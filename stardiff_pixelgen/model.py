"""
StarDiff Model: Two independent JiT_I2I backbones for noise + restoration paths.

Uses PixelGen's JiT_I2I transformer architecture (Vision Transformer with RoPE,
SwiGLU FFN, AdaLN modulation). Each backbone takes 6-channel input [noisy_IHC, H&E].

- Noise path:       predicts ε for stochastic denoising
- Restoration path: predicts I_res = I_ihc - I_he for structure preservation

Architecture matches PixelGen configs_i2i/mist_ki67_pre_fmonly.yaml when model_size="XL".
"""
import sys
import os
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Tuple

# Add parent PixelGen dir to path so we can import src.models
PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from src.models.transformer.JiT_I2I import JiT_I2I


class GradCheckpointJiTBlock(nn.Module):
    """
    Wrapper around JiTBlock that applies gradient checkpointing.
    Handles the @torch.compile decorator on JiTBlock.forward by
    accessing the original uncompiled forward method.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        # Get the original uncompiled forward function.
        # If @torch.compile was applied, the original is stored in _orig_mod
        # or we can use __wrapped__. As fallback, use the compiled version.
        if hasattr(block, '_orig_mod'):
            # torch.compile wraps the whole module
            self._orig_forward = block._orig_mod.forward
        else:
            # Get the underlying function from the class to bypass compile
            self._orig_forward = block.__class__.forward.__wrapped__ if hasattr(
                block.__class__.forward, '__wrapped__') else block.__class__.forward

    def forward(self, x, c, feat_rope=None):
        return torch.utils.checkpoint.checkpoint(
            self._orig_forward,
            self.block, x, c, feat_rope,
            use_reentrant=False,
        )


class StarDiffPixelGenModel(nn.Module):
    """
    StarDiff with two independent JiT_I2I backbones:
      - noise_path:       predicts ε (Gaussian noise)
      - restoration_path: predicts I_res = I_ihc - I_he

    H&E condition is concatenated channel-wise with noisy IHC.
    Each path: forward([noisy_ihc(3ch), he(3ch)]=6ch, timestep) → output(3ch)

    Paper ref: arXiv:2508.02528 Section 3.2
    "we train two networks in parallel:
     - noise prediction network ε_θ(x_t, t, I_he)
     - restoration prediction network r_θ(x_t, t, I_he)"
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        cond_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        bottleneck_dim: int = 128,
        use_bottleneck: bool = True,
        weight_path: str = None,
        load_ema: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        common_kwargs = dict(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            cond_channels=cond_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            bottleneck_dim=bottleneck_dim,
            use_bottleneck=use_bottleneck,
            use_compile=False,
        )

        # Noise prediction path (ε_θ)
        self.noise_path = JiT_I2I(
            weight_path=weight_path, load_ema=load_ema, **common_kwargs,
        )

        # Restoration prediction path (r_θ)
        self.restoration_path = JiT_I2I(
            weight_path=weight_path, load_ema=load_ema, **common_kwargs,
        )

        # Apply gradient checkpointing by wrapping each block
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        noise_params = sum(p.numel() for p in self.noise_path.parameters())
        rest_params = sum(p.numel() for p in self.restoration_path.parameters())
        total = noise_params + rest_params
        print(f"✓ StarDiff-PixelGen: 2 × JiT_I2I backbones")
        print(f"  Noise path:       {noise_params / 1e6:.1f}M params")
        print(f"  Restoration path: {rest_params / 1e6:.1f}M params")
        print(f"  Total:            {total / 1e6:.1f}M params")
        print(f"  Hidden: {hidden_size}, Depth: {depth}, Heads: {num_heads}")
        print(f"  Bottleneck: {use_bottleneck} (dim={bottleneck_dim})")
        print(f"  Gradient checkpointing: {gradient_checkpointing}")
        if weight_path:
            print(f"  Pretrained from: {weight_path}")

    def _enable_gradient_checkpointing(self):
        """Wrap transformer blocks with gradient checkpointing."""
        for path_name in ("noise_path", "restoration_path"):
            path = getattr(self, path_name)
            new_blocks = nn.ModuleList()
            for block in path.blocks:
                new_blocks.append(GradCheckpointJiTBlock(block))
            path.blocks = new_blocks
        print("✓ Gradient checkpointing enabled for both paths")

    def rescale_resolution(self, new_input_size):
        """
        Rescale both paths to a new input resolution.
        Interpolates positional embeddings (bicubic) and rebuilds RoPE.

        Call AFTER loading checkpoint weights and BEFORE training.

        Args:
            new_input_size: Target resolution (e.g. 512 for 512x512)
        """
        old_size = self.noise_path.input_size
        old_tokens = (old_size // self.noise_path.patch_size) ** 2
        new_tokens = (new_input_size // self.noise_path.patch_size) ** 2

        print(f"🔄 Rescaling StarDiff: {old_size}x{old_size} → {new_input_size}x{new_input_size}")
        print(f"   Tokens per path: {old_tokens} → {new_tokens}")

        self.noise_path.rescale_resolution(new_input_size)
        self.restoration_path.rescale_resolution(new_input_size)

        print(f"   ✓ pos_embed interpolated (bicubic)")
        print(f"   ✓ RoPE rebuilt with resolution scaling")

    def forward(
        self,
        x: torch.Tensor,          # noisy IHC [B, 3, H, W]
        t: torch.Tensor,          # timestep [B]
        condition: torch.Tensor,   # H&E condition [B, 3, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (noise_pred, restoration_pred), each [B, 3, H, W].

        Each path independently:
        1. Concatenates [x, condition] → 6ch
        2. Embeds patches + timestep
        3. Runs through transformer blocks
        4. Unpatchifies to image
        """
        noise_pred = self.noise_path(x, t, condition)
        restoration_pred = self.restoration_path(x, t, condition)
        return noise_pred, restoration_pred


# ── EMA (Exponential Moving Average) ──

class SimpleEMA:
    """
    Simple EMA tracker matching PixelGen's src.callbacks.simple_ema.SimpleEMA.
    Tracks a running average of model weights for more stable inference.

    From YAML: decay=0.999, every_n_steps=1
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, every_n_steps: int = 1):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.step_counter = 0
        # Create shadow copy of model parameters ON CPU to save VRAM
        self.shadow = {name: param.clone().detach().cpu()
                       for name, param in model.named_parameters() if param.requires_grad}
        print(f"✓ EMA tracker (Offloaded to CPU): decay={decay}, every_n_steps={every_n_steps}, "
              f"tracking {len(self.shadow)} params (~{len(self.shadow)*4/(1024**3):.1f}GB RAM overhead)")

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights. Call after each optimizer step."""
        self.step_counter += 1
        if self.step_counter % self.every_n_steps != 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Move GPU param to CPU for math, then save back to CPU shadow
                self.shadow[name].mul_(self.decay).add_(param.data.cpu(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation/sampling)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone().cpu()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "step_counter": self.step_counter}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.step_counter = state_dict["step_counter"]


def create_stardiff_model(config, device="cuda"):
    """Create StarDiff model from config and move to device."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    model = StarDiffPixelGenModel(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        cond_channels=config.cond_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        attn_drop=config.attn_drop,
        proj_drop=config.proj_drop,
        bottleneck_dim=config.bottleneck_dim,
        use_bottleneck=config.use_bottleneck,
        weight_path=config.pretrained_weight_path,
        load_ema=True,
        gradient_checkpointing=True,
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model on {device}: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    print(f"  Architecture: JiT_I2I_{config.model_size}")
    return model


def create_stardiff_model_for_finetune(config, checkpoint_path, new_resolution=512, device="cuda"):
    """
    Create StarDiff model for resolution finetuning.

    Steps:
      1. Create model at ORIGINAL checkpoint resolution (e.g. 256)
      2. Load StarDiff checkpoint weights
      3. Rescale pos_embed (bicubic) + RoPE to new_resolution
      4. Move to device

    This preserves all learned features from the 256 training while
    adapting positional encodings for the higher resolution.

    Args:
        config: StarDiffPixelGenConfig (model arch fields used, image_size ignored)
        checkpoint_path: Path to trained StarDiff .pt checkpoint
        new_resolution: Target resolution (e.g. 512)
        device: Target device

    Returns:
        Model ready for finetuning at new_resolution
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load checkpoint to determine original resolution
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    ckpt_config = ckpt.get("config", {})
    original_resolution = int(ckpt_config.get("image_size", "256"))
    print(f"   Original resolution: {original_resolution}x{original_resolution}")

    # Create model at ORIGINAL resolution to match checkpoint weight shapes
    model = StarDiffPixelGenModel(
        input_size=original_resolution,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        cond_channels=config.cond_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        attn_drop=config.attn_drop,
        proj_drop=config.proj_drop,
        bottleneck_dim=config.bottleneck_dim,
        use_bottleneck=config.use_bottleneck,
        weight_path=None,   # Don't load PixelGen pretrained — we load StarDiff ckpt
        load_ema=False,
        gradient_checkpointing=True,
    )

    # Load StarDiff checkpoint
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    n_tensors = sum(1 for _ in ckpt["model_state_dict"])
    print(f"   ✓ Loaded {n_tensors} weight tensors")

    # Rescale to new resolution (interpolate pos_embed + rebuild RoPE)
    if new_resolution != original_resolution:
        model.rescale_resolution(new_resolution)

    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Finetune model on {device}: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    print(f"  Resolution: {new_resolution}x{new_resolution}")
    return model
