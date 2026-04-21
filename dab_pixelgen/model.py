"""
DABPixelGen Model: Single JiT_I2I backbone for DAB density prediction.

Architecture:
  - Input: 4ch [noisy_DAB (1ch), H&E condition (3ch)]
  - Output: 1ch DAB density + per-image H-normalization parameters (a, b)
  - Backbone: PixelGen JiT_I2I (pretrained XL/S/B)

The H-norm head attaches to the final hidden state (global average pooled)
and predicts (a_raw, b_raw) scalars per image.  These constrain to:
  a = sigmoid(a_raw) * 2   in (0, 2)
  b = tanh(b_raw) * 0.5    in (-0.5, 0.5)
so H_IHC = a * H_HE + b (clamped >= 0).
"""
import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Tuple, Optional

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from src.models.transformer.JiT_I2I import JiT_I2I, VisionRotaryEmbeddingFast


class GradCheckpointJiTBlock(nn.Module):
    """Gradient checkpointing wrapper for JiTBlock (handles @torch.compile)."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        if hasattr(block, "_orig_mod"):
            self._orig_forward = block._orig_mod.forward
        else:
            self._orig_forward = (
                block.__class__.forward.__wrapped__
                if hasattr(block.__class__.forward, "__wrapped__")
                else block.__class__.forward
            )

    def forward(self, x, c, feat_rope=None):
        return torch.utils.checkpoint.checkpoint(
            self._orig_forward,
            self.block, x, c, feat_rope,
            use_reentrant=False,
        )


class CombinationNet(nn.Module):
    """
    Lightweight 2-level encoder-decoder that maps (DAB prediction + H&E) → IHC RGB.

    Replaces analytical Beer-Lambert recomposition with a learned spatial mapping.
    This eliminates the dependency on calibrated H-normalization parameters in early
    training and provides direct gradient flow from the IHC loss to spatial features.

    Architecture: ~1.2M params.
    Input:  [B, 4, H, W]  = DAB_pred (1ch, raw density) + HE_condition (3ch, in [-1, 1])
    Output: [B, 3, H, W]  IHC RGB in [-1, 1]

    The output layer is zero-initialized so the network starts as a null correction
    (outputs near 0 = gray) and learns from the IHC reconstruction loss.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        C = base_channels

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, 3, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.SiLU(),
            nn.Conv2d(C * 2, C * 2, 3, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, C * 4),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 3, padding=1),
            nn.GroupNorm(8, C * 4),
            nn.SiLU(),
            nn.Conv2d(C * 4, C * 4, 3, padding=1),
            nn.GroupNorm(8, C * 4),
            nn.SiLU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C * 4 + C * 2, C * 2, 3, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.SiLU(),
            nn.Conv2d(C * 2, C * 2, 3, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.SiLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C * 2 + C, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(C, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, dab_pred: torch.Tensor, he_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dab_pred: [B, 1, H, W]  predicted DAB density (raw OD scale)
            he_cond:  [B, 3, H, W]  H&E condition in [-1, 1]
        Returns:
            [B, 3, H, W] IHC RGB in [-1, 1]
        """
        x  = torch.cat([dab_pred, he_cond], dim=1)  # [B, 4, H, W]
        s1 = self.enc1(x)                            # [B, C, H, W]
        x  = self.down1(s1)                          # [B, 2C, H/2, W/2]
        s2 = self.enc2(x)                            # [B, 2C, H/2, W/2]
        x  = self.down2(s2)                          # [B, 4C, H/4, W/4]
        x  = self.bottleneck(x)                      # [B, 4C, H/4, W/4]
        x  = self.up1(x)
        x  = self.dec1(torch.cat([x, s2], dim=1))   # [B, 2C, H/2, W/2]
        x  = self.up2(x)
        x  = self.dec2(torch.cat([x, s1], dim=1))   # [B, C, H, W]
        return torch.tanh(self.out(x))               # [B, 3, H, W] in [-1, 1]


class HNormHead(nn.Module):
    """
    Per-image H-channel normalization parameter predictor.

    Takes the global-average-pooled final hidden state [B, hidden_size]
    and predicts (a_raw, b_raw) per image.

    Constrained at use-time (not here):
        a = sigmoid(a_raw) * 2   in (0, 2)
        b = tanh(b_raw) * 0.5    in (-0.5, 0.5)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        # Zero-init so the head starts predicting (a_raw=0, b_raw=0)
        # which maps to a=0.5*2=1.0 and b=0 — identity-ish
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: [B, N, hidden_size]  transformer token sequence
        Returns:
            [B, 2]  (a_raw, b_raw)
        """
        pooled = x_seq.mean(dim=1)  # [B, hidden_size]
        return self.net(pooled)     # [B, 2]


class DABPixelGenModel(nn.Module):
    """
    Single PixelGen JiT_I2I backbone for DAB density prediction.

    forward(x_noisy_dab, t, he_condition) -> (dab_pred, h_norm_params)

      x_noisy_dab:  [B, 1, H, W]   noisy DAB density at timestep t
      t:            [B]             flow matching timestep in [0, 1]
      he_condition: [B, 3, H, W]   H&E image (condition, not noised)
      ──────────────────────────────────────────────────────────────
      dab_pred:     [B, 1, H, W]   predicted clean DAB density
      h_norm_params:[B, 2]         (a_raw, b_raw) for H normalization
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        bottleneck_dim: int = 128,
        use_bottleneck: bool = False,
        weight_path: Optional[str] = None,
        load_ema: bool = True,
        gradient_checkpointing: bool = True,
        use_combination_net: bool = True,
        combination_channels: int = 32,
    ):
        super().__init__()

        # DAB = 1 channel; H&E condition = 3 channels -> 4ch patch embed input
        self.backbone = JiT_I2I(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=1,          # noisy DAB
            cond_channels=3,        # H&E condition
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            bottleneck_dim=bottleneck_dim,
            use_bottleneck=use_bottleneck,
            use_compile=False,
            weight_path=None,       # we load below with custom logic
            load_ema=False,
        )

        # Load pretrained PixelGen weights with custom channel expansion
        if weight_path is not None:
            self._load_pretrained(weight_path, load_ema)

        # H-normalization auxiliary head (reads final token sequence)
        self.h_norm_head = HNormHead(hidden_size)
        self._hidden_size = hidden_size

        # Learned combination head: (DAB, H&E) -> IHC RGB
        if use_combination_net:
            self.combination_net = CombinationNet(base_channels=combination_channels)
        else:
            self.combination_net = None

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        combo_params = sum(p.numel() for p in self.combination_net.parameters()) if self.combination_net else 0
        print(f"DABPixelGen: 1 x JiT_I2I backbone (1ch DAB + H-norm head + CombinationNet)")
        print(f"  Hidden: {hidden_size}, Depth: {depth}, Heads: {num_heads}")
        print(f"  CombinationNet: {'enabled' if use_combination_net else 'disabled'} ({combo_params/1e6:.2f}M params)")
        print(f"  Total params: {total / 1e6:.1f}M  Trainable: {trainable / 1e6:.1f}M")
        if weight_path:
            print(f"  Pretrained from: {weight_path}")

    def _load_pretrained(self, weight_path: str, load_ema: bool = True):
        """
        Load PixelGen pretrained weights into the backbone.

        The pretrained model has 3ch noisy input; ours has 1ch noisy + 3ch cond = 4ch.
        Strategy: copy pretrained 3ch weights into first 3 channels of the 4ch proj1.
        The 4th channel (condition) stays zero-initialized.
        """
        import logging
        logger = logging.getLogger(__name__)

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        prefix = "ema_denoiser." if load_ema else "denoiser."
        pretrained = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not pretrained:
            logger.warning(f"No weights with prefix '{prefix}'. Trying bare keys.")
            pretrained = state_dict

        skip_prefixes = ("y_embedder.", "in_context_posemb", "feat_rope_incontext.")
        my_state = self.backbone.state_dict()
        loaded = skipped = 0

        for key, param in pretrained.items():
            if any(key.startswith(sp) for sp in skip_prefixes):
                skipped += 1
                continue

            if key == "x_embedder.proj1.weight" and key in my_state:
                my_shape = my_state[key].shape   # (embed_dim, 4, P, P)
                pt_shape = param.shape            # (embed_dim, 3, P, P)
                if my_shape[1] != pt_shape[1]:
                    logger.info(f"  Channel expand x_embedder.proj1.weight: {pt_shape} -> {my_shape}")
                    new_w = torch.zeros_like(my_state[key])
                    # Copy only first 3 channels (noisy target side) -- skip last ch (condition)
                    # Since JiT_I2I concatenates [noisy, cond], our 4ch is [noisy_dab(1), he(3)].
                    # The pretrained 3ch covers noisy_3ch; we adapt: place pretrained avg into ch0,
                    # then zero-init the condition ch 1..3.
                    pretrained_mean = param.mean(dim=1, keepdim=True)  # (embed_dim, 1, P, P)
                    new_w[:, 0:1, :, :] = pretrained_mean
                    # cond channels (1, 2, 3) stay zero
                    my_state[key].copy_(new_w)
                    loaded += 1
                    continue

            if key in my_state and my_state[key].shape == param.shape:
                my_state[key].copy_(param)
                loaded += 1
            else:
                skipped += 1

        self.backbone.load_state_dict(my_state, strict=False)
        print(f"  Pretrained load: {loaded} loaded, {skipped} skipped")

    def _enable_gradient_checkpointing(self):
        new_blocks = nn.ModuleList()
        for block in self.backbone.blocks:
            new_blocks.append(GradCheckpointJiTBlock(block))
        self.backbone.blocks = new_blocks

    def rescale_resolution(self, new_input_size: int):
        """
        Rescale backbone pos_embed (bicubic) + rebuild RoPE for new_input_size.
        Call AFTER loading checkpoint, BEFORE training at the higher resolution.
        """
        import math
        backbone = self.backbone
        old_size   = backbone.input_size
        patch_size = backbone.patch_size
        old_tokens = (old_size // patch_size) ** 2
        new_tokens = (new_input_size // patch_size) ** 2

        print(f"Rescaling DABPixelGen: {old_size} -> {new_input_size} "
              f"({old_tokens} -> {new_tokens} tokens)")

        # Bicubic interpolation of pos_embed [1, N_old, D] -> [1, N_new, D]
        D = backbone.pos_embed.shape[-1]
        old_hw = int(math.sqrt(old_tokens))
        new_hw = int(math.sqrt(new_tokens))
        pe = backbone.pos_embed.data                    # [1, N_old, D]
        pe = pe.reshape(1, old_hw, old_hw, D).permute(0, 3, 1, 2)  # [1, D, H, W]
        pe = torch.nn.functional.interpolate(
            pe, size=(new_hw, new_hw), mode="bicubic", align_corners=False
        )
        pe = pe.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, D)
        backbone.pos_embed = torch.nn.Parameter(pe, requires_grad=False)

        # Rebuild RoPE for new resolution
        half_head_dim = backbone.hidden_size // backbone.num_heads // 2
        backbone.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=new_hw,
            num_cls_token=0,
        )

        backbone.input_size = new_input_size
        print(f"  pos_embed rescaled and RoPE rebuilt for {new_hw}x{new_hw} grid")

    def forward(
        self,
        x_noisy_dab:  torch.Tensor,   # [B, 1, H, W]
        t:             torch.Tensor,   # [B]
        he_condition:  torch.Tensor,   # [B, 3, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            dab_pred:     [B, 1, H, W]            predicted clean DAB density
            h_norm_params:[B, 2]                   (a_raw, b_raw) for analytical fallback
            ihc_pred:     [B, 3, H, W] or None     IHC RGB in [-1, 1] from CombinationNet
        """
        dab_pred, _, last_seq = self.backbone(
            x_noisy_dab, t, he_condition,
            return_layer=0,
            return_last=True,
        )
        h_norm_params = self.h_norm_head(last_seq)  # [B, 2]

        if self.combination_net is not None:
            ihc_pred = self.combination_net(dab_pred, he_condition)  # [B, 3, H, W]
        else:
            ihc_pred = None

        return dab_pred, h_norm_params, ihc_pred


# ── EMA ──────────────────────────────────────────────────────────────────────

class SimpleEMA:
    """CPU-offloaded EMA tracker (mirrors stardiff_pixelgen.model.SimpleEMA)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, every_n_steps: int = 1):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.step_counter = 0
        self.shadow = {
            name: param.clone().detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        print(f"EMA: decay={decay}, tracking {len(self.shadow)} params")

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step_counter += 1
        if self.step_counter % self.every_n_steps != 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data.cpu(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone().cpu()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "step_counter": self.step_counter}

    def load_state_dict(self, sd):
        self.shadow = sd["shadow"]
        self.step_counter = sd["step_counter"]


# ── Factory helpers ───────────────────────────────────────────────────────────

def create_dab_model(config, device: str = "cuda") -> DABPixelGenModel:
    """Create DABPixelGenModel from a DABPixelGenConfig and move to device."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    model = DABPixelGenModel(
        input_size=config.image_size,
        patch_size=config.patch_size,
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
        use_combination_net=getattr(config, "use_combination_net", True),
        combination_channels=getattr(config, "combination_channels", 32),
    ).to(device)
    return model


def create_dab_model_for_finetune(
    config,
    checkpoint_path: str,
    new_resolution: int = 512,
    device: str = "cuda",
) -> DABPixelGenModel:
    """
    Load a trained DABPixelGen checkpoint and rescale pos_embed + RoPE
    to new_resolution for higher-resolution finetuning.
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {})
    orig_res = int(ckpt_config.get("image_size", 256))
    print(f"Finetuning: checkpoint resolution {orig_res} -> {new_resolution}")

    model = DABPixelGenModel(
        input_size=orig_res,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        attn_drop=config.attn_drop,
        proj_drop=config.proj_drop,
        bottleneck_dim=config.bottleneck_dim,
        use_bottleneck=config.use_bottleneck,
        weight_path=None,
        load_ema=False,
        gradient_checkpointing=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  Loaded {sum(1 for _ in ckpt['model_state_dict'])} tensors from checkpoint")

    if new_resolution != orig_res:
        model.rescale_resolution(new_resolution)

    return model.to(device)
