"""
StarDiff + PixelGen Configuration.
Auto-detects GPU and sets optimal batch size / precision.
Matches PixelGen YAML config: configs_i2i/mist_ki67_pre_fmonly.yaml
"""
import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional


def detect_gpu():
    """Auto-detect GPU type and return optimal settings."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_support = torch.cuda.is_bf16_supported()

    if "H100" in gpu_name:
        gpu_type, batch_size, grad_accum = "H100", 8, 1
    elif "A100" in gpu_name:
        if gpu_memory > 45:
            gpu_type, batch_size, grad_accum = "A100-80GB", 8, 1
        else:
            gpu_type, batch_size, grad_accum = "A100-40GB", 8, 1
    elif "4090" in gpu_name or "A10" in gpu_name:
        gpu_type, batch_size, grad_accum = "RTX4090", 8, 8
    elif "T4" in gpu_name:
        gpu_type, batch_size, grad_accum = "T4", 8, 16
    else:
        gpu_type, batch_size, grad_accum = "Other", 8, 16

    precision = "bf16" if bf16_support else "fp16"

    print("=" * 60)
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"Type: {gpu_type} | BF16: {bf16_support}")
    print(f"Batch: {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    print(f"Precision: {precision}")
    print("=" * 60)

    return gpu_type, batch_size, grad_accum, precision


# ── Model size presets (from PixelGen JiT_I2I.py) ──

MODEL_PRESETS = {
    # XL: matches YAML config exactly (hidden=1152, depth=28, heads=16)
    # ~671M params per path → too large for dual on 24GB
    "XL": dict(hidden_size=1152, depth=28, num_heads=16, use_bottleneck=False,
               bottleneck_dim=128, attn_drop=0.0, proj_drop=0.1),
    # S: ~195M params per path (~390M total dual)
    "S":  dict(hidden_size=768, depth=18, num_heads=12, use_bottleneck=True,
               bottleneck_dim=128, attn_drop=0.0, proj_drop=0.1),
    # B: ~130M params per path (~260M total dual) — good for 24GB GPU
    "B":  dict(hidden_size=768, depth=12, num_heads=12, use_bottleneck=True,
               bottleneck_dim=128, attn_drop=0.0, proj_drop=0.1),
}


@dataclass
class StarDiffPixelGenConfig:
    """
    StarDiff + PixelGen Training Configuration.
    Two independent JiT_I2I backbones (noise + restoration paths).

    Configure model_size to choose architecture:
      - "XL": 671M per path (matches pretrained PixelGen_XL_80ep.ckpt)
              Only use for single-path or 80GB+ GPU
      - "S":  195M per path (~390M dual, fits 40GB)
      - "B":  130M per path (~260M dual, fits 24GB RTX 4090)
    """

    # ── Model Architecture ──
    image_size: int = 256           # from YAML: input_size: 256
    source_image_size: int = 512    # crop_size: 512 in YAML
    patch_size: int = 16            # from YAML: patch_size: 16
    in_channels: int = 3            # from YAML: in_channels: 3
    cond_channels: int = 3          # from YAML: cond_channels: 3
    mlp_ratio: float = 4.0          # from YAML: mlp_ratio: 4.0

    # Model size preset: "XL", "S", or "B"
    # XL matches the pretrained PixelGen_XL_80ep.ckpt architecture
    model_size: str = "XL"

    # These are auto-set by set_model_size() from MODEL_PRESETS
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    attn_drop: float = 0.0
    proj_drop: float = 0.1
    bottleneck_dim: int = 128
    use_bottleneck: bool = False

    # ── Flow Matching (Rectified Flow) ──
    num_timesteps: int = 50       # Flow matching needs ~50 steps (DDPM needed 1000)
    restoration_weight: float = 0.5   # λ for restoration schedule β̄_t

    # ── PixelGen Perceptual Losses ──
    # The YAML "FMonly" config sets these to 0.0 (pure flow matching)
    # For StarDiff, we add perceptual supervision on x̂₀ reconstruction:
    lpips_weight: float = 0 #0.1
    dino_weight: float = 0 #0.01
    noise_gate_threshold: float = 0.3   # from YAML: percept_t_threshold: 0.3
    use_dino: bool = False #True

    # ── Training (from YAML) ──
    batch_size: int = 8              
    learning_rate: float = 1e-4         # from YAML: lr: 0.0001
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0            # from YAML: gradient_clip_val: 1.0

    # ── Loss weights ──
    noise_loss_weight: float = 1.0
    restoration_loss_weight: float = 1.0

    # ── EMA (from YAML: ema_tracker) ──
    use_ema: bool = False                 # Disabled per user request
    ema_decay: float = 0.999              # from YAML: decay: 0.999
    ema_every_n_steps: int = 50           # was 1; increased to 50 to avoid PCIe bottleneck

    # ── DataLoader (from YAML) ──
    num_workers: int = 4                  # Reduced to fix Pin Memory crash
    prefetch_factor: int = 2              # Reduced to fix Pin Memory crash
    preload_to_ram: bool = False          # set to True if you have 32GB+ RAM to eliminate all disk IO bottlenecks

    # ── Mixed Precision ──
    mixed_precision: str = "bf16"         # from YAML: precision: bf16-mixed

    # ── Paths ──
    dataset_root: str = "/home/histo/histo/data"
    output_dir: str = "/home/histo/histo/data"
    # Pretrained PixelGen checkpoint for weight initialization
    # Must match model_size architecture (XL weights only load into XL models)
    pretrained_weight_path: Optional[str] = "./PixelGen_XL_80ep.ckpt"

    # ── Logging (from YAML) ──
    log_every: int = 20                   # from YAML: log_every_n_steps: 20
    save_every: int = 5
    val_every: int = 1
    num_val_samples: int = 8              # from YAML: num_vis_samples: 8

    # ── W&B (from YAML) ──
    wandb_project: str = "star-diff-he-to-ihc"
    wandb_name: str = "stardiff-pixelgen-XL"

    # ── Stains ──
    stains: Tuple[str, ...] = ("KI67",)

    def __post_init__(self):
        """Apply model size preset after initialization."""
        self.set_model_size(self.model_size)

    def set_model_size(self, size: str):
        """Set architecture from preset. Call before creating model."""
        if size not in MODEL_PRESETS:
            raise ValueError(f"Unknown model_size '{size}'. Choose from: {list(MODEL_PRESETS.keys())}")
        self.model_size = size
        for k, v in MODEL_PRESETS[size].items():
            setattr(self, k, v)

    def update_from_gpu(self, gpu_type, batch_size, grad_accum, precision):
        """Update config based on detected GPU. Forcing XL architecture per user request."""
        # Use provided batch_size and grad_accum unless they are None
        if batch_size is not None:
            self.batch_size = batch_size
        if grad_accum is not None:
            self.gradient_accumulation_steps = grad_accum
        
        self.mixed_precision = precision
        # Force XL model size, ignoring GPU memory limits
        self.set_model_size("XL")
        if gpu_type == "T4":
            self.use_dino = False  # Still disable DINOv2 on very low VRAM to try and survive
