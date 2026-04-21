"""
DABPixelGen Training Configuration.

Single-backbone model (one JiT_I2I) that predicts 1-channel DAB density
and 2-scalar H-normalization parameters, then recomposes IHC RGB analytically.

Model sizes mirror PixelGen JiT_I2I presets:
  XL: hidden=1152, depth=28, heads=16  (~671M params)  -- matches PixelGen_XL_80ep.ckpt
  S:  hidden=768,  depth=18, heads=12  (~195M params)
  B:  hidden=768,  depth=12, heads=12  (~130M params)  -- fits 24GB GPU
"""
import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional


MODEL_PRESETS = {
    "XL": dict(hidden_size=1152, depth=28, num_heads=16,
               use_bottleneck=False, bottleneck_dim=128,
               attn_drop=0.0, proj_drop=0.1),
    "S":  dict(hidden_size=768,  depth=18, num_heads=12,
               use_bottleneck=True,  bottleneck_dim=128,
               attn_drop=0.0, proj_drop=0.1),
    "B":  dict(hidden_size=768,  depth=12, num_heads=12,
               use_bottleneck=True,  bottleneck_dim=128,
               attn_drop=0.0, proj_drop=0.1),
}


@dataclass
class DABPixelGenConfig:
    """
    Full configuration for DABPixelGen training.

    Key differences from StarDiffPixelGenConfig:
      - in_channels=1 (DAB is 1ch, not 3ch)
      - No dual-path: single noise_path only
      - New loss weights: fm_weight, dab_weight, recomp_weight
      - noise_gate_threshold controls when DAB/recomp losses activate
    """

    # ── Model architecture ────────────────────────────────────────────────────
    image_size:         int   = 256    # training resolution
    source_image_size:  int   = 512    # source patch size (cropped/resized to image_size)
    patch_size:         int   = 16
    in_channels:        int   = 1      # DAB = 1 channel
    cond_channels:      int   = 3      # H&E condition = 3 channels
    mlp_ratio:          float = 4.0

    model_size: str = "XL"
    # Auto-set by set_model_size():
    hidden_size: int   = 1152
    depth:       int   = 28
    num_heads:   int   = 16
    attn_drop:   float = 0.0
    proj_drop:   float = 0.1
    bottleneck_dim: int = 128
    use_bottleneck: bool = False

    # ── Flow Matching ─────────────────────────────────────────────────────────
    num_timesteps:   int   = 50
    he_init_alpha:   float = 0.3    # H&E warm-start coefficient for noise init

    # ── Loss weights ──────────────────────────────────────────────────────────
    fm_weight:           float = 1.0   # flow matching velocity loss
    dab_weight:          float = 1.0   # direct DAB density MSE
    recomp_weight:       float = 1.0   # recomposition MSE (end-to-end)
    lpips_weight:        float = 0.0   # LPIPS on recomposed RGB (0=disabled)
    noise_gate_threshold: float = 0.7  # min t for DAB/recomp/perceptual losses

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:                   int   = 48
    learning_rate:                float = 1e-4
    num_epochs:                   int   = 80
    warmup_steps:                 int   = 500
    gradient_accumulation_steps:  int   = 1
    max_grad_norm:                float = 1.0

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema:          bool  = False
    ema_decay:        float = 0.999
    ema_every_n_steps: int  = 50

    # ── DataLoader ────────────────────────────────────────────────────────────
    num_workers:     int  = 8
    prefetch_factor: int  = 2
    preload_to_ram:  bool = False

    # ── Mixed precision ───────────────────────────────────────────────────────
    mixed_precision: str = "bf16"

    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_root: str = "/home/ahmed_ayman/data"
    output_dir: str = "/home/ahmed_ayman/ayman/outputs/pixelgen_dab"
    pretrained_weight_path: Optional[str] = "./PixelGen_XL_80ep.ckpt"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every:        int = 20
    save_every:       int = 5
    val_every:        int = 1
    num_val_samples:  int = 8    # how many images to visualise in the W&B grid
    num_val_batches:  int = 30   # max batches to run during validation (0 = full val set)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_project: str = "dab-pixelgen-he-to-ihc"
    wandb_name:    str = "dab-pixelgen-XL"

    # ── Stains ────────────────────────────────────────────────────────────────
    stains: Tuple[str, ...] = ("KI67",)

    # ── Finetuning ────────────────────────────────────────────────────────────
    finetune_checkpoint: Optional[str] = None

    def __post_init__(self):
        self.set_model_size(self.model_size)

    def set_model_size(self, size: str):
        if size not in MODEL_PRESETS:
            raise ValueError(f"Unknown model_size '{size}'. Choose: {list(MODEL_PRESETS)}")
        self.model_size = size
        for k, v in MODEL_PRESETS[size].items():
            setattr(self, k, v)

    def update_from_gpu(self, gpu_type, batch_size, grad_accum, precision):
        if batch_size is not None:
            self.batch_size = batch_size
        if grad_accum is not None:
            self.gradient_accumulation_steps = grad_accum
        self.mixed_precision = precision
        self.set_model_size("XL")   # always use XL (matching pretrained weights)
        if gpu_type == "T4":
            self.lpips_weight = 0.0  # disable LPIPS on tiny GPU

    def configure_for_finetune(self, checkpoint_path: str, target_resolution: int = 512):
        """Configure for higher-resolution finetuning from a trained checkpoint."""
        self.finetune_checkpoint = checkpoint_path
        self.image_size = target_resolution
        self.source_image_size = target_resolution
        self.batch_size = 8
        self.gradient_accumulation_steps = 1

        if target_resolution >= 1024:
            self.learning_rate = 1e-5
            self.num_epochs = 4
        else:
            self.learning_rate = 2e-5
            self.num_epochs = 20

        # Enable LPIPS for finetuning
        self.lpips_weight = 0.1
        self.noise_gate_threshold = 0.7

        self.wandb_name = f"dab-pixelgen-{self.model_size}-finetune-{target_resolution}"
        print(f"Configured for {target_resolution} finetuning from {checkpoint_path}")
        print(f"  Batch: {self.batch_size} x {self.gradient_accumulation_steps} | LR: {self.learning_rate}")
