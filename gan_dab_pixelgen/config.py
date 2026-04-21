"""
GAN DABPixelGen Training Configuration.

Replaces flow-matching with a conditional GAN:
  Generator:     H&E [3ch] → DAB density [1ch] + IHC RGB [3ch]
  Discriminator: multi-scale PatchGAN (H&E + IHC → real/fake)

Training uses TTUR (Two Time-scale Update Rule):
  D lr = 4e-4, G lr = 1e-4, both Adam(beta1=0, beta2=0.9)
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class GANDABConfig:
    """Full configuration for GAN-based DABPixelGen training."""

    # ── Image ─────────────────────────────────────────────────────────────────
    image_size:         int = 256   # training patch resolution
    source_image_size:  int = 512   # source patch before crop/resize

    # ── Generator (UNet) ──────────────────────────────────────────────────────
    gen_base_channels:          int               = 64
    gen_channel_mult:           Tuple[int, ...]   = (1, 2, 4, 8)
    gen_num_res_blocks:         int               = 2
    gen_attention_resolutions:  Tuple[int, ...]   = (32,)   # spatial res where SelfAttn is added
    gen_dropout:                float             = 0.0
    combination_channels:       int               = 32      # CombinationNet base (~0.66M params)

    # ── Discriminator (Multi-scale PatchGAN) ──────────────────────────────────
    disc_base_channels: int = 64
    disc_n_layers:      int = 4     # 4-layer -> ~70x70 receptive field
    disc_num_scales:    int = 2     # number of PatchGAN scales

    # ── GAN objective ─────────────────────────────────────────────────────────
    gan_mode:        str = "hinge"  # "hinge" | "lsgan"
    d_steps_per_g:   int = 1        # D updates per G update

    # ── Loss weights (generator) ──────────────────────────────────────────────
    adv_weight:        float = 1.0    # adversarial loss
    feat_match_weight: float = 10.0   # feature matching loss
    l1_weight:         float = 100.0  # pixel-level L1 on IHC RGB (pix2pix standard)
    dab_weight:        float = 10.0   # FOD-space DAB expression loss
    lpips_weight:      float = 10.0   # LPIPS perceptual loss on IHC RGB

    # ── R1 gradient penalty on D ──────────────────────────────────────────────
    r1_gamma:    float = 10.0  # R1 penalty weight (StyleGAN2-style)
    r1_interval: int   = 16    # lazy R1: apply every N D-steps, scale by interval

    # ── Optimizers (TTUR) ─────────────────────────────────────────────────────
    g_lr:   float = 1e-4
    d_lr:   float = 4e-4
    beta1:  float = 0.0    # GAN-standard beta1 (no momentum = fast transitions)
    beta2:  float = 0.9
    max_grad_norm: float = 1.0

    # ── EMA on generator ──────────────────────────────────────────────────────
    use_ema:   bool  = True
    ema_decay: float = 0.999

    # ── Training schedule ─────────────────────────────────────────────────────
    batch_size:                  int   = 8
    num_epochs:                  int   = 200
    gradient_accumulation_steps: int   = 1
    warmup_steps:                int   = 200   # linear LR warmup for G only

    # ── Instance noise injected into D inputs ─────────────────────────────────
    # Adds Gaussian noise to D inputs to stabilise early GAN training.
    # Decayed linearly to 0 over `instance_noise_decay_steps`.
    instance_noise_std:          float = 0.05
    instance_noise_decay_steps:  int   = 5000

    # ── DataLoader ────────────────────────────────────────────────────────────
    num_workers:     int  = 8
    prefetch_factor: int  = 2
    preload_to_ram:  bool = False

    # ── Mixed precision ───────────────────────────────────────────────────────
    mixed_precision: str = "bf16"

    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_root: str = "/home/ahmed_ayman/data"
    output_dir:   str = "/home/ahmed_ayman/ayman/outputs/gan_dab_pixelgen"

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_every:       int = 20
    save_every:      int = 5    # save checkpoint every N epochs
    val_every:       int = 1
    num_val_samples: int = 8    # images to include in W&B validation grid
    num_val_batches: int = 30   # max validation batches (0 = full val set)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_project: str = "gan-dab-pixelgen-he-to-ihc"
    wandb_name:    str = "gan-dab-pixelgen"

    # ── Stains ────────────────────────────────────────────────────────────────
    stains: Tuple[str, ...] = ("KI67",)

    # ── Finetune ──────────────────────────────────────────────────────────────
    finetune_checkpoint: Optional[str] = None

    def configure_for_finetune(self, checkpoint_path: str, target_resolution: int = 512):
        """Configure for higher-resolution finetuning from a trained GAN checkpoint."""
        self.finetune_checkpoint = checkpoint_path
        self.image_size = target_resolution
        self.source_image_size = target_resolution
        self.batch_size = 4
        self.g_lr = 5e-5
        self.d_lr = 2e-4
        self.num_epochs = 50
        self.wandb_name = f"gan-dab-pixelgen-finetune-{target_resolution}"
        print(f"Configured for {target_resolution} finetuning from {checkpoint_path}")
        print(f"  G lr={self.g_lr}  D lr={self.d_lr}  Epochs={self.num_epochs}")
