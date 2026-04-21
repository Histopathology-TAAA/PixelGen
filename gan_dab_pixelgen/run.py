"""
GAN DABPixelGen: Main entry point.

Usage (script):
    python -m gan_dab_pixelgen.run
    python -m gan_dab_pixelgen.run finetune path/to/gan_dab_best.pt 512
    python -m gan_dab_pixelgen.run infer     path/to/gan_dab_best.pt he_image.png

Usage (notebook):
    from gan_dab_pixelgen.run import main
    main(batch_size=4, num_epochs=100)
"""
import gc
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from accelerate import Accelerator
from diffusers.optimization import get_constant_schedule_with_warmup

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from gan_dab_pixelgen.config        import GANDABConfig
from gan_dab_pixelgen.generator     import DABUNetGenerator, GeneratorEMA, create_generator
from gan_dab_pixelgen.discriminator import MultiScalePatchGAN
from gan_dab_pixelgen.losses        import GANLoss, GANDABLoss
from gan_dab_pixelgen.train         import train_gan, load_checkpoint
from dab_pixelgen.dataset           import create_dataloaders


# ── Environment (.env) ────────────────────────────────────────────────────────

def _load_env():
    """Load .env files from repo root and gan_dab_pixelgen/ if present."""
    for env_path in [
        os.path.join(PIXELGEN_ROOT, ".env"),
        os.path.join(PIXELGEN_ROOT, "gan_dab_pixelgen", ".env"),
    ]:
        try:
            from dotenv import load_dotenv
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
                print(f"Loaded environment file: {env_path}")
        except ImportError:
            pass


_load_env()


# ── Optimizer factory ─────────────────────────────────────────────────────────

def _make_optimizer(model, lr: float, beta1: float, beta2: float):
    """Adam optimizer with GAN-standard betas (beta1=0 by default)."""
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(
            model.parameters(), lr=lr, betas=(beta1, beta2),
        )
        print(f"Using bitsandbytes 8-bit Adam (lr={lr})")
    except ImportError:
        opt = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        print(f"Using standard Adam (lr={lr})")
    return opt


# ── Config printer ────────────────────────────────────────────────────────────

def _print_config(config: GANDABConfig):
    print(f"\n{'='*65}")
    print(f"GAN DABPixelGen Configuration")
    print(f"{'='*65}")
    print(f"  Image:         {config.image_size}px  (source {config.source_image_size}px)")
    print(f"  Generator:     base={config.gen_base_channels}  mult={config.gen_channel_mult}  "
          f"nrb={config.gen_num_res_blocks}")
    print(f"  Discriminator: scales={config.disc_num_scales}  layers={config.disc_n_layers}  "
          f"ndf={config.disc_base_channels}")
    print(f"  GAN mode:      {config.gan_mode}  (R1 gamma={config.r1_gamma}  "
          f"every {config.r1_interval} steps)")
    print(f"  Loss weights:  adv={config.adv_weight}  FM={config.feat_match_weight}  "
          f"L1={config.l1_weight}  DAB={config.dab_weight}  LPIPS={config.lpips_weight}")
    print(f"  Optimizers:    G lr={config.g_lr}  D lr={config.d_lr}  "
          f"(beta1={config.beta1}, beta2={config.beta2})")
    print(f"  Batch:         {config.batch_size}")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Mixed prec:    {config.mixed_precision}")
    print(f"  EMA:           {config.use_ema}  (decay={config.ema_decay})")
    print(f"  Stains:        {config.stains}")
    print(f"{'='*65}\n")


# ── Main training pipeline ────────────────────────────────────────────────────

def main(
    resume_checkpoint:    Optional[str] = None,
    finetune_checkpoint:  Optional[str] = None,
    finetune_resolution:  int           = 512,
    **config_overrides,
):
    """
    Full GAN DABPixelGen training pipeline.

    Args:
        resume_checkpoint:   Path to a previously saved GAN checkpoint to resume.
        finetune_checkpoint: Path to a trained checkpoint to finetune at higher res.
        finetune_resolution: Target resolution for finetuning (default 512).
        **config_overrides:  Any GANDABConfig field overrides (e.g. batch_size=4).
    """
    # ── Config ────────────────────────────────────────────────────────────────
    config = GANDABConfig()

    if finetune_checkpoint:
        config.configure_for_finetune(finetune_checkpoint, finetune_resolution)

    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            print(f"WARNING: Unknown config key '{k}' ignored.")

    _print_config(config)

    # ── Devices ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Reuse DABPairedDataset from dab_pixelgen — it already returns dab_gt_fod etc.
    train_loader, val_loader = create_dataloaders(config)
    print(f"Dataset: {len(train_loader.dataset)} train  |  {len(val_loader.dataset)} val")

    # ── Models ────────────────────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    generator = create_generator(config, device)

    discriminator = MultiScalePatchGAN(
        input_nc   = 6,   # HE (3ch) + IHC (3ch)
        ndf        = config.disc_base_channels,
        n_layers   = config.disc_n_layers,
        num_scales = config.disc_num_scales,
    ).to(device)

    # ── Losses ────────────────────────────────────────────────────────────────
    gan_loss  = GANLoss(mode=config.gan_mode)

    g_loss_fn = GANDABLoss(
        adv_weight        = config.adv_weight,
        feat_match_weight = config.feat_match_weight,
        l1_weight         = config.l1_weight,
        dab_weight        = config.dab_weight,
        lpips_weight      = config.lpips_weight,
        gan_mode          = config.gan_mode,
        device            = device,
    )

    # ── Optimizers (TTUR) ─────────────────────────────────────────────────────
    g_optimizer = _make_optimizer(generator,     config.g_lr, config.beta1, config.beta2)
    d_optimizer = _make_optimizer(discriminator, config.d_lr, config.beta1, config.beta2)

    # Warmup only on G (D needs to be ready quickly)
    g_lr_scheduler = get_constant_schedule_with_warmup(
        g_optimizer, num_warmup_steps=config.warmup_steps
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_tracker = None
    if config.use_ema:
        ema_tracker = GeneratorEMA(generator, decay=config.ema_decay)

    # ── Accelerator ───────────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision              = config.mixed_precision,
        gradient_accumulation_steps  = 1,   # GAN handles accumulation manually
    )

    (
        generator, discriminator,
        g_optimizer, d_optimizer,
        train_loader, val_loader,
        g_lr_scheduler,
    ) = accelerator.prepare(
        generator, discriminator,
        g_optimizer, d_optimizer,
        train_loader, val_loader,
        g_lr_scheduler,
    )
    print(f"Accelerator: mixed_precision={config.mixed_precision}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    checkpoint_to_load = resume_checkpoint or (
        finetune_checkpoint if finetune_checkpoint else None
    )
    if checkpoint_to_load and Path(checkpoint_to_load).exists():
        start_epoch = load_checkpoint(
            checkpoint_to_load,
            accelerator.unwrap_model(generator),
            accelerator.unwrap_model(discriminator),
            g_optim=g_optimizer,
            d_optim=d_optimizer,
            ema=ema_tracker,
        )
        if finetune_checkpoint:
            # Reset epoch counter when finetuning
            start_epoch = 0
        print(f"Resuming from epoch {start_epoch}")

    # ── Train ─────────────────────────────────────────────────────────────────
    train_gan(
        generator       = generator,
        discriminator   = discriminator,
        train_dataloader= train_loader,
        val_dataloader  = val_loader,
        g_optimizer     = g_optimizer,
        d_optimizer     = d_optimizer,
        g_lr_scheduler  = g_lr_scheduler,
        accelerator     = accelerator,
        config          = config,
        g_loss_fn       = g_loss_fn,
        gan_loss        = gan_loss,
        ema_tracker     = ema_tracker,
        start_epoch     = start_epoch,
    )


# ── Inference CLI entry ───────────────────────────────────────────────────────

def infer(
    checkpoint_path: str,
    he_image_path:   str,
    output_path:     str = "gan_ihc_output.png",
    device:          str = "cuda",
):
    """
    Run inference from CLI.

    Args:
        checkpoint_path: Path to saved GAN checkpoint.
        he_image_path:   Path to H&E image (PNG/TIFF).
        output_path:     Where to save the generated IHC image.
    """
    from PIL import Image
    import torchvision.transforms as T
    from gan_dab_pixelgen.inference import run_inference

    config = GANDABConfig()
    generator = create_generator(config, device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Load EMA weights if available, else regular
    if "ema" in ckpt:
        from gan_dab_pixelgen.generator import GeneratorEMA
        ema = GeneratorEMA.__new__(GeneratorEMA)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow(generator)
        print("Using EMA weights for inference.")
    else:
        generator.load_state_dict(ckpt["generator"])

    generator = generator.to(device).eval()

    # Load and preprocess H&E image
    img = Image.open(he_image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ])
    he_tensor = transform(img).unsqueeze(0)   # [1, 3, H, W] in [0, 1]

    with torch.no_grad():
        results = run_inference(he_tensor, generator, device=device)

    ihc_rgb = results["ihc_rgb"][0]   # [3, H, W]
    out_img  = T.ToPILImage()(ihc_rgb.clamp(0, 1).cpu())
    out_img.save(output_path)
    print(f"Generated IHC saved to: {output_path}")


# ── CLI dispatcher ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAN DABPixelGen")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    train_p = subparsers.add_parser("train", help="Start/resume training")
    train_p.add_argument("--resume",         type=str, default=None,
                         help="Resume from checkpoint path")
    train_p.add_argument("--batch-size",     type=int, default=None)
    train_p.add_argument("--num-epochs",     type=int, default=None)
    train_p.add_argument("--g-lr",           type=float, default=None)
    train_p.add_argument("--d-lr",           type=float, default=None)
    train_p.add_argument("--image-size",     type=int, default=None)
    train_p.add_argument("--output-dir",     type=str, default=None)
    train_p.add_argument("--wandb-name",     type=str, default=None)

    # finetune
    ft_p = subparsers.add_parser("finetune", help="Finetune at higher resolution")
    ft_p.add_argument("checkpoint", type=str, help="Trained checkpoint to finetune from")
    ft_p.add_argument("resolution", type=int, nargs="?", default=512,
                      help="Target resolution (default 512)")

    # infer
    infer_p = subparsers.add_parser("infer", help="Run inference on an H&E image")
    infer_p.add_argument("checkpoint",    type=str, help="Trained checkpoint")
    infer_p.add_argument("he_image",      type=str, help="Path to H&E image")
    infer_p.add_argument("--output",      type=str, default="gan_ihc_output.png")
    infer_p.add_argument("--device",      type=str, default="cuda")

    args = parser.parse_args()

    if args.command in (None, "train"):
        overrides = {}
        if hasattr(args, "batch_size")  and args.batch_size:  overrides["batch_size"]  = args.batch_size
        if hasattr(args, "num_epochs")  and args.num_epochs:  overrides["num_epochs"]  = args.num_epochs
        if hasattr(args, "g_lr")        and args.g_lr:        overrides["g_lr"]        = args.g_lr
        if hasattr(args, "d_lr")        and args.d_lr:        overrides["d_lr"]        = args.d_lr
        if hasattr(args, "image_size")  and args.image_size:  overrides["image_size"]  = args.image_size
        if hasattr(args, "output_dir")  and args.output_dir:  overrides["output_dir"]  = args.output_dir
        if hasattr(args, "wandb_name")  and args.wandb_name:  overrides["wandb_name"]  = args.wandb_name
        main(
            resume_checkpoint=getattr(args, "resume", None),
            **overrides,
        )

    elif args.command == "finetune":
        main(
            finetune_checkpoint = args.checkpoint,
            finetune_resolution = args.resolution,
        )

    elif args.command == "infer":
        infer(
            checkpoint_path = args.checkpoint,
            he_image_path   = args.he_image,
            output_path     = args.output,
            device          = args.device,
        )
    else:
        parser.print_help()
