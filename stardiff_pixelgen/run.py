"""
StarDiff + PixelGen: Main entry point.
Can be run as a script or imported from a notebook.

Usage:
    python -m stardiff_pixelgen.run

    # Or from a notebook:
    from stardiff_pixelgen.run import main
    main()
"""
import gc
import os
import sys

import torch
from pathlib import Path
from accelerate import Accelerator
from torch.optim import AdamW
from diffusers.optimization import get_constant_schedule_with_warmup

# Ensure PixelGen src is importable
PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from stardiff_pixelgen.config import StarDiffPixelGenConfig, detect_gpu
from stardiff_pixelgen.dataset import download_dataset, create_dataloaders
from stardiff_pixelgen.model import create_stardiff_model, SimpleEMA

from stardiff_pixelgen.losses import PixelGenPerceptualLoss
from stardiff_pixelgen.train import train_stardiff, resume_from_wandb


def main(
    # Override any config fields here
    pretrained_weight_path: str = None,
    resume_wandb: bool = False,
    resume_run_path: str = None,
    resume_file_name: str = "stardiff_best.pt",
    **config_overrides,
):
    """
    Full StarDiff + PixelGen training pipeline.
    """
    from stardiff_pixelgen.stardiff_scheduler import StarDiffScheduler
    # ── 1. GPU Detection ──
    gpu_type, batch_size, grad_accum, precision = detect_gpu()

    # ── 2. Configuration ──
    config = StarDiffPixelGenConfig()
    config.update_from_gpu(gpu_type, batch_size, grad_accum, precision)
    if pretrained_weight_path:
        config.pretrained_weight_path = pretrained_weight_path
    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # Auto-download XL weights if specified and not found locally
    if config.pretrained_weight_path and "PixelGen_XL" in config.pretrained_weight_path:
        weight_path = os.path.abspath(config.pretrained_weight_path)
        if not os.path.exists(weight_path):
            file_name = os.path.basename(weight_path)
            # Default to zehongma/PixelGen repository on Hugging Face
            url = f"https://huggingface.co/zehongma/PixelGen/resolve/main/{file_name}"
            print(f"\n📥 Pretrained weights not found locally: {weight_path}")
            print(f"   Downloading from Hugging Face: {url}")
            os.system(f"wget -q {url} -O {weight_path}")
            if os.path.exists(weight_path):
                print(f"   ✓ Downloaded successfully.")
            else:
                print(f"   ⚠️ Failed to download weights. Please check connection.")
        else:
            print(f"✓ Found pretrained weights: {weight_path}")

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"StarDiff + PixelGen Config ({gpu_type})")
    print(f"{'='*60}")
    print(f"  Image: {config.image_size}×{config.image_size} from {config.source_image_size}")
    print(f"  Model: JiT_I2I_{config.model_size} (hidden={config.hidden_size}, depth={config.depth})")
    print(f"  Batch: {config.batch_size} × {config.gradient_accumulation_steps} = {effective_batch}")
    print(f"  Precision: {config.mixed_precision}")
    print(f"  LPIPS: {config.lpips_weight} | DINO: {'ON' if config.use_dino else 'OFF'} ({config.dino_weight})")
    print(f"  EMA: {'ON' if config.use_ema else 'OFF'} (decay={config.ema_decay})")
    print(f"  Restoration λ: {config.restoration_weight}")
    print(f"{'='*60}\n")

    # ── 3. Dataset ──
    download_dataset(config.dataset_root, config.stains)
    train_loader, val_loader = create_dataloaders(config)

    # ── 4. Model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_stardiff_model(config, device)

    # ── 5. Scheduler ──
    scheduler = StarDiffScheduler(
        num_timesteps=config.num_timesteps,
        restoration_weight=config.restoration_weight,
    )
    print(f"✓ StarDiff Scheduler: T={config.num_timesteps}, λ={config.restoration_weight}")

    # ── 6. Perceptual Loss ──
    perceptual_loss_fn = PixelGenPerceptualLoss(
        use_dino=config.use_dino,
        lpips_weight=config.lpips_weight,
        dino_weight=config.dino_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        device=device,
    )

    # ── 7. Optimizer + LR Scheduler (from YAML: AdamW, lr=1e-4) ──
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        print("✓ Using bitsandbytes 8-bit optimizer")
    except ImportError:
        print("⚠️ 'bitsandbytes' not found. Falling back to standard AdamW.")
        print("   -> Tip: run 'pip install bitsandbytes' to save massive VRAM.")
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)

    # ── 8. EMA (from YAML: decay=0.999, every_n_steps=1) ──
    ema_tracker = None
    if config.use_ema:
        ema_tracker = SimpleEMA(model, decay=config.ema_decay, every_n_steps=config.ema_every_n_steps)

    # ── 9. Accelerator ──
    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    print(f"✓ Accelerator: {config.mixed_precision}, grad accum: {config.gradient_accumulation_steps}")

    # ── 10. Optional Resume ──
    start_epoch = 0
    if resume_wandb and resume_run_path:
        start_epoch = resume_from_wandb(
            resume_run_path, resume_file_name,
            model, optimizer, lr_scheduler, accelerator,
            ema_tracker=ema_tracker,
        )
    else:
        print(f"Starting fresh training from epoch 1")

    # ── 11. Train! ──
    print(f"\n{'='*60}")
    print(f"Starting StarDiff + PixelGen Training: H&E → IHC")
    print(f"{'='*60}\n")

    trained_model = train_stardiff(
        model=model,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        config=config,
        perceptual_loss_fn=perceptual_loss_fn,
        ema_tracker=ema_tracker,
        start_epoch=start_epoch,
    )

    print(f"\n{'='*60}")
    print("✓ StarDiff + PixelGen Training Complete!")
    print(f"{'='*60}")

    # ── 12. Save final model ──
    final_ckpt = {
        "model_type": "stardiff_pixelgen",
        "model_state_dict": accelerator.unwrap_model(trained_model).state_dict(),
        "config": {k: str(v) for k, v in vars(config).items()},
        "scheduler_config": {
            "num_timesteps": scheduler.num_timesteps,
            "restoration_weight": scheduler.restoration_weight,
        },
    }
    if ema_tracker is not None:
        final_ckpt["ema_state_dict"] = ema_tracker.state_dict()

    final_path = Path(config.output_dir) / "stardiff_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"✓ Final model saved to {final_path}")

    return trained_model, scheduler, config, accelerator


if __name__ == "__main__":
    main()
