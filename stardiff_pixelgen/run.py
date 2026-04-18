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
from stardiff_pixelgen.model import create_stardiff_model, create_stardiff_model_for_finetune, SimpleEMA

from stardiff_pixelgen.losses import PixelGenPerceptualLoss
from stardiff_pixelgen.train import train_stardiff, resume_from_wandb
from src.diffusion.flow_matching.dap_loss import CombinedDABLoss


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
        schedule_type=config.restoration_schedule_type,
        integration_method=config.integration_method,
        he_init_alpha=config.he_init_alpha,
        late_t_threshold=config.late_t_threshold,
        use_ot_coupling=config.use_ot_coupling,
        ot_feature_size=config.ot_feature_size,
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
    import sys
    # Parse --key value overrides from CLI
    args = sys.argv[1:]
    positional = []
    overrides = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:]
            val = args[i + 1]
            # Try to convert to int/float
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            overrides[key] = val
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if len(positional) > 0 and positional[0] == "finetune":
        ckpt = positional[1] if len(positional) > 1 else "stardiff_best.pt"
        res = int(positional[2]) if len(positional) > 2 else 512
        if res >= 1024:
            finetune_1024(checkpoint_path=ckpt, new_resolution=res, **overrides)
        else:
            finetune_512(checkpoint_path=ckpt, new_resolution=res, **overrides)
    else:
        main()


def finetune_512(
    checkpoint_path: str,
    new_resolution: int = 512,
    **config_overrides,
):
    """
    Finetune a trained StarDiff model at higher resolution.

    Loads 256 checkpoint -> rescales pos_embed (bicubic) + RoPE -> trains at new_resolution.
    The preprocessed 512 patches are used at native resolution (no resize).

    Usage:
        # From CLI:
        python -m stardiff_pixelgen.run finetune path/to/stardiff_best.pt 512

        # From notebook:
        from stardiff_pixelgen.run import finetune_512
        finetune_512(checkpoint_path="stardiff_best.pt")

    Args:
        checkpoint_path: Path to trained StarDiff .pt checkpoint (from 256 training)
        new_resolution: Target resolution (default 512)
        **config_overrides: Override any StarDiffPixelGenConfig field
    """
    from stardiff_pixelgen.stardiff_scheduler import StarDiffScheduler

    # GPU detection
    gpu_type, batch_size, grad_accum, precision = detect_gpu()

    # Config for finetuning
    config = StarDiffPixelGenConfig()
    config.configure_for_finetune(checkpoint_path, new_resolution)
    config.mixed_precision = precision

    # Apply any user overrides
    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    effective = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"StarDiff 512 Finetuning from {checkpoint_path}")
    print(f"{'='*60}")
    print(f"  Resolution: {config.image_size}x{config.image_size}")
    print(f"  Model: JiT_I2I_{config.model_size}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective}")
    print(f"  LR: {config.learning_rate} | Epochs: {config.num_epochs}")
    print(f"{'='*60}\n")

    # Dataset (preprocessed 512 patches at native resolution)
    download_dataset(config.dataset_root, config.stains)
    train_loader, val_loader = create_dataloaders(config)

    # Model: load 256 checkpoint -> rescale to 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_stardiff_model_for_finetune(config, checkpoint_path, new_resolution, device)

    # Scheduler
    scheduler = StarDiffScheduler(
        num_timesteps=config.num_timesteps,
        restoration_weight=config.restoration_weight,
        schedule_type=config.restoration_schedule_type,
        integration_method=config.integration_method,
        he_init_alpha=config.he_init_alpha,
        late_t_threshold=config.late_t_threshold,
        use_ot_coupling=config.use_ot_coupling,
        ot_feature_size=config.ot_feature_size,
    )

    # Loss
    perceptual_loss_fn = PixelGenPerceptualLoss(
        use_dino=config.use_dino,
        lpips_weight=config.lpips_weight,
        dino_weight=config.dino_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        device=device,
    )

    # DAB stain-aware loss
    dab_loss_fn = None
    if config.dab_weight > 0:
        dab_loss_fn = CombinedDABLoss(
            patch_sizes=list(config.dab_patch_sizes),
            use_focal=config.dab_use_focal,
            focal_alpha=config.dab_focal_alpha,
            hist_weight=config.dab_hist_weight,
            fod_threshold=config.dab_fod_threshold,
            weight_alpha=config.dab_weight_alpha,
        ).to(device)
        for p in dab_loss_fn.parameters():
            p.requires_grad = False
        dab_loss_fn.eval()

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
    except ImportError:
        optimizer = AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)

    # Accelerator
    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting 512 Finetuning: H&E -> IHC")
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
        dab_loss_fn=dab_loss_fn,
    )

    # Save final
    final_ckpt = {
        "model_type": "stardiff_pixelgen_finetune",
        "model_state_dict": accelerator.unwrap_model(trained_model).state_dict(),
        "config": {k: str(v) for k, v in vars(config).items()},
        "original_checkpoint": checkpoint_path,
        "resolution": new_resolution,
    }
    final_path = Path(config.output_dir) / f"stardiff_finetune_{new_resolution}_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\n✓ Finetuned model saved to {final_path}")

    return trained_model, scheduler, config, accelerator


def finetune_1024(
    checkpoint_path: str,
    new_resolution: int = 1024,
    dataset_root_1024: str = None,
    **config_overrides,
):
    """
    Finetune a trained StarDiff 512 model at 1024×1024 resolution.

    Uses the ORIGINAL MIST 1024×1024 paired images (TrainValAB layout).
    Loads 512 checkpoint → rescales pos_embed (bicubic) + RoPE → trains at 1024.

    ⚠️ 1024×1024 = 4096 tokens per path → ~64× attention cost vs 256.
    Expect: batch_size=1, heavy grad accumulation, ~80GB VRAM minimum.

    Usage:
        python -m stardiff_pixelgen.run finetune path/to/stardiff_512.pt 1024

    Args:
        checkpoint_path: Path to trained StarDiff 512 .pt checkpoint
        new_resolution: Target resolution (default 1024)
        dataset_root_1024: Root dir for MIST 1024 data (default: ./data)
        **config_overrides: Override any config field
    """
    from stardiff_pixelgen.stardiff_scheduler import StarDiffScheduler
    from stardiff_pixelgen.dataset import download_dataset_1024, create_dataloaders_1024

    # GPU detection
    gpu_type, batch_size, grad_accum, precision = detect_gpu()

    # Config for 1024 finetuning
    config = StarDiffPixelGenConfig()
    config.configure_for_finetune(checkpoint_path, new_resolution)
    config.mixed_precision = precision

    # 1024 data comes from MIST original, not the preprocessed 512 Kaggle data
    if dataset_root_1024:
        config.dataset_root = dataset_root_1024

    # Apply any user overrides
    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    effective = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"StarDiff 1024 Finetuning from {checkpoint_path}")
    print(f"{'='*60}")
    print(f"  Resolution: {config.image_size}x{config.image_size}")
    print(f"  Model: JiT_I2I_{config.model_size}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective}")
    print(f"  LR: {config.learning_rate} | Epochs: {config.num_epochs}")
    print(f"  Losses: LPIPS={config.lpips_weight} DINO={config.dino_weight} DAB={config.dab_weight}")
    print(f"{'='*60}\n")

    # Dataset (original MIST 1024×1024 at native resolution)
    download_dataset_1024(config.dataset_root, config.stains)
    train_loader, val_loader = create_dataloaders_1024(config)

    # Model: load 512 checkpoint → rescale to 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_stardiff_model_for_finetune(config, checkpoint_path, new_resolution, device)

    # Scheduler
    scheduler = StarDiffScheduler(
        num_timesteps=config.num_timesteps,
        restoration_weight=config.restoration_weight,
        schedule_type=config.restoration_schedule_type,
        integration_method=config.integration_method,
        he_init_alpha=config.he_init_alpha,
        late_t_threshold=config.late_t_threshold,
        use_ot_coupling=config.use_ot_coupling,
        ot_feature_size=config.ot_feature_size,
    )

    # Perceptual Loss
    perceptual_loss_fn = PixelGenPerceptualLoss(
        use_dino=config.use_dino,
        lpips_weight=config.lpips_weight,
        dino_weight=config.dino_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        device=device,
    )

    # DAB stain-aware loss
    dab_loss_fn = None
    if config.dab_weight > 0:
        dab_loss_fn = CombinedDABLoss(
            patch_sizes=list(config.dab_patch_sizes),
            use_focal=config.dab_use_focal,
            focal_alpha=config.dab_focal_alpha,
            hist_weight=config.dab_hist_weight,
            fod_threshold=config.dab_fod_threshold,
            weight_alpha=config.dab_weight_alpha,
        ).to(device)
        for p in dab_loss_fn.parameters():
            p.requires_grad = False
        dab_loss_fn.eval()

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
    except ImportError:
        optimizer = AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)

    # Accelerator
    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting 1024×1024 Finetuning: H&E → IHC")
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
        dab_loss_fn=dab_loss_fn,
    )

    # Save final
    final_ckpt = {
        "model_type": "stardiff_pixelgen_finetune_1024",
        "model_state_dict": accelerator.unwrap_model(trained_model).state_dict(),
        "config": {k: str(v) for k, v in vars(config).items()},
        "original_checkpoint": checkpoint_path,
        "resolution": new_resolution,
    }
    final_path = Path(config.output_dir) / f"stardiff_finetune_{new_resolution}_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\n✓ 1024 Finetuned model saved to {final_path}")

    return trained_model, scheduler, config, accelerator
