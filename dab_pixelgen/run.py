"""
DABPixelGen: Main entry point.

Usage (script):
    python -m dab_pixelgen.run
    python -m dab_pixelgen.run finetune path/to/dab_pixelgen_best.pt 512
    python -m dab_pixelgen.run finetune path/to/dab_pixelgen_512.pt 1024

Usage (notebook):
    from dab_pixelgen.run import main
    main(pretrained_weight_path="./PixelGen_XL_80ep.ckpt")
"""
import gc
import os
import sys

import torch
from pathlib import Path
from accelerate import Accelerator
from torch.optim import AdamW
from diffusers.optimization import get_constant_schedule_with_warmup

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.config import DABPixelGenConfig, detect_gpu
from dab_pixelgen.dataset import download_dataset, create_dataloaders, create_dataloaders_1024
from dab_pixelgen.model import create_dab_model, create_dab_model_for_finetune, SimpleEMA
from dab_pixelgen.scheduler import DABFlowScheduler
from dab_pixelgen.losses import DABPixelGenLoss
from dab_pixelgen.train import train_dab, resume_from_wandb


# ── Environment loading (.env) ────────────────────────────────────────────────

def _load_env():
    """
    Load environment variables from .env files, if present.

    Search order:
      1) <repo_root>/.env
      2) <repo_root>/dab_pixelgen/.env
    """
    repo_env = os.path.join(PIXELGEN_ROOT, ".env")
    dab_env = os.path.join(PIXELGEN_ROOT, "dab_pixelgen", ".env")

    try:
        from dotenv import load_dotenv
        loaded = False
        if os.path.exists(repo_env):
            load_dotenv(repo_env, override=False)
            loaded = True
            print(f"Loaded environment file: {repo_env}")
        if os.path.exists(dab_env):
            load_dotenv(dab_env, override=False)
            loaded = True
            print(f"Loaded environment file: {dab_env}")
        if not loaded:
            print("No .env file found (continuing with existing environment variables).")
    except ImportError:
        print("python-dotenv is not installed; skipping .env loading.")
        print("Install with: pip install python-dotenv")


_load_env()


# ── Optimizer helper ──────────────────────────────────────────────────────────

def _make_optimizer(model, config):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
        print("Using bitsandbytes 8-bit AdamW")
    except ImportError:
        print("bitsandbytes not found -- using standard AdamW.")
        opt = AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=1e-2,
        )
    return opt


# ── Auto-download PixelGen weights ────────────────────────────────────────────

def _ensure_pretrained_weights(config):
    if not config.pretrained_weight_path:
        return
    path = os.path.abspath(config.pretrained_weight_path)
    if os.path.exists(path):
        print(f"Pretrained weights: {path}")
        return
    if "PixelGen_XL" in path:
        fname = os.path.basename(path)
        url = f"https://huggingface.co/zehongma/PixelGen/resolve/main/{fname}"
        print(f"Downloading {fname} from HuggingFace...")
        os.system(f"wget -q {url} -O {path}")
        if os.path.exists(path):
            print("  Downloaded successfully.")
        else:
            print("  WARNING: Download failed. Starting from random init.")


# ── Main training entry point ─────────────────────────────────────────────────

def main(
    pretrained_weight_path: str = None,
    resume_wandb: bool = False,
    resume_run_path: str = None,
    resume_file_name: str = "dab_pixelgen_best.pt",
    **config_overrides,
):
    """Full DABPixelGen training pipeline."""
    # 1. GPU detection

    # 2. Config
    config = DABPixelGenConfig()
    if pretrained_weight_path:
        config.pretrained_weight_path = pretrained_weight_path
    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    _ensure_pretrained_weights(config)

    effective = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"DABPixelGen Config")
    print(f"{'='*60}")
    print(f"  Image: {config.image_size}x{config.image_size} from {config.source_image_size}")
    print(f"  Model: JiT_I2I_{config.model_size} (hidden={config.hidden_size}, depth={config.depth})")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective}")
    print(f"  Precision: {config.mixed_precision}")
    print(f"  Losses: FM={config.fm_weight} DAB={config.dab_weight} "
          f"Recomp={config.recomp_weight} LPIPS={config.lpips_weight}")
    print(f"  Noise gate: t >= {config.noise_gate_threshold}")
    print(f"  H&E warm start alpha: {config.he_init_alpha}")
    print(f"{'='*60}\n")

    # 3. Dataset
    # download_dataset(config.dataset_root, config.stains)
    train_loader, val_loader = create_dataloaders(config)

    # 4. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dab_model(config, device)

    # 5. Scheduler
    scheduler = DABFlowScheduler(
        num_timesteps=config.num_timesteps,
        he_init_alpha=config.he_init_alpha,
    )

    # 6. Loss
    loss_fn = DABPixelGenLoss(
        fm_weight=config.fm_weight,
        dab_weight=config.dab_weight,
        recomp_weight=config.recomp_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        lpips_weight=config.lpips_weight,
        device=device,
    ).to(device)

    # 7. Optimizer + LR scheduler
    optimizer = _make_optimizer(model, config)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps
    )

    # 8. EMA
    ema_tracker = None
    if config.use_ema:
        ema_tracker = SimpleEMA(model, config.ema_decay, config.ema_every_n_steps)

    # 9. Accelerator
    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    print(f"Accelerator: {config.mixed_precision}, grad_accum={config.gradient_accumulation_steps}")

    # 10. Optional resume
    start_epoch = 0
    if resume_wandb and resume_run_path:
        start_epoch = resume_from_wandb(
            resume_run_path, resume_file_name,
            model, optimizer, lr_scheduler, accelerator,
            ema_tracker=ema_tracker,
        )
    else:
        print("Starting fresh training from epoch 1")

    # 11. Train
    print(f"\n{'='*60}")
    print("Starting DABPixelGen Training: H&E -> DAB -> IHC RGB")
    print(f"{'='*60}\n")

    trained_model = train_dab(
        model=model,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        config=config,
        loss_fn=loss_fn,
        ema_tracker=ema_tracker,
        start_epoch=start_epoch,
    )

    # 12. Save final
    final_ckpt = {
        "model_type":       "dab_pixelgen",
        "model_state_dict": accelerator.unwrap_model(trained_model).state_dict(),
        "config":           {k: str(v) for k, v in vars(config).items()},
        "scheduler_config": {
            "num_timesteps": scheduler.num_timesteps,
            "he_init_alpha": scheduler.he_init_alpha,
        },
    }
    if ema_tracker is not None:
        final_ckpt["ema_state_dict"] = ema_tracker.state_dict()

    final_path = Path(config.output_dir) / "dab_pixelgen_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\nFinal model saved to {final_path}")
    return trained_model, scheduler, config, accelerator


# ── Finetuning entry points ───────────────────────────────────────────────────

def finetune_512(checkpoint_path: str, new_resolution: int = 512, **config_overrides):
    """
    Finetune a trained DABPixelGen model at 512x512 resolution.

    Steps:
      1. Load 256 checkpoint -> rescale pos_embed (bicubic) + RoPE -> train at 512
      2. Enable LPIPS for finetuning

    Usage:
        python -m dab_pixelgen.run finetune path/to/dab_pixelgen_best.pt 512
    """
    _, _, _, precision = detect_gpu()

    config = DABPixelGenConfig()
    config.configure_for_finetune(checkpoint_path, new_resolution)
    config.mixed_precision = precision

    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    effective = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"DABPixelGen 512 Finetuning from {checkpoint_path}")
    print(f"{'='*60}")
    print(f"  Resolution: {config.image_size}x{config.image_size}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective}")
    print(f"  LR: {config.learning_rate} | Epochs: {config.num_epochs}")
    print(f"  Losses: LPIPS={config.lpips_weight}")
    print(f"{'='*60}\n")

    # download_dataset(config.dataset_root, config.stains)
    train_loader, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dab_model_for_finetune(config, checkpoint_path, new_resolution, device)

    scheduler = DABFlowScheduler(
        num_timesteps=config.num_timesteps,
        he_init_alpha=config.he_init_alpha,
    )

    loss_fn = DABPixelGenLoss(
        fm_weight=config.fm_weight,
        dab_weight=config.dab_weight,
        recomp_weight=config.recomp_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        lpips_weight=config.lpips_weight,
        device=device,
    ).to(device)

    optimizer   = _make_optimizer(model, config)
    lr_sched    = get_constant_schedule_with_warmup(optimizer, config.warmup_steps)

    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_sched = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_sched
    )

    print(f"\n{'='*60}")
    print("Starting 512 Finetuning: H&E -> DAB -> IHC RGB")
    print(f"{'='*60}\n")

    trained_model = train_dab(
        model=model,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_sched,
        accelerator=accelerator,
        config=config,
        loss_fn=loss_fn,
    )

    final_ckpt = {
        "model_type":            "dab_pixelgen_finetune",
        "model_state_dict":      accelerator.unwrap_model(trained_model).state_dict(),
        "config":                {k: str(v) for k, v in vars(config).items()},
        "original_checkpoint":   checkpoint_path,
        "resolution":            new_resolution,
    }
    final_path = Path(config.output_dir) / f"dab_pixelgen_finetune_{new_resolution}_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\nFinetuned model saved to {final_path}")
    return trained_model, scheduler, config, accelerator


def finetune_1024(
    checkpoint_path: str,
    new_resolution:  int = 1024,
    dataset_root_1024: str = None,
    **config_overrides,
):
    """
    Finetune at 1024x1024 resolution from a 512 checkpoint.
    Requires the original MIST 1024 dataset (TrainValAB layout).

    Usage:
        python -m dab_pixelgen.run finetune path/to/dab_pixelgen_512.pt 1024
    """
    _, _, _, precision = detect_gpu()

    config = DABPixelGenConfig()
    config.configure_for_finetune(checkpoint_path, new_resolution)
    config.mixed_precision = precision
    if dataset_root_1024:
        config.dataset_root = dataset_root_1024

    for k, v in config_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    effective = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"DABPixelGen 1024 Finetuning from {checkpoint_path}")
    print(f"{'='*60}")
    print(f"  Resolution: {config.image_size}x{config.image_size}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective}")
    print(f"  LR: {config.learning_rate} | Epochs: {config.num_epochs}")
    print(f"{'='*60}\n")

    train_loader, val_loader = create_dataloaders_1024(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dab_model_for_finetune(config, checkpoint_path, new_resolution, device)

    scheduler = DABFlowScheduler(
        num_timesteps=config.num_timesteps,
        he_init_alpha=config.he_init_alpha,
    )

    loss_fn = DABPixelGenLoss(
        fm_weight=config.fm_weight,
        dab_weight=config.dab_weight,
        recomp_weight=config.recomp_weight,
        noise_gate_threshold=config.noise_gate_threshold,
        lpips_weight=config.lpips_weight,
        device=device,
    ).to(device)

    optimizer = _make_optimizer(model, config)
    lr_sched  = get_constant_schedule_with_warmup(optimizer, config.warmup_steps)

    gc.collect()
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    model, optimizer, train_loader, val_loader, lr_sched = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_sched
    )

    print(f"\n{'='*60}")
    print("Starting 1024x1024 Finetuning: H&E -> DAB -> IHC RGB")
    print(f"{'='*60}\n")

    trained_model = train_dab(
        model=model,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_sched,
        accelerator=accelerator,
        config=config,
        loss_fn=loss_fn,
    )

    final_ckpt = {
        "model_type":          "dab_pixelgen_finetune_1024",
        "model_state_dict":    accelerator.unwrap_model(trained_model).state_dict(),
        "config":              {k: str(v) for k, v in vars(config).items()},
        "original_checkpoint": checkpoint_path,
        "resolution":          new_resolution,
    }
    final_path = Path(config.output_dir) / f"dab_pixelgen_finetune_{new_resolution}_final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\n1024 Finetuned model saved to {final_path}")
    return trained_model, scheduler, config, accelerator


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    positional, overrides = [], {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:]
            val = args[i + 1]
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

    if positional and positional[0] == "finetune":
        ckpt = positional[1] if len(positional) > 1 else "dab_pixelgen_best.pt"
        res  = int(positional[2]) if len(positional) > 2 else 512
        if res >= 1024:
            finetune_1024(ckpt, res, **overrides)
        else:
            finetune_512(ckpt, res, **overrides)
    else:
        main(**overrides)
