# # Star-Diff + PixelGen: Dual-Path Restoration Diffusion for H&E → IHC
# ## Using PixelGen JiT_I2I XL Backbone (Pretrained)
# 
# Combines:
# - **Star-Diff** (arXiv:2508.02528): Dual-path restoration diffusion
# - **PixelGen** (arXiv:2602.02493): JiT_I2I ViT backbone + perceptual losses
# 
# Architecture: Two independent **JiT_I2I_XL** backbones:
# - **Noise path** (ε_θ): predicts Gaussian noise ε
# - **Restoration path** (r_θ): predicts I_res = I_ihc - I_he
# 

# Install dependencies
# pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# pip install -q accelerate wandb kaggle lpips einops tqdm
# pip install -q "diffusers[torch]" transformers huggingface_hub
# pip install -q scikit-image pillow matplotlib opencv-python scipy
# pip install -q pytorch-msssim

import os
import sys
import wandb
from huggingface_hub import login as hf_login
import torch
import gc
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers.optimization import get_constant_schedule_with_warmup
import numpy as np

from stardiff_pixelgen.model import create_stardiff_model, SimpleEMA
from stardiff_pixelgen.config import StarDiffPixelGenConfig
from stardiff_pixelgen.stardiff_scheduler import StarDiffScheduler
from stardiff_pixelgen.train import resume_from_wandb

# Ensure PixelGen root is on path
PIXELGEN_ROOT = os.path.abspath(".")
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

# Download the pretrained XL 80ep (FMonly) weights directly
weight_url = "https://huggingface.co/zehongma/PixelGen/resolve/main/PixelGen_XL_80ep.ckpt"
weight_path = os.path.join(PIXELGEN_ROOT, "PixelGen_XL_80ep.ckpt")

if not os.path.exists(weight_path):
    print(f"Downloading weights to {weight_path}...")
    os.system(f"wget -q {weight_url} -O {weight_path}")
else:
    print(f"✓ Weights already downloaded: {weight_path}")

# Set your keys here
os.environ["HF_TOKEN"] = ""
os.environ["WANDB_API_KEY"] = ""
os.environ["KAGGLE_USERNAME"] = "ahmedayman4a77"

hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
wandb.login()
print(f"✓ W&B: {wandb.Api().viewer.username}")




config = StarDiffPixelGenConfig()
config.pretrained_weight_path = weight_path

effective = config.batch_size * config.gradient_accumulation_steps
print(f"\nConfig: JiT_I2I_{config.model_size} (hidden={config.hidden_size}, depth={config.depth})")
print(f"Batch: {config.batch_size} × {config.gradient_accumulation_steps} = {effective}")
print(f"EMA: decay={config.ema_decay} | LPIPS: {config.lpips_weight} | DINO: {config.use_dino}")

from stardiff_pixelgen.dataset import download_dataset, create_dataloaders

# download_dataset(config.dataset_root, config.stains)
train_loader, val_loader = create_dataloaders(config)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_stardiff_model(config, device)

# EMA tracker
ema_tracker = None
if config.use_ema:
    ema_tracker = SimpleEMA(model, decay=config.ema_decay, every_n_steps=config.ema_every_n_steps)



scheduler = StarDiffScheduler(
    num_timesteps=config.num_timesteps,
    restoration_weight=config.restoration_weight,
    he_init_alpha=config.he_init_alpha,
)

from stardiff_pixelgen.losses import PixelGenPerceptualLoss

perceptual_loss_fn = PixelGenPerceptualLoss(
    use_dino=config.use_dino,
    lpips_weight=config.lpips_weight,
    dino_weight=config.dino_weight,
    noise_gate_threshold=config.noise_gate_threshold,
    device=device,
)



gc.collect()
torch.cuda.empty_cache()

try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
except ImportError:
    print('⚠️ bitsandbytes not found. Falling back to standard AdamW.')
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)
model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, lr_scheduler
)
print(f"✓ Accelerator ready: {config.mixed_precision} precision")


# 1. Set the run path and filename
resume_run_id = "dfvt75jy"
run_path = f"mohamed-tarek2607/star-diff-he-to-ihc/{resume_run_id}"
checkpoint_file = "stardiff_epoch_27.pt"

resume_training = False  # Enabled for you
start_epoch = 0

if resume_training and resume_run_id:
    start_epoch = resume_from_wandb(
        run_path=run_path,
        file_name=checkpoint_file,
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        accelerator=accelerator,
        ema_tracker=ema_tracker
    )
    print(f"✓ Ready to resume from epoch {start_epoch}")
else:
    print("Starting fresh training from epoch 0")

from stardiff_pixelgen.train import train_stardiff

print("=" * 60)
print("Starting StarDiff + PixelGen Training: H&E → IHC")
print("=" * 60)

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
    he_init_alpha=config.he_init_alpha,
)


import torch
import numpy as np
from tqdm import tqdm
from src.eval.metrics import (
    compute_fid, compute_kid, compute_ssim_batch, 
    compute_psnr_batch, compute_mad_batch
)
import lpips
import torchvision.transforms.functional as TF

# 1. Setup evaluator and metrics
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_fn = lpips.LPIPS(net="alex").to(device)

model.eval()
all_preds = []
all_gts = []

ssim_scores = []
psnr_scores = []
mad_scores = []
lpips_scores = []

def denorm(x):
    return ((x + 1) / 2).clamp(0, 1)

# 2. Generate predictions on validation set
print(f"Generating predictions for {len(val_loader.dataset)} validation samples...")
for batch in tqdm(val_loader, desc="Evaluation"):
    he = batch["he"].to(device)
    ihc = batch["ihc"].to(device)
    
    with torch.no_grad():
        # Use the accelerator's unwrapped model
        unwrapped_model = accelerator.unwrap_model(model)
        shape = (he.shape[0], 3, config.image_size, config.image_size)
        
        gen = scheduler.sample(
            unwrapped_model,
            he, shape, 
            device=device,
            use_restoration=True, use_noise=True,
            progress=False
        )
        
        gen_01 = denorm(gen)
        ihc_01 = denorm(ihc)
        
        # Batch-based metrics
        ssim_scores.extend(compute_ssim_batch(gen_01, ihc_01))
        psnr_scores.extend(compute_psnr_batch(gen_01, ihc_01))
        mad_scores.extend(compute_mad_batch(gen_01, ihc_01))
        
        # LPIPS expects [-1, 1]
        l_scores = lpips_fn(gen.to(device), ihc.to(device)).view(-1).cpu().tolist()
        lpips_scores.extend(l_scores)
        
        # Collect for FID/KID (CPU to save VRAM)
        all_preds.append(gen_01.cpu())
        all_gts.append(ihc_01.cpu())

# 3. Aggregate images for Distributional Metrics (FID/KID)
all_preds_tensor = torch.cat(all_preds, dim=0)
all_gts_tensor = torch.cat(all_gts, dim=0)

print("\nComputing Distributional Metrics (FID/KID)... Cultivating Inception features...")
fid_value = compute_fid(all_preds_tensor, all_gts_tensor, device=device)
kid_mean, kid_std = compute_kid(all_preds_tensor, all_gts_tensor, device=device)

# 4. Final Results Summary
print(f"\n{'='*60}")
print(f"VAL EVALUATION RESULTS (N={len(ssim_scores)})")
print(f"{'='*60}")
print(f"{'Metric':<15} | {'Mean':<12} | {'Std':<12}")
print(f"{'='*15}-|-{'='*12}-|-{'='*12}")
print(f"{'SSIM ↑':<15} | {np.mean(ssim_scores):<12.4f} | {np.std(ssim_scores):<12.4f}")
print(f"{'PSNR ↑':<15} | {np.mean(psnr_scores):<12.4f} | {np.std(psnr_scores):<12.4f}")
print(f"{'LPIPS ↓':<15} | {np.mean(lpips_scores):<12.4f} | {np.std(lpips_scores):<12.4f}")
print(f"{'MAD ↓':<15} | {np.mean(mad_scores):<12.4f} | {np.std(mad_scores):<12.4f}")
print(f"{'FID ↓':<15} | {fid_value:<12.4f} | {'N/A':<12}")
print(f"{'KID ↓':<15} | {kid_mean:<12.4f} | {kid_std:<12.4f}")
print(f"{'='*60}")



# 1. Prepare metrics dictionary
final_val_metrics = {
    "val/final_ssim": np.mean(ssim_scores),
    "val/final_psnr": np.mean(psnr_scores),
    "val/final_lpips": np.mean(lpips_scores),
    "val/final_mad": np.mean(mad_scores),
    "val/final_fid": fid_value,
    "val/final_kid": kid_mean
}

# 2. Log to active W&B run
if wandb.run is not None:
    wandb.log(final_val_metrics)
    print("\u2713 Successfully reported final metrics to WandB summary:")
    for k, v in final_val_metrics.items():
        print(f"  - {k}: {v:.4f}")
else:
    print("\u26a0\ufe0f No active WandB run found. Metrics printed above but not logged to the cloud.")