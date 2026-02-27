"""
StarDiff Inference & Ablation Study.
Generates samples with different pathway configurations.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Dict
from torch.utils.data import DataLoader
import wandb

from stardiff_pixelgen.metrics import PracticalMetrics
from stardiff_pixelgen.stardiff_scheduler import StarDiffScheduler


@torch.no_grad()
def inference_comparison(model, scheduler, val_dataloader, device="cuda", num_samples=5):
    """Generate samples with all pathway configurations for ablation."""
    model.eval()
    batch = next(iter(val_dataloader))
    he = batch["he"][:num_samples].to(device)
    ihc_real = batch["ihc"][:num_samples].to(device)
    shape = (he.shape[0], 3, he.shape[2], he.shape[3])

    print("Generating: Full Star-Diff (Restoration + Noise)...")
    gen_full = scheduler.sample(model, he, shape, device, use_restoration=True, use_noise=True)

    print("Generating: Restoration Path Only...")
    gen_rest = scheduler.sample(model, he, shape, device, use_restoration=True, use_noise=False)

    print("Generating: Noise Path Only (DDPM baseline)...")
    gen_noise = scheduler.sample(model, he, shape, device, use_restoration=False, use_noise=True)

    denorm = lambda x: ((x + 1) / 2).clamp(0, 1)
    results = {
        "he": denorm(he.cpu()), "ihc_real": denorm(ihc_real.cpu()),
        "gen_full": denorm(gen_full.cpu()),
        "gen_restoration": denorm(gen_rest.cpu()),
        "gen_noise": denorm(gen_noise.cpu()),
    }

    # Metrics
    evaluator = PracticalMetrics(device=device)
    metrics = {}
    for name in ["gen_full", "gen_restoration", "gen_noise"]:
        m = evaluator.compute_all(results["ihc_real"].to(device), results[name].to(device))
        metrics[name] = {
            "ssim_mean": m["ssim"], "ssim_std": m.get("ssim_std", 0),
            "psnr_mean": m["psnr"], "psnr_std": m.get("psnr_std", 0),
            "lpips": m["lpips"],
        }

    return results, metrics


def visualize_comparison(results, metrics, save_path=None):
    """Visualization grid comparing all pathway configurations."""
    num_samples = results["he"].shape[0]
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))
    titles = ["H&E Input", "Real IHC", "Full Star-Diff", "Restoration Only", "Noise Only (DDPM)"]
    keys = ["he", "ihc_real", "gen_full", "gen_restoration", "gen_noise"]

    for i in range(num_samples):
        for j, (title, key) in enumerate(zip(titles, keys)):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(results[key][i].permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<20} {'SSIM ↑':<18} {'PSNR ↑':<15} {'LPIPS ↓':<12}")
    print("-" * 80)
    for name, m in metrics.items():
        display = name.replace("gen_", "").replace("_", " ").title()
        print(f"{display:<20} {m['ssim_mean']:.4f}±{m['ssim_std']:.4f}  "
              f"{m['psnr_mean']:.2f}±{m['psnr_std']:.2f}  {m['lpips']:.4f}")
    print("=" * 80)
