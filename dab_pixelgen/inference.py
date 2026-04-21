"""
DABPixelGen Inference.

Full pipeline:
  1. Stain-deconvolve H&E -> H_HE density
  2. Sample DAB density via flow matching (50 Euler steps)
  3. Apply learned per-image H normalization: H_IHC = a * H_HE + b
  4. Analytically recompose IHC RGB via Beer-Lambert

Also includes:
  - Ablation study: compare GT DAB recomposition vs predicted
  - Batch-level metrics: SSIM, PSNR, LPIPS
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Dict, Optional, Tuple

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.stain_utils import (
    StainDeconvolution,
    normalize_h_density,
    analytical_recompose,
    fit_h_normalization_analytical,
)


# ── Single-image inference ────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    he_image:      torch.Tensor,   # [3, H, W] in [0, 1]  or  [B, 3, H, W]
    model,                         # DABPixelGenModel
    scheduler,                     # DABFlowScheduler
    device:        str = "cuda",
    use_analytical_h_norm: bool = False,   # override learned H-norm with percentile matching
    ihc_h_p5:      float = 0.08,           # training-set calibration for analytical H-norm
    ihc_h_p95:     float = 1.20,
    progress:      bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Run the full DABPixelGen inference pipeline.

    Args:
        he_image: H&E image(s) in [0, 1].
                  Shape [3, H, W] (single) or [B, 3, H, W] (batch).
        model:    Trained DABPixelGenModel (eval mode preferred).
        scheduler: DABFlowScheduler.
        device:   CUDA device string.
        use_analytical_h_norm: If True, override the learned (a, b) with
                  analytical percentile matching (Option A2 from spec).

    Returns dict with:
        ihc_rgb:       [B, 3, H, W] in [0, 1]  -- final recomposed IHC
        dab_pred:      [B, 1, H, W]              -- predicted DAB density
        h_ihc:         [B, 1, H, W]              -- normalized H density
        h_he_density:  [B, 1, H, W]              -- H density from H&E
        h_norm_params: [B, 2]                    -- raw (a, b) from model
    """
    if he_image.dim() == 3:
        he_image = he_image.unsqueeze(0)   # [1, 3, H, W]

    he_image = he_image.to(device)
    B, _, H, W = he_image.shape

    # 1. Stain-deconvolve H&E to get H density
    deconv = StainDeconvolution().to(device)
    h_he_density = deconv(he_image)["hematoxylin"]  # [B, 1, H, W]

    # Normalize H&E to [-1, 1] for model conditioning
    he_11 = he_image * 2 - 1  # [B, 3, H, W]

    # 2. Sample DAB density
    shape = (B, 1, H, W)
    dab_pred, h_norm_params = scheduler.sample(
        model,
        he_condition=he_11,
        shape=shape,
        device=device,
        h_he_density=h_he_density,
        progress=progress,
    )

    # 3. Determine H normalization parameters
    if use_analytical_h_norm:
        h_ihc_list = []
        for b in range(B):
            a, offset = fit_h_normalization_analytical(
                h_he_density[b], ihc_h_p5, ihc_h_p95
            )
            h_norm = (a * h_he_density[b] + offset).clamp(min=0)
            h_ihc_list.append(h_norm)
        h_ihc = torch.stack(h_ihc_list, dim=0)  # [B, 1, H, W]
    else:
        a_raw = h_norm_params[:, 0]
        b_raw = h_norm_params[:, 1]
        h_ihc = normalize_h_density(h_he_density, a_raw, b_raw)

    # 4. Analytical Beer-Lambert recomposition
    ihc_rgb = analytical_recompose(h_ihc, dab_pred, device=device)

    return {
        "ihc_rgb":      ihc_rgb,          # [B, 3, H, W] in [0, 1]
        "dab_pred":     dab_pred,          # [B, 1, H, W]
        "h_ihc":        h_ihc,            # [B, 1, H, W]
        "h_he_density": h_he_density,     # [B, 1, H, W]
        "h_norm_params":h_norm_params,    # [B, 2]
    }


# ── Ablation comparison ───────────────────────────────────────────────────────

@torch.no_grad()
def inference_comparison(
    model,
    scheduler,
    val_dataloader,
    device:      str = "cuda",
    num_samples: int = 5,
):
    """
    Compare multiple inference modes for ablation study:
      A) Predicted DAB + Learned H-norm     (full model)
      B) Predicted DAB + Analytical H-norm  (no learned norm)
      C) GT DAB + Learned H-norm            (DAB oracle)
      D) GT DAB + Analytical H-norm         (full oracle upper bound)
    """
    model.eval()
    batch = next(iter(val_dataloader))
    he      = batch["he"][:num_samples].to(device)          # [-1, 1]
    ihc_gt  = batch["ihc_01"][:num_samples].to(device)      # [0, 1]
    h_he    = batch["h_he_density"][:num_samples].to(device)
    dab_gt  = batch["dab_gt"][:num_samples].to(device)

    shape = (he.shape[0], 1, he.shape[2], he.shape[3])
    dab_pred, h_norm = scheduler.sample(
        model, he, shape, device=device, h_he_density=h_he, progress=True
    )

    a_raw, b_raw = h_norm[:, 0], h_norm[:, 1]

    def recompose_mode(dab, use_gt_dab, use_analytical_norm):
        d = dab_gt if use_gt_dab else dab
        if use_analytical_norm:
            h_list = []
            for b in range(he.shape[0]):
                a, off = fit_h_normalization_analytical(h_he[b])
                h_list.append((a * h_he[b] + off).clamp(0))
            h = torch.stack(h_list, dim=0)
        else:
            h = normalize_h_density(h_he, a_raw, b_raw)
        return analytical_recompose(h, d, device=device)

    results = {
        "he":       ((he + 1) / 2).clamp(0, 1).cpu(),
        "ihc_gt":   ihc_gt.cpu(),
        "A_pred_learned":     recompose_mode(dab_pred, False, False).cpu(),
        "B_pred_analytical":  recompose_mode(dab_pred, False, True).cpu(),
        "C_oracle_learned":   recompose_mode(dab_gt,   True,  False).cpu(),
        "D_oracle_analytical":recompose_mode(dab_gt,   True,  True).cpu(),
    }

    # Metrics
    evaluator = PracticalMetrics(device=device)
    metrics = {}
    for name in ["A_pred_learned", "B_pred_analytical", "C_oracle_learned", "D_oracle_analytical"]:
        metrics[name] = evaluator.compute_all(ihc_gt, results[name].to(device))

    return results, metrics


def visualize_comparison(results, metrics, save_path=None):
    """Grid visualization for ablation study."""
    keys = ["he", "ihc_gt", "A_pred_learned", "B_pred_analytical", "C_oracle_learned", "D_oracle_analytical"]
    titles = ["H&E", "GT IHC", "A: Pred+Learned", "B: Pred+Analytic", "C: Oracle+Learned", "D: Oracle+Analytic"]
    n = results["he"].shape[0]

    fig, axes = plt.subplots(n, len(keys), figsize=(len(keys) * 4, n * 4))
    for i in range(n):
        for j, (key, title) in enumerate(zip(keys, titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(results[key][i].permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n{'='*90}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*90}")
    print(f"{'Mode':<25} {'SSIM':<10} {'PSNR':<10} {'LPIPS':<10}")
    print("-" * 90)
    for name, m in metrics.items():
        print(f"{name:<25} {m['ssim']:.4f}     {m['psnr']:.2f}     {m['lpips']:.4f}")
    print("=" * 90)


# ── Metrics ───────────────────────────────────────────────────────────────────

class PracticalMetrics:
    """SSIM, PSNR, LPIPS for evaluation."""

    def __init__(self, device="cuda"):
        self.device = device
        try:
            import lpips as lpips_lib
            self.lpips_fn = lpips_lib.LPIPS(net="alex").eval().to(device)
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        except ImportError:
            self.lpips_fn = None

    @staticmethod
    def _ssim(img1, img2, C1=1e-4, C2=9e-4):
        mu1 = F.avg_pool2d(img1.unsqueeze(0), 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2.unsqueeze(0), 11, stride=1, padding=5)
        s1  = F.avg_pool2d(img1.unsqueeze(0) ** 2, 11, stride=1, padding=5) - mu1 ** 2
        s2  = F.avg_pool2d(img2.unsqueeze(0) ** 2, 11, stride=1, padding=5) - mu2 ** 2
        s12 = F.avg_pool2d((img1 * img2).unsqueeze(0), 11, stride=1, padding=5) - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
        return (num / den).mean().item()

    @torch.no_grad()
    def compute_all(self, real, generated) -> Dict:
        real      = real.to(self.device)
        generated = generated.to(self.device)
        B = real.shape[0]

        ssim_vals = [self._ssim(generated[i], real[i]) for i in range(B)]
        mse = F.mse_loss(generated, real, reduction="none").reshape(B, -1).mean(1)
        psnr_vals = (-10 * torch.log10(mse.clamp(1e-10))).cpu().numpy()

        lpips_val = 0.0
        if self.lpips_fn is not None:
            lpips_val = self.lpips_fn(
                generated * 2 - 1, real * 2 - 1
            ).mean().item()

        return {
            "ssim":     float(np.mean(ssim_vals)),
            "ssim_std": float(np.std(ssim_vals)),
            "psnr":     float(psnr_vals.mean()),
            "psnr_std": float(psnr_vals.std()),
            "lpips":    float(lpips_val),
        }
