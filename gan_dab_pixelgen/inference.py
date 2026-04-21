"""
GAN DABPixelGen Inference.

Single forward pass: H&E → DAB density + IHC RGB.
No iterative sampling — unlike the diffusion version, inference is O(1).

Usage:
    from gan_dab_pixelgen.inference import run_inference
    results = run_inference(he_image, generator, device="cuda")
    ihc_rgb = results["ihc_rgb"]   # [B, 3, H, W] in [0, 1]
    dab_pred = results["dab_pred"] # [B, 1, H, W]
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Dict, Optional, List

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.stain_utils import (
    StainDeconvolution,
    normalize_h_density,
    analytical_recompose,
    fit_h_normalization_analytical,
)


# ── Single-image / batch inference ───────────────────────────────────────────

@torch.no_grad()
def run_inference(
    he_image:   torch.Tensor,          # [3, H, W] in [0, 1]  or  [B, 3, H, W]
    generator,                         # DABUNetGenerator (eval mode preferred)
    device:     str   = "cuda",
    use_analytical_ihc: bool = False,  # if True, use Beer-Lambert recomp instead of CombinationNet
) -> Dict[str, torch.Tensor]:
    """
    Run GAN DABPixelGen inference.

    Args:
        he_image: H&E image(s) in [0, 1].
                  Shape [3, H, W] (single) or [B, 3, H, W] (batch).
        generator: Trained DABUNetGenerator in eval mode.
        device:    CUDA device string.
        use_analytical_ihc: Override CombinationNet with analytical Beer-Lambert
                  recomposition (useful for ablation / interpretability study).

    Returns dict:
        ihc_rgb:       [B, 3, H, W] in [0, 1]  — final IHC image
        dab_pred:      [B, 1, H, W]             — predicted DAB density
        h_norm_params: [B, 2]                   — (a_raw, b_raw) from HNormHead
        h_ihc:         [B, 1, H, W]             — H density after normalization
                                                  (only populated if analytical)
        h_he_density:  [B, 1, H, W]             — H density from H&E deconvolution
    """
    if he_image.dim() == 3:
        he_image = he_image.unsqueeze(0)

    he_image = he_image.to(device)
    B, _, H, W = he_image.shape

    # Deconvolve H&E to get hematoxylin density (for analytical fallback)
    deconv = StainDeconvolution().to(device)
    h_he_density = deconv(he_image)["hematoxylin"]   # [B, 1, H, W]

    # Normalise H&E to [-1, 1] for generator input
    he_11 = he_image * 2 - 1   # [B, 3, H, W]

    # Single forward pass
    generator.eval()
    dab_pred, ihc_pred_11, h_norm_params = generator(he_11)
    # dab_pred:      [B, 1, H, W]
    # ihc_pred_11:   [B, 3, H, W] in [-1, 1]
    # h_norm_params: [B, 2]

    if use_analytical_ihc:
        # Analytical recomposition via learned H normalisation
        h_ihc = normalize_h_density(h_he_density, h_norm_params[:, 0], h_norm_params[:, 1])
        ihc_rgb = analytical_recompose(h_ihc, dab_pred, device=device)
    else:
        # CombinationNet output (primary path)
        ihc_rgb = (ihc_pred_11.clamp(-1, 1) + 1) / 2   # [-1, 1] → [0, 1]
        h_ihc = normalize_h_density(h_he_density, h_norm_params[:, 0], h_norm_params[:, 1])

    return {
        "ihc_rgb":       ihc_rgb,           # [B, 3, H, W] in [0, 1]
        "dab_pred":      dab_pred,          # [B, 1, H, W]
        "h_norm_params": h_norm_params,     # [B, 2]
        "h_ihc":         h_ihc,             # [B, 1, H, W]
        "h_he_density":  h_he_density,      # [B, 1, H, W]
        "ihc_pred_11":   ihc_pred_11,       # [B, 3, H, W] in [-1, 1]
    }


# ── Ablation comparison ───────────────────────────────────────────────────────

@torch.no_grad()
def inference_comparison(
    generator,
    val_dataloader,
    device:      str = "cuda",
    num_samples: int = 5,
) -> Dict:
    """
    Compare GAN output vs analytical Beer-Lambert recomposition as ablation.

    Modes:
      A) CombinationNet output (full model)
      B) Analytical Beer-Lambert + learned H-norm (interpretable fallback)
      C) Analytical Beer-Lambert + percentile-matching H-norm (no model)
      D) GT DAB + analytical recomposition (DAB oracle upper bound)
    """
    generator.eval()
    batch = next(iter(val_dataloader))
    he       = batch["he"][:num_samples].to(device)       # [-1, 1]
    ihc_gt   = batch["ihc_01"][:num_samples].to(device)   # [0, 1]
    h_he     = batch["h_he_density"][:num_samples].to(device)
    dab_gt   = batch["dab_gt"][:num_samples].to(device)

    dab_pred, ihc_pred_11, h_norm = generator(he)
    ihc_combo = (ihc_pred_11.clamp(-1, 1) + 1) / 2

    # B: analytical with learned H-norm
    h_ihc_learned = normalize_h_density(h_he, h_norm[:, 0], h_norm[:, 1])
    ihc_analytic  = analytical_recompose(h_ihc_learned, dab_pred, device=device)

    # C: analytical with percentile-matching H-norm (no model)
    h_list = []
    for b in range(he.shape[0]):
        a_p, offset = fit_h_normalization_analytical(h_he[b])
        h_list.append((a_p * h_he[b] + offset).clamp(0))
    h_ihc_pct = torch.stack(h_list, 0)
    ihc_analytic_pct = analytical_recompose(h_ihc_pct, dab_pred, device=device)

    # D: GT DAB oracle
    ihc_oracle = analytical_recompose(h_ihc_learned, dab_gt, device=device)

    results = {
        "he":                 ((he + 1) / 2).clamp(0, 1).cpu(),
        "ihc_gt":             ihc_gt.cpu(),
        "A_combo":            ihc_combo.cpu(),
        "B_analytic_learned": ihc_analytic.cpu(),
        "C_analytic_pct":     ihc_analytic_pct.cpu(),
        "D_oracle":           ihc_oracle.cpu(),
    }

    # Metrics
    from gan_dab_pixelgen.train import compute_val_metrics
    metrics = {}
    for key in ["A_combo", "B_analytic_learned", "C_analytic_pct", "D_oracle"]:
        m = compute_val_metrics(
            dab_pred, dab_gt,
            results[key].to(device), ihc_gt,
            device=device,
        )
        metrics[key] = m

    return results, metrics


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_comparison(results: Dict, metrics: Dict, save_path: Optional[str] = None):
    """Grid visualisation for ablation study."""
    keys   = ["he", "ihc_gt", "A_combo", "B_analytic_learned", "C_analytic_pct", "D_oracle"]
    titles = ["H&E", "GT IHC", "A: GAN", "B: Analytic+Learned", "C: Analytic+Pct", "D: Oracle"]
    n = results["he"].shape[0]

    fig, axes = plt.subplots(n, len(keys), figsize=(len(keys) * 4, n * 4))
    for i in range(n):
        for j, (key, title) in enumerate(zip(keys, titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(results[key][i].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if i == 0:
                if key in metrics:
                    m = metrics[key]
                    sub = (f"PSNR={m.get('val/ihc_psnr', 0):.1f}\n"
                           f"SSIM={m.get('val/ihc_ssim', 0):.3f}")
                    ax.set_title(f"{title}\n{sub}", fontsize=9)
                else:
                    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Practical metrics ─────────────────────────────────────────────────────────

class PracticalMetrics:
    """Quick PSNR / SSIM / LPIPS on a batch of images."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lpips_fn = None

    def _lpips(self):
        if self._lpips_fn is None:
            try:
                import lpips
                fn = lpips.LPIPS(net="alex").eval().to(self.device)
                for p in fn.parameters():
                    p.requires_grad = False
                self._lpips_fn = fn
            except ImportError:
                self._lpips_fn = False
        return self._lpips_fn if self._lpips_fn is not False else None

    @torch.no_grad()
    def compute_all(
        self,
        real: torch.Tensor,   # [B, 3, H, W] in [0, 1]
        fake: torch.Tensor,   # [B, 3, H, W] in [0, 1]
    ) -> Dict[str, float]:
        real = real.to(self.device)
        fake = fake.to(self.device)
        B = real.shape[0]

        mse = F.mse_loss(fake, real, reduction="none").reshape(B, -1).mean(1)
        psnr = (-10 * torch.log10(mse.clamp(1e-10))).mean().item()
        mae  = F.l1_loss(fake, real).item()

        lpips_val = float("nan")
        fn = self._lpips()
        if fn is not None:
            lpips_val = fn(fake * 2 - 1, real * 2 - 1).mean().item()

        return {"psnr": psnr, "mae": mae, "lpips": lpips_val}
