"""
Evaluation metrics: SSIM, PSNR, LPIPS.
"""
import torch
import torch.nn.functional as F
import numpy as np
import lpips
from typing import Dict


class PracticalMetrics:
    """Evaluation metrics for validation: SSIM, PSNR, LPIPS."""

    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    @staticmethod
    def _ssim_single(img1, img2, C1=0.01**2, C2=0.03**2):
        """SSIM between two [3,H,W] tensors in [0,1]."""
        mu1 = F.avg_pool2d(img1.unsqueeze(0), 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2.unsqueeze(0), 11, stride=1, padding=5)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1.unsqueeze(0) ** 2, 11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2.unsqueeze(0) ** 2, 11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d((img1 * img2).unsqueeze(0), 11, stride=1, padding=5) - mu1_mu2
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        return (num / den).mean().item()

    @torch.no_grad()
    def compute_all(self, real, generated) -> Dict:
        """
        Compute SSIM, PSNR, LPIPS.
        Inputs: [B, 3, H, W] in [0, 1] range.
        """
        real = real.to(self.device)
        generated = generated.to(self.device)
        b = real.shape[0]

        ssim_vals = [self._ssim_single(generated[i], real[i]) for i in range(b)]

        mse_per = F.mse_loss(generated, real, reduction="none").view(b, -1).mean(1)
        psnr_vals = (-10 * torch.log10(mse_per.clamp(min=1e-10))).cpu().numpy()

        lpips_val = self.lpips_fn(generated * 2 - 1, real * 2 - 1).mean().item()

        return {
            "ssim": float(np.mean(ssim_vals)),
            "ssim_std": float(np.std(ssim_vals)),
            "psnr": float(psnr_vals.mean()),
            "psnr_std": float(psnr_vals.std()),
            "lpips": float(lpips_val),
        }
