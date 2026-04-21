"""
Stain deconvolution and Beer-Lambert recomposition utilities.

Ruifrok & Johnston (2001) color deconvolution for H-DAB histology images.
All operations are differentiable (compatible with PyTorch autograd).

Key operations:
- StainDeconvolution: RGB -> H density + DAB density (+ residual)
- analytical_recompose: H density + DAB density -> RGB
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ruifrok & Johnston (2001) stain vectors [H, DAB, residual] x [R, G, B]
# Each row is a unit vector describing stain absorption per RGB channel.
STAIN_MATRIX = torch.tensor([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # DAB
    [0.711, 0.423, 0.561],  # Residual (background / eosin)
], dtype=torch.float32)


class StainDeconvolution(nn.Module):
    """
    Differentiable H-DAB stain deconvolution.

    Converts an RGB image (white-normalized, range [0, 1]) to per-stain
    optical density maps via the Ruifrok matrix inverse.

    Higher OD = more stain present.  Background pixels (white) -> OD ~ 0.
    """

    def __init__(self):
        super().__init__()
        stain_matrix_inv = torch.linalg.inv(STAIN_MATRIX)
        self.register_buffer("stain_matrix_inv", stain_matrix_inv)

    def forward(self, img: torch.Tensor) -> dict:
        """
        Args:
            img: [B, 3, H, W] RGB in [0, 1]
        Returns:
            dict:
                'hematoxylin': [B, 1, H, W]  H optical density
                'dab':         [B, 1, H, W]  DAB optical density
                'residual':    [B, 1, H, W]  residual OD
        """
        img = img.clamp(min=1e-6, max=1.0)
        od = -torch.log10(img)                              # [B, 3, H, W]
        od_bhw3 = od.permute(0, 2, 3, 1)                   # [B, H, W, 3]
        stains = torch.matmul(od_bhw3, self.stain_matrix_inv.T)  # [B, H, W, 3]
        stains = stains.clamp(min=0)
        stains = stains.permute(0, 3, 1, 2)                # [B, 3, H, W]
        return {
            "hematoxylin": stains[:, 0:1],
            "dab":         stains[:, 1:2],
            "residual":    stains[:, 2:3],
        }


class StainRecomposer(nn.Module):
    """
    Differentiable Beer-Lambert recomposition: density maps -> RGB.

    RGB = exp(-H_density * V_H - DAB_density * V_DAB)

    where V_H and V_DAB are the Ruifrok stain vectors (3-vectors per stain).
    The output is in [0, 1] (white background = 1.0).
    """

    def __init__(self):
        super().__init__()
        # V_H and V_DAB shapes: [1, 1, 1, 3] for broadcasting over [B, H, W, 3]
        self.register_buffer("v_h",   STAIN_MATRIX[0].view(1, 3, 1, 1))  # [1, 3, 1, 1]
        self.register_buffer("v_dab", STAIN_MATRIX[1].view(1, 3, 1, 1))  # [1, 3, 1, 1]

    def forward(
        self,
        h_density:   torch.Tensor,  # [B, 1, H, W]
        dab_density: torch.Tensor,  # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Returns:
            [B, 3, H, W] RGB in [0, 1]
        """
        od = h_density * self.v_h + dab_density * self.v_dab  # [B, 3, H, W]
        rgb = torch.exp(-od)
        return rgb.clamp(0.0, 1.0)


def analytical_recompose(
    h_density:   torch.Tensor,  # [B, 1, H, W]
    dab_density: torch.Tensor,  # [B, 1, H, W]
    device: torch.device = None,
) -> torch.Tensor:
    """
    Functional (stateless) wrapper around StainRecomposer.
    Moves stain vectors to the same device as the inputs automatically.

    Returns [B, 3, H, W] RGB in [0, 1].
    """
    if device is None:
        device = h_density.device
    v_h   = STAIN_MATRIX[0].view(1, 3, 1, 1).to(device)
    v_dab = STAIN_MATRIX[1].view(1, 3, 1, 1).to(device)
    od  = h_density * v_h + dab_density * v_dab
    rgb = torch.exp(-od)
    return rgb.clamp(0.0, 1.0)


def normalize_h_density(
    h_he:     torch.Tensor,  # [B, 1, H, W] H density from H&E deconvolution
    a_raw:    torch.Tensor,  # [B] or [B, 1] raw scale parameter (unconstrained)
    b_raw:    torch.Tensor,  # [B] or [B, 1] raw offset parameter (unconstrained)
) -> torch.Tensor:
    """
    Apply learned per-image linear normalization H_HE -> H_IHC.

    a = sigmoid(a_raw) * 2     -> (0, 2)   scale
    b = tanh(b_raw) * 0.5      -> (-0.5, 0.5)  offset

    H_IHC = (a * H_HE + b).clamp(0)
    """
    a = torch.sigmoid(a_raw) * 2.0    # (0, 2)
    b = torch.tanh(b_raw) * 0.5       # (-0.5, 0.5)

    # Broadcast: a, b are [B] -> [B, 1, 1, 1]
    a = a.view(-1, 1, 1, 1)
    b = b.view(-1, 1, 1, 1)

    return (a * h_he + b).clamp(min=0.0)


def fit_h_normalization_analytical(
    h_he_density: torch.Tensor,  # [H, W] or [B, 1, H, W]
    ihc_h_p5:  float = 0.08,
    ihc_h_p95: float = 1.20,
) -> tuple:
    """
    Fit (a, b) analytically by percentile matching (no learned params).
    ihc_h_p5 / ihc_h_p95 are calibrated from training set IHC H-channel stats.

    Returns (a, b) as Python floats.
    """
    flat = h_he_density.flatten().float()
    he_p5  = torch.quantile(flat, 0.05).item()
    he_p95 = torch.quantile(flat, 0.95).item()
    denom = max(he_p95 - he_p5, 1e-6)
    a = (ihc_h_p95 - ihc_h_p5) / denom
    b = ihc_h_p5 - a * he_p5
    return a, b
