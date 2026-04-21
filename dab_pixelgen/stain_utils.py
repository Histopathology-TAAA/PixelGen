"""
Stain deconvolution and Beer-Lambert recomposition utilities.

Based on the PSPStain paper approach (MLPA_LOSS) for accurate H-DAB separation.

Key improvements over naïve Ruifrok:
  1. Uses proper HED (Hematoxylin, Eosin, DAB) three-stain system with eosin
     as the third stain, NOT a generic residual. This correctly anchors the
     inverse matrix so DAB separation is clean.
  2. Uses PSPStain's log-natural approach (log_adjust = log(1e-6)) consistently
     in BOTH separation and recombination — eliminating the log10 vs exp(-)
     inconsistency that corrupts recomposed colours.
  3. `PSPStainDABExtractor` replicates the full PSPStain round-trip:
       HED deconv -> zero H,E -> recombine -> grayscale -> FOD transform
     This produces the true visual brown DAB signal, not a raw OD number.

Stain matrix convention (PSPStain / skimage `rgb2hed`):
  Row 0 — Hematoxylin  [0.65, 0.70, 0.29]
  Row 1 — Eosin        [0.07, 0.99, 0.11]
  Row 2 — DAB          [0.27, 0.57, 0.78]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Stain matrices (PSPStain / PSPStain-paper values) ─────────────────────────

# Forward matrix: maps stain densities -> RGB optical density contribution
# Shape [3, 3], rows = stains (H, E, DAB), cols = RGB channels
RGB_FROM_HED = torch.tensor([
    [0.65, 0.70, 0.29],   # Hematoxylin
    [0.07, 0.99, 0.11],   # Eosin
    [0.27, 0.57, 0.78],   # DAB
], dtype=torch.float32)

# Inverse: maps RGB optical densities -> stain densities
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)

# Natural log of 1e-6 — PSPStain's normalisation constant.
# All stain values are in this log-natural scale (consistent with combine_stains).
LOG_ADJUST = math.log(1e-6)   # ≈ -13.816

# Standard luminance weights for grayscale conversion (BT.601)
LUMINANCE = torch.tensor([0.2125, 0.7154, 0.0721], dtype=torch.float32)

# FOD hyper-parameters (PSPStain paper defaults)
FOD_ALPHA = 1.8
FOD_CALIBRATION = 10.0 ** (-(math.e) ** (1.0 / FOD_ALPHA))   # ≈ 0.0305
FOD_THRESH = 0.15    # zero out FOD values below this (suppress background noise)


# ── Core separation / recombination ──────────────────────────────────────────

class StainDeconvolution(nn.Module):
    """
    Differentiable H-E-D stain deconvolution (PSPStain approach).

    `separate_stains` formula (PSPStain):
        stains = matmul(log(rgb) / log(1e-6), hed_from_rgb)
        stains = clamp(stains, min=0)

    Returns per-stain densities in the log-natural scale consistent with
    `StainRecomposer.combine_stains`.  Higher value = more stain present.

    Channel order: index-0 = Hematoxylin, index-1 = Eosin, index-2 = DAB.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("hed_from_rgb", HED_FROM_RGB.clone())
        self.log_adjust = LOG_ADJUST      # scalar float, no device needed

    def forward(self, img: torch.Tensor) -> dict:
        """
        Args:
            img: [B, 3, H, W] RGB in [0, 1]
        Returns:
            dict:
                'hematoxylin': [B, 1, H, W]
                'eosin':       [B, 1, H, W]
                'dab':         [B, 1, H, W]
        """
        img = img.clamp(min=1e-6, max=1.0)
        img_bhw3 = img.permute(0, 2, 3, 1)                             # [B, H, W, 3]
        log_img  = torch.log(img_bhw3) / self.log_adjust                # divide by -13.816
        stains   = torch.matmul(log_img, self.hed_from_rgb)             # [B, H, W, 3]
        stains   = stains.clamp(min=0.0)
        stains   = stains.permute(0, 3, 1, 2)                          # [B, 3, H, W]
        return {
            "hematoxylin": stains[:, 0:1],   # H
            "eosin":       stains[:, 1:2],   # E
            "dab":         stains[:, 2:3],   # DAB  ← index 2 in HED
        }


class StainRecomposer(nn.Module):
    """
    Differentiable PSPStain `combine_stains` recomposition.

    `combine_stains` formula (PSPStain):
        log_rgb = -matmul(stains * (-log_adjust), rgb_from_hed)
        rgb     = exp(log_rgb)

    This is the EXACT inverse of `StainDeconvolution.forward`, so the
    round-trip  img -> deconvolve -> recompose -> img  is numerically accurate.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("rgb_from_hed", RGB_FROM_HED.clone())
        self.log_adjust = LOG_ADJUST

    def forward(
        self,
        h_density:   torch.Tensor,  # [B, 1, H, W]
        dab_density: torch.Tensor,  # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Recombine H + DAB densities to RGB.  Eosin is set to zero.

        Returns [B, 3, H, W] in [0, 1].
        """
        B, _, H, W = h_density.shape
        eosin = torch.zeros_like(h_density)
        # HED order: [H, E, DAB]
        stains_bhw3 = torch.cat([h_density, eosin, dab_density], dim=1)  # [B, 3, H, W]
        stains_bhw3 = stains_bhw3.permute(0, 2, 3, 1)                    # [B, H, W, 3]

        # PSPStain combine_stains:
        # log_rgb = -(stains * (-log_adjust)) @ rgb_from_hed
        #         =  stains * |log_adjust| @ rgb_from_hed
        log_rgb = -torch.matmul(stains_bhw3 * (-self.log_adjust), self.rgb_from_hed)
        rgb = torch.exp(log_rgb)
        return rgb.permute(0, 3, 1, 2).clamp(0.0, 1.0)   # [B, 3, H, W]


# ── Functional wrappers ───────────────────────────────────────────────────────

def analytical_recompose(
    h_density:   torch.Tensor,   # [B, 1, H, W]
    dab_density: torch.Tensor,   # [B, 1, H, W]
    device: torch.device = None,
) -> torch.Tensor:
    """
    Stateless Beer-Lambert recomposition (PSPStain consistent).

    Returns [B, 3, H, W] RGB in [0, 1].
    """
    if device is None:
        device = h_density.device

    rgb_from_hed = RGB_FROM_HED.to(device)   # [3, 3]
    B, _, H, W = h_density.shape
    eosin = torch.zeros_like(h_density)

    stains_bhw3 = torch.cat([h_density, eosin, dab_density], dim=1)  # [B, 3, H, W]
    stains_bhw3 = stains_bhw3.permute(0, 2, 3, 1)                    # [B, H, W, 3]

    log_rgb = -torch.matmul(stains_bhw3 * (-LOG_ADJUST), rgb_from_hed)
    rgb = torch.exp(log_rgb)
    return rgb.permute(0, 3, 1, 2).clamp(0.0, 1.0)


# ── PSPStain DAB extractor (round-trip + FOD) ─────────────────────────────────

class PSPStainDABExtractor(nn.Module):
    """
    Full PSPStain DAB isolation pipeline (from MLPA_LOSS.compute_OD).

    Pipeline:
      1. Deconvolve RGB to HED stain densities
      2. Zero out H and E channels → DAB-only stain
      3. Recombine to RGB using `combine_stains` → visually brown DAB image
      4. Convert to grayscale [0.2125, 0.7154, 0.0721]
      5. Apply Focal Optical Density (FOD) transform
      6. Threshold weak values at `thresh_fod`

    Output: FOD map [B, 1, H, W], higher = stronger DAB expression.

    Why the round-trip?  Using raw OD directly confuses DAB with H in
    overlapping spectral regions.  Recombining zeroed H/E back to RGB, then
    converting to greyscale, gives the *visual* brown signal a pathologist sees.

    Args:
        alpha:      FOD power exponent (PSPStain default 1.8)
        thresh_fod: Threshold to suppress weak/background FOD values
    """

    def __init__(self, alpha: float = FOD_ALPHA, thresh_fod: float = FOD_THRESH):
        super().__init__()
        self.alpha     = alpha
        self.thresh_fod = thresh_fod

        calibration = 10.0 ** (-(math.e) ** (1.0 / alpha))
        self.register_buffer("calibration",  torch.tensor(calibration,  dtype=torch.float32))
        self.register_buffer("rgb_from_hed", RGB_FROM_HED.clone())
        self.register_buffer("hed_from_rgb", HED_FROM_RGB.clone())
        self.register_buffer("luminance",    LUMINANCE.clone())
        self.log_adjust = LOG_ADJUST

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [B, 3, H, W] RGB in [0, 1]
        Returns:
            fod: [B, 1, H, W]  FOD map (thresholded, background -> 0)
        """
        img = img.clamp(1e-6, 1.0)
        img_bhw3 = img.permute(0, 2, 3, 1)                             # [B, H, W, 3]

        # 1. Separate stains (PSPStain `separate_stains`)
        stains = torch.matmul(
            torch.log(img_bhw3) / self.log_adjust,
            self.hed_from_rgb,
        ).clamp(min=0.0)                                                # [B, H, W, 3]

        # 2. Zero H and E, keep DAB only
        null = torch.zeros_like(stains[..., 0:1])
        dab_only = torch.cat([null, null, stains[..., 2:3]], dim=-1)   # [B, H, W, 3]

        # 3. Recombine to RGB (`combine_stains`)
        log_rgb_dab = -torch.matmul(dab_only * (-self.log_adjust), self.rgb_from_hed)
        dab_rgb = torch.exp(log_rgb_dab).clamp(0.0, 1.0)              # [B, H, W, 3]

        # 4. Convert to grayscale (BT.601 luminance)
        grey = torch.matmul(dab_rgb, self.luminance.view(3, 1))        # [B, H, W, 1]
        grey = grey.clamp(0.0, 1.0)

        # 5. Focal Optical Density (PSPStain formula)
        fod = torch.log10(1.0 / (grey + self.calibration))
        fod = F.relu(fod) ** self.alpha                                 # [B, H, W, 1]

        # 6. Threshold weak values
        fod = torch.where(fod < self.thresh_fod,
                          torch.zeros_like(fod),
                          fod)

        return fod.permute(0, 3, 1, 2)                                  # [B, 1, H, W]

    def extract_raw_dab(self, img: torch.Tensor) -> torch.Tensor:
        """
        Return the raw PSPStain DAB stain density (no FOD, no threshold).
        Useful when the raw density is needed for Beer-Lambert recomposition.

        Returns [B, 1, H, W], values in PSPStain log-natural scale.
        """
        img = img.clamp(1e-6, 1.0)
        img_bhw3 = img.permute(0, 2, 3, 1)
        stains = torch.matmul(
            torch.log(img_bhw3) / self.log_adjust,
            self.hed_from_rgb,
        ).clamp(min=0.0)                                                # [B, H, W, 3]
        return stains[..., 2:3].permute(0, 3, 1, 2)                    # [B, 1, H, W]

    def extract_raw_h(self, img: torch.Tensor) -> torch.Tensor:
        """Return the raw PSPStain H (hematoxylin) stain density [B, 1, H, W]."""
        img = img.clamp(1e-6, 1.0)
        img_bhw3 = img.permute(0, 2, 3, 1)
        stains = torch.matmul(
            torch.log(img_bhw3) / self.log_adjust,
            self.hed_from_rgb,
        ).clamp(min=0.0)
        return stains[..., 0:1].permute(0, 3, 1, 2)                    # [B, 1, H, W]


# ── H normalization ───────────────────────────────────────────────────────────

def normalize_h_density(
    h_he:  torch.Tensor,   # [B, 1, H, W] H density from H&E deconvolution
    a_raw: torch.Tensor,   # [B] raw scale parameter (unconstrained)
    b_raw: torch.Tensor,   # [B] raw offset parameter (unconstrained)
) -> torch.Tensor:
    """
    Apply per-image linear normalization H_HE -> H_IHC.

    a = sigmoid(a_raw) * 2     in (0, 2)
    b = tanh(b_raw) * 0.5      in (-0.5, 0.5)

    H_IHC = clamp(a * H_HE + b, min=0)
    """
    a = torch.sigmoid(a_raw) * 2.0    # (0, 2)
    b = torch.tanh(b_raw)   * 0.5     # (-0.5, 0.5)
    a = a.view(-1, 1, 1, 1)
    b = b.view(-1, 1, 1, 1)
    return (a * h_he + b).clamp(min=0.0)


def fit_h_normalization_analytical(
    h_he_density: torch.Tensor,   # [H, W] or [B, 1, H, W]
    ihc_h_p5:  float = 0.08,
    ihc_h_p95: float = 1.20,
) -> tuple:
    """
    Fit (a, b) analytically by percentile matching (Option A2 — no neural params).
    ihc_h_p5 / ihc_h_p95 should be calibrated from your training set.

    Returns (a, b) as Python floats.
    """
    flat   = h_he_density.flatten().float()
    he_p5  = torch.quantile(flat, 0.05).item()
    he_p95 = torch.quantile(flat, 0.95).item()
    denom  = max(he_p95 - he_p5, 1e-6)
    a = (ihc_h_p95 - ihc_h_p5) / denom
    b = ihc_h_p5 - a * he_p5
    return a, b
