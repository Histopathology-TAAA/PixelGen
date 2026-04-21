"""
Loss functions for GAN DABPixelGen.

Generator loss (combined):
  L_G = w_adv   * L_adv(G)            adversarial (fool discriminator)
      + w_fm    * L_fm(feats)          feature matching (stable early training)
      + w_l1    * L_l1(ihc, ihc_gt)   pixel-level L1 on IHC RGB
      + w_dab   * L_dab(dab, dab_fod)  DAB expression in FOD space
      + w_lpips * L_lpips(ihc, ihc_gt) perceptual similarity

Discriminator loss:
  L_D = L_hinge(D_real) + L_hinge(D_fake)   [hinge mode]
  L_D = L_lsgan(D_real) + L_lsgan(D_fake)   [lsgan mode]

Feature matching loss (pix2pixHD):
  For each scale s and each intermediate layer l:
    L_fm += (1 / N_layers) * L1(D_s_l(real), D_s_l(fake))

DAB expression loss (carried over from dab_pixelgen):
  Supervises DAB output in FOD (Focal Optical Density) space,
  emphasising strongly positive nuclei/cells and suppressing background.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ── Adversarial loss ──────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Adversarial loss for D and G.

    Supports two objectives:
      "hinge" — hinge loss (pix2pixHD / BigGAN default, numerically stable)
      "lsgan" — least-squares GAN (Mao et al., 2017)

    All logit maps returned by MultiScalePatchGAN are aggregated by summing
    over scales before computing the scalar loss.
    """

    def __init__(self, mode: str = "hinge"):
        super().__init__()
        if mode not in ("hinge", "lsgan"):
            raise ValueError(f"Unknown GAN mode '{mode}'. Choose: hinge, lsgan")
        self.mode = mode

    def _mean_logit(self, per_scale_feats: List[List[torch.Tensor]]) -> torch.Tensor:
        """Average the logit maps across all scales and return a scalar."""
        total = torch.tensor(0.0, device=per_scale_feats[0][-1].device)
        for feats in per_scale_feats:
            total = total + feats[-1].mean()   # feats[-1] is the logit map
        return total / len(per_scale_feats)

    def d_loss(
        self,
        real_feats: List[List[torch.Tensor]],
        fake_feats: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Discriminator loss (maximize real score, minimize fake score).

        Hinge:  L = E[relu(1 - D(real))] + E[relu(1 + D(fake))]
        LSGAN:  L = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[D(fake)^2]
        """
        d_real = self._mean_logit(real_feats)
        d_fake = self._mean_logit(fake_feats)

        if self.mode == "hinge":
            return F.relu(1.0 - d_real) + F.relu(1.0 + d_fake)
        else:   # lsgan
            return 0.5 * (d_real - 1.0) ** 2 + 0.5 * d_fake ** 2

    def g_loss(
        self,
        fake_feats: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Generator adversarial loss (fool discriminator).

        Hinge:  L = -E[D(fake)]
        LSGAN:  L = 0.5 * E[(D(fake) - 1)^2]
        """
        d_fake = self._mean_logit(fake_feats)

        if self.mode == "hinge":
            return -d_fake
        else:   # lsgan
            return 0.5 * (d_fake - 1.0) ** 2


# ── Feature matching loss ─────────────────────────────────────────────────────

class FeatureMatchingLoss(nn.Module):
    """
    L1 feature matching between real and fake discriminator intermediate layers.

    This provides dense per-pixel feedback to the generator at every scale and
    every layer depth, making early training much more stable than pure adversarial.

    From pix2pixHD (Wang et al.):
        L_FM = sum_s sum_l (1/N_l) * L1(D_s_l(real), D_s_l(fake))

    We normalise by the total number of feature tensors (scales × layers − 1)
    so the weight is scale/depth-invariant.
    """

    def forward(
        self,
        real_feats: List[List[torch.Tensor]],
        fake_feats: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            real_feats: per_scale[per_layer[tensor]]  from D(H&E, IHC_real)
            fake_feats: per_scale[per_layer[tensor]]  from D(H&E, IHC_fake)
            NOTE: last layer in each scale (index -1) is the logit map; we skip it
                  to avoid double-counting with the adversarial loss.
        Returns:
            scalar L1 feature matching loss
        """
        loss = torch.tensor(0.0, device=real_feats[0][0].device)
        n_terms = 0

        for r_scale, f_scale in zip(real_feats, fake_feats):
            # Exclude the final logit map (index -1)
            for r_feat, f_feat in zip(r_scale[:-1], f_scale[:-1]):
                loss = loss + F.l1_loss(f_feat, r_feat.detach())
                n_terms += 1

        return loss / max(n_terms, 1)


# ── DAB expression loss ───────────────────────────────────────────────────────

class FocalODLoss(nn.Module):
    """
    PSPStain Focal Optical Density loss with histogram and patch-level supervision.

    Fixes the "always white DAB" failure mode that plagues pixel-only MSE:

      Problem 1 — Wrong OD→intensity mapping:
        The model predicts OD (optical density), where HIGH OD = MORE staining.
        Via Beer-Lambert: intensity I = 10^(-OD).
        Background (OD=0) → I=1 (white).  Heavy staining (OD=3) → I=0.001 (dark).
        The naive `_raw_to_fod(raw)` treated OD as intensity directly, giving
        FOD≈0 for strong staining and FOD≈high for empty prediction — sign error.

      Problem 2 — Dead gradients through ReLU + hard threshold:
        When the model initialises dab_head ≈ 0, FOD(0) lands in the threshold
        dead-zone.  Gradients are exactly zero and the head never wakes up.
        We use softplus instead of relu in the training FOD to preserve gradients
        everywhere.  The hard threshold is only applied at inference / visualisation.

      Problem 3 — Background dominates pixel loss:
        At 256px patches, ~80% of pixels are background (FOD=0).  MSE collapses
        to predicting zero everywhere with low average loss.
        Histogram + patch-level losses operate on image-level statistics and
        are insensitive to pixel-perfect background alignment.

    Args:
        n_bins:        number of bins for differentiable soft histogram (default 64)
        hist_sigma:    soft-bin bandwidth in FOD units (default 0.05)
        patch_sizes:   pooling kernel sizes for patch statistics (default (16, 32, 64))
        pixel_weight:  L1 pixel loss weight
        hist_weight:   histogram EMD/L1 loss weight
        patch_weight:  patch mean-MSE weight
    """

    # PSPStain FOD constants
    _FOD_ALPHA:       float = 1.8
    _FOD_CALIBRATION: float = 10.0 ** (-(math.e) ** (1.0 / 1.8))  # ≈ 0.0181
    _FOD_MAX:         float = 5.0   # upper end of histogram axis

    def __init__(
        self,
        n_bins:       int              = 64,
        hist_sigma:   float            = 0.05,
        patch_sizes:  tuple            = (16, 32, 64),
        pixel_weight: float            = 1.0,
        hist_weight:  float            = 2.0,
        patch_weight: float            = 1.0,
    ):
        super().__init__()
        self.n_bins       = n_bins
        self.hist_sigma   = hist_sigma
        self.patch_sizes  = patch_sizes
        self.pixel_weight = pixel_weight
        self.hist_weight  = hist_weight
        self.patch_weight = patch_weight

        # Bin centres for soft histogram — range [0, FOD_MAX]
        self.register_buffer(
            "bin_centers",
            torch.linspace(0.0, self._FOD_MAX, n_bins),
        )

    def _density_to_fod(self, density: torch.Tensor) -> torch.Tensor:
        """
        Convert raw DAB OD density → FOD for training.

        Beer-Lambert: I = 10^(-OD)   (OD=0 → I=1 white; OD=3 → I=0.001 dark)
        FOD = relu(log10(1/(I + cal)))^alpha

        Uses softplus instead of relu so that gradients are nonzero even when
        log10(1/(I+cal)) ≤ 0 (i.e. for very low-density / background pixels).
        This prevents the DAB head from getting permanently stuck at zero.
        """
        d         = density.clamp(0.0, 5.0)
        intensity = 10.0 ** (-d)                             # Beer-Lambert
        log_fod   = torch.log10(1.0 / (intensity + self._FOD_CALIBRATION))
        # Steep softplus approximates relu but has nonzero gradient below 0
        # softplus(5x)/5 ≈ relu(x) for x > 0.3, smooth and nonzero elsewhere
        fod = (F.softplus(log_fod * 5.0) / 5.0) ** self._FOD_ALPHA
        return fod

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable per-image soft histogram over all spatial positions.

        Returns normalised probability distribution over `n_bins` FOD bins.
        Gaussian kernel assignment:  h[b] ∝ Σ_i exp(-0.5*((x_i - c_b)/σ)²)
        """
        B       = x.shape[0]
        x_flat  = x.reshape(B, -1, 1)                        # [B, N, 1]
        centers = self.bin_centers.to(x.device).view(1, 1, -1)  # [1, 1, n_bins]
        weights = torch.exp(-0.5 * ((x_flat - centers) / self.hist_sigma) ** 2)
        hist    = weights.mean(dim=1)                         # [B, n_bins]
        return hist / (hist.sum(dim=1, keepdim=True) + 1e-8)

    def _histogram_loss(
        self, pred_fod: torch.Tensor, gt_fod: torch.Tensor
    ) -> torch.Tensor:
        """L1 between per-image normalised FOD histograms."""
        return F.l1_loss(self._soft_histogram(pred_fod), self._soft_histogram(gt_fod))

    def _patch_loss(
        self, pred_fod: torch.Tensor, gt_fod: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-scale patch mean-MSE.

        Average-pools both pred and GT FOD to coarser grids, then computes MSE.
        Effective at matching the spatial distribution of staining even when
        pixel-level alignment is poor.
        """
        loss = torch.tensor(0.0, device=pred_fod.device)
        for ps in self.patch_sizes:
            pred_p = F.avg_pool2d(pred_fod, kernel_size=ps, stride=ps)
            gt_p   = F.avg_pool2d(gt_fod,   kernel_size=ps, stride=ps)
            loss   = loss + F.mse_loss(pred_p, gt_p)
        return loss / len(self.patch_sizes)

    def forward(
        self,
        dab_pred:   torch.Tensor,   # [B, 1, H, W]  predicted raw OD density (≥ 0)
        dab_gt_fod: torch.Tensor,   # [B, 1, H, W]  GT FOD from PSPStainDABExtractor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total loss (scalar), log dict with per-component values
        """
        pred_fod = self._density_to_fod(dab_pred)   # [B, 1, H, W]

        pixel_loss = F.l1_loss(pred_fod, dab_gt_fod)
        hist_loss  = self._histogram_loss(pred_fod, dab_gt_fod)
        patch_loss = self._patch_loss(pred_fod, dab_gt_fod)

        total = (
              self.pixel_weight * pixel_loss
            + self.hist_weight  * hist_loss
            + self.patch_weight * patch_loss
        )
        return total, {
            "dab/pixel": pixel_loss.detach(),
            "dab/hist":  hist_loss.detach(),
            "dab/patch": patch_loss.detach(),
        }


# Keep old name as alias for backward compatibility with any existing imports
DABExpressionLoss = FocalODLoss


# ── Combined generator loss ───────────────────────────────────────────────────

class GANDABLoss(nn.Module):
    """
    Combined generator loss for GAN DABPixelGen.

    L_G = adv_weight        * L_adv        (adversarial)
        + feat_match_weight * L_fm         (feature matching)
        + l1_weight         * L_l1         (pixel L1 on IHC RGB)
        + dab_weight        * L_dab        (DAB FOD expression)
        + lpips_weight      * L_lpips      (perceptual, if installed)
    """

    def __init__(
        self,
        adv_weight:        float = 1.0,
        feat_match_weight: float = 10.0,
        l1_weight:         float = 100.0,
        dab_weight:        float = 10.0,
        lpips_weight:      float = 10.0,
        gan_mode:          str   = "hinge",
        device:            str   = "cuda",
    ):
        super().__init__()
        self.adv_weight        = adv_weight
        self.feat_match_weight = feat_match_weight
        self.l1_weight         = l1_weight
        self.dab_weight        = dab_weight
        self.lpips_weight      = lpips_weight

        self.gan_loss    = GANLoss(mode=gan_mode)
        self.fm_loss     = FeatureMatchingLoss()
        self.dab_loss_fn = FocalODLoss()
        self.lpips_fn: Optional[nn.Module] = None

        if lpips_weight > 0.0:
            try:
                import lpips as _lpips_lib
                self.lpips_fn = _lpips_lib.LPIPS(net="vgg").eval().to(device)
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
                print(f"GANDABLoss: LPIPS (VGG) loaded, weight={lpips_weight}")
            except ImportError:
                print("GANDABLoss: 'lpips' not installed — LPIPS term disabled.")

    def forward(
        self,
        # ── Generator outputs ─────────────────────────────────────────────────
        dab_pred:   torch.Tensor,              # [B, 1, H, W]
        ihc_pred:   torch.Tensor,              # [B, 3, H, W] in [-1, 1]
        # ── Discriminator features (D evaluated on fake) ──────────────────────
        fake_feats: List[List[torch.Tensor]],  # returned by D.forward(he, ihc_pred)
        # ── Discriminator features (D evaluated on real) ──────────────────────
        real_feats: List[List[torch.Tensor]],  # returned by D.forward(he, ihc_real)
        # ── Ground truth ──────────────────────────────────────────────────────
        dab_gt_fod: torch.Tensor,              # [B, 1, H, W]  FOD DAB from IHC
        ihc_gt_01:  torch.Tensor,              # [B, 3, H, W]  IHC in [0, 1]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined generator loss.

        Returns:
            total_loss: scalar tensor (backpropagated)
            log_dict:   dict of detached component losses for logging
        """
        device = dab_pred.device

        # ── Convert ihc_pred [-1,1] to [0,1] for L1/LPIPS ────────────────────
        ihc_pred_01 = (ihc_pred.clamp(-1, 1) + 1) / 2   # [0, 1]

        # ── Adversarial ───────────────────────────────────────────────────────
        l_adv = self.gan_loss.g_loss(fake_feats)

        # ── Feature matching ──────────────────────────────────────────────────
        l_fm = self.fm_loss(real_feats, fake_feats)

        # ── Pixel L1 on IHC RGB ───────────────────────────────────────────────
        l_l1 = F.l1_loss(ihc_pred_01, ihc_gt_01)

        # ── DAB expression (FOD histogram + patch + pixel) ───────────────────
        l_dab, dab_log = self.dab_loss_fn(dab_pred, dab_gt_fod)

        # ── LPIPS perceptual ──────────────────────────────────────────────────
        l_lpips = torch.tensor(0.0, device=device)
        if self.lpips_fn is not None and self.lpips_weight > 0.0:
            # LPIPS expects [-1, 1]
            l_lpips = self.lpips_fn(
                ihc_pred.clamp(-1, 1),
                ihc_gt_01 * 2 - 1,
            ).mean()

        # ── Weighted sum ──────────────────────────────────────────────────────
        total = (
              self.adv_weight        * l_adv
            + self.feat_match_weight * l_fm
            + self.l1_weight         * l_l1
            + self.dab_weight        * l_dab
            + self.lpips_weight      * l_lpips
        )

        log = {
            "g/adv":         l_adv.detach(),
            "g/feat_match":  l_fm.detach(),
            "g/l1":          l_l1.detach(),
            "g/dab":         l_dab.detach(),
            "g/dab_pixel":   dab_log["dab/pixel"],
            "g/dab_hist":    dab_log["dab/hist"],
            "g/dab_patch":   dab_log["dab/patch"],
            "g/lpips":       l_lpips.detach(),
            "g/total":       total.detach(),
        }
        return total, log


# ── Discriminator loss (standalone helper) ────────────────────────────────────

def discriminator_loss(
    gan_loss:   GANLoss,
    real_feats: List[List[torch.Tensor]],
    fake_feats: List[List[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute D hinge/LSGAN loss.

    Args:
        gan_loss:   GANLoss instance
        real_feats: D features on real IHC
        fake_feats: D features on fake IHC (detached from G)

    Returns:
        (scalar loss, log dict)
    """
    loss = gan_loss.d_loss(real_feats, fake_feats)
    return loss, {"d/loss": loss.detach()}
