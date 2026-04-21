"""
Loss functions for DABPixelGen training.

Three complementary losses work together:

1. FlowMatchingLoss (velocity MSE)
   Trains the backbone to predict clean DAB (x_1) via flow matching.
   Always active across all timesteps.

2. DABPredictionLoss (MSE on DAB density)
   Direct supervision on the predicted DAB density vs GT DAB.
   Noise-gated: only applied when t >= noise_gate_threshold.

3. RecompositionLoss (MSE + optional LPIPS on RGB)
   End-to-end loss: applies H-normalization, recomposes RGB via Beer-Lambert,
   then compares to GT IHC RGB.  Backpropagates jointly into DAB head and
   H-norm head.
   Noise-gated: only applied when t >= noise_gate_threshold.
"""
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.stain_utils import analytical_recompose, normalize_h_density


class FlowMatchingLoss(nn.Module):
    """
    Rectified Flow velocity MSE.

    The backbone predicts the clean image x_1 directly.
    We convert to velocity: v_pred = (x_1_pred - x_t) / (1 - t)
    and supervise against v_target = dab_gt - x_0.
    """

    def forward(
        self,
        x1_pred:  torch.Tensor,    # [B, 1, H, W] predicted clean DAB
        x_t:      torch.Tensor,    # [B, 1, H, W] noisy DAB at t
        v_target: torch.Tensor,    # [B, 1, H, W] ground truth velocity
        t:        torch.Tensor,    # [B] timesteps in [0, 1]
    ) -> torch.Tensor:
        denom = (1.0 - t.view(-1, 1, 1, 1)).clamp_min(1e-3)
        v_pred = (x1_pred - x_t) / denom
        return F.mse_loss(v_pred, v_target)


class DABPredictionLoss(nn.Module):
    """
    DAB expression-level supervision using PSPStain FOD targets.

    Uses `dab_gt_fod` (FOD-transformed ground truth) rather than raw density:
      - FOD emphasises strongly positive nuclei/cells (the clinically relevant signal)
      - FOD suppresses weak/background staining noise
      - MSE on FOD space is perceptually aligned with what pathologists score

    The model still predicts raw stain density (for use in recomposition), so
    we apply the FOD transform to the prediction here before computing the loss.

    Noise-gated: only active for samples where t >= threshold (clean predictions).
    """

    def __init__(self, noise_gate_threshold: float = 0.7):
        super().__init__()
        self.threshold  = noise_gate_threshold
        self.fod_alpha  = 1.8
        self.fod_calibration = 10.0 ** (-(math.e) ** (1.0 / self.fod_alpha))
        self.fod_thresh = 0.15

    def _raw_to_fod(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Apply PSPStain FOD transform to raw stain density.

        Raw stain -> grey intensity proxy -> FOD.
        The raw stain value in PSPStain scale corresponds to DAB's
        grayscale contribution; we approximate it here directly.
        """
        # clamp to valid range; larger raw value = more DAB = darker brown
        intensity = torch.clamp(raw, 0.0, 5.0)
        # Treat raw density as an approximate intensity proxy
        # (proper round-trip happens in PSPStainDABExtractor in dataset)
        fod = torch.log10(1.0 / (intensity + self.fod_calibration))
        fod = F.relu(fod) ** self.fod_alpha
        fod = torch.where(fod < self.fod_thresh,
                          torch.zeros_like(fod), fod)
        return fod

    def forward(
        self,
        dab_pred:   torch.Tensor,    # [B, 1, H, W] model output (raw PSPStain density)
        dab_gt_fod: torch.Tensor,    # [B, 1, H, W] GT FOD from PSPStainDABExtractor
        t:          torch.Tensor,    # [B]
    ) -> torch.Tensor:
        gate = (t >= self.threshold).float()
        if gate.sum() == 0:
            return torch.tensor(0.0, device=dab_pred.device)

        pred_fod = self._raw_to_fod(dab_pred)
        diff = F.mse_loss(pred_fod, dab_gt_fod, reduction="none")  # [B, 1, H, W]
        diff = diff.mean(dim=(1, 2, 3))                              # [B]
        return (diff * gate).sum() / gate.sum()


class RecompositionLoss(nn.Module):
    """
    End-to-end recomposition loss:
        1. Normalize H_HE -> H_IHC using (a, b) from the H-norm head
        2. Recompose RGB via Beer-Lambert
        3. MSE + optional LPIPS against GT IHC RGB

    Noise-gated: only for t >= threshold, where predictions are clean enough
    for stain deconvolution / recomposition to be meaningful.
    """

    def __init__(
        self,
        noise_gate_threshold: float = 0.7,
        lpips_weight: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.threshold = noise_gate_threshold
        self.lpips_weight = lpips_weight
        self.lpips_fn = None

        if lpips_weight > 0:
            try:
                import lpips as lpips_lib
                self.lpips_fn = lpips_lib.LPIPS(net="vgg").eval().to(device)
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
                print(f"RecompositionLoss: LPIPS (VGG) loaded, weight={lpips_weight}")
            except ImportError:
                print("RecompositionLoss: 'lpips' not installed; LPIPS disabled.")

    def forward(
        self,
        dab_pred:     torch.Tensor,            # [B, 1, H, W] predicted clean DAB density
        h_norm_params:torch.Tensor,            # [B, 2]        (a_raw, b_raw)
        h_he_density: torch.Tensor,            # [B, 1, H, W]  H density from H&E deconv
        ihc_rgb_gt:   torch.Tensor,            # [B, 3, H, W]  GT IHC in [0, 1]
        t:            torch.Tensor,            # [B]
        ihc_pred:     Optional[torch.Tensor] = None,  # [B, 3, H, W] in [-1,1] from CombinationNet
    ) -> Tuple[torch.Tensor, Dict]:
        gate = (t >= self.threshold).float()
        if gate.sum() == 0:
            zero = torch.tensor(0.0, device=dab_pred.device)
            return zero, {"recomp_mse": zero, "recomp_lpips": zero}

        if ihc_pred is not None:
            # CombinationNet output: [-1, 1] → [0, 1]
            recomposed = (ihc_pred.clamp(-1, 1) + 1) / 2
        else:
            a_raw = h_norm_params[:, 0]
            b_raw = h_norm_params[:, 1]
            h_ihc = normalize_h_density(h_he_density, a_raw, b_raw)  # [B, 1, H, W]
            recomposed = analytical_recompose(h_ihc, dab_pred, device=dab_pred.device)

        mse = F.mse_loss(recomposed, ihc_rgb_gt, reduction="none")  # [B, 3, H, W]
        mse = mse.mean(dim=(1, 2, 3))                                 # [B]
        mse_loss = (mse * gate).sum() / gate.sum()

        lpips_loss = torch.tensor(0.0, device=dab_pred.device)
        if self.lpips_fn is not None and self.lpips_weight > 0:
            # LPIPS expects [-1, 1]; recomposed is [0, 1]
            recomp_11 = recomposed * 2 - 1
            gt_11     = ihc_rgb_gt  * 2 - 1
            lpips_vals = self.lpips_fn(recomp_11, gt_11).view(-1)  # [B]
            lpips_loss = (lpips_vals * gate).sum() / gate.sum()

        total = mse_loss + self.lpips_weight * lpips_loss
        return total, {
            "recomp_mse": mse_loss.detach(),
            "recomp_lpips": lpips_loss.detach(),
        }


class DABPixelGenLoss(nn.Module):
    """
    Combined loss: flow matching + DAB prediction + recomposition.

    loss = w_fm * L_fm
         + w_dab * L_dab
         + w_recomp * L_recomp
    """

    def __init__(
        self,
        fm_weight:         float = 1.0,
        dab_weight:        float = 1.0,
        recomp_weight:     float = 1.0,
        noise_gate_threshold: float = 0.7,
        lpips_weight:      float = 0.0,
        device:            str = "cuda",
    ):
        super().__init__()
        self.fm_weight     = fm_weight
        self.dab_weight    = dab_weight
        self.recomp_weight = recomp_weight

        self.fm_loss     = FlowMatchingLoss()
        self.dab_loss_fn = DABPredictionLoss(noise_gate_threshold)
        self.recomp_loss_fn = RecompositionLoss(
            noise_gate_threshold, lpips_weight, device
        )

    def forward(
        self,
        x1_pred:       torch.Tensor,            # [B, 1, H, W] model output (clean DAB density)
        h_norm_params: torch.Tensor,            # [B, 2]
        x_t:           torch.Tensor,            # [B, 1, H, W]
        v_target:      torch.Tensor,            # [B, 1, H, W]
        t:             torch.Tensor,            # [B]
        dab_gt:        torch.Tensor,            # [B, 1, H, W] raw DAB density (for recomposition)
        dab_gt_fod:    torch.Tensor,            # [B, 1, H, W] FOD DAB (for expression loss)
        h_he_density:  torch.Tensor,            # [B, 1, H, W]
        ihc_rgb_gt:    torch.Tensor,            # [B, 3, H, W] in [0, 1]
        ihc_pred:      Optional[torch.Tensor] = None,  # [B, 3, H, W] in [-1,1] from CombinationNet
    ) -> Tuple[torch.Tensor, Dict]:
        l_fm   = self.fm_loss(x1_pred, x_t, v_target, t)
        l_dab  = self.dab_loss_fn(x1_pred, dab_gt_fod, t)   # supervise on FOD space
        l_recomp, recomp_dict = self.recomp_loss_fn(
            x1_pred, h_norm_params, h_he_density, ihc_rgb_gt, t,
            ihc_pred=ihc_pred,
        )

        total = (
            self.fm_weight     * l_fm
            + self.dab_weight    * l_dab
            + self.recomp_weight * l_recomp
        )

        log_dict = {
            "loss_fm":          l_fm.detach(),
            "loss_dab":         l_dab.detach(),
            "loss_recomp":      l_recomp.detach(),
            "loss_recomp_mse":  recomp_dict["recomp_mse"],
            "loss_recomp_lpips":recomp_dict["recomp_lpips"],
            "loss_total":       total.detach(),
        }
        return total, log_dict
