"""
PixelGen-style perceptual losses: LPIPS + DINOv2 + noise gating.
Adapted from PixelGen (arXiv:2602.02493).

Applied on reconstructed x̂₀:
  x̂₀ = (x_t - √(1-ᾱ_t)·ε̂ - β̄_t·r̂) / √ᾱ_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import lpips


class PixelGenPerceptualLoss(nn.Module):
    """
    LPIPS (local textures) + DINOv2 (global semantics) + noise gating.
    Perceptual losses are disabled at high-noise timesteps.
    """

    def __init__(
        self,
        use_dino: bool = True,
        lpips_weight: float = 0.1,
        dino_weight: float = 0.01,
        noise_gate_threshold: float = 0.3,
        device: str = "cuda",
    ):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight
        self.noise_gate_threshold = noise_gate_threshold
        self.use_dino = use_dino

        # LPIPS (frozen VGG)
        self.lpips_fn = lpips.LPIPS(net="vgg").eval().to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        print(f"✓ LPIPS (VGG) loaded — {sum(p.numel() for p in self.lpips_fn.parameters()) / 1e6:.1f}M (frozen)")

        # DINOv2 (optional, frozen)
        self.dino = None
        if use_dino:
            try:
                self.dino = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", verbose=False
                ).eval().to(device)
                for p in self.dino.parameters():
                    p.requires_grad = False
                self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                print(f"✓ DINOv2-S loaded — {sum(p.numel() for p in self.dino.parameters()) / 1e6:.1f}M (frozen)")
            except Exception as e:
                print(f"⚠️ DINOv2 load failed ({e}), using LPIPS only")
                self.dino = None
                self.use_dino = False

    def _prep_dino_input(self, x):
        """[-1,1] → [0,1] → ImageNet-normalized, resize to 224."""
        x = (x + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        return x

    def forward(self, pred, target, timesteps, num_timesteps):
        """
        Compute perceptual losses with noise gating.

        Args:
            pred:   predicted x̂₀ [B, 3, H, W] in [-1, 1]
            target: ground truth x₀ [B, 3, H, W] in [-1, 1]
            timesteps: [B] integer timesteps
            num_timesteps: total T
        Returns:
            (total_loss, loss_dict)
        """
        batch_size = pred.shape[0]
        device = pred.device

        # Noise gating: skip at high-noise timesteps (t near 0 in Flow Matching)
        # Flow Matching t is in [0, 1], where t=0 is noise, t=1 is clean.
        # So noise_level = 1.0 - t.
        noise_level = 1.0 - timesteps.float()
        gate_limit = 1.0 - self.noise_gate_threshold
        # Use torch.where to ensure we stay in Tensor territory and avoid float/bool conversion issues
        gate_mask = torch.where(
            noise_level < gate_limit,
            torch.ones_like(noise_level),
            torch.zeros_like(noise_level)
        )

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        if gate_mask.sum() == 0:
            loss_dict["lpips"] = torch.tensor(0.0, device=device)
            loss_dict["dino"] = torch.tensor(0.0, device=device)
            return total_loss, loss_dict

        # LPIPS
        lpips_vals = self.lpips_fn(pred, target).view(batch_size)
        lpips_loss = (lpips_vals * gate_mask).sum() / gate_mask.sum().clamp(min=1)
        total_loss = total_loss + self.lpips_weight * lpips_loss
        loss_dict["lpips"] = lpips_loss.detach()

        # DINOv2
        if self.use_dino and self.dino is not None:
            pred_dino = self._prep_dino_input(pred)
            target_dino = self._prep_dino_input(target)
            with torch.no_grad():
                target_feats = self.dino(target_dino)
            pred_feats = self.dino(pred_dino)  # grads flow through pred only
            cos_sim = F.cosine_similarity(pred_feats, target_feats, dim=-1)
            dino_loss = ((1 - cos_sim) * gate_mask).sum() / gate_mask.sum().clamp(min=1)
            total_loss = total_loss + self.dino_weight * dino_loss
            loss_dict["dino"] = dino_loss.detach()
        else:
            loss_dict["dino"] = torch.tensor(0.0, device=device)

        return total_loss, loss_dict
