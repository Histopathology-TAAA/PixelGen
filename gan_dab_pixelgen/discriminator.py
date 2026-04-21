"""
Multi-scale PatchGAN Discriminator for GAN DABPixelGen.

Architecture based on pix2pixHD (Wang et al., 2018) with:
  - Spectral normalization on all conv layers (Miyato et al., 2018)
    for Lipschitz constraint without hyperparameter tuning
  - InstanceNorm (not BatchNorm) for compatibility with small batch sizes
  - Feature extraction at each layer for feature matching loss
  - Two-scale design: full resolution + 2x average-pooled input

Input:  cat(H&E [3ch], IHC [3ch]) = 6 channels  (condition + real/fake target)
Output: list of per-scale feature-list, where each list's last tensor is the
        logit map (B, 1, H', W')

Receptive field at n_layers=4:
  Layer 0: k4s2 → 8x8 RF
  Layer 1: k4s2 → 16x16
  Layer 2: k4s2 → 34x34
  Layer 3: k4s2 → 70x70  ← classic 70x70 PatchGAN
  Layer 4: k4s1 → 94x94 (penultimate)
  Layer 5: k4s1 → 118x118 (output)
(Each layer roughly 2x previous RF + kernel overlap)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NLayerDiscriminator(nn.Module):
    """
    N-layer PatchGAN discriminator with spectral normalization.

    Returns a list of intermediate feature maps at each layer, with the
    last entry being the final logit map [B, 1, H', W'].
    This supports feature matching loss and per-scale adversarial losses.

    Args:
        input_nc:  number of input channels (default 6 = HE 3ch + IHC 3ch)
        ndf:       base channel width (default 64)
        n_layers:  number of strided conv layers (default 4 → 70x70+ RF)
    """

    def __init__(
        self,
        input_nc: int = 6,
        ndf:      int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        SN   = nn.utils.spectral_norm
        kw   = 4
        padw = 1

        layers = nn.ModuleList()

        # Layer 0: no InstanceNorm on first layer (pix2pixHD style)
        layers.append(nn.Sequential(
            SN(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Strided intermediate layers (n_layers - 1 of them)
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf      = min(nf * 2, ndf * 8)
            layers.append(nn.Sequential(
                SN(nn.Conv2d(nf_prev, nf, kw, stride=2, padding=padw)),
                nn.InstanceNorm2d(nf, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # Penultimate layer: stride=1 (deepens RF without extra downsampling)
        nf_prev = nf
        nf      = min(nf * 2, ndf * 8)
        layers.append(nn.Sequential(
            SN(nn.Conv2d(nf_prev, nf, kw, stride=1, padding=padw)),
            nn.InstanceNorm2d(nf, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Output layer: 1 channel logit map
        layers.append(SN(nn.Conv2d(nf, 1, kw, stride=1, padding=padw)))

        self.layers = layers
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, input_nc, H, W]  — cat(H&E, IHC) in [-1, 1]
        Returns:
            List of feature maps [f0, f1, ..., logits].
            f0..f_{n-2} are intermediate features (used for feature matching).
            logits [-1] is the final per-patch prediction map [B, 1, H', W'].
        """
        features = []
        out = x
        for layer in self.layers:
            out = layer(out)
            features.append(out)
        return features


class MultiScalePatchGAN(nn.Module):
    """
    Multi-scale PatchGAN discriminator.

    Runs `num_scales` NLayerDiscriminators on progressively 2x downsampled
    versions of the input.  The finer scale focuses on local texture; the
    coarser scale enforces global coherence.

    Following pix2pixHD:
      - Scale 0: full resolution
      - Scale 1: 2x average-pooled input
      - (Scale 2: 4x average-pooled — rarely used)

    Args:
        input_nc:   input channels per discriminator (default 6)
        ndf:        base channel width per scale (default 64)
        n_layers:   layers per PatchGAN (default 4)
        num_scales: number of scales (default 2)
    """

    def __init__(
        self,
        input_nc:   int = 6,
        ndf:        int = 64,
        n_layers:   int = 4,
        num_scales: int = 2,
    ):
        super().__init__()
        self.discs = nn.ModuleList([
            NLayerDiscriminator(input_nc, ndf, n_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        total = sum(p.numel() for p in self.parameters())
        print(
            f"MultiScalePatchGAN: {num_scales} scales × NLayer(ndf={ndf}, n={n_layers}) "
            f"| {total/1e6:.2f}M params"
        )

    def forward(
        self,
        he:  torch.Tensor,    # [B, 3, H, W]  H&E condition
        ihc: torch.Tensor,    # [B, 3, H, W]  real or fake IHC
    ) -> List[List[torch.Tensor]]:
        """
        Args:
            he:  H&E condition [B, 3, H, W] in [-1, 1]
            ihc: IHC target    [B, 3, H, W] in [-1, 1]  (real or fake)

        Returns:
            per_scale_features: list of length num_scales, each element is the
                feature list returned by NLayerDiscriminator.forward().
                Outer index = scale, inner index = layer.
                Last layer (index -1) at each scale is the logit map.
        """
        inp = torch.cat([he, ihc], dim=1)   # [B, 6, H, W]
        per_scale = []
        for disc in self.discs:
            per_scale.append(disc(inp))
            # Downsample input for the next (coarser) scale
            inp = self.downsample(inp)
        return per_scale

    def compute_r1_penalty(
        self,
        he:       torch.Tensor,    # [B, 3, H, W]
        ihc_real: torch.Tensor,    # [B, 3, H, W] real IHC (requires_grad must be True)
        gamma:    float = 10.0,
    ) -> torch.Tensor:
        """
        Lazy R1 gradient penalty (Mescheder et al., 2018).

        Penalises the squared L2 norm of gradients of D(real) w.r.t. real images,
        stabilising D training without mode collapse.

        Call every `r1_interval` steps and scale the returned loss by `r1_interval`
        to match the effective per-step regularisation strength (StyleGAN2 style).

        Args:
            he:       H&E condition [B, 3, H, W] in [-1, 1]
            ihc_real: real IHC images — must have requires_grad=True
            gamma:    R1 penalty coefficient (default 10.0)
        Returns:
            scalar tensor — (gamma / 2) * E[||∇_x D(x)||^2]
        """
        inp = torch.cat([he, ihc_real], dim=1)
        B   = inp.shape[0]

        # Aggregate logit scores across all scales
        total_logit = torch.zeros(B, device=inp.device)
        inp_cur = inp
        for disc in self.discs:
            feats = disc(inp_cur)
            total_logit = total_logit + feats[-1].view(B, -1).mean(1)
            inp_cur = self.downsample(inp_cur)

        # Gradient of sum(D) w.r.t. real IHC
        real_grads = torch.autograd.grad(
            outputs=total_logit.sum(),
            inputs=ihc_real,
            create_graph=True,
            retain_graph=True,
        )[0]

        penalty = (gamma / 2.0) * real_grads.pow(2).view(B, -1).sum(1).mean()
        return penalty
