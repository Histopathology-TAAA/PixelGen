"""
Stain-aware losses for H&E to IHC translation.

Includes differentiable stain deconvolution and patch-level DAB loss.
Incorporates Focal Optical Density (FOD) transform from PSPStain paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_optical_density(dab_od: torch.Tensor, alpha: float = 1.6) -> torch.Tensor:
    """
    Apply Focal Optical Density (FOD) transform from PSPStain paper.
    
    This power transform emphasizes strongly positive regions while 
    suppressing weak signals, similar to focal loss in detection.
    
    Args:
        dab_od: (B, 1, H, W) DAB optical density values
        alpha: Power exponent (default: 1.8 from PSPStain)
        
    Returns:
        (B, 1, H, W) Focal optical density values
    """
    # Calibration factor from PSPStain: 10^(-(e)^(1/alpha))
    calibration = 10 ** (-(math.e) ** (1 / alpha))
    
    # Convert OD back to intensity-like value for FOD computation
    # OD = -log10(I), so I = 10^(-OD)
    intensity = torch.pow(10.0, -dab_od)
    
    # FOD = log10(1 / (intensity + calibration))
    fod = torch.log10(1.0 / (intensity + calibration))
    
    # Clamp negative values and apply power transform
    fod = F.relu(fod) ** alpha
    
    return fod


class StainDeconvolution(nn.Module):
    """
    Differentiable stain deconvolution for H-DAB (Hematoxylin-DAB) images.
    
    Based on Ruifrok & Johnston (2001) color deconvolution.
    Same matrices as skimage.color.hdx_from_rgb / rgb2hed.
    
    Returns optical density (OD) values for each stain channel.
    Higher OD = more stain present.
    """
    
    def __init__(self):
        super().__init__()
        
        # Stain vectors (from skimage hdx_from_rgb, normalized)
        # These define how each stain absorbs R, G, B light
        stain_matrix = torch.tensor([
            [0.650, 0.704, 0.286],  # Hematoxylin
            [0.268, 0.570, 0.776],  # DAB
            [0.711, 0.423, 0.561],  # Residual (background/eosin)
        ], dtype=torch.float32)
        
        # Invert for deconvolution (RGB OD -> stain OD)
        stain_matrix_inv = torch.linalg.inv(stain_matrix)
        
        # Register as buffer (not trainable, but moves with .to(device))
        self.register_buffer('stain_matrix_inv', stain_matrix_inv)
    
    def forward(self, img: torch.Tensor) -> dict:
        """
        Deconvolve RGB image into stain channels.
        
        Args:
            img: (B, 3, H, W) RGB image in [0, 1] range
            
        Returns:
            dict with keys:
                - 'hematoxylin': (B, 1, H, W) Hematoxylin OD
                - 'dab': (B, 1, H, W) DAB OD  
                - 'residual': (B, 1, H, W) Residual OD
        """
        # Clamp to avoid log(0)
        img = img.clamp(min=1e-6, max=1.0)
        
        # Convert RGB to optical density: OD = -log10(I)
        # (assuming I0 = 1, white background)
        od = -torch.log10(img)
        
        # Reshape: (B, 3, H, W) -> (B, H, W, 3)
        od_bhw3 = od.permute(0, 2, 3, 1)
        
        # Apply deconvolution: multiply by inverse stain matrix
        # (B, H, W, 3) @ (3, 3) -> (B, H, W, 3)
        stains = torch.matmul(od_bhw3, self.stain_matrix_inv.T)
        
        # Clamp negative values (can occur due to noise/out-of-gamut colors)
        stains = stains.clamp(min=0)
        
        # Reshape back: (B, H, W, 3) -> (B, 3, H, W)
        stains = stains.permute(0, 3, 1, 2)
        
        return {
            'hematoxylin': stains[:, 0:1],  # (B, 1, H, W)
            'dab': stains[:, 1:2],          # (B, 1, H, W)
            'residual': stains[:, 2:3],     # (B, 1, H, W)
        }


class PatchDABLoss(nn.Module):
    """
    Patch-level DAB loss for handling imperfect spatial alignment.
    
    Instead of pixel-wise comparison (which fails with misalignment),
    this compares DAB statistics within local patches, allowing for
    small spatial shifts while still encouraging correct regional
    expression patterns.
    
    Args:
        patch_size: Size of patches for pooling (default: 32)
        use_weighting: Weight patches by GT DAB intensity (default: True)
        weight_alpha: Weighting strength (default: 5.0)
        loss_type: 'l1' or 'l2' (default: 'l1')
    """
    
    def __init__(
        self, 
        patch_size: int = 32,
        use_weighting: bool = True,
        weight_alpha: float = 5.0,
        loss_type: str = 'l1'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_weighting = use_weighting
        self.weight_alpha = weight_alpha
        self.loss_type = loss_type
        
        self.stain_deconv = StainDeconvolution()
    
    def forward(
        self, 
        pred_img: torch.Tensor, 
        gt_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute patch-level DAB loss.
        
        Args:
            pred_img: (B, 3, H, W) Predicted IHC image in [0, 1]
            gt_img: (B, 3, H, W) Ground truth IHC image in [0, 1]
            
        Returns:
            Scalar loss value
        """
        # Extract DAB channels
        pred_stains = self.stain_deconv(pred_img)
        gt_stains = self.stain_deconv(gt_img)
        
        pred_dab = pred_stains['dab']  # (B, 1, H, W)
        gt_dab = gt_stains['dab']      # (B, 1, H, W)
        
        # Compute patch-level means using average pooling
        pred_patches = F.avg_pool2d(pred_dab, kernel_size=self.patch_size)
        gt_patches = F.avg_pool2d(gt_dab, kernel_size=self.patch_size)
        
        # Compute loss
        if self.loss_type == 'l1':
            patch_diff = torch.abs(pred_patches - gt_patches)
        else:  # l2
            patch_diff = (pred_patches - gt_patches) ** 2
        
        # Optional: weight by GT DAB intensity to emphasize positive regions
        if self.use_weighting:
            # Higher weight for patches with more DAB staining
            weight = 1.0 + self.weight_alpha * gt_patches
            patch_diff = weight * patch_diff
        
        return patch_diff.mean()


class MultiScalePatchDABLoss(nn.Module):
    """
    Multi-scale version of patch DAB loss with optional Focal OD transform.
    
    Computes DAB loss at multiple patch sizes to capture both
    fine-grained and coarse regional patterns.
    
    Args:
        patch_sizes: List of patch sizes for multi-scale pooling
        use_weighting: Weight patches by GT intensity (default: True)
        weight_alpha: Weighting strength (default: 5.0)
        use_focal: Use Focal Optical Density transform (default: True)
        focal_alpha: Power exponent for FOD (default: 1.8 from PSPStain)
        fod_threshold: Threshold to zero out weak FOD values (default: 0.15)
    """
    
    def __init__(
        self,
        patch_sizes: list = [16, 32, 64],
        use_weighting: bool = True,
        weight_alpha: float = 5.0,
        use_focal: bool = True,
        focal_alpha: float = 1.8,
        fod_threshold: float = 0.15,
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.use_weighting = use_weighting
        self.weight_alpha = weight_alpha
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.fod_threshold = fod_threshold
        
        self.stain_deconv = StainDeconvolution()
    
    def forward(
        self, 
        pred_img: torch.Tensor, 
        gt_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale patch DAB loss.
        """
        # Extract DAB channels once
        pred_dab = self.stain_deconv(pred_img)['dab']
        gt_dab = self.stain_deconv(gt_img)['dab']
        
        # Apply Focal Optical Density transform if enabled
        if self.use_focal:
            pred_dab = focal_optical_density(pred_dab, alpha=self.focal_alpha)
            gt_dab = focal_optical_density(gt_dab, alpha=self.focal_alpha)
            
            # Threshold to zero out weak FOD values (reduces noise impact)
            pred_dab = torch.where(
                pred_dab < self.fod_threshold, 
                torch.zeros_like(pred_dab), 
                pred_dab
            )
            gt_dab = torch.where(
                gt_dab < self.fod_threshold, 
                torch.zeros_like(gt_dab), 
                gt_dab
            )
        
        total_loss = 0.0
        
        for patch_size in self.patch_sizes:
            pred_patches = F.avg_pool2d(pred_dab, kernel_size=patch_size)
            gt_patches = F.avg_pool2d(gt_dab, kernel_size=patch_size)
            
            patch_diff = torch.abs(pred_patches - gt_patches)
            
            if self.use_weighting:
                weight = 1.0 + self.weight_alpha * gt_patches
                patch_diff = weight * patch_diff
            
            total_loss = total_loss + patch_diff.mean()
        
        # Average over scales
        return total_loss / len(self.patch_sizes)


class DABHistogramLoss(nn.Module):
    """
    Histogram matching loss for DAB channel with optional Focal OD transform.
    
    Matches the distribution of DAB intensities without requiring
    spatial alignment. Good for ensuring overall expression levels match.
    
    Uses soft histogram (differentiable) via kernel density estimation.
    
    Args:
        num_bins: Number of histogram bins (default: 64)
        sigma: Gaussian kernel width for soft binning (default: 0.02)
        use_focal: Use Focal Optical Density transform (default: True)
        focal_alpha: Power exponent for FOD (default: 1.8)
    """
    
    def __init__(
        self, 
        num_bins: int = 64, 
        sigma: float = 0.02,
        use_focal: bool = True,
        focal_alpha: float = 1.8,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        
        self.stain_deconv = StainDeconvolution()
        
        # Bin centers - adjust range based on whether using focal transform
        # FOD values typically range from 0 to ~e (2.718) with alpha=1.8
        if use_focal:
            bin_centers = torch.linspace(0, math.e, num_bins)
        else:
            bin_centers = torch.linspace(0, 2.5, num_bins)
        self.register_buffer('bin_centers', bin_centers)
    
    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable soft histogram using Gaussian kernels.
        
        Args:
            x: (B, 1, H, W) input values
            
        Returns:
            (B, num_bins) soft histogram
        """
        B = x.shape[0]
        x_flat = x.view(B, -1, 1)  # (B, N, 1)
        
        # Distance to each bin center
        bins = self.bin_centers.view(1, 1, -1)  # (1, 1, num_bins)
        
        # Gaussian kernel
        weights = torch.exp(-0.5 * ((x_flat - bins) / self.sigma) ** 2)
        
        # Sum to get histogram, normalize
        hist = weights.sum(dim=1)  # (B, num_bins)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        
        return hist
    
    def forward(
        self, 
        pred_img: torch.Tensor, 
        gt_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute histogram matching loss for DAB channel.
        """
        pred_dab = self.stain_deconv(pred_img)['dab']
        gt_dab = self.stain_deconv(gt_img)['dab']
        
        # Apply Focal Optical Density transform if enabled
        if self.use_focal:
            pred_dab = focal_optical_density(pred_dab, alpha=self.focal_alpha)
            gt_dab = focal_optical_density(gt_dab, alpha=self.focal_alpha)
        
        pred_hist = self.soft_histogram(pred_dab)
        gt_hist = self.soft_histogram(gt_dab)
        
        # L1 distance between histograms (approx Earth Mover's Distance)
        # Using cumulative histograms for better gradient
        pred_cdf = torch.cumsum(pred_hist, dim=1)
        gt_cdf = torch.cumsum(gt_hist, dim=1)
        
        return torch.abs(pred_cdf - gt_cdf).mean()


class CombinedDABLoss(nn.Module):
    """
    Combined DAB loss using both patch-level and histogram matching,
    with Focal Optical Density (FOD) transform from PSPStain paper.
    
    Args:
        patch_weight: Weight for patch-level loss (default: 1.0)
        hist_weight: Weight for histogram loss (default: 0.5)
        patch_sizes: List of patch sizes for multi-scale (default: [16, 32, 64])
        use_focal: Use Focal Optical Density transform (default: True)
        focal_alpha: Power exponent for FOD (default: 1.8 from PSPStain)
        fod_threshold: Threshold to zero out weak FOD values (default: 0.15)
        weight_alpha: Intensity weighting strength (default: 5.0)
    """
    
    def __init__(
        self,
        patch_weight: float = 1.0,
        hist_weight: float = 1.0,
        patch_sizes: list = [16, 32, 64],
        use_focal: bool = True,
        focal_alpha: float = 1.8,
        fod_threshold: float = 0.15,
        weight_alpha: float = 5.0,
    ):
        super().__init__()
        self.patch_weight = patch_weight
        self.hist_weight = hist_weight
        
        self.patch_loss = MultiScalePatchDABLoss(
            patch_sizes=patch_sizes,
            use_weighting=True,
            weight_alpha=weight_alpha,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            fod_threshold=fod_threshold,
        )
        self.hist_loss = DABHistogramLoss(
            use_focal=use_focal,
            focal_alpha=focal_alpha,
        )
    
    def forward(
        self, 
        pred_img: torch.Tensor, 
        gt_img: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            dict with 'total', 'patch', and 'histogram' losses
        """
        patch = self.patch_loss(pred_img, gt_img)
        hist = self.hist_loss(pred_img, gt_img)
        
        total = self.patch_weight * patch + self.hist_weight * hist
        
        return {
            'total': total,
            'patch': patch,
            'histogram': hist,
        }



