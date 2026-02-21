"""
Wrapper around PSPStain evaluation code for computing KID, FID, SSIM, and PSNR
from in-memory tensors. The underlying FID/KID code is taken directly from PSPStain
without modification — this module only provides a tensor-friendly interface.

SSIM uses pytorch_msssim (same library as PSPStain).
PSNR uses standard formula: 10 * log10(MAX^2 / MSE).
MAD (Mean Absolute DAB Difference) uses stain deconvolution to extract DAB channels.
"""
import os
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim as _ssim_fn

from src.eval.fid import calculate_fid_given_paths
from src.eval.kid_score import calculate_kid_given_paths
from src.diffusion.flow_matching.dap_loss import StainDeconvolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tensor_images_to_dir(images: torch.Tensor, directory: str):
    """Save a batch of image tensors as PNG files into a directory.

    Args:
        images: (N, C, H, W) tensor in [0, 1] float or [0, 255] uint8.
        directory: Path to the output directory.
    """
    os.makedirs(directory, exist_ok=True)

    if images.dtype == torch.uint8:
        images_np = images.cpu().numpy()
    else:
        images_np = (images.float().cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()

    for i in range(images_np.shape[0]):
        img = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
        Image.fromarray(img).save(os.path.join(directory, f"{i:06d}.png"))


# ---------------------------------------------------------------------------
# FID  (from PSPStain/util/fid.py)
# ---------------------------------------------------------------------------

def compute_fid(gen_images: torch.Tensor,
                gt_images: torch.Tensor,
                batch_size: int = 50,
                dims: int = 2048,
                device: str = "cuda") -> float:
    """Compute Frechet Inception Distance between two image sets.

    Both tensors should be (N, 3, H, W) in [0, 1] float or [0, 255] uint8.
    Uses PSPStain's ``calculate_fid_given_paths`` under the hood by saving
    tensors to temporary directories.
    """
    tmpdir = tempfile.mkdtemp(prefix="pixelgen_fid_")
    gen_dir = os.path.join(tmpdir, "gen")
    gt_dir = os.path.join(tmpdir, "gt")

    try:
        _save_tensor_images_to_dir(gen_images, gen_dir)
        _save_tensor_images_to_dir(gt_images, gt_dir)

        fid_value = calculate_fid_given_paths(
            paths=[gt_dir, gen_dir],
            batch_size=batch_size,
            device=torch.device(device) if isinstance(device, str) else device,
            dims=dims,
        )
        return float(fid_value)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# KID  (from PSPStain/util/kid_score.py)
# ---------------------------------------------------------------------------

def compute_kid(gen_images: torch.Tensor,
                gt_images: torch.Tensor,
                batch_size: int = 50,
                dims: int = 2048,
                device: str = "cuda") -> tuple:
    """Compute Kernel Inception Distance between two image sets.

    Both tensors should be (N, 3, H, W) in [0, 1] float or [0, 255] uint8.
    Uses PSPStain's ``calculate_kid_given_paths`` under the hood.

    Returns:
        (kid_mean, kid_std) — mean and standard deviation of KID.
    """
    tmpdir = tempfile.mkdtemp(prefix="pixelgen_kid_")
    gen_dir = os.path.join(tmpdir, "gen")
    gt_dir = os.path.join(tmpdir, "gt")
    use_cuda = "cuda" in str(device)

    try:
        _save_tensor_images_to_dir(gen_images, gen_dir)
        _save_tensor_images_to_dir(gt_images, gt_dir)

        results = calculate_kid_given_paths(
            paths=[gt_dir, gen_dir],
            batch_size=batch_size,
            cuda=use_cuda,
            dims=dims,
        )
        # results is a list of (path, kid_mean, kid_std)
        _, kid_mean, kid_std = results[0]
        return float(kid_mean), float(kid_std)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# SSIM  (same library as PSPStain — pytorch_msssim)
# ---------------------------------------------------------------------------

def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> list:
    """Compute per-image SSIM for a batch of paired images.

    Args:
        pred:   (N, C, H, W) in [0, 1] float.
        target: (N, C, H, W) in [0, 1] float.

    Returns:
        List of SSIM values (one per image).
    """
    scores = []
    for i in range(pred.shape[0]):
        val = _ssim_fn(
            pred[i:i + 1],
            target[i:i + 1],
            data_range=1.0,
            size_average=True,
        )
        scores.append(val.item())
    return scores


# ---------------------------------------------------------------------------
# PSNR  (standard formula — PSPStain does not include PSNR)
# ---------------------------------------------------------------------------

def compute_psnr_batch(pred: torch.Tensor, target: torch.Tensor,
                       data_range: float = 1.0) -> list:
    """Compute per-image PSNR for a batch of paired images.

    Args:
        pred:       (N, C, H, W) in [0, 1] float.
        target:     (N, C, H, W) in [0, 1] float.
        data_range: The dynamic range of the images (1.0 for [0, 1]).

    Returns:
        List of PSNR values (one per image).
    """
    scores = []
    for i in range(pred.shape[0]):
        mse = torch.mean((pred[i] - target[i]) ** 2).item()
        if mse == 0:
            scores.append(float("inf"))
        else:
            psnr = 10.0 * np.log10((data_range ** 2) / mse)
            scores.append(psnr)
    return scores


# ---------------------------------------------------------------------------
# MAD (Mean Absolute DAB Difference)
# ---------------------------------------------------------------------------

# Global stain deconvolution instance (reused across calls for efficiency)
_stain_deconv = None
_stain_deconv_device = None

def _get_stain_deconv(device: torch.device = None) -> StainDeconvolution:
    """Get or create a global StainDeconvolution instance, moving to device if needed."""
    global _stain_deconv, _stain_deconv_device
    if _stain_deconv is None:
        _stain_deconv = StainDeconvolution()
        _stain_deconv_device = None
    
    # Move to device if needed
    if device is not None:
        device_str = str(device)
        if _stain_deconv_device != device_str:
            _stain_deconv = _stain_deconv.to(device)
            _stain_deconv_device = device_str
    
    return _stain_deconv


def compute_mad_batch(pred: torch.Tensor, target: torch.Tensor) -> list:
    """Compute per-image Mean Absolute DAB Difference (MAD).

    Extracts DAB optical density channels from both images using stain deconvolution,
    computes the average DAB intensity per image, and returns the absolute difference.

    Args:
        pred:   (N, C, H, W) RGB image in [0, 1] float.
        target: (N, C, H, W) RGB image in [0, 1] float.

    Returns:
        List of MAD values (one per image). Lower is better (0 = perfect match).
    """
    device = pred.device
    stain_deconv = _get_stain_deconv(device)
    
    # Extract DAB channels: (N, 1, H, W)
    pred_stains = stain_deconv(pred)
    target_stains = stain_deconv(target)
    
    pred_dab = pred_stains['dab']  # (N, 1, H, W)
    target_dab = target_stains['dab']  # (N, 1, H, W)
    
    # Compute average DAB intensity per image: (N,)
    pred_avg_dab = pred_dab.view(pred_dab.shape[0], -1).mean(dim=1)  # (N,)
    target_avg_dab = target_dab.view(target_dab.shape[0], -1).mean(dim=1)  # (N,)
    
    # Compute absolute difference: (N,)
    mad_values = torch.abs(pred_avg_dab - target_avg_dab)
    
    return mad_values.cpu().tolist()
