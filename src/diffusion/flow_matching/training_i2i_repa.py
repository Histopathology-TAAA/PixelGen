"""
I2I REPA Trainer for Image-to-Image virtual staining.
Based on training_repa_JiT_LPIPS_DINO_NoiseGating.py.
Adapted for paired image translation with condition image concatenation.
"""
import torch
import torch.nn as nn

from src.utils.no_grad import no_grad, freeze_model
from typing import Callable
from src.diffusion.base.training import BaseTrainer
from src.diffusion.base.scheduling import BaseScheduler
import lpips


def inverse_sigma(alpha, sigma):
    return 1 / sigma**2

def snr(alpha, sigma):
    return alpha / sigma

def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, min=threshold)

def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, max=threshold)

def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)


class I2IREPATrainer(BaseTrainer):
    """
    REPA trainer for Image-to-Image tasks.
    
    The condition image is retrieved from metadata and passed as `y` to the
    denoiser, which concatenates it channel-wise with the noisy target.
    
    Losses:
    - Flow matching (v-prediction from x-prediction)
    - REPA (cosine alignment with encoder features: DINO or Virchow)
    - LPIPS perceptual loss
    - Encoder perceptual loss
    - Noise gating for perceptual losses
    """
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable = constant,
            feat_loss_weight: float = 0.5,
            lognorm_t=False,
            timeshift=1.0,
            encoder: nn.Module = None,
            align_layer=8,
            proj_denoiser_dim=768,
            proj_hidden_dim=768,
            proj_encoder_dim=768,
            P_mean=-0.8,
            P_std=0.8,
            t_eps=0.05,
            lpips_weight: float = 0.1,
            dino_weight: float = 0.01,
            encoder_percept_weight: float = None,
            encoder_type: str = "dino",
            encoder_layers: list = None,
            percept_t_threshold: float = 0.3,
            noise_scale: float = 1.0,
            patch_size: int = 16,
            percept_ratio: float = 1.0,
            # DAB stain-aware loss parameters
            dab_weight: float = 0.0,
            dab_patch_sizes: list = None,
            dab_use_focal: bool = True,
            dab_focal_alpha: float = 1.8,
            dab_hist_weight: float = 1.0,
            dab_fod_threshold: float = 0.15,
            dab_weight_alpha: float = 5.0,
            # Source flow: interpolate H&E -> IHC instead of noise -> IHC
            source_flow: bool = False,
            # Noise added to the H&E source to prevent degenerate [H&E|H&E] inputs
            source_noise_scale: float = 0.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = encoder
        freeze_model(self.encoder)

        self.lpips_loss_fn = lpips.LPIPS(net='vgg').eval()
        freeze_model(self.lpips_loss_fn)
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(proj_denoiser_dim, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_encoder_dim),
        )
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight  # backward-compatible config key
        self.encoder_percept_weight = dino_weight if encoder_percept_weight is None else encoder_percept_weight
        self.encoder_type = encoder_type.lower()
        if self.encoder_type not in {"dino", "virchow"}:
            raise ValueError(f"Unsupported encoder_type='{encoder_type}'. Use 'dino' or 'virchow'.")
        self.percept_t_threshold = percept_t_threshold
        if encoder_layers is not None:
            self.encoder_layers = encoder_layers
        elif self.encoder_type == "dino":
            self.encoder_layers = [11]
        else:
            self.encoder_layers = [0]
        self.dino_layers = self.encoder_layers  # backward-compatible attribute name
        self.noise_scale = noise_scale
        self.percept_ratio = percept_ratio
        self.cached_percept_weight = 1.0
        self.source_flow = source_flow
        self.source_noise_scale = source_noise_scale

        # DAB stain-aware loss
        self.dab_weight = dab_weight
        if dab_weight > 0:
            from src.diffusion.flow_matching.dap_loss import CombinedDABLoss
            self.dab_loss_fn = CombinedDABLoss(
                patch_sizes=dab_patch_sizes or [16, 32, 64],
                use_focal=dab_use_focal,
                focal_alpha=dab_focal_alpha,
                hist_weight=dab_hist_weight,
                fod_threshold=dab_fod_threshold,
                weight_alpha=dab_weight_alpha,
            )
            freeze_model(self.dab_loss_fn)  # no trainable params, but freeze for safety

    def _calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 100.0).detach()
        return d_weight

    def compute_encoder_loss(self, pred_feats, gt_feats, percept_mask=None):
        cos_losses = {}
        final_cos_loss = 0
        batch_size = pred_feats[0].shape[0]
        for i, (pred_feat, gt_feat) in enumerate(zip(pred_feats, gt_feats)):
            if percept_mask is not None:
                percept_mask_r = percept_mask.reshape(batch_size, 1, 1)
                cos_sim = (torch.nn.functional.cosine_similarity(pred_feat, gt_feat, dim=-1) * percept_mask_r).mean(dim=(1, 2))
                cos_sim = cos_sim.sum() / percept_mask_r.sum() if percept_mask_r.sum() > 0.0 else cos_sim.sum()
                cos_loss = 1 - cos_sim
            else:
                cos_loss = 1 - torch.nn.functional.cosine_similarity(pred_feat, gt_feat, dim=-1).view(batch_size, -1).mean()
            cos_losses[f"inter_cos_{i}"] = cos_loss
            cos_losses[f"{self.encoder_type}_inter_cos_{i}"] = cos_loss
            final_cos_loss += cos_loss
        encoder_percept_loss = final_cos_loss / len(pred_feats)
        cos_losses["encoder_percept_loss"] = encoder_percept_loss
        cos_losses["dino_percept_loss"] = encoder_percept_loss  # backward-compatible logging key
        return cos_losses

    def compute_dino_loss(self, pred_dino_feats, gt_dino_feats, percept_mask=None):
        return self.compute_encoder_loss(pred_dino_feats, gt_dino_feats, percept_mask)

    def compute_lpips_loss(self, pred_img, x, percept_mask=None):
        batch_size, _, height, width = pred_img.shape
        if self.patch_size != 16:
            new_scale = int(height * 16 // self.patch_size)
            pred_img = torch.nn.functional.interpolate(pred_img, size=(new_scale, new_scale), mode='bilinear', align_corners=False, antialias=True)
            x = torch.nn.functional.interpolate(x, size=(new_scale, new_scale), mode='bilinear', align_corners=False, antialias=True)
        if percept_mask is not None:
            lpips_loss = (self.lpips_loss_fn(pred_img, x).view(batch_size, -1) * percept_mask).mean(dim=1)
            lpips_loss = lpips_loss.sum() / percept_mask.sum() if percept_mask.sum() > 0.0 else lpips_loss.sum()
        else:
            lpips_loss = self.lpips_loss_fn(pred_img, x).mean()
        return lpips_loss

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        """
        x: target IHC image (normalized [-1,1])
        y: condition (masked by preprocess - may be H&E image or zeros)
        metadata: dict with 'raw_image' (IHC raw [0,1]), 'condition_image' etc.
        """
        raw_images = metadata["raw_image"]  # IHC target raw [0,1] for DINO features
        condition_image = y  # H&E condition (already processed by preprocess with CFG masking)
        current_step = metadata.get("global_step", 0)
        
        batch_size, c, height, width = x.shape
        self.lpips_loss_fn.eval()

        # Sample timesteps
        if self.lognorm_t:
            base_t = (torch.randn(batch_size, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)

        # Forward diffusion
        if self.source_flow:
            # Flow from H&E -> IHC: interpolate between condition and target
            # Add noise to prevent degenerate [H&E|H&E] input at low-t and
            # to avoid the model learning a simple color-copy shortcut.
            source = condition_image + self.source_noise_scale * torch.randn_like(x)
        else:
            # Flow from noise -> IHC: interpolate between noise and target
            source = self.noise_scale * torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        sigma = self.scheduler.sigma(t)

        x_t = alpha * x + source * sigma

        # v target (x-prediction -> v)
        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        # Forward through denoiser: pass condition_image as y
        # The denoiser will concatenate x_t and condition_image internally
        pred_img, src_feature = net(x_t, t, condition_image, return_layer=self.align_layer)
        src_feature = self.proj(src_feature)

        # Compute v from x-prediction
        out = (pred_img - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        # REPA: align with encoder features of target image
        with torch.no_grad():
            dst_features = self.encoder.get_intermediate_feats(raw_images, n=self.encoder_layers)
        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_features[-1], dim=-1)
        cos_loss = 1 - cos_sim

        # Flow matching loss
        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight * (out - v_t) ** 2

        # Noise gating for perceptual losses
        if self.percept_t_threshold > 0.0:
            percept_mask = (t >= self.percept_t_threshold).float().reshape(batch_size, -1)
        else:
            percept_mask = None

        # LPIPS loss (compare predicted clean image with target)
        lpips_loss = self.compute_lpips_loss(pred_img, x, percept_mask)

        # Encoder perceptual loss (compare encoder features of predicted vs target)
        raw_pred_img = (pred_img + 1) / 2  # Convert to [0,1] for encoder
        pred_feats = self.encoder.get_intermediate_feats(raw_pred_img, n=self.encoder_layers)
        encoder_losses = self.compute_encoder_loss(pred_feats, dst_features, percept_mask)

        # DAB stain-aware loss (operates on [0,1] pixel-space images)
        if self.dab_weight > 0:
            # Use float32 for stain deconvolution (log10, matrix inv need precision)
            dab_losses = self.dab_loss_fn(raw_pred_img.float(), raw_images.float())
            dab_loss = dab_losses['total']
        else:
            dab_loss = torch.tensor(0.0, device=x.device)

        # Combine losses
        rec_loss = fm_loss.mean()
        percept_loss = self.lpips_weight * lpips_loss + self.encoder_percept_weight * encoder_losses["encoder_percept_loss"]

        # Adaptive weight balancing (after warmup)
        if current_step >= 10000 and current_step % 50 == 0:
            last_layer = net.final_layer.linear.weight
            percept_weight = self._calculate_adaptive_weight(rec_loss, percept_loss, last_layer)
            self.cached_percept_weight = 0.8 * self.cached_percept_weight + 0.2 * percept_weight

        final_loss = (
            fm_loss.mean()
            + self.feat_loss_weight * cos_loss.mean()
            + self.percept_ratio * self.cached_percept_weight * percept_loss
            + self.dab_weight * dab_loss
        )

        out = dict(
            fm_loss=fm_loss.mean(),
            cos_loss=cos_loss.mean(),
            percept_weight=self.cached_percept_weight,
            encoder_percept_weight=self.encoder_percept_weight,
            lpips_loss=lpips_loss,
            dab_loss=dab_loss,
            loss=final_loss,
        )
        out.update(encoder_losses)
        if self.dab_weight > 0:
            out['dab_patch'] = dab_losses['patch']
            out['dab_hist'] = dab_losses['histogram']
        return out

    def __call__(self, net, ema_net, solver, x, condition, uncondition=None, metadata=None):
        """Override BaseTrainer to skip CFG null-condition masking for I2I."""
        # No null condition dropout — condition is always the real H&E image
        return self._impl_trainstep(net, ema_net, solver, x, condition, metadata)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)
        return destination

