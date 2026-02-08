"""
Lightning Model for Image-to-Image virtual staining.
Extends the base LightningModel with:
- Condition image handling in training/validation
- SSIM, PSNR, FID metrics during validation
- 8-sample visualization logging to W&B
"""
from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict, List
import os.path
import copy
import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback

from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.conditioner.base import BaseConditioner
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

torch._functorch.config.donated_buffer = False

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class I2ILightningModel(pl.LightningModule):
    def __init__(
        self,
        vae: BaseAE,
        conditioner: BaseConditioner,
        denoiser: nn.Module,
        diffusion_trainer: BaseTrainer,
        diffusion_sampler: BaseSampler,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        num_vis_samples: int = 8,
    ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_original_model = eval_original_model
        self.num_vis_samples = num_vis_samples

        self._strict_loading = False

        # Validation accumulators
        self._val_ssim_scores = []
        self._val_psnr_scores = []
        self._val_vis_samples = []  # For W&B visualization
        self._val_gen_images = []   # For FID
        self._val_gt_images = []    # For FID

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # Disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.ema_denoiser)

        # Skip torch.compile for T4 compatibility
        # self.denoiser.compile()
        # self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser},
            {"params": params_trainer},
            {"params": params_sampler, "lr": 1e-3},
        ]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(optimizer=optimizer)
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler={
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate",
                },
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self._val_ssim_scores = []
        self._val_psnr_scores = []
        self._val_vis_samples = []
        self._val_gen_images = []
        self._val_gt_images = []

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def on_load_checkpoint(self, checkpoint):
        keys_to_check = [
            "denoiser.pos_embed",
            "ema_denoiser.pos_embed",
        ]
        ckpt_state_dict = checkpoint["state_dict"]
        current_state_dict = self.state_dict()
        for key in keys_to_check:
            if key in ckpt_state_dict and key in current_state_dict:
                ckpt_shape = ckpt_state_dict[key].shape
                curr_shape = current_state_dict[key].shape
                if ckpt_shape != curr_shape:
                    print(
                        f"[Warning] Shape mismatch for '{key}': "
                        f"Checkpoint {ckpt_shape} vs Current {curr_shape}. "
                        f"Dropping from checkpoint."
                    )
                    del ckpt_state_dict[key]

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        if metadata is None:
            metadata = {}
        metadata["global_step"] = self.global_step
        with torch.no_grad():
            x = self.vae.encode(x)
            condition, uncondition = self.conditioner(y, metadata)
        loss = self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser,
            self.diffusion_sampler,
            x,
            condition,
            uncondition,
            metadata,
        )
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def _generate_samples(self, noise, condition_image):
        """Generate IHC samples from noise and H&E condition using the EMA model."""
        # condition_image is the H&E input
        uncondition = torch.zeros_like(condition_image)

        if self.eval_original_model:
            net = self.denoiser
        else:
            net = self.ema_denoiser

        samples = self.diffusion_sampler(net, noise, condition_image, uncondition)
        samples = self.vae.decode(samples)
        return samples

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        # Stack metadata condition images
        if isinstance(metadata, (list, tuple)):
            condition_images = torch.stack([m["condition_image"] for m in metadata]).to(xT.device)
        else:
            condition_images = metadata["condition_image"].to(xT.device)

        samples = self._generate_samples(xT, condition_images)
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        xT, y, metadata = batch

        # Extract condition and ground truth from metadata
        if isinstance(metadata, (list, tuple)):
            condition_images = torch.stack([m["condition_image"] for m in metadata]).to(xT.device)
            gt_images = torch.stack([m["gt_image"] for m in metadata]).to(xT.device)
            gt_images_raw = torch.stack([m["gt_image_raw"] for m in metadata]).to(xT.device)
            condition_images_raw = torch.stack([m["condition_image_raw"] for m in metadata]).to(xT.device)
        else:
            condition_images = metadata["condition_image"].to(xT.device)
            gt_images = metadata["gt_image"].to(xT.device)
            gt_images_raw = metadata["gt_image_raw"].to(xT.device)
            condition_images_raw = metadata["condition_image_raw"].to(xT.device)

        # Generate samples
        with torch.no_grad():
            gen_samples = self._generate_samples(xT, condition_images)

        # Convert generated to [0, 1] range
        gen_samples_01 = (gen_samples.float().clamp(-1, 1) + 1) / 2

        # Compute SSIM and PSNR per sample
        ssim_vals = self._compute_ssim_batch(gen_samples_01, gt_images_raw)
        psnr_vals = self._compute_psnr_batch(gen_samples_01, gt_images_raw)

        self._val_ssim_scores.extend(ssim_vals)
        self._val_psnr_scores.extend(psnr_vals)

        # Accumulate for FID (as uint8)
        gen_uint8 = (gen_samples_01 * 255).clamp(0, 255).to(torch.uint8)
        gt_uint8 = (gt_images_raw * 255).clamp(0, 255).to(torch.uint8)
        self._val_gen_images.append(gen_uint8.cpu())
        self._val_gt_images.append(gt_uint8.cpu())

        # Collect visualization samples (up to num_vis_samples)
        if len(self._val_vis_samples) < self.num_vis_samples:
            n_needed = self.num_vis_samples - len(self._val_vis_samples)
            n_take = min(n_needed, gen_samples_01.shape[0])
            for i in range(n_take):
                self._val_vis_samples.append({
                    "condition": (condition_images_raw[i].cpu() * 255).clamp(0, 255).to(torch.uint8),
                    "generated": (gen_samples_01[i].cpu() * 255).clamp(0, 255).to(torch.uint8),
                    "ground_truth": (gt_images_raw[i].cpu() * 255).clamp(0, 255).to(torch.uint8),
                })

        return gen_uint8

    def on_validation_epoch_end(self) -> None:
        # Log mean SSIM and PSNR
        if self._val_ssim_scores:
            mean_ssim = sum(self._val_ssim_scores) / len(self._val_ssim_scores)
            self.log("val/ssim", mean_ssim, prog_bar=True, sync_dist=True)

        if self._val_psnr_scores:
            mean_psnr = sum(self._val_psnr_scores) / len(self._val_psnr_scores)
            self.log("val/psnr", mean_psnr, prog_bar=True, sync_dist=True)

        # Compute and log FID
        if self._val_gen_images and self._val_gt_images:
            try:
                fid_score = self._compute_fid()
                self.log("val/fid", fid_score, prog_bar=True, sync_dist=True)
            except Exception as e:
                print(f"[Warning] FID computation failed: {e}")

        # Log visualization samples to W&B
        if self._val_vis_samples and self.logger is not None:
            self._log_wandb_visualizations()

        # Clear accumulators
        self._val_ssim_scores = []
        self._val_psnr_scores = []
        self._val_vis_samples = []
        self._val_gen_images = []
        self._val_gt_images = []

    def _compute_ssim_batch(self, pred, target):
        """Compute SSIM for a batch. pred and target are [0,1] float tensors."""
        from torchmetrics.functional.image import structural_similarity_index_measure
        scores = []
        for i in range(pred.shape[0]):
            ssim = structural_similarity_index_measure(
                pred[i:i+1], target[i:i+1], data_range=1.0
            )
            scores.append(ssim.item())
        return scores

    def _compute_psnr_batch(self, pred, target):
        """Compute PSNR for a batch. pred and target are [0,1] float tensors."""
        from torchmetrics.functional.image import peak_signal_noise_ratio
        scores = []
        for i in range(pred.shape[0]):
            psnr = peak_signal_noise_ratio(
                pred[i:i+1], target[i:i+1], data_range=1.0
            )
            scores.append(psnr.item())
        return scores

    def _compute_fid(self):
        """Compute FID between generated and ground truth images."""
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        fid = fid.to("cpu")

        gen_images = torch.cat(self._val_gen_images, dim=0)
        gt_images = torch.cat(self._val_gt_images, dim=0)

        # FID needs at least 2 samples
        if gen_images.shape[0] < 2:
            return float("nan")

        # Convert uint8 to float [0,1] for torchmetrics FID with normalize=True
        gen_float = gen_images.float() / 255.0
        gt_float = gt_images.float() / 255.0

        fid.update(gt_float, real=True)
        fid.update(gen_float, real=False)
        fid_score = fid.compute().item()
        return fid_score

    def _log_wandb_visualizations(self):
        """Log 8 side-by-side visualization samples to W&B."""
        try:
            import wandb
        except ImportError:
            return

        if not hasattr(self.logger, "experiment"):
            return

        images_list = []
        for i, sample in enumerate(self._val_vis_samples[:self.num_vis_samples]):
            # Convert CHW -> HWC numpy
            cond_np = sample["condition"].permute(1, 2, 0).numpy()
            gen_np = sample["generated"].permute(1, 2, 0).numpy()
            gt_np = sample["ground_truth"].permute(1, 2, 0).numpy()

            # Concatenate horizontally: H&E | Generated IHC | GT IHC
            concat_img = np.concatenate([cond_np, gen_np, gt_np], axis=1)
            caption = f"Sample {i} | H&E (left) | Generated IHC (center) | GT IHC (right)"
            images_list.append(wandb.Image(concat_img, caption=caption))

        self.logger.experiment.log(
            {"val/visualization": images_list},
            step=self.global_step,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix + "denoiser.",
            keep_vars=keep_vars,
        )
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix + "ema_denoiser.",
            keep_vars=keep_vars,
        )
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix + "diffusion_trainer.",
            keep_vars=keep_vars,
        )
        return destination

