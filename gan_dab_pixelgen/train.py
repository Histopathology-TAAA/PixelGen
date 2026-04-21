"""
GAN DABPixelGen Training Loop.

Alternating D / G updates per batch:

  ── D step ────────────────────────────────────────────────────────────────
  1. G(H&E) → fake IHC  (no_grad on G, or detach fake before D)
  2. D(H&E, real_IHC) → real_feats
  3. D(H&E, fake.detach()) → fake_feats
  4. d_loss = hinge/lsgan(real_feats, fake_feats)
  5. Every r1_interval steps: add lazy R1 gradient penalty

  ── G step ────────────────────────────────────────────────────────────────
  6. G(H&E) → (dab_pred, fake_IHC, h_norm)  [fresh forward with grad]
  7. D(H&E, fake_IHC) → fake_feats_g
  8. D(H&E, real_IHC) → real_feats_g         [no_grad on D]
  9. g_loss = adv + feat_match + L1 + DAB + LPIPS

Optimizer: TTUR — D uses 4x higher LR than G.
EMA: applied to G weights every step (optionally).

Validation: generate IHC from H&E via single G forward pass,
            compare to GT IHC using PSNR / SSIM / LPIPS / MAE.
            Comparison grid: H&E | GT IHC | Generated IHC | Pred DAB | GT DAB
"""
import gc
import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam
from accelerate import Accelerator
from tqdm import tqdm
import wandb

from gan_dab_pixelgen.losses import GANLoss, GANDABLoss, discriminator_loss
from dab_pixelgen.stain_utils import normalize_h_density, analytical_recompose


# ── Validation metrics (reused from dab_pixelgen.train) ──────────────────────

_lpips_fn = None

def _get_lpips(device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips as lpips_lib
            _lpips_fn = lpips_lib.LPIPS(net="alex").eval().to(device)
            for p in _lpips_fn.parameters():
                p.requires_grad = False
        except ImportError:
            _lpips_fn = False
    return _lpips_fn if _lpips_fn is not False else None


@torch.no_grad()
def _ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for i in range(pred.shape[0]):
        p, t = pred[i:i+1], target[i:i+1]
        mu_p = F.avg_pool2d(p, 11, stride=1, padding=5)
        mu_t = F.avg_pool2d(t, 11, stride=1, padding=5)
        s_p  = F.avg_pool2d(p * p, 11, stride=1, padding=5) - mu_p ** 2
        s_t  = F.avg_pool2d(t * t, 11, stride=1, padding=5) - mu_t ** 2
        s_pt = F.avg_pool2d(p * t, 11, stride=1, padding=5) - mu_p * mu_t
        num  = (2 * mu_p * mu_t + C1) * (2 * s_pt + C2)
        den  = (mu_p ** 2 + mu_t ** 2 + C1) * (s_p + s_t + C2)
        vals.append((num / den).mean().item())
    return float(np.mean(vals))


@torch.no_grad()
def compute_val_metrics(
    dab_pred:   torch.Tensor,   # [N, 1, H, W]
    dab_gt:     torch.Tensor,   # [N, 1, H, W]
    ihc_gen:    torch.Tensor,   # [N, 3, H, W] in [0, 1]
    ihc_gt:     torch.Tensor,   # [N, 3, H, W] in [0, 1]
    device:     str = "cuda",
) -> Dict[str, float]:
    dab_pred = dab_pred.to(device)
    dab_gt   = dab_gt.to(device)
    ihc_gen  = ihc_gen.to(device)
    ihc_gt   = ihc_gt.to(device)
    N = ihc_gen.shape[0]

    mse_per  = F.mse_loss(ihc_gen, ihc_gt, reduction="none").reshape(N, -1).mean(1)
    ihc_psnr = (-10 * torch.log10(mse_per.clamp(min=1e-10))).mean().item()
    ihc_mae  = F.l1_loss(ihc_gen, ihc_gt).item()
    ihc_ssim = _ssim_batch(ihc_gen, ihc_gt)

    ihc_lpips = float("nan")
    fn = _get_lpips(device)
    if fn is not None:
        ihc_lpips = fn(ihc_gen * 2 - 1, ihc_gt * 2 - 1).mean().item()

    dab_mae  = F.l1_loss(dab_pred, dab_gt).item()
    dab_mse  = F.mse_loss(dab_pred, dab_gt).item()

    p_flat = dab_pred.reshape(N, -1)
    g_flat = dab_gt.reshape(N, -1)
    p_c    = p_flat - p_flat.mean(1, keepdim=True)
    g_c    = g_flat - g_flat.mean(1, keepdim=True)
    pearsonr = (
        (p_c * g_c).sum(1) / (p_c.norm(1) * g_c.norm(1)).clamp(min=1e-8)
    ).mean().item()

    return {
        "val/ihc_psnr":     ihc_psnr,
        "val/ihc_ssim":     ihc_ssim,
        "val/ihc_lpips":    ihc_lpips,
        "val/ihc_mae":      ihc_mae,
        "val/dab_mae":      dab_mae,
        "val/dab_mse":      dab_mse,
        "val/dab_pearsonr": pearsonr,
    }


def _dab_to_rgb_vis(dab: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W] DAB OD → [B, 3, H, W] brown-on-white pseudo-colour."""
    od = dab.clamp(0, 2) / 2.0
    return torch.cat([1 - 0.65 * od, 1 - 0.85 * od, 1 - 0.95 * od], dim=1).clamp(0, 1)


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate_validation_samples(
    generator,
    discriminator,
    val_dataloader,
    epoch,
    global_step,
    config,
    accelerator,
) -> Dict[str, float]:
    """
    Run validation: single G forward pass for each validation batch.
    Logs metrics and a comparison grid to W&B.

    Grid panels: H&E | GT IHC | Generated IHC | Pred DAB | GT DAB
    """
    if not accelerator.is_main_process:
        return {}

    generator.eval()
    he_list, ihc_list, dab_gt_list = [], [], []
    val_iter = iter(val_dataloader)
    max_batches = config.num_val_batches if config.num_val_batches > 0 else float("inf")
    seen = 0

    while seen < max_batches:
        try:
            batch = next(val_iter)
            he_list.append(batch["he"])
            ihc_list.append(batch["ihc_01"])
            dab_gt_list.append(batch["dab_gt"])
            seen += 1
        except StopIteration:
            break

    if not he_list:
        generator.train()
        return {}

    he_all     = torch.cat(he_list,     0)
    ihc_gt_all = torch.cat(ihc_list,    0)
    dab_gt_all = torch.cat(dab_gt_list, 0)
    N = he_all.shape[0]

    chunk = val_dataloader.batch_size or config.batch_size
    dab_parts, ihc_parts = [], []

    for start in range(0, N, chunk):
        end   = min(start + chunk, N)
        he_c  = he_all[start:end].to(accelerator.device)

        dab_c, ihc_c, _ = accelerator.unwrap_model(generator)(he_c)
        ihc_c_01 = (ihc_c.clamp(-1, 1) + 1) / 2

        dab_parts.append(dab_c.cpu())
        ihc_parts.append(ihc_c_01.cpu())

    dab_pred_all = torch.cat(dab_parts, 0)
    ihc_gen_all  = torch.cat(ihc_parts, 0)

    metrics = compute_val_metrics(
        dab_pred_all, dab_gt_all, ihc_gen_all, ihc_gt_all,
        device=accelerator.device,
    )
    metrics["val/num_images"] = float(N)
    wandb.log({**metrics, "epoch": epoch}, step=global_step)

    lpips_str = (
        f"  LPIPS={metrics['val/ihc_lpips']:.4f}"
        if not np.isnan(metrics["val/ihc_lpips"]) else ""
    )
    print(
        f"  Val [{N} imgs] IHC: PSNR={metrics['val/ihc_psnr']:.2f}  "
        f"SSIM={metrics['val/ihc_ssim']:.4f}{lpips_str}  "
        f"MAE={metrics['val/ihc_mae']:.4f} | "
        f"DAB: MAE={metrics['val/dab_mae']:.4f}  r={metrics['val/dab_pearsonr']:.4f}"
    )

    # Comparison grid
    n_vis = min(config.num_val_samples, N)
    he_vis       = ((he_all[:n_vis] + 1) / 2).clamp(0, 1)
    dab_pred_vis = _dab_to_rgb_vis(dab_pred_all[:n_vis])
    dab_gt_vis   = _dab_to_rgb_vis(dab_gt_all[:n_vis])

    imgs = []
    for i in range(n_vis):
        panels = [
            TF.to_pil_image(he_vis[i]),
            TF.to_pil_image(ihc_gt_all[i]),
            TF.to_pil_image(ihc_gen_all[i]),
            TF.to_pil_image(dab_pred_vis[i]),
            TF.to_pil_image(dab_gt_vis[i]),
        ]
        w, h = panels[0].size
        grid = Image.new("RGB", (w * 5, h))
        for j, img in enumerate(panels):
            grid.paste(img, (j * w, 0))
        imgs.append(wandb.Image(
            grid,
            caption=f"Sample {i+1}: H&E | GT IHC | Generated IHC | Pred DAB | GT DAB",
        ))
    wandb.log({"validation/comparison_grid": imgs, "epoch": epoch}, step=global_step)

    generator.train()
    return metrics


# ── Utility: instance noise for D ─────────────────────────────────────────────

def _instance_noise_std(step: int, config) -> float:
    """Linearly decay instance noise std from init to 0 over decay_steps."""
    if config.instance_noise_std <= 0.0:
        return 0.0
    frac = min(1.0, step / max(config.instance_noise_decay_steps, 1))
    return config.instance_noise_std * (1.0 - frac)


def _add_instance_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise to D input (clamp to [-1, 1] to stay in image range)."""
    if std <= 0.0:
        return x
    return (x + torch.randn_like(x) * std).clamp(-1, 1)


# ── Main training loop ────────────────────────────────────────────────────────

def train_gan(
    generator,
    discriminator,
    train_dataloader,
    val_dataloader,
    g_optimizer,
    d_optimizer,
    g_lr_scheduler,
    accelerator,
    config,
    g_loss_fn:          GANDABLoss,
    gan_loss:           GANLoss,
    ema_tracker=None,
    start_epoch:        int = 0,
):
    """
    Main GAN DABPixelGen training loop.

    Batch keys from dataloader:
        he:           [B, 3, H, W]  H&E in [-1, 1]
        ihc:          [B, 3, H, W]  IHC in [-1, 1]  (used for D real)
        ihc_01:       [B, 3, H, W]  IHC in [0, 1]   (used for G L1 / LPIPS)
        dab_gt:       [B, 1, H, W]  raw DAB density from IHC
        dab_gt_fod:   [B, 1, H, W]  FOD-transformed DAB (for DAB loss)
        h_he_density: [B, 1, H, W]  H density from H&E (not used in GAN path)
    """
    if accelerator.is_main_process:
        wandb.init(
            project = config.wandb_project,
            name    = config.wandb_name,
            config  = {k: str(v) for k, v in vars(config).items()},
            tags    = ["gan-dab-pixelgen", "patchgan", "he-to-ihc"],
            resume  = "allow",
        )

    checkpoint_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step    = start_epoch * len(train_dataloader)
    best_val_ssim  = -float("inf")
    d_step_counter = 0   # tracks D updates for lazy R1 scheduling

    for epoch in range(start_epoch, config.num_epochs):
        generator.train()
        discriminator.train()

        ep_d = ep_g = ep_adv = ep_fm = ep_l1 = ep_dab = 0.0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(progress):
            he       = batch["he"]         # [-1, 1]
            ihc_real = batch["ihc"]        # [-1, 1]
            ihc_01   = batch["ihc_01"]     # [0, 1]
            dab_gt   = batch["dab_gt"]
            dab_gt_fod = batch["dab_gt_fod"]

            noise_std = _instance_noise_std(global_step, config)

            # ── D update ──────────────────────────────────────────────────────
            for _ in range(config.d_steps_per_g):
                with torch.no_grad():
                    _, fake_ihc, _ = accelerator.unwrap_model(generator)(he)
                    fake_ihc = fake_ihc.detach()

                # Add instance noise to D inputs for early-training stability
                real_inp = _add_instance_noise(ihc_real, noise_std)
                fake_inp = _add_instance_noise(fake_ihc, noise_std)

                real_feats = discriminator(he, real_inp)
                fake_feats = discriminator(he, fake_inp)

                d_loss, d_log = discriminator_loss(gan_loss, real_feats, fake_feats)

                # Lazy R1 gradient penalty
                r1_loss = torch.tensor(0.0, device=he.device)
                if d_step_counter % config.r1_interval == 0:
                    # R1 requires gradients w.r.t. real IHC
                    real_r1 = real_inp.detach().requires_grad_(True)
                    r1_loss = discriminator.compute_r1_penalty(
                        he, real_r1, gamma=config.r1_gamma
                    ) * config.r1_interval   # scale to match per-step strength

                d_total = d_loss + r1_loss
                d_optimizer.zero_grad()
                accelerator.backward(d_total)
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), config.max_grad_norm
                )
                d_optimizer.step()
                d_step_counter += 1

            # ── G update ──────────────────────────────────────────────────────
            dab_pred, fake_ihc, h_norm = generator(he)

            real_inp_g = _add_instance_noise(ihc_real, noise_std)
            fake_inp_g = _add_instance_noise(fake_ihc, noise_std)

            # D features for real (no_grad on D params during G update)
            with torch.no_grad():
                real_feats_g = discriminator(he, real_inp_g)
            fake_feats_g = discriminator(he, fake_inp_g)

            g_total, g_log = g_loss_fn(
                dab_pred   = dab_pred,
                ihc_pred   = fake_ihc,
                fake_feats = fake_feats_g,
                real_feats = real_feats_g,
                dab_gt_fod = dab_gt_fod,
                ihc_gt_01  = ihc_01,
            )

            g_optimizer.zero_grad()
            accelerator.backward(g_total)
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), config.max_grad_norm
            )
            g_optimizer.step()
            if g_lr_scheduler is not None:
                g_lr_scheduler.step()

            if ema_tracker is not None:
                ema_tracker.update(accelerator.unwrap_model(generator))

            # ── Accumulate epoch stats ────────────────────────────────────────
            ep_d   += d_loss.item()
            ep_g   += g_total.item()
            ep_adv += g_log["g/adv"].item()
            ep_fm  += g_log["g/feat_match"].item()
            ep_l1  += g_log["g/l1"].item()
            ep_dab += g_log["g/dab"].item()
            global_step += 1

            # ── W&B step logging ──────────────────────────────────────────────
            if global_step % config.log_every == 0 and accelerator.is_main_process:
                wandb.log({
                    "train/d_loss":    d_loss.item(),
                    "train/r1_loss":   r1_loss.item(),
                    "train/g_total":   g_total.item(),
                    "train/g_adv":     g_log["g/adv"].item(),
                    "train/g_fm":      g_log["g/feat_match"].item(),
                    "train/g_l1":      g_log["g/l1"].item(),
                    "train/g_dab":     g_log["g/dab"].item(),
                    "train/g_lpips":   g_log["g/lpips"].item(),
                    "train/noise_std": noise_std,
                    "train/epoch":     epoch + 1,
                }, step=global_step)

            progress.set_postfix(
                D=f"{d_loss.item():.3f}",
                G=f"{g_total.item():.3f}",
                adv=f"{g_log['g/adv'].item():.3f}",
                L1=f"{g_log['g/l1'].item():.3f}",
            )

        n = len(train_dataloader)
        if accelerator.is_main_process:
            wandb.log({
                "epoch/d_loss":  ep_d / n,
                "epoch/g_total": ep_g / n,
                "epoch/g_adv":   ep_adv / n,
                "epoch/g_fm":    ep_fm / n,
                "epoch/g_l1":    ep_l1 / n,
                "epoch/g_dab":   ep_dab / n,
                "epoch/epoch":   epoch + 1,
            }, step=global_step)
        print(
            f"Epoch {epoch+1}: D={ep_d/n:.4f} | G={ep_g/n:.4f} | "
            f"adv={ep_adv/n:.4f} | FM={ep_fm/n:.4f} | "
            f"L1={ep_l1/n:.4f} | DAB={ep_dab/n:.4f}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % config.val_every == 0:
            if ema_tracker is not None:
                ema_tracker.apply_shadow(accelerator.unwrap_model(generator))
            metrics = generate_validation_samples(
                generator, discriminator, val_dataloader,
                epoch, global_step, config, accelerator,
            )
            if ema_tracker is not None:
                ema_tracker.restore(accelerator.unwrap_model(generator))

            if metrics and metrics.get("val/ihc_ssim", 0) > best_val_ssim:
                best_val_ssim = metrics["val/ihc_ssim"]
                if accelerator.is_main_process:
                    _save_checkpoint(
                        accelerator, generator, discriminator,
                        g_optimizer, d_optimizer, ema_tracker, epoch,
                        checkpoint_dir / "gan_dab_best.pt",
                    )
                    print(f"  ↑ Best SSIM={best_val_ssim:.4f} saved.")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % config.save_every == 0 and accelerator.is_main_process:
            _save_checkpoint(
                accelerator, generator, discriminator,
                g_optimizer, d_optimizer, ema_tracker, epoch,
                checkpoint_dir / f"gan_dab_ep{epoch+1:04d}.pt",
            )

    if accelerator.is_main_process:
        wandb.finish()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(accelerator, generator, discriminator,
                     g_optim, d_optim, ema, epoch, path):
    gen_sd  = accelerator.unwrap_model(generator).state_dict()
    disc_sd = accelerator.unwrap_model(discriminator).state_dict()
    ckpt = {
        "epoch":        epoch,
        "generator":    gen_sd,
        "discriminator":disc_sd,
        "g_optimizer":  g_optim.state_dict(),
        "d_optimizer":  d_optim.state_dict(),
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, generator, discriminator, g_optim=None, d_optim=None, ema=None):
    """Load a GAN checkpoint into the provided modules."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    if g_optim is not None and "g_optimizer" in ckpt:
        g_optim.load_state_dict(ckpt["g_optimizer"])
    if d_optim is not None and "d_optimizer" in ckpt:
        d_optim.load_state_dict(ckpt["d_optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    epoch = ckpt.get("epoch", 0)
    print(f"Loaded GAN checkpoint from epoch {epoch}: {path}")
    return epoch
