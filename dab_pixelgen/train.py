"""
DABPixelGen Training Loop.

Single JiT_I2I backbone that predicts 1-channel DAB density and
H-normalization parameters, with end-to-end recomposition loss.

Loss breakdown per step:
  L = w_fm    * L_fm(v_pred, v_target)
    + w_dab   * L_dab(dab_pred, dab_gt)      [gated: t >= threshold]
    + w_recomp * L_recomp(recomposed, ihc_gt) [gated: t >= threshold]
"""
import gc
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import numpy as np

from dab_pixelgen.stain_utils import normalize_h_density, analytical_recompose


# ── Validation metrics ────────────────────────────────────────────────────────

# Lazy LPIPS singleton — loaded once on first call, None if not installed.
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
            _lpips_fn = False   # mark as unavailable
    return _lpips_fn if _lpips_fn is not False else None


@torch.no_grad()
def _ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean SSIM over a [B, C, H, W] batch in [0, 1].
    Uses an 11×11 Gaussian-approximated sliding window.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i:i+1]
        t = target[i:i+1]
        mu_p  = F.avg_pool2d(p, 11, stride=1, padding=5)
        mu_t  = F.avg_pool2d(t, 11, stride=1, padding=5)
        s_p   = F.avg_pool2d(p * p, 11, stride=1, padding=5) - mu_p ** 2
        s_t   = F.avg_pool2d(t * t, 11, stride=1, padding=5) - mu_t ** 2
        s_pt  = F.avg_pool2d(p * t, 11, stride=1, padding=5) - mu_p * mu_t
        num = (2 * mu_p * mu_t + C1) * (2 * s_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (s_p + s_t + C2)
        vals.append((num / den).mean().item())
    return float(np.mean(vals))


@torch.no_grad()
def compute_val_metrics(
    dab_pred:   torch.Tensor,   # [N, 1, H, W]  predicted DAB OD
    dab_gt:     torch.Tensor,   # [N, 1, H, W]  GT DAB OD
    recomposed: torch.Tensor,   # [N, 3, H, W]  recomposed IHC in [0, 1]
    ihc_gt:     torch.Tensor,   # [N, 3, H, W]  GT IHC in [0, 1]
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute pixel-level and perceptual metrics on the validation batch.

    IHC RGB metrics  (recomposed vs GT IHC):
        val/ihc_psnr   ↑   peak signal-to-noise ratio
        val/ihc_ssim   ↑   structural similarity
        val/ihc_lpips  ↓   perceptual distance (Alex-Net, if lpips installed)
        val/ihc_mae    ↓   mean absolute error

    DAB OD metrics  (predicted density vs GT density):
        val/dab_mae    ↓   mean absolute error on optical density values
        val/dab_mse    ↓   mean squared error
        val/dab_pearsonr ↑  Pearson correlation (linear agreement)
    """
    dab_pred   = dab_pred.to(device)
    dab_gt     = dab_gt.to(device)
    recomposed = recomposed.to(device)
    ihc_gt     = ihc_gt.to(device)
    N = recomposed.shape[0]

    # ── IHC RGB ───────────────────────────────────────────────────────────────
    ihc_mse_per = F.mse_loss(recomposed, ihc_gt, reduction="none").view(N, -1).mean(1)
    ihc_psnr    = (-10 * torch.log10(ihc_mse_per.clamp(min=1e-10))).mean().item()
    ihc_mae     = F.l1_loss(recomposed, ihc_gt).item()
    ihc_ssim    = _ssim_batch(recomposed, ihc_gt)

    ihc_lpips = float("nan")
    lpips_fn = _get_lpips(device)
    if lpips_fn is not None:
        # LPIPS expects [-1, 1]
        ihc_lpips = lpips_fn(
            recomposed * 2 - 1, ihc_gt * 2 - 1
        ).mean().item()

    # ── DAB OD ────────────────────────────────────────────────────────────────
    dab_mae = F.l1_loss(dab_pred, dab_gt).item()
    dab_mse = F.mse_loss(dab_pred, dab_gt).item()

    # Pearson r across all spatial positions, averaged over images
    p_flat = dab_pred.view(N, -1)
    g_flat = dab_gt.view(N, -1)
    p_c = p_flat - p_flat.mean(dim=1, keepdim=True)
    g_c = g_flat - g_flat.mean(dim=1, keepdim=True)
    num = (p_c * g_c).sum(dim=1)
    den = (p_c.norm(dim=1) * g_c.norm(dim=1)).clamp(min=1e-8)
    dab_pearsonr = (num / den).mean().item()

    return {
        "val/ihc_psnr":    ihc_psnr,
        "val/ihc_ssim":    ihc_ssim,
        "val/ihc_lpips":   ihc_lpips,
        "val/ihc_mae":     ihc_mae,
        "val/dab_mae":     dab_mae,
        "val/dab_mse":     dab_mse,
        "val/dab_pearsonr":dab_pearsonr,
    }


# ── Validation ────────────────────────────────────────────────────────────────

def _dab_to_rgb_vis(dab: torch.Tensor) -> torch.Tensor:
    """
    Convert a [B, 1, H, W] DAB optical-density map to a 3-channel
    pseudo-colour image in [0, 1] for display.

    We use a brown-on-white colormap matching IHC DAB appearance:
      - low OD  -> white (1, 1, 1)
      - high OD -> brown (0.35, 0.15, 0.05)
    """
    od = dab.clamp(0, 2) / 2.0           # [B, 1, H, W]  0..1
    r = 1.0 - 0.65 * od
    g = 1.0 - 0.85 * od
    b = 1.0 - 0.95 * od
    return torch.cat([r, g, b], dim=1).clamp(0, 1)  # [B, 3, H, W]


@torch.no_grad()
def generate_validation_samples(
    model, scheduler, val_dataloader, epoch, global_step, config, accelerator
) -> Dict[str, float]:
    """
    Run full validation: generate DAB predictions + recompose IHC for up to
    `num_val_batches` batches, compute metrics, log grid + scalars to W&B.

    Returns the metrics dict (empty dict on non-main processes).

    Grid panels per row:
        H&E | GT IHC | Recomposed IHC | Pred DAB | GT DAB

    Metrics logged under val/:
        ihc_psnr, ihc_ssim, ihc_lpips, ihc_mae
        dab_mae,  dab_mse,  dab_pearsonr
    """
    if not accelerator.is_main_process:
        return {}

    model.eval()
    he_list, ihc_list, h_he_list, dab_gt_list = [], [], [], []
    val_iter = iter(val_dataloader)
    max_batches = config.num_val_batches if config.num_val_batches > 0 else float("inf")
    batches_seen = 0

    # Collect batches up to the limit
    while batches_seen < max_batches:
        try:
            batch = next(val_iter)
            he_list.append(batch["he"])
            ihc_list.append(batch["ihc_01"])
            h_he_list.append(batch["h_he_density"])
            dab_gt_list.append(batch["dab_gt"])
            batches_seen += 1
        except StopIteration:
            break

    if not he_list:
        model.train()
        return {}

    he_all       = torch.cat(he_list,     dim=0)
    ihc_gt_all   = torch.cat(ihc_list,    dim=0)
    h_he_all     = torch.cat(h_he_list,   dim=0)
    dab_gt_all   = torch.cat(dab_gt_list, dim=0)
    N_total = he_all.shape[0]

    # ── Run sampling in mini-batches to avoid OOM ─────────────────────────────
    # Use the val dataloader batch size as chunk size
    chunk = val_dataloader.batch_size or config.batch_size
    dab_pred_parts, recomp_parts = [], []

    for start in range(0, N_total, chunk):
        end = min(start + chunk, N_total)
        he_c       = he_all[start:end].to(accelerator.device)
        h_he_c     = h_he_all[start:end].to(accelerator.device)

        shape = (he_c.shape[0], 1, config.image_size, config.image_size)
        dab_pred_c, h_norm_c = scheduler.sample(
            accelerator.unwrap_model(model),
            he_c, shape,
            device=accelerator.device,
            h_he_density=h_he_c,
            progress=False,
        )
        h_ihc_c    = normalize_h_density(h_he_c, h_norm_c[:, 0], h_norm_c[:, 1])
        recomp_c   = analytical_recompose(h_ihc_c, dab_pred_c, device=accelerator.device)
        dab_pred_parts.append(dab_pred_c.cpu())
        recomp_parts.append(recomp_c.cpu())

    dab_pred_all = torch.cat(dab_pred_parts, dim=0)   # [N, 1, H, W]
    recomp_all   = torch.cat(recomp_parts,   dim=0)   # [N, 3, H, W]

    # ── Metrics (on all collected images) ────────────────────────────────────
    metrics = compute_val_metrics(
        dab_pred_all, dab_gt_all,
        recomp_all,   ihc_gt_all,
        device=accelerator.device,
    )
    metrics["val/num_images"] = float(N_total)

    # ── W&B scalar logging ────────────────────────────────────────────────────
    wandb.log({**metrics, "epoch": epoch}, step=global_step)

    # ── Print summary ─────────────────────────────────────────────────────────
    lpips_str = (
        f"  LPIPS={metrics['val/ihc_lpips']:.4f}"
        if not np.isnan(metrics["val/ihc_lpips"]) else ""
    )
    print(
        f"  Val [{N_total} imgs] "
        f"IHC: PSNR={metrics['val/ihc_psnr']:.2f}  "
        f"SSIM={metrics['val/ihc_ssim']:.4f}{lpips_str}  "
        f"MAE={metrics['val/ihc_mae']:.4f} | "
        f"DAB: MAE={metrics['val/dab_mae']:.4f}  "
        f"r={metrics['val/dab_pearsonr']:.4f}"
    )

    # ── Visual grid (first num_val_samples images only) ───────────────────────
    n_vis = min(config.num_val_samples, N_total)
    he_vis       = ((he_all[:n_vis] + 1) / 2).clamp(0, 1)
    dab_pred_vis = _dab_to_rgb_vis(dab_pred_all[:n_vis])
    dab_gt_vis   = _dab_to_rgb_vis(dab_gt_all[:n_vis])

    imgs = []
    for i in range(n_vis):
        panels = [
            TF.to_pil_image(he_vis[i]),
            TF.to_pil_image(ihc_gt_all[i]),
            TF.to_pil_image(recomp_all[i]),
            TF.to_pil_image(dab_pred_vis[i]),
            TF.to_pil_image(dab_gt_vis[i]),
        ]
        w, h = panels[0].size
        combined = Image.new("RGB", (w * 5, h))
        for j, img in enumerate(panels):
            combined.paste(img, (j * w, 0))
        imgs.append(wandb.Image(
            combined,
            caption=f"Sample {i+1}: H&E | GT IHC | Recomposed IHC | Pred DAB | GT DAB",
        ))

    wandb.log({"validation/comparison_grid": imgs, "epoch": epoch}, step=global_step)

    model.train()
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train_dab(
    model,
    scheduler,
    train_dataloader,
    val_dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    config,
    loss_fn,
    ema_tracker=None,
    start_epoch: int = 0,
):
    """
    Main DABPixelGen training loop.

    Batch keys expected from dataloader:
        he:           [B, 3, H, W]  H&E in [-1, 1]
        ihc:          [B, 3, H, W]  IHC in [-1, 1]  (unused)
        ihc_01:       [B, 3, H, W]  IHC in [0, 1]   (for recomp loss)
        h_he_density: [B, 1, H, W]  H density from H&E (PSPStain)
        dab_gt:       [B, 1, H, W]  raw DAB density from IHC (for recomposition)
        dab_gt_fod:   [B, 1, H, W]  FOD DAB from IHC (for expression supervision)
    """
    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config={k: str(v) for k, v in vars(config).items()},
            tags=["dab-pixelgen", "jit-i2i", "he-to-ihc", "beer-lambert"],
            resume="allow",
        )

    checkpoint_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scheduler.to(accelerator.device)

    global_step  = start_epoch * len(train_dataloader)
    best_loss    = float("inf")
    best_val_ssim = -float("inf")   # track best val SSIM for checkpoint saving

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        ep_fm = ep_dab = ep_recomp = ep_total = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                he           = batch["he"]
                ihc_01       = batch["ihc_01"]
                h_he_density = batch["h_he_density"]
                dab_gt       = batch["dab_gt"]        # raw density → recomposition
                dab_gt_fod   = batch["dab_gt_fod"]    # FOD → expression supervision
                B = dab_gt.shape[0]

                # ── Timestep sampling (logit-normal, focuses on mid-t) ────────
                u = torch.randn((B,), device=accelerator.device)
                t = torch.sigmoid(u)

                # ── Build starting distribution (H&E warm-start) ────────────
                noise = torch.randn_like(dab_gt)
                alpha = config.he_init_alpha
                x_0 = (1.0 - alpha) * noise + alpha * h_he_density

                # ── Forward process ───────────────────────────────────────────
                x_t, v_target = scheduler.q_sample(dab_gt, t, x_0)

                # ── Model forward ─────────────────────────────────────────────
                x1_pred, h_norm_params = model(x_t, t, he)

                # ── Combined loss ─────────────────────────────────────────────
                total_loss, log_dict = loss_fn(
                    x1_pred=x1_pred,
                    h_norm_params=h_norm_params,
                    x_t=x_t,
                    v_target=v_target,
                    t=t,
                    dab_gt=dab_gt,
                    dab_gt_fod=dab_gt_fod,
                    h_he_density=h_he_density,
                    ihc_rgb_gt=ihc_01,
                )

                accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if ema_tracker is not None:
                    ema_tracker.update(accelerator.unwrap_model(model))

            ep_fm     += log_dict["loss_fm"].item()
            ep_dab    += log_dict["loss_dab"].item()
            ep_recomp += log_dict["loss_recomp"].item()
            ep_total  += log_dict["loss_total"].item()
            global_step += 1

            if global_step % config.log_every == 0 and accelerator.is_main_process:
                log = {
                    "train/loss":        total_loss.item(),
                    "train/loss_fm":     log_dict["loss_fm"].item(),
                    "train/loss_dab":    log_dict["loss_dab"].item(),
                    "train/loss_recomp": log_dict["loss_recomp"].item(),
                    "train/loss_recomp_mse":   log_dict["loss_recomp_mse"].item(),
                    "train/loss_recomp_lpips": log_dict["loss_recomp_lpips"].item(),
                    "train/lr":          lr_scheduler.get_last_lr()[0],
                    "train/epoch":       epoch + 1,
                }
                wandb.log(log, step=global_step)

            progress_bar.set_postfix(
                total=f"{total_loss.item():.4f}",
                fm=f"{log_dict['loss_fm'].item():.4f}",
                dab=f"{log_dict['loss_dab'].item():.4f}",
                rec=f"{log_dict['loss_recomp'].item():.4f}",
            )

        n = len(train_dataloader)
        avg_fm, avg_dab, avg_rec, avg_tot = (
            ep_fm / n, ep_dab / n, ep_recomp / n, ep_total / n
        )
        if accelerator.is_main_process:
            wandb.log({
                "epoch/loss_fm":     avg_fm,
                "epoch/loss_dab":    avg_dab,
                "epoch/loss_recomp": avg_rec,
                "epoch/loss_total":  avg_tot,
                "epoch/epoch":       epoch + 1,
            }, step=global_step)
        print(
            f"Epoch {epoch+1}: FM={avg_fm:.4f} | DAB={avg_dab:.4f} | "
            f"Recomp={avg_rec:.4f} | Total={avg_tot:.4f}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = {}
        if (epoch + 1) % config.val_every == 0:
            if ema_tracker is not None:
                ema_tracker.apply_shadow(accelerator.unwrap_model(model))
            val_metrics = generate_validation_samples(
                model, scheduler, val_dataloader,
                epoch + 1, global_step, config, accelerator
            )
            if ema_tracker is not None:
                ema_tracker.restore(accelerator.unwrap_model(model))

        # ── Checkpoint ───────────────────────────────────────────────────────
        val_ssim = val_metrics.get("val/ihc_ssim", None)
        is_best  = False

        if val_ssim is not None:
            # Prefer val SSIM as the primary quality signal when available
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                is_best = True
        elif avg_tot < best_loss:
            # Fall back to train loss on epochs without validation
            best_loss = avg_tot
            is_best = True

        if (epoch + 1) % config.save_every == 0 or is_best:
            ckpt = {
                "epoch":           epoch + 1,
                "loss_fm":         avg_fm,
                "loss_dab":        avg_dab,
                "loss_recomp":     avg_rec,
                "loss_total":      avg_tot,
                "val_metrics":     val_metrics,
                "model_state_dict":accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "config":          {k: str(v) for k, v in vars(config).items()},
            }
            if ema_tracker is not None:
                ckpt["ema_state_dict"] = ema_tracker.state_dict()

            latest_path = checkpoint_dir / "dab_pixelgen_latest.pt"
            torch.save(ckpt, latest_path)
            if accelerator.is_main_process:
                wandb.save(str(latest_path), base_path=str(checkpoint_dir), policy="now")

            if is_best:
                best_path = checkpoint_dir / "dab_pixelgen_best.pt"
                torch.save(ckpt, best_path)
                if accelerator.is_main_process:
                    wandb.save(str(best_path), base_path=str(checkpoint_dir), policy="now")
                if val_ssim is not None:
                    print(f"  New best model (val SSIM={val_ssim:.4f})")
                else:
                    print(f"  New best model (train loss={avg_tot:.4f})")

    if accelerator.is_main_process:
        wandb.finish()
    return model


# ── Resume helper ─────────────────────────────────────────────────────────────

def resume_from_wandb(
    run_path, file_name, model, optimizer, lr_scheduler, accelerator,
    ema_tracker=None, force_redownload=False
):
    """Download and load a checkpoint from W&B."""
    download_dir = "./wandb_downloads"
    os.makedirs(download_dir, exist_ok=True)
    ckpt_path = os.path.join(download_dir, file_name)

    if force_redownload and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"Downloading from W&B: {run_path}/{file_name}")
        api = wandb.Api()
        run = api.run(run_path)
        for f in run.files():
            if file_name in f.name:
                f.download(root=download_dir, replace=True)
                ckpt_path = os.path.join(download_dir, f.name)
                print(f"  Downloaded: {ckpt_path}")
                break
        else:
            raise FileNotFoundError(f"'{file_name}' not found in run {run_path}")

    ckpt = torch.load(ckpt_path, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema_tracker is not None and "ema_state_dict" in ckpt:
        ema_tracker.load_state_dict(ckpt["ema_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch
