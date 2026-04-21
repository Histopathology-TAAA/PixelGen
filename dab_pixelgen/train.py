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


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_validation_samples(
    model, scheduler, val_dataloader, epoch, global_step, config, accelerator
):
    """Generate DAB predictions, recompose, and log to W&B."""
    if not accelerator.is_main_process:
        return

    model.eval()
    he_list, ihc_list, h_he_list = [], [], []
    val_iter = iter(val_dataloader)
    while len(he_list) < config.num_val_samples:
        try:
            batch = next(val_iter)
            he_list.append(batch["he"])
            ihc_list.append(batch["ihc_01"])
            h_he_list.append(batch["h_he_density"])
        except StopIteration:
            break

    he       = torch.cat(he_list,    dim=0)[:config.num_val_samples]
    ihc_gt   = torch.cat(ihc_list,   dim=0)[:config.num_val_samples]
    h_he_den = torch.cat(h_he_list,  dim=0)[:config.num_val_samples]
    n = he.shape[0]

    shape = (n, 1, config.image_size, config.image_size)
    dab_pred, h_norm = scheduler.sample(
        accelerator.unwrap_model(model),
        he.to(accelerator.device),
        shape,
        device=accelerator.device,
        h_he_density=h_he_den.to(accelerator.device),
        progress=False,
    )

    # Recompose RGB
    a_raw = h_norm[:, 0]
    b_raw = h_norm[:, 1]
    h_ihc = normalize_h_density(h_he_den.to(accelerator.device), a_raw, b_raw)
    recomposed = analytical_recompose(h_ihc, dab_pred, device=accelerator.device)

    def to_01(x):
        return ((x + 1) / 2).clamp(0, 1)

    he_vis  = to_01(he)
    dab_vis = dab_pred.clamp(0, 2) / 2.0   # scale OD to [0,1] for display
    dab_vis = dab_vis.expand(-1, 3, -1, -1) # grayscale -> 3ch for PIL

    imgs = []
    for i in range(n):
        panels = [
            TF.to_pil_image(he_vis[i].cpu()),
            TF.to_pil_image(ihc_gt[i].cpu()),
            TF.to_pil_image(recomposed[i].cpu()),
            TF.to_pil_image(dab_vis[i].cpu()),
        ]
        w, h = panels[0].size
        combined = Image.new("RGB", (w * 4, h))
        for j, img in enumerate(panels):
            combined.paste(img, (j * w, 0))
        imgs.append(wandb.Image(
            combined,
            caption=f"Sample {i+1}: H&E | GT IHC | Recomposed | DAB map"
        ))

    wandb.log({"validation/comparison_grid": imgs, "epoch": epoch}, step=global_step)
    print(f"  Logged {n} validation samples for epoch {epoch}")
    model.train()


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
        h_he_density: [B, 1, H, W]  H OD from H&E
        dab_gt:       [B, 1, H, W]  DAB OD from IHC
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

    global_step = start_epoch * len(train_dataloader)
    best_loss   = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        ep_fm = ep_dab = ep_recomp = ep_total = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                he           = batch["he"]
                ihc_01       = batch["ihc_01"]
                h_he_density = batch["h_he_density"]
                dab_gt       = batch["dab_gt"]
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
        if (epoch + 1) % config.val_every == 0:
            if ema_tracker is not None:
                ema_tracker.apply_shadow(accelerator.unwrap_model(model))
            generate_validation_samples(
                model, scheduler, val_dataloader,
                epoch + 1, global_step, config, accelerator
            )
            if ema_tracker is not None:
                ema_tracker.restore(accelerator.unwrap_model(model))

        # ── Checkpoint ───────────────────────────────────────────────────────
        if (epoch + 1) % config.save_every == 0 or avg_tot < best_loss:
            ckpt = {
                "epoch":           epoch + 1,
                "loss_fm":         avg_fm,
                "loss_dab":        avg_dab,
                "loss_recomp":     avg_rec,
                "loss_total":      avg_tot,
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

            if avg_tot < best_loss:
                best_loss = avg_tot
                best_path = checkpoint_dir / "dab_pixelgen_best.pt"
                torch.save(ckpt, best_path)
                if accelerator.is_main_process:
                    wandb.save(str(best_path), base_path=str(checkpoint_dir), policy="now")
                print(f"  New best model (loss={avg_tot:.4f})")

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
