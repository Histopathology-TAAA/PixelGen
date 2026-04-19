"""
StarDiff + PixelGen Training Loop.
Integrates dual-path losses with PixelGen perceptual supervision.
"""
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import numpy as np




@torch.no_grad()
def generate_validation_samples(model, scheduler, val_dataloader, epoch, global_step, config, accelerator):
    """Generate and log validation samples to W&B."""
    if not accelerator.is_main_process:
        return

    model.eval()
    num_samples = config.num_val_samples
    he_list, ihc_list = [], []
    val_iter = iter(val_dataloader)
    while len(he_list) < num_samples:
        try:
            batch = next(val_iter)
            he_list.append(batch["he"])
            ihc_list.append(batch["ihc"])
        except StopIteration:
            break

    he = torch.cat(he_list, dim=0)[:num_samples]
    ihc_real = torch.cat(ihc_list, dim=0)[:num_samples]
    actual = he.shape[0]

    shape = (actual, 3, config.image_size, config.image_size)
    generated = scheduler.sample(
        accelerator.unwrap_model(model),
        he.to(accelerator.device), shape,
        device=accelerator.device,
        use_restoration=True, use_noise=True,
        progress=False,
    )

    def denorm(x):
        return ((x + 1) / 2).clamp(0, 1)

    he_vis, ihc_vis, gen_vis = denorm(he), denorm(ihc_real), denorm(generated.cpu())
    imgs = []
    for i in range(actual):
        h_img = TF.to_pil_image(he_vis[i])
        r_img = TF.to_pil_image(ihc_vis[i])
        g_img = TF.to_pil_image(gen_vis[i])
        w, h = h_img.size
        combined = Image.new("RGB", (w * 3, h))
        combined.paste(h_img, (0, 0))
        combined.paste(r_img, (w, 0))
        combined.paste(g_img, (w * 2, 0))
        imgs.append(wandb.Image(combined, caption=f"Sample {i+1}: H&E | Real IHC | Generated"))

    wandb.log({"validation/comparison_grid": imgs, "epoch": epoch}, step=global_step)
    print(f"  ✓ Logged {actual} validation samples for epoch {epoch}")
    model.train()


def train_stardiff(
    model, scheduler, train_dataloader, val_dataloader,
    optimizer, lr_scheduler, accelerator, config, perceptual_loss_fn=None,
    dab_loss_fn=None,
    ema_tracker=None,
    start_epoch=0,
    he_init_alpha=0.3,
):
    """
    Star-Diff training with independent noise + restoration paths.
    - Noise path: MSE(ε̂, ε)
    - Restoration path: MSE(r̂, I_res) + perceptual losses
    """
    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config={k: str(v) for k, v in vars(config).items()},
            tags=["star-diff", "pixelgen", "jit-i2i", "he-to-ihc"],
            resume="allow",
        )

    checkpoint_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move scheduler tensors to device
    scheduler.to(accelerator.device)

    global_step = start_epoch * len(train_dataloader)
    best_loss = float("inf")
    dab_weight = getattr(config, 'dab_weight', 0.0)

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_noise_loss = 0.0
        epoch_rest_loss = 0.0
        epoch_percept_loss = 0.0
        epoch_dab_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                he = batch["he"]
                ihc = batch["ihc"]
                residual = batch["residual"]

                raw_noise = torch.randn_like(ihc)
                bs = ihc.shape[0]

                # H&E warm-start: match the inference starting distribution
                he_init_alpha = he_init_alpha
                x_0 = (1.0 - he_init_alpha) * raw_noise + he_init_alpha * he

                # Flow Matching: logit-normal timestep sampling (focuses on hard mid-timesteps)
                u = torch.randn((bs,), device=accelerator.device) * 1.0  # scale=1.0
                t = torch.sigmoid(u)

                # Star-Diff Flow Matching forward — pass x_0 so v_target = x_1 - x_0
                x_t, v_target = scheduler.q_sample(ihc, t, x_0, residual)

                # Independent path predictions
                x1_noise_pred, x1_rest_pred = model(x_t, t, he)
                
                # REFIX: Convert both image predictions to velocity vectors.
                # This ensures we correctly leverage the JiT_I2I backbone's ImageNet pretraining.
                denom = (1.0 - t.view(-1, 1, 1, 1)).clamp_min(1e-3)
                v_noise_pred = (x1_noise_pred - x_t) / denom
                v_rest_pred = (x1_rest_pred - x_t) / denom

                # 1. Noise path loss: predict ground truth flow velocity (x_1 - noise)
                loss_noise = F.mse_loss(v_noise_pred, v_target)

                # 2. Restoration path loss: predict structural residual I_res = I_ihc - I_he
                # In the stable baseline, we train the restoration head's velocity to match the residual.
                loss_rest = F.mse_loss(v_rest_pred, residual)

                # Perceptual losses on x̂_1 (restoration path)
                percept_loss_val = torch.tensor(0.0, device=accelerator.device)
                percept_dict = {}
                if perceptual_loss_fn is not None:
                    # In Flow Matching, x1_rest_pred IS the predicted clean image x_1!
                    x_1_hat = x1_rest_pred.clamp(-1, 1)

                    percept_loss_val, percept_dict = perceptual_loss_fn(
                        x_1_hat, ihc, t, scheduler.num_timesteps
                    )

                # DAB stain-aware loss on restoration path x̂₁
                # Only active when t >= 0.7 (clean predictions) to avoid noisy stain deconvolution
                dab_loss_val = torch.tensor(0.0, device=accelerator.device)
                dab_dict = {}
                if dab_loss_fn is not None and dab_weight > 0:
                    dab_gate = (t >= 0.7).float()  # 1 for clean steps (t≥0.7), 0 for noisy
                    if dab_gate.sum() > 0:
                        # DAB loss expects [0,1] range images
                        x_1_01 = ((x1_rest_pred.clamp(-1, 1) + 1) / 2)
                        ihc_01 = ((ihc + 1) / 2).clamp(0, 1)
                        # Use float32 for stain deconvolution (log10 needs precision)
                        dab_dict = dab_loss_fn(x_1_01.float(), ihc_01.float())
                        # Gate: only count loss from samples with t >= 0.7
                        dab_loss_val = dab_dict['total'] * (dab_gate.sum() / t.shape[0])

                # Combined loss
                loss = (
                    config.noise_loss_weight * loss_noise
                    + config.restoration_loss_weight * loss_rest
                    + percept_loss_val
                    + dab_weight * dab_loss_val
                )

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA after each optimizer step
                if ema_tracker is not None:
                    ema_tracker.update(accelerator.unwrap_model(model))

            epoch_noise_loss += loss_noise.item()
            epoch_rest_loss += loss_rest.item()
            epoch_percept_loss += percept_loss_val.item()
            epoch_dab_loss += dab_loss_val.item()
            global_step += 1

            # Logging
            if global_step % config.log_every == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/noise_loss": loss_noise.item(),
                    "train/restoration_loss": loss_rest.item(),
                    "train/percept_loss": percept_loss_val.item(),
                    "train/dab_loss": dab_loss_val.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }
                if percept_dict:
                    for k, v in percept_dict.items():
                        log_dict[f"train/{k}_loss"] = v.item() if hasattr(v, "item") else float(v)
                if dab_dict:
                    for k, v in dab_dict.items():
                        if k != 'total':
                            log_dict[f"train/dab_{k}_loss"] = v.item() if hasattr(v, "item") else float(v)
                wandb.log(log_dict, step=global_step)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                noise=f"{loss_noise.item():.4f}",
                rest=f"{loss_rest.item():.4f}",
                perc=f"{percept_loss_val.item():.4f}",
                dab=f"{dab_loss_val.item():.4f}",
            )

        # Epoch summary
        n = len(train_dataloader)
        avg_noise = epoch_noise_loss / n
        avg_rest = epoch_rest_loss / n
        avg_perc = epoch_percept_loss / n
        avg_dab = epoch_dab_loss / n
        avg_total = avg_noise + avg_rest + avg_perc + avg_dab

        if accelerator.is_main_process:
            wandb.log({
                "epoch/noise_loss": avg_noise,
                "epoch/restoration_loss": avg_rest,
                "epoch/percept_loss": avg_perc,
                "epoch/dab_loss": avg_dab,
                "epoch/total_loss": avg_total,
                "epoch/epoch": epoch + 1,
            }, step=global_step)
        print(f"Epoch {epoch+1}: Noise={avg_noise:.4f} | Rest={avg_rest:.4f} | Percept={avg_perc:.4f} | DAB={avg_dab:.4f}")

        # Validation (use EMA weights if available)
        if (epoch + 1) % config.val_every == 0:
            if ema_tracker is not None:
                ema_tracker.apply_shadow(accelerator.unwrap_model(model))
            generate_validation_samples(model, scheduler, val_dataloader, epoch + 1, global_step, config, accelerator)
            if ema_tracker is not None:
                ema_tracker.restore(accelerator.unwrap_model(model))

        # Checkpoint
        if (epoch + 1) % config.save_every == 0 or avg_total < best_loss:
            ckpt = {
                "epoch": epoch + 1,
                "noise_loss": avg_noise,
                "restoration_loss": avg_rest,
                "percept_loss": avg_perc,
                "dab_loss": avg_dab,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "config": {k: str(v) for k, v in vars(config).items()},
            }
            if ema_tracker is not None:
                ckpt["ema_state_dict"] = ema_tracker.state_dict()
            
            # Save Latest (overwrites every time we save)
            latest_path = checkpoint_dir / "stardiff_latest.pt"
            torch.save(ckpt, latest_path)
            wandb.save(str(latest_path), base_path=str(checkpoint_dir), policy="now")

            if avg_total < best_loss:
                best_loss = avg_total
                best_path = checkpoint_dir / "stardiff_best.pt"
                torch.save(ckpt, best_path)
                wandb.save(str(best_path), base_path=str(checkpoint_dir), policy="now")
                print(f"  ✓ New best model (loss={avg_total:.4f})")

    wandb.finish()
    return model


def resume_from_wandb(run_path, file_name, model, optimizer, lr_scheduler, accelerator,
                      ema_tracker=None, force_redownload=False):
    """Download and load checkpoint from W&B."""
    download_dir = "./wandb_downloads"
    os.makedirs(download_dir, exist_ok=True)
    checkpoint_path = os.path.join(download_dir, file_name)

    if force_redownload and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"📥 Downloading from W&B: {run_path}/{file_name}")
        api = wandb.Api()
        run = api.run(run_path)
        for f in run.files():
            if file_name in f.name:
                f.download(root=download_dir, replace=True)
                checkpoint_path = os.path.join(download_dir, f.name)
                print(f"  ✅ Downloaded: {checkpoint_path}")
                break
        else:
            raise FileNotFoundError(f"File '{file_name}' not found in run {run_path}")

    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if ema_tracker is not None and "ema_state_dict" in checkpoint:
        ema_tracker.load_state_dict(checkpoint["ema_state_dict"])
        print("  ✅ EMA state restored")

    start_epoch = checkpoint.get("epoch", 0)
    print(f"✅ Resumed from epoch {start_epoch}")
    return start_epoch
