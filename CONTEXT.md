# Project Context & Memory

> This file is the authoritative memory document for the virtual staining adaptation of PixelGen.
> Keep it updated as experiments progress and conclusions are reached.

---

## 1. What This Project Is

**PixelGen** (upstream paper: arxiv 2602.02493) is a pixel-space flow-matching diffusion model trained on ImageNet with REPA + LPIPS + DINO perceptual losses. It achieves 5.11 FID on ImageNet-256 at 80 epochs without CFG.

**This repo extends PixelGen for computational pathology virtual staining:** given an H&E histology image, generate a corresponding IHC-stained image (e.g. ER, Ki67 markers). This is a paired image-to-image (I2I) translation problem, not unconditional generation.

---

## 2. Key Architecture

### Model: JiT_I2I (XL, ~671M params)
- `src/models/transformer/JiT_I2I.py`
- Adapted from JiT (Joint image Transformer) with an extra condition channel stream
- Input: `concat(x_t [3ch], condition_H&E [3ch])` → denoiser → predicted clean IHC
- Initialized from `PixelGen_XL_80ep.ckpt` (ImageNet pretrained, EMA weights loaded)
- Key params: `hidden_size=1152, depth=28, num_heads=16, patch_size=16`
- Since this is pixel-space: VAE = `PixelAE` (identity, scale=1, shift=0)

### Conditioner: `ImageConditioner`
- `src/models/conditioner/image/ImageConditioner`
- Passes the H&E image as the condition `y` directly

### Trainer: `I2IREPATrainer`
- `src/diffusion/flow_matching/training_i2i_repa.py`
- Losses: FM (flow matching) + REPA alignment (DINOv2 cosine) + LPIPS + encoder perceptual + optional DAB stain-aware
- x-prediction head: model predicts clean `x`, then velocity `v = (pred_x - x_t) / (1 - t).clamp_min(t_eps)`
- Adaptive weight balancing between reconstruction and perceptual losses (after step 10k)

### Sampler: `EulerSamplerJiT`
- `src/diffusion/flow_matching/sampling.py`
- ODE Euler integration, 50 steps
- Also applies x→v conversion: `out = (net_out - x) / (1 - t).clamp_min(t_eps)`

### Scheduler: `LinearScheduler`
- `alpha(t) = t`, `sigma(t) = 1-t`
- Forward: `x_t = t * x_clean + (1-t) * source`
- "source" is either noise (standard) or H&E image (source flow)

---

## 3. Data

### Datasets
| Dataset | Stain | Data Path |
|---|---|---|
| MIST ER | H&E → ER (estrogen receptor IHC) | `/scratch/exp1/a.ayman/diff/data/MIST/ER/TrainValAB` |
| MIST Ki67 | H&E → Ki67 IHC | `/scratch/exp1/a.ayman/diff/data/MIST/Ki67/TrainValAB` |

### Directory Structure
```
TrainValAB/
    trainA/   (H&E images, 1024x1024)
    trainB/   (IHC images, 1024x1024, same filenames as trainA)
    valA/
    valB/
```

### Preprocessing
- Random 512×512 crop from 1024×1024, then resize to 256×256
- Random h-flip, v-flip, 90°/180°/270° rotation (all applied identically to both A and B)
- Normalize to [-1, 1]
- `raw_image` (IHC [0,1]) and `condition_image_raw` (H&E [0,1]) kept in metadata for metrics/DINO

---

## 4. Training Setup

### Launch
```bash
sbatch train_job.sh
# which runs:
python main_i2i.py fit --config configs_i2i/<config>.yaml
```

### SLURM: `train_job.sh`
- Partition: `gpu`, 2 GPUs, 16 CPUs, 24h walltime, account `bcd`
- Conda env: `pixelgen`
- Currently points to: `configs_i2i/mist_er_pre_fm_noise_sourceflow.yaml`

### Standard Training Hyperparameters
| Param | Value |
|---|---|
| Batch size | 8 per GPU × 2 GPUs × 2 accum = effective 32 |
| Optimizer | AdamW, lr=1e-4 (source flow exps: 5e-5), wd=0.01 |
| Max steps | 50,000 |
| Val interval | every 2,000 steps |
| Val batches | 30 |
| Precision | bf16-mixed |
| EMA decay | 0.999 |
| Gradient clip | 1.0 |

### Validation Metrics (logged to W&B)
- **FID** (Fréchet Inception Distance) — primary quality metric
- **SSIM** (structural similarity)
- **PSNR**
- **MAD** (mean absolute deviation)
- 8 visualization samples (H&E | generated | GT IHC) logged as W&B images

---

## 5. The Two Paradigms Being Investigated

### A. Standard Flow (noise → IHC)
```
training:  source = noise,  x_t = t*IHC + (1-t)*noise
inference: start  = noise
```
- Model learns `p(IHC | H&E)` — full distribution
- ✅ Realistic IHC staining, correct colors, FID ≈ 70
- ❌ Pixel-level spatial alignment weaker, SSIM ≈ 0.22

### B. Source Flow (H&E → IHC)
```
training:  source = H&E,  x_t = t*IHC + (1-t)*H&E
inference: start  = H&E
```
- Model learns `E[IHC | H&E]` — conditional mean (deterministic trajectory)
- ✅ Better structural alignment, SSIM ≈ 0.35
- ❌ Under-saturated/washed-out DAB staining, FID ≈ 300

---

## 6. Source Flow Problems & Fixes Applied

### Problem 1: Degenerate [H&E | H&E] input at low-t
At small t, `x_t ≈ H&E`, so the model input is `concat(H&E, H&E)` — identical channels.
→ Model learns to copy H&E structure and do a global color shift (shortcut).

**Fix:** Add noise to the source during training:
```python
# training_i2i_repa.py
source = condition_image + self.source_noise_scale * torch.randn_like(x)
```
Parameter `source_noise_scale` (default `0.0`, backward-compatible).

**Inference must match training:**
```python
# lightning_model_i2i.py predict_step + validation_step
source_noise_scale = getattr(self.diffusion_trainer, "source_noise_scale", 0.0)
start = condition_images + source_noise_scale * torch.randn_like(condition_images)
```

### Problem 2: Lognorm P_mean / uniform scheduler
- Original: `lognorm_t=True, P_mean=-0.8` → `sigmoid(N(-0.8, 0.8))` peaks around t≈0.15–0.35 ✅
- **Do NOT change P_mean to 0.0** — this starves training near t=0, which is where inference starts.
  When P_mean=0.0, the model has almost no training signal at t≈0 → gray outputs.
- Current experiments test `lognorm_t: false` (uniform `t ~ U(0,1)`) to isolate scheduler as a confound.

### Problem 3: Training/inference distribution mismatch (was a bug)
- Training added noise to source, but inference started from pure H&E.
- **Fixed** in `lightning_model_i2i.py` — both `predict_step` and `validation_step` now apply `source_noise_scale`.

---

## 7. Configs & Experiments

### Active MIST ER configs (`configs_i2i/`)

| Config file | W&B run name | Key differences | Status |
|---|---|---|---|
| `mist_er_pre_fmonly.yaml` | `MIST_er_I2I_JiT_XL_pretrained` | Baseline: noise flow, lognorm P_mean=-0.8, LPIPS+DINO | **Best baseline**, FID≈70, SSIM≈0.22 |
| `mist_er_pre_fm_percept_sourceflow.yaml` | `MIST_ER_I2I_XL_pretrained_FMonly_sourceflow` | Source flow, source_noise_scale=0.3, P_mean=-0.8 (reverted), LPIPS+DINO | Running |
| `mist_er_pre_fm_noise_sourceflow.yaml` | `MIST_ER_XL_pretrained_FMonly_noise_sourceflow_uniformNoiseSch0.15` | Source flow, source_noise_scale=0.15, **lognorm_t=false (uniform)**, lr=5e-5 | Running (uniform scheduler ablation) |
| `mist_er_pre_fmonly_sourceflow.yaml` | (earlier) | Source flow without noise fix — **gray output bug** | Abandoned |

### Workdirs (`mist_er_workdirs/`)
```
exp_MIST_er_I2I_JiT_XL_pretrained/              ← baseline
exp_MIST_ER_I2I_XL_pretrained_FMonly_sourceflow/ ← source flow (percept config)
exp_MIST_ER_XL_pretrained_FMonly_noise_sourceflow_uniformNoiseSch/    ← uniform sch, noise_scale=0.3
exp_MIST_ER_XL_pretrained_FMonly_noise_sourceflow_uniformNoiseSch0.15/ ← uniform sch, noise_scale=0.15 (current)
```

### W&B Project
- **Project:** `virtual-stain-er`
- **Entity:** (set in env / config)

### MIST Ki67 configs (`configs_i2i/mist_ki67_*.yaml`)
Ki67 experiments use the same JiT_I2I XL architecture but different data root:
`/scratch/exp1/a.ayman/diff/data/MIST/Ki67/TrainValAB`

Best Ki67 config: `mist_ki67_pretrained_best.yaml` — noise flow, lognorm P_mean=-0.8, LPIPS=0.1, DINO=0.01

---

## 8. Source Code Map

```
src/
├── lightning_model_i2i.py       ← I2ILightningModel: training_step, validation_step, predict_step
│                                   source_flow branching for inference start point
├── lightning_data.py            ← I2IDataModule
├── diffusion/
│   └── flow_matching/
│       ├── training_i2i_repa.py ← I2IREPATrainer: _impl_trainstep, all losses
│       │                           source_flow + source_noise_scale logic here
│       ├── sampling.py          ← EulerSamplerJiT (primary), HeunSamplerJiT, AutoGuidance
│       └── scheduling.py        ← LinearScheduler (alpha=t, sigma=1-t), GVPScheduler
├── models/
│   ├── transformer/JiT_I2I.py  ← Main denoiser: concat(x_t, cond) → pred_x
│   ├── encoder.py               ← DINOv2 wrapper (for REPA + perceptual loss)
│   ├── virchow.py               ← Virchow pathology encoder (alternative to DINO)
│   └── autoencoder/pixel.py    ← PixelAE (identity, no latent compression)
└── data/dataset/mist.py        ← MISTTrainDataset, MISTValDataset
```

---

## 9. Key Hyperparameter Reference

| Param | Location | Meaning |
|---|---|---|
| `source_flow` | config + `I2IREPATrainer` + `I2ILightningModel` | Switch between noise→IHC and H&E→IHC flow |
| `source_noise_scale` | `I2IREPATrainer` + `I2ILightningModel` | Noise added to H&E source (train+inference). 0.0=off, 0.15–0.3=recommended |
| `lognorm_t` | `I2IREPATrainer` | If true: `t ~ sigmoid(N(P_mean, P_std))`. If false: `t ~ U(0,1)` |
| `P_mean` | `I2IREPATrainer` | Lognorm center. **Keep at -0.8**. Setting to 0.0 starves t≈0 → gray output |
| `P_std` | `I2IREPATrainer` | Lognorm spread. Default 0.8 |
| `lpips_weight` | `I2IREPATrainer` | VGG LPIPS loss weight. 0.1 in best configs, 0.0 in ablations |
| `dino_weight` | `I2IREPATrainer` | DINOv2 encoder perceptual loss weight. 0.01 in best configs |
| `feat_loss_weight` | `I2IREPATrainer` | REPA cosine alignment loss weight. 0.3 |
| `percept_t_threshold` | `I2IREPATrainer` | Only apply LPIPS/DINO perceptual loss at t ≥ threshold. 0.3 (noise gates low-t perceptual) |
| `dab_weight` | `I2IREPATrainer` | DAB stain-aware loss weight. 0.0 (disabled), untested >0 |
| `align_layer` | `I2IREPATrainer` | Transformer layer whose features are aligned with DINO. 8 |
| `num_steps` | `EulerSamplerJiT` | ODE integration steps at inference. 50 |
| `timeshift` | sampler + trainer | Respace timesteps: `t' = t/(t+(1-t)*shift)`. 1.0=no shift |

---

## 10. Known Bugs & Lessons Learned

| Bug | Root Cause | Fix |
|---|---|---|
| Gray output with source flow | `P_mean=0.0` starved training near t=0; inference always starts at t=0 | Reverted to `P_mean=-0.8` |
| Gray output with source flow | Inference started from pure H&E but training used `H&E + noise` → distribution mismatch | Both `predict_step` and `validation_step` now add `source_noise_scale * randn` to start |
| Source flow FID≈300 | Deterministic mean-collapse + [H&E\|H&E] shortcut at low-t | `source_noise_scale=0.15–0.3` injects stochasticity and breaks shortcut |
| SSIM and FID anti-correlated | Structural alignment (SSIM) and perceptual realism (FID) have opposing objectives | Use FID as primary metric; SSIM as secondary structure check |

---

## 11. Open Questions / Next Steps

- [ ] Does `source_noise_scale=0.15` (uniform sched) beat the baseline FID≈70?
- [ ] Does `source_noise_scale=0.3` (lognorm P_mean=-0.8) beat baseline?
- [ ] Enable `dab_weight > 0` to explicitly penalize under-staining in source flow
- [ ] Try higher `lpips_weight` (e.g. 0.3) in source flow to combat washed-out DAB
- [ ] Test Virchow pathology encoder instead of DINOv2 for domain-specific perceptual alignment
- [ ] Consider curriculum: start with `source_noise_scale=1.0` (≈noise flow), anneal toward 0.0 over training
