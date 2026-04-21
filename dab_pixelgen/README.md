# DABPixelGen Run Guide

This directory contains a single-backbone DAB-first pipeline for H&E -> IHC generation.

## 1) Environment Setup

Create a `.env` file (either at repo root or inside `dab_pixelgen/`) from the template:

```bash
cp dab_pixelgen/.env.example .env
```

Fill in the values:

- `HF_TOKEN` for Hugging Face downloads (optional, useful for private/rate-limited access)
- `WANDB_API_KEY` for Weights & Biases logging
- `KAGGLE_USERNAME` for Kaggle dataset download

`dab_pixelgen/run.py` loads `.env` automatically at startup if `python-dotenv` is installed.

## 2) Train (fresh)

From repo root:

```bash
uv run python -m dab_pixelgen.run
```

With overrides:

```bash
uv run python -m dab_pixelgen.run --dataset_root ./data --output_dir ./output --batch_size 8
```

## 3) Finetune 512

```bash
uv run python -m dab_pixelgen.run finetune path/to/dab_pixelgen_best.pt 512
```

## 4) Finetune 1024

```bash
uv run python -m dab_pixelgen.run finetune path/to/dab_pixelgen_512.pt 1024
```

## Notes

- CLI format is `--key value`.
- `run.py` supports both root `.env` and `dab_pixelgen/.env`.
- Keep `.env` out of version control.
