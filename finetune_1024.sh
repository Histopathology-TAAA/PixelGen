#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# StarDiff + PixelGen: 1024×1024 Finetuning from 512 Checkpoint
# ═══════════════════════════════════════════════════════════════
#
# Chain: 256 (pretrained) → 512 (finetuned) → 1024 (this script)
#
# Requirements:
#   - 80GB+ GPU (H100 or A100-80GB)
#   - 512 finetuned checkpoint (.pt file)
#   - Original MIST 1024×1024 data in TrainValAB layout
#
# Data layout expected:
#   ./data/KI67/TrainValAB/trainA/  (H&E 1024×1024)
#   ./data/KI67/TrainValAB/trainB/  (IHC 1024×1024)
#   ./data/KI67/TrainValAB/valA/
#   ./data/KI67/TrainValAB/valB/
#
# Usage:
#   bash finetune_1024.sh path/to/stardiff_512_checkpoint.pt
#   bash finetune_1024.sh wandb_downloads/stardiff_512_stardiff_latest.pt
# ═══════════════════════════════════════════════════════════════

set -e

# ── Arguments ──
CHECKPOINT="${1:?Usage: bash finetune_1024.sh <path/to/512_checkpoint.pt>}"
RESOLUTION="${2:-1024}"
DATASET_ROOT="${3:-./data}"
OUTPUT_DIR="${4:-./outputs_1024}"

echo "═══════════════════════════════════════════════════════"
echo " StarDiff 1024×1024 Finetuning"
echo "═══════════════════════════════════════════════════════"
echo " Checkpoint:  ${CHECKPOINT}"
echo " Resolution:  ${RESOLUTION}×${RESOLUTION}"
echo " Data root:   ${DATASET_ROOT}"
echo " Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════"

# ── Verify checkpoint exists ──
if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Checkpoint not found: ${CHECKPOINT}"
    echo "   Download your 512 checkpoint first (from W&B or local training)"
    exit 1
fi

# ── Verify data exists ──
if [ ! -d "${DATASET_ROOT}/KI67/TrainValAB/trainA" ]; then
    echo "⚠️  MIST 1024 data not found at ${DATASET_ROOT}/KI67/TrainValAB/trainA"
    echo "   Will attempt Kaggle download, or download manually from:"
    echo "   https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p"
fi

# ── Create output directory ──
mkdir -p "${OUTPUT_DIR}"

# ── Run finetuning ──
python -m stardiff_pixelgen.run finetune "${CHECKPOINT}" "${RESOLUTION}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}"
