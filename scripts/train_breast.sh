#!/usr/bin/env bash
# Train nnDetection on Breast ABUS data (Task100_BreastABUS)
#
# Usage:
#   bash scripts/train_breast.sh [FOLD]
#
# Prerequisites:
#   1. Run scripts/prepare_breast_data.py first
#   2. Activate the nndet conda environment
#
set -euo pipefail

# ---- Configuration ----
TASK_ID=100
TASK_NAME="Task100_BreastABUS"
FOLD="${1:-0}"

# Paths – adjust these to your setup
export det_data="${det_data:-/Users/langtian/nndet_workspace/data}"
export det_models="${det_models:-/Users/langtian/nndet_workspace/models}"
export OMP_NUM_THREADS=1

echo "============================================="
echo " nnDetection – Breast ABUS Training Pipeline"
echo "============================================="
echo "  det_data   = ${det_data}"
echo "  det_models = ${det_models}"
echo "  Task       = ${TASK_NAME}"
echo "  Fold       = ${FOLD}"
echo "============================================="

# ---- Step 1: Preprocessing ----
echo ""
echo "[Step 1/2] Preprocessing (planning + cropping)..."
nndet_prep "${TASK_ID}" --full_check

# ---- Step 2: Training ----
echo ""
echo "[Step 2/2] Training fold ${FOLD}..."
nndet_train "${TASK_ID}" --fold "${FOLD}"

echo ""
echo "Training complete!"
echo "Models saved to: ${det_models}/${TASK_NAME}"
