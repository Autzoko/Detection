#!/usr/bin/env bash
# Evaluate nnDetection on Breast ABUS test set (Task100_BreastABUS)
#
# Usage:
#   bash scripts/eval_breast.sh [MODEL_NAME] [FOLD]
#
# MODEL_NAME defaults to RetinaUNetV001_D3V001_3d (nnDetection's default 3D model).
# FOLD defaults to 0. Use -1 for consolidated model.
#
set -euo pipefail

# ---- Configuration ----
TASK_ID=100
TASK_NAME="Task100_BreastABUS"
MODEL="${1:-RetinaUNetV001_D3V001_3d}"
FOLD="${2:-0}"

export det_data="${det_data:-/Users/langtian/nndet_workspace/data}"
export det_models="${det_models:-/Users/langtian/nndet_workspace/models}"
export OMP_NUM_THREADS=1

echo "============================================="
echo " nnDetection – Breast ABUS Evaluation"
echo "============================================="
echo "  det_data   = ${det_data}"
echo "  det_models = ${det_models}"
echo "  Task       = ${TASK_NAME}"
echo "  Model      = ${MODEL}"
echo "  Fold       = ${FOLD}"
echo "============================================="

# ---- Step 1: Predict on test set ----
echo ""
echo "[Step 1/2] Running inference on test set..."
nndet_predict "${TASK_ID}" "${MODEL}" \
    --fold "${FOLD}" \
    --check

# ---- Step 2: Evaluate (FROC / mAP) ----
echo ""
echo "[Step 2/2] Evaluating predictions (boxes + case-level)..."
nndet_eval "${TASK_ID}" "${MODEL}" "${FOLD}" \
    --test \
    --boxes \
    --case \
    --analyze_boxes

RESULTS_DIR="${det_models}/${TASK_NAME}/${MODEL}/fold${FOLD}"
echo ""
echo "============================================="
echo " Evaluation complete!"
echo " Results: ${RESULTS_DIR}/test_predictions/"
echo "============================================="
