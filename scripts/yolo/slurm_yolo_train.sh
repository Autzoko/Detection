#!/bin/bash
#SBATCH --job-name=yolo_abus
#SBATCH --output=logs/yolo_train_%j.out
#SBATCH --error=logs/yolo_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load anaconda3
conda activate nndet

# Fix CUDA stub library issue
export LD_LIBRARY_PATH=/lib64:/share/apps/NYUAD5/cuda/11.8.0/lib:$(echo $LD_LIBRARY_PATH | sed 's|/share/apps/NYUAD5/cuda/11.8.0/lib/stubs:||g; s|/share/apps/NYUAD5/cuda/11.8.0/lib:||g')

mkdir -p logs

# Install ultralytics if not already installed
pip install ultralytics --quiet

# ============================================================
# Step 1: Convert shards to YOLO format (if not already done)
# ============================================================
SHARD_DIR="/scratch/$USER/Data/Ultrasound/Shards"
YOLO_DATASET="/scratch/$USER/Data/Ultrasound/yolo_dataset"

if [ ! -d "$YOLO_DATASET/images/train" ]; then
    echo "=== Converting shards to YOLO format ==="
    python /scratch/$USER/Projects/Detection/scripts/yolo/convert_shards_to_yolo.py \
        --shard_dir "$SHARD_DIR" \
        --output_dir "$YOLO_DATASET" \
        --datasets BIrads Class2 Class3 Class4 Abus \
        --val_fraction 0.15 \
        --neg_ratio 0.3 \
        --seed 42
else
    echo "=== YOLO dataset already exists, skipping conversion ==="
fi

# ============================================================
# Step 2: Create dataset YAML with correct HPC paths
# ============================================================
cat > /tmp/abus_lesion_hpc.yaml << EOF
path: $YOLO_DATASET
train: images/train
val: images/val
nc: 1
names:
  0: lesion
EOF

# ============================================================
# Step 3: Train YOLOv8
# ============================================================
echo "=== Training YOLOv8 ==="
cd /scratch/$USER/Projects/Detection

python scripts/yolo/train_yolo_abus.py \
    --data /tmp/abus_lesion_hpc.yaml \
    --model yolov8m.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 32 \
    --device 0 \
    --workers 8 \
    --patience 30 \
    --name abus_lesion_v1 \
    --project /scratch/$USER/Models/yolo \
    --cache \
    --cos_lr

echo "=== Training complete ==="

# ============================================================
# Step 4: Run inference on test volumes
# ============================================================
BEST_MODEL="/scratch/$USER/Models/yolo/abus_lesion_v1/weights/best.pt"
TEST_IMAGES="/scratch/$USER/Data/Ultrasound/nnDet/Task100_BreastABUS/raw_splitted/imagesTs"
TEST_LABELS="/scratch/$USER/Data/Ultrasound/nnDet/Task100_BreastABUS/raw_splitted/labelsTs"
YOLO_PREDS="/scratch/$USER/Data/Ultrasound/nnDet/yolo_predictions"

if [ -f "$BEST_MODEL" ]; then
    echo "=== Running 3D inference and evaluation ==="
    python scripts/yolo/predict_and_reconstruct_3d.py \
        --model "$BEST_MODEL" \
        --image_dir "$TEST_IMAGES" \
        --label_dir "$TEST_LABELS" \
        --output_dir "$YOLO_PREDS" \
        --conf 0.3 \
        --min_slices 3 \
        --max_gap 2 \
        --imgsz 640 \
        --batch_size 64
    echo "=== Inference and evaluation complete ==="
else
    echo "WARNING: Best model not found at $BEST_MODEL"
fi
