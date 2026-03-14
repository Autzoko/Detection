#!/bin/bash
#SBATCH --job-name=nndet_abus
#SBATCH --output=logs/nndet_abus_%j.out
#SBATCH --error=logs/nndet_abus_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --time=90:00:00

module load anaconda3
conda activate nndet

# ---- Paths ----
ABUS_SRC=/scratch/$USER/Data/Ultrasound/ABUS
export det_data=/scratch/$USER/Data/Ultrasound/nnDet_ABUS
export det_models=/scratch/$USER/Models/nnDet_ABUS
export OMP_NUM_THREADS=1
export det_num_threads=6

# Fix CUDA stub library issue
export LD_LIBRARY_PATH=/lib64:/share/apps/NYUAD5/cuda/11.8.0/lib:$(echo $LD_LIBRARY_PATH | sed 's|/share/apps/NYUAD5/cuda/11.8.0/lib/stubs:||g; s|/share/apps/NYUAD5/cuda/11.8.0/lib:||g')

mkdir -p $det_data $det_models logs

cd /scratch/$USER/Projects/Detection

# ---- Step 1: Prepare data ----
echo "=== Step 1: Preparing ABUS data ==="
python scripts/prepare_abus_data.py \
    --abus_root "$ABUS_SRC/data" \
    --output_root "$det_data"

# ---- Step 2: Preprocess ----
echo "=== Step 2: Preprocessing ==="
nndet_prep 200

# ---- Step 3: Train ----
echo "=== Step 3: Training fold 0 ==="
nndet_train 200 -o exp.fold=0

echo "=== Done ==="
