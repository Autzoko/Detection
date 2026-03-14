#!/bin/bash
#SBATCH --job-name=nndet_abus_prep
#SBATCH --output=logs/nndet_abus_prep_%j.out
#SBATCH --error=logs/nndet_abus_prep_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00

module load anaconda3
conda activate nndet

# ---- Paths ----
ABUS_SRC=/scratch/$USER/Data/Ultrasound/ABUS
export det_data=/scratch/$USER/Data/Ultrasound/nnDet_ABUS
export OMP_NUM_THREADS=1

mkdir -p $det_data logs

cd /scratch/$USER/Projects/Detection

# ---- Prepare data ----
echo "=== Preparing ABUS data ==="
python scripts/prepare_abus_data.py \
    --abus_root "$ABUS_SRC/data" \
    --output_root "$det_data"

echo "=== Preparation complete ==="
