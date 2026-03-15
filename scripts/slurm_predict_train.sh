#!/bin/bash
#SBATCH --job-name=nndet_pred_train
#SBATCH --output=logs/nndet_pred_train_%j.out
#SBATCH --error=logs/nndet_pred_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00

module purge
module load cuda/11.8.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/$USER/envs/nndet

# ---- Paths ----
export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export OMP_NUM_THREADS=1

# Fix CUDA stub library issue
export LD_LIBRARY_PATH=/lib64:/share/apps/NYUAD5/cuda/11.8.0/lib:$(echo $LD_LIBRARY_PATH | sed 's|/share/apps/NYUAD5/cuda/11.8.0/lib/stubs:||g; s|/share/apps/NYUAD5/cuda/11.8.0/lib:||g')

mkdir -p logs

cd /scratch/$USER/Projects/Detection

echo "=== Predicting on training set ==="
python scripts/predict_train.py 100 RetinaUNetV001_D3V001_3d --fold 0 --split all -ntta 1

echo "=== Done ==="
