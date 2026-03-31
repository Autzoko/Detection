#!/bin/bash
#SBATCH --job-name=birads_train
#SBATCH --output=logs/birads_train_%j.out
#SBATCH --error=logs/birads_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00

module purge
module load cuda/11.8.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/$USER/envs/nndet

# ---- Paths ----
export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export det_num_threads=6
export OMP_NUM_THREADS=1

# Fix CUDA stub library issue
export LD_LIBRARY_PATH=/lib64:/share/apps/NYUAD5/cuda/11.8.0/lib:$(echo $LD_LIBRARY_PATH | sed 's|/share/apps/NYUAD5/cuda/11.8.0/lib/stubs:||g; s|/share/apps/NYUAD5/cuda/11.8.0/lib:||g')

mkdir -p logs

cd /scratch/$USER/Projects/Detection
git pull

# ---- Train V001 with 3-class detection (BI-RADS 2/3/4) ----
# classifier_classes=3 is auto-derived from dataset.json labels
# PatchClassifier is included for FP suppression
echo "=== Training BI-RADS multi-class detection (V001) ==="
nndet_train Task101_BreastBIRADS

echo "=== Done ==="
