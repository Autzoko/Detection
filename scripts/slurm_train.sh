#!/bin/bash
#SBATCH --job-name=nndet_breast
#SBATCH --output=logs/nndet_train_%j.out
#SBATCH --error=logs/nndet_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module load anaconda3
module load cuda/11.2.67
conda activate nndet

export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export OMP_NUM_THREADS=1
export det_num_threads=6

mkdir -p $det_models logs

cd /scratch/$USER/Projects/Detection

# Step 1: Preprocessing
echo "=== Preprocessing ==="
nndet_prep 100 --full_check

# Step 2: Training
echo "=== Training fold 0 ==="
nndet_train 100 --fold 0
