#!/bin/bash
#SBATCH --job-name=nndet_breast
#SBATCH --output=logs/nndet_train_%j.out
#SBATCH --error=logs/nndet_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module load anaconda3/2023.03
module load cuda/11.8
conda activate nndet

export det_data=$HOME/scratch/nndet_data
export det_models=$HOME/scratch/nndet_models
export OMP_NUM_THREADS=1

mkdir -p $det_models logs

cd $HOME/Detection

# Step 1: Preprocessing
echo "=== Preprocessing ==="
nndet_prep 100 --full_check

# Step 2: Training
echo "=== Training fold 0 ==="
nndet_train 100 --fold 0
