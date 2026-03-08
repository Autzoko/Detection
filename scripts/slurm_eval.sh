#!/bin/bash
#SBATCH --job-name=nndet_eval
#SBATCH --output=logs/nndet_eval_%j.out
#SBATCH --error=logs/nndet_eval_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

module load anaconda3/2023.03
module load cuda/11.8
conda activate nndet

export det_data=$HOME/scratch/nndet_data
export det_models=$HOME/scratch/nndet_models
export OMP_NUM_THREADS=1

cd $HOME/Detection

MODEL="RetinaUNetV001_D3V001_3d"
FOLD=0

# Step 1: Predict on test set
echo "=== Inference ==="
nndet_predict 100 $MODEL --fold $FOLD --check

# Step 2: Evaluate
echo "=== Evaluation ==="
nndet_eval 100 $MODEL $FOLD --test --boxes --case --analyze_boxes
