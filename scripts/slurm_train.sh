#!/bin/bash
#SBATCH --job-name=nndet_breast
#SBATCH --output=logs/nndet_train_%j.out
#SBATCH --error=logs/nndet_train_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --time=90:00:00

module load anaconda3
conda activate nndet

export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export OMP_NUM_THREADS=1
export det_num_threads=6

# Fix CUDA stub library issue
export LD_LIBRARY_PATH=/lib64:/share/apps/NYUAD5/cuda/11.8.0/lib:$(echo $LD_LIBRARY_PATH | sed 's|/share/apps/NYUAD5/cuda/11.8.0/lib/stubs:||g; s|/share/apps/NYUAD5/cuda/11.8.0/lib:||g')

mkdir -p $det_models logs

cd /scratch/$USER/Projects/Detection

echo "=== Training fold 0 ==="
nndet_train 100 -o exp.fold=0
