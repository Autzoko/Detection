#!/bin/bash
#SBATCH --job-name=birads_predict
#SBATCH --output=logs/birads_predict_%j.out
#SBATCH --error=logs/birads_predict_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

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
git pull

# ---- Clean old checkpoints (GAP-based, incompatible) ----
MODEL_DIR=/scratch/$USER/Models/nnDet/Task101_BreastBIRADS/RetinaUNetV003_D3V001_3d/fold0
rm -f "$MODEL_DIR"/model_best-v*.ckpt
echo "Kept checkpoints:"
ls -lt "$MODEL_DIR"/*.ckpt

# ---- Generate plan_inference.pkl ----
if [ ! -f "$MODEL_DIR/plan_inference.pkl" ]; then
    echo "Generating plan_inference.pkl..."
    python3 -c "
import pickle
with open('$MODEL_DIR/plan.pkl','rb') as f: plan=pickle.load(f)
plan['inference_plan']={'model_iou':0.1,'model_score_thresh':0.0,'model_topk':1000,'model_detections_per_image':100,'ensemble_iou':0.5,'ensemble_topk':1000,'remove_small_boxes':0.01,'ensemble_score_thresh':0.0}
with open('$MODEL_DIR/plan_inference.pkl','wb') as f: pickle.dump(plan,f)
print('Done')
"
fi

# ---- Run inference ----
echo "=== BI-RADS Prediction ==="
nndet_predict Task101_BreastBIRADS RetinaUNetV003_D3V001_3d -f 0 -ntta 1

echo "=== Done ==="
