# Breast ABUS 3D Lesion Detection with nnDetection

End-to-end guide for preparing data, training, and evaluating a 3D breast
ultrasound lesion detector using nnDetection on **NYU Jubail HPC**.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Setup on Jubail](#repository-setup-on-jubail)
3. [Environment Setup](#environment-setup)
4. [Data Preparation (Local)](#data-preparation-local)
5. [Transfer Data to HPC](#transfer-data-to-hpc)
6. [Training on HPC](#training-on-hpc)
7. [Evaluation on HPC](#evaluation-on-hpc)
8. [Monitoring and Results](#monitoring-and-results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**Task**: Detect lesions in 3D Automated Breast Ultrasound (ABUS) volumes.

**Data sources** (on external drive):

| Folder | Role | Format |
|--------|------|--------|
| `2类/`, `3类/`, `4类/` | Train+Val (BI-RADS 2/3/4) | `.nii` + `_nii_Label.tar` |
| `已标注及BI-rads分类20260123/` | Test set | `.nii` + `_nii_Label.tar` + Excel |
| `度影AI数据/` | Skipped (no annotations) | `.ai` raw binary |

**Pipeline**:
1. `prepare_breast_data.py` — discovers pairs, deduplicates, converts 2D
   bounding-box annotations → 3D ellipsoid instance masks, writes nnDetection
   format
2. `nndet_prep` — nnDetection preprocessing (planning, cropping, resampling)
3. `nndet_train` — training (RetinaUNet)
4. `nndet_predict` + `nndet_eval` — inference and FROC/mAP evaluation

---

## Repository Setup on Jubail

```bash
# SSH into Jubail
ssh <netid>@jubail.abudhabi.nyu.edu

# Clone the repo
cd ~
git clone https://github.com/Autzoko/Detection.git
cd Detection
```

---

## Environment Setup

> **Important**: nnDetection requires **PyTorch 1.x** (not 2.0+). PyTorch 2.0
> removed `torch._six` and other internals that nnDetection depends on.

### Step 1: Request an interactive GPU node (for compilation)

Building nnDetection compiles C++/CUDA extensions, which needs a GPU. Either
request an interactive session or run the install inside a SLURM job:

```bash
# Interactive session with GPU
srun --partition=nvidia --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
```

### Step 2: Load modules

```bash
module load anaconda3
module load cuda/11.2.67
```

### Step 3: Create conda environment and install

```bash
# Create env with Python 3.8 (as specified in nnDetection README)
conda create -n nndet python=3.8 -y
conda activate nndet

# Install GCC toolchain (required for C++/CUDA extension compilation)
conda install gxx_linux-64==9.3.0 -y

# Set compilers to conda-provided versions
export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

# Install PyTorch 1.11 + CUDA 11.3 (compatible with system CUDA 11.2)
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# Clone and install nnDetection
cd /scratch/$USER/Projects
git clone https://github.com/Autzoko/Detection.git
cd Detection

pip install -r requirements.txt
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

# Build with CUDA extensions (GPU must be available)
pip install -v -e .

# Extra dependencies for data preparation
pip install openpyxl scikit-learn
```

### Step 4: Set environment variables

Add these to your `~/.bashrc` so they persist across sessions:

```bash
cat >> ~/.bashrc << 'EOF'
# nnDetection environment
export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export OMP_NUM_THREADS=1
export det_num_threads=6
export det_verbose=1
EOF

source ~/.bashrc
mkdir -p $det_data $det_models
```

### Step 5: Verify installation

```bash
conda activate nndet

# Check CUDA extension compilation
python -c "import torch; import nndet._C; import nndet; print('OK')"

# Check environment paths
nndet_env
```

If the `import nndet._C` step fails, the CUDA extensions did not compile.
Make sure you ran `pip install -v -e .` on a GPU node (not a login node).

### Troubleshooting installation

| Problem | Fix |
|---------|-----|
| `No module named 'torch._six'` | You installed PyTorch 2.0+. Downgrade: `conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch` |
| `nvcc not found` during build | `module load cuda/11.2.67` before running `pip install -v -e .` |
| `nndet._C` import fails | Rebuild on a GPU node: `pip install -v -e .` (not login node) |
| GCC version errors | Make sure `conda install gxx_linux-64==9.3.0` and the `CC`/`CXX` exports are set |
| `SimpleITK` version conflict | `pip install 'SimpleITK<2.1.0'` |

---

## Data Preparation (Local)

Run this on your **local machine** where the external drive is mounted.

### Step 1: Run the preparation script

```bash
conda activate nndet

# Set the output directory (local staging area)
export det_data="/Volumes/Lang/Research/Data/3D Ultrasound/nnDet"

python scripts/prepare_breast_data.py \
    --det_data "$det_data" \
    --data_root "/Volumes/Autzoko/Dataset/third-party/data"
```

This will:
- Find all `.nii` + `_nii_Label.tar` pairs in `2类/`, `3类/`, `4类/`, and the
  test set
- Deduplicate by MD5 hash (prefers test-set copies)
- Convert 2D bounding-box annotations → 3D ellipsoid instance segmentation masks
- Split: test-set → test, 15% of rest → val (stratified by BI-RADS class),
  remainder → train
- Write everything to `$det_data/Task100_BreastABUS/`

### Step 2: Verify output

```bash
# Check the output structure
ls "$det_data/Task100_BreastABUS/"
# Expected: dataset.json  dataset_statistics.csv  raw_splitted/

ls "$det_data/Task100_BreastABUS/raw_splitted/"
# Expected: imagesTr/  imagesTs/  labelsTr/  labelsTs/

# Check dataset.json
cat "$det_data/Task100_BreastABUS/dataset.json"

# Check statistics
head -5 "$det_data/Task100_BreastABUS/dataset_statistics.csv"
```

### Step 3: Verify sample mapping

The file `dataset_statistics.csv` maps every renamed case back to its original
source:

| Column | Description |
|--------|-------------|
| `volume_id` | New name, e.g. `case_00042` |
| `source_folder` | Original folder (`2类`, `3类`, `4类`, or test set) |
| `image_path` | Full original path on disk |
| `mask_path` | Original label tar path |
| `shape` | Volume dimensions |
| `spacing` | Voxel spacing |
| `num_lesions` | Number of annotated lesions |
| `lesion_class` | BI-RADS class (birads2/3/4 or test_mixed) |
| `split` | train / val / test |
| `is_duplicate` | Whether this was a duplicate (excluded entries) |

---

## Transfer Data to HPC

```bash
# From your local machine, upload the prepared Task directory to Jubail
# Using rsync (recommended for large transfers)

rsync -avhP \
    "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Task100_BreastABUS" \
    <netid>@jubail.abudhabi.nyu.edu:~/scratch/nndet_data/

# Or using scp
scp -r \
    "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Task100_BreastABUS" \
    <netid>@jubail.abudhabi.nyu.edu:~/scratch/nndet_data/
```

### Verify on HPC

```bash
ssh <netid>@jubail.abudhabi.nyu.edu
ls ~/scratch/nndet_data/Task100_BreastABUS/
# Should see: dataset.json  dataset_statistics.csv  raw_splitted/
```

---

## Training on HPC

### SLURM Job Script

Create or use the provided `scripts/slurm_train.sh`:

```bash
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
```

### Submit the job

```bash
cd /scratch/$USER/Projects/Detection
mkdir -p logs
sbatch scripts/slurm_train.sh
```

### Monitor

```bash
# Check job status
squeue -u $USER

# Watch training log in real time
tail -f logs/nndet_train_<jobid>.out
```

---

## Evaluation on HPC

### SLURM Job Script

Create or use the provided `scripts/slurm_eval.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=nndet_eval
#SBATCH --output=logs/nndet_eval_%j.out
#SBATCH --error=logs/nndet_eval_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

module load anaconda3
module load cuda/11.2.67
conda activate nndet

export det_data=/scratch/$USER/Data/Ultrasound/nnDet
export det_models=/scratch/$USER/Models/nnDet
export OMP_NUM_THREADS=1

cd /scratch/$USER/Projects/Detection

# Find the model name from the training output
MODEL="RetinaUNetV001_D3V001_3d"
FOLD=0

# Step 1: Predict on test set
echo "=== Inference ==="
nndet_predict 100 $MODEL --fold $FOLD --check

# Step 2: Evaluate
echo "=== Evaluation ==="
nndet_eval 100 $MODEL $FOLD --test --boxes --case --analyze_boxes
```

### Submit

```bash
sbatch scripts/slurm_eval.sh
```

---

## Monitoring and Results

### Training metrics

nnDetection uses MLflow for experiment tracking. After training:

```bash
# Start MLflow UI (forward port via SSH)
# On HPC:
cd $HOME/scratch/nndet_models/Task100_BreastABUS
mlflow ui --port 5000

# On local machine (SSH tunnel):
ssh -L 5000:localhost:5000 <netid>@jubail.abudhabi.nyu.edu
# Then open http://localhost:5000 in your browser
```

### Evaluation results

Results are saved under the model directory:

```bash
ls $det_models/Task100_BreastABUS/RetinaUNetV001_D3V001_3d/fold0/test_predictions/

# Key output files:
#   - results_boxes.json     → FROC and mAP metrics
#   - results_case.json      → per-case detection results
#   - analysis/              → detailed box analysis plots
```

### Download results to local machine

```bash
# From local machine:
rsync -avhP \
    <netid>@jubail.abudhabi.nyu.edu:~/scratch/nndet_models/Task100_BreastABUS/ \
    ./results/Task100_BreastABUS/
```

---

## Troubleshooting

### Common issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size: add `overwrites='train.batch_size=2'` to `nndet_train` |
| `SimpleITK version conflict` | `pip install SimpleITK==2.0.2` |
| `ModuleNotFoundError: nndet` | Re-run `pip install -e .` from the Detection directory |
| `No space left on device` | Use `$SCRATCH` instead of `$HOME` for data/models |
| `nndet_prep fails with dots in path` | Ensure no dots (`.`) in directory names along the data path |

### Adjusting GPU memory

If the default model is too large for your GPU, you can override settings:

```bash
# Train with smaller batch size and patch size
nndet_train 100 --fold 0 \
    overwrites='train.batch_size=2'
```

### Multi-GPU training

```bash
# In your SLURM script, request multiple GPUs:
#SBATCH --gres=gpu:2

# nnDetection uses PyTorch Lightning, which auto-detects multiple GPUs
nndet_train 100 --fold 0
```

### Check environment

```bash
nndet_env
# Should print det_data, det_models paths and confirm they exist
```
