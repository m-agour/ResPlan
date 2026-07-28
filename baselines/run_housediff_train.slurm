#!/bin/bash
#SBATCH --job-name=housediff-resplan
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=housediff-train-%j.out
#SBATCH --error=housediff-train-%j.err

# ─── HouseDiffusion Training on ResPlan ─────────────────────────────────────
# Trains the HouseDiffusion (CVPR 2023) generative model on ResPlan data.
# Checkpoints saved to OPENAI_LOGDIR every save_interval steps.
# ─────────────────────────────────────────────────────────────────────────────

set -e

# Setup
REPO_DIR="/path/to/resplan/house_diffusion"
SCRIPTS_DIR="${REPO_DIR}/scripts"
LOG_DIR="${REPO_DIR}/logs_resplan"

export OPENAI_LOGDIR="${LOG_DIR}"
mkdir -p "${LOG_DIR}"

# Ensure processed_rplan symlink exists
cd "${SCRIPTS_DIR}"
[ -L processed_rplan ] || ln -sf ../processed_rplan .

echo "=== HouseDiffusion Training on ResPlan ==="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Log dir: ${LOG_DIR}"
echo ""

# Training config
# - batch_size 32 (200-point sequences → 4× memory vs RPLAN's 100-point)
# - target_set=0 (all plans, no room-count filtering)
# - lr 1e-4 (default from paper)
# - save_interval 10000 (checkpoint every 10k steps)
# - cosine noise schedule
MODEL_FLAGS="--dataset rplan --batch_size 32 --set_name train --target_set 0"
TRAIN_FLAGS="--lr 1e-4 --save_interval 10000 --weight_decay 0.05 --log_interval 100"

python image_train.py ${MODEL_FLAGS} ${TRAIN_FLAGS}

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Checkpoints saved to: ${LOG_DIR}"
