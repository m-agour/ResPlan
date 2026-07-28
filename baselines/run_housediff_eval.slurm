#!/bin/bash
#SBATCH --job-name=housediff-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=housediff-eval-%j.out
#SBATCH --error=housediff-eval-%j.err

# ─── HouseDiffusion Evaluation on ResPlan ────────────────────────────────────
# Samples from trained model and computes FID + graph compatibility.
# ─────────────────────────────────────────────────────────────────────────────

set -e

REPO_DIR="/path/to/resplan/house_diffusion"
SCRIPTS_DIR="${REPO_DIR}/scripts"
LOG_DIR="${REPO_DIR}/logs_resplan"

export OPENAI_LOGDIR="${LOG_DIR}"

cd "${SCRIPTS_DIR}"
[ -L processed_rplan ] || ln -sf ../processed_rplan .

# Find the latest checkpoint
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls -t ${LOG_DIR}/model*.pt 2>/dev/null | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: No checkpoint found in ${LOG_DIR}"
        exit 1
    fi
fi

echo "=== HouseDiffusion Evaluation on ResPlan ==="
echo "Start time: $(date)"
echo "Model: ${MODEL_PATH}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Clean previous outputs
rm -rf outputs/pred outputs/gt outputs/gif outputs/graphs_gt outputs/graphs_pred

MODEL_FLAGS="--dataset rplan --batch_size 16 --set_name eval --target_set 0"
SAMPLE_FLAGS="--num_samples 1699 --model_path ${MODEL_PATH} --draw_graph False --save_svg False"

python image_sample.py ${MODEL_FLAGS} ${SAMPLE_FLAGS}

echo ""
echo "=== Evaluation Complete ==="
echo "End time: $(date)"
