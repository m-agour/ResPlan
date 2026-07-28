#!/bin/bash
# ============================================================================
# run_task1_gen.slurm — SLURM job for Task 1 learned generative baselines
#
# Trains CVAE and CVAE+Graph models for room layout generation on GPU.
#
# Usage:
#   cd /path/to/resplan/resplan_release/baselines
#   sbatch run_task1_gen.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f task1-gen-*.out
# ============================================================================
#SBATCH --job-name=task1-gen
#SBATCH --output=task1-gen-%j.out
#SBATCH --error=task1-gen-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Verify GPU
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
echo "Working dir: $(pwd)"

export PYTHONUNBUFFERED=1

echo ""
echo "========================================"
echo "Task 1: Learned Generative Baselines"
echo "========================================"

python3 task1_generation.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device auto \
    --epochs 800 \
    --batch 256 \
    --lr 1e-3 \
    --beta 0.1 \
    --warmup 50 \
    --latent 128 \
    --hidden 512 \
    --patience 150 \
    --samples 10 \
    --seed 42 \
    --output results/task1_generation.json

echo ""
echo "Job completed at $(date)"
