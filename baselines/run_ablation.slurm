#!/bin/bash
# ============================================================================
# run_ablation.slurm — Task 2 feature & architecture ablation on GPU
#
# Runs:
#   Part A: 5 feature subsets × 3 seeds = 15 GraphSAGE trainings
#   Part B: 4 layers + 4 hidden dims + 4 architectures × 3 seeds = 36 trainings
#   Total: ~51 GNN runs, ~500 epochs each
#
# Usage:
#   cd /path/to/resplan/resplan_release/baselines
#   sbatch run_ablation.slurm
# ============================================================================
#SBATCH --job-name=ablation
#SBATCH --output=ablation-%j.out
#SBATCH --error=ablation-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
echo "Working dir: $(pwd)"

export PYTHONUNBUFFERED=1

python3 task2_ablation.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device auto \
    --seeds 42 123 7 \
    --epochs 500 \
    --part AB \
    --output results/task2_ablation.json

echo ""
echo "Job completed at $(date)"
