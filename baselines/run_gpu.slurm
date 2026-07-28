#!/bin/bash
# ============================================================================
# run_gpu.slurm — SLURM job script for GPU-accelerated GNN training
#
# Submits the full reproducibility pipeline with GPU acceleration for
# GCN/GraphSAGE training (Tasks 2 and cross-dataset transfer).
#
# Usage:
#   sbatch run_gpu.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm-*.out
# ============================================================================
#SBATCH --job-name=resplan-reproduce
#SBATCH --output=resplan-reproduce-%j.out
#SBATCH --error=resplan-reproduce-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Date: $(date)"
echo ""

# Load modules if needed (uncomment/modify for your cluster)
# module load cuda/12.1
# module load python/3.10

# Activate conda/venv if needed
# source ~/miniforge3/bin/activate resplan

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

# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

# Run the full pipeline (auto-detects GPU)
bash "$SCRIPT_DIR/reproduce.sh"

echo ""
echo "Job completed at $(date)"
