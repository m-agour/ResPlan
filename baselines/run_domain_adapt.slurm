#!/bin/bash
#SBATCH --job-name=domain-adapt
#SBATCH --output=domain-adapt-%j.out
#SBATCH --error=domain-adapt-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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
"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
echo "Working dir: $(pwd)"

export PYTHONUNBUFFERED=1

python3 task2_domain_adapt.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --rplan ../../rplan_data/Interface/static/Data/data_train_converted.pkl \
    --device auto \
    --pretrain-epochs 300 \
    --finetune-epochs 200 \
    --seeds 42 123 7 \
    --output results/task2_domain_adapt.json

echo ""
echo "Job completed at $(date)"
