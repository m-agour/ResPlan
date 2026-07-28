#!/bin/bash
#SBATCH --job-name=leakage-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=leakage-eval-%j.out
#SBATCH --error=leakage-eval-%j.err

set -e
cd /path/to/resplan/resplan_release/baselines

echo "=== Task 1: modern graph architectures ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

python3 task1_leakage_eval.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device cuda \
    --seeds 42 123 7 \
    --epochs 500 \
    --out results/task1_leakage_eval.json

echo "End: $(date)"
