#!/bin/bash
#SBATCH --job-name=table4-rerun
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=table4-rerun-%j.out
#SBATCH --error=table4-rerun-%j.err
set -e
cd /path/to/resplan/resplan_release/baselines
echo "=== Table 4 re-run on the RELEASED artifact ==="
python3 task2_baselines.py --data ../ResPlan.pkl --split ../split.json \
    --device cuda --seeds 42 123 7 --epochs 500 \
    --output results/task1_table4_released_artifact.json
echo "=== Ablations (Tables 6+7) on the RELEASED artifact ==="
python3 task2_ablation.py --data ../ResPlan.pkl --split ../split.json \
    --device cuda --seeds 42 123 7 \
    --out results/task1_ablation_released_artifact.json 2>/dev/null || \
python3 task2_ablation.py --data ../ResPlan.pkl --split ../split.json \
    --device cuda --seeds 42 123 7
echo "Done: $(date)"
