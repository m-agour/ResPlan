#!/bin/bash
#SBATCH --job-name=hd-metrics
#SBATCH --output=/path/to/resplan/resplan_release/baselines/housediff-metrics-%j.out
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --account=YOUR_ACCOUNT

export PYTHONUNBUFFERED=1

echo "=== HouseDiffusion Task-1 Metrics ==="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /path/to/resplan/house_diffusion/scripts

python /path/to/resplan/resplan_release/baselines/housediff_task1_metrics.py \
    --model_path /path/to/resplan/house_diffusion/logs_resplan/ema_0.9999_200000.pt \
    --dataset rplan \
    --set_name eval \
    --target_set 0 \
    --batch_size 16

echo ""
echo "=== Evaluation Complete ==="
echo "End time: $(date)"
