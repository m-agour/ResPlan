#!/bin/bash
#SBATCH --job-name=repro-diag
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=repro-diag-%j.out
#SBATCH --error=repro-diag-%j.err
set -e
cd /path/to/resplan/resplan_release/baselines
for CFG in "../../dataset/ResPlan_with_aug_fields.pkl ../../dataset/split.json MARCH_DATA_MARCH_SPLIT" \
           "../ResPlan.pkl ../split.json RELEASED_DATA_RELEASED_SPLIT" \
           "../../dataset/ResPlan_with_aug_fields.pkl ../split.json MARCH_DATA_RELEASED_SPLIT"; do
  set -- $CFG
  echo "=== $3 ==="
  python3 task1_leakage_eval.py --data "$1" --split "$2" \
      --leaked ../../leaked_test_ids.json --device cuda --seeds 42 \
      --out "results/repro_diag_$3.json" 2>&1 | grep -E "train=|seed=|Error|Traceback" || true
done
