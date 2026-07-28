#!/bin/bash
#SBATCH --job-name=verify-pkl
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=verify-pkl-%j.out
#SBATCH --error=verify-pkl-%j.err
cd /path/to/resplan/resplan_release/baselines
PKL=../../dataset/ResPlan_v3_augmented.pkl
echo "### Table 4 on ResPlan_v3_augmented.pkl (the pickle the paper describes) ###"
python3 task2_baselines.py --data $PKL --split ../split.json --device cuda \
    --seeds 42 123 7 --epochs 500 --output results/verify_table4_augmented.json 2>&1 | tail -25
echo "### Feature ablation (does the 2.3pp typed-edge gap return?) ###"
python3 task2_ablation.py --data $PKL --split ../split.json --device cuda \
    --seeds 42 123 7 --part A 2>&1 | tail -30
echo "### DONE $(date) ###"
