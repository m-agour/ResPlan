#!/bin/bash
#SBATCH --job-name=full-correct
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=full-correct-%j.out
#SBATCH --error=full-correct-%j.err
cd /path/to/resplan/resplan_release/baselines
PKL=../../dataset/ResPlan_v3_augmented.pkl
SPL=../split.json
echo "### MODERN ARCHITECTURES ###"
python3 task1_modern_arch.py --data $PKL --split $SPL --device cuda --seeds 42 123 7 \
    --epochs 500 --out results/CORRECT_modern_arch.json 2>&1 | tail -20
echo "### LEAKAGE ###"
python3 task1_leakage_eval.py --data $PKL --split $SPL --leaked ../../leaked_test_ids.json \
    --device cuda --seeds 42 123 7 --out results/CORRECT_leakage.json 2>&1 | tail -12
echo "### ARCH ABLATION (part B) ###"
python3 task2_ablation.py --data $PKL --split $SPL --device cuda --seeds 42 123 7 --part B 2>&1 | tail -45
echo "### TASK 3 plan2graph ###"
python3 task3_plan2graph.py --data $PKL --split $SPL 2>&1 | tail -20
echo "### TASK 3 edge classifier ###"
python3 task3_edge_classifier.py --data $PKL --split $SPL 2>&1 | tail -25
echo "### CROSS-DATASET ###"
python3 task2_cross_dataset.py --data $PKL --split $SPL 2>&1 | tail -30
echo "### TASK 2 generation ###"
python3 task1_generation.py --data $PKL --split $SPL 2>&1 | tail -40
echo "### DONE $(date) ###"
