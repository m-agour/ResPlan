#!/bin/bash
#SBATCH --job-name=rest-rerun
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=rest-rerun-%j.out
#SBATCH --error=rest-rerun-%j.err
cd /path/to/resplan/resplan_release/baselines
echo "### TASK 3: plan-to-graph ###"
python3 task3_plan2graph.py --data ../ResPlan.pkl --split ../split.json 2>&1 | tail -40 || echo "TASK3_PLAN2GRAPH_FAILED"
echo "### TASK 3: edge classifier ###"
python3 task3_edge_classifier.py --data ../ResPlan.pkl --split ../split.json 2>&1 | tail -40 || echo "TASK3_EDGECLF_FAILED"
echo "### CROSS-DATASET ###"
python3 task2_cross_dataset.py --data ../ResPlan.pkl --split ../split.json 2>&1 | tail -40 || echo "CROSS_FAILED"
echo "### TASK 2: generation ###"
python3 task1_generation.py --data ../ResPlan.pkl --split ../split.json 2>&1 | tail -60 || echo "GEN_FAILED"
echo "### DONE $(date) ###"
