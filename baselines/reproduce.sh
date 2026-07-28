#!/bin/bash
# ============================================================================
# reproduce.sh — Reproduce ALL paper numbers for ResPlan (NeurIPS 2025 D&B)
#
# Usage (CPU, all experiments):
#   bash reproduce.sh
#
# Usage (GPU for GNNs only, via SLURM):
#   sbatch run_gpu.slurm
#
# Expected output:
#   results/task1_results.json
#   results/task2_results.json
#   results/task2_cross_dataset.json
#   results/task3_results.json
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "ResPlan — Full Reproducibility Pipeline"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo ""

# Check data exists
if [ ! -f "../ResPlan.pkl" ]; then
    echo "ERROR: ../ResPlan.pkl not found. Place ResPlan.pkl in the parent directory."
    exit 1
fi
if [ ! -f "../split.json" ]; then
    echo "ERROR: ../split.json not found."
    exit 1
fi

mkdir -p results

# ── Task 1: Retrieval baselines ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 1: Constrained Floor Plan Generation (Retrieval)"
echo "════════════════════════════════════════"
python3 task1_retrieval.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --output results/task1_results.json

# ── Task 1: Learned generative baselines (CVAE) ─────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 1: Constrained Floor Plan Generation (CVAE)"
echo "════════════════════════════════════════"
python3 task1_generation.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device auto \
    --epochs 800 \
    --batch 256 \
    --beta 0.1 \
    --warmup 50 \
    --patience 150 \
    --samples 10 \
    --seed 42 \
    --output results/task1_generation.json

# ── Task 2: Semantic Room Labeling ───────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 2: Semantic Room Labeling (All 5 methods)"
echo "════════════════════════════════════════"
python3 task2_baselines.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device auto \
    --seeds 42 123 7 \
    --epochs 500 \
    --output results/task2_results.json

# ── Task 2: Ablation studies ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 2: Feature & Architecture Ablation"
echo "════════════════════════════════════════"
python3 task2_ablation.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --device auto \
    --seeds 42 123 7 \
    --epochs 500 \
    --part AB \
    --output results/task2_ablation.json

# ── Task 2: Cross-dataset transfer ──────────────────────────────────────────
# Requires RPLAN data from Graph2Plan
RPLAN_PATH="../../rplan_data/Interface/static/Data/data_train_converted.pkl"
if [ -f "$RPLAN_PATH" ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "Task 2: Cross-Dataset Transfer (ResPlan ↔ RPLAN)"
    echo "════════════════════════════════════════"
    python3 task2_cross_dataset.py \
        --data ../ResPlan.pkl \
        --split ../split.json \
        --rplan "$RPLAN_PATH" \
        --device auto \
        --seed 42 \
        --epochs 300 \
        --output results/task2_cross_dataset.json
else
    echo ""
    echo "SKIP: Cross-dataset transfer (RPLAN data not found at $RPLAN_PATH)"
    echo "  Download Graph2Plan data and set --rplan path to run this experiment."
fi

# ── Task 2: Domain adaptation ───────────────────────────────────────────────
if [ -f "$RPLAN_PATH" ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "Task 2: Domain Adaptation (RPLAN → ResPlan fine-tuning)"
    echo "════════════════════════════════════════"
    python3 task2_domain_adapt.py \
        --data ../ResPlan.pkl \
        --split ../split.json \
        --rplan "$RPLAN_PATH" \
        --device auto \
        --seeds 42 123 7 \
        --pretrain-epochs 300 \
        --finetune-epochs 200 \
        --output results/task2_domain_adapt.json
else
    echo ""
    echo "SKIP: Domain adaptation (RPLAN data not found at $RPLAN_PATH)"
fi

# ── Task 3: Plan-to-Graph extraction ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 3: Plan-to-Graph Extraction"
echo "════════════════════════════════════════"
python3 task3_plan2graph.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --output results/task3_results.json

# ── Task 3: Learned edge-type classifier ────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Task 3: Learned Edge-Type Classification"
echo "════════════════════════════════════════"
python3 task3_edge_classifier.py \
    --data ../ResPlan.pkl \
    --split ../split.json \
    --n-jobs -1 \
    --output results/task3_edge_type.json

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "ALL DONE"
echo "════════════════════════════════════════"
echo "Results saved in: $(realpath results/)"
ls -la results/
echo ""
echo "To verify numbers match the paper, compare:"
echo "  results/task1_results.json       → Table 8 (Task 1 retrieval)"
echo "  results/task1_generation.json    → Table 8 (Task 1 CVAE)"
echo "  results/task2_results.json       → Tables 4 & 5 (Task 2)"
echo "  results/task2_ablation.json      → Tables 6 & 7 (Ablation)"
echo "  results/task2_cross_dataset.json → Table 8 (Cross-dataset)"
echo "  results/task2_domain_adapt.json  → Table 9 (Domain adaptation)"
echo "  results/task3_results.json       → Table 10 (Task 3 heuristics)"
echo "  results/task3_edge_type.json     → Table 10 (Task 3 learned classifier)"
