#!/bin/bash
#SBATCH --job-name=fid-task1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=resplan_release/baselines/fid-%j.out
#SBATCH --error=resplan_release/baselines/fid-%j.err

cd /path/to/resplan
source ~/.bashrc
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1

python compute_task1_fid.py
