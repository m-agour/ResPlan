#!/bin/bash
#SBATCH --job-name=ddpm-resplan
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=resplan_release/baselines/ddpm-%j.out
#SBATCH --error=resplan_release/baselines/ddpm-%j.err

cd /path/to/resplan

# Limit thread-parallelism inside workers to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1

/path/to/miniforge3/bin/python train_ddpm.py
