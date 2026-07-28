#!/bin/bash
#SBATCH --job-name=task3-gnn
#SBATCH --output=task3-gnn-%j.out
#SBATCH --error=task3-gnn-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --account=YOUR_ACCOUNT

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /path/to/resplan
/path/to/miniforge3/bin/python task3_gnn.py
