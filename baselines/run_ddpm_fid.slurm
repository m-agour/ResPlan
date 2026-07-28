#!/bin/bash
#SBATCH --job-name=ddpm-fid
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=resplan_release/baselines/ddpm-fid-%j.out
#SBATCH --error=resplan_release/baselines/ddpm-fid-%j.err

cd /path/to/resplan

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/path/to/miniforge3/bin/python eval_ddpm.py --ckpt ddpm_results/model-20.pt --n 1632 --out fid_eval/ddpm --real fid_eval/real_test
