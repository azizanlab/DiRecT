#!/bin/bash
#SBATCH --job-name=mmd_to_tile
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

mkdir -p slurm_logs

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate direct-mmd

cd "$(dirname "$0")/../.."

python scripts/analyze/mmd_to_single_tile.py
