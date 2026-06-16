#!/bin/bash
#SBATCH --job-name=bench_conv_v1.21
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

mkdir -p slurm_logs

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate direct-mmd

cd "$(dirname "$0")/../.."

python scripts/inference/run_benchmark.py \
    --gpu-id 0 \
    --envs conveyor \
    --vmaxs 1.21 \
    -m "bench_j1of6_conveyor_v1.21"
