#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=<your-partition>
#SBATCH --nodelist=<your-node>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

mkdir -p slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate direct-d3il

cd "$(dirname "$0")"

python train.py --resume --exp-name x0pred
