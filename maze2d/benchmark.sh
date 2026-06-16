#!/bin/bash
#SBATCH --job-name=bench_dyn
#SBATCH --time=4-00:00:00
#SBATCH --output=./logs/slurm/%j_%x.out
#SBATCH --error=./logs/slurm/%j_%x.err
#SBATCH --open-mode=truncate
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -eu

[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR" ] && cd "$SLURM_SUBMIT_DIR"
mkdir -p logs logs/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

set +u
# Adjust this path if your conda installation lives elsewhere
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate direct-maze2d
set -u

SEED=0
NUM_REPEATS=100
EXP_PREFIX="benchmark-dyn"
WEIGHTS="<PATH_TO_CHECKPOINT>"

POLICIES=(
    no-guidance
    classifier-guidance
    gradient-guidance
    projection
    primal-dual
    augmented-lagrangian
    safediffuser-ros
    safediffuser-res
    safediffuser-tvs
    direct
)

OBSTACLE_TYPES=(
    broad
    narrow
)

FAILED=0
TOTAL=0

for OBSTACLE in "${OBSTACLE_TYPES[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        EXP_NAME="${EXP_PREFIX}/${OBSTACLE}/${POLICY}"
        TOTAL=$((TOTAL + 1))

        echo "=================================================================="
        echo "[${TOTAL}] Running: ${POLICY} (${OBSTACLE})"
        echo "=================================================================="

        python eval.py \
            --policy "${POLICY}" \
            --obstacle-type "${OBSTACLE}" \
            --seed "${SEED}" \
            --exp-name "${EXP_NAME}" \
            --weights-path "${WEIGHTS}" \
            --num-random-repeats "${NUM_REPEATS}" \
            --device cuda \
        && echo "[OK] ${POLICY} (${OBSTACLE})" \
        || { echo "[FAIL] ${POLICY} (${OBSTACLE})"; FAILED=$((FAILED + 1)); }

        echo ""
    done
done

echo "=================================================================="
echo "BENCHMARK COMPLETE: ${TOTAL} runs, ${FAILED} failures"
echo "=================================================================="
