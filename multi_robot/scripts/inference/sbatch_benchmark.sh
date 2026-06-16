#!/usr/bin/env bash
# ============================================================
# Submit N SLURM jobs, splitting the (env, vel_max) work evenly.
#
# Usage:
#   cd /path/to/multi_robot
#   bash scripts/inference/sbatch_benchmark.sh              # default 6 jobs
#   bash scripts/inference/sbatch_benchmark.sh -n 3         # 3 jobs
#   bash scripts/inference/sbatch_benchmark.sh -n 4 --dry-run
#
# Extra args after -n <N> are forwarded to run_benchmark.py.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ─── CONFIGURATION (edit these) ───
ENVS=("empty" "highway" "conveyor" "dropregion")

# Per-env vel_maxs (must match run_benchmark.py ENVS dict)
declare -A ENV_VMAXS
ENV_VMAXS[empty]="0.675 0.692 0.703"
ENV_VMAXS[highway]="0.647 0.781 0.878"
ENV_VMAXS[conveyor]="1.21 1.46 1.76"
ENV_VMAXS[dropregion]="0.928 1.13 1.34"

# Default number of SLURM jobs
N_JOBS=6

# SLURM settings
PARTITION="${SLURM_PARTITION:-YOUR_GPU_PARTITION}"  # adjust to your cluster
GRES="gpu:1"
TIME="4-00:00:00"
CPUS=8
MEM="64G"                          # total memory (not per-cpu)
CONDA_ENV="direct-mmd"

# ─── PARSE -n <N_JOBS> and --dry-run ───
DRY_RUN=false
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N_JOBS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; ARGS+=("$1"); shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
EXTRA_ARGS="${ARGS[*]}"

# ─── BUILD WORK LIST: (env, vmax) pairs ───
WORK=()
for ENV in "${ENVS[@]}"; do
    for VMAX in ${ENV_VMAXS[$ENV]}; do
        WORK+=("${ENV}:${VMAX}")
    done
done
TOTAL=${#WORK[@]}

if [[ $N_JOBS -gt $TOTAL ]]; then
    N_JOBS=$TOTAL
fi

# ─── LOG DIRECTORY ───
TIMESTAMP=$(date +%y%m%d_%H%M%S)
SLURM_LOG_DIR="$PROJECT_ROOT/logs/benchmark_${TIMESTAMP}"
mkdir -p "$SLURM_LOG_DIR"

echo "============================================"
echo " Submitting benchmark: ${TOTAL} tasks -> ${N_JOBS} jobs"
echo " Envs:  ${ENVS[*]}"
echo " Tasks: ${WORK[*]}"
echo " Logs:  ${SLURM_LOG_DIR}"
echo "============================================"

# ─── SPLIT WORK INTO N_JOBS CHUNKS AND SUBMIT ───
for ((JOB=0; JOB<N_JOBS; JOB++)); do
    # Collect (env, vmax) pairs for this job (round-robin assignment)
    JOB_ENVS=()
    JOB_VMAXS=()
    for ((I=JOB; I<TOTAL; I+=N_JOBS)); do
        PAIR="${WORK[$I]}"
        E="${PAIR%%:*}"
        V="${PAIR##*:}"
        JOB_ENVS+=("$E")
        JOB_VMAXS+=("$V")
    done

    JOB_NAME="bench_j$((JOB+1))of${N_JOBS}"
    SLURM_OUT="${SLURM_LOG_DIR}/${JOB_NAME}_%j.out"
    SLURM_ERR="${SLURM_LOG_DIR}/${JOB_NAME}_%j.err"

    # Build the run commands for this job
    RUN_CMDS=""
    for ((K=0; K<${#JOB_ENVS[@]}; K++)); do
        E="${JOB_ENVS[$K]}"
        V="${JOB_VMAXS[$K]}"
        RUN_CMDS+="echo \"--- Task $((K+1))/${#JOB_ENVS[@]}: env=${E} vmax=${V} ---\"
python scripts/inference/run_benchmark.py --gpu-id 0 --envs ${E} --vmaxs ${V} -m \"${JOB_NAME}_${E}_v${V}\" ${EXTRA_ARGS}
"
    done

    echo ""
    echo "[Job $((JOB+1))/${N_JOBS}] ${JOB_NAME}: ${#JOB_ENVS[@]} tasks"
    for ((K=0; K<${#JOB_ENVS[@]}; K++)); do
        echo "  - env=${JOB_ENVS[$K]} vmax=${JOB_VMAXS[$K]}"
    done

    if $DRY_RUN; then
        echo "  [DRY RUN] would submit sbatch with: ${EXTRA_ARGS}"
        continue
    fi

    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --gres="$GRES" \
        --time="$TIME" \
        --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output="$SLURM_OUT" \
        --error="$SLURM_ERR" \
        --wrap="$(cat <<EOF
#!/usr/bin/env bash
set -e
echo "Job \$SLURM_JOB_ID: ${JOB_NAME}"
echo "Host: \$(hostname), GPU: \$CUDA_VISIBLE_DEVICES"

source "\${CONDA_BASE:-\$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# JAX cuDNN + XLA memory settings
export LD_LIBRARY_PATH=\$HOME/.local/cudnn89/nvidia/cudnn/lib:\$CONDA_HOME/envs/direct-mmd/lib/python3.8/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd ${PROJECT_ROOT}

${RUN_CMDS}
echo "Job \$SLURM_JOB_ID complete."
EOF
)"

done

echo ""
echo "============================================"
echo " Submitted ${N_JOBS} jobs (${TOTAL} tasks total)."
echo " Monitor: squeue -u \$USER"
echo " Logs:    ${SLURM_LOG_DIR}/"
echo "============================================"
