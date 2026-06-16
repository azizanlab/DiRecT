#!/bin/bash
#SBATCH --job-name=sweep_pcd_scl
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Create log directory
mkdir -p slurm_logs

# Activate conda environment
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate pcdp

# Move to repo directory
cd "$(dirname "$0")/.."

# ============================================================================
# Global parameters
# ============================================================================
GPU_ID=0
N_INIT_STATES=50
TRIAL=200
N_DIFFUSION_STEPS=32
V_MAX=8.4
GROUP_SIZE=2
STP=10
EXP_NAME="sweep_pcd_scl2"

CMD="python scripts/eval_H16_seq.py"

# Sweep values (per method)
SCL_VALUES_LB=(3e-4 1e-3 3e-3 0.01 0.03 0.1 0.3 1.0)
SCL_VALUES_DPP=(3e-4 1e-3 3e-3 0.01 0.03 0.1 0.3 1.0)

echo "============================================"
echo "PCD scl Sweep - $(date)"
echo "SCL_LB values: ${SCL_VALUES_LB[@]}"
echo "SCL_DPP values: ${SCL_VALUES_DPP[@]}"
echo "============================================"

# ============================================================================
# PCD-LB sweep
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=sum_log_l2

for SCL in "${SCL_VALUES_LB[@]}"; do
    COMMENT="${EXP_NAME}-PCD_LB-scl${SCL}"
    echo ">>> PCD-LB scl=${SCL}"
    $CMD \
        --guider ${GUIDER} \
        --projector ${PROJECTOR} \
        --cost-func-key ${COST} \
        -m "${COMMENT}" \
        n_init_states=${N_INIT_STATES} \
        trial=${TRIAL} \
        n_diffusion_steps=${N_DIFFUSION_STEPS} \
        v_max=${V_MAX} \
        group_size=${GROUP_SIZE} \
        stp=${STP} \
        scl=${SCL}
done

# ============================================================================
# PCD-DPP sweep
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=dpp

for SCL in "${SCL_VALUES_DPP[@]}"; do
    COMMENT="${EXP_NAME}-PCD_DPP-scl${SCL}"
    echo ">>> PCD-DPP scl=${SCL}"
    $CMD \
        --guider ${GUIDER} \
        --projector ${PROJECTOR} \
        --cost-func-key ${COST} \
        -m "${COMMENT}" \
        n_init_states=${N_INIT_STATES} \
        trial=${TRIAL} \
        n_diffusion_steps=${N_DIFFUSION_STEPS} \
        v_max=${V_MAX} \
        group_size=${GROUP_SIZE} \
        stp=${STP} \
        scl=${SCL}
done

echo "============================================"
echo "PCD scl Sweep complete - $(date)"
echo "============================================"
