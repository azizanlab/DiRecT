#!/bin/bash
#SBATCH --job-name=pusht_benchmark
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
EXP_NAME="benchmark"

CMD="python scripts/eval_H16_seq.py"

echo "============================================"
echo "PushT Benchmark - $(date)"
echo "============================================"

# ============================================================================
# 1. Diffuser Policy (base)
# ============================================================================
GUIDER=vanilla
PROJECTOR=none
COMMENT="${EXP_NAME}-DP"

echo "[1/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    group_size=${GROUP_SIZE}

# ============================================================================
# 2. CD-DPP
# ============================================================================
GUIDER=coupling
PROJECTOR=none
COST=dpp
STP=1
SCL=0.2
COMMENT="${EXP_NAME}-CD_DPP"

echo "[2/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL}

# ============================================================================
# 3. CD-DPP-PS
# ============================================================================
GUIDER=coupling_ps
PROJECTOR=none
COST=dpp
STP=1
SCL=0.2
COMMENT="${EXP_NAME}-CD_DPP_PS"

echo "[3/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL}

# ============================================================================
# 4. CD-LB
# ============================================================================
GUIDER=coupling
PROJECTOR=none
COST=sum_log_l2
STP=1
SCL=0.02
COMMENT="${EXP_NAME}-CD_LB"

echo "[4/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL}

# ============================================================================
# 5. CD-LB-PS
# ============================================================================
GUIDER=coupling_ps
PROJECTOR=none
COST=sum_log_l2
STP=1
SCL=0.02
COMMENT="${EXP_NAME}-CD_LB_PS"

echo "[5/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL}

# ============================================================================
# 6. PCD-DPP
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=dpp
STP=1
SCL=2
COMMENT="${EXP_NAME}-PCD_DPP"

echo "[6/11] ${COMMENT}"
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

# ============================================================================
# 7. PCD-DPP-PS
# ============================================================================
GUIDER=coupling_ps
PROJECTOR=max_vel_admm
COST=dpp
STP=1
SCL=2
COMMENT="${EXP_NAME}-PCD_DPP_PS"

echo "[7/11] ${COMMENT}"
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

# ============================================================================
# 8. PCD-LB
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=sum_log_l2
STP=1
SCL=0.2
COMMENT="${EXP_NAME}-PCD_LB"

echo "[8/11] ${COMMENT}"
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

# ============================================================================
# 9. PCD-LB-PS
# ============================================================================
GUIDER=coupling_ps
PROJECTOR=max_vel_admm
COST=sum_log_l2
STP=1
SCL=0.2
COMMENT="${EXP_NAME}-PCD_LB_PS"

echo "[9/11] ${COMMENT}"
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

# ============================================================================
# 10. DiRecT-LB
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=sum_log_l2
STP=1
SCL=0.005
T_START_GUIDE=16
T_START_PROJ=16
COMMENT="${EXP_NAME}-DiRecT_LB"

echo "[10/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    --direct \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    v_max=${V_MAX} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL} \
    t_start_guide=${T_START_GUIDE} \
    t_start_projection=${T_START_PROJ}

# ============================================================================
# 11. DiRecT-DPP
# ============================================================================
GUIDER=coupling
PROJECTOR=max_vel_admm
COST=dpp
STP=1
SCL=0.05
T_START_GUIDE=16
T_START_PROJ=16
COMMENT="${EXP_NAME}-DiRecT_DPP"

echo "[11/11] ${COMMENT}"
$CMD \
    --guider ${GUIDER} \
    --projector ${PROJECTOR} \
    --cost-func-key ${COST} \
    --direct \
    -m "${COMMENT}" \
    n_init_states=${N_INIT_STATES} \
    trial=${TRIAL} \
    n_diffusion_steps=${N_DIFFUSION_STEPS} \
    v_max=${V_MAX} \
    group_size=${GROUP_SIZE} \
    stp=${STP} \
    scl=${SCL} \
    t_start_guide=${T_START_GUIDE} \
    t_start_projection=${T_START_PROJ}

echo "============================================"
echo "Benchmark complete - $(date)"
echo "============================================"
