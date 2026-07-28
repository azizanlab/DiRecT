#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=<your-partition>
#SBATCH --nodelist=<your-node>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

mkdir -p slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate direct-d3il

cd "$(dirname "$0")"

echo "============================================"
echo "Eval All Methods (dynamics always on) - $(date)"
echo "============================================"

# ============================================================
# Shared parameters
# ============================================================
CP=10 # Diffusion checkpoint
EXP=x0pred
SEED=0
REPEAT=100
BATCH=1
STEPS=32        # Sampling freq.
STEPS_LONG=256
REPLAN=4        # Replan freq.
MARGIN=0.025
RELAXATION=0 # No relaxation (equality constraints)
KP=0.0  # No pdi tracker (open loop over executed horizon) 
KD=0.0 
KI=0.0

COMMON="--diffusion-exp-name $EXP --diffusion-cp $CP --seed $SEED \
  --random-repeat $REPEAT --batch-size $BATCH --replan-steps $REPLAN \
  --n-sampling-steps $STEPS --obstacle-margin $MARGIN \
  --pd-kp $KP --pd-kd $KD --pd-ki $KI --no-early-stop"

# ============================================================
# 1. Unconditional (baseline)
# ============================================================
echo "=== Unconditional ==="
python eval.py $COMMON \
  --guidance-method no \
  --exp-name uncond

# ============================================================
# 2. Gradient Guidance (DPS)
# ============================================================
echo "=== Gradient Guidance ==="
python eval.py $COMMON \
  --guidance-method gradient \
  --constraints \
  --guidance-lr 0.05 \
  --guidance-steps 5 \
  --gradient-start 0.0 \
  --obstacle-weight 100.0 \
  --dynamics-weight 10.0 \
  --exp-name gradient

# ============================================================
# 3. Classifier Guidance
# ============================================================
echo "=== Classifier Guidance ==="
python eval.py $COMMON \
  --guidance-method classifier \
  --constraints \
  --n-sampling-steps $STEPS \
  --guidance-lr 0.05 \
  --gradient-start 0.0 \
  --obstacle-weight 100.0 \
  --dynamics-weight 10.0 \
  --exp-name classifier

# ============================================================
# 4. SafeDiffuser RoS
# ============================================================
echo "=== SafeDiffuser RoS ==="
python eval.py $COMMON \
  --guidance-method safediffuser \
  --constraints \
  --n-sampling-steps $STEPS_LONG \
  --cbf-algorithm RoS \
  --cbf-coefficient 1.0 \
  --exp-name safediffuser_ros

# ============================================================
# 5. SafeDiffuser ReS
# ============================================================
echo "=== SafeDiffuser ReS ==="
python eval.py $COMMON \
  --guidance-method safediffuser \
  --constraints \
  --n-sampling-steps $STEPS_LONG \
  --cbf-algorithm ReS \
  --cbf-coefficient 1.0 \
  --cbf-slack-weight 1000.0 \
  --cbf-relaxation-end 0.9 \
  --exp-name safediffuser_res

# ============================================================
# 6. SafeDiffuser TVS
# ============================================================
echo "=== SafeDiffuser TVS ==="
python eval.py $COMMON \
  --guidance-method safediffuser \
  --constraints \
  --n-sampling-steps $STEPS_LONG \
  --cbf-algorithm TVS \
  --cbf-coefficient 1.0 \
  --cbf-sigmoid-bias 0.9 \
  --cbf-sigmoid-input 256.0 \
  --cbf-sigmoid-output 1.0 \
  --exp-name safediffuser_tvs

# ============================================================
# 7. Primal-Dual
# ============================================================
echo "=== Primal-Dual ==="
python eval.py $COMMON \
  --guidance-method primal-dual \
  --constraints \
  --n-sampling-steps $STEPS_LONG \
  --pd-learning-rate 0.456 \        # Rescaled vals
  --no-pd-use-cbfs \
  --exp-name primal_dual

# ============================================================
# 8. Augmented Lagrangian
# ============================================================
echo "=== Augmented Lagrangian ==="
python eval.py $COMMON \
  --guidance-method augmented-lagrangian \
  --constraints \
  --n-sampling-steps $STEPS_LONG \
  --pd-learning-rate 0.456 \          # Rescaled vals
  --pd-initial-penalty 1e-6 \
  --pd-final-penalty 1e-5 \
  --no-pd-use-cbfs \
  --exp-name augmented_lagrangian

# ============================================================
# 9. Projection (IPOPT in noisy space)
# ============================================================
echo "=== Projection ==="
python eval.py $COMMON \
  --guidance-method projection \
  --constraints \
  --dynamics-relaxation $RELAXATION \
  --projection-start 0.5 \
  --exp-name projection

# ============================================================
# 10. DiRecT (IPOPT in clean space)
# ============================================================
echo "=== DiRecT ==="
python eval.py $COMMON \
  --guidance-method direct \
  --constraints \
  --dynamics-relaxation $RELAXATION \
  --pred-control-weight 1.0 \         # Cost seeking behavior lambda=0.028
  --pred-objective "" \
  --pred-objective-scale 10.0 \
  --guidance-last-steps 16 \
  --exp-name direct

# echo "=== DiRecT + distance objective ==="
# python eval.py $COMMON \
#   --guidance-method direct \
#   --constraints \
#   --dynamics-relaxation $RELAXATION \
#   --pred-control-weight 1.0 \         # Cost seeking behavior lambda=0.028
#   --pred-objective distance \
#   --pred-objective-scale 10.0 \
#   --guidance-last-steps 16 \
#   --exp-name direct_distance

echo "============================================"
echo "All evaluations complete - $(date)"
echo "============================================"
