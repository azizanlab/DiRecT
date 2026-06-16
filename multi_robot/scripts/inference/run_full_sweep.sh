#!/bin/bash
# Full parameter sweep: all methods x all environments x all agent counts.
#
# Usage:
#   bash run_full_sweep.sh GPU_ID [--dry-run]
#
# Examples:
#   bash run_full_sweep.sh 0
#   bash run_full_sweep.sh 0 --dry-run

set -e

GPU=${1:?Usage: run_full_sweep.sh GPU_ID [--dry-run]}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python run_benchmark.py \
    --gpu-id "$GPU" \
    --envs empty highway conveyor dropregion \
    --agents 4 8 12 16 20 \
    --costs hinge_sqr_l2 sum_log_l2 \
    "$@"
