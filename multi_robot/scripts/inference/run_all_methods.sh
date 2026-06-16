#!/bin/bash
# Run all benchmark methods for a given environment and agent count.
#
# Usage:
#   bash run_all_methods.sh GPU_ID ENV N_AGENTS [--dry-run]
#
# Examples:
#   bash run_all_methods.sh 0 empty 4
#   bash run_all_methods.sh 0 highway 4 --dry-run

set -e

GPU=${1:?Usage: run_all_methods.sh GPU_ID ENV N_AGENTS [--dry-run]}
ENV=${2:?Usage: run_all_methods.sh GPU_ID ENV N_AGENTS [--dry-run]}
N_AGENTS=${3:?Usage: run_all_methods.sh GPU_ID ENV N_AGENTS [--dry-run]}
shift 3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python run_benchmark.py --gpu-id "$GPU" --envs "$ENV" --agents "$N_AGENTS" "$@"
