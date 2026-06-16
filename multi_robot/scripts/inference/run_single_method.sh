#!/bin/bash
# Run a single method from the benchmark suite.
#
# Usage:
#   bash run_single_method.sh GPU_ID METHOD [ENV] [N_AGENTS] [--dry-run]
#
# METHOD is one of: diffusion_policy, cd, pcd, direct, final_projection, mmd_cbs
#
# Examples:
#   bash run_single_method.sh 0 direct empty 4
#   bash run_single_method.sh 0 pcd highway 4 --dry-run
#   bash run_single_method.sh 0 diffusion_policy   # defaults: all envs, all agents

set -e

GPU=${1:?Usage: run_single_method.sh GPU_ID METHOD [ENV] [N_AGENTS] [--dry-run]}
METHOD=${2:?Usage: run_single_method.sh GPU_ID METHOD [ENV] [N_AGENTS] [--dry-run]}
ENV=${3:-}
N_AGENTS=${4:-}
shift 2
[ -n "$ENV" ] && shift
[ -n "$N_AGENTS" ] && shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXTRA_ARGS=()
[ -n "$ENV" ] && EXTRA_ARGS+=(--envs "$ENV")
[ -n "$N_AGENTS" ] && EXTRA_ARGS+=(--agents "$N_AGENTS")

python run_benchmark.py --gpu-id "$GPU" --only "$METHOD" "${EXTRA_ARGS[@]}" "$@"
