#!/bin/bash
# Setup script for direct-d3il conda environment
# Usage: bash setup_env.sh
set -e

ENV_NAME="direct-d3il"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Creating conda env: $ENV_NAME ==="
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda env create -f "$SCRIPT_DIR/environment.yml"

echo "=== Installing qpth (needs --no-build-isolation due to broken numpy specifier) ==="
conda run -n "$ENV_NAME" pip install qpth --no-build-isolation --no-deps
conda run -n "$ENV_NAME" pip install "numpy>=2.2,<2.4"

echo "=== Setting up activation script for LD_LIBRARY_PATH ==="
ACTIVATE_DIR="$(conda info --envs | grep "^$ENV_NAME " | awk '{print $NF}')/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/env_vars.sh" << 'ENVSCRIPT'
#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/nvidia
ENVSCRIPT

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
echo "=== Run training with: python train.py --exp-name x0pred ==="
