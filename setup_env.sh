#!/usr/bin/env bash
# Set up the RAT-CFSR conda environment and install dependencies.
set -euo pipefail

export CONDA_BASE=/home/zjut/miniconda3
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

PROJECT_DIR=/home/zjut/public/zjm/RAT-CFSR

echo "[$(date '+%F %T')] Creating conda env 'RAT-CFSR' (python 3.12) ..."
conda create -n RAT-CFSR python=3.12 -y

echo "[$(date '+%F %T')] Activating env ..."
conda activate RAT-CFSR

echo "[$(date '+%F %T')] Installing PyTorch (CUDA 12.4) ..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "[$(date '+%F %T')] Installing numpy / scikit-learn / scipy / pytest ..."
pip install "numpy>=2.0" scikit-learn scipy pytest

echo "[$(date '+%F %T')] Installing project in editable mode ..."
pip install -e "$PROJECT_DIR[test]"

echo "[$(date '+%F %T')] Verifying ..."
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
python -c "import numpy, sklearn, scipy; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'scipy', scipy.__version__)"
python -c "import rat_cfsr; print('rat_cfsr import OK')"

echo "[$(date '+%F %T')] Environment setup complete."
