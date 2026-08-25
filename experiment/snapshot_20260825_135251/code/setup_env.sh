#!/usr/bin/env bash
# Set up the RAT-CFSR conda environment and install project dependencies.
set -euo pipefail

export CONDA_BASE=${CONDA_BASE:-/home/zjut/miniconda3}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME=${ENV_NAME:-RAT-CFSR}
PROJECT_DIR=${PROJECT_DIR:-/home/zjut/public/zjm/RAT-CFSR}
PROXY_URL=${PROXY_URL:-http://192.168.20.51:7897}

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

if env_exists "$ENV_NAME"; then
    echo "[$(date '+%F %T')] Conda env '$ENV_NAME' already exists."
elif env_exists torchsig; then
    echo "[$(date '+%F %T')] Cloning verified env 'torchsig' into '$ENV_NAME' ..."
    conda create -n "$ENV_NAME" --clone torchsig -y
else
    echo "[$(date '+%F %T')] Creating conda env '$ENV_NAME' (python 3.10) ..."
    conda create -n "$ENV_NAME" python=3.10 -y --override-channels \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    conda activate "$ENV_NAME"
    echo "[$(date '+%F %T')] Installing PyTorch (CUDA 12.6) ..."
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    echo "[$(date '+%F %T')] Installing numpy / scikit-learn / scipy / pytest ..."
    pip install "numpy>=2.0" scikit-learn scipy pytest
    conda deactivate
fi

echo "[$(date '+%F %T')] Setting proxy variables for '$ENV_NAME' ..."
conda env config vars set -n "$ENV_NAME" \
    HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL" \
    http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL"

echo "[$(date '+%F %T')] Activating env ..."
conda activate "$ENV_NAME"

echo "[$(date '+%F %T')] Installing project in editable mode ..."
pip install -e "$PROJECT_DIR[test]"

echo "[$(date '+%F %T')] Verifying ..."
python -c "import sys; print('python', sys.version.split()[0], sys.executable)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
python -c "import numpy, sklearn, scipy; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'scipy', scipy.__version__)"
python -c "import rat_cfsr; print('rat_cfsr import OK')"

echo "[$(date '+%F %T')] Environment setup complete. Use: conda activate $ENV_NAME"
