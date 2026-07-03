#!/usr/bin/env bash
# Set up the local Python environment for Atari-HEAD active-gaze experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CREATE_VENV="${CREATE_VENV:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [ "${CREATE_VENV}" != "0" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if [ "${INSTALL_TORCH}" != "0" ]; then
  python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision
fi

python -m pip install -r requirements.txt

if ! command -v hf >/dev/null 2>&1; then
  curl -LsSf https://hf.co/cli/install.sh | bash -s
  export PATH="${HOME}/.local/bin:${PATH}"
fi

python -c "import torch, gymnasium, ale_py, h5py, PIL; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

