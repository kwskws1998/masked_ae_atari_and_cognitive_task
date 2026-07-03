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
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DATA_GAMES="${DATA_GAMES:-breakout}"
INSTALL_ATARI_ROMS="${INSTALL_ATARI_ROMS:-1}"
ALLOW_CONDA_BASE="${ALLOW_CONDA_BASE:-0}"
ALLOW_SYSTEM_PYTHON="${ALLOW_SYSTEM_PYTHON:-0}"

if [ "${CREATE_VENV}" != "0" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  if [ "${CONDA_DEFAULT_ENV:-}" = "base" ] && [ "${ALLOW_CONDA_BASE}" != "1" ]; then
    echo "Refusing to install into conda base. Run 'conda create -n atari-gaze-dt python=3.12 -y' and 'conda activate atari-gaze-dt', or set ALLOW_CONDA_BASE=1 explicitly." >&2
    exit 2
  fi
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ] && [ "${ALLOW_SYSTEM_PYTHON}" != "1" ]; then
    echo "CREATE_VENV=0 requires an active conda/venv environment. Activate one first, or set ALLOW_SYSTEM_PYTHON=1 explicitly." >&2
    exit 2
  fi
fi

echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-none} venv=${VIRTUAL_ENV:-none}"

python -m pip install --upgrade pip setuptools wheel

if [ "${INSTALL_TORCH}" != "0" ]; then
  python -m pip install --index-url "${PYTORCH_INDEX_URL}" "${TORCH_SPEC}" "${TORCHVISION_SPEC}"
fi

python -m pip install -r requirements.txt

if [ "${INSTALL_ATARI_ROMS}" != "0" ]; then
  AutoROM --accept-license
fi

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install "huggingface_hub[cli]"
fi

python -c "import torch, gymnasium, ale_py, h5py, PIL; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

if [ "${DOWNLOAD_DATA}" != "0" ]; then
  read -r -a GAME_ARGS <<< "${DATA_GAMES}"
  bash scripts/setup_atari_head_v4_data.sh "${GAME_ARGS[@]}"
else
  echo "Skipping Atari-HEAD data download because DOWNLOAD_DATA=0"
fi
