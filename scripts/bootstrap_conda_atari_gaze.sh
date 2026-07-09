#!/usr/bin/env bash
# Bootstrap a conda environment, Atari ROMs, Atari-HEAD data, and HDF5 files for active-gaze Atari runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-atari-gaze-dt}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GAME="${GAME:-breakout}"
HF_REPO="${HF_REPO:-skboy/atari-head-v4}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1}"
INSTALL_ATARI_ROMS="${INSTALL_ATARI_ROMS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
PREPARE_HDF5="${PREPARE_HDF5:-1}"
OVERWRITE_HDF5="${OVERWRITE_HDF5:-0}"
MAX_TRIALS="${MAX_TRIALS:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
NO_COMPRESSION="${NO_COMPRESSION:-0}"
COMBINED="${COMBINED:-0}"
HDF5_PATH="${HDF5_PATH:-external/amsterg_ahead/data/processed/${GAME}.hdf5}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${HOME}/miniconda3"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${HOME}/anaconda3"
else
  echo "conda was not found. Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "=== create conda env ${ENV_NAME} python=${PYTHON_VERSION} ==="
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "=== reuse conda env ${ENV_NAME} ==="
fi

conda activate "${ENV_NAME}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
  echo "failed to activate conda env ${ENV_NAME}" >&2
  exit 2
fi

echo "=== install python dependencies env=${CONDA_DEFAULT_ENV} python=$(command -v python) ==="
CREATE_VENV=0 \
INSTALL_TORCH="${INSTALL_TORCH}" \
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
TORCH_SPEC="${TORCH_SPEC}" \
TORCHVISION_SPEC="${TORCHVISION_SPEC}" \
INSTALL_ATARI_ROMS="${INSTALL_ATARI_ROMS}" \
DOWNLOAD_DATA=0 \
bash scripts/setup_environment.sh

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install -U "huggingface_hub[cli]"
  hash -r
fi

if [ "${DOWNLOAD_DATA}" != "0" ]; then
  echo "=== verify Hugging Face auth ==="
  if ! hf auth whoami >/dev/null 2>&1; then
    if [ -n "${HF_TOKEN:-}" ]; then
      hf auth login --token "${HF_TOKEN}"
    else
      hf auth login
    fi
  fi

  echo "=== download Atari-HEAD archive game=${GAME} ==="
  AUTO_ACTIVATE_VENV=0 HF_REPO="${HF_REPO}" bash scripts/setup_atari_head_v4_data.sh "${GAME}"
else
  echo "=== skip Atari-HEAD download DOWNLOAD_DATA=0 ==="
fi

if [ "${PREPARE_HDF5}" != "0" ]; then
  if [ "${OVERWRITE_HDF5}" = "1" ] || [ ! -f "${HDF5_PATH}" ]; then
    echo "=== prepare HDF5 game=${GAME} path=${HDF5_PATH} ==="
    PREPARE_ARGS=(
      --game "${GAME}"
      --max-trials "${MAX_TRIALS}"
      --max-frames "${MAX_FRAMES}"
      --atomic-output
    )
    if [ "${OVERWRITE_HDF5}" = "1" ]; then
      PREPARE_ARGS+=(--overwrite)
    fi
    if [ "${NO_COMPRESSION}" = "1" ]; then
      PREPARE_ARGS+=(--no-compression)
    fi
    if [ "${COMBINED}" = "1" ]; then
      PREPARE_ARGS+=(--combined)
    fi
    python scripts/prepare_amsterg_hdf5.py "${PREPARE_ARGS[@]}"
  else
    echo "=== reuse existing HDF5 ${HDF5_PATH} ==="
  fi

  HDF5_PATH="${HDF5_PATH}" python -c "import h5py, os; path=os.environ['HDF5_PATH']; handle=h5py.File(path, 'r'); groups=[key for key in handle.keys() if key != 'combined']; print(f'hdf5_ok={path} groups={len(groups)}'); handle.close()"
else
  echo "=== skip HDF5 prepare PREPARE_HDF5=0 ==="
fi

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "=== bootstrap complete ==="
echo "Run experiments with: conda activate ${ENV_NAME}"
