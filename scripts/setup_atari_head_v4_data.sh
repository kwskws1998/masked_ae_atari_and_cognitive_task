#!/usr/bin/env bash
# Set up Atari-HEAD v4 game archives from the configured Hugging Face dataset mirror.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

AUTO_ACTIVATE_VENV="${AUTO_ACTIVATE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv}"
HF_REPO="${HF_REPO:-skboy/atari-head-v4}"
OUT_DIR="${ATARI_HEAD_V4_DIR:-data/atari_head_full/v4}"

if [ "${AUTO_ACTIVATE_VENV}" != "0" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
fi

if [ "$#" -eq 0 ]; then
  GAMES=(breakout)
else
  GAMES=("$@")
fi

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install "huggingface_hub[cli]"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed and could not be installed with pip." >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  hf auth login
fi

python scripts/download_atari_head_v4.py \
  --source hf \
  --hf-repo "${HF_REPO}" \
  --out "${OUT_DIR}" \
  --games "${GAMES[@]}"
