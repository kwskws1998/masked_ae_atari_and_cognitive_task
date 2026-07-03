#!/usr/bin/env bash
# Run the Breakout Active-Gaze Decision Transformer pipeline end to end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

AUTO_ACTIVATE_VENV="${AUTO_ACTIVATE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv}"
PROFILE="${PROFILE:-smoke}"
GAME="${GAME:-breakout}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-active_dt}"
HF_REPO="${HF_REPO:-skboy/atari-head-v4}"
OUT_ROOT="${OUT_ROOT:-artifacts/active_gaze_dt/${GAME}_${PROFILE}}"
HDF5_PATH="${HDF5_PATH:-external/amsterg_ahead/data/processed/${GAME}.hdf5}"

if [ "${AUTO_ACTIVATE_VENV}" != "0" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
fi

case "${PROFILE}" in
  smoke)
    MAX_TRIALS="${MAX_TRIALS:-1}"
    MAX_FRAMES="${MAX_FRAMES:-96}"
    MAX_SAMPLES="${MAX_SAMPLES:-16}"
    EPOCHS="${EPOCHS:-1}"
    LR="${LR:-3e-4}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-0}"
    PIN_MEMORY="${PIN_MEMORY:-0}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-4}"
    EMBED_DIM="${EMBED_DIM:-32}"
    ENCODER_LAYERS="${ENCODER_LAYERS:-1}"
    ENCODER_HEADS="${ENCODER_HEADS:-4}"
    ENCODER_FF_DIM="${ENCODER_FF_DIM:-64}"
    DECODER_DIM="${DECODER_DIM:-32}"
    DECODER_LAYERS="${DECODER_LAYERS:-1}"
    DECODER_HEADS="${DECODER_HEADS:-4}"
    DECODER_FF_DIM="${DECODER_FF_DIM:-64}"
    DT_LAYERS="${DT_LAYERS:-1}"
    DT_HEADS="${DT_HEADS:-4}"
    MAX_TIMESTEP="${MAX_TIMESTEP:-4096}"
    EVAL_EPISODES="${EVAL_EPISODES:-1}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-64}"
    NO_COMPRESSION="${NO_COMPRESSION:-1}"
    COMBINED="${COMBINED:-0}"
    ;;
  pilot)
    MAX_TRIALS="${MAX_TRIALS:-4}"
    MAX_FRAMES="${MAX_FRAMES:-20000}"
    MAX_SAMPLES="${MAX_SAMPLES:-20000}"
    EPOCHS="${EPOCHS:-5}"
    LR="${LR:-3e-4}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    PIN_MEMORY="${PIN_MEMORY:-1}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-8}"
    EMBED_DIM="${EMBED_DIM:-128}"
    ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
    ENCODER_HEADS="${ENCODER_HEADS:-4}"
    ENCODER_FF_DIM="${ENCODER_FF_DIM:-256}"
    DECODER_DIM="${DECODER_DIM:-128}"
    DECODER_LAYERS="${DECODER_LAYERS:-1}"
    DECODER_HEADS="${DECODER_HEADS:-4}"
    DECODER_FF_DIM="${DECODER_FF_DIM:-256}"
    DT_LAYERS="${DT_LAYERS:-4}"
    DT_HEADS="${DT_HEADS:-4}"
    MAX_TIMESTEP="${MAX_TIMESTEP:-30000}"
    EVAL_EPISODES="${EVAL_EPISODES:-5}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-5000}"
    NO_COMPRESSION="${NO_COMPRESSION:-1}"
    COMBINED="${COMBINED:-0}"
    ;;
  real)
    MAX_TRIALS="${MAX_TRIALS:-0}"
    MAX_FRAMES="${MAX_FRAMES:-0}"
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    EPOCHS="${EPOCHS:-20}"
    LR="${LR:-3e-4}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    PIN_MEMORY="${PIN_MEMORY:-1}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-30}"
    EMBED_DIM="${EMBED_DIM:-128}"
    ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
    ENCODER_HEADS="${ENCODER_HEADS:-4}"
    ENCODER_FF_DIM="${ENCODER_FF_DIM:-256}"
    DECODER_DIM="${DECODER_DIM:-128}"
    DECODER_LAYERS="${DECODER_LAYERS:-1}"
    DECODER_HEADS="${DECODER_HEADS:-4}"
    DECODER_FF_DIM="${DECODER_FF_DIM:-256}"
    DT_LAYERS="${DT_LAYERS:-4}"
    DT_HEADS="${DT_HEADS:-4}"
    MAX_TIMESTEP="${MAX_TIMESTEP:-30000}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-108000}"
    NO_COMPRESSION="${NO_COMPRESSION:-0}"
    COMBINED="${COMBINED:-0}"
    ;;
  rtx3090)
    MAX_TRIALS="${MAX_TRIALS:-0}"
    MAX_FRAMES="${MAX_FRAMES:-0}"
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    EPOCHS="${EPOCHS:-30}"
    LR="${LR:-1e-4}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    PIN_MEMORY="${PIN_MEMORY:-1}"
    AMP="${AMP:-1}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-30}"
    EMBED_DIM="${EMBED_DIM:-256}"
    ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
    ENCODER_HEADS="${ENCODER_HEADS:-8}"
    ENCODER_FF_DIM="${ENCODER_FF_DIM:-1024}"
    DECODER_DIM="${DECODER_DIM:-256}"
    DECODER_LAYERS="${DECODER_LAYERS:-2}"
    DECODER_HEADS="${DECODER_HEADS:-8}"
    DECODER_FF_DIM="${DECODER_FF_DIM:-1024}"
    DT_LAYERS="${DT_LAYERS:-6}"
    DT_HEADS="${DT_HEADS:-8}"
    MAX_TIMESTEP="${MAX_TIMESTEP:-30000}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-108000}"
    NO_COMPRESSION="${NO_COMPRESSION:-0}"
    COMBINED="${COMBINED:-0}"
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}. Use smoke, pilot, real, or rtx3090."
    exit 1
    ;;
esac
AMP="${AMP:-0}"

bash scripts/setup_atari_head_v4_data.sh "${GAME}"

PREPARE_ARGS=(
  --game "${GAME}"
  --max-trials "${MAX_TRIALS}"
  --max-frames "${MAX_FRAMES}"
  --overwrite
)
if [ "${COMBINED}" = "1" ]; then
  PREPARE_ARGS+=(--combined)
fi
if [ "${NO_COMPRESSION}" = "1" ]; then
  PREPARE_ARGS+=(--no-compression)
fi
python scripts/prepare_amsterg_hdf5.py "${PREPARE_ARGS[@]}"

TRAIN_ARGS=(
  --mode "${MODE}"
  --hdf5 "${HDF5_PATH}"
  --output-dir "${OUT_ROOT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --context-length "${CONTEXT_LENGTH}"
  --embed-dim "${EMBED_DIM}"
  --encoder-layers "${ENCODER_LAYERS}"
  --encoder-heads "${ENCODER_HEADS}"
  --encoder-ff-dim "${ENCODER_FF_DIM}"
  --decoder-dim "${DECODER_DIM}"
  --decoder-layers "${DECODER_LAYERS}"
  --decoder-heads "${DECODER_HEADS}"
  --decoder-ff-dim "${DECODER_FF_DIM}"
  --dt-layers "${DT_LAYERS}"
  --dt-heads "${DT_HEADS}"
  --max-timestep "${MAX_TIMESTEP}"
  --device "${DEVICE}"
  --require-rewards
)
if [ -n "${MAX_SAMPLES}" ]; then
  TRAIN_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${PIN_MEMORY}" = "1" ]; then
  TRAIN_ARGS+=(--pin-memory)
fi
if [ "${AMP}" = "1" ]; then
  TRAIN_ARGS+=(--amp)
fi
python scripts/train_active_gaze_dt.py "${TRAIN_ARGS[@]}"

python scripts/evaluate_gymnasium_atari_policy.py \
  --game "${GAME}" \
  --model-type active-dt \
  --checkpoint "${OUT_ROOT}/${MODE}.pt" \
  --episodes "${EVAL_EPISODES}" \
  --max-steps "${EVAL_MAX_STEPS}" \
  --frameskip 1 \
  --policy sample \
  --temperature 1.0 \
  --context-length "${CONTEXT_LENGTH}" \
  --target-return 20 \
  --device "${DEVICE}" \
  --output-json "artifacts/gymnasium_eval/${GAME}_${PROFILE}_${MODE}.json"
