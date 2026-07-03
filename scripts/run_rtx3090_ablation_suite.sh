#!/usr/bin/env bash
# Run 3090-oriented Breakout ablations with matched model width where needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
AUTO_ACTIVATE_VENV="${AUTO_ACTIVATE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv}"
SUITE_PROFILE="${SUITE_PROFILE:-pilot}"
GAME="${GAME:-breakout}"
DEVICE="${DEVICE:-cuda}"
HF_REPO="${HF_REPO:-skboy/atari-head-v4}"
HDF5_PATH="${HDF5_PATH:-external/amsterg_ahead/data/processed/${GAME}.hdf5}"
SUITE_ROOT="${SUITE_ROOT:-artifacts/active_gaze_dt/${GAME}_${SUITE_PROFILE}_ablation}"
SEEDS="${SEEDS:-42}"
SETUP_ENV="${SETUP_ENV:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
mkdir -p artifacts "${SUITE_ROOT}" artifacts/gymnasium_eval
EVAL_POLICY="${EVAL_POLICY:-argmax}"
EVAL_START_ACTIONS="${EVAL_START_ACTIONS:-1}"

if [ "${AUTO_ACTIVATE_VENV}" != "0" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
fi

case "${SUITE_PROFILE}" in
  pilot)
    MAX_TRIALS="${MAX_TRIALS:-4}"
    MAX_FRAMES="${MAX_FRAMES:-20000}"
    MAX_SAMPLES="${MAX_SAMPLES:-20000}"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EVAL_EPISODES="${EVAL_EPISODES:-5}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-5000}"
    TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-50}"
    EVAL_STEP_LOG_INTERVAL="${EVAL_STEP_LOG_INTERVAL:-${EVAL_LOG_INTERVAL:-0}}"
    EVAL_EPISODE_LOG_INTERVAL="${EVAL_EPISODE_LOG_INTERVAL:-100}"
    NO_COMPRESSION="${NO_COMPRESSION:-1}"
    SPLIT_STRATEGY="${SPLIT_STRATEGY:-block}"
    VAL_FRACTION="${VAL_FRACTION:-0.1}"
    ;;
  full)
    MAX_TRIALS="${MAX_TRIALS:-0}"
    MAX_FRAMES="${MAX_FRAMES:-0}"
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    EPOCHS="${EPOCHS:-20}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-108000}"
    TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-100}"
    EVAL_STEP_LOG_INTERVAL="${EVAL_STEP_LOG_INTERVAL:-${EVAL_LOG_INTERVAL:-0}}"
    EVAL_EPISODE_LOG_INTERVAL="${EVAL_EPISODE_LOG_INTERVAL:-100}"
    NO_COMPRESSION="${NO_COMPRESSION:-0}"
    SPLIT_STRATEGY="${SPLIT_STRATEGY:-trial}"
    VAL_FRACTION="${VAL_FRACTION:-0.1}"
    ;;
  *)
    echo "Unsupported SUITE_PROFILE=${SUITE_PROFILE}. Use pilot or full."
    exit 1
    ;;
esac

CONTEXT_LENGTH="${CONTEXT_LENGTH:-30}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-1}"
AMP="${AMP:-1}"
LR="${LR:-1e-4}"
MASK_RATIO="${MASK_RATIO:-0.75}"
MAX_TIMESTEP="${MAX_TIMESTEP:-30000}"

if [ "${SETUP_ENV}" = "1" ]; then
  bash scripts/setup_environment.sh
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [ "${SKIP_PREPARE}" != "1" ]; then
  HF_REPO="${HF_REPO}" bash scripts/setup_atari_head_v4_data.sh "${GAME}"
  PREPARE_ARGS=(
    --game "${GAME}"
    --max-trials "${MAX_TRIALS}"
    --max-frames "${MAX_FRAMES}"
    --overwrite
    --atomic-output
  )
  if [ "${NO_COMPRESSION}" = "1" ]; then
    PREPARE_ARGS+=(--no-compression)
  fi
  python scripts/prepare_amsterg_hdf5.py "${PREPARE_ARGS[@]}"
fi

HDF5_PATH="${HDF5_PATH}" python -c "import h5py, os; path=os.environ['HDF5_PATH']; handle=h5py.File(path, 'r'); groups=[key for key in handle.keys() if key != 'combined']; print(f'hdf5_ok={path} groups={len(groups)}'); handle.close()"

run_condition() {
  local name="$1"
  local embed_dim="$2"
  local encoder_layers="$3"
  local encoder_heads="$4"
  local encoder_ff_dim="$5"
  local decoder_dim="$6"
  local decoder_layers="$7"
  local decoder_heads="$8"
  local decoder_ff_dim="$9"
  local dt_layers="${10}"
  local dt_heads="${11}"
  local mask_strategy="${12}"
  local lambda_rec="${13}"
  local lambda_gaze="${14}"
  local disable_rec="${15}"
  local seed="$16"

  local out_dir="${SUITE_ROOT}/${name}/seed_${seed}"
  local checkpoint="${out_dir}/active_dt.pt"
  local eval_json="artifacts/gymnasium_eval/${GAME}_${SUITE_PROFILE}_${name}_seed_${seed}.json"

  local train_args=(
    --mode active_dt
    --hdf5 "${HDF5_PATH}"
    --output-dir "${out_dir}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lr "${LR}"
    --context-length "${CONTEXT_LENGTH}"
    --split-strategy "${SPLIT_STRATEGY}"
    --val-fraction "${VAL_FRACTION}"
    --embed-dim "${embed_dim}"
    --encoder-layers "${encoder_layers}"
    --encoder-heads "${encoder_heads}"
    --encoder-ff-dim "${encoder_ff_dim}"
    --decoder-dim "${decoder_dim}"
    --decoder-layers "${decoder_layers}"
    --decoder-heads "${decoder_heads}"
    --decoder-ff-dim "${decoder_ff_dim}"
    --dt-layers "${dt_layers}"
    --dt-heads "${dt_heads}"
    --mask-ratio "${MASK_RATIO}"
    --mask-strategy "${mask_strategy}"
    --lambda-rec "${lambda_rec}"
    --lambda-gaze "${lambda_gaze}"
    --max-timestep "${MAX_TIMESTEP}"
    --seed "${seed}"
    --device "${DEVICE}"
    --require-rewards
    --log-interval "${TRAIN_LOG_INTERVAL}"
  )
  if [ -n "${MAX_SAMPLES}" ]; then
    train_args+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [ "${PIN_MEMORY}" = "1" ]; then
    train_args+=(--pin-memory)
  fi
  if [ "${AMP}" = "1" ]; then
    train_args+=(--amp)
  fi
  if [ "${disable_rec}" = "1" ]; then
    train_args+=(--disable-reconstruction)
  fi

  echo "=== train ${name} seed=${seed} ==="
  python scripts/train_active_gaze_dt.py "${train_args[@]}"

  echo "=== evaluate ${name} seed=${seed} ==="
  python scripts/evaluate_gymnasium_atari_policy.py \
    --game "${GAME}" \
    --model-type active-dt \
    --checkpoint "${checkpoint}" \
    --episodes "${EVAL_EPISODES}" \
    --max-steps "${EVAL_MAX_STEPS}" \
    --frameskip 1 \
    --policy "${EVAL_POLICY}" \
    --temperature 1.0 \
    --start-actions "${EVAL_START_ACTIONS}" \
    --context-length "${CONTEXT_LENGTH}" \
    --target-return 20 \
    --device "${DEVICE}" \
    --log-interval "${EVAL_STEP_LOG_INTERVAL}" \
    --episode-log-interval "${EVAL_EPISODE_LOG_INTERVAL}" \
    --output-json "${eval_json}"
}

for seed in ${SEEDS}; do
  run_condition active_gaze_d256 256 4 8 1024 256 2 8 1024 6 8 learned 1.0 0.1 0 "${seed}"
  run_condition random_mask_d256 256 4 8 1024 256 2 8 1024 6 8 random 1.0 0.0 0 "${seed}"
  run_condition active_gaze_no_rec_d256 256 4 8 1024 256 1 8 1024 6 8 learned 0.0 0.1 1 "${seed}"
  run_condition active_gaze_d128 128 2 4 512 128 1 4 512 4 4 learned 1.0 0.1 0 "${seed}"
done
