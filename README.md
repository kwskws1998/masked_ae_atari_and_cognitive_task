# Atari-HEAD Gaze C-MAE

This workspace now targets Atari-HEAD rather than EyeBench reading data.

Data sources:

- Trial-level Atari-HEAD record: https://zenodo.org/records/2603190
- Game-level Atari-HEAD v4 archives used by the amsterg/AHEAD pipeline: Zenodo record `3451402`

The Zenodo record describes Atari-HEAD as trial-level image archives plus label files:

> "For every game frame, its corresponding image frame, the human keystroke action, the reaction time to make that action, the gaze positions, and immediate reward returned by the environment were recorded."

It also states:

> "*.tar.bz2 files: contains game image frames."

and:

> "*.txt files: label file for each trial."

Downloaded datasets, HDF5 files, checkpoints, and experiment artifacts are intentionally excluded from git. Recreate them with the scripts below.

## Current Model

The first model is `AtariHeadGazeMAE`:

```text
Atari frame stack -> frame patch tokens
gaze heatmap -> gaze patch tokens -> random mask
frame + gaze + position tokens -> Transformer encoder
encoder tokens -> action head
encoder tokens -> masked gaze patch decoder
```

Training loss:

```text
L = L_action_CE + lambda * L_masked_gaze_reconstruction
```

This is intentionally behavior cloning first, not RL. RL fine-tuning can be added after the supervised Atari-HEAD path works.

## Download Selected Atari-HEAD Trials

Fetch Zenodo file metadata:

```bash
python scripts/download_atari_head.py manifest --out data/atari_head
```

Download one trial by prefix:

```bash
python scripts/download_atari_head.py trial --out data/atari_head --trial 100
```

This should download both `100_*.tar.bz2` and `100_*.txt` if they are present in the Zenodo manifest.

## Download Atari-HEAD v4 Game Archives

For the current Breakout/amsterg/active-DT path, download the game-level v4 archive instead of committing data to the repository.

The default setup path uses the private Hugging Face mirror `skboy/atari-head-v4` and downloads Breakout:

```bash
bash scripts/setup_atari_head_v4_data.sh
```

Download multiple mirrored games:

```bash
bash scripts/setup_atari_head_v4_data.sh breakout asterix seaquest
```

The setup script runs `hf auth login` if the local machine is not authenticated. To override the mirror or output path:

```bash
HF_REPO=skboy/atari-head-v4 \
ATARI_HEAD_V4_DIR=data/atari_head_full/v4 \
bash scripts/setup_atari_head_v4_data.sh breakout
```

The same Hugging Face download path is available through Python:

```bash
python scripts/download_atari_head_v4.py \
  --source hf \
  --hf-repo skboy/atari-head-v4 \
  --games breakout \
  --out data/atari_head_full/v4
```

If the Hugging Face mirror is unavailable, use the original Zenodo v4 record:

```bash
python scripts/download_atari_head_v4.py \
  --games breakout \
  --manifest-out data/atari_head_full/v4/zenodo_manifest.tsv \
  --out data/atari_head_full/v4
```

Multiple games can be selected by name:

```bash
python scripts/download_atari_head_v4.py \
  --games breakout asterix seaquest \
  --out data/atari_head_full/v4
```

To download the full v4 archive, use `--all`. This is several GiB and should stay outside git:

```bash
python scripts/download_atari_head_v4.py \
  --all \
  --out data/atari_head_full/v4
```

The downloader resumes partial files and verifies MD5 checksums by default.

## Smoke Test

The unit tests build a tiny synthetic Atari-HEAD-style tar archive and label file, so they do not require the real dataset:

```bash
python tests/test_atari_head_pipeline.py
```

## Amsterg/AHEAD Baseline Smoke

The amsterg/ahead code is vendored under `external/amsterg_ahead`. Prepare one Breakout trial into an amsterg-compatible HDF5 file:

```bash
python scripts/prepare_amsterg_hdf5.py \
  --game breakout \
  --trials 198_RZ_3877709_Dec-03-16-56-11 \
  --max-frames 96 \
  --overwrite \
  --combined \
  --no-compression
```

Run a short supervised smoke train over the original amsterg model classes:

```bash
python scripts/train_amsterg_models.py \
  --game breakout \
  --groups 198_RZ_3877709_Dec-03-16-56-11 \
  --max-samples 32 \
  --epochs 1 \
  --batch-size 8 \
  --models gaze,bc,agil,sea \
  --output-dir artifacts/amsterg_runs/smoke \
  --device cpu
```

To test AGIL/SEA with a predicted gaze model:

```bash
python scripts/train_amsterg_models.py \
  --game breakout \
  --groups 198_RZ_3877709_Dec-03-16-56-11 \
  --max-samples 24 \
  --epochs 1 \
  --batch-size 8 \
  --models agil,sea \
  --gaze-source predicted \
  --gaze-checkpoint artifacts/amsterg_runs/smoke/cnn_gaze.pt \
  --output-dir artifacts/amsterg_runs/predicted_smoke \
  --device cpu
```

## Atari Environment Smoke

Offline Atari-HEAD training does not need an emulator, but game-score evaluation does. The original amsterg code uses legacy Gym ids such as `Breakout-v0`; this workspace keeps that code intact and uses the current Gymnasium/ALE id for new evaluation code:

Set up the Python environment first:

```bash
bash scripts/setup_environment.sh
```

By default this also downloads the mirrored Breakout archive from `skboy/atari-head-v4` into `data/atari_head_full/v4`. To download multiple games during setup:

```bash
DATA_GAMES="breakout asterix seaquest" bash scripts/setup_environment.sh
```

To install dependencies without downloading data:

```bash
DOWNLOAD_DATA=0 bash scripts/setup_environment.sh
```

On a CUDA machine such as an RTX 3090, the default setup installs PyTorch from the CUDA 12.1 wheel index. If PyTorch is already installed, skip that part:

```bash
INSTALL_TORCH=0 bash scripts/setup_environment.sh
```

```bash
python scripts/smoke_gymnasium_atari.py \
  --env-id ALE/Breakout-v5 \
  --steps 32 \
  --frameskip 1 \
  --full-action-space
```

`--full-action-space` keeps the action space at 18 actions, matching Atari-HEAD action labels and the amsterg model heads.

Evaluate a saved BC checkpoint in the same Gymnasium/ALE environment:

```bash
python scripts/evaluate_gymnasium_atari_policy.py \
  --game breakout \
  --model-type bc \
  --checkpoint artifacts/amsterg_runs/smoke/bc.pt \
  --episodes 1 \
  --max-steps 64 \
  --frameskip 1 \
  --policy sample \
  --temperature 1.0 \
  --output-json artifacts/gymnasium_eval/bc_breakout_sample_smoke.json
```

Evaluate a gaze-augmented checkpoint:

```bash
python scripts/evaluate_gymnasium_atari_policy.py \
  --game breakout \
  --model-type sea \
  --checkpoint artifacts/amsterg_runs/smoke/sea.pt \
  --gaze-checkpoint artifacts/amsterg_runs/smoke/cnn_gaze.pt \
  --episodes 1 \
  --max-steps 64 \
  --frameskip 1 \
  --policy sample \
  --temperature 1.0 \
  --output-json artifacts/gymnasium_eval/sea_breakout_sample_smoke.json
```

For paper-style game-score evaluation, increase `--episodes` and set `--max-steps 108000`. The smoke checkpoints above are intentionally tiny, so reward is not meaningful yet; use them only to verify the emulator-policy path.

## Active-Gaze Masked Decision Transformer

The official Decision Transformer reference files are preserved under `external/decision_transformer`. The native implementation in this workspace ports the causal minGPT-style trajectory model, but replaces the official Atari CNN state encoder with:

```text
Atari frame stack
-> active gaze-supervised mask policy
-> visible 25% MAE encoder
-> state embedding
-> Decision Transformer
-> action logits
```

Rebuild the Breakout smoke HDF5 with reward and episode metadata:

```bash
python scripts/prepare_amsterg_hdf5.py \
  --game breakout \
  --trials 198_RZ_3877709_Dec-03-16-56-11 \
  --max-frames 96 \
  --overwrite \
  --combined \
  --no-compression
```

Run a small one-step active-gaze behavior cloning smoke train:

```bash
python scripts/train_active_gaze_dt.py \
  --mode active_bc \
  --groups 198_RZ_3877709_Dec-03-16-56-11 \
  --max-samples 16 \
  --split-strategy block \
  --context-length 4 \
  --epochs 1 \
  --batch-size 2 \
  --embed-dim 32 \
  --encoder-layers 1 \
  --encoder-ff-dim 64 \
  --decoder-dim 32 \
  --decoder-layers 1 \
  --decoder-ff-dim 64 \
  --dt-layers 1 \
  --output-dir artifacts/active_gaze_dt/smoke \
  --device cpu
```

Run the active-gaze Decision Transformer smoke train:

```bash
python scripts/train_active_gaze_dt.py \
  --mode active_dt \
  --groups 198_RZ_3877709_Dec-03-16-56-11 \
  --max-samples 16 \
  --split-strategy block \
  --context-length 4 \
  --epochs 1 \
  --batch-size 2 \
  --embed-dim 32 \
  --encoder-layers 1 \
  --encoder-ff-dim 64 \
  --decoder-dim 32 \
  --decoder-layers 1 \
  --decoder-ff-dim 64 \
  --dt-layers 1 \
  --output-dir artifacts/active_gaze_dt/smoke \
  --device cpu
```

Evaluate the active-DT checkpoint in Gymnasium/ALE:

```bash
python scripts/evaluate_gymnasium_atari_policy.py \
  --game breakout \
  --model-type active-dt \
  --checkpoint artifacts/active_gaze_dt/smoke/active_dt.pt \
  --episodes 1 \
  --max-steps 64 \
  --frameskip 1 \
  --policy sample \
  --temperature 1.0 \
  --context-length 4 \
  --target-return 20 \
  --output-json artifacts/gymnasium_eval/active_dt_breakout_sample_smoke.json
```

To run the full active-DT Breakout pipeline from data download through evaluation:

```bash
PROFILE=smoke bash scripts/run_breakout_active_dt_experiment.sh
PROFILE=pilot bash scripts/run_breakout_active_dt_experiment.sh
PROFILE=real bash scripts/run_breakout_active_dt_experiment.sh
PROFILE=rtx3090 bash scripts/run_breakout_active_dt_experiment.sh
```

`PROFILE=smoke` verifies the path on a tiny sample. `PROFILE=pilot` uses several trials and a default-width model. `PROFILE=real` processes all selected Breakout data, trains with context length 30, and evaluates 30 episodes with the paper-style 108K frame cutoff. `PROFILE=rtx3090` uses a wider 256-dim model, deeper encoder/DT, CUDA AMP, and 8-head attention for a 24GB GPU. Smoke and pilot runs use contiguous `block` splits with a purge gap between train/validation/test windows. Real and RTX 3090 runs use held-out `trial` splits and report final `test_action_acc` separately from Gymnasium game score.

For a 24GB RTX 3090, start with the default `rtx3090` profile. If VRAM is still underused, increase batch size:

```bash
PROFILE=rtx3090 BATCH_SIZE=96 bash scripts/run_breakout_active_dt_experiment.sh
PROFILE=rtx3090 BATCH_SIZE=128 bash scripts/run_breakout_active_dt_experiment.sh
```

If it OOMs, reduce batch size first:

```bash
PROFILE=rtx3090 BATCH_SIZE=32 bash scripts/run_breakout_active_dt_experiment.sh
```

For full-data runs, the HDF5 preprocessing output can be much larger than the original zip because frame stacks and gaze tensors are materialized. The run script does not create the duplicated `combined` HDF5 group by default; set `COMBINED=1` only if another baseline explicitly needs it.

For paper-facing comparisons, run matched-width ablations rather than only one large model:

```bash
# Quick check on a rented 3090.
SUITE_PROFILE=pilot bash scripts/run_rtx3090_ablation_suite.sh

# Full Breakout ablation suite.
SUITE_PROFILE=full bash scripts/run_rtx3090_ablation_suite.sh
```

The suite prepares the data once, then runs:

```text
active_gaze_d256
random_mask_d256
active_gaze_no_rec_d256
active_gaze_d128
```

The first three use the same 256-dim width, so the main comparison isolates the masking/reconstruction choices. The 128-dim run is a size ablation to check whether gains are only from model capacity.

For more aggressive 3090 use:

```bash
SUITE_PROFILE=full BATCH_SIZE=96 bash scripts/run_rtx3090_ablation_suite.sh
SUITE_PROFILE=full BATCH_SIZE=128 bash scripts/run_rtx3090_ablation_suite.sh
```

## First Real Experiment

Use one or two downloaded trials first:

```bash
python scripts/train_atari_head_bc.py \
  --frame-archive data/atari_head/100_RZ_3592991_Aug-24-11-44-38.tar.bz2 \
  --label-file data/atari_head/100_RZ_3592991_Aug-24-11-44-38.txt \
  --max-samples 1024 \
  --epochs 1
```

The first comparison should be:

1. Frame-only behavior cloning
2. Frame plus observed gaze heatmap
3. Frame-conditioned masked gaze reconstruction plus behavior cloning

Only after this path is stable should we add AGIL/SEA-style predicted-gaze baselines or RL fine-tuning.
