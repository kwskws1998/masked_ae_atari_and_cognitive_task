"""Prepare Atari-HEAD v4 data for the amsterg/ahead model code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AHEAD_ROOT = ROOT / "external" / "amsterg_ahead"
DEFAULT_SOURCE_DIR = ROOT / "data" / "atari_head_full" / "v4"

VALID_ACTIONS: dict[str, list[int]] = {
    "alien": list(range(1, 18)),
    "asterix": [2, 3, 4, 5, 6, 7, 8, 9],
    "bank_heist": list(range(1, 18)),
    "berzerk": list(range(1, 18)),
    "breakout": [1, 3, 4],
    "centipede": list(range(1, 18)),
    "demon_attack": [1, 3, 4, 11, 12],
    "enduro": [1, 3, 4, 5, 8, 9, 11, 12],
    "freeway": [2, 5],
    "frostbite": list(range(1, 18)),
    "hero": list(range(1, 18)),
    "montezuma_revenge": list(range(1, 18)),
    "ms_pacman": [2, 3, 4, 5, 6, 7, 8, 9],
    "name_this_game": [1, 3, 4, 11, 12],
    "phoenix": [1, 3, 4, 5, 11, 12, 13],
    "riverraid": list(range(1, 18)),
    "road_runner": list(range(1, 18)),
    "seaquest": list(range(1, 18)),
    "space_invaders": [1, 3, 4, 11, 12],
    "venture": list(range(1, 18)),
}


@dataclass(frozen=True)
class TrialSpec:
    name: str
    raw_dir: Path


def configure_ahead(ahead_root: Path) -> None:
    required = ahead_root / "src" / "data" / "data_utils.py"
    if not required.exists():
        raise FileNotFoundError(
            "missing amsterg/ahead data utilities at "
            f"{required}. Fetch the complete repository or restore "
            "external/amsterg_ahead/src/data before preparing HDF5 files."
        )
    os.environ.setdefault("MPLCONFIGDIR", str(ahead_root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ahead_root / ".cache"))
    sys.path.insert(0, str(ahead_root))
    os.chdir(ahead_root)


def extract_game_zip(game: str, source_dir: Path, ahead_root: Path) -> None:
    raw_root = ahead_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    action_src = source_dir / "action_enums.txt"
    if action_src.exists():
        (raw_root / "action_enums.txt").write_bytes(action_src.read_bytes())

    game_dir = raw_root / game
    if game_dir.exists():
        return
    zip_path = source_dir / f"{game}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing Atari-HEAD game zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (raw_root / member.filename).resolve()
            if raw_root.resolve() not in target.parents and target != raw_root.resolve():
                raise RuntimeError(f"refusing unsafe zip member: {member.filename}")
        archive.extractall(raw_root)


def discover_trials(raw_game_dir: Path) -> list[TrialSpec]:
    trials = []
    for label_file in sorted(raw_game_dir.rglob("*.txt")):
        trial = label_file.stem
        archive = label_file.with_suffix(".tar.bz2")
        if archive.exists():
            trials.append(TrialSpec(name=trial, raw_dir=label_file.parent))
    return trials


def prepare_interim_trial(game: str, trial: TrialSpec, ahead_root: Path, overwrite_gaze: bool) -> None:
    from src.data.data_utils import process_gaze_data

    interim_game_dir = ahead_root / "data" / "interim" / game
    trial_dir = interim_game_dir / trial.name
    interim_game_dir.mkdir(parents=True, exist_ok=True)
    if not trial_dir.exists():
        subprocess.run(
            ["tar", "-xjf", str(trial.raw_dir / f"{trial.name}.tar.bz2"), "-C", str(interim_game_dir)],
            check=True,
        )
    gaze_csv = trial_dir / f"{trial.name}_gaze_data.csv"
    if overwrite_gaze or not gaze_csv.exists():
        process_gaze_data(
            str(trial.raw_dir / f"{trial.name}.txt"),
            str(gaze_csv),
            VALID_ACTIONS[game],
        )


def build_group_arrays(game: str, trial: str, stack: int, max_frames: int | None) -> dict[str, np.ndarray]:
    from src.data.data_loaders import load_action_data, load_gaze_data
    from src.features.feat_utils import fuse_gazes, reduce_gaze_stack, transform_images

    till_ix = -1 if max_frames is None else max_frames
    images, actions = load_action_data(stack=stack, from_ix=0, till_ix=till_ix, game=game, game_run=trial)
    _, gazes = load_gaze_data(stack=stack, from_ix=0, till_ix=till_ix, game=game, game_run=trial, skip_images=True)
    if not images:
        raise RuntimeError(f"no stacked samples produced for {game}/{trial}")

    image_tensor = transform_images(images, type="torch").to(torch.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    gaze_pdf = torch.stack([reduce_gaze_stack(gaze_stack) for gaze_stack in gazes]).to(torch.float32)
    gaze_fused = fuse_gazes(image_tensor, gazes, gaze_count=1).to(torch.float32)
    rewards, episode_ids = build_stacked_metadata(game, trial, stack=stack, from_ix=0, till_ix=till_ix)
    if rewards.shape[0] != action_array.shape[0]:
        raise RuntimeError(
            f"metadata/action length mismatch for {game}/{trial}: "
            f"rewards={rewards.shape[0]} actions={action_array.shape[0]}"
        )

    return {
        "images": image_tensor.numpy(),
        "actions": action_array,
        "gazes": gaze_pdf.numpy(),
        "gazes_fused_noop": gaze_fused.numpy(),
        "rewards": rewards,
        "episode_ids": episode_ids,
    }


def build_stacked_metadata(
    game: str,
    trial: str,
    *,
    stack: int,
    from_ix: int,
    till_ix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read reward and episode columns from the processed gaze CSV and stack them like actions."""

    gaze_csv = Path("data") / "interim" / game / trial / f"{trial}_gaze_data.csv"
    data = pd.read_csv(gaze_csv)
    rewards = data["unclipped_reward"].fillna(0).astype(np.float32).tolist()[from_ix:till_ix]
    episode_ids = data["episode_id"].fillna(0).astype(np.int64).tolist()[from_ix:till_ix]
    reward_stacks = []
    episode_stacks = []
    for index in range(0, len(rewards) - stack):
        reward_stacks.append(rewards[index : index + stack])
        episode_stacks.append(episode_ids[index : index + stack])
    return (
        np.asarray(reward_stacks, dtype=np.float32),
        np.asarray(episode_stacks, dtype=np.int64),
    )


def write_groups(
    hdf5_path: Path,
    game: str,
    trials: list[TrialSpec],
    stack: int,
    max_frames: int | None,
    overwrite: bool,
    compression: str | None,
) -> list[str]:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with h5py.File(hdf5_path, "a") as handle:
        for trial in trials:
            if trial.name in handle:
                if not overwrite:
                    print(f"skip existing group: {trial.name}")
                    written.append(trial.name)
                    continue
                del handle[trial.name]
            arrays = build_group_arrays(game, trial.name, stack=stack, max_frames=max_frames)
            group = handle.create_group(trial.name)
            for key, value in arrays.items():
                group.create_dataset(key, data=value, compression=compression)
            written.append(trial.name)
            print(
                f"wrote {trial.name}: images={arrays['images'].shape} "
                f"actions={arrays['actions'].shape} gazes={arrays['gazes'].shape}"
            )
    return written


def write_combined_group(hdf5_path: Path, groups: list[str], overwrite: bool, compression: str | None) -> None:
    with h5py.File(hdf5_path, "a") as handle:
        if "combined" in handle:
            if not overwrite:
                print("skip existing group: combined")
                return
            del handle["combined"]
        combined = handle.create_group("combined")
        keys = ["images", "actions", "gazes", "gazes_fused_noop", "rewards", "episode_ids"]
        keys = [key for key in keys if all(key in handle[group] for group in groups)]
        for key in keys:
            arrays = [handle[group][key][:] for group in groups]
            combined.create_dataset(key, data=np.concatenate(arrays, axis=0), compression=compression)
        print(f"wrote combined: n={combined['actions'].shape[0]}")


def prepare_hdf5_output(
    hdf5_path: Path,
    game: str,
    trials: list[TrialSpec],
    *,
    stack: int,
    max_frames: int | None,
    overwrite: bool,
    compression: str | None,
    combined: bool,
    atomic_output: bool,
) -> list[str]:
    output_path = hdf5_path
    temp_path: Path | None = None
    if atomic_output:
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = hdf5_path.with_name(f".{hdf5_path.stem}.{os.getpid()}.tmp{hdf5_path.suffix}")
        if temp_path.exists():
            temp_path.unlink()
        output_path = temp_path
        overwrite = True

    try:
        written = write_groups(
            output_path,
            game,
            trials,
            stack=stack,
            max_frames=max_frames,
            overwrite=overwrite,
            compression=compression,
        )
        if combined:
            write_combined_group(output_path, written, overwrite=overwrite, compression=compression)
        if temp_path is not None:
            temp_path.replace(hdf5_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="breakout", choices=sorted(VALID_ACTIONS))
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--ahead-root", type=Path, default=DEFAULT_AHEAD_ROOT)
    parser.add_argument("--trials", nargs="*", help="Trial ids to process. Defaults to sorted trials.")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=256, help="Use 0 or a negative value for all frames.")
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-gaze", action="store_true")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument(
        "--atomic-output",
        action="store_true",
        help="Write a temporary HDF5 file first, then replace the final output after success.",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None

    extract_game_zip(args.game, args.source_dir, args.ahead_root)
    configure_ahead(args.ahead_root)

    raw_game_dir = args.ahead_root / "data" / "raw" / args.game
    trials = discover_trials(raw_game_dir)
    if args.trials:
        wanted = set(args.trials)
        trials = [trial for trial in trials if trial.name in wanted]
        missing = sorted(wanted - {trial.name for trial in trials})
        if missing:
            raise SystemExit(f"missing requested trials: {missing}")
    elif args.max_trials is not None and args.max_trials > 0:
        trials = trials[: args.max_trials]
    if not trials:
        raise SystemExit(f"no trials selected for {args.game}")

    for trial in trials:
        prepare_interim_trial(args.game, trial, args.ahead_root, overwrite_gaze=args.overwrite_gaze)

    hdf5_path = args.ahead_root / "data" / "processed" / f"{args.game}.hdf5"
    compression = None if args.no_compression else "gzip"
    prepare_hdf5_output(
        hdf5_path,
        args.game,
        trials,
        stack=args.stack,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        compression=compression,
        combined=args.combined,
        atomic_output=args.atomic_output,
    )
    print(hdf5_path)


if __name__ == "__main__":
    main()
