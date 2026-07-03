"""Trajectory-window datasets for Atari-HEAD HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryMetadata(NamedTuple):
    rtg: np.ndarray
    timesteps: np.ndarray
    episode_ids: np.ndarray
    rewards: np.ndarray


def select_hdf5_groups(hdf5_path: str | Path, groups: list[str] | None = None) -> list[str]:
    """Return explicit groups or all non-combined trial groups from an HDF5 file."""

    with h5py.File(hdf5_path, "r") as handle:
        selected = sorted(handle.keys()) if groups is None else groups
        selected = [group for group in selected if group != "combined"]
        missing = [group for group in selected if group not in handle]
    if missing:
        raise KeyError(f"missing groups in {hdf5_path}: {missing}")
    if not selected:
        raise ValueError(f"no usable trial groups found in {hdf5_path}")
    return selected


def _last_column(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    return array[:, -1]


def _compute_group_metadata(group: h5py.Group) -> TrajectoryMetadata:
    length = int(group["actions"].shape[0])
    if "rewards" in group:
        rewards = _last_column(group["rewards"][:]).astype(np.float32)
    else:
        rewards = np.zeros(length, dtype=np.float32)
    if "episode_ids" in group:
        episode_ids = _last_column(group["episode_ids"][:]).astype(np.int64)
    else:
        episode_ids = np.zeros(length, dtype=np.int64)

    rtg = np.zeros(length, dtype=np.float32)
    running = 0.0
    for index in range(length - 1, -1, -1):
        if index == length - 1 or episode_ids[index] != episode_ids[index + 1]:
            running = 0.0
        running += float(rewards[index])
        rtg[index] = running

    timesteps = np.zeros(length, dtype=np.int64)
    counter = 0
    for index in range(length):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            counter = 0
        timesteps[index] = counter
        counter += 1

    return TrajectoryMetadata(
        rtg=rtg,
        timesteps=timesteps,
        episode_ids=episode_ids,
        rewards=rewards,
    )


class AtariHeadHDF5TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length trajectory windows from an amsterg-compatible Atari-HEAD HDF5 file."""

    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        groups: list[str] | None = None,
        context_length: int = 8,
        max_samples: int | None = None,
        require_rewards: bool = False,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.hdf5_path = Path(hdf5_path)
        self.groups = select_hdf5_groups(self.hdf5_path, groups)
        self.context_length = context_length
        self.samples: list[tuple[str, int]] = []
        self.metadata: dict[str, TrajectoryMetadata] = {}
        self._handle: h5py.File | None = None

        with h5py.File(self.hdf5_path, "r") as handle:
            for group_name in self.groups:
                group = handle[group_name]
                if require_rewards and "rewards" not in group:
                    raise KeyError(f"group {group_name!r} has no rewards dataset")
                metadata = _compute_group_metadata(group)
                self.metadata[group_name] = metadata
                length = int(group["actions"].shape[0])
                for start in range(0, length - context_length + 1):
                    stop = start + context_length
                    if np.all(metadata.episode_ids[start:stop] == metadata.episode_ids[start]):
                        self.samples.append((group_name, start))
                        if max_samples is not None and len(self.samples) >= max_samples:
                            return

        if not self.samples:
            raise ValueError("no trajectory windows were produced")

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        group_name, start = self.samples[index]
        stop = start + self.context_length
        group = self.handle[group_name]
        metadata = self.metadata[group_name]

        frames = torch.from_numpy(group["images"][start:stop]).to(torch.float32)
        actions_np = _last_column(group["actions"][start:stop]).astype(np.int64)
        gazes = torch.from_numpy(group["gazes"][start:stop]).to(torch.float32).unsqueeze(1)
        rewards = torch.from_numpy(metadata.rewards[start:stop]).to(torch.float32)
        rtg = torch.from_numpy(metadata.rtg[start:stop]).to(torch.float32)
        timesteps = torch.from_numpy(metadata.timesteps[start:stop]).to(torch.long)

        return {
            "frames": frames,
            "actions": torch.from_numpy(actions_np).to(torch.long),
            "rewards": rewards,
            "rtg": rtg,
            "gazes": gazes,
            "timesteps": timesteps,
        }
