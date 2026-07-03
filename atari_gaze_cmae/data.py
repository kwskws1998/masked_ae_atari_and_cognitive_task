"""Atari-HEAD label parsing, frame loading, and gaze heatmap construction."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import csv
import math
import tarfile
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AtariHeadLabel:
    frame_id: str
    action: int
    gaze_points: tuple[tuple[float, float], ...]
    episode_id: int | None = None
    score: int | None = None
    duration_ms: int | None = None
    unclipped_reward: int | None = None


def _is_header(parts: list[str]) -> bool:
    lower = {part.strip().lower() for part in parts}
    return "frame_id" in lower or "action" in lower


def _split_label_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if "," in line:
        return next(csv.reader([line]))
    return line.split()


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_gaze_points(values: Iterable[str]) -> tuple[tuple[float, float], ...]:
    joined = ",".join(v.strip() for v in values if v.strip())
    if joined.lower() in {"", "null", "none", "nan", "[]"}:
        return ()
    for token in "[]()":
        joined = joined.replace(token, "")
    pieces = [p.strip() for p in joined.replace(";", ",").split(",") if p.strip()]
    coords: list[float] = []
    for piece in pieces:
        try:
            coords.append(float(piece))
        except ValueError:
            continue
    return tuple((coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2))


def _record_from_parts(parts: list[str], header: list[str] | None) -> AtariHeadLabel | None:
    if not parts:
        return None
    if header:
        fields = {name.strip().lower(): idx for idx, name in enumerate(header)}
        frame_idx = fields.get("frame_id", 0)
        action_idx = fields.get("action")
        if action_idx is None:
            return None
        frame_id = parts[frame_idx].strip()
        action = _parse_optional_int(parts[action_idx])
        gaze_start = fields.get("gaze_positions")
        gaze_values = parts[gaze_start:] if gaze_start is not None else []
        return AtariHeadLabel(
            frame_id=frame_id,
            action=-1 if action is None else action,
            gaze_points=_parse_gaze_points(gaze_values),
            episode_id=_parse_optional_int(parts[fields["episode_id"]])
            if "episode_id" in fields and fields["episode_id"] < len(parts)
            else None,
            score=_parse_optional_int(parts[fields["score"]])
            if "score" in fields and fields["score"] < len(parts)
            else None,
            duration_ms=_parse_optional_int(parts[fields["duration(ms)"]])
            if "duration(ms)" in fields and fields["duration(ms)"] < len(parts)
            else None,
            unclipped_reward=_parse_optional_int(parts[fields["unclipped_reward"]])
            if "unclipped_reward" in fields and fields["unclipped_reward"] < len(parts)
            else None,
        )

    if len(parts) < 6:
        return None
    frame_id = parts[0].strip()
    action = _parse_optional_int(parts[5])
    return AtariHeadLabel(
        frame_id=frame_id,
        episode_id=_parse_optional_int(parts[1]),
        score=_parse_optional_int(parts[2]),
        duration_ms=_parse_optional_int(parts[3]),
        unclipped_reward=_parse_optional_int(parts[4]),
        action=-1 if action is None else action,
        gaze_points=_parse_gaze_points(parts[6:]),
    )


def read_atari_head_labels(
    label_file: str | Path,
    *,
    max_rows: int | None = None,
) -> list[AtariHeadLabel]:
    path = Path(label_file)
    labels: list[AtariHeadLabel] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = _split_label_line(line)
            if not parts:
                continue
            if header is None and _is_header(parts):
                header = parts
                continue
            record = _record_from_parts(parts, header)
            if record is None or record.action < 0:
                continue
            labels.append(record)
            if max_rows is not None and len(labels) >= max_rows:
                break
    return labels


def gaze_points_to_heatmap(
    gaze_points: tuple[tuple[float, float], ...],
    *,
    source_width: int,
    source_height: int,
    target_width: int = 84,
    target_height: int = 84,
    sigma: float = 2.0,
) -> np.ndarray:
    heatmap = np.zeros((target_height, target_width), dtype=np.float32)
    if not gaze_points:
        return heatmap

    radius = max(1, int(math.ceil(3.0 * sigma)))
    for x_src, y_src in gaze_points:
        x = x_src * (target_width / source_width)
        y = y_src * (target_height / source_height)
        x0 = max(0, int(math.floor(x)) - radius)
        x1 = min(target_width, int(math.floor(x)) + radius + 1)
        y0 = max(0, int(math.floor(y)) - radius)
        y1 = min(target_height, int(math.floor(y)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        heatmap[y0:y1, x0:x1] += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

    total = float(heatmap.sum())
    if total > 0.0:
        heatmap /= total
    return heatmap


class AtariHeadTrialDataset(Dataset):
    """Lazy trial reader for Atari-HEAD frame archives and matching label files."""

    def __init__(
        self,
        frame_archive: str | Path,
        label_file: str | Path,
        *,
        history: int = 4,
        image_size: tuple[int, int] = (84, 84),
        gaze_sigma: float = 2.0,
        skip_no_gaze: bool = False,
        max_rows: int | None = None,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be positive")
        self.frame_archive = Path(frame_archive)
        self.label_file = Path(label_file)
        self.history = history
        self.image_width, self.image_height = image_size
        self.gaze_sigma = gaze_sigma
        labels = read_atari_head_labels(self.label_file, max_rows=max_rows)
        if skip_no_gaze:
            labels = [label for label in labels if label.gaze_points]
        if len(labels) < history:
            raise ValueError("not enough labels for requested history")
        self.labels = labels
        self.indices = list(range(history - 1, len(labels)))
        self._tar: tarfile.TarFile | None = None
        self._member_index: dict[str, tarfile.TarInfo] | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
        self._tar = None
        self._member_index = None

    def _ensure_tar(self) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        if self._tar is None or self._member_index is None:
            self._tar = tarfile.open(self.frame_archive, mode="r:bz2")
            self._member_index = {}
            for member in self._tar.getmembers():
                if not member.isfile():
                    continue
                path = Path(member.name)
                keys = {path.name, path.stem, member.name}
                for key in keys:
                    self._member_index.setdefault(key, member)
        return self._tar, self._member_index

    def _load_frame(self, frame_id: str) -> tuple[np.ndarray, tuple[int, int]]:
        tar, member_index = self._ensure_tar()
        member = (
            member_index.get(frame_id)
            or member_index.get(Path(frame_id).name)
            or member_index.get(Path(frame_id).stem)
        )
        if member is None:
            raise KeyError(f"frame_id={frame_id!r} was not found in {self.frame_archive}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise KeyError(f"could not extract frame_id={frame_id!r}")
        image = Image.open(BytesIO(extracted.read())).convert("L")
        source_size = image.size
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        frame = np.asarray(image, dtype=np.float32) / 255.0
        return frame, source_size

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        label_idx = self.indices[item]
        current = self.labels[label_idx]
        frame_ids = [self.labels[i].frame_id for i in range(label_idx - self.history + 1, label_idx + 1)]
        frames = []
        source_size: tuple[int, int] | None = None
        for frame_id in frame_ids:
            frame, source_size = self._load_frame(frame_id)
            frames.append(frame)
        assert source_size is not None
        source_width, source_height = source_size
        heatmap = gaze_points_to_heatmap(
            current.gaze_points,
            source_width=source_width,
            source_height=source_height,
            target_width=self.image_width,
            target_height=self.image_height,
            sigma=self.gaze_sigma,
        )
        return {
            "frames": torch.from_numpy(np.stack(frames, axis=0)),
            "gaze_heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "action": torch.tensor(current.action, dtype=torch.long),
            "has_gaze": torch.tensor(bool(current.gaze_points), dtype=torch.bool),
        }


def collate_atari_head_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "frames": torch.stack([sample["frames"] for sample in samples]),
        "gaze_heatmaps": torch.stack([sample["gaze_heatmap"] for sample in samples]),
        "actions": torch.stack([sample["action"] for sample in samples]),
        "has_gaze": torch.stack([sample["has_gaze"] for sample in samples]),
    }
