"""Train amsterg/ahead gaze and imitation baselines from prepared Atari-HEAD HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AHEAD_ROOT = ROOT / "external" / "amsterg_ahead"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "amsterg_runs" / "smoke"


class AtariHeadHDF5Dataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy HDF5 dataset for amsterg-compatible Atari-HEAD groups."""

    def __init__(self, hdf5_path: Path, groups: list[str], max_samples: int | None = None) -> None:
        self.hdf5_path = hdf5_path
        self.groups = groups
        self.samples: list[tuple[str, int]] = []
        self._handle: h5py.File | None = None
        with h5py.File(hdf5_path, "r") as handle:
            for group in groups:
                if group not in handle:
                    raise KeyError(f"group {group!r} not found in {hdf5_path}")
                length = int(handle[group]["actions"].shape[0])
                for index in range(length):
                    self.samples.append((group, index))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        return

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        group, row = self.samples[index]
        data = self.handle[group]
        image = torch.from_numpy(data["images"][row]).to(torch.float32)
        action_stack = torch.from_numpy(data["actions"][row]).to(torch.long)
        gaze = torch.from_numpy(data["gazes"][row]).to(torch.float32)
        gaze_fused = torch.from_numpy(data["gazes_fused_noop"][row]).to(torch.float32)
        return {
            "images": image,
            "actions": action_stack[-1],
            "gazes": gaze,
            "gazes_fused_noop": gaze_fused,
        }


def configure_ahead(ahead_root: Path) -> None:
    """Put amsterg/ahead imports and relative config paths on the active Python path."""

    os.environ.setdefault("MPLCONFIGDIR", str(ahead_root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ahead_root / ".cache"))
    sys.path.insert(0, str(ahead_root))
    os.chdir(ahead_root)


def parse_model_list(value: str) -> list[str]:
    """Parse and validate the comma-separated model list."""

    allowed = {"gaze", "bc", "agil", "sea"}
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model names: {unknown}")
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def select_groups(hdf5_path: Path, requested: list[str] | None) -> list[str]:
    """Return explicit groups or all trial groups, excluding the synthetic combined group."""

    with h5py.File(hdf5_path, "r") as handle:
        groups = sorted(handle.keys()) if requested is None else requested
        groups = [group for group in groups if group != "combined"]
        missing = [group for group in groups if group not in handle]
    if missing:
        raise SystemExit(f"missing groups in {hdf5_path}: {missing}")
    if not groups:
        raise SystemExit(f"no trainable groups found in {hdf5_path}")
    return groups


def split_indices(length: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Create deterministic sample-level train/validation indices."""

    if length < 2:
        raise ValueError("dataset needs at least two samples for train/validation split")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    split = int(round(length * train_fraction))
    split = min(max(split, 1), length - 1)
    return indices[:split], indices[split:]


def make_loader(
    dataset: AtariHeadHDF5Dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a DataLoader over a stable subset of the HDF5 dataset."""

    drop_last = shuffle and len(indices) >= batch_size * 2
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
    )


def normalized_gaze_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Convert CNN_GAZE log probabilities into per-sample normalized 84x84 maps."""

    gaze = torch.exp(log_probs).view(-1, 84, 84)
    flat = gaze.flatten(1)
    min_values = flat.min(dim=1).values.view(-1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1)
    gaze = (gaze - min_values) / (max_values - min_values + 1e-8)
    return gaze / (gaze.flatten(1).sum(dim=1).view(-1, 1, 1) + 1e-8)


def agil_overlay(
    frames: torch.Tensor,
    batch: dict[str, torch.Tensor],
    gaze_source: str,
    gaze_model: nn.Module | None,
) -> torch.Tensor:
    """Create the four-frame gaze overlay consumed by GAZED_ACTION_SL."""

    if gaze_source == "true":
        return batch["gazes_fused_noop"].to(frames.device)
    if gaze_model is None:
        raise RuntimeError("predicted gaze source requires a trained or loaded gaze model")
    with torch.no_grad():
        gaze = normalized_gaze_from_log_probs(gaze_model(frames))
    return frames * gaze.unsqueeze(1).repeat(1, frames.shape[1], 1, 1)


def sea_inputs(
    frames: torch.Tensor,
    batch: dict[str, torch.Tensor],
    gaze_source: str,
    gaze_model: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one-frame inputs consumed by SGAZED_ACTION_SL."""

    last_frame = frames[:, -1:].contiguous()
    if gaze_source == "true":
        return last_frame, batch["gazes_fused_noop"].to(frames.device)[:, -1:].contiguous()
    if gaze_model is None:
        raise RuntimeError("predicted gaze source requires a trained or loaded gaze model")
    with torch.no_grad():
        gaze = normalized_gaze_from_log_probs(gaze_model(frames)).unsqueeze(1)
    return last_frame, (last_frame * gaze).contiguous()


def summarize_class_distribution(dataset: AtariHeadHDF5Dataset, indices: list[int]) -> dict[str, int]:
    """Count final-frame action labels for a subset."""

    counts: dict[str, int] = {}
    for index in indices:
        group, row = dataset.samples[index]
        action = int(dataset.handle[group]["actions"][row][-1])
        counts[str(action)] = counts.get(str(action), 0) + 1
    return counts


def train_gaze_model(
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    val_loader: DataLoader[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Train CNN_GAZE with KL divergence against Atari-HEAD gaze heatmaps."""

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            frames = batch["images"].to(args.device)
            gaze = batch["gazes"].to(args.device)
            target = gaze.reshape(gaze.shape[0], -1)
            optimizer.zero_grad()
            output = model(frames)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * frames.shape[0]
            train_count += frames.shape[0]

        val_metrics = evaluate_gaze_model(model, val_loader, args.device)
        metrics = {
            "epoch": epoch,
            "train_kl": train_loss / max(train_count, 1),
            "val_kl": val_metrics["kl"],
        }
        history.append(metrics)
        print(f"gaze epoch={epoch} train_kl={metrics['train_kl']:.6f} val_kl={metrics['val_kl']:.6f}")

    checkpoint = output_dir / "cnn_gaze.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": history}, checkpoint)
    return {"checkpoint": str(checkpoint), "history": history}


def evaluate_gaze_model(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate CNN_GAZE KL divergence."""

    model.eval()
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            frames = batch["images"].to(device)
            gaze = batch["gazes"].to(device)
            output = model(frames)
            loss = loss_fn(output, gaze.reshape(gaze.shape[0], -1))
            total_loss += float(loss.item()) * frames.shape[0]
            total_count += frames.shape[0]
    return {"kl": total_loss / max(total_count, 1)}


def train_action_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    val_loader: DataLoader[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    output_dir: Path,
    gaze_model: nn.Module | None = None,
) -> dict[str, Any]:
    """Train ACTION_SL, GAZED_ACTION_SL, or SGAZED_ACTION_SL with cross entropy."""

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            frames = batch["images"].to(args.device)
            actions = batch["actions"].to(args.device)
            optimizer.zero_grad()
            logits = forward_action_model(name, model, frames, batch, args.gaze_source, gaze_model)
            loss = loss_fn(logits, actions)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * frames.shape[0]
            correct += int((logits.argmax(dim=1) == actions).sum().item())
            total += frames.shape[0]

        val_metrics = evaluate_action_model(name, model, val_loader, args.device, args.gaze_source, gaze_model)
        metrics = {
            "epoch": epoch,
            "train_ce": train_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            "val_ce": val_metrics["ce"],
            "val_acc": val_metrics["acc"],
        }
        history.append(metrics)
        print(
            f"{name} epoch={epoch} train_ce={metrics['train_ce']:.6f} "
            f"train_acc={metrics['train_acc']:.4f} val_ce={metrics['val_ce']:.6f} "
            f"val_acc={metrics['val_acc']:.4f}"
        )

    checkpoint = output_dir / f"{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": history}, checkpoint)
    return {"checkpoint": str(checkpoint), "history": history}


def forward_action_model(
    name: str,
    model: nn.Module,
    frames: torch.Tensor,
    batch: dict[str, torch.Tensor],
    gaze_source: str,
    gaze_model: nn.Module | None,
) -> torch.Tensor:
    """Dispatch a batch through the requested amsterg action architecture."""

    if name == "bc":
        return model(frames)
    if name == "agil":
        return model(frames, agil_overlay(frames, batch, gaze_source, gaze_model))
    if name == "sea":
        frame, overlay = sea_inputs(frames, batch, gaze_source, gaze_model)
        return model(frame, overlay)
    raise ValueError(f"unsupported action model: {name}")


def evaluate_action_model(
    name: str,
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    gaze_source: str,
    gaze_model: nn.Module | None,
) -> dict[str, float]:
    """Evaluate cross entropy and accuracy for an action model."""

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            frames = batch["images"].to(device)
            actions = batch["actions"].to(device)
            logits = forward_action_model(name, model, frames, batch, gaze_source, gaze_model)
            loss = loss_fn(logits, actions)
            total_loss += float(loss.item()) * frames.shape[0]
            correct += int((logits.argmax(dim=1) == actions).sum().item())
            total += frames.shape[0]
    return {"ce": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def load_gaze_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    """Load a gaze checkpoint saved by this script."""

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--ahead-root", type=Path, default=DEFAULT_AHEAD_ROOT)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--models", type=parse_model_list, default=parse_model_list("gaze,bc,agil,sea"))
    parser.add_argument("--gaze-source", choices=["true", "predicted"], default="true")
    parser.add_argument("--gaze-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device(args.device)

    hdf5_path = (args.hdf5 or (args.ahead_root / "data" / "processed" / f"{args.game}.hdf5")).resolve()
    gaze_checkpoint = args.gaze_checkpoint.resolve() if args.gaze_checkpoint is not None else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = select_groups(hdf5_path, args.groups)
    dataset = AtariHeadHDF5Dataset(hdf5_path, groups=groups, max_samples=args.max_samples)
    train_indices, val_indices = split_indices(len(dataset), args.train_fraction, args.seed)
    train_loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    configure_ahead(args.ahead_root.resolve())

    from src.models.action_sl import ACTION_SL
    from src.models.cnn_gaze import CNN_GAZE
    from src.models.gazed_action_sl import GAZED_ACTION_SL
    from src.models.selective_gaze_only import SGAZED_ACTION_SL

    print(f"hdf5={hdf5_path}")
    print(f"groups={groups}")
    print(f"samples train={len(train_indices)} val={len(val_indices)}")
    print(f"train action counts={summarize_class_distribution(dataset, train_indices)}")
    print(f"val action counts={summarize_class_distribution(dataset, val_indices)}")

    results: dict[str, Any] = {
        "game": args.game,
        "hdf5": str(hdf5_path),
        "groups": groups,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "models": args.models,
        "gaze_source": args.gaze_source,
    }

    gaze_model: nn.Module | None = None
    if "gaze" in args.models or args.gaze_source == "predicted" or gaze_checkpoint is not None:
        gaze_model = CNN_GAZE(game=args.game, mode="eval", device=args.device).to(args.device)
        if gaze_checkpoint is not None:
            load_gaze_checkpoint(gaze_model, gaze_checkpoint)
        if "gaze" in args.models:
            results["gaze"] = train_gaze_model(gaze_model, train_loader, val_loader, args, output_dir)
        gaze_model.eval()

    if args.gaze_source == "predicted" and gaze_model is None:
        raise SystemExit("--gaze-source predicted requires --models gaze or --gaze-checkpoint")

    if "bc" in args.models:
        model = ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
        results["bc"] = train_action_model("bc", model, train_loader, val_loader, args, output_dir)
    if "agil" in args.models:
        model = GAZED_ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
        results["agil"] = train_action_model("agil", model, train_loader, val_loader, args, output_dir, gaze_model)
    if "sea" in args.models:
        model = SGAZED_ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
        results["sea"] = train_action_model("sea", model, train_loader, val_loader, args, output_dir, gaze_model)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics_path)


if __name__ == "__main__":
    main()
