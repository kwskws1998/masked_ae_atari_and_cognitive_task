"""Train active-gaze MAE behavior cloning or Decision Transformer models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atari_gaze_cmae import (  # noqa: E402
    ActiveGazeBehaviorCloner,
    ActiveGazeDecisionTransformer,
    ActiveGazeDecisionTransformerConfig,
    AtariHeadHDF5TrajectoryDataset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = ROOT / "external" / "amsterg_ahead" / "data" / "processed" / "breakout.hdf5"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "active_gaze_dt" / "smoke"


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    val: list[int]
    test: list[int]
    metadata: dict[str, Any]


def _split_counts(length: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if length < 2:
        raise ValueError("dataset needs at least two split units")
    if train_fraction <= 0.0 or val_fraction < 0.0:
        raise ValueError("train_fraction must be positive and val_fraction must be non-negative")
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction < -1e-8:
        raise ValueError("train_fraction + val_fraction must be <= 1.0")

    fractions = [train_fraction, val_fraction, max(test_fraction, 0.0)]
    raw = [fraction * length for fraction in fractions]
    counts = [int(np.floor(value)) for value in raw]
    remainder = length - sum(counts)
    order = sorted(range(3), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    if length >= 3:
        for target in (1, 2):
            if fractions[target] > 0.0 and counts[target] == 0:
                donor = max((idx for idx in range(3) if idx != target), key=lambda idx: counts[idx])
                if counts[donor] <= 1:
                    continue
                counts[donor] -= 1
                counts[target] += 1
    if counts[0] == 0:
        donor = max(range(1, 3), key=lambda idx: counts[idx])
        if counts[donor] <= 1:
            raise ValueError("could not allocate a non-empty train split")
        counts[donor] -= 1
        counts[0] = 1
    return counts[0], counts[1], counts[2]


def split_sample_indices(
    length: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Create deterministic sample-level train/validation/test indices."""

    if length < 2:
        raise ValueError("dataset needs at least two samples for train/validation/test split")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    train_count, val_count, test_count = _split_counts(length, train_fraction, val_fraction)
    train_end = train_count
    val_end = train_end + val_count
    return DatasetSplit(
        train=indices[:train_end],
        val=indices[train_end:val_end],
        test=indices[val_end : val_end + test_count],
        metadata={
            "split_strategy": "sample",
            "leakage_warning": "sample-level random split can overlap adjacent trajectory windows",
        },
    )


def _indices_for_groups(dataset: AtariHeadHDF5TrajectoryDataset, groups: set[str]) -> list[int]:
    return [index for index, (group_name, _) in enumerate(dataset.samples) if group_name in groups]


def split_trial_indices(
    dataset: AtariHeadHDF5TrajectoryDataset,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Split whole Atari-HEAD trials before shuffling windows."""

    groups = list(dataset.groups)
    if len(groups) < 3:
        raise ValueError("trial split needs at least three trial groups; use --split-strategy block for small smoke data")
    random.Random(seed).shuffle(groups)
    train_count, val_count, test_count = _split_counts(len(groups), train_fraction, val_fraction)
    train_groups = set(groups[:train_count])
    val_groups = set(groups[train_count : train_count + val_count])
    test_groups = set(groups[train_count + val_count : train_count + val_count + test_count])
    return DatasetSplit(
        train=_indices_for_groups(dataset, train_groups),
        val=_indices_for_groups(dataset, val_groups),
        test=_indices_for_groups(dataset, test_groups),
        metadata={
            "split_strategy": "trial",
            "train_groups": sorted(train_groups),
            "val_groups": sorted(val_groups),
            "test_groups": sorted(test_groups),
        },
    )


def split_block_indices(
    dataset: AtariHeadHDF5TrajectoryDataset,
    train_fraction: float,
    val_fraction: float,
) -> DatasetSplit:
    """Split contiguous windows inside each trial with a purge gap at boundaries."""

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    purge_gap = max(dataset.context_length - 1, 0)
    by_group: dict[str, list[tuple[int, int]]] = {}
    for index, (group_name, start) in enumerate(dataset.samples):
        by_group.setdefault(group_name, []).append((start, index))

    block_metadata: dict[str, Any] = {}
    for group_name, starts_and_indices in by_group.items():
        starts_and_indices.sort()
        length = len(starts_and_indices)
        available_length = length - (2 * purge_gap)
        if available_length < 3:
            raise ValueError(
                f"group {group_name!r} has only {length} windows, not enough for block split "
                f"with context_length={dataset.context_length}"
            )
        train_count, val_count, test_count = _split_counts(available_length, train_fraction, val_fraction)
        train_end = train_count
        val_start = min(length, train_end + purge_gap)
        val_end = min(length, val_start + val_count)
        test_start = min(length, val_end + purge_gap)
        test_end = min(length, test_start + test_count)

        train.extend(index for _, index in starts_and_indices[:train_end])
        val.extend(index for _, index in starts_and_indices[val_start:val_end])
        test.extend(index for _, index in starts_and_indices[test_start:test_end])
        block_metadata[group_name] = {
            "windows": length,
            "train_windows": train_end,
            "val_windows": max(val_end - val_start, 0),
            "test_windows": max(test_end - test_start, 0),
            "purge_gap_windows": purge_gap,
        }

    return DatasetSplit(
        train=train,
        val=val,
        test=test,
        metadata={
            "split_strategy": "block",
            "purge_gap_windows": purge_gap,
            "groups": block_metadata,
        },
    )


def split_dataset_indices(
    dataset: AtariHeadHDF5TrajectoryDataset,
    split_strategy: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Create train/validation/test indices using the requested split strategy."""

    if split_strategy == "sample":
        return split_sample_indices(len(dataset), train_fraction, val_fraction, seed)
    if split_strategy == "trial":
        return split_trial_indices(dataset, train_fraction, val_fraction, seed)
    if split_strategy == "block":
        return split_block_indices(dataset, train_fraction, val_fraction)
    raise ValueError(f"unsupported split_strategy: {split_strategy}")


def make_loader(
    dataset: AtariHeadHDF5TrajectoryDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a single-process loader over a stable subset."""

    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def make_config(args: argparse.Namespace) -> ActiveGazeDecisionTransformerConfig:
    """Create the model config from CLI arguments."""

    return ActiveGazeDecisionTransformerConfig(
        embed_dim=args.embed_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_ff_dim=args.encoder_ff_dim,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff_dim=args.decoder_ff_dim,
        dt_layers=args.dt_layers,
        dt_heads=args.dt_heads,
        dt_ff_mult=args.dt_ff_mult,
        context_length=args.context_length,
        max_timestep=args.max_timestep,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        reconstruction_loss_weight=args.lambda_rec,
        gaze_loss_weight=args.lambda_gaze,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensor batch values to the requested device."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_mode(
    mode: str,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    compute_auxiliary: bool,
) -> Any:
    """Dispatch active_bc and active_dt training forwards."""

    if mode == "active_dt":
        return model(
            frames=batch["frames"],
            actions=batch["actions"],
            returns_to_go=batch["rtg"],
            timesteps=batch["timesteps"],
            gaze_heatmaps=batch["gazes"],
            compute_auxiliary=compute_auxiliary,
        )
    if mode == "active_bc":
        frames = batch["frames"]
        gazes = batch["gazes"]
        actions = batch["actions"]
        batch_size, context_length = frames.shape[:2]
        return model(
            frames=frames.reshape(batch_size * context_length, *frames.shape[2:]),
            actions=actions.reshape(batch_size * context_length),
            gaze_heatmaps=gazes.reshape(batch_size * context_length, *gazes.shape[2:]),
            compute_auxiliary=compute_auxiliary,
        )
    raise ValueError(f"unsupported mode: {mode}")


def batch_accuracy(mode: str, output: Any, batch: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Count action prediction accuracy for active_bc or active_dt outputs."""

    if mode == "active_dt":
        logits = output.action_logits.reshape(-1, output.action_logits.shape[-1])
        targets = batch["actions"].reshape(-1)
    else:
        logits = output.action_logits
        targets = batch["actions"].reshape(-1)
    correct = int((logits.argmax(dim=-1) == targets).sum().item())
    return correct, int(targets.numel())


def run_epoch(
    phase: str,
    epoch: int,
    mode: str,
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    compute_auxiliary: bool = True,
    log_interval: int = 0,
) -> dict[str, float]:
    """Run one train or validation epoch and return averaged metrics."""

    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "action_loss": 0.0,
        "reconstruction_loss": 0.0,
        "gaze_loss": 0.0,
    }
    correct = 0
    total = 0
    token_count = 0
    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                output = forward_mode(mode, model, batch, compute_auxiliary=compute_auxiliary)
            if output.loss is None:
                raise RuntimeError("model did not return a loss")
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(output.loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            action_tokens = int(batch["actions"].numel())
            totals["loss"] += float(output.loss.detach().cpu()) * action_tokens
            if output.action_loss is not None:
                totals["action_loss"] += float(output.action_loss.detach().cpu()) * action_tokens
            if output.reconstruction_loss is not None:
                totals["reconstruction_loss"] += float(output.reconstruction_loss.detach().cpu()) * action_tokens
            if output.gaze_loss is not None:
                totals["gaze_loss"] += float(output.gaze_loss.detach().cpu()) * action_tokens
            batch_correct, batch_total = batch_accuracy(mode, output, batch)
            correct += batch_correct
            total += batch_total
            token_count += action_tokens
            is_last_batch = batch_idx == len(loader)
            should_log = log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or is_last_batch)
            if should_log:
                running_loss = totals["loss"] / max(token_count, 1)
                running_action_loss = totals["action_loss"] / max(token_count, 1)
                running_reconstruction_loss = totals["reconstruction_loss"] / max(token_count, 1)
                running_gaze_loss = totals["gaze_loss"] / max(token_count, 1)
                running_action_acc = correct / max(total, 1)
                print(
                    f"{phase} epoch={epoch} batch={batch_idx}/{len(loader)} "
                    f"action_tokens={token_count} loss={running_loss:.6f} "
                    f"action_loss={running_action_loss:.6f} "
                    f"rec_loss={running_reconstruction_loss:.6f} "
                    f"gaze_loss={running_gaze_loss:.6f} action_acc={running_action_acc:.4f}",
                    flush=True,
                )

    denom = max(token_count, 1)
    action_acc = correct / max(total, 1)
    return {
        "loss": totals["loss"] / denom,
        "action_loss": totals["action_loss"] / denom,
        "reconstruction_loss": totals["reconstruction_loss"] / denom,
        "gaze_loss": totals["gaze_loss"] / denom,
        "action_acc": action_acc,
        "acc": action_acc,
        "action_tokens": token_count,
    }


def checkpoint_payload(
    args: argparse.Namespace,
    cfg: ActiveGazeDecisionTransformerConfig,
    model: nn.Module,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a portable checkpoint payload."""

    clean_args = {}
    for key, value in vars(args).items():
        clean_args[key] = str(value) if isinstance(value, Path) else value
    return {
        "mode": args.mode,
        "model_config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "history": history,
        "args": clean_args,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["active_bc", "active_dt"], default="active_dt")
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-strategy", choices=["sample", "trial", "block"], default="block")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--require-rewards", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-ff-dim", type=int, default=256)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-ff-dim", type=int, default=256)
    parser.add_argument("--dt-layers", type=int, default=4)
    parser.add_argument("--dt-heads", type=int, default=4)
    parser.add_argument("--dt-ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--mask-strategy", choices=["learned", "random"], default="learned")
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-gaze", type=float, default=0.1)
    parser.add_argument("--disable-reconstruction", action="store_true")
    parser.add_argument("--max-timestep", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=50, help="Print batch progress every N batches.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    args.hdf5 = args.hdf5.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AtariHeadHDF5TrajectoryDataset(
        args.hdf5,
        groups=args.groups,
        context_length=args.context_length,
        max_samples=args.max_samples,
        require_rewards=args.require_rewards,
    )
    split = split_dataset_indices(
        dataset,
        args.split_strategy,
        args.train_fraction,
        args.val_fraction,
        args.seed,
    )
    if not split.train or not split.val or not split.test:
        raise ValueError(
            "train/val/test splits must all be non-empty; adjust fractions, max_samples, or split_strategy"
        )
    train_loader = make_loader(
        dataset,
        split.train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = make_loader(
        dataset,
        split.val,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = make_loader(
        dataset,
        split.test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    cfg = make_config(args)
    if args.mode == "active_dt":
        model: nn.Module = ActiveGazeDecisionTransformer(cfg)
    else:
        model = ActiveGazeBehaviorCloner(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"hdf5={args.hdf5}")
    print(f"groups={dataset.groups}")
    print(
        f"split_strategy={args.split_strategy} train_windows={len(split.train)} "
        f"val_windows={len(split.val)} test_windows={len(split.test)}"
    )
    print(f"split_metadata={json.dumps(split.metadata, sort_keys=True)}")
    print(
        f"mode={args.mode} context_length={args.context_length} "
        f"mask_strategy={args.mask_strategy} reconstruction={not args.disable_reconstruction}"
    )

    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            "train",
            epoch,
            args.mode,
            model,
            train_loader,
            device,
            optimizer,
            scaler=scaler,
            use_amp=args.amp,
            compute_auxiliary=not args.disable_reconstruction,
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            "val",
            epoch,
            args.mode,
            model,
            val_loader,
            device,
            use_amp=args.amp,
            compute_auxiliary=not args.disable_reconstruction,
            log_interval=args.log_interval,
        )
        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(metrics)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"train_action_loss={train_metrics['action_loss']:.6f} "
            f"train_rec_loss={train_metrics['reconstruction_loss']:.6f} "
            f"train_gaze_loss={train_metrics['gaze_loss']:.6f} "
            f"train_action_acc={train_metrics['action_acc']:.4f} val_loss={val_metrics['loss']:.6f} "
            f"val_action_loss={val_metrics['action_loss']:.6f} "
            f"val_rec_loss={val_metrics['reconstruction_loss']:.6f} "
            f"val_gaze_loss={val_metrics['gaze_loss']:.6f} val_action_acc={val_metrics['action_acc']:.4f}"
        )

    test_metrics = run_epoch(
        "test",
        args.epochs,
        args.mode,
        model,
        test_loader,
        device,
        use_amp=args.amp,
        compute_auxiliary=not args.disable_reconstruction,
        log_interval=args.log_interval,
    )
    print(
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_action_loss={test_metrics['action_loss']:.6f} "
        f"test_rec_loss={test_metrics['reconstruction_loss']:.6f} "
        f"test_gaze_loss={test_metrics['gaze_loss']:.6f} "
        f"test_action_acc={test_metrics['action_acc']:.4f}"
    )

    checkpoint = args.output_dir / f"{args.mode}.pt"
    torch.save(checkpoint_payload(args, cfg, model, history), checkpoint)
    metrics_path = args.output_dir / f"{args.mode}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "hdf5": str(args.hdf5),
                "groups": dataset.groups,
                "split_strategy": args.split_strategy,
                "split_metadata": split.metadata,
                "train_size": len(split.train),
                "val_size": len(split.val),
                "test_size": len(split.test),
                "checkpoint": str(checkpoint),
                "history": history,
                "test": test_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dataset.close()
    print(checkpoint)
    print(metrics_path)


if __name__ == "__main__":
    main()
