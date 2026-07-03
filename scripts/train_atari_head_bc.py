"""Minimal behavior-cloning training loop for one Atari-HEAD trial."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atari_gaze_cmae import (
    AtariHeadGazeMAE,
    AtariHeadGazeMAEConfig,
    AtariHeadTrialDataset,
    collate_atari_head_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-archive", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-no-gaze", action="store_true")
    args = parser.parse_args()

    dataset = AtariHeadTrialDataset(
        args.frame_archive,
        args.label_file,
        max_rows=args.max_samples,
        skip_no_gaze=args.skip_no_gaze,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_atari_head_samples,
    )
    model = AtariHeadGazeMAE(AtariHeadGazeMAEConfig()).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_seen = 0
        for batch in loader:
            frames = batch["frames"].to(args.device)
            gaze_heatmaps = batch["gaze_heatmaps"].to(args.device)
            actions = batch["actions"].to(args.device)
            output = model(frames=frames, gaze_heatmaps=gaze_heatmaps, actions=actions)
            assert output.loss is not None
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach().cpu()) * frames.size(0)
            total_seen += frames.size(0)
        avg_loss = total_loss / max(total_seen, 1)
        print(f"epoch={epoch + 1} samples={total_seen} loss={avg_loss:.6f}")
    dataset.close()


if __name__ == "__main__":
    main()
