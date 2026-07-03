"""Smoke-run amsterg/ahead on one Atari-HEAD Breakout trial."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
AHEAD_ROOT = ROOT / "external" / "amsterg_ahead"
TRIAL = "198_RZ_3877709_Dec-03-16-56-11"
os.environ.setdefault("MPLCONFIGDIR", str(AHEAD_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(AHEAD_ROOT / ".cache"))


def prepare_trial() -> None:
    raw_dir = AHEAD_ROOT / "data" / "raw" / "breakout"
    interim_dir = AHEAD_ROOT / "data" / "interim" / "breakout"
    trial_dir = interim_dir / TRIAL
    interim_dir.mkdir(parents=True, exist_ok=True)
    if not trial_dir.exists():
        subprocess.run(
            ["tar", "-xjf", str(raw_dir / f"{TRIAL}.tar.bz2"), "-C", str(interim_dir)],
            check=True,
        )

    sys.path.insert(0, str(AHEAD_ROOT))
    os.chdir(AHEAD_ROOT)
    from src.data.data_utils import process_gaze_data

    gaze_out = trial_dir / f"{TRIAL}_gaze_data.csv"
    if not gaze_out.exists():
        process_gaze_data(str(raw_dir / f"{TRIAL}.txt"), str(gaze_out), [1, 3, 4])


def run_smoke() -> None:
    sys.path.insert(0, str(AHEAD_ROOT))
    os.chdir(AHEAD_ROOT)

    import torch
    from src.data.data_loaders import load_action_data, load_gaze_data
    from src.features.feat_utils import fuse_gazes, reduce_gaze_stack, transform_images
    from src.models.action_sl import ACTION_SL
    from src.models.cnn_gaze import CNN_GAZE
    from src.models.gazed_action_sl import GAZED_ACTION_SL
    from src.models.selective_gaze_only import SGAZED_ACTION_SL

    device = torch.device("cpu")
    images, actions = load_action_data(stack=4, from_ix=0, till_ix=64, game="breakout", game_run=TRIAL)
    _, gazes = load_gaze_data(stack=4, from_ix=0, till_ix=64, game="breakout", game_run=TRIAL, skip_images=True)
    x = transform_images(images, type="torch").to(device)
    y = torch.LongTensor(actions)[:, -1].to(device)
    gaze_pdf = torch.stack([reduce_gaze_stack(gaze_stack) for gaze_stack in gazes]).to(device)
    gaze_stack = fuse_gazes(x, gazes, gaze_count=1).to(device)

    batch = slice(0, 8)
    x = x[batch]
    y = y[batch]
    gaze_pdf = gaze_pdf[batch]
    gaze_stack = gaze_stack[batch]

    gaze_model = CNN_GAZE(game="breakout", mode="eval", device=device)
    gaze_out = gaze_model(x)
    gaze_loss = torch.nn.KLDivLoss(reduction="batchmean")(gaze_out, gaze_pdf.reshape(gaze_pdf.shape[0], -1))

    action_model = ACTION_SL(game="breakout", mode="eval", device=device)
    action_opt = torch.optim.Adam(action_model.parameters(), lr=1e-4)
    action_logits = action_model(x)
    action_loss = torch.nn.CrossEntropyLoss()(action_logits, y)
    action_loss.backward()
    action_opt.step()

    agil_model = GAZED_ACTION_SL(game="breakout", mode="eval", device=device)
    agil_logits = agil_model(x, gaze_stack)

    sea_model = SGAZED_ACTION_SL(game="breakout", mode="eval", device=device)
    sea_logits = sea_model(x[:, -1:].contiguous(), (x[:, -1:] * gaze_stack[:, -1:]).contiguous())

    print(f"x={tuple(x.shape)} y={tuple(y.shape)} gaze_pdf={tuple(gaze_pdf.shape)} gaze_stack={tuple(gaze_stack.shape)}")
    print(f"cnn_gaze={tuple(gaze_out.shape)} kl={gaze_loss.item():.6f}")
    print(f"action_sl={tuple(action_logits.shape)} ce={action_loss.item():.6f}")
    print(f"gazed_action_sl={tuple(agil_logits.shape)}")
    print(f"selective_gaze_only={tuple(sea_logits.shape)} gate_mean={sea_model.gate_output:.6f}")


def main() -> None:
    prepare_trial()
    run_smoke()


if __name__ == "__main__":
    main()
