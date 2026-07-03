"""Synthetic smoke tests for Atari-HEAD parsing, loading, and model forward."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tarfile
import tempfile

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atari_gaze_cmae import (  # noqa: E402
    AtariHeadGazeMAE,
    AtariHeadGazeMAEConfig,
    AtariHeadTrialDataset,
    collate_atari_head_samples,
    files_for_trial,
    read_atari_head_labels,
)
from atari_gaze_cmae.zenodo import ZenodoFile  # noqa: E402


def _write_png_to_tar(tar: tarfile.TarFile, name: str, value: int) -> None:
    image = Image.new("RGB", (160, 210), color=(value, value, value))
    data = BytesIO()
    image.save(data, format="PNG")
    payload = data.getvalue()
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, BytesIO(payload))


def _make_trial(root: Path) -> tuple[Path, Path]:
    archive = root / "100_fake.tar.bz2"
    labels = root / "100_fake.txt"
    with tarfile.open(archive, "w:bz2") as tar:
        for idx in range(6):
            _write_png_to_tar(tar, f"{idx:06d}.png", 20 + idx)
    with labels.open("w", encoding="utf-8") as f:
        f.write("frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n")
        for idx in range(6):
            f.write(f"{idx:06d},0,0,33,0,{idx % 3},80,100\n")
    return archive, labels


def test_label_parser_and_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive, labels = _make_trial(Path(tmp))
        parsed = read_atari_head_labels(labels)
        assert len(parsed) == 6
        assert parsed[0].frame_id == "000000"
        assert parsed[0].gaze_points == ((80.0, 100.0),)

        dataset = AtariHeadTrialDataset(archive, labels, history=4)
        sample = dataset[0]
        assert sample["frames"].shape == (4, 84, 84)
        assert sample["gaze_heatmap"].shape == (1, 84, 84)
        assert sample["gaze_heatmap"].sum() > 0
        batch = collate_atari_head_samples([sample, dataset[1]])
        assert batch["frames"].shape == (2, 4, 84, 84)
        dataset.close()


def test_model_forward_on_synthetic_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive, labels = _make_trial(Path(tmp))
        dataset = AtariHeadTrialDataset(archive, labels, history=4)
        batch = collate_atari_head_samples([dataset[0], dataset[1]])
        cfg = AtariHeadGazeMAEConfig(hidden_dim=64, num_heads=4, ff_dim=128)
        model = AtariHeadGazeMAE(cfg)
        output = model(
            frames=batch["frames"],
            gaze_heatmaps=batch["gaze_heatmaps"],
            actions=batch["actions"],
        )
        assert output.loss is not None
        assert output.action_logits.shape == (2, cfg.action_dim)
        assert output.reconstructed_gaze is not None
        assert output.reconstructed_gaze.shape == (2, 1, 84, 84)
        dataset.close()


def test_trial_file_selection() -> None:
    files = [
        ZenodoFile("100_a.tar.bz2", 1, None, "https://example.com/a"),
        ZenodoFile("100_a.txt", 1, None, "https://example.com/b"),
        ZenodoFile("101_b.txt", 1, None, "https://example.com/c"),
    ]
    selected = files_for_trial(files, "100")
    assert [item.name for item in selected] == ["100_a.tar.bz2", "100_a.txt"]


if __name__ == "__main__":
    test_label_parser_and_dataset()
    test_model_forward_on_synthetic_batch()
    test_trial_file_selection()
