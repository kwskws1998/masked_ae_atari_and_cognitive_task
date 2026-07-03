"""Synthetic tests for active-gaze MAE and Decision Transformer models."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atari_gaze_cmae import (  # noqa: E402
    ActiveGazeDecisionTransformer,
    ActiveGazeDecisionTransformerConfig,
    ActiveGazeMAEVisualEncoder,
    AtariHeadHDF5TrajectoryDataset,
)
from scripts.train_active_gaze_dt import split_dataset_indices  # noqa: E402


def _small_config(context_length: int = 4) -> ActiveGazeDecisionTransformerConfig:
    return ActiveGazeDecisionTransformerConfig(
        embed_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        encoder_ff_dim=64,
        decoder_dim=32,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ff_dim=64,
        dt_layers=1,
        dt_heads=4,
        context_length=context_length,
        dropout=0.0,
    )


def _normalized_gaze(shape: tuple[int, ...]) -> np.ndarray:
    gaze = np.random.rand(*shape).astype(np.float32)
    flat = gaze.reshape(shape[0], -1)
    flat /= flat.sum(axis=1, keepdims=True)
    return flat.reshape(shape)


def _write_synthetic_hdf5(hdf5_path: Path, group_count: int = 5, length: int = 16) -> None:
    with h5py.File(hdf5_path, "w") as handle:
        for group_index in range(group_count):
            group = handle.create_group(f"trial_{group_index}")
            group.create_dataset("images", data=np.random.rand(length, 4, 84, 84).astype(np.float32))
            actions = np.zeros((length, 4), dtype=np.int64)
            actions[:, -1] = np.arange(length) % 18
            group.create_dataset("actions", data=actions)
            group.create_dataset("gazes", data=_normalized_gaze((length, 84, 84)))
            rewards = np.zeros((length, 4), dtype=np.float32)
            rewards[:, -1] = np.arange(length, dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            group.create_dataset("episode_ids", data=np.zeros((length, 4), dtype=np.int64))


def test_active_visual_encoder_masks_to_visible_quarter() -> None:
    cfg = _small_config()
    encoder = ActiveGazeMAEVisualEncoder(cfg)
    frames = torch.rand(2, 4, 84, 84)
    gaze = torch.rand(2, 1, 84, 84)
    gaze = gaze / gaze.flatten(1).sum(dim=1).view(2, 1, 1, 1)
    output = encoder(frames, gaze)

    assert output.encoded_visible_tokens.shape == (2, 36, cfg.embed_dim)
    assert output.visible_indices.shape == (2, 36)
    assert output.mask_probs.shape == (2, 144)
    assert torch.allclose(output.mask_probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert output.gaze_patch_target is not None
    assert torch.allclose(output.gaze_patch_target.sum(dim=1), torch.ones(2), atol=1e-5)
    restored = encoder.unpatchify(encoder.patchify(frames))
    assert torch.allclose(restored, frames, atol=1e-6)


def test_random_visual_encoder_masks_to_visible_quarter_without_gaze_loss() -> None:
    cfg = ActiveGazeDecisionTransformerConfig(
        embed_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        encoder_ff_dim=64,
        decoder_dim=32,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ff_dim=64,
        dt_layers=1,
        dt_heads=4,
        context_length=4,
        dropout=0.0,
        mask_strategy="random",
    )
    encoder = ActiveGazeMAEVisualEncoder(cfg)
    frames = torch.rand(2, 4, 84, 84)
    gaze = torch.rand(2, 1, 84, 84)
    output = encoder(frames, gaze, reconstruct=False)

    assert output.encoded_visible_tokens.shape == (2, 36, cfg.embed_dim)
    assert output.visible_indices.shape == (2, 36)
    assert output.mask_probs.shape == (2, 144)
    assert output.reconstruction_loss is None
    assert output.gaze_patch_target is not None
    assert output.gaze_loss is None


def test_active_gaze_decision_transformer_forward() -> None:
    cfg = _small_config(context_length=4)
    model = ActiveGazeDecisionTransformer(cfg)
    frames = torch.rand(2, 4, 4, 84, 84)
    gaze = torch.rand(2, 4, 1, 84, 84)
    gaze = gaze / gaze.flatten(2).sum(dim=2).view(2, 4, 1, 1, 1)
    actions = torch.randint(0, cfg.action_dim, (2, 4))
    rtg = torch.rand(2, 4)
    timesteps = torch.arange(4).view(1, 4).repeat(2, 1)
    output = model(
        frames=frames,
        actions=actions,
        returns_to_go=rtg,
        timesteps=timesteps,
        gaze_heatmaps=gaze,
    )

    assert output.loss is not None
    assert output.action_logits.shape == (2, 4, cfg.action_dim)
    assert output.visible_indices.shape == (2, 4, 36)
    assert output.mask_probs.shape == (2, 4, 144)


def test_active_gaze_decision_transformer_loss_composition() -> None:
    cfg = ActiveGazeDecisionTransformerConfig(
        embed_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        encoder_ff_dim=64,
        decoder_dim=32,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ff_dim=64,
        dt_layers=1,
        dt_heads=4,
        context_length=3,
        dropout=0.0,
        reconstruction_loss_weight=0.7,
        gaze_loss_weight=0.3,
    )
    model = ActiveGazeDecisionTransformer(cfg)
    frames = torch.rand(2, 3, 4, 84, 84)
    gaze = torch.rand(2, 3, 1, 84, 84)
    gaze = gaze / gaze.flatten(2).sum(dim=2).view(2, 3, 1, 1, 1)
    actions = torch.randint(0, cfg.action_dim, (2, 3))
    rtg = torch.rand(2, 3)
    timesteps = torch.arange(3).view(1, 3).repeat(2, 1)
    output = model(
        frames=frames,
        actions=actions,
        returns_to_go=rtg,
        timesteps=timesteps,
        gaze_heatmaps=gaze,
    )

    assert output.loss is not None
    assert output.action_loss is not None
    assert output.reconstruction_loss is not None
    assert output.gaze_loss is not None
    expected = (
        output.action_loss
        + cfg.reconstruction_loss_weight * output.reconstruction_loss
        + cfg.gaze_loss_weight * output.gaze_loss
    )
    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_active_visual_encoder_autocast_reconstruction_dtype() -> None:
    cfg = _small_config()
    encoder = ActiveGazeMAEVisualEncoder(cfg)
    frames = torch.rand(2, 4, 84, 84)
    gaze = torch.rand(2, 1, 84, 84)
    gaze = gaze / gaze.flatten(1).sum(dim=1).view(2, 1, 1, 1)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(frames, gaze)

    assert output.reconstruction_loss is not None
    assert output.reconstructed_patches is not None
    assert output.reconstructed_patches.dtype == torch.bfloat16


def test_hdf5_trajectory_dataset_shapes_and_rtg() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        hdf5_path = Path(tmp) / "synthetic.hdf5"
        length = 12
        with h5py.File(hdf5_path, "w") as handle:
            group = handle.create_group("trial")
            group.create_dataset("images", data=np.random.rand(length, 4, 84, 84).astype(np.float32))
            actions = np.zeros((length, 4), dtype=np.int64)
            actions[:, -1] = np.arange(length) % 18
            group.create_dataset("actions", data=actions)
            group.create_dataset("gazes", data=_normalized_gaze((length, 84, 84)))
            rewards = np.zeros((length, 4), dtype=np.float32)
            rewards[:, -1] = np.arange(length, dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            group.create_dataset("episode_ids", data=np.zeros((length, 4), dtype=np.int64))

        dataset = AtariHeadHDF5TrajectoryDataset(hdf5_path, context_length=4, max_samples=2)
        sample = dataset[0]
        assert sample["frames"].shape == (4, 4, 84, 84)
        assert sample["actions"].shape == (4,)
        assert sample["gazes"].shape == (4, 1, 84, 84)
        assert sample["rewards"].shape == (4,)
        assert sample["rtg"].shape == (4,)
        expected = torch.tensor([sum(range(i, length)) for i in range(4)], dtype=torch.float32)
        assert torch.allclose(sample["rtg"], expected)
        dataset.close()


def test_trial_split_has_disjoint_trial_groups() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        hdf5_path = Path(tmp) / "synthetic.hdf5"
        _write_synthetic_hdf5(hdf5_path, group_count=5, length=12)
        dataset = AtariHeadHDF5TrajectoryDataset(hdf5_path, context_length=4)
        split = split_dataset_indices(dataset, "trial", train_fraction=0.6, val_fraction=0.2, seed=7)

        train_groups = {dataset.samples[index][0] for index in split.train}
        val_groups = {dataset.samples[index][0] for index in split.val}
        test_groups = {dataset.samples[index][0] for index in split.test}
        assert train_groups
        assert val_groups
        assert test_groups
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)
        dataset.close()


def test_block_split_purges_overlapping_windows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        hdf5_path = Path(tmp) / "synthetic.hdf5"
        _write_synthetic_hdf5(hdf5_path, group_count=1, length=24)
        context_length = 4
        dataset = AtariHeadHDF5TrajectoryDataset(hdf5_path, context_length=context_length)
        split = split_dataset_indices(dataset, "block", train_fraction=0.6, val_fraction=0.2, seed=7)

        split_indices = {"train": split.train, "val": split.val, "test": split.test}
        for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
            for left_index in split_indices[left_name]:
                left_group, left_start = dataset.samples[left_index]
                for right_index in split_indices[right_name]:
                    right_group, right_start = dataset.samples[right_index]
                    if left_group == right_group:
                        assert abs(left_start - right_start) >= context_length
        dataset.close()


if __name__ == "__main__":
    test_active_visual_encoder_masks_to_visible_quarter()
    test_random_visual_encoder_masks_to_visible_quarter_without_gaze_loss()
    test_active_gaze_decision_transformer_forward()
    test_active_gaze_decision_transformer_loss_composition()
    test_active_visual_encoder_autocast_reconstruction_dtype()
    test_hdf5_trajectory_dataset_shapes_and_rtg()
    test_trial_split_has_disjoint_trial_groups()
    test_block_split_purges_overlapping_windows()
