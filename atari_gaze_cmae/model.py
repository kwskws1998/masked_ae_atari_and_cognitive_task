"""Frame-conditioned masked gaze modeling for Atari-HEAD behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


ReconstructionLoss = Literal["mse", "smooth_l1"]


@dataclass(frozen=True)
class AtariHeadGazeMAEConfig:
    frame_channels: int = 4
    action_dim: int = 18
    image_height: int = 84
    image_width: int = 84
    patch_size: int = 7
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    mask_ratio: float = 0.3
    reconstruction_loss: ReconstructionLoss = "smooth_l1"
    reconstruction_loss_weight: float = 0.1
    use_gaze_tokens_for_action: bool = True


class AtariHeadGazeMAEOutput(NamedTuple):
    loss: torch.Tensor | None
    action_loss: torch.Tensor | None
    reconstruction_loss: torch.Tensor | None
    action_logits: torch.Tensor
    reconstructed_gaze: torch.Tensor | None
    masked_positions: torch.Tensor | None
    pooled_representation: torch.Tensor


class AtariHeadGazeMAE(nn.Module):
    """Behavior-cloning policy with frame-conditioned masked gaze reconstruction."""

    def __init__(self, cfg: AtariHeadGazeMAEConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if cfg.image_height % cfg.patch_size != 0 or cfg.image_width % cfg.patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")
        if not 0.0 <= cfg.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")

        self.cfg = cfg
        self.grid_height = cfg.image_height // cfg.patch_size
        self.grid_width = cfg.image_width // cfg.patch_size
        self.num_patches = self.grid_height * self.grid_width
        self.patch_area = cfg.patch_size * cfg.patch_size

        self.frame_patch_embed = nn.Conv2d(
            cfg.frame_channels,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.gaze_patch_embed = nn.Conv2d(
            1,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, cfg.hidden_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.gaze_decoder = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, self.patch_area),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def _patch_tokens(self, x: torch.Tensor, embed: nn.Conv2d) -> torch.Tensor:
        tokens = embed(x)
        return tokens.flatten(2).transpose(1, 2)

    def _sample_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float,
    ) -> torch.Tensor:
        if mask_ratio <= 0.0:
            return torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        masked = torch.rand(batch_size, self.num_patches, device=device) < mask_ratio
        needs_mask = ~masked.any(dim=1)
        if needs_mask.any():
            batch_indices = torch.arange(batch_size, device=device)
            masked[batch_indices[needs_mask], 0] = True
        return masked

    def _patchify_gaze(self, gaze_heatmaps: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            gaze_heatmaps,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )
        return patches.transpose(1, 2)

    def _unpatchify_gaze(self, patches: torch.Tensor) -> torch.Tensor:
        patches = patches.transpose(1, 2)
        return F.fold(
            patches,
            output_size=(self.cfg.image_height, self.cfg.image_width),
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )

    def _reconstruction_loss(
        self,
        reconstructed_patches: torch.Tensor,
        target_patches: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        if not masked_positions.any():
            return reconstructed_patches.new_tensor(0.0)
        pred = reconstructed_patches[masked_positions]
        target = target_patches[masked_positions]
        if self.cfg.reconstruction_loss == "mse":
            return F.mse_loss(pred, target)
        if self.cfg.reconstruction_loss == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported reconstruction_loss={self.cfg.reconstruction_loss}")

    def forward(
        self,
        frames: torch.Tensor,
        gaze_heatmaps: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        mask_ratio: float | None = None,
    ) -> AtariHeadGazeMAEOutput:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [batch, channels, height, width]")
        if frames.shape[1:] != (
            self.cfg.frame_channels,
            self.cfg.image_height,
            self.cfg.image_width,
        ):
            raise ValueError(
                "frames shape must match [batch, frame_channels, image_height, image_width]"
            )
        if gaze_heatmaps is not None and gaze_heatmaps.shape != (
            frames.shape[0],
            1,
            self.cfg.image_height,
            self.cfg.image_width,
        ):
            raise ValueError("gaze_heatmaps must have shape [batch, 1, image_height, image_width]")

        batch_size = frames.shape[0]
        frame_tokens = self._patch_tokens(frames, self.frame_patch_embed)
        gaze_tokens = torch.zeros_like(frame_tokens)
        reconstructed_gaze = None
        reconstruction_loss = None
        masked_positions = None

        if gaze_heatmaps is not None:
            gaze_tokens = self._patch_tokens(gaze_heatmaps, self.gaze_patch_embed)
            effective_mask_ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
            masked_positions = self._sample_mask(batch_size, frames.device, effective_mask_ratio)
            mask_token = self.mask_token.view(1, 1, -1).to(dtype=gaze_tokens.dtype)
            gaze_tokens = torch.where(masked_positions.unsqueeze(-1), mask_token, gaze_tokens)
            if not self.cfg.use_gaze_tokens_for_action:
                gaze_tokens = torch.where(masked_positions.unsqueeze(-1), mask_token, torch.zeros_like(gaze_tokens))

        encoded = self.encoder(
            self.input_norm(frame_tokens + gaze_tokens + self.position_embedding)
        )
        pooled = encoded.mean(dim=1)
        action_logits = self.action_head(pooled)

        if gaze_heatmaps is not None:
            reconstructed_patches = self.gaze_decoder(encoded)
            reconstructed_gaze = self._unpatchify_gaze(reconstructed_patches)
            target_patches = self._patchify_gaze(gaze_heatmaps)
            assert masked_positions is not None
            reconstruction_loss = self._reconstruction_loss(
                reconstructed_patches,
                target_patches,
                masked_positions,
            )

        action_loss = F.cross_entropy(action_logits, actions) if actions is not None else None
        loss = None
        if action_loss is not None and reconstruction_loss is not None:
            loss = action_loss + self.cfg.reconstruction_loss_weight * reconstruction_loss
        elif action_loss is not None:
            loss = action_loss
        elif reconstruction_loss is not None:
            loss = self.cfg.reconstruction_loss_weight * reconstruction_loss

        return AtariHeadGazeMAEOutput(
            loss=loss,
            action_loss=action_loss,
            reconstruction_loss=reconstruction_loss,
            action_logits=action_logits,
            reconstructed_gaze=reconstructed_gaze,
            masked_positions=masked_positions,
            pooled_representation=pooled,
        )
