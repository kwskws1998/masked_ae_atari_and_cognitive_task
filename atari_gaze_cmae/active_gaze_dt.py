"""Active gaze-supervised MAE visual encoder and Decision Transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ActiveGazeDecisionTransformerConfig:
    frame_channels: int = 4
    action_dim: int = 18
    image_height: int = 84
    image_width: int = 84
    patch_size: int = 7
    embed_dim: int = 128
    encoder_layers: int = 2
    encoder_heads: int = 4
    encoder_ff_dim: int = 256
    decoder_dim: int = 128
    decoder_layers: int = 1
    decoder_heads: int = 4
    decoder_ff_dim: int = 256
    dt_layers: int = 4
    dt_heads: int = 4
    dt_ff_mult: int = 4
    context_length: int = 8
    max_timestep: int = 4096
    dropout: float = 0.1
    mask_ratio: float = 0.75
    mask_strategy: str = "learned"
    reconstruction_loss_weight: float = 1.0
    gaze_loss_weight: float = 0.1


class ActiveGazeVisualEncoderOutput(NamedTuple):
    state_embedding: torch.Tensor
    encoded_visible_tokens: torch.Tensor
    reconstructed_patches: torch.Tensor | None
    target_patches: torch.Tensor
    mask_probs: torch.Tensor
    gaze_patch_target: torch.Tensor | None
    visible_indices: torch.Tensor
    masked_positions: torch.Tensor
    reconstruction_loss: torch.Tensor | None
    gaze_loss: torch.Tensor | None


class ActiveGazeDecisionTransformerOutput(NamedTuple):
    loss: torch.Tensor | None
    action_loss: torch.Tensor | None
    reconstruction_loss: torch.Tensor | None
    gaze_loss: torch.Tensor | None
    action_logits: torch.Tensor
    mask_probs: torch.Tensor
    visible_indices: torch.Tensor
    state_embeddings: torch.Tensor


class ActiveGazeBehaviorClonerOutput(NamedTuple):
    loss: torch.Tensor | None
    action_loss: torch.Tensor | None
    reconstruction_loss: torch.Tensor | None
    gaze_loss: torch.Tensor | None
    action_logits: torch.Tensor
    mask_probs: torch.Tensor
    visible_indices: torch.Tensor
    state_embeddings: torch.Tensor


class CausalSelfAttention(nn.Module):
    """minGPT-style causal self-attention block used by the Decision Transformer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_tokens: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_tokens, max_tokens))
        self.register_buffer("mask", mask.view(1, 1, max_tokens, max_tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        key = self.key(x).view(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)
        query = self.query(x).view(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention = attention.masked_fill(self.mask[:, :, :token_count, :token_count] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, token_count, embed_dim)
        return self.resid_drop(self.proj(output))


class CausalBlock(nn.Module):
    """Transformer block matching the causal minGPT pattern used by official DT."""

    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int, dropout: float, max_tokens: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, max_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ff_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ActiveGazeMAEVisualEncoder(nn.Module):
    """MAE visual encoder whose visible patches are selected by a gaze-supervised policy."""

    def __init__(self, cfg: ActiveGazeDecisionTransformerConfig) -> None:
        super().__init__()
        if cfg.image_height % cfg.patch_size != 0 or cfg.image_width % cfg.patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")
        if cfg.embed_dim % cfg.encoder_heads != 0:
            raise ValueError("embed_dim must be divisible by encoder_heads")
        if cfg.decoder_dim % cfg.decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads")
        if not 0.0 <= cfg.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        if cfg.mask_strategy not in {"learned", "random"}:
            raise ValueError("mask_strategy must be 'learned' or 'random'")

        self.cfg = cfg
        self.grid_height = cfg.image_height // cfg.patch_size
        self.grid_width = cfg.image_width // cfg.patch_size
        self.num_patches = self.grid_height * self.grid_width
        self.visible_count = max(1, int(round((1.0 - cfg.mask_ratio) * self.num_patches)))
        self.patch_dim = cfg.frame_channels * cfg.patch_size * cfg.patch_size

        self.patch_embed = nn.Conv2d(
            cfg.frame_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        self.mask_policy = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.encoder_ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.encoder_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.decoder_dim,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=cfg.decoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(cfg.decoder_dim),
            nn.Linear(cfg.decoder_dim, self.patch_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def patchify(self, frames: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            frames,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )
        return patches.transpose(1, 2)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        folded = F.fold(
            patches.transpose(1, 2),
            output_size=(self.cfg.image_height, self.cfg.image_width),
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )
        return folded

    def gaze_to_patch_distribution(self, gaze_heatmaps: torch.Tensor) -> torch.Tensor:
        if gaze_heatmaps.ndim == 3:
            gaze_heatmaps = gaze_heatmaps.unsqueeze(1)
        if gaze_heatmaps.shape[1:] != (1, self.cfg.image_height, self.cfg.image_width):
            raise ValueError("gaze_heatmaps must have shape [batch, 1, image_height, image_width]")
        patch_mass = F.unfold(
            gaze_heatmaps,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        ).sum(dim=1)
        total = patch_mass.sum(dim=1, keepdim=True)
        uniform = torch.full_like(patch_mass, 1.0 / self.num_patches)
        normalized = patch_mass / total.clamp_min(1e-8)
        return torch.where(total > 0, normalized, uniform)

    def _patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(frames)
        return tokens.flatten(2).transpose(1, 2)

    def _select_visible_indices(self, mask_probs: torch.Tensor) -> torch.Tensor:
        indices = torch.topk(mask_probs, k=self.visible_count, dim=1).indices
        return torch.sort(indices, dim=1).values

    def _random_visible_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        scores = torch.rand(batch_size, self.num_patches, device=device)
        indices = torch.topk(scores, k=self.visible_count, dim=1).indices
        return torch.sort(indices, dim=1).values

    def forward(
        self,
        frames: torch.Tensor,
        gaze_heatmaps: torch.Tensor | None = None,
        *,
        reconstruct: bool = True,
    ) -> ActiveGazeVisualEncoderOutput:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [batch, channels, height, width]")
        if frames.shape[1:] != (
            self.cfg.frame_channels,
            self.cfg.image_height,
            self.cfg.image_width,
        ):
            raise ValueError("frames shape does not match the visual encoder config")

        batch_size = frames.shape[0]
        patch_tokens = self._patch_tokens(frames)
        patch_tokens = patch_tokens + self.encoder_pos_embed
        if self.cfg.mask_strategy == "random":
            mask_probs = torch.full(
                (batch_size, self.num_patches),
                1.0 / self.num_patches,
                dtype=patch_tokens.dtype,
                device=patch_tokens.device,
            )
            visible_indices = self._random_visible_indices(batch_size, patch_tokens.device)
        else:
            mask_logits = self.mask_policy(patch_tokens).squeeze(-1)
            mask_probs = F.softmax(mask_logits, dim=1)
            visible_indices = self._select_visible_indices(mask_probs)
        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.cfg.embed_dim)
        visible_tokens = torch.gather(patch_tokens, dim=1, index=gather_index)
        encoded_visible = self.encoder(visible_tokens)
        encoded_visible = self.encoder_norm(encoded_visible)
        state_embedding = encoded_visible.mean(dim=1)

        masked_positions = torch.ones(
            batch_size,
            self.num_patches,
            dtype=torch.bool,
            device=frames.device,
        )
        masked_positions.scatter_(1, visible_indices, False)

        target_patches = self.patchify(frames)
        reconstructed_patches = None
        reconstruction_loss = None
        if reconstruct:
            decoder_visible = self.decoder_embed(encoded_visible)
            decoder_tokens = self.mask_token.to(dtype=decoder_visible.dtype).expand(batch_size, self.num_patches, -1).clone()
            scatter_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.cfg.decoder_dim)
            decoder_tokens.scatter_(1, scatter_index, decoder_visible)
            decoder_tokens = decoder_tokens + self.decoder_pos_embed.to(dtype=decoder_tokens.dtype)
            decoded = self.decoder(decoder_tokens)
            reconstructed_patches = self.decoder_head(decoded)
            if masked_positions.any():
                reconstruction_loss = F.mse_loss(
                    reconstructed_patches[masked_positions],
                    target_patches[masked_positions],
                )
            else:
                reconstruction_loss = reconstructed_patches.new_tensor(0.0)

        gaze_patch_target = None
        gaze_loss = None
        if gaze_heatmaps is not None:
            gaze_patch_target = self.gaze_to_patch_distribution(gaze_heatmaps)
            if self.cfg.mask_strategy == "learned":
                gaze_loss = torch.sum(
                    gaze_patch_target
                    * (torch.log(gaze_patch_target.clamp_min(1e-8)) - torch.log(mask_probs.clamp_min(1e-8))),
                    dim=1,
                ).mean()

        return ActiveGazeVisualEncoderOutput(
            state_embedding=state_embedding,
            encoded_visible_tokens=encoded_visible,
            reconstructed_patches=reconstructed_patches,
            target_patches=target_patches,
            mask_probs=mask_probs,
            gaze_patch_target=gaze_patch_target,
            visible_indices=visible_indices,
            masked_positions=masked_positions,
            reconstruction_loss=reconstruction_loss,
            gaze_loss=gaze_loss,
        )


class GazeMaskedDecisionTransformer(nn.Module):
    """Return-conditioned causal transformer over active-MAE state embeddings."""

    def __init__(self, cfg: ActiveGazeDecisionTransformerConfig) -> None:
        super().__init__()
        if cfg.embed_dim % cfg.dt_heads != 0:
            raise ValueError("embed_dim must be divisible by dt_heads")
        self.cfg = cfg
        self.max_tokens = cfg.context_length * 3
        self.return_embedding = nn.Sequential(nn.Linear(1, cfg.embed_dim), nn.Tanh())
        self.action_embedding = nn.Sequential(nn.Embedding(cfg.action_dim, cfg.embed_dim), nn.Tanh())
        self.local_pos_embedding = nn.Parameter(torch.zeros(1, self.max_tokens, cfg.embed_dim))
        self.timestep_embedding = nn.Embedding(cfg.max_timestep + 1, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(
            *[
                CausalBlock(
                    cfg.embed_dim,
                    cfg.dt_heads,
                    cfg.dt_ff_mult,
                    cfg.dropout,
                    self.max_tokens,
                )
                for _ in range(cfg.dt_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.action_head = nn.Linear(cfg.embed_dim, cfg.action_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.local_pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.action_embedding[0].weight, mean=0.0, std=0.02)

    def forward(
        self,
        state_embeddings: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if state_embeddings.ndim != 3:
            raise ValueError("state_embeddings must have shape [batch, time, embed_dim]")
        batch_size, context_length, embed_dim = state_embeddings.shape
        if context_length > self.cfg.context_length:
            raise ValueError("input context exceeds configured context_length")
        if embed_dim != self.cfg.embed_dim:
            raise ValueError("state embedding dimension does not match config")
        if actions.shape != (batch_size, context_length):
            raise ValueError("actions must have shape [batch, time]")
        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        if returns_to_go.shape != (batch_size, context_length, 1):
            raise ValueError("returns_to_go must have shape [batch, time] or [batch, time, 1]")
        if timesteps.shape != (batch_size, context_length):
            raise ValueError("timesteps must have shape [batch, time]")
        if actions.min().item() < 0 or actions.max().item() >= self.cfg.action_dim:
            raise ValueError("actions contain values outside [0, action_dim)")

        return_tokens = self.return_embedding(returns_to_go.to(dtype=state_embeddings.dtype))
        action_tokens = self.action_embedding(actions)
        token_embeddings = torch.zeros(
            batch_size,
            context_length * 3,
            self.cfg.embed_dim,
            dtype=state_embeddings.dtype,
            device=state_embeddings.device,
        )
        token_embeddings[:, 0::3, :] = return_tokens
        token_embeddings[:, 1::3, :] = state_embeddings
        token_embeddings[:, 2::3, :] = action_tokens

        clamped_timesteps = timesteps.clamp(min=0, max=self.cfg.max_timestep)
        timestep_tokens = clamped_timesteps.repeat_interleave(3, dim=1)
        position_embeddings = self.timestep_embedding(timestep_tokens)
        position_embeddings = position_embeddings + self.local_pos_embedding[:, : token_embeddings.shape[1], :]

        hidden = self.drop(token_embeddings + position_embeddings)
        hidden = self.blocks(hidden)
        hidden = self.ln_f(hidden)
        logits = self.action_head(hidden)
        return logits[:, 1::3, :]


class ActiveGazeDecisionTransformer(nn.Module):
    """End-to-end active-gaze visual encoder plus Decision Transformer."""

    def __init__(self, cfg: ActiveGazeDecisionTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual_encoder = ActiveGazeMAEVisualEncoder(cfg)
        self.decision_transformer = GazeMaskedDecisionTransformer(cfg)

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        gaze_heatmaps: torch.Tensor | None = None,
        *,
        compute_auxiliary: bool = True,
    ) -> ActiveGazeDecisionTransformerOutput:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        batch_size, context_length = frames.shape[:2]
        if context_length > self.cfg.context_length:
            raise ValueError("input context exceeds configured context_length")

        flat_frames = frames.reshape(
            batch_size * context_length,
            self.cfg.frame_channels,
            self.cfg.image_height,
            self.cfg.image_width,
        )
        flat_gaze = None
        if gaze_heatmaps is not None:
            if gaze_heatmaps.ndim == 4:
                gaze_heatmaps = gaze_heatmaps.unsqueeze(2)
            if gaze_heatmaps.shape[:2] != (batch_size, context_length):
                raise ValueError("gaze_heatmaps must match [batch, time, ...]")
            flat_gaze = gaze_heatmaps.reshape(
                batch_size * context_length,
                1,
                self.cfg.image_height,
                self.cfg.image_width,
            )
        visual = self.visual_encoder(
            flat_frames,
            flat_gaze,
            reconstruct=compute_auxiliary,
        )
        state_embeddings = visual.state_embedding.reshape(batch_size, context_length, self.cfg.embed_dim)
        action_logits = self.decision_transformer(
            state_embeddings,
            actions,
            returns_to_go,
            timesteps,
        )
        action_loss = F.cross_entropy(action_logits.reshape(-1, self.cfg.action_dim), actions.reshape(-1))

        reconstruction_loss = visual.reconstruction_loss
        gaze_loss = visual.gaze_loss
        loss = action_loss
        if reconstruction_loss is not None:
            loss = loss + self.cfg.reconstruction_loss_weight * reconstruction_loss
        if gaze_loss is not None:
            loss = loss + self.cfg.gaze_loss_weight * gaze_loss

        return ActiveGazeDecisionTransformerOutput(
            loss=loss,
            action_loss=action_loss,
            reconstruction_loss=reconstruction_loss,
            gaze_loss=gaze_loss,
            action_logits=action_logits,
            mask_probs=visual.mask_probs.reshape(batch_size, context_length, -1),
            visible_indices=visual.visible_indices.reshape(batch_size, context_length, -1),
            state_embeddings=state_embeddings,
        )


class ActiveGazeBehaviorCloner(nn.Module):
    """One-step action model for debugging the active-gaze MAE visual encoder."""

    def __init__(self, cfg: ActiveGazeDecisionTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual_encoder = ActiveGazeMAEVisualEncoder(cfg)
        self.action_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.action_dim),
        )

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor | None = None,
        gaze_heatmaps: torch.Tensor | None = None,
        *,
        compute_auxiliary: bool = True,
    ) -> ActiveGazeBehaviorClonerOutput:
        visual = self.visual_encoder(
            frames,
            gaze_heatmaps,
            reconstruct=compute_auxiliary,
        )
        action_logits = self.action_head(visual.state_embedding)
        action_loss = None
        loss = None
        if actions is not None:
            action_loss = F.cross_entropy(action_logits, actions)
            loss = action_loss
            if visual.reconstruction_loss is not None:
                loss = loss + self.cfg.reconstruction_loss_weight * visual.reconstruction_loss
            if visual.gaze_loss is not None:
                loss = loss + self.cfg.gaze_loss_weight * visual.gaze_loss

        return ActiveGazeBehaviorClonerOutput(
            loss=loss,
            action_loss=action_loss,
            reconstruction_loss=visual.reconstruction_loss,
            gaze_loss=visual.gaze_loss,
            action_logits=action_logits,
            mask_probs=visual.mask_probs,
            visible_indices=visual.visible_indices,
            state_embeddings=visual.state_embedding,
        )
