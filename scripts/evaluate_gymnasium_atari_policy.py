"""Evaluate trained amsterg/ahead policies in current Gymnasium ALE environments."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_AHEAD_ROOT = ROOT / "external" / "amsterg_ahead"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "gymnasium_eval" / "breakout_smoke.json"
GAME_TO_ENV_ID = {
    "alien": "ALE/Alien-v5",
    "asterix": "ALE/Asterix-v5",
    "bank_heist": "ALE/BankHeist-v5",
    "berzerk": "ALE/Berzerk-v5",
    "breakout": "ALE/Breakout-v5",
    "centipede": "ALE/Centipede-v5",
    "demon_attack": "ALE/DemonAttack-v5",
    "enduro": "ALE/Enduro-v5",
    "freeway": "ALE/Freeway-v5",
    "frostbite": "ALE/Frostbite-v5",
    "hero": "ALE/Hero-v5",
    "montezuma_revenge": "ALE/MontezumaRevenge-v5",
    "ms_pacman": "ALE/MsPacman-v5",
    "name_this_game": "ALE/NameThisGame-v5",
    "phoenix": "ALE/Phoenix-v5",
    "riverraid": "ALE/Riverraid-v5",
    "road_runner": "ALE/RoadRunner-v5",
    "seaquest": "ALE/Seaquest-v5",
    "space_invaders": "ALE/SpaceInvaders-v5",
    "venture": "ALE/Venture-v5",
}


def configure_ahead(ahead_root: Path) -> None:
    """Configure imports for the vendored amsterg/ahead model classes."""

    os.environ.setdefault("MPLCONFIGDIR", str(ahead_root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ahead_root / ".cache"))
    sys.path.insert(0, str(ahead_root))
    os.chdir(ahead_root)


def load_state_dict(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    """Load a checkpoint saved by scripts/train_amsterg_models.py."""

    state = torch.load(checkpoint, map_location=device)
    if "model_state_dict" not in state:
        raise KeyError(f"checkpoint does not contain model_state_dict: {checkpoint}")
    model.load_state_dict(state["model_state_dict"])


def load_active_dt_checkpoint(checkpoint: Path, device: torch.device) -> nn.Module:
    """Load an active-gaze Decision Transformer checkpoint."""

    from atari_gaze_cmae import ActiveGazeDecisionTransformer, ActiveGazeDecisionTransformerConfig

    state = torch.load(checkpoint, map_location=device)
    if "model_config" not in state or "model_state_dict" not in state:
        raise KeyError(f"checkpoint is not an active-gaze DT checkpoint: {checkpoint}")
    cfg = ActiveGazeDecisionTransformerConfig(**state["model_config"])
    model = ActiveGazeDecisionTransformer(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def normalized_gaze_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Convert CNN_GAZE log probabilities into normalized 84x84 maps."""

    gaze = torch.exp(log_probs).view(-1, 84, 84)
    flat = gaze.flatten(1)
    min_values = flat.min(dim=1).values.view(-1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1)
    gaze = (gaze - min_values) / (max_values - min_values + 1e-8)
    return gaze / (gaze.flatten(1).sum(dim=1).view(-1, 1, 1) + 1e-8)


def preprocess_frame_stack(frame_stack: deque[np.ndarray], transform: Any, device: torch.device) -> torch.Tensor:
    """Convert the current RGB Atari frame stack to an amsterg-compatible tensor."""

    frames = [transform(frame).squeeze(0) for frame in frame_stack]
    return torch.stack(frames).unsqueeze(0).to(device=device, dtype=torch.float32)


def preprocess_active_dt_frame_stack(frame_stack: deque[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert the current RGB Atari frame stack to the active-DT tensor format."""

    frames = []
    for frame in frame_stack:
        image = Image.fromarray(frame).convert("L").resize((84, 84), Image.BILINEAR)
        frames.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0))
    return torch.stack(frames).unsqueeze(0).to(device=device, dtype=torch.float32)


def make_env(args: argparse.Namespace) -> gym.Env:
    """Create a current Gymnasium ALE environment with reproducible action-space settings."""

    import ale_py

    gym.register_envs(ale_py)
    return gym.make(
        args.env_id,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=not args.reduced_action_space,
    )


def build_models(args: argparse.Namespace) -> tuple[nn.Module | None, nn.Module | None]:
    """Instantiate and load the selected amsterg/ahead action and gaze models."""

    if args.model_type == "random":
        return None, None
    if args.model_type == "active-dt":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for model_type=active-dt")
        return load_active_dt_checkpoint(args.checkpoint, args.device), None

    configure_ahead(args.ahead_root)
    from src.models.action_sl import ACTION_SL
    from src.models.cnn_gaze import CNN_GAZE
    from src.models.gazed_action_sl import GAZED_ACTION_SL
    from src.models.selective_gaze_only import SGAZED_ACTION_SL

    action_model: nn.Module | None = None
    gaze_model: nn.Module | None = None
    if args.model_type == "bc":
        action_model = ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
    elif args.model_type == "agil":
        action_model = GAZED_ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
    elif args.model_type == "sea":
        action_model = SGAZED_ACTION_SL(game=args.game, mode="eval", device=args.device).to(args.device)
    else:
        raise ValueError(f"unsupported model_type: {args.model_type}")

    if action_model is not None:
        if args.checkpoint is None:
            raise SystemExit(f"--checkpoint is required for model_type={args.model_type}")
        load_state_dict(action_model, args.checkpoint, args.device)
        action_model.eval()

    if args.model_type in {"agil", "sea"}:
        if args.gaze_checkpoint is None:
            raise SystemExit(f"--gaze-checkpoint is required for model_type={args.model_type}")
        gaze_model = CNN_GAZE(game=args.game, mode="eval", device=args.device).to(args.device)
        load_state_dict(gaze_model, args.gaze_checkpoint, args.device)
        gaze_model.eval()

    return action_model, gaze_model


def action_logits(
    model_type: str,
    action_model: nn.Module,
    gaze_model: nn.Module | None,
    frames: torch.Tensor,
) -> torch.Tensor:
    """Run the selected amsterg/ahead action model on one frame stack."""

    if model_type == "bc":
        return action_model(frames)
    if gaze_model is None:
        raise RuntimeError("gaze_model is required for gaze-augmented policies")
    with torch.no_grad():
        gaze = normalized_gaze_from_log_probs(gaze_model(frames))
    if model_type == "agil":
        overlay = frames * gaze.unsqueeze(1).repeat(1, frames.shape[1], 1, 1)
        return action_model(frames, overlay)
    if model_type == "sea":
        last_frame = frames[:, -1:].contiguous()
        overlay = (last_frame * gaze.unsqueeze(1)).contiguous()
        return action_model(last_frame, overlay)
    raise ValueError(f"unsupported model_type: {model_type}")


def choose_action(
    logits: torch.Tensor,
    policy: str,
    temperature: float,
    generator: torch.Generator,
) -> tuple[int, float]:
    """Choose an action from logits and return its probability."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits.squeeze(0) / temperature, dim=0)
    if policy == "argmax":
        action = int(torch.argmax(probs).item())
    elif policy == "sample":
        sample_probs = probs.detach().cpu()
        action = int(torch.multinomial(sample_probs, num_samples=1, generator=generator).item())
    else:
        raise ValueError(f"unsupported policy: {policy}")
    return action, float(probs[action].item())


def parse_start_actions(raw_actions: str) -> list[int]:
    """Parse a comma-separated list of environment actions to apply after reset."""

    if not raw_actions.strip():
        return []
    return [int(action.strip()) for action in raw_actions.split(",") if action.strip()]


def run_episode(
    env: gym.Env,
    args: argparse.Namespace,
    action_model: nn.Module | None,
    gaze_model: nn.Module | None,
    transform: Any,
    torch_generator: torch.Generator,
    episode_seed: int,
) -> dict[str, Any]:
    """Run one evaluation episode and return reward, length, and action statistics."""

    observation, info = env.reset(seed=episode_seed)
    total_reward = 0.0
    action_counts: dict[str, int] = {}
    action_prob_sum = 0.0
    steps = 0
    terminated = False
    truncated = False
    start_actions = parse_start_actions(args.start_actions)
    for start_action in start_actions:
        observation, reward, terminated, truncated, info = env.step(start_action)
        total_reward += float(reward)
        if terminated or truncated:
            observation, info = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            break
    frame_stack: deque[np.ndarray] = deque([observation] * args.stack, maxlen=args.stack)
    active_states: list[torch.Tensor] = []
    active_actions: list[int] = []
    active_rtgs: list[float] = []
    active_timesteps: list[int] = []
    current_rtg = float(args.target_return)
    model_context_length = args.context_length
    if args.model_type == "active-dt" and action_model is not None:
        model_context_length = min(model_context_length, int(action_model.cfg.context_length))
    while steps < args.max_steps:
        if args.model_type == "random":
            action = int(env.action_space.sample())
            action_prob = 1.0 / int(env.action_space.n)
        elif args.model_type == "active-dt":
            assert action_model is not None
            current_state = preprocess_active_dt_frame_stack(frame_stack, args.device).squeeze(0)
            active_states.append(current_state)
            active_rtgs.append(current_rtg)
            active_timesteps.append(steps)
            context = min(model_context_length, len(active_states))
            frames = torch.stack(active_states[-context:], dim=0).unsqueeze(0)
            previous_actions = active_actions[-(context - 1) :] if context > 1 else []
            action_tokens = previous_actions + [0]
            actions = torch.tensor([action_tokens], dtype=torch.long, device=args.device)
            rtg = torch.tensor([active_rtgs[-context:]], dtype=torch.float32, device=args.device)
            timesteps = torch.tensor([active_timesteps[-context:]], dtype=torch.long, device=args.device)
            with torch.no_grad():
                output = action_model(
                    frames=frames,
                    actions=actions,
                    returns_to_go=rtg,
                    timesteps=timesteps,
                    gaze_heatmaps=None,
                    compute_auxiliary=False,
                )
                logits = output.action_logits[:, -1, :]
            action, action_prob = choose_action(logits, args.policy, args.temperature, torch_generator)
        else:
            assert action_model is not None
            frames = preprocess_frame_stack(frame_stack, transform, args.device)
            with torch.no_grad():
                logits = action_logits(args.model_type, action_model, gaze_model, frames)
            action, action_prob = choose_action(logits, args.policy, args.temperature, torch_generator)
        observation, reward, terminated, truncated, info = env.step(action)
        if args.model_type == "active-dt":
            active_actions.append(action)
            current_rtg -= float(reward)
            if len(active_states) > model_context_length:
                active_states = active_states[-model_context_length :]
                active_rtgs = active_rtgs[-model_context_length :]
                active_timesteps = active_timesteps[-model_context_length :]
                active_actions = active_actions[-model_context_length :]
        frame_stack.append(observation)
        action_counts[str(action)] = action_counts.get(str(action), 0) + 1
        action_prob_sum += action_prob
        total_reward += float(reward)
        steps += 1
        if args.log_interval > 0 and steps % args.log_interval == 0:
            print(
                f"episode_seed={episode_seed} step={steps} reward={total_reward:.3f} "
                f"last_action={action} action_prob={action_prob:.4f}",
                flush=True,
            )
        if terminated or truncated:
            break
    return {
        "seed": episode_seed,
        "reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "action_counts": action_counts,
        "mean_chosen_action_prob": action_prob_sum / max(steps, 1),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate episode rewards and lengths."""

    rewards = np.asarray([result["reward"] for result in results], dtype=np.float64)
    steps = np.asarray([result["steps"] for result in results], dtype=np.float64)
    return {
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "std_reward": float(rewards.std(ddof=0)) if rewards.size else 0.0,
        "mean_steps": float(steps.mean()) if steps.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="breakout", choices=sorted(GAME_TO_ENV_ID))
    parser.add_argument("--env-id")
    parser.add_argument("--ahead-root", type=Path, default=DEFAULT_AHEAD_ROOT)
    parser.add_argument("--model-type", choices=["random", "bc", "agil", "sea", "active-dt"], default="random")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gaze-checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--target-return", type=float, default=20.0)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.25)
    parser.add_argument("--reduced-action-space", action="store_true")
    parser.add_argument("--policy", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--start-actions", default="", help="Comma-separated env actions to apply after reset.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--log-interval", type=int, default=0, help="Print rollout progress every N env steps.")
    parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print completed episode summaries every N episodes. Use 0 to disable per-episode summaries.",
    )
    args = parser.parse_args()

    args.ahead_root = args.ahead_root.resolve()
    args.checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    args.gaze_checkpoint = args.gaze_checkpoint.resolve() if args.gaze_checkpoint is not None else None
    args.output_json = args.output_json.resolve()
    args.env_id = args.env_id or GAME_TO_ENV_ID[args.game]
    args.device = torch.device(args.device)

    if args.stack != 4:
        raise SystemExit("amsterg/ahead models currently require --stack 4")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    action_model, gaze_model = build_models(args)
    transform = None
    if args.model_type in {"bc", "agil", "sea"}:
        from src.features.feat_utils import image_transforms

        transform = image_transforms()
    env = make_env(args)
    if int(env.action_space.n) != 18 and not args.reduced_action_space:
        raise RuntimeError(f"expected 18-action full action space, got {env.action_space}")

    episode_results = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        result = run_episode(
            env,
            args,
            action_model,
            gaze_model,
            transform,
            torch_generator,
            episode_seed,
        )
        episode_results.append(result)
        completed = episode + 1
        should_log_episode = (
            args.episode_log_interval > 0
            and (
                completed == 1
                or completed % args.episode_log_interval == 0
                or completed == args.episodes
            )
        )
        if should_log_episode:
            running_summary = summarize(episode_results)
            print(
                f"episode={completed}/{args.episodes} seed={episode_seed} "
                f"reward={result['reward']:.3f} steps={result['steps']} "
                f"terminated={result['terminated']} truncated={result['truncated']} "
                f"running_mean_reward={running_summary['mean_reward']:.3f}",
                flush=True,
            )
    env.close()

    output = {
        "game": args.game,
        "env_id": args.env_id,
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "gaze_checkpoint": str(args.gaze_checkpoint) if args.gaze_checkpoint is not None else None,
        "policy": args.policy,
        "temperature": args.temperature,
        "start_actions": parse_start_actions(args.start_actions),
        "target_return": args.target_return,
        "context_length": args.context_length,
        "frameskip": args.frameskip,
        "repeat_action_probability": args.repeat_action_probability,
        "full_action_space": not args.reduced_action_space,
        "episodes": episode_results,
        "summary": summarize(episode_results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    summary = output["summary"]
    print(
        f"mean_reward={summary['mean_reward']:.3f} "
        f"std_reward={summary['std_reward']:.3f} mean_steps={summary['mean_steps']:.1f}"
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
