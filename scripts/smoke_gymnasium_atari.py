"""Smoke-test a Gymnasium ALE Atari environment."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.25)
    parser.add_argument("--full-action-space", action="store_true")
    args = parser.parse_args()

    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(
        args.env_id,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=args.full_action_space,
    )
    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0
    completed_steps = 0
    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        completed_steps += 1
        if terminated or truncated:
            observation, info = env.reset()
    print(f"env_id={args.env_id}")
    print(f"obs_shape={tuple(observation.shape)} obs_dtype={observation.dtype}")
    print(f"action_space={env.action_space}")
    print(f"steps={completed_steps} total_reward={total_reward:.3f}")
    print(f"obs_min={int(np.min(observation))} obs_max={int(np.max(observation))}")
    env.close()


if __name__ == "__main__":
    main()
