"""Run a random policy in the Stage 1 Voyager single-agent environment."""

from __future__ import annotations

import gymnasium as gym

import voyager

ENV_ID = "VoyagerSingleAgent-v0"


def main() -> int:
    """Run a full random rollout and print a compact final summary."""

    print(f"Voyager {voyager.__version__} random rollout")

    env = gym.make(ENV_ID)
    _obs, info = env.reset(seed=0)

    done = False
    total_reward = 0.0
    terminated = False
    truncated = False
    while not done:
        action = env.action_space.sample()
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"steps: {info['step']}")
    print(f"total_reward: {total_reward:.3f}")
    print(
        "final_state: "
        f"health={info['health']:.1f} "
        f"hunger={info['hunger']:.1f} "
        f"energy={info['energy']:.1f}"
    )
    print(f"inventory: {info['inventory']}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
