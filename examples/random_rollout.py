"""Stage 0 placeholder showing the intended future random rollout API."""

from __future__ import annotations

import gymnasium as gym

import voyager

ENV_ID = "VoyagerSingleAgent-v0"


def main() -> int:
    """Create the planned environment and report that Stage 1 will implement it."""

    print(f"Voyager {voyager.__version__} random rollout placeholder")

    env = gym.make(ENV_ID)
    try:
        _obs, _info = env.reset(seed=0)
    except NotImplementedError as exc:
        print(exc)
        return 0

    done = False
    while not done:
        action = env.action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
