"""Stage 0 placeholder environments.

These classes exist so the package can register planned environment IDs without
pretending that the Stage 1 simulation is implemented.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MULTI_AGENT_MESSAGE = (
    "VoyagerSurvival-v0 is planned for a later multi-agent stage and is not implemented yet."
)


class MultiAgentPlaceholderEnv(gym.Env):
    """Gymnasium placeholder for later multi-agent Voyager environments."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        raise NotImplementedError(MULTI_AGENT_MESSAGE)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _ = action
        raise NotImplementedError(MULTI_AGENT_MESSAGE)
