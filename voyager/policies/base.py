"""Policy interfaces for Voyager baseline agents."""

from typing import Protocol

import numpy as np

Observation = dict[str, np.ndarray]
Info = dict[str, object]


class Policy(Protocol):
    """Minimal action-selection interface for non-learning policies."""

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        """Return a discrete Voyager action."""
        ...
