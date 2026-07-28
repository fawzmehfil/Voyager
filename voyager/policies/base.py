"""Policy interfaces for Voyager baseline agents."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Observation = dict[str, np.ndarray]
Info = dict[str, object]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One selected action plus optional pre-mask policy diagnostics."""

    action: int
    raw_action: int
    invalid_probability_mass: float = 0.0


class Policy(Protocol):
    """Minimal action-selection interface for non-learning policies."""

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        """Return a discrete Voyager action."""
        ...
