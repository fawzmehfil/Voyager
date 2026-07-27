"""Dataclasses for Voyager single-agent world state."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class AgentState:
    """Mutable survival state for the Stage 1 agent."""

    x: int
    y: int
    health: float = 100.0
    hunger: float = 15.0
    energy: float = 100.0
    inventory: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )


@dataclass(slots=True)
class WorldState:
    """Complete single-agent island state."""

    terrain: np.ndarray
    resource_ids: np.ndarray
    resource_quantities: np.ndarray
    agent: AgentState
    step_count: int = 0
