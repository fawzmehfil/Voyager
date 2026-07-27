"""Dataclasses for Voyager single-agent world state."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class AgentState:
    """Mutable survival state for the Stage 1 agent."""

    x: int
    y: int
    role: str = "forager"
    health: float = 100.0
    hunger: float = 15.0
    energy: float = 100.0
    alive: bool = True
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


@dataclass(slots=True)
class CampState:
    """Shared camp state for the multi-agent survival environment."""

    x: int
    y: int
    stockpile: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )
    shelter_progress: float = 0.0
    shelter_capacity: int = 0


@dataclass(slots=True)
class MultiAgentWorldState:
    """Complete multi-agent island state."""

    terrain: np.ndarray
    resource_ids: np.ndarray
    resource_quantities: np.ndarray
    agents: dict[str, AgentState]
    camp: CampState
    step_count: int = 0
    deaths: int = 0
