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
    food_origins: list[str | None] = field(default_factory=list)


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
    food_high_watermark: int = 0
    food_origins: list[str] = field(default_factory=list)
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
    achievements: set[str] = field(default_factory=set)
    achievement_steps: dict[str, int] = field(default_factory=dict)
    total_deposits: int = 0
    total_withdrawals: int = 0
    total_build_actions: int = 0
    gathered_resources: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )
    deposited_resources: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )
    consumed_resources: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )
    constructed_resources: dict[str, int] = field(
        default_factory=lambda: {"food": 0, "wood": 0, "stone": 0}
    )
    contributing_roles: set[str] = field(default_factory=set)
    food_security_steps: int = 0
    max_food_security_steps: int = 0
    shelter_completion_step: int | None = None
    storm_was_active: bool = False
