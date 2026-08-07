"""Dataclasses for Voyager single-agent world state."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FoodLot:
    """One deterministic food lot with auditable provenance and expiry."""

    id: str
    kind: str
    quantity: int
    origin_type: str
    origin_id: str
    created_tick: int
    expires_tick: int | None
    preparation: str


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
    inventory: dict[str, int] = field(default_factory=lambda: {"food": 0, "wood": 0, "stone": 0})
    food_origins: list[str | None] = field(default_factory=list)
    tools: set[str] = field(default_factory=set)
    equipped_tool: str | None = None
    sheltered: bool = False
    life_state: str = "active"
    downed_ticks: int = 0
    downed_count: int = 0
    revival_labor: int = 0
    revival_food_lot_id: str | None = None
    food_lots: list[FoodLot] = field(default_factory=list)
    tool_charges: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class GroundPileState:
    """A world item pile created by a recorded simulation event."""

    id: str
    x: int
    y: int
    item: str
    quantity: int
    origin_type: str
    origin_id: str
    created_tick: int
    expires_tick: int | None = None


@dataclass(slots=True)
class StructureState:
    """One public structure in the shared Stage 7A camp."""

    id: str
    type: str
    x: int
    y: int
    required_materials: dict[str, int]
    required_labor: int
    reserved_materials: dict[str, int] = field(default_factory=dict)
    labor: int = 0
    condition: int = 100
    capacity: int = 0
    occupants: set[str] = field(default_factory=set)
    fuel: int = 0
    occupancy_order: list[str] = field(default_factory=list)
    repair_labor: int = 0
    repair_material_reserved: bool = False

    @property
    def progress(self) -> float:
        if self.required_labor <= 0:
            return 1.0
        return min(1.0, self.labor / self.required_labor)

    @property
    def complete(self) -> bool:
        return self.progress >= 1.0


@dataclass(slots=True)
class CreatureState:
    """One huntable or hostile non-agent creature."""

    id: str
    type: str
    x: int
    y: int
    health: int
    max_health: int
    alive: bool = True
    target: str | None = None
    spawn_tick: int = 0
    behavior: str = "idle"


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
    stockpile: dict[str, int] = field(default_factory=lambda: {"food": 0, "wood": 0, "stone": 0})
    food_high_watermark: int = 0
    food_origins: list[str] = field(default_factory=list)
    shelter_progress: float = 0.0
    shelter_capacity: int = 0
    food_lots: list[FoodLot] = field(default_factory=list)
    tool_stockpile: dict[str, list[int]] = field(default_factory=dict)


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
    scenario_id: str = "stage5_5_standard_300_v1"
    structures: dict[str, StructureState] = field(default_factory=dict)
    creatures: dict[str, CreatureState] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    last_spawn_count: int = 0
    last_spawn_positions: list[tuple[int, int]] = field(default_factory=list)
    hunts: int = 0
    cooked_meals: int = 0
    monster_defeats: int = 0
    prevented_damage: int = 0
    full_fire_night_ticks: int = 0
    full_shelter_night_ticks: int = 0
    ground_piles: dict[str, GroundPileState] = field(default_factory=dict)
    ledger: list[dict[str, object]] = field(default_factory=list)
    spoiled_resources: dict[str, int] = field(default_factory=dict)
    rescue_success: bool = False
