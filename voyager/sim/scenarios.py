"""Versioned scenario profiles and deterministic Voyager islands."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from voyager.sim.constants import Resource, Terrain
from voyager.sim.state import AgentState, WorldState

COMPACT_SCENARIO_ID = "stage5_5_standard_300_v1"
CIVILIZATION_SCENARIO_ID = "voyager_civilization_vertical_slice_v1"
CIVILIZATION_MAP_SIZE = 48
CIVILIZATION_MAX_STEPS = 600
CIVILIZATION_CAMP = (24, 31)
CIVILIZATION_WORKBENCH = (23, 31)
CIVILIZATION_CAMPFIRE = (25, 31)
CIVILIZATION_SHELTER = (24, 29)
CIVILIZATION_DEER_SPAWNS = ((14, 18), (17, 15))
CIVILIZATION_STALKER_SPAWNS = (
    (8, 10),
    (39, 10),
    (42, 27),
    (8, 31),
    (17, 8),
    (34, 8),
)

ISLAND_BENCHMARK_SCENARIO_ID = "voyager_island_benchmark_v1"
ISLAND_BENCHMARK_MAP_SIZE = 48
ISLAND_BENCHMARK_MAX_STEPS = 1_200
ISLAND_BENCHMARK_CAMP = (24, 31)
ISLAND_BENCHMARK_WORKBENCH = (23, 31)
ISLAND_BENCHMARK_CAMPFIRE = (25, 31)
ISLAND_BENCHMARK_SHELTER = (24, 29)
ISLAND_BENCHMARK_BEACON = (24, 33)
ISLAND_BENCHMARK_STRUCTURE_SITES = {
    "workbench": ISLAND_BENCHMARK_WORKBENCH,
    "campfire": ISLAND_BENCHMARK_CAMPFIRE,
    "shelter": ISLAND_BENCHMARK_SHELTER,
    "beacon": ISLAND_BENCHMARK_BEACON,
}
ISLAND_BENCHMARK_STRUCTURE_SPECS = {
    "workbench": ({"wood": 3, "stone": 1}, 20, 0),
    "campfire": ({"wood": 2, "stone": 1}, 20, 0),
    "shelter": ({"wood": 4, "stone": 2}, 40, 2),
    "beacon": ({"wood": 4, "stone": 2}, 60, 0),
}
ISLAND_BENCHMARK_TOOL_RECIPES = {
    "axe": {"wood": 2, "stone": 1},
    "spear": {"wood": 2, "stone": 1},
}
ISLAND_BENCHMARK_RESCUE_DELAY = 100


@dataclass(frozen=True, slots=True)
class ScenarioMap:
    """One validated map plus the dynamic spawn sites selected for it."""

    state: WorldState
    deer_spawns: tuple[tuple[int, int], ...] = ()
    stalker_spawns: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Configuration that changes content without changing the shared engine."""

    id: str
    map_size: int
    max_steps: int
    civilization: bool = False
    island_benchmark: bool = False
    day_length: int = 300
    night_start: int = 200
    night_end: int = 300


COMPACT_SCENARIO = ScenarioDefinition(COMPACT_SCENARIO_ID, 32, 300)
CIVILIZATION_SCENARIO = ScenarioDefinition(
    CIVILIZATION_SCENARIO_ID,
    CIVILIZATION_MAP_SIZE,
    CIVILIZATION_MAX_STEPS,
    civilization=True,
)
ISLAND_BENCHMARK_SCENARIO = ScenarioDefinition(
    ISLAND_BENCHMARK_SCENARIO_ID,
    ISLAND_BENCHMARK_MAP_SIZE,
    ISLAND_BENCHMARK_MAX_STEPS,
    civilization=True,
    island_benchmark=True,
)

SCENARIO_REGISTRY = {
    COMPACT_SCENARIO.id: COMPACT_SCENARIO,
    CIVILIZATION_SCENARIO.id: CIVILIZATION_SCENARIO,
    ISLAND_BENCHMARK_SCENARIO.id: ISLAND_BENCHMARK_SCENARIO,
}


def scenario_definition(scenario_id: str) -> ScenarioDefinition:
    """Resolve a public scenario ID without embedding behavior checks in the engine."""

    try:
        return SCENARIO_REGISTRY[scenario_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Voyager scenario: {scenario_id!r}.") from exc


def build_civilization_island() -> WorldState:
    """Build the fixed 48x48 Stage 7A island using the existing array representation."""

    size = CIVILIZATION_MAP_SIZE
    terrain = np.full((size, size), Terrain.WATER, dtype=np.uint8)
    resources = np.full((size, size), Resource.NONE, dtype=np.uint8)
    quantities = np.zeros((size, size), dtype=np.uint8)

    yy, xx = np.mgrid[0:size, 0:size]
    center_x, center_y = 23.5, 24.0
    normalized = ((xx - center_x) / 21.5) ** 2 + ((yy - center_y) / 20.5) ** 2
    land = normalized <= 1.0
    beach = land & (normalized > 0.78)
    interior = land & ~beach
    terrain[beach] = Terrain.BEACH
    terrain[interior] = Terrain.GRASS

    forest = interior & (((xx < 20) & (yy < 32)) | ((xx < 14) & (yy < 38)))
    forest &= ((xx * 3 + yy * 5) % 11) < 8
    terrain[forest] = Terrain.FOREST
    resources[forest] = Resource.WOOD
    quantities[forest] = 3 + ((xx[forest] + yy[forest]) % 2).astype(np.uint8)

    highland = interior & (xx > 29) & (yy < 30)
    highland &= ((xx * 5 + yy * 2) % 9) < 7
    terrain[highland] = Terrain.ROCKY_HIGHLAND
    resources[highland] = Resource.STONE
    quantities[highland] = 3 + ((xx[highland] + yy[highland]) % 2).astype(np.uint8)

    berry_region = interior & (yy > 27) & (xx < 24) & (((xx + yy * 2) % 7) < 2)
    resources[berry_region] = Resource.FOOD
    quantities[berry_region] = 3

    for x, y in CIVILIZATION_STALKER_SPAWNS:
        terrain[y, x] = Terrain.CAVE
        resources[y, x] = Resource.NONE
        quantities[y, x] = 0

    camp_x, camp_y = CIVILIZATION_CAMP
    for y in range(camp_y - 4, camp_y + 4):
        for x in range(camp_x - 5, camp_x + 6):
            if 0 <= x < size and 0 <= y < size:
                terrain[y, x] = Terrain.GRASS
                resources[y, x] = Resource.NONE
                quantities[y, x] = 0

    # Nearby nodes keep the vertical slice focused on progression, while the larger zones
    # demonstrate the travel scale that procedural Stage 7C will exploit.
    nearby = {
        (19, 32): (Resource.FOOD, 8),
        (20, 33): (Resource.FOOD, 8),
        (18, 29): (Resource.WOOD, 12),
        (19, 28): (Resource.WOOD, 12),
        (29, 29): (Resource.STONE, 10),
        (30, 28): (Resource.STONE, 10),
    }
    for (x, y), (resource, quantity) in nearby.items():
        resources[y, x] = resource
        quantities[y, x] = quantity
        if resource == Resource.WOOD:
            terrain[y, x] = Terrain.FOREST
        elif resource == Resource.STONE:
            terrain[y, x] = Terrain.ROCKY_HIGHLAND

    validate_civilization_map(terrain, resources, quantities)
    return WorldState(
        terrain=terrain,
        resource_ids=resources,
        resource_quantities=quantities,
        agent=AgentState(x=camp_x, y=camp_y),
    )


def validate_civilization_map(
    terrain: np.ndarray,
    resources: np.ndarray,
    quantities: np.ndarray,
) -> None:
    """Reject malformed vertical-slice maps at reset time."""

    expected = (CIVILIZATION_MAP_SIZE, CIVILIZATION_MAP_SIZE)
    if terrain.shape != expected or resources.shape != expected or quantities.shape != expected:
        raise ValueError(f"Civilization map arrays must all have shape {expected}.")
    if np.any(terrain[0, :] != Terrain.WATER) or np.any(terrain[-1, :] != Terrain.WATER):
        raise ValueError("Civilization island must have a water border.")
    if np.any(terrain[:, 0] != Terrain.WATER) or np.any(terrain[:, -1] != Terrain.WATER):
        raise ValueError("Civilization island must have a water border.")
    camp_x, camp_y = CIVILIZATION_CAMP
    if terrain[camp_y, camp_x] == Terrain.WATER:
        raise ValueError("Civilization camp must be reachable land.")
    for resource in (Resource.FOOD, Resource.WOOD, Resource.STONE):
        if int(np.sum(quantities[resources == resource])) <= 0:
            raise ValueError(f"Civilization map is missing {resource.name.lower()}.")
    if len(CIVILIZATION_STALKER_SPAWNS) < 2:
        raise ValueError("Civilization map needs at least two stalker spawn candidates.")


def build_island_benchmark(
    rng: np.random.Generator,
    *,
    procedural: bool,
) -> ScenarioMap:
    """Build the fixed debug island or a deterministic procedural benchmark island."""

    size = ISLAND_BENCHMARK_MAP_SIZE
    terrain = np.full((size, size), Terrain.WATER, dtype=np.uint8)
    resources = np.full((size, size), Resource.NONE, dtype=np.uint8)
    quantities = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    normalized = ((xx - 23.5) / 21.5) ** 2 + ((yy - 24.0) / 20.5) ** 2
    land = normalized <= 1.0
    beach = land & (normalized > 0.78)
    terrain[beach] = Terrain.BEACH
    terrain[land & ~beach] = Terrain.GRASS

    reserved = {
        ISLAND_BENCHMARK_CAMP,
        *ISLAND_BENCHMARK_STRUCTURE_SITES.values(),
        (ISLAND_BENCHMARK_CAMP[0] - 1, ISLAND_BENCHMARK_CAMP[1]),
        (ISLAND_BENCHMARK_CAMP[0] + 1, ISLAND_BENCHMARK_CAMP[1]),
    }
    for x, y in reserved:
        terrain[y, x] = Terrain.GRASS

    candidates = [
        (x, y)
        for y in range(1, size - 1)
        for x in range(1, size - 1)
        if terrain[y, x] != Terrain.WATER and (x, y) not in reserved
    ]
    if procedural:
        food_positions = _sample_distance_band(candidates, rng, 4, 6, count=4, excluded=reserved)
        excluded = reserved | set(food_positions)
        wood_positions = _sample_distance_band(candidates, rng, 5, 8, count=4, excluded=excluded)
        excluded |= set(wood_positions)
        stone_positions = _sample_distance_band(candidates, rng, 6, 10, count=4, excluded=excluded)
        excluded |= set(stone_positions)
        deer_spawns = tuple(
            _sample_distance_band(candidates, rng, 8, 12, count=2, excluded=excluded)
        )
        excluded |= set(deer_spawns)
        stalker_spawns = tuple(
            _sample_distance_band(candidates, rng, 10, 30, count=8, excluded=excluded)
        )
    else:
        food_positions = [(20, 31), (24, 26), (29, 31), (24, 37)]
        wood_positions = [(18, 31), (24, 24), (32, 31), (20, 27)]
        stone_positions = [(34, 31), (24, 21), (18, 29), (30, 27)]
        deer_spawns = ((16, 31), (24, 19))
        stalker_spawns = (
            (12, 31),
            (36, 31),
            (24, 17),
            (24, 41),
            (14, 23),
            (34, 23),
            (14, 37),
            (34, 37),
        )

    for index, (x, y) in enumerate(food_positions):
        resources[y, x] = Resource.FOOD
        quantities[y, x] = 5
        terrain[y, x] = Terrain.GRASS
    for index, (x, y) in enumerate(wood_positions):
        resources[y, x] = Resource.WOOD
        quantities[y, x] = 9 if index else 8
        terrain[y, x] = Terrain.FOREST
    for x, y in stone_positions:
        resources[y, x] = Resource.STONE
        quantities[y, x] = 5
        terrain[y, x] = Terrain.ROCKY_HIGHLAND
    for x, y in stalker_spawns:
        terrain[y, x] = Terrain.CAVE

    validate_island_benchmark_map(
        terrain,
        resources,
        quantities,
        deer_spawns=deer_spawns,
        stalker_spawns=stalker_spawns,
    )
    camp_x, camp_y = ISLAND_BENCHMARK_CAMP
    return ScenarioMap(
        state=WorldState(
            terrain=terrain,
            resource_ids=resources,
            resource_quantities=quantities,
            agent=AgentState(x=camp_x, y=camp_y),
        ),
        deer_spawns=tuple(deer_spawns),
        stalker_spawns=tuple(stalker_spawns),
    )


def validate_island_benchmark_map(
    terrain: np.ndarray,
    resources: np.ndarray,
    quantities: np.ndarray,
    *,
    deer_spawns: tuple[tuple[int, int], ...],
    stalker_spawns: tuple[tuple[int, int], ...],
) -> None:
    """Reject islands that violate the frozen resource and reachability contract."""

    expected = (ISLAND_BENCHMARK_MAP_SIZE, ISLAND_BENCHMARK_MAP_SIZE)
    if any(array.shape != expected for array in (terrain, resources, quantities)):
        raise ValueError(f"Island benchmark arrays must all have shape {expected}.")
    if np.any(terrain[0, :] != Terrain.WATER) or np.any(terrain[-1, :] != Terrain.WATER):
        raise ValueError("Island benchmark must have a water border.")
    if np.any(terrain[:, 0] != Terrain.WATER) or np.any(terrain[:, -1] != Terrain.WATER):
        raise ValueError("Island benchmark must have a water border.")

    totals = {
        resource: int(np.sum(quantities[resources == resource]))
        for resource in (Resource.FOOD, Resource.WOOD, Resource.STONE)
    }
    minimums = {Resource.FOOD: 20, Resource.WOOD: 35, Resource.STONE: 20}
    for resource, minimum in minimums.items():
        if totals[resource] < minimum:
            raise ValueError(f"Island benchmark needs at least {minimum} {resource.name.lower()}.")
    if len(deer_spawns) < 2 or len(stalker_spawns) < 2:
        raise ValueError("Island benchmark needs two deer and two stalker candidates.")

    reachable = _reachable_land(terrain, ISLAND_BENCHMARK_CAMP)
    required = {
        *ISLAND_BENCHMARK_STRUCTURE_SITES.values(),
        *deer_spawns,
        *stalker_spawns,
        *((int(x), int(y)) for y, x in np.argwhere(quantities > 0)),
    }
    if not required <= reachable:
        raise ValueError("Island benchmark contains an unreachable required location.")

    bands = {Resource.FOOD: (4, 6), Resource.WOOD: (5, 8), Resource.STONE: (6, 10)}
    camp_x, camp_y = ISLAND_BENCHMARK_CAMP
    for resource, (minimum, maximum) in bands.items():
        positions = [
            (int(x), int(y)) for y, x in np.argwhere((resources == resource) & (quantities > 0))
        ]
        if not positions or any(
            not minimum <= abs(x - camp_x) + abs(y - camp_y) <= maximum for x, y in positions
        ):
            raise ValueError(f"{resource.name.lower()} lies outside its distance band.")


def _sample_distance_band(
    candidates: list[tuple[int, int]],
    rng: np.random.Generator,
    minimum: int,
    maximum: int,
    *,
    count: int,
    excluded: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    camp_x, camp_y = ISLAND_BENCHMARK_CAMP
    eligible = [
        position
        for position in candidates
        if position not in excluded
        and minimum <= abs(position[0] - camp_x) + abs(position[1] - camp_y) <= maximum
    ]
    if len(eligible) < count:
        raise ValueError("Not enough reachable positions in the requested distance band.")
    indexes = np.atleast_1d(rng.choice(len(eligible), size=count, replace=False))
    return [eligible[int(index)] for index in indexes]


def _reachable_land(terrain: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            candidate = (x + dx, y + dy)
            if candidate in visited:
                continue
            cx, cy = candidate
            if not (0 <= cx < terrain.shape[1] and 0 <= cy < terrain.shape[0]):
                continue
            if terrain[cy, cx] == Terrain.WATER:
                continue
            visited.add(candidate)
            queue.append(candidate)
    return visited
