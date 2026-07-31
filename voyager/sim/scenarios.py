"""Scenario definitions and the handcrafted Stage 7A island."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Configuration that changes content without changing the shared engine."""

    id: str
    map_size: int
    max_steps: int
    civilization: bool = False
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
