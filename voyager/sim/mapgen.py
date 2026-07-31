"""Seeded island generation for the Stage 1 Voyager environment."""

import numpy as np

from voyager.sim.constants import Resource, Terrain
from voyager.sim.state import AgentState, WorldState


def generate_island(map_size: int, rng: np.random.Generator) -> WorldState:
    """Generate a deterministic island world from ``rng``."""

    terrain = np.full((map_size, map_size), Terrain.WATER, dtype=np.uint8)
    resource_ids = np.full((map_size, map_size), Resource.NONE, dtype=np.uint8)
    resource_quantities = np.zeros((map_size, map_size), dtype=np.uint8)

    center = (map_size - 1) / 2.0
    radius_x = map_size * 0.42
    radius_y = map_size * 0.38

    yy, xx = np.mgrid[0:map_size, 0:map_size]
    normalized = ((xx - center) / radius_x) ** 2 + ((yy - center) / radius_y) ** 2
    land_mask = normalized <= 1.0
    beach_mask = land_mask & (normalized > 0.72)
    interior_mask = land_mask & ~beach_mask

    terrain[beach_mask] = Terrain.BEACH
    terrain[interior_mask] = Terrain.GRASS

    forest_noise = rng.random((map_size, map_size))
    quarry_noise = rng.random((map_size, map_size))
    forest_mask = interior_mask & (forest_noise < 0.20)
    quarry_mask = interior_mask & ~forest_mask & (quarry_noise < 0.10)

    terrain[forest_mask] = Terrain.FOREST
    terrain[quarry_mask] = Terrain.QUARRY

    resource_ids[forest_mask] = Resource.WOOD
    resource_quantities[forest_mask] = rng.integers(
        2, 5, size=int(forest_mask.sum()), dtype=np.uint8
    )

    resource_ids[quarry_mask] = Resource.STONE
    resource_quantities[quarry_mask] = rng.integers(
        2, 5, size=int(quarry_mask.sum()), dtype=np.uint8
    )

    food_mask = (terrain == Terrain.GRASS) & (rng.random((map_size, map_size)) < 0.14)
    resource_ids[food_mask] = Resource.FOOD
    resource_quantities[food_mask] = rng.integers(1, 4, size=int(food_mask.sum()), dtype=np.uint8)

    agent_x, agent_y = _find_center_spawn(terrain)
    return WorldState(
        terrain=terrain,
        resource_ids=resource_ids,
        resource_quantities=resource_quantities,
        agent=AgentState(x=agent_x, y=agent_y),
    )


def _find_center_spawn(terrain: np.ndarray) -> tuple[int, int]:
    """Pick the valid land tile nearest to the map center."""

    map_size = terrain.shape[0]
    center = np.array([(map_size - 1) / 2.0, (map_size - 1) / 2.0])
    land_positions = np.argwhere(terrain != Terrain.WATER)
    distances = np.sum((land_positions - center) ** 2, axis=1)
    y, x = land_positions[int(np.argmin(distances))]
    return int(x), int(y)
