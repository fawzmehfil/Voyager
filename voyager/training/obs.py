"""Observation flattening for Voyager training loops."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from voyager.sim.constants import Resource, Terrain

Observation = Mapping[str, np.ndarray]

OBSERVATION_KEYS = ("local_view", "stats", "inventory", "role", "camp", "progress")


def flatten_observation(observation: Observation) -> np.ndarray:
    """Convert a Voyager dict observation into one normalized float32 vector."""

    local_view = _normalize_local_view(np.asarray(observation["local_view"]))
    parts = [local_view.reshape(-1)]
    for key in OBSERVATION_KEYS[1:]:
        parts.append(np.clip(np.asarray(observation[key], dtype=np.float32), 0.0, 1.0).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def flatten_observations(
    observations: Mapping[str, Observation],
    agent_ids: Sequence[str],
) -> np.ndarray:
    """Stack flattened observations in the caller-provided agent order."""

    if not agent_ids:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(
        [flatten_observation(observations[agent_id]) for agent_id in agent_ids],
        axis=0,
    )


def flat_observation_size(observation_space: Any) -> int:
    """Return the flattened vector length for Voyager's observation space."""

    spaces = observation_space.spaces
    size = 0
    for key in OBSERVATION_KEYS:
        shape = spaces[key].shape
        size += int(np.prod(shape))
    return size


def _normalize_local_view(local_view: np.ndarray) -> np.ndarray:
    if local_view.ndim != 3 or local_view.shape[-1] < 4:
        raise ValueError("local_view must have shape (height, width, >=4).")

    normalized = local_view.astype(np.float32, copy=True)
    normalized[..., 0] /= max(1, int(Terrain.QUARRY))
    normalized[..., 1] /= max(1, int(Resource.STONE))
    normalized[..., 2] /= 255.0
    normalized[..., 3] /= 255.0
    return np.clip(normalized, 0.0, 1.0)
