"""Observation flattening for Voyager training loops."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from voyager.sim.constants import Resource, Terrain

Observation = Mapping[str, np.ndarray]

COMPACT_OBSERVATION_ENCODER = "compact_structured_210_v1"
CIVILIZATION_V2_OBSERVATION_ENCODER = "civilization_v2_flat_v1"
CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER = "civilization_v2_identity_flat_v2"
CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER = (
    "civilization_v3_navigation_flat_v3"
)

OBSERVATION_KEYS = ("local_view", "stats", "inventory", "role", "camp", "progress")
CIVILIZATION_V2_OBSERVATION_KEYS = (
    "local_tiles",
    "self_state",
    "inventory",
    "tools_owned",
    "tool_equipped",
    "torch_charge",
    "role",
    "time",
    "camp",
    "entity_slots",
)
CIVILIZATION_V2_IDENTITY_OBSERVATION_KEYS = (
    *CIVILIZATION_V2_OBSERVATION_KEYS,
    "agent_identity",
)
CIVILIZATION_V3_NAVIGATION_OBSERVATION_KEYS = (
    *CIVILIZATION_V2_IDENTITY_OBSERVATION_KEYS,
    "camp_bearing",
)


def flatten_observation(
    observation: Observation,
    encoder_id: str = COMPACT_OBSERVATION_ENCODER,
) -> np.ndarray:
    """Convert a Voyager dict observation into one normalized float32 vector."""

    if encoder_id in {
        CIVILIZATION_V2_OBSERVATION_ENCODER,
        CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER,
        CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER,
    }:
        keys: tuple[str, ...] = CIVILIZATION_V2_OBSERVATION_KEYS
        if encoder_id == CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER:
            keys = CIVILIZATION_V2_IDENTITY_OBSERVATION_KEYS
        elif encoder_id == CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER:
            keys = CIVILIZATION_V3_NAVIGATION_OBSERVATION_KEYS
        return _flatten_civilization_v2_observation(observation, keys)
    if encoder_id != COMPACT_OBSERVATION_ENCODER:
        raise ValueError(f"Unknown observation encoder: {encoder_id!r}.")
    local_view = _normalize_local_view(np.asarray(observation["local_view"]))
    parts = [local_view.reshape(-1)]
    for key in OBSERVATION_KEYS[1:]:
        parts.append(
            np.clip(
                np.asarray(observation[key], dtype=np.float32),
                0.0,
                1.0,
            ).reshape(-1)
        )
    return np.concatenate(parts).astype(np.float32)


def flatten_observations(
    observations: Mapping[str, Observation],
    agent_ids: Sequence[str],
    encoder_id: str = COMPACT_OBSERVATION_ENCODER,
) -> np.ndarray:
    """Stack flattened observations in the caller-provided agent order."""

    if not agent_ids:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(
        [flatten_observation(observations[agent_id], encoder_id) for agent_id in agent_ids],
        axis=0,
    )


def flat_observation_size(
    observation_space: Any,
    encoder_id: str = COMPACT_OBSERVATION_ENCODER,
) -> int:
    """Return the flattened vector length for Voyager's observation space."""

    spaces = observation_space.spaces
    keys: tuple[str, ...] = OBSERVATION_KEYS
    if encoder_id == CIVILIZATION_V2_OBSERVATION_ENCODER:
        keys = CIVILIZATION_V2_OBSERVATION_KEYS
    elif encoder_id == CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER:
        keys = CIVILIZATION_V2_IDENTITY_OBSERVATION_KEYS
    elif encoder_id == CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER:
        keys = CIVILIZATION_V3_NAVIGATION_OBSERVATION_KEYS
    if encoder_id not in {
        COMPACT_OBSERVATION_ENCODER,
        CIVILIZATION_V2_OBSERVATION_ENCODER,
        CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER,
        CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER,
    }:
        raise ValueError(f"Unknown observation encoder: {encoder_id!r}.")
    size = 0
    for key in keys:
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


def _flatten_civilization_v2_observation(
    observation: Observation,
    keys: tuple[str, ...],
) -> np.ndarray:
    missing = set(keys) - set(observation)
    if missing:
        raise ValueError(
            "Civilization v2 observation is missing: " + ", ".join(sorted(missing))
        )
    local_tiles = np.asarray(observation["local_tiles"])
    if local_tiles.shape != (7, 7, 7):
        raise ValueError("local_tiles must have shape (7, 7, 7).")
    normalized_tiles = local_tiles.astype(np.float32, copy=True)
    normalized_tiles[..., 0] /= max(1, max(int(value) for value in Terrain))
    normalized_tiles[..., 1] /= max(1, max(int(value) for value in Resource))
    normalized_tiles[..., 2] = np.clip(normalized_tiles[..., 2] / 15.0, 0.0, 1.0)
    normalized_tiles[..., 3] /= 3.0
    normalized_tiles[..., 4] /= 2.0
    normalized_tiles[..., 5] /= 255.0
    normalized_tiles[..., 6] /= 255.0
    parts = [np.clip(normalized_tiles, 0.0, 1.0).reshape(-1)]
    for key in keys[1:]:
        values = np.asarray(observation[key], dtype=np.float32)
        if key in {"entity_slots", "camp_bearing"}:
            values = np.clip(values, -1.0, 1.0)
        elif key == "time":
            values = np.clip(values, 0.0, 2.0)
        else:
            values = np.clip(values, 0.0, 1.0)
        parts.append(values.reshape(-1))
    encoded = np.concatenate(parts).astype(np.float32)
    if not np.all(np.isfinite(encoded)):
        raise ValueError("Civilization v2 observation contains non-finite values.")
    return encoded
