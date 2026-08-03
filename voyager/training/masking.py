"""Action-mask helpers shared by PPO training and checkpoint inference."""

from collections.abc import Mapping, Sequence

import numpy as np

from voyager.sim.constants import ACTION_COUNT


def action_mask_from_info(
    info: Mapping[str, object],
    action_count: int = ACTION_COUNT,
) -> np.ndarray:
    """Return a validated boolean action mask, defaulting to all actions."""

    raw_mask = info.get("action_mask")
    if raw_mask is None:
        return np.ones(action_count, dtype=np.bool_)

    mask = np.asarray(raw_mask, dtype=np.bool_)
    if mask.shape != (action_count,):
        raise ValueError(
            f"action_mask must have shape ({action_count},), got {mask.shape}."
        )
    if not np.any(mask):
        raise ValueError("action_mask must allow at least one action.")
    return mask


def stack_action_masks(
    infos: Mapping[str, Mapping[str, object]],
    agent_ids: Sequence[str],
    action_count: int = ACTION_COUNT,
) -> np.ndarray:
    """Stack agent action masks in a stable caller-provided order."""

    return np.stack(
        [action_mask_from_info(infos[agent_id], action_count) for agent_id in agent_ids],
        axis=0,
    )


def mask_numpy_logits(logits: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Replace invalid logits with a large negative value."""

    logits_array = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.bool_)
    if logits_array.ndim != 1:
        raise ValueError(f"logits must be one-dimensional, got {logits_array.shape}.")
    if mask.shape != logits_array.shape:
        raise ValueError(
            f"action_mask shape {mask.shape} does not match logits {logits_array.shape}."
        )
    if not np.any(mask):
        raise ValueError("action_mask must allow at least one action.")
    return np.where(mask, logits_array, np.float32(-1e9))
