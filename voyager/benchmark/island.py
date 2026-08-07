"""Frozen seed splits and statistical helpers for VoyagerIsland-v1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ISLAND_GENERATOR_VERSION = "voyager_island_generator_v1"
ISLAND_TRAIN_SEEDS = tuple(range(1_000))
ISLAND_DEV_SEEDS = tuple(range(10_000, 10_050))
ISLAND_TEST_SEEDS = tuple(range(20_000, 20_100))


def validate_seed_splits() -> None:
    """Assert the immutable training, selection, and final-evaluation split boundary."""

    sets = [set(ISLAND_TRAIN_SEEDS), set(ISLAND_DEV_SEEDS), set(ISLAND_TEST_SEEDS)]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise RuntimeError("VoyagerIsland-v1 seed manifests overlap.")


def paired_bootstrap_difference(
    learned: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 0,
    samples: int = 10_000,
) -> tuple[float, float, float]:
    """Return mean and paired 95% bootstrap interval over matched island seeds."""

    left = np.asarray(learned, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Bootstrap inputs must be non-empty matched one-dimensional samples.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    differences = left - right
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, differences.size, size=(samples, differences.size))
    bootstrapped = np.mean(differences[indexes], axis=1)
    return (
        float(np.mean(differences)),
        float(np.quantile(bootstrapped, 0.025)),
        float(np.quantile(bootstrapped, 0.975)),
    )


validate_seed_splits()
