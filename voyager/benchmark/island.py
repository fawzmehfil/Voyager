"""Frozen seed splits and statistical helpers for VoyagerIsland-v1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ISLAND_GENERATOR_VERSION = "voyager_island_generator_v1"
ISLAND_TRAIN_SEEDS = tuple(range(1_000))
ISLAND_DEV_SEEDS = tuple(range(10_000, 10_050))
ISLAND_TEST_SEEDS = tuple(range(20_000, 20_100))
ISLAND_MANIFEST_FILENAMES = {
    "train": "island_v1_train.json",
    "development": "island_v1_dev.json",
    "test": "island_v1_test.json",
}


@dataclass(frozen=True, slots=True)
class IslandSeedManifest:
    """Validated seed manifest used by one benchmark split."""

    split: str
    seeds: tuple[int, ...]
    generator_version: str
    path: Path
    sha256: str


def load_island_seed_manifests(
    manifest_root: str | Path | None = None,
) -> dict[str, IslandSeedManifest]:
    """Load the tracked split manifests and prove that they match the frozen contract."""

    root = (
        Path(manifest_root)
        if manifest_root is not None
        else Path(__file__).resolve().parents[2] / "benchmarks" / "manifests"
    )
    expected = {
        "train": ISLAND_TRAIN_SEEDS,
        "development": ISLAND_DEV_SEEDS,
        "test": ISLAND_TEST_SEEDS,
    }
    manifests: dict[str, IslandSeedManifest] = {}
    for split, filename in ISLAND_MANIFEST_FILENAMES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing VoyagerIsland-v1 seed manifest: {path}.")
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("benchmark") != "VoyagerIsland-v1":
            raise ValueError(f"Unexpected benchmark in {path}.")
        if payload.get("generator_version") != ISLAND_GENERATOR_VERSION:
            raise ValueError(f"Unexpected generator version in {path}.")
        if payload.get("split") != split:
            raise ValueError(f"Unexpected split label in {path}.")
        seed_range = payload.get("seed_range")
        if not isinstance(seed_range, dict):
            raise TypeError(f"Missing seed_range in {path}.")
        start = _manifest_integer(seed_range, "start", path)
        stop = _manifest_integer(seed_range, "stop_exclusive", path)
        count = _manifest_integer(seed_range, "count", path)
        seeds = tuple(range(start, stop))
        if count != len(seeds) or seeds != expected[split]:
            raise ValueError(f"Seed range in {path} does not match the frozen {split} split.")
        manifests[split] = IslandSeedManifest(
            split=split,
            seeds=seeds,
            generator_version=ISLAND_GENERATOR_VERSION,
            path=path.resolve(),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    sets = [set(manifest.seeds) for manifest in manifests.values()]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise ValueError("VoyagerIsland-v1 seed manifests overlap.")
    return manifests


def validate_seed_splits() -> None:
    """Assert the immutable training, selection, and final-evaluation split boundary."""

    sets = [set(ISLAND_TRAIN_SEEDS), set(ISLAND_DEV_SEEDS), set(ISLAND_TEST_SEEDS)]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise RuntimeError("VoyagerIsland-v1 seed manifests overlap.")


def _manifest_integer(payload: dict[object, object], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} in {path} must be an integer.")
    return value


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
