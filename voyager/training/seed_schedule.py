"""Deterministic manifest-backed episode seed scheduling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

EPISODE_SEED_SCHEDULE_VERSION = "voyager_island_episode_seed_schedule_v1"


@dataclass(slots=True)
class EpisodeSeedScheduler:
    """Visit every allowed environment seed once per deterministic shuffled cycle."""

    seeds: tuple[int, ...]
    shuffle_seed: int
    manifest_hash: str | None = None
    split: str | None = None
    history: list[int] = field(default_factory=list, init=False)
    _cycle: int = field(default=0, init=False)
    _index: int = field(default=0, init=False)
    _order: tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("Episode seed schedule requires at least one seed.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Episode seed schedule cannot contain duplicate seeds.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Episode seeds must be non-negative.")
        if self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be non-negative.")
        self._order = self._permutation(0)

    @property
    def cycle(self) -> int:
        return self._cycle

    def next_seed(self) -> int:
        """Return the next allowed seed and advance the schedule."""

        if self._index >= len(self._order):
            self._cycle += 1
            self._index = 0
            self._order = self._permutation(self._cycle)
        seed = self._order[self._index]
        self._index += 1
        self.history.append(seed)
        return seed

    def metadata(self) -> dict[str, object]:
        return {
            "version": EPISODE_SEED_SCHEDULE_VERSION,
            "split": self.split,
            "manifest_sha256": self.manifest_hash,
            "shuffle_seed": self.shuffle_seed,
            "seed_pool_size": len(self.seeds),
            "episodes_started": len(self.history),
            "current_cycle": self._cycle,
        }

    def _permutation(self, cycle: int) -> tuple[int, ...]:
        prefix = f"{EPISODE_SEED_SCHEDULE_VERSION}:{self.shuffle_seed}:{cycle}:"
        return tuple(
            sorted(
                self.seeds,
                key=lambda seed: hashlib.sha256(f"{prefix}{seed}".encode()).digest(),
            )
        )
