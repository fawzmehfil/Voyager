"""Semantic achievements and public scoring for VoyagerIsland-v1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

ISLAND_ACHIEVEMENT_VERSION = "voyager_island_achievements_v1"
ISLAND_ACHIEVEMENTS = (
    "collect_food",
    "collect_wood",
    "collect_stone",
    "deposit_wood",
    "deposit_stone",
    "build_workbench",
    "craft_axe",
    "craft_spear",
    "hunt_deer",
    "build_campfire",
    "cook_meat",
    "build_shelter",
    "both_survive_first_night",
    "build_beacon",
    "rescue_both",
)
ISLAND_ACHIEVEMENT_GROUPS = {
    "gathering": ISLAND_ACHIEVEMENTS[0:3],
    "delivery": ISLAND_ACHIEVEMENTS[3:5],
    "production": ISLAND_ACHIEVEMENTS[5:12],
    "survival": ("both_survive_first_night",),
    "rescue": ISLAND_ACHIEVEMENTS[13:15],
}


def geometric_mean_score(
    success_rates: Mapping[str, float],
    *,
    smoothing: float = 0.01,
) -> float:
    """Return the smoothed geometric mean over the frozen achievement set."""

    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive.")
    values = [max(0.0, min(1.0, float(success_rates.get(name, 0.0)))) for name in ISLAND_ACHIEVEMENTS]
    return float(math.exp(sum(math.log(value + smoothing) for value in values) / len(values)) - smoothing)


def achievement_success_rates(episodes: Sequence[set[str] | frozenset[str]]) -> dict[str, float]:
    """Aggregate achievement presence without consulting training rewards."""

    if not episodes:
        return {name: 0.0 for name in ISLAND_ACHIEVEMENTS}
    return {
        name: sum(name in achievements for achievements in episodes) / len(episodes)
        for name in ISLAND_ACHIEVEMENTS
    }

def grouped_scores(success_rates: Mapping[str, float]) -> dict[str, float]:
    """Return interpretable arithmetic means for benchmark capability groups."""

    return {
        group: sum(float(success_rates.get(name, 0.0)) for name in names) / len(names)
        for group, names in ISLAND_ACHIEVEMENT_GROUPS.items()
    }
