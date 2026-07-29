"""Generate Stage 6's five seed-matched curated replays through the public API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from voyager.replay.recorder import record_episode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED = 10_000_010
POLICIES = (
    "random",
    "greedy",
    "cooperative",
    "ppo_seed0_deterministic",
    "ppo_seed0_stochastic",
)
EXPECTED = {
    "random": {"survivors": 8, "deaths": 2, "achievement_count": 6},
    "greedy": {"survivors": 8, "deaths": 2, "achievement_count": 5},
    "cooperative": {"survivors": 2, "deaths": 8, "achievement_count": 6},
    "ppo_seed0_deterministic": {"survivors": 10, "deaths": 0, "achievement_count": 16},
    "ppo_seed0_stochastic": {"survivors": 10, "deaths": 0, "achievement_count": 16},
}
SHOWCASE_CAMERA = [
    {
        "start_tick": 0,
        "end_tick": 23,
        "kind": "overview",
        "target": {"x": 16, "y": 16},
        "zoom": 0.68,
        "label": "The island",
    },
    {
        "start_tick": 24,
        "end_tick": 59,
        "kind": "event",
        "target": {"x": 16, "y": 16},
        "zoom": 1.0,
        "label": "First supplies",
    },
    {
        "start_tick": 60,
        "end_tick": 139,
        "kind": "event",
        "target": {"x": 16, "y": 16},
        "zoom": 0.92,
        "label": "Building shelter",
    },
    {
        "start_tick": 140,
        "end_tick": 199,
        "kind": "team",
        "target": {"x": 16, "y": 16},
        "zoom": 0.78,
        "label": "Working together",
    },
    {
        "start_tick": 200,
        "end_tick": 227,
        "kind": "weather",
        "target": {"x": 16, "y": 16},
        "zoom": 0.66,
        "label": "The storm",
    },
    {
        "start_tick": 228,
        "end_tick": 275,
        "kind": "event",
        "target": {"x": 16, "y": 16},
        "zoom": 0.88,
        "label": "Food security",
    },
    {
        "start_tick": 276,
        "end_tick": 300,
        "kind": "finale",
        "target": {"x": 16, "y": 16},
        "zoom": 0.7,
        "label": "10/10 survived",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/manifests/stage5_6_final.json",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=REPOSITORY_ROOT / "results/benchmark/stage5_6_final_v1/episodes.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/replays/stage6_curated_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    episodes = _load_episodes(args.episodes)
    for policy_id in POLICIES:
        episode = episodes[(policy_id, SEED)]
        expected = EXPECTED[policy_id]
        if (
            episode["survivors"] != expected["survivors"]
            or episode["deaths"] != expected["deaths"]
            or len(episode["achievements"]) != expected["achievement_count"]
        ):
            raise ValueError(f"Benchmark outcome changed for {policy_id!r}.")
        tags = ["curated", "same-island"]
        if policy_id == "ppo_seed0_deterministic":
            tags.extend(("showcase", "default"))
        if policy_id == "cooperative":
            tags.append("failure-case")
        if policy_id == "ppo_seed0_stochastic":
            tags.append("diagnostic")
        target = record_episode(
            args.manifest,
            policy_id=policy_id,
            seed=SEED,
            output_root=args.output_root,
            replay_id=policy_id,
            tags=tuple(tags),
            overwrite=args.overwrite,
            benchmark_episode=episode,
            camera_overrides=(
                SHOWCASE_CAMERA if policy_id == "ppo_seed0_deterministic" else None
            ),
        )
        print(target)


def _load_episodes(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required so curated replays can be verified against Stage 5.6."
        )
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        (str(value["policy_id"]), int(value["seed"])): value
        for value in values
        if int(value["seed"]) == SEED
    }


if __name__ == "__main__":
    main()
