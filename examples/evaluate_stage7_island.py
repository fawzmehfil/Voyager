"""Evaluate VoyagerIsland-v1 legal-random or scripted-oracle baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voyager.benchmark.island import ISLAND_DEV_SEEDS, ISLAND_TEST_SEEDS
from voyager.policies.island_scripted import ScriptedIslandOracle
from voyager.training.island_evaluation import (
    LegalRandomIslandPolicy,
    evaluate_island_policy,
    scripted_oracle_solvability_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("random", "oracle"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = ISLAND_DEV_SEEDS if args.split == "dev" else ISLAND_TEST_SEEDS
    seeds = split if args.episodes is None else split[: args.episodes]
    factory = (
        (lambda _seed: ScriptedIslandOracle())
        if args.policy == "oracle"
        else (lambda seed: LegalRandomIslandPolicy(seed))
    )
    episodes, summary = evaluate_island_policy(factory, seeds=seeds, procedural=True)
    payload = {
        "contract": "voyager_island_evaluation_v1",
        "policy": args.policy,
        "split": args.split,
        "summary": summary,
        "episodes": [episode.as_dict() for episode in episodes],
    }
    if args.policy == "oracle":
        payload["gate"] = scripted_oracle_solvability_gate(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if "gate" in payload:
        print(f"Oracle solvability gate: {'PASS' if payload['gate']['passed'] else 'FAIL'}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
