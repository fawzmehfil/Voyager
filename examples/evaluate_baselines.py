"""Evaluate Voyager's Stage 4 baseline policies."""

from __future__ import annotations

import argparse

from voyager.policies.evaluation import evaluate_baselines, print_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-agents", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = evaluate_baselines(
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
    )
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
