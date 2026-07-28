"""Run or resume a versioned Voyager Stage 5.6 benchmark manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voyager.benchmark.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching partial benchmark output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_benchmark(
            args.manifest,
            args.output,
            resume=args.resume,
            on_progress=_print_progress,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    family = summary.get("ppo_official_family")
    if isinstance(family, dict):
        survivors = family["metrics"]["survivors"]["mean"]
        score = family["civilization_score"]["mean"]
        print(f"official PPO family: survivors={survivors:.2f} score={score:.2f}")
    print(f"benchmark complete: {args.output}")
    return 0


def _print_progress(completed: int, total: int, policy_id: str, seed: int) -> None:
    print(f"[{completed:04d}/{total:04d}] policy={policy_id} seed={seed}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
