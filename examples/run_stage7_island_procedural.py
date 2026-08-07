"""Train, finalize, or verify the frozen VoyagerIsland-v1 procedural baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voyager.training.island_experiment import (
    ALGORITHMS,
    OFFICIAL_TRAINING_SEEDS,
    ProceduralRunSpec,
    finalize_procedural_suite,
    train_procedural_run,
    validate_procedural_run,
    verify_finalized_suite,
)

DEFAULT_ROOT = Path("results/stage7/procedural_baselines_v1")
DEFAULT_SMOKE_ROOT = Path("results/stage7/procedural_baselines_smoke_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one algorithm/seed run.")
    train.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    train.add_argument("--training-seed", type=int, choices=OFFICIAL_TRAINING_SEEDS, required=True)
    train.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Artifact root (official and smoke runs use separate defaults).",
    )
    train.add_argument(
        "--smoke",
        action="store_true",
        help="Run a nonofficial 2,560-transition procedural integration check.",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol without creating files or loading TensorFlow.",
    )

    finalize = subparsers.add_parser(
        "finalize", help="Evaluate all six locked runs on held-out test islands."
    )
    finalize.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)

    verify = subparsers.add_parser(
        "verify", help="Verify an existing final artifact without running episodes."
    )
    verify.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train":
        experiment_root = args.experiment_root or (
            DEFAULT_SMOKE_ROOT if args.smoke else DEFAULT_ROOT
        )
        spec = ProceduralRunSpec(
            experiment_root=experiment_root,
            algorithm=args.algorithm,
            training_seed=args.training_seed,
            official=not args.smoke,
        )
        if args.dry_run:
            protocol = validate_procedural_run(spec, require_clean=spec.official)
            if spec.run_directory.exists():
                raise FileExistsError(f"Procedural run already exists: {spec.run_directory}.")
            print(json.dumps(protocol, indent=2, sort_keys=True))
            print(f"Run directory: {spec.run_directory.resolve()}")
            return 0
        train_procedural_run(spec, require_clean=spec.official)
        return 0
    if args.command == "finalize":
        finalize_procedural_suite(args.experiment_root)
        return 0
    result = verify_finalized_suite(args.experiment_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
