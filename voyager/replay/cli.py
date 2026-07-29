"""Command-line workflows for recording and inspecting Voyager replays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .catalog import ReplayCatalog, catalog_entry_payload
from .loader import ReplayLoader
from .recorder import migrate_legacy_replay, record_checkpoint_episode, record_episode

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/manifests/stage5_6_final.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voyager-replay", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record one policy and seed.")
    record.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    record.add_argument("--policy", required=True)
    record.add_argument("--seed", required=True, type=int)
    record.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "runs/replays")
    record.add_argument("--replay-id")
    record.add_argument("--checkpoint", type=Path)
    record.add_argument("--stochastic", action="store_true")
    record.add_argument("--tag", action="append", default=[])
    record.add_argument("--overwrite", action="store_true")

    list_parser = subparsers.add_parser("list", help="List cataloged recordings.")
    list_parser.add_argument("--root", type=Path, action="append")
    list_parser.add_argument("--policy")
    list_parser.add_argument("--seed", type=int)

    inspect = subparsers.add_parser("inspect", help="Print replay source and summary.")
    inspect.add_argument("path", type=Path)

    validate = subparsers.add_parser("validate", help="Validate replay artifacts.")
    validate.add_argument("path", type=Path)
    validate.add_argument("--deep", action="store_true")

    migrate = subparsers.add_parser("migrate", help="Inspect a Stage 6A legacy fixture.")
    migrate.add_argument("path", type=Path)
    migrate.add_argument(
        "--output",
        type=Path,
        help="Reserved output directory. Legacy loading needs no destructive migration.",
    )
    migrate.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        target = (
            record_checkpoint_episode(
                args.manifest,
                checkpoint=args.checkpoint,
                seed=args.seed,
                output_root=args.output_root,
                policy_id=args.policy,
                replay_id=args.replay_id,
                deterministic=not args.stochastic,
                tags=tuple(args.tag),
                overwrite=args.overwrite,
            )
            if args.checkpoint is not None
            else record_episode(
                args.manifest,
                policy_id=args.policy,
                seed=args.seed,
                output_root=args.output_root,
                replay_id=args.replay_id,
                tags=tuple(args.tag),
                overwrite=args.overwrite,
            )
        )
        print(target)
        return 0
    if args.command == "list":
        roots = args.root or _configured_roots()
        catalog = ReplayCatalog([(f"root-{index}", root) for index, root in enumerate(roots)])
        print(
            json.dumps(
                [
                    catalog_entry_payload(entry)
                    for entry in catalog.query(policy=args.policy, seed=args.seed)
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    loader = ReplayLoader(args.path)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "replay_id": loader.manifest.replay_id,
                    "versions": loader.manifest.versions.model_dump(mode="json"),
                    "source": loader.manifest.source.model_dump(mode="json"),
                    "terminal_summary": loader.manifest.terminal_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        print(json.dumps(loader.validate(deep=args.deep), indent=2, sort_keys=True))
        return 0
    if args.command == "migrate":
        if args.output is None:
            result: dict[str, Any] = loader.validate(deep=True)
            result["message"] = "Pass --output to create a best-effort formal v2 directory."
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        migrated = migrate_legacy_replay(
            args.path,
            args.output,
            overwrite=args.overwrite,
        )
        print(migrated)
        return 0
    return 2


def _configured_roots() -> list[Path]:
    configured = os.environ.get("VOYAGER_REPLAY_ROOTS")
    if configured:
        return [Path(value) for value in configured.split(os.pathsep) if value]
    return [
        REPOSITORY_ROOT / "benchmarks/replays/stage6_curated_v1",
        REPOSITORY_ROOT / "runs/replays",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
