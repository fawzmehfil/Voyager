"""Filesystem-backed, read-only replay catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import ReplayError, ReplayLoader
from .schema import ReplayManifest


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    root: Path
    directory: Path
    manifest: ReplayManifest
    source: str


class ReplayCatalog:
    """Scan curated and local roots without requiring a database."""

    def __init__(self, roots: list[tuple[str, str | Path]], *, cache_size: int = 8) -> None:
        self.roots = [(label, Path(path).resolve()) for label, path in roots]
        self.cache_size = cache_size
        self._manifest_cache: dict[Path, tuple[int, ReplayManifest]] = {}

    def scan(self, *, include_failed: bool = False) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        seen_ids: set[str] = set()
        for source, root in self.roots:
            if not root.exists():
                continue
            candidates = sorted(root.glob("*/manifest.json"))
            if include_failed:
                candidates.extend(sorted((root / ".failed").glob("*/manifest.json")))
            for manifest_path in candidates:
                if any(part.startswith(".") for part in manifest_path.relative_to(root).parts):
                    continue
                try:
                    manifest = self._load_cached(manifest_path)
                except (OSError, ValueError, ReplayError):
                    continue
                if manifest.status != "complete" and not include_failed:
                    continue
                if manifest.replay_id in seen_ids:
                    continue
                seen_ids.add(manifest.replay_id)
                entries.append(
                    CatalogEntry(
                        root=root,
                        directory=manifest_path.parent,
                        manifest=manifest,
                        source=source,
                    )
                )
        return sorted(
            entries,
            key=lambda entry: (
                0 if "showcase" in entry.manifest.tags else 1,
                entry.manifest.replay_id,
            ),
        )

    def get(self, replay_id: str) -> CatalogEntry:
        _validate_replay_id(replay_id)
        for entry in self.scan():
            if entry.manifest.replay_id == replay_id:
                return entry
        raise KeyError(replay_id)

    def loader(self, replay_id: str) -> ReplayLoader:
        return ReplayLoader(self.get(replay_id).directory, cache_size=self.cache_size)

    def query(
        self,
        *,
        policy: str | None = None,
        kind: str | None = None,
        scenario: str | None = None,
        seed: int | None = None,
        inference_mode: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        min_survivors: int | None = None,
        max_survivors: int | None = None,
        achievement: str | None = None,
    ) -> list[CatalogEntry]:
        values = self.scan(include_failed=status == "failed")
        filters = {
            "policy": lambda entry: entry.manifest.source.policy_id == policy,
            "kind": lambda entry: entry.manifest.source.policy_kind == kind,
            "scenario": lambda entry: entry.manifest.versions.scenario == scenario,
            "seed": lambda entry: entry.manifest.source.evaluation_seed == seed,
            "inference_mode": lambda entry: entry.manifest.source.inference_mode == inference_mode,
            "status": lambda entry: entry.manifest.status == status,
            "tag": lambda entry: tag in entry.manifest.tags,
            "min_survivors": lambda entry: int(entry.manifest.terminal_summary["survivors"])
            >= int(min_survivors or 0),
            "max_survivors": lambda entry: int(entry.manifest.terminal_summary["survivors"])
            <= int(max_survivors or 0),
            "achievement": lambda entry: achievement
            in entry.manifest.terminal_summary.get("achievements", []),
        }
        requested: dict[str, Any] = {
            "policy": policy,
            "kind": kind,
            "scenario": scenario,
            "seed": seed,
            "inference_mode": inference_mode,
            "status": status,
            "tag": tag,
            "min_survivors": min_survivors,
            "max_survivors": max_survivors,
            "achievement": achievement,
        }
        for name, value in requested.items():
            if value is not None:
                values = [entry for entry in values if filters[name](entry)]
        return values

    def _load_cached(self, path: Path) -> ReplayManifest:
        modified = path.stat().st_mtime_ns
        cached = self._manifest_cache.get(path)
        if cached is not None and cached[0] == modified:
            return cached[1]
        manifest = ReplayLoader.load_manifest(path.parent)
        self._manifest_cache[path] = (modified, manifest)
        return manifest


def catalog_entry_payload(entry: CatalogEntry) -> dict[str, Any]:
    manifest = entry.manifest
    return {
        "replay_id": manifest.replay_id,
        "source": entry.source,
        "policy_id": manifest.source.policy_id,
        "policy_kind": manifest.source.policy_kind,
        "inference_mode": manifest.source.inference_mode,
        "seed": manifest.source.evaluation_seed,
        "scenario": manifest.versions.scenario,
        "status": manifest.status,
        "tags": manifest.tags,
        "world_steps": manifest.world_steps,
        "tick_rate": manifest.tick_rate,
        "terminal_summary": manifest.terminal_summary,
    }


def _validate_replay_id(replay_id: str) -> None:
    if not replay_id or replay_id in {".", ".."}:
        raise ValueError("Invalid replay id.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in replay_id):
        raise ValueError("Invalid replay id.")
