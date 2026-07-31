"""Simulation-free lazy loading and exact replay reconstruction."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from .recorder import apply_state_delta
from .schema import LEGACY_REPLAY_SCHEMA_VERSION, ReplayManifest, validate_manifest
from .serialization import read_json, read_json_gz, sha256_file, sha256_value

T = TypeVar("T")


class ReplayError(RuntimeError):
    """Base class for structured replay failures."""


class UnsupportedReplayError(ReplayError):
    pass


class CorruptReplayError(ReplayError):
    pass


class IncompleteReplayError(ReplayError):
    pass


class LRUCache(Generic[T]):
    def __init__(self, capacity: int = 8) -> None:
        self.capacity = max(1, capacity)
        self.values: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, key: str, value: T) -> None:
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)


class ReplayLoader:
    """Load a v2 replay directory or the permanent Stage 6A v1 fixture."""

    def __init__(self, path: str | Path, *, cache_size: int = 8) -> None:
        self.path = Path(path).resolve()
        self._chunk_cache: LRUCache[dict[str, Any]] = LRUCache(cache_size)
        self._snapshot_cache: LRUCache[dict[str, Any]] = LRUCache(cache_size)
        self._legacy: LegacyReplayAdapter | None = None
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != LEGACY_REPLAY_SCHEMA_VERSION:
                raise UnsupportedReplayError(
                    f"Unsupported single-file replay {payload.get('schema_version')!r}."
                )
            self._legacy = LegacyReplayAdapter(payload)
            self.manifest = self._legacy.manifest
            return
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise IncompleteReplayError(f"Replay is missing {manifest_path.name}.")
        try:
            self.manifest = validate_manifest(read_json(manifest_path))
        except ValueError as exc:
            raise UnsupportedReplayError(str(exc)) from exc
        if self.manifest.status != "complete":
            raise IncompleteReplayError(
                f"Replay status is {self.manifest.status!r}, not complete."
            )

    @classmethod
    def load_manifest(cls, path: str | Path) -> ReplayManifest:
        """Load only manifest.json; timeline and snapshots remain untouched."""

        return cls(path, cache_size=1).manifest

    def initial(self) -> dict[str, Any]:
        if self._legacy is not None:
            return self._legacy.initial()
        return cast(dict[str, Any], read_json_gz(self.path / "initial.json.gz"))

    def camera(self) -> dict[str, Any]:
        if self._legacy is not None:
            return self._legacy.camera()
        return cast(dict[str, Any], read_json(self.path / "camera.json"))

    def metrics(self) -> dict[str, Any]:
        if self._legacy is not None:
            return self._legacy.metrics()
        return cast(dict[str, Any], read_json_gz(self.path / "metrics.json.gz"))

    def state_at(self, tick: int) -> dict[str, Any]:
        self._validate_tick(tick)
        if self._legacy is not None:
            return self._legacy.state_at(tick)
        snapshot_tick = max(
            entry.start_tick or 0
            for entry in self.manifest.artifacts
            if entry.kind == "snapshots" and (entry.start_tick or 0) <= tick
        )
        state = copy.deepcopy(self._load_snapshot(snapshot_tick))
        if snapshot_tick == tick:
            return state
        for record in self.timeline(snapshot_tick + 1, tick):
            state = apply_state_delta(state, record["state_delta"])
            state["tick"] = record["tick"]
            state["weather"] = record["weather"]
        return state

    def timeline(self, start: int, end: int) -> list[dict[str, Any]]:
        self._validate_range(start, end)
        if self._legacy is not None:
            return self._legacy.timeline(start, end)
        records: list[dict[str, Any]] = []
        for entry in self.manifest.artifacts:
            if entry.kind != "timeline":
                continue
            chunk_start = int(entry.start_tick or 0)
            chunk_end = int(entry.end_tick or 0)
            if chunk_end < start or chunk_start > end:
                continue
            payload = self._load_chunk(entry.path)
            records.extend(
                record for record in payload["records"] if start <= int(record["tick"]) <= end
            )
        return sorted(records, key=lambda record: int(record["tick"]))

    def events(
        self,
        start: int = 0,
        end: int | None = None,
        *,
        event_type: str | None = None,
        agent: str | None = None,
    ) -> list[dict[str, Any]]:
        end = self.manifest.world_steps if end is None else end
        values = [
            event
            for record in self.timeline(max(1, start), end)
            for event in record.get("events", [])
        ]
        if event_type is not None:
            values = [event for event in values if event.get("type") == event_type]
        if agent is not None:
            values = [
                event
                for event in values
                if agent in event.get("actors", []) or agent in event.get("targets", [])
            ]
        return values

    def actions(
        self, start: int = 1, end: int | None = None, *, agent: str | None = None
    ) -> list[dict[str, Any]]:
        end = self.manifest.world_steps if end is None else end
        values = [
            {"tick": record["tick"], **action}
            for record in self.timeline(start, end)
            for action in record.get("actions", [])
        ]
        return (
            [value for value in values if value.get("agent_id") == agent]
            if agent is not None
            else values
        )

    def validate(self, *, deep: bool = False) -> dict[str, Any]:
        if self._legacy is not None:
            return self._legacy.validate(deep=deep)
        failures: list[str] = []
        for entry in self.manifest.artifacts:
            artifact = self.path / entry.path
            if not artifact.is_file():
                failures.append(f"missing:{entry.path}")
                continue
            if artifact.stat().st_size != entry.bytes:
                failures.append(f"size:{entry.path}")
            if sha256_file(artifact) != entry.sha256:
                failures.append(f"checksum:{entry.path}")
        if self.manifest.manifest_sha256:
            payload = self.manifest.model_dump(mode="json")
            expected = payload.pop("manifest_sha256")
            if sha256_value(payload) != expected:
                failures.append("manifest:fingerprint")
        if failures:
            raise CorruptReplayError("; ".join(failures))

        final_state = self.state_at(self.manifest.world_steps)
        summary = self.manifest.terminal_summary
        survivors = sum(bool(agent["alive"]) for agent in final_state["agents"])
        if survivors != int(summary["survivors"]):
            raise CorruptReplayError("Terminal survivor count does not reconstruct.")
        if final_state["camp"]["stockpile"] != summary["camp_stockpile"]:
            raise CorruptReplayError("Terminal camp stockpile does not reconstruct.")
        if final_state["achievements"] != sorted(summary["achievements"]):
            raise CorruptReplayError("Terminal achievements do not reconstruct.")
        checked_ticks = [self.manifest.world_steps]
        if deep:
            checked_ticks = list(range(self.manifest.world_steps + 1))
            for tick in checked_ticks:
                state = self.state_at(tick)
                if int(state["tick"]) != tick:
                    raise CorruptReplayError(f"Tick {tick} reconstructed as {state['tick']}.")
                expected_hash = state.get("extensions", {}).get("state_hash")
                if expected_hash:
                    hashable = {
                        key: value
                        for key, value in state.items()
                        if key not in {"tick", "weather", "terrain", "width", "height"}
                    }
                    hashable.pop("extensions", None)
                    hashable["resources"] = sorted(
                        hashable["resources"], key=lambda item: item["id"]
                    )
                    if sha256_value(hashable) != expected_hash:
                        raise CorruptReplayError(f"Tick {tick} state hash does not reconstruct.")
        return {
            "replay_id": self.manifest.replay_id,
            "status": "valid",
            "checked_artifacts": len(self.manifest.artifacts),
            "checked_ticks": len(checked_ticks),
        }

    def prefetch_adjacent(self, tick: int) -> None:
        if self._legacy is not None:
            return
        chunks = [
            entry
            for entry in self.manifest.artifacts
            if entry.kind == "timeline"
            and int(entry.start_tick or 0) <= tick <= int(entry.end_tick or 0)
        ]
        if not chunks:
            return
        all_chunks = [entry for entry in self.manifest.artifacts if entry.kind == "timeline"]
        index = all_chunks.index(chunks[0])
        for entry in all_chunks[max(0, index - 1) : index + 2]:
            self._load_chunk(entry.path)

    def _load_chunk(self, relative_path: str) -> dict[str, Any]:
        cached = self._chunk_cache.get(relative_path)
        if cached is not None:
            return cached
        payload = cast(dict[str, Any], read_json_gz(self.path / relative_path))
        self._chunk_cache.put(relative_path, payload)
        return payload

    def _load_snapshot(self, tick: int) -> dict[str, Any]:
        key = f"{tick:06d}"
        cached = self._snapshot_cache.get(key)
        if cached is not None:
            return cached
        payload = cast(dict[str, Any], read_json_gz(self.path / "snapshots" / f"{key}.json.gz"))
        self._snapshot_cache.put(key, payload)
        return payload

    def _validate_tick(self, tick: int) -> None:
        if tick < 0 or tick > self.manifest.world_steps:
            raise ValueError(f"Tick must be between 0 and {self.manifest.world_steps}.")

    def _validate_range(self, start: int, end: int) -> None:
        self._validate_tick(start)
        self._validate_tick(end)
        if start > end:
            raise ValueError("Range start cannot exceed range end.")


class LegacyReplayAdapter:
    """Expose Stage 6A's frozen single-file artifact through the v2 loader API."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        source = payload["source"]
        summary = payload["summary"]
        self.manifest = ReplayManifest.model_validate(
            {
                "replay_id": payload["replay_id"],
                "versions": {
                    "replay": "stage6_replay_2.0.0",
                    "environment": "stage5.5",
                    "scenario": source["scenario_id"],
                    "reward": "stage5.5_economy_group_v1",
                    "observation": "structured_210_v1",
                    "action": "discrete_13_v1",
                    "achievement": "stage5.6_16_v1",
                },
                "status": "complete",
                "source": {
                    "policy_kind": source["policy_kind"],
                    "policy_id": source["policy_id"],
                    "inference_mode": source.get("inference_mode"),
                    "evaluation_seed": source["evaluation_seed"],
                    "checkpoint": source.get("checkpoint"),
                    "checkpoint_sha256": source.get("checkpoint_sha256"),
                    "training_seed": source.get("training_seed"),
                    "benchmark_id": source.get("benchmark_id"),
                    "source_fingerprint": source["manifest_sha256"],
                },
                "environment_config": {
                    "map_size": payload["world"]["width"],
                    "max_steps": summary["world_steps"],
                    "num_agents": len(payload["world"]["initial"]["agents"]),
                },
                "tick_rate": payload["tick_rate"],
                "world_steps": summary["world_steps"],
                "agent_steps": summary["agent_steps"],
                "recorded_at": "legacy",
                "tags": ["legacy", "showcase"],
                "terminal_summary": summary,
                "registries": {},
                "artifacts": [],
                "extensions": {"legacy_schema_version": LEGACY_REPLAY_SCHEMA_VERSION},
            }
        )
        self._states = self._build_states()

    def initial(self) -> dict[str, Any]:
        return copy.deepcopy(self._states[0])

    def state_at(self, tick: int) -> dict[str, Any]:
        if tick < 0 or tick >= len(self._states):
            raise ValueError(f"Tick must be between 0 and {len(self._states) - 1}.")
        return copy.deepcopy(self._states[tick])

    def timeline(self, start: int, end: int) -> list[dict[str, Any]]:
        if start < 0 or end >= len(self._states) or start > end:
            raise ValueError("Invalid legacy timeline range.")
        return [
            {
                "tick": frame["step"],
                "actions": [],
                "events": [],
                "state_delta": {},
                "achievements": frame["new_achievements"],
                "weather": {"kind": "storm" if frame["storm"] else "clear", "storm_active": frame["storm"]},
            }
            for frame in self.payload["frames"]
            if start <= frame["step"] <= end
        ]

    def camera(self) -> dict[str, Any]:
        return {"version": "camera_director_legacy", "mode": "showcase", "cues": []}

    def metrics(self) -> dict[str, Any]:
        return {"global": {}, "extensions": {"legacy": True}}

    def validate(self, *, deep: bool = False) -> dict[str, Any]:
        final = self._states[-1]
        if sum(agent["alive"] for agent in final["agents"]) != self.manifest.terminal_summary["survivors"]:
            raise CorruptReplayError("Legacy terminal survivor count does not reconstruct.")
        return {
            "replay_id": self.manifest.replay_id,
            "status": "valid",
            "checked_artifacts": 1,
            "checked_ticks": len(self._states) if deep else 1,
        }

    def _build_states(self) -> list[dict[str, Any]]:
        initial = self.payload["world"]["initial"]
        resources = {
            f"resource-{item['x']}-{item['y']}": {
                "id": f"resource-{item['x']}-{item['y']}",
                **item,
            }
            for item in initial["resources"]
        }
        states = [
            {
                "tick": 0,
                "width": self.payload["world"]["width"],
                "height": self.payload["world"]["height"],
                "terrain": initial["terrain"],
                "camp": initial["camp"],
                "resources": list(resources.values()),
                "structures": [],
                "agents": initial["agents"],
                "achievements": [],
                "weather": {"kind": "clear", "storm_active": False},
                "extensions": {"legacy": True},
            }
        ]
        achievements: list[str] = []
        for frame in self.payload["frames"]:
            for change in frame["resource_changes"]:
                key = f"resource-{change['x']}-{change['y']}"
                if change["quantity"] <= 0 or change["type"] == "none":
                    resources.pop(key, None)
                else:
                    resources[key] = {"id": key, **change}
            achievements.extend(frame["new_achievements"])
            states.append(
                {
                    **states[0],
                    "tick": frame["step"],
                    "camp": frame["camp"],
                    "resources": [resources[key] for key in sorted(resources)],
                    "agents": frame["agents"],
                    "achievements": sorted(set(achievements)),
                    "weather": {
                        "kind": "storm" if frame["storm"] else "clear",
                        "storm_active": frame["storm"],
                    },
                }
            )
        return states
