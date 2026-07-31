"""Policy-agnostic Stage 6 replay recording."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from voyager.benchmark.runner import (
    build_policy_runtime,
    decide_many,
    load_manifest,
    repository_root,
)
from voyager.benchmark.runner import (
    validate_manifest as validate_benchmark_manifest,
)
from voyager.benchmark.schema import BenchmarkManifest, PolicySpec
from voyager.envs import VoyagerParallelEnv
from voyager.policies.base import PolicyDecision
from voyager.sim.achievements import ACHIEVEMENT_IDS
from voyager.sim.constants import Action, Resource, Terrain
from voyager.training.masking import action_mask_from_info

from .director import generate_camera_cues
from .exporter import AGENT_IDENTITIES
from .schema import (
    REPLAY_SCHEMA_VERSION,
    SNAPSHOT_INTERVAL,
    TIMELINE_CHUNK_SIZE,
    ReplayManifest,
    validate_manifest,
)
from .serialization import sha256_file, sha256_value, write_json, write_json_gz

DEFAULT_CATALOG_ROOT = Path("runs/replays")
FAILED_DIRECTORY = ".failed"
ACTION_REGISTRY = [
    {"id": action.name.lower(), "value": int(action), "label": action.name.replace("_", " ").title()}
    for action in Action
]
TERRAIN_REGISTRY = [
    {"id": terrain.name.lower(), "value": int(terrain), "label": terrain.name.title()}
    for terrain in Terrain
]
RESOURCE_REGISTRY = [
    {"id": resource.name.lower(), "value": int(resource), "label": resource.name.title()}
    for resource in Resource
    if resource is not Resource.NONE
]
EVENT_TYPES = (
    "action",
    "gather",
    "deposit",
    "withdraw",
    "build",
    "eat",
    "death",
    "storm_started",
    "storm_ended",
    "achievement",
    "shared_food_transfer",
)


def record_episode(
    manifest_path: str | Path,
    *,
    policy_id: str,
    seed: int,
    output_root: str | Path = DEFAULT_CATALOG_ROOT,
    replay_id: str | None = None,
    tags: tuple[str, ...] = (),
    overwrite: bool = False,
    benchmark_episode: dict[str, Any] | None = None,
    camera_overrides: list[dict[str, Any]] | None = None,
    _benchmark_override: BenchmarkManifest | None = None,
) -> Path:
    """Execute one policy/seed pair and atomically catalog its replay."""

    manifest_path = Path(manifest_path).resolve()
    root = repository_root(manifest_path)
    benchmark = _benchmark_override or load_manifest(manifest_path)
    validate_benchmark_manifest(benchmark, root)
    policy_spec = _find_policy(benchmark, policy_id)
    if seed not in benchmark.seed_suite.seeds:
        raise ValueError(f"Seed {seed} is not part of {benchmark.seed_suite.id!r}.")

    fingerprint = _source_fingerprint(benchmark, policy_spec, seed, manifest_path)
    resolved_id = replay_id or stable_replay_id(
        benchmark.scenario.id,
        policy_spec.id,
        seed,
        policy_spec.inference_mode,
        fingerprint,
    )
    output_root = Path(output_root).resolve()
    target = output_root / resolved_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay {resolved_id!r} already exists.")

    output_root.mkdir(parents=True, exist_ok=True)
    failure_root = output_root / FAILED_DIRECTORY
    failure_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{resolved_id}.", dir=output_root))
    recorder = ReplayRecorder(
        temporary,
        replay_id=resolved_id,
        benchmark=benchmark,
        policy=policy_spec,
        seed=seed,
        source_fingerprint=fingerprint,
        tags=tags,
        benchmark_episode=benchmark_episode,
    )
    try:
        env = _create_environment(benchmark)
        observations, infos = env.reset(seed=seed)
        runtime = build_policy_runtime(policy_spec, seed, root, {})
        recorder.begin(env)
        while env.agents:
            agent_ids = tuple(env.agents)
            decisions = decide_many(runtime, agent_ids, observations, infos)
            actions = {agent_id: int(decisions[agent_id].action) for agent_id in agent_ids}
            observations, rewards, _terms, _truncs, step_infos = env.step(actions)
            recorder.record_tick(env, decisions, actions, rewards, infos, step_infos)
            infos.update(step_infos)
        recorder.finalize(env, camera_overrides=camera_overrides)
        from .loader import ReplayLoader

        ReplayLoader(temporary).validate(deep=True)
        if benchmark_episode is not None:
            _verify_benchmark_episode(recorder.terminal_summary, benchmark_episode)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        return target
    except Exception as exc:
        recorder.mark_failed(exc)
        failed_target = failure_root / f"{resolved_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        if temporary.exists():
            os.replace(temporary, failed_target)
        raise


def record_checkpoint_episode(
    manifest_path: str | Path,
    *,
    checkpoint: str | Path,
    seed: int,
    output_root: str | Path = DEFAULT_CATALOG_ROOT,
    policy_id: str = "ppo_checkpoint_deterministic",
    replay_id: str | None = None,
    deterministic: bool = True,
    tags: tuple[str, ...] = (),
    overwrite: bool = False,
) -> Path:
    """Record an arbitrary compatible PPO training output after training."""

    manifest_path = Path(manifest_path).resolve()
    root = repository_root(manifest_path)
    benchmark = load_manifest(manifest_path)
    checkpoint_path = Path(checkpoint).resolve()
    metadata = cast(
        dict[str, Any],
        json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8")),
    )
    weights_path = checkpoint_path / str(metadata["weights_file"])
    policy = PolicySpec(
        id=policy_id,
        kind="ppo",
        official=deterministic,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(weights_path),
        training_seed=int(metadata["training_seed"]),
        inference_mode="deterministic" if deterministic else "stochastic",
    )
    seed_suite = benchmark.seed_suite.model_copy(
        update={"id": "stage6_custom_recording", "seeds": [seed]}
    )
    custom = benchmark.model_copy(update={"policies": [policy], "seed_suite": seed_suite})
    validate_benchmark_manifest(custom, root)
    return record_episode(
        manifest_path,
        policy_id=policy_id,
        seed=seed,
        output_root=output_root,
        replay_id=replay_id,
        tags=tags,
        overwrite=overwrite,
        _benchmark_override=custom,
    )


def migrate_legacy_replay(
    legacy_path: str | Path,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert the permanent Stage 6A bundle into a best-effort v2 directory."""

    from .loader import ReplayLoader

    legacy_path = Path(legacy_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Replay output already exists at {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    loader = ReplayLoader(legacy_path)
    if not loader.manifest.extensions.get("legacy_schema_version"):
        raise ValueError("migrate requires a Stage 6A single-file replay.")
    try:
        initial = loader.initial()
        write_json_gz(temporary / "initial.json.gz", initial)
        timeline: list[dict[str, Any]] = []
        previous = initial
        for tick in range(1, loader.manifest.world_steps + 1):
            state = loader.state_at(tick)
            previous_achievements = set(previous.get("achievements", []))
            achievements = [
                value
                for value in state.get("achievements", [])
                if value not in previous_achievements
            ]
            timeline.append(
                {
                    "tick": tick,
                    "actions": [],
                    "events": [],
                    "state_delta": _state_delta(previous, state),
                    "achievements": achievements,
                    "weather": state["weather"],
                    "extensions": {"legacy_action_diagnostics_unavailable": True},
                }
            )
            if tick % SNAPSHOT_INTERVAL == 0 or tick == loader.manifest.world_steps:
                write_json_gz(
                    temporary / "snapshots" / f"{tick:06d}.json.gz",
                    state,
                )
            previous = state
        write_json_gz(temporary / "snapshots/000000.json.gz", initial)
        for index in range(0, len(timeline), TIMELINE_CHUNK_SIZE):
            records = timeline[index : index + TIMELINE_CHUNK_SIZE]
            start, end = records[0]["tick"], records[-1]["tick"]
            write_json_gz(
                temporary / "timeline" / f"{start:06d}-{end:06d}.json.gz",
                {"start_tick": start, "end_tick": end, "records": records},
            )
        write_json_gz(
            temporary / "metrics.json.gz",
            {"global": {}, "extensions": {"legacy_metrics_unavailable": True}},
        )
        write_json(temporary / "camera.json", loader.camera())
        payload = loader.manifest.model_dump(mode="json")
        payload.update(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "artifacts": _artifact_index(temporary),
                "extensions": {
                    **payload.get("extensions", {}),
                    "migration": {
                        "source": str(legacy_path),
                        "scientific_limitations": [
                            "policy diagnostics unavailable",
                            "typed events unavailable",
                            "metric series unavailable",
                        ],
                    },
                },
            }
        )
        payload["source"]["source_fingerprint"] = sha256_file(legacy_path)
        provisional = validate_manifest(payload)
        normalized = provisional.model_dump(mode="json")
        normalized.pop("manifest_sha256", None)
        normalized["manifest_sha256"] = sha256_value(normalized)
        manifest = validate_manifest(normalized)
        write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        ReplayLoader(temporary).validate(deep=True)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class ReplayRecorder:
    """Lower-level recorder for custom loops that keep normal env.step semantics."""

    def __init__(
        self,
        directory: Path,
        *,
        replay_id: str,
        benchmark: BenchmarkManifest,
        policy: PolicySpec,
        seed: int,
        source_fingerprint: str,
        tags: tuple[str, ...] = (),
        benchmark_episode: dict[str, Any] | None = None,
    ) -> None:
        self.directory = directory
        self.replay_id = replay_id
        self.benchmark = benchmark
        self.policy = policy
        self.seed = seed
        self.source_fingerprint = source_fingerprint
        self.tags = list(dict.fromkeys(tags))
        self.benchmark_episode = benchmark_episode
        self.timeline: list[dict[str, Any]] = []
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.metric_series: dict[str, list[float]] = defaultdict(list)
        self.agent_steps = 0
        self.dense_return = 0.0
        self.achievement_return = 0.0
        self.reward_totals: dict[str, float] = defaultdict(float)
        self.reward_by_agent: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.reward_by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.initial: dict[str, Any] | None = None
        self.previous_state: dict[str, Any] | None = None
        self.terminal_summary: dict[str, Any] = {}
        self.started_at = datetime.now(UTC)
        self.directory.mkdir(parents=True, exist_ok=True)

    def begin(self, env: VoyagerParallelEnv) -> None:
        state = _world_state(env)
        world = _world_snapshot(env)
        self.initial = {
            "tick": 0,
            "width": env.map_size,
            "height": env.map_size,
            "terrain": [
                [Terrain(int(cell)).name.lower() for cell in row] for row in state.terrain
            ],
            **world,
            "weather": {"kind": "clear", "storm_active": False},
            "extensions": {},
        }
        self.previous_state = world
        self.snapshots[0] = self.initial
        self._append_metrics(env)

    def record_tick(
        self,
        env: VoyagerParallelEnv,
        decisions: dict[str, PolicyDecision],
        actions: dict[str, int],
        rewards: dict[str, float],
        pre_step_infos: dict[str, dict[str, Any]],
        step_infos: dict[str, dict[str, Any]],
    ) -> None:
        if self.previous_state is None:
            raise RuntimeError("ReplayRecorder.begin must be called before record_tick.")
        state = _world_state(env)
        tick = int(state.step_count)
        world = _world_snapshot(env)
        action_records: list[dict[str, Any]] = []
        tick_events: list[dict[str, Any]] = []
        for agent_id, decision in sorted(decisions.items()):
            info_before = pre_step_infos[agent_id]
            info_after = step_infos[agent_id]
            role = str(info_before.get("role", "unknown"))
            mask = action_mask_from_info(info_before)
            selected = int(actions[agent_id])
            dense = _float_dict(info_after.get("dense_reward_components", {}))
            selected_components = _float_dict(info_after.get("reward_components", {}))
            for name, value in dense.items():
                self.reward_totals[name] += value
                self.reward_by_agent[agent_id][name] += value
                self.reward_by_role[role][name] += value
                self.dense_return += value
            action_records.append(
                {
                    "agent_id": agent_id,
                    "role": role,
                    "selected_action": Action(selected).name.lower(),
                    "selected_action_id": selected,
                    "raw_action": Action(int(decision.raw_action)).name.lower(),
                    "raw_action_id": int(decision.raw_action),
                    "action_mask": [int(value) for value in mask.tolist()],
                    "invalid_probability_mass": float(decision.invalid_probability_mass),
                    "reward": float(rewards[agent_id]),
                    "selected_reward_components": selected_components,
                    "dense_reward_components": dense,
                }
            )
            event = _event_from_agent_info(tick, agent_id, info_after, world)
            if event is not None:
                tick_events.append(event)

        new_achievements = _new_achievements(step_infos)
        for achievement_id in new_achievements:
            tick_events.append(
                {
                    "tick": tick,
                    "type": (
                        "shared_food_transfer"
                        if achievement_id == "shared_food_transfer"
                        else "achievement"
                    ),
                    "importance": 76 if achievement_id == "shared_food_transfer" else 70,
                    "actors": [],
                    "targets": [],
                    "position": _position(world["camp"]),
                    "tags": ["achievement", achievement_id],
                    "payload": {"achievement_id": achievement_id},
                }
            )
        was_storm = bool(self.timeline[-1]["weather"]["storm_active"]) if self.timeline else False
        is_storm = env.world.is_storm_active()
        if is_storm != was_storm:
            tick_events.append(
                {
                    "tick": tick,
                    "type": "storm_started" if is_storm else "storm_ended",
                    "importance": 90 if is_storm else 85,
                    "actors": [],
                    "targets": [],
                    "position": _position(world["camp"]),
                    "tags": ["weather"],
                    "payload": {},
                }
            )

        delta = _state_delta(self.previous_state, world)
        record: dict[str, Any] = {
            "tick": tick,
            "actions": action_records,
            "events": sorted(tick_events, key=lambda value: (value["type"], value["actors"])),
            "state_delta": delta,
            "achievements": new_achievements,
            "weather": {"kind": "storm" if is_storm else "clear", "storm_active": is_storm},
            "extensions": {},
        }
        self.timeline.append(record)
        self.events.extend(record["events"])
        self.agent_steps += len(actions)
        self.achievement_return += len(new_achievements) * len(actions)
        self.previous_state = world
        if tick % SNAPSHOT_INTERVAL == 0 or not env.agents:
            self.snapshots[tick] = {
                "tick": tick,
                **world,
                "weather": record["weather"],
                "extensions": {},
            }
        self._append_metrics(env)

    def finalize(
        self,
        env: VoyagerParallelEnv,
        *,
        camera_overrides: list[dict[str, Any]] | None = None,
    ) -> ReplayManifest:
        if self.initial is None or self.previous_state is None:
            raise RuntimeError("Cannot finalize an empty recording.")
        terminal_tick = int(_world_state(env).step_count)
        if terminal_tick not in self.snapshots:
            self.snapshots[terminal_tick] = {
                "tick": terminal_tick,
                **self.previous_state,
                "weather": self.timeline[-1]["weather"],
                "extensions": {},
            }
        metrics = env.metrics()
        camp = cast(dict[str, Any], metrics["camp"])
        self.terminal_summary = {
            "world_steps": terminal_tick,
            "agent_steps": self.agent_steps,
            "survivors": int(cast(int, metrics["active_agents"])),
            "deaths": int(cast(int, metrics["deaths"])),
            "dense_return": self.dense_return,
            "achievement_return": self.achievement_return,
            "achievements": list(cast(list[str], metrics["achievements"])),
            "achievement_steps": dict(cast(dict[str, int], metrics["achievement_steps"])),
            "camp_stockpile": dict(cast(dict[str, Any], camp["stockpile"])),
            "shelter_progress": float(camp["shelter_progress"]),
            "shelter_completion_step": metrics["shelter_completion_step"],
            "resource_flow": dict(cast(dict[str, Any], metrics["resource_flow"])),
        }
        write_json_gz(self.directory / "initial.json.gz", self.initial)
        for start_index in range(0, len(self.timeline), TIMELINE_CHUNK_SIZE):
            records = self.timeline[start_index : start_index + TIMELINE_CHUNK_SIZE]
            start_tick, end_tick = records[0]["tick"], records[-1]["tick"]
            write_json_gz(
                self.directory / "timeline" / f"{start_tick:06d}-{end_tick:06d}.json.gz",
                {"start_tick": start_tick, "end_tick": end_tick, "records": records},
            )
        for tick, snapshot in sorted(self.snapshots.items()):
            write_json_gz(self.directory / "snapshots" / f"{tick:06d}.json.gz", snapshot)
        metrics_payload = {
            "global": dict(self.metric_series),
            "reward_components": dict(sorted(self.reward_totals.items())),
            "reward_components_by_agent": {
                agent: dict(sorted(values.items()))
                for agent, values in sorted(self.reward_by_agent.items())
            },
            "reward_components_by_role": {
                role: dict(sorted(values.items()))
                for role, values in sorted(self.reward_by_role.items())
            },
            "extensions": {},
        }
        write_json_gz(self.directory / "metrics.json.gz", metrics_payload)
        camera = generate_camera_cues(
            self.events,
            terminal_tick=terminal_tick,
            camp=cast(dict[str, Any], self.initial["camp"]),
            agents=cast(list[dict[str, Any]], self.initial["agents"]),
            overrides=camera_overrides,
        )
        write_json(self.directory / "camera.json", camera)
        artifacts = _artifact_index(self.directory)
        manifest_payload: dict[str, Any] = {
            "replay_id": self.replay_id,
            "versions": {
                "replay": REPLAY_SCHEMA_VERSION,
                "environment": self.benchmark.scenario.environment_version,
                "scenario": self.benchmark.scenario.id,
                "reward": self.benchmark.scenario.reward_version,
                "observation": self.benchmark.scenario.observation_version,
                "action": self.benchmark.scenario.action_version,
                "achievement": self.benchmark.scenario.achievement_version,
            },
            "status": "complete",
            "source": {
                "policy_kind": self.policy.kind,
                "policy_id": self.policy.id,
                "inference_mode": self.policy.inference_mode,
                "evaluation_seed": self.seed,
                "checkpoint": self.policy.checkpoint,
                "checkpoint_sha256": self.policy.checkpoint_sha256,
                "training_seed": self.policy.training_seed,
                "git_revision": _git_revision(),
                "dependency_versions": _dependency_versions(),
                "benchmark_id": self.benchmark.benchmark_id,
                "benchmark_episode": self.benchmark_episode,
                "source_fingerprint": self.source_fingerprint,
            },
            "environment_config": dict(self.benchmark.scenario.config),
            "tick_rate": 12,
            "world_steps": terminal_tick,
            "agent_steps": self.agent_steps,
            "recorded_at": self.started_at.isoformat(),
            "tags": self.tags,
            "terminal_summary": self.terminal_summary,
            "registries": _registries(self.initial),
            "artifacts": artifacts,
            "extensions": {},
        }
        provisional = validate_manifest(manifest_payload)
        normalized = provisional.model_dump(mode="json")
        normalized.pop("manifest_sha256", None)
        normalized["manifest_sha256"] = sha256_value(normalized)
        manifest = validate_manifest(normalized)
        write_json(self.directory / "manifest.json", manifest.model_dump(mode="json"))
        return manifest

    def mark_failed(self, exc: Exception) -> None:
        if not self.directory.exists():
            return
        write_json(
            self.directory / "failure.json",
            {
                "replay_id": self.replay_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "recorded_ticks": len(self.timeline),
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )

    def _append_metrics(self, env: VoyagerParallelEnv) -> None:
        state = _world_state(env)
        agents = list(state.agents.values())
        alive = [agent for agent in agents if agent.alive]
        population = alive or agents
        self.metric_series["tick"].append(float(state.step_count))
        self.metric_series["survivors"].append(float(len(alive)))
        self.metric_series["mean_health"].append(float(np.mean([agent.health for agent in population])))
        self.metric_series["mean_hunger"].append(float(np.mean([agent.hunger for agent in population])))
        self.metric_series["mean_energy"].append(float(np.mean([agent.energy for agent in population])))
        self.metric_series["camp_food"].append(float(state.camp.stockpile["food"]))
        self.metric_series["camp_wood"].append(float(state.camp.stockpile["wood"]))
        self.metric_series["camp_stone"].append(float(state.camp.stockpile["stone"]))
        self.metric_series["shelter_progress"].append(float(state.camp.shelter_progress))
        self.metric_series["food_security_duration"].append(float(state.food_security_steps))


def stable_replay_id(
    scenario_id: str,
    policy_id: str,
    seed: int,
    inference_mode: str | None,
    fingerprint: str,
) -> str:
    mode = inference_mode or "default"
    raw = f"{scenario_id}-{policy_id}-{seed}-{mode}-{fingerprint[:10]}"
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in raw)


def _create_environment(manifest: BenchmarkManifest) -> VoyagerParallelEnv:
    config = manifest.scenario.config
    return VoyagerParallelEnv(
        num_agents=int(config["num_agents"]),
        map_size=int(config["map_size"]),
        max_steps=int(config["max_steps"]),
        local_view_size=int(config["local_view_size"]),
        inventory_capacity=int(config["inventory_capacity"]),
        storm_start_step=int(config["storm_start_step"]),
        storm_interval=int(config["storm_interval"]),
        storm_duration=int(config["storm_duration"]),
        storm_damage=float(config["storm_damage"]),
        food_regen_interval=int(config["food_regen_interval"]),
        food_spawn_rate=float(config["food_spawn_rate"]),
        reward_mode="dense",
    )


def _find_policy(manifest: BenchmarkManifest, policy_id: str) -> PolicySpec:
    for policy in manifest.policies:
        if policy.id == policy_id:
            return policy
    raise ValueError(f"Policy {policy_id!r} is absent from the benchmark manifest.")


def _world_state(env: VoyagerParallelEnv) -> Any:
    if env.world.state is None:
        raise RuntimeError("Environment state is unavailable.")
    return env.world.state


def _world_snapshot(env: VoyagerParallelEnv) -> dict[str, Any]:
    state = _world_state(env)
    civilization = bool(state.structures)
    return {
        "camp": {
            "id": "camp",
            "x": state.camp.x,
            "y": state.camp.y,
            "stockpile": dict(state.camp.stockpile),
            "shelter_progress": round(float(state.camp.shelter_progress), 6),
            "shelter_capacity": int(state.camp.shelter_capacity),
        },
        "resources": [
            {
                "id": f"resource-{x}-{y}",
                "x": int(x),
                "y": int(y),
                "type": Resource(int(state.resource_ids[y, x])).name.lower(),
                "quantity": int(state.resource_quantities[y, x]),
            }
            for y, x in np.argwhere(state.resource_quantities > 0)
        ],
        "structures": (
            [
                {
                    "id": structure.id,
                    "type": structure.type,
                    "x": structure.x,
                    "y": structure.y,
                    "progress": round(float(structure.progress), 6),
                    "complete": structure.complete,
                    "labor": structure.labor,
                    "required_labor": structure.required_labor,
                    "condition": structure.condition,
                    "capacity": structure.capacity,
                    "occupants": sorted(structure.occupants),
                    "fuel": structure.fuel,
                }
                for structure in sorted(state.structures.values(), key=lambda value: value.id)
            ]
            if civilization
            else [
                {
                    "id": "shelter",
                    "type": "shelter",
                    "x": state.camp.x,
                    "y": state.camp.y,
                    "progress": round(float(state.camp.shelter_progress), 6),
                    "complete": state.camp.shelter_progress >= 1.0,
                }
            ]
        ),
        "creatures": [
            {
                "id": creature.id,
                "type": creature.type,
                "x": creature.x,
                "y": creature.y,
                "health": creature.health,
                "max_health": creature.max_health,
                "alive": creature.alive,
                "target": creature.target,
                "behavior": creature.behavior,
                "spawn_tick": creature.spawn_tick,
            }
            for creature in sorted(state.creatures.values(), key=lambda value: value.id)
        ],
        "agents": [
            _agent_snapshot(agent_id, agent, index)
            for index, (agent_id, agent) in enumerate(sorted(state.agents.items()))
        ],
        "achievements": sorted(state.achievements),
        "time": env.world.civilization_time() if civilization else None,
        "extensions": {"scenario_id": state.scenario_id},
    }


def _agent_snapshot(agent_id: str, agent: Any, index: int) -> dict[str, Any]:
    base = AGENT_IDENTITIES[index % len(AGENT_IDENTITIES)]
    cycle = index // len(AGENT_IDENTITIES)
    return {
        "id": agent_id,
        "name": str(base["name"]) if cycle == 0 else f"{base['name']} {cycle + 1}",
        "role": agent.role,
        "appearance": {**base, "variant": index, "palette_shift": cycle},
        "x": agent.x,
        "y": agent.y,
        "health": round(float(agent.health), 4),
        "hunger": round(float(agent.hunger), 4),
        "energy": round(float(agent.energy), 4),
        "alive": bool(agent.alive),
        "inventory": dict(agent.inventory),
        "tools": sorted(agent.tools),
        "equipped_tool": agent.equipped_tool,
        "sheltered": agent.sheltered,
        "extensions": {},
    }


def _state_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_resources = {resource["id"]: resource for resource in previous["resources"]}
    current_resources = {resource["id"]: resource for resource in current["resources"]}
    resource_changes = []
    for resource_id in sorted(set(previous_resources) | set(current_resources)):
        if previous_resources.get(resource_id) == current_resources.get(resource_id):
            continue
        resource_changes.append(
            current_resources[resource_id]
            if resource_id in current_resources
            else {**previous_resources[resource_id], "type": "none", "quantity": 0}
        )
    return {
        "camp": current["camp"],
        "agents": current["agents"],
        "structures": current["structures"],
        "creatures": current.get("creatures", []),
        "time": current.get("time"),
        "resource_changes": resource_changes,
        "achievements": current["achievements"],
        "extensions": current.get("extensions", {}),
    }


def apply_state_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Apply one portable state delta without requiring the simulation package."""

    resources = {resource["id"]: dict(resource) for resource in state["resources"]}
    for change in delta.get("resource_changes", []):
        if int(change.get("quantity", 0)) <= 0 or change.get("type") == "none":
            resources.pop(change["id"], None)
        else:
            resources[change["id"]] = dict(change)
    return {
        **state,
        "camp": delta["camp"],
        "agents": delta["agents"],
        "structures": delta["structures"],
        "creatures": delta.get("creatures", state.get("creatures", [])),
        "time": delta.get("time", state.get("time")),
        "resources": [resources[key] for key in sorted(resources)],
        "achievements": delta["achievements"],
        "extensions": {**state.get("extensions", {}), **delta.get("extensions", {})},
    }


def _event_from_agent_info(
    tick: int, agent_id: str, info: dict[str, Any], world: dict[str, Any]
) -> dict[str, Any] | None:
    raw = str(info.get("event", "")).lower()
    if not raw or raw in {"reset", "noop", "moved", "rested"}:
        return None
    event_type = next(
        (name for name in ("gather", "deposit", "withdraw", "build", "eat", "death") if name in raw),
        "action",
    )
    agent = next(value for value in world["agents"] if value["id"] == agent_id)
    importance = {
        "death": 100,
        "build": 58,
        "deposit": 48,
        "gather": 38,
        "withdraw": 35,
        "eat": 30,
        "action": 20,
    }[event_type]
    return {
        "tick": tick,
        "type": event_type,
        "importance": importance,
        "actors": [agent_id],
        "targets": [],
        "position": {"x": agent["x"], "y": agent["y"]},
        "tags": [event_type, str(agent["role"])],
        "payload": {"message": raw},
    }


def _new_achievements(infos: dict[str, dict[str, Any]]) -> list[str]:
    if not infos:
        return []
    value = next(iter(infos.values())).get("new_achievements", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _registries(initial: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    roles = sorted({str(agent["role"]) for agent in initial["agents"]})
    return {
        "terrain": TERRAIN_REGISTRY,
        "resources": RESOURCE_REGISTRY,
        "structures": [
            {"id": "camp", "label": "Camp"},
            {"id": "shelter", "label": "Shelter"},
        ],
        "roles": [{"id": role, "label": role.replace("_", " ").title()} for role in roles],
        "actions": ACTION_REGISTRY,
        "achievements": [
            {"id": achievement, "label": achievement.replace("_", " ").title()}
            for achievement in ACHIEVEMENT_IDS
        ],
        "events": [
            {"id": event, "label": event.replace("_", " ").title()} for event in EVENT_TYPES
        ],
        "metrics": [
            {"id": name, "label": name.replace("_", " ").title()}
            for name in (
                "survivors",
                "mean_health",
                "mean_hunger",
                "mean_energy",
                "camp_food",
                "camp_wood",
                "camp_stone",
                "shelter_progress",
                "food_security_duration",
            )
        ],
    }


def _artifact_index(directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "failure.json"}:
            continue
        relative = path.relative_to(directory).as_posix()
        entry: dict[str, Any] = {
            "path": relative,
            "kind": relative.split("/", maxsplit=1)[0].replace(".json.gz", ""),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if relative.startswith("timeline/"):
            start, end = path.name.removesuffix(".json.gz").split("-", maxsplit=1)
            entry.update(start_tick=int(start), end_tick=int(end))
        elif relative.startswith("snapshots/"):
            tick = int(path.name.removesuffix(".json.gz"))
            entry.update(start_tick=tick, end_tick=tick)
        entries.append(entry)
    return entries


def _source_fingerprint(
    manifest: BenchmarkManifest,
    policy: PolicySpec,
    seed: int,
    manifest_path: Path,
) -> str:
    return sha256_value(
        {
            "benchmark": manifest.model_dump(mode="json"),
            "manifest_sha256": sha256_file(manifest_path),
            "policy": policy.model_dump(mode="json"),
            "seed": seed,
            "recorder": REPLAY_SCHEMA_VERSION,
        }
    )


def _verify_benchmark_episode(summary: dict[str, Any], episode: dict[str, Any]) -> None:
    fields = {
        "world_steps": "world_steps",
        "agent_steps": "agent_steps",
        "survivors": "survivors",
        "deaths": "deaths",
        "achievements": "achievements",
        "achievement_steps": "achievement_steps",
        "camp_stockpile": "camp_stockpile",
        "shelter_completion_step": "shelter_completion_step",
    }
    for summary_key, episode_key in fields.items():
        if summary[summary_key] != episode[episode_key]:
            raise ValueError(f"Replay does not match benchmark field {episode_key!r}.")
    if not np.isclose(summary["dense_return"], episode["dense_return"], atol=1e-9, rtol=0):
        raise ValueError("Replay dense return does not match its benchmark episode.")


def _position(value: dict[str, Any]) -> dict[str, int]:
    return {"x": int(value["x"]), "y": int(value["y"])}


def _float_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in ("numpy", "gymnasium", "pettingzoo", "pydantic", "tensorflow"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            continue
    return values
