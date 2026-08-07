"""Record the public-observation VoyagerIsland-v1 oracle as Replay 2.3."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voyager.envs.island import (
    ISLAND_ENVIRONMENT_VERSION,
    ISLAND_OBSERVATION_VERSION,
    ISLAND_REWARD_VERSION,
    VoyagerIslandEnv,
)
from voyager.policies.island_scripted import ScriptedIslandOracle
from voyager.sim.constants import Terrain
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION, ISLAND_ACHIEVEMENTS
from voyager.sim.island_registry import ISLAND_ACTION_VERSION, IslandAction

from .civilization import _state_hash
from .director import generate_camera_cues
from .recorder import _artifact_index, _registries, _state_delta, _world_snapshot
from .schema import (
    ISLAND_REPLAY_SCHEMA_VERSION,
    SNAPSHOT_INTERVAL,
    TIMELINE_CHUNK_SIZE,
    validate_manifest,
)
from .serialization import sha256_value, write_json, write_json_gz

ISLAND_ORACLE_REPLAY_ID = "island_benchmark_oracle_v1"
ISLAND_ORACLE_REPLAY_SEED = 3


def record_island_oracle_replay(
    output_root: str | Path = "runs/replays",
    *,
    seed: int = ISLAND_ORACLE_REPLAY_SEED,
    overwrite: bool = False,
) -> Path:
    """Atomically write and deeply validate the canonical Replay 2.3 artifact."""

    root = Path(output_root).resolve()
    target = root / ISLAND_ORACLE_REPLAY_ID
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay already exists at {target}.")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ISLAND_ORACLE_REPLAY_ID}.", dir=root))
    try:
        _record_into(temporary, seed=seed)
        from .loader import ReplayLoader

        ReplayLoader(temporary).validate(deep=True)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _record_into(directory: Path, *, seed: int) -> None:
    env = VoyagerIslandEnv(procedural=True)
    observations, infos = env.reset(seed=seed)
    controller = ScriptedIslandOracle()
    controller.reset(tuple(env.possible_agents))
    state = env.world.state
    assert state is not None
    initial_world = _world_snapshot(env)
    initial = {
        "tick": 0,
        "width": env.map_size,
        "height": env.map_size,
        "terrain": [[Terrain(int(cell)).name.lower() for cell in row] for row in state.terrain],
        **initial_world,
        "weather": {"kind": "clear", "storm_active": False},
    }
    initial["extensions"] = {
        **initial.get("extensions", {}),
        "state_hash": _state_hash(initial_world),
        "replay_profile": "voyager_island_v1",
    }
    previous = initial_world
    timeline: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {0: initial}
    all_events: list[dict[str, Any]] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    invalid_actions = 0

    while env.agents:
        actions = controller.act(observations, infos)
        observations, rewards, _terminations, _truncations, step_infos = env.step(actions)
        infos.update(step_infos)
        state = env.world.state
        assert state is not None
        world = _world_snapshot(env)
        state_hash = _state_hash(world)
        world["extensions"] = {**world.get("extensions", {}), "state_hash": state_hash}
        action_records: list[dict[str, Any]] = []
        for agent_id, selected in sorted(actions.items()):
            info = step_infos[agent_id]
            invalid = bool(info.get("invalid_action", False))
            invalid_actions += int(invalid)
            action = IslandAction(selected)
            action_records.append(
                {
                    "agent_id": agent_id,
                    "role": "survivor",
                    "selected_action": action.name.lower(),
                    "selected_action_id": int(action),
                    "invalid": invalid,
                    "reward": float(rewards[agent_id]),
                    "reward_components": dict(info.get("reward_components", {})),
                }
            )
        events = [dict(event) for event in state.events]
        all_events.extend(events)
        night = 200 <= state.step_count % 300 < 300
        record = {
            "tick": state.step_count,
            "actions": action_records,
            "events": events,
            "state_delta": _state_delta(previous, world),
            "achievements": sorted(
                {
                    achievement
                    for info in step_infos.values()
                    for achievement in info.get("new_achievements", ())
                }
            ),
            "weather": {"kind": "night" if night else "clear", "storm_active": False},
            "extensions": {"state_hash": state_hash},
        }
        timeline.append(record)
        previous = world
        if state.step_count % SNAPSHOT_INTERVAL == 0 or not env.agents:
            snapshots[state.step_count] = {
                "tick": state.step_count,
                **world,
                "weather": record["weather"],
            }
        metrics["tick"].append(float(state.step_count))
        metrics["active"].append(float(sum(agent.alive for agent in state.agents.values())))
        metrics["achievements"].append(float(len(state.achievements)))

    write_json_gz(directory / "initial.json.gz", initial)
    for index in range(0, len(timeline), TIMELINE_CHUNK_SIZE):
        records = timeline[index : index + TIMELINE_CHUNK_SIZE]
        start, end = records[0]["tick"], records[-1]["tick"]
        write_json_gz(
            directory / "timeline" / f"{start:06d}-{end:06d}.json.gz",
            {"start_tick": start, "end_tick": end, "records": records},
        )
    for tick, snapshot in sorted(snapshots.items()):
        write_json_gz(directory / "snapshots" / f"{tick:06d}.json.gz", snapshot)
    write_json_gz(directory / "metrics.json.gz", {"global": metrics, "extensions": {}})
    write_json_gz(
        directory / "ledger.json.gz",
        {
            "entries": state.ledger,
            "contributions": env.world.metrics()["contributions"],
            "conservation": env.world.reconcile_v2_ledger(),
        },
    )
    write_json(
        directory / "camera.json",
        generate_camera_cues(
            all_events,
            terminal_tick=state.step_count,
            camp=initial["camp"],
            agents=initial["agents"],
        ),
    )
    terminal_summary = {
        "world_steps": state.step_count,
        "agent_steps": sum(len(record["actions"]) for record in timeline),
        "survivors": sum(agent.alive for agent in state.agents.values()),
        "deaths": state.deaths,
        "rescue_success": state.rescue_success,
        "achievements": sorted(state.achievements),
        "achievement_steps": dict(sorted(state.achievement_steps.items())),
        "camp_stockpile": dict(state.camp.stockpile),
        "structures": {
            name: {"complete": structure.complete, "progress": structure.progress}
            for name, structure in sorted(state.structures.items())
        },
        "invalid_scripted_actions": invalid_actions,
        "conservation": env.world.reconcile_v2_ledger(),
        "state_hash": _state_hash(previous),
    }
    if set(state.achievements) != set(ISLAND_ACHIEVEMENTS):
        raise RuntimeError("Canonical island replay must demonstrate every achievement.")
    if invalid_actions:
        raise RuntimeError("Canonical island replay must contain zero invalid scripted actions.")
    if terminal_summary["conservation"]:
        raise RuntimeError("Canonical island replay must reconcile its resource ledger.")
    registries = _registries(initial)
    registries["actions"] = [
        {"id": action.name.lower(), "value": int(action), "label": action.name.title()}
        for action in IslandAction
    ]
    registries["achievements"] = [
        {"id": achievement, "label": achievement.replace("_", " ").title()}
        for achievement in ISLAND_ACHIEVEMENTS
    ]
    payload: dict[str, Any] = {
        "replay_id": ISLAND_ORACLE_REPLAY_ID,
        "versions": {
            "replay": ISLAND_REPLAY_SCHEMA_VERSION,
            "environment": ISLAND_ENVIRONMENT_VERSION,
            "scenario": state.scenario_id,
            "reward": ISLAND_REWARD_VERSION,
            "observation": ISLAND_OBSERVATION_VERSION,
            "action": ISLAND_ACTION_VERSION,
            "achievement": ISLAND_ACHIEVEMENT_VERSION,
        },
        "status": "complete",
        "source": {
            "policy_kind": "scripted",
            "policy_id": "island_public_observation_oracle_v1",
            "inference_mode": "decentralized_deterministic",
            "evaluation_seed": seed,
            "source_fingerprint": sha256_value(
                {"policy": "island_public_observation_oracle_v1", "seed": seed}
            ),
        },
        "environment_config": {
            "num_agents": 2,
            "map_size": 48,
            "max_steps": 1_200,
            "local_view_size": 7,
            "inventory_capacity": 10,
            "procedural": True,
        },
        "tick_rate": 4,
        "world_steps": state.step_count,
        "agent_steps": terminal_summary["agent_steps"],
        "recorded_at": datetime.now(UTC).isoformat(),
        "tags": ["stage7", "island-benchmark", "oracle", "rescue"],
        "terminal_summary": terminal_summary,
        "registries": registries,
        "artifacts": _artifact_index(directory),
        "extensions": {
            "privileged_policy": False,
            "playback_seconds": 300,
            "ledger_artifact": "ledger.json.gz",
        },
    }
    provisional = validate_manifest(payload)
    normalized = provisional.model_dump(mode="json")
    normalized.pop("manifest_sha256", None)
    normalized["manifest_sha256"] = sha256_value(normalized)
    write_json(directory / "manifest.json", validate_manifest(normalized).model_dump(mode="json"))
