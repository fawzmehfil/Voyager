"""Record the Stage 7B deterministic-core demonstration as Replay 2.2."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voyager.envs.civilization_v2 import VoyagerCivilizationV2Env
from voyager.policies.civilization_v2_scripted import CivilizationV2ScriptedController
from voyager.sim.constants import Terrain
from voyager.sim.registries_v2 import CivilizationV2Argument, CivilizationV2Verb

from .civilization import _state_hash
from .director import generate_camera_cues
from .recorder import _artifact_index, _registries, _state_delta, _world_snapshot
from .schema import (
    DETERMINISTIC_CORE_REPLAY_SCHEMA_VERSION,
    SNAPSHOT_INTERVAL,
    TIMELINE_CHUNK_SIZE,
    validate_manifest,
)
from .serialization import sha256_value, write_json, write_json_gz

DETERMINISTIC_CORE_REPLAY_ID = "civilization_deterministic_core_v1"
DETERMINISTIC_CORE_SHOWCASE_SEED = 7


def record_civilization_deterministic_core(
    output_root: str | Path = "runs/replays",
    *,
    seed: int = DETERMINISTIC_CORE_SHOWCASE_SEED,
    overwrite: bool = False,
) -> Path:
    """Write, validate, and atomically publish the 600-tick Replay 2.2 artifact."""

    output_root = Path(output_root).resolve()
    target = output_root / DETERMINISTIC_CORE_REPLAY_ID
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay already exists at {target}.")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{DETERMINISTIC_CORE_REPLAY_ID}.", dir=output_root))
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
    env = VoyagerCivilizationV2Env()
    env.reset(seed=seed)
    controller = CivilizationV2ScriptedController()
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
    }
    previous = initial_world
    timeline: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {0: initial}
    all_events: list[dict[str, Any]] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    invalid_actions = 0

    while env.agents:
        actions = controller.act_many(env)
        _observations, rewards, _terminations, _truncations, infos = env.step(actions)
        state = env.world.state
        assert state is not None
        world = _world_snapshot(env)
        state_hash = _state_hash(world)
        world["extensions"] = {**world.get("extensions", {}), "state_hash": state_hash}
        action_records = []
        for agent_id, selected in sorted(actions.items()):
            info = infos[agent_id]
            invalid = bool(info.get("invalid_action", False))
            invalid_actions += int(invalid)
            action_records.append(
                {
                    "agent_id": agent_id,
                    "role": state.agents[agent_id].role,
                    "selected_action": CivilizationV2Verb(selected["verb"]).name.lower(),
                    "selected_action_id": selected["verb"],
                    "selected_argument": CivilizationV2Argument(selected["argument"]).name.lower(),
                    "selected_argument_id": selected["argument"],
                    "selected_target": selected["target"],
                    "invalid": invalid,
                    "reward": float(rewards[agent_id]),
                    "reward_components": dict(info.get("reward_components", {})),
                }
            )
        events = [dict(event) for event in state.events]
        all_events.extend(events)
        record = {
            "tick": state.step_count,
            "actions": action_records,
            "events": events,
            "state_delta": _state_delta(previous, world),
            "achievements": sorted(
                {
                    achievement
                    for info in infos.values()
                    for achievement in info.get("new_achievements", ())
                }
            ),
            "weather": {"kind": "clear", "storm_active": False},
            "extensions": {
                "state_hash": state_hash,
                "conflicts": [event for event in events if event["type"] == "intent_rejected"],
            },
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
        metrics["active"].append(float(sum(agent.life_state == "active" for agent in state.agents.values())))
        metrics["downed"].append(float(sum(agent.life_state == "downed" for agent in state.agents.values())))
        metrics["ledger_entries"].append(float(len(state.ledger)))

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
        "downings": sum(event["type"] == "agent_downed" for event in all_events),
        "revivals": sum(event["type"] == "agent_revived" for event in all_events),
        "achievements": sorted(state.achievements),
        "camp_stockpile": dict(state.camp.stockpile),
        "structures": {
            key: {"complete": value.complete, "condition": value.condition}
            for key, value in sorted(state.structures.items())
        },
        "invalid_scripted_actions": invalid_actions,
        "conservation": env.world.reconcile_v2_ledger(),
        "state_hash": _state_hash(previous),
    }
    if terminal_summary["revivals"] < 1:
        raise RuntimeError("Stage 7B demonstration must complete at least one revival.")
    registries = _registries(initial)
    registries.update(
        {
            "verbs": [
                {"id": value.name.lower(), "value": int(value), "label": value.name.title()}
                for value in CivilizationV2Verb
            ],
            "arguments": [
                {"id": value.name.lower(), "value": int(value), "label": value.name.title()}
                for value in CivilizationV2Argument
            ],
        }
    )
    payload: dict[str, Any] = {
        "replay_id": DETERMINISTIC_CORE_REPLAY_ID,
        "versions": {
            "replay": DETERMINISTIC_CORE_REPLAY_SCHEMA_VERSION,
            "environment": "voyager_civilization_v2",
            "scenario": state.scenario_id,
            "reward": "civilization_reward_v1",
            "observation": "civilization_local_observation_v2",
            "action": "civilization_structured_action_v2",
            "achievement": "civilization_achievements_v1",
        },
        "status": "complete",
        "source": {
            "policy_kind": "scripted",
            "policy_id": controller.policy_id,
            "inference_mode": "privileged_deterministic",
            "evaluation_seed": seed,
            "source_fingerprint": sha256_value(
                {"policy": controller.policy_id, "seed": seed, "version": "stage7b_v1"}
            ),
        },
        "environment_config": {
            "num_agents": 10,
            "map_size": 48,
            "max_steps": 600,
            "local_view_size": 7,
            "inventory_capacity": 10,
        },
        "tick_rate": 2,
        "world_steps": state.step_count,
        "agent_steps": terminal_summary["agent_steps"],
        "recorded_at": datetime.now(UTC).isoformat(),
        "tags": ["stage7b", "deterministic-core", "scripted", "public-actions"],
        "terminal_summary": terminal_summary,
        "registries": registries,
        "artifacts": _artifact_index(directory),
        "extensions": {
            "privileged_policy": True,
            "playback_seconds": 300,
            "ledger_artifact": "ledger.json.gz",
        },
    }
    provisional = validate_manifest(payload)
    normalized = provisional.model_dump(mode="json")
    normalized.pop("manifest_sha256", None)
    normalized["manifest_sha256"] = sha256_value(normalized)
    write_json(directory / "manifest.json", validate_manifest(normalized).model_dump(mode="json"))
