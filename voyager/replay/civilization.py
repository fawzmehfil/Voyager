"""Record the deterministic Stage 7A handcrafted vertical slice."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voyager.envs.civilization import VoyagerCivilizationEnv
from voyager.policies.civilization_scripted import CivilizationScriptedController
from voyager.sim.constants import Terrain
from voyager.sim.registries import CivilizationArgument, CivilizationVerb

from .director import generate_camera_cues
from .recorder import _artifact_index, _registries, _state_delta, _world_snapshot
from .schema import (
    CIVILIZATION_REPLAY_SCHEMA_VERSION,
    SNAPSHOT_INTERVAL,
    TIMELINE_CHUNK_SIZE,
    validate_manifest,
)
from .serialization import sha256_value, write_json, write_json_gz

CIVILIZATION_REPLAY_ID = "civilization_vertical_slice_v1"
CIVILIZATION_SHOWCASE_SEED = 7


def record_civilization_vertical_slice(
    output_root: str | Path = "runs/replays",
    *,
    seed: int = CIVILIZATION_SHOWCASE_SEED,
    overwrite: bool = False,
) -> Path:
    """Run the public-action script and atomically write a Replay 2.1 artifact."""

    output_root = Path(output_root).resolve()
    target = output_root / CIVILIZATION_REPLAY_ID
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay already exists at {target}.")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{CIVILIZATION_REPLAY_ID}.", dir=output_root))
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
    env = VoyagerCivilizationEnv()
    env.reset(seed=seed)
    controller = CivilizationScriptedController()
    state = env.world.state
    if state is None:
        raise RuntimeError("Civilization state was not initialized.")

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
    metric_series: dict[str, list[float]] = defaultdict(list)
    invalid_actions = 0
    dense_return = 0.0

    while env.agents:
        actions = controller.act_many(env)
        _observations, rewards, _terms, _truncs, infos = env.step(actions)
        state = env.world.state
        assert state is not None
        world = _world_snapshot(env)
        state_hash = _state_hash(world)
        world["extensions"] = {**world.get("extensions", {}), "state_hash": state_hash}
        action_records = []
        for agent_id, action in sorted(actions.items()):
            info = infos[agent_id]
            invalid_actions += int(bool(info.get("invalid_action", False)))
            dense_return += float(rewards[agent_id])
            action_records.append(
                {
                    "agent_id": agent_id,
                    "role": state.agents[agent_id].role,
                    "selected_action": CivilizationVerb(action["verb"]).name.lower(),
                    "selected_action_id": action["verb"],
                    "selected_argument": CivilizationArgument(action["argument"]).name.lower(),
                    "selected_argument_id": action["argument"],
                    "invalid": bool(info.get("invalid_action", False)),
                    "reward": float(rewards[agent_id]),
                    "reward_components": dict(info.get("reward_components", {})),
                }
            )
        events = [dict(event) for event in state.events]
        all_events.extend(events)
        new_achievements = sorted(
            {
                achievement
                for info in infos.values()
                for achievement in info.get("new_achievements", ())
            }
        )
        record = {
            "tick": state.step_count,
            "actions": action_records,
            "events": events,
            "state_delta": _state_delta(previous, world),
            "achievements": new_achievements,
            "weather": {"kind": "clear", "storm_active": False},
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
        metric_series["tick"].append(float(state.step_count))
        metric_series["survivors"].append(float(len(env.agents)))
        metric_series["campfire_fuel"].append(float(state.structures["campfire"].fuel))
        metric_series["shelter_occupancy"].append(
            float(len(state.structures["shelter"].occupants))
        )
        metric_series["monsters_alive"].append(
            float(
                sum(
                    creature.alive and creature.type == "night_stalker"
                    for creature in state.creatures.values()
                )
            )
        )

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
    write_json_gz(directory / "metrics.json.gz", {"global": metric_series, "extensions": {}})
    write_json(
        directory / "camera.json",
        generate_camera_cues(
            all_events,
            terminal_tick=state.step_count,
            camp=initial["camp"],
            agents=initial["agents"],
        ),
    )

    shelter = state.structures["shelter"]
    terminal_summary = {
        "world_steps": state.step_count,
        "agent_steps": sum(len(record["actions"]) for record in timeline),
        "survivors": sum(agent.alive for agent in state.agents.values()),
        "deaths": state.deaths,
        "dense_return": dense_return,
        "achievement_return": float(len(state.achievements)),
        "achievements": sorted(state.achievements),
        "achievement_steps": dict(state.achievement_steps),
        "camp_stockpile": dict(state.camp.stockpile),
        "shelter_progress": shelter.progress,
        "shelter_completion_step": state.shelter_completion_step,
        "resource_flow": {
            "gathered": state.gathered_resources,
            "deposited": state.deposited_resources,
            "consumed": state.consumed_resources,
            "constructed": state.constructed_resources,
        },
        "structures": {key: value.complete for key, value in state.structures.items()},
        "hunts": state.hunts,
        "cooked_meals": state.cooked_meals,
        "monster_defeats": state.monster_defeats,
        "prevented_damage": state.prevented_damage,
        "invalid_scripted_actions": invalid_actions,
        "state_hash": _state_hash(previous),
    }
    registries = _registries(initial)
    registries.update(
        {
            "verbs": [
                {"id": value.name.lower(), "value": int(value), "label": value.name.title()}
                for value in CivilizationVerb
            ],
            "arguments": [
                {"id": value.name.lower(), "value": int(value), "label": value.name.title()}
                for value in CivilizationArgument
            ],
            "creatures": [
                {"id": "island_deer", "label": "Island Deer"},
                {"id": "night_stalker", "label": "Night Stalker"},
            ],
        }
    )
    payload: dict[str, Any] = {
        "replay_id": CIVILIZATION_REPLAY_ID,
        "versions": {
            "replay": CIVILIZATION_REPLAY_SCHEMA_VERSION,
            "environment": "voyager_civilization_v1",
            "scenario": state.scenario_id,
            "reward": "civilization_reward_v1",
            "observation": "civilization_local_observation_v1",
            "action": "civilization_structured_action_v1",
            "achievement": "civilization_achievements_v1",
        },
        "status": "complete",
        "source": {
            "policy_kind": "scripted",
            "policy_id": controller.policy_id,
            "inference_mode": "privileged_deterministic",
            "evaluation_seed": seed,
            "source_fingerprint": sha256_value(
                {"policy": controller.policy_id, "seed": seed, "version": "stage7a_v1"}
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
        "tags": ["stage7a", "showcase", "scripted", "privileged"],
        "terminal_summary": terminal_summary,
        "registries": registries,
        "artifacts": _artifact_index(directory),
        "extensions": {"privileged_policy": True, "playback_seconds": 300},
    }
    provisional = validate_manifest(payload)
    normalized = provisional.model_dump(mode="json")
    normalized.pop("manifest_sha256", None)
    normalized["manifest_sha256"] = sha256_value(normalized)
    write_json(directory / "manifest.json", validate_manifest(normalized).model_dump(mode="json"))


def _state_hash(world: dict[str, Any]) -> str:
    value = dict(world)
    value.pop("extensions", None)
    value["resources"] = sorted(value["resources"], key=lambda item: item["id"])
    return sha256_value(value)
