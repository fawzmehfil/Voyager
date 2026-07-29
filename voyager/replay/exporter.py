"""Export the locked Stage 6A showcase replay from the frozen Stage 5.6 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from voyager.benchmark.runner import _validate_manifest, load_manifest
from voyager.benchmark.schema import BenchmarkManifest, PolicySpec
from voyager.envs import VoyagerParallelEnv
from voyager.policies.ppo_policy import TensorFlowPPOPolicy
from voyager.sim.achievements import ACHIEVEMENT_IDS
from voyager.sim.constants import Action, Resource, Terrain

REPLAY_SCHEMA_VERSION = "stage6_replay_1.0.0"
LOCKED_POLICY_ID = "ppo_seed0_deterministic"
LOCKED_EVALUATION_SEED = 10_000_010
LOCKED_TICK_RATE = 12
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks/manifests/stage5_6_final.json"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "web/public/replays/stage6_vertical_slice_v1.json"

AGENT_IDENTITIES = (
    {"name": "Miro", "skin": "umber", "hair": "curl", "accent": "fern", "accessory": "satchel"},
    {"name": "Luma", "skin": "copper", "hair": "bob", "accent": "coral", "accessory": "flower"},
    {"name": "Koa", "skin": "deep", "hair": "crop", "accent": "lagoon", "accessory": "bandana"},
    {"name": "Nia", "skin": "golden", "hair": "puffs", "accent": "mango", "accessory": "basket"},
    {"name": "Tavi", "skin": "umber", "hair": "swoop", "accent": "sky", "accessory": "shell"},
    {"name": "Suri", "skin": "deep", "hair": "braid", "accent": "hibiscus", "accessory": "headwrap"},
    {"name": "Beni", "skin": "golden", "hair": "tuft", "accent": "lime", "accessory": "pouch"},
    {"name": "Iko", "skin": "copper", "hair": "mohawk", "accent": "violet", "accessory": "feather"},
    {"name": "Yara", "skin": "deep", "hair": "buns", "accent": "sun", "accessory": "necklace"},
    {"name": "Orin", "skin": "golden", "hair": "waves", "accent": "mint", "accessory": "leaf"},
)

LOCKED_EXPECTATION: dict[str, object] = {
    "world_steps": 300,
    "agent_steps": 3000,
    "survivors": 10,
    "deaths": 0,
    "dense_return": 171.69393333332465,
    "shelter_completion_step": 115,
    "camp_stockpile": {"food": 27, "wood": 0, "stone": 0},
    "achievement_steps": {
        "all_active_agents_alive_100": 100,
        "all_roles_contributed": 112,
        "camp_food_buffer_10": 121,
        "camp_food_buffer_20": 235,
        "first_deposit": 31,
        "first_food_gathered": 3,
        "first_food_withdrawal": 52,
        "first_stone_gathered": 6,
        "first_storm_survived": 225,
        "first_wood_gathered": 1,
        "food_security_100_steps": 220,
        "no_deaths_run": 300,
        "shared_food_transfer": 257,
        "shelter_25_percent": 30,
        "shelter_50_percent": 42,
        "shelter_complete": 115,
    },
}

TERRAIN_NAMES = {
    Terrain.WATER: "water",
    Terrain.BEACH: "beach",
    Terrain.GRASS: "grass",
    Terrain.FOREST: "forest",
    Terrain.QUARRY: "quarry",
}
RESOURCE_NAMES = {
    Resource.NONE: "none",
    Resource.FOOD: "food",
    Resource.WOOD: "wood",
    Resource.STONE: "stone",
}


def export_vertical_slice(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Rerun and export the one locked Stage 6A benchmark episode."""

    manifest_file = Path(manifest_path).resolve()
    repository_root = _repository_root(manifest_file)
    manifest = load_manifest(manifest_file)
    _validate_manifest(manifest, repository_root)
    policy_spec = _locked_policy(manifest)
    if LOCKED_EVALUATION_SEED not in manifest.seed_suite.seeds:
        raise ValueError("Locked replay seed is absent from the final benchmark manifest.")
    if policy_spec.checkpoint is None or policy_spec.checkpoint_sha256 is None:
        raise ValueError("Locked PPO policy is missing checkpoint metadata.")

    config = manifest.scenario.config
    env = _create_environment(config)
    observations, infos = env.reset(seed=LOCKED_EVALUATION_SEED)
    state = _state(env)
    initial_resource_ids = state.resource_ids.copy()
    initial_resource_quantities = state.resource_quantities.copy()
    initial = {
        "terrain": [
            [TERRAIN_NAMES[Terrain(int(value))] for value in row]
            for row in state.terrain
        ],
        "resources": _resource_list(
            state.resource_ids,
            state.resource_quantities,
        ),
        "camp": _camp_snapshot(env),
        "agents": _agent_snapshots(env, {}, {}),
    }

    checkpoint_path = repository_root / policy_spec.checkpoint
    policy = TensorFlowPPOPolicy(
        checkpoint_path,
        deterministic=True,
        seed=LOCKED_EVALUATION_SEED,
    )
    policy.reset(LOCKED_EVALUATION_SEED)
    frames: list[dict[str, object]] = []
    dense_return = 0.0
    agent_steps = 0
    previous_resource_ids = initial_resource_ids
    previous_resource_quantities = initial_resource_quantities

    while env.agents:
        agent_ids = tuple(env.agents)
        decisions = policy.decide_many(agent_ids, observations, infos)
        actions = {
            agent_id: int(decisions[agent_id].action)
            for agent_id in agent_ids
        }
        observations, _rewards, _terms, _truncs, step_infos = env.step(actions)
        agent_steps += len(agent_ids)
        for agent_id in agent_ids:
            dense_components = step_infos[agent_id]["dense_reward_components"]
            dense_return += sum(float(value) for value in dense_components.values())

        state = _state(env)
        frames.append(
            {
                "step": state.step_count,
                "storm": env.world.is_storm_active(),
                "camp": _camp_snapshot(env),
                "agents": _agent_snapshots(env, actions, step_infos),
                "resource_changes": _resource_changes(
                    previous_resource_ids,
                    previous_resource_quantities,
                    state.resource_ids,
                    state.resource_quantities,
                ),
                "new_achievements": _new_achievements(step_infos),
            }
        )
        previous_resource_ids = state.resource_ids.copy()
        previous_resource_quantities = state.resource_quantities.copy()
        infos.update(step_infos)

    metrics = env.metrics()
    summary = {
        "world_steps": int(cast(int, metrics["step"])),
        "agent_steps": agent_steps,
        "dense_return": dense_return,
        "survivors": int(cast(int, metrics["active_agents"])),
        "deaths": int(cast(int, metrics["deaths"])),
        "shelter_completion_step": metrics["shelter_completion_step"],
        "camp_stockpile": dict(
            cast(dict[str, int], _dict(metrics["camp"])["stockpile"])
        ),
        "achievements": list(cast(list[str], metrics["achievements"])),
        "achievement_steps": dict(
            cast(dict[str, int], metrics["achievement_steps"])
        ),
        "resource_flow": dict(cast(dict[str, Any], metrics["resource_flow"])),
    }
    _verify_locked_outcome(summary)

    payload: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_id": "stage6_vertical_slice_v1",
        "tick_rate": LOCKED_TICK_RATE,
        "duration_seconds": len(frames) / LOCKED_TICK_RATE,
        "source": {
            "benchmark_id": manifest.benchmark_id,
            "scenario_id": manifest.scenario.id,
            "policy_id": policy_spec.id,
            "policy_kind": policy_spec.kind,
            "inference_mode": policy_spec.inference_mode,
            "training_seed": policy_spec.training_seed,
            "evaluation_seed": LOCKED_EVALUATION_SEED,
            "checkpoint": policy_spec.checkpoint,
            "checkpoint_sha256": policy_spec.checkpoint_sha256,
            "manifest_sha256": _sha256(manifest_file),
        },
        "world": {
            "width": int(config["map_size"]),
            "height": int(config["map_size"]),
            "initial": initial,
        },
        "frames": frames,
        "summary": summary,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _create_environment(config: dict[str, int | float]) -> VoyagerParallelEnv:
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


def _locked_policy(manifest: BenchmarkManifest) -> PolicySpec:
    for policy in manifest.policies:
        if policy.id == LOCKED_POLICY_ID:
            if policy.inference_mode != "deterministic":
                raise ValueError("The locked replay requires deterministic inference.")
            return policy
    raise ValueError(f"Policy {LOCKED_POLICY_ID!r} is absent from the manifest.")


def _state(env: VoyagerParallelEnv) -> Any:
    state = env.world.state
    if state is None:
        raise RuntimeError("Replay environment state is unavailable.")
    return state


def _camp_snapshot(env: VoyagerParallelEnv) -> dict[str, object]:
    camp = _state(env).camp
    return {
        "x": camp.x,
        "y": camp.y,
        "stockpile": dict(camp.stockpile),
        "shelter_progress": round(float(camp.shelter_progress), 6),
    }


def _agent_snapshots(
    env: VoyagerParallelEnv,
    actions: dict[str, int],
    infos: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    state = _state(env)
    snapshots: list[dict[str, object]] = []
    for index, agent_id in enumerate(env.possible_agents):
        agent = state.agents[agent_id]
        identity = AGENT_IDENTITIES[index]
        action = actions.get(agent_id)
        info = infos.get(agent_id, {})
        snapshots.append(
            {
                "id": agent_id,
                "name": identity["name"],
                "role": agent.role,
                "appearance": {
                    "skin": identity["skin"],
                    "hair": identity["hair"],
                    "accent": identity["accent"],
                    "accessory": identity["accessory"],
                    "variant": index,
                },
                "x": agent.x,
                "y": agent.y,
                "health": round(float(agent.health), 4),
                "hunger": round(float(agent.hunger), 4),
                "energy": round(float(agent.energy), 4),
                "alive": agent.alive,
                "inventory": dict(agent.inventory),
                "action": Action(action).name.lower() if action is not None else "noop",
                "event": str(info.get("event", "reset")),
            }
        )
    return snapshots


def _resource_list(
    resource_ids: np.ndarray,
    resource_quantities: np.ndarray,
) -> list[dict[str, object]]:
    resources: list[dict[str, object]] = []
    for y, x in np.argwhere(resource_quantities > 0):
        resource = Resource(int(resource_ids[y, x]))
        resources.append(
            {
                "x": int(x),
                "y": int(y),
                "type": RESOURCE_NAMES[resource],
                "quantity": int(resource_quantities[y, x]),
            }
        )
    return resources


def _resource_changes(
    previous_ids: np.ndarray,
    previous_quantities: np.ndarray,
    current_ids: np.ndarray,
    current_quantities: np.ndarray,
) -> list[dict[str, object]]:
    changed = (previous_ids != current_ids) | (
        previous_quantities != current_quantities
    )
    changes: list[dict[str, object]] = []
    for y, x in np.argwhere(changed):
        resource = Resource(int(current_ids[y, x]))
        changes.append(
            {
                "x": int(x),
                "y": int(y),
                "type": RESOURCE_NAMES[resource],
                "quantity": int(current_quantities[y, x]),
            }
        )
    return changes


def _new_achievements(infos: dict[str, dict[str, Any]]) -> list[str]:
    if not infos:
        return []
    first = next(iter(infos.values()))
    values = first.get("new_achievements", [])
    if not isinstance(values, list):
        raise TypeError("new_achievements must be a list.")
    return [str(value) for value in values]


def _verify_locked_outcome(summary: dict[str, object]) -> None:
    for key, expected in LOCKED_EXPECTATION.items():
        actual = summary[key]
        if key == "dense_return":
            actual_return = cast(float, actual)
            expected_return = cast(float, expected)
            if not np.isclose(
                actual_return,
                expected_return,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Locked replay mismatch for {key}: {actual!r} != {expected!r}."
                )
        elif actual != expected:
            raise ValueError(
                f"Locked replay mismatch for {key}: {actual!r} != {expected!r}."
            )
    achievements = cast(list[str], summary["achievements"])
    if set(achievements) != set(ACHIEVEMENT_IDS):
        raise ValueError("Locked replay did not unlock all 16 achievements.")


def _repository_root(manifest_path: Path) -> Path:
    for parent in (manifest_path.parent, *manifest_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate the Voyager repository root.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected dictionary.")
    return value
