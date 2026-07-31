from __future__ import annotations

from pathlib import Path

import numpy as np

from voyager.envs.civilization import (
    CivilizationFlattenedActionWrapper,
    VoyagerCivilizationEnv,
)
from voyager.policies.civilization_scripted import CivilizationScriptedController
from voyager.replay.civilization import record_civilization_vertical_slice
from voyager.replay.loader import ReplayLoader
from voyager.sim.registries import (
    MEANINGFUL_ACTION_PAIRS,
    CivilizationAction,
    CivilizationArgument,
    CivilizationVerb,
    flatten_action,
    unflatten_action,
)
from voyager.sim.scenarios import build_civilization_island, validate_civilization_map


def _noop_actions(env: VoyagerCivilizationEnv) -> dict[str, dict[str, int]]:
    return {
        agent_id: {
            "verb": int(CivilizationVerb.NOOP),
            "argument": int(CivilizationArgument.NONE),
        }
        for agent_id in env.agents
    }


def test_handcrafted_island_is_fixed_reachable_and_48_square() -> None:
    first = build_civilization_island()
    second = build_civilization_island()
    validate_civilization_map(
        first.terrain,
        first.resource_ids,
        first.resource_quantities,
    )
    assert first.terrain.shape == (48, 48)
    assert np.array_equal(first.terrain, second.terrain)
    assert np.array_equal(first.resource_quantities, second.resource_quantities)


def test_structured_and_flat_actions_round_trip() -> None:
    for verb, argument in MEANINGFUL_ACTION_PAIRS:
        structured = CivilizationAction(verb, argument)
        assert unflatten_action(flatten_action(int(verb), int(argument))) == (
            int(structured.verb),
            int(structured.argument),
        )


def test_civilization_observations_masks_and_wrapper_are_stable() -> None:
    env = VoyagerCivilizationEnv()
    observations, _infos = env.reset(seed=7)
    for agent_id, observation in observations.items():
        assert env.observation_space(agent_id).contains(observation)
        assert observation["action_mask"].shape == env.action_mask(agent_id).shape
    wrapped = CivilizationFlattenedActionWrapper(VoyagerCivilizationEnv())
    wrapped_observations, _ = wrapped.reset(seed=7)
    assert wrapped.action_space("agent_0").n == len(MEANINGFUL_ACTION_PAIRS)
    assert wrapped_observations["agent_0"]["action_mask"].shape == (
        len(MEANINGFUL_ACTION_PAIRS),
    )


def test_each_night_seededly_samples_one_or_two_stalkers() -> None:
    counts: set[int] = set()
    for seed in range(12):
        env = VoyagerCivilizationEnv()
        env.reset(seed=seed)
        assert env.world.state is not None
        env.world.state.step_count = 199
        env.step(_noop_actions(env))
        counts.add(env.world.state.last_spawn_count)
        positions = env.world.state.last_spawn_positions
        assert len(positions) == len(set(positions)) == env.world.state.last_spawn_count
    assert counts == {1, 2}


def test_unmitigated_stalker_attack_deals_exactly_25_damage() -> None:
    env = VoyagerCivilizationEnv()
    env.reset(seed=7)
    state = env.world.state
    assert state is not None
    state.step_count = 199
    env.step(_noop_actions(env))
    stalker = next(value for value in state.creatures.values() if value.type == "night_stalker")
    target = state.agents["agent_9"]
    target.x, target.y = stalker.x + 1, stalker.y
    for agent_id, agent in state.agents.items():
        if agent_id != "agent_9":
            agent.sheltered = True
    before = target.health
    env.step(_noop_actions(env))
    assert target.health == before - 25


def test_showcase_script_reaches_the_complete_public_progression() -> None:
    env = VoyagerCivilizationEnv()
    env.reset(seed=7)
    controller = CivilizationScriptedController()
    invalid = 0
    while env.agents:
        _obs, _rewards, _terms, _truncs, infos = env.step(controller.act_many(env))
        invalid += sum(int(info["invalid_action"]) for info in infos.values())
    state = env.world.state
    assert state is not None
    assert state.step_count == 600
    assert all(structure.complete for structure in state.structures.values())
    assert state.hunts >= 1
    assert state.cooked_meals >= 1
    assert state.monster_defeats >= 1
    assert state.prevented_damage > 0
    assert invalid == 0
    assert {
        "first_spear_crafted",
        "first_successful_hunt",
        "first_cooked_meal",
        "campfire_full_night",
        "full_shelter_protected_night",
        "joint_construction_multiple_roles",
        "first_ally_defense_kill",
    } <= state.achievements


def test_replay_21_deeply_reconstructs_vertical_slice(tmp_path: Path) -> None:
    replay_path = record_civilization_vertical_slice(tmp_path)
    loader = ReplayLoader(replay_path)
    result = loader.validate(deep=True)
    assert result["checked_ticks"] == 601
    assert loader.manifest.versions.replay == "stage7_replay_2.1.0"
    assert loader.manifest.tick_rate == 2
    assert loader.manifest.world_steps == 600
    assert loader.manifest.terminal_summary["invalid_scripted_actions"] == 0
    assert loader.state_at(200)["time"]["phase"] == "night"
    assert loader.events(event_type="stalkers_spawned")
