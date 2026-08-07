"""Acceptance coverage for the reduced VoyagerIsland-v1 benchmark."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from voyager.benchmark.island import ISLAND_DEV_SEEDS, ISLAND_TEST_SEEDS, ISLAND_TRAIN_SEEDS
from voyager.envs.island import ISLAND_REWARD_VERSION, VoyagerIslandCentralizedEnv, VoyagerIslandEnv
from voyager.replay.island import record_island_oracle_replay
from voyager.replay.loader import ReplayLoader
from voyager.sim.constants import Resource
from voyager.sim.island_achievements import geometric_mean_score
from voyager.sim.island_registry import ISLAND_ACTION_COUNT, IslandAction
from voyager.sim.scenarios import (
    ISLAND_BENCHMARK_CAMP,
    ISLAND_BENCHMARK_STRUCTURE_SPECS,
    ISLAND_BENCHMARK_TOOL_RECIPES,
    build_island_benchmark,
)
from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT, make_training_environment
from voyager.training.island_evaluation import (
    island_checkpoint_selection_key,
    normalize_island_evaluation_milestones,
    scripted_oracle_solvability_gate,
)
from voyager.training.island_reward import (
    CAUSAL_ACHIEVEMENT_REWARD,
    ISLAND_TRAINING_REWARD_V2,
    ISLAND_TRAINING_REWARD_V3,
    ISLAND_TRAINING_REWARD_V4,
    RETURN_MILESTONE_REWARD,
    IslandTrainingRewardV2Wrapper,
    IslandTrainingRewardV3Wrapper,
    IslandTrainingRewardV4Wrapper,
)
from voyager.training.obs import ISLAND_V1_OBSERVATION_ENCODER, flatten_observation


def _hash(env: VoyagerIslandEnv) -> str:
    payload = env.global_state()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _noop(env: VoyagerIslandEnv) -> dict[str, int]:
    return {agent: int(IslandAction.NOOP) for agent in env.agents}


def test_public_contract_and_observation_are_compact() -> None:
    env = VoyagerIslandEnv(procedural=False)
    observations, infos = env.reset(seed=0)

    assert env.possible_agents == ["agent_0", "agent_1"]
    assert env.action_space("agent_0").n == ISLAND_ACTION_COUNT == 19
    assert env.max_steps == 1_200
    assert env.map_size == 48
    assert env.observation_space("agent_0").contains(observations["agent_0"])
    assert flatten_observation(observations["agent_0"], ISLAND_V1_OBSERVATION_ENCODER).shape == (
        373,
    )
    assert infos["agent_0"]["scenario_id"] == "voyager_island_benchmark_v1"
    assert "entity_slots" not in observations["agent_0"]


def test_procedural_maps_are_seeded_valid_and_distinct() -> None:
    first = build_island_benchmark(np.random.default_rng(17), procedural=True)
    repeated = build_island_benchmark(np.random.default_rng(17), procedural=True)
    different = build_island_benchmark(np.random.default_rng(18), procedural=True)

    assert np.array_equal(first.state.terrain, repeated.state.terrain)
    assert np.array_equal(first.state.resource_ids, repeated.state.resource_ids)
    assert np.array_equal(first.state.resource_quantities, repeated.state.resource_quantities)
    assert first.deer_spawns == repeated.deer_spawns
    assert first.stalker_spawns == repeated.stalker_spawns
    assert not np.array_equal(first.state.resource_ids, different.state.resource_ids)
    totals = {
        resource: int(np.sum(first.state.resource_quantities[first.state.resource_ids == resource]))
        for resource in (Resource.FOOD, Resource.WOOD, Resource.STONE)
    }
    assert totals == {Resource.FOOD: 20, Resource.WOOD: 35, Resource.STONE: 20}


def test_submission_order_is_byte_deterministic() -> None:
    left = VoyagerIslandEnv(procedural=True)
    right = VoyagerIslandEnv(procedural=True)
    left.reset(seed=91)
    right.reset(seed=91)

    for _ in range(30):
        left.step({"agent_0": 0, "agent_1": 0})
        right.step({"agent_1": 0, "agent_0": 0})
    assert _hash(left) == _hash(right)


def test_symmetric_collisions_fail_and_swaps_succeed() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    state.agents["agent_0"].x, state.agents["agent_0"].y = 20, 20
    state.agents["agent_1"].x, state.agents["agent_1"].y = 22, 20
    _obs, _rewards, _terms, _truncs, infos = env.step(
        {
            "agent_0": int(IslandAction.MOVE_EAST),
            "agent_1": int(IslandAction.MOVE_WEST),
        }
    )
    assert (state.agents["agent_0"].x, state.agents["agent_1"].x) == (20, 22)
    assert infos["agent_0"]["invalid_action"]
    assert infos["agent_1"]["invalid_action"]

    state.agents["agent_1"].x = 21
    env.step(
        {
            "agent_0": int(IslandAction.MOVE_EAST),
            "agent_1": int(IslandAction.MOVE_WEST),
        }
    )
    assert (state.agents["agent_0"].x, state.agents["agent_1"].x) == (21, 20)


def test_deposit_construction_crafting_and_passive_axe() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    for agent in state.agents.values():
        agent.x, agent.y = ISLAND_BENCHMARK_CAMP
        agent.inventory["wood"] = 2
        agent.inventory["stone"] = 1
    state.resource_quantities[31, 18] -= 4
    state.resource_quantities[31, 34] -= 2
    env.step({agent: int(IslandAction.DEPOSIT_ALL) for agent in env.agents})
    assert state.camp.stockpile["wood"] == 4
    assert state.camp.stockpile["stone"] == 2

    for _ in range(2):
        env.step({agent: int(IslandAction.WORK_WORKBENCH) for agent in env.agents})
    assert state.structures["workbench"].complete
    assert "build_workbench" in state.achievements

    state.resource_quantities[31, 18] -= 1
    state.camp.stockpile.update({"wood": 2, "stone": 1})
    env.step(
        {
            "agent_0": int(IslandAction.CRAFT_AXE),
            "agent_1": int(IslandAction.NOOP),
        }
    )
    assert "axe" in state.agents["agent_0"].tools
    agent = state.agents["agent_0"]
    agent.x, agent.y = 18, 31
    before = int(state.resource_quantities[31, 18])
    env.step({"agent_0": int(IslandAction.INTERACT), "agent_1": int(IslandAction.NOOP)})
    assert state.resource_quantities[31, 18] == before - 2
    assert env.world.reconcile_v2_ledger() == {}


def test_progression_recipes_and_masks_expose_one_coherent_branch() -> None:
    assert ISLAND_BENCHMARK_STRUCTURE_SPECS == {
        "workbench": ({"wood": 3, "stone": 1}, 20, 0),
        "campfire": ({"wood": 2, "stone": 1}, 20, 0),
        "shelter": ({"wood": 4, "stone": 2}, 40, 2),
        "beacon": ({"wood": 4, "stone": 2}, 60, 0),
    }
    assert ISLAND_BENCHMARK_TOOL_RECIPES == {
        "axe": {"wood": 2, "stone": 1},
        "spear": {"wood": 2, "stone": 1},
    }

    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    state.camp.stockpile.update({"wood": 100, "stone": 100})
    agent = state.agents["agent_0"]

    agent.x, agent.y = state.structures["workbench"].x, state.structures["workbench"].y
    mask = env.action_mask("agent_0")
    assert mask[int(IslandAction.WORK_WORKBENCH)] == 1
    assert mask[int(IslandAction.WORK_CAMPFIRE)] == 0
    assert mask[int(IslandAction.WORK_SHELTER)] == 0
    assert mask[int(IslandAction.WORK_BEACON)] == 0

    workbench = state.structures["workbench"]
    workbench.reserved_materials = dict(workbench.required_materials)
    workbench.labor = workbench.required_labor
    mask = env.action_mask("agent_0")
    assert mask[int(IslandAction.CRAFT_AXE)] == 1
    assert mask[int(IslandAction.WORK_CAMPFIRE)] == 0

    state.agents["agent_0"].tools.add("axe")
    state.agents["agent_1"].tools.add("spear")
    agent.x, agent.y = state.structures["campfire"].x, state.structures["campfire"].y
    mask = env.action_mask("agent_0")
    assert mask[int(IslandAction.WORK_CAMPFIRE)] == 1
    assert mask[int(IslandAction.WORK_SHELTER)] == 0

    campfire = state.structures["campfire"]
    campfire.reserved_materials = dict(campfire.required_materials)
    campfire.labor = campfire.required_labor
    agent.x, agent.y = state.structures["shelter"].x, state.structures["shelter"].y
    assert env.action_mask("agent_0")[int(IslandAction.WORK_SHELTER)] == 0
    state.cooked_meals = 1
    assert env.action_mask("agent_0")[int(IslandAction.WORK_SHELTER)] == 1

    shelter = state.structures["shelter"]
    shelter.reserved_materials = dict(shelter.required_materials)
    shelter.labor = shelter.required_labor
    agent.x, agent.y = state.structures["beacon"].x, state.structures["beacon"].y
    mask = env.action_mask("agent_0")
    assert mask[int(IslandAction.WORK_SHELTER)] == 0
    assert mask[int(IslandAction.WORK_BEACON)] == 1


def test_deer_hunting_requires_the_spear() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    deer = next(creature for creature in state.creatures.values() if creature.type == "island_deer")
    agent = state.agents["agent_0"]
    agent.x, agent.y = deer.x - 1, deer.y
    assert env.action_mask("agent_0")[int(IslandAction.ATTACK)] == 0
    agent.tools.add("spear")
    assert env.action_mask("agent_0")[int(IslandAction.ATTACK)] == 1


def test_beacon_requires_prior_infrastructure_and_rescue_requires_both_agents() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    agent.x, agent.y = state.structures["beacon"].x, state.structures["beacon"].y
    state.camp.stockpile.update({"wood": 100, "stone": 100})
    assert env.action_mask("agent_0")[int(IslandAction.WORK_BEACON)] == 0
    state.camp.stockpile.update({"wood": 0, "stone": 0})
    for name in ("workbench", "campfire", "shelter", "beacon"):
        structure = state.structures[name]
        structure.reserved_materials = dict(structure.required_materials)
        structure.labor = structure.required_labor
    state.achievements.update(
        {
            "collect_food",
            "collect_wood",
            "collect_stone",
            "deposit_wood",
            "deposit_stone",
            "build_workbench",
            "craft_axe",
            "craft_spear",
            "hunt_deer",
            "build_campfire",
            "cook_meat",
            "build_shelter",
            "both_survive_first_night",
            "build_beacon",
        }
    )
    state.achievement_steps["build_beacon"] = 100
    state.step_count = 299
    for survivor in state.agents.values():
        survivor.health = 100
        survivor.hunger = 0
    _observations, _rewards, terminations, truncations, _infos = env.step(_noop(env))
    assert state.rescue_success
    assert "rescue_both" in state.achievements
    assert terminations == {"agent_0": True, "agent_1": True}
    assert truncations == {"agent_0": False, "agent_1": False}
    assert env.agents == []


def test_night_spawns_one_or_two_dangerous_stalkers() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=4)
    state = env.world.state
    assert state is not None
    state.step_count = 199
    env.step(_noop(env))
    stalkers = [
        creature
        for creature in state.creatures.values()
        if creature.alive and creature.type == "night_stalker"
    ]
    assert len(stalkers) in {1, 2}
    assert all(creature.health == 3 for creature in stalkers)


def test_completed_campfire_cooks_raw_meat_from_shared_camp_storage() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    campfire = state.structures["campfire"]
    campfire.reserved_materials = dict(campfire.required_materials)
    campfire.labor = campfire.required_labor
    agent.x, agent.y = campfire.x, campfire.y
    raw = agent.food_lots[0]
    raw.kind = "raw_meat"
    raw.origin_type = "animal"
    raw.origin_id = "island-deer-test"
    state.camp.food_lots.append(agent.food_lots.pop())
    initial_balance = state.ledger[0]["balance"]
    initial_balance["food"] -= 1
    initial_balance["raw_meat"] = initial_balance.get("raw_meat", 0) + 1

    assert env.action_mask("agent_0")[int(IslandAction.USE_CAMPFIRE)] == 1
    env.step(
        {
            "agent_0": int(IslandAction.USE_CAMPFIRE),
            "agent_1": int(IslandAction.NOOP),
        }
    )

    assert not any(lot.kind == "raw_meat" for lot in state.camp.food_lots)
    assert any(lot.kind == "cooked_meat" for lot in state.camp.food_lots)
    assert state.cooked_meals == 1


def test_score_is_independent_and_penalizes_missing_achievements() -> None:
    empty = geometric_mean_score({})
    partial = geometric_mean_score({"collect_food": 1.0})
    complete = geometric_mean_score(
        {
            name: 1.0
            for name in (
                "collect_food",
                "collect_wood",
                "collect_stone",
                "deposit_wood",
                "deposit_stone",
                "build_workbench",
                "craft_axe",
                "craft_spear",
                "hunt_deer",
                "build_campfire",
                "cook_meat",
                "build_shelter",
                "both_survive_first_night",
                "build_beacon",
                "rescue_both",
            )
        }
    )
    assert 0.0 <= empty < partial < complete <= 1.0


def test_scripted_oracle_gate_requires_every_achievement() -> None:
    summary: dict[str, object] = {
        "achievement_success_rates": {
            achievement: 0.90
            for achievement in (
                "collect_food",
                "collect_wood",
                "collect_stone",
                "deposit_wood",
                "deposit_stone",
                "build_workbench",
                "craft_axe",
                "craft_spear",
                "hunt_deer",
                "build_campfire",
                "cook_meat",
                "build_shelter",
                "both_survive_first_night",
                "build_beacon",
                "rescue_both",
            )
        },
        "rescue_rate": 0.90,
        "invalid_action_rate": 0.0,
    }
    assert scripted_oracle_solvability_gate(summary)["passed"] is True
    rates = summary["achievement_success_rates"]
    assert isinstance(rates, dict)
    rates["cook_meat"] = 0.89
    assert scripted_oracle_solvability_gate(summary)["passed"] is False


def test_checkpoint_selection_uses_score_then_lower_invalid_rate() -> None:
    weaker = {
        "achievement_geometric_mean": 0.10,
        "invalid_action_rate": 0.0,
    }
    stronger = {
        "achievement_geometric_mean": 0.11,
        "invalid_action_rate": 0.5,
    }
    tied_but_cleaner = {
        "achievement_geometric_mean": 0.11,
        "invalid_action_rate": 0.01,
    }

    assert island_checkpoint_selection_key(stronger) > island_checkpoint_selection_key(weaker)
    assert island_checkpoint_selection_key(tied_but_cleaner) > island_checkpoint_selection_key(
        stronger
    )
    with pytest.raises(ValueError, match="finite"):
        island_checkpoint_selection_key(
            {
                "achievement_geometric_mean": float("nan"),
                "invalid_action_rate": 0.0,
            }
        )


def test_checkpoint_milestones_are_unique_bounded_and_include_budget() -> None:
    assert normalize_island_evaluation_milestones(
        [100_000, -1, 50_000, 100_000, 999_999],
        total_agent_transitions=250_000,
    ) == (50_000, 100_000, 250_000)
    assert normalize_island_evaluation_milestones([], total_agent_transitions=17) == (17,)
    with pytest.raises(ValueError, match="positive"):
        normalize_island_evaluation_milestones([], total_agent_transitions=0)


def test_reward_is_one_time_shared_and_invalid_penalty_is_individual() -> None:
    env = VoyagerIslandEnv(procedural=False)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    state.agents["agent_0"].x, state.agents["agent_0"].y = 20, 31
    _obs, rewards, _terms, _truncs, infos = env.step(
        {"agent_0": int(IslandAction.INTERACT), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards == {"agent_0": 1.0, "agent_1": 1.0}
    assert infos["agent_0"]["new_achievements"] == ["collect_food"]
    _obs, rewards, _terms, _truncs, _infos = env.step(
        {"agent_0": int(IslandAction.INTERACT), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    _obs, rewards, _terms, _truncs, infos = env.step(
        {"agent_0": 999, "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards["agent_0"] == -0.05
    assert rewards["agent_1"] == 0.0
    assert infos["agent_0"]["invalid_action"]
    assert not infos["agent_1"]["invalid_action"]


def test_v2_reward_adds_symmetric_causal_credit_without_changing_v1() -> None:
    base = VoyagerIslandEnv(procedural=False)
    wrapped = IslandTrainingRewardV2Wrapper(base)
    _observations, infos = wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    for agent in state.agents.values():
        agent.x, agent.y = 18, 31

    _observations, rewards, _terminations, _truncations, infos = wrapped.step(
        {agent_id: int(IslandAction.INTERACT) for agent_id in wrapped.agents}
    )

    expected = 1.0 + CAUSAL_ACHIEVEMENT_REWARD + RETURN_MILESTONE_REWARD
    assert rewards == {"agent_0": expected, "agent_1": expected}
    for info in infos.values():
        assert info["reward_version"] == ISLAND_TRAINING_REWARD_V2
        assert info["individual_reward_components"]["causal_collect_wood"] == 0.5
        assert info["individual_reward_components"]["return_distance_6"] == 0.2

    _observations, rewards, _terminations, _truncations, infos = wrapped.step(
        {agent_id: int(IslandAction.INTERACT) for agent_id in wrapped.agents}
    )
    assert all("causal_collect_wood" not in info["reward_components"] for info in infos.values())
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}


def test_v2_return_milestones_are_individual_bounded_and_nonrepeatable() -> None:
    wrapped = IslandTrainingRewardV2Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    state.resource_quantities[31, 18] -= 1
    agent.inventory["wood"] = 1
    agent.x, agent.y = 17, 31

    _obs, rewards, _terms, _truncs, infos = wrapped.step(
        {"agent_0": int(IslandAction.MOVE_EAST), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards["agent_0"] == RETURN_MILESTONE_REWARD
    assert infos["agent_0"]["individual_reward_components"] == {
        "return_distance_6": RETURN_MILESTONE_REWARD
    }

    wrapped.step({"agent_0": int(IslandAction.MOVE_WEST), "agent_1": int(IslandAction.NOOP)})
    _obs, rewards, _terms, _truncs, infos = wrapped.step(
        {"agent_0": int(IslandAction.MOVE_EAST), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards["agent_0"] == 0.0
    assert "return_distance_6" not in infos["agent_0"]["reward_components"]

    agent.x, agent.y = 20, 31
    _obs, rewards, _terms, _truncs, _infos = wrapped.step(
        {"agent_0": int(IslandAction.MOVE_EAST), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards["agent_0"] == RETURN_MILESTONE_REWARD
    agent.x, agent.y = 22, 31
    _obs, rewards, _terms, _truncs, _infos = wrapped.step(
        {"agent_0": int(IslandAction.MOVE_EAST), "agent_1": int(IslandAction.NOOP)}
    )
    assert rewards["agent_0"] == RETURN_MILESTONE_REWARD
    assert sum(wrapped._return_milestones["agent_0"]) == 10


def test_v2_causal_attribution_uses_recorded_actors_and_material_ownership() -> None:
    wrapped = IslandTrainingRewardV2Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=2)
    state = wrapped.world.state
    assert state is not None
    state.events = [
        {"type": "gather", "actors": ["agent_0"], "payload": {"item": "stone"}},
        {"type": "deposit_all", "actors": ["agent_0", "agent_1"], "payload": {}},
        {
            "type": "structure_complete",
            "actors": ["agent_0", "agent_1"],
            "targets": ["workbench"],
            "payload": {},
        },
        {"type": "craft_axe", "actors": ["agent_0"], "payload": {}},
        {
            "type": "creature_defeated",
            "actors": ["agent_1"],
            "targets": ["deer_0"],
            "payload": {},
        },
        {"type": "cook_meat", "actors": ["agent_1"], "payload": {}},
    ]
    actors = wrapped._causal_actors(
        (
            "collect_stone",
            "deposit_wood",
            "build_workbench",
            "craft_axe",
            "hunt_deer",
            "cook_meat",
        ),
        material_before={
            "agent_0": {"wood": 2, "stone": 0},
            "agent_1": {"wood": 0, "stone": 2},
        },
    )
    assert actors == {
        "collect_stone": frozenset({"agent_0"}),
        "deposit_wood": frozenset({"agent_0"}),
        "build_workbench": frozenset({"agent_0", "agent_1"}),
        "craft_axe": frozenset({"agent_0"}),
        "hunt_deer": frozenset({"agent_1"}),
        "cook_meat": frozenset({"agent_1"}),
    }


def test_v2_spawn_assignment_is_seeded_balanced_and_does_not_change_base_env() -> None:
    base_positions: list[dict[str, tuple[int, int]]] = []
    swaps: list[bool] = []
    for seed in range(20):
        base = VoyagerIslandEnv(procedural=False)
        base.reset(seed=seed)
        state = base.world.state
        assert state is not None
        base_positions.append(
            {agent_id: (agent.x, agent.y) for agent_id, agent in state.agents.items()}
        )
        wrapped = IslandTrainingRewardV2Wrapper(VoyagerIslandEnv(procedural=False))
        _observations, infos = wrapped.reset(seed=seed)
        swaps.append(bool(infos["agent_0"]["spawn_assignment_swapped"]))
        _repeat_observations, repeat_infos = wrapped.reset(seed=seed)
        assert repeat_infos["agent_0"]["spawn_assignment_swapped"] == swaps[-1]
    assert all(positions == base_positions[0] for positions in base_positions)
    assert any(swaps) and not all(swaps)


def test_v3_stage_deposit_credit_is_proportional_bounded_and_nonrepeatable() -> None:
    wrapped = IslandTrainingRewardV3Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    for agent in state.agents.values():
        agent.x, agent.y = ISLAND_BENCHMARK_CAMP
        agent.inventory["wood"] = 2
    state.resource_quantities[31, 18] -= 4

    _obs, _rewards, _terms, _truncs, infos = wrapped.step(
        {agent_id: int(IslandAction.DEPOSIT_ALL) for agent_id in wrapped.agents}
    )
    for info in infos.values():
        assert info["reward_version"] == ISLAND_TRAINING_REWARD_V3
        assert info["individual_reward_components"]["stage_workbench_deposit_wood"] == (
            pytest.approx(0.15)
        )

    for agent in state.agents.values():
        agent.inventory["wood"] = 1
    state.resource_quantities[31, 18] -= 2
    _obs, _rewards, _terms, _truncs, infos = wrapped.step(
        {agent_id: int(IslandAction.DEPOSIT_ALL) for agent_id in wrapped.agents}
    )
    assert all(
        "stage_workbench_deposit_wood" not in info["individual_reward_components"]
        for info in infos.values()
    )
    assert wrapped.world.reconcile_v2_ledger() == {}


def test_v3_labor_credit_tracks_only_applied_progress() -> None:
    wrapped = IslandTrainingRewardV3Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    state.resource_quantities[31, 18] -= 3
    state.resource_quantities[31, 34] -= 1
    state.camp.stockpile.update({"wood": 3, "stone": 1})
    for agent in state.agents.values():
        agent.x, agent.y = ISLAND_BENCHMARK_CAMP

    _obs, _rewards, _terms, _truncs, infos = wrapped.step(
        {agent_id: int(IslandAction.WORK_WORKBENCH) for agent_id in wrapped.agents}
    )
    assert state.structures["workbench"].complete
    for info in infos.values():
        assert info["individual_reward_components"]["stage_workbench_labor"] == (
            pytest.approx(0.10)
        )
    _obs, _rewards, _terms, _truncs, infos = wrapped.step(
        {agent_id: int(IslandAction.WORK_WORKBENCH) for agent_id in wrapped.agents}
    )
    assert all(
        "stage_workbench_labor" not in info["individual_reward_components"]
        for info in infos.values()
    )


def test_v4_changes_only_bounded_stage_credit_scale() -> None:
    wrapped = IslandTrainingRewardV4Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    for agent in state.agents.values():
        agent.x, agent.y = ISLAND_BENCHMARK_CAMP
        agent.inventory["wood"] = 2
    state.resource_quantities[31, 18] -= 4

    _obs, _rewards, _terms, _truncs, infos = wrapped.step(
        {agent_id: int(IslandAction.DEPOSIT_ALL) for agent_id in wrapped.agents}
    )
    for info in infos.values():
        assert info["reward_version"] == ISLAND_TRAINING_REWARD_V4
        assert info["individual_reward_components"]["stage_workbench_deposit_wood"] == (
            pytest.approx(0.75)
        )


def test_v4_extraction_milestones_are_shared_and_nonrepeatable() -> None:
    wrapped = IslandTrainingRewardV4Wrapper(VoyagerIslandEnv(procedural=False))
    wrapped.reset(seed=0)
    state = wrapped.world.state
    assert state is not None
    for structure in state.structures.values():
        structure.reserved_materials = dict(structure.required_materials)
        structure.labor = structure.required_labor
    state.agents["agent_0"].tools.add("axe")
    state.agents["agent_1"].tools.add("spear")
    initial_balance = state.ledger[0]["balance"]
    initial_balance["axe"] = 1
    initial_balance["spear"] = 1
    state.cooked_meals = 1
    state.achievements.update(
        {
            "build_workbench",
            "craft_axe",
            "craft_spear",
            "build_campfire",
            "cook_meat",
            "build_shelter",
            "build_beacon",
        }
    )
    state.achievement_steps["build_beacon"] = 0

    infos: dict[str, dict[str, object]] = {}
    for _ in range(25):
        _obs, rewards, _terms, _truncs, infos = wrapped.step(_noop(wrapped.env))
    assert rewards == {"agent_0": 0.25, "agent_1": 0.25}
    assert all(
        info["shared_reward_components"]["shared_extraction_25"] == 0.25 for info in infos.values()
    )

    _obs, rewards, _terms, _truncs, infos = wrapped.step(_noop(wrapped.env))
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert all(
        "shared_extraction_25" not in info["shared_reward_components"] for info in infos.values()
    )


def test_seed_manifests_and_training_adapter_are_frozen() -> None:
    assert len(ISLAND_TRAIN_SEEDS) == 1_000
    assert len(ISLAND_DEV_SEEDS) == 50
    assert len(ISLAND_TEST_SEEDS) == 100
    assert not set(ISLAND_TRAIN_SEEDS) & set(ISLAND_DEV_SEEDS)
    assert not set(ISLAND_TRAIN_SEEDS) & set(ISLAND_TEST_SEEDS)
    assert not set(ISLAND_DEV_SEEDS) & set(ISLAND_TEST_SEEDS)
    training = make_training_environment(
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_REWARD_VERSION,
        num_agents=99,
        map_size=9,
        max_steps=2,
        reward_mode="dense",
        disabled_reward_components=(),
        mask_role_observation=False,
        procedural=False,
    )
    assert (training.num_agents, training.map_size, training.max_steps) == (2, 48, 1_200)
    assert training.observation_encoder == ISLAND_V1_OBSERVATION_ENCODER
    training.env.close()

    remediated = make_training_environment(
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V2,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        reward_mode="dense",
        disabled_reward_components=(),
        mask_role_observation=False,
        procedural=False,
    )
    assert isinstance(remediated.env, IslandTrainingRewardV2Wrapper)
    assert remediated.versions["reward_version"] == ISLAND_TRAINING_REWARD_V2
    remediated.env.close()

    progression = make_training_environment(
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V3,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        reward_mode="dense",
        disabled_reward_components=(),
        mask_role_observation=False,
        procedural=False,
    )
    assert isinstance(progression.env, IslandTrainingRewardV3Wrapper)
    assert progression.versions["reward_version"] == ISLAND_TRAINING_REWARD_V3
    progression.env.close()

    salient_progression = make_training_environment(
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V4,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        reward_mode="dense",
        disabled_reward_components=(),
        mask_role_observation=False,
        procedural=False,
    )
    assert isinstance(salient_progression.env, IslandTrainingRewardV4Wrapper)
    assert salient_progression.versions["reward_version"] == ISLAND_TRAINING_REWARD_V4
    salient_progression.env.close()


def test_parallel_and_centralized_interfaces_are_compliant() -> None:
    parallel_api_test(VoyagerIslandEnv(procedural=False), num_cycles=25)
    env = VoyagerIslandCentralizedEnv(procedural=False)
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, _info = env.step(
        np.array([0, 0], dtype=np.int64)
    )
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated


def test_replay_23_records_complete_public_oracle(tmp_path) -> None:
    path = record_island_oracle_replay(tmp_path)
    loader = ReplayLoader(path)
    validation = loader.validate(deep=True)
    assert validation["status"] == "valid"
    assert loader.manifest.versions.replay == "stage7_replay_2.3.0"
    assert loader.manifest.terminal_summary["rescue_success"] is True
    assert loader.manifest.terminal_summary["invalid_scripted_actions"] == 0
