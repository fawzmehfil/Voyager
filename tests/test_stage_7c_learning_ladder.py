"""Acceptance tests for the controlled Stage 7C learning ladder."""

from __future__ import annotations

import numpy as np
import pytest

from voyager.sim.constants import Resource
from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
    unflatten_v2_action,
)
from voyager.training.civilization_learning_ladder import (
    DELIVERY_DIAGNOSTIC_TASKS,
    LEARNING_TASK_CONTRACTS,
    CivilizationLearningTaskWrapper,
    learning_task_definition,
)
from voyager.training.civilization_learning_ladder_evaluation import (
    diagnose_delivery_components,
    diagnose_learning_ladder,
    learning_task_gate,
    run_learning_task_episode,
)
from voyager.training.environments import (
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.obs import flatten_observation


def test_learning_tasks_keep_v3_actor_contract_and_public_action_registry() -> None:
    for task, contract in LEARNING_TASK_CONTRACTS.items():
        training = make_training_environment(
            environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
            reward_contract=contract,
            num_agents=10,
            map_size=48,
            max_steps=600,
            reward_mode="none",
            disabled_reward_components=(),
            mask_role_observation=False,
        )
        observations, infos = training.env.reset(seed=0)

        assert training.env.action_space("agent_0").n == V2_FLAT_ACTION_COUNT
        assert flatten_observation(
            observations["agent_0"], training.observation_encoder
        ).shape == (604,)
        mask = np.asarray(infos["agent_0"]["action_mask"])
        assert np.all(mask <= learning_task_definition(task).allowed_actions)
        assert infos["agent_0"]["learning_task"] == task
        training.env.close()


def test_delivery_task_masks_unrelated_mechanics_and_rewards_completion_once() -> None:
    env = CivilizationLearningTaskWrapper("delivery")
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    craft = flatten_v2_action(
        CivilizationV2Verb.CRAFT,
        CivilizationV2Argument.AXE_RECIPE,
        0,
    )
    assert not env.env.env.env.action_mask("agent_0")[craft]
    interact = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    deposits = {
        item: flatten_v2_action(
            CivilizationV2Verb.DEPOSIT,
            argument,
            0,
        )
        for item, argument in (
            ("wood", CivilizationV2Argument.WOOD),
            ("stone", CivilizationV2Argument.STONE),
        )
    }
    completion_rewards = 0.0
    for item, resource, required in (
        ("wood", Resource.WOOD, 6),
        ("stone", Resource.STONE, 2),
    ):
        for _index in range(required):
            y, x = np.argwhere(
                (state.resource_ids == resource)
                & (state.resource_quantities > 0)
            )[0]
            agent = state.agents["agent_0"]
            agent.x, agent.y = int(x), int(y)
            agent.energy = 100
            env.step({"agent_0": interact})
            agent.x, agent.y = state.camp.x, state.camp.y
            _obs, _rewards, _terms, _truncs, infos = env.step(
                {"agent_0": deposits[item]}
            )
            completion_rewards += infos["agent_0"][
                "shared_reward_components"
            ].get("delivery_complete", 0.0)

    assert env.success()
    assert env.score() == pytest.approx(1.0)
    assert completion_rewards == pytest.approx(2.0)
    noop = flatten_v2_action(
        CivilizationV2Verb.NOOP,
        CivilizationV2Argument.NONE,
        0,
    )
    _obs, _rewards, _terms, _truncs, repeated = env.step({"agent_0": noop})
    assert "delivery_complete" not in repeated["agent_0"][
        "shared_reward_components"
    ]


def test_construction_task_is_prestocked_conserving_and_completable() -> None:
    env = CivilizationLearningTaskWrapper("construction")
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    assert state.camp.stockpile["wood"] == 6
    assert state.camp.stockpile["stone"] == 2
    assert env.world.reconcile_v2_ledger() == {}
    work = flatten_v2_action(
        CivilizationV2Verb.WORK,
        CivilizationV2Argument.WORKBENCH,
        0,
    )

    total_progress_reward = 0.0
    for _index in range(16):
        _obs, _rewards, _terms, _truncs, infos = env.step(
            {"agent_8": work}
        )
        total_progress_reward += infos["agent_8"][
            "shared_reward_components"
        ].get("workbench_progress", 0.0)
        if env.success():
            break

    assert env.success()
    assert env.score() == pytest.approx(1.0)
    assert total_progress_reward == pytest.approx(2.0)
    assert env.world.reconcile_v2_ledger() == {}


@pytest.mark.parametrize(
    ("task", "item", "resource", "requirement"),
    (
        ("gather_wood", "wood", Resource.WOOD, 6),
        ("gather_stone", "stone", Resource.STONE, 2),
    ),
)
def test_resource_acquisition_tasks_reward_only_the_target_resource(
    task: str,
    item: str,
    resource: Resource,
    requirement: int,
) -> None:
    env = CivilizationLearningTaskWrapper(task)  # type: ignore[arg-type]
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    interact = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    total_credit = 0.0
    completion_reward = 0.0
    for _index in range(requirement):
        y, x = np.argwhere(
            (state.resource_ids == resource) & (state.resource_quantities > 0)
        )[0]
        agent = state.agents["agent_0"]
        agent.x, agent.y = int(x), int(y)
        agent.energy = 100
        _obs, _rewards, _terms, _truncs, infos = env.step(
            {"agent_0": interact}
        )
        total_credit += infos["agent_0"][
            "individual_reward_components"
        ].get(f"gather_{item}", 0.0)
        completion_reward += infos["agent_0"][
            "shared_reward_components"
        ].get(f"gather_{item}_complete", 0.0)

    assert env.success()
    assert env.score() == pytest.approx(1.0)
    assert total_credit == pytest.approx(0.10 * requirement)
    assert completion_reward == pytest.approx(2.0)


def test_return_to_camp_starts_carriers_on_ring_and_completes_by_deposit() -> None:
    env = CivilizationLearningTaskWrapper("return_to_camp")
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    for agent in state.agents.values():
        assert abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y) == 6
        assert agent.inventory["wood"] + agent.inventory["stone"] == 2
    assert env.world.reconcile_v2_ledger() == {}
    interact = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    assert not env.env.env.env.action_mask("agent_0")[interact]

    deposit_wood = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT,
        CivilizationV2Argument.WOOD,
        0,
    )
    deposit_stone = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT,
        CivilizationV2Argument.STONE,
        0,
    )
    carriers = (
        ("agent_0", deposit_wood, (24, 31)),
        ("agent_1", deposit_wood, (25, 31)),
        ("agent_2", deposit_wood, (24, 32)),
        ("agent_5", deposit_stone, (23, 31)),
    )
    completion_reward = 0.0
    for agent_id, action, position in carriers:
        agent = state.agents[agent_id]
        agent.x, agent.y = position
        for _index in range(2):
            _obs, _rewards, _terms, _truncs, infos = env.step(
                {agent_id: action}
            )
            completion_reward += infos[agent_id][
                "shared_reward_components"
            ].get("return_to_camp_complete", 0.0)

    assert env.success()
    assert env.score() == pytest.approx(1.0)
    assert completion_reward == pytest.approx(2.0)
    assert env.world.reconcile_v2_ledger() == {}


def test_survival_task_starts_before_night_with_completed_shelter() -> None:
    env = CivilizationLearningTaskWrapper("survival")
    observations, infos = env.reset(seed=0)
    state = env.world.state
    assert state is not None

    assert state.step_count == 180
    assert state.structures["shelter"].complete
    assert state.structures["shelter"].capacity == 6
    assert infos["agent_0"]["learning_task_horizon"] == 120
    time_values = observations["agent_0"]["time"]
    assert time_values[2] == pytest.approx(1.0)


def test_learning_task_episode_is_seed_reproducible_and_respects_horizon() -> None:
    for task, horizon in (
        ("gather_wood", 100),
        ("gather_stone", 100),
        ("return_to_camp", 60),
        ("delivery", 150),
        ("construction", 60),
        ("survival", 120),
    ):
        first = run_learning_task_episode(
            task=task,
            policy_name="legal_random",
            policy="legal_random",
            seed=17,
        )
        second = run_learning_task_episode(
            task=task,
            policy_name="legal_random",
            policy="legal_random",
            seed=17,
        )
        assert first.as_dict() == second.as_dict()
        assert first.task_steps == horizon


def test_learning_ladder_gate_and_diagnosis_identify_first_failed_skill() -> None:
    learned = {
        "success_rate": 0.7,
        "mean_score": 0.8,
        "mean_completion_step": 10.0,
        "invalid_action_rate": 0.01,
    }
    random = {
        "success_rate": 0.1,
        "mean_score": 0.4,
        "mean_completion_step": 20.0,
        "invalid_action_rate": 0.02,
    }
    delivery = learning_task_gate("delivery", learned, random)
    construction = learning_task_gate(
        "construction", {**learned, "success_rate": 0.9}, random
    )
    survival = learning_task_gate(
        "survival", {**learned, "success_rate": 0.9}, random
    )

    assert delivery["passed"]
    assert construction["passed"]
    assert survival["passed"]
    assert diagnose_learning_ladder(
        {
            "delivery": {"passed": False},
            "construction": construction,
            "survival": survival,
        }
    ) == "navigation_exploration_or_delayed_credit_failure"
    assert diagnose_delivery_components(
        {
            "gather_wood": {"passed": True, "capability_learned": True},
            "gather_stone": {"passed": True, "capability_learned": True},
            "return_to_camp": {
                "passed": False,
                "capability_learned": False,
            },
        }
    ) == {
        "diagnosis": "home_navigation_or_delayed_deposit_credit_failure",
        "failed_components": ["return_to_camp"],
        "efficiency_issues": [],
        "next_step": "remediate only the failed component before composing delivery",
    }


def test_delivery_diagnosis_separates_learned_capability_from_efficiency() -> None:
    diagnosis = diagnose_delivery_components(
        {
            "gather_wood": {"passed": False, "capability_learned": True},
            "gather_stone": {"passed": False, "capability_learned": True},
            "return_to_camp": {"passed": False, "capability_learned": True},
        }
    )

    assert diagnosis == {
        "diagnosis": "component_skills_trainable_combination_or_team_state_failure",
        "failed_components": [],
        "efficiency_issues": [
            "gather_wood",
            "gather_stone",
            "return_to_camp",
        ],
        "next_step": "build a curriculum that composes the passed skills",
    }


def test_allowed_action_sets_contain_only_declared_verbs() -> None:
    allowed_by_task = {
        "gather_wood": {"noop", "move", "interact", "rest"},
        "gather_stone": {"noop", "move", "interact", "rest"},
        "return_to_camp": {"noop", "move", "rest", "deposit"},
        "delivery": {"noop", "move", "interact", "rest", "deposit"},
        "construction": {"noop", "move", "rest", "work"},
        "survival": {
            "noop",
            "move",
            "interact",
            "eat",
            "rest",
            "deposit",
            "withdraw",
            "use",
            "defend",
        },
    }
    for task, expected in allowed_by_task.items():
        definition = learning_task_definition(task)  # type: ignore[arg-type]
        verbs = {
            CivilizationV2Verb(unflatten_v2_action(index)[0]).name.lower()
            for index in np.flatnonzero(definition.allowed_actions)
        }
        assert verbs == expected

    assert DELIVERY_DIAGNOSTIC_TASKS == (
        "gather_wood",
        "gather_stone",
        "return_to_camp",
    )
