"""Stage 5.5 tests for masking, entropy decay, and economy rewards."""

from __future__ import annotations

import numpy as np
import pytest

from voyager.envs import VoyagerParallelEnv
from voyager.policies.evaluation import ppo_policy_specs
from voyager.sim.constants import ACTION_COUNT, Action, Resource
from voyager.training.masking import action_mask_from_info, mask_numpy_logits
from voyager.training.ppo import PPOConfig, PPOTrainer


def test_reset_info_exposes_valid_action_mask() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    _observations, infos = env.reset(seed=0)

    for agent_id in env.agents:
        mask = action_mask_from_info(infos[agent_id])
        assert mask.shape == (ACTION_COUNT,)
        assert mask.dtype == np.bool_
        assert mask[Action.NOOP]
        assert np.any(mask)


def test_mask_blocks_requested_impossible_actions() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    _observations, infos = env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    state.resource_ids[agent.y, agent.x] = Resource.NONE
    state.resource_quantities[agent.y, agent.x] = 0

    mask = env.action_mask("agent_0")

    assert not mask[Action.GATHER]
    assert not mask[Action.EAT]
    assert not mask[Action.DEPOSIT_FOOD]
    assert not mask[Action.DEPOSIT_WOOD]
    assert not mask[Action.DEPOSIT_STONE]
    assert not mask[Action.WITHDRAW_FOOD]
    assert not mask[Action.BUILD_SHELTER]
    np.testing.assert_array_equal(mask, infos["agent_0"]["action_mask"])


def test_mask_enables_actions_when_requirements_are_met() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.inventory.update(food=1, wood=1, stone=1)
    state.camp.stockpile["food"] = 1
    state.resource_ids[agent.y, agent.x] = Resource.FOOD
    state.resource_quantities[agent.y, agent.x] = 1

    mask = env.action_mask("agent_0")

    for action in (
        Action.GATHER,
        Action.EAT,
        Action.DEPOSIT_FOOD,
        Action.DEPOSIT_WOOD,
        Action.DEPOSIT_STONE,
        Action.WITHDRAW_FOOD,
        Action.BUILD_SHELTER,
    ):
        assert mask[action]


def test_mask_blocks_camp_actions_away_from_camp() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.inventory.update(food=1, wood=1, stone=1)
    state.camp.stockpile["food"] = 1
    for y, x in np.argwhere(state.terrain != 0):
        if (int(x), int(y)) != (state.camp.x, state.camp.y):
            agent.x = int(x)
            agent.y = int(y)
            break

    mask = env.action_mask("agent_0")

    assert not mask[Action.DEPOSIT_FOOD]
    assert not mask[Action.DEPOSIT_WOOD]
    assert not mask[Action.DEPOSIT_STONE]
    assert not mask[Action.WITHDRAW_FOOD]
    assert not mask[Action.BUILD_SHELTER]


def test_masked_logits_cannot_select_invalid_highest_logit() -> None:
    logits = np.zeros(ACTION_COUNT, dtype=np.float32)
    logits[Action.GATHER] = 100.0
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[Action.NOOP] = 1
    mask[Action.MOVE_UP] = 1
    logits[Action.MOVE_UP] = 1.0

    masked_logits = mask_numpy_logits(logits, mask)

    assert int(np.argmax(masked_logits)) == Action.MOVE_UP
    assert masked_logits[Action.GATHER] < -1e8


def test_entropy_coefficient_decays_linearly_to_floor() -> None:
    config = PPOConfig(
        total_steps=1_000,
        entropy_coef_start=0.02,
        entropy_coef_end=0.001,
    )

    assert config.entropy_coefficient(0) == pytest.approx(0.02)
    assert config.entropy_coefficient(500) == pytest.approx(0.0105)
    assert config.entropy_coefficient(1_000) == pytest.approx(0.001)
    assert config.entropy_coefficient(2_000) == pytest.approx(0.001)


def test_config_rejects_increasing_entropy_schedule() -> None:
    with pytest.raises(ValueError, match="entropy_coef_start"):
        PPOConfig(entropy_coef_start=0.001, entropy_coef_end=0.02).validate()


def test_useful_food_deposit_and_food_security_are_rewarded() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.inventory["food"] = 1

    _obs, rewards, _terms, _truncs, infos = env.step(
        {"agent_0": Action.DEPOSIT_FOOD}
    )

    components = infos["agent_0"]["reward_components"]
    assert isinstance(components, dict)
    assert components["action"] == pytest.approx(0.20)
    assert components["food_security"] > 0.0
    assert rewards["agent_0"] == pytest.approx(sum(components.values()))


def test_food_withdraw_redeposit_cycle_does_not_repeat_deposit_reward() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.inventory["food"] = 1

    env.step({"agent_0": Action.DEPOSIT_FOOD})
    _obs, _rewards, _terms, _truncs, withdraw_infos = env.step(
        {"agent_0": Action.WITHDRAW_FOOD}
    )
    _obs, _rewards, _terms, _truncs, redeposit_infos = env.step(
        {"agent_0": Action.DEPOSIT_FOOD}
    )

    assert withdraw_infos["agent_0"]["reward_components"]["action"] == 0.0
    assert redeposit_infos["agent_0"]["reward_components"]["action"] == 0.0


def test_builder_can_consume_material_from_shared_camp() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.inventory["wood"] = 0
    agent.inventory["stone"] = 0
    state.camp.stockpile["wood"] = 1

    assert env.action_mask("agent_0")[Action.BUILD_SHELTER]
    _obs, _rewards, _terms, _truncs, infos = env.step(
        {"agent_0": Action.BUILD_SHELTER}
    )

    assert infos["agent_0"]["event"] == "build_shelter_camp_wood"
    assert state.camp.stockpile["wood"] == 0
    assert state.camp.shelter_progress > 0.0


def test_shelter_progress_reward_is_shared_with_group() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    env.reset(seed=0)
    state = env.world.state
    builder = state.agents["agent_0"]
    builder.x = state.camp.x
    builder.y = state.camp.y
    builder.inventory["wood"] = 1

    _obs, _rewards, _terms, _truncs, infos = env.step(
        {
            "agent_0": Action.BUILD_SHELTER,
            "agent_1": Action.NOOP,
        }
    )

    builder_components = infos["agent_0"]["reward_components"]
    teammate_components = infos["agent_1"]["reward_components"]
    assert builder_components["shelter_progress"] > 0.0
    assert teammate_components["shelter_progress"] == pytest.approx(
        builder_components["shelter_progress"]
    )


def test_group_death_penalty_reaches_surviving_teammates() -> None:
    env = VoyagerParallelEnv(num_agents=2, storm_start_step=10_000)
    env.reset(seed=0)
    doomed = env.world.state.agents["agent_0"]
    doomed.health = 0.01
    doomed.hunger = 100.0

    _obs, _rewards, terms, _truncs, infos = env.step(
        {
            "agent_0": Action.NOOP,
            "agent_1": Action.NOOP,
        }
    )

    assert terms["agent_0"]
    assert infos["agent_1"]["reward_components"]["team_death"] < 0.0
    assert infos["agent_1"]["reward_components"]["group_survival"] < 0.01


def test_evaluator_defines_both_ppo_modes() -> None:
    specs = ppo_policy_specs("checkpoints/stage5/latest")

    assert [name for name, _factory in specs] == [
        "ppo_deterministic",
        "ppo_stochastic",
    ]


def test_ppo_rollout_only_collects_masked_actions() -> None:
    pytest.importorskip("tensorflow")
    trainer = PPOTrainer(
        PPOConfig(
            total_steps=8,
            rollout_steps=4,
            num_agents=2,
            max_steps=20,
            hidden_sizes=(8,),
            checkpoint_dir=None,
        )
    )

    batch = trainer.collect_rollout()

    selected_actions_are_valid = batch.action_masks[
        np.arange(batch.actions.shape[0]),
        batch.actions,
    ]
    assert np.all(selected_actions_are_valid)
