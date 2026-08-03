"""Stage 7C trainer, outcome evaluator, and hot-path acceptance tests."""

from __future__ import annotations

import numpy as np
import pytest

from voyager.envs.civilization_v2 import VoyagerCivilizationV2Env
from voyager.sim.constants import Resource
from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
)
from voyager.training.civilization_evaluation import (
    CivilizationEpisodeResult,
    compare_against_random,
    run_civilization_episode,
    summarize_civilization_results,
)
from voyager.training.civilization_probe import CivilizationProbeRewardWrapper
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_PROBE_V1_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.obs import flat_observation_size, flatten_observation
from voyager.training.ppo import PPOConfig, PPOTrainer


def test_v2_training_adapter_exposes_stable_actor_contract() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, infos = env.reset(seed=0)

    observation = observations["agent_0"]
    encoded = flatten_observation(
        observation,
        training_environment.observation_encoder,
    )
    assert encoded.shape == (601,)
    assert encoded.shape == (
        flat_observation_size(
            env.observation_space("agent_0"),
            training_environment.observation_encoder,
        ),
    )
    assert env.action_space("agent_0").n == V2_FLAT_ACTION_COUNT
    assert np.asarray(infos["agent_0"]["action_mask"]).shape == (
        V2_FLAT_ACTION_COUNT,
    )
    assert observation["agent_identity"].shape == (10,)
    assert observation["agent_identity"].sum() == 1
    assert observation["agent_identity"][0] == 1
    assert np.all(np.isfinite(encoded))


def test_v1_probe_contract_retains_591_value_observation() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_V1_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, _infos = env.reset(seed=0)

    assert "agent_identity" not in observations["agent_0"]
    assert flatten_observation(
        observations["agent_0"],
        training_environment.observation_encoder,
    ).shape == (591,)


def test_probe_reward_is_shared_and_rewards_each_finite_gathered_unit() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    wood_y, wood_x = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[0]
    agent.x, agent.y = int(wood_x), int(wood_y)
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )

    _observations, rewards, _terms, _truncs, infos = env.step({"agent_0": gather})

    assert len(set(rewards.values())) == 1
    assert infos["agent_0"]["reward_components"]["gather_wood"] == pytest.approx(0.10)
    agent.energy = 100
    _observations, _rewards, _terms, _truncs, second_infos = env.step(
        {"agent_0": gather}
    )
    assert second_infos["agent_0"]["reward_components"]["gather_wood"] == pytest.approx(
        0.10
    )


def test_delivery_reward_cannot_be_repeated_by_withdraw_and_redeposit() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    wood_y, wood_x = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[0]
    agent.x, agent.y = int(wood_x), int(wood_y)
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    deposit = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT,
        CivilizationV2Argument.WOOD,
        0,
    )
    withdraw = flatten_v2_action(
        CivilizationV2Verb.WITHDRAW,
        CivilizationV2Argument.WOOD,
        0,
    )
    env.step({"agent_0": gather})
    agent.x, agent.y = state.camp.x, state.camp.y

    _obs, _rewards, _terms, _truncs, first_infos = env.step({"agent_0": deposit})
    env.step({"agent_0": withdraw})
    _obs, _rewards, _terms, _truncs, second_infos = env.step({"agent_0": deposit})

    assert first_infos["agent_0"]["reward_components"][
        "first_delivery_wood"
    ] == pytest.approx(0.15)
    assert "first_delivery_wood" not in second_infos["agent_0"]["reward_components"]


def test_work_mask_requires_reservable_materials() -> None:
    env = VoyagerCivilizationV2Env()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    workbench = state.structures["workbench"]
    agent.x, agent.y = workbench.x, workbench.y
    work = flatten_v2_action(
        CivilizationV2Verb.WORK,
        CivilizationV2Argument.WORKBENCH,
        0,
    )

    assert not env.world.v2_action_mask("agent_0")[work]
    state.camp.stockpile.update(wood=6, stone=2)
    assert env.world.v2_action_mask("agent_0")[work]


def test_v2_step_reuses_masks_slots_and_conservation_audits() -> None:
    env = VoyagerCivilizationV2Env()
    mask_calls = 0
    slot_calls = 0
    conservation_calls = 0
    original_mask = env.world.v2_action_mask
    original_slots = env.world.v2_entity_slots
    original_conservation = env.world.reconcile_v2_ledger

    def counted_mask(agent_id: str) -> np.ndarray:
        nonlocal mask_calls
        mask_calls += 1
        return original_mask(agent_id)

    def counted_slots(agent_id: str) -> list[str]:
        nonlocal slot_calls
        slot_calls += 1
        return original_slots(agent_id)

    def counted_conservation() -> dict[str, int]:
        nonlocal conservation_calls
        conservation_calls += 1
        return original_conservation()

    env.world.v2_action_mask = counted_mask  # type: ignore[method-assign]
    env.world.v2_entity_slots = counted_slots  # type: ignore[method-assign]
    env.world.reconcile_v2_ledger = counted_conservation  # type: ignore[method-assign]
    env.reset(seed=0)

    assert mask_calls == 10
    assert slot_calls == 10
    assert conservation_calls == 1


def test_outcome_summary_and_paired_gate_use_independent_metrics() -> None:
    def result(policy: str, seed: int, successes: int) -> CivilizationEpisodeResult:
        flags = [index < successes for index in range(5)]
        return CivilizationEpisodeResult(
            policy=policy,
            seed=seed,
            world_steps=600,
            agent_transitions=6_000,
            shared_return=999.0,
            gathered_workbench_bundle_by_100=flags[0],
            workbench_materials_available_by_300=flags[1],
            workbench_complete=flags[2],
            any_tool_crafted=flags[3],
            majority_active_at_300=flags[4],
            active_at_300=10 if flags[4] else 0,
            final_active=10,
            deaths=0,
            invalid_actions=0,
            submitted_actions=6_000,
        )

    learned = [result("ppo", seed, 5) for seed in range(4)]
    random = [result("random", seed, 0) for seed in range(4)]

    assert summarize_civilization_results(learned)["composite"] == 1.0
    comparison = compare_against_random(
        learned,
        random,
        bootstrap_samples=100,
    )
    assert comparison["overall_passed"] is True
    assert comparison["composite_difference_ci95"] == [1.0, 1.0]


def test_episode_exports_action_resource_and_rejection_diagnostics() -> None:
    result = run_civilization_episode(
        policy_name="legal_random",
        policy="legal_random",
        seed=0,
    )

    assert sum(result.selected_verbs.values()) == result.submitted_actions
    assert sum(result.selected_actions.values()) == result.submitted_actions
    assert sum(result.rejection_reasons.values()) == result.invalid_actions
    assert {"food", "wood", "stone"} <= result.peak_camp_stockpile.keys()


def test_tensorflow_ppo_collects_v2_masked_rollout() -> None:
    pytest.importorskip("tensorflow")
    trainer = PPOTrainer(
        PPOConfig(
            total_steps=20,
            environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
            reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
            rollout_steps=2,
            num_agents=10,
            map_size=48,
            max_steps=600,
            hidden_sizes=(8,),
            train_epochs=1,
            minibatch_size=16,
            checkpoint_dir=None,
            reward_mode="none",
        )
    )

    batch = trainer.collect_rollout()

    assert batch.observations.shape == (20, 601)
    assert batch.action_masks.shape == (20, V2_FLAT_ACTION_COUNT)
    assert np.all(
        batch.action_masks[np.arange(batch.actions.shape[0]), batch.actions]
    )
