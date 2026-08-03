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
    summarize_civilization_results,
)
from voyager.training.civilization_probe import CivilizationProbeRewardWrapper
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
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
    assert encoded.shape == (591,)
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
    assert np.all(np.isfinite(encoded))


def test_probe_reward_is_shared_and_progress_milestones_do_not_repeat() -> None:
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
    assert infos["agent_0"]["reward_components"]["gather_wood"] == pytest.approx(
        0.25
    )
    agent.energy = 100
    _observations, _rewards, _terms, _truncs, second_infos = env.step(
        {"agent_0": gather}
    )
    assert "gather_wood" not in second_infos["agent_0"]["reward_components"]


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
            gathered_wood_and_stone=flags[0],
            deposited_wood_and_stone=flags[1],
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

    assert batch.observations.shape == (20, 591)
    assert batch.action_masks.shape == (20, V2_FLAT_ACTION_COUNT)
    assert np.all(
        batch.action_masks[np.arange(batch.actions.shape[0]), batch.actions]
    )
