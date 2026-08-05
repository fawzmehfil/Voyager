"""Tests for Stage 7C factorized action sampling and PPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    V2_MEANINGFUL_ACTIONS,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
)
from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    ACHIEVEMENT_IDS,
    AchievementEpisodeResult,
    delivery_emergence,
    entered_camp_interaction_range,
    summarize_achievement_results,
    within_camp_interaction_distance,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
)
from voyager.training.factorized_actions import (
    FACTOR_ARGUMENT_COUNT,
    FACTOR_TARGET_COUNT,
    FACTOR_VERB_COUNT,
    action_components,
    argument_masks,
    choose_factorized_actions,
    flatten_components,
    target_masks,
    verb_masks,
)
from voyager.training.factorized_ppo import FactorizedPPOTrainer
from voyager.training.factorized_recurrent_ppo import (
    FactorizedRecurrentPPOTrainer,
)
from voyager.training.model import (
    build_factorized_actor_critic,
    build_factorized_recurrent_actor_critic,
)
from voyager.training.ppo import PPOConfig
from voyager.training.recurrent_ppo import RecurrentPPOConfig


def test_factorized_components_round_trip_entire_public_registry() -> None:
    actions = np.arange(V2_FLAT_ACTION_COUNT, dtype=np.int32)

    verbs, arguments, targets = action_components(actions)

    np.testing.assert_array_equal(
        flatten_components(verbs, arguments, targets),
        actions,
    )


def test_conditional_masks_follow_selected_action_prefix() -> None:
    flat_masks = np.ones((2, V2_FLAT_ACTION_COUNT), dtype=np.bool_)
    selected_verbs = np.asarray(
        [CivilizationV2Verb.MOVE, CivilizationV2Verb.GIVE],
        dtype=np.int32,
    )

    verbs = verb_masks(flat_masks)
    arguments = argument_masks(flat_masks, selected_verbs)
    selected_arguments = np.asarray(
        [CivilizationV2Argument.NORTH, CivilizationV2Argument.WOOD],
        dtype=np.int32,
    )
    targets = target_masks(flat_masks, selected_verbs, selected_arguments)

    assert verbs.shape == (2, FACTOR_VERB_COUNT)
    assert np.all(verbs)
    assert np.flatnonzero(arguments[0]).tolist() == [
        int(CivilizationV2Argument.NORTH),
        int(CivilizationV2Argument.EAST),
        int(CivilizationV2Argument.SOUTH),
        int(CivilizationV2Argument.WEST),
    ]
    assert int(CivilizationV2Argument.WOOD) in np.flatnonzero(arguments[1])
    assert np.flatnonzero(targets[0]).tolist() == [0]
    assert np.flatnonzero(targets[1]).tolist() == list(range(1, 17))


@pytest.mark.parametrize("mode", ["deterministic", "seeded_stochastic"])
def test_factorized_sampler_only_returns_legal_flat_actions(mode: str) -> None:
    rng = np.random.default_rng(9)
    flat_masks = rng.random((64, V2_FLAT_ACTION_COUNT)) < 0.08
    flat_masks[:, 0] = True

    actions, log_probs = choose_factorized_actions(
        verb_logits=rng.normal(size=(64, FACTOR_VERB_COUNT)),
        argument_logits=rng.normal(size=(64, FACTOR_ARGUMENT_COUNT)),
        target_logits=rng.normal(size=(64, FACTOR_TARGET_COUNT)),
        flat_masks=flat_masks,
        inference_mode=mode,  # type: ignore[arg-type]
        rng=rng,
    )

    assert actions.shape == (64,)
    assert log_probs.shape == (64,)
    assert np.all(np.isfinite(log_probs))
    assert np.all(flat_masks[np.arange(64), actions])


def test_factorized_sampling_removes_duplicate_target_mass() -> None:
    noop = flatten_v2_action(
        CivilizationV2Verb.NOOP,
        CivilizationV2Argument.NONE,
        0,
    )
    give_wood = [
        flatten_v2_action(
            CivilizationV2Verb.GIVE,
            CivilizationV2Argument.WOOD,
            target,
        )
        for target in range(1, 17)
    ]
    masks = np.zeros((1, V2_FLAT_ACTION_COUNT), dtype=np.bool_)
    masks[0, noop] = True
    masks[0, give_wood] = True
    zeros = {
        "verb_logits": np.zeros((1, FACTOR_VERB_COUNT)),
        "argument_logits": np.zeros((1, FACTOR_ARGUMENT_COUNT)),
        "target_logits": np.zeros((1, FACTOR_TARGET_COUNT)),
    }
    rng = np.random.default_rng(4)
    give_count = 0
    trials = 4_000
    for _ in range(trials):
        actions, _log_probs = choose_factorized_actions(
            **zeros,
            flat_masks=masks,
            inference_mode="seeded_stochastic",
            rng=rng,
        )
        verb, _argument, _target = V2_MEANINGFUL_ACTIONS[int(actions[0])]
        give_count += verb == CivilizationV2Verb.GIVE

    assert give_count / trials == pytest.approx(0.5, abs=0.04)


def test_factorized_model_shapes_when_tensorflow_is_available() -> None:
    pytest.importorskip("tensorflow")
    model = build_factorized_actor_critic(
        input_dim=12,
        verb_count=FACTOR_VERB_COUNT,
        argument_count=FACTOR_ARGUMENT_COUNT,
        target_count=FACTOR_TARGET_COUNT,
        hidden_sizes=(8,),
        seed=0,
    )

    verb, argument, target, value = model(
        np.zeros((3, 12), dtype=np.float32),
        training=False,
    )

    assert tuple(verb.shape) == (3, FACTOR_VERB_COUNT)
    assert tuple(argument.shape) == (3, FACTOR_ARGUMENT_COUNT)
    assert tuple(target.shape) == (3, FACTOR_TARGET_COUNT)
    assert tuple(value.shape) == (3, 1)


def test_factorized_ppo_rollout_update_and_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    trainer = FactorizedPPOTrainer(
        PPOConfig(
            total_steps=10,
            environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
            reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
            rollout_steps=1,
            num_agents=10,
            map_size=48,
            max_steps=600,
            train_epochs=1,
            minibatch_size=10,
            hidden_sizes=(8,),
            checkpoint_dir=str(tmp_path),
            checkpoint_every=0,
            reward_mode="none",
        )
    )

    batch = trainer.collect_rollout()
    losses = trainer.update_policy(batch)

    assert batch.observations.shape == (10, 610)
    assert np.all(batch.action_masks[np.arange(batch.actions.shape[0]), batch.actions])
    assert all(np.isfinite(value) for value in losses.values())

    checkpoint = trainer.save_named_checkpoint("round_trip", update=1)
    model, metadata = load_policy_checkpoint(checkpoint)
    outputs = model(np.zeros((2, 610), dtype=np.float32), training=False)

    assert metadata["model_type"] == "factorized_feed_forward"
    assert metadata["algorithm"] == "factorized_shared_policy_ppo"
    assert metadata["action_count"] == V2_FLAT_ACTION_COUNT
    assert [tuple(output.shape) for output in outputs] == [
        (2, FACTOR_VERB_COUNT),
        (2, FACTOR_ARGUMENT_COUNT),
        (2, FACTOR_TARGET_COUNT),
        (2, 1),
    ]


def test_camp_return_diagnostics_match_environment_interaction_radius() -> None:
    assert within_camp_interaction_distance(0)
    assert within_camp_interaction_distance(1)
    assert not within_camp_interaction_distance(2)
    assert entered_camp_interaction_range(2, 1)
    assert not entered_camp_interaction_range(1, 0)
    assert not entered_camp_interaction_range(3, 2)


def test_delivery_emergence_rejects_one_lucky_episode() -> None:
    episodes = [_delivery_episode(seed=index, delivered=index < 1) for index in range(20)]

    diagnosis = delivery_emergence(summarize_achievement_results(episodes))

    assert diagnosis["material_deposit_episode_rate"] == pytest.approx(0.05)
    assert not diagnosis["repeatable_delivery_emerged"]


def test_delivery_emergence_accepts_twenty_percent_of_episodes() -> None:
    episodes = [_delivery_episode(seed=index, delivered=index < 4) for index in range(20)]

    diagnosis = delivery_emergence(summarize_achievement_results(episodes))

    assert diagnosis["material_deposit_episode_rate"] == pytest.approx(0.20)
    assert diagnosis["repeatable_delivery_emerged"]


def test_factorized_recurrent_model_shapes() -> None:
    pytest.importorskip("tensorflow")
    model = build_factorized_recurrent_actor_critic(
        input_dim=12,
        verb_count=FACTOR_VERB_COUNT,
        argument_count=FACTOR_ARGUMENT_COUNT,
        target_count=FACTOR_TARGET_COUNT,
        encoder_sizes=(8,),
        recurrent_hidden_size=8,
        seed=0,
    )

    outputs = model(
        [
            np.zeros((2, 3, 12), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
        ],
        training=False,
    )

    assert [tuple(output.shape) for output in outputs] == [
        (2, 3, FACTOR_VERB_COUNT),
        (2, 3, FACTOR_ARGUMENT_COUNT),
        (2, 3, FACTOR_TARGET_COUNT),
        (2, 3, 1),
        (2, 8),
    ]


def test_factorized_recurrent_rollout_update_and_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    trainer = FactorizedRecurrentPPOTrainer(
        RecurrentPPOConfig(
            total_steps=20,
            rollout_steps=2,
            train_epochs=1,
            sequence_length=2,
            sequence_minibatch_size=4,
            encoder_sizes=(8,),
            recurrent_hidden_size=8,
            checkpoint_dir=str(tmp_path),
            reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        )
    )

    batch = trainer.collect_rollout()
    losses = trainer.update_policy(batch, entropy_coef=0.01)

    assert batch.agent_steps == 20
    assert batch.observations.shape == (10, 2, 610)
    sequence_indices, step_indices = np.where(batch.valid)
    assert np.all(
        batch.action_masks[
            sequence_indices,
            step_indices,
            batch.actions[batch.valid],
        ]
    )
    assert all(np.isfinite(value) for value in losses.values())

    checkpoint = trainer.save_named_checkpoint("round_trip", update=1)
    model, metadata = load_policy_checkpoint(checkpoint)
    outputs = model(
        [
            np.zeros((2, 3, 610), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
        ],
        training=False,
    )

    assert metadata["model_type"] == "factorized_recurrent_gru"
    assert metadata["algorithm"] == "factorized_shared_policy_recurrent_ppo"
    assert [tuple(output.shape) for output in outputs] == [
        (2, 3, FACTOR_VERB_COUNT),
        (2, 3, FACTOR_ARGUMENT_COUNT),
        (2, 3, FACTOR_TARGET_COUNT),
        (2, 3, 1),
        (2, 8),
    ]


def _delivery_episode(*, seed: int, delivered: bool) -> AchievementEpisodeResult:
    achievements = {achievement: False for achievement in ACHIEVEMENT_IDS}
    achievements["deposit_wood"] = delivered
    return AchievementEpisodeResult(
        policy="test",
        inference_mode="seeded_stochastic",
        seed=seed,
        world_steps=600,
        agent_transitions=6_000,
        achievements=achievements,
        unlock_ticks={"deposit_wood": 10} if delivered else {},
        invalid_actions=0,
        submitted_actions=6_000,
        gathered_counts={"wood": 1} if delivered else {},
        deposited_counts={"wood": 1} if delivered else {},
        peak_camp_stockpile={"wood": 1} if delivered else {},
        active_at_first_night=10,
        final_active=10,
        deaths=0,
        return_diagnostics={"resource_return_arrivals": float(delivered)},
    )
