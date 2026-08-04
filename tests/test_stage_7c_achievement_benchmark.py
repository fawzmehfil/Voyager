"""Stage 7C achievement spectrum and recurrent baseline tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voyager.sim.registries_v2 import V2_FLAT_ACTION_COUNT
from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    ACHIEVEMENT_IDS,
    AchievementEpisodeResult,
    CivilizationAchievementTracker,
    calibration_gate,
    smoothed_geometric_mean,
    summarize_achievement_results,
)
from voyager.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOTrainer


def test_achievement_tracker_records_full_current_progression_spectrum() -> None:
    tracker = CivilizationAchievementTracker()
    state = _state(tick=300, camp_wood=6, camp_stone=2, workbench_complete=True)
    tracker.observe(
        state=state,
        new_ledger=[
            {"event": "gather", "item": "food", "quantity": 1},
            {"event": "gather", "item": "wood", "quantity": 6},
            {"event": "gather", "item": "stone", "quantity": 2},
            {"event": "deposit", "item": "food", "quantity": 1},
            {"event": "deposit", "item": "wood", "quantity": 6},
            {"event": "deposit", "item": "stone", "quantity": 2},
            {
                "event": "construction_reserve",
                "target": "workbench",
            },
            {"event": "craft_tool", "tool": "axe", "quantity": 1},
            {"event": "give_tool", "tool": "axe", "quantity": 1},
        ],
    )
    tracker.finish(state)

    assert tracker.flags() == {achievement: True for achievement in ACHIEVEMENT_IDS}
    assert tracker.active_at_first_night == 10
    assert set(tracker.unlock_ticks) == set(ACHIEVEMENT_IDS)


def test_achievement_tracker_requires_simultaneous_camp_bundle() -> None:
    tracker = CivilizationAchievementTracker()
    tracker.observe(
        state=_state(tick=10, camp_wood=6, camp_stone=0),
        new_ledger=[
            {"event": "gather", "item": "wood", "quantity": 6},
            {"event": "gather", "item": "stone", "quantity": 2},
        ],
    )
    tracker.observe(
        state=_state(tick=11, camp_wood=0, camp_stone=2),
        new_ledger=[],
    )

    assert tracker.flags()["gather_workbench_bundle"]
    assert not tracker.flags()["assemble_camp_bundle"]


def test_smoothed_geometric_mean_handles_sparse_achievement_rates() -> None:
    assert smoothed_geometric_mean([0.0, 0.0]) == pytest.approx(0.0)
    assert smoothed_geometric_mean([1.0, 1.0]) == pytest.approx(1.0)
    sparse = smoothed_geometric_mean([1.0, 0.0])
    assert 0.0 < sparse < 0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        smoothed_geometric_mean([1.1])


def test_summary_reports_every_rate_and_geometric_score() -> None:
    first = _episode({achievement: True for achievement in ACHIEVEMENT_IDS}, seed=1)
    second = _episode(
        {achievement: achievement == "gather_wood" for achievement in ACHIEVEMENT_IDS},
        seed=2,
    )

    summary = summarize_achievement_results([first, second])

    rates = summary["achievement_rates"]
    assert isinstance(rates, dict)
    assert tuple(rates) == ACHIEVEMENT_IDS
    assert rates["gather_wood"] == pytest.approx(1.0)
    assert rates["complete_workbench"] == pytest.approx(0.5)
    achievement_score = summary["achievement_score"]
    assert isinstance(achievement_score, float)
    assert 0.0 < achievement_score < 1.0


def test_calibration_requires_score_and_meaningful_progression_separation() -> None:
    random_summary = _summary_with_rates({})
    feed_forward = _summary_with_rates(
        {"gather_wood": 1.0, "deposit_wood": 0.5}
    )
    recurrent = _summary_with_rates(
        {
            "gather_wood": 1.0,
            "deposit_wood": 0.8,
            "deposit_stone": 0.5,
        }
    )

    without_recurrent = calibration_gate(
        random_summary=random_summary,
        feed_forward_summary=feed_forward,
    )
    with_recurrent = calibration_gate(
        random_summary=random_summary,
        feed_forward_summary=feed_forward,
        recurrent_summary=recurrent,
    )

    assert without_recurrent["at_least_one_learned_baseline_exceeds_random"]
    assert not without_recurrent["content_work_may_resume"]
    assert without_recurrent["next_action"] == "run_recurrent_ppo"
    assert with_recurrent["strict_ordering_demonstrated"]
    assert with_recurrent["content_work_may_resume"]


def test_recurrent_ppo_collects_episode_safe_masked_sequences() -> None:
    pytest.importorskip("tensorflow")
    trainer = RecurrentPPOTrainer(
        RecurrentPPOConfig(
            total_steps=20,
            rollout_steps=2,
            train_epochs=1,
            sequence_length=2,
            sequence_minibatch_size=4,
            encoder_sizes=(8,),
            recurrent_hidden_size=8,
            checkpoint_dir=None,
        )
    )

    batch = trainer.collect_rollout()

    assert batch.agent_steps == 20
    assert batch.observations.shape == (10, 2, 604)
    assert batch.initial_states.shape == (10, 8)
    assert batch.action_masks.shape == (10, 2, V2_FLAT_ACTION_COUNT)
    sequence_indices, step_indices = np.where(batch.valid)
    assert np.all(
        batch.action_masks[
            sequence_indices,
            step_indices,
            batch.actions[batch.valid],
        ]
    )
    losses = trainer.update_policy(batch, entropy_coef=0.01)
    assert all(np.isfinite(value) for value in losses.values())


def test_recurrent_checkpoint_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    trainer = RecurrentPPOTrainer(
        RecurrentPPOConfig(
            total_steps=10,
            rollout_steps=1,
            train_epochs=1,
            sequence_length=1,
            sequence_minibatch_size=2,
            encoder_sizes=(8,),
            recurrent_hidden_size=8,
            checkpoint_dir=str(tmp_path),
        )
    )
    checkpoint = trainer.save_named_checkpoint("round_trip", update=0)

    model, metadata = load_policy_checkpoint(checkpoint)
    logits, values, final_state = model(
        [
            np.zeros((2, 3, 604), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
        ],
        training=False,
    )

    assert metadata["model_type"] == "recurrent_gru"
    assert tuple(logits.shape) == (2, 3, V2_FLAT_ACTION_COUNT)
    assert tuple(values.shape) == (2, 3, 1)
    assert tuple(final_state.shape) == (2, 8)


def _state(
    *,
    tick: int,
    camp_wood: int = 0,
    camp_stone: int = 0,
    workbench_complete: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        step_count=tick,
        camp=SimpleNamespace(
            stockpile={"food": 0, "wood": camp_wood, "stone": camp_stone}
        ),
        structures={
            "workbench": SimpleNamespace(complete=workbench_complete)
        },
        agents={
            f"agent_{index}": SimpleNamespace(life_state="active")
            for index in range(10)
        },
    )


def _episode(achievements: dict[str, bool], *, seed: int) -> AchievementEpisodeResult:
    return AchievementEpisodeResult(
        policy="test",
        inference_mode="seeded_stochastic",
        seed=seed,
        world_steps=600,
        agent_transitions=6_000,
        achievements=achievements,
        unlock_ticks={name: 10 for name, value in achievements.items() if value},
        invalid_actions=0,
        submitted_actions=6_000,
        gathered_counts={},
        deposited_counts={},
        peak_camp_stockpile={},
        active_at_first_night=10,
        final_active=10,
        deaths=0,
    )


def _summary_with_rates(overrides: dict[str, float]) -> dict[str, object]:
    rates = {achievement: 0.0 for achievement in ACHIEVEMENT_IDS}
    rates.update(overrides)
    return {
        "achievement_rates": rates,
        "achievement_score": smoothed_geometric_mean(list(rates.values())),
    }
