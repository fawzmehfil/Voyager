"""Train recurrent PPO directly on the complete 600-tick Stage 7C island."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    FeedForwardAchievementPolicy,
    LegalRandomAchievementPolicy,
    RecurrentAchievementPolicy,
    calibration_gate,
    evaluate_achievement_policy,
    summarize_achievement_results,
)
from voyager.training.environments import CIVILIZATION_V2_TRAINING_ENVIRONMENT
from voyager.training.obs import CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER
from voyager.training.ppo import PPOUpdateStats
from voyager.training.recurrent_ppo import (
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
)

DEFAULT_FEED_FORWARD_CHECKPOINT = Path(
    "results/stage7c/ppo_probe_v3_250k_seed0/checkpoints/best"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-agent-transitions", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feed-forward-checkpoint",
        type=Path,
        default=DEFAULT_FEED_FORWARD_CHECKPOINT,
    )
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-minibatch-size", type=int, default=16)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--dev-seed-start", type=int, default=40_000)
    parser.add_argument("--test-seed-start", type=int, default=50_000)
    parser.add_argument(
        "--evaluation-milestones",
        type=int,
        nargs="+",
        default=[250_000],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/recurrent_ppo_250k_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_agent_transitions <= 0:
        raise ValueError("--total-agent-transitions must be positive.")
    if args.dev_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("Evaluation episode counts must be positive.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feed_forward_model, feed_forward_metadata = load_policy_checkpoint(
        args.feed_forward_checkpoint
    )
    _validate_feed_forward_checkpoint(feed_forward_metadata)
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / "training_history.jsonl"
    history_path.unlink(missing_ok=True)
    config = RecurrentPPOConfig(
        total_steps=args.total_agent_transitions,
        rollout_steps=args.rollout_world_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        train_epochs=args.train_epochs,
        sequence_length=args.sequence_length,
        sequence_minibatch_size=args.sequence_minibatch_size,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=0,
    )
    _write_json(output_dir / "config.json", asdict(config))
    try:
        trainer = RecurrentPPOTrainer(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dev_seeds = list(
        range(args.dev_seed_start, args.dev_seed_start + args.dev_episodes)
    )
    test_seeds = list(
        range(args.test_seed_start, args.test_seed_start + args.test_episodes)
    )
    print(
        "Stage 7C recurrent PPO: "
        f"transitions={config.total_steps:,} observation={trainer.input_dim} "
        f"actions={trainer.action_count} sequence={config.sequence_length}"
    )
    print("Evaluating legal random on development seeds...")
    random_dev = summarize_achievement_results(
        evaluate_achievement_policy(
            policy=LegalRandomAchievementPolicy(),
            seeds=dev_seeds,
            inference_mode="seeded_stochastic",
        )
    )

    milestones = sorted(
        {
            min(value, config.total_steps)
            for value in args.evaluation_milestones
            if value > 0
        }
        | {config.total_steps}
    )
    pending_milestones = list(milestones)
    evaluations: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_invalid_rate = float("inf")
    best_checkpoint: str | None = None

    def on_update(stats: PPOUpdateStats) -> None:
        nonlocal best_score, best_invalid_rate, best_checkpoint
        _append_json_line(history_path, asdict(stats))
        if stats.update == 1 or stats.update % 10 == 0:
            print(
                f"update={stats.update:04d} transitions={stats.agent_steps:,} "
                f"reward={stats.mean_reward:+.4f} "
                f"throughput={stats.agent_steps_per_second:.1f}/s"
            )
        while pending_milestones and stats.agent_steps >= pending_milestones[0]:
            milestone = pending_milestones.pop(0)
            checkpoint = trainer.save_named_checkpoint(
                f"agent_steps_{milestone:09d}", stats.update
            )
            modes: dict[str, dict[str, object]] = {}
            for inference_mode in ("deterministic", "seeded_stochastic"):
                print(
                    f"Evaluating {milestone:,} transitions ({inference_mode})..."
                )
                results = evaluate_achievement_policy(
                    policy=RecurrentAchievementPolicy(
                        model=trainer.model,
                        hidden_size=config.recurrent_hidden_size,
                        inference_mode=inference_mode,
                    ),
                    seeds=dev_seeds,
                    inference_mode=inference_mode,
                )
                modes[inference_mode] = summarize_achievement_results(results)
            primary = modes["seeded_stochastic"]
            score = _number(primary["achievement_score"])
            invalid_rate = _number(primary["invalid_action_rate"])
            evaluations.append(
                {
                    "milestone_agent_transitions": milestone,
                    "actual_agent_transitions": stats.agent_steps,
                    "checkpoint": checkpoint,
                    "recurrent_ppo": modes,
                    "random": random_dev,
                }
            )
            _write_json(output_dir / "development_evaluations.json", evaluations)
            if score > best_score or (
                score == best_score and invalid_rate < best_invalid_rate
            ):
                best_score = score
                best_invalid_rate = invalid_rate
                best_checkpoint = trainer.save_named_checkpoint("best", stats.update)
            print(
                f"dev achievement_score={score:.3f} "
                f"random={_number(random_dev['achievement_score']):.3f} "
                f"invalid={invalid_rate:.3%}"
            )

    trainer.train(on_update=on_update)
    timing = trainer.timing_report()
    _write_json(output_dir / "timing.json", timing)
    if best_checkpoint is None:
        raise RuntimeError("Training completed without an achievement evaluation.")

    print(f"Evaluating best checkpoint on {len(test_seeds)} held-out seeds...")
    recurrent_model, recurrent_metadata = load_policy_checkpoint(best_checkpoint)
    random_test = summarize_achievement_results(
        evaluate_achievement_policy(
            policy=LegalRandomAchievementPolicy(),
            seeds=test_seeds,
            inference_mode="seeded_stochastic",
        )
    )
    feed_forward_test: dict[str, dict[str, object]] = {}
    recurrent_test: dict[str, dict[str, object]] = {}
    for inference_mode in ("deterministic", "seeded_stochastic"):
        feed_forward_test[inference_mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=FeedForwardAchievementPolicy(
                    model=feed_forward_model,
                    inference_mode=inference_mode,
                ),
                seeds=test_seeds,
                inference_mode=inference_mode,
            )
        )
        recurrent_test[inference_mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=RecurrentAchievementPolicy(
                    model=recurrent_model,
                    hidden_size=config.recurrent_hidden_size,
                    inference_mode=inference_mode,
                ),
                seeds=test_seeds,
                inference_mode=inference_mode,
            )
        )

    calibration = calibration_gate(
        random_summary=random_test,
        feed_forward_summary=feed_forward_test["seeded_stochastic"],
        recurrent_summary=recurrent_test["seeded_stochastic"],
    )
    summary = {
        "contract": "stage7c_recurrent_ppo_achievement_pilot_v1",
        "primary_inference": "seeded_stochastic",
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_metadata": recurrent_metadata,
        "feed_forward_checkpoint": str(args.feed_forward_checkpoint.resolve()),
        "feed_forward_checkpoint_metadata": feed_forward_metadata,
        "development_evaluations": evaluations,
        "held_out": {
            "random": random_test,
            "feed_forward_ppo": feed_forward_test,
            "recurrent_ppo": recurrent_test,
        },
        "calibration": calibration,
        "timing": timing,
    }
    _write_json(output_dir / "summary.json", summary)
    print("Held-out achievement scores (seeded stochastic):")
    print(f"  random:       {_number(random_test['achievement_score']):.3f}")
    print(
        "  feed-forward: "
        f"{_number(feed_forward_test['seeded_stochastic']['achievement_score']):.3f}"
    )
    print(
        "  recurrent:    "
        f"{_number(recurrent_test['seeded_stochastic']['achievement_score']):.3f}"
    )
    print(f"Next action: {calibration['next_action']}")
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Expected a numeric metric.")
    return float(value)


def _validate_feed_forward_checkpoint(metadata: dict[str, object]) -> None:
    if metadata.get("environment_id") != CIVILIZATION_V2_TRAINING_ENVIRONMENT:
        raise ValueError("Feed-forward checkpoint is not from VoyagerCivilization-v2.")
    if metadata.get("observation_encoder") != (
        CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER
    ):
        raise ValueError("Feed-forward checkpoint does not use the v3 actor observation.")
    if str(metadata.get("model_type", "feed_forward")) != "feed_forward":
        raise ValueError("--feed-forward-checkpoint must contain an MLP PPO policy.")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_json_line(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
