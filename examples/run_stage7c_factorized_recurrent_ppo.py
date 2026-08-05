"""Run the final Stage 7C factorized recurrent PPO composition test."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    FactorizedAchievementPolicy,
    FactorizedRecurrentAchievementPolicy,
    InferenceMode,
    LegalRandomAchievementPolicy,
    compare_achievement_summaries,
    delivery_emergence,
    evaluate_achievement_policy,
    summarize_achievement_results,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
)
from voyager.training.factorized_recurrent_ppo import (
    FactorizedRecurrentPPOTrainer,
)
from voyager.training.obs import CIVILIZATION_V4_TEAM_OBJECTIVE_OBSERVATION_ENCODER
from voyager.training.ppo import PPOUpdateStats
from voyager.training.recurrent_ppo import RecurrentPPOConfig

DEFAULT_FACTORIZED_CHECKPOINT = Path(
    "results/stage7c/factorized_ppo_v1_250k_seed0/checkpoints/final"
)
INFERENCE_MODES: tuple[InferenceMode, ...] = (
    "deterministic",
    "seeded_stochastic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-agent-transitions", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--factorized-feed-forward-checkpoint",
        type=Path,
        default=DEFAULT_FACTORIZED_CHECKPOINT,
    )
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-minibatch-size", type=int, default=16)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--dev-seed-start", type=int, default=60_000)
    parser.add_argument("--test-seed-start", type=int, default=70_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/factorized_recurrent_v1_250k_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feed_forward_model, feed_forward_metadata = load_policy_checkpoint(
        args.factorized_feed_forward_checkpoint
    )
    _validate_feed_forward_checkpoint(feed_forward_metadata)

    dev_seeds = list(range(args.dev_seed_start, args.dev_seed_start + args.dev_episodes))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_episodes))
    config = RecurrentPPOConfig(
        total_steps=args.total_agent_transitions,
        rollout_steps=args.rollout_world_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef_start=0.02,
        entropy_coef_end=0.005,
        value_coef=0.5,
        train_epochs=args.train_epochs,
        sequence_length=args.sequence_length,
        sequence_minibatch_size=args.sequence_minibatch_size,
        encoder_sizes=(128,),
        recurrent_hidden_size=128,
        max_gradient_norm=0.5,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_every=0,
        reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    )
    experiment = {
        "contract": "stage7c_factorized_recurrent_composition_test_v1",
        "total_agent_transitions": args.total_agent_transitions,
        "seed": args.seed,
        "dev_seeds": dev_seeds,
        "test_seeds": test_seeds,
        "primary_inference": "seeded_stochastic",
        "minimum_repeatable_delivery_rate": 0.20,
        "factorized_feed_forward_checkpoint": str(
            args.factorized_feed_forward_checkpoint.resolve()
        ),
    }
    _write_json(output_dir / "experiment_config.json", experiment)
    _write_json(output_dir / "config.json", asdict(config))
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    print(
        "Stage 7C final composition test: factorized recurrent PPO, "
        f"{config.total_steps:,} agent transitions"
    )
    try:
        trainer = FactorizedRecurrentPPOTrainer(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if trainer.input_dim != 610:
        raise ValueError(
            f"V4 factorized recurrent actor requires 610 values, got {trainer.input_dim}."
        )
    print(
        f"Contract: observation={trainer.input_dim}, public_actions={trainer.action_count}, "
        f"sequence={config.sequence_length}, memory={config.recurrent_hidden_size}"
    )
    stats = trainer.train(on_update=_update_logger(history_path))
    checkpoint = trainer.save_named_checkpoint("final", update=len(stats))
    _write_json(output_dir / "timing.json", trainer.timing_report())

    print("\nEvaluating development seeds...")
    recurrent_dev = _evaluate_recurrent(
        trainer.model,
        config.recurrent_hidden_size,
        dev_seeds,
    )
    feed_forward_dev = _evaluate_feed_forward(feed_forward_model, dev_seeds)
    development = {
        "factorized_feed_forward": feed_forward_dev,
        "factorized_recurrent": recurrent_dev,
    }
    _write_json(output_dir / "development.json", development)

    print(f"\nEvaluating {len(test_seeds)} held-out seeds...")
    recurrent_model, recurrent_metadata = load_policy_checkpoint(checkpoint)
    random_test = _evaluate_random(test_seeds)
    feed_forward_test = _evaluate_feed_forward(feed_forward_model, test_seeds)
    recurrent_test = _evaluate_recurrent(
        recurrent_model,
        config.recurrent_hidden_size,
        test_seeds,
    )
    diagnosis = _diagnose(
        random_summary=random_test,
        feed_forward=feed_forward_test["seeded_stochastic"],
        recurrent_dev=recurrent_dev["seeded_stochastic"],
        recurrent_test=recurrent_test["seeded_stochastic"],
    )
    summary = {
        **experiment,
        "checkpoint": checkpoint,
        "checkpoint_metadata": recurrent_metadata,
        "factorized_feed_forward_checkpoint_metadata": feed_forward_metadata,
        "development": development,
        "held_out": {
            "legal_random": random_test,
            "factorized_feed_forward": feed_forward_test,
            "factorized_recurrent": recurrent_test,
        },
        "diagnosis": diagnosis,
        "timing": trainer.timing_report(),
    }
    _write_json(output_dir / "summary.json", summary)

    primary = recurrent_test["seeded_stochastic"]
    print("Held-out achievement scores:")
    print(f"  legal random:            {_number(random_test['achievement_score']):.3f}")
    print(
        "  factorized feed-forward: "
        f"{_number(feed_forward_test['seeded_stochastic']['achievement_score']):.3f}"
    )
    print(f"  factorized recurrent:    {_number(primary['achievement_score']):.3f}")
    print(
        "Repeatable delivery on development and held-out seeds: "
        + ("YES" if diagnosis["repeatable_delivery_on_both_sets"] else "NO")
    )
    print(f"Next action: {diagnosis['next_action']}")
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _evaluate_random(seeds: list[int]) -> dict[str, object]:
    print("Evaluating legal-random comparator...")
    return summarize_achievement_results(
        evaluate_achievement_policy(
            policy=LegalRandomAchievementPolicy(),
            seeds=seeds,
            inference_mode="seeded_stochastic",
            reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        )
    )


def _evaluate_feed_forward(
    model: Any,
    seeds: list[int],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for mode in INFERENCE_MODES:
        print(f"Evaluating factorized feed-forward PPO ({mode})...")
        results[mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=FactorizedAchievementPolicy(
                    model=model,
                    inference_mode=mode,
                ),
                seeds=seeds,
                inference_mode=mode,
                reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
            )
        )
    return results


def _evaluate_recurrent(
    model: Any,
    hidden_size: int,
    seeds: list[int],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for mode in INFERENCE_MODES:
        print(f"Evaluating factorized recurrent PPO ({mode})...")
        results[mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=FactorizedRecurrentAchievementPolicy(
                    model=model,
                    hidden_size=hidden_size,
                    inference_mode=mode,
                ),
                seeds=seeds,
                inference_mode=mode,
                reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
            )
        )
    return results


def _diagnose(
    *,
    random_summary: dict[str, object],
    feed_forward: dict[str, object],
    recurrent_dev: dict[str, object],
    recurrent_test: dict[str, object],
) -> dict[str, object]:
    development_delivery = delivery_emergence(recurrent_dev)
    held_out_delivery = delivery_emergence(recurrent_test)
    repeatable_both = bool(
        development_delivery["repeatable_delivery_emerged"]
        and held_out_delivery["repeatable_delivery_emerged"]
    )
    versus_random = compare_achievement_summaries(recurrent_test, random_summary)
    versus_feed_forward = compare_achievement_summaries(recurrent_test, feed_forward)
    useful_separation = bool(
        repeatable_both
        and versus_random["score_above_baseline"]
        and versus_random["meaningful_progression_above_baseline"]
    )
    return {
        "development_delivery": development_delivery,
        "held_out_delivery": held_out_delivery,
        "repeatable_delivery_on_both_sets": repeatable_both,
        "factorized_recurrent_vs_random": versus_random,
        "factorized_recurrent_vs_feed_forward": versus_feed_forward,
        "useful_baseline_separation": useful_separation,
        "mappo_authorized": False,
        "next_action": (
            "replicate_factorized_recurrent_across_training_seeds"
            if useful_separation
            else "stop_algorithm_stacking_simplify_task_or_interface"
        ),
    }


def _validate_feed_forward_checkpoint(metadata: dict[str, object]) -> None:
    if metadata.get("environment_id") != CIVILIZATION_V2_TRAINING_ENVIRONMENT:
        raise ValueError("Checkpoint is not from VoyagerCivilization-v2.")
    if metadata.get("observation_encoder") != (CIVILIZATION_V4_TEAM_OBJECTIVE_OBSERVATION_ENCODER):
        raise ValueError("Checkpoint does not use the 610-value v4 observation.")
    if metadata.get("model_type") != "factorized_feed_forward":
        raise ValueError("Checkpoint must contain factorized feed-forward PPO.")


def _update_logger(history_path: Path) -> Any:
    def on_update(stats: PPOUpdateStats) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(stats), sort_keys=True) + "\n")
        if stats.update == 1 or stats.update % 10 == 0:
            print(
                f"[factorized-recurrent] update={stats.update:04d} "
                f"transitions={stats.agent_steps:,} "
                f"reward={stats.mean_reward:+.4f} "
                f"throughput={stats.agent_steps_per_second:.1f}/s"
            )

    return on_update


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_agent_transitions <= 0:
        raise ValueError("--total-agent-transitions must be positive.")
    if args.dev_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("Evaluation episode counts must be positive.")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Expected a numeric metric.")
    return float(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
