"""Run the paired Stage 7C v4 feed-forward and recurrent PPO calibration pilot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    FeedForwardAchievementPolicy,
    InferenceMode,
    LegalRandomAchievementPolicy,
    RecurrentAchievementPolicy,
    calibration_gate,
    evaluate_achievement_policy,
    summarize_achievement_results,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
)
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats
from voyager.training.recurrent_ppo import (
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
)

Algorithm = Literal["feed_forward", "recurrent"]
INFERENCE_MODES: tuple[InferenceMode, ...] = (
    "deterministic",
    "seeded_stochastic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions-per-policy", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=256)
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
        default=Path("results/stage7c/team_objective_v4_250k_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_seeds = list(
        range(args.dev_seed_start, args.dev_seed_start + args.dev_episodes)
    )
    test_seeds = list(
        range(args.test_seed_start, args.test_seed_start + args.test_episodes)
    )
    experiment_config = {
        "contract": "stage7c_team_objective_v4_paired_pilot",
        "reward_contract": CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        "transitions_per_policy": args.transitions_per_policy,
        "seed": args.seed,
        "dev_seeds": dev_seeds,
        "test_seeds": test_seeds,
        "primary_inference": "seeded_stochastic",
    }
    _write_json(output_dir / "experiment_config.json", experiment_config)

    print(
        "Stage 7C v4 paired pilot: "
        f"{args.transitions_per_policy:,} transitions per learned policy"
    )
    print("Evaluating legal random on development seeds...")
    random_dev = _evaluate_random(dev_seeds)

    print("\nTraining feed-forward PPO on the complete v4 environment...")
    try:
        feed_forward_trainer = _train_feed_forward(args, output_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    feed_forward_checkpoint = feed_forward_trainer.save_named_checkpoint(
        "final", update=len(_history_rows(output_dir / "feed_forward" / "history.jsonl"))
    )
    feed_forward_dev = _evaluate_feed_forward(
        feed_forward_trainer.model, dev_seeds
    )
    _write_json(output_dir / "feed_forward" / "development.json", feed_forward_dev)

    print("\nTraining recurrent PPO on the complete v4 environment...")
    try:
        recurrent_trainer = _train_recurrent(args, output_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    recurrent_checkpoint = recurrent_trainer.save_named_checkpoint(
        "final", update=len(_history_rows(output_dir / "recurrent" / "history.jsonl"))
    )
    recurrent_dev = _evaluate_recurrent(
        recurrent_trainer.model,
        recurrent_trainer.config.recurrent_hidden_size,
        dev_seeds,
    )
    _write_json(output_dir / "recurrent" / "development.json", recurrent_dev)

    print(f"\nEvaluating both final checkpoints on {len(test_seeds)} held-out seeds...")
    feed_forward_model, feed_forward_metadata = load_policy_checkpoint(
        feed_forward_checkpoint
    )
    recurrent_model, recurrent_metadata = load_policy_checkpoint(
        recurrent_checkpoint
    )
    random_test = _evaluate_random(test_seeds)
    feed_forward_test = _evaluate_feed_forward(feed_forward_model, test_seeds)
    recurrent_hidden_size = _positive_int(
        recurrent_metadata.get("recurrent_hidden_size"),
        "recurrent_hidden_size",
    )
    recurrent_test = _evaluate_recurrent(
        recurrent_model,
        recurrent_hidden_size,
        test_seeds,
    )
    calibration = calibration_gate(
        random_summary=random_test,
        feed_forward_summary=feed_forward_test["seeded_stochastic"],
        recurrent_summary=recurrent_test["seeded_stochastic"],
    )
    summary = {
        **experiment_config,
        "feed_forward_checkpoint": feed_forward_checkpoint,
        "feed_forward_checkpoint_metadata": feed_forward_metadata,
        "recurrent_checkpoint": recurrent_checkpoint,
        "recurrent_checkpoint_metadata": recurrent_metadata,
        "development": {
            "random": random_dev,
            "feed_forward_ppo": feed_forward_dev,
            "recurrent_ppo": recurrent_dev,
        },
        "held_out": {
            "random": random_test,
            "feed_forward_ppo": feed_forward_test,
            "recurrent_ppo": recurrent_test,
        },
        "calibration": calibration,
        "timing": {
            "feed_forward": feed_forward_trainer.timing_report(),
            "recurrent": recurrent_trainer.timing_report(),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print("Held-out achievement scores (seeded stochastic):")
    print(f"  legal random: {_number(random_test['achievement_score']):.3f}")
    print(
        "  feed-forward: "
        f"{_number(feed_forward_test['seeded_stochastic']['achievement_score']):.3f}"
    )
    print(
        "  recurrent:    "
        f"{_number(recurrent_test['seeded_stochastic']['achievement_score']):.3f}"
    )
    print(
        "Useful learned baseline separation: "
        + (
            "YES"
            if calibration["at_least_one_learned_baseline_exceeds_random"]
            else "NO"
        )
    )
    print(f"Next action: {calibration['next_action']}")
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _train_feed_forward(args: argparse.Namespace, output_dir: Path) -> PPOTrainer:
    policy_dir = output_dir / "feed_forward"
    policy_dir.mkdir(parents=True, exist_ok=True)
    history_path = policy_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    config = PPOConfig(
        total_steps=args.transitions_per_policy,
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        rollout_steps=args.rollout_world_steps,
        num_agents=10,
        map_size=48,
        max_steps=600,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef_start=0.02,
        entropy_coef_end=0.005,
        value_coef=0.5,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
        hidden_sizes=(128, 128),
        checkpoint_dir=str(policy_dir / "checkpoints"),
        checkpoint_every=0,
        reward_mode="none",
        use_action_mask=True,
    )
    _write_json(policy_dir / "config.json", asdict(config))
    trainer = PPOTrainer(config)
    _print_contract("feed_forward", trainer.input_dim, trainer.action_count)
    trainer.train(on_update=_update_logger("feed_forward", history_path))
    _write_json(policy_dir / "timing.json", trainer.timing_report())
    return trainer


def _train_recurrent(
    args: argparse.Namespace, output_dir: Path
) -> RecurrentPPOTrainer:
    policy_dir = output_dir / "recurrent"
    policy_dir.mkdir(parents=True, exist_ok=True)
    history_path = policy_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    config = RecurrentPPOConfig(
        total_steps=args.transitions_per_policy,
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
        checkpoint_dir=str(policy_dir / "checkpoints"),
        checkpoint_every=0,
        reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    )
    _write_json(policy_dir / "config.json", asdict(config))
    trainer = RecurrentPPOTrainer(config)
    _print_contract("recurrent", trainer.input_dim, trainer.action_count)
    trainer.train(on_update=_update_logger("recurrent", history_path))
    _write_json(policy_dir / "timing.json", trainer.timing_report())
    return trainer


def _evaluate_random(seeds: list[int]) -> dict[str, object]:
    return summarize_achievement_results(
        evaluate_achievement_policy(
            policy=LegalRandomAchievementPolicy(),
            seeds=seeds,
            inference_mode="seeded_stochastic",
            reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        )
    )


def _evaluate_feed_forward(
    model: Any, seeds: list[int]
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for mode in INFERENCE_MODES:
        print(f"Evaluating feed-forward PPO ({mode})...")
        results[mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=FeedForwardAchievementPolicy(
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
        print(f"Evaluating recurrent PPO ({mode})...")
        results[mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=RecurrentAchievementPolicy(
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


def _update_logger(
    algorithm: Algorithm,
    history_path: Path,
) -> Any:
    def on_update(stats: PPOUpdateStats) -> None:
        _append_json_line(history_path, asdict(stats))
        if stats.update == 1 or stats.update % 10 == 0:
            print(
                f"[{algorithm}] update={stats.update:04d} "
                f"transitions={stats.agent_steps:,} "
                f"reward={stats.mean_reward:+.4f} "
                f"throughput={stats.agent_steps_per_second:.1f}/s"
            )

    return on_update


def _print_contract(algorithm: Algorithm, input_dim: int, action_count: int) -> None:
    if input_dim != 610:
        raise ValueError(f"V4 {algorithm} actor must receive 610 values, got {input_dim}.")
    print(f"[{algorithm}] observation={input_dim} actions={action_count}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.transitions_per_policy <= 0:
        raise ValueError("--transitions-per-policy must be positive.")
    if args.dev_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("Evaluation episode counts must be positive.")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Expected a numeric metric.")
    return float(value)


def _history_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
