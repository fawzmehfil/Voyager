"""Run the Stage 7C feed-forward PPO trainability and throughput gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_evaluation import (
    compare_against_random,
    evaluate_civilization_policy,
    pilot_continuation,
    summarize_civilization_results,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
)
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats

DEFAULT_MILESTONES = (250_000, 500_000, 1_000_000, 1_500_000, 2_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-agent-transitions", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/ppo_probe_v3_seed0"),
    )
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=50)
    parser.add_argument("--dev-seed-start", type=int, default=10_000)
    parser.add_argument("--test-seed-start", type=int, default=20_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--evaluation-milestones",
        type=int,
        nargs="+",
        default=list(DEFAULT_MILESTONES),
    )
    parser.add_argument(
        "--stochastic-evaluation",
        action="store_true",
        help="Deprecated: v3 always records deterministic and seeded-stochastic inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_agent_transitions <= 0:
        raise ValueError("--total-agent-transitions must be positive.")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / "training_history.jsonl"
    history_path.unlink(missing_ok=True)

    config = PPOConfig(
        total_steps=args.total_agent_transitions,
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
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
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=0,
        reward_mode="none",
        use_action_mask=True,
    )
    _write_json(output_dir / "config.json", asdict(config))

    try:
        trainer = PPOTrainer(config)
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
        "Stage 7C probe: "
        f"transitions={config.total_steps:,} observation={trainer.input_dim} "
        f"actions={trainer.action_count} dev_seeds={len(dev_seeds)} "
        f"test_seeds={len(test_seeds)}"
    )
    print("Evaluating the seed-matched legal-random development comparator...")
    random_dev = evaluate_civilization_policy(
        policy_name="legal_random",
        policy="legal_random",
        seeds=dev_seeds,
    )
    random_dev_summary = summarize_civilization_results(random_dev)
    _write_json(output_dir / "random_dev.json", random_dev_summary)

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
    best_composite = float("-inf")
    best_invalid_rate = float("inf")
    best_checkpoint: str | None = None

    def on_update(stats: PPOUpdateStats) -> None:
        nonlocal best_composite, best_invalid_rate, best_checkpoint
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
                f"agent_steps_{milestone:09d}",
                stats.update,
            )
            print(
                f"Evaluating milestone {milestone:,} "
                f"(actual transitions {stats.agent_steps:,})..."
            )
            learned_deterministic = evaluate_civilization_policy(
                policy_name="feed_forward_ppo",
                policy="model",
                seeds=dev_seeds,
                model=trainer.model,
                deterministic=True,
            )
            learned_stochastic = evaluate_civilization_policy(
                policy_name="feed_forward_ppo_seeded_stochastic",
                policy="model",
                seeds=dev_seeds,
                model=trainer.model,
                deterministic=False,
            )
            deterministic_summary = summarize_civilization_results(
                learned_deterministic
            )
            stochastic_summary = summarize_civilization_results(learned_stochastic)
            deterministic_comparison = compare_against_random(
                learned_deterministic,
                random_dev,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            stochastic_comparison = compare_against_random(
                learned_stochastic,
                random_dev,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            evaluation = {
                "milestone_agent_transitions": milestone,
                "actual_agent_transitions": stats.agent_steps,
                "checkpoint": checkpoint,
                "learned": {
                    "deterministic": deterministic_summary,
                    "seeded_stochastic": stochastic_summary,
                },
                "random": random_dev_summary,
                "comparison": {
                    "deterministic": deterministic_comparison,
                    "seeded_stochastic": stochastic_comparison,
                },
            }
            evaluations.append(evaluation)
            _write_json(output_dir / "development_evaluations.json", evaluations)
            composite = _float_value(
                deterministic_summary["composite"], "composite"
            )
            invalid_rate = _float_value(
                deterministic_summary["invalid_action_rate"],
                "invalid_action_rate",
            )
            if composite > best_composite or (
                composite == best_composite and invalid_rate < best_invalid_rate
            ):
                best_composite = composite
                best_invalid_rate = invalid_rate
                best_checkpoint = trainer.save_named_checkpoint("best", stats.update)
            print(
                f"dev composite={composite:.3f} "
                "difference="
                f"{_float_value(deterministic_comparison['composite_difference'], 'difference'):+.3f} "
                f"invalid={invalid_rate:.3%} "
                f"rates={deterministic_summary['capability_rates']}"
            )
            print(
                "seeded-stochastic "
                f"composite={_float_value(stochastic_summary['composite'], 'composite'):.3f} "
                "difference="
                f"{_float_value(stochastic_comparison['composite_difference'], 'difference'):+.3f} "
                "invalid="
                f"{_float_value(stochastic_summary['invalid_action_rate'], 'invalid_action_rate'):.3%}"
            )

    trainer.train(on_update=on_update)
    timing = trainer.timing_report()
    _write_json(output_dir / "timing.json", timing)
    if best_checkpoint is None:
        raise RuntimeError("Training completed without a milestone evaluation.")

    print(f"Evaluating best checkpoint on {len(test_seeds)} held-out seeds...")
    best_model, best_metadata = load_policy_checkpoint(best_checkpoint)
    learned_test_deterministic = evaluate_civilization_policy(
        policy_name="feed_forward_ppo",
        policy="model",
        seeds=test_seeds,
        model=best_model,
        deterministic=True,
    )
    learned_test_stochastic = evaluate_civilization_policy(
        policy_name="feed_forward_ppo_seeded_stochastic",
        policy="model",
        seeds=test_seeds,
        model=best_model,
        deterministic=False,
    )
    random_test = evaluate_civilization_policy(
        policy_name="legal_random",
        policy="legal_random",
        seeds=test_seeds,
    )
    deterministic_comparison = compare_against_random(
        learned_test_deterministic,
        random_test,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    stochastic_comparison = compare_against_random(
        learned_test_stochastic,
        random_test,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    deterministic_summary = summarize_civilization_results(
        learned_test_deterministic
    )
    stochastic_summary = summarize_civilization_results(learned_test_stochastic)
    random_test_summary = summarize_civilization_results(random_test)
    continuation = pilot_continuation(
        deterministic_summary,
        deterministic_comparison,
        stochastic_summary,
        stochastic_comparison,
    )
    summary = {
        "contract": "stage7c_handcrafted_trainability_gate_v3",
        "evaluation_inference": ["deterministic", "seeded_stochastic"],
        "primary_inference": "deterministic",
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_metadata": best_metadata,
        "development_evaluations": evaluations,
        "held_out": {
            "learned": {
                "deterministic": deterministic_summary,
                "seeded_stochastic": stochastic_summary,
            },
            "random": random_test_summary,
            "comparison": {
                "deterministic": deterministic_comparison,
                "seeded_stochastic": stochastic_comparison,
            },
        },
        "pilot_continuation": continuation,
        "timing": timing,
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        "Stage 7C gate: "
        + ("PASS" if deterministic_comparison["overall_passed"] else "FAIL")
    )
    print(
        "Continue beyond the 250K pilot: "
        + ("YES" if continuation["continue"] else "NO")
    )
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_json_line(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _float_value(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
