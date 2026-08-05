"""Train and diagnose the Stage 7C v4 factorized feed-forward PPO baseline."""

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
    FeedForwardAchievementPolicy,
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
from voyager.training.factorized_actions import (
    FACTOR_ARGUMENT_COUNT,
    FACTOR_TARGET_COUNT,
    FACTOR_VERB_COUNT,
)
from voyager.training.factorized_ppo import FactorizedPPOTrainer
from voyager.training.ppo import PPOConfig, PPOUpdateStats

INFERENCE_MODES: tuple[InferenceMode, ...] = (
    "deterministic",
    "seeded_stochastic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-agent-transitions", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--dev-seed-start", type=int, default=60_000)
    parser.add_argument("--test-seed-start", type=int, default=70_000)
    parser.add_argument(
        "--atomic-checkpoint",
        type=Path,
        default=Path("results/stage7c/team_objective_v4_250k_seed0/feed_forward/checkpoints/final"),
        help="Prior v4 atomic PPO checkpoint; omitted from comparison if absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/factorized_ppo_v1_250k_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_seeds = list(range(args.dev_seed_start, args.dev_seed_start + args.dev_episodes))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_episodes))
    experiment = {
        "contract": "stage7c_factorized_feed_forward_v1",
        "reward_contract": CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        "total_agent_transitions": args.total_agent_transitions,
        "seed": args.seed,
        "dev_seeds": dev_seeds,
        "test_seeds": test_seeds,
        "factor_counts": {
            "verbs": FACTOR_VERB_COUNT,
            "arguments": FACTOR_ARGUMENT_COUNT,
            "targets": FACTOR_TARGET_COUNT,
        },
        "primary_inference": "seeded_stochastic",
    }
    _write_json(output_dir / "experiment_config.json", experiment)

    print(f"Stage 7C factorized PPO pilot: {args.total_agent_transitions:,} agent transitions")
    print("Policy: verb -> valid argument -> valid target -> public flat action")
    try:
        trainer, update_count = _train(args, output_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    checkpoint = trainer.save_named_checkpoint("final", update=update_count)

    print("\nEvaluating final policy on development seeds...")
    development = _evaluate_factorized(trainer.model, dev_seeds)
    _write_json(output_dir / "development.json", development)

    print(f"\nEvaluating on {len(test_seeds)} held-out seeds...")
    model, metadata = load_policy_checkpoint(checkpoint)
    factorized_test = _evaluate_factorized(model, test_seeds)
    random_test = _evaluate_random(test_seeds)
    atomic_test = _evaluate_atomic_if_available(args.atomic_checkpoint, test_seeds)
    primary = factorized_test["seeded_stochastic"]
    diagnosis = _diagnose(
        factorized=primary,
        random_summary=random_test,
        atomic=atomic_test["seeded_stochastic"] if atomic_test else None,
    )
    summary = {
        **experiment,
        "checkpoint": checkpoint,
        "checkpoint_metadata": metadata,
        "development": development,
        "held_out": {
            "legal_random": random_test,
            "atomic_feed_forward_v4": atomic_test,
            "factorized_feed_forward_v4": factorized_test,
        },
        "diagnosis": diagnosis,
        "timing": trainer.timing_report(),
    }
    _write_json(output_dir / "summary.json", summary)

    print("Held-out achievement scores:")
    print(f"  legal random:          {_number(random_test['achievement_score']):.3f}")
    if atomic_test is not None:
        print(
            "  atomic feed-forward:   "
            f"{_number(atomic_test['seeded_stochastic']['achievement_score']):.3f}"
        )
    print(f"  factorized feed-forward: {_number(primary['achievement_score']):.3f}")
    print(
        "Resource-return behavior emerged: "
        + ("YES" if diagnosis["resource_return_emerged"] else "NO")
    )
    print(f"Next action: {diagnosis['next_action']}")
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _train(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[FactorizedPPOTrainer, int]:
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    config = PPOConfig(
        total_steps=args.total_agent_transitions,
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
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_every=0,
        reward_mode="none",
        use_action_mask=True,
    )
    _write_json(output_dir / "config.json", asdict(config))
    trainer = FactorizedPPOTrainer(config)
    if trainer.input_dim != 610:
        raise ValueError(f"V4 factorized actor must receive 610 values, got {trainer.input_dim}.")
    print(
        f"Contract: observation={trainer.input_dim}, public_actions={trainer.action_count}, "
        f"heads={FACTOR_VERB_COUNT}+{FACTOR_ARGUMENT_COUNT}+{FACTOR_TARGET_COUNT}"
    )
    stats = trainer.train(on_update=_update_logger(history_path))
    _write_json(output_dir / "timing.json", trainer.timing_report())
    return trainer, len(stats)


def _evaluate_factorized(
    model: Any,
    seeds: list[int],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for mode in INFERENCE_MODES:
        print(f"Evaluating factorized PPO ({mode})...")
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


def _evaluate_atomic_if_available(
    checkpoint: Path,
    seeds: list[int],
) -> dict[str, dict[str, object]] | None:
    if not (checkpoint / "metadata.json").is_file():
        print(f"Prior atomic checkpoint not found; skipping: {checkpoint}")
        return None
    print(f"Evaluating prior atomic v4 checkpoint: {checkpoint}")
    model, metadata = load_policy_checkpoint(checkpoint)
    if metadata.get("model_type") != "feed_forward":
        raise ValueError("--atomic-checkpoint must be a feed-forward PPO checkpoint.")
    results: dict[str, dict[str, object]] = {}
    for mode in INFERENCE_MODES:
        results[mode] = summarize_achievement_results(
            evaluate_achievement_policy(
                policy=FeedForwardAchievementPolicy(
                    model=model,
                    inference_mode=mode,
                    name="atomic_feed_forward_ppo_v4",
                ),
                seeds=seeds,
                inference_mode=mode,
                reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
            )
        )
    return results


def _diagnose(
    *,
    factorized: dict[str, object],
    random_summary: dict[str, object],
    atomic: dict[str, object] | None,
) -> dict[str, object]:
    rates = _numeric_map(factorized.get("achievement_rates"))
    delivery = delivery_emergence(factorized)
    deposited = bool(delivery["repeatable_material_deposit"])
    arrived = bool(delivery["repeatable_return_arrival"])
    camp_or_progression = any(
        rates[name] > 0.0
        for name in (
            "assemble_camp_bundle",
            "start_workbench",
            "complete_workbench",
        )
    )
    versus_random = compare_achievement_summaries(factorized, random_summary)
    versus_atomic = (
        compare_achievement_summaries(factorized, atomic) if atomic is not None else None
    )
    return_emerged = deposited or arrived
    if camp_or_progression and bool(versus_random["score_above_baseline"]):
        next_action = "factorized_baseline_separates_run_replication_seeds"
    elif return_emerged:
        next_action = "train_factorized_recurrent_ppo"
    else:
        next_action = "factorization_failed_reassess_navigation_before_mappo"
    return {
        "resource_deposit_emerged": deposited,
        "resource_return_arrival_emerged": arrived,
        "resource_return_emerged": return_emerged,
        "delivery_emergence": delivery,
        "camp_or_workbench_progress_emerged": camp_or_progression,
        "factorized_vs_random": versus_random,
        "factorized_vs_atomic": versus_atomic,
        "next_action": next_action,
    }


def _update_logger(history_path: Path) -> Any:
    def on_update(stats: PPOUpdateStats) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(stats), sort_keys=True) + "\n")
        if stats.update == 1 or stats.update % 10 == 0:
            print(
                f"[factorized] update={stats.update:04d} "
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


def _numeric_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError("Expected a dictionary of numeric metrics.")
    return {str(key): _number(item) for key, item in value.items()}


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
