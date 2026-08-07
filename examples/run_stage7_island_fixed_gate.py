"""Train and evaluate the pre-procedural VoyagerIsland-v1 250K gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voyager.benchmark.island import ISLAND_DEV_SEEDS, ISLAND_TEST_SEEDS
from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT
from voyager.training.island_evaluation import (
    FeedForwardCheckpointIslandPolicy,
    FeedForwardModelIslandPolicy,
    LegalRandomIslandPolicy,
    evaluate_island_policy,
    fixed_island_trainability_gate,
    island_checkpoint_selection_key,
    normalize_island_evaluation_milestones,
)
from voyager.training.island_reward import ISLAND_TRAINING_REWARD_V4
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats

DEFAULT_MILESTONES = (50_000, 100_000, 150_000, 200_000, 250_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-agent-transitions", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--entropy-coef-start", type=float, default=0.02)
    parser.add_argument("--entropy-coef-end", type=float, default=0.005)
    parser.add_argument(
        "--evaluation-milestones",
        type=int,
        nargs="+",
        default=list(DEFAULT_MILESTONES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7/island_progression_v4_250k_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    config = PPOConfig(
        total_steps=args.total_agent_transitions,
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V4,
        rollout_steps=128,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        seed=args.seed,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=0,
        reward_mode="dense",
        use_action_mask=True,
        procedural=False,
    )
    _write_json(args.output_dir / "config.json", asdict(config))
    trainer = PPOTrainer(config)
    dev_seeds = ISLAND_DEV_SEEDS[: args.dev_episodes]
    test_seeds = ISLAND_TEST_SEEDS[: args.eval_episodes]
    milestones = normalize_island_evaluation_milestones(
        args.evaluation_milestones,
        total_agent_transitions=args.total_agent_transitions,
    )
    pending_milestones = list(milestones)
    development_evaluations: list[dict[str, object]] = []
    best_selection_key: tuple[float, float] | None = None
    best_checkpoint: str | None = None
    best_milestone: int | None = None

    print(
        "Stage 7 fixed-island checkpoint-selection gate: "
        f"transitions={args.total_agent_transitions:,} "
        f"dev_episodes={len(dev_seeds)} test_episodes={len(test_seeds)} "
        f"milestones={list(milestones)} "
        f"entropy={args.entropy_coef_start:g}->{args.entropy_coef_end:g}"
    )
    print("Evaluating the legal-random development comparator...")
    random_dev_results, random_dev_summary = evaluate_island_policy(
        lambda seed: LegalRandomIslandPolicy(seed),
        seeds=dev_seeds,
        procedural=False,
    )

    def on_update(update_stats: PPOUpdateStats) -> None:
        nonlocal best_selection_key, best_checkpoint, best_milestone
        _print_update(update_stats)
        while pending_milestones and update_stats.agent_steps >= pending_milestones[0]:
            milestone = pending_milestones.pop(0)
            checkpoint = trainer.save_named_checkpoint(
                f"agent_steps_{milestone:09d}",
                update_stats.update,
            )
            print(
                f"Evaluating milestone {milestone:,} "
                f"(actual transitions {update_stats.agent_steps:,})..."
            )
            evaluation = _evaluate_model(
                trainer.model,
                trainer.observation_encoder,
                checkpoint=checkpoint,
                seeds=dev_seeds,
                include_episodes=True,
            )
            stochastic_summary = _summary(evaluation, "stochastic")
            selection_key = island_checkpoint_selection_key(stochastic_summary)
            selected = best_selection_key is None or selection_key > best_selection_key
            if selected:
                best_selection_key = selection_key
                best_checkpoint = trainer.save_named_checkpoint(
                    "best",
                    update_stats.update,
                )
                best_milestone = milestone
            development_evaluations.append(
                {
                    "milestone_agent_transitions": milestone,
                    "actual_agent_transitions": update_stats.agent_steps,
                    "checkpoint": checkpoint,
                    "selection_mode": "seeded_stochastic",
                    "selection_key": {
                        "achievement_geometric_mean": selection_key[0],
                        "invalid_action_rate": -selection_key[1],
                    },
                    "became_best_so_far": selected,
                    "evaluation": evaluation,
                }
            )
            _write_json(
                args.output_dir / "development_evaluations.json",
                {
                    "contract": "voyager_island_checkpoint_selection_v1",
                    "seeds": list(dev_seeds),
                    "random": {
                        "summary": random_dev_summary,
                        "episodes": [row.as_dict() for row in random_dev_results],
                    },
                    "milestones": development_evaluations,
                    "best_milestone_agent_transitions": best_milestone,
                    "best_checkpoint": best_checkpoint,
                },
            )
            print(
                "dev stochastic score="
                f"{selection_key[0]:.3f} invalid={-selection_key[1]:.3%} "
                f"best={'YES' if selected else 'NO'}"
            )

    stats = trainer.train(on_update=on_update)
    history_path = args.output_dir / "training_history.jsonl"
    history_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in stats),
        encoding="utf-8",
    )
    if best_checkpoint is None or best_milestone is None:
        raise RuntimeError("Training completed without a checkpoint selection.")
    for row in development_evaluations:
        row["is_final_selection"] = row["milestone_agent_transitions"] == best_milestone
    _write_json(
        args.output_dir / "development_evaluations.json",
        {
            "contract": "voyager_island_checkpoint_selection_v1",
            "seeds": list(dev_seeds),
            "random": {
                "summary": random_dev_summary,
                "episodes": [row.as_dict() for row in random_dev_results],
            },
            "milestones": development_evaluations,
            "best_milestone_agent_transitions": best_milestone,
            "best_checkpoint": best_checkpoint,
        },
    )

    print(
        f"Selected milestone {best_milestone:,}; evaluating it once on "
        f"{len(test_seeds)} held-out test seeds..."
    )
    random_results, random_summary = evaluate_island_policy(
        lambda seed: LegalRandomIslandPolicy(seed),
        seeds=test_seeds,
        procedural=False,
    )
    selected_test = _evaluate_checkpoint(
        best_checkpoint,
        seeds=test_seeds,
        include_episodes=True,
    )
    learned_summary = _summary(selected_test, "stochastic")
    deterministic_summary = _summary(selected_test, "deterministic")
    gate = fixed_island_trainability_gate(learned_summary, random_summary)
    latest_checkpoint = checkpoint_dir / "latest"
    latest_test = _evaluate_checkpoint(
        latest_checkpoint,
        seeds=test_seeds,
        include_episodes=False,
    )
    reward_diagnostics = getattr(trainer.env, "reward_diagnostics", None)
    payload = {
        "contract": "voyager_island_fixed_trainability_gate_v4",
        "training_reward_contract": ISLAND_TRAINING_REWARD_V4,
        "selection": {
            "contract": "voyager_island_checkpoint_selection_v1",
            "mode": "seeded_stochastic",
            "primary_metric": "achievement_geometric_mean",
            "tie_breaker": "lower_invalid_action_rate",
            "development_seeds": list(dev_seeds),
            "test_seeds": list(test_seeds),
            "milestones_agent_transitions": list(milestones),
            "best_milestone_agent_transitions": best_milestone,
            "best_checkpoint": best_checkpoint,
            "test_results_never_used_for_selection": True,
        },
        "predecessor": {
            "contract": "voyager_island_fixed_trainability_gate_v3",
            "reward_contract": "voyager_island_progression_reward_v3",
            "artifact": "results/stage7/island_progression_v3_250k_seed0",
            "result": "failed",
            "diagnosis": "bounded_progression_credit_was_under_scaled",
        },
        "training": {
            "agent_transitions": args.total_agent_transitions,
            "world_steps": trainer.world_steps,
            "updates": len(stats),
            "entropy_schedule": {
                "start": args.entropy_coef_start,
                "end": args.entropy_coef_end,
            },
            "config_artifact": "config.json",
            "timing": trainer.timing_report(),
            "reward_diagnostics": (reward_diagnostics() if callable(reward_diagnostics) else {}),
            "history_artifact": history_path.name,
        },
        "random": random_summary,
        "selected_checkpoint": {
            "path": best_checkpoint,
            "milestone_agent_transitions": best_milestone,
        },
        "learned_stochastic": learned_summary,
        "learned_deterministic": deterministic_summary,
        "latest_checkpoint_diagnostic": latest_test,
        "gate": gate,
        "episodes": {
            "random": [result.as_dict() for result in random_results],
            "learned_stochastic": _episodes(selected_test, "stochastic"),
            "learned_deterministic": _episodes(selected_test, "deterministic"),
        },
    }
    _write_json(args.output_dir / "summary.json", payload)
    print(f"Stage 7 fixed-island gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"Continue to procedural calibration: {'YES' if gate['passed'] else 'NO'}")
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0


def _print_update(stats: PPOUpdateStats) -> None:
    if stats.update == 1 or stats.update % 10 == 0:
        print(
            f"update={stats.update:04d} transitions={stats.agent_steps:,} "
            f"reward={stats.mean_reward:+.4f} throughput={stats.agent_steps_per_second:.1f}/s"
        )


def _evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    seeds: tuple[int, ...],
    include_episodes: bool,
) -> dict[str, object]:
    stochastic_results, stochastic_summary = evaluate_island_policy(
        lambda seed: FeedForwardCheckpointIslandPolicy(
            checkpoint,
            deterministic=False,
            seed=seed,
        ),
        seeds=seeds,
        procedural=False,
    )
    deterministic_results, deterministic_summary = evaluate_island_policy(
        lambda seed: FeedForwardCheckpointIslandPolicy(
            checkpoint,
            deterministic=True,
            seed=seed,
        ),
        seeds=seeds,
        procedural=False,
    )
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "stochastic": {"summary": stochastic_summary},
        "deterministic": {"summary": deterministic_summary},
    }
    if include_episodes:
        stochastic = payload["stochastic"]
        deterministic = payload["deterministic"]
        assert isinstance(stochastic, dict)
        assert isinstance(deterministic, dict)
        stochastic["episodes"] = [row.as_dict() for row in stochastic_results]
        deterministic["episodes"] = [row.as_dict() for row in deterministic_results]
    return payload


def _evaluate_model(
    model: Any,
    observation_encoder: str,
    *,
    checkpoint: str,
    seeds: tuple[int, ...],
    include_episodes: bool,
) -> dict[str, object]:
    """Evaluate the live model without rebuilding it or advancing TensorFlow RNG."""

    stochastic_results, stochastic_summary = evaluate_island_policy(
        lambda seed: FeedForwardModelIslandPolicy(
            model,
            observation_encoder,
            deterministic=False,
            seed=seed,
        ),
        seeds=seeds,
        procedural=False,
    )
    deterministic_results, deterministic_summary = evaluate_island_policy(
        lambda seed: FeedForwardModelIslandPolicy(
            model,
            observation_encoder,
            deterministic=True,
            seed=seed,
        ),
        seeds=seeds,
        procedural=False,
    )
    payload: dict[str, object] = {
        "checkpoint": checkpoint,
        "stochastic": {"summary": stochastic_summary},
        "deterministic": {"summary": deterministic_summary},
    }
    if include_episodes:
        stochastic = payload["stochastic"]
        deterministic = payload["deterministic"]
        assert isinstance(stochastic, dict)
        assert isinstance(deterministic, dict)
        stochastic["episodes"] = [row.as_dict() for row in stochastic_results]
        deterministic["episodes"] = [row.as_dict() for row in deterministic_results]
    return payload


def _summary(evaluation: dict[str, object], mode: str) -> dict[str, object]:
    row = evaluation[mode]
    if not isinstance(row, dict):
        raise TypeError(f"Missing {mode} evaluation row.")
    summary = row["summary"]
    if not isinstance(summary, dict):
        raise TypeError(f"Missing {mode} evaluation summary.")
    return summary


def _episodes(evaluation: dict[str, object], mode: str) -> list[object]:
    row = evaluation[mode]
    if not isinstance(row, dict):
        raise TypeError(f"Missing {mode} evaluation row.")
    episodes = row.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"Missing {mode} evaluation episodes.")
    return episodes


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_agent_transitions <= 0:
        raise ValueError("--total-agent-transitions must be positive.")
    if not 0 < args.dev_episodes <= len(ISLAND_DEV_SEEDS):
        raise ValueError(f"--dev-episodes must be between 1 and {len(ISLAND_DEV_SEEDS)}.")
    if not 0 < args.eval_episodes <= len(ISLAND_TEST_SEEDS):
        raise ValueError(f"--eval-episodes must be between 1 and {len(ISLAND_TEST_SEEDS)}.")
    if args.entropy_coef_start < 0.0 or args.entropy_coef_end < 0.0:
        raise ValueError("Entropy coefficients must be non-negative.")
    if args.entropy_coef_start < args.entropy_coef_end:
        raise ValueError("--entropy-coef-start must be >= --entropy-coef-end.")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
