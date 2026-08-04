"""Train isolated Stage 7C tasks to locate the full-probe learning bottleneck."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from voyager.training.civilization_learning_ladder import (
    ALL_LEARNING_TASKS,
    DELIVERY_DIAGNOSTIC_TASKS,
    LEARNING_TASK_CONTRACTS,
    LEARNING_TASKS,
    LearningTask,
    learning_task_definition,
)
from voyager.training.civilization_learning_ladder_evaluation import (
    diagnose_delivery_components,
    diagnose_learning_ladder,
    evaluate_learning_task,
    learning_task_gate,
    summarize_learning_task,
)
from voyager.training.environments import CIVILIZATION_V2_TRAINING_ENVIRONMENT
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats

DEFAULT_TRANSITIONS: dict[LearningTask, int] = {
    "gather_wood": 75_000,
    "gather_stone": 75_000,
    "return_to_camp": 75_000,
    "delivery": 100_000,
    "construction": 50_000,
    "survival": 100_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=(*ALL_LEARNING_TASKS, "all", "delivery_diagnostics"),
        default=["all"],
    )
    parser.add_argument(
        "--gather-wood-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["gather_wood"],
    )
    parser.add_argument(
        "--gather-stone-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["gather_stone"],
    )
    parser.add_argument(
        "--return-to-camp-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["return_to_camp"],
    )
    parser.add_argument(
        "--delivery-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["delivery"],
    )
    parser.add_argument(
        "--construction-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["construction"],
    )
    parser.add_argument(
        "--survival-transitions",
        type=int,
        default=DEFAULT_TRANSITIONS["survival"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=30_000)
    parser.add_argument("--rollout-world-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/learning_ladder_v1_seed0"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _selected_tasks(args.tasks)
    transitions: dict[LearningTask, int] = {
        "gather_wood": args.gather_wood_transitions,
        "gather_stone": args.gather_stone_transitions,
        "return_to_camp": args.return_to_camp_transitions,
        "delivery": args.delivery_transitions,
        "construction": args.construction_transitions,
        "survival": args.survival_transitions,
    }
    if any(transitions[task] <= 0 for task in tasks):
        raise ValueError("Every selected transition budget must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    task_summaries: dict[str, dict[str, object]] = {}

    print(
        "Stage 7C learning ladder: "
        + ", ".join(f"{task}={transitions[task]:,}" for task in tasks)
    )
    for task in tasks:
        print(f"\n[{task}] evaluating legal-random comparator...")
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        random_results = evaluate_learning_task(
            task=task,
            policy_name="legal_random",
            policy="legal_random",
            seeds=seeds,
        )
        random_summary = summarize_learning_task(random_results)
        _write_json(task_dir / "random.json", random_summary)

        config = PPOConfig(
            total_steps=transitions[task],
            environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
            reward_contract=LEARNING_TASK_CONTRACTS[task],
            rollout_steps=args.rollout_world_steps,
            num_agents=10,
            map_size=48,
            max_steps=learning_task_definition(task).horizon,
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
            checkpoint_dir=str(task_dir / "checkpoints"),
            checkpoint_every=0,
            reward_mode="none",
            use_action_mask=True,
        )
        _write_json(task_dir / "config.json", asdict(config))
        history_path = task_dir / "training_history.jsonl"
        history_path.unlink(missing_ok=True)
        try:
            trainer = PPOTrainer(config)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        def on_update(
            stats: PPOUpdateStats,
            *,
            task_name: LearningTask = task,
            selected_history_path: Path = history_path,
        ) -> None:
            _append_json_line(selected_history_path, asdict(stats))
            if stats.update == 1 or stats.update % 10 == 0:
                print(
                    f"[{task_name}] update={stats.update:04d} "
                    f"transitions={stats.agent_steps:,} "
                    f"reward={stats.mean_reward:+.4f} "
                    f"throughput={stats.agent_steps_per_second:.1f}/s"
                )

        training_stats = trainer.train(on_update=on_update)
        checkpoint = trainer.save_named_checkpoint(
            "final", training_stats[-1].update
        )
        timing = trainer.timing_report()
        _write_json(task_dir / "timing.json", timing)

        print(f"[{task}] evaluating trained policy in both inference modes...")
        deterministic_results = evaluate_learning_task(
            task=task,
            policy_name="feed_forward_ppo_deterministic",
            policy="model",
            seeds=seeds,
            model=trainer.model,
            deterministic=True,
        )
        stochastic_results = evaluate_learning_task(
            task=task,
            policy_name="feed_forward_ppo_seeded_stochastic",
            policy="model",
            seeds=seeds,
            model=trainer.model,
            deterministic=False,
        )
        deterministic_summary = summarize_learning_task(deterministic_results)
        stochastic_summary = summarize_learning_task(stochastic_results)
        gate = learning_task_gate(task, stochastic_summary, random_summary)
        task_summary: dict[str, object] = {
            "task": task,
            "contract": LEARNING_TASK_CONTRACTS[task],
            "primary_inference": "seeded_stochastic",
            "checkpoint": checkpoint,
            "learned": {
                "deterministic": deterministic_summary,
                "seeded_stochastic": stochastic_summary,
            },
            "random": random_summary,
            "gate": gate,
            "timing": timing,
        }
        task_summaries[task] = task_summary
        _write_json(task_dir / "summary.json", task_summary)
        print(
            f"[{task}] {'PASS' if gate['passed'] else 'FAIL'}: "
            f"learned success={_number(stochastic_summary['success_rate']):.1%} "
            f"score={_number(stochastic_summary['mean_score']):.3f}; "
            f"random success={_number(random_summary['success_rate']):.1%} "
            f"score={_number(random_summary['mean_score']):.3f}"
        )

    gates: dict[str, dict[str, object]] = {}
    for task_name, summary in task_summaries.items():
        gate_value = summary.get("gate")
        if isinstance(gate_value, dict):
            gates[task_name] = gate_value
    delivery_diagnosis = (
        diagnose_delivery_components(gates)
        if set(tasks) == set(DELIVERY_DIAGNOSTIC_TASKS)
        else None
    )
    diagnosis = None
    if set(tasks) == set(LEARNING_TASKS):
        diagnosis = diagnose_learning_ladder(gates)
    elif delivery_diagnosis is not None:
        diagnosis_value = delivery_diagnosis["diagnosis"]
        if isinstance(diagnosis_value, str):
            diagnosis = diagnosis_value
    overall = {
        "contract": "stage7c_learning_ladder_v1",
        "purpose": "diagnostic_only_not_an_official_benchmark_result",
        "tasks": list(tasks),
        "task_summaries": task_summaries,
        "all_selected_tasks_passed": all(
            bool(gate.get("passed", False)) for gate in gates.values()
        ),
        "diagnosis": diagnosis,
        "delivery_component_diagnosis": delivery_diagnosis,
    }
    _write_json(output_dir / "summary.json", overall)
    print("\nLearning ladder diagnosis: " + (diagnosis or "partial_run"))
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def _selected_tasks(values: list[str]) -> tuple[LearningTask, ...]:
    if "all" in values:
        return LEARNING_TASKS
    if "delivery_diagnostics" in values:
        return DELIVERY_DIAGNOSTIC_TASKS
    ordered: list[LearningTask] = []
    for task in ALL_LEARNING_TASKS:
        if task in values:
            ordered.append(task)
    return tuple(ordered)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_json_line(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected a numeric result.")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
