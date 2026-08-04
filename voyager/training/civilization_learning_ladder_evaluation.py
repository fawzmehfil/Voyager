"""Evaluation and diagnosis for the Stage 7C isolated learning tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Verb,
    unflatten_v2_action,
)
from voyager.training.civilization_learning_ladder import (
    LEARNING_TASK_CONTRACTS,
    CivilizationLearningTaskWrapper,
    LearningTask,
)
from voyager.training.environments import (
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.masking import stack_action_masks
from voyager.training.obs import flatten_observations

DiagnosticPolicy = Literal["legal_random", "model"]


@dataclass(frozen=True, slots=True)
class LearningTaskEpisodeResult:
    """Outcome of one task episode, independent of its training reward."""

    task: LearningTask
    policy: str
    seed: int
    task_steps: int
    agent_transitions: int
    success: bool
    score: float
    completion_step: int | None
    invalid_actions: int
    submitted_actions: int
    final_active: int
    peak_camp_stockpile: dict[str, int] = field(default_factory=dict)
    gathered_counts: dict[str, int] = field(default_factory=dict)
    deposited_counts: dict[str, int] = field(default_factory=dict)
    selected_verbs: dict[str, int] = field(default_factory=dict)
    total_agent_return: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_learning_task_episode(
    *,
    task: LearningTask,
    policy_name: str,
    policy: DiagnosticPolicy,
    seed: int,
    model: Any | None = None,
    deterministic: bool = False,
) -> LearningTaskEpisodeResult:
    """Run one isolated task using legal-random or a supplied actor model."""

    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=LEARNING_TASK_CONTRACTS[task],
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    if not isinstance(env, CivilizationLearningTaskWrapper):
        raise TypeError("Learning-task evaluation requires its diagnostic wrapper.")
    if policy == "model" and model is None:
        raise ValueError("Model evaluation requires an actor-critic model.")

    rng = np.random.default_rng(seed)
    observations, infos = env.reset(seed=seed)
    start_tick = env.task_definition.start_tick
    invalid_actions = 0
    submitted_actions = 0
    agent_transitions = 0
    total_agent_return = 0.0
    completion_step: int | None = None
    peak_camp = Counter[str]()
    gathered = Counter[str]()
    deposited = Counter[str]()
    selected_verbs = Counter[str]()
    ledger_cursor = 0

    while env.agents:
        agent_ids = tuple(env.agents)
        masks = stack_action_masks(infos, agent_ids, V2_FLAT_ACTION_COUNT)
        if policy == "legal_random":
            actions = {
                agent_id: int(rng.choice(np.flatnonzero(masks[index])))
                for index, agent_id in enumerate(agent_ids)
            }
        else:
            if model is None:
                raise RuntimeError("Model policy was selected without a model.")
            flat = flatten_observations(
                observations,
                agent_ids,
                training_environment.observation_encoder,
            )
            logits, _values = model(flat, training=False)
            selected_logits = np.where(masks, np.asarray(logits), -1e9)
            if deterministic:
                selected = np.argmax(selected_logits, axis=1)
            else:
                selected = np.asarray(
                    [
                        rng.choice(len(row), p=_probabilities(row))
                        for row in selected_logits
                    ],
                    dtype=np.int64,
                )
            actions = {
                agent_id: int(selected[index])
                for index, agent_id in enumerate(agent_ids)
            }

        for action in actions.values():
            verb, _argument, _target = unflatten_v2_action(action)
            selected_verbs[CivilizationV2Verb(verb).name.lower()] += 1
        observations, rewards, _terms, _truncs, infos = env.step(actions)
        submitted_actions += len(agent_ids)
        agent_transitions += len(agent_ids)
        total_agent_return += sum(float(value) for value in rewards.values())
        invalid_actions += sum(
            bool(infos[agent_id].get("invalid_action", False))
            for agent_id in agent_ids
        )
        state = env.world.state
        assert state is not None
        for entry in state.ledger[ledger_cursor:]:
            event = str(entry.get("event", ""))
            item = str(entry.get("item", ""))
            quantity = entry.get("quantity", 0)
            if not isinstance(quantity, int) or not item:
                continue
            if event == "gather":
                gathered[item] += quantity
            elif event == "deposit":
                deposited[item] += quantity
        ledger_cursor = len(state.ledger)
        for item in ("wood", "stone"):
            peak_camp[item] = max(
                peak_camp[item], state.camp.stockpile.get(item, 0)
            )
        if completion_step is None and env.success():
            completion_step = state.step_count - start_tick

    state = env.world.state
    assert state is not None
    result = LearningTaskEpisodeResult(
        task=task,
        policy=policy_name,
        seed=seed,
        task_steps=state.step_count - start_tick,
        agent_transitions=agent_transitions,
        success=env.success(),
        score=env.score(),
        completion_step=completion_step,
        invalid_actions=invalid_actions,
        submitted_actions=submitted_actions,
        final_active=sum(
            agent.life_state == "active" for agent in state.agents.values()
        ),
        peak_camp_stockpile=dict(sorted(peak_camp.items())),
        gathered_counts=dict(sorted(gathered.items())),
        deposited_counts=dict(sorted(deposited.items())),
        selected_verbs=dict(sorted(selected_verbs.items())),
        total_agent_return=total_agent_return,
    )
    env.close()
    return result


def evaluate_learning_task(
    *,
    task: LearningTask,
    policy_name: str,
    policy: DiagnosticPolicy,
    seeds: list[int],
    model: Any | None = None,
    deterministic: bool = False,
) -> list[LearningTaskEpisodeResult]:
    """Evaluate a policy on a caller-owned seed set."""

    return [
        run_learning_task_episode(
            task=task,
            policy_name=policy_name,
            policy=policy,
            seed=seed,
            model=model,
            deterministic=deterministic,
        )
        for seed in seeds
    ]


def summarize_learning_task(
    results: list[LearningTaskEpisodeResult],
) -> dict[str, object]:
    """Aggregate task success, efficiency, validity, and behavior."""

    if not results:
        raise ValueError("At least one learning-task result is required.")
    submitted = sum(result.submitted_actions for result in results)
    completed = [
        result.completion_step
        for result in results
        if result.completion_step is not None
    ]
    verb_counts = Counter[str]()
    for result in results:
        verb_counts.update(result.selected_verbs)
    return {
        "task": results[0].task,
        "policy": results[0].policy,
        "episodes": len(results),
        "seeds": [result.seed for result in results],
        "success_rate": float(np.mean([result.success for result in results])),
        "mean_score": float(np.mean([result.score for result in results])),
        "mean_completion_step": (
            None if not completed else float(np.mean(completed))
        ),
        "invalid_action_rate": sum(result.invalid_actions for result in results)
        / max(1, submitted),
        "mean_final_active": float(
            np.mean([result.final_active for result in results])
        ),
        "mean_gathered_counts": _mean_counters(
            [result.gathered_counts for result in results]
        ),
        "mean_deposited_counts": _mean_counters(
            [result.deposited_counts for result in results]
        ),
        "selected_verb_rates": {
            key: value / max(1, submitted)
            for key, value in sorted(verb_counts.items())
        },
        "episodes_detail": [result.as_dict() for result in results],
    }


def learning_task_gate(
    task: LearningTask,
    learned: dict[str, object],
    random: dict[str, object],
) -> dict[str, object]:
    """Return an interpretable diagnostic gate, not a benchmark score."""

    success = _number(learned["success_rate"])
    score = _number(learned["mean_score"])
    random_score = _number(random["mean_score"])
    invalid = _number(learned["invalid_action_rate"])
    if task in {"gather_wood", "gather_stone"}:
        learned_step = learned["mean_completion_step"]
        random_step = random["mean_completion_step"]
        faster = isinstance(learned_step, int | float) and (
            random_step is None
            or (
                isinstance(random_step, int | float)
                and float(learned_step) <= 0.80 * float(random_step)
            )
        )
        checks = {
            "success_rate_at_least_80_percent": success >= 0.80,
            "completion_at_least_20_percent_faster_than_random": faster,
            "invalid_rate_below_10_percent": invalid < 0.10,
        }
        capability_learned = success >= 0.50 and (
            score - random_score >= 0.15 or faster
        )
    elif task in {"return_to_camp", "delivery"}:
        checks = {
            "success_rate_at_least_50_percent": success >= 0.50,
            "score_beats_random_by_0.15": score - random_score >= 0.15,
            "invalid_rate_below_10_percent": invalid < 0.10,
        }
        capability_learned = checks["success_rate_at_least_50_percent"] and checks[
            "score_beats_random_by_0.15"
        ]
    elif task == "construction":
        learned_step = learned["mean_completion_step"]
        random_step = random["mean_completion_step"]
        faster = isinstance(learned_step, int | float) and (
            random_step is None
            or (
                isinstance(random_step, int | float)
                and float(learned_step) <= 0.80 * float(random_step)
            )
        )
        checks = {
            "success_rate_at_least_80_percent": success >= 0.80,
            "completion_at_least_20_percent_faster_than_random": faster,
            "invalid_rate_below_10_percent": invalid < 0.10,
        }
        capability_learned = checks["success_rate_at_least_80_percent"]
    else:
        checks = {
            "majority_survival_rate_at_least_80_percent": success >= 0.80,
            "active_fraction_not_worse_than_random_by_more_than_0.05": (
                score >= random_score - 0.05
            ),
            "invalid_rate_below_10_percent": invalid < 0.10,
        }
        capability_learned = checks[
            "majority_survival_rate_at_least_80_percent"
        ] and checks[
            "active_fraction_not_worse_than_random_by_more_than_0.05"
        ]
    return {
        "task": task,
        "checks": checks,
        "passed": all(checks.values()),
        "capability_learned": capability_learned,
        "score_difference_from_random": score - random_score,
        "purpose": "diagnostic_only_not_an_official_benchmark_result",
    }


def diagnose_learning_ladder(task_gates: dict[str, dict[str, object]]) -> str:
    """Map the first failed isolated skill to the next engineering decision."""

    passed = {
        task: bool(task_gates.get(task, {}).get("passed", False))
        for task in ("delivery", "construction", "survival")
    }
    if not passed["construction"]:
        return "primitive_action_or_optimizer_failure"
    if not passed["delivery"]:
        return "navigation_exploration_or_delayed_credit_failure"
    if not passed["survival"]:
        return "survival_control_or_threat_response_failure"
    return "isolated_skills_trainable_full_task_composition_failure"


def diagnose_delivery_components(
    task_gates: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Explain which component prevents the gather-and-return sequence."""

    failed = [
        task
        for task in ("gather_wood", "gather_stone", "return_to_camp")
        if not bool(
            task_gates.get(task, {}).get(
                "capability_learned",
                task_gates.get(task, {}).get("passed", False),
            )
        )
    ]
    efficiency_issues = [
        task
        for task in ("gather_wood", "gather_stone", "return_to_camp")
        if task not in failed
        and not bool(task_gates.get(task, {}).get("passed", False))
    ]
    if not failed:
        diagnosis = "component_skills_trainable_combination_or_team_state_failure"
    elif "return_to_camp" in failed:
        diagnosis = "home_navigation_or_delayed_deposit_credit_failure"
    else:
        diagnosis = "resource_search_or_acquisition_failure"
    return {
        "diagnosis": diagnosis,
        "failed_components": failed,
        "efficiency_issues": efficiency_issues,
        "next_step": (
            "build a curriculum that composes the passed skills"
            if not failed
            else "remediate only the failed component before composing delivery"
        ),
    }


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable = logits.astype(np.float64) - float(np.max(logits))
    probabilities = np.exp(stable)
    return probabilities / np.sum(probabilities)


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected a numeric diagnostic value.")
    return float(value)


def _mean_counters(rows: list[dict[str, int]]) -> dict[str, float]:
    totals = Counter[str]()
    for row in rows:
        totals.update(row)
    return {key: value / len(rows) for key, value in sorted(totals.items())}
