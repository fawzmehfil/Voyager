"""Outcome and behavior diagnostics for the Stage 7C trainability probe."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    unflatten_v2_action,
)
from voyager.training.civilization_probe import CivilizationProbeRewardWrapper
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.masking import stack_action_masks
from voyager.training.obs import flatten_observations

ProbePolicy = Literal["legal_random", "model"]
WORKBENCH_WOOD = 6
WORKBENCH_STONE = 2
PROBE_METRICS = (
    "gathered_workbench_bundle_by_100",
    "workbench_materials_available_by_300",
    "workbench_complete",
    "any_tool_crafted",
    "majority_active_at_300",
)
PROBE_THRESHOLDS = {
    "gathered_workbench_bundle_by_100": 0.50,
    "workbench_materials_available_by_300": 0.30,
    "workbench_complete": 0.20,
    "any_tool_crafted": 0.10,
    "majority_active_at_300": 0.80,
}


@dataclass(frozen=True, slots=True)
class CivilizationEpisodeResult:
    """One 600-tick episode measured independently of the training reward."""

    policy: str
    seed: int
    world_steps: int
    agent_transitions: int
    shared_return: float
    gathered_workbench_bundle_by_100: bool
    workbench_materials_available_by_300: bool
    workbench_complete: bool
    any_tool_crafted: bool
    majority_active_at_300: bool
    active_at_300: int
    final_active: int
    deaths: int
    invalid_actions: int
    submitted_actions: int
    gathered_wood_and_stone: bool = False
    deposited_wood_and_stone: bool = False
    gathered_counts: dict[str, int] = field(default_factory=dict)
    deposited_counts: dict[str, int] = field(default_factory=dict)
    peak_camp_stockpile: dict[str, int] = field(default_factory=dict)
    workbench_bundle_gather_tick: int | None = None
    workbench_completion_tick: int | None = None
    first_tool_tick: int | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    selected_verbs: dict[str, int] = field(default_factory=dict)
    selected_actions: dict[str, int] = field(default_factory=dict)
    total_agent_return: float = 0.0
    mean_agent_return: float = 0.0
    per_agent_returns: dict[str, float] = field(default_factory=dict)
    shared_reward_component_totals: dict[str, float] = field(default_factory=dict)
    individual_reward_component_totals: dict[str, float] = field(default_factory=dict)
    per_agent_individual_reward_totals: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    @property
    def composite(self) -> float:
        return float(
            np.mean([float(getattr(self, metric)) for metric in PROBE_METRICS])
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["composite"] = self.composite
        return payload


def run_civilization_episode(
    *,
    policy_name: str,
    policy: ProbePolicy,
    seed: int,
    model: Any | None = None,
    deterministic: bool = True,
    reward_contract: str = CIVILIZATION_PROBE_REWARD_CONTRACT,
) -> CivilizationEpisodeResult:
    """Run one handcrafted v2 episode with legal-random or model actions."""

    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=reward_contract,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    if not isinstance(env, CivilizationProbeRewardWrapper):
        raise TypeError("Stage 7C evaluation requires CivilizationProbeRewardWrapper.")
    if policy == "model" and model is None:
        raise ValueError("Model evaluation requires an actor-critic model.")
    selected_model = model

    rng = np.random.default_rng(seed)
    observations, infos = env.reset(seed=seed)
    shared_return = 0.0
    total_agent_return = 0.0
    per_agent_returns: dict[str, float] = {
        agent_id: 0.0 for agent_id in env.possible_agents
    }
    shared_reward_component_totals: Counter[str] = Counter()
    individual_reward_component_totals: Counter[str] = Counter()
    per_agent_individual_reward_totals: dict[str, Counter[str]] = {
        agent_id: Counter() for agent_id in env.possible_agents
    }
    agent_transitions = 0
    invalid_actions = 0
    submitted_actions = 0
    active_at_300 = 0
    gathered_by_100: Counter[str] = Counter()
    gathered_counts: Counter[str] = Counter()
    deposited_counts: Counter[str] = Counter()
    peak_camp_stockpile: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    selected_verbs: Counter[str] = Counter()
    selected_actions: Counter[str] = Counter()
    workbench_materials_available_by_300 = False
    workbench_bundle_gather_tick: int | None = None
    workbench_completion_tick: int | None = None
    first_tool_tick: int | None = None
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
            if selected_model is None:
                raise RuntimeError("Model policy was selected without a model.")
            flat_observations = flatten_observations(
                observations,
                agent_ids,
                training_environment.observation_encoder,
            )
            logits, _values = selected_model(flat_observations, training=False)
            logits_array = np.asarray(logits, dtype=np.float64)
            selected_logits = np.where(masks, logits_array, -1e9)
            if deterministic:
                selected = np.argmax(selected_logits, axis=1)
            else:
                selected = np.asarray(
                    [
                        rng.choice(
                            selected_logits.shape[1],
                            p=_probabilities(row),
                        )
                        for row in selected_logits
                    ],
                    dtype=np.int64,
                )
            actions = {
                agent_id: int(selected[index])
                for index, agent_id in enumerate(agent_ids)
            }

        for action in actions.values():
            verb_value, argument_value, target = unflatten_v2_action(action)
            verb = CivilizationV2Verb(verb_value).name.lower()
            argument = CivilizationV2Argument(argument_value).name.lower()
            selected_verbs[verb] += 1
            selected_actions[
                f"{verb}:{argument}:{'targeted' if target else 'untargeted'}"
            ] += 1

        observations, rewards, _terminations, _truncations, infos = env.step(actions)
        agent_transitions += len(agent_ids)
        submitted_actions += len(agent_ids)
        invalid_actions += sum(
            bool(infos[agent_id].get("invalid_action", False))
            for agent_id in agent_ids
        )
        if rewards:
            first_info = infos[agent_ids[0]]
            shared_return += float(
                first_info.get("shared_reward", next(iter(rewards.values())))
            )
            total_agent_return += sum(float(value) for value in rewards.values())
            for agent_id, value in rewards.items():
                per_agent_returns[agent_id] = (
                    per_agent_returns.get(agent_id, 0.0) + float(value)
                )
            shared_components = first_info.get("shared_reward_components", {})
            if isinstance(shared_components, dict):
                shared_reward_component_totals.update(
                    {
                        str(name): float(value)
                        for name, value in shared_components.items()
                    }
                )
            for agent_id in agent_ids:
                individual_components = infos[agent_id].get(
                    "individual_reward_components", {}
                )
                if not isinstance(individual_components, dict):
                    continue
                normalized_components = {
                    str(name): float(value)
                    for name, value in individual_components.items()
                }
                individual_reward_component_totals.update(normalized_components)
                per_agent_individual_reward_totals[agent_id].update(
                    normalized_components
                )
        state = env.world.state
        assert state is not None
        new_ledger = state.ledger[ledger_cursor:]
        ledger_cursor = len(state.ledger)
        for entry in new_ledger:
            event = str(entry.get("event", ""))
            item = str(entry.get("item", ""))
            quantity = _integer(entry.get("quantity", 0), "ledger quantity")
            if event == "gather" and item:
                gathered_counts[item] += quantity
                if state.step_count <= 100:
                    gathered_by_100[item] += quantity
            elif event == "deposit" and item:
                deposited_counts[item] += quantity
            elif event == "craft_tool" and first_tool_tick is None:
                first_tool_tick = state.step_count
        for event in state.events:
            if event.get("type") != "intent_rejected":
                continue
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                rejection_reasons[str(payload.get("reason", "unknown"))] += 1

        for item in ("food", "wood", "stone"):
            peak_camp_stockpile[item] = max(
                peak_camp_stockpile[item],
                state.camp.stockpile.get(item, 0),
            )
        if (
            workbench_bundle_gather_tick is None
            and gathered_counts["wood"] >= WORKBENCH_WOOD
            and gathered_counts["stone"] >= WORKBENCH_STONE
        ):
            workbench_bundle_gather_tick = state.step_count
        if (
            state.step_count <= 300
            and state.camp.stockpile.get("wood", 0) >= WORKBENCH_WOOD
            and state.camp.stockpile.get("stone", 0) >= WORKBENCH_STONE
        ):
            workbench_materials_available_by_300 = True
        if state.structures["workbench"].complete and workbench_completion_tick is None:
            workbench_completion_tick = state.step_count
        if state.step_count == 300:
            active_at_300 = sum(
                agent.life_state == "active" for agent in state.agents.values()
            )

    state = env.world.state
    assert state is not None
    result = CivilizationEpisodeResult(
        policy=policy_name,
        seed=seed,
        world_steps=state.step_count,
        agent_transitions=agent_transitions,
        shared_return=shared_return,
        gathered_workbench_bundle_by_100=(
            gathered_by_100["wood"] >= WORKBENCH_WOOD
            and gathered_by_100["stone"] >= WORKBENCH_STONE
        ),
        workbench_materials_available_by_300=workbench_materials_available_by_300,
        workbench_complete=state.structures["workbench"].complete,
        any_tool_crafted=first_tool_tick is not None,
        majority_active_at_300=active_at_300 >= 6,
        active_at_300=active_at_300,
        final_active=sum(
            agent.life_state == "active" for agent in state.agents.values()
        ),
        deaths=sum(agent.life_state == "dead" for agent in state.agents.values()),
        invalid_actions=invalid_actions,
        submitted_actions=submitted_actions,
        gathered_wood_and_stone=(
            gathered_counts["wood"] > 0 and gathered_counts["stone"] > 0
        ),
        deposited_wood_and_stone=(
            deposited_counts["wood"] > 0 and deposited_counts["stone"] > 0
        ),
        gathered_counts=dict(sorted(gathered_counts.items())),
        deposited_counts=dict(sorted(deposited_counts.items())),
        peak_camp_stockpile=dict(sorted(peak_camp_stockpile.items())),
        workbench_bundle_gather_tick=workbench_bundle_gather_tick,
        workbench_completion_tick=workbench_completion_tick,
        first_tool_tick=first_tool_tick,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        selected_verbs=dict(sorted(selected_verbs.items())),
        selected_actions=dict(sorted(selected_actions.items())),
        total_agent_return=total_agent_return,
        mean_agent_return=total_agent_return / len(env.possible_agents),
        per_agent_returns={
            agent_id: float(per_agent_returns[agent_id])
            for agent_id in env.possible_agents
        },
        shared_reward_component_totals=dict(
            sorted(shared_reward_component_totals.items())
        ),
        individual_reward_component_totals=dict(
            sorted(individual_reward_component_totals.items())
        ),
        per_agent_individual_reward_totals={
            agent_id: dict(sorted(values.items()))
            for agent_id, values in sorted(per_agent_individual_reward_totals.items())
        },
    )
    env.close()
    return result


def evaluate_civilization_policy(
    *,
    policy_name: str,
    policy: ProbePolicy,
    seeds: list[int],
    model: Any | None = None,
    deterministic: bool = True,
    reward_contract: str = CIVILIZATION_PROBE_REWARD_CONTRACT,
) -> list[CivilizationEpisodeResult]:
    """Evaluate one policy on a caller-owned deterministic seed set."""

    return [
        run_civilization_episode(
            policy_name=policy_name,
            policy=policy,
            seed=seed,
            model=model,
            deterministic=deterministic,
            reward_contract=reward_contract,
        )
        for seed in seeds
    ]


def summarize_civilization_results(
    results: list[CivilizationEpisodeResult],
) -> dict[str, object]:
    """Aggregate predeclared capability rates and behavioral diagnostics."""

    if not results:
        raise ValueError("At least one episode result is required.")
    submitted = sum(result.submitted_actions for result in results)
    return {
        "policy": results[0].policy,
        "episodes": len(results),
        "seeds": [result.seed for result in results],
        "composite": float(np.mean([result.composite for result in results])),
        "capability_rates": {
            metric: float(np.mean([float(getattr(result, metric)) for result in results]))
            for metric in PROBE_METRICS
        },
        "legacy_diagnostic_rates": {
            metric: float(np.mean([float(getattr(result, metric)) for result in results]))
            for metric in ("gathered_wood_and_stone", "deposited_wood_and_stone")
        },
        "mean_active_at_300": float(
            np.mean([result.active_at_300 for result in results])
        ),
        "mean_final_active": float(np.mean([result.final_active for result in results])),
        "invalid_action_rate": sum(result.invalid_actions for result in results)
        / max(1, submitted),
        "rejection_reason_rates": {
            reason: count / max(1, submitted)
            for reason, count in _sum_counters(
                result.rejection_reasons for result in results
            ).items()
        },
        "selected_verb_rates": {
            verb: count / max(1, submitted)
            for verb, count in _sum_counters(
                result.selected_verbs for result in results
            ).items()
        },
        "mean_gathered_counts": _mean_counters(
            [result.gathered_counts for result in results]
        ),
        "mean_deposited_counts": _mean_counters(
            [result.deposited_counts for result in results]
        ),
        "mean_shared_return": float(
            np.mean([result.shared_return for result in results])
        ),
        "mean_total_agent_return": float(
            np.mean([result.total_agent_return for result in results])
        ),
        "mean_agent_return": float(
            np.mean([result.mean_agent_return for result in results])
        ),
        "mean_shared_reward_components": _mean_numeric_maps(
            [result.shared_reward_component_totals for result in results]
        ),
        "mean_individual_reward_components": _mean_numeric_maps(
            [result.individual_reward_component_totals for result in results]
        ),
        "episodes_detail": [result.as_dict() for result in results],
    }


def compare_against_random(
    learned: list[CivilizationEpisodeResult],
    random: list[CivilizationEpisodeResult],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Return paired composite confidence bounds and the Stage 7C pass decision."""

    learned_by_seed = {result.seed: result for result in learned}
    random_by_seed = {result.seed: result for result in random}
    if learned_by_seed.keys() != random_by_seed.keys():
        raise ValueError("Learned and random results must use identical seeds.")
    ordered_seeds = sorted(learned_by_seed)
    differences = np.asarray(
        [
            learned_by_seed[item].composite - random_by_seed[item].composite
            for item in ordered_seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(bootstrap_samples, len(differences)),
    )
    bootstrapped_means = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(bootstrapped_means, (0.025, 0.975))
    learned_summary = summarize_civilization_results(learned)
    capability_rates = learned_summary["capability_rates"]
    assert isinstance(capability_rates, dict)
    capability_passes = {
        metric: float(capability_rates[metric]) >= threshold
        for metric, threshold in PROBE_THRESHOLDS.items()
    }
    difference = float(np.mean(differences))
    return {
        "paired_seeds": ordered_seeds,
        "composite_difference": difference,
        "composite_difference_ci95": [float(lower), float(upper)],
        "composite_gate_passed": difference >= 0.15 and float(lower) > 0.0,
        "capability_thresholds": dict(PROBE_THRESHOLDS),
        "capability_gate_passed": capability_passes,
        "overall_passed": difference >= 0.15
        and float(lower) > 0.0
        and all(capability_passes.values()),
    }


def pilot_continuation(
    deterministic_summary: dict[str, object],
    deterministic_comparison: dict[str, object],
    stochastic_summary: dict[str, object],
    stochastic_comparison: dict[str, object],
) -> dict[str, object]:
    """Apply the deterministic-primary 250K continuation contract."""

    deterministic = _continuation_checks(
        deterministic_summary,
        deterministic_comparison,
    )
    stochastic = _continuation_checks(
        stochastic_summary,
        stochastic_comparison,
    )
    deterministic_continue = all(deterministic.values())
    stochastic_continue = all(stochastic.values())
    return {
        "contract": "stage7c_probe_v3_250k_continuation_v1",
        "primary_inference": "deterministic",
        "checks": deterministic,
        "continue": deterministic_continue,
        "seeded_stochastic_diagnostic": {
            "checks": stochastic,
            "would_continue": stochastic_continue,
        },
        "failure_mode": (
            "deterministic_coordination_collapse"
            if stochastic_continue and not deterministic_continue
            else None
        ),
        "note": "Seeded stochastic inference is diagnostic and cannot authorize continuation.",
    }


def _continuation_checks(
    learned_summary: dict[str, object],
    comparison: dict[str, object],
) -> dict[str, bool]:
    rates = learned_summary.get("capability_rates")
    if not isinstance(rates, dict):
        raise TypeError("capability_rates must be a dictionary.")
    return {
        "composite_above_random": _float_value(
            comparison.get("composite_difference"),
            "composite_difference",
        )
        > 0.0,
        "gathers_bundle_by_100": _float_value(
            rates.get("gathered_workbench_bundle_by_100"),
            "gathered_workbench_bundle_by_100",
        )
        >= 0.50,
        "nonzero_camp_or_later_progress": any(
            _float_value(rates.get(name), name) > 0.0
            for name in (
                "workbench_materials_available_by_300",
                "workbench_complete",
                "any_tool_crafted",
            )
        ),
        "invalid_action_rate_below_10_percent": _float_value(
            learned_summary.get("invalid_action_rate"),
            "invalid_action_rate",
        )
        < 0.10,
    }


def _sum_counters(rows: Any) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update(row)
    return result


def _mean_counters(rows: list[dict[str, int]]) -> dict[str, float]:
    totals = _sum_counters(rows)
    return {key: value / len(rows) for key, value in sorted(totals.items())}


def _mean_numeric_maps(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = {key for row in rows for key in row}
    return {
        key: float(np.mean([row.get(key, 0.0) for row in rows]))
        for key in sorted(keys)
    }


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _float_value(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable_logits = logits - float(np.max(logits))
    probabilities = np.exp(stable_logits)
    return probabilities / np.sum(probabilities)
