"""Reward-independent evaluation utilities for VoyagerIsland-v1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from voyager.envs.island import VoyagerIslandEnv
from voyager.sim.constants import Resource
from voyager.sim.island_achievements import (
    ISLAND_ACHIEVEMENTS,
    achievement_success_rates,
    geometric_mean_score,
    grouped_scores,
)
from voyager.sim.island_registry import IslandAction
from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.masking import mask_numpy_logits
from voyager.training.obs import flatten_observation


class IslandPolicy(Protocol):
    """Minimal decentralized policy interface used by benchmark evaluation."""

    def reset(self, possible_agents: tuple[str, ...]) -> None: ...

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class IslandEpisodeResult:
    seed: int
    achievements: tuple[str, ...]
    achievement_steps: dict[str, int]
    joint_survival: bool
    rescue_success: bool
    invalid_actions: int
    actions: int
    agent_returns: dict[str, float]
    final_active_agents: int
    resource_waste: dict[str, int]
    contributions: dict[str, object]
    action_counts: dict[str, int]
    action_counts_by_agent: dict[str, dict[str, int]]
    carrier_diagnostics: dict[str, dict[str, int | None]]
    conservation_errors: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class LegalRandomIslandPolicy:
    """Uniformly sample from the same legal-action mask available to learners."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        _ = possible_agents

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        _ = infos
        actions: dict[str, int] = {}
        for agent_id, observation in observations.items():
            legal = np.flatnonzero(np.asarray(observation["action_mask"], dtype=np.int8))
            actions[agent_id] = int(self.rng.choice(legal))
        return actions


class FeedForwardModelIslandPolicy:
    """Run an in-memory shared PPO actor for side-effect-free evaluation."""

    def __init__(
        self,
        model: Any,
        observation_encoder: str,
        *,
        deterministic: bool,
        seed: int,
    ) -> None:
        self.model = model
        self.encoder = observation_encoder
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        _ = possible_agents

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        _ = infos
        agent_ids = tuple(observations)
        rows = np.stack(
            [flatten_observation(observations[agent_id], self.encoder) for agent_id in agent_ids]
        )
        logits, _values = self.model(rows, training=False)
        actions: dict[str, int] = {}
        for index, agent_id in enumerate(agent_ids):
            masked = mask_numpy_logits(
                np.asarray(logits[index], dtype=np.float32),
                np.asarray(observations[agent_id]["action_mask"], dtype=np.int8),
            )
            if self.deterministic:
                action = int(np.argmax(masked))
            else:
                shifted = masked - np.max(masked)
                probabilities = np.exp(shifted)
                probabilities /= np.sum(probabilities)
                action = int(self.rng.choice(len(probabilities), p=probabilities))
            actions[agent_id] = action
        return actions


class FeedForwardCheckpointIslandPolicy(FeedForwardModelIslandPolicy):
    """Run a saved shared PPO actor under stochastic or deterministic inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        deterministic: bool,
        seed: int,
    ) -> None:
        model, metadata = load_policy_checkpoint(checkpoint)
        if metadata.get("environment_id") != "VoyagerIsland-v1":
            raise ValueError("Checkpoint was not trained for VoyagerIsland-v1.")
        if metadata.get("model_type") != "feed_forward":
            raise ValueError("FeedForwardCheckpointIslandPolicy requires a feed-forward model.")
        self.metadata = metadata
        super().__init__(
            model,
            str(metadata["observation_encoder"]),
            deterministic=deterministic,
            seed=seed,
        )


@dataclass(slots=True)
class RecurrentModelIslandPolicy:
    """Run an in-memory shared recurrent PPO actor with one state per agent."""

    model: Any
    encoder: str
    hidden_size: int
    deterministic: bool
    seed: int
    rng: np.random.Generator = field(init=False)
    states: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        self.states = {
            agent_id: np.zeros((self.hidden_size,), dtype=np.float32)
            for agent_id in possible_agents
        }

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        _ = infos
        agent_ids = tuple(observations)
        rows = np.stack(
            [flatten_observation(observations[agent_id], self.encoder) for agent_id in agent_ids]
        )
        initial_states = np.stack([self.states[agent_id] for agent_id in agent_ids])
        model = self.model
        logits, _values, final_states = model([rows[:, None, :], initial_states], training=False)
        logits_array = np.asarray(logits, dtype=np.float32)[:, 0, :]
        final_states_array = np.asarray(final_states, dtype=np.float32)
        actions: dict[str, int] = {}
        for index, agent_id in enumerate(agent_ids):
            self.states[agent_id] = final_states_array[index]
            masked = mask_numpy_logits(
                logits_array[index],
                np.asarray(observations[agent_id]["action_mask"], dtype=np.int8),
            )
            if self.deterministic:
                action = int(np.argmax(masked))
            else:
                shifted = masked - np.max(masked)
                probabilities = np.exp(shifted)
                probabilities /= np.sum(probabilities)
                action = int(self.rng.choice(len(probabilities), p=probabilities))
            actions[agent_id] = action
        return actions


class RecurrentCheckpointIslandPolicy(RecurrentModelIslandPolicy):
    """Load and run a saved shared recurrent PPO actor."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        deterministic: bool,
        seed: int,
    ) -> None:
        model, metadata = load_policy_checkpoint(checkpoint)
        if metadata.get("environment_id") != "VoyagerIsland-v1":
            raise ValueError("Checkpoint was not trained for VoyagerIsland-v1.")
        if metadata.get("model_type") != "recurrent_gru":
            raise ValueError("RecurrentCheckpointIslandPolicy requires a recurrent model.")
        self.checkpoint = checkpoint
        self.metadata = metadata
        super().__init__(
            model=model,
            encoder=str(metadata["observation_encoder"]),
            hidden_size=int(cast(Any, metadata["recurrent_hidden_size"])),
            deterministic=deterministic,
            seed=seed,
        )


def run_island_episode(
    policy: IslandPolicy,
    *,
    seed: int,
    procedural: bool,
) -> IslandEpisodeResult:
    """Run one complete episode and collect only outcome-level benchmark metrics."""

    env = VoyagerIslandEnv(procedural=procedural)
    observations, infos = env.reset(seed=seed)
    policy.reset(tuple(env.possible_agents))
    returns = {agent_id: 0.0 for agent_id in env.possible_agents}
    invalid_actions = 0
    actions = 0
    action_counts: Counter[str] = Counter()
    action_counts_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    carrier: dict[str, dict[str, int | None]] = {
        agent_id: {
            "ticks": 0,
            "homeward_moves": 0,
            "outward_moves": 0,
            "stationary_moves": 0,
            "camp_opportunities": 0,
            "deposit_selections": 0,
            "successful_deposits": 0,
            "minimum_camp_distance": None,
        }
        for agent_id in env.possible_agents
    }
    while env.agents:
        selected = policy.act(observations, infos)
        state = env.world.state
        assert state is not None
        before: dict[str, tuple[bool, int]] = {}
        for agent_id, selected_action in selected.items():
            action = IslandAction(selected_action)
            action_name = action.name.lower()
            action_counts[action_name] += 1
            action_counts_by_agent[agent_id][action_name] += 1
            agent = state.agents[agent_id]
            carrying = (agent.inventory.get("wood", 0) + agent.inventory.get("stone", 0)) > 0
            distance = abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y)
            before[agent_id] = (carrying, distance)
            if carrying:
                row = carrier[agent_id]
                row["ticks"] = int(row["ticks"] or 0) + 1
                minimum = row["minimum_camp_distance"]
                row["minimum_camp_distance"] = (
                    distance if minimum is None else min(int(minimum), distance)
                )
                if distance <= 1:
                    row["camp_opportunities"] = int(row["camp_opportunities"] or 0) + 1
                if action == IslandAction.DEPOSIT_ALL:
                    row["deposit_selections"] = int(row["deposit_selections"] or 0) + 1
        observations, rewards, _terminations, _truncations, step_infos = env.step(selected)
        actions += len(rewards)
        for agent_id, reward in rewards.items():
            returns[agent_id] += float(reward)
            invalid_actions += int(bool(step_infos[agent_id].get("invalid_action", False)))
            carrying_before, distance_before = before[agent_id]
            if carrying_before and state.agents[agent_id].alive:
                agent = state.agents[agent_id]
                distance_after = abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y)
                key = (
                    "homeward_moves"
                    if distance_after < distance_before
                    else "outward_moves"
                    if distance_after > distance_before
                    else "stationary_moves"
                )
                carrier[agent_id][key] = int(carrier[agent_id][key] or 0) + 1
            if step_infos[agent_id].get("event") == "deposit_all":
                carrier[agent_id]["successful_deposits"] = (
                    int(carrier[agent_id]["successful_deposits"] or 0) + 1
                )
        infos.update(step_infos)
    state = env.world.state
    assert state is not None
    metrics = env.metrics()
    contributions = metrics.get("contributions", {})
    result = IslandEpisodeResult(
        seed=seed,
        achievements=tuple(name for name in ISLAND_ACHIEVEMENTS if name in state.achievements),
        achievement_steps=dict(state.achievement_steps),
        joint_survival=all(agent.alive for agent in state.agents.values()),
        rescue_success=state.rescue_success,
        invalid_actions=invalid_actions,
        actions=actions,
        agent_returns=returns,
        final_active_agents=len(env.world.alive_agents()),
        resource_waste={
            resource.name.lower(): int(
                np.sum(state.resource_quantities[state.resource_ids == resource])
            )
            for resource in Resource
            if resource != Resource.NONE
        },
        contributions=dict(contributions) if isinstance(contributions, Mapping) else {},
        action_counts=dict(sorted(action_counts.items())),
        action_counts_by_agent={
            agent_id: dict(sorted(counts.items()))
            for agent_id, counts in sorted(action_counts_by_agent.items())
        },
        carrier_diagnostics=carrier,
        conservation_errors={
            str(key): int(value) for key, value in env.world.reconcile_v2_ledger().items()
        },
    )
    env.close()
    return result


def evaluate_island_policy(
    policy_factory: Callable[[int], IslandPolicy],
    *,
    seeds: Sequence[int],
    procedural: bool,
) -> tuple[list[IslandEpisodeResult], dict[str, object]]:
    """Evaluate independent seeded policy instances and aggregate the public score."""

    results = [
        run_island_episode(policy_factory(seed), seed=seed, procedural=procedural) for seed in seeds
    ]
    rates = achievement_success_rates([frozenset(result.achievements) for result in results])
    action_counts: Counter[str] = Counter()
    carrier_totals: Counter[str] = Counter()
    material_carrier_episodes = 0
    for result in results:
        action_counts.update(result.action_counts)
        episode_had_carrier = False
        for diagnostics in result.carrier_diagnostics.values():
            if int(diagnostics["ticks"] or 0) > 0:
                episode_had_carrier = True
            for name, value in diagnostics.items():
                if name != "minimum_camp_distance" and value is not None:
                    carrier_totals[name] += int(value)
        material_carrier_episodes += int(episode_had_carrier)
    summary: dict[str, object] = {
        "episodes": len(results),
        "seeds": list(seeds),
        "achievement_success_rates": rates,
        "achievement_geometric_mean": geometric_mean_score(rates),
        "group_scores": grouped_scores(rates),
        "joint_survival_rate": float(np.mean([result.joint_survival for result in results])),
        "rescue_rate": float(np.mean([result.rescue_success for result in results])),
        "invalid_action_rate": sum(result.invalid_actions for result in results)
        / max(1, sum(result.actions for result in results)),
        "mean_agent_return": float(
            np.mean([value for result in results for value in result.agent_returns.values()])
        ),
        "mean_final_active_agents": float(
            np.mean([result.final_active_agents for result in results])
        ),
        "ledger_reconciliation_rate": float(
            np.mean([not result.conservation_errors for result in results])
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "carrier_diagnostics": {
            **dict(sorted(carrier_totals.items())),
            "episodes_with_material_carrier": material_carrier_episodes,
        },
    }
    return results, summary


def fixed_island_trainability_gate(
    learned: Mapping[str, object],
    random: Mapping[str, object],
) -> dict[str, object]:
    """Apply the predeclared Stage 7 fixed-island trainability gate."""

    rates_value = learned["achievement_success_rates"]
    if not isinstance(rates_value, Mapping):
        raise TypeError("achievement_success_rates must be a mapping.")
    rates = {str(key): _number(value) for key, value in rates_value.items()}
    checks = {
        "gather_food_80": float(rates["collect_food"]) >= 0.80,
        "gather_wood_80": float(rates["collect_wood"]) >= 0.80,
        "gather_stone_80": float(rates["collect_stone"]) >= 0.80,
        "deposit_wood_50": float(rates["deposit_wood"]) >= 0.50,
        "deposit_stone_50": float(rates["deposit_stone"]) >= 0.50,
        "workbench_20": float(rates["build_workbench"]) >= 0.20,
        "invalid_below_5": _number(learned["invalid_action_rate"]) < 0.05,
        "score_margin_002": _number(learned["achievement_geometric_mean"])
        >= _number(random["achievement_geometric_mean"]) + 0.02,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_island_checkpoint(
    checkpoint: str | Path,
    *,
    seeds: Sequence[int],
    procedural: bool,
    include_episodes: bool = True,
) -> dict[str, object]:
    """Load one checkpoint once and evaluate both canonical inference modes."""

    model, metadata = load_policy_checkpoint(checkpoint)
    if metadata.get("environment_id") != "VoyagerIsland-v1":
        raise ValueError("Checkpoint was not trained for VoyagerIsland-v1.")
    model_type = metadata.get("model_type")
    encoder = str(metadata["observation_encoder"])

    def factory(deterministic: bool) -> Callable[[int], IslandPolicy]:
        if model_type == "feed_forward":
            return lambda seed: FeedForwardModelIslandPolicy(
                model,
                encoder,
                deterministic=deterministic,
                seed=seed,
            )
        if model_type == "recurrent_gru":
            hidden_size = int(cast(Any, metadata["recurrent_hidden_size"]))
            return lambda seed: RecurrentModelIslandPolicy(
                model=model,
                encoder=encoder,
                hidden_size=hidden_size,
                deterministic=deterministic,
                seed=seed,
            )
        raise ValueError(f"Unsupported Island checkpoint model_type: {model_type!r}.")

    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
    }
    for mode, deterministic in (("stochastic", False), ("deterministic", True)):
        episodes, summary = evaluate_island_policy(
            factory(deterministic),
            seeds=seeds,
            procedural=procedural,
        )
        row: dict[str, object] = {"summary": summary}
        if include_episodes:
            row["episodes"] = [episode.as_dict() for episode in episodes]
        payload[mode] = row
    return payload


def island_checkpoint_selection_key(
    summary: Mapping[str, object],
) -> tuple[float, float]:
    """Rank a development checkpoint by public score, then lower invalid rate.

    Seeded-stochastic inference is the canonical PPO evaluation mode. Callers are
    responsible for supplying a stochastic development summary and for never
    selecting on held-out test results.
    """

    score = _number(summary["achievement_geometric_mean"])
    invalid_rate = _number(summary["invalid_action_rate"])
    if not math.isfinite(score) or not math.isfinite(invalid_rate):
        raise ValueError("Checkpoint selection metrics must be finite.")
    return score, -invalid_rate


def normalize_island_evaluation_milestones(
    values: Sequence[int],
    *,
    total_agent_transitions: int,
) -> tuple[int, ...]:
    """Return positive, unique milestones bounded by and including the budget."""

    if total_agent_transitions <= 0:
        raise ValueError("total_agent_transitions must be positive.")
    normalized = {min(value, total_agent_transitions) for value in values if value > 0}
    normalized.add(total_agent_transitions)
    return tuple(sorted(normalized))


def scripted_oracle_solvability_gate(summary: Mapping[str, object]) -> dict[str, object]:
    """Apply the frozen held-out safety-oracle threshold."""

    rates_value = summary["achievement_success_rates"]
    if not isinstance(rates_value, Mapping):
        raise TypeError("achievement_success_rates must be a mapping.")
    rates = {str(key): _number(value) for key, value in rates_value.items()}
    checks = {
        "every_achievement_at_least_90": all(
            rates.get(achievement, 0.0) >= 0.90 for achievement in ISLAND_ACHIEVEMENTS
        ),
        "rescue_at_least_90": _number(summary["rescue_rate"]) >= 0.90,
        "invalid_below_5": _number(summary["invalid_action_rate"]) < 0.05,
        "ledger_reconciles": _number(summary["ledger_reconciliation_rate"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected a numeric benchmark summary value.")
    return float(value)
