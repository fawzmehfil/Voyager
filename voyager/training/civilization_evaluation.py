"""Outcome-based evaluation for the Stage 7C trainability probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from voyager.sim.registries_v2 import V2_FLAT_ACTION_COUNT
from voyager.training.civilization_probe import CivilizationProbeRewardWrapper
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.masking import stack_action_masks
from voyager.training.obs import flatten_observations

ProbePolicy = Literal["legal_random", "model"]
PROBE_METRICS = (
    "gathered_wood_and_stone",
    "deposited_wood_and_stone",
    "workbench_complete",
    "any_tool_crafted",
    "majority_active_at_300",
)
PROBE_THRESHOLDS = {
    "gathered_wood_and_stone": 0.50,
    "deposited_wood_and_stone": 0.30,
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
    gathered_wood_and_stone: bool
    deposited_wood_and_stone: bool
    workbench_complete: bool
    any_tool_crafted: bool
    majority_active_at_300: bool
    active_at_300: int
    final_active: int
    deaths: int
    invalid_actions: int
    submitted_actions: int

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
) -> CivilizationEpisodeResult:
    """Run one handcrafted v2 episode with legal-random or model actions."""

    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
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
    agent_transitions = 0
    invalid_actions = 0
    submitted_actions = 0
    active_at_300 = 0

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

        observations, rewards, _terminations, _truncations, infos = env.step(actions)
        agent_transitions += len(agent_ids)
        submitted_actions += len(agent_ids)
        invalid_actions += sum(
            bool(infos[agent_id].get("invalid_action", False))
            for agent_id in agent_ids
        )
        if rewards:
            shared_return += float(next(iter(rewards.values())))
        state = env.world.state
        assert state is not None
        if state.step_count == 300:
            active_at_300 = sum(
                agent.life_state == "active" for agent in state.agents.values()
            )

    state = env.world.state
    assert state is not None
    gathered = {
        str(entry.get("item"))
        for entry in state.ledger
        if entry.get("event") == "gather"
    }
    deposited = {
        str(entry.get("item"))
        for entry in state.ledger
        if entry.get("event") == "deposit"
    }
    result = CivilizationEpisodeResult(
        policy=policy_name,
        seed=seed,
        world_steps=state.step_count,
        agent_transitions=agent_transitions,
        shared_return=shared_return,
        gathered_wood_and_stone={"wood", "stone"} <= gathered,
        deposited_wood_and_stone={"wood", "stone"} <= deposited,
        workbench_complete=state.structures["workbench"].complete,
        any_tool_crafted=any(
            entry.get("event") == "craft_tool" for entry in state.ledger
        ),
        majority_active_at_300=active_at_300 >= 6,
        active_at_300=active_at_300,
        final_active=sum(
            agent.life_state == "active" for agent in state.agents.values()
        ),
        deaths=sum(agent.life_state == "dead" for agent in state.agents.values()),
        invalid_actions=invalid_actions,
        submitted_actions=submitted_actions,
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
) -> list[CivilizationEpisodeResult]:
    """Evaluate one policy on a caller-owned deterministic seed set."""

    return [
        run_civilization_episode(
            policy_name=policy_name,
            policy=policy,
            seed=seed,
            model=model,
            deterministic=deterministic,
        )
        for seed in seeds
    ]


def summarize_civilization_results(
    results: list[CivilizationEpisodeResult],
) -> dict[str, object]:
    """Aggregate the five predeclared capability rates and diagnostics."""

    if not results:
        raise ValueError("At least one episode result is required.")
    return {
        "policy": results[0].policy,
        "episodes": len(results),
        "seeds": [result.seed for result in results],
        "composite": float(np.mean([result.composite for result in results])),
        "capability_rates": {
            metric: float(np.mean([float(getattr(result, metric)) for result in results]))
            for metric in PROBE_METRICS
        },
        "mean_active_at_300": float(
            np.mean([result.active_at_300 for result in results])
        ),
        "mean_final_active": float(np.mean([result.final_active for result in results])),
        "invalid_action_rate": sum(result.invalid_actions for result in results)
        / max(1, sum(result.submitted_actions for result in results)),
        "mean_shared_return": float(
            np.mean([result.shared_return for result in results])
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


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable_logits = logits - float(np.max(logits))
    probabilities = np.exp(stable_logits)
    return probabilities / np.sum(probabilities)
