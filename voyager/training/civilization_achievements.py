"""Achievement-spectrum evaluation for the Stage 7C Civilization island."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    unflatten_v2_action,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.factorized_actions import choose_factorized_actions
from voyager.training.masking import stack_action_masks
from voyager.training.obs import flatten_observations

ACHIEVEMENT_BENCHMARK_VERSION = "civilization_achievement_benchmark_v1"
FIRST_NIGHT_TICK = 300
WORKBENCH_WOOD = 6
WORKBENCH_STONE = 2
CAMP_INTERACTION_DISTANCE = 1
MIN_REPEATABLE_DELIVERY_RATE = 0.20

ACHIEVEMENT_GROUPS: dict[str, tuple[str, ...]] = {
    "gathering": (
        "gather_food",
        "gather_wood",
        "gather_stone",
        "gather_workbench_bundle",
    ),
    "delivery": (
        "deposit_food",
        "deposit_wood",
        "deposit_stone",
        "assemble_camp_bundle",
    ),
    "progression": (
        "start_workbench",
        "complete_workbench",
        "craft_tool",
        "transfer_tool",
    ),
    "survival": (
        "majority_active_first_night",
        "all_active_first_night",
        "majority_active_at_end",
    ),
}
ACHIEVEMENT_IDS = tuple(
    achievement for group in ACHIEVEMENT_GROUPS.values() for achievement in group
)
COMPOSITION_ACHIEVEMENTS = (
    "deposit_wood",
    "deposit_stone",
    "assemble_camp_bundle",
    "start_workbench",
    "complete_workbench",
    "craft_tool",
    "transfer_tool",
)

InferenceMode = Literal["deterministic", "seeded_stochastic"]


class AchievementPolicy(Protocol):
    """Stateful decentralized policy used by the achievement evaluator."""

    name: str

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        """Reset episode-local state."""

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Return one legal flattened action for every acting agent."""


@dataclass(slots=True)
class LegalRandomAchievementPolicy:
    """Uniform legal-action comparator."""

    name: str = "legal_random"

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        del possible_agents

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        del observations, observation_encoder
        masks = stack_action_masks(infos, agent_ids, action_count)
        return {
            agent_id: int(rng.choice(np.flatnonzero(masks[index])))
            for index, agent_id in enumerate(agent_ids)
        }


@dataclass(slots=True)
class FeedForwardAchievementPolicy:
    """Feed-forward actor evaluated with legal-action masking."""

    model: Any
    inference_mode: InferenceMode
    name: str = "feed_forward_ppo"

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        del possible_agents

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        masks = stack_action_masks(infos, agent_ids, action_count)
        flat = flatten_observations(observations, agent_ids, observation_encoder)
        logits, _values = self.model(flat, training=False)
        chosen = _choose_actions(
            np.asarray(logits, dtype=np.float64),
            masks,
            self.inference_mode,
            rng,
        )
        return {agent_id: int(chosen[index]) for index, agent_id in enumerate(agent_ids)}


@dataclass(slots=True)
class FactorizedAchievementPolicy:
    """Feed-forward hierarchical actor evaluated through the public flat registry."""

    model: Any
    inference_mode: InferenceMode
    name: str = "factorized_feed_forward_ppo"

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        del possible_agents

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        masks = stack_action_masks(infos, agent_ids, action_count)
        flat = flatten_observations(observations, agent_ids, observation_encoder)
        verb_logits, argument_logits, target_logits, _values = self.model(
            flat,
            training=False,
        )
        chosen, _joint_log_probs = choose_factorized_actions(
            verb_logits=np.asarray(verb_logits),
            argument_logits=np.asarray(argument_logits),
            target_logits=np.asarray(target_logits),
            flat_masks=masks,
            inference_mode=self.inference_mode,
            rng=rng,
        )
        return {agent_id: int(chosen[index]) for index, agent_id in enumerate(agent_ids)}


@dataclass(slots=True)
class RecurrentAchievementPolicy:
    """Recurrent actor with one hidden state per decentralized agent."""

    model: Any
    hidden_size: int
    inference_mode: InferenceMode
    name: str = "recurrent_ppo"
    _states: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        self._states = {
            agent_id: np.zeros((self.hidden_size,), dtype=np.float32)
            for agent_id in possible_agents
        }

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        masks = stack_action_masks(infos, agent_ids, action_count)
        flat = flatten_observations(observations, agent_ids, observation_encoder)
        initial_states = np.stack(
            [self._states[agent_id] for agent_id in agent_ids],
            axis=0,
        )
        logits, _values, final_states = self.model(
            [flat[:, None, :], initial_states],
            training=False,
        )
        logits_array = np.asarray(logits, dtype=np.float64)[:, 0, :]
        final_states_array = np.asarray(final_states, dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            self._states[agent_id] = final_states_array[index]
        chosen = _choose_actions(logits_array, masks, self.inference_mode, rng)
        return {agent_id: int(chosen[index]) for index, agent_id in enumerate(agent_ids)}


@dataclass(slots=True)
class FactorizedRecurrentAchievementPolicy:
    """Recurrent hierarchical actor with one hidden state per agent."""

    model: Any
    hidden_size: int
    inference_mode: InferenceMode
    name: str = "factorized_recurrent_ppo"
    _states: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        self._states = {
            agent_id: np.zeros((self.hidden_size,), dtype=np.float32)
            for agent_id in possible_agents
        }

    def act(
        self,
        *,
        observations: dict[str, dict[str, np.ndarray]],
        infos: dict[str, dict[str, Any]],
        agent_ids: tuple[str, ...],
        observation_encoder: str,
        action_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        masks = stack_action_masks(infos, agent_ids, action_count)
        flat = flatten_observations(observations, agent_ids, observation_encoder)
        initial_states = np.stack(
            [self._states[agent_id] for agent_id in agent_ids],
            axis=0,
        )
        (
            verb_logits,
            argument_logits,
            target_logits,
            _values,
            final_states,
        ) = self.model(
            [flat[:, None, :], initial_states],
            training=False,
        )
        final_states_array = np.asarray(final_states, dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            self._states[agent_id] = final_states_array[index]
        chosen, _joint_log_probs = choose_factorized_actions(
            verb_logits=np.asarray(verb_logits)[:, 0, :],
            argument_logits=np.asarray(argument_logits)[:, 0, :],
            target_logits=np.asarray(target_logits)[:, 0, :],
            flat_masks=masks,
            inference_mode=self.inference_mode,
            rng=rng,
        )
        return {agent_id: int(chosen[index]) for index, agent_id in enumerate(agent_ids)}


@dataclass(frozen=True, slots=True)
class AchievementEpisodeResult:
    """One episode's reward-independent group achievements and diagnostics."""

    policy: str
    inference_mode: str
    seed: int
    world_steps: int
    agent_transitions: int
    achievements: dict[str, bool]
    unlock_ticks: dict[str, int]
    invalid_actions: int
    submitted_actions: int
    gathered_counts: dict[str, int]
    deposited_counts: dict[str, int]
    peak_camp_stockpile: dict[str, int]
    active_at_first_night: int
    final_active: int
    deaths: int
    selected_verbs: dict[str, int] = field(default_factory=dict)
    selected_actions: dict[str, int] = field(default_factory=dict)
    return_diagnostics: dict[str, float] = field(default_factory=dict)
    peak_team_carried: dict[str, int] = field(default_factory=dict)
    final_team_carried: dict[str, int] = field(default_factory=dict)

    @property
    def invalid_action_rate(self) -> float:
        return self.invalid_actions / max(1, self.submitted_actions)

    @property
    def achievement_count(self) -> int:
        return sum(self.achievements.values())

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["achievement_count"] = self.achievement_count
        payload["invalid_action_rate"] = self.invalid_action_rate
        return payload


@dataclass(slots=True)
class CivilizationAchievementTracker:
    """Derive the frozen spectrum from public episode events and group state."""

    gathered: Counter[str] = field(default_factory=Counter)
    deposited: Counter[str] = field(default_factory=Counter)
    peak_camp: Counter[str] = field(default_factory=Counter)
    unlock_ticks: dict[str, int] = field(default_factory=dict)
    active_at_first_night: int = 0

    def observe(
        self,
        *,
        state: Any,
        new_ledger: list[dict[str, object]],
    ) -> None:
        tick = int(state.step_count)
        for entry in new_ledger:
            event = str(entry.get("event", ""))
            item = str(entry.get("item", ""))
            quantity = _ledger_quantity(entry)
            if event == "gather" and item:
                self.gathered[item] += quantity
                if item in {"food", "wood", "stone"}:
                    self._unlock(f"gather_{item}", tick)
            elif event == "deposit" and item:
                self.deposited[item] += quantity
                if item in {"food", "wood", "stone"}:
                    self._unlock(f"deposit_{item}", tick)
            elif event == "construction_reserve" and entry.get("target") == "workbench":
                self._unlock("start_workbench", tick)
            elif event == "construction_labor" and entry.get("target") == "workbench":
                if quantity > 0:
                    self._unlock("start_workbench", tick)
            elif event == "craft_tool":
                self._unlock("craft_tool", tick)
            elif event in {"give_tool", "deposit_tool"}:
                self._unlock("transfer_tool", tick)

        if self.gathered["wood"] >= WORKBENCH_WOOD and self.gathered["stone"] >= WORKBENCH_STONE:
            self._unlock("gather_workbench_bundle", tick)

        for item in ("food", "wood", "stone"):
            self.peak_camp[item] = max(
                self.peak_camp[item],
                int(state.camp.stockpile.get(item, 0)),
            )
        if (
            state.camp.stockpile.get("wood", 0) >= WORKBENCH_WOOD
            and state.camp.stockpile.get("stone", 0) >= WORKBENCH_STONE
        ):
            self._unlock("assemble_camp_bundle", tick)
        if state.structures["workbench"].complete:
            self._unlock("complete_workbench", tick)

        if tick == FIRST_NIGHT_TICK:
            self.active_at_first_night = _active_count(state)
            if self.active_at_first_night >= 6:
                self._unlock("majority_active_first_night", tick)
            if self.active_at_first_night == len(state.agents):
                self._unlock("all_active_first_night", tick)

    def finish(self, state: Any) -> None:
        if _active_count(state) >= 6:
            self._unlock("majority_active_at_end", int(state.step_count))

    def flags(self) -> dict[str, bool]:
        return {achievement: achievement in self.unlock_ticks for achievement in ACHIEVEMENT_IDS}

    def _unlock(self, achievement: str, tick: int) -> None:
        if achievement not in ACHIEVEMENT_IDS:
            raise ValueError(f"Unknown achievement: {achievement!r}.")
        self.unlock_ticks.setdefault(achievement, tick)


def run_achievement_episode(
    *,
    policy: AchievementPolicy,
    seed: int,
    inference_mode: str,
    reward_contract: str = CIVILIZATION_PROBE_REWARD_CONTRACT,
) -> AchievementEpisodeResult:
    """Run one unchanged 600-tick island episode and score its achievements."""

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
    action_count = int(env.action_space(env.possible_agents[0]).n)
    if action_count != V2_FLAT_ACTION_COUNT:
        raise ValueError("Achievement benchmark requires the frozen v2 action registry.")

    rng = np.random.default_rng(seed)
    observations, infos = env.reset(seed=seed)
    possible_agents = tuple(env.possible_agents)
    policy.reset(possible_agents)
    tracker = CivilizationAchievementTracker()
    ledger_cursor = 0
    agent_transitions = 0
    invalid_actions = 0
    submitted_actions = 0
    selected_verbs: Counter[str] = Counter()
    selected_actions: Counter[str] = Counter()
    carrying_transitions = 0
    carrying_distance_sum = 0.0
    carrying_move_actions = 0
    homeward_moves = 0
    away_moves = 0
    neutral_moves = 0
    carrying_at_camp_opportunities = 0
    resource_deposit_choices = 0
    resource_return_arrivals = 0
    peak_team_carried: Counter[str] = Counter()

    while env.agents:
        agent_ids = tuple(env.agents)
        actions = policy.act(
            observations=observations,
            infos=infos,
            agent_ids=agent_ids,
            observation_encoder=training_environment.observation_encoder,
            action_count=action_count,
            rng=rng,
        )
        if set(actions) != set(agent_ids):
            raise ValueError("Policy must return exactly one action per acting agent.")
        state_before = env.world.state
        assert state_before is not None
        distance_before: dict[str, int] = {}
        move_selected: set[str] = set()
        for agent_id, action in actions.items():
            verb_value, argument_value, target = unflatten_v2_action(action)
            verb = CivilizationV2Verb(verb_value)
            argument = CivilizationV2Argument(argument_value)
            selected_verbs[verb.name.lower()] += 1
            selected_actions[
                f"{verb.name.lower()}:{argument.name.lower()}:"
                f"{'targeted' if target else 'untargeted'}"
            ] += 1
            agent = state_before.agents[agent_id]
            wood = int(agent.inventory.get("wood", 0))
            stone = int(agent.inventory.get("stone", 0))
            if wood + stone <= 0:
                continue
            distance = abs(agent.x - state_before.camp.x) + abs(agent.y - state_before.camp.y)
            distance_before[agent_id] = distance
            carrying_transitions += 1
            carrying_distance_sum += distance
            if within_camp_interaction_distance(distance):
                carrying_at_camp_opportunities += 1
                if verb == CivilizationV2Verb.DEPOSIT and argument in {
                    CivilizationV2Argument.WOOD,
                    CivilizationV2Argument.STONE,
                }:
                    resource_deposit_choices += 1
            if verb == CivilizationV2Verb.MOVE:
                move_selected.add(agent_id)
                carrying_move_actions += 1
        observations, _rewards, _terminations, _truncations, infos = env.step(actions)
        agent_transitions += len(agent_ids)
        submitted_actions += len(agent_ids)
        invalid_actions += sum(
            bool(infos[agent_id].get("invalid_action", False)) for agent_id in agent_ids
        )
        state = env.world.state
        assert state is not None
        for agent_id in move_selected:
            agent = state.agents[agent_id]
            distance_after = abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y)
            before = distance_before[agent_id]
            if distance_after < before:
                homeward_moves += 1
            elif distance_after > before:
                away_moves += 1
            else:
                neutral_moves += 1
            if entered_camp_interaction_range(before, distance_after):
                resource_return_arrivals += 1
        for item in ("wood", "stone"):
            team_carried = sum(
                int(agent.inventory.get(item, 0))
                for agent in state.agents.values()
                if agent.life_state != "dead"
            )
            peak_team_carried[item] = max(peak_team_carried[item], team_carried)
        new_ledger = state.ledger[ledger_cursor:]
        ledger_cursor = len(state.ledger)
        tracker.observe(state=state, new_ledger=new_ledger)

    state = env.world.state
    assert state is not None
    tracker.finish(state)
    final_team_carried = {
        item: sum(
            int(agent.inventory.get(item, 0))
            for agent in state.agents.values()
            if agent.life_state != "dead"
        )
        for item in ("wood", "stone")
    }
    result = AchievementEpisodeResult(
        policy=policy.name,
        inference_mode=inference_mode,
        seed=seed,
        world_steps=int(state.step_count),
        agent_transitions=agent_transitions,
        achievements=tracker.flags(),
        unlock_ticks=dict(sorted(tracker.unlock_ticks.items())),
        invalid_actions=invalid_actions,
        submitted_actions=submitted_actions,
        gathered_counts=dict(sorted(tracker.gathered.items())),
        deposited_counts=dict(sorted(tracker.deposited.items())),
        peak_camp_stockpile=dict(sorted(tracker.peak_camp.items())),
        active_at_first_night=tracker.active_at_first_night,
        final_active=_active_count(state),
        deaths=sum(agent.life_state == "dead" for agent in state.agents.values()),
        selected_verbs=dict(sorted(selected_verbs.items())),
        selected_actions=dict(sorted(selected_actions.items())),
        return_diagnostics={
            "carrying_transitions": float(carrying_transitions),
            "mean_carrying_distance": carrying_distance_sum / max(1, carrying_transitions),
            "carrying_move_actions": float(carrying_move_actions),
            "homeward_moves": float(homeward_moves),
            "away_moves": float(away_moves),
            "neutral_moves": float(neutral_moves),
            "homeward_move_rate": homeward_moves / max(1, carrying_move_actions),
            "carrying_at_camp_opportunities": float(carrying_at_camp_opportunities),
            "resource_deposit_choices": float(resource_deposit_choices),
            "deposit_choice_rate_at_camp": resource_deposit_choices
            / max(1, carrying_at_camp_opportunities),
            "resource_return_arrivals": float(resource_return_arrivals),
        },
        peak_team_carried=dict(sorted(peak_team_carried.items())),
        final_team_carried=final_team_carried,
    )
    env.close()
    return result


def evaluate_achievement_policy(
    *,
    policy: AchievementPolicy,
    seeds: list[int],
    inference_mode: str,
    reward_contract: str = CIVILIZATION_PROBE_REWARD_CONTRACT,
) -> list[AchievementEpisodeResult]:
    """Evaluate one policy on a fixed ordered seed set."""

    if not seeds:
        raise ValueError("Achievement evaluation requires at least one seed.")
    return [
        run_achievement_episode(
            policy=policy,
            seed=seed,
            inference_mode=inference_mode,
            reward_contract=reward_contract,
        )
        for seed in seeds
    ]


def summarize_achievement_results(
    results: list[AchievementEpisodeResult],
) -> dict[str, object]:
    """Report every success rate plus group and overall geometric-mean scores."""

    if not results:
        raise ValueError("Cannot summarize an empty achievement result set.")
    rates = {
        achievement: float(np.mean([result.achievements[achievement] for result in results]))
        for achievement in ACHIEVEMENT_IDS
    }
    group_scores = {
        group: smoothed_geometric_mean([rates[achievement] for achievement in achievements])
        for group, achievements in ACHIEVEMENT_GROUPS.items()
    }
    unlock_tick_means = {
        achievement: float(
            np.mean(
                [
                    result.unlock_ticks[achievement]
                    for result in results
                    if achievement in result.unlock_ticks
                ]
            )
        )
        for achievement in ACHIEVEMENT_IDS
        if any(achievement in result.unlock_ticks for result in results)
    }
    return {
        "benchmark_version": ACHIEVEMENT_BENCHMARK_VERSION,
        "policy": results[0].policy,
        "inference_mode": results[0].inference_mode,
        "episodes": len(results),
        "seeds": [result.seed for result in results],
        "achievement_rates": rates,
        "achievement_score": smoothed_geometric_mean(list(rates.values())),
        "group_scores": group_scores,
        "mean_achievement_count": float(np.mean([result.achievement_count for result in results])),
        "mean_unlock_ticks_on_success": unlock_tick_means,
        "invalid_action_rate": sum(result.invalid_actions for result in results)
        / max(1, sum(result.submitted_actions for result in results)),
        "mean_active_at_first_night": float(
            np.mean([result.active_at_first_night for result in results])
        ),
        "mean_final_active": float(np.mean([result.final_active for result in results])),
        "mean_deaths": float(np.mean([result.deaths for result in results])),
        "mean_selected_verbs_per_episode": _mean_counter_maps(
            [result.selected_verbs for result in results]
        ),
        "mean_selected_actions_per_episode": _mean_counter_maps(
            [result.selected_actions for result in results]
        ),
        "mean_return_diagnostics": _mean_numeric_maps(
            [result.return_diagnostics for result in results]
        ),
        "mean_peak_team_carried": _mean_numeric_maps(
            [result.peak_team_carried for result in results]
        ),
        "mean_final_team_carried": _mean_numeric_maps(
            [result.final_team_carried for result in results]
        ),
        "resource_deposit_conversion": {
            item: sum(result.deposited_counts.get(item, 0) for result in results)
            / max(
                1,
                sum(result.gathered_counts.get(item, 0) for result in results),
            )
            for item in ("wood", "stone")
        },
        "episodes_detail": [result.as_dict() for result in results],
    }


def compare_achievement_summaries(
    learned: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    """Compare two seed-matched achievement summaries without hiding raw rates."""

    learned_rates = _numeric_map(learned.get("achievement_rates"), "learned rates")
    baseline_rates = _numeric_map(baseline.get("achievement_rates"), "baseline rates")
    rate_differences = {
        achievement: learned_rates[achievement] - baseline_rates[achievement]
        for achievement in ACHIEVEMENT_IDS
    }
    learned_score = _number(learned.get("achievement_score"), "learned score")
    baseline_score = _number(baseline.get("achievement_score"), "baseline score")
    meaningful_wins = [
        achievement
        for achievement in COMPOSITION_ACHIEVEMENTS
        if rate_differences[achievement] >= 0.10
    ]
    return {
        "achievement_score_difference": learned_score - baseline_score,
        "achievement_rate_differences": rate_differences,
        "meaningful_achievement_wins": meaningful_wins,
        "score_above_baseline": learned_score > baseline_score,
        "meaningful_progression_above_baseline": bool(meaningful_wins),
    }


def delivery_emergence(
    summary: dict[str, object],
    *,
    minimum_episode_rate: float = MIN_REPEATABLE_DELIVERY_RATE,
) -> dict[str, object]:
    """Require return or material deposit behavior in a repeatable episode fraction."""

    if not 0.0 <= minimum_episode_rate <= 1.0:
        raise ValueError("minimum_episode_rate must be in [0, 1].")
    rates = _numeric_map(summary.get("achievement_rates"), "achievement rates")
    details = summary.get("episodes_detail")
    if not isinstance(details, list) or not details:
        raise TypeError("Achievement summary must include non-empty episodes_detail.")
    arrival_episodes = 0
    for episode in details:
        if not isinstance(episode, dict):
            raise TypeError("Episode detail must be a dictionary.")
        diagnostics = episode.get("return_diagnostics")
        if not isinstance(diagnostics, dict):
            raise TypeError("Episode detail must include return_diagnostics.")
        arrivals = _number(
            diagnostics.get("resource_return_arrivals", 0.0),
            "resource_return_arrivals",
        )
        arrival_episodes += arrivals > 0.0
    arrival_episode_rate = arrival_episodes / len(details)
    material_deposit_rate = max(rates["deposit_wood"], rates["deposit_stone"])
    return {
        "minimum_episode_rate": minimum_episode_rate,
        "material_deposit_episode_rate": material_deposit_rate,
        "return_arrival_episode_rate": arrival_episode_rate,
        "repeatable_material_deposit": material_deposit_rate >= minimum_episode_rate,
        "repeatable_return_arrival": arrival_episode_rate >= minimum_episode_rate,
        "repeatable_delivery_emerged": (
            material_deposit_rate >= minimum_episode_rate
            or arrival_episode_rate >= minimum_episode_rate
        ),
    }


def within_camp_interaction_distance(distance: int) -> bool:
    """Match Civilization's public deposit/withdraw interaction radius."""

    return distance <= CAMP_INTERACTION_DISTANCE


def entered_camp_interaction_range(before: int, after: int) -> bool:
    """Return whether movement crossed into the camp interaction radius."""

    return not within_camp_interaction_distance(before) and within_camp_interaction_distance(after)


def calibration_gate(
    *,
    random_summary: dict[str, object],
    feed_forward_summary: dict[str, object],
    recurrent_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Apply the staged baseline-separation rule without a binary task composite."""

    feed_forward_vs_random = compare_achievement_summaries(feed_forward_summary, random_summary)
    feed_forward_separates = bool(
        feed_forward_vs_random["score_above_baseline"]
        and feed_forward_vs_random["meaningful_progression_above_baseline"]
    )
    recurrent_vs_feed_forward = None
    recurrent_vs_random = None
    strict_ordering = False
    learned_baseline_separates = feed_forward_separates
    if recurrent_summary is not None:
        recurrent_vs_feed_forward = compare_achievement_summaries(
            recurrent_summary, feed_forward_summary
        )
        recurrent_vs_random = compare_achievement_summaries(recurrent_summary, random_summary)
        recurrent_separates_random = bool(
            recurrent_vs_random["score_above_baseline"]
            and recurrent_vs_random["meaningful_progression_above_baseline"]
        )
        learned_baseline_separates = feed_forward_separates or recurrent_separates_random
        strict_ordering = bool(
            feed_forward_separates
            and recurrent_vs_feed_forward["score_above_baseline"]
            and recurrent_vs_feed_forward["meaningful_progression_above_baseline"]
        )

    return {
        "desired_ordering": "legal_random < feed_forward_ppo < recurrent_ppo_or_mappo",
        "feed_forward_vs_random": feed_forward_vs_random,
        "recurrent_vs_feed_forward": recurrent_vs_feed_forward,
        "recurrent_vs_random": recurrent_vs_random,
        "strict_ordering_demonstrated": strict_ordering,
        "at_least_one_learned_baseline_exceeds_random": learned_baseline_separates,
        "content_work_may_resume": (recurrent_summary is not None and learned_baseline_separates),
        "next_action": (
            "run_recurrent_ppo"
            if recurrent_summary is None
            else (
                "resume_stage7c_content"
                if learned_baseline_separates
                else "add_minimal_team_state_or_factorized_actions_before_mappo"
            )
        ),
    }


def smoothed_geometric_mean(rates: list[float]) -> float:
    """Return exp(mean(log(1 + 100r))) - 1 in normalized [0, 1] units."""

    if not rates:
        raise ValueError("Geometric mean requires at least one rate.")
    values = np.asarray(rates, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Achievement rates must be in [0, 1].")
    percentage_score = np.exp(np.mean(np.log1p(100.0 * values))) - 1.0
    return float(percentage_score / 100.0)


def _choose_actions(
    logits: np.ndarray,
    masks: np.ndarray,
    inference_mode: InferenceMode,
    rng: np.random.Generator,
) -> np.ndarray:
    masked_logits = np.where(masks, logits, -1e9)
    if inference_mode == "deterministic":
        return np.argmax(masked_logits, axis=1).astype(np.int64)
    if inference_mode != "seeded_stochastic":
        raise ValueError(f"Unknown inference mode: {inference_mode!r}.")
    return np.asarray(
        [rng.choice(masked_logits.shape[1], p=_probabilities(row)) for row in masked_logits],
        dtype=np.int64,
    )


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable = logits - float(np.max(logits))
    probabilities = np.exp(stable)
    return probabilities / np.sum(probabilities)


def _ledger_quantity(entry: dict[str, object]) -> int:
    value = entry.get("quantity", 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Ledger quantity must be numeric.")
    return int(value)


def _active_count(state: Any) -> int:
    return sum(agent.life_state == "active" for agent in state.agents.values())


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _numeric_map(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary.")
    result = {str(key): _number(item, f"{name}.{key}") for key, item in value.items()}
    missing = set(ACHIEVEMENT_IDS) - set(result)
    if missing:
        raise ValueError(f"{name} is missing achievements: {sorted(missing)}")
    return result


def _mean_counter_maps(rows: list[dict[str, int]]) -> dict[str, float]:
    keys = {key for row in rows for key in row}
    return {key: float(np.mean([row.get(key, 0) for row in rows])) for key in sorted(keys)}


def _mean_numeric_maps(
    rows: Sequence[Mapping[str, int | float]],
) -> dict[str, float]:
    keys = {key for row in rows for key in row}
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in sorted(keys)}
