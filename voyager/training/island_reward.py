"""Versioned bounded training rewards for the VoyagerIsland-v1 benchmark."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.envs.island import VoyagerIslandEnv
from voyager.sim.island_core import (
    island_progress_stage,
    island_stage_material_requirements,
)

ISLAND_TRAINING_REWARD_V2 = "voyager_island_trainability_reward_v2"
ISLAND_TRAINING_REWARD_V3 = "voyager_island_progression_reward_v3"
ISLAND_TRAINING_REWARD_V4 = "voyager_island_progression_reward_v4"
CAUSAL_ACHIEVEMENT_REWARD = 0.50
RETURN_MILESTONE_REWARD = 0.20
RETURN_DISTANCE_MILESTONES = (6, 3, 1)
USEFUL_DEPOSIT_UNIT_REWARD = 0.10
WORK_ACTION_REWARD = 0.10
EXTRACTION_MILESTONE_REWARD = 0.25
EXTRACTION_MILESTONES = (25, 50, 75)


class IslandTrainingRewardV2Wrapper(ParallelEnv[str, dict[str, np.ndarray], int]):
    """Add bounded causal credit without changing public evaluation outcomes."""

    metadata = VoyagerIslandEnv.metadata

    def __init__(self, env: VoyagerIslandEnv | None = None) -> None:
        self.env = env or VoyagerIslandEnv(reward_mode="dense")
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        self.action_spaces = dict(self.env.action_spaces)
        self._return_milestones: dict[str, set[int]] = {
            agent_id: set() for agent_id in self.possible_agents
        }
        self._component_totals: dict[str, float] = defaultdict(float)
        self._component_totals_by_agent: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._spawn_swaps = 0
        self._resets = 0

    @property
    def world(self) -> Any:
        return self.env.world

    @property
    def performance_seconds(self) -> dict[str, float]:
        value = getattr(self.env, "performance_seconds", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        swapped = self._should_swap_spawns(seed)
        if swapped:
            state = self.world.state
            assert state is not None
            left = state.agents[self.possible_agents[0]]
            right = state.agents[self.possible_agents[1]]
            left.x, right.x = right.x, left.x
            left.y, right.y = right.y, left.y
            observations = {
                agent_id: self.env._observation(agent_id) for agent_id in self.env.agents
            }
            infos = {agent_id: self.env._info(agent_id, "reset") for agent_id in self.env.agents}
            self._spawn_swaps += 1
        self._resets += 1
        self.agents = list(self.env.agents)
        self._return_milestones = {agent_id: set() for agent_id in self.possible_agents}
        for info in infos.values():
            info["reward_version"] = ISLAND_TRAINING_REWARD_V2
            info["spawn_assignment_swapped"] = swapped
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        acting_agents = tuple(self.env.agents)
        state = self.world.state
        assert state is not None
        material_before = {
            agent_id: {
                item: state.agents[agent_id].inventory.get(item, 0) for item in ("wood", "stone")
            }
            for agent_id in acting_agents
        }
        observations, _base_rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        state = self.world.state
        assert state is not None
        new_achievements = tuple(
            dict.fromkeys(
                achievement
                for agent_id in acting_agents
                for achievement in infos[agent_id].get("new_achievements", ())
            )
        )
        causal_actors = self._causal_actors(
            new_achievements,
            material_before=material_before,
        )

        rewards: dict[str, float] = {}
        for agent_id in acting_agents:
            info = infos[agent_id]
            base_components = {
                str(name): float(value)
                for name, value in dict(info.get("reward_components", {})).items()
            }
            individual: dict[str, float] = {}
            for achievement in new_achievements:
                if agent_id in causal_actors.get(achievement, frozenset()):
                    individual[f"causal_{achievement}"] = CAUSAL_ACHIEVEMENT_REWARD
            agent = state.agents[agent_id]
            carrying_material = (
                agent.inventory.get("wood", 0) + agent.inventory.get("stone", 0)
            ) > 0
            if agent.alive and carrying_material:
                distance = abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y)
                reached = self._return_milestones[agent_id]
                for threshold in RETURN_DISTANCE_MILESTONES:
                    if distance <= threshold and threshold not in reached:
                        reached.add(threshold)
                        individual[f"return_distance_{threshold}"] = RETURN_MILESTONE_REWARD

            combined = {**base_components, **individual}
            rewards[agent_id] = float(sum(combined.values()))
            shared = {
                name: value for name, value in base_components.items() if name.startswith("shared_")
            }
            base_individual = {
                name: value
                for name, value in base_components.items()
                if not name.startswith("shared_")
            }
            info["environment_reward_components"] = base_components
            info["shared_reward_components"] = shared
            info["individual_reward_components"] = {
                **base_individual,
                **individual,
            }
            info["reward_components"] = combined
            info["reward_version"] = ISLAND_TRAINING_REWARD_V2
            info["causal_achievements"] = sorted(
                achievement for achievement, actors in causal_actors.items() if agent_id in actors
            )
            for name, value in combined.items():
                self._component_totals[name] += value
                self._component_totals_by_agent[agent_id][name] += value
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def metrics(self) -> dict[str, object]:
        return self.env.metrics()

    def render(self) -> str | None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def reward_diagnostics(self) -> dict[str, object]:
        """Return cumulative training-only reward and spawn-assignment diagnostics."""

        return {
            "reward_version": ISLAND_TRAINING_REWARD_V2,
            "component_totals": dict(sorted(self._component_totals.items())),
            "component_totals_by_agent": {
                agent_id: dict(sorted(counter.items()))
                for agent_id, counter in sorted(self._component_totals_by_agent.items())
            },
            "resets": self._resets,
            "spawn_swaps": self._spawn_swaps,
        }

    def _causal_actors(
        self,
        achievements: tuple[str, ...],
        *,
        material_before: Mapping[str, Mapping[str, int]],
    ) -> dict[str, frozenset[str]]:
        state = self.world.state
        assert state is not None
        events = state.events

        def event_actors(
            event_type: str,
            *,
            item: str | None = None,
            target: str | None = None,
        ) -> set[str]:
            found: set[str] = set()
            for event in events:
                if event.get("type") != event_type:
                    continue
                payload = event.get("payload", {})
                if item is not None and (
                    not isinstance(payload, Mapping) or payload.get("item") != item
                ):
                    continue
                targets = event.get("targets", ())
                if target is not None and (not isinstance(targets, list) or target not in targets):
                    continue
                actors = event.get("actors", ())
                if isinstance(actors, list):
                    found.update(str(actor) for actor in actors if str(actor) in state.agents)
            return found

        result: dict[str, frozenset[str]] = {}
        for achievement in achievements:
            actors: set[str] = set()
            if achievement.startswith("collect_"):
                actors = event_actors("gather", item=achievement.removeprefix("collect_"))
            elif achievement.startswith("deposit_"):
                item = achievement.removeprefix("deposit_")
                actors = {
                    agent_id
                    for agent_id in event_actors("deposit_all")
                    if material_before.get(agent_id, {}).get(item, 0) > 0
                }
            elif achievement.startswith("build_"):
                actors = event_actors(
                    "structure_complete",
                    target=achievement.removeprefix("build_"),
                )
            elif achievement in {"craft_axe", "craft_spear", "cook_meat"}:
                actors = event_actors(achievement)
            elif achievement == "hunt_deer":
                for event in events:
                    if event.get("type") != "creature_defeated":
                        continue
                    targets = event.get("targets", ())
                    if not isinstance(targets, list) or not any(
                        target in state.creatures and state.creatures[target].type == "island_deer"
                        for target in targets
                    ):
                        continue
                    raw_actors = event.get("actors", ())
                    if isinstance(raw_actors, list):
                        actors.update(
                            str(actor) for actor in raw_actors if str(actor) in state.agents
                        )
            result[achievement] = frozenset(actors)
        return result

    @staticmethod
    def _should_swap_spawns(seed: int | None) -> bool:
        mixed_seed = None if seed is None else seed ^ 0x5A17_7A1D
        return bool(np.random.default_rng(mixed_seed).integers(0, 2))


class IslandTrainingRewardV3Wrapper(IslandTrainingRewardV2Wrapper):
    """Add bounded stage material, labor, and extraction credit."""

    reward_version = ISLAND_TRAINING_REWARD_V3
    useful_deposit_unit_reward = USEFUL_DEPOSIT_UNIT_REWARD
    work_action_reward = WORK_ACTION_REWARD

    def __init__(self, env: VoyagerIslandEnv | None = None) -> None:
        super().__init__(env)
        self._extraction_milestones: set[int] = set()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = super().reset(seed=seed, options=options)
        self._extraction_milestones.clear()
        for info in infos.values():
            info["reward_version"] = self.reward_version
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        state = self.world.state
        assert state is not None
        acting_agents = tuple(self.env.agents)
        stage_before = island_progress_stage(state)
        requirements = island_stage_material_requirements(state, stage_before)
        stock_before = {item: int(state.camp.stockpile.get(item, 0)) for item in ("wood", "stone")}
        material_before = {
            agent_id: {
                item: int(state.agents[agent_id].inventory.get(item, 0))
                for item in ("wood", "stone")
            }
            for agent_id in acting_agents
        }

        observations, rewards, terminations, truncations, infos = super().step(actions)
        state = self.world.state
        assert state is not None
        additions: dict[str, dict[str, float]] = {agent_id: {} for agent_id in acting_agents}
        successful_depositors = {
            agent_id for agent_id in acting_agents if infos[agent_id].get("event") == "deposit_all"
        }
        for item in ("wood", "stone"):
            useful_capacity = max(0, requirements[item] - stock_before[item])
            deposited = {
                agent_id: material_before[agent_id][item]
                for agent_id in successful_depositors
                if material_before[agent_id][item] > 0
            }
            total_deposited = sum(deposited.values())
            useful_total = min(useful_capacity, total_deposited)
            if useful_total <= 0 or total_deposited <= 0:
                continue
            for agent_id, quantity in deposited.items():
                credited_units = useful_total * quantity / total_deposited
                additions[agent_id][f"stage_{stage_before}_deposit_{item}"] = (
                    self.useful_deposit_unit_reward * credited_units
                )

        for event in state.events:
            if event.get("type") not in {"work", "structure_complete"}:
                continue
            actors = event.get("actors", ())
            payload = event.get("payload", {})
            targets = event.get("targets", ())
            if not isinstance(actors, list) or not actors:
                continue
            if not isinstance(payload, Mapping) or not isinstance(targets, list):
                continue
            applied_labor = float(payload.get("applied_labor", 0.0))
            if applied_labor <= 0:
                continue
            structure = str(targets[0]) if targets else "structure"
            per_actor = self.work_action_reward * (applied_labor / 10.0) / len(actors)
            for agent_id in actors:
                if agent_id in additions:
                    additions[agent_id][f"stage_{structure}_labor"] = per_actor

        beacon_step = state.achievement_steps.get("build_beacon")
        if beacon_step is not None and all(agent.alive for agent in state.agents.values()):
            extraction_elapsed = state.step_count - beacon_step
            for milestone in EXTRACTION_MILESTONES:
                if extraction_elapsed >= milestone and milestone not in self._extraction_milestones:
                    self._extraction_milestones.add(milestone)
                    for agent_id in acting_agents:
                        additions[agent_id][f"shared_extraction_{milestone}"] = (
                            EXTRACTION_MILESTONE_REWARD
                        )

        for agent_id in acting_agents:
            info = infos[agent_id]
            individual = dict(info.get("individual_reward_components", {}))
            shared = dict(info.get("shared_reward_components", {}))
            for name, value in additions[agent_id].items():
                if name.startswith("shared_"):
                    shared[name] = value
                else:
                    individual[name] = value
                self._component_totals[name] += value
                self._component_totals_by_agent[agent_id][name] += value
            combined = {**shared, **individual}
            rewards[agent_id] = float(sum(combined.values()))
            info["shared_reward_components"] = shared
            info["individual_reward_components"] = individual
            info["reward_components"] = combined
            info["reward_version"] = self.reward_version
            info["technology_stage"] = island_progress_stage(state)
        return observations, rewards, terminations, truncations, infos

    def reward_diagnostics(self) -> dict[str, object]:
        diagnostics = super().reward_diagnostics()
        diagnostics["reward_version"] = self.reward_version
        return diagnostics


class IslandTrainingRewardV4Wrapper(IslandTrainingRewardV3Wrapper):
    """Use salient but still finite stage credit after the under-scaled v3 probe."""

    reward_version = ISLAND_TRAINING_REWARD_V4
    useful_deposit_unit_reward = 0.50
    work_action_reward = 0.25
