"""Versioned rewards and actor observations for Stage 7C trainability."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.envs.civilization_v2 import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
)

PROBE_REWARD_V1 = "civilization_trainability_probe_v1"
PROBE_REWARD_V2 = "civilization_trainability_probe_v2"
PROBE_REWARD_VERSION = "civilization_trainability_probe_v3"
PROBE_REWARD_V4 = "civilization_trainability_probe_v4"
PROBE_REWARD_VERSIONS = frozenset(
    {PROBE_REWARD_V1, PROBE_REWARD_V2, PROBE_REWARD_VERSION, PROBE_REWARD_V4}
)

WORKBENCH_RESOURCE_REQUIREMENTS = {"wood": 6, "stone": 2}


class CivilizationProbeRewardWrapper(
    ParallelEnv[str, dict[str, np.ndarray], int]
):
    """Expose versioned probe rewards over the deterministic v2 world."""

    metadata = CivilizationV2FlattenedActionWrapper.metadata

    def __init__(
        self,
        env: CivilizationV2FlattenedActionWrapper | None = None,
        *,
        reward_version: str = PROBE_REWARD_VERSION,
    ) -> None:
        if reward_version not in PROBE_REWARD_VERSIONS:
            raise ValueError(f"Unsupported Civilization probe reward: {reward_version!r}.")
        self.env = env or CivilizationV2FlattenedActionWrapper(
            VoyagerCivilizationV2Env(reward_mode="none")
        )
        self.reward_version = reward_version
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        if reward_version in {
            PROBE_REWARD_V2,
            PROBE_REWARD_VERSION,
            PROBE_REWARD_V4,
        }:
            self.observation_spaces = {
                agent_id: self._augmented_observation_space(
                    space,
                    include_camp_bearing=reward_version
                    in {PROBE_REWARD_VERSION, PROBE_REWARD_V4},
                    include_team_objective=reward_version == PROBE_REWARD_V4,
                )
                for agent_id, space in self.observation_spaces.items()
            }
        self.action_spaces = dict(self.env.action_spaces)
        self._milestones: set[str] = set()
        self._ledger_cursor = 0
        self._gathered_units: Counter[str] = Counter()
        self._credited_deposits: Counter[str] = Counter()
        self._visited_positions: set[tuple[int, int]] = set()
        self._v3_gather_credit: dict[str, float] = {}
        self._v3_camp_high_water: dict[str, float] = {}

    @property
    def world(self) -> Any:
        return self.env.env.world

    @property
    def performance_seconds(self) -> dict[str, float]:
        return self.env.performance_seconds

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        self._milestones.clear()
        self._gathered_units.clear()
        self._credited_deposits.clear()
        self._v3_gather_credit.clear()
        self._v3_camp_high_water.clear()
        self._v3_gather_credit.update(
            {item: 0.0 for item in WORKBENCH_RESOURCE_REQUIREMENTS}
        )
        self._v3_camp_high_water.update(
            {item: 0.0 for item in WORKBENCH_RESOURCE_REQUIREMENTS}
        )
        state = self.world.state
        assert state is not None
        self._ledger_cursor = len(state.ledger)
        self._visited_positions = {
            (agent.x, agent.y)
            for agent in state.agents.values()
            if agent.life_state != "dead"
        }
        for info in infos.values():
            info["reward_version"] = self.reward_version
        return self._augment_observations(observations), infos

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
        before_workbench = state.structures["workbench"].progress
        observations, _rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        state = self.world.state
        assert state is not None
        new_ledger = state.ledger[self._ledger_cursor :]
        self._ledger_cursor = len(state.ledger)
        if self.reward_version in {PROBE_REWARD_VERSION, PROBE_REWARD_V4}:
            shared_components, individual_components = self._reward_components_v3(
                new_ledger=new_ledger,
                before_workbench=before_workbench,
                infos=infos,
            )
            shared_reward = float(sum(shared_components.values()))
            rewards = {}
            for agent_id in acting_agents:
                individual = individual_components.get(agent_id, {})
                individual_reward = float(sum(individual.values()))
                rewards[agent_id] = shared_reward + individual_reward
                info = infos[agent_id]
                info["environment_reward_components"] = dict(
                    info.get("dense_reward_components", {})
                )
                info["shared_reward_components"] = dict(shared_components)
                info["individual_reward_components"] = dict(individual)
                info["reward_components"] = {
                    **{
                        f"shared_{name}": value
                        for name, value in shared_components.items()
                    },
                    **{
                        f"individual_{name}": value
                        for name, value in individual.items()
                    },
                }
                info["shared_reward"] = shared_reward
                info["individual_reward"] = individual_reward
                info["reward_version"] = self.reward_version
        else:
            components = self._reward_components(
                new_ledger=new_ledger,
                before_workbench=before_workbench,
                infos=infos,
            )
            team_reward = float(sum(components.values()))
            rewards = {agent_id: team_reward for agent_id in acting_agents}
            for agent_id in acting_agents:
                info = infos[agent_id]
                info["environment_reward_components"] = dict(
                    info.get("dense_reward_components", {})
                )
                info["reward_components"] = dict(components)
                info["reward_version"] = self.reward_version
        return (
            self._augment_observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def state(self) -> dict[str, object]:
        return self.env.state()

    def close(self) -> None:
        self.env.close()

    def _reward_components(
        self,
        *,
        new_ledger: list[dict[str, object]],
        before_workbench: float,
        infos: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        if self.reward_version == PROBE_REWARD_V1:
            return self._reward_components_v1(
                new_ledger=new_ledger,
                before_workbench=before_workbench,
                infos=infos,
            )
        if self.reward_version != PROBE_REWARD_V2:
            raise RuntimeError("The v3 reward uses per-agent reward components.")
        return self._reward_components_v2(
            new_ledger=new_ledger,
            before_workbench=before_workbench,
            infos=infos,
        )

    def _reward_components_v2(
        self,
        *,
        new_ledger: list[dict[str, object]],
        before_workbench: float,
        infos: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        state = self.world.state
        assert state is not None
        active_fraction = sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)
        invalid_fraction = sum(
            bool(info.get("invalid_action", False)) for info in infos.values()
        ) / max(1, len(infos))
        components: dict[str, float] = {
            "survival": 0.001 * active_fraction,
            "invalid": -0.02 * invalid_fraction,
            "downed": -0.25
            * sum(event["type"] == "agent_downed" for event in state.events),
            "death": -1.0
            * sum(event["type"] == "agent_died" for event in state.events),
        }

        current_positions = {
            (agent.x, agent.y)
            for agent in state.agents.values()
            if agent.life_state == "active"
        }
        new_positions = current_positions - self._visited_positions
        if new_positions:
            components["new_team_tiles"] = 0.002 * len(new_positions)
            self._visited_positions.update(new_positions)

        gather_values = {
            "food": 0.04,
            "wood": 0.10,
            "stone": 0.10,
            "raw_meat": 0.04,
            "cooked_meat": 0.04,
        }
        for entry in new_ledger:
            event = str(entry.get("event", ""))
            item = str(entry.get("item", ""))
            tool = str(entry.get("tool", ""))
            quantity = self._integer(entry.get("quantity", 0), "ledger quantity")
            target = str(entry.get("target", ""))
            if event == "gather" and item in gather_values and quantity > 0:
                self._gathered_units[item] += quantity
                self._add_component(
                    components,
                    f"gather_{item}",
                    gather_values[item] * quantity,
                )
            elif event == "deposit" and item in {"wood", "stone"} and quantity > 0:
                uncredited = max(
                    0,
                    self._gathered_units[item] - self._credited_deposits[item],
                )
                credited = min(quantity, uncredited)
                if credited:
                    self._credited_deposits[item] += credited
                    self._add_component(
                        components,
                        f"first_delivery_{item}",
                        0.15 * credited,
                    )
            elif event == "construction_reserve" and target == "workbench":
                self._award_once(components, "workbench_materials_reserved", 0.5)
            elif event == "construction_labor" and target == "workbench" and quantity > 0:
                self._add_component(
                    components,
                    "workbench_labor",
                    0.005 * quantity,
                )
            elif event == "craft_tool" and tool:
                self._award_once(components, f"first_{tool}_crafted", 1.0)

        workbench_progress = state.structures["workbench"].progress
        if before_workbench < 1.0 <= workbench_progress:
            self._award_once(components, "workbench_complete", 2.0)
        if state.step_count == 300:
            self._award_once(components, "first_night_survival", active_fraction)
        return components

    def _reward_components_v3(
        self,
        *,
        new_ledger: list[dict[str, object]],
        before_workbench: float,
        infos: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        state = self.world.state
        assert state is not None
        active_fraction = sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)
        shared: dict[str, float] = {
            "survival": 0.001 * active_fraction,
            "downed": -0.25
            * sum(event["type"] == "agent_downed" for event in state.events),
            "death": -1.0
            * sum(event["type"] == "agent_died" for event in state.events),
        }
        individual: dict[str, dict[str, float]] = {
            agent_id: {} for agent_id in infos
        }
        for agent_id, info in infos.items():
            if bool(info.get("invalid_action", False)):
                individual[agent_id]["invalid"] = -0.02

        gathered = self._actor_quantities(new_ledger, event="gather")
        deposited = self._actor_quantities(new_ledger, event="deposit")
        for item, requirement in WORKBENCH_RESOURCE_REQUIREMENTS.items():
            gathered_by_actor = gathered.get(item, {})
            gathered_total = sum(gathered_by_actor.values())
            remaining = max(0.0, requirement - self._v3_gather_credit[item])
            gather_credit = min(gathered_total, remaining)
            if gather_credit > 0.0:
                self._distribute_credit(
                    individual,
                    gathered_by_actor,
                    gather_credit,
                    component=f"gather_{item}",
                    value_per_unit=0.05,
                )
                self._v3_gather_credit[item] += gather_credit
            if self._v3_gather_credit[item] >= requirement:
                self._award_once(shared, f"gather_{item}_requirement", 0.25)

            previous_high_water = self._v3_camp_high_water[item]
            current_stock = min(requirement, state.camp.stockpile.get(item, 0))
            current_high_water = max(previous_high_water, current_stock)
            delivery_credit = current_high_water - previous_high_water
            if delivery_credit > 0.0:
                self._distribute_credit(
                    individual,
                    deposited.get(item, {}),
                    delivery_credit,
                    component=f"deliver_{item}",
                    value_per_unit=0.10,
                )
                self._v3_camp_high_water[item] = current_high_water
            if self._v3_camp_high_water[item] >= requirement:
                self._award_once(shared, f"camp_{item}_requirement", 0.50)

        if all(
            self._v3_gather_credit[item] >= requirement
            for item, requirement in WORKBENCH_RESOURCE_REQUIREMENTS.items()
        ):
            self._award_once(shared, "gather_workbench_bundle", 0.50)
        if all(
            self._v3_camp_high_water[item] >= requirement
            for item, requirement in WORKBENCH_RESOURCE_REQUIREMENTS.items()
        ):
            self._award_once(shared, "camp_workbench_bundle", 1.00)

        for entry in new_ledger:
            event = str(entry.get("event", ""))
            target = str(entry.get("target", ""))
            tool = str(entry.get("tool", ""))
            if event == "construction_reserve" and target == "workbench":
                self._award_once(shared, "workbench_materials_reserved", 1.00)
            elif event == "craft_tool" and tool:
                self._award_once(shared, "first_tool_crafted", 0.75)
                self._award_once(shared, f"first_{tool}_crafted", 0.25)

        workbench_progress = state.structures["workbench"].progress
        progress_delta = max(0.0, workbench_progress - before_workbench)
        if progress_delta:
            shared["workbench_progress"] = 2.0 * progress_delta
        if before_workbench < 1.0 <= workbench_progress:
            self._award_once(shared, "workbench_complete", 2.00)
        if state.step_count == 300:
            self._award_once(shared, "first_night_survival", active_fraction)
        return shared, individual

    def _reward_components_v1(
        self,
        *,
        new_ledger: list[dict[str, object]],
        before_workbench: float,
        infos: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        state = self.world.state
        assert state is not None
        active_fraction = sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)
        invalid_fraction = sum(
            bool(info.get("invalid_action", False)) for info in infos.values()
        ) / max(1, len(infos))
        components: dict[str, float] = {
            "survival": 0.001 * active_fraction,
            "invalid": -0.02 * invalid_fraction,
            "downed": -0.25
            * sum(event["type"] == "agent_downed" for event in state.events),
            "death": -1.0
            * sum(event["type"] == "agent_died" for event in state.events),
        }
        for entry in new_ledger:
            event = str(entry.get("event", ""))
            item = str(entry.get("item", ""))
            tool = str(entry.get("tool", ""))
            if event == "gather" and item in {"food", "wood", "stone"}:
                self._award_once(components, f"gather_{item}", 0.25)
            elif event == "deposit" and item in {"food", "wood", "stone"}:
                self._award_once(components, f"deposit_{item}", 0.25)
            elif event == "craft_tool" and tool:
                self._award_once(components, f"craft_{tool}", 0.5)

        workbench_progress = state.structures["workbench"].progress
        progress_delta = max(0.0, workbench_progress - before_workbench)
        if progress_delta:
            components["workbench_progress"] = 2.0 * progress_delta
        if workbench_progress >= 1.0:
            self._award_once(components, "workbench_complete", 1.0)
        if state.step_count == 300:
            self._award_once(components, "first_night_survival", active_fraction)
        return components

    def _augment_observations(
        self,
        observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        if self.reward_version == PROBE_REWARD_V1:
            return observations
        augmented: dict[str, dict[str, np.ndarray]] = {}
        for agent_id, observation in observations.items():
            identity = np.zeros(len(self.possible_agents), dtype=np.int8)
            identity[self.possible_agents.index(agent_id)] = 1
            values = {
                **observation,
                "agent_identity": identity,
            }
            if self.reward_version in {PROBE_REWARD_VERSION, PROBE_REWARD_V4}:
                state = self.world.state
                assert state is not None
                agent = state.agents[agent_id]
                x_scale = max(1, state.terrain.shape[1] - 1)
                y_scale = max(1, state.terrain.shape[0] - 1)
                values["camp_bearing"] = np.array(
                    [
                        (state.camp.x - agent.x) / x_scale,
                        (state.camp.y - agent.y) / y_scale,
                        (
                            abs(state.camp.x - agent.x)
                            + abs(state.camp.y - agent.y)
                        )
                        / (x_scale + y_scale),
                    ],
                    dtype=np.float32,
                )
            if self.reward_version == PROBE_REWARD_V4:
                state = self.world.state
                assert state is not None
                active_fraction = sum(
                    agent.life_state == "active"
                    for agent in state.agents.values()
                ) / len(state.agents)
                values["team_objective"] = np.array(
                    [
                        self._v3_gather_credit["wood"]
                        / WORKBENCH_RESOURCE_REQUIREMENTS["wood"],
                        self._v3_gather_credit["stone"]
                        / WORKBENCH_RESOURCE_REQUIREMENTS["stone"],
                        self._v3_camp_high_water["wood"]
                        / WORKBENCH_RESOURCE_REQUIREMENTS["wood"],
                        self._v3_camp_high_water["stone"]
                        / WORKBENCH_RESOURCE_REQUIREMENTS["stone"],
                        state.structures["workbench"].progress,
                        active_fraction,
                    ],
                    dtype=np.float32,
                )
            augmented[agent_id] = values
        return augmented

    def _augmented_observation_space(
        self,
        space: spaces.Space,
        *,
        include_camp_bearing: bool,
        include_team_objective: bool,
    ) -> spaces.Dict:
        if not isinstance(space, spaces.Dict):
            raise TypeError("Civilization v2 requires a Dict observation space.")
        additions: dict[str, spaces.Space] = {
            "agent_identity": spaces.Box(
                0,
                1,
                shape=(len(self.possible_agents),),
                dtype=np.int8,
            )
        }
        if include_camp_bearing:
            additions["camp_bearing"] = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                dtype=np.float32,
            )
        if include_team_objective:
            additions["team_objective"] = spaces.Box(
                0.0,
                1.0,
                shape=(6,),
                dtype=np.float32,
            )
        return spaces.Dict(
            {
                **space.spaces,
                **additions,
            }
        )

    @staticmethod
    def _actor_quantities(
        entries: list[dict[str, object]],
        *,
        event: str,
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for entry in entries:
            if str(entry.get("event", "")) != event:
                continue
            item = str(entry.get("item", ""))
            quantity = CivilizationProbeRewardWrapper._integer(
                entry.get("quantity", 0),
                "ledger quantity",
            )
            actors = entry.get("actors", [])
            if not isinstance(actors, list) or not actors or quantity <= 0:
                continue
            actor_quantity = quantity / len(actors)
            item_result = result.setdefault(item, {})
            for actor in actors:
                actor_id = str(actor)
                item_result[actor_id] = (
                    item_result.get(actor_id, 0.0) + actor_quantity
                )
        return result

    @staticmethod
    def _distribute_credit(
        components: dict[str, dict[str, float]],
        actor_quantities: dict[str, float],
        credited_quantity: float,
        *,
        component: str,
        value_per_unit: float,
    ) -> None:
        total_quantity = sum(actor_quantities.values())
        if credited_quantity <= 0.0 or total_quantity <= 0.0:
            return
        for agent_id, quantity in actor_quantities.items():
            if agent_id not in components:
                continue
            value = value_per_unit * credited_quantity * quantity / total_quantity
            components[agent_id][component] = (
                components[agent_id].get(component, 0.0) + value
            )

    @staticmethod
    def _add_component(
        components: dict[str, float],
        name: str,
        value: float,
    ) -> None:
        components[name] = components.get(name, 0.0) + value

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        return value

    def _award_once(
        self,
        components: dict[str, float],
        milestone: str,
        value: float,
    ) -> None:
        if milestone in self._milestones:
            return
        self._milestones.add(milestone)
        components[milestone] = value
