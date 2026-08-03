"""Versioned shared rewards and actor observations for Stage 7C trainability."""

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
PROBE_REWARD_VERSION = "civilization_trainability_probe_v2"
PROBE_REWARD_VERSIONS = frozenset({PROBE_REWARD_V1, PROBE_REWARD_VERSION})


class CivilizationProbeRewardWrapper(
    ParallelEnv[str, dict[str, np.ndarray], int]
):
    """Expose a versioned shared probe reward over the deterministic v2 world."""

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
        if reward_version == PROBE_REWARD_VERSION:
            self.observation_spaces = {
                agent_id: self._identity_observation_space(space)
                for agent_id, space in self.observation_spaces.items()
            }
        self.action_spaces = dict(self.env.action_spaces)
        self._milestones: set[str] = set()
        self._ledger_cursor = 0
        self._gathered_units: Counter[str] = Counter()
        self._credited_deposits: Counter[str] = Counter()
        self._visited_positions: set[tuple[int, int]] = set()

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
            augmented[agent_id] = {
                **observation,
                "agent_identity": identity,
            }
        return augmented

    def _identity_observation_space(self, space: spaces.Space) -> spaces.Dict:
        if not isinstance(space, spaces.Dict):
            raise TypeError("Civilization v2 requires a Dict observation space.")
        return spaces.Dict(
            {
                **space.spaces,
                "agent_identity": spaces.Box(
                    0,
                    1,
                    shape=(len(self.possible_agents),),
                    dtype=np.int8,
                ),
            }
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
