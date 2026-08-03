"""Provisional shared reward and outcome metrics for the Stage 7C trainability gate."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.envs.civilization_v2 import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
)

PROBE_REWARD_VERSION = "civilization_trainability_probe_v1"


class CivilizationProbeRewardWrapper(
    ParallelEnv[str, dict[str, np.ndarray], int]
):
    """Replace v2 rewards with a shared, first-time progression probe reward."""

    metadata = CivilizationV2FlattenedActionWrapper.metadata

    def __init__(
        self,
        env: CivilizationV2FlattenedActionWrapper | None = None,
    ) -> None:
        self.env = env or CivilizationV2FlattenedActionWrapper(
            VoyagerCivilizationV2Env(reward_mode="none")
        )
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        self.action_spaces = dict(self.env.action_spaces)
        self._milestones: set[str] = set()
        self._ledger_cursor = 0

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
        state = self.world.state
        assert state is not None
        self._ledger_cursor = len(state.ledger)
        for info in infos.values():
            info["reward_version"] = PROBE_REWARD_VERSION
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
            info["reward_version"] = PROBE_REWARD_VERSION
        return observations, rewards, terminations, truncations, infos

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
        state = self.world.state
        assert state is not None
        active_fraction = sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)
        invalid_fraction = sum(
            bool(info.get("invalid_action", False)) for info in infos.values()
        ) / max(1, len(infos))
        events = state.events
        components: dict[str, float] = {
            "survival": 0.001 * active_fraction,
            "invalid": -0.02 * invalid_fraction,
            "downed": -0.25 * sum(event["type"] == "agent_downed" for event in events),
            "death": -1.0 * sum(event["type"] == "agent_died" for event in events),
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
