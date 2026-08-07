"""Public two-agent reinforcement-learning interface for VoyagerIsland-v1."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from voyager.envs.parallel_env import VoyagerParallelEnv
from voyager.sim.constants import Terrain
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION
from voyager.sim.island_core import island_progress_stage
from voyager.sim.island_registry import ISLAND_ACTION_COUNT, ISLAND_ACTION_VERSION
from voyager.sim.scenarios import (
    ISLAND_BENCHMARK_MAP_SIZE,
    ISLAND_BENCHMARK_MAX_STEPS,
    ISLAND_BENCHMARK_RESCUE_DELAY,
    ISLAND_BENCHMARK_SCENARIO_ID,
)

ISLAND_OBSERVATION_VERSION = "voyager_island_observation_v1"
ISLAND_REWARD_VERSION = "voyager_island_achievement_reward_v1"
ISLAND_ENVIRONMENT_VERSION = "voyager_island_v1"
STRUCTURE_CHANNEL = {"workbench": 1, "campfire": 2, "shelter": 3, "beacon": 4}
CREATURE_CHANNEL = {"island_deer": 1, "night_stalker": 2}


class VoyagerIslandEnv(VoyagerParallelEnv):
    """Canonical PettingZoo ParallelEnv for the frozen island benchmark."""

    metadata = {"render_modes": ["ansi"], "name": "VoyagerIsland-v1"}  # noqa: RUF012

    def __init__(
        self,
        *,
        procedural: bool = True,
        reward_mode: str = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            num_agents=2,
            map_size=ISLAND_BENCHMARK_MAP_SIZE,
            max_steps=ISLAND_BENCHMARK_MAX_STEPS,
            local_view_size=7,
            inventory_capacity=10,
            storm_start_step=10_000,
            storm_interval=0,
            storm_duration=0,
            storm_damage=0.0,
            food_regen_interval=0,
            food_spawn_rate=0.0,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            scenario_id=ISLAND_BENCHMARK_SCENARIO_ID,
            civilization_version=3,
            procedural=procedural,
            render_mode=render_mode,
        )
        observation_space = self._build_observation_space()
        action_space: spaces.Space = spaces.Discrete(ISLAND_ACTION_COUNT)
        self.observation_spaces = {agent_id: observation_space for agent_id in self.possible_agents}
        self.action_spaces = {agent_id: action_space for agent_id in self.possible_agents}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        if options is not None and "procedural" in options:
            self.world.procedural = bool(options["procedural"])
        return super().reset(seed=seed, options=options)

    def action_mask(self, agent_id: str) -> np.ndarray:
        return self.world.island_action_mask(agent_id)

    def global_state(self) -> dict[str, object]:
        return self.world.global_state()

    def _build_observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "local_tiles": spaces.Box(0, 255, shape=(7, 7, 7), dtype=np.uint8),
                "self_state": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
                "inventory": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
                "tools": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
                "identity": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
                "camp_bearing": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "public_board": spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(ISLAND_ACTION_COUNT,), dtype=np.int8),
            }
        )

    def _observation(self, agent_id: str) -> dict[str, np.ndarray]:
        state = self.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before producing observations.")
        agent = state.agents[agent_id]
        local_tiles = np.zeros((7, 7, 7), dtype=np.uint8)
        local_tiles[:, :, 0] = Terrain.WATER
        occupied = self.world.occupied_positions()
        structures = {(value.x, value.y): value for value in state.structures.values()}
        creatures = {(value.x, value.y): value for value in state.creatures.values() if value.alive}
        ambient = float(self.world.civilization_time()["ambient_light"])
        for row, y in enumerate(range(agent.y - 3, agent.y + 4)):
            for column, x in enumerate(range(agent.x - 3, agent.x + 4)):
                if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                    continue
                local_tiles[row, column, 0] = state.terrain[y, x]
                if state.resource_quantities[y, x] > 0:
                    local_tiles[row, column, 1] = state.resource_ids[y, x]
                    local_tiles[row, column, 2] = state.resource_quantities[y, x]
                structure = structures.get((x, y))
                if structure is not None:
                    local_tiles[row, column, 3] = STRUCTURE_CHANNEL[structure.type]
                creature = creatures.get((x, y))
                if creature is not None:
                    local_tiles[row, column, 4] = CREATURE_CHANNEL[creature.type]
                visible_agent = occupied.get((x, y))
                if visible_agent is not None:
                    local_tiles[row, column, 5] = int(visible_agent.rsplit("_", 1)[1]) + 1
                local_tiles[row, column, 6] = round(ambient * 255)

        food_counts = {
            "food": sum(
                lot.quantity for lot in agent.food_lots if lot.kind in {"wreck_ration", "berries"}
            ),
            "raw_meat": sum(lot.quantity for lot in agent.food_lots if lot.kind == "raw_meat"),
            "cooked_meat": sum(
                lot.quantity for lot in agent.food_lots if lot.kind == "cooked_meat"
            ),
        }
        inventory_total = (
            agent.inventory.get("wood", 0)
            + agent.inventory.get("stone", 0)
            + sum(food_counts.values())
        )
        identity = np.zeros(2, dtype=np.int8)
        identity[int(agent_id.rsplit("_", 1)[1])] = 1
        camp_dx = state.camp.x - agent.x
        camp_dy = state.camp.y - agent.y
        structures_progress = [
            state.structures[name].progress
            for name in ("workbench", "campfire", "shelter", "beacon")
        ]
        tick = state.step_count
        return {
            "local_tiles": local_tiles,
            "self_state": np.array(
                [
                    agent.health / 100.0,
                    agent.hunger / 100.0,
                    agent.energy / 100.0,
                    inventory_total / self.inventory_capacity,
                    float(agent.sheltered),
                    float(agent.alive),
                ],
                dtype=np.float32,
            ),
            "inventory": np.array(
                [
                    food_counts["food"] / self.inventory_capacity,
                    agent.inventory.get("wood", 0) / self.inventory_capacity,
                    agent.inventory.get("stone", 0) / self.inventory_capacity,
                    food_counts["raw_meat"] / self.inventory_capacity,
                    food_counts["cooked_meat"] / self.inventory_capacity,
                ],
                dtype=np.float32,
            ),
            "tools": np.array(
                [int("axe" in agent.tools), int("spear" in agent.tools)], dtype=np.int8
            ),
            "identity": identity,
            "camp_bearing": np.array(
                [
                    camp_dx / (self.map_size - 1),
                    camp_dy / (self.map_size - 1),
                    (abs(camp_dx) + abs(camp_dy)) / (2 * (self.map_size - 1)),
                ],
                dtype=np.float32,
            ),
            "public_board": np.array(
                [
                    state.camp.stockpile.get("food", 0) / 40.0,
                    state.camp.stockpile.get("wood", 0) / 40.0,
                    state.camp.stockpile.get("stone", 0) / 40.0,
                    state.camp.stockpile.get("raw_meat", 0) / 40.0,
                    state.camp.stockpile.get("cooked_meat", 0) / 40.0,
                    *structures_progress,
                    len(self.world.alive_agents()) / 2.0,
                    tick / self.max_steps,
                    float(200 <= tick % 300 < 300),
                ],
                dtype=np.float32,
            ),
            "action_mask": self.action_mask(agent_id),
        }

    def _info(
        self,
        agent_id: str,
        event: str,
        reward_components: dict[str, float] | None = None,
        dense_reward_components: dict[str, float] | None = None,
        new_achievements: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        info = super()._info(
            agent_id,
            event,
            reward_components,
            dense_reward_components,
            new_achievements,
        )
        state = self.world.state
        if state is None:
            return info
        beacon_step = state.achievement_steps.get("build_beacon")
        rescue_ticks_remaining = (
            max(
                0,
                ISLAND_BENCHMARK_RESCUE_DELAY - (state.step_count - beacon_step),
                300 - state.step_count,
            )
            if beacon_step is not None
            else None
        )
        info.update(
            {
                "environment_version": ISLAND_ENVIRONMENT_VERSION,
                "scenario_id": ISLAND_BENCHMARK_SCENARIO_ID,
                "observation_version": ISLAND_OBSERVATION_VERSION,
                "action_version": ISLAND_ACTION_VERSION,
                "reward_version": ISLAND_REWARD_VERSION,
                "achievement_version": ISLAND_ACHIEVEMENT_VERSION,
                "time": self.world.civilization_time(),
                "rescue_success": state.rescue_success,
                "technology_stage": island_progress_stage(state),
                "rescue_ticks_remaining": rescue_ticks_remaining,
                "structures": {
                    name: self.world._structure_payload(structure)
                    for name, structure in sorted(state.structures.items())
                },
                "events": list(state.events),
                "action_mask": self.action_mask(agent_id),
            }
        )
        return info


class VoyagerIslandCentralizedEnv(gym.Env[tuple[dict[str, np.ndarray], ...], np.ndarray]):
    """Optional Gymnasium wrapper for a controller that emits both agent actions."""

    metadata = VoyagerIslandEnv.metadata

    def __init__(self, *, procedural: bool = True, render_mode: str | None = None) -> None:
        self.env = VoyagerIslandEnv(procedural=procedural, render_mode=render_mode)
        agent_space = self.env.observation_space(self.env.possible_agents[0])
        self.observation_space = spaces.Tuple((agent_space, agent_space))
        self.action_space = spaces.MultiDiscrete([ISLAND_ACTION_COUNT, ISLAND_ACTION_COUNT])
        self._last_observations: dict[str, dict[str, np.ndarray]] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[dict[str, np.ndarray], ...], dict[str, Any]]:
        super().reset(seed=seed)
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        return self._joint_observation(), {"agents": infos}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[tuple[dict[str, np.ndarray], ...], float, bool, bool, dict[str, Any]]:
        values = np.asarray(action, dtype=np.int64)
        if values.shape != (2,):
            raise ValueError("Centralized island actions must have shape (2,).")
        action_map = {
            agent_id: int(values[index])
            for index, agent_id in enumerate(self.env.possible_agents)
            if agent_id in self.env.agents
        }
        observations, rewards, terminations, truncations, infos = self.env.step(action_map)
        self._last_observations.update(observations)
        terminated = bool(terminations) and all(terminations.values())
        truncated = any(truncations.values())
        return (
            self._joint_observation(),
            float(sum(rewards.values())),
            terminated,
            truncated,
            {"agents": infos},
        )

    def render(self) -> str | None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _joint_observation(self) -> tuple[dict[str, np.ndarray], ...]:
        if self.env.world.state is None:
            return tuple(self._last_observations[agent_id] for agent_id in self.env.possible_agents)
        return tuple(self.env._observation(agent_id) for agent_id in self.env.possible_agents)
