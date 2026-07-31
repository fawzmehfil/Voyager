"""PettingZoo parallel environment for Voyager's shared island."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.sim.constants import ACTION_COUNT, ROLE_COUNT, Resource, Terrain
from voyager.sim.multi_world import ROLE_IDS, MultiAgentWorld
from voyager.sim.registries import CivilizationAction
from voyager.sim.rewards import DENSE_REWARD_COMPONENTS, REWARD_MODES, RewardMode
from voyager.sim.scenarios import COMPACT_SCENARIO_ID


class VoyagerParallelEnv(gym.Env, ParallelEnv[str, dict[str, np.ndarray], int]):
    """Multi-agent stranded-island survival environment."""

    metadata = {"render_modes": ["ansi"], "name": "VoyagerSurvival-v0"}  # noqa: RUF012

    def __init__(
        self,
        num_agents: int = 10,
        map_size: int = 32,
        max_steps: int = 1000,
        local_view_size: int = 7,
        inventory_capacity: int = 10,
        storm_start_step: int = 200,
        storm_interval: int = 200,
        storm_duration: int = 25,
        storm_damage: float = 1.0,
        food_regen_interval: int = 50,
        food_spawn_rate: float = 0.04,
        reward_mode: RewardMode = "dense",
        disabled_reward_components: tuple[str, ...] = (),
        mask_role_observation: bool = False,
        scenario_id: str = COMPACT_SCENARIO_ID,
        render_mode: str | None = None,
    ) -> None:
        if map_size < 9:
            raise ValueError("map_size must be at least 9.")
        if local_view_size < 3 or local_view_size % 2 == 0:
            raise ValueError("local_view_size must be an odd integer >= 3.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward_mode: {reward_mode!r}.")
        unknown_components = set(disabled_reward_components) - DENSE_REWARD_COMPONENTS
        if unknown_components:
            names = ", ".join(sorted(unknown_components))
            raise ValueError(f"Unknown disabled reward components: {names}")

        self._configured_num_agents = num_agents
        self.map_size = map_size
        self.max_steps = max_steps
        self.local_view_size = local_view_size
        self.inventory_capacity = inventory_capacity
        self.reward_mode = reward_mode
        self.disabled_reward_components = frozenset(disabled_reward_components)
        self.mask_role_observation = mask_role_observation
        self.render_mode = render_mode
        self.world = MultiAgentWorld(
            num_agents=num_agents,
            map_size=map_size,
            max_steps=max_steps,
            inventory_capacity=inventory_capacity,
            storm_start_step=storm_start_step,
            storm_interval=storm_interval,
            storm_duration=storm_duration,
            storm_damage=storm_damage,
            food_regen_interval=food_regen_interval,
            food_spawn_rate=food_spawn_rate,
            scenario_id=scenario_id,
        )
        self.possible_agents = list(self.world.possible_agents)
        self.agents: list[str] = []

        observation_space = self._build_observation_space()
        action_space: spaces.Discrete = spaces.Discrete(ACTION_COUNT)
        self.observation_spaces: dict[str, spaces.Space] = {
            agent_id: observation_space for agent_id in self.possible_agents
        }
        self.action_spaces: dict[str, spaces.Space] = {
            agent_id: action_space for agent_id in self.possible_agents
        }
        self._np_random = np.random.default_rng()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        _ = options
        self._np_random = np.random.default_rng(seed)
        self.world.reset(self._np_random)
        self.agents = self.world.alive_agents()
        observations = {agent_id: self._observation(agent_id) for agent_id in self.agents}
        infos = {agent_id: self._info(agent_id, "reset") for agent_id in self.agents}
        return observations, infos

    def step(  # type: ignore[override]
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        acting_agents = list(self.agents)
        world_actions: dict[str, int | CivilizationAction] = dict(actions)
        results = self.world.step(world_actions)
        max_step_reached = (
            self.world.state is not None and self.world.state.step_count >= self.max_steps
        )

        selected_components = {
            agent_id: self._selected_reward_components(results[agent_id])
            for agent_id in acting_agents
        }
        rewards = {
            agent_id: float(sum(selected_components[agent_id].values()))
            for agent_id in acting_agents
        }
        terminations = {agent_id: results[agent_id].terminated for agent_id in acting_agents}
        truncations = {agent_id: results[agent_id].truncated for agent_id in acting_agents}
        infos = {
            agent_id: self._info(
                agent_id,
                results[agent_id].event,
                selected_components[agent_id],
                dense_reward_components=results[agent_id].reward_components,
                new_achievements=results[agent_id].new_achievements,
            )
            for agent_id in acting_agents
        }

        self.agents = [] if max_step_reached else self.world.alive_agents()
        observations = {agent_id: self._observation(agent_id) for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        state = self.world.state
        if state is None:
            return "Voyager parallel environment has not been reset."

        terrain_chars = {
            Terrain.WATER: "~",
            Terrain.BEACH: ".",
            Terrain.GRASS: ",",
            Terrain.FOREST: "T",
            Terrain.QUARRY: "^",
        }
        resource_chars = {
            Resource.FOOD: "f",
            Resource.WOOD: "w",
            Resource.STONE: "s",
        }
        occupied = self.world.occupied_positions()
        rows: list[str] = []
        for y in range(self.map_size):
            chars: list[str] = []
            for x in range(self.map_size):
                if (x, y) in occupied:
                    chars.append(occupied[(x, y)].split("_", maxsplit=1)[1][-1])
                    continue
                if (x, y) == (state.camp.x, state.camp.y):
                    chars.append("C")
                    continue
                resource = Resource(int(state.resource_ids[y, x]))
                quantity = int(state.resource_quantities[y, x])
                if resource != Resource.NONE and quantity > 0:
                    chars.append(resource_chars[resource])
                else:
                    chars.append(terrain_chars[Terrain(int(state.terrain[y, x]))])
            rows.append("".join(chars))

        summary = (
            f"step={state.step_count} active={len(self.agents)} deaths={state.deaths} "
            f"camp=({state.camp.x},{state.camp.y}) stockpile={state.camp.stockpile}"
        )
        return "\n".join([summary, *rows])

    def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self.action_spaces[agent]

    def metrics(self) -> dict[str, object]:
        """Return global survival economy metrics."""

        return self.world.metrics()

    def action_mask(self, agent_id: str) -> np.ndarray:
        """Return currently legal and useful actions for one agent."""

        return self.world.action_mask(agent_id)

    def _build_observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "local_view": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.local_view_size, self.local_view_size, 4),
                    dtype=np.uint8,
                ),
                "stats": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "inventory": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "role": spaces.Box(low=0.0, high=1.0, shape=(ROLE_COUNT,), dtype=np.float32),
                "camp": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "progress": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _observation(self, agent_id: str) -> dict[str, np.ndarray]:
        state = self.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before producing observations.")
        agent = state.agents[agent_id]
        half = self.local_view_size // 2
        local_view = np.zeros(
            (self.local_view_size, self.local_view_size, 4),
            dtype=np.uint8,
        )
        local_view[:, :, 0] = Terrain.WATER
        occupied = self.world.occupied_positions()

        for row, y in enumerate(range(agent.y - half, agent.y + half + 1)):
            for col, x in enumerate(range(agent.x - half, agent.x + half + 1)):
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    local_view[row, col, 0] = state.terrain[y, x]
                    if state.resource_quantities[y, x] > 0:
                        local_view[row, col, 1] = state.resource_ids[y, x]
                    if (x, y) in occupied:
                        local_view[row, col, 2] = 255
                    if (x, y) == (state.camp.x, state.camp.y):
                        local_view[row, col, 3] = 255

        role = np.zeros(ROLE_COUNT, dtype=np.float32)
        if not self.mask_role_observation:
            role[ROLE_IDS[agent.role]] = 1.0

        return {
            "local_view": local_view,
            "stats": np.array(
                [agent.health / 100.0, agent.hunger / 100.0, agent.energy / 100.0],
                dtype=np.float32,
            ),
            "inventory": np.array(
                [
                    agent.inventory["food"] / self.inventory_capacity,
                    agent.inventory["wood"] / self.inventory_capacity,
                    agent.inventory["stone"] / self.inventory_capacity,
                ],
                dtype=np.float32,
            ),
            "role": role,
            "camp": np.array(
                [
                    state.camp.stockpile["food"]
                    / max(1, self.inventory_capacity * self._configured_num_agents),
                    state.camp.stockpile["wood"]
                    / max(1, self.inventory_capacity * self._configured_num_agents),
                    state.camp.stockpile["stone"]
                    / max(1, self.inventory_capacity * self._configured_num_agents),
                    state.camp.shelter_progress,
                ],
                dtype=np.float32,
            ),
            "progress": np.array([state.step_count / self.max_steps], dtype=np.float32),
        }

    def _info(
        self,
        agent_id: str,
        event: str,
        reward_components: dict[str, float] | None = None,
        dense_reward_components: dict[str, float] | None = None,
        new_achievements: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        state = self.world.state
        if state is None:
            return {"event": event}
        agent = state.agents[agent_id]
        return {
            "event": event,
            "invalid_action": event.startswith("invalid_"),
            "step": state.step_count,
            "position": (agent.x, agent.y),
            "role": agent.role,
            "health": agent.health,
            "hunger": agent.hunger,
            "energy": agent.energy,
            "inventory": dict(agent.inventory),
            "camp": {
                "position": (state.camp.x, state.camp.y),
                "stockpile": dict(state.camp.stockpile),
                "shelter_progress": state.camp.shelter_progress,
            },
            "storm_active": self.world.is_storm_active(),
            "achievements": sorted(state.achievements),
            "action_mask": self.action_mask(agent_id),
            "reward_components": dict(reward_components or {}),
            "dense_reward_components": dict(
                dense_reward_components
                if dense_reward_components is not None
                else reward_components or {}
            ),
            "new_achievements": list(new_achievements),
        }

    def _selected_reward_components(self, result: Any) -> dict[str, float]:
        if self.reward_mode == "achievement":
            return {"achievement": float(len(result.new_achievements))}
        if self.reward_mode == "none":
            return {}
        return {
            name: value
            for name, value in result.reward_components.items()
            if name not in self.disabled_reward_components
        }
