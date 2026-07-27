"""Gymnasium environment for the Stage 1 single-agent Voyager prototype."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from voyager.sim.constants import ACTION_COUNT, Action, Resource, Terrain
from voyager.sim.world import SingleAgentWorld


class VoyagerSingleAgentEnv(gym.Env):
    """Single-agent stranded-island survival environment."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}  # noqa: RUF012

    def __init__(
        self,
        map_size: int = 32,
        max_steps: int = 1000,
        local_view_size: int = 7,
        inventory_capacity: int = 10,
        render_mode: str | None = None,
    ) -> None:
        if map_size < 9:
            raise ValueError("map_size must be at least 9.")
        if local_view_size < 3 or local_view_size % 2 == 0:
            raise ValueError("local_view_size must be an odd integer >= 3.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

        self.map_size = map_size
        self.max_steps = max_steps
        self.local_view_size = local_view_size
        self.inventory_capacity = inventory_capacity
        self.render_mode = render_mode
        self.world = SingleAgentWorld(
            map_size=map_size,
            max_steps=max_steps,
            inventory_capacity=inventory_capacity,
        )

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Dict(
            {
                "local_view": spaces.Box(
                    low=0,
                    high=255,
                    shape=(local_view_size, local_view_size, 3),
                    dtype=np.uint8,
                ),
                "stats": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "inventory": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "progress": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        self.world.reset(self.np_random)
        return self._observation(), self._info("reset")

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        try:
            parsed_action = Action(int(action))
            result = self.world.step(parsed_action)
            reward = result.reward
            event = result.event
            terminated = result.terminated
            truncated = result.truncated
        except ValueError:
            result = self.world.step(Action.NOOP)
            reward = result.reward - 0.02
            event = "invalid_action"
            terminated = result.terminated
            truncated = result.truncated

        return self._observation(), reward, terminated, truncated, self._info(event)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        state = self.world.state
        if state is None:
            return "Voyager environment has not been reset."

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

        rows: list[str] = []
        for y in range(self.map_size):
            chars: list[str] = []
            for x in range(self.map_size):
                if x == state.agent.x and y == state.agent.y:
                    chars.append("@")
                    continue
                resource = Resource(int(state.resource_ids[y, x]))
                quantity = int(state.resource_quantities[y, x])
                if resource != Resource.NONE and quantity > 0:
                    chars.append(resource_chars[resource])
                else:
                    chars.append(terrain_chars[Terrain(int(state.terrain[y, x]))])
            rows.append("".join(chars))

        agent = state.agent
        summary = (
            f"step={state.step_count} health={agent.health:.1f} "
            f"hunger={agent.hunger:.1f} energy={agent.energy:.1f} "
            f"inventory={agent.inventory}"
        )
        return "\n".join([summary, *rows])

    def _observation(self) -> dict[str, np.ndarray]:
        state = self.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before producing observations.")

        agent = state.agent
        half = self.local_view_size // 2
        local_view = np.zeros(
            (self.local_view_size, self.local_view_size, 3),
            dtype=np.uint8,
        )
        local_view[:, :, 0] = Terrain.WATER

        for row, y in enumerate(range(agent.y - half, agent.y + half + 1)):
            for col, x in enumerate(range(agent.x - half, agent.x + half + 1)):
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    local_view[row, col, 0] = state.terrain[y, x]
                    quantity = state.resource_quantities[y, x]
                    if quantity > 0:
                        local_view[row, col, 1] = state.resource_ids[y, x]

        local_view[half, half, 2] = 255

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
            "progress": np.array([state.step_count / self.max_steps], dtype=np.float32),
        }

    def _info(self, event: str) -> dict[str, Any]:
        state = self.world.state
        if state is None:
            return {"event": event}

        agent = state.agent
        return {
            "event": event,
            "step": state.step_count,
            "position": (agent.x, agent.y),
            "health": agent.health,
            "hunger": agent.hunger,
            "energy": agent.energy,
            "inventory": dict(agent.inventory),
        }
