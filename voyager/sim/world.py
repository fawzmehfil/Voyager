"""Single-agent survival world mechanics for Voyager Stage 1."""

from dataclasses import dataclass

import numpy as np

from voyager.sim.constants import Action, Resource, Terrain
from voyager.sim.mapgen import generate_island
from voyager.sim.state import WorldState

RESOURCE_NAMES = {
    Resource.FOOD: "food",
    Resource.WOOD: "wood",
    Resource.STONE: "stone",
}


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of applying one simulation action."""

    reward: float
    terminated: bool
    truncated: bool
    event: str


class SingleAgentWorld:
    """Deterministic single-agent island survival simulation."""

    def __init__(self, map_size: int, max_steps: int, inventory_capacity: int) -> None:
        self.map_size = map_size
        self.max_steps = max_steps
        self.inventory_capacity = inventory_capacity
        self.state: WorldState | None = None

    def reset(self, rng: np.random.Generator) -> WorldState:
        """Reset the world with a freshly generated seeded island."""

        self.state = generate_island(self.map_size, rng)
        return self.state

    def step(self, action: Action) -> StepResult:
        """Apply an action and update survival state."""

        state = self._require_state()
        state.step_count += 1

        reward = 0.01
        event = "noop"
        invalid = False

        if action in {
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        }:
            event, invalid = self._move(action)
        elif action == Action.GATHER:
            event, invalid, gather_reward = self._gather()
            reward += gather_reward
        elif action == Action.EAT:
            event, invalid, eat_reward = self._eat()
            reward += eat_reward
        elif action == Action.REST:
            event, invalid, rest_reward = self._rest()
            reward += rest_reward
        elif action == Action.NOOP:
            event = "noop"

        if invalid:
            reward -= 0.02

        self._apply_survival_pressure()
        agent = state.agent
        reward -= agent.hunger * 0.0005

        terminated = agent.health <= 0.0
        truncated = state.step_count >= self.max_steps
        if terminated:
            reward -= 10.0
            event = "death"

        return StepResult(
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            event=event,
        )

    def _move(self, action: Action) -> tuple[str, bool]:
        state = self._require_state()
        agent = state.agent
        dx, dy = {
            Action.MOVE_UP: (0, -1),
            Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (-1, 0),
            Action.MOVE_RIGHT: (1, 0),
        }[action]
        target_x = agent.x + dx
        target_y = agent.y + dy

        if agent.energy < 1.5:
            return "invalid_no_energy", True
        if not self._in_bounds(target_x, target_y):
            return "invalid_out_of_bounds", True
        if state.terrain[target_y, target_x] == Terrain.WATER:
            return "invalid_water_blocked", True

        agent.x = target_x
        agent.y = target_y
        agent.energy = max(0.0, agent.energy - 1.5)
        return "move", False

    def _gather(self) -> tuple[str, bool, float]:
        state = self._require_state()
        agent = state.agent
        resource = Resource(int(state.resource_ids[agent.y, agent.x]))
        quantity = int(state.resource_quantities[agent.y, agent.x])

        if agent.energy < 2.0:
            return "invalid_no_energy", True, 0.0
        if resource == Resource.NONE or quantity <= 0:
            return "invalid_no_resource", True, 0.0

        name = RESOURCE_NAMES[resource]
        if agent.inventory[name] >= self.inventory_capacity:
            return f"invalid_{name}_full", True, 0.0

        agent.inventory[name] += 1
        agent.energy = max(0.0, agent.energy - 2.0)
        state.resource_quantities[agent.y, agent.x] = quantity - 1
        if state.resource_quantities[agent.y, agent.x] == 0:
            state.resource_ids[agent.y, agent.x] = Resource.NONE
        return f"gather_{name}", False, 0.10

    def _eat(self) -> tuple[str, bool, float]:
        agent = self._require_state().agent
        if agent.inventory["food"] <= 0:
            return "invalid_no_food", True, 0.0

        was_meaningfully_hungry = agent.hunger >= 35.0
        agent.inventory["food"] -= 1
        agent.hunger = max(0.0, agent.hunger - 35.0)
        return "eat", False, 0.30 if was_meaningfully_hungry else 0.0

    def _rest(self) -> tuple[str, bool, float]:
        agent = self._require_state().agent
        was_low_energy = agent.energy <= 50.0
        if agent.energy >= 100.0:
            return "invalid_full_energy", True, 0.0
        agent.energy = min(100.0, agent.energy + 10.0)
        return "rest", False, 0.05 if was_low_energy else 0.0

    def _apply_survival_pressure(self) -> None:
        agent = self._require_state().agent
        agent.hunger = min(100.0, agent.hunger + 0.35)
        if agent.hunger > 80.0:
            agent.health = max(0.0, agent.health - ((agent.hunger - 80.0) * 0.05))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _require_state(self) -> WorldState:
        if self.state is None:
            raise RuntimeError("World must be reset before stepping.")
        return self.state
