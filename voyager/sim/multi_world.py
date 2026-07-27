"""Multi-agent island world mechanics for Voyager Stage 2."""

from dataclasses import dataclass

import numpy as np

from voyager.sim.constants import Action, Resource, Role, Terrain
from voyager.sim.mapgen import generate_island
from voyager.sim.state import AgentState, CampState, MultiAgentWorldState
from voyager.sim.world import RESOURCE_NAMES

ROLE_NAMES = {
    Role.FORAGER: "forager",
    Role.WOODCUTTER: "woodcutter",
    Role.BUILDER: "builder",
}
ROLE_IDS = {name: role for role, name in ROLE_NAMES.items()}


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    """Outcome for one agent in a simultaneous multi-agent step."""

    reward: float
    terminated: bool
    truncated: bool
    event: str


class MultiAgentWorld:
    """Shared island simulation for PettingZoo parallel agents."""

    def __init__(
        self,
        num_agents: int,
        map_size: int,
        max_steps: int,
        inventory_capacity: int,
        storm_start_step: int = 200,
        storm_interval: int = 200,
        storm_duration: int = 25,
        storm_damage: float = 1.0,
        food_regen_interval: int = 50,
        food_spawn_rate: float = 0.04,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")
        self.num_agents = num_agents
        self.map_size = map_size
        self.max_steps = max_steps
        self.inventory_capacity = inventory_capacity
        self.storm_start_step = storm_start_step
        self.storm_interval = storm_interval
        self.storm_duration = storm_duration
        self.storm_damage = storm_damage
        self.food_regen_interval = food_regen_interval
        self.food_spawn_rate = food_spawn_rate
        self.possible_agents = [f"agent_{index}" for index in range(num_agents)]
        self.state: MultiAgentWorldState | None = None
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> MultiAgentWorldState:
        """Generate a fresh seeded shared island and spawn all agents."""

        self.rng = rng
        single_state = generate_island(self.map_size, rng)
        camp = CampState(
            x=single_state.agent.x,
            y=single_state.agent.y,
            shelter_capacity=self.num_agents,
        )
        spawns = self._spawn_positions(single_state.terrain, camp.x, camp.y)
        agents: dict[str, AgentState] = {}
        for index, agent_id in enumerate(self.possible_agents):
            x, y = spawns[index]
            role = ROLE_NAMES[Role(index % len(Role))]
            agents[agent_id] = AgentState(x=x, y=y, role=role)

        self.state = MultiAgentWorldState(
            terrain=single_state.terrain,
            resource_ids=single_state.resource_ids,
            resource_quantities=single_state.resource_quantities,
            agents=agents,
            camp=camp,
        )
        return self.state

    def step(self, actions: dict[str, int]) -> dict[str, AgentStepResult]:
        """Apply one stable-order simultaneous step for all currently living agents."""

        state = self._require_state()
        state.step_count += 1
        occupied = {
            (agent.x, agent.y)
            for agent in state.agents.values()
            if agent.alive
        }
        results: dict[str, AgentStepResult] = {}

        for agent_id in self.possible_agents:
            agent = state.agents[agent_id]
            if not agent.alive:
                continue

            action = self._parse_action(actions.get(agent_id, Action.NOOP))
            reward = 0.01
            event = "noop"
            invalid = False

            if action in {
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
            }:
                event, invalid = self._move(agent, action, occupied)
            elif action == Action.GATHER:
                event, invalid, action_reward = self._gather(agent)
                reward += action_reward
            elif action == Action.EAT:
                event, invalid, action_reward = self._eat(agent)
                reward += action_reward
            elif action == Action.REST:
                event, invalid, action_reward = self._rest(agent)
                reward += action_reward
            elif action in {
                Action.DEPOSIT_FOOD,
                Action.DEPOSIT_WOOD,
                Action.DEPOSIT_STONE,
            }:
                event, invalid, action_reward = self._deposit(agent, action)
                reward += action_reward
            elif action == Action.WITHDRAW_FOOD:
                event, invalid, action_reward = self._withdraw_food(agent)
                reward += action_reward
            elif action == Action.BUILD_SHELTER:
                event, invalid, action_reward = self._build_shelter(agent)
                reward += action_reward
            elif action == Action.NOOP:
                event = "noop"

            if invalid:
                reward -= 0.02

            self._apply_survival_pressure(agent)
            reward -= agent.hunger * 0.0005

            terminated = agent.health <= 0.0
            truncated = state.step_count >= self.max_steps
            if terminated:
                agent.alive = False
                state.deaths += 1
                occupied.discard((agent.x, agent.y))
                reward -= 10.0
                event = "death"

            results[agent_id] = AgentStepResult(
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                event=event,
            )

        self._apply_storm_effects(results)
        self._regenerate_food()
        self._update_achievements()
        return results

    def alive_agents(self) -> list[str]:
        """Return live agents in stable possible-agent order."""

        state = self._require_state()
        return [
            agent_id
            for agent_id in self.possible_agents
            if state.agents[agent_id].alive
        ]

    def occupied_positions(self) -> dict[tuple[int, int], str]:
        """Return occupied live-agent positions keyed by coordinate."""

        state = self._require_state()
        return {
            (agent.x, agent.y): agent_id
            for agent_id, agent in state.agents.items()
            if agent.alive
        }

    def is_storm_active(self) -> bool:
        """Return whether a deterministic storm is active at the current step."""

        state = self._require_state()
        if state.step_count < self.storm_start_step:
            return False
        if self.storm_interval <= 0 or self.storm_duration <= 0:
            return False
        return ((state.step_count - self.storm_start_step) % self.storm_interval) < self.storm_duration

    def metrics(self) -> dict[str, object]:
        """Return JSON-like survival economy metrics."""

        state = self._require_state()
        return {
            "step": state.step_count,
            "active_agents": len(self.alive_agents()),
            "deaths": state.deaths,
            "camp": {
                "position": (state.camp.x, state.camp.y),
                "stockpile": dict(state.camp.stockpile),
                "shelter_progress": state.camp.shelter_progress,
                "shelter_capacity": state.camp.shelter_capacity,
            },
            "storm_active": self.is_storm_active(),
            "achievements": sorted(state.achievements),
            "total_deposits": state.total_deposits,
            "total_withdrawals": state.total_withdrawals,
            "total_build_actions": state.total_build_actions,
        }

    def _move(
        self,
        agent: AgentState,
        action: Action,
        occupied: set[tuple[int, int]],
    ) -> tuple[str, bool]:
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
        state = self._require_state()
        if state.terrain[target_y, target_x] == Terrain.WATER:
            return "invalid_water_blocked", True
        if (target_x, target_y) in occupied:
            return "invalid_occupied", True

        occupied.discard((agent.x, agent.y))
        agent.x = target_x
        agent.y = target_y
        occupied.add((agent.x, agent.y))
        agent.energy = max(0.0, agent.energy - 1.5)
        return "move", False

    def _gather(self, agent: AgentState) -> tuple[str, bool, float]:
        state = self._require_state()
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

    def _eat(self, agent: AgentState) -> tuple[str, bool, float]:
        if agent.inventory["food"] <= 0:
            return "invalid_no_food", True, 0.0

        was_meaningfully_hungry = agent.hunger >= 35.0
        agent.inventory["food"] -= 1
        agent.hunger = max(0.0, agent.hunger - 35.0)
        return "eat", False, 0.30 if was_meaningfully_hungry else 0.0

    def _rest(self, agent: AgentState) -> tuple[str, bool, float]:
        was_low_energy = agent.energy <= 50.0
        if agent.energy >= 100.0:
            return "invalid_full_energy", True, 0.0
        agent.energy = min(100.0, agent.energy + 10.0)
        return "rest", False, 0.05 if was_low_energy else 0.0

    def _deposit(self, agent: AgentState, action: Action) -> tuple[str, bool, float]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0

        resource_name = {
            Action.DEPOSIT_FOOD: "food",
            Action.DEPOSIT_WOOD: "wood",
            Action.DEPOSIT_STONE: "stone",
        }[action]
        if agent.inventory[resource_name] <= 0:
            return f"invalid_no_{resource_name}", True, 0.0

        agent.inventory[resource_name] -= 1
        state.camp.stockpile[resource_name] += 1
        state.total_deposits += 1
        state.achievements.add("first_deposit")
        return f"deposit_{resource_name}", False, 0.08

    def _withdraw_food(self, agent: AgentState) -> tuple[str, bool, float]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0
        if state.camp.stockpile["food"] <= 0:
            return "invalid_camp_no_food", True, 0.0
        if agent.inventory["food"] >= self.inventory_capacity:
            return "invalid_food_full", True, 0.0

        state.camp.stockpile["food"] -= 1
        agent.inventory["food"] += 1
        state.total_withdrawals += 1
        state.achievements.add("first_food_withdrawal")
        return "withdraw_food", False, 0.04

    def _build_shelter(self, agent: AgentState) -> tuple[str, bool, float]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0
        if state.camp.shelter_progress >= 1.0:
            return "invalid_shelter_complete", True, 0.0

        material = ""
        if agent.inventory["wood"] > 0:
            material = "wood"
        elif agent.inventory["stone"] > 0:
            material = "stone"
        else:
            return "invalid_no_build_material", True, 0.0

        agent.inventory[material] -= 1
        build_amount = 0.06 if agent.role == "builder" else 0.04
        state.camp.shelter_progress = min(1.0, state.camp.shelter_progress + build_amount)
        state.total_build_actions += 1
        self._update_shelter_achievements()
        return f"build_shelter_{material}", False, 0.15

    def _apply_survival_pressure(self, agent: AgentState) -> None:
        agent.hunger = min(100.0, agent.hunger + 0.35)
        if agent.hunger > 80.0:
            agent.health = max(0.0, agent.health - ((agent.hunger - 80.0) * 0.05))

    def _apply_storm_effects(self, results: dict[str, AgentStepResult]) -> None:
        state = self._require_state()
        storm_active = self.is_storm_active()
        if not storm_active:
            if state.storm_was_active and self.alive_agents():
                state.achievements.add("first_storm_survived")
            state.storm_was_active = False
            return

        state.storm_was_active = True
        damage = self.storm_damage * max(0.0, 1.0 - state.camp.shelter_progress)
        if damage <= 0.0:
            return

        for agent_id in self.possible_agents:
            agent = state.agents[agent_id]
            if not agent.alive:
                continue
            agent.health = max(0.0, agent.health - damage)
            if agent.health <= 0.0:
                agent.alive = False
                state.deaths += 1
                if agent_id in results:
                    previous = results[agent_id]
                    results[agent_id] = AgentStepResult(
                        reward=previous.reward - 10.0,
                        terminated=True,
                        truncated=previous.truncated,
                        event="death",
                    )

    def _regenerate_food(self) -> None:
        state = self._require_state()
        if self.food_regen_interval <= 0:
            return
        if state.step_count == 0 or state.step_count % self.food_regen_interval != 0:
            return

        candidate_positions = np.argwhere(
            ((state.terrain == Terrain.GRASS) | (state.terrain == Terrain.FOREST))
            & (state.resource_quantities == 0)
        )
        if len(candidate_positions) == 0:
            return

        spawn_count = max(1, round(self.map_size * self.food_spawn_rate * 4))
        selected_indexes = self.rng.choice(
            len(candidate_positions),
            size=min(spawn_count, len(candidate_positions)),
            replace=False,
        )
        for index in np.atleast_1d(selected_indexes):
            y, x = candidate_positions[int(index)]
            state.resource_ids[y, x] = Resource.FOOD
            state.resource_quantities[y, x] = 1

    def _update_achievements(self) -> None:
        state = self._require_state()
        self._update_shelter_achievements()
        if state.step_count >= 100 and state.deaths == 0:
            state.achievements.add("all_active_agents_alive_100")

    def _update_shelter_achievements(self) -> None:
        state = self._require_state()
        if state.camp.shelter_progress >= 0.25:
            state.achievements.add("shelter_25_percent")
        if state.camp.shelter_progress >= 0.50:
            state.achievements.add("shelter_50_percent")
        if state.camp.shelter_progress >= 1.0:
            state.achievements.add("shelter_complete")

    def _at_camp(self, agent: AgentState) -> bool:
        state = self._require_state()
        return (agent.x, agent.y) == (state.camp.x, state.camp.y)

    def _spawn_positions(
        self,
        terrain: np.ndarray,
        camp_x: int,
        camp_y: int,
    ) -> list[tuple[int, int]]:
        land_positions = np.argwhere(terrain != Terrain.WATER)
        distances = np.sum((land_positions - np.array([camp_y, camp_x])) ** 2, axis=1)
        ordered = land_positions[np.argsort(distances)]
        spawns: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for y, x in ordered:
            position = (int(x), int(y))
            if position in seen:
                continue
            spawns.append(position)
            seen.add(position)
            if len(spawns) == self.num_agents:
                return spawns
        raise RuntimeError("Could not find enough land spawn positions.")

    def _parse_action(self, action: int | Action) -> Action:
        try:
            return Action(int(action))
        except ValueError:
            return Action.NOOP

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _require_state(self) -> MultiAgentWorldState:
        if self.state is None:
            raise RuntimeError("World must be reset before stepping.")
        return self.state
