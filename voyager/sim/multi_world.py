"""Multi-agent island world mechanics for Voyager."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from voyager.sim.achievements import ACHIEVEMENT_SET
from voyager.sim.constants import ACTION_COUNT, Action, Resource, Role, Terrain
from voyager.sim.mapgen import generate_island
from voyager.sim.registries import (
    DIRECTION_ARGUMENTS,
    ITEM_ARGUMENTS,
    STRUCTURE_ARGUMENTS,
    TARGET_ARGUMENT_START,
    TARGET_SLOT_COUNT,
    CivilizationAction,
    CivilizationArgument,
    CivilizationVerb,
)
from voyager.sim.registries_v2 import CivilizationV2Action
from voyager.sim.scenarios import (
    CIVILIZATION_CAMP,
    CIVILIZATION_CAMPFIRE,
    CIVILIZATION_DEER_SPAWNS,
    CIVILIZATION_SHELTER,
    CIVILIZATION_STALKER_SPAWNS,
    CIVILIZATION_WORKBENCH,
    COMPACT_SCENARIO_ID,
    ISLAND_BENCHMARK_CAMP,
    ISLAND_BENCHMARK_RESCUE_DELAY,
    ISLAND_BENCHMARK_STRUCTURE_SITES,
    ISLAND_BENCHMARK_STRUCTURE_SPECS,
    build_civilization_island,
    build_island_benchmark,
    scenario_definition,
)
from voyager.sim.state import (
    AgentState,
    CampState,
    CreatureState,
    FoodLot,
    MultiAgentWorldState,
    StructureState,
)
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
    reward_components: dict[str, float] = field(default_factory=dict)
    new_achievements: tuple[str, ...] = ()


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
        scenario_id: str = COMPACT_SCENARIO_ID,
        civilization_version: int = 1,
        procedural: bool = True,
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
        self.scenario_id = scenario_id
        self.scenario = scenario_definition(scenario_id)
        self.civilization = self.scenario.civilization
        self.island_benchmark = self.scenario.island_benchmark
        self.civilization_version = civilization_version
        self.procedural = procedural
        self.deer_spawns: tuple[tuple[int, int], ...] = ()
        self.stalker_spawns: tuple[tuple[int, int], ...] = ()
        self.possible_agents = [f"agent_{index}" for index in range(num_agents)]
        self.state: MultiAgentWorldState | None = None
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> MultiAgentWorldState:
        """Generate a fresh seeded shared island and spawn all agents."""

        self.rng = rng
        if self.island_benchmark:
            scenario_map = build_island_benchmark(rng, procedural=self.procedural)
            single_state = scenario_map.state
            self.deer_spawns = scenario_map.deer_spawns
            self.stalker_spawns = scenario_map.stalker_spawns
        elif self.civilization:
            single_state = build_civilization_island()
            self.deer_spawns = CIVILIZATION_DEER_SPAWNS
            self.stalker_spawns = CIVILIZATION_STALKER_SPAWNS
        else:
            single_state = generate_island(self.map_size, rng)
            self.deer_spawns = ()
            self.stalker_spawns = ()
        if self.civilization and self.map_size != single_state.terrain.shape[0]:
            raise ValueError(
                f"Civilization vertical slice requires map_size={single_state.terrain.shape[0]}."
            )
        camp_x, camp_y = (
            ISLAND_BENCHMARK_CAMP
            if self.island_benchmark
            else (
                CIVILIZATION_CAMP
                if self.civilization
                else (single_state.agent.x, single_state.agent.y)
            )
        )
        camp = CampState(
            x=camp_x,
            y=camp_y,
            shelter_capacity=(2 if self.island_benchmark else 6)
            if self.civilization
            else self.num_agents,
        )
        if self.civilization and not self.island_benchmark:
            camp.stockpile["food"] = self.num_agents
            camp.stockpile.update({"raw_meat": 0, "cooked_meat": 0})
        elif self.island_benchmark:
            camp.stockpile.update({"food": 0, "raw_meat": 0, "cooked_meat": 0})
        spawns = self._spawn_positions(single_state.terrain, camp.x, camp.y)
        agents: dict[str, AgentState] = {}
        for index, agent_id in enumerate(self.possible_agents):
            x, y = spawns[index]
            role = "survivor" if self.island_benchmark else ROLE_NAMES[Role(index % len(Role))]
            agent = AgentState(x=x, y=y, role=role)
            if self.civilization:
                agent.inventory.update({"raw_meat": 0, "cooked_meat": 0})
            agents[agent_id] = agent

        structures: dict[str, StructureState] = {}
        creatures: dict[str, CreatureState] = {}
        if self.civilization:
            structures = self._initial_civilization_structures()
            for index, (x, y) in enumerate(self.deer_spawns):
                creature_id = f"deer_{index}"
                creatures[creature_id] = CreatureState(
                    id=creature_id,
                    type="island_deer",
                    x=x,
                    y=y,
                    health=2,
                    max_health=2,
                )

        self.state = MultiAgentWorldState(
            terrain=single_state.terrain,
            resource_ids=single_state.resource_ids,
            resource_quantities=single_state.resource_quantities,
            agents=agents,
            camp=camp,
            scenario_id=self.scenario_id,
            structures=structures,
            creatures=creatures,
        )
        if self.island_benchmark:
            from voyager.sim.island_core import initialize_island_state

            initialize_island_state(self)
        elif self.civilization and self.civilization_version >= 2:
            self._initialize_v2_state()
        return self.state

    def step(
        self,
        actions: Mapping[str, int | CivilizationAction | CivilizationV2Action],
    ) -> dict[str, AgentStepResult]:
        """Apply one stable-order simultaneous step for all currently living agents."""

        if self.island_benchmark:
            from voyager.sim.island_core import step_island

            return step_island(self, actions)
        if self.civilization and self.civilization_version >= 2:
            from voyager.sim.core_v2 import step_civilization_v2

            return step_civilization_v2(self, actions)

        state = self._require_state()
        previous_deaths = state.deaths
        previous_shelter_progress = state.camp.shelter_progress
        previous_achievements = set(state.achievements)
        state.step_count += 1
        state.events = []
        occupied = {
            (agent.x, agent.y)
            for agent in state.agents.values()
            if agent.alive and not agent.sheltered
        }
        results: dict[str, AgentStepResult] = {}
        target_bindings = {
            agent_id: tuple(self.target_slots(agent_id)) for agent_id in self.alive_agents()
        }
        work_intents: dict[str, list[str]] = {}
        attack_intents: dict[str, list[str]] = {}
        defend_intents: dict[str, list[str]] = {}

        for agent_id in self.possible_agents:
            agent = state.agents[agent_id]
            if not agent.alive:
                continue

            raw_action = actions.get(agent_id, Action.NOOP)
            reward_components = {
                "alive": 0.01,
                "action": 0.0,
                "invalid": 0.0,
                "hunger_control": 0.0,
                "death": 0.0,
            }
            if self.civilization:
                reward_components.update(
                    {
                        "tool_progression": 0.0,
                        "food_preparation": 0.0,
                        "public_infrastructure": 0.0,
                        "joint_work": 0.0,
                        "defense": 0.0,
                    }
                )
            event = "noop"
            invalid = False

            if isinstance(raw_action, CivilizationAction):
                event, invalid, component_updates = self._apply_civilization_action(
                    agent_id,
                    agent,
                    raw_action,
                    occupied,
                    target_bindings.get(agent_id, ()),
                    work_intents,
                    attack_intents,
                    defend_intents,
                )
                for name, value in component_updates.items():
                    reward_components[name] = reward_components.get(name, 0.0) + value
            else:
                if isinstance(raw_action, CivilizationV2Action):
                    raw_action = Action.NOOP
                action = self._parse_action(raw_action)
                if action in {
                    Action.MOVE_UP,
                    Action.MOVE_DOWN,
                    Action.MOVE_LEFT,
                    Action.MOVE_RIGHT,
                }:
                    event, invalid = self._move(agent, action, occupied)
                elif action == Action.GATHER:
                    event, invalid, action_reward = self._gather(agent_id, agent)
                    reward_components["action"] += action_reward
                elif action == Action.EAT:
                    event, invalid, action_reward = self._eat(agent_id, agent)
                    reward_components["action"] += action_reward
                elif action == Action.REST:
                    event, invalid, action_reward = self._rest(agent)
                    reward_components["action"] += action_reward
                elif action in {
                    Action.DEPOSIT_FOOD,
                    Action.DEPOSIT_WOOD,
                    Action.DEPOSIT_STONE,
                }:
                    event, invalid, action_reward = self._deposit(agent_id, agent, action)
                    reward_components["action"] += action_reward
                elif action == Action.WITHDRAW_FOOD:
                    event, invalid, action_reward = self._withdraw_food(agent_id, agent)
                    reward_components["action"] += action_reward
                elif action == Action.BUILD_SHELTER:
                    event, invalid, action_reward = self._build_shelter(agent_id, agent)
                    reward_components["action"] += action_reward

            if invalid:
                reward_components["invalid"] = -0.05

            self._apply_survival_pressure(agent)
            reward_components["hunger_control"] = self._hunger_control_reward(agent.hunger)

            terminated = agent.health <= 0.0
            truncated = state.step_count >= self.max_steps
            if terminated:
                agent.alive = False
                state.deaths += 1
                occupied.discard((agent.x, agent.y))
                reward_components["death"] = -10.0
                event = "death"

            results[agent_id] = AgentStepResult(
                reward=float(sum(reward_components.values())),
                terminated=terminated,
                truncated=truncated,
                event=event,
                reward_components=reward_components,
            )

        if self.civilization:
            self._resolve_public_work(work_intents, results)
            self._resolve_agent_combat(attack_intents, defend_intents, results)
            self._spawn_night_stalkers()
            self._update_creatures(defend_intents, results)
            self._advance_civilization_time()
        self._apply_storm_effects(results)
        self._apply_group_rewards(
            results=results,
            previous_deaths=previous_deaths,
            previous_shelter_progress=previous_shelter_progress,
        )
        self._regenerate_food()
        self._update_achievements()
        new_achievements = tuple(sorted(state.achievements - previous_achievements))
        if new_achievements:
            for agent_id, result in tuple(results.items()):
                results[agent_id] = replace(
                    result,
                    new_achievements=new_achievements,
                )
        return results

    def _initialize_v2_state(self) -> None:
        from voyager.sim.core_v2 import initialize_v2_state

        initialize_v2_state(self)

    def v2_entity_slots(self, agent_id: str) -> list[str]:
        from voyager.sim.core_v2 import entity_slots

        return entity_slots(self, agent_id)

    def v2_action_mask(self, agent_id: str) -> np.ndarray:
        from voyager.sim.core_v2 import action_mask

        return action_mask(self, agent_id)

    def reconcile_v2_ledger(self) -> dict[str, int]:
        from voyager.sim.core_v2 import reconcile_ledger

        return reconcile_ledger(self)

    def island_action_mask(self, agent_id: str) -> np.ndarray:
        """Return the compact benchmark's authoritative legal-action mask."""

        from voyager.sim.island_core import action_mask

        return action_mask(self, agent_id)

    def alive_agents(self) -> list[str]:
        """Return live agents in stable possible-agent order."""

        state = self._require_state()
        return [agent_id for agent_id in self.possible_agents if state.agents[agent_id].alive]

    def occupied_positions(self) -> dict[tuple[int, int], str]:
        """Return occupied live-agent positions keyed by coordinate."""

        state = self._require_state()
        return {
            (agent.x, agent.y): agent_id
            for agent_id, agent in state.agents.items()
            if agent.alive and not agent.sheltered
        }

    def action_mask(self, agent_id: str) -> np.ndarray:
        """Return currently legal and useful actions for one live agent."""

        state = self._require_state()
        if agent_id not in state.agents:
            raise KeyError(f"Unknown agent: {agent_id}")

        agent = state.agents[agent_id]
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[Action.NOOP] = 1
        if not agent.alive:
            return mask

        occupied = set(self.occupied_positions())
        for action in (
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        ):
            target_x, target_y = self._movement_target(agent, action)
            if (
                agent.energy >= 1.5
                and self._in_bounds(target_x, target_y)
                and state.terrain[target_y, target_x] != Terrain.WATER
                and (target_x, target_y) not in occupied
            ):
                mask[action] = 1

        resource = Resource(int(state.resource_ids[agent.y, agent.x]))
        quantity = int(state.resource_quantities[agent.y, agent.x])
        if resource != Resource.NONE and quantity > 0 and agent.energy >= 2.0:
            resource_name = RESOURCE_NAMES[resource]
            if agent.inventory[resource_name] < self.inventory_capacity:
                mask[Action.GATHER] = 1

        if agent.inventory["food"] > 0:
            mask[Action.EAT] = 1
        if agent.energy < 100.0:
            mask[Action.REST] = 1

        if self._at_camp(agent):
            if agent.inventory["food"] > 0:
                mask[Action.DEPOSIT_FOOD] = 1
            if state.camp.shelter_progress < 1.0:
                if agent.inventory["wood"] > 0:
                    mask[Action.DEPOSIT_WOOD] = 1
                if agent.inventory["stone"] > 0:
                    mask[Action.DEPOSIT_STONE] = 1
                if (
                    agent.inventory["wood"] > 0
                    or agent.inventory["stone"] > 0
                    or state.camp.stockpile["wood"] > 0
                    or state.camp.stockpile["stone"] > 0
                ):
                    mask[Action.BUILD_SHELTER] = 1
            if (
                state.camp.stockpile["food"] > 0
                and agent.inventory["food"] < self.inventory_capacity
            ):
                mask[Action.WITHDRAW_FOOD] = 1

        return mask

    def target_slots(self, agent_id: str) -> list[str]:
        """Return deterministic visible creature target IDs for one agent."""

        state = self._require_state()
        agent = state.agents[agent_id]
        visible = [
            creature
            for creature in state.creatures.values()
            if creature.alive and abs(creature.x - agent.x) <= 3 and abs(creature.y - agent.y) <= 3
        ]
        visible.sort(
            key=lambda creature: (
                abs(creature.x - agent.x) + abs(creature.y - agent.y),
                creature.type,
                creature.id,
                creature.x,
                creature.y,
            )
        )
        return [creature.id for creature in visible[:TARGET_SLOT_COUNT]]

    def civilization_action_mask(self, agent_id: str) -> np.ndarray:
        """Return the legal Stage 7A verb/argument matrix."""

        from voyager.sim.registries import ARGUMENT_COUNT, VERB_COUNT

        state = self._require_state()
        agent = state.agents[agent_id]
        mask = np.zeros((VERB_COUNT, ARGUMENT_COUNT), dtype=np.int8)
        mask[CivilizationVerb.NOOP, CivilizationArgument.NONE] = 1
        if not agent.alive:
            return mask

        occupied = set(self.occupied_positions())
        for argument, (dx, dy) in DIRECTION_ARGUMENTS.items():
            x, y = agent.x + dx, agent.y + dy
            if (
                agent.energy >= 1.5
                and self._in_bounds(x, y)
                and state.terrain[y, x] != Terrain.WATER
                and (x, y) not in occupied
            ):
                mask[CivilizationVerb.MOVE, argument] = 1

        resource = Resource(int(state.resource_ids[agent.y, agent.x]))
        if (
            resource != Resource.NONE
            and int(state.resource_quantities[agent.y, agent.x]) > 0
            and agent.energy >= 2.0
            and self._inventory_total(agent) < self.inventory_capacity
        ):
            mask[CivilizationVerb.INTERACT, CivilizationArgument.NONE] = 1

        for argument in (
            CivilizationArgument.FOOD,
            CivilizationArgument.RAW_MEAT,
            CivilizationArgument.COOKED_MEAT,
        ):
            if agent.inventory.get(ITEM_ARGUMENTS[argument], 0) > 0:
                mask[CivilizationVerb.EAT, argument] = 1
        if agent.energy < 100.0:
            mask[CivilizationVerb.REST, CivilizationArgument.NONE] = 1

        if self._at_camp(agent):
            for argument, item in ITEM_ARGUMENTS.items():
                if agent.inventory.get(item, 0) > 0:
                    mask[CivilizationVerb.DEPOSIT, argument] = 1
                if (
                    state.camp.stockpile.get(item, 0) > 0
                    and self._inventory_total(agent) < self.inventory_capacity
                ):
                    mask[CivilizationVerb.WITHDRAW, argument] = 1

        workbench = state.structures.get("workbench")
        if (
            workbench
            and workbench.complete
            and self._near(agent, workbench.x, workbench.y)
            and "spear" not in agent.tools
            and self._has_materials(agent, {"wood": 2, "stone": 1})
        ):
            mask[CivilizationVerb.CRAFT, CivilizationArgument.SPEAR_RECIPE] = 1
        campfire = state.structures.get("campfire")
        if (
            campfire
            and campfire.complete
            and self._near(agent, campfire.x, campfire.y)
            and agent.inventory.get("raw_meat", 0) > 0
            and campfire.fuel > 0
        ):
            mask[CivilizationVerb.CRAFT, CivilizationArgument.COOK_MEAT_RECIPE] = 1

        for argument, structure_id in STRUCTURE_ARGUMENTS.items():
            structure = state.structures[structure_id]
            if (
                not structure.complete
                and self._near(agent, structure.x, structure.y)
                and (
                    structure.reserved_materials
                    or all(
                        state.camp.stockpile.get(item, 0) >= quantity
                        for item, quantity in structure.required_materials.items()
                    )
                )
            ):
                mask[CivilizationVerb.WORK, argument] = 1

        if "spear" in agent.tools and agent.equipped_tool != "spear":
            mask[CivilizationVerb.USE, CivilizationArgument.SPEAR] = 1
        if (
            campfire
            and campfire.complete
            and self._near(agent, campfire.x, campfire.y)
            and state.camp.stockpile.get("wood", 0) > 0
            and campfire.fuel < 120
        ):
            mask[CivilizationVerb.USE, CivilizationArgument.CAMPFIRE] = 1
        shelter = state.structures.get("shelter")
        if (
            shelter
            and shelter.complete
            and self._near_shelter(agent, shelter.x, shelter.y)
            and (agent_id in shelter.occupants or len(shelter.occupants) < shelter.capacity)
        ):
            mask[CivilizationVerb.USE, CivilizationArgument.SHELTER] = 1

        for slot, creature_id in enumerate(self.target_slots(agent_id)):
            creature = state.creatures[creature_id]
            if self._near(agent, creature.x, creature.y):
                target_argument = CivilizationArgument(TARGET_ARGUMENT_START + slot)
                if agent.equipped_tool == "spear" and (
                    creature.type != "island_deer"
                    or self._inventory_total(agent) <= self.inventory_capacity - 2
                ):
                    mask[CivilizationVerb.ATTACK, target_argument] = 1
                if creature.type == "night_stalker":
                    mask[CivilizationVerb.DEFEND, target_argument] = 1
        return mask

    def civilization_time(self) -> dict[str, int | float | str]:
        """Return deterministic day, phase, and light values."""

        tick = self._require_state().step_count
        within_day = tick % 300
        if within_day < 100:
            phase = "morning"
            phase_start = 0
            ambient = 1.0
        elif within_day < 200:
            phase = "afternoon"
            phase_start = 100
            ambient = 0.8
        else:
            phase = "night"
            phase_start = 200
            ambient = 0.2
        return {
            "day": tick // 300 + 1,
            "tick_in_day": within_day,
            "phase": phase,
            "phase_progress": (within_day - phase_start) / 100.0,
            "ambient_light": ambient,
        }

    def global_state(self) -> dict[str, object]:
        """Return versioned privileged state for scripts, recording, and debugging."""

        state = self._require_state()
        payload: dict[str, object] = {
            "version": "civilization_global_state_v1",
            "scenario_id": state.scenario_id,
            "tick": state.step_count,
            "time": self.civilization_time() if self.civilization else {},
            "camp": {
                "x": state.camp.x,
                "y": state.camp.y,
                "stockpile": dict(state.camp.stockpile),
            },
            "agents": {
                agent_id: {
                    "x": agent.x,
                    "y": agent.y,
                    "health": agent.health,
                    "hunger": agent.hunger,
                    "energy": agent.energy,
                    "alive": agent.alive,
                    "inventory": dict(agent.inventory),
                    "tools": sorted(agent.tools),
                    "equipped_tool": agent.equipped_tool,
                    "sheltered": agent.sheltered,
                }
                for agent_id, agent in sorted(state.agents.items())
            },
            "structures": {
                structure_id: self._structure_payload(structure)
                for structure_id, structure in sorted(state.structures.items())
            },
            "creatures": {
                creature_id: self._creature_payload(creature)
                for creature_id, creature in sorted(state.creatures.items())
            },
            "events": list(state.events),
            "rng_state": self.rng.bit_generator.state,
        }
        if self.civilization_version < 2:
            return payload
        payload["version"] = (
            "island_benchmark_global_state_v1"
            if self.island_benchmark
            else "civilization_global_state_v2"
        )
        camp = payload["camp"]
        assert isinstance(camp, dict)
        camp.update(
            {
                "food_lots": [self._food_lot_payload(lot) for lot in state.camp.food_lots],
                "tool_stockpile": {
                    tool: list(charges)
                    for tool, charges in sorted(state.camp.tool_stockpile.items())
                },
            }
        )
        agents = payload["agents"]
        assert isinstance(agents, dict)
        for agent_id, agent_payload in agents.items():
            agent = state.agents[agent_id]
            assert isinstance(agent_payload, dict)
            agent_payload.update(
                {
                    "life_state": agent.life_state,
                    "downed_ticks": agent.downed_ticks,
                    "downed_count": agent.downed_count,
                    "revival_labor": agent.revival_labor,
                    "revival_food_lot_id": agent.revival_food_lot_id,
                    "food_lots": [self._food_lot_payload(lot) for lot in agent.food_lots],
                    "tool_charges": dict(sorted(agent.tool_charges.items())),
                }
            )
        payload["ground_piles"] = {
            pile_id: {
                "id": pile.id,
                "x": pile.x,
                "y": pile.y,
                "item": pile.item,
                "quantity": pile.quantity,
                "origin_type": pile.origin_type,
                "origin_id": pile.origin_id,
                "created_tick": pile.created_tick,
                "expires_tick": pile.expires_tick,
            }
            for pile_id, pile in sorted(state.ground_piles.items())
        }
        payload["ledger"] = list(state.ledger)
        payload["spoiled_resources"] = dict(sorted(state.spoiled_resources.items()))
        payload["resources"] = [
            {
                "id": f"resource-{x}-{y}",
                "x": int(x),
                "y": int(y),
                "type": Resource(int(state.resource_ids[y, x])).name.lower(),
                "quantity": int(state.resource_quantities[y, x]),
            }
            for y, x in np.argwhere(state.resource_quantities > 0)
        ]
        payload["conservation"] = self.reconcile_v2_ledger()
        if self.island_benchmark:
            from voyager.sim.island_core import island_progress_stage

            beacon_step = state.achievement_steps.get("build_beacon")
            payload["rescue_success"] = state.rescue_success
            payload["technology_stage"] = island_progress_stage(state)
            payload["rescue_ticks_remaining"] = (
                max(
                    0,
                    ISLAND_BENCHMARK_RESCUE_DELAY - (state.step_count - beacon_step),
                    300 - state.step_count,
                )
                if beacon_step is not None
                else None
            )
        return payload

    @staticmethod
    def _food_lot_payload(lot: FoodLot) -> dict[str, object]:
        return {
            "lot_id": lot.id,
            "kind": lot.kind,
            "quantity": lot.quantity,
            "origin_type": lot.origin_type,
            "origin_id": lot.origin_id,
            "created_tick": lot.created_tick,
            "expires_tick": lot.expires_tick,
            "preparation": lot.preparation,
        }

    def _initial_civilization_structures(self) -> dict[str, StructureState]:
        if self.island_benchmark:
            return {
                structure_id: StructureState(
                    id=structure_id,
                    type=structure_id,
                    x=ISLAND_BENCHMARK_STRUCTURE_SITES[structure_id][0],
                    y=ISLAND_BENCHMARK_STRUCTURE_SITES[structure_id][1],
                    required_materials=materials,
                    required_labor=labor,
                    capacity=capacity,
                )
                for structure_id, (
                    materials,
                    labor,
                    capacity,
                ) in ISLAND_BENCHMARK_STRUCTURE_SPECS.items()
            }
        civilization_specs = {
            "workbench": (CIVILIZATION_WORKBENCH, {"wood": 6, "stone": 2}, 240, 0),
            "campfire": (CIVILIZATION_CAMPFIRE, {"wood": 4, "stone": 4}, 160, 0),
            "shelter": (CIVILIZATION_SHELTER, {"wood": 12, "stone": 6}, 600, 6),
        }
        return {
            structure_id: StructureState(
                id=structure_id,
                type=structure_id,
                x=position[0],
                y=position[1],
                required_materials=materials,
                required_labor=labor,
                capacity=capacity,
            )
            for structure_id, (position, materials, labor, capacity) in civilization_specs.items()
        }

    def _apply_civilization_action(
        self,
        agent_id: str,
        agent: AgentState,
        action: CivilizationAction,
        occupied: set[tuple[int, int]],
        target_bindings: tuple[str, ...],
        work_intents: dict[str, list[str]],
        attack_intents: dict[str, list[str]],
        defend_intents: dict[str, list[str]],
    ) -> tuple[str, bool, dict[str, float]]:
        state = self._require_state()
        verb, argument = action.verb, action.argument
        mask = self.civilization_action_mask(agent_id)
        if mask[int(verb), int(argument)] == 0:
            return "invalid_civilization_action", True, {}

        shelter = state.structures.get("shelter")
        if agent.sheltered and not (
            verb in {CivilizationVerb.NOOP, CivilizationVerb.REST}
            or (verb == CivilizationVerb.USE and argument == CivilizationArgument.SHELTER)
        ):
            agent.sheltered = False
            if shelter:
                shelter.occupants.discard(agent_id)
            self._record_event("shelter_exit", actors=[agent_id])

        if verb == CivilizationVerb.NOOP:
            return "noop", False, {}
        if verb == CivilizationVerb.MOVE:
            dx, dy = DIRECTION_ARGUMENTS[argument]
            event, invalid = self._move_delta(agent, dx, dy, occupied)
            return event, invalid, {}
        if verb == CivilizationVerb.INTERACT:
            event, invalid, reward = self._gather(agent_id, agent)
            if not invalid:
                self._record_event(event, actors=[agent_id], position=(agent.x, agent.y))
            return event, invalid, {"action": reward}
        if verb == CivilizationVerb.EAT:
            event, invalid, reward = self._eat_civilization(
                agent_id, agent, ITEM_ARGUMENTS[argument]
            )
            return event, invalid, {"action": reward}
        if verb == CivilizationVerb.REST:
            event, invalid, reward = self._rest_civilization(agent)
            return event, invalid, {"action": reward}
        if verb == CivilizationVerb.DEPOSIT:
            event, invalid, reward = self._deposit_item(agent_id, agent, ITEM_ARGUMENTS[argument])
            return event, invalid, {"action": reward}
        if verb == CivilizationVerb.WITHDRAW:
            event, invalid = self._withdraw_item(agent, ITEM_ARGUMENTS[argument])
            return event, invalid, {}
        if verb == CivilizationVerb.CRAFT:
            if argument == CivilizationArgument.SPEAR_RECIPE:
                event, invalid = self._craft_spear(agent_id, agent)
                return event, invalid, {"tool_progression": 0.5 if not invalid else 0.0}
            event, invalid = self._cook_meat(agent_id, agent)
            return event, invalid, {"food_preparation": 0.4 if not invalid else 0.0}
        if verb == CivilizationVerb.WORK:
            structure_id = STRUCTURE_ARGUMENTS[argument]
            work_intents.setdefault(structure_id, []).append(agent_id)
            return f"work_{structure_id}", False, {}
        if verb == CivilizationVerb.USE:
            if argument == CivilizationArgument.SPEAR:
                agent.equipped_tool = "spear"
                self._record_event("equip_spear", actors=[agent_id])
                return "equip_spear", False, {}
            if argument == CivilizationArgument.CAMPFIRE:
                return self._fuel_campfire(agent_id), False, {"public_infrastructure": 0.05}
            occupied.discard((agent.x, agent.y))
            return self._enter_shelter(agent_id), False, {"public_infrastructure": 0.02}
        if verb in {CivilizationVerb.ATTACK, CivilizationVerb.DEFEND}:
            slot = int(argument) - TARGET_ARGUMENT_START
            if not 0 <= slot < len(target_bindings):
                return "invalid_target_slot", True, {}
            target = target_bindings[slot]
            intents = attack_intents if verb == CivilizationVerb.ATTACK else defend_intents
            intents.setdefault(target, []).append(agent_id)
            return (
                ("attack" if verb == CivilizationVerb.ATTACK else "defend") + f"_{target}",
                False,
                {},
            )
        return "invalid_civilization_action", True, {}

    def _move_delta(
        self,
        agent: AgentState,
        dx: int,
        dy: int,
        occupied: set[tuple[int, int]],
    ) -> tuple[str, bool]:
        target_x, target_y = agent.x + dx, agent.y + dy
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
        agent.x, agent.y = target_x, target_y
        occupied.add((agent.x, agent.y))
        agent.energy = max(0.0, agent.energy - 1.5)
        return "move", False

    def _eat_civilization(
        self,
        agent_id: str,
        agent: AgentState,
        item: str,
    ) -> tuple[str, bool, float]:
        if agent.inventory.get(item, 0) <= 0:
            return f"invalid_no_{item}", True, 0.0
        if item == "food":
            return self._eat(agent_id, agent)
        agent.inventory[item] -= 1
        if item == "raw_meat":
            agent.hunger = max(0.0, agent.hunger - 15.0)
            agent.health = max(0.0, agent.health - 5.0)
            benefit = 0.05
        else:
            agent.hunger = max(0.0, agent.hunger - 40.0)
            benefit = 0.35
        self._require_state().consumed_resources.setdefault(item, 0)
        self._require_state().consumed_resources[item] += 1
        self._record_event(f"eat_{item}", actors=[agent_id])
        return f"eat_{item}", False, benefit

    def _rest_civilization(self, agent: AgentState) -> tuple[str, bool, float]:
        if agent.energy >= 100.0:
            return "invalid_full_energy", True, 0.0
        restore = 10.0
        if agent.sheltered:
            restore = 15.0
        elif self._inside_fire_radius(agent.x, agent.y):
            restore = 13.0
        agent.energy = min(100.0, agent.energy + restore)
        return "rest", False, 0.05

    def _deposit_item(
        self,
        agent_id: str,
        agent: AgentState,
        item: str,
    ) -> tuple[str, bool, float]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0
        if agent.inventory.get(item, 0) <= 0:
            return f"invalid_no_{item}", True, 0.0
        agent.inventory[item] -= 1
        state.camp.stockpile[item] = state.camp.stockpile.get(item, 0) + 1
        state.total_deposits += 1
        state.deposited_resources.setdefault(item, 0)
        state.deposited_resources[item] += 1
        state.contributing_roles.add(agent.role)
        if item == "food":
            origin = agent.food_origins.pop(0) if agent.food_origins else agent_id
            state.camp.food_origins.append(origin or agent_id)
        self._unlock("first_deposit")
        self._record_event(f"deposit_{item}", actors=[agent_id], position=(agent.x, agent.y))
        return f"deposit_{item}", False, 0.12

    def _withdraw_item(self, agent: AgentState, item: str) -> tuple[str, bool]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True
        if state.camp.stockpile.get(item, 0) <= 0:
            return f"invalid_camp_no_{item}", True
        if self._inventory_total(agent) >= self.inventory_capacity:
            return "invalid_inventory_full", True
        state.camp.stockpile[item] -= 1
        agent.inventory[item] = agent.inventory.get(item, 0) + 1
        state.total_withdrawals += 1
        if item == "food":
            origin = state.camp.food_origins.pop(0) if state.camp.food_origins else None
            agent.food_origins.append(origin)
            self._unlock("first_food_withdrawal")
        return f"withdraw_{item}", False

    def _craft_spear(self, agent_id: str, agent: AgentState) -> tuple[str, bool]:
        if not self._consume_materials(agent, {"wood": 2, "stone": 1}):
            return "invalid_spear_materials", True
        agent.tools.add("spear")
        self._unlock("first_spear_crafted")
        self._record_event("spear_crafted", actors=[agent_id], position=(agent.x, agent.y))
        return "craft_spear", False

    def _cook_meat(self, agent_id: str, agent: AgentState) -> tuple[str, bool]:
        if agent.inventory.get("raw_meat", 0) <= 0:
            return "invalid_no_raw_meat", True
        agent.inventory["raw_meat"] -= 1
        agent.inventory["cooked_meat"] = agent.inventory.get("cooked_meat", 0) + 1
        state = self._require_state()
        state.cooked_meals += 1
        self._unlock("first_cooked_meal")
        self._record_event("meat_cooked", actors=[agent_id], position=(agent.x, agent.y))
        return "cook_meat", False

    def _fuel_campfire(self, agent_id: str) -> str:
        state = self._require_state()
        campfire = state.structures["campfire"]
        state.camp.stockpile["wood"] -= 1
        campfire.fuel = min(120, campfire.fuel + 30)
        self._record_event(
            "campfire_fueled",
            actors=[agent_id],
            position=(campfire.x, campfire.y),
            payload={"fuel": campfire.fuel},
        )
        return "fuel_campfire"

    def _enter_shelter(self, agent_id: str) -> str:
        state = self._require_state()
        shelter = state.structures["shelter"]
        shelter.occupants.add(agent_id)
        state.agents[agent_id].sheltered = True
        self._record_event(
            "shelter_enter",
            actors=[agent_id],
            position=(shelter.x, shelter.y),
            payload={"occupancy": len(shelter.occupants), "capacity": shelter.capacity},
        )
        return "enter_shelter"

    def _resolve_public_work(
        self,
        work_intents: dict[str, list[str]],
        results: dict[str, AgentStepResult],
    ) -> None:
        state = self._require_state()
        for structure_id, contributors in sorted(work_intents.items()):
            structure = state.structures[structure_id]
            if structure.complete:
                continue
            if not structure.reserved_materials:
                if not all(
                    state.camp.stockpile.get(item, 0) >= quantity
                    for item, quantity in structure.required_materials.items()
                ):
                    continue
                for item, quantity in structure.required_materials.items():
                    state.camp.stockpile[item] -= quantity
                    state.constructed_resources.setdefault(item, 0)
                    state.constructed_resources[item] += quantity
                structure.reserved_materials = dict(structure.required_materials)

            raw_labor = sum(
                15 if state.agents[agent_id].role == "builder" else 10 for agent_id in contributors
            )
            multiplier_basis_points = min(15_000, 10_000 + 1_000 * (len(contributors) - 1))
            applied_labor = raw_labor * multiplier_basis_points // 10_000
            was_complete = structure.complete
            structure.labor = min(structure.required_labor, structure.labor + applied_labor)
            state.total_build_actions += len(contributors)
            state.contributing_roles.update(
                state.agents[agent_id].role for agent_id in contributors
            )
            if structure_id == "shelter":
                state.camp.shelter_progress = structure.progress
            roles = {state.agents[agent_id].role for agent_id in contributors}
            if len(contributors) >= 2 and len(roles) >= 2:
                self._unlock("joint_construction_multiple_roles")
            if structure.complete and not was_complete:
                if structure_id == "workbench":
                    self._unlock("workbench_complete")
                if structure_id == "shelter":
                    self._update_shelter_achievements()
                self._record_event(
                    "structure_complete",
                    actors=contributors,
                    position=(structure.x, structure.y),
                    targets=[structure_id],
                    payload={"structure": structure_id},
                )
            else:
                self._record_event(
                    "joint_work" if len(contributors) > 1 else "work",
                    actors=contributors,
                    position=(structure.x, structure.y),
                    targets=[structure_id],
                    payload={
                        "raw_labor": raw_labor,
                        "multiplier_basis_points": multiplier_basis_points,
                        "applied_labor": applied_labor,
                        "progress": structure.progress,
                    },
                )
            for agent_id in contributors:
                result = results[agent_id]
                components = dict(result.reward_components)
                components["public_infrastructure"] = (
                    components.get("public_infrastructure", 0.0) + 0.1
                )
                if len(contributors) > 1:
                    components["joint_work"] = components.get("joint_work", 0.0) + 0.05
                results[agent_id] = replace(
                    result,
                    reward=float(sum(components.values())),
                    event=f"work_{structure_id}",
                    reward_components=components,
                )

    def _resolve_agent_combat(
        self,
        attack_intents: dict[str, list[str]],
        defend_intents: dict[str, list[str]],
        results: dict[str, AgentStepResult],
    ) -> None:
        state = self._require_state()
        for creature_id, attackers in sorted(attack_intents.items()):
            creature = state.creatures.get(creature_id)
            if creature is None or not creature.alive:
                continue
            valid_attackers = [
                agent_id
                for agent_id in sorted(attackers)
                if state.agents[agent_id].alive
                and state.agents[agent_id].equipped_tool == "spear"
                and self._near(state.agents[agent_id], creature.x, creature.y)
            ]
            for agent_id in valid_attackers:
                if not creature.alive:
                    break
                creature.health = max(0, creature.health - 2)
                state.agents[agent_id].energy = max(0.0, state.agents[agent_id].energy - 1.0)
                self._record_event(
                    "creature_attacked",
                    actors=[agent_id],
                    targets=[creature_id],
                    position=(creature.x, creature.y),
                    payload={"damage": 2, "remaining_health": creature.health},
                )
                if creature.health == 0:
                    creature.alive = False
                    creature.behavior = "defeated"
                    if creature.type == "island_deer":
                        hunter = state.agents[agent_id]
                        hunter.inventory["raw_meat"] = hunter.inventory.get("raw_meat", 0) + 2
                        state.hunts += 1
                        state.gathered_resources.setdefault("raw_meat", 0)
                        state.gathered_resources["raw_meat"] += 2
                        self._unlock("first_successful_hunt")
                        self._record_event(
                            "successful_hunt",
                            actors=[agent_id],
                            targets=[creature_id],
                            position=(creature.x, creature.y),
                            payload={"raw_meat": 2},
                        )
                    else:
                        state.monster_defeats += 1
                        self._unlock("first_stalker_defeated")
                        if defend_intents.get(creature_id):
                            self._unlock("first_ally_defense_kill")
                        self._record_event(
                            "stalker_defeated",
                            actors=[agent_id, *defend_intents.get(creature_id, [])],
                            targets=[creature_id],
                            position=(creature.x, creature.y),
                        )
            for agent_id in valid_attackers:
                result = results[agent_id]
                components = dict(result.reward_components)
                components["defense"] = components.get("defense", 0.0) + 0.1
                results[agent_id] = replace(
                    result,
                    reward=float(sum(components.values())),
                    event=f"attack_{creature_id}",
                    reward_components=components,
                )

    def _spawn_night_stalkers(self) -> None:
        state = self._require_state()
        if state.step_count % 300 != 200:
            return
        occupied = set(self.occupied_positions())
        candidates = [
            position
            for position in CIVILIZATION_STALKER_SPAWNS
            if position not in occupied and not self._inside_fire_radius(*position)
        ]
        if not candidates:
            return
        count = min(int(self.rng.integers(1, 3)), len(candidates))
        indexes = np.atleast_1d(self.rng.choice(len(candidates), size=count, replace=False))
        positions = [candidates[int(index)] for index in indexes]
        state.last_spawn_count = count
        state.last_spawn_positions = positions
        spawned: list[str] = []
        for sequence, (x, y) in enumerate(positions):
            creature_id = f"stalker_{state.step_count}_{sequence}"
            state.creatures[creature_id] = CreatureState(
                id=creature_id,
                type="night_stalker",
                x=x,
                y=y,
                health=6,
                max_health=6,
                spawn_tick=state.step_count,
                behavior="hunting",
            )
            spawned.append(creature_id)
        self._record_event(
            "stalkers_spawned",
            targets=spawned,
            payload={"count": count, "positions": positions, "candidate_count": len(candidates)},
        )

    def _update_creatures(
        self,
        defend_intents: dict[str, list[str]],
        results: dict[str, AgentStepResult],
    ) -> None:
        state = self._require_state()
        within_day = state.step_count % 300
        if within_day == 0:
            for creature in state.creatures.values():
                if creature.alive and creature.type == "night_stalker":
                    creature.alive = False
                    creature.behavior = "retreated"
                    self._record_event(
                        "stalker_retreat",
                        targets=[creature.id],
                        position=(creature.x, creature.y),
                    )

        if state.step_count % 4 == 0:
            for creature in sorted(state.creatures.values(), key=lambda value: value.id):
                if creature.alive and creature.type == "island_deer":
                    self._move_deer(creature)

        if not 200 <= within_day < 300:
            return
        for creature in sorted(state.creatures.values(), key=lambda value: value.id):
            if not creature.alive or creature.type != "night_stalker":
                continue
            targets = [
                (agent_id, agent)
                for agent_id, agent in state.agents.items()
                if agent.alive
                and not agent.sheltered
                and not self._inside_fire_radius(agent.x, agent.y)
            ]
            if not targets:
                creature.target = None
                continue
            target_id, target = min(
                targets,
                key=lambda item: (
                    abs(item[1].x - creature.x) + abs(item[1].y - creature.y),
                    (state.step_count + int(item[0].rsplit("_", 1)[-1])) % self.num_agents,
                ),
            )
            creature.target = target_id
            if self._near(target, creature.x, creature.y):
                defenders = [
                    agent_id
                    for agent_id in defend_intents.get(creature.id, [])
                    if state.agents[agent_id].alive
                    and self._near(state.agents[agent_id], creature.x, creature.y)
                ]
                if len(defenders) >= 2:
                    state.prevented_damage += 25
                    self._record_event(
                        "joint_defense",
                        actors=defenders,
                        targets=[creature.id, target_id],
                        position=(creature.x, creature.y),
                        payload={"prevented_damage": 25, "staggered": True},
                    )
                    self._reward_defenders(defenders, results, 25)
                    continue
                prevented = 8 if defenders else 0
                damage = 25 - prevented
                state.prevented_damage += prevented
                self._damage_agent(target_id, damage, results)
                self._record_event(
                    "stalker_attack",
                    actors=[creature.id],
                    targets=[target_id],
                    position=(target.x, target.y),
                    payload={"damage": damage, "prevented_damage": prevented},
                )
                if defenders:
                    self._reward_defenders(defenders, results, prevented)
                continue
            if state.step_count % 2 == 0:
                next_position = self._next_stalker_step(creature, target)
                if next_position != (creature.x, creature.y):
                    creature.x, creature.y = next_position
                    self._record_event(
                        "stalker_pursuit",
                        targets=[creature.id, target_id],
                        position=next_position,
                    )

    def _move_deer(self, creature: CreatureState) -> None:
        state = self._require_state()
        living = [agent for agent in state.agents.values() if agent.alive]
        if not living:
            return
        nearest_distance = min(
            abs(creature.x - agent.x) + abs(creature.y - agent.y) for agent in living
        )
        if nearest_distance > 5:
            return
        occupied = set(self.occupied_positions())
        creature_positions = {
            (other.x, other.y)
            for other in state.creatures.values()
            if other.alive and other.id != creature.id
        }
        candidates = [(creature.x, creature.y)]
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            x, y = creature.x + dx, creature.y + dy
            if (
                self._in_bounds(x, y)
                and state.terrain[y, x] != Terrain.WATER
                and (x, y) not in occupied
                and (x, y) not in creature_positions
            ):
                candidates.append((x, y))
        best = max(
            candidates,
            key=lambda position: (
                min(abs(position[0] - agent.x) + abs(position[1] - agent.y) for agent in living),
                -position[1],
                -position[0],
            ),
        )
        if best != (creature.x, creature.y):
            creature.x, creature.y = best
            self._record_event("deer_flee", targets=[creature.id], position=best)

    def _next_stalker_step(
        self,
        creature: CreatureState,
        target: AgentState,
    ) -> tuple[int, int]:
        state = self._require_state()
        occupied = set(self.occupied_positions())
        creature_positions = {
            (other.x, other.y)
            for other in state.creatures.values()
            if other.alive and other.id != creature.id
        }
        candidates: list[tuple[int, int]] = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            x, y = creature.x + dx, creature.y + dy
            if (
                self._in_bounds(x, y)
                and state.terrain[y, x] != Terrain.WATER
                and (x, y) not in occupied
                and (x, y) not in creature_positions
                and not self._inside_fire_radius(x, y)
            ):
                candidates.append((x, y))
        if not candidates:
            return creature.x, creature.y
        return min(
            candidates,
            key=lambda position: (
                abs(position[0] - target.x) + abs(position[1] - target.y),
                position[1],
                position[0],
            ),
        )

    def _advance_civilization_time(self) -> None:
        state = self._require_state()
        within_day = state.step_count % 300
        campfire = state.structures["campfire"]
        shelter = state.structures["shelter"]
        if 200 <= within_day < 300:
            if campfire.fuel > 0:
                state.full_fire_night_ticks += 1
            if shelter.complete and len(shelter.occupants) == shelter.capacity:
                state.full_shelter_night_ticks += 1
        if within_day == 0:
            if state.full_fire_night_ticks >= 100:
                self._unlock("campfire_full_night")
            if state.full_shelter_night_ticks >= 100:
                self._unlock("full_shelter_protected_night")
            state.full_fire_night_ticks = 0
            state.full_shelter_night_ticks = 0
        if 200 <= within_day < 300 and campfire.fuel > 0:
            campfire.fuel -= 1

    def _damage_agent(
        self,
        agent_id: str,
        damage: int,
        results: dict[str, AgentStepResult],
    ) -> None:
        state = self._require_state()
        agent = state.agents[agent_id]
        agent.health = max(0.0, agent.health - damage)
        if agent.health > 0 or not agent.alive:
            return
        agent.alive = False
        agent.sheltered = False
        state.deaths += 1
        state.structures["shelter"].occupants.discard(agent_id)
        result = results[agent_id]
        components = dict(result.reward_components)
        components["death"] = components.get("death", 0.0) - 10.0
        results[agent_id] = replace(
            result,
            reward=float(sum(components.values())),
            terminated=True,
            event="death",
            reward_components=components,
        )

    def _reward_defenders(
        self,
        defenders: list[str],
        results: dict[str, AgentStepResult],
        prevented: int,
    ) -> None:
        for agent_id in defenders:
            result = results[agent_id]
            components = dict(result.reward_components)
            components["defense"] = components.get("defense", 0.0) + prevented / 100.0
            results[agent_id] = replace(
                result,
                reward=float(sum(components.values())),
                event="defend",
                reward_components=components,
            )

    def _has_materials(self, agent: AgentState, costs: dict[str, int]) -> bool:
        state = self._require_state()
        return all(
            agent.inventory.get(item, 0) + state.camp.stockpile.get(item, 0) >= quantity
            for item, quantity in costs.items()
        )

    def _consume_materials(self, agent: AgentState, costs: dict[str, int]) -> bool:
        if not self._has_materials(agent, costs):
            return False
        state = self._require_state()
        for item, quantity in costs.items():
            from_inventory = min(agent.inventory.get(item, 0), quantity)
            agent.inventory[item] = agent.inventory.get(item, 0) - from_inventory
            state.camp.stockpile[item] -= quantity - from_inventory
            state.constructed_resources.setdefault(item, 0)
            state.constructed_resources[item] += quantity
        return True

    def _inside_fire_radius(self, x: int, y: int) -> bool:
        state = self._require_state()
        campfire = state.structures.get("campfire")
        radius = 3
        if self.civilization_version >= 2 and campfire and campfire.condition < 50:
            radius = 2 if campfire.condition > 0 else 0
        return bool(
            campfire
            and campfire.complete
            and campfire.fuel > 0
            and abs(campfire.x - x) + abs(campfire.y - y) <= radius
        )

    def _inventory_total(self, agent: AgentState) -> int:
        return sum(agent.inventory.values())

    def _near(self, agent: AgentState, x: int, y: int) -> bool:
        return abs(agent.x - x) + abs(agent.y - y) <= 1

    def _near_shelter(self, agent: AgentState, x: int, y: int) -> bool:
        return abs(agent.x - x) + abs(agent.y - y) <= 2

    def _record_event(
        self,
        event_type: str,
        *,
        actors: list[str] | None = None,
        targets: list[str] | None = None,
        position: tuple[int, int] | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        state = self._require_state()
        state.events.append(
            {
                "tick": state.step_count,
                "type": event_type,
                "actors": list(actors or []),
                "targets": list(targets or []),
                "position": None if position is None else {"x": position[0], "y": position[1]},
                "payload": dict(payload or {}),
            }
        )

    def _structure_payload(self, structure: StructureState) -> dict[str, object]:
        return {
            "id": structure.id,
            "type": structure.type,
            "x": structure.x,
            "y": structure.y,
            "progress": structure.progress,
            "complete": structure.complete,
            "condition": structure.condition,
            "capacity": structure.capacity,
            "occupants": sorted(structure.occupants),
            "fuel": structure.fuel,
            "reserved_materials": dict(structure.reserved_materials),
            "occupancy_order": list(structure.occupancy_order),
            "repair_labor": structure.repair_labor,
            "repair_material_reserved": structure.repair_material_reserved,
        }

    def _creature_payload(self, creature: CreatureState) -> dict[str, object]:
        return {
            "id": creature.id,
            "type": creature.type,
            "x": creature.x,
            "y": creature.y,
            "health": creature.health,
            "max_health": creature.max_health,
            "alive": creature.alive,
            "target": creature.target,
            "behavior": creature.behavior,
            "spawn_tick": creature.spawn_tick,
        }

    def is_storm_active(self) -> bool:
        """Return whether a deterministic storm is active at the current step."""

        state = self._require_state()
        if state.step_count < self.storm_start_step:
            return False
        if self.storm_interval <= 0 or self.storm_duration <= 0:
            return False
        return (
            (state.step_count - self.storm_start_step) % self.storm_interval
        ) < self.storm_duration

    def metrics(self) -> dict[str, object]:
        """Return JSON-like survival economy metrics."""

        state = self._require_state()
        payload: dict[str, object] = {
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
            "achievement_steps": dict(sorted(state.achievement_steps.items())),
            "total_deposits": state.total_deposits,
            "total_withdrawals": state.total_withdrawals,
            "total_build_actions": state.total_build_actions,
            "resource_flow": {
                "gathered": dict(state.gathered_resources),
                "deposited": dict(state.deposited_resources),
                "withdrawn": {"food": state.total_withdrawals},
                "consumed": dict(state.consumed_resources),
                "constructed": dict(state.constructed_resources),
            },
            "contributing_roles": sorted(state.contributing_roles),
            "food_security_steps": state.food_security_steps,
            "max_food_security_steps": state.max_food_security_steps,
            "shelter_completion_step": state.shelter_completion_step,
            "scenario_id": state.scenario_id,
            "rescue_success": state.rescue_success,
        }
        if self.civilization_version >= 2:
            from voyager.sim.core_v2 import contribution_metrics

            payload.update(
                {
                    "life_states": {
                        value: sum(agent.life_state == value for agent in state.agents.values())
                        for value in ("active", "downed", "dead")
                    },
                    "tools": {
                        "camp": {
                            tool: len(charges)
                            for tool, charges in sorted(state.camp.tool_stockpile.items())
                        },
                        "agents": {
                            agent_id: {
                                "owned": sorted(agent.tools),
                                "equipped": agent.equipped_tool,
                                "charges": dict(sorted(agent.tool_charges.items())),
                            }
                            for agent_id, agent in sorted(state.agents.items())
                        },
                    },
                    "structures": {
                        name: self._structure_payload(structure)
                        for name, structure in sorted(state.structures.items())
                    },
                    "spoiled": dict(sorted(state.spoiled_resources.items())),
                    "contributions": contribution_metrics(self),
                    "conservation": self.reconcile_v2_ledger(),
                }
            )
        return payload

    def _move(
        self,
        agent: AgentState,
        action: Action,
        occupied: set[tuple[int, int]],
    ) -> tuple[str, bool]:
        target_x, target_y = self._movement_target(agent, action)

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

    def _gather(self, agent_id: str, agent: AgentState) -> tuple[str, bool, float]:
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
        if name == "food":
            agent.food_origins.append(None)
        agent.energy = max(0.0, agent.energy - 2.0)
        state.gathered_resources[name] += 1
        self._unlock(f"first_{name}_gathered")
        state.resource_quantities[agent.y, agent.x] = quantity - 1
        if state.resource_quantities[agent.y, agent.x] == 0:
            state.resource_ids[agent.y, agent.x] = Resource.NONE
        return f"gather_{name}", False, 0.10

    def _eat(self, agent_id: str, agent: AgentState) -> tuple[str, bool, float]:
        if agent.inventory["food"] <= 0:
            return "invalid_no_food", True, 0.0

        was_meaningfully_hungry = agent.hunger >= 35.0
        agent.inventory["food"] -= 1
        origin = agent.food_origins.pop(0) if agent.food_origins else None
        if origin is not None and origin != agent_id:
            self._unlock("shared_food_transfer")
        self._require_state().consumed_resources["food"] += 1
        agent.hunger = max(0.0, agent.hunger - 35.0)
        return "eat", False, 0.30 if was_meaningfully_hungry else 0.0

    def _rest(self, agent: AgentState) -> tuple[str, bool, float]:
        was_low_energy = agent.energy <= 50.0
        if agent.energy >= 100.0:
            return "invalid_full_energy", True, 0.0
        agent.energy = min(100.0, agent.energy + 10.0)
        return "rest", False, 0.05 if was_low_energy else 0.0

    def _deposit(
        self,
        agent_id: str,
        agent: AgentState,
        action: Action,
    ) -> tuple[str, bool, float]:
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
        state.deposited_resources[resource_name] += 1
        state.contributing_roles.add(agent.role)
        self._unlock("first_deposit")
        if resource_name == "food":
            if agent.food_origins:
                agent.food_origins.pop(0)
            state.camp.food_origins.append(agent_id)
        if resource_name == "food":
            food_target = max(1, len(self.alive_agents()) * 2)
            new_reserve = state.camp.stockpile["food"] > state.camp.food_high_watermark
            state.camp.food_high_watermark = max(
                state.camp.food_high_watermark,
                state.camp.stockpile["food"],
            )
            if not new_reserve:
                action_reward = 0.0
            elif state.camp.stockpile["food"] <= food_target:
                action_reward = 0.20
            else:
                action_reward = 0.08
        else:
            action_reward = 0.12 if state.camp.shelter_progress < 1.0 else 0.02
        return f"deposit_{resource_name}", False, action_reward

    def _withdraw_food(self, agent_id: str, agent: AgentState) -> tuple[str, bool, float]:
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0
        if state.camp.stockpile["food"] <= 0:
            return "invalid_camp_no_food", True, 0.0
        if agent.inventory["food"] >= self.inventory_capacity:
            return "invalid_food_full", True, 0.0

        state.camp.stockpile["food"] -= 1
        agent.inventory["food"] += 1
        origin = state.camp.food_origins.pop(0) if state.camp.food_origins else None
        agent.food_origins.append(origin)
        state.total_withdrawals += 1
        self._unlock("first_food_withdrawal")
        return "withdraw_food", False, 0.0

    def _build_shelter(
        self,
        agent_id: str,
        agent: AgentState,
    ) -> tuple[str, bool, float]:
        _ = agent_id
        state = self._require_state()
        if not self._at_camp(agent):
            return "invalid_not_at_camp", True, 0.0
        if state.camp.shelter_progress >= 1.0:
            return "invalid_shelter_complete", True, 0.0

        material = ""
        source = ""
        if agent.inventory["wood"] > 0:
            material = "wood"
            source = "inventory"
        elif agent.inventory["stone"] > 0:
            material = "stone"
            source = "inventory"
        elif state.camp.stockpile["wood"] > 0:
            material = "wood"
            source = "camp"
        elif state.camp.stockpile["stone"] > 0:
            material = "stone"
            source = "camp"
        else:
            return "invalid_no_build_material", True, 0.0

        if source == "inventory":
            agent.inventory[material] -= 1
        else:
            state.camp.stockpile[material] -= 1
        build_amount = 0.06 if agent.role == "builder" else 0.04
        state.camp.shelter_progress = min(1.0, state.camp.shelter_progress + build_amount)
        state.total_build_actions += 1
        state.constructed_resources[material] += 1
        state.contributing_roles.add(agent.role)
        self._update_shelter_achievements()
        event = (
            f"build_shelter_{material}"
            if source == "inventory"
            else f"build_shelter_camp_{material}"
        )
        return event, False, 0.15

    def _apply_survival_pressure(self, agent: AgentState) -> None:
        hunger_increase = 0.10 if self.civilization else 0.35
        agent.hunger = min(100.0, agent.hunger + hunger_increase)
        if agent.hunger > 80.0:
            agent.health = max(0.0, agent.health - ((agent.hunger - 80.0) * 0.05))

    def _hunger_control_reward(self, hunger: float) -> float:
        if hunger <= 60.0:
            return 0.01 * (1.0 - hunger / 60.0)
        return -0.08 * ((hunger - 60.0) / 40.0)

    def _apply_storm_effects(self, results: dict[str, AgentStepResult]) -> None:
        state = self._require_state()
        storm_active = self.is_storm_active()
        if not storm_active:
            if state.storm_was_active and self.alive_agents():
                self._unlock("first_storm_survived")
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
                        reward_components={
                            **previous.reward_components,
                            "death": previous.reward_components.get("death", 0.0) - 10.0,
                        },
                    )

    def _apply_group_rewards(
        self,
        results: dict[str, AgentStepResult],
        previous_deaths: int,
        previous_shelter_progress: float,
    ) -> None:
        state = self._require_state()
        alive_fraction = len(self.alive_agents()) / self.num_agents
        food_per_survivor = state.camp.stockpile["food"] / max(1, len(self.alive_agents()))
        shared_components = {
            "group_survival": 0.01 * alive_fraction,
            "food_security": 0.01 * min(1.0, food_per_survivor / 2.0),
            "shelter_progress": 2.0
            * max(0.0, state.camp.shelter_progress - previous_shelter_progress),
            "team_death": -0.25 * max(0, state.deaths - previous_deaths),
            "episode_survival": 1.0 * alive_fraction if state.step_count >= self.max_steps else 0.0,
        }
        shared_reward = sum(shared_components.values())
        for agent_id, result in tuple(results.items()):
            results[agent_id] = replace(
                result,
                reward=float(result.reward + shared_reward),
                reward_components={**result.reward_components, **shared_components},
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
            self._unlock("all_active_agents_alive_100")

        food = state.camp.stockpile["food"]
        if food >= 10:
            self._unlock("camp_food_buffer_10")
        if food >= 20:
            self._unlock("camp_food_buffer_20")

        alive_count = len(self.alive_agents())
        if alive_count > 0 and food >= alive_count:
            state.food_security_steps += 1
        else:
            state.food_security_steps = 0
        state.max_food_security_steps = max(
            state.max_food_security_steps,
            state.food_security_steps,
        )
        if state.food_security_steps >= 100:
            self._unlock("food_security_100_steps")

        if set(ROLE_NAMES.values()).issubset(state.contributing_roles):
            self._unlock("all_roles_contributed")
        if state.step_count >= self.max_steps and state.deaths == 0:
            self._unlock("no_deaths_run")

    def _update_shelter_achievements(self) -> None:
        state = self._require_state()
        if state.camp.shelter_progress >= 0.25:
            self._unlock("shelter_25_percent")
        if state.camp.shelter_progress >= 0.50:
            self._unlock("shelter_50_percent")
        if state.camp.shelter_progress >= 1.0:
            self._unlock("shelter_complete")
            if state.shelter_completion_step is None:
                state.shelter_completion_step = state.step_count

    def _unlock(self, achievement_id: str) -> None:
        state = self._require_state()
        if achievement_id not in ACHIEVEMENT_SET:
            raise ValueError(f"Unknown achievement: {achievement_id}")
        if achievement_id in state.achievements:
            return
        state.achievements.add(achievement_id)
        state.achievement_steps[achievement_id] = state.step_count

    def _at_camp(self, agent: AgentState) -> bool:
        state = self._require_state()
        distance = abs(agent.x - state.camp.x) + abs(agent.y - state.camp.y)
        return distance <= 1 if self.civilization else distance == 0

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

    def _movement_target(self, agent: AgentState, action: Action) -> tuple[int, int]:
        dx, dy = {
            Action.MOVE_UP: (0, -1),
            Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (-1, 0),
            Action.MOVE_RIGHT: (1, 0),
        }[action]
        return agent.x + dx, agent.y + dy

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _require_state(self) -> MultiAgentWorldState:
        if self.state is None:
            raise RuntimeError("World must be reset before stepping.")
        return self.state
