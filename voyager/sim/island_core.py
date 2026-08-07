"""Deterministic simultaneous-action core for VoyagerIsland-v1."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from voyager.sim.constants import Resource, Terrain
from voyager.sim.core_v2 import (
    _event,
    _ledger,
    _sync_all_food,
    contribution_metrics,
    reconcile_ledger,
)
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENTS
from voyager.sim.island_registry import (
    ISLAND_ACTION_COUNT,
    ISLAND_MOVEMENT_DELTAS,
    ISLAND_WORK_ACTIONS,
    IslandAction,
)
from voyager.sim.scenarios import (
    ISLAND_BENCHMARK_RESCUE_DELAY,
    ISLAND_BENCHMARK_TOOL_RECIPES,
)
from voyager.sim.state import AgentState, CreatureState, FoodLot, GroundPileState
from voyager.sim.world import RESOURCE_NAMES

if TYPE_CHECKING:
    from voyager.sim.multi_world import AgentStepResult, MultiAgentWorld

TOOL_RECIPES = {
    IslandAction.CRAFT_AXE: ("axe", ISLAND_BENCHMARK_TOOL_RECIPES["axe"]),
    IslandAction.CRAFT_SPEAR: ("spear", ISLAND_BENCHMARK_TOOL_RECIPES["spear"]),
}
EDIBLE_KINDS = {"wreck_ration", "berries", "cooked_meat"}
DIRECTION_PRIORITY = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
ISLAND_TECH_STAGES = (
    "workbench",
    "tools",
    "campfire",
    "cooked_meal",
    "shelter",
    "beacon",
    "extraction",
)


def initialize_island_state(world: MultiAgentWorld) -> None:
    """Initialize the reduced economy without enabling the Stage 7B sandbox mechanics."""

    state = _state(world)
    state.camp.food_lots.clear()
    state.camp.tool_stockpile = {"axe": [], "spear": []}
    for index, agent_id in enumerate(world.possible_agents):
        agent = state.agents[agent_id]
        agent.role = "survivor"
        agent.life_state = "active"
        agent.food_lots = [
            FoodLot(
                id=f"lot-wreck-ration-{index}",
                kind="wreck_ration",
                quantity=1,
                origin_type="wreck",
                origin_id="initial_wreck",
                created_tick=0,
                expires_tick=None,
                preparation="ration",
            )
        ]
        agent.tools.clear()
        agent.equipped_tool = None
        agent.sheltered = False
        agent.inventory.update({"food": 0, "wood": 0, "stone": 0, "raw_meat": 0, "cooked_meat": 0})
    _sync_all_food(world)
    initial = {
        "food": int(np.sum(state.resource_quantities[state.resource_ids == Resource.FOOD]))
        + world.num_agents,
        "wood": int(np.sum(state.resource_quantities[state.resource_ids == Resource.WOOD])),
        "stone": int(np.sum(state.resource_quantities[state.resource_ids == Resource.STONE])),
    }
    _ledger(world, "initial_sources", category="source", balance=initial)


def action_mask(world: MultiAgentWorld, agent_id: str) -> np.ndarray:
    """Return an authoritative mask over the frozen nineteen-action registry."""

    mask = np.zeros(ISLAND_ACTION_COUNT, dtype=np.int8)
    for action in IslandAction:
        mask[int(action)] = int(_legal(world, agent_id, action))
    return mask


def step_island(
    world: MultiAgentWorld,
    raw_actions: Mapping[str, object],
) -> dict[str, AgentStepResult]:
    """Resolve one benchmark tick from an immutable set of public intents."""

    from voyager.sim.multi_world import AgentStepResult

    state = _state(world)
    previous_achievements = set(state.achievements)
    previous_deaths = state.deaths
    acting = [agent_id for agent_id in world.possible_agents if state.agents[agent_id].alive]
    health_before = {agent_id: state.agents[agent_id].health for agent_id in acting}
    state.step_count += 1
    state.events = []

    actions: dict[str, IslandAction] = {}
    invalid: set[str] = set()
    results: dict[str, AgentStepResult] = {}
    for agent_id in acting:
        raw = raw_actions.get(agent_id, int(IslandAction.NOOP))
        try:
            action = IslandAction(int(cast(Any, raw)))
        except (TypeError, ValueError):
            action = IslandAction.NOOP
            invalid.add(agent_id)
        actions[agent_id] = action
        if agent_id not in invalid and not _legal(world, agent_id, action):
            invalid.add(agent_id)
        results[agent_id] = AgentStepResult(
            reward=0.0,
            terminated=False,
            truncated=state.step_count >= world.max_steps,
            event="noop",
            reward_components={},
        )

    claims, work = _collect_camp_claims(world, actions, invalid)
    _invalidate_oversubscribed_claims(world, claims, work, invalid)
    _resolve_movements(world, actions, invalid, results)
    _resolve_interactions(world, actions, invalid, results)
    _resolve_economy(world, actions, invalid, results)
    _resolve_work(world, work, invalid, results)
    _resolve_attacks(world, actions, invalid, results)
    _advance_creatures(world, results)

    for agent_id in acting:
        agent = state.agents[agent_id]
        if not agent.alive:
            continue
        world._apply_survival_pressure(agent)
        if agent.health <= 0:
            _kill_agent(world, agent_id, reason="starvation")

    _update_achievements(world)
    _sync_all_food(world)
    new_achievements = tuple(
        name for name in ISLAND_ACHIEVEMENTS if name in state.achievements - previous_achievements
    )
    deaths_this_tick = state.deaths - previous_deaths
    for agent_id in acting:
        agent = state.agents[agent_id]
        components: dict[str, float] = {}
        if agent_id in invalid:
            components["invalid"] = -0.05
            rejected = actions[agent_id]
            _event(
                world,
                "intent_rejected",
                actors=[agent_id],
                payload={
                    "reason": "precondition_or_symmetric_conflict",
                    "action": rejected.name.lower(),
                },
            )
        delta = agent.health - health_before[agent_id]
        if delta > 0:
            components["health_restored"] = 0.1
        elif delta < 0:
            components["health_lost"] = -0.1
        if deaths_this_tick:
            components["shared_death"] = -float(deaths_this_tick)
        if new_achievements:
            components["shared_achievement"] = float(len(new_achievements))
        result = results[agent_id]
        event = "invalid_action" if agent_id in invalid else result.event
        results[agent_id] = replace(
            result,
            reward=float(sum(components.values())),
            terminated=not agent.alive or state.rescue_success,
            truncated=state.step_count >= world.max_steps,
            event=event,
            reward_components=components,
            new_achievements=new_achievements,
        )

    discrepancies = reconcile_ledger(world)
    if discrepancies:
        raise RuntimeError(f"VoyagerIsland-v1 resource ledger failed: {discrepancies}")
    return results


def _legal(world: MultiAgentWorld, agent_id: str, action: IslandAction) -> bool:
    state = _state(world)
    agent = state.agents[agent_id]
    if not agent.alive:
        return False
    if agent.sheltered:
        return action in {IslandAction.NOOP, IslandAction.REST, IslandAction.USE_SHELTER}
    if action == IslandAction.NOOP:
        return True
    if action in ISLAND_MOVEMENT_DELTAS:
        dx, dy = ISLAND_MOVEMENT_DELTAS[action]
        x, y = agent.x + dx, agent.y + dy
        return (
            agent.energy >= 1.5 and world._in_bounds(x, y) and state.terrain[y, x] != Terrain.WATER
        )
    if action == IslandAction.INTERACT:
        available = _interactable_quantity(world, agent)
        return (
            available > 0
            and agent.energy >= 2.0
            and _inventory_total(agent) < world.inventory_capacity
        )
    if action == IslandAction.ATTACK:
        target = _adjacent_target(world, agent)
        return (
            agent.energy >= 1.0
            and target is not None
            and (target.type != "island_deer" or "spear" in agent.tools)
        )
    if action == IslandAction.EAT:
        return any(lot.quantity > 0 and lot.kind in EDIBLE_KINDS for lot in agent.food_lots)
    if action == IslandAction.REST:
        return agent.energy < 100.0
    if action == IslandAction.DEPOSIT_ALL:
        return world._at_camp(agent) and _inventory_total(agent) > 0
    if action == IslandAction.WITHDRAW_FOOD:
        return (
            world._at_camp(agent)
            and any(lot.quantity > 0 and lot.kind in EDIBLE_KINDS for lot in state.camp.food_lots)
            and _inventory_total(agent) < world.inventory_capacity
        )
    if action in TOOL_RECIPES:
        tool, costs = TOOL_RECIPES[action]
        workbench = state.structures["workbench"]
        team_tools = {
            owned_tool for teammate in state.agents.values() for owned_tool in teammate.tools
        }
        return (
            workbench.complete
            and tool not in team_tools
            and world._near(agent, workbench.x, workbench.y)
            and all(
                state.camp.stockpile.get(item, 0) >= quantity for item, quantity in costs.items()
            )
        )
    if action in ISLAND_WORK_ACTIONS:
        structure_id = ISLAND_WORK_ACTIONS[action]
        structure = state.structures[structure_id]
        if not island_structure_unlocked(state, structure_id):
            return False
        material_ready = bool(structure.reserved_materials) or all(
            state.camp.stockpile.get(item, 0) >= quantity
            for item, quantity in structure.required_materials.items()
        )
        return (
            not structure.complete
            and material_ready
            and world._near(agent, structure.x, structure.y)
        )
    if action == IslandAction.USE_CAMPFIRE:
        campfire = state.structures["campfire"]
        raw_available = any(
            lot.quantity > 0 and lot.kind == "raw_meat"
            for lots in (agent.food_lots, state.camp.food_lots)
            for lot in lots
        )
        return campfire.complete and world._near(agent, campfire.x, campfire.y) and raw_available
    if action == IslandAction.USE_SHELTER:
        shelter = state.structures["shelter"]
        return (
            shelter.complete
            and world._near_shelter(agent, shelter.x, shelter.y)
            and len(shelter.occupants) < shelter.capacity
        )
    return False


def _collect_camp_claims(
    world: MultiAgentWorld,
    actions: dict[str, IslandAction],
    invalid: set[str],
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[str]]]:
    state = _state(world)
    claims: dict[str, list[tuple[str, int]]] = defaultdict(list)
    work: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid:
            continue
        if action == IslandAction.WITHDRAW_FOOD:
            claims["food"].append((agent_id, 1))
        elif action == IslandAction.USE_CAMPFIRE and not any(
            lot.quantity > 0 and lot.kind == "raw_meat" for lot in state.agents[agent_id].food_lots
        ):
            claims["raw_meat"].append((agent_id, 1))
        elif action in TOOL_RECIPES:
            _tool, costs = TOOL_RECIPES[action]
            for item, quantity in costs.items():
                claims[item].append((agent_id, quantity))
        elif action in ISLAND_WORK_ACTIONS:
            structure_id = ISLAND_WORK_ACTIONS[action]
            work[structure_id].append(agent_id)
    for structure_id in work:
        structure = state.structures[structure_id]
        if structure.reserved_materials:
            continue
        for item, quantity in structure.required_materials.items():
            claims[item].append((f"work:{structure_id}", quantity))
    return claims, work


def _invalidate_oversubscribed_claims(
    world: MultiAgentWorld,
    claims: dict[str, list[tuple[str, int]]],
    work: dict[str, list[str]],
    invalid: set[str],
) -> None:
    state = _state(world)
    for item, claimants in claims.items():
        available = (
            sum(lot.quantity for lot in state.camp.food_lots if lot.kind in EDIBLE_KINDS)
            if item == "food"
            else sum(lot.quantity for lot in state.camp.food_lots if lot.kind == "raw_meat")
            if item == "raw_meat"
            else state.camp.stockpile.get(item, 0)
        )
        if sum(quantity for _claim, quantity in claimants) <= available:
            continue
        for claimant, _quantity in claimants:
            if claimant.startswith("work:"):
                invalid.update(work[claimant.split(":", 1)[1]])
            else:
                invalid.add(claimant)


def _resolve_movements(
    world: MultiAgentWorld,
    actions: dict[str, IslandAction],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    proposed: dict[str, tuple[int, int]] = {}
    for agent_id, action in actions.items():
        if agent_id in invalid or action not in ISLAND_MOVEMENT_DELTAS:
            continue
        dx, dy = ISLAND_MOVEMENT_DELTAS[action]
        agent = state.agents[agent_id]
        proposed[agent_id] = (agent.x + dx, agent.y + dy)
    counts = Counter(proposed.values())
    for agent_id, destination in tuple(proposed.items()):
        if counts[destination] > 1:
            invalid.add(agent_id)
            proposed.pop(agent_id)
    occupied = {
        (agent.x, agent.y): agent_id
        for agent_id, agent in state.agents.items()
        if agent.alive and not agent.sheltered
    }
    memo: dict[str, bool] = {}
    visiting: set[str] = set()

    def can_move(agent_id: str) -> bool:
        if agent_id in memo:
            return memo[agent_id]
        if agent_id in visiting:
            memo[agent_id] = True
            return True
        visiting.add(agent_id)
        occupant = occupied.get(proposed[agent_id])
        success = occupant is None or (occupant in proposed and can_move(occupant))
        visiting.discard(agent_id)
        memo[agent_id] = success
        return success

    successful = {agent_id for agent_id in proposed if can_move(agent_id)}
    invalid.update(set(proposed) - successful)
    for agent_id in sorted(successful):
        agent = state.agents[agent_id]
        agent.x, agent.y = proposed[agent_id]
        agent.energy = max(0.0, agent.energy - 1.5)
        results[agent_id] = replace(results[agent_id], event="move")
        _event(world, "move", actors=[agent_id], position=proposed[agent_id])


def _resolve_interactions(
    world: MultiAgentWorld,
    actions: dict[str, IslandAction],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action != IslandAction.INTERACT:
            continue
        agent = state.agents[agent_id]
        pile = next(
            (
                value
                for value in state.ground_piles.values()
                if value.quantity > 0 and (value.x, value.y) == (agent.x, agent.y)
            ),
            None,
        )
        remaining = world.inventory_capacity - _inventory_total(agent)
        if pile is not None:
            groups[f"pile:{pile.id}"].append((agent_id, min(1, remaining)))
            continue
        resource = Resource(int(state.resource_ids[agent.y, agent.x]))
        quantity = int(state.resource_quantities[agent.y, agent.x])
        amount = 2 if resource == Resource.WOOD and "axe" in agent.tools else 1
        groups[f"node:{agent.x}:{agent.y}"].append((agent_id, min(amount, quantity, remaining)))
    for key, contenders in groups.items():
        available = (
            state.ground_piles[key.split(":", 1)[1]].quantity
            if key.startswith("pile:")
            else int(state.resource_quantities[int(key.split(":")[2]), int(key.split(":")[1])])
        )
        if (
            any(quantity <= 0 for _agent, quantity in contenders)
            or sum(quantity for _agent, quantity in contenders) > available
        ):
            invalid.update(agent for agent, _quantity in contenders)
            continue
        for agent_id, quantity in contenders:
            agent = state.agents[agent_id]
            if key.startswith("pile:"):
                pile = state.ground_piles[key.split(":", 1)[1]]
                pile.quantity -= quantity
                agent.food_lots.append(
                    FoodLot(
                        id=f"lot-{pile.id}-{agent_id}-{state.step_count}",
                        kind="raw_meat",
                        quantity=quantity,
                        origin_type=pile.origin_type,
                        origin_id=pile.origin_id,
                        created_tick=pile.created_tick,
                        expires_tick=None,
                        preparation="raw",
                    )
                )
                item = "raw_meat"
            else:
                _prefix, x_text, y_text = key.split(":")
                x, y = int(x_text), int(y_text)
                resource = Resource(int(state.resource_ids[y, x]))
                item = RESOURCE_NAMES[resource]
                state.resource_quantities[y, x] -= quantity
                if state.resource_quantities[y, x] == 0:
                    state.resource_ids[y, x] = Resource.NONE
                if item == "food":
                    agent.food_lots.append(
                        FoodLot(
                            id=f"lot-berries-{x}-{y}-{state.step_count}-{agent_id}",
                            kind="berries",
                            quantity=quantity,
                            origin_type="map",
                            origin_id=f"tile-{x}-{y}",
                            created_tick=state.step_count,
                            expires_tick=None,
                            preparation="fresh",
                        )
                    )
                else:
                    agent.inventory[item] += quantity
            agent.energy = max(0.0, agent.energy - 2.0)
            state.gathered_resources[item] = state.gathered_resources.get(item, 0) + quantity
            results[agent_id] = replace(results[agent_id], event=f"gather_{item}")
            _ledger(
                world,
                "gather",
                category="transfer",
                actors=[agent_id],
                item=item,
                quantity=quantity,
            )
            _event(world, "gather", actors=[agent_id], payload={"item": item, "quantity": quantity})
    for pile_id in [pile_id for pile_id, pile in state.ground_piles.items() if pile.quantity <= 0]:
        del state.ground_piles[pile_id]


def _resolve_economy(
    world: MultiAgentWorld,
    actions: dict[str, IslandAction],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    for agent_id in world.possible_agents:
        if agent_id not in actions or agent_id in invalid:
            continue
        action = actions[agent_id]
        agent = state.agents[agent_id]
        event = "noop"
        if action == IslandAction.EAT:
            lot = _take_edible(agent.food_lots)
            assert lot is not None
            benefit = 50.0 if lot.kind == "cooked_meat" else 35.0
            agent.hunger = max(0.0, agent.hunger - benefit)
            if lot.kind == "cooked_meat":
                agent.health = min(100.0, agent.health + 10.0)
            item = _aggregate_food(lot.kind)
            _ledger(
                world,
                "eat",
                category="sink",
                actors=[agent_id],
                item=item,
                quantity=1,
                balance={item: -1},
                lot_id=lot.id,
            )
            event = f"eat_{item}"
        elif action == IslandAction.REST:
            agent.energy = min(100.0, agent.energy + (15.0 if agent.sheltered else 10.0))
            event = "rest"
        elif action == IslandAction.DEPOSIT_ALL:
            balance: dict[str, int] = {}
            for item in ("wood", "stone"):
                quantity = agent.inventory.get(item, 0)
                if quantity:
                    agent.inventory[item] = 0
                    state.camp.stockpile[item] += quantity
                    state.deposited_resources[item] = (
                        state.deposited_resources.get(item, 0) + quantity
                    )
                    balance[item] = quantity
            for lot in list(agent.food_lots):
                agent.food_lots.remove(lot)
                state.camp.food_lots.append(lot)
                item = _aggregate_food(lot.kind)
                state.deposited_resources[item] = (
                    state.deposited_resources.get(item, 0) + lot.quantity
                )
                balance[item] = balance.get(item, 0) + lot.quantity
            state.total_deposits += sum(balance.values())
            _ledger(
                world,
                "deposit_all",
                category="transfer",
                actors=[agent_id],
                target="camp",
                quantity=sum(balance.values()),
                details={"items": balance},
            )
            event = "deposit_all"
        elif action == IslandAction.WITHDRAW_FOOD:
            lot = _take_edible(state.camp.food_lots)
            assert lot is not None
            agent.food_lots.append(lot)
            state.total_withdrawals += 1
            _ledger(
                world,
                "withdraw_food",
                category="transfer",
                actors=[agent_id],
                source="camp",
                item=_aggregate_food(lot.kind),
                quantity=1,
                lot_id=lot.id,
            )
            event = "withdraw_food"
        elif action in TOOL_RECIPES:
            tool, costs = TOOL_RECIPES[action]
            for item, quantity in costs.items():
                state.camp.stockpile[item] -= quantity
            agent.tools.add(tool)
            balance = {item: -quantity for item, quantity in costs.items()}
            balance[tool] = 1
            _ledger(
                world,
                "craft_tool",
                category="transform",
                actors=[agent_id],
                tool=tool,
                quantity=1,
                balance=balance,
            )
            event = f"craft_{tool}"
        elif action == IslandAction.USE_CAMPFIRE:
            source_lots = agent.food_lots
            source = agent_id
            raw = _take_kind(source_lots, "raw_meat")
            if raw is None:
                source_lots = state.camp.food_lots
                source = "camp"
                raw = _take_kind(source_lots, "raw_meat")
            assert raw is not None
            cooked = replace(
                raw,
                id=f"lot-cooked-{state.step_count}-{agent_id}-{len(state.ledger)}",
                kind="cooked_meat",
                created_tick=state.step_count,
                expires_tick=None,
                preparation="cooked",
            )
            source_lots.append(cooked)
            state.cooked_meals += 1
            _ledger(
                world,
                "cook",
                category="transform",
                actors=[agent_id],
                item="raw_meat",
                quantity=1,
                balance={"raw_meat": -1, "cooked_meat": 1},
                lot_id=raw.id,
                source=source,
                target=source,
            )
            event = "cook_meat"
        elif action == IslandAction.USE_SHELTER:
            shelter = state.structures["shelter"]
            if agent.sheltered:
                agent.sheltered = False
                shelter.occupants.discard(agent_id)
                if agent_id in shelter.occupancy_order:
                    shelter.occupancy_order.remove(agent_id)
                event = "exit_shelter"
            else:
                agent.sheltered = True
                shelter.occupants.add(agent_id)
                shelter.occupancy_order.append(agent_id)
                event = "enter_shelter"
        if event != "noop":
            results[agent_id] = replace(results[agent_id], event=event)
            _event(world, event, actors=[agent_id])


def _resolve_work(
    world: MultiAgentWorld,
    work: dict[str, list[str]],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    for structure_id, raw_contributors in sorted(work.items()):
        contributors = sorted(agent for agent in raw_contributors if agent not in invalid)
        if not contributors:
            continue
        structure = state.structures[structure_id]
        if not structure.reserved_materials:
            for item, quantity in structure.required_materials.items():
                state.camp.stockpile[item] -= quantity
                state.constructed_resources[item] = (
                    state.constructed_resources.get(item, 0) + quantity
                )
            structure.reserved_materials = dict(structure.required_materials)
            _ledger(
                world,
                "construction_reserve",
                category="sink",
                actors=contributors,
                target=structure_id,
                quantity=sum(structure.required_materials.values()),
                balance={
                    item: -quantity for item, quantity in structure.required_materials.items()
                },
            )
        before = structure.labor
        applied = min(structure.required_labor - structure.labor, 10 * len(contributors))
        structure.labor += applied
        state.total_build_actions += len(contributors)
        _ledger(
            world,
            "construction_labor",
            category="contribution",
            actors=contributors,
            target=structure_id,
            quantity=applied,
            details={"before": before, "after": structure.labor},
        )
        completed = before < structure.required_labor <= structure.labor
        _event(
            world,
            "structure_complete" if completed else "work",
            actors=contributors,
            targets=[structure_id],
            position=(structure.x, structure.y),
            payload={"applied_labor": applied, "progress": structure.progress},
        )
        for agent_id in contributors:
            results[agent_id] = replace(results[agent_id], event=f"work_{structure_id}")


def _resolve_attacks(
    world: MultiAgentWorld,
    actions: dict[str, IslandAction],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    attacks: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action != IslandAction.ATTACK:
            continue
        target = _adjacent_target(world, state.agents[agent_id])
        if target is None:
            invalid.add(agent_id)
            continue
        attacks[target.id].append(agent_id)
    for creature_id, attackers in sorted(attacks.items()):
        creature = state.creatures[creature_id]
        if not creature.alive:
            invalid.update(attackers)
            continue
        damage = sum(2 if "spear" in state.agents[agent].tools else 1 for agent in attackers)
        creature.health = max(0, creature.health - damage)
        for agent_id in attackers:
            state.agents[agent_id].energy = max(0.0, state.agents[agent_id].energy - 1.0)
            results[agent_id] = replace(results[agent_id], event=f"attack_{creature_id}")
        _ledger(
            world,
            "combat_damage",
            category="contribution",
            actors=sorted(attackers),
            target=creature_id,
            quantity=damage,
        )
        _event(
            world,
            "creature_attacked",
            actors=sorted(attackers),
            targets=[creature_id],
            payload={"damage": damage, "remaining_health": creature.health},
        )
        if creature.health > 0:
            continue
        creature.alive = False
        creature.behavior = "defeated"
        if creature.type == "island_deer":
            pile_id = f"pile-{state.step_count}-{creature.id}"
            state.ground_piles[pile_id] = GroundPileState(
                id=pile_id,
                x=creature.x,
                y=creature.y,
                item="raw_meat",
                quantity=2,
                origin_type="animal",
                origin_id=creature.id,
                created_tick=state.step_count,
                expires_tick=None,
            )
            state.hunts += 1
            _ledger(
                world,
                "animal_yield",
                category="source",
                actors=sorted(attackers),
                source=creature.id,
                item="raw_meat",
                quantity=2,
                balance={"raw_meat": 2},
            )
            _event(
                world,
                "ground_pile_created",
                targets=[pile_id],
                position=(creature.x, creature.y),
                payload={"item": "raw_meat", "quantity": 2},
            )
        else:
            state.monster_defeats += 1
        _event(world, "creature_defeated", actors=sorted(attackers), targets=[creature_id])


def _advance_creatures(
    world: MultiAgentWorld,
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    within_day = state.step_count % 300
    if within_day == 0:
        for creature in state.creatures.values():
            if creature.alive and creature.type == "night_stalker":
                creature.alive = False
                creature.behavior = "retreated"
                _event(world, "stalker_retreat", targets=[creature.id])
    if state.step_count % 4 == 0:
        for creature in sorted(state.creatures.values(), key=lambda value: value.id):
            if creature.alive and creature.type == "island_deer":
                world._move_deer(creature)
    if within_day == 200:
        _spawn_stalkers(world)
    if not 200 <= within_day < 300:
        return
    for creature in sorted(state.creatures.values(), key=lambda value: value.id):
        if not creature.alive or creature.type != "night_stalker":
            continue
        targets = [
            (agent_id, agent)
            for agent_id, agent in state.agents.items()
            if agent.alive and not agent.sheltered
        ]
        if not targets:
            continue
        target_id, target = min(
            targets,
            key=lambda item: (
                abs(item[1].x - creature.x) + abs(item[1].y - creature.y),
                item[1].y,
                item[1].x,
            ),
        )
        creature.target = target_id
        adjacent = abs(target.x - creature.x) + abs(target.y - creature.y) <= 1
        if adjacent and state.step_count % 2 == 0:
            target.health = max(0.0, target.health - 25.0)
            _ledger(
                world,
                "stalker_attack",
                category="damage",
                actors=[creature.id],
                target=target_id,
                quantity=25,
            )
            _event(
                world,
                "stalker_attack",
                actors=[creature.id],
                targets=[target_id],
                payload={"damage": 25},
            )
            if target.health <= 0:
                _kill_agent(world, target_id, reason="stalker_attack")
            continue
        if not adjacent and state.step_count % 2 == 0:
            creature.x, creature.y = _next_stalker_step(world, creature, target)


def _spawn_stalkers(world: MultiAgentWorld) -> None:
    state = _state(world)
    occupied = set(world.occupied_positions()) | {
        (creature.x, creature.y) for creature in state.creatures.values() if creature.alive
    }
    candidates = [position for position in world.stalker_spawns if position not in occupied]
    if not candidates:
        return
    count = min(int(world.rng.integers(1, 3)), len(candidates))
    indexes = np.atleast_1d(world.rng.choice(len(candidates), size=count, replace=False))
    positions = [candidates[int(index)] for index in indexes]
    state.last_spawn_count = count
    state.last_spawn_positions = positions
    ids: list[str] = []
    for index, (x, y) in enumerate(positions):
        creature_id = f"stalker-{state.step_count}-{index}"
        state.creatures[creature_id] = CreatureState(
            id=creature_id,
            type="night_stalker",
            x=x,
            y=y,
            health=3,
            max_health=3,
            spawn_tick=state.step_count,
            behavior="hunting",
        )
        ids.append(creature_id)
    _event(
        world,
        "stalkers_spawned",
        targets=ids,
        payload={"count": count, "positions": positions},
    )


def _next_stalker_step(
    world: MultiAgentWorld,
    creature: CreatureState,
    target: AgentState,
) -> tuple[int, int]:
    state = _state(world)
    occupied = set(world.occupied_positions())
    creatures = {
        (other.x, other.y)
        for other in state.creatures.values()
        if other.alive and other.id != creature.id
    }
    candidates: list[tuple[int, int]] = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        x, y = creature.x + dx, creature.y + dy
        if (
            world._in_bounds(x, y)
            and state.terrain[y, x] != Terrain.WATER
            and (x, y) not in occupied
            and (x, y) not in creatures
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


def _update_achievements(world: MultiAgentWorld) -> None:
    state = _state(world)
    beacon_step = state.achievement_steps.get("build_beacon")
    conditions = {
        "collect_food": state.gathered_resources.get("food", 0) > 0,
        "collect_wood": state.gathered_resources.get("wood", 0) > 0,
        "collect_stone": state.gathered_resources.get("stone", 0) > 0,
        "deposit_wood": state.deposited_resources.get("wood", 0) > 0,
        "deposit_stone": state.deposited_resources.get("stone", 0) > 0,
        "build_workbench": state.structures["workbench"].complete,
        "craft_axe": any("axe" in agent.tools for agent in state.agents.values()),
        "craft_spear": any("spear" in agent.tools for agent in state.agents.values()),
        "hunt_deer": state.hunts > 0,
        "build_campfire": state.structures["campfire"].complete,
        "cook_meat": state.cooked_meals > 0,
        "build_shelter": state.structures["shelter"].complete,
        "both_survive_first_night": state.step_count >= 300
        and all(agent.alive for agent in state.agents.values()),
        "build_beacon": state.structures["beacon"].complete,
        "rescue_both": beacon_step is not None
        and state.step_count >= beacon_step + ISLAND_BENCHMARK_RESCUE_DELAY
        and state.step_count >= 300
        and all(agent.alive for agent in state.agents.values()),
    }
    for name in ISLAND_ACHIEVEMENTS:
        if conditions[name] and name not in state.achievements:
            state.achievements.add(name)
            state.achievement_steps[name] = state.step_count
            _event(world, "achievement_unlocked", payload={"achievement": name})
    state.rescue_success = "rescue_both" in state.achievements


def island_progress_stage(state: Any) -> str:
    """Return the single public technology stage currently blocking rescue."""

    if not state.structures["workbench"].complete:
        return "workbench"
    team_tools = {tool for agent in state.agents.values() for tool in agent.tools}
    if not {"axe", "spear"} <= team_tools:
        return "tools"
    if not state.structures["campfire"].complete:
        return "campfire"
    if state.cooked_meals <= 0:
        return "cooked_meal"
    if not state.structures["shelter"].complete:
        return "shelter"
    if not state.structures["beacon"].complete:
        return "beacon"
    return "extraction"


def island_structure_unlocked(state: Any, structure_id: str) -> bool:
    """Expose only the next coherent construction branch through the legal mask."""

    stage = island_progress_stage(state)
    return stage == structure_id


def island_stage_material_requirements(
    state: Any,
    stage: str | None = None,
) -> dict[str, int]:
    """Return the finite material bundle for the active technology stage."""

    resolved_stage = stage or island_progress_stage(state)
    if resolved_stage == "tools":
        team_tools = {tool for agent in state.agents.values() for tool in agent.tools}
        requirements = {"wood": 0, "stone": 0}
        for tool, recipe in ISLAND_BENCHMARK_TOOL_RECIPES.items():
            if tool in team_tools:
                continue
            for item, quantity in recipe.items():
                requirements[item] += quantity
        return requirements
    if resolved_stage in {"workbench", "campfire", "shelter", "beacon"}:
        structure = state.structures[resolved_stage]
        if structure.reserved_materials:
            return {"wood": 0, "stone": 0}
        return {item: int(structure.required_materials.get(item, 0)) for item in ("wood", "stone")}
    return {"wood": 0, "stone": 0}


def _kill_agent(world: MultiAgentWorld, agent_id: str, *, reason: str) -> None:
    state = _state(world)
    agent = state.agents[agent_id]
    if not agent.alive:
        return
    agent.alive = False
    agent.life_state = "dead"
    agent.health = 0.0
    agent.sheltered = False
    state.deaths += 1
    shelter = state.structures["shelter"]
    shelter.occupants.discard(agent_id)
    if agent_id in shelter.occupancy_order:
        shelter.occupancy_order.remove(agent_id)
    _ledger(
        world,
        "agent_died",
        category="death",
        target=agent_id,
        quantity=1,
        details={"reason": reason},
    )
    _event(world, "agent_died", targets=[agent_id], payload={"reason": reason})


def _adjacent_target(world: MultiAgentWorld, agent: AgentState) -> CreatureState | None:
    state = _state(world)
    candidates = [
        creature
        for creature in state.creatures.values()
        if creature.alive and abs(creature.x - agent.x) + abs(creature.y - agent.y) <= 1
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda creature: (
            DIRECTION_PRIORITY.get((creature.x - agent.x, creature.y - agent.y), 4),
            creature.type,
            creature.y,
            creature.x,
        ),
    )


def _interactable_quantity(world: MultiAgentWorld, agent: AgentState) -> int:
    state = _state(world)
    piles = [
        pile
        for pile in state.ground_piles.values()
        if pile.quantity > 0 and (pile.x, pile.y) == (agent.x, agent.y)
    ]
    if piles:
        return 1
    resource = Resource(int(state.resource_ids[agent.y, agent.x]))
    quantity = int(state.resource_quantities[agent.y, agent.x])
    return min(quantity, 2 if resource == Resource.WOOD and "axe" in agent.tools else 1)


def _inventory_total(agent: AgentState) -> int:
    return (
        agent.inventory.get("wood", 0)
        + agent.inventory.get("stone", 0)
        + sum(lot.quantity for lot in agent.food_lots)
    )


def _take_edible(lots: list[FoodLot]) -> FoodLot | None:
    candidates = [lot for lot in lots if lot.quantity > 0 and lot.kind in EDIBLE_KINDS]
    if not candidates:
        return None
    selected = min(candidates, key=lambda lot: (lot.created_tick, lot.id))
    return _take_unit(lots, selected)


def _take_kind(lots: list[FoodLot], kind: str) -> FoodLot | None:
    candidates = [lot for lot in lots if lot.quantity > 0 and lot.kind == kind]
    if not candidates:
        return None
    selected = min(candidates, key=lambda lot: (lot.created_tick, lot.id))
    return _take_unit(lots, selected)


def _take_unit(lots: list[FoodLot], selected: FoodLot) -> FoodLot:
    unit = replace(selected, quantity=1)
    selected.quantity -= 1
    if selected.quantity == 0:
        lots.remove(selected)
    return unit


def _aggregate_food(kind: str) -> str:
    return "food" if kind in {"wreck_ration", "berries"} else kind


def _state(world: MultiAgentWorld):
    return world._require_state()


__all__ = [
    "action_mask",
    "contribution_metrics",
    "initialize_island_state",
    "reconcile_ledger",
    "step_island",
]
