"""Stage 7B deterministic intent resolver used by the shared Voyager world."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from voyager.sim.constants import Resource, Terrain
from voyager.sim.registries_v2 import (
    V2_ACTION_TO_FLAT,
    V2_DIRECTION_ARGUMENTS,
    V2_ENTITY_SLOT_COUNT,
    V2_FLAT_ACTION_COUNT,
    V2_ITEM_ARGUMENTS,
    V2_MEANINGFUL_ACTIONS,
    V2_RECIPE_ARGUMENTS,
    V2_STRUCTURE_ARGUMENTS,
    V2_TOOL_ARGUMENTS,
    CivilizationV2Action,
    CivilizationV2Argument,
    CivilizationV2Verb,
)
from voyager.sim.state import AgentState, FoodLot, GroundPileState, StructureState
from voyager.sim.world import RESOURCE_NAMES

if TYPE_CHECKING:
    from voyager.sim.multi_world import AgentStepResult, MultiAgentWorld

FOOD_TTLS = {"wreck_ration": None, "berries": 240, "raw_meat": 90, "cooked_meat": 360}
FOOD_ARGUMENT_KINDS = {
    "food": {"wreck_ration", "berries"},
    "raw_meat": {"raw_meat"},
    "cooked_meat": {"cooked_meat"},
}
TOOL_NAMES = ("axe", "pickaxe", "spear", "torch", "pack")
REPAIR_MATERIAL = {"workbench": "wood", "campfire": "stone", "shelter": "wood"}


def initialize_v2_state(world: MultiAgentWorld) -> None:
    state = _state(world)
    state.camp.food_lots = [
        FoodLot(
            id="lot-wreck-rations-0",
            kind="wreck_ration",
            quantity=world.num_agents,
            origin_type="wreck",
            origin_id="initial_wreck",
            created_tick=0,
            expires_tick=None,
            preparation="ration",
        )
    ]
    state.camp.tool_stockpile = {tool: [] for tool in TOOL_NAMES}
    for agent in state.agents.values():
        agent.life_state = "active"
        agent.food_lots.clear()
        agent.tool_charges = {"torch": 0}
    _sync_all_food(world)
    initial = Counter(
        {
            "food": int(np.sum(state.resource_quantities[state.resource_ids == Resource.FOOD]))
            + world.num_agents,
            "wood": int(np.sum(state.resource_quantities[state.resource_ids == Resource.WOOD])),
            "stone": int(np.sum(state.resource_quantities[state.resource_ids == Resource.STONE])),
        }
    )
    _ledger(world, "initial_sources", category="source", balance=dict(initial))


def entity_slots(world: MultiAgentWorld, agent_id: str) -> list[str]:
    state = _state(world)
    agent = state.agents[agent_id]
    values: list[tuple[int, int, str]] = []
    for other_id, other in state.agents.items():
        if other_id == agent_id or other.life_state == "dead":
            continue
        distance = abs(other.x - agent.x) + abs(other.y - agent.y)
        if distance <= 3:
            values.append((distance, 0, f"agent:{other_id}"))
    for creature in state.creatures.values():
        if not creature.alive:
            continue
        distance = abs(creature.x - agent.x) + abs(creature.y - agent.y)
        if distance <= 3:
            values.append((distance, 1, f"creature:{creature.id}"))
    for structure in state.structures.values():
        distance = abs(structure.x - agent.x) + abs(structure.y - agent.y)
        if distance <= 3:
            values.append((distance, 2, f"structure:{structure.id}"))
    for pile in state.ground_piles.values():
        if pile.quantity <= 0:
            continue
        distance = abs(pile.x - agent.x) + abs(pile.y - agent.y)
        if distance <= 3:
            values.append((distance, 3, f"pile:{pile.id}"))
    return [value[2] for value in sorted(values)[:V2_ENTITY_SLOT_COUNT]]


def action_mask(world: MultiAgentWorld, agent_id: str) -> np.ndarray:
    mask = np.zeros(V2_FLAT_ACTION_COUNT, dtype=np.int8)
    slots = entity_slots(world, agent_id)
    for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
        action = CivilizationV2Action(
            CivilizationV2Verb(triple[0]),
            CivilizationV2Argument(triple[1]),
            triple[2],
        )
        if _legal(world, agent_id, action, slots):
            mask[index] = 1
    return mask


def step_civilization_v2(
    world: MultiAgentWorld,
    raw_actions: Mapping[str, object],
) -> dict[str, AgentStepResult]:
    from voyager.sim.multi_world import AgentStepResult

    state = _state(world)
    previous_achievements = set(state.achievements)
    state.step_count += 1
    state.events = []
    _expire_food(world)
    acting = [agent_id for agent_id in world.possible_agents if state.agents[agent_id].alive]
    slots = {agent_id: entity_slots(world, agent_id) for agent_id in acting}
    actions: dict[str, CivilizationV2Action] = {}
    invalid: set[str] = set()
    results: dict[str, AgentStepResult] = {}
    for agent_id in acting:
        raw = raw_actions.get(agent_id)
        action = (
            raw
            if isinstance(raw, CivilizationV2Action)
            else CivilizationV2Action(
                CivilizationV2Verb.NOOP,
                CivilizationV2Argument.NONE,
                0,
            )
        )
        actions[agent_id] = action
        if not _legal(world, agent_id, action, slots[agent_id]):
            invalid.add(agent_id)
        results[agent_id] = AgentStepResult(
            reward=0.01,
            terminated=False,
            truncated=state.step_count >= world.max_steps,
            event="noop",
            reward_components={"alive": 0.01, "invalid": 0.0},
        )
    precondition_failures = set(invalid)

    camp_claims, work_intents, repair_intents = _collect_camp_claims(
        world, actions, invalid, slots
    )
    _invalidate_oversubscribed_claims(world, camp_claims, actions, invalid, slots)
    _resolve_movements(world, actions, invalid, results)
    _resolve_interactions(world, actions, invalid, results)
    _resolve_economy(world, actions, invalid, slots, results)

    valid_work = {
        structure_id: [agent for agent in agents if agent not in invalid]
        for structure_id, agents in work_intents.items()
    }
    valid_work = {key: value for key, value in valid_work.items() if value}
    _reserve_work_materials(world, valid_work)
    labor_before = {
        structure_id: state.structures[structure_id].labor for structure_id in valid_work
    }
    world._resolve_public_work(valid_work, results)
    for structure_id, contributors in sorted(valid_work.items()):
        applied = state.structures[structure_id].labor - labor_before[structure_id]
        raw = sum(
            15 if state.agents[agent_id].role == "builder" else 10
            for agent_id in contributors
        )
        _ledger(
            world,
            "construction_labor",
            category="contribution",
            actors=contributors,
            target=structure_id,
            quantity=applied,
            details={"raw_labor": raw, "applied_labor": applied},
        )
    _resolve_repairs(world, repair_intents, invalid, results)
    newly_revived = _resolve_revivals(world, actions, invalid, slots, results)
    _resolve_combat(world, actions, invalid, slots, results)
    _apply_shelter_exits(world, actions, invalid)
    world._spawn_night_stalkers()
    newly_downed = _update_creatures(world, actions, invalid, slots, results)

    for agent_id in acting:
        agent = state.agents[agent_id]
        if agent.life_state != "active" or agent_id in newly_revived:
            continue
        world._apply_survival_pressure(agent)
        if agent.health <= 0:
            _down_or_kill(world, agent_id, results, newly_downed)
    _advance_downed(world, results, newly_downed)
    _advance_time_and_tools(world)
    world._update_achievements()
    _sync_all_food(world)

    for agent_id in sorted(invalid):
        result = results[agent_id]
        components = dict(result.reward_components)
        components["invalid"] = -0.05
        results[agent_id] = replace(
            result,
            reward=float(sum(components.values())),
            event="invalid_action",
            reward_components=components,
        )
        rejected = actions[agent_id]
        _event(
            world,
            "intent_rejected",
            actors=[agent_id],
            payload={
                "reason": (
                    "precondition_failed"
                    if agent_id in precondition_failures
                    else "symmetric_conflict"
                ),
                "verb": rejected.verb.name.lower(),
                "argument": rejected.argument.name.lower(),
                "target": rejected.target,
            },
        )

    new_achievements = tuple(sorted(state.achievements - previous_achievements))
    for agent_id, result in tuple(results.items()):
        agent = state.agents[agent_id]
        terminated = agent.life_state == "dead"
        results[agent_id] = replace(
            result,
            terminated=terminated,
            truncated=state.step_count >= world.max_steps,
            new_achievements=new_achievements,
        )
    discrepancies = reconcile_ledger(world)
    if discrepancies:
        raise RuntimeError(f"Stage 7B resource ledger failed to reconcile: {discrepancies}")
    return results


def _legal(
    world: MultiAgentWorld,
    agent_id: str,
    action: CivilizationV2Action,
    slots: list[str],
) -> bool:
    state = _state(world)
    agent = state.agents[agent_id]
    triple = (action.verb, action.argument, action.target)
    if triple not in V2_ACTION_TO_FLAT or agent.life_state == "dead":
        return False
    if agent.life_state == "downed":
        return action.verb == CivilizationV2Verb.NOOP
    target = _target(slots, action.target)
    if action.verb == CivilizationV2Verb.NOOP:
        return True
    if action.verb == CivilizationV2Verb.MOVE:
        dx, dy = V2_DIRECTION_ARGUMENTS[action.argument]
        x, y = agent.x + dx, agent.y + dy
        return (
            agent.energy >= 1.5
            and world._in_bounds(x, y)
            and state.terrain[y, x] != Terrain.WATER
        )
    if action.verb == CivilizationV2Verb.INTERACT:
        return _interactable_here(world, agent)
    if action.verb == CivilizationV2Verb.EAT:
        return _has_food(agent.food_lots, V2_ITEM_ARGUMENTS[action.argument])
    if action.verb == CivilizationV2Verb.REST:
        return agent.energy < 100
    if action.verb in {CivilizationV2Verb.DEPOSIT, CivilizationV2Verb.WITHDRAW}:
        if not world._at_camp(agent):
            return False
        if action.argument in V2_ITEM_ARGUMENTS:
            item = V2_ITEM_ARGUMENTS[action.argument]
            if action.verb == CivilizationV2Verb.DEPOSIT:
                return _agent_item_count(agent, item) > 0
            return _camp_item_count(state, item) > 0 and _inventory_total(world, agent) < _capacity(world, agent)
        tool = V2_TOOL_ARGUMENTS[action.argument]
        if action.verb == CivilizationV2Verb.DEPOSIT:
            return tool in agent.tools
        return tool not in agent.tools and bool(state.camp.tool_stockpile.get(tool))
    if action.verb == CivilizationV2Verb.CRAFT:
        if action.argument == CivilizationV2Argument.COOK_MEAT_RECIPE:
            campfire = state.structures["campfire"]
            return campfire.condition > 0 and campfire.fuel > 0 and _has_food(agent.food_lots, "raw_meat") and world._near(agent, campfire.x, campfire.y)
        workbench = state.structures["workbench"]
        tool, costs = V2_RECIPE_ARGUMENTS[action.argument]
        return workbench.complete and workbench.condition > 0 and tool not in agent.tools and world._near(agent, workbench.x, workbench.y) and world._has_materials(agent, costs)
    if action.verb == CivilizationV2Verb.WORK:
        structure = state.structures[V2_STRUCTURE_ARGUMENTS[action.argument]]
        materials_ready = bool(structure.reserved_materials) or all(
            state.camp.stockpile.get(item, 0) >= quantity
            for item, quantity in structure.required_materials.items()
        )
        return (
            not structure.complete
            and materials_ready
            and world._near(agent, structure.x, structure.y)
        )
    if action.verb == CivilizationV2Verb.USE:
        if action.argument == CivilizationV2Argument.SHELTER:
            shelter = state.structures["shelter"]
            return (
                shelter.complete
                and shelter.condition > 0
                and world._near_shelter(agent, shelter.x, shelter.y)
                and (agent.sheltered or len(shelter.occupants) < shelter.capacity)
            )
        campfire = state.structures["campfire"]
        return campfire.complete and campfire.condition > 0 and world._near(agent, campfire.x, campfire.y) and ((agent.equipped_tool == "torch" and campfire.fuel >= 15) or state.camp.stockpile.get("wood", 0) > 0)
    if action.verb == CivilizationV2Verb.EQUIP:
        tool = V2_TOOL_ARGUMENTS[action.argument]
        return tool in agent.tools and tool != "pack" and agent.equipped_tool != tool
    if target is None:
        return False
    kind, entity_id = target.split(":", 1)
    if action.verb == CivilizationV2Verb.GIVE:
        if kind != "agent" or not _adjacent_agent(state, agent, entity_id):
            return False
        if action.argument in V2_ITEM_ARGUMENTS:
            return _agent_item_count(agent, V2_ITEM_ARGUMENTS[action.argument]) > 0
        tool = V2_TOOL_ARGUMENTS[action.argument]
        return tool in agent.tools and tool not in state.agents[entity_id].tools
    if action.verb in {CivilizationV2Verb.ATTACK, CivilizationV2Verb.DEFEND}:
        if kind != "creature":
            return False
        creature = state.creatures[entity_id]
        near = abs(agent.x - creature.x) + abs(agent.y - creature.y) <= 1
        return near and (action.verb == CivilizationV2Verb.DEFEND or agent.equipped_tool == "spear")
    if action.verb == CivilizationV2Verb.REPAIR:
        if kind != "structure":
            return False
        structure = state.structures[entity_id]
        material_ready = (
            structure.repair_material_reserved
            or state.camp.stockpile.get(REPAIR_MATERIAL[entity_id], 0) > 0
        )
        return (
            structure.complete
            and structure.condition < 100
            and material_ready
            and world._near(agent, structure.x, structure.y)
        )
    if action.verb == CivilizationV2Verb.REVIVE:
        return kind == "agent" and state.agents[entity_id].life_state == "downed" and _adjacent_agent(state, agent, entity_id)
    return False


def _collect_camp_claims(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[str]], dict[str, list[str]]]:
    state = _state(world)
    claims: dict[str, list[tuple[str, int]]] = defaultdict(list)
    work: dict[str, list[str]] = defaultdict(list)
    repair: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid:
            continue
        agent = state.agents[agent_id]
        if action.verb == CivilizationV2Verb.WITHDRAW and action.argument in V2_ITEM_ARGUMENTS:
            claims[V2_ITEM_ARGUMENTS[action.argument]].append((agent_id, 1))
        elif action.verb == CivilizationV2Verb.WITHDRAW:
            claims[f"tool:{V2_TOOL_ARGUMENTS[action.argument]}"].append((agent_id, 1))
        elif action.verb == CivilizationV2Verb.CRAFT and action.argument in V2_RECIPE_ARGUMENTS:
            _tool, costs = V2_RECIPE_ARGUMENTS[action.argument]
            for item, quantity in costs.items():
                need = max(0, quantity - agent.inventory.get(item, 0))
                if need:
                    claims[item].append((agent_id, need))
        elif action.verb == CivilizationV2Verb.USE and action.argument == CivilizationV2Argument.CAMPFIRE and not (agent.equipped_tool == "torch" and state.structures["campfire"].fuel >= 15):
            claims["wood"].append((agent_id, 1))
        elif action.verb == CivilizationV2Verb.WORK:
            structure_id = V2_STRUCTURE_ARGUMENTS[action.argument]
            work[structure_id].append(agent_id)
        elif action.verb == CivilizationV2Verb.REPAIR:
            target = _target(slots[agent_id], action.target)
            if target:
                structure_id = target.split(":", 1)[1]
                repair[structure_id].append(agent_id)
    for structure_id in work:
        structure = state.structures[structure_id]
        if not structure.reserved_materials:
            claim_id = f"work:{structure_id}"
            for item, quantity in structure.required_materials.items():
                claims[item].append((claim_id, quantity))
    for structure_id in repair:
        structure = state.structures[structure_id]
        if not structure.repair_material_reserved:
            claims[REPAIR_MATERIAL[structure_id]].append((f"repair:{structure_id}", 1))
    return claims, work, repair


def _invalidate_oversubscribed_claims(
    world: MultiAgentWorld,
    claims: dict[str, list[tuple[str, int]]],
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
) -> None:
    state = _state(world)
    for item, values in claims.items():
        available = (
            len(state.camp.tool_stockpile.get(item.split(":", 1)[1], []))
            if item.startswith("tool:")
            else state.camp.stockpile.get(item, 0)
        )
        if sum(quantity for _claim, quantity in values) <= available:
            continue
        for claim, _quantity in values:
            if claim.startswith(("work:", "repair:")):
                continue
            invalid.add(claim)
        sentinels = {claim for claim, _quantity in values if claim.startswith(("work:", "repair:"))}
        for sentinel in sentinels:
            kind, entity_id = sentinel.split(":", 1)
            matching_verb = (
                CivilizationV2Verb.WORK if kind == "work" else CivilizationV2Verb.REPAIR
            )
            for agent_id, action in actions.items():
                if action.verb != matching_verb:
                    continue
                target = _target(slots[agent_id], action.target)
                matches_repair = (
                    kind == "repair"
                    and target is not None
                    and target == f"structure:{entity_id}"
                )
                if kind == "work" and V2_STRUCTURE_ARGUMENTS[action.argument] == entity_id or matches_repair:
                    invalid.add(agent_id)


def _resolve_movements(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    proposed: dict[str, tuple[int, int]] = {}
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.MOVE:
            continue
        agent = state.agents[agent_id]
        dx, dy = V2_DIRECTION_ARGUMENTS[action.argument]
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
    for agent_id in proposed:
        if agent_id not in successful:
            invalid.add(agent_id)
    for agent_id in sorted(successful):
        agent = state.agents[agent_id]
        agent.x, agent.y = proposed[agent_id]
        agent.energy = max(0.0, agent.energy - 1.5)
        results[agent_id] = replace(results[agent_id], event="move")
        _event(world, "move", actors=[agent_id], position=proposed[agent_id])


def _resolve_interactions(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.INTERACT:
            continue
        agent = state.agents[agent_id]
        pile = next((value for value in state.ground_piles.values() if value.quantity > 0 and (value.x, value.y) == (agent.x, agent.y)), None)
        if pile:
            groups[f"pile:{pile.id}"].append((agent_id, 1))
            continue
        resource = Resource(int(state.resource_ids[agent.y, agent.x]))
        if resource == Resource.NONE:
            invalid.add(agent_id)
            continue
        yield_quantity = 1
        if (
            resource == Resource.WOOD
            and agent.equipped_tool == "axe"
            or resource == Resource.STONE
            and agent.equipped_tool == "pickaxe"
        ):
            yield_quantity = min(2, int(state.resource_quantities[agent.y, agent.x]))
        groups[f"node:{agent.x}:{agent.y}"].append((agent_id, yield_quantity))
    for key, contenders in groups.items():
        available = (
            state.ground_piles[key.split(":", 1)[1]].quantity
            if key.startswith("pile:")
            else int(state.resource_quantities[int(key.split(":")[2]), int(key.split(":")[1])])
        )
        if sum(quantity for _agent, quantity in contenders) > available:
            invalid.update(agent for agent, _quantity in contenders)
            continue
        for agent_id, quantity in contenders:
            agent = state.agents[agent_id]
            if _inventory_total(world, agent) + quantity > _capacity(world, agent):
                invalid.add(agent_id)
                continue
            if key.startswith("pile:"):
                pile = state.ground_piles[key.split(":", 1)[1]]
                pile.quantity -= quantity
                agent.food_lots.append(
                    FoodLot(
                        id=f"lot-{pile.id}-{agent_id}-{state.step_count}-{len(state.ledger)}",
                        kind=pile.item,
                        quantity=quantity,
                        origin_type=pile.origin_type,
                        origin_id=pile.origin_id,
                        created_tick=pile.created_tick,
                        expires_tick=pile.expires_tick,
                        preparation=pile.item,
                    )
                )
                item = pile.item
            else:
                _prefix, x_text, y_text = key.split(":")
                x, y = int(x_text), int(y_text)
                resource = Resource(int(state.resource_ids[y, x]))
                item = RESOURCE_NAMES[resource]
                state.resource_quantities[y, x] -= quantity
                if state.resource_quantities[y, x] == 0:
                    state.resource_ids[y, x] = Resource.NONE
                if item == "food":
                    _add_food_lot(world, agent.food_lots, "berries", quantity, "map", f"tile-{x}-{y}")
                else:
                    agent.inventory[item] = agent.inventory.get(item, 0) + quantity
            agent.energy = max(0.0, agent.energy - 2.0)
            results[agent_id] = replace(results[agent_id], event=f"gather_{item}")
            _ledger(world, "gather", category="transfer", actors=[agent_id], item=item, quantity=quantity)
            _event(world, "gather", actors=[agent_id], payload={"item": item, "quantity": quantity})
    for pile_id in [key for key, pile in state.ground_piles.items() if pile.quantity <= 0]:
        del state.ground_piles[pile_id]


def _resolve_economy(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    incoming: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.GIVE or action.argument not in V2_ITEM_ARGUMENTS:
            continue
        target = _target(slots[agent_id], action.target)
        if target:
            incoming[target.split(":", 1)[1]].append(agent_id)
    for receiver_id, givers in incoming.items():
        receiver = state.agents[receiver_id]
        if _inventory_total(world, receiver) + len(givers) > _capacity(world, receiver):
            invalid.update(givers)
    incoming_tools: dict[tuple[str, str], list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.GIVE:
            continue
        if action.argument not in V2_TOOL_ARGUMENTS:
            continue
        target = _target(slots[agent_id], action.target)
        if target:
            incoming_tools[(target.split(":", 1)[1], V2_TOOL_ARGUMENTS[action.argument])].append(agent_id)
    for givers in incoming_tools.values():
        if len(givers) > 1:
            invalid.update(givers)

    for agent_id in world.possible_agents:
        if agent_id not in actions or agent_id in invalid:
            continue
        action = actions[agent_id]
        agent = state.agents[agent_id]
        event = "noop"
        if action.verb == CivilizationV2Verb.EAT:
            item = V2_ITEM_ARGUMENTS[action.argument]
            lot = _take_food(world, agent.food_lots, item)
            if lot is None:
                invalid.add(agent_id)
                continue
            benefit = 15 if item == "raw_meat" else 40 if item == "cooked_meat" else 35
            agent.hunger = max(0.0, agent.hunger - benefit)
            if item == "raw_meat":
                agent.health = max(0.0, agent.health - 5)
            _ledger(world, "eat", category="sink", actors=[agent_id], item=item, quantity=1, balance={item: -1}, lot_id=lot.id)
            event = f"eat_{item}"
        elif action.verb == CivilizationV2Verb.REST:
            agent.energy = min(100.0, agent.energy + (15 if agent.sheltered else 10))
            event = "rest"
        elif action.verb == CivilizationV2Verb.DEPOSIT:
            event = _deposit(world, agent_id, action.argument)
        elif action.verb == CivilizationV2Verb.WITHDRAW:
            event = _withdraw(world, agent_id, action.argument)
        elif action.verb == CivilizationV2Verb.CRAFT:
            event = _craft(world, agent_id, action.argument)
        elif action.verb == CivilizationV2Verb.USE:
            event = _use(world, agent_id, action.argument)
        elif action.verb == CivilizationV2Verb.EQUIP:
            tool = V2_TOOL_ARGUMENTS[action.argument]
            agent.equipped_tool = tool
            event = f"equip_{tool}"
            _ledger(world, "equip", category="ownership", actors=[agent_id], tool=tool)
        elif action.verb == CivilizationV2Verb.GIVE:
            target = _target(slots[agent_id], action.target)
            assert target is not None
            event = _give(world, agent_id, target.split(":", 1)[1], action.argument)
        if event != "noop":
            results[agent_id] = replace(results[agent_id], event=event)
            target_ref = _target(slots[agent_id], action.target)
            _event(
                world,
                event,
                actors=[agent_id],
                targets=[] if target_ref is None else [target_ref.split(":", 1)[1]],
            )


def _deposit(world: MultiAgentWorld, agent_id: str, argument: CivilizationV2Argument) -> str:
    state = _state(world)
    agent = state.agents[agent_id]
    if argument in V2_ITEM_ARGUMENTS:
        item = V2_ITEM_ARGUMENTS[argument]
        if item in FOOD_ARGUMENT_KINDS:
            lot = _take_food(world, agent.food_lots, item)
            assert lot is not None
            state.camp.food_lots.append(lot)
        else:
            agent.inventory[item] -= 1
            state.camp.stockpile[item] += 1
        _ledger(world, "deposit", category="transfer", actors=[agent_id], target="camp", item=item, quantity=1)
        return f"deposit_{item}"
    tool = V2_TOOL_ARGUMENTS[argument]
    charge = agent.tool_charges.get(tool, 0)
    agent.tools.remove(tool)
    if agent.equipped_tool == tool:
        agent.equipped_tool = None
    state.camp.tool_stockpile.setdefault(tool, []).append(charge)
    _ledger(world, "deposit_tool", category="transfer", actors=[agent_id], target="camp", tool=tool, quantity=1)
    return f"deposit_{tool}"


def _withdraw(world: MultiAgentWorld, agent_id: str, argument: CivilizationV2Argument) -> str:
    state = _state(world)
    agent = state.agents[agent_id]
    if argument in V2_ITEM_ARGUMENTS:
        item = V2_ITEM_ARGUMENTS[argument]
        if item in FOOD_ARGUMENT_KINDS:
            lot = _take_food(world, state.camp.food_lots, item)
            assert lot is not None
            agent.food_lots.append(lot)
        else:
            state.camp.stockpile[item] -= 1
            agent.inventory[item] += 1
        _ledger(world, "withdraw", category="transfer", actors=[agent_id], source="camp", item=item, quantity=1)
        return f"withdraw_{item}"
    tool = V2_TOOL_ARGUMENTS[argument]
    charge = state.camp.tool_stockpile[tool].pop(0)
    agent.tools.add(tool)
    agent.tool_charges[tool] = charge
    _ledger(world, "withdraw_tool", category="transfer", actors=[agent_id], source="camp", tool=tool, quantity=1)
    return f"withdraw_{tool}"


def _craft(world: MultiAgentWorld, agent_id: str, argument: CivilizationV2Argument) -> str:
    state = _state(world)
    agent = state.agents[agent_id]
    if argument == CivilizationV2Argument.COOK_MEAT_RECIPE:
        source = _take_food(world, agent.food_lots, "raw_meat")
        assert source is not None
        _add_food_lot(world, agent.food_lots, "cooked_meat", 1, source.origin_type, source.origin_id, preparation="cooked")
        state.cooked_meals += 1
        _ledger(world, "cook", category="transform", actors=[agent_id], item="raw_meat", quantity=1, balance={"raw_meat": -1, "cooked_meat": 1}, lot_id=source.id)
        return "cook_meat"
    tool, costs = V2_RECIPE_ARGUMENTS[argument]
    balance: dict[str, int] = {item: -quantity for item, quantity in costs.items()}
    balance[tool] = 1
    for item, quantity in costs.items():
        own = min(agent.inventory.get(item, 0), quantity)
        agent.inventory[item] -= own
        state.camp.stockpile[item] -= quantity - own
    agent.tools.add(tool)
    agent.tool_charges[tool] = 0
    _ledger(world, "craft_tool", category="transform", actors=[agent_id], tool=tool, quantity=1, balance=balance)
    return f"craft_{tool}"


def _use(world: MultiAgentWorld, agent_id: str, argument: CivilizationV2Argument) -> str:
    state = _state(world)
    agent = state.agents[agent_id]
    if argument == CivilizationV2Argument.SHELTER:
        shelter = state.structures["shelter"]
        if agent.sheltered:
            return "remain_sheltered"
        agent.sheltered = True
        shelter.occupants.add(agent_id)
        shelter.occupancy_order.append(agent_id)
        _event(world, "shelter_enter", actors=[agent_id])
        return "enter_shelter"
    campfire = state.structures["campfire"]
    if agent.equipped_tool == "torch" and campfire.fuel >= 15:
        campfire.fuel -= 15
        agent.tool_charges["torch"] = agent.tool_charges.get("torch", 0) + 30
        _ledger(world, "charge_torch", category="energy_transfer", actors=[agent_id], tool="torch", quantity=30)
        return "charge_torch"
    state.camp.stockpile["wood"] -= 1
    campfire.fuel = min(120, campfire.fuel + 30)
    _ledger(world, "fuel_campfire", category="sink", actors=[agent_id], item="wood", quantity=1, balance={"wood": -1})
    return "fuel_campfire"


def _give(world: MultiAgentWorld, giver_id: str, receiver_id: str, argument: CivilizationV2Argument) -> str:
    state = _state(world)
    giver, receiver = state.agents[giver_id], state.agents[receiver_id]
    if argument in V2_ITEM_ARGUMENTS:
        item = V2_ITEM_ARGUMENTS[argument]
        if item in FOOD_ARGUMENT_KINDS:
            lot = _take_food(world, giver.food_lots, item)
            assert lot is not None
            receiver.food_lots.append(lot)
        else:
            giver.inventory[item] -= 1
            receiver.inventory[item] += 1
        _ledger(world, "give", category="transfer", actors=[giver_id], target=receiver_id, item=item, quantity=1)
        return f"give_{item}"
    tool = V2_TOOL_ARGUMENTS[argument]
    charge = giver.tool_charges.get(tool, 0)
    giver.tools.remove(tool)
    if giver.equipped_tool == tool:
        giver.equipped_tool = None
    receiver.tools.add(tool)
    receiver.tool_charges[tool] = charge
    _ledger(world, "give_tool", category="transfer", actors=[giver_id], target=receiver_id, tool=tool, quantity=1)
    return f"give_{tool}"


def _reserve_work_materials(world: MultiAgentWorld, work: dict[str, list[str]]) -> None:
    state = _state(world)
    for structure_id, contributors in list(work.items()):
        structure = state.structures[structure_id]
        if structure.reserved_materials:
            continue
        if not all(state.camp.stockpile.get(item, 0) >= quantity for item, quantity in structure.required_materials.items()):
            del work[structure_id]
            continue
        _ledger(
            world,
            "construction_reserve",
            category="sink",
            actors=contributors,
            target=structure_id,
            balance={item: -quantity for item, quantity in structure.required_materials.items()},
        )


def _resolve_repairs(
    world: MultiAgentWorld,
    repair_intents: dict[str, list[str]],
    invalid: set[str],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    for structure_id, contributors in sorted(repair_intents.items()):
        contributors = [agent for agent in contributors if agent not in invalid]
        if not contributors:
            continue
        structure = state.structures[structure_id]
        material = REPAIR_MATERIAL[structure_id]
        if not structure.repair_material_reserved:
            if state.camp.stockpile.get(material, 0) <= 0:
                invalid.update(contributors)
                continue
            state.camp.stockpile[material] -= 1
            structure.repair_material_reserved = True
            _ledger(world, "repair_reserve", category="sink", actors=contributors, target=structure_id, item=material, quantity=1, balance={material: -1})
        labor = sum(15 if state.agents[agent].role == "builder" else 10 for agent in contributors)
        structure.repair_labor += labor
        _ledger(world, "repair_labor", category="contribution", actors=contributors, target=structure_id, quantity=labor)
        if structure.repair_labor >= 40:
            before = structure.condition
            structure.condition = min(100, structure.condition + 20)
            structure.repair_labor -= 40
            structure.repair_material_reserved = False
            _apply_structure_threshold(world, structure)
            _ledger(
                world,
                "repair_completed",
                category="repair",
                actors=contributors,
                target=structure_id,
                quantity=structure.condition - before,
                details={"condition_before": before, "condition_after": structure.condition},
            )
            _event(world, "structure_repaired", actors=contributors, targets=[structure_id], payload={"before": before, "after": structure.condition})
        for agent_id in contributors:
            results[agent_id] = replace(results[agent_id], event=f"repair_{structure_id}")


def _resolve_revivals(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
    results: dict[str, AgentStepResult],
) -> set[str]:
    state = _state(world)
    revived: set[str] = set()
    groups: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.REVIVE:
            continue
        target = _target(slots[agent_id], action.target)
        if target:
            groups[target.split(":", 1)[1]].append(agent_id)
    for target_id, contributors in sorted(groups.items()):
        target_agent = state.agents[target_id]
        if target_agent.revival_food_lot_id is None:
            candidates = [
                (lot.expires_tick if lot.expires_tick is not None else 10**12, _tie_key(state.step_count, target_id, actor), actor, lot)
                for actor in contributors
                for lot in state.agents[actor].food_lots
                if lot.quantity > 0
            ]
            if not candidates:
                invalid.update(contributors)
                continue
            _expiry, _tie, donor, selected = min(candidates, key=lambda value: value[:2])
            reserved = _take_specific_lot(state.agents[donor].food_lots, selected.id)
            target_agent.revival_food_lot_id = reserved.id
            _ledger(world, "revival_food_reserved", category="sink", actors=[donor], target=target_id, item=_aggregate_food_kind(reserved.kind), quantity=1, balance={_aggregate_food_kind(reserved.kind): -1}, lot_id=reserved.id)
        target_agent.revival_labor += 10 * len(contributors)
        _ledger(world, "revival_labor", category="contribution", actors=contributors, target=target_id, quantity=10 * len(contributors))
        for agent_id in contributors:
            results[agent_id] = replace(results[agent_id], event=f"revive_{target_id}")
        if target_agent.revival_labor >= 30:
            target_agent.life_state = "active"
            target_agent.health = 30
            target_agent.hunger = 50
            target_agent.energy = 20
            target_agent.downed_ticks = 0
            target_agent.revival_labor = 0
            target_agent.revival_food_lot_id = None
            revived.add(target_id)
            _ledger(
                world,
                "revival_completed",
                category="revival",
                actors=contributors,
                target=target_id,
                quantity=30,
            )
            _event(world, "agent_revived", actors=contributors, targets=[target_id])
    return revived


def _resolve_combat(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
    results: dict[str, AgentStepResult],
) -> None:
    state = _state(world)
    attacks: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.ATTACK:
            continue
        target = _target(slots[agent_id], action.target)
        if target:
            attacks[target.split(":", 1)[1]].append(agent_id)
    for creature_id, attackers in sorted(attacks.items()):
        creature = state.creatures[creature_id]
        if not creature.alive:
            continue
        damage = 2 * len(attackers)
        creature.health = max(0, creature.health - damage)
        _event(
            world,
            "creature_attacked",
            actors=attackers,
            targets=[creature_id],
            payload={"damage": damage, "remaining_health": creature.health},
        )
        _ledger(world, "combat_damage", category="contribution", actors=attackers, target=creature_id, quantity=damage)
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
                expires_tick=state.step_count + FOOD_TTLS["raw_meat"],
            )
            state.hunts += 1
            _event(
                world,
                "ground_pile_created",
                actors=attackers,
                targets=[pile_id],
                position=(creature.x, creature.y),
                payload={"item": "raw_meat", "quantity": 2},
            )
            _ledger(world, "animal_yield", category="source", actors=attackers, source=creature.id, item="raw_meat", quantity=2, balance={"raw_meat": 2})
        else:
            state.monster_defeats += 1
        _event(world, "creature_defeated", actors=attackers, targets=[creature_id])


def _update_creatures(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
    slots: dict[str, list[str]],
    results: dict[str, AgentStepResult],
) -> set[str]:
    state = _state(world)
    newly_downed: set[str] = set()
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
    if not 200 <= within_day < 300:
        return newly_downed
    defense: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in actions.items():
        if agent_id in invalid or action.verb != CivilizationV2Verb.DEFEND:
            continue
        target = _target(slots[agent_id], action.target)
        if target:
            defense[target.split(":", 1)[1]].append(agent_id)
    for creature in sorted(state.creatures.values(), key=lambda value: value.id):
        if not creature.alive or creature.type != "night_stalker":
            continue
        active_targets = [
            (agent_id, agent)
            for agent_id, agent in state.agents.items()
            if agent.life_state == "active" and not agent.sheltered and not world._inside_fire_radius(agent.x, agent.y)
        ]
        non_torch = [item for item in active_targets if not _lit_torch(item[1])]
        targets = non_torch or active_targets
        if not targets:
            downed = [(agent_id, agent) for agent_id, agent in state.agents.items() if agent.life_state == "downed"]
            targets = downed
        if not targets:
            shelter = state.structures["shelter"]
            if abs(creature.x - shelter.x) + abs(creature.y - shelter.y) <= 1:
                _damage_structure(world, shelter, 10)
            elif state.step_count % 2 == 0:
                proxy = AgentState(x=shelter.x, y=shelter.y)
                creature.x, creature.y = world._next_stalker_step(creature, proxy)
            continue
        target_id, target_agent = min(targets, key=lambda item: (abs(item[1].x - creature.x) + abs(item[1].y - creature.y), _tie_key(state.step_count, creature.id, item[0])))
        creature.target = target_id
        if abs(target_agent.x - creature.x) + abs(target_agent.y - creature.y) <= 1:
            if target_agent.life_state == "downed":
                _kill(world, target_id, results, reason="attacked_while_downed")
                continue
            defenders = [agent_id for agent_id in defense.get(creature.id, []) if state.agents[agent_id].life_state == "active" and abs(state.agents[agent_id].x - creature.x) + abs(state.agents[agent_id].y - creature.y) <= 1]
            if len(defenders) >= 2:
                state.prevented_damage += 25
                _ledger(world, "joint_defense", category="contribution", actors=defenders, target=creature.id, quantity=25)
                _event(world, "joint_defense", actors=defenders, targets=[creature.id, target_id], payload={"prevented_damage": 25})
                continue
            prevented = 8 if defenders else 0
            state.prevented_damage += prevented
            if _damage_agent(world, target_id, 25 - prevented, results):
                newly_downed.add(target_id)
            _ledger(world, "stalker_attack", category="damage", actors=[creature.id], target=target_id, quantity=25 - prevented)
        elif state.step_count % 2 == 0:
            creature.x, creature.y = world._next_stalker_step(creature, target_agent)
    return newly_downed


def _damage_agent(world: MultiAgentWorld, agent_id: str, damage: int, results: dict[str, AgentStepResult]) -> bool:
    state = _state(world)
    agent = state.agents[agent_id]
    agent.health = max(0.0, agent.health - damage)
    if agent.health > 0:
        return False
    newly: set[str] = set()
    _down_or_kill(world, agent_id, results, newly)
    return agent_id in newly


def _down_or_kill(world: MultiAgentWorld, agent_id: str, results: dict[str, AgentStepResult], newly: set[str]) -> None:
    state = _state(world)
    agent = state.agents[agent_id]
    if agent.downed_count > 0:
        _kill(world, agent_id, results, reason="repeat_incapacitation")
        return
    agent.life_state = "downed"
    agent.downed_count += 1
    agent.downed_ticks = 20
    agent.health = 0
    agent.sheltered = False
    shelter = state.structures["shelter"]
    shelter.occupants.discard(agent_id)
    if agent_id in shelter.occupancy_order:
        shelter.occupancy_order.remove(agent_id)
    newly.add(agent_id)
    _ledger(
        world,
        "agent_downed",
        category="downing",
        target=agent_id,
        quantity=20,
    )
    _event(world, "agent_downed", targets=[agent_id], payload={"rescue_ticks": 20})


def _advance_downed(world: MultiAgentWorld, results: dict[str, AgentStepResult], newly: set[str]) -> None:
    state = _state(world)
    for agent_id, agent in state.agents.items():
        if agent.life_state != "downed" or agent_id in newly:
            continue
        agent.downed_ticks -= 1
        if agent.downed_ticks <= 0:
            _kill(world, agent_id, results, reason="revival_timeout")


def _kill(world: MultiAgentWorld, agent_id: str, results: dict[str, AgentStepResult], *, reason: str) -> None:
    state = _state(world)
    agent = state.agents[agent_id]
    if agent.life_state == "dead":
        return
    agent.life_state = "dead"
    agent.alive = False
    agent.sheltered = False
    agent.health = 0
    state.deaths += 1
    state.structures["shelter"].occupants.discard(agent_id)
    _ledger(
        world,
        "agent_died",
        category="death",
        target=agent_id,
        quantity=1,
        details={"reason": reason},
    )
    _event(world, "agent_died", targets=[agent_id], payload={"reason": reason})
    if agent_id in results:
        results[agent_id] = replace(results[agent_id], terminated=True, event="death")


def _advance_time_and_tools(world: MultiAgentWorld) -> None:
    state = _state(world)
    world._advance_civilization_time()
    if 200 <= state.step_count % 300 < 300:
        for agent_id, agent in state.agents.items():
            if agent.life_state == "active" and agent.equipped_tool == "torch" and agent.tool_charges.get("torch", 0) > 0:
                agent.tool_charges["torch"] -= 1
                _ledger(world, "torch_burn", category="energy_sink", actors=[agent_id], tool="torch", quantity=1)


def _apply_shelter_exits(
    world: MultiAgentWorld,
    actions: dict[str, CivilizationV2Action],
    invalid: set[str],
) -> None:
    state = _state(world)
    shelter = state.structures["shelter"]
    for agent_id, action in actions.items():
        agent = state.agents[agent_id]
        if agent_id in invalid or not agent.sheltered:
            continue
        remains = action.verb in {CivilizationV2Verb.NOOP, CivilizationV2Verb.REST} or (
            action.verb == CivilizationV2Verb.USE
            and action.argument == CivilizationV2Argument.SHELTER
        )
        if remains:
            continue
        agent.sheltered = False
        shelter.occupants.discard(agent_id)
        if agent_id in shelter.occupancy_order:
            shelter.occupancy_order.remove(agent_id)
        _event(world, "shelter_exit", actors=[agent_id], targets=["shelter"])


def _damage_structure(world: MultiAgentWorld, structure: StructureState, damage: int) -> None:
    before = structure.condition
    structure.condition = max(0, structure.condition - damage)
    _apply_structure_threshold(world, structure)
    _ledger(
        world,
        "structure_damage",
        category="damage",
        target=structure.id,
        quantity=before - structure.condition,
        details={"condition_before": before, "condition_after": structure.condition},
    )
    _event(world, "structure_damaged", targets=[structure.id], payload={"before": before, "after": structure.condition, "damage": damage})


def _apply_structure_threshold(world: MultiAgentWorld, structure: StructureState) -> None:
    state = _state(world)
    if structure.id != "shelter":
        if structure.id == "campfire" and structure.condition == 0:
            structure.fuel = 0
        return
    capacity = 6 if structure.condition >= 50 else 3 if structure.condition > 0 else 0
    structure.capacity = capacity
    state.camp.shelter_capacity = capacity
    while len(structure.occupancy_order) > capacity:
        agent_id = structure.occupancy_order.pop()
        structure.occupants.discard(agent_id)
        state.agents[agent_id].sheltered = False
        _event(world, "shelter_ejected", targets=[agent_id, structure.id])


def _expire_food(world: MultiAgentWorld) -> None:
    state = _state(world)
    locations: list[tuple[str, list[FoodLot]]] = [("camp", state.camp.food_lots)]
    locations.extend((agent_id, agent.food_lots) for agent_id, agent in state.agents.items())
    for owner, lots in locations:
        for lot in list(lots):
            if lot.expires_tick is None or lot.expires_tick > state.step_count:
                continue
            lots.remove(lot)
            item = _aggregate_food_kind(lot.kind)
            state.spoiled_resources[item] = state.spoiled_resources.get(item, 0) + lot.quantity
            _ledger(world, "food_spoiled", category="sink", source=owner, item=item, quantity=lot.quantity, balance={item: -lot.quantity}, lot_id=lot.id)
            _event(world, "food_spoiled", targets=[owner], payload={"item": item, "quantity": lot.quantity, "lot_id": lot.id})
    for pile_id, pile in list(state.ground_piles.items()):
        if pile.expires_tick is not None and pile.expires_tick <= state.step_count:
            state.spoiled_resources[pile.item] = state.spoiled_resources.get(pile.item, 0) + pile.quantity
            _ledger(world, "ground_food_spoiled", category="sink", source=pile_id, item=pile.item, quantity=pile.quantity, balance={pile.item: -pile.quantity})
            _event(
                world,
                "ground_food_spoiled",
                targets=[pile_id],
                position=(pile.x, pile.y),
                payload={"item": pile.item, "quantity": pile.quantity},
            )
            del state.ground_piles[pile_id]
    _sync_all_food(world)


def reconcile_ledger(world: MultiAgentWorld) -> dict[str, int]:
    state = _state(world)
    expected: Counter[str] = Counter()
    for entry in state.ledger:
        for item, delta in dict(entry.get("balance", {})).items():
            expected[str(item)] += int(delta)
    actual: Counter[str] = Counter()
    actual["food"] += int(np.sum(state.resource_quantities[state.resource_ids == Resource.FOOD]))
    actual["wood"] += int(np.sum(state.resource_quantities[state.resource_ids == Resource.WOOD]))
    actual["stone"] += int(np.sum(state.resource_quantities[state.resource_ids == Resource.STONE]))
    for item in ("wood", "stone"):
        actual[item] += state.camp.stockpile.get(item, 0)
        actual[item] += sum(agent.inventory.get(item, 0) for agent in state.agents.values())
    for lot in state.camp.food_lots:
        actual[_aggregate_food_kind(lot.kind)] += lot.quantity
    for agent in state.agents.values():
        for lot in agent.food_lots:
            actual[_aggregate_food_kind(lot.kind)] += lot.quantity
        for tool in agent.tools:
            actual[tool] += 1
    for pile in state.ground_piles.values():
        actual[pile.item] += pile.quantity
    for tool, values in state.camp.tool_stockpile.items():
        actual[tool] += len(values)
    keys = set(expected) | set(actual)
    return {key: actual[key] - expected[key] for key in sorted(keys) if actual[key] != expected[key]}


def contribution_metrics(world: MultiAgentWorld) -> dict[str, object]:
    """Derive contribution totals from the append-only ledger."""

    state = _state(world)
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in state.ledger:
        category = str(entry.get("category", "unknown"))
        quantity = int(entry.get("quantity", 0))
        for actor in entry.get("actors", []):
            actor_id = str(actor)
            if actor_id not in state.agents:
                continue
            by_agent[actor_id][category] += quantity
            by_role[state.agents[actor_id].role][category] += quantity
    return {
        "by_agent": {
            agent_id: dict(sorted(values.items()))
            for agent_id, values in sorted(by_agent.items())
        },
        "by_role": {
            role: dict(sorted(values.items())) for role, values in sorted(by_role.items())
        },
        "ledger_entries": len(state.ledger),
    }


def _ledger(
    world: MultiAgentWorld,
    event: str,
    *,
    category: str,
    actors: list[str] | None = None,
    source: str | None = None,
    target: str | None = None,
    item: str | None = None,
    tool: str | None = None,
    quantity: int = 0,
    balance: dict[str, int] | None = None,
    lot_id: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    state = _state(world)
    state.ledger.append(
        {
            "id": f"ledger-{len(state.ledger):08d}",
            "tick": state.step_count,
            "event": event,
            "category": category,
            "actors": list(actors or []),
            "source": source,
            "target": target,
            "item": item,
            "tool": tool,
            "quantity": quantity,
            "lot_id": lot_id,
            "balance": dict(balance or {}),
            "details": dict(details or {}),
        }
    )


def _event(
    world: MultiAgentWorld,
    event_type: str,
    *,
    actors: list[str] | None = None,
    targets: list[str] | None = None,
    position: tuple[int, int] | None = None,
    payload: dict[str, object] | None = None,
) -> None:
    world._record_event(event_type, actors=actors, targets=targets, position=position, payload=payload)


def _add_food_lot(
    world: MultiAgentWorld,
    lots: list[FoodLot],
    kind: str,
    quantity: int,
    origin_type: str,
    origin_id: str,
    *,
    preparation: str | None = None,
) -> FoodLot:
    state = _state(world)
    ttl = FOOD_TTLS[kind]
    lot = FoodLot(
        id=f"lot-{state.step_count}-{len(state.ledger)}-{kind}",
        kind=kind,
        quantity=quantity,
        origin_type=origin_type,
        origin_id=origin_id,
        created_tick=state.step_count,
        expires_tick=None if ttl is None else state.step_count + ttl,
        preparation=preparation or kind,
    )
    lots.append(lot)
    return lot


def _take_food(
    world: MultiAgentWorld,
    lots: list[FoodLot],
    aggregate_kind: str,
) -> FoodLot | None:
    candidates = [lot for lot in lots if lot.quantity > 0 and lot.kind in FOOD_ARGUMENT_KINDS[aggregate_kind]]
    if not candidates:
        return None
    selected = min(candidates, key=lambda lot: (lot.expires_tick if lot.expires_tick is not None else 10**12, lot.created_tick, lot.id))
    unit = replace(selected, quantity=1)
    selected.quantity -= 1
    if selected.quantity == 0:
        lots.remove(selected)
    return unit


def _take_specific_lot(lots: list[FoodLot], lot_id: str) -> FoodLot:
    selected = next(lot for lot in lots if lot.id == lot_id)
    unit = replace(selected, quantity=1)
    selected.quantity -= 1
    if selected.quantity == 0:
        lots.remove(selected)
    return unit


def _sync_all_food(world: MultiAgentWorld) -> None:
    state = _state(world)
    for aggregate, kinds in FOOD_ARGUMENT_KINDS.items():
        state.camp.stockpile[aggregate] = sum(
            lot.quantity for lot in state.camp.food_lots if lot.kind in kinds
        )
    for agent in state.agents.values():
        for aggregate, kinds in FOOD_ARGUMENT_KINDS.items():
            agent.inventory[aggregate] = sum(
                lot.quantity for lot in agent.food_lots if lot.kind in kinds
            )


def _target(slots: list[str], target: int) -> str | None:
    return slots[target - 1] if 1 <= target <= len(slots) else None


def _has_food(lots: list[FoodLot], aggregate: str) -> bool:
    return any(lot.quantity > 0 and lot.kind in FOOD_ARGUMENT_KINDS[aggregate] for lot in lots)


def _aggregate_food_kind(kind: str) -> str:
    if kind in {"wreck_ration", "berries"}:
        return "food"
    return kind


def _agent_item_count(agent: AgentState, item: str) -> int:
    if item in FOOD_ARGUMENT_KINDS:
        return sum(lot.quantity for lot in agent.food_lots if lot.kind in FOOD_ARGUMENT_KINDS[item])
    return agent.inventory.get(item, 0)


def _camp_item_count(state: Any, item: str) -> int:
    if item in FOOD_ARGUMENT_KINDS:
        return sum(lot.quantity for lot in state.camp.food_lots if lot.kind in FOOD_ARGUMENT_KINDS[item])
    return state.camp.stockpile.get(item, 0)


def _capacity(world: MultiAgentWorld, agent: AgentState) -> int:
    return world.inventory_capacity + (5 if "pack" in agent.tools else 0)


def _inventory_total(world: MultiAgentWorld, agent: AgentState) -> int:
    return agent.inventory.get("wood", 0) + agent.inventory.get("stone", 0) + sum(lot.quantity for lot in agent.food_lots)


def _interactable_here(world: MultiAgentWorld, agent: AgentState) -> bool:
    state = _state(world)
    if any(pile.quantity > 0 and (pile.x, pile.y) == (agent.x, agent.y) for pile in state.ground_piles.values()):
        return True
    return Resource(int(state.resource_ids[agent.y, agent.x])) != Resource.NONE and int(state.resource_quantities[agent.y, agent.x]) > 0


def _adjacent_agent(state: Any, agent: AgentState, target_id: str) -> bool:
    target = state.agents[target_id]
    return target.life_state != "dead" and abs(agent.x - target.x) + abs(agent.y - target.y) <= 1


def _lit_torch(agent: AgentState) -> bool:
    return agent.equipped_tool == "torch" and agent.tool_charges.get("torch", 0) > 0


def _tie_key(tick: int, subject: str, candidate: str) -> str:
    return hashlib.sha256(f"{tick}:{subject}:{candidate}".encode()).hexdigest()


def _state(world: MultiAgentWorld) -> Any:
    return world._require_state()
