"""Deterministic public-action controller for the Stage 7B replay."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from voyager.envs.civilization_v2 import VoyagerCivilizationV2Env
from voyager.policies.civilization_scripted import CivilizationScriptedController
from voyager.sim.constants import Resource
from voyager.sim.registries import (
    ARGUMENT_COUNT,
    VERB_COUNT,
    CivilizationArgument,
    CivilizationVerb,
)
from voyager.sim.registries_v2 import (
    V2_DIRECTION_ARGUMENTS,
    V2_RECIPE_ARGUMENTS,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
)

ActionPayload = dict[str, int]


class _V1PolicyView:
    """Present v1-shaped masks while forwarding the same shared v2 world."""

    def __init__(self, env: VoyagerCivilizationV2Env) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def action_mask(self, agent_id: str) -> np.ndarray:
        mask = np.zeros((VERB_COUNT, ARGUMENT_COUNT), dtype=np.int8)
        v2_mask = self._env.action_mask(agent_id)
        for verb in CivilizationVerb:
            for argument in CivilizationArgument:
                translated = _translate(self._env, agent_id, int(verb), int(argument))
                if translated is None:
                    continue
                try:
                    flat = flatten_v2_action(
                        translated["verb"], translated["argument"], translated["target"]
                    )
                except ValueError:
                    continue
                mask[int(verb), int(argument)] = v2_mask[flat]
        return mask


class CivilizationV2ScriptedController:
    """Run the established vertical-slice strategy entirely through v2 actions."""

    policy_id = "civilization_deterministic_core_script_v1"

    def __init__(self) -> None:
        self._base = CivilizationScriptedController()
        self._axe_transferred = False
        self._torch_deposited = False
        self._torch_roundtrip = False
        self._revival_seen = False
        self._revival_completed = False
        self._shelter_damage_seen = False
        self._shelter_repair_seen = False
        self._repair_progress_seen = False
        self._last_shelter_condition = 100

    def act_many(self, env: VoyagerCivilizationV2Env) -> dict[str, ActionPayload]:
        state = env.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before scripted control.")
        if any(event["type"] == "agent_revived" for event in state.events):
            self._revival_completed = True
        structures_ready = bool(state.structures) and all(
            structure.complete for structure in state.structures.values()
        )
        time = env.world.civilization_time()
        if structures_ready and not (
            str(time["phase"]) == "night" and state.step_count < 300
        ):
            actions = self._extension_actions(env)
            _filter_failed_movements(env, actions)
            return actions
        legacy_actions = self._base.act_many(_V1PolicyView(env))  # type: ignore[arg-type]
        translated: dict[str, ActionPayload] = {}
        for agent_id, payload in legacy_actions.items():
            action = _translate(env, agent_id, payload["verb"], payload["argument"])
            translated[agent_id] = action or {
                "verb": int(CivilizationV2Verb.NOOP),
                "argument": int(CivilizationV2Argument.NONE),
                "target": 0,
            }
        early_food = self._early_food_action(env)
        if early_food is not None:
            agent_id, selected = early_food
            translated[agent_id] = selected
        _filter_failed_movements(env, translated)
        return translated

    def _extension_actions(
        self, env: VoyagerCivilizationV2Env
    ) -> dict[str, ActionPayload]:
        state = env.world.state
        assert state is not None
        noop = _action(CivilizationV2Verb.NOOP)
        actions = {agent_id: dict(noop) for agent_id in env.agents}
        active = {
            agent_id
            for agent_id in env.agents
            if state.agents[agent_id].life_state == "active"
        }
        shelter = state.structures["shelter"]
        if (
            self._shelter_damage_seen
            and shelter.condition > self._last_shelter_condition
        ):
            self._repair_progress_seen = True
        self._last_shelter_condition = shelter.condition
        if shelter.condition < 100 and not self._shelter_repair_seen:
            self._shelter_damage_seen = True
        elif self._shelter_damage_seen:
            self._shelter_repair_seen = True

        downed = sorted(
            agent_id for agent_id in env.agents if state.agents[agent_id].life_state == "downed"
        )
        if downed and not self._revival_completed:
            self._revival_seen = True
            target_id = downed[0]
            helper_candidates = [
                agent_id
                for agent_id in (
                    "agent_0",
                    "agent_1",
                    "agent_2",
                    "agent_3",
                    "agent_4",
                    "agent_5",
                    "agent_7",
                )
                if agent_id in active and agent_id != target_id
            ]
            helpers = sorted(
                helper_candidates,
                key=lambda helper: (
                    -sum(lot.quantity for lot in state.agents[helper].food_lots),
                    helper,
                ),
            )[:3]
            target_ref = f"agent:{target_id}"
            target_agent = state.agents[target_id]
            ready_helpers = [
                helper
                for helper in helpers
                if abs(state.agents[helper].x - target_agent.x)
                + abs(state.agents[helper].y - target_agent.y)
                <= 1
            ]
            food_available = (
                state.agents[target_id].revival_food_lot_id is not None
                or any(
                    lot.quantity > 0
                    for helper in ready_helpers
                    for lot in state.agents[helper].food_lots
                )
            )
            for helper in helpers:
                slots = env.world.v2_entity_slots(helper)
                if food_available and target_ref in slots:
                    candidate = _action(
                        CivilizationV2Verb.REVIVE,
                        target=slots.index(target_ref) + 1,
                    )
                    if _legal(env, helper, candidate):
                        actions[helper] = candidate
                        continue
                target = state.agents[target_id]
                actions[helper] = self._move_to(
                    env,
                    helper,
                    self._near_positions(env, target.x, target.y, 1),
                )
            return actions

        desired = {
            "agent_0": ("axe", CivilizationV2Argument.AXE_RECIPE),
            "agent_3": ("pickaxe", CivilizationV2Argument.PICKAXE_RECIPE),
            "agent_5": ("torch", CivilizationV2Argument.TORCH_RECIPE),
            "agent_6": ("pack", CivilizationV2Argument.PACK_RECIPE),
        }
        pack_exists = any("pack" in agent.tools for agent in state.agents.values()) or bool(
            state.camp.tool_stockpile.get("pack")
        )
        if not pack_exists and "agent_6" not in active:
            fallback = next(
                (
                    agent_id
                    for agent_id in ("agent_4", "agent_2", "agent_7", "agent_1")
                    if agent_id in active
                ),
                None,
            )
            if fallback is not None:
                desired[fallback] = ("pack", CivilizationV2Argument.PACK_RECIPE)
        missing_tools = [
            agent_id
            for agent_id, (tool, _recipe) in desired.items()
            if agent_id in active and tool not in state.agents[agent_id].tools
        ]
        if missing_tools and not self._torch_deposited:
            workbench = state.structures["workbench"]
            for agent_id in missing_tools:
                _tool, recipe = desired[agent_id]
                _recipe_tool, costs = V2_RECIPE_ARGUMENTS[recipe]
                agent = state.agents[agent_id]
                missing_item = next(
                    (
                        item
                        for item, quantity in costs.items()
                        if agent.inventory.get(item, 0)
                        + state.camp.stockpile.get(item, 0)
                        < quantity
                    ),
                    None,
                )
                if missing_item is not None:
                    actions[agent_id] = self._gather_item(
                        env, agent_id, missing_item
                    )
                    continue
                craft = _action(CivilizationV2Verb.CRAFT, recipe)
                if _legal(env, agent_id, craft):
                    actions[agent_id] = craft
                else:
                    actions[agent_id] = self._move_to(
                        env,
                        agent_id,
                        self._near_positions(env, workbench.x, workbench.y, 1),
                    )
            return actions

        if not self._axe_transferred and {"agent_0", "agent_1"} <= active:
            receiver = state.agents["agent_1"]
            if "axe" in receiver.tools:
                self._axe_transferred = True
            else:
                slots = env.world.v2_entity_slots("agent_0")
                if "agent:agent_1" in slots:
                    give = _action(
                        CivilizationV2Verb.GIVE,
                        CivilizationV2Argument.AXE,
                        slots.index("agent:agent_1") + 1,
                    )
                    if _legal(env, "agent_0", give):
                        actions["agent_0"] = give
                        return actions
                actions["agent_0"] = self._move_to(env, "agent_0", {(23, 31)})
                actions["agent_1"] = self._move_to(env, "agent_1", {(24, 31)})
                return actions

        if not self._torch_roundtrip and "agent_5" in active:
            torchbearer = state.agents["agent_5"]
            if self._torch_deposited and "torch" in torchbearer.tools:
                self._torch_roundtrip = True
            elif self._torch_deposited:
                if (torchbearer.x, torchbearer.y) != (state.camp.x, state.camp.y):
                    actions["agent_5"] = self._move_to(
                        env, "agent_5", {(state.camp.x, state.camp.y)}
                    )
                else:
                    actions["agent_5"] = _action(
                        CivilizationV2Verb.WITHDRAW, CivilizationV2Argument.TORCH
                    )
                return actions
            elif torchbearer.tool_charges.get("torch", 0) > 0:
                if (torchbearer.x, torchbearer.y) != (state.camp.x, state.camp.y):
                    actions["agent_5"] = self._move_to(
                        env, "agent_5", {(state.camp.x, state.camp.y)}
                    )
                else:
                    actions["agent_5"] = _action(
                        CivilizationV2Verb.DEPOSIT, CivilizationV2Argument.TORCH
                    )
                    self._torch_deposited = True
                return actions
            elif torchbearer.equipped_tool != "torch":
                actions["agent_5"] = _action(
                    CivilizationV2Verb.EQUIP, CivilizationV2Argument.TORCH
                )
                return actions
            else:
                campfire = state.structures["campfire"]
                charge = _action(CivilizationV2Verb.USE, CivilizationV2Argument.CAMPFIRE)
                if _legal(env, "agent_5", charge):
                    actions["agent_5"] = charge
                else:
                    actions["agent_5"] = self._move_to(
                        env,
                        "agent_5",
                        self._near_positions(env, campfire.x, campfire.y, 1),
                    )
                return actions

        wood_donor_id = next(
            (
                agent_id
                for agent_id in sorted(active)
                if state.agents[agent_id].inventory.get("wood", 0) > 0
            ),
            None,
        )
        if state.camp.stockpile.get("wood", 0) < 8 and not self._shelter_damage_seen:
            if wood_donor_id is not None:
                donor = state.agents[wood_donor_id]
                if env.world._at_camp(donor):
                    actions[wood_donor_id] = _action(
                        CivilizationV2Verb.DEPOSIT, CivilizationV2Argument.WOOD
                    )
                else:
                    actions[wood_donor_id] = self._move_to(
                        env,
                        wood_donor_id,
                        self._near_positions(env, state.camp.x, state.camp.y, 1),
                    )
            else:
                gatherer_id = next(iter(sorted(active)), None)
                if gatherer_id is not None:
                    actions[gatherer_id] = self._gather_item(env, gatherer_id, "wood")
            return actions

        ration_agent_id = (
            "agent_0"
            if "agent_0" in active
            and sum(lot.quantity for lot in state.agents["agent_0"].food_lots) < 2
            else None
        )
        if ration_agent_id is not None and not self._shelter_damage_seen:
            donor = state.agents[ration_agent_id]
            if env.world._at_camp(donor):
                actions[ration_agent_id] = _action(
                    CivilizationV2Verb.WITHDRAW, CivilizationV2Argument.FOOD
                )
            else:
                actions[ration_agent_id] = self._move_to(
                    env,
                    ration_agent_id,
                    self._near_positions(env, state.camp.x, state.camp.y, 1),
                )
            return actions

        if shelter.condition < 100:
            for agent_id in ("agent_0", "agent_1", "agent_2", "agent_3"):
                if agent_id not in active:
                    continue
                slots = env.world.v2_entity_slots(agent_id)
                if "structure:shelter" in slots:
                    repair = _action(
                        CivilizationV2Verb.REPAIR,
                        target=slots.index("structure:shelter") + 1,
                    )
                    if (
                        shelter.repair_material_reserved
                        or state.camp.stockpile.get("wood", 0) > 0
                    ) and _legal(env, agent_id, repair):
                        actions[agent_id] = repair
                        continue
                actions[agent_id] = self._move_to(
                    env,
                    agent_id,
                    self._near_positions(env, shelter.x, shelter.y, 1),
                )
            if self._repair_progress_seen:
                stalkers = [
                    creature
                    for creature in state.creatures.values()
                    if creature.alive and creature.type == "night_stalker"
                ]
                fighters = [
                    agent_id
                    for agent_id in ("agent_4",)
                    if agent_id in active and "spear" in state.agents[agent_id].tools
                ]
                for fighter_id, stalker in zip(
                    fighters, sorted(stalkers, key=lambda value: value.id), strict=False
                ):
                    fighter = state.agents[fighter_id]
                    if fighter.equipped_tool != "spear":
                        actions[fighter_id] = _action(
                            CivilizationV2Verb.EQUIP, CivilizationV2Argument.SPEAR
                        )
                        continue
                    target_ref = f"creature:{stalker.id}"
                    slots = env.world.v2_entity_slots(fighter_id)
                    if target_ref in slots:
                        attack = _action(
                            CivilizationV2Verb.ATTACK,
                            target=slots.index(target_ref) + 1,
                        )
                        if _legal(env, fighter_id, attack):
                            actions[fighter_id] = attack
                            continue
                    actions[fighter_id] = self._move_to(
                        env,
                        fighter_id,
                        self._near_positions(env, stalker.x, stalker.y, 1),
                    )
            return actions

        hunter_id = next(
            (
                agent_id
                for agent_id in ("agent_7", "agent_4", "agent_2", "agent_1")
                if agent_id in active and "spear" in state.agents[agent_id].tools
            ),
            None,
        )
        if hunter_id is not None and state.cooked_meals == 0:
            hunter = state.agents[hunter_id]
            raw_count = hunter.inventory.get("raw_meat", 0)
            if raw_count > 0:
                campfire = state.structures["campfire"]
                cook = _action(
                    CivilizationV2Verb.CRAFT, CivilizationV2Argument.COOK_MEAT_RECIPE
                )
                if _legal(env, hunter_id, cook):
                    actions[hunter_id] = cook
                else:
                    actions[hunter_id] = self._move_to(
                        env,
                        hunter_id,
                        self._near_positions(env, campfire.x, campfire.y, 1),
                    )
                return actions
            piles = [pile for pile in state.ground_piles.values() if pile.quantity > 0]
            if piles:
                pile = min(piles, key=lambda value: value.id)
                interact = _action(CivilizationV2Verb.INTERACT)
                if (hunter.x, hunter.y) == (pile.x, pile.y) and _legal(
                    env, hunter_id, interact
                ):
                    actions[hunter_id] = interact
                else:
                    actions[hunter_id] = self._move_to(
                        env, hunter_id, {(pile.x, pile.y)}
                    )
                return actions
            deer = [
                creature
                for creature in state.creatures.values()
                if creature.alive and creature.type == "island_deer"
            ]
            if deer:
                creature = min(deer, key=lambda value: value.id)
                slots = env.world.v2_entity_slots(hunter_id)
                target_ref = f"creature:{creature.id}"
                candidate_attack: ActionPayload | None = None
                if target_ref in slots:
                    candidate_attack = _action(
                        CivilizationV2Verb.ATTACK,
                        target=slots.index(target_ref) + 1,
                    )
                if candidate_attack is not None and _legal(
                    env, hunter_id, candidate_attack
                ):
                    actions[hunter_id] = candidate_attack
                else:
                    if hunter.equipped_tool != "spear":
                        actions[hunter_id] = _action(
                            CivilizationV2Verb.EQUIP, CivilizationV2Argument.SPEAR
                        )
                    else:
                        actions[hunter_id] = self._move_to(
                            env,
                            hunter_id,
                            self._near_positions(env, creature.x, creature.y, 1),
                        )
                return actions

        if state.step_count < 450:
            return actions

        if not self._shelter_damage_seen:
            campfire = state.structures["campfire"]
            firekeeper_id = next(
                (agent_id for agent_id in ("agent_4", "agent_7", "agent_5") if agent_id in active),
                None,
            )
            if firekeeper_id is not None and campfire.fuel < 60:
                fuel = _action(
                    CivilizationV2Verb.USE, CivilizationV2Argument.CAMPFIRE
                )
                if _legal(env, firekeeper_id, fuel):
                    actions[firekeeper_id] = fuel
                else:
                    actions[firekeeper_id] = self._move_to(
                        env,
                        firekeeper_id,
                        self._near_positions(env, campfire.x, campfire.y, 1),
                    )
            shelter_group = [
                agent_id for agent_id in sorted(active) if agent_id != firekeeper_id
            ][:6]
            for agent_id in shelter_group:
                enter = _action(
                    CivilizationV2Verb.USE, CivilizationV2Argument.SHELTER
                )
                if _legal(env, agent_id, enter):
                    actions[agent_id] = enter
                elif not state.agents[agent_id].sheltered:
                    actions[agent_id] = self._move_to(
                        env,
                        agent_id,
                        self._near_positions(env, shelter.x, shelter.y, 2),
                    )
            return actions

        victim_id = "agent_1" if "agent_1" in active else hunter_id
        if victim_id is not None and not self._revival_seen:
            formation = {
                victim_id: (20, 31),
                "agent_0": (21, 31),
                "agent_2": (20, 30),
                "agent_3": (20, 32),
            }
            for agent_id, goal in formation.items():
                if agent_id in active:
                    actions[agent_id] = self._move_to(env, agent_id, {goal})
        for agent_id in active:
            if not self._revival_seen and agent_id in {
                victim_id,
                "agent_0",
                "agent_2",
                "agent_3",
            }:
                continue
            enter = _action(
                CivilizationV2Verb.USE, CivilizationV2Argument.SHELTER
            )
            if _legal(env, agent_id, enter):
                actions[agent_id] = enter
            elif not state.agents[agent_id].sheltered:
                actions[agent_id] = self._move_to(
                    env,
                    agent_id,
                    self._near_positions(env, state.camp.x, state.camp.y, 2),
                )
        return actions

    def _move_to(
        self,
        env: VoyagerCivilizationV2Env,
        agent_id: str,
        goals: set[tuple[int, int]],
    ) -> ActionPayload:
        state = env.world.state
        assert state is not None
        if state.agents[agent_id].energy < 5:
            rest = _action(CivilizationV2Verb.REST)
            if _legal(env, agent_id, rest):
                return rest
        occupied = {
            (agent.x, agent.y)
            for key, agent in state.agents.items()
            if key != agent_id and agent.alive and not agent.sheltered
        }
        legacy, _destination = self._base._move_toward(
            _V1PolicyView(env), agent_id, goals, occupied  # type: ignore[arg-type]
        )
        return _translate(env, agent_id, legacy["verb"], legacy["argument"]) or _action(
            CivilizationV2Verb.NOOP
        )

    def _early_food_action(
        self, env: VoyagerCivilizationV2Env
    ) -> tuple[str, ActionPayload] | None:
        state = env.world.state
        assert state is not None
        if state.cooked_meals > 0:
            return None
        hunter_id = next(
            (
                agent_id
                for agent_id in ("agent_7", "agent_4", "agent_2", "agent_1")
                if state.agents[agent_id].life_state == "active"
                and "spear" in state.agents[agent_id].tools
            ),
            None,
        )
        if hunter_id is None:
            return None
        hunter = state.agents[hunter_id]
        if hunter.inventory.get("raw_meat", 0) > 0:
            campfire = state.structures["campfire"]
            if not campfire.complete:
                return None
            cook = _action(
                CivilizationV2Verb.CRAFT, CivilizationV2Argument.COOK_MEAT_RECIPE
            )
            if _legal(env, hunter_id, cook):
                return hunter_id, cook
            return hunter_id, self._move_to(
                env,
                hunter_id,
                self._near_positions(env, campfire.x, campfire.y, 1),
            )
        piles = [pile for pile in state.ground_piles.values() if pile.quantity > 0]
        if not piles:
            return None
        pile = min(piles, key=lambda value: value.id)
        interact = _action(CivilizationV2Verb.INTERACT)
        if (hunter.x, hunter.y) == (pile.x, pile.y) and _legal(
            env, hunter_id, interact
        ):
            return hunter_id, interact
        return hunter_id, self._move_to(env, hunter_id, {(pile.x, pile.y)})

    def _gather_item(
        self, env: VoyagerCivilizationV2Env, agent_id: str, item: str
    ) -> ActionPayload:
        state = env.world.state
        assert state is not None
        agent = state.agents[agent_id]
        resource = {"wood": Resource.WOOD, "stone": Resource.STONE}[item]
        interact = _action(CivilizationV2Verb.INTERACT)
        if (
            Resource(int(state.resource_ids[agent.y, agent.x])) == resource
            and _legal(env, agent_id, interact)
        ):
            return interact
        positions = {
            (int(x), int(y))
            for y, x in np.argwhere(
                (state.resource_ids == resource) & (state.resource_quantities > 0)
            )
        }
        return self._move_to(env, agent_id, positions)

    def _near_positions(
        self,
        env: VoyagerCivilizationV2Env,
        x: int,
        y: int,
        radius: int,
    ) -> set[tuple[int, int]]:
        return self._base._near_positions(
            _V1PolicyView(env), x, y, radius  # type: ignore[arg-type]
        )


def _translate(
    env: VoyagerCivilizationV2Env,
    agent_id: str,
    verb: int,
    argument: int,
) -> ActionPayload | None:
    old_verb = CivilizationVerb(verb)
    old_argument = CivilizationArgument(argument)
    new_verb = CivilizationV2Verb(verb)
    target = 0
    if old_argument.value <= CivilizationArgument.COOKED_MEAT:
        new_argument = CivilizationV2Argument(old_argument.value)
    elif old_argument == CivilizationArgument.SPEAR_RECIPE:
        new_argument = CivilizationV2Argument.SPEAR_RECIPE
    elif old_argument == CivilizationArgument.COOK_MEAT_RECIPE:
        new_argument = CivilizationV2Argument.COOK_MEAT_RECIPE
    elif old_argument == CivilizationArgument.SPEAR:
        new_verb = CivilizationV2Verb.EQUIP
        new_argument = CivilizationV2Argument.SPEAR
    elif old_argument in {
        CivilizationArgument.WORKBENCH,
        CivilizationArgument.CAMPFIRE,
        CivilizationArgument.SHELTER,
    }:
        new_argument = {
            CivilizationArgument.WORKBENCH: CivilizationV2Argument.WORKBENCH,
            CivilizationArgument.CAMPFIRE: CivilizationV2Argument.CAMPFIRE,
            CivilizationArgument.SHELTER: CivilizationV2Argument.SHELTER,
        }[old_argument]
    elif old_verb in {CivilizationVerb.ATTACK, CivilizationVerb.DEFEND}:
        old_slots = env.world.target_slots(agent_id)
        slot = old_argument.value - CivilizationArgument.TARGET_0
        if not 0 <= slot < len(old_slots):
            return None
        entity_ref = f"creature:{old_slots[slot]}"
        entity_slots = env.world.v2_entity_slots(agent_id)
        if entity_ref not in entity_slots:
            return None
        target = entity_slots.index(entity_ref) + 1
        new_argument = CivilizationV2Argument.NONE
    else:
        return None
    return {"verb": int(new_verb), "argument": int(new_argument), "target": target}


def _action(
    verb: CivilizationV2Verb,
    argument: CivilizationV2Argument = CivilizationV2Argument.NONE,
    target: int = 0,
) -> ActionPayload:
    return {"verb": int(verb), "argument": int(argument), "target": target}


def _legal(
    env: VoyagerCivilizationV2Env, agent_id: str, payload: ActionPayload
) -> bool:
    try:
        index = flatten_v2_action(
            payload["verb"], payload["argument"], payload["target"]
        )
    except ValueError:
        return False
    return bool(env.action_mask(agent_id)[index])


def _filter_failed_movements(
    env: VoyagerCivilizationV2Env, actions: dict[str, ActionPayload]
) -> None:
    """Turn resolver-known movement conflicts into intentional public no-ops."""

    state = env.world.state
    assert state is not None
    proposed: dict[str, tuple[int, int]] = {}
    for agent_id, payload in actions.items():
        if payload["verb"] != int(CivilizationV2Verb.MOVE):
            continue
        agent = state.agents[agent_id]
        dx, dy = V2_DIRECTION_ARGUMENTS[CivilizationV2Argument(payload["argument"])]
        proposed[agent_id] = (agent.x + dx, agent.y + dy)
    counts = Counter(proposed.values())
    failed = {agent_id for agent_id, target in proposed.items() if counts[target] > 1}
    proposed = {agent_id: target for agent_id, target in proposed.items() if agent_id not in failed}
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
        success = occupant is None or occupant in proposed and can_move(occupant)
        visiting.discard(agent_id)
        memo[agent_id] = success
        return success

    failed.update(agent_id for agent_id in proposed if not can_move(agent_id))
    for agent_id in failed:
        actions[agent_id] = {
            "verb": int(CivilizationV2Verb.NOOP),
            "argument": int(CivilizationV2Argument.NONE),
            "target": 0,
        }
