"""Privileged deterministic controller for the Stage 7A reachability demonstration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from voyager.envs.civilization import VoyagerCivilizationEnv
from voyager.sim.constants import Resource, Terrain
from voyager.sim.registries import (
    DIRECTION_ARGUMENTS,
    ITEM_ARGUMENTS,
    STRUCTURE_ARGUMENTS,
    TARGET_ARGUMENT_START,
    CivilizationArgument,
    CivilizationVerb,
)

ActionPayload = dict[str, int]


@dataclass(slots=True)
class CivilizationScriptedController:
    """Coordinate the vertical slice using observations plus recorded privileged state."""

    policy_id: str = "civilization_vertical_slice_script_v1"

    def act_many(self, env: VoyagerCivilizationEnv) -> dict[str, ActionPayload]:
        """Return one legal action per live agent without mutating the environment."""

        state = env.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before scripted control.")
        occupied = {
            (agent.x, agent.y)
            for agent in state.agents.values()
            if agent.alive and not agent.sheltered
        }
        actions: dict[str, ActionPayload] = {}
        for agent_id in env.agents:
            agent = state.agents[agent_id]
            occupied.discard((agent.x, agent.y))
            action, destination = self._choose_action(env, agent_id, occupied)
            actions[agent_id] = action
            entering_shelter = (
                action["verb"] == int(CivilizationVerb.USE)
                and action["argument"] == int(CivilizationArgument.SHELTER)
            )
            if not agent.sheltered and not entering_shelter:
                occupied.add(destination if destination is not None else (agent.x, agent.y))
        return actions

    def _choose_action(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        occupied: set[tuple[int, int]],
    ) -> tuple[ActionPayload, tuple[int, int] | None]:
        state = env.world.state
        assert state is not None
        agent = state.agents[agent_id]
        mask = env.action_mask(agent_id)

        time = env.world.civilization_time()
        if str(time["phase"]) == "night" or int(time["tick_in_day"]) >= 185:
            return self._night_action(env, agent_id, occupied)

        if agent.hunger >= 60.0:
            for argument in (
                CivilizationArgument.COOKED_MEAT,
                CivilizationArgument.FOOD,
                CivilizationArgument.RAW_MEAT,
            ):
                action = self._action(CivilizationVerb.EAT, argument)
                if self._legal(mask, action):
                    return action, None
            withdraw = self._action(CivilizationVerb.WITHDRAW, CivilizationArgument.FOOD)
            if self._legal(mask, withdraw):
                return withdraw, None
            return self._move_toward(env, agent_id, {(state.camp.x, state.camp.y)}, occupied)
        if agent.energy <= 18.0:
            rest = self._action(CivilizationVerb.REST, CivilizationArgument.NONE)
            if self._legal(mask, rest):
                return rest, None

        workbench = state.structures["workbench"]
        if not workbench.complete:
            return self._build_goal(env, agent_id, "workbench", occupied)

        spear_holders = [
            value for value in state.agents.values() if value.alive and "spear" in value.tools
        ]
        if len(spear_holders) < 4:
            costs = {"wood": 2, "stone": 1}
            if "spear" not in agent.tools and env.world._has_materials(agent, costs):
                craft = self._action(
                    CivilizationVerb.CRAFT,
                    CivilizationArgument.SPEAR_RECIPE,
                )
                if self._legal(mask, craft):
                    return craft, None
                return self._move_toward(
                    env,
                    agent_id,
                    self._near_positions(env, workbench.x, workbench.y, 1),
                    occupied,
                )
            missing = self._missing_material(state, costs)
            if missing:
                return self._gather_goal(env, agent_id, missing, occupied)

        campfire = state.structures["campfire"]
        if not campfire.complete:
            return self._build_goal(env, agent_id, "campfire", occupied)

        hunters = sorted(
            (
                value
                for value in state.agents.values()
                if value.alive and "spear" in value.tools
            ),
            key=lambda value: next(
                key for key, candidate in state.agents.items() if candidate is value
            ),
            reverse=True,
        )
        designated_hunter = hunters[0] if hunters else None
        shelter = state.structures["shelter"]
        if not shelter.complete and agent is not designated_hunter:
            return self._build_goal(env, agent_id, "shelter", occupied)

        if agent.inventory.get("raw_meat", 0) > 0:
            cook = self._action(
                CivilizationVerb.CRAFT,
                CivilizationArgument.COOK_MEAT_RECIPE,
            )
            if self._legal(mask, cook):
                return cook, None
            return self._move_toward(
                env,
                agent_id,
                self._near_positions(env, campfire.x, campfire.y, 1),
                occupied,
            )

        if campfire.fuel < 110:
            fuel = self._action(CivilizationVerb.USE, CivilizationArgument.CAMPFIRE)
            if self._legal(mask, fuel):
                return fuel, None
            if state.camp.stockpile.get("wood", 0) < 5:
                return self._gather_goal(env, agent_id, "wood", occupied)
            return self._move_toward(
                env,
                agent_id,
                self._near_positions(env, campfire.x, campfire.y, 1),
                occupied,
            )

        deer = [
            creature
            for creature in state.creatures.values()
            if creature.alive and creature.type == "island_deer"
        ]
        if (
            state.hunts == 0
            and deer
            and agent is designated_hunter
            and agent.equipped_tool == "spear"
        ):
            target = min(
                deer,
                key=lambda value: abs(value.x - agent.x) + abs(value.y - agent.y),
            )
            attack = self._target_action(env, agent_id, target.id, CivilizationVerb.ATTACK)
            if attack is not None and self._legal(mask, attack):
                return attack, None
            return self._move_toward(
                env,
                agent_id,
                self._near_positions(env, target.x, target.y, 1),
                occupied,
            )
        if (
            agent is designated_hunter
            and "spear" in agent.tools
            and agent.equipped_tool != "spear"
        ):
            equip = self._action(CivilizationVerb.USE, CivilizationArgument.SPEAR)
            if self._legal(mask, equip):
                return equip, None

        if state.hunts == 0 and deer:
            return self._move_toward(env, agent_id, {(state.camp.x, state.camp.y)}, occupied)
        if not shelter.complete:
            return self._build_goal(env, agent_id, "shelter", occupied)

        if agent.hunger >= 45.0 and agent.inventory.get("food", 0) == 0:
            withdraw = self._action(CivilizationVerb.WITHDRAW, CivilizationArgument.FOOD)
            if self._legal(mask, withdraw):
                return withdraw, None
            return self._move_toward(env, agent_id, {(state.camp.x, state.camp.y)}, occupied)
        return self._move_toward(
            env,
            agent_id,
            self._near_positions(env, state.camp.x, state.camp.y, 3),
            occupied,
        )

    def _night_action(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        occupied: set[tuple[int, int]],
    ) -> tuple[ActionPayload, tuple[int, int] | None]:
        state = env.world.state
        assert state is not None
        agent = state.agents[agent_id]
        mask = env.action_mask(agent_id)
        shelter = state.structures["shelter"]
        campfire = state.structures["campfire"]

        attackers = {"agent_4", "agent_9"}
        defenders = {"agent_7", "agent_8"}
        exposed = attackers | defenders
        sheltered_ids = {"agent_0", "agent_1", "agent_2", "agent_3", "agent_5", "agent_6"}

        if agent_id in sheltered_ids:
            if agent.sheltered:
                rest = self._action(CivilizationVerb.REST, CivilizationArgument.NONE)
                if self._legal(mask, rest):
                    return rest, None
                return self._action(CivilizationVerb.NOOP, CivilizationArgument.NONE), None
            enter = self._action(CivilizationVerb.USE, CivilizationArgument.SHELTER)
            if self._legal(mask, enter):
                return enter, None
            return self._move_toward(
                env,
                agent_id,
                self._near_positions(env, shelter.x, shelter.y, 2),
                occupied,
            )

        if "spear" in agent.tools and agent.equipped_tool != "spear":
            equip = self._action(CivilizationVerb.USE, CivilizationArgument.SPEAR)
            if self._legal(mask, equip):
                return equip, None
        if campfire.fuel < 45:
            fuel = self._action(CivilizationVerb.USE, CivilizationArgument.CAMPFIRE)
            if self._legal(mask, fuel):
                return fuel, None

        stalkers = [
            creature
            for creature in state.creatures.values()
            if creature.alive and creature.type == "night_stalker"
        ]
        if stalkers:
            target = min(stalkers, key=lambda value: value.id)
            camp_distance = abs(target.x - state.camp.x) + abs(target.y - state.camp.y)
            if camp_distance <= 4:
                verb = (
                    CivilizationVerb.DEFEND
                    if agent_id in defenders
                    else CivilizationVerb.ATTACK
                )
                target_action = self._target_action(env, agent_id, target.id, verb)
                if target_action is not None and self._legal(mask, target_action):
                    return target_action, None
                return self._move_toward(
                    env,
                    agent_id,
                    self._near_positions(env, target.x, target.y, 1),
                    occupied,
                )

        exposed_ids = sorted(exposed)
        guard_offsets = ((4, 0), (4, 1), (4, -1), (5, 0))
        guard_index = exposed_ids.index(agent_id)
        dx, dy = guard_offsets[guard_index % len(guard_offsets)]
        return self._move_toward(
            env,
            agent_id,
            {(state.camp.x + dx, state.camp.y + dy)},
            occupied,
        )

    def _build_goal(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        structure_id: str,
        occupied: set[tuple[int, int]],
    ) -> tuple[ActionPayload, tuple[int, int] | None]:
        state = env.world.state
        assert state is not None
        structure = state.structures[structure_id]
        mask = env.action_mask(agent_id)
        missing = self._missing_material(state, structure.required_materials)
        if missing and not structure.reserved_materials:
            return self._gather_goal(env, agent_id, missing, occupied)
        argument = next(key for key, value in STRUCTURE_ARGUMENTS.items() if value == structure_id)
        work = self._action(CivilizationVerb.WORK, argument)
        if self._legal(mask, work):
            return work, None
        return self._move_toward(
            env,
            agent_id,
            self._near_positions(env, structure.x, structure.y, 1),
            occupied,
        )

    def _gather_goal(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        item: str,
        occupied: set[tuple[int, int]],
    ) -> tuple[ActionPayload, tuple[int, int] | None]:
        state = env.world.state
        assert state is not None
        agent = state.agents[agent_id]
        mask = env.action_mask(agent_id)
        if agent.inventory.get(item, 0) >= 3 or (
            item != "food" and sum(agent.inventory.values()) >= env.inventory_capacity - 1
        ):
            argument = next(key for key, value in ITEM_ARGUMENTS.items() if value == item)
            deposit = self._action(CivilizationVerb.DEPOSIT, argument)
            if self._legal(mask, deposit):
                return deposit, None
            return self._move_toward(env, agent_id, {(state.camp.x, state.camp.y)}, occupied)

        resource = {"food": Resource.FOOD, "wood": Resource.WOOD, "stone": Resource.STONE}[item]
        current = Resource(int(state.resource_ids[agent.y, agent.x]))
        interact = self._action(CivilizationVerb.INTERACT, CivilizationArgument.NONE)
        if current == resource and self._legal(mask, interact):
            return interact, None
        positions = {
            (int(x), int(y))
            for y, x in np.argwhere(
                (state.resource_ids == resource) & (state.resource_quantities > 0)
            )
        }
        if not positions:
            return self._move_toward(env, agent_id, {(state.camp.x, state.camp.y)}, occupied)
        return self._move_toward(env, agent_id, positions, occupied)

    def _move_toward(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        goals: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
    ) -> tuple[ActionPayload, tuple[int, int] | None]:
        state = env.world.state
        assert state is not None
        agent = state.agents[agent_id]
        start = (agent.x, agent.y)
        if start in goals:
            return self._action(CivilizationVerb.NOOP, CivilizationArgument.NONE), None
        queue: deque[tuple[int, int]] = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        found: tuple[int, int] | None = None
        while queue:
            position = queue.popleft()
            if position in goals:
                found = position
                break
            for dx, dy in DIRECTION_ARGUMENTS.values():
                candidate = (position[0] + dx, position[1] + dy)
                if candidate in parent or candidate in occupied:
                    continue
                x, y = candidate
                if not env.world._in_bounds(x, y) or state.terrain[y, x] == Terrain.WATER:
                    continue
                parent[candidate] = position
                queue.append(candidate)
        if found is None:
            return self._action(CivilizationVerb.NOOP, CivilizationArgument.NONE), None
        step = found
        while parent[step] not in {None, start}:
            step = parent[step]  # type: ignore[assignment]
        dx, dy = step[0] - start[0], step[1] - start[1]
        argument = next(key for key, delta in DIRECTION_ARGUMENTS.items() if delta == (dx, dy))
        action = self._action(CivilizationVerb.MOVE, argument)
        if not self._legal(env.action_mask(agent_id), action):
            return self._action(CivilizationVerb.NOOP, CivilizationArgument.NONE), None
        return action, step

    def _target_action(
        self,
        env: VoyagerCivilizationEnv,
        agent_id: str,
        creature_id: str,
        verb: CivilizationVerb,
    ) -> ActionPayload | None:
        slots = env.world.target_slots(agent_id)
        if creature_id not in slots:
            return None
        return self._action(
            verb,
            CivilizationArgument(TARGET_ARGUMENT_START + slots.index(creature_id)),
        )

    def _missing_material(self, state: object, costs: dict[str, int]) -> str | None:
        stockpile = state.camp.stockpile  # type: ignore[attr-defined]
        missing = [item for item, quantity in costs.items() if stockpile.get(item, 0) < quantity]
        return missing[0] if missing else None

    def _near_positions(
        self,
        env: VoyagerCivilizationEnv,
        x: int,
        y: int,
        radius: int,
    ) -> set[tuple[int, int]]:
        state = env.world.state
        assert state is not None
        return {
            (sample_x, sample_y)
            for sample_y in range(y - radius, y + radius + 1)
            for sample_x in range(x - radius, x + radius + 1)
            if abs(sample_x - x) + abs(sample_y - y) <= radius
            and env.world._in_bounds(sample_x, sample_y)
            and state.terrain[sample_y, sample_x] != Terrain.WATER
        }

    def _action(
        self,
        verb: CivilizationVerb,
        argument: CivilizationArgument,
    ) -> ActionPayload:
        return {"verb": int(verb), "argument": int(argument)}

    def _legal(self, mask: np.ndarray, action: ActionPayload) -> bool:
        return bool(mask[action["verb"], action["argument"]])
