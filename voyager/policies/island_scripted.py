"""Public-observation scripted solvability policy for VoyagerIsland-v1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from voyager.sim.constants import Resource
from voyager.sim.island_registry import IslandAction
from voyager.sim.scenarios import ISLAND_BENCHMARK_STRUCTURE_SPECS

WORK_ACTION = {
    "workbench": IslandAction.WORK_WORKBENCH,
    "campfire": IslandAction.WORK_CAMPFIRE,
    "shelter": IslandAction.WORK_SHELTER,
    "beacon": IslandAction.WORK_BEACON,
}
SITE = {"workbench": (-1, 0), "campfire": (1, 0), "shelter": (0, -2), "beacon": (0, 2)}
RECIPE = {
    name: (spec[0]["wood"], spec[0]["stone"])
    for name, spec in ISLAND_BENCHMARK_STRUCTURE_SPECS.items()
}
EXPLORATION_WAYPOINTS = (
    (0, -10),
    (-8, -8),
    (-4, -8),
    (0, -8),
    (4, -8),
    (8, -8),
    (8, -4),
    (10, 0),
    (4, -4),
    (0, -4),
    (-4, -4),
    (-8, -4),
    (-8, 0),
    (-4, 0),
    (0, 0),
    (4, 0),
    (8, 0),
    (8, 4),
    (0, 10),
    (4, 4),
    (0, 4),
    (-4, 4),
    (-8, 4),
    (-10, 0),
    (-8, 8),
    (-4, 8),
    (0, 8),
    (4, 8),
    (8, 8),
)
DEER_EXPLORATION_WAYPOINTS = (
    (0, -10),
    (5, -5),
    (10, 0),
    (5, 5),
    (0, 10),
    (-5, 5),
    (-10, 0),
    (-5, -5),
)


@dataclass(slots=True)
class _Memory:
    resources: dict[int, set[tuple[int, int]]] = field(
        default_factory=lambda: {
            int(Resource.FOOD): set(),
            int(Resource.WOOD): set(),
            int(Resource.STONE): set(),
        }
    )
    waypoint: int = 0
    deer_waypoint: int = 0
    deer: set[tuple[int, int]] = field(default_factory=set)
    pile_targets: set[tuple[int, int]] = field(default_factory=set)
    meal_prepared: bool = False


class ScriptedIslandOracle:
    """Coordinate two agents using only actor observations and legal masks."""

    def __init__(self) -> None:
        self.memory: dict[str, _Memory] = {}

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        self.memory = {agent_id: _Memory() for agent_id in possible_agents}

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        _ = infos
        actions = {
            agent_id: int(self._act_one(agent_id, observation))
            for agent_id, observation in observations.items()
        }
        destinations: dict[str, tuple[int, int]] = {}
        deltas = {
            int(IslandAction.MOVE_NORTH): (0, -1),
            int(IslandAction.MOVE_EAST): (1, 0),
            int(IslandAction.MOVE_SOUTH): (0, 1),
            int(IslandAction.MOVE_WEST): (-1, 0),
        }
        for agent_id, action in actions.items():
            if action not in deltas:
                continue
            position = self._relative_position(observations[agent_id])
            dx, dy = deltas[action]
            destinations[agent_id] = (position[0] + dx, position[1] + dy)
        if len(destinations) == 2 and len(set(destinations.values())) == 1:
            actions["agent_1"] = int(IslandAction.NOOP)
            destinations.pop("agent_1", None)
        positions = {
            agent_id: self._relative_position(observation)
            for agent_id, observation in observations.items()
        }
        for mover, destination in tuple(destinations.items()):
            blocker = next(
                (
                    other
                    for other, position in positions.items()
                    if other != mover
                    and position == destination
                    and float(observations[other]["self_state"][4]) < 0.5
                ),
                None,
            )
            if blocker is not None and blocker not in destinations:
                actions[mover] = int(IslandAction.NOOP)
                destinations.pop(mover, None)
        if actions.get("agent_0") == actions.get("agent_1") == int(IslandAction.WITHDRAW_FOOD):
            actions["agent_1"] = int(IslandAction.NOOP)
        crafting = {int(IslandAction.CRAFT_AXE), int(IslandAction.CRAFT_SPEAR)}
        if actions.get("agent_0") in crafting and actions.get("agent_1") in crafting:
            actions["agent_1"] = int(IslandAction.NOOP)
        if (
            actions.get("agent_0") == actions.get("agent_1") == int(IslandAction.INTERACT)
            and len(set(positions.values())) == 1
        ):
            actions["agent_1"] = int(IslandAction.NOOP)
        return actions

    def _act_one(
        self,
        agent_id: str,
        observation: Mapping[str, np.ndarray],
    ) -> IslandAction:
        memory = self.memory[agent_id]
        mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        local = np.asarray(observation["local_tiles"])
        inventory = np.asarray(observation["inventory"], dtype=np.float32) * 10.0
        self_state = np.asarray(observation["self_state"], dtype=np.float32)
        board = np.asarray(observation["public_board"], dtype=np.float32)
        identity = int(np.argmax(observation["identity"]))
        position = self._relative_position(observation)
        self._remember_visible_resources(memory, local, position)
        self._remember_visible_deer(memory, local, position)
        if inventory[4] > 0 or board[4] > 0:
            memory.meal_prepared = True

        if board[11] >= 0.5 and board[7] >= 1.0:
            if self_state[4] >= 0.5:
                return IslandAction.REST
            return self._travel_or_act(
                position, self._site_position("shelter", identity), mask, IslandAction.USE_SHELTER
            )
        if self._legal(mask, IslandAction.ATTACK):
            target = self._nearest_visible_creature(local, position, creature_type=1)
            if target is not None:
                memory.pile_targets.clear()
                memory.pile_targets.add(target)
            return IslandAction.ATTACK
        if self_state[4] >= 0.5 and self._legal(mask, IslandAction.USE_SHELTER):
            return IslandAction.USE_SHELTER
        if self_state[1] >= 0.55 and self._legal(mask, IslandAction.EAT):
            return IslandAction.EAT
        if self_state[1] >= 0.45 and inventory[0] + inventory[4] <= 0 and board[0] > 0:
            return self._travel_or_act(
                position,
                self._camp_position(identity),
                mask,
                IslandAction.WITHDRAW_FOOD,
            )
        if self_state[2] <= 0.20 and self._legal(mask, IslandAction.REST):
            return IslandAction.REST

        carried_material = inventory[1] + inventory[2]
        if carried_material > 0 and (
            carried_material >= 4 or self_state[3] >= 0.8 or self._camp_distance(position) <= 1
        ):
            return self._travel_or_act(
                position, self._camp_position(identity), mask, IslandAction.DEPOSIT_ALL
            )

        workbench_complete = board[5] >= 1.0
        axe_owned, spear_owned = observation["tools"]
        camp_wood, camp_stone = round(float(board[1]) * 40), round(float(board[2]) * 40)
        if (
            workbench_complete
            and identity == 0
            and not axe_owned
            and camp_wood >= 2
            and camp_stone >= 1
        ):
            return self._travel_or_act(position, (-1, 0), mask, IslandAction.CRAFT_AXE)
        if (
            workbench_complete
            and identity == 1
            and not spear_owned
            and camp_wood >= 2
            and camp_stone >= 1
        ):
            return self._travel_or_act(position, (0, 0), mask, IslandAction.CRAFT_SPEAR)

        stage = self._construction_stage(board, memory.meal_prepared)
        if stage is not None:
            required_wood, required_stone = RECIPE[stage]
            work_action = WORK_ACTION[stage]
            if camp_wood >= required_wood and camp_stone >= required_stone:
                return self._travel_or_act(
                    position,
                    self._site_position(stage, identity),
                    mask,
                    work_action,
                )

        if (inventory[3] > 0 or board[3] > 0) and board[6] >= 1.0:
            return self._travel_or_act(
                position,
                self._site_position("campfire", identity),
                mask,
                IslandAction.USE_CAMPFIRE,
            )

        pile_target = self._nearest(memory.pile_targets, position)
        if pile_target is not None:
            if position == pile_target:
                if self._legal(mask, IslandAction.INTERACT):
                    memory.pile_targets.discard(pile_target)
                    return IslandAction.INTERACT
                if (
                    self_state[3] >= 0.99
                    and inventory[0] + inventory[4] > 0
                    and self._legal(mask, IslandAction.EAT)
                ):
                    return IslandAction.EAT
                memory.pile_targets.discard(pile_target)
                pile_target = self._nearest(memory.pile_targets, position)
            if pile_target is not None:
                return self._move_toward(position, pile_target, mask)
        seeking_deer = (
            identity == 1
            and bool(spear_owned)
            and board[6] >= 1.0
            and not memory.meal_prepared
            and inventory[3] + inventory[4] + 40.0 * (board[3] + board[4]) <= 0
        )
        if seeking_deer:
            deer_target = self._nearest(memory.deer, position)
            if deer_target is not None:
                return self._move_toward(position, deer_target, mask)
            waypoint = DEER_EXPLORATION_WAYPOINTS[memory.deer_waypoint]
            if (
                position == waypoint
                or abs(position[0] - waypoint[0]) + abs(position[1] - waypoint[1]) <= 1
            ):
                memory.deer_waypoint = (memory.deer_waypoint + 1) % len(DEER_EXPLORATION_WAYPOINTS)
                waypoint = DEER_EXPLORATION_WAYPOINTS[memory.deer_waypoint]
            return self._move_toward(position, waypoint, mask)

        desired = int(Resource.WOOD if identity == 0 else Resource.STONE)
        if self._legal(mask, IslandAction.INTERACT):
            center_resource = int(local[3, 3, 1])
            if center_resource in {desired, int(Resource.FOOD)}:
                return IslandAction.INTERACT
        known_target = self._nearest(memory.resources[desired], position)
        if known_target is not None:
            return self._move_toward(position, known_target, mask)

        food_target = self._nearest(memory.resources[int(Resource.FOOD)], position)
        if inventory[0] <= 1 and food_target is not None:
            return self._move_toward(position, food_target, mask)
        waypoint_index = (
            memory.waypoint
            if identity == 0
            else (-memory.waypoint - 1) % len(EXPLORATION_WAYPOINTS)
        )
        waypoint = EXPLORATION_WAYPOINTS[waypoint_index]
        if (
            position == waypoint
            or abs(position[0] - waypoint[0]) + abs(position[1] - waypoint[1]) <= 1
        ):
            memory.waypoint = (memory.waypoint + 1) % len(EXPLORATION_WAYPOINTS)
            waypoint_index = (
                memory.waypoint
                if identity == 0
                else (-memory.waypoint - 1) % len(EXPLORATION_WAYPOINTS)
            )
            waypoint = EXPLORATION_WAYPOINTS[waypoint_index]
        return self._move_toward(position, waypoint, mask)

    @staticmethod
    def _construction_stage(
        board: np.ndarray,
        meal_prepared: bool,
    ) -> str | None:
        progress = dict(zip(("workbench", "campfire", "shelter", "beacon"), board[5:9]))
        if progress["workbench"] < 1.0:
            return "workbench"
        if progress["campfire"] < 1.0:
            return "campfire"
        if not meal_prepared:
            return None
        if progress["shelter"] < 1.0:
            return "shelter"
        if progress["beacon"] < 1.0:
            return "beacon"
        return None

    @staticmethod
    def _relative_position(observation: Mapping[str, np.ndarray]) -> tuple[int, int]:
        bearing = np.asarray(observation["camp_bearing"], dtype=np.float32)
        return -round(float(bearing[0]) * 47), -round(float(bearing[1]) * 47)

    @staticmethod
    def _remember_visible_resources(
        memory: _Memory,
        local: np.ndarray,
        position: tuple[int, int],
    ) -> None:
        visible: dict[int, set[tuple[int, int]]] = {
            int(Resource.FOOD): set(),
            int(Resource.WOOD): set(),
            int(Resource.STONE): set(),
        }
        for row in range(7):
            for column in range(7):
                resource = int(local[row, column, 1])
                if resource not in visible or int(local[row, column, 2]) <= 0:
                    continue
                visible[resource].add((position[0] + column - 3, position[1] + row - 3))
        window = {
            (position[0] + column - 3, position[1] + row - 3)
            for row in range(7)
            for column in range(7)
        }
        for resource, remembered in memory.resources.items():
            remembered.difference_update(window)
            remembered.update(visible[resource])

    @staticmethod
    def _remember_visible_deer(
        memory: _Memory,
        local: np.ndarray,
        position: tuple[int, int],
    ) -> None:
        visible: set[tuple[int, int]] = set()
        for row in range(7):
            for column in range(7):
                if int(local[row, column, 4]) == 1:
                    visible.add((position[0] + column - 3, position[1] + row - 3))
        window = {
            (position[0] + column - 3, position[1] + row - 3)
            for row in range(7)
            for column in range(7)
        }
        memory.deer.difference_update(window)
        memory.deer.update(visible)

    @staticmethod
    def _nearest_visible_creature(
        local: np.ndarray,
        position: tuple[int, int],
        *,
        creature_type: int,
    ) -> tuple[int, int] | None:
        candidates = [
            (position[0] + column - 3, position[1] + row - 3)
            for row in range(7)
            for column in range(7)
            if int(local[row, column, 4]) == creature_type
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                abs(target[0] - position[0]) + abs(target[1] - position[1]),
                target[1],
                target[0],
            ),
        )

    @staticmethod
    def _nearest(
        candidates: set[tuple[int, int]], position: tuple[int, int]
    ) -> tuple[int, int] | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                abs(target[0] - position[0]) + abs(target[1] - position[1]),
                target[1],
                target[0],
            ),
        )

    def _travel_or_act(
        self,
        position: tuple[int, int],
        target: tuple[int, int],
        mask: np.ndarray,
        action: IslandAction,
    ) -> IslandAction:
        if self._legal(mask, action):
            return action
        return self._move_toward(position, target, mask)

    @staticmethod
    def _move_toward(
        position: tuple[int, int],
        target: tuple[int, int],
        mask: np.ndarray,
    ) -> IslandAction:
        dx, dy = target[0] - position[0], target[1] - position[1]
        candidates: list[IslandAction] = []
        if abs(dx) >= abs(dy) and dx:
            candidates.append(IslandAction.MOVE_EAST if dx > 0 else IslandAction.MOVE_WEST)
        if dy:
            candidates.append(IslandAction.MOVE_SOUTH if dy > 0 else IslandAction.MOVE_NORTH)
        if dx and (
            not candidates or candidates[0] not in {IslandAction.MOVE_EAST, IslandAction.MOVE_WEST}
        ):
            candidates.append(IslandAction.MOVE_EAST if dx > 0 else IslandAction.MOVE_WEST)
        candidates.extend(
            [
                IslandAction.MOVE_NORTH,
                IslandAction.MOVE_EAST,
                IslandAction.MOVE_SOUTH,
                IslandAction.MOVE_WEST,
            ]
        )
        for candidate in candidates:
            if bool(mask[int(candidate)]):
                return candidate
        return IslandAction.REST if bool(mask[int(IslandAction.REST)]) else IslandAction.NOOP

    @staticmethod
    def _camp_position(identity: int) -> tuple[int, int]:
        return (0, 0) if identity == 0 else (1, 0)

    @staticmethod
    def _site_position(name: str, identity: int) -> tuple[int, int]:
        x, y = SITE[name]
        return (x, y) if identity == 0 else (x + 1, y)

    @staticmethod
    def _camp_distance(position: tuple[int, int]) -> int:
        return abs(position[0]) + abs(position[1])

    @staticmethod
    def _legal(mask: np.ndarray, action: IslandAction) -> bool:
        return bool(mask[int(action)])
