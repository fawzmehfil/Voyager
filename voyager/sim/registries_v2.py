"""Versioned Stage 7B action, item, tool, and target registries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CivilizationV2Verb(IntEnum):
    NOOP = 0
    MOVE = 1
    INTERACT = 2
    EAT = 3
    REST = 4
    DEPOSIT = 5
    WITHDRAW = 6
    CRAFT = 7
    WORK = 8
    USE = 9
    ATTACK = 10
    DEFEND = 11
    GIVE = 12
    EQUIP = 13
    REPAIR = 14
    REVIVE = 15


class CivilizationV2Argument(IntEnum):
    NONE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4
    FOOD = 5
    WOOD = 6
    STONE = 7
    RAW_MEAT = 8
    COOKED_MEAT = 9
    AXE_RECIPE = 10
    PICKAXE_RECIPE = 11
    SPEAR_RECIPE = 12
    TORCH_RECIPE = 13
    PACK_RECIPE = 14
    COOK_MEAT_RECIPE = 15
    AXE = 16
    PICKAXE = 17
    SPEAR = 18
    TORCH = 19
    PACK = 20
    WORKBENCH = 21
    CAMPFIRE = 22
    SHELTER = 23


V2_TARGET_COUNT = 17
V2_ENTITY_SLOT_COUNT = 16


@dataclass(frozen=True, slots=True)
class CivilizationV2Action:
    verb: CivilizationV2Verb
    argument: CivilizationV2Argument
    target: int = 0


V2_DIRECTION_ARGUMENTS = {
    CivilizationV2Argument.NORTH: (0, -1),
    CivilizationV2Argument.EAST: (1, 0),
    CivilizationV2Argument.SOUTH: (0, 1),
    CivilizationV2Argument.WEST: (-1, 0),
}

V2_ITEM_ARGUMENTS = {
    CivilizationV2Argument.FOOD: "food",
    CivilizationV2Argument.WOOD: "wood",
    CivilizationV2Argument.STONE: "stone",
    CivilizationV2Argument.RAW_MEAT: "raw_meat",
    CivilizationV2Argument.COOKED_MEAT: "cooked_meat",
}

V2_TOOL_ARGUMENTS = {
    CivilizationV2Argument.AXE: "axe",
    CivilizationV2Argument.PICKAXE: "pickaxe",
    CivilizationV2Argument.SPEAR: "spear",
    CivilizationV2Argument.TORCH: "torch",
    CivilizationV2Argument.PACK: "pack",
}

V2_RECIPE_ARGUMENTS = {
    CivilizationV2Argument.AXE_RECIPE: ("axe", {"wood": 2, "stone": 1}),
    CivilizationV2Argument.PICKAXE_RECIPE: ("pickaxe", {"wood": 1, "stone": 2}),
    CivilizationV2Argument.SPEAR_RECIPE: ("spear", {"wood": 2, "stone": 1}),
    CivilizationV2Argument.TORCH_RECIPE: ("torch", {"wood": 1}),
    CivilizationV2Argument.PACK_RECIPE: ("pack", {"wood": 3, "stone": 1}),
}

V2_STRUCTURE_ARGUMENTS = {
    CivilizationV2Argument.WORKBENCH: "workbench",
    CivilizationV2Argument.CAMPFIRE: "campfire",
    CivilizationV2Argument.SHELTER: "shelter",
}

_NO_TARGET: tuple[tuple[int, int, int], ...] = (
    (CivilizationV2Verb.NOOP, CivilizationV2Argument.NONE, 0),
    *((CivilizationV2Verb.MOVE, argument, 0) for argument in V2_DIRECTION_ARGUMENTS),
    (CivilizationV2Verb.INTERACT, CivilizationV2Argument.NONE, 0),
    *((CivilizationV2Verb.EAT, argument, 0) for argument in (CivilizationV2Argument.FOOD, CivilizationV2Argument.RAW_MEAT, CivilizationV2Argument.COOKED_MEAT)),
    (CivilizationV2Verb.REST, CivilizationV2Argument.NONE, 0),
    *((CivilizationV2Verb.DEPOSIT, argument, 0) for argument in (*V2_ITEM_ARGUMENTS, *V2_TOOL_ARGUMENTS)),
    *((CivilizationV2Verb.WITHDRAW, argument, 0) for argument in (*V2_ITEM_ARGUMENTS, *V2_TOOL_ARGUMENTS)),
    *((CivilizationV2Verb.CRAFT, argument, 0) for argument in (*V2_RECIPE_ARGUMENTS, CivilizationV2Argument.COOK_MEAT_RECIPE)),
    *((CivilizationV2Verb.WORK, argument, 0) for argument in V2_STRUCTURE_ARGUMENTS),
    (CivilizationV2Verb.USE, CivilizationV2Argument.CAMPFIRE, 0),
    (CivilizationV2Verb.USE, CivilizationV2Argument.SHELTER, 0),
    *((CivilizationV2Verb.EQUIP, argument, 0) for argument in V2_TOOL_ARGUMENTS),
)

_TARGETED: tuple[tuple[int, int, int], ...] = tuple(
    (verb, argument, target)
    for verb, arguments in (
        (CivilizationV2Verb.GIVE, (*V2_ITEM_ARGUMENTS, *V2_TOOL_ARGUMENTS)),
        (CivilizationV2Verb.ATTACK, (CivilizationV2Argument.NONE,)),
        (CivilizationV2Verb.DEFEND, (CivilizationV2Argument.NONE,)),
        (CivilizationV2Verb.REPAIR, (CivilizationV2Argument.NONE,)),
        (CivilizationV2Verb.REVIVE, (CivilizationV2Argument.NONE,)),
    )
    for argument in arguments
    for target in range(1, V2_TARGET_COUNT)
)

V2_MEANINGFUL_ACTIONS = _NO_TARGET + _TARGETED
V2_ACTION_TO_FLAT = {action: index for index, action in enumerate(V2_MEANINGFUL_ACTIONS)}
V2_FLAT_ACTION_COUNT = len(V2_MEANINGFUL_ACTIONS)


def flatten_v2_action(verb: int, argument: int, target: int) -> int:
    try:
        return V2_ACTION_TO_FLAT[
            (CivilizationV2Verb(verb), CivilizationV2Argument(argument), int(target))
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Civilization v2 action: ({verb}, {argument}, {target})") from exc


def unflatten_v2_action(action: int) -> tuple[int, int, int]:
    if not 0 <= action < V2_FLAT_ACTION_COUNT:
        raise ValueError(f"Flattened Civilization v2 action out of range: {action}")
    verb, argument, target = V2_MEANINGFUL_ACTIONS[action]
    return int(verb), int(argument), int(target)
