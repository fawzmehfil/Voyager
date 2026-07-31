"""Stage 7A registries layered on Voyager's stable compact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CivilizationVerb(IntEnum):
    """Structured action verbs for the Civilization scenario."""

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


class CivilizationArgument(IntEnum):
    """Arguments used by Stage 7A structured actions."""

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
    SPEAR_RECIPE = 10
    COOK_MEAT_RECIPE = 11
    SPEAR = 12
    WORKBENCH = 13
    CAMPFIRE = 14
    SHELTER = 15
    TARGET_0 = 16
    TARGET_1 = 17
    TARGET_2 = 18
    TARGET_3 = 19
    TARGET_4 = 20
    TARGET_5 = 21
    TARGET_6 = 22
    TARGET_7 = 23
    TARGET_8 = 24
    TARGET_9 = 25
    TARGET_10 = 26
    TARGET_11 = 27
    TARGET_12 = 28
    TARGET_13 = 29
    TARGET_14 = 30
    TARGET_15 = 31


class StructureType(IntEnum):
    """Public structures implemented by Stage 7A."""

    CAMP = 0
    SHELTER = 1
    WORKBENCH = 2
    CAMPFIRE = 3


class CreatureType(IntEnum):
    """Non-agent creatures implemented by Stage 7A."""

    ISLAND_DEER = 0
    NIGHT_STALKER = 1


VERB_COUNT = len(CivilizationVerb)
ARGUMENT_COUNT = len(CivilizationArgument)
TARGET_ARGUMENT_START = int(CivilizationArgument.TARGET_0)
TARGET_SLOT_COUNT = 16


@dataclass(frozen=True, slots=True)
class CivilizationAction:
    """Validated structured action passed from the environment to the shared world."""

    verb: CivilizationVerb
    argument: CivilizationArgument


ITEM_ARGUMENTS = {
    CivilizationArgument.FOOD: "food",
    CivilizationArgument.WOOD: "wood",
    CivilizationArgument.STONE: "stone",
    CivilizationArgument.RAW_MEAT: "raw_meat",
    CivilizationArgument.COOKED_MEAT: "cooked_meat",
}

DIRECTION_ARGUMENTS = {
    CivilizationArgument.NORTH: (0, -1),
    CivilizationArgument.EAST: (1, 0),
    CivilizationArgument.SOUTH: (0, 1),
    CivilizationArgument.WEST: (-1, 0),
}

STRUCTURE_ARGUMENTS = {
    CivilizationArgument.WORKBENCH: "workbench",
    CivilizationArgument.CAMPFIRE: "campfire",
    CivilizationArgument.SHELTER: "shelter",
}

MEANINGFUL_ACTION_PAIRS: tuple[tuple[int, int], ...] = (
    (CivilizationVerb.NOOP, CivilizationArgument.NONE),
    *((CivilizationVerb.MOVE, argument) for argument in DIRECTION_ARGUMENTS),
    (CivilizationVerb.INTERACT, CivilizationArgument.NONE),
    *(
        (CivilizationVerb.EAT, argument)
        for argument in (
            CivilizationArgument.FOOD,
            CivilizationArgument.RAW_MEAT,
            CivilizationArgument.COOKED_MEAT,
        )
    ),
    (CivilizationVerb.REST, CivilizationArgument.NONE),
    *((CivilizationVerb.DEPOSIT, argument) for argument in ITEM_ARGUMENTS),
    *((CivilizationVerb.WITHDRAW, argument) for argument in ITEM_ARGUMENTS),
    (CivilizationVerb.CRAFT, CivilizationArgument.SPEAR_RECIPE),
    (CivilizationVerb.CRAFT, CivilizationArgument.COOK_MEAT_RECIPE),
    *((CivilizationVerb.WORK, argument) for argument in STRUCTURE_ARGUMENTS),
    (CivilizationVerb.USE, CivilizationArgument.SPEAR),
    (CivilizationVerb.USE, CivilizationArgument.CAMPFIRE),
    (CivilizationVerb.USE, CivilizationArgument.SHELTER),
    *((CivilizationVerb.ATTACK, TARGET_ARGUMENT_START + slot) for slot in range(TARGET_SLOT_COUNT)),
    *((CivilizationVerb.DEFEND, TARGET_ARGUMENT_START + slot) for slot in range(TARGET_SLOT_COUNT)),
)

FLAT_ACTION_COUNT = len(MEANINGFUL_ACTION_PAIRS)
PAIR_TO_FLAT = {pair: index for index, pair in enumerate(MEANINGFUL_ACTION_PAIRS)}


def flatten_action(verb: int, argument: int) -> int:
    """Return the stable flattened ID for a meaningful structured action."""

    try:
        return PAIR_TO_FLAT[(CivilizationVerb(verb), CivilizationArgument(argument))]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Civilization action pair: ({verb}, {argument})") from exc


def unflatten_action(action: int) -> tuple[int, int]:
    """Decode a flattened Stage 7A action."""

    if not 0 <= action < FLAT_ACTION_COUNT:
        raise ValueError(f"Flattened Civilization action out of range: {action}")
    verb, argument = MEANINGFUL_ACTION_PAIRS[action]
    return int(verb), int(argument)
