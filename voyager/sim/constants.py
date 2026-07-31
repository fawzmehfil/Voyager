"""Shared constants and enums for Voyager's survival simulation."""

from enum import IntEnum


class Terrain(IntEnum):
    """Tile terrain identifiers used in observations."""

    WATER = 0
    BEACH = 1
    GRASS = 2
    FOREST = 3
    QUARRY = 4
    ROCKY_HIGHLAND = 5
    CAVE = 6


class Resource(IntEnum):
    """Resource identifiers used in map state and inventories."""

    NONE = 0
    FOOD = 1
    WOOD = 2
    STONE = 3


class Role(IntEnum):
    """Role identifiers for multi-agent observations."""

    FORAGER = 0
    WOODCUTTER = 1
    BUILDER = 2


class Action(IntEnum):
    """Discrete actions for the Stage 1 single-agent environment."""

    NOOP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    GATHER = 5
    EAT = 6
    REST = 7
    DEPOSIT_FOOD = 8
    DEPOSIT_WOOD = 9
    DEPOSIT_STONE = 10
    WITHDRAW_FOOD = 11
    BUILD_SHELTER = 12


ACTION_COUNT = len(Action)
ROLE_COUNT = len(Role)
