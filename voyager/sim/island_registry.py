"""Frozen compact action registry for VoyagerIsland-v1."""

from enum import IntEnum


class IslandAction(IntEnum):
    """The complete public action space for the island benchmark."""

    NOOP = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_SOUTH = 3
    MOVE_WEST = 4
    INTERACT = 5
    ATTACK = 6
    EAT = 7
    REST = 8
    DEPOSIT_ALL = 9
    WITHDRAW_FOOD = 10
    CRAFT_AXE = 11
    CRAFT_SPEAR = 12
    WORK_WORKBENCH = 13
    WORK_CAMPFIRE = 14
    WORK_SHELTER = 15
    WORK_BEACON = 16
    USE_CAMPFIRE = 17
    USE_SHELTER = 18


ISLAND_ACTION_COUNT = len(IslandAction)
ISLAND_MOVEMENT_DELTAS = {
    IslandAction.MOVE_NORTH: (0, -1),
    IslandAction.MOVE_EAST: (1, 0),
    IslandAction.MOVE_SOUTH: (0, 1),
    IslandAction.MOVE_WEST: (-1, 0),
}
ISLAND_WORK_ACTIONS = {
    IslandAction.WORK_WORKBENCH: "workbench",
    IslandAction.WORK_CAMPFIRE: "campfire",
    IslandAction.WORK_SHELTER: "shelter",
    IslandAction.WORK_BEACON: "beacon",
}
ISLAND_ACTION_VERSION = "voyager_island_action_v1"
