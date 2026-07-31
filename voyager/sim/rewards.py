"""Reward interface definitions shared by environments and training."""

from typing import Literal

RewardMode = Literal["dense", "achievement", "none"]

DENSE_REWARD_COMPONENTS = frozenset(
    {
        "alive",
        "action",
        "invalid",
        "hunger_control",
        "death",
        "group_survival",
        "food_security",
        "shelter_progress",
        "team_death",
        "episode_survival",
        "tool_progression",
        "food_preparation",
        "public_infrastructure",
        "joint_work",
        "defense",
    }
)

REWARD_MODES = frozenset({"dense", "achievement", "none"})
