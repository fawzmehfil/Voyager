"""Gymnasium registration for planned Voyager environments."""

from gymnasium.envs.registration import register, registry

ENV_IDS = (
    "VoyagerSingleAgent-v0",
    "VoyagerSurvival-v0",
)


def register_envs() -> None:
    """Register Stage 0 placeholder environment IDs with Gymnasium."""

    for env_id in ENV_IDS:
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="voyager.envs.placeholders:StageOnePlaceholderEnv",
        )
