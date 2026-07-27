"""Gymnasium registration for planned Voyager environments."""

from gymnasium.envs.registration import register, registry


def register_envs() -> None:
    """Register Voyager environment IDs with Gymnasium."""

    if "VoyagerSingleAgent-v0" not in registry:
        register(
            id="VoyagerSingleAgent-v0",
            entry_point="voyager.envs.single_agent:VoyagerSingleAgentEnv",
        )

    if "VoyagerSurvival-v0" not in registry:
        register(
            id="VoyagerSurvival-v0",
            entry_point="voyager.envs.parallel_env:VoyagerParallelEnv",
            order_enforce=False,
            disable_env_checker=True,
        )
