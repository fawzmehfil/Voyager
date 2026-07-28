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

    reward_envs = {
        "VoyagerReward-v0": "dense",
        "VoyagerAchievement-v0": "achievement",
        "VoyagerNoReward-v0": "none",
    }
    for env_id, reward_mode in reward_envs.items():
        if env_id not in registry:
            register(
                id=env_id,
                entry_point="voyager.envs.parallel_env:VoyagerParallelEnv",
                order_enforce=False,
                disable_env_checker=True,
                kwargs={"reward_mode": reward_mode},
            )
