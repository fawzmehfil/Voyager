"""Versioned environment adapters used by Voyager's shared PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voyager.envs import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
    VoyagerParallelEnv,
)
from voyager.sim.scenarios import CIVILIZATION_MAP_SIZE, CIVILIZATION_MAX_STEPS
from voyager.training.civilization_learning_ladder import (
    CONTRACT_TO_LEARNING_TASK,
    LEARNING_TASK_CONTRACTS,
    CivilizationLearningTaskWrapper,
    learning_task_definition,
)
from voyager.training.civilization_probe import (
    PROBE_REWARD_V1,
    PROBE_REWARD_V2,
    PROBE_REWARD_V4,
    PROBE_REWARD_VERSION,
    CivilizationProbeRewardWrapper,
)
from voyager.training.obs import (
    CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER,
    CIVILIZATION_V2_OBSERVATION_ENCODER,
    CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER,
    CIVILIZATION_V4_TEAM_OBJECTIVE_OBSERVATION_ENCODER,
    COMPACT_OBSERVATION_ENCODER,
)
from voyager.versions import (
    ACHIEVEMENT_VERSION,
    ACTION_VERSION,
    DENSE_REWARD_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    SCENARIO_VERSION,
)

COMPACT_TRAINING_ENVIRONMENT = "VoyagerSurvival-v0"
CIVILIZATION_V2_TRAINING_ENVIRONMENT = "VoyagerCivilization-v2"
ENVIRONMENT_REWARD_CONTRACT = "environment"
CIVILIZATION_PROBE_REWARD_CONTRACT = PROBE_REWARD_VERSION
CIVILIZATION_PROBE_V4_REWARD_CONTRACT = PROBE_REWARD_V4
CIVILIZATION_PROBE_V1_REWARD_CONTRACT = PROBE_REWARD_V1
CIVILIZATION_PROBE_V2_REWARD_CONTRACT = PROBE_REWARD_V2
CIVILIZATION_LEARNING_TASK_CONTRACTS = frozenset(LEARNING_TASK_CONTRACTS.values())


@dataclass(frozen=True, slots=True)
class TrainingEnvironment:
    """A constructed environment plus the contracts required to reload its policy."""

    env: Any
    observation_encoder: str
    num_agents: int
    map_size: int
    max_steps: int
    versions: dict[str, str]


def make_training_environment(
    *,
    environment_id: str,
    reward_contract: str,
    num_agents: int,
    map_size: int,
    max_steps: int,
    reward_mode: str,
    disabled_reward_components: tuple[str, ...],
    mask_role_observation: bool,
) -> TrainingEnvironment:
    """Construct one supported PPO environment without changing legacy defaults."""

    if environment_id == COMPACT_TRAINING_ENVIRONMENT:
        if reward_contract != ENVIRONMENT_REWARD_CONTRACT:
            raise ValueError("The compact environment requires reward_contract='environment'.")
        env = VoyagerParallelEnv(
            num_agents=num_agents,
            map_size=map_size,
            max_steps=max_steps,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            disabled_reward_components=disabled_reward_components,
            mask_role_observation=mask_role_observation,
        )
        return TrainingEnvironment(
            env=env,
            observation_encoder=COMPACT_OBSERVATION_ENCODER,
            num_agents=num_agents,
            map_size=map_size,
            max_steps=max_steps,
            versions={
                "environment_version": ENVIRONMENT_VERSION,
                "reward_version": DENSE_REWARD_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "action_version": ACTION_VERSION,
                "achievement_version": ACHIEVEMENT_VERSION,
                "scenario_version": SCENARIO_VERSION,
                "training_revision": "5.6",
            },
        )
    if environment_id == CIVILIZATION_V2_TRAINING_ENVIRONMENT:
        if reward_contract in CIVILIZATION_LEARNING_TASK_CONTRACTS:
            task = CONTRACT_TO_LEARNING_TASK[reward_contract]
            definition = learning_task_definition(task)
            return TrainingEnvironment(
                env=CivilizationLearningTaskWrapper(task),
                observation_encoder=CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER,
                num_agents=10,
                map_size=CIVILIZATION_MAP_SIZE,
                max_steps=definition.horizon,
                versions={
                    "environment_version": "voyager_civilization_v2",
                    "reward_version": reward_contract,
                    "observation_version": "civilization_training_observation_v3",
                    "action_version": "civilization_flattened_action_v2",
                    "achievement_version": "civilization_achievements_v1",
                    "scenario_version": f"stage7c_learning_ladder_{task}_v1",
                    "training_revision": "7C-learning-ladder-v1",
                },
            )
        if reward_contract not in {
            CIVILIZATION_PROBE_V1_REWARD_CONTRACT,
            CIVILIZATION_PROBE_V2_REWARD_CONTRACT,
            CIVILIZATION_PROBE_REWARD_CONTRACT,
            CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        }:
            raise ValueError(
                "VoyagerCivilization-v2 training requires the versioned Stage 7C probe reward."
            )
        base = VoyagerCivilizationV2Env(reward_mode="none")
        env = CivilizationProbeRewardWrapper(
            CivilizationV2FlattenedActionWrapper(base),
            reward_version=reward_contract,
        )
        is_v3_probe = reward_contract == CIVILIZATION_PROBE_REWARD_CONTRACT
        is_v4_probe = reward_contract == CIVILIZATION_PROBE_V4_REWARD_CONTRACT
        is_identity_probe = reward_contract in {
            CIVILIZATION_PROBE_V2_REWARD_CONTRACT,
            CIVILIZATION_PROBE_REWARD_CONTRACT,
            CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        }
        return TrainingEnvironment(
            env=env,
            observation_encoder=(
                CIVILIZATION_V4_TEAM_OBJECTIVE_OBSERVATION_ENCODER
                if is_v4_probe
                else (
                    CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER
                    if is_v3_probe
                    else (
                        CIVILIZATION_V2_IDENTITY_OBSERVATION_ENCODER
                        if is_identity_probe
                        else CIVILIZATION_V2_OBSERVATION_ENCODER
                    )
                )
            ),
            num_agents=10,
            map_size=CIVILIZATION_MAP_SIZE,
            max_steps=CIVILIZATION_MAX_STEPS,
            versions={
                "environment_version": "voyager_civilization_v2",
                "reward_version": reward_contract,
                "observation_version": (
                    "civilization_training_observation_v4"
                    if is_v4_probe
                    else (
                        "civilization_training_observation_v3"
                        if is_v3_probe
                        else (
                            "civilization_training_observation_v2"
                            if is_identity_probe
                            else "civilization_local_observation_v2"
                        )
                    )
                ),
                "action_version": "civilization_flattened_action_v2",
                "achievement_version": "civilization_achievements_v1",
                "scenario_version": "voyager_civilization_vertical_slice_v1",
                "training_revision": (
                    "7C-team-objective-v4"
                    if is_v4_probe
                    else (
                        "7C-trainability-v3"
                        if is_v3_probe
                        else (
                            "7C-trainability-v2"
                            if is_identity_probe
                            else "7C-trainability-v1"
                        )
                    )
                ),
            },
        )
    raise ValueError(f"Unsupported PPO environment: {environment_id!r}.")
