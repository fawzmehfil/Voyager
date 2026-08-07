"""Training utilities for Voyager agents."""

from voyager.training.advantages import compute_gae
from voyager.training.factorized_ppo import FactorizedPPOTrainer
from voyager.training.factorized_recurrent_ppo import FactorizedRecurrentPPOTrainer
from voyager.training.island_evaluation import (
    LegalRandomIslandPolicy,
    RecurrentCheckpointIslandPolicy,
    RecurrentModelIslandPolicy,
    evaluate_island_checkpoint,
    evaluate_island_policy,
    fixed_island_trainability_gate,
    island_checkpoint_selection_key,
    normalize_island_evaluation_milestones,
    scripted_oracle_solvability_gate,
)
from voyager.training.island_reward import (
    ISLAND_TRAINING_REWARD_V2,
    ISLAND_TRAINING_REWARD_V3,
    ISLAND_TRAINING_REWARD_V4,
    IslandTrainingRewardV2Wrapper,
    IslandTrainingRewardV3Wrapper,
    IslandTrainingRewardV4Wrapper,
)
from voyager.training.obs import flat_observation_size, flatten_observation, flatten_observations
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats, RolloutBatch

__all__ = [
    "ISLAND_TRAINING_REWARD_V2",
    "ISLAND_TRAINING_REWARD_V3",
    "ISLAND_TRAINING_REWARD_V4",
    "FactorizedPPOTrainer",
    "FactorizedRecurrentPPOTrainer",
    "IslandTrainingRewardV2Wrapper",
    "IslandTrainingRewardV3Wrapper",
    "IslandTrainingRewardV4Wrapper",
    "LegalRandomIslandPolicy",
    "PPOConfig",
    "PPOTrainer",
    "PPOUpdateStats",
    "RecurrentCheckpointIslandPolicy",
    "RecurrentModelIslandPolicy",
    "RolloutBatch",
    "compute_gae",
    "evaluate_island_checkpoint",
    "evaluate_island_policy",
    "fixed_island_trainability_gate",
    "flat_observation_size",
    "flatten_observation",
    "flatten_observations",
    "island_checkpoint_selection_key",
    "normalize_island_evaluation_milestones",
    "scripted_oracle_solvability_gate",
]
