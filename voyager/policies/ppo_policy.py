"""TensorFlow PPO policy wrapper for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voyager.policies.base import Info, Observation, Policy, PolicyDecision
from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.masking import mask_numpy_logits, stack_action_masks
from voyager.training.model import require_tensorflow
from voyager.training.obs import flatten_observations


@dataclass(slots=True)
class TensorFlowPPOPolicy(Policy):
    """Load a Stage 5 PPO checkpoint and expose the common policy interface."""

    checkpoint_path: str | Path
    deterministic: bool = True
    seed: int = 0
    use_action_mask: bool | None = None
    mask_role_observation: bool | None = None
    model: Any = field(init=False, repr=False)
    metadata: dict[str, object] = field(init=False, repr=False)
    tf: Any = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tf = require_tensorflow()
        self.model, self.metadata = load_policy_checkpoint(self.checkpoint_path)
        self.rng = np.random.default_rng(self.seed)
        if self.use_action_mask is None:
            self.use_action_mask = bool(self.metadata.get("action_masking", True))
        if self.mask_role_observation is None:
            self.mask_role_observation = not bool(
                self.metadata.get("role_observation", True)
            )

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        return self.decide_many(
            (agent_id,),
            {agent_id: observation},
            {agent_id: info},
        )[agent_id].action

    def reset(self, seed: int) -> None:
        """Reset only stochastic inference state while retaining loaded weights."""

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def decide_many(
        self,
        agent_ids: tuple[str, ...],
        observations: dict[str, Observation],
        infos: dict[str, Info],
    ) -> dict[str, PolicyDecision]:
        """Select actions for all live agents with one batched model call."""

        prepared = {
            agent_id: self._prepare_observation(observations[agent_id])
            for agent_id in agent_ids
        }
        flat_obs = flatten_observations(prepared, agent_ids)
        logits, _value = self.model(flat_obs, training=False)
        logits_array = np.asarray(logits.numpy(), dtype=np.float64)
        masks = stack_action_masks(infos, agent_ids)
        decisions: dict[str, PolicyDecision] = {}

        for index, agent_id in enumerate(agent_ids):
            raw_logits = logits_array[index]
            raw_probabilities = _probabilities(raw_logits)
            raw_action = int(np.argmax(raw_logits))
            invalid_probability_mass = float(np.sum(raw_probabilities[~masks[index]]))
            selected_logits = (
                mask_numpy_logits(raw_logits, masks[index])
                if self.use_action_mask
                else raw_logits
            )
            if self.deterministic:
                action = int(np.argmax(selected_logits))
            else:
                probabilities = _probabilities(selected_logits)
                action = int(self.rng.choice(len(probabilities), p=probabilities))
            decisions[agent_id] = PolicyDecision(
                action=action,
                raw_action=raw_action,
                invalid_probability_mass=invalid_probability_mass,
            )
        return decisions

    def _prepare_observation(self, observation: Observation) -> Observation:
        if not self.mask_role_observation:
            return observation
        return {
            **observation,
            "role": np.zeros_like(observation["role"]),
        }


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable_logits = logits - float(np.max(logits))
    probabilities = np.exp(stable_logits)
    return probabilities / np.sum(probabilities)
