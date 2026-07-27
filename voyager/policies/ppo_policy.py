"""TensorFlow PPO policy wrapper for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voyager.policies.base import Info, Observation, Policy
from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.model import require_tensorflow
from voyager.training.obs import flatten_observation


@dataclass(slots=True)
class TensorFlowPPOPolicy(Policy):
    """Load a Stage 5 PPO checkpoint and expose the common policy interface."""

    checkpoint_path: str | Path
    deterministic: bool = True
    model: Any = field(init=False, repr=False)
    metadata: dict[str, object] = field(init=False, repr=False)
    tf: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tf = require_tensorflow()
        self.model, self.metadata = load_policy_checkpoint(self.checkpoint_path)

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        _ = agent_id, info
        flat_obs = flatten_observation(observation)[None, :]
        logits, _value = self.model(flat_obs, training=False)
        if self.deterministic:
            return int(np.argmax(logits.numpy()[0]))
        action = self.tf.random.categorical(logits, num_samples=1)
        return int(action.numpy()[0, 0])
