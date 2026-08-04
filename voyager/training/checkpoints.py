"""Checkpoint helpers for TensorFlow PPO policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyager.training.model import build_actor_critic, build_recurrent_actor_critic

METADATA_FILE = "metadata.json"
WEIGHTS_FILE = "model.weights.h5"


def save_policy_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    metadata: dict[str, object],
) -> Path:
    """Save model weights and metadata in a loadable checkpoint directory."""

    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / WEIGHTS_FILE
    model.save_weights(str(weights_path))
    full_metadata = dict(metadata)
    full_metadata["weights_file"] = WEIGHTS_FILE
    (path / METADATA_FILE).write_text(json.dumps(full_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_policy_checkpoint(checkpoint_path: str | Path) -> tuple[Any, dict[str, object]]:
    """Load an actor-critic model and metadata from a checkpoint directory."""

    path = Path(checkpoint_path)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    model_type = str(metadata.get("model_type", "feed_forward"))
    if model_type == "feed_forward":
        hidden_sizes = tuple(int(value) for value in metadata["hidden_sizes"])
        model = build_actor_critic(
            input_dim=int(metadata["input_dim"]),
            action_count=int(metadata["action_count"]),
            hidden_sizes=hidden_sizes,
        )
    elif model_type == "recurrent_gru":
        encoder_sizes = tuple(int(value) for value in metadata["encoder_sizes"])
        model = build_recurrent_actor_critic(
            input_dim=int(metadata["input_dim"]),
            action_count=int(metadata["action_count"]),
            encoder_sizes=encoder_sizes,
            recurrent_hidden_size=int(metadata["recurrent_hidden_size"]),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}.")
    model.load_weights(str(path / str(metadata["weights_file"])))
    return model, metadata
