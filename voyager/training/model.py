"""TensorFlow actor-critic model helpers for Voyager PPO."""

from typing import Any

from voyager.sim.constants import ACTION_COUNT


def require_tensorflow() -> Any:
    """Import TensorFlow lazily and raise a useful setup error when unavailable."""

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        message = (
            "TensorFlow is required for PPO training. Use Python 3.11 or 3.12 and install "
            'the training extra with: python -m pip install -e ".[dev,train]"'
        )
        raise RuntimeError(message) from exc
    return tf


def build_actor_critic(
    input_dim: int,
    action_count: int = ACTION_COUNT,
    hidden_sizes: tuple[int, ...] = (128, 128),
    seed: int | None = None,
) -> Any:
    """Build a small MLP actor-critic model returning logits and value."""

    tf = require_tensorflow()
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="observation")
    hidden = inputs
    for index, hidden_size in enumerate(hidden_sizes):
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation="tanh",
            name=f"hidden_{index}",
        )(hidden)
    logits = tf.keras.layers.Dense(action_count, name="policy_logits")(hidden)
    value = tf.keras.layers.Dense(1, name="value")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=[logits, value], name="voyager_actor_critic")


def build_recurrent_actor_critic(
    input_dim: int,
    action_count: int = ACTION_COUNT,
    encoder_sizes: tuple[int, ...] = (128,),
    recurrent_hidden_size: int = 128,
    seed: int | None = None,
) -> Any:
    """Build a sequence actor-critic with an explicitly managed GRU state."""

    if recurrent_hidden_size <= 0:
        raise ValueError("recurrent_hidden_size must be positive.")
    if not encoder_sizes:
        raise ValueError("encoder_sizes must not be empty.")
    tf = require_tensorflow()
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    observations = tf.keras.Input(
        shape=(None, input_dim),
        dtype=tf.float32,
        name="observation_sequence",
    )
    initial_state = tf.keras.Input(
        shape=(recurrent_hidden_size,),
        dtype=tf.float32,
        name="initial_recurrent_state",
    )
    hidden = observations
    for index, hidden_size in enumerate(encoder_sizes):
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation="tanh",
            name=f"sequence_encoder_{index}",
        )(hidden)
    sequence, final_state = tf.keras.layers.GRU(
        recurrent_hidden_size,
        return_sequences=True,
        return_state=True,
        name="memory",
    )(hidden, initial_state=initial_state)
    logits = tf.keras.layers.Dense(action_count, name="policy_logits")(sequence)
    value = tf.keras.layers.Dense(1, name="value")(sequence)
    return tf.keras.Model(
        inputs=[observations, initial_state],
        outputs=[logits, value, final_state],
        name="voyager_recurrent_actor_critic",
    )
