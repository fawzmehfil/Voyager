"""Factorized recurrent PPO for the frozen Stage 7C Civilization task."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voyager.sim.registries_v2 import V2_FLAT_ACTION_COUNT
from voyager.training.factorized_actions import (
    FACTOR_ARGUMENT_COUNT,
    FACTOR_TARGET_COUNT,
    FACTOR_VERB_COUNT,
    action_components,
    argument_masks,
    choose_factorized_actions,
    target_masks,
    verb_masks,
)
from voyager.training.masking import stack_action_masks
from voyager.training.model import build_factorized_recurrent_actor_critic
from voyager.training.obs import flatten_observations
from voyager.training.recurrent_ppo import (
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    RecurrentRolloutBatch,
    _StepRecord,
)


class FactorizedRecurrentPPOTrainer(RecurrentPPOTrainer):
    """GRU PPO with a shared verb/argument/target actor and per-agent memory."""

    def __init__(self, config: RecurrentPPOConfig) -> None:
        super().__init__(config)
        if self.action_count != V2_FLAT_ACTION_COUNT:
            raise ValueError("Factorized recurrent PPO requires the Civilization v2 registry.")
        self.model = build_factorized_recurrent_actor_critic(
            input_dim=self.input_dim,
            verb_count=FACTOR_VERB_COUNT,
            argument_count=FACTOR_ARGUMENT_COUNT,
            target_count=FACTOR_TARGET_COUNT,
            encoder_sizes=config.encoder_sizes,
            recurrent_hidden_size=config.recurrent_hidden_size,
            seed=config.seed,
        )
        self.optimizer = self.tf.keras.optimizers.Adam(config.learning_rate)

    def collect_rollout(
        self,
        max_agent_steps: int | None = None,
    ) -> RecurrentRolloutBatch:
        """Collect episode-safe sequences with hierarchical joint log probabilities."""

        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided.")
        records: list[_StepRecord] = []

        for _step in range(self.config.rollout_steps):
            if not self.env.agents:
                self._reset_env()
            agent_ids = tuple(self.env.agents)
            if (
                max_agent_steps is not None
                and records
                and len(records) + len(agent_ids) > max_agent_steps
            ):
                break

            started = time.perf_counter()
            flat_observations = flatten_observations(
                self.observations,
                agent_ids,
                self.observation_encoder,
            )
            self.timing_seconds["observation_encoding"] += time.perf_counter() - started
            started = time.perf_counter()
            action_masks = stack_action_masks(
                self.infos,
                agent_ids,
                self.action_count,
            )
            if not self.config.use_action_mask:
                action_masks = np.ones_like(action_masks, dtype=np.bool_)
            self.timing_seconds["action_masks"] += time.perf_counter() - started

            initial_states = np.stack(
                [self.recurrent_states[agent_id] for agent_id in agent_ids],
                axis=0,
            )
            started = time.perf_counter()
            (
                verb_logits,
                argument_logits,
                target_logits,
                values,
                final_states,
            ) = self.model(
                [flat_observations[:, None, :], initial_states],
                training=False,
            )
            actions, old_log_probs = choose_factorized_actions(
                verb_logits=np.asarray(verb_logits)[:, 0, :],
                argument_logits=np.asarray(argument_logits)[:, 0, :],
                target_logits=np.asarray(target_logits)[:, 0, :],
                flat_masks=action_masks,
                inference_mode="seeded_stochastic",
                rng=self.rng,
            )
            step_values = np.asarray(values, dtype=np.float32)[:, 0, 0]
            final_states_array = np.asarray(final_states, dtype=np.float32)
            self.timing_seconds["actor_inference"] += time.perf_counter() - started

            action_map = {agent_id: int(actions[index]) for index, agent_id in enumerate(agent_ids)}
            started = time.perf_counter()
            (
                next_observations,
                rewards,
                terminations,
                truncations,
                step_infos,
            ) = self.env.step(action_map)
            self.timing_seconds["environment_step"] += time.perf_counter() - started
            self.world_steps += 1

            started = time.perf_counter()
            next_values = self._next_values(
                next_observations,
                final_states_array,
                agent_ids,
            )
            self.timing_seconds["next_value_inference"] += time.perf_counter() - started
            for index, agent_id in enumerate(agent_ids):
                done = bool(terminations[agent_id] or truncations[agent_id])
                records.append(
                    _StepRecord(
                        observation=flat_observations[index],
                        initial_state=initial_states[index],
                        action_mask=action_masks[index],
                        action=int(actions[index]),
                        old_log_prob=float(old_log_probs[index]),
                        value=float(step_values[index]),
                        reward=float(rewards[agent_id]),
                        done=done,
                        next_value=0.0 if done else next_values.get(agent_id, 0.0),
                        agent_id=agent_id,
                        episode=self.episode,
                    )
                )
                self.recurrent_states[agent_id] = (
                    np.zeros(
                        (self.config.recurrent_hidden_size,),
                        dtype=np.float32,
                    )
                    if done
                    else final_states_array[index]
                )
            self.observations = next_observations
            self.infos = step_infos

        if not records:
            raise RuntimeError("Factorized recurrent PPO collected an empty rollout.")
        started = time.perf_counter()
        self._assign_advantages(records)
        batch = self._sequence_batch(records)
        self.timing_seconds["batch_and_gae"] += time.perf_counter() - started
        return batch

    def update_policy(
        self,
        batch: RecurrentRolloutBatch,
        entropy_coef: float,
    ) -> dict[str, float]:
        """Apply clipped PPO to the sequence of joint hierarchical actions."""

        tf = self.tf
        sequence_count = int(batch.observations.shape[0])
        sequence_length = int(batch.observations.shape[1])
        indices = np.arange(sequence_count)
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(
                0,
                sequence_count,
                self.config.sequence_minibatch_size,
            ):
                selected = indices[start : start + self.config.sequence_minibatch_size]
                selected_count = len(selected)
                flat_masks = batch.action_masks[selected].reshape(
                    selected_count * sequence_length,
                    self.action_count,
                )
                flat_actions = batch.actions[selected].reshape(-1)
                verbs, arguments, targets = action_components(flat_actions)
                conditional_verb_masks = verb_masks(flat_masks).reshape(
                    selected_count,
                    sequence_length,
                    FACTOR_VERB_COUNT,
                )
                conditional_argument_masks = argument_masks(
                    flat_masks,
                    verbs,
                ).reshape(
                    selected_count,
                    sequence_length,
                    FACTOR_ARGUMENT_COUNT,
                )
                conditional_target_masks = target_masks(
                    flat_masks,
                    verbs,
                    arguments,
                ).reshape(
                    selected_count,
                    sequence_length,
                    FACTOR_TARGET_COUNT,
                )
                verbs = verbs.reshape(selected_count, sequence_length)
                arguments = arguments.reshape(selected_count, sequence_length)
                targets = targets.reshape(selected_count, sequence_length)

                observations = tf.convert_to_tensor(batch.observations[selected], dtype=tf.float32)
                initial_states = tf.convert_to_tensor(
                    batch.initial_states[selected], dtype=tf.float32
                )
                old_log_probs = tf.convert_to_tensor(
                    batch.old_log_probs[selected], dtype=tf.float32
                )
                advantages = tf.convert_to_tensor(batch.advantages[selected], dtype=tf.float32)
                returns = tf.convert_to_tensor(batch.returns[selected], dtype=tf.float32)
                valid = tf.convert_to_tensor(batch.valid[selected], dtype=tf.float32)
                denominator = tf.maximum(tf.reduce_sum(valid), 1.0)

                with tf.GradientTape() as tape:
                    (
                        verb_logits,
                        argument_logits,
                        target_logits,
                        values,
                        _final_states,
                    ) = self.model([observations, initial_states], training=True)
                    verb_log_probs, verb_entropy = self._component_statistics(
                        verb_logits,
                        conditional_verb_masks,
                        verbs,
                        FACTOR_VERB_COUNT,
                    )
                    argument_log_probs, argument_entropy = self._component_statistics(
                        argument_logits,
                        conditional_argument_masks,
                        arguments,
                        FACTOR_ARGUMENT_COUNT,
                    )
                    target_log_probs, target_entropy = self._component_statistics(
                        target_logits,
                        conditional_target_masks,
                        targets,
                        FACTOR_TARGET_COUNT,
                    )
                    log_probs = verb_log_probs + argument_log_probs + target_log_probs
                    values = tf.squeeze(values, axis=2)
                    ratio = tf.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = (
                        tf.clip_by_value(
                            ratio,
                            1.0 - self.config.clip_ratio,
                            1.0 + self.config.clip_ratio,
                        )
                        * advantages
                    )
                    policy_loss = (
                        -tf.reduce_sum(tf.minimum(unclipped, clipped) * valid) / denominator
                    )
                    value_loss = tf.reduce_sum(tf.square(returns - values) * valid) / denominator
                    entropy_by_step = verb_entropy + argument_entropy + target_entropy
                    entropy = tf.reduce_sum(entropy_by_step * valid) / denominator
                    total_loss = (
                        policy_loss + self.config.value_coef * value_loss - entropy_coef * entropy
                    )

                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                finite_gradients = [gradient for gradient in gradients if gradient is not None]
                clipped_gradients, _norm = tf.clip_by_global_norm(
                    finite_gradients,
                    self.config.max_gradient_norm,
                )
                variables = [
                    variable
                    for gradient, variable in zip(
                        gradients,
                        self.model.trainable_variables,
                        strict=True,
                    )
                    if gradient is not None
                ]
                self.optimizer.apply_gradients(zip(clipped_gradients, variables))
                loss_rows.append(
                    {
                        "policy_loss": float(policy_loss.numpy()),
                        "value_loss": float(value_loss.numpy()),
                        "entropy": float(entropy.numpy()),
                        "total_loss": float(total_loss.numpy()),
                    }
                )

        return {
            key: float(np.mean([row[key] for row in loss_rows]))
            for key in ("policy_loss", "value_loss", "entropy", "total_loss")
        }

    def _component_statistics(
        self,
        logits: Any,
        masks: np.ndarray,
        selected: np.ndarray,
        component_count: int,
    ) -> tuple[Any, Any]:
        tf = self.tf
        mask_tensor = tf.convert_to_tensor(masks, dtype=tf.bool)
        invalid_logits = tf.fill(tf.shape(logits), tf.cast(-1e9, logits.dtype))
        masked_logits = tf.where(mask_tensor, logits, invalid_logits)
        selection = tf.convert_to_tensor(selected, dtype=tf.int32)
        one_hot = tf.one_hot(selection, depth=component_count)
        log_softmax = tf.nn.log_softmax(masked_logits)
        selected_log_probs = tf.reduce_sum(one_hot * log_softmax, axis=2)
        probabilities = tf.nn.softmax(masked_logits)
        entropy_by_step = -tf.reduce_sum(probabilities * log_softmax, axis=2)
        return selected_log_probs, entropy_by_step

    def _next_values(
        self,
        observations: dict[str, dict[str, np.ndarray]],
        final_states: np.ndarray,
        prior_agent_ids: tuple[str, ...],
    ) -> dict[str, float]:
        if not observations:
            return {}
        state_by_agent = {
            agent_id: final_states[index] for index, agent_id in enumerate(prior_agent_ids)
        }
        agent_ids = tuple(observations)
        flat = flatten_observations(
            observations,
            agent_ids,
            self.observation_encoder,
        )
        initial_states = np.stack(
            [state_by_agent[agent_id] for agent_id in agent_ids],
            axis=0,
        )
        _verb, _argument, _target, values, _discarded_states = self.model(
            [flat[:, None, :], initial_states],
            training=False,
        )
        values_array = np.asarray(values, dtype=np.float32)[:, 0, 0]
        return {agent_id: float(values_array[index]) for index, agent_id in enumerate(agent_ids)}

    def _checkpoint_metadata(self, update: int) -> dict[str, object]:
        metadata = super()._checkpoint_metadata(update)
        metadata.update(
            {
                "algorithm": "factorized_shared_policy_recurrent_ppo",
                "model_type": "factorized_recurrent_gru",
                "verb_count": FACTOR_VERB_COUNT,
                "argument_count": FACTOR_ARGUMENT_COUNT,
                "target_count": FACTOR_TARGET_COUNT,
                "factorization": "P(verb)*P(argument|verb)*P(target|verb,argument)",
            }
        )
        return metadata
