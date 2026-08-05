"""Factorized shared-policy PPO for the frozen Civilization v2 action registry."""

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
from voyager.training.model import build_factorized_actor_critic
from voyager.training.obs import flatten_observations
from voyager.training.ppo import PPOConfig, PPOTrainer, RolloutBatch, _normalize_advantages


class FactorizedPPOTrainer(PPOTrainer):
    """PPO whose actor chooses a legal verb, then argument, then target."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)
        if self.action_count != V2_FLAT_ACTION_COUNT:
            raise ValueError("Factorized PPO requires the frozen Civilization v2 action registry.")
        self.model = build_factorized_actor_critic(
            input_dim=self.input_dim,
            verb_count=FACTOR_VERB_COUNT,
            argument_count=FACTOR_ARGUMENT_COUNT,
            target_count=FACTOR_TARGET_COUNT,
            hidden_sizes=config.hidden_sizes,
            seed=config.seed,
        )
        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    def collect_rollout(self, max_agent_steps: int | None = None) -> RolloutBatch:
        """Collect a rollout using exact hierarchical joint log probabilities."""

        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided.")

        obs_rows: list[np.ndarray] = []
        action_mask_rows: list[np.ndarray] = []
        actions_rows: list[int] = []
        log_prob_rows: list[float] = []
        value_rows: list[float] = []
        reward_rows: list[float] = []
        done_rows: list[bool] = []
        next_value_rows: list[float] = []
        agent_rows: list[str] = []

        for _step in range(self.config.rollout_steps):
            if not self.env.agents:
                self._reset_env()

            agent_ids = tuple(self.env.agents)
            if (
                max_agent_steps is not None
                and obs_rows
                and len(obs_rows) + len(agent_ids) > max_agent_steps
            ):
                break
            started = time.perf_counter()
            flat_obs = flatten_observations(
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
            self.timing_seconds["action_masks"] += time.perf_counter() - started
            if not self.config.use_action_mask:
                action_masks = np.ones_like(action_masks, dtype=np.bool_)

            started = time.perf_counter()
            verb_logits, argument_logits, target_logits, values = self.model(
                flat_obs,
                training=False,
            )
            actions, log_probs = choose_factorized_actions(
                verb_logits=np.asarray(verb_logits),
                argument_logits=np.asarray(argument_logits),
                target_logits=np.asarray(target_logits),
                flat_masks=action_masks,
                inference_mode="seeded_stochastic",
                rng=self.rng,
            )
            values_np = np.squeeze(values.numpy(), axis=1).astype(np.float32)
            self.timing_seconds["actor_inference"] += time.perf_counter() - started

            action_map = {agent_id: int(actions[index]) for index, agent_id in enumerate(agent_ids)}
            started = time.perf_counter()
            next_observations, rewards, terminations, truncations, step_infos = self.env.step(
                action_map
            )
            self.timing_seconds["environment_step"] += time.perf_counter() - started
            self.world_steps += 1
            started = time.perf_counter()
            next_values_by_agent = self._next_values(next_observations)
            self.timing_seconds["next_value_inference"] += time.perf_counter() - started

            for index, agent_id in enumerate(agent_ids):
                done = bool(terminations[agent_id] or truncations[agent_id])
                obs_rows.append(flat_obs[index])
                action_mask_rows.append(action_masks[index])
                actions_rows.append(int(actions[index]))
                log_prob_rows.append(float(log_probs[index]))
                value_rows.append(float(values_np[index]))
                reward_rows.append(float(rewards[agent_id]))
                done_rows.append(done)
                next_value_rows.append(0.0 if done else next_values_by_agent.get(agent_id, 0.0))
                agent_rows.append(agent_id)

            self.observations = next_observations
            self.infos.update(step_infos)

        started = time.perf_counter()
        observations = np.asarray(obs_rows, dtype=np.float32)
        action_masks_array = np.asarray(action_mask_rows, dtype=np.bool_)
        actions_array = np.asarray(actions_rows, dtype=np.int32)
        old_log_probs = np.asarray(log_prob_rows, dtype=np.float32)
        values_array = np.asarray(value_rows, dtype=np.float32)
        rewards_array = np.asarray(reward_rows, dtype=np.float32)
        dones_array = np.asarray(done_rows, dtype=np.float32)
        next_values_array = np.asarray(next_value_rows, dtype=np.float32)
        advantages, returns = self._advantages_by_agent(
            rewards=rewards_array,
            dones=dones_array,
            values=values_array,
            next_values=next_values_array,
            agent_ids=agent_rows,
        )
        advantages = _normalize_advantages(advantages)
        self.timing_seconds["batch_and_gae"] += time.perf_counter() - started

        return RolloutBatch(
            observations=observations,
            action_masks=action_masks_array,
            actions=actions_array,
            old_log_probs=old_log_probs,
            values=values_array,
            rewards=rewards_array,
            dones=dones_array,
            next_values=next_values_array,
            advantages=advantages,
            returns=returns,
            agent_ids=tuple(agent_rows),
        )

    def update_policy(
        self,
        batch: RolloutBatch,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        """Apply PPO clipping to the joint hierarchical action probability."""

        tf = self.tf
        if entropy_coef is None:
            entropy_coef = self.config.entropy_coefficient(self.agent_steps)
        sample_count = int(batch.actions.shape[0])
        indices = np.arange(sample_count)
        loss_rows: list[dict[str, float]] = []
        verbs, arguments, targets = action_components(batch.actions)

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(0, sample_count, self.config.minibatch_size):
                minibatch = indices[start : start + self.config.minibatch_size]
                flat_masks = batch.action_masks[minibatch]
                selected_verbs = verbs[minibatch]
                selected_arguments = arguments[minibatch]
                selected_targets = targets[minibatch]
                conditional_verb_masks = verb_masks(flat_masks)
                conditional_argument_masks = argument_masks(
                    flat_masks,
                    selected_verbs,
                )
                conditional_target_masks = target_masks(
                    flat_masks,
                    selected_verbs,
                    selected_arguments,
                )

                obs = tf.convert_to_tensor(batch.observations[minibatch], dtype=tf.float32)
                old_log_probs = tf.convert_to_tensor(
                    batch.old_log_probs[minibatch], dtype=tf.float32
                )
                advantages = tf.convert_to_tensor(batch.advantages[minibatch], dtype=tf.float32)
                returns = tf.convert_to_tensor(batch.returns[minibatch], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    verb_logits, argument_logits, target_logits, values = self.model(
                        obs,
                        training=True,
                    )
                    verb_log_probs, verb_entropy = self._component_statistics(
                        verb_logits,
                        conditional_verb_masks,
                        selected_verbs,
                        FACTOR_VERB_COUNT,
                    )
                    argument_log_probs, argument_entropy = self._component_statistics(
                        argument_logits,
                        conditional_argument_masks,
                        selected_arguments,
                        FACTOR_ARGUMENT_COUNT,
                    )
                    target_log_probs, target_entropy = self._component_statistics(
                        target_logits,
                        conditional_target_masks,
                        selected_targets,
                        FACTOR_TARGET_COUNT,
                    )
                    log_probs = verb_log_probs + argument_log_probs + target_log_probs
                    values = tf.squeeze(values, axis=1)
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
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    value_loss = tf.reduce_mean(tf.square(returns - values))
                    entropy = verb_entropy + argument_entropy + target_entropy
                    total_loss = (
                        policy_loss + self.config.value_coef * value_loss - entropy_coef * entropy
                    )

                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                loss_rows.append(
                    {
                        "policy_loss": float(policy_loss.numpy()),
                        "value_loss": float(value_loss.numpy()),
                        "entropy": float(entropy.numpy()),
                        "total_loss": float(total_loss.numpy()),
                    }
                )

        return {
            key: float(np.mean([loss_row[key] for loss_row in loss_rows]))
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
        selected_log_probs = tf.reduce_sum(one_hot * log_softmax, axis=1)
        probabilities = tf.nn.softmax(masked_logits)
        entropy = -tf.reduce_mean(tf.reduce_sum(probabilities * log_softmax, axis=1))
        return selected_log_probs, entropy

    def _next_values(
        self,
        observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, float]:
        if not observations:
            return {}
        agent_ids = tuple(observations)
        flat_obs = flatten_observations(
            observations,
            agent_ids,
            self.observation_encoder,
        )
        _verb, _argument, _target, values = self.model(flat_obs, training=False)
        values_np = np.squeeze(values.numpy(), axis=1)
        return {agent_id: float(values_np[index]) for index, agent_id in enumerate(agent_ids)}

    def _checkpoint_metadata(self, update: int) -> dict[str, object]:
        metadata = super()._checkpoint_metadata(update)
        metadata.update(
            {
                "algorithm": "factorized_shared_policy_ppo",
                "model_type": "factorized_feed_forward",
                "verb_count": FACTOR_VERB_COUNT,
                "argument_count": FACTOR_ARGUMENT_COUNT,
                "target_count": FACTOR_TARGET_COUNT,
                "factorization": "P(verb)*P(argument|verb)*P(target|verb,argument)",
            }
        )
        return metadata
