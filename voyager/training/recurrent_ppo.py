"""Shared recurrent PPO for the complete Stage 7C Civilization environment."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from voyager.envs.island import ISLAND_REWARD_VERSION
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION
from voyager.training.advantages import compute_gae
from voyager.training.checkpoints import save_policy_checkpoint
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    ISLAND_V1_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.island_reward import (
    ISLAND_TRAINING_REWARD_V2,
    ISLAND_TRAINING_REWARD_V3,
    ISLAND_TRAINING_REWARD_V4,
)
from voyager.training.masking import stack_action_masks
from voyager.training.model import build_recurrent_actor_critic, require_tensorflow
from voyager.training.obs import flat_observation_size, flatten_observations
from voyager.training.ppo import PPOUpdateStats


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    """Configuration for one shared GRU policy on the fixed 600-tick island."""

    total_steps: int = 250_000
    rollout_steps: int = 128
    seed: int = 0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.005
    value_coef: float = 0.5
    train_epochs: int = 4
    sequence_length: int = 32
    sequence_minibatch_size: int = 16
    encoder_sizes: tuple[int, ...] = (128,)
    recurrent_hidden_size: int = 128
    max_gradient_norm: float = 0.5
    checkpoint_dir: str | None = "checkpoints/stage7c_recurrent"
    checkpoint_every: int = 0
    use_action_mask: bool = True
    environment_id: str = CIVILIZATION_V2_TRAINING_ENVIRONMENT
    reward_contract: str = CIVILIZATION_PROBE_REWARD_CONTRACT
    num_agents: int = 10
    map_size: int = 48
    max_steps: int = 600
    procedural: bool = True

    def validate(self) -> None:
        positive_ints = {
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "train_epochs": self.train_epochs,
            "sequence_length": self.sequence_length,
            "sequence_minibatch_size": self.sequence_minibatch_size,
            "recurrent_hidden_size": self.recurrent_hidden_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.environment_id == CIVILIZATION_V2_TRAINING_ENVIRONMENT:
            if self.reward_contract not in {
                CIVILIZATION_PROBE_REWARD_CONTRACT,
                CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
            }:
                raise ValueError("Recurrent Civilization PPO requires a frozen v3/v4 reward.")
            if (self.num_agents, self.map_size, self.max_steps) != (10, 48, 600):
                raise ValueError("Recurrent Civilization PPO uses the unchanged 10-agent island.")
        elif self.environment_id == ISLAND_V1_TRAINING_ENVIRONMENT:
            if self.reward_contract not in {
                ISLAND_REWARD_VERSION,
                ISLAND_TRAINING_REWARD_V2,
                ISLAND_TRAINING_REWARD_V3,
                ISLAND_TRAINING_REWARD_V4,
            }:
                raise ValueError("Recurrent island PPO requires a versioned island reward.")
            if (self.num_agents, self.map_size, self.max_steps) != (2, 48, 1_200):
                raise ValueError("Recurrent island PPO requires the frozen 2-agent contract.")
        else:
            raise ValueError(f"Unsupported recurrent environment: {self.environment_id!r}.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive.")
        if self.entropy_coef_start < self.entropy_coef_end:
            raise ValueError("entropy_coef_start must be >= entropy_coef_end.")
        if self.entropy_coef_end < 0.0:
            raise ValueError("entropy coefficients must be non-negative.")
        if self.value_coef < 0.0:
            raise ValueError("value_coef must be non-negative.")
        if not self.encoder_sizes or any(size <= 0 for size in self.encoder_sizes):
            raise ValueError("encoder_sizes must contain positive values.")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive.")

    def entropy_coefficient(self, agent_steps: int) -> float:
        progress = min(1.0, max(0.0, agent_steps / self.total_steps))
        return self.entropy_coef_start + progress * (
            self.entropy_coef_end - self.entropy_coef_start
        )


@dataclass(frozen=True, slots=True)
class RecurrentRolloutBatch:
    """Padded, episode-safe truncated sequences for recurrent PPO."""

    observations: np.ndarray
    initial_states: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray

    @property
    def agent_steps(self) -> int:
        return int(np.sum(self.valid))


@dataclass(slots=True)
class _StepRecord:
    observation: np.ndarray
    initial_state: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    next_value: float
    agent_id: str
    episode: int
    advantage: float = 0.0
    return_value: float = 0.0


class RecurrentPPOTrainer:
    """Train a shared recurrent policy while preserving per-agent memory streams."""

    def __init__(self, config: RecurrentPPOConfig) -> None:
        config.validate()
        self.config = config
        self.tf = require_tensorflow()
        self.rng = np.random.default_rng(config.seed)
        training_environment = make_training_environment(
            environment_id=config.environment_id,
            reward_contract=config.reward_contract,
            num_agents=config.num_agents,
            map_size=config.map_size,
            max_steps=config.max_steps,
            reward_mode=(
                "dense" if config.environment_id == ISLAND_V1_TRAINING_ENVIRONMENT else "none"
            ),
            disabled_reward_components=(),
            mask_role_observation=False,
            procedural=config.procedural,
        )
        self.env = training_environment.env
        self.observation_encoder = training_environment.observation_encoder
        self.contract_versions = training_environment.versions
        self.observations, self.infos = self.env.reset(seed=config.seed)
        self.reset_count = 1
        self.episode = 0
        first_agent = self.env.possible_agents[0]
        action_space = self.env.action_space(first_agent)
        if not hasattr(action_space, "n"):
            raise ValueError("Recurrent PPO requires a discrete flattened action space.")
        self.action_count = int(action_space.n)
        self.input_dim = flat_observation_size(
            self.env.observation_space(first_agent),
            self.observation_encoder,
        )
        self.model = build_recurrent_actor_critic(
            input_dim=self.input_dim,
            action_count=self.action_count,
            encoder_sizes=config.encoder_sizes,
            recurrent_hidden_size=config.recurrent_hidden_size,
            seed=config.seed,
        )
        self.optimizer = self.tf.keras.optimizers.Adam(config.learning_rate)
        self.recurrent_states = self._zero_states()
        self.agent_steps = 0
        self.world_steps = 0
        self.timing_seconds: dict[str, float] = {
            "observation_encoding": 0.0,
            "action_masks": 0.0,
            "actor_inference": 0.0,
            "environment_step": 0.0,
            "next_value_inference": 0.0,
            "batch_and_gae": 0.0,
            "learner_update": 0.0,
        }

    def train(
        self,
        on_update: Callable[[PPOUpdateStats], None] | None = None,
    ) -> list[PPOUpdateStats]:
        stats: list[PPOUpdateStats] = []
        update = 0
        training_started = time.perf_counter()
        while self.agent_steps < self.config.total_steps:
            update += 1
            rollout_started = time.perf_counter()
            batch = self.collect_rollout(max_agent_steps=self.config.total_steps - self.agent_steps)
            rollout_seconds = time.perf_counter() - rollout_started
            entropy_coef = self.config.entropy_coefficient(self.agent_steps)
            update_started = time.perf_counter()
            losses = self.update_policy(batch, entropy_coef=entropy_coef)
            update_seconds = time.perf_counter() - update_started
            self.timing_seconds["learner_update"] += update_seconds
            self.agent_steps += batch.agent_steps
            checkpoint_path = self._maybe_save_checkpoint(update)
            update_stats = PPOUpdateStats(
                update=update,
                agent_steps=self.agent_steps,
                mean_reward=float(np.mean(batch.rewards[batch.valid])),
                mean_return=float(np.mean(batch.returns[batch.valid])),
                policy_loss=losses["policy_loss"],
                value_loss=losses["value_loss"],
                entropy=losses["entropy"],
                entropy_coef=entropy_coef,
                total_loss=losses["total_loss"],
                rollout_seconds=rollout_seconds,
                update_seconds=update_seconds,
                agent_steps_per_second=self.agent_steps
                / max(time.perf_counter() - training_started, 1e-9),
                checkpoint_path=checkpoint_path,
            )
            stats.append(update_stats)
            if on_update is not None:
                on_update(update_stats)
        self._save_checkpoint(update)
        return stats

    def collect_rollout(self, max_agent_steps: int | None = None) -> RecurrentRolloutBatch:
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
            logits, values, final_states = self.model(
                [flat_observations[:, None, :], initial_states],
                training=False,
            )
            step_logits = logits[:, 0, :]
            step_values = np.asarray(values, dtype=np.float32)[:, 0, 0]
            masked_logits = self._mask_logits(step_logits, action_masks)
            actions = self._sample_actions(masked_logits)
            old_log_probs = np.asarray(
                self._selected_log_probs(masked_logits, actions),
                dtype=np.float32,
            )
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
                    np.zeros((self.config.recurrent_hidden_size,), dtype=np.float32)
                    if done
                    else final_states_array[index]
                )
            self.observations = next_observations
            self.infos = step_infos

        if not records:
            raise RuntimeError("Recurrent PPO collected an empty rollout.")
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
        tf = self.tf
        sequence_count = int(batch.observations.shape[0])
        indices = np.arange(sequence_count)
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(0, sequence_count, self.config.sequence_minibatch_size):
                selected = indices[start : start + self.config.sequence_minibatch_size]
                observations = tf.convert_to_tensor(batch.observations[selected], dtype=tf.float32)
                initial_states = tf.convert_to_tensor(
                    batch.initial_states[selected], dtype=tf.float32
                )
                action_masks = tf.convert_to_tensor(batch.action_masks[selected], dtype=tf.bool)
                actions = tf.convert_to_tensor(batch.actions[selected], dtype=tf.int32)
                old_log_probs = tf.convert_to_tensor(
                    batch.old_log_probs[selected], dtype=tf.float32
                )
                advantages = tf.convert_to_tensor(batch.advantages[selected], dtype=tf.float32)
                returns = tf.convert_to_tensor(batch.returns[selected], dtype=tf.float32)
                valid = tf.convert_to_tensor(batch.valid[selected], dtype=tf.float32)
                denominator = tf.maximum(tf.reduce_sum(valid), 1.0)

                with tf.GradientTape() as tape:
                    logits, values, _final_states = self.model(
                        [observations, initial_states], training=True
                    )
                    masked_logits = self._mask_logits(logits, action_masks)
                    values = tf.squeeze(values, axis=2)
                    log_probs = self._selected_log_probs(masked_logits, actions)
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
                    entropy_by_step = self._entropy_by_step(masked_logits)
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
                        gradients, self.model.trainable_variables, strict=True
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

    def save_named_checkpoint(self, name: str, update: int) -> str:
        if self.config.checkpoint_dir is None:
            raise ValueError("Named checkpoints require checkpoint_dir.")
        return str(
            save_policy_checkpoint(
                self.model,
                Path(self.config.checkpoint_dir) / name,
                self._checkpoint_metadata(update),
            )
        )

    def timing_report(self) -> dict[str, object]:
        measured_seconds = float(sum(self.timing_seconds.values()))
        agent_steps_per_second = self.agent_steps / max(measured_seconds, 1e-9)
        environment_detail = getattr(self.env, "performance_seconds", {})
        if not isinstance(environment_detail, dict):
            environment_detail = {}
        return {
            "agent_steps": self.agent_steps,
            "world_steps": self.world_steps,
            "measured_seconds": measured_seconds,
            "agent_steps_per_second": agent_steps_per_second,
            "projected_five_million_hours": 5_000_000 / max(agent_steps_per_second, 1e-9) / 3_600,
            "components_seconds": dict(self.timing_seconds),
            "environment_detail_seconds": dict(environment_detail),
        }

    def _assign_advantages(self, records: list[_StepRecord]) -> None:
        by_stream: dict[tuple[str, int], list[_StepRecord]] = {}
        for record in records:
            by_stream.setdefault((record.agent_id, record.episode), []).append(record)
        all_advantages: list[float] = []
        for stream in by_stream.values():
            advantages, returns = compute_gae(
                rewards=np.asarray([row.reward for row in stream], dtype=np.float32),
                dones=np.asarray([row.done for row in stream], dtype=np.float32),
                values=np.asarray([row.value for row in stream], dtype=np.float32),
                next_values=np.asarray([row.next_value for row in stream], dtype=np.float32),
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            for row, advantage, return_value in zip(stream, advantages, returns, strict=True):
                row.advantage = float(advantage)
                row.return_value = float(return_value)
                all_advantages.append(float(advantage))
        mean = float(np.mean(all_advantages))
        std = float(np.std(all_advantages))
        denominator = std + 1e-8 if std >= 1e-8 else 1.0
        for record in records:
            record.advantage = (record.advantage - mean) / denominator

    def _sequence_batch(self, records: list[_StepRecord]) -> RecurrentRolloutBatch:
        by_stream: dict[tuple[str, int], list[_StepRecord]] = {}
        for record in records:
            by_stream.setdefault((record.agent_id, record.episode), []).append(record)
        chunks: list[list[_StepRecord]] = []
        for key in sorted(by_stream):
            stream = by_stream[key]
            for start in range(0, len(stream), self.config.sequence_length):
                chunks.append(stream[start : start + self.config.sequence_length])

        count = len(chunks)
        length = self.config.sequence_length
        observations = np.zeros((count, length, self.input_dim), dtype=np.float32)
        initial_states = np.zeros((count, self.config.recurrent_hidden_size), dtype=np.float32)
        action_masks = np.zeros((count, length, self.action_count), dtype=np.bool_)
        action_masks[:, :, 0] = True
        actions = np.zeros((count, length), dtype=np.int32)
        old_log_probs = np.zeros((count, length), dtype=np.float32)
        advantages = np.zeros((count, length), dtype=np.float32)
        returns = np.zeros((count, length), dtype=np.float32)
        rewards = np.zeros((count, length), dtype=np.float32)
        valid = np.zeros((count, length), dtype=np.bool_)
        for sequence_index, chunk in enumerate(chunks):
            initial_states[sequence_index] = chunk[0].initial_state
            for step_index, record in enumerate(chunk):
                observations[sequence_index, step_index] = record.observation
                action_masks[sequence_index, step_index] = record.action_mask
                actions[sequence_index, step_index] = record.action
                old_log_probs[sequence_index, step_index] = record.old_log_prob
                advantages[sequence_index, step_index] = record.advantage
                returns[sequence_index, step_index] = record.return_value
                rewards[sequence_index, step_index] = record.reward
                valid[sequence_index, step_index] = True
        return RecurrentRolloutBatch(
            observations=observations,
            initial_states=initial_states,
            action_masks=action_masks,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
            valid=valid,
        )

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
        initial_states = np.stack([state_by_agent[agent_id] for agent_id in agent_ids], axis=0)
        _logits, values, _discarded_states = self.model(
            [flat[:, None, :], initial_states], training=False
        )
        values_array = np.asarray(values, dtype=np.float32)[:, 0, 0]
        return {agent_id: float(values_array[index]) for index, agent_id in enumerate(agent_ids)}

    def _sample_actions(self, logits: Any) -> np.ndarray:
        sampled = self.tf.random.categorical(logits, num_samples=1)
        return np.squeeze(sampled.numpy(), axis=1).astype(np.int32)

    def _mask_logits(self, logits: Any, action_masks: Any) -> Any:
        masks = self.tf.cast(action_masks, self.tf.bool)
        invalid = self.tf.fill(self.tf.shape(logits), self.tf.cast(-1e9, logits.dtype))
        return self.tf.where(masks, logits, invalid)

    def _selected_log_probs(self, logits: Any, actions: Any) -> Any:
        one_hot = self.tf.one_hot(actions, depth=self.action_count)
        return self.tf.reduce_sum(one_hot * self.tf.nn.log_softmax(logits), axis=-1)

    def _entropy_by_step(self, logits: Any) -> Any:
        probabilities = self.tf.nn.softmax(logits)
        log_probabilities = self.tf.nn.log_softmax(logits)
        return -self.tf.reduce_sum(probabilities * log_probabilities, axis=-1)

    def _zero_states(self) -> dict[str, np.ndarray]:
        return {
            agent_id: np.zeros((self.config.recurrent_hidden_size,), dtype=np.float32)
            for agent_id in self.env.possible_agents
        }

    def _reset_env(self) -> None:
        seed = self.config.seed + self.reset_count
        self.observations, self.infos = self.env.reset(seed=seed)
        self.reset_count += 1
        self.episode += 1
        self.recurrent_states = self._zero_states()

    def _maybe_save_checkpoint(self, update: int) -> str | None:
        if (
            self.config.checkpoint_dir is None
            or self.config.checkpoint_every <= 0
            or update % self.config.checkpoint_every != 0
        ):
            return None
        return self._save_checkpoint(update)

    def _save_checkpoint(self, update: int) -> str | None:
        if self.config.checkpoint_dir is None:
            return None
        root = Path(self.config.checkpoint_dir)
        metadata = self._checkpoint_metadata(update)
        latest = save_policy_checkpoint(self.model, root / "latest", metadata)
        if self.config.checkpoint_every > 0 and update % self.config.checkpoint_every == 0:
            save_policy_checkpoint(self.model, root / f"update_{update:05d}", metadata)
        return str(latest)

    def _checkpoint_metadata(self, update: int) -> dict[str, object]:
        return {
            "stage": 7,
            "algorithm": "shared_policy_recurrent_ppo",
            "model_type": "recurrent_gru",
            "update": update,
            "agent_steps": self.agent_steps,
            "input_dim": self.input_dim,
            "action_count": self.action_count,
            "observation_encoder": self.observation_encoder,
            "environment_id": self.config.environment_id,
            "reward_contract": self.config.reward_contract,
            "encoder_sizes": list(self.config.encoder_sizes),
            "recurrent_hidden_size": self.config.recurrent_hidden_size,
            "num_agents": self.config.num_agents,
            "map_size": self.config.map_size,
            "max_steps": self.config.max_steps,
            "rollout_steps": self.config.rollout_steps,
            "sequence_length": self.config.sequence_length,
            "sequence_minibatch_size": self.config.sequence_minibatch_size,
            "world_steps": self.world_steps,
            "environment_resets": self.reset_count,
            "training_seed": self.config.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_ratio": self.config.clip_ratio,
            "value_coef": self.config.value_coef,
            "train_epochs": self.config.train_epochs,
            "action_masking": self.config.use_action_mask,
            "entropy_coef_start": self.config.entropy_coef_start,
            "entropy_coef_end": self.config.entropy_coef_end,
            "max_gradient_norm": self.config.max_gradient_norm,
            **self.contract_versions,
            "achievement_benchmark_version": (
                ISLAND_ACHIEVEMENT_VERSION
                if self.config.environment_id == ISLAND_V1_TRAINING_ENVIRONMENT
                else "civilization_achievement_benchmark_v1"
            ),
            "timing_seconds": dict(self.timing_seconds),
            "python_version": platform.python_version(),
            "tensorflow_version": self.tf.__version__,
            "numpy_version": version("numpy"),
            "gymnasium_version": version("gymnasium"),
            "pettingzoo_version": version("pettingzoo"),
            "git_revision": _git_revision(),
        }


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"
