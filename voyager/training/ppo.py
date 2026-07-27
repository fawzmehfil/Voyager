"""Shared-policy PPO trainer for Voyager's multi-agent environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voyager.envs import VoyagerParallelEnv
from voyager.sim.constants import ACTION_COUNT
from voyager.training.advantages import compute_gae
from voyager.training.checkpoints import save_policy_checkpoint
from voyager.training.model import build_actor_critic, require_tensorflow
from voyager.training.obs import flat_observation_size, flatten_observations


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Configuration for shared-policy Voyager PPO."""

    total_steps: int = 50_000
    rollout_steps: int = 128
    num_agents: int = 10
    map_size: int = 32
    max_steps: int = 300
    seed: int = 0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    train_epochs: int = 4
    minibatch_size: int = 256
    hidden_sizes: tuple[int, ...] = (128, 128)
    checkpoint_dir: str | None = "checkpoints/stage5"
    checkpoint_every: int = 10

    def validate(self) -> None:
        """Validate numeric config values before creating a trainer."""

        positive_ints = {
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "num_agents": self.num_agents,
            "map_size": self.map_size,
            "max_steps": self.max_steps,
            "train_epochs": self.train_epochs,
            "minibatch_size": self.minibatch_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive.")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One PPO rollout flattened across all agents."""

    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    agent_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    """Summary for one PPO update."""

    update: int
    agent_steps: int
    mean_reward: float
    mean_return: float
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    checkpoint_path: str | None


class PPOTrainer:
    """Train a shared actor-critic policy on VoyagerParallelEnv."""

    def __init__(self, config: PPOConfig) -> None:
        config.validate()
        self.config = config
        self.tf = require_tensorflow()
        self.rng = np.random.default_rng(config.seed)
        self.env = VoyagerParallelEnv(
            num_agents=config.num_agents,
            map_size=config.map_size,
            max_steps=config.max_steps,
        )
        self.observations, self.infos = self.env.reset(seed=config.seed)
        self.reset_count = 1
        first_agent = self.env.possible_agents[0]
        self.input_dim = flat_observation_size(self.env.observation_space(first_agent))
        self.model = build_actor_critic(
            input_dim=self.input_dim,
            action_count=ACTION_COUNT,
            hidden_sizes=config.hidden_sizes,
            seed=config.seed,
        )
        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.agent_steps = 0

    def train(
        self,
        on_update: Callable[[PPOUpdateStats], None] | None = None,
    ) -> list[PPOUpdateStats]:
        """Run PPO updates until the configured number of agent steps is reached."""

        stats: list[PPOUpdateStats] = []
        update = 0
        while self.agent_steps < self.config.total_steps:
            update += 1
            batch = self.collect_rollout()
            losses = self.update_policy(batch)
            self.agent_steps += int(batch.actions.shape[0])
            checkpoint_path = self._maybe_save_checkpoint(update)
            update_stats = PPOUpdateStats(
                update=update,
                agent_steps=self.agent_steps,
                mean_reward=float(np.mean(batch.rewards)),
                mean_return=float(np.mean(batch.returns)),
                policy_loss=losses["policy_loss"],
                value_loss=losses["value_loss"],
                entropy=losses["entropy"],
                total_loss=losses["total_loss"],
                checkpoint_path=checkpoint_path,
            )
            stats.append(update_stats)
            if on_update is not None:
                on_update(update_stats)
        self._save_checkpoint(update)
        return stats

    def collect_rollout(self) -> RolloutBatch:
        """Collect one fixed-length rollout across all live agents."""

        obs_rows: list[np.ndarray] = []
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
            flat_obs = flatten_observations(self.observations, agent_ids)
            logits, values = self.model(flat_obs, training=False)
            actions = self._sample_actions(logits)
            log_probs = self._selected_log_probs(logits, actions).numpy()
            values_np = np.squeeze(values.numpy(), axis=1).astype(np.float32)

            action_map = {
                agent_id: int(actions[index])
                for index, agent_id in enumerate(agent_ids)
            }
            next_observations, rewards, terminations, truncations, step_infos = self.env.step(action_map)
            next_values_by_agent = self._next_values(next_observations)

            for index, agent_id in enumerate(agent_ids):
                done = bool(terminations[agent_id] or truncations[agent_id])
                obs_rows.append(flat_obs[index])
                actions_rows.append(int(actions[index]))
                log_prob_rows.append(float(log_probs[index]))
                value_rows.append(float(values_np[index]))
                reward_rows.append(float(rewards[agent_id]))
                done_rows.append(done)
                next_value_rows.append(0.0 if done else next_values_by_agent.get(agent_id, 0.0))
                agent_rows.append(agent_id)

            self.observations = next_observations
            self.infos.update(step_infos)

        observations = np.asarray(obs_rows, dtype=np.float32)
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

        return RolloutBatch(
            observations=observations,
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

    def update_policy(self, batch: RolloutBatch) -> dict[str, float]:
        """Apply PPO minibatch updates and return mean losses."""

        tf = self.tf
        sample_count = int(batch.actions.shape[0])
        indices = np.arange(sample_count)
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(0, sample_count, self.config.minibatch_size):
                minibatch = indices[start : start + self.config.minibatch_size]
                obs = tf.convert_to_tensor(batch.observations[minibatch], dtype=tf.float32)
                actions = tf.convert_to_tensor(batch.actions[minibatch], dtype=tf.int32)
                old_log_probs = tf.convert_to_tensor(batch.old_log_probs[minibatch], dtype=tf.float32)
                advantages = tf.convert_to_tensor(batch.advantages[minibatch], dtype=tf.float32)
                returns = tf.convert_to_tensor(batch.returns[minibatch], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    logits, values = self.model(obs, training=True)
                    values = tf.squeeze(values, axis=1)
                    log_probs = self._selected_log_probs(logits, actions)
                    ratio = tf.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = tf.clip_by_value(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    ) * advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    value_loss = tf.reduce_mean(tf.square(returns - values))
                    entropy = self._entropy(logits)
                    total_loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
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

    def _advantages_by_agent(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        agent_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        for agent_id in sorted(set(agent_ids)):
            indices = np.asarray(
                [index for index, row_agent_id in enumerate(agent_ids) if row_agent_id == agent_id],
                dtype=np.int64,
            )
            agent_advantages, agent_returns = compute_gae(
                rewards=rewards[indices],
                dones=dones[indices],
                values=values[indices],
                next_values=next_values[indices],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            advantages[indices] = agent_advantages
            returns[indices] = agent_returns
        return advantages, returns

    def _sample_actions(self, logits: Any) -> np.ndarray:
        actions = self.tf.random.categorical(logits, num_samples=1)
        return np.squeeze(actions.numpy(), axis=1).astype(np.int32)

    def _selected_log_probs(self, logits: Any, actions: Any) -> Any:
        action_mask = self.tf.one_hot(actions, depth=ACTION_COUNT)
        return self.tf.reduce_sum(action_mask * self.tf.nn.log_softmax(logits), axis=1)

    def _entropy(self, logits: Any) -> Any:
        probabilities = self.tf.nn.softmax(logits)
        log_probabilities = self.tf.nn.log_softmax(logits)
        return -self.tf.reduce_mean(self.tf.reduce_sum(probabilities * log_probabilities, axis=1))

    def _next_values(self, observations: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
        if not observations:
            return {}
        agent_ids = tuple(observations)
        flat_obs = flatten_observations(observations, agent_ids)
        _logits, values = self.model(flat_obs, training=False)
        values_np = np.squeeze(values.numpy(), axis=1)
        return {
            agent_id: float(values_np[index])
            for index, agent_id in enumerate(agent_ids)
        }

    def _reset_env(self) -> None:
        seed = self.config.seed + self.reset_count
        self.observations, self.infos = self.env.reset(seed=seed)
        self.reset_count += 1

    def _maybe_save_checkpoint(self, update: int) -> str | None:
        if self.config.checkpoint_dir is None:
            return None
        if self.config.checkpoint_every <= 0:
            return None
        if update % self.config.checkpoint_every != 0:
            return None
        return self._save_checkpoint(update)

    def _save_checkpoint(self, update: int) -> str | None:
        if self.config.checkpoint_dir is None:
            return None
        root = Path(self.config.checkpoint_dir)
        metadata = {
            "stage": 5,
            "algorithm": "shared_policy_ppo",
            "update": update,
            "agent_steps": self.agent_steps,
            "input_dim": self.input_dim,
            "action_count": ACTION_COUNT,
            "hidden_sizes": list(self.config.hidden_sizes),
            "num_agents": self.config.num_agents,
            "map_size": self.config.map_size,
            "max_steps": self.config.max_steps,
        }
        latest = save_policy_checkpoint(self.model, root / "latest", metadata)
        if self.config.checkpoint_every > 0 and update % self.config.checkpoint_every == 0:
            save_policy_checkpoint(self.model, root / f"update_{update:05d}", metadata)
        return str(latest)


def _normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    if advantages.size == 0:
        return advantages.astype(np.float32)
    std = float(np.std(advantages))
    if std < 1e-8:
        return (advantages - float(np.mean(advantages))).astype(np.float32)
    return ((advantages - float(np.mean(advantages))) / (std + 1e-8)).astype(np.float32)
