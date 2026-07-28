"""Shared-policy PPO trainer for Voyager's multi-agent environment."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from voyager.envs import VoyagerParallelEnv
from voyager.sim.constants import ACTION_COUNT
from voyager.sim.rewards import DENSE_REWARD_COMPONENTS, REWARD_MODES, RewardMode
from voyager.training.advantages import compute_gae
from voyager.training.checkpoints import save_policy_checkpoint
from voyager.training.masking import stack_action_masks
from voyager.training.model import build_actor_critic, require_tensorflow
from voyager.training.obs import flat_observation_size, flatten_observations
from voyager.versions import (
    ACHIEVEMENT_VERSION,
    ACTION_VERSION,
    DENSE_REWARD_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    SCENARIO_VERSION,
)


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
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5
    train_epochs: int = 4
    minibatch_size: int = 256
    hidden_sizes: tuple[int, ...] = (128, 128)
    checkpoint_dir: str | None = "checkpoints/stage5"
    checkpoint_every: int = 10
    reward_mode: RewardMode = "dense"
    use_action_mask: bool = True
    disabled_reward_components: tuple[str, ...] = ()
    mask_role_observation: bool = False

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
        if self.entropy_coef_start < 0.0 or self.entropy_coef_end < 0.0:
            raise ValueError("entropy coefficients must be non-negative.")
        if self.entropy_coef_start < self.entropy_coef_end:
            raise ValueError("entropy_coef_start must be >= entropy_coef_end.")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode!r}.")
        unknown_components = set(self.disabled_reward_components) - DENSE_REWARD_COMPONENTS
        if unknown_components:
            names = ", ".join(sorted(unknown_components))
            raise ValueError(f"Unknown disabled reward components: {names}")

    def entropy_coefficient(self, agent_steps: int) -> float:
        """Return the linearly decayed entropy coefficient."""

        progress = min(1.0, max(0.0, agent_steps / self.total_steps))
        return self.entropy_coef_start + progress * (
            self.entropy_coef_end - self.entropy_coef_start
        )


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One PPO rollout flattened across all agents."""

    observations: np.ndarray
    action_masks: np.ndarray
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
    entropy_coef: float
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
            reward_mode=config.reward_mode,
            disabled_reward_components=config.disabled_reward_components,
            mask_role_observation=config.mask_role_observation,
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
        self.world_steps = 0

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
            entropy_coef = (
                self.config.entropy_coef_end
                if self.agent_steps + int(batch.actions.shape[0]) >= self.config.total_steps
                else self.config.entropy_coefficient(self.agent_steps)
            )
            losses = self.update_policy(batch, entropy_coef=entropy_coef)
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
                entropy_coef=entropy_coef,
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
            flat_obs = flatten_observations(self.observations, agent_ids)
            action_masks = stack_action_masks(self.infos, agent_ids)
            if not self.config.use_action_mask:
                action_masks = np.ones_like(action_masks, dtype=np.bool_)
            logits, values = self.model(flat_obs, training=False)
            masked_logits = self._mask_logits(logits, action_masks)
            actions = self._sample_actions(masked_logits)
            log_probs = self._selected_log_probs(masked_logits, actions).numpy()
            values_np = np.squeeze(values.numpy(), axis=1).astype(np.float32)

            action_map = {
                agent_id: int(actions[index])
                for index, agent_id in enumerate(agent_ids)
            }
            next_observations, rewards, terminations, truncations, step_infos = self.env.step(action_map)
            self.world_steps += 1
            next_values_by_agent = self._next_values(next_observations)

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
        """Apply PPO minibatch updates and return mean losses."""

        tf = self.tf
        if entropy_coef is None:
            entropy_coef = self.config.entropy_coefficient(self.agent_steps)
        sample_count = int(batch.actions.shape[0])
        indices = np.arange(sample_count)
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(0, sample_count, self.config.minibatch_size):
                minibatch = indices[start : start + self.config.minibatch_size]
                obs = tf.convert_to_tensor(batch.observations[minibatch], dtype=tf.float32)
                action_masks = tf.convert_to_tensor(
                    batch.action_masks[minibatch],
                    dtype=tf.bool,
                )
                actions = tf.convert_to_tensor(batch.actions[minibatch], dtype=tf.int32)
                old_log_probs = tf.convert_to_tensor(batch.old_log_probs[minibatch], dtype=tf.float32)
                advantages = tf.convert_to_tensor(batch.advantages[minibatch], dtype=tf.float32)
                returns = tf.convert_to_tensor(batch.returns[minibatch], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    logits, values = self.model(obs, training=True)
                    masked_logits = self._mask_logits(logits, action_masks)
                    values = tf.squeeze(values, axis=1)
                    log_probs = self._selected_log_probs(masked_logits, actions)
                    ratio = tf.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = tf.clip_by_value(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    ) * advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    value_loss = tf.reduce_mean(tf.square(returns - values))
                    entropy = self._entropy(masked_logits)
                    total_loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - entropy_coef * entropy
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

    def _mask_logits(self, logits: Any, action_masks: Any) -> Any:
        masks = self.tf.cast(action_masks, self.tf.bool)
        invalid_logits = self.tf.fill(self.tf.shape(logits), self.tf.cast(-1e9, logits.dtype))
        return self.tf.where(masks, logits, invalid_logits)

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
            "rollout_steps": self.config.rollout_steps,
            "world_steps": self.world_steps,
            "environment_resets": self.reset_count,
            "training_seed": self.config.seed,
            "training_seed_last": self.config.seed + self.reset_count - 1,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_ratio": self.config.clip_ratio,
            "value_coef": self.config.value_coef,
            "train_epochs": self.config.train_epochs,
            "minibatch_size": self.config.minibatch_size,
            "action_masking": self.config.use_action_mask,
            "entropy_coef_start": self.config.entropy_coef_start,
            "entropy_coef_end": self.config.entropy_coef_end,
            "reward_mode": self.config.reward_mode,
            "disabled_reward_components": list(self.config.disabled_reward_components),
            "role_observation": not self.config.mask_role_observation,
            "environment_version": ENVIRONMENT_VERSION,
            "reward_version": DENSE_REWARD_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_VERSION,
            "achievement_version": ACHIEVEMENT_VERSION,
            "scenario_version": SCENARIO_VERSION,
            "training_revision": "5.6",
            "python_version": platform.python_version(),
            "tensorflow_version": self.tf.__version__,
            "numpy_version": version("numpy"),
            "gymnasium_version": version("gymnasium"),
            "pettingzoo_version": version("pettingzoo"),
            "git_revision": _git_revision(),
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
