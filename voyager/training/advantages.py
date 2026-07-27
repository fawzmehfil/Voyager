"""Advantage estimation for PPO."""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and value targets."""

    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    if not (rewards.shape == dones.shape == values.shape == next_values.shape):
        raise ValueError("rewards, dones, values, and next_values must have matching shapes.")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_values[index] * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)
