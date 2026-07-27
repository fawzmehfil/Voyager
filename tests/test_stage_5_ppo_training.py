"""Stage 5 tests for PPO training utilities."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from voyager.envs import VoyagerParallelEnv
from voyager.policies.base import Info, Observation, Policy
from voyager.policies.evaluation import evaluate_baselines
from voyager.sim.constants import Action
from voyager.training.advantages import compute_gae
from voyager.training.model import build_actor_critic, require_tensorflow
from voyager.training.obs import flat_observation_size, flatten_observation, flatten_observations
from voyager.training.ppo import PPOConfig


class NoopPolicy(Policy):
    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        _ = agent_id, observation, info
        return int(Action.NOOP)


def test_flatten_observation_matches_declared_space_size() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    observations, _infos = env.reset(seed=0)
    agent_id = env.agents[0]

    flattened = flatten_observation(observations[agent_id])

    assert flattened.dtype == np.float32
    assert flattened.shape == (flat_observation_size(env.observation_space(agent_id)),)
    assert np.all(flattened >= 0.0)
    assert np.all(flattened <= 1.0)


def test_flatten_observations_preserves_agent_order() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    observations, _infos = env.reset(seed=0)
    agent_ids = tuple(reversed(env.agents))

    stacked = flatten_observations(observations, agent_ids)

    assert stacked.shape[0] == 2
    np.testing.assert_array_equal(stacked[0], flatten_observation(observations[agent_ids[0]]))


def test_compute_gae_respects_done_boundaries() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    values = np.array([0.5, 0.25, 0.1], dtype=np.float32)
    next_values = np.array([0.25, 0.0, 0.2], dtype=np.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_values=next_values,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert advantages[1] == pytest.approx(0.75)
    assert returns[1] == pytest.approx(1.0)


def test_ppo_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        PPOConfig(total_steps=0).validate()
    with pytest.raises(ValueError, match="hidden_sizes"):
        PPOConfig(hidden_sizes=()).validate()


def test_evaluator_accepts_extra_policy_factories() -> None:
    results = evaluate_baselines(
        episodes=1,
        max_steps=5,
        num_agents=2,
        extra_policies=(("noop", lambda _seed: NoopPolicy()),),
    )

    assert "noop" in {result.policy for result in results}


def test_require_tensorflow_reports_clear_missing_dependency() -> None:
    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("TensorFlow is installed in this environment.")
    with pytest.raises(RuntimeError, match="TensorFlow is required"):
        require_tensorflow()


def test_actor_critic_shapes_when_tensorflow_is_available() -> None:
    pytest.importorskip("tensorflow")
    model = build_actor_critic(input_dim=12, action_count=5, hidden_sizes=(8,), seed=0)

    logits, value = model(np.zeros((3, 12), dtype=np.float32), training=False)

    assert tuple(logits.shape) == (3, 5)
    assert tuple(value.shape) == (3, 1)
