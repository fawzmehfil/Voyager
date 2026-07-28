"""Stage 5.6 tests for achievements, reward APIs, and benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import voyager
from voyager.benchmark.aggregate import civilization_score
from voyager.benchmark.runner import load_manifest, run_benchmark
from voyager.envs import VoyagerParallelEnv
from voyager.policies.ppo_policy import TensorFlowPPOPolicy
from voyager.sim.achievements import ACHIEVEMENT_IDS
from voyager.sim.constants import ACTION_COUNT, Action
from voyager.training.ppo import PPOConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEV_MANIFEST = REPOSITORY_ROOT / "benchmarks/manifests/stage5_6_dev.json"


def test_public_reward_environment_ids_are_registered() -> None:
    _ = voyager
    expected_modes = {
        "VoyagerReward-v0": "dense",
        "VoyagerAchievement-v0": "achievement",
        "VoyagerNoReward-v0": "none",
    }
    for env_id, reward_mode in expected_modes.items():
        env = gym.make(env_id, num_agents=1, max_steps=2)
        assert env.unwrapped.reward_mode == reward_mode
        env.close()


def test_reward_modes_preserve_dynamics_and_dense_diagnostics() -> None:
    dense = VoyagerParallelEnv(num_agents=1, max_steps=2, storm_start_step=10_000)
    sparse = VoyagerParallelEnv(
        num_agents=1,
        max_steps=2,
        storm_start_step=10_000,
        reward_mode="achievement",
    )
    zero = VoyagerParallelEnv(
        num_agents=1,
        max_steps=2,
        storm_start_step=10_000,
        reward_mode="none",
    )
    for env in (dense, sparse, zero):
        env.reset(seed=0)
        state = env.world.state
        agent = state.agents["agent_0"]
        agent.x = state.camp.x
        agent.y = state.camp.y
        agent.inventory["food"] = 1

    dense_step = dense.step({"agent_0": Action.DEPOSIT_FOOD})
    sparse_step = sparse.step({"agent_0": Action.DEPOSIT_FOOD})
    zero_step = zero.step({"agent_0": Action.DEPOSIT_FOOD})

    assert dense.metrics() == sparse.metrics() == zero.metrics()
    assert dense_step[1]["agent_0"] == pytest.approx(
        sum(dense_step[4]["agent_0"]["dense_reward_components"].values())
    )
    assert sparse_step[1]["agent_0"] == 1.0
    assert sparse_step[4]["agent_0"]["reward_components"] == {"achievement": 1.0}
    assert zero_step[1]["agent_0"] == 0.0
    assert zero_step[4]["agent_0"]["dense_reward_components"]


def test_dense_default_and_explicit_mode_are_regression_equivalent() -> None:
    default = VoyagerParallelEnv(num_agents=2, max_steps=10, storm_start_step=10_000)
    explicit = VoyagerParallelEnv(
        num_agents=2,
        max_steps=10,
        storm_start_step=10_000,
        reward_mode="dense",
    )
    default.reset(seed=42)
    explicit.reset(seed=42)
    for _ in range(10):
        actions = {agent_id: int(Action.NOOP) for agent_id in default.agents}
        default_step = default.step(actions)
        explicit_step = explicit.step(actions)
        assert default_step[1] == explicit_step[1]
        for agent_id in default_step[4]:
            default_info = default_step[4][agent_id]
            explicit_info = explicit_step[4][agent_id]
            np.testing.assert_array_equal(
                default_info["action_mask"],
                explicit_info["action_mask"],
            )
            assert {
                key: value
                for key, value in default_info.items()
                if key != "action_mask"
            } == {
                key: value
                for key, value in explicit_info.items()
                if key != "action_mask"
            }
    assert default.metrics() == explicit.metrics()


def test_resource_and_shared_transfer_achievements() -> None:
    env = VoyagerParallelEnv(num_agents=2, max_steps=10, storm_start_step=10_000)
    env.reset(seed=0)
    state = env.world.state
    depositor = state.agents["agent_0"]
    recipient = state.agents["agent_1"]
    depositor.x = recipient.x = state.camp.x
    depositor.y = recipient.y = state.camp.y
    depositor.inventory["food"] = 1

    env.step({"agent_0": Action.DEPOSIT_FOOD, "agent_1": Action.NOOP})
    env.step({"agent_0": Action.NOOP, "agent_1": Action.WITHDRAW_FOOD})
    env.step({"agent_0": Action.NOOP, "agent_1": Action.EAT})

    achievements = set(env.metrics()["achievements"])
    assert "first_deposit" in achievements
    assert "first_food_withdrawal" in achievements
    assert "shared_food_transfer" in achievements
    assert env.metrics()["achievement_steps"]["shared_food_transfer"] == 3


def test_food_security_roles_and_terminal_achievements() -> None:
    env = VoyagerParallelEnv(num_agents=3, max_steps=103, storm_start_step=10_000)
    env.reset(seed=0)
    state = env.world.state
    for index, agent in enumerate(state.agents.values()):
        agent.x = state.camp.x
        agent.y = state.camp.y
        agent.inventory["wood"] = 1
        env.step(
            {
                agent_id: (
                    Action.DEPOSIT_WOOD
                    if agent_id == f"agent_{index}"
                    else Action.NOOP
                )
                for agent_id in env.agents
            }
        )
    state.camp.stockpile["food"] = 20
    while env.agents:
        env.step({agent_id: Action.NOOP for agent_id in env.agents})

    achievements = set(env.metrics()["achievements"])
    assert "all_roles_contributed" in achievements
    assert "camp_food_buffer_10" in achievements
    assert "camp_food_buffer_20" in achievements
    assert "food_security_100_steps" in achievements
    assert "no_deaths_run" in achievements


def test_role_observation_and_reward_component_ablations() -> None:
    env = VoyagerParallelEnv(
        num_agents=1,
        mask_role_observation=True,
        disabled_reward_components=("alive",),
    )
    observations, _infos = env.reset(seed=0)
    assert np.all(observations["agent_0"]["role"] == 0.0)
    _obs, rewards, _terms, _truncs, infos = env.step({"agent_0": Action.NOOP})
    assert "alive" not in infos["agent_0"]["reward_components"]
    assert "alive" in infos["agent_0"]["dense_reward_components"]
    assert rewards["agent_0"] == pytest.approx(
        sum(infos["agent_0"]["reward_components"].values())
    )

    config = PPOConfig(
        use_action_mask=False,
        disabled_reward_components=("alive",),
        mask_role_observation=True,
    )
    config.validate()
    with pytest.raises(ValueError, match="Unknown disabled"):
        PPOConfig(disabled_reward_components=("unknown",)).validate()


def test_civilization_score_formula_and_validation() -> None:
    assert civilization_score(np.zeros(len(ACHIEVEMENT_IDS))) == 0.0
    assert civilization_score(np.ones(len(ACHIEVEMENT_IDS))) == pytest.approx(100.0)
    with pytest.raises(ValueError, match="Expected"):
        civilization_score([1.0])


def test_frozen_policy_exposes_batched_mask_diagnostics() -> None:
    pytest.importorskip("tensorflow")
    env = VoyagerParallelEnv(num_agents=2)
    observations, infos = env.reset(seed=5_000_000)
    policy = TensorFlowPPOPolicy(
        REPOSITORY_ROOT / "benchmarks/checkpoints/stage5_5_seed0",
        deterministic=True,
    )

    decisions = policy.decide_many(tuple(env.agents), observations, infos)

    assert set(decisions) == set(env.agents)
    for decision in decisions.values():
        assert 0 <= decision.action < ACTION_COUNT
        assert 0 <= decision.raw_action < ACTION_COUNT
        assert 0.0 <= decision.invalid_probability_mass <= 1.0


def test_manifest_seed_overlap_and_checkpoint_hash_are_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    payload["seed_suite"] = {"id": "overlap", "seeds": [0]}
    payload["policies"] = [payload["policies"][3]]
    overlap_path = tmp_path / "overlap.json"
    overlap_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        run_benchmark(overlap_path, tmp_path / "overlap-output")

    payload["seed_suite"] = {"id": "held-out", "seeds": [5_000_000]}
    payload["policies"][0]["checkpoint_sha256"] = "0" * 64
    hash_path = tmp_path / "hash.json"
    hash_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_benchmark(hash_path, tmp_path / "hash-output")


def test_tiny_benchmark_writes_valid_resumable_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    payload["benchmark_id"] = "stage5_6_test"
    payload["seed_suite"] = {"id": "test-seed", "seeds": [5_000_000]}
    payload["policies"] = [payload["policies"][0], payload["policies"][3]]
    payload["bootstrap"]["samples"] = 100
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"

    first = run_benchmark(manifest_path, output)
    second = run_benchmark(manifest_path, output, resume=True)

    assert first == second
    assert first["episode_count"] == 2
    assert len((output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    for filename in (
        "manifest.json",
        "episodes.jsonl",
        "summary.json",
        "achievements.csv",
        "policies.csv",
    ):
        assert (output / filename).is_file()
    output_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["status"] == "complete"
    assert output_manifest["completed_episodes"] == 2
    assert load_manifest(manifest_path).benchmark_id == "stage5_6_test"
