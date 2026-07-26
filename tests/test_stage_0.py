"""Stage 0 smoke tests."""

import gymnasium as gym
import pytest

import voyager


def test_package_version() -> None:
    assert voyager.__version__ == "0.0.0"


def test_placeholder_env_reports_stage_one_status() -> None:
    env = gym.make("VoyagerSingleAgent-v0")

    with pytest.raises(NotImplementedError, match="planned for Stage 1"):
        env.reset(seed=0)
