"""Stage 1 tests for the Voyager single-agent environment."""

from collections.abc import Iterator

import gymnasium as gym
import numpy as np
import pytest

import voyager
from voyager.sim.constants import Action, Resource, Terrain


def test_package_version() -> None:
    assert voyager.__version__ == "0.0.0"


def test_single_agent_reset_returns_valid_observation() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs)
    assert info["event"] == "reset"
    env.close()


def test_same_seed_produces_identical_initial_world() -> None:
    env_a = gym.make("VoyagerSingleAgent-v0")
    env_b = gym.make("VoyagerSingleAgent-v0")

    obs_a, _info_a = env_a.reset(seed=7)
    obs_b, _info_b = env_b.reset(seed=7)

    assert np.array_equal(obs_a["local_view"], obs_b["local_view"])
    assert np.array_equal(env_a.unwrapped.world.state.terrain, env_b.unwrapped.world.state.terrain)
    assert np.array_equal(
        env_a.unwrapped.world.state.resource_ids,
        env_b.unwrapped.world.state.resource_ids,
    )

    env_a.close()
    env_b.close()


def test_different_seeds_can_produce_different_worlds() -> None:
    env_a = gym.make("VoyagerSingleAgent-v0")
    env_b = gym.make("VoyagerSingleAgent-v0")

    env_a.reset(seed=1)
    env_b.reset(seed=2)

    terrain_differs = not np.array_equal(
        env_a.unwrapped.world.state.terrain,
        env_b.unwrapped.world.state.terrain,
    )
    resources_differ = not np.array_equal(
        env_a.unwrapped.world.state.resource_ids,
        env_b.unwrapped.world.state.resource_ids,
    )
    assert terrain_differs or resources_differ

    env_a.close()
    env_b.close()


def test_random_rollout_runs_at_least_100_steps() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)

    steps = 0
    done = False
    while not done and steps < 100:
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1

    assert steps == 100
    env.close()


def test_movement_respects_boundaries() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    state.agent.x = 0
    state.agent.y = 0

    _obs, reward, _terminated, _truncated, info = env.step(Action.MOVE_LEFT)

    assert reward < 0.0
    assert info["event"] == "invalid_out_of_bounds"
    assert info["position"] == (0, 0)
    env.close()


def test_movement_respects_water_blocking() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    x, y, action = next(_land_tile_with_water_neighbor(state.terrain))
    state.agent.x = x
    state.agent.y = y

    _obs, reward, _terminated, _truncated, info = env.step(action)

    assert reward < 0.0
    assert info["event"] == "invalid_water_blocked"
    assert info["position"] == (x, y)
    env.close()


def test_gather_changes_inventory_and_resource_state() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    y, x = np.argwhere(state.resource_quantities > 0)[0]
    state.agent.x = int(x)
    state.agent.y = int(y)
    resource = Resource(int(state.resource_ids[y, x]))
    resource_name = {Resource.FOOD: "food", Resource.WOOD: "wood", Resource.STONE: "stone"}[
        resource
    ]
    before_quantity = int(state.resource_quantities[y, x])

    _obs, reward, _terminated, _truncated, info = env.step(Action.GATHER)

    assert reward > 0.0
    assert info["event"] == f"gather_{resource_name}"
    assert info["inventory"][resource_name] == 1
    assert int(state.resource_quantities[y, x]) == before_quantity - 1
    env.close()


def test_eat_reduces_hunger_and_consumes_food() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    state.agent.inventory["food"] = 1
    state.agent.hunger = 60.0

    _obs, reward, _terminated, _truncated, info = env.step(Action.EAT)

    assert reward > 0.0
    assert info["event"] == "eat"
    assert info["inventory"]["food"] == 0
    assert info["hunger"] < 60.0
    env.close()


def test_rest_increases_energy() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    state.agent.energy = 25.0

    _obs, reward, _terminated, _truncated, info = env.step(Action.REST)

    assert reward > 0.0
    assert info["event"] == "rest"
    assert info["energy"] > 25.0
    env.close()


def test_death_sets_terminated() -> None:
    env = gym.make("VoyagerSingleAgent-v0")
    env.reset(seed=0)
    state = env.unwrapped.world.state
    state.agent.health = 0.1
    state.agent.hunger = 100.0

    _obs, reward, terminated, truncated, info = env.step(Action.NOOP)

    assert terminated is True
    assert truncated is False
    assert reward < -9.0
    assert info["event"] == "death"
    env.close()


def test_max_steps_sets_truncated() -> None:
    env = gym.make("VoyagerSingleAgent-v0", max_steps=3)
    env.reset(seed=0)

    terminated = False
    truncated = False
    for _ in range(3):
        _obs, _reward, terminated, truncated, _info = env.step(Action.NOOP)

    assert terminated is False
    assert truncated is True
    env.close()


def test_ansi_render_returns_non_empty_string() -> None:
    env = gym.make("VoyagerSingleAgent-v0", render_mode="ansi")
    env.reset(seed=0)

    rendered = env.render()

    assert isinstance(rendered, str)
    assert "health=" in rendered
    assert "@" in rendered
    env.close()


def test_multi_agent_survival_env_remains_placeholder() -> None:
    env = gym.make("VoyagerSurvival-v0")

    with pytest.raises(NotImplementedError, match="multi-agent stage"):
        env.reset(seed=0)

    env.close()


def _land_tile_with_water_neighbor(terrain: np.ndarray) -> Iterator[tuple[int, int, Action]]:
    directions = (
        (0, -1, Action.MOVE_UP),
        (0, 1, Action.MOVE_DOWN),
        (-1, 0, Action.MOVE_LEFT),
        (1, 0, Action.MOVE_RIGHT),
    )
    height, width = terrain.shape
    for y in range(height):
        for x in range(width):
            if terrain[y, x] == Terrain.WATER:
                continue
            for dx, dy, action in directions:
                target_x = x + dx
                target_y = y + dy
                if (
                    0 <= target_x < width
                    and 0 <= target_y < height
                    and terrain[target_y, target_x] == Terrain.WATER
                ):
                    yield x, y, action
