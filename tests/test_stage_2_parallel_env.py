"""Stage 2 tests for Voyager's PettingZoo parallel environment."""

from collections.abc import Iterator

import gymnasium as gym
import numpy as np

from voyager.envs import VoyagerParallelEnv
from voyager.sim.constants import Action, Terrain


def test_survival_env_id_creates_parallel_env() -> None:
    env = gym.make("VoyagerSurvival-v0", disable_env_checker=True)

    assert isinstance(env.unwrapped, VoyagerParallelEnv)
    env.close()


def test_parallel_reset_returns_all_agents() -> None:
    env = VoyagerParallelEnv(num_agents=5)
    observations, infos = env.reset(seed=0)

    assert env.agents == [f"agent_{index}" for index in range(5)]
    assert set(observations) == set(env.agents)
    assert set(infos) == set(env.agents)
    for agent_id, observation in observations.items():
        assert env.observation_space(agent_id).contains(observation)


def test_parallel_reset_is_seed_deterministic() -> None:
    env_a = VoyagerParallelEnv(num_agents=6)
    env_b = VoyagerParallelEnv(num_agents=6)

    obs_a, infos_a = env_a.reset(seed=42)
    obs_b, infos_b = env_b.reset(seed=42)

    assert np.array_equal(env_a.world.state.terrain, env_b.world.state.terrain)
    assert np.array_equal(env_a.world.state.resource_ids, env_b.world.state.resource_ids)
    assert np.array_equal(obs_a["agent_0"]["local_view"], obs_b["agent_0"]["local_view"])
    assert {
        agent_id: info["position"] for agent_id, info in infos_a.items()
    } == {
        agent_id: info["position"] for agent_id, info in infos_b.items()
    }


def test_parallel_reset_spawns_no_overlaps() -> None:
    env = VoyagerParallelEnv(num_agents=10)
    _observations, infos = env.reset(seed=0)

    positions = [info["position"] for info in infos.values()]
    assert len(positions) == len(set(positions))


def test_parallel_different_seeds_can_change_world() -> None:
    env_a = VoyagerParallelEnv(num_agents=3)
    env_b = VoyagerParallelEnv(num_agents=3)

    env_a.reset(seed=1)
    env_b.reset(seed=2)

    terrain_differs = not np.array_equal(env_a.world.state.terrain, env_b.world.state.terrain)
    resources_differ = not np.array_equal(
        env_a.world.state.resource_ids,
        env_b.world.state.resource_ids,
    )
    assert terrain_differs or resources_differ


def test_parallel_random_rollout_runs_100_steps() -> None:
    env = VoyagerParallelEnv(num_agents=4)
    env.reset(seed=0)

    for _ in range(100):
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        _observations, _rewards, _terminations, _truncations, _infos = env.step(actions)
        if not env.agents:
            break

    assert env.world.state.step_count >= 100 or not env.agents


def test_parallel_movement_blocks_out_of_bounds() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    agent = env.world.state.agents["agent_0"]
    agent.x = 0
    agent.y = 0

    _obs, rewards, _terms, _truncs, infos = env.step({"agent_0": Action.MOVE_LEFT})

    assert rewards["agent_0"] < 0
    assert infos["agent_0"]["event"] == "invalid_out_of_bounds"
    assert infos["agent_0"]["position"] == (0, 0)


def test_parallel_movement_blocks_water() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    x, y, action = next(_land_tile_with_water_neighbor(env.world.state.terrain))
    agent = env.world.state.agents["agent_0"]
    agent.x = x
    agent.y = y

    _obs, rewards, _terms, _truncs, infos = env.step({"agent_0": action})

    assert rewards["agent_0"] < 0
    assert infos["agent_0"]["event"] == "invalid_water_blocked"
    assert infos["agent_0"]["position"] == (x, y)


def test_parallel_movement_blocks_occupied_tile() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    env.reset(seed=0)
    env.world.state.agents["agent_0"].x = 10
    env.world.state.agents["agent_0"].y = 10
    env.world.state.agents["agent_1"].x = 11
    env.world.state.agents["agent_1"].y = 10

    _obs, rewards, _terms, _truncs, infos = env.step({"agent_0": Action.MOVE_RIGHT})

    assert rewards["agent_0"] < 0
    assert infos["agent_0"]["event"] == "invalid_occupied"
    assert infos["agent_0"]["position"] == (10, 10)


def test_parallel_death_removes_agent_from_active_agents() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    env.reset(seed=0)
    dying_agent = env.world.state.agents["agent_0"]
    dying_agent.health = 0.1
    dying_agent.hunger = 100.0

    _obs, _rewards, terminations, _truncs, infos = env.step({"agent_0": Action.NOOP})

    assert terminations["agent_0"] is True
    assert infos["agent_0"]["event"] == "death"
    assert "agent_0" not in env.agents
    assert "agent_1" in env.agents


def test_parallel_ansi_render_returns_map() -> None:
    env = VoyagerParallelEnv(num_agents=3, render_mode="ansi")
    env.reset(seed=0)

    rendered = env.render()

    assert isinstance(rendered, str)
    assert "active=3" in rendered
    assert "camp=" in rendered


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
