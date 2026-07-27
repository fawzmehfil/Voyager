"""Stage 3 tests for camp economy, storms, and metrics."""

import numpy as np

from voyager.envs import VoyagerParallelEnv
from voyager.sim.constants import Action, Resource


def test_deposit_and_withdraw_update_camp_and_inventory() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.inventory["food"] = 1

    _obs, _rewards, _terms, _truncs, infos = env.step({"agent_0": Action.DEPOSIT_FOOD})

    assert infos["agent_0"]["event"] == "deposit_food"
    assert agent.inventory["food"] == 0
    assert state.camp.stockpile["food"] == 1
    assert "first_deposit" in env.metrics()["achievements"]

    _obs, _rewards, _terms, _truncs, infos = env.step({"agent_0": Action.WITHDRAW_FOOD})

    assert infos["agent_0"]["event"] == "withdraw_food"
    assert agent.inventory["food"] == 1
    assert state.camp.stockpile["food"] == 0
    assert "first_food_withdrawal" in env.metrics()["achievements"]


def test_build_shelter_consumes_material_and_increases_progress() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.role = "builder"
    agent.inventory["wood"] = 1

    _obs, rewards, _terms, _truncs, infos = env.step({"agent_0": Action.BUILD_SHELTER})

    assert rewards["agent_0"] > 0
    assert infos["agent_0"]["event"] == "build_shelter_wood"
    assert agent.inventory["wood"] == 0
    assert state.camp.shelter_progress > 0.0
    assert env.metrics()["total_build_actions"] == 1


def test_storm_damage_hits_unsheltered_agent() -> None:
    env = VoyagerParallelEnv(
        num_agents=1,
        storm_start_step=1,
        storm_interval=10,
        storm_duration=5,
        storm_damage=5.0,
    )
    env.reset(seed=0)

    _obs, _rewards, _terms, _truncs, infos = env.step({"agent_0": Action.NOOP})

    assert infos["agent_0"]["storm_active"] is True
    assert infos["agent_0"]["health"] < 100.0


def test_shelter_reduces_storm_damage() -> None:
    exposed = VoyagerParallelEnv(
        num_agents=1,
        storm_start_step=1,
        storm_interval=10,
        storm_duration=5,
        storm_damage=5.0,
    )
    sheltered = VoyagerParallelEnv(
        num_agents=1,
        storm_start_step=1,
        storm_interval=10,
        storm_duration=5,
        storm_damage=5.0,
    )
    exposed.reset(seed=0)
    sheltered.reset(seed=0)
    sheltered.world.state.camp.shelter_progress = 0.8

    exposed.step({"agent_0": Action.NOOP})
    sheltered.step({"agent_0": Action.NOOP})

    assert sheltered.world.state.agents["agent_0"].health > exposed.world.state.agents["agent_0"].health


def test_food_regeneration_adds_food_resources() -> None:
    env = VoyagerParallelEnv(num_agents=1, food_regen_interval=1, food_spawn_rate=0.10)
    env.reset(seed=0)
    state = env.world.state
    food_mask = state.resource_ids == Resource.FOOD
    state.resource_ids[food_mask] = Resource.NONE
    state.resource_quantities[food_mask] = 0

    env.step({"agent_0": Action.NOOP})

    assert int(np.sum(state.resource_ids == Resource.FOOD)) > 0


def test_shelter_achievements_are_recorded() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    env.reset(seed=0)
    state = env.world.state
    agent = state.agents["agent_0"]
    agent.x = state.camp.x
    agent.y = state.camp.y
    agent.role = "builder"
    agent.inventory["wood"] = 20

    for _ in range(17):
        env.step({"agent_0": Action.BUILD_SHELTER})

    achievements = set(env.metrics()["achievements"])
    assert "shelter_25_percent" in achievements
    assert "shelter_50_percent" in achievements
    assert "shelter_complete" in achievements


def test_all_alive_achievement_at_100_steps() -> None:
    env = VoyagerParallelEnv(num_agents=2, storm_start_step=1000)
    env.reset(seed=0)

    for _ in range(100):
        env.step({agent_id: Action.NOOP for agent_id in env.agents})

    assert "all_active_agents_alive_100" in env.metrics()["achievements"]


def test_metrics_returns_stable_json_like_data() -> None:
    env = VoyagerParallelEnv(num_agents=2)
    env.reset(seed=0)

    metrics = env.metrics()

    assert metrics["step"] == 0
    assert metrics["active_agents"] == 2
    assert metrics["deaths"] == 0
    assert "camp" in metrics
    assert "achievements" in metrics
    assert metrics["total_deposits"] == 0
    assert metrics["total_withdrawals"] == 0
    assert metrics["total_build_actions"] == 0
