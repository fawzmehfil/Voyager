"""Stage 4 tests for baseline policies and evaluation."""

from copy import deepcopy

from voyager.envs import VoyagerParallelEnv
from voyager.policies import CooperativePolicy, GreedySurvivalPolicy, RandomPolicy
from voyager.policies.evaluation import evaluate_baselines
from voyager.sim.constants import Action


def test_all_policies_return_valid_actions() -> None:
    env = VoyagerParallelEnv(num_agents=3)
    observations, infos = env.reset(seed=0)
    policies = [
        RandomPolicy(seed=0),
        GreedySurvivalPolicy(),
        CooperativePolicy(),
    ]

    for policy in policies:
        for agent_id in env.agents:
            action = int(policy.act(agent_id, observations[agent_id], infos[agent_id]))
            assert env.action_space(agent_id).contains(action)


def test_greedy_policy_eats_when_hungry_with_food() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    observations, infos = env.reset(seed=0)
    info = deepcopy(infos["agent_0"])
    info["hunger"] = 80.0
    info["inventory"] = {"food": 1, "wood": 0, "stone": 0}

    action = GreedySurvivalPolicy().act("agent_0", observations["agent_0"], info)

    assert action == Action.EAT


def test_cooperative_policy_deposits_surplus_food_at_camp() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    observations, infos = env.reset(seed=0)
    info = deepcopy(infos["agent_0"])
    info["position"] = info["camp"]["position"]
    info["hunger"] = 20.0
    info["inventory"] = {"food": 2, "wood": 0, "stone": 0}

    action = CooperativePolicy().act("agent_0", observations["agent_0"], info)

    assert action == Action.DEPOSIT_FOOD


def test_cooperative_policy_builds_shelter_as_builder_with_material() -> None:
    env = VoyagerParallelEnv(num_agents=1)
    observations, infos = env.reset(seed=0)
    info = deepcopy(infos["agent_0"])
    info["position"] = info["camp"]["position"]
    info["role"] = "builder"
    info["inventory"] = {"food": 0, "wood": 1, "stone": 0}

    action = CooperativePolicy().act("agent_0", observations["agent_0"], info)

    assert action == Action.BUILD_SHELTER


def test_baseline_evaluation_runs_all_policies() -> None:
    results = evaluate_baselines(episodes=1, max_steps=25, num_agents=3)

    assert {result.policy for result in results} == {"random", "greedy", "cooperative"}
    assert len(results) == 3


def test_baseline_evaluation_is_deterministic_for_fixed_seeds() -> None:
    first = evaluate_baselines(episodes=1, max_steps=25, num_agents=3)
    second = evaluate_baselines(episodes=1, max_steps=25, num_agents=3)

    assert first == second
