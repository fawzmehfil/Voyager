"""Evaluation helpers for Voyager baseline policies."""

from dataclasses import dataclass
from numbers import Real
from statistics import mean

from voyager.envs import VoyagerParallelEnv
from voyager.policies.base import Policy
from voyager.policies.heuristics import CooperativePolicy, GreedySurvivalPolicy, RandomPolicy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Aggregated result for one baseline episode."""

    policy: str
    seed: int
    total_reward: float
    survivors: int
    deaths: int
    shelter_progress: float
    camp_food: int
    achievements: int


def run_episode(
    policy_name: str,
    policy: Policy,
    seed: int,
    max_steps: int,
    num_agents: int,
) -> EpisodeResult:
    """Run one baseline episode and return aggregate metrics."""

    env = VoyagerParallelEnv(num_agents=num_agents, max_steps=max_steps)
    observations, infos = env.reset(seed=seed)
    total_reward = 0.0

    while env.agents:
        actions = {
            agent_id: int(policy.act(agent_id, observations[agent_id], infos[agent_id]))
            for agent_id in env.agents
        }
        observations, rewards, _terminations, _truncations, step_infos = env.step(actions)
        total_reward += sum(rewards.values())
        infos.update(step_infos)

    metrics = env.metrics()
    camp = metrics["camp"]
    if not isinstance(camp, dict):
        raise TypeError("metrics['camp'] must be a dictionary")
    stockpile = camp["stockpile"]
    if not isinstance(stockpile, dict):
        raise TypeError("metrics['camp']['stockpile'] must be a dictionary")
    achievements = metrics["achievements"]
    if not isinstance(achievements, list):
        raise TypeError("metrics['achievements'] must be a list")

    return EpisodeResult(
        policy=policy_name,
        seed=seed,
        total_reward=total_reward,
        survivors=_int_metric(metrics["active_agents"]),
        deaths=_int_metric(metrics["deaths"]),
        shelter_progress=float(camp["shelter_progress"]),
        camp_food=int(stockpile["food"]),
        achievements=len(achievements),
    )


def evaluate_baselines(
    episodes: int = 3,
    max_steps: int = 300,
    num_agents: int = 10,
) -> list[EpisodeResult]:
    """Run all Stage 4 baselines over deterministic seeds."""

    results: list[EpisodeResult] = []
    for policy_name, policy_factory in (
        ("random", lambda seed: RandomPolicy(seed=seed)),
        ("greedy", lambda _seed: GreedySurvivalPolicy()),
        ("cooperative", lambda _seed: CooperativePolicy()),
    ):
        for seed in range(episodes):
            policy = policy_factory(seed)
            results.append(
                run_episode(
                    policy_name=policy_name,
                    policy=policy,
                    seed=seed,
                    max_steps=max_steps,
                    num_agents=num_agents,
                )
            )
    return results


def print_summary(results: list[EpisodeResult]) -> None:
    """Print an aggregate table by policy."""

    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.policy, []).append(result)

    header = (
        f"{'policy':<12} {'reward':>10} {'survivors':>10} {'deaths':>8} "
        f"{'shelter':>9} {'camp_food':>10} {'achievements':>12}"
    )
    print(header)
    print("-" * len(header))
    for policy_name in sorted(grouped):
        policy_results = grouped[policy_name]
        print(
            f"{policy_name:<12} "
            f"{mean(result.total_reward for result in policy_results):>10.2f} "
            f"{mean(result.survivors for result in policy_results):>10.2f} "
            f"{mean(result.deaths for result in policy_results):>8.2f} "
            f"{mean(result.shelter_progress for result in policy_results):>9.2f} "
            f"{mean(result.camp_food for result in policy_results):>10.2f} "
            f"{mean(result.achievements for result in policy_results):>12.2f}"
        )


def _int_metric(value: object) -> int:
    if not isinstance(value, Real):
        raise TypeError(f"Expected numeric metric value, got {type(value).__name__}")
    return int(float(value))
