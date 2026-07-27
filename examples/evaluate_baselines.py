"""Evaluate Voyager's Stage 4 baseline policies."""

from __future__ import annotations

import argparse

from voyager.policies.evaluation import evaluate_baselines, print_summary
from voyager.policies.ppo_policy import TensorFlowPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint directory, usually checkpoints/stage5/latest.",
    )
    parser.add_argument(
        "--ppo-stochastic",
        action="store_true",
        help="Sample PPO actions instead of taking the highest-logit action.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_policies = []
    if args.ppo_checkpoint is not None:
        extra_policies.append(
            (
                "ppo",
                lambda _seed: TensorFlowPPOPolicy(
                    args.ppo_checkpoint,
                    deterministic=not args.ppo_stochastic,
                ),
            )
        )
    results = evaluate_baselines(
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        extra_policies=extra_policies,
    )
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
