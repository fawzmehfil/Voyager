"""Train a shared-policy TensorFlow PPO agent on Voyager."""

from __future__ import annotations

import argparse
import sys

from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/stage5")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run training without saving model checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        num_agents=args.num_agents,
        map_size=args.map_size,
        max_steps=args.max_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
        hidden_sizes=tuple(args.hidden_sizes),
        checkpoint_dir=None if args.no_checkpoint else args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    try:
        trainer = PPOTrainer(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        "training shared-policy PPO "
        f"agent_steps={config.total_steps} agents={config.num_agents} "
        f"rollout_steps={config.rollout_steps} max_steps={config.max_steps}"
    )
    stats = trainer.train(on_update=_print_update)
    if config.checkpoint_dir is not None:
        print(f"latest checkpoint: {config.checkpoint_dir}/latest")
    print(f"completed updates: {len(stats)}")
    return 0


def _print_update(stats: PPOUpdateStats) -> None:
    checkpoint = f" checkpoint={stats.checkpoint_path}" if stats.checkpoint_path else ""
    print(
        f"update={stats.update:04d} "
        f"agent_steps={stats.agent_steps:07d} "
        f"reward={stats.mean_reward:+.4f} "
        f"return={stats.mean_return:+.4f} "
        f"policy_loss={stats.policy_loss:+.4f} "
        f"value_loss={stats.value_loss:.4f} "
        f"entropy={stats.entropy:.4f}"
        f"{checkpoint}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
