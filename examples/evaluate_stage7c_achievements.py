"""Re-score Stage 7C checkpoints with the frozen achievement spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from voyager.training.checkpoints import load_policy_checkpoint
from voyager.training.civilization_achievements import (
    FeedForwardAchievementPolicy,
    InferenceMode,
    LegalRandomAchievementPolicy,
    RecurrentAchievementPolicy,
    calibration_gate,
    evaluate_achievement_policy,
    summarize_achievement_results,
)
from voyager.training.environments import CIVILIZATION_V2_TRAINING_ENVIRONMENT
from voyager.training.obs import CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER

DEFAULT_FEED_FORWARD_CHECKPOINT = Path(
    "results/stage7c/ppo_probe_v3_250k_seed0/checkpoints/best"
)
INFERENCE_MODES: tuple[InferenceMode, ...] = (
    "deterministic",
    "seeded_stochastic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feed-forward-checkpoint",
        type=Path,
        default=DEFAULT_FEED_FORWARD_CHECKPOINT,
    )
    parser.add_argument("--recurrent-checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage7c/achievement_rescore_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feed_forward_model, feed_forward_metadata = load_policy_checkpoint(
        args.feed_forward_checkpoint
    )
    _validate_checkpoint(feed_forward_metadata, expected_model_type="feed_forward")

    print(f"Scoring legal random on {len(seeds)} fixed episodes...")
    random_results = evaluate_achievement_policy(
        policy=LegalRandomAchievementPolicy(),
        seeds=seeds,
        inference_mode="seeded_stochastic",
    )
    random_summary = summarize_achievement_results(random_results)

    feed_forward: dict[str, dict[str, object]] = {}
    for inference_mode in INFERENCE_MODES:
        print(f"Scoring feed-forward PPO ({inference_mode})...")
        results = evaluate_achievement_policy(
            policy=FeedForwardAchievementPolicy(
                model=feed_forward_model,
                inference_mode=inference_mode,
            ),
            seeds=seeds,
            inference_mode=inference_mode,
        )
        feed_forward[inference_mode] = summarize_achievement_results(results)

    recurrent: dict[str, dict[str, object]] | None = None
    recurrent_metadata: dict[str, object] | None = None
    if args.recurrent_checkpoint is not None:
        recurrent_model, recurrent_metadata = load_policy_checkpoint(
            args.recurrent_checkpoint
        )
        _validate_checkpoint(recurrent_metadata, expected_model_type="recurrent_gru")
        hidden_size = _positive_int(
            recurrent_metadata.get("recurrent_hidden_size"),
            "recurrent_hidden_size",
        )
        recurrent = {}
        for inference_mode in INFERENCE_MODES:
            print(f"Scoring recurrent PPO ({inference_mode})...")
            results = evaluate_achievement_policy(
                policy=RecurrentAchievementPolicy(
                    model=recurrent_model,
                    hidden_size=hidden_size,
                    inference_mode=inference_mode,
                ),
                seeds=seeds,
                inference_mode=inference_mode,
            )
            recurrent[inference_mode] = summarize_achievement_results(results)

    primary_feed_forward = feed_forward["seeded_stochastic"]
    primary_recurrent = (
        None if recurrent is None else recurrent["seeded_stochastic"]
    )
    gate = calibration_gate(
        random_summary=random_summary,
        feed_forward_summary=primary_feed_forward,
        recurrent_summary=primary_recurrent,
    )
    summary = {
        "contract": "stage7c_achievement_rescore_v1",
        "environment": CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        "episode_horizon": 600,
        "primary_inference": "seeded_stochastic",
        "deterministic_inference_role": "coordination-collapse diagnostic only",
        "seeds": seeds,
        "random": random_summary,
        "feed_forward_ppo": feed_forward,
        "feed_forward_checkpoint": str(args.feed_forward_checkpoint.resolve()),
        "feed_forward_metadata": feed_forward_metadata,
        "recurrent_ppo": recurrent,
        "recurrent_checkpoint": (
            None
            if args.recurrent_checkpoint is None
            else str(args.recurrent_checkpoint.resolve())
        ),
        "recurrent_metadata": recurrent_metadata,
        "calibration": gate,
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "random.json", random_summary)
    _write_json(args.output_dir / "feed_forward.json", feed_forward)
    if recurrent is not None:
        _write_json(args.output_dir / "recurrent.json", recurrent)

    print("Achievement scores:")
    print(f"  legal random:      {_number(random_summary['achievement_score']):.3f}")
    print(
        "  feed-forward PPO:  "
        f"{_number(primary_feed_forward['achievement_score']):.3f}"
    )
    if primary_recurrent is not None:
        print(
            "  recurrent PPO:     "
            f"{_number(primary_recurrent['achievement_score']):.3f}"
        )
    print(
        "Learned baseline exceeds random: "
        + (
            "YES"
            if gate["at_least_one_learned_baseline_exceeds_random"]
            else "NO"
        )
    )
    print(f"Next action: {gate['next_action']}")
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0


def _validate_checkpoint(
    metadata: dict[str, object],
    *,
    expected_model_type: str,
) -> None:
    if metadata.get("environment_id") != CIVILIZATION_V2_TRAINING_ENVIRONMENT:
        raise ValueError("Checkpoint was not trained on VoyagerCivilization-v2.")
    if metadata.get("observation_encoder") != (
        CIVILIZATION_V3_NAVIGATION_OBSERVATION_ENCODER
    ):
        raise ValueError("Checkpoint does not use the 604-value v3 actor observation.")
    model_type = str(metadata.get("model_type", "feed_forward"))
    if model_type != expected_model_type:
        raise ValueError(
            f"Expected model_type={expected_model_type!r}, found {model_type!r}."
        )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Expected a numeric score.")
    return float(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
