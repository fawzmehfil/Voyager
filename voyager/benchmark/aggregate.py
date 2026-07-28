"""Deterministic aggregate statistics for Stage 5.6 benchmark records."""

from __future__ import annotations

from collections.abc import Callable
from statistics import stdev

import numpy as np

from voyager.benchmark.schema import BenchmarkManifest, EpisodeRecord
from voyager.sim.achievements import ACHIEVEMENT_IDS
from voyager.versions import BENCHMARK_SCHEMA_VERSION

METRICS: dict[str, Callable[[EpisodeRecord], float]] = {
    "dense_return": lambda row: row.dense_return,
    "achievement_count": lambda row: float(len(row.achievements)),
    "survivors": lambda row: float(row.survivors),
    "deaths": lambda row: float(row.deaths),
    "survival_rate": lambda row: row.survival_rate,
    "shelter_progress": lambda row: row.shelter_progress,
    "camp_food": lambda row: float(row.camp_stockpile["food"]),
    "selected_invalid_actions": lambda row: float(row.selected_invalid_actions),
    "raw_invalid_actions": lambda row: float(row.raw_invalid_actions),
    "invalid_probability_mass": lambda row: row.invalid_probability_mass,
}


def civilization_score(success_rates: list[float] | np.ndarray) -> float:
    """Return the Stage 5.6 shifted geometric-mean score on a 0-100 scale."""

    rates = np.asarray(success_rates, dtype=np.float64)
    if rates.shape != (len(ACHIEVEMENT_IDS),):
        raise ValueError(f"Expected {len(ACHIEVEMENT_IDS)} achievement rates.")
    if np.any(rates < 0.0) or np.any(rates > 1.0):
        raise ValueError("Achievement success rates must be in [0, 1].")
    return float(100.0 * np.expm1(np.mean(np.log1p(rates))))


def aggregate_records(
    records: list[EpisodeRecord],
    manifest: BenchmarkManifest,
) -> dict[str, object]:
    """Aggregate all episode records using deterministic bootstrap intervals."""

    grouped = {
        policy.id: sorted(
            [record for record in records if record.policy_id == policy.id],
            key=lambda record: record.seed,
        )
        for policy in manifest.policies
    }
    policy_summaries: dict[str, object] = {}
    for index, policy in enumerate(manifest.policies):
        rows = grouped[policy.id]
        if len(rows) != len(manifest.seed_suite.seeds):
            raise ValueError(
                f"Policy {policy.id!r} has {len(rows)} episodes; "
                f"expected {len(manifest.seed_suite.seeds)}."
            )
        policy_summaries[policy.id] = _policy_summary(
            rows,
            samples=manifest.bootstrap.samples,
            confidence=manifest.bootstrap.confidence,
            bootstrap_seed=manifest.bootstrap.seed + index,
        )

    official_ppo = [
        policy
        for policy in manifest.policies
        if policy.kind == "ppo" and policy.official
    ]
    ppo_family = None
    if official_ppo:
        ppo_family = _ppo_family_summary(
            [grouped[policy.id] for policy in official_ppo],
            samples=manifest.bootstrap.samples,
            confidence=manifest.bootstrap.confidence,
            bootstrap_seed=manifest.bootstrap.seed,
        )

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "seed_suite": manifest.seed_suite.model_dump(mode="json"),
        "bootstrap": manifest.bootstrap.model_dump(mode="json"),
        "episode_count": len(records),
        "policies": policy_summaries,
        "ppo_official_family": ppo_family,
    }


def _policy_summary(
    rows: list[EpisodeRecord],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(bootstrap_seed)
    count = len(rows)
    indexes = rng.integers(0, count, size=(samples, count))
    metrics: dict[str, object] = {}
    for name, getter in METRICS.items():
        values = np.asarray([getter(row) for row in rows], dtype=np.float64)
        replicates = values[indexes].mean(axis=1)
        metrics[name] = _stat_row(values, replicates, confidence)

    indicators = _achievement_indicators(rows)
    rates = indicators.mean(axis=0)
    rate_replicates = indicators[indexes].mean(axis=1)
    achievement_rows = {
        achievement_id: {
            "success_rate": float(rates[index]),
            "ci_low": _quantile(rate_replicates[:, index], confidence, lower=True),
            "ci_high": _quantile(rate_replicates[:, index], confidence, lower=False),
        }
        for index, achievement_id in enumerate(ACHIEVEMENT_IDS)
    }
    score_replicates = np.asarray(
        [civilization_score(rate_row) for rate_row in rate_replicates],
        dtype=np.float64,
    )
    score = {
        "mean": civilization_score(rates),
        "ci_low": _quantile(score_replicates, confidence, lower=True),
        "ci_high": _quantile(score_replicates, confidence, lower=False),
    }
    return {
        "episodes": count,
        "metrics": metrics,
        "achievements": achievement_rows,
        "civilization_score": score,
    }


def _ppo_family_summary(
    groups: list[list[EpisodeRecord]],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    if any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("PPO policy seeds must have matching evaluation episode counts.")
    rng = np.random.default_rng(bootstrap_seed)
    policy_count = len(groups)
    episode_count = len(groups[0])
    policy_indexes = rng.integers(0, policy_count, size=(samples, policy_count))
    episode_indexes = rng.integers(
        0,
        episode_count,
        size=(samples, policy_count, episode_count),
    )
    sample_policies = policy_indexes[:, :, None]

    metrics: dict[str, object] = {}
    for name, getter in METRICS.items():
        values = np.asarray(
            [[getter(row) for row in group] for group in groups],
            dtype=np.float64,
        )
        replicates = values[sample_policies, episode_indexes].mean(axis=(1, 2))
        checkpoint_means = values.mean(axis=1)
        metrics[name] = {
            **_stat_row(values.reshape(-1), replicates, confidence),
            "policy_seed_std": _sample_std(checkpoint_means),
        }

    indicators = np.asarray(
        [_achievement_indicators(group) for group in groups],
        dtype=np.float64,
    )
    rates = indicators.mean(axis=(0, 1))
    rate_replicates = np.empty((samples, len(ACHIEVEMENT_IDS)), dtype=np.float64)
    policy_seed_rates = indicators.mean(axis=1)
    for achievement_index in range(len(ACHIEVEMENT_IDS)):
        values = indicators[:, :, achievement_index]
        selected = values[sample_policies, episode_indexes]
        rate_replicates[:, achievement_index] = selected.mean(axis=(1, 2))
    achievement_rows = {
        achievement_id: {
            "success_rate": float(rates[index]),
            "ci_low": _quantile(rate_replicates[:, index], confidence, lower=True),
            "ci_high": _quantile(rate_replicates[:, index], confidence, lower=False),
            "policy_seed_std": _sample_std(policy_seed_rates[:, index]),
        }
        for index, achievement_id in enumerate(ACHIEVEMENT_IDS)
    }
    checkpoint_scores = np.asarray(
        [civilization_score(rate_row) for rate_row in policy_seed_rates],
        dtype=np.float64,
    )
    score_replicates = np.asarray(
        [civilization_score(rate_row) for rate_row in rate_replicates],
        dtype=np.float64,
    )
    return {
        "policy_seeds": policy_count,
        "episodes_per_policy_seed": episode_count,
        "metrics": metrics,
        "achievements": achievement_rows,
        "civilization_score": {
            "mean": civilization_score(rates),
            "ci_low": _quantile(score_replicates, confidence, lower=True),
            "ci_high": _quantile(score_replicates, confidence, lower=False),
            "policy_seed_std": _sample_std(checkpoint_scores),
        },
    }


def _achievement_indicators(rows: list[EpisodeRecord]) -> np.ndarray:
    return np.asarray(
        [
            [float(achievement_id in row.achievements) for achievement_id in ACHIEVEMENT_IDS]
            for row in rows
        ],
        dtype=np.float64,
    )


def _stat_row(
    values: np.ndarray,
    replicates: np.ndarray,
    confidence: float,
) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": _sample_std(values),
        "ci_low": _quantile(replicates, confidence, lower=True),
        "ci_high": _quantile(replicates, confidence, lower=False),
    }


def _sample_std(values: np.ndarray) -> float:
    return float(stdev(float(value) for value in values)) if values.size > 1 else 0.0


def _quantile(values: np.ndarray, confidence: float, *, lower: bool) -> float:
    alpha = (1.0 - confidence) / 2.0
    quantile = alpha if lower else 1.0 - alpha
    return float(np.quantile(values, quantile))
