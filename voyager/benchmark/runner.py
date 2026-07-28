"""Resumable Stage 5.6 benchmark execution and artifact export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from voyager.benchmark.aggregate import aggregate_records
from voyager.benchmark.schema import BenchmarkManifest, EpisodeRecord, PolicySpec
from voyager.envs import VoyagerParallelEnv
from voyager.policies.base import Policy, PolicyDecision
from voyager.policies.heuristics import CooperativePolicy, GreedySurvivalPolicy, RandomPolicy
from voyager.policies.ppo_policy import TensorFlowPPOPolicy
from voyager.sim.achievements import ACHIEVEMENT_IDS
from voyager.sim.constants import ACTION_COUNT, Action
from voyager.training.masking import action_mask_from_info
from voyager.versions import (
    ACHIEVEMENT_VERSION,
    ACTION_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    DENSE_REWARD_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    SCENARIO_VERSION,
)

ProgressCallback = Callable[[int, int, str, int], None]


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load and validate one benchmark input manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return BenchmarkManifest.model_validate(payload)


def run_benchmark(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Execute or resume a complete versioned benchmark."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    repository_root = _repository_root(manifest_path)
    manifest = load_manifest(manifest_path)
    _validate_manifest(manifest, repository_root)
    fingerprint = _manifest_fingerprint(manifest)
    expected_keys = {
        (policy.id, seed)
        for policy in manifest.policies
        for seed in manifest.seed_suite.seeds
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "manifest.json"
    episodes_path = output_dir / "episodes.jsonl"
    existing_records: list[EpisodeRecord] = []
    if output_manifest_path.exists() or episodes_path.exists():
        if not resume:
            raise FileExistsError(
                f"Benchmark output already exists at {output_dir}; use --resume."
            )
        output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        if output_manifest.get("manifest_fingerprint") != fingerprint:
            raise ValueError("Cannot resume: benchmark manifest fingerprint changed.")
        existing_records = _read_episode_records(episodes_path)
    else:
        output_manifest = _new_output_manifest(manifest, fingerprint, len(expected_keys))
        _write_json_atomic(output_manifest_path, output_manifest)

    completed: dict[tuple[str, int], EpisodeRecord] = {}
    for record in existing_records:
        key = (record.policy_id, record.seed)
        if key not in expected_keys:
            raise ValueError(f"Unexpected resumed episode key: {key!r}")
        if key in completed:
            raise ValueError(f"Duplicate resumed episode key: {key!r}")
        completed[key] = record

    runtimes: dict[str, Policy] = {}
    total = len(expected_keys)
    try:
        with episodes_path.open("a", encoding="utf-8") as episode_file:
            for policy in manifest.policies:
                for seed in manifest.seed_suite.seeds:
                    key = (policy.id, seed)
                    if key in completed:
                        continue
                    runtime = _policy_runtime(
                        policy,
                        seed,
                        repository_root,
                        runtimes,
                    )
                    record = _run_episode(manifest, policy, runtime, seed)
                    episode_file.write(record.model_dump_json() + "\n")
                    episode_file.flush()
                    os.fsync(episode_file.fileno())
                    completed[key] = record
                    output_manifest["completed_episodes"] = len(completed)
                    output_manifest["updated_at"] = _now()
                    _write_json_atomic(output_manifest_path, output_manifest)
                    if on_progress is not None:
                        on_progress(len(completed), total, policy.id, seed)

        records = [completed[key] for key in sorted(completed)]
        if set(completed) != expected_keys:
            raise ValueError("Benchmark stopped before all expected episodes completed.")
        summary = aggregate_records(records, manifest)
        summary_path = output_dir / "summary.json"
        achievements_path = output_dir / "achievements.csv"
        policies_path = output_dir / "policies.csv"
        _write_json_atomic(summary_path, summary)
        _write_achievement_csv(achievements_path, summary)
        _write_policy_csv(policies_path, manifest, summary)

        artifact_paths = (episodes_path, summary_path, achievements_path, policies_path)
        output_manifest.update(
            {
                "status": "complete",
                "completed_episodes": total,
                "completed_at": _now(),
                "artifacts": {
                    path.name: {
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in artifact_paths
                },
            }
        )
        _write_json_atomic(output_manifest_path, output_manifest)
        if manifest.reference_output:
            _copy_reference_outputs(
                repository_root / manifest.reference_output,
                (output_manifest_path, summary_path, achievements_path, policies_path),
            )
        return summary
    except Exception as exc:
        output_manifest.update(
            {
                "status": "failed",
                "completed_episodes": len(completed),
                "updated_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json_atomic(output_manifest_path, output_manifest)
        raise


def _run_episode(
    manifest: BenchmarkManifest,
    policy_spec: PolicySpec,
    policy: Policy,
    seed: int,
) -> EpisodeRecord:
    config = manifest.scenario.config
    env = VoyagerParallelEnv(
        num_agents=_config_int(config, "num_agents"),
        map_size=_config_int(config, "map_size"),
        max_steps=_config_int(config, "max_steps"),
        local_view_size=_config_int(config, "local_view_size"),
        inventory_capacity=_config_int(config, "inventory_capacity"),
        storm_start_step=_config_int(config, "storm_start_step"),
        storm_interval=_config_int(config, "storm_interval"),
        storm_duration=_config_int(config, "storm_duration"),
        storm_damage=_config_float(config, "storm_damage"),
        food_regen_interval=_config_int(config, "food_regen_interval"),
        food_spawn_rate=_config_float(config, "food_spawn_rate"),
        reward_mode="dense",
    )
    observations, infos = env.reset(seed=seed)
    dense_return = 0.0
    agent_steps = 0
    action_counts: dict[str, int] = defaultdict(int)
    action_counts_by_role: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    reward_components: dict[str, float] = defaultdict(float)
    reward_components_by_role: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    selected_invalid_actions = 0
    raw_invalid_actions = 0
    invalid_probability_mass = 0.0
    curves: dict[str, list[float]] = {
        "step": [],
        "survivors": [],
        "mean_health": [],
        "mean_hunger": [],
        "mean_energy": [],
        "camp_food": [],
        "shelter_progress": [],
    }

    while env.agents:
        agent_ids = tuple(env.agents)
        roles = {agent_id: str(infos[agent_id]["role"]) for agent_id in agent_ids}
        decisions = _decide_many(policy, agent_ids, observations, infos)
        actions: dict[str, int] = {}
        for agent_id in agent_ids:
            decision = decisions[agent_id]
            action = int(decision.action)
            raw_action = int(decision.raw_action)
            mask = action_mask_from_info(infos[agent_id])
            action_name = _action_name(action)
            role = roles[agent_id]
            actions[agent_id] = action
            action_counts[action_name] += 1
            action_counts_by_role[role][action_name] += 1
            if not _mask_contains(mask, action):
                selected_invalid_actions += 1
            if not _mask_contains(mask, raw_action):
                raw_invalid_actions += 1
            invalid_probability_mass += decision.invalid_probability_mass

        observations, _rewards, _terms, _truncs, step_infos = env.step(actions)
        agent_steps += len(agent_ids)
        for agent_id in agent_ids:
            dense_components = step_infos[agent_id]["dense_reward_components"]
            if not isinstance(dense_components, dict):
                raise TypeError("dense_reward_components must be a dictionary.")
            role = roles[agent_id]
            for name, value in dense_components.items():
                numeric_value = float(value)
                dense_return += numeric_value
                reward_components[str(name)] += numeric_value
                reward_components_by_role[role][str(name)] += numeric_value
        infos.update(step_infos)
        _append_curve_point(curves, env)

    metrics = env.metrics()
    camp = _dict_metric(metrics, "camp")
    stockpile = _nested_dict_metric(camp, "stockpile")
    achievements = _string_list_metric(metrics, "achievements")
    achievement_steps = _string_int_dict_metric(metrics, "achievement_steps")
    resource_flow = _dict_metric(metrics, "resource_flow")
    survivors = _numeric_int(metrics["active_agents"])
    deaths = _numeric_int(metrics["deaths"])
    return EpisodeRecord(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        benchmark_id=manifest.benchmark_id,
        scenario_id=manifest.scenario.id,
        seed_suite_id=manifest.seed_suite.id,
        policy_id=policy_spec.id,
        policy_kind=policy_spec.kind,
        official=policy_spec.official,
        checkpoint=policy_spec.checkpoint,
        training_seed=policy_spec.training_seed,
        inference_mode=policy_spec.inference_mode,
        seed=seed,
        world_steps=_numeric_int(metrics["step"]),
        agent_steps=agent_steps,
        dense_return=dense_return,
        achievement_return=float(len(achievements)),
        survivors=survivors,
        deaths=deaths,
        survival_rate=survivors / _config_int(config, "num_agents"),
        shelter_progress=float(camp["shelter_progress"]),
        shelter_completion_step=_optional_int(metrics["shelter_completion_step"]),
        camp_stockpile={name: _numeric_int(value) for name, value in stockpile.items()},
        achievements=achievements,
        achievement_steps=achievement_steps,
        resource_flow=resource_flow,
        action_counts=_plain_counts(action_counts),
        action_counts_by_role={
            role: _plain_counts(counts)
            for role, counts in sorted(action_counts_by_role.items())
        },
        selected_invalid_actions=selected_invalid_actions,
        raw_invalid_actions=raw_invalid_actions,
        invalid_probability_mass=invalid_probability_mass,
        dense_reward_components={
            name: float(value) for name, value in sorted(reward_components.items())
        },
        dense_reward_components_by_role={
            role: {
                name: float(value)
                for name, value in sorted(components.items())
            }
            for role, components in sorted(reward_components_by_role.items())
        },
        curves=curves,
    )


def _decide_many(
    policy: Policy,
    agent_ids: tuple[str, ...],
    observations: dict[str, dict[str, np.ndarray]],
    infos: dict[str, dict[str, Any]],
) -> dict[str, PolicyDecision]:
    decide_many = getattr(policy, "decide_many", None)
    if callable(decide_many):
        return decide_many(agent_ids, observations, infos)
    decisions: dict[str, PolicyDecision] = {}
    for agent_id in agent_ids:
        action = int(policy.act(agent_id, observations[agent_id], infos[agent_id]))
        invalid = not _mask_contains(action_mask_from_info(infos[agent_id]), action)
        decisions[agent_id] = PolicyDecision(
            action=action,
            raw_action=action,
            invalid_probability_mass=float(invalid),
        )
    return decisions


def _policy_runtime(
    spec: PolicySpec,
    seed: int,
    repository_root: Path,
    runtimes: dict[str, Policy],
) -> Policy:
    if spec.kind == "random":
        return RandomPolicy(seed=seed)
    if spec.kind == "greedy":
        return GreedySurvivalPolicy()
    if spec.kind == "cooperative":
        return CooperativePolicy()
    runtime = runtimes.get(spec.id)
    if runtime is None:
        if spec.checkpoint is None:
            raise ValueError(f"PPO policy {spec.id!r} has no checkpoint.")
        runtime = TensorFlowPPOPolicy(
            repository_root / spec.checkpoint,
            deterministic=spec.inference_mode == "deterministic",
            seed=seed,
        )
        runtimes[spec.id] = runtime
    reset = getattr(runtime, "reset", None)
    if callable(reset):
        reset(seed)
    return runtime


def _validate_manifest(manifest: BenchmarkManifest, repository_root: Path) -> None:
    scenario = manifest.scenario
    expected_versions = {
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": DENSE_REWARD_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "achievement_version": ACHIEVEMENT_VERSION,
    }
    for field, expected in expected_versions.items():
        actual = getattr(scenario, field)
        if actual != expected:
            raise ValueError(f"{field}={actual!r}; expected {expected!r}.")
    if scenario.id != SCENARIO_VERSION:
        raise ValueError(f"Unsupported scenario id: {scenario.id!r}.")

    for policy in manifest.policies:
        if policy.kind != "ppo":
            continue
        _validate_checkpoint(policy, manifest.seed_suite.seeds, repository_root)


def _validate_checkpoint(
    policy: PolicySpec,
    evaluation_seeds: list[int],
    repository_root: Path,
) -> None:
    if policy.checkpoint is None or policy.checkpoint_sha256 is None:
        raise ValueError(f"PPO policy {policy.id!r} is missing checkpoint metadata.")
    checkpoint = repository_root / policy.checkpoint
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_path = checkpoint / str(metadata["weights_file"])
    weights_hash = _sha256(weights_path)
    if weights_hash != policy.checkpoint_sha256:
        raise ValueError(f"Checkpoint hash mismatch for {policy.id!r}.")
    if weights_hash != metadata.get("weights_sha256"):
        raise ValueError(f"Checkpoint metadata hash mismatch for {policy.id!r}.")
    required = {
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": DENSE_REWARD_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "input_dim": 210,
        "action_count": ACTION_COUNT,
        "training_seed": policy.training_seed,
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Checkpoint {policy.id!r} has {field}={metadata.get(field)!r}; "
                f"expected {expected!r}."
            )
    upper_bound = int(metadata["training_seed_upper_bound_exclusive"])
    training_seed = int(metadata["training_seed"])
    overlap = [
        seed
        for seed in evaluation_seeds
        if training_seed <= seed < upper_bound
    ]
    if overlap:
        raise ValueError(
            f"Evaluation seeds overlap the conservative training range for {policy.id!r}."
        )


def _append_curve_point(
    curves: dict[str, list[float]],
    env: VoyagerParallelEnv,
) -> None:
    state = env.world.state
    if state is None:
        raise RuntimeError("Environment state is unavailable.")
    agents = list(state.agents.values())
    curves["step"].append(float(state.step_count))
    curves["survivors"].append(float(sum(agent.alive for agent in agents)))
    curves["mean_health"].append(float(np.mean([agent.health for agent in agents])))
    curves["mean_hunger"].append(float(np.mean([agent.hunger for agent in agents])))
    curves["mean_energy"].append(float(np.mean([agent.energy for agent in agents])))
    curves["camp_food"].append(float(state.camp.stockpile["food"]))
    curves["shelter_progress"].append(float(state.camp.shelter_progress))


def _write_achievement_csv(path: Path, summary: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    policies = summary["policies"]
    if not isinstance(policies, dict):
        raise TypeError("Summary policies must be a dictionary.")
    for policy_id, policy_summary in policies.items():
        if not isinstance(policy_summary, dict):
            continue
        achievements = policy_summary["achievements"]
        if not isinstance(achievements, dict):
            continue
        for achievement_id in ACHIEVEMENT_IDS:
            values = achievements[achievement_id]
            rows.append(
                {
                    "policy_id": policy_id,
                    "achievement_id": achievement_id,
                    **values,
                }
            )
    family = summary.get("ppo_official_family")
    if isinstance(family, dict):
        achievements = family["achievements"]
        for achievement_id in ACHIEVEMENT_IDS:
            rows.append(
                {
                    "policy_id": "ppo_official_family",
                    "achievement_id": achievement_id,
                    **achievements[achievement_id],
                }
            )
    _write_csv_atomic(path, rows)


def _write_policy_csv(
    path: Path,
    manifest: BenchmarkManifest,
    summary: dict[str, object],
) -> None:
    policies = summary["policies"]
    if not isinstance(policies, dict):
        raise TypeError("Summary policies must be a dictionary.")
    rows: list[dict[str, object]] = []
    for spec in manifest.policies:
        policy_summary = policies[spec.id]
        if not isinstance(policy_summary, dict):
            raise TypeError("Policy summary must be a dictionary.")
        metrics = policy_summary["metrics"]
        score = policy_summary["civilization_score"]
        if not isinstance(metrics, dict) or not isinstance(score, dict):
            raise TypeError("Policy metrics and score must be dictionaries.")
        rows.append(
            {
                "policy_id": spec.id,
                "kind": spec.kind,
                "official": spec.official,
                "training_seed": spec.training_seed,
                "inference_mode": spec.inference_mode,
                "civilization_score": score["mean"],
                "civilization_score_ci_low": score["ci_low"],
                "civilization_score_ci_high": score["ci_high"],
                "mean_survivors": metrics["survivors"]["mean"],
                "mean_dense_return": metrics["dense_return"]["mean"],
                "mean_achievements": metrics["achievement_count"]["mean"],
                "mean_shelter_progress": metrics["shelter_progress"]["mean"],
            }
        )
    family = summary.get("ppo_official_family")
    if isinstance(family, dict):
        metrics = family["metrics"]
        score = family["civilization_score"]
        rows.append(
            {
                "policy_id": "ppo_official_family",
                "kind": "ppo_family",
                "official": True,
                "training_seed": "",
                "inference_mode": "deterministic",
                "civilization_score": score["mean"],
                "civilization_score_ci_low": score["ci_low"],
                "civilization_score_ci_high": score["ci_high"],
                "mean_survivors": metrics["survivors"]["mean"],
                "mean_dense_return": metrics["dense_return"]["mean"],
                "mean_achievements": metrics["achievement_count"]["mean"],
                "mean_shelter_progress": metrics["shelter_progress"]["mean"],
            }
        )
    _write_csv_atomic(path, rows)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as temporary:
        fieldnames = list(rows[0])
        extra_names = sorted(
            set().union(*(row.keys() for row in rows)) - set(fieldnames)
        )
        writer = csv.DictWriter(
            temporary,
            fieldnames=[*fieldnames, *extra_names],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _copy_reference_outputs(reference_dir: Path, paths: tuple[Path, ...]) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, reference_dir / path.name)


def _read_episode_records(path: Path) -> list[EpisodeRecord]:
    if not path.exists():
        return []
    records: list[EpisodeRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(EpisodeRecord.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid episode record on line {line_number}: {exc}") from exc
    return records


def _new_output_manifest(
    manifest: BenchmarkManifest,
    fingerprint: str,
    expected_episodes: int,
) -> dict[str, object]:
    return {
        **manifest.model_dump(mode="json"),
        "manifest_fingerprint": fingerprint,
        "status": "running",
        "expected_episodes": expected_episodes,
        "completed_episodes": 0,
        "started_at": _now(),
        "updated_at": _now(),
        "artifacts": {},
    }


def _manifest_fingerprint(manifest: BenchmarkManifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _repository_root(path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    for candidate in (path.parent, *path.parents, Path.cwd(), source_root):
        if (candidate / "pyproject.toml").exists():
            return candidate.resolve()
    raise ValueError(f"Could not find repository root above {path}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_name(action: int) -> str:
    try:
        return Action(action).name.lower()
    except ValueError:
        return f"invalid_{action}"


def _mask_contains(mask: np.ndarray, action: int) -> bool:
    return 0 <= action < len(mask) and bool(mask[action])


def _plain_counts(values: dict[str, int]) -> dict[str, int]:
    return {name: int(value) for name, value in sorted(values.items())}


def _dict_metric(metrics: dict[str, object], key: str) -> dict[str, Any]:
    value = metrics[key]
    if not isinstance(value, dict):
        raise TypeError(f"metrics[{key!r}] must be a dictionary.")
    return value


def _nested_dict_metric(metrics: dict[str, Any], key: str) -> dict[str, Any]:
    value = metrics[key]
    if not isinstance(value, dict):
        raise TypeError(f"Nested metric {key!r} must be a dictionary.")
    return value


def _string_list_metric(metrics: dict[str, object], key: str) -> list[str]:
    value = metrics[key]
    if not isinstance(value, list) or not all(isinstance(row, str) for row in value):
        raise TypeError(f"metrics[{key!r}] must be a string list.")
    return value


def _string_int_dict_metric(metrics: dict[str, object], key: str) -> dict[str, int]:
    value = _dict_metric(metrics, key)
    return {str(name): _numeric_int(step) for name, step in value.items()}


def _optional_int(value: object) -> int | None:
    return None if value is None else _numeric_int(value)


def _numeric_int(value: object) -> int:
    if not isinstance(value, Real):
        raise TypeError(f"Expected numeric value, got {type(value).__name__}.")
    return int(float(value))


def _config_int(config: dict[str, int | float], key: str) -> int:
    return int(config[key])


def _config_float(config: dict[str, int | float], key: str) -> float:
    return float(config[key])


def _now() -> str:
    return datetime.now(UTC).isoformat()
