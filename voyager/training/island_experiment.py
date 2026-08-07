"""Auditable procedural baseline orchestration for VoyagerIsland-v1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from voyager.benchmark.island import IslandSeedManifest, load_island_seed_manifests
from voyager.envs.island import (
    ISLAND_ENVIRONMENT_VERSION,
    ISLAND_OBSERVATION_VERSION,
    ISLAND_REWARD_VERSION,
)
from voyager.policies.island_scripted import ScriptedIslandOracle
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION
from voyager.sim.island_registry import ISLAND_ACTION_VERSION
from voyager.sim.scenarios import ISLAND_BENCHMARK_SCENARIO_ID
from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT
from voyager.training.island_evaluation import (
    FeedForwardModelIslandPolicy,
    IslandPolicy,
    LegalRandomIslandPolicy,
    RecurrentModelIslandPolicy,
    evaluate_island_checkpoint,
    evaluate_island_policy,
    island_checkpoint_selection_key,
    scripted_oracle_solvability_gate,
)
from voyager.training.island_reward import ISLAND_TRAINING_REWARD_V4
from voyager.training.ppo import PPOConfig, PPOTrainer, PPOUpdateStats
from voyager.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOTrainer
from voyager.training.seed_schedule import EPISODE_SEED_SCHEDULE_VERSION

Algorithm = Literal["feed_forward_ppo", "recurrent_ppo"]

PROCEDURAL_EXPERIMENT_VERSION = "voyager_island_procedural_baselines_v1"
PROCEDURAL_FINAL_EVALUATION_VERSION = "voyager_island_final_evaluation_v1"
OFFICIAL_TRAINING_SEEDS = (0, 1, 2)
OFFICIAL_AGENT_TRANSITIONS = 1_000_000
OFFICIAL_MILESTONES = (200_000, 400_000, 600_000, 800_000, 1_000_000)
SMOKE_AGENT_TRANSITIONS = 2_560
SMOKE_MILESTONES = (1_280, 2_560)
ALGORITHMS: tuple[Algorithm, ...] = ("feed_forward_ppo", "recurrent_ppo")


@dataclass(frozen=True, slots=True)
class ProceduralRunSpec:
    """Resolved immutable configuration for one algorithm and training seed."""

    experiment_root: Path
    algorithm: Algorithm
    training_seed: int
    official: bool = True

    @property
    def transitions(self) -> int:
        return OFFICIAL_AGENT_TRANSITIONS if self.official else SMOKE_AGENT_TRANSITIONS

    @property
    def milestones(self) -> tuple[int, ...]:
        return OFFICIAL_MILESTONES if self.official else SMOKE_MILESTONES

    @property
    def development_episodes(self) -> int:
        return 50 if self.official else 2

    @property
    def run_directory(self) -> Path:
        return self.experiment_root / "runs" / self.algorithm / f"seed_{self.training_seed}"

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unsupported procedural algorithm: {self.algorithm!r}.")
        if self.training_seed not in OFFICIAL_TRAINING_SEEDS:
            raise ValueError("training_seed must be one of 0, 1, or 2.")


def validate_procedural_run(
    spec: ProceduralRunSpec,
    *,
    require_clean: bool = True,
    manifest_root: str | Path | None = None,
) -> dict[str, object]:
    """Resolve and validate a run without creating files or importing TensorFlow."""

    spec.validate()
    manifests = load_island_seed_manifests(manifest_root)
    revision = _git_revision()
    clean = _git_is_clean()
    if spec.official and require_clean and not clean:
        raise RuntimeError("Official procedural runs require a clean Git worktree.")
    return _protocol_payload(manifests, revision=revision, official=spec.official, clean=clean)


def train_procedural_run(
    spec: ProceduralRunSpec,
    *,
    require_clean: bool = True,
    manifest_root: str | Path | None = None,
) -> Path:
    """Train one procedural baseline and lock its development-selected checkpoint."""

    protocol = validate_procedural_run(
        spec,
        require_clean=require_clean,
        manifest_root=manifest_root,
    )
    manifests = load_island_seed_manifests(manifest_root)
    _prepare_experiment_root(spec.experiment_root, protocol)
    run_dir = spec.run_directory
    if run_dir.exists():
        raise FileExistsError(f"Procedural run directory already exists: {run_dir}.")
    run_dir.mkdir(parents=True)
    _write_status(run_dir, "initializing")
    try:
        trainer, config = _build_trainer(spec, manifests["train"], run_dir)
        config_payload = {
            "contract": PROCEDURAL_EXPERIMENT_VERSION,
            "official": spec.official,
            "algorithm": spec.algorithm,
            "training_seed": spec.training_seed,
            "git_revision": protocol["git_revision"],
            "environment_seed_schedule_version": EPISODE_SEED_SCHEDULE_VERSION,
            "training_manifest_sha256": manifests["train"].sha256,
            "development_manifest_sha256": manifests["development"].sha256,
            "test_manifest_sha256": manifests["test"].sha256,
            "development_seeds": list(
                manifests["development"].seeds[: spec.development_episodes]
            ),
            "test_seeds_accessed": False,
            "milestones_agent_transitions": list(spec.milestones),
            "trainer": asdict(config),
        }
        write_json(run_dir / "run_config.json", config_payload)
        _write_status(run_dir, "training")
        dev_seeds = manifests["development"].seeds[: spec.development_episodes]
        random_episodes, random_summary = evaluate_island_policy(
            lambda seed: LegalRandomIslandPolicy(seed),
            seeds=dev_seeds,
            procedural=True,
        )
        pending = list(spec.milestones)
        evaluations: list[dict[str, object]] = []
        best_key: tuple[float, float] | None = None
        best_milestone: int | None = None
        best_checkpoint: str | None = None
        history_path = run_dir / "training_history.jsonl"

        with history_path.open("x", encoding="utf-8") as history_stream:

            def on_update(stats: PPOUpdateStats) -> None:
                nonlocal best_key, best_milestone, best_checkpoint
                history_stream.write(json.dumps(asdict(stats), sort_keys=True) + "\n")
                history_stream.flush()
                _print_update(spec.algorithm, stats)
                while pending and stats.agent_steps >= pending[0]:
                    milestone = pending.pop(0)
                    checkpoint = trainer.save_named_checkpoint(
                        f"agent_steps_{milestone:09d}", stats.update
                    )
                    evaluation = _evaluate_live_model(
                        trainer,
                        algorithm=spec.algorithm,
                        seeds=dev_seeds,
                    )
                    stochastic = _evaluation_summary(evaluation, "stochastic")
                    key = island_checkpoint_selection_key(stochastic)
                    selected = best_key is None or key > best_key
                    if selected:
                        best_key = key
                        best_milestone = milestone
                        best_checkpoint = trainer.save_named_checkpoint("best", stats.update)
                    evaluations.append(
                        {
                            "milestone_agent_transitions": milestone,
                            "actual_agent_transitions": stats.agent_steps,
                            "checkpoint": _relative_checkpoint(run_dir, checkpoint),
                            "selection_mode": "seeded_stochastic",
                            "selection_key": {
                                "achievement_geometric_mean": key[0],
                                "invalid_action_rate": -key[1],
                            },
                            "became_best_so_far": selected,
                            "evaluation": evaluation,
                        }
                    )
                    _write_development_artifact(
                        run_dir,
                        dev_seeds=dev_seeds,
                        random_episodes=random_episodes,
                        random_summary=random_summary,
                        evaluations=evaluations,
                        best_milestone=best_milestone,
                        best_checkpoint=best_checkpoint,
                    )

            stats = trainer.train(on_update=on_update)

        if best_checkpoint is None or best_milestone is None:
            raise RuntimeError("Procedural training completed without selecting a checkpoint.")
        for evaluation in evaluations:
            evaluation["is_final_selection"] = (
                evaluation["milestone_agent_transitions"] == best_milestone
            )
        _write_development_artifact(
            run_dir,
            dev_seeds=dev_seeds,
            random_episodes=random_episodes,
            random_summary=random_summary,
            evaluations=evaluations,
            best_milestone=best_milestone,
            best_checkpoint=best_checkpoint,
        )
        _write_training_csv(run_dir / "training_history.csv", stats)
        write_json(
            run_dir / "environment_seed_history.json",
            {
                "contract": EPISODE_SEED_SCHEDULE_VERSION,
                "split": "train",
                "manifest_sha256": manifests["train"].sha256,
                "training_seed": spec.training_seed,
                "seeds": list(trainer.environment_seed_history),
            },
        )
        selection = {
            "contract": "voyager_island_checkpoint_selection_v1",
            "locked": True,
            "primary_inference": "seeded_stochastic",
            "primary_metric": "achievement_geometric_mean",
            "tie_breakers": ["lower_invalid_action_rate", "earlier_milestone"],
            "development_manifest_sha256": manifests["development"].sha256,
            "development_seeds": list(dev_seeds),
            "best_milestone_agent_transitions": best_milestone,
            "best_checkpoint": _relative_checkpoint(run_dir, best_checkpoint),
            "test_seeds_accessed": False,
            "latest_checkpoint_is_diagnostic_only": True,
            "timing": trainer.timing_report(),
        }
        write_json(run_dir / "selection.json", selection)
        reward_diagnostics = getattr(trainer.env, "reward_diagnostics", None)
        if callable(reward_diagnostics):
            write_json(run_dir / "reward_diagnostics.json", reward_diagnostics())
        _write_status(run_dir, "selection_locked")
        _write_artifact_manifest(run_dir)
        print(f"Selection locked: {Path(best_checkpoint).resolve()}")
        print(f"Artifacts: {run_dir.resolve()}")
        return run_dir
    except BaseException as error:
        _write_status(run_dir, "failed", error=f"{type(error).__name__}: {error}")
        raise


def finalize_procedural_suite(
    experiment_root: str | Path,
    *,
    manifest_root: str | Path | None = None,
) -> Path:
    """Evaluate all six locked selections once on the held-out test manifest."""

    root = Path(experiment_root)
    final_dir = root / "final"
    if final_dir.exists():
        raise FileExistsError(f"Procedural suite has already been finalized: {final_dir}.")
    protocol = _read_mapping(root / "protocol.json")
    if protocol.get("contract") != PROCEDURAL_EXPERIMENT_VERSION or not protocol.get("official"):
        raise ValueError("Only an official procedural experiment suite can be finalized.")
    if protocol.get("git_revision") != _git_revision() or not _git_is_clean():
        raise RuntimeError(
            "Finalization requires the clean Git revision that produced the locked runs."
        )
    manifests = load_island_seed_manifests(manifest_root)
    _verify_protocol_manifests(protocol, manifests)
    selections = _load_locked_run_matrix(root, protocol, manifests)
    test_seeds = manifests["test"].seeds
    temporary = Path(tempfile.mkdtemp(prefix=".final-", dir=root))
    try:
        random_episodes, random_summary = evaluate_island_policy(
            lambda seed: LegalRandomIslandPolicy(seed),
            seeds=test_seeds,
            procedural=True,
        )
        oracle_episodes, oracle_summary = evaluate_island_policy(
            lambda _seed: ScriptedIslandOracle(),
            seeds=test_seeds,
            procedural=True,
        )
        write_json(
            temporary / "legal_random.json",
            {
                "contract": PROCEDURAL_FINAL_EVALUATION_VERSION,
                "policy": "legal_random",
                "split": "test",
                "manifest_sha256": manifests["test"].sha256,
                "summary": random_summary,
                "episodes": [episode.as_dict() for episode in random_episodes],
            },
        )
        write_json(
            temporary / "scripted_oracle.json",
            {
                "contract": PROCEDURAL_FINAL_EVALUATION_VERSION,
                "policy": "scripted_oracle",
                "split": "test",
                "manifest_sha256": manifests["test"].sha256,
                "summary": oracle_summary,
                "gate": scripted_oracle_solvability_gate(oracle_summary),
                "episodes": [episode.as_dict() for episode in oracle_episodes],
            },
        )
        learned_index: list[dict[str, object]] = []
        for algorithm, training_seed, checkpoint in selections:
            evaluation = evaluate_island_checkpoint(
                checkpoint,
                seeds=test_seeds,
                procedural=True,
                include_episodes=True,
            )
            relative = Path("learned") / algorithm / f"seed_{training_seed}.json"
            write_json(
                temporary / relative,
                {
                    "contract": PROCEDURAL_FINAL_EVALUATION_VERSION,
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "split": "test",
                    "manifest_sha256": manifests["test"].sha256,
                    "primary_inference": "seeded_stochastic",
                    "evaluation": evaluation,
                },
            )
            learned_index.append(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "checkpoint": str(checkpoint.relative_to(root.resolve())),
                    "artifact": str(relative),
                }
            )
        write_json(
            temporary / "final_evaluation.json",
            {
                "contract": PROCEDURAL_FINAL_EVALUATION_VERSION,
                "experiment_contract": PROCEDURAL_EXPERIMENT_VERSION,
                "git_revision": protocol["git_revision"],
                "test_manifest_sha256": manifests["test"].sha256,
                "test_seeds": list(test_seeds),
                "test_results_used_for_selection": False,
                "comparators": ["legal_random.json", "scripted_oracle.json"],
                "learned_runs": learned_index,
            },
        )
        _write_artifact_manifest(temporary, contract=PROCEDURAL_FINAL_EVALUATION_VERSION)
        os.replace(temporary, final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    for algorithm, training_seed, _checkpoint in selections:
        run_dir = root / "runs" / algorithm / f"seed_{training_seed}"
        _write_status(run_dir, "finalized")
        _write_artifact_manifest(run_dir)
    print(f"Held-out final evaluation: {final_dir.resolve()}")
    return final_dir


def verify_finalized_suite(experiment_root: str | Path) -> dict[str, object]:
    """Validate hashes in an existing final suite without rerunning any episode."""

    final_dir = Path(experiment_root) / "final"
    manifest = _read_mapping(final_dir / "artifact_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("Final artifact manifest has no artifacts list.")
    checked = 0
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise TypeError("Invalid final artifact row.")
        path = final_dir / str(row["path"])
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"Final artifact failed validation: {path}.")
        checked += 1
    return {"status": "valid", "checked_artifacts": checked, "path": str(final_dir.resolve())}


def _build_trainer(
    spec: ProceduralRunSpec,
    train_manifest: IslandSeedManifest,
    run_dir: Path,
) -> tuple[Any, PPOConfig | RecurrentPPOConfig]:
    checkpoint_dir = str((run_dir / "checkpoints").resolve())
    if spec.algorithm == "feed_forward_ppo":
        feed_forward_config = PPOConfig(
            total_steps=spec.transitions,
            environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
            reward_contract=ISLAND_TRAINING_REWARD_V4,
            rollout_steps=128,
            num_agents=2,
            map_size=48,
            max_steps=1_200,
            seed=spec.training_seed,
            entropy_coef_start=0.02,
            entropy_coef_end=0.005,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=0,
            use_action_mask=True,
            procedural=True,
            episode_seed_pool=train_manifest.seeds,
            episode_seed_manifest_hash=train_manifest.sha256,
            episode_seed_split="train",
            reward_mode="dense",
            hidden_sizes=(128, 128),
            minibatch_size=256,
        )
        return PPOTrainer(feed_forward_config), feed_forward_config
    recurrent_config = RecurrentPPOConfig(
        total_steps=spec.transitions,
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V4,
        rollout_steps=128,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        seed=spec.training_seed,
        entropy_coef_start=0.02,
        entropy_coef_end=0.005,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=0,
        use_action_mask=True,
        procedural=True,
        episode_seed_pool=train_manifest.seeds,
        episode_seed_manifest_hash=train_manifest.sha256,
        episode_seed_split="train",
        encoder_sizes=(128,),
        recurrent_hidden_size=128,
        sequence_length=32,
        sequence_minibatch_size=16,
        max_gradient_norm=0.5,
    )
    return RecurrentPPOTrainer(recurrent_config), recurrent_config


def _evaluate_live_model(
    trainer: Any,
    *,
    algorithm: Algorithm,
    seeds: Sequence[int],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for mode, deterministic in (("stochastic", False), ("deterministic", True)):
        factory: Callable[[int], IslandPolicy]
        if algorithm == "feed_forward_ppo":
            factory = _feed_forward_policy_factory(
                trainer.model,
                trainer.observation_encoder,
                deterministic,
            )
        else:
            factory = _recurrent_policy_factory(
                trainer.model,
                trainer.observation_encoder,
                trainer.config.recurrent_hidden_size,
                deterministic,
            )
        episodes, summary = evaluate_island_policy(
            factory,
            seeds=seeds,
            procedural=True,
        )
        payload[mode] = {
            "summary": summary,
            "episodes": [episode.as_dict() for episode in episodes],
        }
    return payload


def _feed_forward_policy_factory(
    model: Any,
    encoder: str,
    deterministic: bool,
) -> Callable[[int], IslandPolicy]:
    def create(seed: int) -> IslandPolicy:
        return FeedForwardModelIslandPolicy(
            model,
            encoder,
            deterministic=deterministic,
            seed=seed,
        )

    return create


def _recurrent_policy_factory(
    model: Any,
    encoder: str,
    hidden_size: int,
    deterministic: bool,
) -> Callable[[int], IslandPolicy]:
    def create(seed: int) -> IslandPolicy:
        return RecurrentModelIslandPolicy(
            model=model,
            encoder=encoder,
            hidden_size=hidden_size,
            deterministic=deterministic,
            seed=seed,
        )

    return create


def _write_development_artifact(
    run_dir: Path,
    *,
    dev_seeds: Sequence[int],
    random_episodes: Sequence[Any],
    random_summary: Mapping[str, object],
    evaluations: Sequence[Mapping[str, object]],
    best_milestone: int | None,
    best_checkpoint: str | None,
) -> None:
    write_json(
        run_dir / "development_evaluations.json",
        {
            "contract": "voyager_island_checkpoint_selection_v1",
            "procedural": True,
            "seeds": list(dev_seeds),
            "random": {
                "summary": dict(random_summary),
                "episodes": [episode.as_dict() for episode in random_episodes],
            },
            "milestones": list(evaluations),
            "best_milestone_agent_transitions": best_milestone,
            "best_checkpoint": (
                _relative_checkpoint(run_dir, best_checkpoint)
                if best_checkpoint is not None
                else None
            ),
            "test_seeds_accessed": False,
        },
    )


def _load_locked_run_matrix(
    root: Path,
    protocol: Mapping[str, object],
    manifests: Mapping[str, IslandSeedManifest],
) -> list[tuple[Algorithm, int, Path]]:
    rows: list[tuple[Algorithm, int, Path]] = []
    for algorithm in ALGORITHMS:
        for training_seed in OFFICIAL_TRAINING_SEEDS:
            run_dir = root / "runs" / algorithm / f"seed_{training_seed}"
            status = _read_mapping(run_dir / "status.json")
            if status.get("status") != "selection_locked":
                raise ValueError(f"Run is not selection-locked: {run_dir}.")
            config = _read_mapping(run_dir / "run_config.json")
            selection = _read_mapping(run_dir / "selection.json")
            if not config.get("official") or config.get("algorithm") != algorithm:
                raise ValueError(f"Run does not match the official matrix: {run_dir}.")
            if config.get("training_seed") != training_seed:
                raise ValueError(f"Training seed mismatch in {run_dir}.")
            if config.get("git_revision") != protocol.get("git_revision"):
                raise ValueError(f"Git revision mismatch in {run_dir}.")
            if config.get("training_manifest_sha256") != manifests["train"].sha256:
                raise ValueError(f"Training manifest mismatch in {run_dir}.")
            if config.get("development_manifest_sha256") != manifests["development"].sha256:
                raise ValueError(f"Development manifest mismatch in {run_dir}.")
            if config.get("test_manifest_sha256") != manifests["test"].sha256:
                raise ValueError(f"Test manifest mismatch in {run_dir}.")
            if config.get("milestones_agent_transitions") != list(OFFICIAL_MILESTONES):
                raise ValueError(f"Official milestone schedule mismatch in {run_dir}.")
            if config.get("test_seeds_accessed") is not False:
                raise ValueError(f"Test split was accessed before finalization in {run_dir}.")
            trainer_config = config.get("trainer")
            if not isinstance(trainer_config, Mapping):
                raise TypeError(f"Missing trainer configuration in {run_dir}.")
            if trainer_config.get("total_steps") != OFFICIAL_AGENT_TRANSITIONS:
                raise ValueError(f"Official transition budget mismatch in {run_dir}.")
            if trainer_config.get("environment_id") != ISLAND_V1_TRAINING_ENVIRONMENT:
                raise ValueError(f"Environment contract mismatch in {run_dir}.")
            if trainer_config.get("reward_contract") != ISLAND_TRAINING_REWARD_V4:
                raise ValueError(f"Training reward contract mismatch in {run_dir}.")
            if trainer_config.get("procedural") is not True:
                raise ValueError(f"Procedural mode is disabled in {run_dir}.")
            if not selection.get("locked") or selection.get("test_seeds_accessed") is not False:
                raise ValueError(f"Invalid locked selection in {run_dir}.")
            checkpoint_value = selection.get("best_checkpoint")
            if not isinstance(checkpoint_value, str):
                raise TypeError(f"Missing selected checkpoint in {run_dir}.")
            checkpoint = (run_dir / checkpoint_value).resolve()
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}.")
            _verify_artifact_manifest(run_dir)
            rows.append((algorithm, training_seed, checkpoint))
    return rows


def _prepare_experiment_root(root: Path, protocol: Mapping[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        existing = _read_mapping(protocol_path)
        if existing != protocol:
            raise ValueError("Experiment root is locked to a different protocol or Git revision.")
    else:
        write_json(protocol_path, dict(protocol))


def _protocol_payload(
    manifests: Mapping[str, IslandSeedManifest],
    *,
    revision: str,
    official: bool,
    clean: bool,
) -> dict[str, object]:
    return {
        "contract": PROCEDURAL_EXPERIMENT_VERSION,
        "official": official,
        "git_revision": revision,
        "git_worktree_clean": clean,
        "environment_version": ISLAND_ENVIRONMENT_VERSION,
        "scenario_version": ISLAND_BENCHMARK_SCENARIO_ID,
        "observation_version": ISLAND_OBSERVATION_VERSION,
        "action_version": ISLAND_ACTION_VERSION,
        "public_reward_version": ISLAND_REWARD_VERSION,
        "training_reward_version": ISLAND_TRAINING_REWARD_V4,
        "achievement_version": ISLAND_ACHIEVEMENT_VERSION,
        "episode_seed_schedule_version": EPISODE_SEED_SCHEDULE_VERSION,
        "algorithms": list(ALGORITHMS),
        "training_seeds": list(OFFICIAL_TRAINING_SEEDS),
        "agent_transition_budget": (
            OFFICIAL_AGENT_TRANSITIONS if official else SMOKE_AGENT_TRANSITIONS
        ),
        "checkpoint_milestones": list(
            OFFICIAL_MILESTONES if official else SMOKE_MILESTONES
        ),
        "manifests": {
            split: {
                "filename": manifest.path.name,
                "sha256": manifest.sha256,
                "count": len(manifest.seeds),
                "generator_version": manifest.generator_version,
            }
            for split, manifest in sorted(manifests.items())
        },
        "test_access": "separate_finalize_command",
    }


def _verify_protocol_manifests(
    protocol: Mapping[str, object],
    manifests: Mapping[str, IslandSeedManifest],
) -> None:
    rows = protocol.get("manifests")
    if not isinstance(rows, Mapping):
        raise TypeError("Protocol is missing manifest metadata.")
    for split, manifest in manifests.items():
        row = rows.get(split)
        if not isinstance(row, Mapping) or row.get("sha256") != manifest.sha256:
            raise ValueError(f"Protocol {split} manifest hash no longer matches disk.")


def _write_training_csv(path: Path, stats: Sequence[PPOUpdateStats]) -> None:
    if not stats:
        raise ValueError("Cannot export an empty training history.")
    fields = list(asdict(stats[0]))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in stats)


def _relative_checkpoint(run_dir: Path, checkpoint: str) -> str:
    return str(Path(checkpoint).resolve().relative_to(run_dir.resolve()))


def _write_status(run_dir: Path, status: str, *, error: str | None = None) -> None:
    payload: dict[str, object] = {
        "contract": PROCEDURAL_EXPERIMENT_VERSION,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    write_json(run_dir / "status.json", payload)


def _write_artifact_manifest(root: Path, *, contract: str = PROCEDURAL_EXPERIMENT_VERSION) -> None:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        root / "artifact_manifest.json",
        {"contract": contract, "artifacts": artifacts},
    )


def _verify_artifact_manifest(root: Path) -> int:
    manifest = _read_mapping(root / "artifact_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"Artifact manifest has no artifacts list: {root}.")
    checked = 0
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise TypeError(f"Invalid artifact row in {root}.")
        path = root / str(row["path"])
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"Artifact failed validation: {path}.")
        checked += 1
    return checked


def _evaluation_summary(evaluation: Mapping[str, object], mode: str) -> Mapping[str, object]:
    row = evaluation.get(mode)
    if not isinstance(row, Mapping) or not isinstance(row.get("summary"), Mapping):
        raise TypeError(f"Missing {mode} evaluation summary.")
    return row["summary"]  # type: ignore[return-value]


def _read_mapping(path: Path) -> dict[str, object]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _print_update(algorithm: Algorithm, stats: PPOUpdateStats) -> None:
    if stats.update == 1 or stats.update % 10 == 0:
        print(
            f"[{algorithm}] update={stats.update:04d} transitions={stats.agent_steps:,} "
            f"reward={stats.mean_reward:+.4f} throughput={stats.agent_steps_per_second:.1f}/s"
        )


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def _git_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return not completed.stdout.strip()


def write_json(path: Path, value: object) -> None:
    """Atomically write one deterministic, human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
