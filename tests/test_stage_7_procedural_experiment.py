from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from voyager.benchmark.island import (
    ISLAND_DEV_SEEDS,
    ISLAND_TEST_SEEDS,
    ISLAND_TRAIN_SEEDS,
    load_island_seed_manifests,
)
from voyager.training.checkpoints import load_policy_checkpoint, save_policy_checkpoint
from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT
from voyager.training.island_experiment import (
    OFFICIAL_AGENT_TRANSITIONS,
    OFFICIAL_MILESTONES,
    PROCEDURAL_EXPERIMENT_VERSION,
    ProceduralRunSpec,
    finalize_procedural_suite,
    validate_procedural_run,
    verify_finalized_suite,
)
from voyager.training.island_reward import ISLAND_TRAINING_REWARD_V4
from voyager.training.model import build_actor_critic
from voyager.training.ppo import PPOConfig, PPOTrainer
from voyager.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOTrainer
from voyager.training.seed_schedule import EpisodeSeedScheduler


def test_seed_manifests_are_loaded_from_tracked_files() -> None:
    manifests = load_island_seed_manifests()
    assert manifests["train"].seeds == ISLAND_TRAIN_SEEDS
    assert manifests["development"].seeds == ISLAND_DEV_SEEDS
    assert manifests["test"].seeds == ISLAND_TEST_SEEDS
    assert len({manifest.sha256 for manifest in manifests.values()}) == 3


def test_manifest_loader_rejects_contract_drift(tmp_path: Path) -> None:
    source = Path("benchmarks/manifests")
    destination = tmp_path / "manifests"
    shutil.copytree(source, destination)
    path = destination / "island_v1_dev.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed_range"]["count"] = 49
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen development split"):
        load_island_seed_manifests(destination)


def test_episode_seed_scheduler_is_deterministic_bounded_and_cyclic() -> None:
    seed_pool = tuple(range(10))
    left = EpisodeSeedScheduler(seed_pool, shuffle_seed=9)
    right = EpisodeSeedScheduler(seed_pool, shuffle_seed=9)
    other = EpisodeSeedScheduler(seed_pool, shuffle_seed=10)
    left_values = [left.next_seed() for _ in range(20)]
    right_values = [right.next_seed() for _ in range(20)]
    other_values = [other.next_seed() for _ in range(10)]
    assert left_values == right_values
    assert set(left_values[:10]) == set(seed_pool)
    assert set(left_values[10:]) == set(seed_pool)
    assert other_values != left_values[:10]
    assert left.metadata()["current_cycle"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        EpisodeSeedScheduler((1, 1), shuffle_seed=0)


def test_procedural_run_defaults_are_frozen_and_dry_validation_writes_nothing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    spec = ProceduralRunSpec(root, "feed_forward_ppo", 0, official=False)
    protocol = validate_procedural_run(spec, require_clean=False)
    assert protocol["contract"] == PROCEDURAL_EXPERIMENT_VERSION
    assert protocol["test_access"] == "separate_finalize_command"
    assert not root.exists()

    official = ProceduralRunSpec(root, "recurrent_ppo", 2)
    assert official.transitions == OFFICIAL_AGENT_TRANSITIONS
    assert official.milestones == OFFICIAL_MILESTONES
    assert official.development_episodes == 50
    with pytest.raises(ValueError, match="0, 1, or 2"):
        ProceduralRunSpec(root, "feed_forward_ppo", 3).validate()


def test_finalizer_rejects_nonofficial_protocol_before_test_access(tmp_path: Path) -> None:
    _write_json(tmp_path / "protocol.json", {"contract": PROCEDURAL_EXPERIMENT_VERSION, "official": False})
    with pytest.raises(ValueError, match="official"):
        finalize_procedural_suite(tmp_path)
    assert not (tmp_path / "final").exists()


def test_final_artifact_verification_is_read_only_and_detects_corruption(tmp_path: Path) -> None:
    final = tmp_path / "final"
    _write_json(final / "result.json", {"score": 0.5})
    _write_json(
        final / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "path": "result.json",
                    "size": (final / "result.json").stat().st_size,
                    "sha256": _sha256_file(final / "result.json"),
                }
            ]
        },
    )
    assert verify_finalized_suite(tmp_path)["checked_artifacts"] == 1
    _write_json(final / "result.json", {"score": 0.6})
    with pytest.raises(ValueError, match="failed validation"):
        verify_finalized_suite(tmp_path)


@pytest.mark.parametrize("recurrent", [False, True])
def test_trainers_consume_supplied_manifest_seed_schedule(recurrent: bool) -> None:
    pytest.importorskip("tensorflow")
    common = {
        "total_steps": 256,
        "environment_id": ISLAND_V1_TRAINING_ENVIRONMENT,
        "reward_contract": ISLAND_TRAINING_REWARD_V4,
        "rollout_steps": 2,
        "num_agents": 2,
        "map_size": 48,
        "max_steps": 1_200,
        "seed": 7,
        "checkpoint_dir": None,
        "procedural": True,
        "episode_seed_pool": (11, 13, 17),
        "episode_seed_manifest_hash": "test-manifest",
        "episode_seed_split": "train",
    }
    if recurrent:
        config = RecurrentPPOConfig(
            **common,
            sequence_length=2,
            sequence_minibatch_size=1,
            encoder_sizes=(8,),
            recurrent_hidden_size=8,
        )
        trainer = RecurrentPPOTrainer(config)
    else:
        config = PPOConfig(**common, hidden_sizes=(8,), minibatch_size=4)
        trainer = PPOTrainer(config)
    trainer._reset_env()
    assert len(trainer.environment_seed_history) == 2
    assert set(trainer.environment_seed_history) <= {11, 13, 17}
    assert trainer.seed_scheduler is not None
    assert trainer.seed_scheduler.metadata()["manifest_sha256"] == "test-manifest"
    trainer.env.close()


def test_checkpoint_hash_is_verified_and_legacy_metadata_remains_loadable(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tensorflow")
    model = build_actor_critic(input_dim=4, action_count=2, hidden_sizes=(8,), seed=0)
    checkpoint = save_policy_checkpoint(
        model,
        tmp_path / "checkpoint",
        {
            "model_type": "feed_forward",
            "input_dim": 4,
            "action_count": 2,
            "hidden_sizes": [8],
        },
    )
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["weights_sha256"] == _sha256_file(checkpoint / "model.weights.h5")
    load_policy_checkpoint(checkpoint)

    expected_hash = metadata.pop("weights_sha256")
    _write_json(metadata_path, metadata)
    load_policy_checkpoint(checkpoint)

    metadata["weights_sha256"] = expected_hash
    _write_json(metadata_path, metadata)
    with (checkpoint / "model.weights.h5").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_policy_checkpoint(checkpoint)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
