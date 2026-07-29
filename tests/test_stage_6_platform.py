"""Stage 6 replay platform contract, reconstruction, catalog, and API tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voyager.replay.catalog import ReplayCatalog
from voyager.replay.director import generate_camera_cues
from voyager.replay.loader import CorruptReplayError, ReplayLoader
from voyager.replay.recorder import (
    ReplayRecorder,
    migrate_legacy_replay,
    record_episode,
)
from voyager.replay.schema import REPLAY_SCHEMA_VERSION, replay_major, validate_manifest
from voyager.replay.serialization import canonical_gzip_bytes, canonical_json_bytes
from voyager.server.app import create_app

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "benchmarks/replays/stage6_curated_v1"


@pytest.mark.parametrize(
    ("replay_id", "survivors", "deaths", "achievements"),
    [
        ("random", 8, 2, 6),
        ("greedy", 8, 2, 5),
        ("cooperative", 2, 8, 6),
        ("ppo_seed0_deterministic", 10, 0, 16),
        ("ppo_seed0_stochastic", 10, 0, 16),
    ],
)
def test_curated_replay_outcomes(
    replay_id: str, survivors: int, deaths: int, achievements: int
) -> None:
    loader = ReplayLoader(CURATED / replay_id)
    summary = loader.manifest.terminal_summary
    assert summary["survivors"] == survivors
    assert summary["deaths"] == deaths
    assert len(summary["achievements"]) == achievements
    assert loader.validate(deep=True)["checked_ticks"] == 301


def test_random_access_reconstructs_snapshot_and_chunk_boundaries() -> None:
    loader = ReplayLoader(CURATED / "ppo_seed0_deterministic", cache_size=2)
    for tick in (0, 1, 24, 25, 30, 99, 100, 115, 200, 225, 257, 299, 300):
        state = loader.state_at(tick)
        assert state["tick"] == tick
        assert len(state["agents"]) == 10
    assert loader.state_at(115)["structures"][0]["complete"] is True
    assert loader.state_at(200)["weather"]["storm_active"] is True
    assert loader.state_at(225)["weather"]["storm_active"] is False


def test_canonical_serialization_is_deterministic() -> None:
    value = {"z": [3, 2, 1], "a": {"future": True}}
    assert canonical_json_bytes(value) == canonical_json_bytes(value)
    assert canonical_gzip_bytes(value) == canonical_gzip_bytes(value)


def test_rerecording_produces_identical_canonical_artifacts(tmp_path: Path) -> None:
    manifest = ROOT / "benchmarks/manifests/stage5_6_final.json"
    first = record_episode(
        manifest,
        policy_id="random",
        seed=10_000_010,
        output_root=tmp_path / "first",
        replay_id="random-a",
    )
    second = record_episode(
        manifest,
        policy_id="random",
        seed=10_000_010,
        output_root=tmp_path / "second",
        replay_id="random-b",
    )
    first_checksums = {
        entry.path: entry.sha256 for entry in ReplayLoader(first).manifest.artifacts
    }
    second_checksums = {
        entry.path: entry.sha256 for entry in ReplayLoader(second).manifest.artifacts
    }
    assert first_checksums == second_checksums


def test_recorder_failure_never_catalogs_a_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_tick(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("intentional recorder interruption")

    monkeypatch.setattr(ReplayRecorder, "record_tick", fail_tick)
    with pytest.raises(RuntimeError, match="intentional"):
        record_episode(
            ROOT / "benchmarks/manifests/stage5_6_final.json",
            policy_id="random",
            seed=10_000_010,
            output_root=tmp_path,
            replay_id="interrupted",
        )
    assert not (tmp_path / "interrupted").exists()
    failures = list((tmp_path / ".failed").glob("interrupted-*"))
    assert len(failures) == 1
    assert json.loads((failures[0] / "failure.json").read_text())["status"] == "failed"


def test_unknown_minor_fields_and_extensions_are_preserved() -> None:
    payload = json.loads((CURATED / "random/manifest.json").read_text(encoding="utf-8"))
    payload["future_minor_field"] = {"new": "structure"}
    payload["extensions"]["stage7.example"] = {
        "day_phase": "night",
        "new_resource": "crystal",
        "agent_count": 24,
        "long_episode_ticks": 10_000,
    }
    manifest = validate_manifest(payload)
    dumped = manifest.model_dump(mode="json")
    assert dumped["future_minor_field"]["new"] == "structure"
    assert dumped["extensions"]["stage7.example"]["agent_count"] == 24


def test_unknown_major_version_fails_clearly() -> None:
    assert replay_major(REPLAY_SCHEMA_VERSION) == 2
    payload = json.loads((CURATED / "random/manifest.json").read_text(encoding="utf-8"))
    payload["versions"]["replay"] = "stage6_replay_3.0.0"
    with pytest.raises(ValueError, match="major version 3"):
        validate_manifest(payload)


def test_corrupt_artifact_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "random"
    shutil.copytree(CURATED / "random", target)
    timeline = next((target / "timeline").glob("*.json.gz"))
    timeline.write_bytes(timeline.read_bytes() + b"corrupt")
    with pytest.raises(CorruptReplayError, match="size|checksum"):
        ReplayLoader(target).validate()


def test_legacy_stage6a_fixture_remains_loadable() -> None:
    loader = ReplayLoader(ROOT / "web/public/replays/stage6_vertical_slice_v1.json")
    assert loader.manifest.replay_id == "stage6_vertical_slice_v1"
    assert loader.state_at(300)["tick"] == 300
    assert loader.validate(deep=True)["checked_ticks"] == 301


def test_legacy_fixture_can_migrate_to_formal_v2(tmp_path: Path) -> None:
    output = migrate_legacy_replay(
        ROOT / "web/public/replays/stage6_vertical_slice_v1.json",
        tmp_path / "migrated",
    )
    loader = ReplayLoader(output)
    assert loader.manifest.versions.replay == REPLAY_SCHEMA_VERSION
    assert loader.manifest.extensions["migration"]["scientific_limitations"]
    assert loader.validate(deep=True)["checked_ticks"] == 301


def test_director_is_deterministic_and_validates_overrides() -> None:
    events = [
        {
            "tick": 100,
            "type": "achievement",
            "importance": 70,
            "position": {"x": 4, "y": 5},
            "payload": {"achievement_id": "new_structure"},
        },
        {
            "tick": 200,
            "type": "storm_started",
            "importance": 90,
            "position": {"x": 8, "y": 9},
        },
    ]
    kwargs = {
        "terminal_tick": 300,
        "camp": {"x": 16, "y": 16},
        "agents": [{"x": 15, "y": 16}, {"x": 17, "y": 16}],
    }
    assert generate_camera_cues(events, **kwargs) == generate_camera_cues(events, **kwargs)
    with pytest.raises(ValueError, match="non-overlapping"):
        generate_camera_cues(
            [],
            **kwargs,
            overrides=[
                {"start_tick": 0, "end_tick": 20, "target": {"x": 1, "y": 1}},
                {"start_tick": 20, "end_tick": 30, "target": {"x": 1, "y": 1}},
            ],
        )


def test_catalog_filters_and_ignores_hidden_failures() -> None:
    catalog = ReplayCatalog([("curated", CURATED)])
    assert len(catalog.scan()) == 5
    assert [entry.manifest.replay_id for entry in catalog.query(kind="cooperative")] == [
        "cooperative"
    ]
    assert len(catalog.query(seed=10_000_010, min_survivors=10)) == 2
    assert len(catalog.query(achievement="shelter_complete")) == 2
    with pytest.raises(ValueError):
        catalog.get("../../manifest.json")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Voyager SPA</main>", encoding="utf-8")
    app = create_app(
        replay_roots=[("curated", CURATED)],
        frontend_dir=frontend,
        cache_size=2,
    )
    return TestClient(app)


def test_api_catalog_random_access_etag_and_spa(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "replays": 5}
    listing = client.get("/api/v1/replays?min_survivors=10").json()
    assert listing["total"] == 2
    response = client.get("/api/v1/replays/random/state/100")
    assert response.status_code == 200
    assert response.json()["tick"] == 100
    etag = response.headers["etag"]
    assert (
        client.get("/api/v1/replays/random/state/100", headers={"If-None-Match": etag}).status_code
        == 304
    )
    assert client.get("/api/v1/replays/random/state/301").status_code == 416
    assert client.get("/replays/random").text == "<main>Voyager SPA</main>"


def test_api_events_metrics_pagination_and_comparison(client: TestClient) -> None:
    first = client.get("/api/v1/replays?limit=2").json()
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    second = client.get(f"/api/v1/replays?limit=2&cursor={first['next_cursor']}").json()
    assert len(second["items"]) == 2
    events = client.get(
        "/api/v1/replays/ppo_seed0_deterministic/events?start=190&end=230"
    ).json()["events"]
    assert any(event["type"] == "storm_started" for event in events)
    metrics = client.get(
        "/api/v1/replays/random/metrics?series=survivors"
    ).json()
    assert len(metrics["global"]["survivors"]) == 301
    comparison = client.get(
        "/api/v1/compare?left=random&right=ppo_seed0_deterministic"
    )
    assert comparison.status_code == 200
    assert comparison.json()["terminal_deltas"]["survivors"] == 2
