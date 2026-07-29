"""Stage 6A replay exporter regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voyager.replay.exporter import (
    LOCKED_EVALUATION_SEED,
    LOCKED_POLICY_ID,
    REPLAY_SCHEMA_VERSION,
    export_vertical_slice,
)


def test_tracked_vertical_slice_contract() -> None:
    replay_path = (
        Path(__file__).resolve().parents[1]
        / "web/public/replays/stage6_vertical_slice_v1.json"
    )
    payload = json.loads(replay_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == REPLAY_SCHEMA_VERSION
    assert payload["source"]["policy_id"] == LOCKED_POLICY_ID
    assert payload["source"]["evaluation_seed"] == LOCKED_EVALUATION_SEED
    assert payload["tick_rate"] == 12
    assert payload["duration_seconds"] == 25.0
    assert len(payload["frames"]) == 300
    assert len(payload["world"]["initial"]["agents"]) == 10
    assert len({agent["name"] for agent in payload["world"]["initial"]["agents"]}) == 10
    assert payload["summary"]["survivors"] == 10
    assert payload["summary"]["deaths"] == 0
    assert len(payload["summary"]["achievements"]) == 16
    assert payload["summary"]["achievement_steps"]["shelter_complete"] == 115
    assert payload["summary"]["achievement_steps"]["first_storm_survived"] == 225
    assert payload["summary"]["achievement_steps"]["shared_food_transfer"] == 257


def test_vertical_slice_export_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    export_vertical_slice(first_path)
    export_vertical_slice(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
