"""Deterministic cinematic camera direction from recorded replay semantics."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

IMPORTANCE = {
    "death": 100,
    "storm_started": 90,
    "storm_ended": 85,
    "shelter_complete": 82,
    "achievement": 70,
    "shared_food_transfer": 68,
    "build": 58,
    "deposit": 48,
    "gather": 38,
}


def generate_camera_cues(
    events: Iterable[dict[str, Any]],
    *,
    terminal_tick: int,
    camp: dict[str, Any],
    agents: list[dict[str, Any]],
    overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create stable shots, merging nearby events and avoiding rapid cuts."""

    if overrides is not None:
        _validate_overrides(overrides, terminal_tick)
        override_cues = [dict(cue) for cue in overrides]
        return {
            "version": "camera_director_1.0.0",
            "mode": "showcase",
            "cues": override_cues,
        }

    camp_target = {"x": int(camp["x"]), "y": int(camp["y"])}
    candidates: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", "unknown"))
        importance = int(event.get("importance", IMPORTANCE.get(event_type, 20)))
        if importance < 35:
            continue
        position = event.get("position")
        target = (
            {"x": int(position["x"]), "y": int(position["y"])}
            if isinstance(position, dict) and "x" in position and "y" in position
            else camp_target
        )
        candidates.append(
            {
                "tick": int(event["tick"]),
                "importance": importance,
                "target": target,
                "label": _label(event_type, event),
            }
        )
    candidates.sort(key=lambda item: (item["tick"], -item["importance"], item["label"]))

    cues: list[dict[str, Any]] = [
        {
            "start_tick": 0,
            "end_tick": min(24, terminal_tick),
            "kind": "overview",
            "target": _population_centroid(agents, camp_target),
            "zoom": 0.66,
            "label": "The island",
        }
    ]
    last_start = 0
    for candidate in candidates:
        tick = candidate["tick"]
        if tick < 18 or tick - last_start < 24:
            if len(cues) > 1 and candidate["importance"] > cues[-1].get("importance", 0):
                cues[-1].update(
                    target=candidate["target"],
                    label=candidate["label"],
                    importance=candidate["importance"],
                )
            continue
        start = min(tick, terminal_tick)
        end = min(terminal_tick, start + 35)
        if end <= start:
            continue
        cues.append(
            {
                "start_tick": start,
                "end_tick": end,
                "kind": "event",
                "target": candidate["target"],
                "zoom": 1.0 if candidate["importance"] < 85 else 0.78,
                "label": candidate["label"],
                "importance": candidate["importance"],
            }
        )
        last_start = start

    finale_start = max(0, terminal_tick - 24)
    cues = [cue for cue in cues if cue["start_tick"] < finale_start]
    cues.append(
        {
            "start_tick": finale_start,
            "end_tick": terminal_tick,
            "kind": "finale",
            "target": camp_target,
            "zoom": 0.72,
            "label": "The survivors",
        }
    )
    for left, right in pairwise(cues):
        left["end_tick"] = min(int(left["end_tick"]), int(right["start_tick"]) - 1)
    return {"version": "camera_director_1.0.0", "mode": "automatic", "cues": cues}


def _validate_overrides(cues: list[dict[str, Any]], terminal_tick: int) -> None:
    previous_end = -1
    for cue in cues:
        start = int(cue["start_tick"])
        end = int(cue["end_tick"])
        if start < 0 or end < start or end > terminal_tick:
            raise ValueError("Camera override contains an invalid tick range.")
        if start <= previous_end:
            raise ValueError("Camera overrides must be ordered and non-overlapping.")
        target = cue.get("target")
        if not isinstance(target, dict) or not {"x", "y"} <= set(target):
            raise ValueError("Camera override requires an x/y target.")
        previous_end = end


def _population_centroid(
    agents: list[dict[str, Any]], fallback: dict[str, int]
) -> dict[str, float]:
    if not agents:
        return {"x": float(fallback["x"]), "y": float(fallback["y"])}
    return {
        "x": round(sum(float(agent["x"]) for agent in agents) / len(agents), 3),
        "y": round(sum(float(agent["y"]) for agent in agents) / len(agents), 3),
    }


def _label(event_type: str, event: dict[str, Any]) -> str:
    if event_type == "achievement":
        return str(event.get("payload", {}).get("achievement_id", "Achievement unlocked"))
    return event_type.replace("_", " ").title()
