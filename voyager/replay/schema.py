"""Versioned contracts and validation for Voyager saved replays."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPLAY_SCHEMA_VERSION = "stage6_replay_2.0.0"
LEGACY_REPLAY_SCHEMA_VERSION = "stage6_replay_1.0.0"
TIMELINE_CHUNK_SIZE = 100
SNAPSHOT_INTERVAL = 25


class ExtensibleModel(BaseModel):
    """Core fields are checked while future minor fields remain loadable."""

    model_config = ConfigDict(extra="allow")


class ArtifactEntry(ExtensibleModel):
    path: str
    kind: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    start_tick: int | None = Field(default=None, ge=0)
    end_tick: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> ArtifactEntry:
        if (
            self.start_tick is not None
            and self.end_tick is not None
            and self.start_tick > self.end_tick
        ):
            raise ValueError("Artifact start_tick cannot exceed end_tick.")
        return self


class ReplaySourceMetadata(ExtensibleModel):
    policy_kind: str
    policy_id: str
    inference_mode: str | None = None
    evaluation_seed: int = Field(ge=0)
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    training_seed: int | None = None
    git_revision: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    benchmark_id: str | None = None
    benchmark_episode: dict[str, Any] | None = None
    source_fingerprint: str


class ReplayVersions(ExtensibleModel):
    replay: str = REPLAY_SCHEMA_VERSION
    environment: str
    scenario: str
    reward: str
    observation: str
    action: str
    achievement: str


class ReplayManifest(ExtensibleModel):
    replay_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    versions: ReplayVersions
    status: Literal["complete", "partial", "failed"]
    source: ReplaySourceMetadata
    environment_config: dict[str, int | float | bool | str]
    tick_rate: int = Field(gt=0)
    world_steps: int = Field(ge=0)
    agent_steps: int = Field(ge=0)
    recorded_at: str
    tags: list[str] = Field(default_factory=list)
    terminal_summary: dict[str, Any]
    registries: dict[str, list[dict[str, Any]]]
    artifacts: list[ArtifactEntry]
    extensions: dict[str, Any] = Field(default_factory=dict)
    manifest_sha256: str | None = None

    @model_validator(mode="after")
    def validate_version_and_artifacts(self) -> ReplayManifest:
        major = replay_major(self.versions.replay)
        if major != 2:
            raise ValueError(
                f"Unsupported replay major version {major}; this loader supports major 2."
            )
        paths = [entry.path for entry in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("Replay manifest contains duplicate artifact paths.")
        if any(path.startswith(("/", "\\")) or ".." in path.split("/") for path in paths):
            raise ValueError("Replay artifact paths must remain inside the replay directory.")
        return self


def replay_major(version: str) -> int:
    prefix = "stage6_replay_"
    if not version.startswith(prefix):
        raise ValueError(f"Unrecognized replay version {version!r}.")
    try:
        return int(version[len(prefix) :].split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise ValueError(f"Unrecognized replay version {version!r}.") from exc


def validate_manifest(payload: object) -> ReplayManifest:
    if not isinstance(payload, dict):
        raise TypeError("Replay manifest must be a JSON object.")
    versions = payload.get("versions")
    if not isinstance(versions, dict):
        raise TypeError("Replay manifest is missing versions.")
    replay_major(str(versions.get("replay", "")))
    return ReplayManifest.model_validate(payload)
