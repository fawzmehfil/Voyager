"""Pydantic contracts for Stage 5.6 benchmark inputs and episode records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voyager.versions import BENCHMARK_SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioSpec(StrictModel):
    id: str
    environment_version: str
    reward_version: str
    observation_version: str
    action_version: str
    achievement_version: str
    config: dict[str, int | float]


class SeedSuite(StrictModel):
    id: str
    seeds: list[int]

    @model_validator(mode="after")
    def validate_seeds(self) -> SeedSuite:
        if not self.seeds:
            raise ValueError("Seed suite must contain at least one seed.")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seed suite contains duplicate seeds.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Evaluation seeds must be non-negative.")
        return self


class PolicySpec(StrictModel):
    id: str
    kind: Literal["random", "greedy", "cooperative", "ppo"]
    official: bool = True
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    training_seed: int | None = None
    inference_mode: Literal["deterministic", "stochastic"] | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> PolicySpec:
        checkpoint_fields = (
            self.checkpoint,
            self.checkpoint_sha256,
            self.training_seed,
            self.inference_mode,
        )
        if self.kind == "ppo" and any(value is None for value in checkpoint_fields):
            raise ValueError(f"PPO policy {self.id!r} requires checkpoint metadata.")
        if self.kind != "ppo" and any(value is not None for value in checkpoint_fields):
            raise ValueError(f"Non-PPO policy {self.id!r} cannot declare a checkpoint.")
        if self.kind == "ppo":
            expected_official = self.inference_mode == "deterministic"
            if self.official != expected_official:
                raise ValueError("Only deterministic PPO policies may be official.")
        return self


class BootstrapSpec(StrictModel):
    samples: int = Field(default=10_000, ge=100)
    seed: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class BenchmarkManifest(StrictModel):
    schema_version: str
    benchmark_id: str
    scenario: ScenarioSpec
    seed_suite: SeedSuite
    policies: list[PolicySpec]
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)
    reference_output: str | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> BenchmarkManifest:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported benchmark schema {self.schema_version!r}; "
                f"expected {BENCHMARK_SCHEMA_VERSION!r}."
            )
        policy_ids = [policy.id for policy in self.policies]
        if not policy_ids:
            raise ValueError("Benchmark must include at least one policy.")
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("Benchmark contains duplicate policy ids.")
        return self


class EpisodeRecord(StrictModel):
    schema_version: str
    benchmark_id: str
    scenario_id: str
    seed_suite_id: str
    policy_id: str
    policy_kind: str
    official: bool
    checkpoint: str | None = None
    training_seed: int | None = None
    inference_mode: str | None = None
    seed: int
    world_steps: int
    agent_steps: int
    dense_return: float
    achievement_return: float
    survivors: int
    deaths: int
    survival_rate: float
    shelter_progress: float
    shelter_completion_step: int | None
    camp_stockpile: dict[str, int]
    achievements: list[str]
    achievement_steps: dict[str, int]
    resource_flow: dict[str, Any]
    action_counts: dict[str, int]
    action_counts_by_role: dict[str, dict[str, int]]
    selected_invalid_actions: int
    raw_invalid_actions: int
    invalid_probability_mass: float
    dense_reward_components: dict[str, float]
    dense_reward_components_by_role: dict[str, dict[str, float]]
    curves: dict[str, list[float]]

    @model_validator(mode="after")
    def validate_version(self) -> EpisodeRecord:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("Episode record schema does not match the current benchmark.")
        return self
