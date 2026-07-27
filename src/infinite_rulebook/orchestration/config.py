"""Frozen, versioned experiment configuration models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeVar

from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.orchestration.hashing import scientific_hash

CONFIG_SCHEMA_VERSION = 1


class EnvironmentKind(StrEnum):
    IND = "IND"
    RED_C = "RED-C"
    MIX = "MIX"
    ALEA = "ALEA"
    TRIVIA = "TRIVIA"
    PUBLIC_C = "PUBLIC-C"


class AgentKind(StrEnum):
    FIXED = "fixed"
    REWARD = "reward"
    NOVELTY = "novelty"
    TOTAL_INFORMATION = "total-information"


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    kind: EnvironmentKind
    projection_size: int = 2
    core_dimensions: int = 2
    max_redundant_support: int = 2
    distractor_dimensions: int = 2
    public_reward_cap: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "projection_size",
            "core_dimensions",
            "distractor_dimensions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        value = self.max_redundant_support
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("max_redundant_support must be a nonnegative integer")
        if not math.isfinite(self.public_reward_cap) or self.public_reward_cap < 0:
            raise ValueError("public_reward_cap must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    protocol: str = "P1"
    epsilon: float = 0.1
    query_budget: int = 1

    def __post_init__(self) -> None:
        if self.protocol != "P1":
            raise ValueError("the symbolic pilot currently supports protocol P1")
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must satisfy 0 <= epsilon < 1")
        if (
            isinstance(self.query_budget, bool)
            or not isinstance(self.query_budget, int)
            or self.query_budget < 1
        ):
            raise ValueError("query_budget must be a positive integer")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    q: int = 4
    u: float = 1.0
    c: float = 1.0
    units: str = "nats"

    def __post_init__(self) -> None:
        RewardSpec(q=self.q, u=self.u, c=self.c)
        if self.units != "nats":
            raise ValueError("stored scientific information units must be nats")

    def to_spec(self) -> RewardSpec:
        return RewardSpec(q=self.q, u=self.u, c=self.c)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    kind: AgentKind
    target_size: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_size, bool)
            or not isinstance(self.target_size, int)
            or self.target_size < 1
        ):
            raise ValueError("target_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    rounds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("at least one checkpoint is required")
        if any(
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
            for round_index in self.rounds
        ):
            raise ValueError("checkpoint rounds must be nonnegative integers")
        if tuple(sorted(set(self.rounds))) != self.rounds:
            raise ValueError("checkpoint rounds must be sorted and unique")


@dataclass(frozen=True, slots=True)
class RunCell:
    environment: EnvironmentConfig
    feedback: FeedbackConfig
    reward: RewardConfig
    agent: AgentConfig
    environment_replica: int
    algorithm_replica: int

    @property
    def cell_hash(self) -> str:
        return scientific_hash(asdict(self), domain="run-cell")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    environments: tuple[EnvironmentConfig, ...]
    agents: tuple[AgentConfig, ...]
    checkpoints: CheckpointConfig
    horizon: int
    master_seed: int | str
    feedback: FeedbackConfig = FeedbackConfig()
    reward: RewardConfig = RewardConfig()
    environment_replicas: int = 1
    algorithm_replicas: int = 1
    phase: str = "pilot"
    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config schema_version: {self.schema_version}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.phase != "pilot":
            raise ValueError("this runner does not freeze confirmatory configurations")
        if not self.environments or not self.agents:
            raise ValueError("experiments require environments and agents")
        for name in ("horizon", "environment_replicas", "algorithm_replicas"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.checkpoints.rounds[-1] > self.horizon:
            raise ValueError("checkpoint rounds must not exceed horizon")
        if isinstance(self.master_seed, bool) or not isinstance(
            self.master_seed, (int, str)
        ):
            raise TypeError("master_seed must be an integer or string")
        if self.feedback.epsilon >= (self.reward.q - 1) / self.reward.q:
            raise ValueError(
                "epsilon must define an informative q-ary feedback channel"
            )

    def resolved_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_run_settings(self) -> dict[str, Any]:
        """Return settings shared by cells, excluding unrelated sweep members."""

        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "horizon": self.horizon,
            "checkpoints": asdict(self.checkpoints),
            "master_seed": self.master_seed,
        }

    @property
    def config_hash(self) -> str:
        return scientific_hash(self.resolved_dict(), domain="experiment-config")

    def cells(self) -> tuple[RunCell, ...]:
        cells = (
            RunCell(environment, self.feedback, self.reward, agent, env_rep, alg_rep)
            for environment in self.environments
            for agent in self.agents
            for env_rep in range(self.environment_replicas)
            for alg_rep in range(self.algorithm_replicas)
        )
        return tuple(sorted(cells, key=lambda cell: cell.cell_hash))


T = TypeVar("T")


def _model(
    model: type[T],
    raw: object,
    *,
    enums: dict[str, type[Enum]] | None = None,
) -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"{model.__name__} must be an object")
    names = {field.name for field in fields(model)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {model.__name__} fields: {sorted(unknown)}")
    values = dict(raw)
    for name, enum in (enums or {}).items():
        if name in values:
            values[name] = enum(values[name])
    return model(**values)


def experiment_config_from_dict(raw: object) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise TypeError("experiment config must be an object")
    names = {field.name for field in fields(ExperimentConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown ExperimentConfig fields: {sorted(unknown)}")
    values = dict(raw)
    values["environments"] = tuple(
        _model(EnvironmentConfig, item, enums={"kind": EnvironmentKind})
        for item in values.get("environments", ())
    )
    values["agents"] = tuple(
        _model(AgentConfig, item, enums={"kind": AgentKind})
        for item in values.get("agents", ())
    )
    if "feedback" in values:
        values["feedback"] = _model(FeedbackConfig, values["feedback"])
    if "reward" in values:
        values["reward"] = _model(RewardConfig, values["reward"])
    checkpoint_values = values["checkpoints"]
    if isinstance(checkpoint_values, dict) and "rounds" in checkpoint_values:
        checkpoint_values = {
            **checkpoint_values,
            "rounds": tuple(checkpoint_values["rounds"]),
        }
    values["checkpoints"] = _model(CheckpointConfig, checkpoint_values)
    return ExperimentConfig(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return experiment_config_from_dict(json.load(stream))
