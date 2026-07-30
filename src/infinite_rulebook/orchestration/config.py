"""Frozen, versioned experiment configuration models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    ConfirmatoryFreezeRecord,
    confirmatory_freeze_from_dict,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.jsonio import load_json_strict

CONFIG_SCHEMA_VERSION = 1
SYMBOLIC_ADAPTER_CONTRACT_V1 = "exact-symbolic-adapter.v1"
SYMBOLIC_ADAPTER_CONTRACT_V2 = "exact-symbolic-adapter.v2"
SYMBOLIC_ADAPTER_CONTRACT_VERSION = SYMBOLIC_ADAPTER_CONTRACT_V1
SYMBOLIC_V2_CALIBRATION_EXPERIMENT_NAME = "symbolic-construct-calibration-v2"
SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME = "symbolic-construct-confirmatory-v2"
REPRODUCIBILITY_OPERATIONAL_DIRECTORY = ".infinite-rulebook-reproducibility"
RESERVED_EXPERIMENT_NAMES = frozenset(
    {"_frontiers", REPRODUCIBILITY_OPERATIONAL_DIRECTORY}
)


def registered_symbolic_v2_phase(name: str) -> str | None:
    """Return the phase bound to an exact registered v2 experiment name."""

    if name == SYMBOLIC_V2_CALIBRATION_EXPERIMENT_NAME:
        return "calibration"
    if name == SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME:
        return "confirmatory"
    return None


def symbolic_adapter_contract(name: str) -> str:
    """Resolve the adapter contract without changing the legacy name space."""

    if registered_symbolic_v2_phase(name) is not None:
        return SYMBOLIC_ADAPTER_CONTRACT_V2
    return SYMBOLIC_ADAPTER_CONTRACT_V1


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
    RELEVANT_INFORMATION = "relevant-information"
    SCHEDULED = "scheduled"


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    kind: EnvironmentKind
    projection_size: int = 2
    core_dimensions: int = 2
    max_redundant_support: int = 2
    distractor_dimensions: int = 2
    public_reward_cap: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EnvironmentKind):
            raise TypeError("kind must be an EnvironmentKind")
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
        if self.kind is EnvironmentKind.RED_C and value == 0:
            raise ValueError("RED-C max_redundant_support must be positive")
        if (
            isinstance(self.public_reward_cap, bool)
            or not isinstance(self.public_reward_cap, (int, float))
            or not math.isfinite(self.public_reward_cap)
            or self.public_reward_cap < 0
        ):
            raise ValueError("public_reward_cap must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    protocol: str = "P1"
    epsilon: float = 0.1
    query_budget: int = 1

    def __post_init__(self) -> None:
        if self.protocol != "P1":
            raise ValueError("the symbolic pilot currently supports protocol P1")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(self.epsilon)
            or not 0 <= self.epsilon < 1
        ):
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
    growth_step: int | None = None
    growth_interval: int | None = None
    maximum_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AgentKind):
            raise TypeError("kind must be an AgentKind")
        if (
            isinstance(self.target_size, bool)
            or not isinstance(self.target_size, int)
            or self.target_size < 1
        ):
            raise ValueError("target_size must be a positive integer")
        schedule = (self.growth_step, self.growth_interval, self.maximum_size)
        if self.kind is AgentKind.SCHEDULED:
            if any(value is None for value in schedule):
                raise ValueError(
                    "scheduled agents require growth_step, growth_interval, "
                    "and maximum_size"
                )
            for name in ("growth_step", "growth_interval", "maximum_size"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if self.maximum_size <= self.target_size:
                raise ValueError("scheduled agent maximum_size must exceed target_size")
        elif any(value is not None for value in schedule):
            raise ValueError("schedule parameters are only valid for scheduled agents")


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
class SolverConfig:
    """Numerical settings recorded in, and sealed with, experiment configs."""

    tolerance: float = 1e-9
    bound_tolerance: float = 1e-7
    lagrangian_tolerance: float = 1e-12
    max_iterations: int = 96
    lagrangian_max_iterations: int = 100_000
    reward_grid_points: int = 3
    contract_version: str = "certified-finite-frontier.v1"

    def __post_init__(self) -> None:
        for name in ("tolerance", "bound_tolerance", "lagrangian_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_iterations",
            "lagrangian_max_iterations",
            "reward_grid_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.reward_grid_points < 3:
            raise ValueError("reward_grid_points must be at least 3")
        if not isinstance(self.contract_version, str) or not self.contract_version:
            raise ValueError("contract_version must not be empty")


@dataclass(frozen=True, slots=True)
class RunCell:
    environment: EnvironmentConfig
    feedback: FeedbackConfig
    reward: RewardConfig
    agent: AgentConfig
    solver: SolverConfig
    environment_replica: int
    algorithm_replica: int

    def __post_init__(self) -> None:
        for name, expected in (
            ("environment", EnvironmentConfig),
            ("feedback", FeedbackConfig),
            ("reward", RewardConfig),
            ("agent", AgentConfig),
            ("solver", SolverConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        for name in ("environment_replica", "algorithm_replica"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

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
    algorithm_master_seed: int | str | None = None
    feedback: FeedbackConfig = FeedbackConfig()
    reward: RewardConfig = RewardConfig()
    solver: SolverConfig = SolverConfig()
    environment_replicas: int = 1
    algorithm_replicas: int = 1
    phase: str = "pilot"
    schema_version: int = CONFIG_SCHEMA_VERSION
    confirmatory_freeze: ConfirmatoryFreezeRecord | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported config schema_version: {self.schema_version}"
            )
        if not isinstance(self.name, str) or not self.name or not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.name in {".", ".."} or "/" in self.name or "\\" in self.name:
            raise ValueError("experiment name must be a single safe path component")
        if self.name in RESERVED_EXPERIMENT_NAMES:
            raise ValueError("experiment name collides with a reserved artifact path")
        if self.phase not in {"pilot", "calibration", "confirmatory"}:
            raise ValueError("phase must be 'pilot', 'calibration', or 'confirmatory'")
        registered_v2_phase = registered_symbolic_v2_phase(self.name)
        if registered_v2_phase is not None and self.phase != registered_v2_phase:
            raise ValueError(f"{self.name} requires phase={registered_v2_phase!r}")
        if self.phase == "confirmatory":
            if self.confirmatory_freeze is None:
                raise ConfirmatoryFreezeError(
                    "phase='confirmatory' requires a valid confirmatory freeze record"
                )
            self.confirmatory_freeze.verify_config(self)
        elif self.confirmatory_freeze is not None:
            raise ConfirmatoryFreezeError(
                "pilot and calibration configs must not contain a confirmatory seal"
            )
        if (
            not isinstance(self.environments, tuple)
            or not isinstance(self.agents, tuple)
            or not self.environments
            or not self.agents
        ):
            raise ValueError("experiments require environments and agents")
        if any(not isinstance(value, EnvironmentConfig) for value in self.environments):
            raise TypeError("environments must contain EnvironmentConfig values")
        if any(not isinstance(value, AgentConfig) for value in self.agents):
            raise TypeError("agents must contain AgentConfig values")
        if not isinstance(self.checkpoints, CheckpointConfig):
            raise TypeError("checkpoints must be a CheckpointConfig")
        if not isinstance(self.feedback, FeedbackConfig):
            raise TypeError("feedback must be a FeedbackConfig")
        if not isinstance(self.reward, RewardConfig):
            raise TypeError("reward must be a RewardConfig")
        if not isinstance(self.solver, SolverConfig):
            raise TypeError("solver must be a SolverConfig")
        if len(set(self.environments)) != len(self.environments):
            raise ValueError("duplicate environment configs are not allowed")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("duplicate agent configs are not allowed")
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
        if isinstance(self.algorithm_master_seed, bool) or (
            self.algorithm_master_seed is not None
            and not isinstance(self.algorithm_master_seed, (int, str))
        ):
            raise TypeError("algorithm_master_seed must be an integer, string, or None")
        if self.feedback.epsilon >= (self.reward.q - 1) / self.reward.q:
            raise ValueError(
                "epsilon must define an informative q-ary feedback channel"
            )

    def freeze_payload(self) -> dict[str, Any]:
        """Return the canonical config payload without a confirmatory seal."""

        payload = asdict(self)
        payload.pop("confirmatory_freeze")
        return payload

    def resolved_dict(self) -> dict[str, Any]:
        payload = self.freeze_payload()
        if self.confirmatory_freeze is not None:
            payload["confirmatory_freeze"] = self.confirmatory_freeze.to_dict()
        return payload

    @property
    def confirmatory_frozen(self) -> bool:
        return (
            self.confirmatory_freeze is not None
            and self.confirmatory_freeze.confirmatory_frozen
        )

    def resolved_run_settings(self) -> dict[str, Any]:
        """Return settings shared by cells, excluding unrelated sweep members."""

        adapter_contract = symbolic_adapter_contract(self.name)
        settings = {
            "schema_version": self.schema_version,
            "adapter_contract": adapter_contract,
            "phase": self.phase,
            "horizon": self.horizon,
            "checkpoints": asdict(self.checkpoints),
            "master_seed": self.master_seed,
            "algorithm_master_seed": self.effective_algorithm_master_seed,
        }
        if adapter_contract == SYMBOLIC_ADAPTER_CONTRACT_V2:
            settings["experiment_name"] = self.name
        if self.confirmatory_freeze is not None:
            settings["confirmatory_frozen"] = True
            settings["confirmatory_freeze_hash"] = self.confirmatory_freeze.seal_hash
            settings["analysis_registration_hash"] = (
                self.confirmatory_freeze.analysis_version
            )
        return settings

    @property
    def effective_algorithm_master_seed(self) -> int | str:
        """Return the explicit nuisance-bank seed, or the legacy phase seed."""

        if self.algorithm_master_seed is None:
            return self.master_seed
        return self.algorithm_master_seed

    @property
    def config_hash(self) -> str:
        return scientific_hash(self.resolved_dict(), domain="experiment-config")

    def cells(self) -> tuple[RunCell, ...]:
        return _experiment_cells(self)

    def contains_cell(self, cell: object) -> bool:
        """Return whether ``cell`` is one exact member of this experiment grid."""

        return (
            isinstance(cell, RunCell)
            and cell.environment in self.environments
            and cell.feedback == self.feedback
            and cell.reward == self.reward
            and cell.agent in self.agents
            and cell.solver == self.solver
            and cell.environment_replica < self.environment_replicas
            and cell.algorithm_replica < self.algorithm_replicas
        )


@lru_cache(maxsize=16)
def _experiment_cells(experiment: ExperimentConfig) -> tuple[RunCell, ...]:
    cells = (
        RunCell(
            environment,
            experiment.feedback,
            experiment.reward,
            agent,
            experiment.solver,
            environment_replica,
            algorithm_replica,
        )
        for environment in experiment.environments
        for agent in experiment.agents
        for environment_replica in range(experiment.environment_replicas)
        for algorithm_replica in range(experiment.algorithm_replicas)
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
    if "solver" in values:
        values["solver"] = _model(SolverConfig, values["solver"])
    if "confirmatory_freeze" in values:
        values["confirmatory_freeze"] = confirmatory_freeze_from_dict(
            values["confirmatory_freeze"]
        )
    checkpoint_values = values["checkpoints"]
    if isinstance(checkpoint_values, dict) and "rounds" in checkpoint_values:
        checkpoint_values = {
            **checkpoint_values,
            "rounds": tuple(checkpoint_values["rounds"]),
        }
    values["checkpoints"] = _model(CheckpointConfig, checkpoint_values)
    return ExperimentConfig(**values)


def run_cell_from_dict(raw: object) -> RunCell:
    """Reconstruct one strict run cell from its persisted JSON representation."""

    if not isinstance(raw, dict):
        raise TypeError("run cell must be an object")
    names = {field.name for field in fields(RunCell)}
    unknown = set(raw) - names
    missing = names - set(raw)
    if unknown:
        raise ValueError(f"unknown RunCell fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing RunCell fields: {sorted(missing)}")
    values = dict(raw)
    values["environment"] = _model(
        EnvironmentConfig,
        values["environment"],
        enums={"kind": EnvironmentKind},
    )
    values["feedback"] = _model(FeedbackConfig, values["feedback"])
    values["reward"] = _model(RewardConfig, values["reward"])
    values["agent"] = _model(
        AgentConfig,
        values["agent"],
        enums={"kind": AgentKind},
    )
    values["solver"] = _model(SolverConfig, values["solver"])
    return RunCell(**values)


def run_cell_identity_payload(raw: object) -> dict[str, Any]:
    """Restore typed identity values while preserving the persisted key schema."""

    if not isinstance(raw, dict):
        raise TypeError("run cell must be an object")
    typed = asdict(run_cell_from_dict(raw))

    def project(value: Any, template: Any) -> Any:
        if isinstance(template, dict):
            return {key: project(value[key], template[key]) for key in template}
        if isinstance(template, list):
            return [
                project(item, template_item)
                for item, template_item in zip(value, template, strict=True)
            ]
        return value

    return project(typed, raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return experiment_config_from_dict(
        load_json_strict(path, label="experiment config")
    )
