"""Deterministic scientific canaries over authenticated checkpoint trajectories."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, TypeAlias

from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CheckpointObservation,
    GroupSelector,
)
from infinite_rulebook.orchestration.hashing import scientific_hash


class CanaryKind(StrEnum):
    FRONTIER_IDENTITY = "frontier-semantic-identity"
    METRIC_IDENTITY = "metric-trajectory-identity"
    ADDITIVE_SHIFT = "constant-additive-metric-shift"
    EXACT_ZERO = "exact-zero-metric"


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")


def _selector(name: str, value: GroupSelector) -> None:
    if not isinstance(value, GroupSelector):
        raise TypeError(f"{name} must be a GroupSelector")


def _checkpoints(value: tuple[int, ...]) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError("canary checkpoints must be a nonempty tuple")
    if any(
        isinstance(checkpoint, bool)
        or not isinstance(checkpoint, int)
        or checkpoint < 0
        for checkpoint in value
    ):
        raise ValueError("canary checkpoints must be nonnegative integers")
    if value != tuple(sorted(set(value))):
        raise ValueError("canary checkpoints must be sorted and unique")


def _finite(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class FrontierIdentityCanary:
    name: str
    left: GroupSelector
    right: GroupSelector
    checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier("canary name", self.name)
        _selector("left", self.left)
        _selector("right", self.right)
        if self.left == self.right:
            raise ValueError("frontier canary selectors must differ")
        _checkpoints(self.checkpoints)


@dataclass(frozen=True, slots=True)
class MetricTrajectoryIdentityCanary:
    name: str
    metric: str
    left: GroupSelector
    right: GroupSelector
    checkpoints: tuple[int, ...]
    tolerance: float

    def __post_init__(self) -> None:
        _identifier("canary name", self.name)
        _identifier("metric", self.metric)
        _selector("left", self.left)
        _selector("right", self.right)
        if self.left == self.right:
            raise ValueError("metric-identity canary selectors must differ")
        _checkpoints(self.checkpoints)
        object.__setattr__(
            self,
            "tolerance",
            _finite("tolerance", self.tolerance, nonnegative=True),
        )


@dataclass(frozen=True, slots=True)
class ConstantAdditiveMetricCanary:
    name: str
    selector: GroupSelector
    total_metric: str
    base_metric: str
    shift_metric: str
    expected_shift: float
    checkpoints: tuple[int, ...]
    tolerance: float

    def __post_init__(self) -> None:
        _identifier("canary name", self.name)
        _selector("selector", self.selector)
        for name in ("total_metric", "base_metric", "shift_metric"):
            _identifier(name, getattr(self, name))
        if len({self.total_metric, self.base_metric, self.shift_metric}) != 3:
            raise ValueError("additive canary metric names must be distinct")
        _checkpoints(self.checkpoints)
        object.__setattr__(
            self,
            "expected_shift",
            _finite("expected_shift", self.expected_shift),
        )
        object.__setattr__(
            self,
            "tolerance",
            _finite("tolerance", self.tolerance, nonnegative=True),
        )


@dataclass(frozen=True, slots=True)
class ExactZeroMetricCanary:
    name: str
    selector: GroupSelector
    metric: str
    checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier("canary name", self.name)
        _selector("selector", self.selector)
        _identifier("metric", self.metric)
        _checkpoints(self.checkpoints)


CanarySpec: TypeAlias = (
    FrontierIdentityCanary
    | MetricTrajectoryIdentityCanary
    | ConstantAdditiveMetricCanary
    | ExactZeroMetricCanary
)
_SPEC_TYPES = (
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
)


@dataclass(frozen=True, slots=True)
class CanaryPlan:
    name: str
    phase: AnalysisPhase
    canaries: tuple[CanarySpec, ...]

    def __post_init__(self) -> None:
        _identifier("canary plan name", self.name)
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        if (
            not isinstance(self.canaries, tuple)
            or not self.canaries
            or any(not isinstance(canary, _SPEC_TYPES) for canary in self.canaries)
        ):
            raise TypeError("canaries must be a nonempty tuple of canary specs")
        names = tuple(canary.name for canary in self.canaries)
        if len(set(names)) != len(names):
            raise ValueError("registered canary names must be unique")

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self, domain="scientific-canary-plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "scientific-canary-plan",
            "schema_version": 1,
            "name": self.name,
            "phase": self.phase.value,
            "canaries": _payload(self.canaries),
            "scientific_hash": self.scientific_hash,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True, order=True)
class MetricResidual:
    environment_replica: int
    algorithm_replica: int
    checkpoint: int
    residual: float


@dataclass(frozen=True, slots=True, order=True)
class FrontierComparison:
    environment_replica: int
    algorithm_replica: int
    checkpoint: int
    left_semantic_hash: str
    right_semantic_hash: str


@dataclass(frozen=True, slots=True)
class FrontierIdentityResult:
    name: str
    kind: CanaryKind
    passed: bool
    environment_cluster_count: int
    cell_pair_count: int
    checkpoint_count: int
    mismatch_count: int
    comparisons: tuple[FrontierComparison, ...]


@dataclass(frozen=True, slots=True)
class MetricTrajectoryIdentityResult:
    name: str
    kind: CanaryKind
    metric: str
    passed: bool
    tolerance: float
    environment_cluster_count: int
    cell_pair_count: int
    checkpoint_count: int
    mismatch_count: int
    maximum_absolute_error: float
    residuals: tuple[MetricResidual, ...]


@dataclass(frozen=True, slots=True)
class ConstantAdditiveMetricResult:
    name: str
    kind: CanaryKind
    total_metric: str
    base_metric: str
    shift_metric: str
    expected_shift: float
    passed: bool
    tolerance: float
    environment_cluster_count: int
    cell_count: int
    checkpoint_count: int
    mismatch_count: int
    maximum_decomposition_error: float
    maximum_shift_error: float
    decomposition_residuals: tuple[MetricResidual, ...]
    shift_residuals: tuple[MetricResidual, ...]


@dataclass(frozen=True, slots=True)
class ExactZeroMetricResult:
    name: str
    kind: CanaryKind
    metric: str
    passed: bool
    environment_cluster_count: int
    cell_count: int
    checkpoint_count: int
    nonzero_count: int
    maximum_absolute_value: float
    values: tuple[MetricResidual, ...]


CanaryResult: TypeAlias = (
    FrontierIdentityResult
    | MetricTrajectoryIdentityResult
    | ConstantAdditiveMetricResult
    | ExactZeroMetricResult
)


def _payload(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _payload(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CanaryReport:
    phase: AnalysisPhase
    dataset_hash: str
    plan_hash: str
    results: tuple[CanaryResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def _scientific_payload(self) -> dict[str, Any]:
        return {
            "artifact_type": "scientific-canary-report",
            "schema_version": 1,
            "phase": self.phase.value,
            "dataset_hash": self.dataset_hash,
            "plan_hash": self.plan_hash,
            "passed": self.passed,
            "results": _payload(self.results),
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(
            self._scientific_payload(),
            domain="scientific-canary-report",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._scientific_payload(),
            "scientific_hash": self.scientific_hash,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True)
class _SelectedTrajectory:
    group_key: tuple[str, str, str, str]
    observations: dict[tuple[int, int, int], CheckpointObservation]
    pair_keys: tuple[tuple[int, int], ...]
    environment_cluster_count: int


def _select_trajectory(
    dataset: AnalysisDataset,
    selector: GroupSelector,
    checkpoints: tuple[int, ...],
) -> _SelectedTrajectory:
    selected = [
        observation
        for observation in dataset.observations
        if selector.matches(observation)
    ]
    if not selected:
        raise AnalysisError(
            f"canary selector {selector.label!r} matched no checkpoints"
        )
    groups = {
        (
            observation.condition_hash,
            observation.agent_hash,
            observation.environment_kind,
            observation.agent_kind,
        )
        for observation in selected
    }
    if len(groups) != 1:
        raise AnalysisError(
            f"canary selector {selector.label!r} is ambiguous across groups"
        )
    if {observation.phase for observation in selected} != {dataset.phase}:
        raise AnalysisError("canary selection mixes experiment phases")

    observations: dict[tuple[int, int, int], CheckpointObservation] = {}
    all_pair_keys: set[tuple[int, int]] = set()
    for observation in selected:
        pair_key = observation.pair_key
        all_pair_keys.add(pair_key)
        if observation.round_index not in checkpoints:
            continue
        key = (*pair_key, observation.round_index)
        if key in observations:
            raise AnalysisError(
                "canary selector contains duplicate replica/checkpoint cells"
            )
        observations[key] = observation
    missing = [
        (*pair_key, checkpoint)
        for pair_key in sorted(all_pair_keys)
        for checkpoint in checkpoints
        if (*pair_key, checkpoint) not in observations
    ]
    if missing:
        raise AnalysisError(
            f"canary selector {selector.label!r} is missing registered "
            f"replica/checkpoints: {missing}"
        )
    algorithms_by_environment: dict[int, set[int]] = {}
    for environment_replica, algorithm_replica in all_pair_keys:
        algorithms_by_environment.setdefault(environment_replica, set()).add(
            algorithm_replica
        )
    algorithm_sets = {
        frozenset(algorithms) for algorithms in algorithms_by_environment.values()
    }
    if len(algorithm_sets) != 1:
        raise AnalysisError(
            "canary selector does not form a complete environment/algorithm grid"
        )
    return _SelectedTrajectory(
        group_key=next(iter(groups)),
        observations=observations,
        pair_keys=tuple(sorted(all_pair_keys)),
        environment_cluster_count=len(algorithms_by_environment),
    )


def _paired_trajectories(
    dataset: AnalysisDataset,
    left: GroupSelector,
    right: GroupSelector,
    checkpoints: tuple[int, ...],
) -> tuple[_SelectedTrajectory, _SelectedTrajectory]:
    left_group = _select_trajectory(dataset, left, checkpoints)
    right_group = _select_trajectory(dataset, right, checkpoints)
    if left_group.group_key == right_group.group_key:
        raise AnalysisError("canary selectors resolve to the same registered group")
    if left_group.pair_keys != right_group.pair_keys:
        left_only = sorted(set(left_group.pair_keys) - set(right_group.pair_keys))
        right_only = sorted(set(right_group.pair_keys) - set(left_group.pair_keys))
        raise AnalysisError(
            "canary selectors have unmatched replica pairs: "
            f"left_only={left_only}, right_only={right_only}"
        )
    return left_group, right_group


def _frontier_identity(
    dataset: AnalysisDataset,
    spec: FrontierIdentityCanary,
) -> FrontierIdentityResult:
    left, right = _paired_trajectories(
        dataset,
        spec.left,
        spec.right,
        spec.checkpoints,
    )
    comparisons = tuple(
        FrontierComparison(
            environment_replica,
            algorithm_replica,
            checkpoint,
            left.observations[
                (environment_replica, algorithm_replica, checkpoint)
            ].frontier.semantic_hash,
            right.observations[
                (environment_replica, algorithm_replica, checkpoint)
            ].frontier.semantic_hash,
        )
        for environment_replica, algorithm_replica in left.pair_keys
        for checkpoint in spec.checkpoints
    )
    canonical_hash = comparisons[0].left_semantic_hash
    mismatch_count = sum(
        not (item.left_semantic_hash == item.right_semantic_hash == canonical_hash)
        for item in comparisons
    )
    return FrontierIdentityResult(
        name=spec.name,
        kind=CanaryKind.FRONTIER_IDENTITY,
        passed=mismatch_count == 0,
        environment_cluster_count=left.environment_cluster_count,
        cell_pair_count=len(left.pair_keys),
        checkpoint_count=len(spec.checkpoints),
        mismatch_count=mismatch_count,
        comparisons=comparisons,
    )


def _metric_identity(
    dataset: AnalysisDataset,
    spec: MetricTrajectoryIdentityCanary,
) -> MetricTrajectoryIdentityResult:
    left, right = _paired_trajectories(
        dataset,
        spec.left,
        spec.right,
        spec.checkpoints,
    )
    residuals = tuple(
        MetricResidual(
            environment_replica,
            algorithm_replica,
            checkpoint,
            math.fsum(
                (
                    left.observations[
                        (environment_replica, algorithm_replica, checkpoint)
                    ].metric(spec.metric),
                    -right.observations[
                        (environment_replica, algorithm_replica, checkpoint)
                    ].metric(spec.metric),
                )
            ),
        )
        for environment_replica, algorithm_replica in left.pair_keys
        for checkpoint in spec.checkpoints
    )
    errors = tuple(abs(item.residual) for item in residuals)
    mismatch_count = sum(error > spec.tolerance for error in errors)
    return MetricTrajectoryIdentityResult(
        name=spec.name,
        kind=CanaryKind.METRIC_IDENTITY,
        metric=spec.metric,
        passed=mismatch_count == 0,
        tolerance=spec.tolerance,
        environment_cluster_count=left.environment_cluster_count,
        cell_pair_count=len(left.pair_keys),
        checkpoint_count=len(spec.checkpoints),
        mismatch_count=mismatch_count,
        maximum_absolute_error=max(errors),
        residuals=residuals,
    )


def _additive_shift(
    dataset: AnalysisDataset,
    spec: ConstantAdditiveMetricCanary,
) -> ConstantAdditiveMetricResult:
    selected = _select_trajectory(dataset, spec.selector, spec.checkpoints)
    decomposition_residuals = []
    shift_residuals = []
    for environment_replica, algorithm_replica in selected.pair_keys:
        for checkpoint in spec.checkpoints:
            observation = selected.observations[
                (environment_replica, algorithm_replica, checkpoint)
            ]
            total = observation.metric(spec.total_metric)
            base = observation.metric(spec.base_metric)
            shift = observation.metric(spec.shift_metric)
            decomposition_residuals.append(
                MetricResidual(
                    environment_replica,
                    algorithm_replica,
                    checkpoint,
                    math.fsum((total, -base, -shift)),
                )
            )
            shift_residuals.append(
                MetricResidual(
                    environment_replica,
                    algorithm_replica,
                    checkpoint,
                    math.fsum((shift, -spec.expected_shift)),
                )
            )
    decomposition = tuple(decomposition_residuals)
    shifts = tuple(shift_residuals)
    decomposition_errors = tuple(abs(item.residual) for item in decomposition)
    shift_errors = tuple(abs(item.residual) for item in shifts)
    mismatch_count = sum(
        error > spec.tolerance for error in (*decomposition_errors, *shift_errors)
    )
    return ConstantAdditiveMetricResult(
        name=spec.name,
        kind=CanaryKind.ADDITIVE_SHIFT,
        total_metric=spec.total_metric,
        base_metric=spec.base_metric,
        shift_metric=spec.shift_metric,
        expected_shift=spec.expected_shift,
        passed=mismatch_count == 0,
        tolerance=spec.tolerance,
        environment_cluster_count=selected.environment_cluster_count,
        cell_count=len(selected.pair_keys),
        checkpoint_count=len(spec.checkpoints),
        mismatch_count=mismatch_count,
        maximum_decomposition_error=max(decomposition_errors),
        maximum_shift_error=max(shift_errors),
        decomposition_residuals=decomposition,
        shift_residuals=shifts,
    )


def _exact_zero(
    dataset: AnalysisDataset,
    spec: ExactZeroMetricCanary,
) -> ExactZeroMetricResult:
    selected = _select_trajectory(dataset, spec.selector, spec.checkpoints)
    values = tuple(
        MetricResidual(
            environment_replica,
            algorithm_replica,
            checkpoint,
            selected.observations[
                (environment_replica, algorithm_replica, checkpoint)
            ].metric(spec.metric),
        )
        for environment_replica, algorithm_replica in selected.pair_keys
        for checkpoint in spec.checkpoints
    )
    absolute_values = tuple(abs(item.residual) for item in values)
    nonzero_count = sum(item.residual != 0.0 for item in values)
    return ExactZeroMetricResult(
        name=spec.name,
        kind=CanaryKind.EXACT_ZERO,
        metric=spec.metric,
        passed=nonzero_count == 0,
        environment_cluster_count=selected.environment_cluster_count,
        cell_count=len(selected.pair_keys),
        checkpoint_count=len(spec.checkpoints),
        nonzero_count=nonzero_count,
        maximum_absolute_value=max(absolute_values),
        values=values,
    )


def evaluate_canaries(
    dataset: AnalysisDataset,
    plan: CanaryPlan,
) -> CanaryReport:
    """Evaluate registered deterministic gates without inferential p-values."""

    if not isinstance(dataset, AnalysisDataset):
        raise TypeError("dataset must be an authenticated AnalysisDataset")
    if not isinstance(plan, CanaryPlan):
        raise TypeError("plan must be a CanaryPlan")
    phases = {observation.phase for observation in dataset.observations}
    if phases != {dataset.phase}:
        raise AnalysisError("scientific canaries cannot mix experiment phases")
    if dataset.phase is not plan.phase:
        raise AnalysisError(
            f"canary plan phase {plan.phase.value!r} does not match "
            f"dataset phase {dataset.phase.value!r}"
        )
    results: list[CanaryResult] = []
    for spec in plan.canaries:
        if isinstance(spec, FrontierIdentityCanary):
            result = _frontier_identity(dataset, spec)
        elif isinstance(spec, MetricTrajectoryIdentityCanary):
            result = _metric_identity(dataset, spec)
        elif isinstance(spec, ConstantAdditiveMetricCanary):
            result = _additive_shift(dataset, spec)
        elif isinstance(spec, ExactZeroMetricCanary):
            result = _exact_zero(dataset, spec)
        else:
            raise TypeError(f"unsupported canary spec: {type(spec).__name__}")
        results.append(result)
    return CanaryReport(
        phase=dataset.phase,
        dataset_hash=dataset.scientific_hash,
        plan_hash=plan.scientific_hash,
        results=tuple(results),
    )


__all__ = [
    "CanaryKind",
    "CanaryPlan",
    "CanaryReport",
    "CanaryResult",
    "CanarySpec",
    "ConstantAdditiveMetricCanary",
    "ConstantAdditiveMetricResult",
    "ExactZeroMetricCanary",
    "ExactZeroMetricResult",
    "FrontierComparison",
    "FrontierIdentityCanary",
    "FrontierIdentityResult",
    "MetricResidual",
    "MetricTrajectoryIdentityCanary",
    "MetricTrajectoryIdentityResult",
    "evaluate_canaries",
]
