"""Chunk-authenticated compact evidence for v2 deterministic canaries."""

from __future__ import annotations

import csv
import io
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from typing import Any

from infinite_rulebook.analysis.canaries import (
    CanaryKind,
    CanaryPlan,
    ConstantAdditiveMetricCanary,
    ConstantAdditiveMetricResult,
    ExactZeroMetricCanary,
    ExactZeroMetricResult,
    FrontierIdentityCanary,
    FrontierIdentityResult,
    MetricTrajectoryIdentityCanary,
    MetricTrajectoryIdentityResult,
    evaluate_canaries,
)
from infinite_rulebook.analysis.evidence_common import (
    canonical_json,
    checkpoints,
    count,
    exact_selector,
    expected_group_map,
    finite,
    identifier,
    parse_artifact,
    payload,
    record_payload,
    sha256,
    validate_expected_group_inventory,
)
from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CheckpointObservation,
    ExpectedGroup,
    GroupSelector,
)
from infinite_rulebook.orchestration.hashing import scientific_hash

COMPACT_CANARY_DETAIL_LIMIT = 4096
COMPACT_CANARY_FAILURE_LIMIT = 8
COMPACT_CANARY_SCHEMA_VERSION = 1

_PLAN_TYPE = "compact-canary-plan.v2"
_REPORT_TYPE = "compact-canary-report.v2"
_CHUNK_TYPE = "compact-canary-detail-chunk.v2"
_AGGREGATE_KIND = "aggregate-metric-derivation"
_KINDS = {item.value for item in CanaryKind} | {_AGGREGATE_KIND}


@dataclass(frozen=True, slots=True)
class AggregateMetricCanary:
    """Check a cumulative mean against authenticated all-round checkpoints."""

    name: str
    selectors: tuple[GroupSelector, ...]
    aggregate_metric: str
    source_metric: str
    checkpoints: tuple[int, ...]
    tolerance: float

    def __post_init__(self) -> None:
        identifier("canary name", self.name)
        if (
            not isinstance(self.selectors, tuple)
            or not self.selectors
            or any(not isinstance(item, GroupSelector) for item in self.selectors)
        ):
            raise TypeError("selectors must be a nonempty tuple of GroupSelector")
        for item in self.selectors:
            exact_selector("aggregate selector", item)
        ordered = tuple(
            sorted(
                self.selectors,
                key=lambda item: (
                    item.condition_hash or "",
                    item.agent_hash or "",
                    item.environment_kind or "",
                    item.agent_kind or "",
                ),
            )
        )
        object.__setattr__(self, "selectors", ordered)
        if len(set(ordered)) != len(ordered):
            raise ValueError("aggregate selectors must be unique")
        identifier("aggregate_metric", self.aggregate_metric)
        identifier("source_metric", self.source_metric)
        if self.aggregate_metric == self.source_metric:
            raise ValueError("aggregate and source metrics must differ")
        checkpoints(self.checkpoints)
        if self.checkpoints[0] == 0:
            raise ValueError("aggregate checkpoints must be positive")
        object.__setattr__(
            self,
            "tolerance",
            finite("tolerance", self.tolerance, nonnegative=True),
        )


_BaseCanary = (
    FrontierIdentityCanary
    | MetricTrajectoryIdentityCanary
    | ConstantAdditiveMetricCanary
    | ExactZeroMetricCanary
)
_BASE_TYPES = (
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
)


def _base_kind(spec: _BaseCanary) -> str:
    if isinstance(spec, FrontierIdentityCanary):
        return CanaryKind.FRONTIER_IDENTITY.value
    if isinstance(spec, MetricTrajectoryIdentityCanary):
        return CanaryKind.METRIC_IDENTITY.value
    if isinstance(spec, ConstantAdditiveMetricCanary):
        return CanaryKind.ADDITIVE_SHIFT.value
    return CanaryKind.EXACT_ZERO.value


@dataclass(frozen=True, slots=True)
class CompactCanaryPlan:
    name: str
    phase: AnalysisPhase
    canaries: tuple[_BaseCanary, ...]
    aggregate_canaries: tuple[AggregateMetricCanary, ...] = ()
    expected_groups: tuple[ExpectedGroup, ...] = ()

    def __post_init__(self) -> None:
        identifier("plan name", self.name)
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        if (
            not isinstance(self.canaries, tuple)
            or not self.canaries
            or any(not isinstance(item, _BASE_TYPES) for item in self.canaries)
        ):
            raise TypeError("canaries must contain existing deterministic canary specs")
        if not isinstance(self.aggregate_canaries, tuple) or any(
            not isinstance(item, AggregateMetricCanary)
            for item in self.aggregate_canaries
        ):
            raise TypeError("aggregate_canaries must contain AggregateMetricCanary")
        object.__setattr__(
            self,
            "canaries",
            tuple(sorted(self.canaries, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "aggregate_canaries",
            tuple(sorted(self.aggregate_canaries, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "expected_groups",
            tuple(sorted(self.expected_groups)),
        )
        groups = expected_group_map(self.expected_groups)
        for spec in self.canaries:
            selectors = (
                (spec.left, spec.right)
                if isinstance(
                    spec,
                    (FrontierIdentityCanary, MetricTrajectoryIdentityCanary),
                )
                else (spec.selector,)
            )
            for selector in selectors:
                exact_selector("compact canary selector", selector)
                group = groups.get(selector)
                if group is None or not set(spec.checkpoints) <= set(group.checkpoints):
                    raise ValueError(
                        "compact canary selector/checkpoints are outside "
                        "expected_groups"
                    )
        for spec in self.aggregate_canaries:
            if set(spec.selectors) != set(groups) or any(
                not set(spec.checkpoints) <= set(group.checkpoints)
                for group in groups.values()
            ):
                raise ValueError(
                    "aggregate canary must cover every exact expected group"
                )
        names = tuple(item.name for item in (*self.canaries, *self.aggregate_canaries))
        if len(set(names)) != len(names):
            raise ValueError("compact canary names must be unique")

    def _body(self) -> dict[str, Any]:
        return {
            "artifact_type": _PLAN_TYPE,
            "schema_version": COMPACT_CANARY_SCHEMA_VERSION,
            **record_payload(self)["fields"],
            "detail_chunk_record_limit": COMPACT_CANARY_DETAIL_LIMIT,
            "failure_example_limit": COMPACT_CANARY_FAILURE_LIMIT,
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self._body(), domain=_PLAN_TYPE)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompactCanaryDetail:
    gate_name: str
    kind: str
    comparison: str
    group_hash: str
    environment_replica: int
    algorithm_replica: int
    checkpoint: int
    residual: float
    tolerance: float
    violated: bool
    left_semantic_hash: str | None = None
    right_semantic_hash: str | None = None

    def __post_init__(self) -> None:
        identifier("gate_name", self.gate_name)
        if self.kind not in _KINDS:
            raise ValueError("compact detail kind is invalid")
        identifier("comparison", self.comparison)
        sha256("group_hash", self.group_hash)
        count("environment_replica", self.environment_replica)
        count("algorithm_replica", self.algorithm_replica)
        count("checkpoint", self.checkpoint)
        residual = finite("residual", self.residual)
        tolerance = finite("tolerance", self.tolerance, nonnegative=True)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "tolerance", tolerance)
        if not isinstance(self.violated, bool):
            raise TypeError("violated must be a boolean")
        if self.violated != (abs(residual) > tolerance):
            raise ValueError("violated does not match residual and tolerance")
        frontier = self.kind == CanaryKind.FRONTIER_IDENTITY.value
        if frontier and (
            self.comparison != "frontier-semantic-hash"
            or self.left_semantic_hash is None
            or self.right_semantic_hash is None
            or residual not in (0.0, 1.0)
            or tolerance != 0.0
        ):
            raise ValueError("frontier details require exact semantic-hash evidence")
        if not frontier and (
            self.left_semantic_hash is not None or self.right_semantic_hash is not None
        ):
            raise ValueError("semantic hashes are valid only for frontier details")
        if frontier:
            sha256("left_semantic_hash", self.left_semantic_hash)
            sha256("right_semantic_hash", self.right_semantic_hash)

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            self.gate_name,
            self.kind,
            self.comparison,
            self.group_hash,
            self.environment_replica,
            self.algorithm_replica,
            self.checkpoint,
        )


@dataclass(frozen=True, slots=True)
class CompactGateResult:
    name: str
    kind: str
    passed: bool
    environment_cluster_count: int
    cell_count: int
    checkpoint_count: int
    record_count: int
    violation_count: int
    tolerance: float
    minimum_residual: float
    maximum_residual: float
    maximum_absolute_error: float
    violations: tuple[CompactCanaryDetail, ...]

    def __post_init__(self) -> None:
        identifier("result name", self.name)
        if self.kind not in _KINDS:
            raise ValueError("compact result kind is invalid")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        for name in (
            "environment_cluster_count",
            "cell_count",
            "checkpoint_count",
            "record_count",
        ):
            count(name, getattr(self, name), positive=True)
        count("violation_count", self.violation_count)
        if self.violation_count > self.record_count:
            raise ValueError("violation_count exceeds record_count")
        if self.passed != (self.violation_count == 0):
            raise ValueError("passed does not match violation_count")
        tolerance = finite("tolerance", self.tolerance, nonnegative=True)
        minimum = finite("minimum_residual", self.minimum_residual)
        maximum = finite("maximum_residual", self.maximum_residual)
        absolute = finite(
            "maximum_absolute_error",
            self.maximum_absolute_error,
            nonnegative=True,
        )
        if minimum > maximum or absolute != max(abs(minimum), abs(maximum)):
            raise ValueError("residual extrema are not canonical")
        if (self.violation_count == 0) != (absolute <= tolerance):
            raise ValueError("violation count does not match residual extrema")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "minimum_residual", minimum)
        object.__setattr__(self, "maximum_residual", maximum)
        object.__setattr__(self, "maximum_absolute_error", absolute)
        if (
            not isinstance(self.violations, tuple)
            or len(self.violations)
            != min(
                self.violation_count,
                COMPACT_CANARY_FAILURE_LIMIT,
            )
            or any(
                not item.violated
                or item.gate_name != self.name
                or item.kind != self.kind
                or item.tolerance != tolerance
                for item in self.violations
            )
            or self.violations
            != tuple(sorted(self.violations, key=lambda item: item.sort_key))
        ):
            raise ValueError("violations are not the bounded canonical examples")


@dataclass(frozen=True, slots=True)
class DetailChunkReference:
    index: int
    record_count: int
    scientific_hash: str

    def __post_init__(self) -> None:
        count("chunk index", self.index)
        count("chunk record_count", self.record_count, positive=True)
        if self.record_count > COMPACT_CANARY_DETAIL_LIMIT:
            raise ValueError("detail chunk exceeds the registered limit")
        sha256("chunk scientific_hash", self.scientific_hash)


@dataclass(frozen=True, slots=True)
class CompactCanaryDetailChunk:
    index: int
    records: tuple[CompactCanaryDetail, ...]

    def __post_init__(self) -> None:
        count("chunk index", self.index)
        if (
            not isinstance(self.records, tuple)
            or not self.records
            or len(self.records) > COMPACT_CANARY_DETAIL_LIMIT
            or any(not isinstance(item, CompactCanaryDetail) for item in self.records)
        ):
            raise ValueError("detail chunk must contain 1..4096 records")
        if self.records != tuple(sorted(self.records, key=lambda item: item.sort_key)):
            raise ValueError("detail chunk records are not canonical")
        if len({item.sort_key for item in self.records}) != len(self.records):
            raise ValueError("detail chunk contains duplicate identities")

    def _body(self) -> dict[str, Any]:
        return {
            "artifact_type": _CHUNK_TYPE,
            "schema_version": COMPACT_CANARY_SCHEMA_VERSION,
            **record_payload(self)["fields"],
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self._body(), domain=_CHUNK_TYPE)

    @property
    def reference(self) -> DetailChunkReference:
        return DetailChunkReference(
            self.index,
            len(self.records),
            self.scientific_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def detail_inventory_hash(references: tuple[DetailChunkReference, ...]) -> str:
    if not references or tuple(item.index for item in references) != tuple(
        range(len(references))
    ):
        raise ValueError("detail references are not a contiguous inventory")
    return scientific_hash(payload(references), domain="compact-canary-inventory.v2")


@dataclass(frozen=True, slots=True)
class CompactCanaryReport:
    phase: AnalysisPhase
    dataset_hash: str
    plan_hash: str
    results: tuple[CompactGateResult, ...]
    detail_record_count: int
    detail_chunks: tuple[DetailChunkReference, ...]
    detail_root_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        sha256("dataset_hash", self.dataset_hash)
        sha256("plan_hash", self.plan_hash)
        if (
            not isinstance(self.results, tuple)
            or not self.results
            or any(not isinstance(item, CompactGateResult) for item in self.results)
            or self.results != tuple(sorted(self.results, key=lambda item: item.name))
            or len({item.name for item in self.results}) != len(self.results)
        ):
            raise ValueError("results are not a nonempty canonical inventory")
        count("detail_record_count", self.detail_record_count, positive=True)
        if (
            not isinstance(self.detail_chunks, tuple)
            or not self.detail_chunks
            or any(
                not isinstance(item, DetailChunkReference)
                for item in self.detail_chunks
            )
            or tuple(item.index for item in self.detail_chunks)
            != tuple(range(len(self.detail_chunks)))
            or sum(item.record_count for item in self.detail_chunks)
            != self.detail_record_count
            or sum(item.record_count for item in self.results)
            != self.detail_record_count
        ):
            raise ValueError("detail record counts are inconsistent")
        sha256("detail_root_hash", self.detail_root_hash)
        if self.detail_root_hash != detail_inventory_hash(self.detail_chunks):
            raise ValueError("detail_root_hash does not authenticate the inventory")

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    def _body(self) -> dict[str, Any]:
        return {
            "artifact_type": _REPORT_TYPE,
            "schema_version": COMPACT_CANARY_SCHEMA_VERSION,
            **record_payload(self)["fields"],
            "passed": self.passed,
            "detail_chunk_record_limit": COMPACT_CANARY_DETAIL_LIMIT,
            "failure_example_limit": COMPACT_CANARY_FAILURE_LIMIT,
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self._body(), domain=_REPORT_TYPE)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompactCanaryEvidence:
    report: CompactCanaryReport
    detail_chunks: tuple[CompactCanaryDetailChunk, ...]

    @property
    def scientific_hash(self) -> str:
        return self.report.scientific_hash

    @property
    def passed(self) -> bool:
        return self.report.passed

    def __post_init__(self) -> None:
        if (
            not isinstance(self.report, CompactCanaryReport)
            or not isinstance(self.detail_chunks, tuple)
            or any(
                not isinstance(item, CompactCanaryDetailChunk)
                for item in self.detail_chunks
            )
        ):
            raise TypeError("compact evidence requires a report and chunk tuple")
        _validate_detail_chunks(self.report, self.detail_chunks)


@dataclass(frozen=True, slots=True)
class SpooledCompactCanaryEvidence:
    """Bounded-memory compact evidence whose chunks live in an artifact directory."""

    report: CompactCanaryReport

    def __post_init__(self) -> None:
        if not isinstance(self.report, CompactCanaryReport):
            raise TypeError("spooled compact evidence requires a compact report")

    @property
    def scientific_hash(self) -> str:
        return self.report.scientific_hash

    @property
    def passed(self) -> bool:
        return self.report.passed


def _validate_detail_chunks(
    report: CompactCanaryReport,
    chunks: Iterable[CompactCanaryDetailChunk],
) -> None:
    expected_references = iter(report.detail_chunks)
    previous: tuple[object, ...] | None = None

    def records() -> Iterable[CompactCanaryDetail]:
        nonlocal previous
        for chunk in chunks:
            if not isinstance(chunk, CompactCanaryDetailChunk):
                raise TypeError("compact evidence chunks must be typed detail chunks")
            try:
                expected = next(expected_references)
            except StopIteration:
                raise ValueError("detail chunks exceed the report inventory") from None
            if chunk.reference != expected:
                raise ValueError("detail chunks do not match the report inventory")
            for item in chunk.records:
                if previous is not None and item.sort_key <= previous:
                    raise ValueError("detail chunks are not globally canonical")
                previous = item.sort_key
                yield item
        try:
            next(expected_references)
        except StopIteration:
            return
        raise ValueError("detail chunks do not complete the report inventory")

    def validated_summary(
        group: Iterable[CompactCanaryDetail],
    ) -> CompactGateResult:
        iterator = iter(group)
        first = next(iterator)
        canonical = first.left_semantic_hash

        def validated() -> Iterable[CompactCanaryDetail]:
            for item in chain((first,), iterator):
                if (
                    item.kind == CanaryKind.FRONTIER_IDENTITY.value
                    and item.residual
                    != float(
                        not (
                            item.left_semantic_hash
                            == item.right_semantic_hash
                            == canonical
                        )
                    )
                ):
                    raise ValueError(
                        "frontier detail residual contradicts semantic hashes"
                    )
                yield item

        return _summarize_gate(validated())

    summaries = tuple(
        validated_summary(group)
        for _, group in groupby(records(), key=lambda item: item.gate_name)
    )
    if summaries != report.results:
        raise ValueError("gate summary does not match authenticated details")


def _compact_detail(
    *,
    name: str,
    kind: str,
    comparison: str,
    group_hash: str,
    environment: int,
    algorithm: int,
    checkpoint: int,
    residual: float,
    tolerance: float,
    left_hash: str | None = None,
    right_hash: str | None = None,
) -> CompactCanaryDetail:
    return CompactCanaryDetail(
        name,
        kind,
        comparison,
        group_hash,
        environment,
        algorithm,
        checkpoint,
        residual,
        tolerance,
        abs(residual) > tolerance,
        left_hash,
        right_hash,
    )


def _base_details(
    dataset: AnalysisDataset,
    spec: _BaseCanary,
) -> tuple[Iterable[CompactCanaryDetail], int, int, int, bool]:
    result = evaluate_canaries(
        dataset,
        CanaryPlan(
            f"{spec.name}.compact-evaluation",
            dataset.phase,
            (spec,),
        ),
    ).results[0]
    group_hash = scientific_hash(
        record_payload(spec),
        domain="compact-canary-group.v2",
    )
    if isinstance(result, FrontierIdentityResult):
        canonical = result.comparisons[0].left_semantic_hash
        details = (
            _compact_detail(
                name=spec.name,
                kind=result.kind.value,
                comparison="frontier-semantic-hash",
                group_hash=group_hash,
                environment=item.environment_replica,
                algorithm=item.algorithm_replica,
                checkpoint=item.checkpoint,
                residual=float(
                    not (
                        item.left_semantic_hash == item.right_semantic_hash == canonical
                    )
                ),
                tolerance=0.0,
                left_hash=item.left_semantic_hash,
                right_hash=item.right_semantic_hash,
            )
            for item in result.comparisons
        )
        counts = (
            result.environment_cluster_count,
            result.cell_pair_count,
            result.checkpoint_count,
        )
    elif isinstance(result, MetricTrajectoryIdentityResult):
        details = (
            _compact_detail(
                name=spec.name,
                kind=result.kind.value,
                comparison=result.metric,
                group_hash=group_hash,
                environment=item.environment_replica,
                algorithm=item.algorithm_replica,
                checkpoint=item.checkpoint,
                residual=item.residual,
                tolerance=result.tolerance,
            )
            for item in result.residuals
        )
        counts = (
            result.environment_cluster_count,
            result.cell_pair_count,
            result.checkpoint_count,
        )
    elif isinstance(result, ConstantAdditiveMetricResult):
        details = (
            _compact_detail(
                name=spec.name,
                kind=result.kind.value,
                comparison=comparison,
                group_hash=group_hash,
                environment=item.environment_replica,
                algorithm=item.algorithm_replica,
                checkpoint=item.checkpoint,
                residual=item.residual,
                tolerance=result.tolerance,
            )
            for comparison, values in (
                ("decomposition", result.decomposition_residuals),
                ("registered-shift", result.shift_residuals),
            )
            for item in values
        )
        counts = (
            result.environment_cluster_count,
            result.cell_count,
            result.checkpoint_count,
        )
    elif isinstance(result, ExactZeroMetricResult):
        details = (
            _compact_detail(
                name=spec.name,
                kind=result.kind.value,
                comparison=result.metric,
                group_hash=group_hash,
                environment=item.environment_replica,
                algorithm=item.algorithm_replica,
                checkpoint=item.checkpoint,
                residual=item.residual,
                tolerance=0.0,
            )
            for item in result.values
        )
        counts = (
            result.environment_cluster_count,
            result.cell_count,
            result.checkpoint_count,
        )
    else:
        raise TypeError(f"unsupported canary result: {type(result).__name__}")
    return details, *counts, result.passed


def _aggregate_details(
    dataset: AnalysisDataset,
    spec: AggregateMetricCanary,
) -> Iterable[CompactCanaryDetail]:
    observations = dataset.observations
    group_ranges: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    previous_key: tuple[str, str, str, str] | None = None
    start = 0
    for index, item in enumerate(observations):
        key = (
            item.condition_hash,
            item.agent_hash,
            item.environment_kind,
            item.agent_kind,
        )
        if key != previous_key:
            if previous_key is not None:
                if previous_key in group_ranges:
                    raise AnalysisError("aggregate dataset groups are noncontiguous")
                group_ranges[previous_key] = (start, index)
            previous_key = key
            start = index
    if previous_key is None:
        raise AnalysisError("aggregate dataset is empty")
    group_ranges[previous_key] = (start, len(observations))

    claimed_groups = []
    for selector in spec.selectors:
        group_key = (
            selector.condition_hash,
            selector.agent_hash,
            selector.environment_kind,
            selector.agent_kind,
        )
        if group_key not in group_ranges:
            raise AnalysisError(
                f"aggregate selector {selector.label!r} must match exactly one group"
            )
        if group_key in claimed_groups:
            raise AnalysisError("aggregate selectors overlap one registered group")
        claimed_groups.append(group_key)
    if set(claimed_groups) != set(group_ranges):
        raise AnalysisError(
            "aggregate selector inventory omits or adds registered dataset groups"
        )

    ordered_groups = sorted(
        (
            (
                scientific_hash(group_key, domain="compact-canary-group.v2"),
                group_key,
            )
            for group_key in claimed_groups
        ),
        key=lambda item: item[0],
    )
    comparison = f"{spec.aggregate_metric}=mean({spec.source_metric}[1..checkpoint])"
    for group_hash, group_key in ordered_groups:
        histories: dict[
            tuple[int, int],
            dict[int, CheckpointObservation],
        ] = {}
        first, last = group_ranges[group_key]
        for index in range(first, last):
            item = observations[index]
            history = histories.setdefault(item.pair_key, {})
            if item.round_index in history:
                raise AnalysisError(
                    "aggregate group contains duplicate replica/checkpoint cells"
                )
            history[item.round_index] = item
        algorithms: dict[int, set[int]] = {}
        for environment, algorithm in histories:
            algorithms.setdefault(environment, set()).add(algorithm)
        if len({frozenset(items) for items in algorithms.values()}) != 1:
            raise AnalysisError("aggregate selector has an incomplete replica grid")
        for (environment, algorithm), history in sorted(histories.items()):
            for checkpoint in spec.checkpoints:
                source = tuple(
                    history.get(round_index) for round_index in range(1, checkpoint + 1)
                )
                if any(item is None for item in source) or checkpoint not in history:
                    raise AnalysisError(
                        "aggregate derivation is missing authenticated source "
                        "checkpoints"
                    )
                expected = (
                    math.fsum(
                        item.metric(spec.source_metric)
                        for item in source
                        if item is not None
                    )
                    / checkpoint
                )
                residual = math.fsum(
                    (history[checkpoint].metric(spec.aggregate_metric), -expected)
                )
                yield _compact_detail(
                    name=spec.name,
                    kind=_AGGREGATE_KIND,
                    comparison=comparison,
                    group_hash=group_hash,
                    environment=environment,
                    algorithm=algorithm,
                    checkpoint=checkpoint,
                    residual=residual,
                    tolerance=spec.tolerance,
                )


def _summarize_gate(
    details: Iterable[CompactCanaryDetail],
    emit: Callable[[CompactCanaryDetail], None] | None = None,
) -> CompactGateResult:
    iterator = iter(details)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise AnalysisError("compact gate produced no detail records") from error
    name, kind, tolerance = first.gate_name, first.kind, first.tolerance
    environments = {(first.group_hash, first.environment_replica)}
    cells = {(first.group_hash, first.environment_replica, first.algorithm_replica)}
    registered = {first.checkpoint}
    minimum = maximum = first.residual
    record_count = 1
    violation_count = int(first.violated)
    violations = [first] if first.violated else []
    previous = first.sort_key
    if emit is not None:
        emit(first)
    for item in iterator:
        if (
            item.gate_name != name
            or item.kind != kind
            or item.tolerance != tolerance
            or item.sort_key <= previous
        ):
            raise AnalysisError("compact gate details are mixed or noncanonical")
        if emit is not None:
            emit(item)
        environments.add((item.group_hash, item.environment_replica))
        cells.add((item.group_hash, item.environment_replica, item.algorithm_replica))
        registered.add(item.checkpoint)
        minimum = min(minimum, item.residual)
        maximum = max(maximum, item.residual)
        record_count += 1
        if item.violated:
            violation_count += 1
            if len(violations) < COMPACT_CANARY_FAILURE_LIMIT:
                violations.append(item)
        previous = item.sort_key
    return CompactGateResult(
        name,
        kind,
        violation_count == 0,
        len(environments),
        len(cells),
        len(registered),
        record_count,
        violation_count,
        tolerance,
        minimum,
        maximum,
        max(abs(minimum), abs(maximum)),
        tuple(violations),
    )


def _evaluate_compact_canary_report(
    dataset: AnalysisDataset,
    plan: CompactCanaryPlan,
    emit_chunk: Callable[[CompactCanaryDetailChunk], None],
) -> CompactCanaryReport:
    if not isinstance(dataset, AnalysisDataset):
        raise TypeError("dataset must be an AnalysisDataset")
    if not isinstance(plan, CompactCanaryPlan):
        raise TypeError("plan must be a CompactCanaryPlan")
    if dataset.phase is not plan.phase:
        raise AnalysisError("compact canary plan and dataset phases differ")
    validate_expected_group_inventory(dataset, plan.expected_groups)
    results = []
    references = []
    buffer = []
    previous: tuple[object, ...] | None = None

    def flush() -> None:
        chunk = CompactCanaryDetailChunk(len(references), tuple(buffer))
        emit_chunk(chunk)
        references.append(chunk.reference)
        buffer.clear()

    def emit(item: CompactCanaryDetail) -> None:
        nonlocal previous
        if previous is not None and item.sort_key <= previous:
            raise AnalysisError("compact detail stream is not globally canonical")
        buffer.append(item)
        previous = item.sort_key
        if len(buffer) == COMPACT_CANARY_DETAIL_LIMIT:
            flush()

    specs = sorted(
        (*plan.canaries, *plan.aggregate_canaries),
        key=lambda item: item.name,
    )
    for spec in specs:
        if isinstance(spec, AggregateMetricCanary):
            result = _summarize_gate(_aggregate_details(dataset, spec), emit)
            expected = (
                spec.name,
                _AGGREGATE_KIND,
                len(spec.checkpoints),
            )
            observed = (result.name, result.kind, result.checkpoint_count)
        else:
            details, environments, cells, registered, passed = _base_details(
                dataset,
                spec,
            )
            result = _summarize_gate(details, emit)
            expected = (
                spec.name,
                _base_kind(spec),
                environments,
                cells,
                registered,
                passed,
            )
            observed = (
                result.name,
                result.kind,
                result.environment_cluster_count,
                result.cell_count,
                result.checkpoint_count,
                result.passed,
            )
        if observed != expected:
            raise AnalysisError("compact result differs from registered gate")
        results.append(result)
    if buffer:
        flush()
    reference_inventory = tuple(references)
    return CompactCanaryReport(
        dataset.phase,
        dataset.scientific_hash,
        plan.scientific_hash,
        tuple(results),
        sum(item.record_count for item in results),
        reference_inventory,
        detail_inventory_hash(reference_inventory),
    )


def evaluate_compact_canaries(
    dataset: AnalysisDataset,
    plan: CompactCanaryPlan,
) -> CompactCanaryEvidence:
    """Evaluate compact canaries in memory for bounded datasets and unit tests."""

    chunks = []
    report = _evaluate_compact_canary_report(dataset, plan, chunks.append)
    return CompactCanaryEvidence(report, tuple(chunks))


def _detail_chunk_filename(index: int) -> str:
    return f"canary-details-{index:06d}.json"


def compact_canary_artifact_names(
    evidence: CompactCanaryEvidence | SpooledCompactCanaryEvidence,
) -> tuple[str, ...]:
    if not isinstance(
        evidence,
        (CompactCanaryEvidence, SpooledCompactCanaryEvidence),
    ):
        raise TypeError("evidence must be compact canary evidence")
    return (
        "canaries.json",
        *(
            _detail_chunk_filename(reference.index)
            for reference in evidence.report.detail_chunks
        ),
    )


def _write_spooled_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_compact_canaries_spooled(
    dataset: AnalysisDataset,
    plan: CompactCanaryPlan,
    directory: Path,
) -> SpooledCompactCanaryEvidence:
    """Evaluate and write canonical detail chunks with bounded detail memory."""

    if not isinstance(directory, Path):
        raise TypeError("compact evidence directory must be a Path")
    if directory.is_symlink():
        raise ValueError("compact evidence directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("compact evidence directory must be a directory")
    if (directory / "canaries.json").exists() or any(
        directory.glob("canary-details-*.json")
    ):
        raise ValueError("compact evidence output already exists")
    created: list[Path] = []

    def write_chunk(chunk: CompactCanaryDetailChunk) -> None:
        path = directory / _detail_chunk_filename(chunk.index)
        created.append(path)
        _write_spooled_text(path, chunk.to_json())

    try:
        report = _evaluate_compact_canary_report(dataset, plan, write_chunk)
        report_path = directory / "canaries.json"
        created.append(report_path)
        _write_spooled_text(report_path, report.to_json())
        evidence = SpooledCompactCanaryEvidence(report)
        verify_compact_canary_artifact_directory(directory, evidence)
        return evidence
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def compact_canary_artifacts(
    evidence: CompactCanaryEvidence,
) -> tuple[tuple[str, str], ...]:
    """Return deterministic filenames and JSON for one complete evidence bundle."""

    if not isinstance(evidence, CompactCanaryEvidence):
        raise TypeError("evidence must be CompactCanaryEvidence")
    return (
        ("canaries.json", evidence.report.to_json()),
        *(
            (
                _detail_chunk_filename(chunk.index),
                chunk.to_json(),
            )
            for chunk in evidence.detail_chunks
        ),
    )


def verify_compact_canary_artifact_directory(
    directory: Path,
    evidence: CompactCanaryEvidence | SpooledCompactCanaryEvidence,
) -> None:
    """Stream-verify one canonical compact evidence bundle against a report."""

    if not isinstance(directory, Path):
        raise TypeError("compact evidence directory must be a Path")
    expected_names = compact_canary_artifact_names(evidence)
    expected_details = set(expected_names[1:])
    observed_details = {path.name for path in directory.glob("canary-details-*.json")}
    if observed_details != expected_details:
        raise ValueError("compact detail files differ from the report inventory")
    paths = tuple(directory / name for name in expected_names)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("compact evidence artifacts must be regular files")
    report_content = paths[0].read_text(encoding="utf-8")
    report = parse_compact_canary_report_json(report_content)
    if report != evidence.report or report_content != report.to_json():
        raise ValueError("compact canary report differs from raw evidence")

    def chunks() -> Iterable[CompactCanaryDetailChunk]:
        for path, reference in zip(
            paths[1:],
            report.detail_chunks,
            strict=True,
        ):
            content = path.read_text(encoding="utf-8")
            chunk = parse_compact_canary_detail_chunk_json(content)
            if chunk.reference != reference or content != chunk.to_json():
                raise ValueError(
                    "compact detail chunk differs from the report inventory"
                )
            yield chunk

    _validate_detail_chunks(report, chunks())


def compact_canary_results_csv(report: CompactCanaryReport) -> str:
    """Return one bounded summary row per authenticated v2 canary gate."""

    if not isinstance(report, CompactCanaryReport):
        raise TypeError("report must be a CompactCanaryReport")
    fields = (
        "phase",
        "canary_report_hash",
        "dataset_hash",
        "canary_plan_hash",
        "detail_root_hash",
        "all_canaries_passed",
        "name",
        "kind",
        "gate_passed",
        "environment_clusters",
        "cell_count",
        "checkpoint_count",
        "record_count",
        "violation_count",
        "tolerance",
        "minimum_residual",
        "maximum_residual",
        "maximum_absolute_error",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    for result in report.results:
        writer.writerow(
            {
                "phase": report.phase.value,
                "canary_report_hash": report.scientific_hash,
                "dataset_hash": report.dataset_hash,
                "canary_plan_hash": report.plan_hash,
                "detail_root_hash": report.detail_root_hash,
                "all_canaries_passed": str(report.passed).lower(),
                "name": result.name,
                "kind": result.kind,
                "gate_passed": str(result.passed).lower(),
                "environment_clusters": result.environment_cluster_count,
                "cell_count": result.cell_count,
                "checkpoint_count": result.checkpoint_count,
                "record_count": result.record_count,
                "violation_count": result.violation_count,
                "tolerance": repr(result.tolerance),
                "minimum_residual": repr(result.minimum_residual),
                "maximum_residual": repr(result.maximum_residual),
                "maximum_absolute_error": repr(result.maximum_absolute_error),
            }
        )
    return stream.getvalue()


_COMPACT_RECORD_TYPES = (
    AggregateMetricCanary,
    CompactCanaryPlan,
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
    ExpectedGroup,
    GroupSelector,
    CompactCanaryDetail,
    CompactCanaryDetailChunk,
    CompactGateResult,
    DetailChunkReference,
    CompactCanaryReport,
)


def parse_compact_canary_plan_json(content: str) -> CompactCanaryPlan:
    return parse_artifact(
        content,
        label="compact canary plan",
        artifact_type=_PLAN_TYPE,
        schema_version=COMPACT_CANARY_SCHEMA_VERSION,
        record_type=CompactCanaryPlan,
        allowed_types=_COMPACT_RECORD_TYPES,
        fixed={
            "detail_chunk_record_limit": COMPACT_CANARY_DETAIL_LIMIT,
            "failure_example_limit": COMPACT_CANARY_FAILURE_LIMIT,
        },
        derived=set(),
    )


def parse_compact_canary_detail_chunk_json(
    content: str,
) -> CompactCanaryDetailChunk:
    return parse_artifact(
        content,
        label="compact canary detail chunk",
        artifact_type=_CHUNK_TYPE,
        schema_version=COMPACT_CANARY_SCHEMA_VERSION,
        record_type=CompactCanaryDetailChunk,
        allowed_types=_COMPACT_RECORD_TYPES,
        fixed={},
        derived=set(),
    )


def parse_compact_canary_report_json(content: str) -> CompactCanaryReport:
    return parse_artifact(
        content,
        label="compact canary report",
        artifact_type=_REPORT_TYPE,
        schema_version=COMPACT_CANARY_SCHEMA_VERSION,
        record_type=CompactCanaryReport,
        allowed_types=_COMPACT_RECORD_TYPES,
        fixed={
            "detail_chunk_record_limit": COMPACT_CANARY_DETAIL_LIMIT,
            "failure_example_limit": COMPACT_CANARY_FAILURE_LIMIT,
        },
        derived={"passed"},
    )


__all__ = [
    "COMPACT_CANARY_DETAIL_LIMIT",
    "COMPACT_CANARY_FAILURE_LIMIT",
    "COMPACT_CANARY_SCHEMA_VERSION",
    "AggregateMetricCanary",
    "CompactCanaryDetail",
    "CompactCanaryDetailChunk",
    "CompactCanaryEvidence",
    "CompactCanaryPlan",
    "CompactCanaryReport",
    "CompactGateResult",
    "DetailChunkReference",
    "SpooledCompactCanaryEvidence",
    "compact_canary_artifact_names",
    "compact_canary_artifacts",
    "compact_canary_results_csv",
    "detail_inventory_hash",
    "evaluate_compact_canaries",
    "evaluate_compact_canaries_spooled",
    "parse_compact_canary_detail_chunk_json",
    "parse_compact_canary_plan_json",
    "parse_compact_canary_report_json",
    "verify_compact_canary_artifact_directory",
]
