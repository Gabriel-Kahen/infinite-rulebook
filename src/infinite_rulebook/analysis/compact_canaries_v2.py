"""Chunk-authenticated compact evidence for v2 deterministic canaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby
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
    finite,
    identifier,
    parse_artifact,
    payload,
    record_payload,
    sha256,
)
from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CheckpointObservation,
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
        if (self.left_semantic_hash is None) != (self.right_semantic_hash is None):
            raise ValueError("semantic hashes must be supplied together")
        if self.left_semantic_hash is not None:
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
            not self.results
            or self.results != tuple(sorted(self.results, key=lambda item: item.name))
            or len({item.name for item in self.results}) != len(self.results)
        ):
            raise ValueError("results are not a nonempty canonical inventory")
        count("detail_record_count", self.detail_record_count, positive=True)
        if (
            not self.detail_chunks
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
            or not isinstance(
                self.detail_chunks,
                tuple,
            )
            or any(
                not isinstance(item, CompactCanaryDetailChunk)
                for item in self.detail_chunks
            )
        ):
            raise TypeError("compact evidence requires a report and chunk tuple")
        references = tuple(item.reference for item in self.detail_chunks)
        if references != self.report.detail_chunks:
            raise ValueError("detail chunks do not match the report inventory")
        previous: tuple[object, ...] | None = None

        def records() -> Iterable[CompactCanaryDetail]:
            nonlocal previous
            for chunk in self.detail_chunks:
                for item in chunk.records:
                    if previous is not None and item.sort_key <= previous:
                        raise ValueError("detail chunks are not globally canonical")
                    previous = item.sort_key
                    yield item

        summaries = tuple(
            _summarize_gate(group)
            for _, group in groupby(records(), key=lambda item: item.gate_name)
        )
        if summaries != self.report.results:
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
    histories_by_group: dict[
        tuple[str, str, str, str],
        dict[tuple[int, int], dict[int, CheckpointObservation]],
    ] = {}
    representatives: dict[
        tuple[str, str, str, str],
        CheckpointObservation,
    ] = {}
    for item in dataset.observations:
        group_key = (
            item.condition_hash,
            item.agent_hash,
            item.environment_kind,
            item.agent_kind,
        )
        representatives.setdefault(group_key, item)
        history = histories_by_group.setdefault(group_key, {}).setdefault(
            item.pair_key,
            {},
        )
        if item.round_index in history:
            raise AnalysisError(
                "aggregate group contains duplicate replica/checkpoint cells"
            )
        history[item.round_index] = item

    claimed_groups = []
    for selector in spec.selectors:
        matched_groups = [
            key for key, item in representatives.items() if selector.matches(item)
        ]
        if len(matched_groups) != 1:
            raise AnalysisError(
                f"aggregate selector {selector.label!r} must match exactly one group"
            )
        group_key = matched_groups[0]
        if group_key in claimed_groups:
            raise AnalysisError("aggregate selectors overlap one registered group")
        claimed_groups.append(group_key)
    if set(claimed_groups) != set(histories_by_group):
        raise AnalysisError(
            "aggregate selector inventory omits or adds registered dataset groups"
        )

    ordered_groups = sorted(
        (
            (
                scientific_hash(group_key, domain="compact-canary-group.v2"),
                histories_by_group[group_key],
            )
            for group_key in claimed_groups
        ),
        key=lambda item: item[0],
    )
    comparison = f"{spec.aggregate_metric}=mean({spec.source_metric}[1..checkpoint])"
    for group_hash, histories in ordered_groups:
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


def evaluate_compact_canaries(
    dataset: AnalysisDataset,
    plan: CompactCanaryPlan,
) -> CompactCanaryEvidence:
    """Reuse v1 gate evaluators and compact their authenticated detail output."""

    if not isinstance(dataset, AnalysisDataset):
        raise TypeError("dataset must be an AnalysisDataset")
    if not isinstance(plan, CompactCanaryPlan):
        raise TypeError("plan must be a CompactCanaryPlan")
    if dataset.phase is not plan.phase:
        raise AnalysisError("compact canary plan and dataset phases differ")
    results = []
    chunks = []
    buffer = []
    previous: tuple[object, ...] | None = None

    def emit(item: CompactCanaryDetail) -> None:
        nonlocal previous
        if previous is not None and item.sort_key <= previous:
            raise AnalysisError("compact detail stream is not globally canonical")
        buffer.append(item)
        previous = item.sort_key
        if len(buffer) == COMPACT_CANARY_DETAIL_LIMIT:
            chunks.append(CompactCanaryDetailChunk(len(chunks), tuple(buffer)))
            buffer.clear()

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
        chunks.append(CompactCanaryDetailChunk(len(chunks), tuple(buffer)))
    detail_chunks = tuple(chunks)
    references = tuple(item.reference for item in detail_chunks)
    report = CompactCanaryReport(
        dataset.phase,
        dataset.scientific_hash,
        plan.scientific_hash,
        tuple(results),
        sum(item.record_count for item in results),
        references,
        detail_inventory_hash(references),
    )
    return CompactCanaryEvidence(report, detail_chunks)


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
                f"canary-details-{chunk.index:06d}.json",
                chunk.to_json(),
            )
            for chunk in evidence.detail_chunks
        ),
    )


_COMPACT_RECORD_TYPES = (
    AggregateMetricCanary,
    CompactCanaryPlan,
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
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
    "compact_canary_artifacts",
    "detail_inventory_hash",
    "evaluate_compact_canaries",
    "parse_compact_canary_detail_chunk_json",
    "parse_compact_canary_plan_json",
    "parse_compact_canary_report_json",
]
