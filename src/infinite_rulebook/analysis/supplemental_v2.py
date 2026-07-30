"""Registered paired v2 evidence kept outside the primary Holm family."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from infinite_rulebook.analysis.evidence_common import (
    canonical_json,
    count,
    exact_selector,
    expected_group_map,
    finite,
    identifier,
    parse_artifact,
    record_payload,
    sha256,
    validate_expected_group_inventory,
)
from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    ContrastInterpretation,
    ContrastSpec,
    ExpectedGroup,
    GroupSelector,
)
from infinite_rulebook.analysis.statistics import (
    _evaluate_contrast_registered,
    exact_median_interval,
    exact_sign_p_value,
)
from infinite_rulebook.orchestration.hashing import scientific_hash

SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION = 1
_PLAN_TYPE = "registered-supplemental-plan.v2"
_REPORT_TYPE = "registered-supplemental-report.v2"


def _exact_contrast(spec: ContrastSpec) -> None:
    if not isinstance(spec, ContrastSpec):
        raise TypeError("supplemental comparisons must be ContrastSpec values")
    exact_selector("comparison left", spec.left)
    exact_selector("comparison right", spec.right)
    if spec.left == spec.right:
        raise ValueError("supplemental comparison selectors must differ")
    count("comparison checkpoint", spec.checkpoint, positive=True)
    if spec.required_equivalence_gates:
        raise ValueError("supplemental comparisons cannot depend on primary gates")


def _alpha(value: object) -> float:
    result = finite("interval_alpha", value)
    if not 0.0 < result < 1.0:
        raise ValueError("interval_alpha must lie in (0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class SupplementalEvidencePlan:
    name: str
    phase: AnalysisPhase
    legacy_replications: tuple[ContrastSpec, ...]
    descriptive_comparisons: tuple[ContrastSpec, ...]
    interval_alpha: float = 0.05
    expected_groups: tuple[ExpectedGroup, ...] = ()

    def __post_init__(self) -> None:
        identifier("supplemental plan name", self.name)
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        for name, values in (
            ("legacy_replications", self.legacy_replications),
            ("descriptive_comparisons", self.descriptive_comparisons),
        ):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            for item in values:
                _exact_contrast(item)
            object.__setattr__(
                self,
                name,
                tuple(sorted(values, key=lambda item: item.name)),
            )
        if any(
            item.interpretation is not ContrastInterpretation.INFERENTIAL
            for item in self.legacy_replications
        ):
            raise ValueError("legacy replications must retain inferential role")
        if any(
            item.interpretation is not ContrastInterpretation.TELEMETRY_ONLY
            for item in self.descriptive_comparisons
        ):
            raise ValueError("descriptive comparisons must be telemetry-only")
        names = tuple(
            item.name
            for values in (
                self.legacy_replications,
                self.descriptive_comparisons,
            )
            for item in values
        )
        if len(set(names)) != len(names):
            raise ValueError("supplemental comparison names must be unique")
        object.__setattr__(
            self,
            "expected_groups",
            tuple(sorted(self.expected_groups)),
        )
        groups = expected_group_map(self.expected_groups)
        for spec in (*self.legacy_replications, *self.descriptive_comparisons):
            if (
                spec.left not in groups
                or spec.right not in groups
                or spec.checkpoint not in groups[spec.left].checkpoints
                or spec.checkpoint not in groups[spec.right].checkpoints
            ):
                raise ValueError(
                    "supplemental selector/checkpoint is outside expected_groups"
                )
        object.__setattr__(self, "interval_alpha", _alpha(self.interval_alpha))

    def _body(self) -> dict[str, Any]:
        return {
            "artifact_type": _PLAN_TYPE,
            "schema_version": SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION,
            **record_payload(self)["fields"],
            "outside_primary_holm": True,
            "may_rescue_compound_s2": False,
            "holm_decisions": [],
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self._body(), domain=_PLAN_TYPE)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExactIntervalSummary:
    lower: float | str
    upper: float | str
    coverage: float
    method: str


def _interval(values: tuple[float, ...], alpha: float) -> ExactIntervalSummary:
    result = exact_median_interval(values, alpha=alpha)
    return ExactIntervalSummary(
        "-infinity" if result.lower == -math.inf else result.lower,
        "infinity" if result.upper == math.inf else result.upper,
        result.coverage,
        result.method,
    )


@dataclass(frozen=True, slots=True)
class PairedComparisonSummary:
    environment_cluster_count: int
    mean_difference: float
    median_difference: float
    minimum_difference: float
    maximum_difference: float
    above_null_count: int
    tie_count: int
    below_null_count: int
    median_interval: ExactIntervalSummary
    exact_sign_p_value: float


@dataclass(frozen=True, slots=True)
class PairedComparisonEvidence:
    specification: ContrastSpec
    environment_differences: tuple[float, ...]
    cell_pair_count: int
    interval_alpha: float
    summary: PairedComparisonSummary = field(init=False)

    def __post_init__(self) -> None:
        _exact_contrast(self.specification)
        if not isinstance(self.environment_differences, tuple) or not (
            self.environment_differences
        ):
            raise ValueError("paired differences cannot be empty")
        values = tuple(
            finite("environment difference", item)
            for item in self.environment_differences
        )
        object.__setattr__(self, "environment_differences", values)
        count("cell_pair_count", self.cell_pair_count, positive=True)
        if self.cell_pair_count % len(values):
            raise ValueError("cell_pair_count is not balanced across environments")
        alpha = _alpha(self.interval_alpha)
        object.__setattr__(self, "interval_alpha", alpha)
        null = self.specification.null_margin
        object.__setattr__(
            self,
            "summary",
            PairedComparisonSummary(
                len(values),
                math.fsum(values) / len(values),
                float(statistics.median(values)),
                min(values),
                max(values),
                sum(item > null for item in values),
                sum(item == null for item in values),
                sum(item < null for item in values),
                _interval(values, alpha),
                exact_sign_p_value(
                    values,
                    null=null,
                    alternative=self.specification.alternative,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class SupplementalEvidenceReport:
    phase: AnalysisPhase
    dataset_hash: str
    plan_hash: str
    interval_alpha: float
    legacy_replications: tuple[PairedComparisonEvidence, ...]
    descriptive_comparisons: tuple[PairedComparisonEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        sha256("dataset_hash", self.dataset_hash)
        sha256("plan_hash", self.plan_hash)
        object.__setattr__(self, "interval_alpha", _alpha(self.interval_alpha))
        for name, values in (
            ("legacy_replications", self.legacy_replications),
            ("descriptive_comparisons", self.descriptive_comparisons),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(item, PairedComparisonEvidence) for item in values
                )
            ):
                raise ValueError(f"{name} must contain paired evidence")
            object.__setattr__(
                self,
                name,
                tuple(sorted(values, key=lambda item: item.specification.name)),
            )
        names = tuple(
            item.specification.name
            for values in (
                self.legacy_replications,
                self.descriptive_comparisons,
            )
            for item in values
        )
        if len(set(names)) != len(names):
            raise ValueError("supplemental report names must be unique")
        if any(
            item.interval_alpha != self.interval_alpha
            for values in (
                self.legacy_replications,
                self.descriptive_comparisons,
            )
            for item in values
        ):
            raise ValueError("supplemental summaries mix interval alpha values")
        if any(
            item.specification.interpretation is not ContrastInterpretation.INFERENTIAL
            for item in self.legacy_replications
        ):
            raise ValueError("legacy report entries must retain inferential role")
        if any(
            item.specification.interpretation
            is not ContrastInterpretation.TELEMETRY_ONLY
            for item in self.descriptive_comparisons
        ):
            raise ValueError("descriptive report entries must be telemetry-only")

    def _body(self) -> dict[str, Any]:
        return {
            "artifact_type": _REPORT_TYPE,
            "schema_version": SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION,
            **record_payload(self)["fields"],
            "outside_primary_holm": True,
            "may_rescue_compound_s2": False,
            "holm_decisions": [],
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self._body(), domain=_REPORT_TYPE)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _evaluate(
    dataset: AnalysisDataset,
    specs: tuple[ContrastSpec, ...],
    alpha: float,
) -> tuple[PairedComparisonEvidence, ...]:
    results = []
    for spec in specs:
        groups = tuple(
            {
                (
                    item.condition_hash,
                    item.agent_hash,
                    item.environment_kind,
                    item.agent_kind,
                )
                for item in dataset.observations
                if selector.matches(item)
            }
            for selector in (spec.left, spec.right)
        )
        if any(len(group) != 1 for group in groups) or groups[0] == groups[1]:
            raise AnalysisError(
                "supplemental selectors must resolve to two distinct exact groups"
            )
        result = _evaluate_contrast_registered(
            dataset,
            spec,
            interval_alpha=alpha,
        )
        results.append(
            PairedComparisonEvidence(
                spec,
                result.differences,
                result.cell_pair_count,
                alpha,
            )
        )
    return tuple(results)


def evaluate_supplemental_evidence(
    dataset: AnalysisDataset,
    plan: SupplementalEvidencePlan,
) -> SupplementalEvidenceReport:
    if not isinstance(dataset, AnalysisDataset):
        raise TypeError("dataset must be an AnalysisDataset")
    if not isinstance(plan, SupplementalEvidencePlan):
        raise TypeError("plan must be a SupplementalEvidencePlan")
    if dataset.phase is not plan.phase:
        raise AnalysisError("supplemental plan and dataset phases differ")
    validate_expected_group_inventory(dataset, plan.expected_groups)
    return SupplementalEvidenceReport(
        dataset.phase,
        dataset.scientific_hash,
        plan.scientific_hash,
        plan.interval_alpha,
        _evaluate(dataset, plan.legacy_replications, plan.interval_alpha),
        _evaluate(dataset, plan.descriptive_comparisons, plan.interval_alpha),
    )


def supplemental_evidence_artifacts(
    plan: SupplementalEvidencePlan,
    report: SupplementalEvidenceReport,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(plan, SupplementalEvidencePlan) or not isinstance(
        report,
        SupplementalEvidenceReport,
    ):
        raise TypeError("supplemental artifacts require a plan and report")
    if (
        report.phase is not plan.phase
        or report.plan_hash != plan.scientific_hash
        or report.interval_alpha != plan.interval_alpha
        or tuple(item.specification for item in report.legacy_replications)
        != plan.legacy_replications
        or tuple(item.specification for item in report.descriptive_comparisons)
        != plan.descriptive_comparisons
    ):
        raise ValueError("supplemental report does not derive from the supplied plan")
    return (
        ("supplemental-plan.json", plan.to_json()),
        ("supplemental.json", report.to_json()),
    )


_SUPPLEMENTAL_TYPES = (
    SupplementalEvidencePlan,
    ContrastSpec,
    ExpectedGroup,
    GroupSelector,
    ExactIntervalSummary,
    PairedComparisonSummary,
    PairedComparisonEvidence,
    SupplementalEvidenceReport,
)


def parse_supplemental_evidence_plan_json(
    content: str,
) -> SupplementalEvidencePlan:
    return parse_artifact(
        content,
        label="supplemental evidence plan",
        artifact_type=_PLAN_TYPE,
        schema_version=SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION,
        record_type=SupplementalEvidencePlan,
        allowed_types=_SUPPLEMENTAL_TYPES,
        fixed={
            "outside_primary_holm": True,
            "may_rescue_compound_s2": False,
            "holm_decisions": [],
        },
        derived=set(),
    )


def parse_supplemental_evidence_report_json(
    content: str,
) -> SupplementalEvidenceReport:
    return parse_artifact(
        content,
        label="supplemental evidence report",
        artifact_type=_REPORT_TYPE,
        schema_version=SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION,
        record_type=SupplementalEvidenceReport,
        allowed_types=_SUPPLEMENTAL_TYPES,
        fixed={
            "outside_primary_holm": True,
            "may_rescue_compound_s2": False,
            "holm_decisions": [],
        },
        derived=set(),
    )


__all__ = [
    "SUPPLEMENTAL_EVIDENCE_SCHEMA_VERSION",
    "PairedComparisonEvidence",
    "SupplementalEvidencePlan",
    "SupplementalEvidenceReport",
    "evaluate_supplemental_evidence",
    "parse_supplemental_evidence_plan_json",
    "parse_supplemental_evidence_report_json",
    "supplemental_evidence_artifacts",
]
