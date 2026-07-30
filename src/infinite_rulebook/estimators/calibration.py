"""Synthetic calibration of bounded behavioral-frontier estimates."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

from infinite_rulebook.estimators.behavioral import (
    BehavioralEstimatorConfig,
    BehavioralFrontierPoint,
    IdentificationStatus,
    _semantic_digest,
    estimate_behavioral_frontier,
)
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.inversion import solve_frontier


class CalibrationSplit(StrEnum):
    """Prespecified role of a synthetic calibration case."""

    DEVELOPMENT = "development"
    HELD_OUT = "held-out"


_CALIBRATION_LIMITATIONS = (
    "coverage describes only the prespecified finite synthetic grid",
    "grid points and cases need not be independent or exchangeable",
    "the descriptive coverage fraction has no population interpretation",
    "development-case tuning can bias descriptive grid coverage",
    "upper-error intervals inherit the certified exact solver tolerance",
    "record constructors are structural; only factory outputs are problem-bound",
    "no Monte Carlo, inversion-slope, or scaling gate is evaluated",
)


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result == 0.0:
        result = 0.0
    return result


def _name(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{name} must be nonempty without surrounding whitespace")
    return normalized


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """One named finite problem and reward grid."""

    name: str
    split: CalibrationSplit
    problem: FiniteDecisionProblem
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        name = _name("name", self.name)
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("split must be a CalibrationSplit")
        if not isinstance(self.problem, FiniteDecisionProblem):
            raise TypeError("problem must be a FiniteDecisionProblem")
        targets = tuple(
            sorted(
                {
                    _finite(f"targets[{index}]", target)
                    for index, target in enumerate(self.targets)
                }
            )
        )
        if not targets:
            raise ValueError("targets must not be empty")
        if targets[-1] > self.problem.maximum_reward:
            raise ValueError("calibration targets must be feasible")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """One estimated interval compared with a certified exact envelope."""

    case_name: str
    split: CalibrationSplit
    problem_semantic_hash: str
    estimate: BehavioralFrontierPoint
    exact_lower_bound: float
    exact_upper_bound: float
    exact_converged: bool
    envelope_covered: bool | None
    upper_excess_lower_bound: float
    upper_excess_upper_bound: float
    normalized_reward_excess: float

    def __post_init__(self) -> None:
        case_name = _name("case_name", self.case_name)
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("split must be a CalibrationSplit")
        problem_hash = _semantic_digest(
            "problem_semantic_hash",
            self.problem_semantic_hash,
        )
        if not isinstance(self.estimate, BehavioralFrontierPoint):
            raise TypeError("estimate must be a BehavioralFrontierPoint")
        if self.estimate.identification is IdentificationStatus.INFEASIBLE:
            raise ValueError("calibration points must retain feasible estimates")
        exact_lower = _finite("exact_lower_bound", self.exact_lower_bound)
        exact_upper = _finite("exact_upper_bound", self.exact_upper_bound)
        if exact_lower < 0.0 or exact_upper < exact_lower:
            raise ValueError("exact bounds must be nonnegative and ordered")
        if not isinstance(self.exact_converged, bool):
            raise TypeError("exact_converged must be a bool")
        if self.envelope_covered is not None and not isinstance(
            self.envelope_covered,
            bool,
        ):
            raise TypeError("envelope_covered must be a bool or None")
        if self.exact_converged != (self.envelope_covered is not None):
            raise ValueError(
                "envelope_covered must be present exactly when the solver converged"
            )
        excess_lower = _finite(
            "upper_excess_lower_bound",
            self.upper_excess_lower_bound,
        )
        excess_upper = _finite(
            "upper_excess_upper_bound",
            self.upper_excess_upper_bound,
        )
        if excess_lower != self.estimate.upper_bound - exact_upper:
            raise ValueError("upper_excess_lower_bound is inconsistent")
        if excess_upper != self.estimate.upper_bound - exact_lower:
            raise ValueError("upper_excess_upper_bound is inconsistent")
        normalized_excess = _finite(
            "normalized_reward_excess",
            self.normalized_reward_excess,
        )
        if normalized_excess < 0.0:
            raise ValueError("normalized_reward_excess must be nonnegative")
        object.__setattr__(self, "case_name", case_name)
        object.__setattr__(self, "problem_semantic_hash", problem_hash)
        object.__setattr__(self, "exact_lower_bound", exact_lower)
        object.__setattr__(self, "exact_upper_bound", exact_upper)
        object.__setattr__(self, "upper_excess_lower_bound", excess_lower)
        object.__setattr__(self, "upper_excess_upper_bound", excess_upper)
        object.__setattr__(self, "normalized_reward_excess", normalized_excess)


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Aggregate deterministic errors and grid coverage by split."""

    split: CalibrationSplit
    point_count: int
    exact_converged_count: int
    covered_count: int
    descriptive_grid_coverage: float | None
    maximum_upper_excess: float
    maximum_interval_width: float
    maximum_normalized_reward_excess: float

    def __post_init__(self) -> None:
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("split must be a CalibrationSplit")
        point_count = _count("point_count", self.point_count)
        converged_count = _count(
            "exact_converged_count",
            self.exact_converged_count,
        )
        covered_count = _count("covered_count", self.covered_count)
        if covered_count > converged_count or converged_count > point_count:
            raise ValueError("summary counts are inconsistent")
        coverage = self.descriptive_grid_coverage
        if point_count == 0:
            if coverage is not None:
                raise ValueError("empty summaries must have no coverage fraction")
        else:
            coverage = _finite("descriptive_grid_coverage", coverage)
            if coverage != covered_count / point_count:
                raise ValueError("descriptive_grid_coverage is inconsistent")
        maximum_upper_excess = _finite(
            "maximum_upper_excess",
            self.maximum_upper_excess,
        )
        maximum_interval_width = _finite(
            "maximum_interval_width",
            self.maximum_interval_width,
        )
        maximum_normalized_excess = _finite(
            "maximum_normalized_reward_excess",
            self.maximum_normalized_reward_excess,
        )
        if maximum_interval_width < 0.0 or maximum_normalized_excess < 0.0:
            raise ValueError("maximum widths and normalized excess must be nonnegative")
        if point_count == 0 and (
            maximum_upper_excess != 0.0
            or maximum_interval_width != 0.0
            or maximum_normalized_excess != 0.0
        ):
            raise ValueError("empty summaries must have zero maxima")
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "exact_converged_count", converged_count)
        object.__setattr__(self, "covered_count", covered_count)
        object.__setattr__(self, "descriptive_grid_coverage", coverage)
        object.__setattr__(self, "maximum_upper_excess", maximum_upper_excess)
        object.__setattr__(self, "maximum_interval_width", maximum_interval_width)
        object.__setattr__(
            self,
            "maximum_normalized_reward_excess",
            maximum_normalized_excess,
        )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Deterministic development/held-out synthetic calibration report."""

    config: BehavioralEstimatorConfig
    exact_solver_tolerance: float
    case_count: int
    points: tuple[CalibrationPoint, ...]
    summaries: tuple[CalibrationSummary, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, BehavioralEstimatorConfig):
            raise TypeError("config must be a BehavioralEstimatorConfig")
        tolerance = _finite("exact_solver_tolerance", self.exact_solver_tolerance)
        if tolerance <= 0.0:
            raise ValueError("exact_solver_tolerance must be strictly positive")
        case_count = _count("case_count", self.case_count)
        try:
            points = tuple(self.points)
            summaries = tuple(self.summaries)
            limitations = tuple(
                _name(f"limitations[{index}]", limitation)
                for index, limitation in enumerate(self.limitations)
            )
        except TypeError as error:
            raise TypeError(
                "points, summaries, and limitations must be finite sequences"
            ) from error
        if any(not isinstance(point, CalibrationPoint) for point in points):
            raise TypeError("points must contain CalibrationPoint records")
        if any(not isinstance(summary, CalibrationSummary) for summary in summaries):
            raise TypeError("summaries must contain CalibrationSummary records")
        canonical_points = tuple(
            sorted(
                points,
                key=lambda item: (
                    item.split,
                    item.case_name,
                    item.estimate.target_reward,
                ),
            )
        )
        if points != canonical_points:
            raise ValueError("points must be in canonical order")
        point_keys = tuple(
            (point.split, point.case_name, point.estimate.target_reward)
            for point in points
        )
        if len(set(point_keys)) != len(point_keys):
            raise ValueError("calibration points must be unique")
        names = {point.case_name for point in points}
        if case_count != len(names) or case_count < 1:
            raise ValueError("case_count does not match calibration points")
        case_bindings = {
            point.case_name: (point.split, point.problem_semantic_hash)
            for point in points
        }
        if any(
            case_bindings[point.case_name] != (point.split, point.problem_semantic_hash)
            for point in points
        ):
            raise ValueError("case names must bind one split and problem hash")
        for point in points:
            expected_coverage = (
                point.estimate.lower_bound <= point.exact_lower_bound + tolerance
                and point.estimate.upper_bound + tolerance >= point.exact_upper_bound
                if point.exact_converged
                else None
            )
            if point.envelope_covered is not expected_coverage:
                raise ValueError("envelope_covered does not match report tolerance")
        expected_splits = tuple(sorted({point.split for point in points}))
        if tuple(summary.split for summary in summaries) != expected_splits:
            raise ValueError("summaries must cover represented splits in order")
        expected_summaries = tuple(
            _summarize(split, points) for split in expected_splits
        )
        if summaries != expected_summaries:
            raise ValueError("summaries do not match calibration points")
        if limitations != _CALIBRATION_LIMITATIONS:
            raise ValueError("limitations must retain the complete scientific boundary")
        object.__setattr__(self, "exact_solver_tolerance", tolerance)
        object.__setattr__(self, "case_count", case_count)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(self, "limitations", limitations)


def _summarize(
    split: CalibrationSplit,
    points: tuple[CalibrationPoint, ...],
) -> CalibrationSummary:
    selected = tuple(point for point in points if point.split is split)
    exact_converged_count = sum(
        point.envelope_covered is not None for point in selected
    )
    covered_count = sum(point.envelope_covered is True for point in selected)
    coverage = covered_count / len(selected) if selected else None
    return CalibrationSummary(
        split=split,
        point_count=len(selected),
        exact_converged_count=exact_converged_count,
        covered_count=covered_count,
        descriptive_grid_coverage=coverage,
        maximum_upper_excess=max(
            (point.upper_excess_upper_bound for point in selected),
            default=0.0,
        ),
        maximum_interval_width=max(
            (point.estimate.interval_width for point in selected),
            default=0.0,
        ),
        maximum_normalized_reward_excess=max(
            (point.normalized_reward_excess for point in selected),
            default=0.0,
        ),
    )


def calibrate_behavioral_estimator(
    cases: tuple[CalibrationCase, ...],
    *,
    config: BehavioralEstimatorConfig | None = None,
    exact_solver_tolerance: Real = 1e-8,
) -> CalibrationReport:
    """Compare bounded estimates with certified exact finite frontiers.

    This function performs no tuning.  Callers must prespecify ``config``
    before evaluating cases tagged ``HELD_OUT``.
    """

    supplied_cases = tuple(cases)
    if not supplied_cases:
        raise ValueError("cases must not be empty")
    if any(not isinstance(case, CalibrationCase) for case in supplied_cases):
        raise TypeError("cases must contain CalibrationCase records")
    normalized_cases = tuple(
        sorted(supplied_cases, key=lambda item: (item.split, item.name))
    )
    names = tuple(case.name for case in normalized_cases)
    if len(set(names)) != len(names):
        raise ValueError("calibration case names must be unique")
    settings = BehavioralEstimatorConfig() if config is None else config
    if not isinstance(settings, BehavioralEstimatorConfig):
        raise TypeError("config must be a BehavioralEstimatorConfig")
    exact_tolerance = _finite("exact_solver_tolerance", exact_solver_tolerance)
    if exact_tolerance <= 0.0:
        raise ValueError("exact_solver_tolerance must be strictly positive")

    points = []
    for case in normalized_cases:
        estimate = estimate_behavioral_frontier(
            case.problem,
            case.targets,
            config=settings,
        )
        for estimated in estimate.points:
            exact = solve_frontier(
                case.problem,
                estimated.target_reward,
                tolerance=exact_tolerance,
                bound_tolerance=exact_tolerance,
                lagrangian_tolerance=min(1e-12, exact_tolerance),
            )
            covered = None
            if exact.converged:
                covered = (
                    estimated.lower_bound <= exact.lower_bound + exact_tolerance
                    and estimated.upper_bound + exact_tolerance >= exact.upper_bound
                )
            witness = estimated.witness
            reward_span = (
                case.problem.maximum_reward - case.problem.zero_information_reward
            )
            normalized_reward_excess = (
                max(0.0, witness.expected_reward - estimated.target_reward)
                / reward_span
                if witness is not None and reward_span > 0.0
                else 0.0
            )
            points.append(
                CalibrationPoint(
                    case_name=case.name,
                    split=case.split,
                    problem_semantic_hash=estimate.problem_semantic_hash,
                    estimate=estimated,
                    exact_lower_bound=exact.lower_bound,
                    exact_upper_bound=exact.upper_bound,
                    exact_converged=exact.converged,
                    envelope_covered=covered,
                    upper_excess_lower_bound=(
                        estimated.upper_bound - exact.upper_bound
                    ),
                    upper_excess_upper_bound=(
                        estimated.upper_bound - exact.lower_bound
                    ),
                    normalized_reward_excess=normalized_reward_excess,
                )
            )
    canonical_points = tuple(
        sorted(
            points,
            key=lambda item: (
                item.split,
                item.case_name,
                item.estimate.target_reward,
            ),
        )
    )
    splits = tuple(sorted({case.split for case in normalized_cases}))
    summaries = tuple(_summarize(split, canonical_points) for split in splits)
    return CalibrationReport(
        config=settings,
        exact_solver_tolerance=exact_tolerance,
        case_count=len(normalized_cases),
        points=canonical_points,
        summaries=summaries,
        limitations=_CALIBRATION_LIMITATIONS,
    )
