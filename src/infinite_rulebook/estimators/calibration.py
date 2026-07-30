"""Synthetic calibration of bounded behavioral-frontier estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

from infinite_rulebook.estimators.behavioral import (
    BehavioralEstimatorConfig,
    BehavioralFrontierPoint,
    estimate_behavioral_frontier,
)
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.inversion import solve_frontier


class CalibrationSplit(StrEnum):
    """Prespecified role of a synthetic calibration case."""

    DEVELOPMENT = "development"
    HELD_OUT = "held-out"


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """One named finite problem and reward grid."""

    name: str
    split: CalibrationSplit
    problem: FiniteDecisionProblem
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
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


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Deterministic development/held-out synthetic calibration report."""

    config: BehavioralEstimatorConfig
    exact_solver_tolerance: float
    case_count: int
    points: tuple[CalibrationPoint, ...]
    summaries: tuple[CalibrationSummary, ...]
    limitations: tuple[str, ...]


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
        limitations=(
            "coverage describes only the prespecified finite synthetic grid",
            "grid points and cases need not be independent or exchangeable",
            "the descriptive coverage fraction has no population interpretation",
            "development-case tuning can bias descriptive grid coverage",
            "upper-error intervals inherit the certified exact solver tolerance",
            "no Monte Carlo, inversion-slope, or scaling gate is evaluated",
        ),
    )
