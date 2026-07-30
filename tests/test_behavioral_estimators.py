"""Scientific-contract tests for bounded behavioral frontier estimation."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.estimators import (
    BehavioralEstimatorConfig,
    CalibrationCase,
    CalibrationSplit,
    IdentificationStatus,
    calibrate_behavioral_estimator,
    estimate_behavioral_frontier,
    fit_behavioral_channel,
)
from infinite_rulebook.frontier import (
    FiniteDecisionProblem,
    one_coordinate_problem,
    solve_frontier,
)

FAST_CONFIG = BehavioralEstimatorConfig(optimizer_steps=64)


def test_reference_kl_identity_and_lagrangian_certificate() -> None:
    problem = one_coordinate_problem()
    fit = fit_behavioral_channel(problem, 2.0, config=FAST_CONFIG)

    recomputed = problem.evaluate(fit.witness.channel)
    assert recomputed == fit.witness
    assert fit.reference_kl_upper_bound >= fit.witness.mutual_information
    assert fit.reference_kl_upper_bound == pytest.approx(
        fit.witness.mutual_information + fit.reference_compression_gap,
        abs=1e-12,
    )
    assert fit.direct_reference_kl == pytest.approx(
        fit.reference_kl_upper_bound,
        abs=1e-12,
    )
    assert fit.reference_identity_residual <= 1e-12
    assert fit.objective_lower_bound <= fit.objective_upper_bound + 1e-12
    assert fit.certified_objective_gap >= 0.0
    assert all(probability > 0.0 for probability in fit.reference_marginal)
    assert fit.diagnostics.valid


def test_estimator_is_deterministic_and_hash_stable() -> None:
    problem = FiniteDecisionProblem(
        prior=(0.4, 0.6),
        rewards=((0.0, 1.0, -1.0), (0.0, -1.0, 1.0)),
    )
    left = estimate_behavioral_frontier(
        problem,
        (0.2, 0.5, 0.8),
        config=FAST_CONFIG,
    )
    right = estimate_behavioral_frontier(
        problem,
        (0.8, 0.2, 0.5),
        config=FAST_CONFIG,
    )

    assert left == right
    assert semantic_hash(left) == semantic_hash(right)


@pytest.mark.parametrize(
    ("problem", "targets"),
    [
        (one_coordinate_problem(2, 1.0, 2.0), (0.1, 0.5, 0.9, 1.0)),
        (one_coordinate_problem(4, 1.0, 1.0), (0.1, 0.5, 0.8, 1.0)),
    ],
)
def test_partial_intervals_cover_certified_exact_frontier(
    problem: FiniteDecisionProblem,
    targets: tuple[float, ...],
) -> None:
    estimate = estimate_behavioral_frontier(
        problem,
        targets,
        config=FAST_CONFIG,
    )

    for point in estimate.points:
        exact = solve_frontier(problem, point.target_reward, tolerance=1e-8)
        assert exact.converged
        assert point.witness is not None
        assert point.witness.expected_reward >= point.target_reward
        assert problem.evaluate(point.witness.channel) == point.witness
        assert point.lower_bound <= exact.lower_bound + 1e-8
        assert point.upper_bound >= exact.upper_bound - 1e-8
        assert point.lower_bound <= point.upper_bound


def test_weak_grid_reports_partial_identification_without_false_convergence() -> None:
    problem = one_coordinate_problem()
    config = BehavioralEstimatorConfig(betas=(0.0,), optimizer_steps=1)
    estimate = estimate_behavioral_frontier(problem, (0.8,), config=config)
    point = estimate.points[0]
    exact = solve_frontier(problem, 0.8)

    assert point.identification is IdentificationStatus.CERTIFIED_PARTIAL
    assert point.lower_bound == 0.0
    assert point.witness is not None
    assert point.witness.expected_reward >= 0.8
    assert point.upper_bound >= exact.upper_bound
    assert point.interval_width > 0.0
    assert "exactly-evaluated-feasible" in point.upper_bound_method


def test_zero_information_and_infeasible_targets_are_explicit() -> None:
    problem = one_coordinate_problem()
    estimate = estimate_behavioral_frontier(
        problem,
        (0.0, 1.1),
        config=FAST_CONFIG,
    )
    zero, infeasible = estimate.points

    assert zero.identification is IdentificationStatus.EXACT_ZERO_INFORMATION
    assert zero.lower_bound == zero.upper_bound == 0.0
    assert zero.witness is not None
    assert infeasible.identification is IdentificationStatus.INFEASIBLE
    assert infeasible.witness is None
    assert math.isinf(infeasible.lower_bound)
    assert math.isinf(infeasible.upper_bound)


def test_all_trivial_targets_skip_pathological_channel_fits() -> None:
    problem = FiniteDecisionProblem(
        (0.5, 0.5),
        ((1e308, 0.0), (0.0, 1e308)),
    )
    config = BehavioralEstimatorConfig(
        betas=(0.0, 2.0),
        optimizer_steps=1,
    )
    estimate = estimate_behavioral_frontier(
        problem,
        (0.0, math.nextafter(problem.maximum_reward, math.inf)),
        config=config,
    )
    zero, infeasible = estimate.points

    assert estimate.fits == ()
    assert zero.identification is IdentificationStatus.EXACT_ZERO_INFORMATION
    assert zero.witness == problem.constant_channel()
    assert zero.lower_bound == zero.upper_bound == 0.0
    assert infeasible.identification is IdentificationStatus.INFEASIBLE
    assert infeasible.witness is None
    assert infeasible.lower_bound == infeasible.upper_bound == math.inf


def test_maximum_reward_target_remains_nontrivial() -> None:
    problem = one_coordinate_problem()
    config = BehavioralEstimatorConfig(betas=(0.0,), optimizer_steps=1)

    estimate = estimate_behavioral_frontier(
        problem,
        (problem.maximum_reward,),
        config=config,
    )

    assert len(estimate.fits) == 1
    assert estimate.points[0].identification is IdentificationStatus.CERTIFIED_PARTIAL
    assert estimate.points[0].witness is not None
    assert estimate.points[0].witness.expected_reward >= problem.maximum_reward


def test_estimates_are_invariant_to_state_and_action_labels() -> None:
    original = FiniteDecisionProblem(
        (0.4, 0.6),
        ((0.0, 1.0, -1.0), (0.0, -1.0, 1.0)),
    )
    permuted = FiniteDecisionProblem(
        (0.6, 0.4),
        ((1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)),
    )
    targets = (0.2, 0.5, 0.8)
    left = estimate_behavioral_frontier(
        original,
        targets,
        config=FAST_CONFIG,
    )
    right = estimate_behavioral_frontier(
        permuted,
        targets,
        config=FAST_CONFIG,
    )

    assert tuple(point.lower_bound for point in left.points) == pytest.approx(
        tuple(point.lower_bound for point in right.points),
        abs=1e-12,
    )
    assert tuple(point.upper_bound for point in left.points) == pytest.approx(
        tuple(point.upper_bound for point in right.points),
        abs=1e-12,
    )


def test_calibration_reports_descriptive_split_coverage_and_errors() -> None:
    cases = (
        CalibrationCase(
            "q2-development",
            CalibrationSplit.DEVELOPMENT,
            one_coordinate_problem(2, 1.0, 2.0),
            (0.1, 0.5, 0.9),
        ),
        CalibrationCase(
            "q4-held-out",
            CalibrationSplit.HELD_OUT,
            one_coordinate_problem(4, 1.0, 1.0),
            (0.1, 0.5, 0.8),
        ),
    )
    report = calibrate_behavioral_estimator(cases, config=FAST_CONFIG)

    assert report.case_count == 2
    assert len(report.points) == 6
    assert all(point.exact_converged for point in report.points)
    assert all(point.envelope_covered for point in report.points)
    assert all(point.upper_excess_upper_bound >= -1e-8 for point in report.points)
    assert len(report.summaries) == 2
    for summary in report.summaries:
        assert summary.point_count == 3
        assert summary.exact_converged_count == 3
        assert summary.covered_count == 3
        assert summary.descriptive_grid_coverage == 1.0
        assert summary.maximum_interval_width >= 0.0
        assert summary.maximum_normalized_reward_excess <= 1e-12
    assert any(
        "not be independent or exchangeable" in item for item in report.limitations
    )
    assert any("no population interpretation" in item for item in report.limitations)


def test_calibration_report_is_deterministic() -> None:
    case = CalibrationCase(
        "held-out",
        CalibrationSplit.HELD_OUT,
        one_coordinate_problem(2, 1.0, 2.0),
        (0.2, 0.7),
    )

    left = calibrate_behavioral_estimator((case,), config=FAST_CONFIG)
    right = calibrate_behavioral_estimator((case,), config=FAST_CONFIG)

    assert left == right
    assert semantic_hash(left) == semantic_hash(right)


@pytest.mark.parametrize(
    "config",
    [
        BehavioralEstimatorConfig,
        lambda: BehavioralEstimatorConfig(betas=(0.1,)),
        lambda: BehavioralEstimatorConfig(betas=(0.0, 0.0)),
        lambda: BehavioralEstimatorConfig(optimizer_steps=0),
        lambda: BehavioralEstimatorConfig(reference_update_rate=0.0),
        lambda: BehavioralEstimatorConfig(reference_smoothing=0.0),
        lambda: BehavioralEstimatorConfig(reference_smoothing=1.0),
        lambda: BehavioralEstimatorConfig(
            reference_smoothing=float.fromhex("0x0.0000000000001p-1022")
        ),
        lambda: BehavioralEstimatorConfig(maximum_states=0),
        lambda: BehavioralEstimatorConfig(maximum_actions=0),
    ],
)
def test_invalid_estimator_configs_fail_closed(config: object) -> None:
    if config is BehavioralEstimatorConfig:
        assert config().betas[0] == 0.0
    else:
        with pytest.raises((TypeError, ValueError)):
            config()  # type: ignore[operator]


def test_calibration_rejects_infeasible_targets_and_duplicate_names() -> None:
    problem = one_coordinate_problem()
    with pytest.raises(ValueError, match="must be feasible"):
        CalibrationCase(
            "bad",
            CalibrationSplit.DEVELOPMENT,
            problem,
            (1.1,),
        )

    case = CalibrationCase(
        "duplicate",
        CalibrationSplit.DEVELOPMENT,
        problem,
        (0.5,),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        calibrate_behavioral_estimator((case, case), config=FAST_CONFIG)


def test_problem_size_caps_enforce_the_synthetic_boundary() -> None:
    state_limited = BehavioralEstimatorConfig(maximum_states=1)
    action_limited = BehavioralEstimatorConfig(maximum_actions=2)
    problem = one_coordinate_problem(2, 1.0, 2.0)

    with pytest.raises(ValueError, match="maximum_states"):
        estimate_behavioral_frontier(problem, (0.5,), config=state_limited)
    with pytest.raises(ValueError, match="maximum_actions"):
        estimate_behavioral_frontier(problem, (0.5,), config=action_limited)
    with pytest.raises(TypeError, match="CalibrationCase"):
        calibrate_behavioral_estimator((object(),))  # type: ignore[arg-type]


def test_positive_smoothing_preserves_reference_support_under_underflow() -> None:
    problem = FiniteDecisionProblem(
        (0.5, 0.5),
        ((0.0, -1_000.0), (0.0, 1_000.0)),
    )
    config = BehavioralEstimatorConfig(
        betas=(0.0, 1.0),
        optimizer_steps=4,
        reference_smoothing=1e-12,
    )

    fit = fit_behavioral_channel(problem, 1.0, config=config)

    assert all(probability > 0.0 for probability in fit.reference_marginal)
    assert math.isfinite(fit.reference_kl_upper_bound)
