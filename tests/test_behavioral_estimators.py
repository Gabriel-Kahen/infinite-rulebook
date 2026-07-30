"""Scientific-contract tests for bounded behavioral frontier estimation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.estimators import (
    BehavioralEstimatorConfig,
    BehavioralFrontierEstimate,
    CalibrationCase,
    CalibrationReport,
    CalibrationSplit,
    EstimatorError,
    IdentificationStatus,
    calibrate_behavioral_estimator,
    estimate_behavioral_frontier,
    fit_behavioral_channel,
)
from infinite_rulebook.estimators.calibration import _summarize
from infinite_rulebook.frontier import (
    ChannelWitness,
    FiniteDecisionProblem,
    one_coordinate_problem,
    solve_frontier,
)
from infinite_rulebook.validation import (
    DiagnosticSeverity,
    ValidationDiagnostic,
    ValidationReport,
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


def test_public_kl_fields_reconcile_roundoff_to_nonnegative_values() -> None:
    fit = fit_behavioral_channel(
        one_coordinate_problem(),
        0.0,
        config=BehavioralEstimatorConfig(
            betas=(0.0, 1.0),
            optimizer_steps=2,
        ),
    )

    assert fit.direct_reference_kl >= 0.0
    assert fit.reference_kl_upper_bound >= 0.0
    assert fit.reference_compression_gap >= 0.0
    assert fit.reference_identity_residual >= 0.0


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


def test_signed_zero_has_one_config_and_estimate_state() -> None:
    problem = FiniteDecisionProblem(
        (0.8158734926586414, 0.19560785280390003),
        ((0.0, 1.0), (0.0, -1.0)),
    )
    positive = BehavioralEstimatorConfig(
        betas=(0.0,),
        optimizer_steps=1,
        diagnostic_tolerance=1e-30,
    )
    negative = BehavioralEstimatorConfig(
        betas=(-0.0,),
        optimizer_steps=1,
        diagnostic_tolerance=1e-30,
    )
    target = problem.zero_information_reward + 0.5 * (
        problem.maximum_reward - problem.zero_information_reward
    )

    left = estimate_behavioral_frontier(problem, (target,), config=positive)
    right = estimate_behavioral_frontier(problem, (target,), config=negative)

    assert negative.betas[0].hex() == "0x0.0p+0"
    assert positive == negative
    assert semantic_hash(positive) == semantic_hash(negative)
    assert left == right
    assert semantic_hash(left) == semantic_hash(right)


def test_estimator_evidence_breaks_aliases_and_rejects_forged_state() -> None:
    source = estimate_behavioral_frontier(
        one_coordinate_problem(),
        (0.5,),
        config=BehavioralEstimatorConfig(
            betas=(0.0, 1.0),
            optimizer_steps=2,
        ),
    )
    fits = list(source.fits)
    points = list(source.points)
    limitations = list(source.limitations)
    rebuilt = BehavioralFrontierEstimate(
        source.problem_semantic_hash,
        source.config,
        fits,  # type: ignore[arg-type]
        points,  # type: ignore[arg-type]
        limitations,  # type: ignore[arg-type]
    )
    before = semantic_hash(rebuilt)

    fits.clear()
    points.clear()
    limitations.clear()

    assert rebuilt == source
    assert semantic_hash(rebuilt) == before
    assert issubclass(EstimatorError, RuntimeError)

    fit = source.fits[0]
    fit_rows = [list(row) for row in fit.witness.channel]
    fit_marginal = list(fit.witness.action_marginal)
    reference = list(fit.reference_marginal)
    rebuilt_fit = replace(
        fit,
        witness=ChannelWitness(
            fit_rows,  # type: ignore[arg-type]
            fit_marginal,  # type: ignore[arg-type]
            fit.witness.expected_reward,
            fit.witness.mutual_information,
        ),
        reference_marginal=reference,  # type: ignore[arg-type]
    )
    fit_hash = semantic_hash(rebuilt_fit)
    fit_rows[0][0] = 99.0
    fit_marginal[0] = 99.0
    reference[0] = 99.0
    assert rebuilt_fit == fit
    assert semantic_hash(rebuilt_fit) == fit_hash

    with pytest.raises(ValueError, match="SHA-256"):
        replace(source, problem_semantic_hash="not-a-digest")
    with pytest.raises(ValueError, match="upper_bound"):
        replace(
            source.points[0],
            upper_bound=math.nextafter(source.points[0].upper_bound, math.inf),
        )
    with pytest.raises(ValueError, match="exactly when a point is partial"):
        replace(source, fits=())
    with pytest.raises(ValueError, match="scientific boundary"):
        replace(source, limitations=("claim transfer is proven",))
    with pytest.raises(ValueError, match="full-support"):
        replace(fit, reference_marginal=(1.0, *(0.0 for _ in reference[1:])))
    with pytest.raises(ValueError, match="certified_objective_gap"):
        replace(
            fit,
            certified_objective_gap=fit.certified_objective_gap + 1.0,
        )
    with pytest.raises(ValueError, match="certified_objective_gap"):
        replace(fit, certified_objective_gap=1e-14)
    with pytest.raises(ValueError, match="direct reference KL"):
        replace(fit, direct_reference_kl=99.0)
    with pytest.raises(ValueError, match="direct reference KL"):
        replace(fit, reference_identity_residual=1e-12)
    with pytest.raises(ValueError, match="objective_upper_bound"):
        replace(source.fits[1], objective_upper_bound=0.0)
    with pytest.raises(ValueError, match="fixed_point_residual"):
        replace(fit, fixed_point_residual=0.0)
    with pytest.raises(ValueError, match="diagnostics"):
        replace(source.fits[1], converged=True)
    forged_fit = replace(
        source.fits[1],
        converged=True,
        diagnostics=ValidationReport(),
    )
    with pytest.raises(ValueError, match="convergence"):
        replace(source, fits=(source.fits[0], forged_fit))
    with pytest.raises(ValueError, match="bound methods"):
        replace(source.points[0], upper_bound_method="claimed-witness")
    with pytest.raises(ValueError, match="lower_bound"):
        replace(
            source.points[0],
            lower_bound=source.points[0].upper_bound + 1.0,
        )
    forged_point = replace(
        source.points[0],
        lower_bound=0.0,
        lower_bound_beta=0.0,
    )
    with pytest.raises(ValueError, match="lower certificate"):
        replace(source, points=(forged_point,))
    forged_diagnostics = ValidationReport(
        (
            ValidationDiagnostic(
                DiagnosticSeverity.INFO,
                "bound-roundoff-reconciled",
                f"target[{source.points[0].target_reward.hex()}]",
                "bounds crossed only within floating-point roundoff",
            ),
        )
    )
    with pytest.raises(ValueError, match="diagnostics"):
        replace(source.points[0], diagnostics=forged_diagnostics)


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
    assert "directly-evaluated-feasible" in point.upper_bound_method


def test_retained_mixed_witness_is_exactly_reproducible_and_feasible() -> None:
    problem = FiniteDecisionProblem(
        prior=(
            0.26608193554876375,
            0.31114972852923584,
            0.2738592564308415,
            0.14890907949115892,
        ),
        rewards=(
            (9.657750052104575, -5.483867461874411, -9.07042011017882),
            (-6.958036943680277, 5.151793337152245, 6.100361128514216),
            (-8.052067886344672, 13.381747969330256, -12.83163368163406),
            (-19.44085306061497, -3.2334031048107974, -13.018592006116968),
        ),
    )
    config = BehavioralEstimatorConfig(
        betas=(0.0, 0.2, 1.0, 4.0, 16.0),
        optimizer_steps=9,
        reference_update_rate=0.63,
        reference_smoothing=1e-10,
    )
    target = float.fromhex("0x1.3b533a1fa479dp+2")

    point = estimate_behavioral_frontier(
        problem,
        (target,),
        config=config,
    ).points[0]

    assert point.witness is not None
    assert all(math.fsum(row) == 1.0 for row in point.witness.channel)
    assert problem.evaluate(point.witness.channel) == point.witness
    assert point.witness.expected_reward >= target
    assert point.upper_bound == point.witness.mutual_information


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


def test_zero_information_point_retains_constant_witness_roundoff() -> None:
    problem = FiniteDecisionProblem(
        (0.8158734926586414, 0.19560785280390003),
        ((0.0, 1.0), (0.0, -1.0)),
    )

    point = estimate_behavioral_frontier(
        problem,
        (problem.zero_information_reward,),
    ).points[0]

    assert point.identification is IdentificationStatus.EXACT_ZERO_INFORMATION
    assert point.witness is not None
    assert problem.evaluate(point.witness.channel) == point.witness
    assert point.witness.expected_reward >= point.target_reward
    assert point.lower_bound == 0.0
    assert point.upper_bound == point.witness.mutual_information
    assert point.upper_bound > 0.0
    assert point.upper_bound <= 64.0 * math.ulp(1.0)
    assert {diagnostic.code for diagnostic in point.diagnostics.diagnostics} == {
        "constant-channel-evaluation-roundoff"
    }


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

    nonconverged = replace(
        left.points[1],
        exact_converged=False,
        envelope_covered=None,
    )
    conservative = _summarize(
        CalibrationSplit.HELD_OUT,
        (left.points[0], nonconverged),
    )
    assert conservative.point_count == 2
    assert conservative.exact_converged_count == 1
    assert conservative.covered_count == 1
    assert conservative.descriptive_grid_coverage == 0.5


def test_calibration_evidence_normalizes_names_and_breaks_aliases() -> None:
    problem = one_coordinate_problem(2, 1.0, 2.0)
    composed = CalibrationCase(
        "é",
        CalibrationSplit.HELD_OUT,
        problem,
        (0.2,),
    )
    decomposed = CalibrationCase(
        "e\u0301",
        CalibrationSplit.HELD_OUT,
        problem,
        (0.2,),
    )

    assert composed == decomposed
    assert semantic_hash(composed) == semantic_hash(decomposed)
    with pytest.raises(ValueError, match="names must be unique"):
        calibrate_behavioral_estimator(
            (composed, decomposed),
            config=BehavioralEstimatorConfig(optimizer_steps=2),
        )

    report = calibrate_behavioral_estimator(
        (composed,),
        config=BehavioralEstimatorConfig(
            betas=(0.0, 1.0),
            optimizer_steps=2,
        ),
    )
    points = list(report.points)
    summaries = list(report.summaries)
    limitations = list(report.limitations)
    rebuilt = CalibrationReport(
        report.config,
        report.exact_solver_tolerance,
        report.case_count,
        points,  # type: ignore[arg-type]
        summaries,  # type: ignore[arg-type]
        limitations,  # type: ignore[arg-type]
    )
    before = semantic_hash(rebuilt)

    points.clear()
    summaries.clear()
    limitations.clear()

    assert rebuilt == report
    assert semantic_hash(rebuilt) == before
    with pytest.raises(ValueError, match="case_count"):
        replace(report, case_count=report.case_count + 1)
    with pytest.raises(ValueError, match="scientific boundary"):
        replace(report, limitations=("coverage generalizes",))
    with pytest.raises(ValueError, match="upper_excess_lower_bound"):
        replace(
            report.points[0],
            upper_excess_lower_bound=math.nextafter(
                report.points[0].upper_excess_lower_bound,
                math.inf,
            ),
        )
    with pytest.raises(ValueError, match="coverage"):
        replace(
            report.summaries[0],
            covered_count=0,
        )
    forged_coverage = replace(
        report.points[0],
        envelope_covered=not report.points[0].envelope_covered,
    )
    with pytest.raises(ValueError, match="report tolerance"):
        replace(report, points=(forged_coverage,))


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


def test_subnormal_reference_kl_uses_finite_log_differences() -> None:
    subnormal = float.fromhex("0x0.0000000000001p-1022")
    problem = FiniteDecisionProblem(
        (1.0, subnormal),
        ((0.0, -1_000.0), (-1_000.0, 0.0)),
    )
    config = BehavioralEstimatorConfig(
        betas=(0.0, 1.0),
        optimizer_steps=2,
        reference_smoothing=1e-310,
        maximum_actions=2,
    )

    fit = fit_behavioral_channel(problem, 1.0, config=config)
    estimate = estimate_behavioral_frontier(
        problem,
        (problem.maximum_reward,),
        config=config,
    )

    assert math.isfinite(fit.direct_reference_kl)
    assert math.isfinite(fit.reference_kl_upper_bound)
    assert fit.reference_kl_upper_bound >= fit.witness.mutual_information
    assert estimate.points[0].witness is not None
    assert problem.evaluate(estimate.points[0].witness.channel) == (
        estimate.points[0].witness
    )
