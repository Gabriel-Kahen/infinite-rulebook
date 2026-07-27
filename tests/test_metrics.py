"""Exact symbolic metric and interval propagation tests."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.frontier import (
    FiniteDecisionProblem,
    FrontierSolution,
    one_coordinate_problem,
    solve_frontier,
)
from infinite_rulebook.metrics import (
    CheckpointInterpolation,
    FrontierCurve,
    FrontierPoint,
    MetricInterval,
    NoveltyMetrics,
    PopulationInformationEstimate,
    RewardMetrics,
    SupportMetrics,
    TimedReward,
    UpperEnvelopeCertificate,
    frontier_regret,
    integrate_bit_equivalent,
    lookup_bit_equivalent,
    useful_information_efficiency,
)


def _frontier_point(
    problem: FiniteDecisionProblem,
    reward: float,
) -> FrontierPoint:
    solution = solve_frontier(problem, reward)
    assert solution.converged
    return FrontierPoint.from_frontier_solution(problem, solution)


@pytest.fixture
def exact_curve() -> FrontierCurve:
    problem = one_coordinate_problem(q=2)
    return FrontierCurve(
        points=(
            _frontier_point(problem, 0.0),
            _frontier_point(problem, 0.5),
            _frontier_point(problem, 1.0),
        ),
        zero_information_reward=0.0,
        maximum_reward=1.0,
        semantic_hash=semantic_hash(problem),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )


def test_bit_equivalent_lookup_handles_baseline_grid_and_infeasibility(
    exact_curve: FrontierCurve,
) -> None:
    middle = exact_curve.points[1].information
    final = exact_curve.points[2].information
    assert lookup_bit_equivalent(exact_curve, -1.0) == MetricInterval(0.0, 0.0, "nats")
    assert lookup_bit_equivalent(exact_curve, 0.25) == MetricInterval(
        0.0, 0.5 * middle.upper, "nats"
    )
    bounds = lookup_bit_equivalent(exact_curve, 0.75)
    assert bounds.lower == middle.lower
    assert bounds.upper == pytest.approx(0.5 * (middle.upper + final.upper))
    assert math.isinf(lookup_bit_equivalent(exact_curve, 1.1).lower)


def test_sparse_bit_equivalent_integration_uses_elapsed_round_weights(
    exact_curve: FrontierCurve,
) -> None:
    series = integrate_bit_equivalent(
        exact_curve,
        (
            TimedReward(0, 0.0),
            TimedReward(1, 0.5),
            TimedReward(4, 1.0),
        ),
        horizon=4,
        interpolation=CheckpointInterpolation.LEFT_HOLD,
    )

    middle = exact_curve.points[1].information
    assert series.average == MetricInterval(
        0.75 * middle.lower, 0.75 * middle.upper, "nats"
    )
    assert series.average.lower != pytest.approx(
        math.fsum(point.information.lower for point in exact_curve.points) / 3.0
    )


def test_linear_integration_splits_at_frontier_knots(
    exact_curve: FrontierCurve,
) -> None:
    series = integrate_bit_equivalent(
        exact_curve,
        (TimedReward(0, 0.0), TimedReward(2, 1.0)),
        horizon=2,
        interpolation=CheckpointInterpolation.LINEAR,
    )

    middle = exact_curve.points[1].information
    final = exact_curve.points[2].information
    assert series.average == MetricInterval(
        0.5 * middle.lower,
        0.5 * middle.upper + 0.25 * final.upper,
        "nats",
    )


def test_integration_rejects_ambiguous_checkpoint_coverage(
    exact_curve: FrontierCurve,
) -> None:
    with pytest.raises(ValueError, match="span exactly"):
        integrate_bit_equivalent(
            exact_curve,
            (TimedReward(1, 0.0), TimedReward(4, 1.0)),
            horizon=4,
            interpolation=CheckpointInterpolation.LEFT_HOLD,
        )


def test_useful_information_efficiency_and_validity_diagnostics() -> None:
    information = PopulationInformationEstimate(0.4, 0.1, 0.5, 0.0, 1.0, 20)
    metric = useful_information_efficiency(
        MetricInterval(0.4, 0.6, "nats"),
        information,
        complete_history_manifest=True,
    )

    assert metric.interval == MetricInterval(0.4, 0.6, "ratio")
    assert metric.validation.valid

    violation = useful_information_efficiency(
        MetricInterval(1.1, 1.2, "nats"),
        information,
        complete_history_manifest=False,
    )
    assert not violation.validation.valid
    assert {item.code for item in violation.validation.diagnostics} == {
        "EFFICIENCY_OUT_OF_RANGE",
        "INCOMPLETE_HISTORY_MANIFEST",
    }

    roundoff = useful_information_efficiency(
        MetricInterval(1.0, 1.0 + 5e-13, "nats"),
        PopulationInformationEstimate(1.0, 0.0, 0.0, 0.0, 1.0, 20),
        complete_history_manifest=True,
        tolerance=1e-12,
    )
    assert roundoff.validation.valid


def test_zero_information_efficiency_is_explicitly_undefined() -> None:
    information = PopulationInformationEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 3)

    metric = useful_information_efficiency(
        MetricInterval(0.0, 0.0, "nats"),
        information,
        complete_history_manifest=True,
    )

    assert metric.interval is None
    assert tuple(item.code for item in metric.validation.diagnostics) == (
        "EFFICIENCY_UNDEFINED",
    )

    incomplete = useful_information_efficiency(
        MetricInterval(0.0, 0.0, "nats"),
        information,
        complete_history_manifest=False,
    )
    assert not incomplete.validation.valid
    assert {item.code for item in incomplete.validation.diagnostics} == {
        "EFFICIENCY_UNDEFINED",
        "INCOMPLETE_HISTORY_MANIFEST",
    }


def test_frontier_regret_inverts_certified_curve(
    exact_curve: FrontierCurve,
) -> None:
    middle_budget = exact_curve.points[1].information.upper
    assert frontier_regret(
        exact_curve,
        attained_reward=0.3,
        information_budget_nats=middle_budget,
    ) == MetricInterval(0.2, 0.7, "reward")

    relevant = frontier_regret(
        exact_curve,
        attained_reward=0.3,
        information_budget_nats=middle_budget,
    )
    larger_budget = 0.5 * (middle_budget + exact_curve.points[-1].information.lower)
    full_with_distractor = frontier_regret(
        exact_curve,
        attained_reward=0.3,
        information_budget_nats=larger_budget,
    )
    assert full_with_distractor.lower > relevant.lower


def test_certified_lookup_contains_nonlinear_exact_frontier() -> None:
    from infinite_rulebook.frontier import OneCoordinateFrontier

    exact = OneCoordinateFrontier()
    problem = one_coordinate_problem(q=exact.q, u=exact.u, c=exact.c)
    curve = FrontierCurve(
        points=(
            _frontier_point(problem, 0.0),
            _frontier_point(problem, exact.r_star),
            _frontier_point(problem, exact.u),
        ),
        zero_information_reward=0.0,
        maximum_reward=exact.u,
        semantic_hash=semantic_hash(problem),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )
    target = 0.75
    bounds = lookup_bit_equivalent(curve, target)

    assert bounds.lower <= exact.bit_equivalent(target) <= bounds.upper
    assert bounds.lower < bounds.upper


def test_frontier_rejects_inconsistent_zero_information_endpoint() -> None:
    from dataclasses import replace

    problem = one_coordinate_problem(q=2)
    zero = _frontier_point(problem, 0.0)
    with pytest.raises(ValueError, match="certified solver value"):
        replace(
            zero,
            information=MetricInterval(0.1, 0.1, "nats"),
        )

    foreign_problem = one_coordinate_problem(q=4)
    with pytest.raises(ValueError, match="different problem"):
        FrontierCurve(
            points=(
                zero,
                _frontier_point(problem, 1.0),
            ),
            zero_information_reward=0.0,
            maximum_reward=1.0,
            semantic_hash=semantic_hash(foreign_problem),
            upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
        )


def test_frontier_rejects_inconsistent_requested_endpoint(
    exact_curve: FrontierCurve,
) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="final requested reward"):
        replace(exact_curve, maximum_reward=1.1)


def test_frontier_point_re_evaluates_and_binds_existing_solver_witness() -> None:
    from dataclasses import replace

    problem = one_coordinate_problem()
    solution = solve_frontier(problem, 0.0)
    assert solution.witness is not None
    forged = replace(solution.witness, expected_reward=100.0, mutual_information=100.0)
    point = FrontierPoint.from_frontier_solution(
        problem, replace(solution, witness=forged)
    )

    assert point.reward == 0.0
    assert point.information == MetricInterval(0.0, 0.0, "nats")
    assert point.upper_witness.problem_semantic_hash == semantic_hash(problem)
    assert point.lower_certificate.problem_semantic_hash == semantic_hash(problem)
    assert point.lower_certificate.certificate_hash != (
        point.lower_certificate.source_solution_hash
    )
    assert point.upper_witness.expected_reward == 0.0
    assert point.upper_witness.mutual_information_nats == 0.0


def test_frontier_point_rejects_foreign_lower_bound_certificate() -> None:
    foreign = one_coordinate_problem(q=2, u=1.0, c=1.0)
    target = one_coordinate_problem(q=2, u=2.0, c=1.0)
    solution = solve_frontier(foreign, 0.75)

    with pytest.raises(ValueError, match="different problem"):
        FrontierPoint.from_frontier_solution(target, solution)


def test_frontier_point_rejects_nonconverged_lower_bound() -> None:
    from dataclasses import replace

    problem = one_coordinate_problem(q=2)
    solution = solve_frontier(problem, 0.5)

    with pytest.raises(ValueError, match="converged"):
        FrontierPoint.from_frontier_solution(
            problem,
            replace(solution, converged=False),
        )


def test_frontier_point_rejects_forged_lower_bound() -> None:
    from dataclasses import replace

    problem = one_coordinate_problem(q=2)
    solution = solve_frontier(problem, 0.5)
    forged = replace(
        solution,
        lower_bound=solution.upper_bound,
        duality_gap=0.0,
    )

    with pytest.raises(ValueError, match="exceeds Lagrangian certificate"):
        FrontierPoint.from_frontier_solution(problem, forged)


def test_frontier_solution_keeps_legacy_constructor_compatibility() -> None:
    problem = one_coordinate_problem(q=2)
    witness = problem.constant_channel()
    solution = FrontierSolution(
        0.0,
        0.0,
        witness,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        True,
        semantic_hash(problem),
    )

    assert solution.lower_certificate_marginal is None
    assert solution.lower_certificate_supports is None


def test_reward_support_and_novelty_records_enforce_semantics() -> None:
    reward = RewardMetrics(1.0, 10.0, 0.25, ((0.1, -0.5),))
    support = SupportMetrics(3, 2, 1, 7, 2)
    novelty = NoveltyMetrics(0.7, 0.1, 2.0, 3.0, 0.4, 0.6, 0.2)

    assert reward.lower_quantiles == ((0.1, -0.5),)
    assert support.deployment_support == 3
    assert novelty.aleatoric_observation_novelty != (novelty.persistent_trivia_novelty)

    with pytest.raises(ValueError, match="correct plus incorrect"):
        SupportMetrics(4, 2, 1, 0)

    with pytest.raises(ValueError, match="cannot be negative"):
        NoveltyMetrics(-0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
