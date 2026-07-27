"""Regression tests for the finite stochastic-channel frontier solver."""

from __future__ import annotations

import math
import random

import pytest

from infinite_rulebook.frontier.blahut_arimoto import (
    _channel_from_marginal,
    _marginal_objective_bounds,
    solve_lagrangian,
)
from infinite_rulebook.frontier.finite_problem import (
    FiniteDecisionProblem,
    one_coordinate_problem,
)
from infinite_rulebook.frontier.inversion import invert_frontier, solve_frontier
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier


@pytest.mark.parametrize(
    ("q", "u", "c", "target"),
    [
        (2, 1.0, 2.0, 0.1),
        (2, 1.0, 2.0, 0.55),
        (2, 1.0, 2.0, 0.9),
        (4, 1.0, 1.0, 0.1),
        (4, 1.0, 1.0, 0.5),
        (4, 1.0, 1.0, 0.8),
    ],
)
def test_finite_solver_matches_analytic_frontier(
    q: int, u: float, c: float, target: float
) -> None:
    problem = one_coordinate_problem(q, u, c)
    analytic = OneCoordinateFrontier(q, u, c).bit_equivalent(target)
    result = solve_frontier(problem, target, tolerance=2e-8)

    assert result.converged
    assert result.witness is not None
    assert result.witness.expected_reward >= target - 2e-10
    assert result.lower_bound <= analytic + 2e-8
    assert result.upper_bound >= analytic - 2e-8
    assert result.upper_bound == pytest.approx(analytic, abs=2e-7)


def test_direct_channel_witness_uses_nats() -> None:
    problem = one_coordinate_problem(q=4, u=1.0, c=1.0)
    witness = problem.maximizing_channel()

    assert witness.expected_reward == pytest.approx(1.0)
    assert witness.mutual_information == pytest.approx(math.log(4.0))
    assert math.fsum(witness.action_marginal) == pytest.approx(1.0)


def test_zero_information_endpoint_selects_best_constant_action() -> None:
    problem = FiniteDecisionProblem(
        prior=(0.8, 0.2),
        rewards=((2.0, 0.0), (-1.0, 1.0)),
    )
    result = solve_frontier(problem, 1.39)

    assert problem.zero_information_reward == pytest.approx(1.4)
    assert result.converged
    assert result.lower_bound == 0.0
    assert result.upper_bound == 0.0
    assert result.witness is not None
    assert result.witness.mutual_information == 0.0
    assert result.witness.expected_reward == pytest.approx(1.4)


def test_infeasible_target_is_explicit() -> None:
    problem = one_coordinate_problem()
    result = solve_frontier(problem, math.inf)

    assert result.witness is None
    assert math.isinf(result.lower_bound)
    assert math.isinf(result.upper_bound)


def test_reward_endpoints_are_exact() -> None:
    problem = one_coordinate_problem()

    lower = solve_frontier(problem, -math.inf)
    upper = solve_frontier(problem, problem.maximum_reward)

    assert lower.upper_bound == 0.0
    assert upper.converged
    assert upper.lower_bound == pytest.approx(math.log(4.0), abs=1e-10)
    assert upper.upper_bound == pytest.approx(math.log(4.0), abs=1e-10)


def test_lagrangian_reports_certified_bounds() -> None:
    problem = one_coordinate_problem()
    result = solve_lagrangian(problem, math.log(3.0) + 0.2)

    assert result.converged
    assert result.objective_lower_bound <= result.objective + 1e-11
    assert result.objective <= result.objective_upper_bound + 1e-11
    assert result.duality_gap <= 2e-11
    assert result.fixed_point_residual <= 2e-6


def test_active_set_eliminates_slowly_decaying_actions() -> None:
    generator = random.Random(20260727)
    for _ in range(13):
        prior = tuple(generator.expovariate(1.0) for _ in range(6))
        rewards = tuple(
            tuple(generator.uniform(-2.0, 2.0) for _ in range(8)) for _ in range(6)
        )
    result = solve_lagrangian(
        FiniteDecisionProblem(prior, rewards),
        0.3,
        tolerance=1e-12,
        max_iterations=1_000,
    )

    assert result.converged
    assert result.iterations < 1_000
    assert result.duality_gap <= 1e-12


def test_log_space_certificate_survives_large_excluded_action_slack() -> None:
    problem = FiniteDecisionProblem((1.0,), ((0.0, 1_000.0),))
    marginal = (1.0, 0.0)
    _, log_normalizers = _channel_from_marginal(problem, 1.0, marginal)
    lower, upper, log_slacks = _marginal_objective_bounds(
        problem, 1.0, marginal, log_normalizers
    )

    assert math.isfinite(lower)
    assert lower == pytest.approx(-1_000.0)
    assert upper == pytest.approx(0.0)
    assert log_slacks == pytest.approx((0.0, 1_000.0))


def test_unconverged_inner_solve_does_not_claim_frontier_convergence() -> None:
    problem = FiniteDecisionProblem(
        (0.4, 0.6),
        ((0.0, 1.0, -1.0), (0.0, -1.0, 1.0)),
    )
    result = solve_frontier(
        problem,
        0.4,
        max_iterations=4,
        lagrangian_max_iterations=1,
    )

    assert not result.converged
    assert result.witness is not None
    assert result.witness.expected_reward >= 0.4
    assert result.lower_bound <= result.upper_bound


def test_frontier_is_invariant_to_state_action_and_duplicate_columns() -> None:
    problems = (
        FiniteDecisionProblem((0.4, 0.6), ((0.0, 1.0, -1.0), (0.0, -1.0, 1.0))),
        FiniteDecisionProblem((0.6, 0.4), ((0.0, -1.0, 1.0), (0.0, 1.0, -1.0))),
        FiniteDecisionProblem((0.4, 0.6), ((-1.0, 0.0, 1.0), (1.0, 0.0, -1.0))),
        FiniteDecisionProblem(
            (0.4, 0.6),
            ((0.0, 1.0, -1.0, 1.0), (0.0, -1.0, 1.0, -1.0)),
        ),
    )
    solutions = tuple(
        solve_frontier(problem, 0.4, tolerance=1e-8) for problem in problems
    )

    assert all(solution.converged for solution in solutions)
    assert (
        max(solution.upper_bound for solution in solutions)
        - min(solution.lower_bound for solution in solutions)
        <= 2e-8
    )
    for problem, solution in zip(problems, solutions, strict=True):
        assert solution.witness is not None
        recomputed = problem.evaluate(solution.witness.channel)
        assert recomputed.action_marginal == pytest.approx(
            solution.witness.action_marginal,
            abs=1e-15,
        )
        assert recomputed.expected_reward == pytest.approx(
            solution.witness.expected_reward,
            abs=1e-15,
        )
        assert recomputed.mutual_information == pytest.approx(
            solution.witness.mutual_information,
            abs=1e-15,
        )
        assert all(
            math.fsum(row) == pytest.approx(1.0, abs=1e-12)
            for row in solution.witness.channel
        )


def test_maximum_reward_endpoint_handles_overlapping_ties() -> None:
    problem = FiniteDecisionProblem(
        (1.0, 1.0, 1.0),
        ((1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)),
    )
    solution = solve_frontier(problem, 1.0, tolerance=1e-9)

    assert solution.converged
    assert solution.witness is not None
    assert solution.witness.expected_reward == pytest.approx(1.0)
    assert solution.lower_bound == pytest.approx(math.log(1.5), abs=1e-9)
    assert solution.upper_bound == pytest.approx(math.log(1.5), abs=1e-9)


@pytest.mark.parametrize("q", [2, 4])
def test_frontier_inversion_round_trip(q: int) -> None:
    u = 1.0
    c = 2.0 if q == 2 else 1.0
    analytic = OneCoordinateFrontier(q, u, c)
    target = 0.65
    budget = analytic.bit_equivalent(target)
    result = invert_frontier(
        one_coordinate_problem(q, u, c),
        budget,
        tolerance=2e-5,
    )

    assert result.converged
    assert result.witness.mutual_information <= budget + 2e-8
    assert result.reward_lower_bound <= target + 2e-5
    assert result.reward_upper_bound >= target - 2e-5


def test_unlimited_information_budget_attains_maximum() -> None:
    problem = one_coordinate_problem()
    result = invert_frontier(problem, math.inf)

    assert result.converged
    assert result.reward_lower_bound == problem.maximum_reward
    assert result.reward_upper_bound == problem.maximum_reward


def test_problem_validation_and_channel_validation() -> None:
    with pytest.raises(ValueError, match="one row per prior"):
        FiniteDecisionProblem((0.5, 0.5), ((1.0,),))
    with pytest.raises(ValueError, match="equal length"):
        FiniteDecisionProblem((0.5, 0.5), ((1.0,), (1.0, 2.0)))

    problem = one_coordinate_problem()
    with pytest.raises(ValueError, match="one row per state"):
        problem.evaluate(((1.0, 0.0, 0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="nonnegative"):
        problem.evaluate(tuple((1.0, -1.0, 0.0, 0.0, 0.0) for _ in range(4)))
