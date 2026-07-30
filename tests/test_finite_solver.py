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


def test_normal_channel_preserves_legacy_mutual_information_float() -> None:
    problem = FiniteDecisionProblem(
        prior=(0.25, 0.75),
        rewards=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    channel = ((0.125, 0.625, 0.25), (0.5, 0.25, 0.25))
    marginal = (0.40625, 0.34375, 0.25)
    expected = max(
        0.0,
        math.fsum(
            problem.prior[state]
            * conditional
            * math.log(conditional / marginal[action])
            for state, row in enumerate(channel)
            for action, conditional in enumerate(row)
        ),
    )

    witness = problem.evaluate(channel)

    assert expected.hex() == "0x1.321c043e5ca5dp-4"
    assert witness.mutual_information == expected


def test_direct_channel_ignores_joint_mass_below_float_range() -> None:
    smallest_positive = math.ulp(0.0)
    problem = FiniteDecisionProblem(
        prior=(0.7, 0.3),
        rewards=((0.0, 0.0), (0.0, 0.0)),
    )

    witness = problem.evaluate(
        (
            (0.0, 1.0),
            (smallest_positive, 1.0),
        )
    )

    assert witness.action_marginal == (0.0, 1.0)
    assert witness.mutual_information == 0.0


def test_direct_channel_uses_log_difference_for_extreme_probability_ratio() -> None:
    smallest_positive = math.ulp(0.0)
    problem = FiniteDecisionProblem(
        prior=(smallest_positive, 1.0),
        rewards=((0.0, 0.0), (0.0, 0.0)),
    )

    witness = problem.evaluate(((1.0, 0.0), (0.0, 1.0)))

    assert math.isfinite(witness.mutual_information)
    assert witness.mutual_information > 0.0


def test_direct_channel_recovers_representable_information_from_tiny_joint() -> None:
    smallest_positive = math.ulp(0.0)
    problem = FiniteDecisionProblem(
        prior=(smallest_positive, 1.0),
        rewards=((0.0, 0.0), (0.0, 0.0)),
    )

    witness = problem.evaluate(((0.25, 0.75), (0.0, 1.0)))

    assert witness.action_marginal == (0.0, 1.0)
    assert math.isfinite(witness.mutual_information)
    assert witness.mutual_information > 0.0


def test_warm_start_subnormal_channel_retains_certified_solver_semantics() -> None:
    problem = FiniteDecisionProblem(
        prior=(0.7, 0.3),
        rewards=(
            (-1.5, -1.75, 1.7),
            (0.83, -1.84, -0.32),
        ),
    )
    face = solve_lagrangian(problem, 1.0, tolerance=1e-12)

    assert face.converged
    assert face.witness.action_marginal == (0.0, 0.0, 1.0)

    warm = solve_lagrangian(
        problem,
        2.0,
        tolerance=1e-12,
        initial_action_marginal=face.witness.action_marginal,
    )
    cold = solve_lagrangian(problem, 2.0, tolerance=1e-12)

    assert warm.converged
    assert warm.objective_lower_bound <= warm.objective <= warm.objective_upper_bound
    assert warm.duality_gap <= 1e-12
    assert warm.witness.expected_reward == pytest.approx(
        cold.witness.expected_reward,
        abs=1e-11,
    )
    assert warm.witness.mutual_information == pytest.approx(
        cold.witness.mutual_information,
        abs=1e-11,
    )
    target = problem.zero_information_reward + 0.37 * (
        problem.maximum_reward - problem.zero_information_reward
    )
    frontier = solve_frontier(
        problem,
        target,
        tolerance=2e-7,
        bound_tolerance=2e-7,
    )
    assert frontier.converged
    assert frontier.witness is not None
    assert frontier.witness.expected_reward >= target
    assert frontier.lower_bound <= frontier.upper_bound


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


def test_randomized_small_frontiers_remain_finite_and_certified() -> None:
    generator = random.Random(20260730)
    for _ in range(20):
        state_count = generator.randint(2, 5)
        action_count = generator.randint(2, 6)
        problem = FiniteDecisionProblem(
            tuple(generator.random() + 0.1 for _ in range(state_count)),
            tuple(
                tuple(generator.uniform(-2.0, 2.0) for _ in range(action_count))
                for _ in range(state_count)
            ),
        )
        target = problem.zero_information_reward + 0.37 * (
            problem.maximum_reward - problem.zero_information_reward
        )

        result = solve_frontier(
            problem,
            target,
            tolerance=2e-7,
            bound_tolerance=2e-7,
        )

        assert result.converged
        assert result.witness is not None
        assert result.witness.expected_reward >= target - 1e-12
        assert math.isfinite(result.witness.mutual_information)
        assert result.lower_bound <= result.upper_bound
        assert result.duality_gap <= 2e-7


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


def test_maximum_reward_endpoint_is_reward_scale_invariant() -> None:
    problem = FiniteDecisionProblem(
        (0.5, 0.5),
        ((1e-12, 0.0), (0.0, 1e-12)),
    )
    solution = solve_frontier(problem, problem.maximum_reward)

    assert solution.converged
    assert solution.witness is not None
    assert solution.witness.expected_reward == problem.maximum_reward
    assert solution.lower_bound == pytest.approx(math.log(2.0), abs=1e-10)
    assert solution.upper_bound == pytest.approx(math.log(2.0), abs=1e-10)


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
