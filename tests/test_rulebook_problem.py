"""Integration tests for finite Rulebook enumeration."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.frontier.inversion import solve_frontier
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.rulebook_problem import (
    enumerate_independent_rulebook,
)
from infinite_rulebook.frontier.tensorized import TensorizedFrontier


def test_one_coordinate_enumeration_matches_semantics() -> None:
    enumerated = enumerate_independent_rulebook(1)

    assert enumerated.problem.state_count == 4
    assert enumerated.problem.action_count == 5
    assert enumerated.states == ((1,), (2,), (3,), (4,))
    assert enumerated.actions[0].entries == ()
    assert enumerated.actions[-1].entries == ((1, 4),)
    assert enumerated.problem.rewards[0] == (0.0, 1.0, -1.0, -1.0, -1.0)


@pytest.mark.parametrize("reward", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_finite_solver_matches_one_coordinate_frontier(reward: float) -> None:
    enumerated = enumerate_independent_rulebook(1)
    expected = OneCoordinateFrontier().bit_equivalent(reward)
    solution = solve_frontier(
        enumerated.problem,
        reward,
        tolerance=1e-11,
        bound_tolerance=2e-9,
    )

    assert solution.lower_bound <= expected + 2e-9
    assert solution.upper_bound >= expected - 2e-9
    assert solution.duality_gap <= 2e-8
    assert solution.witness.expected_reward >= reward - 1e-10


@pytest.mark.parametrize("reward", [0.5, 1.0, 1.5])
def test_two_coordinate_solver_matches_tensorization(reward: float) -> None:
    enumerated = enumerate_independent_rulebook(2)
    expected = TensorizedFrontier(OneCoordinateFrontier(), 2).bit_equivalent(reward)
    solution = solve_frontier(
        enumerated.problem,
        reward,
        tolerance=2e-11,
        bound_tolerance=2e-8,
    )

    assert solution.lower_bound <= expected + 5e-8
    assert solution.upper_bound >= expected - 5e-8
    assert solution.duality_gap <= 1e-6


def test_enumeration_size_guard() -> None:
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_independent_rulebook(5, max_matrix_entries=1_000)


def test_custom_reward_spec() -> None:
    spec = RewardSpec(q=3, u=2.0, c=1.5)
    enumerated = enumerate_independent_rulebook(1, spec)

    assert enumerated.problem.maximum_reward == pytest.approx(2.0)
    assert enumerated.problem.constant_action_reward == pytest.approx(0.0)
    assert math.isclose(sum(enumerated.problem.prior), 1.0)
