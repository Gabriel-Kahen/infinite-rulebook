"""Mathematical regression tests for the analytic frontiers."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.tensorized import (
    TensorizedFrontier,
    infinite_bit_equivalent,
)

BASE = OneCoordinateFrontier(q=4, u=1.0, c=1.0)


def test_baseline_closed_form() -> None:
    assert BASE.tau == 0.5
    assert BASE.p_star == 0.75
    assert BASE.r_star == 0.5
    assert BASE.kappa == pytest.approx(math.log(3.0), abs=1e-15)

    for reward in (0.0, 0.1, 0.25, 0.5):
        assert BASE.bit_equivalent(reward) == pytest.approx(
            reward * math.log(3.0), abs=1e-15
        )

    for reward in (0.5, 0.625, 0.8, 1.0):
        p = (reward + 1.0) / 2.0
        expected = p * math.log(4.0 * p)
        if p < 1.0:
            expected += (1.0 - p) * math.log(4.0 * (1.0 - p) / 3.0)
        assert BASE.bit_equivalent(reward) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("reward", "expected"),
    [
        (-math.inf, 0.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, math.log(4.0)),
        (1.000_000_1, math.inf),
        (math.inf, math.inf),
    ],
)
def test_one_coordinate_endpoints(reward: float, expected: float) -> None:
    assert BASE.bit_equivalent(reward) == pytest.approx(expected)


def test_information_endpoints() -> None:
    assert BASE.information_at_accuracy(0.25) == 0.0
    assert BASE.information_at_accuracy(1.0) == pytest.approx(math.log(4.0))
    assert BASE.value_at_accuracy(BASE.tau) == pytest.approx(0.0)
    assert BASE.value_at_accuracy(1.0) == pytest.approx(BASE.u)


def test_frontier_is_smooth_at_r_star() -> None:
    step = 1e-6
    center = BASE.bit_equivalent(BASE.r_star)
    left_slope = (center - BASE.bit_equivalent(BASE.r_star - step)) / step
    right_slope = (BASE.bit_equivalent(BASE.r_star + step) - center) / step
    assert left_slope == pytest.approx(BASE.kappa, rel=1e-10)
    assert right_slope == pytest.approx(BASE.kappa, rel=2e-5)


@given(
    rewards=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=2,
        max_size=30,
    )
)
def test_frontier_is_monotone(rewards: list[float]) -> None:
    ordered = sorted(rewards)
    values = [BASE.bit_equivalent(reward) for reward in ordered]
    assert all(left <= right + 1e-14 for left, right in pairwise(values))


@given(
    left=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    right=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_frontier_is_convex(left: float, right: float, weight: float) -> None:
    mixture = weight * left + (1.0 - weight) * right
    lhs = BASE.bit_equivalent(mixture)
    rhs = weight * BASE.bit_equivalent(left)
    rhs += (1.0 - weight) * BASE.bit_equivalent(right)
    assert lhs <= rhs + 2e-14


@given(
    q=st.integers(min_value=2, max_value=20),
    u=st.floats(
        min_value=0.05,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    margin_multiplier=st.floats(
        min_value=1.001,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_bisection_finds_unique_interior_solution(
    q: int, u: float, margin_multiplier: float
) -> None:
    c = (u / (q - 1)) * margin_multiplier
    frontier = OneCoordinateFrontier(q=q, u=u, c=c)
    p = frontier.p_star
    j_prime = math.log((q - 1) * p / (1.0 - p))
    stationarity = frontier.value_at_accuracy(p) * j_prime
    stationarity -= (u + c) * frontier.information_at_accuracy(p)

    assert frontier.tau < p < 1.0
    assert stationarity == pytest.approx(0.0, abs=2e-11 * (u + c))
    assert frontier.r_star > 0.0
    assert frontier.kappa > 0.0


def test_unrepresentable_interior_solution_uses_nearest_lower_float() -> None:
    frontier = OneCoordinateFrontier(q=3, u=0.25, c=9.125)

    assert frontier.p_star == math.nextafter(1.0, 0.0)
    assert 0.0 < frontier.r_star < frontier.u
    assert math.isfinite(frontier.kappa)


@pytest.mark.parametrize("dimensions", [1, 2, 4, 17])
@pytest.mark.parametrize("per_coordinate_reward", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_tensorization(dimensions: int, per_coordinate_reward: float) -> None:
    frontier = TensorizedFrontier(BASE, dimensions)
    expected = dimensions * BASE.bit_equivalent(per_coordinate_reward)
    assert frontier.bit_equivalent(dimensions * per_coordinate_reward) == (
        pytest.approx(expected)
    )


def test_tensorized_endpoints() -> None:
    frontier = TensorizedFrontier(BASE, 5)
    assert frontier.maximum_reward == 5.0
    assert frontier.bit_equivalent(-1.0) == 0.0
    assert frontier.bit_equivalent(5.0) == pytest.approx(5.0 * math.log(4.0))
    assert math.isinf(frontier.bit_equivalent(5.0001))


@given(
    reward=st.floats(
        min_value=0.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_infinite_frontier_is_linear(reward: float) -> None:
    expected = reward * math.log(3.0)
    assert BASE.infinite_bit_equivalent(reward) == pytest.approx(expected)
    assert infinite_bit_equivalent(BASE, reward) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("q", "u", "c"),
    [
        (4, 1.0, 1.0 / 3.0),
        (4, 1.0, 0.2),
        (2, 1.0, 1.0),
        (2, 2.0, 1.0),
    ],
)
def test_strict_margin_is_required(q: int, u: float, c: float) -> None:
    with pytest.raises(ValueError, match="strict negative uninformed margin"):
        OneCoordinateFrontier(q=q, u=u, c=c)


@pytest.mark.parametrize("q", [True, 1.5, "4"])
def test_q_must_be_an_integer(q: object) -> None:
    with pytest.raises(TypeError, match="q must be an integer"):
        OneCoordinateFrontier(q=q)  # type: ignore[arg-type]


@pytest.mark.parametrize("q", [-1, 0, 1])
def test_q_must_have_at_least_two_labels(q: int) -> None:
    with pytest.raises(ValueError, match="q must be at least 2"):
        OneCoordinateFrontier(q=q)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("u", 0.0),
        ("u", -1.0),
        ("c", 0.0),
        ("c", -1.0),
        ("u", math.nan),
        ("c", math.inf),
    ],
)
def test_reward_parameter_validation(field: str, value: float) -> None:
    arguments = {"u": 1.0, "c": 1.0, field: value}
    with pytest.raises((TypeError, ValueError)):
        OneCoordinateFrontier(**arguments)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_scalar_methods_reject_non_real_values(value: object) -> None:
    with pytest.raises(TypeError):
        BASE.bit_equivalent(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BASE.information_at_accuracy(value)  # type: ignore[arg-type]


def test_scalar_methods_reject_nan_and_invalid_accuracy() -> None:
    with pytest.raises(ValueError, match="must not be NaN"):
        BASE.bit_equivalent(math.nan)
    with pytest.raises(ValueError, match="must lie"):
        BASE.information_at_accuracy(0.2)
    with pytest.raises(ValueError, match="must lie"):
        BASE.value_at_accuracy(1.1)


@pytest.mark.parametrize("dimensions", [True, 1.5, "2"])
def test_tensor_dimensions_must_be_integer(dimensions: object) -> None:
    with pytest.raises(TypeError, match="dimensions must be an integer"):
        TensorizedFrontier(BASE, dimensions)  # type: ignore[arg-type]


@pytest.mark.parametrize("dimensions", [-1, 0])
def test_tensor_dimensions_must_be_positive(dimensions: int) -> None:
    with pytest.raises(ValueError, match="dimensions must be at least 1"):
        TensorizedFrontier(BASE, dimensions)
