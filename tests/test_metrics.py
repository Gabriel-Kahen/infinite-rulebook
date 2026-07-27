"""Exact symbolic metric and interval propagation tests."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.metrics import (
    CheckpointInterpolation,
    FrontierCurve,
    FrontierPoint,
    FrontierUpperWitness,
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


def _frontier_point(reward: float, lower: float, upper: float) -> FrontierPoint:
    witness_payload = {"reward": reward, "information_nats": upper}
    return FrontierPoint(
        reward,
        MetricInterval(lower, upper, "nats"),
        FrontierUpperWitness(reward, upper, semantic_hash(witness_payload)),
    )


@pytest.fixture
def exact_curve() -> FrontierCurve:
    return FrontierCurve(
        points=(
            _frontier_point(0.0, 0.0, 0.0),
            _frontier_point(1.0, 1.0, 1.0),
            _frontier_point(2.0, 3.0, 3.0),
        ),
        zero_information_reward=0.0,
        maximum_reward=2.0,
        semantic_hash=semantic_hash({"problem": "test-frontier"}),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )


def test_bit_equivalent_lookup_handles_baseline_grid_and_infeasibility(
    exact_curve: FrontierCurve,
) -> None:
    assert lookup_bit_equivalent(exact_curve, -1.0) == MetricInterval(0.0, 0.0, "nats")
    assert lookup_bit_equivalent(exact_curve, 0.5) == MetricInterval(0.0, 0.5, "nats")
    assert lookup_bit_equivalent(exact_curve, 1.5) == MetricInterval(1.0, 2.0, "nats")
    assert math.isinf(lookup_bit_equivalent(exact_curve, 2.1).lower)


def test_sparse_bit_equivalent_integration_uses_elapsed_round_weights(
    exact_curve: FrontierCurve,
) -> None:
    series = integrate_bit_equivalent(
        exact_curve,
        (
            TimedReward(0, 0.0),
            TimedReward(1, 1.0),
            TimedReward(4, 2.0),
        ),
        horizon=4,
        interpolation=CheckpointInterpolation.LEFT_HOLD,
    )

    assert series.average == MetricInterval(0.75, 0.75, "nats")
    assert series.average.lower != pytest.approx((0.0 + 1.0 + 3.0) / 3.0)


def test_linear_integration_splits_at_frontier_knots(
    exact_curve: FrontierCurve,
) -> None:
    series = integrate_bit_equivalent(
        exact_curve,
        (TimedReward(0, 0.0), TimedReward(2, 2.0)),
        horizon=2,
        interpolation=CheckpointInterpolation.LINEAR,
    )

    assert series.average == MetricInterval(0.5, 1.25, "nats")


def test_integration_rejects_ambiguous_checkpoint_coverage(
    exact_curve: FrontierCurve,
) -> None:
    with pytest.raises(ValueError, match="span exactly"):
        integrate_bit_equivalent(
            exact_curve,
            (TimedReward(1, 0.0), TimedReward(4, 2.0)),
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
    assert frontier_regret(
        exact_curve,
        attained_reward=0.6,
        information_budget_nats=1.0,
    ) == MetricInterval(0.4, 1.4, "reward")

    relevant = frontier_regret(
        exact_curve,
        attained_reward=0.6,
        information_budget_nats=1.0,
    )
    full_with_distractor = frontier_regret(
        exact_curve,
        attained_reward=0.6,
        information_budget_nats=2.0,
    )
    assert full_with_distractor.lower > relevant.lower


def test_certified_lookup_contains_nonlinear_exact_frontier() -> None:
    from infinite_rulebook.frontier import OneCoordinateFrontier

    exact = OneCoordinateFrontier()
    curve = FrontierCurve(
        points=(
            _frontier_point(0.0, 0.0, 0.0),
            _frontier_point(
                exact.r_star,
                exact.bit_equivalent(exact.r_star),
                exact.bit_equivalent(exact.r_star),
            ),
            _frontier_point(
                exact.u,
                math.log(exact.q),
                math.log(exact.q),
            ),
        ),
        zero_information_reward=0.0,
        maximum_reward=exact.u,
        semantic_hash=semantic_hash({"problem": "one-coordinate"}),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )
    target = 0.75
    bounds = lookup_bit_equivalent(curve, target)

    assert bounds.lower <= exact.bit_equivalent(target) <= bounds.upper
    assert bounds.lower < bounds.upper


def test_frontier_rejects_inconsistent_zero_information_endpoint() -> None:
    with pytest.raises(ValueError, match="exactly"):
        FrontierCurve(
            points=(
                _frontier_point(0.0, 0.1, 0.1),
                _frontier_point(1.0, 1.0, 1.0),
            ),
            zero_information_reward=0.0,
            maximum_reward=1.0,
            semantic_hash=semantic_hash({"problem": "bad"}),
            upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
        )

    with pytest.raises(ValueError, match="feasible witness information"):
        FrontierPoint(
            0.5,
            MetricInterval(0.1, 0.2, "nats"),
            FrontierUpperWitness(
                0.5,
                0.1,
                semantic_hash({"invalid": "witness"}),
            ),
        )


def test_frontier_point_binds_existing_solver_witness() -> None:
    from infinite_rulebook.frontier import one_coordinate_problem, solve_frontier

    solution = solve_frontier(one_coordinate_problem(), 0.0)
    point = FrontierPoint.from_frontier_solution(solution)

    assert point.reward == 0.0
    assert point.information == MetricInterval(0.0, 0.0, "nats")
    assert point.upper_witness.witness_hash == semantic_hash(solution.witness)


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
