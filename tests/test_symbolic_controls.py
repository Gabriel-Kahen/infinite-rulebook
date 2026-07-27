"""Semantic and mathematical tests for ALEA, TRIVIA, and PUBLIC."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.controls import (
    AleaRulebook,
    CappedPublicRulebook,
    PublicBonusSchedule,
    PublicDeploymentAction,
    QueryNamespace,
    SymbolicQuery,
    TriviaRulebook,
    UnboundedPublicRulebook,
)
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.frontier.controls import (
    PublicCFrontier,
    alea_frontier_problem,
    alea_persistent_information_nats,
    enumerate_public_c_rulebook,
    enumerate_trivia_rulebook,
    public_c_bit_equivalent,
    public_u_bit_equivalent,
    public_u_witness,
    trivia_invariant_bit_equivalent,
)
from infinite_rulebook.frontier.inversion import solve_frontier
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.rulebook_problem import (
    enumerate_independent_rulebook,
)

BASE = OneCoordinateFrontier()


def correct_action(
    environment: IndependentRulebook,
    indices: tuple[int, ...],
) -> DeploymentAction:
    return DeploymentAction((index, environment.label(index)) for index in indices)


def test_alea_is_stationary_cosmetic_observation_noise_only() -> None:
    seed = "paired-alea"
    environment = AleaRulebook(seed, cosmetic_alphabet=65_537)
    base = IndependentRulebook(seed)
    channel = QarySymmetricChannel(q=4, epsilon=0.2)
    indices = (1, 7, 1000)
    keys = tuple(
        SemanticObservationKey(
            seed,
            round_index=round_index,
            rule_index=index,
            query_ordinal=ordinal,
            channel="alea-useful",
        )
        for ordinal, (round_index, index) in enumerate(
            zip(range(3), indices, strict=True)
        )
    )

    forward = {
        key: environment.observe(index, channel, key)
        for index, key in zip(indices, keys, strict=True)
    }
    reverse = {
        key: environment.observe(index, channel, key)
        for index, key in reversed(tuple(zip(indices, keys, strict=True)))
    }

    assert forward == reverse
    assert environment.labels(indices) == base.labels(indices)
    action = correct_action(environment, indices)
    assert environment.evaluate(action) == base.evaluate(action)


def test_alea_has_exactly_zero_persistent_theta_information() -> None:
    # An explicit finite factorization P(theta, u)=P(theta)P(u) has zero MI.
    theta_prior = (0.25,) * 4
    cosmetic_prior = (0.2, 0.3, 0.5)
    information = math.fsum(
        theta * cosmetic * math.log((theta * cosmetic) / (theta * cosmetic))
        for theta in theta_prior
        for cosmetic in cosmetic_prior
    )
    base_problem = enumerate_independent_rulebook(1).problem

    assert information == 0.0
    assert alea_persistent_information_nats() == 0.0
    assert alea_frontier_problem(base_problem) is base_problem


def test_trivia_is_persistent_queryable_and_reward_irrelevant() -> None:
    environment = TriviaRulebook("useful", trivia_seed="distractor")
    channel = QarySymmetricChannel(q=4, epsilon=0.1)
    useful_query = SymbolicQuery(QueryNamespace.REWARD, 11)
    trivia_query = SymbolicQuery(QueryNamespace.TRIVIA, 11)
    useful_key = SemanticObservationKey(
        environment.seed,
        round_index=3,
        rule_index=11,
        channel="useful",
    )
    trivia_key = SemanticObservationKey(
        environment.seed,
        round_index=3,
        rule_index=11,
        channel="trivia",
    )

    assert environment.query_label(trivia_query) == environment.trivia_label(11)
    assert environment.trivia_label(11) == environment.trivia_label(11)
    assert environment.observe(useful_query, channel, useful_key).query == useful_query
    assert environment.observe(trivia_query, channel, trivia_key).query == trivia_query

    base = IndependentRulebook("useful")
    action = correct_action(environment, (1, 2, 3))
    assert environment.labels((1, 2, 3)) == base.labels((1, 2, 3))
    assert environment.evaluate(action) == base.evaluate(action)


def test_trivia_labels_are_query_order_invariant() -> None:
    indices = (1, 2, 19, 10**6)
    forward = TriviaRulebook(1, trivia_seed=2)
    reverse = TriviaRulebook(1, trivia_seed=2)
    expected = {index: forward.trivia_label(index) for index in indices}
    observed = {index: reverse.trivia_label(index) for index in reversed(indices)}
    assert observed == expected


def test_trivia_projection_has_exact_counts_rows_and_natural_log_endpoint() -> None:
    augmented = enumerate_trivia_rulebook(1, 2)
    base = enumerate_independent_rulebook(1)

    assert augmented.problem.state_count == 4**3
    assert augmented.problem.action_count == 5
    assert len(set(augmented.states)) == 4**3
    assert all(probability == 1 / 4**3 for probability in augmented.problem.prior)
    for useful_label in range(1, 5):
        matching_rows = {
            augmented.problem.rewards[index]
            for index, state in enumerate(augmented.states)
            if state[0] == useful_label
        }
        assert matching_rows == {base.problem.rewards[useful_label - 1]}
    assert augmented.problem.maximizing_channel().mutual_information == (
        pytest.approx(math.log(4))
    )


@pytest.mark.parametrize("target", [-1.0, 0.0, 0.25, 0.75, 1.0, 1.01])
def test_trivia_leaves_exact_reward_information_frontier_invariant(
    target: float,
) -> None:
    augmented = enumerate_trivia_rulebook(1, 1)
    expected = BASE.bit_equivalent(target)
    assert trivia_invariant_bit_equivalent(BASE.bit_equivalent, target) == (
        pytest.approx(expected)
    )

    solution = solve_frontier(
        augmented.problem,
        target,
        tolerance=2e-10,
        bound_tolerance=2e-8,
    )
    if math.isinf(expected):
        assert math.isinf(solution.lower_bound)
        assert math.isinf(solution.upper_bound)
    else:
        assert solution.lower_bound <= expected + 3e-8
        assert solution.upper_bound >= expected - 3e-8
        assert solution.duality_gap <= 2e-6


@pytest.mark.parametrize(
    "target",
    [-math.inf, -1.0, 0.0, 1.0, 1e12, math.nextafter(math.inf, 0.0)],
)
def test_public_u_is_zero_at_every_finite_threshold(target: float) -> None:
    assert public_u_bit_equivalent(target) == 0.0
    if math.isfinite(target):
        witness = public_u_witness(target, public_unit_reward=0.25)
        assert witness.expected_reward >= target
        assert witness.mutual_information == 0.0
        assert isinstance(witness.public_choice, int)


def test_public_u_infinite_threshold_is_infeasible() -> None:
    assert math.isinf(public_u_bit_equivalent(math.inf))
    with pytest.raises(ValueError, match="finite"):
        public_u_witness(math.inf)


def test_public_u_runtime_has_finite_state_independent_bonus() -> None:
    first = UnboundedPublicRulebook("first", public_unit_reward=0.5)
    second = UnboundedPublicRulebook("second", public_unit_reward=0.5)
    public_only = PublicDeploymentAction(public_choice=17)

    assert first.evaluate(public_only) == 8.5
    assert second.evaluate(public_only) == 8.5
    with pytest.raises(ValueError, match="non-finite"):
        first.public_reward(10**400)


def test_public_c_runtime_is_fixed_bounded_and_composable() -> None:
    schedule = PublicBonusSchedule((0.0, 0.25, 0.1))
    environment = CappedPublicRulebook("public-c", public_schedule=schedule)
    hidden = correct_action(environment, (1, 2))

    assert schedule.maximum_reward == 0.25
    assert schedule.maximizing_choice == 1
    assert environment.maximum_public_reward == 0.25
    assert environment.evaluate(PublicDeploymentAction(hidden, 1)) == 2.25
    assert environment.evaluate(PublicDeploymentAction(hidden, 2)) == 2.1
    with pytest.raises(ValueError, match="outside"):
        environment.evaluate(PublicDeploymentAction(hidden, 3))


@pytest.mark.parametrize(
    "target",
    [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.250_001, math.inf],
)
def test_public_c_exact_shifted_and_truncated_identity(target: float) -> None:
    expected = BASE.bit_equivalent(target - 0.25)
    assert public_c_bit_equivalent(
        BASE.bit_equivalent,
        target,
        maximum_public_reward=0.25,
    ) == pytest.approx(expected)
    assert PublicCFrontier(BASE.bit_equivalent, 0.25).bit_equivalent(target) == (
        pytest.approx(expected)
    )


@pytest.mark.parametrize("target", [0.25, 0.5, 0.75, 1.0, 1.25, 1.251])
def test_public_c_finite_solver_matches_shifted_base_frontier(
    target: float,
) -> None:
    enumerated = enumerate_public_c_rulebook(
        1,
        PublicBonusSchedule((0.0, 0.1, 0.25)),
    )
    expected = BASE.bit_equivalent(target - 0.25)
    solution = solve_frontier(
        enumerated.problem,
        target,
        tolerance=2e-10,
        bound_tolerance=2e-8,
    )

    assert enumerated.problem.state_count == 4
    assert enumerated.problem.action_count == 15
    if math.isinf(expected):
        assert math.isinf(solution.lower_bound)
        assert math.isinf(solution.upper_bound)
    else:
        assert solution.lower_bound <= expected + 3e-8
        assert solution.upper_bound >= expected - 3e-8
        assert solution.duality_gap <= 2e-6


def test_control_projection_allocation_guards_precede_materialization() -> None:
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_trivia_rulebook(5, 20, max_matrix_entries=100)
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_public_c_rulebook(
            10,
            PublicBonusSchedule((0.0, 1.0)),
            max_matrix_entries=100,
        )


@pytest.mark.parametrize(
    "rewards",
    [(), (math.nan,), (math.inf,), (-0.1,), (True,)],
)
def test_public_c_rejects_invalid_bonus_schedules(
    rewards: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PublicBonusSchedule(rewards)  # type: ignore[arg-type]


def test_control_runtime_types_reject_ambiguous_values() -> None:
    with pytest.raises(TypeError):
        SymbolicQuery("trivia", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PublicDeploymentAction(public_choice=True)
    with pytest.raises(ValueError):
        PublicDeploymentAction(public_choice=-1)
    with pytest.raises(ValueError, match="must match"):
        AleaRulebook(1).observe(
            2,
            QarySymmetricChannel(4, 0.1),
            SemanticObservationKey(1, 0, 1),
        )
