"""Semantic and mathematical tests for ALEA, TRIVIA, and PUBLIC."""

from __future__ import annotations

import math
from collections import Counter
from itertools import product

import pytest

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.controls import (
    AleaRulebook,
    CappedPublicRulebook,
    PublicBonusSchedule,
    PublicDeploymentAction,
    QueryNamespace,
    RulebookRuntime,
    SymbolicQuery,
    TriviaRulebook,
    UnboundedPublicRulebook,
)
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.environments.redundant import CappedRedundantRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.frontier.controls import (
    PublicCFrontier,
    alea_frontier_problem,
    alea_persistent_information_nats,
    augment_with_independent_trivia,
    augment_with_public_c,
    enumerate_public_c_rulebook,
    enumerate_trivia_rulebook,
    public_c_bit_equivalent,
    public_u_bit_equivalent,
    public_u_witness,
    trivia_invariant_bit_equivalent,
)
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.inversion import solve_frontier
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.redundancy import enumerate_redundant_rulebook
from infinite_rulebook.frontier.rulebook_problem import (
    enumerate_independent_rulebook,
)

BASE = OneCoordinateFrontier()


def correct_action(
    environment: RulebookRuntime,
    indices: tuple[int, ...],
) -> DeploymentAction:
    return DeploymentAction((index, environment.label(index)) for index in indices)


def test_alea_is_stationary_cosmetic_observation_noise_only() -> None:
    seed = "paired-alea"
    base = IndependentRulebook(seed)
    environment = AleaRulebook(
        base,
        cosmetic_seed="paired-alea-tape",
        cosmetic_alphabet=65_537,
    )
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


def mutual_information_of_uniform_pairs(pairs: list[tuple[int, int]]) -> float:
    joint = Counter(pairs)
    left = Counter(value[0] for value in pairs)
    right = Counter(value[1] for value in pairs)
    total = len(pairs)
    return math.fsum(
        count
        / total
        * math.log((count * total) / (left[left_value] * right[right_value]))
        for (left_value, right_value), count in joint.items()
    )


def test_separate_alea_seed_has_exactly_zero_persistent_theta_information() -> None:
    key = SemanticObservationKey("public-tape-key", 2, 1, 0, "alea")
    pairs = []
    for latent_seed, cosmetic_seed in product(range(4), repeat=2):
        base = IndependentRulebook(latent_seed)
        environment = AleaRulebook(base, cosmetic_seed=cosmetic_seed)
        pairs.append((base.label(1), environment.cosmetic_value(key)))
    base_problem = enumerate_independent_rulebook(1).problem

    assert mutual_information_of_uniform_pairs(pairs) == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert alea_persistent_information_nats() == 0.0
    assert alea_frontier_problem(base_problem) is base_problem


def test_trivia_is_persistent_queryable_and_reward_irrelevant() -> None:
    base = IndependentRulebook("useful")
    environment = TriviaRulebook(base, trivia_seed="distractor")
    channel = QarySymmetricChannel(q=4, epsilon=0.1)
    useful_query = SymbolicQuery(QueryNamespace.REWARD, 11)
    trivia_query = SymbolicQuery(QueryNamespace.TRIVIA, 11)
    shared_key = SemanticObservationKey(
        base.seed,
        round_index=3,
        rule_index=11,
        channel="p1",
    )

    assert environment.query_label(trivia_query) == environment.trivia_label(11)
    assert environment.trivia_label(11) == environment.trivia_label(11)
    assert (
        environment.observation_key(
            useful_query,
            shared_key,
        )
        is shared_key
    )
    assert shared_key != environment.observation_key(trivia_query, shared_key)
    expected_useful = channel.observe(base.label(11), shared_key)
    assert (
        environment.observe(
            useful_query,
            channel,
            shared_key,
        ).value
        == expected_useful
    )
    assert environment.observe(useful_query, channel, shared_key).query == useful_query
    assert environment.observe(trivia_query, channel, shared_key).query == trivia_query

    action = correct_action(environment, (1, 2, 3))
    assert environment.labels((1, 2, 3)) == base.labels((1, 2, 3))
    assert environment.evaluate(action) == base.evaluate(action)


def test_trivia_labels_are_query_order_invariant() -> None:
    indices = (1, 2, 19, 10**6)
    forward = TriviaRulebook(IndependentRulebook(1), trivia_seed=2)
    reverse = TriviaRulebook(IndependentRulebook(1), trivia_seed=2)
    expected = {index: forward.trivia_label(index) for index in indices}
    observed = {index: reverse.trivia_label(index) for index in reversed(indices)}
    assert observed == expected


def test_separate_trivia_seed_matches_independent_product_semantics() -> None:
    pairs = []
    for latent_seed, trivia_seed in product(range(4), repeat=2):
        environment = TriviaRulebook(
            IndependentRulebook(latent_seed),
            trivia_seed=trivia_seed,
        )
        pairs.append((environment.label(1), environment.trivia_label(1)))
    assert mutual_information_of_uniform_pairs(pairs) == pytest.approx(
        0.0,
        abs=1e-15,
    )


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
    first = UnboundedPublicRulebook(
        IndependentRulebook("first"),
        public_unit_reward=0.5,
    )
    second = UnboundedPublicRulebook(
        IndependentRulebook("second"),
        public_unit_reward=0.5,
    )
    public_only = PublicDeploymentAction(public_choice=17)

    assert first.evaluate(public_only) == 8.5
    assert second.evaluate(public_only) == 8.5
    with pytest.raises(ValueError, match="non-finite"):
        first.public_reward(10**400)


def test_public_u_extreme_witness_agrees_with_runtime() -> None:
    unit = 1e-308
    witness = public_u_witness(100.0, public_unit_reward=unit)
    environment = UnboundedPublicRulebook(
        IndependentRulebook(1),
        public_unit_reward=unit,
    )
    assert environment.public_reward(witness.public_choice) == (witness.expected_reward)
    assert witness.expected_reward >= 100.0


def test_public_c_runtime_is_fixed_bounded_and_composable() -> None:
    schedule = PublicBonusSchedule((0.0, 0.25, 0.1))
    environment = CappedPublicRulebook(
        IndependentRulebook("public-c"),
        public_schedule=schedule,
    )
    hidden = correct_action(environment, (1, 2))

    assert schedule.maximum_reward == 0.25
    assert schedule.maximizing_choice == 1
    assert environment.maximum_public_reward == 0.25
    assert environment.evaluate(PublicDeploymentAction(hidden, 1)) == 2.25
    assert environment.evaluate(PublicDeploymentAction(hidden, 2)) == 2.1
    with pytest.raises(ValueError, match="outside"):
        environment.evaluate(PublicDeploymentAction(hidden, 3))


def test_runtime_controls_compose_over_redundancy_and_each_other() -> None:
    redundant = CappedRedundantRulebook(
        "composed",
        core_dimensions=2,
        max_derived_support=3,
    )
    trivia = TriviaRulebook(redundant, trivia_seed="trivia")
    alea = AleaRulebook(trivia, cosmetic_seed="alea")
    environment = CappedPublicRulebook(
        alea,
        public_schedule=PublicBonusSchedule((0.0, 0.5)),
    )
    hidden = correct_action(environment, (1, 2, 3))
    assert environment.evaluate(PublicDeploymentAction(hidden, 1)) == 3.5

    query = SymbolicQuery(QueryNamespace.TRIVIA, 7)
    key = SemanticObservationKey("composed", 0, 7)
    observed = environment.observe_query(
        query,
        QarySymmetricChannel(4, 0.1),
        key,
    )
    assert observed.symbolic.query == query
    assert observed.cosmetic_value is not None
    assert 0 <= observed.cosmetic_value < alea.cosmetic_alphabet


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
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_trivia_rulebook(
            1,
            10**9,
            max_matrix_entries=100,
        )


def test_generic_finite_controls_compose_over_redundant_base() -> None:
    redundant = enumerate_redundant_rulebook(1, 1, 1)
    trivia = augment_with_independent_trivia(
        redundant.problem,
        trivia_alphabet_size=4,
        trivia_dimensions=1,
    )
    public = augment_with_public_c(
        trivia,
        PublicBonusSchedule((0.0, 0.25)),
    )
    solution = solve_frontier(public, 0.5, tolerance=1e-9)
    expected = BASE.bit_equivalent(0.25)

    assert trivia.state_count == redundant.problem.state_count * 4
    assert trivia.action_count == redundant.problem.action_count
    assert public.action_count == trivia.action_count * 2
    assert solution.lower_bound <= expected + 2e-7
    assert solution.upper_bound >= expected - 2e-7


def test_generic_control_endpoints_tolerate_only_float_roundoff() -> None:
    base = FiniteDecisionProblem(
        prior=(0.2, 0.8),
        rewards=((0.0, 2.0, -1.0), (0.0, -1.0, 1.0)),
    )
    trivia = augment_with_independent_trivia(
        base,
        trivia_alphabet_size=3,
        trivia_dimensions=2,
    )
    schedule = PublicBonusSchedule((0.0, 0.4, 0.1))
    public = augment_with_public_c(base, schedule)
    targets = (
        (trivia, base.maximum_reward),
        (
            public,
            math.fsum((base.maximum_reward, schedule.maximum_reward)),
        ),
    )

    for problem, requested_target in targets:
        assert requested_target >= problem.maximum_reward
        solution = solve_frontier(problem, requested_target)
        assert solution.target_reward == requested_target
        assert solution.effective_target_reward == problem.maximum_reward
        assert solution.witness is not None
        assert solution.witness == problem.evaluate(solution.witness.channel)
        assert solution.witness.expected_reward == problem.maximum_reward
        assert math.isfinite(solution.upper_bound)

    ordinary_infeasible = math.nextafter(base.maximum_reward, math.inf)
    ordinary_solution = solve_frontier(base, ordinary_infeasible)
    assert ordinary_solution.effective_target_reward == ordinary_infeasible
    assert ordinary_solution.witness is None
    assert math.isinf(ordinary_solution.upper_bound)

    # The solver's documented numerical endpoint tolerance is deliberately
    # larger than a few ulps, so move beyond it for this assertion.
    far_infeasible = public.maximum_reward + 256 * math.ulp(abs(public.maximum_reward))
    solution = solve_frontier(public, far_infeasible)
    assert solution.witness is None
    assert math.isinf(solution.upper_bound)


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
        AleaRulebook(
            IndependentRulebook(1),
            cosmetic_seed=2,
        ).observe(
            2,
            QarySymmetricChannel(4, 0.1),
            SemanticObservationKey(1, 0, 1),
        )
