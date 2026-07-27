from __future__ import annotations

import math
from itertools import pairwise

import pytest

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.mixed import MixedRulebook
from infinite_rulebook.environments.redundant import (
    CappedRedundantRulebook,
    UnrestrictedRedundantRulebook,
)
from infinite_rulebook.frontier.inversion import solve_frontier
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.redundancy import (
    RareBurstWitness,
    capped_redundant_information_upper_bound,
    enumerate_mixed_rulebook,
    enumerate_redundant_rulebook,
    rare_burst_sequence,
    unrestricted_redundant_bit_equivalent,
)
from infinite_rulebook.frontier.rulebook_problem import (
    enumerate_independent_rulebook,
)


def correct_action(environment: object, indices: range) -> DeploymentAction:
    return DeploymentAction(
        (index, environment.label(index))  # type: ignore[attr-defined]
        for index in indices
    )


def test_redundant_labels_are_stationary_and_order_invariant() -> None:
    indices = (1, 2, 5, 19, 10**6, 2**70)
    forward = UnrestrictedRedundantRulebook("red", core_dimensions=3)
    reverse = UnrestrictedRedundantRulebook("red", core_dimensions=3)
    expected = {index: forward.label(index) for index in indices}
    observed = {index: reverse.label(index) for index in reversed(indices)}
    assert observed == expected
    assert forward.core == reverse.core


def test_public_balanced_surface_functions_repeat() -> None:
    environment = UnrestrictedRedundantRulebook(9, core_dimensions=3)
    period = environment.core_dimensions * environment.reward_spec.q
    for index in range(1, period + 1):
        assert environment.surface_map(index) == environment.surface_map(index + period)
        assert environment.label(index) == environment.label(index + period)
    for component in range(environment.core_dimensions):
        labels = [
            environment.label(1 + component + offset * environment.core_dimensions)
            for offset in range(environment.reward_spec.q)
        ]
        assert sorted(labels) == list(range(1, environment.reward_spec.q + 1))


def test_unrestricted_redundant_reward_is_exact_and_unbounded() -> None:
    environment = UnrestrictedRedundantRulebook(22, core_dimensions=2)
    action = correct_action(environment, range(1, 101))
    assert environment.evaluate(action) == 100 * environment.reward_spec.u

    wrong = environment.label(1) % environment.reward_spec.q + 1
    mixed = DeploymentAction([(1, wrong), (2, environment.label(2))])
    assert environment.evaluate(mixed) == (
        environment.reward_spec.u - environment.reward_spec.c
    )


def test_red_c_enforces_fixed_support_cap_and_maximum_reward() -> None:
    environment = CappedRedundantRulebook(12, core_dimensions=2, max_derived_support=3)
    assert environment.maximum_reward == 3.0
    assert environment.evaluate(correct_action(environment, range(1, 4))) == 3.0
    with pytest.raises(ValueError, match="max_derived_support"):
        environment.evaluate(correct_action(environment, range(1, 5)))


def test_zero_redundant_caps_disable_redundant_deployment() -> None:
    redundant = CappedRedundantRulebook("red-zero", max_derived_support=0)
    mixed = MixedRulebook("mix-zero", max_redundant_support=0)

    assert redundant.maximum_reward == 0.0
    assert redundant.evaluate(DeploymentAction()) == 0.0
    assert mixed.maximum_redundant_reward == 0.0
    assert mixed.evaluate(correct_action(mixed, range(1, 6, 2))) == 3.0
    with pytest.raises(ValueError, match="max_derived_support"):
        redundant.evaluate(correct_action(redundant, range(1, 2)))
    with pytest.raises(ValueError, match="max_redundant_support"):
        mixed.evaluate(correct_action(mixed, range(2, 3)))
    assert (
        capped_redundant_information_upper_bound(
            0.0,
            max_derived_support=0,
            core_dimensions=1,
            q=4,
        )
        == 0.0
    )
    assert math.isinf(
        capped_redundant_information_upper_bound(
            0.1,
            max_derived_support=0,
            core_dimensions=1,
            q=4,
        )
    )


def test_redundant_core_entropy_and_capped_information_bounds() -> None:
    environment = CappedRedundantRulebook(5, core_dimensions=3, max_derived_support=8)
    entropy = 3 * math.log(4)
    assert environment.core_entropy == entropy
    assert (
        capped_redundant_information_upper_bound(
            0.0, max_derived_support=8, core_dimensions=3, q=4
        )
        == 0.0
    )
    assert (
        capped_redundant_information_upper_bound(
            8.0, max_derived_support=8, core_dimensions=3, q=4
        )
        == entropy
    )
    assert math.isinf(
        capped_redundant_information_upper_bound(
            8.1, max_derived_support=8, core_dimensions=3, q=4
        )
    )


def test_rare_burst_witness_preserves_reward_as_information_vanishes() -> None:
    witnesses = rare_burst_sequence(
        10.0,
        [10, 100, 1000, 10_000],
        core_dimensions=4,
        q=4,
    )
    assert all(witness.expected_reward == 10.0 for witness in witnesses)
    assert all(witness.expected_support == 10.0 for witness in witnesses)
    information = [witness.information_upper_bound for witness in witnesses]
    assert all(bound <= 4 * math.log(4) for bound in information)
    assert all(left > right for left, right in pairwise(information))
    assert information[-1] == pytest.approx(4 * math.log(4) / 1000)
    assert unrestricted_redundant_bit_equivalent(10.0) == 0.0
    assert unrestricted_redundant_bit_equivalent(-1.0) == 0.0
    assert math.isinf(unrestricted_redundant_bit_equivalent(math.inf))


def test_rare_burst_rejects_infeasible_support() -> None:
    with pytest.raises(ValueError, match="too small"):
        RareBurstWitness(
            target_reward=3.0,
            support=2,
            core_entropy=math.log(4),
        )


def test_mix_is_order_invariant_and_even_rules_repeat() -> None:
    environment = MixedRulebook("mix", core_dimensions=2)
    clone = MixedRulebook("mix", core_dimensions=2)
    indices = (1, 2, 3, 4, 10**5, 10**6)
    assert environment.labels(indices) == clone.labels(reversed(indices))[::-1]
    period_in_even_indices = 2 * environment.core_dimensions * environment.reward_spec.q
    for index in range(2, 20, 2):
        assert environment.label(index) == environment.label(
            index + period_in_even_indices
        )


def test_mix_caps_only_redundant_support() -> None:
    environment = MixedRulebook(17, core_dimensions=2, max_redundant_support=2)
    allowed_indices = (1, 2, 3, 4, 5, 7, 9, 11)
    assert environment.evaluate(
        DeploymentAction((index, environment.label(index)) for index in allowed_indices)
    ) == float(len(allowed_indices))
    with pytest.raises(ValueError, match="max_redundant_support"):
        environment.evaluate(correct_action(environment, range(2, 8, 2)))


def test_mix_independent_reward_growth_is_unbounded() -> None:
    environment = MixedRulebook(81, core_dimensions=2, max_redundant_support=1)
    rewards = []
    for count in (1, 10, 100):
        odd_indices = range(1, 2 * count, 2)
        rewards.append(environment.evaluate(correct_action(environment, odd_indices)))
    assert rewards == [1.0, 10.0, 100.0]
    assert rewards[-1] > environment.maximum_redundant_reward
    assert environment.redundant_core_entropy == 2 * math.log(4)


def test_redundant_projection_has_exact_state_and_action_counts() -> None:
    capped = enumerate_redundant_rulebook(
        core_dimensions=2,
        derived_rules=3,
        max_derived_support=2,
    )
    uncapped = enumerate_redundant_rulebook(
        core_dimensions=2,
        derived_rules=3,
        max_derived_support=None,
    )
    expected_capped_actions = 1 + math.comb(3, 1) * 4 + math.comb(3, 2) * 4**2
    assert capped.problem.state_count == 4**2
    assert capped.problem.action_count == expected_capped_actions
    assert uncapped.problem.action_count == 5**3
    assert len(set(capped.states)) == 4**2
    assert all(probability == 1 / 4**2 for probability in capped.problem.prior)


def test_redundant_projection_cap_controls_maximum_reward() -> None:
    capped = enumerate_redundant_rulebook(2, 5, 2)
    uncapped = enumerate_redundant_rulebook(2, 5, None)
    assert capped.problem.maximum_reward == 2.0
    assert uncapped.problem.maximum_reward == 5.0
    assert max(len(action) for action in capped.actions) == 2
    core_revealing = enumerate_redundant_rulebook(2, 2, 2)
    assert core_revealing.problem.maximizing_channel().mutual_information == (
        pytest.approx(2 * math.log(4))
    )


def test_one_rule_redundant_projection_reduces_to_independent_semantics() -> None:
    redundant = enumerate_redundant_rulebook(1, 1, 1)
    independent = enumerate_independent_rulebook(1)
    assert redundant.states == independent.states
    assert redundant.actions == independent.actions
    assert redundant.problem == independent.problem


def test_mixed_projection_has_exact_counts_and_separate_cap() -> None:
    mixed = enumerate_mixed_rulebook(
        primitive_dimensions=2,
        core_dimensions=1,
        derived_rules=3,
        max_redundant_support=1,
    )
    redundant_actions = 1 + math.comb(3, 1) * 4
    assert mixed.problem.state_count == 4 ** (2 + 1)
    assert mixed.problem.action_count == 5**2 * redundant_actions
    assert mixed.problem.maximum_reward == 3.0
    assert len(set(mixed.states)) == mixed.problem.state_count
    assert all(
        sum(index % 2 == 0 for index, _ in action) <= 1 for action in mixed.actions
    )
    assert (
        max(sum(index % 2 == 1 for index, _ in action) for action in mixed.actions) == 2
    )


def test_finite_projection_size_guard_runs_before_allocation() -> None:
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_redundant_rulebook(5, 20, None, max_matrix_entries=100)
    with pytest.raises(ValueError, match="max_matrix_entries"):
        enumerate_mixed_rulebook(5, 2, 10, 5, max_matrix_entries=100)


def test_redundant_projection_is_solver_ready() -> None:
    enumerated = enumerate_redundant_rulebook(1, 1, 1)
    target = 0.25
    solution = solve_frontier(
        enumerated.problem,
        target,
        tolerance=1e-6,
        lagrangian_tolerance=1e-9,
    )
    expected = OneCoordinateFrontier().bit_equivalent(target)
    assert solution.lower_bound <= expected + 1e-5
    assert solution.upper_bound >= expected - 1e-5
    assert solution.witness is not None
    assert solution.witness.expected_reward >= target - 1e-8


def test_redundant_projection_frontier_respects_cap_and_core_entropy() -> None:
    enumerated = enumerate_redundant_rulebook(1, 3, 1)
    endpoint = solve_frontier(enumerated.problem, 1.0, tolerance=1e-8)
    infeasible = solve_frontier(enumerated.problem, 1.01)

    assert endpoint.converged
    assert endpoint.upper_bound <= math.log(4) + 1e-8
    assert math.isinf(infeasible.lower_bound)
    assert math.isinf(infeasible.upper_bound)


def test_mixed_projection_reduces_to_registered_edge_cases() -> None:
    redundant = enumerate_redundant_rulebook(1, 1, 1)
    mixed_redundant = enumerate_mixed_rulebook(0, 1, 1, 1)
    assert mixed_redundant.problem == redundant.problem

    mixed_independent = enumerate_mixed_rulebook(1, 1, 0, 0)
    target = 0.25
    solution = solve_frontier(mixed_independent.problem, target, tolerance=1e-8)
    expected = OneCoordinateFrontier().bit_equivalent(target)
    assert solution.converged
    assert solution.lower_bound <= expected + 1e-8
    assert solution.upper_bound >= expected - 1e-8
