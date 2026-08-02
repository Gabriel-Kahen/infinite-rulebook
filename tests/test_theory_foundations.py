"""Executable sanity checks for the theory draft; these are not proofs.

The finite checks exercise the repository's certified frontier solver.  The
unbounded constructions are represented by finite truncations whose reward
and information have closed forms.
"""

from __future__ import annotations

import math
from itertools import pairwise, product

import pytest

from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.inversion import FrontierSolution, solve_frontier


def _solve(problem: FiniteDecisionProblem, target: float) -> FrontierSolution:
    result = solve_frontier(
        problem,
        target,
        tolerance=2e-8,
        bound_tolerance=2e-8,
    )
    assert result.converged
    assert result.lower_bound <= result.upper_bound + 1e-12
    return result


def _binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log(probability) - (1.0 - probability) * math.log(
        1.0 - probability
    )


def _binary_classification_problem(weight: float = 1.0) -> FiniteDecisionProblem:
    return FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((weight, 0.0), (0.0, weight)),
    )


def test_positive_affine_frontier_conjugacy_sanity_check() -> None:
    """Numerically check affine conjugacy on one finite nonbinary problem."""

    problem = FiniteDecisionProblem(
        prior=(0.2, 0.5, 0.3),
        rewards=((1.5, -0.5, 0.2), (-0.2, 1.1, 0.1), (0.0, 0.3, 1.8)),
    )
    scale = 2.75
    shift = -1.2
    transformed = FiniteDecisionProblem(
        prior=problem.prior,
        rewards=tuple(
            tuple(scale * reward + shift for reward in row) for row in problem.rewards
        ),
    )
    target = 0.95

    original = _solve(problem, target)
    conjugate = _solve(transformed, scale * target + shift)

    assert conjugate.lower_bound == pytest.approx(original.lower_bound, abs=3e-7)
    assert conjugate.upper_bound == pytest.approx(original.upper_bound, abs=3e-7)
    assert conjugate.witness is not None
    assert original.witness is not None
    assert conjugate.witness.mutual_information == pytest.approx(
        original.witness.mutual_information,
        abs=3e-7,
    )


def test_reward_sufficient_source_reduction_sanity_check() -> None:
    """Numerically check reduction after duplicating each payoff state."""

    expanded = FiniteDecisionProblem(
        prior=(0.35, 0.15, 0.10, 0.40),
        rewards=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
    )
    reduced = _binary_classification_problem()

    expanded_frontier = _solve(expanded, 0.82)
    reduced_frontier = _solve(reduced, 0.82)

    assert expanded_frontier.lower_bound == pytest.approx(
        reduced_frontier.lower_bound,
        abs=3e-7,
    )
    assert expanded_frontier.upper_bound == pytest.approx(
        reduced_frontier.upper_bound,
        abs=3e-7,
    )


def test_behavioral_action_quotient_sanity_check() -> None:
    """Numerically check equality when every quotient action has a raw lift."""

    raw = FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)),
    )
    quotient = _binary_classification_problem()

    raw_frontier = _solve(raw, 0.81)
    quotient_frontier = _solve(quotient, 0.81)

    assert raw_frontier.lower_bound == pytest.approx(
        quotient_frontier.lower_bound,
        abs=3e-7,
    )
    assert raw_frontier.upper_bound == pytest.approx(
        quotient_frontier.upper_bound,
        abs=3e-7,
    )


def test_quotient_equality_can_fail_without_lift_sanity_check() -> None:
    """Check strict failure when the enlarged quotient has unliftable actions."""

    raw = FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((0.0,), (0.0,)),
    )
    enlarged_quotient = FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    raw_frontier = solve_frontier(raw, 0.75)
    quotient_frontier = _solve(enlarged_quotient, 0.75)

    assert math.isinf(raw_frontier.lower_bound)
    assert math.isfinite(quotient_frontier.upper_bound)
    assert quotient_frontier.upper_bound < math.log(2.0)


def test_finite_product_infimal_convolution_sanity_check() -> None:
    """Check a nonidentical two-factor product at a known interior optimum."""

    weights = (1.0, 2.0)
    states = tuple(product((0, 1), repeat=2))
    actions = tuple(product((0, 1), repeat=2))
    product_problem = FiniteDecisionProblem(
        prior=(0.25,) * len(states),
        rewards=tuple(
            tuple(
                math.fsum(
                    weights[index] * (state[index] == action[index])
                    for index in range(2)
                )
                for action in actions
            )
            for state in states
        ),
    )

    # f'(p) = log(p / (1-p)) for f(p) = log(2) - h(p).  These
    # probabilities obey f'(p_2) = 2 f'(p_1), the allocation's first-order
    # condition for reward weights one and two.
    correct_probabilities = (4.0 / 5.0, 16.0 / 17.0)
    target = correct_probabilities[0] + 2.0 * correct_probabilities[1]
    expected = math.fsum(
        math.log(2.0) - _binary_entropy(probability)
        for probability in correct_probabilities
    )

    product_frontier = _solve(product_problem, target)
    component_frontiers = (
        _solve(_binary_classification_problem(1.0), correct_probabilities[0]),
        _solve(
            _binary_classification_problem(2.0),
            2.0 * correct_probabilities[1],
        ),
    )

    assert math.fsum(result.upper_bound for result in component_frontiers) == (
        pytest.approx(expected, abs=5e-7)
    )
    assert product_frontier.lower_bound == pytest.approx(expected, abs=5e-7)
    assert product_frontier.upper_bound == pytest.approx(expected, abs=5e-7)


def test_cubic_transformation_collapse_witness_sanity_check() -> None:
    """Check finite-prefix witnesses for the cubic classification reversal."""

    target = 1.0
    information_costs = []
    for prefix_length in (1, 2, 4, 7):
        state_count = 1 << prefix_length
        activation_probability = target / prefix_length**3
        channel = tuple(
            tuple(
                1.0 - activation_probability
                if action == 0
                else activation_probability
                if action == state + 1
                else 0.0
                for action in range(state_count + 1)
            )
            for state in range(state_count)
        )
        raw = FiniteDecisionProblem(
            prior=(1.0 / state_count,) * state_count,
            rewards=tuple(
                tuple(
                    float(prefix_length) if action == state + 1 else 0.0
                    for action in range(state_count + 1)
                )
                for state in range(state_count)
            ),
        )
        cubed = FiniteDecisionProblem(
            prior=raw.prior,
            rewards=tuple(tuple(reward**3 for reward in row) for row in raw.rewards),
        )

        raw_witness = raw.evaluate(channel)
        cubed_witness = cubed.evaluate(channel)
        expected_information = target * math.log(2.0) / prefix_length**2

        assert cubed_witness.expected_reward == pytest.approx(target)
        assert raw_witness.expected_reward == pytest.approx(target / prefix_length**2)
        assert cubed_witness.mutual_information == pytest.approx(expected_information)
        assert raw_witness.mutual_information == pytest.approx(expected_information)
        information_costs.append(cubed_witness.mutual_information)

    assert all(later < earlier for earlier, later in pairwise(information_costs))
    assert information_costs[-1] < 0.015


def test_rare_burst_collapse_witness_sanity_check() -> None:
    """Check strict-margin rare bursts with fixed reward and vanishing bits."""

    state_count = 3
    target = 0.75
    information_costs = []
    burst_ratios = []
    for magnitude in (1.0, 2.0, 4.0, 8.0, 16.0):
        activation_probability = target / magnitude
        problem = FiniteDecisionProblem(
            prior=(1.0 / state_count,) * state_count,
            rewards=tuple(
                tuple(
                    0.0
                    if action == 0
                    else magnitude
                    if action == state + 1
                    else -magnitude
                    for action in range(state_count + 1)
                )
                for state in range(state_count)
            ),
        )
        channel = tuple(
            tuple(
                1.0 - activation_probability
                if action == 0
                else activation_probability
                if action == state + 1
                else 0.0
                for action in range(state_count + 1)
            )
            for state in range(state_count)
        )

        witness = problem.evaluate(channel)
        expected_information = target * math.log(state_count) / magnitude

        assert problem.zero_information_reward == pytest.approx(0.0)
        assert witness.expected_reward == pytest.approx(target)
        assert witness.mutual_information == pytest.approx(expected_information)
        information_costs.append(witness.mutual_information)
        burst_ratios.append(math.log(state_count) / magnitude)

    assert all(later < earlier for earlier, later in pairwise(information_costs))
    assert burst_ratios[-1] < burst_ratios[0] / 10.0


def test_bounded_reward_pinsker_certificate_sanity_check() -> None:
    """Check the Pinsker lower bound against a certified finite frontier."""

    problem = _binary_classification_problem()
    target = 0.75
    reward_range = max(map(max, problem.rewards)) - min(map(min, problem.rewards))
    pinsker_lower_bound = (
        2.0 * ((target - problem.zero_information_reward) / reward_range) ** 2
    )
    analytic_frontier = math.log(2.0) - _binary_entropy(target)

    result = _solve(problem, target)

    assert result.witness is not None
    assert result.witness.expected_reward >= target
    assert result.lower_bound <= analytic_frontier <= result.upper_bound + 1e-12
    # Because this certified numerical lower endpoint already exceeds the
    # Pinsker expression, the whole certified frontier interval does too.
    assert result.lower_bound >= pinsker_lower_bound
    assert analytic_frontier > pinsker_lower_bound


def test_positive_local_slope_tensorization_sanity_check() -> None:
    """Check finite-n convergence to an explicit positive local price."""

    tangent_probability = 4.0 / 5.0
    local_price = math.log(8.0 / 5.0)
    mismatch_penalty = math.log(5.0 / 2.0) / local_price
    problem = FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((0.0, 1.0, -mismatch_penalty), (0.0, -mismatch_penalty, 1.0)),
    )
    tangent_reward = (1.0 + mismatch_penalty) * tangent_probability - mismatch_penalty

    def analytic_component_frontier(target: float) -> float:
        if target <= tangent_reward:
            return local_price * target
        correct_probability = (target + mismatch_penalty) / (1.0 + mismatch_penalty)
        return math.log(2.0) - _binary_entropy(correct_probability)

    assert problem.zero_information_reward == pytest.approx(0.0)
    assert tangent_reward == pytest.approx(
        (math.log(2.0) - _binary_entropy(tangent_probability)) / local_price
    )

    total_reward = 2.0
    component_counts = (2, 3, 4, 5, 8, 16, 32)
    tensorized_values = []
    for component_count in component_counts:
        component_target = total_reward / component_count
        expected_component = analytic_component_frontier(component_target)
        component_result = _solve(problem, component_target)

        assert component_result.lower_bound == pytest.approx(
            expected_component,
            abs=3e-7,
        )
        assert component_result.upper_bound == pytest.approx(
            expected_component,
            abs=3e-7,
        )
        tensorized_values.append(component_count * expected_component)

    countable_limit = local_price * total_reward
    assert tensorized_values[0] > countable_limit
    assert all(
        later <= earlier + 1e-12 for earlier, later in pairwise(tensorized_values)
    )
    assert tensorized_values[-1] == pytest.approx(countable_limit, abs=1e-12)


def test_bounded_noncompact_boundary_nonattainment_sanity_check() -> None:
    """Check finite truncations of a bounded, unattained baseline boundary."""

    baselines = []
    information_costs = []
    for cutoff in (2, 4, 8, 16, 32, 64):
        states = (-1, 1)
        actions = tuple(
            (index, sign) for index in range(2, cutoff + 1) for sign in states
        )
        problem = FiniteDecisionProblem(
            prior=(0.5, 0.5),
            rewards=tuple(
                tuple(1.0 - 1.0 / index + 0.5 * state * sign for index, sign in actions)
                for state in states
            ),
        )
        correct_probability = 0.5 + 1.0 / cutoff
        selected_actions = {sign: actions.index((cutoff, sign)) for sign in states}
        channel = tuple(
            tuple(
                correct_probability
                if action == selected_actions[state]
                else 1.0 - correct_probability
                if action == selected_actions[-state]
                else 0.0
                for action in range(len(actions))
            )
            for state in states
        )

        witness = problem.evaluate(channel)
        expected_information = math.log(2.0) - _binary_entropy(correct_probability)

        assert problem.zero_information_reward == pytest.approx(1.0 - 1.0 / cutoff)
        assert problem.zero_information_reward < 1.0
        assert witness.expected_reward == pytest.approx(1.0)
        assert witness.mutual_information == pytest.approx(expected_information)
        baselines.append(problem.zero_information_reward)
        information_costs.append(witness.mutual_information)

    assert all(later > earlier for earlier, later in pairwise(baselines))
    assert baselines[-1] > 0.98
    assert all(later < earlier for earlier, later in pairwise(information_costs))
    assert information_costs[-1] < 0.0005


def test_dependent_source_breaks_countable_composition_sanity_check() -> None:
    """Check shared-bit bursts despite a positive component DV slope."""

    mismatch_penalty = 3.0
    dv_slope = 0.5
    fixed_guess_exponential_moment = (
        math.exp(dv_slope) + math.exp(-dv_slope * mismatch_penalty)
    ) / 2.0
    component = FiniteDecisionProblem(
        prior=(0.5, 0.5),
        rewards=((0.0, 1.0, -mismatch_penalty), (0.0, -mismatch_penalty, 1.0)),
    )

    # The null action has exponential moment one and both guesses have the
    # displayed smaller moment.  Donsker--Varadhan therefore certifies
    # B_1(rho) >= dv_slope * rho for every positive feasible rho.
    assert component.zero_information_reward == pytest.approx(0.0)
    assert fixed_guess_exponential_moment < 1.0

    target = 1.0
    information_costs = []
    for copy_count in (1, 2, 4, 8, 16, 32, 64):
        activation_probability = target / copy_count
        shared_bit_problem = FiniteDecisionProblem(
            prior=(0.5, 0.5),
            rewards=(
                (0.0, float(copy_count), -mismatch_penalty * copy_count),
                (0.0, -mismatch_penalty * copy_count, float(copy_count)),
            ),
        )
        channel = (
            (1.0 - activation_probability, activation_probability, 0.0),
            (1.0 - activation_probability, 0.0, activation_probability),
        )

        witness = shared_bit_problem.evaluate(channel)
        expected_support = activation_probability * copy_count
        expected_information = activation_probability * math.log(2.0)

        assert witness.expected_reward == pytest.approx(target)
        assert expected_support == pytest.approx(target)
        assert witness.mutual_information == pytest.approx(expected_information)
        assert dv_slope * target > 0.0
        information_costs.append(witness.mutual_information)

    assert all(later < earlier for earlier, later in pairwise(information_costs))
    assert information_costs[-1] < 0.011
