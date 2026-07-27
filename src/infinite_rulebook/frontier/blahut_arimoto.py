"""Blahut--Arimoto minimization for finite reward-information problems."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from infinite_rulebook.frontier.finite_problem import (
    ChannelWitness,
    FiniteDecisionProblem,
)


def _nonnegative_finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True, slots=True)
class LagrangianSolution:
    """A feasible BA witness plus certified objective bounds."""

    beta: float
    witness: ChannelWitness
    objective: float
    objective_lower_bound: float
    objective_upper_bound: float
    duality_gap: float
    fixed_point_residual: float
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class SupportedInformationSolution:
    """A minimum-information channel constrained to row-wise action supports."""

    witness: ChannelWitness
    lower_bound: float
    upper_bound: float
    duality_gap: float
    fixed_point_residual: float
    iterations: int
    converged: bool


def _normalized_marginal(
    problem: FiniteDecisionProblem,
    marginal: Sequence[Real] | None,
) -> tuple[float, ...]:
    if marginal is None:
        return tuple(1.0 / problem.action_count for _ in range(problem.action_count))
    if len(marginal) != problem.action_count:
        raise ValueError("initial_action_marginal has the wrong length")
    values = tuple(
        _nonnegative_finite(f"initial_action_marginal[{index}]", value)
        for index, value in enumerate(marginal)
    )
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("initial_action_marginal must have positive mass")
    # A tiny floor lets BA discover an action absent from a warm start.
    floor = 1e-300
    floored = tuple(max(floor, value / total) for value in values)
    normalization = math.fsum(floored)
    return tuple(value / normalization for value in floored)


def _channel_from_marginal(
    problem: FiniteDecisionProblem,
    beta: float,
    marginal: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    channel = []
    log_normalizers = []
    log_marginal = tuple(
        math.log(value) if value > 0.0 else -math.inf for value in marginal
    )
    for rewards in problem.rewards:
        scaled_rewards = tuple(beta * reward for reward in rewards)
        if any(not math.isfinite(value) for value in scaled_rewards):
            raise ValueError("beta times reward exceeds finite float range")
        logits = tuple(
            log_marginal[action] + scaled_rewards[action]
            for action in range(problem.action_count)
        )
        log_normalizer = _logsumexp(logits)
        log_normalizers.append(log_normalizer)
        channel.append(tuple(math.exp(value - log_normalizer) for value in logits))
    return tuple(channel), tuple(log_normalizers)


def _marginal_objective_bounds(
    problem: FiniteDecisionProblem,
    beta: float,
    marginal: tuple[float, ...],
    log_normalizers: tuple[float, ...],
) -> tuple[float, float, tuple[float, ...]]:
    """Bound the marginal objective using log multiplicative slacks."""

    value = -math.fsum(
        probability * log_normalizer
        for probability, log_normalizer in zip(
            problem.prior, log_normalizers, strict=True
        )
    )
    action_log_slacks = []
    for action in range(problem.action_count):
        terms = []
        for state in range(problem.state_count):
            probability = problem.prior[state]
            if probability > 0.0:
                terms.append(
                    math.log(probability)
                    + beta * problem.rewards[state][action]
                    - log_normalizers[state]
                )
        action_log_slacks.append(_logsumexp(terms))
    # This scaled-dual certificate includes every action, including those
    # outside a temporarily restricted active support.
    certificate = value - max(action_log_slacks)
    return certificate, value, tuple(action_log_slacks)


def _certificate_marginal(
    problem: FiniteDecisionProblem,
    marginal: Sequence[Real],
) -> tuple[float, ...]:
    if len(marginal) != problem.action_count:
        raise ValueError("certificate marginal has the wrong action count")
    values = tuple(
        _nonnegative_finite(f"marginal[{index}]", value)
        for index, value in enumerate(marginal)
    )
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("certificate marginal must sum to one")
    return tuple(value / total for value in values)


def lagrangian_certificate_lower_bound(
    problem: FiniteDecisionProblem,
    beta: Real,
    action_marginal: Sequence[Real],
) -> float:
    """Re-evaluate a Lagrangian lower certificate from its sufficient data."""

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    multiplier = _nonnegative_finite("beta", beta)
    marginal = _certificate_marginal(problem, action_marginal)
    _, log_normalizers = _channel_from_marginal(problem, multiplier, marginal)
    lower, _, _ = _marginal_objective_bounds(
        problem,
        multiplier,
        marginal,
        log_normalizers,
    )
    return lower


def supported_certificate_lower_bound(
    problem: FiniteDecisionProblem,
    allowed_actions: Sequence[Sequence[int]],
    action_marginal: Sequence[Real],
) -> float:
    """Re-evaluate a supported-endpoint certificate from its sufficient data."""

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    if len(allowed_actions) != problem.state_count:
        raise ValueError("allowed_actions must have one support per state")
    supports = []
    for state, support in enumerate(allowed_actions):
        canonical = frozenset(support)
        if not canonical:
            raise ValueError(f"allowed_actions[{state}] must not be empty")
        if any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < problem.action_count
            for action in canonical
        ):
            raise ValueError(f"allowed_actions[{state}] contains an invalid action")
        supports.append(canonical)
    marginal = _certificate_marginal(problem, action_marginal)
    support_mass = tuple(
        math.fsum(marginal[action] for action in support) for support in supports
    )
    if any(
        probability > 0.0 and support_mass[state] <= 0.0
        for state, probability in enumerate(problem.prior)
    ):
        raise ValueError("certificate marginal gives zero mass to a required support")
    marginal_objective = -math.fsum(
        probability * math.log(support_mass[state])
        for state, probability in enumerate(problem.prior)
        if probability > 0.0
    )
    slacks = tuple(
        math.fsum(
            probability / support_mass[state]
            for state, (probability, support) in enumerate(
                zip(problem.prior, supports, strict=True)
            )
            if probability > 0.0 and action in support
        )
        for action in range(problem.action_count)
    )
    return marginal_objective - math.log(max(slacks))


def _normalize_on_support(
    marginal: Sequence[float],
    support: set[int],
) -> tuple[float, ...]:
    total = math.fsum(marginal[action] for action in support)
    if total <= 0.0:
        return tuple(
            1.0 / len(support) if action in support else 0.0
            for action in range(len(marginal))
        )
    return tuple(
        marginal[action] / total if action in support else 0.0
        for action in range(len(marginal))
    )


def _certified_gap(lower: float, upper: float) -> float:
    difference = upper - lower
    roundoff = 64.0 * math.ulp(max(1.0, abs(lower), abs(upper)))
    if difference >= 0.0:
        return difference
    return 0.0 if -difference <= roundoff else math.inf


def _certified_face_candidate(
    problem: FiniteDecisionProblem,
    beta: float,
    marginal: tuple[float, ...],
    action_log_slacks: tuple[float, ...],
    tolerance: float,
) -> tuple[float, ...] | None:
    """Project an evident face, accepting it only with a global certificate."""

    threshold = math.log1p(-min(0.5, max(10.0 * tolerance, 1e-14)))
    support = {
        action
        for action, log_slack in enumerate(action_log_slacks)
        if log_slack >= threshold
    }
    if not support or len(support) == problem.action_count:
        return None
    candidate = _normalize_on_support(marginal, support)
    channel, log_normalizers = _channel_from_marginal(problem, beta, candidate)
    witness = problem.evaluate(channel)
    lower, marginal_upper, _ = _marginal_objective_bounds(
        problem, beta, candidate, log_normalizers
    )
    objective = witness.mutual_information - beta * witness.expected_reward
    if _certified_gap(lower, min(objective, marginal_upper)) <= tolerance:
        return witness.action_marginal
    return None


def solve_lagrangian(
    problem: FiniteDecisionProblem,
    beta: Real,
    *,
    tolerance: Real = 1e-12,
    max_iterations: int = 100_000,
    initial_action_marginal: Sequence[Real] | None = None,
) -> LagrangianSolution:
    """Minimize ``I(Theta; A) - beta E[R]``.

    The returned witness is always feasible.  ``objective_lower_bound`` uses
    the scaled-dual ``-E[log Z] - log(max_a s_a)`` certificate over every
    action; ``objective_upper_bound`` is the better of the marginal objective
    and the direct feasible-channel objective.
    """

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    multiplier = _nonnegative_finite("beta", beta)
    accuracy = _nonnegative_finite("tolerance", tolerance)
    if accuracy == 0.0:
        raise ValueError("tolerance must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    if multiplier == 0.0:
        witness = problem.constant_channel()
        return LagrangianSolution(
            beta=0.0,
            witness=witness,
            objective=0.0,
            objective_lower_bound=0.0,
            objective_upper_bound=0.0,
            duality_gap=0.0,
            fixed_point_residual=0.0,
            iterations=0,
            converged=True,
        )

    marginal = _normalized_marginal(problem, initial_action_marginal)
    converged = False
    support = set(range(problem.action_count))
    iterations = 0
    residual = math.inf
    lower_bound = -math.inf
    marginal_upper = math.inf
    channel: tuple[tuple[float, ...], ...]

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        channel, log_normalizers = _channel_from_marginal(problem, multiplier, marginal)
        witness = problem.evaluate(channel)
        new_marginal = witness.action_marginal
        residual = max(
            abs(left - right)
            for left, right in zip(marginal, new_marginal, strict=True)
        )
        lower_bound, marginal_upper, action_log_slacks = _marginal_objective_bounds(
            problem, multiplier, marginal, log_normalizers
        )
        objective = witness.mutual_information - multiplier * witness.expected_reward
        objective_upper = min(objective, marginal_upper)
        gap = _certified_gap(lower_bound, objective_upper)
        restricted_lower = marginal_upper - max(
            action_log_slacks[action] for action in support
        )
        restricted_gap = _certified_gap(restricted_lower, objective_upper)
        residual_limit = math.sqrt(accuracy)

        face_candidate = _certified_face_candidate(
            problem,
            multiplier,
            marginal,
            action_log_slacks,
            accuracy,
        )
        if face_candidate is not None:
            marginal = face_candidate
            break

        if restricted_gap <= accuracy and residual <= residual_limit:
            if gap <= accuracy:
                marginal = new_marginal
                converged = True
                break
            violating = {
                action
                for action in range(problem.action_count)
                if action not in support and action_log_slacks[action] > accuracy
            }
            if violating:
                support.update(violating)
                seed = min(1e-6, 0.01 / len(violating))
                retained = 1.0 - seed * len(violating)
                marginal = tuple(
                    seed if action in violating else retained * new_marginal[action]
                    for action in range(problem.action_count)
                )
                continue

        # Multiplicative BA can spend tens of thousands of iterations driving
        # a provably weak action to zero.  Periodically solve on the suggested
        # face instead; the global certificate above prevents false success,
        # and a violating excluded action is re-entered after face convergence.
        if iteration % 32 == 0 and len(support) > 1:
            drop_threshold = math.log1p(-min(0.5, max(10.0 * accuracy, 1e-14)))
            removable = {
                action
                for action in support
                if new_marginal[action] < 0.05
                and action_log_slacks[action] < drop_threshold
            }
            if removable and len(removable) < len(support):
                support.difference_update(removable)
                marginal = _normalize_on_support(new_marginal, support)
                continue

        marginal = _normalize_on_support(new_marginal, support)

    # Recompute from the final marginal so diagnostics describe the witness.
    channel, log_normalizers = _channel_from_marginal(problem, multiplier, marginal)
    witness = problem.evaluate(channel)
    output = witness.action_marginal
    residual = max(
        abs(left - right) for left, right in zip(marginal, output, strict=True)
    )
    lower_bound, marginal_upper, _ = _marginal_objective_bounds(
        problem, multiplier, marginal, log_normalizers
    )
    objective = witness.mutual_information - multiplier * witness.expected_reward
    objective_upper = min(objective, marginal_upper)
    gap = _certified_gap(lower_bound, objective_upper)
    converged = gap <= accuracy and residual <= math.sqrt(accuracy)

    return LagrangianSolution(
        beta=multiplier,
        witness=witness,
        objective=objective,
        objective_lower_bound=lower_bound,
        objective_upper_bound=objective_upper,
        duality_gap=gap,
        fixed_point_residual=residual,
        iterations=iterations,
        converged=converged,
    )


def solve_supported_minimum_information(
    problem: FiniteDecisionProblem,
    allowed_actions: Sequence[Sequence[int]],
    *,
    tolerance: Real = 1e-12,
    max_iterations: int = 100_000,
) -> SupportedInformationSolution:
    """Minimize mutual information with a declared action support per state.

    For an action marginal ``q``, the row update is ``q`` conditioned on the
    state's allowed set. The global certificate is
    ``-E[log q(S_theta)] - log(max_a s_a)`` with
    ``s_a = sum_{theta: a in S_theta} p_theta / q(S_theta)``.
    """

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    accuracy = _nonnegative_finite("tolerance", tolerance)
    if accuracy == 0.0:
        raise ValueError("tolerance must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if len(allowed_actions) != problem.state_count:
        raise ValueError("allowed_actions must have one support per state")

    supports = []
    for state, support in enumerate(allowed_actions):
        try:
            canonical = frozenset(support)
        except TypeError as error:
            raise TypeError(f"allowed_actions[{state}] must be iterable") from error
        if not canonical:
            raise ValueError(f"allowed_actions[{state}] must not be empty")
        if any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < problem.action_count
            for action in canonical
        ):
            raise ValueError(f"allowed_actions[{state}] contains an invalid action")
        supports.append(canonical)
    canonical_supports = tuple(supports)

    relevant = set().union(
        *(
            support
            for probability, support in zip(
                problem.prior, canonical_supports, strict=True
            )
            if probability > 0.0
        )
    )
    marginal = tuple(
        1.0 / len(relevant) if action in relevant else 0.0
        for action in range(problem.action_count)
    )
    iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        support_mass = tuple(
            math.fsum(marginal[action] for action in support)
            for support in canonical_supports
        )
        channel = tuple(
            tuple(
                (
                    marginal[action] / support_mass[state]
                    if action in canonical_supports[state] and support_mass[state] > 0.0
                    else 0.0
                )
                for action in range(problem.action_count)
            )
            for state in range(problem.state_count)
        )
        # Zero-prior rows do not affect the objective but still need a valid
        # stochastic row.
        channel = tuple(
            row
            if math.fsum(row) > 0.0
            else tuple(
                1.0 / len(canonical_supports[state])
                if action in canonical_supports[state]
                else 0.0
                for action in range(problem.action_count)
            )
            for state, row in enumerate(channel)
        )
        witness = problem.evaluate(channel)
        objective = witness.mutual_information
        residual = max(
            abs(left - right)
            for left, right in zip(marginal, witness.action_marginal, strict=True)
        )
        marginal_objective = -math.fsum(
            probability * math.log(support_mass[state])
            for state, probability in enumerate(problem.prior)
            if probability > 0.0
        )
        slacks = tuple(
            math.fsum(
                probability / support_mass[state]
                for state, (probability, support) in enumerate(
                    zip(problem.prior, canonical_supports, strict=True)
                )
                if probability > 0.0 and action in support
            )
            for action in range(problem.action_count)
        )
        lower = marginal_objective - math.log(max(slacks))
        upper = objective
        gap = _certified_gap(lower, upper)
        if gap <= accuracy and residual <= math.sqrt(accuracy):
            marginal = witness.action_marginal
            converged = True
            break
        marginal = witness.action_marginal

    support_mass = tuple(
        math.fsum(marginal[action] for action in support)
        for support in canonical_supports
    )
    channel = tuple(
        tuple(
            (
                marginal[action] / support_mass[state]
                if action in canonical_supports[state] and support_mass[state] > 0.0
                else 0.0
            )
            for action in range(problem.action_count)
        )
        for state in range(problem.state_count)
    )
    channel = tuple(
        row
        if math.fsum(row) > 0.0
        else tuple(
            1.0 / len(canonical_supports[state])
            if action in canonical_supports[state]
            else 0.0
            for action in range(problem.action_count)
        )
        for state, row in enumerate(channel)
    )
    witness = problem.evaluate(channel)
    residual = max(
        abs(left - right)
        for left, right in zip(marginal, witness.action_marginal, strict=True)
    )
    marginal_objective = -math.fsum(
        probability * math.log(support_mass[state])
        for state, probability in enumerate(problem.prior)
        if probability > 0.0
    )
    slacks = tuple(
        math.fsum(
            probability / support_mass[state]
            for state, (probability, support) in enumerate(
                zip(problem.prior, canonical_supports, strict=True)
            )
            if probability > 0.0 and action in support
        )
        for action in range(problem.action_count)
    )
    lower = marginal_objective - math.log(max(slacks))
    upper = witness.mutual_information
    gap = _certified_gap(lower, upper)
    converged = gap <= accuracy and residual <= math.sqrt(accuracy)
    return SupportedInformationSolution(
        witness=witness,
        lower_bound=lower,
        upper_bound=upper,
        duality_gap=gap,
        fixed_point_residual=residual,
        iterations=iterations,
        converged=converged,
    )
