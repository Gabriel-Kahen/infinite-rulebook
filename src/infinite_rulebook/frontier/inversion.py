"""Certified finite frontier targeting and information-budget inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from infinite_rulebook.frontier.blahut_arimoto import (
    LagrangianSolution,
    solve_lagrangian,
    solve_supported_minimum_information,
)
from infinite_rulebook.frontier.finite_problem import (
    ChannelWitness,
    FiniteDecisionProblem,
)


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _extended_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


@dataclass(frozen=True, slots=True)
class FrontierSolution:
    """Bounds and a feasible channel for a reward threshold."""

    target_reward: float
    witness: ChannelWitness | None
    lower_bound: float
    upper_bound: float
    duality_gap: float
    dual_beta: float
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class FrontierInversion:
    """Certified reward bounds for a mutual-information budget."""

    information_budget: float
    witness: ChannelWitness
    reward_lower_bound: float
    reward_upper_bound: float
    iterations: int
    converged: bool


def _mixed_witness(
    problem: FiniteDecisionProblem,
    below: ChannelWitness,
    above: ChannelWitness,
    target: float,
) -> ChannelWitness:
    span = above.expected_reward - below.expected_reward
    if span <= 0.0:
        return above
    weight = min(1.0, max(0.0, (target - below.expected_reward) / span))
    for _ in range(4):
        channel = tuple(
            tuple(
                (1.0 - weight) * below.channel[state][action]
                + weight * above.channel[state][action]
                for action in range(problem.action_count)
            )
            for state in range(problem.state_count)
        )
        witness = problem.evaluate(channel)
        if witness.expected_reward >= target or weight >= 1.0:
            return witness
        weight = math.nextafter(weight, 1.0)
    return above


def _frontier_lower_bound(
    target: float,
    solution: LagrangianSolution,
) -> float:
    return solution.beta * target + solution.objective_lower_bound


def _bounds_converged(lower: float, upper: float, tolerance: float) -> bool:
    difference = upper - lower
    scale = max(1.0, abs(lower), abs(upper))
    roundoff = 64.0 * math.ulp(scale)
    return -roundoff <= difference <= tolerance


def solve_frontier(
    problem: FiniteDecisionProblem,
    target_reward: Real,
    *,
    tolerance: Real = 1e-9,
    bound_tolerance: Real | None = None,
    lagrangian_tolerance: Real = 1e-12,
    max_iterations: int = 96,
    lagrangian_max_iterations: int = 100_000,
) -> FrontierSolution:
    """Bound ``min I(Theta; A)`` subject to ``E[R] >= target_reward``."""

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    target = _extended_real("target_reward", target_reward)
    search_accuracy = _finite_real("tolerance", tolerance)
    if search_accuracy <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    accuracy = (
        search_accuracy
        if bound_tolerance is None
        else _finite_real("bound_tolerance", bound_tolerance)
    )
    if accuracy <= 0.0:
        raise ValueError("bound_tolerance must be strictly positive")
    lagrangian_accuracy = _finite_real("lagrangian_tolerance", lagrangian_tolerance)
    if lagrangian_accuracy <= 0.0:
        raise ValueError("lagrangian_tolerance must be strictly positive")
    lagrangian_accuracy = min(lagrangian_accuracy, search_accuracy)
    for name, value in (
        ("max_iterations", max_iterations),
        ("lagrangian_max_iterations", lagrangian_max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")

    constant = problem.constant_channel()
    if target <= constant.expected_reward:
        return FrontierSolution(target, constant, 0.0, 0.0, 0.0, 0.0, 0, True)
    if target > problem.maximum_reward:
        return FrontierSolution(
            target, None, math.inf, math.inf, 0.0, math.inf, 0, True
        )
    if target == problem.maximum_reward:
        maximizing_supports = tuple(
            tuple(action for action, reward in enumerate(row) if reward == max(row))
            for row in problem.rewards
        )
        endpoint = solve_supported_minimum_information(
            problem,
            maximizing_supports,
            tolerance=lagrangian_accuracy,
            max_iterations=lagrangian_max_iterations,
        )
        lower = max(0.0, endpoint.lower_bound)
        upper = endpoint.witness.mutual_information
        scale = max(1.0, abs(lower), abs(upper))
        roundoff = 64.0 * math.ulp(scale)
        if lower > upper and lower - upper <= roundoff:
            lower = upper
        gap = upper - lower if upper >= lower else math.inf
        return FrontierSolution(
            target_reward=target,
            witness=endpoint.witness,
            lower_bound=lower,
            upper_bound=upper,
            duality_gap=gap,
            dual_beta=math.inf,
            iterations=endpoint.iterations,
            converged=endpoint.converged
            and math.isfinite(gap)
            and _bounds_converged(lower, upper, accuracy),
        )

    maximizing = problem.maximizing_channel()
    lower_solution = solve_lagrangian(problem, 0.0)
    below = lower_solution.witness
    above = maximizing
    beta_low = 0.0
    beta_high = 1.0
    best_lower = 0.0
    best_beta = 0.0
    total_iterations = 0
    search_blocked = False

    high_solution = solve_lagrangian(
        problem,
        beta_high,
        tolerance=lagrangian_accuracy,
        max_iterations=lagrangian_max_iterations,
    )
    total_iterations += high_solution.iterations
    candidate = _frontier_lower_bound(target, high_solution)
    if candidate > best_lower:
        best_lower, best_beta = candidate, beta_high
    if not high_solution.converged:
        search_blocked = True
    while high_solution.witness.expected_reward < target:
        if not high_solution.converged:
            search_blocked = True
            break
        below = high_solution.witness
        beta_low = beta_high
        beta_high *= 2.0
        if not math.isfinite(beta_high) or beta_high > 1e12:
            search_blocked = True
            break
        high_solution = solve_lagrangian(
            problem,
            beta_high,
            tolerance=lagrangian_accuracy,
            max_iterations=lagrangian_max_iterations,
            initial_action_marginal=high_solution.witness.action_marginal,
        )
        total_iterations += high_solution.iterations
        candidate = _frontier_lower_bound(target, high_solution)
        if candidate > best_lower:
            best_lower, best_beta = candidate, beta_high

    if high_solution.converged and high_solution.witness.expected_reward >= target:
        above = high_solution.witness

    witness = _mixed_witness(problem, below, above, target)
    upper_bound = witness.mutual_information
    converged = not search_blocked and _bounds_converged(
        best_lower, upper_bound, accuracy
    )
    for _ in range(max_iterations):
        if converged or search_blocked:
            break
        beta = (beta_low + beta_high) / 2.0
        solution = solve_lagrangian(
            problem,
            beta,
            tolerance=lagrangian_accuracy,
            max_iterations=lagrangian_max_iterations,
        )
        total_iterations += solution.iterations
        candidate = _frontier_lower_bound(target, solution)
        if candidate > best_lower:
            best_lower, best_beta = candidate, beta
        if not solution.converged:
            search_blocked = True
            break
        if solution.witness.expected_reward < target:
            beta_low = beta
            below = solution.witness
        else:
            beta_high = beta
            above = solution.witness
        candidate_witness = _mixed_witness(problem, below, above, target)
        if candidate_witness.mutual_information < upper_bound:
            witness = candidate_witness
            upper_bound = witness.mutual_information
        converged = _bounds_converged(best_lower, upper_bound, accuracy)

    scale = max(1.0, abs(best_lower), abs(upper_bound))
    roundoff = 64.0 * math.ulp(scale)
    if best_lower > upper_bound:
        if best_lower - upper_bound <= roundoff:
            best_lower = upper_bound
        else:
            converged = False
    gap = upper_bound - best_lower if upper_bound >= best_lower else math.inf
    converged = (
        converged
        and not search_blocked
        and _bounds_converged(best_lower, upper_bound, accuracy)
    )
    return FrontierSolution(
        target_reward=target,
        witness=witness,
        lower_bound=best_lower,
        upper_bound=upper_bound,
        duality_gap=gap,
        dual_beta=best_beta,
        iterations=total_iterations,
        converged=converged,
    )


def invert_frontier(
    problem: FiniteDecisionProblem,
    information_budget: Real,
    *,
    tolerance: Real = 1e-7,
    frontier_tolerance: Real = 1e-9,
    max_iterations: int = 64,
) -> FrontierInversion:
    """Bound the largest reward attainable within an information budget."""

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    budget = _extended_real("information_budget", information_budget)
    if budget < 0.0:
        raise ValueError("information_budget must be nonnegative")
    accuracy = _finite_real("tolerance", tolerance)
    if accuracy <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    frontier_accuracy = _finite_real("frontier_tolerance", frontier_tolerance)
    if frontier_accuracy <= 0.0:
        raise ValueError("frontier_tolerance must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    constant = problem.constant_channel()
    maximizing = problem.maximizing_channel()
    if budget >= maximizing.mutual_information:
        return FrontierInversion(
            budget,
            maximizing,
            problem.maximum_reward,
            problem.maximum_reward,
            0,
            True,
        )

    reward_low = constant.expected_reward
    reward_high = problem.maximum_reward
    witness = constant
    iterations = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        if reward_high - reward_low <= accuracy:
            converged = True
            break
        target = (reward_low + reward_high) / 2.0
        solution = solve_frontier(problem, target, tolerance=frontier_accuracy)
        if solution.upper_bound <= budget:
            reward_low = target
            if solution.witness is not None:
                witness = solution.witness
        elif solution.lower_bound > budget:
            reward_high = target
        else:
            # Neither direction is certified at the requested tolerance.
            break

    return FrontierInversion(
        information_budget=budget,
        witness=witness,
        reward_lower_bound=reward_low,
        reward_upper_bound=reward_high,
        iterations=iterations,
        converged=converged,
    )
