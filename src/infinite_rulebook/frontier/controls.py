"""Exact frontier identities and finite projections for symbolic controls.

All information quantities are measured in nats.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.environments.controls import (
    PublicBonusSchedule,
    PublicDeploymentAction,
)
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.rulebook_problem import EnumeratedRulebookProblem

BitEquivalent = Callable[[Real], float]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _real_not_nan(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


def _finite_nonnegative(value: Real, name: str) -> float:
    result = _real_not_nan(value, name)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _reward_spec(value: RewardSpec | None) -> RewardSpec:
    if value is None:
        return RewardSpec()
    if not isinstance(value, RewardSpec):
        raise TypeError("reward_spec must be a RewardSpec")
    return value


def _guard_matrix(
    state_count: int,
    action_count: int,
    max_matrix_entries: int,
) -> None:
    limit = _integer(max_matrix_entries, "max_matrix_entries", minimum=1)
    if state_count * action_count > limit:
        raise ValueError(
            "enumerated reward matrix would exceed max_matrix_entries: "
            f"{state_count} * {action_count}"
        )


def alea_frontier_problem(
    base_problem: FiniteDecisionProblem,
) -> FiniteDecisionProblem:
    """Return the unchanged decision problem after adding ALEA observations.

    ALEA is fresh observation noise, not persistent ``Theta``, an action
    coordinate, or a reward input. It is therefore absent from the frontier
    problem by construction.
    """

    if not isinstance(base_problem, FiniteDecisionProblem):
        raise TypeError("base_problem must be a FiniteDecisionProblem")
    return base_problem


def alea_persistent_information_nats() -> float:
    """Return exact ``I(Theta; U_history)`` for independent ALEA noise."""

    return 0.0


def trivia_invariant_bit_equivalent(
    base_bit_equivalent: BitEquivalent,
    target_reward: Real,
) -> float:
    """Apply ``B_TRIVIA(rho) = B_base(rho)`` exactly."""

    if not callable(base_bit_equivalent):
        raise TypeError("base_bit_equivalent must be callable")
    target = _real_not_nan(target_reward, "target_reward")
    return float(base_bit_equivalent(target))


def public_u_bit_equivalent(target_reward: Real) -> float:
    """Return the exact PUBLIC-U frontier for an unbounded public reward."""

    target = _real_not_nan(target_reward, "target_reward")
    if target == math.inf:
        return math.inf
    return 0.0


@dataclass(frozen=True, slots=True)
class PublicUWitness:
    """A finite state-independent PUBLIC-U action attaining one threshold."""

    target_reward: float
    public_choice: int
    expected_reward: float

    @property
    def mutual_information(self) -> float:
        return 0.0


def public_u_witness(
    target_reward: Real,
    *,
    public_unit_reward: Real = 1.0,
) -> PublicUWitness:
    """Construct a finite zero-information witness for a finite threshold."""

    target = _real_not_nan(target_reward, "target_reward")
    unit = _finite_nonnegative(public_unit_reward, "public_unit_reward")
    if unit == 0.0:
        raise ValueError("public_unit_reward must be positive")
    if not math.isfinite(target):
        raise ValueError("target_reward must be finite")
    if target <= 0.0:
        choice = 0
    else:
        ratio = Fraction.from_float(target) / Fraction.from_float(unit)
        choice = -(-ratio.numerator // ratio.denominator)
    try:
        reward = float(choice * Fraction.from_float(unit))
    except OverflowError as error:
        raise ValueError("target_reward requires a non-finite public reward") from error
    if not math.isfinite(reward):
        raise ValueError("target_reward requires a non-finite public reward")
    return PublicUWitness(target, choice, reward)


def public_c_bit_equivalent(
    base_bit_equivalent: BitEquivalent,
    target_reward: Real,
    *,
    maximum_public_reward: Real,
) -> float:
    """Apply ``B_PUBLIC-C(rho) = B_base(rho - G_max)`` exactly."""

    if not callable(base_bit_equivalent):
        raise TypeError("base_bit_equivalent must be callable")
    target = _real_not_nan(target_reward, "target_reward")
    maximum = _finite_nonnegative(
        maximum_public_reward,
        "maximum_public_reward",
    )
    return float(base_bit_equivalent(target - maximum))


@dataclass(frozen=True, slots=True)
class PublicCFrontier:
    """A bounded-public-contribution transform of any exact base frontier."""

    base_bit_equivalent: BitEquivalent
    maximum_public_reward: float

    def __post_init__(self) -> None:
        if not callable(self.base_bit_equivalent):
            raise TypeError("base_bit_equivalent must be callable")
        maximum = _finite_nonnegative(
            self.maximum_public_reward,
            "maximum_public_reward",
        )
        object.__setattr__(self, "maximum_public_reward", maximum)

    def bit_equivalent(self, target_reward: Real) -> float:
        return public_c_bit_equivalent(
            self.base_bit_equivalent,
            target_reward,
            maximum_public_reward=self.maximum_public_reward,
        )


@dataclass(frozen=True, slots=True)
class EnumeratedPublicProblem:
    """A finite PUBLIC-C problem with typed composite actions."""

    problem: FiniteDecisionProblem
    states: tuple[tuple[int, ...], ...]
    actions: tuple[PublicDeploymentAction, ...]


def enumerate_trivia_rulebook(
    reward_dimensions: int,
    trivia_dimensions: int,
    reward_spec: RewardSpec | None = None,
    *,
    max_matrix_entries: int = 2_000_000,
) -> EnumeratedRulebookProblem:
    """Exhaustively project independent useful rules plus static trivia.

    State tuples store useful coordinates first and trivia coordinates second.
    Actions contain only reward-relevant deployment behavior. A solver may
    still choose a distinct action channel for every full ``(Z, D)`` state,
    which is the unrestricted augmented frontier optimization.
    """

    useful = _integer(reward_dimensions, "reward_dimensions", minimum=1)
    trivia = _integer(trivia_dimensions, "trivia_dimensions", minimum=0)
    spec = _reward_spec(reward_spec)
    state_count = spec.q ** (useful + trivia)
    action_count = (spec.q + 1) ** useful
    _guard_matrix(state_count, action_count, max_matrix_entries)

    states = tuple(itertools.product(range(1, spec.q + 1), repeat=useful + trivia))
    action_vectors = tuple(itertools.product(range(0, spec.q + 1), repeat=useful))
    actions = tuple(
        DeploymentAction(
            (index + 1, prediction) for index, prediction in enumerate(vector)
        )
        for vector in action_vectors
    )
    rewards = tuple(
        tuple(
            math.fsum(
                spec.contribution(prediction, truth)
                for prediction, truth in zip(
                    action_vector,
                    state[:useful],
                    strict=True,
                )
            )
            for action_vector in action_vectors
        )
        for state in states
    )
    prior = (1.0 / state_count,) * state_count
    return EnumeratedRulebookProblem(
        problem=FiniteDecisionProblem(prior=prior, rewards=rewards),
        states=states,
        actions=actions,
    )


def enumerate_public_c_rulebook(
    reward_dimensions: int,
    public_schedule: PublicBonusSchedule,
    reward_spec: RewardSpec | None = None,
    *,
    max_matrix_entries: int = 2_000_000,
) -> EnumeratedPublicProblem:
    """Exhaustively cross a finite Rulebook with PUBLIC-C choices."""

    dimensions = _integer(reward_dimensions, "reward_dimensions", minimum=1)
    if not isinstance(public_schedule, PublicBonusSchedule):
        raise TypeError("public_schedule must be a PublicBonusSchedule")
    spec = _reward_spec(reward_spec)
    state_count = spec.q**dimensions
    hidden_action_count = (spec.q + 1) ** dimensions
    action_count = hidden_action_count * len(public_schedule.rewards)
    _guard_matrix(state_count, action_count, max_matrix_entries)

    states = tuple(itertools.product(range(1, spec.q + 1), repeat=dimensions))
    hidden_vectors = tuple(itertools.product(range(0, spec.q + 1), repeat=dimensions))
    hidden_actions = tuple(
        DeploymentAction(
            (index + 1, prediction) for index, prediction in enumerate(vector)
        )
        for vector in hidden_vectors
    )
    actions = tuple(
        PublicDeploymentAction(hidden, public_choice)
        for hidden in hidden_actions
        for public_choice in range(len(public_schedule.rewards))
    )
    rewards = tuple(
        tuple(
            math.fsum(
                spec.contribution(prediction, truth)
                for prediction, truth in zip(
                    hidden_vector,
                    state,
                    strict=True,
                )
            )
            + public_schedule.reward(public_choice)
            for hidden_vector in hidden_vectors
            for public_choice in range(len(public_schedule.rewards))
        )
        for state in states
    )
    prior = (1.0 / state_count,) * state_count
    return EnumeratedPublicProblem(
        problem=FiniteDecisionProblem(prior=prior, rewards=rewards),
        states=states,
        actions=actions,
    )


__all__ = [
    "EnumeratedPublicProblem",
    "PublicCFrontier",
    "PublicUWitness",
    "alea_frontier_problem",
    "alea_persistent_information_nats",
    "enumerate_public_c_rulebook",
    "enumerate_trivia_rulebook",
    "public_c_bit_equivalent",
    "public_u_bit_equivalent",
    "public_u_witness",
    "trivia_invariant_bit_equivalent",
]
