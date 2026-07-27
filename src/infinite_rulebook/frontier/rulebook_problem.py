"""Finite enumerations of independent Rulebook decision problems."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem


@dataclass(frozen=True, slots=True)
class EnumeratedRulebookProblem:
    """A finite decision problem with its semantic states and actions."""

    problem: FiniteDecisionProblem
    states: tuple[tuple[int, ...], ...]
    actions: tuple[DeploymentAction, ...]


def enumerate_independent_rulebook(
    dimensions: int,
    reward_spec: RewardSpec | None = None,
    *,
    max_matrix_entries: int = 2_000_000,
) -> EnumeratedRulebookProblem:
    """Enumerate a finite independent Rulebook projection.

    The action set contains every abstain-or-predict vector on the projection.
    The explicit size guard prevents an accidental combinatorial allocation.
    """

    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise TypeError("dimensions must be an integer")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if (
        isinstance(max_matrix_entries, bool)
        or not isinstance(max_matrix_entries, int)
        or max_matrix_entries < 1
    ):
        raise ValueError("max_matrix_entries must be a positive integer")

    if reward_spec is None:
        reward_spec = RewardSpec()

    q = reward_spec.q
    state_count = q**dimensions
    action_count = (q + 1) ** dimensions
    if state_count * action_count > max_matrix_entries:
        raise ValueError(
            "enumerated reward matrix would exceed max_matrix_entries: "
            f"{state_count} * {action_count}"
        )

    states = tuple(itertools.product(range(1, q + 1), repeat=dimensions))
    action_vectors = tuple(itertools.product(range(0, q + 1), repeat=dimensions))
    actions = tuple(
        DeploymentAction(
            (index + 1, prediction) for index, prediction in enumerate(vector)
        )
        for vector in action_vectors
    )

    rewards = tuple(
        tuple(
            sum(
                reward_spec.contribution(prediction, truth)
                for prediction, truth in zip(
                    action_vector,
                    state,
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
