"""Strict-margin additive Rulebook rewards."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from infinite_rulebook.core.behavior import DeploymentAction


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Reward parameters for a q-ary prediction with abstention."""

    q: int = 4
    u: float = 1.0
    c: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.q, int) or isinstance(self.q, bool) or self.q < 2:
            raise ValueError("q must be an integer of at least two")
        if isinstance(self.u, bool) or not math.isfinite(self.u) or self.u <= 0:
            raise ValueError("u must be finite and positive")
        if isinstance(self.c, bool) or not math.isfinite(self.c) or self.c <= 0:
            raise ValueError("c must be finite and positive")
        if self.c <= self.u / (self.q - 1):
            raise ValueError("strict margin requires c > u / (q - 1)")

    @property
    def profitability_threshold(self) -> float:
        return self.c / (self.u + self.c)

    @property
    def uninformed_reward(self) -> float:
        return self.u / self.q - self.c * (1.0 - 1.0 / self.q)

    def contribution(self, prediction: int, truth: int) -> float:
        if prediction == 0:
            return 0.0
        if (
            not isinstance(prediction, int)
            or isinstance(prediction, bool)
            or not isinstance(truth, int)
            or isinstance(truth, bool)
            or not 1 <= prediction <= self.q
            or not 1 <= truth <= self.q
        ):
            raise ValueError(f"predictions and labels must lie in 1..{self.q}")
        return self.u if prediction == truth else -self.c

    def from_counts(self, correct: int, incorrect: int) -> float:
        if (
            not isinstance(correct, int)
            or isinstance(correct, bool)
            or not isinstance(incorrect, int)
            or isinstance(incorrect, bool)
            or correct < 0
            or incorrect < 0
        ):
            raise ValueError("reward counts must be nonnegative")
        return correct * self.u - incorrect * self.c


def additive_reward(
    action: DeploymentAction,
    label_for: Callable[[int], int],
    spec: RewardSpec,
) -> float:
    """Return the exact realized reward of one finite deployment."""

    action.validate_alphabet(spec.q)
    correct = 0
    for index, prediction in action:
        truth = label_for(index)
        if not isinstance(truth, int) or isinstance(truth, bool):
            raise TypeError("labels must be integers")
        if not 1 <= truth <= spec.q:
            raise ValueError(f"label {truth} is outside alphabet 1..{spec.q}")
        correct += prediction == truth
    return spec.from_counts(correct, len(action) - correct)
