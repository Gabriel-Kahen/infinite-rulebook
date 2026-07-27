"""Tensorized exact frontiers for independent Rulebook coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier


@dataclass(frozen=True, slots=True)
class TensorizedFrontier:
    """Exact finite-product frontier for exchangeable independent rules."""

    base: OneCoordinateFrontier
    dimensions: int

    def __post_init__(self) -> None:
        if not isinstance(self.base, OneCoordinateFrontier):
            raise TypeError("base must be a OneCoordinateFrontier")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise TypeError("dimensions must be an integer")
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")

    @property
    def maximum_reward(self) -> float:
        """Maximum attainable total reward."""

        return self.dimensions * self.base.u

    def bit_equivalent(self, total_reward: Real) -> float:
        """Return ``N B_1(total_reward / N)`` in nats."""

        if isinstance(total_reward, bool) or not isinstance(total_reward, Real):
            raise TypeError("total_reward must be a real number")
        target = float(total_reward)
        if math.isnan(target):
            raise ValueError("total_reward must not be NaN")
        if target <= 0.0:
            return 0.0
        if target > self.maximum_reward:
            return math.inf
        return self.dimensions * self.base.bit_equivalent(target / self.dimensions)


def infinite_bit_equivalent(base: OneCoordinateFrontier, reward: Real) -> float:
    """Return the infinite independent-coordinate frontier in nats."""

    if not isinstance(base, OneCoordinateFrontier):
        raise TypeError("base must be a OneCoordinateFrontier")
    return base.infinite_bit_equivalent(reward)
