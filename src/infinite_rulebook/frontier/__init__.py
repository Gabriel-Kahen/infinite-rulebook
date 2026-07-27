"""Reward-information frontier implementations."""

from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.tensorized import (
    TensorizedFrontier,
    infinite_bit_equivalent,
)

__all__ = [
    "OneCoordinateFrontier",
    "TensorizedFrontier",
    "infinite_bit_equivalent",
]
