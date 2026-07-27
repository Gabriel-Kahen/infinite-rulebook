"""Mixed independent-rank and finite-core redundant Rulebook."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec, additive_reward
from infinite_rulebook.core.rng import CounterRNG, Seed
from infinite_rulebook.environments.redundant import cyclic_surface_map


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class MixedRulebook:
    """Odd rules are IID primitives; even rules derive from a finite core.

    Only even (redundant) entries count against ``max_redundant_support``.
    Consequently the redundant reward contribution is bounded while the
    independent odd-index contribution can grow without bound.
    """

    seed: Seed
    core_dimensions: int = 4
    max_redundant_support: int = 32
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    _primitive_rng: CounterRNG = field(init=False, repr=False, compare=False)
    _core_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_integer(self.core_dimensions, "core_dimensions")
        _nonnegative_integer(
            self.max_redundant_support,
            "max_redundant_support",
        )
        object.__setattr__(
            self,
            "_primitive_rng",
            CounterRNG(self.seed, stream="mixed.independent.v1"),
        )
        object.__setattr__(
            self,
            "_core_rng",
            CounterRNG(self.seed, stream="mixed.redundant-core.v1"),
        )

    @property
    def redundant_core(self) -> tuple[int, ...]:
        return tuple(
            1 + self._core_rng.randbelow(self.reward_spec.q, component)
            for component in range(self.core_dimensions)
        )

    @property
    def redundant_core_entropy(self) -> float:
        return self.core_dimensions * math.log(self.reward_spec.q)

    @property
    def maximum_redundant_reward(self) -> float:
        return self.max_redundant_support * self.reward_spec.u

    @staticmethod
    def is_independent(index: int) -> bool:
        _positive_integer(index, "rule indices")
        return index % 2 == 1

    def redundant_surface_map(self, index: int) -> tuple[int, int]:
        """Return the public map for an even-indexed derived rule."""

        _positive_integer(index, "rule indices")
        if index % 2:
            raise ValueError("redundant surface maps are defined only for even rules")
        derived_index = index // 2
        return cyclic_surface_map(
            derived_index,
            self.core_dimensions,
            self.reward_spec.q,
        )

    def label(self, index: int) -> int:
        _positive_integer(index, "rule indices")
        if index % 2:
            primitive_index = (index + 1) // 2
            return 1 + self._primitive_rng.randbelow(
                self.reward_spec.q, primitive_index
            )
        component, offset = self.redundant_surface_map(index)
        return 1 + ((self.redundant_core[component] - 1 + offset) % self.reward_spec.q)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(self.label(index) for index in indices)

    def evaluate(self, action: DeploymentAction) -> float:
        redundant_support = sum(index % 2 == 0 for index, _ in action)
        if redundant_support > self.max_redundant_support:
            raise ValueError(
                "MIX deployment exceeds max_redundant_support "
                f"({redundant_support} > {self.max_redundant_support})"
            )
        return additive_reward(action, self.label, self.reward_spec)
