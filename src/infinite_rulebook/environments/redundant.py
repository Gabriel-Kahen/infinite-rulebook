"""Stationary Rulebooks derived from a finite q-ary latent core."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec, additive_reward
from infinite_rulebook.core.rng import CounterRNG, Seed


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def cyclic_surface_map(
    index: int,
    core_dimensions: int,
    q: int,
) -> tuple[int, int]:
    """Return the registered balanced public map for a derived-rule ordinal."""

    _positive_integer(index, "rule indices")
    _positive_integer(core_dimensions, "core_dimensions")
    _positive_integer(q, "q")
    if q < 2:
        raise ValueError("q must be at least two")
    function = (index - 1) % (core_dimensions * q)
    return function % core_dimensions, function // core_dimensions


@dataclass(frozen=True, slots=True)
class UnrestrictedRedundantRulebook:
    """Countably many balanced public functions of one finite latent core.

    Surface rule ``i`` selects a core coordinate and a public cyclic offset.
    There are only ``core_dimensions * q`` distinct functions, repeated
    forever. The finite core is sampled once with a counter RNG, so labels do
    not depend on query or evaluation order.
    """

    seed: Seed
    core_dimensions: int = 4
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    _core_rng: CounterRNG = field(
        init=False,
        repr=False,
        compare=False,
        metadata={"artifact_exclude": True},
    )

    def __post_init__(self) -> None:
        _positive_integer(self.core_dimensions, "core_dimensions")
        object.__setattr__(
            self,
            "_core_rng",
            CounterRNG(self.seed, stream="redundant.core.v1"),
        )

    @property
    def core(self) -> tuple[int, ...]:
        """The fixed q-ary latent core."""

        return tuple(
            1 + self._core_rng.randbelow(self.reward_spec.q, component)
            for component in range(self.core_dimensions)
        )

    @property
    def core_entropy(self) -> float:
        """Prior entropy of the latent core, in nats."""

        return self.core_dimensions * math.log(self.reward_spec.q)

    def surface_map(self, index: int) -> tuple[int, int]:
        """Return public ``(core_coordinate, cyclic_offset)`` for a rule."""

        return cyclic_surface_map(index, self.core_dimensions, self.reward_spec.q)

    def label(self, index: int) -> int:
        component, offset = self.surface_map(index)
        return 1 + ((self.core[component] - 1 + offset) % self.reward_spec.q)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(self.label(index) for index in indices)

    def evaluate(self, action: DeploymentAction) -> float:
        """Return unrestricted additive reward without changing state."""

        return additive_reward(action, self.label, self.reward_spec)


@dataclass(frozen=True, slots=True)
class CappedRedundantRulebook(UnrestrictedRedundantRulebook):
    """RED-C: redundant reward with a stationary deployment-support cap."""

    max_derived_support: int = 32

    def __post_init__(self) -> None:
        super(CappedRedundantRulebook, self).__post_init__()
        _nonnegative_integer(self.max_derived_support, "max_derived_support")

    @property
    def maximum_reward(self) -> float:
        return self.max_derived_support * self.reward_spec.u

    def evaluate(self, action: DeploymentAction) -> float:
        if len(action) > self.max_derived_support:
            raise ValueError(
                "RED-C deployment exceeds max_derived_support "
                f"({len(action)} > {self.max_derived_support})"
            )
        return super(CappedRedundantRulebook, self).evaluate(action)
