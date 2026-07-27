"""The stationary independent q-ary Rulebook environment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec, additive_reward
from infinite_rulebook.core.rng import CounterRNG, Seed


@dataclass(frozen=True, slots=True)
class IndependentRulebook:
    """Countably many IID labels, sampled once and generated lazily."""

    seed: Seed
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    _latent_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_latent_rng",
            CounterRNG(self.seed, stream="independent.latent.v1"),
        )

    def label(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            raise ValueError("rule indices must be positive integers")
        return 1 + self._latent_rng.randbelow(self.reward_spec.q, index)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(self.label(index) for index in indices)

    def evaluate(self, action: DeploymentAction) -> float:
        """Evaluate without observations, mutation, or RNG-state consumption."""

        return additive_reward(action, self.label, self.reward_spec)
