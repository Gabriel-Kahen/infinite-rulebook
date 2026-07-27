"""Stationary symbolic controls for novelty, trivia, and public reward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.rng import CounterRNG, Seed
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


class QueryNamespace(StrEnum):
    """Disjoint namespaces for useful and reward-irrelevant coordinates."""

    REWARD = "reward"
    TRIVIA = "trivia"


@dataclass(frozen=True, slots=True)
class SymbolicQuery:
    """A typed query for one persistent symbolic coordinate."""

    namespace: QueryNamespace
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, QueryNamespace):
            raise TypeError("namespace must be a QueryNamespace")
        _positive_integer(self.index, "index")


@dataclass(frozen=True, slots=True)
class SymbolicObservation:
    """One noisy observation of a persistent symbolic coordinate."""

    query: SymbolicQuery
    value: int


@dataclass(frozen=True, slots=True)
class AleaObservation:
    """A useful observation with independent, nonpersistent cosmetic novelty."""

    reward_value: int
    cosmetic_value: int


@dataclass(frozen=True, slots=True)
class AleaRulebook(IndependentRulebook):
    """IND plus fresh reward-irrelevant cosmetic observations.

    Cosmetic values are keyed by the complete semantic observation coordinate.
    Replaying one coordinate is deterministic, but distinct rounds/ordinals
    are distinct draws. The cosmetic tape is not part of the persistent
    environment latent and never enters deployment reward.
    """

    cosmetic_alphabet: int = 256
    _cosmetic_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super(AleaRulebook, self).__post_init__()
        _positive_integer(self.cosmetic_alphabet, "cosmetic_alphabet")
        object.__setattr__(
            self,
            "_cosmetic_rng",
            CounterRNG(self.seed, stream="alea.cosmetic-observation.v1"),
        )

    def observe(
        self,
        rule_index: int,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> AleaObservation:
        """Observe a useful label and fresh cosmetic value without mutation."""

        _positive_integer(rule_index, "rule_index")
        if not isinstance(channel, QarySymmetricChannel):
            raise TypeError("channel must be a QarySymmetricChannel")
        if channel.q != self.reward_spec.q:
            raise ValueError("channel alphabet must match reward_spec.q")
        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        if key.rule_index != rule_index:
            raise ValueError("key.rule_index must match rule_index")
        reward_value = channel.observe(self.label(rule_index), key)
        cosmetic_value = self._cosmetic_rng.randbelow(
            self.cosmetic_alphabet,
            key.round_index,
            key.rule_index,
            key.query_ordinal,
            key.channel,
        )
        return AleaObservation(reward_value, cosmetic_value)


@dataclass(frozen=True, slots=True)
class TriviaRulebook(IndependentRulebook):
    """IND plus countably many persistent, queryable irrelevant labels."""

    trivia_seed: Seed | None = None
    _trivia_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super(TriviaRulebook, self).__post_init__()
        seed = self.seed if self.trivia_seed is None else self.trivia_seed
        object.__setattr__(
            self,
            "_trivia_rng",
            CounterRNG(seed, stream="trivia.persistent-latent.v1"),
        )

    def trivia_label(self, index: int) -> int:
        """Return one fixed reward-irrelevant q-ary latent coordinate."""

        _positive_integer(index, "trivia index")
        return 1 + self._trivia_rng.randbelow(self.reward_spec.q, index)

    def query_label(self, query: SymbolicQuery) -> int:
        """Return the persistent label selected by a typed query."""

        if not isinstance(query, SymbolicQuery):
            raise TypeError("query must be a SymbolicQuery")
        if query.namespace is QueryNamespace.REWARD:
            return self.label(query.index)
        return self.trivia_label(query.index)

    def observe(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> SymbolicObservation:
        """Observe useful rules and trivia through the same channel family."""

        if not isinstance(query, SymbolicQuery):
            raise TypeError("query must be a SymbolicQuery")
        if not isinstance(channel, QarySymmetricChannel):
            raise TypeError("channel must be a QarySymmetricChannel")
        if channel.q != self.reward_spec.q:
            raise ValueError("channel alphabet must match reward_spec.q")
        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        if key.rule_index != query.index:
            raise ValueError("key.rule_index must match query.index")
        value = channel.observe(self.query_label(query), key)
        return SymbolicObservation(query, value)


@dataclass(frozen=True, slots=True)
class PublicDeploymentAction:
    """A hidden-label deployment paired with a public integer choice."""

    deployment: DeploymentAction = field(default_factory=DeploymentAction)
    public_choice: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, DeploymentAction):
            raise TypeError("deployment must be a DeploymentAction")
        _nonnegative_integer(self.public_choice, "public_choice")


@dataclass(frozen=True, slots=True)
class PublicBonusSchedule:
    """A fixed finite public action set for PUBLIC-C."""

    rewards: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        try:
            rewards = tuple(
                _finite_nonnegative(value, f"rewards[{index}]")
                for index, value in enumerate(self.rewards)
            )
        except TypeError as error:
            raise TypeError("rewards must be a finite sequence") from error
        if not rewards:
            raise ValueError("rewards must contain at least one public choice")
        object.__setattr__(self, "rewards", rewards)

    @property
    def maximum_reward(self) -> float:
        return max(self.rewards)

    @property
    def maximizing_choice(self) -> int:
        return max(range(len(self.rewards)), key=self.rewards.__getitem__)

    def reward(self, choice: int) -> float:
        _nonnegative_integer(choice, "public_choice")
        if choice >= len(self.rewards):
            raise ValueError("public_choice is outside the bounded public action set")
        return self.rewards[choice]


@dataclass(frozen=True, slots=True)
class CappedPublicRulebook(IndependentRulebook):
    """PUBLIC-C with a fixed, bounded, attained public contribution."""

    public_schedule: PublicBonusSchedule = field(default_factory=PublicBonusSchedule)

    def __post_init__(self) -> None:
        super(CappedPublicRulebook, self).__post_init__()
        if not isinstance(self.public_schedule, PublicBonusSchedule):
            raise TypeError("public_schedule must be a PublicBonusSchedule")

    @property
    def maximum_public_reward(self) -> float:
        return self.public_schedule.maximum_reward

    def evaluate(self, action: PublicDeploymentAction) -> float:
        if not isinstance(action, PublicDeploymentAction):
            raise TypeError("action must be a PublicDeploymentAction")
        hidden_reward = super(CappedPublicRulebook, self).evaluate(action.deployment)
        return hidden_reward + self.public_schedule.reward(action.public_choice)


@dataclass(frozen=True, slots=True)
class UnboundedPublicRulebook(IndependentRulebook):
    """PUBLIC-U with public reward ``unit_reward * k`` for any finite ``k``."""

    public_unit_reward: float = 1.0

    def __post_init__(self) -> None:
        super(UnboundedPublicRulebook, self).__post_init__()
        reward = _finite_nonnegative(self.public_unit_reward, "public_unit_reward")
        if reward == 0.0:
            raise ValueError("public_unit_reward must be positive")
        object.__setattr__(self, "public_unit_reward", reward)

    def public_reward(self, choice: int) -> float:
        allocation = _nonnegative_integer(choice, "public_choice")
        try:
            reward = allocation * self.public_unit_reward
        except OverflowError as error:
            raise ValueError("public_choice produces non-finite reward") from error
        if not math.isfinite(reward):
            raise ValueError("public_choice produces non-finite reward")
        return reward

    def evaluate(self, action: PublicDeploymentAction) -> float:
        if not isinstance(action, PublicDeploymentAction):
            raise TypeError("action must be a PublicDeploymentAction")
        hidden_reward = super(UnboundedPublicRulebook, self).evaluate(action.deployment)
        return hidden_reward + self.public_reward(action.public_choice)


__all__ = [
    "AleaObservation",
    "AleaRulebook",
    "CappedPublicRulebook",
    "PublicBonusSchedule",
    "PublicDeploymentAction",
    "QueryNamespace",
    "SymbolicObservation",
    "SymbolicQuery",
    "TriviaRulebook",
    "UnboundedPublicRulebook",
]
