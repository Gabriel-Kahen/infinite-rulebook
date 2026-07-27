"""Stationary symbolic controls for novelty, trivia, and public reward."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from numbers import Real
from typing import Generic, Protocol, TypeVar, runtime_checkable

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.core.rng import CounterRNG, Seed
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


ActionT = TypeVar("ActionT", contravariant=True)


@runtime_checkable
class RulebookRuntime(Protocol[ActionT]):
    """Structural runtime contract accepted by composable controls."""

    @property
    def reward_spec(self) -> RewardSpec: ...

    def label(self, index: int) -> int: ...

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]: ...

    def evaluate(self, action: ActionT) -> float: ...


def _runtime(value: object) -> RulebookRuntime[object]:
    if not isinstance(value, RulebookRuntime):
        raise TypeError("base must implement RulebookRuntime")
    if not isinstance(value.reward_spec, RewardSpec):
        raise TypeError("base.reward_spec must be a RewardSpec")
    return value


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
class ControlObservation:
    """A unified query result for arbitrarily composed symbolic controls."""

    symbolic: SymbolicObservation
    cosmetic_value: int | None = None


@dataclass(frozen=True, slots=True)
class AleaObservation:
    """A base observation with independent, nonpersistent cosmetic novelty."""

    reward_value: int
    cosmetic_value: int


def _base_observe_query(
    base: RulebookRuntime[object],
    query: SymbolicQuery,
    channel: QarySymmetricChannel,
    key: SemanticObservationKey,
) -> ControlObservation:
    if not isinstance(query, SymbolicQuery):
        raise TypeError("query must be a SymbolicQuery")
    if not isinstance(channel, QarySymmetricChannel):
        raise TypeError("channel must be a QarySymmetricChannel")
    if not isinstance(key, SemanticObservationKey):
        raise TypeError("key must be a SemanticObservationKey")
    observer = getattr(base, "observe_query", None)
    if callable(observer):
        result = observer(query, channel, key)
        if not isinstance(result, ControlObservation):
            raise TypeError("base.observe_query must return ControlObservation")
        return result
    if query.namespace is not QueryNamespace.REWARD:
        raise ValueError("base runtime does not expose a trivia namespace")
    if channel.q != base.reward_spec.q:
        raise ValueError("channel alphabet must match reward_spec.q")
    if key.rule_index != query.index:
        raise ValueError("key.rule_index must match query.index")
    value = channel.observe(base.label(query.index), key)
    return ControlObservation(SymbolicObservation(query, value))


@dataclass(frozen=True, slots=True)
class AleaRulebook(Generic[ActionT]):
    """A composable rulebook plus fresh cosmetic observations.

    Cosmetic values are keyed by the complete semantic observation coordinate.
    Replaying one coordinate is deterministic, but distinct rounds/ordinals
    are distinct draws. The cosmetic tape is not part of the persistent
    environment latent and never enters deployment reward. ``cosmetic_seed``
    is a separately sampled runtime-noise seed, independent of the base
    environment seed under the experiment's declared product seed law.
    """

    base: RulebookRuntime[ActionT]
    cosmetic_seed: Seed
    cosmetic_alphabet: int = 256
    _cosmetic_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _runtime(self.base)
        _positive_integer(self.cosmetic_alphabet, "cosmetic_alphabet")
        object.__setattr__(
            self,
            "_cosmetic_rng",
            CounterRNG(
                self.cosmetic_seed,
                stream="alea.cosmetic-observation.v1",
            ),
        )

    @property
    def reward_spec(self) -> RewardSpec:
        return self.base.reward_spec

    def label(self, index: int) -> int:
        return self.base.label(index)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return self.base.labels(indices)

    def evaluate(self, action: ActionT) -> float:
        return self.base.evaluate(action)

    def cosmetic_value(self, key: SemanticObservationKey) -> int:
        """Return fresh-event novelty keyed only by the ALEA tape."""

        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        return self._cosmetic_rng.randbelow(
            self.cosmetic_alphabet,
            key.round_index,
            key.rule_index,
            key.query_ordinal,
            key.channel,
        )

    def augment_observation(
        self,
        reward_value: int,
        key: SemanticObservationKey,
    ) -> AleaObservation:
        """Append cosmetic novelty to any q-ary base observation."""

        if (
            not isinstance(reward_value, int)
            or isinstance(reward_value, bool)
            or not 1 <= reward_value <= self.reward_spec.q
        ):
            raise ValueError(
                f"reward_value must be an integer in 1..{self.reward_spec.q}"
            )
        return AleaObservation(reward_value, self.cosmetic_value(key))

    def observe_query(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> ControlObservation:
        """Observe any base namespace and append ALEA cosmetic novelty."""

        if not isinstance(query, SymbolicQuery):
            raise TypeError("query must be a SymbolicQuery")
        if not isinstance(channel, QarySymmetricChannel):
            raise TypeError("channel must be a QarySymmetricChannel")
        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        base_observation = _base_observe_query(self.base, query, channel, key)
        cosmetic_key = SemanticObservationKey(
            environment_seed=key.environment_seed,
            round_index=key.round_index,
            rule_index=key.rule_index,
            query_ordinal=key.query_ordinal,
            channel=f"{key.channel}.{query.namespace.value}",
        )
        return ControlObservation(
            symbolic=base_observation.symbolic,
            cosmetic_value=self.cosmetic_value(cosmetic_key),
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
        observation = self.observe_query(
            SymbolicQuery(QueryNamespace.REWARD, rule_index),
            channel,
            key,
        )
        if observation.cosmetic_value is None:
            raise RuntimeError("ALEA observation is missing cosmetic novelty")
        return AleaObservation(
            observation.symbolic.value,
            observation.cosmetic_value,
        )


@dataclass(frozen=True, slots=True)
class TriviaRulebook(Generic[ActionT]):
    """A composable rulebook plus persistent, queryable irrelevant labels."""

    base: RulebookRuntime[ActionT]
    trivia_seed: Seed
    _trivia_rng: CounterRNG = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _runtime(self.base)
        object.__setattr__(
            self,
            "_trivia_rng",
            CounterRNG(
                self.trivia_seed,
                stream="trivia.persistent-latent.v1",
            ),
        )

    @property
    def reward_spec(self) -> RewardSpec:
        return self.base.reward_spec

    def label(self, index: int) -> int:
        return self.base.label(index)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return self.base.labels(indices)

    def evaluate(self, action: ActionT) -> float:
        return self.base.evaluate(action)

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
        namespaced_key = self.observation_key(query, key)
        value = channel.observe(self.query_label(query), namespaced_key)
        return SymbolicObservation(query, value)

    def observe_query(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> ControlObservation:
        return ControlObservation(self.observe(query, channel, key))

    def observation_key(
        self,
        query: SymbolicQuery,
        key: SemanticObservationKey,
    ) -> SemanticObservationKey:
        """Domain-separate useful and trivia observation-noise draws."""

        if not isinstance(query, SymbolicQuery):
            raise TypeError("query must be a SymbolicQuery")
        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        if key.rule_index != query.index:
            raise ValueError("key.rule_index must match query.index")
        if query.namespace is QueryNamespace.REWARD:
            return key
        return SemanticObservationKey(
            environment_seed=key.environment_seed,
            round_index=key.round_index,
            rule_index=key.rule_index,
            query_ordinal=key.query_ordinal,
            channel=f"{key.channel}.trivia",
        )


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
class CappedPublicRulebook:
    """A composable PUBLIC-C wrapper with a fixed public contribution."""

    base: RulebookRuntime[DeploymentAction]
    public_schedule: PublicBonusSchedule = field(default_factory=PublicBonusSchedule)

    def __post_init__(self) -> None:
        _runtime(self.base)
        if not isinstance(self.public_schedule, PublicBonusSchedule):
            raise TypeError("public_schedule must be a PublicBonusSchedule")

    @property
    def reward_spec(self) -> RewardSpec:
        return self.base.reward_spec

    def label(self, index: int) -> int:
        return self.base.label(index)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return self.base.labels(indices)

    @property
    def maximum_public_reward(self) -> float:
        return self.public_schedule.maximum_reward

    def evaluate(self, action: PublicDeploymentAction) -> float:
        if not isinstance(action, PublicDeploymentAction):
            raise TypeError("action must be a PublicDeploymentAction")
        hidden_reward = self.base.evaluate(action.deployment)
        return hidden_reward + self.public_schedule.reward(action.public_choice)

    def observe_query(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> ControlObservation:
        """Forward training queries without exposing public reward to feedback."""

        return _base_observe_query(self.base, query, channel, key)


@dataclass(frozen=True, slots=True)
class UnboundedPublicRulebook:
    """A composable PUBLIC-U wrapper with reward ``unit_reward * k``."""

    base: RulebookRuntime[DeploymentAction]
    public_unit_reward: float = 1.0

    def __post_init__(self) -> None:
        _runtime(self.base)
        reward = _finite_nonnegative(self.public_unit_reward, "public_unit_reward")
        if reward == 0.0:
            raise ValueError("public_unit_reward must be positive")
        object.__setattr__(self, "public_unit_reward", reward)

    @property
    def reward_spec(self) -> RewardSpec:
        return self.base.reward_spec

    def label(self, index: int) -> int:
        return self.base.label(index)

    def labels(self, indices: Iterable[int]) -> tuple[int, ...]:
        return self.base.labels(indices)

    def public_reward(self, choice: int) -> float:
        allocation = _nonnegative_integer(choice, "public_choice")
        try:
            reward = float(allocation * Fraction.from_float(self.public_unit_reward))
        except OverflowError as error:
            raise ValueError("public_choice produces non-finite reward") from error
        if not math.isfinite(reward):
            raise ValueError("public_choice produces non-finite reward")
        return reward

    def evaluate(self, action: PublicDeploymentAction) -> float:
        if not isinstance(action, PublicDeploymentAction):
            raise TypeError("action must be a PublicDeploymentAction")
        hidden_reward = self.base.evaluate(action.deployment)
        return hidden_reward + self.public_reward(action.public_choice)

    def observe_query(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> ControlObservation:
        """Forward training queries without exposing public reward to feedback."""

        return _base_observe_query(self.base, query, channel, key)


__all__ = [
    "AleaObservation",
    "AleaRulebook",
    "CappedPublicRulebook",
    "ControlObservation",
    "PublicBonusSchedule",
    "PublicDeploymentAction",
    "QueryNamespace",
    "RulebookRuntime",
    "SymbolicObservation",
    "SymbolicQuery",
    "TriviaRulebook",
    "UnboundedPublicRulebook",
]
