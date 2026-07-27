"""Factorized symbolic agents for matched acquisition comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from infinite_rulebook.agents.objectives import (
    bayes_deployment_value,
    decision_value_gain,
    expected_entropy_reduction,
    prediction_error_novelty,
)
from infinite_rulebook.agents.protocols import (
    AcquisitionContext,
    AgentCheckpoint,
    CapabilityManifest,
    ObservationBatch,
    QueryAction,
    QueryTarget,
    TargetKey,
)
from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.core.rng import CounterRNG, Seed
from infinite_rulebook.environments.controls import QueryNamespace
from infinite_rulebook.posteriors.categorical import CategoricalPosterior

_FACTOR_CAPABILITIES = CapabilityManifest(
    # All matched agents receive the same relevance metadata for deployment.
    # The experiment varies whether the acquisition objective uses that mask.
    knows_relevance_mask=True,
    knows_coordinate_factorization=True,
    knows_true_posterior_family=True,
    knows_reward_parameters=True,
)
_TARGET_CAPABILITIES = CapabilityManifest(
    knows_relevance_mask=True,
    knows_coordinate_factorization=True,
    knows_target_hierarchy=True,
    knows_true_posterior_family=True,
    knows_reward_parameters=True,
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True, slots=True)
class ExpandingTargetSchedule:
    """A target size that changes only as a pure function of round index."""

    initial_size: int
    growth_step: int
    growth_interval: int = 1
    maximum_size: int | None = None

    def __post_init__(self) -> None:
        _integer(self.initial_size, "initial_size")
        _integer(self.growth_step, "growth_step", minimum=1)
        _integer(self.growth_interval, "growth_interval", minimum=1)
        if self.maximum_size is not None:
            _integer(self.maximum_size, "maximum_size")
            if self.maximum_size < self.initial_size:
                raise ValueError("maximum_size must not be below initial_size")

    def size_at(self, round_index: int) -> int:
        _integer(round_index, "round_index")
        size = self.initial_size + self.growth_step * (
            round_index // self.growth_interval
        )
        return size if self.maximum_size is None else min(size, self.maximum_size)


class AcquisitionPolicy(Protocol):
    """A pure scoring rule over exposed factorized targets."""

    capabilities: CapabilityManifest

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None: ...


@dataclass(frozen=True, slots=True)
class FixedTargetPolicy:
    """Information-directed allocation within a fixed useful prefix."""

    target_size: int
    namespace: str = QueryNamespace.REWARD.value
    capabilities: CapabilityManifest = field(
        default=_TARGET_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _integer(self.target_size, "target_size", minimum=1)
        if not isinstance(self.namespace, str):
            raise TypeError("namespace must be a string")
        if not self.namespace:
            raise ValueError("namespace must not be empty")

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        del reward_spec, round_index
        if (
            target.key.namespace != self.namespace
            or target.key.index > self.target_size
            or target.rule_index is None
            or target.relevance_weight <= 0.0
        ):
            return None
        return expected_entropy_reduction(posterior)


@dataclass(frozen=True, slots=True)
class ScheduledTargetPolicy:
    """Information-directed allocation within an expanding useful prefix."""

    schedule: ExpandingTargetSchedule
    namespace: str = QueryNamespace.REWARD.value
    capabilities: CapabilityManifest = field(
        default=_TARGET_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ExpandingTargetSchedule):
            raise TypeError("schedule must be an ExpandingTargetSchedule")
        if not isinstance(self.namespace, str):
            raise TypeError("namespace must be a string")
        if not self.namespace:
            raise ValueError("namespace must not be empty")

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        del reward_spec
        if (
            target.key.namespace != self.namespace
            or target.key.index > self.schedule.size_at(round_index)
            or target.rule_index is None
            or target.relevance_weight <= 0.0
        ):
            return None
        return expected_entropy_reduction(posterior)


@dataclass(frozen=True, slots=True)
class RewardDirectedPolicy:
    """Select queries by expected improvement in Bayes deployment reward."""

    capabilities: CapabilityManifest = field(
        default=_FACTOR_CAPABILITIES,
        init=False,
        repr=False,
    )

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        del round_index
        if target.rule_index is None or target.relevance_weight <= 0.0:
            return None
        gain = decision_value_gain(posterior, reward_spec)
        if gain > 0.0:
            return 1.0 + target.relevance_weight * gain / reward_spec.u
        information_gain = expected_entropy_reduction(posterior)
        if (
            information_gain <= 0.0
            or bayes_deployment_value(
                posterior,
                reward_spec,
            )
            > 0.0
        ):
            return 0.0
        # An informative observation may need several repetitions before the
        # strict deployment threshold is crossed. Prefer a started coordinate
        # so a broad target does not spread all queries across fresh rules.
        return (
            target.relevance_weight * math.ulp(1.0) * (1 + posterior.total_observations)
        )


@dataclass(frozen=True, slots=True)
class NoveltyDirectedPolicy:
    """Maximize prediction error of the full symbolic-plus-cosmetic observation."""

    cosmetic_alphabet: int = 1
    capabilities: CapabilityManifest = field(
        default=_FACTOR_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _integer(self.cosmetic_alphabet, "cosmetic_alphabet", minimum=1)

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float:
        del target, reward_spec, round_index
        symbolic_error = prediction_error_novelty(posterior)
        return 1.0 - (1.0 - symbolic_error) / self.cosmetic_alphabet


@dataclass(frozen=True, slots=True)
class TotalInformationDirectedPolicy:
    """Maximize persistent full-posterior information gain."""

    capabilities: CapabilityManifest = field(
        default=_FACTOR_CAPABILITIES,
        init=False,
        repr=False,
    )

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        del reward_spec, round_index
        if not target.persistent:
            return None
        return expected_entropy_reduction(posterior)


@dataclass(frozen=True, slots=True)
class RelevantInformationDirectedPolicy:
    """Privileged oracle maximizing information about useful coordinates."""

    capabilities: CapabilityManifest = field(
        default=_FACTOR_CAPABILITIES,
        init=False,
        repr=False,
    )

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        del reward_spec, round_index
        if not target.persistent or target.relevance_weight <= 0.0:
            return None
        return target.relevance_weight * expected_entropy_reduction(posterior)


@dataclass(slots=True)
class FactorizedQueryAgent:
    """Shared posterior state with separate acquisition and deployment surfaces."""

    policy: AcquisitionPolicy
    q: int = 4
    epsilon: float = 0.2
    query_budget: int = 1
    seed: Seed = 0
    reward_spec: RewardSpec | None = None
    _posteriors: dict[TargetKey, CategoricalPosterior] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _targets: dict[TargetKey, QueryTarget] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _query_counts: dict[TargetKey, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _completed_rounds: int = field(default=0, init=False, repr=False)
    _pending_action: QueryAction | None = field(default=None, init=False, repr=False)
    _tie_rng: CounterRNG = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _integer(self.q, "q", minimum=2)
        _integer(self.query_budget, "query_budget")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(self.epsilon)
            or not 0.0 <= self.epsilon < (self.q - 1) / self.q
        ):
            raise ValueError("epsilon must satisfy 0 <= epsilon < (q - 1) / q")
        if not isinstance(
            getattr(self.policy, "capabilities", None),
            CapabilityManifest,
        ):
            raise TypeError("policy must declare a CapabilityManifest")
        if self.reward_spec is None:
            self.reward_spec = RewardSpec(q=self.q)
        elif not isinstance(self.reward_spec, RewardSpec):
            raise TypeError("reward_spec must be a RewardSpec")
        elif self.reward_spec.q != self.q:
            raise ValueError("reward_spec and agent alphabet sizes must match")
        self._tie_rng = CounterRNG(self.seed, stream="agent.tie-break.v1")

    @property
    def capabilities(self) -> CapabilityManifest:
        return self.policy.capabilities

    @property
    def completed_rounds(self) -> int:
        return self._completed_rounds

    def acquisition_context(
        self,
        candidates: Sequence[QueryTarget],
        *,
        query_budget: int | None = None,
    ) -> AcquisitionContext:
        """Return an immutable context without changing training state."""

        budget = self.query_budget if query_budget is None else query_budget
        _integer(budget, "query_budget")
        if budget > self.query_budget:
            raise ValueError("context query budget exceeds the agent budget")
        candidate_tuple = tuple(candidates)
        return AcquisitionContext(
            round_index=self._completed_rounds,
            query_budget=budget,
            candidates=candidate_tuple,
        )

    def select_train_action(self, context: AcquisitionContext) -> QueryAction:
        """Choose at most the configured budget using stateless tie-breaking."""

        if not isinstance(context, AcquisitionContext):
            raise TypeError("context must be an AcquisitionContext")
        if context.round_index != self._completed_rounds:
            raise ValueError("context round does not match agent state")
        if context.query_budget > self.query_budget:
            raise ValueError("context query budget exceeds the agent budget")
        if self._pending_action is not None:
            context.validate_action(self._pending_action)
            return self._pending_action

        ranked: list[tuple[float, int, TargetKey, QueryTarget]] = []
        for target in sorted(context.candidates, key=lambda candidate: candidate.key):
            posterior = self._posterior_for(target)
            score = self.policy.score(
                target,
                posterior,
                self.reward_spec,
                context.round_index,
            )
            if score is None:
                continue
            if not math.isfinite(score):
                raise ValueError("policy scores must be finite")
            if score <= 0.0:
                continue
            tie_break = self._tie_rng.uint64(
                context.round_index,
                target.key.namespace,
                target.key.index,
            )
            ranked.append((score, tie_break, target.key, target))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        action = QueryAction(
            context.round_index,
            tuple(item[3] for item in ranked[: context.query_budget]),
        )
        context.validate_action(action)
        self._pending_action = action
        return action

    def observe(self, batch: ObservationBatch) -> None:
        """Apply exactly one bounded P1 batch to persistent target posteriors."""

        if not isinstance(batch, ObservationBatch):
            raise TypeError("batch must be an ObservationBatch")
        if batch.action.round_index != self._completed_rounds:
            raise ValueError("observation round does not match agent state")
        if self._pending_action is None or batch.action != self._pending_action:
            raise ValueError("observations must match the selected acquisition action")
        if len(batch.action.targets) > self.query_budget:
            raise ValueError("observation batch exceeds the agent budget")
        if any(observation > self.q for observation in batch.observations):
            raise ValueError(f"observations must lie in 1..{self.q}")

        rule_owners = {
            target.rule_index: key
            for key, target in self._targets.items()
            if target.rule_index is not None
        }
        for target in batch.action.targets:
            known = self._targets.get(target.key)
            if known is not None and known != target:
                raise ValueError("target metadata changed across rounds")
            owner = (
                None
                if target.rule_index is None
                else rule_owners.get(target.rule_index)
            )
            if owner is not None and owner != target.key:
                raise ValueError("one deployment rule cannot have multiple targets")

        for target, observation in batch.items:
            self._targets[target.key] = target
            self._query_counts[target.key] = self._query_counts.get(target.key, 0) + 1
            if target.persistent:
                self._posteriors.setdefault(
                    target.key,
                    CategoricalPosterior(q=self.q, epsilon=self.epsilon),
                ).update(observation)
        self._completed_rounds += 1
        self._pending_action = None

    def deployment(self) -> DeploymentAction:
        """Return a fresh immutable behavioral snapshot without acquisition."""

        predictions: dict[int, int] = {}
        for key, posterior in self._posteriors.items():
            target = self._targets.get(key)
            if (
                target is None
                or target.rule_index is None
                or target.relevance_weight <= 0.0
            ):
                continue
            prediction = posterior.deployment(self.reward_spec.profitability_threshold)
            if not prediction:
                continue
            previous = predictions.setdefault(target.rule_index, prediction)
            if previous != prediction:
                raise ValueError("targets disagree about one deployment rule")
        return DeploymentAction(predictions.items())

    def checkpoint(self) -> AgentCheckpoint:
        """Freeze state without mutating posteriors, schedules, or RNG."""

        return AgentCheckpoint(
            round_index=self._completed_rounds,
            deployment=self.deployment(),
            posterior_snapshots={
                key: posterior.probabilities
                for key, posterior in self._posteriors.items()
            },
            query_counts=self._query_counts,
        )

    def posterior(self, key: TargetKey) -> CategoricalPosterior:
        """Return an independent posterior copy for diagnostics."""

        if not isinstance(key, TargetKey):
            raise TypeError("key must be a TargetKey")
        posterior = self._posteriors.get(key)
        return (
            CategoricalPosterior(q=self.q, epsilon=self.epsilon)
            if posterior is None
            else posterior.copy()
        )

    def query_count(self, key: TargetKey) -> int:
        """Return the completed acquisition count for one target."""

        if not isinstance(key, TargetKey):
            raise TypeError("key must be a TargetKey")
        return self._query_counts.get(key, 0)

    def _posterior_for(self, target: QueryTarget) -> CategoricalPosterior:
        if target.persistent and target.key in self._posteriors:
            return self._posteriors[target.key]
        return CategoricalPosterior(q=self.q, epsilon=self.epsilon)


def useful_targets(
    count: int,
    *,
    namespace: str = QueryNamespace.REWARD.value,
) -> tuple[QueryTarget, ...]:
    """Return a canonical useful prefix for IND-style comparisons."""

    _integer(count, "count")
    _validate_namespace(namespace)
    return tuple(
        QueryTarget(
            key=TargetKey(namespace, index),
            rule_index=index,
            relevance_weight=1.0,
            persistent=True,
        )
        for index in range(1, count + 1)
    )


def distractor_targets(
    count: int,
    *,
    namespace: str = QueryNamespace.TRIVIA.value,
    persistent: bool = True,
) -> tuple[QueryTarget, ...]:
    """Return reward-irrelevant persistent or aleatoric query targets."""

    _integer(count, "count")
    _validate_namespace(namespace)
    if not isinstance(persistent, bool):
        raise TypeError("persistent must be a boolean")
    return tuple(
        QueryTarget(
            key=TargetKey(namespace, index),
            relevance_weight=0.0,
            persistent=persistent,
        )
        for index in range(1, count + 1)
    )


def _validate_namespace(namespace: object) -> None:
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if not namespace:
        raise ValueError("namespace must not be empty")
