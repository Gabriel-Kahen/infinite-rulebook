"""Shared immutable contracts for symbolic acquisition and deployment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from infinite_rulebook.core.behavior import DeploymentAction


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _freeze_posteriors(
    snapshots: Mapping[TargetKey, Sequence[float]],
) -> Mapping[TargetKey, tuple[float, ...]]:
    frozen: dict[TargetKey, tuple[float, ...]] = {}
    for key, probabilities in snapshots.items():
        if not isinstance(key, TargetKey):
            raise TypeError("posterior keys must be TargetKey instances")
        values = tuple(probabilities)
        if len(values) < 2:
            raise ValueError("posterior snapshots need at least two probabilities")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in values
        ):
            raise ValueError("posterior probabilities must be finite and nonnegative")
        if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("posterior probabilities must sum to one")
        frozen[key] = tuple(float(value) for value in values)
    return MappingProxyType(frozen)


def _freeze_counts(
    counts: Mapping[TargetKey, int],
) -> Mapping[TargetKey, int]:
    frozen: dict[TargetKey, int] = {}
    for key, count in counts.items():
        if not isinstance(key, TargetKey):
            raise TypeError("count keys must be TargetKey instances")
        _nonnegative_int(count, "query count")
        frozen[key] = count
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Structural privileges that must accompany every reported agent."""

    knows_relevance_mask: bool = False
    knows_coordinate_factorization: bool = False
    knows_latent_dependency_graph: bool = False
    knows_target_hierarchy: bool = False
    knows_true_posterior_family: bool = False
    knows_exact_frontier: bool = False
    knows_approximate_frontier: bool = False
    knows_reward_parameters: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True, order=True)
class TargetKey:
    """Stable identity for a queryable target in a named namespace."""

    namespace: str
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str):
            raise TypeError("namespace must be a string")
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        _nonnegative_int(self.index, "index")
        if self.index == 0:
            raise ValueError("index must be positive")


@dataclass(frozen=True, slots=True)
class QueryTarget:
    """One exposed acquisition target and its declared reward relevance."""

    key: TargetKey
    rule_index: int | None = None
    relevance_weight: float = 1.0
    persistent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.key, TargetKey):
            raise TypeError("key must be a TargetKey")
        if self.rule_index is not None:
            _nonnegative_int(self.rule_index, "rule_index")
            if self.rule_index == 0:
                raise ValueError("rule_index must be positive")
        weight = self.relevance_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0.0 <= weight <= 1.0
        ):
            raise ValueError("relevance_weight must be finite and lie in [0, 1]")
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a boolean")
        object.__setattr__(self, "relevance_weight", float(weight))


@dataclass(frozen=True, slots=True)
class QueryAction:
    """A finite, duplicate-free P1 acquisition action."""

    round_index: int
    targets: tuple[QueryTarget, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.round_index, "round_index")
        targets = tuple(self.targets)
        if any(not isinstance(target, QueryTarget) for target in targets):
            raise TypeError("targets must contain QueryTarget instances")
        keys = [target.key for target in targets]
        if len(set(keys)) != len(keys):
            raise ValueError("query targets must be unique")
        rule_indices = [
            target.rule_index for target in targets if target.rule_index is not None
        ]
        if len(set(rule_indices)) != len(rule_indices):
            raise ValueError("concrete rule indices must be unique")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class AcquisitionContext:
    """Read-only information available when selecting a training action."""

    round_index: int
    query_budget: int
    candidates: tuple[QueryTarget, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.round_index, "round_index")
        _nonnegative_int(self.query_budget, "query_budget")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, QueryTarget) for candidate in candidates):
            raise TypeError("candidates must contain QueryTarget instances")
        keys = [candidate.key for candidate in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate keys must be unique")
        object.__setattr__(self, "candidates", candidates)

    def validate_action(self, action: QueryAction) -> None:
        """Reject actions that do not belong to this bounded context."""

        if not isinstance(action, QueryAction):
            raise TypeError("action must be a QueryAction")
        if action.round_index != self.round_index:
            raise ValueError("action round does not match acquisition context")
        if len(action.targets) > self.query_budget:
            raise ValueError("action exceeds query budget")
        candidates = {candidate.key: candidate for candidate in self.candidates}
        if any(candidates.get(target.key) != target for target in action.targets):
            raise ValueError("action contains a target not exposed by the context")


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Ordered P1 observations bound to the action that generated them."""

    action: QueryAction
    observations: tuple[int, ...]
    cosmetic_observations: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, QueryAction):
            raise TypeError("action must be a QueryAction")
        observations = tuple(self.observations)
        if len(observations) != len(self.action.targets):
            raise ValueError("one observation is required for each query target")
        for observation in observations:
            if isinstance(observation, bool) or not isinstance(observation, int):
                raise TypeError("observations must be integers")
            if observation < 1:
                raise ValueError("observations must be positive labels")
        cosmetics = (
            (None,) * len(observations)
            if self.cosmetic_observations is None
            else tuple(self.cosmetic_observations)
        )
        if len(cosmetics) != len(observations):
            raise ValueError("cosmetic observations must align with observations")
        for cosmetic in cosmetics:
            if cosmetic is not None and (
                isinstance(cosmetic, bool)
                or not isinstance(cosmetic, int)
                or cosmetic < 0
            ):
                raise ValueError(
                    "cosmetic observations must be nonnegative integers or None"
                )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "cosmetic_observations", cosmetics)

    @property
    def items(self) -> tuple[tuple[QueryTarget, int], ...]:
        return tuple(zip(self.action.targets, self.observations, strict=True))

    @property
    def full_items(
        self,
    ) -> tuple[tuple[QueryTarget, int, int | None], ...]:
        return tuple(
            zip(
                self.action.targets,
                self.observations,
                self.cosmetic_observations,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class AgentCheckpoint:
    """Immutable acquisition state and behavioral deployment snapshot."""

    round_index: int
    deployment: DeploymentAction
    posterior_snapshots: Mapping[TargetKey, Sequence[float]]
    query_counts: Mapping[TargetKey, int]

    def __post_init__(self) -> None:
        _nonnegative_int(self.round_index, "round_index")
        if not isinstance(self.deployment, DeploymentAction):
            raise TypeError("deployment must be a DeploymentAction")
        object.__setattr__(
            self,
            "posterior_snapshots",
            _freeze_posteriors(self.posterior_snapshots),
        )
        object.__setattr__(self, "query_counts", _freeze_counts(self.query_counts))


@runtime_checkable
class SymbolicAgent(Protocol):
    """An agent whose acquisition, observation, and deployment are distinct."""

    capabilities: CapabilityManifest

    def select_train_action(self, context: AcquisitionContext) -> QueryAction: ...

    def observe(self, batch: ObservationBatch) -> None: ...

    def checkpoint(self) -> AgentCheckpoint: ...

    def deployment(self) -> DeploymentAction: ...
