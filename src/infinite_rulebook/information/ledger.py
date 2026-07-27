"""Finite-closure information ledger.

The ledger evaluates posterior-to-prior KL directly.  It never subtracts
entropies, so a countably infinite untouched environment does not produce an
``infinity - infinity`` expression.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from numbers import Real

from infinite_rulebook.validation import (
    DiagnosticSeverity,
    ValidationDiagnostic,
    ValidationReport,
)


class InformationCategory(StrEnum):
    """Disjoint scientific roles for primitive latent variables."""

    REWARD_RELEVANT = "reward_relevant"
    SHARED_CORE = "shared_core"
    PERSISTENT_DISTRACTOR = "persistent_distractor"
    DYNAMIC_STATE = "dynamic_state"


_CATEGORY_ORDER = {
    InformationCategory.REWARD_RELEVANT: 0,
    InformationCategory.SHARED_CORE: 1,
    InformationCategory.PERSISTENT_DISTRACTOR: 2,
    InformationCategory.DYNAMIC_STATE: 3,
}


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return unicodedata.normalize("NFC", value)


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _distribution(name: str, values: Sequence[Real]) -> tuple[float, ...]:
    try:
        result = tuple(
            _finite_real(f"{name}[{index}]", value)
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise TypeError(f"{name} must be a finite sequence") from error
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0.0 for value in result):
        raise ValueError(f"{name} must be nonnegative")
    total = math.fsum(result)
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} probabilities must sum to one")
    return tuple(value / total for value in result)


def posterior_to_prior_kl(
    posterior: Sequence[Real],
    prior: Sequence[Real],
) -> float:
    """Return ``D_KL(posterior || prior)`` in nats.

    Zero posterior terms contribute zero.  Positive posterior mass outside the
    prior support returns positive infinity, as required by KL semantics.
    """

    p = _distribution("posterior", posterior)
    q = _distribution("prior", prior)
    if len(p) != len(q):
        raise ValueError("posterior and prior must have equal support")
    terms = []
    for posterior_mass, prior_mass in zip(p, q, strict=True):
        if posterior_mass == 0.0:
            continue
        if prior_mass == 0.0:
            return math.inf
        terms.append(posterior_mass * (math.log(posterior_mass) - math.log(prior_mass)))
    return max(0.0, math.fsum(terms))


@dataclass(frozen=True, slots=True)
class LatentAxis:
    """One unique primitive latent in a finite posterior closure."""

    latent_id: str
    cardinality: int
    category: InformationCategory

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_id", _identifier("latent_id", self.latent_id))
        if isinstance(self.cardinality, bool) or not isinstance(self.cardinality, int):
            raise TypeError("cardinality must be an integer")
        if self.cardinality < 2:
            raise ValueError("cardinality must be at least two")
        if not isinstance(self.category, InformationCategory):
            raise TypeError("category must be an InformationCategory")


@dataclass(frozen=True, slots=True)
class SurfaceDependency:
    """Audit-only mapping from a surface rule to its primitive closure."""

    surface_rule_id: str
    latent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_rule_id",
            _identifier("surface_rule_id", self.surface_rule_id),
        )
        latent_ids = tuple(
            sorted(_identifier("latent_id", value) for value in self.latent_ids)
        )
        if not latent_ids:
            raise ValueError("latent_ids must not be empty")
        if len(set(latent_ids)) != len(latent_ids):
            raise ValueError("a surface dependency cannot repeat a latent")
        object.__setattr__(self, "latent_ids", latent_ids)


def _state_count(axes: Sequence[LatentAxis]) -> int:
    return math.prod(axis.cardinality for axis in axes)


def _chain_contributions(
    axes: Sequence[LatentAxis],
    posterior: tuple[float, ...],
    prior: tuple[float, ...],
) -> tuple[float, ...]:
    """Attribute joint KL by the chain rule in the declared axis order."""

    posterior_prefix: list[defaultdict[tuple[int, ...], float]] = [
        defaultdict(float) for _ in range(len(axes) + 1)
    ]
    prior_prefix: list[defaultdict[tuple[int, ...], float]] = [
        defaultdict(float) for _ in range(len(axes) + 1)
    ]
    states = product(*(range(axis.cardinality) for axis in axes))
    for state, posterior_mass, prior_mass in zip(states, posterior, prior, strict=True):
        for length in range(len(axes) + 1):
            prefix = state[:length]
            posterior_prefix[length][prefix] += posterior_mass
            prior_prefix[length][prefix] += prior_mass

    contributions = []
    for length in range(1, len(axes) + 1):
        terms = []
        for prefix, posterior_mass in posterior_prefix[length].items():
            if posterior_mass == 0.0:
                continue
            parent = prefix[:-1]
            posterior_parent = posterior_prefix[length - 1][parent]
            prior_mass = prior_prefix[length][prefix]
            prior_parent = prior_prefix[length - 1][parent]
            if prior_mass == 0.0:
                return tuple(
                    (*contributions, math.inf, *((0.0,) * (len(axes) - length)))
                )
            terms.append(
                posterior_mass
                * (
                    math.log(posterior_mass)
                    - math.log(posterior_parent)
                    - math.log(prior_mass)
                    + math.log(prior_parent)
                )
            )
        contribution = math.fsum(terms)
        contributions.append(max(0.0, contribution))
    return tuple(contributions)


def _permute_joint_distribution(
    values: tuple[float, ...],
    axes: tuple[LatentAxis, ...],
    canonical_indices: tuple[int, ...],
) -> tuple[float, ...]:
    if canonical_indices == tuple(range(len(axes))):
        return values
    canonical_axes = tuple(axes[index] for index in canonical_indices)
    result = [0.0] * len(values)
    states = product(*(range(axis.cardinality) for axis in axes))
    for state, mass in zip(states, values, strict=True):
        new_state = tuple(state[index] for index in canonical_indices)
        flat_index = 0
        for value, axis in zip(new_state, canonical_axes, strict=True):
            flat_index = flat_index * axis.cardinality + value
        result[flat_index] = mass
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PosteriorBlock:
    """An exact joint posterior over a unique finite latent closure.

    State probabilities use lexicographic product order over ``axes``.  Surface
    dependencies are provenance only: adding aliases never adds information.
    Separate blocks assert independent prior and posterior factorization.
    """

    block_id: str
    axes: tuple[LatentAxis, ...]
    prior: tuple[float, ...]
    posterior: tuple[float, ...]
    surface_dependencies: tuple[SurfaceDependency, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _identifier("block_id", self.block_id))
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("a posterior block needs at least one latent axis")
        if any(not isinstance(axis, LatentAxis) for axis in axes):
            raise TypeError("axes must contain LatentAxis records")
        latent_ids = tuple(axis.latent_id for axis in axes)
        if len(set(latent_ids)) != len(latent_ids):
            raise ValueError("latent axes must have unique identifiers")
        expected = _state_count(axes)
        prior = _distribution("prior", self.prior)
        posterior = _distribution("posterior", self.posterior)
        if len(prior) != expected or len(posterior) != expected:
            raise ValueError(
                f"prior and posterior must each contain {expected} joint states"
            )
        canonical_indices = tuple(
            sorted(
                range(len(axes)),
                key=lambda index: (
                    _CATEGORY_ORDER[axes[index].category],
                    axes[index].latent_id,
                ),
            )
        )
        prior = _permute_joint_distribution(prior, axes, canonical_indices)
        posterior = _permute_joint_distribution(posterior, axes, canonical_indices)
        axes = tuple(axes[index] for index in canonical_indices)
        latent_ids = tuple(axis.latent_id for axis in axes)
        dependencies = tuple(self.surface_dependencies)
        if any(not isinstance(item, SurfaceDependency) for item in dependencies):
            raise TypeError(
                "surface_dependencies must contain SurfaceDependency records"
            )
        dependencies = tuple(
            sorted(dependencies, key=lambda item: item.surface_rule_id)
        )
        surface_ids = tuple(item.surface_rule_id for item in dependencies)
        if len(set(surface_ids)) != len(surface_ids):
            raise ValueError("surface rule identifiers must be unique within a block")
        known = set(latent_ids)
        for dependency in dependencies:
            unknown = set(dependency.latent_ids) - known
            if unknown:
                raise ValueError(
                    f"surface rule {dependency.surface_rule_id!r} references "
                    f"unknown latents: {sorted(unknown)!r}"
                )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "surface_dependencies", dependencies)

    @property
    def total_nats(self) -> float:
        return posterior_to_prior_kl(self.posterior, self.prior)

    @property
    def chain_contributions(self) -> tuple[float, ...]:
        return _chain_contributions(self.axes, self.posterior, self.prior)


@dataclass(frozen=True, slots=True)
class InformationBreakdown:
    """Immutable run-level Bayesian-surprise decomposition in nats."""

    reward_relevant_nats: float
    shared_core_nats: float
    persistent_distractor_nats: float
    dynamic_state_nats: float
    approximation_residual_nats: float
    total_acquired_nats: float

    def __post_init__(self) -> None:
        exact_names = (
            "reward_relevant_nats",
            "shared_core_nats",
            "persistent_distractor_nats",
            "dynamic_state_nats",
        )
        for name in exact_names:
            value = _finite_real(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        residual = _finite_real(
            "approximation_residual_nats", self.approximation_residual_nats
        )
        total = _finite_real("total_acquired_nats", self.total_acquired_nats)
        if total < 0.0:
            raise ValueError("total_acquired_nats cannot be negative")
        object.__setattr__(self, "approximation_residual_nats", residual)
        object.__setattr__(self, "total_acquired_nats", total)

    @property
    def reconciled_total_nats(self) -> float:
        return math.fsum(
            (
                self.reward_relevant_nats,
                self.shared_core_nats,
                self.persistent_distractor_nats,
                self.dynamic_state_nats,
                self.approximation_residual_nats,
            )
        )

    @property
    def relevant_nats(self) -> float:
        """All reward-relevant information without adding another ledger bucket."""

        return self.reward_relevant_nats + self.shared_core_nats

    def reconciles(self, tolerance: Real = 1e-12) -> bool:
        accuracy = _finite_real("tolerance", tolerance)
        if accuracy < 0.0:
            raise ValueError("tolerance must be nonnegative")
        return math.isclose(
            self.reconciled_total_nats,
            self.total_acquired_nats,
            rel_tol=accuracy,
            abs_tol=accuracy,
        )


@dataclass(frozen=True, slots=True)
class InformationLedger:
    """A collection of disjoint exact finite posterior closures."""

    blocks: tuple[PosteriorBlock, ...]
    approximation_residual_nats: float = 0.0

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if any(not isinstance(block, PosteriorBlock) for block in blocks):
            raise TypeError("blocks must contain PosteriorBlock records")
        blocks = tuple(sorted(blocks, key=lambda block: block.block_id))
        block_ids = tuple(block.block_id for block in blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("posterior block identifiers must be unique")
        latent_ids = [axis.latent_id for block in blocks for axis in block.axes]
        if len(set(latent_ids)) != len(latent_ids):
            raise ValueError("a primitive latent cannot occur in multiple blocks")
        surface_ids = [
            dependency.surface_rule_id
            for block in blocks
            for dependency in block.surface_dependencies
        ]
        if len(set(surface_ids)) != len(surface_ids):
            raise ValueError("a surface rule cannot occur in multiple blocks")
        residual = _finite_real(
            "approximation_residual_nats", self.approximation_residual_nats
        )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "approximation_residual_nats", residual)

    @classmethod
    def from_blocks(
        cls,
        blocks: Iterable[PosteriorBlock],
        *,
        approximation_residual_nats: Real = 0.0,
    ) -> InformationLedger:
        return cls(
            tuple(blocks),
            _finite_real("approximation_residual_nats", approximation_residual_nats),
        )

    @property
    def breakdown(self) -> InformationBreakdown:
        block_values = tuple(
            (block, block.total_nats, block.chain_contributions)
            for block in self.blocks
        )
        return self._breakdown_from(block_values)

    def _breakdown_from(
        self,
        block_values: tuple[tuple[PosteriorBlock, float, tuple[float, ...]], ...],
    ) -> InformationBreakdown:
        totals = {category: 0.0 for category in InformationCategory}
        for block, _, contributions in block_values:
            for axis, contribution in zip(block.axes, contributions, strict=True):
                totals[axis.category] += contribution
        exact_total = math.fsum(total for _, total, _ in block_values)
        total = exact_total + self.approximation_residual_nats
        return InformationBreakdown(
            reward_relevant_nats=totals[InformationCategory.REWARD_RELEVANT],
            shared_core_nats=totals[InformationCategory.SHARED_CORE],
            persistent_distractor_nats=totals[
                InformationCategory.PERSISTENT_DISTRACTOR
            ],
            dynamic_state_nats=totals[InformationCategory.DYNAMIC_STATE],
            approximation_residual_nats=self.approximation_residual_nats,
            total_acquired_nats=total,
        )

    def validate(self, tolerance: Real = 1e-12) -> ValidationReport:
        """Return stable diagnostics for cross-component scientific invariants."""

        accuracy = _finite_real("tolerance", tolerance)
        if accuracy < 0.0:
            raise ValueError("tolerance must be nonnegative")
        diagnostics = []
        support_violation = False
        block_values = tuple(
            (block, block.total_nats, block.chain_contributions)
            for block in self.blocks
        )
        for index, (_, total, contributions) in enumerate(block_values):
            if math.isinf(total):
                support_violation = True
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "ABSOLUTE_CONTINUITY_VIOLATION",
                        f"blocks[{index}]",
                        "posterior has positive mass outside prior support",
                    )
                )
                continue
            chain_total = math.fsum(contributions)
            if not math.isclose(
                chain_total,
                total,
                rel_tol=accuracy,
                abs_tol=accuracy,
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "CHAIN_RULE_TOTAL_MISMATCH",
                        f"blocks[{index}]",
                        f"chain total {chain_total!r} differs from joint KL {total!r}",
                    )
                )
        if support_violation:
            return ValidationReport(tuple(diagnostics))
        raw_total = (
            math.fsum(total for _, total, _ in block_values)
            + self.approximation_residual_nats
        )
        if raw_total < 0.0:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "NEGATIVE_TOTAL_INFORMATION",
                    "breakdown.total_acquired_nats",
                    f"total acquired information {raw_total!r} is negative",
                )
            )
            return ValidationReport(tuple(diagnostics))
        breakdown = self._breakdown_from(block_values)
        if not breakdown.reconciles(accuracy):
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "COMPONENT_TOTAL_MISMATCH",
                    "breakdown.total_acquired_nats",
                    "information buckets and residual do not reconcile",
                )
            )
        return ValidationReport(tuple(diagnostics))
