"""Replay-derived information telemetry for symbolic pilot runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from infinite_rulebook.environments.redundant import cyclic_surface_map
from infinite_rulebook.feedback.qary import QarySymmetricChannel
from infinite_rulebook.information import (
    InformationBreakdown,
    InformationCategory,
    InformationLedger,
    LatentAxis,
    PosteriorBlock,
    SurfaceDependency,
)
from infinite_rulebook.orchestration.config import EnvironmentKind
from infinite_rulebook.posteriors import CategoricalPosterior


class ObservationNamespace(StrEnum):
    """Scientific role of one persisted query result."""

    REWARD = "reward"
    TRIVIA = "trivia"
    COSMETIC = "cosmetic"


@dataclass(frozen=True, slots=True)
class ReplayQueryObservation:
    """Immutable observation record sufficient for deterministic replay."""

    round_index: int
    query_ordinal: int
    namespace: ObservationNamespace
    rule_index: int
    value: int

    def __post_init__(self) -> None:
        for name in ("round_index", "query_ordinal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.namespace, ObservationNamespace):
            raise TypeError("namespace must be an ObservationNamespace")
        if (
            isinstance(self.rule_index, bool)
            or not isinstance(self.rule_index, int)
            or self.rule_index < 1
        ):
            raise ValueError("rule_index must be a positive integer")
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("value must be an integer")
        if self.namespace is ObservationNamespace.COSMETIC:
            if self.value < 0:
                raise ValueError("cosmetic values must be nonnegative")
        elif self.value < 1:
            raise ValueError("symbolic values must be positive")

    @property
    def event_identity(self) -> tuple[int, int, ObservationNamespace, int]:
        return (
            self.round_index,
            self.query_ordinal,
            self.namespace,
            self.rule_index,
        )


@dataclass(slots=True)
class _LatentEvidence:
    category: InformationCategory
    counts: list[int]
    surfaces: set[str]


def _latent_coordinate(
    observation: ReplayQueryObservation,
    *,
    environment: EnvironmentKind,
    q: int,
    core_dimensions: int | None,
) -> tuple[str, InformationCategory, int, str | None]:
    if observation.namespace is ObservationNamespace.TRIVIA:
        if environment is not EnvironmentKind.TRIVIA:
            raise ValueError("trivia observations require the TRIVIA environment")
        return (
            f"{environment.value}:trivia:{observation.rule_index}",
            InformationCategory.PERSISTENT_DISTRACTOR,
            observation.value,
            None,
        )

    if observation.namespace is not ObservationNamespace.REWARD:
        raise AssertionError("cosmetic observations must be filtered first")
    if environment is EnvironmentKind.RED_C:
        assert core_dimensions is not None
        component, offset = cyclic_surface_map(
            observation.rule_index, core_dimensions, q
        )
        return (
            f"{environment.value}:core:{component}",
            InformationCategory.SHARED_CORE,
            1 + ((observation.value - 1 - offset) % q),
            f"{environment.value}:surface:{observation.rule_index}",
        )
    if environment is EnvironmentKind.MIX and observation.rule_index % 2 == 0:
        assert core_dimensions is not None
        component, offset = cyclic_surface_map(
            observation.rule_index // 2, core_dimensions, q
        )
        return (
            f"{environment.value}:core:{component}",
            InformationCategory.SHARED_CORE,
            1 + ((observation.value - 1 - offset) % q),
            f"{environment.value}:surface:{observation.rule_index}",
        )
    primitive_index = (
        (observation.rule_index + 1) // 2
        if environment is EnvironmentKind.MIX
        else observation.rule_index
    )
    return (
        f"{environment.value}:reward:{primitive_index}",
        InformationCategory.REWARD_RELEVANT,
        observation.value,
        None,
    )


def information_ledger_from_observations(
    observations: Iterable[ReplayQueryObservation],
    *,
    environment: EnvironmentKind,
    q: int,
    epsilon: float,
    core_dimensions: int | None = None,
) -> InformationLedger:
    """Build an exact finite-closure ledger from persisted query results."""

    if not isinstance(environment, EnvironmentKind):
        raise TypeError("environment must be an EnvironmentKind")
    QarySymmetricChannel(q, epsilon)
    if environment in {EnvironmentKind.RED_C, EnvironmentKind.MIX} and (
        isinstance(core_dimensions, bool)
        or not isinstance(core_dimensions, int)
        or core_dimensions < 1
    ):
        raise ValueError("core_dimensions must be positive for RED-C and MIX telemetry")

    evidence: dict[str, _LatentEvidence] = {}
    identities: set[tuple[int, int, ObservationNamespace, int]] = set()
    for observation in observations:
        if not isinstance(observation, ReplayQueryObservation):
            raise TypeError("observations must contain ReplayQueryObservation records")
        if observation.event_identity in identities:
            raise ValueError(
                f"duplicate replay observation: {observation.event_identity!r}"
            )
        identities.add(observation.event_identity)
        if observation.namespace is ObservationNamespace.COSMETIC:
            if environment is not EnvironmentKind.ALEA:
                raise ValueError("cosmetic observations require the ALEA environment")
            continue
        if observation.value > q:
            raise ValueError(f"symbolic observation values must lie in 1..{q}")

        latent_id, category, value, surface = _latent_coordinate(
            observation,
            environment=environment,
            q=q,
            core_dimensions=core_dimensions,
        )
        latent = evidence.setdefault(
            latent_id,
            _LatentEvidence(category, [0] * q, set()),
        )
        latent.counts[value - 1] += 1
        if surface is not None:
            latent.surfaces.add(surface)

    blocks = []
    uniform_prior = (1.0 / q,) * q
    for latent_id, latent in sorted(evidence.items()):
        posterior = CategoricalPosterior.from_counts(
            latent.counts,
            epsilon=epsilon,
            prior=uniform_prior,
        )
        blocks.append(
            PosteriorBlock(
                block_id=latent_id,
                axes=(LatentAxis(latent_id, q, latent.category),),
                prior=uniform_prior,
                posterior=posterior.probabilities,
                surface_dependencies=tuple(
                    SurfaceDependency(surface, (latent_id,))
                    for surface in sorted(latent.surfaces)
                ),
            )
        )
    return InformationLedger.from_blocks(blocks)


def information_breakdown_from_observations(
    observations: Iterable[ReplayQueryObservation],
    *,
    environment: EnvironmentKind,
    q: int,
    epsilon: float,
    core_dimensions: int | None = None,
) -> InformationBreakdown:
    """Return the canonical breakdown for replay-derived telemetry."""

    return information_ledger_from_observations(
        observations,
        environment=environment,
        q=q,
        epsilon=epsilon,
        core_dimensions=core_dimensions,
    ).breakdown
