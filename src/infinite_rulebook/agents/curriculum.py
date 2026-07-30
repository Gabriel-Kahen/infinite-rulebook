"""Deterministic target-expansion curricula for later, non-registered studies."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from infinite_rulebook.agents.objectives import expected_entropy_reduction
from infinite_rulebook.agents.protocols import (
    AcquisitionContext,
    CapabilityManifest,
    QueryAction,
    QueryTarget,
    TargetKey,
)
from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.posteriors.categorical import CategoricalPosterior


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if result == 0.0:
        result = 0.0
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _name(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{name} must be nonempty without surrounding whitespace")
    return normalized


def _digest(value: object, name: str) -> str:
    digest = _name(value, name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _information_sum(values: tuple[float, ...]) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("planned information total must be finite") from error
    if not math.isfinite(result):
        raise ValueError("planned information total must be finite")
    return result


def _within_information_limit(
    values: tuple[float, ...],
    limit: float,
) -> bool:
    try:
        residual = math.fsum((*values, -limit))
    except OverflowError:
        return False
    return math.isfinite(residual) and residual <= 0.0


class CurriculumEvidence(StrEnum):
    """Privilege class of one curriculum estimate."""

    ORACLE = "oracle"
    POSTERIOR = "posterior"
    DISCOVERY = "discovery"


class CurriculumAction(StrEnum):
    """One bounded curriculum transition."""

    HOLD = "hold"
    EXPAND = "expand"
    PROBE = "probe"


class CurriculumReason(StrEnum):
    """Stable audit reason for a curriculum decision."""

    EXPANSION_SELECTED = "expansion-selected"
    FRONTIER_GAP_OPEN = "frontier-gap-open"
    RESIDUAL_VALUE_OPEN = "residual-value-open"
    UNCERTAINTY_TOO_WIDE = "uncertainty-too-wide"
    DISCOVERY_PROBE = "discovery-probe"
    INSUFFICIENT_DISCOVERY_EVIDENCE = "insufficient-discovery-evidence"
    NO_POSITIVE_VALUE_PER_NAT = "no-positive-value-per-nat"
    NO_ELIGIBLE_TARGET = "no-eligible-target"
    BUDGET_OR_SUPPORT_LIMIT = "budget-or-support-limit"


class CurriculumDeploymentRewardContract(StrEnum):
    """Executable deployment-reward contract for an estimand."""

    FACTORIZED_ADDITIVE_V1 = "factorized-query-agent-additive-v1"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class CurriculumEstimand:
    """Pinned reward, information, uncertainty, and data-split semantics."""

    name: str
    reward_spec_semantic_hash: str
    deployment_reward_contract: CurriculumDeploymentRewardContract
    reward_transform: str
    reward_aggregation: str
    reward_horizon: int
    information_variable: str
    information_conditioning: str
    confidence_method: str
    confidence_level: float
    confidence_family: str
    data_split: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.deployment_reward_contract,
            CurriculumDeploymentRewardContract,
        ):
            raise TypeError(
                "deployment_reward_contract must be a "
                "CurriculumDeploymentRewardContract"
            )
        for field_name in (
            "name",
            "reward_transform",
            "reward_aggregation",
            "information_variable",
            "information_conditioning",
            "confidence_method",
            "confidence_family",
            "data_split",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name),
            )
        if (
            self.deployment_reward_contract
            is CurriculumDeploymentRewardContract.FACTORIZED_ADDITIVE_V1
            and (
                self.reward_transform != "identity"
                or self.reward_aggregation
                not in {"fixed-horizon mean", "fixed-horizon sum"}
            )
        ):
            raise ValueError(
                "factorized additive deployment requires identity reward and "
                "fixed-horizon mean or sum aggregation"
            )
        object.__setattr__(
            self,
            "reward_spec_semantic_hash",
            _digest(
                self.reward_spec_semantic_hash,
                "reward_spec_semantic_hash",
            ),
        )
        _integer(self.reward_horizon, "reward_horizon", minimum=1)
        confidence_level = _real(
            self.confidence_level,
            "confidence_level",
            positive=True,
        )
        if confidence_level > 1.0:
            raise ValueError("confidence_level must not exceed one")
        object.__setattr__(self, "confidence_level", confidence_level)


@dataclass(frozen=True, slots=True)
class CurriculumEvidenceBasis:
    """Auditable source and exact active-support basis for one update."""

    estimand: CurriculumEstimand
    active_members: tuple[TargetKey, ...]
    source_scientific_payload_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.estimand, CurriculumEstimand):
            raise TypeError("estimand must be a CurriculumEstimand")
        active_members = tuple(self.active_members)
        if any(not isinstance(member, TargetKey) for member in active_members):
            raise TypeError("active_members must contain TargetKey instances")
        if len(set(active_members)) != len(active_members):
            raise ValueError("active_members must be unique")
        object.__setattr__(self, "active_members", tuple(sorted(active_members)))
        object.__setattr__(
            self,
            "source_scientific_payload_hash",
            _digest(
                self.source_scientific_payload_hash, "source_scientific_payload_hash"
            ),
        )


@dataclass(frozen=True, slots=True)
class CurriculumTarget:
    """A named target increment over stable query identities."""

    name: str
    members: tuple[TargetKey, ...]
    parent: str | None = None

    def __post_init__(self) -> None:
        name = _name(self.name, "target name")
        members = tuple(self.members)
        if not members or any(not isinstance(member, TargetKey) for member in members):
            raise ValueError("target members must be nonempty TargetKey instances")
        if len(set(members)) != len(members):
            raise ValueError("target members must be unique")
        parent = self.parent
        if self.parent is not None:
            parent = _name(self.parent, "target parent")
            if parent == name:
                raise ValueError("target parent must name a different target")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "members", tuple(sorted(members)))


@dataclass(frozen=True, slots=True)
class CurriculumCatalog:
    """Immutable target definitions, including optional hierarchy edges."""

    targets: tuple[CurriculumTarget, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not targets or any(
            not isinstance(target, CurriculumTarget) for target in targets
        ):
            raise ValueError(
                "catalog targets must be nonempty CurriculumTarget records"
            )
        by_name = {target.name: target for target in targets}
        if len(by_name) != len(targets):
            raise ValueError("catalog target names must be unique")
        for target in targets:
            if target.parent is not None and target.parent not in by_name:
                raise ValueError(f"unknown target parent: {target.parent}")
            seen = {target.name}
            parent = target.parent
            while parent is not None:
                if parent in seen:
                    raise ValueError("target hierarchy must be acyclic")
                seen.add(parent)
                parent = by_name[parent].parent
        descendants: dict[str, set[str]] = {target.name: set() for target in targets}
        for candidate in targets:
            parent = candidate.parent
            while parent is not None:
                descendants[parent].add(candidate.name)
                parent = by_name[parent].parent
        for target in targets:
            preempting_members = {
                member
                for candidate in targets
                if candidate.name != target.name
                and candidate.name not in descendants[target.name]
                for member in candidate.members
            }
            if not set(target.members) - preempting_members:
                raise ValueError(
                    "target overlap can eliminate a required support increment"
                )
        object.__setattr__(
            self,
            "targets",
            tuple(sorted(targets, key=lambda target: target.name)),
        )

    def target(self, name: str) -> CurriculumTarget:
        """Return one named target or fail closed."""

        name = _name(name, "target name")
        for target in self.targets:
            if target.name == name:
                return target
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class FrontierEstimate:
    """Current-target frontier or posterior value-of-information bounds."""

    evidence: CurriculumEvidence
    frontier_gap_upper: float
    residual_value_upper: float
    bound_width: float

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CurriculumEvidence):
            raise TypeError("frontier evidence must be CurriculumEvidence")
        for name in (
            "frontier_gap_upper",
            "residual_value_upper",
            "bound_width",
        ):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name, minimum=0.0),
            )


@dataclass(frozen=True, slots=True)
class CurriculumAssessment:
    """Conservative reward and information bounds for one target increment."""

    target: str
    evidence: CurriculumEvidence
    marginal_reward_lower: float
    marginal_reward_upper: float
    information_nats_lower: float
    information_nats_upper: float
    probe_information_nats: float = 0.0
    evidence_count: int = 0

    def __post_init__(self) -> None:
        target = _name(self.target, "assessment target")
        if not isinstance(self.evidence, CurriculumEvidence):
            raise TypeError("assessment evidence must be CurriculumEvidence")
        lower_reward = _real(self.marginal_reward_lower, "marginal_reward_lower")
        upper_reward = _real(self.marginal_reward_upper, "marginal_reward_upper")
        lower_information = _real(
            self.information_nats_lower,
            "information_nats_lower",
            positive=True,
        )
        upper_information = _real(
            self.information_nats_upper,
            "information_nats_upper",
            positive=True,
        )
        if upper_reward < lower_reward:
            raise ValueError("marginal reward bounds are reversed")
        if upper_information < lower_information:
            raise ValueError("information bounds are reversed")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "marginal_reward_lower", lower_reward)
        object.__setattr__(self, "marginal_reward_upper", upper_reward)
        object.__setattr__(self, "information_nats_lower", lower_information)
        object.__setattr__(self, "information_nats_upper", upper_information)
        object.__setattr__(
            self,
            "probe_information_nats",
            _real(
                self.probe_information_nats,
                "probe_information_nats",
                minimum=0.0,
            ),
        )
        _integer(self.evidence_count, "evidence_count")
        if not math.isfinite(self.conservative_value_per_nat):
            raise ValueError("conservative value per nat must be finite")

    @property
    def reward_bound_width(self) -> float:
        return self.marginal_reward_upper - self.marginal_reward_lower

    @property
    def conservative_value_per_nat(self) -> float:
        return self.marginal_reward_lower / self.information_nats_upper


@dataclass(frozen=True, slots=True)
class CurriculumUpdate:
    """One immutable estimator update presented to a curriculum."""

    round_index: int
    assessments: tuple[CurriculumAssessment, ...]
    evidence_basis: CurriculumEvidenceBasis
    frontier: FrontierEstimate | None = None

    def __post_init__(self) -> None:
        _integer(self.round_index, "round_index")
        assessments = tuple(self.assessments)
        if any(
            not isinstance(assessment, CurriculumAssessment)
            for assessment in assessments
        ):
            raise TypeError("assessments must contain CurriculumAssessment records")
        if len({assessment.target for assessment in assessments}) != len(assessments):
            raise ValueError("an update may assess each target at most once")
        if not isinstance(self.evidence_basis, CurriculumEvidenceBasis):
            raise TypeError("evidence_basis must be a CurriculumEvidenceBasis")
        if self.frontier is not None and not isinstance(
            self.frontier,
            FrontierEstimate,
        ):
            raise TypeError("frontier must be a FrontierEstimate or None")
        object.__setattr__(
            self,
            "assessments",
            tuple(sorted(assessments, key=lambda assessment: assessment.target)),
        )


@dataclass(frozen=True, slots=True)
class CurriculumLimits:
    """Hard support and conservative information limits."""

    maximum_support: int
    maximum_planned_information_nats: float

    def __post_init__(self) -> None:
        _integer(self.maximum_support, "maximum_support")
        object.__setattr__(
            self,
            "maximum_planned_information_nats",
            _real(
                self.maximum_planned_information_nats,
                "maximum_planned_information_nats",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    """One policy decision with stable structural audit fields."""

    round_index: int
    action: CurriculumAction
    reason: CurriculumReason
    target: str | None = None
    score: float | None = None
    planned_information_nats: float = 0.0
    support_added: int = 0

    def __post_init__(self) -> None:
        _integer(self.round_index, "round_index")
        if not isinstance(self.action, CurriculumAction):
            raise TypeError("action must be CurriculumAction")
        if not isinstance(self.reason, CurriculumReason):
            raise TypeError("reason must be CurriculumReason")
        information = _real(
            self.planned_information_nats,
            "planned_information_nats",
            minimum=0.0,
        )
        _integer(self.support_added, "support_added")
        if self.action is CurriculumAction.HOLD:
            if (
                self.target is not None
                or self.score is not None
                or information != 0.0
                or self.support_added != 0
            ):
                raise ValueError("hold decisions cannot carry a target or cost")
            if self.reason in {
                CurriculumReason.EXPANSION_SELECTED,
                CurriculumReason.DISCOVERY_PROBE,
            }:
                raise ValueError("hold decision reason is incompatible with its action")
        else:
            target = _name(self.target, "decision target")
            if self.score is None:
                raise ValueError("active decisions require a score")
            object.__setattr__(self, "score", _real(self.score, "score"))
            if information <= 0.0:
                raise ValueError("active decisions require positive information cost")
            if self.action is CurriculumAction.EXPAND and self.support_added < 1:
                raise ValueError("expansion decisions must add support")
            if self.action is CurriculumAction.PROBE and self.support_added != 0:
                raise ValueError("probe decisions cannot add persistent support")
            expected_reason = (
                CurriculumReason.EXPANSION_SELECTED
                if self.action is CurriculumAction.EXPAND
                else CurriculumReason.DISCOVERY_PROBE
            )
            if self.reason is not expected_reason:
                raise ValueError("decision reason is incompatible with its action")
            object.__setattr__(self, "target", target)
        object.__setattr__(self, "planned_information_nats", information)


@dataclass(frozen=True, slots=True)
class CurriculumState:
    """Immutable structural state after zero or more curriculum updates."""

    next_round: int = 0
    active_targets: tuple[str, ...] = ()
    active_members: tuple[TargetKey, ...] = ()
    planned_information_nats: float = 0.0
    probe_counts: tuple[tuple[str, int], ...] = ()
    evidence_counts: tuple[tuple[str, int], ...] = ()
    decisions: tuple[CurriculumDecision, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.next_round, "next_round")
        active_targets = tuple(
            _name(name, "active target name") for name in self.active_targets
        )
        if len(set(active_targets)) != len(active_targets):
            raise ValueError("active target names must be unique")
        if any(not isinstance(member, TargetKey) for member in self.active_members):
            raise TypeError("active members must be TargetKey instances")
        if len(set(self.active_members)) != len(self.active_members):
            raise ValueError("active members must be unique")
        object.__setattr__(
            self,
            "planned_information_nats",
            _real(
                self.planned_information_nats,
                "planned_information_nats",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "probe_counts",
            _validated_counts(self.probe_counts, "probe_counts"),
        )
        object.__setattr__(
            self,
            "evidence_counts",
            _validated_counts(self.evidence_counts, "evidence_counts"),
        )
        decisions = tuple(self.decisions)
        if any(not isinstance(item, CurriculumDecision) for item in decisions):
            raise TypeError("decisions must contain CurriculumDecision records")
        if len(decisions) != self.next_round or any(
            decision.round_index != round_index
            for round_index, decision in enumerate(decisions)
        ):
            raise ValueError("decision history must cover every completed round")
        reconstructed_information = _information_sum(
            tuple(decision.planned_information_nats for decision in decisions)
        )
        if reconstructed_information != self.planned_information_nats:
            raise ValueError("planned information must match decision history")
        object.__setattr__(self, "active_targets", tuple(sorted(active_targets)))
        object.__setattr__(self, "active_members", tuple(sorted(self.active_members)))
        object.__setattr__(self, "decisions", decisions)

    def probe_count(self, target: str) -> int:
        return dict(self.probe_counts).get(_name(target, "target name"), 0)

    def evidence_count(self, target: str) -> int:
        return dict(self.evidence_counts).get(_name(target, "target name"), 0)


def _validated_counts(
    values: tuple[tuple[str, int], ...],
    name: str,
) -> tuple[tuple[str, int], ...]:
    pairs = tuple(
        (_name(key, f"{name} key"), _integer(value, name)) for key, value in values
    )
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} keys must be unique")
    return tuple(sorted(pairs))


@runtime_checkable
class CurriculumPolicy(Protocol):
    """Pure curriculum decision rule over immutable state and estimates."""

    capabilities: CapabilityManifest

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: CurriculumState,
        update: CurriculumUpdate,
        limits: CurriculumLimits,
    ) -> CurriculumDecision: ...


_ORACLE_CAPABILITIES = CapabilityManifest(
    knows_relevance_mask=True,
    knows_coordinate_factorization=True,
    knows_target_hierarchy=True,
    knows_true_posterior_family=True,
    knows_exact_frontier=True,
    knows_reward_parameters=True,
)
_ESTIMATED_CAPABILITIES = CapabilityManifest(
    knows_target_hierarchy=True,
    knows_approximate_frontier=True,
    knows_reward_parameters=True,
)
_CANDIDATE_GROUP_CAPABILITIES = CapabilityManifest(
    knows_approximate_frontier=True,
    knows_reward_parameters=True,
)


def _validate_evidence_capabilities(
    capabilities: CapabilityManifest,
    update: CurriculumUpdate,
) -> None:
    evidence = {assessment.evidence for assessment in update.assessments}
    if update.frontier is not None:
        evidence.add(update.frontier.evidence)
    if CurriculumEvidence.ORACLE in evidence and not capabilities.knows_exact_frontier:
        raise ValueError("oracle evidence requires exact-frontier capability")
    if (
        evidence
        & {
            CurriculumEvidence.POSTERIOR,
            CurriculumEvidence.DISCOVERY,
        }
        and not capabilities.knows_approximate_frontier
    ):
        raise ValueError(
            "posterior or discovery evidence requires approximate-frontier capability"
        )
    if update.assessments and not capabilities.knows_reward_parameters:
        raise ValueError("reward assessments require reward-parameter capability")


@dataclass(frozen=True, slots=True)
class OracleFrontierPolicy:
    """Expand after the exact current-target frontier gap is sufficiently small."""

    frontier_gap_threshold: float
    minimum_value_per_nat: float = 0.0
    capabilities: CapabilityManifest = field(
        default=_ORACLE_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frontier_gap_threshold",
            _real(
                self.frontier_gap_threshold,
                "frontier_gap_threshold",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_value_per_nat",
            _real(
                self.minimum_value_per_nat,
                "minimum_value_per_nat",
                minimum=0.0,
            ),
        )

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: CurriculumState,
        update: CurriculumUpdate,
        limits: CurriculumLimits,
    ) -> CurriculumDecision:
        frontier = _frontier(update, CurriculumEvidence.ORACLE)
        assessments = _assessments(update, CurriculumEvidence.ORACLE)
        if frontier.bound_width != 0.0 or any(
            assessment.marginal_reward_lower != assessment.marginal_reward_upper
            or assessment.information_nats_lower != assessment.information_nats_upper
            for assessment in assessments
        ):
            raise ValueError("oracle estimates must be exact")
        if frontier.frontier_gap_upper > self.frontier_gap_threshold:
            return _hold(update, CurriculumReason.FRONTIER_GAP_OPEN)
        return _select_expansion(
            catalog,
            state,
            update,
            limits,
            assessments,
            self.minimum_value_per_nat,
        )


@dataclass(frozen=True, slots=True)
class EstimatedFrontierPolicy:
    """Follow posterior frontier bounds only after uncertainty and VPI are small."""

    residual_value_threshold: float
    bound_width_tolerance: float
    minimum_value_per_nat: float = 0.0
    capabilities: CapabilityManifest = field(
        default=_ESTIMATED_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "residual_value_threshold",
            "bound_width_tolerance",
            "minimum_value_per_nat",
        ):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name, minimum=0.0),
            )

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: CurriculumState,
        update: CurriculumUpdate,
        limits: CurriculumLimits,
    ) -> CurriculumDecision:
        frontier = _frontier(update, CurriculumEvidence.POSTERIOR)
        assessments = _assessments(update, CurriculumEvidence.POSTERIOR)
        if frontier.bound_width > self.bound_width_tolerance:
            return _hold(update, CurriculumReason.UNCERTAINTY_TOO_WIDE)
        if frontier.residual_value_upper > self.residual_value_threshold:
            return _hold(update, CurriculumReason.RESIDUAL_VALUE_OPEN)
        _require_complete_selection(catalog, state, assessments)
        if any(
            assessment.reward_bound_width > self.bound_width_tolerance
            for assessment in assessments
            if _structurally_eligible(catalog, state, assessment)
        ):
            return _hold(update, CurriculumReason.UNCERTAINTY_TOO_WIDE)
        return _select_expansion(
            catalog,
            state,
            update,
            limits,
            assessments,
            self.minimum_value_per_nat,
        )


@dataclass(frozen=True, slots=True)
class MarginalValuePerBitPolicy:
    """Implement the plan's value-per-bit concept in repository-standard nats."""

    minimum_value_per_nat: float = 0.0
    bound_width_tolerance: float | None = None
    capabilities: CapabilityManifest = field(
        default=_ESTIMATED_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_value_per_nat",
            _real(
                self.minimum_value_per_nat,
                "minimum_value_per_nat",
                minimum=0.0,
            ),
        )
        tolerance = self.bound_width_tolerance
        if tolerance is not None:
            tolerance = _real(tolerance, "bound_width_tolerance", minimum=0.0)
        object.__setattr__(self, "bound_width_tolerance", tolerance)

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: CurriculumState,
        update: CurriculumUpdate,
        limits: CurriculumLimits,
    ) -> CurriculumDecision:
        assessments = _assessments(update, CurriculumEvidence.POSTERIOR)
        _require_complete_selection(catalog, state, assessments)
        if any(
            self.bound_width_tolerance is not None
            and assessment.reward_bound_width > self.bound_width_tolerance
            for assessment in assessments
            if _structurally_eligible(catalog, state, assessment)
        ):
            return _hold(update, CurriculumReason.UNCERTAINTY_TOO_WIDE)
        return _select_expansion(
            catalog,
            state,
            update,
            limits,
            assessments,
            self.minimum_value_per_nat,
        )


@dataclass(frozen=True, slots=True)
class CandidateGroupDiscoveryPolicy:
    """Probe externally generated groups before conservative expansion."""

    minimum_evidence: int
    bound_width_tolerance: float
    minimum_value_per_nat: float = 0.0
    maximum_probes_per_target: int = 1
    capabilities: CapabilityManifest = field(
        default=_CANDIDATE_GROUP_CAPABILITIES,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _integer(self.minimum_evidence, "minimum_evidence", minimum=1)
        _integer(
            self.maximum_probes_per_target,
            "maximum_probes_per_target",
            minimum=1,
        )
        for name in ("bound_width_tolerance", "minimum_value_per_nat"):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name, minimum=0.0),
            )

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: CurriculumState,
        update: CurriculumUpdate,
        limits: CurriculumLimits,
    ) -> CurriculumDecision:
        assessments = _assessments(update, CurriculumEvidence.DISCOVERY)
        if any(target.parent is not None for target in catalog.targets):
            raise ValueError(
                "candidate-group discovery cannot consume a declared hierarchy"
            )
        _require_complete_discovery(catalog, state, assessments)
        inactive = tuple(
            assessment
            for assessment in assessments
            if assessment.target not in state.active_targets
        )
        unresolved = tuple(
            assessment
            for assessment in inactive
            if assessment.evidence_count < self.minimum_evidence
            or assessment.reward_bound_width > self.bound_width_tolerance
        )
        if unresolved:
            probes = tuple(
                assessment
                for assessment in unresolved
                if state.probe_count(assessment.target) < self.maximum_probes_per_target
                and _feasible(
                    catalog,
                    state,
                    assessment,
                    limits,
                    CurriculumAction.PROBE,
                )
            )
            if probes:
                selected = min(
                    probes,
                    key=lambda item: (
                        item.evidence_count >= self.minimum_evidence,
                        item.evidence_count,
                        state.probe_count(item.target),
                        item.target,
                    ),
                )
                return _active_decision(
                    catalog,
                    state,
                    update,
                    selected,
                    CurriculumAction.PROBE,
                    selected.conservative_value_per_nat,
                    CurriculumReason.DISCOVERY_PROBE,
                )
            if all(
                state.probe_count(item.target) >= self.maximum_probes_per_target
                for item in unresolved
            ):
                reason = CurriculumReason.INSUFFICIENT_DISCOVERY_EVIDENCE
            else:
                reason = CurriculumReason.BUDGET_OR_SUPPORT_LIMIT
            return _hold(update, reason)
        return _select_expansion(
            catalog,
            state,
            update,
            limits,
            inactive,
            self.minimum_value_per_nat,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CurriculumController:
    """Fail-closed mutable shell around a pure curriculum policy."""

    catalog: CurriculumCatalog
    policy: CurriculumPolicy
    limits: CurriculumLimits
    estimand: CurriculumEstimand
    state: CurriculumState = field(default_factory=CurriculumState)

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not isinstance(self.catalog, CurriculumCatalog):
            raise TypeError("catalog must be a CurriculumCatalog")
        if not isinstance(self.policy, CurriculumPolicy):
            raise TypeError("policy must satisfy CurriculumPolicy")
        if not isinstance(
            getattr(self.policy, "capabilities", None),
            CapabilityManifest,
        ):
            raise TypeError("curriculum policy must declare a CapabilityManifest")
        if not isinstance(self.limits, CurriculumLimits):
            raise TypeError("limits must be CurriculumLimits")
        if not isinstance(self.estimand, CurriculumEstimand):
            raise TypeError("estimand must be a CurriculumEstimand")
        if not isinstance(self.state, CurriculumState):
            raise TypeError("state must be CurriculumState")
        if any(target.parent is not None for target in self.catalog.targets) and not (
            self.policy.capabilities.knows_target_hierarchy
        ):
            raise ValueError(
                "hierarchical catalogs require target-hierarchy capability"
            )
        self._validate_state()

    @property
    def capabilities(self) -> CapabilityManifest:
        return self.policy.capabilities

    @property
    def queryable_members(self) -> tuple[TargetKey, ...]:
        """Persistent support, or only the latest transient discovery probe."""

        if self.state.decisions:
            decision = self.state.decisions[-1]
            if decision.action is CurriculumAction.PROBE:
                target_members = set(self.catalog.target(decision.target).members)
                return tuple(sorted(target_members - set(self.state.active_members)))
        return self.state.active_members

    def update(self, update: CurriculumUpdate) -> CurriculumDecision:
        """Validate one estimator update, decide, and advance immutable state."""

        if not isinstance(update, CurriculumUpdate):
            raise TypeError("update must be a CurriculumUpdate")
        self._validate_configuration()
        if update.round_index != self.state.next_round:
            raise ValueError("curriculum update round does not match state")
        if update.evidence_basis.estimand != self.estimand:
            raise ValueError("curriculum update estimand does not match controller")
        if update.evidence_basis.active_members != self.state.active_members:
            raise ValueError("curriculum evidence basis does not match active support")
        known = {target.name for target in self.catalog.targets}
        if any(item.target not in known for item in update.assessments):
            raise ValueError("curriculum update contains an unknown target")
        for item in update.assessments:
            if item.evidence_count < self.state.evidence_count(item.target):
                raise ValueError("curriculum evidence counts cannot decrease")
        capabilities = getattr(self.policy, "capabilities", None)
        if not isinstance(capabilities, CapabilityManifest):
            raise TypeError("curriculum policy must declare a CapabilityManifest")
        if any(target.parent is not None for target in self.catalog.targets) and not (
            capabilities.knows_target_hierarchy
        ):
            raise ValueError(
                "hierarchical catalogs require target-hierarchy capability"
            )
        _validate_evidence_capabilities(capabilities, update)
        decision = self.policy.decide(
            self.catalog,
            self.state,
            update,
            self.limits,
        )
        if not isinstance(decision, CurriculumDecision):
            raise TypeError("curriculum policy must return CurriculumDecision")
        object.__setattr__(self, "state", self._transition(update, decision))
        return decision

    def _transition(
        self,
        update: CurriculumUpdate,
        decision: CurriculumDecision,
    ) -> CurriculumState:
        if decision.round_index != update.round_index:
            raise ValueError("decision round does not match update")
        active_targets = set(self.state.active_targets)
        active_members = set(self.state.active_members)
        probes = dict(self.state.probe_counts)
        if decision.action is not CurriculumAction.HOLD:
            assessment = next(
                (item for item in update.assessments if item.target == decision.target),
                None,
            )
            if assessment is None:
                raise ValueError("active decision target was not assessed")
            if (
                decision.action is CurriculumAction.PROBE
                and assessment.evidence is not CurriculumEvidence.DISCOVERY
            ):
                raise ValueError("probe decisions require discovery evidence")
            if decision.action is CurriculumAction.EXPAND:
                _require_complete_selection(
                    self.catalog,
                    self.state,
                    update.assessments,
                )
            else:
                _require_complete_discovery(
                    self.catalog,
                    self.state,
                    update.assessments,
                )
            target = self.catalog.target(decision.target)
            new_members = set(target.members) - active_members
            expected_information = (
                assessment.probe_information_nats
                if decision.action is CurriculumAction.PROBE
                else assessment.information_nats_upper
            )
            expected_support = (
                0 if decision.action is CurriculumAction.PROBE else len(new_members)
            )
            if decision.planned_information_nats != expected_information:
                raise ValueError("decision information cost differs from assessment")
            if decision.support_added != expected_support:
                raise ValueError("decision support differs from target increment")
            if decision.score != assessment.conservative_value_per_nat:
                raise ValueError("decision score differs from assessment")
            if not _feasible(
                self.catalog,
                self.state,
                assessment,
                self.limits,
                decision.action,
            ):
                raise ValueError(
                    "decision exceeds hierarchy, support, or budget limits"
                )
            if decision.action is CurriculumAction.EXPAND:
                active_targets.add(target.name)
                active_members.update(new_members)
            else:
                probes[target.name] = probes.get(target.name, 0) + 1
        evidence = dict(self.state.evidence_counts)
        evidence.update(
            {item.target: item.evidence_count for item in update.assessments}
        )
        decisions = (*self.state.decisions, decision)
        decision_costs = tuple(item.planned_information_nats for item in decisions)
        if not _within_information_limit(
            decision_costs,
            self.limits.maximum_planned_information_nats,
        ):
            raise ValueError("decision exceeds the information limit")
        return CurriculumState(
            next_round=self.state.next_round + 1,
            active_targets=tuple(active_targets),
            active_members=tuple(active_members),
            planned_information_nats=_information_sum(decision_costs),
            probe_counts=tuple(probes.items()),
            evidence_counts=tuple(evidence.items()),
            decisions=decisions,
        )

    def _validate_state(self) -> None:
        known = {target.name for target in self.catalog.targets}
        active: set[str] = set()
        members: set[TargetKey] = set()
        probes: dict[str, int] = {}
        for decision in self.state.decisions:
            if decision.action is CurriculumAction.HOLD:
                continue
            if decision.target not in known:
                raise ValueError("decision history contains an unknown target")
            target = self.catalog.target(decision.target)
            if target.name in active:
                raise ValueError("decision history reuses an active target")
            if target.parent is not None and target.parent not in active:
                raise ValueError("decision history violates target hierarchy")
            new_members = set(target.members) - members
            if not new_members:
                raise ValueError("decision history contains an empty target increment")
            if len(members | set(target.members)) > self.limits.maximum_support:
                raise ValueError("decision history exceeds the support limit")
            if decision.action is CurriculumAction.EXPAND:
                if decision.support_added != len(new_members):
                    raise ValueError("decision history has an invalid support delta")
                active.add(target.name)
                members.update(new_members)
            else:
                probes[target.name] = probes.get(target.name, 0) + 1
        if active != set(self.state.active_targets):
            raise ValueError("active targets do not match decision history")
        if members != set(self.state.active_members):
            raise ValueError("active members do not match decision history")
        if dict(self.state.probe_counts) != probes:
            raise ValueError("probe counts do not match decision history")
        if isinstance(self.policy, CandidateGroupDiscoveryPolicy) and any(
            count > self.policy.maximum_probes_per_target for count in probes.values()
        ):
            raise ValueError("probe history exceeds the per-target limit")
        decision_costs = tuple(
            decision.planned_information_nats for decision in self.state.decisions
        )
        _information_sum(decision_costs)
        if not _within_information_limit(
            decision_costs,
            self.limits.maximum_planned_information_nats,
        ):
            raise ValueError("state exceeds the information limit")
        if any(
            key not in known
            for key, _ in (*self.state.probe_counts, *self.state.evidence_counts)
        ):
            raise ValueError("state counts contain targets outside the catalog")


@dataclass(frozen=True, slots=True, eq=False)
class CurriculumAcquisitionPolicy:
    """Expose the controller's bounded support as an information-gain policy."""

    controller: CurriculumController

    def __post_init__(self) -> None:
        if not isinstance(self.controller, CurriculumController):
            raise TypeError("controller must be a CurriculumController")
        if (
            self.controller.estimand.deployment_reward_contract
            is not CurriculumDeploymentRewardContract.FACTORIZED_ADDITIVE_V1
        ):
            raise ValueError(
                "CurriculumAcquisitionPolicy requires the factorized additive "
                "deployment reward contract"
            )

    @property
    def capabilities(self) -> CapabilityManifest:
        scheduler = self.controller.capabilities
        return CapabilityManifest(
            knows_relevance_mask=True,
            knows_coordinate_factorization=True,
            knows_latent_dependency_graph=scheduler.knows_latent_dependency_graph,
            knows_target_hierarchy=scheduler.knows_target_hierarchy,
            knows_true_posterior_family=True,
            knows_exact_frontier=scheduler.knows_exact_frontier,
            knows_approximate_frontier=scheduler.knows_approximate_frontier,
            knows_reward_parameters=True,
        )

    def update_curriculum(self, update: CurriculumUpdate) -> CurriculumDecision:
        return self.controller.update(update)

    def validate_action(self, action: QueryAction) -> None:
        """Require a transient probe round to query its selected candidate group."""

        if not isinstance(action, QueryAction):
            raise TypeError("action must be a QueryAction")
        if not self.controller.state.decisions:
            return
        decision = self.controller.state.decisions[-1]
        if action.round_index != decision.round_index:
            raise ValueError("action round does not match curriculum decision")
        allowed = set(self.controller.queryable_members)
        selected = {target.key for target in action.targets}
        if decision.action is not CurriculumAction.PROBE:
            if not selected <= allowed:
                raise ValueError("action contains support outside the curriculum")
            return
        if not selected or not selected <= allowed:
            raise ValueError("probe decision requires a selected candidate member")

    def validate_context(self, context: AcquisitionContext) -> None:
        """Require complete exposure of the current curriculum support."""

        if not isinstance(context, AcquisitionContext):
            raise TypeError("context must be an AcquisitionContext")
        candidates = {target.key: target for target in context.candidates}
        required = set(self.controller.queryable_members)
        missing = required - set(candidates)
        if missing:
            raise ValueError("candidate context omits curriculum members")
        if any(not candidates[key].persistent for key in required):
            raise ValueError("curriculum members must be exposed as persistent targets")

    def allows_deployment(self, key: TargetKey) -> bool:
        """Exclude transient probe-only information from deployment."""

        if not isinstance(key, TargetKey):
            raise TypeError("deployment key must be a TargetKey")
        return key in self.controller.state.active_members

    def score(
        self,
        target: QueryTarget,
        posterior: CategoricalPosterior,
        reward_spec: RewardSpec,
        round_index: int,
    ) -> float | None:
        if (
            semantic_hash(reward_spec)
            != self.controller.estimand.reward_spec_semantic_hash
        ):
            raise ValueError("execution reward spec does not match curriculum estimand")
        del round_index
        if target.key not in self.controller.queryable_members or not target.persistent:
            return None
        score = expected_entropy_reduction(posterior)
        if (
            self.controller.state.decisions
            and self.controller.state.decisions[-1].action is CurriculumAction.PROBE
            and score <= 0.0
        ):
            return math.ulp(0.0)
        return score


def _frontier(
    update: CurriculumUpdate,
    evidence: CurriculumEvidence,
) -> FrontierEstimate:
    if update.frontier is None:
        raise ValueError("frontier-following policies require a frontier estimate")
    if update.frontier.evidence is not evidence:
        raise ValueError(f"frontier evidence must be {evidence.value}")
    return update.frontier


def _assessments(
    update: CurriculumUpdate,
    evidence: CurriculumEvidence,
) -> tuple[CurriculumAssessment, ...]:
    if any(item.evidence is not evidence for item in update.assessments):
        raise ValueError(f"assessment evidence must be {evidence.value}")
    return update.assessments


def _hold(
    update: CurriculumUpdate,
    reason: CurriculumReason,
) -> CurriculumDecision:
    return CurriculumDecision(update.round_index, CurriculumAction.HOLD, reason)


def _new_members(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    assessment: CurriculumAssessment,
) -> set[TargetKey]:
    return set(catalog.target(assessment.target).members) - set(state.active_members)


def _structurally_eligible(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    assessment: CurriculumAssessment,
) -> bool:
    target = catalog.target(assessment.target)
    return (
        target.name not in state.active_targets
        and bool(_new_members(catalog, state, assessment))
        and (target.parent is None or target.parent in state.active_targets)
    )


def _feasible(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    assessment: CurriculumAssessment,
    limits: CurriculumLimits,
    action: CurriculumAction,
) -> bool:
    if action is CurriculumAction.HOLD or not _structurally_eligible(
        catalog,
        state,
        assessment,
    ):
        return False
    members = set(state.active_members) | set(catalog.target(assessment.target).members)
    information = (
        assessment.probe_information_nats
        if action is CurriculumAction.PROBE
        else assessment.information_nats_upper
    )
    proposed_costs = (
        *(decision.planned_information_nats for decision in state.decisions),
        information,
    )
    return (
        information > 0.0
        and len(members) <= limits.maximum_support
        and _within_information_limit(
            proposed_costs,
            limits.maximum_planned_information_nats,
        )
    )


def _select_expansion(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    update: CurriculumUpdate,
    limits: CurriculumLimits,
    assessments: tuple[CurriculumAssessment, ...],
    minimum_value_per_nat: float,
) -> CurriculumDecision:
    _require_complete_selection(catalog, state, assessments)
    positive = tuple(
        item
        for item in assessments
        if item.conservative_value_per_nat > minimum_value_per_nat
        and _structurally_eligible(catalog, state, item)
    )
    feasible = tuple(
        item
        for item in positive
        if _feasible(
            catalog,
            state,
            item,
            limits,
            CurriculumAction.EXPAND,
        )
    )
    if not feasible:
        if positive:
            reason = CurriculumReason.BUDGET_OR_SUPPORT_LIMIT
        elif any(_structurally_eligible(catalog, state, item) for item in assessments):
            reason = CurriculumReason.NO_POSITIVE_VALUE_PER_NAT
        else:
            reason = CurriculumReason.NO_ELIGIBLE_TARGET
        return _hold(update, reason)
    selected = min(
        feasible,
        key=lambda item: (-item.conservative_value_per_nat, item.target),
    )
    return _active_decision(
        catalog,
        state,
        update,
        selected,
        CurriculumAction.EXPAND,
        selected.conservative_value_per_nat,
        CurriculumReason.EXPANSION_SELECTED,
    )


def _require_complete_selection(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    assessments: tuple[CurriculumAssessment, ...],
) -> None:
    eligible_names = {
        target.name
        for target in catalog.targets
        if target.name not in state.active_targets
        and (target.parent is None or target.parent in state.active_targets)
        and bool(set(target.members) - set(state.active_members))
    }
    assessed_names = {assessment.target for assessment in assessments}
    if not eligible_names <= assessed_names:
        raise ValueError("selection updates must assess every eligible target")


def _require_complete_discovery(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    assessments: tuple[CurriculumAssessment, ...],
) -> None:
    inactive_names = {
        target.name
        for target in catalog.targets
        if target.name not in state.active_targets
    }
    assessed_inactive_names = {
        assessment.target
        for assessment in assessments
        if assessment.target not in state.active_targets
        and assessment.evidence is CurriculumEvidence.DISCOVERY
    }
    if assessed_inactive_names != inactive_names:
        raise ValueError("discovery updates must assess every inactive candidate group")


def _active_decision(
    catalog: CurriculumCatalog,
    state: CurriculumState,
    update: CurriculumUpdate,
    assessment: CurriculumAssessment,
    action: CurriculumAction,
    score: float,
    reason: CurriculumReason,
) -> CurriculumDecision:
    return CurriculumDecision(
        round_index=update.round_index,
        action=action,
        reason=reason,
        target=assessment.target,
        score=score,
        planned_information_nats=(
            assessment.probe_information_nats
            if action is CurriculumAction.PROBE
            else assessment.information_nats_upper
        ),
        support_added=(
            0
            if action is CurriculumAction.PROBE
            else len(_new_members(catalog, state, assessment))
        ),
    )
