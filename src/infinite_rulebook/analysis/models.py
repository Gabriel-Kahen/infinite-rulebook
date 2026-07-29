"""Immutable models for registered experiment analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import pairwise

from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash


class AnalysisError(ValueError):
    """Raised when an analysis would violate its registered contract."""


class AnalysisPhase(StrEnum):
    PILOT = "pilot"
    CALIBRATION = "calibration"
    CONFIRMATORY = "confirmatory"


class Alternative(StrEnum):
    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two-sided"


class ContrastInterpretation(StrEnum):
    INFERENTIAL = "inferential"
    TELEMETRY_ONLY = "telemetry-only"


class Interpolation(StrEnum):
    LEFT_HOLD = "left-hold"
    LINEAR = "linear"


class MarginSource(StrEnum):
    PREREGISTERED = "preregistered"
    PILOT = "pilot"
    CALIBRATION = "calibration"


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class CertifiedFrontier:
    """Validated finite-curve summary used only after pooling reward."""

    semantic_hash: str
    zero_information_reward: float
    maximum_reward: float
    points: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not is_sha256(self.semantic_hash):
            raise ValueError("frontier semantic_hash must be a SHA-256 digest")
        zero = _finite("zero_information_reward", self.zero_information_reward)
        maximum = _finite("maximum_reward", self.maximum_reward)
        if zero >= maximum:
            raise ValueError("frontier reward endpoints are invalid")
        normalized = tuple(
            (
                _finite("frontier reward", reward),
                _finite("frontier lower bound", lower),
                _finite("frontier upper bound", upper),
            )
            for reward, lower, upper in self.points
        )
        if len(normalized) < 2:
            raise ValueError("frontier needs at least two points")
        rewards = tuple(point[0] for point in normalized)
        if rewards[0] != zero or rewards[-1] > maximum:
            raise ValueError("frontier points do not span the reward range")
        if any(left >= right for left, right in pairwise(rewards)):
            raise ValueError("frontier rewards must be strictly increasing")
        for _, lower, upper in normalized:
            if lower < 0.0 or lower > upper:
                raise ValueError("frontier information bounds are invalid")
        if normalized[0][1:] != (0.0, 0.0):
            raise ValueError("zero-information frontier point must be [0, 0]")
        for bound_index in (1, 2):
            values = tuple(point[bound_index] for point in normalized)
            if any(left > right for left, right in pairwise(values)):
                raise ValueError("frontier information bounds must be nondecreasing")
        endpoint_scale = max(abs(normalized[-1][0]), abs(maximum))
        if maximum - normalized[-1][0] > 64.0 * math.ulp(endpoint_scale):
            raise ValueError("frontier endpoint does not attain maximum reward")
        object.__setattr__(self, "zero_information_reward", zero)
        object.__setattr__(self, "maximum_reward", maximum)
        object.__setattr__(self, "points", normalized)

    def lookup(self, reward: float) -> tuple[float, float]:
        """Return the registered step-lower/chord-upper certified envelope."""

        value = _finite("reward", reward)
        if value <= self.zero_information_reward:
            return 0.0, 0.0
        if value > self.maximum_reward:
            return math.inf, math.inf
        if value > self.points[-1][0]:
            return self.points[-1][1], self.points[-1][2]
        for index in range(1, len(self.points)):
            right = self.points[index]
            if value > right[0]:
                continue
            left = self.points[index - 1]
            if value == right[0]:
                return right[1], right[2]
            weight = (value - left[0]) / (right[0] - left[0])
            upper = left[2] + weight * (right[2] - left[2])
            return left[1], upper
        raise AssertionError("frontier lookup did not locate a segment")


@dataclass(frozen=True, slots=True)
class CheckpointObservation:
    """One authenticated run checkpoint; never a population estimate."""

    run_hash: str
    content_hash: str
    phase: AnalysisPhase
    confirmatory_frozen: bool
    freeze_hash: str | None
    analysis_registration_hash: str | None
    condition_hash: str
    environment_kind: str
    agent_kind: str
    agent_hash: str
    environment_replica: int
    algorithm_replica: int
    round_index: int
    metrics: tuple[tuple[str, float], ...]
    semantic_hashes: tuple[tuple[str, str], ...]
    frontier: CertifiedFrontier
    run_settings_hash: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    cell_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_hash", "content_hash", "condition_hash", "agent_hash"):
            if not is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        if not isinstance(self.confirmatory_frozen, bool):
            raise TypeError("confirmatory_frozen must be a boolean")
        if self.freeze_hash is not None and not is_sha256(self.freeze_hash):
            raise ValueError("freeze_hash must be a SHA-256 digest or None")
        if self.analysis_registration_hash is not None and not is_sha256(
            self.analysis_registration_hash
        ):
            raise ValueError(
                "analysis_registration_hash must be a SHA-256 digest or None"
            )
        if self.phase is AnalysisPhase.CONFIRMATORY:
            if (
                not self.confirmatory_frozen
                or self.freeze_hash is None
                or self.analysis_registration_hash is None
            ):
                raise AnalysisError(
                    "confirmatory checkpoints require freeze and registration hashes"
                )
        elif (
            self.confirmatory_frozen
            or self.freeze_hash is not None
            or self.analysis_registration_hash is not None
        ):
            raise AnalysisError(
                "pilot/calibration checkpoints cannot carry confirmatory bindings"
            )
        _identifier("environment_kind", self.environment_kind)
        _identifier("agent_kind", self.agent_kind)
        for name in ("environment_replica", "algorithm_replica", "round_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        metrics = tuple(sorted(self.metrics))
        if not metrics or len({name for name, _ in metrics}) != len(metrics):
            raise ValueError("metrics must be nonempty with unique names")
        for name, value in metrics:
            _identifier("metric name", name)
            _finite(name, value)
        semantics = tuple(sorted(self.semantic_hashes))
        if len({name for name, _ in semantics}) != len(semantics):
            raise ValueError("semantic hash names must be unique")
        if any(not is_sha256(value) for _, value in semantics):
            raise ValueError("semantic hashes must be SHA-256 digests")
        if dict(semantics).get("frontier") != self.frontier.semantic_hash:
            raise ValueError("checkpoint and frontier semantic hashes differ")
        provenance = tuple(sorted(self.provenance))
        if self.run_settings_hash is None:
            if provenance:
                raise AnalysisError(
                    "checkpoint provenance requires a run-settings hash"
                )
        elif not is_sha256(self.run_settings_hash):
            raise ValueError("run_settings_hash must be a SHA-256 digest or None")
        if self.cell_hash is not None and not is_sha256(self.cell_hash):
            raise ValueError("cell_hash must be a SHA-256 digest or None")
        if provenance:
            if len({name for name, _ in provenance}) != len(provenance):
                raise ValueError("provenance fields must be unique")
            if any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
                for name, value in provenance
            ):
                raise ValueError("provenance fields must be nonempty strings")
            for name in (
                "analysis_code_hash",
                "dependency_lock_hash",
                "dirty_tree_hash",
                "environment_digest",
            ):
                if not is_sha256(dict(provenance).get(name)):
                    raise ValueError(f"provenance {name} must be a SHA-256 digest")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "semantic_hashes", semantics)
        object.__setattr__(self, "provenance", provenance)

    @property
    def pair_key(self) -> tuple[int, int]:
        return self.environment_replica, self.algorithm_replica

    def metric(self, name: str) -> float:
        try:
            return dict(self.metrics)[name]
        except KeyError as error:
            raise AnalysisError(
                f"checkpoint does not contain metric {name!r}"
            ) from error


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    observations: tuple[CheckpointObservation, ...]
    _scientific_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        observations = tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.condition_hash,
                    item.agent_hash,
                    item.agent_kind,
                    item.round_index,
                    item.pair_key,
                    item.run_hash,
                ),
            )
        )
        if not observations:
            raise AnalysisError("analysis dataset is empty")
        first = observations[0]
        phase = first.phase
        seal = first.freeze_hash, first.analysis_registration_hash
        binding = first.run_settings_hash, first.provenance
        rounds_by_run: dict[str, set[int]] = {}
        for item in observations:
            if item.phase is not phase:
                raise AnalysisError(
                    "analysis cannot mix pilot, calibration, and confirmatory data"
                )
            if (
                phase is AnalysisPhase.CONFIRMATORY
                and (
                    item.freeze_hash,
                    item.analysis_registration_hash,
                )
                != seal
            ):
                raise AnalysisError(
                    "confirmatory dataset mixes freeze or registration hashes"
                )
            if (item.run_settings_hash, item.provenance) != binding:
                raise AnalysisError(
                    "analysis dataset mixes run settings or scientific provenance"
                )
            observed_rounds = rounds_by_run.setdefault(item.run_hash, set())
            if item.round_index in observed_rounds:
                raise AnalysisError("analysis dataset contains duplicate checkpoints")
            observed_rounds.add(item.round_index)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(
            self,
            "_scientific_hash",
            scientific_hash(observations, domain="analysis-dataset"),
        )

    @property
    def phase(self) -> AnalysisPhase:
        return self.observations[0].phase

    @property
    def scientific_hash(self) -> str:
        return self._scientific_hash

    @property
    def run_settings_hash(self) -> str | None:
        return self.observations[0].run_settings_hash

    @property
    def provenance(self) -> tuple[tuple[str, str], ...]:
        return self.observations[0].provenance


@dataclass(frozen=True, slots=True)
class GroupSelector:
    environment_kind: str | None = None
    agent_kind: str | None = None
    condition_hash: str | None = None
    agent_hash: str | None = None

    def __post_init__(self) -> None:
        if self.environment_kind is not None:
            _identifier("environment_kind", self.environment_kind)
        if self.agent_kind is not None:
            _identifier("agent_kind", self.agent_kind)
        if self.condition_hash is not None and not is_sha256(self.condition_hash):
            raise ValueError("condition_hash must be a SHA-256 digest or None")
        if self.agent_hash is not None and not is_sha256(self.agent_hash):
            raise ValueError("agent_hash must be a SHA-256 digest or None")
        if (
            self.environment_kind is None
            and self.agent_kind is None
            and self.condition_hash is None
            and self.agent_hash is None
        ):
            raise ValueError("selector must constrain at least one field")

    def matches(self, item: CheckpointObservation) -> bool:
        return (
            (
                self.environment_kind is None
                or self.environment_kind == item.environment_kind
            )
            and (self.agent_kind is None or self.agent_kind == item.agent_kind)
            and (
                self.condition_hash is None
                or self.condition_hash == item.condition_hash
            )
            and (self.agent_hash is None or self.agent_hash == item.agent_hash)
        )

    @property
    def label(self) -> str:
        parts = (
            self.environment_kind or "*",
            self.agent_kind or "*",
            self.condition_hash[:12] if self.condition_hash else "*",
            self.agent_hash[:12] if self.agent_hash else "*",
        )
        return "/".join(parts)


@dataclass(frozen=True, slots=True, order=True)
class ExpectedGroup:
    """Exact frozen inventory for one condition/agent analysis group."""

    condition_hash: str
    agent_hash: str
    environment_kind: str
    agent_kind: str
    checkpoints: tuple[int, ...]
    environment_replicas: int
    algorithm_replicas: int

    def __post_init__(self) -> None:
        for name in ("condition_hash", "agent_hash"):
            if not is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        _identifier("environment_kind", self.environment_kind)
        _identifier("agent_kind", self.agent_kind)
        if (
            not isinstance(self.checkpoints, tuple)
            or not self.checkpoints
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.checkpoints
            )
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
        ):
            raise ValueError(
                "checkpoints must be a nonempty sorted tuple of unique "
                "nonnegative integers"
            )
        for name in ("environment_replicas", "algorithm_replicas"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def identities(
        self,
    ) -> tuple[tuple[str, str, str, str, int, int, int], ...]:
        return tuple(
            (
                self.condition_hash,
                self.agent_hash,
                self.environment_kind,
                self.agent_kind,
                environment,
                algorithm,
                checkpoint,
            )
            for environment in range(self.environment_replicas)
            for algorithm in range(self.algorithm_replicas)
            for checkpoint in self.checkpoints
        )


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    name: str
    metric: str
    left: GroupSelector
    right: GroupSelector
    checkpoint: int
    alternative: Alternative = Alternative.GREATER
    null_margin: float = 0.0
    required_equivalence_gates: tuple[str, ...] = ()
    interpretation: ContrastInterpretation = ContrastInterpretation.INFERENTIAL

    def __post_init__(self) -> None:
        _identifier("contrast name", self.name)
        _identifier("metric", self.metric)
        if not isinstance(self.left, GroupSelector) or not isinstance(
            self.right,
            GroupSelector,
        ):
            raise TypeError("contrast sides must be GroupSelector values")
        if isinstance(self.checkpoint, bool) or not isinstance(self.checkpoint, int):
            raise TypeError("checkpoint must be an integer")
        if self.checkpoint < 0:
            raise ValueError("checkpoint must be nonnegative")
        if not isinstance(self.alternative, Alternative):
            raise TypeError("alternative must be an Alternative")
        if not isinstance(self.interpretation, ContrastInterpretation):
            raise TypeError("interpretation must be a ContrastInterpretation")
        object.__setattr__(
            self, "null_margin", _finite("null_margin", self.null_margin)
        )
        gates = self.required_equivalence_gates
        if (
            not isinstance(gates, tuple)
            or tuple(sorted(set(gates))) != gates
            or any(not isinstance(gate, str) or not gate.strip() for gate in gates)
        ):
            raise ValueError(
                "required_equivalence_gates must be a sorted tuple of unique names"
            )


@dataclass(frozen=True, slots=True)
class EquivalenceSpec:
    name: str
    metric: str
    left: GroupSelector
    right: GroupSelector
    checkpoint: int
    margin: float
    margin_source: MarginSource
    margin_provenance_hash: str

    def __post_init__(self) -> None:
        _identifier("equivalence name", self.name)
        _identifier("metric", self.metric)
        if not isinstance(self.left, GroupSelector) or not isinstance(
            self.right,
            GroupSelector,
        ):
            raise TypeError("equivalence sides must be GroupSelector values")
        if isinstance(self.checkpoint, bool) or not isinstance(self.checkpoint, int):
            raise TypeError("checkpoint must be an integer")
        if self.checkpoint < 0:
            raise ValueError("checkpoint must be nonnegative")
        margin = _finite("margin", self.margin)
        if margin <= 0.0:
            raise ValueError("equivalence margin must be positive")
        if not isinstance(self.margin_source, MarginSource):
            raise TypeError("margin_source must be a MarginSource")
        if not is_sha256(self.margin_provenance_hash):
            raise ValueError("margin_provenance_hash must be a SHA-256 digest")
        object.__setattr__(self, "margin", margin)


@dataclass(frozen=True, slots=True)
class ScalingSpec:
    name: str
    metric: str
    selector: GroupSelector
    horizon: int
    interpolation: Interpolation = Interpolation.LEFT_HOLD

    def __post_init__(self) -> None:
        _identifier("scaling name", self.name)
        _identifier("metric", self.metric)
        if not isinstance(self.selector, GroupSelector):
            raise TypeError("selector must be a GroupSelector")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation")


@dataclass(frozen=True, slots=True)
class AnalysisPlan:
    name: str
    phase: AnalysisPhase
    contrasts: tuple[ContrastSpec, ...] = ()
    equivalences: tuple[EquivalenceSpec, ...] = ()
    scalings: tuple[ScalingSpec, ...] = ()
    expected_groups: tuple[ExpectedGroup, ...] = ()
    family_alpha: float = 0.05
    interval_alpha: float = 0.05
    frozen: bool = False
    freeze_hash: str | None = None

    def __post_init__(self) -> None:
        _identifier("analysis plan name", self.name)
        if not isinstance(self.phase, AnalysisPhase):
            raise TypeError("phase must be an AnalysisPhase")
        if not isinstance(self.frozen, bool):
            raise TypeError("frozen must be a boolean")
        expected_types = (
            (self.contrasts, ContrastSpec, "contrasts"),
            (self.equivalences, EquivalenceSpec, "equivalences"),
            (self.scalings, ScalingSpec, "scalings"),
            (self.expected_groups, ExpectedGroup, "expected_groups"),
        )
        for values, expected, name in expected_types:
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected) for item in values
            ):
                raise TypeError(f"{name} must be a tuple of {expected.__name__}")
        for name in ("family_alpha", "interval_alpha"):
            value = _finite(name, getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
            object.__setattr__(self, name, value)
        names = [
            item.name
            for collection in (self.contrasts, self.equivalences, self.scalings)
            for item in collection
        ]
        if len(set(names)) != len(names):
            raise ValueError("registered analysis names must be unique")
        equivalence_names = {item.name for item in self.equivalences}
        unknown_dependencies = {
            gate
            for contrast in self.contrasts
            for gate in contrast.required_equivalence_gates
            if gate not in equivalence_names
        }
        if unknown_dependencies:
            raise ValueError(
                "contrast dependencies name unregistered equivalence gates: "
                f"{sorted(unknown_dependencies)}"
            )
        if tuple(sorted(self.expected_groups)) != self.expected_groups:
            raise ValueError("expected_groups must be in canonical sorted order")
        group_keys = tuple(
            (item.condition_hash, item.agent_hash) for item in self.expected_groups
        )
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("expected_groups must have unique condition/agent hashes")
        if self.phase is AnalysisPhase.CONFIRMATORY:
            if not self.frozen or not is_sha256(self.freeze_hash):
                raise AnalysisError(
                    "confirmatory analysis requires a frozen external freeze hash"
                )
            if not self.expected_groups:
                raise AnalysisError(
                    "confirmatory analysis requires a frozen expected run inventory"
                )
        elif self.frozen or self.freeze_hash is not None:
            raise AnalysisError("only confirmatory plans may carry a freeze seal")

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(self, domain="analysis-plan")

    @property
    def registration_hash(self) -> str:
        """Hash registrations without creating a cycle through the freeze seal."""

        registered = (
            replace(self, freeze_hash="0" * 64)
            if self.phase is AnalysisPhase.CONFIRMATORY
            else self
        )
        return scientific_hash(registered, domain="analysis-registration")


__all__ = [
    "Alternative",
    "AnalysisDataset",
    "AnalysisError",
    "AnalysisPhase",
    "AnalysisPlan",
    "CertifiedFrontier",
    "CheckpointObservation",
    "ContrastInterpretation",
    "ContrastSpec",
    "EquivalenceSpec",
    "ExpectedGroup",
    "GroupSelector",
    "Interpolation",
    "MarginSource",
    "ScalingSpec",
]
