"""Split-sample certified design assurance and diagnostic power simulation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from functools import cache
from numbers import Real
from typing import Self

from infinite_rulebook.analysis.models import Alternative
from infinite_rulebook.analysis.statistics import exact_sign_p_value, holm_adjust
from infinite_rulebook.core.rng import CounterRNG, Seed

DEFAULT_POWER_SIMULATIONS = 10_000
DEFAULT_SIMULATION_ERROR_ALPHA = 0.01
DEFAULT_DESIGN_CONFIDENCE_ALPHA = 0.01


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(name: str, value: Real) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _open_probability(name: str, value: Real) -> float:
    result = _finite(name, value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _alpha(value: Real) -> float:
    return _open_probability("alpha", value)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _environment_key(value: int | str) -> tuple[int, int | str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("environment_id must be an integer or string")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer environment_id values must be nonnegative")
        return 0, value
    if not value:
        raise ValueError("string environment_id values must not be empty")
    return 1, value


@dataclass(frozen=True, slots=True)
class EnvironmentCluster:
    """Paired algorithm differences, or one pre-averaged cluster difference."""

    environment_id: int | str
    algorithm_differences: tuple[float, ...]
    source_algorithm_replicas: int | None = None

    def __post_init__(self) -> None:
        _environment_key(self.environment_id)
        try:
            differences = tuple(
                _finite("algorithm difference", value)
                for value in self.algorithm_differences
            )
        except TypeError as error:
            raise TypeError(
                "algorithm_differences must be a sequence of real numbers"
            ) from error
        if not differences:
            raise ValueError("an environment cluster needs an algorithm difference")
        source_count = (
            len(differences)
            if self.source_algorithm_replicas is None
            else _positive_integer(
                "source_algorithm_replicas",
                self.source_algorithm_replicas,
            )
        )
        object.__setattr__(self, "algorithm_differences", differences)
        object.__setattr__(self, "source_algorithm_replicas", source_count)

    @property
    def paired_difference(self) -> float:
        """Average crossed algorithm replicas without inflating sample size."""

        return math.fsum(self.algorithm_differences) / len(self.algorithm_differences)


def _validated_clusters(value: object) -> tuple[EnvironmentCluster, ...]:
    try:
        clusters = tuple(
            sorted(
                value,  # type: ignore[arg-type]
                key=lambda cluster: _environment_key(cluster.environment_id),
            )
        )
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "clusters must contain EnvironmentCluster records; "
            "environment x algorithm cells may not be passed as flat replicates"
        ) from error
    if len(clusters) < 2:
        raise ValueError("power calibration needs at least two environments")
    if any(not isinstance(cluster, EnvironmentCluster) for cluster in clusters):
        raise TypeError("clusters must contain EnvironmentCluster records")
    environment_ids = tuple(cluster.environment_id for cluster in clusters)
    if len(set(environment_ids)) != len(environment_ids):
        raise ValueError("environment cluster identifiers must be unique")
    algorithm_counts = {cluster.source_algorithm_replicas for cluster in clusters}
    if len(algorithm_counts) != 1:
        raise ValueError("algorithm replicas must be fully crossed over environments")
    return clusters


@dataclass(frozen=True, slots=True)
class PowerHypothesis:
    """One registered location hypothesis with calibration residual clusters."""

    name: str
    clusters: tuple[EnvironmentCluster, ...]
    minimum_effect: float
    alternative: Alternative = Alternative.GREATER
    null_value: float = 0.0

    def __post_init__(self) -> None:
        _identifier("hypothesis name", self.name)
        if not isinstance(self.alternative, Alternative):
            raise TypeError("alternative must be an Alternative")
        clusters = _validated_clusters(self.clusters)

        null = _finite("null_value", self.null_value)
        effect = _finite("minimum_effect", self.minimum_effect)
        if self.alternative is Alternative.GREATER and effect <= 0.0:
            raise ValueError("a greater alternative requires a positive effect")
        if self.alternative is Alternative.LESS and effect >= 0.0:
            raise ValueError("a less alternative requires a negative effect")
        if self.alternative is Alternative.TWO_SIDED and effect == 0.0:
            raise ValueError("a two-sided alternative requires a nonzero effect")
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "null_value", null)
        object.__setattr__(self, "minimum_effect", effect)

    @classmethod
    def from_cluster_differences(
        cls,
        name: str,
        differences: tuple[float, ...],
        *,
        minimum_effect: float,
        alternative: Alternative = Alternative.GREATER,
        null_value: float = 0.0,
        algorithm_replicas_per_environment: int = 1,
    ) -> Self:
        """Build from values already averaged once per environment seed."""

        try:
            values = tuple(differences)
        except TypeError as error:
            raise TypeError("differences must be a sequence") from error
        return cls(
            name=name,
            clusters=tuple(
                EnvironmentCluster(
                    index,
                    (value,),
                    source_algorithm_replicas=algorithm_replicas_per_environment,
                )
                for index, value in enumerate(values)
            ),
            minimum_effect=minimum_effect,
            alternative=alternative,
            null_value=null_value,
        )

    @property
    def environment_ids(self) -> tuple[int | str, ...]:
        return tuple(cluster.environment_id for cluster in self.clusters)

    @property
    def algorithm_replicas(self) -> int:
        count = self.clusters[0].source_algorithm_replicas
        assert count is not None
        return count

    @property
    def cluster_differences(self) -> tuple[float, ...]:
        return tuple(cluster.paired_difference for cluster in self.clusters)

    @property
    def centered_residuals(self) -> tuple[float, ...]:
        values = self.cluster_differences
        center = float(statistics.median(values))
        return tuple(value - center for value in values)


@dataclass(frozen=True, slots=True)
class EquivalencePowerHypothesis:
    """Exact-sign TOST assurance plus a separate shifted-bootstrap diagnostic.

    Certified bounds target the stationary raw paired-difference population
    represented by the held-out calibration clusters. ``diagnostic_location``
    affects only the explicitly conditional location-shift bootstrap.
    """

    name: str
    clusters: tuple[EnvironmentCluster, ...]
    margin: float
    diagnostic_location: float = 0.0

    def __post_init__(self) -> None:
        _identifier("equivalence hypothesis name", self.name)
        clusters = _validated_clusters(self.clusters)
        margin = _finite("equivalence margin", self.margin)
        diagnostic_location = _finite(
            "diagnostic_location",
            self.diagnostic_location,
        )
        if margin <= 0.0:
            raise ValueError("equivalence margin must be positive")
        if not -margin < diagnostic_location < margin:
            raise ValueError("diagnostic_location must lie strictly inside the margin")
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "diagnostic_location", diagnostic_location)

    @classmethod
    def from_cluster_differences(
        cls,
        name: str,
        differences: tuple[float, ...],
        *,
        margin: float,
        diagnostic_location: float = 0.0,
        algorithm_replicas_per_environment: int = 1,
    ) -> Self:
        try:
            values = tuple(differences)
        except TypeError as error:
            raise TypeError("differences must be a sequence") from error
        return cls(
            name=name,
            clusters=tuple(
                EnvironmentCluster(
                    index,
                    (value,),
                    source_algorithm_replicas=algorithm_replicas_per_environment,
                )
                for index, value in enumerate(values)
            ),
            margin=margin,
            diagnostic_location=diagnostic_location,
        )

    @property
    def environment_ids(self) -> tuple[int | str, ...]:
        return tuple(cluster.environment_id for cluster in self.clusters)

    @property
    def algorithm_replicas(self) -> int:
        count = self.clusters[0].source_algorithm_replicas
        assert count is not None
        return count

    @property
    def cluster_differences(self) -> tuple[float, ...]:
        return tuple(cluster.paired_difference for cluster in self.clusters)

    @property
    def centered_residuals(self) -> tuple[float, ...]:
        values = self.cluster_differences
        center = float(statistics.median(values))
        return tuple(value - center for value in values)


@dataclass(frozen=True, slots=True)
class HypothesisPower:
    name: str
    rejections: int
    simulations: int
    favorable_sign_successes: int = 0
    favorable_sign_trials: int = 0
    favorable_sign_probability_lower_bound: float = 0.0
    certified_power_lower_bound: float = 0.0

    def __post_init__(self) -> None:
        _identifier("hypothesis name", self.name)
        simulations = _positive_integer("simulations", self.simulations)
        if (
            isinstance(self.rejections, bool)
            or not isinstance(self.rejections, int)
            or not 0 <= self.rejections <= simulations
        ):
            raise ValueError("rejections must lie in [0, simulations]")
        if (
            isinstance(self.favorable_sign_trials, bool)
            or not isinstance(self.favorable_sign_trials, int)
            or self.favorable_sign_trials < 0
            or isinstance(self.favorable_sign_successes, bool)
            or not isinstance(self.favorable_sign_successes, int)
            or not 0 <= self.favorable_sign_successes <= self.favorable_sign_trials
        ):
            raise ValueError("favorable sign counts are invalid")
        _probability(
            "favorable_sign_probability_lower_bound",
            self.favorable_sign_probability_lower_bound,
        )
        _probability(
            "certified_power_lower_bound",
            self.certified_power_lower_bound,
        )

    @property
    def power(self) -> float:
        return self.rejections / self.simulations


@dataclass(frozen=True, slots=True)
class EquivalencePowerBound:
    """Conditional diagnostic plus simultaneous certified TOST power bounds."""

    name: str
    rejections: int
    simulations: int
    lower_successes: int
    upper_successes: int
    favorable_sign_trials: int
    lower_probability_bound: float
    upper_probability_bound: float
    lower_test_power_bound: float
    upper_test_power_bound: float
    certified_power_lower_bound: float

    def __post_init__(self) -> None:
        _identifier("equivalence hypothesis name", self.name)
        simulations = _positive_integer("simulations", self.simulations)
        if (
            isinstance(self.rejections, bool)
            or not isinstance(self.rejections, int)
            or not 0 <= self.rejections <= simulations
        ):
            raise ValueError("equivalence rejections must lie in [0, simulations]")
        trials = _positive_integer(
            "favorable_sign_trials",
            self.favorable_sign_trials,
        )
        for name, value in (
            ("lower_successes", self.lower_successes),
            ("upper_successes", self.upper_successes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= trials
            ):
                raise ValueError(f"{name} must lie in [0, favorable_sign_trials]")
        for name, value in (
            ("lower_probability_bound", self.lower_probability_bound),
            ("upper_probability_bound", self.upper_probability_bound),
            ("lower_test_power_bound", self.lower_test_power_bound),
            ("upper_test_power_bound", self.upper_test_power_bound),
            ("certified_power_lower_bound", self.certified_power_lower_bound),
        ):
            _probability(name, value)

    @property
    def power(self) -> float:
        """Return the explicitly conditional shifted-bootstrap diagnostic."""

        return self.rejections / self.simulations


@dataclass(frozen=True, slots=True)
class EffectAdequacy:
    """Simultaneous order-statistic interval and preregistered effect gate."""

    name: str
    kind: str
    interval_lower: float
    interval_upper: float
    threshold_lower: float | None
    threshold_upper: float | None
    center_environment_count: int
    passes: bool

    def __post_init__(self) -> None:
        _identifier("effect adequacy name", self.name)
        if self.kind not in {"directional", "equivalence"}:
            raise ValueError("effect adequacy kind is invalid")
        lower = _finite("interval_lower", self.interval_lower)
        upper = _finite("interval_upper", self.interval_upper)
        if lower > upper:
            raise ValueError("effect adequacy interval is reversed")
        for name, value in (
            ("threshold_lower", self.threshold_lower),
            ("threshold_upper", self.threshold_upper),
        ):
            if value is not None:
                _finite(name, value)
        _positive_integer(
            "center_environment_count",
            self.center_environment_count,
        )
        if not isinstance(self.passes, bool):
            raise TypeError("effect adequacy passes must be a boolean")


@dataclass(frozen=True, slots=True)
class CandidatePower:
    environment_count: int
    hypotheses: tuple[HypothesisPower, ...]
    equivalence_hypotheses: tuple[EquivalencePowerBound, ...]
    registered_joint_rejections: int
    directional_global_null_rejections: int
    equivalence_lower_boundary_rejections: int
    equivalence_upper_boundary_rejections: int
    simulations: int
    certified_registered_joint_power_lower_bound: float
    directional_global_null_fwer_upper_bound: float
    equivalence_lower_boundary_error_upper_bound: float
    equivalence_upper_boundary_error_upper_bound: float
    meets_targets: bool

    def __post_init__(self) -> None:
        _positive_integer("environment_count", self.environment_count)
        simulations = _positive_integer("simulations", self.simulations)
        hypotheses = tuple(sorted(self.hypotheses, key=lambda item: item.name))
        if not hypotheses:
            raise ValueError("candidate power requires hypotheses")
        if len({item.name for item in hypotheses}) != len(hypotheses):
            raise ValueError("candidate hypothesis names must be unique")
        if any(item.simulations != simulations for item in hypotheses):
            raise ValueError("hypothesis simulation counts must agree")
        equivalences = tuple(
            sorted(self.equivalence_hypotheses, key=lambda item: item.name)
        )
        if len({item.name for item in equivalences}) != len(equivalences):
            raise ValueError("candidate equivalence hypothesis names must be unique")
        if any(item.simulations != simulations for item in equivalences):
            raise ValueError("equivalence simulation counts must agree")
        if set(item.name for item in hypotheses) & set(
            item.name for item in equivalences
        ):
            raise ValueError("directional and equivalence hypothesis names must differ")
        for name, value in (
            ("registered_joint_rejections", self.registered_joint_rejections),
            (
                "directional_global_null_rejections",
                self.directional_global_null_rejections,
            ),
            (
                "equivalence_lower_boundary_rejections",
                self.equivalence_lower_boundary_rejections,
            ),
            (
                "equivalence_upper_boundary_rejections",
                self.equivalence_upper_boundary_rejections,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= simulations
            ):
                raise ValueError(f"{name} must lie in [0, simulations]")
        for name, value in (
            (
                "certified_registered_joint_power_lower_bound",
                self.certified_registered_joint_power_lower_bound,
            ),
            (
                "directional_global_null_fwer_upper_bound",
                self.directional_global_null_fwer_upper_bound,
            ),
            (
                "equivalence_lower_boundary_error_upper_bound",
                self.equivalence_lower_boundary_error_upper_bound,
            ),
            (
                "equivalence_upper_boundary_error_upper_bound",
                self.equivalence_upper_boundary_error_upper_bound,
            ),
        ):
            _probability(name, value)
        if not isinstance(self.meets_targets, bool):
            raise TypeError("meets_targets must be a boolean")
        if not equivalences and (
            self.equivalence_lower_boundary_rejections
            or self.equivalence_upper_boundary_rejections
        ):
            raise ValueError(
                "equivalence boundary errors require equivalence hypotheses"
            )
        individual_rejections = tuple(
            item.rejections for item in (*hypotheses, *equivalences)
        )
        if self.registered_joint_rejections > min(individual_rejections):
            raise ValueError(
                "registered joint rejections cannot exceed an individual count"
            )
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "equivalence_hypotheses", equivalences)

    @property
    def registered_joint_power(self) -> float:
        return self.registered_joint_rejections / self.simulations

    @property
    def directional_global_null_fwer(self) -> float:
        return self.directional_global_null_rejections / self.simulations

    @property
    def equivalence_lower_boundary_error(self) -> float:
        return self.equivalence_lower_boundary_rejections / self.simulations

    @property
    def equivalence_upper_boundary_error(self) -> float:
        return self.equivalence_upper_boundary_rejections / self.simulations


@dataclass(frozen=True, slots=True)
class PowerCalibration:
    candidates: tuple[CandidatePower, ...]
    selected_environment_count: int | None
    calibration_environment_count: int
    center_environment_count: int
    probability_environment_count: int
    algorithm_replicas_per_environment: int
    effect_adequacy: tuple[EffectAdequacy, ...]
    simulations: int
    alpha: float
    minimum_power: float
    minimum_joint_power: float
    maximum_fwer: float
    minimum_equivalence_power: float = 0.90
    simulation_error_alpha: float = DEFAULT_SIMULATION_ERROR_ALPHA
    design_confidence_alpha: float = DEFAULT_DESIGN_CONFIDENCE_ALPHA
    simultaneous_design_event_count: int = field(init=False)
    per_event_confidence_alpha: float = field(init=False)
    simultaneous_decision_count: int = field(init=False)
    simulation_error_bound: float = field(init=False)

    def __post_init__(self) -> None:
        candidates = tuple(
            sorted(self.candidates, key=lambda item: item.environment_count)
        )
        if not candidates:
            raise ValueError("power calibration needs candidate results")
        counts = tuple(item.environment_count for item in candidates)
        if len(set(counts)) != len(counts):
            raise ValueError("candidate environment counts must be unique")
        simulations = _positive_integer("simulations", self.simulations)
        if any(item.simulations != simulations for item in candidates):
            raise ValueError("candidate simulation counts must agree")
        hypothesis_families = {
            tuple(item.name for item in candidate.hypotheses)
            for candidate in candidates
        }
        if len(hypothesis_families) != 1:
            raise ValueError("power candidates must share one hypothesis family")
        equivalence_families = {
            tuple(item.name for item in candidate.equivalence_hypotheses)
            for candidate in candidates
        }
        if len(equivalence_families) != 1:
            raise ValueError(
                "power candidates must share one equivalence hypothesis family"
            )
        _positive_integer(
            "calibration_environment_count",
            self.calibration_environment_count,
        )
        center_count = _positive_integer(
            "center_environment_count",
            self.center_environment_count,
        )
        probability_count = _positive_integer(
            "probability_environment_count",
            self.probability_environment_count,
        )
        if center_count + probability_count != self.calibration_environment_count:
            raise ValueError(
                "center and probability splits must exhaust calibration environments"
            )
        _positive_integer(
            "algorithm_replicas_per_environment",
            self.algorithm_replicas_per_environment,
        )
        adequacy = tuple(sorted(self.effect_adequacy, key=lambda item: item.name))
        if (
            not adequacy
            or any(not isinstance(item, EffectAdequacy) for item in adequacy)
            or len({item.name for item in adequacy}) != len(adequacy)
            or any(item.center_environment_count != center_count for item in adequacy)
        ):
            raise ValueError("effect adequacy records are incomplete or invalid")
        _alpha(self.alpha)
        _probability("minimum_power", self.minimum_power)
        _probability(
            "minimum_equivalence_power",
            self.minimum_equivalence_power,
        )
        _probability("minimum_joint_power", self.minimum_joint_power)
        _probability("maximum_fwer", self.maximum_fwer)
        simulation_error_alpha = _open_probability(
            "simulation_error_alpha",
            self.simulation_error_alpha,
        )
        design_confidence_alpha = _open_probability(
            "design_confidence_alpha",
            self.design_confidence_alpha,
        )
        hypothesis_count = len(next(iter(hypothesis_families)))
        equivalence_count = len(next(iter(equivalence_families)))
        decisions_per_candidate = (
            hypothesis_count + equivalence_count + 4
            if equivalence_count
            else hypothesis_count + 2
        )
        decision_count = len(candidates) * decisions_per_candidate
        error_bound = simultaneous_hoeffding_bound(
            simulations=simulations,
            decision_count=decision_count,
            simulation_error_alpha=simulation_error_alpha,
        )
        design_event_count = (
            2 * len(adequacy) + hypothesis_count + 2 * equivalence_count
        )
        per_event_alpha = design_confidence_alpha / design_event_count
        for candidate in candidates:
            expected = _meets_targets(
                candidate,
                minimum_power=self.minimum_power,
                minimum_equivalence_power=self.minimum_equivalence_power,
                minimum_joint_power=self.minimum_joint_power,
                maximum_fwer=self.maximum_fwer,
                effect_adequate=all(item.passes for item in adequacy),
            )
            if candidate.meets_targets is not expected:
                raise ValueError(
                    "candidate decision does not match certified population bounds"
                )
        passing = tuple(
            item.environment_count for item in candidates if item.meets_targets
        )
        expected = passing[0] if passing else None
        if self.selected_environment_count != expected:
            raise ValueError(
                "selected_environment_count must be the smallest passing design"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "effect_adequacy", adequacy)
        object.__setattr__(self, "simulation_error_alpha", simulation_error_alpha)
        object.__setattr__(self, "design_confidence_alpha", design_confidence_alpha)
        object.__setattr__(
            self,
            "simultaneous_design_event_count",
            design_event_count,
        )
        object.__setattr__(
            self,
            "per_event_confidence_alpha",
            per_event_alpha,
        )
        object.__setattr__(self, "simultaneous_decision_count", decision_count)
        object.__setattr__(self, "simulation_error_bound", error_bound)

    def lower_probability_bound(self, estimate: Real) -> float:
        return _lower_bound(
            _probability("probability estimate", estimate),
            self.simulation_error_bound,
        )

    def upper_probability_bound(self, estimate: Real) -> float:
        return _upper_bound(
            _probability("probability estimate", estimate),
            self.simulation_error_bound,
        )


def simultaneous_hoeffding_bound(
    *,
    simulations: int,
    decision_count: int,
    simulation_error_alpha: float = DEFAULT_SIMULATION_ERROR_ALPHA,
) -> float:
    """Return a union-bound radius for simultaneous one-sided decisions."""

    simulations = _positive_integer("simulations", simulations)
    decision_count = _positive_integer("decision_count", decision_count)
    error_alpha = _open_probability(
        "simulation_error_alpha",
        simulation_error_alpha,
    )
    radius = math.sqrt(math.log(decision_count / error_alpha) / (2 * simulations))
    return math.nextafter(radius, math.inf)


def _lower_bound(estimate: float, error_bound: float) -> float:
    value = estimate - error_bound
    return 0.0 if value <= 0.0 else math.nextafter(value, -math.inf)


def _upper_bound(estimate: float, error_bound: float) -> float:
    value = estimate + error_bound
    return 1.0 if value >= 1.0 else math.nextafter(value, math.inf)


def _binomial_right_tail_probability(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    probability = _probability("binomial probability", probability)
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    return min(
        1.0,
        math.fsum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        ),
    )


def _one_sided_clopper_pearson_lower(
    successes: int,
    trials: int,
    error_alpha: float,
) -> float:
    """Return the exact one-sided lower bound with strict successes."""

    trials = _positive_integer("trials", trials)
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes must lie in [0, trials]")
    error_alpha = _open_probability("error_alpha", error_alpha)
    if successes == 0:
        return 0.0
    lower = 0.0
    upper = 1.0
    for _ in range(128):
        candidate = (lower + upper) / 2.0
        tail = _binomial_right_tail_probability(
            successes,
            trials,
            candidate,
        )
        if tail < error_alpha:
            lower = candidate
        else:
            upper = candidate
    return max(0.0, math.nextafter(lower, -math.inf))


def _sign_rejection_cutoff(
    trials: int,
    *,
    raw_alpha: float,
) -> int:
    trials = _positive_integer("trials", trials)
    raw_alpha = _open_probability("raw_alpha", raw_alpha)
    for successes in range(trials + 1):
        if _binomial_right_tail_probability(successes, trials, 0.5) <= raw_alpha:
            return successes
    return trials + 1


def _certified_sign_power(
    trials: int,
    *,
    favorable_probability_lower_bound: float,
    raw_alpha: float,
) -> float:
    cutoff = _sign_rejection_cutoff(trials, raw_alpha=raw_alpha)
    return _binomial_right_tail_probability(
        cutoff,
        trials,
        favorable_probability_lower_bound,
    )


def _simultaneous_median_interval(
    values: tuple[float, ...],
    *,
    per_tail_alpha: float,
) -> tuple[float, float]:
    """Return an exact distribution-free interval for a median set."""

    ordered = tuple(sorted(_finite("center value", value) for value in values))
    trials = _positive_integer("center_environment_count", len(ordered))
    per_tail_alpha = _open_probability("per_tail_alpha", per_tail_alpha)
    upper_rank = next(
        (
            rank
            for rank in range(1, trials + 1)
            if _binomial_right_tail_probability(rank, trials, 0.5) <= per_tail_alpha
        ),
        None,
    )
    if upper_rank is None:
        raise ValueError(
            "center split is too small for the registered confidence budget"
        )
    lower_rank = trials - upper_rank + 1
    return ordered[lower_rank - 1], ordered[upper_rank - 1]


def _meets_targets(
    candidate: CandidatePower,
    *,
    minimum_power: float,
    minimum_equivalence_power: float,
    minimum_joint_power: float,
    maximum_fwer: float,
    effect_adequate: bool,
) -> bool:
    directional = (
        effect_adequate
        and all(
            item.certified_power_lower_bound >= minimum_power
            for item in candidate.hypotheses
        )
        and candidate.certified_registered_joint_power_lower_bound
        >= minimum_joint_power
        and candidate.directional_global_null_fwer_upper_bound <= maximum_fwer
    )
    if not candidate.equivalence_hypotheses:
        return directional
    return (
        directional
        and all(
            item.certified_power_lower_bound >= minimum_equivalence_power
            for item in candidate.equivalence_hypotheses
        )
        and candidate.equivalence_lower_boundary_error_upper_bound <= maximum_fwer
        and candidate.equivalence_upper_boundary_error_upper_bound <= maximum_fwer
    )


def _sample(
    hypothesis: PowerHypothesis,
    indices: tuple[int, ...],
    *,
    under_null: bool,
    residuals: tuple[float, ...],
) -> tuple[float, ...]:
    location = (
        hypothesis.null_value
        if under_null
        else hypothesis.null_value + hypothesis.minimum_effect
    )
    return tuple(location + residuals[index] for index in indices)


def _equivalence_sample(
    indices: tuple[int, ...],
    *,
    location: float,
    residuals: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(location + residuals[index] for index in indices)


@cache
def _sign_p_value_from_counts(
    above: int,
    below: int,
    trials: int,
    alternative: Alternative,
) -> float:
    """Reuse exact sign tails across bootstrap samples with equal counts."""

    signs = (1.0,) * above + (-1.0,) * below + (0.0,) * (trials - above - below)
    return exact_sign_p_value(signs, null=0.0, alternative=alternative)


def _sign_p_value(
    values: tuple[float, ...],
    *,
    null: float,
    alternative: Alternative,
) -> float:
    above = sum(value > null for value in values)
    below = sum(value < null for value in values)
    return _sign_p_value_from_counts(above, below, len(values), alternative)


def _equivalence_p_value(
    values: tuple[float, ...],
    *,
    margin: float,
) -> float:
    lower = _sign_p_value(
        values,
        null=-margin,
        alternative=Alternative.GREATER,
    )
    upper = _sign_p_value(
        values,
        null=margin,
        alternative=Alternative.LESS,
    )
    return max(lower, upper)


def _resample_indices(
    rng: CounterRNG,
    *,
    population: int,
    sample_size: int,
    candidate: int,
    simulation: int,
    regime: str,
) -> tuple[int, ...]:
    return tuple(
        rng.randbelow(
            population,
            regime,
            candidate,
            simulation,
            draw,
        )
        for draw in range(sample_size)
    )


def calibrate_environment_count(
    hypotheses: tuple[PowerHypothesis, ...],
    candidate_environment_counts: tuple[int, ...],
    *,
    equivalence_hypotheses: tuple[EquivalencePowerHypothesis, ...] = (),
    seed: Seed,
    simulations: int = DEFAULT_POWER_SIMULATIONS,
    alpha: float = 0.05,
    minimum_power: float = 0.90,
    minimum_equivalence_power: float = 0.90,
    minimum_joint_power: float = 0.80,
    maximum_fwer: float = 0.05,
    simulation_error_alpha: float = DEFAULT_SIMULATION_ERROR_ALPHA,
    design_confidence_alpha: float = DEFAULT_DESIGN_CONFIDENCE_ALPHA,
    center_environment_count: int | None = None,
) -> PowerCalibration:
    """Choose the smallest environment count with certified operating bounds.

    The outer cluster is the environment realization. Crossed algorithm
    replicas are averaged within that cluster before resampling, so supplying
    more algorithm cells cannot masquerade as more independent environments.
    A preregistered environment-index split separates simultaneous
    order-statistic effect-adequacy intervals from simultaneous exact
    Clopper-Pearson bounds on favorable-sign probabilities. Candidate powers
    are conservative sufficient-event bounds for the registered exact sign
    tests. The shifted paired bootstrap is retained only as a conditional
    working-model diagnostic and never determines selection.
    """

    try:
        family = tuple(hypotheses)
    except TypeError as error:
        raise TypeError("hypotheses must be a sequence") from error
    if not family:
        raise ValueError("registered power family must not be empty")
    if any(not isinstance(item, PowerHypothesis) for item in family):
        raise TypeError("hypotheses must contain PowerHypothesis records")
    family = tuple(sorted(family, key=lambda item: item.name))
    if len({item.name for item in family}) != len(family):
        raise ValueError("registered hypothesis names must be unique")
    environment_ids = family[0].environment_ids
    algorithm_replicas = family[0].algorithm_replicas
    for item in family[1:]:
        if item.environment_ids != environment_ids:
            raise ValueError("hypotheses must use the same paired environment clusters")
        if item.algorithm_replicas != algorithm_replicas:
            raise ValueError(
                "hypotheses must use the same fully crossed algorithm replicas"
            )
    try:
        equivalence_family = tuple(equivalence_hypotheses)
    except TypeError as error:
        raise TypeError("equivalence_hypotheses must be a sequence") from error
    if any(
        not isinstance(item, EquivalencePowerHypothesis) for item in equivalence_family
    ):
        raise TypeError(
            "equivalence_hypotheses must contain EquivalencePowerHypothesis records"
        )
    equivalence_family = tuple(sorted(equivalence_family, key=lambda item: item.name))
    if len({item.name for item in equivalence_family}) != len(equivalence_family):
        raise ValueError("registered equivalence hypothesis names must be unique")
    if {item.name for item in family} & {item.name for item in equivalence_family}:
        raise ValueError("directional and equivalence hypothesis names must differ")
    for item in equivalence_family:
        if item.environment_ids != environment_ids:
            raise ValueError(
                "equivalence hypotheses must use the same paired environment clusters"
            )
        if item.algorithm_replicas != algorithm_replicas:
            raise ValueError(
                "equivalence hypotheses must use the same fully crossed "
                "algorithm replicas"
            )

    simulations = _positive_integer("simulations", simulations)
    alpha = _alpha(alpha)
    minimum_power = _probability("minimum_power", minimum_power)
    minimum_equivalence_power = _probability(
        "minimum_equivalence_power",
        minimum_equivalence_power,
    )
    minimum_joint_power = _probability(
        "minimum_joint_power",
        minimum_joint_power,
    )
    maximum_fwer = _probability("maximum_fwer", maximum_fwer)
    simulation_error_alpha = _open_probability(
        "simulation_error_alpha",
        simulation_error_alpha,
    )
    design_confidence_alpha = _open_probability(
        "design_confidence_alpha",
        design_confidence_alpha,
    )
    try:
        candidates = tuple(candidate_environment_counts)
    except TypeError as error:
        raise TypeError("candidate_environment_counts must be a sequence") from error
    candidates = tuple(
        _positive_integer("candidate environment count", value) for value in candidates
    )
    if not candidates:
        raise ValueError("candidate_environment_counts must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate environment counts must be unique")
    candidates = tuple(sorted(candidates))
    rng = CounterRNG(seed, stream="analysis.cluster-power.v1")
    population = len(environment_ids)
    center_count = (
        population // 2
        if center_environment_count is None
        else _positive_integer(
            "center_environment_count",
            center_environment_count,
        )
    )
    if center_count >= population:
        raise ValueError(
            "center_environment_count must leave an independent probability split"
        )
    probability_count = population - center_count
    design_event_count = (
        2 * (len(family) + len(equivalence_family))
        + len(family)
        + 2 * len(equivalence_family)
    )
    per_event_alpha = design_confidence_alpha / design_event_count
    effect_adequacy: list[EffectAdequacy] = []
    favorable_bounds: dict[str, tuple[int, float]] = {}
    for item in family:
        center = item.cluster_differences[:center_count]
        probability = item.cluster_differences[center_count:]
        lower, upper = _simultaneous_median_interval(
            center,
            per_tail_alpha=per_event_alpha,
        )
        if item.alternative is Alternative.GREATER:
            successes = sum(value > item.null_value for value in probability)
            threshold_lower = item.null_value + item.minimum_effect
            threshold_upper = None
            adequate = lower >= threshold_lower
        elif item.alternative is Alternative.LESS:
            successes = sum(value < item.null_value for value in probability)
            threshold_lower = None
            threshold_upper = item.null_value + item.minimum_effect
            adequate = upper <= threshold_upper
        else:
            direction = 1.0 if item.minimum_effect > 0.0 else -1.0
            successes = sum(
                direction * (value - item.null_value) > 0.0 for value in probability
            )
            threshold_lower = (
                item.null_value + item.minimum_effect if direction > 0.0 else None
            )
            threshold_upper = (
                item.null_value + item.minimum_effect if direction < 0.0 else None
            )
            adequate = (
                lower >= threshold_lower
                if threshold_lower is not None
                else upper <= threshold_upper  # type: ignore[operator]
            )
        effect_adequacy.append(
            EffectAdequacy(
                name=item.name,
                kind="directional",
                interval_lower=lower,
                interval_upper=upper,
                threshold_lower=threshold_lower,
                threshold_upper=threshold_upper,
                center_environment_count=center_count,
                passes=adequate,
            )
        )
        favorable_bounds[item.name] = (
            successes,
            _one_sided_clopper_pearson_lower(
                successes,
                probability_count,
                per_event_alpha,
            ),
        )

    equivalence_bounds: dict[str, tuple[int, int, float, float]] = {}
    for item in equivalence_family:
        center = item.cluster_differences[:center_count]
        probability = item.cluster_differences[center_count:]
        lower, upper = _simultaneous_median_interval(
            center,
            per_tail_alpha=per_event_alpha,
        )
        effect_adequacy.append(
            EffectAdequacy(
                name=item.name,
                kind="equivalence",
                interval_lower=lower,
                interval_upper=upper,
                threshold_lower=-item.margin,
                threshold_upper=item.margin,
                center_environment_count=center_count,
                passes=lower > -item.margin and upper < item.margin,
            )
        )
        lower_successes = sum(value > -item.margin for value in probability)
        upper_successes = sum(value < item.margin for value in probability)
        equivalence_bounds[item.name] = (
            lower_successes,
            upper_successes,
            _one_sided_clopper_pearson_lower(
                lower_successes,
                probability_count,
                per_event_alpha,
            ),
            _one_sided_clopper_pearson_lower(
                upper_successes,
                probability_count,
                per_event_alpha,
            ),
        )
    all_effects_adequate = all(item.passes for item in effect_adequacy)
    directional_residuals = {item.name: item.centered_residuals for item in family}
    equivalence_residuals = {
        item.name: item.centered_residuals for item in equivalence_family
    }
    candidate_results = []
    for environment_count in candidates:
        rejections = {item.name: 0 for item in family}
        equivalence_rejections = {item.name: 0 for item in equivalence_family}
        registered_joint_rejections = 0
        directional_null_rejections = 0
        lower_boundary_rejections = 0
        upper_boundary_rejections = 0
        for simulation in range(simulations):
            power_indices = _resample_indices(
                rng,
                population=population,
                sample_size=environment_count,
                candidate=environment_count,
                simulation=simulation,
                regime="alternative",
            )
            power_p_values = tuple(
                (
                    item.name,
                    _sign_p_value(
                        _sample(
                            item,
                            power_indices,
                            under_null=False,
                            residuals=directional_residuals[item.name],
                        ),
                        null=item.null_value,
                        alternative=item.alternative,
                    ),
                )
                for item in family
            )
            decisions = holm_adjust(power_p_values, alpha=alpha)
            rejected = {item.name for item in decisions if item.reject_null}
            for name in rejected:
                rejections[name] += 1
            equivalence_p_values = tuple(
                (
                    item.name,
                    _equivalence_p_value(
                        _equivalence_sample(
                            power_indices,
                            location=item.diagnostic_location,
                            residuals=equivalence_residuals[item.name],
                        ),
                        margin=item.margin,
                    ),
                )
                for item in equivalence_family
            )
            equivalence_rejected = {
                item.name
                for item in holm_adjust(equivalence_p_values, alpha=alpha)
                if item.reject_null
            }
            for name in equivalence_rejected:
                equivalence_rejections[name] += 1
            registered_joint_rejections += len(rejected) == len(family) and len(
                equivalence_rejected
            ) == len(equivalence_family)

            error_indices = _resample_indices(
                rng,
                population=population,
                sample_size=environment_count,
                candidate=environment_count,
                simulation=simulation,
                regime="global-null",
            )
            null_p_values = tuple(
                (
                    item.name,
                    _sign_p_value(
                        _sample(
                            item,
                            error_indices,
                            under_null=True,
                            residuals=directional_residuals[item.name],
                        ),
                        null=item.null_value,
                        alternative=item.alternative,
                    ),
                )
                for item in family
            )
            directional_null_rejections += any(
                item.reject_null for item in holm_adjust(null_p_values, alpha=alpha)
            )
            for boundary, target in (
                ("lower", -1.0),
                ("upper", 1.0),
            ):
                boundary_p_values = tuple(
                    (
                        item.name,
                        _equivalence_p_value(
                            _equivalence_sample(
                                error_indices,
                                location=target * item.margin,
                                residuals=equivalence_residuals[item.name],
                            ),
                            margin=item.margin,
                        ),
                    )
                    for item in equivalence_family
                )
                false_equivalence = any(
                    item.reject_null
                    for item in holm_adjust(boundary_p_values, alpha=alpha)
                )
                if boundary == "lower":
                    lower_boundary_rejections += false_equivalence
                else:
                    upper_boundary_rejections += false_equivalence

        directional_raw_alpha = (
            alpha / len(family)
            if all(item.alternative is not Alternative.TWO_SIDED for item in family)
            else alpha / (2 * len(family))
        )
        hypothesis_results = tuple(
            HypothesisPower(
                name,
                rejections[name],
                simulations,
                favorable_sign_successes=favorable_bounds[name][0],
                favorable_sign_trials=probability_count,
                favorable_sign_probability_lower_bound=favorable_bounds[name][1],
                certified_power_lower_bound=_certified_sign_power(
                    environment_count,
                    favorable_probability_lower_bound=favorable_bounds[name][1],
                    raw_alpha=directional_raw_alpha,
                ),
            )
            for name in sorted(rejections)
        )
        equivalence_results_list = []
        endpoint_powers = []
        equivalence_raw_alpha = (
            alpha / len(equivalence_family) if equivalence_family else alpha
        )
        for name in sorted(equivalence_rejections):
            (
                lower_successes,
                upper_successes,
                lower_probability,
                upper_probability,
            ) = equivalence_bounds[name]
            lower_power = _certified_sign_power(
                environment_count,
                favorable_probability_lower_bound=lower_probability,
                raw_alpha=equivalence_raw_alpha,
            )
            upper_power = _certified_sign_power(
                environment_count,
                favorable_probability_lower_bound=upper_probability,
                raw_alpha=equivalence_raw_alpha,
            )
            endpoint_powers.extend((lower_power, upper_power))
            equivalence_results_list.append(
                EquivalencePowerBound(
                    name=name,
                    rejections=equivalence_rejections[name],
                    simulations=simulations,
                    lower_successes=lower_successes,
                    upper_successes=upper_successes,
                    favorable_sign_trials=probability_count,
                    lower_probability_bound=lower_probability,
                    upper_probability_bound=upper_probability,
                    lower_test_power_bound=lower_power,
                    upper_test_power_bound=upper_power,
                    certified_power_lower_bound=max(
                        0.0,
                        1.0 - (1.0 - lower_power) - (1.0 - upper_power),
                    ),
                )
            )
        equivalence_results = tuple(equivalence_results_list)
        joint_lower_bound = max(
            0.0,
            1.0
            - math.fsum(
                1.0 - item.certified_power_lower_bound for item in hypothesis_results
            )
            - math.fsum(1.0 - power for power in endpoint_powers),
        )
        candidate = CandidatePower(
            environment_count=environment_count,
            hypotheses=hypothesis_results,
            equivalence_hypotheses=equivalence_results,
            registered_joint_rejections=registered_joint_rejections,
            directional_global_null_rejections=directional_null_rejections,
            equivalence_lower_boundary_rejections=lower_boundary_rejections,
            equivalence_upper_boundary_rejections=upper_boundary_rejections,
            simulations=simulations,
            certified_registered_joint_power_lower_bound=joint_lower_bound,
            directional_global_null_fwer_upper_bound=alpha,
            equivalence_lower_boundary_error_upper_bound=(
                alpha if equivalence_family else 0.0
            ),
            equivalence_upper_boundary_error_upper_bound=(
                alpha if equivalence_family else 0.0
            ),
            meets_targets=False,
        )
        meets_targets = _meets_targets(
            candidate,
            minimum_power=minimum_power,
            minimum_equivalence_power=minimum_equivalence_power,
            minimum_joint_power=minimum_joint_power,
            maximum_fwer=maximum_fwer,
            effect_adequate=all_effects_adequate,
        )
        candidate_results.append(
            CandidatePower(
                environment_count=environment_count,
                hypotheses=hypothesis_results,
                equivalence_hypotheses=equivalence_results,
                registered_joint_rejections=registered_joint_rejections,
                directional_global_null_rejections=directional_null_rejections,
                equivalence_lower_boundary_rejections=lower_boundary_rejections,
                equivalence_upper_boundary_rejections=upper_boundary_rejections,
                simulations=simulations,
                certified_registered_joint_power_lower_bound=joint_lower_bound,
                directional_global_null_fwer_upper_bound=alpha,
                equivalence_lower_boundary_error_upper_bound=(
                    alpha if equivalence_family else 0.0
                ),
                equivalence_upper_boundary_error_upper_bound=(
                    alpha if equivalence_family else 0.0
                ),
                meets_targets=meets_targets,
            )
        )

    selected = next(
        (item.environment_count for item in candidate_results if item.meets_targets),
        None,
    )
    return PowerCalibration(
        candidates=tuple(candidate_results),
        selected_environment_count=selected,
        calibration_environment_count=population,
        center_environment_count=center_count,
        probability_environment_count=probability_count,
        algorithm_replicas_per_environment=algorithm_replicas,
        effect_adequacy=tuple(effect_adequacy),
        simulations=simulations,
        alpha=alpha,
        minimum_power=minimum_power,
        minimum_equivalence_power=minimum_equivalence_power,
        minimum_joint_power=minimum_joint_power,
        maximum_fwer=maximum_fwer,
        simulation_error_alpha=simulation_error_alpha,
        design_confidence_alpha=design_confidence_alpha,
    )


__all__ = [
    "DEFAULT_DESIGN_CONFIDENCE_ALPHA",
    "DEFAULT_POWER_SIMULATIONS",
    "DEFAULT_SIMULATION_ERROR_ALPHA",
    "CandidatePower",
    "EffectAdequacy",
    "EnvironmentCluster",
    "EquivalencePowerBound",
    "EquivalencePowerHypothesis",
    "HypothesisPower",
    "PowerCalibration",
    "PowerHypothesis",
    "calibrate_environment_count",
    "simultaneous_hoeffding_bound",
]
