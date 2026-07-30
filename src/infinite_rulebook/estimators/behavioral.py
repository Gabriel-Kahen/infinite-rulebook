"""Deterministic finite-table behavioral-channel estimation.

This module is a bounded foundation for learned frontier work.  It evaluates
complete tables for small finite problems, retains every feasible channel
witness, and uses global Lagrangian certificates for conservative lower
bounds.  It does not implement Monte Carlo, autoregressive rulebooks, or
infinite-frontier claims.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from numbers import Real

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.frontier.blahut_arimoto import (
    lagrangian_certificate_lower_bound,
)
from infinite_rulebook.frontier.finite_problem import (
    ChannelWitness,
    FiniteDecisionProblem,
)
from infinite_rulebook.validation import (
    DiagnosticSeverity,
    ValidationDiagnostic,
    ValidationReport,
)


class EstimatorError(RuntimeError):
    """Raised when an estimator result cannot be reported safely."""


class IdentificationStatus(StrEnum):
    """Scientific interpretation of a reported frontier point."""

    EXACT_ZERO_INFORMATION = "exact-zero-information"
    CERTIFIED_PARTIAL = "certified-partial-identification"
    INFEASIBLE = "infeasible"


_ESTIMATOR_LIMITATIONS = (
    "finite tabular state and canonical-action spaces only",
    "complete finite-channel floating-point evaluation; no Monte Carlo claim",
    "fixed beta grid and bounded optimization may leave wide intervals",
    "record constructors are structural; only factory outputs are problem-bound",
    "no autoregressive, infinite-N, bottleneck, or scaling claim",
)

_POINT_METHODS = {
    IdentificationStatus.EXACT_ZERO_INFORMATION: (
        "zero-information-optimum",
        "directly-evaluated-constant-action-witness",
    ),
    IdentificationStatus.CERTIFIED_PARTIAL: (
        "global-lagrangian-reference-certificate",
        "directly-evaluated-feasible-finite-channel",
    ),
    IdentificationStatus.INFEASIBLE: (
        "finite-maximum-reward",
        "no-feasible-channel",
    ),
}


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result == 0.0:
        result = 0.0
    return result


def _nonnegative(name: str, value: Real) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{name} must be nonempty without surrounding whitespace")
    return normalized


def _semantic_digest(name: str, value: str) -> str:
    digest = _text(name, value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _canonical_witness(name: str, witness: ChannelWitness) -> ChannelWitness:
    if not isinstance(witness, ChannelWitness):
        raise TypeError(f"{name} must be a ChannelWitness")
    try:
        channel = tuple(
            tuple(
                _nonnegative(f"{name}.channel[{state}][{action}]", probability)
                for action, probability in enumerate(row)
            )
            for state, row in enumerate(witness.channel)
        )
    except TypeError as error:
        raise TypeError(f"{name}.channel must be a finite matrix") from error
    if not channel or not channel[0]:
        raise ValueError(f"{name}.channel must not be empty")
    action_count = len(channel[0])
    if any(len(row) != action_count for row in channel):
        raise ValueError(f"{name}.channel rows must have equal length")
    if any(math.fsum(row) != 1.0 for row in channel):
        raise ValueError(f"{name}.channel rows must sum exactly to one")
    try:
        marginal = tuple(
            _nonnegative(f"{name}.action_marginal[{action}]", probability)
            for action, probability in enumerate(witness.action_marginal)
        )
    except TypeError as error:
        raise TypeError(f"{name}.action_marginal must be a finite sequence") from error
    if len(marginal) != action_count:
        raise ValueError(f"{name}.action_marginal has the wrong length")
    if not math.isclose(
        math.fsum(marginal),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name}.action_marginal must sum to one")
    return ChannelWitness(
        channel=channel,
        action_marginal=marginal,
        expected_reward=_finite(f"{name}.expected_reward", witness.expected_reward),
        mutual_information=_nonnegative(
            f"{name}.mutual_information",
            witness.mutual_information,
        ),
    )


@dataclass(frozen=True, slots=True)
class BehavioralEstimatorConfig:
    """Prespecified deterministic optimization settings."""

    betas: tuple[float, ...] = (
        0.0,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    )
    optimizer_steps: int = 256
    reference_update_rate: float = 1.0
    reference_smoothing: float = 1e-12
    diagnostic_tolerance: float = 1e-8
    maximum_states: int = 128
    maximum_actions: int = 512

    def __post_init__(self) -> None:
        betas = tuple(
            _finite(f"betas[{index}]", beta) for index, beta in enumerate(self.betas)
        )
        if not betas or betas[0] != 0.0:
            raise ValueError("betas must start at zero")
        if any(beta < 0.0 for beta in betas):
            raise ValueError("betas must be nonnegative")
        if any(left >= right for left, right in pairwise(betas)):
            raise ValueError("betas must be strictly increasing")
        for name in ("optimizer_steps", "maximum_states", "maximum_actions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        update_rate = _finite("reference_update_rate", self.reference_update_rate)
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("reference_update_rate must lie in (0, 1]")
        smoothing = _finite("reference_smoothing", self.reference_smoothing)
        if not 0.0 < smoothing < 1.0:
            raise ValueError("reference_smoothing must lie in (0, 1)")
        if smoothing / self.maximum_actions == 0.0:
            raise ValueError("reference_smoothing is too small for maximum_actions")
        tolerance = _finite("diagnostic_tolerance", self.diagnostic_tolerance)
        if tolerance <= 0.0:
            raise ValueError("diagnostic_tolerance must be strictly positive")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "reference_update_rate", update_rate)
        object.__setattr__(self, "reference_smoothing", smoothing)
        object.__setattr__(self, "diagnostic_tolerance", tolerance)


def _validate_problem_size(
    problem: FiniteDecisionProblem,
    config: BehavioralEstimatorConfig,
) -> None:
    if problem.state_count > config.maximum_states:
        raise ValueError("problem exceeds config.maximum_states")
    if problem.action_count > config.maximum_actions:
        raise ValueError("problem exceeds config.maximum_actions")


def _unit_sum_row(row: tuple[float, ...]) -> tuple[float, ...]:
    values = list(row)
    pivot = max(range(len(values)), key=values.__getitem__)
    other_total = math.fsum(
        value for index, value in enumerate(values) if index != pivot
    )
    values[pivot] = max(0.0, 1.0 - other_total)
    for _ in range(16):
        total = math.fsum(values)
        if total == 1.0:
            return tuple(values)
        direction = math.inf if total < 1.0 else 0.0
        updated = math.nextafter(values[pivot], direction)
        if updated == values[pivot] or updated < 0.0:
            break
        values[pivot] = updated
    raise EstimatorError("could not construct an idempotent probability row")


def _public_witness(
    problem: FiniteDecisionProblem,
    channel: Sequence[Sequence[Real]],
) -> ChannelWitness:
    canonical = problem.validate_channel(channel)
    stabilized = tuple(_unit_sum_row(row) for row in canonical)
    witness = problem.evaluate(stabilized)
    if witness.channel != stabilized or problem.evaluate(witness.channel) != witness:
        raise EstimatorError("retained channel witness is not exactly reproducible")
    return witness


@dataclass(frozen=True, slots=True)
class BehavioralFit:
    """One bounded direct-channel fit at a fixed Lagrange multiplier."""

    beta: float
    steps: int
    witness: ChannelWitness
    reference_marginal: tuple[float, ...]
    direct_reference_kl: float
    reference_kl_upper_bound: float
    reference_compression_gap: float
    reference_identity_residual: float
    objective_lower_bound: float
    objective_upper_bound: float
    certified_objective_gap: float
    fixed_point_residual: float
    converged: bool
    diagnostics: ValidationReport

    def __post_init__(self) -> None:
        beta = _nonnegative("beta", self.beta)
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 1
        ):
            raise ValueError("steps must be a positive integer")
        witness = _canonical_witness("witness", self.witness)
        try:
            reference = tuple(
                _finite(f"reference_marginal[{index}]", probability)
                for index, probability in enumerate(self.reference_marginal)
            )
        except TypeError as error:
            raise TypeError("reference_marginal must be a finite sequence") from error
        if (
            len(reference) != len(witness.action_marginal)
            or any(probability <= 0.0 for probability in reference)
            or not math.isclose(
                math.fsum(reference),
                1.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("reference_marginal must be full-support and normalized")
        direct_reference_kl = _nonnegative(
            "direct_reference_kl",
            self.direct_reference_kl,
        )
        nonnegative = {
            name: _nonnegative(name, getattr(self, name))
            for name in (
                "reference_kl_upper_bound",
                "reference_compression_gap",
                "reference_identity_residual",
                "certified_objective_gap",
                "fixed_point_residual",
            )
        }
        objective_lower = _finite("objective_lower_bound", self.objective_lower_bound)
        objective_upper = _finite("objective_upper_bound", self.objective_upper_bound)
        roundoff = 64.0 * math.ulp(max(1.0, abs(objective_lower), abs(objective_upper)))
        if objective_lower > objective_upper + roundoff:
            raise ValueError("objective bounds are inconsistent")
        expected_gap = max(0.0, objective_upper - objective_lower)
        if not math.isclose(
            nonnegative["certified_objective_gap"],
            expected_gap,
            rel_tol=1e-12,
            abs_tol=roundoff,
        ):
            raise ValueError("certified_objective_gap does not match objective bounds")
        if (
            nonnegative["reference_kl_upper_bound"]
            != witness.mutual_information + nonnegative["reference_compression_gap"]
        ):
            raise ValueError("reference KL upper bound is inconsistent")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        if not isinstance(self.diagnostics, ValidationReport):
            raise TypeError("diagnostics must be a ValidationReport")
        if not self.diagnostics.valid:
            raise ValueError("fit diagnostics must be valid")
        expected_diagnostics = (
            ValidationReport()
            if self.converged
            else ValidationReport(
                (
                    ValidationDiagnostic(
                        DiagnosticSeverity.WARNING,
                        "bounded-optimization-not-converged",
                        f"beta[{beta.hex()}]",
                        "the fixed optimization budget ended before the "
                        "diagnostic tolerance",
                    ),
                )
            )
        )
        if self.diagnostics != expected_diagnostics:
            raise ValueError("fit diagnostics are inconsistent with convergence")
        if nonnegative["fixed_point_residual"] > 1.0 + 1e-12:
            raise ValueError("fixed_point_residual exceeds its probability range")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "witness", witness)
        object.__setattr__(self, "reference_marginal", reference)
        object.__setattr__(self, "direct_reference_kl", direct_reference_kl)
        for name, value in nonnegative.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "objective_lower_bound", objective_lower)
        object.__setattr__(self, "objective_upper_bound", objective_upper)


@dataclass(frozen=True, slots=True)
class BehavioralFrontierPoint:
    """A feasible upper witness and conservative frontier interval."""

    target_reward: float
    witness: ChannelWitness | None
    lower_bound: float
    upper_bound: float
    lower_bound_beta: float | None
    identification: IdentificationStatus
    lower_bound_method: str
    upper_bound_method: str
    diagnostics: ValidationReport

    @property
    def interval_width(self) -> float:
        if not math.isfinite(self.lower_bound) or not math.isfinite(self.upper_bound):
            return math.inf
        return self.upper_bound - self.lower_bound

    def __post_init__(self) -> None:
        target = _finite("target_reward", self.target_reward)
        if not isinstance(self.identification, IdentificationStatus):
            raise TypeError("identification must be an IdentificationStatus")
        lower_method = _text("lower_bound_method", self.lower_bound_method)
        upper_method = _text("upper_bound_method", self.upper_bound_method)
        if (lower_method, upper_method) != _POINT_METHODS[self.identification]:
            raise ValueError("bound methods are inconsistent with identification")
        if not isinstance(self.diagnostics, ValidationReport):
            raise TypeError("diagnostics must be a ValidationReport")
        if not self.diagnostics.valid:
            raise ValueError("point diagnostics must be valid")
        lower_beta = self.lower_bound_beta
        if lower_beta is not None:
            lower_beta = _nonnegative("lower_bound_beta", lower_beta)

        if self.identification is IdentificationStatus.INFEASIBLE:
            if self.witness is not None:
                raise ValueError("an infeasible point must not retain a witness")
            if self.lower_bound != math.inf or self.upper_bound != math.inf:
                raise ValueError(
                    "an infeasible point must have positive-infinite bounds"
                )
            if lower_beta is not None:
                raise ValueError("an infeasible point must not select a lower beta")
            if self.diagnostics != ValidationReport():
                raise ValueError("an infeasible point must not retain diagnostics")
            lower = upper = math.inf
            witness = None
        else:
            witness = _canonical_witness("witness", self.witness)
            lower = _nonnegative("lower_bound", self.lower_bound)
            upper = _nonnegative("upper_bound", self.upper_bound)
            if lower > upper:
                raise ValueError("lower_bound must not exceed upper_bound")
            if upper != witness.mutual_information:
                raise ValueError("upper_bound must match the retained witness")
            if witness.expected_reward < target:
                raise ValueError("retained witness does not clear target_reward")
            if lower_beta is None:
                raise ValueError("a witnessed point must select a lower beta")
            if self.identification is IdentificationStatus.EXACT_ZERO_INFORMATION:
                if lower != 0.0 or lower_beta != 0.0:
                    raise ValueError("zero-information point must start at zero")
                if len(set(witness.channel)) != 1:
                    raise ValueError(
                        "zero-information witness must be state-independent"
                    )
                roundoff = 64.0 * math.ulp(max(1.0, abs(upper)))
                if upper > roundoff:
                    raise ValueError("zero-information witness exceeds roundoff")
                expected_diagnostics = (
                    ValidationReport()
                    if upper == 0.0
                    else ValidationReport(
                        (
                            ValidationDiagnostic(
                                DiagnosticSeverity.INFO,
                                "constant-channel-evaluation-roundoff",
                                f"target[{target.hex()}]",
                                "the exactly replayed constant witness retained a "
                                "numerical information residual",
                            ),
                        )
                    )
                )
                if self.diagnostics != expected_diagnostics:
                    raise ValueError(
                        "zero-information diagnostics do not match its witness"
                    )
            elif (
                any(
                    diagnostic.severity is not DiagnosticSeverity.INFO
                    or diagnostic.code != "bound-roundoff-reconciled"
                    or diagnostic.path != f"target[{target.hex()}]"
                    or diagnostic.message
                    != "bounds crossed only within floating-point roundoff"
                    for diagnostic in self.diagnostics.diagnostics
                )
                or len(self.diagnostics.diagnostics) > 1
            ):
                raise ValueError("partial-identification diagnostics are inconsistent")

        object.__setattr__(self, "target_reward", target)
        object.__setattr__(self, "witness", witness)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "lower_bound_beta", lower_beta)
        object.__setattr__(self, "lower_bound_method", lower_method)
        object.__setattr__(self, "upper_bound_method", upper_method)


@dataclass(frozen=True, slots=True)
class BehavioralFrontierEstimate:
    """A deterministic finite-problem estimate over a requested reward grid."""

    problem_semantic_hash: str
    config: BehavioralEstimatorConfig
    fits: tuple[BehavioralFit, ...]
    points: tuple[BehavioralFrontierPoint, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        problem_hash = _semantic_digest(
            "problem_semantic_hash",
            self.problem_semantic_hash,
        )
        if not isinstance(self.config, BehavioralEstimatorConfig):
            raise TypeError("config must be a BehavioralEstimatorConfig")
        try:
            fits = tuple(self.fits)
            points = tuple(self.points)
            limitations = tuple(
                _text(f"limitations[{index}]", limitation)
                for index, limitation in enumerate(self.limitations)
            )
        except TypeError as error:
            raise TypeError(
                "fits, points, and limitations must be finite sequences"
            ) from error
        if any(not isinstance(fit, BehavioralFit) for fit in fits):
            raise TypeError("fits must contain BehavioralFit records")
        if any(not isinstance(point, BehavioralFrontierPoint) for point in points):
            raise TypeError("points must contain BehavioralFrontierPoint records")
        if not points:
            raise ValueError("points must not be empty")
        targets = tuple(point.target_reward for point in points)
        if targets != tuple(sorted(set(targets))):
            raise ValueError("points must have unique increasing targets")
        if fits and tuple(fit.beta for fit in fits) != self.config.betas:
            raise ValueError("fits must cover the configured beta grid in order")
        has_partial = any(
            point.identification is IdentificationStatus.CERTIFIED_PARTIAL
            for point in points
        )
        if bool(fits) != has_partial:
            raise ValueError("fits must be present exactly when a point is partial")
        if any(fit.steps != self.config.optimizer_steps for fit in fits):
            raise ValueError("fit steps must match the estimator configuration")
        if any(
            point.identification is IdentificationStatus.CERTIFIED_PARTIAL
            and point.lower_bound_beta not in self.config.betas
            for point in points
        ):
            raise ValueError("point lower betas must belong to the configured grid")
        shapes = {
            (len(witness.channel), len(witness.channel[0]))
            for witness in (
                *(fit.witness for fit in fits),
                *(point.witness for point in points if point.witness is not None),
            )
        }
        if len(shapes) > 1:
            raise ValueError("all retained witnesses must have one problem shape")
        status_rank = {
            IdentificationStatus.EXACT_ZERO_INFORMATION: 0,
            IdentificationStatus.CERTIFIED_PARTIAL: 1,
            IdentificationStatus.INFEASIBLE: 2,
        }
        ranks = tuple(status_rank[point.identification] for point in points)
        if ranks != tuple(sorted(ranks)):
            raise ValueError("point statuses are inconsistent with target order")
        if limitations != _ESTIMATOR_LIMITATIONS:
            raise ValueError("limitations must retain the complete scientific boundary")
        object.__setattr__(self, "problem_semantic_hash", problem_hash)
        object.__setattr__(self, "fits", fits)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "limitations", limitations)


def _gibbs_channel(
    problem: FiniteDecisionProblem,
    beta: float,
    reference: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    log_reference = tuple(math.log(value) for value in reference)
    rows = []
    for state, rewards in enumerate(problem.rewards):
        logits = tuple(
            log_reference[action] + beta * rewards[action]
            for action in range(problem.action_count)
        )
        if any(not math.isfinite(value) for value in logits):
            raise ValueError(f"beta-scaled reward is not finite for state {state}")
        maximum = max(logits)
        weights = tuple(math.exp(value - maximum) for value in logits)
        total = math.fsum(weights)
        rows.append(tuple(value / total for value in weights))
    return tuple(rows)


def _updated_reference(
    marginal: tuple[float, ...],
    reference: tuple[float, ...],
    config: BehavioralEstimatorConfig,
) -> tuple[float, ...]:
    action_count = len(marginal)
    smoothed = tuple(
        (1.0 - config.reference_smoothing) * value
        + config.reference_smoothing / action_count
        for value in marginal
    )
    updated = tuple(
        (1.0 - config.reference_update_rate) * old + config.reference_update_rate * new
        for old, new in zip(reference, smoothed, strict=True)
    )
    total = math.fsum(updated)
    return tuple(value / total for value in updated)


def _weighted_log_ratio(
    factors: tuple[float, ...],
    numerator: float,
    denominator: float,
) -> float:
    ratio = numerator / denominator
    log_ratio = (
        math.log(ratio)
        if math.isfinite(ratio) and ratio > 0.0
        else math.log(numerator) - math.log(denominator)
    )
    if log_ratio == 0.0:
        return 0.0
    weight = 1.0
    for factor in factors:
        weight *= factor
    term = weight * log_ratio
    if weight > 0.0 and term != 0.0:
        return term
    log_magnitude = math.fsum(math.log(factor) for factor in factors) + math.log(
        abs(log_ratio)
    )
    return math.copysign(math.exp(log_magnitude), log_ratio)


def _reference_kl(
    problem: FiniteDecisionProblem,
    witness: ChannelWitness,
    reference: tuple[float, ...],
) -> float:
    return math.fsum(
        _weighted_log_ratio(
            (problem.prior[state], conditional),
            conditional,
            reference[action],
        )
        for state, row in enumerate(witness.channel)
        if problem.prior[state] > 0.0
        for action, conditional in enumerate(row)
        if conditional > 0.0
    )


def _marginal_kl(
    marginal: tuple[float, ...],
    reference: tuple[float, ...],
) -> float:
    return math.fsum(
        _weighted_log_ratio(
            (probability,),
            probability,
            reference[action],
        )
        for action, probability in enumerate(marginal)
        if probability > 0.0
    )


def fit_behavioral_channel(
    problem: FiniteDecisionProblem,
    beta: Real,
    *,
    config: BehavioralEstimatorConfig | None = None,
) -> BehavioralFit:
    """Fit a finite-table channel with bounded alternating reference updates.

    The returned channel is always evaluated directly by ``problem``.  The
    reference KL is an upper bound on its mutual information, while the
    Lagrangian lower certificate remains valid even if optimization does not
    converge.
    """

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    settings = BehavioralEstimatorConfig() if config is None else config
    if not isinstance(settings, BehavioralEstimatorConfig):
        raise TypeError("config must be a BehavioralEstimatorConfig")
    _validate_problem_size(problem, settings)
    multiplier = _finite("beta", beta)
    if multiplier < 0.0:
        raise ValueError("beta must be nonnegative")

    reference = tuple(1.0 / problem.action_count for _ in range(problem.action_count))
    for _ in range(settings.optimizer_steps):
        witness = problem.evaluate(_gibbs_channel(problem, multiplier, reference))
        reference = _updated_reference(
            witness.action_marginal,
            reference,
            settings,
        )
    witness = _public_witness(
        problem,
        _gibbs_channel(problem, multiplier, reference),
    )

    raw_direct_reference_kl = _reference_kl(problem, witness, reference)
    raw_compression_gap = _marginal_kl(witness.action_marginal, reference)
    compression_roundoff = 64.0 * math.ulp(
        max(1.0, abs(raw_direct_reference_kl), abs(witness.mutual_information))
    )
    if raw_direct_reference_kl < -compression_roundoff:
        raise EstimatorError("direct reference KL is materially negative")
    if raw_compression_gap < -compression_roundoff:
        raise EstimatorError("reference-marginal KL is materially negative")
    direct_reference_kl = max(0.0, raw_direct_reference_kl)
    compression_gap = max(0.0, raw_compression_gap)
    reference_kl = witness.mutual_information + compression_gap
    identity_residual = abs(
        raw_direct_reference_kl - witness.mutual_information - raw_compression_gap
    )
    objective_lower = lagrangian_certificate_lower_bound(
        problem,
        multiplier,
        reference,
    )
    objective_upper = witness.mutual_information - multiplier * witness.expected_reward
    gap = objective_upper - objective_lower
    roundoff = 64.0 * math.ulp(max(1.0, abs(objective_lower), abs(objective_upper)))
    diagnostics = []
    if identity_residual > settings.diagnostic_tolerance:
        diagnostics.append(
            ValidationDiagnostic(
                DiagnosticSeverity.ERROR,
                "reference-kl-identity",
                f"beta[{multiplier.hex()}]",
                "reference KL identity failed",
            )
        )
    if gap < -roundoff:
        diagnostics.append(
            ValidationDiagnostic(
                DiagnosticSeverity.ERROR,
                "invalid-lagrangian-certificate",
                f"beta[{multiplier.hex()}]",
                "Lagrangian lower certificate exceeds the feasible objective",
            )
        )
    gap = max(0.0, gap)
    residual = max(
        abs(left - right)
        for left, right in zip(
            reference,
            witness.action_marginal,
            strict=True,
        )
    )
    converged = (
        gap <= settings.diagnostic_tolerance
        and residual <= math.sqrt(settings.diagnostic_tolerance)
        and not diagnostics
    )
    if not converged and not diagnostics:
        diagnostics.append(
            ValidationDiagnostic(
                DiagnosticSeverity.WARNING,
                "bounded-optimization-not-converged",
                f"beta[{multiplier.hex()}]",
                "the fixed optimization budget ended before the diagnostic tolerance",
            )
        )
    report = ValidationReport(tuple(diagnostics))
    if not report.valid:
        raise EstimatorError("behavioral fit failed deterministic consistency checks")
    return BehavioralFit(
        beta=multiplier,
        steps=settings.optimizer_steps,
        witness=witness,
        reference_marginal=reference,
        direct_reference_kl=direct_reference_kl,
        reference_kl_upper_bound=reference_kl,
        reference_compression_gap=compression_gap,
        reference_identity_residual=identity_residual,
        objective_lower_bound=objective_lower,
        objective_upper_bound=objective_upper,
        certified_objective_gap=gap,
        fixed_point_residual=residual,
        converged=converged,
        diagnostics=report,
    )


def _mixed_witness(
    problem: FiniteDecisionProblem,
    below: ChannelWitness,
    above: ChannelWitness,
    target: float,
) -> ChannelWitness:
    if below.expected_reward >= target:
        return below
    span = above.expected_reward - below.expected_reward
    if span <= 0.0:
        return above
    weight = min(1.0, max(0.0, (target - below.expected_reward) / span))
    for _ in range(4):
        channel = tuple(
            tuple(
                (1.0 - weight) * below.channel[state][action]
                + weight * above.channel[state][action]
                for action in range(problem.action_count)
            )
            for state in range(problem.state_count)
        )
        witness = _public_witness(problem, channel)
        if witness.expected_reward >= target or weight >= 1.0:
            return witness
        weight = math.nextafter(weight, 1.0)
    return above


def _upper_witness(
    problem: FiniteDecisionProblem,
    target: float,
    candidates: tuple[ChannelWitness, ...],
) -> ChannelWitness:
    below = tuple(item for item in candidates if item.expected_reward < target)
    above = tuple(item for item in candidates if item.expected_reward >= target)
    feasible = list(above)
    feasible.extend(
        _mixed_witness(problem, left, right, target)
        for left in below
        for right in above
        if right.expected_reward > left.expected_reward
    )
    if not feasible:
        raise EstimatorError("no feasible upper-bound witness was constructed")
    valid = tuple(item for item in feasible if item.expected_reward >= target)
    if not valid:
        raise EstimatorError("constructed witnesses do not clear the reward target")
    return min(
        valid,
        key=lambda item: (
            item.mutual_information,
            item.expected_reward,
            item.channel,
        ),
    )


def _validate_frontier_point(
    problem: FiniteDecisionProblem,
    point: BehavioralFrontierPoint,
) -> None:
    witness = point.witness
    if witness is None:
        if (
            point.identification is not IdentificationStatus.INFEASIBLE
            or point.lower_bound != math.inf
            or point.upper_bound != math.inf
        ):
            raise EstimatorError("a witness-free point must be explicitly infeasible")
        return
    if point.identification is IdentificationStatus.INFEASIBLE:
        raise EstimatorError("an infeasible point must not retain a witness")
    if problem.evaluate(witness.channel) != witness:
        raise EstimatorError("frontier witness is not exactly reproducible")
    if witness.expected_reward < point.target_reward:
        raise EstimatorError("frontier witness does not clear its reward target")
    if point.upper_bound != witness.mutual_information:
        raise EstimatorError("frontier upper bound does not match its retained witness")
    if point.lower_bound > point.upper_bound:
        raise EstimatorError("frontier lower bound exceeds its upper bound")
    if point.identification is IdentificationStatus.EXACT_ZERO_INFORMATION:
        if point.lower_bound != 0.0 or len(set(witness.channel)) != 1:
            raise EstimatorError("zero-information point has inconsistent evidence")
        roundoff = 64.0 * math.ulp(max(1.0, abs(point.upper_bound)))
        if point.upper_bound > roundoff:
            raise EstimatorError("constant-channel information exceeds roundoff")


def estimate_behavioral_frontier(
    problem: FiniteDecisionProblem,
    targets: Sequence[Real],
    *,
    config: BehavioralEstimatorConfig | None = None,
) -> BehavioralFrontierEstimate:
    """Estimate a finite frontier with explicit partial-identification bounds."""

    if not isinstance(problem, FiniteDecisionProblem):
        raise TypeError("problem must be a FiniteDecisionProblem")
    settings = BehavioralEstimatorConfig() if config is None else config
    if not isinstance(settings, BehavioralEstimatorConfig):
        raise TypeError("config must be a BehavioralEstimatorConfig")
    _validate_problem_size(problem, settings)
    requested = tuple(
        sorted(
            {_finite(f"targets[{index}]", value) for index, value in enumerate(targets)}
        )
    )
    if not requested:
        raise ValueError("targets must not be empty")

    requires_estimation = any(
        problem.zero_information_reward < target <= problem.maximum_reward
        for target in requested
    )
    fits = (
        tuple(
            fit_behavioral_channel(problem, beta, config=settings)
            for beta in settings.betas
        )
        if requires_estimation
        else ()
    )
    candidates = (
        tuple(
            {
                item.channel: item
                for item in (
                    problem.constant_channel(),
                    *(fit.witness for fit in fits),
                    problem.maximizing_channel(),
                )
            }.values()
        )
        if requires_estimation
        else ()
    )
    points = []
    for target in requested:
        if target <= problem.zero_information_reward:
            witness = problem.constant_channel()
            diagnostics = []
            if witness.mutual_information != 0.0:
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.INFO,
                        "constant-channel-evaluation-roundoff",
                        f"target[{target.hex()}]",
                        "the exactly replayed constant witness retained a numerical "
                        "information residual",
                    )
                )
            points.append(
                BehavioralFrontierPoint(
                    target_reward=target,
                    witness=witness,
                    lower_bound=0.0,
                    upper_bound=witness.mutual_information,
                    lower_bound_beta=0.0,
                    identification=IdentificationStatus.EXACT_ZERO_INFORMATION,
                    lower_bound_method="zero-information-optimum",
                    upper_bound_method="directly-evaluated-constant-action-witness",
                    diagnostics=ValidationReport(tuple(diagnostics)),
                )
            )
            continue
        if target > problem.maximum_reward:
            points.append(
                BehavioralFrontierPoint(
                    target_reward=target,
                    witness=None,
                    lower_bound=math.inf,
                    upper_bound=math.inf,
                    lower_bound_beta=None,
                    identification=IdentificationStatus.INFEASIBLE,
                    lower_bound_method="finite-maximum-reward",
                    upper_bound_method="no-feasible-channel",
                    diagnostics=ValidationReport(),
                )
            )
            continue

        witness = _upper_witness(problem, target, candidates)
        lower_candidates = tuple(
            (max(0.0, fit.beta * target + fit.objective_lower_bound), fit.beta)
            for fit in fits
        )
        lower, lower_beta = max(lower_candidates)
        upper = witness.mutual_information
        roundoff = 64.0 * math.ulp(max(1.0, abs(lower), abs(upper)))
        diagnostics = []
        if lower > upper + roundoff:
            raise EstimatorError(
                "certified lower bound exceeds the feasible upper-bound witness"
            )
        if lower > upper:
            lower = upper
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.INFO,
                    "bound-roundoff-reconciled",
                    f"target[{target.hex()}]",
                    "bounds crossed only within floating-point roundoff",
                )
            )
        points.append(
            BehavioralFrontierPoint(
                target_reward=target,
                witness=witness,
                lower_bound=lower,
                upper_bound=upper,
                lower_bound_beta=lower_beta,
                identification=IdentificationStatus.CERTIFIED_PARTIAL,
                lower_bound_method="global-lagrangian-reference-certificate",
                upper_bound_method="directly-evaluated-feasible-finite-channel",
                diagnostics=ValidationReport(tuple(diagnostics)),
            )
        )

    validated_points = tuple(points)
    for point in validated_points:
        _validate_frontier_point(problem, point)
    return BehavioralFrontierEstimate(
        problem_semantic_hash=semantic_hash(problem),
        config=settings,
        fits=fits,
        points=validated_points,
        limitations=(*_ESTIMATOR_LIMITATIONS,),
    )
