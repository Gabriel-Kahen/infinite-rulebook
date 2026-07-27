"""Core symbolic metrics with explicit estimand and interval types."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING

from infinite_rulebook.validation import (
    DiagnosticSeverity,
    ValidationDiagnostic,
    ValidationReport,
)

if TYPE_CHECKING:
    from infinite_rulebook.frontier.finite_problem import (
        Channel,
        ChannelWitness,
        FiniteDecisionProblem,
    )


def _real(name: str, value: Real, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result) or (finite and not math.isfinite(result)):
        qualifier = "finite " if finite else ""
        raise ValueError(f"{name} must be a {qualifier}non-NaN number")
    return result


def _validate_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MetricInterval:
    """A closed lower/upper interval for one scalar estimand."""

    lower: float
    upper: float
    units: str

    def __post_init__(self) -> None:
        lower = _real("lower", self.lower, finite=False)
        upper = _real("upper", self.upper, finite=False)
        if lower > upper:
            raise ValueError("interval lower bound exceeds upper bound")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a nonempty string")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True, init=False)
class FrontierUpperWitness:
    """Auditable feasible witness backing one frontier upper endpoint."""

    problem_semantic_hash: str
    channel_witness: ChannelWitness
    witness_hash: str

    def __init__(
        self,
        problem: FiniteDecisionProblem,
        channel: Channel | ChannelWitness,
    ) -> None:
        """Re-evaluate a channel on its bound finite decision problem."""

        from infinite_rulebook.artifacts import semantic_hash
        from infinite_rulebook.frontier.finite_problem import (
            ChannelWitness,
            FiniteDecisionProblem,
        )

        if not isinstance(problem, FiniteDecisionProblem):
            raise TypeError("problem must be a FiniteDecisionProblem")
        raw_channel = (
            channel.channel if isinstance(channel, ChannelWitness) else channel
        )
        evaluated = problem.evaluate(raw_channel)
        problem_hash = semantic_hash(problem)
        object.__setattr__(self, "problem_semantic_hash", problem_hash)
        object.__setattr__(self, "channel_witness", evaluated)
        object.__setattr__(
            self,
            "witness_hash",
            semantic_hash({"problem": problem_hash, "witness": evaluated}),
        )

    @property
    def expected_reward(self) -> float:
        return self.channel_witness.expected_reward

    @property
    def mutual_information_nats(self) -> float:
        return self.channel_witness.mutual_information


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """Certified information bounds at one expected-reward threshold."""

    reward: float
    information: MetricInterval
    upper_witness: FrontierUpperWitness
    requested_reward: float | None = None

    @classmethod
    def from_frontier_solution(
        cls,
        problem: FiniteDecisionProblem,
        solution: object,
    ) -> FrontierPoint:
        """Construct a point directly from a certified finite solver result."""

        from infinite_rulebook.frontier.inversion import FrontierSolution

        if not isinstance(solution, FrontierSolution):
            raise TypeError("solution must be a FrontierSolution")
        if solution.witness is None:
            raise ValueError("an infeasible frontier solution has no upper witness")
        from infinite_rulebook.artifacts import semantic_hash

        if solution.problem_semantic_hash != semantic_hash(problem):
            raise ValueError("frontier solution is bound to a different problem")
        return cls(
            reward=solution.effective_target_reward,
            information=MetricInterval(
                solution.lower_bound,
                solution.upper_bound,
                "nats",
            ),
            upper_witness=FrontierUpperWitness(problem, solution.witness),
            requested_reward=solution.target_reward,
        )

    def __post_init__(self) -> None:
        reward = _real("reward", self.reward)
        requested = (
            reward
            if self.requested_reward is None
            else _real("requested_reward", self.requested_reward)
        )
        if requested < reward:
            raise ValueError("requested reward cannot be below effective reward")
        if not isinstance(self.information, MetricInterval):
            raise TypeError("information must be a MetricInterval")
        if self.information.units != "nats":
            raise ValueError("frontier information must be measured in nats")
        if self.information.lower < 0.0:
            raise ValueError("frontier information cannot be negative")
        if not isinstance(self.upper_witness, FrontierUpperWitness):
            raise TypeError("upper_witness must be a FrontierUpperWitness")
        if self.upper_witness.expected_reward < reward:
            raise ValueError("upper witness does not attain the reward threshold")
        if not math.isclose(
            self.upper_witness.mutual_information_nats,
            self.information.upper,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("upper bound must equal its feasible witness information")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "requested_reward", requested)


class UpperEnvelopeCertificate(StrEnum):
    """Why interpolation between upper endpoints remains a feasible upper bound."""

    WITNESS_MIXTURE = "witness_mixture"


@dataclass(frozen=True, slots=True)
class FrontierCurve:
    """Certified point bounds with a safe step lower/chord upper envelope."""

    points: tuple[FrontierPoint, ...]
    zero_information_reward: float
    maximum_reward: float
    semantic_hash: str
    upper_certificate: UpperEnvelopeCertificate
    _rewards: tuple[float, ...] = field(
        init=False,
        repr=False,
        compare=False,
        metadata={"artifact_exclude": True},
    )

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) < 2:
            raise ValueError("a frontier curve needs at least two points")
        if any(not isinstance(point, FrontierPoint) for point in points):
            raise TypeError("points must contain FrontierPoint records")
        rewards = tuple(point.reward for point in points)
        if any(left >= right for left, right in pairwise(rewards)):
            raise ValueError("frontier rewards must be strictly increasing")
        zero = _real("zero_information_reward", self.zero_information_reward)
        maximum = _real("maximum_reward", self.maximum_reward)
        if zero >= maximum:
            raise ValueError("zero-information reward must be below maximum reward")
        if rewards[0] != zero or rewards[-1] != maximum:
            raise ValueError("frontier points must span the declared reward range")
        if points[0].information != MetricInterval(0.0, 0.0, "nats"):
            raise ValueError("zero-information endpoint must be exactly [0, 0] nats")
        for bound in ("lower", "upper"):
            values = tuple(getattr(point.information, bound) for point in points)
            if any(left > right for left, right in pairwise(values)):
                raise ValueError(f"frontier {bound} information must be nondecreasing")
        _validate_sha256("semantic_hash", self.semantic_hash)
        if not isinstance(self.upper_certificate, UpperEnvelopeCertificate):
            raise TypeError("upper_certificate must be an UpperEnvelopeCertificate")
        for point in points:
            if point.upper_witness.problem_semantic_hash != self.semantic_hash:
                raise ValueError("frontier witness is bound to a different problem")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "zero_information_reward", zero)
        object.__setattr__(self, "maximum_reward", maximum)
        object.__setattr__(self, "_rewards", rewards)


def _interpolate(left: float, right: float, weight: float) -> float:
    if left == right:
        return left
    if math.isinf(left) or math.isinf(right):
        return left if weight == 0.0 else right
    return left + weight * (right - left)


def lookup_bit_equivalent(
    curve: FrontierCurve, expected_reward: Real
) -> MetricInterval:
    """Look up certified bit-equivalent bounds for pooled expected reward."""

    if not isinstance(curve, FrontierCurve):
        raise TypeError("curve must be a FrontierCurve")
    reward = _real("expected_reward", expected_reward)
    if reward <= curve.zero_information_reward:
        return MetricInterval(0.0, 0.0, "nats")
    if reward > curve.maximum_reward:
        return MetricInterval(math.inf, math.inf, "nats")
    right_index = bisect_left(curve._rewards, reward)
    right = curve.points[right_index]
    left = curve.points[right_index - 1]
    weight = (reward - left.reward) / (right.reward - left.reward)
    return MetricInterval(
        left.information.lower if reward < right.reward else right.information.lower,
        _interpolate(left.information.upper, right.information.upper, weight),
        "nats",
    )


@dataclass(frozen=True, slots=True)
class TimedReward:
    """Pooled expected reward at an explicit elapsed-round boundary."""

    round_index: int
    expected_reward: float

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int):
            raise TypeError("round_index must be an integer")
        if self.round_index < 0:
            raise ValueError("round_index must be nonnegative")
        object.__setattr__(
            self, "expected_reward", _real("expected_reward", self.expected_reward)
        )


class CheckpointInterpolation(StrEnum):
    """Declared interpolation between sparse checkpoint estimates."""

    LEFT_HOLD = "left_hold"
    LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class BitEquivalentSeries:
    """Time-average bit-equivalent with its weighting provenance."""

    average: MetricInterval
    horizon: int
    interpolation: CheckpointInterpolation
    checkpoint_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.average, MetricInterval):
            raise TypeError("average must be a MetricInterval")
        if self.average.units != "nats" or self.average.lower < 0.0:
            raise ValueError("average must be nonnegative and measured in nats")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not isinstance(self.interpolation, CheckpointInterpolation):
            raise TypeError("interpolation must be a CheckpointInterpolation")
        if isinstance(self.checkpoint_count, bool) or not isinstance(
            self.checkpoint_count, int
        ):
            raise TypeError("checkpoint_count must be an integer")
        if not 2 <= self.checkpoint_count <= self.horizon + 1:
            raise ValueError("checkpoint_count must lie in [2, horizon + 1]")


def _segment_integral(
    curve: FrontierCurve,
    left: TimedReward,
    right: TimedReward,
    interpolation: CheckpointInterpolation,
) -> tuple[float, float]:
    duration = right.round_index - left.round_index
    if interpolation is CheckpointInterpolation.LEFT_HOLD:
        value = lookup_bit_equivalent(curve, left.expected_reward)
        return duration * value.lower, duration * value.upper

    fractions = [0.0, 1.0]
    reward_span = right.expected_reward - left.expected_reward
    if reward_span != 0.0:
        low = min(left.expected_reward, right.expected_reward)
        high = max(left.expected_reward, right.expected_reward)
        for point in curve.points:
            if low < point.reward < high:
                fractions.append((point.reward - left.expected_reward) / reward_span)
    fractions.sort()
    lower_integral = 0.0
    upper_integral = 0.0
    for start, stop in pairwise(fractions):
        start_reward = left.expected_reward + start * reward_span
        stop_reward = left.expected_reward + stop * reward_span
        start_value = lookup_bit_equivalent(curve, start_reward)
        stop_value = lookup_bit_equivalent(curve, stop_reward)
        midpoint_reward = left.expected_reward + (start + stop) / 2.0 * reward_span
        midpoint_value = lookup_bit_equivalent(curve, midpoint_reward)
        subduration = duration * (stop - start)
        lower_integral += subduration * midpoint_value.lower
        upper_integral += subduration * (start_value.upper + stop_value.upper) / 2.0
    return lower_integral, upper_integral


def integrate_bit_equivalent(
    curve: FrontierCurve,
    checkpoints: tuple[TimedReward, ...],
    *,
    horizon: int,
    interpolation: CheckpointInterpolation,
) -> BitEquivalentSeries:
    """Integrate sparse pooled rewards with explicit elapsed-round weighting."""

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not isinstance(interpolation, CheckpointInterpolation):
        raise TypeError("interpolation must be a CheckpointInterpolation")
    points = tuple(checkpoints)
    if len(points) < 2 or any(not isinstance(point, TimedReward) for point in points):
        raise ValueError("checkpoints must contain at least two TimedReward records")
    rounds = tuple(point.round_index for point in points)
    if rounds[0] != 0 or rounds[-1] != horizon:
        raise ValueError("checkpoints must span exactly [0, horizon]")
    if any(left >= right for left, right in pairwise(rounds)):
        raise ValueError("checkpoint rounds must be strictly increasing")
    lower = 0.0
    upper = 0.0
    for left, right in pairwise(points):
        segment_lower, segment_upper = _segment_integral(
            curve, left, right, interpolation
        )
        lower += segment_lower
        upper += segment_upper
    return BitEquivalentSeries(
        average=MetricInterval(lower / horizon, upper / horizon, "nats"),
        horizon=horizon,
        interpolation=interpolation,
        checkpoint_count=len(points),
    )


@dataclass(frozen=True, slots=True)
class PopulationInformationEstimate:
    """Ensemble information estimate; never a single-history Bayesian surprise."""

    reward_relevant_nats: float
    shared_core_nats: float
    persistent_distractor_nats: float
    dynamic_state_nats: float
    total_nats: float
    run_count: int

    def __post_init__(self) -> None:
        names = (
            "reward_relevant_nats",
            "shared_core_nats",
            "persistent_distractor_nats",
            "dynamic_state_nats",
            "total_nats",
        )
        values = tuple(_real(name, getattr(self, name)) for name in names)
        if any(value < 0.0 for value in values):
            raise ValueError("population information cannot be negative")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)
        if isinstance(self.run_count, bool) or not isinstance(self.run_count, int):
            raise TypeError("run_count must be an integer")
        if self.run_count < 1:
            raise ValueError("run_count must be positive")

    @property
    def relevant_nats(self) -> float:
        return self.reward_relevant_nats + self.shared_core_nats

    def validate(self, tolerance: Real = 1e-12) -> ValidationReport:
        accuracy = _real("tolerance", tolerance)
        if accuracy < 0.0:
            raise ValueError("tolerance must be nonnegative")
        subtotal = math.fsum(
            (
                self.reward_relevant_nats,
                self.shared_core_nats,
                self.persistent_distractor_nats,
                self.dynamic_state_nats,
            )
        )
        diagnostics = ()
        if not math.isclose(
            subtotal, self.total_nats, rel_tol=accuracy, abs_tol=accuracy
        ):
            diagnostics = (
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "COMPONENT_TOTAL_MISMATCH",
                    "total_nats",
                    f"component sum {subtotal!r} does not equal "
                    f"total {self.total_nats!r}",
                ),
            )
        return ValidationReport(diagnostics)


@dataclass(frozen=True, slots=True)
class EfficiencyMetric:
    """Useful-information efficiency and any scientific validity finding."""

    interval: MetricInterval | None
    validation: ValidationReport
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        tolerance = _real("tolerance", self.tolerance)
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        object.__setattr__(self, "tolerance", tolerance)
        if not isinstance(self.validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport")
        codes = {item.code for item in self.validation.diagnostics}
        if self.interval is None:
            if "EFFICIENCY_UNDEFINED" not in codes:
                raise ValueError("an undefined efficiency needs EFFICIENCY_UNDEFINED")
            return
        if not isinstance(self.interval, MetricInterval):
            raise TypeError("interval must be a MetricInterval or None")
        if self.interval.units != "ratio" or self.interval.lower < 0.0:
            raise ValueError("efficiency must be a nonnegative ratio")
        if (
            self.interval.upper > 1.0 + tolerance
            and "EFFICIENCY_OUT_OF_RANGE" not in codes
        ):
            raise ValueError("efficiency above one needs EFFICIENCY_OUT_OF_RANGE")


def useful_information_efficiency(
    bit_equivalent: MetricInterval,
    information: PopulationInformationEstimate,
    *,
    complete_history_manifest: bool,
    tolerance: Real = 1e-12,
) -> EfficiencyMetric:
    """Return ``B(E[R]) / I(Theta;H)`` without seedwise ratio averaging."""

    if bit_equivalent.units != "nats":
        raise ValueError("bit-equivalent interval must be measured in nats")
    if bit_equivalent.lower < 0.0:
        raise ValueError("bit-equivalent information cannot be negative")
    if not isinstance(information, PopulationInformationEstimate):
        raise TypeError("information must be a PopulationInformationEstimate")
    if not isinstance(complete_history_manifest, bool):
        raise TypeError("complete_history_manifest must be a boolean")
    accuracy = _real("tolerance", tolerance)
    diagnostics = list(information.validate(accuracy).diagnostics)
    if not complete_history_manifest:
        diagnostics.append(
            ValidationDiagnostic(
                DiagnosticSeverity.ERROR,
                "INCOMPLETE_HISTORY_MANIFEST",
                "complete_history_manifest",
                "the deployment history omits a Theta-dependent input",
            )
        )
    denominator = information.total_nats
    if denominator <= 0.0 or not math.isfinite(denominator):
        inconsistent = bit_equivalent.upper > accuracy and denominator <= 0.0
        diagnostics.append(
            ValidationDiagnostic(
                (
                    DiagnosticSeverity.ERROR
                    if inconsistent
                    else DiagnosticSeverity.WARNING
                ),
                ("EFFICIENCY_INCONSISTENT" if inconsistent else "EFFICIENCY_UNDEFINED"),
                "information.total_nats",
                (
                    "positive bit-equivalent cannot be supported by zero information"
                    if inconsistent
                    else "efficiency requires finite positive acquired information"
                ),
            )
        )
        if inconsistent:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.WARNING,
                    "EFFICIENCY_UNDEFINED",
                    "efficiency",
                    "the inconsistent ratio is undefined",
                )
            )
        return EfficiencyMetric(
            None,
            ValidationReport(tuple(diagnostics)),
            tolerance=accuracy,
        )
    interval = MetricInterval(
        bit_equivalent.lower / denominator,
        bit_equivalent.upper / denominator,
        "ratio",
    )
    if interval.upper > 1.0 + accuracy:
        diagnostics.append(
            ValidationDiagnostic(
                DiagnosticSeverity.ERROR,
                "EFFICIENCY_OUT_OF_RANGE",
                "efficiency",
                f"upper efficiency {interval.upper!r} exceeds one",
            )
        )
    return EfficiencyMetric(
        interval,
        ValidationReport(tuple(diagnostics)),
        tolerance=accuracy,
    )


def _reward_at_budget(
    curve: FrontierCurve,
    information_budget: float,
    bound: str,
) -> float:
    values = tuple(getattr(point.information, bound) for point in curve.points)
    if information_budget < values[0]:
        return curve.zero_information_reward
    if bound == "lower":
        for index, value in enumerate(values[1:], start=1):
            if information_budget < value:
                return curve.points[index].reward
        return curve.maximum_reward
    for index, (left, right) in enumerate(pairwise(values)):
        if information_budget < right:
            if right == left:
                return curve.points[index + 1].reward
            weight = (information_budget - left) / (right - left)
            return _interpolate(
                curve.points[index].reward, curve.points[index + 1].reward, weight
            )
    return curve.maximum_reward


def frontier_regret(
    curve: FrontierCurve,
    *,
    attained_reward: Real,
    information_budget_nats: Real,
) -> MetricInterval:
    """Return certified ``R*(C) - rho`` bounds without substantive clamping."""

    reward = _real("attained_reward", attained_reward)
    budget = _real("information_budget_nats", information_budget_nats)
    if budget < 0.0:
        raise ValueError("information budget must be nonnegative")
    reward_lower = _reward_at_budget(curve, budget, "upper")
    reward_upper = _reward_at_budget(curve, budget, "lower")
    return MetricInterval(reward_lower - reward, reward_upper - reward, "reward")


@dataclass(frozen=True, slots=True)
class FrontierRegretMetrics:
    """Frontier regret under total and reward-relevant information budgets."""

    full_information: MetricInterval
    relevant_information: MetricInterval

    def __post_init__(self) -> None:
        for name in ("full_information", "relevant_information"):
            value = getattr(self, name)
            if not isinstance(value, MetricInterval):
                raise TypeError(f"{name} must be a MetricInterval")
            if value.units != "reward":
                raise ValueError(f"{name} must be measured in reward units")

    def validate(self, tolerance: Real = 1e-12) -> ValidationReport:
        accuracy = _real("tolerance", tolerance)
        diagnostics = []
        for name in ("full_information", "relevant_information"):
            interval = getattr(self, name)
            if interval.upper < -accuracy:
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "NEGATIVE_FRONTIER_REGRET",
                        name,
                        f"upper regret bound {interval.upper!r} is negative",
                    )
                )
            elif interval.lower < -accuracy:
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.WARNING,
                        "NEGATIVE_FRONTIER_REGRET_BOUND",
                        name,
                        f"lower regret bound {interval.lower!r} is negative",
                    )
                )
        return ValidationReport(tuple(diagnostics))


@dataclass(frozen=True, slots=True)
class RewardMetrics:
    """Pooled reward summaries; lower quantiles are labeled explicitly."""

    expected_reward: float
    cumulative_reward: float
    variance: float
    lower_quantiles: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("expected_reward", "cumulative_reward", "variance"):
            object.__setattr__(self, name, _real(name, getattr(self, name)))
        if self.variance < 0.0:
            raise ValueError("variance cannot be negative")
        quantiles = []
        for raw_probability, raw_value in self.lower_quantiles:
            probability = _real("quantile probability", raw_probability)
            value = _real("quantile value", raw_value)
            if not 0.0 < probability < 0.5:
                raise ValueError("lower quantile probabilities must lie in (0, 0.5)")
            quantiles.append((probability, value))
        if len({probability for probability, _ in quantiles}) != len(quantiles):
            raise ValueError("lower quantile probabilities must be unique")
        object.__setattr__(self, "lower_quantiles", tuple(sorted(quantiles)))


@dataclass(frozen=True, slots=True)
class SupportMetrics:
    """Deployment support and outcome counts."""

    deployment_support: int
    correct_deployments: int
    incorrect_deployments: int
    abstentions: int
    mastered_independent_rules: int = 0

    def __post_init__(self) -> None:
        for name in (
            "deployment_support",
            "correct_deployments",
            "incorrect_deployments",
            "abstentions",
            "mastered_independent_rules",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.deployment_support != (
            self.correct_deployments + self.incorrect_deployments
        ):
            raise ValueError(
                "deployment support must equal correct plus incorrect deployments"
            )


@dataclass(frozen=True, slots=True)
class ComputeMetrics:
    """Scientifically meaningful operation counts; timings are runtime metadata."""

    queries: int
    environment_steps: int
    posterior_updates: int
    frontier_solver_calls: int
    deployment_evaluations: int

    def __post_init__(self) -> None:
        for name in (
            "queries",
            "environment_steps",
            "posterior_updates",
            "frontier_solver_calls",
            "deployment_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class NoveltyMetrics:
    """Non-collapsed novelty measures; fresh and persistent remain distinct."""

    observation_prediction_error: float
    compression_improvement: float
    count_novelty: float
    latent_visitation_novelty: float
    behavioral_novelty: float
    aleatoric_observation_novelty: float
    persistent_trivia_novelty: float

    def __post_init__(self) -> None:
        nonnegative = (
            "observation_prediction_error",
            "count_novelty",
            "latent_visitation_novelty",
            "behavioral_novelty",
            "aleatoric_observation_novelty",
            "persistent_trivia_novelty",
        )
        for name in (*nonnegative, "compression_improvement"):
            value = _real(name, getattr(self, name))
            if name in nonnegative and value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
