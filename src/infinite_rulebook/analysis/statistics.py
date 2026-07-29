"""Deterministic pooling and small-ensemble registered statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import pairwise

from infinite_rulebook.analysis.models import (
    Alternative,
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CheckpointObservation,
    ContrastSpec,
    EquivalenceSpec,
    GroupSelector,
    Interpolation,
    ScalingSpec,
)


@dataclass(frozen=True, slots=True)
class ExactInterval:
    lower: float
    upper: float
    coverage: float
    method: str


@dataclass(frozen=True, slots=True)
class MetricSummary:
    name: str
    count: int
    cell_count: int
    algorithm_replicas_per_environment: int
    mean: float
    median: float
    minimum: float
    maximum: float
    sample_standard_deviation: float
    standard_error: float
    median_interval: ExactInterval


@dataclass(frozen=True, slots=True)
class PoolKey:
    condition_hash: str
    agent_hash: str
    environment_kind: str
    agent_kind: str
    round_index: int


@dataclass(frozen=True, slots=True)
class PooledCheckpoint:
    key: PoolKey
    metrics: tuple[MetricSummary, ...]
    bit_equivalent_lower_nats: float
    bit_equivalent_upper_nats: float
    run_hashes: tuple[str, ...]

    def metric(self, name: str) -> MetricSummary:
        try:
            return next(item for item in self.metrics if item.name == name)
        except StopIteration as error:
            raise AnalysisError(f"pool does not contain metric {name!r}") from error


@dataclass(frozen=True, slots=True)
class ContrastResult:
    name: str
    metric: str
    left_label: str
    right_label: str
    checkpoint: int
    alternative: Alternative
    null_margin: float
    pair_count: int
    cell_pair_count: int
    differences: tuple[float, ...]
    mean_difference: float
    median_difference: float
    standardized_mean_difference: float | None
    median_interval: ExactInterval
    unadjusted_p_value: float


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    name: str
    metric: str
    left_label: str
    right_label: str
    checkpoint: int
    margin: float
    margin_source: str
    margin_provenance_hash: str
    pair_count: int
    cell_pair_count: int
    differences: tuple[float, ...]
    mean_difference: float
    median_difference: float
    median_interval: ExactInterval
    lower_tost_p_value: float
    upper_tost_p_value: float
    unadjusted_p_value: float


@dataclass(frozen=True, slots=True)
class HolmDecision:
    name: str
    raw_p_value: float
    adjusted_p_value: float
    alpha: float
    reject_null: bool


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    round_index: int
    count: int
    cell_count: int
    mean: float
    median_interval: ExactInterval | None


@dataclass(frozen=True, slots=True)
class DyadicSlope:
    start_round: int
    end_round: int
    slope: float


@dataclass(frozen=True, slots=True)
class ScalingSummary:
    name: str
    metric: str
    selector_label: str
    horizon: int
    interpolation: Interpolation
    points: tuple[ScalingPoint, ...]
    elapsed_weighted_average: float
    terminal_value: float
    terminal_per_round: float
    terminal_per_log_horizon: float
    dyadic_slopes: tuple[DyadicSlope, ...]


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def exact_median_interval(
    values: tuple[float, ...],
    *,
    alpha: float,
) -> ExactInterval:
    """Distribution-free sign interval for a population median.

    Tiny samples may not support the requested finite interval; in that case the
    honest exact interval is unbounded rather than a mislabeled min/max range.
    """

    if not values:
        raise AnalysisError("cannot summarize an empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    ordered = tuple(sorted(values))
    count = len(ordered)
    selected = 0
    tail = 0.0
    cumulative = 0
    denominator = 2**count
    for rank in range(1, count // 2 + 1):
        cumulative += math.comb(count, rank - 1)
        candidate_tail = cumulative / denominator
        if 2.0 * candidate_tail <= alpha:
            selected = rank
            tail = candidate_tail
        else:
            break
    if selected == 0:
        return ExactInterval(
            -math.inf,
            math.inf,
            1.0,
            "exact-sign-median-unbounded",
        )
    return ExactInterval(
        ordered[selected - 1],
        ordered[count - selected],
        1.0 - 2.0 * tail,
        "exact-sign-median",
    )


def summarize_metric(
    name: str,
    values: tuple[float, ...],
    *,
    interval_alpha: float,
    cell_count: int | None = None,
    algorithm_replicas_per_environment: int = 1,
) -> MetricSummary:
    if not values:
        raise AnalysisError("cannot summarize an empty metric")
    mean = _mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return MetricSummary(
        name=name,
        count=len(values),
        cell_count=len(values) if cell_count is None else cell_count,
        algorithm_replicas_per_environment=algorithm_replicas_per_environment,
        mean=mean,
        median=float(statistics.median(values)),
        minimum=min(values),
        maximum=max(values),
        sample_standard_deviation=deviation,
        standard_error=deviation / math.sqrt(len(values)),
        median_interval=exact_median_interval(values, alpha=interval_alpha),
    )


def _environment_cluster_values(
    observations: list[CheckpointObservation],
    metric: str,
) -> tuple[tuple[float, ...], int]:
    by_environment: dict[int, dict[int, CheckpointObservation]] = {}
    for item in observations:
        algorithms = by_environment.setdefault(item.environment_replica, {})
        if item.algorithm_replica in algorithms:
            raise AnalysisError("pool contains duplicate replica pairs")
        algorithms[item.algorithm_replica] = item
    algorithm_sets = {frozenset(items) for items in by_environment.values()}
    if len(algorithm_sets) != 1:
        raise AnalysisError(
            "algorithm replicas must be fully crossed over environment replicas"
        )
    algorithm_count = len(next(iter(algorithm_sets)))
    if algorithm_count == 0:
        raise AnalysisError("environment cluster contains no algorithm replicas")
    values = tuple(
        _mean(
            tuple(
                item.metric(metric)
                for _, item in sorted(by_environment[environment].items())
            )
        )
        for environment in sorted(by_environment)
    )
    return values, algorithm_count


def _pooled_metric_values(
    observations: list[CheckpointObservation],
) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], int]:
    """Transpose one pool once before summarizing every registered metric."""

    metric_names = tuple(name for name, _ in observations[0].metrics)
    by_environment: dict[
        int,
        dict[int, tuple[tuple[str, float], ...]],
    ] = {}
    for item in observations:
        if tuple(name for name, _ in item.metrics) != metric_names:
            raise AnalysisError("pool checkpoints have incompatible metric schemas")
        algorithms = by_environment.setdefault(item.environment_replica, {})
        if item.algorithm_replica in algorithms:
            raise AnalysisError("pool contains duplicate replica pairs")
        algorithms[item.algorithm_replica] = item.metrics
    algorithm_sets = {frozenset(items) for items in by_environment.values()}
    if len(algorithm_sets) != 1:
        raise AnalysisError(
            "algorithm replicas must be fully crossed over environment replicas"
        )
    algorithm_count = len(next(iter(algorithm_sets)))
    if algorithm_count == 0:
        raise AnalysisError("environment cluster contains no algorithm replicas")
    ordered_environments = tuple(
        tuple(metrics for _, metrics in sorted(by_environment[environment].items()))
        for environment in sorted(by_environment)
    )
    transposed = tuple(
        (
            name,
            tuple(
                _mean(tuple(metrics[index][1] for metrics in algorithms))
                for algorithms in ordered_environments
            ),
        )
        for index, name in enumerate(metric_names)
    )
    return transposed, algorithm_count


def pool_checkpoints(
    dataset: AnalysisDataset,
    *,
    interval_alpha: float = 0.05,
) -> tuple[PooledCheckpoint, ...]:
    """Pool only registered condition/agent/checkpoint replicates."""

    grouped: dict[
        tuple[str, str, str, str, int],
        list[CheckpointObservation],
    ] = {}
    for item in dataset.observations:
        key = (
            item.condition_hash,
            item.agent_hash,
            item.environment_kind,
            item.agent_kind,
            item.round_index,
        )
        grouped.setdefault(key, []).append(item)
    result = []
    for raw_key, observations in sorted(grouped.items()):
        pair_keys = [item.pair_key for item in observations]
        if len(set(pair_keys)) != len(pair_keys):
            raise AnalysisError("pool contains duplicate replica pairs")
        frontiers = {item.frontier for item in observations}
        if len(frontiers) != 1:
            raise AnalysisError("pool checkpoints have incompatible frontiers")
        pooled_values, algorithm_count = _pooled_metric_values(observations)
        summaries_list = []
        for name, values in pooled_values:
            summaries_list.append(
                summarize_metric(
                    name,
                    values,
                    interval_alpha=interval_alpha,
                    cell_count=len(observations),
                    algorithm_replicas_per_environment=algorithm_count,
                )
            )
        summaries = tuple(summaries_list)
        expected_reward = next(
            item.mean for item in summaries if item.name == "expected_reward"
        )
        frontier = next(iter(frontiers))
        bit_lower, bit_upper = frontier.lookup(expected_reward)
        result.append(
            PooledCheckpoint(
                key=PoolKey(*raw_key),
                metrics=summaries,
                bit_equivalent_lower_nats=bit_lower,
                bit_equivalent_upper_nats=bit_upper,
                run_hashes=tuple(sorted(item.run_hash for item in observations)),
            )
        )
    return tuple(result)


def _select_pairs(
    dataset: AnalysisDataset,
    left: GroupSelector,
    right: GroupSelector,
    checkpoint: int,
    metric: str,
) -> tuple[tuple[float, ...], int, str, str]:
    if metric.startswith("bit_equivalent"):
        raise AnalysisError(
            "bit-equivalent must be inverted after pooling and cannot be tested "
            "as seedwise paired observations"
        )

    def select(selector: GroupSelector) -> list[CheckpointObservation]:
        return [
            item
            for item in dataset.observations
            if item.round_index == checkpoint and selector.matches(item)
        ]

    left_items = select(left)
    right_items = select(right)
    if not left_items or not right_items:
        raise AnalysisError("registered comparison selector matched no checkpoints")
    for selector, items in ((left, left_items), (right, right_items)):
        groups = {(item.condition_hash, item.agent_hash) for item in items}
        if len(groups) != 1:
            raise AnalysisError(
                f"selector {selector.label!r} is ambiguous across registered groups"
            )

    def by_environment(
        items: list[CheckpointObservation],
    ) -> dict[int, dict[int, CheckpointObservation]]:
        result: dict[int, dict[int, CheckpointObservation]] = {}
        for item in items:
            algorithms = result.setdefault(item.environment_replica, {})
            if item.algorithm_replica in algorithms:
                raise AnalysisError("comparison side contains duplicate replica pairs")
            algorithms[item.algorithm_replica] = item
        return result

    left_by_environment = by_environment(left_items)
    right_by_environment = by_environment(right_items)
    if set(left_by_environment) != set(right_by_environment):
        raise AnalysisError("paired comparison has unmatched environment replicas")
    cluster_differences = []
    cell_pair_count = 0
    crossed_algorithms: frozenset[int] | None = None
    for environment_replica in sorted(left_by_environment):
        left_algorithms = left_by_environment[environment_replica]
        right_algorithms = right_by_environment[environment_replica]
        if set(left_algorithms) != set(right_algorithms):
            raise AnalysisError(
                "paired comparison has unmatched algorithm replicas within an "
                "environment"
            )
        algorithms = frozenset(left_algorithms)
        if crossed_algorithms is None:
            crossed_algorithms = algorithms
        elif algorithms != crossed_algorithms:
            raise AnalysisError(
                "algorithm replicas must be fully crossed over environment replicas"
            )
        cell_differences = tuple(
            left_algorithms[algorithm].metric(metric)
            - right_algorithms[algorithm].metric(metric)
            for algorithm in sorted(left_algorithms)
        )
        cell_pair_count += len(cell_differences)
        cluster_differences.append(_mean(cell_differences))
    return (
        tuple(cluster_differences),
        cell_pair_count,
        left.label,
        right.label,
    )


def _binomial_right_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return math.fsum(
        math.comb(trials, value) for value in range(successes, trials + 1)
    ) / (2**trials)


def exact_sign_p_value(
    values: tuple[float, ...],
    *,
    null: float,
    alternative: Alternative,
) -> float:
    above = sum(value > null for value in values)
    below = sum(value < null for value in values)
    # Boundary ties count against rejection. Dropping them is anti-conservative
    # for discrete symbolic metrics whose null distribution can have point mass.
    trials = len(values)
    if trials == 0:
        return 1.0
    if alternative is Alternative.GREATER:
        return _binomial_right_tail(above, trials)
    if alternative is Alternative.LESS:
        return _binomial_right_tail(below, trials)
    upper = _binomial_right_tail(above, trials)
    lower = _binomial_right_tail(below, trials)
    return min(1.0, 2.0 * min(upper, lower))


def evaluate_contrast(
    dataset: AnalysisDataset,
    spec: ContrastSpec,
    *,
    interval_alpha: float,
) -> ContrastResult:
    if dataset.phase is AnalysisPhase.CONFIRMATORY:
        raise AnalysisError(
            "confirmatory contrasts must run through a sealed AnalysisPlan"
        )
    return _evaluate_contrast_registered(
        dataset,
        spec,
        interval_alpha=interval_alpha,
    )


def _evaluate_contrast_registered(
    dataset: AnalysisDataset,
    spec: ContrastSpec,
    *,
    interval_alpha: float,
) -> ContrastResult:
    differences, cell_pair_count, left_label, right_label = _select_pairs(
        dataset,
        spec.left,
        spec.right,
        spec.checkpoint,
        spec.metric,
    )
    mean = _mean(differences)
    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    standardized = None if deviation == 0.0 else mean / deviation
    return ContrastResult(
        name=spec.name,
        metric=spec.metric,
        left_label=left_label,
        right_label=right_label,
        checkpoint=spec.checkpoint,
        alternative=spec.alternative,
        null_margin=spec.null_margin,
        pair_count=len(differences),
        cell_pair_count=cell_pair_count,
        differences=differences,
        mean_difference=mean,
        median_difference=float(statistics.median(differences)),
        standardized_mean_difference=standardized,
        median_interval=exact_median_interval(
            differences,
            alpha=interval_alpha,
        ),
        unadjusted_p_value=exact_sign_p_value(
            differences,
            null=spec.null_margin,
            alternative=spec.alternative,
        ),
    )


def evaluate_equivalence(
    dataset: AnalysisDataset,
    spec: EquivalenceSpec,
    *,
    interval_alpha: float,
) -> EquivalenceResult:
    if dataset.phase is AnalysisPhase.CONFIRMATORY:
        raise AnalysisError(
            "confirmatory equivalence must run through a sealed AnalysisPlan"
        )
    return _evaluate_equivalence_registered(
        dataset,
        spec,
        interval_alpha=interval_alpha,
    )


def _evaluate_equivalence_registered(
    dataset: AnalysisDataset,
    spec: EquivalenceSpec,
    *,
    interval_alpha: float,
) -> EquivalenceResult:
    differences, cell_pair_count, left_label, right_label = _select_pairs(
        dataset,
        spec.left,
        spec.right,
        spec.checkpoint,
        spec.metric,
    )
    lower = exact_sign_p_value(
        differences,
        null=-spec.margin,
        alternative=Alternative.GREATER,
    )
    upper = exact_sign_p_value(
        differences,
        null=spec.margin,
        alternative=Alternative.LESS,
    )
    return EquivalenceResult(
        name=spec.name,
        metric=spec.metric,
        left_label=left_label,
        right_label=right_label,
        checkpoint=spec.checkpoint,
        margin=spec.margin,
        margin_source=spec.margin_source.value,
        margin_provenance_hash=spec.margin_provenance_hash,
        pair_count=len(differences),
        cell_pair_count=cell_pair_count,
        differences=differences,
        mean_difference=_mean(differences),
        median_difference=float(statistics.median(differences)),
        median_interval=exact_median_interval(
            differences,
            alpha=interval_alpha,
        ),
        lower_tost_p_value=lower,
        upper_tost_p_value=upper,
        unadjusted_p_value=max(lower, upper),
    )


def holm_adjust(
    p_values: tuple[tuple[str, float], ...],
    *,
    alpha: float,
) -> tuple[HolmDecision, ...]:
    """Return deterministic Holm step-down adjusted p-values and decisions."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if len({name for name, _ in p_values}) != len(p_values):
        raise ValueError("Holm family names must be unique")
    if any(not 0.0 <= value <= 1.0 for _, value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * value))
        adjusted[name] = running
    raw = dict(p_values)
    return tuple(
        HolmDecision(
            name=name,
            raw_p_value=raw[name],
            adjusted_p_value=adjusted[name],
            alpha=alpha,
            reject_null=adjusted[name] <= alpha,
        )
        for name in sorted(raw)
    )


def _scaling_value(
    items: list[CheckpointObservation],
    metric: str,
    *,
    interval_alpha: float,
) -> tuple[float, ExactInterval | None, int]:
    if metric in {"bit_equivalent_lower_nats", "bit_equivalent_upper_nats"}:
        frontiers = {item.frontier for item in items}
        if len(frontiers) != 1:
            raise AnalysisError("scaling group has incompatible frontiers")
        rewards, _ = _environment_cluster_values(items, "expected_reward")
        reward = _mean(rewards)
        bounds = next(iter(frontiers)).lookup(reward)
        return bounds[metric == "bit_equivalent_upper_nats"], None, len(rewards)
    values, _ = _environment_cluster_values(items, metric)
    return (
        _mean(values),
        exact_median_interval(values, alpha=interval_alpha),
        len(values),
    )


def summarize_scaling(
    dataset: AnalysisDataset,
    spec: ScalingSpec,
    *,
    interval_alpha: float = 0.05,
) -> ScalingSummary:
    if dataset.phase is AnalysisPhase.CONFIRMATORY:
        raise AnalysisError(
            "confirmatory scaling must run through a sealed AnalysisPlan"
        )
    return _summarize_scaling_registered(
        dataset,
        spec,
        interval_alpha=interval_alpha,
    )


def _summarize_scaling_registered(
    dataset: AnalysisDataset,
    spec: ScalingSpec,
    *,
    interval_alpha: float = 0.05,
) -> ScalingSummary:
    selected = [
        item
        for item in dataset.observations
        if item.round_index <= spec.horizon and spec.selector.matches(item)
    ]
    if not selected:
        raise AnalysisError("scaling selector matched no checkpoints")
    groups = {(item.condition_hash, item.agent_hash) for item in selected}
    if len(groups) != 1:
        raise AnalysisError("scaling selector is ambiguous across registered groups")
    by_round: dict[int, list[CheckpointObservation]] = {}
    for item in selected:
        by_round.setdefault(item.round_index, []).append(item)
    if 0 not in by_round or spec.horizon not in by_round:
        raise AnalysisError("scaling checkpoints must span exactly [0, horizon]")
    pair_sets = {
        round_index: {item.pair_key for item in items}
        for round_index, items in by_round.items()
    }
    if len({frozenset(value) for value in pair_sets.values()}) != 1:
        raise AnalysisError("scaling trajectory has incomplete replica histories")
    points = []
    for round_index, items in sorted(by_round.items()):
        if round_index > spec.horizon:
            continue
        value, interval, cluster_count = _scaling_value(
            items,
            spec.metric,
            interval_alpha=interval_alpha,
        )
        points.append(
            ScalingPoint(
                round_index,
                cluster_count,
                len(items),
                value,
                interval,
            )
        )
    if points[-1].round_index != spec.horizon:
        raise AnalysisError("scaling trajectory does not end at its horizon")
    area = 0.0
    for left, right in pairwise(points):
        duration = right.round_index - left.round_index
        if spec.interpolation is Interpolation.LEFT_HOLD:
            area += duration * left.mean
        else:
            area += duration * (left.mean + right.mean) / 2.0
    by_index = {point.round_index: point.mean for point in points}
    slopes = []
    for end_round, end_value in sorted(by_index.items()):
        if end_round <= 0 or end_round % 2:
            continue
        start_round = end_round // 2
        start_value = by_index.get(start_round)
        if (
            start_round > 0
            and start_value is not None
            and min(start_value, end_value) > 0
        ):
            slopes.append(
                DyadicSlope(
                    start_round,
                    end_round,
                    math.log(end_value / start_value) / math.log(2.0),
                )
            )
    terminal = points[-1].mean
    return ScalingSummary(
        name=spec.name,
        metric=spec.metric,
        selector_label=spec.selector.label,
        horizon=spec.horizon,
        interpolation=spec.interpolation,
        points=tuple(points),
        elapsed_weighted_average=area / spec.horizon,
        terminal_value=terminal,
        terminal_per_round=terminal / spec.horizon,
        terminal_per_log_horizon=terminal / math.log1p(spec.horizon),
        dyadic_slopes=tuple(slopes),
    )


__all__ = [
    "ContrastResult",
    "DyadicSlope",
    "EquivalenceResult",
    "ExactInterval",
    "HolmDecision",
    "MetricSummary",
    "PoolKey",
    "PooledCheckpoint",
    "ScalingPoint",
    "ScalingSummary",
    "evaluate_contrast",
    "evaluate_equivalence",
    "exact_median_interval",
    "exact_sign_p_value",
    "holm_adjust",
    "pool_checkpoints",
    "summarize_metric",
    "summarize_scaling",
]
