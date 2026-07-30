"""Deterministic, dependency-free tabular and SVG study outputs."""

from __future__ import annotations

import csv
import html
import io
import math

from infinite_rulebook.analysis.canaries import (
    CanaryKind,
    CanaryReport,
    ConstantAdditiveMetricResult,
    ExactZeroMetricResult,
    FrontierIdentityResult,
    MetricTrajectoryIdentityResult,
)
from infinite_rulebook.analysis.models import ContrastInterpretation
from infinite_rulebook.analysis.power import (
    EquivalencePowerHypothesis,
    PowerCalibration,
    PowerHypothesis,
)
from infinite_rulebook.analysis.reporting import AnalysisReport
from infinite_rulebook.analysis.statistics import (
    ContrastResult,
    EquivalenceResult,
    PooledCheckpoint,
    ScalingSummary,
)
from infinite_rulebook.orchestration.hashing import is_sha256

CANARY_RESULTS_FILENAME = "canary-results.csv"
POWER_CALIBRATION_FILENAME = "power-calibration.csv"
REGISTERED_GATES_FILENAME = "registered-gates.csv"
TERMINAL_SUMMARY_FILENAME = "terminal-summary.csv"
TRAJECTORIES_FILENAME = "trajectories.svg"

_CSV_FIELDS = (
    "plan_name",
    "phase",
    "report_hash",
    "dataset_hash",
    "environment_kind",
    "agent_kind",
    "condition_hash",
    "agent_hash",
    "terminal_round",
    "metric",
    "environment_clusters",
    "algorithm_cells",
    "algorithm_replicas_per_environment",
    "cluster_mean",
    "cluster_median",
    "cluster_minimum",
    "cluster_maximum",
    "cluster_sample_standard_deviation",
    "cluster_standard_error",
    "exact_median_interval_lower",
    "exact_median_interval_upper",
    "exact_median_interval_coverage",
    "exact_median_interval_method",
    "bit_equivalent_lower_nats",
    "bit_equivalent_upper_nats",
    "inference_scope",
)

_SCOPE = (
    "Descriptive pooled summary: environment replicas are independent clusters; "
    "algorithm replicas are averaged within each environment cluster."
)

_INFERENCE_SCOPE = (
    "Environment replicas are independent clusters; paired algorithm-replica "
    "differences are averaged within environment; inference is conditional on the "
    "fixed algorithm-seed bank."
)

_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)


def _number(value: float) -> str:
    if math.isinf(value):
        return "infinity" if value > 0.0 else "-infinity"
    return repr(float(value))


def _boolean(value: bool) -> str:
    return "true" if value else "false"


def _csv(
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _terminal_pools(
    pools: tuple[PooledCheckpoint, ...],
) -> tuple[PooledCheckpoint, ...]:
    groups: dict[tuple[str, str, str, str], PooledCheckpoint] = {}
    for pool in pools:
        key = (
            pool.key.condition_hash,
            pool.key.agent_hash,
            pool.key.environment_kind,
            pool.key.agent_kind,
        )
        current = groups.get(key)
        if current is None or pool.key.round_index > current.key.round_index:
            groups[key] = pool
    return tuple(groups[key] for key in sorted(groups))


def terminal_summary_csv(report: AnalysisReport) -> str:
    """Return a canonical long-form table of terminal cluster summaries."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=_CSV_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for pool in _terminal_pools(report.pools):
        for metric in sorted(pool.metrics, key=lambda item: item.name):
            writer.writerow(
                {
                    "plan_name": report.plan.name,
                    "phase": report.phase.value,
                    "report_hash": report.scientific_hash,
                    "dataset_hash": report.dataset_hash,
                    "environment_kind": pool.key.environment_kind,
                    "agent_kind": pool.key.agent_kind,
                    "condition_hash": pool.key.condition_hash,
                    "agent_hash": pool.key.agent_hash,
                    "terminal_round": pool.key.round_index,
                    "metric": metric.name,
                    "environment_clusters": metric.count,
                    "algorithm_cells": metric.cell_count,
                    "algorithm_replicas_per_environment": (
                        metric.algorithm_replicas_per_environment
                    ),
                    "cluster_mean": _number(metric.mean),
                    "cluster_median": _number(metric.median),
                    "cluster_minimum": _number(metric.minimum),
                    "cluster_maximum": _number(metric.maximum),
                    "cluster_sample_standard_deviation": _number(
                        metric.sample_standard_deviation
                    ),
                    "cluster_standard_error": _number(metric.standard_error),
                    "exact_median_interval_lower": _number(
                        metric.median_interval.lower
                    ),
                    "exact_median_interval_upper": _number(
                        metric.median_interval.upper
                    ),
                    "exact_median_interval_coverage": _number(
                        metric.median_interval.coverage
                    ),
                    "exact_median_interval_method": metric.median_interval.method,
                    "bit_equivalent_lower_nats": _number(
                        pool.bit_equivalent_lower_nats
                    ),
                    "bit_equivalent_upper_nats": _number(
                        pool.bit_equivalent_upper_nats
                    ),
                    "inference_scope": _SCOPE,
                }
            )
    return output.getvalue()


def _gate_row(
    report: AnalysisReport,
    result: ContrastResult | EquivalenceResult,
    *,
    gate_type: str,
) -> dict[str, object]:
    decisions = (
        report.family_decisions
        if isinstance(result, ContrastResult)
        else report.equivalence_decisions
    )
    matches = tuple(item for item in decisions if item.name == result.name)
    if len(matches) != 1:
        raise ValueError(
            f"registered gate {result.name!r} needs exactly one decision in its "
            "multiplicity family"
        )
    decision = matches[0]
    required_gates: tuple[str, ...] = ()
    required_gates_passed = True
    interpretation_role = ContrastInterpretation.INFERENTIAL
    if isinstance(result, ContrastResult):
        specification = next(
            item for item in report.plan.contrasts if item.name == result.name
        )
        required_gates = specification.required_equivalence_gates
        interpretation_role = specification.interpretation
        equivalence_decisions = {
            item.name: item.reject_null for item in report.equivalence_decisions
        }
        required_gates_passed = all(
            equivalence_decisions[name] for name in required_gates
        )
    interpretation_eligible = (
        decision.reject_null
        and required_gates_passed
        and report.interpretation_eligible
        and interpretation_role is ContrastInterpretation.INFERENTIAL
    )
    common: dict[str, object] = {
        "plan_name": report.plan.name,
        "phase": report.phase.value,
        "report_hash": report.scientific_hash,
        "dataset_hash": report.dataset_hash,
        "analysis_plan_hash": report.plan.scientific_hash,
        "analysis_registration_hash": report.plan.registration_hash,
        "gate_type": gate_type,
        "multiplicity_family": (
            "primary-directional"
            if isinstance(result, ContrastResult)
            else "equivalence"
        ),
        "name": result.name,
        "metric": result.metric,
        "checkpoint": result.checkpoint,
        "left_group": result.left_label,
        "right_group": result.right_label,
        "environment_clusters": result.pair_count,
        "algorithm_cell_pairs": result.cell_pair_count,
        "mean_difference": _number(result.mean_difference),
        "median_difference": _number(result.median_difference),
        "exact_median_interval_lower": _number(result.median_interval.lower),
        "exact_median_interval_upper": _number(result.median_interval.upper),
        "exact_median_interval_coverage": _number(result.median_interval.coverage),
        "exact_median_interval_method": result.median_interval.method,
        "unadjusted_p_value": _number(result.unadjusted_p_value),
        "holm_adjusted_p_value": _number(decision.adjusted_p_value),
        "family_alpha": _number(decision.alpha),
        "gate_passed": _boolean(decision.reject_null),
        "required_equivalence_gates": ";".join(required_gates),
        "required_equivalence_gates_passed": _boolean(required_gates_passed),
        "deterministic_canaries_passed": (
            "" if report.canaries_passed is None else _boolean(report.canaries_passed)
        ),
        "interpretation_role": interpretation_role.value,
        "interpretation_eligible": _boolean(interpretation_eligible),
        "inference_scope": _INFERENCE_SCOPE,
        "alternative": "",
        "null_margin": "",
        "standardized_mean_difference": "",
        "equivalence_margin": "",
        "margin_source": "",
        "margin_provenance_hash": "",
        "lower_tost_p_value": "",
        "upper_tost_p_value": "",
        "gate_semantics": "",
    }
    if isinstance(result, ContrastResult):
        common.update(
            {
                "alternative": result.alternative.value,
                "null_margin": _number(result.null_margin),
                "standardized_mean_difference": (
                    ""
                    if result.standardized_mean_difference is None
                    else _number(result.standardized_mean_difference)
                ),
                "gate_semantics": (
                    "Telemetry-only registered contrast; pass is descriptive "
                    "construct telemetry and supports no acquisition or capability "
                    "claim."
                    if interpretation_role is ContrastInterpretation.TELEMETRY_ONLY
                    else "Pass means the Holm-adjusted exact sign test rejects the "
                    "registered cluster-median sign null at family alpha; it is not "
                    "proof of the alternative."
                ),
            }
        )
    else:
        common.update(
            {
                "equivalence_margin": _number(result.margin),
                "margin_source": result.margin_source,
                "margin_provenance_hash": result.margin_provenance_hash,
                "lower_tost_p_value": _number(result.lower_tost_p_value),
                "upper_tost_p_value": _number(result.upper_tost_p_value),
                "gate_semantics": (
                    "Pass means the separately Holm-adjusted two-one-sided exact "
                    "sign family supports equivalence within the externally frozen "
                    "margin at its family alpha."
                ),
            }
        )
    return common


def registered_gates_csv(report: AnalysisReport) -> str:
    """Return registered directional and equivalence decisions in one table."""

    fields = (
        "plan_name",
        "phase",
        "report_hash",
        "dataset_hash",
        "analysis_plan_hash",
        "analysis_registration_hash",
        "gate_type",
        "multiplicity_family",
        "name",
        "metric",
        "checkpoint",
        "left_group",
        "right_group",
        "environment_clusters",
        "algorithm_cell_pairs",
        "alternative",
        "null_margin",
        "equivalence_margin",
        "margin_source",
        "margin_provenance_hash",
        "mean_difference",
        "median_difference",
        "standardized_mean_difference",
        "exact_median_interval_lower",
        "exact_median_interval_upper",
        "exact_median_interval_coverage",
        "exact_median_interval_method",
        "unadjusted_p_value",
        "lower_tost_p_value",
        "upper_tost_p_value",
        "holm_adjusted_p_value",
        "family_alpha",
        "gate_passed",
        "required_equivalence_gates",
        "required_equivalence_gates_passed",
        "deterministic_canaries_passed",
        "interpretation_role",
        "interpretation_eligible",
        "gate_semantics",
        "inference_scope",
    )
    rows = [
        _gate_row(report, result, gate_type="directional-contrast")
        for result in sorted(report.contrasts, key=lambda item: item.name)
    ]
    rows.extend(
        _gate_row(report, result, gate_type="equivalence")
        for result in sorted(report.equivalences, key=lambda item: item.name)
    )
    return _csv(fields, rows)


def _canary_semantics(kind: CanaryKind) -> str:
    return {
        CanaryKind.FRONTIER_IDENTITY: (
            "Pass iff every registered paired cell/checkpoint shares one canonical "
            "frontier semantic hash."
        ),
        CanaryKind.METRIC_IDENTITY: (
            "Pass iff every registered paired metric residual is within the "
            "registered absolute tolerance."
        ),
        CanaryKind.ADDITIVE_SHIFT: (
            "Pass iff every registered decomposition and shift residual is within "
            "the registered absolute tolerance."
        ),
        CanaryKind.EXACT_ZERO: (
            "Pass iff every registered metric value is exactly zero; no tolerance "
            "or p-value is used."
        ),
    }[kind]


def canary_results_csv(report: CanaryReport) -> str:
    """Return one auditable row per deterministic scientific canary gate."""

    fields = (
        "phase",
        "canary_report_hash",
        "dataset_hash",
        "canary_plan_hash",
        "all_canaries_passed",
        "name",
        "kind",
        "gate_passed",
        "metric",
        "total_metric",
        "base_metric",
        "shift_metric",
        "expected_shift",
        "tolerance",
        "environment_clusters",
        "cell_count",
        "cell_unit",
        "checkpoint_count",
        "violation_count",
        "maximum_absolute_error",
        "maximum_decomposition_error",
        "maximum_shift_error",
        "gate_semantics",
    )
    rows = []
    for result in sorted(report.results, key=lambda item: (item.name, item.kind.value)):
        row: dict[str, object] = {
            "phase": report.phase.value,
            "canary_report_hash": report.scientific_hash,
            "dataset_hash": report.dataset_hash,
            "canary_plan_hash": report.plan_hash,
            "all_canaries_passed": _boolean(report.passed),
            "name": result.name,
            "kind": result.kind.value,
            "gate_passed": _boolean(result.passed),
            "metric": "",
            "total_metric": "",
            "base_metric": "",
            "shift_metric": "",
            "expected_shift": "",
            "tolerance": "",
            "environment_clusters": result.environment_cluster_count,
            "cell_count": "",
            "cell_unit": "",
            "checkpoint_count": result.checkpoint_count,
            "violation_count": "",
            "maximum_absolute_error": "",
            "maximum_decomposition_error": "",
            "maximum_shift_error": "",
            "gate_semantics": _canary_semantics(result.kind),
        }
        if isinstance(result, FrontierIdentityResult):
            row.update(
                {
                    "cell_count": result.cell_pair_count,
                    "cell_unit": "paired condition cells",
                    "violation_count": result.mismatch_count,
                }
            )
        elif isinstance(result, MetricTrajectoryIdentityResult):
            row.update(
                {
                    "metric": result.metric,
                    "tolerance": _number(result.tolerance),
                    "cell_count": result.cell_pair_count,
                    "cell_unit": "paired condition cells",
                    "violation_count": result.mismatch_count,
                    "maximum_absolute_error": _number(result.maximum_absolute_error),
                }
            )
        elif isinstance(result, ConstantAdditiveMetricResult):
            row.update(
                {
                    "total_metric": result.total_metric,
                    "base_metric": result.base_metric,
                    "shift_metric": result.shift_metric,
                    "expected_shift": _number(result.expected_shift),
                    "tolerance": _number(result.tolerance),
                    "cell_count": result.cell_count,
                    "cell_unit": "condition cells",
                    "violation_count": result.mismatch_count,
                    "maximum_decomposition_error": _number(
                        result.maximum_decomposition_error
                    ),
                    "maximum_shift_error": _number(result.maximum_shift_error),
                }
            )
        elif isinstance(result, ExactZeroMetricResult):
            row.update(
                {
                    "metric": result.metric,
                    "cell_count": result.cell_count,
                    "cell_unit": "condition cells",
                    "violation_count": result.nonzero_count,
                    "maximum_absolute_error": _number(result.maximum_absolute_value),
                }
            )
        else:
            raise TypeError(f"unsupported canary result: {type(result).__name__}")
        rows.append(row)
    return _csv(fields, rows)


def power_calibration_csv(
    report: AnalysisReport,
    calibration: PowerCalibration,
    hypotheses: tuple[PowerHypothesis, ...],
    *,
    equivalence_hypotheses: tuple[EquivalencePowerHypothesis, ...] = (),
    calibration_hash: str,
) -> str:
    """Return candidate-by-hypothesis operating characteristics."""

    if not is_sha256(calibration_hash):
        raise ValueError("calibration_hash must be a SHA-256 digest")
    directional_specs = {item.name: item for item in hypotheses}
    equivalence_specs = {item.name: item for item in equivalence_hypotheses}
    if len(directional_specs) != len(hypotheses):
        raise ValueError("power hypotheses must have unique names")
    if len(equivalence_specs) != len(equivalence_hypotheses):
        raise ValueError("equivalence power hypotheses must have unique names")
    if set(directional_specs) & set(equivalence_specs):
        raise ValueError("directional and equivalence hypothesis names must differ")
    candidate_directional_names = {
        item.name
        for candidate in calibration.candidates
        for item in candidate.hypotheses
    }
    candidate_equivalence_names = {
        item.name
        for candidate in calibration.candidates
        for item in candidate.equivalence_hypotheses
    }
    if (
        set(directional_specs) != candidate_directional_names
        or set(equivalence_specs) != candidate_equivalence_names
    ):
        raise ValueError(
            "power hypotheses do not match the calibrated candidate family"
        )
    if any(
        set(item.name for item in candidate.hypotheses) != candidate_directional_names
        or set(item.name for item in candidate.equivalence_hypotheses)
        != candidate_equivalence_names
        for candidate in calibration.candidates
    ):
        raise ValueError("power candidates do not share registered hypothesis families")
    if any(
        len(item.clusters) != calibration.calibration_environment_count
        or item.algorithm_replicas != calibration.algorithm_replicas_per_environment
        for item in (*hypotheses, *equivalence_hypotheses)
    ):
        raise ValueError(
            "power hypothesis source clusters do not match calibration metadata"
        )
    fields = (
        "phase",
        "power_calibration_hash",
        "analysis_report_hash",
        "dataset_hash",
        "analysis_plan_hash",
        "analysis_registration_hash",
        "candidate_environment_count",
        "selected_environment_count",
        "candidate_is_selected",
        "hypothesis_name",
        "hypothesis_family",
        "hypothesis_alternative",
        "hypothesis_null_value",
        "hypothesis_minimum_effect",
        "equivalence_margin",
        "certified_equivalence_population",
        "diagnostic_equivalence_location",
        "hypothesis_rejections",
        "simulations",
        "conditional_working_model_power",
        "simulation_error_alpha",
        "simultaneous_decision_count",
        "simulation_error_bound",
        "hypothesis_power",
        "hypothesis_power_lower_bound",
        "favorable_sign_successes",
        "favorable_sign_trials",
        "favorable_sign_probability_lower_bound",
        "lower_tost_successes",
        "upper_tost_successes",
        "lower_tost_probability_bound",
        "upper_tost_probability_bound",
        "lower_tost_power_bound",
        "upper_tost_power_bound",
        "design_confidence_alpha",
        "simultaneous_design_event_count",
        "per_event_confidence_alpha",
        "effect_interval_lower",
        "effect_interval_upper",
        "effect_adequacy_passed",
        "minimum_hypothesis_power_target",
        "hypothesis_meets_target",
        "registered_joint_rejections",
        "registered_joint_power",
        "registered_joint_power_lower_bound",
        "minimum_registered_joint_power_target",
        "registered_joint_power_meets_target",
        "directional_global_null_rejections",
        "directional_global_null_fwer",
        "directional_global_null_fwer_upper_bound",
        "equivalence_lower_boundary_rejections",
        "equivalence_lower_boundary_error",
        "equivalence_lower_boundary_error_upper_bound",
        "equivalence_upper_boundary_rejections",
        "equivalence_upper_boundary_error",
        "equivalence_upper_boundary_error_upper_bound",
        "maximum_fwer_target",
        "directional_global_null_fwer_meets_target",
        "equivalence_lower_boundary_error_meets_target",
        "equivalence_upper_boundary_error_meets_target",
        "candidate_meets_all_targets",
        "calibration_environment_clusters",
        "algorithm_replicas_per_environment",
        "family_alpha",
        "design_semantics",
    )
    adequacy_by_name = {item.name: item for item in calibration.effect_adequacy}
    rows = []
    for candidate in calibration.candidates:
        records = tuple(
            ("directional", item, directional_specs[item.name])
            for item in candidate.hypotheses
        ) + tuple(
            ("equivalence", item, equivalence_specs[item.name])
            for item in candidate.equivalence_hypotheses
        )
        joint_lower = candidate.certified_registered_joint_power_lower_bound
        directional_fwer_upper = candidate.directional_global_null_fwer_upper_bound
        lower_boundary_upper = candidate.equivalence_lower_boundary_error_upper_bound
        upper_boundary_upper = candidate.equivalence_upper_boundary_error_upper_bound
        for family, hypothesis, specification in records:
            directional = isinstance(specification, PowerHypothesis)
            adequacy = adequacy_by_name[hypothesis.name]
            hypothesis_lower = hypothesis.certified_power_lower_bound
            minimum_power = (
                calibration.minimum_power
                if directional
                else calibration.minimum_equivalence_power
            )
            rows.append(
                {
                    "phase": report.phase.value,
                    "power_calibration_hash": calibration_hash,
                    "analysis_report_hash": report.scientific_hash,
                    "dataset_hash": report.dataset_hash,
                    "analysis_plan_hash": report.plan.scientific_hash,
                    "analysis_registration_hash": report.plan.registration_hash,
                    "candidate_environment_count": candidate.environment_count,
                    "selected_environment_count": (
                        ""
                        if calibration.selected_environment_count is None
                        else calibration.selected_environment_count
                    ),
                    "candidate_is_selected": _boolean(
                        candidate.environment_count
                        == calibration.selected_environment_count
                    ),
                    "hypothesis_name": hypothesis.name,
                    "hypothesis_family": family,
                    "hypothesis_alternative": (
                        specification.alternative.value
                        if directional
                        else "exact-sign-tost"
                    ),
                    "hypothesis_null_value": (
                        _number(specification.null_value) if directional else ""
                    ),
                    "hypothesis_minimum_effect": (
                        _number(specification.minimum_effect) if directional else ""
                    ),
                    "equivalence_margin": (
                        "" if directional else _number(specification.margin)
                    ),
                    "certified_equivalence_population": (
                        "" if directional else "raw-held-out-calibration-population"
                    ),
                    "diagnostic_equivalence_location": (
                        ""
                        if directional
                        else _number(specification.diagnostic_location)
                    ),
                    "hypothesis_rejections": hypothesis.rejections,
                    "simulations": hypothesis.simulations,
                    "conditional_working_model_power": _number(hypothesis.power),
                    "simulation_error_alpha": _number(
                        calibration.simulation_error_alpha
                    ),
                    "simultaneous_decision_count": (
                        calibration.simultaneous_decision_count
                    ),
                    "simulation_error_bound": _number(
                        calibration.simulation_error_bound
                    ),
                    "hypothesis_power": _number(hypothesis.power),
                    "hypothesis_power_lower_bound": _number(hypothesis_lower),
                    "favorable_sign_successes": (
                        hypothesis.favorable_sign_successes if directional else ""
                    ),
                    "favorable_sign_trials": hypothesis.favorable_sign_trials,
                    "favorable_sign_probability_lower_bound": (
                        _number(hypothesis.favorable_sign_probability_lower_bound)
                        if directional
                        else ""
                    ),
                    "lower_tost_successes": (
                        "" if directional else hypothesis.lower_successes
                    ),
                    "upper_tost_successes": (
                        "" if directional else hypothesis.upper_successes
                    ),
                    "lower_tost_probability_bound": (
                        ""
                        if directional
                        else _number(hypothesis.lower_probability_bound)
                    ),
                    "upper_tost_probability_bound": (
                        ""
                        if directional
                        else _number(hypothesis.upper_probability_bound)
                    ),
                    "lower_tost_power_bound": (
                        ""
                        if directional
                        else _number(hypothesis.lower_test_power_bound)
                    ),
                    "upper_tost_power_bound": (
                        ""
                        if directional
                        else _number(hypothesis.upper_test_power_bound)
                    ),
                    "design_confidence_alpha": _number(
                        calibration.design_confidence_alpha
                    ),
                    "simultaneous_design_event_count": (
                        calibration.simultaneous_design_event_count
                    ),
                    "per_event_confidence_alpha": _number(
                        calibration.per_event_confidence_alpha
                    ),
                    "effect_interval_lower": _number(adequacy.interval_lower),
                    "effect_interval_upper": _number(adequacy.interval_upper),
                    "effect_adequacy_passed": _boolean(adequacy.passes),
                    "minimum_hypothesis_power_target": _number(minimum_power),
                    "hypothesis_meets_target": _boolean(
                        hypothesis_lower >= minimum_power
                    ),
                    "registered_joint_rejections": (
                        candidate.registered_joint_rejections
                    ),
                    "registered_joint_power": _number(candidate.registered_joint_power),
                    "registered_joint_power_lower_bound": _number(joint_lower),
                    "minimum_registered_joint_power_target": _number(
                        calibration.minimum_joint_power
                    ),
                    "registered_joint_power_meets_target": _boolean(
                        joint_lower >= calibration.minimum_joint_power
                    ),
                    "directional_global_null_rejections": (
                        candidate.directional_global_null_rejections
                    ),
                    "directional_global_null_fwer": _number(
                        candidate.directional_global_null_fwer
                    ),
                    "directional_global_null_fwer_upper_bound": _number(
                        directional_fwer_upper
                    ),
                    "equivalence_lower_boundary_rejections": (
                        candidate.equivalence_lower_boundary_rejections
                    ),
                    "equivalence_lower_boundary_error": _number(
                        candidate.equivalence_lower_boundary_error
                    ),
                    "equivalence_lower_boundary_error_upper_bound": _number(
                        lower_boundary_upper
                    ),
                    "equivalence_upper_boundary_rejections": (
                        candidate.equivalence_upper_boundary_rejections
                    ),
                    "equivalence_upper_boundary_error": _number(
                        candidate.equivalence_upper_boundary_error
                    ),
                    "equivalence_upper_boundary_error_upper_bound": _number(
                        upper_boundary_upper
                    ),
                    "maximum_fwer_target": _number(calibration.maximum_fwer),
                    "directional_global_null_fwer_meets_target": _boolean(
                        directional_fwer_upper <= calibration.maximum_fwer
                    ),
                    "equivalence_lower_boundary_error_meets_target": _boolean(
                        lower_boundary_upper <= calibration.maximum_fwer
                    ),
                    "equivalence_upper_boundary_error_meets_target": _boolean(
                        upper_boundary_upper <= calibration.maximum_fwer
                    ),
                    "candidate_meets_all_targets": _boolean(candidate.meets_targets),
                    "calibration_environment_clusters": (
                        calibration.calibration_environment_count
                    ),
                    "algorithm_replicas_per_environment": (
                        calibration.algorithm_replicas_per_environment
                    ),
                    "family_alpha": _number(calibration.alpha),
                    "design_semantics": (
                        "Split-sample, distribution-free certification conditional "
                        "on the preregistered iid stationary simulator-seed "
                        "population. Exact order-statistic effect gates and "
                        "Clopper-Pearson favorable-sign bounds determine selection; "
                        "equivalence certification targets the raw held-out "
                        "calibration population, while the shifted cluster bootstrap "
                        "location is diagnostic only."
                    ),
                }
            )
    return _csv(fields, rows)


def _coordinate(value: float) -> str:
    return format(value, ".3f")


def _tick(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10_000 or magnitude < 0.001:
        return format(value, ".2e")
    return format(value, ".4g")


def _plot_bounds(summary: ScalingSummary) -> tuple[float, float]:
    values = [point.mean for point in summary.points]
    values.extend(
        endpoint
        for point in summary.points
        if point.median_interval is not None
        for endpoint in (
            point.median_interval.lower,
            point.median_interval.upper,
        )
        if math.isfinite(endpoint)
    )
    lower = min(0.0, *values)
    upper = max(0.0, *values)
    if lower == upper:
        return lower - 0.5, upper + 0.5
    padding = 0.08 * (upper - lower)
    return (
        lower - padding if lower < 0.0 else lower,
        upper + padding if upper > 0.0 else upper,
    )


def _panel(
    summary: ScalingSummary,
    *,
    index: int,
    left: float,
    top: float,
    width: float,
    height: float,
) -> list[str]:
    plot_left = left + 72
    plot_top = top + 66
    plot_width = width - 94
    plot_height = height - 126
    plot_bottom = plot_top + plot_height
    lower, upper = _plot_bounds(summary)
    horizon = max(summary.horizon, 1)

    def x(round_index: int) -> float:
        return plot_left + plot_width * round_index / horizon

    def y(value: float) -> float:
        return plot_bottom - plot_height * (value - lower) / (upper - lower)

    title = html.escape(summary.name)
    selector = html.escape(summary.selector_label)
    metric = html.escape(summary.metric)
    color = _PALETTE[index % len(_PALETTE)]
    panel_id = f"trajectory-panel-{index}"
    lines = [
        f'<g role="group" aria-labelledby="{panel_id}-title">',
        f'<title id="{panel_id}-title">{title}: {metric}</title>',
        (
            f'<rect x="{_coordinate(left)}" y="{_coordinate(top)}" '
            f'width="{_coordinate(width)}" height="{_coordinate(height)}" '
            'rx="8" class="panel"/>'
        ),
        (
            f'<text x="{_coordinate(left + 18)}" y="{_coordinate(top + 27)}" '
            f'class="panel-title">{title}</text>'
        ),
        (
            f'<text x="{_coordinate(left + 18)}" y="{_coordinate(top + 47)}" '
            f'class="panel-subtitle">{metric} · {selector}</text>'
        ),
    ]
    for tick_index in range(5):
        value = lower + (upper - lower) * tick_index / 4
        tick_y = y(value)
        lines.extend(
            [
                (
                    f'<line x1="{_coordinate(plot_left)}" '
                    f'y1="{_coordinate(tick_y)}" '
                    f'x2="{_coordinate(plot_left + plot_width)}" '
                    f'y2="{_coordinate(tick_y)}" class="grid"/>'
                ),
                (
                    f'<text x="{_coordinate(plot_left - 8)}" '
                    f'y="{_coordinate(tick_y + 4)}" '
                    f'class="tick y-tick">{html.escape(_tick(value))}</text>'
                ),
            ]
        )
    x_ticks = tuple(
        sorted({round(horizon * tick_index / 4) for tick_index in range(5)})
    )
    for tick_round in x_ticks:
        tick_x = x(tick_round)
        lines.extend(
            [
                (
                    f'<line x1="{_coordinate(tick_x)}" '
                    f'y1="{_coordinate(plot_bottom)}" '
                    f'x2="{_coordinate(tick_x)}" '
                    f'y2="{_coordinate(plot_bottom + 5)}" class="axis"/>'
                ),
                (
                    f'<text x="{_coordinate(tick_x)}" '
                    f'y="{_coordinate(plot_bottom + 20)}" '
                    f'class="tick x-tick">{tick_round}</text>'
                ),
            ]
        )
    lines.extend(
        [
            (
                f'<line x1="{_coordinate(plot_left)}" '
                f'y1="{_coordinate(plot_top)}" '
                f'x2="{_coordinate(plot_left)}" '
                f'y2="{_coordinate(plot_bottom)}" class="axis"/>'
            ),
            (
                f'<line x1="{_coordinate(plot_left)}" '
                f'y1="{_coordinate(plot_bottom)}" '
                f'x2="{_coordinate(plot_left + plot_width)}" '
                f'y2="{_coordinate(plot_bottom)}" class="axis"/>'
            ),
            (
                f'<text x="{_coordinate(plot_left + plot_width / 2)}" '
                f'y="{_coordinate(plot_bottom + 42)}" '
                'class="axis-label">Round</text>'
            ),
        ]
    )
    points = " ".join(
        f"{_coordinate(x(point.round_index))},{_coordinate(y(point.mean))}"
        for point in summary.points
    )
    lines.append(
        f'<polyline points="{points}" style="stroke:{color}" class="trajectory"/>'
    )
    for point in summary.points:
        interval = point.median_interval
        if (
            interval is not None
            and math.isfinite(interval.lower)
            and math.isfinite(interval.upper)
        ):
            tick_x = x(point.round_index)
            top_y = y(interval.upper)
            bottom_y = y(interval.lower)
            lines.extend(
                [
                    (
                        f'<line x1="{_coordinate(tick_x)}" '
                        f'y1="{_coordinate(top_y)}" '
                        f'x2="{_coordinate(tick_x)}" '
                        f'y2="{_coordinate(bottom_y)}" '
                        f'style="stroke:{color}" class="interval"/>'
                    ),
                    (
                        f'<line x1="{_coordinate(tick_x - 4)}" '
                        f'y1="{_coordinate(top_y)}" '
                        f'x2="{_coordinate(tick_x + 4)}" '
                        f'y2="{_coordinate(top_y)}" '
                        f'style="stroke:{color}" class="interval"/>'
                    ),
                    (
                        f'<line x1="{_coordinate(tick_x - 4)}" '
                        f'y1="{_coordinate(bottom_y)}" '
                        f'x2="{_coordinate(tick_x + 4)}" '
                        f'y2="{_coordinate(bottom_y)}" '
                        f'style="stroke:{color}" class="interval"/>'
                    ),
                ]
            )
        interval_text = (
            "unbounded"
            if interval is None
            or not math.isfinite(interval.lower)
            or not math.isfinite(interval.upper)
            else f"[{_tick(interval.lower)}, {_tick(interval.upper)}]"
        )
        lines.extend(
            [
                (
                    f'<circle cx="{_coordinate(x(point.round_index))}" '
                    f'cy="{_coordinate(y(point.mean))}" r="4" '
                    f'style="fill:{color}" class="point">'
                ),
                (
                    "<title>"
                    f"Round {point.round_index}; cluster mean {_tick(point.mean)}; "
                    f"exact median interval {html.escape(interval_text)}; "
                    f"{point.count} environment clusters; {point.cell_count} cells"
                    "</title>"
                ),
                "</circle>",
            ]
        )
    lines.append("</g>")
    return lines


def trajectories_svg(report: AnalysisReport) -> str:
    """Return an accessible small-multiple SVG for registered trajectories."""

    width = 1200
    panel_height = 300
    columns = 2
    rows = max(1, math.ceil(len(report.scaling) / columns))
    height = 132 + rows * panel_height + (rows - 1) * 22 + 88
    escaped_plan = html.escape(report.plan.name)
    escaped_phase = html.escape(report.phase.value)
    description = (
        "Registered trajectory summaries for the bounded symbolic study. Lines are "
        "environment-cluster means. Whiskers are pointwise exact median intervals "
        "where finite. Algorithm replicas are averaged within each environment "
        "cluster; trajectories alone do not establish a registered contrast."
    )
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="figure-title figure-description">'
        ),
        (
            f'<title id="figure-title">{escaped_plan}: registered trajectories '
            f"({escaped_phase})</title>"
        ),
        f'<desc id="figure-description">{html.escape(description)}</desc>',
        (
            "<metadata>"
            f"phase={escaped_phase}; report_hash={report.scientific_hash}; "
            f"dataset_hash={html.escape(report.dataset_hash)}"
            "</metadata>"
        ),
        "<style>",
        "text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #17202a; }",
        ".figure-title { font-size: 24px; font-weight: 700; }",
        ".figure-subtitle, .note { font-size: 13px; fill: #43515e; }",
        ".panel { fill: #ffffff; stroke: #c9d2d9; stroke-width: 1; }",
        ".panel-title { font-size: 15px; font-weight: 700; }",
        ".panel-subtitle { font-size: 11px; fill: #52616d; }",
        ".axis { stroke: #45535f; stroke-width: 1; }",
        ".grid { stroke: #dfe5e9; stroke-width: 1; }",
        ".tick { font-size: 10px; fill: #52616d; }",
        ".x-tick { text-anchor: middle; }",
        ".y-tick { text-anchor: end; }",
        ".axis-label { font-size: 11px; text-anchor: middle; font-weight: 600; }",
        ".trajectory { fill: none; stroke-width: 2.5; stroke-linejoin: round; }",
        ".interval { stroke-width: 1.25; }",
        ".point { stroke: #ffffff; stroke-width: 1.25; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#f5f7f8"/>',
        (
            f'<text id="visible-title" x="48" y="42" class="figure-title">'
            f"{escaped_plan}</text>"
        ),
        (
            f'<text x="48" y="67" class="figure-subtitle">Phase: '
            f"{escaped_phase} · descriptive registered trajectories</text>"
        ),
        (
            '<text x="48" y="90" class="figure-subtitle">'
            "Cluster means with pointwise exact median intervals where finite."
            "</text>"
        ),
    ]
    if report.scaling:
        panel_width = (width - 96 - 22) / columns
        for index, summary in enumerate(report.scaling):
            column = index % columns
            row = index // columns
            left = 48 + column * (panel_width + 22)
            top = 112 + row * (panel_height + 22)
            lines.extend(
                _panel(
                    summary,
                    index=index,
                    left=left,
                    top=top,
                    width=panel_width,
                    height=panel_height,
                )
            )
    else:
        lines.extend(
            [
                '<rect x="48" y="112" width="1104" height="300" rx="8" class="panel"/>',
                (
                    '<text x="600" y="262" text-anchor="middle" '
                    'class="panel-title">No scaling trajectories were registered.'
                    "</text>"
                ),
            ]
        )
    note_y = height - 50
    lines.extend(
        [
            (
                f'<text x="48" y="{note_y}" class="note">'
                "Inference scope: environment replicas are independent clusters; "
                "algorithm replicas are averaged within cluster."
                "</text>"
            ),
            (
                f'<text x="48" y="{note_y + 22}" class="note">Report '
                f"{report.scientific_hash[:12]} · figure contains no additional "
                "hypothesis tests.</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CANARY_RESULTS_FILENAME",
    "POWER_CALIBRATION_FILENAME",
    "REGISTERED_GATES_FILENAME",
    "TERMINAL_SUMMARY_FILENAME",
    "TRAJECTORIES_FILENAME",
    "canary_results_csv",
    "power_calibration_csv",
    "registered_gates_csv",
    "terminal_summary_csv",
    "trajectories_svg",
]
