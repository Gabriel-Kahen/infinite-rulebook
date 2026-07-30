"""Deterministic dependency-free exports for registered study reports."""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest

from infinite_rulebook.analysis.canaries import (
    CanaryKind,
    CanaryReport,
    ConstantAdditiveMetricResult,
    ExactZeroMetricResult,
    FrontierIdentityResult,
    MetricTrajectoryIdentityResult,
)
from infinite_rulebook.analysis.models import (
    Alternative,
    AnalysisPhase,
    AnalysisPlan,
    ContrastInterpretation,
    ContrastSpec,
    EquivalenceSpec,
    ExpectedGroup,
    GroupSelector,
    Interpolation,
    MarginSource,
)
from infinite_rulebook.analysis.power import (
    CandidatePower,
    EffectAdequacy,
    EquivalencePowerBound,
    EquivalencePowerHypothesis,
    HypothesisPower,
    PowerCalibration,
    PowerHypothesis,
)
from infinite_rulebook.analysis.reporting import AnalysisReport
from infinite_rulebook.analysis.statistics import (
    ContrastResult,
    EquivalenceResult,
    ExactInterval,
    HolmDecision,
    MetricSummary,
    PooledCheckpoint,
    PoolKey,
    ScalingPoint,
    ScalingSummary,
)
from infinite_rulebook.analysis.visualization import (
    canary_results_csv,
    power_calibration_csv,
    registered_gates_csv,
    terminal_summary_csv,
    trajectories_svg,
)


def _metric(name: str, mean: float) -> MetricSummary:
    return MetricSummary(
        name=name,
        count=4,
        cell_count=12,
        algorithm_replicas_per_environment=3,
        mean=mean,
        median=mean,
        minimum=mean - 0.1,
        maximum=mean + 0.1,
        sample_standard_deviation=0.1,
        standard_error=0.05,
        median_interval=ExactInterval(
            mean - 0.1,
            mean + 0.1,
            0.875,
            "exact-sign-median",
        ),
    )


def _pool(round_index: int, mean: float) -> PooledCheckpoint:
    return PooledCheckpoint(
        key=PoolKey(
            condition_hash="a" * 64,
            agent_hash="b" * 64,
            environment_kind="IND",
            agent_kind="reward",
            round_index=round_index,
        ),
        metrics=(
            _metric("expected_reward", mean),
            _metric("relevant_information_nats", mean + 0.5),
        ),
        bit_equivalent_lower_nats=mean / 2,
        bit_equivalent_upper_nats=mean,
        run_hashes=("c" * 64,),
    )


def _report(*, include_scaling: bool = True) -> AnalysisReport:
    scaling = ()
    if include_scaling:
        scaling = (
            ScalingSummary(
                name="Reward <trajectory>",
                metric="hidden_expected_reward",
                selector_label="IND/reward/*/*",
                horizon=12,
                interpolation=Interpolation.LEFT_HOLD,
                points=(
                    ScalingPoint(
                        0,
                        4,
                        12,
                        0.0,
                        ExactInterval(
                            float("-inf"),
                            float("inf"),
                            1.0,
                            "exact-sign-median-unbounded",
                        ),
                    ),
                    ScalingPoint(
                        12,
                        4,
                        12,
                        2.0,
                        ExactInterval(1.8, 2.2, 0.875, "exact-sign-median"),
                    ),
                ),
                elapsed_weighted_average=0.0,
                terminal_value=2.0,
                terminal_per_round=1 / 6,
                terminal_per_log_horizon=0.8,
                dyadic_slopes=(),
            ),
        )
    return AnalysisReport(
        phase=AnalysisPhase.CALIBRATION,
        dataset_hash="d" * 64,
        plan=AnalysisPlan(
            "Symbolic <study>",
            AnalysisPhase.CALIBRATION,
        ),
        pools=(_pool(0, 0.0), _pool(12, 2.0)),
        contrasts=(),
        equivalences=(),
        scaling=scaling,
        family_decisions=(),
    )


def _gated_report() -> AnalysisReport:
    base = _report()
    left = GroupSelector(environment_kind="IND", agent_kind="scheduled")
    right = GroupSelector(environment_kind="IND", agent_kind="fixed")
    equivalence_left = GroupSelector(
        environment_kind="IND",
        agent_kind="reward",
    )
    equivalence_right = GroupSelector(
        environment_kind="ALEA",
        agent_kind="reward",
    )
    plan = AnalysisPlan(
        base.plan.name,
        base.phase,
        contrasts=(
            ContrastSpec(
                "reward-direction",
                "hidden_expected_reward",
                left,
                right,
                12,
                Alternative.GREATER,
            ),
        ),
        equivalences=(
            EquivalenceSpec(
                "control-equivalence",
                "hidden_expected_reward",
                equivalence_left,
                equivalence_right,
                12,
                0.05,
                MarginSource.CALIBRATION,
                "e" * 64,
            ),
        ),
    )
    contrast = ContrastResult(
        name="reward-direction",
        metric="hidden_expected_reward",
        left_label="IND/scheduled",
        right_label="IND/fixed",
        checkpoint=12,
        alternative=Alternative.GREATER,
        null_margin=0.25,
        pair_count=16,
        cell_pair_count=48,
        differences=(0.4,) * 16,
        mean_difference=0.4,
        median_difference=0.4,
        standardized_mean_difference=2.0,
        median_interval=ExactInterval(0.3, 0.5, 0.95, "exact-sign-median"),
        unadjusted_p_value=0.001,
    )
    equivalence = EquivalenceResult(
        name="control-equivalence",
        metric="hidden_expected_reward",
        left_label="IND/reward",
        right_label="ALEA/reward",
        checkpoint=12,
        margin=0.05,
        margin_source="calibration",
        margin_provenance_hash="e" * 64,
        pair_count=16,
        cell_pair_count=48,
        differences=(0.0,) * 16,
        mean_difference=0.0,
        median_difference=0.0,
        median_interval=ExactInterval(-0.01, 0.01, 0.95, "exact-sign-median"),
        lower_tost_p_value=0.001,
        upper_tost_p_value=0.001,
        unadjusted_p_value=0.001,
    )
    return AnalysisReport(
        phase=base.phase,
        dataset_hash=base.dataset_hash,
        plan=plan,
        pools=base.pools,
        contrasts=(contrast,),
        equivalences=(equivalence,),
        scaling=base.scaling,
        family_decisions=(HolmDecision("reward-direction", 0.001, 0.001, 0.05, True),),
        equivalence_decisions=(
            HolmDecision("control-equivalence", 0.001, 0.001, 0.05, True),
        ),
    )


def _confirmatory_gated_report(
    *,
    canaries_passed: bool,
    deviation_count: int,
) -> AnalysisReport:
    base = _gated_report()
    plan = replace(
        base.plan,
        phase=AnalysisPhase.CONFIRMATORY,
        frozen=True,
        freeze_hash="f" * 64,
        expected_groups=(
            ExpectedGroup(
                "0" * 64,
                "1" * 64,
                "IND",
                "scheduled",
                (12,),
                1,
                1,
            ),
        ),
    )
    return replace(
        base,
        phase=AnalysisPhase.CONFIRMATORY,
        plan=plan,
        canary_report_hash="c" * 64,
        canaries_passed=canaries_passed,
        deviation_log_hash="d" * 64,
        deviation_count=deviation_count,
    )


def _canary_report() -> CanaryReport:
    return CanaryReport(
        phase=AnalysisPhase.CALIBRATION,
        dataset_hash="d" * 64,
        plan_hash="e" * 64,
        results=(
            FrontierIdentityResult(
                "frontier",
                CanaryKind.FRONTIER_IDENTITY,
                True,
                4,
                12,
                13,
                0,
                (),
            ),
            MetricTrajectoryIdentityResult(
                "trajectory",
                CanaryKind.METRIC_IDENTITY,
                "hidden_expected_reward",
                True,
                1e-12,
                4,
                12,
                13,
                0,
                0.0,
                (),
            ),
            ConstantAdditiveMetricResult(
                "decomposition",
                CanaryKind.ADDITIVE_SHIFT,
                "expected_reward",
                "hidden_expected_reward",
                "public_reward",
                0.5,
                True,
                1e-12,
                4,
                12,
                13,
                0,
                0.0,
                0.0,
                (),
                (),
            ),
            ExactZeroMetricResult(
                "exact-zero",
                CanaryKind.EXACT_ZERO,
                "distractor_information_nats",
                True,
                4,
                12,
                13,
                0,
                0.0,
                (),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("canaries_passed", "deviation_count"),
    ((False, 0), (True, 1)),
)
def test_deterministic_gates_block_confirmatory_claims(
    canaries_passed: bool,
    deviation_count: int,
) -> None:
    report = _confirmatory_gated_report(
        canaries_passed=canaries_passed,
        deviation_count=deviation_count,
    )

    assert not report.interpretation_eligible
    assert not report.registered_family_passed
    assert "supports registered alternative" not in report.to_markdown()
    assert "equivalent within frozen margin" not in report.to_markdown()
    rows = list(csv.DictReader(io.StringIO(registered_gates_csv(report))))
    assert rows
    assert {row["interpretation_eligible"] for row in rows} == {"false"}


def test_clean_confirmatory_gates_allow_registered_claims() -> None:
    report = _confirmatory_gated_report(
        canaries_passed=True,
        deviation_count=0,
    )

    assert report.interpretation_eligible
    assert report.registered_family_passed
    assert "supports registered alternative" in report.to_markdown()
    assert "equivalent within frozen margin" in report.to_markdown()


def test_telemetry_role_is_explicit_and_never_interpretation_eligible() -> None:
    base = _confirmatory_gated_report(
        canaries_passed=True,
        deviation_count=0,
    )
    plan = replace(
        base.plan,
        contrasts=(
            replace(
                base.plan.contrasts[0],
                interpretation=ContrastInterpretation.TELEMETRY_ONLY,
            ),
        ),
    )
    report = replace(base, plan=plan)

    row = next(csv.DictReader(io.StringIO(registered_gates_csv(report))))
    assert row["interpretation_role"] == "telemetry-only"
    assert row["interpretation_eligible"] == "false"
    assert "supports registered alternative" not in report.to_markdown()
    assert "telemetry-only" in report.to_markdown()


def _power_calibration() -> PowerCalibration:
    def directional(name: str, diagnostic: int, certified: float) -> HypothesisPower:
        return HypothesisPower(
            name,
            diagnostic,
            10_000,
            favorable_sign_successes=6,
            favorable_sign_trials=6,
            favorable_sign_probability_lower_bound=0.9,
            certified_power_lower_bound=certified,
        )

    def equivalence(diagnostic: int, certified: float) -> EquivalencePowerBound:
        return EquivalencePowerBound(
            "reward-equivalence",
            diagnostic,
            10_000,
            lower_successes=6,
            upper_successes=6,
            favorable_sign_trials=6,
            lower_probability_bound=0.9,
            upper_probability_bound=0.9,
            lower_test_power_bound=0.98,
            upper_test_power_bound=0.98,
            certified_power_lower_bound=certified,
        )

    return PowerCalibration(
        candidates=(
            CandidatePower(
                4,
                (
                    directional("information", 6000, 0.6),
                    directional("reward", 5000, 0.5),
                ),
                equivalence_hypotheses=(equivalence(5000, 0.5),),
                registered_joint_rejections=3500,
                directional_global_null_rejections=1000,
                equivalence_lower_boundary_rejections=1000,
                equivalence_upper_boundary_rejections=1000,
                simulations=10_000,
                certified_registered_joint_power_lower_bound=0.3,
                directional_global_null_fwer_upper_bound=0.05,
                equivalence_lower_boundary_error_upper_bound=0.05,
                equivalence_upper_boundary_error_upper_bound=0.05,
                meets_targets=False,
            ),
            CandidatePower(
                8,
                (
                    directional("information", 9600, 0.96),
                    directional("reward", 9500, 0.95),
                ),
                equivalence_hypotheses=(equivalence(9500, 0.95),),
                registered_joint_rejections=8800,
                directional_global_null_rejections=100,
                equivalence_lower_boundary_rejections=100,
                equivalence_upper_boundary_rejections=100,
                simulations=10_000,
                certified_registered_joint_power_lower_bound=0.85,
                directional_global_null_fwer_upper_bound=0.05,
                equivalence_lower_boundary_error_upper_bound=0.05,
                equivalence_upper_boundary_error_upper_bound=0.05,
                meets_targets=True,
            ),
        ),
        selected_environment_count=8,
        calibration_environment_count=12,
        center_environment_count=6,
        probability_environment_count=6,
        algorithm_replicas_per_environment=3,
        effect_adequacy=(
            EffectAdequacy(
                "information",
                "directional",
                0.5,
                0.75,
                0.5,
                None,
                6,
                True,
            ),
            EffectAdequacy(
                "reward",
                "directional",
                0.25,
                0.5,
                0.25,
                None,
                6,
                True,
            ),
            EffectAdequacy(
                "reward-equivalence",
                "equivalence",
                -0.1,
                0.1,
                -0.25,
                0.25,
                6,
                True,
            ),
        ),
        simulations=10_000,
        alpha=0.05,
        minimum_power=0.9,
        minimum_joint_power=0.8,
        maximum_fwer=0.05,
    )


def _power_hypotheses() -> tuple[PowerHypothesis, ...]:
    return tuple(
        PowerHypothesis.from_cluster_differences(
            name,
            (0.0,) * 12,
            minimum_effect=minimum_effect,
            algorithm_replicas_per_environment=3,
        )
        for name, minimum_effect in (
            ("information", 0.5),
            ("reward", 0.25),
        )
    )


def _equivalence_power_hypotheses() -> tuple[EquivalencePowerHypothesis, ...]:
    return (
        EquivalencePowerHypothesis.from_cluster_differences(
            "reward-equivalence",
            (0.0,) * 12,
            margin=0.25,
            diagnostic_location=0.0,
            algorithm_replicas_per_environment=3,
        ),
    )


def test_terminal_summary_is_deterministic_long_form_and_terminal_only() -> None:
    report = _report()

    rendered = terminal_summary_csv(report)
    rows = list(csv.DictReader(io.StringIO(rendered)))

    assert rendered == terminal_summary_csv(report)
    assert len(rows) == 2
    assert {row["terminal_round"] for row in rows} == {"12"}
    assert {row["metric"] for row in rows} == {
        "expected_reward",
        "relevant_information_nats",
    }
    assert {row["phase"] for row in rows} == {"calibration"}
    assert {row["environment_clusters"] for row in rows} == {"4"}
    assert {row["algorithm_cells"] for row in rows} == {"12"}
    assert {row["report_hash"] for row in rows} == {report.scientific_hash}
    assert all("independent clusters" in row["inference_scope"] for row in rows)


def test_trajectory_svg_is_deterministic_valid_accessible_and_caveated() -> None:
    report = _report()

    rendered = trajectories_svg(report)

    assert rendered == trajectories_svg(report)
    ET.fromstring(rendered)
    assert 'role="img"' in rendered
    assert 'aria-labelledby="figure-title figure-description"' in rendered
    assert "<desc " in rendered
    assert "Symbolic &lt;study&gt;" in rendered
    assert "Reward &lt;trajectory&gt;" in rendered
    assert "environment replicas are independent clusters" in rendered
    assert "algorithm replicas are averaged within" in rendered
    assert "trajectories alone do not establish a registered contrast" in rendered
    assert "<polyline " in rendered
    assert "<circle " in rendered
    assert "exact median interval unbounded" in rendered
    assert report.scientific_hash in rendered


def test_trajectory_svg_discloses_when_no_scaling_was_registered() -> None:
    rendered = trajectories_svg(_report(include_scaling=False))

    ET.fromstring(rendered)
    assert "No scaling trajectories were registered." in rendered
    assert "<polyline " not in rendered


def test_registered_gate_table_distinguishes_directional_and_equivalence() -> None:
    report = _gated_report()

    rendered = registered_gates_csv(report)
    rows = list(csv.DictReader(io.StringIO(rendered)))
    by_type = {row["gate_type"]: row for row in rows}

    assert rendered == registered_gates_csv(report)
    assert set(by_type) == {"directional-contrast", "equivalence"}
    assert by_type["directional-contrast"]["alternative"] == "greater"
    assert (
        by_type["directional-contrast"]["multiplicity_family"] == "primary-directional"
    )
    assert by_type["directional-contrast"]["null_margin"] == "0.25"
    assert by_type["directional-contrast"]["gate_passed"] == "true"
    assert "not proof" in by_type["directional-contrast"]["gate_semantics"]
    assert by_type["equivalence"]["equivalence_margin"] == "0.05"
    assert by_type["equivalence"]["multiplicity_family"] == "equivalence"
    assert by_type["equivalence"]["margin_provenance_hash"] == "e" * 64
    assert "externally frozen margin" in by_type["equivalence"]["gate_semantics"]
    assert "separately Holm-adjusted" in by_type["equivalence"]["gate_semantics"]
    assert all(row["phase"] == "calibration" for row in rows)
    assert all(row["report_hash"] == report.scientific_hash for row in rows)
    assert all("independent clusters" in row["inference_scope"] for row in rows)


def test_canary_table_preserves_each_exact_gate_rule_without_p_values() -> None:
    report = _canary_report()

    rendered = canary_results_csv(report)
    rows = list(csv.DictReader(io.StringIO(rendered)))
    by_kind = {row["kind"]: row for row in rows}

    assert rendered == canary_results_csv(report)
    assert len(rows) == 4
    assert all(row["gate_passed"] == "true" for row in rows)
    assert all(row["canary_report_hash"] == report.scientific_hash for row in rows)
    assert "p_value" not in rows[0]
    metric = by_kind["metric-trajectory-identity"]
    assert metric["metric"] == "hidden_expected_reward"
    assert metric["tolerance"] == "1e-12"
    additive = by_kind["constant-additive-metric-shift"]
    assert additive["expected_shift"] == "0.5"
    assert additive["maximum_decomposition_error"] == "0.0"
    exact = by_kind["exact-zero-metric"]
    assert "no tolerance or p-value" in exact["gate_semantics"]


def test_power_table_is_candidate_by_hypothesis_and_labels_design_evidence() -> None:
    report = _report()
    calibration = _power_calibration()

    rendered = power_calibration_csv(
        report,
        calibration,
        _power_hypotheses(),
        equivalence_hypotheses=_equivalence_power_hypotheses(),
        calibration_hash="f" * 64,
    )
    rows = list(csv.DictReader(io.StringIO(rendered)))

    assert rendered == power_calibration_csv(
        report,
        calibration,
        _power_hypotheses(),
        equivalence_hypotheses=_equivalence_power_hypotheses(),
        calibration_hash="f" * 64,
    )
    assert len(rows) == 6
    assert {row["candidate_environment_count"] for row in rows} == {"4", "8"}
    assert {row["hypothesis_name"] for row in rows} == {
        "information",
        "reward",
        "reward-equivalence",
    }
    assert {
        row["hypothesis_name"]: row["hypothesis_minimum_effect"]
        for row in rows
        if row["hypothesis_family"] == "directional"
    } == {"information": "0.5", "reward": "0.25"}
    assert {row["hypothesis_alternative"] for row in rows} == {
        "greater",
        "exact-sign-tost",
    }
    equivalence_rows = [
        row for row in rows if row["hypothesis_family"] == "equivalence"
    ]
    assert {row["equivalence_margin"] for row in equivalence_rows} == {"0.25"}
    assert {row["certified_equivalence_population"] for row in equivalence_rows} == {
        "raw-held-out-calibration-population"
    }
    assert {row["diagnostic_equivalence_location"] for row in equivalence_rows} == {
        "0.0"
    }
    assert {
        row["candidate_environment_count"]
        for row in rows
        if row["candidate_is_selected"] == "true"
    } == {"8"}
    assert all(row["selected_environment_count"] == "8" for row in rows)
    assert all(row["phase"] == "calibration" for row in rows)
    assert all(row["power_calibration_hash"] == "f" * 64 for row in rows)
    assert {row["simulation_error_alpha"] for row in rows} == {"0.01"}
    assert {row["simultaneous_decision_count"] for row in rows} == {"14"}
    assert all(float(row["simulation_error_bound"]) > 0.0 for row in rows)
    assert all(
        float(row["hypothesis_power_lower_bound"]) <= float(row["hypothesis_power"])
        for row in rows
    )
    assert {row["directional_global_null_fwer_upper_bound"] for row in rows} == {"0.05"}
    assert {row["equivalence_lower_boundary_error_upper_bound"] for row in rows} == {
        "0.05"
    }
    assert all(
        "Split-sample, distribution-free certification" in row["design_semantics"]
        for row in rows
    )
    assert all("diagnostic only" in row["design_semantics"] for row in rows)


def test_power_table_rejects_unbound_calibration_hash() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        power_calibration_csv(
            _report(),
            _power_calibration(),
            _power_hypotheses(),
            equivalence_hypotheses=_equivalence_power_hypotheses(),
            calibration_hash="not-a-hash",
        )
