"""Canonical machine-readable and Markdown registered-analysis reports."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    AnalysisPlan,
    ContrastInterpretation,
)
from infinite_rulebook.analysis.statistics import (
    ContrastResult,
    EquivalenceResult,
    HolmDecision,
    PooledCheckpoint,
    ScalingSummary,
    _evaluate_contrast_registered,
    _evaluate_equivalence_registered,
    _summarize_scaling_registered,
    holm_adjust,
    pool_checkpoints,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash


def _payload(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _payload(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "-infinity"
    return value


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    phase: AnalysisPhase
    dataset_hash: str
    plan: AnalysisPlan
    pools: tuple[PooledCheckpoint, ...]
    contrasts: tuple[ContrastResult, ...]
    equivalences: tuple[EquivalenceResult, ...]
    scaling: tuple[ScalingSummary, ...]
    family_decisions: tuple[HolmDecision, ...]
    equivalence_decisions: tuple[HolmDecision, ...] = ()
    run_settings_hash: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    canary_report_hash: str | None = None
    canaries_passed: bool | None = None
    config_hash: str | None = None
    deviation_log_hash: str | None = None
    deviation_count: int = 0

    def __post_init__(self) -> None:
        if self.run_settings_hash is not None and not is_sha256(self.run_settings_hash):
            raise ValueError("run_settings_hash must be a SHA-256 digest or None")
        if (self.canary_report_hash is None) != (self.canaries_passed is None):
            raise ValueError(
                "canary_report_hash and canaries_passed must be supplied together"
            )
        if self.canary_report_hash is not None and not is_sha256(
            self.canary_report_hash
        ):
            raise ValueError("canary_report_hash must be a SHA-256 digest or None")
        if self.canaries_passed is not None and not isinstance(
            self.canaries_passed,
            bool,
        ):
            raise TypeError("canaries_passed must be a boolean or None")
        if self.config_hash is not None and not is_sha256(self.config_hash):
            raise ValueError("config_hash must be a SHA-256 digest or None")
        if self.deviation_log_hash is not None and not is_sha256(
            self.deviation_log_hash
        ):
            raise ValueError("deviation_log_hash must be a SHA-256 digest or None")
        if (
            isinstance(self.deviation_count, bool)
            or not isinstance(self.deviation_count, int)
            or self.deviation_count < 0
        ):
            raise ValueError("deviation_count must be a nonnegative integer")
        if self.deviation_log_hash is None and self.deviation_count:
            raise ValueError("a nonzero deviation_count requires a deviation log")

    @property
    def interpretation_eligible(self) -> bool:
        return (
            self.phase is AnalysisPhase.CONFIRMATORY
            and self.canaries_passed is True
            and self.deviation_log_hash is not None
            and self.deviation_count == 0
        )

    @property
    def registered_family_passed(self) -> bool:
        return (
            self.interpretation_eligible
            and all(item.reject_null for item in self.family_decisions)
            and all(item.reject_null for item in self.equivalence_decisions)
        )

    def _scientific_payload(self) -> dict[str, Any]:
        return {
            "artifact_type": "registered-analysis-report",
            "schema_version": 2,
            "phase": self.phase.value,
            "dataset_hash": self.dataset_hash,
            "inference_scope": {
                "environment_replicas": "independent-clusters",
                "algorithm_replicas": "conditional-fixed-seed-bank",
            },
            "plan_hash": self.plan.scientific_hash,
            "analysis_registration_hash": self.plan.registration_hash,
            "plan": _payload(self.plan),
            "pools": _payload(self.pools),
            "contrasts": _payload(self.contrasts),
            "equivalences": _payload(self.equivalences),
            "scaling": _payload(self.scaling),
            "family_decisions": _payload(self.family_decisions),
            "equivalence_decisions": _payload(self.equivalence_decisions),
            "run_settings_hash": self.run_settings_hash,
            "provenance": dict(self.provenance),
            "canary_report_hash": self.canary_report_hash,
            "canaries_passed": self.canaries_passed,
            "config_hash": self.config_hash,
            "deviation_log_hash": self.deviation_log_hash,
            "deviation_count": self.deviation_count,
            "interpretation_eligible": self.interpretation_eligible,
            "registered_family_passed": self.registered_family_passed,
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(
            self._scientific_payload(),
            domain="registered-analysis-report",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._scientific_payload(),
            "scientific_hash": self.scientific_hash,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_payload(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        contrast_decision_by_name = {item.name: item for item in self.family_decisions}
        equivalence_decision_by_name = {
            item.name: item for item in self.equivalence_decisions
        }
        lines = [
            f"# {self.plan.name}",
            "",
            f"- Phase: `{self.phase.value}`",
            f"- Dataset hash: `{self.dataset_hash}`",
            f"- Analysis-plan hash: `{self.plan.scientific_hash}`",
            f"- Analysis-registration hash: `{self.plan.registration_hash}`",
            f"- Report hash: `{self.scientific_hash}`",
        ]
        if self.config_hash is not None:
            lines.append(f"- Experiment-config hash: `{self.config_hash}`")
        if self.run_settings_hash is not None:
            lines.append(f"- Run-settings hash: `{self.run_settings_hash}`")
        if self.provenance:
            provenance = dict(self.provenance)
            lines.extend(
                [
                    f"- Source commit: `{provenance['code_commit']}`",
                    f"- Analysis-source hash: `{provenance['analysis_code_hash']}`",
                    f"- Dependency-lock hash: `{provenance['dependency_lock_hash']}`",
                    "- Execution-environment hash: "
                    f"`{provenance['environment_digest']}`",
                ]
            )
        if self.canaries_passed is not None:
            lines.append(f"- Canary-report hash: `{self.canary_report_hash}`")
        if self.deviation_log_hash is not None:
            lines.extend(
                [
                    f"- Deviation-log hash: `{self.deviation_log_hash}`",
                    f"- Registered deviations: {self.deviation_count}",
                ]
            )
        if self.phase is AnalysisPhase.CALIBRATION:
            lines.append(
                "- Interpretive status: **descriptive calibration; not confirmatory**"
            )
        elif self.interpretation_eligible:
            lines.append(
                "- Interpretive status: **eligible under all deterministic gates**"
            )
        else:
            blockers = []
            if self.canaries_passed is not True:
                blockers.append("deterministic canaries did not pass")
            if self.deviation_count:
                blockers.append("the registered protocol has deviations")
            if not blockers:
                blockers.append("deterministic gate evidence is incomplete")
            lines.append(
                "- Interpretive status: **BLOCKED: " + "; ".join(blockers) + "**"
            )
        if self.plan.freeze_hash is not None:
            lines.append(f"- Confirmatory freeze: `{self.plan.freeze_hash}`")
        lines.extend(
            [
                "",
                "## Pooled checkpoints",
                "",
                "| Environment | Agent | Condition | Agent config | Round | "
                "n env (cells) | Reward | Relevant nats | Total nats | "
                "Bit-equivalent nats |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pool in self.pools:
            metrics = {item.name: item for item in pool.metrics}
            reward = metrics["expected_reward"].mean
            relevant = metrics.get("relevant_information_nats")
            total = metrics.get("total_information_nats")
            reward_summary = metrics["expected_reward"]
            lines.append(
                "| "
                f"{pool.key.environment_kind} | {pool.key.agent_kind} | "
                f"`{pool.key.condition_hash[:12]}` | "
                f"`{pool.key.agent_hash[:12]}` | {pool.key.round_index} | "
                f"{reward_summary.count} "
                f"({reward_summary.cell_count}) | "
                f"{_number(reward)} | "
                f"{_number(relevant.mean) if relevant else '—'} | "
                f"{_number(total.mean) if total else '—'} | "
                f"[{_number(pool.bit_equivalent_lower_nats)}, "
                f"{_number(pool.bit_equivalent_upper_nats)}] |"
            )
        lines.extend(
            [
                "",
                "## Registered contrasts",
                "",
                "| Name | Metric | Registered test | n env (cells) | "
                "Mean difference | Exact median interval | Raw p | Holm p | "
                "Decision |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for result in self.contrasts:
            decision = contrast_decision_by_name[result.name]
            interval = _interval(
                result.median_interval.lower,
                result.median_interval.upper,
            )
            specification = next(
                (item for item in self.plan.contrasts if item.name == result.name),
                None,
            )
            if specification is None:
                raise AnalysisError(
                    f"contrast result {result.name!r} is absent from the plan"
                )
            missing_gates = tuple(
                name
                for name in specification.required_equivalence_gates
                if not equivalence_decision_by_name[name].reject_null
            )
            telemetry_only = (
                specification.interpretation is ContrastInterpretation.TELEMETRY_ONLY
            )
            if self.plan.phase is AnalysisPhase.CALIBRATION:
                conclusion = (
                    "descriptive calibration telemetry only"
                    if telemetry_only
                    else "descriptive calibration result; not confirmatory"
                )
            elif not self.interpretation_eligible:
                conclusion = "blocked by deterministic interpretation gates"
            elif telemetry_only:
                conclusion = "telemetry-only diagnostic; no construct claim"
            elif not decision.reject_null:
                conclusion = "not resolved"
            elif missing_gates:
                conclusion = "directional test passes; required equivalence unresolved"
            else:
                conclusion = "supports registered alternative"
            lines.append(
                f"| {result.name} | {result.metric} | "
                f"{result.alternative.value} vs {_number(result.null_margin)} | "
                f"{result.pair_count} ({result.cell_pair_count}) | "
                f"{_number(result.mean_difference)} | "
                f"{interval} | "
                f"{_number(result.unadjusted_p_value)} | "
                f"{_number(decision.adjusted_p_value)} | "
                f"{conclusion} |"
            )
        lines.extend(
            [
                "",
                "## Registered equivalence checks",
                "",
                "| Name | Metric | Margin | n env (cells) | Mean difference | "
                "Exact median interval | Raw TOST p | Holm p | Decision |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for result in self.equivalences:
            decision = equivalence_decision_by_name[result.name]
            interval = _interval(
                result.median_interval.lower,
                result.median_interval.upper,
            )
            if self.plan.phase is AnalysisPhase.CALIBRATION:
                conclusion = "descriptive calibration result; not confirmatory"
            elif not self.interpretation_eligible:
                conclusion = "blocked by deterministic interpretation gates"
            elif decision.reject_null:
                conclusion = "equivalent within frozen margin"
            else:
                conclusion = "not resolved"
            lines.append(
                f"| {result.name} | {result.metric} | ±{_number(result.margin)} | "
                f"{result.pair_count} ({result.cell_pair_count}) | "
                f"{_number(result.mean_difference)} | "
                f"{interval} | "
                f"{_number(result.unadjusted_p_value)} | "
                f"{_number(decision.adjusted_p_value)} | "
                f"{conclusion} |"
            )
        lines.extend(
            [
                "",
                "## Scaling summaries",
                "",
                "**Descriptive only:** the fixed horizon does not identify a scaling "
                "class or establish open-endedness.",
                "",
            ]
        )
        if self.scaling:
            lines.extend(
                [
                    "| Name | Metric | Horizon | Weighted average | Terminal | "
                    "Terminal/T | Terminal/log(1+T) |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for result in self.scaling:
                lines.append(
                    f"| {result.name} | {result.metric} | {result.horizon} | "
                    f"{_number(result.elapsed_weighted_average)} | "
                    f"{_number(result.terminal_value)} | "
                    f"{_number(result.terminal_per_round)} | "
                    f"{_number(result.terminal_per_log_horizon)} |"
                )
        else:
            lines.append("No scaling summaries were registered.")
        lines.extend(
            [
                "",
                "## Analysis contract",
                "",
                "- Reward and information are pooled across complete registered "
                "condition/agent/checkpoint ensembles.",
                "- Bit-equivalent bounds invert pooled reward through the "
                "authenticated certified frontier; seedwise nonlinear ratios are "
                "not averaged.",
                "- Paired algorithm-replica differences are averaged within each "
                "environment replica. Exact sign tests and distribution-free median "
                "intervals use environment replicas as independent clusters.",
                "- Contrast p-values test the registered cluster-median sign null; "
                "reported mean differences and standardized means are descriptive.",
                "- The S5 sign estimands are probabilities of strict paired "
                "differences under the registered common-random-number coupling. "
                "They are not claims of marginal mean equality; ties count against "
                "the asserted direction or equivalence endpoint.",
                "- IND and RED-C share algorithm/query noise but not latent labels. "
                "S5 information and reward conclusions are conditional on this "
                "registered coupling.",
                "- Inference is conditional on the registered fixed algorithm-seed "
                "bank; algorithm replicas are not counted as independent population "
                "replicates.",
                "- The registered superiority family uses Holm adjustment. "
                "Equivalence gates form a separate registered family, with margins "
                "frozen externally and never estimated from confirmatory outcomes.",
                "",
            ]
        )
        return "\n".join(lines)


def _number(value: float) -> str:
    if math.isinf(value):
        return "infinity" if value > 0.0 else "-infinity"
    return format(value, ".8g")


def _interval(lower: float, upper: float) -> str:
    return f"[{_number(lower)}, {_number(upper)}]"


def _validate_phase_contract(
    dataset: AnalysisDataset,
    plan: AnalysisPlan,
) -> None:
    if dataset.phase is not plan.phase:
        raise AnalysisError(
            f"analysis plan phase {plan.phase.value!r} does not match "
            f"dataset phase {dataset.phase.value!r}"
        )
    if plan.phase is AnalysisPhase.CONFIRMATORY:
        if any(
            not item.confirmatory_frozen
            or item.freeze_hash != plan.freeze_hash
            or item.analysis_registration_hash != plan.registration_hash
            for item in dataset.observations
        ):
            raise AnalysisError(
                "confirmatory data do not all match the freeze seal and registered "
                "analysis hash"
            )
    elif any(
        item.confirmatory_frozen
        or item.freeze_hash is not None
        or item.analysis_registration_hash is not None
        for item in dataset.observations
    ):
        raise AnalysisError("pilot/calibration data cannot carry confirmatory bindings")


def _validate_inventory(
    dataset: AnalysisDataset,
    plan: AnalysisPlan,
) -> None:
    if not plan.expected_groups:
        return

    def expected_identities():
        for group in plan.expected_groups:
            for checkpoint in group.checkpoints:
                for environment in range(group.environment_replicas):
                    for algorithm in range(group.algorithm_replicas):
                        yield (
                            group.condition_hash,
                            group.agent_hash,
                            group.agent_kind,
                            checkpoint,
                            environment,
                            algorithm,
                            group.environment_kind,
                        )

    def actual_identities():
        previous = None
        for item in dataset.observations:
            identity = (
                item.condition_hash,
                item.agent_hash,
                item.agent_kind,
                item.round_index,
                item.environment_replica,
                item.algorithm_replica,
                item.environment_kind,
            )
            if identity == previous:
                raise AnalysisError(
                    "dataset contains duplicate registered checkpoint identities"
                )
            previous = identity
            yield identity

    expected = iter(expected_identities())
    actual = iter(actual_identities())
    expected_item = next(expected, None)
    actual_item = next(actual, None)
    missing = 0
    unexpected = 0
    while expected_item is not None or actual_item is not None:
        if expected_item is None:
            unexpected += 1
            actual_item = next(actual, None)
        elif actual_item is None:
            missing += 1
            expected_item = next(expected, None)
        elif expected_item == actual_item:
            expected_item = next(expected, None)
            actual_item = next(actual, None)
        elif expected_item < actual_item:
            missing += 1
            expected_item = next(expected, None)
        else:
            unexpected += 1
            actual_item = next(actual, None)
    if missing or unexpected:
        raise AnalysisError(
            "dataset does not match the frozen run inventory "
            f"({missing} missing, {unexpected} unexpected checkpoints)"
        )


def build_report(
    dataset: AnalysisDataset,
    plan: AnalysisPlan,
    *,
    canary_report_hash: str | None = None,
    canaries_passed: bool | None = None,
    config_hash: str | None = None,
) -> AnalysisReport:
    """Evaluate exactly the frozen registrations; no data-driven tuning occurs."""

    _validate_phase_contract(dataset, plan)
    _validate_inventory(dataset, plan)
    pools = pool_checkpoints(dataset, interval_alpha=plan.interval_alpha)
    contrasts = tuple(
        _evaluate_contrast_registered(
            dataset,
            spec,
            interval_alpha=plan.interval_alpha,
        )
        for spec in plan.contrasts
    )
    equivalences = tuple(
        _evaluate_equivalence_registered(
            dataset,
            spec,
            interval_alpha=plan.interval_alpha,
        )
        for spec in plan.equivalences
    )
    contrast_p_values = tuple(
        (result.name, result.unadjusted_p_value) for result in contrasts
    )
    decisions = holm_adjust(contrast_p_values, alpha=plan.family_alpha)
    equivalence_decisions = holm_adjust(
        tuple((result.name, result.unadjusted_p_value) for result in equivalences),
        alpha=plan.family_alpha,
    )
    scaling = tuple(
        _summarize_scaling_registered(
            dataset,
            spec,
            interval_alpha=plan.interval_alpha,
        )
        for spec in plan.scalings
    )
    return AnalysisReport(
        phase=plan.phase,
        dataset_hash=dataset.scientific_hash,
        plan=plan,
        pools=pools,
        contrasts=contrasts,
        equivalences=equivalences,
        scaling=scaling,
        family_decisions=decisions,
        equivalence_decisions=equivalence_decisions,
        run_settings_hash=dataset.run_settings_hash,
        provenance=dataset.provenance,
        canary_report_hash=canary_report_hash,
        canaries_passed=canaries_passed,
        config_hash=config_hash,
    )


__all__ = ["AnalysisReport", "build_report"]
