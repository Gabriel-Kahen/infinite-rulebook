"""Strict, hashed JSON contract for checked-in analysis plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infinite_rulebook.analysis.models import (
    Alternative,
    AnalysisError,
    AnalysisPhase,
    AnalysisPlan,
    ContrastInterpretation,
    ContrastSpec,
    EquivalenceSpec,
    ExpectedGroup,
    GroupSelector,
    Interpolation,
    MarginSource,
    ScalingSpec,
)

ANALYSIS_PLAN_SCHEMA_VERSION = 2
_ARTIFACT_TYPE = "infinite-rulebook-analysis-plan"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnalysisError(f"{name} must be a JSON object")
    return value


def _keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise AnalysisError(f"{name} must contain exactly {sorted(expected)}")


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnalysisError(f"{name} must be a JSON array")
    return value


def _selector_payload(selector: GroupSelector) -> dict[str, str | None]:
    return {
        "environment_kind": selector.environment_kind,
        "agent_kind": selector.agent_kind,
        "condition_hash": selector.condition_hash,
        "agent_hash": selector.agent_hash,
    }


def _selector(value: object, name: str) -> GroupSelector:
    payload = _mapping(value, name)
    expected = {
        "environment_kind",
        "agent_kind",
        "condition_hash",
        "agent_hash",
    }
    _keys(payload, expected, name)
    return GroupSelector(**payload)


def analysis_plan_to_dict(plan: AnalysisPlan) -> dict[str, Any]:
    """Serialize every registration and its external provenance explicitly."""

    if not isinstance(plan, AnalysisPlan):
        raise TypeError("plan must be an AnalysisPlan")
    payload = {
        "artifact_type": _ARTIFACT_TYPE,
        "schema_version": ANALYSIS_PLAN_SCHEMA_VERSION,
        "name": plan.name,
        "phase": plan.phase.value,
        "contrasts": [
            {
                "name": item.name,
                "metric": item.metric,
                "left": _selector_payload(item.left),
                "right": _selector_payload(item.right),
                "checkpoint": item.checkpoint,
                "alternative": item.alternative.value,
                "null_margin": item.null_margin,
                "required_equivalence_gates": list(item.required_equivalence_gates),
                "interpretation": item.interpretation.value,
            }
            for item in plan.contrasts
        ],
        "equivalences": [
            {
                "name": item.name,
                "metric": item.metric,
                "left": _selector_payload(item.left),
                "right": _selector_payload(item.right),
                "checkpoint": item.checkpoint,
                "margin": item.margin,
                "margin_source": item.margin_source.value,
                "margin_provenance_hash": item.margin_provenance_hash,
            }
            for item in plan.equivalences
        ],
        "scalings": [
            {
                "name": item.name,
                "metric": item.metric,
                "selector": _selector_payload(item.selector),
                "horizon": item.horizon,
                "interpolation": item.interpolation.value,
            }
            for item in plan.scalings
        ],
        "expected_groups": [
            {
                "condition_hash": item.condition_hash,
                "agent_hash": item.agent_hash,
                "environment_kind": item.environment_kind,
                "agent_kind": item.agent_kind,
                "checkpoints": list(item.checkpoints),
                "environment_replicas": item.environment_replicas,
                "algorithm_replicas": item.algorithm_replicas,
            }
            for item in plan.expected_groups
        ],
        "family_alpha": plan.family_alpha,
        "interval_alpha": plan.interval_alpha,
        "frozen": plan.frozen,
        "freeze_hash": plan.freeze_hash,
        "registration_hash": plan.registration_hash,
        "scientific_hash": plan.scientific_hash,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload


def analysis_plan_from_dict(value: object) -> AnalysisPlan:
    """Parse a plan with fail-closed fields, enums, and scientific hash."""

    payload = _mapping(value, "analysis plan")
    expected = {
        "artifact_type",
        "schema_version",
        "name",
        "phase",
        "contrasts",
        "equivalences",
        "scalings",
        "expected_groups",
        "family_alpha",
        "interval_alpha",
        "frozen",
        "freeze_hash",
        "registration_hash",
        "scientific_hash",
    }
    _keys(payload, expected, "analysis plan")
    if payload["artifact_type"] != _ARTIFACT_TYPE:
        raise AnalysisError("analysis plan artifact_type is not recognized")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != ANALYSIS_PLAN_SCHEMA_VERSION
    ):
        raise AnalysisError("analysis plan schema_version is not supported")

    contrasts = []
    contrast_keys = {
        "name",
        "metric",
        "left",
        "right",
        "checkpoint",
        "alternative",
        "null_margin",
        "required_equivalence_gates",
        "interpretation",
    }
    for index, raw in enumerate(_array(payload["contrasts"], "contrasts")):
        item = _mapping(raw, f"contrasts[{index}]")
        _keys(item, contrast_keys, f"contrasts[{index}]")
        contrasts.append(
            ContrastSpec(
                name=item["name"],
                metric=item["metric"],
                left=_selector(item["left"], f"contrasts[{index}].left"),
                right=_selector(item["right"], f"contrasts[{index}].right"),
                checkpoint=item["checkpoint"],
                alternative=Alternative(item["alternative"]),
                null_margin=item["null_margin"],
                required_equivalence_gates=tuple(
                    _array(
                        item["required_equivalence_gates"],
                        f"contrasts[{index}].required_equivalence_gates",
                    )
                ),
                interpretation=ContrastInterpretation(item["interpretation"]),
            )
        )

    equivalences = []
    equivalence_keys = {
        "name",
        "metric",
        "left",
        "right",
        "checkpoint",
        "margin",
        "margin_source",
        "margin_provenance_hash",
    }
    for index, raw in enumerate(_array(payload["equivalences"], "equivalences")):
        item = _mapping(raw, f"equivalences[{index}]")
        _keys(item, equivalence_keys, f"equivalences[{index}]")
        equivalences.append(
            EquivalenceSpec(
                name=item["name"],
                metric=item["metric"],
                left=_selector(item["left"], f"equivalences[{index}].left"),
                right=_selector(item["right"], f"equivalences[{index}].right"),
                checkpoint=item["checkpoint"],
                margin=item["margin"],
                margin_source=MarginSource(item["margin_source"]),
                margin_provenance_hash=item["margin_provenance_hash"],
            )
        )

    scalings = []
    scaling_keys = {
        "name",
        "metric",
        "selector",
        "horizon",
        "interpolation",
    }
    for index, raw in enumerate(_array(payload["scalings"], "scalings")):
        item = _mapping(raw, f"scalings[{index}]")
        _keys(item, scaling_keys, f"scalings[{index}]")
        scalings.append(
            ScalingSpec(
                name=item["name"],
                metric=item["metric"],
                selector=_selector(item["selector"], f"scalings[{index}].selector"),
                horizon=item["horizon"],
                interpolation=Interpolation(item["interpolation"]),
            )
        )

    expected_groups = []
    expected_group_keys = {
        "condition_hash",
        "agent_hash",
        "environment_kind",
        "agent_kind",
        "checkpoints",
        "environment_replicas",
        "algorithm_replicas",
    }
    for index, raw in enumerate(_array(payload["expected_groups"], "expected_groups")):
        item = _mapping(raw, f"expected_groups[{index}]")
        _keys(item, expected_group_keys, f"expected_groups[{index}]")
        checkpoints = _array(
            item["checkpoints"],
            f"expected_groups[{index}].checkpoints",
        )
        expected_groups.append(
            ExpectedGroup(
                condition_hash=item["condition_hash"],
                agent_hash=item["agent_hash"],
                environment_kind=item["environment_kind"],
                agent_kind=item["agent_kind"],
                checkpoints=tuple(checkpoints),
                environment_replicas=item["environment_replicas"],
                algorithm_replicas=item["algorithm_replicas"],
            )
        )

    try:
        plan = AnalysisPlan(
            name=payload["name"],
            phase=AnalysisPhase(payload["phase"]),
            contrasts=tuple(contrasts),
            equivalences=tuple(equivalences),
            scalings=tuple(scalings),
            expected_groups=tuple(expected_groups),
            family_alpha=payload["family_alpha"],
            interval_alpha=payload["interval_alpha"],
            frozen=payload["frozen"],
            freeze_hash=payload["freeze_hash"],
        )
    except (TypeError, ValueError) as error:
        raise AnalysisError("analysis plan field validation failed") from error
    if payload["scientific_hash"] != plan.scientific_hash:
        raise AnalysisError("analysis plan scientific hash mismatch")
    if payload["registration_hash"] != plan.registration_hash:
        raise AnalysisError("analysis plan registration hash mismatch")
    return plan


def analysis_plan_json(plan: AnalysisPlan) -> str:
    """Return canonical human-readable JSON with a trailing newline."""

    return (
        json.dumps(
            analysis_plan_to_dict(plan),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parse_analysis_plan_json(content: str) -> AnalysisPlan:
    if not isinstance(content, str):
        raise TypeError("analysis plan JSON content must be a string")

    def reject_constant(value: str) -> None:
        raise AnalysisError(f"analysis plan JSON contains non-finite {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise AnalysisError(f"analysis plan JSON repeats key {name!r}")
            result[name] = value
        return result

    try:
        raw = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise AnalysisError("analysis plan is not valid JSON") from error
    return analysis_plan_from_dict(raw)


def load_analysis_plan(path: str | Path) -> AnalysisPlan:
    """Load and authenticate a checked-in plan without accepting JSON extensions."""

    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise AnalysisError(f"cannot read analysis plan: {path}") from error
    return parse_analysis_plan_json(content)


__all__ = [
    "ANALYSIS_PLAN_SCHEMA_VERSION",
    "analysis_plan_from_dict",
    "analysis_plan_json",
    "analysis_plan_to_dict",
    "load_analysis_plan",
    "parse_analysis_plan_json",
]
