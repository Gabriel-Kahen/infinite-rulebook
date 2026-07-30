from __future__ import annotations

import json
from dataclasses import replace

import pytest

from infinite_rulebook.analysis.canaries import (
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
)
from infinite_rulebook.analysis.compact_canaries_v2 import (
    COMPACT_CANARY_DETAIL_LIMIT,
    COMPACT_CANARY_FAILURE_LIMIT,
    AggregateMetricCanary,
    CompactCanaryDetailChunk,
    CompactCanaryEvidence,
    CompactCanaryPlan,
    compact_canary_artifacts,
    evaluate_compact_canaries,
    parse_compact_canary_detail_chunk_json,
    parse_compact_canary_plan_json,
    parse_compact_canary_report_json,
)
from infinite_rulebook.analysis.models import (
    Alternative,
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CertifiedFrontier,
    CheckpointObservation,
    ContrastSpec,
    GroupSelector,
)
from infinite_rulebook.analysis.supplemental_v2 import (
    SupplementalEvidencePlan,
    evaluate_supplemental_evidence,
    parse_supplemental_evidence_plan_json,
    parse_supplemental_evidence_report_json,
    supplemental_evidence_artifacts,
)
from infinite_rulebook.orchestration.hashing import scientific_hash


def _hash(label: str) -> str:
    return scientific_hash(label, domain="compact-v2-test")


_FRONTIER = CertifiedFrontier(
    semantic_hash=_hash("frontier"),
    zero_information_reward=0.0,
    maximum_reward=10.0,
    points=((0.0, 0.0, 0.0), (10.0, 1.0, 1.0)),
)


def _observation(
    environment: str,
    agent: str,
    environment_replica: int,
    algorithm_replica: int,
    checkpoint: int,
    *,
    frontier: CertifiedFrontier = _FRONTIER,
) -> CheckpointObservation:
    offset = environment_replica / 10.0 + algorithm_replica / 100.0
    agent_bonus = 0.5 if agent == "relevant" else 0.0
    load_bonus = {"D6": 0.06, "D12": 0.12, "D24": 0.24}.get(
        environment,
        0.0,
    )
    hidden = checkpoint + offset + agent_bonus + load_bonus
    public = 0.25 if environment == "PUBLIC-C" else 0.0
    metrics = [
        ("distractor_information_nats", 0.0),
        ("expected_reward", hidden + public),
        ("hidden_expected_reward", hidden),
        ("path_metric", hidden),
        ("public_reward", public),
    ]
    if checkpoint > 0:
        post_query = checkpoint + offset + agent_bonus + load_bonus
        metrics.extend(
            (
                ("post_query_hidden_expected_reward", post_query),
                (
                    "post_query_mean_hidden_expected_reward",
                    (checkpoint + 1.0) / 2.0 + offset + agent_bonus + load_bonus,
                ),
            )
        )
    run_label = f"{environment}:{agent}:{environment_replica}:{algorithm_replica}"
    return CheckpointObservation(
        run_hash=_hash(f"run:{run_label}"),
        content_hash=_hash(f"content:{run_label}"),
        phase=AnalysisPhase.CALIBRATION,
        confirmatory_frozen=False,
        freeze_hash=None,
        analysis_registration_hash=None,
        condition_hash=_hash(f"condition:{environment}"),
        environment_kind=environment,
        agent_kind=agent,
        agent_hash=_hash(f"agent:{agent}"),
        environment_replica=environment_replica,
        algorithm_replica=algorithm_replica,
        round_index=checkpoint,
        metrics=tuple(metrics),
        semantic_hashes=(("frontier", frontier.semantic_hash),),
        frontier=frontier,
    )


def _dataset() -> AnalysisDataset:
    return AnalysisDataset(
        tuple(
            _observation(environment, agent, environment_replica, algorithm_replica, t)
            for environment in ("IND", "ALEA", "PUBLIC-C", "D6", "D12", "D24")
            for agent in ("relevant", "total")
            for environment_replica in range(2)
            for algorithm_replica in range(2)
            for t in range(4)
        )
    )


def _selector(environment: str, agent: str = "relevant") -> GroupSelector:
    return GroupSelector(
        environment_kind=environment,
        agent_kind=agent,
        condition_hash=_hash(f"condition:{environment}"),
        agent_hash=_hash(f"agent:{agent}"),
    )


def _all_group_selectors() -> tuple[GroupSelector, ...]:
    return tuple(
        _selector(environment, agent)
        for environment in ("IND", "ALEA", "PUBLIC-C", "D6", "D12", "D24")
        for agent in ("relevant", "total")
    )


def _plan() -> CompactCanaryPlan:
    return CompactCanaryPlan(
        "compact-symbolic-canaries.v2",
        AnalysisPhase.CALIBRATION,
        (
            FrontierIdentityCanary(
                "01-frontier",
                _selector("IND"),
                _selector("ALEA"),
                (0, 1, 2, 3),
            ),
            MetricTrajectoryIdentityCanary(
                "02-path",
                "path_metric",
                _selector("IND"),
                _selector("ALEA"),
                (0, 1, 2, 3),
                1e-12,
            ),
            ConstantAdditiveMetricCanary(
                "03-public",
                _selector("PUBLIC-C"),
                "expected_reward",
                "hidden_expected_reward",
                "public_reward",
                0.25,
                (0, 1, 2, 3),
                1e-12,
            ),
            ExactZeroMetricCanary(
                "04-zero",
                _selector("ALEA"),
                "distractor_information_nats",
                (0, 1, 2, 3),
            ),
        ),
        (
            AggregateMetricCanary(
                "05-aggregate",
                _all_group_selectors(),
                "post_query_mean_hidden_expected_reward",
                "post_query_hidden_expected_reward",
                (1, 2, 3),
                1e-12,
            ),
        ),
    )


def _replace_metric(
    observation: CheckpointObservation,
    name: str,
    value: float,
) -> CheckpointObservation:
    return replace(
        observation,
        metrics=tuple(
            (metric, value if metric == name else current)
            for metric, current in observation.metrics
        ),
    )


def test_compact_v2_evidence_round_trips_without_embedded_detail_stream() -> None:
    dataset = _dataset()
    plan = _plan()
    evidence = evaluate_compact_canaries(dataset, plan)
    reversed_evidence = evaluate_compact_canaries(
        AnalysisDataset(tuple(reversed(dataset.observations))),
        plan,
    )

    assert evidence.report.passed
    assert evidence.report.dataset_hash == dataset.scientific_hash
    assert evidence.report.plan_hash == plan.scientific_hash
    assert evidence.report.scientific_hash == reversed_evidence.report.scientific_hash
    assert evidence.detail_chunks == reversed_evidence.detail_chunks
    assert len(evidence.detail_chunks) == 1
    assert evidence.report.detail_record_count == sum(
        item.record_count for item in evidence.report.results
    )
    aggregate = next(
        item for item in evidence.report.results if item.name == "05-aggregate"
    )
    assert aggregate.environment_cluster_count == 24
    assert aggregate.maximum_absolute_error <= aggregate.tolerance

    parsed_plan = parse_compact_canary_plan_json(plan.to_json())
    parsed_report = parse_compact_canary_report_json(evidence.report.to_json())
    parsed_chunks = tuple(
        parse_compact_canary_detail_chunk_json(chunk.to_json())
        for chunk in evidence.detail_chunks
    )
    assert parsed_plan == plan
    assert parsed_report == evidence.report
    assert parsed_chunks == evidence.detail_chunks
    assert CompactCanaryEvidence(parsed_report, parsed_chunks) == evidence
    assert evidence.passed
    assert evidence.scientific_hash == evidence.report.scientific_hash
    assert not hasattr(evidence, "to_dict")

    report_payload = json.loads(evidence.report.to_json())
    assert "records" not in report_payload
    assert report_payload["detail_chunks"][0]["fields"]["record_count"] > 0
    assert report_payload["detail_root_hash"] == evidence.report.detail_root_hash
    artifacts = compact_canary_artifacts(evidence)
    assert artifacts[0][0] == "canaries.json"
    assert artifacts[1][0] == "canary-details-000000.json"
    assert json.loads(artifacts[1][1])["scientific_hash"] == (
        evidence.detail_chunks[0].scientific_hash
    )


def test_planned_twenty_seven_gate_shape_is_supported() -> None:
    frontier = tuple(
        FrontierIdentityCanary(
            f"frontier-{index:02d}",
            _selector("IND"),
            _selector("ALEA"),
            (0, 3),
        )
        for index in range(4)
    )
    paths = tuple(
        MetricTrajectoryIdentityCanary(
            f"path-{index:02d}",
            "path_metric",
            _selector("IND"),
            _selector("ALEA"),
            (0, 1, 2, 3),
            1e-12,
        )
        for index in range(10)
    )
    public = tuple(
        ConstantAdditiveMetricCanary(
            f"public-{index:02d}",
            _selector("PUBLIC-C"),
            "expected_reward",
            "hidden_expected_reward",
            "public_reward",
            0.25,
            (0, 1, 2, 3),
            1e-12,
        )
        for index in range(6)
    )
    zero = tuple(
        ExactZeroMetricCanary(
            f"zero-{index:02d}",
            _selector("ALEA"),
            "distractor_information_nats",
            (0, 1, 2, 3),
        )
        for index in range(6)
    )
    aggregate = (
        AggregateMetricCanary(
            "aggregate-00",
            _all_group_selectors(),
            "post_query_mean_hidden_expected_reward",
            "post_query_hidden_expected_reward",
            (1, 2, 3),
            1e-12,
        ),
    )
    plan = CompactCanaryPlan(
        "planned-27.v2",
        AnalysisPhase.CALIBRATION,
        (*frontier, *paths, *public, *zero),
        aggregate,
    )
    evidence = evaluate_compact_canaries(_dataset(), plan)

    assert len(plan.canaries) + len(plan.aggregate_canaries) == 27
    assert len(evidence.report.results) == 27
    assert evidence.report.passed


def test_aggregate_tamper_and_missing_authenticated_history_fail_closed() -> None:
    observations = list(_dataset().observations)
    target = next(
        index
        for index, item in enumerate(observations)
        if item.environment_kind == "D12"
        and item.agent_kind == "relevant"
        and item.environment_replica == 0
        and item.algorithm_replica == 0
        and item.round_index == 3
    )
    item = observations[target]
    observations[target] = _replace_metric(
        item,
        "post_query_mean_hidden_expected_reward",
        item.metric("post_query_mean_hidden_expected_reward") + 0.25,
    )
    evidence = evaluate_compact_canaries(
        AnalysisDataset(tuple(observations)),
        _plan(),
    )
    aggregate = next(
        result for result in evidence.report.results if result.name == "05-aggregate"
    )
    assert not aggregate.passed
    assert aggregate.violation_count == 1
    assert len(aggregate.violations) == 1

    incomplete = tuple(
        item
        for item in _dataset().observations
        if not (
            item.environment_kind == "D12"
            and item.agent_kind == "relevant"
            and item.environment_replica == 0
            and item.algorithm_replica == 0
            and item.round_index == 2
        )
    )
    with pytest.raises(AnalysisError, match="missing authenticated source"):
        evaluate_compact_canaries(AnalysisDataset(incomplete), _plan())


def test_aggregate_exact_group_inventory_rejects_omission_and_overlap() -> None:
    plan = _plan()
    aggregate = plan.aggregate_canaries[0]
    omitted = replace(
        plan,
        aggregate_canaries=(replace(aggregate, selectors=aggregate.selectors[:-1]),),
    )
    with pytest.raises(AnalysisError, match="omits"):
        evaluate_compact_canaries(_dataset(), omitted)

    d6 = _selector("D6", "relevant")
    overlapping = GroupSelector(
        condition_hash=d6.condition_hash,
        agent_hash=d6.agent_hash,
    )
    overlap_plan = replace(
        plan,
        aggregate_canaries=(
            replace(
                aggregate,
                selectors=(*aggregate.selectors, overlapping),
            ),
        ),
    )
    with pytest.raises(AnalysisError, match="overlap"):
        evaluate_compact_canaries(_dataset(), overlap_plan)

    with pytest.raises(ValueError, match="unique"):
        replace(
            aggregate,
            selectors=(*aggregate.selectors, aggregate.selectors[0]),
        )


def test_failures_are_bounded_canonical_examples() -> None:
    observations = [
        (
            _replace_metric(
                item,
                "path_metric",
                item.metric("path_metric") + 1.0,
            )
            if item.environment_kind == "ALEA" and item.agent_kind == "relevant"
            else item
        )
        for item in _dataset().observations
    ]
    evidence = evaluate_compact_canaries(
        AnalysisDataset(tuple(observations)),
        _plan(),
    )
    result = next(item for item in evidence.report.results if item.name == "02-path")

    assert result.violation_count == 16
    assert len(result.violations) == COMPACT_CANARY_FAILURE_LIMIT
    assert result.violations == tuple(
        sorted(result.violations, key=lambda item: item.sort_key)
    )
    assert not result.passed


def test_detail_chunks_are_limited_ordered_and_inventory_bound() -> None:
    observations = tuple(
        _observation("ALEA", "relevant", environment, 0, 0)
        for environment in range(COMPACT_CANARY_DETAIL_LIMIT + 1)
    )
    dataset = AnalysisDataset(observations)
    plan = CompactCanaryPlan(
        "chunk-boundary.v2",
        AnalysisPhase.CALIBRATION,
        (
            ExactZeroMetricCanary(
                "zero",
                _selector("ALEA"),
                "distractor_information_nats",
                (0,),
            ),
        ),
    )
    evidence = evaluate_compact_canaries(dataset, plan)

    assert tuple(len(chunk.records) for chunk in evidence.detail_chunks) == (
        COMPACT_CANARY_DETAIL_LIMIT,
        1,
    )
    assert tuple(item.index for item in evidence.report.detail_chunks) == (0, 1)
    with pytest.raises(ValueError, match="4096"):
        CompactCanaryDetailChunk(
            0,
            (*evidence.detail_chunks[0].records, evidence.detail_chunks[1].records[0]),
        )
    with pytest.raises(ValueError, match="inventory"):
        CompactCanaryEvidence(
            evidence.report,
            tuple(reversed(evidence.detail_chunks)),
        )


def test_json_parsers_reject_duplicate_keys_reordering_and_tampering() -> None:
    plan = _plan()
    evidence = evaluate_compact_canaries(_dataset(), plan)

    duplicate = plan.to_json().replace(
        '"artifact_type":',
        '"artifact_type": "forged", "artifact_type":',
        1,
    )
    with pytest.raises(AnalysisError, match="repeats key"):
        parse_compact_canary_plan_json(duplicate)

    plan_payload = plan.to_dict()
    plan_payload["scientific_hash"] = "0" * 64
    with pytest.raises(AnalysisError, match="tampered"):
        parse_compact_canary_plan_json(json.dumps(plan_payload))

    plan_payload = plan.to_dict()
    plan_payload["canaries"][0]["record_type"] = "builtins.dict"
    with pytest.raises(AnalysisError, match=r"registered union|unregistered"):
        parse_compact_canary_plan_json(json.dumps(plan_payload))

    chunk_payload = evidence.detail_chunks[0].to_dict()
    chunk_payload["records"] = list(reversed(chunk_payload["records"]))
    with pytest.raises(
        AnalysisError,
        match=r"fields|canonical|tampered|validation",
    ):
        parse_compact_canary_detail_chunk_json(json.dumps(chunk_payload))

    report_payload = evidence.report.to_dict()
    report_payload["detail_root_hash"] = "0" * 64
    with pytest.raises(AnalysisError, match=r"fields|authenticate|validation"):
        parse_compact_canary_report_json(json.dumps(report_payload))

    forged_result = replace(
        evidence.report.results[0],
        minimum_residual=1.0,
        maximum_residual=1.0,
        maximum_absolute_error=1.0,
    )
    forged_report = replace(
        evidence.report,
        results=(forged_result, *evidence.report.results[1:]),
    )
    with pytest.raises(ValueError, match="summary"):
        CompactCanaryEvidence(forged_report, evidence.detail_chunks)

    plan_payload = plan.to_dict()
    plan_payload["schema_version"] = 1.0
    with pytest.raises(AnalysisError, match="schema"):
        parse_compact_canary_plan_json(json.dumps(plan_payload))


def _supplemental_plan() -> SupplementalEvidencePlan:
    return SupplementalEvidencePlan(
        "registered-supplemental.v2",
        AnalysisPhase.CALIBRATION,
        (
            ContrastSpec(
                "legacy-d6-terminal",
                "hidden_expected_reward",
                _selector("D6", "relevant"),
                _selector("D6", "total"),
                3,
                Alternative.GREATER,
                0.0,
            ),
        ),
        (
            ContrastSpec(
                "d12-relevant-minus-total",
                "hidden_expected_reward",
                _selector("D12", "relevant"),
                _selector("D12", "total"),
                3,
                Alternative.GREATER,
                0.0,
            ),
        ),
    )


def test_supplemental_evidence_is_exact_and_cannot_enter_or_rescue_holm() -> None:
    plan = _supplemental_plan()
    report = evaluate_supplemental_evidence(_dataset(), plan)
    payload = report.to_dict()

    assert payload["outside_primary_holm"] is True
    assert payload["may_rescue_compound_s2"] is False
    assert payload["holm_decisions"] == []
    assert "passed" not in payload
    replication = report.legacy_replications[0]
    assert replication.environment_differences == (0.5, 0.5)
    assert replication.summary.above_null_count == 2
    assert replication.summary.tie_count == 0
    assert replication.summary.exact_sign_p_value == 0.25
    descriptive = report.descriptive_comparisons[0]
    assert descriptive.environment_differences == (0.5, 0.5)
    assert descriptive.summary.median_difference == 0.5
    assert descriptive.summary.above_null_count == 2

    assert parse_supplemental_evidence_plan_json(plan.to_json()) == plan
    parsed = parse_supplemental_evidence_report_json(report.to_json())
    assert parsed == report
    assert parsed.scientific_hash == report.scientific_hash
    artifacts = supplemental_evidence_artifacts(plan, report)
    assert tuple(name for name, _ in artifacts) == (
        "supplemental-plan.json",
        "supplemental.json",
    )
    with pytest.raises(ValueError, match="derive"):
        supplemental_evidence_artifacts(
            replace(plan, name="other-supplemental"),
            report,
        )
    with pytest.raises(ValueError, match="derive"):
        supplemental_evidence_artifacts(
            plan,
            replace(
                report,
                legacy_replications=report.descriptive_comparisons,
                descriptive_comparisons=report.legacy_replications,
            ),
        )


def test_supplemental_contract_rejects_inexact_selectors_and_tampering() -> None:
    with pytest.raises(ValueError, match="exact"):
        SupplementalEvidencePlan(
            "bad",
            AnalysisPhase.CALIBRATION,
            (
                ContrastSpec(
                    "legacy",
                    "hidden_expected_reward",
                    GroupSelector(environment_kind="D6", agent_kind="relevant"),
                    _selector("D6", "total"),
                    3,
                    Alternative.GREATER,
                    0.0,
                ),
            ),
            _supplemental_plan().descriptive_comparisons,
        )

    report = evaluate_supplemental_evidence(_dataset(), _supplemental_plan())
    payload = report.to_dict()
    payload["may_rescue_compound_s2"] = True
    with pytest.raises(AnalysisError, match="boundary"):
        parse_supplemental_evidence_report_json(json.dumps(payload))

    payload = report.to_dict()
    payload["legacy_replications"][0]["fields"]["summary"]["fields"][
        "mean_difference"
    ] = 0.0
    with pytest.raises(
        AnalysisError,
        match=r"derived|invalid|tampered|canonical",
    ):
        parse_supplemental_evidence_report_json(json.dumps(payload))


def test_descriptive_d12_comparison_requires_exact_paired_replica_grid() -> None:
    plan = _supplemental_plan()
    d12 = plan.descriptive_comparisons[0]
    with pytest.raises(ValueError, match="differ"):
        replace(
            plan,
            descriptive_comparisons=(replace(d12, right=d12.left),),
        )

    unmatched = tuple(
        item
        for item in _dataset().observations
        if not (
            item.environment_kind == "D12"
            and item.agent_kind == "total"
            and item.environment_replica == 1
            and item.algorithm_replica == 1
            and item.round_index == 3
        )
    )
    with pytest.raises(AnalysisError, match="unmatched algorithm"):
        evaluate_supplemental_evidence(AnalysisDataset(unmatched), plan)

    same_group = replace(
        plan,
        descriptive_comparisons=(
            replace(
                d12,
                right=GroupSelector(
                    condition_hash=d12.left.condition_hash,
                    agent_hash=d12.left.agent_hash,
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisError, match="distinct exact groups"):
        evaluate_supplemental_evidence(_dataset(), same_group)
