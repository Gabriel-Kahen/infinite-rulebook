from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import infinite_rulebook.cli as cli
from infinite_rulebook.analysis import (
    AnalysisPhase,
    AnalysisReport,
    ContrastResult,
    EquivalenceResult,
    ExactInterval,
    HolmDecision,
    load_analysis_plan,
)
from infinite_rulebook.analysis.canaries import CanaryKind, ExactZeroMetricCanary
from infinite_rulebook.analysis.compact_canaries_v2 import (
    CompactCanaryDetail,
    CompactCanaryDetailChunk,
    CompactCanaryEvidence,
    CompactCanaryReport,
    CompactGateResult,
    SpooledCompactCanaryEvidence,
    compact_canary_artifact_names,
    compact_canary_artifacts,
    detail_inventory_hash,
    parse_compact_canary_detail_chunk_json,
    parse_compact_canary_report_json,
)
from infinite_rulebook.analysis.supplemental_v2 import (
    PairedComparisonEvidence,
    SupplementalEvidenceReport,
)
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.inventory import (
    RawArtifactInventory,
    RawArtifactTree,
)
from infinite_rulebook.orchestration.provenance import collect_provenance
from infinite_rulebook.orchestration.release import (
    STUDY_RELEASE_MANIFEST_FILENAME,
    StudyReleaseManifest,
    load_study_release_manifest,
)
from infinite_rulebook.orchestration.reproducibility import (
    ReproducibilityReport,
    ReproducibilityRun,
    _create_execution_receipt_pair,
)
from infinite_rulebook.studies import symbolic_construct_v2 as v2
from infinite_rulebook.studies.symbolic_registry import SYMBOLIC_STUDY_V2

_CONFIG = Path(__file__).parents[1] / "configs" / "symbolic-calibration-v2.json"
_CALIBRATION_ENVIRONMENTS = 32
_POWER_SIMULATIONS = 20


@dataclass(frozen=True)
class _Lifecycle:
    config_path: Path
    config: ExperimentConfig
    analysis_plan: Path
    canary_plan: Path
    supplemental_plan: Path
    report_root: Path
    study: Any
    dataset: SimpleNamespace
    reproducibility: ReproducibilityReport
    serial_inventory: RawArtifactInventory
    parallel_inventory: RawArtifactInventory
    smoke: SimpleNamespace


def _compact_evidence(dataset: SimpleNamespace, plan: Any) -> CompactCanaryEvidence:
    specs = tuple(
        item for item in plan.canaries if isinstance(item, ExactZeroMetricCanary)
    )[:2]
    assert len(specs) == 2
    details = tuple(
        CompactCanaryDetail(
            gate_name=spec.name,
            kind=CanaryKind.EXACT_ZERO.value,
            comparison=spec.metric,
            group_hash=scientific_hash(
                spec.name,
                domain="v2-lifecycle-compact-group",
            ),
            environment_replica=0,
            algorithm_replica=0,
            checkpoint=0,
            residual=0.0,
            tolerance=0.0,
            violated=False,
        )
        for spec in specs
    )
    chunks = tuple(
        CompactCanaryDetailChunk(index, (detail,))
        for index, detail in enumerate(details)
    )
    results = tuple(
        CompactGateResult(
            name=detail.gate_name,
            kind=detail.kind,
            passed=True,
            environment_cluster_count=1,
            cell_count=1,
            checkpoint_count=1,
            record_count=1,
            violation_count=0,
            tolerance=0.0,
            minimum_residual=0.0,
            maximum_residual=0.0,
            maximum_absolute_error=0.0,
            violations=(),
        )
        for detail in details
    )
    references = tuple(chunk.reference for chunk in chunks)
    report = CompactCanaryReport(
        phase=AnalysisPhase.CALIBRATION,
        dataset_hash=dataset.scientific_hash,
        plan_hash=plan.scientific_hash,
        results=results,
        detail_record_count=len(details),
        detail_chunks=references,
        detail_root_hash=detail_inventory_hash(references),
    )
    return CompactCanaryEvidence(report, chunks)


def _spool_compact_evidence(
    dataset: SimpleNamespace,
    plan: Any,
    directory: Path,
) -> SpooledCompactCanaryEvidence:
    evidence = _compact_evidence(dataset, plan)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in compact_canary_artifacts(evidence):
        (directory / name).write_text(content, encoding="utf-8")
    return SpooledCompactCanaryEvidence(evidence.report)


def _supplemental_report(
    dataset: SimpleNamespace,
    plan: Any,
    *,
    environment_replicas: int,
    algorithm_replicas: int,
) -> SupplementalEvidenceReport:
    def evidence(spec: Any) -> PairedComparisonEvidence:
        return PairedComparisonEvidence(
            spec,
            (1.0,) * environment_replicas,
            environment_replicas * algorithm_replicas,
            plan.interval_alpha,
        )

    return SupplementalEvidenceReport(
        phase=AnalysisPhase.CALIBRATION,
        dataset_hash=dataset.scientific_hash,
        plan_hash=plan.scientific_hash,
        interval_alpha=plan.interval_alpha,
        legacy_replications=tuple(evidence(item) for item in plan.legacy_replications),
        descriptive_comparisons=tuple(
            evidence(item) for item in plan.descriptive_comparisons
        ),
    )


def _analysis_report(
    dataset: SimpleNamespace,
    plan: Any,
    *,
    config: ExperimentConfig,
    canary_report_hash: str,
    canaries_passed: bool,
    config_hash: str,
) -> AnalysisReport:
    contrasts = tuple(
        ContrastResult(
            name=spec.name,
            metric=spec.metric,
            left_label=spec.left.label,
            right_label=spec.right.label,
            checkpoint=spec.checkpoint,
            alternative=spec.alternative,
            null_margin=spec.null_margin,
            pair_count=config.environment_replicas,
            cell_pair_count=(config.environment_replicas * config.algorithm_replicas),
            differences=(v2.PRIMARY_MINIMUM_EFFECTS[spec.name] + 1.0,)
            * config.environment_replicas,
            mean_difference=v2.PRIMARY_MINIMUM_EFFECTS[spec.name] + 1.0,
            median_difference=v2.PRIMARY_MINIMUM_EFFECTS[spec.name] + 1.0,
            standardized_mean_difference=None,
            median_interval=ExactInterval(
                v2.PRIMARY_MINIMUM_EFFECTS[spec.name] + 1.0,
                v2.PRIMARY_MINIMUM_EFFECTS[spec.name] + 1.0,
                0.95,
                "bounded-v2-lifecycle-fixture",
            ),
            unadjusted_p_value=0.001,
        )
        for spec in plan.contrasts
    )
    equivalences = tuple(
        EquivalenceResult(
            name=spec.name,
            metric=spec.metric,
            left_label=spec.left.label,
            right_label=spec.right.label,
            checkpoint=spec.checkpoint,
            margin=spec.margin,
            margin_source=spec.margin_source.value,
            margin_provenance_hash=spec.margin_provenance_hash,
            pair_count=config.environment_replicas,
            cell_pair_count=(config.environment_replicas * config.algorithm_replicas),
            differences=(0.0,) * config.environment_replicas,
            mean_difference=0.0,
            median_difference=0.0,
            median_interval=ExactInterval(
                0.0,
                0.0,
                0.95,
                "bounded-v2-lifecycle-fixture",
            ),
            lower_tost_p_value=0.001,
            upper_tost_p_value=0.001,
            unadjusted_p_value=0.001,
        )
        for spec in plan.equivalences
    )
    return AnalysisReport(
        phase=AnalysisPhase.CALIBRATION,
        dataset_hash=dataset.scientific_hash,
        plan=plan,
        pools=(),
        contrasts=contrasts,
        equivalences=equivalences,
        scaling=(),
        family_decisions=tuple(
            HolmDecision(spec.name, 0.001, 0.001, plan.family_alpha, True)
            for spec in plan.contrasts
        ),
        equivalence_decisions=tuple(
            HolmDecision(spec.name, 0.001, 0.001, plan.family_alpha, True)
            for spec in plan.equivalences
        ),
        run_settings_hash=scientific_hash(
            config.resolved_run_settings(),
            domain="resolved-run-settings",
        ),
        provenance=dataset.provenance,
        canary_report_hash=canary_report_hash,
        canaries_passed=canaries_passed,
        config_hash=config_hash,
    )


def _test_study(config: ExperimentConfig) -> Any:
    def build_analysis(
        selected: ExperimentConfig,
        *,
        phase: AnalysisPhase,
        freeze_hash: str | None = None,
    ) -> Any:
        return v2._build_analysis_plan(
            selected,
            phase=phase,
            freeze_hash=freeze_hash,
        )

    def registration(selected: ExperimentConfig) -> str:
        freeze_hash = (
            None
            if selected.confirmatory_freeze is None
            else selected.confirmatory_freeze.seal_hash
        )
        return build_analysis(
            selected,
            phase=AnalysisPhase.CONFIRMATORY,
            freeze_hash=freeze_hash,
        ).registration_hash

    def tolerances(selected: ExperimentConfig) -> dict[str, float]:
        return {
            "aggregate_metric_absolute_error": (v2.AGGREGATE_METRIC_ABSOLUTE_TOLERANCE),
            "artifact_completion_fraction": v2.ARTIFACT_COMPLETION_FRACTION,
            "frontier_bound_tolerance_nats": selected.solver.bound_tolerance,
            "ledger_reconciliation_nats": v2.LEDGER_RECONCILIATION_TOLERANCE,
            "paired_path_absolute_error": v2.PAIRED_PATH_ABSOLUTE_TOLERANCE,
        }

    def verify_calibration(selected: ExperimentConfig) -> None:
        if (
            selected.name != config.name
            or selected.phase != "calibration"
            or selected.confirmatory_freeze is not None
        ):
            raise ValueError("invalid bounded v2 calibration fixture")

    def verify_confirmatory(
        selected: ExperimentConfig,
        *,
        analysis_code_hash: str | None = None,
        dependency_lock_hash: str | None = None,
        environment_digest: str | None = None,
    ) -> Any:
        record = selected.confirmatory_freeze
        if selected.phase != "confirmatory" or record is None:
            raise ValueError("invalid bounded v2 confirmatory fixture")
        record.verify_config(selected)
        record.verify_semantic_contract(
            analysis_contract=v2.STUDY_CONTRACT,
            analysis_version=registration(selected),
            analysis_code_hash=analysis_code_hash,
            dependency_lock_hash=dependency_lock_hash,
            environment_digest=environment_digest,
            tolerances=tolerances(selected),
            margins=v2.expected_confirmatory_margins(),
        )
        return record

    def evidence_hash(
        *,
        config: ExperimentConfig,
        report: AnalysisReport,
        **fields: Any,
    ) -> str:
        return v2.calibration_evidence_hash_from_hashes(
            config_hash=config.config_hash,
            analysis_report_hash=report.scientific_hash,
            analysis_code_hash=dict(report.provenance).get("analysis_code_hash"),
            run_settings_hash=report.run_settings_hash,
            **fields,
        )

    supplemental = replace(
        SYMBOLIC_STUDY_V2.evidence.supplemental,
        build_plan=lambda selected, *, phase: v2._build_symbolic_supplemental_plan(
            selected,
            phase=phase,
        ),
        evaluate=lambda dataset, plan: _supplemental_report(
            dataset,
            plan,
            environment_replicas=config.environment_replicas,
            algorithm_replicas=config.algorithm_replicas,
        ),
    )
    evidence = replace(
        SYMBOLIC_STUDY_V2.evidence,
        build_canary_plan=lambda selected, *, phase: v2._build_symbolic_canary_plan(
            selected,
            phase=phase,
        ),
        evaluate_canaries=_compact_evidence,
        evaluate_canaries_to_directory=_spool_compact_evidence,
        supplemental=supplemental,
    )
    power = replace(
        SYMBOLIC_STUDY_V2.power,
        candidate_environment_counts=(2,),
        center_environment_count=_CALIBRATION_ENVIRONMENTS // 2,
        probability_environment_count=_CALIBRATION_ENVIRONMENTS // 2,
        simulations=_POWER_SIMULATIONS,
        minimum_individual_power=0.0,
        minimum_equivalence_power=0.0,
        minimum_joint_power=0.0,
        maximum_global_null_fwer=1.0,
    )
    return replace(
        SYMBOLIC_STUDY_V2,
        verify_calibration=verify_calibration,
        verify_confirmatory=verify_confirmatory,
        build_analysis_plan=build_analysis,
        expected_confirmatory_registration=registration,
        expected_confirmatory_tolerances=tolerances,
        calibration_evidence_hash=evidence_hash,
        evidence=evidence,
        power=power,
    )


def _raw_inventory(
    config: ExperimentConfig,
    runs: tuple[ReproducibilityRun, ...],
    *,
    side: str,
    receipt: Any,
) -> RawArtifactInventory:
    frontier_hash = scientific_hash("frontier", domain="v2-lifecycle-frontier")
    trees = tuple(
        sorted(
            (
                RawArtifactTree(
                    tree_type="frontier",
                    path=f"_frontiers/{frontier_hash}",
                    identity_hash=frontier_hash,
                    scientific_content_hash=scientific_hash(
                        "frontier-content",
                        domain="v2-lifecycle-frontier-content",
                    ),
                    file_count=1,
                    byte_size=0,
                    tree_byte_sha256=scientific_hash(
                        (side, "frontier"),
                        domain="v2-lifecycle-tree-bytes",
                    ),
                ),
                *(
                    RawArtifactTree(
                        tree_type="run",
                        path=f"{config.name}/{run.run_hash}",
                        identity_hash=run.run_hash,
                        scientific_content_hash=run.scientific_content_hash,
                        file_count=1,
                        byte_size=0,
                        tree_byte_sha256=scientific_hash(
                            (side, run.run_hash),
                            domain="v2-lifecycle-tree-bytes",
                        ),
                    )
                    for run in runs
                ),
            ),
            key=lambda item: item.path,
        )
    )
    body = {
        "artifact_type": "infinite-rulebook-raw-artifact-inventory",
        "schema_version": 2,
        "experiment_name": config.name,
        "config_hash": config.config_hash,
        "side": side,
        "execution_receipt": receipt.to_dict(),
        "trees": [tree.to_dict() for tree in trees],
    }
    return RawArtifactInventory(
        experiment_name=config.name,
        config_hash=config.config_hash,
        side=side,
        trees=trees,
        execution_receipt=receipt,
        scientific_hash=scientific_hash(body, domain="raw-artifact-inventory"),
    )


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Lifecycle:
    config = replace(
        load_experiment_config(_CONFIG),
        environment_replicas=_CALIBRATION_ENVIRONMENTS,
        algorithm_replicas=1,
    )
    config_path = tmp_path / "symbolic-calibration-v2.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    study = _test_study(config)
    provenance_record = collect_provenance()
    provenance = tuple(sorted(provenance_record.to_dict().items()))
    cells = config.cells()
    runs = tuple(
        ReproducibilityRun(
            cell_hash=cell.cell_hash,
            run_hash=scientific_hash(
                cell.cell_hash,
                domain="v2-lifecycle-run",
            ),
            scientific_content_hash=scientific_hash(
                cell.cell_hash,
                domain="v2-lifecycle-content",
            ),
        )
        for cell in cells
    )
    serial_receipt, parallel_receipt = _create_execution_receipt_pair(
        config,
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=2,
        provenance=provenance_record,
        smoke_prerequisite_hash=v2.SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
    )
    reproducibility = ReproducibilityReport(
        config_hash=config.config_hash,
        runs=runs,
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=2,
        invocation_id=serial_receipt.invocation_id,
        serial_receipt_hash=serial_receipt.scientific_hash,
        parallel_receipt_hash=parallel_receipt.scientific_hash,
    )
    serial_inventory = _raw_inventory(
        config,
        runs,
        side="serial",
        receipt=serial_receipt,
    )
    parallel_inventory = _raw_inventory(
        config,
        runs,
        side="parallel",
        receipt=parallel_receipt,
    )
    dataset = SimpleNamespace(
        observations=tuple(
            SimpleNamespace(
                run_hash=run.run_hash,
                cell_hash=run.cell_hash,
                content_hash=run.scientific_content_hash,
            )
            for run in runs
        ),
        provenance=provenance,
        scientific_hash=scientific_hash(
            [run.to_dict() for run in runs],
            domain="v2-lifecycle-dataset",
        ),
    )
    smoke = SimpleNamespace(
        config=SimpleNamespace(config_hash=v2.SYMBOLIC_V2_SMOKE_CONFIG_HASH),
        reproducibility=SimpleNamespace(
            scientific_hash=scientific_hash("smoke-repro", domain="v2-lifecycle"),
            serial_root=tmp_path / "smoke-serial",
            parallel_root=tmp_path / "smoke-parallel",
        ),
        serial_inventory=SimpleNamespace(
            scientific_hash=scientific_hash(
                "smoke-serial",
                domain="v2-lifecycle",
            )
        ),
        parallel_inventory=SimpleNamespace(
            scientific_hash=scientific_hash(
                "smoke-parallel",
                domain="v2-lifecycle",
            )
        ),
        engineering_anomalies=(),
        scientific_hash=v2.SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
        to_dict=lambda: {
            "artifact_type": "bounded-v2-lifecycle-smoke-fixture",
            "scientific_hash": v2.SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
        },
    )
    roots = tuple(tmp_path / "runs" / run.run_hash for run in runs)

    def create_inventory(
        _cls: type[RawArtifactInventory],
        _root: Path,
        selected: ExperimentConfig,
        *,
        side: str,
    ) -> RawArtifactInventory:
        assert selected == config
        return {"serial": serial_inventory, "parallel": parallel_inventory}[side]

    def verify_inventory(
        inventory: RawArtifactInventory,
        _root: Path,
        selected: ExperimentConfig,
        *,
        side: str | None = None,
    ) -> None:
        assert selected == config
        assert side is None or side == inventory.side

    original_registered = cli.registered_symbolic_study
    monkeypatch.setattr(
        cli,
        "registered_symbolic_study",
        lambda name: (
            study if name in study.registered_names else original_registered(name)
        ),
    )
    monkeypatch.setattr(cli, "collect_provenance", lambda: provenance_record)
    monkeypatch.setattr(
        cli,
        "_completed_run_roots",
        lambda _root, experiment: (
            roots
            if experiment == config.name
            else (_ for _ in ()).throw(AssertionError(experiment))
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_reproducibility_report",
        lambda *_args, **_kwargs: reproducibility,
    )
    monkeypatch.setattr(cli, "_load_smoke_prerequisite", lambda *_args: smoke)
    monkeypatch.setattr(cli, "load_run_trees", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(
        cli,
        "build_report",
        lambda loaded, plan, **kwargs: _analysis_report(
            loaded,
            plan,
            config=config,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        RawArtifactInventory,
        "create",
        classmethod(create_inventory),
    )
    monkeypatch.setattr(RawArtifactInventory, "verify", verify_inventory)

    analysis_plan = tmp_path / "analysis-v2.json"
    canary_plan = tmp_path / "canaries-v2.json"
    supplemental_plan = tmp_path / "supplemental-v2.json"
    assert (
        cli.main(
            [
                "plan",
                str(config_path),
                str(analysis_plan),
                str(canary_plan),
                "--supplemental-output",
                str(supplemental_plan),
            ]
        )
        == 0
    )
    return _Lifecycle(
        config_path=config_path,
        config=config,
        analysis_plan=analysis_plan,
        canary_plan=canary_plan,
        supplemental_plan=supplemental_plan,
        report_root=tmp_path / "calibration-release-v2",
        study=study,
        dataset=dataset,
        reproducibility=reproducibility,
        serial_inventory=serial_inventory,
        parallel_inventory=parallel_inventory,
        smoke=smoke,
    )


def _report_arguments(lifecycle: _Lifecycle, output: Path) -> list[str]:
    return [
        "report",
        str(lifecycle.config_path),
        str(lifecycle.analysis_plan),
        str(lifecycle.canary_plan),
        str(lifecycle.reproducibility.parallel_root),
        str(output),
        "--supplemental-plan",
        str(lifecycle.supplemental_plan),
        "--power-simulations",
        str(_POWER_SIMULATIONS),
        "--reproducibility-report",
        str(output.parent / "reproducibility-input.json"),
        "--smoke-evidence",
        str(output.parent / "smoke-input.json"),
    ]


def _freeze_arguments(
    lifecycle: _Lifecycle,
    release_root: Path,
    output_root: Path,
    *,
    include_supplemental: bool = True,
) -> list[str]:
    arguments = [
        "freeze",
        str(lifecycle.config_path),
        str(release_root / "summary.json"),
        str(output_root / "confirmatory-v2.json"),
        str(output_root / "analysis-v2.json"),
        str(output_root / "canaries-v2.json"),
    ]
    if include_supplemental:
        arguments.extend(
            [
                "--output-supplemental-plan",
                str(output_root / "supplemental-v2.json"),
            ]
        )
    return arguments


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash(payload: dict[str, Any], *, domain: str) -> str:
    payload["scientific_hash"] = scientific_hash(
        {name: value for name, value in payload.items() if name != "scientific_hash"},
        domain=domain,
    )
    return payload["scientific_hash"]


def _reseal_release(root: Path) -> None:
    old = load_study_release_manifest(
        root / STUDY_RELEASE_MANIFEST_FILENAME,
        verify_files=False,
    )
    release = StudyReleaseManifest.create(
        root,
        tuple(member.path for member in old.members),
        phase=old.phase,
        study_contract=old.study_contract,
        config_hash=old.config_hash,
        freeze_hash=old.freeze_hash,
        calibration_evidence_hash=_read(root / "summary.json")[
            "calibration_evidence_hash"
        ],
    )
    _write(
        root / STUDY_RELEASE_MANIFEST_FILENAME,
        release.to_dict(),
    )


def _rebind_summary(root: Path) -> None:
    analysis = _read(root / "analysis.json")
    canaries = _read(root / "canaries.json")
    supplemental_plan = _read(root / "supplemental-plan.json")
    supplemental = _read(root / "supplemental.json")
    summary = _read(root / "summary.json")
    summary.update(
        {
            "analysis_hash": analysis["scientific_hash"],
            "canary_hash": canaries["scientific_hash"],
            "canary_detail_root_hash": canaries["detail_root_hash"],
            "canary_detail_record_count": canaries["detail_record_count"],
            "supplemental_plan_hash": supplemental_plan["scientific_hash"],
            "supplemental_report_hash": supplemental["scientific_hash"],
        }
    )
    summary["calibration_evidence_hash"] = v2.calibration_evidence_hash_from_hashes(
        config_hash=summary["config_hash"],
        analysis_report_hash=summary["analysis_hash"],
        canary_report_hash=summary["canary_hash"],
        canary_detail_root_hash=summary["canary_detail_root_hash"],
        canary_detail_record_count=summary["canary_detail_record_count"],
        supplemental_plan_hash=summary["supplemental_plan_hash"],
        supplemental_report_hash=summary["supplemental_report_hash"],
        power_calibration_hash=summary["power_hash"],
        reproducibility_report_hash=summary["reproducibility_hash"],
        raw_serial_inventory_hash=summary["raw_serial_inventory_hash"],
        raw_parallel_inventory_hash=summary["raw_parallel_inventory_hash"],
        deviation_log_hash=summary["deviation_log_hash"],
        smoke_prerequisite_hash=summary["smoke_prerequisite_hash"],
        smoke_config_hash=summary["smoke_config_hash"],
        smoke_reproducibility_hash=summary["smoke_reproducibility_hash"],
        smoke_raw_serial_inventory_hash=summary["smoke_raw_serial_inventory_hash"],
        smoke_raw_parallel_inventory_hash=summary["smoke_raw_parallel_inventory_hash"],
        analysis_code_hash=summary["analysis_code_hash"],
        run_settings_hash=summary["run_settings_hash"],
    )
    _rehash(summary, domain="symbolic-study-summary")
    _write(root / "summary.json", summary)
    _reseal_release(root)


def test_v2_report_release_freeze_contract_binds_dynamic_and_supplemental_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _prepare(tmp_path, monkeypatch)
    missing_supplemental = tmp_path / "missing-supplemental-release"
    arguments = _report_arguments(lifecycle, missing_supplemental)
    plan_flag = arguments.index("--supplemental-plan")
    with pytest.raises(ValueError, match="requires --supplemental-plan"):
        cli.main(arguments[:plan_flag] + arguments[plan_flag + 2 :])
    assert not missing_supplemental.exists()

    assert cli.main(_report_arguments(lifecycle, lifecycle.report_root)) == 0
    root = lifecycle.report_root
    summary = _read(root / "summary.json")
    canaries = _read(root / "canaries.json")
    supplemental = _read(root / "supplemental.json")
    release = load_study_release_manifest(root / STUDY_RELEASE_MANIFEST_FILENAME)
    detail_names = tuple(
        name
        for name in compact_canary_artifact_names(
            _compact_evidence(
                lifecycle.dataset,
                lifecycle.study.evidence.build_canary_plan(
                    lifecycle.config,
                    phase=AnalysisPhase.CALIBRATION,
                ),
            )
        )
        if name.startswith("canary-details-")
    )
    assert detail_names == (
        "canary-details-000000.json",
        "canary-details-000001.json",
    )
    assert set(detail_names) <= {member.path for member in release.members}
    assert {"supplemental-plan.json", "supplemental.json"} <= {
        member.path for member in release.members
    }
    assert summary["canary_detail_root_hash"] == canaries["detail_root_hash"]
    assert summary["canary_detail_record_count"] == canaries["detail_record_count"]
    assert (
        summary["supplemental_plan_hash"]
        == _read(root / "supplemental-plan.json")["scientific_hash"]
    )
    assert summary["supplemental_report_hash"] == supplemental["scientific_hash"]
    assert summary["calibration_evidence_hash"] == (
        v2.calibration_evidence_hash_from_hashes(
            config_hash=lifecycle.config.config_hash,
            analysis_report_hash=summary["analysis_hash"],
            canary_report_hash=summary["canary_hash"],
            canary_detail_root_hash=summary["canary_detail_root_hash"],
            canary_detail_record_count=summary["canary_detail_record_count"],
            supplemental_plan_hash=summary["supplemental_plan_hash"],
            supplemental_report_hash=summary["supplemental_report_hash"],
            power_calibration_hash=summary["power_hash"],
            reproducibility_report_hash=summary["reproducibility_hash"],
            raw_serial_inventory_hash=summary["raw_serial_inventory_hash"],
            raw_parallel_inventory_hash=summary["raw_parallel_inventory_hash"],
            deviation_log_hash=summary["deviation_log_hash"],
            smoke_prerequisite_hash=summary["smoke_prerequisite_hash"],
            smoke_config_hash=summary["smoke_config_hash"],
            smoke_reproducibility_hash=summary["smoke_reproducibility_hash"],
            smoke_raw_serial_inventory_hash=summary["smoke_raw_serial_inventory_hash"],
            smoke_raw_parallel_inventory_hash=summary[
                "smoke_raw_parallel_inventory_hash"
            ],
            analysis_code_hash=summary["analysis_code_hash"],
            run_settings_hash=summary["run_settings_hash"],
        )
    )
    assert release.calibration_evidence_hash == summary["calibration_evidence_hash"]

    with pytest.raises(ValueError, match="requires --output-supplemental-plan"):
        cli.main(
            _freeze_arguments(
                lifecycle,
                root,
                tmp_path / "missing-frozen-supplemental",
                include_supplemental=False,
            )
        )
    frozen_root = tmp_path / "frozen"
    assert cli.main(_freeze_arguments(lifecycle, root, frozen_root)) == 0
    sealed = load_experiment_config(frozen_root / "confirmatory-v2.json")
    assert sealed.confirmatory_freeze is not None
    assert (
        sealed.confirmatory_freeze.calibration_evidence_hash
        == summary["calibration_evidence_hash"]
    )
    assert load_analysis_plan(frozen_root / "analysis-v2.json").registration_hash == (
        sealed.confirmatory_freeze.analysis_version
    )
    expected_supplemental = lifecycle.study.evidence.supplemental.build_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
    )
    lifecycle.study.evidence.supplemental.verify_plan_json(
        (frozen_root / "supplemental-v2.json").read_text(encoding="utf-8"),
        expected_supplemental,
    )

    detail_tamper = tmp_path / "detail-tamper"
    shutil.copytree(root, detail_tamper)
    detail_path = detail_tamper / detail_names[0]
    detail_chunk = parse_compact_canary_detail_chunk_json(
        detail_path.read_text(encoding="utf-8")
    )
    detail_chunk = replace(
        detail_chunk,
        records=(
            replace(detail_chunk.records[0], comparison="tampered-metric"),
            *detail_chunk.records[1:],
        ),
    )
    detail_path.write_text(detail_chunk.to_json(), encoding="utf-8")
    detail_chunks = tuple(
        parse_compact_canary_detail_chunk_json(
            (detail_tamper / name).read_text(encoding="utf-8")
        )
        for name in detail_names
    )
    references = tuple(chunk.reference for chunk in detail_chunks)
    canary_report = parse_compact_canary_report_json(
        (detail_tamper / "canaries.json").read_text(encoding="utf-8")
    )
    canary_report = replace(
        canary_report,
        detail_chunks=references,
        detail_root_hash=detail_inventory_hash(references),
    )
    (detail_tamper / "canaries.json").write_text(
        canary_report.to_json(),
        encoding="utf-8",
    )
    analysis_payload = _read(detail_tamper / "analysis.json")
    analysis_payload["canary_report_hash"] = canary_report.scientific_hash
    _rehash(analysis_payload, domain="registered-analysis-report")
    _write(detail_tamper / "analysis.json", analysis_payload)
    _rebind_summary(detail_tamper)
    detail_outputs = tmp_path / "detail-tamper-outputs"
    with pytest.raises(ValueError):
        cli.main(_freeze_arguments(lifecycle, detail_tamper, detail_outputs))
    assert not (detail_outputs / "confirmatory-v2.json").exists()

    supplemental_tamper = tmp_path / "supplemental-tamper"
    shutil.copytree(root, supplemental_tamper)
    supplemental_payload = _read(supplemental_tamper / "supplemental.json")
    supplemental_payload["dataset_hash"] = "f" * 64
    _rehash(supplemental_payload, domain="registered-supplemental-report.v2")
    _write(supplemental_tamper / "supplemental.json", supplemental_payload)
    _rebind_summary(supplemental_tamper)
    supplemental_outputs = tmp_path / "supplemental-tamper-outputs"
    with pytest.raises(ValueError):
        cli.main(
            _freeze_arguments(
                lifecycle,
                supplemental_tamper,
                supplemental_outputs,
            )
        )
    assert not (supplemental_outputs / "confirmatory-v2.json").exists()
