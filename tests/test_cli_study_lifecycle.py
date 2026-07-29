from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import infinite_rulebook.cli as cli
from infinite_rulebook.analysis import (
    AnalysisPhase,
    AnalysisReport,
    CanaryReport,
    ContrastResult,
    EquivalenceResult,
    ExactInterval,
    HolmDecision,
    load_analysis_plan,
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
from infinite_rulebook.studies.symbolic_construct import (
    build_symbolic_canary_plan,
    calibration_evidence_hash_from_hashes,
    expected_confirmatory_registration,
)

_CONFIG = Path("configs/symbolic-calibration-v1.json")
_TEST_POWER_SIMULATIONS = 2_000
_SMOKE_PAYLOAD = {"fixture": "authenticated-smoke-prerequisite"}
_SMOKE_PREREQUISITE_HASH = scientific_hash(
    _SMOKE_PAYLOAD,
    domain="symbolic-smoke-prerequisite",
)


@dataclass(frozen=True)
class _Lifecycle:
    config: ExperimentConfig
    analysis_plan: Path
    canary_plan: Path
    report_root: Path
    reproducibility: ReproducibilityReport
    serial_inventory: RawArtifactInventory
    parallel_inventory: RawArtifactInventory
    smoke: SimpleNamespace
    dataset: SimpleNamespace
    calls: dict[str, list[dict[str, Any]]]


def _fake_report(
    dataset: SimpleNamespace,
    plan: Any,
    *,
    canary_report_hash: str,
    canaries_passed: bool,
    config_hash: str,
) -> AnalysisReport:
    environment_replicas = load_experiment_config(_CONFIG).environment_replicas
    algorithm_replicas = load_experiment_config(_CONFIG).algorithm_replicas
    contrasts = tuple(
        ContrastResult(
            name=spec.name,
            metric=spec.metric,
            left_label=spec.left.label,
            right_label=spec.right.label,
            checkpoint=spec.checkpoint,
            alternative=spec.alternative,
            null_margin=spec.null_margin,
            pair_count=environment_replicas,
            cell_pair_count=environment_replicas * algorithm_replicas,
            differences=(cli.PRIMARY_MINIMUM_EFFECTS[spec.name],)
            * environment_replicas,
            mean_difference=cli.PRIMARY_MINIMUM_EFFECTS[spec.name],
            median_difference=cli.PRIMARY_MINIMUM_EFFECTS[spec.name],
            standardized_mean_difference=None,
            median_interval=ExactInterval(
                cli.PRIMARY_MINIMUM_EFFECTS[spec.name],
                cli.PRIMARY_MINIMUM_EFFECTS[spec.name],
                0.95,
                "lifecycle-fixture",
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
            pair_count=environment_replicas,
            cell_pair_count=environment_replicas * algorithm_replicas,
            differences=(0.0,) * environment_replicas,
            mean_difference=0.0,
            median_difference=0.0,
            median_interval=ExactInterval(
                0.0,
                0.0,
                0.95,
                "lifecycle-fixture",
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
            HolmDecision(spec.name, 0.001, 0.005, plan.family_alpha, True)
            for spec in plan.contrasts
        ),
        equivalence_decisions=tuple(
            HolmDecision(spec.name, 0.001, 0.001, plan.family_alpha, True)
            for spec in plan.equivalences
        ),
        run_settings_hash=scientific_hash(
            load_experiment_config(_CONFIG).resolved_run_settings(),
            domain="resolved-run-settings",
        ),
        provenance=dataset.provenance,
        canary_report_hash=canary_report_hash,
        canaries_passed=canaries_passed,
        config_hash=config_hash,
    )


def _prepare_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Lifecycle:
    config = load_experiment_config(_CONFIG)
    analysis_plan = tmp_path / "registered-analysis.json"
    canary_plan = tmp_path / "registered-canaries.json"
    assert (
        cli.main(
            [
                "plan",
                str(_CONFIG),
                str(analysis_plan),
                str(canary_plan),
            ]
        )
        == 0
    )

    runs = tuple(
        ReproducibilityRun(
            cell_hash=cell.cell_hash,
            run_hash=scientific_hash(
                cell.cell_hash,
                domain="cli-lifecycle-run",
            ),
            scientific_content_hash=scientific_hash(
                cell.cell_hash,
                domain="cli-lifecycle-content",
            ),
        )
        for cell in config.cells()
    )
    receipt_provenance = collect_provenance()
    serial_receipt, parallel_receipt = _create_execution_receipt_pair(
        config,
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=4,
        provenance=receipt_provenance,
        smoke_prerequisite_hash=_SMOKE_PREREQUISITE_HASH,
    )
    reproducibility = ReproducibilityReport(
        config_hash=config.config_hash,
        runs=runs,
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=4,
        invocation_id=serial_receipt.invocation_id,
        serial_receipt_hash=serial_receipt.scientific_hash,
        parallel_receipt_hash=parallel_receipt.scientific_hash,
    )
    reproducibility_path = tmp_path / "reproducibility.json"
    reproducibility_path.write_text(
        json.dumps(reproducibility.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = tuple(sorted(collect_provenance().to_dict().items()))
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
            domain="cli-lifecycle-dataset",
        ),
    )

    def raw_inventory(side: str) -> RawArtifactInventory:
        frontier_hash = scientific_hash("frontier", domain="cli-lifecycle-frontier")
        trees = tuple(
            sorted(
                (
                    RawArtifactTree(
                        tree_type="frontier",
                        path=f"_frontiers/{frontier_hash}",
                        identity_hash=frontier_hash,
                        scientific_content_hash=scientific_hash(
                            "frontier-content",
                            domain="cli-lifecycle-frontier-content",
                        ),
                        file_count=1,
                        byte_size=0,
                        tree_byte_sha256=scientific_hash(
                            (side, "frontier-tree"),
                            domain="cli-lifecycle-tree-bytes",
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
                                (side, "run-tree", run.run_hash),
                                domain="cli-lifecycle-tree-bytes",
                            ),
                        )
                        for run in runs
                    ),
                ),
                key=lambda tree: tree.path,
            )
        )
        body = {
            "artifact_type": "infinite-rulebook-raw-artifact-inventory",
            "schema_version": 2,
            "experiment_name": config.name,
            "config_hash": config.config_hash,
            "side": side,
            "execution_receipt": {
                "serial": serial_receipt,
                "parallel": parallel_receipt,
            }[side].to_dict(),
            "trees": [tree.to_dict() for tree in trees],
        }
        return RawArtifactInventory(
            experiment_name=config.name,
            config_hash=config.config_hash,
            side=side,
            trees=trees,
            execution_receipt={
                "serial": serial_receipt,
                "parallel": parallel_receipt,
            }[side],
            scientific_hash=scientific_hash(
                body,
                domain="raw-artifact-inventory",
            ),
        )

    serial_inventory = raw_inventory("serial")
    parallel_inventory = raw_inventory("parallel")
    smoke_config = SimpleNamespace(
        config_hash=scientific_hash(
            "smoke-config",
            domain="cli-lifecycle-smoke",
        )
    )
    smoke_reproducibility = SimpleNamespace(
        scientific_hash=scientific_hash(
            "smoke-reproducibility",
            domain="cli-lifecycle-smoke",
        ),
        serial_root=tmp_path / "smoke-serial",
        parallel_root=tmp_path / "smoke-parallel",
    )
    smoke_serial_inventory = SimpleNamespace(
        scientific_hash=scientific_hash(
            "smoke-serial-inventory",
            domain="cli-lifecycle-smoke",
        )
    )
    smoke_parallel_inventory = SimpleNamespace(
        scientific_hash=scientific_hash(
            "smoke-parallel-inventory",
            domain="cli-lifecycle-smoke",
        )
    )
    smoke_payload = dict(_SMOKE_PAYLOAD)
    smoke = SimpleNamespace(
        config=smoke_config,
        reproducibility=smoke_reproducibility,
        serial_inventory=smoke_serial_inventory,
        parallel_inventory=smoke_parallel_inventory,
        engineering_anomalies=(),
        scientific_hash=_SMOKE_PREREQUISITE_HASH,
        to_dict=lambda: smoke_payload,
    )
    calls: dict[str, list[dict[str, Any]]] = {
        "load": [],
        "build": [],
        "power": [],
    }
    original_calibrate = cli.calibrate_environment_count

    def completed_roots(_root: Path, experiment: str) -> tuple[Path, ...]:
        assert experiment == config.name
        return tuple(tmp_path / "runs" / run.run_hash for run in runs)

    def load_run_trees(
        roots: tuple[Path, ...],
        *,
        expected_phase: AnalysisPhase,
        expected_freeze_hash: str | None,
        expected_run_settings: dict[str, Any],
    ) -> SimpleNamespace:
        calls["load"].append(
            {
                "root_count": len(roots),
                "expected_phase": expected_phase,
                "expected_freeze_hash": expected_freeze_hash,
                "expected_run_settings": expected_run_settings,
            }
        )
        return dataset

    def evaluate_canaries(
        loaded: SimpleNamespace,
        registered: Any,
    ) -> CanaryReport:
        assert loaded is dataset
        return CanaryReport(
            phase=AnalysisPhase.CALIBRATION,
            dataset_hash=dataset.scientific_hash,
            plan_hash=registered.scientific_hash,
            results=(),
        )

    def build_report(
        loaded: SimpleNamespace,
        plan: Any,
        *,
        canary_report_hash: str,
        canaries_passed: bool,
        config_hash: str,
    ) -> AnalysisReport:
        calls["build"].append(
            {
                "canary_report_hash": canary_report_hash,
                "canaries_passed": canaries_passed,
                "config_hash": config_hash,
            }
        )
        return _fake_report(
            loaded,
            plan,
            canary_report_hash=canary_report_hash,
            canaries_passed=canaries_passed,
            config_hash=config_hash,
        )

    def calibrate(*args: Any, **kwargs: Any) -> Any:
        calls["power"].append(dict(kwargs))
        return original_calibrate(*args, **kwargs)

    def create_inventory(
        _cls: type[RawArtifactInventory],
        _root: Path,
        experiment: ExperimentConfig,
        *,
        side: str,
    ) -> RawArtifactInventory:
        assert experiment == config
        return {
            "serial": serial_inventory,
            "parallel": parallel_inventory,
        }[side]

    def verify_inventory(
        inventory: RawArtifactInventory,
        _root: Path,
        experiment: ExperimentConfig,
        *,
        side: str | None = None,
    ) -> None:
        assert experiment == config
        assert side is None or side == inventory.side

    monkeypatch.setattr(cli, "_completed_run_roots", completed_roots)
    monkeypatch.setattr(
        cli,
        "_load_reproducibility_report",
        lambda *_args, **_kwargs: reproducibility,
    )
    monkeypatch.setattr(
        cli,
        "_load_smoke_prerequisite",
        lambda *_args, **_kwargs: smoke,
    )
    monkeypatch.setattr(cli, "load_run_trees", load_run_trees)
    monkeypatch.setattr(cli, "evaluate_canaries", evaluate_canaries)
    monkeypatch.setattr(cli, "build_report", build_report)
    monkeypatch.setattr(cli, "calibrate_environment_count", calibrate)
    monkeypatch.setattr(
        RawArtifactInventory,
        "create",
        classmethod(create_inventory),
    )
    monkeypatch.setattr(RawArtifactInventory, "verify", verify_inventory)
    monkeypatch.setattr(cli, "POWER_SIMULATIONS", _TEST_POWER_SIMULATIONS)
    monkeypatch.setattr(cli, "POWER_CANDIDATE_ENVIRONMENTS", (32,))

    report_root = tmp_path / "calibration-release"
    assert (
        cli.main(
            [
                "report",
                str(_CONFIG),
                str(analysis_plan),
                str(canary_plan),
                str(tmp_path / "parallel"),
                str(report_root),
                "--power-simulations",
                str(_TEST_POWER_SIMULATIONS),
                "--reproducibility-report",
                str(reproducibility_path),
                "--smoke-evidence",
                str(tmp_path / "smoke-prerequisite.json"),
            ]
        )
        == 0
    )
    return _Lifecycle(
        config=config,
        analysis_plan=analysis_plan,
        canary_plan=canary_plan,
        report_root=report_root,
        reproducibility=reproducibility,
        serial_inventory=serial_inventory,
        parallel_inventory=parallel_inventory,
        smoke=smoke,
        dataset=dataset,
        calls=calls,
    )


def _freeze(
    lifecycle: _Lifecycle,
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    output_config = tmp_path / "symbolic-confirmatory-v1.json"
    output_plan = tmp_path / "symbolic-confirmatory-analysis-v1.json"
    output_canaries = tmp_path / "symbolic-confirmatory-canaries-v1.json"
    cli.main(
        [
            "freeze",
            str(_CONFIG),
            str(lifecycle.report_root / "summary.json"),
            str(output_config),
            str(output_plan),
            str(output_canaries),
        ]
    )
    return output_config, output_plan, output_canaries


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash(payload: dict[str, Any], *, domain: str) -> None:
    body = {key: value for key, value in payload.items() if key != "scientific_hash"}
    payload["scientific_hash"] = scientific_hash(body, domain=domain)


def _reseal_calibration_package(
    lifecycle: _Lifecycle,
    *,
    analysis: dict[str, Any],
    canaries: dict[str, Any],
    power: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    root = lifecycle.report_root
    _rewrite(root / "analysis.json", analysis)
    _rewrite(root / "canaries.json", canaries)
    _rewrite(root / "power.json", power)
    _rewrite(root / "summary.json", summary)
    old_release = load_study_release_manifest(
        root / STUDY_RELEASE_MANIFEST_FILENAME,
        verify_files=False,
    )
    release = StudyReleaseManifest.create(
        root,
        tuple(member.path for member in old_release.members),
        phase="calibration",
        study_contract=old_release.study_contract,
        config_hash=lifecycle.config.config_hash,
        freeze_hash=None,
        calibration_evidence_hash=summary["calibration_evidence_hash"],
    )
    _rewrite(root / STUDY_RELEASE_MANIFEST_FILENAME, release.to_dict())
    release.verify_files(root)


def test_report_release_freeze_lifecycle_serializes_and_binds_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _prepare_report(tmp_path, monkeypatch)
    root = lifecycle.report_root
    summary = _read(root / "summary.json")
    power = _read(root / "power.json")
    release = load_study_release_manifest(root / STUDY_RELEASE_MANIFEST_FILENAME)

    rows = tuple(
        csv.DictReader(
            io.StringIO((root / "power-calibration.csv").read_text(encoding="utf-8"))
        )
    )
    assert len(rows) == 6
    assert {row["hypothesis_family"] for row in rows} == {
        "directional",
        "equivalence",
    }
    assert {row["power_calibration_hash"] for row in rows} == {power["scientific_hash"]}
    assert summary["analysis_hash"] == _read(root / "analysis.json")["scientific_hash"]
    assert summary["canary_hash"] == _read(root / "canaries.json")["scientific_hash"]
    assert summary["power_hash"] == power["scientific_hash"]
    assert summary["reproducibility_hash"] == (
        lifecycle.reproducibility.scientific_hash
    )
    assert summary["raw_serial_inventory_hash"] == (
        lifecycle.serial_inventory.scientific_hash
    )
    assert summary["raw_parallel_inventory_hash"] == (
        lifecycle.parallel_inventory.scientific_hash
    )
    assert summary["calibration_evidence_hash"] == (
        calibration_evidence_hash_from_hashes(
            config_hash=lifecycle.config.config_hash,
            analysis_report_hash=summary["analysis_hash"],
            canary_report_hash=summary["canary_hash"],
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
    assert summary["freeze_eligible"] is True
    assert release.calibration_evidence_hash == summary["calibration_evidence_hash"]
    assert {member.path for member in release.members} == set(
        cli._expected_release_members(AnalysisPhase.CALIBRATION)
    )

    output_config, output_plan, output_canaries = _freeze(lifecycle, tmp_path)
    sealed = load_experiment_config(output_config)
    assert sealed.confirmatory_freeze is not None
    assert (
        sealed.confirmatory_freeze.calibration_evidence_hash
        == (summary["calibration_evidence_hash"])
    )
    assert sealed.environment_replicas == 32
    assert load_analysis_plan(output_plan).registration_hash == (
        expected_confirmatory_registration(sealed)
    )
    assert (
        _read(output_canaries)
        == build_symbolic_canary_plan(
            sealed,
            phase=AnalysisPhase.CONFIRMATORY,
        ).to_dict()
    )

    assert len(lifecycle.calls["load"]) == 2
    assert all(
        call["root_count"] == len(lifecycle.config.cells())
        for call in lifecycle.calls["load"]
    )
    assert len(lifecycle.calls["build"]) == 2
    assert all(call["canaries_passed"] is True for call in lifecycle.calls["build"])
    assert len(lifecycle.calls["power"]) == 2
    assert all(
        call["candidate_environment_counts"] == (32,)
        and call["simulations"] == _TEST_POWER_SIMULATIONS
        and "equivalence_hypotheses" in call
        and "simulation_error_alpha" in call
        for call in lifecycle.calls["power"]
    )


def test_calibration_report_rejects_a_substituted_nonreceipt_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _prepare_report(tmp_path, monkeypatch)
    substituted_output = tmp_path / "substituted-release"

    with pytest.raises(ValueError, match="exact receipt-bound"):
        cli.main(
            [
                "report",
                str(_CONFIG),
                str(lifecycle.analysis_plan),
                str(lifecycle.canary_plan),
                str(tmp_path / "substituted-artifacts"),
                str(substituted_output),
                "--power-simulations",
                str(_TEST_POWER_SIMULATIONS),
                "--reproducibility-report",
                str(tmp_path / "reproducibility.json"),
                "--smoke-evidence",
                str(tmp_path / "smoke-prerequisite.json"),
            ]
        )
    assert not substituted_output.exists()


def test_freeze_rejects_release_member_tampering_before_writing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _prepare_report(tmp_path, monkeypatch)
    analysis_markdown = lifecycle.report_root / "analysis.md"
    analysis_markdown.write_text(
        analysis_markdown.read_text(encoding="utf-8") + "\ntampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="members do not match"):
        _freeze(lifecycle, tmp_path)
    assert not (tmp_path / "symbolic-confirmatory-v1.json").exists()


def test_freeze_rejects_boolean_deviation_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _prepare_report(tmp_path, monkeypatch)
    root = lifecycle.report_root
    deviations = _read(root / "deviations.json")
    summary = _read(root / "summary.json")
    deviations["schema_version"] = True
    _rehash(deviations, domain="symbolic-study-deviation-log")
    _rewrite(root / "deviations.json", deviations)
    summary["deviation_log_hash"] = deviations["scientific_hash"]
    summary["calibration_evidence_hash"] = calibration_evidence_hash_from_hashes(
        config_hash=lifecycle.config.config_hash,
        analysis_report_hash=summary["analysis_hash"],
        canary_report_hash=summary["canary_hash"],
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
    _rewrite(root / "summary.json", summary)
    old_release = load_study_release_manifest(
        root / STUDY_RELEASE_MANIFEST_FILENAME,
        verify_files=False,
    )
    release = StudyReleaseManifest.create(
        root,
        tuple(member.path for member in old_release.members),
        phase="calibration",
        study_contract=old_release.study_contract,
        config_hash=lifecycle.config.config_hash,
        freeze_hash=None,
        calibration_evidence_hash=summary["calibration_evidence_hash"],
    )
    _rewrite(root / STUDY_RELEASE_MANIFEST_FILENAME, release.to_dict())

    with pytest.raises(ValueError, match="deviation log"):
        _freeze(lifecycle, tmp_path)
    assert not (tmp_path / "symbolic-confirmatory-v1.json").exists()


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("analysis", "analysis evidence does not derive from raw roots"),
        ("canaries", "canary evidence does not derive from raw roots"),
    ],
)
def test_freeze_rejects_rehashed_resealed_evidence_not_derived_from_raw_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    message: str,
) -> None:
    lifecycle = _prepare_report(tmp_path, monkeypatch)
    root = lifecycle.report_root
    analysis = _read(root / "analysis.json")
    canaries = _read(root / "canaries.json")
    power = _read(root / "power.json")
    summary = _read(root / "summary.json")

    if component == "analysis":
        contrast = analysis["contrasts"][0]
        contrast["differences"] = [0.5] * lifecycle.config.environment_replicas
        contrast["mean_difference"] = 0.5
        contrast["median_difference"] = 0.5
        contrast["median_interval"]["lower"] = 0.5
        contrast["median_interval"]["upper"] = 0.5
        _rehash(analysis, domain="registered-analysis-report")
        hypotheses, equivalences = cli._reconstruct_registered_power(
            analysis,
            lifecycle.config.environment_replicas,
            lifecycle.config.algorithm_replicas,
        )
        calibration = cli.calibrate_environment_count(
            hypotheses,
            candidate_environment_counts=(32,),
            equivalence_hypotheses=equivalences,
            simulations=_TEST_POWER_SIMULATIONS,
            seed=cli.POWER_SEED,
            alpha=cli.POWER_ALPHA,
            minimum_power=cli.MINIMUM_INDIVIDUAL_POWER,
            minimum_equivalence_power=cli.MINIMUM_EQUIVALENCE_POWER,
            minimum_joint_power=cli.MINIMUM_JOINT_POWER,
            maximum_fwer=cli.MAXIMUM_GLOBAL_NULL_FWER,
            simulation_error_alpha=cli.POWER_SIMULATION_ERROR_ALPHA,
        )
        power["hypotheses"] = cli._jsonable(hypotheses)
        power["equivalence_hypotheses"] = cli._jsonable(equivalences)
        power["calibration"] = cli._jsonable(calibration)
        _rehash(power, domain="symbolic-power-calibration")
    else:
        canaries["results"] = [{"fabricated-but-passing": True}]
        _rehash(canaries, domain="scientific-canary-report")

    summary["analysis_hash"] = analysis["scientific_hash"]
    summary["canary_hash"] = canaries["scientific_hash"]
    summary["power_hash"] = power["scientific_hash"]
    summary["calibration_evidence_hash"] = calibration_evidence_hash_from_hashes(
        config_hash=lifecycle.config.config_hash,
        analysis_report_hash=analysis["scientific_hash"],
        canary_report_hash=canaries["scientific_hash"],
        power_calibration_hash=power["scientific_hash"],
        reproducibility_report_hash=lifecycle.reproducibility.scientific_hash,
        raw_serial_inventory_hash=lifecycle.serial_inventory.scientific_hash,
        raw_parallel_inventory_hash=lifecycle.parallel_inventory.scientific_hash,
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
    _reseal_calibration_package(
        lifecycle,
        analysis=analysis,
        canaries=canaries,
        power=power,
        summary=summary,
    )

    with pytest.raises(ValueError, match=message):
        _freeze(lifecycle, tmp_path)
    assert not (tmp_path / "symbolic-confirmatory-v1.json").exists()
