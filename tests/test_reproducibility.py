from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copytree

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ArtifactStore,
    ScientificArtifactError,
    artifact_root_lock,
    read_artifact,
)
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.provenance import collect_provenance
from infinite_rulebook.orchestration.reproducibility import (
    EXECUTION_RECEIPT_FILENAME,
    MAX_EXECUTION_RECEIPT_BYTES,
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    ReproducibilityError,
    ReproducibilityReport,
    _create_execution_receipt_pair,
    _execution_pair_lock,
    authenticate_reproducibility_roots,
    compare_reproducibility_results,
    load_execution_receipt,
    run_reproducibility_check,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def tiny_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="reproducibility-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(
            AgentConfig(AgentKind.FIXED, target_size=1),
            AgentConfig(AgentKind.REWARD, target_size=1),
        ),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="reproducibility-test-seed",
    )


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_mode,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted((root, *root.rglob("*")))
    )


def test_complete_serial_parallel_check_is_authenticated_and_json_safe(
    tmp_path: Path,
) -> None:
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    report = run_reproducibility_check(
        tiny_experiment(),
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )

    raw = report.to_dict()
    json.dumps(raw, allow_nan=False, sort_keys=True)
    assert raw["scientific"]["exact_match"] is True
    assert raw["scientific"]["run_count"] == 2
    assert raw["operational"] == {
        "serial": {
            "artifact_root": str(serial_root.absolute()),
            "max_workers": 1,
        },
        "parallel": {
            "artifact_root": str(parallel_root.absolute()),
            "max_workers": 2,
        },
    }
    assert [run.cell_hash for run in report.runs] == sorted(
        run.cell_hash for run in report.runs
    )
    relocated = replace(
        report,
        serial_root=tmp_path / "relocated-serial",
        parallel_root=tmp_path / "relocated-parallel",
    )
    assert relocated.scientific_hash == report.scientific_hash
    assert str(serial_root.absolute()) not in json.dumps(report.scientific_payload())
    for run in report.runs:
        assert len(run.run_hash) == 64
        assert len(run.scientific_content_hash) == 64

    parsed = ReproducibilityReport.from_dict(
        raw,
        experiment=tiny_experiment(),
        expected_serial_root=serial_root,
        expected_parallel_root=parallel_root,
        expected_parallel_workers=2,
    )
    assert parsed == report
    sorted_json = json.loads(json.dumps(raw, allow_nan=False, sort_keys=True))
    assert (
        ReproducibilityReport.from_dict(
            sorted_json,
            experiment=tiny_experiment(),
            expected_serial_root=serial_root,
            expected_parallel_root=parallel_root,
            expected_parallel_workers=2,
        )
        == report
    )
    before = (_snapshot(serial_root), _snapshot(parallel_root))
    authenticated = authenticate_reproducibility_roots(
        tiny_experiment(),
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )
    assert authenticated == report
    assert (_snapshot(serial_root), _snapshot(parallel_root)) == before


def test_report_ingestion_rejects_noncanonical_or_forged_evidence(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    report = run_reproducibility_check(
        experiment,
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )
    raw = report.to_dict()

    forged_root = deepcopy(raw)
    forged_root["operational"]["serial"]["artifact_root"] = str(
        (tmp_path / "forged").absolute()
    )
    with pytest.raises(ReproducibilityError, match="trusted roots"):
        ReproducibilityReport.from_dict(
            forged_root,
            expected_serial_root=serial_root,
            expected_parallel_root=parallel_root,
            expected_parallel_workers=2,
        )

    same_roots = deepcopy(raw)
    same_roots["operational"]["parallel"]["artifact_root"] = str(serial_root.absolute())
    with pytest.raises(ReproducibilityError, match="non-overlapping"):
        ReproducibilityReport.from_dict(same_roots)

    nested_roots = deepcopy(raw)
    nested_roots["operational"]["parallel"]["artifact_root"] = str(
        (serial_root / "nested").absolute()
    )
    with pytest.raises(ReproducibilityError, match="non-overlapping"):
        ReproducibilityReport.from_dict(nested_roots)

    missing_parallel = deepcopy(raw)
    del missing_parallel["operational"]["parallel"]
    with pytest.raises(ReproducibilityError, match="fields"):
        ReproducibilityReport.from_dict(missing_parallel)

    extra_field = deepcopy(raw)
    extra_field["unexpected"] = True
    with pytest.raises(ReproducibilityError, match="fields"):
        ReproducibilityReport.from_dict(extra_field)

    reordered = deepcopy(raw)
    scientific = reordered["scientific"]
    reordered["scientific"] = {
        key: scientific[key] for key in reversed(tuple(scientific))
    }
    with pytest.raises(ReproducibilityError, match="canonical field order"):
        ReproducibilityReport.from_dict(reordered)

    tampered = deepcopy(raw)
    tampered["scientific"]["runs"][0]["scientific_content_hash"] = "0" * 64
    tampered["scientific_hash"] = scientific_hash(
        {
            "report_format": tampered["report_format"],
            "schema_version": tampered["schema_version"],
            "scientific": tampered["scientific"],
        },
        domain="reproducibility-report",
    )
    with pytest.raises(ReproducibilityError, match="authenticated roots"):
        ReproducibilityReport.from_dict(
            tampered,
            experiment=experiment,
            expected_serial_root=serial_root,
            expected_parallel_root=parallel_root,
            expected_parallel_workers=2,
        )


def test_relative_roots_round_trip_and_reauthenticate_portably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = tiny_experiment()
    serial_root = Path("portable/serial")
    parallel_root = Path("portable/parallel")
    report = run_reproducibility_check(
        experiment,
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )

    raw = json.loads(json.dumps(report.to_dict(), sort_keys=True))
    assert raw["operational"]["serial"]["artifact_root"] == "portable/serial"
    assert raw["operational"]["parallel"]["artifact_root"] == "portable/parallel"
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    copytree(tmp_path / "portable", relocated / "portable")
    monkeypatch.chdir(relocated)
    assert (
        ReproducibilityReport.from_dict(
            raw,
            experiment=experiment,
            expected_serial_root=serial_root,
            expected_parallel_root=parallel_root,
            expected_parallel_workers=2,
        )
        == report
    )
    assert (
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )
        == report
    )


@pytest.mark.parametrize(
    "unsafe_root",
    (
        ".",
        "..",
        "../serial",
        "serial/../escape",
        "./serial",
        "serial//nested",
        "serial/./nested",
        "serial/",
    ),
)
def test_relative_roots_reject_traversal_and_noncanonical_forms(
    tmp_path: Path,
    unsafe_root: str,
) -> None:
    report = run_reproducibility_check(
        tiny_experiment(),
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=2,
    )
    raw = report.to_dict()
    raw["operational"]["serial"]["artifact_root"] = unsafe_root

    with pytest.raises(ReproducibilityError, match="canonical path"):
        ReproducibilityReport.from_dict(raw)
    with pytest.raises((TypeError, ValueError), match="canonical path"):
        authenticate_reproducibility_roots(
            tiny_experiment(),
            serial_root=unsafe_root,
            parallel_root=tmp_path / "other",
            parallel_workers=2,
        )


def test_execution_rejects_copied_or_preexisting_experiment_directories(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    source_serial = tmp_path / "source-serial"
    source_parallel = tmp_path / "source-parallel"
    run_reproducibility_check(
        experiment,
        serial_root=source_serial,
        parallel_root=source_parallel,
        parallel_workers=2,
    )
    reused_serial = tmp_path / "reused-serial"
    fresh_parallel = tmp_path / "fresh-parallel"
    copytree(
        source_serial / experiment.name,
        reused_serial / experiment.name,
    )

    with pytest.raises(ReproducibilityError, match="fresh"):
        run_reproducibility_check(
            experiment,
            serial_root=reused_serial,
            parallel_root=fresh_parallel,
            parallel_workers=2,
        )
    assert not (fresh_parallel / experiment.name).exists()


def test_root_authentication_rejects_missing_alias_and_extra_inventories(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    run_reproducibility_check(
        experiment,
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )

    with pytest.raises(ValueError, match="distinct"):
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial_root,
            parallel_root=serial_root,
            parallel_workers=2,
        )
    with pytest.raises(ValueError, match="non-overlapping"):
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial_root,
            parallel_root=serial_root / "nested",
            parallel_workers=2,
        )
    with pytest.raises(ReproducibilityError, match="cannot inspect"):
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial_root,
            parallel_root=tmp_path / "missing",
            parallel_workers=2,
        )
    extra = parallel_root / experiment.name / ("f" * 64)
    extra.mkdir()
    with pytest.raises(ReproducibilityError, match="extra entries"):
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )


def test_comparison_fails_closed_on_bad_or_incomplete_inventory(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
        provenance=provenance,
    )
    serial = SweepRunner(
        RunExecutor(
            serial_root,
            ExactSymbolicAdapter,
            provenance=provenance,
            reproducibility_mode=True,
        )
    ).run(experiment)
    parallel = SweepRunner(
        RunExecutor(
            parallel_root,
            ExactSymbolicAdapter,
            provenance=provenance,
            reproducibility_mode=True,
        )
    ).run(experiment, max_workers=2)

    report = compare_reproducibility_results(
        experiment,
        serial,
        parallel,
        serial_root=serial_root,
        parallel_root=parallel_root,
        parallel_workers=2,
    )
    assert len(report.runs) == len(experiment.cells())

    with pytest.raises(ReproducibilityError, match="incomplete"):
        compare_reproducibility_results(
            experiment,
            serial[:-1],
            parallel,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )
    with pytest.raises(ReproducibilityError, match="duplicate"):
        compare_reproducibility_results(
            experiment,
            (serial[0], serial[0]),
            parallel,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )
    mismatched = (
        replace(parallel[0], scientific_content_hash="0" * 64),
        *parallel[1:],
    )
    with pytest.raises(ReproducibilityError, match="metadata differs"):
        compare_reproducibility_results(
            experiment,
            serial,
            mismatched,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )

    mixed_serial_root = tmp_path / "mixed-serial"
    different_provenance_root = tmp_path / "different-provenance"
    _create_execution_receipt_pair(
        experiment,
        serial_root=mixed_serial_root,
        parallel_root=different_provenance_root,
        parallel_workers=2,
        provenance=provenance,
    )
    mixed_serial = SweepRunner(
        RunExecutor(
            mixed_serial_root,
            ExactSymbolicAdapter,
            provenance=provenance,
            reproducibility_mode=True,
        )
    ).run(experiment)
    different_provenance = SweepRunner(
        RunExecutor(
            different_provenance_root,
            ExactSymbolicAdapter,
            provenance=replace(provenance, code_commit="different"),
            reproducibility_mode=True,
        )
    ).run(experiment, max_workers=2)
    with pytest.raises(ReproducibilityError, match="receipt provenance"):
        compare_reproducibility_results(
            experiment,
            mixed_serial,
            different_provenance,
            serial_root=mixed_serial_root,
            parallel_root=different_provenance_root,
            parallel_workers=2,
        )

    class DivergentCheckpointAdapter(ExactSymbolicAdapter):
        def checkpoint(self, *args, **kwargs):
            payload = super().checkpoint(*args, **kwargs)
            return {**payload, "divergent_test_marker": True}

    divergent_serial_root = tmp_path / "divergent-serial"
    divergent_root = tmp_path / "divergent-content"
    _create_execution_receipt_pair(
        experiment,
        serial_root=divergent_serial_root,
        parallel_root=divergent_root,
        parallel_workers=2,
        provenance=provenance,
    )
    divergent_serial = SweepRunner(
        RunExecutor(
            divergent_serial_root,
            ExactSymbolicAdapter,
            provenance=provenance,
            reproducibility_mode=True,
        )
    ).run(experiment)
    divergent = SweepRunner(
        RunExecutor(
            divergent_root,
            DivergentCheckpointAdapter,
            provenance=provenance,
            reproducibility_mode=True,
        )
    ).run(experiment, max_workers=2)
    with pytest.raises(ReproducibilityError, match="scientific content mismatch"):
        compare_reproducibility_results(
            experiment,
            divergent_serial,
            divergent,
            serial_root=divergent_serial_root,
            parallel_root=divergent_root,
            parallel_workers=2,
        )

    unexpected = parallel_root / experiment.name / ("f" * 64)
    unexpected.mkdir()
    with pytest.raises(ReproducibilityError, match="unexpected"):
        compare_reproducibility_results(
            experiment,
            serial,
            parallel,
            serial_root=serial_root,
            parallel_root=parallel_root,
            parallel_workers=2,
        )


def test_distinct_roots_and_real_parallelism_are_required(tmp_path: Path) -> None:
    experiment = tiny_experiment()
    with pytest.raises(ValueError, match="distinct"):
        run_reproducibility_check(
            experiment,
            serial_root=tmp_path,
            parallel_root=tmp_path,
        )
    with pytest.raises(ValueError, match="at least 2"):
        run_reproducibility_check(
            experiment,
            serial_root=tmp_path / "serial",
            parallel_root=tmp_path / "parallel",
            parallel_workers=1,
        )


def test_receipts_are_role_bound_while_scientific_runs_stay_stable(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    first = run_reproducibility_check(
        experiment,
        serial_root=tmp_path / "first-serial",
        parallel_root=tmp_path / "first-parallel",
        parallel_workers=2,
        provenance=provenance,
    )
    second = run_reproducibility_check(
        experiment,
        serial_root=tmp_path / "second-serial",
        parallel_root=tmp_path / "second-parallel",
        parallel_workers=3,
        provenance=provenance,
    )

    assert first.invocation_id != second.invocation_id
    assert first.scientific_hash != second.scientific_hash
    assert first.runs == second.runs
    serial = load_execution_receipt(tmp_path / "first-serial")
    parallel = load_execution_receipt(tmp_path / "first-parallel")
    assert (serial.role, serial.max_workers) == ("serial", 1)
    assert (parallel.role, parallel.max_workers) == ("parallel", 2)
    assert serial.invocation_id == parallel.invocation_id == first.invocation_id
    assert serial.scientific_hash == first.serial_receipt_hash
    assert parallel.scientific_hash == first.parallel_receipt_hash


def test_receipt_publication_rejects_symlinked_ancestors_without_external_writes(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReproducibilityError, match="immutable execution receipt"):
        _create_execution_receipt_pair(
            experiment,
            serial_root=alias / "serial",
            parallel_root=tmp_path / "parallel",
            parallel_workers=2,
            provenance=collect_provenance(),
        )

    assert tuple(outside.iterdir()) == ()
    assert not (tmp_path / "parallel").exists()


def test_execution_receipts_and_report_runs_are_deeply_immutable(
    tmp_path: Path,
) -> None:
    report = run_reproducibility_check(
        tiny_experiment(),
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=2,
    )
    serial = load_execution_receipt(tmp_path / "serial")
    parallel = load_execution_receipt(tmp_path / "parallel")
    original_hash = serial.scientific_hash

    with pytest.raises(TypeError):
        serial.pair["serial"] = "0" * 64  # type: ignore[index]
    assert serial.scientific_hash == original_hash
    assert serial.pair is not parallel.pair

    mutable_runs = list(report.runs)
    normalized = ReproducibilityReport(
        config_hash=report.config_hash,
        runs=mutable_runs,  # type: ignore[arg-type]
        serial_root=report.serial_root,
        parallel_root=report.parallel_root,
        parallel_workers=report.parallel_workers,
        invocation_id=report.invocation_id,
        serial_receipt_hash=report.serial_receipt_hash,
        parallel_receipt_hash=report.parallel_receipt_hash,
    )
    mutable_runs.clear()
    assert normalized.runs == report.runs


def test_calibration_receipts_bind_exact_smoke_prerequisite_and_resume(
    tmp_path: Path,
) -> None:
    experiment = replace(tiny_experiment(), phase="calibration")
    smoke_hash = scientific_hash("smoke-a", domain="test-smoke-prerequisite")
    other_smoke_hash = scientific_hash("smoke-b", domain="test-smoke-prerequisite")
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    with pytest.raises(ReproducibilityError, match="Stage-0 prerequisite"):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
        )

    run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        smoke_prerequisite_hash=smoke_hash,
    )
    assert load_execution_receipt(serial).smoke_prerequisite_hash == smoke_hash
    assert load_execution_receipt(parallel).smoke_prerequisite_hash == smoke_hash

    with pytest.raises(ReproducibilityError, match="different Stage-0"):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
            resume=True,
            smoke_prerequisite_hash=other_smoke_hash,
        )


def test_workflow_root_ownership_rejects_competing_modes_without_waiting(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    with (
        artifact_root_lock(serial, shared=True),
        pytest.raises(ReproducibilityError, match="already owned"),
    ):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
        )
    assert not (serial / REPRODUCIBILITY_OPERATIONAL_DIRECTORY).exists()
    assert not (parallel / REPRODUCIBILITY_OPERATIONAL_DIRECTORY).exists()

    with (
        artifact_root_lock(serial),
        pytest.raises(ScientificArtifactError, match="already owned"),
    ):
        RunExecutor(serial, ExactSymbolicAdapter).execute(
            experiment,
            experiment.cells()[0],
        )


def test_copied_serial_root_cannot_masquerade_as_parallel_root(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
    )
    copied_serial = tmp_path / "copied-serial-as-parallel"
    copytree(serial, copied_serial)

    with pytest.raises(ReproducibilityError, match="root role"):
        authenticate_reproducibility_roots(
            experiment,
            serial_root=serial,
            parallel_root=copied_serial,
            parallel_workers=2,
        )


def test_receipt_bytes_and_size_are_fail_closed(tmp_path: Path) -> None:
    experiment = tiny_experiment()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
    )
    receipt_path = (
        serial / REPRODUCIBILITY_OPERATIONAL_DIRECTORY / EXECUTION_RECEIPT_FILENAME
    )
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    with pytest.raises(ReproducibilityError, match="canonical"):
        load_execution_receipt(serial)

    parallel_receipt_path = (
        parallel / REPRODUCIBILITY_OPERATIONAL_DIRECTORY / EXECUTION_RECEIPT_FILENAME
    )
    parallel_receipt_path.chmod(0o600)
    parallel_receipt_path.write_bytes(b"x" * (MAX_EXECUTION_RECEIPT_BYTES + 1))
    with pytest.raises(ReproducibilityError, match="bounded"):
        load_execution_receipt(parallel)


def test_environment_fingerprint_reproduces_the_receipt_digest() -> None:
    provenance = collect_provenance()
    payload = json.loads(provenance.environment_fingerprint)
    payload["machine"] = "tampered-machine"
    tampered = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    with pytest.raises(ValueError, match="does not reproduce"):
        replace(provenance, environment_fingerprint=tampered)


def test_resume_completes_an_authenticated_mid_run_interruption(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )
    partial = RunExecutor(
        serial,
        ExactSymbolicAdapter,
        provenance=provenance,
        reproducibility_mode=True,
    ).execute(
        experiment,
        experiment.cells()[0],
        stop_after_new_events=1,
    )
    assert partial.complete is False
    with pytest.raises(ScientificArtifactError, match="paired reproducibility"):
        RunExecutor(
            serial,
            ExactSymbolicAdapter,
            provenance=provenance,
        ).execute(experiment, experiment.cells()[1])

    report = run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
        resume=True,
    )
    assert len(report.runs) == len(experiment.cells())


@pytest.mark.parametrize("tree", ("run", "frontier"))
def test_authenticated_resume_removes_orphaned_publication_temporaries(
    tmp_path: Path,
    tree: str,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )
    partial = RunExecutor(
        serial,
        ExactSymbolicAdapter,
        provenance=provenance,
        reproducibility_mode=True,
    ).execute(
        experiment,
        experiment.cells()[0],
        stop_after_new_events=1,
    )
    if tree == "run":
        temporary = partial.path / ".metrics.json.0123456789abcdef01234567"
    else:
        reference = read_artifact(partial.path / "frontier-reference.json")
        temporary = (
            serial
            / "_frontiers"
            / reference.payload["frontier_hash"]
            / "frontier"
            / ".diagnostics.json.0123456789abcdef01234567"
        )
    temporary.write_text('{"interrupted":', encoding="utf-8")

    report = run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
        resume=True,
    )

    assert report.scientific_payload()["exact_match"] is True
    assert not temporary.exists()


def test_resume_rejects_wrong_workers_and_concurrent_invocation(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )

    with pytest.raises(ReproducibilityError, match="matching pair"):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=3,
            provenance=provenance,
            resume=True,
        )
    with (
        _execution_pair_lock(serial, parallel),
        pytest.raises(ReproducibilityError, match="already running"),
    ):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
            provenance=provenance,
            resume=True,
        )


def test_failed_receipt_republication_preserves_the_existing_pair(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )
    before = (load_execution_receipt(serial), load_execution_receipt(parallel))

    with pytest.raises(ReproducibilityError, match="publish"):
        _create_execution_receipt_pair(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
            provenance=provenance,
        )

    assert (load_execution_receipt(serial), load_execution_receipt(parallel)) == before
    assert {path.name for root in (serial, parallel) for path in root.iterdir()} == {
        REPRODUCIBILITY_OPERATIONAL_DIRECTORY
    }


def test_resume_preflight_rejects_unexpected_partial_artifact_without_mutation(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )
    partial = RunExecutor(
        serial,
        ExactSymbolicAdapter,
        provenance=provenance,
        reproducibility_mode=True,
    ).execute(
        experiment,
        experiment.cells()[0],
        stop_after_new_events=1,
    )
    ArtifactStore(partial.path).write(
        "unexpected.json",
        "unexpected-test-artifact",
        read_artifact(partial.path / "config.resolved.json").semantic_hashes,
        {"unexpected": True},
    )
    before = (_snapshot(serial), _snapshot(parallel))

    with pytest.raises(ReproducibilityError, match="unexpected artifact"):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
            provenance=provenance,
            resume=True,
        )
    assert (_snapshot(serial), _snapshot(parallel)) == before


def test_resume_preflight_binds_partial_run_to_frontier_without_mutation(
    tmp_path: Path,
) -> None:
    experiment = tiny_experiment()
    provenance = collect_provenance()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _create_execution_receipt_pair(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
        provenance=provenance,
    )
    partial = RunExecutor(
        serial,
        ExactSymbolicAdapter,
        provenance=provenance,
        reproducibility_mode=True,
    ).execute(
        experiment,
        experiment.cells()[0],
        stop_after_new_events=1,
    )
    reference_path = partial.path / "frontier-reference.json"
    reference = read_artifact(reference_path)
    changed = ArtifactEnvelope.create(
        reference.artifact_type,
        reference.semantic_hashes,
        {**reference.payload, "artifact_hash": "0" * 64},
    )
    reference_path.chmod(0o600)
    reference_path.write_text(
        json.dumps(changed.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = (_snapshot(serial), _snapshot(parallel))

    with pytest.raises(ReproducibilityError, match="different frontier"):
        run_reproducibility_check(
            experiment,
            serial_root=serial,
            parallel_root=parallel,
            parallel_workers=2,
            provenance=provenance,
            resume=True,
        )
    assert (_snapshot(serial), _snapshot(parallel)) == before


@pytest.mark.parametrize(
    "reserved",
    ("_frontiers", REPRODUCIBILITY_OPERATIONAL_DIRECTORY),
)
def test_experiment_names_reserve_operational_root_directories(
    reserved: str,
) -> None:
    with pytest.raises(ValueError, match="reserved artifact path"):
        replace(tiny_experiment(), name=reserved)


def test_environment_digest_binds_machine_and_float_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.orchestration.provenance as provenance_module

    original = collect_provenance().environment_digest
    monkeypatch.setattr(
        provenance_module.platform,
        "machine",
        lambda: "scientifically-different-machine",
    )

    assert collect_provenance().environment_digest != original
