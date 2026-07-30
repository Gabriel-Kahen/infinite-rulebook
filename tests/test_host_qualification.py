from __future__ import annotations

import copy
import json
import math
import os
import runpy
import signal
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import infinite_rulebook.operations.host_qualification as qualification
import scripts.qualify_symbolic_v2_host as qualification_cli
from infinite_rulebook.operations.host_qualification import (
    GIB,
    V2_E192_DATASET_HASH,
    V2_E768_DATASET_HASH,
    V2_PROBE_CONFIG_HASH,
    V2_PROBE_DATASET_HASH,
    assess_qualification,
    bind_probe_result,
    build_exact_plan,
    inspect_host,
    run_capacity_benchmark,
    run_probe_step,
    validate_output_path,
    verify_assessment_bundle,
    verify_record,
    write_record,
)
from infinite_rulebook.orchestration.artifacts import artifact_root_lock

COMMIT = qualification.APPROVED_V2_EXECUTION_COMMITS[0]


def _fake_host_files(
    root: Path,
    storage: Path,
    *,
    physical_gib: int = 96,
    filesystem: str = "ext4",
    device: str = "/dev/nvme0n1p1",
) -> tuple[Path, Path, Path]:
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc / "meminfo").write_text(
        "\n".join(
            (
                f"MemTotal: {physical_gib * GIB // 1024} kB",
                f"MemAvailable: {80 * GIB // 1024} kB",
                "SwapTotal: 0 kB",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (proc / "vmstat").write_text(
        "pswpin 10\npswpout 20\n",
        encoding="utf-8",
    )
    (proc / "sys" / "kernel" / "random" / "boot_id").write_text(
        "11111111-2222-3333-4444-555555555555\n",
        encoding="utf-8",
    )
    (proc / "self" / "mountinfo").write_text(
        f"36 25 259:1 / {storage} rw,noatime - {filesystem} {device} rw\n",
        encoding="utf-8",
    )
    sys_class_block = root / "sys" / "class" / "block"
    sys_class_block.mkdir(parents=True)
    machine_id = root / "machine-id"
    machine_id.write_text("0123456789abcdef0123456789abcdef\n", encoding="utf-8")
    return proc, sys_class_block, machine_id


def _mock_static_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_bytes: int = 512 * GIB,
    available_inodes: int = 30_000_000,
) -> None:
    repository = Path(__file__).parents[1]
    fake_uv = repository / ".venv" / "bin" / "python"

    def run_text(arguments, *, cwd=None):
        current = repository if cwd is None else Path(cwd)
        if "rev-parse" in arguments and arguments[-1] == "HEAD":
            return COMMIT
        if "status" in arguments and "--porcelain=v2" in arguments:
            return f"# branch.oid {COMMIT}\n# branch.head (detached)"
        if "status" in arguments and "--porcelain" in arguments:
            return ""
        if arguments[-1] == "--version":
            return "uv 0.8.0"
        if tuple(arguments[1:3]) == ("sync", "--frozen"):
            return ""
        if arguments[-2:] == (
            "-c",
            "import platform; print(platform.python_version())",
        ):
            return "3.11.15"
        if arguments[-2:] == ("-c", "import sys; print(sys.prefix)"):
            return str(current / ".venv")
        raise AssertionError(arguments)

    monkeypatch.setattr(qualification, "_run_text", run_text)
    monkeypatch.setattr(qualification.shutil, "which", lambda _: str(fake_uv))
    monkeypatch.setattr(
        qualification,
        "_dependency_environment_hash",
        lambda: "9" * 64,
    )
    monkeypatch.setattr(
        qualification.os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=available_bytes,
            f_frsize=1,
            f_favail=available_inodes,
        ),
    )
    monkeypatch.setattr(
        qualification.os,
        "access",
        lambda *_args, **_kwargs: True,
    )


def _inspect(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    physical_gib: int = 96,
    filesystem: str = "ext4",
) -> dict[str, object]:
    storage = tmp_path / "storage"
    storage.mkdir(parents=True)
    proc, sys_class_block, machine_id = _fake_host_files(
        tmp_path,
        storage,
        physical_gib=physical_gib,
        filesystem=filesystem,
        device=("/dev/nvme0n1p1" if filesystem != "nfs" else "server:/volume"),
    )
    _mock_static_environment(monkeypatch)
    return inspect_host(
        repo_root=repo_root,
        execution_commit=COMMIT,
        storage_root=storage,
        additional_storage_bytes=GIB,
        additional_inodes=10_000,
        proc_root=proc,
        sys_class_block=sys_class_block,
        machine_id_path=machine_id,
    )


def _benchmark_result(stage: str) -> dict[str, object]:
    if stage == "e192":
        replicas = 192
        observations = 958_464
        dataset_hash = V2_E192_DATASET_HASH
        rss = 3500.0
    else:
        replicas = 768
        observations = 3_833_856
        dataset_hash = V2_E768_DATASET_HASH
        rss = 14_000.0
    return {
        "algorithm_replicas": 8,
        "checkpoint_count": 13,
        "checkpoint_zero_metric_count": 29,
        "dataset_elapsed_seconds": 1.0,
        "positive_checkpoint_metric_count": 31,
        "environment_replicas": replicas,
        "observation_count": observations,
        "pooled_checkpoint_count": 624,
        "dataset_hash": dataset_hash,
        "maximum_rss_mib": rss,
        "metric_count": 29,
        "pool_elapsed_seconds": 1.0,
        "rss_before_mib": 100.0,
        "rss_increment_upper_bound_mib": rss - 100.0,
        "total_elapsed_seconds": 2.0,
    }


def _write_fake_process_output(process, stdout, stderr):
    result = _benchmark_result("e192")
    result.update(
        {
            "dataset_elapsed_seconds": 0.001,
            "pool_elapsed_seconds": 0.0,
            "total_elapsed_seconds": 0.001,
        }
    )
    stdout.write(json.dumps(result))
    stdout.flush()
    stderr.flush()
    return process


def _capacity_record(
    stage: str,
    host: dict[str, object],
    *,
    passed: bool = True,
    recorded_at: str | None = None,
) -> dict[str, object]:
    snapshot = {
        "commit": COMMIT,
        "clean": True,
        "status": "",
        "uv_lock_sha256": host["execution"]["uv_lock_sha256"],
        "execution_python_sha256": host["runtime"]["execution_python_sha256"],
    }
    result = _benchmark_result(stage)
    faults = 0 if passed else 1
    decision = qualification._capacity_decision(
        stage,
        result=result,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=faults,
        swapout_delta_pages=0,
        checkout_unchanged=True,
    )
    execution = qualification._capacity_execution_identity(host)
    timestamp = qualification._utc_now() if recorded_at is None else recorded_at
    return qualification._signed(
        {
            "schema_version": 1,
            "record_type": "symbolic-v2-capacity-qualification",
            "recorded_at": timestamp,
            "stage": stage,
            "execution": execution,
            "host_identity": host["host_identity"],
            "host_identity_hash": host["host_identity_hash"],
            "tool_identity_hash": host["tool_identity_hash"],
            "static_record_hash": host["record_hash"],
            "command": qualification._command(
                qualification._capacity_arguments(
                    stage,
                    uv_path=execution["uv_path"],
                    execution_python=execution["execution_python_path"],
                )
            ),
            "started_at": timestamp,
            "elapsed_seconds": 2.0,
            "timeout_seconds": qualification.DEFAULT_CAPACITY_TIMEOUT_SECONDS,
            "exit_code": 0,
            "timed_out": False,
            "output_limit_exceeded": False,
            "benchmark_result": result,
            "prelaunch_checkout": snapshot,
            "postrun_checkout": dict(snapshot),
            "host_memory": {
                "before_available_memory_bytes": 80 * GIB,
                "after_available_memory_bytes": 79 * GIB,
                "minimum_available_memory_bytes": 79 * GIB,
            },
            "host_swap": {
                "page_size_bytes": 4096,
                "pswpin_delta_pages": 0,
                "pswpin_delta_bytes": 0,
                "pswpout_delta_pages": 0,
                "pswpout_delta_bytes": 0,
            },
            "process_major_faults": faults,
            "stderr": "",
            "decision": decision,
            "scientific_boundary": {
                "synthetic_dataset_only": True,
                "creates_study_artifacts": False,
                "executes_registered_study": False,
                "scientific_use": "prohibited",
            },
        }
    )


def _probe_execution_result(host: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_root": host["probe_storage"]["artifact_root"],
        "config_hash": V2_PROBE_CONFIG_HASH,
        "phase": "calibration",
        "run_count": 48,
    }


def _probe_benchmark_result(host: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_root": host["probe_storage"]["artifact_root"],
        "budget_multiplier": 2.0,
        "dataset_hash": V2_PROBE_DATASET_HASH,
        "fixed_frontier_cache_copy_seconds": 0.0,
        "fixed_frontier_raw_hash_seconds": 0.0,
        "fixed_frontier_validation_seconds": 0.0,
        "fixed_report_ingestion_seconds": 0.0,
        "frontier_tree_count": 4,
        "inventory_elapsed_seconds": 10.0,
        "load_elapsed_seconds": 3.437488,
        "maximum_rss_mib": 100.0,
        "modeled_probe_report_seconds": 23.437488,
        "marginal_run_load_seconds_per_run": 0.488281,
        "marginal_run_raw_hash_seconds_per_run": 0.0,
        "marginal_run_validation_seconds_per_run": 0.0,
        "observation_count": 624,
        "observed_probe_report_seconds": 23.437488,
        "operational_budget_hours": 80.0,
        "projected_report_ingestion_hours": 40.0,
        "projected_runs_per_root": 294_912,
        "raw_run_byte_size": 1,
        "raw_run_file_count": 1_392,
        "residual_seconds_per_run": 0.0,
        "rss_before_mib": 50.0,
        "rss_increment_upper_bound_mib": 50.0,
        "run_count": 48,
    }


def _bound_probe_records(
    host: dict[str, object],
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    artifact_root.mkdir(parents=True)
    (artifact_root / "synthetic-artifact.json").write_text(
        '{"synthetic":true}\n',
        encoding="utf-8",
    )
    descriptor = os.open(
        artifact_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        execution = qualification._bind_probe_result(
            kind="execution",
            static_record=host,
            result=_probe_execution_result(host),
            artifact_directory_fd=descriptor,
            executed_command=qualification._command(
                qualification._secured_probe_arguments(
                    "execution",
                    uv_path=host["runtime"]["uv_path"],
                    execution_python=host["runtime"]["execution_python_path"],
                    artifact_descriptor=descriptor,
                    repository=Path(host["execution"]["repository"]),
                )
            ),
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )
        benchmark = qualification._bind_probe_result(
            kind="benchmark",
            static_record=host,
            result=_probe_benchmark_result(host),
            probe_execution_record=execution,
            artifact_directory_fd=descriptor,
            executed_command=qualification._command(
                qualification._secured_probe_arguments(
                    "benchmark",
                    uv_path=host["runtime"]["uv_path"],
                    execution_python=host["runtime"]["execution_python_path"],
                    artifact_descriptor=descriptor,
                    repository=Path(host["execution"]["repository"]),
                )
            ),
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )
    finally:
        os.close(descriptor)
    return execution, benchmark


def _resign(record: dict[str, object]) -> dict[str, object]:
    changed = copy.deepcopy(record)
    changed.pop("record_hash", None)
    return qualification._signed(changed)


def test_exact_plan_is_print_only_and_excludes_registered_runner(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    plan = build_exact_plan(
        repo_root=repo_root,
        execution_commit=COMMIT,
        probe_root=tmp_path / "probe",
    )

    assert plan["execution_mode"] == "print-only"
    assert "--environment-replicas 768" in plan["capacity"]["e768"]
    assert "--positive-checkpoint-metrics 31" in plan["capacity"]["e192"]
    assert "scripts.run_ingestion_probe" in plan["probe"]["underlying_execute"]
    assert (
        "--projected-runs 294912 --budget-multiplier 2"
        in plan["probe"]["underlying_benchmark"]
    )
    assert "run-probe" in plan["probe"]["qualifying_execute_template"]
    assert "infinite-rulebook run" not in json.dumps(plan)
    with pytest.raises(ValueError, match="not approved"):
        build_exact_plan(
            repo_root=repo_root,
            execution_commit="a" * 40,
            probe_root=tmp_path / "probe",
        )
    with pytest.raises(ValueError, match="outside the execution repository"):
        build_exact_plan(
            repo_root=repo_root,
            execution_commit=COMMIT,
            probe_root=repo_root / "ignored-probe",
        )
    repository_link = tmp_path / "repository-link"
    repository_link.symlink_to(repo_root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the execution repository"):
        build_exact_plan(
            repo_root=repo_root,
            execution_commit=COMMIT,
            probe_root=repository_link / "ignored-probe",
        )


def test_static_inspection_rejects_storage_inside_execution_repository() -> None:
    repo_root = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="outside the execution repository"):
        inspect_host(
            repo_root=repo_root,
            execution_commit=COMMIT,
            storage_root=repo_root / "ignored-storage",
            additional_storage_bytes=GIB,
            additional_inodes=10_000,
        )


def test_static_inspection_passes_only_a_pinned_sized_local_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).parents[1]
    record = _inspect(repo_root, tmp_path, monkeypatch)

    assert verify_record(record) == record
    assert record["decision"]["static_prerequisites_passed"]
    assert not record["decision"]["registered_execution_authorized"]
    assert record["storage"]["local_filesystem"]
    assert record["storage"]["solid_state"]
    assert not record["scientific_boundary"]["executes_registered_study"]

    tampered = dict(record)
    tampered["recorded_at"] = "changed"
    with pytest.raises(ValueError, match="does not match"):
        verify_record(tampered)


@pytest.mark.parametrize(
    ("physical_gib", "filesystem", "failed_requirement"),
    ((32, "ext4", "physical-memory"), (96, "nfs", "local-filesystem")),
)
def test_static_inspection_fails_unsafe_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_gib: int,
    filesystem: str,
    failed_requirement: str,
) -> None:
    record = _inspect(
        Path(__file__).parents[1],
        tmp_path,
        monkeypatch,
        physical_gib=physical_gib,
        filesystem=filesystem,
    )

    decisions = {item["name"]: item["passed"] for item in record["requirements"]}
    assert not decisions[failed_requirement]
    assert not record["decision"]["static_prerequisites_passed"]


def test_capacity_runner_refuses_e768_without_explicit_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not launch"),
    )

    with pytest.raises(ValueError, match="explicit"):
        run_capacity_benchmark(stage="e768", static_record=host)


def test_capacity_runner_never_launches_e768_on_an_undersized_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(
        Path(__file__).parents[1],
        tmp_path,
        monkeypatch,
        physical_gib=32,
    )
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not launch"),
    )

    with pytest.raises(ValueError, match="static host prerequisites"):
        run_capacity_benchmark(
            stage="e768",
            static_record=host,
            acknowledge_e768=True,
        )


def test_capacity_runner_captures_synthetic_evidence_with_mocked_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification,
        "_current_host_resources_match",
        lambda *_args, **_kwargs: True,
    )
    readings = iter(
        (
            {"MemAvailable": 80 * GIB},
            {"pswpin": 10, "pswpout": 20},
            {"MemAvailable": 79 * GIB},
            {"pswpin": 10, "pswpout": 20},
        )
    )
    monkeypatch.setattr(qualification, "_read_key_values", lambda _path: next(readings))
    usage = iter((SimpleNamespace(ru_majflt=5), SimpleNamespace(ru_majflt=5)))
    monkeypatch.setattr(qualification.resource, "getrusage", lambda _: next(usage))

    class Process:
        returncode = 0
        pid = 123

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout=None) -> int:
            del timeout
            return 0

    captured: dict[str, object] = {}

    def popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["environment"] = kwargs["env"]
        return _write_fake_process_output(
            Process(),
            kwargs["stdout"],
            kwargs["stderr"],
        )

    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/untrusted")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted.so")
    monkeypatch.setattr(
        qualification.os,
        "killpg",
        lambda *_: pytest.fail("completed process group must not be signaled"),
    )
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        popen,
    )
    record = run_capacity_benchmark(
        stage="e192",
        static_record=host,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )

    assert verify_record(record) == record
    assert record["decision"]["passed"]
    assert record["process_major_faults"] == 0
    assert record["host_swap"]["pswpout_delta_pages"] == 0
    assert record["benchmark_result"]["dataset_hash"] == V2_E192_DATASET_HASH
    assert captured["arguments"][0] == host["runtime"]["uv_path"]
    assert host["runtime"]["execution_python_path"] in captured["arguments"]
    assert not any(
        name.startswith(("PYTHON", "UV_")) or name in {"LD_PRELOAD", "LD_LIBRARY_PATH"}
        for name in captured["environment"]
    )


@pytest.mark.parametrize("maximum_rss_mib", (0.0, -1.0))
def test_capacity_decision_rejects_nonpositive_rss(
    maximum_rss_mib: float,
) -> None:
    result = _benchmark_result("e192")
    result["maximum_rss_mib"] = maximum_rss_mib

    decision = qualification._capacity_decision(
        "e192",
        result=result,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )

    assert not decision["equal_physical_reserve_passed"]
    assert not decision["passed"]


def test_capacity_decision_rejects_integral_float_shape_counts() -> None:
    result = _benchmark_result("e192")
    result["algorithm_replicas"] = 8.0

    decision = qualification._capacity_decision(
        "e192",
        result=result,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )

    assert not decision["exact_shape_passed"]
    assert not decision["passed"]


def test_assessment_requires_every_gate_and_never_authorizes_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    probe_execution, probe_benchmark = _bound_probe_records(host, tmp_path)
    record = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=probe_execution,
        probe_benchmark_result=probe_benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )

    assert record["decision"]["qualification_passed"]
    assert verify_record(record) == record
    assert (
        verify_assessment_bundle(
            assessment_record=record,
            static_record=host,
            e192_record=e192,
            e768_record=e768,
            probe_execution_record=probe_execution,
            probe_benchmark_record=probe_benchmark,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )
        == record
    )
    assert not record["decision"]["registered_execution_authorized_by_this_record"]
    assert not record["scientific_boundary"]["inspects_probe_metrics"]

    failed = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=probe_execution,
        probe_benchmark_result=probe_benchmark,
        available_window_hours=90.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not failed["decision"]["qualification_passed"]
    assert not failed["checks"]["time_window_passed"]


def test_assessment_bundle_rejects_rehashed_checks_for_different_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    failing_e192 = _capacity_record("e192", host, passed=False)
    forged = copy.deepcopy(assessment)
    forged["inputs"]["e192_record_hash"] = failing_e192["record_hash"]
    forged = _resign(forged)

    assert verify_record(forged) == forged
    with pytest.raises(ValueError, match="does not match its bound"):
        verify_assessment_bundle(
            assessment_record=forged,
            static_record=host,
            e192_record=failing_e192,
            e768_record=e768,
            probe_execution_record=execution,
            probe_benchmark_record=benchmark,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )

    inconsistent_runtime = copy.deepcopy(assessment)
    inconsistent_runtime["runtime"]["required_window_hours"] = 91.0
    with pytest.raises(ValueError, match="required window"):
        verify_record(_resign(inconsistent_runtime))


def test_records_are_write_once_and_owner_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    payload = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)

    write_record(path, payload)
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    write_record(path, payload)
    changed = copy.deepcopy(payload)
    changed["recorded_at"] = qualification._utc_now()
    changed = qualification._signed(changed)
    with pytest.raises(ValueError, match="overwrite"):
        write_record(path, changed)


def test_verify_record_rejects_bool_schema_and_skeletal_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    bool_schema = _resign({**host, "schema_version": True})
    with pytest.raises(ValueError, match="unsupported"):
        verify_record(bool_schema)

    skeletal = qualification._signed(
        {
            "schema_version": 1,
            "record_type": "symbolic-v2-capacity-qualification",
            "stage": "e192",
            "decision": {"passed": True},
        }
    )
    with pytest.raises(ValueError, match="fields do not match schema"):
        verify_record(skeletal)

    missing_requirements = copy.deepcopy(host)
    missing_requirements.pop("requirements")
    with pytest.raises(ValueError, match=r"missing=.*requirements"):
        verify_record(_resign(missing_requirements))


def test_verify_record_rejects_unapproved_sha_and_fabricated_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    capacity = _capacity_record("e192", host)

    unapproved = copy.deepcopy(capacity)
    unapproved["execution"]["commit"] = "a" * 40
    with pytest.raises(ValueError, match="not approved"):
        verify_record(_resign(unapproved))

    fabricated = copy.deepcopy(capacity)
    fabricated["decision"]["passed"] = False
    with pytest.raises(ValueError, match="does not match recomputed"):
        verify_record(_resign(fabricated))

    missing_benchmark = copy.deepcopy(capacity)
    missing_benchmark.pop("benchmark_result")
    with pytest.raises(ValueError, match=r"missing=.*benchmark_result"):
        verify_record(_resign(missing_benchmark))


def test_assessment_rejects_plain_probe_dicts_and_other_static_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)

    with pytest.raises(ValueError, match="unsupported qualification record schema"):
        assess_qualification(
            static_record=host,
            e192_record=e192,
            e768_record=e768,
            probe_execution_result=_probe_execution_result(host),
            probe_benchmark_result=_probe_benchmark_result(host),
            available_window_hours=96.0,
            recovery_margin_hours=12.0,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )

    other_static = "f" * 64
    mixed = copy.deepcopy(e768)
    mixed["static_record_hash"] = other_static
    mixed = _resign(mixed)
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=mixed,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["capacity_static_binding_passed"]
    assert not assessment["decision"]["qualification_passed"]


def test_assessment_rejects_cross_host_and_out_of_order_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)

    mixed_host = copy.deepcopy(benchmark)
    mixed_host["host_identity"]["boot_id_hash"] = "a" * 64
    mixed_host["host_identity_hash"] = "b" * 64
    mixed_host = _resign(mixed_host)
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=mixed_host,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["host_binding_passed"]
    assert not assessment["decision"]["qualification_passed"]

    mixed_tool = copy.deepcopy(benchmark)
    mixed_tool["tool_identity_hash"] = "c" * 64
    mixed_tool = _resign(mixed_tool)
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=mixed_tool,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["tool_binding_passed"]
    assert not assessment["decision"]["qualification_passed"]

    out_of_order = copy.deepcopy(e768)
    out_of_order["started_at"] = host["recorded_at"]
    out_of_order = _resign(out_of_order)
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=out_of_order,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["input_freshness_order_passed"]
    assert not assessment["decision"]["qualification_passed"]


def test_assessment_rejects_stale_coherently_rebound_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    stale = copy.deepcopy(host)
    stale["recorded_at"] = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    stale = _resign(stale)
    rebound = []
    for record in (e192, e768, execution, benchmark):
        changed = copy.deepcopy(record)
        changed["static_record_hash"] = stale["record_hash"]
        rebound.append(_resign(changed))

    assessment = assess_qualification(
        static_record=stale,
        e192_record=rebound[0],
        e768_record=rebound[1],
        probe_execution_result=rebound[2],
        probe_benchmark_result=rebound[3],
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert assessment["checks"]["capacity_static_binding_passed"]
    assert not assessment["checks"]["input_freshness_order_passed"]
    assert not assessment["decision"]["qualification_passed"]


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("sample_interval_seconds", True),
        ("sample_interval_seconds", 1.1),
        ("sample_interval_seconds", math.nan),
        ("sample_interval_seconds", math.inf),
        ("timeout_seconds", True),
        ("timeout_seconds", None),
        ("timeout_seconds", math.nan),
        ("timeout_seconds", math.inf),
    ),
)
def test_capacity_runner_rejects_invalid_controls_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: object,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not launch"),
    )

    with pytest.raises(ValueError, match="must be"):
        run_capacity_benchmark(
            stage="e192",
            static_record=host,
            **{keyword: value},
        )


def test_capacity_runner_terminates_and_reaps_on_sampling_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification,
        "_current_host_resources_match",
        lambda *_args, **_kwargs: True,
    )
    calls = 0

    def read_values(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"MemAvailable": 80 * GIB}
        if calls == 2:
            return {"pswpin": 10, "pswpout": 20}
        raise KeyboardInterrupt

    monkeypatch.setattr(qualification, "_read_key_values", read_values)
    monkeypatch.setattr(
        qualification.resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_majflt=0),
    )

    class Process:
        pid = 321
        returncode = None
        alive = True
        wait_count = 0

        @classmethod
        def poll(cls):
            return None if cls.alive else cls.returncode

        @classmethod
        def wait(cls, timeout=None):
            assert timeout == 10
            cls.wait_count += 1
            if cls.alive:
                raise subprocess.TimeoutExpired("mock", timeout)
            return cls.returncode

    signals: list[int] = []

    def killpg(_pid, sent_signal):
        signals.append(sent_signal)
        if sent_signal == signal.SIGKILL:
            Process.alive = False
            Process.returncode = -sent_signal

    monkeypatch.setattr(qualification.os, "killpg", killpg)
    monkeypatch.setattr(qualification.subprocess, "Popen", lambda *_a, **_k: Process())

    with pytest.raises(KeyboardInterrupt):
        run_capacity_benchmark(
            stage="e192",
            static_record=host,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert Process.wait_count == 2


@pytest.mark.parametrize(
    "postrun_change",
    (
        {"clean": False, "status": "1 .M source.py"},
        {"commit": "d" * 40},
        {"uv_lock_sha256": "e" * 64},
        {"execution_python_sha256": "f" * 64},
    ),
)
def test_capacity_record_fails_when_checkout_changes_after_completed_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postrun_change: dict[str, object],
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification,
        "_current_host_resources_match",
        lambda *_args, **_kwargs: True,
    )
    clean_snapshot = {
        "commit": COMMIT,
        "clean": True,
        "status": "",
        "uv_lock_sha256": host["execution"]["uv_lock_sha256"],
        "execution_python_sha256": host["runtime"]["execution_python_sha256"],
    }
    changed_snapshot = {**clean_snapshot, **postrun_change}
    snapshots = iter((clean_snapshot, changed_snapshot))
    monkeypatch.setattr(
        qualification,
        "_repository_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    readings = iter(
        (
            {"MemAvailable": 80 * GIB},
            {"pswpin": 10, "pswpout": 20},
            {"MemAvailable": 79 * GIB},
            {"pswpin": 10, "pswpout": 20},
        )
    )
    monkeypatch.setattr(qualification, "_read_key_values", lambda _path: next(readings))
    usage = iter((SimpleNamespace(ru_majflt=0), SimpleNamespace(ru_majflt=0)))
    monkeypatch.setattr(qualification.resource, "getrusage", lambda _: next(usage))

    class Process:
        pid = 123
        returncode = 0

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait(timeout=None):
            assert timeout == 10
            return 0

    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *_args, **kwargs: _write_fake_process_output(
            Process(),
            kwargs["stdout"],
            kwargs["stderr"],
        ),
    )
    record = run_capacity_benchmark(
        stage="e192",
        static_record=host,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )

    assert not record["decision"]["checkout_unchanged_passed"]
    assert not record["decision"]["passed"]
    assert verify_record(record) == record


@pytest.mark.parametrize("value", ("nan", "inf", "-inf"))
def test_cli_float_parsers_reject_nonfinite_values(value: str) -> None:
    with pytest.raises(Exception, match="finite"):
        qualification_cli._positive_float(value)
    with pytest.raises(Exception, match="finite"):
        qualification_cli._nonnegative_float(value)


def test_probe_binding_rejects_symlinked_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    real_root = tmp_path / "real-probe"
    real_root.mkdir()
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked component"):
        bind_probe_result(
            kind="execution",
            static_record=host,
            result=_probe_execution_result(host),
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )


def test_probe_binding_rejects_same_inode_content_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    artifact_root.mkdir(parents=True)
    artifact = artifact_root / "synthetic-artifact.json"
    artifact.write_text('{"version":1}\n', encoding="utf-8")
    descriptor = os.open(
        artifact_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        execution = qualification._bind_probe_result(
            kind="execution",
            static_record=host,
            result=_probe_execution_result(host),
            artifact_directory_fd=descriptor,
            executed_command=qualification._command(
                qualification._secured_probe_arguments(
                    "execution",
                    uv_path=host["runtime"]["uv_path"],
                    execution_python=host["runtime"]["execution_python_path"],
                    artifact_descriptor=descriptor,
                    repository=Path(host["execution"]["repository"]),
                )
            ),
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )

        artifact.write_text('{"version":2}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="content changed"):
            qualification._bind_probe_result(
                kind="benchmark",
                static_record=host,
                result=_probe_benchmark_result(host),
                probe_execution_record=execution,
                artifact_directory_fd=descriptor,
                executed_command=qualification._command(
                    qualification._secured_probe_arguments(
                        "benchmark",
                        uv_path=host["runtime"]["uv_path"],
                        execution_python=host["runtime"]["execution_python_path"],
                        artifact_descriptor=descriptor,
                        repository=Path(host["execution"]["repository"]),
                    )
                ),
                proc_root=tmp_path / "proc",
                machine_id_path=tmp_path / "machine-id",
            )
    finally:
        os.close(descriptor)


def test_probe_runner_anchors_execution_and_benchmark_to_open_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    anchored_paths: list[str] = []

    class Process:
        pid = 123
        returncode = 0

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait(timeout=None):
            assert timeout == 10
            return 0

    def popen(arguments, **kwargs):
        config_index = next(
            index
            for index, argument in enumerate(arguments)
            if argument.endswith("configs/symbolic-artifact-ingestion-probe-v2.json")
        )
        reported_root = arguments[config_index + 1]
        assert reported_root == "."
        assert kwargs["pass_fds"]
        descriptor = kwargs["pass_fds"][0]
        anchored_root = f"/proc/self/fd/{descriptor}"
        anchored_paths.append(anchored_root)
        root = Path(anchored_root)
        assert root.is_dir()
        if qualification._PRECREATED_PROBE_ROOT_BOOTSTRAP in arguments:
            (root / "synthetic-artifact.json").write_text(
                '{"synthetic":true}\n',
                encoding="utf-8",
            )
            result = {
                **_probe_execution_result(host),
                "artifact_root": reported_root,
            }
        else:
            assert qualification._ANCHORED_PROBE_BENCHMARK_BOOTSTRAP in arguments
            result = {
                **_probe_benchmark_result(host),
                "artifact_root": reported_root,
            }
        kwargs["stdout"].write(json.dumps(result))
        kwargs["stdout"].flush()
        kwargs["stderr"].flush()
        return Process()

    monkeypatch.setattr(qualification.subprocess, "Popen", popen)
    execution = run_probe_step(
        kind="execution",
        static_record=host,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    benchmark = run_probe_step(
        kind="benchmark",
        static_record=host,
        probe_execution_record=execution,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )

    assert execution["decision"]["shape_passed"]
    assert benchmark["decision"]["shape_passed"]
    assert execution["secure_artifact_access"]
    assert benchmark["artifact_manifest"] == execution["artifact_manifest"]
    assert "scripts.run_ingestion_probe" in execution["planned_command"]
    assert "runpy.run_module" in execution["executed_command"]
    assert len(anchored_paths) == 2


def test_probe_descriptor_cwd_is_compatible_with_artifact_nofollow_walker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe-root"
    root.mkdir()
    previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fchdir(descriptor)
        with artifact_root_lock(Path("."), create=False) as opened:
            observed = os.fstat(opened)
            expected = os.fstat(descriptor)
            assert (observed.st_dev, observed.st_ino) == (
                expected.st_dev,
                expected.st_ino,
            )
    finally:
        os.fchdir(previous)
        os.close(previous)
        os.close(descriptor)


def test_precreated_probe_bootstrap_anchors_cwd_and_masks_only_root_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "probe-root"
    root.mkdir()
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    original_argv = sys.argv

    def fake_run_module(name, *, run_name):
        assert name == "scripts.run_ingestion_probe"
        assert run_name == "__main__"
        assert sys.argv[1:] == [
            "/execution/config.json",
            "/execution/probe.json",
            ".",
            "--workers",
            "4",
        ]
        observed = os.stat(".")
        expected = os.fstat(descriptor)
        assert (observed.st_dev, observed.st_ino) == (
            expected.st_dev,
            expected.st_ino,
        )
        assert not os.path.lexists(".")
        assert os.path.lexists(".")
        raise SystemExit(0)

    monkeypatch.setattr(runpy, "run_module", fake_run_module)
    sys.argv = [
        "-c",
        str(descriptor),
        "/execution",
        "/execution/config.json",
        "/execution/probe.json",
        ".",
        "--workers",
        "4",
    ]
    try:
        with pytest.raises(SystemExit) as stopped:
            exec(qualification._PRECREATED_PROBE_ROOT_BOOTSTRAP, {})
        assert stopped.value.code == 0
    finally:
        sys.argv = original_argv
        os.fchdir(previous)
        os.close(previous)
        os.close(descriptor)


def test_manual_probe_binding_is_hash_sealed_but_nonqualifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    artifact_root.mkdir(parents=True)
    (artifact_root / "synthetic-artifact.json").write_text(
        '{"synthetic":true}\n',
        encoding="utf-8",
    )

    record = bind_probe_result(
        kind="execution",
        static_record=host,
        result=_probe_execution_result(host),
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )

    assert verify_record(record) == record
    assert not record["secure_artifact_access"]
    assert not record["decision"]["shape_passed"]
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "manual evidence must be rejected before benchmark launch"
        ),
    )
    with pytest.raises(ValueError, match="not valid for benchmark launch"):
        run_probe_step(
            kind="benchmark",
            static_record=host,
            probe_execution_record=record,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )


def test_probe_binding_rejects_symlink_inside_artifact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    artifact_root.mkdir(parents=True)
    (artifact_root / "linked-artifact").symlink_to(tmp_path / "outside")

    with pytest.raises(ValueError, match="non-regular file"):
        bind_probe_result(
            kind="execution",
            static_record=host,
            result=_probe_execution_result(host),
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )


def test_record_outputs_reject_protected_and_symlinked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    repository = Path(host["execution"]["repository"])
    with pytest.raises(ValueError, match="outside protected root"):
        validate_output_path(
            repository / "qualification.json",
            forbidden_roots=(repository,),
        )
    for protected in (
        Path(host["storage"]["storage_root"]),
        Path(host["probe_storage"]["artifact_root"]),
    ):
        with pytest.raises(ValueError, match="outside protected root"):
            validate_output_path(
                protected / "qualification.json",
                forbidden_roots=(protected,),
            )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        write_record(linked_parent / "record.json", host)

    real_target = tmp_path / "real-record.json"
    real_target.write_text("placeholder", encoding="utf-8")
    linked_target = tmp_path / "linked-record.json"
    linked_target.symlink_to(real_target)
    with pytest.raises(ValueError, match="contains a symlink"):
        write_record(linked_target, host)


def test_record_write_validates_first_and_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_path = tmp_path / "invalid" / "record.json"
    with pytest.raises(ValueError, match="unsupported qualification record type"):
        write_record(
            invalid_path,
            qualification._signed({"schema_version": 1, "record_type": "fabricated"}),
        )
    assert not invalid_path.parent.exists()

    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    failed_path = tmp_path / "failed" / "record.json"
    monkeypatch.setattr(
        qualification.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mock link failure")),
    )
    with pytest.raises(OSError, match="mock link failure"):
        write_record(failed_path, host)
    assert not failed_path.exists()
    assert not tuple(failed_path.parent.iterdir())


def test_assessment_rejects_coherently_forged_capacity_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)

    forged = copy.deepcopy(e192)
    forged["execution"].update(
        {
            "repository": "/forged/execution",
            "git_directory": "/forged/execution.git",
            "git_path": "/forged/bin/git",
            "git_sha256": "1" * 64,
            "uv_lock_sha256": "2" * 64,
            "uv_path": "/forged/bin/uv",
            "uv_sha256": "3" * 64,
            "execution_python_path": "/forged/bin/python",
            "execution_python_sha256": "4" * 64,
        }
    )
    for snapshot_name in ("prelaunch_checkout", "postrun_checkout"):
        forged[snapshot_name]["uv_lock_sha256"] = "2" * 64
        forged[snapshot_name]["execution_python_sha256"] = "4" * 64
    forged["command"] = qualification._command(
        qualification._capacity_arguments(
            "e192",
            uv_path="/forged/bin/uv",
            execution_python="/forged/bin/python",
        )
    )
    forged = _resign(forged)

    assert verify_record(forged) == forged
    assessment = assess_qualification(
        static_record=host,
        e192_record=forged,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["execution_binding_passed"]
    assert not assessment["decision"]["qualification_passed"]


def test_assessment_rejects_coherently_forged_probe_static_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    forged_root = str(tmp_path / "forged-probe")

    forged_execution = copy.deepcopy(execution)
    forged_execution["expected_artifact_root"] = forged_root
    forged_execution["artifact_storage_identity"].update(
        {
            "canonical_path": forged_root,
            "device": host["storage"]["directory_device_id"] + 1,
        }
    )
    forged_execution["probe_config_hash"] = "5" * 64
    forged_execution["planned_command"] = "forged execution plan"
    forged_execution["executed_command"] = "forged execution argv"
    forged_execution["result"]["artifact_root"] = forged_root
    forged_execution["source_result_hash"] = qualification.scientific_hash(
        forged_execution["result"],
        domain="operations.v2-probe-execution-result.v1",
    )
    forged_execution = _resign(forged_execution)

    forged_benchmark = copy.deepcopy(benchmark)
    forged_benchmark["expected_artifact_root"] = forged_root
    forged_benchmark["artifact_storage_identity"].update(
        {
            "canonical_path": forged_root,
            "device": host["storage"]["directory_device_id"] + 1,
        }
    )
    forged_benchmark["probe_config_hash"] = "5" * 64
    forged_benchmark["planned_command"] = "forged benchmark plan"
    forged_benchmark["executed_command"] = "forged benchmark argv"
    forged_benchmark["probe_execution_record_hash"] = forged_execution["record_hash"]
    forged_benchmark["result"]["artifact_root"] = forged_root
    forged_benchmark["source_result_hash"] = qualification.scientific_hash(
        forged_benchmark["result"],
        domain="operations.v2-probe-benchmark-result.v1",
    )
    forged_benchmark = _resign(forged_benchmark)

    assert verify_record(forged_execution) == forged_execution
    assert verify_record(forged_benchmark) == forged_benchmark
    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=forged_execution,
        probe_benchmark_result=forged_benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["probe_static_binding_passed"]
    assert not assessment["checks"]["probe_artifacts_current"]
    assert not assessment["decision"]["qualification_passed"]


def test_probe_projection_is_recomputed_instead_of_trusting_claimed_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    _, benchmark = _bound_probe_records(host, tmp_path)
    forged = copy.deepcopy(benchmark)
    forged["result"]["projected_report_ingestion_hours"] = 0.001
    forged["result"]["operational_budget_hours"] = 0.002
    forged["source_result_hash"] = qualification.scientific_hash(
        forged["result"],
        domain="operations.v2-probe-benchmark-result.v1",
    )

    with pytest.raises(ValueError, match="decision does not match"):
        verify_record(_resign(forged))

    zero_pre_rss = copy.deepcopy(benchmark)
    zero_pre_rss["result"]["rss_before_mib"] = 0.0
    zero_pre_rss["result"]["rss_increment_upper_bound_mib"] = zero_pre_rss["result"][
        "maximum_rss_mib"
    ]
    zero_pre_rss["source_result_hash"] = qualification.scientific_hash(
        zero_pre_rss["result"],
        domain="operations.v2-probe-benchmark-result.v1",
    )
    with pytest.raises(ValueError, match="decision does not match"):
        verify_record(_resign(zero_pre_rss))


def test_git_identity_commands_ignore_inherited_repository_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-tree"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "attacker-config"))
    monkeypatch.setenv("PATH", str(tmp_path / "attacker-bin"))
    assert qualification._sanitized_environment() == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }

    repository = tmp_path / "qualified"
    repository.mkdir()
    lock = repository / "uv.lock"
    python = repository / "python"
    lock.write_text("lock", encoding="utf-8")
    python.write_text("python", encoding="utf-8")
    git_directory = tmp_path / "qualified.git"
    git_directory.mkdir()
    captured: list[tuple[str, ...]] = []

    def run_text(arguments, *, cwd=None):
        assert cwd == repository
        captured.append(tuple(arguments))
        return f"# branch.oid {COMMIT}\n# branch.head (detached)"

    monkeypatch.setattr(qualification, "_run_text", run_text)
    qualification._repository_snapshot(
        repository,
        lock=lock,
        execution_python=python,
        git_path="/trusted/bin/git",
        git_directory=git_directory,
    )

    assert captured == [
        (
            "/trusted/bin/git",
            "--no-optional-locks",
            f"--git-dir={git_directory}",
            f"--work-tree={repository}",
            "status",
            "--porcelain=v2",
            "--branch",
            "--untracked-files=normal",
        )
    ]


def test_capacity_decision_rejects_metric_timing_and_memory_forgery() -> None:
    wrong_metrics = _benchmark_result("e192")
    wrong_metrics["metric_count"] = 999
    decision = qualification._capacity_decision(
        "e192",
        result=wrong_metrics,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )
    assert not decision["exact_shape_passed"]
    assert not decision["passed"]

    bad_memory = qualification._capacity_decision(
        "e192",
        result=_benchmark_result("e192"),
        before_available_memory_bytes=0,
        after_available_memory_bytes=0,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )
    assert not bad_memory["memory_measurements_consistent_passed"]
    assert not bad_memory["passed"]

    bad_timing_result = _benchmark_result("e192")
    bad_timing_result["total_elapsed_seconds"] = 1.0
    bad_timing = qualification._capacity_decision(
        "e192",
        result=bad_timing_result,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )
    assert not bad_timing["benchmark_metrics_consistent_passed"]
    assert not bad_timing["passed"]

    zero_pre_rss_result = _benchmark_result("e192")
    zero_pre_rss_result["rss_before_mib"] = 0.0
    zero_pre_rss_result["rss_increment_upper_bound_mib"] = zero_pre_rss_result[
        "maximum_rss_mib"
    ]
    zero_pre_rss = qualification._capacity_decision(
        "e192",
        result=zero_pre_rss_result,
        before_available_memory_bytes=80 * GIB,
        after_available_memory_bytes=79 * GIB,
        minimum_available_memory_bytes=79 * GIB,
        process_major_faults=0,
        swapout_delta_pages=0,
    )
    assert not zero_pre_rss["benchmark_metrics_consistent_passed"]
    assert not zero_pre_rss["passed"]


def test_capacity_record_rejects_impossible_wrapper_and_child_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    forged = _capacity_record("e192", host)
    forged["elapsed_seconds"] = 0.0
    forged["timeout_seconds"] = 0.001

    with pytest.raises(ValueError, match="wrapper elapsed time"):
        verify_record(_resign(forged))

    forged["elapsed_seconds"] = 2.0
    with pytest.raises(ValueError, match="credible timeout"):
        verify_record(_resign(forged))


def test_installed_dependency_hash_binds_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = tmp_path / "package.py"
    installed.write_text("VALUE = 1\n", encoding="utf-8")
    distribution = SimpleNamespace(
        metadata={"Name": "example"},
        version="1.0",
        files=(Path("package.py"),),
        locate_file=lambda declared: tmp_path / declared,
    )
    monkeypatch.setattr(
        qualification.importlib_metadata,
        "distributions",
        lambda: (distribution,),
    )

    before = qualification._dependency_environment_hash()
    installed.write_text("VALUE = 2\n", encoding="utf-8")
    after = qualification._dependency_environment_hash()
    assert before != after


def test_current_execution_python_requires_executable_bound_bytes(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o600)
    host = {
        "runtime": {
            "execution_python_path": str(python),
            "execution_python_sha256": qualification._sha256_file(python),
        }
    }
    assert not qualification._current_execution_python_matches_host(host)

    python.chmod(0o700)
    assert qualification._current_execution_python_matches_host(host)


def test_current_resource_revalidation_checks_storage_mount_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    proc = tmp_path / "proc"
    assert qualification._current_host_resources_match(host, proc_root=proc)

    wrong_inode = copy.deepcopy(host)
    wrong_inode["storage"]["directory_inode"] += 1
    assert not qualification._current_host_resources_match(
        wrong_inode,
        proc_root=proc,
    )

    mountinfo = proc / "self" / "mountinfo"
    original_mount = mountinfo.read_text(encoding="utf-8")
    mountinfo.write_text(
        original_mount.replace("/dev/nvme0n1p1", "/dev/nvme9n9p9"),
        encoding="utf-8",
    )
    assert not qualification._current_host_resources_match(host, proc_root=proc)
    mountinfo.write_text(original_mount, encoding="utf-8")

    original_statvfs = qualification.os.statvfs
    monkeypatch.setattr(
        qualification.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=1, f_frsize=1, f_favail=1),
    )
    assert not qualification._current_host_resources_match(host, proc_root=proc)
    monkeypatch.setattr(qualification.os, "statvfs", original_statvfs)

    meminfo = proc / "meminfo"
    original_memory = meminfo.read_text(encoding="utf-8")
    meminfo.write_text(
        original_memory.replace(
            f"MemTotal: {96 * GIB // 1024} kB",
            f"MemTotal: {32 * GIB // 1024} kB",
        ),
        encoding="utf-8",
    )
    assert not qualification._current_host_resources_match(host, proc_root=proc)


def test_assessment_wires_current_resource_revalidation_into_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    monkeypatch.setattr(
        qualification.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=1, f_frsize=1, f_favail=1),
    )

    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not assessment["checks"]["host_resources_revalidated"]
    assert not assessment["decision"]["qualification_passed"]


def test_assessment_timestamp_is_deterministic_fresh_and_not_redatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    arguments = {
        "static_record": host,
        "e192_record": e192,
        "e768_record": e768,
        "probe_execution_result": execution,
        "probe_benchmark_result": benchmark,
        "available_window_hours": 96.0,
        "recovery_margin_hours": 12.0,
        "proc_root": tmp_path / "proc",
        "machine_id_path": tmp_path / "machine-id",
    }
    assessment = assess_qualification(**arguments)
    repeated = assess_qualification(**arguments)
    assert repeated == assessment
    assert assessment["recorded_at"] == max(
        record["recorded_at"] for record in (host, e192, e768, execution, benchmark)
    )

    redated = copy.deepcopy(assessment)
    redated["recorded_at"] = (
        datetime.fromisoformat(assessment["recorded_at"]) + timedelta(seconds=1)
    ).isoformat()
    redated = _resign(redated)
    assert verify_record(redated) == redated
    with pytest.raises(ValueError, match="does not match its bound"):
        verify_assessment_bundle(
            assessment_record=redated,
            static_record=host,
            e192_record=e192,
            e768_record=e768,
            probe_execution_record=execution,
            probe_benchmark_record=benchmark,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )

    for timestamp in (
        datetime.now(UTC) - timedelta(hours=25),
        datetime.now(UTC) + timedelta(minutes=6),
    ):
        invalid = copy.deepcopy(assessment)
        invalid["recorded_at"] = timestamp.isoformat()
        with pytest.raises(ValueError, match="stale or future-dated"):
            verify_record(_resign(invalid))


def test_detailed_probe_manifest_is_validated_and_rechecked_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _inspect(Path(__file__).parents[1], tmp_path, monkeypatch)
    e192 = _capacity_record("e192", host)
    e768 = _capacity_record("e768", host)
    execution, benchmark = _bound_probe_records(host, tmp_path)
    manifest = execution["artifact_manifest"]
    assert manifest["entries"] == [
        {
            "kind": "file",
            "path": "synthetic-artifact.json",
            "sha256": qualification._sha256_file(
                Path(host["probe_storage"]["artifact_root"]) / "synthetic-artifact.json"
            ),
            "size_bytes": 19,
        }
    ]

    inconsistent = copy.deepcopy(execution)
    inconsistent["artifact_manifest"]["file_count"] = 2
    with pytest.raises(ValueError, match="aggregates do not match"):
        verify_record(_resign(inconsistent))

    assessment = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    artifact = Path(host["probe_storage"]["artifact_root"]) / "synthetic-artifact.json"
    artifact.write_text('{"synthetic":false}\n', encoding="utf-8")
    changed = assess_qualification(
        static_record=host,
        e192_record=e192,
        e768_record=e768,
        probe_execution_result=execution,
        probe_benchmark_result=benchmark,
        available_window_hours=96.0,
        recovery_margin_hours=12.0,
        proc_root=tmp_path / "proc",
        machine_id_path=tmp_path / "machine-id",
    )
    assert not changed["checks"]["probe_artifacts_current"]
    assert not changed["decision"]["qualification_passed"]
    with pytest.raises(ValueError, match="does not match its bound"):
        verify_assessment_bundle(
            assessment_record=assessment,
            static_record=host,
            e192_record=e192,
            e768_record=e768,
            probe_execution_record=execution,
            probe_benchmark_record=benchmark,
            proc_root=tmp_path / "proc",
            machine_id_path=tmp_path / "machine-id",
        )


def test_live_probe_verification_keeps_one_descriptor_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "storage"
    root = storage / "probe"
    root.mkdir(parents=True)
    (root / "original.json").write_text("{}", encoding="utf-8")
    displaced = storage / "displaced"
    original_manifest = qualification._artifact_manifest_from_descriptor

    def replace_root_after_open(descriptor):
        root.rename(displaced)
        root.mkdir()
        (root / "replacement.json").write_text("{}", encoding="utf-8")
        return original_manifest(descriptor)

    monkeypatch.setattr(
        qualification,
        "_artifact_manifest_from_descriptor",
        replace_root_after_open,
    )
    with pytest.raises(ValueError, match="changed during verification"):
        qualification._current_artifact_evidence(
            root,
            storage_root=storage,
        )


def test_cli_protects_separate_tool_repository_outputs(tmp_path: Path) -> None:
    tool_repository = tmp_path / "tool"
    tool_git_directory = tmp_path / "git" / "tool"
    host = {
        "execution": {
            "repository": str(tmp_path / "execution"),
            "git_directory": str(tmp_path / "git" / "execution"),
        },
        "tool": {
            "repository": str(tool_repository),
            "git_directory": str(tool_git_directory),
        },
        "storage": {"storage_root": str(tmp_path / "storage")},
        "probe_storage": {"artifact_root": str(tmp_path / "storage" / "probe")},
    }

    roots = qualification_cli._protected_roots(host)
    assert tool_repository in roots
    assert tool_git_directory in roots
    for protected in (tool_repository, tool_git_directory):
        with pytest.raises(ValueError, match="outside protected root"):
            validate_output_path(
                protected / "qualification.json",
                forbidden_roots=roots,
            )
