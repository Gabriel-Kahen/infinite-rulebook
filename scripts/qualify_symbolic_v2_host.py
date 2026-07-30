"""Qualify an intended symbolic-v2 host without running a registered study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from infinite_rulebook.operations.host_qualification import (
    DEFAULT_CAPACITY_TIMEOUT_SECONDS,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    assess_qualification,
    bind_probe_result,
    build_exact_plan,
    inspect_host,
    run_capacity_benchmark,
    run_probe_step,
    validate_output_path,
    verify_assessment_bundle,
    verify_current_context,
    verify_record,
    write_record,
)
from infinite_rulebook.orchestration.jsonio import load_json_strict


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _static_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument(
        "--additional-storage-bytes",
        type=_positive_integer,
        required=True,
    )
    parser.add_argument(
        "--additional-inodes",
        type=_positive_integer,
        required=True,
    )
    parser.add_argument("--probe-root", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read-only host checks and synthetic capacity evidence for v2"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print exact qualification commands")
    plan.add_argument("--execution-commit", required=True)
    plan.add_argument("--repo-root", type=Path, default=Path("."))
    plan.add_argument("--probe-root", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="capture static host evidence")
    _static_arguments(inspect)
    inspect.add_argument("--output", type=Path)

    capacity = commands.add_parser(
        "run-capacity",
        help="run one synthetic analysis capacity benchmark",
    )
    capacity.add_argument("--host-record", type=Path, required=True)
    capacity.add_argument("--stage", choices=("e192", "e768"), required=True)
    capacity.add_argument(
        "--acknowledge-e768-synthetic-memory-pressure",
        action="store_true",
    )
    capacity.add_argument(
        "--sample-interval-seconds",
        type=_positive_float,
        default=0.25,
    )
    capacity.add_argument(
        "--timeout-hours",
        type=_positive_float,
        default=DEFAULT_CAPACITY_TIMEOUT_SECONDS / 3600,
    )
    capacity.add_argument("--output", type=Path, required=True)

    bind_probe = commands.add_parser(
        "bind-probe",
        help="seal imported plain probe output as nonqualifying evidence",
    )
    bind_probe.add_argument(
        "--kind",
        choices=("execution", "benchmark"),
        required=True,
    )
    bind_probe.add_argument("--host-record", type=Path, required=True)
    bind_probe.add_argument("--result", type=Path, required=True)
    bind_probe.add_argument("--probe-execution-record", type=Path)
    bind_probe.add_argument("--output", type=Path, required=True)

    run_probe = commands.add_parser(
        "run-probe",
        help="run one probe step through a descriptor-anchored artifact path",
    )
    run_probe.add_argument(
        "--kind",
        choices=("execution", "benchmark"),
        required=True,
    )
    run_probe.add_argument("--host-record", type=Path, required=True)
    run_probe.add_argument("--probe-execution-record", type=Path)
    run_probe.add_argument(
        "--timeout-hours",
        type=_positive_float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS / 3600,
    )
    run_probe.add_argument("--output", type=Path, required=True)

    assess = commands.add_parser(
        "assess",
        help="combine completed qualification records without running a study",
    )
    assess.add_argument("--host-record", type=Path, required=True)
    assess.add_argument("--e192-record", type=Path, required=True)
    assess.add_argument("--e768-record", type=Path, required=True)
    assess.add_argument("--probe-execution-record", type=Path, required=True)
    assess.add_argument("--probe-benchmark-record", type=Path, required=True)
    assess.add_argument("--available-window-hours", type=_positive_float, required=True)
    assess.add_argument(
        "--recovery-margin-hours",
        type=_nonnegative_float,
        required=True,
    )
    assess.add_argument("--output", type=Path)

    verify = commands.add_parser(
        "verify-assessment",
        help="verify an assessment against all five bound input records",
    )
    verify.add_argument("--assessment-record", type=Path, required=True)
    verify.add_argument("--host-record", type=Path, required=True)
    verify.add_argument("--e192-record", type=Path, required=True)
    verify.add_argument("--e768-record", type=Path, required=True)
    verify.add_argument("--probe-execution-record", type=Path, required=True)
    verify.add_argument("--probe-benchmark-record", type=Path, required=True)
    return parser


def _protected_roots(host: dict[str, object]) -> tuple[Path, ...]:
    return (
        Path(host["execution"]["repository"]),  # type: ignore[index]
        Path(host["execution"]["git_directory"]),  # type: ignore[index]
        Path(host["tool"]["repository"]),  # type: ignore[index]
        Path(host["tool"]["git_directory"]),  # type: ignore[index]
        Path(host["storage"]["storage_root"]),  # type: ignore[index]
        Path(host["probe_storage"]["artifact_root"]),  # type: ignore[index]
    )


def _emit(
    payload: object,
    output: Path | None,
    *,
    host: dict[str, object] | None = None,
) -> None:
    if output is not None:
        write_record(
            output,
            payload,  # type: ignore[arg-type]
            forbidden_roots=() if host is None else _protected_roots(host),
        )
        if host is not None:
            verify_current_context(host)
    print(json.dumps(payload, sort_keys=True))


def _load(path: Path, label: str) -> dict[str, object]:
    payload = load_json_strict(path, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "plan":
        _emit(
            build_exact_plan(
                repo_root=arguments.repo_root,
                execution_commit=arguments.execution_commit,
                probe_root=arguments.probe_root,
            ),
            None,
        )
        return 0
    if arguments.command in {"inspect", "run-capacity"}:
        if arguments.command == "run-capacity":
            host = verify_record(
                _load(arguments.host_record, "host record"),
                record_type="symbolic-v2-host-static-qualification",
            )
            validate_output_path(
                arguments.output,
                forbidden_roots=_protected_roots(host),
            )
            record = run_capacity_benchmark(
                stage=arguments.stage,
                static_record=host,
                acknowledge_e768=(arguments.acknowledge_e768_synthetic_memory_pressure),
                sample_interval_seconds=arguments.sample_interval_seconds,
                timeout_seconds=arguments.timeout_hours * 3600,
            )
            _emit(record, arguments.output, host=host)
            return 0 if record["decision"]["passed"] else 2
        forbidden = (
            arguments.repo_root,
            arguments.storage_root,
            arguments.probe_root
            or arguments.storage_root / "symbolic-v2-ingestion-probe",
        )
        if arguments.output is not None:
            validate_output_path(arguments.output, forbidden_roots=forbidden)
        host = inspect_host(
            repo_root=arguments.repo_root,
            execution_commit=arguments.execution_commit,
            storage_root=arguments.storage_root,
            additional_storage_bytes=arguments.additional_storage_bytes,
            additional_inodes=arguments.additional_inodes,
            probe_root=arguments.probe_root,
        )
        _emit(host, arguments.output, host=host)
        return 0 if host["decision"]["static_prerequisites_passed"] else 2
    if arguments.command in {"bind-probe", "run-probe"}:
        host = verify_record(
            _load(arguments.host_record, "host record"),
            record_type="symbolic-v2-host-static-qualification",
        )
        validate_output_path(
            arguments.output,
            forbidden_roots=_protected_roots(host),
        )
        execution_record = (
            None
            if arguments.probe_execution_record is None
            else _load(
                arguments.probe_execution_record,
                "probe execution record",
            )
        )
        if arguments.kind == "benchmark" and execution_record is None:
            raise ValueError(
                "--probe-execution-record is required for benchmark binding"
            )
        if arguments.kind == "execution" and execution_record is not None:
            raise ValueError(
                "--probe-execution-record is only valid for benchmark binding"
            )
        if arguments.command == "run-probe":
            record = run_probe_step(
                kind=arguments.kind,
                static_record=host,
                probe_execution_record=execution_record,
                timeout_seconds=arguments.timeout_hours * 3600,
            )
        else:
            record = bind_probe_result(
                kind=arguments.kind,
                static_record=host,
                result=_load(arguments.result, "plain probe result"),
                probe_execution_record=execution_record,
            )
        _emit(record, arguments.output, host=host)
        return 0 if record["decision"]["shape_passed"] else 2
    host = verify_record(
        _load(arguments.host_record, "host record"),
        record_type="symbolic-v2-host-static-qualification",
    )
    if arguments.command == "verify-assessment":
        record = verify_assessment_bundle(
            assessment_record=_load(
                arguments.assessment_record,
                "assessment record",
            ),
            static_record=host,
            e192_record=_load(arguments.e192_record, "E192 record"),
            e768_record=_load(arguments.e768_record, "E768 record"),
            probe_execution_record=_load(
                arguments.probe_execution_record,
                "probe execution record",
            ),
            probe_benchmark_record=_load(
                arguments.probe_benchmark_record,
                "probe benchmark record",
            ),
        )
        _emit(record, None)
        return 0
    if arguments.output is not None:
        validate_output_path(
            arguments.output,
            forbidden_roots=_protected_roots(host),
        )
    record = assess_qualification(
        static_record=host,
        e192_record=_load(arguments.e192_record, "E192 record"),
        e768_record=_load(arguments.e768_record, "E768 record"),
        probe_execution_result=_load(
            arguments.probe_execution_record,
            "probe execution record",
        ),
        probe_benchmark_result=_load(
            arguments.probe_benchmark_record,
            "probe benchmark record",
        ),
        available_window_hours=arguments.available_window_hours,
        recovery_margin_hours=arguments.recovery_margin_hours,
    )
    _emit(record, arguments.output, host=host)
    return 0 if record["decision"]["qualification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
