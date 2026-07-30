"""Command-line entry points for sealed symbolic experiments and artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import secrets
import shutil
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from infinite_rulebook.analysis import (
    Alternative,
    AnalysisPhase,
    EquivalencePowerHypothesis,
    PowerHypothesis,
    analysis_plan_json,
    build_report,
    calibrate_environment_count,
    canary_results_csv,
    evaluate_canaries,
    load_analysis_plan,
    load_run_trees,
    power_calibration_csv,
    registered_gates_csv,
    terminal_summary_csv,
    trajectories_svg,
)
from infinite_rulebook.orchestration.artifacts import validate_artifact_tree
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.orchestration.freeze import (
    SeedBankIdentities,
    freeze_experiment_config,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.inventory import RawArtifactInventory
from infinite_rulebook.orchestration.provenance import collect_provenance
from infinite_rulebook.orchestration.release import (
    STUDY_RELEASE_MANIFEST_FILENAME,
    StudyReleaseManifest,
    load_study_release_manifest,
)
from infinite_rulebook.orchestration.reproducibility import (
    ReproducibilityReport,
    run_reproducibility_check,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.studies.smoke_prerequisite import (
    SmokePrerequisiteEvidence,
)
from infinite_rulebook.studies.symbolic_construct import (
    STUDY_CONTRACT,
    build_symbolic_analysis_plan,
    build_symbolic_canary_plan,
)
from infinite_rulebook.studies.symbolic_registry import (
    SYMBOLIC_STUDY_V1,
    SymbolicStudySpec,
    execution_adapter_factory,
    registered_symbolic_study,
)

# Backward-compatible module constants for callers that inspected the v1 CLI.
POWER_SIMULATIONS = SYMBOLIC_STUDY_V1.power.simulations
POWER_CANDIDATE_ENVIRONMENTS = (
    SYMBOLIC_STUDY_V1.power.candidate_environment_counts
)
POWER_SEED = SYMBOLIC_STUDY_V1.power.seed
POWER_ALPHA = SYMBOLIC_STUDY_V1.power.alpha
POWER_SIMULATION_ERROR_ALPHA = SYMBOLIC_STUDY_V1.power.simulation_error_alpha
MINIMUM_INDIVIDUAL_POWER = SYMBOLIC_STUDY_V1.power.minimum_individual_power
MINIMUM_EQUIVALENCE_POWER = SYMBOLIC_STUDY_V1.power.minimum_equivalence_power
MINIMUM_JOINT_POWER = SYMBOLIC_STUDY_V1.power.minimum_joint_power
MAXIMUM_GLOBAL_NULL_FWER = SYMBOLIC_STUDY_V1.power.maximum_global_null_fwer
PRIMARY_MINIMUM_EFFECTS = SYMBOLIC_STUDY_V1.power.minimum_effects
S5_REWARD_EQUIVALENCE_MARGIN = SYMBOLIC_STUDY_V1.power.equivalence_margin


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infinite-rulebook")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("config", type=Path)
        command.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
        command.add_argument("--workers", type=int, default=1)
        command.add_argument(
            "--smoke-evidence",
            type=Path,
            help="authenticated Stage-0 evidence (required for calibration)",
        )

    run = subparsers.add_parser(
        "run",
        help="run a pilot, calibration, or sealed confirmatory experiment",
    )
    add_run_arguments(run)
    pilot = subparsers.add_parser(
        "pilot",
        help="run a bounded pilot (legacy phase-strict alias)",
    )
    add_run_arguments(pilot)
    validate = subparsers.add_parser("validate", help="validate an artifact tree")
    validate.add_argument("path", type=Path)
    inventory = subparsers.add_parser(
        "inventory",
        help="write a portable checksum inventory for one complete raw root",
    )
    inventory.add_argument("config", type=Path)
    inventory.add_argument("artifact_root", type=Path)
    inventory.add_argument("side", choices=("serial", "parallel"))
    inventory.add_argument("output", type=Path)
    verify_inventory = subparsers.add_parser(
        "verify-inventory",
        help="verify a raw root against a portable checksum inventory",
    )
    verify_inventory.add_argument("config", type=Path)
    verify_inventory.add_argument("artifact_root", type=Path)
    verify_inventory.add_argument("inventory", type=Path)
    verify_inventory.add_argument(
        "--side",
        choices=("serial", "parallel"),
    )
    plan = subparsers.add_parser(
        "plan",
        help="write the registered bounded symbolic analysis plan",
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("output", type=Path)
    plan.add_argument("canary_output", type=Path)
    report = subparsers.add_parser(
        "report",
        help="validate and report a complete symbolic study artifact root",
    )
    report.add_argument("config", type=Path)
    report.add_argument("plan", type=Path)
    report.add_argument("canary_plan", type=Path)
    report.add_argument("artifact_root", type=Path)
    report.add_argument("output_dir", type=Path)
    report.add_argument(
        "--power-simulations",
        type=int,
        default=10_000,
    )
    report.add_argument(
        "--reproducibility-report",
        type=Path,
        required=True,
    )
    report.add_argument(
        "--smoke-evidence",
        type=Path,
        help="authenticated Stage-0 evidence (required for calibration)",
    )
    report.add_argument(
        "--deviation",
        action="append",
        default=[],
        help="record one protocol or execution deviation in the hashed package",
    )
    smoke = subparsers.add_parser(
        "smoke-evidence",
        help="authenticate and bind the registered Stage-0 smoke prerequisite",
    )
    smoke.add_argument("config", type=Path)
    smoke.add_argument("reproducibility_report", type=Path)
    smoke.add_argument("serial_inventory", type=Path)
    smoke.add_argument("parallel_inventory", type=Path)
    smoke.add_argument("output", type=Path)
    smoke.add_argument(
        "--anomaly",
        action="append",
        default=[],
        help="record one non-invalidating Stage-0 engineering anomaly",
    )
    reproduce = subparsers.add_parser(
        "reproduce",
        help="run and authenticate complete serial and parallel sweeps",
    )
    reproduce.add_argument("config", type=Path)
    reproduce.add_argument("serial_root", type=Path)
    reproduce.add_argument("parallel_root", type=Path)
    reproduce.add_argument("output", type=Path)
    reproduce.add_argument("--workers", type=int, default=4)
    reproduce.add_argument(
        "--smoke-evidence",
        type=Path,
        help="authenticated Stage-0 evidence (required for calibration)",
    )
    reproduce.add_argument(
        "--resume",
        action="store_true",
        help="resume only a matching receipt-bound interrupted invocation",
    )
    freeze = subparsers.add_parser(
        "freeze",
        help="seal a passing calibration as a confirmatory config and plan",
    )
    freeze.add_argument("calibration_config", type=Path)
    freeze.add_argument("calibration_summary", type=Path)
    freeze.add_argument("output_config", type=Path)
    freeze.add_argument("output_plan", type=Path)
    freeze.add_argument("output_canary_plan", type=Path)
    return parser


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(
    path: Path,
    payload: object,
    *,
    verifier: Callable[[Path], None] | None = None,
) -> None:
    normalized = _jsonable(payload)

    def verify(candidate: Path) -> None:
        if _read_json_object(candidate, label="persisted JSON output") != normalized:
            raise ValueError(f"persisted JSON output differs from its payload: {path}")
        if verifier is not None:
            verifier(candidate)

    _write_immutable_text(
        path,
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        verifier=verify,
    )


def _write_immutable_text(
    path: Path,
    content: str,
    *,
    verifier: Callable[[Path], None] | None = None,
) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite immutable output: {path}")
        if verifier is not None:
            verifier(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if verifier is not None:
            verifier(temporary)
        try:
            os.link(temporary, path)
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(
                    f"refusing to overwrite immutable output: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _expected_release_members(phase: AnalysisPhase) -> tuple[str, ...]:
    members = (
        "analysis.json",
        "analysis.md",
        "analysis-plan.json",
        "canary-plan.json",
        "canaries.json",
        "deviations.json",
        "experiment-config.json",
        "reproducibility.json",
        "raw-parallel-inventory.json",
        "raw-serial-inventory.json",
        "registered-gates.csv",
        "canary-results.csv",
        "terminal-summary.csv",
        "trajectories.svg",
        "summary.json",
    )
    if phase is AnalysisPhase.CALIBRATION:
        return (
            *members,
            "power.json",
            "power-calibration.csv",
            "smoke-prerequisite.json",
        )
    if phase is AnalysisPhase.CONFIRMATORY:
        return members
    raise ValueError("study release packages require calibration or confirmation")


def _load_registered_canary_plan(path: Path, expected: object) -> None:
    payload = _read_json_object(path, label="scientific canary plan")
    if payload != _jsonable(expected):
        raise ValueError("canary plan differs from the registered symbolic study plan")


def _deviation_log(
    experiment: ExperimentConfig,
    deviations: Sequence[str],
    *,
    study_contract: str = STUDY_CONTRACT,
) -> dict[str, object]:
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in deviations
    ):
        raise ValueError("deviations must be nonempty strings without outer whitespace")
    entries = tuple(sorted(deviations))
    if len(entries) != len(set(entries)):
        raise ValueError("deviations must be unique")
    payload: dict[str, object] = {
        "artifact_type": "symbolic-study-deviation-log",
        "schema_version": 1,
        "study_contract": study_contract,
        "phase": experiment.phase,
        "config_hash": experiment.config_hash,
        "deviations": list(entries),
    }
    payload["scientific_hash"] = scientific_hash(
        payload,
        domain="symbolic-study-deviation-log",
    )
    return payload


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _preflight_output_paths(
    *,
    output_files: tuple[Path, ...] = (),
    output_directories: tuple[Path, ...] = (),
    protected_files: tuple[Path, ...] = (),
    protected_directories: tuple[Path, ...] = (),
) -> None:
    files = tuple(_resolved(path) for path in output_files)
    directories = tuple(_resolved(path) for path in output_directories)
    all_outputs = (*files, *directories)
    if len(set(all_outputs)) != len(all_outputs):
        raise ValueError("output paths must be distinct")
    for index, left in enumerate(files):
        if any(
            _within(left, right) or _within(right, left) for right in files[index + 1 :]
        ):
            raise ValueError("output files must not overlap")
    if any(
        _within(file, directory) or _within(directory, file)
        for file in files
        for directory in directories
    ):
        raise ValueError(
            "output files must not be inside output directories or contain them"
        )
    for index, left in enumerate(directories):
        if any(
            _within(left, right) or _within(right, left)
            for right in directories[index + 1 :]
        ):
            raise ValueError("output directories must not overlap")
    protected_file_paths = tuple(_resolved(path) for path in protected_files)
    protected_directory_paths = tuple(_resolved(path) for path in protected_directories)
    for output in files:
        if output in protected_file_paths or any(
            _within(output, directory) for directory in protected_directory_paths
        ):
            raise ValueError("output file overlaps an authenticated input")
    for output in directories:
        if any(
            _within(path, output) or _within(output, path)
            for path in protected_file_paths
        ) or any(
            _within(output, directory) or _within(directory, output)
            for directory in protected_directory_paths
        ):
            raise ValueError("output directory overlaps an authenticated input")


def _completed_run_roots(artifact_root: Path, experiment: str) -> tuple[Path, ...]:
    root = artifact_root / experiment
    if not root.is_dir():
        raise ValueError(f"experiment artifact directory does not exist: {root}")
    entries = tuple(sorted(root.iterdir()))
    if any(
        path.is_symlink()
        or not path.is_dir()
        or not is_sha256(path.name)
        or not (path / "manifest.json").is_file()
        for path in entries
    ):
        raise ValueError("experiment artifact inventory contains an invalid entry")
    return entries


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"{label} repeats key {name!r}")
            result[name] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_hashed_json(
    path: Path,
    *,
    label: str,
    domain: str,
) -> dict[str, object]:
    payload = _read_json_object(path, label=label)
    recorded = payload.get("scientific_hash")
    body = {name: value for name, value in payload.items() if name != "scientific_hash"}
    if not is_sha256(recorded) or recorded != scientific_hash(
        body,
        domain=domain,
    ):
        raise ValueError(f"{label} scientific hash is invalid")
    return payload


def _load_calibration_summary(path: Path) -> dict[str, object]:
    return _load_hashed_json(
        path,
        label="calibration summary",
        domain="symbolic-study-summary",
    )


def _load_reproducibility_report(
    path: Path,
    *,
    experiment: ExperimentConfig,
) -> ReproducibilityReport:
    payload = _read_json_object(path, label="reproducibility report")
    report = ReproducibilityReport.from_dict(payload)
    if report.config_hash != experiment.config_hash or len(report.runs) != len(
        experiment.cells()
    ):
        raise ValueError("reproducibility report does not match the experiment")
    return report


def _trusted_reproducibility_root(
    requested: Path,
    reproducibility: ReproducibilityReport,
) -> Path:
    resolved = _resolved(requested)
    candidates = (
        _resolved(reproducibility.serial_root),
        _resolved(reproducibility.parallel_root),
    )
    if resolved not in candidates:
        raise ValueError(
            "reporting requires an exact receipt-bound reproducibility root"
        )
    return candidates[candidates.index(resolved)]


def _load_smoke_prerequisite(path: Path) -> SmokePrerequisiteEvidence:
    return SmokePrerequisiteEvidence.from_dict(
        _read_json_object(path, label="smoke prerequisite"),
        verify_roots=True,
    )


def _require_equal(observed: object, expected: object, *, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} differs from its source")


def _verify_report_json_outputs(
    output: Path,
    *,
    experiment: ExperimentConfig,
    plan: object,
    canary_plan: object,
    report: object,
    canaries: object,
    deviations: dict[str, object],
    reproducibility: ReproducibilityReport,
    serial_inventory: RawArtifactInventory,
    parallel_inventory: RawArtifactInventory,
    smoke_evidence: SmokePrerequisiteEvidence | None,
    power: dict[str, object] | None,
    summary: dict[str, object],
) -> None:
    _require_equal(
        load_experiment_config(output / "experiment-config.json"),
        experiment,
        label="persisted report config",
    )
    _require_equal(
        load_analysis_plan(output / "analysis-plan.json"),
        plan,
        label="persisted report analysis plan",
    )
    _load_registered_canary_plan(
        output / "canary-plan.json",
        canary_plan.to_dict(),
    )
    for name, filename, expected, domain in (
        (
            "analysis",
            "analysis.json",
            report.to_payload(),
            "registered-analysis-report",
        ),
        (
            "canary",
            "canaries.json",
            canaries.to_dict(),
            "scientific-canary-report",
        ),
        (
            "deviation",
            "deviations.json",
            deviations,
            "symbolic-study-deviation-log",
        ),
        ("summary", "summary.json", summary, "symbolic-study-summary"),
    ):
        _require_equal(
            _load_hashed_json(
                output / filename,
                label=f"persisted {name} evidence",
                domain=domain,
            ),
            expected,
            label=f"persisted {name} evidence",
        )
    persisted_reproducibility = ReproducibilityReport.from_dict(
        _read_json_object(
            output / "reproducibility.json",
            label="persisted reproducibility report",
        ),
        expected_serial_root=reproducibility.serial_root,
        expected_parallel_root=reproducibility.parallel_root,
        expected_parallel_workers=reproducibility.parallel_workers,
    )
    _require_equal(
        persisted_reproducibility,
        reproducibility,
        label="persisted reproducibility report",
    )
    for name, expected in (
        ("serial", serial_inventory),
        ("parallel", parallel_inventory),
    ):
        persisted = RawArtifactInventory.from_dict(
            _read_json_object(
                output / f"raw-{name}-inventory.json",
                label=f"persisted {name} raw inventory",
            )
        )
        _require_equal(
            persisted,
            expected,
            label=f"persisted {name} raw inventory",
        )
    if smoke_evidence is not None:
        _require_equal(
            _load_smoke_prerequisite(output / "smoke-prerequisite.json"),
            smoke_evidence,
            label="persisted smoke prerequisite",
        )
    if power is not None:
        _require_equal(
            _load_hashed_json(
                output / "power.json",
                label="persisted power calibration",
                domain="symbolic-power-calibration",
            ),
            power,
            label="persisted power calibration",
        )


def _smoke_prerequisite_for_execution(
    experiment: ExperimentConfig,
    path: Path | None,
    *,
    study: SymbolicStudySpec | None = None,
) -> SmokePrerequisiteEvidence | None:
    if experiment.phase == "calibration":
        if path is None:
            raise ValueError(
                "registered calibration execution requires Stage-0 smoke evidence"
            )
        selected = study or registered_symbolic_study(experiment.name)
        evidence = _load_smoke_prerequisite(path)
        if selected.smoke_prerequisite_hash is not None and (
            evidence.config.config_hash != selected.smoke_config_hash
            or evidence.scientific_hash != selected.smoke_prerequisite_hash
        ):
            raise ValueError(
                f"symbolic v{selected.version} calibration requires its exact "
                "registered Stage-0 evidence"
            )
        return evidence
    if path is not None:
        raise ValueError("smoke evidence is accepted only for calibration execution")
    return None


def _verify_registered_execution(
    experiment: ExperimentConfig,
) -> SymbolicStudySpec | None:
    if experiment.phase not in {"calibration", "confirmatory"}:
        return None
    study = registered_symbolic_study(experiment.name)
    if experiment.phase == "calibration":
        study.verify_calibration(experiment)
    else:
        current = collect_provenance()
        study.verify_confirmatory(
            experiment,
            analysis_code_hash=current.analysis_code_hash,
            dependency_lock_hash=current.dependency_lock_hash,
            environment_digest=current.environment_digest,
        )
    return study


def _validate_inventory_receipts(
    reproducibility: ReproducibilityReport,
    serial_inventory: RawArtifactInventory,
    parallel_inventory: RawArtifactInventory,
    *,
    smoke_prerequisite_hash: str | None = None,
) -> None:
    serial = serial_inventory.execution_receipt
    parallel = parallel_inventory.execution_receipt
    if (
        serial is None
        or parallel is None
        or serial.scientific_hash != reproducibility.serial_receipt_hash
        or parallel.scientific_hash != reproducibility.parallel_receipt_hash
        or serial.invocation_id != reproducibility.invocation_id
        or parallel.invocation_id != reproducibility.invocation_id
        or serial.pair != parallel.pair
        or serial.parallel_workers != reproducibility.parallel_workers
        or parallel.parallel_workers != reproducibility.parallel_workers
        or serial.smoke_prerequisite_hash != smoke_prerequisite_hash
        or parallel.smoke_prerequisite_hash != smoke_prerequisite_hash
    ):
        raise ValueError(
            "raw inventories do not match the reproducibility receipt pair"
        )
    expected_runs = {
        (run.run_hash, run.scientific_content_hash) for run in reproducibility.runs
    }
    serial_runs = {
        (tree.identity_hash, tree.scientific_content_hash)
        for tree in serial_inventory.trees
        if tree.tree_type == "run"
    }
    parallel_runs = {
        (tree.identity_hash, tree.scientific_content_hash)
        for tree in parallel_inventory.trees
        if tree.tree_type == "run"
    }
    if serial_runs != expected_runs or parallel_runs != expected_runs:
        raise ValueError(
            "raw inventories do not match the reproducibility run evidence"
        )


def _reconstruct_registered_power(
    analysis: dict[str, object],
    calibration_environment_count: int,
    algorithm_replicas: int,
    *,
    study: SymbolicStudySpec = SYMBOLIC_STUDY_V1,
) -> tuple[
    tuple[PowerHypothesis, ...],
    tuple[EquivalencePowerHypothesis, ...],
]:
    """Rebuild frozen power inputs from the authenticated analysis artifact."""

    raw_contrasts = analysis.get("contrasts")
    minimum_effects = study.power.minimum_effects
    if not isinstance(raw_contrasts, list) or len(raw_contrasts) != len(
        minimum_effects
    ):
        raise ValueError("calibration analysis primary contrasts are invalid")
    by_name = {
        item.get("name"): item
        for item in raw_contrasts
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != set(minimum_effects):
        raise ValueError("calibration analysis changed the registered primary family")
    directional = []
    for name in sorted(minimum_effects):
        item = by_name[name]
        differences = item.get("differences")
        if (
            item.get("pair_count") != calibration_environment_count
            or item.get("cell_pair_count")
            != calibration_environment_count * algorithm_replicas
            or not isinstance(differences, list)
            or len(differences) != calibration_environment_count
        ):
            raise ValueError(
                f"calibration contrast {name!r} has an invalid cluster inventory"
            )
        directional.append(
            PowerHypothesis.from_cluster_differences(
                name,
                tuple(differences),
                minimum_effect=minimum_effects[name],
                alternative=Alternative.GREATER,
                algorithm_replicas_per_environment=algorithm_replicas,
            )
        )

    raw_equivalences = analysis.get("equivalences")
    if not isinstance(raw_equivalences, list) or len(raw_equivalences) != 1:
        raise ValueError("calibration analysis equivalence family is invalid")
    item = raw_equivalences[0]
    if not isinstance(item, dict):
        raise ValueError("calibration analysis equivalence result is invalid")
    differences = item.get("differences")
    if (
        item.get("name") != study.power.equivalence_name
        or item.get("margin") != study.power.equivalence_margin
        or item.get("pair_count") != calibration_environment_count
        or item.get("cell_pair_count")
        != calibration_environment_count * algorithm_replicas
        or not isinstance(differences, list)
        or len(differences) != calibration_environment_count
    ):
        raise ValueError("calibration analysis changed the registered S5 equivalence")
    equivalences = (
        EquivalencePowerHypothesis.from_cluster_differences(
            item["name"],
            tuple(differences),
            margin=study.power.equivalence_margin,
            diagnostic_location=study.power.equivalence_diagnostic_location,
            algorithm_replicas_per_environment=algorithm_replicas,
        ),
    )
    return tuple(directional), equivalences


def _freeze_study(arguments: argparse.Namespace) -> int:
    _preflight_output_paths(
        output_files=(
            arguments.output_config,
            arguments.output_plan,
            arguments.output_canary_plan,
        ),
        protected_files=(arguments.calibration_config,),
        protected_directories=(arguments.calibration_summary.parent,),
    )
    calibration = load_experiment_config(arguments.calibration_config)
    if calibration.phase != "calibration":
        raise ValueError("only a calibration config can be frozen")
    study = registered_symbolic_study(calibration.name)
    study.verify_calibration(calibration)
    summary = _load_calibration_summary(arguments.calibration_summary)
    if summary.get("phase") != "calibration":
        raise ValueError("summary is not calibration evidence")
    if summary.get("config_hash") != calibration.config_hash:
        raise ValueError("summary does not match the calibration config")
    if summary.get("run_count") != len(calibration.cells()):
        raise ValueError("summary does not cover the complete calibration inventory")
    expected_calibration_plan = study.build_analysis_plan(
        calibration,
        phase=AnalysisPhase.CALIBRATION,
    )
    if (
        summary.get("analysis_plan_hash") != expected_calibration_plan.scientific_hash
        or summary.get("analysis_registration_hash")
        != expected_calibration_plan.registration_hash
    ):
        raise ValueError("summary does not use the registered calibration analysis")
    expected_calibration_canaries = build_symbolic_canary_plan(
        calibration,
        phase=AnalysisPhase.CALIBRATION,
    )
    if summary.get("canary_plan_hash") != expected_calibration_canaries.scientific_hash:
        raise ValueError("summary does not use the registered calibration canaries")
    if summary.get("canaries_passed") is not True:
        raise ValueError("failed scientific canaries block confirmation")
    if summary.get("freeze_eligible") is not True:
        raise ValueError("calibration summary is not eligible for confirmation")
    expected_power_design = study.power.summary_fields(
        include_rng_stream=study.records_power_rng_stream
    )
    for name, expected in expected_power_design.items():
        if summary.get(name) != expected:
            raise ValueError(f"calibration summary changed registered {name}")
    expected_run_settings_hash = scientific_hash(
        calibration.resolved_run_settings(),
        domain="resolved-run-settings",
    )
    if summary.get("run_settings_hash") != expected_run_settings_hash:
        raise ValueError("calibration run settings do not match the supplied config")
    evidence_dir = arguments.calibration_summary.parent
    release = load_study_release_manifest(
        evidence_dir / STUDY_RELEASE_MANIFEST_FILENAME
    )
    if (
        release.phase != "calibration"
        or release.study_contract != study.study_contract
        or release.config_hash != calibration.config_hash
        or release.freeze_hash is not None
        or release.calibration_evidence_hash != summary.get("calibration_evidence_hash")
        or {member.path for member in release.members}
        != set(_expected_release_members(AnalysisPhase.CALIBRATION))
    ):
        raise ValueError("calibration release package does not match the study")
    smoke_evidence = _load_smoke_prerequisite(evidence_dir / "smoke-prerequisite.json")
    if (
        summary.get("smoke_prerequisite_passed") is not True
        or summary.get("smoke_prerequisite_hash") != smoke_evidence.scientific_hash
        or summary.get("smoke_config_hash") != smoke_evidence.config.config_hash
        or summary.get("smoke_reproducibility_hash")
        != smoke_evidence.reproducibility.scientific_hash
        or summary.get("smoke_raw_serial_inventory_hash")
        != smoke_evidence.serial_inventory.scientific_hash
        or summary.get("smoke_raw_parallel_inventory_hash")
        != smoke_evidence.parallel_inventory.scientific_hash
        or summary.get("smoke_engineering_anomalies")
        != list(smoke_evidence.engineering_anomalies)
    ):
        raise ValueError("calibration summary does not bind its smoke prerequisite")
    analysis = _load_hashed_json(
        evidence_dir / "analysis.json",
        label="calibration analysis",
        domain="registered-analysis-report",
    )
    packaged_config = _read_json_object(
        evidence_dir / "experiment-config.json",
        label="packaged calibration config",
    )
    packaged_plan = _read_json_object(
        evidence_dir / "analysis-plan.json",
        label="packaged calibration analysis plan",
    )
    packaged_canary_plan = _read_json_object(
        evidence_dir / "canary-plan.json",
        label="packaged calibration canary plan",
    )
    if (
        packaged_config != _jsonable(calibration.resolved_dict())
        or packaged_plan != json.loads(analysis_plan_json(expected_calibration_plan))
        or packaged_canary_plan != expected_calibration_canaries.to_dict()
    ):
        raise ValueError("packaged calibration registrations are inconsistent")
    canaries = _load_hashed_json(
        evidence_dir / "canaries.json",
        label="calibration canaries",
        domain="scientific-canary-report",
    )
    power = _load_hashed_json(
        evidence_dir / "power.json",
        label="power calibration",
        domain="symbolic-power-calibration",
    )
    deviations = _load_hashed_json(
        evidence_dir / "deviations.json",
        label="calibration deviation log",
        domain="symbolic-study-deviation-log",
    )
    deviation_entries = deviations.get("deviations")
    if (
        set(deviations)
        != {
            "artifact_type",
            "schema_version",
            "study_contract",
            "phase",
            "config_hash",
            "deviations",
            "scientific_hash",
        }
        or deviations.get("artifact_type") != "symbolic-study-deviation-log"
        or isinstance(deviations.get("schema_version"), bool)
        or not isinstance(deviations.get("schema_version"), int)
        or deviations.get("schema_version") != 1
        or deviations.get("study_contract") != study.study_contract
        or deviations.get("phase") != "calibration"
        or deviations.get("config_hash") != calibration.config_hash
        or not isinstance(deviation_entries, list)
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in deviation_entries
        )
        or deviation_entries != sorted(set(deviation_entries))
        or summary.get("deviation_log_hash") != deviations["scientific_hash"]
        or summary.get("deviation_count") != len(deviation_entries)
    ):
        raise ValueError("calibration deviation log is invalid or inconsistent")
    if deviation_entries:
        raise ValueError(
            f"symbolic v{study.version} confirmation cannot be frozen after any "
            "deviation"
        )
    serial_inventory = RawArtifactInventory.from_dict(
        _read_json_object(
            evidence_dir / "raw-serial-inventory.json",
            label="calibration serial raw inventory",
        )
    )
    parallel_inventory = RawArtifactInventory.from_dict(
        _read_json_object(
            evidence_dir / "raw-parallel-inventory.json",
            label="calibration parallel raw inventory",
        )
    )
    reproducibility = _load_reproducibility_report(
        evidence_dir / "reproducibility.json",
        experiment=calibration,
    )
    _validate_inventory_receipts(
        reproducibility,
        serial_inventory,
        parallel_inventory,
        smoke_prerequisite_hash=smoke_evidence.scientific_hash,
    )
    _preflight_output_paths(
        output_files=(
            arguments.output_config,
            arguments.output_plan,
            arguments.output_canary_plan,
        ),
        protected_files=(arguments.calibration_config,),
        protected_directories=(
            evidence_dir,
            reproducibility.serial_root,
            reproducibility.parallel_root,
            smoke_evidence.reproducibility.serial_root,
            smoke_evidence.reproducibility.parallel_root,
        ),
    )
    serial_inventory.verify(
        reproducibility.serial_root,
        calibration,
        side="serial",
    )
    parallel_inventory.verify(
        reproducibility.parallel_root,
        calibration,
        side="parallel",
    )
    if (
        summary.get("raw_serial_inventory_hash") != serial_inventory.scientific_hash
        or summary.get("raw_parallel_inventory_hash")
        != parallel_inventory.scientific_hash
    ):
        raise ValueError("calibration raw inventories do not match the summary")
    authenticated_roots = _completed_run_roots(
        reproducibility.parallel_root,
        calibration.name,
    )
    if len(authenticated_roots) != len(calibration.cells()):
        raise ValueError("calibration raw-root inventory is incomplete")
    authenticated_dataset = load_run_trees(
        authenticated_roots,
        expected_phase=AnalysisPhase.CALIBRATION,
        expected_freeze_hash=None,
        expected_run_settings=calibration.resolved_run_settings(),
    )
    rebuilt_canaries = evaluate_canaries(
        authenticated_dataset,
        expected_calibration_canaries,
    )
    rebuilt_analysis = build_report(
        authenticated_dataset,
        expected_calibration_plan,
        canary_report_hash=rebuilt_canaries.scientific_hash,
        canaries_passed=rebuilt_canaries.passed,
        config_hash=calibration.config_hash,
    )
    rebuilt_analysis = dataclasses.replace(
        rebuilt_analysis,
        deviation_log_hash=deviations["scientific_hash"],
        deviation_count=len(deviation_entries),
    )
    if rebuilt_canaries.to_dict() != canaries:
        raise ValueError("calibration canary evidence does not derive from raw roots")
    if rebuilt_analysis.to_payload() != analysis:
        raise ValueError("calibration analysis evidence does not derive from raw roots")
    if summary.get("dataset_hash") != authenticated_dataset.scientific_hash:
        raise ValueError("calibration summary dataset does not derive from raw roots")
    for summary_name, artifact, artifact_name in (
        ("analysis_hash", analysis, "analysis"),
        ("canary_hash", canaries, "canary"),
        ("power_hash", power, "power"),
    ):
        if summary.get(summary_name) != artifact.get("scientific_hash"):
            raise ValueError(
                f"summary {artifact_name} hash does not match its artifact"
            )
    if summary.get("reproducibility_hash") != reproducibility.scientific_hash:
        raise ValueError("summary reproducibility hash does not match its artifact")
    power_result = power.get("calibration")
    hypotheses = power.get("hypotheses")
    equivalence_hypotheses = power.get("equivalence_hypotheses")
    if (
        not isinstance(power_result, dict)
        or not isinstance(hypotheses, list)
        or not isinstance(equivalence_hypotheses, list)
    ):
        raise ValueError("power calibration artifact structure is invalid")
    reconstructed, reconstructed_equivalences = _reconstruct_registered_power(
        analysis,
        calibration.environment_replicas,
        calibration.algorithm_replicas,
        study=study,
    )
    if hypotheses != _jsonable(reconstructed) or equivalence_hypotheses != _jsonable(
        reconstructed_equivalences
    ):
        raise ValueError("power calibration hypotheses do not derive from analysis")
    recomputed_power = calibrate_environment_count(
        reconstructed,
        candidate_environment_counts=study.power.candidate_environment_counts,
        equivalence_hypotheses=reconstructed_equivalences,
        simulations=study.power.simulations,
        seed=study.power.seed,
        rng_stream=study.power.rng_stream,
        alpha=study.power.alpha,
        minimum_power=study.power.minimum_individual_power,
        minimum_equivalence_power=study.power.minimum_equivalence_power,
        minimum_joint_power=study.power.minimum_joint_power,
        maximum_fwer=study.power.maximum_global_null_fwer,
        simulation_error_alpha=study.power.simulation_error_alpha,
        design_confidence_alpha=study.power.design_confidence_alpha,
        center_environment_count=study.power.center_environment_count,
    )
    if power_result != _jsonable(recomputed_power):
        raise ValueError(
            "power calibration does not exactly reproduce the registered simulation"
        )
    expected_selected = recomputed_power.selected_environment_count
    if summary.get("selected_environment_replicas") != expected_selected:
        raise ValueError("summary selected a different confirmatory environment count")
    if (
        analysis.get("phase") != "calibration"
        or analysis.get("config_hash") != calibration.config_hash
        or analysis.get("plan_hash") != expected_calibration_plan.scientific_hash
        or analysis.get("analysis_registration_hash")
        != expected_calibration_plan.registration_hash
        or analysis.get("dataset_hash") != canaries.get("dataset_hash")
        or analysis.get("canary_report_hash") != canaries.get("scientific_hash")
        or analysis.get("canaries_passed") is not True
        or analysis.get("run_settings_hash") != expected_run_settings_hash
        or canaries.get("phase") != "calibration"
        or canaries.get("passed") is not True
        or canaries.get("plan_hash") != expected_calibration_canaries.scientific_hash
    ):
        raise ValueError("calibration analysis/canary evidence is inconsistent")
    provenance = analysis.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("calibration analysis provenance is missing")
    calibration_source_hash = provenance.get("analysis_code_hash")
    calibration_dependency_lock_hash = provenance.get("dependency_lock_hash")
    calibration_environment_digest = provenance.get("environment_digest")
    if (
        not is_sha256(calibration_source_hash)
        or not is_sha256(calibration_dependency_lock_hash)
        or not is_sha256(calibration_environment_digest)
        or summary.get("analysis_code_hash") != calibration_source_hash
        or summary.get("dependency_lock_hash") != calibration_dependency_lock_hash
        or summary.get("environment_digest") != calibration_environment_digest
    ):
        raise ValueError("calibration provenance evidence is inconsistent")
    current_provenance = collect_provenance()
    current_source_hash = current_provenance.analysis_code_hash
    if (
        current_source_hash != calibration_source_hash
        or current_provenance.dependency_lock_hash != calibration_dependency_lock_hash
        or current_provenance.environment_digest != calibration_environment_digest
    ):
        raise ValueError(
            "scientific source, dependency lock, or execution environment changed "
            "after calibration"
        )
    recomputed_evidence = study.calibration_evidence_hash_from_hashes(
        config_hash=calibration.config_hash,
        analysis_report_hash=analysis["scientific_hash"],
        canary_report_hash=canaries["scientific_hash"],
        power_calibration_hash=power["scientific_hash"],
        reproducibility_report_hash=reproducibility.scientific_hash,
        raw_serial_inventory_hash=serial_inventory.scientific_hash,
        raw_parallel_inventory_hash=parallel_inventory.scientific_hash,
        deviation_log_hash=deviations["scientific_hash"],
        smoke_prerequisite_hash=smoke_evidence.scientific_hash,
        smoke_config_hash=smoke_evidence.config.config_hash,
        smoke_reproducibility_hash=(smoke_evidence.reproducibility.scientific_hash),
        smoke_raw_serial_inventory_hash=(
            smoke_evidence.serial_inventory.scientific_hash
        ),
        smoke_raw_parallel_inventory_hash=(
            smoke_evidence.parallel_inventory.scientific_hash
        ),
        analysis_code_hash=calibration_source_hash,
        run_settings_hash=expected_run_settings_hash,
    )
    selected = summary.get("selected_environment_replicas")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
        raise ValueError("power calibration did not select an environment count")
    evidence_hash = summary.get("calibration_evidence_hash")
    if not is_sha256(evidence_hash) or evidence_hash != recomputed_evidence:
        raise ValueError("calibration evidence hash is missing or inconsistent")
    design = dataclasses.replace(
        calibration,
        environment_replicas=selected,
    )
    registration_hash = study.expected_confirmatory_registration(design)
    if calibration.algorithm_master_seed is None:
        raise ValueError(
            "calibration config must declare a fixed algorithm_master_seed"
        )
    banks = SeedBankIdentities.bind(
        calibration_master_seed=calibration.master_seed,
        confirmatory_master_seed=study.confirmatory_master_seed,
        algorithm_master_seed=calibration.algorithm_master_seed,
        calibration_namespace=study.seed_namespaces[0],
        confirmatory_namespace=study.seed_namespaces[1],
        algorithm_namespace=study.seed_namespaces[2],
        evaluation_namespace=study.seed_namespaces[3],
    )
    sealed = freeze_experiment_config(
        design,
        name=study.confirmatory_name,
        confirmatory_master_seed=study.confirmatory_master_seed,
        calibration_evidence_hash=evidence_hash,
        analysis_contract=study.study_contract,
        analysis_version=registration_hash,
        analysis_code_hash=current_source_hash,
        dependency_lock_hash=current_provenance.dependency_lock_hash,
        environment_digest=current_provenance.environment_digest,
        seed_banks=banks,
        tolerances=study.expected_confirmatory_tolerances(design),
        margins=study.expected_confirmatory_margins(),
    )
    study.verify_confirmatory(
        sealed,
        analysis_code_hash=current_source_hash,
        dependency_lock_hash=current_provenance.dependency_lock_hash,
        environment_digest=current_provenance.environment_digest,
    )
    assert sealed.confirmatory_freeze is not None
    plan = study.build_analysis_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
        freeze_hash=sealed.confirmatory_freeze.seal_hash,
    )
    if plan.registration_hash != sealed.confirmatory_freeze.analysis_version:
        raise AssertionError("sealed analysis registration changed during freezing")
    canary_plan = build_symbolic_canary_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
    )
    _write_json(
        arguments.output_config,
        sealed.resolved_dict(),
        verifier=lambda candidate: _require_equal(
            load_experiment_config(candidate),
            sealed,
            label="persisted confirmatory config",
        ),
    )
    _write_immutable_text(
        arguments.output_plan,
        analysis_plan_json(plan),
        verifier=lambda candidate: _require_equal(
            load_analysis_plan(candidate),
            plan,
            label="persisted confirmatory analysis plan",
        ),
    )
    _write_immutable_text(
        arguments.output_canary_plan,
        canary_plan.to_json(),
        verifier=lambda candidate: _load_registered_canary_plan(
            candidate,
            canary_plan.to_dict(),
        ),
    )
    result = {
        "config": str(arguments.output_config),
        "plan": str(arguments.output_plan),
        "canary_plan": str(arguments.output_canary_plan),
        "config_hash": sealed.config_hash,
        "freeze_hash": sealed.confirmatory_freeze.seal_hash,
        "analysis_registration_hash": plan.registration_hash,
        "canary_plan_hash": canary_plan.scientific_hash,
        "analysis_code_hash": sealed.confirmatory_freeze.analysis_code_hash,
        "dependency_lock_hash": sealed.confirmatory_freeze.dependency_lock_hash,
        "environment_digest": sealed.confirmatory_freeze.environment_digest,
        "environment_replicas": selected,
        "algorithm_replicas": sealed.algorithm_replicas,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _write_study_report(arguments: argparse.Namespace) -> int:
    target = arguments.output_dir
    _preflight_output_paths(
        output_directories=(target,),
        protected_files=(
            arguments.config,
            arguments.plan,
            arguments.canary_plan,
            arguments.reproducibility_report,
            *((arguments.smoke_evidence,) if arguments.smoke_evidence else ()),
        ),
        protected_directories=(arguments.artifact_root,),
    )
    experiment = load_experiment_config(arguments.config)
    study = registered_symbolic_study(experiment.name)
    smoke_evidence = (
        _smoke_prerequisite_for_execution(
            experiment,
            arguments.smoke_evidence,
            study=study,
        )
        if experiment.phase == "calibration"
        else None
    )
    if experiment.phase != "calibration" and arguments.smoke_evidence is not None:
        raise ValueError("smoke evidence is accepted only by calibration reporting")
    reproducibility = _load_reproducibility_report(
        arguments.reproducibility_report,
        experiment=experiment,
    )
    trusted_artifact_root = _trusted_reproducibility_root(
        arguments.artifact_root,
        reproducibility,
    )
    arguments = argparse.Namespace(**vars(arguments))
    arguments.artifact_root = trusted_artifact_root
    protected_roots = (
        arguments.artifact_root,
        reproducibility.serial_root,
        reproducibility.parallel_root,
        *(
            ()
            if smoke_evidence is None
            else (
                smoke_evidence.reproducibility.serial_root,
                smoke_evidence.reproducibility.parallel_root,
            )
        ),
    )
    _preflight_output_paths(
        output_directories=(target,),
        protected_files=(
            arguments.config,
            arguments.plan,
            arguments.canary_plan,
            arguments.reproducibility_report,
            *((arguments.smoke_evidence,) if arguments.smoke_evidence else ()),
        ),
        protected_directories=protected_roots,
    )
    if target.exists():
        raise ValueError(f"study output directory must not already exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{secrets.token_hex(16)}.tmp"
    staging.mkdir(mode=0o700)
    staged_arguments = argparse.Namespace(**vars(arguments))
    staged_arguments.output_dir = staging
    staged_arguments._preloaded_experiment = experiment
    staged_arguments._study = study
    staged_arguments._preloaded_smoke_evidence = smoke_evidence
    staged_arguments._preloaded_reproducibility = reproducibility
    try:
        result = _build_study_report(staged_arguments)
        if target.exists():
            raise ValueError(f"refusing to overwrite immutable study output: {target}")
        os.rename(staging, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


def _build_study_report(arguments: argparse.Namespace) -> dict[str, object]:
    experiment = arguments._preloaded_experiment
    study = arguments._study
    smoke_evidence = arguments._preloaded_smoke_evidence
    reproducibility = arguments._preloaded_reproducibility
    protected_files = (
        arguments.config,
        arguments.plan,
        arguments.canary_plan,
        arguments.reproducibility_report,
        *((arguments.smoke_evidence,) if arguments.smoke_evidence is not None else ()),
    )
    smoke_roots = (
        ()
        if smoke_evidence is None
        else (
            smoke_evidence.reproducibility.serial_root,
            smoke_evidence.reproducibility.parallel_root,
        )
    )
    _preflight_output_paths(
        output_directories=(arguments.output_dir,),
        protected_files=protected_files,
        protected_directories=(arguments.artifact_root, *smoke_roots),
    )
    if experiment.phase == "confirmatory":
        current_provenance = collect_provenance()
        study.verify_confirmatory(
            experiment,
            analysis_code_hash=current_provenance.analysis_code_hash,
            dependency_lock_hash=current_provenance.dependency_lock_hash,
            environment_digest=current_provenance.environment_digest,
        )
    plan = load_analysis_plan(arguments.plan)
    if plan.expected_groups != study.expected_groups(experiment):
        raise ValueError("analysis plan inventory does not match the experiment config")
    if (
        experiment.confirmatory_freeze is not None
        and plan.registration_hash != experiment.confirmatory_freeze.analysis_version
    ):
        raise ValueError(
            "analysis plan does not match the config's frozen registration hash"
        )
    roots = _completed_run_roots(arguments.artifact_root, experiment.name)
    expected_runs = len(experiment.cells())
    if len(roots) != expected_runs:
        raise ValueError(f"expected {expected_runs} completed runs, found {len(roots)}")
    _preflight_output_paths(
        output_directories=(arguments.output_dir,),
        protected_files=protected_files,
        protected_directories=(
            arguments.artifact_root,
            reproducibility.serial_root,
            reproducibility.parallel_root,
            *smoke_roots,
        ),
    )
    serial_inventory = RawArtifactInventory.create(
        reproducibility.serial_root,
        experiment,
        side="serial",
    )
    parallel_inventory = RawArtifactInventory.create(
        reproducibility.parallel_root,
        experiment,
        side="parallel",
    )
    _validate_inventory_receipts(
        reproducibility,
        serial_inventory,
        parallel_inventory,
        smoke_prerequisite_hash=(
            None if smoke_evidence is None else smoke_evidence.scientific_hash
        ),
    )
    phase = AnalysisPhase(experiment.phase)
    expected_freeze = (
        None
        if experiment.confirmatory_freeze is None
        else experiment.confirmatory_freeze.seal_hash
    )
    expected_plan = study.build_analysis_plan(
        experiment,
        phase=phase,
        freeze_hash=expected_freeze,
    )
    if plan != expected_plan:
        raise ValueError(
            "analysis plan differs from the registered symbolic study plan"
        )
    expected_canary_plan = build_symbolic_canary_plan(
        experiment,
        phase=phase,
    )
    _load_registered_canary_plan(
        arguments.canary_plan,
        expected_canary_plan.to_dict(),
    )
    dataset = load_run_trees(
        roots,
        expected_phase=phase,
        expected_freeze_hash=expected_freeze,
        expected_run_settings=experiment.resolved_run_settings(),
    )
    current_provenance = tuple(sorted(collect_provenance().to_dict().items()))
    if dataset.provenance != current_provenance:
        raise ValueError(
            "reporting source or execution environment differs from run provenance"
        )
    current_reproducibility: dict[str, tuple[str, str]] = {}
    for observation in dataset.observations:
        if observation.cell_hash is None:
            raise ValueError("loaded run is missing its cell identity")
        value = observation.cell_hash, observation.content_hash
        previous = current_reproducibility.setdefault(observation.run_hash, value)
        if previous != value:
            raise ValueError("loaded checkpoints disagree on run identity")
    registered_reproducibility = {
        run["run_hash"]: (run["cell_hash"], run["scientific_content_hash"])
        for run in reproducibility.to_dict()["scientific"]["runs"]
    }
    if current_reproducibility != registered_reproducibility:
        raise ValueError(
            "reproducibility evidence does not match the analyzed artifact trees"
        )
    canaries = evaluate_canaries(dataset, expected_canary_plan)
    deviations = _deviation_log(
        experiment,
        arguments.deviation,
        study_contract=study.study_contract,
    )
    report = build_report(
        dataset,
        plan,
        canary_report_hash=canaries.scientific_hash,
        canaries_passed=canaries.passed,
        config_hash=experiment.config_hash,
    )
    report = dataclasses.replace(
        report,
        deviation_log_hash=deviations["scientific_hash"],
        deviation_count=len(deviations["deviations"]),
    )
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "experiment-config.json", experiment.resolved_dict())
    _write_immutable_text(output / "analysis-plan.json", analysis_plan_json(plan))
    _write_immutable_text(
        output / "canary-plan.json",
        expected_canary_plan.to_json(),
    )
    _write_immutable_text(output / "analysis.json", report.to_json())
    _write_immutable_text(output / "analysis.md", report.to_markdown())
    _write_immutable_text(output / "canaries.json", canaries.to_json())
    _write_json(output / "deviations.json", deviations)
    _write_json(output / "reproducibility.json", reproducibility.to_dict())
    _write_immutable_text(
        output / "raw-serial-inventory.json",
        serial_inventory.to_json(),
    )
    _write_immutable_text(
        output / "raw-parallel-inventory.json",
        parallel_inventory.to_json(),
    )
    if smoke_evidence is not None:
        _write_json(
            output / "smoke-prerequisite.json",
            smoke_evidence.to_dict(),
        )
    _write_immutable_text(
        output / "registered-gates.csv",
        registered_gates_csv(report),
    )
    _write_immutable_text(
        output / "canary-results.csv",
        canary_results_csv(canaries),
    )
    _write_immutable_text(
        output / "terminal-summary.csv",
        terminal_summary_csv(report),
    )
    _write_immutable_text(
        output / "trajectories.svg",
        trajectories_svg(report),
    )
    primary_decisions = {
        item.name: item.reject_null for item in report.family_decisions
    }
    equivalence_decisions = {
        item.name: item.reject_null for item in report.equivalence_decisions
    }
    summary: dict[str, object] = {
        "phase": phase.value,
        "config_hash": experiment.config_hash,
        "analysis_registration_hash": plan.registration_hash,
        "analysis_plan_hash": plan.scientific_hash,
        "freeze_hash": expected_freeze,
        "run_settings_hash": report.run_settings_hash,
        "analysis_code_hash": dict(report.provenance).get("analysis_code_hash"),
        "dependency_lock_hash": dict(report.provenance).get("dependency_lock_hash"),
        "environment_digest": dict(report.provenance).get("environment_digest"),
        "run_count": len(roots),
        "dataset_hash": dataset.scientific_hash,
        "analysis_hash": report.scientific_hash,
        "canary_plan_hash": expected_canary_plan.scientific_hash,
        "canary_hash": canaries.scientific_hash,
        "reproducibility_hash": reproducibility.scientific_hash,
        "reproducibility_passed": True,
        "deviation_log_hash": deviations["scientific_hash"],
        "deviation_count": len(deviations["deviations"]),
        "raw_serial_inventory_hash": serial_inventory.scientific_hash,
        "raw_parallel_inventory_hash": parallel_inventory.scientific_hash,
        "canaries_passed": canaries.passed,
        "interpretation_eligible": report.interpretation_eligible,
    }
    if smoke_evidence is not None:
        summary.update(
            {
                "smoke_prerequisite_passed": True,
                "smoke_prerequisite_hash": smoke_evidence.scientific_hash,
                "smoke_config_hash": smoke_evidence.config.config_hash,
                "smoke_reproducibility_hash": (
                    smoke_evidence.reproducibility.scientific_hash
                ),
                "smoke_raw_serial_inventory_hash": (
                    smoke_evidence.serial_inventory.scientific_hash
                ),
                "smoke_raw_parallel_inventory_hash": (
                    smoke_evidence.parallel_inventory.scientific_hash
                ),
                "smoke_engineering_anomalies": list(
                    smoke_evidence.engineering_anomalies
                ),
            }
        )
    if phase is AnalysisPhase.CALIBRATION:
        summary.update(
            {
                "descriptive_primary_rejections": sorted(
                    name for name, rejected in primary_decisions.items() if rejected
                ),
                "descriptive_equivalence_rejections": sorted(
                    name for name, rejected in equivalence_decisions.items() if rejected
                ),
                "calibration_tests_are_confirmatory": False,
            }
        )
    else:
        summary.update(
            {
                "primary_family_passed": all(
                    primary_decisions[result.name] for result in report.contrasts
                ),
                "equivalence_gates_passed": all(
                    equivalence_decisions[result.name] for result in report.equivalences
                ),
                "registered_family_passed": report.registered_family_passed,
            }
        )
    if experiment.confirmatory_freeze is not None:
        summary.update(
            {
                "calibration_evidence_hash": (
                    experiment.confirmatory_freeze.calibration_evidence_hash
                ),
                "seed_banks": experiment.confirmatory_freeze.seed_banks.to_dict(),
                "frozen_tolerances": {
                    item.name: item.value
                    for item in experiment.confirmatory_freeze.tolerances
                },
                "frozen_margins": {
                    item.name: item.value
                    for item in experiment.confirmatory_freeze.margins
                },
            }
        )
    power_evidence: dict[str, object] | None = None
    if phase is AnalysisPhase.CALIBRATION and canaries.passed:
        assert smoke_evidence is not None
        hypotheses = study.power.hypotheses(report)
        equivalence_hypotheses = study.power.equivalence_hypotheses(report)
        power = calibrate_environment_count(
            hypotheses,
            candidate_environment_counts=study.power.candidate_environment_counts,
            equivalence_hypotheses=equivalence_hypotheses,
            simulations=arguments.power_simulations,
            seed=study.power.seed,
            rng_stream=study.power.rng_stream,
            alpha=study.power.alpha,
            minimum_power=study.power.minimum_individual_power,
            minimum_equivalence_power=study.power.minimum_equivalence_power,
            minimum_joint_power=study.power.minimum_joint_power,
            maximum_fwer=study.power.maximum_global_null_fwer,
            simulation_error_alpha=study.power.simulation_error_alpha,
            design_confidence_alpha=study.power.design_confidence_alpha,
            center_environment_count=study.power.center_environment_count,
        )
        power_payload = {
            "artifact_type": "symbolic-power-calibration",
            "schema_version": 2,
            "hypotheses": hypotheses,
            "equivalence_hypotheses": equivalence_hypotheses,
            "calibration": power,
        }
        plain_power = _jsonable(power_payload)
        plain_power["scientific_hash"] = scientific_hash(
            plain_power,
            domain="symbolic-power-calibration",
        )
        power_evidence = plain_power
        _write_json(output / "power.json", plain_power)
        _write_immutable_text(
            output / "power-calibration.csv",
            power_calibration_csv(
                report,
                power,
                hypotheses,
                equivalence_hypotheses=equivalence_hypotheses,
                calibration_hash=plain_power["scientific_hash"],
            ),
        )
        evidence = study.calibration_evidence_hash(
            config=experiment,
            report=report,
            canary_report_hash=canaries.scientific_hash,
            power_calibration_hash=plain_power["scientific_hash"],
            reproducibility_report_hash=reproducibility.scientific_hash,
            raw_serial_inventory_hash=serial_inventory.scientific_hash,
            raw_parallel_inventory_hash=parallel_inventory.scientific_hash,
            deviation_log_hash=deviations["scientific_hash"],
            smoke_prerequisite_hash=smoke_evidence.scientific_hash,
            smoke_config_hash=smoke_evidence.config.config_hash,
            smoke_reproducibility_hash=(smoke_evidence.reproducibility.scientific_hash),
            smoke_raw_serial_inventory_hash=(
                smoke_evidence.serial_inventory.scientific_hash
            ),
            smoke_raw_parallel_inventory_hash=(
                smoke_evidence.parallel_inventory.scientific_hash
            ),
        )
        summary.update(
            {
                "power_hash": plain_power["scientific_hash"],
                "calibration_evidence_hash": evidence,
                "selected_environment_replicas": (power.selected_environment_count),
            }
        )
    elif phase is AnalysisPhase.CALIBRATION:
        blocked_power = {
            "artifact_type": "symbolic-power-calibration",
            "schema_version": 2,
            "status": "blocked-by-failed-canaries",
        }
        blocked_power["scientific_hash"] = scientific_hash(
            blocked_power,
            domain="symbolic-power-calibration",
        )
        power_evidence = blocked_power
        _write_json(output / "power.json", blocked_power)
        _write_immutable_text(
            output / "power-calibration.csv",
            "status\nblocked-by-failed-canaries\n",
        )
        summary["power_hash"] = blocked_power["scientific_hash"]
    if phase is AnalysisPhase.CALIBRATION:
        summary.update(
            study.power.summary_fields(
                include_rng_stream=study.records_power_rng_stream
            )
        )
        summary["power_simulations"] = arguments.power_simulations
        summary["freeze_eligible"] = (
            canaries.passed
            and summary["reproducibility_passed"] is True
            and summary.get("smoke_prerequisite_passed") is True
            and summary.get("selected_environment_replicas") is not None
            and arguments.power_simulations == study.power.simulations
            and summary["deviation_count"] == 0
        )
    summary["scientific_hash"] = scientific_hash(
        summary,
        domain="symbolic-study-summary",
    )
    _write_json(output / "summary.json", summary)
    _verify_report_json_outputs(
        output,
        experiment=experiment,
        plan=plan,
        canary_plan=expected_canary_plan,
        report=report,
        canaries=canaries,
        deviations=deviations,
        reproducibility=reproducibility,
        serial_inventory=serial_inventory,
        parallel_inventory=parallel_inventory,
        smoke_evidence=smoke_evidence,
        power=power_evidence,
        summary=summary,
    )
    member_names = _expected_release_members(phase)
    observed_names = {path.name for path in output.iterdir()}
    allowed_names = {*member_names, STUDY_RELEASE_MANIFEST_FILENAME}
    if observed_names - allowed_names or not set(member_names) <= observed_names:
        raise ValueError("study output directory has an incomplete or extra inventory")
    evidence_hash = summary.get("calibration_evidence_hash")
    if evidence_hash is not None and not is_sha256(evidence_hash):
        raise ValueError("study summary calibration evidence hash is invalid")
    release = StudyReleaseManifest.create(
        output,
        member_names,
        phase=phase.value,
        study_contract=study.study_contract,
        config_hash=experiment.config_hash,
        freeze_hash=expected_freeze,
        calibration_evidence_hash=evidence_hash,
    )
    _write_json(
        output / STUDY_RELEASE_MANIFEST_FILENAME,
        release.to_dict(),
    )
    if load_study_release_manifest(output / STUDY_RELEASE_MANIFEST_FILENAME) != release:
        raise ValueError("persisted release manifest differs from its source")
    return {
        **summary,
        "release_manifest_hash": release.scientific_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        artifacts = validate_artifact_tree(arguments.path)
        print(json.dumps({"artifacts": len(artifacts), "valid": True}))
        return 0
    if arguments.command == "smoke-evidence":
        _preflight_output_paths(
            output_files=(arguments.output,),
            protected_files=(
                arguments.config,
                arguments.reproducibility_report,
                arguments.serial_inventory,
                arguments.parallel_inventory,
            ),
        )
        if arguments.output.exists():
            raise ValueError(
                f"smoke evidence output must not already exist: {arguments.output}"
            )
        experiment = load_experiment_config(arguments.config)
        reproducibility = _load_reproducibility_report(
            arguments.reproducibility_report,
            experiment=experiment,
        )
        serial_inventory = RawArtifactInventory.from_dict(
            _read_json_object(
                arguments.serial_inventory,
                label="smoke serial raw inventory",
            )
        )
        parallel_inventory = RawArtifactInventory.from_dict(
            _read_json_object(
                arguments.parallel_inventory,
                label="smoke parallel raw inventory",
            )
        )
        evidence = SmokePrerequisiteEvidence.create(
            experiment,
            reproducibility,
            serial_inventory,
            parallel_inventory,
            engineering_anomalies=tuple(sorted(set(arguments.anomaly))),
        )
        _preflight_output_paths(
            output_files=(arguments.output,),
            protected_files=(
                arguments.config,
                arguments.reproducibility_report,
                arguments.serial_inventory,
                arguments.parallel_inventory,
            ),
            protected_directories=(
                reproducibility.serial_root,
                reproducibility.parallel_root,
            ),
        )
        _write_json(
            arguments.output,
            evidence.to_dict(),
            verifier=lambda candidate: _require_equal(
                _load_smoke_prerequisite(candidate),
                evidence,
                label="persisted smoke prerequisite",
            ),
        )
        print(
            json.dumps(
                {
                    "path": str(arguments.output),
                    "scientific_hash": evidence.scientific_hash,
                    "engineering_anomalies": list(evidence.engineering_anomalies),
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "inventory":
        _preflight_output_paths(
            output_files=(arguments.output,),
            protected_files=(arguments.config,),
            protected_directories=(arguments.artifact_root,),
        )
        if arguments.output.exists():
            raise ValueError(
                f"raw inventory output must not already exist: {arguments.output}"
            )
        experiment = load_experiment_config(arguments.config)
        inventory = RawArtifactInventory.create(
            arguments.artifact_root,
            experiment,
            side=arguments.side,
        )
        _write_immutable_text(
            arguments.output,
            inventory.to_json(),
            verifier=lambda candidate: _require_equal(
                RawArtifactInventory.from_dict(
                    _read_json_object(
                        candidate,
                        label="persisted raw artifact inventory",
                    )
                ),
                inventory,
                label="persisted raw inventory",
            ),
        )
        print(
            json.dumps(
                {
                    "path": str(arguments.output),
                    "scientific_hash": inventory.scientific_hash,
                    "trees": len(inventory.trees),
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-inventory":
        experiment = load_experiment_config(arguments.config)
        inventory = RawArtifactInventory.from_dict(
            _read_json_object(
                arguments.inventory,
                label="raw artifact inventory",
            )
        )
        inventory.verify(
            arguments.artifact_root,
            experiment,
            side=arguments.side,
        )
        print(
            json.dumps(
                {
                    "scientific_hash": inventory.scientific_hash,
                    "trees": len(inventory.trees),
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "plan":
        _preflight_output_paths(
            output_files=(arguments.output, arguments.canary_output),
            protected_files=(arguments.config,),
        )
        experiment = load_experiment_config(arguments.config)
        phase = AnalysisPhase(experiment.phase)
        plan = build_symbolic_analysis_plan(
            experiment,
            phase=phase,
            freeze_hash=(
                None
                if experiment.confirmatory_freeze is None
                else experiment.confirmatory_freeze.seal_hash
            ),
        )
        if (
            experiment.confirmatory_freeze is not None
            and plan.registration_hash
            != experiment.confirmatory_freeze.analysis_version
        ):
            raise ValueError(
                "confirmatory config is bound to a different analysis registration"
            )
        canary_plan = build_symbolic_canary_plan(experiment, phase=phase)
        _write_immutable_text(
            arguments.output,
            analysis_plan_json(plan),
            verifier=lambda candidate: _require_equal(
                load_analysis_plan(candidate),
                plan,
                label="persisted analysis plan",
            ),
        )
        _write_immutable_text(
            arguments.canary_output,
            canary_plan.to_json(),
            verifier=lambda candidate: _load_registered_canary_plan(
                candidate,
                canary_plan.to_dict(),
            ),
        )
        print(
            json.dumps(
                {
                    "path": str(arguments.output),
                    "canary_path": str(arguments.canary_output),
                    "registration_hash": plan.registration_hash,
                    "scientific_hash": plan.scientific_hash,
                    "canary_scientific_hash": canary_plan.scientific_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "report":
        return _write_study_report(arguments)
    if arguments.command == "freeze":
        return _freeze_study(arguments)
    if arguments.command == "reproduce":
        _preflight_output_paths(
            output_files=(arguments.output,),
            output_directories=(
                arguments.serial_root,
                arguments.parallel_root,
            ),
            protected_files=(
                arguments.config,
                *((arguments.smoke_evidence,) if arguments.smoke_evidence else ()),
            ),
        )
        experiment = load_experiment_config(arguments.config)
        study = (
            registered_symbolic_study(experiment.name)
            if experiment.phase == "calibration"
            else None
        )
        smoke_evidence = _smoke_prerequisite_for_execution(
            experiment,
            arguments.smoke_evidence,
            study=study,
        )
        extra_protected_files = (
            () if arguments.smoke_evidence is None else (arguments.smoke_evidence,)
        )
        extra_protected_directories = (
            ()
            if smoke_evidence is None
            else (
                smoke_evidence.reproducibility.serial_root,
                smoke_evidence.reproducibility.parallel_root,
            )
        )
        _preflight_output_paths(
            output_files=(arguments.output,),
            output_directories=(
                arguments.serial_root,
                arguments.parallel_root,
            ),
            protected_files=(arguments.config, *extra_protected_files),
            protected_directories=extra_protected_directories,
        )
        study = _verify_registered_execution(experiment)
        if os.path.lexists(arguments.output):
            if not arguments.resume:
                raise ValueError(
                    "fresh reproducibility report output must not already exist"
                )
            ReproducibilityReport.from_dict(
                _read_json_object(
                    arguments.output,
                    label="existing reproducibility report",
                ),
                experiment=experiment,
                expected_serial_root=arguments.serial_root,
                expected_parallel_root=arguments.parallel_root,
                expected_parallel_workers=arguments.workers,
            )
        report = run_reproducibility_check(
            experiment,
            serial_root=arguments.serial_root,
            parallel_root=arguments.parallel_root,
            parallel_workers=arguments.workers,
            adapter_factory=(
                execution_adapter_factory(experiment.name)
                if study is None
                else study.adapter_factory
            ),
            resume=arguments.resume,
            smoke_prerequisite_hash=(
                None if smoke_evidence is None else smoke_evidence.scientific_hash
            ),
        )
        _write_json(
            arguments.output,
            report.to_dict(),
            verifier=lambda candidate: _require_equal(
                ReproducibilityReport.from_dict(
                    _read_json_object(
                        candidate,
                        label="persisted reproducibility report",
                    ),
                    expected_serial_root=arguments.serial_root,
                    expected_parallel_root=arguments.parallel_root,
                    expected_parallel_workers=arguments.workers,
                ),
                report,
                label="persisted reproducibility report",
            ),
        )
        print(
            json.dumps(
                {
                    "path": str(arguments.output),
                    "config_hash": experiment.config_hash,
                    "run_count": len(report.runs),
                    "scientific_hash": report.scientific_hash,
                    "exact_match": True,
                },
                sort_keys=True,
            )
        )
        return 0
    experiment = load_experiment_config(arguments.config)
    if arguments.command == "pilot" and experiment.phase != "pilot":
        raise ValueError("the pilot command accepts only phase='pilot' configs")
    study = (
        registered_symbolic_study(experiment.name)
        if experiment.phase == "calibration"
        else None
    )
    smoke_evidence = _smoke_prerequisite_for_execution(
        experiment,
        arguments.smoke_evidence,
        study=study,
    )
    study = _verify_registered_execution(experiment)
    _preflight_output_paths(
        output_directories=(arguments.artifact_root,),
        protected_files=(
            arguments.config,
            *((arguments.smoke_evidence,) if arguments.smoke_evidence else ()),
        ),
        protected_directories=(
            ()
            if smoke_evidence is None
            else (
                smoke_evidence.reproducibility.serial_root,
                smoke_evidence.reproducibility.parallel_root,
            )
        ),
    )
    executor = RunExecutor(
        arguments.artifact_root,
        (
            execution_adapter_factory(experiment.name)
            if study is None
            else study.adapter_factory
        ),
    )
    results = SweepRunner(executor).run(
        experiment,
        max_workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "phase": experiment.phase,
                "config_hash": experiment.config_hash,
                "confirmatory_frozen": experiment.confirmatory_frozen,
                "confirmatory_freeze_hash": (
                    None
                    if experiment.confirmatory_freeze is None
                    else experiment.confirmatory_freeze.seal_hash
                ),
                "analysis_registration_hash": (
                    None
                    if experiment.confirmatory_freeze is None
                    else experiment.confirmatory_freeze.analysis_version
                ),
                "runs": [
                    {
                        "run_hash": result.run_hash,
                        "complete": result.complete,
                        "scientific_content_hash": result.scientific_content_hash,
                        "path": str(result.path),
                    }
                    for result in results
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
