from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.cli import main
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.orchestration.inventory import RawArtifactInventory
from infinite_rulebook.orchestration.reproducibility import (
    ReproducibilityRun,
    run_reproducibility_check,
)
from infinite_rulebook.studies.smoke_prerequisite import (
    SmokePrerequisiteEvidence,
)
from infinite_rulebook.studies.symbolic_construct import (
    verify_symbolic_smoke_design,
)


def _tiny_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke-prerequisite-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="smoke-prerequisite-test",
    )


def _evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SmokePrerequisiteEvidence:
    import infinite_rulebook.studies.smoke_prerequisite as smoke_module

    monkeypatch.setattr(
        smoke_module,
        "verify_symbolic_smoke_design",
        lambda _config: None,
    )
    config = _tiny_experiment()
    report = run_reproducibility_check(
        config,
        serial_root=tmp_path / "serial",
        parallel_root=tmp_path / "parallel",
        parallel_workers=2,
    )
    serial = RawArtifactInventory.create(report.serial_root, config, side="serial")
    parallel = RawArtifactInventory.create(
        report.parallel_root,
        config,
        side="parallel",
    )
    return SmokePrerequisiteEvidence.create(
        config,
        report,
        serial,
        parallel,
        engineering_anomalies=("non-invalidating note",),
    )


def test_smoke_prerequisite_is_portable_and_reauthenticates_relocated_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    raw = evidence.to_dict()
    relocated_serial = tmp_path / "relocated-serial"
    relocated_parallel = tmp_path / "relocated-parallel"
    shutil.copytree(evidence.reproducibility.serial_root, relocated_serial)
    shutil.copytree(evidence.reproducibility.parallel_root, relocated_parallel)
    raw["operational"]["serial"]["artifact_root"] = str(relocated_serial)
    raw["operational"]["parallel"]["artifact_root"] = str(relocated_parallel)

    relocated = SmokePrerequisiteEvidence.from_dict(raw)

    assert relocated.scientific_hash == evidence.scientific_hash
    assert relocated.engineering_anomalies == ("non-invalidating note",)


def test_smoke_prerequisite_rejects_a_fabricated_reproducibility_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    run = evidence.reproducibility.runs[0]
    fabricated = replace(
        evidence.reproducibility,
        runs=(
            ReproducibilityRun(
                cell_hash=run.cell_hash,
                run_hash=run.run_hash,
                scientific_content_hash="0" * 64,
            ),
        ),
    )

    with pytest.raises(ValueError, match="authenticated roots"):
        SmokePrerequisiteEvidence.create(
            evidence.config,
            fabricated,
            evidence.serial_inventory,
            evidence.parallel_inventory,
        )


def test_smoke_prerequisite_schema_and_anomalies_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    raw = copy.deepcopy(evidence.to_dict())
    raw["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        SmokePrerequisiteEvidence.from_dict(raw)

    with pytest.raises(ValueError, match="sorted unique"):
        SmokePrerequisiteEvidence.create(
            evidence.config,
            evidence.reproducibility,
            evidence.serial_inventory,
            evidence.parallel_inventory,
            engineering_anomalies=("second", "first"),
        )


@pytest.mark.parametrize("command", ("run", "reproduce"))
def test_registered_calibration_refuses_to_start_without_smoke_evidence(
    tmp_path: Path,
    command: str,
) -> None:
    config = Path("configs/symbolic-calibration-v1.json")
    if command == "run":
        arguments = [
            "run",
            str(config),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    else:
        arguments = [
            "reproduce",
            str(config),
            str(tmp_path / "serial"),
            str(tmp_path / "parallel"),
            str(tmp_path / "reproducibility.json"),
        ]

    with pytest.raises(ValueError, match="requires Stage-0 smoke evidence"):
        main(arguments)
    assert not tuple(tmp_path.iterdir())


def test_registered_smoke_config_is_exactly_pinned() -> None:
    config = load_experiment_config("configs/pilot-smoke.json")
    verify_symbolic_smoke_design(config)

    with pytest.raises(ValueError, match="Stage-0 design"):
        verify_symbolic_smoke_design(replace(config, name="other-smoke"))
