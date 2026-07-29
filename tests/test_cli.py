from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infinite_rulebook.analysis import AnalysisPhase, load_analysis_plan
from infinite_rulebook.cli import (
    _completed_run_roots,
    _deviation_log,
    _expected_release_members,
    _parser,
    _reconstruct_registered_power,
    _trusted_reproducibility_root,
    _validate_inventory_receipts,
    main,
)
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
    run_reproducibility_check,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter
from infinite_rulebook.studies.symbolic_construct import (
    PRIMARY_MINIMUM_EFFECTS,
    S5_REWARD_EQUIVALENCE_MARGIN,
    build_symbolic_analysis_plan,
    build_symbolic_canary_plan,
)


def test_trusted_report_root_is_stable_after_alias_retarget(
    tmp_path: Path,
) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    replacement = tmp_path / "replacement"
    for path in (serial, parallel, replacement):
        path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parallel, target_is_directory=True)
    reproducibility = SimpleNamespace(
        serial_root=serial,
        parallel_root=parallel,
    )

    trusted = _trusted_reproducibility_root(alias, reproducibility)
    alias.unlink()
    alias.symlink_to(replacement, target_is_directory=True)

    assert trusted == parallel.resolve()
    assert trusted != alias.resolve()


def test_run_command_accepts_phase_aware_arguments(tmp_path: Path) -> None:
    arguments = _parser().parse_args(
        [
            "run",
            "configs/symbolic-calibration-v1.json",
            "--artifact-root",
            str(tmp_path),
            "--workers",
            "3",
        ]
    )

    assert arguments.command == "run"
    assert arguments.workers == 3
    assert arguments.artifact_root == tmp_path


def test_legacy_pilot_alias_rejects_calibration_config() -> None:
    with pytest.raises(ValueError, match="phase='pilot'"):
        main(["pilot", "configs/symbolic-calibration-v1.json"])


def test_plan_command_writes_and_authenticates_analysis_and_canary_plans(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "analysis.json"
    canary_path = tmp_path / "canaries.json"
    arguments = [
        "plan",
        "configs/symbolic-calibration-v1.json",
        str(analysis_path),
        str(canary_path),
    ]

    assert main(arguments) == 0
    config = load_experiment_config("configs/symbolic-calibration-v1.json")
    assert load_analysis_plan(analysis_path) == build_symbolic_analysis_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    assert json.loads(canary_path.read_text(encoding="utf-8")) == (
        build_symbolic_canary_plan(
            config,
            phase=AnalysisPhase.CALIBRATION,
        ).to_dict()
    )
    assert main(arguments) == 0

    canary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        main(arguments)


def test_cli_lifecycle_requires_explicit_reproducibility_and_canary_inputs(
    tmp_path: Path,
) -> None:
    report = _parser().parse_args(
        [
            "report",
            "config.json",
            "analysis.json",
            "canaries.json",
            "parallel-artifacts",
            "results",
            "--reproducibility-report",
            "reproducibility.json",
        ]
    )
    freeze = _parser().parse_args(
        [
            "freeze",
            "calibration.json",
            "summary.json",
            "confirmatory.json",
            "confirmatory-analysis.json",
            "confirmatory-canaries.json",
        ]
    )
    reproduce = _parser().parse_args(
        [
            "reproduce",
            "config.json",
            "serial",
            "parallel",
            str(tmp_path / "reproducibility.json"),
        ]
    )

    assert report.canary_plan == Path("canaries.json")
    assert report.reproducibility_report == Path("reproducibility.json")
    assert freeze.output_canary_plan == Path("confirmatory-canaries.json")
    assert reproduce.serial_root == Path("serial")
    assert reproduce.parallel_root == Path("parallel")
    assert reproduce.resume is False
    assert (
        _parser()
        .parse_args(
            [
                "reproduce",
                "config.json",
                "serial",
                "parallel",
                "reproducibility.json",
                "--resume",
            ]
        )
        .resume
        is True
    )


def _analysis_power_payload(environment_count: int = 2) -> dict[str, object]:
    contrasts = [
        {
            "name": name,
            "pair_count": environment_count,
            "cell_pair_count": environment_count * 3,
            "differences": [effect, effect + 0.1],
        }
        for name, effect in PRIMARY_MINIMUM_EFFECTS.items()
    ]
    return {
        "contrasts": contrasts,
        "equivalences": [
            {
                "name": "ind-red-terminal-hidden-reward-equivalence",
                "margin": S5_REWARD_EQUIVALENCE_MARGIN,
                "pair_count": environment_count,
                "cell_pair_count": environment_count * 3,
                "differences": [0.0] * environment_count,
            }
        ],
    }


def test_freeze_power_reconstruction_uses_exact_analysis_cluster_inventory() -> None:
    directional, equivalences = _reconstruct_registered_power(
        _analysis_power_payload(),
        2,
        3,
    )

    assert {item.name for item in directional} == set(PRIMARY_MINIMUM_EFFECTS)
    assert len(equivalences) == 1
    assert equivalences[0].margin == S5_REWARD_EQUIVALENCE_MARGIN
    assert all(item.algorithm_replicas == 3 for item in (*directional, *equivalences))

    tampered = _analysis_power_payload()
    tampered["contrasts"][0]["differences"] = [0.25]
    with pytest.raises(ValueError, match="cluster inventory"):
        _reconstruct_registered_power(tampered, 2, 3)


def test_release_package_inventory_is_fixed_by_phase() -> None:
    calibration = _expected_release_members(AnalysisPhase.CALIBRATION)
    confirmatory = _expected_release_members(AnalysisPhase.CONFIRMATORY)

    assert {"power.json", "power-calibration.csv"} <= set(calibration)
    assert "deviations.json" in confirmatory
    assert set(calibration) - set(confirmatory) == {
        "power.json",
        "power-calibration.csv",
        "smoke-prerequisite.json",
    }


def test_deviation_log_is_explicit_deterministic_and_strict() -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")
    first = _deviation_log(config, ["second", "first"])
    second = _deviation_log(config, ["first", "second"])

    assert first == second
    assert first["deviations"] == ["first", "second"]
    with pytest.raises(ValueError, match="unique"):
        _deviation_log(config, ["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="nonempty"):
        _deviation_log(config, [" padded "])


def test_cli_creates_and_verifies_portable_raw_inventory(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="cli-inventory-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="cli-inventory-seed",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), sort_keys=True),
        encoding="utf-8",
    )
    root = tmp_path / "raw"
    RunExecutor(root, ExactSymbolicAdapter).execute(config, config.cells()[0])
    output = tmp_path / "inventory.json"

    assert (
        main(
            [
                "inventory",
                str(config_path),
                str(root),
                "serial",
                str(output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "verify-inventory",
                str(config_path),
                str(root),
                str(output),
                "--side",
                "serial",
            ]
        )
        == 0
    )


def test_reproduce_rejects_preexisting_report_output_before_execution(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="cli-reproduce-output-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="cli-reproduce-output-seed",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "reproducibility.json"
    output.write_text("{}\n", encoding="utf-8")
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    with pytest.raises(ValueError, match="must not already exist"):
        main(
            [
                "reproduce",
                str(config_path),
                str(serial),
                str(parallel),
                str(output),
            ]
        )
    assert not serial.exists()
    assert not parallel.exists()
    with pytest.raises(ValueError, match="reproducibility report"):
        main(
            [
                "reproduce",
                str(config_path),
                str(serial),
                str(parallel),
                str(output),
                "--resume",
            ]
        )
    assert not serial.exists()
    assert not parallel.exists()


def test_reproduce_cli_reuses_only_its_matching_completed_report_on_resume(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="cli-reproduce-resume-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="cli-reproduce-resume-seed",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), sort_keys=True),
        encoding="utf-8",
    )
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    output = tmp_path / "reproducibility.json"
    arguments = [
        "reproduce",
        str(config_path),
        str(serial),
        str(parallel),
        str(output),
        "--workers",
        "2",
    ]

    assert main(arguments) == 0
    before = output.read_bytes()
    assert main([*arguments, "--resume"]) == 0
    assert output.read_bytes() == before


def test_inventory_receipts_must_match_the_exact_report_pair(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="cli-receipt-pair-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="cli-receipt-pair-seed",
    )
    first = run_reproducibility_check(
        config,
        serial_root=tmp_path / "first-serial",
        parallel_root=tmp_path / "first-parallel",
        parallel_workers=2,
    )
    run_reproducibility_check(
        config,
        serial_root=tmp_path / "second-serial",
        parallel_root=tmp_path / "second-parallel",
        parallel_workers=2,
    )
    serial_inventory = RawArtifactInventory.create(
        tmp_path / "first-serial",
        config,
        side="serial",
    )
    wrong_parallel_inventory = RawArtifactInventory.create(
        tmp_path / "second-parallel",
        config,
        side="parallel",
    )

    with pytest.raises(ValueError, match="receipt pair"):
        _validate_inventory_receipts(
            first,
            serial_inventory,
            wrong_parallel_inventory,
        )


def test_lifecycle_rejects_output_input_overlap_before_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output files must not overlap"):
        main(
            [
                "plan",
                "missing-config.json",
                str(tmp_path / "registered"),
                str(tmp_path / "registered" / "canaries.json"),
            ]
        )
    assert not (tmp_path / "registered").exists()

    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    with pytest.raises(ValueError, match="inside output directories"):
        main(
            [
                "reproduce",
                "missing-config.json",
                str(serial),
                str(parallel),
                str(serial / "reproducibility.json"),
            ]
        )
    assert not serial.exists()
    assert not parallel.exists()

    report_parent = tmp_path / "job"
    with pytest.raises(ValueError, match="or contain them"):
        main(
            [
                "reproduce",
                "missing-config.json",
                str(report_parent / "serial"),
                str(tmp_path / "other-parallel"),
                str(report_parent),
            ]
        )
    assert not report_parent.exists()

    artifact_root = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="overlaps"):
        main(
            [
                "report",
                "config.json",
                "analysis.json",
                "canaries.json",
                str(artifact_root),
                str(artifact_root / "results"),
                "--reproducibility-report",
                "reproducibility.json",
            ]
        )
    assert not artifact_root.exists()

    evidence = tmp_path / "evidence"
    with pytest.raises(ValueError, match="overlaps"):
        main(
            [
                "freeze",
                "calibration.json",
                str(evidence / "summary.json"),
                str(evidence / "confirmatory.json"),
                str(tmp_path / "analysis.json"),
                str(tmp_path / "canaries.json"),
            ]
        )
    assert not evidence.exists()


def test_completed_run_inventory_rejects_unexpected_entries(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "experiment"
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("not a run", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid entry"):
        _completed_run_roots(tmp_path / "artifacts", "experiment")


def test_report_rejects_output_inside_declared_reproducibility_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/symbolic-calibration-v1.json")
    config = load_experiment_config(config_path)
    analysis_path = tmp_path / "analysis.json"
    canary_path = tmp_path / "canaries.json"
    assert (
        main(
            [
                "plan",
                str(config_path),
                str(analysis_path),
                str(canary_path),
            ]
        )
        == 0
    )
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    monkeypatch.setattr(
        "infinite_rulebook.cli._completed_run_roots",
        lambda *_: tuple(
            tmp_path / f"run-{index}" for index in range(len(config.cells()))
        ),
    )
    monkeypatch.setattr(
        "infinite_rulebook.cli._load_reproducibility_report",
        lambda *_args, **_kwargs: SimpleNamespace(
            serial_root=serial,
            parallel_root=parallel,
        ),
    )
    monkeypatch.setattr(
        "infinite_rulebook.cli._load_smoke_prerequisite",
        lambda *_args, **_kwargs: SimpleNamespace(
            reproducibility=SimpleNamespace(
                serial_root=tmp_path / "smoke-serial",
                parallel_root=tmp_path / "smoke-parallel",
            )
        ),
    )

    with pytest.raises(ValueError, match="overlaps"):
        main(
            [
                "report",
                str(config_path),
                str(analysis_path),
                str(canary_path),
                str(parallel),
                str(serial / "report"),
                "--reproducibility-report",
                str(tmp_path / "reproducibility.json"),
                "--smoke-evidence",
                str(tmp_path / "smoke.json"),
            ]
        )
    assert not serial.exists()
