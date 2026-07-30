from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    FeedbackConfig,
    SolverConfig,
    experiment_config_from_dict,
)
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    SeedBankIdentities,
    SeedBankIdentity,
    confirmatory_config_hash,
    confirmatory_freeze_from_dict,
    freeze_experiment_config,
    load_confirmatory_freeze,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.provenance import collect_provenance
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def calibration_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="symbolic-calibration-v1",
        environments=(
            EnvironmentConfig(EnvironmentKind.IND, projection_size=2),
            EnvironmentConfig(
                EnvironmentKind.RED_C,
                projection_size=2,
                max_redundant_support=2,
            ),
        ),
        agents=(
            AgentConfig(AgentKind.REWARD, target_size=2),
            AgentConfig(AgentKind.RELEVANT_INFORMATION, target_size=2),
            AgentConfig(
                AgentKind.SCHEDULED,
                target_size=1,
                growth_step=1,
                growth_interval=2,
                maximum_size=4,
            ),
        ),
        checkpoints=CheckpointConfig((0, 2, 4)),
        horizon=4,
        master_seed="calibration-seed-v1",
        algorithm_master_seed="algorithm-seed-v1",
        environment_replicas=2,
        algorithm_replicas=2,
        phase="calibration",
    )


def evidence_hash() -> str:
    return scientific_hash(
        {
            "artifact_manifest_hashes": ("a" * 64, "b" * 64),
            "calibration_contract": "symbolic-pilot-analysis.v1",
        },
        domain="calibration-evidence",
    )


def analysis_hash() -> str:
    return scientific_hash(
        "symbolic-confirmatory-analysis.v1",
        domain="analysis-registration",
    )


def sealed_config() -> ExperimentConfig:
    calibration = calibration_config()
    banks = SeedBankIdentities.bind(
        calibration_master_seed=calibration.master_seed,
        confirmatory_master_seed="confirmatory-seed-v1",
        algorithm_master_seed=calibration.algorithm_master_seed,
    )
    return freeze_experiment_config(
        calibration,
        name="symbolic-confirmatory-v1",
        confirmatory_master_seed="confirmatory-seed-v1",
        calibration_evidence_hash=evidence_hash(),
        analysis_contract="symbolic-confirmatory-analysis",
        analysis_version=analysis_hash(),
        analysis_code_hash=analysis_hash(),
        dependency_lock_hash=analysis_hash(),
        environment_digest=analysis_hash(),
        seed_banks=banks,
        tolerances={
            "frontier_certificate_gap_nats": 1e-7,
            "parallel_hash_mismatches": 0.0,
        },
        margins={
            "distractor_leakage_nats": 0.02,
            "late_slope_equivalence": 0.1,
        },
    )


def test_confirmatory_freeze_round_trip_is_deterministic(tmp_path: Path) -> None:
    config = sealed_config()
    record = config.confirmatory_freeze
    assert record is not None
    assert record.confirmatory_frozen
    assert config.confirmatory_frozen
    assert record.seed_banks.algorithm == SeedBankIdentity.bind(
        config.algorithm_master_seed,
        namespace="algorithm.v1",
    )
    assert record.config_hash == confirmatory_config_hash(config)
    assert config.resolved_run_settings()["confirmatory_frozen"] is True
    assert config.resolved_run_settings()["confirmatory_freeze_hash"] == (
        record.seal_hash
    )
    assert (
        config.resolved_run_settings()["analysis_registration_hash"] == analysis_hash()
    )

    raw = config.resolved_dict()
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    assert "timestamp" not in encoded
    loaded = experiment_config_from_dict(json.loads(encoded))
    assert loaded == config
    assert loaded.resolved_dict() == raw
    assert (
        json.dumps(loaded.resolved_dict(), sort_keys=True, separators=(",", ":"))
        == encoded
    )

    record_path = tmp_path / "confirmatory-freeze.json"
    record_path.write_text(
        json.dumps(record.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    assert load_confirmatory_freeze(record_path) == record
    assert confirmatory_freeze_from_dict(record.to_dict()) == record


def test_freeze_semantic_contract_requires_exact_registered_values() -> None:
    record = sealed_config().confirmatory_freeze
    assert record is not None
    record.verify_semantic_contract(
        analysis_contract=record.analysis_contract,
        analysis_version=record.analysis_version,
        analysis_code_hash=record.analysis_code_hash,
        dependency_lock_hash=record.dependency_lock_hash,
        environment_digest=record.environment_digest,
        tolerances=record.tolerance_values,
        margins=record.margin_values,
    )

    changed = {**record.tolerance_values, "paired_path_absolute_error": 1e-6}
    with pytest.raises(ConfirmatoryFreezeError, match="tolerances"):
        record.verify_semantic_contract(
            analysis_contract=record.analysis_contract,
            analysis_version=record.analysis_version,
            tolerances=changed,
            margins=record.margin_values,
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"schema_version":1,"schema_version":1}', "repeats key"),
        ('{"schema_version":NaN}', "non-finite"),
    ],
)
def test_freeze_loader_rejects_noncanonical_json(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "freeze.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_confirmatory_freeze(path)


def test_confirmatory_config_mutation_invalidates_seal() -> None:
    config = sealed_config()
    raw = config.resolved_dict()
    raw["horizon"] = 5
    raw["checkpoints"] = {"rounds": [0, 2, 4, 5]}
    with pytest.raises(ConfirmatoryFreezeError, match="config hash mismatch"):
        experiment_config_from_dict(raw)

    with pytest.raises(ConfirmatoryFreezeError, match="config hash mismatch"):
        replace(config, algorithm_replicas=config.algorithm_replicas + 1)


def test_confirmatory_execution_rejects_unregistered_study_design(
    tmp_path: Path,
) -> None:
    config = sealed_config()
    with pytest.raises(ConfirmatoryFreezeError, match="exact symbolic v1"):
        RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
            config,
            config.cells()[0],
        )
    assert not (tmp_path / config.name).exists()


def test_confirmatory_execution_rejects_caller_forged_provenance(
    tmp_path: Path,
) -> None:
    config = sealed_config()
    assert config.confirmatory_freeze is not None
    forged = replace(
        collect_provenance(),
        analysis_code_hash=config.confirmatory_freeze.analysis_code_hash,
    )

    with pytest.raises(ConfirmatoryFreezeError, match="current scientific source"):
        RunExecutor(
            tmp_path,
            ExactSymbolicAdapter,
            provenance=forged,
        ).execute(config, config.cells()[0])


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("tolerances", "frontier_certificate_gap_nats"),
            2e-7,
            "seal_hash mismatch",
        ),
        (("analysis_version",), "c" * 64, "seal_hash mismatch"),
        (("analysis_code_hash",), "c" * 64, "seal_hash mismatch"),
        (("dependency_lock_hash",), "c" * 64, "seal_hash mismatch"),
        (("environment_digest",), "c" * 64, "seal_hash mismatch"),
        (
            ("seed_banks", "algorithm", "identity_hash"),
            "c" * 64,
            "seal_hash mismatch",
        ),
        (("calibration_evidence_hash",), "c" * 64, "seal_hash mismatch"),
        (("confirmatory_frozen",), False, "confirmatory_frozen=true"),
    ],
)
def test_freeze_record_tampering_fails_closed(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw = sealed_config().resolved_dict()
    record = raw["confirmatory_freeze"]
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfirmatoryFreezeError, match=message):
        experiment_config_from_dict(raw)


def test_confirmatory_phase_requires_seal_and_rejects_wrong_phase() -> None:
    config = sealed_config()
    missing = config.freeze_payload()
    with pytest.raises(ConfirmatoryFreezeError, match="requires"):
        experiment_config_from_dict(missing)

    pilot_with_seal = config.resolved_dict()
    pilot_with_seal["phase"] = "pilot"
    with pytest.raises(ConfirmatoryFreezeError, match="must not contain"):
        experiment_config_from_dict(pilot_with_seal)

    calibration_with_seal = config.resolved_dict()
    calibration_with_seal["phase"] = "calibration"
    with pytest.raises(ConfirmatoryFreezeError, match="must not contain"):
        experiment_config_from_dict(calibration_with_seal)

    with pytest.raises(ConfirmatoryFreezeError, match="only a phase='calibration'"):
        freeze_experiment_config(
            replace(calibration_config(), phase="pilot"),
            confirmatory_master_seed="confirmatory-seed-v1",
            calibration_evidence_hash=evidence_hash(),
            analysis_contract="analysis",
            analysis_version=analysis_hash(),
            analysis_code_hash=analysis_hash(),
            dependency_lock_hash=analysis_hash(),
            environment_digest=analysis_hash(),
            seed_banks=SeedBankIdentities.bind(
                calibration_master_seed="calibration-seed-v1",
                confirmatory_master_seed="confirmatory-seed-v1",
                algorithm_master_seed="algorithm-seed-v1",
            ),
            tolerances={"numeric": 1e-7},
            margins={"effect": 0.01},
        )


def test_seed_namespaces_and_bound_identities_must_be_disjoint() -> None:
    calibration = SeedBankIdentity.bind("cal", namespace="shared")
    confirmatory = SeedBankIdentity.bind("confirm", namespace="shared")
    algorithm = SeedBankIdentity.bind("algorithm", namespace="algorithm")
    evaluation = SeedBankIdentity.bind("confirm", namespace="evaluation")
    with pytest.raises(ConfirmatoryFreezeError, match="namespaces must be disjoint"):
        SeedBankIdentities(calibration, confirmatory, algorithm, evaluation)
    overlapping_algorithm = SeedBankIdentity.bind(
        "algorithm",
        namespace=evaluation.namespace,
    )
    with pytest.raises(ConfirmatoryFreezeError, match="namespaces must be disjoint"):
        SeedBankIdentities(
            SeedBankIdentity.bind("cal", namespace="calibration"),
            SeedBankIdentity.bind("confirm", namespace="confirmatory"),
            overlapping_algorithm,
            evaluation,
        )

    banks = SeedBankIdentities.bind(
        calibration_master_seed="wrong-calibration",
        confirmatory_master_seed="confirmatory-seed-v1",
        algorithm_master_seed="algorithm-seed-v1",
    )
    with pytest.raises(ConfirmatoryFreezeError, match=r"calibration.*master_seed"):
        freeze_experiment_config(
            calibration_config(),
            confirmatory_master_seed="confirmatory-seed-v1",
            calibration_evidence_hash=evidence_hash(),
            analysis_contract="analysis",
            analysis_version=analysis_hash(),
            analysis_code_hash=analysis_hash(),
            dependency_lock_hash=analysis_hash(),
            environment_digest=analysis_hash(),
            seed_banks=banks,
            tolerances={"numeric": 1e-7},
            margins={"effect": 0.01},
        )

    wrong_confirmatory = SeedBankIdentities.bind(
        calibration_master_seed="calibration-seed-v1",
        confirmatory_master_seed="other-confirmatory",
        algorithm_master_seed="algorithm-seed-v1",
    )
    with pytest.raises(
        ConfirmatoryFreezeError,
        match="configured master seeds",
    ):
        freeze_experiment_config(
            calibration_config(),
            confirmatory_master_seed="confirmatory-seed-v1",
            calibration_evidence_hash=evidence_hash(),
            analysis_contract="analysis",
            analysis_version=analysis_hash(),
            analysis_code_hash=analysis_hash(),
            dependency_lock_hash=analysis_hash(),
            environment_digest=analysis_hash(),
            seed_banks=wrong_confirmatory,
            tolerances={"numeric": 1e-7},
            margins={"effect": 0.01},
        )

    wrong_algorithm = SeedBankIdentities.bind(
        calibration_master_seed="calibration-seed-v1",
        confirmatory_master_seed="confirmatory-seed-v1",
        algorithm_master_seed="wrong-algorithm",
    )
    with pytest.raises(ConfirmatoryFreezeError, match="configured master seeds"):
        freeze_experiment_config(
            calibration_config(),
            confirmatory_master_seed="confirmatory-seed-v1",
            calibration_evidence_hash=evidence_hash(),
            analysis_contract="analysis",
            analysis_version=analysis_hash(),
            analysis_code_hash=analysis_hash(),
            dependency_lock_hash=analysis_hash(),
            environment_digest=analysis_hash(),
            seed_banks=wrong_algorithm,
            tolerances={"numeric": 1e-7},
            margins={"effect": 0.01},
        )

    with pytest.raises(ConfirmatoryFreezeError, match="master seeds must be distinct"):
        SeedBankIdentities.bind(
            calibration_master_seed="calibration-seed-v1",
            confirmatory_master_seed="calibration-seed-v1",
            algorithm_master_seed="algorithm-seed-v1",
        )


def test_legacy_pilot_and_explicit_calibration_remain_unsealed() -> None:
    pilot_raw = {
        "schema_version": 1,
        "name": "legacy-pilot",
        "phase": "pilot",
        "master_seed": "pilot-seed",
        "horizon": 2,
        "checkpoints": {"rounds": [0, 2]},
        "environments": [{"kind": "IND", "projection_size": 1}],
        "agents": [{"kind": "reward", "target_size": 1}],
    }
    pilot = experiment_config_from_dict(pilot_raw)
    assert pilot.phase == "pilot"
    assert not pilot.confirmatory_frozen
    assert pilot.confirmatory_freeze is None
    assert "confirmatory_freeze" not in pilot.resolved_dict()
    assert "confirmatory_freeze_hash" not in pilot.resolved_run_settings()

    calibration = experiment_config_from_dict(
        {**pilot_raw, "name": "calibration", "phase": "calibration"}
    )
    assert calibration.phase == "calibration"
    assert calibration.confirmatory_freeze is None

    with pytest.raises(ConfirmatoryFreezeError, match="requires"):
        experiment_config_from_dict({**pilot_raw, "phase": "confirmatory"})


def test_schedule_parameters_are_strictly_kind_specific() -> None:
    with pytest.raises(ValueError, match="require growth_step"):
        AgentConfig(AgentKind.SCHEDULED, target_size=1)
    with pytest.raises(ValueError, match="must exceed"):
        AgentConfig(
            AgentKind.SCHEDULED,
            target_size=2,
            growth_step=1,
            growth_interval=1,
            maximum_size=2,
        )
    with pytest.raises(ValueError, match="only valid"):
        AgentConfig(AgentKind.REWARD, target_size=2, growth_step=1)

    relevant = AgentConfig(AgentKind.RELEVANT_INFORMATION, target_size=2)
    assert relevant.growth_step is None


@pytest.mark.parametrize("invalid", [True, "0.01", float("inf"), -0.01])
def test_threshold_values_are_strict_finite_numbers(invalid: object) -> None:
    calibration = calibration_config()
    banks = SeedBankIdentities.bind(
        calibration_master_seed=calibration.master_seed,
        confirmatory_master_seed="confirmatory-seed-v1",
        algorithm_master_seed=calibration.algorithm_master_seed,
    )
    with pytest.raises((ConfirmatoryFreezeError, TypeError)):
        freeze_experiment_config(
            calibration,
            confirmatory_master_seed="confirmatory-seed-v1",
            calibration_evidence_hash=evidence_hash(),
            analysis_contract="analysis",
            analysis_version=analysis_hash(),
            analysis_code_hash=analysis_hash(),
            dependency_lock_hash=analysis_hash(),
            environment_digest=analysis_hash(),
            seed_banks=banks,
            tolerances={"invalid": invalid},  # type: ignore[dict-item]
            margins={"effect": 0.01},
        )


def test_programmatic_config_models_reject_bool_and_string_aliases() -> None:
    with pytest.raises(TypeError, match="EnvironmentKind"):
        EnvironmentConfig("IND")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AgentKind"):
        AgentConfig("reward")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="public_reward_cap"):
        EnvironmentConfig(EnvironmentKind.PUBLIC_C, public_reward_cap=True)
    with pytest.raises(ValueError, match="epsilon"):
        FeedbackConfig(epsilon=False)
    with pytest.raises(ValueError, match="tolerance"):
        SolverConfig(tolerance=True)
    raw = calibration_config().resolved_dict()
    raw["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        experiment_config_from_dict(raw)


@pytest.mark.parametrize(
    "reserved_name",
    ["_frontiers", ".infinite-rulebook-reproducibility"],
)
def test_experiment_names_cannot_collide_with_reserved_artifact_paths(
    reserved_name: str,
) -> None:
    with pytest.raises(ValueError, match="reserved artifact path"):
        replace(calibration_config(), name=reserved_name)
