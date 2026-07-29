from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.analysis import (
    AnalysisPhase,
    ContrastInterpretation,
    analysis_plan_json,
)
from infinite_rulebook.analysis.canaries import (
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
    MetricTrajectoryIdentityCanary,
)
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    ConfirmatoryFreezeRecord,
    SeedBankIdentities,
    SeedBankIdentity,
    confirmatory_config_hash,
    freeze_experiment_config,
)
from infinite_rulebook.studies.symbolic_construct import (
    PAIRED_PATH_ABSOLUTE_TOLERANCE,
    PRIMARY_MINIMUM_EFFECTS,
    STUDY_CONTRACT,
    SYMBOLIC_V1_ALGORITHM_MASTER_SEED,
    SYMBOLIC_V1_ALGORITHM_REPLICAS,
    SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS,
    SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
    SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED,
    SYMBOLIC_V1_CONFIRMATORY_NAME,
    SYMBOLIC_V1_DESIGN_HASH,
    build_symbolic_analysis_plan,
    build_symbolic_canary_plan,
    expected_analysis_groups,
    expected_confirmatory_margins,
    expected_confirmatory_registration,
    expected_confirmatory_tolerances,
    symbolic_v1_design_hash,
    symbolic_v1_design_payload,
    verify_symbolic_calibration_design,
    verify_symbolic_confirmatory_contract,
)


def test_symbolic_study_registration_matches_calibration_matrix() -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")

    plan = build_symbolic_analysis_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    canaries = build_symbolic_canary_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )

    assert len(plan.expected_groups) == len(config.environments) * len(config.agents)
    assert plan.expected_groups == expected_analysis_groups(config)
    assert {contrast.name for contrast in plan.contrasts} == set(
        PRIMARY_MINIMUM_EFFECTS
    )
    assert {contrast.name: contrast.interpretation for contrast in plan.contrasts} == {
        name: (
            ContrastInterpretation.TELEMETRY_ONLY
            if name == "alea-over-ind-prediction-error-novelty"
            else ContrastInterpretation.INFERENTIAL
        )
        for name in PRIMARY_MINIMUM_EFFECTS
    }
    assert len(plan.scalings) == 6
    agents = {agent.kind.value for agent in config.agents}
    additive = tuple(
        canary
        for canary in canaries.canaries
        if isinstance(canary, ConstantAdditiveMetricCanary)
    )
    exact_zero = tuple(
        canary
        for canary in canaries.canaries
        if isinstance(canary, ExactZeroMetricCanary)
    )
    assert len(canaries.canaries) == 8 + 2 * len(config.agents)
    assert {canary.selector.agent_kind for canary in additive} == agents
    assert {canary.selector.agent_kind for canary in exact_zero} == agents
    assert all(
        canary.tolerance == PAIRED_PATH_ABSOLUTE_TOLERANCE
        for canary in canaries.canaries
        if isinstance(
            canary,
            (ConstantAdditiveMetricCanary, MetricTrajectoryIdentityCanary),
        )
    )


def test_checked_in_calibration_plans_match_current_generators() -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")
    analysis = build_symbolic_analysis_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    canaries = build_symbolic_canary_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )

    assert Path("configs/symbolic-calibration-analysis-v1.json").read_text(
        encoding="utf-8"
    ) == analysis_plan_json(analysis)
    assert (
        Path("configs/symbolic-calibration-canaries-v1.json").read_text(
            encoding="utf-8"
        )
        == canaries.to_json()
    )


def test_symbolic_v1_design_hash_covers_every_phase_independent_field() -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")
    payload = symbolic_v1_design_payload(config)

    assert symbolic_v1_design_hash(config) == SYMBOLIC_V1_DESIGN_HASH
    assert config.algorithm_master_seed == SYMBOLIC_V1_ALGORITHM_MASTER_SEED
    assert config.algorithm_replicas == SYMBOLIC_V1_ALGORITHM_REPLICAS
    assert config.environment_replicas == SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS
    assert set(payload) == {
        "agents",
        "algorithm_master_seed",
        "algorithm_replicas",
        "checkpoints",
        "environments",
        "feedback",
        "horizon",
        "reward",
        "schema_version",
        "solver",
    }
    assert set(payload).isdisjoint(
        {"name", "phase", "master_seed", "environment_replicas", "confirmatory_freeze"}
    )
    assert payload["algorithm_master_seed"] == SYMBOLIC_V1_ALGORITHM_MASTER_SEED
    verify_symbolic_calibration_design(config)


@pytest.mark.parametrize(
    "changed",
    [
        lambda config: replace(
            config,
            environments=tuple(reversed(config.environments)),
        ),
        lambda config: replace(
            config,
            agents=tuple(reversed(config.agents)),
        ),
        lambda config: replace(
            config,
            checkpoints=replace(
                config.checkpoints,
                rounds=config.checkpoints.rounds[:-1],
            ),
        ),
        lambda config: replace(config, horizon=config.horizon + 1),
        lambda config: replace(
            config,
            feedback=replace(config.feedback, epsilon=0.2),
        ),
        lambda config: replace(
            config,
            reward=replace(config.reward, u=2.0),
        ),
        lambda config: replace(
            config,
            solver=replace(config.solver, bound_tolerance=2e-7),
        ),
        lambda config: replace(config, algorithm_master_seed="other-algorithm-bank"),
        lambda config: replace(config, algorithm_replicas=4),
    ],
)
def test_symbolic_v1_design_rejects_any_scientific_field_change(changed) -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")

    with pytest.raises(ConfirmatoryFreezeError, match="exact symbolic v1"):
        verify_symbolic_calibration_design(changed(config))


@pytest.mark.parametrize(
    "changed",
    [
        lambda config: replace(config, name="another-safe-name"),
        lambda config: replace(config, phase="pilot"),
        lambda config: replace(config, master_seed="another-phase-seed"),
        lambda config: replace(config, environment_replicas=48),
    ],
)
def test_symbolic_v1_design_hash_excludes_registered_non_design_fields(changed) -> None:
    config = load_experiment_config("configs/symbolic-calibration-v1.json")

    assert symbolic_v1_design_hash(changed(config)) == SYMBOLIC_V1_DESIGN_HASH


def test_symbolic_calibration_requires_exact_development_replica_count() -> None:
    config = replace(
        load_experiment_config("configs/symbolic-calibration-v1.json"),
        environment_replicas=32,
    )

    with pytest.raises(ConfirmatoryFreezeError, match="exactly 192"):
        verify_symbolic_calibration_design(config)


def test_confirmatory_registration_hash_is_seal_independent() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")
    design = replace(
        calibration,
        environment_replicas=32,
        algorithm_replicas=3,
    )
    draft = build_symbolic_analysis_plan(
        design,
        phase=AnalysisPhase.CONFIRMATORY,
    )
    sealed = replace(draft, freeze_hash="f" * 64)

    assert draft.registration_hash == sealed.registration_hash
    assert draft.scientific_hash != sealed.scientific_hash
    assert all(group.environment_replicas == 32 for group in sealed.expected_groups)


@pytest.mark.parametrize(
    "changed",
    [
        lambda config: replace(config, master_seed="unregistered-calibration-seed"),
        lambda config: replace(config, phase="pilot"),
    ],
)
def test_confirmatory_draft_requires_registered_calibration_source(changed) -> None:
    design = replace(
        load_experiment_config("configs/symbolic-calibration-v1.json"),
        environment_replicas=32,
    )

    with pytest.raises(ConfirmatoryFreezeError, match="calibration design"):
        expected_confirmatory_registration(changed(design))


def _sealed_symbolic_config(
    *,
    analysis_contract: str = STUDY_CONTRACT,
    analysis_version: str | None = None,
    tolerances: dict[str, float] | None = None,
    margins: dict[str, float] | None = None,
    analysis_code_hash: str = "a" * 64,
):
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")
    design = replace(calibration, environment_replicas=32)
    registration = expected_confirmatory_registration(design)
    return freeze_experiment_config(
        design,
        name=SYMBOLIC_V1_CONFIRMATORY_NAME,
        confirmatory_master_seed=SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED,
        calibration_evidence_hash="b" * 64,
        analysis_contract=analysis_contract,
        analysis_version=analysis_version or registration,
        analysis_code_hash=analysis_code_hash,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
        seed_banks=SeedBankIdentities.bind(
            calibration_master_seed=design.master_seed,
            confirmatory_master_seed=SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED,
            algorithm_master_seed=design.algorithm_master_seed,
        ),
        tolerances=tolerances or expected_confirmatory_tolerances(design),
        margins=margins or expected_confirmatory_margins(),
    )


def test_symbolic_confirmatory_contract_is_exact_and_self_consistent() -> None:
    sealed = _sealed_symbolic_config()
    record = verify_symbolic_confirmatory_contract(
        sealed,
        analysis_code_hash="a" * 64,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
    )

    assert record.analysis_contract == STUDY_CONTRACT
    assert record.analysis_version == expected_confirmatory_registration(sealed)
    assert record.tolerance_values == expected_confirmatory_tolerances(sealed)
    assert record.margin_values == expected_confirmatory_margins()
    plan = build_symbolic_analysis_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
        freeze_hash=record.seal_hash,
    )
    assert plan.registration_hash == record.analysis_version
    assert build_symbolic_canary_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
    ).canaries


@pytest.mark.parametrize(
    ("overrides", "mismatch"),
    [
        ({"analysis_contract": "other-study.v1"}, "analysis_contract"),
        ({"analysis_version": "c" * 64}, "analysis_version"),
        (
            {
                "tolerances": {
                    **expected_confirmatory_tolerances(
                        load_experiment_config("configs/symbolic-calibration-v1.json")
                    ),
                    "paired_path_absolute_error": 1e-6,
                }
            },
            "tolerances",
        ),
        (
            {
                "margins": {
                    **expected_confirmatory_margins(),
                    "ind-red-terminal-hidden-reward-equivalence": 0.5,
                }
            },
            "margins",
        ),
    ],
)
def test_symbolic_confirmatory_contract_rejects_valid_but_wrong_seals(
    overrides: dict[str, object],
    mismatch: str,
) -> None:
    sealed = _sealed_symbolic_config(**overrides)  # type: ignore[arg-type]

    with pytest.raises(ConfirmatoryFreezeError, match=mismatch):
        verify_symbolic_confirmatory_contract(sealed)
    record = sealed.confirmatory_freeze
    assert record is not None
    with pytest.raises(ConfirmatoryFreezeError, match=mismatch):
        build_symbolic_analysis_plan(
            sealed,
            phase=AnalysisPhase.CONFIRMATORY,
            freeze_hash=record.seal_hash,
        )


def test_symbolic_confirmatory_contract_rejects_analysis_code_drift() -> None:
    sealed = _sealed_symbolic_config()

    with pytest.raises(ConfirmatoryFreezeError, match="analysis_code_hash"):
        verify_symbolic_confirmatory_contract(
            sealed,
            analysis_code_hash="d" * 64,
            dependency_lock_hash="c" * 64,
            environment_digest="d" * 64,
        )

    with pytest.raises(ConfirmatoryFreezeError, match="dependency_lock_hash"):
        verify_symbolic_confirmatory_contract(
            sealed,
            analysis_code_hash="a" * 64,
            dependency_lock_hash="e" * 64,
            environment_digest="d" * 64,
        )

    with pytest.raises(ConfirmatoryFreezeError, match="environment_digest"):
        verify_symbolic_confirmatory_contract(
            sealed,
            analysis_code_hash="a" * 64,
            dependency_lock_hash="c" * 64,
            environment_digest="e" * 64,
        )


def test_symbolic_contract_independently_rejects_reused_calibration_seed() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")
    design = replace(calibration, environment_replicas=32)
    payload = design.freeze_payload()
    payload.update(
        {
            "name": "reused-calibration-seed",
            "phase": "confirmatory",
            "master_seed": SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
        }
    )
    banks = SeedBankIdentities(
        calibration=SeedBankIdentity.bind(
            SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
            namespace="calibration.v1",
        ),
        confirmatory=SeedBankIdentity.bind(
            SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
            namespace="confirmatory.v1",
        ),
        algorithm=SeedBankIdentity.bind(
            SYMBOLIC_V1_ALGORITHM_MASTER_SEED,
            namespace="algorithm.v1",
        ),
        evaluation=SeedBankIdentity.bind(
            SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
            namespace="evaluation.v1",
        ),
    )
    record = ConfirmatoryFreezeRecord.create(
        config_hash=confirmatory_config_hash(payload),
        calibration_evidence_hash="b" * 64,
        analysis_contract=STUDY_CONTRACT,
        analysis_version=expected_confirmatory_registration(design),
        analysis_code_hash="a" * 64,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
        seed_banks=banks,
        tolerances=expected_confirmatory_tolerances(design),
        margins=expected_confirmatory_margins(),
    )
    sealed = replace(
        design,
        name=payload["name"],
        phase="confirmatory",
        master_seed=SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
        confirmatory_freeze=record,
    )
    record.verify_config(sealed)

    with pytest.raises(ConfirmatoryFreezeError, match="registered name"):
        verify_symbolic_confirmatory_contract(sealed)


def test_symbolic_contract_verifies_calibration_seed_identity() -> None:
    sealed = _sealed_symbolic_config()
    record = sealed.confirmatory_freeze
    assert record is not None
    wrong_banks = replace(
        record.seed_banks,
        calibration=SeedBankIdentity.bind(
            "unregistered-calibration-bank",
            namespace=record.seed_banks.calibration.namespace,
        ),
    )
    wrong_record = ConfirmatoryFreezeRecord.create(
        config_hash=record.config_hash,
        calibration_evidence_hash=record.calibration_evidence_hash,
        analysis_contract=record.analysis_contract,
        analysis_version=record.analysis_version,
        analysis_code_hash=record.analysis_code_hash,
        dependency_lock_hash=record.dependency_lock_hash,
        environment_digest=record.environment_digest,
        seed_banks=wrong_banks,
        tolerances=record.tolerance_values,
        margins=record.margin_values,
    )
    malicious = replace(sealed, confirmatory_freeze=wrong_record)
    wrong_record.verify_config(malicious)

    with pytest.raises(ConfirmatoryFreezeError, match="seed-bank identities"):
        verify_symbolic_confirmatory_contract(malicious)


def test_symbolic_confirmation_requires_and_seals_fixed_algorithm_bank() -> None:
    registered = load_experiment_config("configs/symbolic-calibration-v1.json")
    tolerances = expected_confirmatory_tolerances(registered)
    calibration = replace(
        registered,
        algorithm_master_seed=None,
    )
    design = replace(calibration, environment_replicas=32)
    with pytest.raises(ConfirmatoryFreezeError, match="algorithm_master_seed"):
        freeze_experiment_config(
            design,
            name="missing-algorithm-bank",
            confirmatory_master_seed="confirmatory-phase-seed",
            calibration_evidence_hash="b" * 64,
            analysis_contract=STUDY_CONTRACT,
            analysis_version="c" * 64,
            analysis_code_hash="a" * 64,
            dependency_lock_hash="c" * 64,
            environment_digest="d" * 64,
            seed_banks=SeedBankIdentities.bind(
                calibration_master_seed=design.master_seed,
                confirmatory_master_seed="confirmatory-phase-seed",
                algorithm_master_seed="placeholder-algorithm-bank",
            ),
            tolerances=tolerances,
            margins=expected_confirmatory_margins(),
        )

    sealed = _sealed_symbolic_config()
    with pytest.raises(ConfirmatoryFreezeError, match="config hash mismatch"):
        replace(sealed, algorithm_master_seed="changed-fixed-bank")


def test_symbolic_confirmation_rejects_unregistered_environment_count() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")
    design = replace(calibration, environment_replicas=17)
    with pytest.raises(ConfirmatoryFreezeError, match="power-candidate"):
        expected_confirmatory_registration(design)
    with pytest.raises(ConfirmatoryFreezeError, match="power-candidate"):
        build_symbolic_analysis_plan(
            design,
            phase=AnalysisPhase.CONFIRMATORY,
        )

    confirmatory_seed = SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED
    sealed = freeze_experiment_config(
        design,
        name=SYMBOLIC_V1_CONFIRMATORY_NAME,
        confirmatory_master_seed=confirmatory_seed,
        calibration_evidence_hash="b" * 64,
        analysis_contract=STUDY_CONTRACT,
        analysis_version="c" * 64,
        analysis_code_hash="a" * 64,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
        seed_banks=SeedBankIdentities.bind(
            calibration_master_seed=design.master_seed,
            confirmatory_master_seed=confirmatory_seed,
            algorithm_master_seed=design.algorithm_master_seed,
        ),
        tolerances=expected_confirmatory_tolerances(design),
        margins=expected_confirmatory_margins(),
    )

    with pytest.raises(ConfirmatoryFreezeError, match="power-candidate"):
        verify_symbolic_confirmatory_contract(sealed)


def test_symbolic_confirmation_rejects_unregistered_master_seed() -> None:
    sealed = _sealed_symbolic_config()
    wrong_seed = "unregistered-confirmatory-master"
    wrong_banks = SeedBankIdentities.bind(
        calibration_master_seed=SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
        confirmatory_master_seed=wrong_seed,
        algorithm_master_seed=SYMBOLIC_V1_ALGORITHM_MASTER_SEED,
    )
    record = sealed.confirmatory_freeze
    assert record is not None
    wrong_payload = sealed.freeze_payload()
    wrong_payload["master_seed"] = wrong_seed
    wrong_record = ConfirmatoryFreezeRecord.create(
        config_hash=confirmatory_config_hash(wrong_payload),
        calibration_evidence_hash=record.calibration_evidence_hash,
        analysis_contract=record.analysis_contract,
        analysis_version=record.analysis_version,
        analysis_code_hash=record.analysis_code_hash,
        dependency_lock_hash=record.dependency_lock_hash,
        environment_digest=record.environment_digest,
        seed_banks=wrong_banks,
        tolerances=record.tolerance_values,
        margins=record.margin_values,
    )
    malicious = replace(
        sealed,
        master_seed=wrong_seed,
        confirmatory_freeze=wrong_record,
    )

    with pytest.raises(ConfirmatoryFreezeError, match="registered name"):
        verify_symbolic_confirmatory_contract(malicious)
