from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.analysis import analysis_plan_json, load_analysis_plan
from infinite_rulebook.analysis.models import AnalysisPhase
from infinite_rulebook.analysis.power import (
    EnvironmentCluster,
    PowerHypothesis,
    calibrate_environment_count,
)
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentKind,
    load_experiment_config,
)
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    SeedBankIdentities,
    freeze_experiment_config,
)
from infinite_rulebook.studies import symbolic_construct_v2 as v2
from infinite_rulebook.studies.symbolic_construct_v2 import (
    LEGACY_D6_REPLICATION,
    POWER_CANDIDATE_ENVIRONMENTS,
    POWER_RNG_STREAM,
    PRIMARY_MINIMUM_EFFECTS,
    S2_EARLY,
    S2_LOAD,
    SECONDARY_D12,
    STUDY_CONTRACT,
    SYMBOLIC_V2_ALGORITHM_MASTER_SEED,
    SYMBOLIC_V2_CALIBRATION_MASTER_SEED,
    SYMBOLIC_V2_COMPONENT_HASH,
    SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED,
    SYMBOLIC_V2_CONFIRMATORY_NAME,
    SYMBOLIC_V2_DESIGN_HASH,
    SYMBOLIC_V2_SMOKE_CONFIG_HASH,
    SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
    build_symbolic_analysis_plan,
    build_symbolic_canary_plan,
    build_symbolic_supplemental_plan,
    calibration_evidence_hash_from_hashes,
    expected_confirmatory_margins,
    expected_confirmatory_registration,
    expected_confirmatory_tolerances,
    registration_component_hash,
    registration_component_payload,
    symbolic_v2_design_hash,
    verify_symbolic_calibration_design,
    verify_symbolic_confirmatory_contract,
)

_CONFIG = Path(__file__).parents[1] / "configs" / "symbolic-calibration-v2.json"
_ANALYSIS_PLAN = (
    Path(__file__).parents[1] / "configs" / "symbolic-calibration-analysis-v2.json"
)
_CANARY_PLAN = (
    Path(__file__).parents[1] / "configs" / "symbolic-calibration-canaries-v2.json"
)
_SUPPLEMENTAL_PLAN = (
    Path(__file__).parents[1] / "configs" / "symbolic-calibration-supplemental-v2.json"
)


def _power_hypothesis() -> PowerHypothesis:
    return PowerHypothesis(
        "stream-test",
        tuple(EnvironmentCluster(index, (0.5 + 0.01 * index,)) for index in range(32)),
        minimum_effect=0.25,
    )


def test_v2_registration_matches_the_ambitious_matrix() -> None:
    config = load_experiment_config(_CONFIG)
    verify_symbolic_calibration_design(config)

    assert config.config_hash == (
        "c0f4cf5bf09e6b516379c0fec26ccd4a8780d8b6d52226093ef5a96cc0437508"
    )
    assert symbolic_v2_design_hash(config) == SYMBOLIC_V2_DESIGN_HASH
    assert registration_component_hash(config) == SYMBOLIC_V2_COMPONENT_HASH
    assert config.master_seed == SYMBOLIC_V2_CALIBRATION_MASTER_SEED
    assert config.algorithm_master_seed == SYMBOLIC_V2_ALGORITHM_MASTER_SEED
    assert config.horizon == 12
    assert config.checkpoints.rounds == tuple(range(13))
    assert config.environment_replicas == 192
    assert config.algorithm_replicas == 8
    assert len(config.environments) == 8
    assert len(config.agents) == 6
    assert len(config.cells()) == 73_728
    trivia_loads = {
        environment.distractor_dimensions
        for environment in config.environments
        if environment.kind is EnvironmentKind.TRIVIA
    }
    assert trivia_loads == {6, 12, 24}


def test_v2_checked_in_registration_artifacts_are_exact_builder_outputs() -> None:
    config = load_experiment_config(_CONFIG)
    analysis = build_symbolic_analysis_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    canaries = build_symbolic_canary_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    supplemental = build_symbolic_supplemental_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )

    assert load_analysis_plan(_ANALYSIS_PLAN) == analysis
    assert _ANALYSIS_PLAN.read_text(encoding="utf-8") == analysis_plan_json(analysis)
    assert _CANARY_PLAN.read_text(encoding="utf-8") == canaries.to_json()
    assert _SUPPLEMENTAL_PLAN.read_text(encoding="utf-8") == supplemental.to_json()


def test_v2_plan_has_six_exact_primaries_and_no_scope_reduction() -> None:
    config = load_experiment_config(_CONFIG)
    plan = build_symbolic_analysis_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )

    assert len(plan.expected_groups) == 48
    assert len(plan.contrasts) == 6
    assert {contrast.name for contrast in plan.contrasts} == set(
        PRIMARY_MINIMUM_EFFECTS
    )
    assert plan.scientific_hash == (
        "aef2100b60636a86f73f871a2b9f99346b2207762b872ec8262a512226a1f6fc"
    )
    assert plan.registration_hash == (
        "747b53dc6fafbf354c595e20d57269270230a5f9d05a655a7df9da0e4903a1d0"
    )
    by_name = {contrast.name: contrast for contrast in plan.contrasts}
    assert by_name[S2_EARLY].metric == ("post_query_mean_hidden_expected_reward")
    assert by_name[S2_LOAD].metric == "hidden_expected_reward"
    assert by_name[S2_EARLY].left.condition_hash != (
        by_name[S2_LOAD].left.condition_hash
    )
    for contrast in plan.contrasts:
        for selector in (contrast.left, contrast.right):
            assert selector.condition_hash is not None
            assert selector.agent_hash is not None
    trivia_conditions = {
        group.condition_hash
        for group in plan.expected_groups
        if group.environment_kind == EnvironmentKind.TRIVIA.value
    }
    assert len(trivia_conditions) == 3


def test_v2_component_binds_compound_non_rescue_and_compact_evidence() -> None:
    config = load_experiment_config(_CONFIG)
    component = registration_component_payload(config)

    assert component["compound_s2"] == {
        "components": [S2_EARLY, S2_LOAD],
        "decision": "both-primary-components-required",
        "legacy_replication_may_rescue": False,
    }
    legacy = component["legacy_replication"]
    assert legacy["name"] == LEGACY_D6_REPLICATION
    assert legacy["family_membership"] == "outside-primary-holm"
    assert legacy["may_rescue_compound_s2"] is False
    secondary = component["d12_secondary"]
    assert secondary["name"] == SECONDARY_D12
    assert secondary["role"] == "registered-descriptive-paired-contrast"
    assert secondary["paired_by"] == [
        "environment_replica",
        "algorithm_replica",
    ]
    assert secondary["may_rescue_compound_s2"] is False
    assert component["post_query_metric"]["source_checkpoint_t_positive"] == (
        "equals the authenticated post-query training-event reward at round t"
    )
    compact = component["compact_canaries"]
    assert compact["gate_count"] == 27
    assert compact["aggregate_metric_coverage"] == (
        "all-exact-registered-condition-agent-groups"
    )
    assert compact["detail_chunk_records"] == 4096
    assert compact["calibration_plan_hash"] == (
        "645a509da5f3c66563df8d88796a7a10ca87e817fd0e4549536fd68095e06a9a"
    )
    assert (
        "sealed selected environment-replica count"
        in (compact["confirmatory_inventory_rule"])
    )
    assert component["supplemental_evidence"] == {
        "calibration_plan_hash": (
            "1856efab9c7d4c0518dbb4004f3915bb9c70bb9376fbc255825bf4451c96c7f8"
        ),
        "confirmatory_inventory_rule": (
            "same registered comparisons and exact groups with the sealed "
            "selected environment-replica count"
        ),
        "legacy_replication": LEGACY_D6_REPLICATION,
        "descriptive_comparison": SECONDARY_D12,
        "family_membership": "outside-primary-holm",
        "may_rescue_compound_s2": False,
    }
    assert component["power"]["rng_stream"] == POWER_RNG_STREAM
    assert component["stage_0_prerequisite"] == {
        "role": "operational-not-inferential",
        "required": True,
        "config_hash": SYMBOLIC_V2_SMOKE_CONFIG_HASH,
        "evidence_hash": SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
        "does_not_replace_v2_adapter_validation": True,
    }


def test_v2_component_binds_complete_power_and_freeze_decision_contract() -> None:
    config = load_experiment_config(_CONFIG)
    component = registration_component_payload(config)

    assert component["power"] == {
        "candidate_environment_counts": list(v2.POWER_CANDIDATE_ENVIRONMENTS),
        "calibration_environment_count": 192,
        "center_environment_count": 64,
        "probability_environment_count": 128,
        "simulations": 10_000,
        "seed": "bounded-symbolic-power-v2",
        "rng_stream": "analysis.cluster-power.v2",
        "alpha": 0.05,
        "simulation_error_alpha": v2.POWER_SIMULATION_ERROR_ALPHA,
        "design_confidence_alpha": v2.POWER_DESIGN_CONFIDENCE_ALPHA,
        "minimum_individual_power": 0.90,
        "minimum_equivalence_power": 0.90,
        "minimum_joint_power": 0.80,
        "maximum_global_null_fwer": 0.05,
        "maximum_false_equivalence_boundary_error": 0.05,
        "primary_minimum_effects": {
            name: v2.PRIMARY_MINIMUM_EFFECTS[name]
            for name in sorted(v2.PRIMARY_MINIMUM_EFFECTS)
        },
        "s5_equivalence": {
            "name": v2.S5_EQUIVALENCE,
            "margin": 0.25,
            "margin_provenance_hash": v2.S5_REWARD_MARGIN_PROVENANCE_HASH,
            "diagnostic_location": 0.0,
        },
        "selection_rule": "smallest-candidate-meeting-every-certified-target",
    }
    assert component["confirmatory_freeze"] == {
        "tolerances": {
            "aggregate_metric_absolute_error": 1e-12,
            "artifact_completion_fraction": 1.0,
            "frontier_bound_tolerance_nats": config.solver.bound_tolerance,
            "ledger_reconciliation_nats": 1e-12,
            "paired_path_absolute_error": 1e-12,
        },
        "margins": {
            **v2.PRIMARY_MINIMUM_EFFECTS,
            v2.S5_EQUIVALENCE: 0.25,
        },
    }


def test_v2_component_hash_changes_with_every_decision_parameter_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(_CONFIG)
    baseline = registration_component_hash(config)
    mutations = (
        {
            "POWER_CANDIDATE_ENVIRONMENTS": (
                *v2.POWER_CANDIDATE_ENVIRONMENTS,
                1024,
            )
        },
        {"SYMBOLIC_V2_CALIBRATION_ENVIRONMENT_REPLICAS": 193},
        {"POWER_CENTER_ENVIRONMENTS": 63},
        {"POWER_PROBABILITY_ENVIRONMENTS": 129},
        {"POWER_SIMULATIONS": 9_999},
        {"POWER_SEED": "changed-power-seed"},
        {"POWER_RNG_STREAM": "analysis.cluster-power.changed"},
        {"POWER_ALPHA": 0.04},
        {"POWER_SIMULATION_ERROR_ALPHA": 0.02},
        {"POWER_DESIGN_CONFIDENCE_ALPHA": 0.02},
        {"MINIMUM_INDIVIDUAL_POWER": 0.89},
        {"MINIMUM_EQUIVALENCE_POWER": 0.89},
        {"MINIMUM_JOINT_POWER": 0.79},
        {"MAXIMUM_GLOBAL_NULL_FWER": 0.04},
        {
            "PRIMARY_MINIMUM_EFFECTS": {
                **v2.PRIMARY_MINIMUM_EFFECTS,
                v2.S2_EARLY: 0.24,
            }
        },
        {"S5_REWARD_EQUIVALENCE_MARGIN": 0.24},
        {"S5_REWARD_MARGIN_PROVENANCE_HASH": "0" * 64},
        {"S5_BOOTSTRAP_DIAGNOSTIC_LOCATION": 0.01},
        {"POWER_SELECTION_RULE": "changed-selection-rule"},
        {"AGGREGATE_METRIC_ABSOLUTE_TOLERANCE": 1e-11},
        {"ARTIFACT_COMPLETION_FRACTION": 0.99},
        {"LEDGER_RECONCILIATION_TOLERANCE": 1e-11},
        {"PAIRED_PATH_ABSOLUTE_TOLERANCE": 1e-11},
    )
    for mutation in mutations:
        with monkeypatch.context() as context:
            for name, value in mutation.items():
                context.setattr(v2, name, value)
            assert registration_component_hash(config) != baseline, mutation

    changed_solver = replace(
        config.solver,
        bound_tolerance=config.solver.bound_tolerance * 2.0,
    )
    assert (
        registration_component_hash(replace(config, solver=changed_solver)) != baseline
    )


def test_v2_evidence_plans_are_exact_and_registration_bound() -> None:
    config = load_experiment_config(_CONFIG)
    canaries = build_symbolic_canary_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    supplemental = build_symbolic_supplemental_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )

    assert len(canaries.canaries) == 26
    assert len(canaries.aggregate_canaries) == 1
    aggregate = canaries.aggregate_canaries[0]
    assert len(aggregate.selectors) == 48
    assert len(canaries.expected_groups) == 48
    assert len(supplemental.expected_groups) == 48
    assert aggregate.checkpoints == tuple(range(1, 13))
    assert aggregate.source_metric == "post_query_hidden_expected_reward"
    assert aggregate.aggregate_metric == "post_query_mean_hidden_expected_reward"
    assert {item.name for item in supplemental.legacy_replications} == {
        LEGACY_D6_REPLICATION
    }
    assert {item.name for item in supplemental.descriptive_comparisons} == {
        SECONDARY_D12
    }


def test_v2_unsuffixed_canary_names_retain_the_registered_agents() -> None:
    config = load_experiment_config(_CONFIG)
    plan = build_symbolic_canary_plan(
        config,
        phase=AnalysisPhase.CALIBRATION,
    )
    by_name = {canary.name: canary for canary in plan.canaries}

    assert by_name["public-reward-decomposition"].selector.agent_kind == (
        AgentKind.REWARD.value
    )
    assert (
        by_name["alea-has-no-persistent-distractor-information"].selector.agent_kind
        == AgentKind.NOVELTY.value
    )


def test_v2_component_and_evidence_plans_support_an_actual_sealed_e32_config() -> None:
    calibration = load_experiment_config(_CONFIG)
    design = replace(calibration, environment_replicas=32)
    registration = expected_confirmatory_registration(design)
    sealed = freeze_experiment_config(
        design,
        name=SYMBOLIC_V2_CONFIRMATORY_NAME,
        confirmatory_master_seed=SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED,
        calibration_evidence_hash="b" * 64,
        analysis_contract=STUDY_CONTRACT,
        analysis_version=registration,
        analysis_code_hash="a" * 64,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
        seed_banks=SeedBankIdentities.bind(
            calibration_master_seed=SYMBOLIC_V2_CALIBRATION_MASTER_SEED,
            confirmatory_master_seed=SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED,
            algorithm_master_seed=SYMBOLIC_V2_ALGORITHM_MASTER_SEED,
            calibration_namespace="calibration.v2",
            confirmatory_namespace="confirmatory.v2",
            algorithm_namespace="algorithm.v2",
            evaluation_namespace="evaluation.v2",
        ),
        tolerances=expected_confirmatory_tolerances(design),
        margins=expected_confirmatory_margins(),
    )

    verify_symbolic_confirmatory_contract(
        sealed,
        analysis_code_hash="a" * 64,
        dependency_lock_hash="c" * 64,
        environment_digest="d" * 64,
    )
    assert registration_component_hash(sealed) == SYMBOLIC_V2_COMPONENT_HASH
    canaries = build_symbolic_canary_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
    )
    supplemental = build_symbolic_supplemental_plan(
        sealed,
        phase=AnalysisPhase.CONFIRMATORY,
    )
    assert {group.environment_replicas for group in canaries.expected_groups} == {32}
    assert {group.environment_replicas for group in supplemental.expected_groups} == {
        32
    }


def test_v2_mutations_and_lookalike_designs_fail_closed() -> None:
    config = load_experiment_config(_CONFIG)
    trivia_d12 = next(
        environment
        for environment in config.environments
        if environment.kind is EnvironmentKind.TRIVIA
        and environment.distractor_dimensions == 12
    )
    changed_environment = replace(trivia_d12, distractor_dimensions=18)
    changed = replace(
        config,
        environments=tuple(
            changed_environment if item == trivia_d12 else item
            for item in config.environments
        ),
    )
    with pytest.raises(ConfirmatoryFreezeError, match="scientific design"):
        verify_symbolic_calibration_design(changed)
    with pytest.raises(ConfirmatoryFreezeError, match="name, seed"):
        verify_symbolic_calibration_design(
            replace(config, master_seed="lookalike-v2-seed")
        )


def test_v2_calibration_evidence_requires_the_registered_stage_zero() -> None:
    hashes = {
        "config_hash": "0" * 64,
        "analysis_report_hash": "1" * 64,
        "canary_report_hash": "2" * 64,
        "canary_detail_root_hash": "d" * 64,
        "canary_detail_record_count": 1,
        "supplemental_plan_hash": "e" * 64,
        "supplemental_report_hash": "f" * 64,
        "power_calibration_hash": "3" * 64,
        "reproducibility_report_hash": "4" * 64,
        "raw_serial_inventory_hash": "5" * 64,
        "raw_parallel_inventory_hash": "6" * 64,
        "deviation_log_hash": "7" * 64,
        "smoke_prerequisite_hash": SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
        "smoke_config_hash": SYMBOLIC_V2_SMOKE_CONFIG_HASH,
        "smoke_reproducibility_hash": "8" * 64,
        "smoke_raw_serial_inventory_hash": "9" * 64,
        "smoke_raw_parallel_inventory_hash": "a" * 64,
        "analysis_code_hash": "b" * 64,
        "run_settings_hash": "c" * 64,
    }
    assert len(calibration_evidence_hash_from_hashes(**hashes)) == 64
    with pytest.raises(ValueError, match="Stage-0"):
        calibration_evidence_hash_from_hashes(
            **{**hashes, "smoke_prerequisite_hash": "d" * 64}
        )
    with pytest.raises(ValueError, match="SHA-256 hash inputs"):
        calibration_evidence_hash_from_hashes(**{**hashes, "config_hash": "not-a-hash"})
    with pytest.raises(ValueError, match="optional provenance"):
        calibration_evidence_hash_from_hashes(
            **{**hashes, "analysis_code_hash": "not-a-hash"}
        )
    with pytest.raises(ValueError, match="record count"):
        calibration_evidence_hash_from_hashes(
            **{**hashes, "canary_detail_record_count": 0}
        )


def test_power_rng_stream_defaults_to_v1_and_separates_v2() -> None:
    arguments = {
        "hypotheses": (_power_hypothesis(),),
        "candidate_environment_counts": (8,),
        "seed": "stream-separation",
        "simulations": 64,
        "center_environment_count": 16,
    }
    default = calibrate_environment_count(**arguments)
    explicit_v1 = calibrate_environment_count(
        **arguments,
        rng_stream="analysis.cluster-power.v1",
    )
    first_v2 = calibrate_environment_count(
        **arguments,
        rng_stream=POWER_RNG_STREAM,
    )
    second_v2 = calibrate_environment_count(
        **arguments,
        rng_stream=POWER_RNG_STREAM,
    )

    assert default == explicit_v1
    assert first_v2 == second_v2
    assert first_v2 != default
    assert POWER_CANDIDATE_ENVIRONMENTS[-1] == 768
    with pytest.raises(ValueError, match="rng_stream"):
        calibrate_environment_count(**arguments, rng_stream="")
