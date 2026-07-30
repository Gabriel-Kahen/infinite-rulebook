from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.analysis.models import AnalysisPhase
from infinite_rulebook.analysis.power import (
    EnvironmentCluster,
    PowerHypothesis,
    calibrate_environment_count,
)
from infinite_rulebook.orchestration.config import (
    EnvironmentKind,
    load_experiment_config,
)
from infinite_rulebook.orchestration.freeze import ConfirmatoryFreezeError
from infinite_rulebook.studies.symbolic_construct_v2 import (
    LEGACY_D6_REPLICATION,
    POWER_CANDIDATE_ENVIRONMENTS,
    POWER_RNG_STREAM,
    PRIMARY_MINIMUM_EFFECTS,
    S2_EARLY,
    S2_LOAD,
    SECONDARY_D12,
    SYMBOLIC_V2_ALGORITHM_MASTER_SEED,
    SYMBOLIC_V2_CALIBRATION_MASTER_SEED,
    SYMBOLIC_V2_COMPONENT_HASH,
    SYMBOLIC_V2_DESIGN_HASH,
    build_symbolic_analysis_plan,
    registration_component_hash,
    registration_component_payload,
    symbolic_v2_design_hash,
    verify_symbolic_calibration_design,
)

_CONFIG = Path(__file__).parents[1] / "configs" / "symbolic-calibration-v2.json"


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
        "287d4c1877b0eaf359389045b64a0adb7955f068c23dfbd37a2cf4586a4d4c4a"
    )
    assert plan.registration_hash == (
        "0df9beb6007949984a13e32df40132516a6c1f6e29b1d0a13ebf9f2aa2ab9fa7"
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
    assert secondary["role"] == "registered-descriptive-load-sweep"
    assert secondary["may_rescue_compound_s2"] is False
    compact = component["compact_canaries"]
    assert compact["gate_count"] == 27
    assert compact["detail_chunk_records"] == 4096
    assert component["power"]["rng_stream"] == POWER_RNG_STREAM


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
