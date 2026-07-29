"""Registered analysis stays deterministic, paired, and phase-safe."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.analysis import (
    Alternative,
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    AnalysisPlan,
    CertifiedFrontier,
    CheckpointObservation,
    ContrastSpec,
    EquivalenceSpec,
    ExpectedGroup,
    GroupSelector,
    Interpolation,
    MarginSource,
    ScalingSpec,
    analysis_plan_from_dict,
    analysis_plan_json,
    build_report,
    evaluate_contrast,
    exact_sign_p_value,
    expected_groups_from_experiment,
    holm_adjust,
    load_analysis_plan,
    load_run_tree,
    load_run_trees,
    parse_analysis_plan_json,
    pool_checkpoints,
)
from infinite_rulebook.orchestration.artifacts import ScientificArtifactError
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    SeedBankIdentities,
    freeze_experiment_config,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.provenance import collect_provenance
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _hash(value: object) -> str:
    return scientific_hash(value, domain="analysis-test")


_FRONTIER = CertifiedFrontier(
    semantic_hash=_hash("frontier"),
    zero_information_reward=0.0,
    maximum_reward=2.0,
    points=((0.0, 0.0, 0.0), (1.0, 0.5, 1.0), (2.0, 2.0, 3.0)),
)


def _observation(
    *,
    environment: str,
    agent: str = "reward",
    environment_replica: int = 0,
    algorithm_replica: int = 0,
    round_index: int = 2,
    reward: float = 1.0,
    relevant: float = 0.5,
    shared_core: float = 0.0,
    distractor: float = 0.0,
    condition_variant: int = 0,
    agent_variant: int = 0,
    phase: AnalysisPhase = AnalysisPhase.PILOT,
    frozen: bool = False,
    freeze_hash: str | None = None,
) -> CheckpointObservation:
    condition_hash = _hash((environment, condition_variant))
    identity = (
        environment,
        agent,
        environment_replica,
        algorithm_replica,
        round_index,
        condition_variant,
        agent_variant,
        phase,
    )
    semantics = (
        ("action", _hash("action")),
        ("environment", _hash((environment, condition_variant))),
        ("feedback", _hash("feedback")),
        ("frontier", _FRONTIER.semantic_hash),
        ("reward", _hash("reward")),
    )
    return CheckpointObservation(
        run_hash=_hash(("run", identity)),
        content_hash=_hash(("content", identity, reward, relevant, distractor)),
        phase=phase,
        confirmatory_frozen=frozen,
        freeze_hash=freeze_hash,
        analysis_registration_hash=None,
        condition_hash=condition_hash,
        environment_kind=environment,
        agent_kind=agent,
        agent_hash=_hash((agent, agent_variant)),
        environment_replica=environment_replica,
        algorithm_replica=algorithm_replica,
        round_index=round_index,
        metrics=(
            ("distractor_information_nats", distractor),
            ("expected_reward", reward),
            ("information.shared_core_nats", shared_core),
            ("information.reward_relevant_nats", relevant - shared_core),
            ("relevant_information_nats", relevant),
            ("total_information_nats", relevant + distractor),
        ),
        semantic_hashes=semantics,
        frontier=_FRONTIER,
    )


def _paired_dataset(
    differences: tuple[float, ...],
    *,
    algorithm_replicas: int = 1,
    rounds: tuple[int, ...] = (2,),
) -> AnalysisDataset:
    observations = []
    for environment_replica, difference in enumerate(differences):
        for algorithm_replica in range(algorithm_replicas):
            for round_index in rounds:
                scale = round_index / max(rounds)
                observations.extend(
                    (
                        _observation(
                            environment="IND",
                            environment_replica=environment_replica,
                            algorithm_replica=algorithm_replica,
                            round_index=round_index,
                            reward=1.0 + difference * scale,
                            relevant=1.0 + difference * scale,
                        ),
                        _observation(
                            environment="RED-C",
                            environment_replica=environment_replica,
                            algorithm_replica=algorithm_replica,
                            round_index=round_index,
                            reward=1.0,
                            relevant=1.0,
                            shared_core=0.75,
                        ),
                    )
                )
    return AnalysisDataset(tuple(observations))


def _contrast(*, checkpoint: int = 2) -> ContrastSpec:
    return ContrastSpec(
        "ind-over-red",
        "relevant_information_nats",
        GroupSelector(environment_kind="IND", agent_kind="reward"),
        GroupSelector(environment_kind="RED-C", agent_kind="reward"),
        checkpoint,
        Alternative.GREATER,
    )


def test_dataset_scientific_hash_is_computed_once_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.analysis.models as model_module

    calls = 0
    original = model_module.scientific_hash

    def counted(value: object, *, domain: str = "payload") -> str:
        nonlocal calls
        if domain == "analysis-dataset":
            calls += 1
        return original(value, domain=domain)

    monkeypatch.setattr(model_module, "scientific_hash", counted)
    dataset = _paired_dataset((1.0, -1.0))

    first = dataset.scientific_hash
    second = dataset.scientific_hash

    assert first == second
    assert calls == 1


def test_dataset_duplicate_check_handles_large_round_indices_without_bitsets() -> None:
    observation = replace(_observation(environment="IND"), round_index=10**9)

    dataset = AnalysisDataset((observation,))

    assert dataset.observations[0].round_index == 10**9


def test_pooling_uses_complete_registered_groups_and_pooled_frontier_reward() -> None:
    dataset = AnalysisDataset(
        (
            _observation(environment="IND", environment_replica=0, reward=0.0),
            _observation(environment="IND", environment_replica=1, reward=2.0),
            _observation(environment="RED-C", environment_replica=0, reward=1.0),
        )
    )

    pools = pool_checkpoints(dataset)
    ind = next(pool for pool in pools if pool.key.environment_kind == "IND")

    assert ind.metric("expected_reward").mean == 1.0
    assert ind.bit_equivalent_lower_nats == 0.5
    assert ind.bit_equivalent_upper_nats == 1.0
    assert (
        ind.bit_equivalent_lower_nats
        != (_FRONTIER.lookup(0.0)[0] + _FRONTIER.lookup(2.0)[0]) / 2.0
    )


def test_pooling_rejects_incompatible_metric_schemas_after_transposition() -> None:
    first = _observation(environment="IND", environment_replica=0)
    second = replace(
        _observation(environment="IND", environment_replica=1),
        metrics=(("expected_reward", 1.0), ("replacement_metric", 0.5)),
    )

    with pytest.raises(AnalysisError, match="incompatible metric schemas"):
        pool_checkpoints(AnalysisDataset((first, second)))


def test_crossed_algorithm_cells_are_clustered_by_environment_replica() -> None:
    dataset = _paired_dataset(
        (1.0, 1.0, 1.0, 1.0, -1.0),
        algorithm_replicas=10,
    )

    result = evaluate_contrast(dataset, _contrast(), interval_alpha=0.05)

    assert result.pair_count == 5
    assert result.cell_pair_count == 50
    assert result.differences == (1.0, 1.0, 1.0, 1.0, -1.0)
    assert result.unadjusted_p_value == 0.1875
    assert result.unadjusted_p_value > 0.05
    ind_pool = next(
        pool for pool in pool_checkpoints(dataset) if pool.key.environment_kind == "IND"
    )
    summary = ind_pool.metric("relevant_information_nats")
    assert summary.count == 5
    assert summary.cell_count == 50
    assert summary.algorithm_replicas_per_environment == 10


def test_holm_adjustment_is_step_down_monotone_and_order_invariant() -> None:
    left = holm_adjust((("b", 0.04), ("a", 0.01), ("c", 0.03)), alpha=0.05)
    right = holm_adjust((("c", 0.03), ("a", 0.01), ("b", 0.04)), alpha=0.05)

    assert left == right
    assert {item.name: item.adjusted_p_value for item in left} == {
        "a": 0.03,
        "b": 0.06,
        "c": 0.06,
    }
    assert [item.name for item in left if item.reject_null] == ["a"]


def test_exact_sign_test_counts_boundary_ties_against_rejection() -> None:
    values = (1.0,) * 5 + (0.0,) * 95

    assert (
        exact_sign_p_value(
            values,
            null=0.0,
            alternative=Alternative.GREATER,
        )
        > 0.99
    )
    assert (
        exact_sign_p_value(
            (0.5,) * 100,
            null=0.5,
            alternative=Alternative.LESS,
        )
        > 0.99
    )


def test_symbolic_scaling_defaults_to_discrete_left_hold_weighting() -> None:
    dataset = _paired_dataset((2.0,) * 6, rounds=(0, 1, 2))
    report = build_report(
        dataset,
        AnalysisPlan(
            "discrete scaling",
            AnalysisPhase.PILOT,
            scalings=(
                ScalingSpec(
                    "symbolic-growth",
                    "relevant_information_nats",
                    GroupSelector(environment_kind="IND", agent_kind="reward"),
                    horizon=2,
                ),
            ),
        ),
    )

    summary = report.scaling[0]
    assert summary.interpolation is Interpolation.LEFT_HOLD
    assert summary.elapsed_weighted_average == 1.5


def test_equivalence_uses_frozen_margin_and_separate_registered_family() -> None:
    dataset = _paired_dataset((0.0,) * 6)
    equivalence = EquivalenceSpec(
        "ind-red-equivalent",
        "expected_reward",
        GroupSelector(environment_kind="IND", agent_kind="reward"),
        GroupSelector(environment_kind="RED-C", agent_kind="reward"),
        2,
        margin=0.5,
        margin_source=MarginSource.CALIBRATION,
        margin_provenance_hash=_hash("calibration"),
    )
    report = build_report(
        dataset,
        AnalysisPlan(
            "pilot-analysis",
            AnalysisPhase.PILOT,
            equivalences=(equivalence,),
        ),
    )

    result = report.equivalences[0]
    assert result.lower_tost_p_value == 1 / 64
    assert result.upper_tost_p_value == 1 / 64
    assert report.family_decisions == ()
    assert report.equivalence_decisions[0].reject_null


def test_report_serialization_and_scaling_are_deterministic() -> None:
    dataset = _paired_dataset((0.2,) * 6, rounds=(0, 1, 2, 4))
    reverse = AnalysisDataset(tuple(reversed(dataset.observations)))
    equivalence = EquivalenceSpec(
        "reward-match",
        "expected_reward",
        GroupSelector(environment_kind="IND", agent_kind="reward"),
        GroupSelector(environment_kind="RED-C", agent_kind="reward"),
        4,
        margin=0.5,
        margin_source=MarginSource.PILOT,
        margin_provenance_hash=_hash("margin"),
    )
    plan = AnalysisPlan(
        "deterministic pilot report",
        AnalysisPhase.PILOT,
        contrasts=(_contrast(checkpoint=4),),
        equivalences=(equivalence,),
        scalings=(
            ScalingSpec(
                "ind-growth",
                "relevant_information_nats",
                GroupSelector(environment_kind="IND", agent_kind="reward"),
                horizon=4,
                interpolation=Interpolation.LINEAR,
            ),
            ScalingSpec(
                "ind-bit-growth",
                "bit_equivalent_lower_nats",
                GroupSelector(environment_kind="IND", agent_kind="reward"),
                horizon=4,
            ),
        ),
    )

    first = build_report(dataset, plan)
    second = build_report(reverse, plan)

    assert first.scientific_hash == second.scientific_hash
    assert first.to_payload() == second.to_payload()
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.scaling[0].dyadic_slopes
    assert '"scientific_hash":' in first.to_json()
    assert "seedwise nonlinear ratios are not averaged" in first.to_markdown()


def test_analysis_plan_json_is_strict_hashed_and_loadable(tmp_path: Path) -> None:
    plan = AnalysisPlan(
        "checked-in plan",
        AnalysisPhase.CALIBRATION,
        contrasts=(_contrast(),),
        scalings=(
            ScalingSpec(
                "growth",
                "expected_reward",
                GroupSelector(environment_kind="IND", agent_kind="reward"),
                horizon=2,
            ),
        ),
    )
    encoded = analysis_plan_json(plan)
    path = tmp_path / "analysis-plan.json"
    path.write_text(encoded, encoding="utf-8")

    assert parse_analysis_plan_json(encoded) == plan
    assert load_analysis_plan(path) == plan
    assert analysis_plan_json(load_analysis_plan(path)) == encoded

    tampered = json.loads(encoded)
    tampered["family_alpha"] = 0.1
    with pytest.raises(AnalysisError, match="hash mismatch"):
        analysis_plan_from_dict(tampered)
    unknown = json.loads(encoded)
    unknown["post_hoc_threshold"] = 1.0
    with pytest.raises(AnalysisError, match="exactly"):
        analysis_plan_from_dict(unknown)
    duplicate = encoded.replace(
        "{\n",
        '{\n  "name": "duplicate",\n',
        1,
    )
    with pytest.raises(AnalysisError, match="repeats key"):
        parse_analysis_plan_json(duplicate)
    nonfinite = encoded.replace('"family_alpha": 0.05', '"family_alpha": NaN')
    with pytest.raises(AnalysisError, match="non-finite"):
        parse_analysis_plan_json(nonfinite)


def test_phase_and_margin_leakage_are_rejected() -> None:
    pilot = _observation(environment="IND")
    with pytest.raises(AnalysisError, match="confirmatory bindings"):
        replace(pilot, analysis_registration_hash=_hash("post-hoc-registration"))
    confirmatory = replace(
        pilot,
        run_hash=_hash("confirmatory-run"),
        phase=AnalysisPhase.CONFIRMATORY,
        confirmatory_frozen=True,
        freeze_hash=_hash("freeze"),
        analysis_registration_hash=_hash("registration"),
    )
    with pytest.raises(AnalysisError, match="cannot mix"):
        AnalysisDataset((pilot, confirmatory))
    with pytest.raises(AnalysisError, match="frozen external"):
        AnalysisPlan("bad", AnalysisPhase.CONFIRMATORY)
    with pytest.raises(TypeError, match="MarginSource"):
        EquivalenceSpec(
            "leak",
            "expected_reward",
            GroupSelector(environment_kind="IND"),
            GroupSelector(environment_kind="RED-C"),
            2,
            0.1,
            AnalysisPhase.CONFIRMATORY,
            _hash("leaked-margin"),
        )

    dataset = AnalysisDataset((confirmatory,))
    plan = AnalysisPlan(
        "sealed",
        AnalysisPhase.CONFIRMATORY,
        frozen=True,
        freeze_hash=_hash("other-freeze"),
        expected_groups=(
            ExpectedGroup(
                condition_hash=confirmatory.condition_hash,
                agent_hash=confirmatory.agent_hash,
                environment_kind="IND",
                agent_kind="reward",
                checkpoints=(2,),
                environment_replicas=1,
                algorithm_replicas=1,
            ),
        ),
    )
    with pytest.raises(AnalysisError, match="freeze seal"):
        build_report(dataset, plan)


def test_confirmatory_registration_and_full_inventory_are_mandatory() -> None:
    freeze_hash = _hash("confirmatory-freeze")
    expected_groups = tuple(
        sorted(
            ExpectedGroup(
                condition_hash=_hash((environment, 0)),
                agent_hash=_hash(("reward", 0)),
                environment_kind=environment,
                agent_kind="reward",
                checkpoints=(2,),
                environment_replicas=2,
                algorithm_replicas=1,
            )
            for environment in ("IND", "RED-C")
        )
    )
    plan = AnalysisPlan(
        "registered confirmatory analysis",
        AnalysisPhase.CONFIRMATORY,
        contrasts=(_contrast(),),
        expected_groups=expected_groups,
        frozen=True,
        freeze_hash=freeze_hash,
    )
    other_seal = replace(plan, freeze_hash=_hash("another-seal"))
    assert other_seal.registration_hash == plan.registration_hash
    altered = replace(
        plan,
        contrasts=(replace(_contrast(), null_margin=0.1),),
    )
    assert altered.registration_hash != plan.registration_hash
    pilot = _paired_dataset((0.5, 0.5))
    observations = tuple(
        replace(
            item,
            phase=AnalysisPhase.CONFIRMATORY,
            confirmatory_frozen=True,
            freeze_hash=freeze_hash,
            analysis_registration_hash=plan.registration_hash,
        )
        for item in pilot.observations
    )
    dataset = AnalysisDataset(observations)

    report = build_report(dataset, plan)

    assert report.plan.registration_hash == plan.registration_hash
    with pytest.raises(AnalysisError, match="sealed AnalysisPlan"):
        evaluate_contrast(dataset, _contrast(), interval_alpha=0.05)
    selected = AnalysisDataset(
        tuple(item for item in observations if item.environment_replica == 0)
    )
    with pytest.raises(AnalysisError, match="frozen run inventory"):
        build_report(selected, plan)
    wrong_registration = AnalysisDataset(
        tuple(
            replace(item, analysis_registration_hash=_hash("wrong-registration"))
            for item in observations
        )
    )
    with pytest.raises(AnalysisError, match="registered analysis hash"):
        build_report(wrong_registration, plan)


def test_pairing_ambiguity_missing_cells_and_seedwise_inversion_are_rejected() -> None:
    incomplete = list(_paired_dataset((1.0, 1.0), algorithm_replicas=2).observations)
    incomplete.pop()
    with pytest.raises(AnalysisError, match="unmatched algorithm"):
        evaluate_contrast(
            AnalysisDataset(tuple(incomplete)),
            _contrast(),
            interval_alpha=0.05,
        )

    variants = AnalysisDataset(
        (
            _observation(environment="IND", condition_variant=0),
            _observation(environment="IND", condition_variant=1),
            _observation(environment="RED-C"),
        )
    )
    with pytest.raises(AnalysisError, match="ambiguous"):
        evaluate_contrast(variants, _contrast(), interval_alpha=0.05)

    agent_variants = AnalysisDataset(
        (
            _observation(environment="IND", agent_variant=0),
            _observation(environment="IND", agent_variant=1),
            _observation(environment="RED-C"),
        )
    )
    with pytest.raises(AnalysisError, match="ambiguous"):
        evaluate_contrast(agent_variants, _contrast(), interval_alpha=0.05)

    with pytest.raises(AnalysisError, match="after pooling"):
        evaluate_contrast(
            _paired_dataset((1.0,)),
            replace(_contrast(), metric="bit_equivalent_lower_nats"),
            interval_alpha=0.05,
        )


def test_loader_uses_existing_full_tree_validator_and_relevant_core_sum(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    cell = next(
        item for item in config.cells() if item.environment.kind.value == "RED-C"
    )
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)

    observations = load_run_tree(
        result.path,
        expected_phase=AnalysisPhase.PILOT,
    )

    assert observations
    assert tuple(item.round_index for item in observations) == (0, 2, 4)
    final = observations[-1]
    expected = final.metric("information.reward_relevant_nats") + final.metric(
        "information.shared_core_nats"
    )
    assert math.isclose(final.metric("relevant_information_nats"), expected)
    assert final.frontier.semantic_hash == dict(final.semantic_hashes)["frontier"]
    expected_groups = expected_groups_from_experiment(config)
    assert any(
        group.condition_hash == final.condition_hash
        and group.agent_hash == final.agent_hash
        for group in expected_groups
    )

    copied = result.path.parent / ("0" * 64)
    shutil.copytree(result.path, copied)
    with pytest.raises(ScientificArtifactError, match="run identity"):
        load_run_tree(copied)


def test_loader_validates_each_shared_frontier_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.orchestration.artifacts as artifact_module

    base = load_experiment_config("configs/pilot-foundation.json")
    config = replace(
        base,
        environments=(base.environments[0],),
        agents=(base.agents[0],),
        environment_replicas=2,
    )
    executor = RunExecutor(tmp_path, ExactSymbolicAdapter)
    roots = tuple(executor.execute(config, cell).path for cell in config.cells())
    calls = 0
    original = artifact_module._validate_frontier_records

    def counted(*args: object, **kwargs: object) -> None:
        nonlocal calls
        records = args[1]
        if any(
            artifact.artifact_type == "frontier-manifest"
            for _, artifact in records  # type: ignore[union-attr]
        ):
            calls += 1
        original(*args, **kwargs)

    monkeypatch.setattr(artifact_module, "_validate_frontier_records", counted)
    dataset = load_run_trees(roots, expected_phase=AnalysisPhase.PILOT)

    assert dataset.observations
    assert calls == 1


def test_loader_keeps_public_and_hidden_reward_components_separate(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-smoke.json")
    cell = next(
        item
        for item in config.cells()
        if item.environment.kind.value == "PUBLIC-C"
        and item.agent.kind.value == "reward"
    )
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)

    final = load_run_tree(result.path)[-1]

    assert final.metric("expected_reward") == pytest.approx(
        final.metric("hidden_expected_reward") + final.metric("public_reward")
    )


def test_generic_self_seal_cannot_execute_as_the_registered_symbolic_study(
    tmp_path: Path,
) -> None:
    calibration = ExperimentConfig(
        name="analysis-calibration",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="calibration-analysis-seed",
        algorithm_master_seed="shared-analysis-algorithm-seed",
        phase="calibration",
    )
    expected_groups = expected_groups_from_experiment(calibration)
    placeholder = AnalysisPlan(
        "confirmatory analysis",
        AnalysisPhase.CONFIRMATORY,
        expected_groups=expected_groups,
        frozen=True,
        freeze_hash="0" * 64,
    )
    confirmatory_seed = "confirmatory-analysis-seed"
    sealed = freeze_experiment_config(
        calibration,
        name="analysis-confirmatory",
        confirmatory_master_seed=confirmatory_seed,
        calibration_evidence_hash=_hash("calibration-evidence"),
        analysis_contract="registered-analysis-report.v1",
        analysis_version=placeholder.registration_hash,
        analysis_code_hash=collect_provenance().analysis_code_hash,
        dependency_lock_hash=collect_provenance().dependency_lock_hash,
        environment_digest=collect_provenance().environment_digest,
        seed_banks=SeedBankIdentities.bind(
            calibration_master_seed=calibration.master_seed,
            confirmatory_master_seed=confirmatory_seed,
            algorithm_master_seed=calibration.algorithm_master_seed,
        ),
        tolerances={"frontier_gap_nats": 1e-7},
        margins={"equivalence_nats": 0.1},
    )
    assert sealed.confirmatory_freeze is not None
    plan = replace(
        placeholder,
        freeze_hash=sealed.confirmatory_freeze.seal_hash,
    )
    assert plan.freeze_hash == sealed.confirmatory_freeze.seal_hash
    with pytest.raises(ConfirmatoryFreezeError, match="exact symbolic v1"):
        RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
            sealed,
            sealed.cells()[0],
        )
