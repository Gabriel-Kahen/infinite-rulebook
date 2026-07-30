from __future__ import annotations

from dataclasses import replace

import pytest

from infinite_rulebook.analysis import AnalysisPhase, load_run_trees
from infinite_rulebook.orchestration.artifacts import validate_artifact_tree
from infinite_rulebook.orchestration.config import (
    SYMBOLIC_ADAPTER_CONTRACT_V2,
    SYMBOLIC_V2_INGESTION_PROBE_EXPERIMENT_NAME,
    load_experiment_config,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.studies.symbolic_registry import registered_symbolic_study
from scripts.benchmark_analysis_scale import build_scale_dataset
from scripts.benchmark_artifact_ingestion import project_report_ingestion
from scripts.generate_ingestion_probe import build_ingestion_probe


@pytest.mark.parametrize(("version", "run_count"), ((1, 36), (2, 48)))
def test_ingestion_probe_preserves_production_shape_and_uses_disjoint_seeds(
    version: int,
    run_count: int,
) -> None:
    calibration = load_experiment_config(
        f"configs/symbolic-calibration-v{version}.json"
    )

    probe = build_ingestion_probe(calibration)
    checked_in = load_experiment_config(
        f"configs/symbolic-artifact-ingestion-probe-v{version}.json"
    )

    assert checked_in == probe
    assert probe.phase == "calibration"
    assert probe.master_seed != calibration.master_seed
    assert probe.algorithm_master_seed != calibration.algorithm_master_seed
    assert probe.environment_replicas == probe.algorithm_replicas == 1
    assert len(probe.cells()) == run_count
    for name in (
        "environments",
        "agents",
        "horizon",
        "checkpoints",
        "feedback",
        "reward",
        "solver",
    ):
        assert getattr(probe, name) == getattr(calibration, name)


def test_v2_ingestion_probe_executes_validates_and_loads_one_cell(tmp_path) -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v2.json")
    study = registered_symbolic_study(calibration.name)
    probe = build_ingestion_probe(calibration)
    settings = probe.resolved_run_settings()

    assert probe.name == SYMBOLIC_V2_INGESTION_PROBE_EXPERIMENT_NAME
    assert settings["adapter_contract"] == SYMBOLIC_ADAPTER_CONTRACT_V2
    assert settings["experiment_name"] == probe.name
    with pytest.raises(ValueError, match="unregistered symbolic study"):
        registered_symbolic_study(probe.name)
    with pytest.raises(ValueError, match="requires phase='calibration'"):
        replace(probe, phase="pilot")

    result = RunExecutor(tmp_path, study.adapter_factory).execute(
        probe,
        probe.cells()[0],
    )
    assert result.complete
    validate_artifact_tree(result.path)
    dataset = load_run_trees(
        (result.path,),
        expected_phase=AnalysisPhase.CALIBRATION,
        expected_freeze_hash=None,
        expected_run_settings=settings,
    )

    assert len(dataset.observations) == len(probe.checkpoints.rounds)
    assert {item.round_index for item in dataset.observations} == set(
        probe.checkpoints.rounds
    )
    assert {
        "post_query_hidden_expected_reward",
        "post_query_mean_hidden_expected_reward",
    } <= dict(dataset.observations[-1].metrics).keys()


def test_ingestion_probe_rejects_a_nonregistered_source_design() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")

    with pytest.raises(ValueError):
        build_ingestion_probe(replace(calibration, horizon=11))


def test_scale_benchmark_models_v2_positive_checkpoint_metrics() -> None:
    calibration = replace(
        load_experiment_config("configs/symbolic-calibration-v2.json"),
        algorithm_replicas=1,
    )
    dataset = build_scale_dataset(
        calibration,
        environment_replicas=1,
        metric_count=29,
        positive_checkpoint_metric_count=31,
    )

    assert {
        len(item.metrics) for item in dataset.observations if item.round_index == 0
    } == {29}
    assert {
        len(item.metrics) for item in dataset.observations if item.round_index > 0
    } == {31}
    with pytest.raises(ValueError, match="cannot be below"):
        build_scale_dataset(
            calibration,
            environment_replicas=1,
            metric_count=31,
            positive_checkpoint_metric_count=29,
        )


def test_ingestion_projection_does_not_scale_fixed_frontier_work() -> None:
    fixed, marginal, residual, projected = project_report_ingestion(
        frontier_validation=10.0,
        frontier_copy=2.0,
        frontier_raw=1.0,
        run_validation=0.1,
        run_raw=0.01,
        run_load=0.2,
        observed_probe=100.0,
        probe_runs=10,
        projected_runs=100,
    )

    assert fixed == 78.0
    assert marginal == pytest.approx(0.62)
    assert residual == pytest.approx(1.58)
    assert projected == pytest.approx(298.0)
