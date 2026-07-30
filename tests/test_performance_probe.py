from __future__ import annotations

from dataclasses import replace

import pytest

from infinite_rulebook.orchestration.config import load_experiment_config
from scripts.benchmark_artifact_ingestion import project_report_ingestion
from scripts.generate_ingestion_probe import build_ingestion_probe


def test_ingestion_probe_preserves_production_shape_and_uses_disjoint_seeds() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")

    probe = build_ingestion_probe(calibration)
    checked_in = load_experiment_config(
        "configs/symbolic-artifact-ingestion-probe-v1.json"
    )

    assert checked_in == probe
    assert probe.phase == "calibration"
    assert probe.master_seed != calibration.master_seed
    assert probe.algorithm_master_seed != calibration.algorithm_master_seed
    assert probe.environment_replicas == probe.algorithm_replicas == 1
    assert len(probe.cells()) == 36
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


def test_ingestion_probe_rejects_a_nonregistered_source_design() -> None:
    calibration = load_experiment_config("configs/symbolic-calibration-v1.json")

    with pytest.raises(ValueError):
        build_ingestion_probe(replace(calibration, horizon=11))


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
