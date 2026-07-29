from __future__ import annotations

from dataclasses import replace

import pytest

from infinite_rulebook.orchestration.config import load_experiment_config
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
