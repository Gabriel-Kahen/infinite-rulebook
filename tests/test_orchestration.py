from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, astuple, replace
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ArtifactStore,
    EventJournal,
    ScientificArtifactError,
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    FeedbackConfig,
    experiment_config_from_dict,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.seeds import SeedBank
from infinite_rulebook.orchestration.semantics import semantic_hashes
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def experiment(*, replicas: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-pilot",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD),),
        checkpoints=CheckpointConfig((0, 2, 3)),
        horizon=3,
        master_seed="test-seed",
        environment_replicas=replicas,
    )


def test_config_is_frozen_versioned_and_strict() -> None:
    config = experiment()
    with pytest.raises(FrozenInstanceError):
        config.horizon = 2  # type: ignore[misc]
    raw = config.resolved_dict()
    assert experiment_config_from_dict(raw) == config
    with pytest.raises(ValueError, match="unknown"):
        experiment_config_from_dict({**raw, "typo": True})
    with pytest.raises(ValueError, match="schema_version"):
        experiment_config_from_dict({**raw, "schema_version": 99})


def test_run_identity_is_independent_of_unrelated_sweep_cells(
    tmp_path: Path,
) -> None:
    base = experiment()
    expanded = replace(
        base,
        environments=(
            EnvironmentConfig(EnvironmentKind.RED_C, projection_size=1),
            *base.environments,
        ),
    )
    base_result = RunExecutor(tmp_path / "base", ExactSymbolicAdapter()).execute(
        base, base.cells()[0]
    )
    ind_cell = next(
        cell
        for cell in expanded.cells()
        if cell.environment.kind is EnvironmentKind.IND
    )
    expanded_result = RunExecutor(
        tmp_path / "expanded", ExactSymbolicAdapter()
    ).execute(expanded, ind_cell)
    assert base_result.run_hash == expanded_result.run_hash
    assert (
        base_result.scientific_content_hash == expanded_result.scientific_content_hash
    )


def test_seed_tree_is_stable_and_streams_are_separate() -> None:
    cell = experiment().cells()[0]
    first = SeedBank("master").for_cell(cell)
    second = SeedBank("master").for_cell(cell)
    assert first == second
    assert len(set(astuple(first))) == len(astuple(first))
    paired = replace(
        cell,
        environment=EnvironmentConfig(EnvironmentKind.TRIVIA),
        agent=AgentConfig(AgentKind.TOTAL_INFORMATION),
    )
    assert SeedBank("master").for_cell(paired) == first


def test_scientific_hash_is_canonical_and_runtime_metadata_is_excluded() -> None:
    left = scientific_hash({"b": 2, "a": 1.0})
    right = scientific_hash({"a": 1.0, "b": 2})
    assert left == right
    semantics = {"environment": scientific_hash("environment")}
    first = ArtifactEnvelope.create(
        "metric",
        semantics,
        {"value": 1.0},
        runtime_metadata={"timestamp": "today", "hardware": "cpu-a"},
    )
    second = ArtifactEnvelope.create(
        "metric",
        semantics,
        {"value": 1.0},
        runtime_metadata={"timestamp": "tomorrow", "hardware": "cpu-b"},
    )
    assert first.scientific_hash == second.scientific_hash


def test_frontier_identity_excludes_feedback_and_irrelevant_augmentation() -> None:
    base = experiment().cells()[0]
    noisy = replace(base, feedback=FeedbackConfig(epsilon=0.2))
    alea = replace(
        base,
        environment=EnvironmentConfig(EnvironmentKind.ALEA, projection_size=1),
    )
    trivia = replace(
        base,
        environment=EnvironmentConfig(EnvironmentKind.TRIVIA, projection_size=1),
    )
    hashes = [semantic_hashes(cell) for cell in (base, noisy, alea, trivia)]
    assert len({item["frontier"] for item in hashes}) == 1
    assert hashes[0]["environment"] != hashes[2]["environment"]


def test_event_journal_rejects_duplicate_content_and_semantic_mismatch(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    semantics = {"environment": scientific_hash("one")}
    journal = EventJournal(store, semantics)
    first = journal.append("round:0", "training", {"value": 1})
    assert journal.append("round:0", "training", {"value": 1}) == first
    assert len(journal.events()) == 1
    with pytest.raises(ScientificArtifactError, match="different content"):
        journal.append("round:0", "training", {"value": 2})
    with pytest.raises(ScientificArtifactError, match="incompatible"):
        read_artifact(
            store.path / "events/00000000.json",
            expected_semantic_hashes={"environment": scientific_hash("two")},
        )


def test_checkpoint_is_side_effect_free_and_frontier_components_persist(
    tmp_path: Path,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter()).execute(
        config, config.cells()[0]
    )
    artifacts = validate_artifact_tree(result.path)
    checkpoints = [
        artifact for artifact in artifacts if artifact.artifact_type == "run-checkpoint"
    ]
    assert checkpoints
    assert all(
        checkpoint.payload["training_state_before"]
        == checkpoint.payload["training_state_after"]
        for checkpoint in checkpoints
    )
    frontier_reference = next(
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "frontier-reference"
    )
    frontier_path = (
        tmp_path / "_frontiers" / frontier_reference.payload["frontier_hash"]
    )
    frontier_artifacts = validate_artifact_tree(frontier_path)
    assert {
        "frontier-witness",
        "frontier-certificate",
        "frontier-diagnostics",
    } <= {artifact.artifact_type for artifact in frontier_artifacts}


def test_interrupted_run_resumes_without_duplicates_or_hash_changes(
    tmp_path: Path,
) -> None:
    config = experiment()
    cell = config.cells()[0]
    interrupted = RunExecutor(tmp_path / "resumed", ExactSymbolicAdapter())
    partial = interrupted.execute(config, cell, stop_after_new_events=1)
    assert not partial.complete
    assert partial.event_count == 1
    resumed = interrupted.execute(config, cell)
    fresh = RunExecutor(tmp_path / "fresh", ExactSymbolicAdapter()).execute(
        config, cell
    )
    assert resumed.complete
    assert resumed.event_count == config.horizon
    assert resumed.scientific_content_hash == fresh.scientific_content_hash
    store = ArtifactStore(resumed.path)
    events = EventJournal(store, semantic_hashes(cell)).events()
    assert [event.sequence for event in events] == list(range(config.horizon))
    assert len({event.event_key for event in events}) == config.horizon


def test_parallel_and_serial_sweeps_are_semantically_equal(tmp_path: Path) -> None:
    config = experiment(replicas=2)
    serial = SweepRunner(RunExecutor(tmp_path / "serial", ExactSymbolicAdapter())).run(
        config
    )
    parallel = SweepRunner(
        RunExecutor(tmp_path / "parallel", ExactSymbolicAdapter())
    ).run(config, max_workers=2)
    assert [(result.run_hash, result.scientific_content_hash) for result in serial] == [
        (result.run_hash, result.scientific_content_hash) for result in parallel
    ]


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter()).execute(
        config, config.cells()[0]
    )
    metric_path = result.path / "metrics.json"
    raw = json.loads(metric_path.read_text())
    raw["payload"]["event_count"] = 999
    metric_path.chmod(0o644)
    metric_path.write_text(json.dumps(raw))
    with pytest.raises(ScientificArtifactError, match="hash mismatch"):
        validate_artifact_tree(result.path)


def test_immutable_artifact_cannot_be_replaced(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    semantics = {"environment": scientific_hash("environment")}
    first = store.write("metric.json", "metric", semantics, {"value": 1})
    assert store.write("metric.json", "metric", semantics, {"value": 1}) == first
    with pytest.raises(ScientificArtifactError, match="refusing to mutate"):
        store.write("metric.json", "metric", semantics, {"value": 2})
