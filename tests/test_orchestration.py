from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, astuple, replace
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ArtifactStore,
    ArtifactValidationSession,
    EventJournal,
    ScientificArtifactError,
    read_artifact,
    validate_artifact_tree,
    write_frontier_bundle,
)
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    FeedbackConfig,
    RunCell,
    experiment_config_from_dict,
    load_experiment_config,
)
from infinite_rulebook.orchestration.freeze import ConfirmatoryFreezeError
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


def test_boolean_artifact_schema_version_is_rejected() -> None:
    envelope = ArtifactEnvelope.create(
        "metric",
        {"environment": "a" * 64},
        {"value": 1},
    )
    raw = envelope.to_dict()
    raw["schema_version"] = True
    raw["scientific_hash"] = scientific_hash(
        {
            "artifact_type": raw["artifact_type"],
            "schema_version": True,
            "semantic_hashes": raw["semantic_hashes"],
            "payload": raw["payload"],
        },
        domain="artifact-envelope",
    )

    with pytest.raises(ScientificArtifactError, match="schema"):
        ArtifactEnvelope.from_dict(raw)


@pytest.mark.parametrize("name", (".", "..", "nested/name", r"nested\name"))
def test_experiment_name_must_be_one_safe_path_component(name: str) -> None:
    with pytest.raises(ValueError, match="path component"):
        replace(experiment(), name=name)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"name":"first","name":"second"}', "repeats key"),
        ('{"name":NaN}', "non-finite"),
        ('{"name":Infinity}', "non-finite"),
    ],
)
def test_config_loader_rejects_noncanonical_json(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


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
    base_result = RunExecutor(tmp_path / "base", ExactSymbolicAdapter).execute(
        base, base.cells()[0]
    )
    ind_cell = next(
        cell
        for cell in expanded.cells()
        if cell.environment.kind is EnvironmentKind.IND
    )
    expanded_result = RunExecutor(tmp_path / "expanded", ExactSymbolicAdapter).execute(
        expanded, ind_cell
    )
    assert base_result.run_hash == expanded_result.run_hash
    assert (
        base_result.scientific_content_hash == expanded_result.scientific_content_hash
    )


def test_executor_rejects_cell_outside_experiment_inventory(
    tmp_path: Path,
) -> None:
    config = experiment()
    foreign = replace(config.cells()[0], environment_replica=1)
    with pytest.raises(ValueError, match="experiment inventory"):
        RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, foreign)
    assert not (tmp_path / config.name).exists()


def test_executor_rejects_unregistered_adapter_contract(tmp_path: Path) -> None:
    class WrongContractAdapter(ExactSymbolicAdapter):
        contract_version = "unregistered.v1"

    config = experiment()
    with pytest.raises(ValueError, match="configured contract"):
        RunExecutor(tmp_path, WrongContractAdapter).execute(
            config,
            config.cells()[0],
        )


def test_calibration_rejects_adapter_substitution(tmp_path: Path) -> None:
    class LookalikeAdapter(ExactSymbolicAdapter):
        pass

    config = replace(experiment(), phase="calibration")
    with pytest.raises(ConfirmatoryFreezeError, match="registered exact adapter"):
        RunExecutor(tmp_path, LookalikeAdapter).execute(config, config.cells()[0])


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


def test_algorithm_nuisance_bank_can_be_shared_across_phase_seed_banks() -> None:
    cell = experiment().cells()[0]
    calibration = SeedBank("calibration", "shared-algorithms").for_cell(cell)
    confirmatory = SeedBank("confirmatory", "shared-algorithms").for_cell(cell)

    assert calibration.algorithm == confirmatory.algorithm
    assert calibration.deployment == confirmatory.deployment
    assert calibration.environment != confirmatory.environment
    assert calibration.query_observation != confirmatory.query_observation
    assert calibration.evaluation != confirmatory.evaluation


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


def test_scientific_hash_type_tags_cannot_collide_with_user_mappings() -> None:
    assert scientific_hash(1.0) != scientific_hash({"$float": "0x1.0000000000000p+0"})
    assert scientific_hash(b"abc") != scientific_hash({"$bytes": "616263"})
    assert scientific_hash(Path("x")) != scientific_hash({"$path": "x"})


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
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    artifacts = validate_artifact_tree(result.path)
    resolved = next(
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "resolved-run-config"
    )
    assert {
        "analysis_code_hash",
        "code_commit",
        "dependency_lock_hash",
        "dirty_tree_hash",
        "environment_digest",
    } <= set(resolved.payload["provenance"])
    manifest = next(
        artifact for artifact in artifacts if artifact.artifact_type == "run-manifest"
    )
    assert {
        "hardware",
        "timestamp_utc",
        "wall_time_seconds",
    } <= set(manifest.runtime_metadata)
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
    final = max(checkpoints, key=lambda checkpoint: checkpoint.payload["round"])
    assert (
        final.payload["result"]["support"]["deployment_support"]
        <= config.environments[0].projection_size
    )
    curve = next(
        artifact
        for artifact in frontier_artifacts
        if artifact.artifact_type == "frontier-curve"
    )
    assert (
        final.payload["result"]["expected_reward"]
        <= curve.payload["problem"]["feasible_reward_range"][1]
    )


def test_checkpoint_schedule_cannot_change_future_training_events(
    tmp_path: Path,
) -> None:
    frequent = experiment()
    final_only = replace(
        frequent,
        checkpoints=CheckpointConfig((frequent.horizon,)),
    )
    frequent_result = RunExecutor(tmp_path / "frequent", ExactSymbolicAdapter).execute(
        frequent, frequent.cells()[0]
    )
    final_result = RunExecutor(tmp_path / "final", ExactSymbolicAdapter).execute(
        final_only, final_only.cells()[0]
    )

    def events(result_path: Path) -> tuple[object, ...]:
        store = ArtifactStore(result_path)
        hashes = store.read("config.resolved.json").semantic_hashes
        return tuple(event.payload for event in EventJournal(store, hashes).events())

    assert events(frequent_result.path) == events(final_result.path)


def test_interrupted_run_resumes_without_duplicates_or_hash_changes(
    tmp_path: Path,
) -> None:
    config = experiment()
    cell = config.cells()[0]
    interrupted = RunExecutor(tmp_path / "resumed", ExactSymbolicAdapter)
    partial = interrupted.execute(config, cell, stop_after_new_events=1)
    assert not partial.complete
    assert partial.event_count == 1
    resumed = interrupted.execute(
        config,
        cell,
        runtime_metadata={"hardware": "resume-host"},
    )
    fresh = RunExecutor(tmp_path / "fresh", ExactSymbolicAdapter).execute(
        config,
        cell,
        runtime_metadata={"hardware": "fresh-host"},
    )
    assert resumed.complete
    assert resumed.event_count == config.horizon
    assert resumed.scientific_content_hash == fresh.scientific_content_hash
    store = ArtifactStore(resumed.path)
    recorded_hashes = store.read("config.resolved.json").semantic_hashes
    events = EventJournal(store, recorded_hashes).events()
    assert [event.sequence for event in events] == list(range(config.horizon))
    assert len({event.event_key for event in events}) == config.horizon
    resumed_manifest = ArtifactStore(resumed.path).read("manifest.json")
    fresh_manifest = ArtifactStore(fresh.path).read("manifest.json")
    assert resumed_manifest.scientific_hash == fresh_manifest.scientific_hash
    assert resumed_manifest.runtime_metadata != fresh_manifest.runtime_metadata


def test_run_validation_requires_its_external_frontier(tmp_path: Path) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    reference = ArtifactStore(result.path).read("frontier-reference.json")
    shutil.rmtree(tmp_path / "_frontiers" / reference.payload["frontier_hash"])
    with pytest.raises(ScientificArtifactError, match="does not exist"):
        validate_artifact_tree(result.path)


def test_validation_session_never_trusts_a_different_artifact_root(
    tmp_path: Path,
) -> None:
    config = experiment()
    first = RunExecutor(tmp_path / "first", ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    second = RunExecutor(tmp_path / "second", ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    session = ArtifactValidationSession()
    validate_artifact_tree(first.path, session=session)
    reference = ArtifactStore(second.path).read("frontier-reference.json")
    shutil.rmtree(
        tmp_path / "second" / "_frontiers" / reference.payload["frontier_hash"]
    )

    with pytest.raises(ScientificArtifactError, match="does not exist"):
        validate_artifact_tree(second.path, session=session)


def test_incomplete_artifact_tree_is_not_publishable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "partial")
    semantics = {"environment": scientific_hash("environment")}
    store.write("metrics.json", "run-metrics", semantics, {"value": 1})
    with pytest.raises(ScientificArtifactError, match="recognized manifest"):
        validate_artifact_tree(store.path)
    with pytest.raises(ScientificArtifactError, match="incomplete run"):
        store.finalize(semantics)


@pytest.mark.parametrize(
    ("extra_path", "artifact_type"),
    [
        ("extra.json", "unexpected"),
        ("other-metrics.json", "run-metrics"),
        ("checkpoints/not-a-checkpoint.json", "run-checkpoint"),
    ],
)
def test_invalid_run_inventory_never_publishes_a_manifest(
    tmp_path: Path,
    extra_path: str,
    artifact_type: str,
) -> None:
    store = ArtifactStore(tmp_path / "partial")
    semantics = {"environment": scientific_hash("environment")}
    store.write("config.resolved.json", "resolved-run-config", semantics, {})
    store.write("frontier-reference.json", "frontier-reference", semantics, {})
    store.write("events/00000000.json", "training-event", semantics, {})
    store.write("checkpoints/00000000.json", "run-checkpoint", semantics, {})
    store.write("metrics.json", "run-metrics", semantics, {})
    store.write(extra_path, artifact_type, semantics, {})

    with pytest.raises(ScientificArtifactError, match="cannot finalize"):
        store.finalize(semantics)
    assert not (store.path / "manifest.json").exists()


def test_invalid_frontier_inventory_never_publishes_a_manifest(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "frontier")
    semantics = {"frontier": scientific_hash("frontier")}
    store.write("extra.json", "frontier-diagnostics", semantics, {})

    with pytest.raises(ScientificArtifactError, match="unexpected artifact inventory"):
        write_frontier_bundle(
            store,
            semantics,
            curve={"points": []},
            witnesses={},
            certificates={},
            diagnostics={},
        )
    assert not (store.path / "frontier/manifest.json").exists()


@pytest.mark.parametrize(
    ("extra_path", "artifact_type"),
    [
        ("events/00000003.json", "training-event"),
        ("checkpoints/00000001.json", "run-checkpoint"),
    ],
)
def test_extra_validly_named_run_member_never_publishes_a_manifest(
    tmp_path: Path,
    extra_path: str,
    artifact_type: str,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    manifest_path = result.path / "manifest.json"
    manifest_path.unlink()
    store = ArtifactStore(result.path)
    semantics = store.read("config.resolved.json").semantic_hashes
    store.write(extra_path, artifact_type, semantics, {"extra": True})

    with pytest.raises(ScientificArtifactError, match="extra event/checkpoint"):
        store.finalize(semantics)
    assert not manifest_path.exists()


def test_nested_manifest_name_never_bypasses_finalization_inventory(
    tmp_path: Path,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    manifest_path = result.path / "manifest.json"
    manifest_path.unlink()
    store = ArtifactStore(result.path)
    semantics = store.read("config.resolved.json").semantic_hashes
    store.write(
        "events/manifest.json",
        "training-event",
        semantics,
        {"unexpected": True},
    )

    with pytest.raises(ScientificArtifactError, match="training-event inventory"):
        store.finalize(semantics)
    assert not manifest_path.exists()


@pytest.mark.parametrize("extra_kind", ("file", "directory"))
def test_artifact_validation_rejects_uninventoried_tree_members(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    if extra_kind == "file":
        (result.path / "notes.txt").write_text("not in the manifest", encoding="utf-8")
    else:
        (result.path / "empty-extra").mkdir()

    with pytest.raises(
        ScientificArtifactError, match=r"unexpected member|empty directory"
    ):
        validate_artifact_tree(result.path)


def test_run_validation_parses_each_run_artifact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.orchestration.artifacts as artifact_module

    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    expected = {
        path.resolve() for path in result.path.rglob("*.json") if path.is_file()
    }
    reads: Counter[Path] = Counter()
    original = artifact_module.read_artifact

    def counted(path: str | Path, **kwargs: object):
        resolved = Path(path).resolve()
        if resolved in expected:
            reads[resolved] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(artifact_module, "read_artifact", counted)
    validate_artifact_tree(result.path)

    assert set(reads) == expected
    assert set(reads.values()) == {1}


def test_frontier_validation_requires_complete_points_and_valid_duals(
    tmp_path: Path,
) -> None:
    cell = experiment().cells()[0]
    frontier = ExactSymbolicAdapter().frontier(cell)
    hashes = {"frontier": scientific_hash("complete-frontier")}
    incomplete = ArtifactStore(tmp_path / "incomplete")
    witnesses = dict(frontier["witnesses"])
    certificates = dict(frontier["certificates"])
    witnesses.pop("point-001")
    certificates.pop("point-001")
    write_frontier_bundle(
        incomplete,
        hashes,
        curve=frontier["curve"],
        witnesses=witnesses,
        certificates=certificates,
        diagnostics=frontier["diagnostics"],
    )
    with pytest.raises(ScientificArtifactError, match="differ"):
        validate_artifact_tree(incomplete.path)

    invalid_dual = ArtifactStore(tmp_path / "invalid-dual")
    certificates = {
        name: dict(certificate)
        for name, certificate in frontier["certificates"].items()
    }
    certificates["point-001"]["dual_beta"] = 999.0
    write_frontier_bundle(
        invalid_dual,
        hashes,
        curve=frontier["curve"],
        witnesses=frontier["witnesses"],
        certificates=certificates,
        diagnostics=frontier["diagnostics"],
    )
    with pytest.raises(ScientificArtifactError, match="certificate"):
        validate_artifact_tree(invalid_dual.path)


def test_parallel_and_serial_sweeps_are_semantically_equal(tmp_path: Path) -> None:
    config = experiment(replicas=2)
    serial = SweepRunner(RunExecutor(tmp_path / "serial", ExactSymbolicAdapter)).run(
        config
    )
    parallel = SweepRunner(
        RunExecutor(tmp_path / "parallel", ExactSymbolicAdapter)
    ).run(config, max_workers=2)
    assert [(result.run_hash, result.scientific_content_hash) for result in serial] == [
        (result.run_hash, result.scientific_content_hash) for result in parallel
    ]


def test_parallel_sweep_bounds_inflight_work_and_stops_submitting_on_error() -> None:
    cells = tuple(
        SimpleNamespace(index=index, cell_hash=f"cell-{index:04d}")
        for index in range(100)
    )

    class Grid:
        def cells(self):
            return cells

    class FailingExecutor:
        def __init__(self) -> None:
            self.started: list[int] = []
            self.lock = Lock()
            self.failure_started = Event()
            self.hold_other_workers = Event()

        def execute(self, _experiment: object, cell: SimpleNamespace):
            with self.lock:
                self.started.append(cell.index)
            if cell.index == 0:
                self.failure_started.set()
                raise RuntimeError("first cell failed")
            self.failure_started.wait(timeout=0.1)
            self.hold_other_workers.wait(timeout=0.1)
            return SimpleNamespace(run_hash=f"run-{cell.index:04d}")

    executor = FailingExecutor()
    with pytest.raises(RuntimeError, match="first cell failed"):
        SweepRunner(executor).run(Grid(), max_workers=4)  # type: ignore[arg-type]

    assert 0 in executor.started
    assert len(executor.started) <= 4


def test_shared_frontier_is_solved_once_per_artifact_root(tmp_path: Path) -> None:
    class CountingAdapter(ExactSymbolicAdapter):
        calls = 0

        def frontier(self, cell):
            type(self).calls += 1
            return super().frontier(cell)

    config = replace(
        experiment(),
        environments=(
            EnvironmentConfig(EnvironmentKind.IND, projection_size=1),
            EnvironmentConfig(EnvironmentKind.ALEA, projection_size=1),
        ),
    )
    results = SweepRunner(RunExecutor(tmp_path, CountingAdapter)).run(config)

    assert len(results) == 2
    assert CountingAdapter.calls == 1


def test_duplicate_sweep_cells_are_rejected(tmp_path: Path) -> None:
    config = experiment()
    del tmp_path
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            config,
            environments=(config.environments[0], config.environments[0]),
        )


def test_same_run_executors_are_serialized(tmp_path: Path) -> None:
    config = experiment()
    cell = config.cells()[0]
    executor = RunExecutor(tmp_path, ExactSymbolicAdapter)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: executor.execute(config, cell), range(2)))
    assert all(result.complete for result in results)
    assert len({result.scientific_content_hash for result in results}) == 1
    store = ArtifactStore(results[0].path)
    hashes = store.read("config.resolved.json").semantic_hashes
    assert len(EventJournal(store, hashes).events()) == config.horizon


def test_stale_checkpoint_seed_is_rejected_on_resume(tmp_path: Path) -> None:
    config = experiment()
    cell = config.cells()[0]
    executor = RunExecutor(tmp_path, ExactSymbolicAdapter)
    partial = executor.execute(config, cell, stop_after_new_events=1)
    checkpoint_path = partial.path / "checkpoints/00000000.json"
    checkpoint = read_artifact(checkpoint_path)
    changed_payload = {**checkpoint.payload, "evaluation_seed": -1}
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        changed_payload,
    )
    checkpoint_path.chmod(0o644)
    checkpoint_path.write_text(json.dumps(changed.to_dict()))
    with pytest.raises(ScientificArtifactError, match="replayed"):
        executor.execute(config, cell)


@pytest.mark.parametrize("tree", ("run", "frontier"))
def test_resume_removes_reserved_orphan_publication_temporaries(
    tmp_path: Path,
    tree: str,
) -> None:
    config = experiment()
    cell = config.cells()[0]
    executor = RunExecutor(tmp_path, ExactSymbolicAdapter)
    partial = executor.execute(config, cell, stop_after_new_events=1)
    if tree == "run":
        temporary = partial.path / "events" / ".00000001.json.0123456789abcdef01234567"
    else:
        reference = read_artifact(partial.path / "frontier-reference.json")
        temporary = (
            tmp_path
            / "_frontiers"
            / reference.payload["frontier_hash"]
            / "frontier"
            / ".diagnostics.json.0123456789abcdef01234567"
        )
    temporary.write_text('{"interrupted":', encoding="utf-8")

    completed = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)

    assert completed.complete
    assert not temporary.exists()


def test_self_consistent_checkpoint_result_tamper_is_rejected_on_resume(
    tmp_path: Path,
) -> None:
    config = experiment()
    cell = config.cells()[0]
    executor = RunExecutor(tmp_path, ExactSymbolicAdapter)
    partial = executor.execute(config, cell, stop_after_new_events=1)
    checkpoint_path = partial.path / "checkpoints/00000000.json"
    checkpoint = read_artifact(checkpoint_path)
    changed_payload = {
        **checkpoint.payload,
        "result": {
            **checkpoint.payload["result"],
            "expected_reward": 12345.0,
        },
    }
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        changed_payload,
    )
    checkpoint_path.chmod(0o644)
    checkpoint_path.write_text(json.dumps(changed.to_dict()))

    with pytest.raises(ScientificArtifactError, match="replayed evaluation"):
        executor.execute(config, cell)


def test_execute_does_not_rematerialize_the_experiment_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = experiment()
    cell = config.cells()[0]

    def fail_if_called(self: ExperimentConfig) -> tuple[RunCell, ...]:
        del self
        raise AssertionError("execute rematerialized the experiment grid")

    monkeypatch.setattr(ExperimentConfig, "cells", fail_if_called)
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)

    assert result.complete


def test_executor_validates_each_shared_frontier_once_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.orchestration.artifacts as artifact_module

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
    config = experiment(replicas=3)
    results = SweepRunner(
        RunExecutor(tmp_path, ExactSymbolicAdapter),
    ).run(config, max_workers=2)

    assert len(results) == 3
    assert calls == 1


def test_frontier_validation_cache_is_isolated_from_mutable_results(
    tmp_path: Path,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    reference = read_artifact(result.path / "frontier-reference.json")
    frontier_hash = reference.payload["frontier_hash"]
    frontier_root = tmp_path / "_frontiers" / frontier_hash
    session = ArtifactValidationSession()
    expected = {"frontier": frontier_hash}

    first = validate_artifact_tree(
        frontier_root,
        expected_semantic_hashes=expected,
        session=session,
    )
    first_curve = next(
        artifact for artifact in first if artifact.artifact_type == "frontier-curve"
    )
    original_payload = json.loads(json.dumps(first_curve.payload))
    first_curve.payload["points"][0]["target_reward"] = 999.0
    first_curve.semantic_hashes["frontier"] = "0" * 64

    second = validate_artifact_tree(
        frontier_root,
        expected_semantic_hashes=expected,
        session=session,
    )
    second_curve = next(
        artifact for artifact in second if artifact.artifact_type == "frontier-curve"
    )
    assert second_curve.payload == original_payload
    assert second_curve.semantic_hashes == expected

    second_curve.payload["points"][0]["target_reward"] = 888.0
    third = validate_artifact_tree(
        frontier_root,
        expected_semantic_hashes=expected,
        session=session,
    )
    third_curve = next(
        artifact for artifact in third if artifact.artifact_type == "frontier-curve"
    )
    assert third_curve.payload == original_payload


def test_parallel_resume_single_flights_shared_frontier_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinite_rulebook.orchestration.artifacts as artifact_module

    config = experiment(replicas=4)
    SweepRunner(RunExecutor(tmp_path, ExactSymbolicAdapter)).run(config)
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
    results = SweepRunner(
        RunExecutor(tmp_path, ExactSymbolicAdapter),
    ).run(config, max_workers=4)

    assert len(results) == 4
    assert calls == 1


def test_artifact_paths_cannot_escape_store(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    semantics = {"environment": scientific_hash("environment")}
    with pytest.raises(ScientificArtifactError, match="inside"):
        store.write("../outside.json", "metric", semantics, {"value": 1})
    with pytest.raises(ScientificArtifactError, match="inside"):
        store.read(tmp_path / "outside.json")


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    metric_path = result.path / "metrics.json"
    raw = json.loads(metric_path.read_text())
    raw["payload"]["event_count"] = 999
    metric_path.chmod(0o644)
    metric_path.write_text(json.dumps(raw))
    with pytest.raises(ScientificArtifactError, match="hash mismatch"):
        validate_artifact_tree(result.path)


def test_self_consistent_false_manifest_content_hash_is_rejected(
    tmp_path: Path,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    manifest_path = result.path / "manifest.json"
    manifest = read_artifact(manifest_path)
    changed = ArtifactEnvelope.create(
        manifest.artifact_type,
        manifest.semantic_hashes,
        {**manifest.payload, "scientific_content_hash": "false"},
        runtime_metadata=manifest.runtime_metadata,
    )
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(changed.to_dict()))
    with pytest.raises(ScientificArtifactError, match="content hash"):
        validate_artifact_tree(result.path)


def test_self_consistent_manifest_cannot_mix_semantic_conditions(
    tmp_path: Path,
) -> None:
    config = experiment()
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config, config.cells()[0]
    )
    metric_path = result.path / "metrics.json"
    metric = read_artifact(metric_path)
    changed_metric = ArtifactEnvelope.create(
        metric.artifact_type,
        {**metric.semantic_hashes, "environment": scientific_hash("other")},
        metric.payload,
    )
    metric_path.chmod(0o644)
    metric_path.write_text(json.dumps(changed_metric.to_dict()))

    manifest_path = result.path / "manifest.json"
    manifest = read_artifact(manifest_path)
    members = [
        {
            **member,
            "scientific_hash": (
                changed_metric.scientific_hash
                if member["path"] == "metrics.json"
                else member["scientific_hash"]
            ),
        }
        for member in manifest.payload["members"]
    ]
    changed_manifest = ArtifactEnvelope.create(
        manifest.artifact_type,
        manifest.semantic_hashes,
        {
            "members": members,
            "scientific_content_hash": scientific_hash(
                members,
                domain="run-scientific-content",
            ),
        },
        runtime_metadata=manifest.runtime_metadata,
    )
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(changed_manifest.to_dict()))

    with pytest.raises(ScientificArtifactError, match="semantic hashes differ"):
        validate_artifact_tree(result.path)


def test_immutable_artifact_cannot_be_replaced(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    semantics = {"environment": scientific_hash("environment")}
    first = store.write("metric.json", "metric", semantics, {"value": 1})
    assert store.write("metric.json", "metric", semantics, {"value": 1}) == first
    with pytest.raises(ScientificArtifactError, match="refusing to mutate"):
        store.write("metric.json", "metric", semantics, {"value": 2})


def test_finalize_rejects_member_from_another_semantic_condition(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    expected = {"environment": scientific_hash("one")}
    other = {"environment": scientific_hash("two")}
    store.write("config.json", "resolved-run-config", expected, {})
    store.write("frontier.json", "frontier-reference", expected, {})
    store.write("event.json", "training-event", expected, {})
    store.write("checkpoint.json", "run-checkpoint", expected, {})
    store.write("metrics.json", "run-metrics", other, {})

    with pytest.raises(ScientificArtifactError, match="semantic hashes"):
        store.finalize(expected)
