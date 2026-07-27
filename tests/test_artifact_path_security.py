from pathlib import Path

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactStore,
    ScientificArtifactError,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
)
from infinite_rulebook.orchestration.run import RunExecutor, run_identity
from infinite_rulebook.orchestration.seeds import SeedBank
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _store(path: Path) -> ArtifactStore:
    return ArtifactStore(path)


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="path-security",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="path-security",
    )


def test_symlinked_store_root_cannot_read_write_or_list(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-store"
    linked_root.symlink_to(outside, target_is_directory=True)
    store = _store(linked_root)

    with pytest.raises(ScientificArtifactError):
        store.write("escaped.json", "test", {}, {"value": 1})
    with pytest.raises(ScientificArtifactError):
        store.read("missing.json")
    with pytest.raises(ScientificArtifactError):
        store.list_artifacts()
    with pytest.raises(ScientificArtifactError):
        validate_artifact_tree(linked_root)

    assert tuple(outside.iterdir()) == ()


def test_symlinked_store_ancestor_cannot_redirect_artifact_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    store = _store(linked_parent / "store")

    with pytest.raises(ScientificArtifactError):
        store.write("escaped.json", "test", {}, {"value": 1})
    with pytest.raises(ScientificArtifactError):
        store.read("missing.json")

    assert tuple(outside.iterdir()) == ()


def test_executor_lock_rejects_symlinked_artifact_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    config = _experiment()

    with pytest.raises(ScientificArtifactError):
        RunExecutor(linked_root, ExactSymbolicAdapter).execute(
            config, config.cells()[0]
        )

    assert tuple(outside.iterdir()) == ()


def test_executor_lock_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    config = _experiment()
    cell = config.cells()[0]
    executor = RunExecutor(root, ExactSymbolicAdapter)
    run_hash = run_identity(
        config,
        cell,
        SeedBank(config.master_seed).for_cell(cell),
        executor.provenance,
    )
    experiment_root = root / config.name
    experiment_root.mkdir(parents=True)
    (experiment_root / run_hash).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScientificArtifactError):
        executor.execute(config, cell)

    assert tuple(outside.iterdir()) == ()


def test_symlinked_parent_cannot_redirect_artifact_write(tmp_path: Path) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _store(outside).write("existing.json", "test", {}, {"protected": True})
    original = (outside / "existing.json").read_bytes()
    (root / "events").symlink_to(outside, target_is_directory=True)
    store = _store(root)

    with pytest.raises(ScientificArtifactError):
        store.read("events/existing.json")
    with pytest.raises(ScientificArtifactError):
        store.write("events/escaped.json", "test", {}, {"value": 1})

    assert not (outside / "escaped.json").exists()
    assert (outside / "existing.json").read_bytes() == original


def test_symlinked_artifact_target_cannot_be_read_or_replaced(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    protected = _store(outside).write("protected.json", "test", {}, {"protected": True})
    protected_path = outside / "protected.json"
    original = protected_path.read_bytes()
    (root / "record.json").symlink_to(protected_path)
    store = _store(root)

    with pytest.raises(ScientificArtifactError):
        store.read("record.json")
    with pytest.raises(ScientificArtifactError):
        store.write("record.json", "test", {}, {"protected": False})

    assert protected.scientific_hash
    assert protected_path.read_bytes() == original


@pytest.mark.parametrize("member", ["linked.json", "linked-directory"])
def test_listing_and_tree_validation_reject_symlink_members(
    tmp_path: Path,
    member: str,
) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if member.endswith(".json"):
        target = outside / "artifact.json"
        _store(outside).write("artifact.json", "test", {}, {"value": 1})
        (root / member).symlink_to(target)
    else:
        (root / member).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScientificArtifactError):
        _store(root).list_artifacts()
    with pytest.raises(ScientificArtifactError):
        validate_artifact_tree(root)
