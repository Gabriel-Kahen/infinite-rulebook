from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import infinite_rulebook.orchestration.artifacts as artifact_module
import infinite_rulebook.orchestration.inventory as inventory_module
from infinite_rulebook.orchestration.artifacts import ScientificArtifactError
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
)
from infinite_rulebook.orchestration.inventory import (
    RawArtifactInventory,
    RawArtifactInventoryError,
    create_raw_artifact_inventory,
    load_raw_artifact_inventory,
    verify_raw_artifact_inventory,
)
from infinite_rulebook.orchestration.reproducibility import (
    EXECUTION_RECEIPT_FILENAME,
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    run_reproducibility_check,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="raw-inventory-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.FIXED, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="raw-inventory-environment",
        algorithm_master_seed="raw-inventory-algorithm",
    )


@pytest.fixture
def raw_case(
    tmp_path: Path,
) -> tuple[Path, ExperimentConfig, RawArtifactInventory]:
    root = tmp_path / "raw"
    experiment = _experiment()
    SweepRunner(RunExecutor(root, ExactSymbolicAdapter)).run(
        experiment,
        max_workers=1,
    )
    inventory = create_raw_artifact_inventory(
        root,
        experiment,
        side="serial",
    )
    return root, experiment, inventory


def _run_tree(inventory: RawArtifactInventory) -> str:
    return next(tree.path for tree in inventory.trees if tree.tree_type == "run")


def _frontier_tree(inventory: RawArtifactInventory) -> str:
    return next(tree.path for tree in inventory.trees if tree.tree_type == "frontier")


def test_inventory_roundtrip_is_strict_and_authenticates_complete_root(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    tmp_path: Path,
) -> None:
    root, experiment, inventory = raw_case
    assert [tree.path for tree in inventory.trees] == sorted(
        tree.path for tree in inventory.trees
    )
    assert {tree.tree_type for tree in inventory.trees} == {"run", "frontier"}
    assert all(tree.file_count > 0 and tree.byte_size > 0 for tree in inventory.trees)

    parsed = RawArtifactInventory.from_dict(inventory.to_dict())
    assert parsed == inventory
    path = tmp_path / "raw-inventory.json"
    path.write_text(inventory.to_json(), encoding="utf-8")
    assert load_raw_artifact_inventory(path) == inventory
    verify_raw_artifact_inventory(
        parsed,
        root,
        experiment,
        side="serial",
    )

    raw = inventory.to_dict()
    raw["unexpected"] = True
    with pytest.raises(RawArtifactInventoryError, match="fields"):
        RawArtifactInventory.from_dict(raw)
    path.write_text('{"side":"serial","side":"parallel"}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeats key"):
        load_raw_artifact_inventory(path)


def test_inventory_reuses_only_the_run_revalidation_frontier_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    experiment = replace(_experiment(), environment_replicas=3)
    SweepRunner(RunExecutor(root, ExactSymbolicAdapter)).run(experiment)
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

    create_raw_artifact_inventory(root, experiment, side="serial")

    assert calls == 3


def test_inventory_binds_but_excludes_validated_execution_receipt(
    tmp_path: Path,
) -> None:
    experiment = _experiment()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    report = run_reproducibility_check(
        experiment,
        serial_root=serial,
        parallel_root=parallel,
        parallel_workers=2,
    )
    inventory = create_raw_artifact_inventory(
        serial,
        experiment,
        side="serial",
    )

    assert inventory.execution_receipt is not None
    assert inventory.execution_receipt.scientific_hash == report.serial_receipt_hash
    assert all(
        not tree.path.startswith(REPRODUCIBILITY_OPERATIONAL_DIRECTORY)
        for tree in inventory.trees
    )

    receipt = (
        serial / REPRODUCIBILITY_OPERATIONAL_DIRECTORY / EXECUTION_RECEIPT_FILENAME
    )
    receipt.chmod(0o600)
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(RawArtifactInventoryError, match="receipt"):
        inventory.verify(serial, experiment, side="serial")


def test_inventory_is_portable_across_root_relocation(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    tmp_path: Path,
) -> None:
    root, experiment, inventory = raw_case
    relocated = tmp_path / "relocated" / "raw"
    shutil.copytree(root, relocated)

    verify_raw_artifact_inventory(inventory, relocated, experiment)
    assert (
        create_raw_artifact_inventory(
            relocated,
            experiment,
            side="serial",
        )
        == inventory
    )


def test_tree_byte_hash_detects_semantically_inert_byte_tampering(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
) -> None:
    root, experiment, inventory = raw_case
    metrics = root / _run_tree(inventory) / "metrics.json"
    metrics.chmod(0o644)
    metrics.write_bytes(metrics.read_bytes() + b"\n")

    with pytest.raises(RawArtifactInventoryError, match="exactly match"):
        inventory.verify(root, experiment)


def test_inventory_rejects_change_between_validation_and_byte_snapshot(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, experiment, _ = raw_case
    original = inventory_module._tree_metrics
    changed = False

    def mutate_after_validation(tree: Path) -> tuple[int, int, str]:
        nonlocal changed
        if not changed and tree.parent.name == experiment.name:
            (tree / "post-validation.json").write_text("{}", encoding="utf-8")
            changed = True
        return original(tree)

    monkeypatch.setattr(inventory_module, "_tree_metrics", mutate_after_validation)
    with pytest.raises(RawArtifactInventoryError, match=r"safely inspect|changed"):
        create_raw_artifact_inventory(root, experiment, side="serial")
    assert changed


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_inventory_rejects_missing_or_extra_trees(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    tmp_path: Path,
    mutation: str,
) -> None:
    root, experiment, inventory = raw_case
    run = root / _run_tree(inventory)
    if mutation == "missing":
        run.rename(tmp_path / "removed-run")
    else:
        frontier = root / _frontier_tree(inventory)
        shutil.copytree(frontier, root / "_frontiers" / ("f" * 64))

    with pytest.raises(
        RawArtifactInventoryError,
        match=r"missing|unexpected|unreferenced",
    ):
        create_raw_artifact_inventory(root, experiment, side="serial")


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_inventory_rejects_missing_or_extra_files(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    mutation: str,
) -> None:
    root, experiment, inventory = raw_case
    run = root / _run_tree(inventory)
    if mutation == "missing":
        (run / "metrics.json").unlink()
    else:
        (run / "unexpected.txt").write_text("not an artifact", encoding="utf-8")

    with pytest.raises(
        ScientificArtifactError,
        match=r"manifest|unexpected|completed run",
    ):
        create_raw_artifact_inventory(root, experiment, side="serial")


def test_inventory_rejects_symlinked_root_entries(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
    tmp_path: Path,
) -> None:
    root, experiment, _ = raw_case
    frontiers = root / "_frontiers"
    target = tmp_path / "frontiers-real"
    frontiers.rename(target)
    frontiers.symlink_to(target, target_is_directory=True)

    with pytest.raises(RawArtifactInventoryError, match="symbolic link"):
        create_raw_artifact_inventory(root, experiment, side="serial")


def test_inventory_missing_root_raises_only_the_public_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(RawArtifactInventoryError, match="safely inspect"):
        create_raw_artifact_inventory(
            tmp_path / "missing",
            _experiment(),
            side="serial",
        )


def test_inventory_rejects_duplicate_and_unsafe_tree_records(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
) -> None:
    _, _, inventory = raw_case
    duplicate = inventory.to_dict()
    duplicate["trees"].append(dict(duplicate["trees"][0]))
    with pytest.raises(RawArtifactInventoryError, match=r"sorted|duplicate"):
        RawArtifactInventory.from_dict(duplicate)

    unsafe = inventory.to_dict()
    unsafe["trees"][0]["path"] = "../outside"
    with pytest.raises(RawArtifactInventoryError, match="unsafe"):
        RawArtifactInventory.from_dict(unsafe)


def test_inventory_binds_side_and_exact_experiment(
    raw_case: tuple[Path, ExperimentConfig, RawArtifactInventory],
) -> None:
    root, experiment, inventory = raw_case
    with pytest.raises(RawArtifactInventoryError, match="side"):
        inventory.verify(root, experiment, side="parallel")
    with pytest.raises(RawArtifactInventoryError, match="exact experiment"):
        inventory.verify(
            root,
            replace(experiment, master_seed="different-seed"),
        )

    raw = inventory.to_dict()
    raw["side"] = "unknown"
    with pytest.raises(RawArtifactInventoryError, match="side"):
        RawArtifactInventory.from_dict(raw)
