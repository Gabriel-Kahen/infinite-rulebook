"""Portable, authenticated inventories for complete raw artifact roots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ArtifactRootBusyError,
    ArtifactValidationSession,
    ScientificArtifactError,
    artifact_root_lock,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    run_cell_from_dict,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import load_json_strict
from infinite_rulebook.orchestration.reproducibility import (
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    ExecutionReceipt,
    authenticate_execution_receipt,
)

RAW_ARTIFACT_INVENTORY_VERSION = 2
RAW_ARTIFACT_INVENTORY_FORMAT = "infinite-rulebook-raw-artifact-inventory"
_INVENTORY_DOMAIN = "raw-artifact-inventory"
_TREE_HASH_PREFIX = b"infinite-rulebook-raw-artifact-tree-v1\0"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_INVENTORY_FIELDS = {
    "artifact_type",
    "schema_version",
    "experiment_name",
    "config_hash",
    "side",
    "execution_receipt",
    "trees",
    "scientific_hash",
}
_TREE_FIELDS = {
    "tree_type",
    "path",
    "identity_hash",
    "scientific_content_hash",
    "file_count",
    "byte_size",
    "tree_byte_sha256",
}


class RawArtifactInventoryError(ScientificArtifactError):
    """Raised when a raw artifact inventory or root is invalid."""


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RawArtifactInventoryError(f"{label} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise RawArtifactInventoryError(f"{label} must be one safe relative component")
    return value


@dataclass(frozen=True, slots=True)
class RawArtifactTree:
    tree_type: str
    path: str
    identity_hash: str
    scientific_content_hash: str
    file_count: int
    byte_size: int
    tree_byte_sha256: str

    def __post_init__(self) -> None:
        if self.tree_type not in {"run", "frontier"}:
            raise RawArtifactInventoryError("raw artifact tree_type is invalid")
        if not all(
            is_sha256(value)
            for value in (
                self.identity_hash,
                self.scientific_content_hash,
                self.tree_byte_sha256,
            )
        ):
            raise RawArtifactInventoryError(
                "raw artifact tree hashes must be SHA-256 digests"
            )
        relative = PurePosixPath(self.path)
        expected_prefix = "_frontiers" if self.tree_type == "frontier" else None
        if (
            not isinstance(self.path, str)
            or not self.path
            or relative.is_absolute()
            or len(relative.parts) != 2
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in self.path
            or relative.parts[1] != self.identity_hash
            or (expected_prefix is not None and relative.parts[0] != expected_prefix)
            or (self.tree_type == "run" and relative.parts[0] == "_frontiers")
        ):
            raise RawArtifactInventoryError(
                "raw artifact tree path is unsafe or inconsistent"
            )
        for label, value, minimum in (
            ("file_count", self.file_count, 1),
            ("byte_size", self.byte_size, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise RawArtifactInventoryError(f"raw artifact tree {label} is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "tree_type": self.tree_type,
            "path": self.path,
            "identity_hash": self.identity_hash,
            "scientific_content_hash": self.scientific_content_hash,
            "file_count": self.file_count,
            "byte_size": self.byte_size,
            "tree_byte_sha256": self.tree_byte_sha256,
        }

    @classmethod
    def from_dict(cls, raw: object) -> RawArtifactTree:
        if not isinstance(raw, dict) or set(raw) != _TREE_FIELDS:
            raise RawArtifactInventoryError("raw artifact tree fields are invalid")
        try:
            return cls(**raw)
        except TypeError as error:
            raise RawArtifactInventoryError(
                "raw artifact tree values are invalid"
            ) from error


@dataclass(frozen=True, slots=True)
class RawArtifactInventory:
    experiment_name: str
    config_hash: str
    side: str
    trees: tuple[RawArtifactTree, ...]
    scientific_hash: str
    execution_receipt: ExecutionReceipt | None = None

    def __post_init__(self) -> None:
        experiment_name = _safe_component(
            self.experiment_name,
            "raw artifact experiment_name",
        )
        if experiment_name == "_frontiers":
            raise RawArtifactInventoryError("raw artifact experiment_name is reserved")
        if not is_sha256(self.config_hash):
            raise RawArtifactInventoryError(
                "raw artifact config_hash must be a SHA-256 digest"
            )
        if self.side not in {"serial", "parallel"}:
            raise RawArtifactInventoryError(
                "raw artifact side must be serial or parallel"
            )
        if self.execution_receipt is not None and (
            not isinstance(self.execution_receipt, ExecutionReceipt)
            or self.execution_receipt.config_hash != self.config_hash
            or self.execution_receipt.role != self.side
        ):
            raise RawArtifactInventoryError(
                "raw artifact execution receipt does not match its inventory"
            )
        if not self.trees or any(
            not isinstance(tree, RawArtifactTree) for tree in self.trees
        ):
            raise RawArtifactInventoryError(
                "raw artifact trees must be a nonempty tuple"
            )
        if tuple(sorted(self.trees, key=lambda tree: tree.path)) != self.trees:
            raise RawArtifactInventoryError("raw artifact trees must be sorted by path")
        paths = [tree.path for tree in self.trees]
        identities = [(tree.tree_type, tree.identity_hash) for tree in self.trees]
        if len(set(paths)) != len(paths) or len(set(identities)) != len(identities):
            raise RawArtifactInventoryError(
                "raw artifact trees contain duplicate records"
            )
        run_trees = tuple(tree for tree in self.trees if tree.tree_type == "run")
        frontier_trees = tuple(
            tree for tree in self.trees if tree.tree_type == "frontier"
        )
        if (
            not run_trees
            or not frontier_trees
            or any(
                PurePosixPath(tree.path).parts[0] != experiment_name
                for tree in run_trees
            )
        ):
            raise RawArtifactInventoryError(
                "raw artifact trees do not match the experiment"
            )
        if self.scientific_hash != scientific_hash(
            self.body_dict(),
            domain=_INVENTORY_DOMAIN,
        ):
            raise RawArtifactInventoryError(
                "raw artifact inventory scientific_hash is invalid"
            )

    def body_dict(self) -> dict[str, object]:
        return {
            "artifact_type": RAW_ARTIFACT_INVENTORY_FORMAT,
            "schema_version": RAW_ARTIFACT_INVENTORY_VERSION,
            "experiment_name": self.experiment_name,
            "config_hash": self.config_hash,
            "side": self.side,
            "execution_receipt": (
                None
                if self.execution_receipt is None
                else self.execution_receipt.to_dict()
            ),
            "trees": [tree.to_dict() for tree in self.trees],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "scientific_hash": self.scientific_hash}

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def create(
        cls,
        artifact_root: str | Path,
        experiment: ExperimentConfig,
        *,
        side: str,
    ) -> RawArtifactInventory:
        return create_raw_artifact_inventory(
            artifact_root,
            experiment,
            side=side,
        )

    @classmethod
    def from_dict(cls, raw: object) -> RawArtifactInventory:
        if not isinstance(raw, dict) or set(raw) != _INVENTORY_FIELDS:
            raise RawArtifactInventoryError("raw artifact inventory fields are invalid")
        if (
            raw["artifact_type"] != RAW_ARTIFACT_INVENTORY_FORMAT
            or raw["schema_version"] != RAW_ARTIFACT_INVENTORY_VERSION
            or isinstance(raw["schema_version"], bool)
            or not isinstance(raw["trees"], list)
        ):
            raise RawArtifactInventoryError(
                "raw artifact inventory type or schema is invalid"
            )
        trees = tuple(RawArtifactTree.from_dict(item) for item in raw["trees"])
        try:
            execution_receipt = (
                None
                if raw["execution_receipt"] is None
                else ExecutionReceipt.from_dict(raw["execution_receipt"])
            )
            inventory = cls(
                experiment_name=raw["experiment_name"],
                config_hash=raw["config_hash"],
                side=raw["side"],
                execution_receipt=execution_receipt,
                trees=trees,
                scientific_hash=raw["scientific_hash"],
            )
        except TypeError as error:
            raise RawArtifactInventoryError(
                "raw artifact inventory values are invalid"
            ) from error
        if inventory.to_dict() != raw:
            raise RawArtifactInventoryError("raw artifact inventory is not canonical")
        return inventory

    def verify(
        self,
        artifact_root: str | Path,
        experiment: ExperimentConfig,
        *,
        side: str | None = None,
    ) -> None:
        if side is not None and side != self.side:
            raise RawArtifactInventoryError(
                "raw artifact inventory side differs from the expected side"
            )
        if (
            not isinstance(experiment, ExperimentConfig)
            or experiment.name != self.experiment_name
            or experiment.config_hash != self.config_hash
        ):
            raise RawArtifactInventoryError(
                "raw artifact inventory does not match the exact experiment"
            )
        observed = create_raw_artifact_inventory(
            artifact_root,
            experiment,
            side=self.side,
        )
        if observed != self:
            raise RawArtifactInventoryError(
                "raw artifact root does not exactly match its inventory"
            )


def _directory_entries(path: Path, label: str) -> tuple[os.DirEntry[str], ...]:
    try:
        if path.is_symlink():
            raise RawArtifactInventoryError(f"{label} must not be a symbolic link")
        with os.scandir(path) as stream:
            entries = tuple(sorted(stream, key=lambda entry: entry.name))
    except RawArtifactInventoryError:
        raise
    except OSError as error:
        raise RawArtifactInventoryError(f"cannot inspect {label}: {path}") from error
    if any(
        entry.is_symlink() or not entry.is_dir(follow_symlinks=False)
        for entry in entries
    ):
        raise RawArtifactInventoryError(
            f"{label} contains a symbolic link or non-directory entry"
        )
    return entries


def _tree_metrics(root: Path) -> tuple[int, int, str]:
    def open_directory(path: Path) -> int:
        absolute = Path(os.path.abspath(path))
        descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
        try:
            for component in absolute.parts[1:]:
                child = os.open(
                    component,
                    _DIRECTORY_FLAGS,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
        except OSError as error:
            os.close(descriptor)
            raise RawArtifactInventoryError(
                f"cannot safely open raw artifact tree: {path}"
            ) from error
        return descriptor

    def same_state(left: os.stat_result, right: os.stat_result) -> bool:
        return all(
            getattr(left, field) == getattr(right, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )

    digest = hashlib.sha256(_TREE_HASH_PREFIX)
    file_count = 0
    byte_size = 0

    def visit(descriptor: int, relative: PurePosixPath) -> None:
        nonlocal byte_size, file_count
        before_directory = os.fstat(descriptor)
        names = tuple(sorted(os.listdir(descriptor)))
        for name in names:
            child_relative = relative / name
            try:
                metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise RawArtifactInventoryError(
                    f"cannot inspect raw artifact member: {root / child_relative}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise RawArtifactInventoryError(
                    "raw artifact tree contains a symbolic link: "
                    f"{root / child_relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as error:
                    raise RawArtifactInventoryError(
                        "cannot safely traverse raw artifact directory: "
                        f"{root / child_relative}"
                    ) from error
                try:
                    visit(child, child_relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RawArtifactInventoryError(
                    "raw artifact tree contains a non-regular file: "
                    f"{root / child_relative}"
                )
            if child_relative == PurePosixPath(".run.lock"):
                continue
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(
                    name,
                    _FILE_FLAGS,
                    dir_fd=descriptor,
                )
                opened = os.fstat(file_descriptor)
                if not stat.S_ISREG(opened.st_mode) or not same_state(metadata, opened):
                    raise RawArtifactInventoryError(
                        "raw artifact member changed during inventory"
                    )
                encoded = child_relative.as_posix().encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                digest.update(opened.st_size.to_bytes(8, "big"))
                observed_size = 0
                while True:
                    chunk = os.read(file_descriptor, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    observed_size += len(chunk)
                after = os.fstat(file_descriptor)
                if observed_size != opened.st_size or not same_state(opened, after):
                    raise RawArtifactInventoryError(
                        "raw artifact member changed during inventory"
                    )
            except RawArtifactInventoryError:
                raise
            except OSError as error:
                raise RawArtifactInventoryError(
                    f"cannot safely read raw artifact member: {root / child_relative}"
                ) from error
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)
            file_count += 1
            byte_size += opened.st_size
        if tuple(sorted(os.listdir(descriptor))) != names or not same_state(
            before_directory, os.fstat(descriptor)
        ):
            raise RawArtifactInventoryError(
                f"raw artifact directory changed during inventory: {root / relative}"
            )

    root_descriptor = open_directory(root)
    try:
        visit(root_descriptor, PurePosixPath())
    finally:
        os.close(root_descriptor)
    return file_count, byte_size, digest.hexdigest()


def _manifest(
    artifacts: tuple[ArtifactEnvelope, ...],
    artifact_type: str,
) -> ArtifactEnvelope:
    matches = tuple(
        artifact for artifact in artifacts if artifact.artifact_type == artifact_type
    )
    if len(matches) != 1:
        raise RawArtifactInventoryError(f"raw artifact tree lacks one {artifact_type}")
    return matches[0]


def _tree_record(
    root: Path,
    *,
    tree_type: str,
    relative_path: str,
    identity_hash: str,
    content_hash: object,
    validated_artifacts: tuple[ArtifactEnvelope, ...],
    expected_semantic_hashes: dict[str, str] | None = None,
    revalidation_session: ArtifactValidationSession | None = None,
) -> RawArtifactTree:
    if not is_sha256(content_hash):
        raise RawArtifactInventoryError(
            "raw artifact tree scientific content hash is invalid"
        )
    file_count, byte_size, tree_byte_sha256 = _tree_metrics(root)
    revalidated = validate_artifact_tree(
        root,
        expected_semantic_hashes=expected_semantic_hashes,
        session=revalidation_session or ArtifactValidationSession(),
    )
    if revalidated != validated_artifacts:
        raise RawArtifactInventoryError(
            "raw artifact tree changed across its authenticated snapshot"
        )
    return RawArtifactTree(
        tree_type=tree_type,
        path=relative_path,
        identity_hash=identity_hash,
        scientific_content_hash=content_hash,
        file_count=file_count,
        byte_size=byte_size,
        tree_byte_sha256=tree_byte_sha256,
    )


def _normalized(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def create_raw_artifact_inventory(
    artifact_root: str | Path,
    experiment: ExperimentConfig,
    *,
    side: str,
) -> RawArtifactInventory:
    """Acquire snapshot ownership and authenticate one complete raw root."""

    root = Path(artifact_root)
    try:
        with artifact_root_lock(
            root,
            nonblocking=True,
            create=False,
        ):
            return _create_raw_artifact_inventory_locked(
                root,
                experiment,
                side=side,
            )
    except ArtifactRootBusyError as error:
        raise RawArtifactInventoryError(
            "raw artifact root is already owned by another workflow"
        ) from error
    except RawArtifactInventoryError:
        raise
    except ScientificArtifactError as error:
        raise RawArtifactInventoryError(
            f"cannot safely inspect raw artifact root: {error}"
        ) from error


def _create_raw_artifact_inventory_locked(
    artifact_root: str | Path,
    experiment: ExperimentConfig,
    *,
    side: str,
) -> RawArtifactInventory:
    """Authenticate one complete raw root and return its portable inventory."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    if side not in {"serial", "parallel"}:
        raise RawArtifactInventoryError("raw artifact side must be serial or parallel")
    if experiment.name in {
        "_frontiers",
        REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    }:
        raise RawArtifactInventoryError("raw artifact experiment name is reserved")

    root = Path(artifact_root)
    entries = _directory_entries(root, "raw artifact root")
    entry_names = {entry.name for entry in entries}
    expected_names = {experiment.name, "_frontiers"}
    if entry_names not in (
        expected_names,
        expected_names | {REPRODUCIBILITY_OPERATIONAL_DIRECTORY},
    ):
        raise RawArtifactInventoryError(
            "raw artifact root contains missing or unexpected entries"
        )
    execution_receipt = None
    if REPRODUCIBILITY_OPERATIONAL_DIRECTORY in entry_names:
        try:
            execution_receipt = authenticate_execution_receipt(
                root,
                experiment,
                role=side,
            )
        except ScientificArtifactError as error:
            raise RawArtifactInventoryError(
                "raw artifact execution receipt is invalid"
            ) from error
    run_parent = root / experiment.name
    frontier_parent = root / "_frontiers"
    run_entries = _directory_entries(run_parent, "raw run inventory")
    if len(run_entries) != len(experiment.cells()) or any(
        not is_sha256(entry.name) for entry in run_entries
    ):
        raise RawArtifactInventoryError(
            "raw run inventory contains missing or unexpected trees"
        )

    expected_cells = {cell.cell_hash: cell for cell in experiment.cells()}
    expected_settings = _normalized(experiment.resolved_run_settings())
    observed_cells: set[str] = set()
    referenced_frontiers: set[str] = set()
    trees: list[RawArtifactTree] = []
    session = ArtifactValidationSession()
    run_revalidation_session = ArtifactValidationSession()

    for entry in run_entries:
        run_root = Path(entry.path)
        artifacts = validate_artifact_tree(run_root, session=session)
        manifest = _manifest(artifacts, "run-manifest")
        config = _manifest(artifacts, "resolved-run-config").payload
        reference = _manifest(artifacts, "frontier-reference").payload
        if (
            not isinstance(config, dict)
            or set(config)
            != {"cell", "provenance", "run_hash", "run_settings", "seeds"}
            or config["run_hash"] != entry.name
            or config["run_settings"] != expected_settings
        ):
            raise RawArtifactInventoryError(
                "raw run identity does not match the exact experiment"
            )
        try:
            cell = run_cell_from_dict(config["cell"])
        except (TypeError, ValueError) as error:
            raise RawArtifactInventoryError("raw run cell is malformed") from error
        if (
            expected_cells.get(cell.cell_hash) != cell
            or cell.cell_hash in observed_cells
        ):
            raise RawArtifactInventoryError(
                "raw run cell inventory does not match the exact experiment"
            )
        if (
            not isinstance(reference, dict)
            or set(reference) != {"artifact_hash", "frontier_hash"}
            or not is_sha256(reference["frontier_hash"])
        ):
            raise RawArtifactInventoryError("raw frontier reference is invalid")
        observed_cells.add(cell.cell_hash)
        referenced_frontiers.add(reference["frontier_hash"])
        trees.append(
            _tree_record(
                run_root,
                tree_type="run",
                relative_path=f"{experiment.name}/{entry.name}",
                identity_hash=entry.name,
                content_hash=manifest.payload.get("scientific_content_hash"),
                validated_artifacts=artifacts,
                revalidation_session=run_revalidation_session,
            )
        )

    if observed_cells != set(expected_cells):
        raise RawArtifactInventoryError("raw run cell inventory is incomplete")
    frontier_entries = _directory_entries(
        frontier_parent,
        "raw frontier inventory",
    )
    if {entry.name for entry in frontier_entries} != referenced_frontiers or any(
        not is_sha256(entry.name) for entry in frontier_entries
    ):
        raise RawArtifactInventoryError(
            "raw frontier inventory contains missing, unreferenced, or unexpected trees"
        )
    for entry in frontier_entries:
        frontier_root = Path(entry.path)
        artifacts = validate_artifact_tree(
            frontier_root,
            expected_semantic_hashes={"frontier": entry.name},
            session=session,
        )
        manifest = _manifest(artifacts, "frontier-manifest")
        trees.append(
            _tree_record(
                frontier_root,
                tree_type="frontier",
                relative_path=f"_frontiers/{entry.name}",
                identity_hash=entry.name,
                content_hash=manifest.payload.get("scientific_content_hash"),
                validated_artifacts=artifacts,
                expected_semantic_hashes={"frontier": entry.name},
            )
        )

    ordered = tuple(sorted(trees, key=lambda tree: tree.path))
    body = {
        "artifact_type": RAW_ARTIFACT_INVENTORY_FORMAT,
        "schema_version": RAW_ARTIFACT_INVENTORY_VERSION,
        "experiment_name": experiment.name,
        "config_hash": experiment.config_hash,
        "side": side,
        "execution_receipt": (
            None if execution_receipt is None else execution_receipt.to_dict()
        ),
        "trees": [tree.to_dict() for tree in ordered],
    }
    return RawArtifactInventory(
        experiment_name=experiment.name,
        config_hash=experiment.config_hash,
        side=side,
        execution_receipt=execution_receipt,
        trees=ordered,
        scientific_hash=scientific_hash(body, domain=_INVENTORY_DOMAIN),
    )


def raw_artifact_inventory_from_dict(raw: object) -> RawArtifactInventory:
    """Parse a strict, canonical raw-artifact inventory object."""

    return RawArtifactInventory.from_dict(raw)


def load_raw_artifact_inventory(path: str | Path) -> RawArtifactInventory:
    """Load a raw-artifact inventory with duplicate-key rejection."""

    return RawArtifactInventory.from_dict(
        load_json_strict(path, label="raw artifact inventory")
    )


def verify_raw_artifact_inventory(
    inventory: RawArtifactInventory,
    artifact_root: str | Path,
    experiment: ExperimentConfig,
    *,
    side: str | None = None,
) -> None:
    """Recompute one complete raw root and exact-compare its inventory."""

    if not isinstance(inventory, RawArtifactInventory):
        raise TypeError("inventory must be a RawArtifactInventory")
    inventory.verify(artifact_root, experiment, side=side)


__all__ = [
    "RAW_ARTIFACT_INVENTORY_FORMAT",
    "RAW_ARTIFACT_INVENTORY_VERSION",
    "RawArtifactInventory",
    "RawArtifactInventoryError",
    "RawArtifactTree",
    "create_raw_artifact_inventory",
    "load_raw_artifact_inventory",
    "raw_artifact_inventory_from_dict",
    "verify_raw_artifact_inventory",
]
