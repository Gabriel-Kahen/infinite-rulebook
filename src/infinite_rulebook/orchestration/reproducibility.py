"""Authenticated serial-versus-parallel experiment reproducibility checks."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import secrets
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from infinite_rulebook.orchestration.artifacts import (
    ArtifactRootBusyError,
    ArtifactStore,
    ArtifactValidationSession,
    EventJournal,
    ScientificArtifactError,
    artifact_root_lock,
    artifact_tree_lock,
    cleanup_orphaned_artifact_temporaries,
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    ExperimentConfig,
    run_cell_from_dict,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import parse_json_strict
from infinite_rulebook.orchestration.provenance import (
    ScientificProvenance,
    collect_provenance,
)
from infinite_rulebook.orchestration.run import (
    ExperimentAdapter,
    RunExecutor,
    RunResult,
    run_identity,
)
from infinite_rulebook.orchestration.seeds import RunSeeds, SeedBank
from infinite_rulebook.orchestration.semantics import semantic_hashes
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter

REPRODUCIBILITY_REPORT_VERSION = 2
REPRODUCIBILITY_REPORT_FORMAT = "infinite-rulebook-reproducibility"
EXECUTION_RECEIPT_FILENAME = "execution-receipt.json"
EXECUTION_RECEIPT_FORMAT = "infinite-rulebook-execution-receipt"
EXECUTION_RECEIPT_VERSION = 2
MAX_EXECUTION_RECEIPT_BYTES = 64 * 1024
_REPORT_FIELDS = (
    "operational",
    "report_format",
    "schema_version",
    "scientific",
    "scientific_hash",
)
_SCIENTIFIC_FIELDS = (
    "comparison",
    "config_hash",
    "exact_match",
    "execution_receipts",
    "run_count",
    "runs",
)
_RUN_FIELDS = ("cell_hash", "run_hash", "scientific_content_hash")
_OPERATIONAL_FIELDS = ("parallel", "serial")
_EXECUTION_FIELDS = ("artifact_root", "max_workers")
_RECEIPT_FIELDS = (
    "receipt_format",
    "schema_version",
    "scientific",
    "scientific_hash",
)
_RECEIPT_SCIENTIFIC_FIELDS = (
    "config_hash",
    "invocation_id",
    "max_workers",
    "pair",
    "parallel_workers",
    "provenance",
    "role",
    "smoke_prerequisite_hash",
)
_RECEIPT_PAIR_FIELDS = ("parallel", "serial")
_RECEIPT_REPORT_FIELDS = ("invocation_id", "parallel", "serial")
_PROVENANCE_FIELDS = (
    "analysis_code_hash",
    "blas",
    "code_commit",
    "cuda",
    "cudnn",
    "dependency_lock_hash",
    "deterministic_mode",
    "dirty_tree_hash",
    "environment_digest",
    "environment_fingerprint",
    "numeric_precision",
    "python_implementation",
    "python_version",
)


class ReproducibilityError(ScientificArtifactError):
    """Raised unless both complete authenticated sweeps match exactly."""


def _receipt_role_identity(
    *,
    config_hash: str,
    invocation_id: str,
    max_workers: int,
    provenance: ScientificProvenance,
    role: str,
    smoke_prerequisite_hash: str | None,
) -> str:
    return scientific_hash(
        {
            "config_hash": config_hash,
            "invocation_id": invocation_id,
            "max_workers": max_workers,
            "provenance": provenance.to_dict(),
            "role": role,
            "smoke_prerequisite_hash": smoke_prerequisite_hash,
        },
        domain="reproducibility-execution-role",
    )


def _receipt_pair_identities(
    *,
    config_hash: str,
    invocation_id: str,
    parallel_workers: int,
    provenance: ScientificProvenance,
    smoke_prerequisite_hash: str | None,
) -> dict[str, str]:
    return {
        "parallel": _receipt_role_identity(
            config_hash=config_hash,
            invocation_id=invocation_id,
            max_workers=parallel_workers,
            provenance=provenance,
            role="parallel",
            smoke_prerequisite_hash=smoke_prerequisite_hash,
        ),
        "serial": _receipt_role_identity(
            config_hash=config_hash,
            invocation_id=invocation_id,
            max_workers=1,
            provenance=provenance,
            role="serial",
            smoke_prerequisite_hash=smoke_prerequisite_hash,
        ),
    }


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Immutable pre-execution identity for one side of a paired sweep."""

    config_hash: str
    invocation_id: str
    max_workers: int
    pair: Mapping[str, str]
    parallel_workers: int
    provenance: ScientificProvenance
    role: str
    smoke_prerequisite_hash: str | None

    def __post_init__(self) -> None:
        try:
            immutable_pair = MappingProxyType(dict(self.pair))
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                "execution receipt pair identity is invalid"
            ) from error
        object.__setattr__(self, "pair", immutable_pair)
        if not is_sha256(self.config_hash):
            raise ReproducibilityError("execution receipt config_hash is invalid")
        if not is_sha256(self.invocation_id):
            raise ReproducibilityError("execution receipt invocation_id is invalid")
        if self.role not in {"serial", "parallel"}:
            raise ReproducibilityError("execution receipt role is invalid")
        if self.smoke_prerequisite_hash is not None and not is_sha256(
            self.smoke_prerequisite_hash
        ):
            raise ReproducibilityError(
                "execution receipt smoke prerequisite hash is invalid"
            )
        if not isinstance(self.provenance, ScientificProvenance):
            raise ReproducibilityError("execution receipt provenance is invalid")
        provenance = self.provenance.to_dict()
        if (
            any(
                not isinstance(value, str) or not value for value in provenance.values()
            )
            or any(
                not is_sha256(getattr(self.provenance, field))
                for field in (
                    "analysis_code_hash",
                    "dependency_lock_hash",
                    "dirty_tree_hash",
                    "environment_digest",
                )
            )
            or set(provenance) != set(_PROVENANCE_FIELDS)
        ):
            raise ReproducibilityError("execution receipt provenance is invalid")
        try:
            _validate_parallel_workers(self.parallel_workers)
            if self.role == "parallel":
                if self.max_workers != self.parallel_workers:
                    raise ValueError(
                        "parallel execution receipt workers do not match its pair"
                    )
            else:
                _validate_serial_workers(self.max_workers)
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                "execution receipt max_workers is invalid"
            ) from error
        if (
            tuple(self.pair) != _RECEIPT_PAIR_FIELDS
            or not all(is_sha256(value) for value in self.pair.values())
            or self.pair
            != _receipt_pair_identities(
                config_hash=self.config_hash,
                invocation_id=self.invocation_id,
                parallel_workers=self.parallel_workers,
                provenance=self.provenance,
                smoke_prerequisite_hash=self.smoke_prerequisite_hash,
            )
        ):
            raise ReproducibilityError("execution receipt pair identity is invalid")

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "invocation_id": self.invocation_id,
            "max_workers": self.max_workers,
            "pair": dict(self.pair),
            "parallel_workers": self.parallel_workers,
            "provenance": {
                field: self.provenance.to_dict()[field] for field in _PROVENANCE_FIELDS
            },
            "role": self.role,
            "smoke_prerequisite_hash": self.smoke_prerequisite_hash,
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(
            {
                "receipt_format": EXECUTION_RECEIPT_FORMAT,
                "schema_version": EXECUTION_RECEIPT_VERSION,
                "scientific": self.scientific_payload(),
            },
            domain="reproducibility-execution-receipt",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_format": EXECUTION_RECEIPT_FORMAT,
            "schema_version": EXECUTION_RECEIPT_VERSION,
            "scientific": self.scientific_payload(),
            "scientific_hash": self.scientific_hash,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_dict(cls, raw: object) -> ExecutionReceipt:
        receipt = _exact_mapping(raw, _RECEIPT_FIELDS, "execution receipt")
        if (
            receipt["receipt_format"] != EXECUTION_RECEIPT_FORMAT
            or receipt["schema_version"] != EXECUTION_RECEIPT_VERSION
            or isinstance(receipt["schema_version"], bool)
        ):
            raise ReproducibilityError("unsupported execution receipt")
        scientific = _exact_mapping(
            receipt["scientific"],
            _RECEIPT_SCIENTIFIC_FIELDS,
            "execution receipt scientific payload",
        )
        pair = _exact_mapping(
            scientific["pair"],
            _RECEIPT_PAIR_FIELDS,
            "execution receipt pair",
        )
        provenance_raw = _exact_mapping(
            scientific["provenance"],
            _PROVENANCE_FIELDS,
            "execution receipt provenance",
        )
        try:
            provenance = ScientificProvenance(**provenance_raw)
            parsed = cls(
                config_hash=scientific["config_hash"],
                invocation_id=scientific["invocation_id"],
                max_workers=scientific["max_workers"],
                pair=pair,
                parallel_workers=scientific["parallel_workers"],
                provenance=provenance,
                role=scientific["role"],
                smoke_prerequisite_hash=scientific["smoke_prerequisite_hash"],
            )
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                "execution receipt values are invalid"
            ) from error
        if (
            not is_sha256(receipt["scientific_hash"])
            or receipt["scientific_hash"] != parsed.scientific_hash
            or receipt != parsed.to_dict()
        ):
            raise ReproducibilityError(
                "execution receipt is noncanonical or scientifically invalid"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class ReproducibilityRun:
    cell_hash: str
    run_hash: str
    scientific_content_hash: str

    def __post_init__(self) -> None:
        if not all(
            is_sha256(value)
            for value in (
                self.cell_hash,
                self.run_hash,
                self.scientific_content_hash,
            )
        ):
            raise ValueError("reproducibility run hashes must be SHA-256 digests")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Exact-match evidence with operational locations kept out of its hash."""

    config_hash: str
    runs: tuple[ReproducibilityRun, ...]
    serial_root: Path
    parallel_root: Path
    parallel_workers: int
    invocation_id: str
    serial_receipt_hash: str
    parallel_receipt_hash: str

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if any(not isinstance(run, ReproducibilityRun) for run in runs):
            raise TypeError("runs must contain ReproducibilityRun values")
        object.__setattr__(self, "runs", runs)
        serial_root = _lexical_root(self.serial_root, "serial artifact root")
        parallel_root = _lexical_root(self.parallel_root, "parallel artifact root")
        object.__setattr__(self, "serial_root", serial_root)
        object.__setattr__(self, "parallel_root", parallel_root)
        if not is_sha256(self.config_hash):
            raise ValueError("config_hash must be a SHA-256 digest")
        if not is_sha256(self.invocation_id):
            raise ValueError("invocation_id must be a 64-character hexadecimal token")
        if (
            not all(
                is_sha256(value)
                for value in (
                    self.serial_receipt_hash,
                    self.parallel_receipt_hash,
                )
            )
            or self.serial_receipt_hash == self.parallel_receipt_hash
        ):
            raise ValueError("receipt hashes must be distinct SHA-256 digests")
        if not runs:
            raise ValueError("a reproducibility report requires at least one run")
        if len({run.cell_hash for run in runs}) != len(runs):
            raise ValueError("reproducibility report contains duplicate cells")
        if len({run.run_hash for run in runs}) != len(runs):
            raise ValueError("reproducibility report contains duplicate runs")
        _validate_parallel_workers(self.parallel_workers)
        _validate_disjoint_roots(serial_root, parallel_root)

    def scientific_payload(self) -> dict[str, Any]:
        ordered_runs = sorted(self.runs, key=lambda run: run.cell_hash)
        return {
            "comparison": "exact-serial-vs-parallel",
            "config_hash": self.config_hash,
            "exact_match": True,
            "execution_receipts": {
                "invocation_id": self.invocation_id,
                "parallel": self.parallel_receipt_hash,
                "serial": self.serial_receipt_hash,
            },
            "run_count": len(self.runs),
            "runs": [run.to_dict() for run in ordered_runs],
        }

    @property
    def scientific_hash(self) -> str:
        return scientific_hash(
            {
                "report_format": REPRODUCIBILITY_REPORT_FORMAT,
                "schema_version": REPRODUCIBILITY_REPORT_VERSION,
                "scientific": self.scientific_payload(),
            },
            domain="reproducibility-report",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operational": {
                "parallel": {
                    "artifact_root": str(self.parallel_root),
                    "max_workers": self.parallel_workers,
                },
                "serial": {
                    "artifact_root": str(self.serial_root),
                    "max_workers": 1,
                },
            },
            "report_format": REPRODUCIBILITY_REPORT_FORMAT,
            "schema_version": REPRODUCIBILITY_REPORT_VERSION,
            "scientific": self.scientific_payload(),
            "scientific_hash": self.scientific_hash,
        }

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        experiment: ExperimentConfig | None = None,
        expected_serial_root: str | Path | None = None,
        expected_parallel_root: str | Path | None = None,
        expected_parallel_workers: int | None = None,
    ) -> ReproducibilityReport:
        """Parse canonical evidence and optionally re-authenticate trusted roots."""

        report = _exact_mapping(raw, _REPORT_FIELDS, "reproducibility report")
        schema_version = report["schema_version"]
        if (
            report["report_format"] != REPRODUCIBILITY_REPORT_FORMAT
            or isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != REPRODUCIBILITY_REPORT_VERSION
        ):
            raise ReproducibilityError("unsupported reproducibility report")
        scientific = _exact_mapping(
            report["scientific"],
            _SCIENTIFIC_FIELDS,
            "reproducibility scientific payload",
        )
        if (
            scientific["comparison"] != "exact-serial-vs-parallel"
            or scientific["exact_match"] is not True
            or not is_sha256(scientific["config_hash"])
        ):
            raise ReproducibilityError("reproducibility scientific identity is invalid")
        run_count = scientific["run_count"]
        raw_runs = scientific["runs"]
        if (
            isinstance(run_count, bool)
            or not isinstance(run_count, int)
            or run_count < 1
            or not isinstance(raw_runs, list)
            or len(raw_runs) != run_count
        ):
            raise ReproducibilityError("reproducibility run count is invalid")
        try:
            runs = tuple(
                ReproducibilityRun(
                    **_exact_mapping(item, _RUN_FIELDS, "reproducibility run")
                )
                for item in raw_runs
            )
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                "reproducibility run identity is invalid"
            ) from error
        if tuple(item.cell_hash for item in runs) != tuple(
            sorted(item.cell_hash for item in runs)
        ):
            raise ReproducibilityError(
                "reproducibility runs are not in canonical cell order"
            )
        execution_receipts = _exact_mapping(
            scientific["execution_receipts"],
            _RECEIPT_REPORT_FIELDS,
            "reproducibility execution receipts",
        )
        if not all(
            is_sha256(execution_receipts[field]) for field in _RECEIPT_REPORT_FIELDS
        ):
            raise ReproducibilityError(
                "reproducibility execution receipt identities are invalid"
            )

        operational = _exact_mapping(
            report["operational"],
            _OPERATIONAL_FIELDS,
            "reproducibility operational payload",
        )
        serial = _exact_mapping(
            operational["serial"],
            _EXECUTION_FIELDS,
            "serial operation",
        )
        parallel = _exact_mapping(
            operational["parallel"],
            _EXECUTION_FIELDS,
            "parallel operation",
        )
        serial_root = _canonical_root(serial["artifact_root"], "serial artifact root")
        parallel_root = _canonical_root(
            parallel["artifact_root"],
            "parallel artifact root",
        )
        try:
            _validate_disjoint_roots(serial_root, parallel_root)
        except ValueError as error:
            raise ReproducibilityError(str(error)) from error
        if serial["max_workers"] != 1 or isinstance(serial["max_workers"], bool):
            raise ReproducibilityError("serial max_workers must equal 1")
        try:
            _validate_parallel_workers(parallel["max_workers"])
        except (TypeError, ValueError) as error:
            raise ReproducibilityError("parallel max_workers is invalid") from error

        expected = (
            expected_serial_root,
            expected_parallel_root,
            expected_parallel_workers,
        )
        if any(value is not None for value in expected):
            if any(value is None for value in expected):
                raise ValueError(
                    "all expected operational values must be supplied together"
                )
            assert expected_serial_root is not None
            assert expected_parallel_root is not None
            assert expected_parallel_workers is not None
            if (
                serial_root
                != _lexical_root(expected_serial_root, "expected serial artifact root")
                or parallel_root
                != _lexical_root(
                    expected_parallel_root,
                    "expected parallel artifact root",
                )
                or parallel["max_workers"] != expected_parallel_workers
            ):
                raise ReproducibilityError(
                    "reproducibility operational evidence differs from trusted roots"
                )

        try:
            parsed = cls(
                config_hash=scientific["config_hash"],
                runs=runs,
                serial_root=serial_root,
                parallel_root=parallel_root,
                parallel_workers=parallel["max_workers"],
                invocation_id=execution_receipts["invocation_id"],
                serial_receipt_hash=execution_receipts["serial"],
                parallel_receipt_hash=execution_receipts["parallel"],
            )
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                "reproducibility report metadata is invalid"
            ) from error
        if (
            not is_sha256(report["scientific_hash"])
            or report["scientific_hash"] != parsed.scientific_hash
            or parsed.to_dict() != report
        ):
            raise ReproducibilityError(
                "reproducibility report is noncanonical or scientifically invalid"
            )
        if experiment is not None:
            if not isinstance(experiment, ExperimentConfig):
                raise TypeError("experiment must be an ExperimentConfig or None")
            authenticated = authenticate_reproducibility_roots(
                experiment,
                serial_root=serial_root,
                parallel_root=parallel_root,
                parallel_workers=parallel["max_workers"],
            )
            if parsed != authenticated:
                raise ReproducibilityError(
                    "reproducibility report differs from authenticated roots"
                )
        return parsed


@dataclass(frozen=True, slots=True)
class _AuthenticatedRun:
    cell_hash: str
    run_hash: str
    content_hash: str


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(path))


def _lexical_root(value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path")
    encoded = str(value)
    path = Path(encoded)
    if (
        not encoded
        or encoded in {".", ".."}
        or str(path) != encoded
        or ".." in path.parts
    ):
        raise ValueError(f"{label} must be a canonical path without traversal")
    return path


def _validate_disjoint_roots(left: str | Path, right: str | Path) -> None:
    left_path = _absolute(left).resolve(strict=False)
    right_path = _absolute(right).resolve(strict=False)
    if (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    ):
        raise ValueError(
            "serial and parallel artifact roots must be distinct and non-overlapping"
        )


def _validate_parallel_workers(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("parallel_workers must be an integer")
    if value < 2:
        raise ValueError("parallel_workers must be at least 2")


def _validate_serial_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("serial max_workers must equal 1")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReproducibilityError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object,
    fields: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or tuple(value) != fields:
        raise ReproducibilityError(
            f"{label} fields or canonical field order are invalid"
        )
    return value


def _canonical_root(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ReproducibilityError(f"{label} must be a canonical path")
    try:
        return _lexical_root(value, label)
    except (TypeError, ValueError) as error:
        raise ReproducibilityError(
            f"{label} must be a canonical path without traversal"
        ) from error


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _receipt_path(root: Path) -> Path:
    return root / REPRODUCIBILITY_OPERATIONAL_DIRECTORY / EXECUTION_RECEIPT_FILENAME


@contextmanager
def _execution_root_pair_lock(
    serial_root: Path,
    parallel_root: Path,
) -> Iterator[None]:
    """Exclude ordinary root writers while paired ownership is established."""

    first, second = sorted(
        (serial_root, parallel_root),
        key=lambda value: os.path.abspath(value),
    )
    stack = ExitStack()
    try:
        stack.enter_context(artifact_root_lock(first, nonblocking=True))
        stack.enter_context(artifact_root_lock(second, nonblocking=True))
    except ArtifactRootBusyError as error:
        stack.close()
        raise ReproducibilityError(
            "reproducibility artifact root is already owned by another workflow"
        ) from error
    except BaseException:
        stack.close()
        raise
    try:
        yield
    finally:
        stack.close()


_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory_chain(path: str | Path, *, create: bool) -> int:
    """Open one directory without following any component-level symlink."""

    absolute = _absolute(path)
    descriptor = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_receipt_descriptor(root: str | Path) -> int:
    root_descriptor = _open_directory_chain(root, create=False)
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=root_descriptor,
        )
        receipt_descriptor = os.open(
            EXECUTION_RECEIPT_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(root_descriptor)
    return receipt_descriptor


@contextmanager
def _execution_pair_lock(
    serial_root: Path,
    parallel_root: Path,
) -> Iterator[None]:
    """Hold nonblocking locks on both immutable receipts for one invocation."""

    descriptors: list[int] = []
    try:
        for root in sorted(
            (serial_root, parallel_root),
            key=lambda value: os.path.abspath(value),
        ):
            descriptor = _open_receipt_descriptor(root)
            descriptors.append(descriptor)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ReproducibilityError(
                    "execution receipt lock target is not a regular file"
                )
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                raise ReproducibilityError(
                    "reproducibility invocation is already running"
                ) from error
        yield
    except ReproducibilityError:
        raise
    except OSError as error:
        raise ReproducibilityError(
            "cannot lock paired reproducibility invocation"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def load_execution_receipt(root: str | Path) -> ExecutionReceipt:
    """Load and byte-authenticate one exact operational execution receipt."""

    root_path = _absolute(root)
    directory = _receipt_path(root_path).parent
    root_descriptor: int | None = None
    directory_descriptor: int | None = None
    receipt_descriptor: int | None = None
    try:
        root_descriptor = _open_directory_chain(root_path, create=False)
        directory_descriptor = os.open(
            REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=root_descriptor,
        )
        entries = os.listdir(directory_descriptor)
        if entries != [EXECUTION_RECEIPT_FILENAME]:
            raise ReproducibilityError(
                "execution receipt directory contains missing or unexpected entries"
            )
        receipt_descriptor = os.open(
            EXECUTION_RECEIPT_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(receipt_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_EXECUTION_RECEIPT_BYTES
        ):
            raise ReproducibilityError(
                "execution receipt is not a bounded regular file"
            )
        content_bytes = bytearray()
        while len(content_bytes) <= MAX_EXECUTION_RECEIPT_BYTES:
            chunk = os.read(
                receipt_descriptor,
                min(
                    8192,
                    MAX_EXECUTION_RECEIPT_BYTES + 1 - len(content_bytes),
                ),
            )
            if not chunk:
                break
            content_bytes.extend(chunk)
        if len(content_bytes) > MAX_EXECUTION_RECEIPT_BYTES:
            raise ReproducibilityError("execution receipt exceeds its size limit")
        content = bytes(content_bytes).decode("utf-8")
        receipt = ExecutionReceipt.from_dict(
            parse_json_strict(content, label="execution receipt")
        )
    except ReproducibilityError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise ReproducibilityError(
            f"cannot inspect execution receipt directory: {directory}"
        ) from error
    finally:
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
    if content != receipt.to_json():
        raise ReproducibilityError("execution receipt bytes are not canonical")
    return receipt


def authenticate_execution_receipt(
    root: str | Path,
    experiment: ExperimentConfig,
    *,
    role: str,
) -> ExecutionReceipt:
    """Authenticate one receipt against an exact experiment and root role."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    if role not in {"serial", "parallel"}:
        raise ValueError("role must be serial or parallel")
    receipt = load_execution_receipt(root)
    if receipt.config_hash != experiment.config_hash or receipt.role != role:
        raise ReproducibilityError(
            "execution receipt does not match the exact experiment and root role"
        )
    if (experiment.phase == "calibration") != (
        receipt.smoke_prerequisite_hash is not None
    ):
        raise ReproducibilityError(
            "execution receipt smoke prerequisite binding does not match the phase"
        )
    return receipt


def _authenticate_execution_receipt_pair(
    experiment: ExperimentConfig,
    *,
    serial_root: Path,
    parallel_root: Path,
    parallel_workers: int,
    expected_provenance: ScientificProvenance | None = None,
    expected_smoke_prerequisite_hash: str | None = None,
) -> tuple[ExecutionReceipt, ExecutionReceipt]:
    serial = authenticate_execution_receipt(
        serial_root,
        experiment,
        role="serial",
    )
    parallel = authenticate_execution_receipt(
        parallel_root,
        experiment,
        role="parallel",
    )
    if (
        serial.invocation_id != parallel.invocation_id
        or serial.pair != parallel.pair
        or serial.parallel_workers != parallel.parallel_workers
        or serial.parallel_workers != parallel_workers
        or serial.max_workers != 1
        or parallel.max_workers != parallel_workers
        or serial.provenance != parallel.provenance
        or serial.smoke_prerequisite_hash != parallel.smoke_prerequisite_hash
    ):
        raise ReproducibilityError(
            "serial and parallel execution receipts are not one matching pair"
        )
    if expected_provenance is not None and serial.provenance != expected_provenance:
        raise ReproducibilityError(
            "execution receipts differ from current scientific provenance"
        )
    if (
        expected_smoke_prerequisite_hash is not None
        and serial.smoke_prerequisite_hash != expected_smoke_prerequisite_hash
    ):
        raise ReproducibilityError(
            "execution receipts bind a different Stage-0 smoke prerequisite"
        )
    return serial, parallel


@dataclass(slots=True)
class _PreparedReceipt:
    root: Path
    root_descriptor: int
    temporary_name: str
    temporary_descriptor: int


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("receipt write made no progress")
        view = view[written:]


def _discard_receipt_directory(root_descriptor: int, name: str) -> None:
    """Best-effort removal of one exact receipt directory created by this call."""

    directory_descriptor: int | None = None
    receipt_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=root_descriptor,
        )
        os.fchmod(directory_descriptor, 0o700)
        entries = os.listdir(directory_descriptor)
        if entries:
            if entries != [EXECUTION_RECEIPT_FILENAME]:
                return
            receipt_descriptor = os.open(
                EXECUTION_RECEIPT_FILENAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            if not stat.S_ISREG(os.fstat(receipt_descriptor).st_mode):
                return
            os.fchmod(receipt_descriptor, 0o600)
            os.close(receipt_descriptor)
            receipt_descriptor = None
            os.unlink(
                EXECUTION_RECEIPT_FILENAME,
                dir_fd=directory_descriptor,
            )
        os.close(directory_descriptor)
        directory_descriptor = None
        os.rmdir(name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
    except OSError:
        return
    finally:
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _prepare_receipt_directory(
    root: Path,
    receipt: ExecutionReceipt,
) -> _PreparedReceipt:
    root_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name = (
        f".{REPRODUCIBILITY_OPERATIONAL_DIRECTORY}.{secrets.token_hex(16)}.tmp"
    )
    try:
        root_descriptor = _open_directory_chain(root, create=True)
        os.mkdir(temporary_name, mode=0o700, dir_fd=root_descriptor)
        temporary_descriptor = os.open(
            temporary_name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=root_descriptor,
        )
        receipt_descriptor = os.open(
            EXECUTION_RECEIPT_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=temporary_descriptor,
        )
        try:
            _write_all(receipt_descriptor, receipt.to_json().encode("utf-8"))
            os.fsync(receipt_descriptor)
            os.fchmod(receipt_descriptor, 0o444)
        finally:
            os.close(receipt_descriptor)
        os.fsync(temporary_descriptor)
        os.fsync(root_descriptor)
        return _PreparedReceipt(
            root=root,
            root_descriptor=root_descriptor,
            temporary_name=temporary_name,
            temporary_descriptor=temporary_descriptor,
        )
    except (OSError, ReproducibilityError) as error:
        if root_descriptor is not None:
            _discard_receipt_directory(root_descriptor, temporary_name)
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        raise ReproducibilityError(
            "cannot create immutable execution receipt"
        ) from error


def _publish_prepared_receipt(prepared: _PreparedReceipt) -> None:
    final_descriptor: int | None = None
    destination_created = False
    try:
        os.mkdir(
            REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            mode=0o700,
            dir_fd=prepared.root_descriptor,
        )
        destination_created = True
        final_descriptor = os.open(
            REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=prepared.root_descriptor,
        )
        os.rename(
            EXECUTION_RECEIPT_FILENAME,
            EXECUTION_RECEIPT_FILENAME,
            src_dir_fd=prepared.temporary_descriptor,
            dst_dir_fd=final_descriptor,
        )
        os.fsync(final_descriptor)
        os.fchmod(final_descriptor, 0o555)
        os.fchmod(prepared.temporary_descriptor, 0o700)
        os.rmdir(
            prepared.temporary_name,
            dir_fd=prepared.root_descriptor,
        )
        os.fsync(prepared.root_descriptor)
    except BaseException:
        if final_descriptor is not None:
            os.close(final_descriptor)
            final_descriptor = None
        if destination_created:
            _discard_receipt_directory(
                prepared.root_descriptor,
                REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            )
        raise
    finally:
        if final_descriptor is not None:
            os.close(final_descriptor)


def _close_prepared_receipt(prepared: _PreparedReceipt) -> None:
    os.close(prepared.temporary_descriptor)
    os.close(prepared.root_descriptor)


def _create_execution_receipt_pair(
    experiment: ExperimentConfig,
    *,
    serial_root: Path,
    parallel_root: Path,
    parallel_workers: int,
    provenance: ScientificProvenance,
    smoke_prerequisite_hash: str | None = None,
) -> tuple[ExecutionReceipt, ExecutionReceipt]:
    if (experiment.phase == "calibration") != (smoke_prerequisite_hash is not None):
        raise ReproducibilityError(
            "calibration receipt creation requires one Stage-0 prerequisite hash"
        )
    if smoke_prerequisite_hash is not None and not is_sha256(smoke_prerequisite_hash):
        raise ReproducibilityError("smoke_prerequisite_hash is invalid")
    invocation_id = secrets.token_hex(32)
    pair = _receipt_pair_identities(
        config_hash=experiment.config_hash,
        invocation_id=invocation_id,
        parallel_workers=parallel_workers,
        provenance=provenance,
        smoke_prerequisite_hash=smoke_prerequisite_hash,
    )
    serial = ExecutionReceipt(
        config_hash=experiment.config_hash,
        invocation_id=invocation_id,
        max_workers=1,
        pair=pair,
        parallel_workers=parallel_workers,
        provenance=provenance,
        role="serial",
        smoke_prerequisite_hash=smoke_prerequisite_hash,
    )
    parallel = ExecutionReceipt(
        config_hash=experiment.config_hash,
        invocation_id=invocation_id,
        max_workers=parallel_workers,
        pair=pair,
        parallel_workers=parallel_workers,
        provenance=provenance,
        role="parallel",
        smoke_prerequisite_hash=smoke_prerequisite_hash,
    )
    serial_temporary: _PreparedReceipt | None = None
    parallel_temporary: _PreparedReceipt | None = None
    serial_published_by_call = False
    parallel_published_by_call = False
    try:
        serial_temporary = _prepare_receipt_directory(serial_root, serial)
        parallel_temporary = _prepare_receipt_directory(parallel_root, parallel)
        _publish_prepared_receipt(serial_temporary)
        serial_published_by_call = True
        _publish_prepared_receipt(parallel_temporary)
        parallel_published_by_call = True
    except (OSError, ReproducibilityError) as error:
        if serial_temporary is not None:
            _discard_receipt_directory(
                serial_temporary.root_descriptor,
                serial_temporary.temporary_name,
            )
        if parallel_temporary is not None:
            _discard_receipt_directory(
                parallel_temporary.root_descriptor,
                parallel_temporary.temporary_name,
            )
        if serial_published_by_call:
            _discard_receipt_directory(
                serial_temporary.root_descriptor,
                REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            )
        if parallel_published_by_call:
            _discard_receipt_directory(
                parallel_temporary.root_descriptor,
                REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
            )
        if isinstance(error, ReproducibilityError):
            raise
        raise ReproducibilityError(
            "cannot publish paired immutable execution receipts"
        ) from error
    finally:
        if serial_temporary is not None:
            _close_prepared_receipt(serial_temporary)
        if parallel_temporary is not None:
            _close_prepared_receipt(parallel_temporary)
    return serial, parallel


def _artifacts_by_type(
    result: RunResult,
    *,
    side: str,
    session: ArtifactValidationSession,
) -> dict[str, list[Any]]:
    try:
        artifacts = validate_artifact_tree(result.path, session=session)
    except ScientificArtifactError as error:
        raise ReproducibilityError(
            f"{side} run {result.run_hash} failed artifact authentication: {error}"
        ) from error
    by_type: dict[str, list[Any]] = {}
    for artifact in artifacts:
        by_type.setdefault(artifact.artifact_type, []).append(artifact)
    return by_type


def _authenticate_side(
    experiment: ExperimentConfig,
    results: Iterable[RunResult],
    *,
    expected_receipt: ExecutionReceipt,
    root: Path,
    session: ArtifactValidationSession,
    side: str,
) -> dict[str, _AuthenticatedRun]:
    observed = tuple(results)
    cells = experiment.cells()
    expected_cells = {cell.cell_hash: cell for cell in cells}
    if len(expected_cells) != len(cells):
        raise ReproducibilityError("experiment contains duplicate run cells")
    if len(observed) != len(cells):
        raise ReproducibilityError(
            f"{side} inventory is incomplete: expected {len(cells)} runs, "
            f"received {len(observed)}"
        )
    if any(not isinstance(result, RunResult) for result in observed):
        raise TypeError(f"{side} results must contain only RunResult values")
    run_hashes = [result.run_hash for result in observed]
    if len(set(run_hashes)) != len(run_hashes):
        raise ReproducibilityError(
            f"{side} inventory contains duplicate run identities"
        )

    expected_settings = _json_safe(experiment.resolved_run_settings())
    authenticated: dict[str, _AuthenticatedRun] = {}
    for result in sorted(observed, key=lambda item: item.run_hash):
        if (
            not result.complete
            or result.event_count != experiment.horizon
            or not is_sha256(result.run_hash)
            or not is_sha256(result.scientific_content_hash)
        ):
            raise ReproducibilityError(
                f"{side} result {result.run_hash} is incomplete or malformed"
            )
        expected_path = ArtifactStore.for_run(
            root, experiment.name, result.run_hash
        ).path
        if _absolute(result.path) != _absolute(expected_path):
            raise ReproducibilityError(
                f"{side} result path is outside its operational artifact root"
            )

        by_type = _artifacts_by_type(result, side=side, session=session)
        configs = by_type.get("resolved-run-config", [])
        manifests = by_type.get("run-manifest", [])
        if len(configs) != 1 or len(manifests) != 1:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} lacks singleton identity artifacts"
            )
        config_envelope = configs[0]
        manifest = manifests[0]
        config = _mapping(config_envelope.payload, "resolved run config")
        if set(config) != {
            "cell",
            "provenance",
            "run_hash",
            "run_settings",
            "seeds",
        }:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} has invalid identity fields"
            )
        if config["run_settings"] != expected_settings:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} used different experiment settings"
            )
        try:
            cell = run_cell_from_dict(config["cell"])
            seeds = RunSeeds(**_mapping(config["seeds"], "run seeds"))
            provenance = ScientificProvenance(
                **_mapping(config["provenance"], "run provenance")
            )
        except (TypeError, ValueError) as error:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} has malformed identity inputs"
            ) from error
        if provenance != expected_receipt.provenance:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} differs from its execution receipt "
                "provenance"
            )
        expected_cell = expected_cells.get(cell.cell_hash)
        if expected_cell is None or cell != expected_cell:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} is not in the experiment inventory"
            )
        if cell.cell_hash in authenticated:
            raise ReproducibilityError(f"{side} inventory contains a duplicate cell")
        if seeds != SeedBank(
            experiment.master_seed,
            experiment.algorithm_master_seed,
        ).for_cell(cell):
            raise ReproducibilityError(
                f"{side} run {result.run_hash} used a different seed bank"
            )
        expected_semantics = semantic_hashes(
            cell,
            analysis_code_hash=provenance.analysis_code_hash,
        )
        if config_envelope.semantic_hashes != expected_semantics:
            raise ReproducibilityError(
                f"{side} run {result.run_hash} has incompatible semantics"
            )
        expected_run_hash = run_identity(experiment, cell, seeds, provenance)
        content_hash = manifest.payload.get("scientific_content_hash")
        if (
            config["run_hash"] != expected_run_hash
            or result.run_hash != expected_run_hash
            or result.scientific_content_hash != content_hash
        ):
            raise ReproducibilityError(
                f"{side} result metadata differs from its authenticated tree"
            )
        authenticated[cell.cell_hash] = _AuthenticatedRun(
            cell_hash=cell.cell_hash,
            run_hash=result.run_hash,
            content_hash=result.scientific_content_hash,
        )

    if set(authenticated) != set(expected_cells):
        raise ReproducibilityError(f"{side} cell inventory is incomplete")
    experiment_path = ArtifactStore.for_run(root, experiment.name, "0" * 64).path.parent
    try:
        entries = tuple(experiment_path.iterdir())
    except OSError as error:
        raise ReproducibilityError(
            f"cannot inspect {side} operational inventory"
        ) from error
    if any(entry.is_symlink() or not entry.is_dir() for entry in entries):
        raise ReproducibilityError(
            f"{side} operational inventory contains an invalid entry"
        )
    stored_run_hashes = {entry.name for entry in entries}
    if stored_run_hashes != set(run_hashes):
        raise ReproducibilityError(
            f"{side} operational inventory contains missing or unexpected runs"
        )
    return authenticated


def _discover_results(
    experiment: ExperimentConfig,
    *,
    root: Path,
    side: str,
) -> tuple[RunResult, ...]:
    """Read existing result metadata without creating or locking artifacts."""

    experiment_path = ArtifactStore.for_run(
        root,
        experiment.name,
        "0" * 64,
    ).path.parent
    try:
        entries = tuple(sorted(experiment_path.iterdir(), key=lambda item: item.name))
    except OSError as error:
        raise ReproducibilityError(
            f"cannot inspect {side} operational inventory"
        ) from error
    if len(entries) != len(experiment.cells()):
        raise ReproducibilityError(
            f"{side} operational inventory is incomplete or has extra entries"
        )
    if any(
        entry.is_symlink() or not entry.is_dir() or not is_sha256(entry.name)
        for entry in entries
    ):
        raise ReproducibilityError(
            f"{side} operational inventory contains an invalid entry"
        )

    results = []
    for entry in entries:
        try:
            manifest = read_artifact(entry / "manifest.json")
            metrics = read_artifact(entry / "metrics.json")
        except ScientificArtifactError as error:
            raise ReproducibilityError(
                f"{side} run {entry.name} lacks readable completion metadata"
            ) from error
        if (
            manifest.artifact_type != "run-manifest"
            or metrics.artifact_type != "run-metrics"
        ):
            raise ReproducibilityError(
                f"{side} run {entry.name} has invalid completion metadata"
            )
        manifest_payload = _mapping(manifest.payload, "run manifest")
        metrics_payload = _mapping(metrics.payload, "run metrics")
        results.append(
            RunResult(
                run_hash=entry.name,
                path=entry,
                complete=True,
                scientific_content_hash=manifest_payload.get("scientific_content_hash"),
                event_count=metrics_payload.get("event_count"),
            )
        )
    return tuple(results)


def compare_reproducibility_results(
    experiment: ExperimentConfig,
    serial_results: Iterable[RunResult],
    parallel_results: Iterable[RunResult],
    *,
    serial_root: str | Path,
    parallel_root: str | Path,
    parallel_workers: int,
    validation_session: ArtifactValidationSession | None = None,
) -> ReproducibilityReport:
    """Authenticate two complete sweep results and require exact equality."""

    _validate_parallel_workers(parallel_workers)
    if validation_session is not None and not isinstance(
        validation_session,
        ArtifactValidationSession,
    ):
        raise TypeError("validation_session must be ArtifactValidationSession or None")
    session = validation_session or ArtifactValidationSession()
    serial_declaration = _lexical_root(serial_root, "serial artifact root")
    parallel_declaration = _lexical_root(parallel_root, "parallel artifact root")
    serial_path = _absolute(serial_declaration)
    parallel_path = _absolute(parallel_declaration)
    _validate_disjoint_roots(serial_path, parallel_path)
    serial_receipt, parallel_receipt = _authenticate_execution_receipt_pair(
        experiment,
        serial_root=serial_path,
        parallel_root=parallel_path,
        parallel_workers=parallel_workers,
    )
    serial = _authenticate_side(
        experiment,
        serial_results,
        expected_receipt=serial_receipt,
        root=serial_path,
        session=session,
        side="serial",
    )
    parallel = _authenticate_side(
        experiment,
        parallel_results,
        expected_receipt=parallel_receipt,
        root=parallel_path,
        session=session,
        side="parallel",
    )
    runs = []
    for cell_hash in sorted(serial):
        left = serial[cell_hash]
        right = parallel[cell_hash]
        if left.run_hash != right.run_hash:
            raise ReproducibilityError(f"run identity mismatch for cell {cell_hash}")
        if left.content_hash != right.content_hash:
            raise ReproducibilityError(
                f"scientific content mismatch for cell {cell_hash}"
            )
        runs.append(
            ReproducibilityRun(
                cell_hash=cell_hash,
                run_hash=left.run_hash,
                scientific_content_hash=left.content_hash,
            )
        )
    return ReproducibilityReport(
        config_hash=experiment.config_hash,
        runs=tuple(runs),
        serial_root=serial_declaration,
        parallel_root=parallel_declaration,
        parallel_workers=parallel_workers,
        invocation_id=serial_receipt.invocation_id,
        serial_receipt_hash=serial_receipt.scientific_hash,
        parallel_receipt_hash=parallel_receipt.scientific_hash,
    )


def authenticate_reproducibility_roots(
    experiment: ExperimentConfig,
    *,
    serial_root: str | Path,
    parallel_root: str | Path,
    parallel_workers: int,
    validation_session: ArtifactValidationSession | None = None,
) -> ReproducibilityReport:
    """Rebuild canonical reproducibility evidence from two existing roots."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    _validate_parallel_workers(parallel_workers)
    serial_declaration = _lexical_root(serial_root, "serial artifact root")
    parallel_declaration = _lexical_root(parallel_root, "parallel artifact root")
    serial_path = _absolute(serial_declaration)
    parallel_path = _absolute(parallel_declaration)
    _validate_disjoint_roots(serial_path, parallel_path)
    serial = _discover_results(
        experiment,
        root=serial_path,
        side="serial",
    )
    parallel = _discover_results(
        experiment,
        root=parallel_path,
        side="parallel",
    )
    return compare_reproducibility_results(
        experiment,
        serial,
        parallel,
        serial_root=serial_declaration,
        parallel_root=parallel_declaration,
        parallel_workers=parallel_workers,
        validation_session=validation_session,
    )


def _validate_fresh_execution_roots(roots: tuple[Path, Path]) -> None:
    for root in roots:
        try:
            if os.path.lexists(root) and (
                root.is_symlink() or not root.is_dir() or any(root.iterdir())
            ):
                raise ReproducibilityError(
                    "reproducibility runs require absent or empty fresh artifact roots"
                )
        except ReproducibilityError:
            raise
        except OSError as error:
            raise ReproducibilityError(
                f"cannot inspect fresh reproducibility root: {root}"
            ) from error


def _preflight_resume_run_tree(
    experiment: ExperimentConfig,
    *,
    root: Path,
    receipt: ExecutionReceipt,
    expected_cells: Mapping[str, Any],
    expected_settings: object,
    adapter_factory: Callable[[], ExperimentAdapter],
    session: ArtifactValidationSession,
    side: str,
) -> None:
    store = ArtifactStore(root)
    manifest_path = root / "manifest.json"
    try:
        if manifest_path.exists():
            artifacts = validate_artifact_tree(root, session=session)
            paths = store.list_artifacts()
        else:
            paths = store.list_artifacts()
            artifacts = tuple(read_artifact(path) for path in paths)
    except ScientificArtifactError as error:
        raise ReproducibilityError(
            f"{side} existing run {root.name} is not safely resumable: {error}"
        ) from error
    if not artifacts:
        return
    records = dict(zip(paths, artifacts, strict=True))
    configs = tuple(
        (path, artifact)
        for path, artifact in records.items()
        if artifact.artifact_type == "resolved-run-config"
    )
    if len(configs) != 1:
        raise ReproducibilityError(
            f"{side} partial run {root.name} lacks one resolved config"
        )
    config_path, config_envelope = configs[0]
    if config_path != root / "config.resolved.json":
        raise ReproducibilityError(
            f"{side} partial run {root.name} stores its config at an invalid path"
        )
    config = _mapping(config_envelope.payload, "resolved run config")
    if (
        set(config) != {"cell", "provenance", "run_hash", "run_settings", "seeds"}
        or config["run_hash"] != root.name
        or config["run_settings"] != expected_settings
    ):
        raise ReproducibilityError(
            f"{side} partial run {root.name} has incompatible identity metadata"
        )
    try:
        cell = run_cell_from_dict(config["cell"])
        seeds = RunSeeds(**_mapping(config["seeds"], "run seeds"))
        provenance = ScientificProvenance(
            **_mapping(config["provenance"], "run provenance")
        )
    except (TypeError, ValueError) as error:
        raise ReproducibilityError(
            f"{side} partial run {root.name} has malformed identity metadata"
        ) from error
    if (
        expected_cells.get(cell.cell_hash) != cell
        or provenance != receipt.provenance
        or seeds
        != SeedBank(
            experiment.master_seed,
            experiment.algorithm_master_seed,
        ).for_cell(cell)
        or run_identity(experiment, cell, seeds, provenance) != root.name
    ):
        raise ReproducibilityError(
            f"{side} partial run {root.name} is outside this invocation"
        )
    if manifest_path.exists():
        return

    hashes = semantic_hashes(
        cell,
        analysis_code_hash=provenance.analysis_code_hash,
    )
    checkpoint_paths: dict[int, Any] = {}
    metrics = None
    frontier_reference = None
    for path, artifact in records.items():
        relative = path.relative_to(root)
        if artifact.semantic_hashes != hashes:
            raise ReproducibilityError(
                f"{side} partial run {root.name} has incompatible semantics"
            )
        if artifact.artifact_type == "resolved-run-config":
            continue
        if (
            artifact.artifact_type == "frontier-reference"
            and relative == Path("frontier-reference.json")
            and frontier_reference is None
        ):
            frontier_reference = artifact
            continue
        if (
            artifact.artifact_type == "run-metrics"
            and relative == Path("metrics.json")
            and metrics is None
        ):
            metrics = artifact
            continue
        if (
            artifact.artifact_type == "training-event"
            and relative.parent == Path("events")
            and relative.stem.isdigit()
            and len(relative.stem) == 8
        ):
            continue
        if (
            artifact.artifact_type == "run-checkpoint"
            and relative.parent == Path("checkpoints")
            and relative.stem.isdigit()
            and len(relative.stem) == 8
        ):
            round_index = int(relative.stem)
            if round_index in checkpoint_paths:
                raise ReproducibilityError(
                    f"{side} partial run {root.name} duplicates a checkpoint"
                )
            checkpoint_paths[round_index] = artifact
            continue
        raise ReproducibilityError(
            f"{side} partial run {root.name} contains an unexpected artifact"
        )
    if frontier_reference is not None:
        reference = _mapping(
            frontier_reference.payload,
            "partial frontier reference",
        )
        if (
            set(reference) != {"artifact_hash", "frontier_hash"}
            or reference["frontier_hash"] != hashes["frontier"]
            or not is_sha256(reference["artifact_hash"])
        ):
            raise ReproducibilityError(
                f"{side} partial run {root.name} has an invalid frontier reference"
            )
        frontier_root = root.parents[1] / "_frontiers" / hashes["frontier"]
        try:
            frontier_manifest = read_artifact(
                frontier_root / "frontier/manifest.json",
                expected_semantic_hashes={"frontier": hashes["frontier"]},
            )
        except ScientificArtifactError as error:
            raise ReproducibilityError(
                f"{side} partial run {root.name} references an incomplete frontier"
            ) from error
        if (
            frontier_manifest.artifact_type != "frontier-manifest"
            or frontier_manifest.scientific_hash != reference["artifact_hash"]
        ):
            raise ReproducibilityError(
                f"{side} partial run {root.name} references a different frontier"
            )

    try:
        events = EventJournal(store, hashes).events()
    except ScientificArtifactError as error:
        raise ReproducibilityError(
            f"{side} partial run {root.name} has an invalid event journal"
        ) from error
    if len(events) > experiment.horizon:
        raise ReproducibilityError(
            f"{side} partial run {root.name} exceeds its registered horizon"
        )
    checkpoint_rounds = set(experiment.checkpoints.rounds)
    if any(
        round_index not in checkpoint_rounds or round_index > len(events)
        for round_index in checkpoint_paths
    ):
        raise ReproducibilityError(
            f"{side} partial run {root.name} has an impossible checkpoint"
        )
    adapter = adapter_factory()
    state = adapter.initial_state(cell, seeds)
    for round_index in range(len(events) + 1):
        checkpoint = checkpoint_paths.get(round_index)
        if checkpoint is not None:
            current = adapter.state_fingerprint(state)
            evaluation_adapter, evaluation_state = copy.deepcopy((adapter, state))
            result = evaluation_adapter.checkpoint(
                evaluation_state,
                round_index,
                cell,
                seeds,
                hashes,
            )
            expected_checkpoint = {
                "round": round_index,
                "training_state_before": current,
                "training_state_after": current,
                "evaluation_seed": seeds.evaluation,
                "deployment_seed": seeds.deployment,
                "result": result,
            }
            if (
                evaluation_adapter.state_fingerprint(evaluation_state) != current
                or checkpoint.payload != expected_checkpoint
            ):
                raise ReproducibilityError(
                    f"{side} partial run {root.name} has an invalid checkpoint"
                )
        if round_index == len(events):
            break
        event = events[round_index]
        expected_event = adapter.training_event(
            state,
            round_index,
            cell,
            seeds,
        )
        if (
            event.sequence != round_index
            or event.event_key != f"round:{round_index}"
            or event.event_kind != "training-step"
            or event.payload != expected_event
        ):
            raise ReproducibilityError(
                f"{side} partial run {root.name} differs from deterministic replay"
            )
        state = adapter.apply_training_event(state, event.payload)
    if metrics is not None:
        expected_metrics = {
            "completed_rounds": experiment.horizon,
            "event_count": experiment.horizon,
            "final_state_hash": adapter.state_fingerprint(state),
            "phase": experiment.phase,
            "confirmatory_frozen": experiment.confirmatory_frozen,
        }
        if len(events) != experiment.horizon or metrics.payload != expected_metrics:
            raise ReproducibilityError(
                f"{side} partial run {root.name} has invalid completion metrics"
            )


def _preflight_resume_frontier_tree(
    cell: Any,
    *,
    root: Path,
    adapter_factory: Callable[[], ExperimentAdapter],
    session: ArtifactValidationSession,
    side: str,
) -> None:
    hashes = {"frontier": root.name}
    try:
        if (root / "frontier/manifest.json").exists():
            validate_artifact_tree(
                root,
                expected_semantic_hashes=hashes,
                session=session,
            )
            return
        store = ArtifactStore(root)
        paths = store.list_artifacts()
        artifacts = tuple(
            read_artifact(path, expected_semantic_hashes=hashes) for path in paths
        )
    except ScientificArtifactError as error:
        raise ReproducibilityError(
            f"{side} existing frontier {root.name} is not safely resumable: {error}"
        ) from error
    if not artifacts:
        return
    frontier = adapter_factory().frontier(cell)
    expected: dict[str, tuple[str, object]] = {
        "frontier/curve.json": ("frontier-curve", frontier["curve"]),
        "frontier/diagnostics.json": (
            "frontier-diagnostics",
            frontier["diagnostics"],
        ),
    }
    expected.update(
        {
            f"frontier/witnesses/{name}.json": ("frontier-witness", payload)
            for name, payload in frontier["witnesses"].items()
        }
    )
    expected.update(
        {
            f"frontier/certificates/{name}.json": (
                "frontier-certificate",
                payload,
            )
            for name, payload in frontier["certificates"].items()
        }
    )
    for path, artifact in zip(paths, artifacts, strict=True):
        relative = path.relative_to(root).as_posix()
        expected_record = expected.get(relative)
        if (
            expected_record is None
            or (
                artifact.artifact_type,
                artifact.payload,
            )
            != expected_record
        ):
            raise ReproducibilityError(
                f"{side} partial frontier {root.name} contains an unexpected "
                "or incompatible artifact"
            )


def _validate_resume_root(
    experiment: ExperimentConfig,
    *,
    root: Path,
    receipt: ExecutionReceipt,
    adapter_factory: Callable[[], ExperimentAdapter],
    session: ArtifactValidationSession,
    side: str,
) -> None:
    try:
        root_metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ReproducibilityError(f"{side} resume root is not a real directory")
        with os.scandir(root) as stream:
            entries = tuple(stream)
    except ReproducibilityError:
        raise
    except OSError as error:
        raise ReproducibilityError(f"cannot inspect {side} resume root") from error
    allowed = {
        REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
        "_frontiers",
        experiment.name,
    }
    if REPRODUCIBILITY_OPERATIONAL_DIRECTORY not in {
        entry.name for entry in entries
    } or any(
        entry.name not in allowed
        or entry.is_symlink()
        or not entry.is_dir(follow_symlinks=False)
        for entry in entries
    ):
        raise ReproducibilityError(
            f"{side} resume root contains an invalid operational entry"
        )

    expected_run_hashes = {
        run_identity(
            experiment,
            cell,
            SeedBank(
                experiment.master_seed,
                experiment.algorithm_master_seed,
            ).for_cell(cell),
            receipt.provenance,
        )
        for cell in experiment.cells()
    }
    expected_frontier_cells = {
        semantic_hashes(
            cell,
            analysis_code_hash=receipt.provenance.analysis_code_hash,
        )["frontier"]: cell
        for cell in experiment.cells()
    }
    expected_frontier_hashes = set(expected_frontier_cells)
    expected_cells = {cell.cell_hash: cell for cell in experiment.cells()}
    expected_settings = _json_safe(experiment.resolved_run_settings())
    for directory_name, expected, label in (
        (experiment.name, expected_run_hashes, "run"),
        ("_frontiers", expected_frontier_hashes, "frontier"),
    ):
        directory = root / directory_name
        if not os.path.lexists(directory):
            continue
        try:
            with os.scandir(directory) as stream:
                children = tuple(stream)
        except OSError as error:
            raise ReproducibilityError(
                f"cannot inspect {side} partial {label} inventory"
            ) from error
        if any(
            child.name not in expected
            or child.is_symlink()
            or not child.is_dir(follow_symlinks=False)
            for child in children
        ):
            raise ReproducibilityError(
                f"{side} partial {label} inventory contains an unexpected entry"
            )
        for child in children:
            child_path = Path(child.path)
            try:
                with artifact_tree_lock(child_path):
                    cleanup_orphaned_artifact_temporaries(child_path)
                    if label == "run":
                        _preflight_resume_run_tree(
                            experiment,
                            root=child_path,
                            receipt=receipt,
                            expected_cells=expected_cells,
                            expected_settings=expected_settings,
                            adapter_factory=adapter_factory,
                            session=session,
                            side=side,
                        )
                        continue
                    _preflight_resume_frontier_tree(
                        expected_frontier_cells[child.name],
                        root=child_path,
                        adapter_factory=adapter_factory,
                        session=session,
                        side=side,
                    )
            except ReproducibilityError:
                raise
            except ScientificArtifactError as error:
                raise ReproducibilityError(
                    f"{side} existing {label} {child.name} is not safely "
                    f"resumable: {error}"
                ) from error


def run_reproducibility_check(
    experiment: ExperimentConfig,
    *,
    serial_root: str | Path,
    parallel_root: str | Path,
    parallel_workers: int = 2,
    adapter_factory: Callable[[], ExperimentAdapter] = ExactSymbolicAdapter,
    provenance: ScientificProvenance | None = None,
    resume: bool = False,
    smoke_prerequisite_hash: str | None = None,
) -> ReproducibilityReport:
    """Run or explicitly resume paired sweeps, then authenticate exact equality."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    if (experiment.phase == "calibration") != (smoke_prerequisite_hash is not None):
        raise ReproducibilityError(
            "calibration reproducibility requires one Stage-0 prerequisite hash"
        )
    if smoke_prerequisite_hash is not None and not is_sha256(smoke_prerequisite_hash):
        raise ReproducibilityError("smoke_prerequisite_hash is invalid")
    _validate_parallel_workers(parallel_workers)
    serial_declaration = _lexical_root(serial_root, "serial artifact root")
    parallel_declaration = _lexical_root(parallel_root, "parallel artifact root")
    serial_path = _absolute(serial_declaration)
    parallel_path = _absolute(parallel_declaration)
    _validate_disjoint_roots(serial_path, parallel_path)
    if provenance is not None and not isinstance(provenance, ScientificProvenance):
        raise TypeError("provenance must be ScientificProvenance or None")
    frozen_provenance = provenance or collect_provenance()
    roots = (serial_path, parallel_path)
    with _execution_root_pair_lock(serial_path, parallel_path):
        if not resume:
            _validate_fresh_execution_roots(roots)
        experiment_paths = (
            ArtifactStore.for_run(
                serial_path,
                experiment.name,
                "0" * 64,
            ).path.parent,
            ArtifactStore.for_run(
                parallel_path,
                experiment.name,
                "0" * 64,
            ).path.parent,
        )
        if not resume and any(os.path.lexists(path) for path in experiment_paths):
            raise ReproducibilityError(
                "reproducibility runs require absent, fresh experiment directories"
            )
        if not resume:
            _create_execution_receipt_pair(
                experiment,
                serial_root=serial_path,
                parallel_root=parallel_path,
                parallel_workers=parallel_workers,
                provenance=frozen_provenance,
                smoke_prerequisite_hash=smoke_prerequisite_hash,
            )
        with _execution_pair_lock(serial_path, parallel_path):
            validation_session = ArtifactValidationSession()
            serial_receipt, parallel_receipt = _authenticate_execution_receipt_pair(
                experiment,
                serial_root=serial_path,
                parallel_root=parallel_path,
                parallel_workers=parallel_workers,
                expected_provenance=frozen_provenance,
                expected_smoke_prerequisite_hash=smoke_prerequisite_hash,
            )
            if resume:
                for root, side, receipt in (
                    (serial_path, "serial", serial_receipt),
                    (parallel_path, "parallel", parallel_receipt),
                ):
                    _validate_resume_root(
                        experiment,
                        root=root,
                        receipt=receipt,
                        adapter_factory=adapter_factory,
                        session=validation_session,
                        side=side,
                    )
            serial = SweepRunner(
                RunExecutor(
                    serial_path,
                    adapter_factory,
                    provenance=frozen_provenance,
                    validation_session=validation_session,
                    reproducibility_mode=True,
                )
            ).run(experiment, max_workers=1)
            parallel = SweepRunner(
                RunExecutor(
                    parallel_path,
                    adapter_factory,
                    provenance=frozen_provenance,
                    validation_session=validation_session,
                    reproducibility_mode=True,
                )
            ).run(experiment, max_workers=parallel_workers)
            return compare_reproducibility_results(
                experiment,
                serial,
                parallel,
                serial_root=serial_declaration,
                parallel_root=parallel_declaration,
                parallel_workers=parallel_workers,
                validation_session=validation_session,
            )
