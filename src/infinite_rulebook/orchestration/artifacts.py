"""Validated immutable artifacts and restart-safe event journals."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from infinite_rulebook.agents import CapabilityManifest
from infinite_rulebook.artifacts import semantic_hash as typed_semantic_hash
from infinite_rulebook.frontier.blahut_arimoto import solve_lagrangian
from infinite_rulebook.frontier.finite_problem import (
    ChannelWitness,
    FiniteDecisionProblem,
)
from infinite_rulebook.frontier.inversion import FrontierSolution
from infinite_rulebook.metrics import FrontierPoint
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import parse_json_strict

ARTIFACT_SCHEMA_VERSION = 1
_EVENT_NAME = re.compile(r"^(\d{8})\.json$")
_CHECKPOINT_NAME = re.compile(r"^\d{8}\.json$")
_ORPHAN_TEMP_NAME = re.compile(r"^\.(?P<target>[^/]+\.json)\.[0-9a-f]{24}$")


class ScientificArtifactError(ValueError):
    """Raised when an artifact is invalid, incompatible, or would be mutated."""


class ArtifactRootBusyError(ScientificArtifactError):
    """Raised when another workflow already owns an artifact root."""


@dataclass(slots=True)
class ArtifactValidationSession:
    """Caller-owned cache populated only by successful frontier validation."""

    _validated_frontiers: dict[tuple[str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _validated_frontier_trees: dict[tuple[str, str], tuple[ArtifactEnvelope, ...]] = (
        field(default_factory=dict, init=False, repr=False)
    )
    _frontier_inflight: dict[tuple[str, str], threading.Event] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )


class _ArtifactNotFound(ScientificArtifactError):
    """Internal signal used to distinguish an absent immutable artifact."""


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    artifact_type: str
    semantic_hashes: dict[str, str]
    payload: Any
    scientific_hash: str
    runtime_metadata: dict[str, Any]
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        artifact_type: str,
        semantic_hashes: dict[str, str],
        payload: Any,
        *,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        if not isinstance(semantic_hashes, dict):
            raise ScientificArtifactError("semantic_hashes must be an object")
        try:
            normalized_payload = json.loads(
                json.dumps(payload, allow_nan=False, sort_keys=True)
            )
            normalized_runtime = json.loads(
                json.dumps(runtime_metadata or {}, allow_nan=False, sort_keys=True)
            )
        except (TypeError, ValueError) as error:
            raise ScientificArtifactError(
                "artifact payload and runtime metadata must be JSON-safe"
            ) from error
        scientific_payload = {
            "artifact_type": artifact_type,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "semantic_hashes": dict(semantic_hashes),
            "payload": normalized_payload,
        }
        envelope = cls(
            artifact_type=artifact_type,
            semantic_hashes=dict(semantic_hashes),
            payload=normalized_payload,
            scientific_hash=scientific_hash(
                scientific_payload, domain="artifact-envelope"
            ),
            runtime_metadata=normalized_runtime,
        )
        envelope.validate()
        return envelope

    def validate(
        self,
        expected_semantic_hashes: dict[str, str] | None = None,
    ) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ARTIFACT_SCHEMA_VERSION
        ):
            raise ScientificArtifactError("unsupported artifact schema version")
        if not isinstance(self.artifact_type, str) or not self.artifact_type:
            raise ScientificArtifactError("artifact_type must not be empty")
        if not isinstance(self.semantic_hashes, dict):
            raise ScientificArtifactError("semantic_hashes must be an object")
        if any(
            not isinstance(name, str) or not name or not is_sha256(value)
            for name, value in self.semantic_hashes.items()
        ):
            raise ScientificArtifactError("invalid semantic hash")
        scientific_payload = {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "semantic_hashes": self.semantic_hashes,
            "payload": self.payload,
        }
        expected_hash = scientific_hash(scientific_payload, domain="artifact-envelope")
        if self.scientific_hash != expected_hash:
            raise ScientificArtifactError("scientific artifact hash mismatch")
        if expected_semantic_hashes is not None:
            for name, expected in expected_semantic_hashes.items():
                if self.semantic_hashes.get(name) != expected:
                    raise ScientificArtifactError(
                        f"incompatible semantic hash for {name}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format": "infinite-rulebook-artifact",
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "semantic_hashes": self.semantic_hashes,
            "payload": self.payload,
            "scientific_hash": self.scientific_hash,
            "runtime_metadata": self.runtime_metadata,
        }

    @classmethod
    def from_dict(cls, raw: object) -> ArtifactEnvelope:
        if not isinstance(raw, dict):
            raise ScientificArtifactError("artifact must be a JSON object")
        expected = {
            "artifact_format",
            "artifact_type",
            "schema_version",
            "semantic_hashes",
            "payload",
            "scientific_hash",
            "runtime_metadata",
        }
        if set(raw) != expected:
            raise ScientificArtifactError("artifact fields are invalid")
        if raw["artifact_format"] != "infinite-rulebook-artifact":
            raise ScientificArtifactError("unrecognized artifact format")
        if not isinstance(raw["semantic_hashes"], dict) or not isinstance(
            raw["runtime_metadata"], dict
        ):
            raise ScientificArtifactError("artifact metadata must be objects")
        envelope = cls(
            artifact_type=raw["artifact_type"],
            schema_version=raw["schema_version"],
            semantic_hashes=raw["semantic_hashes"],
            payload=raw["payload"],
            scientific_hash=raw["scientific_hash"],
            runtime_metadata=raw["runtime_metadata"],
        )
        envelope.validate()
        return envelope


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory(
    path: Path,
    *,
    create: bool = False,
    missing_ok: bool = False,
) -> int | None:
    """Open a directory through no-follow directory descriptors."""

    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError as error:
                if not create:
                    if missing_ok:
                        os.close(descriptor)
                        return None
                    raise _ArtifactNotFound(
                        f"artifact path does not exist: {path}"
                    ) from error
                created = False
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(descriptor)
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except _ArtifactNotFound:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise ScientificArtifactError(
            f"artifact path contains an unsafe component: {path}"
        ) from error


@contextmanager
def artifact_tree_lock(path: Path) -> Iterator[None]:
    """Lock an artifact directory without following path or file symlinks."""

    parent = _open_directory(path, create=True)
    assert parent is not None
    descriptor: int | None = None
    try:
        descriptor = os.open(
            ".run.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode=0o600,
            dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ScientificArtifactError("artifact lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except ScientificArtifactError:
        raise
    except OSError as error:
        raise ScientificArtifactError(f"cannot lock artifact tree: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


@contextmanager
def artifact_root_lock(
    path: Path,
    *,
    shared: bool = False,
    nonblocking: bool = False,
    create: bool = True,
) -> Iterator[int]:
    """Coordinate workflow ownership through the artifact-root directory inode."""

    descriptor = _open_directory(path, create=create)
    assert descriptor is not None
    try:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if nonblocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise ArtifactRootBusyError(
                f"artifact root is already owned by another workflow: {path}"
            ) from error
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError as error:
        raise ScientificArtifactError(f"cannot lock artifact root: {path}") from error
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> object:
    parent = _open_directory(path.parent)
    assert parent is not None
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ScientificArtifactError(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = None
            return parse_json_strict(
                stream.read(),
                label=f"artifact {path}",
            )
    except FileNotFoundError as error:
        raise _ArtifactNotFound(f"artifact does not exist: {path}") from error
    except ScientificArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise ScientificArtifactError(f"cannot read artifact {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _list_json_files(path: Path, *, missing_ok: bool = False) -> tuple[Path, ...]:
    """List JSON files without following symlinks anywhere in the tree."""

    root = _open_directory(path, missing_ok=missing_ok)
    if root is None:
        return ()
    result: list[Path] = []

    def walk(descriptor: int, relative: Path) -> None:
        for name in sorted(os.listdir(descriptor)):
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ScientificArtifactError(
                    f"cannot inspect artifact tree member: {path / relative / name}"
                ) from error
            member = relative / name
            if stat.S_ISLNK(metadata.st_mode):
                raise ScientificArtifactError(
                    f"artifact tree contains a symbolic link: {path / member}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                before = len(result)
                try:
                    child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as error:
                    raise ScientificArtifactError(
                        f"cannot safely traverse artifact tree: {path / member}"
                    ) from error
                try:
                    walk(child, member)
                finally:
                    os.close(child)
                if len(result) == before:
                    raise ScientificArtifactError(
                        f"artifact tree contains an empty directory: {path / member}"
                    )
            elif name.endswith(".json"):
                if not stat.S_ISREG(metadata.st_mode):
                    raise ScientificArtifactError(
                        f"artifact is not a regular file: {path / member}"
                    )
                result.append(path / member)
            elif member == Path(".run.lock") and stat.S_ISREG(metadata.st_mode):
                continue
            else:
                raise ScientificArtifactError(
                    f"artifact tree contains an unexpected member: {path / member}"
                )

    try:
        walk(root, Path())
    finally:
        os.close(root)
    return tuple(result)


def cleanup_orphaned_artifact_temporaries(path: Path) -> None:
    """Remove only this store's interrupted private JSON publications.

    The caller must hold ``artifact_tree_lock(path)``. The private filename
    pattern is reserved to ``_exclusive_json`` and is never a user artifact.
    """

    root = _open_directory(path, missing_ok=True)
    if root is None:
        return

    def walk(descriptor: int) -> None:
        for name in sorted(os.listdir(descriptor)):
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ScientificArtifactError(
                    "cannot inspect interrupted artifact publication"
                ) from error
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                try:
                    walk(child)
                finally:
                    os.close(child)
                continue
            match = _ORPHAN_TEMP_NAME.fullmatch(name)
            if match is None:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ScientificArtifactError(
                    "interrupted artifact publication is not a regular file"
                )
            target = match.group("target")
            try:
                target_metadata = os.stat(
                    target,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                target_metadata = None
            except OSError as error:
                raise ScientificArtifactError(
                    "cannot inspect interrupted artifact target"
                ) from error
            if target_metadata is not None and not stat.S_ISREG(
                target_metadata.st_mode
            ):
                raise ScientificArtifactError(
                    "interrupted artifact target is not a regular file"
                )
            try:
                os.unlink(name, dir_fd=descriptor)
                os.fsync(descriptor)
            except OSError as error:
                raise ScientificArtifactError(
                    "cannot remove interrupted artifact publication"
                ) from error

    try:
        walk(root)
    finally:
        os.close(root)


def read_artifact(
    path: str | Path,
    *,
    expected_semantic_hashes: dict[str, str] | None = None,
) -> ArtifactEnvelope:
    envelope = ArtifactEnvelope.from_dict(_read_json(Path(path)))
    envelope.validate(expected_semantic_hashes)
    return envelope


def _exclusive_json(path: Path, content: dict[str, Any]) -> None:
    encoded = (
        json.dumps(content, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    parent = _open_directory(path.parent, create=True)
    assert parent is not None
    temporary_name = f".{path.name}.{secrets.token_hex(12)}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode=0o600,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            os.fsync(parent)
        except FileExistsError as error:
            raise ScientificArtifactError(
                f"immutable artifact already exists: {path}"
            ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent)
        os.close(parent)


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    path: Path

    @classmethod
    def for_run(
        cls,
        artifact_root: str | Path,
        experiment: str,
        run_hash: str,
    ) -> ArtifactStore:
        if not is_sha256(run_hash):
            raise ValueError("run_hash must be a SHA-256 digest")
        safe_name = experiment.replace("/", "_").replace("\\", "_")
        if safe_name in {"", ".", ".."}:
            raise ValueError("invalid experiment name")
        return cls(Path(artifact_root) / safe_name / run_hash)

    def _artifact_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ScientificArtifactError(
                "artifact paths must remain inside the artifact store"
            )
        return self.path / relative

    def write(
        self,
        relative_path: str | Path,
        artifact_type: str,
        semantic_hashes: dict[str, str],
        payload: Any,
        *,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        path = self._artifact_path(relative_path)
        envelope = ArtifactEnvelope.create(
            artifact_type,
            semantic_hashes,
            payload,
            runtime_metadata=runtime_metadata,
        )
        try:
            existing = read_artifact(path)
            if existing.scientific_hash != envelope.scientific_hash:
                raise ScientificArtifactError(
                    f"refusing to mutate immutable artifact: {path}"
                )
            return existing
        except _ArtifactNotFound:
            pass
        try:
            _exclusive_json(path, envelope.to_dict())
        except ScientificArtifactError as error:
            try:
                existing = read_artifact(path)
            except _ArtifactNotFound:
                raise
            if existing.scientific_hash != envelope.scientific_hash:
                raise ScientificArtifactError(
                    f"refusing to mutate immutable artifact: {path}"
                ) from error
            return existing
        return envelope

    def read(
        self,
        relative_path: str | Path,
        *,
        expected_semantic_hashes: dict[str, str] | None = None,
    ) -> ArtifactEnvelope:
        return read_artifact(
            self._artifact_path(relative_path),
            expected_semantic_hashes=expected_semantic_hashes,
        )

    def list_artifacts(self) -> tuple[Path, ...]:
        return _list_json_files(self.path, missing_ok=True)

    def finalize(
        self,
        semantic_hashes: dict[str, str],
        *,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        members = []
        resolved_configs: list[Any] = []
        for path in self.list_artifacts():
            if path == self.path / "manifest.json":
                continue
            artifact = read_artifact(path)
            if artifact.semantic_hashes != semantic_hashes:
                raise ScientificArtifactError(
                    "run artifact semantic hashes differ from finalization semantics"
                )
            members.append(
                {
                    "path": path.relative_to(self.path).as_posix(),
                    "artifact_type": artifact.artifact_type,
                    "scientific_hash": artifact.scientific_hash,
                }
            )
            if artifact.artifact_type == "resolved-run-config":
                resolved_configs.append(artifact.payload)
        required = {
            "resolved-run-config",
            "frontier-reference",
            "training-event",
            "run-checkpoint",
            "run-metrics",
        }
        present = {member["artifact_type"] for member in members}
        missing = required - present
        if missing:
            raise ScientificArtifactError(
                f"cannot finalize incomplete run; missing {sorted(missing)}"
            )
        _validate_run_member_inventory(
            self.path,
            members,
            resolved_configs=resolved_configs,
        )
        payload = {
            "members": members,
            "scientific_content_hash": scientific_hash(
                members, domain="run-scientific-content"
            ),
        }
        return self.write(
            "manifest.json",
            "run-manifest",
            semantic_hashes,
            payload,
            runtime_metadata=runtime_metadata,
        )


def _validate_run_member_inventory(
    root: Path,
    members: list[dict[str, str]],
    *,
    resolved_configs: list[Any],
) -> None:
    by_type: dict[str, list[str]] = {}
    for member in members:
        by_type.setdefault(member["artifact_type"], []).append(member["path"])
    singleton_paths = {
        "resolved-run-config": "config.resolved.json",
        "frontier-reference": "frontier-reference.json",
        "run-metrics": "metrics.json",
    }
    for artifact_type, expected_path in singleton_paths.items():
        if by_type.get(artifact_type) != [expected_path]:
            raise ScientificArtifactError(
                f"cannot finalize run with invalid {artifact_type} inventory"
            )
    events = by_type.get("training-event", [])
    checkpoints = by_type.get("run-checkpoint", [])
    if not events or any(
        Path(path).parent != Path("events")
        or _EVENT_NAME.fullmatch(Path(path).name) is None
        for path in events
    ):
        raise ScientificArtifactError(
            "cannot finalize run with invalid training-event inventory"
        )
    if not checkpoints or any(
        Path(path).parent != Path("checkpoints")
        or _CHECKPOINT_NAME.fullmatch(Path(path).name) is None
        for path in checkpoints
    ):
        raise ScientificArtifactError(
            "cannot finalize run with invalid run-checkpoint inventory"
        )
    allowed = {*singleton_paths, "training-event", "run-checkpoint"}
    unexpected = set(by_type) - allowed
    if unexpected:
        raise ScientificArtifactError(
            f"cannot finalize run with unexpected artifact types: {sorted(unexpected)}"
        )
    if len({member["path"] for member in members}) != len(members):
        raise ScientificArtifactError(
            f"cannot finalize run with duplicate member paths under {root}"
        )
    if len(resolved_configs) != 1 or not isinstance(resolved_configs[0], dict):
        raise ScientificArtifactError(
            "cannot finalize run with an invalid resolved config"
        )
    run_settings = resolved_configs[0].get("run_settings")
    if not isinstance(run_settings, dict):
        raise ScientificArtifactError(
            "cannot finalize run with invalid resolved run settings"
        )
    horizon = run_settings.get("horizon")
    checkpoint_config = run_settings.get("checkpoints")
    rounds = (
        None
        if not isinstance(checkpoint_config, dict)
        else checkpoint_config.get("rounds")
    )
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or not isinstance(rounds, list)
        or any(
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or not 0 <= round_index <= horizon
            for round_index in rounds
        )
        or rounds != sorted(set(rounds))
    ):
        raise ScientificArtifactError(
            "cannot finalize run with invalid horizon or checkpoints"
        )
    expected_events = [
        f"events/{round_index:08d}.json" for round_index in range(horizon)
    ]
    expected_checkpoints = [
        f"checkpoints/{round_index:08d}.json" for round_index in rounds
    ]
    if events != expected_events or checkpoints != expected_checkpoints:
        raise ScientificArtifactError(
            "cannot finalize run with an incomplete or extra event/checkpoint inventory"
        )


@dataclass(frozen=True, slots=True)
class JournalEvent:
    sequence: int
    event_key: str
    event_kind: str
    previous_hash: str | None
    payload: Any
    event_hash: str


def _validated_journal_events(
    records: Iterator[tuple[int, ArtifactEnvelope]],
) -> list[JournalEvent]:
    result: list[JournalEvent] = []
    event_keys: set[str] = set()
    previous_hash = None
    required = {
        "sequence",
        "event_key",
        "event_kind",
        "previous_hash",
        "payload",
        "event_hash",
    }
    for expected_sequence, (sequence, envelope) in enumerate(records):
        if sequence != expected_sequence:
            raise ScientificArtifactError("event journal has a sequence gap")
        if envelope.artifact_type != "training-event":
            raise ScientificArtifactError("event journal contains a wrong type")
        payload = envelope.payload
        if not isinstance(payload, dict):
            raise ScientificArtifactError("event payload must be an object")
        if set(payload) != required:
            raise ScientificArtifactError("training event fields are invalid")
        if (
            isinstance(payload["sequence"], bool)
            or not isinstance(payload["sequence"], int)
            or not isinstance(payload["event_key"], str)
            or not payload["event_key"]
            or not isinstance(payload["event_kind"], str)
            or not payload["event_kind"]
            or (
                payload["previous_hash"] is not None
                and not is_sha256(payload["previous_hash"])
            )
            or not is_sha256(payload["event_hash"])
        ):
            raise ScientificArtifactError("training event values are invalid")
        scientific_event = {name: payload[name] for name in required - {"event_hash"}}
        event_hash = scientific_hash(scientific_event, domain="training-event")
        if payload["sequence"] != sequence or payload["previous_hash"] != previous_hash:
            raise ScientificArtifactError("event journal hash chain is invalid")
        if payload["event_hash"] != event_hash:
            raise ScientificArtifactError("training event hash mismatch")
        event = JournalEvent(**payload)
        if event.event_key in event_keys:
            raise ScientificArtifactError("event journal has duplicate event keys")
        event_keys.add(event.event_key)
        result.append(event)
        previous_hash = event.event_hash
    return result


class EventJournal:
    """One immutable file per event, forming a validated hash chain."""

    def __init__(
        self,
        store: ArtifactStore,
        semantic_hashes: dict[str, str],
    ) -> None:
        self.store = store
        self.semantic_hashes = semantic_hashes
        self._events: list[JournalEvent] | None = None
        self._events_by_key: dict[str, JournalEvent] = {}

    def events(self) -> tuple[JournalEvent, ...]:
        return tuple(self._load())

    def _load(self) -> list[JournalEvent]:
        if self._events is not None:
            return self._events
        directory = self.store.path / "events"
        paths = []
        for path in _list_json_files(directory, missing_ok=True):
            if path.parent != directory:
                continue
            match = _EVENT_NAME.match(path.name)
            if match:
                paths.append((int(match.group(1)), path))
        paths.sort()
        result = _validated_journal_events(
            (
                (
                    sequence,
                    read_artifact(
                        path,
                        expected_semantic_hashes=self.semantic_hashes,
                    ),
                )
                for sequence, path in paths
            )
        )
        self._events = result
        self._events_by_key = {event.event_key: event for event in result}
        return result

    def append(self, event_key: str, event_kind: str, payload: Any) -> JournalEvent:
        if not event_key or not event_kind:
            raise ValueError("event key and kind must not be empty")
        events = self._load()
        if event_key in self._events_by_key:
            existing = self._events_by_key[event_key]
            if existing.event_kind != event_kind or existing.payload != payload:
                raise ScientificArtifactError(
                    "duplicate event key has different content"
                )
            return existing
        sequence = len(events)
        previous_hash = events[-1].event_hash if events else None
        scientific_event = {
            "sequence": sequence,
            "event_key": event_key,
            "event_kind": event_kind,
            "previous_hash": previous_hash,
            "payload": payload,
        }
        event_hash = scientific_hash(scientific_event, domain="training-event")
        event_payload = {**scientific_event, "event_hash": event_hash}
        self.store.write(
            f"events/{sequence:08d}.json",
            "training-event",
            self.semantic_hashes,
            event_payload,
        )
        event = JournalEvent(**event_payload)
        events.append(event)
        self._events_by_key[event_key] = event
        return event


def write_frontier_bundle(
    store: ArtifactStore,
    semantic_hashes: dict[str, str],
    *,
    curve: Any,
    witnesses: dict[str, Any],
    certificates: dict[str, Any],
    diagnostics: Any,
) -> ArtifactEnvelope:
    """Persist every scientific component of a frontier before its manifest."""

    members: list[dict[str, str]] = []

    def write_member(
        relative_path: str,
        artifact_type: str,
        payload: Any,
    ) -> None:
        artifact = store.write(
            relative_path,
            artifact_type,
            semantic_hashes,
            payload,
        )
        members.append(
            {
                "path": relative_path,
                "artifact_type": artifact_type,
                "scientific_hash": artifact.scientific_hash,
            }
        )

    write_member("frontier/curve.json", "frontier-curve", curve)
    for name, payload in sorted(witnesses.items()):
        write_member(
            f"frontier/witnesses/{name}.json",
            "frontier-witness",
            payload,
        )
    for name, payload in sorted(certificates.items()):
        write_member(
            f"frontier/certificates/{name}.json",
            "frontier-certificate",
            payload,
        )
    write_member(
        "frontier/diagnostics.json",
        "frontier-diagnostics",
        diagnostics,
    )
    members.sort(key=lambda member: member["path"])
    observed = []
    for path in store.list_artifacts():
        if path == store.path / "frontier/manifest.json":
            continue
        artifact = read_artifact(path)
        observed.append(
            {
                "path": path.relative_to(store.path).as_posix(),
                "artifact_type": artifact.artifact_type,
                "scientific_hash": artifact.scientific_hash,
            }
        )
    if observed != members:
        raise ScientificArtifactError(
            "cannot finalize frontier with an unexpected artifact inventory"
        )
    return store.write(
        "frontier/manifest.json",
        "frontier-manifest",
        semantic_hashes,
        {
            "members": members,
            "scientific_content_hash": scientific_hash(
                members,
                domain="frontier-scientific-content",
            ),
        },
    )


def validate_artifact_tree(
    root: str | Path,
    *,
    expected_semantic_hashes: dict[str, str] | None = None,
    session: ArtifactValidationSession | None = None,
) -> tuple[ArtifactEnvelope, ...]:
    """Validate a tree, single-flighting shared frontier work per session."""

    if session is not None and not isinstance(session, ArtifactValidationSession):
        raise TypeError("session must be an ArtifactValidationSession or None")
    path = Path(root)
    if (
        session is None
        or expected_semantic_hashes is None
        or set(expected_semantic_hashes) != {"frontier"}
        or not is_sha256(expected_semantic_hashes["frontier"])
    ):
        return _validate_artifact_tree_uncached(
            path,
            expected_semantic_hashes=expected_semantic_hashes,
            session=session,
        )
    frontier_cache_key = (
        os.path.abspath(path),
        expected_semantic_hashes["frontier"],
    )
    while True:
        with session._lock:
            cached = session._validated_frontier_trees.get(frontier_cache_key)
            if cached is not None:
                return _copy_artifact_envelopes(cached)
            pending = session._frontier_inflight.get(frontier_cache_key)
            if pending is None:
                pending = threading.Event()
                session._frontier_inflight[frontier_cache_key] = pending
                break
        pending.wait()
    try:
        artifacts = _validate_artifact_tree_uncached(
            path,
            expected_semantic_hashes=expected_semantic_hashes,
            session=session,
        )
    except BaseException:
        with session._lock:
            session._frontier_inflight.pop(frontier_cache_key, None)
            pending.set()
        raise
    with session._lock:
        session._validated_frontier_trees[frontier_cache_key] = (
            _copy_artifact_envelopes(artifacts)
        )
        session._frontier_inflight.pop(frontier_cache_key, None)
        pending.set()
    return artifacts


def _copy_artifact_envelopes(
    artifacts: tuple[ArtifactEnvelope, ...],
) -> tuple[ArtifactEnvelope, ...]:
    """Keep mutable public envelope graphs outside the trusted cache."""

    return tuple(
        ArtifactEnvelope.from_dict(copy.deepcopy(artifact.to_dict()))
        for artifact in artifacts
    )


def _validate_artifact_tree_uncached(
    path: Path,
    *,
    expected_semantic_hashes: dict[str, str] | None,
    session: ArtifactValidationSession | None,
) -> tuple[ArtifactEnvelope, ...]:
    try:
        files = _list_json_files(path)
    except _ArtifactNotFound as error:
        raise ScientificArtifactError(
            f"artifact tree does not exist: {path}"
        ) from error
    records = tuple(
        (
            file,
            read_artifact(file, expected_semantic_hashes=expected_semantic_hashes),
        )
        for file in files
    )
    artifacts = tuple(artifact for _, artifact in records)
    if not artifacts:
        raise ScientificArtifactError("artifact tree is empty")
    manifests = [
        artifact for artifact in artifacts if artifact.artifact_type == "run-manifest"
    ]
    frontier_manifests = [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "frontier-manifest"
    ]
    if len(manifests) + len(frontier_manifests) != 1:
        raise ScientificArtifactError(
            "artifact tree must contain exactly one recognized manifest"
        )
    manifest = (manifests or frontier_manifests)[0]
    if any(
        artifact.semantic_hashes != manifest.semantic_hashes for artifact in artifacts
    ):
        raise ScientificArtifactError(
            "artifact semantic hashes differ from the manifest"
        )
    if manifests:
        manifest = manifests[0]
        manifest_records = [
            file
            for file, artifact in records
            if artifact.artifact_type == "run-manifest"
        ]
        if manifest_records != [path / "manifest.json"]:
            raise ScientificArtifactError("run manifest must be at the tree root")
        try:
            expected_members = manifest.payload["members"]
        except (KeyError, TypeError) as error:
            raise ScientificArtifactError(
                "run manifest scientific structure is malformed"
            ) from error
        actual_members = [
            {
                "path": file.relative_to(path).as_posix(),
                "artifact_type": artifact.artifact_type,
                "scientific_hash": artifact.scientific_hash,
            }
            for file, artifact in records
            if file != path / "manifest.json"
        ]
        if actual_members != expected_members:
            raise ScientificArtifactError("run manifest member list is invalid")
        expected_content_hash = scientific_hash(
            actual_members, domain="run-scientific-content"
        )
        if manifest.payload.get("scientific_content_hash") != expected_content_hash:
            raise ScientificArtifactError(
                "run manifest scientific content hash is invalid"
            )
        _validate_completed_run(path, records, manifest)
    try:
        _validate_frontier_records(path, records)
        if manifests:
            _validate_frontier_reference(
                path,
                records,
                session=session,
            )
    except ScientificArtifactError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ScientificArtifactError(
            "artifact tree scientific structure is malformed"
        ) from error
    return artifacts


def _validate_completed_run(
    root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
    manifest: ArtifactEnvelope,
) -> None:
    try:
        _validate_completed_run_payload(root, records, manifest)
    except ScientificArtifactError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ScientificArtifactError(
            "completed run scientific structure is malformed"
        ) from error


def _validate_completed_run_payload(
    root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
    manifest: ArtifactEnvelope,
) -> None:
    by_type: dict[str, list[tuple[Path, ArtifactEnvelope]]] = {}
    for file, artifact in records:
        by_type.setdefault(artifact.artifact_type, []).append((file, artifact))
    singleton_types = (
        "resolved-run-config",
        "frontier-reference",
        "run-metrics",
        "run-manifest",
    )
    for artifact_type in singleton_types:
        if len(by_type.get(artifact_type, [])) != 1:
            raise ScientificArtifactError(
                f"completed run must contain one {artifact_type}"
            )
    if not by_type.get("training-event") or not by_type.get("run-checkpoint"):
        raise ScientificArtifactError(
            "completed run must contain events and checkpoints"
        )
    allowed_types = {
        *singleton_types,
        "training-event",
        "run-checkpoint",
    }
    unexpected_types = set(by_type) - allowed_types
    if unexpected_types:
        raise ScientificArtifactError(
            f"completed run contains unexpected artifact types: "
            f"{sorted(unexpected_types)}"
        )
    expected_singletons = {
        "resolved-run-config": root / "config.resolved.json",
        "frontier-reference": root / "frontier-reference.json",
        "run-metrics": root / "metrics.json",
        "run-manifest": root / "manifest.json",
    }
    for artifact_type, expected_path in expected_singletons.items():
        if by_type[artifact_type][0][0] != expected_path:
            raise ScientificArtifactError(
                f"completed run {artifact_type} is stored at an invalid path"
            )
    if any(
        file.parent != root / "events" or _EVENT_NAME.fullmatch(file.name) is None
        for file, _ in by_type["training-event"]
    ):
        raise ScientificArtifactError(
            "completed run training events must be stored under events/"
        )
    if any(
        file.parent != root / "checkpoints" for file, _ in by_type["run-checkpoint"]
    ):
        raise ScientificArtifactError(
            "completed run checkpoints must be stored under checkpoints/"
        )

    config = by_type["resolved-run-config"][0][1].payload
    base_config_fields = {
        "cell",
        "provenance",
        "run_hash",
        "run_settings",
        "seeds",
    }
    if set(config) != base_config_fields:
        raise ScientificArtifactError(
            "completed run resolved config fields are invalid"
        )
    metrics = by_type["run-metrics"][0][1].payload
    run_settings = config["run_settings"]
    cell = config["cell"]
    seeds = config["seeds"]
    provenance = config["provenance"]
    recorded_run_hash = config.get("run_hash")
    try:
        from infinite_rulebook.orchestration.config import (
            SYMBOLIC_ADAPTER_CONTRACT_V1,
            SYMBOLIC_ADAPTER_CONTRACT_V2,
            registered_symbolic_v2_phase,
            run_cell_from_dict,
            run_cell_identity_payload,
            symbolic_adapter_contract,
        )
        from infinite_rulebook.orchestration.provenance import ScientificProvenance
        from infinite_rulebook.orchestration.seeds import RunSeeds, SeedBank
        from infinite_rulebook.orchestration.semantics import semantic_hashes

        typed_cell = run_cell_from_dict(cell)
        typed_cell_payload = run_cell_identity_payload(cell)
        typed_seeds = RunSeeds(**seeds)
        typed_provenance = ScientificProvenance(**provenance)
    except (TypeError, ValueError) as error:
        raise ScientificArtifactError(
            "completed run identity inputs are malformed"
        ) from error
    seed_bank = SeedBank(
        run_settings["master_seed"],
        run_settings.get("algorithm_master_seed"),
    )
    expected_seeds = (
        seed_bank.for_cell(typed_cell)
        if "algorithm_master_seed" in run_settings
        else seed_bank.legacy_for_cell(typed_cell)
    )
    if typed_seeds != expected_seeds:
        raise ScientificArtifactError(
            "completed run seeds do not match the registered seed-bank policy"
        )
    expected_semantics = semantic_hashes(
        typed_cell,
        analysis_code_hash=typed_provenance.analysis_code_hash,
        solver_identity_payload=cell["solver"],
    )
    if manifest.semantic_hashes != expected_semantics:
        raise ScientificArtifactError(
            "completed run semantics do not match its typed cell and provenance"
        )
    expected_run_hash = scientific_hash(
        {
            "runner_version": "symbolic-runner.v1",
            "run_settings": run_settings,
            "cell": typed_cell_payload,
            "seeds": asdict(typed_seeds),
            "provenance": typed_provenance.to_dict(),
        },
        domain="run-identity",
    )
    if recorded_run_hash != expected_run_hash or root.name != expected_run_hash:
        raise ScientificArtifactError(
            "completed run identity does not match its scientific inputs"
        )
    horizon = run_settings["horizon"]
    phase = run_settings.get("phase")
    adapter_contract = run_settings.get("adapter_contract")
    experiment_name = run_settings.get("experiment_name")
    if adapter_contract == SYMBOLIC_ADAPTER_CONTRACT_V1:
        if experiment_name is not None:
            raise ScientificArtifactError(
                "legacy adapter runs cannot claim a v2 experiment name"
            )
    elif adapter_contract == SYMBOLIC_ADAPTER_CONTRACT_V2:
        if (
            not isinstance(experiment_name, str)
            or registered_symbolic_v2_phase(experiment_name) != phase
            or symbolic_adapter_contract(experiment_name) != adapter_contract
        ):
            raise ScientificArtifactError(
                "v2 adapter contract is not bound to an exact registered experiment"
            )
    else:
        raise ScientificArtifactError(
            "completed run has an unregistered adapter contract"
        )
    confirmatory = phase == "confirmatory"
    freeze_hash = run_settings.get("confirmatory_freeze_hash")
    registration_hash = run_settings.get("analysis_registration_hash")
    if (
        phase not in {"pilot", "calibration", "confirmatory"}
        or metrics.get("phase") != phase
        or metrics.get("confirmatory_frozen") is not confirmatory
        or run_settings.get("confirmatory_frozen", False) is not confirmatory
        or (confirmatory and not is_sha256(freeze_hash))
        or (confirmatory and not is_sha256(registration_hash))
        or (
            confirmatory
            and adapter_contract
            not in {SYMBOLIC_ADAPTER_CONTRACT_V1, SYMBOLIC_ADAPTER_CONTRACT_V2}
        )
        or (not confirmatory and freeze_hash is not None)
        or (not confirmatory and registration_hash is not None)
    ):
        raise ScientificArtifactError(
            "completed run phase and confirmatory freeze metadata are invalid"
        )
    events = _validated_journal_events(
        (
            (int(file.stem), envelope)
            for file, envelope in sorted(
                by_type["training-event"],
                key=lambda item: item[0].name,
            )
        )
    )
    if (
        len(events) != horizon
        or metrics.get("event_count") != horizon
        or metrics.get("completed_rounds") != horizon
    ):
        raise ScientificArtifactError(
            "completed run event count does not match its horizon"
        )
    post_query_hidden_rewards: tuple[float, ...] = ()
    if adapter_contract == SYMBOLIC_ADAPTER_CONTRACT_V2:
        parsed_rewards = []
        for event in events:
            payload = event.payload
            value = (
                payload.get("post_query_hidden_expected_reward")
                if isinstance(payload, dict)
                else None
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ScientificArtifactError(
                    "v2 training event has an invalid post-query hidden reward"
                )
            parsed_rewards.append(float(value))
        post_query_hidden_rewards = tuple(parsed_rewards)
    elif any(
        isinstance(event.payload, dict)
        and "post_query_hidden_expected_reward" in event.payload
        for event in events
    ):
        raise ScientificArtifactError(
            "legacy training event contains a v2 post-query hidden reward"
        )
    expected_rounds = list(run_settings["checkpoints"]["rounds"])
    checkpoint_rounds = []
    for file, checkpoint in by_type["run-checkpoint"]:
        from infinite_rulebook.environments.controls import PublicDeploymentAction
        from infinite_rulebook.orchestration.records import (
            validate_checkpoint_record,
        )

        payload = checkpoint.payload
        round_index = payload.get("round")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or file.name != f"{round_index:08d}.json"
            or payload.get("training_state_before")
            != payload.get("training_state_after")
            or payload.get("evaluation_seed") != seeds["evaluation"]
            or payload.get("deployment_seed") != seeds["deployment"]
        ):
            raise ScientificArtifactError(
                "completed run checkpoint metadata is invalid"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ScientificArtifactError("completed run checkpoint result is invalid")
        post_query_source_name = "post_query_hidden_expected_reward"
        post_query_mean_name = "post_query_mean_hidden_expected_reward"
        if adapter_contract == SYMBOLIC_ADAPTER_CONTRACT_V2:
            if round_index == 0:
                valid_post_query_metrics = (
                    post_query_source_name not in result
                    and post_query_mean_name not in result
                )
            else:
                post_query_source = result.get(post_query_source_name)
                post_query_mean = result.get(post_query_mean_name)
                valid_post_query_metrics = (
                    not isinstance(post_query_source, bool)
                    and isinstance(post_query_source, (int, float))
                    and math.isfinite(post_query_source)
                    and post_query_source == post_query_hidden_rewards[round_index - 1]
                    and not isinstance(post_query_mean, bool)
                    and isinstance(post_query_mean, (int, float))
                    and math.isfinite(post_query_mean)
                    and post_query_mean
                    == math.fsum(post_query_hidden_rewards[:round_index]) / round_index
                )
            if not valid_post_query_metrics:
                raise ScientificArtifactError(
                    "v2 checkpoint post-query hidden-reward metrics are invalid"
                )
        elif post_query_source_name in result or post_query_mean_name in result:
            raise ScientificArtifactError(
                "legacy checkpoint contains a v2 post-query hidden-reward metric"
            )
        try:
            typed = validate_checkpoint_record(result["scientific_records"])
        except (KeyError, TypeError, ValueError) as error:
            raise ScientificArtifactError(
                "completed run typed checkpoint record is invalid"
            ) from error
        from infinite_rulebook.orchestration.symbolic import (
            recompute_reward_components,
        )

        recomputed_reward = recompute_reward_components(
            typed_cell,
            typed_seeds,
            typed.deployment_witness,
        )
        typed_semantics = {
            name: getattr(typed.semantic_hashes, name)
            for name in ("environment", "reward", "action", "feedback", "frontier")
        }
        deployment = typed.deployment_witness
        readable_deployment: object
        if isinstance(deployment, PublicDeploymentAction):
            readable_deployment = {
                "entries": [list(entry) for entry in deployment.deployment.entries],
                "public_choice": deployment.public_choice,
            }
        else:
            readable_deployment = [list(entry) for entry in deployment]
        if (
            typed.round_index != round_index
            or typed_semantics != checkpoint.semantic_hashes
            or typed.deployment_seed != payload["deployment_seed"]
            or result.get("expected_reward") != typed.reward_samples[0]
            or (
                result.get("expected_reward"),
                result.get("hidden_expected_reward"),
                result.get("public_reward"),
            )
            != recomputed_reward
            or result.get("deployment") != readable_deployment
            or result.get("information") != asdict(typed.realized_information)
            or result.get("novelty") != asdict(typed.novelty)
            or result.get("support") != asdict(typed.support)
            or result.get("compute") != asdict(typed.compute)
        ):
            raise ScientificArtifactError(
                "completed run checkpoint result differs from its typed record"
            )
        capabilities = result.get("agent_capabilities")
        try:
            capability_manifest = CapabilityManifest(**capabilities)
        except (TypeError, ValueError):
            raise ScientificArtifactError(
                "completed run checkpoint agent capabilities are invalid"
            ) from None
        if capabilities != asdict(capability_manifest):
            raise ScientificArtifactError(
                "completed run checkpoint agent capabilities are not canonical"
            )
        checkpoint_rounds.append(round_index)
    if sorted(checkpoint_rounds) != expected_rounds:
        raise ScientificArtifactError("completed run checkpoint schedule is incomplete")
    if phase in {"calibration", "confirmatory"}:
        _validate_exact_symbolic_replay(
            typed_cell=typed_cell,
            typed_seeds=typed_seeds,
            semantic_hashes=manifest.semantic_hashes,
            events=events,
            checkpoints=tuple(artifact for _, artifact in by_type["run-checkpoint"]),
            metrics=metrics,
            horizon=horizon,
            expected_rounds=tuple(expected_rounds),
            phase=phase,
            confirmatory=confirmatory,
            adapter_contract=adapter_contract,
        )


def _validate_exact_symbolic_replay(
    *,
    typed_cell: Any,
    typed_seeds: Any,
    semantic_hashes: dict[str, str],
    events: tuple[Any, ...],
    checkpoints: tuple[ArtifactEnvelope, ...],
    metrics: dict[str, Any],
    horizon: int,
    expected_rounds: tuple[int, ...],
    phase: str,
    confirmatory: bool,
    adapter_contract: str,
) -> None:
    """Replay every registered study transition and evaluation from first principles."""

    from infinite_rulebook.orchestration.symbolic import exact_symbolic_adapter_class

    adapter = exact_symbolic_adapter_class(adapter_contract)()
    state = adapter.initial_state(typed_cell, typed_seeds)
    checkpoint_by_round = {
        checkpoint.payload["round"]: checkpoint for checkpoint in checkpoints
    }
    for round_index in range(horizon + 1):
        if round_index in expected_rounds:
            current = adapter.state_fingerprint(state)
            evaluation_adapter, evaluation_state = copy.deepcopy((adapter, state))
            if evaluation_adapter.state_fingerprint(evaluation_state) != current:
                raise ScientificArtifactError(
                    "registered checkpoint clone differs during exact replay"
                )
            result = evaluation_adapter.checkpoint(
                evaluation_state,
                round_index,
                typed_cell,
                typed_seeds,
                semantic_hashes,
            )
            if evaluation_adapter.state_fingerprint(evaluation_state) != current:
                raise ScientificArtifactError(
                    "registered checkpoint mutates state during exact replay"
                )
            expected_checkpoint = {
                "round": round_index,
                "training_state_before": current,
                "training_state_after": current,
                "evaluation_seed": typed_seeds.evaluation,
                "deployment_seed": typed_seeds.deployment,
                "result": result,
            }
            if checkpoint_by_round[round_index].payload != expected_checkpoint:
                raise ScientificArtifactError(
                    "completed run checkpoint differs from exact adapter replay"
                )
        if round_index == horizon:
            break
        event = events[round_index]
        expected_event = adapter.training_event(
            state,
            round_index,
            typed_cell,
            typed_seeds,
        )
        if (
            event.sequence != round_index
            or event.event_key != f"round:{round_index}"
            or event.event_kind != "training-step"
            or event.payload != expected_event
        ):
            raise ScientificArtifactError(
                "completed run event differs from exact adapter replay"
            )
        state = adapter.apply_training_event(state, event.payload)
    expected_metrics = {
        "completed_rounds": horizon,
        "event_count": horizon,
        "final_state_hash": adapter.state_fingerprint(state),
        "phase": phase,
        "confirmatory_frozen": confirmatory,
    }
    if metrics != expected_metrics:
        raise ScientificArtifactError(
            "completed run metrics differ from exact adapter replay"
        )


def _validate_frontier_reference(
    run_root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
    *,
    session: ArtifactValidationSession | None,
) -> None:
    references = [
        artifact
        for _, artifact in records
        if artifact.artifact_type == "frontier-reference"
    ]
    if len(references) != 1:
        raise ScientificArtifactError(
            "run artifact must contain one frontier reference"
        )
    reference = references[0]
    frontier_hash = reference.payload["frontier_hash"]
    if reference.semantic_hashes.get("frontier") != frontier_hash:
        raise ScientificArtifactError("frontier reference semantic hash mismatch")
    try:
        artifact_root = run_root.parents[1]
    except IndexError as error:
        raise ScientificArtifactError("run artifact path is invalid") from error
    frontier_root = artifact_root / "_frontiers" / frontier_hash
    cache_key = (os.path.abspath(frontier_root), frontier_hash)
    validated_frontiers = None if session is None else session._validated_frontiers
    if validated_frontiers is not None:
        assert session is not None
        with session._lock:
            cached_hash = validated_frontiers.get(cache_key)
        if cached_hash is not None:
            if cached_hash != reference.payload["artifact_hash"]:
                raise ScientificArtifactError(
                    "cached frontier manifest hash is incompatible"
                )
            return
    frontier_artifacts = validate_artifact_tree(
        frontier_root,
        expected_semantic_hashes={"frontier": frontier_hash},
        session=session,
    )
    manifest = next(
        artifact
        for artifact in frontier_artifacts
        if artifact.artifact_type == "frontier-manifest"
    )
    if manifest.scientific_hash != reference.payload["artifact_hash"]:
        raise ScientificArtifactError(
            "referenced frontier manifest hash is incompatible"
        )
    if validated_frontiers is not None:
        assert session is not None
        with session._lock:
            validated_frontiers[cache_key] = manifest.scientific_hash


def _validate_frontier_records(
    root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
) -> None:
    manifests = [
        artifact
        for _, artifact in records
        if artifact.artifact_type == "frontier-manifest"
    ]
    if not manifests:
        return
    if len(manifests) != 1:
        raise ScientificArtifactError("frontier tree must have one manifest")
    manifest_records = [
        file
        for file, artifact in records
        if artifact.artifact_type == "frontier-manifest"
    ]
    if manifest_records != [root / "frontier/manifest.json"]:
        raise ScientificArtifactError(
            "frontier manifest must be at frontier/manifest.json"
        )
    members = [
        {
            "path": file.relative_to(root).as_posix(),
            "artifact_type": artifact.artifact_type,
            "scientific_hash": artifact.scientific_hash,
        }
        for file, artifact in records
        if artifact.artifact_type != "frontier-manifest"
    ]
    manifest_payload = manifests[0].payload
    if manifest_payload.get("members") != members:
        raise ScientificArtifactError("frontier manifest member list is invalid")
    if manifest_payload.get("scientific_content_hash") != scientific_hash(
        members,
        domain="frontier-scientific-content",
    ):
        raise ScientificArtifactError(
            "frontier manifest scientific content hash is invalid"
        )

    curves = [
        artifact
        for _, artifact in records
        if artifact.artifact_type == "frontier-curve"
    ]
    if len(curves) != 1:
        raise ScientificArtifactError("frontier tree must have one curve")
    problem_payload = curves[0].payload["problem"]
    recorded_provenance = problem_payload["provenance_hash"]
    unhashed_problem = {
        key: value for key, value in problem_payload.items() if key != "provenance_hash"
    }
    if recorded_provenance != scientific_hash(
        unhashed_problem, domain="frontier-problem"
    ):
        raise ScientificArtifactError("frontier problem provenance is invalid")
    problem = FiniteDecisionProblem(
        prior=tuple(problem_payload["prior"]),
        rewards=tuple(tuple(row) for row in problem_payload["reward_matrix"]),
    )
    problem_semantic_hash = typed_semantic_hash(problem)
    if curves[0].payload.get("problem_semantic_hash") != problem_semantic_hash:
        raise ScientificArtifactError("frontier problem semantic hash is invalid")
    points = curves[0].payload["points"]
    expected_names = {f"point-{index:03d}" for index in range(len(points))}
    witnesses = {
        file.stem: artifact
        for file, artifact in records
        if artifact.artifact_type == "frontier-witness"
    }
    certificates = {
        file.stem: artifact
        for file, artifact in records
        if artifact.artifact_type == "frontier-certificate"
    }
    raw_curve = curves[0].payload.get("raw_curve")
    if not isinstance(raw_curve, list) or len(raw_curve) != len(points):
        raise ScientificArtifactError(
            "frontier raw curve must be retained for every point"
        )
    diagnostics = [
        artifact.payload
        for _, artifact in records
        if artifact.artifact_type == "frontier-diagnostics"
    ]
    if len(diagnostics) != 1:
        raise ScientificArtifactError(
            "frontier tree must have one diagnostics artifact"
        )
    diagnostic_by_name = {
        diagnostic["name"]: diagnostic
        for diagnostic in diagnostics[0].get("points", [])
    }
    diagnostic_names = set(diagnostic_by_name)
    if (
        set(witnesses) != expected_names
        or set(certificates) != expected_names
        or diagnostic_names != expected_names
    ):
        raise ScientificArtifactError(
            "frontier points, witnesses, certificates, and diagnostics differ"
        )
    solver_settings = diagnostics[0]["solver_settings"]
    for point_index, point in enumerate(points):
        name = f"point-{point_index:03d}"
        witness_artifact = witnesses[name]
        certificate = certificates[name].payload
        witness_payload = witness_artifact.payload
        persisted_witness = ChannelWitness(
            channel=tuple(tuple(row) for row in witness_payload["channel"]),
            action_marginal=tuple(witness_payload["action_marginal"]),
            expected_reward=witness_payload["expected_reward"],
            mutual_information=witness_payload["mutual_information"],
        )
        evaluated = problem.evaluate(persisted_witness.channel)
        if not math.isclose(
            evaluated.expected_reward,
            witness_payload["expected_reward"],
            abs_tol=1e-10,
        ) or not math.isclose(
            evaluated.mutual_information,
            witness_payload["mutual_information"],
            abs_tol=1e-10,
        ):
            raise ScientificArtifactError("frontier witness quantities are invalid")
        recorded_marginal = witness_payload["action_marginal"]
        if not isinstance(recorded_marginal, list) or len(recorded_marginal) != len(
            evaluated.action_marginal
        ):
            raise ScientificArtifactError("frontier witness action marginal is invalid")
        if any(
            not math.isclose(left, right, abs_tol=1e-10)
            for left, right in zip(
                evaluated.action_marginal,
                recorded_marginal,
                strict=True,
            )
        ):
            raise ScientificArtifactError("frontier witness action marginal is invalid")
        if evaluated.expected_reward < point["target_reward"] - 1e-9:
            raise ScientificArtifactError("frontier witness is reward-infeasible")
        if (
            certificate["target_reward"] != point["target_reward"]
            or certificate["lower_bound"] != point["lower_information"]
            or certificate["upper_bound"] != point["upper_information"]
        ):
            raise ScientificArtifactError(
                "frontier curve and certificate are inconsistent"
            )
        if certificate["lower_bound"] > evaluated.mutual_information + 1e-8:
            raise ScientificArtifactError("frontier certificate exceeds witness")
        if not math.isclose(
            certificate["upper_bound"],
            evaluated.mutual_information,
            abs_tol=1e-8,
        ):
            raise ScientificArtifactError("frontier certificate upper bound is invalid")
        beta = certificate["dual_beta"]
        parsed_beta = math.inf if beta == "infinity" else beta
        try:
            solution = FrontierSolution(
                target_reward=certificate["requested_target_reward"],
                effective_target_reward=certificate["effective_target_reward"],
                witness=persisted_witness,
                lower_bound=certificate["lower_bound"],
                upper_bound=certificate["upper_bound"],
                duality_gap=(
                    math.inf
                    if certificate["duality_gap"] == "infinity"
                    else certificate["duality_gap"]
                ),
                dual_beta=parsed_beta,
                iterations=diagnostic_by_name[name]["iterations"],
                converged=diagnostic_by_name[name]["converged"],
                problem_semantic_hash=problem_semantic_hash,
                lower_certificate_marginal=(
                    None
                    if certificate["dual_action_marginal"] is None
                    else tuple(certificate["dual_action_marginal"])
                ),
                lower_certificate_supports=(
                    None
                    if certificate["supported_actions"] is None
                    else tuple(
                        tuple(support) for support in certificate["supported_actions"]
                    )
                ),
            )
            typed_point = FrontierPoint.from_frontier_solution(problem, solution)
        except (TypeError, ValueError) as error:
            raise ScientificArtifactError(
                "frontier lower certificate evidence is invalid"
            ) from error
        if (
            typed_point.upper_witness.witness_hash != witness_payload["witness_hash"]
            or typed_point.lower_certificate.source_solution_hash
            != certificate["source_solution_hash"]
            or typed_point.lower_certificate.certificate_hash
            != certificate["certificate_hash"]
            or typed_point.lower_certificate.method.value != certificate["method"]
            or typed_point.lower_certificate.dual_objective_lower_bound
            != certificate["dual_objective_lower_bound"]
        ):
            raise ScientificArtifactError(
                "frontier typed certificate hashes are invalid"
            )
        if beta == "infinity":
            if (
                not math.isclose(
                    point["target_reward"],
                    problem.maximum_reward,
                    abs_tol=1e-12,
                )
                or certificate["dual_objective_lower_bound"] is not None
            ):
                raise ScientificArtifactError(
                    "infinite dual beta is valid only at the reward endpoint"
                )
            continue
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(beta)
            or beta < 0
        ):
            raise ScientificArtifactError("frontier dual beta is invalid")
        dual = solve_lagrangian(
            problem,
            beta,
            tolerance=solver_settings["lagrangian_tolerance"],
            max_iterations=solver_settings["lagrangian_max_iterations"],
        )
        if not math.isclose(
            dual.objective_lower_bound,
            certificate["dual_objective_lower_bound"],
            abs_tol=1e-9,
        ):
            raise ScientificArtifactError("frontier dual objective is invalid")
        recomputed_lower = max(
            0.0,
            beta * point["target_reward"] + dual.objective_lower_bound,
        )
        if not math.isclose(
            recomputed_lower,
            certificate["lower_bound"],
            abs_tol=max(1e-8, solver_settings["bound_tolerance"]),
        ):
            raise ScientificArtifactError("frontier dual lower bound is invalid")
