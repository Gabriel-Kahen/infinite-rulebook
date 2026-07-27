"""Validated immutable artifacts and restart-safe event journals."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from infinite_rulebook.agents import CapabilityManifest
from infinite_rulebook.artifacts import semantic_hash as typed_semantic_hash
from infinite_rulebook.frontier.blahut_arimoto import solve_lagrangian
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.inversion import FrontierSolution
from infinite_rulebook.metrics import FrontierPoint
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash

ARTIFACT_SCHEMA_VERSION = 1
_EVENT_NAME = re.compile(r"^(\d{8})\.json$")


class ScientificArtifactError(ValueError):
    """Raised when an artifact is invalid, incompatible, or would be mutated."""


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
            "semantic_hashes": semantic_hashes,
            "payload": normalized_payload,
        }
        return cls(
            artifact_type=artifact_type,
            semantic_hashes=dict(semantic_hashes),
            payload=normalized_payload,
            scientific_hash=scientific_hash(
                scientific_payload, domain="artifact-envelope"
            ),
            runtime_metadata=normalized_runtime,
        )

    def validate(
        self,
        expected_semantic_hashes: dict[str, str] | None = None,
    ) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ScientificArtifactError("unsupported artifact schema version")
        if not self.artifact_type:
            raise ScientificArtifactError("artifact_type must not be empty")
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
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
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
            return json.load(stream)
    except FileNotFoundError as error:
        raise _ArtifactNotFound(f"artifact does not exist: {path}") from error
    except ScientificArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as error:
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
            elif name.endswith(".json"):
                if not stat.S_ISREG(metadata.st_mode):
                    raise ScientificArtifactError(
                        f"artifact is not a regular file: {path / member}"
                    )
                result.append(path / member)

    try:
        walk(root, Path())
    finally:
        os.close(root)
    return tuple(result)


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
        for path in self.list_artifacts():
            if path.name == "manifest.json":
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


@dataclass(frozen=True, slots=True)
class JournalEvent:
    sequence: int
    event_key: str
    event_kind: str
    previous_hash: str | None
    payload: Any
    event_hash: str


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
        result: list[JournalEvent] = []
        previous_hash = None
        for expected_sequence, (sequence, path) in enumerate(paths):
            if sequence != expected_sequence:
                raise ScientificArtifactError("event journal has a sequence gap")
            envelope = read_artifact(
                path, expected_semantic_hashes=self.semantic_hashes
            )
            if envelope.artifact_type != "training-event":
                raise ScientificArtifactError("event journal contains a wrong type")
            payload = envelope.payload
            if not isinstance(payload, dict):
                raise ScientificArtifactError("event payload must be an object")
            required = {
                "sequence",
                "event_key",
                "event_kind",
                "previous_hash",
                "payload",
                "event_hash",
            }
            if set(payload) != required:
                raise ScientificArtifactError("training event fields are invalid")
            scientific_event = {
                name: payload[name] for name in required - {"event_hash"}
            }
            event_hash = scientific_hash(scientific_event, domain="training-event")
            invalid_chain = (
                payload["sequence"] != sequence
                or payload["previous_hash"] != previous_hash
            )
            if invalid_chain:
                raise ScientificArtifactError("event journal hash chain is invalid")
            if payload["event_hash"] != event_hash:
                raise ScientificArtifactError("training event hash mismatch")
            event = JournalEvent(**payload)
            result.append(event)
            previous_hash = event.event_hash
        self._events = result
        self._events_by_key = {event.event_key: event for event in result}
        if len(self._events_by_key) != len(result):
            raise ScientificArtifactError("event journal has duplicate event keys")
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

    members = []
    members.append(
        store.write("frontier/curve.json", "frontier-curve", semantic_hashes, curve)
    )
    for name, payload in sorted(witnesses.items()):
        members.append(
            store.write(
                f"frontier/witnesses/{name}.json",
                "frontier-witness",
                semantic_hashes,
                payload,
            )
        )
    for name, payload in sorted(certificates.items()):
        members.append(
            store.write(
                f"frontier/certificates/{name}.json",
                "frontier-certificate",
                semantic_hashes,
                payload,
            )
        )
    members.append(
        store.write(
            "frontier/diagnostics.json",
            "frontier-diagnostics",
            semantic_hashes,
            diagnostics,
        )
    )
    return store.write(
        "frontier/manifest.json",
        "frontier-manifest",
        semantic_hashes,
        {"member_hashes": [member.scientific_hash for member in members]},
    )


def validate_artifact_tree(
    root: str | Path,
    *,
    expected_semantic_hashes: dict[str, str] | None = None,
) -> tuple[ArtifactEnvelope, ...]:
    path = Path(root)
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
        expected_members = manifest.payload["members"]
        actual_members = []
        for file in files:
            if file.name == "manifest.json" and file.parent == path:
                continue
            artifact = read_artifact(file)
            actual_members.append(
                {
                    "path": file.relative_to(path).as_posix(),
                    "artifact_type": artifact.artifact_type,
                    "scientific_hash": artifact.scientific_hash,
                }
            )
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
    _validate_frontier_records(path, records)
    if manifests:
        _validate_frontier_reference(path, records)
    return artifacts


def _validate_completed_run(
    root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
    manifest: ArtifactEnvelope,
) -> None:
    del manifest
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

    config = by_type["resolved-run-config"][0][1].payload
    metrics = by_type["run-metrics"][0][1].payload
    run_settings = config["run_settings"]
    seeds = config["seeds"]
    horizon = run_settings["horizon"]
    journal = EventJournal(
        ArtifactStore(root),
        by_type["run-manifest"][0][1].semantic_hashes,
    )
    events = journal.events()
    if (
        len(events) != horizon
        or metrics.get("event_count") != horizon
        or metrics.get("completed_rounds") != horizon
    ):
        raise ScientificArtifactError(
            "completed run event count does not match its horizon"
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
        try:
            typed = validate_checkpoint_record(result["scientific_records"])
        except (KeyError, TypeError, ValueError) as error:
            raise ScientificArtifactError(
                "completed run typed checkpoint record is invalid"
            ) from error
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


def _validate_frontier_reference(
    run_root: Path,
    records: tuple[tuple[Path, ArtifactEnvelope], ...],
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
    frontier_artifacts = validate_artifact_tree(
        artifact_root / "_frontiers" / frontier_hash,
        expected_semantic_hashes={"frontier": frontier_hash},
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
    members = [
        artifact
        for _, artifact in records
        if artifact.artifact_type.startswith("frontier-")
        and artifact.artifact_type != "frontier-manifest"
    ]
    if sorted(manifests[0].payload["member_hashes"]) != sorted(
        member.scientific_hash for member in members
    ):
        raise ScientificArtifactError("frontier manifest member list is invalid")

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
        evaluated = problem.evaluate(witness_payload["channel"])
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
                witness=evaluated,
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
