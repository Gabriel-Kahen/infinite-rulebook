"""Validated immutable artifacts and restart-safe event journals."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infinite_rulebook.frontier.blahut_arimoto import solve_lagrangian
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash

ARTIFACT_SCHEMA_VERSION = 1
_EVENT_NAME = re.compile(r"^(\d{8})\.json$")


class ScientificArtifactError(ValueError):
    """Raised when an artifact is invalid, incompatible, or would be mutated."""


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
        scientific_payload = {
            "artifact_type": artifact_type,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "semantic_hashes": semantic_hashes,
            "payload": payload,
        }
        return cls(
            artifact_type=artifact_type,
            semantic_hashes=dict(semantic_hashes),
            payload=payload,
            scientific_hash=scientific_hash(
                scientific_payload, domain="artifact-envelope"
            ),
            runtime_metadata=dict(runtime_metadata or {}),
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


def _read_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ScientificArtifactError(f"cannot read artifact {path}") from error


def read_artifact(
    path: str | Path,
    *,
    expected_semantic_hashes: dict[str, str] | None = None,
) -> ArtifactEnvelope:
    envelope = ArtifactEnvelope.from_dict(_read_json(Path(path)))
    envelope.validate(expected_semantic_hashes)
    return envelope


def _exclusive_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(content, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ScientificArtifactError(
                f"immutable artifact already exists: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


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
        if relative.is_absolute() or ".." in relative.parts:
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
        if path.exists():
            existing = read_artifact(path)
            if existing.scientific_hash != envelope.scientific_hash:
                raise ScientificArtifactError(
                    f"refusing to mutate immutable artifact: {path}"
                )
            return existing
        try:
            _exclusive_json(path, envelope.to_dict())
        except ScientificArtifactError as error:
            if not path.exists():
                raise
            existing = read_artifact(path)
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
        if not self.path.exists():
            return ()
        return tuple(
            sorted(path for path in self.path.rglob("*.json") if path.is_file())
        )

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
        if not directory.exists():
            self._events = []
            return self._events
        paths = []
        for path in directory.iterdir():
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
    if not path.is_dir():
        raise ScientificArtifactError(f"artifact tree does not exist: {path}")
    records = tuple(
        (
            file,
            read_artifact(file, expected_semantic_hashes=expected_semantic_hashes),
        )
        for file in sorted(path.rglob("*.json"))
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
        for file in sorted(path.rglob("*.json")):
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
    diagnostic_names = {
        diagnostic["name"] for diagnostic in diagnostics[0].get("points", [])
    }
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
