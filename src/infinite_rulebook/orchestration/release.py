"""Authenticated manifests for deterministic study-report packages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import load_json_strict

STUDY_RELEASE_MANIFEST_VERSION = 1
STUDY_RELEASE_MANIFEST_FILENAME = "release-manifest.json"
_MANIFEST_DOMAIN = "symbolic-study-release-manifest"


@dataclass(frozen=True, slots=True)
class ReleaseMember:
    path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if (
            not self.path
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.parts[0] in {".", ".."}
        ):
            raise ValueError("release member path must be one safe relative component")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise ValueError("release member byte_size must be nonnegative")
        if not is_sha256(self.sha256):
            raise ValueError("release member sha256 must be a SHA-256 digest")

    @classmethod
    def from_file(cls, root: Path, name: str) -> ReleaseMember:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release member is not a regular file: {path}")
        content = path.read_bytes()
        return cls(name, len(content), hashlib.sha256(content).hexdigest())

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class StudyReleaseManifest:
    phase: str
    study_contract: str
    config_hash: str
    freeze_hash: str | None
    calibration_evidence_hash: str | None
    members: tuple[ReleaseMember, ...]
    scientific_hash: str

    def __post_init__(self) -> None:
        if self.phase not in {"calibration", "confirmatory"}:
            raise ValueError("release manifest phase is invalid")
        if not isinstance(self.study_contract, str) or not self.study_contract:
            raise ValueError("release manifest study_contract must not be empty")
        if not is_sha256(self.config_hash):
            raise ValueError("release manifest config_hash is invalid")
        for name, value in (
            ("freeze_hash", self.freeze_hash),
            ("calibration_evidence_hash", self.calibration_evidence_hash),
        ):
            if value is not None and not is_sha256(value):
                raise ValueError(f"release manifest {name} is invalid")
        members = tuple(sorted(self.members, key=lambda item: item.path))
        if not members or len({item.path for item in members}) != len(members):
            raise ValueError("release manifest members must be nonempty and unique")
        object.__setattr__(self, "members", members)
        if self.scientific_hash != scientific_hash(
            self.body_dict(),
            domain=_MANIFEST_DOMAIN,
        ):
            raise ValueError("release manifest scientific_hash is invalid")

    def body_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "symbolic-study-release-manifest",
            "schema_version": STUDY_RELEASE_MANIFEST_VERSION,
            "phase": self.phase,
            "study_contract": self.study_contract,
            "config_hash": self.config_hash,
            "freeze_hash": self.freeze_hash,
            "calibration_evidence_hash": self.calibration_evidence_hash,
            "members": [member.to_dict() for member in self.members],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "scientific_hash": self.scientific_hash}

    @classmethod
    def create(
        cls,
        root: str | Path,
        member_names: tuple[str, ...],
        *,
        phase: str,
        study_contract: str,
        config_hash: str,
        freeze_hash: str | None,
        calibration_evidence_hash: str | None,
    ) -> StudyReleaseManifest:
        directory = Path(root)
        members = tuple(
            ReleaseMember.from_file(directory, name) for name in sorted(member_names)
        )
        body = {
            "artifact_type": "symbolic-study-release-manifest",
            "schema_version": STUDY_RELEASE_MANIFEST_VERSION,
            "phase": phase,
            "study_contract": study_contract,
            "config_hash": config_hash,
            "freeze_hash": freeze_hash,
            "calibration_evidence_hash": calibration_evidence_hash,
            "members": [member.to_dict() for member in members],
        }
        return cls(
            phase=phase,
            study_contract=study_contract,
            config_hash=config_hash,
            freeze_hash=freeze_hash,
            calibration_evidence_hash=calibration_evidence_hash,
            members=members,
            scientific_hash=scientific_hash(body, domain=_MANIFEST_DOMAIN),
        )

    def verify_files(
        self,
        root: str | Path,
        *,
        manifest_name: str = STUDY_RELEASE_MANIFEST_FILENAME,
    ) -> None:
        directory = Path(root)
        entries = tuple(directory.iterdir())
        expected_names = {manifest_name, *(member.path for member in self.members)}
        if {entry.name for entry in entries} != expected_names or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise ValueError(
                "release package inventory does not exactly match the manifest"
            )
        observed = tuple(
            ReleaseMember.from_file(directory, member.path) for member in self.members
        )
        if observed != self.members:
            raise ValueError("release package members do not match the manifest")


def study_release_manifest_from_dict(raw: object) -> StudyReleaseManifest:
    if not isinstance(raw, dict):
        raise TypeError("release manifest must be an object")
    expected = {
        "artifact_type",
        "schema_version",
        "phase",
        "study_contract",
        "config_hash",
        "freeze_hash",
        "calibration_evidence_hash",
        "members",
        "scientific_hash",
    }
    if set(raw) != expected:
        raise ValueError("release manifest fields are invalid")
    if (
        raw["artifact_type"] != "symbolic-study-release-manifest"
        or isinstance(raw["schema_version"], bool)
        or not isinstance(raw["schema_version"], int)
        or raw["schema_version"] != STUDY_RELEASE_MANIFEST_VERSION
    ):
        raise ValueError("release manifest type or schema is invalid")
    member_values = raw["members"]
    if not isinstance(member_values, list):
        raise TypeError("release manifest members must be an array")
    members = []
    for value in member_values:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "byte_size",
            "sha256",
        }:
            raise ValueError("release manifest member fields are invalid")
        members.append(ReleaseMember(**value))
    manifest = StudyReleaseManifest(
        phase=raw["phase"],
        study_contract=raw["study_contract"],
        config_hash=raw["config_hash"],
        freeze_hash=raw["freeze_hash"],
        calibration_evidence_hash=raw["calibration_evidence_hash"],
        members=tuple(members),
        scientific_hash=raw["scientific_hash"],
    )
    if manifest.to_dict() != raw:
        raise ValueError("release manifest is not canonical")
    return manifest


def load_study_release_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> StudyReleaseManifest:
    source = Path(path)
    manifest = study_release_manifest_from_dict(
        load_json_strict(source, label="study release manifest")
    )
    if verify_files:
        manifest.verify_files(source.parent, manifest_name=source.name)
    return manifest


__all__ = [
    "STUDY_RELEASE_MANIFEST_FILENAME",
    "STUDY_RELEASE_MANIFEST_VERSION",
    "ReleaseMember",
    "StudyReleaseManifest",
    "load_study_release_manifest",
    "study_release_manifest_from_dict",
]
