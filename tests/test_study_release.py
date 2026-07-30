from __future__ import annotations

import json
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.release import (
    StudyReleaseManifest,
    load_study_release_manifest,
    study_release_manifest_from_dict,
)


def _manifest(tmp_path: Path) -> StudyReleaseManifest:
    (tmp_path / "analysis.json").write_text('{"result":true}\n', encoding="utf-8")
    (tmp_path / "summary.json").write_text('{"passed":true}\n', encoding="utf-8")
    return StudyReleaseManifest.create(
        tmp_path,
        ("summary.json", "analysis.json"),
        phase="calibration",
        study_contract="study.v1",
        config_hash="a" * 64,
        freeze_hash=None,
        calibration_evidence_hash="b" * 64,
    )


def test_release_manifest_authenticates_exact_file_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "release-manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    assert load_study_release_manifest(path) == manifest

    (tmp_path / "summary.json").write_text('{"passed":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_study_release_manifest(path)


def test_release_manifest_rejects_unlisted_package_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    (tmp_path / "unlisted.txt").write_text("not authenticated", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        load_study_release_manifest(path)


def test_release_manifest_rejects_tampering_and_unsafe_members(
    tmp_path: Path,
) -> None:
    raw = _manifest(tmp_path).to_dict()
    raw["config_hash"] = "c" * 64
    with pytest.raises(ValueError, match="scientific_hash"):
        study_release_manifest_from_dict(raw)

    raw = _manifest(tmp_path).to_dict()
    raw["members"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="safe relative"):
        study_release_manifest_from_dict(raw)


def test_release_manifest_json_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "release-manifest.json"
    path.write_text('{"phase":"calibration","phase":"confirmatory"}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeats key"):
        load_study_release_manifest(path)


def test_release_manifest_rejects_boolean_schema_and_noncanonical_order(
    tmp_path: Path,
) -> None:
    raw = _manifest(tmp_path).to_dict()
    raw["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        study_release_manifest_from_dict(raw)

    raw = _manifest(tmp_path).to_dict()
    raw["members"].reverse()
    with pytest.raises(ValueError, match="canonical"):
        study_release_manifest_from_dict(raw)
