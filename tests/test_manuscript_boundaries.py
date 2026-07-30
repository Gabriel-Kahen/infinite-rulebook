from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from infinite_rulebook.orchestration.release import load_study_release_manifest

ROOT = Path(__file__).resolve().parents[1]
V1_RELEASE_COMMIT = "af4d0587c071fd451d6f4ef450676a7e0566563e"
V1_RELEASE_SCIENTIFIC_HASH = (
    "e48c6194ce5cb63ce9d504e3a5b4efa7ef3e96f647b7e3c30eef10db48975d32"
)
MANUSCRIPT_SHA256 = "1e2671762c439a198cf9af7a4c4069e9e97cc81960ee6f2340cec0a5fea499db"
EVIDENCE_MAP_SHA256 = "27115cf2198aca04d9f700bf37c54f5eebb7b416c381f92ab5d71190cced315c"
REGISTERED_COMMITS = {
    "registration_commit": "3909db8b9913fc0031710b9f4d4803fad00dd0bf",
    "operational_record_commit": "c9b6297b63b572d9e6d106de4add1dae436c00d3",
    "approved_execution_commit": "c9b6297b63b572d9e6d106de4add1dae436c00d3",
}
PRE_DATA_MARKER = "**PRE-DATA — intentionally blank.**"
V2_RESULTS_HEADING = "## 8. Version 2 results"
V2_NEXT_HEADING = "## 9. Implemented foundations and planned extensions"
V2_ALLOWED_BODY = """\
**PRE-DATA — intentionally blank.**

No v2 effect estimate, hypothesis decision, selected sample size, seal, or
confirmatory result exists. This section may be populated only from the
authenticated registered report and must retain unfavorable and stopped
outcomes."""
FORBIDDEN_V2_PATHS = {
    "artifacts/symbolic-calibration-v2-serial",
    "artifacts/symbolic-calibration-v2-parallel",
    "evidence/symbolic-calibration-v2-reproducibility.json",
    "results/symbolic-calibration-v2",
    "configs/symbolic-confirmatory-v2.json",
    "configs/symbolic-confirmatory-analysis-v2.json",
    "configs/symbolic-confirmatory-canaries-v2.json",
    "configs/symbolic-confirmatory-supplemental-v2.json",
    "artifacts/symbolic-confirmatory-v2-serial",
    "artifacts/symbolic-confirmatory-v2-parallel",
    "evidence/symbolic-confirmatory-v2-reproducibility.json",
    "results/symbolic-confirmatory-v2",
}
FROZEN_REGISTRATION_PATHS = {
    "configs/symbolic-artifact-ingestion-probe-v2.json",
    "configs/symbolic-calibration-analysis-v2.json",
    "configs/symbolic-calibration-canaries-v2.json",
    "configs/symbolic-calibration-supplemental-v2.json",
    "configs/symbolic-calibration-v2.json",
    "docs/symbolic-confirmatory-v2.md",
}
FROZEN_OPERATIONAL_RECORD_PATHS = {
    "benchmarks/symbolic-v2-operational-preflight.json",
    "docs/symbolic-v2-operational-preflight.md",
}


def _repository_path(relative: str) -> Path:
    path = Path(relative)
    assert relative
    assert not path.is_absolute()
    assert ".." not in path.parts
    return ROOT / path


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads(_repository_path(relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _is_git_checkout() -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return (
        result.returncode == 0
        and Path(result.stdout.strip()).resolve() == ROOT.resolve()
    )


def _section_body(markdown: str, *, heading: str, next_heading: str) -> str:
    lines = markdown.splitlines()
    assert lines.count(heading) == 1
    assert lines.count(next_heading) == 1
    start = lines.index(heading) + 1
    end = lines.index(next_heading, start)
    return "\n".join(lines[start:end]).strip()


def test_evidence_documents_and_release_are_pinned() -> None:
    status = _load_json("paper/evidence-status.json")
    assert status["schema_version"] == 1
    assert status["manuscript"] == "paper/manuscript.md"
    assert status["evidence_map"] == "paper/evidence-map.md"

    evidence_map = _repository_path(status["evidence_map"])
    assert status["evidence_map_sha256"] == EVIDENCE_MAP_SHA256
    assert _sha256(evidence_map) == EVIDENCE_MAP_SHA256

    v1 = status["studies"]["symbolic-v1"]
    assert v1 == {
        "status": "calibration_stopped",
        "release": "v0.1.0",
        "release_commit": V1_RELEASE_COMMIT,
        "release_manifest": "results/symbolic-calibration-v1/release-manifest.json",
        "release_manifest_sha256": (
            "42da03348145d166ea0b748e3a2cc79b0c973619f51ccc4de066b46d75b77fea"
        ),
        "release_scientific_hash": V1_RELEASE_SCIENTIFIC_HASH,
        "confirmatory_generated": False,
    }
    assert (
        _sha256(_repository_path(v1["release_manifest"]))
        == (v1["release_manifest_sha256"])
    )
    release = load_study_release_manifest(_repository_path(v1["release_manifest"]))
    assert release.phase == "calibration"
    assert release.study_contract == "bounded-symbolic-construct-validation.v1"
    assert release.scientific_hash == v1["release_scientific_hash"]


@pytest.mark.skipif(not _is_git_checkout(), reason="requires repository history")
def test_evidence_commits_are_in_repository_history() -> None:
    status = _load_json("paper/evidence-status.json")
    assert _git("rev-parse", "v0.1.0^{}").stdout.strip() == V1_RELEASE_COMMIT
    _git("merge-base", "--is-ancestor", V1_RELEASE_COMMIT, "HEAD")

    v2 = status["studies"]["symbolic-v2"]
    assert {name: v2[name] for name in REGISTERED_COMMITS} == REGISTERED_COMMITS
    for commit in set(REGISTERED_COMMITS.values()):
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    _git(
        "merge-base",
        "--is-ancestor",
        v2["registration_commit"],
        v2["operational_record_commit"],
    )
    _git("merge-base", "--is-ancestor", v2["approved_execution_commit"], "HEAD")
    assert set(v2["frozen_registration_paths"]) == FROZEN_REGISTRATION_PATHS
    _git(
        "diff",
        "--quiet",
        v2["registration_commit"],
        "HEAD",
        "--",
        *sorted(FROZEN_REGISTRATION_PATHS),
    )
    assert set(v2["frozen_operational_record_paths"]) == FROZEN_OPERATIONAL_RECORD_PATHS
    _git(
        "diff",
        "--quiet",
        v2["operational_record_commit"],
        "HEAD",
        "--",
        *sorted(FROZEN_OPERATIONAL_RECORD_PATHS),
    )


def test_v1_released_claims_match_artifacts() -> None:
    summary = _load_json("results/symbolic-calibration-v1/summary.json")
    assert {
        name: summary[name]
        for name in (
            "phase",
            "smoke_prerequisite_passed",
            "run_count",
            "reproducibility_passed",
            "canaries_passed",
            "deviation_count",
            "selected_environment_replicas",
            "freeze_eligible",
            "interpretation_eligible",
        )
    } == {
        "phase": "calibration",
        "smoke_prerequisite_passed": True,
        "run_count": 20_736,
        "reproducibility_passed": True,
        "canaries_passed": True,
        "deviation_count": 0,
        "selected_environment_replicas": None,
        "freeze_eligible": False,
        "interpretation_eligible": False,
    }

    calibration = _load_json("results/symbolic-calibration-v1/power.json")[
        "calibration"
    ]
    adequacy = {record["name"]: record for record in calibration["effect_adequacy"]}
    assert len(adequacy) == 6
    failed = {name: record for name, record in adequacy.items() if not record["passes"]}
    assert set(failed) == {"relevant-over-total-trivia-hidden-reward"}
    trivia = failed["relevant-over-total-trivia-hidden-reward"]
    assert (
        trivia["interval_lower"],
        trivia["interval_upper"],
        trivia["threshold_lower"],
    ) == (0.0, 2.0 / 3.0, 0.25)
    assert calibration["selected_environment_count"] is None

    center = next(
        candidate
        for candidate in calibration["candidates"]
        if candidate["environment_count"] == calibration["center_environment_count"]
    )
    signs = next(
        hypothesis
        for hypothesis in center["hypotheses"]
        if hypothesis["name"] == "relevant-over-total-trivia-hidden-reward"
    )
    assert (
        signs["favorable_sign_successes"],
        signs["favorable_sign_trials"],
    ) == (58, 128)


def test_pre_data_manuscript_boundary_is_explicit() -> None:
    status = _load_json("paper/evidence-status.json")
    v2 = status["studies"]["symbolic-v2"]
    assert v2["status"] == "pre_data"
    manuscript_path = _repository_path(status["manuscript"])
    manuscript_bytes = manuscript_path.read_bytes()
    manuscript = manuscript_bytes.decode("utf-8")
    assert v2["manuscript_sha256"] == MANUSCRIPT_SHA256
    assert _sha256(manuscript_path) == MANUSCRIPT_SHA256
    assert v2["required_marker"] == PRE_DATA_MARKER
    assert manuscript.count(PRE_DATA_MARKER) == 1

    results_section = v2["results_section"]
    assert results_section == {
        "heading": V2_RESULTS_HEADING,
        "next_heading": V2_NEXT_HEADING,
        "allowed_body": V2_ALLOWED_BODY,
    }
    assert (
        _section_body(
            manuscript,
            heading=V2_RESULTS_HEADING,
            next_heading=V2_NEXT_HEADING,
        )
        == V2_ALLOWED_BODY
    )

    assert set(v2["forbidden_result_roots"]) == FORBIDDEN_V2_PATHS
    for relative_root in FORBIDDEN_V2_PATHS:
        path = _repository_path(relative_root)
        assert not path.exists()
        assert not path.is_symlink()
