from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "paper" / "evidence-status.json"
REGISTERED_COMMITS = {
    "registration_commit": "3909db8b9913fc0031710b9f4d4803fad00dd0bf",
    "operational_record_commit": "c9b6297b63b572d9e6d106de4add1dae436c00d3",
}


def _section_body(markdown: str, *, heading: str, next_heading: str) -> str:
    assert markdown.count(heading) == 1
    assert markdown.count(next_heading) == 1
    start = markdown.index(heading) + len(heading)
    end = markdown.index(next_heading, start)
    return markdown[start:end].strip()


def test_pre_data_manuscript_boundary_is_explicit() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    assert status["schema_version"] == 1

    v1 = status["studies"]["symbolic-v1"]
    assert v1 == {
        "status": "calibration_stopped",
        "release": "v0.1.0",
        "confirmatory_generated": False,
    }

    v2 = status["studies"]["symbolic-v2"]
    assert v2["status"] == "pre_data"
    manuscript_bytes = (ROOT / status["manuscript"]).read_bytes()
    manuscript = manuscript_bytes.decode("utf-8")
    assert hashlib.sha256(manuscript_bytes).hexdigest() == v2["manuscript_sha256"]
    assert manuscript.count(v2["required_marker"]) == 1

    results_section = v2["results_section"]
    assert (
        _section_body(
            manuscript,
            heading=results_section["heading"],
            next_heading=results_section["next_heading"],
        )
        == results_section["allowed_body"]
    )

    assert {name: v2[name] for name in REGISTERED_COMMITS} == REGISTERED_COMMITS
    for commit in REGISTERED_COMMITS.values():
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            v2["registration_commit"],
            v2["operational_record_commit"],
        ],
        cwd=ROOT,
        check=True,
    )

    for relative_root in v2["forbidden_result_roots"]:
        assert not (ROOT / relative_root).exists()
