from __future__ import annotations

import json
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ScientificArtifactError,
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _replace(path: Path, envelope: ArtifactEnvelope) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(envelope.to_dict()), encoding="utf-8")


def test_completed_tree_revalidates_typed_checkpoint_content(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    cell = config.cells()[0]
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)
    checkpoint_path = result.path / "checkpoints/00000004.json"
    checkpoint = read_artifact(checkpoint_path)
    changed_result = {**checkpoint.payload["result"], "expected_reward": 999.0}
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {**checkpoint.payload, "result": changed_result},
    )
    _replace(checkpoint_path, changed)

    manifest_path = result.path / "manifest.json"
    manifest = read_artifact(manifest_path)
    members = [
        (
            {
                **member,
                "scientific_hash": changed.scientific_hash,
            }
            if member["path"] == "checkpoints/00000004.json"
            else member
        )
        for member in manifest.payload["members"]
    ]
    changed_manifest = ArtifactEnvelope.create(
        "run-manifest",
        manifest.semantic_hashes,
        {
            "members": members,
            "scientific_content_hash": scientific_hash(
                members,
                domain="run-scientific-content",
            ),
        },
        runtime_metadata=manifest.runtime_metadata,
    )
    _replace(manifest_path, changed_manifest)

    with pytest.raises(ScientificArtifactError, match="typed record"):
        validate_artifact_tree(result.path)
