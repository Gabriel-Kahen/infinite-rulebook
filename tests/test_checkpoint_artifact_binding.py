from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ScientificArtifactError,
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    EnvironmentConfig,
    EnvironmentKind,
    load_experiment_config,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _replace(path: Path, envelope: ArtifactEnvelope) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(envelope.to_dict()), encoding="utf-8")


def _replace_checkpoint_and_manifest(
    run_root: Path,
    checkpoint_path: Path,
    changed: ArtifactEnvelope,
) -> None:
    _replace(checkpoint_path, changed)
    manifest_path = run_root / "manifest.json"
    manifest = read_artifact(manifest_path)
    members = [
        (
            {**member, "scientific_hash": changed.scientific_hash}
            if member["path"] == checkpoint_path.relative_to(run_root).as_posix()
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
    _replace_checkpoint_and_manifest(result.path, checkpoint_path, changed)

    with pytest.raises(ScientificArtifactError, match="typed record"):
        validate_artifact_tree(result.path)


@pytest.mark.parametrize(
    "environment_kind",
    [EnvironmentKind.IND, EnvironmentKind.PUBLIC_C],
)
def test_completed_tree_recomputes_reward_decomposition(
    tmp_path: Path,
    environment_kind: EnvironmentKind,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    config = replace(
        config,
        environments=(
            EnvironmentConfig(
                environment_kind,
                projection_size=1,
                public_reward_cap=0.5,
            ),
        ),
    )
    cell = config.cells()[0]
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(config, cell)
    checkpoint_path = result.path / "checkpoints/00000004.json"
    checkpoint = read_artifact(checkpoint_path)
    original = checkpoint.payload["result"]
    changed_result = {
        **original,
        "hidden_expected_reward": original["hidden_expected_reward"] + 1.0,
        "public_reward": original["public_reward"] - 1.0,
    }
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {**checkpoint.payload, "result": changed_result},
    )
    _replace_checkpoint_and_manifest(result.path, checkpoint_path, changed)

    with pytest.raises(ScientificArtifactError, match="typed record"):
        validate_artifact_tree(result.path)


def test_calibration_checkpoint_must_match_exact_adapter_replay(
    tmp_path: Path,
) -> None:
    config = replace(
        load_experiment_config("configs/pilot-foundation.json"),
        phase="calibration",
    )
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    checkpoint_path = result.path / "checkpoints/00000004.json"
    checkpoint = read_artifact(checkpoint_path)
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {
            **checkpoint.payload,
            "result": {
                **checkpoint.payload["result"],
                "evaluation": "fabricated-but-rehashed",
            },
        },
    )
    _replace_checkpoint_and_manifest(result.path, checkpoint_path, changed)

    with pytest.raises(ScientificArtifactError, match="exact adapter replay"):
        validate_artifact_tree(result.path)


@pytest.mark.parametrize(
    ("relative_path", "missing_field"),
    [
        ("config.resolved.json", "run_settings"),
        ("frontier-reference.json", "frontier_hash"),
    ],
)
def test_rehashed_malformed_run_payload_raises_domain_error(
    tmp_path: Path,
    relative_path: str,
    missing_field: str,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    result = RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
        config,
        config.cells()[0],
    )
    member_path = result.path / relative_path
    member = read_artifact(member_path)
    changed_payload = dict(member.payload)
    changed_payload.pop(missing_field)
    changed = ArtifactEnvelope.create(
        member.artifact_type,
        member.semantic_hashes,
        changed_payload,
    )
    _replace_checkpoint_and_manifest(result.path, member_path, changed)

    with pytest.raises(
        ScientificArtifactError,
        match=r"structure|resolved config fields",
    ):
        validate_artifact_tree(result.path)
