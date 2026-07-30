from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from infinite_rulebook.analysis.loading import load_run_tree
from infinite_rulebook.orchestration.artifacts import (
    ArtifactEnvelope,
    ArtifactStore,
    EventJournal,
    ScientificArtifactError,
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    SYMBOLIC_ADAPTER_CONTRACT_V1,
    SYMBOLIC_ADAPTER_CONTRACT_V2,
    SYMBOLIC_V2_CALIBRATION_EXPERIMENT_NAME,
    SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME,
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    run_cell_identity_payload,
    symbolic_adapter_contract,
)
from infinite_rulebook.orchestration.freeze import ConfirmatoryFreezeError
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.provenance import ScientificProvenance
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.seeds import SeedBank
from infinite_rulebook.orchestration.semantics import semantic_hashes
from infinite_rulebook.orchestration.symbolic import (
    ExactSymbolicAdapter,
    ExactSymbolicAdapterV2,
)


def _v2_config() -> ExperimentConfig:
    return ExperimentConfig(
        name=SYMBOLIC_V2_CALIBRATION_EXPERIMENT_NAME,
        environments=(
            EnvironmentConfig(
                EnvironmentKind.TRIVIA,
                projection_size=2,
                distractor_dimensions=6,
            ),
        ),
        agents=(AgentConfig(AgentKind.RELEVANT_INFORMATION, target_size=2),),
        checkpoints=CheckpointConfig((0, 1, 2, 3)),
        horizon=3,
        master_seed="v2-adapter-test",
        algorithm_master_seed="v2-adapter-algorithms",
        phase="calibration",
    )


def _golden_v1_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="v1-golden",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1, 2)),
        horizon=2,
        master_seed="v1-golden-seed",
        algorithm_master_seed="v1-golden-algorithm",
    )


def _fixed_provenance() -> ScientificProvenance:
    return ScientificProvenance(
        code_commit="c" * 40,
        dirty_tree_hash="d" * 64,
        dependency_lock_hash="e" * 64,
        analysis_code_hash="a" * 64,
        environment_digest="f" * 64,
        python_implementation="CPython",
        python_version="3.11.15",
    )


def _replace_member_and_manifest(
    run_root: Path,
    member_path: Path,
    changed: ArtifactEnvelope,
) -> None:
    member_path.chmod(0o644)
    member_path.write_text(json.dumps(changed.to_dict()), encoding="utf-8")
    manifest_path = run_root / "manifest.json"
    manifest = read_artifact(manifest_path)
    relative_path = member_path.relative_to(run_root).as_posix()
    members = [
        (
            {**member, "scientific_hash": changed.scientific_hash}
            if member["path"] == relative_path
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
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(changed_manifest.to_dict()),
        encoding="utf-8",
    )


def test_v2_contract_mapping_is_closed_and_phase_bound(tmp_path: Path) -> None:
    config = _v2_config()
    settings = config.resolved_run_settings()
    assert settings["adapter_contract"] == SYMBOLIC_ADAPTER_CONTRACT_V2
    assert settings["experiment_name"] == SYMBOLIC_V2_CALIBRATION_EXPERIMENT_NAME
    assert (
        symbolic_adapter_contract(SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME)
        == SYMBOLIC_ADAPTER_CONTRACT_V2
    )
    assert (
        symbolic_adapter_contract("symbolic-construct-calibration-v2-lookalike")
        == SYMBOLIC_ADAPTER_CONTRACT_V1
    )
    with pytest.raises(ValueError, match="requires phase"):
        replace(config, phase="pilot")
    with pytest.raises(ConfirmatoryFreezeError, match="requires a valid"):
        replace(
            config,
            name=SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME,
            phase="confirmatory",
        )
    with pytest.raises(ValueError, match="configured contract"):
        RunExecutor(tmp_path, ExactSymbolicAdapter).execute(
            config,
            config.cells()[0],
        )

    class LookalikeV2(ExactSymbolicAdapterV2):
        pass

    with pytest.raises(ConfirmatoryFreezeError, match="registered exact adapter"):
        RunExecutor(tmp_path, LookalikeV2).execute(config, config.cells()[0])


def test_v2_event_reward_timing_and_checkpoint_mean() -> None:
    config = _v2_config()
    cell = config.cells()[0]
    seeds = SeedBank(
        config.master_seed,
        config.algorithm_master_seed,
    ).for_cell(cell)
    hashes = semantic_hashes(cell)
    adapter = ExactSymbolicAdapterV2()
    state = adapter.initial_state(cell, seeds)

    checkpoint = adapter.checkpoint(state, 0, cell, seeds, hashes)
    assert "post_query_hidden_expected_reward" not in checkpoint
    assert "post_query_mean_hidden_expected_reward" not in checkpoint
    rewards = []
    for round_index in range(config.horizon):
        before = adapter.state_fingerprint(state)
        event = adapter.training_event(state, round_index, cell, seeds)
        assert adapter.state_fingerprint(state) == before
        rewards.append(event["post_query_hidden_expected_reward"])
        state = adapter.apply_training_event(state, event)
        checkpoint = adapter.checkpoint(
            state,
            round_index + 1,
            cell,
            seeds,
            hashes,
        )
        assert (
            event["post_query_hidden_expected_reward"]
            == checkpoint["post_query_hidden_expected_reward"]
            == checkpoint["hidden_expected_reward"]
        )
        assert checkpoint["post_query_mean_hidden_expected_reward"] == (
            math.fsum(rewards) / len(rewards)
        )

    fresh = adapter.initial_state(cell, seeds)
    event = adapter.training_event(fresh, 0, cell, seeds)
    with pytest.raises(ValueError, match="deterministic replay"):
        adapter.apply_training_event(
            fresh,
            {
                **event,
                "post_query_hidden_expected_reward": (
                    event["post_query_hidden_expected_reward"] + 0.25
                ),
            },
        )


def test_v2_run_resumes_replays_and_loads_authenticated_metric(
    tmp_path: Path,
) -> None:
    config = _v2_config()
    cell = config.cells()[0]
    resumed_executor = RunExecutor(tmp_path / "resumed", ExactSymbolicAdapterV2)
    partial = resumed_executor.execute(config, cell, stop_after_new_events=1)
    assert not partial.complete
    assert partial.event_count == 1

    resumed = resumed_executor.execute(config, cell)
    fresh = RunExecutor(tmp_path / "fresh", ExactSymbolicAdapterV2).execute(
        config,
        cell,
    )
    assert resumed.scientific_content_hash == fresh.scientific_content_hash
    validate_artifact_tree(resumed.path)

    store = ArtifactStore(resumed.path)
    hashes = store.read("config.resolved.json").semantic_hashes
    events = EventJournal(store, hashes).events()
    rewards = [event.payload["post_query_hidden_expected_reward"] for event in events]
    observations = load_run_tree(resumed.path)
    assert "post_query_hidden_expected_reward" not in dict(observations[0].metrics)
    assert "post_query_mean_hidden_expected_reward" not in dict(observations[0].metrics)
    for observation in observations[1:]:
        assert (
            dict(observation.metrics)["post_query_hidden_expected_reward"]
            == rewards[observation.round_index - 1]
        )
        assert (
            dict(observation.metrics)["post_query_mean_hidden_expected_reward"]
            == math.fsum(rewards[: observation.round_index]) / observation.round_index
        )


def test_v2_artifacts_reject_rehashed_metric_and_replay_tampering(
    tmp_path: Path,
) -> None:
    config = _v2_config()
    result = RunExecutor(tmp_path / "mean", ExactSymbolicAdapterV2).execute(
        config,
        config.cells()[0],
    )
    checkpoint_path = result.path / "checkpoints/00000002.json"
    checkpoint = read_artifact(checkpoint_path)
    changed_result = dict(checkpoint.payload["result"])
    changed_result["post_query_mean_hidden_expected_reward"] += 0.25
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {**checkpoint.payload, "result": changed_result},
    )
    _replace_member_and_manifest(result.path, checkpoint_path, changed)
    with pytest.raises(ScientificArtifactError, match="post-query"):
        validate_artifact_tree(result.path)

    result = RunExecutor(tmp_path / "source", ExactSymbolicAdapterV2).execute(
        config,
        config.cells()[0],
    )
    checkpoint_path = result.path / "checkpoints/00000002.json"
    checkpoint = read_artifact(checkpoint_path)
    changed_result = dict(checkpoint.payload["result"])
    changed_result["post_query_hidden_expected_reward"] += 0.25
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {**checkpoint.payload, "result": changed_result},
    )
    _replace_member_and_manifest(result.path, checkpoint_path, changed)
    with pytest.raises(ScientificArtifactError, match="post-query"):
        validate_artifact_tree(result.path)

    result = RunExecutor(tmp_path / "replay", ExactSymbolicAdapterV2).execute(
        config,
        config.cells()[0],
    )
    checkpoint_path = result.path / "checkpoints/00000002.json"
    checkpoint = read_artifact(checkpoint_path)
    changed = ArtifactEnvelope.create(
        checkpoint.artifact_type,
        checkpoint.semantic_hashes,
        {
            **checkpoint.payload,
            "result": {
                **checkpoint.payload["result"],
                "evaluation": "rehashed-fabrication",
            },
        },
    )
    _replace_member_and_manifest(result.path, checkpoint_path, changed)
    with pytest.raises(ScientificArtifactError, match="exact adapter replay"):
        validate_artifact_tree(result.path)


def test_v2_artifacts_reject_rehashed_registered_name_tampering(
    tmp_path: Path,
) -> None:
    config = _v2_config()
    result = RunExecutor(tmp_path, ExactSymbolicAdapterV2).execute(
        config,
        config.cells()[0],
    )
    config_path = result.path / "config.resolved.json"
    resolved = read_artifact(config_path)
    run_settings = {
        **resolved.payload["run_settings"],
        "experiment_name": SYMBOLIC_V2_CONFIRMATORY_EXPERIMENT_NAME,
    }
    changed_run_hash = scientific_hash(
        {
            "runner_version": "symbolic-runner.v1",
            "run_settings": run_settings,
            "cell": run_cell_identity_payload(resolved.payload["cell"]),
            "seeds": resolved.payload["seeds"],
            "provenance": resolved.payload["provenance"],
        },
        domain="run-identity",
    )
    changed = ArtifactEnvelope.create(
        resolved.artifact_type,
        resolved.semantic_hashes,
        {
            **resolved.payload,
            "run_hash": changed_run_hash,
            "run_settings": run_settings,
        },
    )
    _replace_member_and_manifest(result.path, config_path, changed)
    tampered_root = result.path.parent / changed_run_hash
    result.path.rename(tampered_root)
    with pytest.raises(ScientificArtifactError, match="registered experiment"):
        validate_artifact_tree(tampered_root)


def test_v1_resolved_settings_and_artifact_hashes_remain_golden(
    tmp_path: Path,
) -> None:
    config = _golden_v1_config()
    assert config.config_hash == (
        "0d4a0eaae49512c943207e4b46028ddd539fc8b6e984961fd074cce8134455bf"
    )
    assert config.resolved_run_settings() == {
        "schema_version": 1,
        "adapter_contract": "exact-symbolic-adapter.v1",
        "phase": "pilot",
        "horizon": 2,
        "checkpoints": {"rounds": (0, 1, 2)},
        "master_seed": "v1-golden-seed",
        "algorithm_master_seed": "v1-golden-algorithm",
    }
    result = RunExecutor(
        tmp_path,
        ExactSymbolicAdapter,
        provenance=_fixed_provenance(),
    ).execute(config, config.cells()[0], runtime_metadata={"stable": "ignored"})
    assert result.run_hash == (
        "090a43fb27b74bba2f944fd6882ba4b53ba0753414c583ef46503f48bf3902e8"
    )
    assert result.scientific_content_hash == (
        "ac955413030cfd994a5a89a4e15e944f64cc73ae8bb9980f96acc746f8281834"
    )
    expected_hashes = {
        "config.resolved.json": (
            "fab9d9435a4dd10df454d4dac8807630e2005a79e1b33d6d3d0a3b4ae8e20923"
        ),
        "events/00000000.json": (
            "6e1cf2d36619444936e316792316f6a5777e1004d947b64595b70b2fcb545cde"
        ),
        "events/00000001.json": (
            "e3e27a106fe0fd060dddc87c5044bbdf8479e00f33334054834c7817e12970eb"
        ),
        "checkpoints/00000000.json": (
            "51534b22a63a8e1332bc904bbe44f299e9162fa65caaae6506504927486418ab"
        ),
        "checkpoints/00000001.json": (
            "79b133adec7675ebbf36bb2f5a14d6622db32d6e4ca005360c8c8589dc38b9e5"
        ),
        "checkpoints/00000002.json": (
            "a435343efca98dd65fe9092ed2118a8f983ffaf132bf83aec04550fcd1306eb7"
        ),
        "metrics.json": (
            "74aaab9507029e118b494b4dca78289b7955a4e9e559070d964eb21e06872d9b"
        ),
        "manifest.json": (
            "9ef305ba4219e7741224a65333e276b42d04f7341a02e5599b47bab3a9203e60"
        ),
    }
    assert {
        relative: read_artifact(result.path / relative).scientific_hash
        for relative in expected_hashes
    } == expected_hashes
    for checkpoint in (result.path / "checkpoints").glob("*.json"):
        assert (
            "post_query_mean_hidden_expected_reward"
            not in read_artifact(checkpoint).payload["result"]
        )
    for event in (result.path / "events").glob("*.json"):
        assert (
            "post_query_hidden_expected_reward"
            not in read_artifact(event).payload["payload"]
        )
