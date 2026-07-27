"""Deterministic artifact serialization and scientific hash boundaries."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from infinite_rulebook.artifacts import (
    ArtifactEnvelope,
    CheckpointEstimate,
    FrozenArray,
    FrozenFloat,
    RunCheckpoint,
    canonical_json_bytes,
    scientific_payload_hash,
    semantic_hash,
)
from infinite_rulebook.core import DeploymentAction
from infinite_rulebook.information import InformationBreakdown
from infinite_rulebook.metrics import (
    ComputeMetrics,
    EfficiencyMetric,
    FrontierRegretMetrics,
    MetricInterval,
    NoveltyMetrics,
    PopulationInformationEstimate,
    RewardMetrics,
    SupportMetrics,
)
from infinite_rulebook.validation import ValidationReport


def test_canonical_serialization_is_mapping_order_invariant() -> None:
    left = {"z": [1, 2.0], "a": {"beta": True, "alpha": None}}
    right = {"a": {"alpha": None, "beta": True}, "z": [1, 2.0]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert semantic_hash(left) == semantic_hash(right)
    assert scientific_payload_hash(left) != semantic_hash(left)


def test_float_serialization_normalizes_negative_zero_and_tags_infinity() -> None:
    assert canonical_json_bytes(-0.0) == canonical_json_bytes(0.0)
    assert canonical_json_bytes(math.inf) == b'{"$float":"+inf"}'

    with pytest.raises(ValueError, match="NaN"):
        canonical_json_bytes(math.nan)


def test_serialization_is_stable_across_hash_seeds() -> None:
    script = (
        "from infinite_rulebook.artifacts import semantic_hash;"
        "payload={key:key for key in {'alpha','beta','gamma'}};"
        "print(semantic_hash(payload))"
    )
    outputs = []
    for seed in ("1", "999"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1]


def test_runtime_metadata_does_not_change_scientific_hashes() -> None:
    common = {
        "kind": "checkpoint_estimate",
        "schema_version": 1,
        "semantic_payload": {"environment": "IND", "reward": "additive"},
        "scientific_payload": {"expected_reward": 1.5, "information_nats": 0.75},
    }
    workstation = ArtifactEnvelope(
        **common,
        runtime_metadata={"hostname": "workstation", "wall_seconds": 10.0},
    )
    cluster = ArtifactEnvelope(
        **common,
        runtime_metadata={"hostname": "cluster", "wall_seconds": 2.0},
    )

    assert workstation.semantic_hash == cluster.semantic_hash
    assert workstation.scientific_payload_hash == cluster.scientific_payload_hash
    assert workstation.canonical_bytes() != cluster.canonical_bytes()
    assert b'"semantic_payload":{"environment":"IND"' in workstation.canonical_bytes()


def test_scientific_or_semantic_changes_move_the_correct_hashes() -> None:
    baseline = ArtifactEnvelope(
        kind="ledger",
        schema_version=1,
        semantic_payload={"latent_order": ["x", "y"]},
        scientific_payload={"total_nats": 1.0},
    )
    changed_value = ArtifactEnvelope(
        kind="ledger",
        schema_version=1,
        semantic_payload={"latent_order": ["x", "y"]},
        scientific_payload={"total_nats": 2.0},
    )
    changed_order = ArtifactEnvelope(
        kind="ledger",
        schema_version=1,
        semantic_payload={"latent_order": ["y", "x"]},
        scientific_payload={"total_nats": 1.0},
    )

    assert baseline.semantic_hash == changed_value.semantic_hash
    assert baseline.scientific_payload_hash != changed_value.scientific_payload_hash
    assert baseline.semantic_hash != changed_order.semantic_hash


def test_artifact_compatibility_returns_typed_diagnostics() -> None:
    left = ArtifactEnvelope(
        kind="frontier",
        schema_version=1,
        semantic_payload={"problem": "a"},
        scientific_payload={"lower": 0.0},
    )
    right = ArtifactEnvelope(
        kind="frontier",
        schema_version=1,
        semantic_payload={"problem": "b"},
        scientific_payload={"lower": 0.0},
    )

    report = left.validate_compatible(right)

    assert not report.valid
    assert tuple(item.code for item in report.diagnostics) == (
        "INCOMPATIBLE_SEMANTIC_HASH",
    )


def test_checkpoint_schemas_keep_run_and_population_information_distinct() -> None:
    novelty = NoveltyMetrics(0.1, 0.0, 1.0, 0.0, 0.2, 0.1, 0.0)
    support = SupportMetrics(1, 1, 0, 2)
    witness = DeploymentAction(((1, 1),))
    run = RunCheckpoint(
        schema_version=1,
        round_index=4,
        reward_samples=(1.0, 0.5),
        realized_information=InformationBreakdown(
            math.log(2), 0.0, 0.0, 0.0, 0.0, math.log(2)
        ),
        deployment_witness=witness,
        deployment_semantic_hash=semantic_hash(witness),
        deployment_seed=b"seed",
        novelty=novelty,
        support=support,
        compute=ComputeMetrics(2, 4, 4, 1, 1),
        target_size=3,
    )
    population = PopulationInformationEstimate(0.5, 0.0, 0.5, 0.0, 1.0, 10)
    uncertainty_pair = ["reward", MetricInterval(0.05, 0.1, "reward")]
    semantic_pair = ["frontier", "abc"]
    estimate = CheckpointEstimate(
        schema_version=1,
        reward=RewardMetrics(0.75, 3.0, 0.1),
        bit_equivalent=MetricInterval(0.4, 0.5, "nats"),
        population_information=population,
        efficiency=EfficiencyMetric(
            MetricInterval(0.4, 0.5, "ratio"), ValidationReport()
        ),
        novelty=novelty,
        support=support,
        frontier_regret=FrontierRegretMetrics(
            MetricInterval(0.1, 0.2, "reward"),
            MetricInterval(0.0, 0.1, "reward"),
        ),
        uncertainty=(uncertainty_pair,),  # type: ignore[arg-type]
        semantic_hashes=(semantic_pair,),  # type: ignore[arg-type]
    )
    uncertainty_pair[0] = "changed"
    semantic_pair[1] = "changed"

    assert run.envelope(semantic_payload={"run": 1}).kind == "run_checkpoint"
    assert estimate.envelope(semantic_payload={"cell": 1}).kind == (
        "checkpoint_estimate"
    )
    assert estimate.validate().valid
    assert estimate.uncertainty[0][0] == "reward"
    assert estimate.semantic_hashes == (("frontier", "abc"),)

    with pytest.raises(TypeError, match="InformationBreakdown"):
        RunCheckpoint(
            1,
            4,
            (1.0,),
            population,  # type: ignore[arg-type]
            witness,
            semantic_hash(witness),
            9,
            novelty,
            support,
            ComputeMetrics(1, 1, 1, 0, 1),
            3,
        )

    with pytest.raises(ValueError, match="target_size"):
        RunCheckpoint(
            schema_version=1,
            round_index=4,
            reward_samples=(1.0,),
            realized_information=run.realized_information,
            deployment_witness=witness,
            deployment_semantic_hash=semantic_hash(witness),
            deployment_seed=9,
            novelty=novelty,
            support=support,
            compute=ComputeMetrics(1, 1, 1, 0, 1),
            target_size=4,
        )


def test_frozen_wrappers_copy_nested_mutable_values() -> None:
    caller_owned = [1]
    frozen = FrozenArray((caller_owned,))
    before = canonical_json_bytes(frozen)

    caller_owned.append(2)

    assert canonical_json_bytes(frozen) == before
    with pytest.raises(ValueError, match="float token"):
        FrozenFloat("not-a-float")


def test_byte_seed_has_deterministic_tagged_serialization() -> None:
    assert canonical_json_bytes(b"\x00\xff") == b'{"$bytes":"00ff"}'


def test_scientific_hash_is_independent_of_process_hash_seed() -> None:
    code = (
        "from infinite_rulebook.artifacts import scientific_payload_hash;"
        "print(scientific_payload_hash({key:key for key in {'a','b','c','d'}}))"
    )
    outputs = []
    for seed in ("1", "98765"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                env=environment,
                text=True,
            ).strip()
        )

    assert outputs[0] == outputs[1]


def test_nested_metric_pairs_are_copied_into_immutable_tuples() -> None:
    quantile = [0.1, -0.5]
    reward = RewardMetrics(1.0, 2.0, 0.0, (quantile,))  # type: ignore[arg-type]

    quantile[1] = 99.0

    assert reward.lower_quantiles == ((0.1, -0.5),)
