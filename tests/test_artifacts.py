"""Deterministic artifact serialization and scientific hash boundaries."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace

import pytest

from infinite_rulebook.artifacts import (
    ArtifactEnvelope,
    CheckpointEstimate,
    FrozenArray,
    FrozenFloat,
    RunCheckpoint,
    ScientificSemantics,
    canonical_json_bytes,
    scientific_payload_hash,
    semantic_hash,
)
from infinite_rulebook.core import CounterRNG, DeploymentAction
from infinite_rulebook.environments import PublicDeploymentAction
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
    useful_information_efficiency,
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
    assert canonical_json_bytes(math.inf) == b'["f","+inf"]'

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
    assert b"semantic_payload" in workstation.canonical_bytes()


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
    semantics = ScientificSemantics(
        environment=semantic_hash({"environment": 1}),
        reward=semantic_hash({"reward": 1}),
        action=semantic_hash({"action": 1}),
        feedback=semantic_hash({"feedback": 1}),
        frontier=semantic_hash({"frontier": 1}),
    )
    run = RunCheckpoint(
        schema_version=1,
        semantic_hashes=semantics,
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
    estimate = CheckpointEstimate(
        schema_version=1,
        semantic_hashes=semantics,
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
    )
    uncertainty_pair[0] = "changed"

    assert run.envelope().kind == "run_checkpoint"
    assert estimate.envelope().kind == "checkpoint_estimate"
    assert estimate.validate().valid
    assert estimate.uncertainty[0][0] == "reward"
    assert estimate.semantic_hashes.frontier == semantics.frontier
    mismatched = replace(
        estimate,
        efficiency=EfficiencyMetric(
            MetricInterval(0.3, 0.5, "ratio"), ValidationReport()
        ),
    )
    assert {item.code for item in mismatched.validate().diagnostics} == {
        "EFFICIENCY_VALUE_MISMATCH"
    }

    zero_information = PopulationInformationEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 10)
    undefined = useful_information_efficiency(
        MetricInterval(0.0, 0.0, "nats"),
        zero_information,
        complete_history_manifest=True,
    )
    missing_ratio = replace(estimate, efficiency=undefined)
    assert "EFFICIENCY_VALUE_MISMATCH" in {
        item.code for item in missing_ratio.validate().diagnostics
    }
    inconsistent_zero = replace(
        estimate,
        population_information=zero_information,
        efficiency=undefined,
    )
    assert not inconsistent_zero.validate().valid
    assert "EFFICIENCY_INCONSISTENT" in {
        item.code for item in inconsistent_zero.validate().diagnostics
    }
    within_tolerance = replace(
        estimate,
        efficiency=EfficiencyMetric(
            MetricInterval(0.4 + 5e-13, 0.5 - 5e-13, "ratio"),
            ValidationReport(),
            tolerance=1e-12,
        ),
    )
    assert within_tolerance.validate().valid

    empty = DeploymentAction()
    with pytest.raises(ValueError, match="witness support"):
        replace(
            run,
            deployment_witness=empty,
            deployment_semantic_hash=semantic_hash(empty),
        )

    with pytest.raises(TypeError, match="InformationBreakdown"):
        RunCheckpoint(
            schema_version=1,
            semantic_hashes=semantics,
            round_index=4,
            reward_samples=(1.0,),
            realized_information=population,  # type: ignore[arg-type]
            deployment_witness=witness,
            deployment_semantic_hash=semantic_hash(witness),
            deployment_seed=9,
            novelty=novelty,
            support=support,
            compute=ComputeMetrics(1, 1, 1, 0, 1),
            target_size=3,
        )

    with pytest.raises(ValueError, match="target_size"):
        RunCheckpoint(
            schema_version=1,
            semantic_hashes=semantics,
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

    with pytest.raises(ValueError, match="unique"):
        replace(
            estimate,
            uncertainty=(
                ("same", MetricInterval(0.0, 0.1, "reward")),
                ("same", MetricInterval(0.1, 0.2, "reward")),
            ),
        )

    with pytest.raises(ValueError, match="nonnegative"):
        replace(
            estimate,
            bit_equivalent=MetricInterval(-1.0, 0.0, "nats"),
        )

    changed_semantics = replace(
        semantics,
        environment=semantic_hash({"environment": 2}),
    )
    changed_estimate = replace(estimate, semantic_hashes=changed_semantics)
    assert (
        not estimate.envelope().validate_compatible(changed_estimate.envelope()).valid
    )

    changed_feedback = replace(
        semantics,
        feedback=semantic_hash({"feedback": 2}),
    )
    assert (
        not run.envelope()
        .validate_compatible(replace(run, semantic_hashes=changed_feedback).envelope())
        .valid
    )

    public_witness = PublicDeploymentAction(witness, public_choice=2)
    public_run = replace(
        run,
        deployment_witness=public_witness,
        deployment_semantic_hash=semantic_hash(public_witness),
    )
    assert public_run.support.deployment_support == len(public_witness.deployment)
    assert public_run.envelope().semantic_hash == run.envelope().semantic_hash
    assert public_run.envelope().scientific_payload_hash != (
        run.envelope().scientific_payload_hash
    )

    with pytest.raises(ValueError, match="SHA-256"):
        ScientificSemantics(
            "abc",
            semantics.reward,
            semantics.action,
            semantics.feedback,
            semantics.frontier,
        )
    with pytest.raises(TypeError, match="ScientificSemantics"):
        replace(estimate, semantic_hashes=())


def test_frozen_wrappers_copy_nested_mutable_values() -> None:
    caller_owned = [1]
    frozen = FrozenArray((caller_owned,))
    before = canonical_json_bytes(frozen)

    caller_owned.append(2)

    assert canonical_json_bytes(frozen) == before
    with pytest.raises(ValueError, match="float token"):
        FrozenFloat("not-a-float")


def test_byte_seed_has_deterministic_tagged_serialization() -> None:
    assert canonical_json_bytes(b"\x00\xff") == b'["y","00ff"]'


def test_unicode_seed_semantics_match_canonical_hashing() -> None:
    composed = "é"
    decomposed = "e\u0301"

    assert CounterRNG(composed).digest(1) == CounterRNG(decomposed).digest(1)
    assert semantic_hash(composed) == semantic_hash(decomposed)


def test_tagged_serialization_is_injective_across_payload_types() -> None:
    float_token_map = {"$float": "0x1.0000000000000p+0"}
    byte_token_map = {"$bytes": "00ff"}

    assert semantic_hash(1.0) != semantic_hash(float_token_map)
    assert semantic_hash(b"\x00\xff") != semantic_hash(byte_token_map)
    assert semantic_hash(witness := DeploymentAction()) != semantic_hash(
        {
            "$type": f"{type(witness).__module__}.{type(witness).__qualname__}",
            "entries": (),
        }
    )


def test_dataclass_comparison_policy_does_not_change_hash_boundaries() -> None:
    @dataclass(frozen=True)
    class Payload:
        value: int
        evidence: str = field(compare=False)

    @dataclass(frozen=True)
    class PayloadWithCache:
        value: int
        cache: str = field(metadata={"artifact_exclude": True})

    assert semantic_hash(Payload(1, "A")) != semantic_hash(Payload(1, "B"))
    assert semantic_hash(PayloadWithCache(1, "A")) == semantic_hash(
        PayloadWithCache(1, "B")
    )


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
