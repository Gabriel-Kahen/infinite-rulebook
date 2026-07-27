"""Typed Track B records for one symbolic pilot checkpoint."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from numbers import Real
from typing import Any

from infinite_rulebook.artifacts import (
    ArtifactEnvelope,
    CheckpointEstimate,
    RunCheckpoint,
    ScientificSemantics,
    semantic_hash,
)
from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.rng import Seed
from infinite_rulebook.environments.controls import PublicDeploymentAction
from infinite_rulebook.information import InformationBreakdown
from infinite_rulebook.metrics import (
    ComputeMetrics,
    FrontierCurve,
    FrontierRegretMetrics,
    NoveltyMetrics,
    PopulationInformationEstimate,
    RewardMetrics,
    SupportMetrics,
    frontier_regret,
    lookup_bit_equivalent,
    useful_information_efficiency,
)

_SEMANTIC_FIELDS = ("environment", "reward", "action", "feedback", "frontier")


def _tagged_envelope(envelope: ArtifactEnvelope) -> object:
    return json.loads(envelope.canonical_bytes())


def _diagnostics(record: CheckpointEstimate) -> list[dict[str, object]]:
    return [
        {
            "severity": diagnostic.severity.name.lower(),
            "code": diagnostic.code,
            "path": diagnostic.path,
            "message": diagnostic.message,
        }
        for diagnostic in record.validate().diagnostics
    ]


def _seed_summary(seed: Seed) -> dict[str, object]:
    if isinstance(seed, bytes):
        return {"type": "bytes", "hex": seed.hex()}
    if isinstance(seed, bool):
        raise TypeError("deployment_seed must not be a boolean")
    return {"type": type(seed).__name__, "value": seed}


def _deployment_summary(
    witness: DeploymentAction | PublicDeploymentAction,
) -> dict[str, object]:
    deployment = (
        witness.deployment if isinstance(witness, PublicDeploymentAction) else witness
    )
    summary: dict[str, object] = {
        "type": (
            "public_deployment"
            if isinstance(witness, PublicDeploymentAction)
            else "deployment"
        ),
        "entries": [list(entry) for entry in deployment.entries],
    }
    if isinstance(witness, PublicDeploymentAction):
        summary["public_choice"] = witness.public_choice
    return summary


def _record_payload(
    *,
    summary: dict[str, object],
    envelope: ArtifactEnvelope,
) -> dict[str, object]:
    return {
        "summary": json.loads(json.dumps(summary, allow_nan=False, sort_keys=True)),
        "envelope": _tagged_envelope(envelope),
        "semantic_hash": envelope.semantic_hash,
        "scientific_hash": envelope.scientific_payload_hash,
    }


def build_checkpoint_record(
    *,
    semantic_hashes: Mapping[str, str],
    round_index: int,
    reward_sample: Real,
    information: InformationBreakdown,
    deployment: DeploymentAction | PublicDeploymentAction,
    deployment_seed: Seed,
    novelty: NoveltyMetrics,
    support: SupportMetrics,
    compute: ComputeMetrics,
    frontier: FrontierCurve,
    runtime_metadata: object | None = None,
) -> dict[str, object]:
    """Build validated run and one-run population records as JSON-safe data."""

    if set(semantic_hashes) != set(_SEMANTIC_FIELDS):
        raise ValueError(
            "semantic_hashes must contain environment, reward, action, feedback, "
            "and frontier"
        )
    semantics = ScientificSemantics(
        **{name: semantic_hashes[name] for name in _SEMANTIC_FIELDS}
    )
    if not isinstance(information, InformationBreakdown):
        raise TypeError("information must be an InformationBreakdown")
    if information.approximation_residual_nats != 0.0:
        raise ValueError(
            "an exact one-run population estimate cannot include an "
            "approximation residual"
        )
    if not isinstance(frontier, FrontierCurve):
        raise TypeError("frontier must be a FrontierCurve")

    reward = RewardMetrics(
        expected_reward=reward_sample,
        cumulative_reward=reward_sample,
        variance=0.0,
    )
    population = PopulationInformationEstimate(
        reward_relevant_nats=information.reward_relevant_nats,
        shared_core_nats=information.shared_core_nats,
        persistent_distractor_nats=information.persistent_distractor_nats,
        dynamic_state_nats=information.dynamic_state_nats,
        total_nats=information.total_acquired_nats,
        run_count=1,
    )
    bit_equivalent = lookup_bit_equivalent(frontier, reward.expected_reward)
    efficiency = useful_information_efficiency(
        bit_equivalent,
        population,
        complete_history_manifest=True,
    )
    regret = FrontierRegretMetrics(
        full_information=frontier_regret(
            frontier,
            attained_reward=reward.expected_reward,
            information_budget_nats=population.total_nats,
        ),
        relevant_information=frontier_regret(
            frontier,
            attained_reward=reward.expected_reward,
            information_budget_nats=population.relevant_nats,
        ),
    )
    run = RunCheckpoint(
        schema_version=1,
        semantic_hashes=semantics,
        round_index=round_index,
        reward_samples=(reward.expected_reward,),
        realized_information=information,
        deployment_witness=deployment,
        deployment_semantic_hash=semantic_hash(deployment),
        deployment_seed=deployment_seed,
        novelty=novelty,
        support=support,
        compute=compute,
        target_size=support.deployment_support + support.abstentions,
    )
    estimate = CheckpointEstimate(
        schema_version=1,
        semantic_hashes=semantics,
        reward=reward,
        bit_equivalent=bit_equivalent,
        population_information=population,
        efficiency=efficiency,
        novelty=novelty,
        support=support,
        frontier_regret=regret,
        uncertainty=(),
    )
    validation = estimate.validate()

    run_envelope = run.envelope(runtime_metadata=runtime_metadata)
    estimate_envelope = estimate.envelope(runtime_metadata=runtime_metadata)
    common_summary: dict[str, Any] = {
        "semantics": {name: getattr(semantics, name) for name in _SEMANTIC_FIELDS},
        "round_index": round_index,
        "reward_sample": reward.expected_reward,
        "information": dataclasses.asdict(information),
        "deployment": _deployment_summary(deployment),
        "deployment_seed": _seed_summary(deployment_seed),
        "novelty": dataclasses.asdict(novelty),
        "support": dataclasses.asdict(support),
        "compute": dataclasses.asdict(compute),
    }
    estimate_summary = {
        **common_summary,
        "reward": dataclasses.asdict(reward),
        "bit_equivalent": dataclasses.asdict(bit_equivalent),
        "population_information": dataclasses.asdict(population),
        "efficiency": (
            None
            if efficiency.interval is None
            else dataclasses.asdict(efficiency.interval)
        ),
        "frontier_regret": dataclasses.asdict(regret),
        "pilot_population_status": "single-run-diagnostic-not-confirmatory",
        "validation_valid": validation.valid,
        "validation_diagnostics": _diagnostics(estimate),
    }
    payload = {
        "schema_version": 1,
        "run_checkpoint": _record_payload(
            summary=common_summary,
            envelope=run_envelope,
        ),
        "checkpoint_estimate": _record_payload(
            summary=estimate_summary,
            envelope=estimate_envelope,
        ),
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload
