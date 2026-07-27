"""Typed Track B records for one symbolic pilot checkpoint."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from numbers import Real
from typing import Any

from infinite_rulebook.artifacts import (
    ArtifactEnvelope,
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
    NoveltyMetrics,
    SupportMetrics,
)

_SEMANTIC_FIELDS = ("environment", "reward", "action", "feedback", "frontier")
_POPULATION_STATUS = (
    "Population CheckpointEstimate, efficiency, and frontier regret are not "
    "emitted until complete histories are pooled."
)


def _tagged_envelope(envelope: ArtifactEnvelope) -> object:
    return json.loads(envelope.canonical_bytes())


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


def _checkpoint_summary(run: RunCheckpoint) -> dict[str, object]:
    return {
        "semantics": {
            name: getattr(run.semantic_hashes, name) for name in _SEMANTIC_FIELDS
        },
        "round_index": run.round_index,
        "reward_sample": run.reward_samples[0],
        "information": dataclasses.asdict(run.realized_information),
        "deployment": _deployment_summary(run.deployment_witness),
        "deployment_seed": _seed_summary(run.deployment_seed),
        "novelty": dataclasses.asdict(run.novelty),
        "support": dataclasses.asdict(run.support),
        "compute": dataclasses.asdict(run.compute),
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
) -> dict[str, object]:
    """Build a validated per-run record without inventing a population estimand."""

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
            "an exact per-run checkpoint cannot include an approximation residual"
        )
    run = RunCheckpoint(
        schema_version=1,
        semantic_hashes=semantics,
        round_index=round_index,
        reward_samples=(float(reward_sample),),
        realized_information=information,
        deployment_witness=deployment,
        deployment_semantic_hash=semantic_hash(deployment),
        deployment_seed=deployment_seed,
        novelty=novelty,
        support=support,
        compute=compute,
        target_size=support.deployment_support + support.abstentions,
    )
    run_envelope = run.envelope()
    payload = {
        "schema_version": 1,
        "run_checkpoint": _record_payload(
            summary=_checkpoint_summary(run),
            envelope=run_envelope,
        ),
        "population_status": _POPULATION_STATUS,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    validate_checkpoint_record(payload)
    return payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _reconstruct_seed(summary: object) -> Seed:
    tagged = _mapping(summary, "deployment_seed")
    seed_type = tagged.get("type")
    if seed_type == "bytes":
        _require_keys(tagged, {"type", "hex"}, "deployment_seed")
        encoded = tagged["hex"]
        if not isinstance(encoded, str):
            raise TypeError("deployment_seed.hex must be a string")
        try:
            return bytes.fromhex(encoded)
        except ValueError as error:
            raise ValueError("deployment_seed.hex must encode bytes") from error
    _require_keys(tagged, {"type", "value"}, "deployment_seed")
    value = tagged["value"]
    if seed_type == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if seed_type == "str" and isinstance(value, str):
        return value
    raise TypeError("deployment_seed must encode bytes, an integer, or a string")


def _reconstruct_deployment(
    summary: object,
) -> DeploymentAction | PublicDeploymentAction:
    readable = _mapping(summary, "deployment")
    deployment_type = readable.get("type")
    if deployment_type == "deployment":
        _require_keys(readable, {"type", "entries"}, "deployment")
    elif deployment_type == "public_deployment":
        _require_keys(
            readable,
            {"type", "entries", "public_choice"},
            "deployment",
        )
    else:
        raise ValueError("deployment.type is not recognized")
    entries = readable["entries"]
    if not isinstance(entries, list):
        raise TypeError("deployment.entries must be a JSON array")
    deployment = DeploymentAction(tuple(tuple(entry) for entry in entries))
    if deployment_type == "public_deployment":
        return PublicDeploymentAction(deployment, readable["public_choice"])
    return deployment


def _reconstruct_dataclass(
    summary: object,
    name: str,
    record_type: type[Any],
) -> Any:
    return record_type(**dict(_mapping(summary, name)))


def validate_checkpoint_record(payload: object) -> RunCheckpoint:
    """Reconstruct and authenticate a readable per-run checkpoint record."""

    record = _mapping(payload, "checkpoint record")
    _require_keys(
        record,
        {"schema_version", "run_checkpoint", "population_status"},
        "checkpoint record",
    )
    if record["schema_version"] != 1:
        raise ValueError("checkpoint record schema_version must be 1")
    if record["population_status"] != _POPULATION_STATUS:
        raise ValueError("checkpoint record population_status is not recognized")

    checkpoint = _mapping(record["run_checkpoint"], "run_checkpoint")
    _require_keys(
        checkpoint,
        {"summary", "envelope", "semantic_hash", "scientific_hash"},
        "run_checkpoint",
    )
    summary = _mapping(checkpoint["summary"], "run_checkpoint.summary")
    _require_keys(
        summary,
        {
            "semantics",
            "round_index",
            "reward_sample",
            "information",
            "deployment",
            "deployment_seed",
            "novelty",
            "support",
            "compute",
        },
        "run_checkpoint.summary",
    )
    semantic_summary = _mapping(summary["semantics"], "semantics")
    _require_keys(semantic_summary, set(_SEMANTIC_FIELDS), "semantics")
    semantics = ScientificSemantics(
        **{name: semantic_summary[name] for name in _SEMANTIC_FIELDS}
    )
    information = _reconstruct_dataclass(
        summary["information"],
        "information",
        InformationBreakdown,
    )
    novelty = _reconstruct_dataclass(summary["novelty"], "novelty", NoveltyMetrics)
    support = _reconstruct_dataclass(summary["support"], "support", SupportMetrics)
    compute = _reconstruct_dataclass(summary["compute"], "compute", ComputeMetrics)
    deployment = _reconstruct_deployment(summary["deployment"])
    run = RunCheckpoint(
        schema_version=1,
        semantic_hashes=semantics,
        round_index=summary["round_index"],
        reward_samples=(summary["reward_sample"],),
        realized_information=information,
        deployment_witness=deployment,
        deployment_semantic_hash=semantic_hash(deployment),
        deployment_seed=_reconstruct_seed(summary["deployment_seed"]),
        novelty=novelty,
        support=support,
        compute=compute,
        target_size=support.deployment_support + support.abstentions,
    )
    canonical_summary = json.loads(
        json.dumps(_checkpoint_summary(run), allow_nan=False, sort_keys=True)
    )
    if checkpoint["summary"] != canonical_summary:
        raise ValueError("run_checkpoint summary is not canonical")
    envelope = run.envelope()
    if checkpoint["envelope"] != _tagged_envelope(envelope):
        raise ValueError("run_checkpoint envelope does not match its summary")
    if checkpoint["semantic_hash"] != envelope.semantic_hash:
        raise ValueError("run_checkpoint semantic_hash does not match its envelope")
    if checkpoint["scientific_hash"] != envelope.scientific_payload_hash:
        raise ValueError("run_checkpoint scientific_hash does not match its envelope")
    return run
