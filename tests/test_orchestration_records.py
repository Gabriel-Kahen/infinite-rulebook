from __future__ import annotations

import copy
import json
import math

import pytest

from infinite_rulebook.artifacts import RunCheckpoint, semantic_hash
from infinite_rulebook.core import DeploymentAction
from infinite_rulebook.environments import PublicDeploymentAction
from infinite_rulebook.information import InformationBreakdown
from infinite_rulebook.metrics import (
    ComputeMetrics,
    NoveltyMetrics,
    SupportMetrics,
)
from infinite_rulebook.orchestration.records import (
    build_checkpoint_record,
    validate_checkpoint_record,
)


def _inputs() -> dict[str, object]:
    amount = math.log(2.0)
    return {
        "semantic_hashes": {
            "environment": semantic_hash({"environment": "IND"}),
            "reward": semantic_hash({"reward": "exact"}),
            "action": semantic_hash({"action": "finite"}),
            "feedback": semantic_hash({"feedback": "qary", "epsilon": 0.0}),
            "frontier": semantic_hash({"cache_identity": "pilot"}),
        },
        "round_index": 2,
        "reward_sample": 1.0,
        "information": InformationBreakdown(
            amount,
            0.0,
            0.0,
            0.0,
            0.0,
            amount,
        ),
        "deployment": DeploymentAction(((1, 2),)),
        "deployment_seed": b"deployment",
        "novelty": NoveltyMetrics(0.1, 0.0, 1.0, 1.0, 0.2, 0.0, 0.0),
        "support": SupportMetrics(1, 1, 0, 0, 1),
        "compute": ComputeMetrics(2, 2, 2, 0, 1),
    }


def test_record_is_deterministic_and_defers_population_estimands() -> None:
    inputs = _inputs()
    payload = build_checkpoint_record(**inputs)

    assert json.loads(json.dumps(payload)) == payload
    assert build_checkpoint_record(**inputs) == payload
    assert set(payload) == {
        "schema_version",
        "run_checkpoint",
        "population_status",
    }
    assert (
        payload["population_status"]
        == "Population CheckpointEstimate, efficiency, and frontier regret are not "
        "emitted until complete histories are pooled."
    )
    assert isinstance(validate_checkpoint_record(payload), RunCheckpoint)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("run_checkpoint", "summary", "reward_sample"), 0.25),
        (("run_checkpoint", "semantic_hash"), "0" * 64),
        (("run_checkpoint", "scientific_hash"), "f" * 64),
        (("run_checkpoint", "envelope"), ["tampered"]),
    ],
)
def test_record_validation_rejects_tampering(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    payload = build_checkpoint_record(**_inputs())
    tampered = copy.deepcopy(payload)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValueError):
        validate_checkpoint_record(tampered)


def test_record_validation_rejects_nested_runtime_metadata() -> None:
    payload = build_checkpoint_record(**_inputs())
    tampered = copy.deepcopy(payload)
    envelope = tampered["run_checkpoint"]["envelope"]
    envelope[-1][2][-1][1] = ["m", [["hostname", ["s", "workstation"]]]]

    with pytest.raises(ValueError, match="envelope"):
        validate_checkpoint_record(tampered)


def test_record_validation_rejects_noncanonical_readable_summary() -> None:
    payload = build_checkpoint_record(**_inputs())
    tampered = copy.deepcopy(payload)
    tampered["run_checkpoint"]["summary"]["deployment_seed"] = {
        "type": "bytes",
        "hex": "6465706C6F796D656E74",
    }

    with pytest.raises(ValueError, match="canonical"):
        validate_checkpoint_record(tampered)


def test_public_deployment_is_preserved_in_typed_and_readable_records() -> None:
    inputs = _inputs()
    inputs["deployment"] = PublicDeploymentAction(
        inputs["deployment"],
        public_choice=2,
    )

    payload = build_checkpoint_record(**inputs)
    run = validate_checkpoint_record(payload)

    summary = payload["run_checkpoint"]["summary"]["deployment"]
    assert summary == {
        "type": "public_deployment",
        "entries": [[1, 2]],
        "public_choice": 2,
    }
    assert isinstance(run.deployment_witness, PublicDeploymentAction)
    assert run.deployment_witness.public_choice == 2
    json.dumps(payload, allow_nan=False)
