from __future__ import annotations

import json
import math

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.core import DeploymentAction
from infinite_rulebook.environments import PublicDeploymentAction
from infinite_rulebook.frontier import one_coordinate_problem, solve_frontier
from infinite_rulebook.information import InformationBreakdown
from infinite_rulebook.metrics import (
    ComputeMetrics,
    FrontierCurve,
    FrontierPoint,
    NoveltyMetrics,
    SupportMetrics,
    UpperEnvelopeCertificate,
)
from infinite_rulebook.orchestration.records import build_checkpoint_record


def _curve() -> FrontierCurve:
    problem = one_coordinate_problem(q=2)
    points = tuple(
        FrontierPoint.from_frontier_solution(
            problem,
            solve_frontier(problem, reward),
        )
        for reward in (0.0, 1.0)
    )
    return FrontierCurve(
        points=points,
        zero_information_reward=0.0,
        maximum_reward=1.0,
        semantic_hash=semantic_hash(problem),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )


def _inputs() -> dict[str, object]:
    curve = _curve()
    amount = math.log(2.0)
    return {
        "semantic_hashes": {
            "environment": semantic_hash({"environment": "IND"}),
            "reward": semantic_hash({"reward": "exact"}),
            "action": semantic_hash({"action": "finite"}),
            "feedback": semantic_hash({"feedback": "qary", "epsilon": 0.0}),
            "frontier": semantic_hash(
                {"cache_identity": curve.semantic_hash, "solver": "pilot"}
            ),
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
        "frontier": curve,
    }


def test_record_is_deterministic_and_excludes_runtime_from_scientific_hashes() -> None:
    inputs = _inputs()
    workstation = build_checkpoint_record(
        **inputs,
        runtime_metadata={"hostname": "workstation", "elapsed_seconds": 9.0},
    )
    cluster = build_checkpoint_record(
        **inputs,
        runtime_metadata={"hostname": "cluster", "elapsed_seconds": 1.0},
    )

    assert json.loads(json.dumps(workstation)) == workstation
    for kind in ("run_checkpoint", "checkpoint_estimate"):
        assert workstation[kind]["semantic_hash"] == cluster[kind]["semantic_hash"]
        assert workstation[kind]["scientific_hash"] == cluster[kind]["scientific_hash"]
        assert workstation[kind]["envelope"] != cluster[kind]["envelope"]
    repeated = build_checkpoint_record(
        **inputs,
        runtime_metadata={"hostname": "workstation", "elapsed_seconds": 9.0},
    )
    assert repeated == workstation
    assert (
        workstation["checkpoint_estimate"]["summary"]["population_information"][
            "run_count"
        ]
        == 1
    )


def test_public_deployment_is_preserved_in_typed_and_readable_records() -> None:
    inputs = _inputs()
    inputs["deployment"] = PublicDeploymentAction(
        inputs["deployment"],
        public_choice=2,
    )

    payload = build_checkpoint_record(**inputs)

    summary = payload["run_checkpoint"]["summary"]["deployment"]
    assert summary == {
        "type": "public_deployment",
        "entries": [[1, 2]],
        "public_choice": 2,
    }
    json.dumps(payload, allow_nan=False)
