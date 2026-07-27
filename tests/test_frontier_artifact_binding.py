from __future__ import annotations

import copy
from pathlib import Path

import pytest

from infinite_rulebook.orchestration.artifacts import (
    ArtifactStore,
    ScientificArtifactError,
    validate_artifact_tree,
    write_frontier_bundle,
)
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _validate_tamper(
    tmp_path: Path,
    name: str,
    frontier: dict[str, object],
) -> None:
    store = ArtifactStore(tmp_path / name)
    write_frontier_bundle(
        store,
        {"frontier": "f" * 64},
        curve=frontier["curve"],
        witnesses=frontier["witnesses"],
        certificates=frontier["certificates"],
        diagnostics=frontier["diagnostics"],
    )
    with pytest.raises(ScientificArtifactError):
        validate_artifact_tree(store.path)


def test_frontier_validation_binds_typed_certificate_evidence(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    cell = next(cell for cell in config.cells() if cell.environment.kind.value == "IND")
    source = ExactSymbolicAdapter().frontier(cell)

    wrong_problem = copy.deepcopy(source)
    wrong_problem["curve"]["problem_semantic_hash"] = "0" * 64
    _validate_tamper(tmp_path, "problem", wrong_problem)

    wrong_witness = copy.deepcopy(source)
    wrong_witness["witnesses"]["point-001"]["witness_hash"] = "0" * 64
    _validate_tamper(tmp_path, "witness", wrong_witness)

    wrong_dual = copy.deepcopy(source)
    wrong_dual["certificates"]["point-001"]["dual_action_marginal"][0] += 0.01
    _validate_tamper(tmp_path, "dual", wrong_dual)

    wrong_endpoint = copy.deepcopy(source)
    wrong_endpoint["certificates"]["point-002"]["supported_actions"][0] = []
    _validate_tamper(tmp_path, "endpoint", wrong_endpoint)

    wrong_certificate = copy.deepcopy(source)
    wrong_certificate["certificates"]["point-001"]["certificate_hash"] = "0" * 64
    _validate_tamper(tmp_path, "certificate", wrong_certificate)

    wrong_source = copy.deepcopy(source)
    wrong_source["certificates"]["point-001"]["source_solution_hash"] = "0" * 64
    _validate_tamper(tmp_path, "source", wrong_source)


def test_frontier_manifest_rejects_undeclared_extra_artifact(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-foundation.json")
    cell = next(cell for cell in config.cells() if cell.environment.kind.value == "IND")
    frontier = ExactSymbolicAdapter().frontier(cell)
    store = ArtifactStore(tmp_path / "extra")
    semantics = {"frontier": "f" * 64}
    write_frontier_bundle(
        store,
        semantics,
        curve=frontier["curve"],
        witnesses=frontier["witnesses"],
        certificates=frontier["certificates"],
        diagnostics=frontier["diagnostics"],
    )
    validate_artifact_tree(store.path)
    store.write(
        "frontier/undeclared.json",
        "unrelated-diagnostic",
        semantics,
        {"value": 1},
    )

    with pytest.raises(ScientificArtifactError, match="member list"):
        validate_artifact_tree(store.path)
