from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

from infinite_rulebook.agents import distractor_targets, useful_targets
from infinite_rulebook.orchestration.artifacts import (
    read_artifact,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentKind,
    load_experiment_config,
)
from infinite_rulebook.orchestration.records import validate_checkpoint_record
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.seeds import SeedBank
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import (
    ExactSymbolicAdapter,
    _execute_round,
)


def test_full_smoke_pilot_is_valid_and_restart_deterministic(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("configs/pilot-smoke.json")
    runner = SweepRunner(RunExecutor(tmp_path, ExactSymbolicAdapter))
    results = runner.run(config, max_workers=4)

    assert len(results) == 24
    assert all(result.complete for result in results)
    assert len({result.run_hash for result in results}) == 24
    observed = set()
    typed_checkpoints = 0
    alea_cosmetics = []
    trivia_information = []
    prediction_errors = {}
    for result in results:
        artifacts = validate_artifact_tree(result.path)
        resolved = read_artifact(result.path / "config.resolved.json")
        cell = resolved.payload["cell"]
        environment = cell["environment"]["kind"]
        agent = cell["agent"]["kind"]
        observed.add((environment, agent))
        assert set(resolved.semantic_hashes) == {
            "environment",
            "reward",
            "action",
            "feedback",
            "frontier",
        }
        checkpoints = [
            artifact
            for artifact in artifacts
            if artifact.artifact_type == "run-checkpoint"
        ]
        typed_checkpoints += len(checkpoints)
        for checkpoint in checkpoints:
            records = checkpoint.payload["result"]["scientific_records"]
            assert records["run_checkpoint"]["scientific_hash"]
            assert (
                records["population_status"]
                == "Population CheckpointEstimate, efficiency, and frontier regret "
                "are not emitted until complete histories are pooled."
            )
            validate_checkpoint_record(records)
        final = read_artifact(result.path / "checkpoints/00000004.json")
        capabilities = final.payload["result"]["agent_capabilities"]
        assert capabilities["knows_relevance_mask"]
        assert capabilities["knows_coordinate_factorization"]
        prediction_errors[(environment, agent)] = final.payload["result"]["novelty"][
            "observation_prediction_error"
        ]
        if environment == EnvironmentKind.ALEA.value:
            alea_cosmetics.append(
                final.payload["result"]["novelty"]["aleatoric_observation_novelty"]
            )
            assert (
                final.payload["result"]["information"]["persistent_distractor_nats"]
                == 0.0
            )
        if (
            environment == EnvironmentKind.TRIVIA.value
            and agent == AgentKind.TOTAL_INFORMATION.value
        ):
            trivia_information.append(
                final.payload["result"]["information"]["persistent_distractor_nats"]
            )

    assert observed == set(
        product(
            (kind.value for kind in EnvironmentKind),
            (kind.value for kind in AgentKind),
        )
    )
    assert typed_checkpoints == 72
    assert all(value > 0.0 for value in alea_cosmetics)
    assert trivia_information and trivia_information[0] > 0.0
    for agent in AgentKind:
        assert (
            prediction_errors[(EnvironmentKind.ALEA.value, agent.value)]
            > (prediction_errors[(EnvironmentKind.IND.value, agent.value)])
        )
    frontiers = tuple(path for path in (tmp_path / "_frontiers").iterdir())
    assert len(frontiers) == 4
    for frontier in frontiers:
        validate_artifact_tree(frontier)

    restarted = runner.run(config, max_workers=4)
    assert [
        (result.run_hash, result.scientific_content_hash) for result in restarted
    ] == [(result.run_hash, result.scientific_content_hash) for result in results]


def test_failed_query_preflight_does_not_poison_pending_agent_state() -> None:
    config = load_experiment_config("configs/pilot-smoke.json")
    cell = next(
        cell
        for cell in config.cells()
        if cell.environment.kind is EnvironmentKind.IND
        and cell.agent.kind is AgentKind.REWARD
    )
    state = ExactSymbolicAdapter().initial_state(
        cell,
        SeedBank(config.master_seed).for_cell(cell),
    )
    state.candidates = distractor_targets(1)
    with pytest.raises(ValueError, match="trivia"):
        _execute_round(state)

    state.candidates = useful_targets(1)
    trace, _ = _execute_round(state)
    assert trace.action.round_index == 0
    assert state.agent.completed_rounds == 1
