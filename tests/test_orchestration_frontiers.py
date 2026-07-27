from __future__ import annotations

from dataclasses import replace

import pytest

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.frontier import (
    enumerate_independent_rulebook,
    enumerate_mixed_rulebook,
)
from infinite_rulebook.metrics import FrontierCurve
from infinite_rulebook.orchestration.artifacts import (
    ArtifactStore,
    validate_artifact_tree,
    write_frontier_bundle,
)
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    EnvironmentConfig,
    EnvironmentKind,
    FeedbackConfig,
    RewardConfig,
    RunCell,
    SolverConfig,
)
from infinite_rulebook.orchestration.frontiers import build_pilot_frontier


def _cell(kind: EnvironmentKind) -> RunCell:
    return RunCell(
        environment=EnvironmentConfig(
            kind=kind,
            projection_size=1,
            core_dimensions=1,
            max_redundant_support=1,
            distractor_dimensions=1,
            public_reward_cap=2.0,
        ),
        feedback=FeedbackConfig(),
        reward=RewardConfig(),
        agent=AgentConfig(AgentKind.REWARD),
        solver=SolverConfig(),
        environment_replica=0,
        algorithm_replica=0,
    )


@pytest.mark.parametrize("kind", tuple(EnvironmentKind))
def test_builds_valid_typed_and_persisted_frontier(tmp_path, kind) -> None:
    result = build_pilot_frontier(_cell(kind))

    assert isinstance(result.curve, FrontierCurve)
    assert len(result.curve.points) == 3
    assert all(
        point["requested_target_reward"] >= point["effective_target_reward"]
        for point in result.bundle["curve"]["points"]
    )
    store = ArtifactStore(tmp_path / kind.value)
    write_frontier_bundle(
        store,
        {"frontier": "f" * 64},
        curve=result.bundle["curve"],
        witnesses=result.bundle["witnesses"],
        certificates=result.bundle["certificates"],
        diagnostics=result.bundle["diagnostics"],
    )
    validate_artifact_tree(
        store.path,
        expected_semantic_hashes={"frontier": "f" * 64},
    )


def test_alea_and_trivia_reuse_canonical_ind_problem_identity() -> None:
    ind = build_pilot_frontier(_cell(EnvironmentKind.IND))
    alea = build_pilot_frontier(_cell(EnvironmentKind.ALEA))
    trivia = build_pilot_frontier(_cell(EnvironmentKind.TRIVIA))

    assert alea.curve == ind.curve == trivia.curve
    assert (
        alea.bundle["curve"]["problem"]
        == ind.bundle["curve"]["problem"]
        == trivia.bundle["curve"]["problem"]
    )
    assert alea.bundle == ind.bundle == trivia.bundle
    assert ind.bundle["diagnostics"]["control_invariance"] == {
        "canonical_environment": "IND",
        "frontier_problem_reused": True,
        "registered_invariances": [
            "fresh-cosmetic-noise-is-not-persistent-information",
            "reward-irrelevant-trivia-does-not-change-frontier",
        ],
    }


def test_public_c_persists_composite_actions_and_cap() -> None:
    result = build_pilot_frontier(_cell(EnvironmentKind.PUBLIC_C))
    actions = result.bundle["curve"]["problem"]["actions"]

    assert all(set(action) == {"deployment", "public_choice"} for action in actions)
    assert {action["public_choice"] for action in actions} == {0, 1}
    assert result.curve.maximum_reward == 3.0
    assert result.curve.zero_information_reward == 2.0


def test_invariance_diagnostic_does_not_change_trivia_frontier() -> None:
    cell = _cell(EnvironmentKind.TRIVIA)
    one = build_pilot_frontier(cell)
    many = build_pilot_frontier(
        replace(
            cell,
            environment=replace(cell.environment, distractor_dimensions=5),
        )
    )

    assert one.curve == many.curve
    assert (
        one.bundle["curve"]["problem"]["provenance_hash"]
        == many.bundle["curve"]["problem"]["provenance_hash"]
    )
    assert one.bundle == many.bundle


def test_one_rule_mix_reuses_exact_ind_problem_without_spurious_core() -> None:
    mix = build_pilot_frontier(_cell(EnvironmentKind.MIX))
    independent_problem = enumerate_independent_rulebook(1, RewardConfig().to_spec())

    assert mix.curve.semantic_hash == semantic_hash(independent_problem.problem)
    assert mix.curve.maximum_reward == 1.0
    assert mix.bundle["curve"]["problem"]["structural_assumptions"] == "canonical-IND"
    assert {
        index
        for action in mix.bundle["curve"]["problem"]["actions"]
        for index, _prediction in action
    } == {1}


def test_three_rule_mix_matches_exact_odd_even_projection() -> None:
    cell = replace(
        _cell(EnvironmentKind.MIX),
        environment=replace(
            _cell(EnvironmentKind.MIX).environment,
            projection_size=3,
        ),
    )
    mix = build_pilot_frontier(cell)
    exact_problem = enumerate_mixed_rulebook(
        2,
        cell.environment.core_dimensions,
        1,
        cell.environment.max_redundant_support,
        cell.reward.to_spec(),
    )

    assert mix.curve.semantic_hash == semantic_hash(exact_problem.problem)
    assert mix.curve.maximum_reward == 3.0
    assert {
        index
        for action in mix.bundle["curve"]["problem"]["actions"]
        for index, _prediction in action
    } == {1, 2, 3}


def test_red_c_rejects_zero_redundant_support() -> None:
    with pytest.raises(
        ValueError,
        match="RED-C max_redundant_support must be positive",
    ):
        EnvironmentConfig(
            kind=EnvironmentKind.RED_C,
            max_redundant_support=0,
        )
