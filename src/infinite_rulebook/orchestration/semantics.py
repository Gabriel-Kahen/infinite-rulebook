"""Explicit semantic identities for configs, actions, and frontiers."""

from __future__ import annotations

from dataclasses import asdict

from infinite_rulebook.orchestration.config import EnvironmentKind, RunCell
from infinite_rulebook.orchestration.hashing import scientific_hash


def semantic_hashes(
    cell: RunCell,
    *,
    analysis_code_hash: str | None = None,
) -> dict[str, str]:
    environment_payload = asdict(cell.environment)
    reward_payload = asdict(cell.reward)
    action_payload = {
        "contract": "canonical-finite-deployment.v1",
        "alphabet": cell.reward.q,
        "abstention": 0,
        "duplicates": "rejected",
        "ordering": "sorted-by-rule-index",
    }
    if cell.environment.kind is EnvironmentKind.PUBLIC_C:
        action_payload = {
            **action_payload,
            "public_contract": "bounded-public-choice.v1",
            "public_rewards": (0.0, cell.environment.public_reward_cap),
        }
    feedback_payload = {
        **asdict(cell.feedback),
        "alphabet": cell.reward.q,
        "semantic_observation_key": "p1-semantic-coordinate.v1",
    }
    environment_hash = scientific_hash(environment_payload, domain="environment")
    reward_hash = scientific_hash(reward_payload, domain="reward")
    action_hash = scientific_hash(action_payload, domain="behavioral-action")
    feedback_hash = scientific_hash(feedback_payload, domain="feedback")
    if cell.environment.kind in {EnvironmentKind.ALEA, EnvironmentKind.TRIVIA}:
        decision_kind = EnvironmentKind.IND
    else:
        decision_kind = cell.environment.kind
    frontier_environment: dict[str, object] = {
        "kind": decision_kind.value,
        "projection_size": cell.environment.projection_size,
    }
    if decision_kind in {EnvironmentKind.RED_C, EnvironmentKind.MIX}:
        frontier_environment.update(
            {
                "core_dimensions": cell.environment.core_dimensions,
                "max_redundant_support": (cell.environment.max_redundant_support),
            }
        )
    if decision_kind is EnvironmentKind.PUBLIC_C:
        frontier_environment["public_reward_cap"] = cell.environment.public_reward_cap
    frontier_payload = {
        "environment": frontier_environment,
        "reward_hash": reward_hash,
        "action_hash": action_hash,
        "solver": asdict(cell.solver),
        "analysis_code_hash": analysis_code_hash or "unspecified",
    }
    return {
        "environment": environment_hash,
        "reward": reward_hash,
        "action": action_hash,
        "feedback": feedback_hash,
        "frontier": scientific_hash(frontier_payload, domain="frontier"),
    }
