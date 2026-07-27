"""Small exact symbolic adapter used to exercise the orchestration foundation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.environments.mixed import MixedRulebook
from infinite_rulebook.environments.redundant import CappedRedundantRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.frontier.inversion import solve_frontier
from infinite_rulebook.frontier.redundancy import (
    enumerate_mixed_rulebook,
    enumerate_redundant_rulebook,
)
from infinite_rulebook.frontier.rulebook_problem import enumerate_independent_rulebook
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentKind,
    RunCell,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.seeds import RunSeeds


def _extended_number(value: float) -> float | str:
    if math.isinf(value):
        return "infinity" if value > 0 else "-infinity"
    return value


@dataclass(slots=True)
class ExactSymbolicAdapter:
    """A bounded P1/reward-directed adapter for harness smoke tests.

    Richer environments and acquisition objectives are intentionally left to
    their owning simulator and agent APIs. This adapter is a real executable
    control, not a placeholder implementation of those missing conditions.
    """

    def _require_supported(self, cell: RunCell) -> None:
        if cell.agent.kind is not AgentKind.REWARD:
            raise ValueError("foundation adapter supports only the reward agent")
        supported = {
            EnvironmentKind.IND,
            EnvironmentKind.RED_C,
            EnvironmentKind.MIX,
        }
        if cell.environment.kind not in supported:
            raise ValueError(
                f"foundation adapter does not implement {cell.environment.kind.value}"
            )

    def _environment(self, cell: RunCell, seeds: RunSeeds) -> Any:
        self._require_supported(cell)
        spec = cell.reward.to_spec()
        if cell.environment.kind is EnvironmentKind.IND:
            return IndependentRulebook(seeds.environment, spec)
        if cell.environment.kind is EnvironmentKind.RED_C:
            return CappedRedundantRulebook(
                seeds.environment,
                cell.environment.core_dimensions,
                spec,
                max_derived_support=cell.environment.max_redundant_support,
            )
        return MixedRulebook(
            seeds.environment,
            cell.environment.core_dimensions,
            cell.environment.max_redundant_support,
            spec,
        )

    def initial_state(self, cell: RunCell, seeds: RunSeeds) -> dict[int, int]:
        self._require_supported(cell)
        del seeds
        return {}

    def training_event(
        self,
        state: dict[int, int],
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> dict[str, Any]:
        del state
        environment = self._environment(cell, seeds)
        channel = QarySymmetricChannel(cell.reward.q, cell.feedback.epsilon)
        if cell.environment.kind is EnvironmentKind.RED_C:
            rule_count = max(1, cell.environment.projection_size)
            indices = tuple(
                1 + ((round_index * cell.feedback.query_budget + ordinal) % rule_count)
                for ordinal in range(cell.feedback.query_budget)
            )
        elif cell.environment.kind is EnvironmentKind.MIX:
            indices = tuple(
                1 + 2 * (round_index * cell.feedback.query_budget + ordinal)
                for ordinal in range(cell.feedback.query_budget)
            )
        else:
            indices = tuple(
                1 + round_index * cell.feedback.query_budget + ordinal
                for ordinal in range(cell.feedback.query_budget)
            )
        observations = []
        for ordinal, index in enumerate(indices):
            key = SemanticObservationKey(
                environment_seed=seeds.query_observation,
                round_index=round_index,
                rule_index=index,
                query_ordinal=ordinal,
                channel="p1-useful",
            )
            observations.append([index, channel.observe(environment.label(index), key)])
        return {"round": round_index, "observations": observations}

    def apply_training_event(
        self,
        state: dict[int, int],
        payload: dict[str, Any],
    ) -> dict[int, int]:
        updated = dict(state)
        for index, observation in payload["observations"]:
            updated[index] = observation
        return updated

    def checkpoint(
        self,
        state: dict[int, int],
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> dict[str, Any]:
        environment = self._environment(cell, seeds)
        entries = sorted(state.items())
        if cell.environment.kind is EnvironmentKind.RED_C:
            entries = entries[: cell.environment.max_redundant_support]
        deployment = DeploymentAction(entries)
        return {
            "expected_reward": environment.evaluate(deployment),
            "deployment": [list(entry) for entry in deployment.entries],
            "support": len(deployment),
            "round": round_index,
            "evaluation": "exact-no-feedback",
            "action_sample_count": 0,
        }

    def state_fingerprint(self, state: dict[int, int]) -> str:
        return scientific_hash(sorted(state.items()), domain="symbolic-agent-state")

    def _problem(self, cell: RunCell) -> Any:
        spec = cell.reward.to_spec()
        environment = cell.environment
        if environment.kind is EnvironmentKind.IND:
            return enumerate_independent_rulebook(environment.projection_size, spec)
        if environment.kind is EnvironmentKind.RED_C:
            return enumerate_redundant_rulebook(
                environment.core_dimensions,
                environment.projection_size,
                environment.max_redundant_support,
                spec,
            )
        primitive_dimensions = max(1, environment.projection_size // 2)
        derived_rules = max(1, environment.projection_size - primitive_dimensions)
        return enumerate_mixed_rulebook(
            primitive_dimensions,
            environment.core_dimensions,
            derived_rules,
            environment.max_redundant_support,
            spec,
        )

    def frontier(self, cell: RunCell) -> dict[str, Any]:
        enumerated = self._problem(cell)
        problem = enumerated.problem
        targets = (
            problem.zero_information_reward,
            (problem.zero_information_reward + problem.maximum_reward) / 2,
            problem.maximum_reward,
        )
        curve = []
        witnesses = {}
        certificates = {}
        diagnostics = {"solver": "certified-finite-frontier", "points": []}
        for index, target in enumerate(targets):
            solution = solve_frontier(problem, target, bound_tolerance=1e-7)
            if solution.witness is None:
                raise ValueError("pilot frontier target unexpectedly infeasible")
            point_name = f"point-{index:03d}"
            curve.append(
                {
                    "target_reward": target,
                    "lower_information": solution.lower_bound,
                    "upper_information": solution.upper_bound,
                    "units": "nats",
                    "classification": "certified-interval",
                }
            )
            witnesses[point_name] = {
                "channel": [list(row) for row in solution.witness.channel],
                "action_marginal": list(solution.witness.action_marginal),
                "expected_reward": solution.witness.expected_reward,
                "mutual_information": solution.witness.mutual_information,
            }
            certificates[point_name] = {
                "dual_beta": _extended_number(solution.dual_beta),
                "lower_bound": solution.lower_bound,
                "upper_bound": solution.upper_bound,
                "duality_gap": _extended_number(solution.duality_gap),
            }
            diagnostics["points"].append(
                {
                    "name": point_name,
                    "iterations": solution.iterations,
                    "converged": solution.converged,
                }
            )
        problem_payload = {
            "prior": list(problem.prior),
            "reward_matrix": [list(row) for row in problem.rewards],
            "actions": [
                [list(entry) for entry in action.entries]
                for action in enumerated.actions
            ],
            "units": "nats",
            "feasible_reward_range": [
                problem.zero_information_reward,
                problem.maximum_reward,
            ],
            "structural_assumptions": cell.environment.kind.value,
        }
        problem_payload["provenance_hash"] = scientific_hash(
            problem_payload, domain="frontier-problem"
        )
        return {
            "curve": {"problem": problem_payload, "points": curve, "raw_curve": curve},
            "witnesses": witnesses,
            "certificates": certificates,
            "diagnostics": diagnostics,
        }
