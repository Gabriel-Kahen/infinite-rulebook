"""Certified finite frontier bundles for the bounded symbolic pilot."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.controls import (
    PublicBonusSchedule,
    PublicDeploymentAction,
)
from infinite_rulebook.frontier import (
    EnumeratedPublicProblem,
    EnumeratedRulebookProblem,
    enumerate_independent_rulebook,
    enumerate_mixed_rulebook,
    enumerate_public_c_rulebook,
    enumerate_redundant_rulebook,
    solve_frontier,
    solve_lagrangian,
)
from infinite_rulebook.metrics import (
    FrontierCurve,
    FrontierPoint,
    UpperEnvelopeCertificate,
)
from infinite_rulebook.orchestration.config import EnvironmentKind, RunCell
from infinite_rulebook.orchestration.hashing import scientific_hash


@dataclass(frozen=True, slots=True)
class PilotFrontier:
    """A typed frontier and its immutable-artifact representation."""

    curve: FrontierCurve
    bundle: dict[str, Any]


def _enumerated_problem(
    cell: RunCell,
) -> EnumeratedRulebookProblem | EnumeratedPublicProblem:
    environment = cell.environment
    spec = cell.reward.to_spec()
    if environment.kind in {
        EnvironmentKind.IND,
        EnvironmentKind.ALEA,
        EnvironmentKind.TRIVIA,
    }:
        # ALEA noise is fresh and TRIVIA is reward-irrelevant. Both controls
        # therefore use the canonical IND decision problem, not an augmented
        # source that could accidentally charge for irrelevant observations.
        return enumerate_independent_rulebook(environment.projection_size, spec)
    if environment.kind is EnvironmentKind.RED_C:
        return enumerate_redundant_rulebook(
            environment.core_dimensions,
            environment.projection_size,
            environment.max_redundant_support,
            spec,
        )
    if environment.kind is EnvironmentKind.MIX:
        primitive_dimensions = max(1, environment.projection_size // 2)
        derived_rules = max(1, environment.projection_size - primitive_dimensions)
        return enumerate_mixed_rulebook(
            primitive_dimensions,
            environment.core_dimensions,
            derived_rules,
            environment.max_redundant_support,
            spec,
        )
    if environment.kind is EnvironmentKind.PUBLIC_C:
        return enumerate_public_c_rulebook(
            environment.projection_size,
            PublicBonusSchedule((0.0, environment.public_reward_cap)),
            spec,
        )
    raise ValueError(f"unsupported pilot environment: {environment.kind}")


def _serialized_action(
    action: DeploymentAction | PublicDeploymentAction,
) -> list[list[int]] | dict[str, Any]:
    if isinstance(action, PublicDeploymentAction):
        return {
            "deployment": [list(entry) for entry in action.deployment.entries],
            "public_choice": action.public_choice,
        }
    return [list(entry) for entry in action.entries]


def _invariance_diagnostic(cell: RunCell) -> dict[str, Any] | None:
    environment = cell.environment
    if environment.kind in {
        EnvironmentKind.IND,
        EnvironmentKind.ALEA,
        EnvironmentKind.TRIVIA,
    }:
        return {
            "canonical_environment": EnvironmentKind.IND.value,
            "frontier_problem_reused": True,
            "registered_invariances": [
                "fresh-cosmetic-noise-is-not-persistent-information",
                "reward-irrelevant-trivia-does-not-change-frontier",
            ],
        }
    return None


def _extended_number(value: float) -> float | str:
    if math.isinf(value):
        return "infinity" if value > 0 else "-infinity"
    return value


def build_pilot_frontier(cell: RunCell) -> PilotFrontier:
    """Solve and serialize the zero, midpoint, and maximum pilot frontier."""

    enumerated = _enumerated_problem(cell)
    problem = enumerated.problem
    targets = (
        problem.zero_information_reward,
        (problem.zero_information_reward + problem.maximum_reward) / 2.0,
        problem.maximum_reward,
    )
    solutions = tuple(
        solve_frontier(
            problem,
            target,
            tolerance=cell.solver.tolerance,
            bound_tolerance=cell.solver.bound_tolerance,
            lagrangian_tolerance=cell.solver.lagrangian_tolerance,
            max_iterations=cell.solver.max_iterations,
            lagrangian_max_iterations=cell.solver.lagrangian_max_iterations,
        )
        for target in targets
    )
    points = tuple(
        FrontierPoint.from_frontier_solution(problem, solution)
        for solution in solutions
    )
    curve = FrontierCurve(
        points=points,
        zero_information_reward=problem.zero_information_reward,
        maximum_reward=problem.maximum_reward,
        semantic_hash=semantic_hash(problem),
        upper_certificate=UpperEnvelopeCertificate.WITNESS_MIXTURE,
    )

    persisted_points: list[dict[str, Any]] = []
    raw_curve: list[dict[str, Any]] = []
    witnesses: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    point_diagnostics: list[dict[str, Any]] = []
    for index, (point, solution) in enumerate(zip(points, solutions, strict=True)):
        if solution.witness is None:
            raise ValueError("pilot frontier target unexpectedly infeasible")
        name = f"point-{index:03d}"
        persisted_point = {
            "target_reward": solution.effective_target_reward,
            "requested_target_reward": solution.target_reward,
            "effective_target_reward": solution.effective_target_reward,
            "lower_information": point.information.lower,
            "upper_information": point.information.upper,
            "units": point.information.units,
            "classification": "certified-interval",
        }
        persisted_points.append(persisted_point)
        raw_curve.append(
            {
                **persisted_point,
                "duality_gap": solution.duality_gap,
                "solver_iterations": solution.iterations,
                "solver_converged": solution.converged,
            }
        )
        witnesses[name] = {
            "channel": [list(row) for row in solution.witness.channel],
            "action_marginal": list(solution.witness.action_marginal),
            "expected_reward": solution.witness.expected_reward,
            "mutual_information": solution.witness.mutual_information,
            "witness_hash": point.upper_witness.witness_hash,
        }
        dual_objective = None
        if math.isfinite(solution.dual_beta):
            dual_objective = solve_lagrangian(
                problem,
                solution.dual_beta,
                tolerance=cell.solver.lagrangian_tolerance,
                max_iterations=cell.solver.lagrangian_max_iterations,
            ).objective_lower_bound
        certificates[name] = {
            "target_reward": solution.effective_target_reward,
            "requested_target_reward": solution.target_reward,
            "effective_target_reward": solution.effective_target_reward,
            "dual_beta": _extended_number(solution.dual_beta),
            "dual_objective_lower_bound": dual_objective,
            "dual_action_marginal": (
                None
                if solution.lower_certificate_marginal is None
                else list(solution.lower_certificate_marginal)
            ),
            "supported_actions": (
                None
                if solution.lower_certificate_supports is None
                else [list(support) for support in solution.lower_certificate_supports]
            ),
            "lower_bound": solution.lower_bound,
            "upper_bound": solution.upper_bound,
            "duality_gap": _extended_number(solution.duality_gap),
            "certificate_hash": point.lower_certificate.certificate_hash,
            "method": point.lower_certificate.method.value,
        }
        point_diagnostics.append(
            {
                "name": name,
                "requested_target_reward": solution.target_reward,
                "effective_target_reward": solution.effective_target_reward,
                "iterations": solution.iterations,
                "converged": solution.converged,
            }
        )

    problem_payload = {
        "prior": list(problem.prior),
        "reward_matrix": [list(row) for row in problem.rewards],
        "actions": [_serialized_action(action) for action in enumerated.actions],
        "units": "nats",
        "feasible_reward_range": [
            problem.zero_information_reward,
            problem.maximum_reward,
        ],
        "structural_assumptions": (
            "canonical-IND"
            if cell.environment.kind
            in {EnvironmentKind.IND, EnvironmentKind.ALEA, EnvironmentKind.TRIVIA}
            else cell.environment.kind.value
        ),
    }
    problem_payload["provenance_hash"] = scientific_hash(
        problem_payload,
        domain="frontier-problem",
    )
    diagnostics: dict[str, Any] = {
        "solver": cell.solver.contract_version,
        "solver_settings": asdict(cell.solver),
        "confirmatory_frozen": False,
        "points": point_diagnostics,
    }
    invariance = _invariance_diagnostic(cell)
    if invariance is not None:
        diagnostics["control_invariance"] = invariance
    bundle = {
        "curve": {
            "problem": problem_payload,
            "problem_semantic_hash": curve.semantic_hash,
            "zero_information_reward": curve.zero_information_reward,
            "maximum_reward": curve.maximum_reward,
            "upper_certificate": curve.upper_certificate.value,
            "points": persisted_points,
            "raw_curve": raw_curve,
        },
        "witnesses": witnesses,
        "certificates": certificates,
        "diagnostics": diagnostics,
    }
    return PilotFrontier(curve=curve, bundle=bundle)


__all__ = ["PilotFrontier", "build_pilot_frontier"]
