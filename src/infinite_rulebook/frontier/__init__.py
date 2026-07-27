"""Reward-information frontier implementations."""

from infinite_rulebook.frontier.blahut_arimoto import (
    LagrangianSolution,
    solve_lagrangian,
)
from infinite_rulebook.frontier.finite_problem import (
    ChannelWitness,
    FiniteDecisionProblem,
    one_coordinate_problem,
)
from infinite_rulebook.frontier.inversion import (
    FrontierInversion,
    FrontierSolution,
    invert_frontier,
    solve_frontier,
)
from infinite_rulebook.frontier.one_coordinate import OneCoordinateFrontier
from infinite_rulebook.frontier.redundancy import (
    RareBurstWitness,
    capped_redundant_information_upper_bound,
    enumerate_mixed_rulebook,
    enumerate_redundant_rulebook,
    rare_burst_sequence,
    unrestricted_redundant_bit_equivalent,
)
from infinite_rulebook.frontier.rulebook_problem import (
    EnumeratedRulebookProblem,
    enumerate_independent_rulebook,
)
from infinite_rulebook.frontier.tensorized import (
    TensorizedFrontier,
    infinite_bit_equivalent,
)

__all__ = [
    "ChannelWitness",
    "EnumeratedRulebookProblem",
    "FiniteDecisionProblem",
    "FrontierInversion",
    "FrontierSolution",
    "LagrangianSolution",
    "OneCoordinateFrontier",
    "RareBurstWitness",
    "TensorizedFrontier",
    "capped_redundant_information_upper_bound",
    "enumerate_independent_rulebook",
    "enumerate_mixed_rulebook",
    "enumerate_redundant_rulebook",
    "infinite_bit_equivalent",
    "invert_frontier",
    "one_coordinate_problem",
    "rare_burst_sequence",
    "solve_frontier",
    "solve_lagrangian",
    "unrestricted_redundant_bit_equivalent",
]
