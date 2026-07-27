"""Reward-information frontier implementations."""

from infinite_rulebook.frontier.blahut_arimoto import (
    LagrangianSolution,
    SupportedInformationSolution,
    solve_lagrangian,
    solve_supported_minimum_information,
)
from infinite_rulebook.frontier.controls import (
    EnumeratedPublicProblem,
    PublicCFrontier,
    PublicUWitness,
    alea_frontier_problem,
    alea_persistent_information_nats,
    augment_with_independent_trivia,
    augment_with_public_c,
    enumerate_public_c_rulebook,
    enumerate_trivia_rulebook,
    public_c_bit_equivalent,
    public_u_bit_equivalent,
    public_u_witness,
    trivia_invariant_bit_equivalent,
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
    "EnumeratedPublicProblem",
    "EnumeratedRulebookProblem",
    "FiniteDecisionProblem",
    "FrontierInversion",
    "FrontierSolution",
    "LagrangianSolution",
    "OneCoordinateFrontier",
    "PublicCFrontier",
    "PublicUWitness",
    "RareBurstWitness",
    "SupportedInformationSolution",
    "TensorizedFrontier",
    "alea_frontier_problem",
    "alea_persistent_information_nats",
    "augment_with_independent_trivia",
    "augment_with_public_c",
    "capped_redundant_information_upper_bound",
    "enumerate_independent_rulebook",
    "enumerate_mixed_rulebook",
    "enumerate_public_c_rulebook",
    "enumerate_redundant_rulebook",
    "enumerate_trivia_rulebook",
    "infinite_bit_equivalent",
    "invert_frontier",
    "one_coordinate_problem",
    "public_c_bit_equivalent",
    "public_u_bit_equivalent",
    "public_u_witness",
    "rare_burst_sequence",
    "solve_frontier",
    "solve_lagrangian",
    "solve_supported_minimum_information",
    "trivia_invariant_bit_equivalent",
    "unrestricted_redundant_bit_equivalent",
]
