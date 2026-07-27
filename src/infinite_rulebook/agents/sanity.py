"""The one-observation fresh-coordinate analytic sanity agent."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from infinite_rulebook.core.behavior import DeploymentAction


def _validate_reward_inputs(
    *,
    u: float,
    c: float,
    epsilon: float,
) -> None:
    if not math.isfinite(u) or u <= 0.0:
        raise ValueError("u must be finite and positive")
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("c must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must lie in [0, 1)")


def expected_coordinate_reward(*, u: float, c: float, epsilon: float) -> float:
    """Expected reward from deploying one noisy observed label."""

    _validate_reward_inputs(u=u, c=c, epsilon=epsilon)
    return (u + c) * (1.0 - epsilon) - c


def expected_reward_slope(
    *,
    query_budget: int,
    u: float,
    c: float,
    epsilon: float,
) -> float:
    """Expected reward gained per completed training round."""

    if isinstance(query_budget, bool) or not isinstance(query_budget, int):
        raise TypeError("query_budget must be an integer")
    if query_budget < 1:
        raise ValueError("query_budget must be positive")
    return query_budget * expected_coordinate_reward(u=u, c=c, epsilon=epsilon)


def bit_equivalent_slope(
    *,
    kappa: float,
    query_budget: int,
    u: float,
    c: float,
    epsilon: float,
) -> float:
    """Expected infinite-frontier bit-equivalent gain per round, in nats."""

    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    return kappa * expected_reward_slope(
        query_budget=query_budget,
        u=u,
        c=c,
        epsilon=epsilon,
    )


def average_bit_equivalent(
    horizon: int,
    *,
    kappa: float,
    query_budget: int,
    u: float,
    c: float,
    epsilon: float,
) -> float:
    """Exact average over checkpoints ``t=0, ..., horizon-1``.

    At checkpoint ``t``, exactly ``query_budget * t`` observations have been
    acquired. Thus the finite-horizon result is
    ``kappa * query_budget * s * (horizon - 1) / 2``.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return (
        bit_equivalent_slope(
            kappa=kappa,
            query_budget=query_budget,
            u=u,
            c=c,
            epsilon=epsilon,
        )
        * (horizon - 1)
        / 2.0
    )


def average_bit_equivalent_slope(
    *,
    kappa: float,
    query_budget: int,
    u: float,
    c: float,
    epsilon: float,
) -> float:
    """Asymptotic coefficient of average bit-equivalent versus horizon."""

    return (
        bit_equivalent_slope(
            kappa=kappa,
            query_budget=query_budget,
            u=u,
            c=c,
            epsilon=epsilon,
        )
        / 2.0
    )


@dataclass(slots=True)
class FreshCoordinateSanityAgent:
    """Queries ``b`` fresh coordinates and deploys their observed labels.

    Query selection is a pure function of the round, and ``deployment`` returns
    a read-only snapshot. Both choices make checkpoint evaluation side-effect
    free and independent of training RNG state.
    """

    q: int
    epsilon: float
    query_budget: int
    u: float = 1.0
    c: float = 1.0
    first_rule_index: int = 1
    _predictions: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, int) or self.q < 2:
            raise ValueError("q must be an integer of at least 2")
        if (
            isinstance(self.query_budget, bool)
            or not isinstance(self.query_budget, int)
            or self.query_budget < 1
        ):
            raise ValueError("query_budget must be a positive integer")
        if (
            isinstance(self.first_rule_index, bool)
            or not isinstance(self.first_rule_index, int)
            or self.first_rule_index < 1
        ):
            raise ValueError("first_rule_index must be a positive integer")
        _validate_reward_inputs(u=self.u, c=self.c, epsilon=self.epsilon)
        if not self.epsilon < (self.q - 1) / self.q:
            raise ValueError("epsilon must define an informative q-ary channel")
        if self.expected_coordinate_reward <= 0.0:
            raise ValueError(
                "one observation must exceed the deployment profitability threshold"
            )

    @property
    def threshold(self) -> float:
        return self.c / (self.u + self.c)

    @property
    def expected_coordinate_reward(self) -> float:
        return expected_coordinate_reward(u=self.u, c=self.c, epsilon=self.epsilon)

    @property
    def learned_count(self) -> int:
        return len(self._predictions)

    def queries_for_round(self, round_index: int) -> tuple[int, ...]:
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise TypeError("round_index must be an integer")
        if round_index < 0:
            raise ValueError("round_index must be nonnegative")
        start = self.first_rule_index + round_index * self.query_budget
        return tuple(range(start, start + self.query_budget))

    select_queries = queries_for_round

    def observe(self, rule_index: int, observation: int) -> None:
        if (
            isinstance(rule_index, bool)
            or not isinstance(rule_index, int)
            or rule_index < 1
        ):
            raise ValueError("rule_index must be a positive integer")
        if (
            isinstance(observation, bool)
            or not isinstance(observation, int)
            or not 1 <= observation <= self.q
        ):
            raise ValueError(f"observation must be an integer in 1..{self.q}")
        self._predictions[rule_index] = observation

    def observe_many(
        self,
        observations: Mapping[int, int] | Iterable[tuple[int, int]],
    ) -> None:
        items = (
            observations.items() if isinstance(observations, Mapping) else observations
        )
        for rule_index, observation in items:
            self.observe(rule_index, observation)

    def deployment(self) -> DeploymentAction:
        """Return a new immutable canonical snapshot of learned predictions."""

        return DeploymentAction(self._predictions.items())

    def expected_reward_after_rounds(self, completed_rounds: int) -> float:
        if isinstance(completed_rounds, bool) or not isinstance(completed_rounds, int):
            raise TypeError("completed_rounds must be an integer")
        if completed_rounds < 0:
            raise ValueError("completed_rounds must be nonnegative")
        return completed_rounds * expected_reward_slope(
            query_budget=self.query_budget,
            u=self.u,
            c=self.c,
            epsilon=self.epsilon,
        )

    def expected_bit_equivalent_after_rounds(
        self,
        completed_rounds: int,
        *,
        kappa: float,
    ) -> float:
        if isinstance(completed_rounds, bool) or not isinstance(completed_rounds, int):
            raise TypeError("completed_rounds must be an integer")
        if completed_rounds < 0:
            raise ValueError("completed_rounds must be nonnegative")
        return completed_rounds * bit_equivalent_slope(
            kappa=kappa,
            query_budget=self.query_budget,
            u=self.u,
            c=self.c,
            epsilon=self.epsilon,
        )
