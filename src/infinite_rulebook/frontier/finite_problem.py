"""Finite source/action problems and direct stochastic-channel witnesses.

All information quantities are measured in nats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

Channel = tuple[tuple[float, ...], ...]


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _log_action_marginal(
    prior: tuple[float, ...],
    channel: Channel,
    action: int,
) -> float:
    terms = tuple(
        math.log(state_probability) + math.log(channel[state][action])
        for state, state_probability in enumerate(prior)
        if state_probability > 0.0 and channel[state][action] > 0.0
    )
    if not terms:
        return -math.inf
    maximum = max(terms)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in terms))


@dataclass(frozen=True, slots=True)
class ChannelWitness:
    """A feasible behavioral channel and its exact derived quantities."""

    channel: Channel
    action_marginal: tuple[float, ...]
    expected_reward: float
    mutual_information: float


@dataclass(frozen=True, slots=True)
class FiniteDecisionProblem:
    """A finite prior and state-by-action reward matrix."""

    prior: tuple[float, ...]
    rewards: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        try:
            prior = tuple(
                _finite_real(f"prior[{index}]", value)
                for index, value in enumerate(self.prior)
            )
        except TypeError as error:
            raise TypeError("prior must be a finite sequence") from error
        if not prior:
            raise ValueError("prior must contain at least one state")
        if any(probability < 0.0 for probability in prior):
            raise ValueError("prior probabilities must be nonnegative")
        total = math.fsum(prior)
        if total <= 0.0:
            raise ValueError("prior must have positive total mass")
        prior = tuple(probability / total for probability in prior)

        try:
            rows = tuple(tuple(row) for row in self.rewards)
        except TypeError as error:
            raise TypeError("rewards must be a finite matrix") from error
        if len(rows) != len(prior):
            raise ValueError("rewards must have one row per prior state")
        if not rows or not rows[0]:
            raise ValueError("rewards must contain at least one action")
        action_count = len(rows[0])
        if any(len(row) != action_count for row in rows):
            raise ValueError("reward rows must have equal length")
        rewards = tuple(
            tuple(
                _finite_real(f"rewards[{state}][{action}]", value)
                for action, value in enumerate(row)
            )
            for state, row in enumerate(rows)
        )

        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "rewards", rewards)

    @property
    def state_count(self) -> int:
        return len(self.prior)

    @property
    def action_count(self) -> int:
        return len(self.rewards[0])

    @property
    def constant_action_rewards(self) -> tuple[float, ...]:
        """Expected rewards of state-independent deterministic actions."""

        return tuple(
            math.fsum(
                self.prior[state] * self.rewards[state][action]
                for state in range(self.state_count)
            )
            for action in range(self.action_count)
        )

    @property
    def zero_information_reward(self) -> float:
        """Largest reward attainable without revealing the state."""

        return max(self.constant_action_rewards)

    @property
    def constant_action_reward(self) -> float:
        """Compatibility name for the best state-independent reward."""

        return self.zero_information_reward

    @property
    def maximum_reward(self) -> float:
        """Largest reward attainable by an unrestricted state-aware channel."""

        return math.fsum(
            probability * max(row)
            for probability, row in zip(self.prior, self.rewards, strict=True)
        )

    def constant_channel(self, action: int | None = None) -> ChannelWitness:
        """Return a best, or explicitly selected, constant-action witness."""

        if action is None:
            action = max(
                range(self.action_count),
                key=self.constant_action_rewards.__getitem__,
            )
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer")
        if not 0 <= action < self.action_count:
            raise ValueError("action index is out of range")
        row = tuple(
            1.0 if candidate == action else 0.0
            for candidate in range(self.action_count)
        )
        return self.evaluate(tuple(row for _ in range(self.state_count)))

    def maximizing_channel(self) -> ChannelWitness:
        """Return a deterministic channel attaining ``maximum_reward``."""

        rows = []
        for rewards in self.rewards:
            best = max(range(self.action_count), key=rewards.__getitem__)
            rows.append(
                tuple(
                    1.0 if action == best else 0.0
                    for action in range(self.action_count)
                )
            )
        return self.evaluate(tuple(rows))

    def validate_channel(self, channel: Sequence[Sequence[Real]]) -> Channel:
        """Validate and canonically normalize a state-by-action channel."""

        try:
            rows = tuple(tuple(row) for row in channel)
        except TypeError as error:
            raise TypeError("channel must be a finite matrix") from error
        if len(rows) != self.state_count:
            raise ValueError("channel must have one row per state")

        normalized = []
        for state, row in enumerate(rows):
            if len(row) != self.action_count:
                raise ValueError("channel rows must have one value per action")
            values = tuple(
                _finite_real(f"channel[{state}][{action}]", value)
                for action, value in enumerate(row)
            )
            if any(value < 0.0 for value in values):
                raise ValueError("channel probabilities must be nonnegative")
            total = math.fsum(values)
            if total <= 0.0:
                raise ValueError("each channel row must have positive mass")
            normalized.append(tuple(value / total for value in values))
        return tuple(normalized)

    def evaluate(self, channel: Sequence[Sequence[Real]]) -> ChannelWitness:
        """Return reward and mutual information for a feasible channel."""

        canonical = self.validate_channel(channel)
        marginal = tuple(
            math.fsum(
                self.prior[state] * canonical[state][action]
                for state in range(self.state_count)
            )
            for action in range(self.action_count)
        )
        reward = math.fsum(
            self.prior[state] * canonical[state][action] * self.rewards[state][action]
            for state in range(self.state_count)
            for action in range(self.action_count)
        )
        information_terms = []
        underflow_log_marginals = {
            action: _log_action_marginal(self.prior, canonical, action)
            for action in range(self.action_count)
            if any(
                state_probability > 0.0
                and canonical[state][action] > 0.0
                and state_probability * canonical[state][action] == 0.0
                for state, state_probability in enumerate(self.prior)
            )
        }
        for state, state_probability in enumerate(self.prior):
            if state_probability == 0.0:
                continue
            for action, conditional in enumerate(canonical[state]):
                if conditional == 0.0:
                    continue
                joint_probability = state_probability * conditional
                action_probability = marginal[action]
                if joint_probability > 0.0 and action_probability > 0.0:
                    ratio = conditional / action_probability
                    if math.isfinite(ratio) and ratio > 0.0:
                        information_terms.append(joint_probability * math.log(ratio))
                        continue
                if action in underflow_log_marginals:
                    log_action_probability = underflow_log_marginals[action]
                else:
                    if action_probability <= 0.0:
                        raise ArithmeticError(
                            "positive conditional has zero action marginal"
                        )
                    log_action_probability = math.log(action_probability)
                if not math.isfinite(log_action_probability):
                    raise ArithmeticError(
                        "positive conditional has zero action marginal"
                    )
                log_ratio = math.log(conditional) - log_action_probability
                if log_ratio == 0.0:
                    continue
                if joint_probability > 0.0:
                    information_terms.append(joint_probability * log_ratio)
                    continue
                # Recover a representable MI term even when p(theta)q(a|theta)
                # itself falls below the float range.
                log_joint = math.log(state_probability) + math.log(conditional)
                information_terms.append(
                    math.copysign(
                        math.exp(log_joint + math.log(abs(log_ratio))),
                        log_ratio,
                    )
                )
        information = max(0.0, math.fsum(information_terms))
        return ChannelWitness(canonical, marginal, reward, information)


def one_coordinate_problem(
    q: int = 4,
    u: Real = 1.0,
    c: Real = 1.0,
) -> FiniteDecisionProblem:
    """Construct the finite abstain-or-predict problem for one ``q``-ary rule."""

    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError("q must be an integer")
    if q < 2:
        raise ValueError("q must be at least 2")
    utility = _finite_real("u", u)
    cost = _finite_real("c", c)
    if utility <= 0.0 or cost <= 0.0:
        raise ValueError("u and c must be strictly positive")

    rewards = []
    for state in range(q):
        rewards.append(
            (
                0.0,
                *(utility if prediction == state else -cost for prediction in range(q)),
            )
        )
    return FiniteDecisionProblem(
        prior=tuple(1.0 / q for _ in range(q)),
        rewards=tuple(rewards),
    )
