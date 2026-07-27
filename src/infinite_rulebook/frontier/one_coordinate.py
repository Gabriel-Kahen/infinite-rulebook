"""Exact reward-information frontier for one independent rule.

All information quantities in this module use natural logarithms and are
therefore measured in nats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real


def _as_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_reward(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reward must be a real number")
    result = float(value)
    if math.isnan(result):
        raise ValueError("reward must not be NaN")
    return result


@dataclass(frozen=True, slots=True)
class OneCoordinateFrontier:
    """Analytic frontier for a uniformly distributed ``q``-ary rule.

    A correct prediction earns ``u``, an incorrect prediction earns ``-c``,
    and abstention earns zero.  The strict margin ``c > u / (q - 1)`` is
    required: without it, the infinite-coordinate frontier collapses.
    """

    q: int = 4
    u: float = 1.0
    c: float = 1.0
    _tau: float = field(init=False, repr=False, compare=False)
    _p_star: float = field(init=False, repr=False, compare=False)
    _r_star: float = field(init=False, repr=False, compare=False)
    _kappa: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, int):
            raise TypeError("q must be an integer")
        if self.q < 2:
            raise ValueError("q must be at least 2")

        u = _as_real("u", self.u)
        c = _as_real("c", self.c)
        if u <= 0.0:
            raise ValueError("u must be strictly positive")
        if c <= 0.0:
            raise ValueError("c must be strictly positive")
        if c <= u / (self.q - 1):
            raise ValueError(
                "strict negative uninformed margin required: c > u / (q - 1)"
            )
        if not math.isfinite(u + c):
            raise ValueError("u + c must be finite")

        # This form avoids overflow in c / (u + c) when c is much larger than u.
        tau = 1.0 / (1.0 + u / c) if c >= u else (c / u) / (1.0 + c / u)
        if not 1.0 / self.q < tau < 1.0:
            raise ValueError("u and c are too ill-conditioned for float arithmetic")

        object.__setattr__(self, "u", u)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "_tau", tau)

        p_star = self._solve_p_star()
        r_star = self.value_at_accuracy(p_star)
        kappa = self.information_at_accuracy(p_star) / r_star
        object.__setattr__(self, "_p_star", p_star)
        object.__setattr__(self, "_r_star", r_star)
        object.__setattr__(self, "_kappa", kappa)

    @property
    def tau(self) -> float:
        """Minimum conditional accuracy for profitable deployment."""

        return self._tau

    @property
    def p_star(self) -> float:
        """Accuracy at the end of the frontier's linear segment."""

        return self._p_star

    @property
    def r_star(self) -> float:
        """Reward at the end of the frontier's linear segment."""

        return self._r_star

    @property
    def kappa(self) -> float:
        """Information cost per reward in the linear segment, in nats."""

        return self._kappa

    def value_at_accuracy(self, p: Real) -> float:
        """Return conditional expected reward at deployment accuracy ``p``."""

        accuracy = self._validated_accuracy(p)
        return (self.u + self.c) * accuracy - self.c

    def information_at_accuracy(self, p: Real) -> float:
        """Return ``J_q(p)`` for the symmetric ``q``-ary channel, in nats."""

        accuracy = self._validated_accuracy(p)
        chance = 1.0 / self.q
        if accuracy == chance:
            return 0.0
        if accuracy == 1.0:
            return math.log(self.q)

        incorrect = 1.0 - accuracy
        information = accuracy * math.log(self.q * accuracy)
        information += incorrect * math.log(self.q * incorrect / (self.q - 1))
        # Roundoff can only make this negative extremely close to chance.
        return max(0.0, information)

    def bit_equivalent(self, reward: Real) -> float:
        """Return the minimum action information for a reward threshold.

        Nonpositive thresholds are attainable by abstaining and cost zero
        information.  Thresholds above the one-rule maximum return infinity.
        """

        target = _as_reward(reward)
        if target <= 0.0:
            return 0.0
        if target > self.u:
            return math.inf
        if target == self.u:
            return math.log(self.q)
        if target <= self.r_star:
            return self.kappa * target

        accuracy = (target + self.c) / (self.u + self.c)
        return self.information_at_accuracy(accuracy)

    def infinite_bit_equivalent(self, reward: Real) -> float:
        """Return the exact frontier for countably many independent rules."""

        target = _as_reward(reward)
        if target <= 0.0:
            return 0.0
        if math.isinf(target):
            return math.inf
        return self.kappa * target

    def _validated_accuracy(self, p: Real) -> float:
        accuracy = _as_real("p", p)
        chance = 1.0 / self.q
        if not chance <= accuracy <= 1.0:
            raise ValueError(f"p must lie in [1 / q, 1], got {accuracy}")
        return accuracy

    def _information_derivative(self, p: float) -> float:
        if p == 1.0:
            return math.inf
        return math.log((self.q - 1) * p / (1.0 - p))

    def _stationarity(self, p: float) -> float:
        value = (self.u + self.c) * p - self.c
        return value * self._information_derivative(p) - (
            self.u + self.c
        ) * self.information_at_accuracy(p)

    def _solve_p_star(self) -> float:
        # The baseline oracle is exact and is used throughout regression tests.
        if self.q == 4 and self.u == self.c:
            return 0.75

        low = self.tau
        high = 1.0
        for _ in range(256):
            midpoint = (low + high) / 2.0
            if midpoint in (low, high):
                # The mathematical root is interior, but for extreme reward
                # ratios it can lie between 1.0 and the next lower float.
                return low if high == 1.0 else high
            if self._stationarity(midpoint) < 0.0:
                low = midpoint
            else:
                high = midpoint
        return (low + high) / 2.0
