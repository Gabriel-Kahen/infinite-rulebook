"""Exact categorical posterior for q-ary symmetric observations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from infinite_rulebook.feedback.qary import QarySymmetricChannel


def _normalize_prior(prior: Sequence[float]) -> tuple[float, ...]:
    if len(prior) < 2:
        raise ValueError("a categorical prior needs at least two labels")
    if any(not math.isfinite(value) or value < 0.0 for value in prior):
        raise ValueError("prior entries must be finite and nonnegative")
    total = math.fsum(prior)
    if total <= 0.0:
        raise ValueError("prior must have positive mass")
    return tuple(value / total for value in prior)


def _validate_counts(counts: Sequence[int], q: int) -> list[int]:
    if len(counts) != q:
        raise ValueError(f"counts must contain exactly {q} entries")
    result: list[int] = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("observation counts must be integers")
        if count < 0:
            raise ValueError("observation counts must be nonnegative")
        result.append(count)
    return result


def thresholded_deployment(
    probabilities: Sequence[float],
    threshold: float,
) -> int:
    """Return the 1-based MAP label, or zero to abstain.

    Deployment is deliberately strict: confidence equal to the profitability
    threshold abstains, matching ``max_j P(Y=j|H) > tau`` in the protocol.
    Ties are resolved toward the lowest label for deterministic behavior.
    """

    normalized = _normalize_prior(probabilities)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    map_index = max(range(len(normalized)), key=normalized.__getitem__)
    if normalized[map_index] <= threshold:
        return 0
    return map_index + 1


@dataclass(slots=True)
class CategoricalPosterior:
    """A sufficient-statistic posterior for one independent coordinate."""

    q: int
    epsilon: float
    prior: Sequence[float] | None = None
    counts: Sequence[int] | None = None
    _prior: tuple[float, ...] = field(init=False, repr=False)
    _counts: list[int] = field(init=False, repr=False)
    _channel: QarySymmetricChannel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._channel = QarySymmetricChannel(self.q, self.epsilon)
        if self.prior is None:
            self._prior = (1.0 / self.q,) * self.q
        else:
            if len(self.prior) != self.q:
                raise ValueError(f"prior must contain exactly {self.q} entries")
            self._prior = _normalize_prior(self.prior)
        self._counts = _validate_counts(
            (0,) * self.q if self.counts is None else self.counts,
            self.q,
        )

    @classmethod
    def from_counts(
        cls,
        counts: Sequence[int],
        *,
        epsilon: float,
        prior: Sequence[float] | None = None,
    ) -> CategoricalPosterior:
        return cls(
            q=len(counts),
            epsilon=epsilon,
            prior=prior,
            counts=counts,
        )

    @property
    def prior_probabilities(self) -> tuple[float, ...]:
        return self._prior

    @property
    def observation_counts(self) -> tuple[int, ...]:
        return tuple(self._counts)

    @property
    def total_observations(self) -> int:
        return sum(self._counts)

    @property
    def probabilities(self) -> tuple[float, ...]:
        log_weights: list[float] = []
        error_probability = self.epsilon / (self.q - 1)
        for candidate_index, prior_mass in enumerate(self._prior):
            if prior_mass == 0.0:
                log_weights.append(-math.inf)
                continue
            log_weight = math.log(prior_mass)
            for observation_index, count in enumerate(self._counts):
                if count == 0:
                    continue
                probability = (
                    self._channel.accuracy
                    if candidate_index == observation_index
                    else error_probability
                )
                if probability == 0.0:
                    log_weight = -math.inf
                    break
                log_weight += count * math.log(probability)
            log_weights.append(log_weight)

        maximum = max(log_weights)
        if maximum == -math.inf:
            raise ValueError("the supplied counts have zero probability")
        weights = [
            0.0 if value == -math.inf else math.exp(value - maximum)
            for value in log_weights
        ]
        total = math.fsum(weights)
        return tuple(value / total for value in weights)

    @property
    def map_label(self) -> int:
        probabilities = self.probabilities
        return max(range(self.q), key=probabilities.__getitem__) + 1

    @property
    def confidence(self) -> float:
        return max(self.probabilities)

    @property
    def entropy(self) -> float:
        """Posterior entropy in nats."""

        return -math.fsum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    @property
    def kl_from_prior(self) -> float:
        """Realized ``D_KL(posterior || prior)`` in nats."""

        posterior = self.probabilities
        return math.fsum(
            probability * math.log(probability / prior)
            for probability, prior in zip(
                posterior,
                self._prior,
                strict=True,
            )
            if probability > 0.0
        )

    def update(self, observation: int) -> None:
        self._channel._validate_label(observation)
        self._counts[observation - 1] += 1

    def update_many(self, observations: Iterable[int]) -> None:
        for observation in observations:
            self.update(observation)

    def deployment(self, threshold: float) -> int:
        return thresholded_deployment(self.probabilities, threshold)

    def copy(self) -> CategoricalPosterior:
        return type(self)(
            q=self.q,
            epsilon=self.epsilon,
            prior=self._prior,
            counts=self._counts,
        )
