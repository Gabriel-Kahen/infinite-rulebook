"""Pure one-step objectives for factorized q-ary agents."""

from __future__ import annotations

import math

from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.feedback.qary import QarySymmetricChannel
from infinite_rulebook.posteriors.categorical import CategoricalPosterior


def _validate_compatibility(
    posterior: CategoricalPosterior,
    reward_spec: RewardSpec,
) -> None:
    if not isinstance(posterior, CategoricalPosterior):
        raise TypeError("posterior must be a CategoricalPosterior")
    if not isinstance(reward_spec, RewardSpec):
        raise TypeError("reward_spec must be a RewardSpec")
    if posterior.q != reward_spec.q:
        raise ValueError("posterior and reward alphabet sizes must match")


def _clamp_roundoff(value: float, scale: float) -> float:
    tolerance = 1e-12 * max(1.0, abs(scale))
    return 0.0 if -tolerance <= value < 0.0 else value


def posterior_predictive(
    posterior: CategoricalPosterior,
) -> tuple[float, ...]:
    """Return ``P(O=o | H)`` for the next observation."""

    if not isinstance(posterior, CategoricalPosterior):
        raise TypeError("posterior must be a CategoricalPosterior")
    channel = QarySymmetricChannel(posterior.q, posterior.epsilon)
    probabilities = posterior.probabilities
    return tuple(
        math.fsum(
            truth_probability * channel.likelihood(observation, truth)
            for truth, truth_probability in enumerate(probabilities, start=1)
        )
        for observation in range(1, posterior.q + 1)
    )


def bayes_deployment_value(
    posterior: CategoricalPosterior,
    reward_spec: RewardSpec,
) -> float:
    """Return the current optimal expected value, including abstention."""

    _validate_compatibility(posterior, reward_spec)
    confidence = max(posterior.probabilities)
    predicted_value = reward_spec.u * confidence - reward_spec.c * (1.0 - confidence)
    return max(0.0, predicted_value)


def expected_post_query_bayes_value(
    posterior: CategoricalPosterior,
    reward_spec: RewardSpec,
) -> float:
    """Return Bayes deployment value averaged over the next observation."""

    _validate_compatibility(posterior, reward_spec)
    terms = []
    for observation, probability in enumerate(posterior_predictive(posterior), start=1):
        if probability == 0.0:
            continue
        updated = posterior.copy()
        updated.update(observation)
        terms.append(probability * bayes_deployment_value(updated, reward_spec))
    return math.fsum(terms)


def decision_value_gain(
    posterior: CategoricalPosterior,
    reward_spec: RewardSpec,
) -> float:
    """Return the expected one-query improvement in Bayes deployment value."""

    current = bayes_deployment_value(posterior, reward_spec)
    expected = expected_post_query_bayes_value(posterior, reward_spec)
    return _clamp_roundoff(expected - current, expected + current)


def expected_entropy_reduction(posterior: CategoricalPosterior) -> float:
    """Return expected posterior entropy reduction in nats."""

    if not isinstance(posterior, CategoricalPosterior):
        raise TypeError("posterior must be a CategoricalPosterior")
    expected_entropy = math.fsum(
        probability * updated.entropy
        for observation, probability in enumerate(
            posterior_predictive(posterior),
            start=1,
        )
        if probability > 0.0
        for updated in (_updated_posterior(posterior, observation),)
    )
    return _clamp_roundoff(
        posterior.entropy - expected_entropy,
        posterior.entropy + expected_entropy,
    )


def prediction_error_novelty(posterior: CategoricalPosterior) -> float:
    """Return the minimum expected 0-1 error when predicting the observation."""

    predictive = posterior_predictive(posterior)
    return 1.0 - max(predictive)


def _updated_posterior(
    posterior: CategoricalPosterior,
    observation: int,
) -> CategoricalPosterior:
    updated = posterior.copy()
    updated.update(observation)
    return updated
