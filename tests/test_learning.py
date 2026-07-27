"""Tests for P1 feedback, categorical inference, and the sanity agent."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.agents.sanity import (
    FreshCoordinateSanityAgent,
    average_bit_equivalent,
    average_bit_equivalent_slope,
    bit_equivalent_slope,
    expected_coordinate_reward,
)
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.posteriors.categorical import (
    CategoricalPosterior,
    thresholded_deployment,
)


def test_semantic_noise_is_order_invariant() -> None:
    channel = QarySymmetricChannel(q=4, epsilon=0.2)
    queries = [
        (
            (index % 4) + 1,
            SemanticObservationKey(
                environment_seed="paired-environment",
                round_index=index // 3,
                rule_index=index * 17,
                query_ordinal=index % 3,
                channel="useful",
            ),
        )
        for index in range(100)
    ]

    forward = {key: channel.observe(label, key) for label, key in queries}
    reverse = {key: channel.observe(label, key) for label, key in reversed(queries)}

    assert forward == reverse
    assert all(1 <= observation <= 4 for observation in forward.values())


def test_semantic_key_fields_select_distinct_noise_draws() -> None:
    base = SemanticObservationKey(17, 2, 9, 0, "p1")
    changed = SemanticObservationKey(17, 2, 9, 1, "p1")

    assert base.canonical_bytes() != changed.canonical_bytes()
    assert (
        base.canonical_bytes()
        == SemanticObservationKey(17, 2, 9, 0, "p1").canonical_bytes()
    )


def test_semantic_feedback_key_normalizes_unicode_equivalents() -> None:
    composed = SemanticObservationKey("é", 0, 3, channel="café")
    decomposed = SemanticObservationKey("e\u0301", 0, 3, channel="cafe\u0301")
    channel = QarySymmetricChannel(q=4, epsilon=0.5)

    assert composed == decomposed
    assert composed.canonical_bytes() == decomposed.canonical_bytes()
    assert channel.observe(3, composed) == channel.observe(3, decomposed)


def test_qary_channel_distribution_and_capacity() -> None:
    channel = QarySymmetricChannel(q=4, epsilon=0.25)

    assert channel.probabilities(2) == pytest.approx((1 / 12, 3 / 4, 1 / 12, 1 / 12))
    assert math.fsum(channel.probabilities(2)) == pytest.approx(1.0)
    assert channel.capacity == pytest.approx(
        math.log(4) + 0.25 * math.log(0.25) + 0.75 * math.log(0.75) - 0.25 * math.log(3)
    )


def test_noiseless_channel_always_returns_true_label() -> None:
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    for index in range(20):
        key = SemanticObservationKey(3, index, index)
        assert channel.observe((index % 4) + 1, key) == (index % 4) + 1


def test_categorical_posterior_one_observation_is_exact() -> None:
    posterior = CategoricalPosterior(q=4, epsilon=0.2)
    posterior.update(3)

    assert posterior.probabilities == pytest.approx((1 / 15, 1 / 15, 4 / 5, 1 / 15))
    assert posterior.map_label == 3
    assert posterior.confidence == pytest.approx(0.8)
    assert posterior.observation_counts == (0, 0, 1, 0)
    assert posterior.total_observations == 1
    assert posterior.entropy + posterior.kl_from_prior == pytest.approx(math.log(4))


def test_repeated_evidence_increases_map_confidence_and_information() -> None:
    posterior = CategoricalPosterior(q=4, epsilon=0.2)
    prior_entropy = posterior.entropy
    posterior.update(2)
    first_confidence = posterior.confidence
    first_kl = posterior.kl_from_prior
    posterior.update_many([2, 2])

    assert posterior.map_label == 2
    assert posterior.confidence > first_confidence
    assert posterior.entropy < prior_entropy
    assert posterior.kl_from_prior > first_kl


def test_posterior_from_counts_and_nonuniform_prior() -> None:
    posterior = CategoricalPosterior.from_counts(
        [0, 1, 0],
        epsilon=0.3,
        prior=[0.5, 0.25, 0.25],
    )
    unnormalized = (0.5 * 0.15, 0.25 * 0.7, 0.25 * 0.15)
    total = math.fsum(unnormalized)

    assert posterior.probabilities == pytest.approx(
        tuple(value / total for value in unnormalized)
    )


def test_thresholded_deployment_abstains_at_equality() -> None:
    posterior = CategoricalPosterior(q=4, epsilon=0.2)
    posterior.update(1)

    assert posterior.deployment(posterior.confidence) == 0
    assert posterior.deployment(0.5) == 1
    assert thresholded_deployment([0.4, 0.4, 0.2], 0.4) == 0


def test_copy_is_independent() -> None:
    original = CategoricalPosterior(q=4, epsilon=0.2)
    copy = original.copy()
    copy.update(4)

    assert original.total_observations == 0
    assert copy.total_observations == 1


def test_sanity_agent_queries_fresh_coordinates_and_snapshots_deployment() -> None:
    agent = FreshCoordinateSanityAgent(
        q=4,
        epsilon=0.2,
        query_budget=3,
    )
    assert agent.queries_for_round(0) == (1, 2, 3)
    assert agent.queries_for_round(2) == (7, 8, 9)

    agent.observe_many({1: 4, 2: 3, 3: 2})
    deployment = agent.deployment()
    assert deployment.entries == ((1, 4), (2, 3), (3, 2))
    agent.observe(4, 1)
    assert deployment.entries == ((1, 4), (2, 3), (3, 2))


def test_sanity_agent_rejects_unprofitable_single_observation() -> None:
    with pytest.raises(ValueError, match="profitability"):
        FreshCoordinateSanityAgent(
            q=4,
            epsilon=0.6,
            query_budget=1,
            u=1.0,
            c=1.0,
        )


def test_analytic_sanity_slopes_match_closed_form_baseline() -> None:
    q = 4
    del q
    epsilon = 0.2
    query_budget = 3
    kappa = math.log(3)
    coordinate_reward = expected_coordinate_reward(
        u=1.0,
        c=1.0,
        epsilon=epsilon,
    )

    assert coordinate_reward == pytest.approx(0.6)
    assert bit_equivalent_slope(
        kappa=kappa,
        query_budget=query_budget,
        u=1.0,
        c=1.0,
        epsilon=epsilon,
    ) == pytest.approx(kappa * query_budget * coordinate_reward)
    assert average_bit_equivalent_slope(
        kappa=kappa,
        query_budget=query_budget,
        u=1.0,
        c=1.0,
        epsilon=epsilon,
    ) == pytest.approx(kappa * query_budget * coordinate_reward / 2)
    assert average_bit_equivalent(
        10,
        kappa=kappa,
        query_budget=query_budget,
        u=1.0,
        c=1.0,
        epsilon=epsilon,
    ) == pytest.approx(kappa * query_budget * coordinate_reward * 4.5)


def test_sanity_agent_empirical_reward_matches_analytic_slope() -> None:
    environment = IndependentRulebook(seed="sanity-integration")
    channel = QarySymmetricChannel(q=4, epsilon=0.2)
    agent = FreshCoordinateSanityAgent(q=4, epsilon=0.2, query_budget=20)
    rounds = 500

    for round_index in range(rounds):
        for ordinal, rule_index in enumerate(agent.queries_for_round(round_index)):
            key = SemanticObservationKey(
                environment_seed=environment.seed,
                round_index=round_index,
                rule_index=rule_index,
                query_ordinal=ordinal,
            )
            agent.observe(
                rule_index,
                channel.observe(environment.label(rule_index), key),
            )

    empirical_reward = environment.evaluate(agent.deployment())
    analytic_reward = agent.expected_reward_after_rounds(rounds)
    assert empirical_reward == pytest.approx(analytic_reward, abs=150.0)


@pytest.mark.parametrize(
    ("q", "epsilon"),
    [
        (1, 0.0),
        (4, -0.1),
        (4, 0.75),
    ],
)
def test_invalid_channel_parameters_are_rejected(q: int, epsilon: float) -> None:
    with pytest.raises(ValueError):
        QarySymmetricChannel(q=q, epsilon=epsilon)
