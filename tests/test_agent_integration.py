"""End-to-end symbolic agent, control, ledger, and frontier checks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from infinite_rulebook.agents.comparison import (
    ExpandingTargetSchedule,
    FactorizedQueryAgent,
    FixedTargetPolicy,
    NoveltyDirectedPolicy,
    RelevantInformationDirectedPolicy,
    RewardDirectedPolicy,
    ScheduledTargetPolicy,
    TotalInformationDirectedPolicy,
    distractor_targets,
    useful_targets,
)
from infinite_rulebook.agents.integration import (
    P1RoundTrace,
    exact_ind_bit_equivalent,
    execute_p1_round,
    factorized_information_ledger,
    population_information_estimate,
)
from infinite_rulebook.agents.protocols import QueryTarget, TargetKey
from infinite_rulebook.environments import (
    AleaRulebook,
    IndependentRulebook,
    QueryNamespace,
    RulebookRuntime,
    TriviaRulebook,
)
from infinite_rulebook.feedback import QarySymmetricChannel
from infinite_rulebook.frontier import (
    alea_persistent_information_nats,
    trivia_invariant_bit_equivalent,
)
from infinite_rulebook.metrics import MetricInterval

P1_SEED = "matched-p1-noise"


def _registry(
    candidates: Sequence[QueryTarget],
) -> dict[TargetKey, QueryTarget]:
    return {target.key: target for target in candidates}


def _execute_pure_round(
    agent: FactorizedQueryAgent,
    environment: RulebookRuntime[object],
    candidates: Sequence[QueryTarget],
    channel: QarySymmetricChannel,
) -> P1RoundTrace:
    context = agent.acquisition_context(candidates)
    selected = agent.select_train_action(context)
    checkpoint = agent.checkpoint()

    assert agent.checkpoint() == checkpoint
    assert agent.select_train_action(context) == selected
    environment.evaluate(checkpoint.deployment)
    assert agent.checkpoint() == checkpoint
    assert agent.select_train_action(context) == selected

    trace = execute_p1_round(
        agent,
        environment,
        context,
        channel,
        environment_seed=P1_SEED,
    )
    assert trace.action == selected

    completed = agent.checkpoint()
    environment.evaluate(completed.deployment)
    assert agent.checkpoint() == completed
    return trace


def test_alea_novelty_requeries_after_reward_and_information_saturate() -> None:
    environment = AleaRulebook(
        IndependentRulebook("alea-reward"),
        cosmetic_seed="alea-cosmetic",
        cosmetic_alphabet=32,
    )
    candidates = useful_targets(1)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    agent = FactorizedQueryAgent(
        NoveltyDirectedPolicy(cosmetic_alphabet=32),
        epsilon=0.0,
        query_budget=1,
        seed="alea-agent",
    )
    rewards = []
    bit_equivalents = []
    relevant_information = []
    cosmetics = []

    for _ in range(8):
        trace = _execute_pure_round(agent, environment, candidates, channel)
        checkpoint = agent.checkpoint()
        breakdown = factorized_information_ledger(checkpoint, registry).breakdown
        reward = environment.evaluate(checkpoint.deployment)

        assert trace.queries[0].namespace is QueryNamespace.REWARD
        cosmetics.append(trace.cosmetic_values[0])
        rewards.append(reward)
        bit_equivalents.append(exact_ind_bit_equivalent(reward).lower)
        relevant_information.append(breakdown.relevant_nats)
        assert breakdown.persistent_distractor_nats == 0.0
        assert breakdown.total_acquired_nats == pytest.approx(math.log(4))

    assert all(value is not None for value in cosmetics)
    assert alea_persistent_information_nats() == 0.0
    assert agent.query_count(candidates[0].key) == 8
    assert rewards == [1.0] * 8
    assert bit_equivalents == pytest.approx([math.log(3)] * 8)
    assert relevant_information == pytest.approx([math.log(4)] * 8)


def test_total_information_chases_trivia_before_reward() -> None:
    environment = AleaRulebook(
        TriviaRulebook(
            IndependentRulebook("total-reward"),
            trivia_seed="total-trivia",
        ),
        cosmetic_seed="total-alea",
        cosmetic_alphabet=32,
    )
    candidates = useful_targets(1) + distractor_targets(7)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    agent = FactorizedQueryAgent(
        TotalInformationDirectedPolicy(),
        epsilon=0.0,
        query_budget=1,
        seed=3,
    )
    namespaces = []
    rewards = []
    distractor_information = []
    bit_equivalents = []

    for _ in range(8):
        trace = _execute_pure_round(agent, environment, candidates, channel)
        checkpoint = agent.checkpoint()
        breakdown = factorized_information_ledger(checkpoint, registry).breakdown
        reward = environment.evaluate(checkpoint.deployment)

        namespaces.append(trace.queries[0].namespace)
        rewards.append(reward)
        distractor_information.append(breakdown.persistent_distractor_nats)
        bit_equivalents.append(
            trivia_invariant_bit_equivalent(
                lambda value: exact_ind_bit_equivalent(value).lower,
                reward,
            )
        )
        assert trace.cosmetic_values[0] is not None

    assert namespaces == [QueryNamespace.TRIVIA] * 7 + [QueryNamespace.REWARD]
    assert rewards == [0.0] * 7 + [1.0]
    assert distractor_information[:7] == pytest.approx(
        [math.log(4) * count for count in range(1, 8)]
    )
    assert distractor_information[-1] == pytest.approx(7 * math.log(4))
    assert bit_equivalents == pytest.approx([0.0] * 7 + [math.log(3)])

    final = factorized_information_ledger(agent.checkpoint(), registry).breakdown
    assert final.relevant_nats == pytest.approx(math.log(4))
    assert final.total_acquired_nats == pytest.approx(8 * math.log(4))


@pytest.mark.parametrize(
    "policy",
    [RewardDirectedPolicy(), RelevantInformationDirectedPolicy()],
    ids=["reward", "relevant-information"],
)
def test_useful_objectives_choose_reward_over_trivia(policy: object) -> None:
    environment = TriviaRulebook(
        IndependentRulebook("useful-reward"),
        trivia_seed="useful-trivia",
    )
    candidates = useful_targets(1) + distractor_targets(7)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    agent = FactorizedQueryAgent(
        policy,  # type: ignore[arg-type]
        epsilon=0.0,
        query_budget=1,
        seed=3,
    )

    trace = _execute_pure_round(agent, environment, candidates, channel)
    checkpoint = agent.checkpoint()
    breakdown = factorized_information_ledger(checkpoint, registry).breakdown
    reward = environment.evaluate(checkpoint.deployment)

    assert trace.queries[0].namespace is QueryNamespace.REWARD
    assert reward == 1.0
    assert breakdown.relevant_nats == pytest.approx(math.log(4))
    assert breakdown.persistent_distractor_nats == 0.0
    assert breakdown.total_acquired_nats == pytest.approx(math.log(4))
    bit_equivalent = exact_ind_bit_equivalent(reward)
    assert isinstance(bit_equivalent, MetricInterval)
    assert bit_equivalent.units == "nats"
    assert bit_equivalent.lower == bit_equivalent.upper
    assert bit_equivalent.lower == pytest.approx(math.log(3))


def test_multi_query_round_uses_target_local_semantic_ordinals() -> None:
    environment = TriviaRulebook(
        IndependentRulebook("batch-reward"),
        trivia_seed="batch-trivia",
    )
    candidates = useful_targets(1) + distractor_targets(1)
    agent = FactorizedQueryAgent(
        TotalInformationDirectedPolicy(),
        epsilon=0.2,
        query_budget=2,
        seed="batch-agent",
    )

    trace = execute_p1_round(
        agent,
        environment,
        agent.acquisition_context(candidates),
        QarySymmetricChannel(q=4, epsilon=0.2),
        environment_seed=P1_SEED,
    )

    assert len(trace.action.targets) == 2
    assert tuple(key.query_ordinal for key in trace.observation_keys) == (0, 0)
    assert len(set(zip(trace.queries, trace.observation_keys, strict=True))) == 2
    effective_keys = tuple(
        environment.observation_key(query, key)
        for query, key in zip(
            trace.queries,
            trace.observation_keys,
            strict=True,
        )
    )
    assert len(set(effective_keys)) == 2


def test_fixed_target_saturates_real_ind_trajectory() -> None:
    environment = IndependentRulebook("fixed-integration")
    candidates = useful_targets(8)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    agent = FactorizedQueryAgent(
        FixedTargetPolicy(2),
        epsilon=0.0,
        query_budget=1,
        seed="fixed-integration-agent",
    )
    rewards = [0.0]
    relevant_information = [0.0]
    bit_equivalents = [0.0]

    for _ in range(5):
        _execute_pure_round(agent, environment, candidates, channel)
        checkpoint = agent.checkpoint()
        breakdown = factorized_information_ledger(checkpoint, registry).breakdown
        reward = environment.evaluate(checkpoint.deployment)
        rewards.append(reward)
        relevant_information.append(breakdown.relevant_nats)
        bit_equivalents.append(exact_ind_bit_equivalent(reward).lower)

    assert rewards == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert relevant_information == pytest.approx(
        [value * math.log(4) for value in (0, 1, 2, 2, 2, 2)]
    )
    assert bit_equivalents == pytest.approx(
        [value * math.log(3) for value in (0, 1, 2, 2, 2, 2)]
    )


def test_scheduled_target_sustains_linear_ind_growth() -> None:
    environment = IndependentRulebook("scheduled-integration")
    candidates = useful_targets(5)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    agent = FactorizedQueryAgent(
        ScheduledTargetPolicy(ExpandingTargetSchedule(1, 1)),
        epsilon=0.0,
        query_budget=1,
        seed="scheduled-integration-agent",
    )
    rewards = [0.0]
    relevant_information = [0.0]
    bit_equivalents = [0.0]

    for _ in range(5):
        _execute_pure_round(agent, environment, candidates, channel)
        checkpoint = agent.checkpoint()
        breakdown = factorized_information_ledger(checkpoint, registry).breakdown
        reward = environment.evaluate(checkpoint.deployment)
        rewards.append(reward)
        relevant_information.append(breakdown.relevant_nats)
        bit_equivalents.append(exact_ind_bit_equivalent(reward).lower)

    assert rewards == [float(value) for value in range(6)]
    assert relevant_information == pytest.approx(
        [value * math.log(4) for value in range(6)]
    )
    assert bit_equivalents == pytest.approx([value * math.log(3) for value in range(6)])


def test_population_information_adapter_averages_complete_histories() -> None:
    candidates = useful_targets(1) + distractor_targets(1)
    registry = _registry(candidates)
    channel = QarySymmetricChannel(q=4, epsilon=0.0)
    histories = []

    for environment_seed in ("ensemble-a", "ensemble-b"):
        environment = TriviaRulebook(
            IndependentRulebook(environment_seed),
            trivia_seed=f"{environment_seed}-trivia",
        )
        agent = FactorizedQueryAgent(
            TotalInformationDirectedPolicy(),
            epsilon=0.0,
            seed=0,
        )
        execute_p1_round(
            agent,
            environment,
            agent.acquisition_context(candidates),
            channel,
            environment_seed=environment_seed,
        )
        histories.append(
            factorized_information_ledger(
                agent.checkpoint(),
                registry,
            ).breakdown
        )

    estimate = population_information_estimate(histories)
    assert estimate.run_count == 2
    assert estimate.validate().valid
    assert estimate.total_nats == pytest.approx(math.log(4))


def test_runner_uses_target_local_ordinals_and_preserves_cosmetics() -> None:
    environment = AleaRulebook(
        IndependentRulebook("ordinal-environment"),
        cosmetic_seed="ordinal-cosmetics",
        cosmetic_alphabet=8,
    )
    candidates = useful_targets(2)
    agent = FactorizedQueryAgent(
        NoveltyDirectedPolicy(cosmetic_alphabet=8),
        epsilon=0.0,
        query_budget=2,
        seed="ordinal-agent",
    )

    trace = execute_p1_round(
        agent,
        environment,
        agent.acquisition_context(candidates),
        QarySymmetricChannel(q=4, epsilon=0.0),
        environment_seed="ordinal-observations",
    )

    assert len(trace.action.targets) == 2
    assert [key.query_ordinal for key in trace.observation_keys] == [0, 0]
    assert all(value is not None for value in trace.cosmetic_values)
    assert agent.query_count(candidates[0].key) == 1
    assert agent.query_count(candidates[1].key) == 1


def test_runner_rejects_non_control_targets_before_selection() -> None:
    environment = IndependentRulebook("adapter-validation")
    agent = FactorizedQueryAgent(TotalInformationDirectedPolicy(), epsilon=0.0)
    invalid = QueryTarget(
        TargetKey("alea", 1),
        relevance_weight=0.0,
        persistent=False,
    )

    with pytest.raises(ValueError, match="persistent"):
        execute_p1_round(
            agent,
            environment,
            agent.acquisition_context((invalid,)),
            QarySymmetricChannel(q=4, epsilon=0.0),
            environment_seed="adapter-validation",
        )

    assert agent.completed_rounds == 0
    valid = useful_targets(1)
    action = agent.select_train_action(agent.acquisition_context(valid))
    assert action.targets == valid
