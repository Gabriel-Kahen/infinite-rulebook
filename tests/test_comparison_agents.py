"""Deterministic comparisons for the symbolic acquisition agents."""

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
from infinite_rulebook.agents.objectives import (
    bayes_deployment_value,
    decision_value_gain,
    expected_entropy_reduction,
    expected_post_query_bayes_value,
    posterior_predictive,
    prediction_error_novelty,
)
from infinite_rulebook.agents.protocols import (
    AcquisitionContext,
    ObservationBatch,
    QueryAction,
    QueryTarget,
    TargetKey,
)
from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.feedback.qary import QarySymmetricChannel, SemanticObservationKey
from infinite_rulebook.posteriors.categorical import CategoricalPosterior


def _observe(
    agent: FactorizedQueryAgent,
    targets: Sequence[QueryTarget],
    observations: Sequence[int],
) -> None:
    expected = {
        target.key: observation
        for target, observation in zip(targets, observations, strict=True)
    }
    action = agent.select_train_action(agent.acquisition_context(targets))
    assert {target.key for target in action.targets} == set(expected)
    agent.observe(
        ObservationBatch(
            action,
            tuple(expected[target.key] for target in action.targets),
        )
    )


def _train_on_ind(
    agent: FactorizedQueryAgent,
    environment: IndependentRulebook,
    candidates: Sequence[QueryTarget],
    rounds: int,
) -> tuple[list[QueryAction], list[DeploymentAction]]:
    actions = []
    deployments = []
    for _ in range(rounds):
        context = agent.acquisition_context(candidates)
        action = agent.select_train_action(context)
        actions.append(action)
        agent.observe(
            ObservationBatch(
                action,
                tuple(
                    environment.label(target.rule_index)
                    for target in action.targets
                    if target.rule_index is not None
                ),
            )
        )
        deployments.append(agent.deployment())
    return actions, deployments


def test_fixed_target_respects_boundary_and_saturates() -> None:
    environment = IndependentRulebook("fixed")
    candidates = useful_targets(6)
    agent = FactorizedQueryAgent(
        FixedTargetPolicy(target_size=3),
        epsilon=0.0,
        query_budget=2,
        seed="fixed-agent",
    )

    actions, _ = _train_on_ind(agent, environment, candidates, rounds=3)
    queried = {target.key.index for action in actions for target in action.targets}

    assert [len(action.targets) for action in actions] == [2, 1, 0]
    assert queried == {1, 2, 3}
    assert agent.deployment().support == (1, 2, 3)
    assert environment.evaluate(agent.deployment()) == 3.0


def test_scheduled_target_boundaries_sustain_noiseless_ind_growth() -> None:
    environment = IndependentRulebook("scheduled")
    candidates = useful_targets(8)
    schedule = ExpandingTargetSchedule(initial_size=2, growth_step=2)
    agent = FactorizedQueryAgent(
        ScheduledTargetPolicy(schedule),
        epsilon=0.0,
        query_budget=2,
        seed="scheduled-agent",
    )

    actions, deployments = _train_on_ind(agent, environment, candidates, rounds=4)
    rewards = [environment.evaluate(deployment) for deployment in deployments]

    assert [schedule.size_at(round_index) for round_index in range(5)] == [
        2,
        4,
        6,
        8,
        10,
    ]
    assert [{target.key.index for target in action.targets} for action in actions] == [
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    ]
    assert rewards == [2.0, 4.0, 6.0, 8.0]
    assert environment.evaluate(agent.deployment()) == 8.0


def test_reward_directed_prefers_useful_target_to_trivia() -> None:
    useful = useful_targets(1)[0]
    trivia = distractor_targets(1)[0]
    agent = FactorizedQueryAgent(
        RewardDirectedPolicy(),
        query_budget=1,
        seed="reward",
    )

    action = agent.select_train_action(agent.acquisition_context((trivia, useful)))

    assert action.targets == (useful,)


def test_reward_directed_does_not_deadlock_before_threshold() -> None:
    useful = useful_targets(1)[0]
    agent = FactorizedQueryAgent(
        RewardDirectedPolicy(),
        epsilon=0.6,
        query_budget=1,
    )

    assert decision_value_gain(agent.posterior(useful.key), agent.reward_spec) == 0.0
    action = agent.select_train_action(agent.acquisition_context((useful,)))

    assert action.targets == (useful,)


def test_reward_directed_repeats_evidence_in_a_broad_weak_channel() -> None:
    candidates = useful_targets(100)
    agent = FactorizedQueryAgent(
        RewardDirectedPolicy(),
        epsilon=0.65,
        query_budget=1,
        seed="weak-channel",
    )
    queried = []

    for _ in range(8):
        action = agent.select_train_action(agent.acquisition_context(candidates))
        queried.append(action.targets[0].key)
        agent.observe(ObservationBatch(action, (1,)))

    assert queried[0] == queried[1] == queried[2]
    assert agent.deployment().support


def test_novelty_directed_prefers_fresh_alea_to_learned_useful() -> None:
    useful = useful_targets(1)[0]
    alea = distractor_targets(1, namespace="alea", persistent=False)[0]
    agent = FactorizedQueryAgent(
        NoveltyDirectedPolicy(),
        query_budget=1,
        seed="novelty",
    )
    _observe(agent, (useful,), (1,))

    action = agent.select_train_action(agent.acquisition_context((useful, alea)))

    assert action.targets == (alea,)


def test_total_information_prefers_persistent_trivia_and_excludes_alea() -> None:
    useful = useful_targets(1)[0]
    trivia = distractor_targets(1)[0]
    alea = distractor_targets(1, namespace="alea", persistent=False)[0]
    agent = FactorizedQueryAgent(
        TotalInformationDirectedPolicy(),
        epsilon=0.0,
        query_budget=1,
        seed="total-information",
    )
    _observe(agent, (useful,), (1,))

    action = agent.select_train_action(
        agent.acquisition_context((alea, useful, trivia))
    )

    assert action.targets == (trivia,)


def test_total_information_ties_use_seeded_candidate_choice() -> None:
    candidates = useful_targets(1) + distractor_targets(3)

    def choice(seed: int, order: Sequence[QueryTarget]) -> TargetKey:
        agent = FactorizedQueryAgent(
            TotalInformationDirectedPolicy(),
            query_budget=1,
            seed=seed,
        )
        return (
            agent.select_train_action(agent.acquisition_context(order)).targets[0].key
        )

    assert choice(0, candidates) == TargetKey("useful", 1)
    assert choice(1, candidates) == TargetKey("trivia", 1)
    assert choice(1, tuple(reversed(candidates))) == TargetKey("trivia", 1)


def test_information_policy_retains_tiny_positive_channel_gain() -> None:
    useful = useful_targets(1)[0]
    agent = FactorizedQueryAgent(
        TotalInformationDirectedPolicy(),
        epsilon=0.74999999,
        query_budget=1,
    )

    action = agent.select_train_action(agent.acquisition_context((useful,)))

    assert expected_entropy_reduction(agent.posterior(useful.key)) > 0.0
    assert action.targets == (useful,)


def test_relevant_information_prefers_useful_target_to_trivia() -> None:
    useful = useful_targets(1)[0]
    trivia = distractor_targets(1)[0]
    agent = FactorizedQueryAgent(
        RelevantInformationDirectedPolicy(),
        query_budget=1,
        seed="relevant-information",
    )

    action = agent.select_train_action(agent.acquisition_context((trivia, useful)))

    assert action.targets == (useful,)


def test_budget_is_strict_and_context_rejects_invalid_actions() -> None:
    candidates = useful_targets(3)
    agent = FactorizedQueryAgent(
        FixedTargetPolicy(3),
        query_budget=1,
    )
    context = agent.acquisition_context(candidates)
    action = agent.select_train_action(context)

    assert len(action.targets) == 1
    with pytest.raises(ValueError, match="query budget"):
        context.validate_action(QueryAction(0, candidates[:2]))
    with pytest.raises(ValueError, match="agent budget"):
        agent.acquisition_context(candidates, query_budget=2)
    with pytest.raises(ValueError, match="selected acquisition action"):
        agent.observe(ObservationBatch(QueryAction(0, candidates[:2]), (1, 1)))


def test_candidate_order_and_same_seed_leave_query_trace_unchanged() -> None:
    environment = IndependentRulebook("trace-environment")
    channel = QarySymmetricChannel(q=4, epsilon=0.2)
    candidates = useful_targets(7)

    def run(
        order: Sequence[QueryTarget],
        *,
        checkpoint_each_round: bool,
    ) -> tuple[list[QueryAction], DeploymentAction]:
        agent = FactorizedQueryAgent(
            FixedTargetPolicy(7),
            epsilon=0.2,
            query_budget=2,
            seed="trace-agent",
        )
        actions = []
        for round_index in range(5):
            if checkpoint_each_round:
                agent.checkpoint()
            action = agent.select_train_action(agent.acquisition_context(order))
            observations = tuple(
                channel.observe(
                    environment.label(target.rule_index),
                    SemanticObservationKey(
                        environment.seed,
                        round_index,
                        target.rule_index,
                        ordinal,
                        target.key.namespace,
                    ),
                )
                for ordinal, target in enumerate(action.targets)
            )
            agent.observe(ObservationBatch(action, observations))
            actions.append(action)
        return actions, agent.deployment()

    forward_actions, forward_deployment = run(
        candidates,
        checkpoint_each_round=False,
    )
    reverse_actions, reverse_deployment = run(
        tuple(reversed(candidates)),
        checkpoint_each_round=True,
    )

    assert forward_actions == reverse_actions
    assert forward_deployment == reverse_deployment


def test_checkpoint_is_pure_and_deployment_snapshot_is_immutable() -> None:
    environment = IndependentRulebook("checkpoint")
    candidates = useful_targets(3)
    agent = FactorizedQueryAgent(
        FixedTargetPolicy(3),
        epsilon=0.0,
        query_budget=1,
        seed="checkpoint-agent",
    )
    first = candidates[0]
    _observe(agent, (first,), (environment.label(first.rule_index),))
    context_before = agent.acquisition_context(candidates)
    action_before = agent.select_train_action(context_before)
    counts_before = agent.posterior(first.key).observation_counts
    query_count_before = agent.query_count(first.key)

    checkpoint = agent.checkpoint()
    assert agent.checkpoint() == checkpoint
    assert environment.evaluate(checkpoint.deployment) == 1.0
    assert agent.completed_rounds == 1
    assert agent.posterior(first.key).observation_counts == counts_before
    assert agent.query_count(first.key) == query_count_before
    action_after = agent.select_train_action(agent.acquisition_context(candidates))
    assert action_after == action_before

    second = action_before.targets[0]
    _observe(agent, (second,), (environment.label(second.rule_index),))
    assert checkpoint.deployment.support == (1,)
    assert checkpoint.query_counts[first.key] == 1
    assert second.key not in checkpoint.query_counts
    assert len(agent.deployment().support) == 2


def test_objectives_match_one_query_closed_forms() -> None:
    posterior = CategoricalPosterior(q=4, epsilon=0.2)
    reward = RewardSpec()
    capacity = (
        math.log(4) + 0.2 * math.log(0.2) + 0.8 * math.log(0.8) - 0.2 * math.log(3)
    )

    assert posterior_predictive(posterior) == pytest.approx((0.25,) * 4)
    assert bayes_deployment_value(posterior, reward) == 0.0
    assert expected_post_query_bayes_value(posterior, reward) == pytest.approx(0.6)
    assert decision_value_gain(posterior, reward) == pytest.approx(0.6)
    assert expected_entropy_reduction(posterior) == pytest.approx(capacity)
    assert prediction_error_novelty(posterior) == pytest.approx(0.75)


def test_objectives_use_nonuniform_posterior_without_mutation() -> None:
    posterior = CategoricalPosterior.from_counts([2, 1, 0, 0], epsilon=0.2)
    counts = posterior.observation_counts
    predictive = posterior_predictive(posterior)

    assert math.fsum(predictive) == pytest.approx(1.0)
    assert predictive != pytest.approx((0.25,) * 4)
    assert expected_entropy_reduction(posterior) > 0.0
    assert decision_value_gain(posterior, RewardSpec()) >= 0.0
    assert posterior.observation_counts == counts


def test_comparison_contract_validation() -> None:
    with pytest.raises(TypeError, match="query_budget"):
        FactorizedQueryAgent(FixedTargetPolicy(1), query_budget=True)
    with pytest.raises(ValueError, match="epsilon"):
        FactorizedQueryAgent(FixedTargetPolicy(1), epsilon=0.75)
    with pytest.raises(ValueError, match="epsilon"):
        FactorizedQueryAgent(FixedTargetPolicy(1), epsilon=False)
    with pytest.raises(ValueError, match="target_size"):
        FixedTargetPolicy(0)
    with pytest.raises(ValueError, match="maximum_size"):
        ExpandingTargetSchedule(2, 1, maximum_size=1)
    with pytest.raises(ValueError, match="namespace"):
        TargetKey("", 1)
    with pytest.raises(ValueError, match="namespace"):
        useful_targets(0, namespace="")
    with pytest.raises(TypeError, match="persistent"):
        distractor_targets(0, persistent=1)
    with pytest.raises(ValueError, match="relevance_weight"):
        QueryTarget(TargetKey("useful", 1), relevance_weight=1.1)
    with pytest.raises(ValueError, match="unique"):
        AcquisitionContext(
            round_index=0,
            query_budget=1,
            candidates=useful_targets(1) * 2,
        )
