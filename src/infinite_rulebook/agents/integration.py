"""Adapters from factorized agents to symbolic controls and information ledgers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from infinite_rulebook.agents.comparison import NoveltyDirectedPolicy
from infinite_rulebook.agents.protocols import (
    AcquisitionContext,
    AgentCheckpoint,
    ObservationBatch,
    QueryAction,
    QueryTarget,
    SymbolicAgent,
    TargetKey,
)
from infinite_rulebook.core.rng import Seed
from infinite_rulebook.environments.controls import (
    AleaRulebook,
    CappedPublicRulebook,
    ControlObservation,
    QueryNamespace,
    RulebookRuntime,
    SymbolicObservation,
    SymbolicQuery,
    TriviaRulebook,
    UnboundedPublicRulebook,
)
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.frontier import OneCoordinateFrontier
from infinite_rulebook.information import (
    InformationBreakdown,
    InformationCategory,
    InformationLedger,
    LatentAxis,
    PosteriorBlock,
    SurfaceDependency,
)
from infinite_rulebook.metrics import MetricInterval, PopulationInformationEstimate


def target_to_symbolic_query(target: QueryTarget) -> SymbolicQuery:
    """Convert a validated persistent agent target to a control query."""

    if not isinstance(target, QueryTarget):
        raise TypeError("target must be a QueryTarget")
    if not target.persistent:
        raise ValueError("symbolic queries must target persistent coordinates")
    if target.relevance_weight not in (0.0, 1.0):
        raise ValueError("symbolic targets require binary relevance")
    if target.key.namespace == QueryNamespace.REWARD.value:
        if target.relevance_weight != 1.0 or target.rule_index != target.key.index:
            raise ValueError(
                "reward targets require unit relevance and matching rule index"
            )
        namespace = QueryNamespace.REWARD
    elif target.key.namespace == QueryNamespace.TRIVIA.value:
        if target.relevance_weight != 0.0 or target.rule_index is not None:
            raise ValueError("trivia targets must be irrelevant and nondeployable")
        namespace = QueryNamespace.TRIVIA
    else:
        raise ValueError("target namespace must be 'reward' or 'trivia'")
    return SymbolicQuery(namespace, target.key.index)


def symbolic_query_to_target(query: SymbolicQuery) -> QueryTarget:
    """Convert a control query to canonical factorized-agent metadata."""

    if not isinstance(query, SymbolicQuery):
        raise TypeError("query must be a SymbolicQuery")
    key = TargetKey(query.namespace.value, query.index)
    if query.namespace is QueryNamespace.REWARD:
        return QueryTarget(key, rule_index=query.index)
    return QueryTarget(key, rule_index=None, relevance_weight=0.0)


@dataclass(frozen=True, slots=True)
class P1RoundTrace:
    """Immutable audit trace for one bounded symbolic P1 round."""

    action: QueryAction
    queries: tuple[SymbolicQuery, ...]
    observation_keys: tuple[SemanticObservationKey, ...]
    symbolic_observations: tuple[SymbolicObservation, ...]
    cosmetic_values: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action, QueryAction):
            raise TypeError("action must be a QueryAction")
        queries = tuple(self.queries)
        keys = tuple(self.observation_keys)
        observations = tuple(self.symbolic_observations)
        cosmetics = tuple(self.cosmetic_values)
        expected = len(self.action.targets)
        if any(
            len(values) != expected
            for values in (queries, keys, observations, cosmetics)
        ):
            raise ValueError("trace fields must contain one item per query target")
        for target, query, key, observation, cosmetic in zip(
            self.action.targets,
            queries,
            keys,
            observations,
            cosmetics,
            strict=True,
        ):
            if target_to_symbolic_query(target) != query:
                raise ValueError("trace query does not match its agent target")
            if not isinstance(key, SemanticObservationKey):
                raise TypeError(
                    "observation keys must be SemanticObservationKey records"
                )
            if (
                key.round_index != self.action.round_index
                or key.rule_index != query.index
                or key.query_ordinal != 0
            ):
                raise ValueError("observation key does not match its semantic query")
            if not isinstance(observation, SymbolicObservation):
                raise TypeError(
                    "symbolic observations must be SymbolicObservation records"
                )
            if observation.query != query:
                raise ValueError("symbolic observation does not match its query")
            if (
                isinstance(observation.value, bool)
                or not isinstance(observation.value, int)
                or observation.value < 1
            ):
                raise ValueError(
                    "symbolic observation values must be positive integers"
                )
            if cosmetic is not None and (
                isinstance(cosmetic, bool)
                or not isinstance(cosmetic, int)
                or cosmetic < 0
            ):
                raise ValueError("cosmetic values must be nonnegative integers or None")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "observation_keys", keys)
        object.__setattr__(self, "symbolic_observations", observations)
        object.__setattr__(self, "cosmetic_values", cosmetics)

    @property
    def symbolic_values(self) -> tuple[int, ...]:
        return tuple(observation.value for observation in self.symbolic_observations)


def execute_p1_round(
    agent: SymbolicAgent,
    environment: RulebookRuntime[object],
    context: AcquisitionContext,
    channel: QarySymmetricChannel,
    *,
    environment_seed: Seed,
    channel_name: str = "p1",
) -> P1RoundTrace:
    """Execute one bounded P1 action without mixing cosmetic and symbolic data."""

    if not isinstance(agent, SymbolicAgent):
        raise TypeError("agent must implement SymbolicAgent")
    if not isinstance(environment, RulebookRuntime):
        raise TypeError("environment must implement RulebookRuntime")
    if not isinstance(context, AcquisitionContext):
        raise TypeError("context must be an AcquisitionContext")
    if not isinstance(channel, QarySymmetricChannel):
        raise TypeError("channel must be a QarySymmetricChannel")
    if channel.q != environment.reward_spec.q:
        raise ValueError("channel alphabet must match environment reward alphabet")
    if getattr(agent, "q", channel.q) != channel.q:
        raise ValueError("agent, channel, and environment alphabets must match")
    if getattr(agent, "epsilon", channel.epsilon) != channel.epsilon:
        raise ValueError("agent and channel epsilon values must match")
    if (
        getattr(agent, "reward_spec", environment.reward_spec)
        != environment.reward_spec
    ):
        raise ValueError("agent and environment reward specifications must match")
    policy = getattr(agent, "policy", None)
    if isinstance(policy, NoveltyDirectedPolicy):
        cosmetic_alphabet = _cosmetic_alphabet(environment)
        if policy.cosmetic_alphabet != cosmetic_alphabet:
            raise ValueError(
                "novelty policy cosmetic alphabet must match the environment"
            )
    if isinstance(environment_seed, bool) or not isinstance(
        environment_seed,
        int | str | bytes,
    ):
        raise TypeError("environment_seed must be an integer, string, or bytes")
    if not isinstance(channel_name, str) or not channel_name:
        raise ValueError("channel_name must be a nonempty string")

    # Reject malformed adapter metadata before the agent records a pending action.
    candidate_queries = tuple(
        target_to_symbolic_query(target) for target in context.candidates
    )
    _validate_query_support(environment, candidate_queries)
    action = agent.select_train_action(context)
    context.validate_action(action)
    queries = tuple(target_to_symbolic_query(target) for target in action.targets)
    keys = tuple(
        SemanticObservationKey(
            environment_seed=environment_seed,
            round_index=action.round_index,
            rule_index=query.index,
            query_ordinal=0,
            channel=channel_name,
        )
        for query in queries
    )
    controls = tuple(
        _observe_control(environment, query, channel, key)
        for query, key in zip(queries, keys, strict=True)
    )
    trace = P1RoundTrace(
        action=action,
        queries=queries,
        observation_keys=keys,
        symbolic_observations=tuple(item.symbolic for item in controls),
        cosmetic_values=tuple(item.cosmetic_value for item in controls),
    )
    agent.observe(
        ObservationBatch(
            action,
            trace.symbolic_values,
            trace.cosmetic_values,
        )
    )
    return trace


def _cosmetic_alphabet(environment: RulebookRuntime[object]) -> int:
    if isinstance(environment, AleaRulebook):
        return environment.cosmetic_alphabet
    if isinstance(environment, (CappedPublicRulebook, UnboundedPublicRulebook)):
        return _cosmetic_alphabet(environment.base)
    return 1


def _validate_query_support(
    environment: RulebookRuntime[object],
    queries: Sequence[SymbolicQuery],
) -> None:
    if isinstance(environment, TriviaRulebook):
        return
    if isinstance(
        environment,
        (AleaRulebook, CappedPublicRulebook, UnboundedPublicRulebook),
    ):
        _validate_query_support(environment.base, queries)
        return
    if isinstance(environment, IndependentRulebook):
        if any(query.namespace is not QueryNamespace.REWARD for query in queries):
            raise ValueError("bare IndependentRulebook does not expose trivia queries")
        return
    if not callable(getattr(environment, "observe_query", None)):
        raise TypeError("environment does not expose symbolic queries")


def _observe_control(
    environment: RulebookRuntime[object],
    query: SymbolicQuery,
    channel: QarySymmetricChannel,
    key: SemanticObservationKey,
) -> ControlObservation:
    observer = getattr(environment, "observe_query", None)
    if callable(observer):
        observation = observer(query, channel, key)
        if not isinstance(observation, ControlObservation):
            raise TypeError("environment.observe_query must return ControlObservation")
        return observation
    if not isinstance(environment, IndependentRulebook):
        raise TypeError(
            "environments without observe_query must be bare IndependentRulebook"
        )
    if query.namespace is not QueryNamespace.REWARD:
        raise ValueError("bare IndependentRulebook exposes only reward queries")
    value = channel.observe(environment.label(query.index), key)
    return ControlObservation(SymbolicObservation(query, value))


def factorized_information_ledger(
    checkpoint: AgentCheckpoint,
    target_registry: Mapping[TargetKey, QueryTarget],
) -> InformationLedger:
    """Build an exact ledger for independent reward and trivia coordinates."""

    if not isinstance(checkpoint, AgentCheckpoint):
        raise TypeError("checkpoint must be an AgentCheckpoint")
    if not isinstance(target_registry, Mapping):
        raise TypeError("target_registry must be a mapping")

    registry: dict[TargetKey, QueryTarget] = {}
    rule_owners: dict[int, TargetKey] = {}
    for key, target in target_registry.items():
        if not isinstance(key, TargetKey) or not isinstance(target, QueryTarget):
            raise TypeError("target_registry must map TargetKey to QueryTarget")
        if target.key != key:
            raise ValueError("target registry keys must match target metadata")
        if target.relevance_weight not in (0.0, 1.0):
            raise ValueError("factorized ledger requires binary relevance")
        if target.persistent:
            target_to_symbolic_query(target)
            if target.rule_index is not None:
                owner = rule_owners.setdefault(target.rule_index, key)
                if owner != key:
                    raise ValueError("one reward rule cannot have multiple targets")
        registry[key] = target

    recorded = set(checkpoint.posterior_snapshots) | set(checkpoint.query_counts)
    missing = recorded - set(registry)
    if missing:
        raise ValueError(
            f"target registry is missing recorded keys: {sorted(missing)!r}"
        )

    blocks = []
    for key, posterior in sorted(checkpoint.posterior_snapshots.items()):
        target = registry[key]
        if not target.persistent:
            raise ValueError("nonpersistent targets cannot have posterior snapshots")
        count = checkpoint.query_counts.get(key, 0)
        if count < 1:
            raise ValueError("posterior snapshots require a positive query count")
        query = target_to_symbolic_query(target)
        latent_id = f"{query.namespace.value}:{query.index}"
        if query.namespace is QueryNamespace.REWARD:
            category = InformationCategory.REWARD_RELEVANT
            dependencies = (
                SurfaceDependency(f"rule:{target.rule_index}", (latent_id,)),
            )
        else:
            category = InformationCategory.PERSISTENT_DISTRACTOR
            dependencies = ()
        cardinality = len(posterior)
        blocks.append(
            PosteriorBlock(
                block_id=latent_id,
                axes=(LatentAxis(latent_id, cardinality, category),),
                prior=(1.0 / cardinality,) * cardinality,
                posterior=tuple(posterior),
                surface_dependencies=dependencies,
            )
        )

    for key, count in checkpoint.query_counts.items():
        target = registry[key]
        if (
            target.persistent
            and count > 0
            and key not in checkpoint.posterior_snapshots
        ):
            raise ValueError("queried persistent targets require posterior snapshots")
    ledger = InformationLedger.from_blocks(blocks)
    if not ledger.validate().valid:
        raise ValueError("factorized information ledger failed validation")
    return ledger


def population_information_estimate(
    histories: Sequence[InformationBreakdown],
) -> PopulationInformationEstimate:
    """Average complete-history ledger outcomes into a population estimand."""

    values = tuple(histories)
    if not values:
        raise ValueError("histories must not be empty")
    if any(not isinstance(value, InformationBreakdown) for value in values):
        raise TypeError("histories must contain InformationBreakdown records")
    if any(
        not value.reconciles() or value.approximation_residual_nats != 0.0
        for value in values
    ):
        raise ValueError(
            "population estimates require reconciled histories without residuals"
        )
    count = len(values)
    return PopulationInformationEstimate(
        reward_relevant_nats=math.fsum(value.reward_relevant_nats for value in values)
        / count,
        shared_core_nats=math.fsum(value.shared_core_nats for value in values) / count,
        persistent_distractor_nats=math.fsum(
            value.persistent_distractor_nats for value in values
        )
        / count,
        dynamic_state_nats=math.fsum(value.dynamic_state_nats for value in values)
        / count,
        total_nats=math.fsum(value.total_acquired_nats for value in values) / count,
        run_count=count,
    )


def exact_ind_bit_equivalent(
    expected_reward: float,
    *,
    q: int = 4,
    u: float = 1.0,
    c: float = 1.0,
) -> MetricInterval:
    """Return the registered exact infinite-IND bit-equivalent interval."""

    value = OneCoordinateFrontier(q=q, u=u, c=c).infinite_bit_equivalent(
        expected_reward
    )
    return MetricInterval(value, value, "nats")
