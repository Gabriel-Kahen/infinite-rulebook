"""Dependency-integrated symbolic adapter for the bounded smoke pilot."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from infinite_rulebook.agents import (
    FactorizedQueryAgent,
    FixedTargetPolicy,
    NoveltyDirectedPolicy,
    P1RoundTrace,
    QueryTarget,
    RewardDirectedPolicy,
    TotalInformationDirectedPolicy,
    distractor_targets,
    execute_p1_round,
    useful_targets,
)
from infinite_rulebook.agents.objectives import prediction_error_novelty
from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.environments.controls import (
    AleaRulebook,
    CappedPublicRulebook,
    ControlObservation,
    PublicBonusSchedule,
    PublicDeploymentAction,
    QueryNamespace,
    SymbolicObservation,
    SymbolicQuery,
    TriviaRulebook,
)
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.environments.mixed import MixedRulebook
from infinite_rulebook.environments.redundant import CappedRedundantRulebook
from infinite_rulebook.feedback.qary import (
    QarySymmetricChannel,
    SemanticObservationKey,
)
from infinite_rulebook.metrics import ComputeMetrics, NoveltyMetrics, SupportMetrics
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentKind,
    RunCell,
)
from infinite_rulebook.orchestration.frontiers import (
    PilotFrontier,
    build_pilot_frontier,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.records import build_checkpoint_record
from infinite_rulebook.orchestration.seeds import RunSeeds
from infinite_rulebook.orchestration.telemetry import (
    ObservationNamespace,
    ReplayQueryObservation,
    information_breakdown_from_observations,
)


@dataclass(frozen=True, slots=True)
class _RewardQueryRuntime:
    """Expose P1 reward queries for runtimes without a control wrapper."""

    base: Any

    @property
    def reward_spec(self) -> Any:
        return self.base.reward_spec

    def label(self, index: int) -> int:
        return self.base.label(index)

    def labels(self, indices: Any) -> tuple[int, ...]:
        return self.base.labels(indices)

    def evaluate(self, action: DeploymentAction) -> float:
        return self.base.evaluate(action)

    def observe_query(
        self,
        query: SymbolicQuery,
        channel: QarySymmetricChannel,
        key: SemanticObservationKey,
    ) -> ControlObservation:
        if query.namespace is not QueryNamespace.REWARD:
            raise ValueError("this environment exposes only reward queries")
        value = channel.observe(self.label(query.index), key)
        return ControlObservation(SymbolicObservation(query, value))


@dataclass(slots=True)
class PilotState:
    agent: FactorizedQueryAgent
    environment: Any
    channel: QarySymmetricChannel
    candidates: tuple[QueryTarget, ...]
    query_observation_seed: int
    observations: tuple[ReplayQueryObservation, ...] = ()
    prediction_errors: tuple[float, ...] = ()


def _public_schedule(cell: RunCell) -> PublicBonusSchedule:
    return PublicBonusSchedule((0.0, cell.environment.public_reward_cap))


def _environment(cell: RunCell, seeds: RunSeeds) -> Any:
    spec = cell.reward.to_spec()
    kind = cell.environment.kind
    if kind in {
        EnvironmentKind.IND,
        EnvironmentKind.ALEA,
        EnvironmentKind.TRIVIA,
        EnvironmentKind.PUBLIC_C,
    }:
        base: Any = IndependentRulebook(seeds.environment, spec)
    elif kind is EnvironmentKind.RED_C:
        base = CappedRedundantRulebook(
            seeds.environment,
            cell.environment.core_dimensions,
            spec,
            max_derived_support=cell.environment.max_redundant_support,
        )
    else:
        base = MixedRulebook(
            seeds.environment,
            cell.environment.core_dimensions,
            cell.environment.max_redundant_support,
            spec,
        )
    if kind is EnvironmentKind.ALEA:
        return AleaRulebook(
            base,
            seeds.aleatoric,
            cosmetic_alphabet=cell.environment.distractor_dimensions,
        )
    if kind is EnvironmentKind.TRIVIA:
        return TriviaRulebook(base, seeds.persistent_distractor)
    if kind is EnvironmentKind.PUBLIC_C:
        return CappedPublicRulebook(base, _public_schedule(cell))
    if kind in {EnvironmentKind.RED_C, EnvironmentKind.MIX}:
        return _RewardQueryRuntime(base)
    return base


def _candidates(cell: RunCell) -> tuple[QueryTarget, ...]:
    candidates = useful_targets(cell.environment.projection_size)
    if cell.environment.kind is EnvironmentKind.TRIVIA:
        candidates += distractor_targets(cell.environment.distractor_dimensions)
    return candidates


def _policy(cell: RunCell) -> Any:
    kind = cell.agent.kind
    if kind is AgentKind.FIXED:
        return FixedTargetPolicy(cell.agent.target_size)
    if kind is AgentKind.REWARD:
        return RewardDirectedPolicy()
    if kind is AgentKind.NOVELTY:
        cosmetic_alphabet = (
            cell.environment.distractor_dimensions
            if cell.environment.kind is EnvironmentKind.ALEA
            else 1
        )
        return NoveltyDirectedPolicy(cosmetic_alphabet=cosmetic_alphabet)
    return TotalInformationDirectedPolicy()


def _target_payload(target: QueryTarget) -> dict[str, Any]:
    return {
        "namespace": target.key.namespace,
        "index": target.key.index,
        "rule_index": target.rule_index,
        "relevance_weight": target.relevance_weight,
        "persistent": target.persistent,
    }


def _trace_payload(
    trace: P1RoundTrace,
    prediction_errors: tuple[float, ...],
) -> dict[str, Any]:
    observations = []
    for ordinal, (target, query, key, symbolic, cosmetic, error) in enumerate(
        zip(
            trace.action.targets,
            trace.queries,
            trace.observation_keys,
            trace.symbolic_observations,
            trace.cosmetic_values,
            prediction_errors,
            strict=True,
        )
    ):
        observations.append(
            {
                "query_ordinal": ordinal,
                "target": _target_payload(target),
                "query": {
                    "namespace": query.namespace.value,
                    "index": query.index,
                },
                "observation_key": asdict(key),
                "symbolic_value": symbolic.value,
                "cosmetic_value": cosmetic,
                "prediction_error": error,
            }
        )
    return {"round": trace.action.round_index, "observations": observations}


def _execute_round(state: PilotState) -> tuple[P1RoundTrace, tuple[float, ...]]:
    context = state.agent.acquisition_context(state.candidates)
    action = state.agent.select_train_action(context)
    errors = tuple(
        prediction_error_novelty(state.agent.posterior(target.key))
        for target in action.targets
    )
    trace = execute_p1_round(
        state.agent,
        state.environment,
        context,
        state.channel,
        environment_seed=state.query_observation_seed,
        channel_name="pilot.p1",
    )
    return trace, errors


def _append_trace(
    state: PilotState,
    trace: P1RoundTrace,
    prediction_errors: tuple[float, ...],
) -> None:
    replay = list(state.observations)
    for ordinal, (query, symbolic, cosmetic) in enumerate(
        zip(
            trace.queries,
            trace.symbolic_observations,
            trace.cosmetic_values,
            strict=True,
        )
    ):
        namespace = (
            ObservationNamespace.REWARD
            if query.namespace is QueryNamespace.REWARD
            else ObservationNamespace.TRIVIA
        )
        replay.append(
            ReplayQueryObservation(
                trace.action.round_index,
                ordinal,
                namespace,
                query.index,
                symbolic.value,
            )
        )
        if cosmetic is not None:
            replay.append(
                ReplayQueryObservation(
                    trace.action.round_index,
                    ordinal,
                    ObservationNamespace.COSMETIC,
                    query.index,
                    cosmetic,
                )
            )
    state.observations = tuple(replay)
    state.prediction_errors += prediction_errors


def _agent_state_payload(state: PilotState) -> dict[str, Any]:
    checkpoint = state.agent.checkpoint()
    return {
        "round": checkpoint.round_index,
        "deployment": [list(entry) for entry in checkpoint.deployment.entries],
        "posteriors": [
            {
                "namespace": key.namespace,
                "index": key.index,
                "probabilities": list(probabilities),
            }
            for key, probabilities in sorted(checkpoint.posterior_snapshots.items())
        ],
        "query_counts": [
            {
                "namespace": key.namespace,
                "index": key.index,
                "count": count,
            }
            for key, count in sorted(checkpoint.query_counts.items())
        ],
        "observations": [asdict(item) for item in state.observations],
        "prediction_errors": list(state.prediction_errors),
    }


def _project_deployment(
    deployment: DeploymentAction,
    cell: RunCell,
) -> DeploymentAction | PublicDeploymentAction:
    kind = cell.environment.kind
    entries = list(deployment.entries)
    if kind is EnvironmentKind.RED_C:
        entries = entries[: cell.environment.max_redundant_support]
    elif kind is EnvironmentKind.MIX:
        remaining = cell.environment.max_redundant_support
        projected = []
        for entry in entries:
            if entry[0] % 2 == 0:
                if remaining == 0:
                    continue
                remaining -= 1
            projected.append(entry)
        entries = projected
    hidden = DeploymentAction(entries)
    if kind is EnvironmentKind.PUBLIC_C:
        schedule = _public_schedule(cell)
        return PublicDeploymentAction(hidden, schedule.maximizing_choice)
    return hidden


def _hidden_deployment(
    deployment: DeploymentAction | PublicDeploymentAction,
) -> DeploymentAction:
    return (
        deployment.deployment
        if isinstance(deployment, PublicDeploymentAction)
        else deployment
    )


def _support_metrics(
    deployment: DeploymentAction | PublicDeploymentAction,
    state: PilotState,
    cell: RunCell,
) -> SupportMetrics:
    hidden = _hidden_deployment(deployment)
    correct = sum(
        prediction == state.environment.label(index) for index, prediction in hidden
    )
    independent = sum(
        prediction == state.environment.label(index)
        and (cell.environment.kind is not EnvironmentKind.MIX or index % 2 == 1)
        and cell.environment.kind is not EnvironmentKind.RED_C
        for index, prediction in hidden
    )
    return SupportMetrics(
        len(hidden),
        correct,
        len(hidden) - correct,
        cell.environment.projection_size - len(hidden),
        independent,
    )


def _novelty_metrics(
    state: PilotState,
    deployment: DeploymentAction | PublicDeploymentAction,
    cell: RunCell,
    total_information: float,
    persistent_trivia: float,
) -> NoveltyMetrics:
    symbolic = [
        item
        for item in state.observations
        if item.namespace is not ObservationNamespace.COSMETIC
    ]
    cosmetics = [
        item.value
        for item in state.observations
        if item.namespace is ObservationNamespace.COSMETIC
    ]
    keys = {(item.namespace, item.rule_index) for item in symbolic}
    reward_rules = {
        item.rule_index
        for item in symbolic
        if item.namespace is ObservationNamespace.REWARD
    }
    return NoveltyMetrics(
        math.fsum(state.prediction_errors) / max(1, len(state.prediction_errors)),
        total_information,
        len(keys) / max(1, len(symbolic)),
        len(reward_rules) / cell.environment.projection_size,
        len(_hidden_deployment(deployment)) / cell.environment.projection_size,
        len(set(cosmetics)) / max(1, len(cosmetics)),
        persistent_trivia,
    )


@dataclass(slots=True)
class ExactSymbolicAdapter:
    """Execute all registered smoke-pilot conditions against merged APIs."""

    _frontiers: dict[str, PilotFrontier] = field(default_factory=dict)

    def initial_state(self, cell: RunCell, seeds: RunSeeds) -> PilotState:
        environment = _environment(cell, seeds)
        return PilotState(
            agent=FactorizedQueryAgent(
                _policy(cell),
                q=cell.reward.q,
                epsilon=cell.feedback.epsilon,
                query_budget=cell.feedback.query_budget,
                seed=seeds.algorithm,
                reward_spec=cell.reward.to_spec(),
            ),
            environment=environment,
            channel=QarySymmetricChannel(
                cell.reward.q,
                cell.feedback.epsilon,
            ),
            candidates=_candidates(cell),
            query_observation_seed=seeds.query_observation,
        )

    def training_event(
        self,
        state: PilotState,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> dict[str, Any]:
        del cell, seeds
        if state.agent.completed_rounds != round_index:
            raise ValueError("training round does not match replayed agent state")
        clone = copy.deepcopy(state)
        trace, errors = _execute_round(clone)
        return _trace_payload(trace, errors)

    def apply_training_event(
        self,
        state: PilotState,
        payload: dict[str, Any],
    ) -> PilotState:
        trace, errors = _execute_round(state)
        if _trace_payload(trace, errors) != payload:
            raise ValueError(
                "persisted training event does not match deterministic replay"
            )
        _append_trace(state, trace, errors)
        return state

    def checkpoint(
        self,
        state: PilotState,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
        semantic_hashes: dict[str, str],
    ) -> dict[str, Any]:
        deployment = _project_deployment(state.agent.deployment(), cell)
        reward = state.environment.evaluate(deployment)
        breakdown = information_breakdown_from_observations(
            state.observations,
            environment=cell.environment.kind,
            q=cell.reward.q,
            epsilon=cell.feedback.epsilon,
            core_dimensions=cell.environment.core_dimensions,
        )
        support = _support_metrics(deployment, state, cell)
        novelty = _novelty_metrics(
            state,
            deployment,
            cell,
            breakdown.total_acquired_nats,
            breakdown.persistent_distractor_nats,
        )
        compute = ComputeMetrics(
            queries=len(
                [
                    item
                    for item in state.observations
                    if item.namespace is not ObservationNamespace.COSMETIC
                ]
            ),
            environment_steps=round_index,
            posterior_updates=sum(state.agent.checkpoint().query_counts.values()),
            frontier_solver_calls=3,
            deployment_evaluations=1,
        )
        frontier = self._frontiers.get(cell.cell_hash)
        if frontier is None:
            raise RuntimeError("checkpoint evaluation requires a prepared frontier")
        scientific_records = build_checkpoint_record(
            semantic_hashes=semantic_hashes,
            round_index=round_index,
            reward_sample=reward,
            information=breakdown,
            deployment=deployment,
            deployment_seed=seeds.deployment,
            novelty=novelty,
            support=support,
            compute=compute,
            frontier=frontier.curve,
        )
        return {
            "expected_reward": reward,
            "deployment": (
                {
                    "entries": [list(entry) for entry in deployment.deployment],
                    "public_choice": deployment.public_choice,
                }
                if isinstance(deployment, PublicDeploymentAction)
                else [list(entry) for entry in deployment]
            ),
            "support": asdict(support),
            "novelty": asdict(novelty),
            "information": asdict(breakdown),
            "compute": asdict(compute),
            "scientific_records": scientific_records,
            "round": round_index,
            "evaluation": "exact-no-feedback",
            "action_sample_count": 0,
            "deployment_seed": seeds.deployment,
        }

    def state_fingerprint(self, state: PilotState) -> str:
        return scientific_hash(
            _agent_state_payload(state),
            domain="symbolic-agent-state",
        )

    def frontier(self, cell: RunCell) -> dict[str, Any]:
        result = build_pilot_frontier(cell)
        self._frontiers[cell.cell_hash] = result
        return result.bundle
