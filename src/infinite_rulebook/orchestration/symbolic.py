"""Dependency-integrated symbolic adapter for the bounded smoke pilot."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

from infinite_rulebook.agents import (
    ExpandingTargetSchedule,
    FactorizedQueryAgent,
    FixedTargetPolicy,
    NoveltyDirectedPolicy,
    P1RoundTrace,
    QueryTarget,
    RelevantInformationDirectedPolicy,
    RewardDirectedPolicy,
    ScheduledTargetPolicy,
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
    SYMBOLIC_ADAPTER_CONTRACT_V1,
    SYMBOLIC_ADAPTER_CONTRACT_V2,
    AgentKind,
    EnvironmentKind,
    RunCell,
)
from infinite_rulebook.orchestration.frontiers import build_pilot_frontier
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


@dataclass(slots=True)
class PilotStateV2(PilotState):
    post_query_hidden_expected_rewards: tuple[float, ...] = ()
    cell: RunCell | None = None


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
    if kind is AgentKind.SCHEDULED:
        assert cell.agent.growth_step is not None
        assert cell.agent.growth_interval is not None
        assert cell.agent.maximum_size is not None
        return ScheduledTargetPolicy(
            ExpandingTargetSchedule(
                initial_size=cell.agent.target_size,
                growth_step=cell.agent.growth_step,
                growth_interval=cell.agent.growth_interval,
                maximum_size=cell.agent.maximum_size,
            )
        )
    if kind is AgentKind.REWARD:
        return RewardDirectedPolicy()
    if kind is AgentKind.NOVELTY:
        cosmetic_alphabet = (
            cell.environment.distractor_dimensions
            if cell.environment.kind is EnvironmentKind.ALEA
            else 1
        )
        return NoveltyDirectedPolicy(cosmetic_alphabet=cosmetic_alphabet)
    if kind is AgentKind.TOTAL_INFORMATION:
        return TotalInformationDirectedPolicy()
    if kind is AgentKind.RELEVANT_INFORMATION:
        return RelevantInformationDirectedPolicy()
    raise ValueError(f"unsupported symbolic agent kind: {kind}")


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
    symbolic_errors = {
        target.key: prediction_error_novelty(state.agent.posterior(target.key))
        for target in context.candidates
    }
    cosmetic_alphabet = (
        state.environment.cosmetic_alphabet
        if isinstance(state.environment, AleaRulebook)
        else 1
    )
    trace = execute_p1_round(
        state.agent,
        state.environment,
        context,
        state.channel,
        environment_seed=state.query_observation_seed,
        channel_name="pilot.p1",
    )
    errors = tuple(
        1.0 - (1.0 - symbolic_errors[target.key]) / cosmetic_alphabet
        for target in trace.action.targets
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


def _reward_components(
    environment: Any,
    deployment: DeploymentAction | PublicDeploymentAction,
) -> tuple[float, float, float]:
    total = environment.evaluate(deployment)
    if isinstance(deployment, PublicDeploymentAction):
        hidden = environment.base.evaluate(deployment.deployment)
        return total, hidden, total - hidden
    return total, total, 0.0


def recompute_reward_components(
    cell: RunCell,
    seeds: RunSeeds,
    deployment: DeploymentAction | PublicDeploymentAction,
) -> tuple[float, float, float]:
    """Recompute registered total, hidden, and public checkpoint rewards."""

    return _reward_components(_environment(cell, seeds), deployment)


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

    contract_version = SYMBOLIC_ADAPTER_CONTRACT_V1

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
        reward, hidden_reward, public_reward = _reward_components(
            state.environment,
            deployment,
        )
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
            frontier_solver_calls=cell.solver.reward_grid_points,
            deployment_evaluations=1,
        )
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
        )
        return {
            "expected_reward": reward,
            "hidden_expected_reward": hidden_reward,
            "public_reward": public_reward,
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
            "agent_capabilities": asdict(state.agent.capabilities),
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
        return build_pilot_frontier(cell).bundle


@dataclass(slots=True)
class ExactSymbolicAdapterV2(ExactSymbolicAdapter):
    """Authenticate the v2 post-query hidden-reward trajectory."""

    contract_version = SYMBOLIC_ADAPTER_CONTRACT_V2

    def initial_state(self, cell: RunCell, seeds: RunSeeds) -> PilotStateV2:
        state = ExactSymbolicAdapter.initial_state(self, cell, seeds)
        return PilotStateV2(
            agent=state.agent,
            environment=state.environment,
            channel=state.channel,
            candidates=state.candidates,
            query_observation_seed=state.query_observation_seed,
            observations=state.observations,
            prediction_errors=state.prediction_errors,
            cell=cell,
        )

    def training_event(
        self,
        state: PilotState,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> dict[str, Any]:
        del seeds
        if not isinstance(state, PilotStateV2):
            raise TypeError("v2 execution requires PilotStateV2")
        if state.cell != cell:
            raise ValueError("v2 state belongs to a different run cell")
        if state.agent.completed_rounds != round_index:
            raise ValueError("training round does not match replayed agent state")
        clone = copy.deepcopy(state)
        trace, errors = _execute_round(clone)
        deployment = _project_deployment(clone.agent.deployment(), cell)
        _, hidden_reward, _ = _reward_components(clone.environment, deployment)
        return {
            **_trace_payload(trace, errors),
            "post_query_hidden_expected_reward": hidden_reward,
        }

    def apply_training_event(
        self,
        state: PilotState,
        payload: dict[str, Any],
    ) -> PilotStateV2:
        if not isinstance(state, PilotStateV2):
            raise TypeError("v2 execution requires PilotStateV2")
        if state.cell is None:
            raise ValueError("v2 state is missing its run cell")
        trace, errors = _execute_round(state)
        deployment = _project_deployment(
            state.agent.deployment(),
            state.cell,
        )
        _, hidden_reward, _ = _reward_components(state.environment, deployment)
        expected = {
            **_trace_payload(trace, errors),
            "post_query_hidden_expected_reward": hidden_reward,
        }
        if expected != payload:
            raise ValueError(
                "persisted training event does not match deterministic replay"
            )
        _append_trace(state, trace, errors)
        state.post_query_hidden_expected_rewards += (hidden_reward,)
        return state

    def checkpoint(
        self,
        state: PilotState,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
        semantic_hashes: dict[str, str],
    ) -> dict[str, Any]:
        if not isinstance(state, PilotStateV2):
            raise TypeError("v2 execution requires PilotStateV2")
        if state.cell != cell:
            raise ValueError("v2 state belongs to a different run cell")
        if (
            state.agent.completed_rounds != round_index
            or len(state.post_query_hidden_expected_rewards) != round_index
        ):
            raise ValueError("checkpoint round does not match v2 reward history")
        result = ExactSymbolicAdapter.checkpoint(
            self,
            state,
            round_index,
            cell,
            seeds,
            semantic_hashes,
        )
        if round_index > 0:
            result["post_query_mean_hidden_expected_reward"] = (
                math.fsum(state.post_query_hidden_expected_rewards) / round_index
            )
        return result

    def state_fingerprint(self, state: PilotState) -> str:
        if not isinstance(state, PilotStateV2):
            raise TypeError("v2 execution requires PilotStateV2")
        return scientific_hash(
            {
                **_agent_state_payload(state),
                "post_query_hidden_expected_rewards": list(
                    state.post_query_hidden_expected_rewards
                ),
            },
            domain="symbolic-agent-state.v2",
        )


def exact_symbolic_adapter_class(
    contract_version: str,
) -> type[ExactSymbolicAdapter] | type[ExactSymbolicAdapterV2]:
    """Return the only exact adapter registered for a recorded contract."""

    if contract_version == SYMBOLIC_ADAPTER_CONTRACT_V1:
        return ExactSymbolicAdapter
    if contract_version == SYMBOLIC_ADAPTER_CONTRACT_V2:
        return ExactSymbolicAdapterV2
    raise ValueError(
        f"unregistered exact symbolic adapter contract: {contract_version}"
    )
