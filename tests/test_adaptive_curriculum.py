"""Synthetic contracts for deterministic, non-registered curricula."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

import pytest

from infinite_rulebook.agents import (
    CandidateGroupDiscoveryPolicy,
    CapabilityManifest,
    CurriculumAcquisitionPolicy,
    CurriculumAction,
    CurriculumAssessment,
    CurriculumCatalog,
    CurriculumController,
    CurriculumDecision,
    CurriculumEvidence,
    CurriculumLimits,
    CurriculumReason,
    CurriculumState,
    CurriculumTarget,
    CurriculumUpdate,
    EstimatedFrontierPolicy,
    FactorizedQueryAgent,
    FrontierEstimate,
    MarginalValuePerBitPolicy,
    ObservationBatch,
    OracleFrontierPolicy,
    TargetKey,
    useful_targets,
)
from infinite_rulebook.artifacts import semantic_hash
from infinite_rulebook.environments.independent import IndependentRulebook
from infinite_rulebook.orchestration.hashing import (
    canonical_json_bytes,
    scientific_hash,
)


def _target(
    name: str,
    *indices: int,
    parent: str | None = None,
) -> CurriculumTarget:
    return CurriculumTarget(
        name,
        tuple(TargetKey("reward", index) for index in indices),
        parent,
    )


def _assessment(
    target: str,
    *,
    evidence: CurriculumEvidence = CurriculumEvidence.POSTERIOR,
    reward: tuple[float, float] = (0.5, 0.5),
    information: tuple[float, float] = (1.0, 1.0),
    probe_nats: float = 0.0,
    count: int = 0,
) -> CurriculumAssessment:
    return CurriculumAssessment(
        target=target,
        evidence=evidence,
        marginal_reward_lower=reward[0],
        marginal_reward_upper=reward[1],
        information_nats_lower=information[0],
        information_nats_upper=information[1],
        probe_information_nats=probe_nats,
        evidence_count=count,
    )


def _frontier(
    evidence: CurriculumEvidence,
    *,
    gap: float = 0.0,
    residual: float = 0.0,
    width: float = 0.0,
) -> FrontierEstimate:
    return FrontierEstimate(evidence, gap, residual, width)


def _controller(
    catalog: CurriculumCatalog,
    policy: object,
    *,
    support: int = 16,
    information: float = 16.0,
) -> CurriculumController:
    return CurriculumController(
        catalog,
        policy,  # type: ignore[arg-type]
        CurriculumLimits(support, information),
    )


def test_catalog_canonicalizes_members_and_rejects_hierarchy_cycles() -> None:
    catalog = CurriculumCatalog(
        (_target("root", 2, 1), _target("next", 3, parent="root"))
    )

    assert catalog.target("root").members == (
        TargetKey("reward", 1),
        TargetKey("reward", 2),
    )
    with pytest.raises(ValueError, match="acyclic"):
        CurriculumCatalog(
            (
                _target("left", 1, parent="right"),
                _target("right", 2, parent="left"),
            )
        )


def test_catalog_rejects_stranding_overlap_but_allows_safe_partial_overlap() -> None:
    with pytest.raises(ValueError, match="overlap can eliminate"):
        CurriculumCatalog(
            (
                _target("peer", 1),
                _target("parent", 1),
                _target("child", 2, parent="parent"),
            )
        )

    catalog = CurriculumCatalog(
        (
            _target("left", 1, 2),
            _target("right", 2, 3),
        )
    )
    assert catalog.target("left").members == (
        TargetKey("reward", 1),
        TargetKey("reward", 2),
    )


def test_oracle_waits_for_gap_then_expands_exact_hierarchy_deterministically() -> None:
    catalog = CurriculumCatalog(
        (
            _target("root", 1),
            _target("next-a", 2, parent="root"),
            _target("next-b", 3, parent="root"),
        )
    )
    controller = _controller(
        catalog,
        OracleFrontierPolicy(
            frontier_gap_threshold=0.05,
            minimum_value_per_nat=0.1,
        ),
    )
    oracle = CurriculumEvidence.ORACLE

    held = controller.update(
        CurriculumUpdate(
            0,
            (_assessment("root", evidence=oracle),),
            _frontier(oracle, gap=0.1),
        )
    )
    root = controller.update(
        CurriculumUpdate(
            1,
            (_assessment("root", evidence=oracle),),
            _frontier(oracle, gap=0.01),
        )
    )
    tied = (
        _assessment("next-b", evidence=oracle, reward=(0.4, 0.4)),
        _assessment("next-a", evidence=oracle, reward=(0.4, 0.4)),
    )
    child = controller.update(CurriculumUpdate(2, tied, _frontier(oracle, gap=0.0)))

    assert held.reason is CurriculumReason.FRONTIER_GAP_OPEN
    assert root.action is CurriculumAction.EXPAND
    assert root.target == "root"
    assert child.target == "next-a"
    assert controller.state.active_targets == ("next-a", "root")
    assert controller.state.active_members == (
        TargetKey("reward", 1),
        TargetKey("reward", 2),
    )
    assert controller.capabilities.knows_exact_frontier


def test_oracle_rejects_uncertain_or_nonoracle_inputs() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    controller = _controller(catalog, OracleFrontierPolicy(0.1))

    with pytest.raises(ValueError, match="oracle estimates must be exact"):
        controller.update(
            CurriculumUpdate(
                0,
                (
                    _assessment(
                        "root",
                        evidence=CurriculumEvidence.ORACLE,
                        reward=(0.2, 0.3),
                    ),
                ),
                _frontier(CurriculumEvidence.ORACLE),
            )
        )


def test_estimated_frontier_requires_narrow_bounds_and_low_residual_value() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    controller = _controller(
        catalog,
        EstimatedFrontierPolicy(
            residual_value_threshold=0.1,
            bound_width_tolerance=0.05,
            minimum_value_per_nat=0.1,
        ),
    )
    estimate = _assessment(
        "root",
        reward=(0.3, 0.34),
        information=(1.0, 1.2),
    )

    wide = controller.update(
        CurriculumUpdate(
            0,
            (estimate,),
            _frontier(CurriculumEvidence.POSTERIOR, residual=0.05, width=0.1),
        )
    )
    unresolved = controller.update(
        CurriculumUpdate(
            1,
            (estimate,),
            _frontier(CurriculumEvidence.POSTERIOR, residual=0.2, width=0.01),
        )
    )
    expanded = controller.update(
        CurriculumUpdate(
            2,
            (estimate,),
            _frontier(CurriculumEvidence.POSTERIOR, residual=0.05, width=0.01),
        )
    )

    assert wide.reason is CurriculumReason.UNCERTAINTY_TOO_WIDE
    assert unresolved.reason is CurriculumReason.RESIDUAL_VALUE_OPEN
    assert expanded.action is CurriculumAction.EXPAND
    assert expanded.planned_information_nats == 1.2
    assert not controller.capabilities.knows_exact_frontier
    assert controller.capabilities.knows_approximate_frontier


def test_marginal_value_per_nat_uses_conservative_bounds_not_point_order() -> None:
    catalog = CurriculumCatalog((_target("large", 1), _target("small", 2)))
    controller = _controller(
        catalog,
        MarginalValuePerBitPolicy(minimum_value_per_nat=0.1),
    )
    decision = controller.update(
        CurriculumUpdate(
            0,
            (
                _assessment(
                    "large",
                    reward=(0.5, 0.9),
                    information=(1.0, 2.0),
                ),
                _assessment(
                    "small",
                    reward=(0.3, 0.31),
                    information=(0.8, 1.0),
                ),
            ),
        )
    )

    assert decision.target == "small"
    assert decision.score == pytest.approx(0.3)


def test_marginal_value_policy_default_round_trips_and_hashes_canonically() -> None:
    policy = MarginalValuePerBitPolicy()
    payload = {
        item.name: getattr(policy, item.name) for item in fields(policy) if item.init
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    restored = MarginalValuePerBitPolicy(**json.loads(encoded))

    assert policy.bound_width_tolerance is None
    assert restored == policy
    assert canonical_json_bytes(restored) == canonical_json_bytes(policy)
    assert semantic_hash(restored) == semantic_hash(policy)
    assert scientific_hash(restored, domain="curriculum-policy") == scientific_hash(
        policy,
        domain="curriculum-policy",
    )
    with pytest.raises(ValueError, match="must be finite"):
        MarginalValuePerBitPolicy(bound_width_tolerance=float("inf"))


def test_marginal_value_policy_applies_an_optional_finite_width_gate() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    controller = _controller(
        catalog,
        MarginalValuePerBitPolicy(bound_width_tolerance=0.1),
    )

    decision = controller.update(
        CurriculumUpdate(
            0,
            (_assessment("root", reward=(0.2, 0.4)),),
        )
    )

    assert decision.reason is CurriculumReason.UNCERTAINTY_TOO_WIDE


def test_support_information_and_hierarchy_limits_fail_closed() -> None:
    catalog = CurriculumCatalog(
        (_target("root", 1, 2), _target("child", 3, parent="root"))
    )
    controller = _controller(
        catalog,
        MarginalValuePerBitPolicy(),
        support=1,
        information=0.5,
    )

    limited = controller.update(
        CurriculumUpdate(
            0,
            (_assessment("root", information=(1.0, 1.0)),),
        )
    )
    child_first = controller.update(
        CurriculumUpdate(
            1,
            (_assessment("child", information=(0.1, 0.1)),),
        )
    )

    assert limited.reason is CurriculumReason.BUDGET_OR_SUPPORT_LIMIT
    assert child_first.reason is CurriculumReason.NO_ELIGIBLE_TARGET
    assert controller.state.active_targets == ()
    assert controller.state.planned_information_nats == 0.0


def test_evidence_counts_cannot_regress_across_updates() -> None:
    catalog = CurriculumCatalog((_target("root", 1, 2),))
    controller = _controller(
        catalog,
        MarginalValuePerBitPolicy(),
        support=1,
    )
    controller.update(CurriculumUpdate(0, (_assessment("root", count=2),)))

    with pytest.raises(ValueError, match="cannot decrease"):
        controller.update(CurriculumUpdate(1, (_assessment("root", count=1),)))


def test_candidate_group_policy_probes_every_group_before_expansion() -> None:
    catalog = CurriculumCatalog((_target("alpha", 1), _target("beta", 2)))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(
            minimum_evidence=1,
            bound_width_tolerance=0.1,
            minimum_value_per_nat=0.05,
            maximum_probes_per_target=1,
        ),
    )
    discovery = CurriculumEvidence.DISCOVERY

    first = controller.update(
        CurriculumUpdate(
            0,
            (
                _assessment("beta", evidence=discovery, probe_nats=0.2),
                _assessment("alpha", evidence=discovery, probe_nats=0.2),
            ),
        )
    )
    second = controller.update(
        CurriculumUpdate(
            1,
            (
                _assessment(
                    "alpha",
                    evidence=discovery,
                    probe_nats=0.2,
                    count=1,
                ),
                _assessment("beta", evidence=discovery, probe_nats=0.2),
            ),
        )
    )
    expanded = controller.update(
        CurriculumUpdate(
            2,
            (
                _assessment(
                    "alpha",
                    evidence=discovery,
                    reward=(0.3, 0.35),
                    information=(1.0, 1.2),
                    probe_nats=0.2,
                    count=1,
                ),
                _assessment(
                    "beta",
                    evidence=discovery,
                    reward=(-0.1, -0.05),
                    probe_nats=0.2,
                    count=1,
                ),
            ),
        )
    )

    assert (first.action, first.target) == (CurriculumAction.PROBE, "alpha")
    assert (second.action, second.target) == (CurriculumAction.PROBE, "beta")
    assert (expanded.action, expanded.target) == (
        CurriculumAction.EXPAND,
        "alpha",
    )
    assert controller.state.probe_counts == (("alpha", 1), ("beta", 1))
    assert controller.state.planned_information_nats == pytest.approx(1.6)
    assert not controller.capabilities.knows_relevance_mask
    assert not controller.capabilities.knows_target_hierarchy


def test_candidate_group_policy_stops_when_probes_do_not_produce_evidence() -> None:
    catalog = CurriculumCatalog((_target("unknown", 1),))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(
            minimum_evidence=1,
            bound_width_tolerance=0.1,
            maximum_probes_per_target=1,
        ),
    )
    assessment = _assessment(
        "unknown",
        evidence=CurriculumEvidence.DISCOVERY,
        probe_nats=0.25,
    )

    assert (
        controller.update(CurriculumUpdate(0, (assessment,))).action
        is CurriculumAction.PROBE
    )
    stopped = controller.update(CurriculumUpdate(1, (assessment,)))

    assert stopped.reason is CurriculumReason.INSUFFICIENT_DISCOVERY_EVIDENCE
    assert controller.state.active_targets == ()


def test_candidate_group_policy_rejects_a_privileged_declared_hierarchy() -> None:
    catalog = CurriculumCatalog(
        (_target("root", 1), _target("hidden-child", 2, parent="root"))
    )
    with pytest.raises(ValueError, match="target-hierarchy capability"):
        _controller(
            catalog,
            CandidateGroupDiscoveryPolicy(1, 0.1),
        )


def test_curriculum_acquisition_adapter_respects_active_support_and_query_budget() -> (
    None
):
    catalog = CurriculumCatalog((_target("initial", 1, 2),))
    controller = _controller(catalog, MarginalValuePerBitPolicy())
    policy = CurriculumAcquisitionPolicy(controller)
    policy.update_curriculum(CurriculumUpdate(0, (_assessment("initial"),)))
    agent = FactorizedQueryAgent(
        policy,
        epsilon=0.0,
        query_budget=1,
        seed="synthetic-curriculum",
    )
    candidates = useful_targets(3)
    action = agent.select_train_action(agent.acquisition_context(candidates))
    environment = IndependentRulebook("synthetic-curriculum-environment")
    agent.observe(
        ObservationBatch(
            action,
            tuple(environment.label(target.rule_index) for target in action.targets),
        )
    )

    assert len(action.targets) == 1
    assert action.targets[0].key in {
        TargetKey("reward", 1),
        TargetKey("reward", 2),
    }
    assert TargetKey("reward", 3) not in controller.queryable_members
    assert environment.evaluate(agent.deployment()) == 1.0


def test_multi_member_discovery_probe_preserves_one_query_budget() -> None:
    catalog = CurriculumCatalog((_target("candidate", 2, 3),))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(
            minimum_evidence=1,
            bound_width_tolerance=0.1,
            maximum_probes_per_target=1,
        ),
    )
    policy = CurriculumAcquisitionPolicy(controller)
    assessment = _assessment(
        "candidate",
        evidence=CurriculumEvidence.DISCOVERY,
        probe_nats=0.25,
    )
    assert (
        policy.update_curriculum(CurriculumUpdate(0, (assessment,))).action
        is CurriculumAction.PROBE
    )
    agent = FactorizedQueryAgent(policy, epsilon=0.0, query_budget=1)
    candidates = useful_targets(3)
    probed = agent.select_train_action(agent.acquisition_context(candidates))
    agent.observe(ObservationBatch(probed, (1,)))

    assert len(probed.targets) == 1
    assert probed.targets[0] in candidates[1:]
    assert (
        policy.update_curriculum(CurriculumUpdate(1, (assessment,))).action
        is CurriculumAction.HOLD
    )
    after_probe = agent.select_train_action(agent.acquisition_context(candidates))
    assert after_probe.targets == ()


def test_discovery_probe_temporarily_excludes_unrelated_active_support() -> None:
    catalog = CurriculumCatalog((_target("active", 1), _target("candidate", 2, 3)))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(1, 0.1),
    )
    policy = CurriculumAcquisitionPolicy(controller)
    agent = FactorizedQueryAgent(policy, epsilon=0.0, query_budget=1, seed=0)
    candidates = useful_targets(3)

    expanded = policy.update_curriculum(
        CurriculumUpdate(
            0,
            (
                _assessment(
                    "active",
                    evidence=CurriculumEvidence.DISCOVERY,
                    count=1,
                ),
            ),
        )
    )
    assert expanded.action is CurriculumAction.EXPAND
    active_action = agent.select_train_action(agent.acquisition_context(candidates))
    agent.observe(ObservationBatch(active_action, (1,)))

    probed = policy.update_curriculum(
        CurriculumUpdate(
            1,
            (
                _assessment(
                    "candidate",
                    evidence=CurriculumEvidence.DISCOVERY,
                    probe_nats=0.25,
                ),
            ),
        )
    )
    probe_action = agent.select_train_action(agent.acquisition_context(candidates))

    assert probed.action is CurriculumAction.PROBE
    assert controller.queryable_members == (
        TargetKey("reward", 2),
        TargetKey("reward", 3),
    )
    assert len(probe_action.targets) == 1
    assert probe_action.targets[0].key in controller.queryable_members


def test_discovery_probe_fails_when_no_candidate_member_is_exposed() -> None:
    catalog = CurriculumCatalog((_target("candidate", 2, 3),))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(1, 0.1),
    )
    policy = CurriculumAcquisitionPolicy(controller)
    policy.update_curriculum(
        CurriculumUpdate(
            0,
            (
                _assessment(
                    "candidate",
                    evidence=CurriculumEvidence.DISCOVERY,
                    probe_nats=0.25,
                ),
            ),
        )
    )
    agent = FactorizedQueryAgent(policy, epsilon=0.0, query_budget=1)

    with pytest.raises(ValueError, match="requires a selected candidate"):
        agent.select_train_action(agent.acquisition_context(useful_targets(1)))


def test_pending_action_is_revalidated_after_a_curriculum_probe_update() -> None:
    catalog = CurriculumCatalog((_target("active", 1), _target("candidate", 2)))
    controller = _controller(
        catalog,
        CandidateGroupDiscoveryPolicy(1, 0.1),
    )
    policy = CurriculumAcquisitionPolicy(controller)
    policy.update_curriculum(
        CurriculumUpdate(
            0,
            (
                _assessment(
                    "active",
                    evidence=CurriculumEvidence.DISCOVERY,
                    count=1,
                ),
            ),
        )
    )
    agent = FactorizedQueryAgent(policy, epsilon=0.0, query_budget=1)
    context = agent.acquisition_context(useful_targets(2))
    pending = agent.select_train_action(context)
    policy.update_curriculum(
        CurriculumUpdate(
            1,
            (
                _assessment(
                    "candidate",
                    evidence=CurriculumEvidence.DISCOVERY,
                    probe_nats=0.25,
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="round does not match"):
        agent.observe(ObservationBatch(pending, (1,)))
    with pytest.raises(ValueError, match="round does not match"):
        agent.select_train_action(context)


@dataclass(frozen=True, slots=True)
class _UnderpricedPolicy:
    capabilities: CapabilityManifest = field(
        default_factory=lambda: CapabilityManifest(
            knows_approximate_frontier=True,
            knows_reward_parameters=True,
        )
    )

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: object,
        update: CurriculumUpdate,
        limits: object,
    ) -> CurriculumDecision:
        del catalog, state, limits
        return CurriculumDecision(
            update.round_index,
            CurriculumAction.EXPAND,
            CurriculumReason.EXPANSION_SELECTED,
            target="root",
            score=1.0,
            planned_information_nats=0.5,
            support_added=1,
        )


def test_controller_rejects_a_policy_that_underprices_an_expansion() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    controller = _controller(catalog, _UnderpricedPolicy())

    with pytest.raises(ValueError, match="differs from assessment"):
        controller.update(
            CurriculumUpdate(
                0,
                (_assessment("root", information=(1.0, 1.0)),),
            )
        )


@dataclass(frozen=True, slots=True)
class _OracleConsumerWithoutPrivileges:
    capabilities: CapabilityManifest = field(default_factory=CapabilityManifest)

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: object,
        update: CurriculumUpdate,
        limits: object,
    ) -> CurriculumDecision:
        del catalog, state, limits
        return CurriculumDecision(
            update.round_index,
            CurriculumAction.EXPAND,
            CurriculumReason.EXPANSION_SELECTED,
            target="root",
            score=0.5,
            planned_information_nats=1.0,
            support_added=1,
        )


@dataclass(frozen=True, slots=True)
class _InvalidCapabilityPolicy:
    capabilities: object = "not-a-manifest"

    def decide(
        self,
        catalog: CurriculumCatalog,
        state: object,
        update: CurriculumUpdate,
        limits: object,
    ) -> CurriculumDecision:
        del catalog, state, limits
        return CurriculumDecision(
            update.round_index,
            CurriculumAction.HOLD,
            CurriculumReason.NO_ELIGIBLE_TARGET,
        )


def test_controller_binds_evidence_to_valid_declared_capabilities() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    with pytest.raises(TypeError, match="CapabilityManifest"):
        _controller(catalog, _InvalidCapabilityPolicy())

    controller = _controller(catalog, _OracleConsumerWithoutPrivileges())
    with pytest.raises(ValueError, match="exact-frontier capability"):
        controller.update(
            CurriculumUpdate(
                0,
                (
                    _assessment(
                        "root",
                        evidence=CurriculumEvidence.ORACLE,
                    ),
                ),
                frontier=_frontier(CurriculumEvidence.ORACLE),
            )
        )


def test_decision_reasons_and_scores_are_auditable() -> None:
    with pytest.raises(ValueError, match="reason is incompatible"):
        CurriculumDecision(
            0,
            CurriculumAction.EXPAND,
            CurriculumReason.NO_ELIGIBLE_TARGET,
            target="root",
            score=-99.0,
            planned_information_nats=1.0,
            support_added=1,
        )

    @dataclass(frozen=True, slots=True)
    class WrongScorePolicy:
        capabilities: CapabilityManifest = field(
            default_factory=lambda: CapabilityManifest(
                knows_approximate_frontier=True,
                knows_reward_parameters=True,
            )
        )

        def decide(
            self,
            catalog: CurriculumCatalog,
            state: object,
            update: CurriculumUpdate,
            limits: object,
        ) -> CurriculumDecision:
            del catalog, state, limits
            return CurriculumDecision(
                update.round_index,
                CurriculumAction.EXPAND,
                CurriculumReason.EXPANSION_SELECTED,
                target="root",
                score=-99.0,
                planned_information_nats=1.0,
                support_added=1,
            )

    controller = _controller(
        CurriculumCatalog((_target("root", 1),)),
        WrongScorePolicy(),
    )
    with pytest.raises(ValueError, match="score differs from assessment"):
        controller.update(CurriculumUpdate(0, (_assessment("root"),)))


def test_replay_state_requires_complete_decision_and_information_history() -> None:
    with pytest.raises(ValueError, match="every completed round"):
        CurriculumState(next_round=1)
    with pytest.raises(ValueError, match="planned information"):
        CurriculumState(
            next_round=1,
            planned_information_nats=1.0,
            decisions=(
                CurriculumDecision(
                    0,
                    CurriculumAction.HOLD,
                    CurriculumReason.NO_ELIGIBLE_TARGET,
                ),
            ),
        )


def test_replay_state_rejects_scale_dependent_information_understatement() -> None:
    decision = CurriculumDecision(
        0,
        CurriculumAction.EXPAND,
        CurriculumReason.EXPANSION_SELECTED,
        target="root",
        score=1.0,
        planned_information_nats=1e12,
        support_added=1,
    )
    with pytest.raises(ValueError, match="planned information"):
        CurriculumState(
            next_round=1,
            active_targets=("root",),
            active_members=(TargetKey("reward", 1),),
            planned_information_nats=1e12 - 0.5,
            decisions=(decision,),
        )


def test_full_history_residual_enforces_large_hard_information_limit() -> None:
    catalog = CurriculumCatalog((_target("first", 1), _target("second", 2)))
    controller = _controller(
        catalog,
        MarginalValuePerBitPolicy(),
        information=1e16,
    )
    first = controller.update(
        CurriculumUpdate(
            0,
            (_assessment("first", information=(1e16, 1e16)),),
        )
    )
    second = controller.update(
        CurriculumUpdate(
            1,
            (_assessment("second", information=(1.0, 1.0)),),
        )
    )

    assert first.action is CurriculumAction.EXPAND
    assert second.action is CurriculumAction.HOLD
    assert second.reason is CurriculumReason.BUDGET_OR_SUPPORT_LIMIT
    assert controller.state.active_targets == ("first",)
    assert controller.state.planned_information_nats == 1e16

    over_limit_state = CurriculumState(
        next_round=2,
        active_targets=("first", "second"),
        active_members=(TargetKey("reward", 1), TargetKey("reward", 2)),
        planned_information_nats=1e16,
        decisions=(
            CurriculumDecision(
                0,
                CurriculumAction.EXPAND,
                CurriculumReason.EXPANSION_SELECTED,
                target="first",
                score=1.0,
                planned_information_nats=1e16,
                support_added=1,
            ),
            CurriculumDecision(
                1,
                CurriculumAction.EXPAND,
                CurriculumReason.EXPANSION_SELECTED,
                target="second",
                score=1.0,
                planned_information_nats=1.0,
                support_added=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="information limit"):
        CurriculumController(
            catalog,
            MarginalValuePerBitPolicy(),
            CurriculumLimits(2, 1e16),
            over_limit_state,
        )


def _expansion_decision(
    round_index: int,
    target: str,
    support_added: int,
) -> CurriculumDecision:
    return CurriculumDecision(
        round_index,
        CurriculumAction.EXPAND,
        CurriculumReason.EXPANSION_SELECTED,
        target=target,
        score=1.0,
        planned_information_nats=1.0,
        support_added=support_added,
    )


def test_restored_state_rejects_duplicate_expansion_history() -> None:
    catalog = CurriculumCatalog((_target("root", 1),))
    state = CurriculumState(
        next_round=2,
        active_targets=("root",),
        active_members=(TargetKey("reward", 1),),
        planned_information_nats=2.0,
        decisions=(
            _expansion_decision(0, "root", 1),
            _expansion_decision(1, "root", 1),
        ),
    )

    with pytest.raises(ValueError, match="reuses an active target"):
        CurriculumController(
            catalog,
            MarginalValuePerBitPolicy(),
            CurriculumLimits(4, 4.0),
            state,
        )


def test_restored_state_rejects_child_before_parent() -> None:
    catalog = CurriculumCatalog(
        (_target("root", 1), _target("child", 2, parent="root"))
    )
    state = CurriculumState(
        next_round=1,
        active_targets=("child",),
        active_members=(TargetKey("reward", 2),),
        planned_information_nats=1.0,
        decisions=(_expansion_decision(0, "child", 1),),
    )

    with pytest.raises(ValueError, match="violates target hierarchy"):
        CurriculumController(
            catalog,
            MarginalValuePerBitPolicy(),
            CurriculumLimits(4, 4.0),
            state,
        )


def test_restored_state_rejects_wrong_support_delta() -> None:
    catalog = CurriculumCatalog((_target("root", 1, 2),))
    state = CurriculumState(
        next_round=1,
        active_targets=("root",),
        active_members=(TargetKey("reward", 1), TargetKey("reward", 2)),
        planned_information_nats=1.0,
        decisions=(_expansion_decision(0, "root", 1),),
    )

    with pytest.raises(ValueError, match="invalid support delta"):
        CurriculumController(
            catalog,
            MarginalValuePerBitPolicy(),
            CurriculumLimits(4, 4.0),
            state,
        )
