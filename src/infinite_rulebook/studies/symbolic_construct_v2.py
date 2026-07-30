"""Preregistered symbolic construct-validation study, version 2."""

from __future__ import annotations

from dataclasses import asdict, replace

from infinite_rulebook.analysis.canaries import (
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
)
from infinite_rulebook.analysis.compact_canaries_v2 import (
    AggregateMetricCanary,
    CompactCanaryPlan,
)
from infinite_rulebook.analysis.loading import (
    analysis_agent_hash,
    analysis_condition_hash,
    expected_groups_from_experiment,
)
from infinite_rulebook.analysis.models import (
    Alternative,
    AnalysisPhase,
    AnalysisPlan,
    ContrastInterpretation,
    ContrastSpec,
    EquivalenceSpec,
    ExpectedGroup,
    GroupSelector,
    Interpolation,
    MarginSource,
    ScalingSpec,
)
from infinite_rulebook.analysis.power import (
    DEFAULT_DESIGN_CONFIDENCE_ALPHA,
    DEFAULT_SIMULATION_ERROR_ALPHA,
    EquivalencePowerHypothesis,
    PowerHypothesis,
)
from infinite_rulebook.analysis.reporting import AnalysisReport
from infinite_rulebook.analysis.supplemental_v2 import SupplementalEvidencePlan
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
    RunCell,
)
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    ConfirmatoryFreezeRecord,
    SeedBankIdentities,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash

STUDY_CONTRACT = "bounded-symbolic-construct-validation.v2"
SYMBOLIC_V2_CALIBRATION_NAME = "symbolic-construct-calibration-v2"
SYMBOLIC_V2_CONFIRMATORY_NAME = "symbolic-construct-confirmatory-v2"
SYMBOLIC_V2_ALGORITHM_MASTER_SEED = "irb-symbolic-fixed-algorithm-bank-v2"
SYMBOLIC_V2_CALIBRATION_MASTER_SEED = "irb-symbolic-calibration-v2"
SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED = "irb-symbolic-confirmatory-v2"
SYMBOLIC_V2_ALGORITHM_REPLICAS = 8
SYMBOLIC_V2_CALIBRATION_ENVIRONMENT_REPLICAS = 192
SYMBOLIC_V2_SMOKE_CONFIG_HASH = (
    "fae70beb1e57206d77cf192e437eb9d8baef2fb0f877a29f04181b0412edbec2"
)
SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH = (
    "0ab32994d8c75c4ab36eb8de171f67ec802ae54c09bbb250b773918d5d892249"
)
SYMBOLIC_V2_DESIGN_HASH = (
    "a7d38ff66ff113f0c4a1aaae89e73a39df95294ede8c073b04437de064f88114"
)
SYMBOLIC_V2_COMPONENT_HASH = (
    "b5ff912ecf7d1c070dccf433d605566d5db9bbc962f524a2409077094f1d8986"
)

PAIRED_PATH_ABSOLUTE_TOLERANCE = 1e-12
LEDGER_RECONCILIATION_TOLERANCE = 1e-12
AGGREGATE_METRIC_ABSOLUTE_TOLERANCE = 1e-12
ARTIFACT_COMPLETION_FRACTION = 1.0
POWER_CANDIDATE_ENVIRONMENTS = (
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
)
POWER_CENTER_ENVIRONMENTS = 64
POWER_PROBABILITY_ENVIRONMENTS = 128
POWER_SIMULATIONS = 10_000
POWER_SEED = "bounded-symbolic-power-v2"
POWER_RNG_STREAM = "analysis.cluster-power.v2"
POWER_ALPHA = 0.05
POWER_SIMULATION_ERROR_ALPHA = DEFAULT_SIMULATION_ERROR_ALPHA
POWER_DESIGN_CONFIDENCE_ALPHA = DEFAULT_DESIGN_CONFIDENCE_ALPHA
MINIMUM_INDIVIDUAL_POWER = 0.90
MINIMUM_EQUIVALENCE_POWER = 0.90
MINIMUM_JOINT_POWER = 0.80
MAXIMUM_GLOBAL_NULL_FWER = 0.05
S5_REWARD_EQUIVALENCE_MARGIN = 0.25
S5_BOOTSTRAP_DIAGNOSTIC_LOCATION = 0.0
COMPACT_CANARY_DETAIL_CHUNK_RECORDS = 4096

S1 = "scheduled-over-fixed-hidden-reward"
S2_EARLY = "relevant-over-total-trivia-d6-post-query-mean-hidden-reward"
S2_LOAD = "relevant-over-total-trivia-d24-terminal-hidden-reward"
S3 = "total-over-relevant-trivia-d6-distractor-information"
S4 = "alea-over-ind-prediction-error-novelty"
S5 = "ind-over-red-useful-information"
S5_EQUIVALENCE = "ind-red-terminal-hidden-reward-equivalence"
LEGACY_D6_REPLICATION = "legacy-d6-terminal-hidden-reward-replication"
SECONDARY_D12 = "trivia-d12-terminal-hidden-reward-descriptive-comparison"

PRIMARY_MINIMUM_EFFECTS = {
    S1: 0.25,
    S2_EARLY: 0.25,
    S2_LOAD: 0.25,
    S3: 0.50,
    S4: 0.10,
    S5: 0.50,
}

S5_REWARD_MARGIN_PROVENANCE_HASH = scientific_hash(
    {
        "study_contract": STUDY_CONTRACT,
        "metric": "hidden_expected_reward",
        "margin": S5_REWARD_EQUIVALENCE_MARGIN,
        "rationale": (
            "one quarter of one unit-reward coordinate, and one twelfth of the "
            "registered three-coordinate terminal maximum"
        ),
        "registered_before_calibration": True,
    },
    domain="symbolic-equivalence-margin-registration",
)


def expected_analysis_groups(config: ExperimentConfig) -> tuple[ExpectedGroup, ...]:
    """Return the exact condition/agent/checkpoint inventory."""

    return expected_groups_from_experiment(config)


def symbolic_v2_design_payload(config: ExperimentConfig) -> dict[str, object]:
    """Return phase-independent fields covered by the v2 design."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    payload = config.freeze_payload()
    for name in ("name", "phase", "master_seed", "environment_replicas"):
        payload.pop(name)
    return payload


def symbolic_v2_design_hash(config: ExperimentConfig) -> str:
    return scientific_hash(
        symbolic_v2_design_payload(config),
        domain="symbolic-study-design.v2",
    )


def _environment(
    config: ExperimentConfig,
    kind: EnvironmentKind,
    *,
    distractor_dimensions: int | None = None,
) -> EnvironmentConfig:
    matches = tuple(
        environment
        for environment in config.environments
        if environment.kind is kind
        and (
            distractor_dimensions is None
            or environment.distractor_dimensions == distractor_dimensions
        )
    )
    if len(matches) != 1:
        raise ConfirmatoryFreezeError(
            "v2 selector does not identify exactly one registered environment"
        )
    return matches[0]


def _selector(
    config: ExperimentConfig,
    kind: EnvironmentKind,
    agent_kind: AgentKind,
    *,
    distractor_dimensions: int | None = None,
) -> GroupSelector:
    environment = _environment(
        config,
        kind,
        distractor_dimensions=distractor_dimensions,
    )
    agents = tuple(agent for agent in config.agents if agent.kind is agent_kind)
    if len(agents) != 1:
        raise ConfirmatoryFreezeError(
            "v2 selector does not identify exactly one registered agent"
        )
    cell = RunCell(
        environment=environment,
        feedback=config.feedback,
        reward=config.reward,
        agent=agents[0],
        solver=config.solver,
        environment_replica=0,
        algorithm_replica=0,
    )
    return GroupSelector(
        environment_kind=kind.value,
        agent_kind=agent_kind.value,
        condition_hash=analysis_condition_hash(cell),
        agent_hash=analysis_agent_hash(asdict(agents[0])),
    )


def _selector_payload(selector: GroupSelector) -> dict[str, str | None]:
    return {
        "environment_kind": selector.environment_kind,
        "agent_kind": selector.agent_kind,
        "condition_hash": selector.condition_hash,
        "agent_hash": selector.agent_hash,
    }


def _expected_group_selector(group: ExpectedGroup) -> GroupSelector:
    return GroupSelector(
        environment_kind=group.environment_kind,
        agent_kind=group.agent_kind,
        condition_hash=group.condition_hash,
        agent_hash=group.agent_hash,
    )


def _build_symbolic_canary_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
) -> CompactCanaryPlan:
    checkpoints = config.checkpoints.rounds
    positive_checkpoints = tuple(item for item in checkpoints if item > 0)
    ind = _selector(config, EnvironmentKind.IND, AgentKind.REWARD)
    paired_environments = (
        (
            "alea",
            _selector(config, EnvironmentKind.ALEA, AgentKind.REWARD),
        ),
        (
            "trivia-d6",
            _selector(
                config,
                EnvironmentKind.TRIVIA,
                AgentKind.REWARD,
                distractor_dimensions=6,
            ),
        ),
        (
            "trivia-d12",
            _selector(
                config,
                EnvironmentKind.TRIVIA,
                AgentKind.REWARD,
                distractor_dimensions=12,
            ),
        ),
        (
            "trivia-d24",
            _selector(
                config,
                EnvironmentKind.TRIVIA,
                AgentKind.REWARD,
                distractor_dimensions=24,
            ),
        ),
    )
    public = _selector(config, EnvironmentKind.PUBLIC_C, AgentKind.REWARD)
    public_config = _environment(config, EnvironmentKind.PUBLIC_C)
    canaries = [
        FrontierIdentityCanary(
            f"{name}-frontier-is-ind",
            ind,
            selector,
            checkpoints,
        )
        for name, selector in paired_environments
    ]
    for name, selector in (*paired_environments, ("public", public)):
        canaries.extend(
            (
                MetricTrajectoryIdentityCanary(
                    f"{name}-hidden-reward-path-is-ind",
                    "hidden_expected_reward",
                    ind,
                    selector,
                    checkpoints,
                    PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
                MetricTrajectoryIdentityCanary(
                    f"{name}-useful-information-path-is-ind",
                    "relevant_information_nats",
                    ind,
                    selector,
                    checkpoints,
                    PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
            )
        )
    for agent in config.agents:
        public_suffix = "" if agent.kind is AgentKind.REWARD else f"-{agent.kind.value}"
        alea_suffix = "" if agent.kind is AgentKind.NOVELTY else f"-{agent.kind.value}"
        canaries.extend(
            (
                ConstantAdditiveMetricCanary(
                    f"public-reward-decomposition{public_suffix}",
                    _selector(config, EnvironmentKind.PUBLIC_C, agent.kind),
                    total_metric="expected_reward",
                    base_metric="hidden_expected_reward",
                    shift_metric="public_reward",
                    expected_shift=public_config.public_reward_cap,
                    checkpoints=checkpoints,
                    tolerance=PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
                ExactZeroMetricCanary(
                    f"alea-has-no-persistent-distractor-information{alea_suffix}",
                    _selector(config, EnvironmentKind.ALEA, agent.kind),
                    "distractor_information_nats",
                    checkpoints,
                ),
            )
        )
    return CompactCanaryPlan(
        name=f"{STUDY_CONTRACT}-compact-canaries-{phase.value}",
        phase=phase,
        canaries=tuple(canaries),
        aggregate_canaries=(
            AggregateMetricCanary(
                "post-query-mean-hidden-reward-derivation",
                tuple(
                    _expected_group_selector(group)
                    for group in expected_analysis_groups(config)
                ),
                aggregate_metric="post_query_mean_hidden_expected_reward",
                source_metric="post_query_hidden_expected_reward",
                checkpoints=positive_checkpoints,
                tolerance=AGGREGATE_METRIC_ABSOLUTE_TOLERANCE,
            ),
        ),
        expected_groups=expected_analysis_groups(config),
    )


def _build_symbolic_supplemental_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
) -> SupplementalEvidencePlan:
    relevant_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=6,
    )
    total_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=6,
    )
    relevant_d12 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=12,
    )
    total_d12 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=12,
    )
    return SupplementalEvidencePlan(
        name=f"{STUDY_CONTRACT}-supplemental-{phase.value}",
        phase=phase,
        legacy_replications=(
            ContrastSpec(
                LEGACY_D6_REPLICATION,
                "hidden_expected_reward",
                relevant_d6,
                total_d6,
                config.horizon,
                Alternative.GREATER,
            ),
        ),
        descriptive_comparisons=(
            ContrastSpec(
                SECONDARY_D12,
                "hidden_expected_reward",
                relevant_d12,
                total_d12,
                config.horizon,
                Alternative.GREATER,
                interpretation=ContrastInterpretation.TELEMETRY_ONLY,
            ),
        ),
        expected_groups=expected_analysis_groups(config),
    )


def registration_component_payload(config: ExperimentConfig) -> dict[str, object]:
    """Bind registered v2 features that live outside the primary plan schema."""

    relevant_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=6,
    )
    total_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=6,
    )
    relevant_d12 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=12,
    )
    total_d12 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=12,
    )
    registered_calibration = replace(
        config,
        name=SYMBOLIC_V2_CALIBRATION_NAME,
        phase=AnalysisPhase.CALIBRATION.value,
        master_seed=SYMBOLIC_V2_CALIBRATION_MASTER_SEED,
        environment_replicas=SYMBOLIC_V2_CALIBRATION_ENVIRONMENT_REPLICAS,
        confirmatory_freeze=None,
    )
    canary_plan_hash = _build_symbolic_canary_plan(
        registered_calibration,
        phase=AnalysisPhase.CALIBRATION,
    ).scientific_hash
    supplemental_plan_hash = _build_symbolic_supplemental_plan(
        registered_calibration,
        phase=AnalysisPhase.CALIBRATION,
    ).scientific_hash
    return {
        "study_contract": STUDY_CONTRACT,
        "post_query_metric": {
            "name": "post_query_mean_hidden_expected_reward",
            "source": "post_query_hidden_expected_reward",
            "checkpoint_zero": "absent",
            "source_checkpoint_t_positive": (
                "equals the authenticated post-query training-event reward at round t"
            ),
            "checkpoint_t_positive": (
                "math.fsum(authenticated post-query hidden rewards for rounds "
                "1 through t) divided by t"
            ),
        },
        "compound_s2": {
            "components": [S2_EARLY, S2_LOAD],
            "decision": "both-primary-components-required",
            "legacy_replication_may_rescue": False,
        },
        "legacy_replication": {
            "name": LEGACY_D6_REPLICATION,
            "metric": "hidden_expected_reward",
            "checkpoint": config.horizon,
            "left": _selector_payload(relevant_d6),
            "right": _selector_payload(total_d6),
            "alternative": "greater",
            "family_membership": "outside-primary-holm",
            "may_rescue_compound_s2": False,
        },
        "d12_secondary": {
            "name": SECONDARY_D12,
            "metric": "hidden_expected_reward",
            "checkpoint": config.horizon,
            "left": _selector_payload(relevant_d12),
            "right": _selector_payload(total_d12),
            "alternative": "greater",
            "paired_by": ["environment_replica", "algorithm_replica"],
            "role": "registered-descriptive-paired-contrast",
            "family_membership": "outside-primary-holm",
            "may_rescue_compound_s2": False,
        },
        "compact_canaries": {
            "gate_count": 27,
            "frontier_identities": 4,
            "paired_path_identities": 10,
            "public_decompositions": 6,
            "alea_persistent_information_zero": 6,
            "aggregate_metric_derivations": 1,
            "aggregate_metric_coverage": (
                "all-exact-registered-condition-agent-groups"
            ),
            "detail_chunk_records": COMPACT_CANARY_DETAIL_CHUNK_RECORDS,
            "calibration_plan_hash": canary_plan_hash,
            "confirmatory_inventory_rule": (
                "same registered gates and exact groups with the sealed selected "
                "environment-replica count"
            ),
            "raw_roots_remain_authoritative": True,
        },
        "supplemental_evidence": {
            "calibration_plan_hash": supplemental_plan_hash,
            "confirmatory_inventory_rule": (
                "same registered comparisons and exact groups with the sealed "
                "selected environment-replica count"
            ),
            "legacy_replication": LEGACY_D6_REPLICATION,
            "descriptive_comparison": SECONDARY_D12,
            "family_membership": "outside-primary-holm",
            "may_rescue_compound_s2": False,
        },
        "power": {
            "seed": POWER_SEED,
            "rng_stream": POWER_RNG_STREAM,
            "candidate_environment_counts": list(POWER_CANDIDATE_ENVIRONMENTS),
        },
        "stage_0_prerequisite": {
            "role": "operational-not-inferential",
            "required": True,
            "config_hash": SYMBOLIC_V2_SMOKE_CONFIG_HASH,
            "evidence_hash": SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
            "does_not_replace_v2_adapter_validation": True,
        },
    }


def registration_component_hash(config: ExperimentConfig) -> str:
    return scientific_hash(
        registration_component_payload(config),
        domain="symbolic-v2-registration-components",
    )


def build_symbolic_canary_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
) -> CompactCanaryPlan:
    """Build the exact 27-gate compact v2 canary registration."""

    plan = _build_symbolic_canary_plan(config, phase=phase)
    if config.phase != phase.value:
        raise ValueError("canary phase must match the experiment config")
    if phase is AnalysisPhase.CALIBRATION:
        verify_symbolic_calibration_design(config)
    elif phase is AnalysisPhase.CONFIRMATORY:
        verify_symbolic_confirmatory_contract(config)
    else:
        raise ValueError("v2 canaries require calibration or confirmation")
    return plan


def build_symbolic_supplemental_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
) -> SupplementalEvidencePlan:
    """Build the exact outside-Holm paired v2 evidence registration."""

    plan = _build_symbolic_supplemental_plan(config, phase=phase)
    if config.phase != phase.value:
        raise ValueError("supplemental phase must match the experiment config")
    if phase is AnalysisPhase.CALIBRATION:
        verify_symbolic_calibration_design(config)
    elif phase is AnalysisPhase.CONFIRMATORY:
        verify_symbolic_confirmatory_contract(config)
    else:
        raise ValueError(
            "v2 supplemental evidence requires calibration or confirmation"
        )
    return plan


def _verify_scientific_design(config: ExperimentConfig) -> None:
    if (
        config.algorithm_master_seed != SYMBOLIC_V2_ALGORITHM_MASTER_SEED
        or config.algorithm_replicas != SYMBOLIC_V2_ALGORITHM_REPLICAS
        or symbolic_v2_design_hash(config) != SYMBOLIC_V2_DESIGN_HASH
        or registration_component_hash(config) != SYMBOLIC_V2_COMPONENT_HASH
    ):
        raise ConfirmatoryFreezeError(
            "config does not match the exact symbolic v2 scientific design"
        )


def verify_symbolic_calibration_design(config: ExperimentConfig) -> None:
    """Require the exact unsealed v2 calibration matrix."""

    if config.phase != "calibration" or config.confirmatory_freeze is not None:
        raise ConfirmatoryFreezeError(
            "symbolic v2 calibration must be unsealed phase='calibration'"
        )
    _verify_scientific_design(config)
    if (
        config.name != SYMBOLIC_V2_CALIBRATION_NAME
        or config.master_seed != SYMBOLIC_V2_CALIBRATION_MASTER_SEED
        or config.environment_replicas != SYMBOLIC_V2_CALIBRATION_ENVIRONMENT_REPLICAS
    ):
        raise ConfirmatoryFreezeError(
            "symbolic v2 calibration changed its name, seed, or environment count"
        )


def _verify_confirmatory_design(config: ExperimentConfig) -> None:
    _verify_scientific_design(config)
    if config.phase == "calibration":
        if (
            config.name != SYMBOLIC_V2_CALIBRATION_NAME
            or config.master_seed != SYMBOLIC_V2_CALIBRATION_MASTER_SEED
        ):
            raise ConfirmatoryFreezeError(
                "v2 confirmatory draft must derive from registered calibration"
            )
    elif config.phase == "confirmatory":
        if (
            config.name != SYMBOLIC_V2_CONFIRMATORY_NAME
            or config.master_seed != SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED
        ):
            raise ConfirmatoryFreezeError(
                "symbolic v2 confirmation changed its registered name or seed"
            )
    else:
        raise ConfirmatoryFreezeError(
            "v2 confirmatory analysis requires calibration or confirmation"
        )
    if config.environment_replicas not in POWER_CANDIDATE_ENVIRONMENTS:
        raise ConfirmatoryFreezeError(
            "v2 confirmatory environment count is outside the registered grid"
        )


def _build_analysis_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
    freeze_hash: str | None,
) -> AnalysisPlan:
    if phase is AnalysisPhase.CONFIRMATORY:
        frozen = True
        seal = freeze_hash or "0" * 64
    else:
        if config.phase != phase.value:
            raise ValueError("analysis phase must match the experiment config")
        frozen = False
        seal = None
    checkpoint = config.horizon
    ind_scheduled = _selector(config, EnvironmentKind.IND, AgentKind.SCHEDULED)
    ind_fixed = _selector(config, EnvironmentKind.IND, AgentKind.FIXED)
    ind_reward = _selector(config, EnvironmentKind.IND, AgentKind.REWARD)
    red_reward = _selector(config, EnvironmentKind.RED_C, AgentKind.REWARD)
    alea_novelty = _selector(config, EnvironmentKind.ALEA, AgentKind.NOVELTY)
    ind_novelty = _selector(config, EnvironmentKind.IND, AgentKind.NOVELTY)
    relevant_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=6,
    )
    total_d6 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=6,
    )
    relevant_d24 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.RELEVANT_INFORMATION,
        distractor_dimensions=24,
    )
    total_d24 = _selector(
        config,
        EnvironmentKind.TRIVIA,
        AgentKind.TOTAL_INFORMATION,
        distractor_dimensions=24,
    )
    contrasts = (
        ContrastSpec(
            S1,
            "hidden_expected_reward",
            ind_scheduled,
            ind_fixed,
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            S2_EARLY,
            "post_query_mean_hidden_expected_reward",
            relevant_d6,
            total_d6,
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            S2_LOAD,
            "hidden_expected_reward",
            relevant_d24,
            total_d24,
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            S3,
            "distractor_information_nats",
            total_d6,
            relevant_d6,
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            S4,
            "novelty.observation_prediction_error",
            alea_novelty,
            ind_novelty,
            checkpoint,
            Alternative.GREATER,
            interpretation=ContrastInterpretation.TELEMETRY_ONLY,
        ),
        ContrastSpec(
            S5,
            "relevant_information_nats",
            ind_reward,
            red_reward,
            checkpoint,
            Alternative.GREATER,
            required_equivalence_gates=(S5_EQUIVALENCE,),
        ),
    )
    trajectory_specs = (
        ("ind-scheduled-hidden-reward", "hidden_expected_reward", ind_scheduled),
        ("ind-fixed-hidden-reward", "hidden_expected_reward", ind_fixed),
        ("trivia-d6-relevant-hidden-reward", "hidden_expected_reward", relevant_d6),
        ("trivia-d6-total-hidden-reward", "hidden_expected_reward", total_d6),
        (
            "trivia-d24-relevant-hidden-reward",
            "hidden_expected_reward",
            relevant_d24,
        ),
        ("trivia-d24-total-hidden-reward", "hidden_expected_reward", total_d24),
        ("ind-reward-useful-information", "relevant_information_nats", ind_reward),
        ("red-reward-useful-information", "relevant_information_nats", red_reward),
    )
    component_hash = registration_component_hash(config)
    return AnalysisPlan(
        name=f"{STUDY_CONTRACT}-{component_hash}-{phase.value}",
        phase=phase,
        contrasts=contrasts,
        equivalences=(
            EquivalenceSpec(
                S5_EQUIVALENCE,
                "hidden_expected_reward",
                ind_reward,
                red_reward,
                checkpoint,
                margin=S5_REWARD_EQUIVALENCE_MARGIN,
                margin_source=MarginSource.PREREGISTERED,
                margin_provenance_hash=S5_REWARD_MARGIN_PROVENANCE_HASH,
            ),
        ),
        scalings=tuple(
            ScalingSpec(
                name,
                metric,
                selector,
                horizon=config.horizon,
                interpolation=Interpolation.LEFT_HOLD,
            )
            for name, metric, selector in trajectory_specs
        ),
        expected_groups=expected_analysis_groups(config),
        family_alpha=0.05,
        interval_alpha=0.05,
        frozen=frozen,
        freeze_hash=seal,
    )


def build_symbolic_analysis_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
    freeze_hash: str | None = None,
) -> AnalysisPlan:
    """Build and verify the six-primary v2 analysis family."""

    plan = _build_analysis_plan(config, phase=phase, freeze_hash=freeze_hash)
    _verify_scientific_design(config)
    if phase is AnalysisPhase.CALIBRATION:
        verify_symbolic_calibration_design(config)
    elif phase is AnalysisPhase.CONFIRMATORY:
        _verify_confirmatory_design(config)
        if config.phase == "confirmatory":
            record = verify_symbolic_confirmatory_contract(config)
            if freeze_hash != record.seal_hash:
                raise ConfirmatoryFreezeError(
                    "v2 analysis plan must use the config's freeze seal"
                )
    return plan


def expected_confirmatory_tolerances(
    config: ExperimentConfig,
) -> dict[str, float]:
    _verify_scientific_design(config)
    return {
        "aggregate_metric_absolute_error": AGGREGATE_METRIC_ABSOLUTE_TOLERANCE,
        "artifact_completion_fraction": ARTIFACT_COMPLETION_FRACTION,
        "frontier_bound_tolerance_nats": config.solver.bound_tolerance,
        "ledger_reconciliation_nats": LEDGER_RECONCILIATION_TOLERANCE,
        "paired_path_absolute_error": PAIRED_PATH_ABSOLUTE_TOLERANCE,
    }


def expected_confirmatory_margins() -> dict[str, float]:
    return {
        **PRIMARY_MINIMUM_EFFECTS,
        S5_EQUIVALENCE: S5_REWARD_EQUIVALENCE_MARGIN,
    }


def expected_confirmatory_registration(config: ExperimentConfig) -> str:
    _verify_confirmatory_design(config)
    plan = _build_analysis_plan(
        config,
        phase=AnalysisPhase.CONFIRMATORY,
        freeze_hash=(
            None
            if config.confirmatory_freeze is None
            else config.confirmatory_freeze.seal_hash
        ),
    )
    return plan.registration_hash


def verify_symbolic_confirmatory_contract(
    config: ExperimentConfig,
    *,
    analysis_code_hash: str | None = None,
    dependency_lock_hash: str | None = None,
    environment_digest: str | None = None,
) -> ConfirmatoryFreezeRecord:
    """Fail closed unless the config carries the exact v2 seal."""

    if config.phase != "confirmatory" or config.confirmatory_freeze is None:
        raise ConfirmatoryFreezeError(
            "symbolic v2 confirmation requires a sealed confirmatory config"
        )
    _verify_confirmatory_design(config)
    record = config.confirmatory_freeze
    expected_banks = SeedBankIdentities.bind(
        calibration_master_seed=SYMBOLIC_V2_CALIBRATION_MASTER_SEED,
        confirmatory_master_seed=SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED,
        algorithm_master_seed=SYMBOLIC_V2_ALGORITHM_MASTER_SEED,
        calibration_namespace="calibration.v2",
        confirmatory_namespace="confirmatory.v2",
        algorithm_namespace="algorithm.v2",
        evaluation_namespace="evaluation.v2",
    )
    if record.seed_banks != expected_banks:
        raise ConfirmatoryFreezeError(
            "seed-bank identities or namespaces do not match symbolic v2"
        )
    record.verify_config(config)
    record.verify_semantic_contract(
        analysis_contract=STUDY_CONTRACT,
        analysis_version=expected_confirmatory_registration(config),
        analysis_code_hash=analysis_code_hash,
        dependency_lock_hash=dependency_lock_hash,
        environment_digest=environment_digest,
        tolerances=expected_confirmatory_tolerances(config),
        margins=expected_confirmatory_margins(),
    )
    return record


def power_hypotheses(report: AnalysisReport) -> tuple[PowerHypothesis, ...]:
    """Turn v2 calibration clusters into the registered six-primary family."""

    if report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("power calibration requires a calibration report")
    observed = {result.name: result for result in report.contrasts}
    if set(observed) != set(PRIMARY_MINIMUM_EFFECTS):
        raise ValueError("calibration report does not match the v2 primary family")
    if len({result.pair_count for result in observed.values()}) != 1:
        raise ValueError("v2 primary contrasts use different seed inventories")
    algorithm_counts = {
        result.cell_pair_count // result.pair_count for result in observed.values()
    }
    if len(algorithm_counts) != 1 or any(
        result.cell_pair_count % result.pair_count for result in observed.values()
    ):
        raise ValueError("v2 primaries do not use a fully crossed algorithm bank")
    algorithm_replicas = next(iter(algorithm_counts))
    return tuple(
        PowerHypothesis.from_cluster_differences(
            name,
            observed[name].differences,
            minimum_effect=PRIMARY_MINIMUM_EFFECTS[name],
            alternative=Alternative.GREATER,
            algorithm_replicas_per_environment=algorithm_replicas,
        )
        for name in sorted(PRIMARY_MINIMUM_EFFECTS)
    )


def power_equivalence_hypotheses(
    report: AnalysisReport,
) -> tuple[EquivalencePowerHypothesis, ...]:
    if report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("power calibration requires a calibration report")
    if len(report.equivalences) != 1:
        raise ValueError("v2 calibration needs exactly one equivalence gate")
    result = report.equivalences[0]
    if (
        result.name != S5_EQUIVALENCE
        or result.margin != S5_REWARD_EQUIVALENCE_MARGIN
        or result.margin_source != MarginSource.PREREGISTERED.value
        or result.margin_provenance_hash != S5_REWARD_MARGIN_PROVENANCE_HASH
        or result.pair_count < 1
        or result.cell_pair_count % result.pair_count
    ):
        raise ValueError("calibration report changed the v2 S5 equivalence")
    return (
        EquivalencePowerHypothesis.from_cluster_differences(
            result.name,
            result.differences,
            margin=S5_REWARD_EQUIVALENCE_MARGIN,
            diagnostic_location=S5_BOOTSTRAP_DIAGNOSTIC_LOCATION,
            algorithm_replicas_per_environment=(
                result.cell_pair_count // result.pair_count
            ),
        ),
    )


def calibration_evidence_hash(
    *,
    config: ExperimentConfig,
    report: AnalysisReport,
    canary_report_hash: str,
    canary_detail_root_hash: str,
    canary_detail_record_count: int,
    supplemental_plan_hash: str,
    supplemental_report_hash: str,
    power_calibration_hash: str,
    reproducibility_report_hash: str,
    raw_serial_inventory_hash: str,
    raw_parallel_inventory_hash: str,
    deviation_log_hash: str,
    smoke_prerequisite_hash: str,
    smoke_config_hash: str,
    smoke_reproducibility_hash: str,
    smoke_raw_serial_inventory_hash: str,
    smoke_raw_parallel_inventory_hash: str,
) -> str:
    """Bind the authenticated v2 calibration decision inputs."""

    if config.phase != "calibration" or report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("v2 calibration evidence requires calibration inputs")
    verify_symbolic_calibration_design(config)
    return calibration_evidence_hash_from_hashes(
        config_hash=config.config_hash,
        analysis_report_hash=report.scientific_hash,
        canary_report_hash=canary_report_hash,
        canary_detail_root_hash=canary_detail_root_hash,
        canary_detail_record_count=canary_detail_record_count,
        supplemental_plan_hash=supplemental_plan_hash,
        supplemental_report_hash=supplemental_report_hash,
        power_calibration_hash=power_calibration_hash,
        reproducibility_report_hash=reproducibility_report_hash,
        raw_serial_inventory_hash=raw_serial_inventory_hash,
        raw_parallel_inventory_hash=raw_parallel_inventory_hash,
        deviation_log_hash=deviation_log_hash,
        smoke_prerequisite_hash=smoke_prerequisite_hash,
        smoke_config_hash=smoke_config_hash,
        smoke_reproducibility_hash=smoke_reproducibility_hash,
        smoke_raw_serial_inventory_hash=smoke_raw_serial_inventory_hash,
        smoke_raw_parallel_inventory_hash=smoke_raw_parallel_inventory_hash,
        analysis_code_hash=dict(report.provenance).get("analysis_code_hash"),
        run_settings_hash=report.run_settings_hash,
    )


def calibration_evidence_hash_from_hashes(
    *,
    config_hash: str,
    analysis_report_hash: str,
    canary_report_hash: str,
    canary_detail_root_hash: str,
    canary_detail_record_count: int,
    supplemental_plan_hash: str,
    supplemental_report_hash: str,
    power_calibration_hash: str,
    reproducibility_report_hash: str,
    raw_serial_inventory_hash: str,
    raw_parallel_inventory_hash: str,
    deviation_log_hash: str,
    smoke_prerequisite_hash: str,
    smoke_config_hash: str,
    smoke_reproducibility_hash: str,
    smoke_raw_serial_inventory_hash: str,
    smoke_raw_parallel_inventory_hash: str,
    analysis_code_hash: str | None,
    run_settings_hash: str | None,
) -> str:
    """Bind every v2 input that can authorize a confirmatory seal."""

    required_hashes = {
        "config_hash": config_hash,
        "analysis_report_hash": analysis_report_hash,
        "canary_report_hash": canary_report_hash,
        "canary_detail_root_hash": canary_detail_root_hash,
        "supplemental_plan_hash": supplemental_plan_hash,
        "supplemental_report_hash": supplemental_report_hash,
        "power_calibration_hash": power_calibration_hash,
        "reproducibility_report_hash": reproducibility_report_hash,
        "raw_serial_inventory_hash": raw_serial_inventory_hash,
        "raw_parallel_inventory_hash": raw_parallel_inventory_hash,
        "deviation_log_hash": deviation_log_hash,
        "smoke_prerequisite_hash": smoke_prerequisite_hash,
        "smoke_config_hash": smoke_config_hash,
        "smoke_reproducibility_hash": smoke_reproducibility_hash,
        "smoke_raw_serial_inventory_hash": smoke_raw_serial_inventory_hash,
        "smoke_raw_parallel_inventory_hash": smoke_raw_parallel_inventory_hash,
    }
    if any(not is_sha256(value) for value in required_hashes.values()):
        raise ValueError("v2 calibration evidence requires SHA-256 hash inputs")
    if any(
        value is not None and not is_sha256(value)
        for value in (analysis_code_hash, run_settings_hash)
    ):
        raise ValueError("v2 optional provenance identities must be SHA-256 or None")
    if (
        smoke_prerequisite_hash != SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH
        or smoke_config_hash != SYMBOLIC_V2_SMOKE_CONFIG_HASH
    ):
        raise ValueError("v2 calibration changed its registered Stage-0 prerequisite")
    if (
        isinstance(canary_detail_record_count, bool)
        or not isinstance(canary_detail_record_count, int)
        or canary_detail_record_count < 1
    ):
        raise ValueError("v2 canary detail record count must be positive")
    return scientific_hash(
        {
            "study_contract": STUDY_CONTRACT,
            "scientific_design_hash": SYMBOLIC_V2_DESIGN_HASH,
            "registration_component_hash": SYMBOLIC_V2_COMPONENT_HASH,
            "config_hash": config_hash,
            "analysis_report_hash": analysis_report_hash,
            "canary_report_hash": canary_report_hash,
            "canary_detail_root_hash": canary_detail_root_hash,
            "canary_detail_record_count": canary_detail_record_count,
            "supplemental_plan_hash": supplemental_plan_hash,
            "supplemental_report_hash": supplemental_report_hash,
            "power_calibration_hash": power_calibration_hash,
            "reproducibility_report_hash": reproducibility_report_hash,
            "raw_serial_inventory_hash": raw_serial_inventory_hash,
            "raw_parallel_inventory_hash": raw_parallel_inventory_hash,
            "deviation_log_hash": deviation_log_hash,
            "smoke_prerequisite_hash": smoke_prerequisite_hash,
            "smoke_config_hash": smoke_config_hash,
            "smoke_reproducibility_hash": smoke_reproducibility_hash,
            "smoke_raw_serial_inventory_hash": smoke_raw_serial_inventory_hash,
            "smoke_raw_parallel_inventory_hash": smoke_raw_parallel_inventory_hash,
            "analysis_code_hash": analysis_code_hash,
            "run_settings_hash": run_settings_hash,
        },
        domain="symbolic-calibration-evidence.v2",
    )


__all__ = [
    "AGGREGATE_METRIC_ABSOLUTE_TOLERANCE",
    "ARTIFACT_COMPLETION_FRACTION",
    "COMPACT_CANARY_DETAIL_CHUNK_RECORDS",
    "LEDGER_RECONCILIATION_TOLERANCE",
    "MAXIMUM_GLOBAL_NULL_FWER",
    "MINIMUM_EQUIVALENCE_POWER",
    "MINIMUM_INDIVIDUAL_POWER",
    "MINIMUM_JOINT_POWER",
    "PAIRED_PATH_ABSOLUTE_TOLERANCE",
    "POWER_ALPHA",
    "POWER_CANDIDATE_ENVIRONMENTS",
    "POWER_CENTER_ENVIRONMENTS",
    "POWER_DESIGN_CONFIDENCE_ALPHA",
    "POWER_PROBABILITY_ENVIRONMENTS",
    "POWER_RNG_STREAM",
    "POWER_SEED",
    "POWER_SIMULATIONS",
    "POWER_SIMULATION_ERROR_ALPHA",
    "PRIMARY_MINIMUM_EFFECTS",
    "STUDY_CONTRACT",
    "SYMBOLIC_V2_ALGORITHM_MASTER_SEED",
    "SYMBOLIC_V2_ALGORITHM_REPLICAS",
    "SYMBOLIC_V2_CALIBRATION_ENVIRONMENT_REPLICAS",
    "SYMBOLIC_V2_CALIBRATION_MASTER_SEED",
    "SYMBOLIC_V2_CALIBRATION_NAME",
    "SYMBOLIC_V2_COMPONENT_HASH",
    "SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED",
    "SYMBOLIC_V2_CONFIRMATORY_NAME",
    "SYMBOLIC_V2_DESIGN_HASH",
    "SYMBOLIC_V2_SMOKE_CONFIG_HASH",
    "SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH",
    "build_symbolic_analysis_plan",
    "build_symbolic_canary_plan",
    "build_symbolic_supplemental_plan",
    "calibration_evidence_hash",
    "calibration_evidence_hash_from_hashes",
    "expected_analysis_groups",
    "expected_confirmatory_margins",
    "expected_confirmatory_registration",
    "expected_confirmatory_tolerances",
    "power_equivalence_hypotheses",
    "power_hypotheses",
    "registration_component_hash",
    "registration_component_payload",
    "symbolic_v2_design_hash",
    "symbolic_v2_design_payload",
    "verify_symbolic_calibration_design",
    "verify_symbolic_confirmatory_contract",
]
