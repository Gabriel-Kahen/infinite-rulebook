"""Preregistered bounded symbolic construct-validation study, version 1."""

from __future__ import annotations

from infinite_rulebook.analysis.canaries import (
    CanaryPlan,
    ConstantAdditiveMetricCanary,
    ExactZeroMetricCanary,
    FrontierIdentityCanary,
    MetricTrajectoryIdentityCanary,
)
from infinite_rulebook.analysis.loading import expected_groups_from_experiment
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
from infinite_rulebook.orchestration.config import (
    AgentKind,
    EnvironmentKind,
    ExperimentConfig,
)
from infinite_rulebook.orchestration.freeze import (
    ConfirmatoryFreezeError,
    ConfirmatoryFreezeRecord,
    SeedBankIdentities,
)
from infinite_rulebook.orchestration.hashing import scientific_hash

STUDY_CONTRACT = "bounded-symbolic-construct-validation.v1"
SYMBOLIC_V1_SMOKE_CONFIG_HASH = (
    "fae70beb1e57206d77cf192e437eb9d8baef2fb0f877a29f04181b0412edbec2"
)
SYMBOLIC_V1_CALIBRATION_NAME = "symbolic-construct-calibration-v1"
SYMBOLIC_V1_CONFIRMATORY_NAME = "symbolic-construct-confirmatory-v1"
SYMBOLIC_V1_DESIGN_HASH = (
    "160bc39f81fd3661318ae336c60d2fe5765becd2dc108d8692de24e66cc84a31"
)
SYMBOLIC_V1_ALGORITHM_MASTER_SEED = "irb-symbolic-fixed-algorithm-bank-v1"
SYMBOLIC_V1_CALIBRATION_MASTER_SEED = "irb-symbolic-calibration-v1"
SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED = "irb-symbolic-confirmatory-v1"
SYMBOLIC_V1_ALGORITHM_REPLICAS = 3
SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS = 192
PAIRED_PATH_ABSOLUTE_TOLERANCE = 1e-12
LEDGER_RECONCILIATION_TOLERANCE = 1e-12
ARTIFACT_COMPLETION_FRACTION = 1.0
POWER_CANDIDATE_ENVIRONMENTS = (32, 48, 64, 96, 128, 192, 256, 384, 512)
POWER_CENTER_ENVIRONMENTS = 64
POWER_PROBABILITY_ENVIRONMENTS = 128
POWER_SIMULATIONS = 10_000
POWER_SEED = "bounded-symbolic-power-v1"
POWER_ALPHA = 0.05
POWER_SIMULATION_ERROR_ALPHA = DEFAULT_SIMULATION_ERROR_ALPHA
POWER_DESIGN_CONFIDENCE_ALPHA = DEFAULT_DESIGN_CONFIDENCE_ALPHA
MINIMUM_INDIVIDUAL_POWER = 0.90
MINIMUM_EQUIVALENCE_POWER = 0.90
MINIMUM_JOINT_POWER = 0.80
MAXIMUM_GLOBAL_NULL_FWER = 0.05
S5_REWARD_EQUIVALENCE_MARGIN = 0.25
S5_BOOTSTRAP_DIAGNOSTIC_LOCATION = 0.0
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
PRIMARY_MINIMUM_EFFECTS = {
    "scheduled-over-fixed-hidden-reward": 0.25,
    "relevant-over-total-trivia-hidden-reward": 0.25,
    "total-over-relevant-trivia-distractor-information": 0.50,
    "alea-over-ind-prediction-error-novelty": 0.10,
    "ind-over-red-useful-information": 0.50,
}


def _selector(environment: EnvironmentKind, agent: AgentKind) -> GroupSelector:
    return GroupSelector(
        environment_kind=environment.value,
        agent_kind=agent.value,
    )


def expected_analysis_groups(
    config: ExperimentConfig,
) -> tuple[ExpectedGroup, ...]:
    """Return the exact condition/agent/checkpoint inventory for a config."""

    return expected_groups_from_experiment(config)


def verify_symbolic_smoke_design(config: ExperimentConfig) -> None:
    """Fail closed unless Stage 0 uses the exact preregistered smoke matrix."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if (
        config.phase != "pilot"
        or config.name != "symbolic-smoke-pilot"
        or config.config_hash != SYMBOLIC_V1_SMOKE_CONFIG_HASH
        or len(config.cells()) != 24
    ):
        raise ValueError("smoke evidence changed the registered Stage-0 design")


def symbolic_v1_design_payload(config: ExperimentConfig) -> dict[str, object]:
    """Return phase-independent scientific fields covered by the v1 contract."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    payload = config.freeze_payload()
    for name in ("name", "phase", "master_seed", "environment_replicas"):
        payload.pop(name)
    return payload


def symbolic_v1_design_hash(config: ExperimentConfig) -> str:
    return scientific_hash(
        symbolic_v1_design_payload(config),
        domain="symbolic-study-design.v1",
    )


def _verify_symbolic_v1_scientific_design(config: ExperimentConfig) -> None:
    if (
        config.algorithm_master_seed != SYMBOLIC_V1_ALGORITHM_MASTER_SEED
        or config.algorithm_replicas != SYMBOLIC_V1_ALGORITHM_REPLICAS
        or symbolic_v1_design_hash(config) != SYMBOLIC_V1_DESIGN_HASH
    ):
        raise ConfirmatoryFreezeError(
            "config does not match the exact symbolic v1 scientific design"
        )


def verify_symbolic_calibration_design(config: ExperimentConfig) -> None:
    """Require the checked-in v1 calibration design and development inventory."""

    if config.phase != "calibration" or config.confirmatory_freeze is not None:
        raise ConfirmatoryFreezeError(
            "symbolic calibration design must be unsealed phase='calibration'"
        )
    _verify_symbolic_v1_scientific_design(config)
    if (
        config.name != SYMBOLIC_V1_CALIBRATION_NAME
        or config.master_seed != SYMBOLIC_V1_CALIBRATION_MASTER_SEED
        or config.environment_replicas != SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS
    ):
        raise ConfirmatoryFreezeError(
            "symbolic v1 calibration requires the registered master seed and exactly "
            f"{SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS} "
            "environment replicas"
        )


def _verify_symbolic_confirmatory_design(config: ExperimentConfig) -> None:
    _verify_symbolic_v1_scientific_design(config)
    if config.phase == "calibration":
        if config.master_seed != SYMBOLIC_V1_CALIBRATION_MASTER_SEED:
            raise ConfirmatoryFreezeError(
                "confirmatory analysis draft must derive from the registered "
                "calibration design"
            )
    elif config.phase == "confirmatory":
        if (
            config.name != SYMBOLIC_V1_CONFIRMATORY_NAME
            or config.master_seed != SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED
        ):
            raise ConfirmatoryFreezeError(
                "symbolic v1 confirmation requires the registered name and "
                "confirmatory master seed"
            )
    else:
        raise ConfirmatoryFreezeError(
            "confirmatory analysis requires a calibration design or sealed "
            "confirmatory config"
        )
    if config.environment_replicas not in POWER_CANDIDATE_ENVIRONMENTS:
        raise ConfirmatoryFreezeError(
            "confirmatory environment replicas must be a registered "
            "power-candidate count"
        )


def _build_symbolic_analysis_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
    freeze_hash: str | None = None,
) -> AnalysisPlan:
    """Build the fixed primary family and descriptive trajectory summaries."""

    if phase is AnalysisPhase.CONFIRMATORY:
        frozen = True
        seal = freeze_hash or "0" * 64
    else:
        if config.phase != phase.value:
            raise ValueError("analysis phase must match the unsealed experiment config")
        frozen = False
        seal = None
    checkpoint = config.horizon
    contrasts = (
        ContrastSpec(
            "scheduled-over-fixed-hidden-reward",
            "hidden_expected_reward",
            _selector(EnvironmentKind.IND, AgentKind.SCHEDULED),
            _selector(EnvironmentKind.IND, AgentKind.FIXED),
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            "relevant-over-total-trivia-hidden-reward",
            "hidden_expected_reward",
            _selector(EnvironmentKind.TRIVIA, AgentKind.RELEVANT_INFORMATION),
            _selector(EnvironmentKind.TRIVIA, AgentKind.TOTAL_INFORMATION),
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            "total-over-relevant-trivia-distractor-information",
            "distractor_information_nats",
            _selector(EnvironmentKind.TRIVIA, AgentKind.TOTAL_INFORMATION),
            _selector(EnvironmentKind.TRIVIA, AgentKind.RELEVANT_INFORMATION),
            checkpoint,
            Alternative.GREATER,
        ),
        ContrastSpec(
            "alea-over-ind-prediction-error-novelty",
            "novelty.observation_prediction_error",
            _selector(EnvironmentKind.ALEA, AgentKind.NOVELTY),
            _selector(EnvironmentKind.IND, AgentKind.NOVELTY),
            checkpoint,
            Alternative.GREATER,
            interpretation=ContrastInterpretation.TELEMETRY_ONLY,
        ),
        ContrastSpec(
            "ind-over-red-useful-information",
            "relevant_information_nats",
            _selector(EnvironmentKind.IND, AgentKind.REWARD),
            _selector(EnvironmentKind.RED_C, AgentKind.REWARD),
            checkpoint,
            Alternative.GREATER,
            required_equivalence_gates=("ind-red-terminal-hidden-reward-equivalence",),
        ),
    )
    trajectories = (
        (
            "ind-scheduled-hidden-reward-trajectory",
            "hidden_expected_reward",
            EnvironmentKind.IND,
            AgentKind.SCHEDULED,
        ),
        (
            "ind-fixed-hidden-reward-trajectory",
            "hidden_expected_reward",
            EnvironmentKind.IND,
            AgentKind.FIXED,
        ),
        (
            "trivia-total-distractor-information-trajectory",
            "distractor_information_nats",
            EnvironmentKind.TRIVIA,
            AgentKind.TOTAL_INFORMATION,
        ),
        (
            "trivia-relevant-distractor-information-trajectory",
            "distractor_information_nats",
            EnvironmentKind.TRIVIA,
            AgentKind.RELEVANT_INFORMATION,
        ),
        (
            "ind-reward-bit-equivalent-upper-trajectory",
            "bit_equivalent_upper_nats",
            EnvironmentKind.IND,
            AgentKind.REWARD,
        ),
        (
            "red-reward-bit-equivalent-upper-trajectory",
            "bit_equivalent_upper_nats",
            EnvironmentKind.RED_C,
            AgentKind.REWARD,
        ),
    )
    return AnalysisPlan(
        name=f"{STUDY_CONTRACT}-{phase.value}",
        phase=phase,
        contrasts=contrasts,
        equivalences=(
            EquivalenceSpec(
                "ind-red-terminal-hidden-reward-equivalence",
                "hidden_expected_reward",
                _selector(EnvironmentKind.IND, AgentKind.REWARD),
                _selector(EnvironmentKind.RED_C, AgentKind.REWARD),
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
                _selector(environment, agent),
                horizon=config.horizon,
                interpolation=Interpolation.LEFT_HOLD,
            )
            for name, metric, environment, agent in trajectories
        ),
        expected_groups=expected_analysis_groups(config),
        family_alpha=0.05,
        interval_alpha=0.05,
        frozen=frozen,
        freeze_hash=seal,
    )


def expected_confirmatory_tolerances(
    config: ExperimentConfig,
) -> dict[str, float]:
    """Return the exact numerical gates committed by the symbolic seal."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    _verify_symbolic_v1_scientific_design(config)
    return {
        "artifact_completion_fraction": ARTIFACT_COMPLETION_FRACTION,
        "frontier_bound_tolerance_nats": config.solver.bound_tolerance,
        "ledger_reconciliation_nats": LEDGER_RECONCILIATION_TOLERANCE,
        "paired_path_absolute_error": PAIRED_PATH_ABSOLUTE_TOLERANCE,
    }


def expected_confirmatory_margins() -> dict[str, float]:
    """Return the exact practical-effect and equivalence registrations."""

    return {
        **PRIMARY_MINIMUM_EFFECTS,
        "ind-red-terminal-hidden-reward-equivalence": (S5_REWARD_EQUIVALENCE_MARGIN),
    }


def expected_confirmatory_registration(config: ExperimentConfig) -> str:
    """Return the seal-independent confirmatory analysis registration."""

    _verify_symbolic_confirmatory_design(config)
    plan = _build_symbolic_analysis_plan(
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
    """Fail closed unless a config carries the exact registered study seal."""

    if config.phase != "confirmatory" or config.confirmatory_freeze is None:
        raise ConfirmatoryFreezeError(
            "symbolic confirmatory study requires a sealed confirmatory config"
        )
    _verify_symbolic_confirmatory_design(config)
    if config.algorithm_master_seed is None:
        raise ConfirmatoryFreezeError(
            "symbolic confirmatory study requires an explicit fixed "
            "algorithm_master_seed"
        )
    record = config.confirmatory_freeze
    expected_banks = SeedBankIdentities.bind(
        calibration_master_seed=SYMBOLIC_V1_CALIBRATION_MASTER_SEED,
        confirmatory_master_seed=SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED,
        algorithm_master_seed=SYMBOLIC_V1_ALGORITHM_MASTER_SEED,
    )
    if record.seed_banks != expected_banks:
        raise ConfirmatoryFreezeError(
            "seed-bank identities or namespaces do not match the registered "
            "symbolic v1 banks"
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


def build_symbolic_analysis_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
    freeze_hash: str | None = None,
) -> AnalysisPlan:
    """Build and, for confirmation, verify the fixed registered analysis."""

    plan = _build_symbolic_analysis_plan(
        config,
        phase=phase,
        freeze_hash=freeze_hash,
    )
    _verify_symbolic_v1_scientific_design(config)
    if phase is AnalysisPhase.CALIBRATION:
        verify_symbolic_calibration_design(config)
    elif phase is AnalysisPhase.CONFIRMATORY:
        _verify_symbolic_confirmatory_design(config)
        if config.phase == "confirmatory":
            record = verify_symbolic_confirmatory_contract(config)
            if freeze_hash != record.seal_hash:
                raise ConfirmatoryFreezeError(
                    "confirmatory analysis plan must use the config's freeze seal"
                )
    return plan


def build_symbolic_canary_plan(
    config: ExperimentConfig,
    *,
    phase: AnalysisPhase,
) -> CanaryPlan:
    """Build deterministic semantic and paired-path gates without p-values."""

    if config.phase != phase.value:
        raise ValueError("canary phase must match the experiment config")
    if phase is AnalysisPhase.CALIBRATION:
        verify_symbolic_calibration_design(config)
    elif phase is AnalysisPhase.CONFIRMATORY:
        verify_symbolic_confirmatory_contract(config)
    checkpoints = config.checkpoints.rounds
    reward = AgentKind.REWARD
    ind = _selector(EnvironmentKind.IND, reward)
    alea = _selector(EnvironmentKind.ALEA, reward)
    trivia = _selector(EnvironmentKind.TRIVIA, reward)
    public = _selector(EnvironmentKind.PUBLIC_C, reward)
    public_config = next(
        environment
        for environment in config.environments
        if environment.kind is EnvironmentKind.PUBLIC_C
    )
    canaries = [
        FrontierIdentityCanary(
            "alea-frontier-is-ind",
            ind,
            alea,
            checkpoints,
        ),
        FrontierIdentityCanary(
            "trivia-frontier-is-ind",
            ind,
            trivia,
            checkpoints,
        ),
    ]
    for name, right in (
        ("alea", alea),
        ("trivia", trivia),
        ("public", public),
    ):
        canaries.extend(
            (
                MetricTrajectoryIdentityCanary(
                    f"{name}-hidden-reward-path-is-ind",
                    "hidden_expected_reward",
                    ind,
                    right,
                    checkpoints,
                    tolerance=PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
                MetricTrajectoryIdentityCanary(
                    f"{name}-useful-information-path-is-ind",
                    "relevant_information_nats",
                    ind,
                    right,
                    checkpoints,
                    tolerance=PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
            )
        )
    for agent in config.agents:
        public_name = (
            "public-reward-decomposition"
            if agent.kind is AgentKind.REWARD
            else f"public-reward-decomposition-{agent.kind.value}"
        )
        alea_name = (
            "alea-has-no-persistent-distractor-information"
            if agent.kind is AgentKind.NOVELTY
            else (f"alea-has-no-persistent-distractor-information-{agent.kind.value}")
        )
        canaries.extend(
            (
                ConstantAdditiveMetricCanary(
                    public_name,
                    _selector(EnvironmentKind.PUBLIC_C, agent.kind),
                    total_metric="expected_reward",
                    base_metric="hidden_expected_reward",
                    shift_metric="public_reward",
                    expected_shift=public_config.public_reward_cap,
                    checkpoints=checkpoints,
                    tolerance=PAIRED_PATH_ABSOLUTE_TOLERANCE,
                ),
                ExactZeroMetricCanary(
                    alea_name,
                    _selector(EnvironmentKind.ALEA, agent.kind),
                    "distractor_information_nats",
                    checkpoints,
                ),
            )
        )
    return CanaryPlan(
        name=f"{STUDY_CONTRACT}-canaries-{phase.value}",
        phase=phase,
        canaries=tuple(canaries),
    )


def power_hypotheses(report: AnalysisReport) -> tuple[PowerHypothesis, ...]:
    """Turn calibration residual clusters into the registered power family."""

    if report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("power calibration requires a calibration analysis report")
    observed = {result.name: result for result in report.contrasts}
    if set(observed) != set(PRIMARY_MINIMUM_EFFECTS):
        raise ValueError("calibration report does not match the primary family")
    if len({result.pair_count for result in observed.values()}) != 1:
        raise ValueError("primary calibration contrasts use different seed inventories")
    algorithm_counts = {
        result.cell_pair_count // result.pair_count for result in observed.values()
    }
    if len(algorithm_counts) != 1 or any(
        result.cell_pair_count % result.pair_count for result in observed.values()
    ):
        raise ValueError("primary contrasts do not use a fully crossed algorithm bank")
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
    """Build the preregistered S5 exact-sign TOST power hypothesis."""

    if report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("power calibration requires a calibration analysis report")
    if len(report.equivalences) != 1:
        raise ValueError("calibration report must contain exactly one equivalence gate")
    result = report.equivalences[0]
    if (
        result.name != "ind-red-terminal-hidden-reward-equivalence"
        or result.margin != S5_REWARD_EQUIVALENCE_MARGIN
        or result.margin_source != MarginSource.PREREGISTERED.value
        or result.margin_provenance_hash != S5_REWARD_MARGIN_PROVENANCE_HASH
    ):
        raise ValueError("calibration report changed the registered S5 equivalence")
    if result.pair_count < 1 or result.cell_pair_count % result.pair_count:
        raise ValueError("S5 equivalence does not use a fully crossed algorithm bank")
    algorithm_replicas = result.cell_pair_count // result.pair_count
    return (
        EquivalencePowerHypothesis.from_cluster_differences(
            result.name,
            result.differences,
            margin=S5_REWARD_EQUIVALENCE_MARGIN,
            diagnostic_location=S5_BOOTSTRAP_DIAGNOSTIC_LOCATION,
            algorithm_replicas_per_environment=algorithm_replicas,
        ),
    )


def calibration_evidence_hash(
    *,
    config: ExperimentConfig,
    report: AnalysisReport,
    canary_report_hash: str,
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
    """Bind every input used to decide whether confirmation may be frozen."""

    if config.phase != "calibration" or report.phase is not AnalysisPhase.CALIBRATION:
        raise ValueError("calibration evidence requires calibration inputs")
    verify_symbolic_calibration_design(config)
    return calibration_evidence_hash_from_hashes(
        config_hash=config.config_hash,
        analysis_report_hash=report.scientific_hash,
        canary_report_hash=canary_report_hash,
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
    """Recompute the calibration decision identity from released artifacts."""

    return scientific_hash(
        {
            "study_contract": STUDY_CONTRACT,
            "scientific_design_hash": SYMBOLIC_V1_DESIGN_HASH,
            "config_hash": config_hash,
            "analysis_report_hash": analysis_report_hash,
            "canary_report_hash": canary_report_hash,
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
        domain="symbolic-calibration-evidence",
    )


__all__ = [
    "ARTIFACT_COMPLETION_FRACTION",
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
    "POWER_SEED",
    "POWER_SIMULATIONS",
    "POWER_SIMULATION_ERROR_ALPHA",
    "PRIMARY_MINIMUM_EFFECTS",
    "S5_BOOTSTRAP_DIAGNOSTIC_LOCATION",
    "S5_REWARD_EQUIVALENCE_MARGIN",
    "S5_REWARD_MARGIN_PROVENANCE_HASH",
    "STUDY_CONTRACT",
    "SYMBOLIC_V1_ALGORITHM_MASTER_SEED",
    "SYMBOLIC_V1_ALGORITHM_REPLICAS",
    "SYMBOLIC_V1_CALIBRATION_ENVIRONMENT_REPLICAS",
    "SYMBOLIC_V1_CALIBRATION_MASTER_SEED",
    "SYMBOLIC_V1_CALIBRATION_NAME",
    "SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED",
    "SYMBOLIC_V1_CONFIRMATORY_NAME",
    "SYMBOLIC_V1_DESIGN_HASH",
    "SYMBOLIC_V1_SMOKE_CONFIG_HASH",
    "build_symbolic_analysis_plan",
    "build_symbolic_canary_plan",
    "calibration_evidence_hash",
    "calibration_evidence_hash_from_hashes",
    "expected_analysis_groups",
    "expected_confirmatory_margins",
    "expected_confirmatory_registration",
    "expected_confirmatory_tolerances",
    "power_equivalence_hypotheses",
    "power_hypotheses",
    "symbolic_v1_design_hash",
    "symbolic_v1_design_payload",
    "verify_symbolic_calibration_design",
    "verify_symbolic_confirmatory_contract",
    "verify_symbolic_smoke_design",
]
