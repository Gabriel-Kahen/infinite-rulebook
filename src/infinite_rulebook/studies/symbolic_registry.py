"""Versioned registry for the executable symbolic studies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from infinite_rulebook.analysis.models import AnalysisPhase
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    symbolic_adapter_contract,
)
from infinite_rulebook.orchestration.freeze import ConfirmatoryFreezeRecord
from infinite_rulebook.orchestration.symbolic import (
    ExactSymbolicAdapter,
    ExactSymbolicAdapterV2,
    exact_symbolic_adapter_class,
)
from infinite_rulebook.studies import symbolic_construct as v1
from infinite_rulebook.studies import symbolic_construct_v2 as v2

AnalysisPlanBuilder = Callable[..., Any]
CalibrationVerifier = Callable[[ExperimentConfig], None]
ConfirmatoryVerifier = Callable[..., ConfirmatoryFreezeRecord]
EvidenceHashBuilder = Callable[..., str]
ExpectedGroupsBuilder = Callable[[ExperimentConfig], tuple[Any, ...]]
PowerHypothesesBuilder = Callable[[Any], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class SymbolicPowerDesign:
    """All registered inputs to one study's deterministic power simulation."""

    candidate_environment_counts: tuple[int, ...]
    center_environment_count: int
    probability_environment_count: int
    simulations: int
    seed: str
    rng_stream: str
    alpha: float
    simulation_error_alpha: float
    design_confidence_alpha: float
    minimum_individual_power: float
    minimum_equivalence_power: float
    minimum_joint_power: float
    maximum_global_null_fwer: float
    minimum_effects: Mapping[str, float]
    equivalence_name: str
    equivalence_margin: float
    equivalence_diagnostic_location: float
    hypotheses: PowerHypothesesBuilder
    equivalence_hypotheses: PowerHypothesesBuilder

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_effects",
            MappingProxyType(dict(self.minimum_effects)),
        )

    def summary_fields(self, *, include_rng_stream: bool) -> dict[str, object]:
        fields: dict[str, object] = {
            "power_simulations": self.simulations,
            "power_seed": self.seed,
            "power_candidate_environment_counts": list(
                self.candidate_environment_counts
            ),
            "power_center_environment_count": self.center_environment_count,
            "power_probability_environment_count": (self.probability_environment_count),
            "power_design_confidence_alpha": self.design_confidence_alpha,
            "power_alpha": self.alpha,
            "power_simulation_error_alpha": self.simulation_error_alpha,
            "minimum_individual_power": self.minimum_individual_power,
            "minimum_equivalence_power": self.minimum_equivalence_power,
            "minimum_joint_power": self.minimum_joint_power,
            "maximum_global_null_fwer": self.maximum_global_null_fwer,
        }
        if include_rng_stream:
            fields["power_rng_stream"] = self.rng_stream
        return fields


@dataclass(frozen=True, slots=True)
class SymbolicStudySpec:
    """Executable contract for one registered symbolic study generation."""

    version: int
    study_contract: str
    calibration_name: str
    confirmatory_name: str
    confirmatory_master_seed: str
    adapter_factory: type[ExactSymbolicAdapter]
    verify_calibration: CalibrationVerifier
    verify_confirmatory: ConfirmatoryVerifier
    expected_groups: ExpectedGroupsBuilder
    build_analysis_plan: AnalysisPlanBuilder
    expected_confirmatory_registration: Callable[[ExperimentConfig], str]
    expected_confirmatory_tolerances: Callable[[ExperimentConfig], dict[str, float]]
    expected_confirmatory_margins: Callable[[], dict[str, float]]
    calibration_evidence_hash: EvidenceHashBuilder
    calibration_evidence_hash_from_hashes: EvidenceHashBuilder
    power: SymbolicPowerDesign
    smoke_config_hash: str
    smoke_prerequisite_hash: str | None
    seed_namespaces: tuple[str, str, str, str]

    @property
    def registered_names(self) -> tuple[str, str]:
        return self.calibration_name, self.confirmatory_name

    @property
    def records_power_rng_stream(self) -> bool:
        return self.version >= 2

    def verify_phase(
        self,
        experiment: ExperimentConfig,
        *,
        analysis_code_hash: str | None = None,
        dependency_lock_hash: str | None = None,
        environment_digest: str | None = None,
    ) -> None:
        if experiment.phase == AnalysisPhase.CALIBRATION.value:
            self.verify_calibration(experiment)
            return
        if experiment.phase == AnalysisPhase.CONFIRMATORY.value:
            self.verify_confirmatory(
                experiment,
                analysis_code_hash=analysis_code_hash,
                dependency_lock_hash=dependency_lock_hash,
                environment_digest=environment_digest,
            )


def _power_design(module: Any, *, rng_stream: str) -> SymbolicPowerDesign:
    return SymbolicPowerDesign(
        candidate_environment_counts=module.POWER_CANDIDATE_ENVIRONMENTS,
        center_environment_count=module.POWER_CENTER_ENVIRONMENTS,
        probability_environment_count=module.POWER_PROBABILITY_ENVIRONMENTS,
        simulations=module.POWER_SIMULATIONS,
        seed=module.POWER_SEED,
        rng_stream=rng_stream,
        alpha=module.POWER_ALPHA,
        simulation_error_alpha=module.POWER_SIMULATION_ERROR_ALPHA,
        design_confidence_alpha=module.POWER_DESIGN_CONFIDENCE_ALPHA,
        minimum_individual_power=module.MINIMUM_INDIVIDUAL_POWER,
        minimum_equivalence_power=module.MINIMUM_EQUIVALENCE_POWER,
        minimum_joint_power=module.MINIMUM_JOINT_POWER,
        maximum_global_null_fwer=module.MAXIMUM_GLOBAL_NULL_FWER,
        minimum_effects=module.PRIMARY_MINIMUM_EFFECTS,
        equivalence_name="ind-red-terminal-hidden-reward-equivalence",
        equivalence_margin=module.S5_REWARD_EQUIVALENCE_MARGIN,
        equivalence_diagnostic_location=module.S5_BOOTSTRAP_DIAGNOSTIC_LOCATION,
        hypotheses=module.power_hypotheses,
        equivalence_hypotheses=module.power_equivalence_hypotheses,
    )


SYMBOLIC_STUDY_V1 = SymbolicStudySpec(
    version=1,
    study_contract=v1.STUDY_CONTRACT,
    calibration_name=v1.SYMBOLIC_V1_CALIBRATION_NAME,
    confirmatory_name=v1.SYMBOLIC_V1_CONFIRMATORY_NAME,
    confirmatory_master_seed=v1.SYMBOLIC_V1_CONFIRMATORY_MASTER_SEED,
    adapter_factory=ExactSymbolicAdapter,
    verify_calibration=v1.verify_symbolic_calibration_design,
    verify_confirmatory=v1.verify_symbolic_confirmatory_contract,
    expected_groups=v1.expected_analysis_groups,
    build_analysis_plan=v1.build_symbolic_analysis_plan,
    expected_confirmatory_registration=v1.expected_confirmatory_registration,
    expected_confirmatory_tolerances=v1.expected_confirmatory_tolerances,
    expected_confirmatory_margins=v1.expected_confirmatory_margins,
    calibration_evidence_hash=v1.calibration_evidence_hash,
    calibration_evidence_hash_from_hashes=v1.calibration_evidence_hash_from_hashes,
    power=_power_design(v1, rng_stream="analysis.cluster-power.v1"),
    smoke_config_hash=v1.SYMBOLIC_V1_SMOKE_CONFIG_HASH,
    smoke_prerequisite_hash=None,
    seed_namespaces=(
        "calibration.v1",
        "confirmatory.v1",
        "algorithm.v1",
        "evaluation.v1",
    ),
)

SYMBOLIC_STUDY_V2 = SymbolicStudySpec(
    version=2,
    study_contract=v2.STUDY_CONTRACT,
    calibration_name=v2.SYMBOLIC_V2_CALIBRATION_NAME,
    confirmatory_name=v2.SYMBOLIC_V2_CONFIRMATORY_NAME,
    confirmatory_master_seed=v2.SYMBOLIC_V2_CONFIRMATORY_MASTER_SEED,
    adapter_factory=ExactSymbolicAdapterV2,
    verify_calibration=v2.verify_symbolic_calibration_design,
    verify_confirmatory=v2.verify_symbolic_confirmatory_contract,
    expected_groups=v2.expected_analysis_groups,
    build_analysis_plan=v2.build_symbolic_analysis_plan,
    expected_confirmatory_registration=v2.expected_confirmatory_registration,
    expected_confirmatory_tolerances=v2.expected_confirmatory_tolerances,
    expected_confirmatory_margins=v2.expected_confirmatory_margins,
    calibration_evidence_hash=v2.calibration_evidence_hash,
    calibration_evidence_hash_from_hashes=v2.calibration_evidence_hash_from_hashes,
    power=_power_design(v2, rng_stream=v2.POWER_RNG_STREAM),
    smoke_config_hash=v2.SYMBOLIC_V2_SMOKE_CONFIG_HASH,
    smoke_prerequisite_hash=v2.SYMBOLIC_V2_SMOKE_PREREQUISITE_HASH,
    seed_namespaces=(
        "calibration.v2",
        "confirmatory.v2",
        "algorithm.v2",
        "evaluation.v2",
    ),
)

_REGISTERED_BY_NAME = MappingProxyType(
    {
        name: spec
        for spec in (SYMBOLIC_STUDY_V1, SYMBOLIC_STUDY_V2)
        for name in spec.registered_names
    }
)


def registered_symbolic_study(name: str) -> SymbolicStudySpec:
    """Resolve only exact preregistered calibration or confirmation names."""

    try:
        return _REGISTERED_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unregistered symbolic study name: {name!r}") from None


def execution_adapter_factory(
    name: str,
) -> type[ExactSymbolicAdapter] | type[ExactSymbolicAdapterV2]:
    """Resolve the exact adapter while retaining the legacy pilot namespace."""

    return exact_symbolic_adapter_class(symbolic_adapter_contract(name))


__all__ = [
    "SYMBOLIC_STUDY_V1",
    "SYMBOLIC_STUDY_V2",
    "SymbolicPowerDesign",
    "SymbolicStudySpec",
    "execution_adapter_factory",
    "registered_symbolic_study",
]
