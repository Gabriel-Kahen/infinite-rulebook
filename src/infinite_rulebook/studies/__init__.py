"""Versioned, preregistered study protocols."""

from infinite_rulebook.studies.symbolic_construct import (
    MINIMUM_EQUIVALENCE_POWER,
    POWER_CANDIDATE_ENVIRONMENTS,
    POWER_SIMULATION_ERROR_ALPHA,
    PRIMARY_MINIMUM_EFFECTS,
    S5_BOOTSTRAP_DIAGNOSTIC_LOCATION,
    SYMBOLIC_V1_DESIGN_HASH,
    build_symbolic_analysis_plan,
    build_symbolic_canary_plan,
    calibration_evidence_hash,
    expected_analysis_groups,
    expected_confirmatory_margins,
    expected_confirmatory_registration,
    expected_confirmatory_tolerances,
    power_equivalence_hypotheses,
    power_hypotheses,
    symbolic_v1_design_hash,
    symbolic_v1_design_payload,
    verify_symbolic_calibration_design,
    verify_symbolic_confirmatory_contract,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    POWER_RNG_STREAM as SYMBOLIC_V2_POWER_RNG_STREAM,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    STUDY_CONTRACT as SYMBOLIC_V2_STUDY_CONTRACT,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    SYMBOLIC_V2_COMPONENT_HASH,
    SYMBOLIC_V2_DESIGN_HASH,
    symbolic_v2_design_hash,
    symbolic_v2_design_payload,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    build_symbolic_analysis_plan as build_symbolic_v2_analysis_plan,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    calibration_evidence_hash as symbolic_v2_calibration_evidence_hash,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    registration_component_hash as symbolic_v2_registration_component_hash,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    verify_symbolic_calibration_design as verify_symbolic_v2_calibration_design,
)
from infinite_rulebook.studies.symbolic_construct_v2 import (
    verify_symbolic_confirmatory_contract as verify_symbolic_v2_confirmatory_contract,
)

__all__ = [
    "MINIMUM_EQUIVALENCE_POWER",
    "POWER_CANDIDATE_ENVIRONMENTS",
    "POWER_SIMULATION_ERROR_ALPHA",
    "PRIMARY_MINIMUM_EFFECTS",
    "S5_BOOTSTRAP_DIAGNOSTIC_LOCATION",
    "SYMBOLIC_V1_DESIGN_HASH",
    "SYMBOLIC_V2_COMPONENT_HASH",
    "SYMBOLIC_V2_DESIGN_HASH",
    "SYMBOLIC_V2_POWER_RNG_STREAM",
    "SYMBOLIC_V2_STUDY_CONTRACT",
    "build_symbolic_analysis_plan",
    "build_symbolic_canary_plan",
    "build_symbolic_v2_analysis_plan",
    "calibration_evidence_hash",
    "expected_analysis_groups",
    "expected_confirmatory_margins",
    "expected_confirmatory_registration",
    "expected_confirmatory_tolerances",
    "power_equivalence_hypotheses",
    "power_hypotheses",
    "symbolic_v1_design_hash",
    "symbolic_v1_design_payload",
    "symbolic_v2_calibration_evidence_hash",
    "symbolic_v2_design_hash",
    "symbolic_v2_design_payload",
    "symbolic_v2_registration_component_hash",
    "verify_symbolic_calibration_design",
    "verify_symbolic_confirmatory_contract",
    "verify_symbolic_v2_calibration_design",
    "verify_symbolic_v2_confirmatory_contract",
]
