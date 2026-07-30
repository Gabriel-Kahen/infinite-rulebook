from __future__ import annotations

import pytest

from infinite_rulebook.orchestration.symbolic import (
    ExactSymbolicAdapter,
    ExactSymbolicAdapterV2,
)
from infinite_rulebook.studies.symbolic_registry import (
    SYMBOLIC_STUDY_V1,
    SYMBOLIC_STUDY_V2,
    execution_adapter_factory,
    registered_symbolic_study,
)


def test_registry_resolves_only_exact_registered_study_names() -> None:
    assert (
        registered_symbolic_study("symbolic-construct-calibration-v1")
        is SYMBOLIC_STUDY_V1
    )
    assert (
        registered_symbolic_study("symbolic-construct-confirmatory-v2")
        is SYMBOLIC_STUDY_V2
    )
    with pytest.raises(ValueError, match="unregistered"):
        registered_symbolic_study("symbolic-construct-calibration-v2-lookalike")


def test_execution_adapter_preserves_legacy_namespace_and_exact_v2_names() -> None:
    assert execution_adapter_factory("an-existing-pilot") is ExactSymbolicAdapter
    assert (
        execution_adapter_factory("symbolic-construct-calibration-v2")
        is ExactSymbolicAdapterV2
    )
    assert (
        execution_adapter_factory("symbolic-construct-calibration-v2-lookalike")
        is ExactSymbolicAdapter
    )


def test_power_streams_and_seed_namespaces_are_versioned() -> None:
    assert SYMBOLIC_STUDY_V1.power.rng_stream == "analysis.cluster-power.v1"
    assert SYMBOLIC_STUDY_V2.power.rng_stream == "analysis.cluster-power.v2"
    assert SYMBOLIC_STUDY_V1.seed_namespaces[0] == "calibration.v1"
    assert SYMBOLIC_STUDY_V2.seed_namespaces[0] == "calibration.v2"
    assert SYMBOLIC_STUDY_V1.smoke_prerequisite_hash is None
    assert len(SYMBOLIC_STUDY_V2.smoke_prerequisite_hash or "") == 64
