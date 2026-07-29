from __future__ import annotations

import math
from dataclasses import asdict, replace

import pytest

from infinite_rulebook.analysis.models import Alternative
from infinite_rulebook.analysis.power import (
    DEFAULT_DESIGN_CONFIDENCE_ALPHA,
    DEFAULT_POWER_SIMULATIONS,
    DEFAULT_SIMULATION_ERROR_ALPHA,
    EnvironmentCluster,
    EquivalencePowerHypothesis,
    PowerHypothesis,
    _equivalence_p_value,
    calibrate_environment_count,
    simultaneous_hoeffding_bound,
)
from infinite_rulebook.analysis.power import _sign_p_value as bootstrap_sign_p_value
from infinite_rulebook.analysis.statistics import exact_sign_p_value


def _zero_residual_hypothesis(
    name: str,
    *,
    environments: int = 192,
    algorithms: int = 3,
    observed_difference: float = 1.0,
) -> PowerHypothesis:
    return PowerHypothesis(
        name=name,
        clusters=tuple(
            EnvironmentCluster(
                environment,
                (observed_difference,) * algorithms,
            )
            for environment in range(environments)
        ),
        minimum_effect=1.0,
    )


def _zero_residual_equivalence(
    name: str = "reward-equivalence",
    *,
    environments: int = 192,
    algorithms: int = 3,
) -> EquivalencePowerHypothesis:
    return EquivalencePowerHypothesis(
        name=name,
        clusters=tuple(
            EnvironmentCluster(
                environment,
                (0.0,) * algorithms,
            )
            for environment in range(environments)
        ),
        margin=0.25,
        diagnostic_location=0.0,
    )


def test_power_calibration_selects_smallest_holm_passing_design() -> None:
    family = (
        _zero_residual_hypothesis("structure"),
        _zero_residual_hypothesis("efficiency"),
    )

    result = calibrate_environment_count(
        family,
        (16, 8),
        seed="power-selection",
        simulations=128,
        minimum_power=0.90,
        minimum_joint_power=0.80,
        center_environment_count=64,
    )

    assert result.selected_environment_count == 16
    assert [item.environment_count for item in result.candidates] == [8, 16]
    small, large = result.candidates
    assert not small.meets_targets
    assert all(item.power == 1.0 for item in small.hypotheses)
    assert large.meets_targets
    assert all(item.power == 1.0 for item in large.hypotheses)
    assert all(item.certified_power_lower_bound >= 0.90 for item in large.hypotheses)
    assert large.registered_joint_power == 1.0
    assert large.certified_registered_joint_power_lower_bound >= 0.80
    assert large.directional_global_null_fwer == 0.0
    assert result.simulation_error_alpha == 0.01
    assert result.simultaneous_decision_count == 8
    expected_bound = math.nextafter(
        math.sqrt(math.log(8 / 0.01) / (2 * 128)),
        math.inf,
    )
    assert result.simulation_error_bound == expected_bound
    assert expected_bound == simultaneous_hoeffding_bound(
        simulations=128,
        decision_count=8,
        simulation_error_alpha=0.01,
    )
    assert result.simultaneous_design_event_count == 6
    assert result.design_confidence_alpha == DEFAULT_DESIGN_CONFIDENCE_ALPHA
    assert result.calibration_environment_count == 192
    assert result.center_environment_count == 64
    assert result.probability_environment_count == 128
    assert result.algorithm_replicas_per_environment == 3


def test_algorithm_cells_cannot_masquerade_as_environment_replicates() -> None:
    hypothesis = _zero_residual_hypothesis(
        "clustered",
        environments=192,
        algorithms=8,
    )

    result = calibrate_environment_count(
        (hypothesis,),
        (16,),
        seed="no-pseudoreplication",
        simulations=8,
        minimum_power=0.5,
        minimum_joint_power=0.5,
        center_environment_count=64,
    )

    assert result.calibration_environment_count == 192
    assert result.algorithm_replicas_per_environment == 8
    assert result.selected_environment_count == 16
    assert result.candidates[0].hypotheses[0].favorable_sign_trials == 128


def test_counter_bootstrap_is_reproducible_and_family_order_independent() -> None:
    first_values = (1.0, 1.25, 1.5, 2.0) * 48
    second_values = (-1.0, -1.25, -1.5, -2.0) * 48
    first = PowerHypothesis.from_cluster_differences(
        "first",
        first_values,
        minimum_effect=0.4,
    )
    second = PowerHypothesis.from_cluster_differences(
        "second",
        second_values,
        minimum_effect=-0.4,
        alternative=Alternative.LESS,
    )
    arguments = {
        "candidate_environment_counts": (12, 8),
        "seed": "reproducible-power",
        "simulations": 41,
        "minimum_power": 0.0,
        "minimum_joint_power": 0.0,
        "maximum_fwer": 1.0,
        "center_environment_count": 64,
    }

    forward = calibrate_environment_count((first, second), **arguments)
    reverse = calibrate_environment_count((second, first), **arguments)

    assert forward == reverse
    assert [item.environment_count for item in forward.candidates] == [8, 12]
    assert DEFAULT_POWER_SIMULATIONS == 10_000
    assert DEFAULT_SIMULATION_ERROR_ALPHA == 0.01


def test_conditional_bootstrap_cannot_override_failed_effect_adequacy() -> None:
    result = calibrate_environment_count(
        (
            _zero_residual_hypothesis(
                "off-support-working-model",
                observed_difference=0.0,
            ),
        ),
        (32,),
        seed="simulation-uncertainty",
        simulations=32,
        center_environment_count=64,
    )

    candidate = result.candidates[0]
    assert candidate.hypotheses[0].power == 1.0
    assert candidate.registered_joint_power == 1.0
    assert candidate.directional_global_null_fwer == 0.0
    assert not result.effect_adequacy[0].passes
    assert candidate.hypotheses[0].certified_power_lower_bound == 0.0
    assert not candidate.meets_targets
    assert result.selected_environment_count is None
    serialized = asdict(result)
    assert serialized["simulation_error_alpha"] == 0.01
    assert serialized["simultaneous_decision_count"] == 3
    assert serialized["simulation_error_bound"] == result.simulation_error_bound


def test_registered_equivalence_power_and_both_boundaries_gate_selection() -> None:
    result = calibrate_environment_count(
        (_zero_residual_hypothesis("directional"),),
        (16,),
        equivalence_hypotheses=(_zero_residual_equivalence(),),
        seed="registered-joint",
        simulations=128,
        center_environment_count=64,
    )

    candidate = result.candidates[0]
    assert candidate.equivalence_hypotheses[0].power == 1.0
    assert candidate.registered_joint_power == 1.0
    assert candidate.equivalence_lower_boundary_error == 0.0
    assert candidate.equivalence_upper_boundary_error == 0.0
    assert candidate.meets_targets
    assert result.selected_environment_count == 16
    assert result.simultaneous_decision_count == 6
    assert result.simultaneous_design_event_count == 7
    serialized = asdict(result)["candidates"][0]
    assert serialized["equivalence_hypotheses"][0]["rejections"] == 128
    assert serialized["registered_joint_rejections"] == 128
    assert serialized["equivalence_lower_boundary_rejections"] == 0
    assert serialized["equivalence_upper_boundary_rejections"] == 0


def test_certified_equivalence_bounds_ignore_diagnostic_bootstrap_location() -> None:
    base = _zero_residual_equivalence()

    def calibration(location: float):
        return calibrate_environment_count(
            (_zero_residual_hypothesis("directional"),),
            (16,),
            equivalence_hypotheses=(replace(base, diagnostic_location=location),),
            seed="diagnostic-location-separation",
            simulations=16,
            center_environment_count=64,
        )

    centered = calibration(0.0)
    shifted = calibration(0.2)
    centered_bound = centered.candidates[0].equivalence_hypotheses[0]
    shifted_bound = shifted.candidates[0].equivalence_hypotheses[0]

    assert centered.effect_adequacy == shifted.effect_adequacy
    assert (
        centered_bound.lower_successes,
        centered_bound.upper_successes,
        centered_bound.lower_probability_bound,
        centered_bound.upper_probability_bound,
        centered_bound.lower_test_power_bound,
        centered_bound.upper_test_power_bound,
        centered_bound.certified_power_lower_bound,
    ) == (
        shifted_bound.lower_successes,
        shifted_bound.upper_successes,
        shifted_bound.lower_probability_bound,
        shifted_bound.upper_probability_bound,
        shifted_bound.lower_test_power_bound,
        shifted_bound.upper_test_power_bound,
        shifted_bound.certified_power_lower_bound,
    )


def test_current_registered_design_allocates_all_confidence_events() -> None:
    directional = tuple(
        _zero_residual_hypothesis(f"directional-{index}") for index in range(5)
    )
    result = calibrate_environment_count(
        directional,
        (32, 48, 64, 96, 128, 192, 256, 384, 512),
        equivalence_hypotheses=(_zero_residual_equivalence(),),
        seed="registered-decision-count",
        simulations=16,
        minimum_power=0.0,
        minimum_equivalence_power=0.0,
        minimum_joint_power=0.0,
        maximum_fwer=1.0,
        center_environment_count=64,
    )

    assert result.simultaneous_decision_count == 90
    assert result.simultaneous_design_event_count == 19
    assert len(result.candidates) == 9


def test_bootstrap_residuals_are_computed_once_per_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directional = tuple(
        _zero_residual_hypothesis(f"directional-{index}") for index in range(2)
    )
    equivalence = _zero_residual_equivalence()
    directional_calls: dict[str, int] = {}
    equivalence_calls: dict[str, int] = {}
    directional_getter = PowerHypothesis.centered_residuals.fget
    equivalence_getter = EquivalencePowerHypothesis.centered_residuals.fget
    assert directional_getter is not None
    assert equivalence_getter is not None

    def tracked_directional(item: PowerHypothesis) -> tuple[float, ...]:
        directional_calls[item.name] = directional_calls.get(item.name, 0) + 1
        return directional_getter(item)

    def tracked_equivalence(
        item: EquivalencePowerHypothesis,
    ) -> tuple[float, ...]:
        equivalence_calls[item.name] = equivalence_calls.get(item.name, 0) + 1
        return equivalence_getter(item)

    monkeypatch.setattr(
        PowerHypothesis,
        "centered_residuals",
        property(tracked_directional),
    )
    monkeypatch.setattr(
        EquivalencePowerHypothesis,
        "centered_residuals",
        property(tracked_equivalence),
    )

    calibrate_environment_count(
        directional,
        (16, 32),
        equivalence_hypotheses=(equivalence,),
        seed="residual-cache",
        simulations=8,
        minimum_power=0.0,
        minimum_equivalence_power=0.0,
        minimum_joint_power=0.0,
        maximum_fwer=1.0,
        center_environment_count=64,
    )

    assert directional_calls == {item.name: 1 for item in directional}
    assert equivalence_calls == {equivalence.name: 1}


def test_equivalence_power_can_block_an_otherwise_passing_directional_design() -> None:
    noisy_equivalence = EquivalencePowerHypothesis.from_cluster_differences(
        "noisy-equivalence",
        (-1.0, 1.0) * 96,
        margin=0.25,
        algorithm_replicas_per_environment=3,
    )
    result = calibrate_environment_count(
        (_zero_residual_hypothesis("directional"),),
        (32,),
        equivalence_hypotheses=(noisy_equivalence,),
        seed="equivalence-block",
        simulations=64,
        minimum_power=0.0,
        minimum_equivalence_power=0.90,
        minimum_joint_power=0.0,
        maximum_fwer=1.0,
        center_environment_count=64,
    )

    candidate = result.candidates[0]
    assert candidate.hypotheses[0].certified_power_lower_bound > 0.9
    assert not next(
        item for item in result.effect_adequacy if item.name == "noisy-equivalence"
    ).passes
    assert candidate.equivalence_hypotheses[0].certified_power_lower_bound == 0.0
    assert not candidate.meets_targets
    assert result.selected_environment_count is None


def test_analytic_error_bounds_replace_bootstrap_error_gates() -> None:
    result = calibrate_environment_count(
        tuple(_zero_residual_hypothesis(f"directional-{index}") for index in range(5)),
        (16,),
        equivalence_hypotheses=(_zero_residual_equivalence(),),
        seed="separate-equivalence-family",
        simulations=16,
        minimum_power=0.0,
        minimum_equivalence_power=0.0,
        minimum_joint_power=0.0,
        maximum_fwer=1.0,
        center_environment_count=64,
    )

    candidate = result.candidates[0]
    assert all(item.power == 1.0 for item in candidate.hypotheses)
    assert candidate.equivalence_hypotheses[0].power == 1.0
    assert candidate.directional_global_null_fwer_upper_bound == 0.05
    assert candidate.equivalence_lower_boundary_error_upper_bound == 0.05
    assert candidate.equivalence_upper_boundary_error_upper_bound == 0.05


def test_exact_sign_tost_counts_boundary_ties_against_equivalence() -> None:
    assert _equivalence_p_value((0.0,) * 6, margin=0.25) == 1 / 64
    assert _equivalence_p_value((0.25,) * 6, margin=0.25) == 1.0


def test_equivalence_power_requires_shared_clusters_and_interior_truth() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        EquivalencePowerHypothesis(
            "boundary-truth",
            _zero_residual_equivalence().clusters,
            margin=0.25,
            diagnostic_location=0.25,
        )
    with pytest.raises(ValueError, match="same paired environment"):
        calibrate_environment_count(
            (_zero_residual_hypothesis("directional"),),
            (8,),
            equivalence_hypotheses=(_zero_residual_equivalence(environments=5),),
            seed="unmatched-equivalence",
            simulations=2,
        )


def test_cluster_vector_is_averaged_once() -> None:
    cluster = EnvironmentCluster("environment-a", (1.0, 2.0, 6.0))

    assert cluster.paired_difference == 3.0


def test_bootstrap_sign_test_counts_boundary_ties_against_rejection() -> None:
    values = (0.0, 1.0)

    assert bootstrap_sign_p_value(
        values,
        null=0.0,
        alternative=Alternative.GREATER,
    ) == exact_sign_p_value(
        values,
        null=0.0,
        alternative=Alternative.GREATER,
    )


@pytest.mark.parametrize(
    ("clusters", "message"),
    [
        (
            (
                EnvironmentCluster(0, (0.0,)),
                EnvironmentCluster(0, (1.0,)),
            ),
            "identifiers",
        ),
        (
            (
                EnvironmentCluster(0, (0.0,)),
                EnvironmentCluster(1, (1.0, 2.0)),
            ),
            "fully crossed",
        ),
    ],
)
def test_hypothesis_rejects_invalid_environment_clustering(
    clusters: tuple[EnvironmentCluster, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PowerHypothesis("invalid", clusters, minimum_effect=1.0)


def test_power_family_requires_matching_paired_environment_ids() -> None:
    left = PowerHypothesis(
        "left",
        (
            EnvironmentCluster("a", (0.0,)),
            EnvironmentCluster("b", (1.0,)),
        ),
        minimum_effect=1.0,
    )
    right = PowerHypothesis(
        "right",
        (
            EnvironmentCluster("a", (0.0,)),
            EnvironmentCluster("c", (1.0,)),
        ),
        minimum_effect=1.0,
    )

    with pytest.raises(ValueError, match="same paired environment"):
        calibrate_environment_count(
            (left, right),
            (8,),
            seed=1,
            simulations=2,
        )


@pytest.mark.parametrize(
    ("effect", "alternative", "message"),
    [
        (0.0, Alternative.GREATER, "positive"),
        (-1.0, Alternative.GREATER, "positive"),
        (1.0, Alternative.LESS, "negative"),
        (0.0, Alternative.TWO_SIDED, "nonzero"),
    ],
)
def test_minimum_effect_must_match_registered_direction(
    effect: float,
    alternative: Alternative,
    message: str,
) -> None:
    clusters = (
        EnvironmentCluster(0, (0.0,)),
        EnvironmentCluster(1, (1.0,)),
    )

    with pytest.raises(ValueError, match=message):
        PowerHypothesis(
            "direction",
            clusters,
            minimum_effect=effect,
            alternative=alternative,
        )


@pytest.mark.parametrize(
    "differences",
    [
        (),
        (math.nan,),
        (math.inf,),
    ],
)
def test_environment_cluster_rejects_invalid_algorithm_vectors(
    differences: tuple[float, ...],
) -> None:
    error = ValueError
    with pytest.raises(error):
        EnvironmentCluster(0, differences)


def test_flat_environment_algorithm_cells_are_rejected() -> None:
    with pytest.raises(TypeError, match="EnvironmentCluster"):
        PowerHypothesis(
            "flat",
            (0.0, 1.0),  # type: ignore[arg-type]
            minimum_effect=1.0,
        )


def test_calibration_rejects_duplicate_candidates_and_invalid_rates() -> None:
    hypothesis = _zero_residual_hypothesis("strict")

    with pytest.raises(ValueError, match="unique"):
        calibrate_environment_count(
            (hypothesis,),
            (8, 8),
            seed=0,
            simulations=2,
        )
    with pytest.raises(ValueError, match="minimum_power"):
        calibrate_environment_count(
            (hypothesis,),
            (8,),
            seed=0,
            simulations=2,
            minimum_power=1.1,
        )
    with pytest.raises(ValueError, match="alpha"):
        calibrate_environment_count(
            (hypothesis,),
            (8,),
            seed=0,
            simulations=2,
            alpha=1.0,
        )
    with pytest.raises(ValueError, match="simulation_error_alpha"):
        calibrate_environment_count(
            (hypothesis,),
            (8,),
            seed=0,
            simulations=2,
            simulation_error_alpha=0.0,
        )
