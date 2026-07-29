from __future__ import annotations

import json
from dataclasses import replace

import pytest

from infinite_rulebook.analysis.canaries import (
    CanaryPlan,
    ConstantAdditiveMetricCanary,
    ConstantAdditiveMetricResult,
    ExactZeroMetricCanary,
    ExactZeroMetricResult,
    FrontierIdentityCanary,
    FrontierIdentityResult,
    MetricTrajectoryIdentityCanary,
    MetricTrajectoryIdentityResult,
    evaluate_canaries,
)
from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CertifiedFrontier,
    CheckpointObservation,
    GroupSelector,
)
from infinite_rulebook.orchestration.hashing import scientific_hash


def _hash(label: str) -> str:
    return scientific_hash(label, domain="canary-test")


def _frontier(label: str = "shared") -> CertifiedFrontier:
    return CertifiedFrontier(
        semantic_hash=_hash(f"frontier:{label}"),
        zero_information_reward=0.0,
        maximum_reward=1.0,
        points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


def _observation(
    environment: str,
    environment_replica: int,
    algorithm_replica: int,
    checkpoint: int,
    *,
    condition_variant: int = 0,
    frontier: CertifiedFrontier | None = None,
    phase: AnalysisPhase = AnalysisPhase.CALIBRATION,
) -> CheckpointObservation:
    curve = frontier or _frontier()
    hidden = checkpoint / 10.0 + environment_replica / 100.0
    hidden += algorithm_replica / 1000.0
    bonus = 0.25 if environment == "PUBLIC-C" else 0.0
    metrics = (
        ("distractor_information_nats", 0.0),
        ("expected_reward", hidden + bonus),
        ("hidden_expected_reward", hidden),
        ("public_reward", bonus),
        ("trajectory_metric", hidden),
    )
    run_label = (
        f"{environment}:{condition_variant}:{environment_replica}:{algorithm_replica}"
    )
    return CheckpointObservation(
        run_hash=_hash(f"run:{run_label}"),
        content_hash=_hash(f"content:{run_label}"),
        phase=phase,
        confirmatory_frozen=False,
        freeze_hash=None,
        analysis_registration_hash=None,
        condition_hash=_hash(f"condition:{environment}:{condition_variant}"),
        environment_kind=environment,
        agent_kind="reward",
        agent_hash=_hash("agent:reward"),
        environment_replica=environment_replica,
        algorithm_replica=algorithm_replica,
        round_index=checkpoint,
        metrics=metrics,
        semantic_hashes=(("frontier", curve.semantic_hash),),
        frontier=curve,
    )


def _dataset() -> AnalysisDataset:
    observations = tuple(
        _observation(environment, environment_replica, algorithm_replica, checkpoint)
        for environment in ("IND", "ALEA", "PUBLIC-C")
        for environment_replica in range(2)
        for algorithm_replica in range(2)
        for checkpoint in (0, 2)
    )
    return AnalysisDataset(observations)


def _plan() -> CanaryPlan:
    ind = GroupSelector(environment_kind="IND", agent_kind="reward")
    alea = GroupSelector(environment_kind="ALEA", agent_kind="reward")
    public = GroupSelector(environment_kind="PUBLIC-C", agent_kind="reward")
    return CanaryPlan(
        name="symbolic-control-canaries.v1",
        phase=AnalysisPhase.CALIBRATION,
        canaries=(
            FrontierIdentityCanary(
                "alea-frontier-invariance",
                ind,
                alea,
                (0, 2),
            ),
            MetricTrajectoryIdentityCanary(
                "alea-reward-trajectory-invariance",
                "trajectory_metric",
                ind,
                alea,
                (0, 2),
                tolerance=1e-12,
            ),
            ConstantAdditiveMetricCanary(
                "public-reward-decomposition",
                public,
                total_metric="expected_reward",
                base_metric="hidden_expected_reward",
                shift_metric="public_reward",
                expected_shift=0.25,
                checkpoints=(0, 2),
                tolerance=1e-12,
            ),
            ExactZeroMetricCanary(
                "alea-persistent-distractor-zero",
                alea,
                "distractor_information_nats",
                (0, 2),
            ),
        ),
    )


def test_canary_plan_has_deterministic_authenticated_json() -> None:
    plan = _plan()
    payload = json.loads(plan.to_json())

    assert payload == plan.to_dict()
    assert payload["scientific_hash"] == plan.scientific_hash
    assert plan.to_json() == plan.to_json()


def _replace_metric(
    observation: CheckpointObservation,
    metric: str,
    value: float,
) -> CheckpointObservation:
    metrics = tuple(
        (name, value if name == metric else current)
        for name, current in observation.metrics
    )
    return replace(observation, metrics=metrics)


def test_registered_canaries_pass_and_serialize_deterministically() -> None:
    dataset = _dataset()
    report = evaluate_canaries(dataset, _plan())
    reversed_report = evaluate_canaries(
        AnalysisDataset(tuple(reversed(dataset.observations))),
        _plan(),
    )

    assert report.passed
    assert report.scientific_hash == reversed_report.scientific_hash
    assert report.to_dict() == reversed_report.to_dict()
    assert report.to_json() == reversed_report.to_json()
    assert json.loads(report.to_json())["scientific_hash"] == report.scientific_hash
    assert "p_value" not in report.to_json()

    frontier, metric, additive, zero = report.results
    assert isinstance(frontier, FrontierIdentityResult)
    assert frontier.environment_cluster_count == 2
    assert frontier.cell_pair_count == 4
    assert len(frontier.comparisons) == 8
    assert isinstance(metric, MetricTrajectoryIdentityResult)
    assert metric.maximum_absolute_error == 0.0
    assert isinstance(additive, ConstantAdditiveMetricResult)
    assert additive.maximum_decomposition_error <= additive.tolerance
    assert additive.maximum_shift_error == 0.0
    assert isinstance(zero, ExactZeroMetricResult)
    assert zero.maximum_absolute_value == 0.0


def test_scientific_canary_violations_return_failed_evidence() -> None:
    observations = list(_dataset().observations)
    for index, observation in enumerate(observations):
        if (
            observation.environment_kind == "ALEA"
            and observation.environment_replica == 0
            and observation.algorithm_replica == 0
            and observation.round_index == 2
        ):
            observations[index] = _replace_metric(
                observation,
                "trajectory_metric",
                observation.metric("trajectory_metric") + 1e-5,
            )
        if (
            observation.environment_kind == "ALEA"
            and observation.environment_replica == 1
            and observation.algorithm_replica == 1
            and observation.round_index == 2
        ):
            observations[index] = _replace_metric(
                observation,
                "distractor_information_nats",
                1e-15,
            )
        if (
            observation.environment_kind == "PUBLIC-C"
            and observation.environment_replica == 0
            and observation.algorithm_replica == 1
            and observation.round_index == 0
        ):
            observations[index] = _replace_metric(
                observation,
                "public_reward",
                0.2,
            )
    report = evaluate_canaries(AnalysisDataset(tuple(observations)), _plan())

    assert not report.passed
    assert report.results[0].passed
    metric = report.results[1]
    additive = report.results[2]
    zero = report.results[3]
    assert isinstance(metric, MetricTrajectoryIdentityResult)
    assert metric.mismatch_count == 1
    assert isinstance(additive, ConstantAdditiveMetricResult)
    assert additive.mismatch_count == 2
    assert isinstance(zero, ExactZeroMetricResult)
    assert zero.nonzero_count == 1


def test_frontier_semantic_mismatch_is_a_failed_canary() -> None:
    observations = list(_dataset().observations)
    for index, observation in enumerate(observations):
        if (
            observation.environment_kind == "ALEA"
            and observation.environment_replica == 0
            and observation.algorithm_replica == 0
            and observation.round_index == 0
        ):
            curve = _frontier("tampered")
            observations[index] = replace(
                observation,
                frontier=curve,
                semantic_hashes=(("frontier", curve.semantic_hash),),
            )
            break
    report = evaluate_canaries(AnalysisDataset(tuple(observations)), _plan())

    result = report.results[0]
    assert isinstance(result, FrontierIdentityResult)
    assert not result.passed
    assert result.mismatch_count == 1


def test_ambiguous_and_overlapping_selectors_are_rejected() -> None:
    dataset = _dataset()
    variant = tuple(
        _observation(
            "IND",
            environment_replica,
            algorithm_replica,
            checkpoint,
            condition_variant=1,
        )
        for environment_replica in range(2)
        for algorithm_replica in range(2)
        for checkpoint in (0, 2)
    )
    with pytest.raises(AnalysisError, match="ambiguous"):
        evaluate_canaries(
            AnalysisDataset((*dataset.observations, *variant)),
            _plan(),
        )

    condition = _hash("condition:IND:0")
    overlapping = CanaryPlan(
        "overlap",
        AnalysisPhase.CALIBRATION,
        (
            MetricTrajectoryIdentityCanary(
                "same-group",
                "trajectory_metric",
                GroupSelector(environment_kind="IND"),
                GroupSelector(condition_hash=condition),
                (0, 2),
                0.0,
            ),
        ),
    )
    with pytest.raises(AnalysisError, match="same registered group"):
        evaluate_canaries(dataset, overlapping)


def test_missing_pairs_checkpoints_and_incomplete_grids_are_rejected() -> None:
    dataset = _dataset()
    missing_checkpoint = tuple(
        observation
        for observation in dataset.observations
        if not (
            observation.environment_kind == "ALEA"
            and observation.environment_replica == 1
            and observation.algorithm_replica == 1
            and observation.round_index == 2
        )
    )
    with pytest.raises(AnalysisError, match="missing registered"):
        evaluate_canaries(AnalysisDataset(missing_checkpoint), _plan())

    missing_pair = tuple(
        observation
        for observation in dataset.observations
        if not (
            observation.environment_kind == "ALEA"
            and observation.environment_replica == 1
            and observation.algorithm_replica == 1
        )
    )
    with pytest.raises(AnalysisError, match=r"complete.*grid|unmatched"):
        evaluate_canaries(AnalysisDataset(missing_pair), _plan())


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1e-9, True])
def test_nonfinite_or_invalid_tolerances_are_rejected(tolerance: object) -> None:
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        MetricTrajectoryIdentityCanary(
            "invalid",
            "trajectory_metric",
            GroupSelector(environment_kind="IND"),
            GroupSelector(environment_kind="ALEA"),
            (0, 2),
            tolerance,  # type: ignore[arg-type]
        )


def test_phase_mixing_and_plan_phase_mismatch_are_rejected() -> None:
    calibration = _observation("IND", 0, 0, 0)
    confirmatory = replace(
        _observation("ALEA", 0, 0, 0),
        phase=AnalysisPhase.CONFIRMATORY,
        confirmatory_frozen=True,
        freeze_hash=_hash("freeze"),
        analysis_registration_hash=_hash("registration"),
    )
    with pytest.raises(AnalysisError, match="cannot mix"):
        AnalysisDataset((calibration, confirmatory))

    with pytest.raises(AnalysisError, match="plan phase"):
        evaluate_canaries(
            _dataset(),
            replace(_plan(), phase=AnalysisPhase.PILOT),
        )


def test_missing_metric_is_rejected_as_malformed_evidence() -> None:
    observations = list(_dataset().observations)
    target = next(
        index
        for index, observation in enumerate(observations)
        if observation.environment_kind == "IND"
    )
    observation = observations[target]
    observations[target] = replace(
        observation,
        metrics=tuple(
            item for item in observation.metrics if item[0] != "trajectory_metric"
        ),
    )

    with pytest.raises(AnalysisError, match="does not contain metric"):
        evaluate_canaries(AnalysisDataset(tuple(observations)), _plan())
