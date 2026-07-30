"""Measure eager analysis-dataset memory and runtime at a registered grid size."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisPhase,
    CertifiedFrontier,
    CheckpointObservation,
)
from infinite_rulebook.analysis.statistics import pool_checkpoints
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.orchestration.hashing import scientific_hash


def _digest(value: object, *, domain: str) -> str:
    return scientific_hash(value, domain=domain)


def _current_rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def build_scale_dataset(
    experiment: ExperimentConfig,
    *,
    environment_replicas: int,
    metric_count: int,
    positive_checkpoint_metric_count: int | None = None,
) -> AnalysisDataset:
    positive_metric_count = (
        metric_count
        if positive_checkpoint_metric_count is None
        else positive_checkpoint_metric_count
    )
    if environment_replicas < 1 or metric_count < 1:
        raise ValueError("benchmark dimensions must be positive")
    if positive_metric_count < metric_count:
        raise ValueError(
            "positive checkpoint metric count cannot be below round-zero count"
        )
    metric_names = (
        "expected_reward",
        *(f"metric_{index:02d}" for index in range(1, positive_metric_count)),
    )
    run_settings_hash = _digest("settings", domain="analysis-scale-benchmark")
    provenance = tuple(
        (
            name,
            _digest(name, domain="analysis-scale-benchmark-provenance"),
        )
        for name in (
            "analysis_code_hash",
            "dependency_lock_hash",
            "dirty_tree_hash",
            "environment_digest",
        )
    )
    observations: list[CheckpointObservation] = []
    for environment_index, environment in enumerate(experiment.environments):
        frontier_hash = _digest(
            environment_index,
            domain="analysis-scale-benchmark-frontier",
        )
        frontier = CertifiedFrontier(
            semantic_hash=frontier_hash,
            zero_information_reward=0.0,
            maximum_reward=1.0,
            points=((0.0, 0.0, 0.0), (1.0, 0.5, 0.75)),
        )
        semantics = (("frontier", frontier_hash),)
        condition_hash = _digest(
            environment_index,
            domain="analysis-scale-benchmark-condition",
        )
        for agent_index, agent in enumerate(experiment.agents):
            agent_hash = _digest(
                agent_index,
                domain="analysis-scale-benchmark-agent",
            )
            for environment_replica in range(environment_replicas):
                for algorithm_replica in range(experiment.algorithm_replicas):
                    run_key = (
                        environment_index,
                        agent_index,
                        environment_replica,
                        algorithm_replica,
                    )
                    run_hash = _digest(
                        run_key,
                        domain="analysis-scale-benchmark-run",
                    )
                    content_hash = _digest(
                        run_key,
                        domain="analysis-scale-benchmark-content",
                    )
                    cell_hash = _digest(
                        run_key,
                        domain="analysis-scale-benchmark-cell",
                    )
                    for round_index in experiment.checkpoints.rounds:
                        checkpoint_metric_count = (
                            metric_count if round_index == 0 else positive_metric_count
                        )
                        base = (
                            environment_replica
                            + algorithm_replica / 10
                            + round_index / 100
                        )
                        metrics = tuple(
                            (name, base + index / 1_000)
                            for index, name in enumerate(
                                metric_names[:checkpoint_metric_count]
                            )
                        )
                        observations.append(
                            CheckpointObservation(
                                run_hash=run_hash,
                                content_hash=content_hash,
                                phase=AnalysisPhase.CALIBRATION,
                                confirmatory_frozen=False,
                                freeze_hash=None,
                                analysis_registration_hash=None,
                                condition_hash=condition_hash,
                                environment_kind=environment.kind.value,
                                agent_kind=agent.kind.value,
                                agent_hash=agent_hash,
                                environment_replica=environment_replica,
                                algorithm_replica=algorithm_replica,
                                round_index=round_index,
                                metrics=metrics,
                                semantic_hashes=semantics,
                                frontier=frontier,
                                run_settings_hash=run_settings_hash,
                                provenance=provenance,
                                cell_hash=cell_hash,
                            )
                        )
    return AnalysisDataset(tuple(observations))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/symbolic-calibration-v1.json"),
    )
    parser.add_argument("--environment-replicas", type=int)
    parser.add_argument("--metrics", type=int, default=29)
    parser.add_argument(
        "--positive-checkpoint-metrics",
        type=int,
        help="metric count at checkpoints after round zero (defaults to --metrics)",
    )
    parser.add_argument(
        "--skip-pooling",
        action="store_true",
        help="measure dataset construction only",
    )
    arguments = parser.parse_args()
    experiment = load_experiment_config(arguments.config)
    environment_replicas = (
        experiment.environment_replicas
        if arguments.environment_replicas is None
        else arguments.environment_replicas
    )
    expected = (
        len(experiment.environments)
        * len(experiment.agents)
        * environment_replicas
        * experiment.algorithm_replicas
        * len(experiment.checkpoints.rounds)
    )
    rss_before = _current_rss_mib()
    started = time.perf_counter()
    dataset = build_scale_dataset(
        experiment,
        environment_replicas=environment_replicas,
        metric_count=arguments.metrics,
        positive_checkpoint_metric_count=arguments.positive_checkpoint_metrics,
    )
    dataset_elapsed = time.perf_counter() - started
    pool_started = time.perf_counter()
    pools = () if arguments.skip_pooling else pool_checkpoints(dataset)
    pool_elapsed = time.perf_counter() - pool_started
    elapsed = time.perf_counter() - started
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "algorithm_replicas": experiment.algorithm_replicas,
                "checkpoint_count": len(experiment.checkpoints.rounds),
                "dataset_hash": dataset.scientific_hash,
                "dataset_elapsed_seconds": round(dataset_elapsed, 3),
                "environment_replicas": environment_replicas,
                "maximum_rss_mib": round(maximum_rss, 3),
                "checkpoint_zero_metric_count": arguments.metrics,
                "metric_count": arguments.metrics,
                "positive_checkpoint_metric_count": (
                    arguments.metrics
                    if arguments.positive_checkpoint_metrics is None
                    else arguments.positive_checkpoint_metrics
                ),
                "observation_count": len(dataset.observations),
                "pool_elapsed_seconds": round(pool_elapsed, 3),
                "pooled_checkpoint_count": len(pools),
                "rss_before_mib": round(rss_before, 3),
                "rss_increment_upper_bound_mib": round(
                    max(0.0, maximum_rss - rss_before),
                    3,
                ),
                "total_elapsed_seconds": round(elapsed, 3),
            },
            sort_keys=True,
        )
    )
    if len(dataset.observations) != expected:
        raise AssertionError("benchmark observation inventory is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
