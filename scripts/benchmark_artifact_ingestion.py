"""Benchmark authenticated artifact ingestion without producing study evidence."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import time
from pathlib import Path

from infinite_rulebook.analysis.loading import (
    _load_run_tree,
    _LoadingCaches,
    load_run_trees,
)
from infinite_rulebook.orchestration.artifacts import (
    ArtifactValidationSession,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.inventory import (
    RawArtifactInventory,
    _tree_metrics,
)


def _current_rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def _elapsed(operation) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def project_report_ingestion(
    *,
    frontier_validation: float,
    frontier_copy: float,
    frontier_raw: float,
    run_validation: float,
    run_raw: float,
    run_load: float,
    observed_probe: float,
    probe_runs: int,
    projected_runs: int,
) -> tuple[float, float, float, float]:
    """Separate fixed shared-frontier work from marginal run work."""

    fixed = 7.0 * frontier_validation + 3.0 * frontier_copy + 2.0 * frontier_raw
    marginal = 4.0 * run_validation + 2.0 * run_raw + run_load
    modeled_probe = fixed + probe_runs * marginal
    residual = max(0.0, (observed_probe - modeled_probe) / probe_runs)
    projected = fixed + projected_runs * (marginal + residual)
    return fixed, marginal, residual, projected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--side", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--experiment-name")
    parser.add_argument("--environment-replicas", type=int)
    parser.add_argument("--algorithm-replicas", type=int)
    parser.add_argument("--projected-runs", type=int)
    parser.add_argument("--budget-multiplier", type=float, default=2.0)
    arguments = parser.parse_args()
    if arguments.projected_runs is not None and arguments.projected_runs < 1:
        parser.error("--projected-runs must be positive")
    if arguments.budget_multiplier < 1.0:
        parser.error("--budget-multiplier must be at least 1")

    experiment = load_experiment_config(arguments.config)
    overrides = {
        name: value
        for name, value in (
            ("name", arguments.experiment_name),
            ("environment_replicas", arguments.environment_replicas),
            ("algorithm_replicas", arguments.algorithm_replicas),
        )
        if value is not None
    }
    if any(isinstance(value, int) and value < 1 for value in overrides.values()):
        parser.error("replica overrides must be positive")
    if overrides:
        experiment = dataclasses.replace(experiment, **overrides)
    rss_before = _current_rss_mib()
    inventory_started = time.perf_counter()
    inventory = RawArtifactInventory.create(
        arguments.artifact_root,
        experiment,
        side=arguments.side,
    )
    inventory_elapsed = time.perf_counter() - inventory_started
    run_trees = tuple(tree for tree in inventory.trees if tree.tree_type == "run")
    frontier_trees = tuple(
        tree for tree in inventory.trees if tree.tree_type == "frontier"
    )
    roots = tuple(arguments.artifact_root / tree.path for tree in run_trees)
    frontier_roots = tuple(
        (arguments.artifact_root / tree.path, tree.identity_hash)
        for tree in frontier_trees
    )
    load_started = time.perf_counter()
    dataset = load_run_trees(
        roots,
        expected_run_settings=experiment.resolved_run_settings(),
    )
    load_elapsed = time.perf_counter() - load_started

    frontier_session = ArtifactValidationSession()

    def validate_frontiers() -> None:
        for root, identity_hash in frontier_roots:
            validate_artifact_tree(
                root,
                expected_semantic_hashes={"frontier": identity_hash},
                session=frontier_session,
            )

    frontier_validation_elapsed = _elapsed(validate_frontiers)
    frontier_copy_elapsed = _elapsed(validate_frontiers)
    frontier_raw_elapsed = _elapsed(
        lambda: tuple(_tree_metrics(root) for root, _ in frontier_roots)
    )

    validation_session = ArtifactValidationSession()
    for root in roots:
        validate_artifact_tree(root, session=validation_session)
    marginal_validation_elapsed = _elapsed(
        lambda: tuple(
            validate_artifact_tree(root, session=validation_session) for root in roots
        )
    )
    marginal_raw_elapsed = _elapsed(
        lambda: tuple(_tree_metrics(root) for root in roots)
    )

    loading_session = ArtifactValidationSession()
    loading_caches = _LoadingCaches.create()

    def load_marginal_runs() -> None:
        for root in roots:
            _load_run_tree(
                root,
                expected_phase=None,
                expected_freeze_hash=None,
                expected_run_settings=experiment.resolved_run_settings(),
                validation_session=loading_session,
                caches=loading_caches,
            )

    load_marginal_runs()
    marginal_load_elapsed = _elapsed(load_marginal_runs)

    run_count = len(run_trees)
    projected_runs = arguments.projected_runs or run_count
    validation_per_run = marginal_validation_elapsed / run_count
    raw_per_run = marginal_raw_elapsed / run_count
    load_per_run = marginal_load_elapsed / run_count
    observed_probe_seconds = 2.0 * inventory_elapsed + load_elapsed
    (
        fixed_seconds,
        marginal_seconds_per_run,
        residual_seconds_per_run,
        projected_seconds,
    ) = project_report_ingestion(
        frontier_validation=frontier_validation_elapsed,
        frontier_copy=frontier_copy_elapsed,
        frontier_raw=frontier_raw_elapsed,
        run_validation=validation_per_run,
        run_raw=raw_per_run,
        run_load=load_per_run,
        observed_probe=observed_probe_seconds,
        probe_runs=run_count,
        projected_runs=projected_runs,
    )
    modeled_probe_seconds = fixed_seconds + run_count * marginal_seconds_per_run
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "artifact_root": str(arguments.artifact_root),
                "budget_multiplier": arguments.budget_multiplier,
                "dataset_hash": dataset.scientific_hash,
                "fixed_frontier_cache_copy_seconds": round(
                    frontier_copy_elapsed,
                    6,
                ),
                "fixed_frontier_raw_hash_seconds": round(
                    frontier_raw_elapsed,
                    6,
                ),
                "fixed_frontier_validation_seconds": round(
                    frontier_validation_elapsed,
                    6,
                ),
                "fixed_report_ingestion_seconds": round(fixed_seconds, 6),
                "frontier_tree_count": len(frontier_trees),
                "inventory_elapsed_seconds": round(inventory_elapsed, 6),
                "load_elapsed_seconds": round(load_elapsed, 6),
                "maximum_rss_mib": round(maximum_rss, 3),
                "modeled_probe_report_seconds": round(
                    modeled_probe_seconds,
                    6,
                ),
                "marginal_run_load_seconds_per_run": round(load_per_run, 6),
                "marginal_run_raw_hash_seconds_per_run": round(raw_per_run, 6),
                "marginal_run_validation_seconds_per_run": round(
                    validation_per_run,
                    6,
                ),
                "observation_count": len(dataset.observations),
                "observed_probe_report_seconds": round(
                    observed_probe_seconds,
                    6,
                ),
                "operational_budget_hours": round(
                    projected_seconds * arguments.budget_multiplier / 3600,
                    3,
                ),
                "projected_report_ingestion_hours": round(
                    projected_seconds / 3600,
                    3,
                ),
                "projected_runs_per_root": projected_runs,
                "raw_run_byte_size": sum(tree.byte_size for tree in run_trees),
                "raw_run_file_count": sum(tree.file_count for tree in run_trees),
                "residual_seconds_per_run": round(
                    residual_seconds_per_run,
                    6,
                ),
                "rss_before_mib": round(rss_before, 3),
                "rss_increment_upper_bound_mib": round(
                    max(0.0, maximum_rss - rss_before),
                    3,
                ),
                "run_count": run_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
