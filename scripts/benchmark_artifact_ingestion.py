"""Benchmark authenticated artifact ingestion without producing study evidence."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import time
from pathlib import Path

from infinite_rulebook.analysis.loading import load_run_trees
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.inventory import RawArtifactInventory


def _current_rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


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
    roots = tuple(arguments.artifact_root / tree.path for tree in run_trees)
    load_started = time.perf_counter()
    dataset = load_run_trees(
        roots,
        expected_run_settings=experiment.resolved_run_settings(),
    )
    load_elapsed = time.perf_counter() - load_started

    run_count = len(run_trees)
    projected_runs = arguments.projected_runs or run_count
    inventory_per_run = inventory_elapsed / run_count
    load_per_run = load_elapsed / run_count
    projected_seconds = projected_runs * (2.0 * inventory_per_run + load_per_run)
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "artifact_root": str(arguments.artifact_root),
                "budget_multiplier": arguments.budget_multiplier,
                "dataset_hash": dataset.scientific_hash,
                "inventory_elapsed_seconds": round(inventory_elapsed, 6),
                "inventory_seconds_per_run": round(inventory_per_run, 6),
                "load_elapsed_seconds": round(load_elapsed, 6),
                "load_seconds_per_run": round(load_per_run, 6),
                "maximum_rss_mib": round(maximum_rss, 3),
                "observation_count": len(dataset.observations),
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
