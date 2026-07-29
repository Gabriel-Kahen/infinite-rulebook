"""Run only the registered-shape, disjoint-seed ingestion performance probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter
from scripts.generate_ingestion_probe import build_ingestion_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_config", type=Path)
    parser.add_argument("probe_config", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    expected = build_ingestion_probe(
        load_experiment_config(arguments.calibration_config)
    )
    observed = load_experiment_config(arguments.probe_config)
    if observed != expected:
        raise ValueError("probe config differs from the registered-shape derivation")
    if os.path.lexists(arguments.artifact_root):
        raise ValueError(
            f"probe artifact root must not already exist: {arguments.artifact_root}"
        )
    results = SweepRunner(
        RunExecutor(arguments.artifact_root, ExactSymbolicAdapter)
    ).run(observed, max_workers=arguments.workers)
    print(
        json.dumps(
            {
                "artifact_root": str(arguments.artifact_root),
                "config_hash": observed.config_hash,
                "phase": observed.phase,
                "run_count": len(results),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
