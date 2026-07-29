"""Derive a disjoint-seed, full-shape artifact-ingestion performance probe."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    load_experiment_config,
)
from infinite_rulebook.studies.symbolic_construct import (
    verify_symbolic_calibration_design,
)

PROBE_NAME = "symbolic-artifact-ingestion-probe-v1"
PROBE_MASTER_SEED = "irb-symbolic-artifact-ingestion-probe-v1"
PROBE_ALGORITHM_MASTER_SEED = "irb-symbolic-artifact-ingestion-probe-algorithm-v1"


def build_ingestion_probe(calibration: ExperimentConfig) -> ExperimentConfig:
    """Keep the producer shape while excluding every registered study seed."""

    verify_symbolic_calibration_design(calibration)
    return dataclasses.replace(
        calibration,
        name=PROBE_NAME,
        phase="calibration",
        master_seed=PROBE_MASTER_SEED,
        algorithm_master_seed=PROBE_ALGORITHM_MASTER_SEED,
        environment_replicas=1,
        algorithm_replicas=1,
        confirmatory_freeze=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_config", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    probe = build_ingestion_probe(load_experiment_config(arguments.calibration_config))
    content = (
        json.dumps(
            probe.resolved_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if arguments.output.exists():
        if (
            not arguments.output.is_file()
            or arguments.output.read_text(encoding="utf-8") != content
        ):
            raise ValueError(
                f"refusing to overwrite a different probe: {arguments.output}"
            )
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(arguments.output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    print(
        json.dumps(
            {
                "algorithm_replicas": probe.algorithm_replicas,
                "config_hash": probe.config_hash,
                "environment_replicas": probe.environment_replicas,
                "output": str(arguments.output),
                "run_count": len(probe.cells()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
