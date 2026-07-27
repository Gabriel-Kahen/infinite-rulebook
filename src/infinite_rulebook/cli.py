"""Command-line entry points for symbolic pilot artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from infinite_rulebook.orchestration.artifacts import validate_artifact_tree
from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.sweep import SweepRunner
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infinite-rulebook")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run a bounded symbolic pilot")
    pilot.add_argument("config", type=Path)
    pilot.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    pilot.add_argument("--workers", type=int, default=1)
    validate = subparsers.add_parser("validate", help="validate an artifact tree")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        artifacts = validate_artifact_tree(arguments.path)
        print(json.dumps({"artifacts": len(artifacts), "valid": True}))
        return 0
    experiment = load_experiment_config(arguments.config)
    executor = RunExecutor(arguments.artifact_root, ExactSymbolicAdapter)
    results = SweepRunner(executor).run(
        experiment,
        max_workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "phase": experiment.phase,
                "confirmatory_frozen": False,
                "runs": [
                    {
                        "run_hash": result.run_hash,
                        "complete": result.complete,
                        "scientific_content_hash": result.scientific_content_hash,
                        "path": str(result.path),
                    }
                    for result in results
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
