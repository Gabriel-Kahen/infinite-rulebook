"""Order-independent local execution of typed experiment sweeps."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from infinite_rulebook.orchestration.config import ExperimentConfig
from infinite_rulebook.orchestration.run import RunExecutor, RunResult


@dataclass(slots=True)
class SweepRunner:
    executor: RunExecutor

    def run(
        self,
        experiment: ExperimentConfig,
        *,
        max_workers: int = 1,
    ) -> tuple[RunResult, ...]:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        cells = experiment.cells()
        if max_workers == 1:
            results = [self.executor.execute(experiment, cell) for cell in cells]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(
                    pool.map(
                        lambda cell: self.executor.execute(experiment, cell),
                        cells,
                    )
                )
        return tuple(sorted(results, key=lambda result: result.run_hash))
