"""Order-independent local execution of typed experiment sweeps."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
        if len({cell.cell_hash for cell in cells}) != len(cells):
            raise ValueError("sweep contains duplicate run cells")
        if max_workers == 1:
            results = [self.executor.execute(experiment, cell) for cell in cells]
        else:
            results = []
            remaining = iter(cells)
            pool = ThreadPoolExecutor(max_workers=max_workers)
            pending: set[Future[RunResult]] = set()
            try:
                for _ in range(max_workers):
                    try:
                        cell = next(remaining)
                    except StopIteration:
                        break
                    pending.add(pool.submit(self.executor.execute, experiment, cell))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    completed = [future.result() for future in done]
                    results.extend(completed)
                    for _ in completed:
                        try:
                            cell = next(remaining)
                        except StopIteration:
                            break
                        pending.add(
                            pool.submit(self.executor.execute, experiment, cell)
                        )
            except BaseException:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        return tuple(sorted(results, key=lambda result: result.run_hash))
