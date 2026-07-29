"""Deterministic, semantically named seed banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from infinite_rulebook.core.rng import CounterRNG
from infinite_rulebook.orchestration.config import RunCell
from infinite_rulebook.orchestration.hashing import scientific_hash


@dataclass(frozen=True, slots=True)
class RunSeeds:
    environment: int
    persistent_distractor: int
    aleatoric: int
    query_observation: int
    algorithm: int
    deployment: int
    evaluation: int
    frontier: int

    @property
    def seed_hash(self) -> str:
        return scientific_hash(asdict(self), domain="run-seeds")


@dataclass(frozen=True, slots=True)
class SeedBank:
    master_seed: int | str
    algorithm_master_seed: int | str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.master_seed, bool) or not isinstance(
            self.master_seed, (int, str)
        ):
            raise TypeError("master_seed must be an integer or string")
        if isinstance(self.algorithm_master_seed, bool) or (
            self.algorithm_master_seed is not None
            and not isinstance(self.algorithm_master_seed, (int, str))
        ):
            raise TypeError("algorithm_master_seed must be an integer, string, or None")

    def _derive(self, stream: str, replica: int) -> int:
        rng = CounterRNG(self.master_seed, stream="experiment.seed-bank.v1")
        return rng.uint64(stream, replica)

    def _derive_algorithm(self, stream: str, replica: int) -> int:
        master_seed = (
            self.master_seed
            if self.algorithm_master_seed is None
            else self.algorithm_master_seed
        )
        rng = CounterRNG(master_seed, stream="experiment.algorithm-seed-bank.v1")
        return rng.uint64(stream, replica)

    def for_cell(self, cell: RunCell) -> RunSeeds:
        environment_replica = cell.environment_replica
        algorithm_replica = cell.algorithm_replica
        return RunSeeds(
            environment=self._derive("environment", environment_replica),
            persistent_distractor=self._derive(
                "persistent-distractor", environment_replica
            ),
            aleatoric=self._derive("aleatoric", environment_replica),
            query_observation=self._derive("query-observation", environment_replica),
            algorithm=self._derive_algorithm("algorithm", algorithm_replica),
            deployment=self._derive_algorithm("deployment", algorithm_replica),
            evaluation=self._derive("evaluation", environment_replica),
            frontier=self._derive("frontier", 0),
        )

    def legacy_for_cell(self, cell: RunCell) -> RunSeeds:
        """Reconstruct pre-split seed trees for validating released v1 artifacts."""

        environment_replica = cell.environment_replica
        algorithm_replica = cell.algorithm_replica
        return RunSeeds(
            environment=self._derive("environment", environment_replica),
            persistent_distractor=self._derive(
                "persistent-distractor", environment_replica
            ),
            aleatoric=self._derive("aleatoric", environment_replica),
            query_observation=self._derive("query-observation", environment_replica),
            algorithm=self._derive("algorithm", algorithm_replica),
            deployment=self._derive("deployment", algorithm_replica),
            evaluation=self._derive("evaluation", environment_replica),
            frontier=self._derive("frontier", 0),
        )
