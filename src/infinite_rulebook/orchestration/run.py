"""Restartable training execution with isolated checkpoint evaluation."""

from __future__ import annotations

import copy
import fcntl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from infinite_rulebook.orchestration.artifacts import (
    ArtifactStore,
    EventJournal,
    ScientificArtifactError,
    validate_artifact_tree,
    write_frontier_bundle,
)
from infinite_rulebook.orchestration.config import ExperimentConfig, RunCell
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.provenance import (
    ScientificProvenance,
    collect_provenance,
    collect_runtime_metadata,
)
from infinite_rulebook.orchestration.seeds import RunSeeds, SeedBank
from infinite_rulebook.orchestration.semantics import semantic_hashes

RUNNER_VERSION = "symbolic-runner.v1"


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    """Serialize executors that resolve to the same run directory."""

    path.mkdir(parents=True, exist_ok=True)
    with (path / ".run.lock").open("ab") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class ExperimentAdapter(Protocol):
    """Replayable scientific logic owned by a simulator/agent integration."""

    def initial_state(self, cell: RunCell, seeds: RunSeeds) -> Any: ...

    def training_event(
        self,
        state: Any,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> Any: ...

    def apply_training_event(self, state: Any, payload: Any) -> Any: ...

    def checkpoint(
        self,
        state: Any,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
        semantic_hashes: dict[str, str],
    ) -> Any: ...

    def state_fingerprint(self, state: Any) -> str: ...

    def frontier(self, cell: RunCell) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RunResult:
    run_hash: str
    path: Path
    complete: bool
    scientific_content_hash: str | None
    event_count: int


def run_identity(
    experiment: ExperimentConfig,
    cell: RunCell,
    seeds: RunSeeds,
    provenance: ScientificProvenance,
) -> str:
    return scientific_hash(
        {
            "runner_version": RUNNER_VERSION,
            "run_settings": experiment.resolved_run_settings(),
            "cell": asdict(cell),
            "seeds": asdict(seeds),
            "provenance": provenance.to_dict(),
        },
        domain="run-identity",
    )


@dataclass(slots=True)
class RunExecutor:
    artifact_root: Path
    adapter_factory: Callable[[], ExperimentAdapter]
    provenance: ScientificProvenance = field(default_factory=collect_provenance)

    def _frontier_store(
        self,
        experiment: ExperimentConfig,
        cell: RunCell,
        hashes: dict[str, str],
    ) -> ArtifactStore:
        del experiment, cell
        return ArtifactStore(
            Path(self.artifact_root) / "_frontiers" / hashes["frontier"]
        )

    def execute(
        self,
        experiment: ExperimentConfig,
        cell: RunCell,
        *,
        stop_after_new_events: int | None = None,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        adapter = self.adapter_factory()
        seeds = SeedBank(experiment.master_seed).for_cell(cell)
        provenance = self.provenance
        run_hash = run_identity(experiment, cell, seeds, provenance)
        hashes = semantic_hashes(cell, analysis_code_hash=provenance.analysis_code_hash)
        store = ArtifactStore.for_run(self.artifact_root, experiment.name, run_hash)
        with _run_lock(store.path):
            return self._execute_locked(
                experiment,
                cell,
                adapter,
                seeds,
                provenance,
                run_hash,
                hashes,
                store,
                started=started,
                stop_after_new_events=stop_after_new_events,
                runtime_metadata=runtime_metadata,
            )

    def _execute_locked(
        self,
        experiment: ExperimentConfig,
        cell: RunCell,
        adapter: ExperimentAdapter,
        seeds: RunSeeds,
        provenance: ScientificProvenance,
        run_hash: str,
        hashes: dict[str, str],
        store: ArtifactStore,
        *,
        started: float,
        stop_after_new_events: int | None,
        runtime_metadata: dict[str, Any] | None,
    ) -> RunResult:
        manifest_path = store.path / "manifest.json"
        if manifest_path.exists():
            artifacts = validate_artifact_tree(
                store.path, expected_semantic_hashes=hashes
            )
            manifest = next(
                artifact
                for artifact in artifacts
                if artifact.artifact_type == "run-manifest"
            )
            return RunResult(
                run_hash,
                store.path,
                True,
                manifest.payload["scientific_content_hash"],
                len(EventJournal(store, hashes).events()),
            )

        store.write(
            "config.resolved.json",
            "resolved-run-config",
            hashes,
            {
                "run_settings": experiment.resolved_run_settings(),
                "cell": asdict(cell),
                "seeds": asdict(seeds),
                "provenance": provenance.to_dict(),
                "run_hash": run_hash,
            },
        )
        frontier_store = self._frontier_store(experiment, cell, hashes)
        frontier = adapter.frontier(cell)
        write_frontier_bundle(
            frontier_store,
            {"frontier": hashes["frontier"]},
            curve=frontier["curve"],
            witnesses=frontier["witnesses"],
            certificates=frontier["certificates"],
            diagnostics=frontier["diagnostics"],
        )
        frontier_artifacts = validate_artifact_tree(
            frontier_store.path,
            expected_semantic_hashes={"frontier": hashes["frontier"]},
        )
        frontier_manifest = next(
            artifact
            for artifact in frontier_artifacts
            if artifact.artifact_type == "frontier-manifest"
        )
        store.write(
            "frontier-reference.json",
            "frontier-reference",
            hashes,
            {
                "frontier_hash": hashes["frontier"],
                "artifact_hash": frontier_manifest.scientific_hash,
            },
        )

        journal = EventJournal(store, hashes)
        state = adapter.initial_state(cell, seeds)
        existing_events = journal.events()
        checkpoint_rounds = set(experiment.checkpoints.rounds)
        for event in existing_events:
            if event.sequence in checkpoint_rounds:
                self._checkpoint(
                    adapter,
                    store,
                    hashes,
                    state,
                    event.sequence,
                    cell,
                    seeds,
                )
            state = adapter.apply_training_event(state, event.payload)

        existing_event_count = len(existing_events)
        for round_index in range(existing_event_count, experiment.horizon):
            if round_index in checkpoint_rounds:
                self._checkpoint(
                    adapter, store, hashes, state, round_index, cell, seeds
                )
            payload = adapter.training_event(state, round_index, cell, seeds)
            journal.append(f"round:{round_index}", "training-step", payload)
            state = adapter.apply_training_event(state, payload)
            new_events = round_index - existing_event_count + 1
            if (
                stop_after_new_events is not None
                and new_events >= stop_after_new_events
            ):
                return RunResult(
                    run_hash,
                    store.path,
                    False,
                    None,
                    len(journal.events()),
                )

        if experiment.horizon in checkpoint_rounds:
            self._checkpoint(
                adapter,
                store,
                hashes,
                state,
                experiment.horizon,
                cell,
                seeds,
            )
        final_fingerprint = adapter.state_fingerprint(state)
        store.write(
            "metrics.json",
            "run-metrics",
            hashes,
            {
                "completed_rounds": experiment.horizon,
                "event_count": len(journal.events()),
                "final_state_hash": final_fingerprint,
                "phase": experiment.phase,
                "confirmatory_frozen": False,
            },
        )
        recorded_runtime = collect_runtime_metadata(
            wall_time_seconds=time.perf_counter() - started
        )
        recorded_runtime.update(runtime_metadata or {})
        manifest = store.finalize(
            hashes,
            runtime_metadata=recorded_runtime,
        )
        validate_artifact_tree(store.path, expected_semantic_hashes=hashes)
        return RunResult(
            run_hash,
            store.path,
            True,
            manifest.payload["scientific_content_hash"],
            len(journal.events()),
        )

    def _checkpoint(
        self,
        adapter: ExperimentAdapter,
        store: ArtifactStore,
        hashes: dict[str, str],
        state: Any,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> None:
        relative_path = f"checkpoints/{round_index:08d}.json"
        path = store.path / relative_path
        current = adapter.state_fingerprint(state)
        if path.exists():
            checkpoint = store.read(relative_path, expected_semantic_hashes=hashes)
            payload = checkpoint.payload
            if (
                payload["round"] != round_index
                or payload["training_state_before"] != current
                or payload["training_state_after"] != current
                or payload["evaluation_seed"] != seeds.evaluation
                or payload["deployment_seed"] != seeds.deployment
            ):
                raise ScientificArtifactError(
                    "stored checkpoint is incompatible with replayed state"
                )
            return
        evaluation_adapter, evaluation_state = copy.deepcopy((adapter, state))
        evaluation_before = evaluation_adapter.state_fingerprint(evaluation_state)
        if evaluation_before != current:
            raise ScientificArtifactError(
                "checkpoint clone changed the scientific training state"
            )
        payload = evaluation_adapter.checkpoint(
            evaluation_state,
            round_index,
            cell,
            seeds,
            hashes,
        )
        evaluation_after = evaluation_adapter.state_fingerprint(evaluation_state)
        if evaluation_before != evaluation_after:
            raise ScientificArtifactError(
                "checkpoint evaluation mutated its frozen state"
            )
        after = adapter.state_fingerprint(state)
        if current != after:
            raise ScientificArtifactError("checkpoint changed training state")
        store.write(
            relative_path,
            "run-checkpoint",
            hashes,
            {
                "round": round_index,
                "training_state_before": current,
                "training_state_after": after,
                "evaluation_seed": seeds.evaluation,
                "deployment_seed": seeds.deployment,
                "result": payload,
            },
        )
