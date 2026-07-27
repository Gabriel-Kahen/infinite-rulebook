"""Restartable training execution with isolated checkpoint evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from infinite_rulebook.orchestration.seeds import RunSeeds, SeedBank
from infinite_rulebook.orchestration.semantics import semantic_hashes

RUNNER_VERSION = "symbolic-runner.v1"


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
) -> str:
    return scientific_hash(
        {
            "runner_version": RUNNER_VERSION,
            "run_settings": experiment.resolved_run_settings(),
            "cell": asdict(cell),
            "seeds": asdict(seeds),
        },
        domain="run-identity",
    )


@dataclass(slots=True)
class RunExecutor:
    artifact_root: Path
    adapter: ExperimentAdapter

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
        seeds = SeedBank(experiment.master_seed).for_cell(cell)
        run_hash = run_identity(experiment, cell, seeds)
        hashes = semantic_hashes(cell)
        store = ArtifactStore.for_run(self.artifact_root, experiment.name, run_hash)
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
                "run_hash": run_hash,
            },
        )
        frontier_store = self._frontier_store(experiment, cell, hashes)
        frontier = self.adapter.frontier(cell)
        write_frontier_bundle(
            frontier_store,
            {"frontier": hashes["frontier"]},
            curve=frontier["curve"],
            witnesses=frontier["witnesses"],
            certificates=frontier["certificates"],
            diagnostics=frontier["diagnostics"],
        )
        frontier_manifest = frontier_store.read("frontier/manifest.json")
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
        state = self.adapter.initial_state(cell, seeds)
        for event in journal.events():
            state = self.adapter.apply_training_event(state, event.payload)

        existing_event_count = len(journal.events())
        checkpoint_rounds = set(experiment.checkpoints.rounds)
        for round_index in range(existing_event_count, experiment.horizon):
            if round_index in checkpoint_rounds:
                self._checkpoint(store, hashes, state, round_index, cell, seeds)
            payload = self.adapter.training_event(state, round_index, cell, seeds)
            journal.append(f"round:{round_index}", "training-step", payload)
            state = self.adapter.apply_training_event(state, payload)
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
            self._checkpoint(store, hashes, state, experiment.horizon, cell, seeds)
        final_fingerprint = self.adapter.state_fingerprint(state)
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
        manifest = store.finalize(hashes, runtime_metadata=runtime_metadata)
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
        store: ArtifactStore,
        hashes: dict[str, str],
        state: Any,
        round_index: int,
        cell: RunCell,
        seeds: RunSeeds,
    ) -> None:
        relative_path = f"checkpoints/{round_index:08d}.json"
        path = store.path / relative_path
        if path.exists():
            store.read(relative_path, expected_semantic_hashes=hashes)
            return
        before = self.adapter.state_fingerprint(state)
        payload = self.adapter.checkpoint(state, round_index, cell, seeds)
        after = self.adapter.state_fingerprint(state)
        if before != after:
            raise ScientificArtifactError(
                "checkpoint evaluation mutated training state"
            )
        store.write(
            relative_path,
            "run-checkpoint",
            hashes,
            {
                "round": round_index,
                "training_state_before": before,
                "training_state_after": after,
                "evaluation_seed": seeds.evaluation,
                "deployment_seed": seeds.deployment,
                "result": payload,
            },
        )
