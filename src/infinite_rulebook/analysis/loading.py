"""Authenticated loading of completed experiment run trees."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    AnalysisPhase,
    CertifiedFrontier,
    CheckpointObservation,
    ExpectedGroup,
)
from infinite_rulebook.orchestration.artifacts import (
    ArtifactValidationSession,
    validate_artifact_tree,
)
from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    RunCell,
    run_cell_from_dict,
    run_cell_identity_payload,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.provenance import ScientificProvenance
from infinite_rulebook.orchestration.run import RUNNER_VERSION
from infinite_rulebook.orchestration.seeds import RunSeeds


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnalysisError(f"{name} must be an object with string keys")
    return value


def _normalized_record(value: object, name: str) -> dict[str, Any]:
    raw = (
        asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
    )
    try:
        normalized = json.loads(json.dumps(raw, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{name} must be a JSON-safe scientific record") from error
    return _mapping(normalized, name)


def analysis_condition_hash(cell: object) -> str:
    """Return the agent/replica-independent registered condition identity."""

    normalized = _normalized_record(cell, "run cell")
    payload = {
        name: value
        for name, value in normalized.items()
        if name not in {"agent", "environment_replica", "algorithm_replica"}
    }
    return scientific_hash(payload, domain="analysis-condition")


def analysis_agent_hash(agent: object) -> str:
    """Return the complete registered agent-configuration identity."""

    return scientific_hash(
        _normalized_record(agent, "agent config"),
        domain="analysis-agent",
    )


def expected_groups_from_experiment(
    experiment: ExperimentConfig,
) -> tuple[ExpectedGroup, ...]:
    """Derive the exact checkpoint inventory sealed by an experiment config."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    groups: dict[tuple[str, str], ExpectedGroup] = {}
    for environment in experiment.environments:
        for agent in experiment.agents:
            cell = RunCell(
                environment=environment,
                feedback=experiment.feedback,
                reward=experiment.reward,
                agent=agent,
                solver=experiment.solver,
                environment_replica=0,
                algorithm_replica=0,
            )
            normalized = _normalized_record(cell, "run cell")
            environment_record = _mapping(
                normalized["environment"],
                "cell environment",
            )
            agent_record = _mapping(normalized["agent"], "cell agent")
            condition_hash = analysis_condition_hash(normalized)
            agent_hash = analysis_agent_hash(agent_record)
            expected = ExpectedGroup(
                condition_hash=condition_hash,
                agent_hash=agent_hash,
                environment_kind=environment_record["kind"],
                agent_kind=agent_record["kind"],
                checkpoints=experiment.checkpoints.rounds,
                environment_replicas=experiment.environment_replicas,
                algorithm_replicas=experiment.algorithm_replicas,
            )
            key = condition_hash, agent_hash
            if key in groups and groups[key] != expected:
                raise AnalysisError(
                    "experiment contains an inconsistent analysis group"
                )
            groups[key] = expected
    return tuple(sorted(groups.values()))


def _phase(value: object) -> AnalysisPhase:
    try:
        return AnalysisPhase(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"unrecognized experiment phase: {value!r}") from error


def _freeze_hash(*records: dict[str, Any]) -> str | None:
    candidates: set[str] = set()
    direct_names = {
        "freeze_hash",
        "confirmatory_freeze_hash",
        "confirmatory_seal_hash",
        "seal_hash",
    }

    def visit(value: object, *, inside_freeze: bool = False) -> None:
        if not isinstance(value, dict):
            return
        for name, child in value.items():
            relevant = inside_freeze or "freeze" in name or "seal" in name
            if (
                relevant
                and name in direct_names
                and isinstance(child, str)
                and is_sha256(child)
            ):
                candidates.add(child)
            if isinstance(child, dict) and relevant:
                visit(child, inside_freeze=True)

    for record in records:
        visit(record)
    if len(candidates) > 1:
        raise AnalysisError(
            "run records contain conflicting confirmatory freeze hashes"
        )
    return next(iter(candidates), None)


def _numeric_metrics(result: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    metrics: dict[str, float] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        number = float(value)
        if not math.isfinite(number):
            raise AnalysisError(f"checkpoint metric {name!r} must be finite")
        metrics[sys.intern(name)] = number

    for name in (
        "expected_reward",
        "hidden_expected_reward",
        "post_query_mean_hidden_expected_reward",
        "public_reward",
    ):
        if name in result:
            add(name, result[name])
    for section in ("information", "novelty", "support", "compute"):
        values = _mapping(result.get(section), f"checkpoint {section}")
        for name, value in values.items():
            add(f"{section}.{name}", value)

    information = _mapping(result.get("information"), "checkpoint information")
    reward_relevant = information.get("reward_relevant_nats")
    shared_core = information.get("shared_core_nats")
    total = information.get("total_acquired_nats")
    distractor = information.get("persistent_distractor_nats")
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (reward_relevant, shared_core)
    ):
        add(
            "relevant_information_nats",
            float(reward_relevant) + float(shared_core),
        )
    add("total_information_nats", total)
    add("distractor_information_nats", distractor)
    if "expected_reward" not in metrics:
        raise AnalysisError("checkpoint is missing expected_reward")
    return tuple(sorted(metrics.items()))


def _frontier_summary(
    run_root: Path,
    frontier_hash: str,
    frontier_manifest_hash: str,
    validation_session: ArtifactValidationSession,
    summary_cache: dict[tuple[str, str, str], CertifiedFrontier],
) -> CertifiedFrontier:
    try:
        artifact_root = Path(os.path.abspath(run_root)).parents[1]
    except IndexError as error:
        raise AnalysisError("run tree is not under an experiment directory") from error
    frontier_root = artifact_root / "_frontiers" / frontier_hash
    key = os.path.abspath(frontier_root), frontier_hash, frontier_manifest_hash
    cached = summary_cache.get(key)
    if cached is None:
        cached = _authenticated_frontier_summary(
            frontier_root,
            frontier_hash,
            frontier_manifest_hash,
            validation_session,
        )
        summary_cache[key] = cached
    return cached


@dataclass(slots=True)
class _LoadingCaches:
    frontiers: dict[tuple[str, str, str], CertifiedFrontier]
    provenances: dict[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ]
    semantics: dict[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ]

    @classmethod
    def create(cls) -> _LoadingCaches:
        return cls({}, {}, {})


def _authenticated_frontier_summary(
    frontier_root: Path,
    frontier_hash: str,
    frontier_manifest_hash: str,
    validation_session: ArtifactValidationSession,
) -> CertifiedFrontier:
    """Derive a summary from one fully validated immutable frontier tree."""

    artifacts = validate_artifact_tree(
        frontier_root,
        expected_semantic_hashes={"frontier": frontier_hash},
        session=validation_session,
    )
    manifests = [
        item for item in artifacts if item.artifact_type == "frontier-manifest"
    ]
    curves = [item for item in artifacts if item.artifact_type == "frontier-curve"]
    if (
        len(manifests) != 1
        or manifests[0].scientific_hash != frontier_manifest_hash
        or len(curves) != 1
    ):
        raise AnalysisError(
            "validated frontier does not match the run's manifest reference"
        )
    curve = curves[0]
    payload = _mapping(curve.payload, "frontier curve")
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise AnalysisError("frontier points must be an array")
    points = []
    for raw in raw_points:
        point = _mapping(raw, "frontier point")
        points.append(
            (
                point["target_reward"],
                point["lower_information"],
                point["upper_information"],
            )
        )
    return CertifiedFrontier(
        semantic_hash=frontier_hash,
        zero_information_reward=payload["zero_information_reward"],
        maximum_reward=payload["maximum_reward"],
        points=tuple(points),
    )


def _load_run_tree(
    root: str | Path,
    *,
    expected_phase: AnalysisPhase | None,
    expected_freeze_hash: str | None,
    expected_run_settings: dict[str, Any] | None,
    validation_session: ArtifactValidationSession,
    caches: _LoadingCaches,
) -> tuple[CheckpointObservation, ...]:
    """Load one completed run only after the existing full-tree validator passes."""

    path = Path(root)
    artifacts = validate_artifact_tree(
        path,
        session=validation_session,
    )
    by_type: dict[str, list[Any]] = {}
    for artifact in artifacts:
        by_type.setdefault(artifact.artifact_type, []).append(artifact)
    try:
        config_envelope = by_type["resolved-run-config"][0]
        metrics_envelope = by_type["run-metrics"][0]
        manifest = by_type["run-manifest"][0]
        frontier_reference = by_type["frontier-reference"][0]
        checkpoints = by_type["run-checkpoint"]
    except (KeyError, IndexError) as error:
        raise AnalysisError("validated run is missing an analysis artifact") from error
    if any(
        len(by_type[name]) != 1
        for name in (
            "resolved-run-config",
            "run-metrics",
            "run-manifest",
            "frontier-reference",
        )
    ):
        raise AnalysisError("validated run contains duplicate singleton artifacts")

    config = _mapping(config_envelope.payload, "resolved run config")
    run_settings = _mapping(config.get("run_settings"), "resolved run settings")
    if expected_run_settings is not None and _normalized_record(
        run_settings, "resolved run settings"
    ) != _normalized_record(expected_run_settings, "expected run settings"):
        raise AnalysisError("run settings do not match the supplied experiment config")
    cell = _mapping(config.get("cell"), "resolved run cell")
    seeds = _mapping(config.get("seeds"), "resolved run seeds")
    provenance = _mapping(config.get("provenance"), "resolved run provenance")
    metrics = _mapping(metrics_envelope.payload, "run metrics")
    config_phase = _phase(run_settings.get("phase"))
    metric_phase = _phase(metrics.get("phase"))
    if config_phase is not metric_phase:
        raise AnalysisError("resolved config and metrics disagree on experiment phase")
    if expected_phase is not None and config_phase is not expected_phase:
        raise AnalysisError(
            f"expected {expected_phase.value} data, got {config_phase.value}"
        )
    frozen = metrics.get("confirmatory_frozen", False)
    if not isinstance(frozen, bool):
        raise AnalysisError("confirmatory_frozen must be a boolean")
    seal = _freeze_hash(run_settings, config, metrics)
    registration_hash = run_settings.get("analysis_registration_hash")
    if registration_hash is not None and not is_sha256(registration_hash):
        raise AnalysisError("analysis_registration_hash must be a SHA-256 digest")
    if expected_freeze_hash is not None:
        if not is_sha256(expected_freeze_hash):
            raise ValueError("expected_freeze_hash must be a SHA-256 digest")
        if seal != expected_freeze_hash:
            raise AnalysisError("run is bound to a different confirmatory freeze")
    if config_phase is AnalysisPhase.CONFIRMATORY and (not frozen or seal is None):
        raise AnalysisError("confirmatory run is not bound to a frozen protocol")
    if config_phase is not AnalysisPhase.CONFIRMATORY and frozen:
        raise AnalysisError("non-confirmatory run is incorrectly marked frozen")

    run_hash = config.get("run_hash")
    content_hash = manifest.payload.get("scientific_content_hash")
    if not is_sha256(run_hash) or not is_sha256(content_hash):
        raise AnalysisError("run identity hashes are invalid")
    try:
        typed_cell = run_cell_from_dict(cell)
        typed_cell_payload = run_cell_identity_payload(cell)
        typed_seeds = RunSeeds(**seeds)
        typed_provenance = ScientificProvenance(**provenance)
    except (TypeError, ValueError) as error:
        raise AnalysisError("run identity inputs are malformed") from error
    expected_run_hash = scientific_hash(
        {
            "runner_version": RUNNER_VERSION,
            "run_settings": run_settings,
            "cell": typed_cell_payload,
            "seeds": asdict(typed_seeds),
            "provenance": typed_provenance.to_dict(),
        },
        domain="run-identity",
    )
    if run_hash != expected_run_hash:
        raise AnalysisError(
            "embedded run identity does not match its scientific inputs"
        )
    if Path(os.path.abspath(path)).name != run_hash:
        raise AnalysisError("run directory name does not match its embedded identity")
    environment = _mapping(cell.get("environment"), "cell environment")
    agent = _mapping(cell.get("agent"), "cell agent")
    environment_kind = environment.get("kind")
    agent_kind = agent.get("kind")
    if not isinstance(environment_kind, str) or not isinstance(agent_kind, str):
        raise AnalysisError("run cell kinds must be strings")
    condition_hash = sys.intern(analysis_condition_hash(cell))
    agent_hash = sys.intern(analysis_agent_hash(agent))
    environment_replica = cell.get("environment_replica")
    algorithm_replica = cell.get("algorithm_replica")
    if (
        isinstance(environment_replica, bool)
        or not isinstance(environment_replica, int)
        or isinstance(algorithm_replica, bool)
        or not isinstance(algorithm_replica, int)
    ):
        raise AnalysisError("run cell replica identifiers must be integers")
    raw_semantics = tuple(
        sorted(
            (sys.intern(name), sys.intern(value))
            for name, value in manifest.semantic_hashes.items()
        )
    )
    semantics = caches.semantics.setdefault(raw_semantics, raw_semantics)
    frontier_hash = manifest.semantic_hashes.get("frontier")
    if not isinstance(frontier_hash, str):
        raise AnalysisError("run manifest is missing its frontier semantic hash")
    reference = _mapping(frontier_reference.payload, "frontier reference")
    if reference.get("frontier_hash") != frontier_hash or not is_sha256(
        reference.get("artifact_hash")
    ):
        raise AnalysisError("run frontier reference is invalid")
    frontier = _frontier_summary(
        path,
        frontier_hash,
        reference["artifact_hash"],
        validation_session,
        caches.frontiers,
    )
    raw_provenance = tuple(
        sorted(
            (sys.intern(name), sys.intern(value))
            for name, value in typed_provenance.to_dict().items()
        )
    )
    shared_provenance = caches.provenances.setdefault(
        raw_provenance,
        raw_provenance,
    )
    run_settings_hash = sys.intern(
        scientific_hash(
            run_settings,
            domain="resolved-run-settings",
        )
    )
    cell_hash = sys.intern(typed_cell.cell_hash)

    observations = []
    for checkpoint in checkpoints:
        payload = _mapping(checkpoint.payload, "run checkpoint")
        result = _mapping(payload.get("result"), "checkpoint result")
        round_index = payload.get("round")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise AnalysisError("checkpoint round must be an integer")
        observations.append(
            CheckpointObservation(
                run_hash=run_hash,
                content_hash=content_hash,
                phase=config_phase,
                confirmatory_frozen=frozen,
                freeze_hash=seal,
                analysis_registration_hash=registration_hash,
                condition_hash=condition_hash,
                environment_kind=sys.intern(environment_kind),
                agent_kind=sys.intern(agent_kind),
                agent_hash=agent_hash,
                environment_replica=environment_replica,
                algorithm_replica=algorithm_replica,
                round_index=round_index,
                metrics=_numeric_metrics(result),
                semantic_hashes=semantics,
                frontier=frontier,
                run_settings_hash=run_settings_hash,
                provenance=shared_provenance,
                cell_hash=cell_hash,
            )
        )
    return tuple(sorted(observations, key=lambda item: item.round_index))


def load_run_tree(
    root: str | Path,
    *,
    expected_phase: AnalysisPhase | None = None,
    expected_freeze_hash: str | None = None,
    expected_run_settings: dict[str, Any] | None = None,
) -> tuple[CheckpointObservation, ...]:
    """Load one run with a fresh, non-injectable validation session."""

    return _load_run_tree(
        root,
        expected_phase=expected_phase,
        expected_freeze_hash=expected_freeze_hash,
        expected_run_settings=expected_run_settings,
        validation_session=ArtifactValidationSession(),
        caches=_LoadingCaches.create(),
    )


def load_run_trees(
    roots: tuple[str | Path, ...] | list[str | Path],
    *,
    expected_phase: AnalysisPhase | None = None,
    expected_freeze_hash: str | None = None,
    expected_run_settings: dict[str, Any] | None = None,
) -> AnalysisDataset:
    """Load and combine completed runs without accepting duplicate checkpoints."""

    validation_session = ArtifactValidationSession()
    caches = _LoadingCaches.create()
    observations = tuple(
        checkpoint
        for root in roots
        for checkpoint in _load_run_tree(
            root,
            expected_phase=expected_phase,
            expected_freeze_hash=expected_freeze_hash,
            expected_run_settings=expected_run_settings,
            validation_session=validation_session,
            caches=caches,
        )
    )
    return AnalysisDataset(observations)


__all__ = [
    "analysis_agent_hash",
    "analysis_condition_hash",
    "expected_groups_from_experiment",
    "load_run_tree",
    "load_run_trees",
]
