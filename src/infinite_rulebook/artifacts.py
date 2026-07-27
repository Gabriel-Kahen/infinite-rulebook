"""Immutable artifacts, canonical serialization, and scientific hashes."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from typing import TypeAlias

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.rng import Seed
from infinite_rulebook.information import InformationBreakdown
from infinite_rulebook.metrics import (
    ComputeMetrics,
    EfficiencyMetric,
    FrontierRegretMetrics,
    MetricInterval,
    NoveltyMetrics,
    PopulationInformationEstimate,
    RewardMetrics,
    SupportMetrics,
)
from infinite_rulebook.validation import (
    DiagnosticSeverity,
    ValidationDiagnostic,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class FrozenFloat:
    """A float with a platform-stable JSON representation."""

    token: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str):
            raise TypeError("float token must be a string")
        if self.token in {"+inf", "-inf"}:
            return
        try:
            value = float.fromhex(self.token)
        except ValueError as error:
            raise ValueError("invalid exact float token") from error
        if not math.isfinite(value):
            raise ValueError("nonfinite floats must use +inf or -inf")
        if value == 0.0:
            value = 0.0
        if self.token != value.hex():
            raise ValueError("float token is not canonical")

    @classmethod
    def from_value(cls, value: Real) -> FrozenFloat:
        result = float(value)
        if math.isnan(result):
            raise ValueError("NaN is not a valid scientific artifact value")
        if result == math.inf:
            return cls("+inf")
        if result == -math.inf:
            return cls("-inf")
        if result == 0.0:
            result = 0.0
        return cls(result.hex())


@dataclass(frozen=True, slots=True)
class FrozenArray:
    """An immutable JSON array."""

    items: tuple[FrozenJson, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "items", tuple(freeze_json(item) for item in self.items)
        )


@dataclass(frozen=True, slots=True)
class FrozenBytes:
    """Immutable bytes with an unambiguous tagged JSON representation."""

    value: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.value, bytes | bytearray):
            raise TypeError("FrozenBytes value must be bytes")
        object.__setattr__(self, "value", bytes(self.value))


@dataclass(frozen=True, slots=True)
class FrozenMap:
    """An immutable JSON object stored in canonical key order."""

    items: tuple[tuple[str, FrozenJson], ...]

    def __post_init__(self) -> None:
        items = []
        for key, value in self.items:
            if not isinstance(key, str):
                raise TypeError("FrozenMap keys must be strings")
            items.append((_normalized_string(key), freeze_json(value)))
        items.sort(key=lambda pair: pair[0])
        keys = tuple(key for key, _ in items)
        if len(set(keys)) != len(keys):
            raise ValueError("FrozenMap items must have unique normalized keys")
        object.__setattr__(self, "items", tuple(items))


@dataclass(frozen=True, slots=True)
class FrozenRecord:
    """A typed dataclass payload, distinct from an ordinary mapping."""

    type_name: str
    fields: FrozenMap

    def __post_init__(self) -> None:
        if not isinstance(self.type_name, str) or not self.type_name:
            raise ValueError("record type_name must be a nonempty string")
        if not isinstance(self.fields, FrozenMap):
            raise TypeError("record fields must be a FrozenMap")


@dataclass(frozen=True, slots=True)
class FrozenEnum:
    """A typed enum payload, distinct from its underlying scalar."""

    type_name: str
    value: FrozenJson

    def __post_init__(self) -> None:
        if not isinstance(self.type_name, str) or not self.type_name:
            raise ValueError("enum type_name must be a nonempty string")
        object.__setattr__(self, "value", freeze_json(self.value))


FrozenJson: TypeAlias = (
    bool
    | int
    | str
    | FrozenBytes
    | FrozenFloat
    | FrozenArray
    | FrozenMap
    | FrozenRecord
    | FrozenEnum
    | None
)


def _normalized_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def freeze_json(value: object) -> FrozenJson:
    """Recursively convert supported values to an immutable canonical tree."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(
        value,
        FrozenBytes | FrozenFloat | FrozenArray | FrozenMap | FrozenRecord | FrozenEnum,
    ):
        return value
    if isinstance(value, Enum):
        return FrozenEnum(
            f"{type(value).__module__}.{type(value).__qualname__}",
            freeze_json(value.value),
        )
    if isinstance(value, str):
        return _normalized_string(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return FrozenFloat.from_value(value)
    if isinstance(value, bytes | bytearray):
        return FrozenBytes(bytes(value))
    if is_dataclass(value) and not isinstance(value, type):
        return FrozenRecord(
            f"{type(value).__module__}.{type(value).__qualname__}",
            FrozenMap(
                tuple(
                    (field.name, freeze_json(getattr(value, field.name)))
                    for field in fields(value)
                    if field.compare
                )
            ),
        )
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("artifact mapping keys must be strings")
            items.append((_normalized_string(key), freeze_json(item)))
        items.sort(key=lambda pair: pair[0])
        return FrozenMap(tuple(items))
    if isinstance(value, Sequence):
        return FrozenArray(tuple(freeze_json(item) for item in value))
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _wire_value(value: FrozenJson) -> object:
    if value is None:
        return ["n"]
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, int):
        return ["i", str(value)]
    if isinstance(value, str):
        return ["s", value]
    if isinstance(value, FrozenBytes):
        return ["y", value.value.hex()]
    if isinstance(value, FrozenFloat):
        return ["f", value.token]
    if isinstance(value, FrozenArray):
        return ["a", [_wire_value(item) for item in value.items]]
    if isinstance(value, FrozenMap):
        return ["m", [[key, _wire_value(item)] for key, item in value.items]]
    if isinstance(value, FrozenRecord):
        return [
            "r",
            value.type_name,
            [[key, _wire_value(item)] for key, item in value.fields.items],
        ]
    if isinstance(value, FrozenEnum):
        return ["e", value.type_name, _wire_value(value.value)]
    raise TypeError(f"unsupported frozen value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON with exact tagged floats."""

    frozen = freeze_json(value)
    return json.dumps(
        _wire_value(frozen),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _domain_hash(domain: str, payload: object) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError("hash domain must be a nonempty string")
    prefix = f"infinite-rulebook:{domain}:v1\0".encode()
    return hashlib.sha256(prefix + canonical_json_bytes(payload)).hexdigest()


def semantic_hash(payload: object) -> str:
    """Hash problem meaning for compatibility and cache reuse."""

    return _domain_hash("semantic", payload)


def scientific_payload_hash(payload: object) -> str:
    """Hash reported scientific content, excluding runtime metadata."""

    return _domain_hash("scientific", payload)


def _validate_sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ScientificSemantics:
    """Required semantic provenance for run and checkpoint artifacts."""

    environment: str
    reward: str
    action: str
    frontier: str

    def __post_init__(self) -> None:
        for name in ("environment", "reward", "action", "frontier"):
            object.__setattr__(
                self,
                name,
                _validate_sha256(name, getattr(self, name)),
            )


@dataclass(frozen=True, slots=True, init=False)
class ArtifactEnvelope:
    """A versioned immutable artifact with two explicit hash boundaries."""

    kind: str
    schema_version: int
    semantic_payload: FrozenJson
    scientific_payload: FrozenJson
    runtime_metadata: FrozenJson
    semantic_hash: str
    scientific_payload_hash: str

    def __init__(
        self,
        *,
        kind: str,
        schema_version: int,
        semantic_payload: object,
        scientific_payload: object,
        runtime_metadata: object | None = None,
    ) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a nonempty string")
        kind = _normalized_string(kind)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("schema_version must be an integer")
        if schema_version < 1:
            raise ValueError("schema_version must be positive")
        semantic = freeze_json(semantic_payload)
        scientific = freeze_json(scientific_payload)
        runtime = freeze_json({} if runtime_metadata is None else runtime_metadata)
        semantic_input = {
            "kind": kind,
            "schema_version": schema_version,
            "semantic_payload": semantic,
        }
        semantic_digest = semantic_hash(semantic_input)
        scientific_input = {
            "kind": kind,
            "schema_version": schema_version,
            "semantic_hash": semantic_digest,
            "scientific_payload": scientific,
        }
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "semantic_payload", semantic)
        object.__setattr__(self, "scientific_payload", scientific)
        object.__setattr__(self, "runtime_metadata", runtime)
        object.__setattr__(self, "semantic_hash", semantic_digest)
        object.__setattr__(
            self,
            "scientific_payload_hash",
            scientific_payload_hash(scientific_input),
        )

    def canonical_bytes(self) -> bytes:
        """Serialize the full envelope, including non-scientific runtime data."""

        return canonical_json_bytes(self)

    def validate_compatible(self, other: ArtifactEnvelope) -> ValidationReport:
        """Diagnose whether two artifacts describe the same scientific problem."""

        if not isinstance(other, ArtifactEnvelope):
            raise TypeError("other must be an ArtifactEnvelope")
        diagnostics = []
        if self.kind != other.kind:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "ARTIFACT_KIND_MISMATCH",
                    "kind",
                    f"{self.kind!r} does not match {other.kind!r}",
                )
            )
        if self.schema_version != other.schema_version:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "SCHEMA_VERSION_MISMATCH",
                    "schema_version",
                    f"{self.schema_version} does not match {other.schema_version}",
                )
            )
        if self.semantic_hash != other.semantic_hash:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "INCOMPATIBLE_SEMANTIC_HASH",
                    "semantic_hash",
                    "artifact semantic hashes differ",
                )
            )
        return ValidationReport(tuple(diagnostics))


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Immutable per-run checkpoint; information is realized Bayesian surprise."""

    schema_version: int
    semantic_hashes: ScientificSemantics
    round_index: int
    reward_samples: tuple[float, ...]
    realized_information: InformationBreakdown
    deployment_witness: DeploymentAction
    deployment_semantic_hash: str
    deployment_seed: Seed
    novelty: NoveltyMetrics
    support: SupportMetrics
    compute: ComputeMetrics
    target_size: int

    def __post_init__(self) -> None:
        for name in ("schema_version", "round_index", "target_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name == "schema_version" else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not isinstance(self.semantic_hashes, ScientificSemantics):
            raise TypeError("semantic_hashes must be ScientificSemantics")
        samples = tuple(
            float(_artifact_finite("reward sample", value))
            for value in self.reward_samples
        )
        if not samples:
            raise ValueError("reward_samples must not be empty")
        if not isinstance(self.realized_information, InformationBreakdown):
            raise TypeError("realized_information must be an InformationBreakdown")
        if not self.realized_information.reconciles():
            raise ValueError("realized information buckets do not reconcile")
        if not isinstance(self.deployment_witness, DeploymentAction):
            raise TypeError("deployment_witness must be a DeploymentAction")
        if (
            not isinstance(self.deployment_semantic_hash, str)
            or not self.deployment_semantic_hash
        ):
            raise ValueError("deployment_semantic_hash must be a nonempty string")
        if self.deployment_semantic_hash != semantic_hash(self.deployment_witness):
            raise ValueError("deployment_semantic_hash does not match witness")
        if isinstance(self.deployment_seed, bool) or not isinstance(
            self.deployment_seed, str | int | bytes
        ):
            raise TypeError("deployment_seed must be a string, integer, or bytes")
        if not isinstance(self.novelty, NoveltyMetrics):
            raise TypeError("novelty must be NoveltyMetrics")
        if not isinstance(self.support, SupportMetrics):
            raise TypeError("support must be SupportMetrics")
        if not isinstance(self.compute, ComputeMetrics):
            raise TypeError("compute must be ComputeMetrics")
        if len(self.deployment_witness) != self.support.deployment_support:
            raise ValueError(
                "deployment witness support does not match support metrics"
            )
        if (
            self.support.deployment_support + self.support.abstentions
            != self.target_size
        ):
            raise ValueError(
                "target_size must equal deployment support plus abstentions"
            )
        object.__setattr__(self, "reward_samples", samples)

    def envelope(
        self,
        *,
        runtime_metadata: object | None = None,
    ) -> ArtifactEnvelope:
        """Wrap this checkpoint with an explicit semantic/runtime boundary."""

        return ArtifactEnvelope(
            kind="run_checkpoint",
            schema_version=self.schema_version,
            semantic_payload=self.semantic_hashes,
            scientific_payload=self,
            runtime_metadata=runtime_metadata,
        )


def _artifact_finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CheckpointEstimate:
    """Immutable pooled checkpoint estimate with distinct population estimands."""

    schema_version: int
    semantic_hashes: ScientificSemantics
    reward: RewardMetrics
    bit_equivalent: MetricInterval
    population_information: PopulationInformationEstimate
    efficiency: EfficiencyMetric
    novelty: NoveltyMetrics
    support: SupportMetrics
    frontier_regret: FrontierRegretMetrics
    uncertainty: tuple[tuple[str, MetricInterval], ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be an integer")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not isinstance(self.semantic_hashes, ScientificSemantics):
            raise TypeError("semantic_hashes must be ScientificSemantics")
        expected_types = (
            ("reward", self.reward, RewardMetrics),
            ("bit_equivalent", self.bit_equivalent, MetricInterval),
            (
                "population_information",
                self.population_information,
                PopulationInformationEstimate,
            ),
            ("efficiency", self.efficiency, EfficiencyMetric),
            ("novelty", self.novelty, NoveltyMetrics),
            ("support", self.support, SupportMetrics),
            ("frontier_regret", self.frontier_regret, FrontierRegretMetrics),
        )
        for name, value, expected in expected_types:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.bit_equivalent.units != "nats" or self.bit_equivalent.lower < 0.0:
            raise ValueError("bit_equivalent must be a nonnegative interval in nats")
        uncertainty_items = []
        for name, interval in self.uncertainty:
            if not isinstance(name, str) or not name:
                raise ValueError("uncertainty names must be nonempty strings")
            if not isinstance(interval, MetricInterval):
                raise TypeError("uncertainty values must be MetricInterval records")
            uncertainty_items.append((_normalized_string(name), interval))
        uncertainty = tuple(sorted(uncertainty_items, key=lambda item: item[0]))
        if len({name for name, _ in uncertainty}) != len(uncertainty):
            raise ValueError("uncertainty component names must be unique")
        object.__setattr__(self, "uncertainty", uncertainty)

    def validate(self) -> ValidationReport:
        diagnostics = [
            *self.population_information.validate().diagnostics,
            *self.efficiency.validation.diagnostics,
            *self.frontier_regret.validate().diagnostics,
        ]
        denominator = self.population_information.total_nats
        interval = self.efficiency.interval
        if interval is not None and denominator > 0.0 and math.isfinite(denominator):
            expected = MetricInterval(
                self.bit_equivalent.lower / denominator,
                self.bit_equivalent.upper / denominator,
                "ratio",
            )
            if interval != expected:
                diagnostics.append(
                    ValidationDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "EFFICIENCY_VALUE_MISMATCH",
                        "efficiency.interval",
                        "efficiency is not the pooled bit-equivalent/information ratio",
                    )
                )
        elif interval is not None:
            diagnostics.append(
                ValidationDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "EFFICIENCY_VALUE_MISMATCH",
                    "efficiency.interval",
                    "efficiency must be undefined without finite positive information",
                )
            )
        return ValidationReport(tuple(diagnostics))

    def envelope(
        self,
        *,
        runtime_metadata: object | None = None,
    ) -> ArtifactEnvelope:
        """Wrap this estimate with an explicit semantic/runtime boundary."""

        return ArtifactEnvelope(
            kind="checkpoint_estimate",
            schema_version=self.schema_version,
            semantic_payload=self.semantic_hashes,
            scientific_payload=self,
            runtime_metadata=runtime_metadata,
        )
