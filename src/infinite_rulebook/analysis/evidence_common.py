"""Shared fail-closed helpers for authenticated v2 evidence records."""

from __future__ import annotations

import dataclasses
import json
import math
import types
from enum import Enum
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from infinite_rulebook.analysis.models import (
    AnalysisDataset,
    AnalysisError,
    ExpectedGroup,
    GroupSelector,
)
from infinite_rulebook.orchestration.hashing import is_sha256
from infinite_rulebook.orchestration.jsonio import parse_json_strict

_T = TypeVar("_T")


def mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnalysisError(f"{name} must be a JSON object")
    return value


def keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise AnalysisError(f"{name} must contain exactly {sorted(expected)}")


def identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    return value


def finite(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def count(name: str, value: object, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def sha256(name: str, value: object) -> str:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def checkpoints(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError(
            "checkpoints must be a nonempty sorted tuple of unique nonnegative integers"
        )
    return value


def exact_selector(name: str, value: GroupSelector) -> None:
    if not isinstance(value, GroupSelector):
        raise TypeError(f"{name} must be a GroupSelector")
    if value.condition_hash is None or value.agent_hash is None:
        raise ValueError(f"{name} must bind exact condition_hash and agent_hash values")


def expected_group_map(
    value: tuple[ExpectedGroup, ...],
) -> dict[GroupSelector, ExpectedGroup]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, ExpectedGroup) for item in value)
    ):
        raise TypeError("expected_groups must be a nonempty tuple of ExpectedGroup")
    result = {
        GroupSelector(
            environment_kind=item.environment_kind,
            agent_kind=item.agent_kind,
            condition_hash=item.condition_hash,
            agent_hash=item.agent_hash,
        ): item
        for item in value
    }
    if len(result) != len(value):
        raise ValueError("expected_groups contains duplicate group identities")
    return result


def validate_expected_group_inventory(
    dataset: AnalysisDataset,
    expected_groups: tuple[ExpectedGroup, ...],
) -> None:
    """Require the exact registered group/replica/checkpoint Cartesian product."""

    if not isinstance(dataset, AnalysisDataset):
        raise TypeError("dataset must be an AnalysisDataset")
    expected = {
        (
            selector.environment_kind,
            selector.agent_kind,
            selector.condition_hash,
            selector.agent_hash,
        ): group
        for selector, group in expected_group_map(expected_groups).items()
    }
    checkpoint_positions = {
        identity: {
            checkpoint: index for index, checkpoint in enumerate(group.checkpoints)
        }
        for identity, group in expected.items()
    }
    observed = {
        identity: bytearray(
            group.environment_replicas
            * group.algorithm_replicas
            * len(group.checkpoints)
        )
        for identity, group in expected.items()
    }
    counts = dict.fromkeys(expected, 0)
    for item in dataset.observations:
        identity = (
            item.environment_kind,
            item.agent_kind,
            item.condition_hash,
            item.agent_hash,
        )
        group = expected.get(identity)
        if group is None:
            raise AnalysisError(
                "evidence dataset groups differ from the registered inventory"
            )
        checkpoint = checkpoint_positions[identity].get(item.round_index)
        if (
            checkpoint is None
            or item.environment_replica >= group.environment_replicas
            or item.algorithm_replica >= group.algorithm_replicas
        ):
            raise AnalysisError(
                "evidence dataset does not match the registered replica/checkpoint grid"
            )
        index = (
            item.environment_replica * group.algorithm_replicas + item.algorithm_replica
        ) * len(group.checkpoints) + checkpoint
        if observed[identity][index]:
            raise AnalysisError(
                "evidence dataset does not match the registered replica/checkpoint grid"
            )
        observed[identity][index] = 1
        counts[identity] += 1
    for identity, group in expected.items():
        required_count = (
            group.environment_replicas
            * group.algorithm_replicas
            * len(group.checkpoints)
        )
        if counts[identity] != required_count:
            raise AnalysisError(
                "evidence dataset does not match the registered replica/checkpoint grid"
            )


def payload(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: payload(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [payload(item) for item in value]
    return value


def record_payload(value: Any) -> Any:
    """Encode typed evidence records with explicit, allow-listed type tags."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "record_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: record_payload(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [record_payload(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def decode_record(
    value: object,
    expected: Any,
    *,
    allowed_types: tuple[type[Any], ...],
    name: str,
) -> Any:
    """Decode one strict tagged record and reconstruct all derived fields."""

    registry = {
        f"{item.__module__}.{item.__qualname__}": item for item in allowed_types
    }

    def decode(raw: object, annotation: Any, path: str) -> Any:
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is types.UnionType:
            for choice in arguments:
                try:
                    return decode(raw, choice, path)
                except AnalysisError:
                    continue
            raise AnalysisError(f"{path} does not match its registered union")
        if origin is tuple:
            if not isinstance(raw, list):
                raise AnalysisError(f"{path} must be a JSON array")
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return tuple(
                    decode(item, arguments[0], f"{path}[{index}]")
                    for index, item in enumerate(raw)
                )
            if len(raw) != len(arguments):
                raise AnalysisError(f"{path} has the wrong tuple length")
            return tuple(
                decode(item, item_type, f"{path}[{index}]")
                for index, (item, item_type) in enumerate(
                    zip(raw, arguments, strict=True)
                )
            )
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            try:
                return annotation(raw)
            except (TypeError, ValueError) as error:
                raise AnalysisError(f"{path} has an invalid enum value") from error
        if isinstance(raw, dict):
            record = mapping(raw, path)
            keys(record, {"record_type", "fields"}, path)
            tag = record["record_type"]
            candidate = registry.get(tag) if isinstance(tag, str) else None
            if candidate is None or (
                dataclasses.is_dataclass(annotation) and candidate is not annotation
            ):
                raise AnalysisError(f"{path} has an unregistered record type")
            fields = mapping(record["fields"], f"{path}.fields")
            definitions = dataclasses.fields(candidate)
            keys(
                fields,
                {field.name for field in definitions},
                f"{path}.fields",
            )
            hints = get_type_hints(candidate)
            kwargs = {
                field.name: decode(
                    fields[field.name],
                    hints[field.name],
                    f"{path}.{field.name}",
                )
                for field in definitions
                if field.init
            }
            try:
                result = candidate(**kwargs)
            except (AttributeError, TypeError, ValueError) as error:
                raise AnalysisError(f"{path} failed record validation") from error
            if record_payload(result) != record:
                raise AnalysisError(f"{path} has noncanonical derived fields")
            return result
        if annotation is type(None) and raw is None:
            return None
        if annotation is str and isinstance(raw, str):
            return raw
        if annotation is bool and isinstance(raw, bool):
            return raw
        if annotation is int and type(raw) is int:
            return raw
        if annotation is float and type(raw) in (int, float):
            return raw
        raise AnalysisError(f"{path} has an invalid primitive value")

    return decode(value, expected, name)


def canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def strict_json(content: str, *, label: str) -> object:
    try:
        return parse_json_strict(content, label=label)
    except (TypeError, ValueError) as error:
        raise AnalysisError(str(error)) from error


def parse_artifact(
    content: str,
    *,
    label: str,
    artifact_type: str,
    schema_version: int,
    record_type: type[_T],
    allowed_types: tuple[type[Any], ...],
    fixed: dict[str, object],
    derived: set[str],
) -> _T:
    """Strictly reconstruct and authenticate one tagged evidence artifact."""

    raw = mapping(strict_json(content, label=label), label)
    record_fields = {item.name for item in dataclasses.fields(record_type)}
    keys(
        raw,
        {
            "artifact_type",
            "schema_version",
            "scientific_hash",
            *record_fields,
            *fixed,
            *derived,
        },
        label,
    )
    if (
        raw["artifact_type"] != artifact_type
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != schema_version
        or any(raw[name] != expected for name, expected in fixed.items())
    ):
        raise AnalysisError(f"{label} type, schema, or fixed boundary is invalid")
    result = decode_record(
        {
            "record_type": f"{record_type.__module__}.{record_type.__qualname__}",
            "fields": {name: raw[name] for name in record_fields},
        },
        record_type,
        allowed_types=allowed_types,
        name=label,
    )
    if canonical_json(result.to_dict()) != canonical_json(raw):
        raise AnalysisError(f"{label} is noncanonical or tampered")
    return result


__all__ = [
    "canonical_json",
    "checkpoints",
    "count",
    "decode_record",
    "exact_selector",
    "expected_group_map",
    "finite",
    "identifier",
    "keys",
    "mapping",
    "parse_artifact",
    "payload",
    "record_payload",
    "sha256",
    "strict_json",
    "validate_expected_group_inventory",
]
