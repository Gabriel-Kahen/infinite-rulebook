"""Canonical, domain-separated hashes for scientific payloads."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

SCIENTIFIC_HASH_VERSION = "infinite-rulebook.scientific.v2"


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            "record",
            f"{type(value).__module__}.{type(value).__qualname__}",
            [
                [field.name, _canonical(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        ]
    if isinstance(value, Enum):
        return [
            "enum",
            f"{type(value).__module__}.{type(value).__qualname__}",
            _canonical(value.value),
        ]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("scientific mappings must have string keys")
        return [
            "map",
            [[key, _canonical(value[key])] for key in sorted(value)],
        ]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_canonical(item) for item in value]]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Path):
        return ["path", value.as_posix()]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payloads must not contain non-finite floats")
        return ["float", value.hex()]
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"unsupported scientific payload value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a scientific payload without mapping-order or float ambiguity."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def scientific_hash(value: Any, *, domain: str = "payload") -> str:
    """Return a stable SHA-256 digest for a canonical scientific payload."""

    if not isinstance(domain, str) or not domain:
        raise ValueError("hash domain must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(SCIENTIFIC_HASH_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
