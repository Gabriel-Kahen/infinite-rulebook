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

    return b"".join(_canonical_json_chunks(value))


def _json_scalar(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical_json_chunks(value: Any):
    """Yield the exact v2 canonical JSON encoding without building its object graph."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield b'["record",'
        yield _json_scalar(f"{type(value).__module__}.{type(value).__qualname__}")
        yield b",["
        for index, field in enumerate(dataclasses.fields(value)):
            if index:
                yield b","
            yield b"["
            yield _json_scalar(field.name)
            yield b","
            yield from _canonical_json_chunks(getattr(value, field.name))
            yield b"]"
        yield b"]]"
        return
    if isinstance(value, Enum):
        yield b'["enum",'
        yield _json_scalar(f"{type(value).__module__}.{type(value).__qualname__}")
        yield b","
        yield from _canonical_json_chunks(value.value)
        yield b"]"
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("scientific mappings must have string keys")
        yield b'["map",['
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield b"["
            yield _json_scalar(key)
            yield b","
            yield from _canonical_json_chunks(value[key])
            yield b"]"
        yield b"]]"
        return
    if isinstance(value, (tuple, list)):
        yield b'["sequence",['
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _canonical_json_chunks(item)
        yield b"]]"
        return
    if isinstance(value, bytes):
        yield b'["bytes",'
        yield _json_scalar(value.hex())
        yield b"]"
        return
    if isinstance(value, Path):
        yield b'["path",'
        yield _json_scalar(value.as_posix())
        yield b"]"
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payloads must not contain non-finite floats")
        yield b'["float",'
        yield _json_scalar(value.hex())
        yield b"]"
        return
    if value is None:
        yield b'["none"]'
        return
    if isinstance(value, bool):
        yield b'["bool",'
        yield b"true" if value else b"false"
        yield b"]"
        return
    if isinstance(value, int):
        yield b'["int",'
        yield _json_scalar(str(value))
        yield b"]"
        return
    if isinstance(value, str):
        yield b'["str",'
        yield _json_scalar(value)
        yield b"]"
        return
    raise TypeError(f"unsupported scientific payload value: {type(value).__name__}")


def scientific_hash(value: Any, *, domain: str = "payload") -> str:
    """Return a stable SHA-256 digest for a canonical scientific payload."""

    if not isinstance(domain, str) or not domain:
        raise ValueError("hash domain must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(SCIENTIFIC_HASH_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for chunk in _canonical_json_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
