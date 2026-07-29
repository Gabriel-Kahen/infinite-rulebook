"""Strict JSON parsing for scientific configuration and evidence files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_json_strict(content: str, *, label: str) -> object:
    """Parse RFC-style JSON while rejecting duplicate keys and non-finite values."""

    if not isinstance(content, str):
        raise TypeError("JSON content must be a string")
    if not isinstance(label, str) or not label:
        raise ValueError("JSON label must be a nonempty string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"{label} repeats key {name!r}")
            result[name] = value
        return result

    try:
        return json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def load_json_strict(path: str | Path, *, label: str) -> object:
    """Read and strictly parse one JSON file."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read {label}: {source}") from error
    return parse_json_strict(content, label=label)
