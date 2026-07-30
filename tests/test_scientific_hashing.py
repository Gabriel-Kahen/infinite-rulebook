from __future__ import annotations

import tracemalloc

from infinite_rulebook.orchestration.hashing import scientific_hash


def test_large_scientific_hash_has_bounded_temporary_memory() -> None:
    payload = tuple(
        {
            "index": index,
            "metrics": (("a", float(index)), ("b", float(index + 1))),
            "shared": "x" * 64,
        }
        for index in range(10_000)
    )

    tracemalloc.start()
    try:
        digest = scientific_hash(payload, domain="streaming-memory-regression")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(digest) == 64
    assert peak < 1_000_000
