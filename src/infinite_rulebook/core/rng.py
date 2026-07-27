"""Stateless, process-independent counter-based pseudorandom generation."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import TypeAlias

Seed: TypeAlias = int | str | bytes
Counter: TypeAlias = int | str | bytes


def _encode(value: Seed) -> bytes:
    if isinstance(value, bool):
        raise TypeError("boolean seeds and counters are not supported")
    if isinstance(value, int):
        sign = b"-" if value < 0 else b"+"
        magnitude = abs(value)
        width = max(1, (magnitude.bit_length() + 7) // 8)
        payload = sign + magnitude.to_bytes(width, "big")
        tag = b"i"
    elif isinstance(value, str):
        payload = unicodedata.normalize("NFC", value).encode("utf-8")
        tag = b"s"
    elif isinstance(value, bytes):
        payload = value
        tag = b"b"
    else:
        raise TypeError("seeds and counters must be int, str, or bytes")
    return tag + len(payload).to_bytes(8, "big") + payload


@dataclass(frozen=True, slots=True)
class CounterRNG:
    """A deterministic random function keyed by a seed and named stream."""

    seed: Seed
    stream: str = ""
    _key: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stream, str):
            raise TypeError("stream must be a string")
        key = hashlib.sha256(b"infinite-rulebook.seed.v1" + _encode(self.seed)).digest()
        object.__setattr__(self, "_key", key)

    def digest(self, *counter: Counter, block: int = 0) -> bytes:
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            raise ValueError("block must be a nonnegative integer")
        message = bytearray(b"infinite-rulebook.counter.v1")
        message.extend(_encode(self.stream))
        message.extend(_encode(block))
        for component in counter:
            message.extend(_encode(component))
        return hashlib.blake2b(message, key=self._key, digest_size=32).digest()

    def uint64(self, *counter: Counter, block: int = 0) -> int:
        return int.from_bytes(self.digest(*counter, block=block)[:8], "big")

    def randbelow(self, upper_bound: int, *counter: Counter) -> int:
        """Draw uniformly from ``range(upper_bound)`` without modulo bias."""

        if (
            not isinstance(upper_bound, int)
            or isinstance(upper_bound, bool)
            or upper_bound <= 0
        ):
            raise ValueError("upper_bound must be a positive integer")
        range_size = 1 << 256
        if upper_bound > range_size:
            raise ValueError("upper_bound must not exceed 2**256")
        acceptance_limit = range_size - range_size % upper_bound
        block = 0
        while True:
            candidate = int.from_bytes(self.digest(*counter, block=block), "big")
            if candidate < acceptance_limit:
                return candidate % upper_bound
            block += 1
