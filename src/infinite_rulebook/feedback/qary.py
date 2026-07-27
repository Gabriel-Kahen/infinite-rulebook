"""The deterministic-noise implementation of the P1 q-ary query channel."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

Seed = int | str | bytes


def _seed_record(seed: Seed) -> tuple[str, str]:
    if isinstance(seed, bool):
        raise TypeError("environment_seed must not be a boolean")
    if isinstance(seed, bytes):
        return ("bytes", seed.hex())
    if isinstance(seed, int):
        return ("int", str(seed))
    if isinstance(seed, str):
        return ("str", seed)
    raise TypeError("environment_seed must be an int, str, or bytes")


@dataclass(frozen=True, slots=True)
class SemanticObservationKey:
    """Identity of one observation-noise draw.

    The key deliberately contains no sequential RNG position. Consequently,
    evaluating the same semantic query produces the same observation regardless
    of execution order, batching, or process count.
    """

    environment_seed: Seed
    round_index: int
    rule_index: int
    query_ordinal: int = 0
    channel: str = "p1"

    def __post_init__(self) -> None:
        for name in ("round_index", "rule_index", "query_ordinal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.channel, str):
            raise TypeError("channel must be a string")
        if not self.channel:
            raise ValueError("channel must not be empty")
        _seed_record(self.environment_seed)

    def canonical_bytes(self) -> bytes:
        record = [
            "infinite-rulebook-observation-v1",
            _seed_record(self.environment_seed),
            self.round_index,
            self.rule_index,
            self.query_ordinal,
            self.channel,
        ]
        return json.dumps(
            record,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")


@dataclass(frozen=True, slots=True)
class QarySymmetricChannel:
    """An informative q-ary symmetric observation channel.

    Labels use the benchmark convention ``1, ..., q``. Noise is a pure
    function of a :class:`SemanticObservationKey`; the class owns no RNG state.
    """

    q: int
    epsilon: float

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, int):
            raise TypeError("q must be an integer")
        if self.q < 2:
            raise ValueError("q must be at least 2")
        if not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite")
        if not 0.0 <= self.epsilon < (self.q - 1) / self.q:
            raise ValueError("epsilon must satisfy 0 <= epsilon < (q - 1) / q")

    @property
    def accuracy(self) -> float:
        return 1.0 - self.epsilon

    @property
    def capacity(self) -> float:
        """Uniform-input channel capacity in nats."""

        if self.epsilon == 0.0:
            return math.log(self.q)
        binary_entropy = -(
            self.epsilon * math.log(self.epsilon)
            + (1.0 - self.epsilon) * math.log1p(-self.epsilon)
        )
        return math.log(self.q) - binary_entropy - self.epsilon * math.log(self.q - 1)

    def probabilities(self, true_label: int) -> tuple[float, ...]:
        """Return ``P(O=j | Y=true_label)`` in ascending label order."""

        self._validate_label(true_label)
        error_probability = self.epsilon / (self.q - 1)
        return tuple(
            self.accuracy if label == true_label else error_probability
            for label in range(1, self.q + 1)
        )

    def likelihood(self, observation: int, true_label: int) -> float:
        self._validate_label(observation)
        self._validate_label(true_label)
        if observation == true_label:
            return self.accuracy
        return self.epsilon / (self.q - 1)

    def observe(
        self,
        true_label: int,
        key: SemanticObservationKey,
    ) -> int:
        """Return the deterministic observation associated with ``key``."""

        self._validate_label(true_label)
        if not isinstance(key, SemanticObservationKey):
            raise TypeError("key must be a SemanticObservationKey")
        digest = hashlib.sha256(key.canonical_bytes()).digest()
        draw = int.from_bytes(digest, "big") / (1 << (8 * len(digest)))
        if draw < self.accuracy:
            return true_label

        # Conditional on an error, divide the remaining interval equally among
        # incorrect labels. The rank-to-label conversion skips true_label.
        error_draw = (draw - self.accuracy) / self.epsilon
        incorrect_rank = min(int(error_draw * (self.q - 1)), self.q - 2)
        candidate = incorrect_rank + 1
        return candidate if candidate < true_label else candidate + 1

    sample = observe

    def _validate_label(self, label: int) -> None:
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError("labels must be integers")
        if not 1 <= label <= self.q:
            raise ValueError(f"label must be in 1..{self.q}")


def deterministic_qary_observation(
    true_label: int,
    *,
    q: int,
    epsilon: float,
    environment_seed: Seed,
    round_index: int,
    rule_index: int,
    query_ordinal: int = 0,
    channel: str = "p1",
) -> int:
    """Convenience wrapper for a single semantically keyed observation."""

    key = SemanticObservationKey(
        environment_seed=environment_seed,
        round_index=round_index,
        rule_index=rule_index,
        query_ordinal=query_ordinal,
        channel=channel,
    )
    return QarySymmetricChannel(q=q, epsilon=epsilon).observe(true_label, key)
