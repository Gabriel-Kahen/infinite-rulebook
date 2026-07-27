"""Canonical behavioral deployment actions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

RuleEntry = tuple[int, int]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True, init=False)
class DeploymentAction:
    """An immutable finite-support rulebook.

    Entries are ``(rule_index, prediction)`` pairs. Rule indices start at one,
    and prediction zero denotes abstention. Abstentions are omitted from the
    canonical representation; all remaining entries are sorted by rule index.
    """

    entries: tuple[RuleEntry, ...]

    def __init__(self, entries: Iterable[RuleEntry] = ()) -> None:
        seen: set[int] = set()
        deployed: list[RuleEntry] = []
        for entry in entries:
            try:
                index, prediction = entry
            except (TypeError, ValueError) as error:
                raise TypeError("entries must be (index, prediction) pairs") from error
            if not _is_int(index) or index < 1:
                raise ValueError("rule indices must be positive integers")
            if not _is_int(prediction) or prediction < 0:
                raise ValueError("predictions must be nonnegative integers")
            if index in seen:
                raise ValueError(f"duplicate rule index: {index}")
            seen.add(index)
            if prediction:
                deployed.append((index, prediction))
        object.__setattr__(self, "entries", tuple(sorted(deployed)))

    @classmethod
    def from_mapping(cls, predictions: Mapping[int, int]) -> DeploymentAction:
        return cls(predictions.items())

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.entries)

    def prediction_for(self, index: int) -> int:
        if not _is_int(index) or index < 1:
            raise ValueError("rule indices must be positive integers")
        for candidate, prediction in self.entries:
            if candidate == index:
                return prediction
            if candidate > index:
                break
        return 0

    def validate_alphabet(self, q: int) -> None:
        if not _is_int(q) or q < 2:
            raise ValueError("q must be an integer of at least two")
        for _, prediction in self.entries:
            if prediction > q:
                raise ValueError(f"prediction {prediction} is outside alphabet 1..{q}")

    def __iter__(self) -> Iterator[RuleEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
