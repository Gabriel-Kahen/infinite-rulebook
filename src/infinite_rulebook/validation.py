"""Stable, machine-readable validation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DiagnosticSeverity(IntEnum):
    """Diagnostic severity, ordered from informational to fatal."""

    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True, slots=True, order=True)
class ValidationDiagnostic:
    """One deterministic artifact or metric validation finding."""

    severity: DiagnosticSeverity
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.severity, DiagnosticSeverity):
            raise TypeError("severity must be a DiagnosticSeverity")
        for name, value in (
            ("code", self.code),
            ("path", self.path),
            ("message", self.message),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """An immutable, deterministically ordered set of diagnostics."""

    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        diagnostics = tuple(sorted(self.diagnostics))
        if any(not isinstance(item, ValidationDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain ValidationDiagnostic records")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def valid(self) -> bool:
        return not any(
            item.severity >= DiagnosticSeverity.ERROR for item in self.diagnostics
        )
