"""Authenticated Stage-0 prerequisite for the symbolic v1 study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from infinite_rulebook.orchestration.config import (
    ExperimentConfig,
    experiment_config_from_dict,
)
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.inventory import RawArtifactInventory
from infinite_rulebook.orchestration.reproducibility import (
    ReproducibilityReport,
    authenticate_reproducibility_roots,
)
from infinite_rulebook.studies.symbolic_construct import (
    STUDY_CONTRACT,
    verify_symbolic_smoke_design,
)

SMOKE_PREREQUISITE_SCHEMA_VERSION = 1
SMOKE_PREREQUISITE_FORMAT = "symbolic-smoke-prerequisite"
_FIELDS = {
    "evidence_format",
    "schema_version",
    "scientific",
    "operational",
    "scientific_hash",
}
_SCIENTIFIC_FIELDS = {
    "study_contract",
    "passed",
    "engineering_anomalies",
    "smoke_config",
    "reproducibility",
    "serial_inventory",
    "parallel_inventory",
}
_REPRODUCIBILITY_EVIDENCE_FIELDS = {
    "report_format",
    "schema_version",
    "scientific",
    "scientific_hash",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _validate_anomalies(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("engineering_anomalies must be an array")
    anomalies = tuple(value)
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in anomalies
    ) or anomalies != tuple(sorted(set(anomalies))):
        raise ValueError("engineering_anomalies must be sorted unique nonempty strings")
    return anomalies


def _validate_receipt_pair(
    report: ReproducibilityReport,
    serial_inventory: RawArtifactInventory,
    parallel_inventory: RawArtifactInventory,
) -> None:
    serial = serial_inventory.execution_receipt
    parallel = parallel_inventory.execution_receipt
    if (
        serial is None
        or parallel is None
        or serial.scientific_hash != report.serial_receipt_hash
        or parallel.scientific_hash != report.parallel_receipt_hash
        or serial.invocation_id != report.invocation_id
        or parallel.invocation_id != report.invocation_id
        or serial.pair != parallel.pair
        or serial.parallel_workers != report.parallel_workers
        or parallel.parallel_workers != report.parallel_workers
    ):
        raise ValueError(
            "smoke inventories do not match the reproducibility receipt pair"
        )


def _portable_reproducibility(
    report: ReproducibilityReport,
) -> dict[str, object]:
    raw = report.to_dict()
    return {
        name: raw[name]
        for name in (
            "report_format",
            "schema_version",
            "scientific",
            "scientific_hash",
        )
    }


@dataclass(frozen=True, slots=True)
class SmokePrerequisiteEvidence:
    """Self-contained evidence that Stage 0 passed its integrity checks."""

    config: ExperimentConfig
    reproducibility: ReproducibilityReport
    serial_inventory: RawArtifactInventory
    parallel_inventory: RawArtifactInventory
    engineering_anomalies: tuple[str, ...]
    scientific_hash: str

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        reproducibility: ReproducibilityReport,
        serial_inventory: RawArtifactInventory,
        parallel_inventory: RawArtifactInventory,
        *,
        engineering_anomalies: tuple[str, ...] = (),
    ) -> SmokePrerequisiteEvidence:
        verify_symbolic_smoke_design(config)
        anomalies = _validate_anomalies(engineering_anomalies)
        authenticated = authenticate_reproducibility_roots(
            config,
            serial_root=reproducibility.serial_root,
            parallel_root=reproducibility.parallel_root,
            parallel_workers=reproducibility.parallel_workers,
        )
        if authenticated != reproducibility:
            raise ValueError(
                "smoke reproducibility report differs from authenticated roots"
            )
        if (
            reproducibility.config_hash != config.config_hash
            or len(reproducibility.runs) != len(config.cells())
            or serial_inventory.config_hash != config.config_hash
            or parallel_inventory.config_hash != config.config_hash
            or serial_inventory.side != "serial"
            or parallel_inventory.side != "parallel"
        ):
            raise ValueError("smoke evidence does not cover the registered design")
        _validate_receipt_pair(
            reproducibility,
            serial_inventory,
            parallel_inventory,
        )
        serial_inventory.verify(
            reproducibility.serial_root,
            config,
            side="serial",
        )
        parallel_inventory.verify(
            reproducibility.parallel_root,
            config,
            side="parallel",
        )
        scientific = cls._scientific(
            config,
            reproducibility,
            serial_inventory,
            parallel_inventory,
            anomalies,
        )
        return cls(
            config=config,
            reproducibility=reproducibility,
            serial_inventory=serial_inventory,
            parallel_inventory=parallel_inventory,
            engineering_anomalies=anomalies,
            scientific_hash=scientific_hash(
                {
                    "evidence_format": SMOKE_PREREQUISITE_FORMAT,
                    "schema_version": SMOKE_PREREQUISITE_SCHEMA_VERSION,
                    "scientific": scientific,
                },
                domain="symbolic-smoke-prerequisite",
            ),
        )

    @staticmethod
    def _scientific(
        config: ExperimentConfig,
        reproducibility: ReproducibilityReport,
        serial_inventory: RawArtifactInventory,
        parallel_inventory: RawArtifactInventory,
        anomalies: tuple[str, ...],
    ) -> dict[str, object]:
        payload = {
            "study_contract": STUDY_CONTRACT,
            "passed": True,
            "engineering_anomalies": list(anomalies),
            "smoke_config": config.resolved_dict(),
            "reproducibility": _portable_reproducibility(reproducibility),
            "serial_inventory": serial_inventory.to_dict(),
            "parallel_inventory": parallel_inventory.to_dict(),
        }
        return json.loads(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        report = self.reproducibility.to_dict()
        return {
            "evidence_format": SMOKE_PREREQUISITE_FORMAT,
            "schema_version": SMOKE_PREREQUISITE_SCHEMA_VERSION,
            "scientific": self._scientific(
                self.config,
                self.reproducibility,
                self.serial_inventory,
                self.parallel_inventory,
                self.engineering_anomalies,
            ),
            "operational": report["operational"],
            "scientific_hash": self.scientific_hash,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        verify_roots: bool = True,
    ) -> SmokePrerequisiteEvidence:
        raw = _mapping(value, "smoke prerequisite")
        if set(raw) != _FIELDS:
            raise ValueError("smoke prerequisite fields are invalid")
        schema = raw["schema_version"]
        if (
            raw["evidence_format"] != SMOKE_PREREQUISITE_FORMAT
            or isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema != SMOKE_PREREQUISITE_SCHEMA_VERSION
        ):
            raise ValueError("smoke prerequisite type or schema is invalid")
        scientific = _mapping(raw["scientific"], "smoke scientific evidence")
        if (
            set(scientific) != _SCIENTIFIC_FIELDS
            or scientific["study_contract"] != STUDY_CONTRACT
            or scientific["passed"] is not True
        ):
            raise ValueError("smoke prerequisite scientific fields are invalid")
        anomalies = _validate_anomalies(scientific["engineering_anomalies"])
        recorded_hash = raw["scientific_hash"]
        hash_input = {
            "evidence_format": raw["evidence_format"],
            "schema_version": schema,
            "scientific": scientific,
        }
        if not is_sha256(recorded_hash) or recorded_hash != scientific_hash(
            hash_input,
            domain="symbolic-smoke-prerequisite",
        ):
            raise ValueError("smoke prerequisite scientific hash is invalid")
        config = experiment_config_from_dict(
            _mapping(scientific["smoke_config"], "smoke config")
        )
        verify_symbolic_smoke_design(config)
        portable_report = _mapping(
            scientific["reproducibility"],
            "portable smoke reproducibility",
        )
        if set(portable_report) != _REPRODUCIBILITY_EVIDENCE_FIELDS:
            raise ValueError("portable smoke reproducibility fields are invalid")
        complete_report = {
            "operational": _mapping(
                raw["operational"],
                "smoke operational evidence",
            ),
            **portable_report,
        }
        reproducibility = ReproducibilityReport.from_dict(
            complete_report,
            experiment=config if verify_roots else None,
        )
        serial = RawArtifactInventory.from_dict(scientific["serial_inventory"])
        parallel = RawArtifactInventory.from_dict(scientific["parallel_inventory"])
        _validate_receipt_pair(reproducibility, serial, parallel)
        if verify_roots:
            serial.verify(reproducibility.serial_root, config, side="serial")
            parallel.verify(reproducibility.parallel_root, config, side="parallel")
        evidence = cls(
            config=config,
            reproducibility=reproducibility,
            serial_inventory=serial,
            parallel_inventory=parallel,
            engineering_anomalies=anomalies,
            scientific_hash=recorded_hash,
        )
        if evidence.to_dict() != raw:
            raise ValueError("smoke prerequisite is not canonical")
        return evidence


__all__ = [
    "SMOKE_PREREQUISITE_FORMAT",
    "SMOKE_PREREQUISITE_SCHEMA_VERSION",
    "SmokePrerequisiteEvidence",
]
