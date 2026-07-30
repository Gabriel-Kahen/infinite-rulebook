"""Immutable, deterministic seals for confirmatory experiment configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import load_json_strict

if TYPE_CHECKING:
    from infinite_rulebook.orchestration.config import ExperimentConfig

CONFIRMATORY_FREEZE_SCHEMA_VERSION = 2
_CONFIG_HASH_DOMAIN = "confirmatory-config.v1"
_FREEZE_HASH_DOMAIN = "confirmatory-freeze-record.v2"
_SEED_IDENTITY_DOMAIN = "seed-bank-identity.v1"


class ConfirmatoryFreezeError(ValueError):
    """Raised when a confirmatory seal is absent, malformed, or mismatched."""


@dataclass(frozen=True, slots=True, order=True)
class FrozenThreshold:
    """A named nonnegative tolerance or decision margin."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ConfirmatoryFreezeError(
                "threshold names must be nonempty strings without outer whitespace"
            )
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or self.value < 0
        ):
            raise ConfirmatoryFreezeError(
                f"threshold {self.name!r} must be finite and nonnegative"
            )


def _thresholds(
    values: Mapping[str, float] | tuple[FrozenThreshold, ...],
    *,
    label: str,
) -> tuple[FrozenThreshold, ...]:
    if isinstance(values, Mapping):
        normalized = []
        for name, value in values.items():
            threshold = FrozenThreshold(name, value)
            normalized.append(FrozenThreshold(threshold.name, float(threshold.value)))
        result = tuple(
            sorted(
                normalized,
                key=lambda threshold: threshold.name,
            )
        )
    elif isinstance(values, tuple) and all(
        isinstance(value, FrozenThreshold) for value in values
    ):
        result = values
    else:
        raise TypeError(f"{label} must be a mapping or tuple of FrozenThreshold")
    names = tuple(threshold.name for threshold in result)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ConfirmatoryFreezeError(
            f"{label} must have unique names in canonical sorted order"
        )
    if not result:
        raise ConfirmatoryFreezeError(f"{label} must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class SeedBankIdentity:
    """A public identity for one master-seed/namespace pair."""

    namespace: str
    identity_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace
            or self.namespace != self.namespace.strip()
        ):
            raise ConfirmatoryFreezeError(
                "seed namespaces must be nonempty strings without outer whitespace"
            )
        if not is_sha256(self.identity_hash):
            raise ConfirmatoryFreezeError(
                "seed-bank identity_hash must be a lowercase SHA-256 digest"
            )

    @classmethod
    def bind(
        cls,
        master_seed: int | str,
        *,
        namespace: str,
    ) -> SeedBankIdentity:
        if isinstance(master_seed, bool) or not isinstance(master_seed, (int, str)):
            raise TypeError("master_seed must be an integer or string")
        return cls(
            namespace=namespace,
            identity_hash=scientific_hash(
                {"master_seed": master_seed, "namespace": namespace},
                domain=_SEED_IDENTITY_DOMAIN,
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "namespace": self.namespace,
            "identity_hash": self.identity_hash,
        }


@dataclass(frozen=True, slots=True)
class SeedBankIdentities:
    """Disjoint phase, fixed-algorithm, and evaluation seed namespaces."""

    calibration: SeedBankIdentity
    confirmatory: SeedBankIdentity
    algorithm: SeedBankIdentity
    evaluation: SeedBankIdentity

    def __post_init__(self) -> None:
        banks = (
            self.calibration,
            self.confirmatory,
            self.algorithm,
            self.evaluation,
        )
        if not all(isinstance(bank, SeedBankIdentity) for bank in banks):
            raise TypeError("all seed-bank identities must be SeedBankIdentity values")
        namespaces = tuple(bank.namespace for bank in banks)
        if len(set(namespaces)) != len(namespaces):
            raise ConfirmatoryFreezeError(
                "calibration, confirmatory, algorithm, and evaluation seed "
                "namespaces must be disjoint"
            )
        identities = tuple(bank.identity_hash for bank in banks)
        if len(set(identities)) != len(identities):
            raise ConfirmatoryFreezeError(
                "calibration, confirmatory, algorithm, and evaluation seed-bank "
                "identities must be distinct"
            )

    @classmethod
    def bind(
        cls,
        *,
        calibration_master_seed: int | str,
        confirmatory_master_seed: int | str,
        algorithm_master_seed: int | str,
        calibration_namespace: str = "calibration.v1",
        confirmatory_namespace: str = "confirmatory.v1",
        algorithm_namespace: str = "algorithm.v1",
        evaluation_namespace: str = "evaluation.v1",
    ) -> SeedBankIdentities:
        if calibration_master_seed == confirmatory_master_seed:
            raise ConfirmatoryFreezeError(
                "calibration and confirmatory master seeds must be distinct"
            )
        return cls(
            calibration=SeedBankIdentity.bind(
                calibration_master_seed,
                namespace=calibration_namespace,
            ),
            confirmatory=SeedBankIdentity.bind(
                confirmatory_master_seed,
                namespace=confirmatory_namespace,
            ),
            algorithm=SeedBankIdentity.bind(
                algorithm_master_seed,
                namespace=algorithm_namespace,
            ),
            evaluation=SeedBankIdentity.bind(
                confirmatory_master_seed,
                namespace=evaluation_namespace,
            ),
        )

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {
            "calibration": self.calibration.to_dict(),
            "confirmatory": self.confirmatory.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def _threshold_dict(
    thresholds: tuple[FrozenThreshold, ...],
) -> dict[str, float]:
    return {threshold.name: float(threshold.value) for threshold in thresholds}


def confirmatory_config_hash(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """Hash the canonical confirmatory config payload, excluding its seal."""

    if isinstance(config, Mapping):
        payload = dict(config)
        payload.pop("confirmatory_freeze", None)
    else:
        payload = config.freeze_payload()
    if payload.get("phase") != "confirmatory":
        raise ConfirmatoryFreezeError(
            "only a phase='confirmatory' payload can receive a confirmatory hash"
        )
    return scientific_hash(payload, domain=_CONFIG_HASH_DOMAIN)


@dataclass(frozen=True, slots=True)
class ConfirmatoryFreezeRecord:
    """Self-verifying record that seals one confirmatory protocol."""

    schema_version: int
    confirmatory_frozen: bool
    config_hash: str
    calibration_evidence_hash: str
    analysis_contract: str
    analysis_version: str
    analysis_code_hash: str
    dependency_lock_hash: str
    environment_digest: str
    seed_banks: SeedBankIdentities
    tolerances: tuple[FrozenThreshold, ...]
    margins: tuple[FrozenThreshold, ...]
    seal_hash: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONFIRMATORY_FREEZE_SCHEMA_VERSION
        ):
            raise ConfirmatoryFreezeError(
                f"unsupported confirmatory freeze schema_version: {self.schema_version}"
            )
        if self.confirmatory_frozen is not True:
            raise ConfirmatoryFreezeError(
                "confirmatory freeze records require confirmatory_frozen=true"
            )
        for name in (
            "config_hash",
            "calibration_evidence_hash",
            "analysis_code_hash",
            "dependency_lock_hash",
            "environment_digest",
            "seal_hash",
        ):
            if not is_sha256(getattr(self, name)):
                raise ConfirmatoryFreezeError(
                    f"{name} must be a lowercase SHA-256 digest"
                )
        value = self.analysis_contract
        if not isinstance(value, str) or not value or value != value.strip():
            raise ConfirmatoryFreezeError(
                "analysis_contract must be a nonempty string without outer whitespace"
            )
        if not is_sha256(self.analysis_version):
            raise ConfirmatoryFreezeError(
                "analysis_version must be the registered analysis SHA-256 hash"
            )
        if not isinstance(self.seed_banks, SeedBankIdentities):
            raise TypeError("seed_banks must be SeedBankIdentities")
        _thresholds(self.tolerances, label="tolerances")
        _thresholds(self.margins, label="margins")
        if self.seal_hash != self.expected_seal_hash:
            raise ConfirmatoryFreezeError("confirmatory freeze seal_hash mismatch")

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "confirmatory_frozen": self.confirmatory_frozen,
            "config_hash": self.config_hash,
            "calibration_evidence_hash": self.calibration_evidence_hash,
            "analysis_contract": self.analysis_contract,
            "analysis_version": self.analysis_version,
            "analysis_code_hash": self.analysis_code_hash,
            "dependency_lock_hash": self.dependency_lock_hash,
            "environment_digest": self.environment_digest,
            "seed_banks": self.seed_banks.to_dict(),
            "tolerances": _threshold_dict(self.tolerances),
            "margins": _threshold_dict(self.margins),
        }

    @property
    def expected_seal_hash(self) -> str:
        return scientific_hash(self._body_dict(), domain=_FREEZE_HASH_DOMAIN)

    @property
    def tolerance_values(self) -> dict[str, float]:
        return _threshold_dict(self.tolerances)

    @property
    def margin_values(self) -> dict[str, float]:
        return _threshold_dict(self.margins)

    def verify_semantic_contract(
        self,
        *,
        analysis_contract: str,
        analysis_version: str,
        tolerances: Mapping[str, float] | tuple[FrozenThreshold, ...],
        margins: Mapping[str, float] | tuple[FrozenThreshold, ...],
        analysis_code_hash: str | None = None,
        dependency_lock_hash: str | None = None,
        environment_digest: str | None = None,
    ) -> None:
        """Require exact preregistered semantics, not merely a valid self-seal."""

        expected_tolerances = _thresholds(tolerances, label="tolerances")
        expected_margins = _thresholds(margins, label="margins")
        mismatches = []
        if self.analysis_contract != analysis_contract:
            mismatches.append("analysis_contract")
        if self.analysis_version != analysis_version:
            mismatches.append("analysis_version")
        if self.tolerances != expected_tolerances:
            mismatches.append("tolerances")
        if self.margins != expected_margins:
            mismatches.append("margins")
        if (
            analysis_code_hash is not None
            and self.analysis_code_hash != analysis_code_hash
        ):
            mismatches.append("analysis_code_hash")
        if (
            dependency_lock_hash is not None
            and self.dependency_lock_hash != dependency_lock_hash
        ):
            mismatches.append("dependency_lock_hash")
        if (
            environment_digest is not None
            and self.environment_digest != environment_digest
        ):
            mismatches.append("environment_digest")
        if mismatches:
            raise ConfirmatoryFreezeError(
                "confirmatory freeze differs from the expected semantic contract: "
                f"{', '.join(mismatches)}"
            )

    @classmethod
    def create(
        cls,
        *,
        config_hash: str,
        calibration_evidence_hash: str,
        analysis_contract: str,
        analysis_version: str,
        analysis_code_hash: str,
        dependency_lock_hash: str,
        environment_digest: str,
        seed_banks: SeedBankIdentities,
        tolerances: Mapping[str, float] | tuple[FrozenThreshold, ...],
        margins: Mapping[str, float] | tuple[FrozenThreshold, ...],
    ) -> ConfirmatoryFreezeRecord:
        canonical_tolerances = _thresholds(tolerances, label="tolerances")
        canonical_margins = _thresholds(margins, label="margins")
        body = {
            "schema_version": CONFIRMATORY_FREEZE_SCHEMA_VERSION,
            "confirmatory_frozen": True,
            "config_hash": config_hash,
            "calibration_evidence_hash": calibration_evidence_hash,
            "analysis_contract": analysis_contract,
            "analysis_version": analysis_version,
            "analysis_code_hash": analysis_code_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "environment_digest": environment_digest,
            "seed_banks": seed_banks.to_dict(),
            "tolerances": _threshold_dict(canonical_tolerances),
            "margins": _threshold_dict(canonical_margins),
        }
        return cls(
            schema_version=CONFIRMATORY_FREEZE_SCHEMA_VERSION,
            confirmatory_frozen=True,
            config_hash=config_hash,
            calibration_evidence_hash=calibration_evidence_hash,
            analysis_contract=analysis_contract,
            analysis_version=analysis_version,
            analysis_code_hash=analysis_code_hash,
            dependency_lock_hash=dependency_lock_hash,
            environment_digest=environment_digest,
            seed_banks=seed_banks,
            tolerances=canonical_tolerances,
            margins=canonical_margins,
            seal_hash=scientific_hash(body, domain=_FREEZE_HASH_DOMAIN),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "seal_hash": self.seal_hash}

    def verify_config(self, config: ExperimentConfig) -> None:
        """Fail closed unless this record seals the supplied config and seed banks."""

        if config.phase != "confirmatory":
            raise ConfirmatoryFreezeError(
                "a confirmatory freeze record cannot seal a non-confirmatory config"
            )
        if confirmatory_config_hash(config) != self.config_hash:
            raise ConfirmatoryFreezeError("confirmatory config hash mismatch")
        expected_confirmatory = SeedBankIdentity.bind(
            config.master_seed,
            namespace=self.seed_banks.confirmatory.namespace,
        )
        expected_evaluation = SeedBankIdentity.bind(
            config.master_seed,
            namespace=self.seed_banks.evaluation.namespace,
        )
        if config.algorithm_master_seed is None:
            raise ConfirmatoryFreezeError(
                "confirmatory config requires an explicit algorithm_master_seed"
            )
        expected_algorithm = SeedBankIdentity.bind(
            config.algorithm_master_seed,
            namespace=self.seed_banks.algorithm.namespace,
        )
        if expected_confirmatory != self.seed_banks.confirmatory:
            raise ConfirmatoryFreezeError(
                "confirmatory seed-bank identity does not match master_seed"
            )
        if expected_evaluation != self.seed_banks.evaluation:
            raise ConfirmatoryFreezeError(
                "evaluation seed-bank identity does not match master_seed"
            )
        if expected_algorithm != self.seed_banks.algorithm:
            raise ConfirmatoryFreezeError(
                "algorithm seed-bank identity does not match algorithm_master_seed"
            )


def freeze_experiment_config(
    calibration_config: ExperimentConfig,
    *,
    confirmatory_master_seed: int | str,
    calibration_evidence_hash: str,
    analysis_contract: str,
    analysis_version: str,
    analysis_code_hash: str,
    dependency_lock_hash: str,
    environment_digest: str,
    seed_banks: SeedBankIdentities,
    tolerances: Mapping[str, float] | tuple[FrozenThreshold, ...],
    margins: Mapping[str, float] | tuple[FrozenThreshold, ...],
    name: str | None = None,
) -> ExperimentConfig:
    """Promote a calibration config to a sealed confirmatory config."""

    if calibration_config.phase != "calibration":
        raise ConfirmatoryFreezeError(
            "only a phase='calibration' config can be promoted to confirmatory"
        )
    if calibration_config.confirmatory_freeze is not None:
        raise ConfirmatoryFreezeError("calibration config must not contain a seal")
    if confirmatory_master_seed == calibration_config.master_seed:
        raise ConfirmatoryFreezeError(
            "calibration and confirmatory master seeds must be distinct"
        )
    if calibration_config.algorithm_master_seed is None:
        raise ConfirmatoryFreezeError(
            "calibration config requires an explicit algorithm_master_seed "
            "before confirmatory promotion"
        )
    expected_calibration = SeedBankIdentity.bind(
        calibration_config.master_seed,
        namespace=seed_banks.calibration.namespace,
    )
    if expected_calibration != seed_banks.calibration:
        raise ConfirmatoryFreezeError(
            "calibration seed-bank identity does not match calibration master_seed"
        )
    expected_confirmatory = SeedBankIdentity.bind(
        confirmatory_master_seed,
        namespace=seed_banks.confirmatory.namespace,
    )
    expected_evaluation = SeedBankIdentity.bind(
        confirmatory_master_seed,
        namespace=seed_banks.evaluation.namespace,
    )
    expected_algorithm = SeedBankIdentity.bind(
        calibration_config.algorithm_master_seed,
        namespace=seed_banks.algorithm.namespace,
    )
    if (
        expected_confirmatory != seed_banks.confirmatory
        or expected_evaluation != seed_banks.evaluation
        or expected_algorithm != seed_banks.algorithm
    ):
        raise ConfirmatoryFreezeError(
            "confirmatory/evaluation/algorithm seed-bank identities do not match "
            "their configured master seeds"
        )

    payload = calibration_config.freeze_payload()
    payload.update(
        {
            "phase": "confirmatory",
            "master_seed": confirmatory_master_seed,
        }
    )
    if name is not None:
        payload["name"] = name
    record = ConfirmatoryFreezeRecord.create(
        config_hash=confirmatory_config_hash(payload),
        calibration_evidence_hash=calibration_evidence_hash,
        analysis_contract=analysis_contract,
        analysis_version=analysis_version,
        analysis_code_hash=analysis_code_hash,
        dependency_lock_hash=dependency_lock_hash,
        environment_digest=environment_digest,
        seed_banks=seed_banks,
        tolerances=tolerances,
        margins=margins,
    )
    return replace(
        calibration_config,
        name=payload["name"],
        phase="confirmatory",
        master_seed=confirmatory_master_seed,
        confirmatory_freeze=record,
    )


def _strict_mapping(
    raw: object,
    *,
    label: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be an object")
    unknown = set(raw) - fields
    missing = fields - set(raw)
    if unknown:
        raise ConfirmatoryFreezeError(f"unknown {label} fields: {sorted(unknown)}")
    if missing:
        raise ConfirmatoryFreezeError(f"missing {label} fields: {sorted(missing)}")
    return dict(raw)


def _seed_identity_from_dict(raw: object) -> SeedBankIdentity:
    values = _strict_mapping(
        raw,
        label="SeedBankIdentity",
        fields={"namespace", "identity_hash"},
    )
    return SeedBankIdentity(**values)


def _seed_banks_from_dict(raw: object) -> SeedBankIdentities:
    values = _strict_mapping(
        raw,
        label="SeedBankIdentities",
        fields={"calibration", "confirmatory", "algorithm", "evaluation"},
    )
    return SeedBankIdentities(
        calibration=_seed_identity_from_dict(values["calibration"]),
        confirmatory=_seed_identity_from_dict(values["confirmatory"]),
        algorithm=_seed_identity_from_dict(values["algorithm"]),
        evaluation=_seed_identity_from_dict(values["evaluation"]),
    )


def _thresholds_from_dict(raw: object, *, label: str) -> tuple[FrozenThreshold, ...]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be an object")
    if not all(isinstance(name, str) for name in raw):
        raise TypeError(f"{label} names must be strings")
    return _thresholds(raw, label=label)


def confirmatory_freeze_from_dict(raw: object) -> ConfirmatoryFreezeRecord:
    """Parse and immediately verify a strict freeze-record object."""

    values = _strict_mapping(
        raw,
        label="ConfirmatoryFreezeRecord",
        fields={
            "schema_version",
            "confirmatory_frozen",
            "config_hash",
            "calibration_evidence_hash",
            "analysis_contract",
            "analysis_version",
            "analysis_code_hash",
            "dependency_lock_hash",
            "environment_digest",
            "seed_banks",
            "tolerances",
            "margins",
            "seal_hash",
        },
    )
    values["seed_banks"] = _seed_banks_from_dict(values["seed_banks"])
    values["tolerances"] = _thresholds_from_dict(
        values["tolerances"], label="tolerances"
    )
    values["margins"] = _thresholds_from_dict(values["margins"], label="margins")
    return ConfirmatoryFreezeRecord(**values)


def load_confirmatory_freeze(path: str | Path) -> ConfirmatoryFreezeRecord:
    """Load a deterministic freeze record; timestamps are neither read nor required."""

    return confirmatory_freeze_from_dict(
        load_json_strict(path, label="confirmatory freeze")
    )
