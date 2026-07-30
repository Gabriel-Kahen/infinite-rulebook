"""Reproducible source and dependency provenance for scientific runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from infinite_rulebook.orchestration.hashing import scientific_hash


def _command(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            arguments,
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return b""


def _repository_root() -> Path:
    package = Path(__file__).resolve()
    output = _command(package.parent, "git", "rev-parse", "--show-toplevel")
    return Path(output.decode().strip()) if output else package.parents[3]


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return scientific_hash("missing", domain="file-content")
    return scientific_hash(path.read_bytes(), domain="file-content")


def _analysis_hash(root: Path) -> str:
    paths = sorted((root / "src").rglob("*.py"))
    scripts = root / "scripts"
    if scripts.is_dir():
        paths.extend(sorted(scripts.rglob("*.py")))
    return scientific_hash(
        {path.relative_to(root).as_posix(): path.read_bytes() for path in paths},
        domain="analysis-code",
    )


def _dirty_tree_hash(root: Path) -> str:
    digest = hashlib.sha256(b"infinite-rulebook.dirty-tree.v1\0")
    digest.update(_command(root, "git", "diff", "--binary", "HEAD"))
    untracked = _command(
        root,
        "git",
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "configs",
        "tests",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in sorted(filter(None, untracked.decode().splitlines())):
        path = root / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _execution_environment_payload(
    dependency_lock_hash: str,
) -> dict[str, object]:
    libc_name, libc_version = platform.libc_ver()
    python_build_number, python_build_date = platform.python_build()
    return {
        "dependency_lock_hash": dependency_lock_hash,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build_number": python_build_number,
        "python_build_date": python_build_date,
        "python_compiler": platform.python_compiler(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc_name": libc_name,
        "libc_version": libc_version,
        "byteorder": sys.byteorder,
        "float_info": {
            "radix": sys.float_info.radix,
            "mant_dig": sys.float_info.mant_dig,
            "max_exp": sys.float_info.max_exp,
            "min_exp": sys.float_info.min_exp,
            "rounds": sys.float_info.rounds,
        },
    }


@dataclass(frozen=True, slots=True)
class ScientificProvenance:
    code_commit: str
    dirty_tree_hash: str
    dependency_lock_hash: str
    analysis_code_hash: str
    environment_digest: str
    python_implementation: str
    python_version: str
    environment_fingerprint: str = ""
    numeric_precision: str = "python-float64"
    deterministic_mode: str = "counter-rng+semantic-keys"
    blas: str = "not-applicable-stdlib"
    cuda: str = "not-applicable"
    cudnn: str = "not-applicable"

    def __post_init__(self) -> None:
        if not self.environment_fingerprint:
            return
        try:
            payload = json.loads(self.environment_fingerprint)
        except json.JSONDecodeError as error:
            raise ValueError("environment_fingerprint is not valid JSON") from error
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if (
            not isinstance(payload, dict)
            or canonical != self.environment_fingerprint
            or payload.get("dependency_lock_hash") != self.dependency_lock_hash
            or payload.get("python_implementation") != self.python_implementation
            or payload.get("python_version") != self.python_version
            or scientific_hash(payload, domain="execution-environment")
            != self.environment_digest
        ):
            raise ValueError(
                "environment_fingerprint does not reproduce environment_digest"
            )

    def to_dict(self) -> dict[str, str]:
        result = asdict(self)
        if not self.environment_fingerprint:
            del result["environment_fingerprint"]
        return result


def collect_provenance() -> ScientificProvenance:
    root = _repository_root()
    commit = _command(root, "git", "rev-parse", "HEAD").decode().strip()
    dependency_lock_hash = _file_hash(root / "uv.lock")
    python_implementation = platform.python_implementation()
    python_version = platform.python_version()
    environment_payload = _execution_environment_payload(dependency_lock_hash)
    environment_fingerprint = json.dumps(
        environment_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    environment_digest = scientific_hash(
        environment_payload,
        domain="execution-environment",
    )
    return ScientificProvenance(
        code_commit=commit or "unavailable",
        dirty_tree_hash=_dirty_tree_hash(root),
        dependency_lock_hash=dependency_lock_hash,
        analysis_code_hash=_analysis_hash(root),
        environment_digest=environment_digest,
        python_implementation=python_implementation,
        python_version=python_version,
        environment_fingerprint=environment_fingerprint,
    )


def collect_runtime_metadata(*, wall_time_seconds: float) -> dict[str, object]:
    """Return operational metadata excluded from scientific hashes."""

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "wall_time_seconds": wall_time_seconds,
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "python_executable": sys.executable,
    }
