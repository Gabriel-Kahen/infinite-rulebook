"""Reproducible source and dependency provenance for scientific runs."""

from __future__ import annotations

import hashlib
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


@dataclass(frozen=True, slots=True)
class ScientificProvenance:
    code_commit: str
    dirty_tree_hash: str
    dependency_lock_hash: str
    analysis_code_hash: str
    environment_digest: str
    python_implementation: str
    python_version: str
    numeric_precision: str = "python-float64"
    deterministic_mode: str = "counter-rng+semantic-keys"
    blas: str = "not-applicable-stdlib"
    cuda: str = "not-applicable"
    cudnn: str = "not-applicable"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def collect_provenance() -> ScientificProvenance:
    root = _repository_root()
    commit = _command(root, "git", "rev-parse", "HEAD").decode().strip()
    dependency_lock_hash = _file_hash(root / "uv.lock")
    python_implementation = platform.python_implementation()
    python_version = platform.python_version()
    environment_digest = scientific_hash(
        {
            "dependency_lock_hash": dependency_lock_hash,
            "python_implementation": python_implementation,
            "python_version": python_version,
        },
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
