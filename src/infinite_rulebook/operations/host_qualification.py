"""Read-only host checks and synthetic capacity evidence for symbolic v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from importlib import metadata as importlib_metadata
from itertools import pairwise
from pathlib import Path
from typing import Any

from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.jsonio import parse_json_strict
from infinite_rulebook.studies.symbolic_registry import registered_symbolic_study

GIB = 1024**3
MIB = 1024**2
APPROVED_V2_EXECUTION_COMMITS = ("c9b6297b63b572d9e6d106de4add1dae436c00d3",)
MINIMUM_PHYSICAL_MEMORY_BYTES = 64 * GIB
REFERENCE_PAIR_RAW_BYTES = 171_002_155_008
REFERENCE_PAIR_RAW_FILES = 17_104_896
V2_CALIBRATION_CONFIG_HASH = (
    "c0f4cf5bf09e6b516379c0fec26ccd4a8780d8b6d52226093ef5a96cc0437508"
)
V2_PROBE_CONFIG_HASH = (
    "9120115e05a126127ce639169b15a27f9e39c09c7e277301f2e4dedd189de9aa"
)
V2_E192_DATASET_HASH = (
    "f8eb1d5d130f283959c469049ea5ec1e53dbfc053e656e0c4817ae8c1534d1ee"
)
V2_E768_DATASET_HASH = (
    "b4de258dc8e72d78e5324c072dbfc7fab19f4534d40aaf6175c3e596873cf58a"
)
V2_PROBE_DATASET_HASH = (
    "881f1c3ac959baedb22ab961ad35aaa2654ec092407da725de79fe858265a043"
)
V2_PROJECTED_RUNS = 294_912
V2_PROBE_RAW_RUN_FILES = 1_392
MAXIMUM_RECORD_AGE = timedelta(hours=24)
MAXIMUM_CLOCK_SKEW = timedelta(minutes=5)
DEFAULT_CAPACITY_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_PROBE_TIMEOUT_SECONDS = 24 * 60 * 60
MAXIMUM_SAMPLE_INTERVAL_SECONDS = 1.0

_STATIC_RECORD_TYPE = "symbolic-v2-host-static-qualification"
_CAPACITY_RECORD_TYPE = "symbolic-v2-capacity-qualification"
_PROBE_EXECUTION_RECORD_TYPE = "symbolic-v2-probe-execution-qualification"
_PROBE_BENCHMARK_RECORD_TYPE = "symbolic-v2-probe-benchmark-qualification"
_ASSESSMENT_RECORD_TYPE = "symbolic-v2-host-qualification-assessment"

_PRECREATED_PROBE_ROOT_BOOTSTRAP = """
import os
import runpy
import sys

descriptor = int(sys.argv.pop(1))
repository = sys.argv.pop(1)
sys.path.insert(0, repository)
os.fchdir(descriptor)
target = os.fspath(sys.argv[3])
original_lexists = os.path.lexists
calls = 0

def allow_precreated_root_once(path):
    global calls
    if calls == 0 and os.fspath(path) == target:
        calls = 1
        os.path.lexists = original_lexists
        return False
    return original_lexists(path)

os.path.lexists = allow_precreated_root_once
try:
    runpy.run_module("scripts.run_ingestion_probe", run_name="__main__")
except SystemExit:
    if calls != 1:
        raise RuntimeError("probe runner did not validate the anchored root")
    raise
finally:
    os.path.lexists = original_lexists
""".strip()

_ANCHORED_PROBE_BENCHMARK_BOOTSTRAP = """
import os
import runpy
import sys

descriptor = int(sys.argv.pop(1))
repository = sys.argv.pop(1)
sys.path.insert(0, repository)
os.fchdir(descriptor)
runpy.run_module("scripts.benchmark_artifact_ingestion", run_name="__main__")
""".strip()

_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "ceph",
        "cifs",
        "fuse.sshfs",
        "gcsfuse",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
        "sshfs",
    }
)
_NON_STORAGE_FILESYSTEMS = frozenset(
    {
        "cgroup",
        "cgroup2",
        "devpts",
        "devtmpfs",
        "proc",
        "sysfs",
        "tmpfs",
    }
)
_CAPACITY_SPECS = {
    "e192": {
        "environment_replicas": 192,
        "observation_count": 958_464,
        "dataset_hash": V2_E192_DATASET_HASH,
    },
    "e768": {
        "environment_replicas": 768,
        "observation_count": 3_833_856,
        "dataset_hash": V2_E768_DATASET_HASH,
    },
}
_ALLOWED_PTH_EXECUTABLE_LINE_HASHES = frozenset(
    {
        "2cbb286eeaf39b2cd7d68cb09c1bc3cfb0cc8d27da949004d0c22698f127f270",
        "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _record_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    return scientific_hash(unsigned, domain="operations.host-qualification.v1")


def _signed(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["record_hash"] = _record_hash(result)
    return result


def _expect_keys(
    value: object,
    *,
    label: str,
    keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    observed = frozenset(value)
    if observed != keys:
        missing = sorted(keys - observed)
        unexpected = sorted(observed - keys)
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


def _expect_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _expect_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _expect_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (
        result < minimum or (strict_minimum and result == minimum)
    ):
        comparator = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {comparator} {minimum}")
    return result


def _expect_string(value: object, *, label: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{label} must be a {qualifier}string")
    return value


def _expect_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parse_timestamp(value: object, *, label: str) -> datetime:
    text = _expect_string(value, label=label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC)


def verify_record(payload: object, *, record_type: str | None = None) -> dict[str, Any]:
    """Validate one operational record without treating it as study evidence."""

    if not isinstance(payload, dict):
        raise ValueError("qualification record must be a JSON object")
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
    ):
        raise ValueError("unsupported qualification record schema")
    if record_type is not None and payload.get("record_type") != record_type:
        raise ValueError(f"expected {record_type!r} qualification record")
    if not is_sha256(payload.get("record_hash")):
        raise ValueError("qualification record hash is missing or malformed")
    if payload["record_hash"] != _record_hash(payload):
        raise ValueError("qualification record hash does not match its payload")
    validators = {
        _STATIC_RECORD_TYPE: _validate_static_record,
        _CAPACITY_RECORD_TYPE: _validate_capacity_record,
        _PROBE_EXECUTION_RECORD_TYPE: _validate_probe_execution_record,
        _PROBE_BENCHMARK_RECORD_TYPE: _validate_probe_benchmark_record,
        _ASSESSMENT_RECORD_TYPE: _validate_assessment_record,
    }
    observed_type = payload.get("record_type")
    validator = validators.get(observed_type)
    if validator is None:
        raise ValueError(f"unsupported qualification record type: {observed_type!r}")
    validator(payload)
    return payload


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_output_path(
    path: Path,
    *,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    """Reject qualification outputs that can contaminate bound data roots."""

    target = _absolute_path(path)
    for root in forbidden_roots:
        forbidden = Path(root).resolve(strict=False)
        if _path_within(target, forbidden) or _path_within(
            target.resolve(strict=False),
            forbidden,
        ):
            raise ValueError(
                f"qualification output must be outside protected root: {forbidden}"
            )
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"qualification output path contains a symlink: {current}")
    return target


def _open_directory_nofollow(path: Path) -> int:
    target = _absolute_path(path)
    descriptor = os.open(
        target.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        for component in target.parts[1:]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_record_identity_at(
    directory_fd: int,
    name: str,
) -> tuple[str, tuple[int, int, int, int]] | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("qualification record target must be a regular file")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as stream:
            content = stream.read()
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("qualification record changed while being read")
        return content, (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
        )
    finally:
        os.close(descriptor)


def _make_record_read_only_at(
    directory_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != expected_identity[:2]
            or before.st_nlink != 1
        ):
            raise ValueError("qualification record target must be a regular file")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o400
            or after.st_nlink != 1
        ):
            raise ValueError("qualification record changed while being hardened")
        return (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
        )
    finally:
        os.close(descriptor)


def _verify_record_publication(
    target: Path,
    *,
    directory_fd: int,
    content: str,
    expected_identity: tuple[int, int, int, int],
) -> None:
    verification_fd = _open_existing_directory_nofollow(target.parent)
    try:
        if not _same_open_directory(directory_fd, verification_fd):
            raise ValueError("qualification record parent changed during publication")
        published = _read_record_identity_at(verification_fd, target.name)
        if published is None:
            raise ValueError("qualification record is absent after publication")
        observed_content, observed_identity = published
        if (
            observed_identity != expected_identity
            or observed_content != content
            or not stat.S_ISREG(observed_identity[2])
            or stat.S_IMODE(observed_identity[2]) != 0o400
            or observed_identity[3] != 1
        ):
            raise ValueError("qualification record changed during publication")
    finally:
        os.close(verification_fd)


def write_record(
    path: Path,
    payload: Mapping[str, Any],
    *,
    forbidden_roots: Sequence[Path] = (),
) -> None:
    """Create one write-once operational JSON record with owner-read-only mode."""

    checked = verify_record(dict(payload))
    content = (
        json.dumps(
            checked,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    target = validate_output_path(path, forbidden_roots=forbidden_roots)
    directory_fd = _open_directory_nofollow(target.parent)
    temporary_name = f".{target.name}.{secrets.token_hex(16)}.tmp"
    temporary_exists = False
    try:
        existing = _read_record_identity_at(directory_fd, target.name)
        if existing is not None:
            existing_content, published_identity = existing
            if existing_content != content:
                raise ValueError(
                    f"refusing to overwrite qualification record: {target}"
                )
            published_identity = _make_record_read_only_at(
                directory_fd,
                target.name,
                expected_identity=published_identity,
            )
        else:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            temporary_exists = True
            try:
                with os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8",
                    closefd=False,
                ) as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                temporary_metadata = os.fstat(descriptor)
                published_identity = (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                    temporary_metadata.st_mode,
                    temporary_metadata.st_nlink,
                )
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary_name,
                    target.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raced = _read_record_identity_at(directory_fd, target.name)
                if raced is None or raced[0] != content:
                    raise ValueError(
                        f"refusing to overwrite qualification record: {target}"
                    ) from None
                published_identity = _make_record_read_only_at(
                    directory_fd,
                    target.name,
                    expected_identity=raced[1],
                )
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)
        _verify_record_publication(
            target,
            directory_fd=directory_fd,
            content=content,
            expected_identity=published_identity,
        )
    finally:
        if temporary_exists:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _full_git_commit(value: str) -> str:
    value = _full_git_sha(value, label="execution commit")
    if value not in APPROVED_V2_EXECUTION_COMMITS:
        approved = ", ".join(APPROVED_V2_EXECUTION_COMMITS)
        raise ValueError(
            f"execution commit is not approved for symbolic v2; expected {approved}"
        )
    return value


def _full_git_sha(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a full lowercase 40-hex Git SHA")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_directory(repository: Path) -> Path:
    marker = repository / ".git"
    metadata = marker.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("Git metadata marker cannot be a symlink")
    if stat.S_ISDIR(metadata.st_mode):
        return marker.resolve(strict=True)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Git metadata marker must be a directory or gitdir file")
    line = marker.read_text(encoding="utf-8").strip()
    prefix = "gitdir: "
    if not line.startswith(prefix) or "\n" in line:
        raise ValueError("Git worktree metadata marker is malformed")
    candidate = Path(line.removeprefix(prefix))
    directory = (
        candidate if candidate.is_absolute() else repository / candidate
    ).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Git metadata directory does not exist")
    return directory


def _git_command(
    git_path: str,
    repository: Path,
    git_directory: Path,
    *arguments: str,
) -> tuple[str, ...]:
    return (
        git_path,
        "--no-optional-locks",
        f"--git-dir={git_directory}",
        f"--work-tree={repository}",
        *arguments,
    )


def _stable_regular_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    descriptor = os.open(
        resolved,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"installed dependency is not a regular file: {resolved}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, MIB):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(
                f"installed dependency changed while being hashed: {resolved}"
            )
        return {
            "resolved_path": str(resolved),
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        }
    finally:
        os.close(descriptor)


def _dependency_environment_hash() -> str:
    distributions: list[dict[str, Any]] = []
    for distribution in importlib_metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower()
        version = distribution.version
        declared_files = distribution.files
        if not name or not version or declared_files is None:
            raise ValueError("installed dependency metadata is incomplete")
        files = []
        for declared_path in sorted(declared_files, key=os.fspath):
            files.append(
                {
                    "declared_path": os.fspath(declared_path),
                    **_stable_regular_file_identity(
                        Path(distribution.locate_file(declared_path))
                    ),
                }
            )
        distributions.append(
            {
                "name": name,
                "version": version,
                "files": files,
            }
        )
    distributions.sort(
        key=lambda item: (
            item["name"],
            item["version"],
            scientific_hash(
                item["files"],
                domain="operations.host-qualification-dependency-files.v1",
            ),
        )
    )
    return scientific_hash(
        distributions,
        domain="operations.host-qualification-dependencies.v1",
    )


def _tool_execution_environment_hash() -> str:
    identity = _execution_environment_identity(
        Path(sys.prefix),
        python_version=platform.python_version(),
        reject_import_bytecode=False,
    )
    if identity["includes_system_site_packages"]:
        raise ValueError("tool environment must disable system site-packages")
    return _execution_environment_hash(identity)


def _tool_identity(*, git_path: str) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    git_directory = _git_directory(repository)
    commit = _full_git_sha(
        _run_text(
            _git_command(
                git_path,
                repository,
                git_directory,
                "rev-parse",
                "HEAD",
            ),
            cwd=repository,
        ),
        label="qualification tool commit",
    )
    dirty = _run_text(
        _git_command(
            git_path,
            repository,
            git_directory,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ),
        cwd=repository,
    )
    python = Path(sys.executable).resolve(strict=True)
    sources = (
        Path(__file__).resolve(),
        repository / "scripts" / "qualify_symbolic_v2_host.py",
    )
    source_hash = scientific_hash(
        {
            str(source.relative_to(repository)): _sha256_file(source)
            for source in sources
        },
        domain="operations.host-qualification-tool-source.v1",
    )
    return {
        "repository": str(repository),
        "commit": commit,
        "worktree_clean": not dirty,
        "source_hash": source_hash,
        "git_directory": str(git_directory),
        "git_path": git_path,
        "git_sha256": _sha256_file(Path(git_path)),
        "python_path": str(python),
        "python_sha256": _sha256_file(python),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": platform.python_version(),
        "dependency_lock_sha256": _sha256_file(repository / "uv.lock"),
        "dependency_environment_hash": _dependency_environment_hash(),
        "startup_environment_hash": _tool_execution_environment_hash(),
    }


def _tool_identity_hash(tool: Mapping[str, Any]) -> str:
    return scientific_hash(tool, domain="operations.host-qualification-tool.v1")


def _machine_boot_identity(
    *,
    proc_root: Path,
    machine_id_path: Path,
) -> dict[str, str]:
    machine_id = machine_id_path.read_text(encoding="utf-8").strip()
    boot_id = (
        (proc_root / "sys" / "kernel" / "random" / "boot_id")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not machine_id or not boot_id:
        raise ValueError("machine and boot identity values must be nonempty")
    return {
        "machine_id_hash": scientific_hash(
            machine_id,
            domain="operations.host-machine-id.v1",
        ),
        "boot_id_hash": scientific_hash(
            boot_id,
            domain="operations.host-boot-id.v1",
        ),
    }


def _repository_snapshot(
    repository: Path,
    *,
    lock: Path,
    execution_python: Path,
    git_path: str,
    git_directory: Path,
) -> dict[str, Any]:
    status = _run_text(
        _git_command(
            git_path,
            repository,
            git_directory,
            "status",
            "--porcelain=v2",
            "--branch",
            "--untracked-files=normal",
        ),
        cwd=repository,
    )
    lines = status.splitlines()
    prefix = "# branch.oid "
    commit_lines = [line for line in lines if line.startswith(prefix)]
    if len(commit_lines) != 1:
        raise ValueError("cannot determine checkout commit from atomic status snapshot")
    commit = _full_git_sha(
        commit_lines[0].removeprefix(prefix),
        label="observed execution commit",
    )
    changes = tuple(line for line in lines if not line.startswith("# "))
    return {
        "commit": commit,
        "clean": not changes,
        "status": "\n".join(changes),
        "uv_lock_sha256": _sha256_file(lock) if lock.is_file() else None,
        "execution_python_sha256": (
            _sha256_file(execution_python.resolve())
            if execution_python.is_file()
            else None
        ),
    }


def _command(arguments: Sequence[object]) -> str:
    return shlex.join(str(argument) for argument in arguments)


def _runtime_prefix(
    uv_path: str,
    execution_python: str,
) -> tuple[str, ...]:
    return (
        uv_path,
        "run",
        "--frozen",
        "--no-sync",
        "--python",
        execution_python,
        "python",
    )


def _capacity_arguments(
    stage: str,
    *,
    uv_path: str,
    execution_python: str,
) -> tuple[str, ...]:
    if stage not in _CAPACITY_SPECS:
        raise ValueError("capacity stage must be e192 or e768")
    arguments = [
        *_runtime_prefix(uv_path, execution_python),
        "scripts/benchmark_analysis_scale.py",
        "--config",
        "configs/symbolic-calibration-v2.json",
    ]
    if stage == "e768":
        arguments.extend(("--environment-replicas", "768"))
    arguments.extend(("--metrics", "29", "--positive-checkpoint-metrics", "31"))
    return tuple(arguments)


def _secured_probe_arguments(
    kind: str,
    *,
    uv_path: str,
    execution_python: str,
    artifact_descriptor: int | str,
    repository: Path,
) -> tuple[str, ...]:
    runtime = _runtime_prefix(uv_path, execution_python)
    descriptor = str(artifact_descriptor)
    if kind == "execution":
        return (
            *runtime,
            "-c",
            _PRECREATED_PROBE_ROOT_BOOTSTRAP,
            descriptor,
            str(repository),
            str(repository / "configs" / "symbolic-calibration-v2.json"),
            str(repository / "configs" / "symbolic-artifact-ingestion-probe-v2.json"),
            ".",
            "--workers",
            "4",
        )
    if kind == "benchmark":
        return (
            *runtime,
            "-c",
            _ANCHORED_PROBE_BENCHMARK_BOOTSTRAP,
            descriptor,
            str(repository),
            str(repository / "configs" / "symbolic-artifact-ingestion-probe-v2.json"),
            ".",
            "--projected-runs",
            str(V2_PROJECTED_RUNS),
            "--budget-multiplier",
            "2",
        )
    raise ValueError("probe step kind must be execution or benchmark")


def _external_root(repo_root: Path, candidate: Path, *, label: str) -> Path:
    root = candidate.resolve(strict=False)
    if root == repo_root or repo_root in root.parents:
        raise ValueError(f"{label} must be outside the execution repository")
    return root


def _safe_probe_root(repo_root: Path, probe_root: Path) -> Path:
    return _external_root(repo_root, probe_root, label="probe root")


def build_exact_plan(
    *,
    repo_root: Path,
    execution_commit: str,
    probe_root: Path,
    uv_path: Path | None = None,
    execution_python: Path | None = None,
) -> dict[str, Any]:
    """Return the exact registered v2 qualification commands without running them."""

    repository = repo_root.resolve()
    commit = _full_git_commit(execution_commit)
    probe = _safe_probe_root(repository, probe_root)
    located_uv = shutil.which("uv") if uv_path is None else str(uv_path)
    if located_uv is None:
        raise ValueError("uv executable is required to build a qualification plan")
    uv = (
        Path(located_uv).resolve(strict=True)
        if uv_path is None
        else _absolute_path(Path(located_uv))
    )
    if not uv.is_absolute() or (
        uv_path is None and (not uv.is_file() or not os.access(uv, os.X_OK))
    ):
        raise ValueError("uv executable must be an absolute executable file")
    python = _absolute_path(
        repository / ".venv" / "bin" / "python"
        if execution_python is None
        else execution_python
    )
    located_git = shutil.which("git")
    if located_git is None:
        raise ValueError("git executable is required to build a qualification plan")
    git = Path(located_git).resolve(strict=True)
    if not git.is_file() or not os.access(git, os.X_OK):
        raise ValueError("git executable must be an absolute executable file")
    return _exact_plan_payload(
        repository=repository,
        commit=commit,
        probe=probe,
        uv=uv,
        python=python,
        git=git,
        git_directory=_git_directory(repository),
    )


def _exact_plan_payload(
    *,
    repository: Path,
    commit: str,
    probe: Path,
    uv: Path,
    python: Path,
    git: Path,
    git_directory: Path,
) -> dict[str, Any]:
    runtime = _runtime_prefix(str(uv), str(python))
    probe_execute = (
        *runtime,
        "-m",
        "scripts.run_ingestion_probe",
        "configs/symbolic-calibration-v2.json",
        "configs/symbolic-artifact-ingestion-probe-v2.json",
        str(probe),
        "--workers",
        "4",
    )
    probe_benchmark = (
        *runtime,
        "scripts/benchmark_artifact_ingestion.py",
        "configs/symbolic-artifact-ingestion-probe-v2.json",
        str(probe),
        "--projected-runs",
        str(V2_PROJECTED_RUNS),
        "--budget-multiplier",
        "2",
    )
    return {
        "execution_mode": "print-only",
        "working_directory": str(repository),
        "execution_commit": commit,
        "checkout": [
            _command(
                _git_command(
                    str(git),
                    repository,
                    git_directory,
                    "checkout",
                    "--detach",
                    commit,
                )
            ),
            _command(
                _git_command(
                    str(git),
                    repository,
                    git_directory,
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                )
            ),
        ],
        "capacity": {
            "e192": _command(
                _capacity_arguments(
                    "e192",
                    uv_path=str(uv),
                    execution_python=str(python),
                )
            ),
            "e768": _command(
                _capacity_arguments(
                    "e768",
                    uv_path=str(uv),
                    execution_python=str(python),
                )
            ),
        },
        "probe": {
            "artifact_root": str(probe),
            "underlying_execute": _command(probe_execute),
            "underlying_benchmark": _command(probe_benchmark),
            "qualifying_execute_template": _command(
                (
                    "<TOOL_PYTHON>",
                    "<TOOL_CHECKOUT>/scripts/qualify_symbolic_v2_host.py",
                    "run-probe",
                    "--kind",
                    "execution",
                    "--host-record",
                    "<HOST_RECORD>",
                    "--output",
                    "<PROBE_EXECUTION_RECORD>",
                )
            ),
            "qualifying_benchmark_template": _command(
                (
                    "<TOOL_PYTHON>",
                    "<TOOL_CHECKOUT>/scripts/qualify_symbolic_v2_host.py",
                    "run-probe",
                    "--kind",
                    "benchmark",
                    "--host-record",
                    "<HOST_RECORD>",
                    "--probe-execution-record",
                    "<PROBE_EXECUTION_RECORD>",
                    "--output",
                    "<PROBE_BENCHMARK_RECORD>",
                )
            ),
        },
        "order_before_calibration": [
            "static-host-inspection",
            "e192",
            "e768",
            "probe-execute",
            "probe-benchmark",
            "independent-review",
        ],
        "repeat_before_selected_confirmation": ["e192", "e768"],
    }


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        key = fields[0].removesuffix(":")
        value = int(fields[1])
        if len(fields) > 2 and fields[2] == "kB":
            value *= 1024
        values[key] = value
    return values


def _decode_mount_field(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _mount_for(path: Path, mountinfo: Path) -> dict[str, str]:
    target = path.resolve()
    candidates: list[tuple[int, dict[str, str]]] = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        mount_point = Path(_decode_mount_field(fields[4]))
        if target != mount_point and mount_point not in target.parents:
            continue
        candidates.append(
            (
                len(mount_point.parts),
                {
                    "device": _decode_mount_field(fields[separator + 2]),
                    "mount_options": fields[5],
                    "mount_point": str(mount_point),
                    "type": fields[separator + 1],
                },
            )
        )
    if not candidates:
        raise ValueError(f"cannot resolve mount for storage root: {target}")
    return max(candidates, key=lambda item: item[0])[1]


def _block_is_solid_state(device: str, sys_class_block: Path) -> bool | None:
    name = Path(os.path.realpath(device)).name
    if name.startswith("nvme"):
        return True
    node = sys_class_block / name
    if not node.exists():
        return None
    partition = node / "partition"
    if partition.exists():
        resolved = node.resolve()
        node = sys_class_block / resolved.parent.name
    rotational = node / "queue" / "rotational"
    if rotational.is_file():
        return rotational.read_text(encoding="utf-8").strip() == "0"
    slaves = node / "slaves"
    if slaves.is_dir():
        values = [
            _block_is_solid_state(f"/dev/{child.name}", sys_class_block)
            for child in slaves.iterdir()
        ]
        known = [value for value in values if value is not None]
        return all(known) if known and len(known) == len(values) else None
    return None


def _run_text(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=_sanitized_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if len(result.stdout) > MIB or len(result.stderr) > MIB:
        raise ValueError("qualification identity command output exceeded 1 MiB")
    return result.stdout.strip()


def _requirement(
    name: str,
    *,
    passed: bool,
    observed: object,
    required: object,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "required": required,
    }


def _python_version_supported(value: str | None) -> bool:
    if value is None:
        return False
    try:
        major, minor, *_ = (int(part) for part in value.split("."))
    except ValueError:
        return False
    return (3, 11) <= (major, minor) < (4, 0)


def _host_identity_hash(
    *,
    execution: Mapping[str, Any],
    host_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    host: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> str:
    return scientific_hash(
        {
            "execution_commit": execution["observed_commit"],
            "host_identity": host_identity,
            "host": {
                "physical_memory_bytes": host["physical_memory_bytes"],
                "kernel": runtime["kernel"],
                "logical_cpu_count": runtime["logical_cpu_count"],
            },
            "runtime": {
                "execution_python": runtime["execution_python"],
                "execution_python_path": runtime["execution_python_path"],
                "execution_python_sha256": runtime["execution_python_sha256"],
                "execution_environment_hash": runtime["execution_environment_hash"],
                "git": runtime["git"],
                "git_path": runtime["git_path"],
                "git_sha256": runtime["git_sha256"],
                "uv": runtime["uv"],
                "uv_path": runtime["uv_path"],
                "uv_sha256": runtime["uv_sha256"],
            },
            "storage": {
                key: storage[key]
                for key in (
                    "device",
                    "directory_device_id",
                    "mount_point",
                    "storage_root",
                    "type",
                )
            },
        },
        domain="operations.host-identity.v2",
    )


def _static_requirements(
    *,
    execution: Mapping[str, Any],
    tool: Mapping[str, Any],
    runtime: Mapping[str, Any],
    host_identity: Mapping[str, Any],
    host: Mapping[str, Any],
    storage: Mapping[str, Any],
    registered_inputs: Mapping[str, Any],
    probe_storage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _requirement(
            "execution-commit",
            passed=execution["observed_commit"] == execution["expected_commit"],
            observed=execution["observed_commit"],
            required=execution["expected_commit"],
        ),
        _requirement(
            "clean-worktree",
            passed=execution["worktree_clean"],
            observed=execution["status"] or "clean",
            required="clean",
        ),
        _requirement(
            "linux-runtime",
            passed=runtime["platform"].startswith("linux"),
            observed=runtime["platform"],
            required="linux",
        ),
        _requirement(
            "tool-python",
            passed=_python_version_supported(runtime["tool_python"]),
            observed=runtime["tool_python"],
            required=">=3.11,<4",
        ),
        _requirement(
            "execution-python",
            passed=(
                _python_version_supported(runtime["execution_python"])
                and runtime["execution_python_sha256"] is not None
                and runtime["execution_python_path"]
                == str(Path(execution["repository"]) / ".venv" / "bin" / "python")
                and runtime["execution_prefix"]
                == str(Path(execution["repository"]) / ".venv")
            ),
            observed={
                "path": runtime["execution_python_path"],
                "prefix": runtime["execution_prefix"],
                "version": runtime["execution_python"],
            },
            required="synced checkout .venv with Python >=3.11,<4",
        ),
        _requirement(
            "uv",
            passed=(
                runtime["uv"] is not None
                and runtime["uv_path"] is not None
                and runtime["uv_sha256"] is not None
                and Path(runtime["uv_path"]).is_absolute()
            ),
            observed={
                "path": runtime["uv_path"],
                "version": runtime["uv"],
            },
            required="absolute executable with recorded SHA-256",
        ),
        _requirement(
            "git",
            passed=(
                runtime["git"] is not None
                and runtime["git_path"] is not None
                and runtime["git_sha256"] is not None
                and Path(runtime["git_path"]).is_absolute()
            ),
            observed={
                "path": runtime["git_path"],
                "version": runtime["git"],
            },
            required="absolute executable with recorded SHA-256",
        ),
        _requirement(
            "dependency-lock",
            passed=is_sha256(execution["uv_lock_sha256"]),
            observed=execution["uv_lock_sha256"] or "missing",
            required="SHA-256 of checked-in uv.lock",
        ),
        _requirement(
            "execution-environment-synced",
            passed=runtime["environment_synced"],
            observed=runtime["environment_synced"],
            required=True,
        ),
        _requirement(
            "execution-environment-bound",
            passed=(
                isinstance(runtime["execution_environment"], dict)
                and is_sha256(runtime["execution_environment_hash"])
                and _execution_environment_hash(runtime["execution_environment"])
                == runtime["execution_environment_hash"]
                and not runtime["execution_environment"][
                    "includes_system_site_packages"
                ]
                and runtime["execution_environment"]["site_packages"]["file_count"] > 0
                and all(
                    (
                        _path_within(
                            Path(path),
                            Path(execution["repository"]) / "src",
                        )
                        or _path_within(
                            Path(path),
                            Path(
                                runtime["execution_environment"]["site_packages"][
                                    "path"
                                ]
                            ),
                        )
                    )
                    for path in runtime["execution_environment"]["pth_path_entries"]
                )
            ),
            observed={
                "hash": runtime["execution_environment_hash"],
                "includes_system_site_packages": (
                    runtime["execution_environment"]["includes_system_site_packages"]
                    if isinstance(runtime["execution_environment"], dict)
                    else None
                ),
                "pth_path_entries": (
                    runtime["execution_environment"]["pth_path_entries"]
                    if isinstance(runtime["execution_environment"], dict)
                    else []
                ),
            },
            required=(
                "metadata-and-byte-bound checkout .venv with system site-packages "
                "disabled and path entries confined to the checkout"
            ),
        ),
        _requirement(
            "qualification-tool-clean",
            passed=tool["worktree_clean"],
            observed=tool["worktree_clean"],
            required=True,
        ),
        _requirement(
            "qualification-tool-source",
            passed=is_sha256(tool["source_hash"]),
            observed=tool["source_hash"],
            required="SHA-256-bound reviewed tool sources",
        ),
        _requirement(
            "qualification-tool-dependencies",
            passed=(
                is_sha256(tool["dependency_lock_sha256"])
                and is_sha256(tool["dependency_environment_hash"])
                and is_sha256(tool["startup_environment_hash"])
                and is_sha256(tool["python_sha256"])
            ),
            observed={
                "lock": tool["dependency_lock_sha256"],
                "environment": tool["dependency_environment_hash"],
                "startup_environment": tool["startup_environment_hash"],
                "python": tool["python_sha256"],
            },
            required="hash-bound tool interpreter, lock, and installed environment",
        ),
        _requirement(
            "host-machine-identity",
            passed=is_sha256(host_identity["machine_id_hash"]),
            observed=host_identity["machine_id_hash"],
            required="hashed machine identity",
        ),
        _requirement(
            "host-boot-identity",
            passed=is_sha256(host_identity["boot_id_hash"]),
            observed=host_identity["boot_id_hash"],
            required="hashed boot identity",
        ),
        _requirement(
            "physical-memory",
            passed=host["physical_memory_bytes"] >= MINIMUM_PHYSICAL_MEMORY_BYTES,
            observed=host["physical_memory_bytes"],
            required=MINIMUM_PHYSICAL_MEMORY_BYTES,
        ),
        _requirement(
            "local-filesystem",
            passed=storage["local_filesystem"],
            observed=storage["type"],
            required="local non-pseudo filesystem",
        ),
        _requirement(
            "ssd-or-nvme",
            passed=storage["solid_state"] is True,
            observed=storage["solid_state"],
            required=True,
        ),
        _requirement(
            "read-write-mount",
            passed=storage["read_write_mount"],
            observed=storage["mount_options"],
            required="rw",
        ),
        _requirement(
            "available-storage",
            passed=storage["available_bytes"] >= storage["required_bytes"],
            observed=storage["available_bytes"],
            required=storage["required_bytes"],
        ),
        _requirement(
            "declared-storage-margin",
            passed=storage["additional_storage_bytes"] > 0,
            observed=storage["additional_storage_bytes"],
            required="positive bytes for reports, staging, and recovery",
        ),
        _requirement(
            "available-inodes",
            passed=storage["available_inodes"] >= storage["required_inodes"],
            observed=storage["available_inodes"],
            required=storage["required_inodes"],
        ),
        _requirement(
            "declared-inode-margin",
            passed=storage["additional_inodes"] > 0,
            observed=storage["additional_inodes"],
            required="positive inodes for reports, staging, and recovery",
        ),
        _requirement(
            "writable-storage-root",
            passed=storage["writable"],
            observed=storage["writable"],
            required=True,
        ),
        _requirement(
            "registered-calibration-config",
            passed=(
                registered_inputs["calibration_config_hash"]
                == V2_CALIBRATION_CONFIG_HASH
            ),
            observed=registered_inputs["calibration_config_hash"],
            required=V2_CALIBRATION_CONFIG_HASH,
        ),
        _requirement(
            "registered-probe-config",
            passed=registered_inputs["probe_config_hash"] == V2_PROBE_CONFIG_HASH,
            observed=registered_inputs["probe_config_hash"],
            required=V2_PROBE_CONFIG_HASH,
        ),
        _requirement(
            "probe-root-absent",
            passed=not probe_storage["existed_at_inspection"],
            observed=probe_storage["existed_at_inspection"],
            required=False,
        ),
        _requirement(
            "probe-on-intended-storage",
            passed=probe_storage["on_intended_storage"],
            observed=probe_storage["artifact_root"],
            required=f"within {storage['storage_root']}",
        ),
    ]


def inspect_host(
    *,
    repo_root: Path,
    execution_commit: str,
    storage_root: Path,
    additional_storage_bytes: int,
    additional_inodes: int,
    probe_root: Path | None = None,
    proc_root: Path = Path("/proc"),
    sys_class_block: Path = Path("/sys/class/block"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Capture static host evidence; never execute a benchmark or study."""

    repository = repo_root.resolve()
    commit = _full_git_commit(execution_commit)
    storage = _external_root(repository, storage_root, label="storage root")
    margins = (additional_storage_bytes, additional_inodes)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in margins
    ):
        raise ValueError("storage and inode margins must be positive")
    required_storage_bytes = REFERENCE_PAIR_RAW_BYTES + additional_storage_bytes
    required_inodes = REFERENCE_PAIR_RAW_FILES + additional_inodes

    lock = repository / "uv.lock"
    execution_python = repository / ".venv" / "bin" / "python"
    located_git = shutil.which("git")
    if located_git is None:
        raise ValueError("git executable is required for host inspection")
    git_path = str(Path(located_git).resolve(strict=True))
    if not os.access(git_path, os.X_OK):
        raise ValueError("git executable must be executable")
    git_directory = _git_directory(repository)
    snapshot = _repository_snapshot(
        repository,
        lock=lock,
        execution_python=execution_python,
        git_path=git_path,
        git_directory=git_directory,
    )
    calibration = load_experiment_config(
        repository / "configs" / "symbolic-calibration-v2.json"
    )
    registered_symbolic_study(calibration.name).verify_calibration(calibration)
    checked_probe = load_experiment_config(
        repository / "configs" / "symbolic-artifact-ingestion-probe-v2.json"
    )

    memory = _read_key_values(proc_root / "meminfo")
    vmstat = _read_key_values(proc_root / "vmstat")
    mount = _mount_for(storage, proc_root / "self" / "mountinfo")
    stat = os.statvfs(storage)
    available_bytes = stat.f_bavail * stat.f_frsize
    available_inodes = stat.f_favail
    filesystem_type = mount["type"]
    local_filesystem = (
        filesystem_type not in _NETWORK_FILESYSTEMS
        and filesystem_type not in _NON_STORAGE_FILESYSTEMS
    )
    read_write_mount = "rw" in mount["mount_options"].split(",")
    solid_state = _block_is_solid_state(mount["device"], sys_class_block)
    storage_metadata = os.stat(storage, follow_symlinks=False)
    located_uv = shutil.which("uv")
    uv_path = (
        str(Path(located_uv).resolve(strict=True)) if located_uv is not None else None
    )
    uv_version = None if uv_path is None else _run_text((uv_path, "--version"))
    uv_sha256 = None if uv_path is None else _sha256_file(Path(uv_path))
    git_version = _run_text((git_path, "--version"))
    git_sha256 = _sha256_file(Path(git_path))
    execution_python_version = (
        _run_text(
            (
                str(execution_python),
                "-c",
                "import platform; print(platform.python_version())",
            ),
            cwd=repository,
        )
        if execution_python.is_file() and os.access(execution_python, os.X_OK)
        else None
    )
    execution_prefix = (
        _run_text(
            (
                str(execution_python),
                "-c",
                "import sys; print(sys.prefix)",
            ),
            cwd=repository,
        )
        if execution_python_version is not None
        else None
    )
    environment_synced = False
    if uv_path is not None and execution_python_version is not None:
        try:
            _run_text(
                (uv_path, "sync", "--frozen", "--all-groups", "--check"),
                cwd=repository,
            )
            environment_synced = (
                Path(execution_prefix).resolve() == (repository / ".venv").resolve()
            )
        except (OSError, subprocess.CalledProcessError):
            environment_synced = False
    execution_environment: dict[str, Any] | None = None
    execution_environment_hash: str | None = None
    if execution_python_version is not None and execution_prefix is not None:
        try:
            execution_environment = _execution_environment_identity(
                Path(execution_prefix),
                python_version=execution_python_version,
            )
            execution_environment_hash = _execution_environment_hash(
                execution_environment
            )
        except (OSError, ValueError):
            execution_environment = None
            execution_environment_hash = None
    if uv_path is None:
        raise ValueError("uv executable is required for host inspection")
    plan = _exact_plan_payload(
        repository=repository,
        commit=commit,
        probe=_safe_probe_root(
            repository,
            probe_root or storage / "symbolic-v2-ingestion-probe",
        ),
        uv=Path(uv_path),
        python=_absolute_path(execution_python),
        git=Path(git_path),
        git_directory=git_directory,
    )
    storage_writable = os.access(storage, os.W_OK | os.X_OK)
    planned_probe_root = Path(plan["probe"]["artifact_root"])
    probe_on_storage = (
        planned_probe_root == storage or storage in planned_probe_root.parents
    )
    runtime_record = {
        "execution_python": execution_python_version,
        "execution_python_path": str(execution_python),
        "execution_python_sha256": snapshot["execution_python_sha256"],
        "execution_prefix": execution_prefix,
        "environment_synced": environment_synced,
        "execution_environment": execution_environment,
        "execution_environment_hash": execution_environment_hash,
        "kernel": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "platform": sys.platform,
        "tool_python": platform.python_version(),
        "git_path": git_path,
        "git": git_version,
        "git_sha256": git_sha256,
        "uv_path": uv_path,
        "uv": uv_version,
        "uv_sha256": uv_sha256,
    }
    host_record = {
        "available_memory_bytes": memory.get("MemAvailable", 0),
        "physical_memory_bytes": memory.get("MemTotal", 0),
        "swap_bytes": memory.get("SwapTotal", 0),
        "vmstat": {
            "pswpin_pages": vmstat.get("pswpin", 0),
            "pswpout_pages": vmstat.get("pswpout", 0),
        },
    }
    storage_record = {
        **mount,
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
        "additional_storage_bytes": additional_storage_bytes,
        "additional_inodes": additional_inodes,
        "directory_device_id": storage_metadata.st_dev,
        "directory_inode": storage_metadata.st_ino,
        "local_filesystem": local_filesystem,
        "paired_raw_reference_bytes": REFERENCE_PAIR_RAW_BYTES,
        "paired_raw_reference_files": REFERENCE_PAIR_RAW_FILES,
        "required_bytes": required_storage_bytes,
        "required_inodes": required_inodes,
        "solid_state": solid_state,
        "read_write_mount": read_write_mount,
        "storage_root": str(storage),
        "writable": storage_writable,
    }
    execution_record = {
        "expected_commit": commit,
        "observed_commit": snapshot["commit"],
        "repository": str(repository),
        "git_directory": str(git_directory),
        "status": snapshot["status"],
        "worktree_clean": snapshot["clean"],
        "uv_lock_sha256": snapshot["uv_lock_sha256"],
    }
    tool_record = _tool_identity(git_path=git_path)
    tool_hash = _tool_identity_hash(tool_record)
    machine_identity = _machine_boot_identity(
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )
    probe_storage_record = {
        "artifact_root": str(planned_probe_root),
        "existed_at_inspection": os.path.lexists(planned_probe_root),
        "on_intended_storage": probe_on_storage,
    }
    registered_inputs = {
        "calibration_config_hash": calibration.config_hash,
        "probe_config_hash": checked_probe.config_hash,
    }
    requirements = _static_requirements(
        execution=execution_record,
        tool=tool_record,
        runtime=runtime_record,
        host_identity=machine_identity,
        host=host_record,
        storage=storage_record,
        registered_inputs=registered_inputs,
        probe_storage=probe_storage_record,
    )
    passed = all(item["passed"] for item in requirements)
    identity_hash = _host_identity_hash(
        execution=execution_record,
        host_identity=machine_identity,
        runtime=runtime_record,
        host=host_record,
        storage=storage_record,
    )
    payload = {
        "schema_version": 1,
        "record_type": _STATIC_RECORD_TYPE,
        "recorded_at": _utc_now(),
        "execution": execution_record,
        "tool": tool_record,
        "tool_identity_hash": tool_hash,
        "host_identity": machine_identity,
        "host_identity_hash": identity_hash,
        "runtime": runtime_record,
        "host": host_record,
        "storage": storage_record,
        "probe_storage": probe_storage_record,
        "registered_inputs": registered_inputs,
        "requirements": requirements,
        "plan": plan,
        "decision": {
            "qualification_steps_executed": False,
            "registered_execution_authorized": False,
            "static_prerequisites_passed": passed,
        },
        "scientific_boundary": {
            "creates_study_artifacts": False,
            "executes_registered_study": False,
            "inspects_probe_metrics": False,
            "scientific_use": "prohibited",
        },
    }
    return _signed(payload)


def _capacity_decision(
    stage: str,
    *,
    result: Mapping[str, Any],
    before_available_memory_bytes: int,
    after_available_memory_bytes: int,
    minimum_available_memory_bytes: int,
    process_major_faults: int,
    swapout_delta_pages: int,
    checkout_unchanged: bool = True,
    execution_environment_unchanged: bool = True,
) -> dict[str, Any]:
    spec = _CAPACITY_SPECS[stage]
    expected_integers = {
        "algorithm_replicas": 8,
        "checkpoint_count": 13,
        "checkpoint_zero_metric_count": 29,
        "metric_count": 29,
        "positive_checkpoint_metric_count": 31,
        "environment_replicas": spec["environment_replicas"],
        "observation_count": spec["observation_count"],
        "pooled_checkpoint_count": 624,
    }
    shape_passed = all(
        type(result.get(name)) is int and result.get(name) == expected
        for name, expected in expected_integers.items()
    ) and (
        isinstance(result.get("dataset_hash"), str)
        and result.get("dataset_hash") == spec["dataset_hash"]
    )
    maximum_rss = result.get("maximum_rss_mib")
    rss_before = result.get("rss_before_mib")
    rss_increment = result.get("rss_increment_upper_bound_mib")
    dataset_elapsed = result.get("dataset_elapsed_seconds")
    pool_elapsed = result.get("pool_elapsed_seconds")
    total_elapsed = result.get("total_elapsed_seconds")
    valid_rss = (
        isinstance(maximum_rss, (int, float))
        and not isinstance(maximum_rss, bool)
        and math.isfinite(maximum_rss)
        and maximum_rss > 0
    )
    benchmark_metrics_consistent = (
        valid_rss
        and isinstance(rss_before, (int, float))
        and not isinstance(rss_before, bool)
        and math.isfinite(rss_before)
        and 0 < rss_before <= maximum_rss
        and isinstance(rss_increment, (int, float))
        and not isinstance(rss_increment, bool)
        and math.isfinite(rss_increment)
        and rss_increment >= 0
        and abs(rss_increment - max(0.0, maximum_rss - rss_before)) <= 0.002
        and isinstance(dataset_elapsed, (int, float))
        and not isinstance(dataset_elapsed, bool)
        and math.isfinite(dataset_elapsed)
        and dataset_elapsed > 0
        and isinstance(pool_elapsed, (int, float))
        and not isinstance(pool_elapsed, bool)
        and math.isfinite(pool_elapsed)
        and pool_elapsed >= 0
        and isinstance(total_elapsed, (int, float))
        and not isinstance(total_elapsed, bool)
        and math.isfinite(total_elapsed)
        and total_elapsed > 0
        and total_elapsed + 0.003 >= dataset_elapsed
        and total_elapsed + 0.003 >= pool_elapsed
        and abs(total_elapsed - dataset_elapsed - pool_elapsed) <= 0.01
    )
    memory_measurements_consistent = (
        type(before_available_memory_bytes) is int
        and type(after_available_memory_bytes) is int
        and type(minimum_available_memory_bytes) is int
        and before_available_memory_bytes > 0
        and after_available_memory_bytes > 0
        and minimum_available_memory_bytes > 0
        and minimum_available_memory_bytes <= before_available_memory_bytes
        and minimum_available_memory_bytes <= after_available_memory_bytes
    )
    equal_reserve = (
        valid_rss
        and memory_measurements_consistent
        and minimum_available_memory_bytes >= maximum_rss * MIB
    )
    no_swap_dependence = process_major_faults == 0 and swapout_delta_pages == 0
    return {
        "checkout_unchanged_passed": checkout_unchanged,
        "execution_environment_unchanged_passed": (execution_environment_unchanged),
        "exact_shape_passed": shape_passed,
        "benchmark_metrics_consistent_passed": benchmark_metrics_consistent,
        "memory_measurements_consistent_passed": memory_measurements_consistent,
        "equal_physical_reserve_passed": equal_reserve,
        "no_swap_dependence_observed": no_swap_dependence,
        "passed": (
            checkout_unchanged
            and execution_environment_unchanged
            and shape_passed
            and benchmark_metrics_consistent
            and memory_measurements_consistent
            and equal_reserve
            and no_swap_dependence
        ),
    }


def _sanitized_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _terminate_and_reap(
    process: subprocess.Popen[str],
    *,
    process_group: int,
) -> None:
    if (
        isinstance(process_group, bool)
        or process_group <= 0
        or process_group != process.pid
    ):
        raise ValueError("detached process group must equal its positive leader PID")
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
        process.wait(timeout=10)
        if not _wait_for_process_group_exit(
            process_group,
            timeout_seconds=1.0,
        ):
            raise RuntimeError("detached process group survived SIGKILL") from None
        return
    if not _wait_for_process_group_exit(process_group, timeout_seconds=1.0):
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
        if not _wait_for_process_group_exit(
            process_group,
            timeout_seconds=1.0,
        ):
            raise RuntimeError("detached process group survived cleanup")


def _run_bounded_detached_json(
    arguments: Sequence[str],
    *,
    cwd: Path,
    storage: Path,
    timeout_seconds: float,
    label: str,
    pass_fds: Sequence[int] = (),
) -> dict[str, Any]:
    timeout = _expect_number(
        timeout_seconds,
        label=f"{label} timeout",
        minimum=0.0,
        strict_minimum=True,
    )
    started = time.monotonic()
    timed_out = False
    output_limit_exceeded = False
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=storage) as stdout,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=storage) as stderr,
    ):
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            env=_sanitized_environment(),
            text=True,
            start_new_session=True,
            pass_fds=tuple(pass_fds),
        )
        process_group = process.pid
        try:
            while process.poll() is None:
                if (
                    os.fstat(stdout.fileno()).st_size > MIB
                    or os.fstat(stderr.fileno()).st_size > MIB
                ):
                    output_limit_exceeded = True
                    break
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    timed_out = True
                    break
                time.sleep(min(0.25, timeout - elapsed))
        finally:
            _terminate_and_reap(process, process_group=process_group)
        stdout.seek(0, os.SEEK_END)
        stdout_size = stdout.tell()
        stdout.seek(0)
        output = stdout.read(MIB + 1)
        stderr.seek(0, os.SEEK_END)
        stderr_size = stderr.tell()
        stderr.seek(max(0, stderr_size - 4000))
        error_excerpt = stderr.read()
    if (
        process.returncode != 0
        or timed_out
        or output_limit_exceeded
        or stdout_size > MIB
        or stderr_size > MIB
    ):
        detail = error_excerpt[-4000:] if error_excerpt else "no stderr"
        raise ValueError(f"{label} failed or exceeded its resource limits: {detail}")
    parsed = parse_json_strict(output, label=f"{label} output")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return parsed


def _snapshot_matches_host(
    snapshot: Mapping[str, Any],
    host: Mapping[str, Any],
) -> bool:
    return all(
        (
            snapshot["commit"] == host["execution"]["expected_commit"],
            snapshot["clean"] is True,
            snapshot["uv_lock_sha256"] == host["execution"]["uv_lock_sha256"],
            snapshot["execution_python_sha256"]
            == host["runtime"]["execution_python_sha256"],
        )
    )


def _capacity_execution_identity(host: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commit": host["execution"]["observed_commit"],
        "repository": host["execution"]["repository"],
        "git_directory": host["execution"]["git_directory"],
        "git_path": host["runtime"]["git_path"],
        "git_sha256": host["runtime"]["git_sha256"],
        "uv_lock_sha256": host["execution"]["uv_lock_sha256"],
        "uv_path": host["runtime"]["uv_path"],
        "uv_sha256": host["runtime"]["uv_sha256"],
        "execution_python_path": host["runtime"]["execution_python_path"],
        "execution_python_sha256": host["runtime"]["execution_python_sha256"],
        "execution_environment_hash": host["runtime"]["execution_environment_hash"],
    }


def _current_uv_matches_host(host: Mapping[str, Any]) -> bool:
    path = Path(host["runtime"]["uv_path"])
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        return False
    return _sha256_file(path.resolve(strict=True)) == host["runtime"]["uv_sha256"]


def _current_git_matches_host(host: Mapping[str, Any]) -> bool:
    path = Path(host["runtime"]["git_path"])
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        return False
    return _sha256_file(path.resolve(strict=True)) == host["runtime"]["git_sha256"]


def _current_execution_python_matches_host(host: Mapping[str, Any]) -> bool:
    path = Path(host["runtime"]["execution_python_path"])
    expected_path = Path(host["execution"]["repository"]) / ".venv" / "bin" / "python"
    if (
        path != expected_path
        or not path.is_absolute()
        or not path.is_file()
        or not os.access(path, os.X_OK)
    ):
        return False
    try:
        resolved = path.resolve(strict=True)
        observed_version = _run_text(
            (
                str(path),
                "-c",
                "import platform; print(platform.python_version())",
            ),
            cwd=Path(host["execution"]["repository"]),
        )
        observed_prefix = _run_text(
            (
                str(path),
                "-c",
                "import sys; print(sys.prefix)",
            ),
            cwd=Path(host["execution"]["repository"]),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return (
        resolved.is_file()
        and os.access(resolved, os.X_OK)
        and _sha256_file(resolved) == host["runtime"]["execution_python_sha256"]
        and observed_version == host["runtime"]["execution_python"]
        and observed_prefix == host["runtime"]["execution_prefix"]
    )


def _current_execution_environment_hash(
    host: Mapping[str, Any],
) -> str | None:
    try:
        observed = _execution_environment_identity(
            Path(host["runtime"]["execution_prefix"]),
            python_version=host["runtime"]["execution_python"],
        )
    except (OSError, TypeError, ValueError):
        return None
    return _execution_environment_hash(observed)


def _record_is_fresh(record: Mapping[str, Any]) -> bool:
    timestamp = _parse_timestamp(record["recorded_at"], label="recorded_at")
    now = datetime.now(UTC)
    return now - MAXIMUM_RECORD_AGE <= timestamp <= now + MAXIMUM_CLOCK_SKEW


def run_capacity_benchmark(
    *,
    stage: str,
    static_record: Mapping[str, Any],
    acknowledge_e768: bool = False,
    sample_interval_seconds: float = 0.25,
    timeout_seconds: float = DEFAULT_CAPACITY_TIMEOUT_SECONDS,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Run only one synthetic capacity benchmark after static gates pass."""

    host = verify_record(dict(static_record), record_type=_STATIC_RECORD_TYPE)
    if stage not in _CAPACITY_SPECS:
        raise ValueError("capacity stage must be e192 or e768")
    if not isinstance(acknowledge_e768, bool):
        raise ValueError("E768 acknowledgement must be a boolean")
    if not host["decision"]["static_prerequisites_passed"]:
        raise ValueError("static host prerequisites did not pass")
    if not _record_is_fresh(host):
        raise ValueError("static host qualification is stale or future-dated")
    if stage == "e768" and not acknowledge_e768:
        raise ValueError(
            "E768 requires explicit synthetic memory-pressure acknowledgement"
        )
    interval = _expect_number(
        sample_interval_seconds,
        label="sample interval",
        minimum=0.0,
        strict_minimum=True,
    )
    if interval > MAXIMUM_SAMPLE_INTERVAL_SECONDS:
        raise ValueError(
            "sample interval must be at most "
            f"{MAXIMUM_SAMPLE_INTERVAL_SECONDS:g} second"
        )
    timeout = _expect_number(
        timeout_seconds,
        label="timeout",
        minimum=0.0,
        strict_minimum=True,
    )

    repository = Path(host["execution"]["repository"])
    lock = repository / "uv.lock"
    execution_python = Path(host["runtime"]["execution_python_path"])
    before_snapshot = _repository_snapshot(
        repository,
        lock=lock,
        execution_python=execution_python,
        git_path=host["runtime"]["git_path"],
        git_directory=Path(host["execution"]["git_directory"]),
    )
    if not _snapshot_matches_host(before_snapshot, host):
        raise ValueError("execution checkout changed after static inspection")
    if not _current_uv_matches_host(host):
        raise ValueError("uv executable changed after static inspection")
    if not _current_git_matches_host(host):
        raise ValueError("git executable changed after static inspection")
    if not _current_execution_python_matches_host(host):
        raise ValueError("execution Python changed after static inspection")
    prelaunch_environment_hash = _current_execution_environment_hash(host)
    if prelaunch_environment_hash != host["runtime"][
        "execution_environment_hash"
    ] or not _current_execution_environment_matches_host(host):
        raise ValueError("execution environment changed after static inspection")
    if (
        _tool_identity_hash(_tool_identity(git_path=host["runtime"]["git_path"]))
        != host["tool_identity_hash"]
    ):
        raise ValueError("qualification tool source changed after static inspection")
    if (
        _machine_boot_identity(
            proc_root=proc_root,
            machine_id_path=machine_id_path,
        )
        != host["host_identity"]
    ):
        raise ValueError("host machine or boot changed after static inspection")
    if not _current_host_resources_match(host, proc_root=proc_root):
        raise ValueError("host memory or qualified storage changed after inspection")
    command = _capacity_arguments(
        stage,
        uv_path=host["runtime"]["uv_path"],
        execution_python=host["runtime"]["execution_python_path"],
    )
    meminfo = proc_root / "meminfo"
    vmstat_path = proc_root / "vmstat"
    before_memory = _read_key_values(meminfo)
    before_vmstat = _read_key_values(vmstat_path)
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_at = _utc_now()
    started = time.monotonic()
    minimum_available = before_memory.get("MemAvailable", 0)
    timed_out = False
    output_limit_exceeded = False
    storage = Path(host["storage"]["storage_root"])
    with (
        tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            dir=storage,
        ) as stdout_file,
        tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            dir=storage,
        ) as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=stdout_file,
            stderr=stderr_file,
            env=_sanitized_environment(),
            text=True,
            start_new_session=True,
        )
        process_group = process.pid
        try:
            while process.poll() is None:
                current = _read_key_values(meminfo)
                minimum_available = min(
                    minimum_available,
                    current.get("MemAvailable", minimum_available),
                )
                if (
                    os.fstat(stdout_file.fileno()).st_size > MIB
                    or os.fstat(stderr_file.fileno()).st_size > MIB
                ):
                    output_limit_exceeded = True
                    break
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    timed_out = True
                    break
                time.sleep(min(interval, timeout - elapsed))
        finally:
            _terminate_and_reap(process, process_group=process_group)
        stdout_file.seek(0, os.SEEK_END)
        stdout_size = stdout_file.tell()
        output_limit_exceeded = output_limit_exceeded or stdout_size > MIB
        stdout_file.seek(0)
        stdout = stdout_file.read(MIB + 1)
        stderr_file.seek(0, os.SEEK_END)
        stderr_size = stderr_file.tell()
        output_limit_exceeded = output_limit_exceeded or stderr_size > MIB
        stderr_file.seek(max(0, stderr_size - 4000))
        stderr = stderr_file.read()

    after_snapshot = _repository_snapshot(
        repository,
        lock=lock,
        execution_python=execution_python,
        git_path=host["runtime"]["git_path"],
        git_directory=Path(host["execution"]["git_directory"]),
    )
    checkout_unchanged = after_snapshot == before_snapshot and _snapshot_matches_host(
        after_snapshot, host
    )
    postrun_environment_hash = _current_execution_environment_hash(host)
    execution_environment_unchanged = (
        postrun_environment_hash
        == prelaunch_environment_hash
        == host["runtime"]["execution_environment_hash"]
        and _current_execution_environment_matches_host(host)
    )
    after_memory = _read_key_values(meminfo)
    after_vmstat = _read_key_values(vmstat_path)
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    minimum_available = min(
        minimum_available,
        after_memory.get("MemAvailable", minimum_available),
    )
    process_major_faults = max(0, after_usage.ru_majflt - before_usage.ru_majflt)
    swapin_delta = max(
        0,
        after_vmstat.get("pswpin", 0) - before_vmstat.get("pswpin", 0),
    )
    swapout_delta = max(
        0,
        after_vmstat.get("pswpout", 0) - before_vmstat.get("pswpout", 0),
    )
    result: dict[str, Any] = {}
    if process.returncode == 0 and not timed_out and not output_limit_exceeded:
        try:
            parsed = parse_json_strict(
                stdout,
                label=f"symbolic v2 {stage.upper()} capacity output",
            )
        except ValueError as error:
            raise ValueError(
                "capacity benchmark did not emit one JSON object"
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError("capacity benchmark output must be a JSON object")
        result = parsed
    decision = _capacity_decision(
        stage,
        result=result,
        before_available_memory_bytes=before_memory.get("MemAvailable", 0),
        after_available_memory_bytes=after_memory.get("MemAvailable", 0),
        minimum_available_memory_bytes=minimum_available,
        process_major_faults=process_major_faults,
        swapout_delta_pages=swapout_delta,
        checkout_unchanged=checkout_unchanged,
        execution_environment_unchanged=execution_environment_unchanged,
    )
    page_size = os.sysconf("SC_PAGE_SIZE")
    execution_identity = _capacity_execution_identity(host)
    payload = {
        "schema_version": 1,
        "record_type": _CAPACITY_RECORD_TYPE,
        "recorded_at": _utc_now(),
        "stage": stage,
        "execution": execution_identity,
        "host_identity": host["host_identity"],
        "host_identity_hash": host["host_identity_hash"],
        "tool_identity_hash": host["tool_identity_hash"],
        "static_record_hash": host["record_hash"],
        "command": _command(command),
        "started_at": started_at,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timeout_seconds": timeout,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "output_limit_exceeded": output_limit_exceeded,
        "benchmark_result": result,
        "prelaunch_checkout": before_snapshot,
        "postrun_checkout": after_snapshot,
        "prelaunch_execution_environment_hash": prelaunch_environment_hash,
        "postrun_execution_environment_hash": postrun_environment_hash,
        "host_memory": {
            "before_available_memory_bytes": before_memory.get("MemAvailable", 0),
            "after_available_memory_bytes": after_memory.get("MemAvailable", 0),
            "minimum_available_memory_bytes": minimum_available,
        },
        "host_swap": {
            "page_size_bytes": page_size,
            "pswpin_delta_pages": swapin_delta,
            "pswpin_delta_bytes": swapin_delta * page_size,
            "pswpout_delta_pages": swapout_delta,
            "pswpout_delta_bytes": swapout_delta * page_size,
        },
        "process_major_faults": process_major_faults,
        "stderr": stderr[-4000:] if stderr else "",
        "decision": decision,
        "scientific_boundary": {
            "synthetic_dataset_only": True,
            "creates_study_artifacts": False,
            "executes_registered_study": False,
            "scientific_use": "prohibited",
        },
    }
    return _signed(payload)


_PROBE_EXECUTION_RESULT_KEYS = frozenset(
    {
        "artifact_root",
        "config_hash",
        "phase",
        "run_count",
    }
)
_PROBE_BENCHMARK_RESULT_KEYS = frozenset(
    {
        "artifact_root",
        "budget_multiplier",
        "dataset_hash",
        "fixed_frontier_cache_copy_seconds",
        "fixed_frontier_raw_hash_seconds",
        "fixed_frontier_validation_seconds",
        "fixed_report_ingestion_seconds",
        "frontier_tree_count",
        "inventory_elapsed_seconds",
        "load_elapsed_seconds",
        "maximum_rss_mib",
        "modeled_probe_report_seconds",
        "marginal_run_load_seconds_per_run",
        "marginal_run_raw_hash_seconds_per_run",
        "marginal_run_validation_seconds_per_run",
        "observation_count",
        "observed_probe_report_seconds",
        "operational_budget_hours",
        "projected_report_ingestion_hours",
        "projected_runs_per_root",
        "raw_run_byte_size",
        "raw_run_file_count",
        "residual_seconds_per_run",
        "rss_before_mib",
        "rss_increment_upper_bound_mib",
        "run_count",
    }
)


def _validate_probe_execution_result(value: object) -> dict[str, Any]:
    result = _expect_keys(
        value,
        label="probe execution result",
        keys=_PROBE_EXECUTION_RESULT_KEYS,
    )
    root = Path(_expect_string(result["artifact_root"], label="probe artifact_root"))
    if not root.is_absolute():
        raise ValueError("probe artifact_root must be absolute")
    _expect_sha256(result["config_hash"], label="probe config_hash")
    _expect_string(result["phase"], label="probe phase")
    _expect_int(result["run_count"], label="probe run_count", minimum=0)
    return result


def _validate_probe_benchmark_result(value: object) -> dict[str, Any]:
    result = _expect_keys(
        value,
        label="probe benchmark result",
        keys=_PROBE_BENCHMARK_RESULT_KEYS,
    )
    root = Path(_expect_string(result["artifact_root"], label="probe artifact_root"))
    if not root.is_absolute():
        raise ValueError("probe artifact_root must be absolute")
    _expect_sha256(result["dataset_hash"], label="probe dataset_hash")
    for name in (
        "frontier_tree_count",
        "observation_count",
        "projected_runs_per_root",
        "raw_run_byte_size",
        "raw_run_file_count",
        "run_count",
    ):
        _expect_int(result[name], label=f"probe {name}", minimum=0)
    for name in _PROBE_BENCHMARK_RESULT_KEYS - {
        "artifact_root",
        "dataset_hash",
        "frontier_tree_count",
        "observation_count",
        "projected_runs_per_root",
        "raw_run_byte_size",
        "raw_run_file_count",
        "run_count",
    }:
        _expect_number(result[name], label=f"probe {name}", minimum=0.0)
    return result


def _probe_execution_shape(
    result: Mapping[str, Any],
    expected_artifact_root: str,
) -> bool:
    return all(
        (
            result["artifact_root"] == expected_artifact_root,
            result["config_hash"] == V2_PROBE_CONFIG_HASH,
            result["phase"] == "calibration",
            type(result["run_count"]) is int,
            result["run_count"] == 48,
        )
    )


def _probe_projection(result: Mapping[str, Any]) -> dict[str, float] | None:
    probe_runs = result.get("run_count")
    projected_runs = result.get("projected_runs_per_root")
    if (
        type(probe_runs) is not int
        or probe_runs <= 0
        or type(projected_runs) is not int
        or projected_runs <= 0
    ):
        return None
    try:
        frontier_validation = float(result["fixed_frontier_validation_seconds"])
        frontier_copy = float(result["fixed_frontier_cache_copy_seconds"])
        frontier_raw = float(result["fixed_frontier_raw_hash_seconds"])
        run_validation = float(result["marginal_run_validation_seconds_per_run"])
        run_raw = float(result["marginal_run_raw_hash_seconds_per_run"])
        run_load = float(result["marginal_run_load_seconds_per_run"])
        observed_probe = float(result["observed_probe_report_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    components = (
        frontier_validation,
        frontier_copy,
        frontier_raw,
        run_validation,
        run_raw,
        run_load,
        observed_probe,
    )
    if any(not math.isfinite(value) or value < 0 for value in components):
        return None
    fixed = 7.0 * frontier_validation + 3.0 * frontier_copy + 2.0 * frontier_raw
    marginal = 4.0 * run_validation + 2.0 * run_raw + run_load
    modeled_probe = fixed + probe_runs * marginal
    residual = max(0.0, (observed_probe - modeled_probe) / probe_runs)
    projected_seconds = fixed + projected_runs * (marginal + residual)
    return {
        "fixed_seconds": fixed,
        "modeled_probe_seconds": modeled_probe,
        "residual_seconds_per_run": residual,
        "projected_hours": projected_seconds / 3600,
        "operational_budget_hours": (
            projected_seconds * float(result.get("budget_multiplier", math.nan)) / 3600
        ),
    }


def _probe_benchmark_metrics_consistent(result: Mapping[str, Any]) -> bool:
    projection = _probe_projection(result)
    if projection is None:
        return False
    try:
        inventory = float(result["inventory_elapsed_seconds"])
        load = float(result["load_elapsed_seconds"])
        observed = float(result["observed_probe_report_seconds"])
        maximum_rss = float(result["maximum_rss_mib"])
        rss_before = float(result["rss_before_mib"])
        rss_increment = float(result["rss_increment_upper_bound_mib"])
        raw_bytes = result["raw_run_byte_size"]
    except (KeyError, TypeError, ValueError):
        return False
    values = (
        inventory,
        load,
        observed,
        maximum_rss,
        rss_before,
        rss_increment,
    )
    return (
        all(math.isfinite(value) and value >= 0 for value in values)
        and maximum_rss > 0
        and 0 < rss_before <= maximum_rss
        and abs(rss_increment - max(0.0, maximum_rss - rss_before)) <= 0.002
        and type(raw_bytes) is int
        and raw_bytes > 0
        and abs(observed - (2.0 * inventory + load)) <= 0.000005
        and abs(
            float(result["fixed_report_ingestion_seconds"])
            - projection["fixed_seconds"]
        )
        <= 0.00001
        and abs(
            float(result["modeled_probe_report_seconds"])
            - projection["modeled_probe_seconds"]
        )
        <= 0.00025
        and abs(
            float(result["residual_seconds_per_run"])
            - projection["residual_seconds_per_run"]
        )
        <= 0.00001
        and abs(
            float(result["projected_report_ingestion_hours"])
            - projection["projected_hours"]
        )
        <= 0.002
        and abs(
            float(result["operational_budget_hours"])
            - projection["operational_budget_hours"]
        )
        <= 0.002
    )


def _probe_benchmark_shape(
    result: Mapping[str, Any],
    expected_artifact_root: str,
) -> bool:
    expected_integers = {
        "frontier_tree_count": 4,
        "observation_count": 624,
        "projected_runs_per_root": V2_PROJECTED_RUNS,
        "raw_run_file_count": V2_PROBE_RAW_RUN_FILES,
        "run_count": 48,
    }
    return all(
        type(result[name]) is int and result[name] == expected
        for name, expected in expected_integers.items()
    ) and all(
        (
            result["artifact_root"] == expected_artifact_root,
            result["dataset_hash"] == V2_PROBE_DATASET_HASH,
            result["budget_multiplier"] == 2.0,
            _probe_benchmark_metrics_consistent(result),
        )
    )


def _assert_no_symlink_components(path: Path) -> Path:
    target = _absolute_path(path)
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"path contains a symlinked component: {current}")
    return target.resolve(strict=True)


def _artifact_storage_identity(
    artifact_root: Path,
    *,
    storage_root: Path,
) -> dict[str, Any]:
    canonical = _assert_no_symlink_components(artifact_root)
    storage = _assert_no_symlink_components(storage_root)
    if not _path_within(canonical, storage):
        raise ValueError("probe artifact root is outside qualified storage")
    metadata = canonical.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("probe artifact root must be a directory")
    return {
        "canonical_path": str(canonical),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, MIB):
        digest.update(chunk)
    return digest.hexdigest()


def _stat_stability_token(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _raise_walk_error(error: OSError) -> None:
    raise error


def _build_artifact_manifest(
    root: Path,
    *,
    base_directory_fd: int | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    for directory, directory_names, file_names, current_directory_fd in os.fwalk(
        root,
        topdown=True,
        onerror=_raise_walk_error,
        follow_symlinks=False,
        dir_fd=base_directory_fd,
    ):
        directory_names.sort()
        file_names.sort()
        relative_directory = Path(directory).relative_to(root)
        for name in directory_names:
            metadata = os.stat(
                name,
                dir_fd=current_directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("probe artifact tree contains a non-directory entry")
            entries.append(
                {
                    "kind": "directory",
                    "path": (relative_directory / name).as_posix(),
                }
            )
        for name in file_names:
            metadata = os.stat(
                name,
                dir_fd=current_directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("probe artifact tree contains a non-regular file")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_directory_fd,
            )
            try:
                opened = os.fstat(descriptor)
                if _stat_stability_token(opened) != _stat_stability_token(metadata):
                    raise ValueError("probe artifact changed while building manifest")
                file_hash = _sha256_descriptor(descriptor)
                after = os.fstat(descriptor)
                if _stat_stability_token(after) != _stat_stability_token(opened):
                    raise ValueError("probe artifact changed while being hashed")
            finally:
                os.close(descriptor)
            relative = (relative_directory / name).as_posix()
            entries.append(
                {
                    "kind": "file",
                    "path": relative,
                    "sha256": file_hash,
                    "size_bytes": metadata.st_size,
                }
            )
            file_count += 1
            total_bytes += metadata.st_size
    return {
        "entries": entries,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "manifest_hash": scientific_hash(
            entries,
            domain="operations.v2-probe-artifact-manifest.v1",
        ),
    }


def _artifact_manifest(artifact_root: Path) -> dict[str, Any]:
    return _stable_manifest_at_path(_assert_no_symlink_components(artifact_root))


def _artifact_manifest_from_descriptor(directory_fd: int) -> dict[str, Any]:
    return _build_artifact_manifest(Path("."), base_directory_fd=directory_fd)


def _directory_stability_state_from_descriptor(
    directory_fd: int,
) -> tuple[tuple[object, ...], ...]:
    state: list[tuple[object, ...]] = []
    root = Path(".")
    for directory, directory_names, file_names, current_fd in os.fwalk(
        root,
        topdown=True,
        onerror=_raise_walk_error,
        follow_symlinks=False,
        dir_fd=directory_fd,
    ):
        directory_names.sort()
        file_names.sort()
        metadata = os.fstat(current_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("manifest traversal left the open directory tree")
        state.append(
            (
                "directory",
                Path(directory).relative_to(root).as_posix(),
                *_stat_stability_token(metadata),
                tuple(directory_names),
                tuple(file_names),
            )
        )
        relative_directory = Path(directory).relative_to(root)
        for name in file_names:
            metadata = os.stat(
                name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("manifest stability tree contains a non-regular file")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            try:
                opened = os.fstat(descriptor)
                if _stat_stability_token(opened) != _stat_stability_token(metadata):
                    raise ValueError(
                        "manifest entry changed during stability verification"
                    )
            finally:
                os.close(descriptor)
            state.append(
                (
                    "file",
                    (relative_directory / name).as_posix(),
                    *_stat_stability_token(metadata),
                )
            )
    return tuple(state)


def _stable_manifest_and_seal_from_descriptor(
    directory_fd: int,
) -> tuple[dict[str, Any], str]:
    state_before = _directory_stability_state_from_descriptor(directory_fd)
    manifest = _artifact_manifest_from_descriptor(directory_fd)
    repeated = _artifact_manifest_from_descriptor(directory_fd)
    state_after = _directory_stability_state_from_descriptor(directory_fd)
    if manifest != repeated or state_before != state_after:
        raise ValueError("directory tree changed during manifest verification")
    seal = scientific_hash(
        {
            "manifest_hash": manifest["manifest_hash"],
            "stability_state": state_after,
        },
        domain="operations.stable-directory-tree.v1",
    )
    return manifest, seal


def _stable_manifest_from_descriptor(directory_fd: int) -> dict[str, Any]:
    return _stable_manifest_and_seal_from_descriptor(directory_fd)[0]


def _open_existing_directory_nofollow(path: Path) -> int:
    target = _absolute_path(path)
    descriptor = os.open(
        target.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        for component in target.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_manifest_and_seal_at_path(
    path: Path,
) -> tuple[dict[str, Any], str]:
    target = _absolute_path(path)
    descriptor = _open_existing_directory_nofollow(target)
    verification_descriptor: int | None = None
    try:
        manifest, seal = _stable_manifest_and_seal_from_descriptor(descriptor)
        verification_descriptor = _open_existing_directory_nofollow(target)
        if not _same_open_directory(descriptor, verification_descriptor):
            raise ValueError("directory root changed during manifest verification")
        repeated, repeated_seal = _stable_manifest_and_seal_from_descriptor(
            verification_descriptor
        )
        if repeated != manifest or repeated_seal != seal:
            raise ValueError("directory tree changed during manifest verification")
        return manifest, seal
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        os.close(descriptor)


def _stable_manifest_at_path(path: Path) -> dict[str, Any]:
    return _stable_manifest_and_seal_at_path(path)[0]


def _stable_directory_tree_snapshot(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _absolute_path(path)
    descriptor = _open_existing_directory_nofollow(target)
    verification_descriptor: int | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"stable tree root must be a directory: {target}")
        manifest, stability_hash = _stable_manifest_and_seal_from_descriptor(descriptor)
        verification_descriptor = _open_existing_directory_nofollow(target)
        if not _same_open_directory(descriptor, verification_descriptor):
            raise ValueError("directory root changed during stable tree verification")
        repeated, repeated_stability_hash = _stable_manifest_and_seal_from_descriptor(
            verification_descriptor
        )
        if repeated != manifest or repeated_stability_hash != stability_hash:
            raise ValueError("directory tree changed during stable tree verification")
        final_metadata = os.fstat(verification_descriptor)
        if _stat_stability_token(final_metadata) != _stat_stability_token(metadata):
            raise ValueError("directory root changed during stable tree verification")
        directory_count = 1 + sum(
            entry["kind"] == "directory" for entry in manifest["entries"]
        )
        identity = {
            "path": str(target),
            "device": final_metadata.st_dev,
            "inode": final_metadata.st_ino,
            "mode": final_metadata.st_mode,
            "link_count": final_metadata.st_nlink,
            "file_count": manifest["file_count"],
            "directory_count": directory_count,
            "total_bytes": manifest["total_bytes"],
            "manifest_hash": manifest["manifest_hash"],
            "stability_hash": stability_hash,
        }
        return identity, manifest
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        os.close(descriptor)


def _stable_directory_tree_identity(path: Path) -> dict[str, Any]:
    return _stable_directory_tree_snapshot(path)[0]


def _stable_import_target_identity(
    path: Path,
    *,
    reject_bytecode: bool,
) -> dict[str, Any]:
    identity, manifest = _stable_directory_tree_snapshot(path)
    if reject_bytecode and any(
        "__pycache__" in Path(entry["path"]).parts
        or Path(entry["path"]).suffix == ".pyc"
        for entry in manifest["entries"]
    ):
        raise ValueError(
            "execution import path contains unbound Python bytecode caches"
        )
    return identity


def _read_descriptor_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, MIB):
        chunks.append(chunk)
    return b"".join(chunks)


def _stable_file_bytes_and_identity(path: Path) -> tuple[bytes, dict[str, Any]]:
    target = _absolute_path(path)
    directory_fd = _open_existing_directory_nofollow(target.parent)
    verification_fd: int | None = None
    descriptor: int | None = None
    repeated_descriptor: int | None = None
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"stable file must be regular: {target}")
        content = _read_descriptor_bytes(descriptor)
        after = os.fstat(descriptor)
        if _stat_stability_token(before) != _stat_stability_token(after):
            raise ValueError(f"file changed while being read: {target}")
        verification_fd = _open_existing_directory_nofollow(target.parent)
        if not _same_open_directory(directory_fd, verification_fd):
            raise ValueError("file parent changed during stable verification")
        repeated_descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=verification_fd,
        )
        repeated_metadata = os.fstat(repeated_descriptor)
        repeated_content = _read_descriptor_bytes(repeated_descriptor)
        repeated_after = os.fstat(repeated_descriptor)
        if (
            _stat_stability_token(before) != _stat_stability_token(repeated_metadata)
            or _stat_stability_token(repeated_metadata)
            != _stat_stability_token(repeated_after)
            or content != repeated_content
        ):
            raise ValueError(f"file changed during stable verification: {target}")
        return content, {
            "path": str(target),
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode": before.st_mode,
            "link_count": before.st_nlink,
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    finally:
        if repeated_descriptor is not None:
            os.close(repeated_descriptor)
        if verification_fd is not None:
            os.close(verification_fd)
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _current_artifact_evidence(
    artifact_root: Path,
    *,
    storage_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    artifact = _absolute_path(artifact_root)
    storage = _absolute_path(storage_root)
    if (
        str(artifact) != str(artifact_root)
        or str(storage) != str(storage_root)
        or not _path_within(artifact, storage)
    ):
        raise ValueError("probe artifact path is not normalized qualified storage")
    descriptor = _open_existing_directory_nofollow(artifact)
    verification_descriptor: int | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("probe artifact root must be a directory")
        identity = {
            "canonical_path": str(artifact),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
        manifest, stability_hash = _stable_manifest_and_seal_from_descriptor(descriptor)
        verification_descriptor = _open_existing_directory_nofollow(artifact)
        if not _same_open_directory(descriptor, verification_descriptor):
            raise ValueError("probe artifact root changed during verification")
        verification_manifest, verification_stability_hash = (
            _stable_manifest_and_seal_from_descriptor(verification_descriptor)
        )
        if (
            verification_manifest != manifest
            or verification_stability_hash != stability_hash
        ):
            raise ValueError("probe artifact tree changed during verification")
        return identity, manifest, stability_hash
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        os.close(descriptor)


def _open_probe_parent(
    artifact_root: Path,
    *,
    storage_root: Path,
) -> tuple[int, str]:
    artifact = _absolute_path(artifact_root)
    storage = _absolute_path(storage_root)
    try:
        relative = artifact.relative_to(storage)
    except ValueError as error:
        raise ValueError("probe artifact root is outside qualified storage") from error
    if not relative.parts:
        raise ValueError("probe artifact root cannot equal qualified storage root")
    descriptor = _open_existing_directory_nofollow(storage)
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        leaf = relative.parts[-1]
        try:
            os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return descriptor, leaf
        raise ValueError("probe artifact root must not already exist")
    except BaseException:
        os.close(descriptor)
        raise


def _same_open_directory(left: int, right: int) -> bool:
    left_stat = os.fstat(left)
    right_stat = os.fstat(right)
    return (
        stat.S_ISDIR(left_stat.st_mode)
        and stat.S_ISDIR(right_stat.st_mode)
        and (left_stat.st_dev, left_stat.st_ino)
        == (right_stat.st_dev, right_stat.st_ino)
    )


def _execution_environment_identity(
    prefix: Path,
    *,
    python_version: str,
    reject_import_bytecode: bool = True,
) -> dict[str, Any]:
    canonical_prefix = _assert_no_symlink_components(prefix)
    version_parts = python_version.split(".")
    if (
        len(version_parts) < 2
        or not version_parts[0].isdigit()
        or not version_parts[1].isdigit()
    ):
        raise ValueError("execution Python version must include major and minor")
    python_directory = f"python{version_parts[0]}.{version_parts[1]}"
    config_content, config_identity = _stable_file_bytes_and_identity(
        canonical_prefix / "pyvenv.cfg"
    )
    try:
        config_text = config_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pyvenv.cfg must be UTF-8") from error
    config_fields: dict[str, str] = {}
    for line in config_text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        normalized_key = key.strip().lower()
        if not separator or not normalized_key or normalized_key in config_fields:
            raise ValueError("pyvenv.cfg fields must be unique key/value pairs")
        config_fields[normalized_key] = value.strip()
    include_system = config_fields.get("include-system-site-packages", "").lower()
    if include_system not in {"true", "false"}:
        raise ValueError("pyvenv.cfg must declare include-system-site-packages")
    site_packages_path = canonical_prefix / "lib" / python_directory / "site-packages"
    tree_before = _stable_directory_tree_identity(site_packages_path)
    pth_path_entries: set[str] = set()
    pth_executable_line_hashes: set[str] = set()
    for candidate in sorted(site_packages_path.iterdir(), key=lambda item: item.name):
        if candidate.suffix != ".pth":
            continue
        pth_content, _ = _stable_file_bytes_and_identity(candidate)
        try:
            pth_text = pth_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"site path file must be UTF-8: {candidate}") from error
        for line in pth_text.splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if entry.startswith(("import ", "import\t")):
                line_hash = hashlib.sha256(entry.encode("utf-8")).hexdigest()
                if line_hash not in _ALLOWED_PTH_EXECUTABLE_LINE_HASHES:
                    raise ValueError(
                        "execution environment contains an unapproved "
                        "executable .pth line"
                    )
                pth_executable_line_hashes.add(line_hash)
                continue
            pth_path_entries.add(str(_absolute_path(site_packages_path / entry)))
    tree_after = _stable_directory_tree_identity(site_packages_path)
    if tree_after != tree_before:
        raise ValueError(
            "execution site-packages changed during environment inspection"
        )
    pth_target_identities = [
        _stable_import_target_identity(
            Path(path),
            reject_bytecode=reject_import_bytecode,
        )
        for path in sorted(pth_path_entries)
        if not _path_within(Path(path), site_packages_path)
    ]
    target_by_path = {target["path"]: target for target in pth_target_identities}
    checkout_import_trees = []
    for path in (
        canonical_prefix.parent / "src",
        canonical_prefix.parent / "scripts",
    ):
        target = target_by_path.get(str(path))
        checkout_import_trees.append(
            target
            if target is not None
            else _stable_import_target_identity(
                path,
                reject_bytecode=reject_import_bytecode,
            )
        )
    final_tree = _stable_directory_tree_identity(site_packages_path)
    if final_tree != tree_before:
        raise ValueError(
            "execution site-packages changed during environment inspection"
        )
    repeated_config_content, repeated_config_identity = _stable_file_bytes_and_identity(
        canonical_prefix / "pyvenv.cfg"
    )
    if (
        repeated_config_content != config_content
        or repeated_config_identity != config_identity
    ):
        raise ValueError("pyvenv.cfg changed during execution environment inspection")
    repeated_targets = [
        _stable_import_target_identity(
            Path(target["path"]),
            reject_bytecode=reject_import_bytecode,
        )
        for target in pth_target_identities
    ]
    if repeated_targets != pth_target_identities:
        raise ValueError("execution import path changed during environment inspection")
    repeated_checkout_trees = [
        _stable_import_target_identity(
            Path(target["path"]),
            reject_bytecode=reject_import_bytecode,
        )
        for target in checkout_import_trees
    ]
    if repeated_checkout_trees != checkout_import_trees:
        raise ValueError("checkout import tree changed during environment inspection")
    return {
        "prefix": str(canonical_prefix),
        "python_version": python_version,
        "pyvenv_config": config_identity,
        "includes_system_site_packages": include_system == "true",
        "site_packages": final_tree,
        "pth_path_entries": sorted(pth_path_entries),
        "pth_target_identities": pth_target_identities,
        "checkout_import_trees": checkout_import_trees,
        "pth_executable_line_hashes": sorted(pth_executable_line_hashes),
    }


def _execution_environment_hash(identity: Mapping[str, Any]) -> str:
    return scientific_hash(
        identity,
        domain="operations.host-execution-environment.v1",
    )


def _current_execution_environment_matches_host(
    host: Mapping[str, Any],
) -> bool:
    expected_identity = host["runtime"].get("execution_environment")
    expected_hash = host["runtime"].get("execution_environment_hash")
    if not isinstance(expected_identity, dict) or not is_sha256(expected_hash):
        return False
    try:
        observed = _execution_environment_identity(
            Path(host["runtime"]["execution_prefix"]),
            python_version=host["runtime"]["execution_python"],
        )
    except (OSError, ValueError):
        return False
    return (
        observed == expected_identity
        and _execution_environment_hash(observed) == expected_hash
    )


def _current_host_resources_match(
    host: Mapping[str, Any],
    *,
    proc_root: Path,
) -> bool:
    storage = Path(host["storage"]["storage_root"])
    descriptor: int | None = None
    verification_descriptor: int | None = None
    try:
        canonical = _absolute_path(storage)
        if str(canonical) != str(storage):
            return False
        descriptor = _open_existing_directory_nofollow(canonical)
        metadata = os.fstat(descriptor)
        mount = _mount_for(canonical, proc_root / "self" / "mountinfo")
        filesystem = os.statvfs(descriptor)
        memory = _read_key_values(proc_root / "meminfo")
        writable = os.access(
            ".",
            os.W_OK | os.X_OK,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        verification_descriptor = _open_existing_directory_nofollow(canonical)
        path_stable = _same_open_directory(descriptor, verification_descriptor)
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        if descriptor is not None:
            os.close(descriptor)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    available_inodes = filesystem.f_favail
    return all(
        (
            stat.S_ISDIR(metadata.st_mode),
            metadata.st_dev == host["storage"]["directory_device_id"],
            metadata.st_ino == host["storage"]["directory_inode"],
            mount["device"] == host["storage"]["device"],
            mount["mount_point"] == host["storage"]["mount_point"],
            mount["type"] == host["storage"]["type"],
            mount["type"] not in _NETWORK_FILESYSTEMS,
            mount["type"] not in _NON_STORAGE_FILESYSTEMS,
            "rw" in mount["mount_options"].split(","),
            available_bytes >= host["storage"]["required_bytes"],
            available_inodes >= host["storage"]["required_inodes"],
            memory.get("MemTotal", 0) == host["host"]["physical_memory_bytes"],
            memory.get("MemTotal", 0) >= MINIMUM_PHYSICAL_MEMORY_BYTES,
            writable,
            path_stable,
        )
    )


def _assert_current_context(
    host: Mapping[str, Any],
    *,
    proc_root: Path,
    machine_id_path: Path,
) -> dict[str, Any]:
    repository = Path(host["execution"]["repository"])
    snapshot = _repository_snapshot(
        repository,
        lock=repository / "uv.lock",
        execution_python=Path(host["runtime"]["execution_python_path"]),
        git_path=host["runtime"]["git_path"],
        git_directory=Path(host["execution"]["git_directory"]),
    )
    if not _snapshot_matches_host(snapshot, host):
        raise ValueError("execution checkout no longer matches static qualification")
    if not _current_uv_matches_host(host):
        raise ValueError("uv executable no longer matches static qualification")
    if not _current_git_matches_host(host):
        raise ValueError("git executable no longer matches static qualification")
    if not _current_execution_python_matches_host(host):
        raise ValueError("execution Python no longer matches static qualification")
    if not _current_execution_environment_matches_host(host):
        raise ValueError("execution environment no longer matches static qualification")
    if (
        _tool_identity_hash(_tool_identity(git_path=host["runtime"]["git_path"]))
        != host["tool_identity_hash"]
    ):
        raise ValueError("qualification tool no longer matches static qualification")
    if (
        _machine_boot_identity(
            proc_root=proc_root,
            machine_id_path=machine_id_path,
        )
        != host["host_identity"]
    ):
        raise ValueError("host machine or boot no longer matches static qualification")
    if not _current_host_resources_match(host, proc_root=proc_root):
        raise ValueError("host memory or qualified storage no longer passes")
    return snapshot


def verify_current_context(
    static_record: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Recheck the checkout, executable, tool, machine, and boot bindings."""

    host = verify_record(dict(static_record), record_type=_STATIC_RECORD_TYPE)
    _assert_current_context(
        host,
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )
    return host


def _bind_probe_result(
    *,
    kind: str,
    static_record: Mapping[str, Any],
    result: Mapping[str, Any],
    probe_execution_record: Mapping[str, Any] | None = None,
    artifact_directory_fd: int | None = None,
    executed_command: str | None = None,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Bind one plain probe output to the qualified execution context."""

    host = verify_record(dict(static_record), record_type=_STATIC_RECORD_TYPE)
    if not host["decision"]["static_prerequisites_passed"]:
        raise ValueError("static host prerequisites did not pass")
    if not _record_is_fresh(host):
        raise ValueError("static host qualification is stale or future-dated")
    _assert_current_context(
        host,
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )
    artifact_root = Path(result.get("artifact_root", ""))
    artifact_identity = _artifact_storage_identity(
        artifact_root,
        storage_root=Path(host["storage"]["storage_root"]),
    )
    if artifact_identity["device"] != host["storage"]["directory_device_id"]:
        raise ValueError("probe artifact root is not on the qualified storage device")
    secure_artifact_access = artifact_directory_fd is not None
    if secure_artifact_access != (executed_command is not None):
        raise ValueError(
            "secure probe binding requires both an open directory and executed command"
        )
    if artifact_directory_fd is None:
        artifact_manifest, artifact_stability_hash = _stable_manifest_and_seal_at_path(
            artifact_root
        )
    else:
        if isinstance(artifact_directory_fd, bool) or artifact_directory_fd < 0:
            raise ValueError("artifact directory descriptor must be nonnegative")
        opened = os.fstat(artifact_directory_fd)
        if not stat.S_ISDIR(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (
            artifact_identity["device"],
            artifact_identity["inode"],
        ):
            raise ValueError("open artifact directory differs from expected root")
        artifact_manifest, artifact_stability_hash = (
            _stable_manifest_and_seal_from_descriptor(artifact_directory_fd)
        )
    common = {
        "schema_version": 1,
        "recorded_at": _utc_now(),
        "execution_commit": host["execution"]["observed_commit"],
        "host_identity": host["host_identity"],
        "host_identity_hash": host["host_identity_hash"],
        "tool_identity_hash": host["tool_identity_hash"],
        "static_record_hash": host["record_hash"],
        "probe_config_hash": host["registered_inputs"]["probe_config_hash"],
        "expected_artifact_root": host["probe_storage"]["artifact_root"],
        "artifact_storage_identity": artifact_identity,
        "artifact_manifest": artifact_manifest,
        "artifact_stability_hash": artifact_stability_hash,
        "secure_artifact_access": secure_artifact_access,
        "executed_command": executed_command,
        "scientific_boundary": {
            "creates_study_artifacts": False,
            "executes_registered_study": False,
            "inspects_probe_metrics": False,
            "scientific_use": "prohibited",
        },
    }
    if kind == "execution":
        checked = _validate_probe_execution_result(dict(result))
        if probe_execution_record is not None:
            raise ValueError("execution probe binding cannot name a prior probe record")
        payload = {
            **common,
            "record_type": _PROBE_EXECUTION_RECORD_TYPE,
            "planned_command": host["plan"]["probe"]["underlying_execute"],
            "source_result_hash": scientific_hash(
                checked,
                domain="operations.v2-probe-execution-result.v1",
            ),
            "result": checked,
            "decision": {
                "shape_passed": _probe_execution_shape(
                    checked,
                    host["probe_storage"]["artifact_root"],
                )
                and artifact_manifest["file_count"] > 0
                and artifact_manifest["total_bytes"] > 0
                and secure_artifact_access,
            },
        }
    elif kind == "benchmark":
        checked = _validate_probe_benchmark_result(dict(result))
        if probe_execution_record is None:
            raise ValueError("benchmark probe binding requires its execution record")
        execution = verify_record(
            dict(probe_execution_record),
            record_type=_PROBE_EXECUTION_RECORD_TYPE,
        )
        for field in (
            "expected_artifact_root",
            "host_identity",
            "host_identity_hash",
            "tool_identity_hash",
            "static_record_hash",
            "probe_config_hash",
            "secure_artifact_access",
        ):
            if execution[field] != common[field]:
                raise ValueError(f"probe execution {field} binding differs")
        if execution["artifact_storage_identity"] != artifact_identity:
            raise ValueError("probe artifact storage identity changed before benchmark")
        if execution["artifact_manifest"] != artifact_manifest:
            raise ValueError("probe artifact content changed before benchmark binding")
        if execution["artifact_stability_hash"] != artifact_stability_hash:
            raise ValueError("probe artifact metadata changed before benchmark binding")
        if execution["result"]["artifact_root"] != checked["artifact_root"]:
            raise ValueError("probe benchmark artifact root differs from execution")
        payload = {
            **common,
            "record_type": _PROBE_BENCHMARK_RECORD_TYPE,
            "planned_command": host["plan"]["probe"]["underlying_benchmark"],
            "probe_execution_record_hash": execution["record_hash"],
            "source_result_hash": scientific_hash(
                checked,
                domain="operations.v2-probe-benchmark-result.v1",
            ),
            "result": checked,
            "decision": {
                "shape_passed": _probe_benchmark_shape(
                    checked,
                    host["probe_storage"]["artifact_root"],
                )
                and artifact_manifest["file_count"] > 0
                and artifact_manifest["total_bytes"] > 0
                and secure_artifact_access,
            },
        }
    else:
        raise ValueError("probe result kind must be execution or benchmark")
    return _signed(payload)


def bind_probe_result(
    *,
    kind: str,
    static_record: Mapping[str, Any],
    result: Mapping[str, Any],
    probe_execution_record: Mapping[str, Any] | None = None,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Hash-seal imported probe JSON without qualifying its prior path access."""

    return _bind_probe_result(
        kind=kind,
        static_record=static_record,
        result=result,
        probe_execution_record=probe_execution_record,
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )


def _secured_probe_command_matches(
    host: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    kind: str,
) -> bool:
    command = record.get("executed_command")
    if not isinstance(command, str):
        return False
    try:
        arguments = tuple(shlex.split(command))
    except ValueError:
        return False
    runtime = _runtime_prefix(
        host["runtime"]["uv_path"],
        host["runtime"]["execution_python_path"],
    )
    descriptor_index = len(runtime) + 2
    if len(arguments) <= descriptor_index or not arguments[descriptor_index].isdigit():
        return False
    return arguments == _secured_probe_arguments(
        kind,
        uv_path=host["runtime"]["uv_path"],
        execution_python=host["runtime"]["execution_python_path"],
        artifact_descriptor=arguments[descriptor_index],
        repository=Path(host["execution"]["repository"]),
    )


def _probe_execution_for_benchmark(
    host: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    execution = verify_record(
        dict(record),
        record_type=_PROBE_EXECUTION_RECORD_TYPE,
    )
    expected = {
        "execution_commit": host["execution"]["observed_commit"],
        "host_identity": host["host_identity"],
        "host_identity_hash": host["host_identity_hash"],
        "tool_identity_hash": host["tool_identity_hash"],
        "static_record_hash": host["record_hash"],
        "probe_config_hash": host["registered_inputs"]["probe_config_hash"],
        "expected_artifact_root": host["probe_storage"]["artifact_root"],
        "planned_command": host["plan"]["probe"]["underlying_execute"],
    }
    if any(execution[name] != value for name, value in expected.items()):
        raise ValueError("probe execution record differs from the qualified host")
    if (
        not execution["secure_artifact_access"]
        or not execution["decision"]["shape_passed"]
        or not _secured_probe_command_matches(host, execution, kind="execution")
        or not _record_is_fresh(execution)
        or _parse_timestamp(execution["recorded_at"], label="probe recorded_at")
        < _parse_timestamp(host["recorded_at"], label="static recorded_at")
    ):
        raise ValueError("probe execution record is not valid for benchmark launch")
    return execution


def run_probe_step(
    *,
    kind: str,
    static_record: Mapping[str, Any],
    probe_execution_record: Mapping[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Run one probe step through a no-follow descriptor-anchored path."""

    host = verify_record(dict(static_record), record_type=_STATIC_RECORD_TYPE)
    if not host["decision"]["static_prerequisites_passed"]:
        raise ValueError("static host prerequisites did not pass")
    if not _record_is_fresh(host):
        raise ValueError("static host qualification is stale or future-dated")
    _assert_current_context(
        host,
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )
    repository = Path(host["execution"]["repository"])
    storage = Path(host["storage"]["storage_root"])
    artifact_root = Path(host["probe_storage"]["artifact_root"])
    if kind == "execution":
        if probe_execution_record is not None:
            raise ValueError("execution probe cannot name a prior probe record")
        parent_fd, leaf = _open_probe_parent(
            artifact_root,
            storage_root=storage,
        )
        root_fd: int | None = None
        expected_fd: int | None = None
        try:
            os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
            root_fd = os.open(
                leaf,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            if os.fstat(root_fd).st_dev != host["storage"]["directory_device_id"]:
                raise ValueError(
                    "probe artifact root is not on the qualified storage device"
                )
            command = _secured_probe_arguments(
                "execution",
                uv_path=host["runtime"]["uv_path"],
                execution_python=host["runtime"]["execution_python_path"],
                artifact_descriptor=root_fd,
                repository=repository,
            )
            result = _run_bounded_detached_json(
                command,
                cwd=repository,
                storage=storage,
                timeout_seconds=timeout_seconds,
                label="symbolic v2 probe execution",
                pass_fds=(root_fd,),
            )
            if result.get("artifact_root") != ".":
                raise ValueError("probe execution reported an unexpected artifact root")
            expected_fd = _open_existing_directory_nofollow(artifact_root)
            if not _same_open_directory(root_fd, expected_fd):
                raise ValueError("probe root path changed during execution")
            normalized = {**result, "artifact_root": str(artifact_root)}
            return _bind_probe_result(
                kind="execution",
                static_record=host,
                result=normalized,
                artifact_directory_fd=root_fd,
                executed_command=_command(command),
                proc_root=proc_root,
                machine_id_path=machine_id_path,
            )
        finally:
            if expected_fd is not None:
                os.close(expected_fd)
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
    if kind == "benchmark":
        if probe_execution_record is None:
            raise ValueError("benchmark probe requires its execution record")
        execution = _probe_execution_for_benchmark(host, probe_execution_record)
        root_fd = _open_existing_directory_nofollow(artifact_root)
        try:
            identity = _artifact_storage_identity(
                artifact_root,
                storage_root=storage,
            )
            if identity["device"] != host["storage"]["directory_device_id"]:
                raise ValueError(
                    "probe artifact root is not on the qualified storage device"
                )
            opened = os.fstat(root_fd)
            if (opened.st_dev, opened.st_ino) != (
                identity["device"],
                identity["inode"],
            ) or identity != execution["artifact_storage_identity"]:
                raise ValueError("probe artifact identity changed before benchmark")
            current_manifest, current_stability_hash = (
                _stable_manifest_and_seal_from_descriptor(root_fd)
            )
            if current_manifest != execution["artifact_manifest"]:
                raise ValueError("probe artifact content changed before benchmark")
            if current_stability_hash != execution["artifact_stability_hash"]:
                raise ValueError("probe artifact metadata changed before benchmark")
            command = _secured_probe_arguments(
                "benchmark",
                uv_path=host["runtime"]["uv_path"],
                execution_python=host["runtime"]["execution_python_path"],
                artifact_descriptor=root_fd,
                repository=repository,
            )
            result = _run_bounded_detached_json(
                command,
                cwd=repository,
                storage=storage,
                timeout_seconds=timeout_seconds,
                label="symbolic v2 probe benchmark",
                pass_fds=(root_fd,),
            )
            if result.get("artifact_root") != ".":
                raise ValueError("probe benchmark reported an unexpected artifact root")
            normalized = {**result, "artifact_root": str(artifact_root)}
            return _bind_probe_result(
                kind="benchmark",
                static_record=host,
                result=normalized,
                probe_execution_record=execution,
                artifact_directory_fd=root_fd,
                executed_command=_command(command),
                proc_root=proc_root,
                machine_id_path=machine_id_path,
            )
        finally:
            os.close(root_fd)
    raise ValueError("probe step kind must be execution or benchmark")


def _fresh_and_ordered(
    *,
    host: Mapping[str, Any],
    e192: Mapping[str, Any],
    e768: Mapping[str, Any],
    probe_execution: Mapping[str, Any],
    probe_benchmark: Mapping[str, Any],
) -> bool:
    now = datetime.now(UTC)
    timestamps = (
        _parse_timestamp(host["recorded_at"], label="static recorded_at"),
        _parse_timestamp(e192["started_at"], label="E192 started_at"),
        _parse_timestamp(e192["recorded_at"], label="E192 recorded_at"),
        _parse_timestamp(e768["started_at"], label="E768 started_at"),
        _parse_timestamp(e768["recorded_at"], label="E768 recorded_at"),
        _parse_timestamp(
            probe_execution["recorded_at"],
            label="probe execution recorded_at",
        ),
        _parse_timestamp(
            probe_benchmark["recorded_at"],
            label="probe benchmark recorded_at",
        ),
    )
    return (
        all(left <= right for left, right in pairwise(timestamps))
        and now - MAXIMUM_RECORD_AGE <= timestamps[0]
        and timestamps[-1] <= now + MAXIMUM_CLOCK_SKEW
    )


def _assessment_recorded_at(
    *,
    host: Mapping[str, Any],
    e192: Mapping[str, Any],
    e768: Mapping[str, Any],
    probe_execution: Mapping[str, Any],
    probe_benchmark: Mapping[str, Any],
) -> str:
    return max(
        _parse_timestamp(record["recorded_at"], label="input recorded_at")
        for record in (host, e192, e768, probe_execution, probe_benchmark)
    ).isoformat()


def assess_qualification(
    *,
    static_record: Mapping[str, Any],
    e192_record: Mapping[str, Any],
    e768_record: Mapping[str, Any],
    probe_execution_result: Mapping[str, Any],
    probe_benchmark_result: Mapping[str, Any],
    available_window_hours: float,
    recovery_margin_hours: float,
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Combine completed operational gates without launching study execution."""

    host = verify_record(dict(static_record), record_type=_STATIC_RECORD_TYPE)
    e192 = verify_record(dict(e192_record), record_type=_CAPACITY_RECORD_TYPE)
    e768 = verify_record(dict(e768_record), record_type=_CAPACITY_RECORD_TYPE)
    probe_execution = verify_record(
        dict(probe_execution_result),
        record_type=_PROBE_EXECUTION_RECORD_TYPE,
    )
    probe_benchmark = verify_record(
        dict(probe_benchmark_result),
        record_type=_PROBE_BENCHMARK_RECORD_TYPE,
    )
    window = _expect_number(
        available_window_hours,
        label="available runtime window",
        minimum=0.0,
        strict_minimum=True,
    )
    margin = _expect_number(
        recovery_margin_hours,
        label="recovery margin",
        minimum=0.0,
    )

    commit = host["execution"]["observed_commit"]
    try:
        _assert_current_context(
            host,
            proc_root=proc_root,
            machine_id_path=machine_id_path,
        )
        checkout_pinned = True
    except (OSError, subprocess.SubprocessError, ValueError):
        checkout_pinned = False
    host_resources_revalidated = _current_host_resources_match(
        host,
        proc_root=proc_root,
    )
    static_binding_passed = (
        e192["static_record_hash"]
        == e768["static_record_hash"]
        == probe_execution["static_record_hash"]
        == probe_benchmark["static_record_hash"]
        == host["record_hash"]
    )
    host_binding_passed = all(
        record["host_identity_hash"] == host["host_identity_hash"]
        and record["host_identity"] == host["host_identity"]
        for record in (e192, e768, probe_execution, probe_benchmark)
    )
    tool_binding_passed = all(
        record["tool_identity_hash"] == host["tool_identity_hash"]
        for record in (e192, e768, probe_execution, probe_benchmark)
    )
    expected_capacity_execution = _capacity_execution_identity(host)
    execution_binding_passed = all(
        (
            e192["execution"] == expected_capacity_execution,
            e768["execution"] == expected_capacity_execution,
            probe_execution["execution_commit"] == commit,
            probe_benchmark["execution_commit"] == commit,
        )
    )
    capacity_passed = all(
        (
            e192.get("stage") == "e192",
            e768.get("stage") == "e768",
            e192["decision"]["passed"] is True,
            e768["decision"]["passed"] is True,
        )
    )
    expected_probe_root = host["probe_storage"]["artifact_root"]
    probe_static_binding_passed = all(
        (
            probe_execution["expected_artifact_root"] == expected_probe_root,
            probe_benchmark["expected_artifact_root"] == expected_probe_root,
            probe_execution["result"]["artifact_root"] == expected_probe_root,
            probe_benchmark["result"]["artifact_root"] == expected_probe_root,
            probe_execution["artifact_storage_identity"]["canonical_path"]
            == expected_probe_root,
            probe_benchmark["artifact_storage_identity"]["canonical_path"]
            == expected_probe_root,
            probe_execution["artifact_storage_identity"]["device"]
            == host["storage"]["directory_device_id"],
            probe_benchmark["artifact_storage_identity"]["device"]
            == host["storage"]["directory_device_id"],
            probe_execution["result"]["artifact_root"]
            == probe_benchmark["result"]["artifact_root"],
            probe_execution["probe_config_hash"]
            == probe_benchmark["probe_config_hash"]
            == host["registered_inputs"]["probe_config_hash"],
            probe_execution["planned_command"]
            == host["plan"]["probe"]["underlying_execute"],
            probe_benchmark["planned_command"]
            == host["plan"]["probe"]["underlying_benchmark"],
            _secured_probe_command_matches(
                host,
                probe_execution,
                kind="execution",
            ),
            _secured_probe_command_matches(
                host,
                probe_benchmark,
                kind="benchmark",
            ),
        )
    )
    try:
        (
            current_artifact_identity,
            current_artifact_manifest,
            current_artifact_stability_hash,
        ) = _current_artifact_evidence(
            Path(expected_probe_root),
            storage_root=Path(host["storage"]["storage_root"]),
        )
        probe_artifacts_current = all(
            (
                current_artifact_identity
                == probe_execution["artifact_storage_identity"],
                current_artifact_identity
                == probe_benchmark["artifact_storage_identity"],
                current_artifact_manifest == probe_execution["artifact_manifest"],
                current_artifact_manifest == probe_benchmark["artifact_manifest"],
                current_artifact_stability_hash
                == probe_execution["artifact_stability_hash"],
                current_artifact_stability_hash
                == probe_benchmark["artifact_stability_hash"],
            )
        )
    except (OSError, ValueError):
        probe_artifacts_current = False
    probe_shape_passed = all(
        (
            probe_execution["decision"]["shape_passed"],
            probe_benchmark["decision"]["shape_passed"],
            probe_benchmark["probe_execution_record_hash"]
            == probe_execution["record_hash"],
            probe_execution["artifact_storage_identity"]
            == probe_benchmark["artifact_storage_identity"],
            probe_execution["artifact_manifest"]
            == probe_benchmark["artifact_manifest"],
            probe_execution["artifact_stability_hash"]
            == probe_benchmark["artifact_stability_hash"],
            probe_execution["secure_artifact_access"] is True,
            probe_benchmark["secure_artifact_access"] is True,
            probe_static_binding_passed,
            probe_artifacts_current,
        )
    )
    projection = _probe_projection(probe_benchmark["result"])
    consistent_budget = (
        projection is not None
        and projection["projected_hours"] > 0
        and projection["operational_budget_hours"] > 0
        and _probe_benchmark_metrics_consistent(probe_benchmark["result"])
    )
    budget = projection["operational_budget_hours"] if consistent_budget else None
    time_passed = (
        consistent_budget and budget is not None and window >= float(budget) + margin
    )
    freshness_order_passed = _fresh_and_ordered(
        host=host,
        e192=e192,
        e768=e768,
        probe_execution=probe_execution,
        probe_benchmark=probe_benchmark,
    )
    passed = all(
        (
            host["decision"]["static_prerequisites_passed"],
            checkout_pinned,
            host_resources_revalidated,
            static_binding_passed,
            host_binding_passed,
            tool_binding_passed,
            execution_binding_passed,
            capacity_passed,
            probe_shape_passed,
            freshness_order_passed,
            time_passed,
        )
    )
    payload = {
        "schema_version": 1,
        "record_type": _ASSESSMENT_RECORD_TYPE,
        "recorded_at": _assessment_recorded_at(
            host=host,
            e192=e192,
            e768=e768,
            probe_execution=probe_execution,
            probe_benchmark=probe_benchmark,
        ),
        "execution_commit": commit,
        "host_identity": host["host_identity"],
        "host_identity_hash": host["host_identity_hash"],
        "tool_identity_hash": host["tool_identity_hash"],
        "inputs": {
            "static_record_hash": host["record_hash"],
            "e192_record_hash": e192["record_hash"],
            "e768_record_hash": e768["record_hash"],
            "probe_execution_record_hash": probe_execution["record_hash"],
            "probe_benchmark_record_hash": probe_benchmark["record_hash"],
        },
        "checks": {
            "capacity_passed": capacity_passed,
            "capacity_static_binding_passed": static_binding_passed,
            "execution_binding_passed": execution_binding_passed,
            "execution_checkout_pinned": checkout_pinned,
            "host_resources_revalidated": host_resources_revalidated,
            "host_binding_passed": host_binding_passed,
            "input_freshness_order_passed": freshness_order_passed,
            "probe_budget_consistent": consistent_budget,
            "probe_shape_passed": probe_shape_passed,
            "probe_static_binding_passed": probe_static_binding_passed,
            "probe_artifacts_current": probe_artifacts_current,
            "static_prerequisites_passed": host["decision"][
                "static_prerequisites_passed"
            ],
            "time_window_passed": time_passed,
            "tool_binding_passed": tool_binding_passed,
        },
        "runtime": {
            "available_window_hours": window,
            "measured_two_times_projection_hours": budget,
            "recovery_margin_hours": margin,
            "required_window_hours": (
                float(budget) + margin
                if consistent_budget and budget is not None
                else None
            ),
        },
        "decision": {
            "qualification_passed": passed,
            "registered_execution_authorized_by_this_record": False,
            "next_action": (
                "Independent review may authorize the separately invoked registered "
                "calibration."
                if passed
                else "Resolve failed operational gates on the intended host; do not "
                "run the registered study."
            ),
        },
        "scientific_boundary": {
            "creates_study_artifacts": False,
            "executes_registered_study": False,
            "inspects_probe_metrics": False,
            "scientific_use": "prohibited",
        },
    }
    return _signed(payload)


def verify_assessment_bundle(
    *,
    assessment_record: Mapping[str, Any],
    static_record: Mapping[str, Any],
    e192_record: Mapping[str, Any],
    e768_record: Mapping[str, Any],
    probe_execution_record: Mapping[str, Any],
    probe_benchmark_record: Mapping[str, Any],
    proc_root: Path = Path("/proc"),
    machine_id_path: Path = Path("/etc/machine-id"),
) -> dict[str, Any]:
    """Verify an assessment against all five records it claims to summarize."""

    assessment = verify_record(
        dict(assessment_record),
        record_type=_ASSESSMENT_RECORD_TYPE,
    )
    recomputed = assess_qualification(
        static_record=static_record,
        e192_record=e192_record,
        e768_record=e768_record,
        probe_execution_result=probe_execution_record,
        probe_benchmark_result=probe_benchmark_record,
        available_window_hours=assessment["runtime"]["available_window_hours"],
        recovery_margin_hours=assessment["runtime"]["recovery_margin_hours"],
        proc_root=proc_root,
        machine_id_path=machine_id_path,
    )
    if assessment != recomputed:
        raise ValueError("assessment does not match its bound qualification records")
    return assessment


_STATIC_BOUNDARY = {
    "creates_study_artifacts": False,
    "executes_registered_study": False,
    "inspects_probe_metrics": False,
    "scientific_use": "prohibited",
}
_CAPACITY_BOUNDARY = {
    "synthetic_dataset_only": True,
    "creates_study_artifacts": False,
    "executes_registered_study": False,
    "scientific_use": "prohibited",
}
_CAPACITY_BENCHMARK_KEYS = frozenset(
    {
        "algorithm_replicas",
        "checkpoint_count",
        "checkpoint_zero_metric_count",
        "dataset_elapsed_seconds",
        "dataset_hash",
        "environment_replicas",
        "maximum_rss_mib",
        "metric_count",
        "observation_count",
        "pool_elapsed_seconds",
        "pooled_checkpoint_count",
        "positive_checkpoint_metric_count",
        "rss_before_mib",
        "rss_increment_upper_bound_mib",
        "total_elapsed_seconds",
    }
)


def _validate_host_identity(value: object, *, label: str) -> dict[str, Any]:
    identity = _expect_keys(
        value,
        label=label,
        keys=frozenset({"machine_id_hash", "boot_id_hash"}),
    )
    _expect_sha256(identity["machine_id_hash"], label=f"{label}.machine_id_hash")
    _expect_sha256(identity["boot_id_hash"], label=f"{label}.boot_id_hash")
    return identity


def _validate_tool_identity(value: object) -> dict[str, Any]:
    tool = _expect_keys(
        value,
        label="tool",
        keys=frozenset(
            {
                "repository",
                "commit",
                "worktree_clean",
                "source_hash",
                "git_directory",
                "git_path",
                "git_sha256",
                "python_path",
                "python_sha256",
                "python_prefix",
                "python_version",
                "dependency_lock_sha256",
                "dependency_environment_hash",
                "startup_environment_hash",
            }
        ),
    )
    repository = Path(_expect_string(tool["repository"], label="tool.repository"))
    if not repository.is_absolute():
        raise ValueError("tool.repository must be absolute")
    _full_git_sha(tool["commit"], label="qualification tool commit")
    _expect_bool(tool["worktree_clean"], label="tool.worktree_clean")
    for name in (
        "source_hash",
        "git_sha256",
        "python_sha256",
        "dependency_lock_sha256",
        "dependency_environment_hash",
        "startup_environment_hash",
    ):
        _expect_sha256(tool[name], label=f"tool.{name}")
    for name in ("git_directory", "git_path", "python_path", "python_prefix"):
        path = Path(_expect_string(tool[name], label=f"tool.{name}"))
        if not path.is_absolute():
            raise ValueError(f"tool.{name} must be absolute")
    tool_python = _expect_string(
        tool["python_version"],
        label="tool.python_version",
    )
    if not _python_version_supported(tool_python):
        raise ValueError("tool.python_version is unsupported")
    return tool


def _validate_stable_tree_identity(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    identity = _expect_keys(
        value,
        label=label,
        keys=frozenset(
            {
                "path",
                "device",
                "inode",
                "mode",
                "link_count",
                "file_count",
                "directory_count",
                "total_bytes",
                "manifest_hash",
                "stability_hash",
            }
        ),
    )
    path = Path(_expect_string(identity["path"], label=f"{label}.path"))
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    for name in (
        "device",
        "inode",
        "mode",
        "link_count",
        "file_count",
        "directory_count",
        "total_bytes",
    ):
        _expect_int(identity[name], label=f"{label}.{name}", minimum=0)
    if (
        not stat.S_ISDIR(identity["mode"])
        or identity["inode"] < 1
        or identity["link_count"] < 1
        or identity["directory_count"] < 1
    ):
        raise ValueError(f"{label} metadata is invalid")
    for name in ("manifest_hash", "stability_hash"):
        _expect_sha256(identity[name], label=f"{label}.{name}")
    return identity


def _validate_execution_environment_identity(
    value: object,
) -> dict[str, Any]:
    identity = _expect_keys(
        value,
        label="execution environment",
        keys=frozenset(
            {
                "prefix",
                "python_version",
                "pyvenv_config",
                "includes_system_site_packages",
                "site_packages",
                "pth_path_entries",
                "pth_target_identities",
                "checkout_import_trees",
                "pth_executable_line_hashes",
            }
        ),
    )
    prefix = Path(
        _expect_string(identity["prefix"], label="execution environment.prefix")
    )
    if not prefix.is_absolute():
        raise ValueError("execution environment.prefix must be absolute")
    python_version = _expect_string(
        identity["python_version"],
        label="execution environment.python_version",
    )
    if not _python_version_supported(python_version):
        raise ValueError("execution environment Python version is unsupported")
    config = _expect_keys(
        identity["pyvenv_config"],
        label="execution environment.pyvenv_config",
        keys=frozenset(
            {
                "path",
                "device",
                "inode",
                "mode",
                "link_count",
                "size_bytes",
                "mtime_ns",
                "ctime_ns",
                "sha256",
            }
        ),
    )
    config_path = Path(
        _expect_string(
            config["path"],
            label="execution environment.pyvenv_config.path",
        )
    )
    if config_path != prefix / "pyvenv.cfg":
        raise ValueError("execution environment pyvenv.cfg path is inconsistent")
    for name in (
        "device",
        "inode",
        "mode",
        "link_count",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    ):
        _expect_int(
            config[name],
            label=f"execution environment.pyvenv_config.{name}",
            minimum=0,
        )
    if (
        not stat.S_ISREG(config["mode"])
        or config["inode"] < 1
        or config["link_count"] < 1
    ):
        raise ValueError("execution environment pyvenv.cfg metadata is invalid")
    _expect_sha256(
        config["sha256"],
        label="execution environment.pyvenv_config.sha256",
    )
    _expect_bool(
        identity["includes_system_site_packages"],
        label="execution environment.includes_system_site_packages",
    )
    site_packages = _validate_stable_tree_identity(
        identity["site_packages"],
        label="execution environment.site_packages",
    )
    site_path = Path(site_packages["path"])
    major, minor, *_ = python_version.split(".")
    expected_site_path = prefix / "lib" / f"python{major}.{minor}" / "site-packages"
    if site_path != expected_site_path:
        raise ValueError("execution environment site-packages path is inconsistent")
    path_entries = identity["pth_path_entries"]
    if not isinstance(path_entries, list):
        raise ValueError("execution environment.pth_path_entries must be a list")
    checked_entries = [
        _expect_string(
            entry,
            label=f"execution environment.pth_path_entries[{index}]",
        )
        for index, entry in enumerate(path_entries)
    ]
    if checked_entries != sorted(set(checked_entries)) or any(
        not Path(entry).is_absolute() or str(_absolute_path(Path(entry))) != entry
        for entry in checked_entries
    ):
        raise ValueError(
            "execution environment path entries must be unique normalized absolutes"
        )
    target_identities = identity["pth_target_identities"]
    if not isinstance(target_identities, list):
        raise ValueError("execution environment.pth_target_identities must be a list")
    checked_targets = [
        _validate_stable_tree_identity(
            target,
            label=f"execution environment.pth_target_identities[{index}]",
        )
        for index, target in enumerate(target_identities)
    ]
    expected_target_paths = [
        entry for entry in checked_entries if not _path_within(Path(entry), site_path)
    ]
    if [target["path"] for target in checked_targets] != expected_target_paths:
        raise ValueError("execution environment .pth targets do not match path entries")
    checkout_import_trees = identity["checkout_import_trees"]
    if not isinstance(checkout_import_trees, list):
        raise ValueError("execution environment.checkout_import_trees must be a list")
    checked_checkout_trees = [
        _validate_stable_tree_identity(
            target,
            label=f"execution environment.checkout_import_trees[{index}]",
        )
        for index, target in enumerate(checkout_import_trees)
    ]
    if [target["path"] for target in checked_checkout_trees] != [
        str(prefix.parent / "src"),
        str(prefix.parent / "scripts"),
    ]:
        raise ValueError("execution environment checkout import trees are incomplete")
    executable_hashes = identity["pth_executable_line_hashes"]
    if (
        not isinstance(executable_hashes, list)
        or executable_hashes != sorted(set(executable_hashes))
        or any(not is_sha256(value) for value in executable_hashes)
        or not set(executable_hashes) <= _ALLOWED_PTH_EXECUTABLE_LINE_HASHES
    ):
        raise ValueError("execution environment executable .pth lines are not approved")
    return identity


def _validate_static_record(record: Mapping[str, Any]) -> None:
    _expect_keys(
        record,
        label="static qualification record",
        keys=frozenset(
            {
                "schema_version",
                "record_type",
                "recorded_at",
                "execution",
                "tool",
                "tool_identity_hash",
                "host_identity",
                "host_identity_hash",
                "runtime",
                "host",
                "storage",
                "probe_storage",
                "registered_inputs",
                "requirements",
                "plan",
                "decision",
                "scientific_boundary",
                "record_hash",
            }
        ),
    )
    _parse_timestamp(record["recorded_at"], label="static recorded_at")
    execution = _expect_keys(
        record["execution"],
        label="static execution",
        keys=frozenset(
            {
                "expected_commit",
                "observed_commit",
                "repository",
                "git_directory",
                "status",
                "worktree_clean",
                "uv_lock_sha256",
            }
        ),
    )
    _full_git_commit(execution["expected_commit"])
    _full_git_sha(execution["observed_commit"], label="observed execution commit")
    repository = Path(
        _expect_string(execution["repository"], label="execution.repository")
    )
    if not repository.is_absolute():
        raise ValueError("execution.repository must be absolute")
    git_directory = Path(
        _expect_string(execution["git_directory"], label="execution.git_directory")
    )
    if not git_directory.is_absolute():
        raise ValueError("execution.git_directory must be absolute")
    status = _expect_string(
        execution["status"],
        label="execution.status",
        nonempty=False,
    )
    clean = _expect_bool(
        execution["worktree_clean"],
        label="execution.worktree_clean",
    )
    if clean != (status == ""):
        raise ValueError("execution cleanliness disagrees with recorded status")
    if execution["uv_lock_sha256"] is not None:
        _expect_sha256(
            execution["uv_lock_sha256"],
            label="execution.uv_lock_sha256",
        )

    tool = _validate_tool_identity(record["tool"])
    tool_hash = _expect_sha256(
        record["tool_identity_hash"],
        label="tool_identity_hash",
    )
    if tool_hash != _tool_identity_hash(tool):
        raise ValueError("tool identity hash does not match tool fields")
    host_identity = _validate_host_identity(
        record["host_identity"],
        label="host_identity",
    )
    _expect_sha256(record["host_identity_hash"], label="host_identity_hash")
    runtime = _expect_keys(
        record["runtime"],
        label="runtime",
        keys=frozenset(
            {
                "execution_python",
                "execution_python_path",
                "execution_python_sha256",
                "execution_prefix",
                "environment_synced",
                "execution_environment",
                "execution_environment_hash",
                "kernel",
                "logical_cpu_count",
                "platform",
                "tool_python",
                "git",
                "git_path",
                "git_sha256",
                "uv",
                "uv_path",
                "uv_sha256",
            }
        ),
    )
    for name in ("execution_python", "execution_prefix"):
        if runtime[name] is not None:
            _expect_string(runtime[name], label=f"runtime.{name}")
    python_path = Path(
        _expect_string(
            runtime["execution_python_path"],
            label="runtime.execution_python_path",
        )
    )
    if not python_path.is_absolute():
        raise ValueError("runtime.execution_python_path must be absolute")
    expected_python_path = repository / ".venv" / "bin" / "python"
    if python_path != expected_python_path:
        raise ValueError(
            "runtime.execution_python_path must name the checkout .venv Python"
        )
    if runtime["execution_python_sha256"] is not None:
        _expect_sha256(
            runtime["execution_python_sha256"],
            label="runtime.execution_python_sha256",
        )
    _expect_bool(runtime["environment_synced"], label="runtime.environment_synced")
    if runtime["execution_environment"] is None:
        if runtime["execution_environment_hash"] is not None:
            raise ValueError("runtime execution environment hash requires an identity")
    else:
        environment = _validate_execution_environment_identity(
            runtime["execution_environment"]
        )
        environment_hash = _expect_sha256(
            runtime["execution_environment_hash"],
            label="runtime.execution_environment_hash",
        )
        if environment_hash != _execution_environment_hash(environment):
            raise ValueError(
                "runtime execution environment hash does not match identity"
            )
        if (
            environment["prefix"] != runtime["execution_prefix"]
            or environment["python_version"] != runtime["execution_python"]
        ):
            raise ValueError(
                "runtime execution environment differs from Python runtime"
            )
    _expect_string(runtime["kernel"], label="runtime.kernel")
    if runtime["logical_cpu_count"] is not None:
        _expect_int(
            runtime["logical_cpu_count"],
            label="runtime.logical_cpu_count",
            minimum=1,
        )
    _expect_string(runtime["platform"], label="runtime.platform")
    _expect_string(runtime["tool_python"], label="runtime.tool_python")
    if runtime["tool_python"] != tool["python_version"]:
        raise ValueError("runtime tool Python version differs from tool identity")
    _expect_string(runtime["git"], label="runtime.git")
    _expect_string(runtime["git_path"], label="runtime.git_path")
    _expect_sha256(runtime["git_sha256"], label="runtime.git_sha256")
    if not Path(runtime["git_path"]).is_absolute():
        raise ValueError("runtime.git_path must be absolute")
    _expect_string(runtime["uv"], label="runtime.uv")
    _expect_string(runtime["uv_path"], label="runtime.uv_path")
    _expect_sha256(runtime["uv_sha256"], label="runtime.uv_sha256")
    if not Path(runtime["uv_path"]).is_absolute():
        raise ValueError("runtime.uv_path must be absolute")
    if (
        tool["git_path"] != runtime["git_path"]
        or tool["git_sha256"] != runtime["git_sha256"]
    ):
        raise ValueError("tool Git identity differs from inspected Git identity")

    host = _expect_keys(
        record["host"],
        label="host",
        keys=frozenset(
            {
                "available_memory_bytes",
                "physical_memory_bytes",
                "swap_bytes",
                "vmstat",
            }
        ),
    )
    for name in (
        "available_memory_bytes",
        "physical_memory_bytes",
        "swap_bytes",
    ):
        _expect_int(host[name], label=f"host.{name}", minimum=0)
    vmstat = _expect_keys(
        host["vmstat"],
        label="host.vmstat",
        keys=frozenset({"pswpin_pages", "pswpout_pages"}),
    )
    for name in vmstat:
        _expect_int(vmstat[name], label=f"host.vmstat.{name}", minimum=0)

    storage = _expect_keys(
        record["storage"],
        label="storage",
        keys=frozenset(
            {
                "device",
                "mount_options",
                "mount_point",
                "type",
                "available_bytes",
                "available_inodes",
                "additional_storage_bytes",
                "additional_inodes",
                "directory_device_id",
                "directory_inode",
                "local_filesystem",
                "paired_raw_reference_bytes",
                "paired_raw_reference_files",
                "required_bytes",
                "required_inodes",
                "solid_state",
                "read_write_mount",
                "storage_root",
                "writable",
            }
        ),
    )
    for name in ("device", "mount_options", "mount_point", "type", "storage_root"):
        _expect_string(storage[name], label=f"storage.{name}")
    if not Path(storage["storage_root"]).is_absolute():
        raise ValueError("storage.storage_root must be absolute")
    for name in (
        "available_bytes",
        "available_inodes",
        "additional_storage_bytes",
        "additional_inodes",
        "directory_device_id",
        "paired_raw_reference_bytes",
        "paired_raw_reference_files",
        "required_bytes",
        "required_inodes",
    ):
        _expect_int(storage[name], label=f"storage.{name}", minimum=0)
    _expect_int(storage["directory_inode"], label="storage.directory_inode", minimum=1)
    if (
        storage["paired_raw_reference_bytes"] != REFERENCE_PAIR_RAW_BYTES
        or storage["paired_raw_reference_files"] != REFERENCE_PAIR_RAW_FILES
    ):
        raise ValueError("storage reference constants do not match")
    if storage["required_bytes"] != (
        REFERENCE_PAIR_RAW_BYTES + storage["additional_storage_bytes"]
    ) or storage["required_inodes"] != (
        REFERENCE_PAIR_RAW_FILES + storage["additional_inodes"]
    ):
        raise ValueError("storage requirements do not include declared margins")
    for name in ("local_filesystem", "read_write_mount", "writable"):
        _expect_bool(storage[name], label=f"storage.{name}")
    if storage["solid_state"] is not None:
        _expect_bool(storage["solid_state"], label="storage.solid_state")

    probe_storage = _expect_keys(
        record["probe_storage"],
        label="probe_storage",
        keys=frozenset(
            {"artifact_root", "existed_at_inspection", "on_intended_storage"}
        ),
    )
    artifact_root = Path(
        _expect_string(
            probe_storage["artifact_root"],
            label="probe_storage.artifact_root",
        )
    )
    if not artifact_root.is_absolute():
        raise ValueError("probe_storage.artifact_root must be absolute")
    for name in ("existed_at_inspection", "on_intended_storage"):
        _expect_bool(probe_storage[name], label=f"probe_storage.{name}")

    registered = _expect_keys(
        record["registered_inputs"],
        label="registered_inputs",
        keys=frozenset({"calibration_config_hash", "probe_config_hash"}),
    )
    for name in registered:
        _expect_sha256(registered[name], label=f"registered_inputs.{name}")
    expected_plan = _exact_plan_payload(
        repository=repository,
        commit=execution["expected_commit"],
        probe=artifact_root,
        uv=Path(runtime["uv_path"]),
        python=python_path,
        git=Path(runtime["git_path"]),
        git_directory=git_directory,
    )
    if record["plan"] != expected_plan:
        raise ValueError("static exact plan does not match bound executables and paths")
    expected_requirements = _static_requirements(
        execution=execution,
        tool=tool,
        runtime=runtime,
        host_identity=host_identity,
        host=host,
        storage=storage,
        registered_inputs=registered,
        probe_storage=probe_storage,
    )
    if record["requirements"] != expected_requirements:
        raise ValueError("static requirements are incomplete or inconsistent")
    expected_host_hash = _host_identity_hash(
        execution=execution,
        host_identity=host_identity,
        runtime=runtime,
        host=host,
        storage=storage,
    )
    if record["host_identity_hash"] != expected_host_hash:
        raise ValueError("host identity hash does not match static fields")
    decision = _expect_keys(
        record["decision"],
        label="static decision",
        keys=frozenset(
            {
                "qualification_steps_executed",
                "registered_execution_authorized",
                "static_prerequisites_passed",
            }
        ),
    )
    if decision != {
        "qualification_steps_executed": False,
        "registered_execution_authorized": False,
        "static_prerequisites_passed": all(
            item["passed"] for item in expected_requirements
        ),
    }:
        raise ValueError("static decision does not match recomputed requirements")
    if record["scientific_boundary"] != _STATIC_BOUNDARY:
        raise ValueError("static scientific boundary is invalid")


def _validate_checkout_snapshot(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    snapshot = _expect_keys(
        value,
        label=label,
        keys=frozenset(
            {
                "commit",
                "clean",
                "status",
                "uv_lock_sha256",
                "execution_python_sha256",
            }
        ),
    )
    _full_git_sha(snapshot["commit"], label=f"{label}.commit")
    clean = _expect_bool(snapshot["clean"], label=f"{label}.clean")
    status = _expect_string(
        snapshot["status"],
        label=f"{label}.status",
        nonempty=False,
    )
    if clean != (status == ""):
        raise ValueError(f"{label} cleanliness disagrees with status")
    for name in ("uv_lock_sha256", "execution_python_sha256"):
        if snapshot[name] is not None:
            _expect_sha256(snapshot[name], label=f"{label}.{name}")
    return snapshot


def _validate_capacity_benchmark_result(value: object) -> dict[str, Any]:
    if value == {}:
        return {}
    result = _expect_keys(
        value,
        label="capacity benchmark_result",
        keys=_CAPACITY_BENCHMARK_KEYS,
    )
    for name in (
        "algorithm_replicas",
        "checkpoint_count",
        "checkpoint_zero_metric_count",
        "environment_replicas",
        "metric_count",
        "observation_count",
        "pooled_checkpoint_count",
        "positive_checkpoint_metric_count",
    ):
        _expect_int(result[name], label=f"benchmark_result.{name}", minimum=0)
    _expect_sha256(result["dataset_hash"], label="benchmark_result.dataset_hash")
    for name in _CAPACITY_BENCHMARK_KEYS - {
        "algorithm_replicas",
        "checkpoint_count",
        "checkpoint_zero_metric_count",
        "dataset_hash",
        "environment_replicas",
        "metric_count",
        "observation_count",
        "pooled_checkpoint_count",
        "positive_checkpoint_metric_count",
    }:
        _expect_number(
            result[name],
            label=f"benchmark_result.{name}",
            minimum=0.0,
        )
    return result


def _validate_capacity_record(record: Mapping[str, Any]) -> None:
    _expect_keys(
        record,
        label="capacity qualification record",
        keys=frozenset(
            {
                "schema_version",
                "record_type",
                "recorded_at",
                "stage",
                "execution",
                "host_identity",
                "host_identity_hash",
                "tool_identity_hash",
                "static_record_hash",
                "command",
                "started_at",
                "elapsed_seconds",
                "timeout_seconds",
                "exit_code",
                "timed_out",
                "output_limit_exceeded",
                "benchmark_result",
                "prelaunch_checkout",
                "postrun_checkout",
                "prelaunch_execution_environment_hash",
                "postrun_execution_environment_hash",
                "host_memory",
                "host_swap",
                "process_major_faults",
                "stderr",
                "decision",
                "scientific_boundary",
                "record_hash",
            }
        ),
    )
    recorded = _parse_timestamp(record["recorded_at"], label="capacity recorded_at")
    started = _parse_timestamp(record["started_at"], label="capacity started_at")
    if started > recorded + MAXIMUM_CLOCK_SKEW:
        raise ValueError("capacity started_at follows recorded_at")
    stage = _expect_string(record["stage"], label="capacity stage")
    if stage not in _CAPACITY_SPECS:
        raise ValueError("capacity stage must be e192 or e768")
    execution = _expect_keys(
        record["execution"],
        label="capacity execution",
        keys=frozenset(
            {
                "commit",
                "repository",
                "git_directory",
                "git_path",
                "git_sha256",
                "uv_lock_sha256",
                "uv_path",
                "uv_sha256",
                "execution_python_path",
                "execution_python_sha256",
                "execution_environment_hash",
            }
        ),
    )
    _full_git_commit(execution["commit"])
    for name in (
        "repository",
        "git_directory",
        "git_path",
        "uv_path",
        "execution_python_path",
    ):
        path = Path(_expect_string(execution[name], label=f"execution.{name}"))
        if not path.is_absolute():
            raise ValueError(f"execution.{name} must be absolute")
    for name in (
        "git_sha256",
        "uv_lock_sha256",
        "uv_sha256",
        "execution_python_sha256",
        "execution_environment_hash",
    ):
        _expect_sha256(execution[name], label=f"execution.{name}")
    host_identity = _validate_host_identity(
        record["host_identity"],
        label="capacity host_identity",
    )
    _expect_sha256(record["host_identity_hash"], label="host_identity_hash")
    _expect_sha256(record["tool_identity_hash"], label="tool_identity_hash")
    _expect_sha256(record["static_record_hash"], label="static_record_hash")
    expected_command = _command(
        _capacity_arguments(
            stage,
            uv_path=execution["uv_path"],
            execution_python=execution["execution_python_path"],
        )
    )
    if record["command"] != expected_command:
        raise ValueError("capacity command does not match pinned executables and stage")
    elapsed_seconds = _expect_number(
        record["elapsed_seconds"],
        label="elapsed_seconds",
        minimum=0.0,
    )
    timeout_seconds = _expect_number(
        record["timeout_seconds"],
        label="timeout_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    exit_code = _expect_int(record["exit_code"], label="exit_code")
    timed_out = _expect_bool(record["timed_out"], label="timed_out")
    output_limit = _expect_bool(
        record["output_limit_exceeded"],
        label="output_limit_exceeded",
    )
    result = _validate_capacity_benchmark_result(record["benchmark_result"])
    if result and (exit_code != 0 or timed_out or output_limit):
        raise ValueError("failed capacity process cannot retain a benchmark result")
    if result:
        child_elapsed = float(result["total_elapsed_seconds"])
        if elapsed_seconds + 0.002 < child_elapsed:
            raise ValueError("capacity wrapper elapsed time is below benchmark time")
        if child_elapsed > timeout_seconds + 0.01:
            raise ValueError("capacity benchmark time exceeds its credible timeout")
    before = _validate_checkout_snapshot(
        record["prelaunch_checkout"],
        label="prelaunch_checkout",
    )
    after = _validate_checkout_snapshot(
        record["postrun_checkout"],
        label="postrun_checkout",
    )
    prelaunch_environment_hash = _expect_sha256(
        record["prelaunch_execution_environment_hash"],
        label="prelaunch_execution_environment_hash",
    )
    postrun_environment_hash = record["postrun_execution_environment_hash"]
    if postrun_environment_hash is not None:
        _expect_sha256(
            postrun_environment_hash,
            label="postrun_execution_environment_hash",
        )
    if (
        before["commit"] != execution["commit"]
        or before["uv_lock_sha256"] != execution["uv_lock_sha256"]
        or before["execution_python_sha256"] != execution["execution_python_sha256"]
        or prelaunch_environment_hash != execution["execution_environment_hash"]
        or not before["clean"]
    ):
        raise ValueError("capacity prelaunch snapshot differs from execution binding")
    memory = _expect_keys(
        record["host_memory"],
        label="host_memory",
        keys=frozenset(
            {
                "before_available_memory_bytes",
                "after_available_memory_bytes",
                "minimum_available_memory_bytes",
            }
        ),
    )
    for name in memory:
        _expect_int(memory[name], label=f"host_memory.{name}", minimum=0)
    swap = _expect_keys(
        record["host_swap"],
        label="host_swap",
        keys=frozenset(
            {
                "page_size_bytes",
                "pswpin_delta_pages",
                "pswpin_delta_bytes",
                "pswpout_delta_pages",
                "pswpout_delta_bytes",
            }
        ),
    )
    for name in swap:
        _expect_int(swap[name], label=f"host_swap.{name}", minimum=0)
    if (
        swap["pswpin_delta_bytes"]
        != swap["pswpin_delta_pages"] * swap["page_size_bytes"]
        or swap["pswpout_delta_bytes"]
        != swap["pswpout_delta_pages"] * swap["page_size_bytes"]
    ):
        raise ValueError("capacity swap byte and page deltas disagree")
    faults = _expect_int(
        record["process_major_faults"],
        label="process_major_faults",
        minimum=0,
    )
    stderr = _expect_string(record["stderr"], label="stderr", nonempty=False)
    if len(stderr) > 4000:
        raise ValueError("capacity stderr excerpt exceeds 4000 characters")
    expected_decision = _capacity_decision(
        stage,
        result=result,
        before_available_memory_bytes=memory["before_available_memory_bytes"],
        after_available_memory_bytes=memory["after_available_memory_bytes"],
        minimum_available_memory_bytes=memory["minimum_available_memory_bytes"],
        process_major_faults=faults,
        swapout_delta_pages=swap["pswpout_delta_pages"],
        checkout_unchanged=before == after and before["clean"] and after["clean"],
        execution_environment_unchanged=(
            prelaunch_environment_hash
            == postrun_environment_hash
            == execution["execution_environment_hash"]
        ),
    )
    if record["decision"] != expected_decision:
        raise ValueError("capacity decision does not match recomputed evidence")
    if record["scientific_boundary"] != _CAPACITY_BOUNDARY:
        raise ValueError("capacity scientific boundary is invalid")
    del host_identity


def _validate_artifact_storage_identity(value: object) -> dict[str, Any]:
    identity = _expect_keys(
        value,
        label="artifact_storage_identity",
        keys=frozenset({"canonical_path", "device", "inode"}),
    )
    path = Path(
        _expect_string(
            identity["canonical_path"],
            label="artifact_storage_identity.canonical_path",
        )
    )
    if not path.is_absolute():
        raise ValueError("artifact storage canonical path must be absolute")
    _expect_int(identity["device"], label="artifact_storage_identity.device", minimum=0)
    _expect_int(identity["inode"], label="artifact_storage_identity.inode", minimum=1)
    return identity


def _validate_artifact_manifest(value: object) -> dict[str, Any]:
    manifest = _expect_keys(
        value,
        label="artifact_manifest",
        keys=frozenset({"entries", "file_count", "total_bytes", "manifest_hash"}),
    )
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ValueError("artifact_manifest.entries must be a list")
    observed_paths: set[str] = set()
    observed_file_count = 0
    observed_total_bytes = 0
    for index, value in enumerate(entries):
        if not isinstance(value, dict):
            raise ValueError(f"artifact_manifest.entries[{index}] must be an object")
        kind = value.get("kind")
        keys = (
            frozenset({"kind", "path"})
            if kind == "directory"
            else frozenset({"kind", "path", "sha256", "size_bytes"})
            if kind == "file"
            else frozenset()
        )
        if not keys:
            raise ValueError(
                f"artifact_manifest.entries[{index}].kind must be file or directory"
            )
        entry = _expect_keys(
            value,
            label=f"artifact_manifest.entries[{index}]",
            keys=keys,
        )
        path_text = _expect_string(
            entry["path"],
            label=f"artifact_manifest.entries[{index}].path",
        )
        path = Path(path_text)
        if (
            path.is_absolute()
            or path.as_posix() != path_text
            or path_text in {".", ".."}
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact manifest paths must be normalized and relative")
        if path_text in observed_paths:
            raise ValueError("artifact manifest paths must be unique")
        observed_paths.add(path_text)
        if kind == "file":
            _expect_sha256(
                entry["sha256"],
                label=f"artifact_manifest.entries[{index}].sha256",
            )
            observed_total_bytes += _expect_int(
                entry["size_bytes"],
                label=f"artifact_manifest.entries[{index}].size_bytes",
                minimum=0,
            )
            observed_file_count += 1
    file_count = _expect_int(
        manifest["file_count"],
        label="artifact_manifest.file_count",
        minimum=0,
    )
    total_bytes = _expect_int(
        manifest["total_bytes"],
        label="artifact_manifest.total_bytes",
        minimum=0,
    )
    manifest_hash = _expect_sha256(
        manifest["manifest_hash"],
        label="artifact_manifest.manifest_hash",
    )
    if file_count != observed_file_count or total_bytes != observed_total_bytes:
        raise ValueError("artifact manifest aggregates do not match detailed entries")
    if manifest_hash != scientific_hash(
        entries,
        domain="operations.v2-probe-artifact-manifest.v1",
    ):
        raise ValueError("artifact manifest hash does not match detailed entries")
    return manifest


def _validate_probe_record_common(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, bool]:
    _parse_timestamp(record["recorded_at"], label="probe recorded_at")
    _full_git_commit(record["execution_commit"])
    host_identity = _validate_host_identity(
        record["host_identity"],
        label="probe host_identity",
    )
    _expect_sha256(record["host_identity_hash"], label="host_identity_hash")
    _expect_sha256(record["tool_identity_hash"], label="tool_identity_hash")
    _expect_sha256(record["static_record_hash"], label="static_record_hash")
    _expect_sha256(record["probe_config_hash"], label="probe_config_hash")
    _expect_string(record["planned_command"], label="planned probe command")
    if record["executed_command"] is not None:
        _expect_string(record["executed_command"], label="executed probe command")
    expected_root = _expect_string(
        record["expected_artifact_root"],
        label="expected_artifact_root",
    )
    if not Path(expected_root).is_absolute():
        raise ValueError("expected_artifact_root must be absolute")
    artifact_identity = _validate_artifact_storage_identity(
        record["artifact_storage_identity"]
    )
    artifact_manifest = _validate_artifact_manifest(record["artifact_manifest"])
    _expect_sha256(
        record["artifact_stability_hash"],
        label="artifact_stability_hash",
    )
    secure_artifact_access = _expect_bool(
        record["secure_artifact_access"],
        label="secure_artifact_access",
    )
    if secure_artifact_access != (record["executed_command"] is not None):
        raise ValueError("probe execution provenance is incomplete")
    if artifact_identity["canonical_path"] != expected_root:
        raise ValueError("artifact storage identity differs from expected root")
    _expect_sha256(record["source_result_hash"], label="source_result_hash")
    if record["scientific_boundary"] != _STATIC_BOUNDARY:
        raise ValueError("probe scientific boundary is invalid")
    return (
        host_identity,
        artifact_identity,
        artifact_manifest,
        expected_root,
        secure_artifact_access,
    )


def _validate_probe_execution_record(record: Mapping[str, Any]) -> None:
    _expect_keys(
        record,
        label="probe execution qualification record",
        keys=frozenset(
            {
                "schema_version",
                "record_type",
                "recorded_at",
                "execution_commit",
                "host_identity",
                "host_identity_hash",
                "tool_identity_hash",
                "static_record_hash",
                "probe_config_hash",
                "planned_command",
                "executed_command",
                "expected_artifact_root",
                "artifact_storage_identity",
                "artifact_manifest",
                "artifact_stability_hash",
                "secure_artifact_access",
                "source_result_hash",
                "result",
                "decision",
                "scientific_boundary",
                "record_hash",
            }
        ),
    )
    (
        _,
        _,
        artifact_manifest,
        expected_root,
        secure_artifact_access,
    ) = _validate_probe_record_common(record)
    result = _validate_probe_execution_result(record["result"])
    expected_hash = scientific_hash(
        result,
        domain="operations.v2-probe-execution-result.v1",
    )
    if record["source_result_hash"] != expected_hash:
        raise ValueError("probe execution result hash does not match result")
    if record["decision"] != {
        "shape_passed": (
            _probe_execution_shape(result, expected_root)
            and artifact_manifest["file_count"] > 0
            and artifact_manifest["total_bytes"] > 0
            and secure_artifact_access
        )
    }:
        raise ValueError("probe execution decision does not match result")


def _validate_probe_benchmark_record(record: Mapping[str, Any]) -> None:
    _expect_keys(
        record,
        label="probe benchmark qualification record",
        keys=frozenset(
            {
                "schema_version",
                "record_type",
                "recorded_at",
                "execution_commit",
                "host_identity",
                "host_identity_hash",
                "tool_identity_hash",
                "static_record_hash",
                "probe_config_hash",
                "planned_command",
                "executed_command",
                "expected_artifact_root",
                "artifact_storage_identity",
                "artifact_manifest",
                "artifact_stability_hash",
                "secure_artifact_access",
                "probe_execution_record_hash",
                "source_result_hash",
                "result",
                "decision",
                "scientific_boundary",
                "record_hash",
            }
        ),
    )
    (
        _,
        _,
        artifact_manifest,
        expected_root,
        secure_artifact_access,
    ) = _validate_probe_record_common(record)
    _expect_sha256(
        record["probe_execution_record_hash"],
        label="probe_execution_record_hash",
    )
    result = _validate_probe_benchmark_result(record["result"])
    expected_hash = scientific_hash(
        result,
        domain="operations.v2-probe-benchmark-result.v1",
    )
    if record["source_result_hash"] != expected_hash:
        raise ValueError("probe benchmark result hash does not match result")
    if record["decision"] != {
        "shape_passed": (
            _probe_benchmark_shape(result, expected_root)
            and artifact_manifest["file_count"] > 0
            and artifact_manifest["total_bytes"] > 0
            and secure_artifact_access
        )
    }:
        raise ValueError("probe benchmark decision does not match result")


def _validate_assessment_record(record: Mapping[str, Any]) -> None:
    _expect_keys(
        record,
        label="host qualification assessment",
        keys=frozenset(
            {
                "schema_version",
                "record_type",
                "recorded_at",
                "execution_commit",
                "host_identity",
                "host_identity_hash",
                "tool_identity_hash",
                "inputs",
                "checks",
                "runtime",
                "decision",
                "scientific_boundary",
                "record_hash",
            }
        ),
    )
    _parse_timestamp(record["recorded_at"], label="assessment recorded_at")
    if not _record_is_fresh(record):
        raise ValueError("assessment is stale or future-dated")
    _full_git_commit(record["execution_commit"])
    _validate_host_identity(record["host_identity"], label="assessment host_identity")
    _expect_sha256(record["host_identity_hash"], label="host_identity_hash")
    _expect_sha256(record["tool_identity_hash"], label="tool_identity_hash")
    inputs = _expect_keys(
        record["inputs"],
        label="assessment inputs",
        keys=frozenset(
            {
                "static_record_hash",
                "e192_record_hash",
                "e768_record_hash",
                "probe_execution_record_hash",
                "probe_benchmark_record_hash",
            }
        ),
    )
    for name in inputs:
        _expect_sha256(inputs[name], label=f"assessment inputs.{name}")
    check_names = frozenset(
        {
            "capacity_passed",
            "capacity_static_binding_passed",
            "execution_binding_passed",
            "execution_checkout_pinned",
            "host_resources_revalidated",
            "host_binding_passed",
            "input_freshness_order_passed",
            "probe_artifacts_current",
            "probe_budget_consistent",
            "probe_shape_passed",
            "probe_static_binding_passed",
            "static_prerequisites_passed",
            "time_window_passed",
            "tool_binding_passed",
        }
    )
    checks = _expect_keys(record["checks"], label="assessment checks", keys=check_names)
    for name in checks:
        _expect_bool(checks[name], label=f"assessment checks.{name}")
    runtime = _expect_keys(
        record["runtime"],
        label="assessment runtime",
        keys=frozenset(
            {
                "available_window_hours",
                "measured_two_times_projection_hours",
                "recovery_margin_hours",
                "required_window_hours",
            }
        ),
    )
    _expect_number(
        runtime["available_window_hours"],
        label="available_window_hours",
        minimum=0.0,
        strict_minimum=True,
    )
    measured = runtime["measured_two_times_projection_hours"]
    if measured is not None:
        _expect_number(
            measured,
            label="measured_two_times_projection_hours",
            minimum=0.0,
            strict_minimum=True,
        )
    _expect_number(
        runtime["recovery_margin_hours"],
        label="recovery_margin_hours",
        minimum=0.0,
    )
    if runtime["required_window_hours"] is not None:
        _expect_number(
            runtime["required_window_hours"],
            label="required_window_hours",
            minimum=0.0,
            strict_minimum=True,
        )
    expected_required_window = (
        float(measured) + float(runtime["recovery_margin_hours"])
        if checks["probe_budget_consistent"] and measured is not None
        else None
    )
    if checks["probe_budget_consistent"] != (measured is not None):
        raise ValueError("assessment budget consistency disagrees with projection")
    if runtime["required_window_hours"] != expected_required_window:
        raise ValueError("assessment required window does not match runtime values")
    expected_time_passed = (
        checks["probe_budget_consistent"]
        and expected_required_window is not None
        and float(runtime["available_window_hours"]) >= expected_required_window
    )
    if checks["time_window_passed"] != expected_time_passed:
        raise ValueError("assessment time-window check does not match runtime values")
    decision = _expect_keys(
        record["decision"],
        label="assessment decision",
        keys=frozenset(
            {
                "qualification_passed",
                "registered_execution_authorized_by_this_record",
                "next_action",
            }
        ),
    )
    passed = all(checks.values())
    expected_action = (
        "Independent review may authorize the separately invoked registered "
        "calibration."
        if passed
        else "Resolve failed operational gates on the intended host; do not run the "
        "registered study."
    )
    if decision != {
        "qualification_passed": passed,
        "registered_execution_authorized_by_this_record": False,
        "next_action": expected_action,
    }:
        raise ValueError("assessment decision does not match recomputed checks")
    if record["scientific_boundary"] != _STATIC_BOUNDARY:
        raise ValueError("assessment scientific boundary is invalid")
