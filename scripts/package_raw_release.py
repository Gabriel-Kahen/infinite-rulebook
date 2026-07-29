"""Create and verify deterministic, chunked archives of raw artifact roots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from infinite_rulebook.orchestration.config import load_experiment_config
from infinite_rulebook.orchestration.hashing import is_sha256, scientific_hash
from infinite_rulebook.orchestration.inventory import (
    RawArtifactInventory,
    load_raw_artifact_inventory,
    raw_artifact_inventory_from_dict,
)
from infinite_rulebook.orchestration.jsonio import parse_json_strict
from infinite_rulebook.orchestration.reproducibility import (
    EXECUTION_RECEIPT_FILENAME,
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
)

ASSET_MANIFEST_VERSION = 2
ASSET_MANIFEST_TYPE = "infinite-rulebook-raw-release-assets"
DEFAULT_CHUNK_BYTES = 1_900_000_000
_MANIFEST_DOMAIN = "raw-release-asset-manifest"
_TREE_HASH_PREFIX = b"infinite-rulebook-raw-artifact-tree-v1\0"
_STREAM_BYTES = 1 << 20
_MAX_MANIFEST_BYTES = 1 << 28
_MAX_PARTS = 100_000
_PART_INDEX_WIDTH = 8
_GZIP_HEADER = bytes.fromhex("1f8b08000000000000ff")
_RECONSTRUCTION = "cat <parts in manifest order> | gzip -dc | tar -xf -"
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FIELDS = {
    "artifact_type",
    "schema_version",
    "asset_label",
    "experiment_name",
    "config_hash",
    "raw_inventory_hash",
    "raw_inventory",
    "side",
    "archive_format",
    "compression",
    "excluded_files",
    "excluded_root_directories",
    "chunk_bytes",
    "tar_byte_size",
    "tar_sha256",
    "compressed_byte_size",
    "compressed_sha256",
    "parts",
    "reconstruction",
    "scientific_hash",
}
_PART_FIELDS = {"path", "byte_size", "sha256"}


class RawReleasePackagingError(ValueError):
    """Raised when an archive package is incomplete, unsafe, or inconsistent."""


def _safe_label(value: object) -> str:
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise RawReleasePackagingError("asset label is not a safe filename stem")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RawReleasePackagingError(f"{label} must be a positive integer")
    return value


def _canonical_path(path: str | Path, *, must_exist: bool) -> Path:
    try:
        return Path(path).resolve(strict=must_exist)
    except OSError as error:
        raise RawReleasePackagingError(f"cannot resolve path: {path}") from error


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _durably_create_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        os.close(descriptor)
        raise RawReleasePackagingError(
            f"cannot durably create raw release directory: {path}"
        ) from error
    os.close(descriptor)


def _validate_output_location(
    output: Path,
    *,
    artifact_root: Path,
    config_path: Path,
    inventory_path: Path,
) -> None:
    for label, source in (
        ("artifact root", artifact_root),
        ("configuration input", config_path),
        ("inventory input", inventory_path),
    ):
        if _paths_overlap(output, source):
            raise RawReleasePackagingError(
                f"raw release output must not overlap the {label}"
            )


class _ChunkWriter(io.RawIOBase):
    def __init__(self, output: Path, label: str, chunk_bytes: int) -> None:
        self.output = output
        self.label = label
        self.chunk_bytes = chunk_bytes
        self.stream: BinaryIO | None = None
        self.chunk_hash: Any = None
        self.chunk_size = 0
        self.index = 0
        self.parts: list[dict[str, object]] = []
        self.aggregate = hashlib.sha256()
        self.total = 0

    def writable(self) -> bool:
        return True

    def _open(self) -> None:
        if self.index >= _MAX_PARTS:
            raise RawReleasePackagingError(
                "raw release exceeds the bounded part-count limit"
            )
        name = f"{self.label}.tar.gz.part-{self.index:0{_PART_INDEX_WIDTH}d}"
        self.stream = (self.output / name).open("xb")
        self.chunk_hash = hashlib.sha256()
        self.chunk_size = 0

    def _finish(self) -> None:
        if self.stream is None:
            return
        stream = self.stream
        chunk_hash = self.chunk_hash
        chunk_size = self.chunk_size
        self.stream = None
        self.chunk_hash = None
        self.chunk_size = 0
        name = Path(stream.name).name
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except OSError as error:
            raise RawReleasePackagingError(
                "cannot durably write raw release part"
            ) from error
        finally:
            stream.close()
        self.parts.append(
            {
                "path": name,
                "byte_size": chunk_size,
                "sha256": chunk_hash.hexdigest(),
            }
        )
        self.index += 1

    def write(self, data: bytes | bytearray) -> int:
        if self.closed:
            raise ValueError("cannot write to a closed chunk stream")
        view = memoryview(data)
        written = 0
        while written < len(view):
            if self.stream is None:
                self._open()
            capacity = self.chunk_bytes - self.chunk_size
            piece = view[written : written + capacity]
            assert self.stream is not None
            self.stream.write(piece)
            self.chunk_hash.update(piece)
            self.aggregate.update(piece)
            size = len(piece)
            self.chunk_size += size
            self.total += size
            written += size
            if self.chunk_size == self.chunk_bytes:
                self._finish()
        return written

    def flush(self) -> None:
        if self.stream is not None:
            self.stream.flush()

    def close(self) -> None:
        if not self.closed:
            self._finish()
        super().close()


class _DigestWriter:
    def __init__(self, target: BinaryIO) -> None:
        self.target = target
        self.digest = hashlib.sha256()
        self.total = 0

    def write(self, data: bytes) -> int:
        written = self.target.write(data)
        if written != len(data):
            raise RawReleasePackagingError("short write while creating raw archive")
        self.digest.update(data)
        self.total += written
        return written

    def flush(self) -> None:
        self.target.flush()


def _archive_members(
    root: Path,
    experiment_name: str,
    *,
    include_receipt: bool,
) -> Iterator[tuple[Path, str, os.stat_result]]:
    def visit(path: Path) -> Iterator[tuple[Path, str, os.stat_result]]:
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as error:
            raise RawReleasePackagingError(
                f"cannot inspect raw archive member: {path}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RawReleasePackagingError("raw archive contains a symbolic link")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISREG(metadata.st_mode):
            yield path, relative, metadata
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise RawReleasePackagingError("raw archive contains a non-regular member")
        yield path, relative, metadata
        try:
            with os.scandir(path) as stream:
                names = sorted(entry.name for entry in stream)
        except OSError as error:
            raise RawReleasePackagingError(
                f"cannot inspect raw archive directory: {path}"
            ) from error
        for name in names:
            child = path / name
            if name == ".run.lock":
                try:
                    lock_metadata = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise RawReleasePackagingError(
                        "cannot inspect excluded run lock"
                    ) from error
                if not stat.S_ISREG(lock_metadata.st_mode):
                    raise RawReleasePackagingError(
                        "excluded run lock is not a regular file"
                    )
                continue
            yield from visit(child)

    top_names = ["_frontiers", experiment_name]
    if include_receipt:
        top_names.append(REPRODUCIBILITY_OPERATIONAL_DIRECTORY)
    for top_name in top_names:
        yield from visit(root / top_name)


def _tar_info(
    relative: str,
    metadata: os.stat_result,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(relative)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if stat.S_ISDIR(metadata.st_mode):
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    elif stat.S_ISREG(metadata.st_mode):
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = metadata.st_size
    else:
        raise RawReleasePackagingError("raw archive member changed type")
    return info


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _open_source(path: Path, expected: os.stat_result) -> tuple[BinaryIO, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise RawReleasePackagingError(
            f"cannot safely open raw archive member: {path}"
        ) from error
    if not stat.S_ISREG(observed.st_mode) or not _same_file_state(expected, observed):
        os.close(descriptor)
        raise RawReleasePackagingError("raw archive member changed during packaging")
    return os.fdopen(descriptor, "rb", buffering=0), descriptor


def _write_archive(
    *,
    root: Path,
    experiment_name: str,
    output: Path,
    label: str,
    chunk_bytes: int,
    include_receipt: bool,
) -> tuple[_ChunkWriter, _DigestWriter]:
    writer = _ChunkWriter(output, label, chunk_bytes)
    tar_digest: _DigestWriter | None = None
    try:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=writer,
            mtime=0,
        ) as compressed:
            tar_digest = _DigestWriter(compressed)
            with tarfile.open(
                mode="w|",
                fileobj=tar_digest,
                format=tarfile.GNU_FORMAT,
            ) as archive:
                for path, relative, expected in _archive_members(
                    root,
                    experiment_name,
                    include_receipt=include_receipt,
                ):
                    info = _tar_info(relative, expected)
                    if info.isfile():
                        stream, descriptor = _open_source(path, expected)
                        with stream:
                            archive.addfile(info, stream)
                            after = os.fstat(descriptor)
                        if not _same_file_state(expected, after):
                            raise RawReleasePackagingError(
                                "raw archive member changed during packaging"
                            )
                    else:
                        archive.addfile(info)
    finally:
        writer.close()
    if tar_digest is None or not writer.parts:
        raise RawReleasePackagingError("raw release archive is empty")
    return writer, tar_digest


def _manifest_body(
    *,
    label: str,
    inventory: RawArtifactInventory,
    chunk_bytes: int,
    writer: _ChunkWriter,
    tar_digest: _DigestWriter,
) -> dict[str, object]:
    return {
        "artifact_type": ASSET_MANIFEST_TYPE,
        "schema_version": ASSET_MANIFEST_VERSION,
        "asset_label": label,
        "experiment_name": inventory.experiment_name,
        "config_hash": inventory.config_hash,
        "raw_inventory_hash": inventory.scientific_hash,
        "raw_inventory": inventory.to_dict(),
        "side": inventory.side,
        "archive_format": "deterministic-gnu-tar",
        "compression": "gzip-level-6-mtime-0",
        "excluded_files": [".run.lock"],
        "excluded_root_directories": [],
        "chunk_bytes": chunk_bytes,
        "tar_byte_size": tar_digest.total,
        "tar_sha256": tar_digest.digest.hexdigest(),
        "compressed_byte_size": writer.total,
        "compressed_sha256": writer.aggregate.hexdigest(),
        "parts": writer.parts,
        "reconstruction": _RECONSTRUCTION,
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise RawReleasePackagingError("cannot write raw release manifest") from error


def package_raw_release(
    *,
    config_path: str | Path,
    inventory_path: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    asset_label: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Path:
    """Authenticate and transactionally package one raw artifact root."""

    label = _safe_label(asset_label)
    chunk_limit = _positive_int(chunk_bytes, "chunk_bytes")
    config_source = _canonical_path(config_path, must_exist=True)
    inventory_source = _canonical_path(inventory_path, must_exist=True)
    root = _canonical_path(artifact_root, must_exist=True)
    output = _canonical_path(output_dir, must_exist=False)
    _validate_output_location(
        output,
        artifact_root=root,
        config_path=config_source,
        inventory_path=inventory_source,
    )
    if os.path.lexists(output):
        raise RawReleasePackagingError("raw release output directory must be absent")

    config = load_experiment_config(config_source)
    inventory = load_raw_artifact_inventory(inventory_source)
    inventory.verify(root, config)

    try:
        _durably_create_directory(output.parent)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.tmp-",
                dir=output.parent,
            )
        )
    except OSError as error:
        raise RawReleasePackagingError(
            "cannot create transactional raw release directory"
        ) from error

    try:
        writer, tar_digest = _write_archive(
            root=root,
            experiment_name=config.name,
            output=temporary,
            label=label,
            chunk_bytes=chunk_limit,
            include_receipt=inventory.execution_receipt is not None,
        )
        try:
            inventory.verify(root, config)
        except (OSError, ValueError) as error:
            raise RawReleasePackagingError(
                "raw artifact root changed during packaging"
            ) from error
        body = _manifest_body(
            label=label,
            inventory=inventory,
            chunk_bytes=chunk_limit,
            writer=writer,
            tar_digest=tar_digest,
        )
        manifest = {
            **body,
            "scientific_hash": scientific_hash(body, domain=_MANIFEST_DOMAIN),
        }
        temporary_manifest = temporary / f"{label}.manifest.json"
        _write_manifest(temporary_manifest, manifest)
        verify_asset_manifest(temporary_manifest)
        _fsync_directory(temporary)
        if os.path.lexists(output):
            raise RawReleasePackagingError(
                "raw release output appeared during packaging"
            )
        temporary.rename(output)
        _fsync_directory(output.parent)
    except BaseException:
        if os.path.lexists(temporary):
            shutil.rmtree(temporary)
        raise
    return output / f"{label}.manifest.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_text(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_MANIFEST_BYTES:
            os.close(descriptor)
            descriptor = None
            raise RawReleasePackagingError(
                "raw release manifest is not a bounded regular file"
            )
        with os.fdopen(descriptor, "rb") as stream:
            content = stream.read(_MAX_MANIFEST_BYTES + 1)
    except RawReleasePackagingError:
        raise
    except OSError as error:
        raise RawReleasePackagingError("cannot read raw release manifest") from error
    if len(content) > _MAX_MANIFEST_BYTES:
        raise RawReleasePackagingError("raw release manifest is too large")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RawReleasePackagingError("raw release manifest is not UTF-8") from error


def _load_manifest(path: Path) -> tuple[dict[str, Any], RawArtifactInventory]:
    content = _read_regular_text(path)
    try:
        raw = parse_json_strict(content, label="raw release asset manifest")
    except ValueError as error:
        raise RawReleasePackagingError(str(error)) from error
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise RawReleasePackagingError("raw release manifest fields are invalid")
    if (
        content
        != json.dumps(
            raw,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ):
        raise RawReleasePackagingError("raw release manifest is not canonical JSON")
    try:
        inventory = raw_artifact_inventory_from_dict(raw["raw_inventory"])
    except (TypeError, ValueError) as error:
        raise RawReleasePackagingError(
            "embedded raw artifact inventory is invalid"
        ) from error
    if (
        raw["artifact_type"] != ASSET_MANIFEST_TYPE
        or raw["schema_version"] != ASSET_MANIFEST_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["archive_format"] != "deterministic-gnu-tar"
        or raw["compression"] != "gzip-level-6-mtime-0"
        or raw["excluded_files"] != [".run.lock"]
        or raw["excluded_root_directories"] != []
        or raw["reconstruction"] != _RECONSTRUCTION
        or raw["experiment_name"] != inventory.experiment_name
        or raw["config_hash"] != inventory.config_hash
        or raw["raw_inventory_hash"] != inventory.scientific_hash
        or raw["side"] != inventory.side
        or not is_sha256(raw["compressed_sha256"])
        or not is_sha256(raw["tar_sha256"])
        or not isinstance(raw["parts"], list)
        or not raw["parts"]
        or len(raw["parts"]) > _MAX_PARTS
    ):
        raise RawReleasePackagingError("raw release manifest values are invalid")
    label = _safe_label(raw["asset_label"])
    _safe_label(raw["experiment_name"])
    if path.name != f"{label}.manifest.json":
        raise RawReleasePackagingError("raw release manifest filename is not canonical")
    chunk_bytes = _positive_int(raw["chunk_bytes"], "chunk_bytes")
    compressed_size = _positive_int(
        raw["compressed_byte_size"],
        "compressed_byte_size",
    )
    tar_size = _positive_int(raw["tar_byte_size"], "tar_byte_size")
    part_names: list[str] = []
    part_total = 0
    for index, part in enumerate(raw["parts"]):
        if (
            not isinstance(part, dict)
            or set(part) != _PART_FIELDS
            or part["path"] != (f"{label}.tar.gz.part-{index:0{_PART_INDEX_WIDTH}d}")
            or isinstance(part["byte_size"], bool)
            or not isinstance(part["byte_size"], int)
            or part["byte_size"] < 1
            or part["byte_size"] > chunk_bytes
            or (index < len(raw["parts"]) - 1 and part["byte_size"] != chunk_bytes)
            or not is_sha256(part["sha256"])
        ):
            raise RawReleasePackagingError("raw release part record is invalid")
        part_names.append(part["path"])
        part_total += part["byte_size"]
    if len(set(part_names)) != len(part_names) or part_total != compressed_size:
        raise RawReleasePackagingError("raw release part inventory is invalid")
    body = {name: raw[name] for name in raw if name != "scientific_hash"}
    if not is_sha256(raw["scientific_hash"]) or raw[
        "scientific_hash"
    ] != scientific_hash(body, domain=_MANIFEST_DOMAIN):
        raise RawReleasePackagingError("raw release manifest hash is invalid")
    if tar_size % tarfile.RECORDSIZE:
        raise RawReleasePackagingError("raw release tar byte size is not canonical")
    return raw, inventory


class _ChunkReader(io.RawIOBase):
    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        try:
            self.directory_fd = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError as error:
            raise RawReleasePackagingError(
                "cannot open raw release package directory"
            ) from error
        self.parts = manifest["parts"]
        self.expected_total = manifest["compressed_byte_size"]
        self.expected_hash = manifest["compressed_sha256"]
        self.index = 0
        self.stream: BinaryIO | None = None
        self.current_hash: Any = None
        self.current_size = 0
        self.aggregate = hashlib.sha256()
        self.total = 0
        self.complete = False

    def readable(self) -> bool:
        return True

    def _open_part(self) -> None:
        part = self.parts[self.index]
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                part["path"],
                flags,
                dir_fd=self.directory_fd,
            )
            metadata = os.fstat(descriptor)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise RawReleasePackagingError(
                "cannot safely open raw release part"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != part["byte_size"]:
            os.close(descriptor)
            raise RawReleasePackagingError(
                "raw release part bytes do not match the manifest"
            )
        self.stream = os.fdopen(descriptor, "rb", buffering=0)
        self.current_hash = hashlib.sha256()
        self.current_size = 0

    def _finish_part(self) -> None:
        assert self.stream is not None
        part = self.parts[self.index]
        if self.stream.read(1):
            raise RawReleasePackagingError(
                "raw release part bytes do not match the manifest"
            )
        metadata = os.fstat(self.stream.fileno())
        self.stream.close()
        self.stream = None
        if (
            metadata.st_size != part["byte_size"]
            or self.current_size != part["byte_size"]
            or self.current_hash.hexdigest() != part["sha256"]
        ):
            raise RawReleasePackagingError(
                "raw release part bytes do not match the manifest"
            )
        self.current_hash = None
        self.current_size = 0
        self.index += 1

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self.index < len(self.parts):
            if self.stream is None:
                self._open_part()
            part = self.parts[self.index]
            remaining = part["byte_size"] - self.current_size
            if remaining == 0:
                self._finish_part()
                continue
            assert self.stream is not None
            count = self.stream.readinto(view[written : written + remaining])
            if not count:
                raise RawReleasePackagingError(
                    "raw release part bytes do not match the manifest"
                )
            piece = view[written : written + count]
            self.current_hash.update(piece)
            self.aggregate.update(piece)
            self.current_size += count
            self.total += count
            written += count
        if self.index == len(self.parts) and self.stream is None:
            self.complete = True
        return written

    def finish(self) -> None:
        scratch = bytearray(_STREAM_BYTES)
        while self.readinto(scratch):
            pass
        if (
            not self.complete
            or self.total != self.expected_total
            or self.aggregate.hexdigest() != self.expected_hash
        ):
            raise RawReleasePackagingError(
                "raw release compressed stream does not match the manifest"
            )

    def close(self) -> None:
        if not self.closed:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            os.close(self.directory_fd)
        super().close()


class _GzipReader(io.RawIOBase):
    def __init__(self, source: _ChunkReader) -> None:
        self.source = source
        self.decompressor = zlib.decompressobj(wbits=31)
        self.output = bytearray()
        self.header = bytearray()
        self.finished = False

    def readable(self) -> bool:
        return True

    def _compressed_block(self) -> bytes:
        block = self.source.read(_STREAM_BYTES)
        if block and len(self.header) < len(_GZIP_HEADER):
            needed = len(_GZIP_HEADER) - len(self.header)
            self.header.extend(block[:needed])
            if (
                len(self.header) == len(_GZIP_HEADER)
                and bytes(self.header) != _GZIP_HEADER
            ):
                raise RawReleasePackagingError(
                    "raw release gzip header is not canonical"
                )
        return block

    def _pump(self) -> None:
        if self.finished:
            return
        block = (
            self.decompressor.unconsumed_tail
            if self.decompressor.unconsumed_tail
            else self._compressed_block()
        )
        if not block:
            raise RawReleasePackagingError("raw release gzip stream is incomplete")
        try:
            produced = self.decompressor.decompress(block, _STREAM_BYTES)
        except zlib.error as error:
            raise RawReleasePackagingError(
                "raw release gzip stream is invalid"
            ) from error
        self.output.extend(produced)
        if self.decompressor.eof:
            if self.decompressor.unused_data or self.source.read(1):
                raise RawReleasePackagingError(
                    "raw release gzip stream has trailing data"
                )
            self.source.finish()
            flushed = self.decompressor.flush()
            if len(flushed) > _STREAM_BYTES:
                raise RawReleasePackagingError("raw release gzip stream is not bounded")
            self.output.extend(flushed)
            self.finished = True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer).cast("B")
        while not self.output and not self.finished:
            self._pump()
        count = min(len(view), len(self.output))
        if count:
            view[:count] = self.output[:count]
            del self.output[:count]
        return count

    def close(self) -> None:
        if not self.closed:
            self.source.close()
        super().close()


class _DigestReader(io.RawIOBase):
    def __init__(self, source: _GzipReader) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.total = 0
        self.tail = bytearray()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        count = self.source.readinto(buffer)
        if count:
            piece = bytes(memoryview(buffer)[:count])
            self.digest.update(piece)
            self.total += count
            self.tail.extend(piece)
            if len(self.tail) > 2 * tarfile.RECORDSIZE:
                del self.tail[: -2 * tarfile.RECORDSIZE]
        return count

    def finish(
        self,
        *,
        expected_size: int,
        expected_hash: str,
        content_end: int,
    ) -> None:
        scratch = bytearray(_STREAM_BYTES)
        while self.readinto(scratch):
            pass
        canonical_size = (
            (content_end + 2 * tarfile.BLOCKSIZE + tarfile.RECORDSIZE - 1)
            // tarfile.RECORDSIZE
            * tarfile.RECORDSIZE
        )
        zero_suffix = self.total - content_end
        if (
            self.total != expected_size
            or self.total != canonical_size
            or self.digest.hexdigest() != expected_hash
            or zero_suffix < 2 * tarfile.BLOCKSIZE
            or zero_suffix > len(self.tail)
            or any(self.tail[-zero_suffix:])
        ):
            raise RawReleasePackagingError("raw release tar stream is not canonical")

    def close(self) -> None:
        if not self.closed:
            self.source.close()
        super().close()


class _TreeState:
    def __init__(self) -> None:
        self.digest = hashlib.sha256(_TREE_HASH_PREFIX)
        self.file_count = 0
        self.byte_size = 0
        self.last_path: str | None = None
        self.root_seen = False

    def add(self, relative: str, member: tarfile.TarInfo, stream: BinaryIO) -> None:
        if self.last_path is not None and relative <= self.last_path:
            raise RawReleasePackagingError(
                "raw release tree members are not canonically ordered"
            )
        self.last_path = relative
        encoded = relative.encode("utf-8")
        self.digest.update(len(encoded).to_bytes(8, "big"))
        self.digest.update(encoded)
        self.digest.update(member.size.to_bytes(8, "big"))
        remaining = member.size
        while remaining:
            content = stream.read(min(_STREAM_BYTES, remaining))
            if not content:
                raise RawReleasePackagingError("raw release tar member is truncated")
            self.digest.update(content)
            self.byte_size += len(content)
            remaining -= len(content)
        if stream.read(1):
            raise RawReleasePackagingError(
                "raw release tar member exceeds its declared size"
            )
        self.file_count += 1


class _DirectoryState:
    def __init__(self, parts: tuple[str, ...]) -> None:
        self.parts = parts
        self.has_child = False


def _member_path(name: object) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise RawReleasePackagingError("raw release tar path is unsafe")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawReleasePackagingError("raw release tar path is unsafe")
    return path.parts


def _canonical_member_metadata(member: tarfile.TarInfo) -> bool:
    expected_mode = 0o755 if member.type == tarfile.DIRTYPE else 0o644
    return (
        member.type in {tarfile.DIRTYPE, tarfile.REGTYPE}
        and member.mode == expected_mode
        and member.uid == 0
        and member.gid == 0
        and member.uname == ""
        and member.gname == ""
        and member.mtime == 0
        and member.linkname == ""
        and member.devmajor == 0
        and member.devminor == 0
        and not member.pax_headers
        and not member.sparse
        and (member.type != tarfile.DIRTYPE or member.size == 0)
    )


def _verify_tar_stream(
    stream: _DigestReader,
    manifest: dict[str, Any],
    inventory: RawArtifactInventory,
) -> None:
    expected = {tree.path: tree for tree in inventory.trees}
    states = {path: _TreeState() for path in expected}
    directory_stack: list[_DirectoryState] = []
    top_directories_seen: set[str] = set()
    previous_order: tuple[int, tuple[str, ...]] | None = None
    content_end = 0
    top_names = {"_frontiers", inventory.experiment_name}
    receipt = inventory.execution_receipt
    if receipt is not None:
        top_names.add(REPRODUCIBILITY_OPERATIONAL_DIRECTORY)
    receipt_member = (
        f"{REPRODUCIBILITY_OPERATIONAL_DIRECTORY}/{EXECUTION_RECEIPT_FILENAME}"
    )
    receipt_seen = False
    try:
        with tarfile.open(mode="r|", fileobj=stream) as archive:
            for member in archive:
                archive.members.clear()
                parts = _member_path(member.name)
                if parts[0] not in top_names or ".run.lock" in parts:
                    raise RawReleasePackagingError(
                        "raw release tar contains an unexpected path"
                    )
                top_order = {
                    "_frontiers": 0,
                    inventory.experiment_name: 1,
                    REPRODUCIBILITY_OPERATIONAL_DIRECTORY: 2,
                }[parts[0]]
                order = (top_order, tuple(parts[1:]))
                if previous_order is not None and order <= previous_order:
                    raise RawReleasePackagingError(
                        "raw release tar members are not canonically ordered"
                    )
                previous_order = order
                if not _canonical_member_metadata(member):
                    raise RawReleasePackagingError(
                        "raw release tar metadata is not canonical"
                    )
                while directory_stack and (
                    len(parts) <= len(directory_stack[-1].parts)
                    or parts[: len(directory_stack[-1].parts)]
                    != directory_stack[-1].parts
                ):
                    completed = directory_stack.pop()
                    if not completed.has_child:
                        raise RawReleasePackagingError(
                            "raw release tar contains an empty directory"
                        )
                if len(parts) > 1:
                    if not directory_stack or directory_stack[-1].parts != parts[:-1]:
                        raise RawReleasePackagingError(
                            "raw release tar omits a parent directory"
                        )
                    directory_stack[-1].has_child = True
                if len(parts) == 1:
                    if member.type != tarfile.DIRTYPE:
                        raise RawReleasePackagingError(
                            "raw release top-level member is not a directory"
                        )
                    top_directories_seen.add(member.name)
                    directory_stack.append(_DirectoryState(parts))
                    content_end = member.offset_data
                    continue
                if parts[0] == REPRODUCIBILITY_OPERATIONAL_DIRECTORY:
                    if (
                        receipt is None
                        or member.name != receipt_member
                        or member.type != tarfile.REGTYPE
                    ):
                        raise RawReleasePackagingError(
                            "raw release receipt inventory is invalid"
                        )
                    expected_receipt = receipt.to_json().encode("utf-8")
                    extracted = archive.extractfile(member)
                    if (
                        extracted is None
                        or member.size != len(expected_receipt)
                        or extracted.read(len(expected_receipt) + 1) != expected_receipt
                    ):
                        raise RawReleasePackagingError(
                            "raw release receipt bytes are invalid"
                        )
                    receipt_seen = True
                    content_end = member.offset_data + (
                        (member.size + tarfile.BLOCKSIZE - 1)
                        // tarfile.BLOCKSIZE
                        * tarfile.BLOCKSIZE
                    )
                    continue
                tree_path = "/".join(parts[:2])
                tree = expected.get(tree_path)
                if tree is None or (
                    (parts[0] == "_frontiers") != (tree.tree_type == "frontier")
                ):
                    raise RawReleasePackagingError(
                        "raw release tar tree inventory is invalid"
                    )
                if member.type == tarfile.DIRTYPE:
                    if len(parts) == 2:
                        states[tree_path].root_seen = True
                    directory_stack.append(_DirectoryState(parts))
                    content_end = member.offset_data
                    continue
                if len(parts) < 3:
                    raise RawReleasePackagingError(
                        "raw release tree root is not a directory"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RawReleasePackagingError("cannot read raw release tar member")
                states[tree_path].add(
                    "/".join(parts[2:]),
                    member,
                    extracted,
                )
                content_end = member.offset_data + (
                    (member.size + tarfile.BLOCKSIZE - 1)
                    // tarfile.BLOCKSIZE
                    * tarfile.BLOCKSIZE
                )
    except RawReleasePackagingError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise RawReleasePackagingError("raw release tar stream is invalid") from error

    while directory_stack:
        completed = directory_stack.pop()
        if not completed.has_child:
            raise RawReleasePackagingError(
                "raw release tar contains an empty directory"
            )

    stream.finish(
        expected_size=manifest["tar_byte_size"],
        expected_hash=manifest["tar_sha256"],
        content_end=content_end,
    )
    if top_directories_seen != top_names:
        raise RawReleasePackagingError(
            "raw release tar has missing or empty directories"
        )
    if receipt_seen != (receipt is not None):
        raise RawReleasePackagingError(
            "raw release tar receipt does not match the embedded inventory"
        )
    for path, tree in expected.items():
        state = states[path]
        if (
            not state.root_seen
            or state.file_count != tree.file_count
            or state.byte_size != tree.byte_size
            or state.digest.hexdigest() != tree.tree_byte_sha256
        ):
            raise RawReleasePackagingError(
                "raw release tar bytes do not match the embedded inventory"
            )


def verify_asset_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Stream-verify the manifest, chunks, gzip/tar, and embedded inventory."""

    path = Path(manifest_path)
    manifest, inventory = _load_manifest(path)
    directory = path.parent
    expected_files = {
        path.name,
        *(part["path"] for part in manifest["parts"]),
    }
    try:
        with os.scandir(directory) as entries:
            observed = tuple(entries)
    except OSError as error:
        raise RawReleasePackagingError("cannot inspect raw release package") from error
    if {entry.name for entry in observed} != expected_files or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False)
        for entry in observed
    ):
        raise RawReleasePackagingError(
            "raw release package inventory does not match the manifest"
        )

    chunks = _ChunkReader(directory, manifest)
    compressed = _GzipReader(chunks)
    tar_stream = _DigestReader(compressed)
    try:
        _verify_tar_stream(tar_stream, manifest, inventory)
    finally:
        tar_stream.close()
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("config", type=Path)
    package.add_argument("inventory", type=Path)
    package.add_argument("artifact_root", type=Path)
    package.add_argument("output_dir", type=Path)
    package.add_argument("asset_label")
    package.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "package":
        path = package_raw_release(
            config_path=arguments.config,
            inventory_path=arguments.inventory,
            artifact_root=arguments.artifact_root,
            output_dir=arguments.output_dir,
            asset_label=arguments.asset_label,
            chunk_bytes=arguments.chunk_bytes,
        )
        manifest = verify_asset_manifest(path)
        print(
            json.dumps(
                {
                    "manifest": str(path),
                    "parts": len(manifest["parts"]),
                    "scientific_hash": manifest["scientific_hash"],
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = verify_asset_manifest(arguments.manifest)
    print(
        json.dumps(
            {
                "parts": len(manifest["parts"]),
                "scientific_hash": manifest["scientific_hash"],
                "valid": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
