from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import tarfile
import tracemalloc
from pathlib import Path, PurePosixPath

import pytest

import scripts.package_raw_release as raw_release
from infinite_rulebook.orchestration.config import (
    AgentConfig,
    AgentKind,
    CheckpointConfig,
    EnvironmentConfig,
    EnvironmentKind,
    ExperimentConfig,
)
from infinite_rulebook.orchestration.hashing import scientific_hash
from infinite_rulebook.orchestration.inventory import (
    RAW_ARTIFACT_INVENTORY_FORMAT,
    RAW_ARTIFACT_INVENTORY_VERSION,
    RawArtifactInventory,
    RawArtifactTree,
)
from infinite_rulebook.orchestration.reproducibility import (
    REPRODUCIBILITY_OPERATIONAL_DIRECTORY,
    run_reproducibility_check,
)
from infinite_rulebook.orchestration.run import RunExecutor
from infinite_rulebook.orchestration.symbolic import ExactSymbolicAdapter
from scripts.package_raw_release import (
    RawReleasePackagingError,
    package_raw_release,
    verify_asset_manifest,
)

_MANIFEST_DOMAIN = "raw-release-asset-manifest"


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = ExperimentConfig(
        name="raw-release-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="raw-release-test",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), sort_keys=True),
        encoding="utf-8",
    )
    root = tmp_path / "raw"
    RunExecutor(root, ExactSymbolicAdapter).execute(config, config.cells()[0])
    inventory = RawArtifactInventory.create(root, config, side="serial")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(inventory.to_json(), encoding="utf-8")
    return config_path, inventory_path, root


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    body = {
        name: value for name, value in manifest.items() if name != "scientific_hash"
    }
    manifest["scientific_hash"] = scientific_hash(
        body,
        domain=_MANIFEST_DOMAIN,
    )
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_chunk_fsync_failure_preserves_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = raw_release._ChunkWriter(tmp_path, "durability", 1024)
    writer.write(b"content")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(raw_release.os, "fsync", fail_fsync)
    with pytest.raises(RawReleasePackagingError, match="durably write") as captured:
        writer.close()
    assert isinstance(captured.value.__cause__, OSError)
    assert writer.stream is None
    writer.close()


def test_tar_verification_memory_is_bounded_by_depth_not_member_count() -> None:
    experiment_name = "raw-release-scale-test"
    frontier_hash = scientific_hash("frontier", domain="raw-release-scale")
    run_hash = scientific_hash("run", domain="raw-release-scale")
    frontier_files = tuple(
        (f"file-{index:05d}.json", f'{{"index":{index}}}\n'.encode())
        for index in range(12_000)
    )
    run_files = (("manifest.json", b'{"complete":true}\n'),)

    def tree_record(
        tree_type: str,
        path: str,
        identity_hash: str,
        files: tuple[tuple[str, bytes], ...],
    ) -> RawArtifactTree:
        digest = hashlib.sha256(raw_release._TREE_HASH_PREFIX)
        for relative, content in files:
            encoded = relative.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return RawArtifactTree(
            tree_type=tree_type,
            path=path,
            identity_hash=identity_hash,
            scientific_content_hash=scientific_hash(
                path,
                domain="raw-release-scale-content",
            ),
            file_count=len(files),
            byte_size=sum(len(content) for _, content in files),
            tree_byte_sha256=digest.hexdigest(),
        )

    trees = (
        tree_record(
            "frontier",
            f"_frontiers/{frontier_hash}",
            frontier_hash,
            frontier_files,
        ),
        tree_record(
            "run",
            f"{experiment_name}/{run_hash}",
            run_hash,
            run_files,
        ),
    )
    config_hash = scientific_hash("config", domain="raw-release-scale")
    body = {
        "artifact_type": RAW_ARTIFACT_INVENTORY_FORMAT,
        "schema_version": RAW_ARTIFACT_INVENTORY_VERSION,
        "experiment_name": experiment_name,
        "config_hash": config_hash,
        "side": "serial",
        "execution_receipt": None,
        "trees": [tree.to_dict() for tree in trees],
    }
    inventory = RawArtifactInventory(
        experiment_name=experiment_name,
        config_hash=config_hash,
        side="serial",
        execution_receipt=None,
        trees=trees,
        scientific_hash=scientific_hash(
            body,
            domain="raw-artifact-inventory",
        ),
    )

    tar_output = io.BytesIO()
    with tarfile.open(
        fileobj=tar_output,
        mode="w:",
        format=tarfile.GNU_FORMAT,
    ) as archive:
        for directory in (
            "_frontiers",
            f"_frontiers/{frontier_hash}",
        ):
            archive.addfile(_canonical_tar_member(directory, directory=True))
        for relative, content in frontier_files:
            member = _canonical_tar_member(
                f"_frontiers/{frontier_hash}/{relative}",
                size=len(content),
            )
            archive.addfile(member, io.BytesIO(content))
        for directory in (
            experiment_name,
            f"{experiment_name}/{run_hash}",
        ):
            archive.addfile(_canonical_tar_member(directory, directory=True))
        for relative, content in run_files:
            member = _canonical_tar_member(
                f"{experiment_name}/{run_hash}/{relative}",
                size=len(content),
            )
            archive.addfile(member, io.BytesIO(content))
    tar_bytes = tar_output.getvalue()
    manifest = {
        "tar_byte_size": len(tar_bytes),
        "tar_sha256": hashlib.sha256(tar_bytes).hexdigest(),
    }
    stream = raw_release._DigestReader(io.BytesIO(tar_bytes))

    tracemalloc.start()
    try:
        raw_release._verify_tar_stream(stream, manifest, inventory)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        stream.close()
    assert peak < 6_000_000


def _canonical_tar_member(
    name: str,
    *,
    directory: bool = False,
    size: int = 0,
) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    member.mode = 0o755 if directory else 0o644
    member.size = size
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    member.pax_headers = {}
    return member


def _rewrite_tar(
    manifest_path: Path,
    mutation: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compressed = b"".join(
        (manifest_path.parent / part["path"]).read_bytes() for part in manifest["parts"]
    )
    source = io.BytesIO(gzip.decompress(compressed))
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(fileobj=source, mode="r:") as archive:
        for member in archive:
            stream = archive.extractfile(member) if member.isfile() else None
            members.append(
                (
                    copy.copy(member),
                    stream.read() if stream is not None else None,
                )
            )

    regular_index = next(
        index for index, (member, _) in enumerate(members) if member.isfile()
    )
    if mutation == "unrelated":
        members[0][0].name = "unrelated"
    elif mutation == "traversal":
        members[regular_index][0].name = "../escape"
    elif mutation == "symlink":
        member = members[regular_index][0]
        member.type = tarfile.SYMTYPE
        member.linkname = "../escape"
        member.size = 0
        members[regular_index] = (member, None)
    elif mutation == "metadata":
        members[regular_index][0].uid = 7
    elif mutation == "missing":
        members.pop(regular_index)
    elif mutation == "duplicate":
        member, content = members[regular_index]
        members.append((copy.copy(member), content))
    elif mutation == "extra":
        member, content = members[regular_index]
        extra = copy.copy(member)
        extra.name = f"{PurePosixPath(member.name).parent.as_posix()}/zz-extra.json"
        members.append((extra, content))
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    tar_output = io.BytesIO()
    with tarfile.open(
        fileobj=tar_output,
        mode="w",
        format=tarfile.GNU_FORMAT,
    ) as archive:
        for member, content in members:
            archive.addfile(
                member,
                io.BytesIO(content) if content is not None else None,
            )
    tar_bytes = tar_output.getvalue()
    compressed_output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=6,
        fileobj=compressed_output,
        mtime=0,
    ) as archive:
        archive.write(tar_bytes)
    compressed_bytes = compressed_output.getvalue()

    for part in manifest["parts"]:
        (manifest_path.parent / part["path"]).unlink()
    part_name = f"{manifest['asset_label']}.tar.gz.part-00000000"
    (manifest_path.parent / part_name).write_bytes(compressed_bytes)
    manifest["chunk_bytes"] = max(manifest["chunk_bytes"], len(compressed_bytes))
    manifest["tar_byte_size"] = len(tar_bytes)
    manifest["tar_sha256"] = hashlib.sha256(tar_bytes).hexdigest()
    manifest["compressed_byte_size"] = len(compressed_bytes)
    manifest["compressed_sha256"] = hashlib.sha256(compressed_bytes).hexdigest()
    manifest["parts"] = [
        {
            "path": part_name,
            "byte_size": len(compressed_bytes),
            "sha256": hashlib.sha256(compressed_bytes).hexdigest(),
        }
    ]
    _write_manifest(manifest_path, manifest)


def test_raw_release_archive_is_deterministic_and_chunked(tmp_path: Path) -> None:
    config, inventory, root = _inputs(tmp_path)
    first = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "first",
        asset_label="raw-test",
        chunk_bytes=1024,
    )
    second = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "second",
        asset_label="raw-test",
        chunk_bytes=1024,
    )

    left = verify_asset_manifest(first)
    right = verify_asset_manifest(second)
    assert left == right
    assert len(left["parts"]) > 1
    for part in left["parts"]:
        assert (first.parent / part["path"]).read_bytes() == (
            second.parent / part["path"]
        ).read_bytes()


def test_raw_release_archive_consumes_member_inventory_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    del config, inventory
    members = list(
        raw_release._archive_members(
            root,
            "raw-release-test",
            include_receipt=False,
        )
    )
    first_relative = members[0][1]
    first_processed = False
    original_tar_info = raw_release._tar_info

    def streaming_members(*args: object, **kwargs: object):
        del args, kwargs
        yield members[0]
        assert first_processed
        yield from members[1:]

    def observed_tar_info(relative: str, metadata: object):
        nonlocal first_processed
        if relative == first_relative:
            first_processed = True
        return original_tar_info(relative, metadata)  # type: ignore[arg-type]

    monkeypatch.setattr(raw_release, "_archive_members", streaming_members)
    monkeypatch.setattr(raw_release, "_tar_info", observed_tar_info)
    output = tmp_path / "streamed"
    output.mkdir()

    writer, _ = raw_release._write_archive(
        root=root,
        experiment_name="raw-release-test",
        output=output,
        label="raw-test",
        chunk_bytes=4096,
        include_receipt=False,
    )

    assert first_processed
    assert writer.parts


def test_raw_release_verification_rejects_tampering_missing_and_extra(
    tmp_path: Path,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    manifest_path = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "assets",
        asset_label="raw-test",
        chunk_bytes=4096,
    )
    manifest = verify_asset_manifest(manifest_path)
    first_part = manifest_path.parent / manifest["parts"][0]["path"]
    first_part.write_bytes(first_part.read_bytes() + b"tampered")
    with pytest.raises(RawReleasePackagingError, match="part bytes"):
        verify_asset_manifest(manifest_path)

    first_part.write_bytes(first_part.read_bytes()[: -len(b"tampered")])
    extra = manifest_path.parent / "extra"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(RawReleasePackagingError, match="inventory"):
        verify_asset_manifest(manifest_path)


def test_raw_release_refuses_overwrite_and_unsafe_labels(tmp_path: Path) -> None:
    config, inventory, root = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RawReleasePackagingError, match="must be absent"):
        package_raw_release(
            config_path=config,
            inventory_path=inventory,
            artifact_root=root,
            output_dir=output,
            asset_label="raw-test",
        )
    with pytest.raises(RawReleasePackagingError, match="safe filename"):
        package_raw_release(
            config_path=config,
            inventory_path=inventory,
            artifact_root=root,
            output_dir=tmp_path / "unsafe",
            asset_label="../unsafe",
        )


@pytest.mark.parametrize(
    "output_kind",
    ("root", "nested", "ancestor", "config", "inventory-child"),
)
def test_raw_release_rejects_output_overlap(
    tmp_path: Path,
    output_kind: str,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    output = {
        "root": root,
        "nested": root / "release",
        "ancestor": tmp_path,
        "config": config,
        "inventory-child": inventory / "release",
    }[output_kind]

    with pytest.raises(RawReleasePackagingError, match="overlap"):
        package_raw_release(
            config_path=config,
            inventory_path=inventory,
            artifact_root=root,
            output_dir=output,
            asset_label="raw-test",
        )


def test_manifest_embeds_the_exact_canonical_raw_inventory(
    tmp_path: Path,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    manifest_path = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "assets",
        asset_label="raw-test",
    )

    manifest = verify_asset_manifest(manifest_path)
    expected = json.loads(inventory.read_text(encoding="utf-8"))
    assert manifest["raw_inventory"] == expected
    assert manifest["raw_inventory_hash"] == expected["scientific_hash"]
    assert manifest["excluded_root_directories"] == []


def test_receipt_bearing_archive_extracts_to_a_verifiable_raw_root(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="raw-release-receipt-test",
        environments=(EnvironmentConfig(EnvironmentKind.IND, projection_size=1),),
        agents=(AgentConfig(AgentKind.REWARD, target_size=1),),
        checkpoints=CheckpointConfig((0, 1)),
        horizon=1,
        master_seed="raw-release-receipt-test",
    )
    config_path = tmp_path / "receipt-config.json"
    config_path.write_text(
        json.dumps(config.resolved_dict(), sort_keys=True),
        encoding="utf-8",
    )
    report = run_reproducibility_check(
        config,
        serial_root=tmp_path / "receipt-serial",
        parallel_root=tmp_path / "receipt-parallel",
        parallel_workers=2,
    )
    inventory = RawArtifactInventory.create(
        report.serial_root,
        config,
        side="serial",
    )
    inventory_path = tmp_path / "receipt-inventory.json"
    inventory_path.write_text(inventory.to_json(), encoding="utf-8")
    manifest_path = package_raw_release(
        config_path=config_path,
        inventory_path=inventory_path,
        artifact_root=report.serial_root,
        output_dir=tmp_path / "receipt-assets",
        asset_label="receipt-raw",
        chunk_bytes=4096,
    )
    manifest = verify_asset_manifest(manifest_path)
    compressed = b"".join(
        (manifest_path.parent / part["path"]).read_bytes() for part in manifest["parts"]
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(
        fileobj=io.BytesIO(gzip.decompress(compressed)),
        mode="r:",
    ) as archive:
        names = []
        for member in archive:
            names.append(member.name)
            target = extracted / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                target.write_bytes(stream.read())

    assert f"{REPRODUCIBILITY_OPERATIONAL_DIRECTORY}/execution-receipt.json" in names
    inventory.verify(extracted, config, side="serial")


def test_self_hashed_unrelated_archive_is_rejected(tmp_path: Path) -> None:
    config, inventory, root = _inputs(tmp_path)
    manifest_path = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "assets",
        asset_label="raw-test",
    )
    _rewrite_tar(manifest_path, "unrelated")

    with pytest.raises(RawReleasePackagingError):
        verify_asset_manifest(manifest_path)


@pytest.mark.parametrize(
    "mutation",
    ("traversal", "symlink", "metadata", "missing", "duplicate", "extra"),
)
def test_tar_stream_rejects_unsafe_noncanonical_or_wrong_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    manifest_path = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "assets",
        asset_label="raw-test",
    )
    _rewrite_tar(manifest_path, mutation)

    with pytest.raises(RawReleasePackagingError):
        verify_asset_manifest(manifest_path)


def test_source_mutation_is_detected_and_partial_output_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    original_verify = RawArtifactInventory.verify
    calls = 0

    def mutate_after_initial_verification(
        self: RawArtifactInventory,
        artifact_root: str | Path,
        experiment: ExperimentConfig,
        *,
        side: str | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_verify(
                self,
                artifact_root,
                experiment,
                side=side,
            )
            target = next(
                path for path in root.rglob("*.json") if path.name != ".run.lock"
            )
            target.chmod(0o644)
            target.write_text(
                target.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        RawArtifactInventory,
        "verify",
        mutate_after_initial_verification,
    )
    output = tmp_path / "assets"
    with pytest.raises(RawReleasePackagingError, match="embedded inventory"):
        package_raw_release(
            config_path=config,
            inventory_path=inventory,
            artifact_root=root,
            output_dir=output,
            asset_label="raw-test",
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".assets.tmp-*"))


def test_verification_streams_parts_without_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, inventory, root = _inputs(tmp_path)
    manifest_path = package_raw_release(
        config_path=config,
        inventory_path=inventory,
        artifact_root=root,
        output_dir=tmp_path / "assets",
        asset_label="raw-test",
        chunk_bytes=1024,
    )
    original_read_bytes = Path.read_bytes

    def reject_part_read_bytes(path: Path) -> bytes:
        if ".tar.gz.part-" in path.name:
            raise AssertionError("part verification must be streaming")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_part_read_bytes)
    assert verify_asset_manifest(manifest_path)["parts"]
