"""Adversarial tests for the private matched-v3 atomic content primitive.

These tests publish only tiny synthetic byte strings.  They execute no candidate,
network operation, container, scientific protocol, or evidence workflow.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    _forager_matched_v3_atomic_publication as atomic,
)

pytestmark = pytest.mark.integration

_ADDRESS = hashlib.sha256(b"caller-carried-publication-address").hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "publication-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    return parent.resolve(strict=True)


def _payload_case() -> tuple[dict[str, bytes], tuple[atomic.ExactFileRecord, ...]]:
    payloads = {
        "manifest.json": b'{"classification":"synthetic"}',
        "empty.bin": b"",
        "trace.bin": b"\x01\x00\xff\x01",
    }
    # Deliberately not sorted: normalization must be deterministic and name-based.
    records = tuple(
        atomic.ExactFileRecord(name, len(raw), _sha256(raw))
        for name, raw in payloads.items()
    )
    return payloads, records


def _publish(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, bytes],
    tuple[atomic.ExactFileRecord, ...],
    atomic.ContentVerifiedFlatPublication,
]:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    result = atomic.publish_exact_flat_publication(
        parent,
        address=_ADDRESS,
        expected_files=records,
        payloads=payloads,
    )
    return parent, payloads, records, result


def _staging_names(parent: Path) -> list[str]:
    return sorted(
        child.name
        for child in parent.iterdir()
        if child.name.startswith(".forager-matched-v3-atomic-partial-")
    )


def test_descriptor_is_private_content_only_and_non_authorizing() -> None:
    descriptor = atomic.atomic_publication_descriptor()
    assert descriptor["status"] == "private_content_only_primitive"
    assert descriptor["classification"] == (
        "private_content_only_non_authorizing_primitive"
    )
    assert descriptor["scope"] == {
        "layout": "one_sha256_named_directory_with_exact_flat_files",
        "address_semantics": "externally_carried_opaque_sha256_namespace_key",
        "publisher_specific_semantics": False,
        "default_publication_root": False,
        "recursive_deletion": False,
        "network_process_or_workload_execution": False,
    }
    assert all(value is False for value in descriptor["claims"].values())
    raw = atomic.canonical_atomic_publication_descriptor_bytes()
    assert _sha256(raw) == atomic.ATOMIC_PUBLICATION_DESCRIPTOR_SHA256


def test_publish_and_reload_exact_inventory(tmp_path: Path) -> None:
    parent, payloads, records, published = _publish(tmp_path)
    root = parent / _ADDRESS

    assert published.root == root
    assert published.address == _ADDRESS
    assert published.records == tuple(sorted(records, key=lambda item: item.name))
    assert dict(published.files) == payloads
    assert isinstance(published.files, MappingProxyType)
    with pytest.raises(TypeError):
        published.files["new.bin"] = b"forbidden"  # type: ignore[index]

    assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert set(child.name for child in root.iterdir()) == set(payloads)
    for child in root.iterdir():
        metadata = child.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1

    loaded = atomic.load_exact_flat_publication(
        parent,
        address=_ADDRESS,
        expected_files=records,
    )
    assert loaded == published
    assert _staging_names(parent) == []


def test_zero_length_file_is_written_fsynced_and_reloaded(tmp_path: Path) -> None:
    parent = _safe_parent(tmp_path)
    payloads = {"zero.bin": b""}
    records = (atomic.ExactFileRecord("zero.bin", 0, _sha256(b"")),)

    published = atomic.publish_exact_flat_publication(
        parent,
        address=_ADDRESS,
        expected_files=records,
        payloads=payloads,
    )

    assert (published.root / "zero.bin").stat().st_size == 0
    assert atomic.load_exact_flat_publication(
        parent,
        address=_ADDRESS,
        expected_files=records,
    ).files["zero.bin"] == b""


@pytest.mark.parametrize(
    "address",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "0" * 64,
        "../" + "a" * 61,
    ],
)
def test_address_must_be_one_lowercase_nonzero_sha256(
    tmp_path: Path,
    address: str,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="address",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=address,
            expected_files=records,
            payloads=payloads,
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        ".hidden",
        "nested/file.bin",
        "nested\\file.bin",
        "white space.bin",
        "é.bin",
        "a" * 129,
    ],
)
def test_unsafe_or_nested_record_names_are_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    parent = _safe_parent(tmp_path)
    raw = b"x"
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="safe portable flat name",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=(atomic.ExactFileRecord(name, 1, _sha256(raw)),),
            payloads={name: raw},
        )


def test_duplicate_record_names_and_non_tuple_records_are_rejected(
    tmp_path: Path,
) -> None:
    parent = _safe_parent(tmp_path)
    record = atomic.ExactFileRecord("same.bin", 1, _sha256(b"x"))
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="duplicate",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=(record, record),
            payloads={"same.bin": b"x"},
        )
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="exact tuple",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=[record],  # type: ignore[arg-type]
            payloads={"same.bin": b"x"},
        )


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (atomic.ExactFileRecord("a.bin", -1, _sha256(b"")), "size"),
        (atomic.ExactFileRecord("a.bin", True, _sha256(b"x")), "size"),
        (
            atomic.ExactFileRecord("a.bin", atomic.MAX_FILE_BYTES + 1, _sha256(b"x")),
            "size",
        ),
        (atomic.ExactFileRecord("a.bin", 1, "A" * 64), "digest"),
        (atomic.ExactFileRecord("a.bin", 1, "0" * 64), "digest"),
    ],
)
def test_bad_expected_size_or_digest_is_rejected(
    tmp_path: Path,
    record: atomic.ExactFileRecord,
    match: str,
) -> None:
    parent = _safe_parent(tmp_path)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError, match=match):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=(record,),
            payloads={"a.bin": b"x"},
        )


def test_count_and_total_bounds_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    empty_digest = _sha256(b"")
    too_many = tuple(
        atomic.ExactFileRecord(f"f{index}.bin", 0, empty_digest)
        for index in range(atomic.MAX_FILES + 1)
    )
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="file count",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=too_many,
            payloads={record.name: b"" for record in too_many},
        )

    monkeypatch.setattr(atomic, "MAX_TOTAL_BYTES", 1)
    records = (
        atomic.ExactFileRecord("a.bin", 1, _sha256(b"a")),
        atomic.ExactFileRecord("b.bin", 1, _sha256(b"b")),
    )
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="total byte bound",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads={"a.bin": b"a", "b.bin": b"b"},
        )


@pytest.mark.parametrize(
    "payloads",
    [
        {"manifest.json": b"wrong", "empty.bin": b"", "trace.bin": b"\x01\x00\xff\x01"},
        {"manifest.json": b'{"classification":"synthetic"}', "empty.bin": b""},
        {
            "manifest.json": b'{"classification":"synthetic"}',
            "empty.bin": b"",
            "trace.bin": b"\x01\x00\xff\x01",
            "extra.bin": b"x",
        },
        {
            "manifest.json": bytearray(b'{"classification":"synthetic"}'),
            "empty.bin": b"",
            "trace.bin": b"\x01\x00\xff\x01",
        },
    ],
)
def test_payload_inventory_type_size_and_digest_are_exact(
    tmp_path: Path,
    payloads: dict[str, Any],
) -> None:
    parent = _safe_parent(tmp_path)
    _, records = _payload_case()
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mutation", ["parent", "root", "file"])
def test_unsafe_modes_are_rejected_on_reload(tmp_path: Path, mutation: str) -> None:
    parent, _, records, published = _publish(tmp_path)
    target = {
        "parent": parent,
        "root": published.root,
        "file": published.root / "manifest.json",
    }[mutation]
    target.chmod(0o755 if mutation != "file" else 0o644)

    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="unsafe|mode",
    ):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )


def test_symlink_parent_and_symlink_publication_are_rejected(tmp_path: Path) -> None:
    real_parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="canonical non-symlink",
    ):
        atomic.publish_exact_flat_publication(
            alias.absolute(),
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (real_parent / _ADDRESS).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            real_parent,
            address=_ADDRESS,
            expected_files=records,
        )


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_symlink_special_or_nested_entry_is_rejected(
    tmp_path: Path,
    kind: str,
) -> None:
    parent, _, records, published = _publish(tmp_path)
    target = published.root / "trace.bin"
    target.unlink()
    if kind == "symlink":
        external = parent / "external.bin"
        external.write_bytes(b"\x01\x00\xff\x01")
        external.chmod(0o600)
        target.symlink_to(external)
    elif kind == "fifo":
        os.mkfifo(target, 0o600)
        target.chmod(0o600)
    else:
        target.mkdir(mode=0o700)

    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )


def test_hard_link_is_rejected_even_when_outside_exact_root(tmp_path: Path) -> None:
    parent, _, records, published = _publish(tmp_path)
    os.link(published.root / "manifest.json", parent / "outside-hard-link")

    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="linked|safe inode|unexpected",
    ):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exact_inventory_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    parent, _, records, published = _publish(tmp_path)
    if mutation == "missing":
        (published.root / "trace.bin").unlink()
    else:
        extra = published.root / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)

    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )


def test_wrong_caller_carried_records_fail_reload(tmp_path: Path) -> None:
    parent, _, records, _ = _publish(tmp_path)
    wrong = tuple(
        atomic.ExactFileRecord(record.name, record.size_bytes, _sha256(b"different"))
        if record.name == "trace.bin"
        else record
        for record in records
    )
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=wrong,
        )


def test_relative_root_filesystem_root_and_non_private_parent_are_rejected(
    tmp_path: Path,
) -> None:
    payloads, records = _payload_case()
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="absolute non-root",
    ):
        atomic.publish_exact_flat_publication(
            Path("relative"),
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="absolute non-root",
    ):
        atomic.publish_exact_flat_publication(
            Path("/"),
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    parent = _safe_parent(tmp_path)
    parent.chmod(0o750)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="unsafe ownership/mode",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )


def test_existing_address_collision_never_replaces_content(tmp_path: Path) -> None:
    parent, payloads, records, published = _publish(tmp_path)
    before = {name: (published.root / name).read_bytes() for name in payloads}

    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationCollisionError):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )

    assert {name: (published.root / name).read_bytes() for name in payloads} == before
    assert _staging_names(parent) == []


def test_concurrent_publishers_have_exactly_one_winner(tmp_path: Path) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    barrier = threading.Barrier(2)

    def attempt() -> object:
        barrier.wait()
        try:
            return atomic.publish_exact_flat_publication(
                parent,
                address=_ADDRESS,
                expected_files=records,
                payloads=payloads,
            )
        except BaseException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(2)))

    assert sum(
        isinstance(item, atomic.ContentVerifiedFlatPublication) for item in outcomes
    ) == 1
    assert sum(
        isinstance(item, atomic.ForagerMatchedV3AtomicPublicationCollisionError)
        for item in outcomes
    ) == 1
    loaded = atomic.load_exact_flat_publication(
        parent,
        address=_ADDRESS,
        expected_files=records,
    )
    assert dict(loaded.files) == payloads
    assert _staging_names(parent) == []


def test_missing_renameat2_fails_closed_without_overwrite_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()

    class _LibcWithoutRename:
        pass

    monkeypatch.setattr(atomic.ctypes, "CDLL", lambda *args, **kwargs: _LibcWithoutRename())
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationError,
        match="no overwrite fallback",
    ):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert not (parent / _ADDRESS).exists()
    assert _staging_names(parent) == []


def test_precommit_rename_failure_cleans_staging_and_is_not_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()

    def fail_before_rename(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic precommit failure")

    monkeypatch.setattr(atomic, "_rename_no_replace", fail_before_rename)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert not isinstance(
        caught.value,
        atomic.ForagerMatchedV3AtomicPublicationUncertainError,
    )
    assert not (parent / _ADDRESS).exists()
    assert _staging_names(parent) == []


def test_rename_reports_error_after_commit_with_recoverable_true_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    real_rename = atomic._rename_no_replace

    def rename_then_fail(
        opened_parent: Any,
        source_name: str,
        destination_name: str,
    ) -> None:
        real_rename(opened_parent, source_name, destination_name)
        raise OSError("synthetic lost rename response")

    monkeypatch.setattr(atomic, "_rename_no_replace", rename_then_fail)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert caught.value.address == _ADDRESS
    assert caught.value.destination == parent / _ADDRESS
    assert caught.value.committed is True
    assert dict(
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        ).files
    ) == payloads


def test_disappeared_staging_reports_unknown_commit_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    abandoned_name = ".synthetic-ambiguous-staging"

    def move_elsewhere_then_fail(
        opened_parent: Any,
        source_name: str,
        destination_name: str,
    ) -> None:
        del destination_name
        os.rename(
            source_name,
            abandoned_name,
            src_dir_fd=opened_parent.descriptor,
            dst_dir_fd=opened_parent.descriptor,
        )
        raise OSError("synthetic indeterminate rename")

    monkeypatch.setattr(atomic, "_rename_no_replace", move_elsewhere_then_fail)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert caught.value.address == _ADDRESS
    assert caught.value.committed is None
    assert not (parent / _ADDRESS).exists()


def test_parent_fsync_failure_after_move_reports_committed_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()

    def fail_parent_sync(opened_parent: Any) -> None:
        del opened_parent
        raise OSError("synthetic parent fsync failure")

    monkeypatch.setattr(atomic, "_sync_publication_parent", fail_parent_sync)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert caught.value.committed is True
    assert (parent / _ADDRESS).is_dir()


def test_final_replay_failure_reports_committed_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    real_load = atomic._load_from_open_root
    calls = 0

    def fail_second_replay(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic final replay failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(atomic, "_load_from_open_root", fail_second_replay)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert calls == 2
    assert caught.value.committed is True
    assert (parent / _ADDRESS).is_dir()


def test_cleanup_failure_cannot_mask_hostile_precommit_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()

    class _HostilePrimary(BaseException):
        pass

    def hostile_rename(*args: object, **kwargs: object) -> None:
        raise _HostilePrimary("primary")

    def hostile_cleanup(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cleanup must not mask primary")

    monkeypatch.setattr(atomic, "_rename_no_replace", hostile_rename)
    monkeypatch.setattr(atomic, "_cleanup_owned_staging", hostile_cleanup)
    with pytest.raises(_HostilePrimary, match="primary"):
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )


def test_file_swap_between_stat_and_open_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _, records, published = _publish(tmp_path)
    target = published.root / "trace.bin"
    real_open = atomic.os.open
    swapped = False

    def swap_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if path == "trace.bin" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            target.unlink()
            target.write_bytes(b"\x02\x00\xfe\x02")
            target.chmod(0o600)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(atomic.os, "open", swap_then_open)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )
    assert swapped is True


def test_address_swap_during_load_is_detected_through_held_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _, records, _ = _publish(tmp_path)
    real_read = atomic._read_stable_regular_at
    moved = parent / f"{_ADDRESS}.moved"
    swapped = False

    def read_then_swap(*args: Any, **kwargs: Any) -> bytes:
        nonlocal swapped
        raw = real_read(*args, **kwargs)
        if not swapped:
            swapped = True
            (parent / _ADDRESS).rename(moved)
            replacement = parent / _ADDRESS
            replacement.mkdir(mode=0o700)
            replacement.chmod(0o700)
        return raw

    monkeypatch.setattr(atomic, "_read_stable_regular_at", read_then_swap)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )
    assert swapped is True


def test_mutation_after_pre_replay_is_caught_by_post_move_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    payloads, records = _payload_case()
    real_sync = atomic._durably_sync_open_tree

    def sync_then_mutate(opened_root: Any, expected: Any) -> None:
        real_sync(opened_root, expected)
        target = opened_root.path / "trace.bin"
        target.write_bytes(b"\x02\x00\xfe\x02")
        target.chmod(0o600)

    monkeypatch.setattr(atomic, "_durably_sync_open_tree", sync_then_mutate)
    with pytest.raises(
        atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as caught:
        atomic.publish_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
            payloads=payloads,
        )
    assert caught.value.committed is True
    assert (parent / _ADDRESS).is_dir()


def test_owner_mismatch_is_rejected_when_privileged(tmp_path: Path) -> None:
    if os.geteuid() != 0:
        pytest.skip("changing file ownership requires privilege")
    parent, _, records, published = _publish(tmp_path)
    target = published.root / "trace.bin"
    os.chown(target, 1, 1)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        atomic.load_exact_flat_publication(
            parent,
            address=_ADDRESS,
            expected_files=records,
        )
