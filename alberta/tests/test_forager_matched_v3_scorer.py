"""Synthetic-only tests for strict matched-v3 raw-reward score ingestion."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import warnings
import zipfile
import zlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

_TRACE_DOMAIN = b"alberta.forager.matched_v3.raw_reward_trace.int8.v1"


@pytest.fixture(scope="module")
def valid_trace() -> bytes:
    assert protocol.MATCHED_V3_HORIZON % 4 == 0
    return bytes((255, 0, 1, 30)) * (protocol.MATCHED_V3_HORIZON // 4)


def _write_npz(
    path: Path,
    values: np.ndarray[Any, Any],
    *,
    key: str = "rewards",
    compressed: bool = False,
) -> None:
    with path.open("wb") as handle:
        if key == "rewards":
            if compressed:
                np.savez_compressed(handle, rewards=values)
            else:
                np.savez(handle, rewards=values)
        elif key == "reward":
            if compressed:
                np.savez_compressed(handle, reward=values)
            else:
                np.savez(handle, reward=values)
        else:
            raise AssertionError("test helper only supports its two synthetic keys")


def _write_trace_npz(path: Path, trace: bytes) -> None:
    _write_npz(path, np.frombuffer(trace, dtype=np.int8))


def _ingest_path(path: Path) -> scorer.MatchedV3ScoreReceipt:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        return scorer.ingest_reward_npz_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _canonical_receipt_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    canonical = (
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    return hashlib.sha256(canonical).hexdigest()


def _replace_member_bytes(raw: bytes, old: bytes, new: bytes) -> bytes:
    assert len(old) == len(new)
    changed = bytearray(raw)
    member_offset = raw.index(b"\x93NUMPY")
    member_size = 128 + protocol.MATCHED_V3_HORIZON
    relative = raw[member_offset : member_offset + member_size].index(old)
    start = member_offset + relative
    changed[start : start + len(old)] = new
    crc32 = zlib.crc32(changed[member_offset : member_offset + member_size]) & 0xFFFFFFFF
    struct.pack_into("<I", changed, 14, crc32)
    central_offset = raw.index(b"PK\x01\x02")
    struct.pack_into("<I", changed, central_offset + 16, crc32)
    return bytes(changed)


@pytest.mark.unit
def test_ingestion_binds_metric_exact_score_and_deterministic_trace_digest(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    assert scorer.CANONICAL_NPZ_SIZE_BYTES == 499_980
    path = tmp_path / "valid.npz"
    _write_trace_npz(path, valid_trace)
    encoded = scorer.canonical_reward_npz_bytes(valid_trace)
    assert encoded == path.read_bytes()

    receipt = _ingest_path(path)
    assert scorer.ingest_reward_npz_bytes(encoded) == receipt
    assert scorer.extract_canonical_reward_trace(encoded) == valid_trace
    expected_preimage = b"".join(
        (
            len(_TRACE_DOMAIN).to_bytes(4, "big"),
            _TRACE_DOMAIN,
            len(valid_trace).to_bytes(8, "big"),
            valid_trace,
        )
    )
    expected_trace_sha256 = hashlib.sha256(expected_preimage).hexdigest()
    expected_score = 30 * (protocol.MATCHED_V3_HORIZON // 4)

    assert receipt.cumulative_score == expected_score
    assert receipt.raw_trace_sha256 == expected_trace_sha256
    assert expected_trace_sha256 == (
        "5d40663e010645c17b1f9b85bba2f4a2411caf0914a1b615142ad191be149f3b"
    )
    assert receipt.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt.artifact_size_bytes == scorer.CANONICAL_NPZ_SIZE_BYTES
    assert scorer.canonical_raw_reward_trace_sha256(valid_trace) == expected_trace_sha256
    payload = receipt.to_payload()
    assert payload["metric"] == {
        "descriptor": protocol.cumulative_reward_metric_descriptor(),
        "sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
    }
    assert payload["authority"] == {
        "task_identity_authority": False,
        "configuration_identity_authority": False,
        "candidate_identity_authority": False,
        "scientific_evidence_authority": False,
        "qualification_authority": False,
        "execution_authority": False,
        "promotion_authority": False,
    }
    assert receipt.canonical_json().endswith(b"\n")
    assert scorer.parse_score_receipt(receipt.canonical_json()) == receipt


@pytest.mark.unit
def test_receipt_is_frozen_detached_content_addressed_and_replayable(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    path = tmp_path / "receipt.npz"
    _write_trace_npz(path, valid_trace)
    receipt = _ingest_path(path)
    payload = receipt.to_payload()
    payload["metric"]["descriptor"]["horizon"] = 1
    payload["authority"]["scientific_evidence_authority"] = True

    assert receipt.to_payload()["metric"]["descriptor"]["horizon"] == (
        protocol.MATCHED_V3_HORIZON
    )
    assert set(receipt.to_payload()["authority"].values()) == {False}
    assert receipt.receipt_sha256 == hashlib.sha256(receipt.canonical_body()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        receipt.cumulative_score = 0  # type: ignore[misc]

    descriptor = os.open(path, os.O_RDONLY)
    try:
        assert scorer.replay_reward_npz_descriptor(descriptor, receipt.to_payload()) == receipt
    finally:
        os.close(descriptor)


@pytest.mark.unit
def test_open_descriptor_defeats_path_substitution_and_preserves_offset(
    tmp_path: Path,
) -> None:
    original = tmp_path / "active.npz"
    replacement = tmp_path / "replacement.npz"
    _write_trace_npz(original, bytes(protocol.MATCHED_V3_HORIZON))
    _write_trace_npz(replacement, bytes((30,)) * protocol.MATCHED_V3_HORIZON)
    descriptor = os.open(original, os.O_RDONLY)
    try:
        os.lseek(descriptor, 17, os.SEEK_SET)
        retained = tmp_path / "retained-original.npz"
        os.rename(original, retained)
        os.replace(replacement, original)
        receipt = scorer.ingest_reward_npz_descriptor(descriptor)
        assert receipt.cumulative_score == 0
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 17
        os.fstat(descriptor)
    finally:
        os.close(descriptor)
    assert _ingest_path(original).cumulative_score == 30 * protocol.MATCHED_V3_HORIZON


@pytest.mark.unit
def test_out_of_support_raw_reward_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "support.npz"
    rewards = np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.int8)
    rewards[-1] = np.int8(2)
    _write_npz(path, rewards)

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="raw reward support"):
        _ingest_path(path)


@pytest.mark.unit
@pytest.mark.parametrize("length_delta", [-1, 1])
def test_horizon_is_exact(tmp_path: Path, length_delta: int) -> None:
    path = tmp_path / f"horizon-{length_delta}.npz"
    rewards = np.zeros((protocol.MATCHED_V3_HORIZON + length_delta,), dtype=np.int8)
    _write_npz(path, rewards)

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="byte size|horizon"):
        _ingest_path(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.uint8),
        np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.dtype(">i2")),
        np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.float32),
        np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.bool_),
        np.asarray([object()], dtype=object),
    ],
    ids=["uint8", "big-endian-int16", "float32", "bool", "object"],
)
def test_dtype_endian_and_object_aliases_are_rejected(
    tmp_path: Path,
    values: np.ndarray[Any, Any],
) -> None:
    path = tmp_path / f"alias-{values.dtype.str.replace('>', 'be')}.npz"
    _write_npz(path, values)

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="byte size|NPY header"):
        _ingest_path(path)


@pytest.mark.unit
def test_same_size_dtype_alias_and_malformed_npy_header_are_rejected(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    valid = tmp_path / "base.npz"
    _write_trace_npz(valid, valid_trace)
    raw = valid.read_bytes()
    candidates = {
        "uint8-alias": _replace_member_bytes(raw, b"'|i1'", b"'|u1'"),
        "bad-magic": _replace_member_bytes(raw, b"\x93NUMPY", b"\x93XUMPY"),
    }

    for name, candidate in candidates.items():
        path = tmp_path / f"{name}.npz"
        path.write_bytes(candidate)
        with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="NPY header"):
            _ingest_path(path)


@pytest.mark.unit
def test_compression_and_zip_bomb_shape_are_rejected_before_decompression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compressed.npz"
    rewards = np.zeros((protocol.MATCHED_V3_HORIZON,), dtype=np.int8)
    _write_npz(path, rewards, compressed=True)
    assert path.stat().st_size < scorer.CANONICAL_NPZ_SIZE_BYTES

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="byte size"):
        _ingest_path(path)


@pytest.mark.unit
def test_duplicate_path_like_extra_and_trailing_zip_members_are_rejected(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    valid = tmp_path / "inventory-base.npz"
    _write_trace_npz(valid, valid_trace)
    with zipfile.ZipFile(valid, mode="r") as archive:
        member = archive.read("rewards.npy")

    malformed: dict[str, bytes] = {}
    for label, names in {
        "duplicate": ("rewards.npy", "rewards.npy"),
        "path-like": ("../ward.npy",),
        "wrong-key": ("reward.npy",),
        "extra": ("rewards.npy", "extra.npy"),
    }.items():
        target = tmp_path / f"build-{label}.npz"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED) as archive:
                for name in names:
                    archive.writestr(name, member)
        malformed[label] = target.read_bytes()
    malformed["trailing"] = valid.read_bytes() + b"overlay"

    for label, raw in malformed.items():
        path = tmp_path / f"invalid-{label}.npz"
        path.write_bytes(raw)
        with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="byte size|ZIP"):
            _ingest_path(path)


@pytest.mark.unit
def test_same_size_path_member_and_zip_metadata_drift_are_rejected(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    valid = tmp_path / "metadata-base.npz"
    _write_trace_npz(valid, valid_trace)
    raw = valid.read_bytes()
    path_like = raw.replace(b"rewards.npy", b"../ward.npy")
    timestamp = bytearray(raw)
    timestamp[10] = 1

    for label, candidate in {"path": path_like, "timestamp": bytes(timestamp)}.items():
        path = tmp_path / f"metadata-{label}.npz"
        path.write_bytes(candidate)
        with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="ZIP"):
            _ingest_path(path)


@pytest.mark.unit
def test_hard_size_bound_precedes_artifact_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.npz"
    with path.open("wb") as handle:
        handle.truncate(scorer.CANONICAL_NPZ_SIZE_BYTES + 1)
    called = False

    def forbidden_read(descriptor: int, byte_count: int) -> bytes:
        del descriptor, byte_count
        nonlocal called
        called = True
        raise AssertionError("oversized artifact must not be read")

    monkeypatch.setattr(scorer, "_read_descriptor_bytes", forbidden_read)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="byte size"):
        _ingest_path(path)
    assert called is False


@pytest.mark.unit
def test_descriptor_identity_drift_is_rejected(
    tmp_path: Path,
    valid_trace: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "drift.npz"
    _write_trace_npz(path, valid_trace)
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> Any:
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_size=metadata.st_size,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino + 1,
                st_nlink=metadata.st_nlink,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
            )
        return metadata

    monkeypatch.setattr(os, "fstat", drifting_fstat)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="changed during ingestion"):
        _ingest_path(path)


@pytest.mark.unit
def test_multiple_hard_links_are_rejected(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    path = tmp_path / "linked.npz"
    alias = tmp_path / "linked-alias.npz"
    _write_trace_npz(path, valid_trace)
    os.link(path, alias)

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="exactly one filesystem link"):
        _ingest_path(path)


@pytest.mark.unit
def test_nonregular_and_boolean_descriptors_are_rejected(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="regular file"):
            scorer.ingest_reward_npz_descriptor(descriptor)
    finally:
        os.close(descriptor)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="descriptor"):
        scorer.ingest_reward_npz_descriptor(True)


@pytest.mark.unit
def test_receipt_rejects_aliases_authority_mutation_and_artifact_mismatch(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    path = tmp_path / "replay.npz"
    _write_trace_npz(path, valid_trace)
    receipt = _ingest_path(path)

    authority = receipt.to_payload()
    authority["authority"]["qualification_authority"] = True
    authority["receipt_sha256"] = _canonical_receipt_sha256(authority)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="authority"):
        scorer.parse_score_receipt(authority)

    boolean_score = receipt.to_payload()
    boolean_score["score"]["cumulative_reward"] = True
    boolean_score["receipt_sha256"] = _canonical_receipt_sha256(boolean_score)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="exact integer"):
        scorer.parse_score_receipt(boolean_score)

    detached_change = receipt.to_payload()
    detached_change["score"]["cumulative_reward"] += 1
    detached_change["receipt_sha256"] = _canonical_receipt_sha256(detached_change)
    changed = scorer.parse_score_receipt(detached_change)
    assert changed != receipt
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="does not replay"):
            scorer.replay_reward_npz_descriptor(descriptor, detached_change)
    finally:
        os.close(descriptor)

    digest_mutation = receipt.to_payload()
    digest_mutation["raw_trace"]["sha256"] = "0" * 64
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="receipt_sha256"):
        scorer.parse_score_receipt(digest_mutation)


@pytest.mark.unit
def test_raw_trace_digest_helper_rejects_wrong_horizon_and_support(
    valid_trace: bytes,
) -> None:
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="horizon"):
        scorer.canonical_raw_reward_trace_sha256(valid_trace[:-1])
    invalid = bytearray(valid_trace)
    invalid[0] = 2
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="raw reward support"):
        scorer.canonical_raw_reward_trace_sha256(bytes(invalid))


@pytest.mark.unit
def test_receipt_json_is_canonical_and_duplicate_free(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    path = tmp_path / "canonical.npz"
    _write_trace_npz(path, valid_trace)
    receipt = _ingest_path(path)
    payload = receipt.to_payload()
    noncanonical = json.dumps(payload, indent=2).encode()
    duplicate = receipt.canonical_json().replace(
        b'{"artifact":',
        b'{"artifact":null,"artifact":',
        1,
    )

    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="canonical"):
        scorer.parse_score_receipt(noncanonical)
    with pytest.raises(scorer.ForagerMatchedV3ScorerError, match="duplicate JSON object key"):
        scorer.parse_score_receipt(duplicate)


@pytest.mark.unit
def test_metric_descriptor_returned_by_receipt_is_detached(
    tmp_path: Path,
    valid_trace: bytes,
) -> None:
    path = tmp_path / "metric-detached.npz"
    _write_trace_npz(path, valid_trace)
    receipt = _ingest_path(path)
    first = receipt.to_payload()
    second = copy.deepcopy(first)
    second["metric"]["descriptor"]["raw_reward_values"].append(999)

    assert first == receipt.to_payload()
    assert second != receipt.to_payload()


@pytest.mark.unit
def test_authority_schema_ignores_mutated_private_denial_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = scorer.MatchedV3ScoreReceipt(
        cumulative_score=0,
        raw_trace_sha256="1" * 64,
        artifact_sha256="2" * 64,
        artifact_size_bytes=scorer.CANONICAL_NPZ_SIZE_BYTES,
    )
    monkeypatch.setattr(scorer, "_AUTHORITY_DENIAL", {}, raising=False)

    payload = receipt.to_payload()
    assert payload["authority"] == {
        "task_identity_authority": False,
        "configuration_identity_authority": False,
        "candidate_identity_authority": False,
        "scientific_evidence_authority": False,
        "qualification_authority": False,
        "execution_authority": False,
        "promotion_authority": False,
    }
    assert scorer.parse_score_receipt(payload) == receipt
