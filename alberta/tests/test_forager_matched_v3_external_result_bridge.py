"""Cheap adversarial tests for the content-only external reward bridge."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import stat
import struct
import warnings
import zipfile
from dataclasses import FrozenInstanceError, replace
from functools import cache
from types import MappingProxyType
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    _forager_matched_v3_external_result_bridge as bridge,
)
from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

_EXPECTED_DESCRIPTOR_SHA256 = "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
_ZIP_CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")
_ZIP_END = struct.Struct("<IHHHHIIH")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _npy(
    *,
    descriptor: object,
    data: bytes,
    shape: object = (bridge.MATCHED_V3_REWARD_HORIZON,),
    fortran_order: object = False,
    version: tuple[int, int] = (1, 0),
    minimum_header_size: int = 0,
) -> bytes:
    dictionary = repr(
        {
            "descr": descriptor,
            "fortran_order": fortran_order,
            "shape": shape,
        }
    ).encode("ascii")
    length_size = 2 if version == (1, 0) else 4
    prefix_size = 8 + length_size
    padding = (-((prefix_size + len(dictionary) + 1) % 64)) % 64
    while len(dictionary) + padding + 1 < minimum_header_size:
        padding += 64
    header = dictionary + b" " * padding + b"\n"
    if length_size == 2:
        length = struct.pack("<H", len(header))
    else:
        length = struct.pack("<I", len(header))
    return b"\x93NUMPY" + bytes(version) + length + header + data


def _npy_literal(dictionary: bytes, data: bytes) -> bytes:
    prefix_size = 10
    padding = (-((prefix_size + len(dictionary) + 1) % 64)) % 64
    header = dictionary + b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def _data_offset(member: bytes | bytearray) -> int:
    version = (member[6], member[7])
    if version == (1, 0):
        return 10 + int(struct.unpack_from("<H", member, 8)[0])
    return 12 + int(struct.unpack_from("<I", member, 8)[0])


@cache
def _valid_member(descriptor: str) -> bytes:
    item_size = 2 if descriptor == "<f2" else 4
    pack_format = "<e" if descriptor == "<f2" else "<f"
    data = bytearray(bridge.MATCHED_V3_REWARD_HORIZON * item_size)
    for index, value in ((0, -1), (1, 0), (2, 1), (3, 30), (499_711, -1)):
        struct.pack_into(pack_format, data, index * item_size, value)
    return _npy(descriptor=descriptor, data=bytes(data))


def _member_with_value(descriptor: str, index: int, value: float) -> bytes:
    member = bytearray(_valid_member(descriptor))
    item_size = 2 if descriptor == "<f2" else 4
    pack_format = "<e" if descriptor == "<f2" else "<f"
    struct.pack_into(pack_format, member, _data_offset(member) + index * item_size, value)
    return bytes(member)


def _zip(
    entries: list[tuple[str, bytes, int, int]] | None = None,
    *,
    reward_descriptor: str = "<f2",
    reward_member: bytes | None = None,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    if entries is None:
        entries = [
            (
                "rewards.npy",
                _valid_member(reward_descriptor) if reward_member is None else reward_member,
                compression,
                (stat.S_IFREG | 0o600) << 16,
            )
        ]
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
            for name, member, method, attributes in entries:
                info = zipfile.ZipInfo(name)
                info.compress_type = method
                info.create_system = 3
                info.external_attr = attributes
                archive.writestr(info, member)
    return output.getvalue()


def _zip64_reward(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        info = zipfile.ZipInfo("rewards.npy")
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        with archive.open(info, "w", force_zip64=True) as target:
            target.write(member)
    return output.getvalue()


def _central_records(raw: bytes) -> list[tuple[int, tuple[Any, ...]]]:
    end_offset = len(raw) - _ZIP_END.size
    end = _ZIP_END.unpack_from(raw, end_offset)
    count = int(end[4])
    cursor = int(end[6])
    result: list[tuple[int, tuple[Any, ...]]] = []
    for _ in range(count):
        record = _ZIP_CENTRAL.unpack_from(raw, cursor)
        result.append((cursor, record))
        cursor += _ZIP_CENTRAL.size + int(record[10]) + int(record[11]) + int(record[12])
    return result


def _patch_flags(raw: bytes, flags: int) -> bytes:
    changed = bytearray(raw)
    central_offset, central = _central_records(raw)[0]
    local_offset = int(central[16])
    struct.pack_into("<H", changed, local_offset + 6, flags)
    struct.pack_into("<H", changed, central_offset + 8, flags)
    return bytes(changed)


def _patch_method(raw: bytes, method: int) -> bytes:
    changed = bytearray(raw)
    central_offset, central = _central_records(raw)[0]
    local_offset = int(central[16])
    struct.pack_into("<H", changed, local_offset + 8, method)
    struct.pack_into("<H", changed, central_offset + 10, method)
    return bytes(changed)


def _corrupt_stored_reward(raw: bytes) -> bytes:
    changed = bytearray(raw)
    _central_offset, central = _central_records(raw)[0]
    local_offset = int(central[16])
    name_size, extra_size = struct.unpack_from("<HH", raw, local_offset + 26)
    data_offset = local_offset + 30 + name_size + extra_size
    changed[data_offset] ^= 1
    return bytes(changed)


@pytest.fixture(scope="module")
def continuing_case() -> tuple[bytes, bridge.ExternalRewardConversion]:
    raw = _zip(reward_descriptor="<f2")
    return raw, bridge.convert_external_reward_npz(
        candidate_id="external_dqn_plain", external_npz=raw
    )


@pytest.mark.unit
def test_descriptor_is_literal_self_pinned_detached_and_non_authorizing() -> None:
    raw = bridge.canonical_external_result_bridge_descriptor_bytes()
    descriptor = bridge.external_result_bridge_descriptor()
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256
    assert bridge.EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256 == _EXPECTED_DESCRIPTOR_SHA256
    assert bridge.external_result_bridge_descriptor_sha256() == _EXPECTED_DESCRIPTOR_SHA256
    assert bridge.parse_external_result_bridge_descriptor(raw) == descriptor
    assert descriptor["status"] == (
        "implemented_score_reward_bearing_permanently_nonqualifying_non_authorizing"
    )
    assert all(value is False for value in descriptor["claims"].values())
    assert all(value is False for value in descriptor["apis"].values())
    assert descriptor["exposure_contract"] == {
        "conversion_input_npz_contains_raw_rewards": True,
        "conversion_trace_contains_complete_raw_rewards": True,
        "conversion_canonical_npz_contains_complete_raw_rewards": True,
        "conversion_cumulative_score_is_plaintext": True,
        "receipt_cumulative_score_is_plaintext": True,
        "receipt_contains_reward_and_score_commitments": True,
        "score_blind_controller_input_allowed": False,
        "score_blind_publisher_input_allowed": False,
        "fresh_isolated_post_qualification_outcome_consumer_required": True,
        "fresh_process_isolation_enforced_by_this_module": False,
        "permanently_nonqualifying": True,
    }
    descriptor["claims"]["result_accepted"] = True
    assert bridge.external_result_bridge_descriptor()["claims"]["result_accepted"] is False


@pytest.mark.unit
def test_exact_twelve_candidate_family_dtype_mapping_is_immutable() -> None:
    expected_order = (
        "external_dqn_plain",
        "external_dqn_crelu",
        "external_dqn_redo",
        "external_dqn_reward_trace",
        "external_dqn_l2_init",
        "external_pt_dqn_xfinal",
        "external_drqn_xfinal",
        "isolated_ppo_generic",
        "isolated_rtu_paper_scale",
        "random_policy",
        "search_nearest",
        "search_oracle",
    )
    continuing = (
        "external_dqn_plain",
        "external_dqn_crelu",
        "external_dqn_redo",
        "external_dqn_reward_trace",
        "external_dqn_l2_init",
        "external_pt_dqn_xfinal",
        "external_drqn_xfinal",
        "random_policy",
        "search_nearest",
        "search_oracle",
    )
    ppo = ("isolated_ppo_generic", "isolated_rtu_paper_scale")
    assert bridge.EXTERNAL_RESULT_CANDIDATE_IDS == expected_order
    assert bridge.external_result_bridge_descriptor()["candidate_contract"][
        "candidate_ids"
    ] == list(expected_order)
    assert bridge.EXTERNAL_RESULT_CANDIDATE_FORMATS == {
        **{candidate_id: ("continuing", "<f2") for candidate_id in continuing},
        **{candidate_id: ("ppo", "<f4") for candidate_id in ppo},
    }
    assert isinstance(bridge.EXTERNAL_RESULT_CANDIDATE_FORMATS, MappingProxyType)
    with pytest.raises(TypeError):
        bridge.EXTERNAL_RESULT_CANDIDATE_FORMATS["external_dqn_plain"] = (  # type: ignore[index]
            "ppo",
            "<f4",
        )


@pytest.mark.unit
def test_all_twelve_candidates_convert_in_exact_frozen_order() -> None:
    raw_by_dtype = {
        "<f2": _zip(reward_descriptor="<f2"),
        "<f4": _zip(reward_descriptor="<f4"),
    }
    observed: list[str] = []
    for candidate_id in bridge.EXTERNAL_RESULT_CANDIDATE_IDS:
        family, descriptor = bridge.EXTERNAL_RESULT_CANDIDATE_FORMATS[candidate_id]
        conversion = bridge.convert_external_reward_npz(
            candidate_id=candidate_id,
            external_npz=raw_by_dtype[descriptor],
        )
        observed.append(conversion.candidate_id)
        assert conversion.family == family
        assert conversion.external_dtype == descriptor
        assert conversion.trace[:4] == b"\xff\x00\x01\x1e"
        assert conversion.cumulative_score == 29
    assert tuple(observed) == bridge.EXTERNAL_RESULT_CANDIDATE_IDS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate_id", "descriptor", "compression"),
    [
        ("external_dqn_plain", "<f2", zipfile.ZIP_STORED),
        ("isolated_ppo_generic", "<f4", zipfile.ZIP_DEFLATED),
    ],
)
def test_both_dtype_happy_paths_roundtrip_exact_scorer_order_and_receipt(
    candidate_id: str, descriptor: str, compression: int
) -> None:
    raw = _zip(reward_descriptor=descriptor, compression=compression)
    conversion = bridge.convert_external_reward_npz(candidate_id=candidate_id, external_npz=raw)
    assert conversion.external_dtype == descriptor
    assert conversion.trace[:4] == b"\xff\x00\x01\x1e"
    assert conversion.trace[-1:] == b"\xff"
    assert len(conversion.trace) == 499_712
    assert conversion.cumulative_score == 29
    assert len(conversion.canonical_scorer_npz) == 499_980
    assert (
        scorer.extract_canonical_reward_trace(conversion.canonical_scorer_npz) == conversion.trace
    )
    scored = scorer.ingest_reward_npz_bytes(conversion.canonical_scorer_npz)
    assert scored.cumulative_score == 29
    assert scored.raw_trace_sha256 == conversion.trace_sha256
    receipt_raw = bridge.canonical_external_reward_conversion_receipt_bytes(conversion)
    receipt_sha256 = bridge.external_reward_conversion_receipt_sha256(conversion)
    assert (
        bridge.parse_external_reward_conversion_receipt(
            receipt_raw,
            expected_file_sha256=receipt_sha256,
            candidate_id=candidate_id,
            external_npz=raw,
        )
        == conversion
    )


@pytest.mark.unit
def test_numpy_style_zip64_local_sizes_and_npy_v2_are_supported() -> None:
    zip64 = _zip64_reward(_valid_member("<f2"))
    zip64_conversion = bridge.convert_external_reward_npz(
        candidate_id="external_dqn_crelu", external_npz=zip64
    )
    assert zip64_conversion.cumulative_score == 29

    valid_float32 = _valid_member("<f4")
    data = valid_float32[_data_offset(valid_float32) :]
    v2_member = _npy(descriptor="<f4", data=data, version=(2, 0))
    v2 = bridge.convert_external_reward_npz(
        candidate_id="isolated_rtu_paper_scale",
        external_npz=_zip(reward_member=v2_member),
    )
    assert v2.trace[:4] == b"\xff\x00\x01\x1e"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate_id", "descriptor", "member_count"),
    [
        ("external_dqn_plain", "<f2", 11),
        ("isolated_ppo_generic", "<f4", 81),
    ],
)
def test_observed_upstream_multi_member_layouts_are_inventoried_but_rewards_only_expanded(
    candidate_id: str,
    descriptor: str,
    member_count: int,
) -> None:
    entries = [("rewards.npy", _valid_member(descriptor), zipfile.ZIP_DEFLATED, 0o600 << 16)] + [
        (f"metadata_{index:03d}.npy", b"not-an-npy", zipfile.ZIP_DEFLATED, 0o600 << 16)
        for index in range(member_count - 1)
    ]
    raw = _zip(entries)
    assert len(_central_records(raw)) == member_count
    conversion = bridge.convert_external_reward_npz(candidate_id=candidate_id, external_npz=raw)
    assert conversion.cumulative_score == 29
    assert conversion.trace[:4] == b"\xff\x00\x01\x1e"


@pytest.mark.unit
def test_conversion_and_receipt_bind_every_content_and_scorer_identity(
    continuing_case: tuple[bytes, bridge.ExternalRewardConversion],
) -> None:
    raw, conversion = continuing_case
    receipt = bridge.external_reward_conversion_receipt(conversion)
    assert conversion.input_npz_sha256 == hashlib.sha256(raw).hexdigest()
    assert conversion.trace_bytes_sha256 == hashlib.sha256(conversion.trace).hexdigest()
    assert (
        conversion.canonical_scorer_npz_sha256
        == hashlib.sha256(conversion.canonical_scorer_npz).hexdigest()
    )
    assert receipt["input_npz"] == {
        "kind": "caller_supplied_immutable_bytes",
        "sha256": conversion.input_npz_sha256,
        "size_bytes": len(raw),
    }
    assert receipt["trace"]["length"] == 499_712
    assert receipt["score"]["cumulative_reward"] == 29
    assert receipt["canonical_scorer_npz"]["size_bytes"] == 499_980
    assert receipt["bindings"] == {
        "scorer_source_sha256": (
            "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
        ),
        "scorer_protocol_source_sha256": (
            "dd5db9a657ad167abf192942489642130b08bd065f724f7ad1b80743b1103720"
        ),
        "metric_schema_version": "alberta.forager_cumulative_reward_metric.v1",
        "metric_sha256": ("ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"),
    }
    assert all(value is False for value in receipt["claims"].values())
    assert receipt["claims"]["live_execution_completed"] is False
    assert receipt["claims"]["result_accepted"] is False
    assert receipt["exposure"] == {
        "raw_reward_bytes_exposed_by_conversion": True,
        "plaintext_cumulative_score_exposed": True,
        "reward_and_score_commitments_exposed": True,
        "score_blind_controller_input_allowed": False,
        "score_blind_publisher_input_allowed": False,
        "fresh_isolated_post_qualification_outcome_consumer_required": True,
        "permanently_nonqualifying": True,
    }
    with pytest.raises(FrozenInstanceError):
        conversion.family = "ppo"  # type: ignore[misc]


@pytest.mark.unit
def test_order_is_preserved_and_reordered_input_cannot_replay_original_receipt(
    continuing_case: tuple[bytes, bridge.ExternalRewardConversion],
) -> None:
    original_npz, original = continuing_case
    reordered_member = bytearray(_valid_member("<f2"))
    offset = _data_offset(reordered_member)
    for index, value in enumerate((30, 1, 0, -1)):
        struct.pack_into("<e", reordered_member, offset + index * 2, value)
    reordered_npz = _zip(reward_member=bytes(reordered_member))
    reordered = bridge.convert_external_reward_npz(
        candidate_id="external_dqn_plain", external_npz=reordered_npz
    )
    assert reordered.trace[:4] == b"\x1e\x01\x00\xff"
    assert reordered.cumulative_score == original.cumulative_score
    assert reordered.trace_sha256 != original.trace_sha256
    receipt = bridge.canonical_external_reward_conversion_receipt_bytes(original)
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.parse_external_reward_conversion_receipt(
            receipt,
            expected_file_sha256=hashlib.sha256(receipt).hexdigest(),
            candidate_id="external_dqn_plain",
            external_npz=reordered_npz,
        )
    assert original_npz != reordered_npz


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate_id", "descriptor"),
    [
        ("external_dqn_plain", ">f2"),
        ("external_dqn_plain", "=f2"),
        ("external_dqn_plain", "<f4"),
        ("isolated_ppo_generic", ">f4"),
        ("isolated_ppo_generic", "=f4"),
        ("isolated_ppo_generic", "<f2"),
    ],
)
def test_wrong_endian_native_or_candidate_dtype_fails(candidate_id: str, descriptor: str) -> None:
    item_size = 2 if descriptor.endswith("2") else 4
    member = _npy(descriptor=descriptor, data=b"\0" * (499_712 * item_size))
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id=candidate_id,
            external_npz=_zip(reward_member=member),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("descriptor", "shape", "fortran_order"),
    [
        ("|O", (499_712,), False),
        ([("reward", "<f2")], (499_712,), False),
        (("<f2", (1,)), (499_712,), False),
        ("<f2", (499_711,), False),
        ("<f2", (499_712, 1), False),
        ("<f2", (499_712,), True),
    ],
)
def test_object_structured_subarray_shape_and_fortran_headers_fail(
    descriptor: object, shape: object, fortran_order: object
) -> None:
    member = _npy(
        descriptor=descriptor,
        shape=shape,
        fortran_order=fortran_order,
        data=b"\0" * (499_712 * 2),
    )
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain", external_npz=_zip(reward_member=member)
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinities_fail(value: float) -> None:
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(reward_member=_member_with_value("<f2", 7, value)),
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [0.5, -0.5, 2.0, -2.0, 31.0])
def test_fractional_and_out_of_support_values_fail(value: float) -> None:
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(reward_member=_member_with_value("<f2", 11, value)),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "member",
    [
        _valid_member("<f2")[:-1],
        _valid_member("<f2") + b"\0",
        _npy(
            descriptor="<f2",
            data=b"\0" * (499_712 * 2),
            minimum_header_size=bridge.MAX_NPY_HEADER_BYTES + 1,
        ),
        _npy_literal(
            (b"{'descr': '<f2', 'descr': '<f2', 'fortran_order': False, 'shape': (499712,)}"),
            b"\0" * (499_712 * 2),
        ),
        b"\x93NUMPY\x03\x00" + struct.pack("<I", 64) + b" " * 64,
    ],
)
def test_truncated_trailing_oversized_header_and_unsupported_npy_version_fail(
    member: bytes,
) -> None:
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain", external_npz=_zip(reward_member=member)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "entries",
    [
        [("other.npy", b"x", zipfile.ZIP_STORED, (stat.S_IFREG | 0o600) << 16)],
        [
            ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16),
            ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16),
        ],
        [
            ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16),
            ("Rewards.npy", b"x", zipfile.ZIP_STORED, 0o600 << 16),
        ],
        [("../rewards.npy", b"x", zipfile.ZIP_STORED, 0o600 << 16)],
        [("dir\\rewards.npy", b"x", zipfile.ZIP_STORED, 0o600 << 16)],
        [("/rewards.npy", b"x", zipfile.ZIP_STORED, 0o600 << 16)],
        [("folder/", b"", zipfile.ZIP_STORED, (stat.S_IFDIR | 0o755) << 16)],
    ],
)
def test_missing_duplicate_casefold_traversal_backslash_absolute_and_directory_fail(
    entries: list[tuple[str, bytes, int, int]],
) -> None:
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain", external_npz=_zip(entries)
        )


@pytest.mark.unit
@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600])
def test_symlink_and_special_external_attributes_fail(mode: int) -> None:
    raw = _zip(
        [
            (
                "rewards.npy",
                _valid_member("<f2"),
                zipfile.ZIP_STORED,
                mode << 16,
            )
        ]
    )
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(candidate_id="external_dqn_plain", external_npz=raw)


@pytest.mark.unit
def test_encrypted_unsupported_compression_and_bad_crc_fail() -> None:
    raw = _zip()
    for malformed in (_patch_flags(raw, 1), _patch_method(raw, 99), _corrupt_stored_reward(raw)):
        with pytest.raises(bridge.ExternalResultBridgeError):
            bridge.convert_external_reward_npz(
                candidate_id="external_dqn_plain", external_npz=malformed
            )


@pytest.mark.unit
def test_zip_prefix_trailing_data_descriptor_local_name_zip64_and_deflate_crc_fail() -> None:
    raw = _zip()
    local_name_mismatch = bytearray(raw)
    _central_offset, central = _central_records(raw)[0]
    local_offset = int(central[16])
    local_name_mismatch[local_offset + 30] = ord("R")

    deflated = _zip(compression=zipfile.ZIP_DEFLATED)
    bad_deflate_crc = bytearray(deflated)
    deflate_central_offset, deflate_central = _central_records(deflated)[0]
    deflate_local_offset = int(deflate_central[16])
    wrong_crc = int(deflate_central[7]) ^ 1
    struct.pack_into("<I", bad_deflate_crc, deflate_local_offset + 14, wrong_crc)
    struct.pack_into("<I", bad_deflate_crc, deflate_central_offset + 16, wrong_crc)

    zip64 = _zip64_reward(_valid_member("<f2"))
    malformed_zip64 = bytearray(zip64)
    _zip64_central_offset, zip64_central = _central_records(zip64)[0]
    zip64_local_offset = int(zip64_central[16])
    zip64_name_size = int(struct.unpack_from("<H", zip64, zip64_local_offset + 26)[0])
    struct.pack_into(
        "<H",
        malformed_zip64,
        zip64_local_offset + 30 + zip64_name_size,
        2,
    )

    malformed_inputs = (
        b"prefix" + raw,
        raw + b"trailing",
        _patch_flags(raw, 1 << 3),
        bytes(local_name_mismatch),
        bytes(bad_deflate_crc),
        bytes(malformed_zip64),
    )
    for malformed in malformed_inputs:
        with pytest.raises(bridge.ExternalResultBridgeError):
            bridge.convert_external_reward_npz(
                candidate_id="external_dqn_plain",
                external_npz=malformed,
            )


@pytest.mark.unit
def test_npy_misaligned_header_and_float32_invalid_reward_fail() -> None:
    misaligned = bytearray(_valid_member("<f2"))
    original_header_size = int(struct.unpack_from("<H", misaligned, 8)[0])
    struct.pack_into("<H", misaligned, 8, original_header_size - 1)
    with pytest.raises(bridge.ExternalResultBridgeError, match="header length or alignment"):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(reward_member=bytes(misaligned)),
        )
    for value in (0.5, 2.0, float("nan"), float("inf")):
        with pytest.raises(bridge.ExternalResultBridgeError):
            bridge.convert_external_reward_npz(
                candidate_id="isolated_ppo_generic",
                external_npz=_zip(reward_member=_member_with_value("<f4", 17, value)),
            )


@pytest.mark.unit
def test_truncation_central_size_overlap_member_count_and_zip_bomb_bounds_fail() -> None:
    raw = _zip()
    central_size = bytearray(raw)
    struct.pack_into(
        "<I",
        central_size,
        len(central_size) - _ZIP_END.size + 12,
        _ZIP_END.unpack_from(raw, len(raw) - _ZIP_END.size)[5] + 1,
    )

    two = _zip(
        [
            ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16),
            ("extra.npy", b"x", zipfile.ZIP_STORED, 0o600 << 16),
        ]
    )
    overlap = bytearray(two)
    second_central_offset, _second = _central_records(two)[1]
    struct.pack_into("<I", overlap, second_central_offset + 42, 0)

    too_many = _zip(
        [("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16)]
        + [
            (f"unused{index}.bin", b"x", zipfile.ZIP_STORED, 0o600 << 16)
            for index in range(bridge.MAX_ZIP_MEMBER_COUNT)
        ]
    )
    declared_bomb = bytearray(
        _zip(
            [
                ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_DEFLATED, 0o600 << 16),
                ("unused.bin", b"small", zipfile.ZIP_DEFLATED, 0o600 << 16),
            ]
        )
    )
    bomb_central_offset, _bomb_record = _central_records(bytes(declared_bomb))[1]
    struct.pack_into(
        "<I",
        declared_bomb,
        bomb_central_offset + 24,
        bridge.MAX_ZIP_TOTAL_EXPANDED_BYTES + 1,
    )
    compressed_bomb = bytearray(
        _zip(
            [
                ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_DEFLATED, 0o600 << 16),
                ("unused.bin", b"small", zipfile.ZIP_DEFLATED, 0o600 << 16),
            ]
        )
    )
    compressed_central_offset, _compressed_record = _central_records(bytes(compressed_bomb))[1]
    struct.pack_into(
        "<I",
        compressed_bomb,
        compressed_central_offset + 20,
        bridge.MAX_ZIP_TOTAL_COMPRESSED_BYTES + 1,
    )
    malformed_inputs = (
        raw[:-1],
        bytes(central_size),
        bytes(overlap),
        too_many,
        bytes(declared_bomb),
        bytes(compressed_bomb),
    )
    for malformed in malformed_inputs:
        with pytest.raises(bridge.ExternalResultBridgeError):
            bridge.convert_external_reward_npz(
                candidate_id="external_dqn_plain", external_npz=malformed
            )


@pytest.mark.unit
def test_input_byte_bound_is_enforced_without_allocating_the_full_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _zip()
    monkeypatch.setattr(bridge, "MAX_EXTERNAL_NPZ_BYTES", len(raw) - 1)
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(candidate_id="external_dqn_plain", external_npz=raw)


@pytest.mark.unit
def test_inventory_reads_only_reward_member_but_still_bounds_all_members() -> None:
    raw = _zip(
        [
            ("rewards.npy", _valid_member("<f2"), zipfile.ZIP_STORED, 0o600 << 16),
            ("unused.bin", b"unused", zipfile.ZIP_STORED, 0o600 << 16),
        ]
    )
    changed = bytearray(raw)
    _unused_central_offset, unused = _central_records(raw)[1]
    local_offset = int(unused[16])
    name_size, extra_size = struct.unpack_from("<HH", raw, local_offset + 26)
    changed[local_offset + 30 + name_size + extra_size] ^= 1
    conversion = bridge.convert_external_reward_npz(
        candidate_id="external_dqn_plain", external_npz=bytes(changed)
    )
    assert conversion.cumulative_score == 29


@pytest.mark.unit
def test_receipt_rejects_bool_int_alias_noncanonical_duplicate_and_wrong_pin(
    continuing_case: tuple[bytes, bridge.ExternalRewardConversion],
) -> None:
    raw_npz, conversion = continuing_case
    receipt_raw = bridge.canonical_external_reward_conversion_receipt_bytes(conversion)
    payload = json.loads(receipt_raw)
    payload["trace"]["length"] = True
    body = copy.deepcopy(payload)
    body.pop("receipt_body_sha256")
    payload["receipt_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    bool_alias = _canonical(payload)
    duplicate = receipt_raw.replace(
        b'"status":',
        b'"status":"forged","status":',
        1,
    )
    noncanonical = json.dumps(json.loads(receipt_raw), indent=2, sort_keys=True).encode() + b"\n"
    for malformed in (bool_alias, duplicate, noncanonical):
        with pytest.raises(bridge.ExternalResultBridgeError):
            bridge.parse_external_reward_conversion_receipt(
                malformed,
                expected_file_sha256=hashlib.sha256(malformed).hexdigest(),
                candidate_id="external_dqn_plain",
                external_npz=raw_npz,
            )
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.parse_external_reward_conversion_receipt(
            receipt_raw,
            expected_file_sha256="1" * 64,
            candidate_id="external_dqn_plain",
            external_npz=raw_npz,
        )


@pytest.mark.unit
def test_mutated_scorer_function_cannot_mint_an_impossible_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeReceipt:
        cumulative_score = 99_999_999
        raw_trace_sha256 = "1" * 64
        artifact_sha256 = "2" * 64
        artifact_size_bytes = bridge.CANONICAL_SCORER_NPZ_SIZE_BYTES
        receipt_sha256 = "3" * 64

        def to_payload(self) -> dict[str, object]:
            return {
                "schema_version": scorer.SCORE_RECEIPT_SCHEMA_VERSION,
                "metric": {
                    "descriptor": {
                        "schema_version": bridge.SCORER_METRIC_SCHEMA_VERSION,
                    },
                    "sha256": bridge.SCORER_METRIC_SHA256,
                },
            }

    monkeypatch.setattr(scorer, "ingest_reward_npz_bytes", lambda _artifact: FakeReceipt())
    with pytest.raises(bridge.ExternalResultBridgeError, match="function closure drifted"):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(),
        )


@pytest.mark.unit
def test_mutated_scorer_code_object_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replacement(_artifact: bytes) -> object:
        return object()

    monkeypatch.setattr(
        scorer.ingest_reward_npz_bytes,
        "__code__",
        replacement.__code__,
    )
    with pytest.raises(bridge.ExternalResultBridgeError, match="function closure drifted"):
        bridge.external_result_bridge_descriptor()


@pytest.mark.unit
def test_mutated_scorer_receipt_property_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scorer.MatchedV3ScoreReceipt,
        "receipt_sha256",
        property(lambda _self: "1" * 64),
    )
    with pytest.raises(bridge.ExternalResultBridgeError, match="module/class identity drifted"):
        bridge.external_result_bridge_descriptor()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("owner", "name", "value"),
    [
        (scorer, "RAW_TRACE_ENCODING", "mutated_encoding"),
        (scorer, "NPZ_MEMBER_NAME", "not_rewards.npy"),
        (protocol, "MATCHED_V3_SCORE_MAXIMUM", 99_999_999),
    ],
)
def test_mutated_scorer_or_protocol_semantic_global_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    owner: object,
    name: str,
    value: object,
) -> None:
    monkeypatch.setattr(owner, name, value)
    with pytest.raises(bridge.ExternalResultBridgeError, match="semantic global surface drifted"):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(),
        )


@pytest.mark.unit
def test_mutated_dependency_source_path_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scorer, "__file__", __file__)
    with pytest.raises(bridge.ExternalResultBridgeError, match="loaded source path identity"):
        bridge.external_result_bridge_descriptor_sha256()


@pytest.mark.unit
def test_reassigned_bridge_candidate_order_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge,
        "EXTERNAL_RESULT_CANDIDATE_IDS",
        tuple(reversed(bridge.EXTERNAL_RESULT_CANDIDATE_IDS)),
    )
    with pytest.raises(bridge.ExternalResultBridgeError, match="bridge candidate"):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(),
        )


@pytest.mark.unit
def test_replaced_independent_score_function_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge,
        "_independent_trace_score_and_sha256",
        lambda _trace: (99_999_999, "1" * 64),
    )
    with pytest.raises(bridge.ExternalResultBridgeError, match="bridge function closure drifted"):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=_zip(),
        )


@pytest.mark.unit
def test_direct_and_post_init_bypassing_conversion_forgeries_are_rejected(
    continuing_case: tuple[bytes, bridge.ExternalRewardConversion],
) -> None:
    _raw, conversion = continuing_case
    with pytest.raises(bridge.ExternalResultBridgeError, match="cumulative score"):
        replace(conversion, cumulative_score=99_999_999)

    forged = object.__new__(bridge.ExternalRewardConversion)
    for field_name in (
        "candidate_id",
        "family",
        "external_dtype",
        "input_npz",
        "trace",
        "canonical_scorer_npz",
        "input_npz_sha256",
        "trace_bytes_sha256",
        "trace_sha256",
        "canonical_scorer_npz_sha256",
        "scorer_receipt_sha256",
    ):
        object.__setattr__(forged, field_name, getattr(conversion, field_name))
    object.__setattr__(forged, "cumulative_score", 99_999_999)
    with pytest.raises(bridge.ExternalResultBridgeError, match="cumulative score"):
        bridge.external_reward_conversion_receipt(forged)


@pytest.mark.unit
def test_unknown_candidate_mutable_input_and_default_input_are_rejected() -> None:
    raw = _zip()
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(candidate_id="other", external_npz=raw)
    with pytest.raises(bridge.ExternalResultBridgeError):
        bridge.convert_external_reward_npz(
            candidate_id="external_dqn_plain",
            external_npz=bytearray(raw),  # type: ignore[arg-type]
        )
    signature = inspect.signature(bridge.convert_external_reward_npz)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError):
        bridge.convert_external_reward_npz()  # type: ignore[call-arg]


@pytest.mark.unit
def test_no_execution_publication_path_capability_or_workload_imports_and_apis() -> None:
    source = inspect.getsource(bridge)
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    forbidden_import_fragments = (
        "subprocess",
        "pathlib",
        "zipfile",
        "configuration_plan",
        "external_materialization",
        "compiled_runner",
        "publication",
        "oci",
    )
    assert not any(
        fragment in imported for imported in imports for fragment in forbidden_import_fragments
    )
    public_names = set(bridge.__all__)
    assert not any(
        fragment in name.lower()
        for name in public_names
        for fragment in ("run", "workload", "publish", "path", "capability", "accept")
    )
    for name in (
        "run",
        "execute",
        "open_workload",
        "publish",
        "accept_result",
        "issue_capability",
        "from_path",
        "DEFAULT_INPUT",
    ):
        assert not hasattr(bridge, name)
