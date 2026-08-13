"""Score- and reward-bearing bridge from bounded external NPZ bytes to the v3 scorer.

The bridge inventories one generic in-memory ZIP before expanding anything, reads
only its sole ``rewards.npy`` member, and parses a deliberately narrow NPY subset.
The twelve upstream candidate identities select exactly one little-endian floating
reward dtype.  Finite integral rewards in ``{-1, 0, 1, 30}`` are converted in order
to immutable signed-int8 bytes and immediately round-tripped through the canonical
matched-v3 scorer artifact.

Every conversion exposes the input archive, the complete raw reward trace, the
canonical scorer archive, their content commitments, and the cumulative score.
It is therefore permanently forbidden as an input to any score-blind controller
or publisher.  A production caller must place it inside a fresh isolated,
post-qualification outcome-consumer/publisher process; the in-process integrity
checks below do not replace that isolation boundary.

This module has no runner, workload, process, OCI, caller-supplied filesystem path,
capability, publication, default-input, result-acceptance, qualification, or
evidence authority.  Its receipt records only a deterministic, score-bearing
content transformation and remains permanently nonqualifying and nonauthorizing.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import hmac
import json
import math
import os
import re
import stat
import struct
import types
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as _scorer
from alberta_framework.benchmarks import forager_matched_v3_protocol as _protocol

EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_result_bridge_descriptor.v1"
)
EXTERNAL_RESULT_BRIDGE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_result_conversion_receipt.v1"
)
EXTERNAL_RESULT_BRIDGE_STATUS: Final = (
    "implemented_score_reward_bearing_permanently_nonqualifying_non_authorizing"
)

EXTERNAL_RESULT_CANDIDATE_IDS: Final = (
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

_CONTINUING_CANDIDATE_IDS: Final = (
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
_PPO_CANDIDATE_IDS: Final = (
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
)
EXTERNAL_RESULT_CANDIDATE_FORMATS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "external_dqn_plain": ("continuing", "<f2"),
        "external_dqn_crelu": ("continuing", "<f2"),
        "external_dqn_redo": ("continuing", "<f2"),
        "external_dqn_reward_trace": ("continuing", "<f2"),
        "external_dqn_l2_init": ("continuing", "<f2"),
        "external_pt_dqn_xfinal": ("continuing", "<f2"),
        "external_drqn_xfinal": ("continuing", "<f2"),
        "isolated_ppo_generic": ("ppo", "<f4"),
        "isolated_rtu_paper_scale": ("ppo", "<f4"),
        "random_policy": ("continuing", "<f2"),
        "search_nearest": ("continuing", "<f2"),
        "search_oracle": ("continuing", "<f2"),
    }
)

MATCHED_V3_REWARD_HORIZON: Final = 499_712
CANONICAL_SCORER_NPZ_SIZE_BYTES: Final = 499_980
# Full-horizon upstream archives extrapolate to roughly 18--30 MiB from the
# inspected smoke outputs, and PPO carries 81 members.  Sixty-four MiB and 128
# members retain more than 2x size margin without making either dimension open.
MAX_EXTERNAL_NPZ_BYTES: Final = 64 * 1024 * 1024
MAX_ZIP_MEMBER_COUNT: Final = 128
MAX_ZIP_TOTAL_COMPRESSED_BYTES: Final = 64 * 1024 * 1024
MAX_ZIP_TOTAL_EXPANDED_BYTES: Final = 64 * 1024 * 1024
MAX_NPY_HEADER_BYTES: Final = 4 * 1024

SCORER_SOURCE_SHA256: Final = "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
SCORER_PROTOCOL_SOURCE_SHA256: Final = (
    "dd5db9a657ad167abf192942489642130b08bd065f724f7ad1b80743b1103720"
)
SCORER_METRIC_SCHEMA_VERSION: Final = "alberta.forager_cumulative_reward_metric.v1"
SCORER_METRIC_SHA256: Final = "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"

_SCORER_SOURCE_RELATIVE_PATH: Final = "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
_PROTOCOL_SOURCE_RELATIVE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_protocol.py"
)
_SOURCE_DIRECTORY_AT_IMPORT: Final = os.path.dirname(os.path.realpath(__file__))
_SCORER_SOURCE_PATH: Final = os.path.join(
    _SOURCE_DIRECTORY_AT_IMPORT, "_forager_matched_v3_scorer.py"
)
_PROTOCOL_SOURCE_PATH: Final = os.path.join(
    _SOURCE_DIRECTORY_AT_IMPORT, "forager_matched_v3_protocol.py"
)
_MAX_DEPENDENCY_SOURCE_BYTES: Final = 1024 * 1024
_SOURCE_READ_BYTES: Final = 64 * 1024

_PINNED_SCORE_RECEIPT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_score_receipt.v1"
_PINNED_NPZ_CONTAINER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_reward_npz.v1"
_PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_raw_reward_trace.int8.v1"
)
_PINNED_RAW_TRACE_ENCODING: Final = "signed_int8_twos_complement_c_order_one_byte_per_step"
_PINNED_RAW_TRACE_DIGEST_DOMAIN: Final = b"alberta.forager.matched_v3.raw_reward_trace.int8.v1"
_PINNED_CANONICAL_MEMBER_NAME: Final = "rewards.npy"
_PINNED_METRIC_DESCRIPTOR_BYTES: Final = (
    b'{"accumulation":"ordered_exact_integer_sum","aperture_size":9,'
    b'"environment_id":"ForagaxTwoBiomeLarge-v1","horizon":499712,'
    b'"observation_type":"color","ordered_difference_bounds":'
    b'{"maximum":15491072,"minimum":-15491072,"range_width":30982144},'
    b'"out_of_set_reward_rejected":true,"raw_reward_values":[-1,0,1,30],'
    b'"schema_version":"alberta.forager_cumulative_reward_metric.v1",'
    b'"score_bounds":{"maximum":14991360,"minimum":-499712},'
    b'"tail_or_ema_metric":false,"trace_completeness_required":true}\n'
)

_REWARD_MEMBER_NAME: Final = "rewards.npy"
_REWARD_SUPPORT: Final = (-1, 0, 1, 30)
_MAX_MEMBER_NAME_BYTES: Final = 255
_MAX_CENTRAL_DIRECTORY_BYTES: Final = 64 * 1024
_MAX_RECEIPT_BYTES: Final = 64 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 20_000
_MAX_JSON_TEXT: Final = 16 * 1024
_MAX_JSON_INTEGER_DIGITS: Final = 19
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

_ZIP_LOCAL = struct.Struct("<IHHHHHIIIHH")
_ZIP_CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")
_ZIP_END = struct.Struct("<IHHHHIIH")
_ZIP64_EXTRA = struct.Struct("<HHQQ")
_ZIP_LOCAL_SIGNATURE: Final = 0x04034B50
_ZIP_CENTRAL_SIGNATURE: Final = 0x02014B50
_ZIP_END_SIGNATURE: Final = 0x06054B50
_ZIP64_EXTRA_ID: Final = 1
_ZIP64_EXTRA_DATA_SIZE: Final = 16
_ZIP_SIZE_SENTINEL: Final = 0xFFFFFFFF
_ZIP_UTF8_FLAG: Final = 1 << 11
_ZIP_ENCRYPTED_FLAG: Final = 1
_ZIP_DATA_DESCRIPTOR_FLAG: Final = 1 << 3
_ZIP_ALLOWED_FLAGS: Final = _ZIP_UTF8_FLAG
_ZIP_STORED: Final = 0
_ZIP_DEFLATED: Final = 8
_ZIP_ALLOWED_METHODS: Final = (_ZIP_STORED, _ZIP_DEFLATED)

_CANONICAL_MEMBER_NAME_BYTES: Final = _PINNED_CANONICAL_MEMBER_NAME.encode("ascii")
_CANONICAL_NPY_DICTIONARY: Final = (
    f"{{'descr': '|i1', 'fortran_order': False, 'shape': ({MATCHED_V3_REWARD_HORIZON},), }}"
).encode("ascii")
_CANONICAL_NPY_HEADER_PAYLOAD_SIZE: Final = 118
_CANONICAL_NPY_HEADER_PAYLOAD: Final = (
    _CANONICAL_NPY_DICTIONARY
    + b" " * (_CANONICAL_NPY_HEADER_PAYLOAD_SIZE - len(_CANONICAL_NPY_DICTIONARY) - 1)
    + b"\n"
)
_CANONICAL_NPY_HEADER: Final = (
    b"\x93NUMPY\x01\x00"
    + struct.pack("<H", _CANONICAL_NPY_HEADER_PAYLOAD_SIZE)
    + _CANONICAL_NPY_HEADER_PAYLOAD
)
_CANONICAL_NPY_MEMBER_SIZE: Final = len(_CANONICAL_NPY_HEADER) + MATCHED_V3_REWARD_HORIZON
_CANONICAL_ZIP64_LOCAL_EXTRA: Final = struct.pack(
    "<HHQQ",
    _ZIP64_EXTRA_ID,
    _ZIP64_EXTRA_DATA_SIZE,
    _CANONICAL_NPY_MEMBER_SIZE,
    _CANONICAL_NPY_MEMBER_SIZE,
)
_CANONICAL_ZIP_DOS_DATE: Final = 33
_CANONICAL_ZIP_EXTERNAL_ATTRIBUTES: Final = 0x01800000
_CANONICAL_ZIP_DATA_OFFSET: Final = (
    _ZIP_LOCAL.size + len(_CANONICAL_MEMBER_NAME_BYTES) + len(_CANONICAL_ZIP64_LOCAL_EXTRA)
)
_CANONICAL_ZIP_CENTRAL_OFFSET: Final = _CANONICAL_ZIP_DATA_OFFSET + _CANONICAL_NPY_MEMBER_SIZE
_CANONICAL_ZIP_CENTRAL_SIZE: Final = _ZIP_CENTRAL.size + len(_CANONICAL_MEMBER_NAME_BYTES)
_CANONICAL_ZIP_END_OFFSET: Final = _CANONICAL_ZIP_CENTRAL_OFFSET + _CANONICAL_ZIP_CENTRAL_SIZE
if _CANONICAL_ZIP_END_OFFSET + _ZIP_END.size != CANONICAL_SCORER_NPZ_SIZE_BYTES:
    raise AssertionError("independent canonical scorer NPZ geometry drifted")


class ExternalResultBridgeError(ValueError):
    """External reward content violated the sealed bridge contract."""


_BRIDGE_ERROR_TYPE_AT_IMPORT: Final = ExternalResultBridgeError


@dataclass(frozen=True, slots=True)
class _FunctionIntegrity:
    function: types.FunctionType
    code: types.CodeType
    defaults: tuple[Any, ...] | None
    keyword_defaults: dict[str, Any] | None
    closure: tuple[types.CellType, ...] | None
    closure_contents: tuple[object, ...] | None
    builtins_mapping: dict[str, Any]


def _source_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_pinned_dependency_source(
    path: str,
    *,
    module: types.ModuleType,
    expected_sha256: str,
    label: str,
) -> bytes:
    module_path = getattr(module, "__file__", None)
    if (
        type(module_path) is not str
        or not os.path.isabs(module_path)
        or os.path.realpath(module_path) != path
    ):
        raise ExternalResultBridgeError(f"{label} loaded source path identity drifted")
    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_DEPENDENCY_SOURCE_BYTES
        ):
            raise ExternalResultBridgeError(f"{label} source descriptor identity is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_SOURCE_READ_BYTES, remaining))
            if not chunk:
                raise ExternalResultBridgeError(f"{label} source truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ExternalResultBridgeError(f"{label} source grew during read")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        raw = b"".join(chunks)
        if (
            _source_identity(before) != _source_identity(after)
            or _source_identity(before) != _source_identity(current)
            or len(raw) != before.st_size
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256)
        ):
            raise ExternalResultBridgeError(f"{label} source bytes or identity drifted")
        return raw
    except ExternalResultBridgeError:
        raise
    except OSError as exc:
        raise ExternalResultBridgeError(f"{label} source could not be read exactly") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stable_code_constant(value: Any) -> tuple[Any, ...]:
    if type(value) is types.CodeType:
        return ("code", _code_shape(value))
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value.hex())
    if type(value) is tuple:
        return ("tuple", tuple(_stable_code_constant(item) for item in value))
    if type(value) is frozenset:
        items = (_stable_code_constant(item) for item in value)
        return ("frozenset", tuple(sorted(items, key=repr)))
    return ("other", type(value).__module__, type(value).__qualname__, repr(value))


def _code_shape(code: types.CodeType) -> tuple[Any, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code.hex(),
        tuple(_stable_code_constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )


def _function_code_sha256(function: types.FunctionType) -> str:
    return hashlib.sha256(repr(_code_shape(function.__code__)).encode("ascii")).hexdigest()


def _current_dependency_functions() -> dict[str, types.FunctionType]:
    receipt_type = getattr(_scorer, "MatchedV3ScoreReceipt", None)
    if type(receipt_type) is not type:
        raise ExternalResultBridgeError("scorer receipt class identity drifted")
    receipt_property = vars(receipt_type).get("receipt_sha256")
    if (
        type(receipt_property) is not property
        or type(receipt_property.fget) is not types.FunctionType
    ):
        raise ExternalResultBridgeError("scorer receipt digest property identity drifted")
    candidates: dict[str, object] = {
        "scorer.canonical_reward_npz_bytes": getattr(_scorer, "canonical_reward_npz_bytes", None),
        "scorer.extract_canonical_reward_trace": getattr(
            _scorer, "extract_canonical_reward_trace", None
        ),
        "scorer.ingest_reward_npz_bytes": getattr(_scorer, "ingest_reward_npz_bytes", None),
        "scorer._trace_score_and_sha256": getattr(_scorer, "_trace_score_and_sha256", None),
        "scorer._extract_canonical_trace": getattr(_scorer, "_extract_canonical_trace", None),
        "scorer._bound_metric_descriptor": getattr(_scorer, "_bound_metric_descriptor", None),
        "scorer._unpack_exact": getattr(_scorer, "_unpack_exact", None),
        "scorer._sha256": getattr(_scorer, "_sha256", None),
        "scorer._require_exact_integer": getattr(_scorer, "_require_exact_integer", None),
        "scorer._require_sha256": getattr(_scorer, "_require_sha256", None),
        "scorer._authority_denial": getattr(_scorer, "_authority_denial", None),
        "scorer._canonical_json_bytes": getattr(_scorer, "_canonical_json_bytes", None),
        "scorer.MatchedV3ScoreReceipt.__init__": getattr(receipt_type, "__init__", None),
        "scorer.MatchedV3ScoreReceipt.__post_init__": getattr(receipt_type, "__post_init__", None),
        "scorer.MatchedV3ScoreReceipt.to_body": getattr(receipt_type, "to_body", None),
        "scorer.MatchedV3ScoreReceipt.canonical_body": getattr(
            receipt_type, "canonical_body", None
        ),
        "scorer.MatchedV3ScoreReceipt.to_payload": getattr(receipt_type, "to_payload", None),
        "scorer.MatchedV3ScoreReceipt.canonical_json": getattr(
            receipt_type, "canonical_json", None
        ),
        "scorer.MatchedV3ScoreReceipt.receipt_sha256.fget": receipt_property.fget,
        "protocol.cumulative_reward_metric_descriptor": getattr(
            _protocol, "cumulative_reward_metric_descriptor", None
        ),
        "protocol.canonical_cumulative_reward_metric_bytes": getattr(
            _protocol, "canonical_cumulative_reward_metric_bytes", None
        ),
        "protocol.validate_cumulative_reward_score": getattr(
            _protocol, "validate_cumulative_reward_score", None
        ),
    }
    result: dict[str, types.FunctionType] = {}
    for label, candidate in candidates.items():
        if type(candidate) is not types.FunctionType:
            raise ExternalResultBridgeError(f"dependency callable identity drifted: {label}")
        result[label] = candidate
    return result


_EXPECTED_DEPENDENCY_CODE_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scorer.canonical_reward_npz_bytes": (
            "a87f85bdc96dcae4da51bd24381d6e47c31a2bbad94bc4765de3ae4d0ecda83d"
        ),
        "scorer.extract_canonical_reward_trace": (
            "eeb0f1a41ad64fbd3d8ac6a1fd626bcf69ea9e1501228ce07c17e62e585664cd"
        ),
        "scorer.ingest_reward_npz_bytes": (
            "0341f7c0bf6c9abd4f60c9ef3399ee54a4bc8fdc9efd38512934fdabd0ccad6d"
        ),
        "scorer._trace_score_and_sha256": (
            "0dcfe3bf8f370d6620896f3c201d0daba0c192d44eca3d43d6ab24ba26b6895e"
        ),
        "scorer._extract_canonical_trace": (
            "a8c4551b7fd57b8b7a3531beae1981856990f864cc5ab66843fc4b03cf870edd"
        ),
        "scorer._bound_metric_descriptor": (
            "526bef9aeeed6f0387da5755a36e5b47e81eced6d40ce84007e838bcd0a2d6d9"
        ),
        "scorer._unpack_exact": (
            "421628cea2da15696b08a93819a06c63feb75872881ba06b3e81bd0ffad47508"
        ),
        "scorer._sha256": "7b1d8513201e539a177bf62a4452e48db49ccb5e52d05092fa6941f9dd68276a",
        "scorer._require_exact_integer": (
            "52e75d7e98de04f6dfd69debb8b3462965c1c6f7eec39a4c853090bb66bfe38f"
        ),
        "scorer._require_sha256": (
            "fbf72d0e823fa065c151a9785eb33755853afef883b9356038ed2a2b5f729ce6"
        ),
        "scorer._authority_denial": (
            "809d678bf1fa2d5dadf18b6e12caba87aea263b31a0421b51eb9c93464fa04cf"
        ),
        "scorer._canonical_json_bytes": (
            "45abcde0a5806bc6cccf258f80ad6bc40f77b4a3836032b36c0043cee8d02e16"
        ),
        "scorer.MatchedV3ScoreReceipt.__init__": (
            "5a327f370bc8bef6c50121f597e81431eacbe751f029a96b9ab22c33ed9262a4"
        ),
        "scorer.MatchedV3ScoreReceipt.__post_init__": (
            "8665b14a55bd5638c42ccbea90beff3f4a71cf3192268cf2d2a1620a872d4006"
        ),
        "scorer.MatchedV3ScoreReceipt.to_body": (
            "ea925178be77723c667d194e2cbfa516ba57b32db3cdeb3b74cae4d76882333f"
        ),
        "scorer.MatchedV3ScoreReceipt.canonical_body": (
            "3f1c68b6daa4666e123ec28e586ad242b26c625146ce6a377e2cdca5badedb6c"
        ),
        "scorer.MatchedV3ScoreReceipt.to_payload": (
            "acc289df3405720f7901df3ffabc3a3d8d052f1587c97992838eaddc9d1c377e"
        ),
        "scorer.MatchedV3ScoreReceipt.canonical_json": (
            "f050000a66c93df1ae29eac5489acfcf3293708f4caaa18d8606509edc36d488"
        ),
        "scorer.MatchedV3ScoreReceipt.receipt_sha256.fget": (
            "c4c331da4f749732b621182e5ba52c75a632f8ed7e351c4ab1ced1f0e4544b58"
        ),
        "protocol.cumulative_reward_metric_descriptor": (
            "fee3b22faa3e9af1c915539d502d3d3bc7f44e50dae10f099c840d5a7003a9e8"
        ),
        "protocol.canonical_cumulative_reward_metric_bytes": (
            "beb1489e817921811c3a92380410e2308c6d6d254b5dab038de9f02c95ee3b21"
        ),
        "protocol.validate_cumulative_reward_score": (
            "92dc0a56aaf1f691f200291748e4e25c167199548996170180cfbb1f1cf3162b"
        ),
    }
)


def _capture_function_integrity(function: types.FunctionType) -> _FunctionIntegrity:
    closure = function.__closure__
    closure_contents = None if closure is None else tuple(cell.cell_contents for cell in closure)
    return _FunctionIntegrity(
        function=function,
        code=function.__code__,
        defaults=function.__defaults__,
        keyword_defaults=function.__kwdefaults__,
        closure=closure,
        closure_contents=closure_contents,
        builtins_mapping=cast(dict[str, Any], getattr(function, "__builtins__")),
    )


_SCORER_MODULE_AT_IMPORT: Final = _scorer
_PROTOCOL_MODULE_AT_IMPORT: Final = _protocol
_SCORER_RECEIPT_TYPE_AT_IMPORT: Final = _scorer.MatchedV3ScoreReceipt
_SCORER_ERROR_TYPE_AT_IMPORT: Final = _scorer.ForagerMatchedV3ScorerError
_PROTOCOL_ERROR_TYPE_AT_IMPORT: Final = _protocol.ForagerMatchedV3ProtocolError
_DEPENDENCY_FUNCTION_BASELINE: Final[Mapping[str, _FunctionIntegrity]] = MappingProxyType(
    {
        label: _capture_function_integrity(function)
        for label, function in _current_dependency_functions().items()
    }
)
_SCORER_RECEIPT_PROPERTY_AT_IMPORT: Final = vars(_SCORER_RECEIPT_TYPE_AT_IMPORT)["receipt_sha256"]
_SCORER_RECEIPT_CLASS_SURFACE_AT_IMPORT: Final[Mapping[str, object]] = MappingProxyType(
    dict(vars(_SCORER_RECEIPT_TYPE_AT_IMPORT))
)
_DEPENDENCY_OBJECT_BASELINE: Final[Mapping[str, object]] = MappingProxyType(
    {
        "scorer.protocol": getattr(_scorer, "protocol"),
        "scorer.hashlib": getattr(_scorer, "hashlib"),
        "scorer.json": getattr(_scorer, "json"),
        "scorer.struct": getattr(_scorer, "struct"),
        "scorer.zlib": getattr(_scorer, "zlib"),
        "scorer.ForagerMatchedV3ScorerError": _SCORER_ERROR_TYPE_AT_IMPORT,
        "scorer.MatchedV3ScoreReceipt": _SCORER_RECEIPT_TYPE_AT_IMPORT,
        "scorer._SHA256": _scorer._SHA256,
        "scorer._ZIP_LOCAL_HEADER": _scorer._ZIP_LOCAL_HEADER,
        "scorer._ZIP_CENTRAL_HEADER": _scorer._ZIP_CENTRAL_HEADER,
        "scorer._ZIP_END_RECORD": _scorer._ZIP_END_RECORD,
        "protocol.hashlib": getattr(_protocol, "hashlib"),
        "protocol.json": getattr(_protocol, "json"),
        "protocol.cast": getattr(_protocol, "cast"),
        "protocol.Any": getattr(_protocol, "Any"),
        "protocol.ForagerMatchedV3ProtocolError": _PROTOCOL_ERROR_TYPE_AT_IMPORT,
        "stdlib.hashlib.sha256": hashlib.sha256,
        "stdlib.hmac.compare_digest": hmac.compare_digest,
        "stdlib.json.dumps": json.dumps,
        "stdlib.json.loads": json.loads,
        "stdlib.struct.iter_unpack": struct.iter_unpack,
        "stdlib.struct.error": struct.error,
        "stdlib.zlib.crc32": zlib.crc32,
        "stdlib.zlib.decompressobj": zlib.decompressobj,
        "builtins.len": builtins.len,
        "builtins.type": builtins.type,
        "builtins.int": builtins.int,
        "builtins.bytes": builtins.bytes,
        "builtins.list": builtins.list,
        "builtins.dict": builtins.dict,
        "builtins.str": builtins.str,
        "builtins.enumerate": builtins.enumerate,
        "builtins.AssertionError": builtins.AssertionError,
    }
)


def _dependency_semantic_payload() -> tuple[tuple[str, type[Any], Any], ...]:
    values: dict[str, Any] = {
        "scorer.SCORE_RECEIPT_SCHEMA_VERSION": _scorer.SCORE_RECEIPT_SCHEMA_VERSION,
        "scorer.NPZ_CONTAINER_SCHEMA_VERSION": _scorer.NPZ_CONTAINER_SCHEMA_VERSION,
        "scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION": _scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION,
        "scorer.RAW_TRACE_ENCODING": _scorer.RAW_TRACE_ENCODING,
        "scorer.RAW_TRACE_DIGEST_DOMAIN": _scorer.RAW_TRACE_DIGEST_DOMAIN,
        "scorer.NPZ_MEMBER_NAME": _scorer.NPZ_MEMBER_NAME,
        "scorer.CANONICAL_NPZ_SIZE_BYTES": _scorer.CANONICAL_NPZ_SIZE_BYTES,
        "scorer._MAX_RECEIPT_BYTES": _scorer._MAX_RECEIPT_BYTES,
        "scorer._NPY_FORMAT_HORIZON": _scorer._NPY_FORMAT_HORIZON,
        "scorer._CANONICAL_NPY_HEADER": _scorer._CANONICAL_NPY_HEADER,
        "scorer._NPY_HEADER_SIZE": _scorer._NPY_HEADER_SIZE,
        "scorer._NPY_MEMBER_SIZE": _scorer._NPY_MEMBER_SIZE,
        "scorer._ZIP64_LOCAL_EXTRA": _scorer._ZIP64_LOCAL_EXTRA,
        "scorer._MEMBER_NAME_BYTES": _scorer._MEMBER_NAME_BYTES,
        "scorer._ZIP_LOCAL_SIGNATURE": _scorer._ZIP_LOCAL_SIGNATURE,
        "scorer._ZIP_CENTRAL_SIGNATURE": _scorer._ZIP_CENTRAL_SIGNATURE,
        "scorer._ZIP_END_SIGNATURE": _scorer._ZIP_END_SIGNATURE,
        "scorer._ZIP_DOS_DATE_1980_01_01": _scorer._ZIP_DOS_DATE_1980_01_01,
        "scorer._ZIP_EXTERNAL_ATTR": _scorer._ZIP_EXTERNAL_ATTR,
        "scorer._ZIP_DATA_OFFSET": _scorer._ZIP_DATA_OFFSET,
        "scorer._ZIP_CENTRAL_OFFSET": _scorer._ZIP_CENTRAL_OFFSET,
        "scorer._ZIP_CENTRAL_SIZE": _scorer._ZIP_CENTRAL_SIZE,
        "scorer._ZIP_END_OFFSET": _scorer._ZIP_END_OFFSET,
        "protocol.MATCHED_V3_HORIZON": _protocol.MATCHED_V3_HORIZON,
        "protocol.MATCHED_V3_RAW_REWARD_VALUES": _protocol.MATCHED_V3_RAW_REWARD_VALUES,
        "protocol.MATCHED_V3_SCORE_MINIMUM": _protocol.MATCHED_V3_SCORE_MINIMUM,
        "protocol.MATCHED_V3_SCORE_MAXIMUM": _protocol.MATCHED_V3_SCORE_MAXIMUM,
        "protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION": (
            _protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
        ),
        "protocol.CUMULATIVE_REWARD_METRIC_SHA256": (_protocol.CUMULATIVE_REWARD_METRIC_SHA256),
        "protocol._CUMULATIVE_REWARD_METRIC_BYTES": (_protocol._CUMULATIVE_REWARD_METRIC_BYTES),
    }
    return tuple((name, type(value), value) for name, value in sorted(values.items()))


_DEPENDENCY_SEMANTIC_BASELINE: Final = _dependency_semantic_payload()


def _require_bridge_semantics() -> None:
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
    expected_continuing = (
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
    expected_ppo = ("isolated_ppo_generic", "isolated_rtu_paper_scale")
    expected_formats = (
        ("external_dqn_plain", ("continuing", "<f2")),
        ("external_dqn_crelu", ("continuing", "<f2")),
        ("external_dqn_redo", ("continuing", "<f2")),
        ("external_dqn_reward_trace", ("continuing", "<f2")),
        ("external_dqn_l2_init", ("continuing", "<f2")),
        ("external_pt_dqn_xfinal", ("continuing", "<f2")),
        ("external_drqn_xfinal", ("continuing", "<f2")),
        ("isolated_ppo_generic", ("ppo", "<f4")),
        ("isolated_rtu_paper_scale", ("ppo", "<f4")),
        ("random_policy", ("continuing", "<f2")),
        ("search_nearest", ("continuing", "<f2")),
        ("search_oracle", ("continuing", "<f2")),
    )
    if (
        type(EXTERNAL_RESULT_CANDIDATE_IDS) is not tuple
        or EXTERNAL_RESULT_CANDIDATE_IDS != expected_order
        or type(_CONTINUING_CANDIDATE_IDS) is not tuple
        or _CONTINUING_CANDIDATE_IDS != expected_continuing
        or type(_PPO_CANDIDATE_IDS) is not tuple
        or _PPO_CANDIDATE_IDS != expected_ppo
        or type(EXTERNAL_RESULT_CANDIDATE_FORMATS) is not MappingProxyType
        or tuple(EXTERNAL_RESULT_CANDIDATE_FORMATS.items()) != expected_formats
        or MATCHED_V3_REWARD_HORIZON != 499_712
        or CANONICAL_SCORER_NPZ_SIZE_BYTES != 499_980
        or MAX_EXTERNAL_NPZ_BYTES != 64 * 1024 * 1024
        or MAX_ZIP_MEMBER_COUNT != 128
        or MAX_ZIP_TOTAL_COMPRESSED_BYTES != 64 * 1024 * 1024
        or MAX_ZIP_TOTAL_EXPANDED_BYTES != 64 * 1024 * 1024
        or MAX_NPY_HEADER_BYTES != 4 * 1024
        or _REWARD_SUPPORT != (-1, 0, 1, 30)
        or _REWARD_MEMBER_NAME != "rewards.npy"
        or _ZIP_LOCAL.format != "<IHHHHHIIIHH"
        or _ZIP_CENTRAL.format != "<IHHHHHHIIIHHHHHII"
        or _ZIP_END.format != "<IHHHHIIH"
        or _PINNED_SCORE_RECEIPT_SCHEMA_VERSION != "alberta.forager_matched_v3_score_receipt.v1"
        or _PINNED_NPZ_CONTAINER_SCHEMA_VERSION != "alberta.forager_matched_v3_reward_npz.v1"
        or _PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION
        != "alberta.forager_matched_v3_raw_reward_trace.int8.v1"
        or _PINNED_RAW_TRACE_ENCODING != "signed_int8_twos_complement_c_order_one_byte_per_step"
        or _PINNED_RAW_TRACE_DIGEST_DOMAIN != b"alberta.forager.matched_v3.raw_reward_trace.int8.v1"
        or _PINNED_CANONICAL_MEMBER_NAME != "rewards.npy"
        or SCORER_SOURCE_SHA256
        != "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
        or SCORER_PROTOCOL_SOURCE_SHA256
        != "dd5db9a657ad167abf192942489642130b08bd065f724f7ad1b80743b1103720"
        or SCORER_METRIC_SCHEMA_VERSION != "alberta.forager_cumulative_reward_metric.v1"
        or SCORER_METRIC_SHA256
        != "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
        or hashlib.sha256(_PINNED_METRIC_DESCRIPTOR_BYTES).hexdigest() != SCORER_METRIC_SHA256
        or EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION
        != "alberta.forager_matched_v3.external_result_bridge_descriptor.v1"
        or EXTERNAL_RESULT_BRIDGE_RECEIPT_SCHEMA_VERSION
        != "alberta.forager_matched_v3.external_result_conversion_receipt.v1"
        or EXTERNAL_RESULT_BRIDGE_STATUS
        != "implemented_score_reward_bearing_permanently_nonqualifying_non_authorizing"
        or EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256
        != "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
        or type(globals().get("_DESCRIPTOR_BYTES")) is not bytes
        or hashlib.sha256(cast(bytes, globals().get("_DESCRIPTOR_BYTES"))).hexdigest()
        != EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256
        or globals().get("ExternalResultBridgeError") is not _BRIDGE_ERROR_TYPE_AT_IMPORT
        or globals().get("_ZipMember") is not _ZIP_MEMBER_TYPE_AT_IMPORT
        or globals().get("ExternalRewardConversion")
        is not _EXTERNAL_REWARD_CONVERSION_TYPE_AT_IMPORT
    ):
        raise ExternalResultBridgeError("bridge candidate, parser, or scorer semantics drifted")


def _current_bridge_function_surface() -> dict[str, types.FunctionType]:
    names = (
        "_source_identity",
        "_read_pinned_dependency_source",
        "_stable_code_constant",
        "_code_shape",
        "_function_code_sha256",
        "_current_dependency_functions",
        "_capture_function_integrity",
        "_dependency_semantic_payload",
        "_require_bridge_semantics",
        "_current_bridge_function_surface",
        "_require_dependency_integrity",
        "_raise_json_constant",
        "_raise_json_float",
        "_parse_bounded_json_int",
        "_object_without_duplicates",
        "_assert_plain_unaliased_json",
        "_exact_json_equal",
        "_canonical_json",
        "_strict_json_load",
        "_require_sha256",
        "_unpack",
        "_safe_member_name",
        "_validate_external_attributes",
        "_validate_local_record",
        "_inventory_zip",
        "_read_reward_member",
        "_parse_npy_header",
        "_trace_from_npy",
        "_independent_trace_score_and_sha256",
        "_independent_canonical_reward_npz",
        "_expected_scorer_receipt_payload",
        "_claims",
        "_limitations",
        "_candidate_contract",
        "_descriptor",
        "_verify_scorer_receipt",
        "_derive_conversion",
        "_validate_conversion",
        "_receipt_body",
        "_receipt_payload",
        "external_result_bridge_descriptor",
        "canonical_external_result_bridge_descriptor_bytes",
        "external_result_bridge_descriptor_sha256",
        "parse_external_result_bridge_descriptor",
        "convert_external_reward_npz",
        "external_reward_conversion_receipt",
        "canonical_external_reward_conversion_receipt_bytes",
        "external_reward_conversion_receipt_sha256",
        "parse_external_reward_conversion_receipt",
    )
    result: dict[str, types.FunctionType] = {}
    for name in names:
        candidate = globals().get(name)
        if type(candidate) is not types.FunctionType:
            raise ExternalResultBridgeError(f"bridge function identity drifted: {name}")
        result[name] = candidate
    return result


def _require_dependency_integrity() -> None:
    _require_bridge_semantics()
    if (
        _scorer is not _SCORER_MODULE_AT_IMPORT
        or _protocol is not _PROTOCOL_MODULE_AT_IMPORT
        or getattr(_scorer, "protocol", None) is not _PROTOCOL_MODULE_AT_IMPORT
        or getattr(_scorer, "MatchedV3ScoreReceipt", None) is not _SCORER_RECEIPT_TYPE_AT_IMPORT
        or getattr(_scorer, "ForagerMatchedV3ScorerError", None) is not _SCORER_ERROR_TYPE_AT_IMPORT
        or getattr(_protocol, "ForagerMatchedV3ProtocolError", None)
        is not _PROTOCOL_ERROR_TYPE_AT_IMPORT
        or vars(_SCORER_RECEIPT_TYPE_AT_IMPORT).get("receipt_sha256")
        is not _SCORER_RECEIPT_PROPERTY_AT_IMPORT
    ):
        raise ExternalResultBridgeError("scorer or protocol module/class identity drifted")
    current_objects = {
        "scorer.protocol": getattr(_scorer, "protocol", None),
        "scorer.hashlib": getattr(_scorer, "hashlib", None),
        "scorer.json": getattr(_scorer, "json", None),
        "scorer.struct": getattr(_scorer, "struct", None),
        "scorer.zlib": getattr(_scorer, "zlib", None),
        "scorer.ForagerMatchedV3ScorerError": getattr(_scorer, "ForagerMatchedV3ScorerError", None),
        "scorer.MatchedV3ScoreReceipt": getattr(_scorer, "MatchedV3ScoreReceipt", None),
        "scorer._SHA256": getattr(_scorer, "_SHA256", None),
        "scorer._ZIP_LOCAL_HEADER": getattr(_scorer, "_ZIP_LOCAL_HEADER", None),
        "scorer._ZIP_CENTRAL_HEADER": getattr(_scorer, "_ZIP_CENTRAL_HEADER", None),
        "scorer._ZIP_END_RECORD": getattr(_scorer, "_ZIP_END_RECORD", None),
        "protocol.hashlib": getattr(_protocol, "hashlib", None),
        "protocol.json": getattr(_protocol, "json", None),
        "protocol.cast": getattr(_protocol, "cast", None),
        "protocol.Any": getattr(_protocol, "Any", None),
        "protocol.ForagerMatchedV3ProtocolError": getattr(
            _protocol, "ForagerMatchedV3ProtocolError", None
        ),
        "stdlib.hashlib.sha256": getattr(hashlib, "sha256", None),
        "stdlib.hmac.compare_digest": getattr(hmac, "compare_digest", None),
        "stdlib.json.dumps": getattr(json, "dumps", None),
        "stdlib.json.loads": getattr(json, "loads", None),
        "stdlib.struct.iter_unpack": getattr(struct, "iter_unpack", None),
        "stdlib.struct.error": getattr(struct, "error", None),
        "stdlib.zlib.crc32": getattr(zlib, "crc32", None),
        "stdlib.zlib.decompressobj": getattr(zlib, "decompressobj", None),
        "builtins.len": getattr(builtins, "len", None),
        "builtins.type": getattr(builtins, "type", None),
        "builtins.int": getattr(builtins, "int", None),
        "builtins.bytes": getattr(builtins, "bytes", None),
        "builtins.list": getattr(builtins, "list", None),
        "builtins.dict": getattr(builtins, "dict", None),
        "builtins.str": getattr(builtins, "str", None),
        "builtins.enumerate": getattr(builtins, "enumerate", None),
        "builtins.AssertionError": getattr(builtins, "AssertionError", None),
    }
    for label, expected in _DEPENDENCY_OBJECT_BASELINE.items():
        if current_objects.get(label) is not expected:
            raise ExternalResultBridgeError(f"dependency object identity drifted: {label}")
    if _dependency_semantic_payload() != _DEPENDENCY_SEMANTIC_BASELINE:
        raise ExternalResultBridgeError("scorer or protocol semantic global surface drifted")
    if (
        _scorer.SCORE_RECEIPT_SCHEMA_VERSION != _PINNED_SCORE_RECEIPT_SCHEMA_VERSION
        or _scorer.NPZ_CONTAINER_SCHEMA_VERSION != _PINNED_NPZ_CONTAINER_SCHEMA_VERSION
        or _scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION != _PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION
        or _scorer.RAW_TRACE_ENCODING != _PINNED_RAW_TRACE_ENCODING
        or _scorer.RAW_TRACE_DIGEST_DOMAIN != _PINNED_RAW_TRACE_DIGEST_DOMAIN
        or _scorer.NPZ_MEMBER_NAME != _PINNED_CANONICAL_MEMBER_NAME
        or _scorer.CANONICAL_NPZ_SIZE_BYTES != CANONICAL_SCORER_NPZ_SIZE_BYTES
        or _protocol.MATCHED_V3_HORIZON != MATCHED_V3_REWARD_HORIZON
        or _protocol.MATCHED_V3_RAW_REWARD_VALUES != _REWARD_SUPPORT
        or _protocol.MATCHED_V3_SCORE_MINIMUM != -MATCHED_V3_REWARD_HORIZON
        or _protocol.MATCHED_V3_SCORE_MAXIMUM != 30 * MATCHED_V3_REWARD_HORIZON
        or _protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION != SCORER_METRIC_SCHEMA_VERSION
        or _protocol.CUMULATIVE_REWARD_METRIC_SHA256 != SCORER_METRIC_SHA256
        or _protocol._CUMULATIVE_REWARD_METRIC_BYTES != _PINNED_METRIC_DESCRIPTOR_BYTES
    ):
        raise ExternalResultBridgeError("pinned scorer or protocol semantics drifted")
    fields = dataclass_fields(_SCORER_RECEIPT_TYPE_AT_IMPORT)
    current_class_surface = vars(_SCORER_RECEIPT_TYPE_AT_IMPORT)
    if (
        current_class_surface.keys() != _SCORER_RECEIPT_CLASS_SURFACE_AT_IMPORT.keys()
        or any(
            current_class_surface[name] is not expected
            for name, expected in _SCORER_RECEIPT_CLASS_SURFACE_AT_IMPORT.items()
        )
        or tuple(field.name for field in fields)
        != ("cumulative_score", "raw_trace_sha256", "artifact_sha256", "artifact_size_bytes")
        or tuple(getattr(_SCORER_RECEIPT_TYPE_AT_IMPORT, "__slots__", ()))
        != ("cumulative_score", "raw_trace_sha256", "artifact_sha256", "artifact_size_bytes")
    ):
        raise ExternalResultBridgeError("scorer receipt dataclass surface drifted")
    current_functions = _current_dependency_functions()
    if current_functions.keys() != _DEPENDENCY_FUNCTION_BASELINE.keys():
        raise ExternalResultBridgeError("dependency function inventory drifted")
    for label, current in current_functions.items():
        expected = _DEPENDENCY_FUNCTION_BASELINE[label]
        expected_globals = (
            _PROTOCOL_MODULE_AT_IMPORT.__dict__
            if label.startswith("protocol.")
            else _SCORER_MODULE_AT_IMPORT.__dict__
        )
        if (
            current is not expected.function
            or current.__code__ is not expected.code
            or current.__globals__ is not expected_globals
            or current.__defaults__ is not expected.defaults
            or current.__kwdefaults__ is not expected.keyword_defaults
            or current.__closure__ is not expected.closure
            or getattr(current, "__builtins__", None) is not expected.builtins_mapping
            or getattr(current, "__builtins__", None) is not builtins.__dict__
            or _function_code_sha256(current) != _EXPECTED_DEPENDENCY_CODE_SHA256[label]
        ):
            raise ExternalResultBridgeError(f"dependency function closure drifted: {label}")
        if expected.closure is not None:
            assert expected.closure_contents is not None
            for cell, expected_cell, expected_content in zip(
                current.__closure__ or (),
                expected.closure,
                expected.closure_contents,
                strict=True,
            ):
                if cell is not expected_cell or cell.cell_contents is not expected_content:
                    raise ExternalResultBridgeError(
                        f"dependency function closure cell drifted: {label}"
                    )
    _read_pinned_dependency_source(
        _SCORER_SOURCE_PATH,
        module=_SCORER_MODULE_AT_IMPORT,
        expected_sha256=SCORER_SOURCE_SHA256,
        label="matched-v3 scorer",
    )
    _read_pinned_dependency_source(
        _PROTOCOL_SOURCE_PATH,
        module=_PROTOCOL_MODULE_AT_IMPORT,
        expected_sha256=SCORER_PROTOCOL_SOURCE_SHA256,
        label="matched-v3 scorer protocol",
    )
    try:
        bridge_baseline = _BRIDGE_FUNCTION_BASELINE
    except NameError as exc:
        raise ExternalResultBridgeError(
            "bridge function integrity baseline is unavailable"
        ) from exc
    current_bridge = _current_bridge_function_surface()
    if current_bridge.keys() != bridge_baseline.keys():
        raise ExternalResultBridgeError("bridge function inventory drifted")
    for name, expected in bridge_baseline.items():
        current = current_bridge[name]
        if (
            current is not expected.function
            or current.__code__ is not expected.code
            or current.__globals__ is not globals()
            or current.__defaults__ is not expected.defaults
            or current.__kwdefaults__ is not expected.keyword_defaults
            or current.__closure__ is not expected.closure
            or getattr(current, "__builtins__", None) is not expected.builtins_mapping
        ):
            raise ExternalResultBridgeError(f"bridge function closure drifted: {name}")


@dataclass(frozen=True, slots=True)
class _ZipMember:
    name: str
    compression: int
    flags: int
    crc32: int
    compressed_size: int
    expanded_size: int
    local_offset: int
    data_offset: int
    data_end: int


_ZIP_MEMBER_TYPE_AT_IMPORT: Final = _ZipMember


@dataclass(frozen=True, slots=True)
class ExternalRewardConversion:
    """Immutable score/reward-bearing transformation with no result authority."""

    candidate_id: str
    family: str
    external_dtype: str
    input_npz: bytes = field(repr=False)
    trace: bytes = field(repr=False)
    canonical_scorer_npz: bytes = field(repr=False)
    input_npz_sha256: str
    trace_bytes_sha256: str
    trace_sha256: str
    canonical_scorer_npz_sha256: str
    cumulative_score: int
    scorer_receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_conversion(self)


_EXTERNAL_REWARD_CONVERSION_TYPE_AT_IMPORT: Final = ExternalRewardConversion


def _raise_json_constant(value: str) -> NoReturn:
    raise ExternalResultBridgeError(f"receipt contains forbidden constant {value!r}")


def _raise_json_float(value: str) -> NoReturn:
    raise ExternalResultBridgeError(f"receipt contains forbidden float {value!r}")


def _parse_bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        raise ExternalResultBridgeError("receipt integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalResultBridgeError(f"receipt contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ExternalResultBridgeError("receipt exceeds its JSON node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ExternalResultBridgeError("receipt exceeds its JSON depth bound")
        if type(item) is str:
            if len(item) > _MAX_JSON_TEXT or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                raise ExternalResultBridgeError("receipt strings must be bounded printable ASCII")
            continue
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) not in {dict, list}:
            raise ExternalResultBridgeError("receipt contains a non-plain JSON value")
        identity = id(item)
        if identity in seen:
            raise ExternalResultBridgeError("receipt contains an aliased or cyclic container")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
            continue
        mapping = cast(dict[Any, Any], item)
        if any(type(key) is not str for key in mapping):
            raise ExternalResultBridgeError("receipt keys must be exact strings")
        for key, child in mapping.items():
            pending.append((key, depth + 1))
            pending.append((child, depth + 1))


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[str, Any], left)
        right_mapping = cast(dict[str, Any], right)
        return left_mapping.keys() == right_mapping.keys() and all(
            _exact_json_equal(left_mapping[key], right_mapping[key]) for key in left_mapping
        )
    if type(left) is list:
        left_list = left
        right_list = cast(list[Any], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ExternalResultBridgeError("canonical JSON root must be a plain object")
    _assert_plain_unaliased_json(value)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ExternalResultBridgeError("value is not finite canonical ASCII JSON") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ExternalResultBridgeError("canonical JSON exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_RECEIPT_BYTES:
        raise ExternalResultBridgeError("receipt input must be bounded exact bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ExternalResultBridgeError("receipt must have one canonical trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_json_int,
        )
    except ExternalResultBridgeError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ExternalResultBridgeError("receipt is not strict bounded ASCII JSON") from exc
    if type(value) is not dict:
        raise ExternalResultBridgeError("receipt root must be a plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ExternalResultBridgeError("receipt is not in exact canonical form")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ExternalResultBridgeError(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _unpack(record: struct.Struct, raw: bytes, offset: int, label: str) -> tuple[Any, ...]:
    try:
        return record.unpack_from(raw, offset)
    except struct.error as exc:
        raise ExternalResultBridgeError(f"external NPZ {label} is truncated") from exc


def _safe_member_name(raw_name: bytes) -> str:
    if not raw_name or len(raw_name) > _MAX_MEMBER_NAME_BYTES:
        raise ExternalResultBridgeError("ZIP member name length is invalid")
    try:
        name = raw_name.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExternalResultBridgeError("ZIP member names must be exact ASCII") from exc
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in name):
        raise ExternalResultBridgeError("ZIP member names must be printable ASCII")
    if (
        "/" in name
        or "\\" in name
        or ":" in name
        or name in {".", ".."}
        or name.startswith(("/", "\\"))
    ):
        raise ExternalResultBridgeError("ZIP member names must be safe flat relative names")
    return name


def _validate_external_attributes(external_attributes: int) -> None:
    dos_attributes = external_attributes & 0xFFFF
    unix_mode = (external_attributes >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if dos_attributes & 0x10 or file_type not in {0, stat.S_IFREG}:
        raise ExternalResultBridgeError("ZIP member has directory, symlink, or special attributes")


def _validate_local_record(
    raw: bytes,
    *,
    central_offset: int,
    central_name: bytes,
    flags: int,
    compression: int,
    crc32: int,
    compressed_size: int,
    expanded_size: int,
) -> tuple[int, int]:
    local = _unpack(_ZIP_LOCAL, raw, central_offset, "local record")
    (
        signature,
        _extract_version,
        local_flags,
        local_compression,
        _time,
        _date,
        local_crc32,
        local_compressed_size,
        local_expanded_size,
        name_size,
        extra_size,
    ) = local
    if signature != _ZIP_LOCAL_SIGNATURE:
        raise ExternalResultBridgeError("ZIP central entry does not name a local record")
    if (
        local_flags != flags
        or local_compression != compression
        or local_crc32 != crc32
        or name_size != len(central_name)
    ):
        raise ExternalResultBridgeError("ZIP local and central metadata disagree")
    name_start = central_offset + _ZIP_LOCAL.size
    name_end = name_start + name_size
    extra_end = name_end + extra_size
    if extra_end > len(raw) or raw[name_start:name_end] != central_name:
        raise ExternalResultBridgeError("ZIP local member name or extra field is truncated")
    extra = raw[name_end:extra_end]
    if local_compressed_size == _ZIP_SIZE_SENTINEL or local_expanded_size == _ZIP_SIZE_SENTINEL:
        if (
            local_compressed_size != _ZIP_SIZE_SENTINEL
            or local_expanded_size != _ZIP_SIZE_SENTINEL
            or len(extra) != _ZIP64_EXTRA.size
        ):
            raise ExternalResultBridgeError("ZIP64 local size representation is ambiguous")
        extra_id, extra_data_size, zip64_expanded, zip64_compressed = cast(
            tuple[int, int, int, int], _ZIP64_EXTRA.unpack(extra)
        )
        if (
            extra_id != _ZIP64_EXTRA_ID
            or extra_data_size != _ZIP64_EXTRA_DATA_SIZE
            or zip64_expanded != expanded_size
            or zip64_compressed != compressed_size
        ):
            raise ExternalResultBridgeError("ZIP64 local sizes disagree with inventory")
    elif local_compressed_size != compressed_size or local_expanded_size != expanded_size or extra:
        raise ExternalResultBridgeError("ZIP local sizes or extra data are noncanonical")
    data_end = extra_end + compressed_size
    if data_end > len(raw):
        raise ExternalResultBridgeError("ZIP member data is truncated")
    return extra_end, data_end


def _inventory_zip(raw: bytes) -> tuple[_ZipMember, ...]:
    if type(raw) is not bytes:
        raise ExternalResultBridgeError("external NPZ must be one immutable exact byte string")
    if len(raw) < _ZIP_END.size or len(raw) > MAX_EXTERNAL_NPZ_BYTES:
        raise ExternalResultBridgeError("external NPZ byte length is outside its bound")
    end_offset = len(raw) - _ZIP_END.size
    end = _unpack(_ZIP_END, raw, end_offset, "end record")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_size,
    ) = end
    if (
        signature != _ZIP_END_SIGNATURE
        or disk_number != 0
        or central_disk != 0
        or entries_on_disk != entry_count
        or comment_size != 0
        or entry_count < 1
        or entry_count > MAX_ZIP_MEMBER_COUNT
        or central_size > _MAX_CENTRAL_DIRECTORY_BYTES
        or central_offset + central_size != end_offset
    ):
        raise ExternalResultBridgeError(
            "external NPZ end record or central-directory bound is invalid"
        )

    members: list[_ZipMember] = []
    exact_names: set[str] = set()
    casefold_names: set[str] = set()
    total_compressed = 0
    total_expanded = 0
    cursor = central_offset
    for _ in range(entry_count):
        central = _unpack(_ZIP_CENTRAL, raw, cursor, "central record")
        (
            central_signature,
            _create_version,
            _extract_version,
            flags,
            compression,
            _time,
            _date,
            crc32,
            compressed_size,
            expanded_size,
            name_size,
            extra_size,
            member_comment_size,
            member_disk,
            _internal_attributes,
            external_attributes,
            local_offset,
        ) = central
        if central_signature != _ZIP_CENTRAL_SIGNATURE:
            raise ExternalResultBridgeError("external NPZ central inventory is malformed")
        if (
            flags & _ZIP_ENCRYPTED_FLAG
            or flags & _ZIP_DATA_DESCRIPTOR_FLAG
            or flags & ~_ZIP_ALLOWED_FLAGS
        ):
            raise ExternalResultBridgeError("ZIP encryption or unsupported flags are forbidden")
        if compression not in _ZIP_ALLOWED_METHODS:
            raise ExternalResultBridgeError("ZIP compression method is unsupported")
        if (
            compressed_size == _ZIP_SIZE_SENTINEL
            or expanded_size == _ZIP_SIZE_SENTINEL
            or name_size < 1
            or name_size > _MAX_MEMBER_NAME_BYTES
            or extra_size != 0
            or member_comment_size != 0
            or member_disk != 0
            or local_offset >= central_offset
        ):
            raise ExternalResultBridgeError("ZIP central member metadata is ambiguous")
        if compression == _ZIP_STORED and compressed_size != expanded_size:
            raise ExternalResultBridgeError("stored ZIP member sizes disagree")
        name_start = cursor + _ZIP_CENTRAL.size
        name_end = name_start + name_size
        if name_end > end_offset:
            raise ExternalResultBridgeError("ZIP central member name is truncated")
        raw_name = raw[name_start:name_end]
        name = _safe_member_name(raw_name)
        folded = name.casefold()
        if name in exact_names or folded in casefold_names:
            raise ExternalResultBridgeError(
                "ZIP member names contain duplicates or casefold aliases"
            )
        exact_names.add(name)
        casefold_names.add(folded)
        _validate_external_attributes(external_attributes)

        total_compressed += compressed_size
        total_expanded += expanded_size
        if (
            compressed_size > MAX_ZIP_TOTAL_COMPRESSED_BYTES
            or expanded_size > MAX_ZIP_TOTAL_EXPANDED_BYTES
            or total_compressed > MAX_ZIP_TOTAL_COMPRESSED_BYTES
            or total_expanded > MAX_ZIP_TOTAL_EXPANDED_BYTES
        ):
            raise ExternalResultBridgeError(
                "ZIP declared compressed or expanded size exceeds its bound"
            )
        data_offset, data_end = _validate_local_record(
            raw,
            central_offset=local_offset,
            central_name=raw_name,
            flags=flags,
            compression=compression,
            crc32=crc32,
            compressed_size=compressed_size,
            expanded_size=expanded_size,
        )
        members.append(
            _ZipMember(
                name=name,
                compression=compression,
                flags=flags,
                crc32=crc32,
                compressed_size=compressed_size,
                expanded_size=expanded_size,
                local_offset=local_offset,
                data_offset=data_offset,
                data_end=data_end,
            )
        )
        cursor = name_end
    if cursor != end_offset or cursor - central_offset != central_size:
        raise ExternalResultBridgeError("ZIP central-directory size or inventory disagrees")

    ordered_spans = sorted((member.local_offset, member.data_end) for member in members)
    expected_start = 0
    for span_start, span_end in ordered_spans:
        if span_start != expected_start or span_end <= span_start or span_end > central_offset:
            raise ExternalResultBridgeError("ZIP local records overlap or leave ambiguous bytes")
        expected_start = span_end
    if expected_start != central_offset:
        raise ExternalResultBridgeError(
            "ZIP local inventory does not exactly reach the central directory"
        )

    rewards = [member for member in members if member.name == _REWARD_MEMBER_NAME]
    if len(rewards) != 1:
        raise ExternalResultBridgeError("external NPZ must contain exactly one rewards.npy member")
    return tuple(members)


def _read_reward_member(raw: bytes, members: tuple[_ZipMember, ...]) -> bytes:
    reward = next(member for member in members if member.name == _REWARD_MEMBER_NAME)
    compressed = raw[reward.data_offset : reward.data_end]
    if len(compressed) != reward.compressed_size:
        raise ExternalResultBridgeError("reward member compressed length disagrees")
    if reward.compression == _ZIP_STORED:
        expanded = compressed
    else:
        try:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            expanded = decompressor.decompress(compressed, reward.expanded_size + 1)
            if len(expanded) > reward.expanded_size or decompressor.unconsumed_tail:
                raise ExternalResultBridgeError("reward member expands beyond its declared bound")
            remainder = decompressor.flush()
        except zlib.error as exc:
            raise ExternalResultBridgeError("reward member deflate stream is invalid") from exc
        if len(expanded) + len(remainder) > reward.expanded_size:
            raise ExternalResultBridgeError("reward member expands beyond its declared bound")
        expanded += remainder
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            raise ExternalResultBridgeError("reward member deflate framing is ambiguous")
    if len(expanded) != reward.expanded_size:
        raise ExternalResultBridgeError("reward member expanded length disagrees")
    if zlib.crc32(expanded) & 0xFFFFFFFF != reward.crc32:
        raise ExternalResultBridgeError("reward member CRC is invalid")
    return expanded


def _parse_npy_header(member: bytes, expected_dtype: str) -> tuple[int, str]:
    if len(member) < 10 or member[:6] != b"\x93NUMPY":
        raise ExternalResultBridgeError("reward member lacks an exact NPY preamble")
    version = (member[6], member[7])
    if version == (1, 0):
        length_size = 2
        header_size = struct.unpack_from("<H", member, 8)[0]
    elif version == (2, 0):
        length_size = 4
        if len(member) < 12:
            raise ExternalResultBridgeError("reward NPY v2 preamble is truncated")
        header_size = struct.unpack_from("<I", member, 8)[0]
    else:
        raise ExternalResultBridgeError("reward NPY version is unsupported")
    prefix_size = 8 + length_size
    header_end = prefix_size + header_size
    if (
        header_size < 1
        or header_size > MAX_NPY_HEADER_BYTES
        or header_end > len(member)
        or header_end % 64 != 0
    ):
        raise ExternalResultBridgeError("reward NPY header length or alignment is invalid")
    header = member[prefix_size:header_end]
    if not header.endswith(b"\n") or any(
        byte != 0x0A and (byte < 0x20 or byte > 0x7E) for byte in header
    ):
        raise ExternalResultBridgeError("reward NPY header must be bounded printable ASCII")
    try:
        expression = ast.parse(header.decode("ascii"), mode="eval")
        if not isinstance(expression.body, ast.Dict):
            raise ExternalResultBridgeError("reward NPY header must be one exact dictionary")
        header_keys = [
            key.value
            for key in expression.body.keys
            if isinstance(key, ast.Constant) and type(key.value) is str
        ]
        if len(header_keys) != 3 or len(set(header_keys)) != 3:
            raise ExternalResultBridgeError("reward NPY header keys must be exact and unique")
        decoded = ast.literal_eval(expression)
    except ExternalResultBridgeError:
        raise
    except (MemoryError, RecursionError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise ExternalResultBridgeError("reward NPY header literal is invalid") from exc
    if type(decoded) is not dict or set(decoded) != {"descr", "fortran_order", "shape"}:
        raise ExternalResultBridgeError("reward NPY header fields are not exact")
    header_map = cast(dict[str, Any], decoded)
    descriptor = header_map["descr"]
    shape = header_map["shape"]
    if type(descriptor) is not str or descriptor != expected_dtype:
        raise ExternalResultBridgeError(
            "reward NPY dtype is wrong-endian, native, object, structured, subarray, or mismatched"
        )
    if header_map["fortran_order"] is not False:
        raise ExternalResultBridgeError("reward NPY must be C-order, not Fortran-order")
    if (
        type(shape) is not tuple
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] != MATCHED_V3_REWARD_HORIZON
    ):
        raise ExternalResultBridgeError("reward NPY shape must be exactly (499712,)")
    return header_end, descriptor


def _trace_from_npy(member: bytes, expected_dtype: str) -> bytes:
    header_end, descriptor = _parse_npy_header(member, expected_dtype)
    item_size = 2 if descriptor == "<f2" else 4
    expected_data_size = MATCHED_V3_REWARD_HORIZON * item_size
    if len(member) - header_end != expected_data_size:
        raise ExternalResultBridgeError("reward NPY data length is truncated or has trailing bytes")
    data = member[header_end:]
    unpack_format = "<e" if descriptor == "<f2" else "<f"
    trace = bytearray(MATCHED_V3_REWARD_HORIZON)
    try:
        values = struct.iter_unpack(unpack_format, data)
        for index, (value,) in enumerate(values):
            if not math.isfinite(value):
                raise ExternalResultBridgeError(f"reward value at index {index} is NaN or infinite")
            integer = int(value)
            if value != integer:
                raise ExternalResultBridgeError(
                    f"reward value at index {index} is not exactly integral"
                )
            if integer not in _REWARD_SUPPORT:
                raise ExternalResultBridgeError(
                    f"reward value at index {index} is outside the exact support"
                )
            trace[index] = integer & 0xFF
    except struct.error as exc:
        raise ExternalResultBridgeError("reward NPY data cannot be decoded exactly") from exc
    return bytes(trace)


def _independent_trace_score_and_sha256(trace: bytes) -> tuple[int, str]:
    if type(trace) is not bytes or len(trace) != MATCHED_V3_REWARD_HORIZON:
        raise ExternalResultBridgeError("independent trace replay requires one complete byte trace")
    score = 0
    for index, encoded in enumerate(trace):
        if encoded == 255:
            score -= 1
        elif encoded == 0:
            continue
        elif encoded == 1:
            score += 1
        elif encoded == 30:
            score += 30
        else:
            raise ExternalResultBridgeError(
                f"independent trace replay found invalid reward at index {index}"
            )
    if not -MATCHED_V3_REWARD_HORIZON <= score <= 30 * MATCHED_V3_REWARD_HORIZON:
        raise ExternalResultBridgeError("independent cumulative score is outside exact bounds")
    preimage = b"".join(
        (
            len(_PINNED_RAW_TRACE_DIGEST_DOMAIN).to_bytes(4, "big"),
            _PINNED_RAW_TRACE_DIGEST_DOMAIN,
            len(trace).to_bytes(8, "big"),
            trace,
        )
    )
    return score, hashlib.sha256(preimage).hexdigest()


def _independent_canonical_reward_npz(trace: bytes) -> bytes:
    _independent_trace_score_and_sha256(trace)
    member = _CANONICAL_NPY_HEADER + trace
    if len(member) != _CANONICAL_NPY_MEMBER_SIZE:
        raise ExternalResultBridgeError("independent canonical NPY member geometry drifted")
    crc32 = zlib.crc32(member) & 0xFFFFFFFF
    local = _ZIP_LOCAL.pack(
        _ZIP_LOCAL_SIGNATURE,
        45,
        0,
        _ZIP_STORED,
        0,
        _CANONICAL_ZIP_DOS_DATE,
        crc32,
        _ZIP_SIZE_SENTINEL,
        _ZIP_SIZE_SENTINEL,
        len(_CANONICAL_MEMBER_NAME_BYTES),
        len(_CANONICAL_ZIP64_LOCAL_EXTRA),
    )
    central = _ZIP_CENTRAL.pack(
        _ZIP_CENTRAL_SIGNATURE,
        (3 << 8) | 45,
        45,
        0,
        _ZIP_STORED,
        0,
        _CANONICAL_ZIP_DOS_DATE,
        crc32,
        _CANONICAL_NPY_MEMBER_SIZE,
        _CANONICAL_NPY_MEMBER_SIZE,
        len(_CANONICAL_MEMBER_NAME_BYTES),
        0,
        0,
        0,
        0,
        _CANONICAL_ZIP_EXTERNAL_ATTRIBUTES,
        0,
    )
    end = _ZIP_END.pack(
        _ZIP_END_SIGNATURE,
        0,
        0,
        1,
        1,
        _CANONICAL_ZIP_CENTRAL_SIZE,
        _CANONICAL_ZIP_CENTRAL_OFFSET,
        0,
    )
    artifact = b"".join(
        (
            local,
            _CANONICAL_MEMBER_NAME_BYTES,
            _CANONICAL_ZIP64_LOCAL_EXTRA,
            member,
            central,
            _CANONICAL_MEMBER_NAME_BYTES,
            end,
        )
    )
    if len(artifact) != CANONICAL_SCORER_NPZ_SIZE_BYTES:
        raise ExternalResultBridgeError("independent canonical scorer NPZ size drifted")
    return artifact


def _expected_scorer_receipt_payload(
    *,
    trace: bytes,
    canonical_npz: bytes,
) -> dict[str, Any]:
    score, trace_sha256 = _independent_trace_score_and_sha256(trace)
    artifact_sha256 = hashlib.sha256(canonical_npz).hexdigest()
    metric_descriptor = _strict_json_load(_PINNED_METRIC_DESCRIPTOR_BYTES)
    body: dict[str, Any] = {
        "schema_version": _PINNED_SCORE_RECEIPT_SCHEMA_VERSION,
        "metric": {
            "descriptor": metric_descriptor,
            "sha256": SCORER_METRIC_SHA256,
        },
        "score": {
            "accumulation": "ordered_exact_integer_sum",
            "cumulative_reward": score,
        },
        "raw_trace": {
            "encoding_schema_version": _PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION,
            "encoding": _PINNED_RAW_TRACE_ENCODING,
            "digest_domain": _PINNED_RAW_TRACE_DIGEST_DOMAIN.decode("ascii"),
            "digest_framing": (
                "uint32be_domain_length_then_ascii_domain_then_"
                "uint64be_trace_length_then_trace_bytes"
            ),
            "horizon": MATCHED_V3_REWARD_HORIZON,
            "raw_reward_values": list(_REWARD_SUPPORT),
            "sha256": trace_sha256,
        },
        "artifact": {
            "container_schema_version": _PINNED_NPZ_CONTAINER_SCHEMA_VERSION,
            "member_name": _PINNED_CANONICAL_MEMBER_NAME,
            "compression": "stored",
            "npy_format_version": "1.0",
            "dtype": "|i1",
            "fortran_order": False,
            "shape": [MATCHED_V3_REWARD_HORIZON],
            "sha256": artifact_sha256,
            "size_bytes": len(canonical_npz),
        },
        "authority": {
            "task_identity_authority": False,
            "configuration_identity_authority": False,
            "candidate_identity_authority": False,
            "scientific_evidence_authority": False,
            "qualification_authority": False,
            "execution_authority": False,
            "promotion_authority": False,
        },
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def _claims() -> dict[str, bool]:
    return {
        "capability_accepted": False,
        "capability_issued": False,
        "candidate_qualified": False,
        "execution_authorized": False,
        "live_execution_completed": False,
        "performance_claim_allowed": False,
        "publication_authorized": False,
        "publisher_invoked": False,
        "qualification_authority": False,
        "result_accepted": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "workload_executed": False,
    }


def _limitations() -> list[str]:
    return [
        (
            "This conversion and receipt expose raw reward material, deterministic reward "
            "commitments, and the plaintext cumulative score."
        ),
        (
            "They are permanently forbidden as inputs to a score-blind controller or "
            "score-blind publisher."
        ),
        (
            "Production use requires a fresh isolated post-qualification outcome consumer; "
            "same-process integrity checks are not an isolation boundary."
        ),
        "This receipt records only a deterministic score-bearing transformation of bytes.",
        (
            "No runner, workload, process, OCI image, caller-supplied result path, or "
            "publisher is used; only pinned dependency source paths are read internally."
        ),
        "No candidate, source closure, runtime, result, or scientific claim is qualified.",
        "The canonical scorer artifact is not evidence and is not accepted as a result here.",
        "Future acceptance requires an independently authorized validator and full provenance.",
    ]


def _candidate_contract() -> dict[str, Any]:
    return {
        "candidate_count": len(EXTERNAL_RESULT_CANDIDATE_IDS),
        "candidate_ids": list(EXTERNAL_RESULT_CANDIDATE_IDS),
        "formats": {
            candidate_id: {
                "family": EXTERNAL_RESULT_CANDIDATE_FORMATS[candidate_id][0],
                "npy_descr": EXTERNAL_RESULT_CANDIDATE_FORMATS[candidate_id][1],
            }
            for candidate_id in EXTERNAL_RESULT_CANDIDATE_IDS
        },
        "continuing": {
            "candidate_ids": list(_CONTINUING_CANDIDATE_IDS),
            "npy_descr": "<f2",
            "semantic_dtype": "little_endian_float16",
        },
        "ppo": {
            "candidate_ids": list(_PPO_CANDIDATE_IDS),
            "npy_descr": "<f4",
            "semantic_dtype": "little_endian_float32",
        },
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
        "receipt_schema_version": EXTERNAL_RESULT_BRIDGE_RECEIPT_SCHEMA_VERSION,
        "status": EXTERNAL_RESULT_BRIDGE_STATUS,
        "classification": (
            "score_reward_bearing_content_transformation_only_"
            "permanently_nonqualifying_non_authorizing"
        ),
        "candidate_contract": _candidate_contract(),
        "input_contract": {
            "input_kind": "one_immutable_exact_npz_byte_string",
            "default_input_available": False,
            "maximum_input_bytes": MAX_EXTERNAL_NPZ_BYTES,
            "maximum_member_count": MAX_ZIP_MEMBER_COUNT,
            "maximum_total_compressed_bytes": MAX_ZIP_TOTAL_COMPRESSED_BYTES,
            "maximum_total_expanded_bytes": MAX_ZIP_TOTAL_EXPANDED_BYTES,
            "allowed_compression_methods": ["stored", "raw_deflate"],
            "required_member": _REWARD_MEMBER_NAME,
            "inventory_before_expansion": True,
            "only_reward_member_expanded": True,
        },
        "npy_contract": {
            "allowed_versions": ["1.0", "2.0"],
            "maximum_header_bytes": MAX_NPY_HEADER_BYTES,
            "shape": [MATCHED_V3_REWARD_HORIZON],
            "fortran_order": False,
            "native_endian_allowed": False,
            "pickle_or_object_allowed": False,
            "structured_or_subarray_allowed": False,
            "finite_required": True,
            "exactly_integral_required": True,
            "ordered_reward_values": list(_REWARD_SUPPORT),
        },
        "output_contract": {
            "trace_encoding": _PINNED_RAW_TRACE_ENCODING,
            "trace_length": MATCHED_V3_REWARD_HORIZON,
            "canonical_scorer_npz_size_bytes": CANONICAL_SCORER_NPZ_SIZE_BYTES,
            "canonical_scorer_npz_reextraction_required": True,
            "canonical_scorer_npz_reingestion_required": True,
            "independent_score_recomputation_required": True,
            "independent_trace_digest_recomputation_required": True,
            "independent_canonical_npz_reconstruction_required": True,
            "complete_scorer_receipt_payload_replay_required": True,
        },
        "exposure_contract": {
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
        },
        "bindings": {
            "scorer": {
                "module": "alberta_framework.benchmarks._forager_matched_v3_scorer",
                "source_sha256": SCORER_SOURCE_SHA256,
                "source_path": _SCORER_SOURCE_RELATIVE_PATH,
                "score_receipt_schema_version": _PINNED_SCORE_RECEIPT_SCHEMA_VERSION,
                "source_api_globals_checked_before_and_after_use": True,
            },
            "protocol": {
                "module": "alberta_framework.benchmarks.forager_matched_v3_protocol",
                "source_sha256": SCORER_PROTOCOL_SOURCE_SHA256,
                "source_path": _PROTOCOL_SOURCE_RELATIVE_PATH,
                "source_api_globals_checked_before_and_after_use": True,
            },
            "metric": {
                "schema_version": SCORER_METRIC_SCHEMA_VERSION,
                "sha256": SCORER_METRIC_SHA256,
            },
        },
        "apis": {
            "runner_exposed": False,
            "workload_exposed": False,
            "filesystem_path_exposed": False,
            "capability_issuance_or_acceptance_exposed": False,
            "publisher_exposed": False,
            "default_input_exposed": False,
            "result_acceptance_exposed": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256,
):
    raise AssertionError("external-result bridge descriptor identity drifted")


def _verify_scorer_receipt(
    *,
    trace: bytes,
    canonical_npz: bytes,
) -> _scorer.MatchedV3ScoreReceipt:
    _require_dependency_integrity()
    if len(canonical_npz) != CANONICAL_SCORER_NPZ_SIZE_BYTES:
        raise ExternalResultBridgeError("canonical scorer NPZ size identity drifted")
    independent_npz = _independent_canonical_reward_npz(trace)
    if not hmac.compare_digest(independent_npz, canonical_npz):
        raise ExternalResultBridgeError("scorer NPZ differs from independent reconstruction")
    try:
        extracted = _scorer.extract_canonical_reward_trace(canonical_npz)
        receipt = _scorer.ingest_reward_npz_bytes(canonical_npz)
    except _SCORER_ERROR_TYPE_AT_IMPORT as exc:
        raise ExternalResultBridgeError("canonical scorer NPZ failed strict replay") from exc
    _require_dependency_integrity()
    if not hmac.compare_digest(extracted, trace):
        raise ExternalResultBridgeError("canonical scorer NPZ changed reward order")
    if type(receipt) is not _SCORER_RECEIPT_TYPE_AT_IMPORT:
        raise ExternalResultBridgeError("scorer returned a non-exact receipt type")
    payload = receipt.to_payload()
    if (
        type(payload) is not dict
        or not _exact_json_equal(
            payload,
            _expected_scorer_receipt_payload(trace=trace, canonical_npz=canonical_npz),
        )
        or not hmac.compare_digest(
            receipt.canonical_json(),
            _canonical_json(
                _expected_scorer_receipt_payload(trace=trace, canonical_npz=canonical_npz)
            ),
        )
    ):
        raise ExternalResultBridgeError("complete scorer receipt payload identity drifted")
    _require_dependency_integrity()
    return receipt


def _derive_conversion(candidate_id: str, external_npz: bytes) -> ExternalRewardConversion:
    _require_dependency_integrity()
    if type(candidate_id) is not str or candidate_id not in EXTERNAL_RESULT_CANDIDATE_FORMATS:
        raise ExternalResultBridgeError("candidate_id is not one exact upstream candidate")
    if type(external_npz) is not bytes:
        raise ExternalResultBridgeError("external NPZ must be one immutable exact byte string")
    family, expected_dtype = EXTERNAL_RESULT_CANDIDATE_FORMATS[candidate_id]
    inventory = _inventory_zip(external_npz)
    member = _read_reward_member(external_npz, inventory)
    trace = _trace_from_npy(member, expected_dtype)
    independent_npz = _independent_canonical_reward_npz(trace)
    try:
        canonical_npz = _scorer.canonical_reward_npz_bytes(trace)
    except _SCORER_ERROR_TYPE_AT_IMPORT as exc:
        raise ExternalResultBridgeError(
            "signed-int8 trace failed canonical scorer construction"
        ) from exc
    _require_dependency_integrity()
    if not hmac.compare_digest(canonical_npz, independent_npz):
        raise ExternalResultBridgeError("scorer construction differs from independent NPZ")
    score_receipt = _verify_scorer_receipt(trace=trace, canonical_npz=canonical_npz)
    independent_score, independent_trace_sha256 = _independent_trace_score_and_sha256(trace)
    independent_artifact_sha256 = hashlib.sha256(canonical_npz).hexdigest()
    expected_scorer_payload = _expected_scorer_receipt_payload(
        trace=trace,
        canonical_npz=canonical_npz,
    )
    if (
        type(score_receipt.cumulative_score) is not int
        or score_receipt.cumulative_score != independent_score
        or score_receipt.raw_trace_sha256 != independent_trace_sha256
        or score_receipt.artifact_sha256 != independent_artifact_sha256
        or score_receipt.artifact_size_bytes != CANONICAL_SCORER_NPZ_SIZE_BYTES
        or score_receipt.receipt_sha256 != expected_scorer_payload["receipt_sha256"]
    ):
        raise ExternalResultBridgeError("scorer receipt fields differ from independent replay")
    _require_dependency_integrity()
    return ExternalRewardConversion(
        candidate_id=candidate_id,
        family=family,
        external_dtype=expected_dtype,
        input_npz=external_npz,
        trace=trace,
        canonical_scorer_npz=canonical_npz,
        input_npz_sha256=hashlib.sha256(external_npz).hexdigest(),
        trace_bytes_sha256=hashlib.sha256(trace).hexdigest(),
        trace_sha256=independent_trace_sha256,
        canonical_scorer_npz_sha256=independent_artifact_sha256,
        cumulative_score=independent_score,
        scorer_receipt_sha256=cast(str, expected_scorer_payload["receipt_sha256"]),
    )


def _validate_conversion(conversion: ExternalRewardConversion) -> None:
    _require_dependency_integrity()
    if type(conversion.candidate_id) is not str or (
        conversion.candidate_id not in EXTERNAL_RESULT_CANDIDATE_FORMATS
    ):
        raise ExternalResultBridgeError("conversion candidate identity is invalid")
    family, expected_dtype = EXTERNAL_RESULT_CANDIDATE_FORMATS[conversion.candidate_id]
    if type(conversion.family) is not str or conversion.family != family:
        raise ExternalResultBridgeError("conversion family binding is invalid")
    if type(conversion.external_dtype) is not str or conversion.external_dtype != expected_dtype:
        raise ExternalResultBridgeError("conversion dtype binding is invalid")
    if type(conversion.input_npz) is not bytes or type(conversion.trace) is not bytes:
        raise ExternalResultBridgeError("conversion input and trace must be immutable exact bytes")
    if type(conversion.canonical_scorer_npz) is not bytes:
        raise ExternalResultBridgeError("conversion scorer artifact must be immutable exact bytes")
    expected_trace = _trace_from_npy(
        _read_reward_member(conversion.input_npz, _inventory_zip(conversion.input_npz)),
        expected_dtype,
    )
    if not hmac.compare_digest(expected_trace, conversion.trace):
        raise ExternalResultBridgeError("conversion trace does not match its input NPZ")
    independent_npz = _independent_canonical_reward_npz(conversion.trace)
    try:
        expected_npz = _scorer.canonical_reward_npz_bytes(conversion.trace)
    except _SCORER_ERROR_TYPE_AT_IMPORT as exc:
        raise ExternalResultBridgeError(
            "conversion trace cannot build the canonical scorer NPZ"
        ) from exc
    _require_dependency_integrity()
    if not hmac.compare_digest(expected_npz, independent_npz) or not hmac.compare_digest(
        independent_npz, conversion.canonical_scorer_npz
    ):
        raise ExternalResultBridgeError("conversion canonical scorer NPZ binding is invalid")
    _verify_scorer_receipt(
        trace=conversion.trace,
        canonical_npz=conversion.canonical_scorer_npz,
    )
    independent_score, independent_trace_sha256 = _independent_trace_score_and_sha256(
        conversion.trace
    )
    scorer_payload = _expected_scorer_receipt_payload(
        trace=conversion.trace,
        canonical_npz=conversion.canonical_scorer_npz,
    )
    exact_fields: tuple[tuple[Any, Any, str], ...] = (
        (
            conversion.input_npz_sha256,
            hashlib.sha256(conversion.input_npz).hexdigest(),
            "input NPZ digest",
        ),
        (
            conversion.trace_bytes_sha256,
            hashlib.sha256(conversion.trace).hexdigest(),
            "trace byte digest",
        ),
        (conversion.trace_sha256, independent_trace_sha256, "framed trace digest"),
        (
            conversion.canonical_scorer_npz_sha256,
            hashlib.sha256(conversion.canonical_scorer_npz).hexdigest(),
            "canonical scorer NPZ digest",
        ),
        (conversion.cumulative_score, independent_score, "cumulative score"),
        (
            conversion.scorer_receipt_sha256,
            scorer_payload["receipt_sha256"],
            "scorer receipt digest",
        ),
    )
    for supplied, expected, label in exact_fields:
        if type(supplied) is not type(expected) or supplied != expected:
            raise ExternalResultBridgeError(f"conversion {label} binding is invalid")
    _require_dependency_integrity()


def _receipt_body(conversion: ExternalRewardConversion) -> dict[str, Any]:
    _validate_conversion(conversion)
    return {
        "schema_version": EXTERNAL_RESULT_BRIDGE_RECEIPT_SCHEMA_VERSION,
        "status": EXTERNAL_RESULT_BRIDGE_STATUS,
        "classification": (
            "score_reward_bearing_content_transformation_record_"
            "permanently_nonqualifying_non_authorizing"
        ),
        "descriptor_binding": {
            "schema_version": EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256,
        },
        "candidate": {
            "candidate_id": conversion.candidate_id,
            "family": conversion.family,
            "input_npy_descr": conversion.external_dtype,
        },
        "input_npz": {
            "kind": "caller_supplied_immutable_bytes",
            "sha256": conversion.input_npz_sha256,
            "size_bytes": len(conversion.input_npz),
        },
        "trace": {
            "encoding_schema_version": _PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION,
            "encoding": _PINNED_RAW_TRACE_ENCODING,
            "length": len(conversion.trace),
            "reward_values": list(_REWARD_SUPPORT),
            "bytes_sha256": conversion.trace_bytes_sha256,
            "framed_sha256": conversion.trace_sha256,
        },
        "score": {
            "accumulation": "ordered_exact_integer_sum",
            "cumulative_reward": conversion.cumulative_score,
        },
        "canonical_scorer_npz": {
            "container_schema_version": _PINNED_NPZ_CONTAINER_SCHEMA_VERSION,
            "member_name": _PINNED_CANONICAL_MEMBER_NAME,
            "sha256": conversion.canonical_scorer_npz_sha256,
            "size_bytes": len(conversion.canonical_scorer_npz),
            "scorer_receipt_sha256": conversion.scorer_receipt_sha256,
        },
        "bindings": {
            "scorer_source_sha256": SCORER_SOURCE_SHA256,
            "scorer_protocol_source_sha256": SCORER_PROTOCOL_SOURCE_SHA256,
            "metric_schema_version": SCORER_METRIC_SCHEMA_VERSION,
            "metric_sha256": SCORER_METRIC_SHA256,
        },
        "exposure": {
            "raw_reward_bytes_exposed_by_conversion": True,
            "plaintext_cumulative_score_exposed": True,
            "reward_and_score_commitments_exposed": True,
            "score_blind_controller_input_allowed": False,
            "score_blind_publisher_input_allowed": False,
            "fresh_isolated_post_qualification_outcome_consumer_required": True,
            "permanently_nonqualifying": True,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _receipt_payload(conversion: ExternalRewardConversion) -> dict[str, Any]:
    body = _receipt_body(conversion)
    return {
        **body,
        "receipt_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def external_result_bridge_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the self-pinned content-only descriptor."""

    _require_dependency_integrity()
    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_external_result_bridge_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes, including the newline."""

    _require_dependency_integrity()
    return _DESCRIPTOR_BYTES


def external_result_bridge_descriptor_sha256() -> str:
    """Return the exact frozen descriptor digest."""

    _require_dependency_integrity()
    return EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256


def parse_external_result_bridge_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact self-pinned bridge descriptor."""

    _require_dependency_integrity()
    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256
    ):
        raise ExternalResultBridgeError("external-result bridge descriptor identity drifted")
    return value


def convert_external_reward_npz(
    *, candidate_id: str, external_npz: bytes
) -> ExternalRewardConversion:
    """Return score/reward-bearing content without granting result authority."""

    return _derive_conversion(candidate_id, external_npz)


def external_reward_conversion_receipt(conversion: ExternalRewardConversion) -> dict[str, Any]:
    """Return a detached score-bearing, permanently nonqualifying receipt."""

    if type(conversion) is not ExternalRewardConversion:
        raise ExternalResultBridgeError("receipt requires one exact conversion dataclass")
    return _strict_json_load(_canonical_json(_receipt_payload(conversion)))


def canonical_external_reward_conversion_receipt_bytes(
    conversion: ExternalRewardConversion,
) -> bytes:
    """Return strict score-bearing bytes for one content-bound conversion receipt."""

    if type(conversion) is not ExternalRewardConversion:
        raise ExternalResultBridgeError("receipt requires one exact conversion dataclass")
    return _canonical_json(_receipt_payload(conversion))


def external_reward_conversion_receipt_sha256(conversion: ExternalRewardConversion) -> str:
    """Return the full-file digest of one strict conversion receipt."""

    return hashlib.sha256(
        canonical_external_reward_conversion_receipt_bytes(conversion)
    ).hexdigest()


def parse_external_reward_conversion_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
    candidate_id: str,
    external_npz: bytes,
) -> ExternalRewardConversion:
    """Replay a receipt only from independently pinned caller-supplied content."""

    expected_file_sha256 = _require_sha256(expected_file_sha256, "expected receipt file")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_file_sha256
    ):
        raise ExternalResultBridgeError("conversion receipt full-file digest disagrees")
    supplied = _strict_json_load(raw)
    conversion = _derive_conversion(candidate_id, external_npz)
    expected = _receipt_payload(conversion)
    if not _exact_json_equal(supplied, expected) or not hmac.compare_digest(
        _canonical_json(expected), raw
    ):
        raise ExternalResultBridgeError(
            "conversion receipt differs from independently replayed content"
        )
    return conversion


_BRIDGE_FUNCTION_BASELINE: Final[Mapping[str, _FunctionIntegrity]] = MappingProxyType(
    {
        name: _capture_function_integrity(function)
        for name, function in _current_bridge_function_surface().items()
    }
)


__all__ = [
    "CANONICAL_SCORER_NPZ_SIZE_BYTES",
    "EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256",
    "EXTERNAL_RESULT_BRIDGE_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_RESULT_BRIDGE_STATUS",
    "EXTERNAL_RESULT_CANDIDATE_FORMATS",
    "EXTERNAL_RESULT_CANDIDATE_IDS",
    "ExternalResultBridgeError",
    "ExternalRewardConversion",
    "MATCHED_V3_REWARD_HORIZON",
    "MAX_EXTERNAL_NPZ_BYTES",
    "MAX_NPY_HEADER_BYTES",
    "MAX_ZIP_MEMBER_COUNT",
    "MAX_ZIP_TOTAL_COMPRESSED_BYTES",
    "MAX_ZIP_TOTAL_EXPANDED_BYTES",
    "SCORER_METRIC_SCHEMA_VERSION",
    "SCORER_METRIC_SHA256",
    "SCORER_PROTOCOL_SOURCE_SHA256",
    "SCORER_SOURCE_SHA256",
    "canonical_external_result_bridge_descriptor_bytes",
    "canonical_external_reward_conversion_receipt_bytes",
    "convert_external_reward_npz",
    "external_result_bridge_descriptor",
    "external_result_bridge_descriptor_sha256",
    "external_reward_conversion_receipt",
    "external_reward_conversion_receipt_sha256",
    "parse_external_result_bridge_descriptor",
    "parse_external_reward_conversion_receipt",
]
