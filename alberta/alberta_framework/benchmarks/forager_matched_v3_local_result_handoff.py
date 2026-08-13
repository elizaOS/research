"""Isolated, nonauthorizing handoff for matched-v3 local result bytes.

This module is descriptor-only under a normal package import.  Capability
creation and content access require an exact isolated direct-byte load, exact
source-hash injections for this module and the already-loaded local execution
bootstrap, and a clean stdlib-only module boundary.

The handoff never executes a workload and never accepts a completion or
serialized result as authority.  It consumes one authentic bootstrap outcome
internally, replays the bootstrap's strict parser, and returns only a new
opaque, PID-bound, single-use capability.  A second explicit opt-in consumes
that capability and exposes immutable, nonauthorizing content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import sys
import threading
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_result_handoff_descriptor.v1"
)
LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_result_handoff_record.v1"
)
LOCAL_RESULT_HANDOFF_STATUS: Final = "implemented_unexecuted_non_authorizing"
LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_result_handoff_isolated_v1"
)

PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_execution_bootstrap_descriptor.v1"
)
PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256: Final = (
    "6e62e2c6f2e1d157bee74c0866c96eededda21c5d77073d2adc05cc40dc72733"
)
PINNED_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256: Final = (
    "b9e49f5a97665bf0a6438404d36cff6e47073d5c778f6a72596a53cb3cbbb6d8"
)
PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
)

_HANDOFF_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256"
)
_BOOTSTRAP_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")

_FORBIDDEN_PREFIXES: Final = (
    "alberta_framework",
    "chex",
    "foragax",
    "jax",
    "jaxlib",
    "ml_dtypes",
    "numpy",
    "scipy",
)
_MODULE_KEYS_AT_LOAD: Final = tuple(sys.modules)
_NONEXACT_MODULE_KEYS_AT_LOAD: Final = tuple(
    type(name).__name__ for name in _MODULE_KEYS_AT_LOAD if type(name) is not str
)
_PRELOADED_FORBIDDEN_AT_LOAD: Final = tuple(
    sorted(
        name
        for name in _MODULE_KEYS_AT_LOAD
        if type(name) is str
        and any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES)
    )
)
_SELF_MODULE_AT_LOAD: Final = (
    sys.modules.get(_MODULE_NAME_INPUT) if type(_MODULE_NAME_INPUT) is str else None
)
_ISOLATED_HANDOFF_BOUNDARY: Final = (
    type(_MODULE_NAME_INPUT) is str
    and _MODULE_NAME_INPUT == LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
    and (
        _MODULE_PACKAGE_INPUT is None
        or (type(_MODULE_PACKAGE_INPUT) is str and _MODULE_PACKAGE_INPUT == "")
    )
    and type(_SELF_MODULE_AT_LOAD) is types.ModuleType
    and _SELF_MODULE_AT_LOAD.__dict__ is globals()
    and not _NONEXACT_MODULE_KEYS_AT_LOAD
    and not _PRELOADED_FORBIDDEN_AT_LOAD
)

_BOOTSTRAP_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
)
_BOOTSTRAP_CONSUMER_AT_LOAD: Final = getattr(
    _BOOTSTRAP_MODULE_AT_LOAD,
    "consume_matched_v3_local_bootstrap_outcome",
    None,
)
_BOOTSTRAP_PARSER_AT_LOAD: Final = getattr(
    _BOOTSTRAP_MODULE_AT_LOAD,
    "parse_matched_v3_local_bootstrap_receipt",
    None,
)
_BOOTSTRAP_OUTCOME_TYPE_AT_LOAD: Final = getattr(
    _BOOTSTRAP_MODULE_AT_LOAD,
    "_ParentOutcomeCapability",
    None,
)
_BOOTSTRAP_COMPLETION_TYPE_AT_LOAD: Final = getattr(
    _BOOTSTRAP_MODULE_AT_LOAD,
    "MatchedV3LocalBootstrapCompletion",
    None,
)
_BOOTSTRAP_FUNCTION_IDENTITIES_AT_LOAD: Final = tuple(
    sorted(
        (
            (name, value, value.__code__)
            for name, value in vars(_BOOTSTRAP_MODULE_AT_LOAD).items()
            if type(_BOOTSTRAP_MODULE_AT_LOAD) is types.ModuleType
            and type(name) is str
            and type(value) is types.FunctionType
            and value.__module__ == PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
        ),
        key=lambda item: item[0],
    )
) if type(_BOOTSTRAP_MODULE_AT_LOAD) is types.ModuleType else ()

_PINNED_BOOTSTRAP_CONSUMER_CODE_SHA256: Final = (
    "620f5707793498887991ff5986f12db5e6f73ded16112d130dd8fc1ba730230f"
)
_PINNED_BOOTSTRAP_PARSER_CODE_SHA256: Final = (
    "3b755a96b71456880ccb4879f5f749f89ff061dd72f60748dc2b72eee3feae2b"
)

_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_RECORD_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_STRING_BYTES: Final = 24 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAX_BOOTSTRAP_RECEIPT_BYTES: Final = 32 * 1024 * 1024
_MAX_CHILD_RECORD_BYTES: Final = 32 * 1024 * 1024
_MAX_LOCAL_RECEIPT_BYTES: Final = 32 * 1024 * 1024
_MAX_REWARD_TRACE_BYTES: Final = 32 * 1024 * 1024
_MAX_STDIO_BYTES: Final = 32 * 1024 * 1024
_UINT31_MAX: Final = 2**31 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ForagerMatchedV3LocalResultHandoffError(RuntimeError):
    """The isolated handoff, capability, source, or structural replay failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalResultHandoffError(
        f"local handoff JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalResultHandoffError(
                f"local handoff JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3LocalResultHandoffError(
                "local handoff JSON exceeds its node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalResultHandoffError(
                "local handoff JSON exceeds its depth bound"
            )
        if type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3LocalResultHandoffError(
                    "local handoff JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES or any(
                byte < 0x20 or byte > 0x7E for byte in encoded
            ):
                raise ForagerMatchedV3LocalResultHandoffError(
                    "local handoff JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalResultHandoffError(
                "local handoff JSON contains a non-plain value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalResultHandoffError(
                "local handoff JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalResultHandoffError(
                        "local handoff JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: dict[str, Any], *, maximum_bytes: int = _MAX_JSON_BYTES) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff canonical root must be a plain object"
        )
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
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff canonical artifact exceeds its byte ceiling"
        )
    return raw


def _strict_json_load(raw: bytes, *, maximum_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact must be bounded nonempty exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact must have one trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalResultHandoffError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact is not strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result, maximum_bytes=maximum_bytes), raw):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff artifact is not exactly canonical"
        )
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        right_list = cast(list[Any], right)
        return len(left) == len(right_list) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right_list, strict=True)
        )
    if type(left) is dict:
        left_map = cast(dict[str, Any], left)
        right_map = cast(dict[str, Any], right)
        return left_map.keys() == right_map.keys() and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    return False


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} keys are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3LocalResultHandoffError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_uint31(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3LocalResultHandoffError(
            f"{label} must be one exact uint31 integer"
        )
    return value


def _require_candidate_id(value: Any) -> str:
    if type(value) is not str or _CANDIDATE_RE.fullmatch(value) is None:
        raise ForagerMatchedV3LocalResultHandoffError(
            "candidate_id must be one bounded exact ASCII identifier"
        )
    return value


def _require_exact_bytes(
    value: Any,
    *,
    label: str,
    maximum_bytes: int,
    require_nonempty: bool,
) -> bytes:
    if (
        type(value) is not bytes
        or len(value) > maximum_bytes
        or (require_nonempty and not value)
    ):
        qualifier = "nonempty " if require_nonempty else ""
        raise ForagerMatchedV3LocalResultHandoffError(
            f"{label} must be bounded {qualifier}exact bytes"
        )
    return value


def _claims() -> dict[str, bool]:
    return {
        "authorization_issuer": False,
        "execution_authority_granted": False,
        "execution_authority_serialized": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "A handoff carries one unqualified local execution result only.",
        "The handoff does not execute a workload or issue execution authority.",
        "No serialized completion, record, receipt, trace, or descriptor grants capability.",
        "Runtime, dependency, interpreter, toolchain, and hardware qualification remain external.",
        "No handoff is scientific evidence or authorizes publication, promotion, or a SOTA claim.",
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_RESULT_HANDOFF_STATUS,
        "classification": "isolated_local_result_handoff_plumbing_non_authorizing",
        "pinned_bootstrap": {
            "descriptor_schema_version": (
                PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
            "isolated_module_name": PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME,
            "source_sha256": PINNED_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256,
            "authentic_outcome_consumed_internally": True,
            "strict_bootstrap_parser_replayed": True,
        },
        "module_import": {
            "scope": "exact_isolated_direct_byte_load_only_for_handoff_capabilities",
            "handoff_source_sha256_injection_required": True,
            "bootstrap_source_sha256_injection_required": True,
            "preloaded_alberta_jax_numpy_foragax_allowed": False,
            "filesystem_output": False,
            "subprocess_launch": False,
            "workload_execution": False,
            "default_workload_paths": False,
        },
        "handoff": {
            "creation_explicit_opt_in": True,
            "content_access_second_explicit_opt_in": True,
            "opaque": True,
            "weak_registry": True,
            "pid_bound": True,
            "single_use": True,
            "serializable": False,
            "copyable": False,
            "plain_bootstrap_completion_accepted": False,
            "serialized_bytes_accepted": False,
            "completion_returned_to_creation_caller": False,
        },
        "retained_content": {
            "canonical_handoff_record": True,
            "canonical_bootstrap_receipt": True,
            "canonical_bootstrap_child_record": True,
            "canonical_local_runner_receipt": True,
            "raw_reward_trace": True,
            "stdout_including_zero_length": True,
            "stderr_including_zero_length": True,
            "full_file_body_and_content_digests": True,
            "candidate_environment_agent_seed_binding": True,
            "bootstrap_and_handoff_descriptor_source_binding": True,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(),
    maximum_bytes=_MAX_DESCRIPTOR_BYTES,
)
LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256: Final = (
    "dc488f74d50ef224309e89968559df4671f4a3f954144530a9e4424e3cabba03"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 local result handoff descriptor identity drifted")


def _live_forbidden_modules() -> tuple[str, ...]:
    try:
        items = tuple(sys.modules)
    except RuntimeError as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            "runtime module registry changed during local handoff boundary observation"
        ) from exc
    if any(type(name) is not str for name in items):
        raise ForagerMatchedV3LocalResultHandoffError(
            "runtime module registry contains a non-exact-string key"
        )
    return tuple(
        sorted(
            name
            for name in items
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_PREFIXES
            )
        )
    )


def _read_exact_source_sha256(module: types.ModuleType, label: str) -> str:
    raw_path = getattr(module, "__file__", None)
    if type(raw_path) is not str:
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} has no exact source path")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.anchor != os.sep
        or path == Path(path.anchor)
        or os.path.abspath(raw_path) != raw_path
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            f"{label} source path is not exact absolute"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_SOURCE_BYTES
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
        ):
            raise ForagerMatchedV3LocalResultHandoffError(
                f"{label} source is not one stable bounded single-link regular file"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3LocalResultHandoffError(
                    f"{label} source ended while being hashed"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalResultHandoffError(
                f"{label} source grew while being hashed"
            )
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identities = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identities != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or identities != (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ):
            raise ForagerMatchedV3LocalResultHandoffError(
                f"{label} source changed while being hashed"
            )
        return digest.hexdigest()
    except OSError as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            f"{label} source could not be read exactly"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _current_handoff_source_sha256() -> str:
    module = sys.modules.get(LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME)
    if type(module) is not types.ModuleType or module is not _SELF_MODULE_AT_LOAD:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff isolated module identity is stale"
        )
    return _read_exact_source_sha256(module, "local handoff")


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


def _function_code_sha256(function: Any) -> str:
    if type(function) is not types.FunctionType:
        raise ForagerMatchedV3LocalResultHandoffError(
            "bootstrap boundary callable is not one exact Python function"
        )
    return hashlib.sha256(repr(_code_shape(function.__code__)).encode("ascii")).hexdigest()


def _require_handoff_boundary(*, require_current_source: bool) -> str:
    if not _ISOLATED_HANDOFF_BOUNDARY:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff requires the exact isolated direct-byte module boundary"
        )
    expected = _require_sha256(_HANDOFF_SOURCE_SHA256_INPUT, "local handoff direct-byte source")
    forbidden = _live_forbidden_modules()
    if forbidden:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff rejects preloaded runtime dependencies: "
            f"{', '.join(forbidden[:8])}"
        )
    if require_current_source:
        current = _current_handoff_source_sha256()
        if not hmac.compare_digest(current, expected):
            raise ForagerMatchedV3LocalResultHandoffError(
                "local result handoff direct-byte source identity is stale or forged"
            )
    return expected


def _validated_bootstrap_function(
    function: Any,
    *,
    captured: Any,
    module: types.ModuleType,
    name: str,
    code_sha256: str,
) -> types.FunctionType:
    if type(function) is not types.FunctionType:
        raise ForagerMatchedV3LocalResultHandoffError(
            f"local bootstrap function identity drifted: {name}"
        )
    exact = function
    if (
        exact is not captured
        or exact.__name__ != name
        or exact.__qualname__ != name
        or exact.__module__ != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
        or exact.__globals__ is not module.__dict__
        or exact.__defaults__ is not None
        or exact.__kwdefaults__ is not None
        or not hmac.compare_digest(_function_code_sha256(exact), code_sha256)
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            f"local bootstrap function identity drifted: {name}"
        )
    return cast(types.FunctionType, exact)


def _require_exact_bootstrap_module() -> tuple[
    types.ModuleType,
    types.FunctionType,
    types.FunctionType,
    str,
]:
    expected_source = _require_sha256(
        _BOOTSTRAP_SOURCE_SHA256_INPUT,
        "local bootstrap direct-byte source",
    )
    if not hmac.compare_digest(
        expected_source,
        PINNED_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256,
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap source injection differs from the pinned handoff dependency"
        )
    current = sys.modules.get(PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME)
    if (
        type(_BOOTSTRAP_MODULE_AT_LOAD) is not types.ModuleType
        or current is not _BOOTSTRAP_MODULE_AT_LOAD
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap module is absent, replaced, or was not present at handoff load"
        )
    module = current
    if (
        module.__name__ != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
        or module.__package__ not in {None, ""}
        or getattr(module, "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256", None)
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256
        or getattr(module, "LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME", None)
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
        or getattr(module, "_BOOTSTRAP_SOURCE_SHA256_INPUT", None) != expected_source
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap module identity or source injection drifted"
        )
    observed_source = _read_exact_source_sha256(module, "local bootstrap")
    if not hmac.compare_digest(observed_source, expected_source):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap source bytes are stale or forged"
        )
    current_functions = tuple(
        sorted(
            (
                (name, value)
                for name, value in module.__dict__.items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    if (
        len(current_functions) != len(_BOOTSTRAP_FUNCTION_IDENTITIES_AT_LOAD)
        or any(
            current_name != expected_name
            or current_function is not expected_function
            or current_function.__code__ is not expected_code
            for (
                current_name,
                current_function,
            ), (
                expected_name,
                expected_function,
                expected_code,
            ) in zip(
                current_functions,
                _BOOTSTRAP_FUNCTION_IDENTITIES_AT_LOAD,
                strict=True,
            )
        )
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap in-memory function surface drifted"
        )
    consumer = _validated_bootstrap_function(
        getattr(module, "consume_matched_v3_local_bootstrap_outcome", None),
        captured=_BOOTSTRAP_CONSUMER_AT_LOAD,
        module=module,
        name="consume_matched_v3_local_bootstrap_outcome",
        code_sha256=_PINNED_BOOTSTRAP_CONSUMER_CODE_SHA256,
    )
    parser = _validated_bootstrap_function(
        getattr(module, "parse_matched_v3_local_bootstrap_receipt", None),
        captured=_BOOTSTRAP_PARSER_AT_LOAD,
        module=module,
        name="parse_matched_v3_local_bootstrap_receipt",
        code_sha256=_PINNED_BOOTSTRAP_PARSER_CODE_SHA256,
    )
    return (
        module,
        consumer,
        parser,
        expected_source,
    )


def _json_body_sha256(
    raw: bytes,
    *,
    body_key: str,
    label: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], str]:
    value = _strict_json_load(raw, maximum_bytes=maximum_bytes)
    if body_key not in value:
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} body digest is absent")
    body = dict(value)
    supplied = _require_sha256(body.pop(body_key), f"{label} body")
    calculated = hashlib.sha256(_canonical_json(body, maximum_bytes=maximum_bytes)).hexdigest()
    if not hmac.compare_digest(supplied, calculated):
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} body digest drifted")
    return value, supplied


def _artifact_file_record(raw: bytes, body_sha256: str) -> dict[str, Any]:
    return {
        "size_bytes": len(raw),
        "full_file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": body_sha256,
    }


def _artifact_content_record(raw: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(raw),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _build_handoff_record(
    *,
    handoff_source_sha256: str,
    bootstrap_source_sha256: str,
    creation_pid: int,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    bootstrap_receipt: bytes,
    bootstrap_receipt_body_sha256: str,
    child_record: bytes,
    child_record_body_sha256: str,
    local_receipt: bytes,
    local_receipt_body_sha256: str,
    reward_trace: bytes,
    stdout: bytes,
    stderr: bytes,
) -> bytes:
    body: dict[str, Any] = {
        "schema_version": LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION,
        "status": "created_unconsumed_non_authorizing",
        "classification": "opaque_pid_bound_local_result_content_handoff",
        "descriptor_binding": {
            "schema_version": LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
        },
        "source_binding": {
            "handoff_source_sha256": handoff_source_sha256,
            "bootstrap": {
                "descriptor_schema_version": (
                    PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION
                ),
                "descriptor_sha256": PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
                "source_sha256": bootstrap_source_sha256,
                "isolated_module_name": PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME,
            },
        },
        "cell": {
            "candidate_id": candidate_id,
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
        },
        "provenance": {
            "creation_pid": creation_pid,
            "authentic_bootstrap_outcome_consumed": True,
            "bootstrap_completion_returned_to_creation_caller": False,
            "handoff_pid_bound": True,
            "handoff_single_use": True,
            "content_access_requires_second_explicit_opt_in": True,
        },
        "artifacts": {
            "bootstrap_receipt": _artifact_file_record(
                bootstrap_receipt,
                bootstrap_receipt_body_sha256,
            ),
            "bootstrap_child_record": _artifact_file_record(
                child_record,
                child_record_body_sha256,
            ),
            "local_runner_receipt": _artifact_file_record(
                local_receipt,
                local_receipt_body_sha256,
            ),
            "raw_reward_trace": _artifact_content_record(reward_trace),
            "stdout": _artifact_content_record(stdout),
            "stderr": _artifact_content_record(stderr),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return _canonical_json(
        {
            **body,
            "handoff_record_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
        },
        maximum_bytes=_MAX_RECORD_BYTES,
    )


def _validate_artifact_file_record(
    value: Any,
    *,
    raw: bytes,
    body_sha256: str,
    label: str,
) -> None:
    exact = _require_exact_keys(
        value,
        frozenset({"size_bytes", "full_file_sha256", "body_sha256"}),
        label,
    )
    expected = _artifact_file_record(raw, body_sha256)
    if not _exact_json_equal(exact, expected):
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} digest linkage drifted")


def _validate_artifact_content_record(value: Any, *, raw: bytes, label: str) -> None:
    exact = _require_exact_keys(
        value,
        frozenset({"size_bytes", "content_sha256"}),
        label,
    )
    if not _exact_json_equal(exact, _artifact_content_record(raw)):
        raise ForagerMatchedV3LocalResultHandoffError(f"{label} digest linkage drifted")


def parse_matched_v3_local_result_handoff_record(
    raw: bytes,
    *,
    expected_record_sha256: str,
    bootstrap_receipt_bytes: bytes,
    bootstrap_child_record_bytes: bytes,
    local_runner_receipt_bytes: bytes,
    raw_reward_trace_bytes: bytes,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
) -> dict[str, Any]:
    """Replay detached structural content; serialized content grants no capability."""

    expected = _require_sha256(expected_record_sha256, "expected handoff record")
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff record full-file digest disagrees"
        )
    bootstrap_receipt = _require_exact_bytes(
        bootstrap_receipt_bytes,
        label="bootstrap receipt",
        maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
        require_nonempty=True,
    )
    child_record = _require_exact_bytes(
        bootstrap_child_record_bytes,
        label="bootstrap child record",
        maximum_bytes=_MAX_CHILD_RECORD_BYTES,
        require_nonempty=True,
    )
    local_receipt = _require_exact_bytes(
        local_runner_receipt_bytes,
        label="local runner receipt",
        maximum_bytes=_MAX_LOCAL_RECEIPT_BYTES,
        require_nonempty=True,
    )
    reward_trace = _require_exact_bytes(
        raw_reward_trace_bytes,
        label="raw reward trace",
        maximum_bytes=_MAX_REWARD_TRACE_BYTES,
        require_nonempty=False,
    )
    stdout = _require_exact_bytes(
        stdout_bytes,
        label="stdout",
        maximum_bytes=_MAX_STDIO_BYTES,
        require_nonempty=False,
    )
    stderr = _require_exact_bytes(
        stderr_bytes,
        label="stderr",
        maximum_bytes=_MAX_STDIO_BYTES,
        require_nonempty=False,
    )
    receipt_value, receipt_body = _json_body_sha256(
        bootstrap_receipt,
        body_key="receipt_body_sha256",
        label="bootstrap receipt",
        maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
    )
    child_value, child_body = _json_body_sha256(
        child_record,
        body_key="child_record_body_sha256",
        label="bootstrap child record",
        maximum_bytes=_MAX_CHILD_RECORD_BYTES,
    )
    local_value, local_body = _json_body_sha256(
        local_receipt,
        body_key="receipt_body_sha256",
        label="local runner receipt",
        maximum_bytes=_MAX_LOCAL_RECEIPT_BYTES,
    )
    value = _strict_json_load(raw, maximum_bytes=_MAX_RECORD_BYTES)
    record = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "source_binding",
                "cell",
                "provenance",
                "artifacts",
                "claims",
                "limitations",
                "handoff_record_body_sha256",
            }
        ),
        "local handoff record",
    )
    if (
        record["schema_version"] != LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION
        or record["status"] != "created_unconsumed_non_authorizing"
        or record["classification"] != "opaque_pid_bound_local_result_content_handoff"
        or not _exact_json_equal(
            record["descriptor_binding"],
            {
                "schema_version": LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
            },
        )
    ):
        raise ForagerMatchedV3LocalResultHandoffError("local handoff record identity drifted")
    source = _require_exact_keys(
        record["source_binding"],
        frozenset({"handoff_source_sha256", "bootstrap"}),
        "local handoff source binding",
    )
    _require_sha256(source["handoff_source_sha256"], "handoff record source")
    bootstrap_source = _require_exact_keys(
        source["bootstrap"],
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "source_sha256",
                "isolated_module_name",
            }
        ),
        "local handoff bootstrap source binding",
    )
    if (
        bootstrap_source["descriptor_schema_version"]
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION
        or bootstrap_source["descriptor_sha256"]
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256
        or bootstrap_source["isolated_module_name"]
        != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff bootstrap descriptor binding drifted"
        )
    bootstrap_source_sha256 = _require_sha256(
        bootstrap_source["source_sha256"],
        "handoff record bootstrap source",
    )
    cell = _require_exact_keys(
        record["cell"],
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "local handoff cell",
    )
    candidate_id = _require_candidate_id(cell["candidate_id"])
    environment_seed = _require_uint31(cell["environment_seed"], "handoff environment seed")
    agent_seed = _require_uint31(cell["agent_seed"], "handoff agent seed")
    provenance = _require_exact_keys(
        record["provenance"],
        frozenset(
            {
                "creation_pid",
                "authentic_bootstrap_outcome_consumed",
                "bootstrap_completion_returned_to_creation_caller",
                "handoff_pid_bound",
                "handoff_single_use",
                "content_access_requires_second_explicit_opt_in",
            }
        ),
        "local handoff provenance",
    )
    if (
        type(provenance["creation_pid"]) is not int
        or provenance["creation_pid"] <= 0
        or provenance["authentic_bootstrap_outcome_consumed"] is not True
        or provenance["bootstrap_completion_returned_to_creation_caller"] is not False
        or provenance["handoff_pid_bound"] is not True
        or provenance["handoff_single_use"] is not True
        or provenance["content_access_requires_second_explicit_opt_in"] is not True
    ):
        raise ForagerMatchedV3LocalResultHandoffError("local handoff provenance drifted")
    artifacts = _require_exact_keys(
        record["artifacts"],
        frozenset(
            {
                "bootstrap_receipt",
                "bootstrap_child_record",
                "local_runner_receipt",
                "raw_reward_trace",
                "stdout",
                "stderr",
            }
        ),
        "local handoff artifacts",
    )
    _validate_artifact_file_record(
        artifacts["bootstrap_receipt"],
        raw=bootstrap_receipt,
        body_sha256=receipt_body,
        label="local handoff bootstrap receipt",
    )
    _validate_artifact_file_record(
        artifacts["bootstrap_child_record"],
        raw=child_record,
        body_sha256=child_body,
        label="local handoff bootstrap child record",
    )
    _validate_artifact_file_record(
        artifacts["local_runner_receipt"],
        raw=local_receipt,
        body_sha256=local_body,
        label="local handoff local runner receipt",
    )
    _validate_artifact_content_record(
        artifacts["raw_reward_trace"],
        raw=reward_trace,
        label="local handoff raw reward trace",
    )
    _validate_artifact_content_record(artifacts["stdout"], raw=stdout, label="local handoff stdout")
    _validate_artifact_content_record(artifacts["stderr"], raw=stderr, label="local handoff stderr")
    receipt_cell = _require_exact_keys(
        receipt_value.get("cell"),
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "bootstrap receipt cell",
    )
    child_cell = _require_exact_keys(
        child_value.get("cell"),
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "bootstrap child record cell",
    )
    local_candidate = _require_exact_keys(
        local_value.get("candidate"),
        frozenset({"candidate_id", "implementation_kind"}),
        "local runner receipt candidate",
    )
    local_seeds = _require_exact_keys(
        local_value.get("seed_transport"),
        frozenset(
            {
                "environment_seed",
                "agent_seed",
                "environment_transport",
                "agent_transport",
                "environment_agent_seed_collision",
                "environment_agent_seed_collisions_allowed",
            }
        ),
        "local runner receipt seed transport",
    )
    exact_cell = {
        "candidate_id": candidate_id,
        "environment_seed": environment_seed,
        "agent_seed": agent_seed,
    }
    if (
        not _exact_json_equal(receipt_cell, exact_cell)
        or not _exact_json_equal(child_cell, exact_cell)
        or local_candidate["candidate_id"] != candidate_id
        or local_seeds["environment_seed"] != environment_seed
        or local_seeds["agent_seed"] != agent_seed
        or receipt_value.get("bootstrap_source_sha256") != bootstrap_source_sha256
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff candidate, seed, or bootstrap source linkage drifted"
        )
    if not _exact_json_equal(record["claims"], _claims()) or not _exact_json_equal(
        record["limitations"], _limitations()
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff claims or limitations drifted"
        )
    body = dict(record)
    supplied_body = _require_sha256(
        body.pop("handoff_record_body_sha256"),
        "local handoff record body",
    )
    if not hmac.compare_digest(
        supplied_body,
        hashlib.sha256(_canonical_json(body)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff record body identity drifted"
        )
    return record


@dataclass(frozen=True, slots=True)
class MatchedV3LocalResultHandoffContent:
    """Immutable nonauthorizing bytes exposed only by a consumed live handoff."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    creation_pid: int
    handoff_source_sha256: str
    bootstrap_source_sha256: str
    canonical_handoff_record_bytes: bytes
    handoff_record_sha256: str
    canonical_bootstrap_receipt_bytes: bytes
    bootstrap_receipt_sha256: str
    bootstrap_receipt_body_sha256: str
    canonical_bootstrap_child_record_bytes: bytes
    bootstrap_child_record_sha256: str
    bootstrap_child_record_body_sha256: str
    canonical_local_runner_receipt_bytes: bytes
    local_runner_receipt_sha256: str
    local_runner_receipt_body_sha256: str
    raw_reward_trace_bytes: bytes
    raw_reward_trace_sha256: str
    stdout_bytes: bytes
    stdout_sha256: str
    stderr_bytes: bytes
    stderr_sha256: str

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id)
        _require_uint31(self.environment_seed, "handoff content environment seed")
        _require_uint31(self.agent_seed, "handoff content agent seed")
        if type(self.creation_pid) is not int or self.creation_pid <= 0:
            raise ForagerMatchedV3LocalResultHandoffError(
                "handoff content creation PID is invalid"
            )
        for supplied, raw, label in (
            (
                self.handoff_record_sha256,
                self.canonical_handoff_record_bytes,
                "handoff record",
            ),
            (
                self.bootstrap_receipt_sha256,
                self.canonical_bootstrap_receipt_bytes,
                "bootstrap receipt",
            ),
            (
                self.bootstrap_child_record_sha256,
                self.canonical_bootstrap_child_record_bytes,
                "bootstrap child record",
            ),
            (
                self.local_runner_receipt_sha256,
                self.canonical_local_runner_receipt_bytes,
                "local runner receipt",
            ),
            (
                self.raw_reward_trace_sha256,
                self.raw_reward_trace_bytes,
                "raw reward trace",
            ),
            (self.stdout_sha256, self.stdout_bytes, "stdout"),
            (self.stderr_sha256, self.stderr_bytes, "stderr"),
        ):
            expected = _require_sha256(supplied, f"handoff content {label}")
            if type(raw) is not bytes or not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(),
                expected,
            ):
                raise ForagerMatchedV3LocalResultHandoffError(
                    f"handoff content {label} digest drifted"
                )
        for body, label in (
            (self.bootstrap_receipt_body_sha256, "bootstrap receipt body"),
            (self.bootstrap_child_record_body_sha256, "bootstrap child record body"),
            (self.local_runner_receipt_body_sha256, "local runner receipt body"),
        ):
            _require_sha256(body, f"handoff content {label}")
        record = parse_matched_v3_local_result_handoff_record(
            self.canonical_handoff_record_bytes,
            expected_record_sha256=self.handoff_record_sha256,
            bootstrap_receipt_bytes=self.canonical_bootstrap_receipt_bytes,
            bootstrap_child_record_bytes=self.canonical_bootstrap_child_record_bytes,
            local_runner_receipt_bytes=self.canonical_local_runner_receipt_bytes,
            raw_reward_trace_bytes=self.raw_reward_trace_bytes,
            stdout_bytes=self.stdout_bytes,
            stderr_bytes=self.stderr_bytes,
        )
        if (
            record["cell"]
            != {
                "candidate_id": self.candidate_id,
                "environment_seed": self.environment_seed,
                "agent_seed": self.agent_seed,
            }
            or record["provenance"]["creation_pid"] != self.creation_pid
            or record["source_binding"]["handoff_source_sha256"]
            != self.handoff_source_sha256
            or record["source_binding"]["bootstrap"]["source_sha256"]
            != self.bootstrap_source_sha256
        ):
            raise ForagerMatchedV3LocalResultHandoffError(
                "handoff immutable content identity drifted"
            )

    def record(self) -> dict[str, Any]:
        """Return detached structural content; this grants no capability."""

        return parse_matched_v3_local_result_handoff_record(
            self.canonical_handoff_record_bytes,
            expected_record_sha256=self.handoff_record_sha256,
            bootstrap_receipt_bytes=self.canonical_bootstrap_receipt_bytes,
            bootstrap_child_record_bytes=self.canonical_bootstrap_child_record_bytes,
            local_runner_receipt_bytes=self.canonical_local_runner_receipt_bytes,
            raw_reward_trace_bytes=self.raw_reward_trace_bytes,
            stdout_bytes=self.stdout_bytes,
            stderr_bytes=self.stderr_bytes,
        )


class _LocalResultHandoffCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local result handoff capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("local result handoff capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("local result handoff capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("local result handoff capabilities cannot be serialized")

@dataclass(slots=True)
class _HandoffState:
    pid: int
    status: Literal["live", "consumed"]
    bootstrap_outcome_capability: object
    bootstrap_outcome_identity: int
    handoff_source_sha256: str
    bootstrap_source_sha256: str
    content_identity: int
    content_sha256: tuple[str, ...]
    content: MatchedV3LocalResultHandoffContent


_CAPABILITY_LOCK: Final = threading.Lock()
_HANDOFF_CAPABILITIES: Final[
    weakref.WeakKeyDictionary[_LocalResultHandoffCapability, _HandoffState]
] = weakref.WeakKeyDictionary()


def _build_content_from_bootstrap_completion(
    *,
    completion: Any,
    bootstrap_receipt: dict[str, Any],
    handoff_source_sha256: str,
    bootstrap_source_sha256: str,
    creation_pid: int,
) -> MatchedV3LocalResultHandoffContent:
    candidate_id = _require_candidate_id(getattr(completion, "candidate_id", None))
    environment_seed = _require_uint31(
        getattr(completion, "environment_seed", None),
        "bootstrap completion environment seed",
    )
    agent_seed = _require_uint31(
        getattr(completion, "agent_seed", None),
        "bootstrap completion agent seed",
    )
    bootstrap_receipt_raw = _require_exact_bytes(
        getattr(completion, "canonical_receipt_bytes", None),
        label="bootstrap completion receipt",
        maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
        require_nonempty=True,
    )
    child_record_raw = _require_exact_bytes(
        getattr(completion, "canonical_child_record_bytes", None),
        label="bootstrap completion child record",
        maximum_bytes=_MAX_CHILD_RECORD_BYTES,
        require_nonempty=True,
    )
    local_receipt_raw = _require_exact_bytes(
        getattr(completion, "local_completion_receipt_bytes", None),
        label="bootstrap completion local runner receipt",
        maximum_bytes=_MAX_LOCAL_RECEIPT_BYTES,
        require_nonempty=True,
    )
    reward_trace = _require_exact_bytes(
        getattr(completion, "reward_trace", None),
        label="bootstrap completion reward trace",
        maximum_bytes=_MAX_REWARD_TRACE_BYTES,
        require_nonempty=False,
    )
    stdout = _require_exact_bytes(
        getattr(completion, "stdout", None),
        label="bootstrap completion stdout",
        maximum_bytes=_MAX_STDIO_BYTES,
        require_nonempty=False,
    )
    stderr = _require_exact_bytes(
        getattr(completion, "stderr", None),
        label="bootstrap completion stderr",
        maximum_bytes=_MAX_STDIO_BYTES,
        require_nonempty=False,
    )
    receipt_value, receipt_body = _json_body_sha256(
        bootstrap_receipt_raw,
        body_key="receipt_body_sha256",
        label="bootstrap receipt",
        maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
    )
    if not _exact_json_equal(receipt_value, bootstrap_receipt):
        raise ForagerMatchedV3LocalResultHandoffError(
            "bootstrap strict parser replay disagrees with local structural replay"
        )
    child_value, child_body = _json_body_sha256(
        child_record_raw,
        body_key="child_record_body_sha256",
        label="bootstrap child record",
        maximum_bytes=_MAX_CHILD_RECORD_BYTES,
    )
    local_value, local_body = _json_body_sha256(
        local_receipt_raw,
        body_key="receipt_body_sha256",
        label="local runner receipt",
        maximum_bytes=_MAX_LOCAL_RECEIPT_BYTES,
    )
    expected_cell = {
        "candidate_id": candidate_id,
        "environment_seed": environment_seed,
        "agent_seed": agent_seed,
    }
    if (
        not _exact_json_equal(receipt_value.get("cell"), expected_cell)
        or not _exact_json_equal(child_value.get("cell"), expected_cell)
        or receipt_value.get("bootstrap_source_sha256") != bootstrap_source_sha256
        or local_value.get("candidate", {}).get("candidate_id") != candidate_id
        or local_value.get("seed_transport", {}).get("environment_seed") != environment_seed
        or local_value.get("seed_transport", {}).get("agent_seed") != agent_seed
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "bootstrap completion candidate, seed, or source binding drifted"
        )
    record_raw = _build_handoff_record(
        handoff_source_sha256=handoff_source_sha256,
        bootstrap_source_sha256=bootstrap_source_sha256,
        creation_pid=creation_pid,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        bootstrap_receipt=bootstrap_receipt_raw,
        bootstrap_receipt_body_sha256=receipt_body,
        child_record=child_record_raw,
        child_record_body_sha256=child_body,
        local_receipt=local_receipt_raw,
        local_receipt_body_sha256=local_body,
        reward_trace=reward_trace,
        stdout=stdout,
        stderr=stderr,
    )
    return MatchedV3LocalResultHandoffContent(
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        creation_pid=creation_pid,
        handoff_source_sha256=handoff_source_sha256,
        bootstrap_source_sha256=bootstrap_source_sha256,
        canonical_handoff_record_bytes=record_raw,
        handoff_record_sha256=hashlib.sha256(record_raw).hexdigest(),
        canonical_bootstrap_receipt_bytes=bootstrap_receipt_raw,
        bootstrap_receipt_sha256=hashlib.sha256(bootstrap_receipt_raw).hexdigest(),
        bootstrap_receipt_body_sha256=receipt_body,
        canonical_bootstrap_child_record_bytes=child_record_raw,
        bootstrap_child_record_sha256=hashlib.sha256(child_record_raw).hexdigest(),
        bootstrap_child_record_body_sha256=child_body,
        canonical_local_runner_receipt_bytes=local_receipt_raw,
        local_runner_receipt_sha256=hashlib.sha256(local_receipt_raw).hexdigest(),
        local_runner_receipt_body_sha256=local_body,
        raw_reward_trace_bytes=reward_trace,
        raw_reward_trace_sha256=hashlib.sha256(reward_trace).hexdigest(),
        stdout_bytes=stdout,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_bytes=stderr,
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
    )


def issue_matched_v3_local_result_handoff(
    *,
    bootstrap_outcome_capability: object,
    explicit_handoff_opt_in: bool,
) -> object:
    """Consume one authentic bootstrap outcome and issue one opaque handoff."""

    if type(explicit_handoff_opt_in) is not bool or explicit_handoff_opt_in is not True:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff creation requires exact explicit opt-in"
        )
    handoff_source_sha256 = _require_handoff_boundary(require_current_source=True)
    module, consumer, parser, bootstrap_source_sha256 = _require_exact_bootstrap_module()
    outcome_type = getattr(module, "_ParentOutcomeCapability", None)
    completion_type = getattr(module, "MatchedV3LocalBootstrapCompletion", None)
    if (
        type(outcome_type) is not type
        or outcome_type is not _BOOTSTRAP_OUTCOME_TYPE_AT_LOAD
        or outcome_type.__module__ != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
        or type(bootstrap_outcome_capability) is not outcome_type
        or type(completion_type) is not type
        or completion_type is not _BOOTSTRAP_COMPLETION_TYPE_AT_LOAD
        or completion_type.__module__ != PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff requires an authentic opaque bootstrap outcome"
        )
    try:
        completion: Any = consumer(
            outcome_capability=bootstrap_outcome_capability,
            explicit_content_access_opt_in=True,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            "authentic local bootstrap outcome consumption failed"
        ) from exc
    if type(completion) is not completion_type:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap consumer returned non-authentic completion content"
        )
    exact_completion = cast(Any, completion)
    module_after, consumer_after, parser_after, source_after = _require_exact_bootstrap_module()
    if (
        module_after is not module
        or consumer_after is not consumer
        or parser_after is not parser
        or not hmac.compare_digest(source_after, bootstrap_source_sha256)
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap boundary changed across outcome consumption"
        )
    try:
        bootstrap_receipt = parser(
            exact_completion.canonical_receipt_bytes,
            expected_receipt_sha256=exact_completion.receipt_sha256,
            canonical_child_record_bytes=exact_completion.canonical_child_record_bytes,
            local_completion_receipt_bytes=exact_completion.local_completion_receipt_bytes,
            reward_trace=exact_completion.reward_trace,
            stdout=exact_completion.stdout,
            stderr=exact_completion.stderr,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap strict parser replay failed during handoff"
        ) from exc
    if type(bootstrap_receipt) is not dict:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local bootstrap strict parser returned a non-plain receipt"
        )
    creation_pid = os.getpid()
    content = _build_content_from_bootstrap_completion(
        completion=exact_completion,
        bootstrap_receipt=cast(dict[str, Any], bootstrap_receipt),
        handoff_source_sha256=handoff_source_sha256,
        bootstrap_source_sha256=bootstrap_source_sha256,
        creation_pid=creation_pid,
    )
    if not hmac.compare_digest(
        _require_handoff_boundary(require_current_source=True),
        handoff_source_sha256,
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local handoff source changed before capability issuance"
        )
    handoff = _LocalResultHandoffCapability()
    content_digests = (
        content.handoff_record_sha256,
        content.bootstrap_receipt_sha256,
        content.bootstrap_child_record_sha256,
        content.local_runner_receipt_sha256,
        content.raw_reward_trace_sha256,
        content.stdout_sha256,
        content.stderr_sha256,
    )
    with _CAPABILITY_LOCK:
        _HANDOFF_CAPABILITIES[handoff] = _HandoffState(
            pid=creation_pid,
            status="live",
            bootstrap_outcome_capability=bootstrap_outcome_capability,
            bootstrap_outcome_identity=id(bootstrap_outcome_capability),
            handoff_source_sha256=handoff_source_sha256,
            bootstrap_source_sha256=bootstrap_source_sha256,
            content_identity=id(content),
            content_sha256=content_digests,
            content=content,
        )
    return handoff


def consume_matched_v3_local_result_handoff(
    *,
    handoff_capability: object,
    explicit_content_access_opt_in: bool,
) -> MatchedV3LocalResultHandoffContent:
    """Consume one authentic handoff and expose immutable nonauthorizing content."""

    if (
        type(explicit_content_access_opt_in) is not bool
        or explicit_content_access_opt_in is not True
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff content access requires exact explicit opt-in"
        )
    if type(handoff_capability) is not _LocalResultHandoffCapability:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff content access requires an authentic opaque capability"
        )
    exact = handoff_capability
    with _CAPABILITY_LOCK:
        state = _HANDOFF_CAPABILITIES.get(exact)
        if state is None or state.status != "live":
            raise ForagerMatchedV3LocalResultHandoffError(
                "local result handoff capability is unknown, stale, or already consumed"
            )
        current_pid = os.getpid()
        if state.pid != current_pid:
            state.status = "consumed"
            raise ForagerMatchedV3LocalResultHandoffError(
                "local result handoff capability cannot cross a PID boundary"
            )
        if (
            id(state.bootstrap_outcome_capability) != state.bootstrap_outcome_identity
            or id(state.content) != state.content_identity
        ):
            state.status = "consumed"
            raise ForagerMatchedV3LocalResultHandoffError(
                "local result handoff provenance identity is stale"
            )
        state.status = "consumed"
    current_source = _require_handoff_boundary(require_current_source=True)
    if not hmac.compare_digest(current_source, state.handoff_source_sha256):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff source is stale"
        )
    _module, _consumer, _parser, bootstrap_source = _require_exact_bootstrap_module()
    if not hmac.compare_digest(bootstrap_source, state.bootstrap_source_sha256):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff bootstrap source is stale"
        )
    content = state.content
    if type(content) is not MatchedV3LocalResultHandoffContent:
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff immutable content type is stale"
        )
    observed = (
        hashlib.sha256(content.canonical_handoff_record_bytes).hexdigest(),
        hashlib.sha256(content.canonical_bootstrap_receipt_bytes).hexdigest(),
        hashlib.sha256(content.canonical_bootstrap_child_record_bytes).hexdigest(),
        hashlib.sha256(content.canonical_local_runner_receipt_bytes).hexdigest(),
        hashlib.sha256(content.raw_reward_trace_bytes).hexdigest(),
        hashlib.sha256(content.stdout_bytes).hexdigest(),
        hashlib.sha256(content.stderr_bytes).hexdigest(),
    )
    if len(observed) != len(state.content_sha256) or any(
        not hmac.compare_digest(actual, expected)
        for actual, expected in zip(observed, state.content_sha256, strict=True)
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff content bytes are stale"
        )
    record = content.record()
    if (
        record["provenance"]["creation_pid"] != state.pid
        or record["source_binding"]["handoff_source_sha256"]
        != state.handoff_source_sha256
        or record["source_binding"]["bootstrap"]["source_sha256"]
        != state.bootstrap_source_sha256
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff provenance record drifted"
        )
    return content


def matched_v3_local_result_handoff_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing handoff descriptor content."""

    return _strict_json_load(_DESCRIPTOR_BYTES, maximum_bytes=_MAX_DESCRIPTOR_BYTES)


def canonical_matched_v3_local_result_handoff_descriptor_bytes() -> bytes:
    """Return the exact canonical handoff descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_local_result_handoff_descriptor_sha256() -> str:
    """Return the exact handoff descriptor digest."""

    return LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256


def parse_matched_v3_local_result_handoff_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen nonauthorizing descriptor."""

    value = _strict_json_load(raw, maximum_bytes=_MAX_DESCRIPTOR_BYTES)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3LocalResultHandoffError(
            "local result handoff descriptor differs from its frozen identity"
        )
    return value


__all__ = [
    "ForagerMatchedV3LocalResultHandoffError",
    "LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256",
    "LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME",
    "LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION",
    "LOCAL_RESULT_HANDOFF_STATUS",
    "MatchedV3LocalResultHandoffContent",
    "PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION",
    "PINNED_LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME",
    "PINNED_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256",
    "canonical_matched_v3_local_result_handoff_descriptor_bytes",
    "consume_matched_v3_local_result_handoff",
    "issue_matched_v3_local_result_handoff",
    "matched_v3_local_result_handoff_descriptor",
    "matched_v3_local_result_handoff_descriptor_sha256",
    "parse_matched_v3_local_result_handoff_descriptor",
    "parse_matched_v3_local_result_handoff_record",
]
