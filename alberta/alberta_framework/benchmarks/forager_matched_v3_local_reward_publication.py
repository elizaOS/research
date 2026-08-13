"""Direct, score-field-not-decoding publication for one matched-v3 local capability.

An ordinary package import exposes only detached descriptor and metadata types.
The publishing interface works only after this source was direct-loaded under
its exact isolated name after the exact private atomic helper and before the
exact local reward bundle.  The public publisher accepts a live bundle
capability, expected cell/source identities, and an absolute publication
parent.  It never accepts bundle objects, callbacks, reward bytes, score bytes,
or serialized bundle content.

The bundle captures the private sink in this module at its own load boundary.
That one-way source trust avoids a source-hash cycle: the bundle pins this
publisher, while this publisher validates the live bundle source and descriptor
at call time.  Publication uses the exact atomic helper once for exactly nine
files and uses the full-file SHA-256 of ``publication.json`` as the address.
Raw helper results remain local and are discarded after validation; public
publish and reload calls return immutable digest metadata only.

This is nonauthorizing plumbing.  Same-process Python mutation and same-UID
filesystem access remain outside its security boundary; a qualification worker
must own the full capability chain in a fresh isolated process.  Publication
does not grant campaign ingestion, evidence, qualification, or promotion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_metadata.v1"
)
LOCAL_REWARD_PUBLICATION_STATUS: Final = "implemented_unexecuted_non_authorizing"
LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_reward_publication_isolated_v1"
)

PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_atomic_publication_isolated_v1"
)
PINNED_ATOMIC_PUBLICATION_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_atomic_publication.py"
)
PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256: Final = (
    "8e7ccf6333c7cd8d932a190bc69aed969be93fdad450df7d5b6f8cbb785fc587"
)
PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "b224fe9fdc438ccab0df5bfd3199e1d264feacbb99147970cc68a9c703b9e98e"
)

PINNED_LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_reward_bundle_isolated_v1"
)
PINNED_LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_bundle_descriptor.v1"
)
PINNED_LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_bundle_manifest.v1"
)
PINNED_LOCAL_REWARD_PAYLOAD_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_payload.v1"
)

PUBLICATION_MANIFEST_FILENAME: Final = "publication.json"
LOCAL_BUNDLE_MANIFEST_FILENAME: Final = "local-bundle-manifest.json"
_ROLE_PATHS: Final[tuple[tuple[str, str], ...]] = (
    ("publication_manifest", PUBLICATION_MANIFEST_FILENAME),
    ("local_bundle_manifest", LOCAL_BUNDLE_MANIFEST_FILENAME),
    ("bootstrap_receipt", "bootstrap-receipt.json"),
    ("bootstrap_child_record", "bootstrap-child-record.json"),
    ("local_runner_receipt", "local-runner-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
_PUBLICATION_BOUND_ROLE_PATHS: Final = _ROLE_PATHS[1:]
_PAYLOAD_ROLE_PATHS: Final = _ROLE_PATHS[2:]
_EXACT_FILENAMES: Final = tuple(path for _role, path in _ROLE_PATHS)

_PUBLICATION_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")
_MODULE_KEYS_AT_LOAD: Final = tuple(sys.modules)
_NONEXACT_MODULE_KEYS_AT_LOAD: Final = tuple(
    type(name).__name__ for name in _MODULE_KEYS_AT_LOAD if type(name) is not str
)
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
_ISOLATED_PUBLICATION_BOUNDARY: Final = (
    type(_MODULE_NAME_INPUT) is str
    and _MODULE_NAME_INPUT == LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
    and (
        _MODULE_PACKAGE_INPUT is None
        or (type(_MODULE_PACKAGE_INPUT) is str and _MODULE_PACKAGE_INPUT == "")
    )
    and type(_SELF_MODULE_AT_LOAD) is types.ModuleType
    and _SELF_MODULE_AT_LOAD.__dict__ is globals()
    and not _NONEXACT_MODULE_KEYS_AT_LOAD
    and not _PRELOADED_FORBIDDEN_AT_LOAD
)

_ATOMIC_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
)
_ATOMIC_PUBLISH_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "publish_exact_flat_publication", None
)
_ATOMIC_LOAD_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "load_exact_flat_publication", None
)
_ATOMIC_RECORD_TYPE_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "ExactFileRecord", None)
_ATOMIC_RESULT_TYPE_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "ContentVerifiedFlatPublication", None
)
_ATOMIC_OPEN_DIRECTORY_TYPE_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "_OpenDirectory", None
)
_ATOMIC_OPEN_PARENT_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "_open_parent", None)
_ATOMIC_CLOSE_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "_close_no_raise", None)
_ATOMIC_FUNCTION_SURFACE_AT_LOAD: Final = (
    tuple(
        sorted(
            (
                (
                    name,
                    value,
                    value.__code__,
                    value.__defaults__,
                    value.__kwdefaults__,
                )
                for name, value in vars(_ATOMIC_MODULE_AT_LOAD).items()
                if type(_ATOMIC_MODULE_AT_LOAD) is types.ModuleType
                and type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    if type(_ATOMIC_MODULE_AT_LOAD) is types.ModuleType
    else ()
)

_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAX_METADATA_BYTES: Final = 1024 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UINT31_MAX: Final = 2**31 - 1
_PATH_TYPE: Final = type(Path())


class ForagerMatchedV3LocalRewardPublicationError(RuntimeError):
    """A publisher boundary, live binding, inventory, or metadata failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalRewardPublicationError(
        f"local publication JSON contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalRewardPublicationError(
        f"local publication JSON contains forbidden float {value!r}"
    )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"local publication JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_METADATA_BYTES:
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} bytes are invalid")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
        )
    except ForagerMatchedV3LocalRewardPublicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} is not an exact object")
    return cast(dict[str, Any], value)


def _canonical_json(value: object) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local publication value is not canonical JSON"
        ) from exc
    if not 1 <= len(raw) <= _MAX_METADATA_BYTES:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local publication canonical JSON exceeds its bound"
        )
    return raw


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _require_candidate_id(value: object) -> str:
    if type(value) is not str or _CANDIDATE_RE.fullmatch(value) is None:
        raise ForagerMatchedV3LocalRewardPublicationError("candidate id is invalid")
    return value


def _require_uint31(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} is invalid")
    return value


def _require_parent(value: object) -> Path:
    if type(value) is not _PATH_TYPE:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication parent must be an exact pathlib.Path"
        )
    parent = value
    raw = str(parent)
    if (
        not parent.is_absolute()
        or parent.anchor != os.sep
        or parent == Path(parent.anchor)
        or os.path.abspath(raw) != raw
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication parent must be an exact absolute non-root path"
        )
    return parent


def _read_stable_source_sha256(module: types.ModuleType, *, label: str, suffix: str) -> str:
    raw_path = getattr(module, "__file__", None)
    if type(raw_path) is not str:
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} has no exact source path")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.anchor != os.sep
        or path == Path(path.anchor)
        or os.path.abspath(raw_path) != raw_path
        or not raw_path.endswith(suffix)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(f"{label} source path is invalid")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAX_SOURCE_BYTES
            or before_identity != opened_identity
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"{label} is not a stable bounded single-link regular source"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3LocalRewardPublicationError(
                    f"{label} ended during hashing"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalRewardPublicationError(f"{label} grew during hashing")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if before_identity != after_identity or before_identity != current_identity:
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"{label} changed during hashing"
            )
        return digest.hexdigest()
    except OSError as exc:
        raise ForagerMatchedV3LocalRewardPublicationError(
            f"{label} could not be hashed exactly"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _live_forbidden_modules() -> tuple[str, ...]:
    names = tuple(sys.modules)
    if any(type(name) is not str for name in names):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "runtime module registry contains a non-string key"
        )
    return tuple(
        sorted(
            name
            for name in names
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_PREFIXES
            )
        )
    )


_SELF_FUNCTION_SURFACE_AT_READY: tuple[
    tuple[str, types.FunctionType, types.CodeType, object, object], ...
] | None = None


def _current_self_function_surface() -> tuple[
    tuple[str, types.FunctionType, types.CodeType, object, object], ...
]:
    if type(_MODULE_NAME_INPUT) is not str:
        raise ForagerMatchedV3LocalRewardPublicationError("publisher module name is invalid")
    return tuple(
        sorted(
            (
                (name, value, value.__code__, value.__defaults__, value.__kwdefaults__)
                for name, value in globals().items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == _MODULE_NAME_INPUT
            ),
            key=lambda item: item[0],
        )
    )


def _require_self_function_surface() -> None:
    expected = _SELF_FUNCTION_SURFACE_AT_READY
    if expected is None:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publisher own-function surface is not ready"
        )
    current = _current_self_function_surface()
    if len(current) != len(expected) or any(
        a_name != e_name
        or a_function is not e_function
        or a_code is not e_code
        or a_defaults != e_defaults
        or a_kwdefaults != e_kwdefaults
        for (a_name, a_function, a_code, a_defaults, a_kwdefaults), (
            e_name,
            e_function,
            e_code,
            e_defaults,
            e_kwdefaults,
        ) in zip(current, expected, strict=True)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publisher own-function surface drifted in memory"
        )


def _require_publication_boundary(*, reject_runtime_modules: bool) -> str:
    _require_self_function_surface()
    if not _ISOLATED_PUBLICATION_BOUNDARY:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publisher was not loaded through its exact isolated boundary"
        )
    expected = _require_sha256(_PUBLICATION_SOURCE_SHA256_INPUT, "publisher injected source")
    if not hmac.compare_digest(
        _read_stable_source_sha256(
            cast(types.ModuleType, _SELF_MODULE_AT_LOAD),
            label="publisher source",
            suffix="/alberta_framework/benchmarks/forager_matched_v3_local_reward_publication.py",
        ),
        expected,
    ):
        raise ForagerMatchedV3LocalRewardPublicationError("publisher source digest drifted")
    if not hmac.compare_digest(
        hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
        LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    ) or not hmac.compare_digest(_canonical_json(_descriptor()), _DESCRIPTOR_BYTES):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publisher descriptor bytes drifted"
        )
    if reject_runtime_modules and _live_forbidden_modules():
        raise ForagerMatchedV3LocalRewardPublicationError(
            "forbidden runtime modules are live at the publisher boundary"
        )
    return expected


def _require_atomic_module() -> types.ModuleType:
    _require_publication_boundary(reject_runtime_modules=False)
    module = _ATOMIC_MODULE_AT_LOAD
    if (
        type(module) is not types.ModuleType
        or sys.modules.get(PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME) is not module
        or module.__name__ != PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
        or _ATOMIC_PUBLISH_AT_LOAD is not getattr(module, "publish_exact_flat_publication", None)
        or _ATOMIC_LOAD_AT_LOAD is not getattr(module, "load_exact_flat_publication", None)
        or _ATOMIC_RECORD_TYPE_AT_LOAD is not getattr(module, "ExactFileRecord", None)
        or _ATOMIC_RESULT_TYPE_AT_LOAD
        is not getattr(module, "ContentVerifiedFlatPublication", None)
        or _ATOMIC_OPEN_DIRECTORY_TYPE_AT_LOAD is not getattr(module, "_OpenDirectory", None)
        or _ATOMIC_OPEN_PARENT_AT_LOAD is not getattr(module, "_open_parent", None)
        or _ATOMIC_CLOSE_AT_LOAD is not getattr(module, "_close_no_raise", None)
        or getattr(module, "ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "ATOMIC_PUBLICATION_DESCRIPTOR_SHA256", None)
        != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "captured atomic publication module identity drifted"
        )
    observed_source = _read_stable_source_sha256(
        module,
        label="atomic publication source",
        suffix=f"/{PINNED_ATOMIC_PUBLICATION_SOURCE_PATH}",
    )
    if not hmac.compare_digest(observed_source, PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic publication source digest drifted"
        )
    current = tuple(
        sorted(
            (
                (name, value, value.__code__, value.__defaults__, value.__kwdefaults__)
                for name, value in vars(module).items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    if len(current) != len(_ATOMIC_FUNCTION_SURFACE_AT_LOAD) or any(
        a_name != e_name
        or a_function is not e_function
        or a_code is not e_code
        or a_defaults != e_defaults
        or a_kwdefaults != e_kwdefaults
        for (a_name, a_function, a_code, a_defaults, a_kwdefaults), (
            e_name,
            e_function,
            e_code,
            e_defaults,
            e_kwdefaults,
        ) in zip(current, _ATOMIC_FUNCTION_SURFACE_AT_LOAD, strict=True)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic publication function surface drifted"
        )
    canonical_descriptor = getattr(module, "canonical_atomic_publication_descriptor_bytes", None)
    detached_descriptor = getattr(module, "atomic_publication_descriptor", None)
    if (
        type(canonical_descriptor) is not types.FunctionType
        or type(detached_descriptor) is not types.FunctionType
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic publication descriptor functions drifted"
        )
    try:
        descriptor_raw = canonical_descriptor()
        descriptor = detached_descriptor()
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic publication descriptor replay failed"
        ) from exc
    if (
        type(descriptor_raw) is not bytes
        or type(descriptor) is not dict
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_raw).hexdigest(),
            PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        )
        or not hmac.compare_digest(_canonical_json(descriptor), descriptor_raw + b"\n")
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic publication descriptor identity drifted"
        )
    return module


def _preflight_publication_parent(*, publication_parent: Path) -> Path:
    """Open and validate the exact parent before the bundle capability is claimed.

    The atomic publisher independently reopens and revalidates the path during its
    single commit attempt.  This no-write preflight therefore narrows predictable
    failures without claiming to eliminate filesystem TOCTOU.
    """

    _require_atomic_module()
    parent = _require_parent(publication_parent)
    open_parent = _ATOMIC_OPEN_PARENT_AT_LOAD
    close = _ATOMIC_CLOSE_AT_LOAD
    opened_type = _ATOMIC_OPEN_DIRECTORY_TYPE_AT_LOAD
    if (
        type(open_parent) is not types.FunctionType
        or type(close) is not types.FunctionType
        or type(opened_type) is not type
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "captured atomic parent preflight surface is unavailable"
        )
    opened = open_parent(parent)
    try:
        if type(opened) is not opened_type:
            raise ForagerMatchedV3LocalRewardPublicationError(
                "atomic parent preflight returned a non-exact result"
            )
        exact_opened = cast(Any, opened)
        if (
            exact_opened.path != parent
            or type(exact_opened.descriptor) is not int
            or exact_opened.descriptor < 0
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                "atomic parent preflight returned a non-exact result"
            )
    finally:
        descriptor = getattr(opened, "descriptor", None)
        if type(descriptor) is int and descriptor >= 0:
            close(descriptor)
    return parent


def _require_bundle_module() -> types.ModuleType:
    module = sys.modules.get(PINNED_LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME)
    if type(module) is not types.ModuleType:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "exact isolated local reward bundle is not live"
        )
    if (
        getattr(module, "LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION", None)
        != PINNED_LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        or getattr(module, "LOCAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION", None)
        != PINNED_LOCAL_REWARD_PAYLOAD_MANIFEST_SCHEMA_VERSION
        or getattr(module, "_PUBLISHER_SINK_AT_LOAD", None)
        is not _publish_consumed_local_reward_payload
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local reward bundle runtime binding drifted"
        )
    source_input = getattr(module, "_BUNDLE_SOURCE_SHA256_INPUT", None)
    source = _require_sha256(source_input, "bundle injected source")
    observed = _read_stable_source_sha256(
        module,
        label="local reward bundle source",
        suffix="/alberta_framework/benchmarks/forager_matched_v3_local_reward_bundle.py",
    )
    if not hmac.compare_digest(source, observed):
        raise ForagerMatchedV3LocalRewardPublicationError("local reward bundle source drifted")
    require_boundary = getattr(module, "_require_bundle_boundary", None)
    require_surface = getattr(module, "_require_self_function_surface", None)
    canonical_descriptor = getattr(
        module,
        "canonical_matched_v3_local_reward_bundle_descriptor_bytes",
        None,
    )
    parse_descriptor = getattr(
        module,
        "parse_matched_v3_local_reward_bundle_descriptor",
        None,
    )
    descriptor_sha256_function = getattr(
        module,
        "matched_v3_local_reward_bundle_descriptor_sha256",
        None,
    )
    if (
        type(require_boundary) is not types.FunctionType
        or type(require_surface) is not types.FunctionType
        or type(canonical_descriptor) is not types.FunctionType
        or type(parse_descriptor) is not types.FunctionType
        or type(descriptor_sha256_function) is not types.FunctionType
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local reward bundle guards are not exact functions"
        )
    require_surface()
    if not hmac.compare_digest(require_boundary(reject_runtime_modules=False), source):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local reward bundle guard disagrees with source"
        )
    descriptor_sha256 = _require_sha256(
        getattr(module, "LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256", None),
        "bundle descriptor",
    )
    try:
        descriptor_raw = canonical_descriptor()
        parsed_descriptor = parse_descriptor(descriptor_raw)
        reported_descriptor_sha256 = descriptor_sha256_function()
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local reward bundle descriptor replay failed"
        ) from exc
    if (
        type(descriptor_raw) is not bytes
        or type(parsed_descriptor) is not dict
        or type(reported_descriptor_sha256) is not str
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_raw).hexdigest(),
            descriptor_sha256,
        )
        or not hmac.compare_digest(reported_descriptor_sha256, descriptor_sha256)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "local reward bundle descriptor identity drifted"
        )
    return module


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_REWARD_PUBLICATION_STATUS,
        "classification": (
            "score_fields_not_decoded_atomic_publication_metadata_only_non_authorizing"
        ),
        "load_order": {
            "atomic_before_publisher": True,
            "publisher_before_bundle": True,
            "bundle_pins_publisher_source_and_sink": True,
            "publisher_binds_bundle_source_and_descriptor_at_call_time": True,
            "static_mutual_source_hash_cycle": False,
        },
        "public_publish_interface": {
            "accepts_live_capability": True,
            "accepts_public_bundle_object": False,
            "accepts_serialized_bundle_bytes": False,
            "accepts_callback_or_sink": False,
            "returns_immutable_metadata_only": True,
            "explicit_opt_in_required": True,
        },
        "publication": {
            "atomic_descriptor_schema_version": (
                PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "atomic_descriptor_sha256": PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
            "atomic_source_path": PINNED_ATOMIC_PUBLICATION_SOURCE_PATH,
            "atomic_source_sha256": PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
            "exact_file_count": len(_EXACT_FILENAMES),
            "exact_filenames": list(_EXACT_FILENAMES),
            "address": "full_sha256_of_publication_json",
            "atomic_helper_call_count": 1,
            "collision_retry": False,
            "uncertain_state_retry": False,
            "raw_helper_result_publicly_exposed": False,
        },
        "publication_parent": {
            "exact_platform_path_required": True,
            "absolute_non_root_canonical_path_required": True,
            "must_exist_before_capability_claim": True,
            "symlinks_allowed": False,
            "effective_uid_ownership_required": True,
            "required_mode": "0700",
            "captured_atomic_open_and_close_primitives": True,
            "safe_preflight_before_bundle_capability_claim": True,
            "preflight_performs_writes": False,
            "atomic_commit_reopens_and_reverifies_parent": True,
            "preflight_eliminates_toctou": False,
        },
        "reload": {
            "caller_carried_address_required": True,
            "caller_carried_exact_file_records_required": True,
            "cell_and_source_tree_replayed": True,
            "returns_immutable_metadata_only": True,
        },
        "metadata_receipt": {
            "canonical_score_field_free_bytes_exported": True,
            "strict_parser_exported": True,
            "caller_carried_full_file_sha256_required": True,
            "body_digest_is_not_substituted_for_full_file_digest": True,
        },
        "metadata_visibility": {
            "plaintext_score_returned": False,
            "published_file_bytes_returned": False,
            "exact_file_digests_and_sizes_returned": True,
            "information_theoretic_score_opacity_claimed": False,
            "qualification_controller_may_branch_on_content_digests_or_sizes": False,
        },
        "process_boundary": {
            "fresh_isolated_worker_required_for_qualification": True,
            "capability_chain_must_remain_in_one_pid": True,
            "same_process_python_is_not_a_hostile_code_boundary": True,
            "same_uid_filesystem_confidentiality_claimed": False,
        },
        "claims": {
            "campaign_ingestion_authorized": False,
            "evidence_authority": False,
            "execution_authorized": False,
            "qualification_authority": False,
            "scientific_promotion_allowed": False,
        },
        "limitations": [
            "Persisted content remains score-bearing even when returned metadata is not.",
            "File sizes and digests are not information-theoretically score opaque.",
            "Same-process Python mutation is outside the security boundary.",
            "Same-UID filesystem confidentiality is not claimed.",
            (
                "The pre-claim parent preflight does not eliminate filesystem TOCTOU; "
                "the atomic commit independently reopens and revalidates the parent."
            ),
            "A fresh isolated qualification worker remains externally required.",
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
):
    raise AssertionError(
        "matched-v3 local reward publication descriptor identity drifted: "
        f"{hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest()}"
    )


@dataclass(frozen=True, slots=True)
class MatchedV3LocalRewardPublicationFile:
    """One immutable caller-carried digest record; no file content is retained."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or _SAFE_NAME_RE.fullmatch(self.name) is None
            or self.name not in _EXACT_FILENAMES
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                "publication file metadata name is invalid"
            )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ForagerMatchedV3LocalRewardPublicationError(
                "publication file metadata size is invalid"
            )
        _require_sha256(self.sha256, "publication file metadata digest")


@dataclass(frozen=True, slots=True)
class MatchedV3LocalRewardPublicationMetadata:
    """Immutable digest metadata.  It intentionally retains no published bytes."""

    schema_version: str
    operation: Literal["published", "reloaded"]
    publication_root: Path
    address: str
    candidate_id: str
    environment_seed: int
    agent_seed: int
    creation_pid: int
    publication_manifest_sha256: str
    publication_manifest_body_sha256: str
    local_manifest_sha256: str
    local_manifest_body_sha256: str
    handoff_sha256: str
    local_tree_sha256: str
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    bundle_descriptor_sha256: str
    bundle_source_sha256: str
    atomic_descriptor_sha256: str
    atomic_source_sha256: str
    file_count: int
    total_size_bytes: int
    inventory_sha256: str
    files: tuple[MatchedV3LocalRewardPublicationFile, ...]
    metadata_body_sha256: str

    def __post_init__(self) -> None:
        _validate_metadata(self)


def _records_body(records: tuple[MatchedV3LocalRewardPublicationFile, ...]) -> list[dict[str, Any]]:
    return [
        {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in records
    ]


def _metadata_body(metadata: MatchedV3LocalRewardPublicationMetadata) -> dict[str, Any]:
    return {
        "schema_version": metadata.schema_version,
        "operation": metadata.operation,
        "publication_root": str(metadata.publication_root),
        "address": metadata.address,
        "candidate_id": metadata.candidate_id,
        "environment_seed": metadata.environment_seed,
        "agent_seed": metadata.agent_seed,
        "creation_pid": metadata.creation_pid,
        "publication_manifest_sha256": metadata.publication_manifest_sha256,
        "publication_manifest_body_sha256": metadata.publication_manifest_body_sha256,
        "local_manifest_sha256": metadata.local_manifest_sha256,
        "local_manifest_body_sha256": metadata.local_manifest_body_sha256,
        "handoff_sha256": metadata.handoff_sha256,
        "local_tree_sha256": metadata.local_tree_sha256,
        "publisher_descriptor_sha256": metadata.publisher_descriptor_sha256,
        "publisher_source_sha256": metadata.publisher_source_sha256,
        "bundle_descriptor_sha256": metadata.bundle_descriptor_sha256,
        "bundle_source_sha256": metadata.bundle_source_sha256,
        "atomic_descriptor_sha256": metadata.atomic_descriptor_sha256,
        "atomic_source_sha256": metadata.atomic_source_sha256,
        "file_count": metadata.file_count,
        "total_size_bytes": metadata.total_size_bytes,
        "inventory_sha256": metadata.inventory_sha256,
        "files": _records_body(metadata.files),
    }


def _metadata_payload(metadata: MatchedV3LocalRewardPublicationMetadata) -> dict[str, Any]:
    payload = _metadata_body(metadata)
    payload["metadata_body_sha256"] = metadata.metadata_body_sha256
    return payload


def _validate_records(
    value: object,
) -> tuple[MatchedV3LocalRewardPublicationFile, ...]:
    if type(value) is not tuple:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication file records must be an exact tuple"
        )
    records = cast(tuple[object, ...], value)
    if (
        len(records) != len(_EXACT_FILENAMES)
        or any(type(item) is not MatchedV3LocalRewardPublicationFile for item in records)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication file record inventory is not exact"
        )
    exact = cast(tuple[MatchedV3LocalRewardPublicationFile, ...], records)
    if tuple(item.name for item in exact) != _EXACT_FILENAMES:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication file record order is not exact"
        )
    return exact


def _validate_metadata(value: object) -> MatchedV3LocalRewardPublicationMetadata:
    if type(value) is not MatchedV3LocalRewardPublicationMetadata:
        raise ForagerMatchedV3LocalRewardPublicationError("publication metadata type is not exact")
    exact = value
    if (
        exact.schema_version != LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION
        or exact.operation not in {"published", "reloaded"}
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication metadata identity is invalid"
        )
    if (
        type(exact.publication_root) is not _PATH_TYPE
        or not exact.publication_root.is_absolute()
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication metadata root is invalid"
        )
    _require_sha256(exact.address, "publication address")
    _require_candidate_id(exact.candidate_id)
    _require_uint31(exact.environment_seed, "environment seed")
    _require_uint31(exact.agent_seed, "agent seed")
    if type(exact.creation_pid) is not int or exact.creation_pid <= 0:
        raise ForagerMatchedV3LocalRewardPublicationError("creation PID is invalid")
    for label, digest in (
        ("publication manifest", exact.publication_manifest_sha256),
        ("publication manifest body", exact.publication_manifest_body_sha256),
        ("local manifest", exact.local_manifest_sha256),
        ("local manifest body", exact.local_manifest_body_sha256),
        ("handoff", exact.handoff_sha256),
        ("local tree", exact.local_tree_sha256),
        ("publisher descriptor", exact.publisher_descriptor_sha256),
        ("publisher source", exact.publisher_source_sha256),
        ("bundle descriptor", exact.bundle_descriptor_sha256),
        ("bundle source", exact.bundle_source_sha256),
        ("atomic descriptor", exact.atomic_descriptor_sha256),
        ("atomic source", exact.atomic_source_sha256),
        ("inventory", exact.inventory_sha256),
        ("metadata body", exact.metadata_body_sha256),
    ):
        _require_sha256(digest, label)
    records = _validate_records(exact.files)
    if (
        exact.address != exact.publication_manifest_sha256
        or exact.publisher_descriptor_sha256 != LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or exact.atomic_descriptor_sha256 != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
        or exact.atomic_source_sha256 != PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256
        or type(exact.file_count) is not int
        or exact.file_count != len(records)
        or type(exact.total_size_bytes) is not int
        or exact.total_size_bytes != sum(item.size_bytes for item in records)
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication metadata fixed bindings drifted"
        )
    inventory = hashlib.sha256(_canonical_json(_records_body(records))).hexdigest()
    if not hmac.compare_digest(inventory, exact.inventory_sha256):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication metadata inventory digest does not replay"
        )
    body = hashlib.sha256(_canonical_json(_metadata_body(exact))).hexdigest()
    if not hmac.compare_digest(body, exact.metadata_body_sha256):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication metadata body digest does not replay"
        )
    return exact


def canonical_matched_v3_local_reward_publication_metadata_bytes(
    metadata: MatchedV3LocalRewardPublicationMetadata,
) -> bytes:
    """Serialize one validated score-field-free metadata receipt canonically."""

    _require_self_function_surface()
    exact = _validate_metadata(metadata)
    return _canonical_json(_metadata_payload(exact))


def parse_matched_v3_local_reward_publication_metadata(
    raw: bytes,
    *,
    expected_full_file_sha256: str,
) -> MatchedV3LocalRewardPublicationMetadata:
    """Parse one canonical metadata receipt under a caller-carried full-file digest."""

    _require_self_function_surface()
    expected = _require_sha256(expected_full_file_sha256, "metadata full-file digest")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        expected,
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "metadata receipt disagrees with its caller-carried full-file digest"
        )
    payload = _strict_json(raw, label="local publication metadata receipt")
    expected_keys = {
        "schema_version",
        "operation",
        "publication_root",
        "address",
        "candidate_id",
        "environment_seed",
        "agent_seed",
        "creation_pid",
        "publication_manifest_sha256",
        "publication_manifest_body_sha256",
        "local_manifest_sha256",
        "local_manifest_body_sha256",
        "handoff_sha256",
        "local_tree_sha256",
        "publisher_descriptor_sha256",
        "publisher_source_sha256",
        "bundle_descriptor_sha256",
        "bundle_source_sha256",
        "atomic_descriptor_sha256",
        "atomic_source_sha256",
        "file_count",
        "total_size_bytes",
        "inventory_sha256",
        "files",
        "metadata_body_sha256",
    }
    if set(payload) != expected_keys or not hmac.compare_digest(_canonical_json(payload), raw):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "metadata receipt fields or canonical encoding are not exact"
        )
    root = payload["publication_root"]
    files = payload["files"]
    if type(root) is not str or type(files) is not list:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "metadata receipt root or file inventory type is invalid"
        )
    records: list[MatchedV3LocalRewardPublicationFile] = []
    for item in files:
        if type(item) is not dict or set(item) != {"name", "sha256", "size_bytes"}:
            raise ForagerMatchedV3LocalRewardPublicationError(
                "metadata receipt file record fields are not exact"
            )
        record = cast(dict[str, Any], item)
        records.append(
            MatchedV3LocalRewardPublicationFile(
                name=record["name"],
                size_bytes=record["size_bytes"],
                sha256=record["sha256"],
            )
        )
    metadata = MatchedV3LocalRewardPublicationMetadata(
        schema_version=payload["schema_version"],
        operation=payload["operation"],
        publication_root=Path(root),
        address=payload["address"],
        candidate_id=payload["candidate_id"],
        environment_seed=payload["environment_seed"],
        agent_seed=payload["agent_seed"],
        creation_pid=payload["creation_pid"],
        publication_manifest_sha256=payload["publication_manifest_sha256"],
        publication_manifest_body_sha256=payload["publication_manifest_body_sha256"],
        local_manifest_sha256=payload["local_manifest_sha256"],
        local_manifest_body_sha256=payload["local_manifest_body_sha256"],
        handoff_sha256=payload["handoff_sha256"],
        local_tree_sha256=payload["local_tree_sha256"],
        publisher_descriptor_sha256=payload["publisher_descriptor_sha256"],
        publisher_source_sha256=payload["publisher_source_sha256"],
        bundle_descriptor_sha256=payload["bundle_descriptor_sha256"],
        bundle_source_sha256=payload["bundle_source_sha256"],
        atomic_descriptor_sha256=payload["atomic_descriptor_sha256"],
        atomic_source_sha256=payload["atomic_source_sha256"],
        file_count=payload["file_count"],
        total_size_bytes=payload["total_size_bytes"],
        inventory_sha256=payload["inventory_sha256"],
        files=tuple(records),
        metadata_body_sha256=payload["metadata_body_sha256"],
    )
    if not hmac.compare_digest(
        canonical_matched_v3_local_reward_publication_metadata_bytes(metadata),
        raw,
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "metadata receipt does not round-trip exactly"
        )
    return metadata


def _exact_role_payloads(value: object) -> tuple[tuple[str, bytes], ...]:
    if type(value) is not tuple:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "private publication payload inventory is not an exact tuple"
        )
    items = cast(tuple[object, ...], value)
    if len(items) != len(_EXACT_FILENAMES):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "private publication payload count is not exact"
        )
    exact: list[tuple[str, bytes]] = []
    for index, item in enumerate(items):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bytes
            or item[0] != _EXACT_FILENAMES[index]
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                "private publication payload order or byte type is invalid"
            )
        exact.append((item[0], item[1]))
    return tuple(exact)


def _manifest_record_matches(
    value: object,
    *,
    role: str,
    path: str,
    record: MatchedV3LocalRewardPublicationFile,
) -> bool:
    """Compare one parsed manifest record with caller-carried verified content exactly."""

    if type(value) is not dict or set(value) != {"path", "role", "sha256", "size_bytes"}:
        return False
    exact = cast(dict[str, Any], value)
    return bool(
        type(exact["path"]) is str
        and exact["path"] == path
        and type(exact["role"]) is str
        and exact["role"] == role
        and type(exact["sha256"]) is str
        and hmac.compare_digest(exact["sha256"], record.sha256)
        and type(exact["size_bytes"]) is int
        and exact["size_bytes"] == record.size_bytes
    )


def _metadata_from_verified_bytes(
    *,
    operation: Literal["published", "reloaded"],
    root: Path,
    address: str,
    records: tuple[MatchedV3LocalRewardPublicationFile, ...],
    files: dict[str, bytes],
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
) -> MatchedV3LocalRewardPublicationMetadata:
    module = _require_bundle_module()
    publication_raw = files[PUBLICATION_MANIFEST_FILENAME]
    local_raw = files[LOCAL_BUNDLE_MANIFEST_FILENAME]
    if not hmac.compare_digest(hashlib.sha256(publication_raw).hexdigest(), address):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication address is not the full publication manifest digest"
        )
    parse_publication = getattr(module, "parse_matched_v3_local_reward_publication_manifest")
    parse_local = getattr(module, "parse_matched_v3_local_reward_bundle_manifest")
    publication = parse_publication(publication_raw, expected_full_file_sha256=address)
    local_sha256 = hashlib.sha256(local_raw).hexdigest()
    local = parse_local(local_raw, expected_full_file_sha256=local_sha256)
    cell = {
        "candidate_id": expected_candidate_id,
        "environment_seed": expected_environment_seed,
        "agent_seed": expected_agent_seed,
    }
    source = cast(dict[str, Any], local["source_binding"])
    source_tree = cast(dict[str, Any], source["local_source_tree"])
    handoff = cast(dict[str, Any], source["handoff"])
    provenance = cast(dict[str, Any], local["provenance"])
    binding = cast(dict[str, Any], publication["bundle_binding"])
    local_descriptor_binding = cast(dict[str, Any], local["descriptor_binding"])
    bundle_descriptor_sha256 = _require_sha256(
        getattr(module, "LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256", None),
        "live bundle descriptor",
    )
    bundle_source_sha256 = _require_sha256(
        getattr(module, "_BUNDLE_SOURCE_SHA256_INPUT", None),
        "live bundle source",
    )
    if (
        publication["cell"] != cell
        or local["cell"] != cell
        or publication["local_source_tree_sha256"] != expected_local_source_tree_sha256
        or source_tree["tree_sha256"] != expected_local_source_tree_sha256
        or publication["handoff_record_sha256"] != handoff["record_full_file_sha256"]
        or binding["manifest_full_file_sha256"] != local_sha256
        or binding["manifest_body_sha256"] != local["manifest_body_sha256"]
        or binding["implementation_source_sha256"] != source["bundle_source_sha256"]
        or binding["descriptor_sha256"] != bundle_descriptor_sha256
        or local_descriptor_binding["sha256"] != bundle_descriptor_sha256
        or source["bundle_source_sha256"] != bundle_source_sha256
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "publication manifests disagree with expected cell or source bindings"
        )
    records_by_name = {record.name: record for record in records}
    publication_files = cast(dict[str, Any], publication["files"])
    local_files = cast(dict[str, Any], local["files"])
    for role, path in _PUBLICATION_BOUND_ROLE_PATHS:
        if not _manifest_record_matches(
            publication_files.get(role),
            role=role,
            path=path,
            record=records_by_name[path],
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"publication manifest disagrees with verified file content: {role}"
            )
    for role, path in _PAYLOAD_ROLE_PATHS:
        if not _manifest_record_matches(
            local_files.get(role),
            role=role,
            path=path,
            record=records_by_name[path],
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"local bundle manifest disagrees with verified file content: {role}"
            )
    publisher_source = _require_publication_boundary(reject_runtime_modules=False)
    inventory_sha256 = hashlib.sha256(_canonical_json(_records_body(records))).hexdigest()
    body = {
        "schema_version": LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION,
        "operation": operation,
        "publication_root": str(root),
        "address": address,
        "candidate_id": expected_candidate_id,
        "environment_seed": expected_environment_seed,
        "agent_seed": expected_agent_seed,
        "creation_pid": cast(int, provenance["creation_pid"]),
        "publication_manifest_sha256": address,
        "publication_manifest_body_sha256": cast(
            str, publication["publication_body_sha256"]
        ),
        "local_manifest_sha256": local_sha256,
        "local_manifest_body_sha256": cast(str, local["manifest_body_sha256"]),
        "handoff_sha256": cast(str, publication["handoff_record_sha256"]),
        "local_tree_sha256": expected_local_source_tree_sha256,
        "publisher_descriptor_sha256": LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        "publisher_source_sha256": publisher_source,
        "bundle_descriptor_sha256": cast(str, binding["descriptor_sha256"]),
        "bundle_source_sha256": cast(str, binding["implementation_source_sha256"]),
        "atomic_descriptor_sha256": PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        "atomic_source_sha256": PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
        "file_count": len(records),
        "total_size_bytes": sum(item.size_bytes for item in records),
        "inventory_sha256": inventory_sha256,
        "files": _records_body(records),
    }
    metadata_digest = hashlib.sha256(_canonical_json(body)).hexdigest()
    return MatchedV3LocalRewardPublicationMetadata(
        schema_version=LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION,
        operation=operation,
        publication_root=root,
        address=address,
        candidate_id=expected_candidate_id,
        environment_seed=expected_environment_seed,
        agent_seed=expected_agent_seed,
        creation_pid=cast(int, provenance["creation_pid"]),
        publication_manifest_sha256=address,
        publication_manifest_body_sha256=cast(
            str, publication["publication_body_sha256"]
        ),
        local_manifest_sha256=local_sha256,
        local_manifest_body_sha256=cast(str, local["manifest_body_sha256"]),
        handoff_sha256=cast(str, publication["handoff_record_sha256"]),
        local_tree_sha256=expected_local_source_tree_sha256,
        publisher_descriptor_sha256=LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        publisher_source_sha256=publisher_source,
        bundle_descriptor_sha256=cast(str, binding["descriptor_sha256"]),
        bundle_source_sha256=cast(str, binding["implementation_source_sha256"]),
        atomic_descriptor_sha256=PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        atomic_source_sha256=PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
        file_count=len(records),
        total_size_bytes=sum(item.size_bytes for item in records),
        inventory_sha256=inventory_sha256,
        files=records,
        metadata_body_sha256=metadata_digest,
    )


def _atomic_records(
    records: tuple[MatchedV3LocalRewardPublicationFile, ...],
) -> tuple[object, ...]:
    record_type = _ATOMIC_RECORD_TYPE_AT_LOAD
    if type(record_type) is not type:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "captured atomic file record type is unavailable"
        )
    return tuple(
        record_type(name=item.name, size_bytes=item.size_bytes, sha256=item.sha256)
        for item in records
    )


def _validated_atomic_result(
    *,
    result: object,
    expected_root: Path,
    address: str,
    atomic_records: tuple[object, ...],
    records: tuple[MatchedV3LocalRewardPublicationFile, ...],
) -> dict[str, bytes]:
    if type(result) is not _ATOMIC_RESULT_TYPE_AT_LOAD:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic helper returned a non-exact result type"
        )
    exact = cast(Any, result)
    normalized_atomic_records = tuple(
        sorted(
            atomic_records,
            key=lambda item: os.fsencode(cast(Any, item).name),
        )
    )
    normalized_names = tuple(cast(Any, item).name for item in normalized_atomic_records)
    if (
        type(exact.root) is not _PATH_TYPE
        or exact.root != expected_root / address
        or exact.address != address
        or exact.records != normalized_atomic_records
        or type(exact.files) is not types.MappingProxyType
        or tuple(exact.files) != normalized_names
    ):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "atomic helper result identity or inventory drifted"
        )
    loaded: dict[str, bytes] = {}
    for record in records:
        raw = exact.files.get(record.name)
        if (
            type(raw) is not bytes
            or len(raw) != record.size_bytes
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record.sha256)
        ):
            raise ForagerMatchedV3LocalRewardPublicationError(
                f"atomic helper result bytes drifted for {record.name}"
            )
        loaded[record.name] = raw
    return loaded


def _publish_consumed_local_reward_payload(
    *,
    publication_parent: Path,
    role_payloads: tuple[tuple[str, bytes], ...],
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
) -> MatchedV3LocalRewardPublicationMetadata:
    """Captured private sink.  It is not a public byte-ingestion interface."""

    _require_publication_boundary(reject_runtime_modules=False)
    _require_atomic_module()
    _require_bundle_module()
    parent = _require_parent(publication_parent)
    candidate_id = _require_candidate_id(expected_candidate_id)
    environment_seed = _require_uint31(expected_environment_seed, "environment seed")
    agent_seed = _require_uint31(expected_agent_seed, "agent seed")
    source_tree = _require_sha256(expected_local_source_tree_sha256, "local source tree")
    payload_items = _exact_role_payloads(role_payloads)
    payloads = dict(payload_items)
    records = tuple(
        MatchedV3LocalRewardPublicationFile(
            name=name,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        for name, raw in payload_items
    )
    address = records[0].sha256
    atomic_records = _atomic_records(records)
    publish = _ATOMIC_PUBLISH_AT_LOAD
    if type(publish) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "captured atomic publish function is unavailable"
        )
    # Exactly one call.  Collision and uncertain exceptions intentionally pass through.
    raw_result = publish(
        parent,
        address=address,
        expected_files=atomic_records,
        payloads=payloads,
    )
    loaded = _validated_atomic_result(
        result=raw_result,
        expected_root=parent,
        address=address,
        atomic_records=atomic_records,
        records=records,
    )
    metadata = _metadata_from_verified_bytes(
        operation="published",
        root=parent / address,
        address=address,
        records=records,
        files=loaded,
        expected_candidate_id=candidate_id,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
        expected_local_source_tree_sha256=source_tree,
    )
    del loaded
    del raw_result
    return _validate_metadata(metadata)


def publish_matched_v3_local_reward_capability(
    *,
    bundle_capability: object,
    publication_parent: Path,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
    explicit_publication_opt_in: bool,
) -> MatchedV3LocalRewardPublicationMetadata:
    """Consume a live capability through the bundle-captured sink and publish once."""

    _require_publication_boundary(reject_runtime_modules=True)
    _require_atomic_module()
    bundle = _require_bundle_module()
    if type(explicit_publication_opt_in) is not bool or explicit_publication_opt_in is not True:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "direct local publication requires exact explicit opt-in"
        )
    consume = getattr(bundle, "_consume_matched_v3_local_reward_capability_to_captured_sink", None)
    if type(consume) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "bundle private captured-sink consumer is unavailable"
        )
    metadata = consume(
        bundle_capability=bundle_capability,
        publication_parent=_require_parent(publication_parent),
        expected_candidate_id=_require_candidate_id(expected_candidate_id),
        expected_environment_seed=_require_uint31(
            expected_environment_seed, "environment seed"
        ),
        expected_agent_seed=_require_uint31(expected_agent_seed, "agent seed"),
        expected_local_source_tree_sha256=_require_sha256(
            expected_local_source_tree_sha256, "local source tree"
        ),
        explicit_publication_opt_in=True,
    )
    return _validate_metadata(metadata)


def load_matched_v3_local_reward_publication(
    *,
    publication_parent: Path,
    expected_address: str,
    expected_file_records: tuple[MatchedV3LocalRewardPublicationFile, ...],
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
) -> MatchedV3LocalRewardPublicationMetadata:
    """Strictly reload one caller-addressed inventory and return digest metadata."""

    _require_publication_boundary(reject_runtime_modules=True)
    _require_atomic_module()
    _require_bundle_module()
    parent = _require_parent(publication_parent)
    address = _require_sha256(expected_address, "expected publication address")
    records = _validate_records(expected_file_records)
    if not hmac.compare_digest(records[0].sha256, address):
        raise ForagerMatchedV3LocalRewardPublicationError(
            "expected address disagrees with publication manifest record"
        )
    atomic_records = _atomic_records(records)
    load = _ATOMIC_LOAD_AT_LOAD
    if type(load) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardPublicationError(
            "captured atomic load function is unavailable"
        )
    raw_result = load(parent, address=address, expected_files=atomic_records)
    loaded = _validated_atomic_result(
        result=raw_result,
        expected_root=parent,
        address=address,
        atomic_records=atomic_records,
        records=records,
    )
    metadata = _metadata_from_verified_bytes(
        operation="reloaded",
        root=parent / address,
        address=address,
        records=records,
        files=loaded,
        expected_candidate_id=_require_candidate_id(expected_candidate_id),
        expected_environment_seed=_require_uint31(
            expected_environment_seed, "environment seed"
        ),
        expected_agent_seed=_require_uint31(expected_agent_seed, "agent seed"),
        expected_local_source_tree_sha256=_require_sha256(
            expected_local_source_tree_sha256, "local source tree"
        ),
    )
    del loaded
    del raw_result
    return _validate_metadata(metadata)


def matched_v3_local_reward_publication_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing publisher descriptor content."""

    _require_self_function_surface()
    return _strict_json(_DESCRIPTOR_BYTES, label="local publication descriptor")


def canonical_matched_v3_local_reward_publication_descriptor_bytes() -> bytes:
    """Return exact canonical publisher descriptor bytes."""

    _require_self_function_surface()
    return _DESCRIPTOR_BYTES


def matched_v3_local_reward_publication_descriptor_sha256() -> str:
    """Return the frozen publisher descriptor digest."""

    _require_self_function_surface()
    return LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256


__all__ = [
    "ForagerMatchedV3LocalRewardPublicationError",
    "LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME",
    "LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION",
    "LOCAL_REWARD_PUBLICATION_STATUS",
    "MatchedV3LocalRewardPublicationFile",
    "MatchedV3LocalRewardPublicationMetadata",
    "canonical_matched_v3_local_reward_publication_descriptor_bytes",
    "canonical_matched_v3_local_reward_publication_metadata_bytes",
    "load_matched_v3_local_reward_publication",
    "matched_v3_local_reward_publication_descriptor",
    "matched_v3_local_reward_publication_descriptor_sha256",
    "parse_matched_v3_local_reward_publication_metadata",
    "publish_matched_v3_local_reward_capability",
]


_SELF_FUNCTION_SURFACE_AT_READY = _current_self_function_surface()
