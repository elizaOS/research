"""Additive metadata-only atomic publication for matched-v3 adapter results.

The two public fused entrypoints run the existing exact full-horizon *unqualified*
adapter surfaces, consume their live in-process results immediately, convert them with
the frozen adapter bundle, and persist the existing five-file publication layout.
Only canonical digest/size metadata is returned.  This module does not turn either
runner into an authorized production or qualification surface.

Digest-only return values are not a confidentiality boundary.  A same-UID process can
still read the published files, and their names, sizes, and hashes are content
fingerprints.  Qualification therefore remains blocked on a fresh host worker/cgroup
executor that exposes only the canonical metadata receipt across its IPC boundary.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib
import json
import marshal
import os
import re
import stat
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_atomic_publication as atomic

ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_atomic_publication_descriptor.v2"
)
ADAPTER_ATOMIC_PUBLICATION_V2_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_atomic_publication_metadata.v2"
)
ADAPTER_ATOMIC_PUBLICATION_V2_STATUS: Final = (
    "implemented_unexecuted_unqualified_surfaces_host_isolation_unproven"
)

PUBLICATION_MANIFEST_FILENAME: Final = "publication.json"
ADAPTER_BUNDLE_MANIFEST_FILENAME: Final = "adapter-bundle-manifest.json"
RUNNER_RESULT_RECEIPT_FILENAME: Final = "runner-result-receipt.json"
REWARD_TRACE_FILENAME: Final = "reward-trace.npz"
SCORE_RECEIPT_FILENAME: Final = "score-receipt.json"

ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES: Final = (
    PUBLICATION_MANIFEST_FILENAME,
    ADAPTER_BUNDLE_MANIFEST_FILENAME,
    RUNNER_RESULT_RECEIPT_FILENAME,
    REWARD_TRACE_FILENAME,
    SCORE_RECEIPT_FILENAME,
)
ADAPTER_ATOMIC_PUBLICATION_V2_CANDIDATE_IDS: Final = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)

_ATOMIC_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_atomic_publication.py"
)
_ATOMIC_SOURCE_SHA256: Final = (
    "8e7ccf6333c7cd8d932a190bc69aed969be93fdad450df7d5b6f8cbb785fc587"
)
_SCORER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
)
_SCORER_SOURCE_SHA256: Final = (
    "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
)
_BUNDLE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py"
)
_BUNDLE_SOURCE_SHA256: Final = (
    "22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469"
)
_LEGACY_PUBLICATION_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
)
_LEGACY_PUBLICATION_SOURCE_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)
_FULL_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
)
_FULL_RUNNER_SOURCE_SHA256: Final = (
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
)
_PPO_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
)
_PPO_RUNNER_SOURCE_SHA256: Final = (
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
)
_ATOMIC_DESCRIPTOR_SHA256: Final = (
    "b224fe9fdc438ccab0df5bfd3199e1d264feacbb99147970cc68a9c703b9e98e"
)
_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5"
)
_LEGACY_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
_FULL_RUNNER_DESCRIPTOR_SHA256: Final = (
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
)
_PPO_RUNNER_DESCRIPTOR_SHA256: Final = (
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
)
_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_atomic_publication_v2.py"
)

_SCORER_MODULE: Final = "alberta_framework.benchmarks._forager_matched_v3_scorer"
_BUNDLE_MODULE: Final = (
    "alberta_framework.benchmarks.forager_matched_v3_adapter_reward_bundle"
)
_LEGACY_PUBLICATION_MODULE: Final = (
    "alberta_framework.benchmarks.forager_matched_v3_adapter_reward_publication"
)
_FULL_RUNNER_MODULE: Final = (
    "alberta_framework.benchmarks.forager_matched_v3_full_rainbow_runner"
)
_PPO_RUNNER_MODULE: Final = (
    "alberta_framework.benchmarks.forager_matched_v3_ppo_gru_runner"
)

_MAX_DESCRIPTOR_BYTES: Final = 128 * 1024
_MAX_METADATA_BYTES: Final = 256 * 1024
_MAX_JSON_NODES: Final = 8_192
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_STRING_CHARACTERS: Final = 32 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_PUBLICATION_FILE_BYTES: Final = 512 * 1024 * 1024
_MAX_PUBLICATION_TOTAL_BYTES: Final = 1024 * 1024 * 1024
_MAX_INTEGER_DIGITS: Final = 19
_UINT31_MAX: Final = (1 << 31) - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_TYPE: Final = type(Path())


class ForagerMatchedV3AdapterAtomicPublicationV2Error(ValueError):
    """The additive publisher, metadata, or exact dependency closure failed."""


class ForagerMatchedV3AdapterAtomicPublicationV2CollisionError(
    ForagerMatchedV3AdapterAtomicPublicationV2Error
):
    """The requested content address already exists; publication was not retried."""


class PublishedAdapterAtomicPublicationV2UncertainError(
    ForagerMatchedV3AdapterAtomicPublicationV2Error
):
    """Publication committed, or may have committed, before metadata returned."""

    def __init__(
        self,
        destination: Path,
        address: str,
        detail: str,
        *,
        committed: Literal[True] | None,
    ) -> None:
        if committed is not True and committed is not None:
            raise TypeError("committed must be exactly True or None")
        self.destination = destination
        self.address = address
        self.committed = committed
        state = "committed" if committed is True else "commit state unknown"
        super().__init__(
            f"adapter atomic-v2 publication {address} at {destination}: "
            f"{detail}; {state}"
        )


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3AdapterAtomicPublicationV2Error(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_uint31(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        _fail(f"{label} must be one uint31 exact integer")
    return value


def _require_true(value: object, label: str) -> None:
    if type(value) is not bool or value is not True:
        _fail(f"{label} requires exact explicit True")


def _require_candidate(value: object) -> str:
    if type(value) is not str or value not in ADAPTER_ATOMIC_PUBLICATION_V2_CANDIDATE_IDS:
        _fail("adapter atomic-v2 candidate differs")
    return value


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            "adapter atomic-v2 value is not canonical ASCII JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _reject_float(value: str) -> NoReturn:
    _fail(f"adapter atomic-v2 JSON contains forbidden float {value!r}")


def _reject_constant(value: str) -> NoReturn:
    _fail(f"adapter atomic-v2 JSON contains non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_INTEGER_DIGITS:
        _fail("adapter atomic-v2 JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"adapter atomic-v2 JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_json(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("adapter atomic-v2 JSON exceeds its complexity bound")
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) is str:
            text = item
            if len(text) > _MAX_JSON_STRING_CHARACTERS or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in text
            ):
                _fail("adapter atomic-v2 JSON string is not bounded printable ASCII")
            continue
        if type(item) not in {dict, list}:
            _fail("adapter atomic-v2 JSON contains an inexact value type")
        identity = id(item)
        if identity in seen:
            _fail("adapter atomic-v2 JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                _fail("adapter atomic-v2 JSON object key is not exact text")
            stack.extend((child, depth + 1) for child in mapping.values())
        else:
            stack.extend((child, depth + 1) for child in cast(list[object], item))


def _decode_canonical_json(raw: bytes, *, maximum: int, newline: bool) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        _fail("adapter atomic-v2 JSON bytes are empty or outside their bound")
    if newline:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail("adapter atomic-v2 metadata requires one canonical trailing newline")
        encoded = raw[:-1]
    else:
        encoded = raw
    try:
        text = encoded.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_int=_parse_int,
            parse_constant=_reject_constant,
        )
    except ForagerMatchedV3AdapterAtomicPublicationV2Error:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            "adapter atomic-v2 input is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("adapter atomic-v2 JSON root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if raw != _canonical_json(result, newline=newline):
        _fail("adapter atomic-v2 JSON is not canonical")
    return result


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _read_stable_source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith("/" + expected_suffix):
        _fail(f"cannot resolve exact source path {expected_suffix}")
    path = Path(module_file)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or before_identity != named_identity
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_SOURCE_BYTES
        ):
            _fail(f"source identity is unsafe for {expected_suffix}")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"source truncated while reading {expected_suffix}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"source grew while reading {expected_suffix}")
        after = os.fstat(descriptor)
        final_named = os.stat(path, follow_symlinks=False)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        final_named_identity = (
            final_named.st_dev,
            final_named.st_ino,
            final_named.st_mode,
            final_named.st_size,
            final_named.st_mtime_ns,
            final_named.st_ctime_ns,
            final_named.st_nlink,
        )
        if (
            not stat.S_ISREG(final_named.st_mode)
            or before_identity != after_identity
            or before_identity != final_named_identity
            or after.st_nlink != 1
        ):
            _fail(f"source changed while reading {expected_suffix}")
        return _sha256(b"".join(chunks))
    except OSError as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            f"cannot read exact source {expected_suffix}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


_ATOMIC_MODULE: Final = "alberta_framework.benchmarks._forager_matched_v3_atomic_publication"
_SOURCE_BINDING_SPECS: Final = (
    ("atomic_publication", _ATOMIC_MODULE, _ATOMIC_SOURCE_PATH, _ATOMIC_SOURCE_SHA256),
    ("strict_scorer", _SCORER_MODULE, _SCORER_SOURCE_PATH, _SCORER_SOURCE_SHA256),
    (
        "adapter_reward_bundle_v1",
        _BUNDLE_MODULE,
        _BUNDLE_SOURCE_PATH,
        _BUNDLE_SOURCE_SHA256,
    ),
    (
        "adapter_reward_publication_v1_layout",
        _LEGACY_PUBLICATION_MODULE,
        _LEGACY_PUBLICATION_SOURCE_PATH,
        _LEGACY_PUBLICATION_SOURCE_SHA256,
    ),
    (
        "full_rainbow_runner",
        _FULL_RUNNER_MODULE,
        _FULL_RUNNER_SOURCE_PATH,
        _FULL_RUNNER_SOURCE_SHA256,
    ),
    (
        "ppo_gru_runner",
        _PPO_RUNNER_MODULE,
        _PPO_RUNNER_SOURCE_PATH,
        _PPO_RUNNER_SOURCE_SHA256,
    ),
)
_GETPID_AT_LOAD: Final = os.getpid
_ATOMIC_OPEN_PARENT_AT_LOAD: Final = atomic._open_parent
_ATOMIC_CLOSE_AT_LOAD: Final = atomic._close_no_raise
_ATOMIC_OPEN_PARENT_CODE_SHA256: Final = (
    "f9fabf16590fe7c9d3c873c31864177dc1699e6040bf84652d2c77348560781a"
)
_ATOMIC_CLOSE_CODE_SHA256: Final = (
    "7c93375033818754e1fede5434fd6225194fd1f9934831c816686761e067cda2"
)
_MODULE_SOURCE_PATHS: Final = {
    _ATOMIC_MODULE: _ATOMIC_SOURCE_PATH,
    _SCORER_MODULE: _SCORER_SOURCE_PATH,
    _BUNDLE_MODULE: _BUNDLE_SOURCE_PATH,
    _LEGACY_PUBLICATION_MODULE: _LEGACY_PUBLICATION_SOURCE_PATH,
    _FULL_RUNNER_MODULE: _FULL_RUNNER_SOURCE_PATH,
    _PPO_RUNNER_MODULE: _PPO_RUNNER_SOURCE_PATH,
}


@dataclass(frozen=True, slots=True)
class _BoundFunction:
    module: types.ModuleType
    name: str
    function: Callable[..., Any]
    code: types.CodeType
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class _DependencyClosure:
    scorer: types.ModuleType
    reward_bundle: types.ModuleType
    legacy_publication: types.ModuleType
    full_runner: types.ModuleType
    ppo_runner: types.ModuleType
    bundle_type: type[Any]
    run_full: Callable[..., Any]
    open_ppo: Callable[..., Any]
    run_ppo: Callable[..., Any]
    build_full: Callable[..., Any]
    build_ppo: Callable[..., Any]
    validate_bundle: Callable[..., Any]
    legacy_manifest: Callable[..., Any]
    legacy_payloads: Callable[..., Any]
    parse_legacy_manifest: Callable[..., Any]
    parse_full_receipt: Callable[..., Any]
    parse_ppo_receipt: Callable[..., Any]
    atomic_publish: Callable[..., Any]
    atomic_load: Callable[..., Any]
    bindings: tuple[_BoundFunction, ...]
    publisher_source_sha256: str


_FunctionSpec = tuple[
    str,
    str,
    str,
    tuple[object, ...] | None,
    dict[str, object] | None,
]
_FUNCTION_SPECS: Final[tuple[_FunctionSpec, ...]] = (
    (
        _ATOMIC_MODULE,
        "_open_parent",
        _ATOMIC_OPEN_PARENT_CODE_SHA256,
        None,
        None,
    ),
    (
        _ATOMIC_MODULE,
        "_close_no_raise",
        _ATOMIC_CLOSE_CODE_SHA256,
        None,
        None,
    ),
    (
        _ATOMIC_MODULE,
        "publish_exact_flat_publication",
        "fdc03419208f2a1bdf313c99a2f8f971c43bbba5363847bd4b22b619bbb7361a",
        None,
        None,
    ),
    (
        _ATOMIC_MODULE,
        "load_exact_flat_publication",
        "0347efc72a354f913a350e47cb22c8ea2b3d24bbd759bf611fc68da14add804b",
        None,
        None,
    ),
    (
        _BUNDLE_MODULE,
        "build_full_rainbow_reward_bundle",
        "41533e9c0c664b587aaea7287cc10817259514df0b2021081203308e666b59df",
        None,
        None,
    ),
    (
        _BUNDLE_MODULE,
        "build_ppo_gru_reward_bundle",
        "2ada0abcffc244d6caa9e0484a157da75f16ea5b525cd49eb1c209f56a13d943",
        None,
        None,
    ),
    (
        _BUNDLE_MODULE,
        "validate_adapter_reward_bundle",
        "a2cf16ceab86aaeebe6ef0a09be3e9980f30751245ee8c70687e15ffb85dcd83",
        None,
        None,
    ),
    (
        _LEGACY_PUBLICATION_MODULE,
        "_publication_manifest_bytes",
        "a05c3492dcea55e135a30a08acf5b43cae993b169de5d34959438fbbe7695224",
        None,
        None,
    ),
    (
        _LEGACY_PUBLICATION_MODULE,
        "_payload_bytes",
        "dacdb1a73d2c66e2e81333cbb43e799942760956285b5493b515a8bd26196a57",
        None,
        None,
    ),
    (
        _LEGACY_PUBLICATION_MODULE,
        "parse_adapter_reward_publication_manifest",
        "adbbcc847602422b0144e9b8f697c06c030dd453801fda18b24cd57d65f018ff",
        None,
        None,
    ),
    (
        _FULL_RUNNER_MODULE,
        "run_matched_v3_full_rainbow",
        "757789124c6bbd34fcf9c481967743937d74328cdd0df73ab8b0ee05861dba40",
        None,
        {"unqualified_engineering": False},
    ),
    (
        _FULL_RUNNER_MODULE,
        "parse_full_rainbow_result_receipt",
        "815b864e69bbc0c85a997643257a37eb5fd6e6c133aeca38ce904a834ff79365",
        None,
        None,
    ),
    (
        _PPO_RUNNER_MODULE,
        "open_matched_v3_ppo_gru_runner_runtime",
        "2e2762b4405a30fe5777c0b8e9c9d78e9071bf96268a77155a94f003c15d709f",
        None,
        None,
    ),
    (
        _PPO_RUNNER_MODULE,
        "run_matched_v3_ppo_gru_production",
        "d0a5ebfb1612915219ab8ea9e3c62370e8071ef5d4e4694476aa58b79d7dd6e2",
        None,
        None,
    ),
    (
        _PPO_RUNNER_MODULE,
        "parse_ppo_gru_result_receipt",
        "c94dce8f3d7622de54d15aaa839b70eb8ea47a9c349f927739ced746f7d10277",
        None,
        {"expected_receipt_sha256": None},
    ),
)


def _portable_normalized_code(code: object, source_path: str) -> types.CodeType:
    if type(code) is not types.CodeType:
        _fail("adapter atomic-v2 nested dependency code type drifted")
    exact = code
    normalized_constants = tuple(
        (
            _portable_normalized_code(constant, source_path)
            if type(constant) is types.CodeType
            else constant
        )
        for constant in exact.co_consts
    )
    return exact.replace(co_filename=source_path, co_consts=normalized_constants)


def _portable_code_sha256(code: types.CodeType, source_path: str) -> str:
    return _sha256(marshal.dumps(_portable_normalized_code(code, source_path)))


def _bind_function(
    module: types.ModuleType,
    name: str,
    code_sha256: str,
    expected_defaults: tuple[object, ...] | None,
    expected_kwdefaults: dict[str, object] | None,
) -> _BoundFunction:
    value = getattr(module, name, None)
    if type(value) is not types.FunctionType:
        _fail(f"adapter atomic-v2 dependency function {name} type drifted")
    function = value
    if (
        function.__module__ != module.__name__
        or function.__qualname__ != name
        or function.__defaults__ != expected_defaults
        or function.__kwdefaults__ != expected_kwdefaults
        or not hmac.compare_digest(
            _portable_code_sha256(
                function.__code__,
                _MODULE_SOURCE_PATHS[module.__name__],
            ),
            code_sha256,
        )
    ):
        _fail(f"adapter atomic-v2 dependency function {name} identity drifted")
    return _BoundFunction(
        module=module,
        name=name,
        function=function,
        code=function.__code__,
        defaults=copy.deepcopy(function.__defaults__),
        kwdefaults=copy.deepcopy(function.__kwdefaults__),
    )


def _require_dependency_closure(closure: _DependencyClosure) -> str:
    modules = {
        _ATOMIC_MODULE: atomic,
        _SCORER_MODULE: closure.scorer,
        _BUNDLE_MODULE: closure.reward_bundle,
        _LEGACY_PUBLICATION_MODULE: closure.legacy_publication,
        _FULL_RUNNER_MODULE: closure.full_runner,
        _PPO_RUNNER_MODULE: closure.ppo_runner,
    }
    for _component, module_name, path, expected in _SOURCE_BINDING_SPECS:
        module = modules[module_name]
        if module.__name__ != module_name:
            _fail(f"adapter atomic-v2 dependency module identity drifted for {path}")
        observed = _read_stable_source_sha256(getattr(module, "__file__", None), path)
        if not hmac.compare_digest(observed, expected):
            _fail(f"adapter atomic-v2 dependency source drifted for {path}")
    descriptor_checks = (
        (
            getattr(atomic, "ATOMIC_PUBLICATION_DESCRIPTOR_SHA256", None),
            _ATOMIC_DESCRIPTOR_SHA256,
        ),
        (
            getattr(closure.reward_bundle, "ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256", None),
            _BUNDLE_DESCRIPTOR_SHA256,
        ),
        (
            getattr(
                closure.legacy_publication,
                "ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
                None,
            ),
            _LEGACY_PUBLICATION_DESCRIPTOR_SHA256,
        ),
        (
            getattr(closure.full_runner, "FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256", None),
            _FULL_RUNNER_DESCRIPTOR_SHA256,
        ),
        (
            getattr(closure.ppo_runner, "PPO_GRU_RUNNER_DESCRIPTOR_SHA256", None),
            _PPO_RUNNER_DESCRIPTOR_SHA256,
        ),
    )
    if any(
        type(actual) is not str or not hmac.compare_digest(actual, expected)
        for actual, expected in descriptor_checks
    ):
        _fail("adapter atomic-v2 dependency descriptor drifted")
    for binding in closure.bindings:
        function = binding.function
        if (
            getattr(binding.module, binding.name, None) is not function
            or type(function) is not types.FunctionType
            or function.__code__ is not binding.code
            or function.__defaults__ != binding.defaults
            or function.__kwdefaults__ != binding.kwdefaults
        ):
            _fail(f"adapter atomic-v2 dependency function {binding.name} was replaced")
    bound = {
        (binding.module.__name__, binding.name): binding.function
        for binding in closure.bindings
    }
    callable_fields = (
        (closure.run_full, bound[(_FULL_RUNNER_MODULE, "run_matched_v3_full_rainbow")]),
        (
            closure.open_ppo,
            bound[(_PPO_RUNNER_MODULE, "open_matched_v3_ppo_gru_runner_runtime")],
        ),
        (closure.run_ppo, bound[(_PPO_RUNNER_MODULE, "run_matched_v3_ppo_gru_production")]),
        (closure.build_full, bound[(_BUNDLE_MODULE, "build_full_rainbow_reward_bundle")]),
        (closure.build_ppo, bound[(_BUNDLE_MODULE, "build_ppo_gru_reward_bundle")]),
        (closure.validate_bundle, bound[(_BUNDLE_MODULE, "validate_adapter_reward_bundle")]),
        (
            closure.legacy_manifest,
            bound[(_LEGACY_PUBLICATION_MODULE, "_publication_manifest_bytes")],
        ),
        (
            closure.legacy_payloads,
            bound[(_LEGACY_PUBLICATION_MODULE, "_payload_bytes")],
        ),
        (
            closure.parse_legacy_manifest,
            bound[
                (
                    _LEGACY_PUBLICATION_MODULE,
                    "parse_adapter_reward_publication_manifest",
                )
            ],
        ),
        (
            closure.parse_full_receipt,
            bound[(_FULL_RUNNER_MODULE, "parse_full_rainbow_result_receipt")],
        ),
        (
            closure.parse_ppo_receipt,
            bound[(_PPO_RUNNER_MODULE, "parse_ppo_gru_result_receipt")],
        ),
        (
            closure.atomic_publish,
            bound[(_ATOMIC_MODULE, "publish_exact_flat_publication")],
        ),
        (closure.atomic_load, bound[(_ATOMIC_MODULE, "load_exact_flat_publication")]),
    )
    if any(actual is not expected for actual, expected in callable_fields):
        _fail("adapter atomic-v2 dependency closure callable binding drifted")
    if (
        getattr(closure.reward_bundle, "MatchedV3AdapterRewardBundle", None)
        is not closure.bundle_type
        or closure.bundle_type.__module__ != _BUNDLE_MODULE
        or closure.bundle_type.__qualname__ != "MatchedV3AdapterRewardBundle"
    ):
        _fail("adapter atomic-v2 dependency bundle type was replaced")
    if os.getpid is not _GETPID_AT_LOAD:
        _fail("adapter atomic-v2 PID primitive was replaced")
    if (
        _canonical_json(_descriptor()) != _DESCRIPTOR_BYTES
        or not hmac.compare_digest(
            _sha256(_DESCRIPTOR_BYTES),
            ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256,
        )
    ):
        _fail("adapter atomic-v2 own descriptor failed exact replay")
    source_sha256 = _read_stable_source_sha256(__file__, _IMPLEMENTATION_PATH)
    if not hmac.compare_digest(source_sha256, closure.publisher_source_sha256):
        _fail("adapter atomic-v2 publisher source changed during operation")
    return source_sha256


def _load_dependency_closure() -> _DependencyClosure:
    modules = {
        _ATOMIC_MODULE: atomic,
        _SCORER_MODULE: importlib.import_module(_SCORER_MODULE),
        _BUNDLE_MODULE: importlib.import_module(_BUNDLE_MODULE),
        _LEGACY_PUBLICATION_MODULE: importlib.import_module(_LEGACY_PUBLICATION_MODULE),
        _FULL_RUNNER_MODULE: importlib.import_module(_FULL_RUNNER_MODULE),
        _PPO_RUNNER_MODULE: importlib.import_module(_PPO_RUNNER_MODULE),
    }
    bindings = tuple(
        _bind_function(
            modules[module_name],
            name,
            code_sha256,
            expected_defaults,
            expected_kwdefaults,
        )
        for (
            module_name,
            name,
            code_sha256,
            expected_defaults,
            expected_kwdefaults,
        ) in _FUNCTION_SPECS
    )
    by_name = {(item.module.__name__, item.name): item.function for item in bindings}
    bundle_type = getattr(modules[_BUNDLE_MODULE], "MatchedV3AdapterRewardBundle", None)
    if type(bundle_type) is not type:
        _fail("adapter atomic-v2 dependency bundle type drifted")
    publisher_source = _read_stable_source_sha256(__file__, _IMPLEMENTATION_PATH)
    closure = _DependencyClosure(
        scorer=modules[_SCORER_MODULE],
        reward_bundle=modules[_BUNDLE_MODULE],
        legacy_publication=modules[_LEGACY_PUBLICATION_MODULE],
        full_runner=modules[_FULL_RUNNER_MODULE],
        ppo_runner=modules[_PPO_RUNNER_MODULE],
        bundle_type=cast(type[Any], bundle_type),
        run_full=by_name[(_FULL_RUNNER_MODULE, "run_matched_v3_full_rainbow")],
        open_ppo=by_name[
            (_PPO_RUNNER_MODULE, "open_matched_v3_ppo_gru_runner_runtime")
        ],
        run_ppo=by_name[(_PPO_RUNNER_MODULE, "run_matched_v3_ppo_gru_production")],
        build_full=by_name[(_BUNDLE_MODULE, "build_full_rainbow_reward_bundle")],
        build_ppo=by_name[(_BUNDLE_MODULE, "build_ppo_gru_reward_bundle")],
        validate_bundle=by_name[(_BUNDLE_MODULE, "validate_adapter_reward_bundle")],
        legacy_manifest=by_name[
            (_LEGACY_PUBLICATION_MODULE, "_publication_manifest_bytes")
        ],
        legacy_payloads=by_name[(_LEGACY_PUBLICATION_MODULE, "_payload_bytes")],
        parse_legacy_manifest=by_name[
            (_LEGACY_PUBLICATION_MODULE, "parse_adapter_reward_publication_manifest")
        ],
        parse_full_receipt=by_name[
            (_FULL_RUNNER_MODULE, "parse_full_rainbow_result_receipt")
        ],
        parse_ppo_receipt=by_name[(_PPO_RUNNER_MODULE, "parse_ppo_gru_result_receipt")],
        atomic_publish=by_name[(_ATOMIC_MODULE, "publish_exact_flat_publication")],
        atomic_load=by_name[(_ATOMIC_MODULE, "load_exact_flat_publication")],
        bindings=bindings,
        publisher_source_sha256=publisher_source,
    )
    _require_dependency_closure(closure)
    return closure


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "campaign_ingestion_authorized": False,
        "evidence_authority": False,
        "execution_authority_granted": False,
        "performance_claim_allowed": False,
        "qualification_authority": False,
        "runner_surface_authorized": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        (
            "The Full Rainbow entrypoint is exact full-horizon but explicitly "
            "unqualified engineering, not an authorized qualification runner."
        ),
        (
            "The PPO-GRU production adapter runtime is explicitly unqualified and "
            "grants no qualification authority."
        ),
        (
            "This is not an implemented strict qualification publisher; the "
            "adapter-publisher v3 registry gap remains open."
        ),
        (
            "The public path is callable repeatedly; it makes exactly one atomic "
            "publish call per invocation and provides no campaign-wide single-use "
            "or retry coordinator."
        ),
        (
            "Returning only filenames, digests and sizes does not decode score fields "
            "or magnitudes, but is not proof of content opacity."
        ),
        (
            "A same-UID or privileged process can read published score-bearing files; "
            "hashes and sizes are content fingerprints."
        ),
        (
            "Neither publication nor normal reload proves a fresh host worker, fresh "
            "build, empty cgroup, network namespace, or all-descendant cleanup."
        ),
        (
            "Safe-parent preflight rejects predictable hazards before runner import, "
            "but cannot freeze the directory during a workload; the atomic primitive "
            "revalidates it at publication."
        ),
        (
            "No existing stale or unqualified image is converted into a fresh "
            "qualified build by this module."
        ),
        (
            "The local source-tree digest is caller-carried and must be checked by an "
            "external source and build closure."
        ),
        (
            "No function or receipt here authorizes execution, ingestion, evidence "
            "acceptance, comparison, or promotion."
        ),
    ]


def _source_binding_records() -> list[dict[str, str]]:
    return [
        {
            "component_id": name,
            "implementation_path": path,
            "implementation_source_sha256": source_sha256,
        }
        for name, _module_name, path, source_sha256 in _SOURCE_BINDING_SPECS
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SCHEMA_VERSION,
        "status": ADAPTER_ATOMIC_PUBLICATION_V2_STATUS,
        "classification": "additive_fused_unqualified_metadata_only_non_authorizing",
        "implementation_path": _IMPLEMENTATION_PATH,
        "candidate_ids": list(ADAPTER_ATOMIC_PUBLICATION_V2_CANDIDATE_IDS),
        "source_bindings": _source_binding_records(),
        "descriptor_bindings": {
            "atomic_publication": _ATOMIC_DESCRIPTOR_SHA256,
            "adapter_reward_bundle_v1": _BUNDLE_DESCRIPTOR_SHA256,
            "adapter_reward_publication_v1_layout": (
                _LEGACY_PUBLICATION_DESCRIPTOR_SHA256
            ),
            "full_rainbow_runner": _FULL_RUNNER_DESCRIPTOR_SHA256,
            "ppo_gru_runner": _PPO_RUNNER_DESCRIPTOR_SHA256,
        },
        "fused_contract": {
            "one_pid_from_runner_entry_through_atomic_publication": True,
            "public_result_or_outcome_input": False,
            "public_bundle_input": False,
            "public_payload_bytes_input": False,
            "public_callback_input": False,
            "metadata_only_return": True,
            "canonical_metadata_receipt_and_strict_parser": True,
            "exact_base_file_layout": list(ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES),
            "exact_safe_parent_preflight_before_runner_import": True,
            "one_atomic_publish_call_per_public_invocation": True,
            "global_retry_or_single_use_coordinator": False,
            "full_rainbow_surface": "exact_full_horizon_unqualified_engineering",
            "ppo_gru_surface": "production_adapter_runtime_unqualified",
        },
        "trust_graph": {
            "direction": "atomic_v2_to_frozen_v1_bundle_runners_layout_and_atomic_primitive",
            "reverse_source_pin_required": False,
            "existing_v1_sources_modified": False,
        },
        "qualification_registry": {
            "implemented_strict_qualification_publisher": False,
            "v3_adapter_publisher_registry_gap_remains": True,
            "only_executable_public_surfaces_are_explicitly_unqualified": True,
        },
        "readiness": {
            "authorized_full_rainbow_runner_surface_available": False,
            "authorized_ppo_gru_runner_surface_available": False,
            "fresh_host_worker_proven": False,
            "empty_cgroup_and_descendant_cleanup_proven": False,
            "metadata_only_controller_ipc_proven": False,
            "fresh_qualified_build_proven": False,
            "qualification_ready": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256: Final = (
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6"
)
if not hmac.compare_digest(
    _sha256(_DESCRIPTOR_BYTES),
    ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256,
):
    raise RuntimeError("adapter atomic-v2 descriptor identity drifted")


def adapter_atomic_publication_v2_descriptor() -> dict[str, Any]:
    """Return detached additive publisher descriptor content."""

    return copy.deepcopy(_descriptor())


def canonical_adapter_atomic_publication_v2_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_adapter_atomic_publication_v2_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen descriptor; it grants no execution authority."""

    parsed = _decode_canonical_json(raw, maximum=_MAX_DESCRIPTOR_BYTES, newline=False)
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        _sha256(raw), ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256
    ):
        _fail("adapter atomic-v2 descriptor identity drifted")
    return copy.deepcopy(parsed)


@dataclass(frozen=True, slots=True)
class MatchedV3AdapterAtomicPublicationFileV2:
    """One filename, byte length, and content fingerprint."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES:
            _fail("adapter atomic-v2 file name differs")
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 1
            or self.size_bytes > _MAX_PUBLICATION_FILE_BYTES
        ):
            _fail("adapter atomic-v2 file size is outside the atomic bound")
        _require_sha256(self.sha256, "adapter atomic-v2 file digest")


@dataclass(frozen=True, slots=True)
class MatchedV3AdapterAtomicPublicationMetadataV2:
    """Canonical digest/size metadata retaining no publication payload bytes."""

    schema_version: str
    operation: Literal["published", "reloaded"]
    publication_root: Path
    address: str
    candidate_id: str
    environment_seed: int
    agent_seed: int
    operation_pid: int
    local_source_tree_sha256: str
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    legacy_publication_descriptor_sha256: str
    legacy_publication_source_sha256: str
    bundle_descriptor_sha256: str
    bundle_source_sha256: str
    runner_descriptor_sha256: str
    runner_source_sha256: str
    atomic_descriptor_sha256: str
    atomic_source_sha256: str
    publication_manifest_sha256: str
    publication_manifest_body_sha256: str
    bundle_manifest_sha256: str
    bundle_manifest_body_sha256: str
    runner_receipt_sha256: str
    reward_artifact_sha256: str
    score_receipt_sha256: str
    file_count: int
    total_size_bytes: int
    inventory_sha256: str
    content_projection_sha256: str
    files: tuple[MatchedV3AdapterAtomicPublicationFileV2, ...]
    metadata_body_sha256: str

    def __post_init__(self) -> None:
        _validate_metadata(self)


def _record_body(
    records: tuple[MatchedV3AdapterAtomicPublicationFileV2, ...],
) -> list[dict[str, Any]]:
    return [
        {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in records
    ]


def _validate_records(
    value: object,
) -> tuple[MatchedV3AdapterAtomicPublicationFileV2, ...]:
    if type(value) is not tuple:
        _fail("adapter atomic-v2 file records must be one exact tuple")
    records = cast(tuple[object, ...], value)
    if (
        len(records) != len(ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES)
        or any(type(item) is not MatchedV3AdapterAtomicPublicationFileV2 for item in records)
    ):
        _fail("adapter atomic-v2 file record inventory differs")
    exact = cast(tuple[MatchedV3AdapterAtomicPublicationFileV2, ...], records)
    if tuple(item.name for item in exact) != ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES:
        _fail("adapter atomic-v2 file record order differs")
    return exact


def _content_projection_body(
    metadata: MatchedV3AdapterAtomicPublicationMetadataV2,
) -> dict[str, Any]:
    return {
        "address": metadata.address,
        "candidate_id": metadata.candidate_id,
        "environment_seed": metadata.environment_seed,
        "agent_seed": metadata.agent_seed,
        "local_source_tree_sha256": metadata.local_source_tree_sha256,
        "publisher_descriptor_sha256": metadata.publisher_descriptor_sha256,
        "publisher_source_sha256": metadata.publisher_source_sha256,
        "legacy_publication_descriptor_sha256": (
            metadata.legacy_publication_descriptor_sha256
        ),
        "legacy_publication_source_sha256": metadata.legacy_publication_source_sha256,
        "bundle_descriptor_sha256": metadata.bundle_descriptor_sha256,
        "bundle_source_sha256": metadata.bundle_source_sha256,
        "runner_descriptor_sha256": metadata.runner_descriptor_sha256,
        "runner_source_sha256": metadata.runner_source_sha256,
        "atomic_descriptor_sha256": metadata.atomic_descriptor_sha256,
        "atomic_source_sha256": metadata.atomic_source_sha256,
        "publication_manifest_sha256": metadata.publication_manifest_sha256,
        "publication_manifest_body_sha256": metadata.publication_manifest_body_sha256,
        "bundle_manifest_sha256": metadata.bundle_manifest_sha256,
        "bundle_manifest_body_sha256": metadata.bundle_manifest_body_sha256,
        "runner_receipt_sha256": metadata.runner_receipt_sha256,
        "reward_artifact_sha256": metadata.reward_artifact_sha256,
        "score_receipt_sha256": metadata.score_receipt_sha256,
        "file_count": metadata.file_count,
        "total_size_bytes": metadata.total_size_bytes,
        "inventory_sha256": metadata.inventory_sha256,
        "files": _record_body(metadata.files),
    }


def _metadata_body(
    metadata: MatchedV3AdapterAtomicPublicationMetadataV2,
) -> dict[str, Any]:
    return {
        "schema_version": metadata.schema_version,
        "status": ADAPTER_ATOMIC_PUBLICATION_V2_STATUS,
        "classification": "digest_size_metadata_non_authorizing",
        "operation": metadata.operation,
        "publication_root": str(metadata.publication_root),
        "operation_pid": metadata.operation_pid,
        **_content_projection_body(metadata),
        "content_projection_sha256": metadata.content_projection_sha256,
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _validate_metadata(
    value: object,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    if type(value) is not MatchedV3AdapterAtomicPublicationMetadataV2:
        _fail("adapter atomic-v2 metadata type differs")
    exact = value
    if (
        exact.schema_version != ADAPTER_ATOMIC_PUBLICATION_V2_METADATA_SCHEMA_VERSION
        or type(exact.operation) is not str
        or exact.operation not in {"published", "reloaded"}
    ):
        _fail("adapter atomic-v2 metadata identity differs")
    if (
        type(exact.publication_root) is not _PATH_TYPE
        or not exact.publication_root.is_absolute()
        or exact.publication_root == Path("/")
        or exact.publication_root.name != exact.address
    ):
        _fail("adapter atomic-v2 metadata publication root differs")
    _require_candidate(exact.candidate_id)
    _require_uint31(exact.environment_seed, "environment seed")
    _require_uint31(exact.agent_seed, "agent seed")
    if type(exact.operation_pid) is not int or exact.operation_pid <= 0:
        _fail("adapter atomic-v2 metadata PID differs")
    for label, digest in (
        ("address", exact.address),
        ("local source tree", exact.local_source_tree_sha256),
        ("publisher descriptor", exact.publisher_descriptor_sha256),
        ("publisher source", exact.publisher_source_sha256),
        ("legacy publication descriptor", exact.legacy_publication_descriptor_sha256),
        ("legacy publication source", exact.legacy_publication_source_sha256),
        ("bundle descriptor", exact.bundle_descriptor_sha256),
        ("bundle source", exact.bundle_source_sha256),
        ("runner descriptor", exact.runner_descriptor_sha256),
        ("runner source", exact.runner_source_sha256),
        ("atomic descriptor", exact.atomic_descriptor_sha256),
        ("atomic source", exact.atomic_source_sha256),
        ("publication manifest", exact.publication_manifest_sha256),
        ("publication manifest body", exact.publication_manifest_body_sha256),
        ("bundle manifest", exact.bundle_manifest_sha256),
        ("bundle manifest body", exact.bundle_manifest_body_sha256),
        ("runner receipt", exact.runner_receipt_sha256),
        ("reward artifact", exact.reward_artifact_sha256),
        ("score receipt", exact.score_receipt_sha256),
        ("inventory", exact.inventory_sha256),
        ("content projection", exact.content_projection_sha256),
        ("metadata body", exact.metadata_body_sha256),
    ):
        _require_sha256(digest, label)
    records = _validate_records(exact.files)
    expected_runner = (
        (
            _FULL_RUNNER_DESCRIPTOR_SHA256,
            _FULL_RUNNER_SOURCE_SHA256,
        )
        if exact.candidate_id == "adapted_full_rainbow"
        else (_PPO_RUNNER_DESCRIPTOR_SHA256, _PPO_RUNNER_SOURCE_SHA256)
    )
    by_name = {item.name: item for item in records}
    if (
        exact.address != exact.publication_manifest_sha256
        or exact.publisher_descriptor_sha256
        != ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256
        or exact.legacy_publication_descriptor_sha256
        != _LEGACY_PUBLICATION_DESCRIPTOR_SHA256
        or exact.legacy_publication_source_sha256 != _LEGACY_PUBLICATION_SOURCE_SHA256
        or exact.bundle_descriptor_sha256
        != _BUNDLE_DESCRIPTOR_SHA256
        or exact.bundle_source_sha256 != _BUNDLE_SOURCE_SHA256
        or (exact.runner_descriptor_sha256, exact.runner_source_sha256) != expected_runner
        or exact.atomic_descriptor_sha256 != _ATOMIC_DESCRIPTOR_SHA256
        or exact.atomic_source_sha256 != _ATOMIC_SOURCE_SHA256
        or exact.publication_manifest_sha256
        != by_name[PUBLICATION_MANIFEST_FILENAME].sha256
        or exact.bundle_manifest_sha256 != by_name[ADAPTER_BUNDLE_MANIFEST_FILENAME].sha256
        or exact.runner_receipt_sha256 != by_name[RUNNER_RESULT_RECEIPT_FILENAME].sha256
        or exact.reward_artifact_sha256 != by_name[REWARD_TRACE_FILENAME].sha256
        or exact.score_receipt_sha256 != by_name[SCORE_RECEIPT_FILENAME].sha256
        or type(exact.file_count) is not int
        or exact.file_count != len(records)
        or type(exact.total_size_bytes) is not int
        or exact.total_size_bytes != sum(item.size_bytes for item in records)
        or exact.total_size_bytes > _MAX_PUBLICATION_TOTAL_BYTES
    ):
        _fail("adapter atomic-v2 metadata fixed binding differs")
    inventory = _sha256(_canonical_json(_record_body(records)))
    if not hmac.compare_digest(inventory, exact.inventory_sha256):
        _fail("adapter atomic-v2 metadata inventory does not replay")
    projection = _sha256(_canonical_json(_content_projection_body(exact)))
    if not hmac.compare_digest(projection, exact.content_projection_sha256):
        _fail("adapter atomic-v2 content projection does not replay")
    body_sha256 = _sha256(_canonical_json(_metadata_body(exact)))
    if not hmac.compare_digest(body_sha256, exact.metadata_body_sha256):
        _fail("adapter atomic-v2 metadata body does not replay")
    return exact


def canonical_adapter_atomic_publication_v2_metadata_bytes(
    metadata: MatchedV3AdapterAtomicPublicationMetadataV2,
) -> bytes:
    """Serialize payload-free digest/size metadata without decoding score magnitudes."""

    exact = _validate_metadata(metadata)
    body = _metadata_body(exact)
    receipt = dict(body)
    receipt["metadata_body_sha256"] = exact.metadata_body_sha256
    raw = _canonical_json(receipt, newline=True)
    if len(raw) > _MAX_METADATA_BYTES:
        _fail("adapter atomic-v2 metadata receipt exceeds its byte bound")
    return raw


def _file_record_from_dict(value: object) -> MatchedV3AdapterAtomicPublicationFileV2:
    item = _require_exact_keys(
        value,
        frozenset({"name", "sha256", "size_bytes"}),
        "adapter atomic-v2 metadata file record",
    )
    return MatchedV3AdapterAtomicPublicationFileV2(
        name=item["name"], size_bytes=item["size_bytes"], sha256=item["sha256"]
    )


_METADATA_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "classification",
        "operation",
        "publication_root",
        "operation_pid",
        "address",
        "candidate_id",
        "environment_seed",
        "agent_seed",
        "local_source_tree_sha256",
        "publisher_descriptor_sha256",
        "publisher_source_sha256",
        "legacy_publication_descriptor_sha256",
        "legacy_publication_source_sha256",
        "bundle_descriptor_sha256",
        "bundle_source_sha256",
        "runner_descriptor_sha256",
        "runner_source_sha256",
        "atomic_descriptor_sha256",
        "atomic_source_sha256",
        "publication_manifest_sha256",
        "publication_manifest_body_sha256",
        "bundle_manifest_sha256",
        "bundle_manifest_body_sha256",
        "runner_receipt_sha256",
        "reward_artifact_sha256",
        "score_receipt_sha256",
        "file_count",
        "total_size_bytes",
        "inventory_sha256",
        "content_projection_sha256",
        "files",
        "claims",
        "limitations",
    }
)


def parse_adapter_atomic_publication_v2_metadata(
    raw: bytes,
    *,
    expected_full_file_sha256: str,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    """Strictly parse metadata under an independently carried full-file digest."""

    expected = _require_sha256(expected_full_file_sha256, "metadata full-file digest")
    if type(raw) is not bytes or not hmac.compare_digest(_sha256(raw), expected):
        _fail("adapter atomic-v2 metadata full-file digest differs")
    payload = _decode_canonical_json(raw, maximum=_MAX_METADATA_BYTES, newline=True)
    receipt = _require_exact_keys(
        payload,
        frozenset({*_METADATA_BODY_KEYS, "metadata_body_sha256"}),
        "adapter atomic-v2 metadata receipt",
    )
    supplied_body = _require_sha256(
        receipt["metadata_body_sha256"], "metadata body digest"
    )
    body = dict(receipt)
    del body["metadata_body_sha256"]
    if not hmac.compare_digest(_sha256(_canonical_json(body)), supplied_body):
        _fail("adapter atomic-v2 metadata receipt body digest differs")
    if (
        body["status"] != ADAPTER_ATOMIC_PUBLICATION_V2_STATUS
        or body["classification"] != "digest_size_metadata_non_authorizing"
        or body["claims"] != _claims()
        or body["limitations"] != _limitations()
    ):
        _fail("adapter atomic-v2 metadata policy differs")
    raw_records = body["files"]
    if type(raw_records) is not list:
        _fail("adapter atomic-v2 metadata files must be one list")
    records = tuple(_file_record_from_dict(item) for item in raw_records)
    publication_root_text = body["publication_root"]
    if type(publication_root_text) is not str:
        _fail("adapter atomic-v2 metadata publication root must be exact text")
    metadata = MatchedV3AdapterAtomicPublicationMetadataV2(
        schema_version=body["schema_version"],
        operation=body["operation"],
        publication_root=Path(publication_root_text),
        address=body["address"],
        candidate_id=body["candidate_id"],
        environment_seed=body["environment_seed"],
        agent_seed=body["agent_seed"],
        operation_pid=body["operation_pid"],
        local_source_tree_sha256=body["local_source_tree_sha256"],
        publisher_descriptor_sha256=body["publisher_descriptor_sha256"],
        publisher_source_sha256=body["publisher_source_sha256"],
        legacy_publication_descriptor_sha256=body[
            "legacy_publication_descriptor_sha256"
        ],
        legacy_publication_source_sha256=body["legacy_publication_source_sha256"],
        bundle_descriptor_sha256=body["bundle_descriptor_sha256"],
        bundle_source_sha256=body["bundle_source_sha256"],
        runner_descriptor_sha256=body["runner_descriptor_sha256"],
        runner_source_sha256=body["runner_source_sha256"],
        atomic_descriptor_sha256=body["atomic_descriptor_sha256"],
        atomic_source_sha256=body["atomic_source_sha256"],
        publication_manifest_sha256=body["publication_manifest_sha256"],
        publication_manifest_body_sha256=body["publication_manifest_body_sha256"],
        bundle_manifest_sha256=body["bundle_manifest_sha256"],
        bundle_manifest_body_sha256=body["bundle_manifest_body_sha256"],
        runner_receipt_sha256=body["runner_receipt_sha256"],
        reward_artifact_sha256=body["reward_artifact_sha256"],
        score_receipt_sha256=body["score_receipt_sha256"],
        file_count=body["file_count"],
        total_size_bytes=body["total_size_bytes"],
        inventory_sha256=body["inventory_sha256"],
        content_projection_sha256=body["content_projection_sha256"],
        files=records,
        metadata_body_sha256=supplied_body,
    )
    if canonical_adapter_atomic_publication_v2_metadata_bytes(metadata) != raw:
        _fail("adapter atomic-v2 metadata receipt does not replay")
    return metadata


@dataclass(frozen=True, slots=True)
class _LoadedFacts:
    candidate_id: str
    environment_seed: int
    agent_seed: int
    publication_body_sha256: str
    bundle_manifest_body_sha256: str
    runner_descriptor_sha256: str
    runner_source_sha256: str


def _atomic_records(
    records: tuple[MatchedV3AdapterAtomicPublicationFileV2, ...],
) -> tuple[atomic.ExactFileRecord, ...]:
    return tuple(
        atomic.ExactFileRecord(
            name=item.name, size_bytes=item.size_bytes, sha256=item.sha256
        )
        for item in records
    )


def _records_from_payloads(
    payloads: Mapping[str, bytes],
) -> tuple[MatchedV3AdapterAtomicPublicationFileV2, ...]:
    if set(payloads) != set(ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES):
        _fail("adapter atomic-v2 payload inventory differs")
    return tuple(
        MatchedV3AdapterAtomicPublicationFileV2(
            name=name,
            size_bytes=len(payloads[name]),
            sha256=_sha256(payloads[name]),
        )
        for name in ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES
    )


def _runner_seed_facts(
    closure: _DependencyClosure,
    candidate_id: str,
    receipt: bytes,
) -> tuple[int, int, str, str]:
    try:
        if candidate_id == "adapted_full_rainbow":
            parsed = closure.parse_full_receipt(receipt)
            descriptor = _FULL_RUNNER_DESCRIPTOR_SHA256
            source = _FULL_RUNNER_SOURCE_SHA256
        elif candidate_id == "adapted_ppo_gru":
            parsed = closure.parse_ppo_receipt(receipt)
            descriptor = _PPO_RUNNER_DESCRIPTOR_SHA256
            source = _PPO_RUNNER_SOURCE_SHA256
        else:
            _fail("adapter atomic-v2 runner candidate differs")
    except Exception as exc:
        expected_errors = (
            getattr(closure.full_runner, "FullRainbowRunnerContractError"),
            getattr(closure.ppo_runner, "ForagerMatchedV3PPOGRURunnerError"),
        )
        if isinstance(exc, expected_errors):
            raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
                "adapter atomic-v2 runner receipt failed its exact parser"
            ) from exc
        raise
    seeds = cast(dict[str, Any], parsed["seeds"])
    return (
        _require_uint31(seeds["environment_seed"], "runner environment seed"),
        _require_uint31(seeds["agent_seed"], "runner agent seed"),
        descriptor,
        source,
    )


def _validate_loaded_payloads(
    *,
    closure: _DependencyClosure,
    address: str,
    files: Mapping[str, bytes],
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
) -> _LoadedFacts:
    if type(files) not in {dict, types.MappingProxyType} or set(files) != set(
        ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES
    ):
        _fail("adapter atomic-v2 loaded byte inventory differs")
    publication_raw = files[PUBLICATION_MANIFEST_FILENAME]
    bundle_raw = files[ADAPTER_BUNDLE_MANIFEST_FILENAME]
    runner_raw = files[RUNNER_RESULT_RECEIPT_FILENAME]
    reward_raw = files[REWARD_TRACE_FILENAME]
    score_raw = files[SCORE_RECEIPT_FILENAME]
    try:
        outer = closure.parse_legacy_manifest(
            publication_raw,
            expected_publication_file_sha256=address,
        )
        dependency = cast(dict[str, Any], outer["adapter_reward_bundle"])
        reconstructed = closure.bundle_type(
            candidate_id=cast(str, outer["candidate_id"]),
            runner_receipt_bytes=runner_raw,
            reward_artifact_bytes=reward_raw,
            score_receipt_bytes=score_raw,
            manifest_bytes=bundle_raw,
            manifest_sha256=cast(str, dependency["manifest_body_sha256"]),
        )
        validated = closure.validate_bundle(reconstructed)
        replayed, replayed_body, replayed_file = closure.legacy_manifest(validated)
    except Exception as exc:
        expected_errors = (
            getattr(
                closure.legacy_publication,
                "ForagerMatchedV3AdapterRewardPublicationError",
            ),
            getattr(
                closure.reward_bundle,
                "ForagerMatchedV3AdapterRewardBundleError",
            ),
        )
        if isinstance(exc, expected_errors):
            raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
                "adapter atomic-v2 persisted content failed exact structural replay"
            ) from exc
        raise
    candidate = _require_candidate(outer["candidate_id"])
    environment_seed, agent_seed, runner_descriptor, runner_source = _runner_seed_facts(
        closure, candidate, runner_raw
    )
    if (
        replayed != publication_raw
        or replayed_file != address
        or replayed_body != outer["publication_body_sha256"]
        or candidate != expected_candidate_id
        or environment_seed != expected_environment_seed
        or agent_seed != expected_agent_seed
    ):
        _fail("adapter atomic-v2 content differs from its expected cell or replay")
    return _LoadedFacts(
        candidate_id=candidate,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        publication_body_sha256=cast(str, outer["publication_body_sha256"]),
        bundle_manifest_body_sha256=cast(str, dependency["manifest_body_sha256"]),
        runner_descriptor_sha256=runner_descriptor,
        runner_source_sha256=runner_source,
    )


def _metadata_from_flat_publication(
    *,
    closure: _DependencyClosure,
    operation: Literal["published", "reloaded"],
    operation_pid: int,
    publisher_source_sha256: str,
    local_source_tree_sha256: str,
    flat: atomic.ContentVerifiedFlatPublication,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    address = _require_sha256(flat.address, "atomic publication address")
    files = flat.files
    facts = _validate_loaded_payloads(
        closure=closure,
        address=address,
        files=files,
        expected_candidate_id=expected_candidate_id,
        expected_environment_seed=expected_environment_seed,
        expected_agent_seed=expected_agent_seed,
    )
    records = _records_from_payloads(files)
    by_name = {item.name: item for item in records}
    inventory = _sha256(_canonical_json(_record_body(records)))
    partial = MatchedV3AdapterAtomicPublicationMetadataV2.__new__(
        MatchedV3AdapterAtomicPublicationMetadataV2
    )
    values: dict[str, Any] = {
        "schema_version": ADAPTER_ATOMIC_PUBLICATION_V2_METADATA_SCHEMA_VERSION,
        "operation": operation,
        "publication_root": flat.root,
        "address": address,
        "candidate_id": facts.candidate_id,
        "environment_seed": facts.environment_seed,
        "agent_seed": facts.agent_seed,
        "operation_pid": operation_pid,
        "local_source_tree_sha256": local_source_tree_sha256,
        "publisher_descriptor_sha256": ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256,
        "publisher_source_sha256": publisher_source_sha256,
        "legacy_publication_descriptor_sha256": _LEGACY_PUBLICATION_DESCRIPTOR_SHA256,
        "legacy_publication_source_sha256": _LEGACY_PUBLICATION_SOURCE_SHA256,
        "bundle_descriptor_sha256": _BUNDLE_DESCRIPTOR_SHA256,
        "bundle_source_sha256": _BUNDLE_SOURCE_SHA256,
        "runner_descriptor_sha256": facts.runner_descriptor_sha256,
        "runner_source_sha256": facts.runner_source_sha256,
        "atomic_descriptor_sha256": _ATOMIC_DESCRIPTOR_SHA256,
        "atomic_source_sha256": _ATOMIC_SOURCE_SHA256,
        "publication_manifest_sha256": address,
        "publication_manifest_body_sha256": facts.publication_body_sha256,
        "bundle_manifest_sha256": by_name[ADAPTER_BUNDLE_MANIFEST_FILENAME].sha256,
        "bundle_manifest_body_sha256": facts.bundle_manifest_body_sha256,
        "runner_receipt_sha256": by_name[RUNNER_RESULT_RECEIPT_FILENAME].sha256,
        "reward_artifact_sha256": by_name[REWARD_TRACE_FILENAME].sha256,
        "score_receipt_sha256": by_name[SCORE_RECEIPT_FILENAME].sha256,
        "file_count": len(records),
        "total_size_bytes": sum(item.size_bytes for item in records),
        "inventory_sha256": inventory,
        "files": records,
    }
    for name, item in values.items():
        object.__setattr__(partial, name, item)
    projection = _sha256(_canonical_json(_content_projection_body(partial)))
    object.__setattr__(partial, "content_projection_sha256", projection)
    body_sha256 = _sha256(_canonical_json(_metadata_body(partial)))
    object.__setattr__(partial, "metadata_body_sha256", body_sha256)
    return _validate_metadata(partial)


def _payloads_from_bundle(
    closure: _DependencyClosure,
    adapter_bundle: object,
) -> tuple[str, dict[str, bytes]]:
    try:
        validated = closure.validate_bundle(adapter_bundle)
        publication_raw, _body_sha256, address = closure.legacy_manifest(validated)
        role_payloads = closure.legacy_payloads(validated)
    except Exception as exc:
        expected_errors = (
            getattr(
                closure.reward_bundle,
                "ForagerMatchedV3AdapterRewardBundleError",
            ),
            getattr(
                closure.legacy_publication,
                "ForagerMatchedV3AdapterRewardPublicationError",
            ),
        )
        if isinstance(exc, expected_errors):
            raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
                "adapter atomic-v2 input failed its private live bundle conversion"
            ) from exc
        raise
    payloads = {
        PUBLICATION_MANIFEST_FILENAME: publication_raw,
        ADAPTER_BUNDLE_MANIFEST_FILENAME: role_payloads["adapter_bundle_manifest"],
        RUNNER_RESULT_RECEIPT_FILENAME: role_payloads["runner_result_receipt"],
        REWARD_TRACE_FILENAME: role_payloads["reward_trace"],
        SCORE_RECEIPT_FILENAME: role_payloads["score_receipt"],
    }
    _records_from_payloads(payloads)
    return address, payloads


def _atomic_publish_once(
    closure: _DependencyClosure,
    publication_parent: Path,
    *,
    address: str,
    records: tuple[MatchedV3AdapterAtomicPublicationFileV2, ...],
    payloads: dict[str, bytes],
) -> atomic.ContentVerifiedFlatPublication:
    return cast(
        atomic.ContentVerifiedFlatPublication,
        closure.atomic_publish(
            publication_parent,
            address=address,
            expected_files=_atomic_records(records),
            payloads=payloads,
        ),
    )


def _publish_adapter_bundle_with_closure(
    *,
    closure: _DependencyClosure,
    adapter_bundle: object,
    publication_parent: Path,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
    required_pid: int,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    if type(adapter_bundle) is not closure.bundle_type:
        _fail("adapter atomic-v2 private sink requires its exact bundle type")
    if type(required_pid) is not int or required_pid <= 0 or _GETPID_AT_LOAD() != required_pid:
        _fail("adapter atomic-v2 private sink crossed its required PID boundary")
    parent = _preflight_publication_parent(publication_parent)
    environment_seed = _require_uint31(expected_environment_seed, "environment seed")
    agent_seed = _require_uint31(expected_agent_seed, "agent seed")
    source_tree = _require_sha256(
        expected_local_source_tree_sha256, "caller-carried local source tree"
    )
    publisher_source = _require_dependency_closure(closure)
    address, payloads = _payloads_from_bundle(closure, adapter_bundle)
    records = _records_from_payloads(payloads)
    candidate_id = _require_candidate(getattr(adapter_bundle, "candidate_id", None))
    _validate_loaded_payloads(
        closure=closure,
        address=address,
        files=payloads,
        expected_candidate_id=candidate_id,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
    )
    if _GETPID_AT_LOAD() != required_pid:
        _fail("adapter atomic-v2 PID changed before atomic publication")
    try:
        flat = _atomic_publish_once(
            closure,
            parent,
            address=address,
            records=records,
            payloads=dict(payloads),
        )
    except atomic.ForagerMatchedV3AtomicPublicationCollisionError as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2CollisionError(
            f"adapter atomic-v2 publication address {address} already exists; not retried"
        ) from exc
    except atomic.ForagerMatchedV3AtomicPublicationUncertainError as exc:
        raise PublishedAdapterAtomicPublicationV2UncertainError(
            exc.destination,
            exc.address,
            "atomic primitive did not return a content-verified publication",
            committed=exc.committed,
        ) from exc
    except atomic.ForagerMatchedV3AtomicPublicationError as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            "adapter atomic-v2 content publication failed before a committed result"
        ) from exc
    try:
        if _GETPID_AT_LOAD() != required_pid:
            _fail("adapter atomic-v2 PID changed after atomic publication")
        _require_dependency_closure(closure)
        return _metadata_from_flat_publication(
            closure=closure,
            operation="published",
            operation_pid=required_pid,
            publisher_source_sha256=publisher_source,
            local_source_tree_sha256=source_tree,
            flat=flat,
            expected_candidate_id=candidate_id,
            expected_environment_seed=environment_seed,
            expected_agent_seed=agent_seed,
        )
    except BaseException as exc:
        raise PublishedAdapterAtomicPublicationV2UncertainError(
            flat.root,
            address,
            "content committed but final metadata replay failed",
            committed=True,
        ) from exc


def _publish_validated_adapter_bundle(
    *,
    adapter_bundle: object,
    publication_parent: Path,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
    required_pid: int,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    """Private synthetic-test sink; public fused entrypoints do not expose bundles."""

    closure = _load_dependency_closure()
    return _publish_adapter_bundle_with_closure(
        closure=closure,
        adapter_bundle=adapter_bundle,
        publication_parent=publication_parent,
        expected_environment_seed=expected_environment_seed,
        expected_agent_seed=expected_agent_seed,
        expected_local_source_tree_sha256=expected_local_source_tree_sha256,
        required_pid=required_pid,
    )


def _preflight_publication_parent(value: object) -> Path:
    if type(value) is not _PATH_TYPE:
        _fail("adapter atomic-v2 publication parent must be the exact platform Path type")
    parent = value
    if not parent.is_absolute() or parent == Path("/"):
        _fail("adapter atomic-v2 publication parent must be absolute and non-root")
    observed_atomic_source = _read_stable_source_sha256(
        getattr(atomic, "__file__", None),
        _ATOMIC_SOURCE_PATH,
    )
    if (
        not hmac.compare_digest(observed_atomic_source, _ATOMIC_SOURCE_SHA256)
        or getattr(atomic, "ATOMIC_PUBLICATION_DESCRIPTOR_SHA256", None)
        != _ATOMIC_DESCRIPTOR_SHA256
        or getattr(atomic, "_open_parent", None) is not _ATOMIC_OPEN_PARENT_AT_LOAD
        or getattr(atomic, "_close_no_raise", None) is not _ATOMIC_CLOSE_AT_LOAD
        or type(_ATOMIC_OPEN_PARENT_AT_LOAD) is not types.FunctionType
        or type(_ATOMIC_CLOSE_AT_LOAD) is not types.FunctionType
        or _ATOMIC_OPEN_PARENT_AT_LOAD.__defaults__ is not None
        or _ATOMIC_OPEN_PARENT_AT_LOAD.__kwdefaults__ is not None
        or _ATOMIC_CLOSE_AT_LOAD.__defaults__ is not None
        or _ATOMIC_CLOSE_AT_LOAD.__kwdefaults__ is not None
        or not hmac.compare_digest(
            _portable_code_sha256(
                _ATOMIC_OPEN_PARENT_AT_LOAD.__code__,
                _ATOMIC_SOURCE_PATH,
            ),
            _ATOMIC_OPEN_PARENT_CODE_SHA256,
        )
        or not hmac.compare_digest(
            _portable_code_sha256(
                _ATOMIC_CLOSE_AT_LOAD.__code__,
                _ATOMIC_SOURCE_PATH,
            ),
            _ATOMIC_CLOSE_CODE_SHA256,
        )
    ):
        _fail("adapter atomic-v2 atomic parent preflight closure drifted")
    opened: Any | None = None
    try:
        opened = _ATOMIC_OPEN_PARENT_AT_LOAD(parent)
    except atomic.ForagerMatchedV3AtomicPublicationError as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            "adapter atomic-v2 publication parent failed safe atomic preflight"
        ) from exc
    finally:
        if opened is not None:
            _ATOMIC_CLOSE_AT_LOAD(opened.descriptor)
    return parent


def _require_public_entry_inputs(
    *,
    environment_seed: object,
    agent_seed: object,
    publication_parent: object,
    expected_local_source_tree_sha256: object,
    explicit_unqualified_execution: object,
    explicit_publication_opt_in: object,
) -> tuple[int, int, Path, str, int]:
    _require_true(explicit_unqualified_execution, "unqualified full-horizon execution")
    _require_true(explicit_publication_opt_in, "adapter atomic-v2 publication")
    environment = _require_uint31(environment_seed, "environment seed")
    agent = _require_uint31(agent_seed, "agent seed")
    parent = _preflight_publication_parent(publication_parent)
    source_tree = _require_sha256(
        expected_local_source_tree_sha256, "caller-carried local source tree"
    )
    pid = _GETPID_AT_LOAD()
    return environment, agent, parent, source_tree, pid


def run_and_publish_matched_v3_full_rainbow_adapter_v2(
    *,
    environment_seed: int,
    agent_seed: int,
    publication_parent: Path,
    expected_local_source_tree_sha256: str,
    explicit_unqualified_execution: bool,
    explicit_publication_opt_in: bool,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    """Run the exact unqualified Full Rainbow horizon and return metadata only."""

    environment, agent, parent, source_tree, pid = _require_public_entry_inputs(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        publication_parent=publication_parent,
        expected_local_source_tree_sha256=expected_local_source_tree_sha256,
        explicit_unqualified_execution=explicit_unqualified_execution,
        explicit_publication_opt_in=explicit_publication_opt_in,
    )
    closure = _load_dependency_closure()
    if _GETPID_AT_LOAD() != pid:
        _fail("Full Rainbow fused entry crossed its PID boundary before runner entry")
    try:
        result = closure.run_full(
            environment_seed=environment,
            agent_seed=agent,
            unqualified_engineering=True,
        )
        if _GETPID_AT_LOAD() != pid:
            _fail("Full Rainbow result crossed the fused PID boundary")
        adapter_bundle = closure.build_full(result)
        del result
        metadata = _publish_adapter_bundle_with_closure(
            closure=closure,
            adapter_bundle=adapter_bundle,
            publication_parent=parent,
            expected_environment_seed=environment,
            expected_agent_seed=agent,
            expected_local_source_tree_sha256=source_tree,
            required_pid=pid,
        )
        del adapter_bundle
        return metadata
    except ForagerMatchedV3AdapterAtomicPublicationV2Error:
        raise
    except Exception as exc:
        expected_errors = (
            getattr(closure.full_runner, "FullRainbowRunnerContractError"),
            getattr(closure.full_runner, "FullRainbowRunnerExecutionBlockedError"),
            getattr(
                closure.reward_bundle,
                "ForagerMatchedV3AdapterRewardBundleError",
            ),
        )
        if isinstance(exc, expected_errors):
            raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
                "fused unqualified Full Rainbow adapter publication failed"
            ) from exc
        raise


def run_and_publish_matched_v3_ppo_gru_adapter_v2(
    *,
    environment_seed: int,
    agent_seed: int,
    publication_parent: Path,
    expected_local_source_tree_sha256: str,
    explicit_unqualified_execution: bool,
    explicit_publication_opt_in: bool,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    """Run the unqualified PPO-GRU adapter horizon and return metadata only."""

    environment, agent, parent, source_tree, pid = _require_public_entry_inputs(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        publication_parent=publication_parent,
        expected_local_source_tree_sha256=expected_local_source_tree_sha256,
        explicit_unqualified_execution=explicit_unqualified_execution,
        explicit_publication_opt_in=explicit_publication_opt_in,
    )
    closure = _load_dependency_closure()
    if _GETPID_AT_LOAD() != pid:
        _fail("PPO-GRU fused entry crossed its PID boundary before runtime open")
    try:
        runtime = closure.open_ppo()
        if _GETPID_AT_LOAD() != pid:
            _fail("PPO-GRU runtime crossed the fused PID boundary")
        outcome = closure.run_ppo(
            environment_seed=environment,
            agent_seed=agent,
            runtime=runtime,
        )
        if _GETPID_AT_LOAD() != pid:
            _fail("PPO-GRU outcome crossed the fused PID boundary")
        adapter_bundle = closure.build_ppo(outcome)
        del outcome
        metadata = _publish_adapter_bundle_with_closure(
            closure=closure,
            adapter_bundle=adapter_bundle,
            publication_parent=parent,
            expected_environment_seed=environment,
            expected_agent_seed=agent,
            expected_local_source_tree_sha256=source_tree,
            required_pid=pid,
        )
        del adapter_bundle
        del runtime
        return metadata
    except ForagerMatchedV3AdapterAtomicPublicationV2Error:
        raise
    except Exception as exc:
        expected_errors = (
            getattr(closure.ppo_runner, "ForagerMatchedV3PPOGRURunnerError"),
            getattr(
                closure.reward_bundle,
                "ForagerMatchedV3AdapterRewardBundleError",
            ),
        )
        if isinstance(exc, expected_errors):
            raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
                "fused unqualified PPO-GRU adapter publication failed"
            ) from exc
        raise


def reload_matched_v3_adapter_atomic_publication_v2(
    *,
    publication_parent: Path,
    expected_address: str,
    expected_file_records: tuple[MatchedV3AdapterAtomicPublicationFileV2, ...],
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
) -> MatchedV3AdapterAtomicPublicationMetadataV2:
    """Reload one caller-addressed publication and return canonical metadata only."""

    pid = _GETPID_AT_LOAD()
    parent = _preflight_publication_parent(publication_parent)
    address = _require_sha256(expected_address, "expected publication address")
    records = _validate_records(expected_file_records)
    candidate = _require_candidate(expected_candidate_id)
    environment = _require_uint31(expected_environment_seed, "environment seed")
    agent = _require_uint31(expected_agent_seed, "agent seed")
    source_tree = _require_sha256(
        expected_local_source_tree_sha256, "caller-carried local source tree"
    )
    if records[0].sha256 != address:
        _fail("adapter atomic-v2 address differs from its publication record")
    closure = _load_dependency_closure()
    publisher_source = _require_dependency_closure(closure)
    if _GETPID_AT_LOAD() != pid:
        _fail("adapter atomic-v2 reload crossed its PID boundary before content load")
    try:
        flat = closure.atomic_load(
            parent,
            address=address,
            expected_files=_atomic_records(records),
        )
    except atomic.ForagerMatchedV3AtomicPublicationError as exc:
        raise ForagerMatchedV3AdapterAtomicPublicationV2Error(
            "adapter atomic-v2 content reload failed"
        ) from exc
    if _GETPID_AT_LOAD() != pid:
        _fail("adapter atomic-v2 reload crossed its PID boundary")
    _require_dependency_closure(closure)
    return _metadata_from_flat_publication(
        closure=closure,
        operation="reloaded",
        operation_pid=pid,
        publisher_source_sha256=publisher_source,
        local_source_tree_sha256=source_tree,
        flat=flat,
        expected_candidate_id=candidate,
        expected_environment_seed=environment,
        expected_agent_seed=agent,
    )


__all__ = [
    "ADAPTER_ATOMIC_PUBLICATION_V2_CANDIDATE_IDS",
    "ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256",
    "ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES",
    "ADAPTER_ATOMIC_PUBLICATION_V2_METADATA_SCHEMA_VERSION",
    "ADAPTER_ATOMIC_PUBLICATION_V2_STATUS",
    "ForagerMatchedV3AdapterAtomicPublicationV2Error",
    "ForagerMatchedV3AdapterAtomicPublicationV2CollisionError",
    "MatchedV3AdapterAtomicPublicationFileV2",
    "MatchedV3AdapterAtomicPublicationMetadataV2",
    "PublishedAdapterAtomicPublicationV2UncertainError",
    "adapter_atomic_publication_v2_descriptor",
    "canonical_adapter_atomic_publication_v2_descriptor_bytes",
    "canonical_adapter_atomic_publication_v2_metadata_bytes",
    "parse_adapter_atomic_publication_v2_descriptor",
    "parse_adapter_atomic_publication_v2_metadata",
    "reload_matched_v3_adapter_atomic_publication_v2",
    "run_and_publish_matched_v3_full_rainbow_adapter_v2",
    "run_and_publish_matched_v3_ppo_gru_adapter_v2",
]
