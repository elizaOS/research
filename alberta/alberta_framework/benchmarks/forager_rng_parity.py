"""Fail-closed fixed-action RNG parity probe for the current Forager FOV task.

The probe compares the exact audited ``continual-foragax-agents`` wrapper with
Alberta's direct Foragax call shape under one explicit action sequence.  It
never trains an agent and never emits raw observations, rewards, state, or
``info`` values.  Those values are compared through canonical tree hashes.

The result's self-hash is content identity only.  It is not an attestation,
does not authenticate the OCI executor, and cannot authorize promotion.  A
caller must independently pin and verify the required OCI image and probe
source before relying on a replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

PARITY_RESULT_SCHEMA_VERSION: Final = "alberta.forager_fixed_action_rng_parity.v1"
TRACE_SCHEMA_VERSION: Final = "alberta.forager_fixed_action_environment_trace.v1"
ACTION_SEQUENCE_SCHEMA_VERSION: Final = "alberta.forager_fixed_action_sequence.v1"
TASK_SCHEMA_VERSION: Final = "alberta.forager_fov_task_identity.v1"
RNG_CONTRACT_SCHEMA_VERSION: Final = "alberta.forager_environment_rng_contract.v1"

CONTENT_IDENTITY_BOUNDARY: Final = (
    "content_identity_only_external_image_and_source_verification_required"
)
MATCH_STATUS: Final = "exact_fixed_action_parity_match"

REQUIRED_OCI_IMAGE_ID: Final = (
    "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
)
REQUIRED_BUILD_ATTESTATION_PATH: Final = Path("/opt/alberta-attestations/build-attestation.json")
REQUIRED_BUILD_ATTESTATION_SHA256: Final = (
    "94e0e164191ba872bcc514cd3167741afa36f3b9d076a8844d391b370dbb9ca5"
)
REQUIRED_BUILD_ATTESTATION_SCHEMA: Final = "alberta.official_foragax.oci_build.v4"
REQUIRED_SOURCE_REPOSITORY: Final = "https://github.com/steventango/continual-foragax-agents"
REQUIRED_SOURCE_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
REQUIRED_SOURCE_TREE_GIT_SHA1: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
REQUIRED_SOURCE_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256: Final = (
    "fcab40b01123250e837d9feb222d1c086303192dd24b806ab8cb8405cd7300d9"
)
REQUIRED_DEPENDENCY_LOCK_SHA256: Final = (
    "46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd"
)
REQUIRED_SOURCE_ROOT: Final = Path("/opt/continual-foragax-agents")
REQUIRED_SOURCE_SRC_ROOT: Final = REQUIRED_SOURCE_ROOT / "src"
REQUIRED_WRAPPER_PATH: Final = REQUIRED_SOURCE_SRC_ROOT / "environments/Foragax.py"
REQUIRED_WRAPPER_SHA256: Final = "91c4e34ee3d477f52bedafb5526785ea138ae2b187df0f1d720a805927ab67dc"

REQUIRED_FORAGAX_DISTRIBUTION: Final = "continual-foragax"
REQUIRED_FORAGAX_VERSION: Final = "0.55.0"
REQUIRED_FORAGAX_WHEEL_SHA256: Final = (
    "79b20f234d651feed2736873192fa6e3b224bce9bf6e9674f1ed52a227b073d2"
)
REQUIRED_FORAGAX_INSTALL_TREE_SHA256: Final = (
    "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
)
REQUIRED_FORAGAX_INSTALL_TREE_HASH_SCHEME: Final = "relative-path+size+bytes-v1"

REQUIRED_PYTHON_VERSION: Final = "3.12.3"
REQUIRED_PYTHON_EXECUTABLE: Final = Path("/opt/alberta-runtime/bin/python")
REQUIRED_PYTHON_EXECUTABLE_SHA256: Final = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
)
REQUIRED_JAX_VERSION: Final = "0.9.0.1"
REQUIRED_JAXLIB_VERSION: Final = "0.9.0.1"
REQUIRED_PRNG_IMPL: Final = "threefry2x32"
REQUIRED_BACKEND: Final = "cpu"
REQUIRED_SOURCE_MOUNT_MODE: Final = "read_only_content_addressed_oci_layer"
REQUIRED_ENVIRONMENT_RNG_SCHEDULE: Final = "dedicated_environment_split_chain_v1"

MAX_SEED: Final = 2**31 - 1
MAX_ACTIONS: Final = 1_024
MAX_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_JSON_NODES: Final = 100_000
MAX_JSON_DEPTH: Final = 64
MAX_SOURCE_BYTES: Final = 8 * 1024 * 1024

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_KEY_WORD_MAX: Final = 2**32 - 1

_TASK_BASE: Final = MappingProxyType(
    {
        "schema_version": TASK_SCHEMA_VERSION,
        "preset": "field_of_view",
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "aperture_size": 9,
        "observation_type": "color",
        "reward_delay": 0,
        "random_shift_max_steps": 0,
        "extra_kwargs": {},
        "foragax_distribution": REQUIRED_FORAGAX_DISTRIBUTION,
        "foragax_version": REQUIRED_FORAGAX_VERSION,
    }
)

_RNG_CONTRACT_BASE: Final = MappingProxyType(
    {
        "schema_version": RNG_CONTRACT_SCHEMA_VERSION,
        "identity": REQUIRED_ENVIRONMENT_RNG_SCHEDULE,
        "prng_impl": REQUIRED_PRNG_IMPL,
        "jax_threefry_partitionable": True,
        "backend": REQUIRED_BACKEND,
        "root": "jax.random.key(seed, impl=threefry2x32)",
        "reset": "next_key, reset_key = jax.random.split(input_key)",
        "transition": "next_key, step_key = jax.random.split(input_key)",
        "reset_during_trace": False,
        "agent_rng_present": False,
    }
)


class ForagerRngParityError(ValueError):
    """The fixed-action parity contract or an executor trace is invalid."""


class ForagerRngParityMismatchError(ForagerRngParityError):
    """The exact wrapper and direct environment traces differ."""


@dataclass(frozen=True)
class FixedActionProbeConfig:
    """One explicit seed and immutable fixed-action sequence."""

    seed: int
    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ForagerRngParityError("seed must be an integer")
        if not 0 <= self.seed <= MAX_SEED:
            raise ForagerRngParityError(f"seed must lie in [0, {MAX_SEED}]")
        if not isinstance(self.actions, tuple):
            raise ForagerRngParityError("actions must be an immutable tuple")
        if not self.actions or len(self.actions) > MAX_ACTIONS:
            raise ForagerRngParityError(f"actions must contain between 1 and {MAX_ACTIONS} entries")
        for index, action in enumerate(self.actions):
            if isinstance(action, bool) or not isinstance(action, int):
                raise ForagerRngParityError(f"actions[{index}] must be an integer")
            if not 0 <= action < 4:
                raise ForagerRngParityError(f"actions[{index}] must lie in [0, 3]")

    @property
    def action_sequence_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": ACTION_SEQUENCE_SCHEMA_VERSION,
                "actions": list(self.actions),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "actions": list(self.actions),
            "action_count": len(self.actions),
            "action_sequence_sha256": self.action_sequence_sha256,
        }


@dataclass(frozen=True)
class KeyFrame:
    """One exact input/carry/environment-key split frame."""

    input_key: tuple[int, int]
    next_key: tuple[int, int]
    environment_key: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_key": list(self.input_key),
            "next_key": list(self.next_key),
            "environment_key": list(self.environment_key),
        }


@dataclass(frozen=True)
class TreeDigest:
    """Structure and exact-value hashes for one finite numeric pytree."""

    leaf_count: int
    structure_sha256: str
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_count": self.leaf_count,
            "structure_sha256": self.structure_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class RawResetRecord:
    """Nonserializable in-memory reset record; values are never emitted."""

    keys: KeyFrame
    observation: Any
    state: Any


@dataclass(frozen=True)
class RawTransitionRecord:
    """Nonserializable in-memory transition record; values are never emitted."""

    index: int
    action: int
    keys: KeyFrame
    observation: Any
    reward: Any
    done: Any
    info: Any
    state: Any


@dataclass(frozen=True)
class RawEnvironmentTrace:
    """One executor's raw in-memory trace before irreversible hashing."""

    reset: RawResetRecord
    transitions: tuple[RawTransitionRecord, ...]


@dataclass(frozen=True)
class TransitionDigest:
    """Hash-only normalized transition."""

    index: int
    action: int
    keys: KeyFrame
    observation: TreeDigest
    reward: TreeDigest
    done: TreeDigest
    info: TreeDigest
    state: TreeDigest

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "keys": self.keys.to_dict(),
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
            "done": self.done.to_dict(),
            "info": self.info.to_dict(),
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class EnvironmentTraceDigest:
    """Canonical hash-only reset and fixed-action transition trace."""

    seed: int
    action_sequence_sha256: str
    reset_keys: KeyFrame
    reset_observation: TreeDigest
    reset_state: TreeDigest
    transitions: tuple[TransitionDigest, ...]
    trace_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "seed": self.seed,
            "action_sequence_sha256": self.action_sequence_sha256,
            "reset": {
                "keys": self.reset_keys.to_dict(),
                "observation": self.reset_observation.to_dict(),
                "state": self.reset_state.to_dict(),
            },
            "transitions": [item.to_dict() for item in self.transitions],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["trace_sha256"] = self.trace_sha256
        return payload


@dataclass(frozen=True)
class VerifiedRuntimeIdentity:
    """Exact identities observed in the read-only qualified runtime."""

    required_oci_image_id: str
    build_attestation_sha256: str
    source_repository: str
    source_commit: str
    source_tree_git_sha1: str
    source_archive_sha256: str
    source_archive_inventory_sha256: str
    dependency_lock_sha256: str
    wrapper_source_path: str
    wrapper_source_sha256: str
    source_mount_mode: str
    foragax_distribution: str
    foragax_version: str
    foragax_wheel_sha256: str
    foragax_install_tree_hash_scheme: str
    foragax_install_tree_sha256: str
    python_version: str
    python_executable_sha256: str
    jax_version: str
    jaxlib_version: str
    backend: str
    cpu_device_count: int
    prng_impl: str
    threefry_partitionable: bool
    jax_enable_x64: bool
    probe_module_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_oci_image_id": self.required_oci_image_id,
            "build_attestation_sha256": self.build_attestation_sha256,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "source_tree_git_sha1": self.source_tree_git_sha1,
            "source_archive_sha256": self.source_archive_sha256,
            "source_archive_inventory_sha256": self.source_archive_inventory_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "wrapper_source_path": self.wrapper_source_path,
            "wrapper_source_sha256": self.wrapper_source_sha256,
            "source_mount_mode": self.source_mount_mode,
            "foragax_distribution": self.foragax_distribution,
            "foragax_version": self.foragax_version,
            "foragax_wheel_sha256": self.foragax_wheel_sha256,
            "foragax_install_tree_hash_scheme": self.foragax_install_tree_hash_scheme,
            "foragax_install_tree_sha256": self.foragax_install_tree_sha256,
            "python_version": self.python_version,
            "python_executable_sha256": self.python_executable_sha256,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "backend": self.backend,
            "cpu_device_count": self.cpu_device_count,
            "prng_impl": self.prng_impl,
            "threefry_partitionable": self.threefry_partitionable,
            "jax_enable_x64": self.jax_enable_x64,
            "probe_module_sha256": self.probe_module_sha256,
        }


@dataclass(frozen=True)
class ParityProbeResult:
    """Canonical matched result.  Its self-hash is not a trust assertion."""

    runtime: VerifiedRuntimeIdentity
    config: FixedActionProbeConfig
    matched_trace: EnvironmentTraceDigest
    wrapper_trace_sha256: str
    direct_trace_sha256: str
    payload_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PARITY_RESULT_SCHEMA_VERSION,
            "status": MATCH_STATUS,
            "evidence_boundary": CONTENT_IDENTITY_BOUNDARY,
            "promotion_authorized": False,
            "runtime": self.runtime.to_dict(),
            "task": task_descriptor(),
            "rng_contract": rng_contract_descriptor(),
            "probe": self.config.to_dict(),
            "matched_trace": self.matched_trace.to_dict(),
            "wrapper_trace_sha256": self.wrapper_trace_sha256,
            "direct_trace_sha256": self.direct_trace_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _duplicate_free_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerRngParityError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerRngParityError(f"non-finite JSON number {token!r} is forbidden")


def _parse_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ForagerRngParityError(f"non-finite JSON number {token!r} is forbidden")
    return value


def _validate_json_value(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ForagerRngParityError("JSON value exceeds the node limit")
        if depth > MAX_JSON_DEPTH:
            raise ForagerRngParityError("JSON value exceeds the nesting limit")
        if isinstance(item, Mapping):
            for key in item:
                if type(key) is not str:
                    raise ForagerRngParityError("JSON object keys must be strings")
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ForagerRngParityError(
                        "JSON object keys must contain valid Unicode"
                    ) from exc
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ForagerRngParityError("JSON strings must contain valid Unicode") from exc
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerRngParityError("non-finite JSON numbers are forbidden")
        elif item is not None and type(item) not in (bool, int):
            raise ForagerRngParityError(f"value contains non-JSON type {type(item).__name__}")


def decode_strict_json(data: bytes | str) -> Any:
    """Decode duplicate-free finite UTF-8 JSON under fixed resource limits."""
    try:
        if isinstance(data, bytes):
            if len(data) > MAX_JSON_BYTES:
                raise ForagerRngParityError("JSON input exceeds the byte limit")
            text = data.decode("utf-8")
        else:
            if len(data) > MAX_JSON_BYTES or len(data.encode("utf-8")) > MAX_JSON_BYTES:
                raise ForagerRngParityError("JSON input exceeds the byte limit")
            text = data
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_free_object,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_json_float,
        )
    except ForagerRngParityError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ForagerRngParityError(f"input is not strict UTF-8 JSON: {exc}") from exc
    _validate_json_value(value)
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return compact deterministic UTF-8 JSON bytes without a newline."""
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ForagerRngParityError(f"value is not canonical JSON data: {exc}") from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def task_descriptor() -> dict[str, Any]:
    """Return the exact immutable FOV task identity and its content hash."""
    payload = dict(_TASK_BASE)
    payload["extra_kwargs"] = {}
    payload["task_sha256"] = _canonical_sha256(dict(_TASK_BASE))
    return payload


def rng_contract_descriptor() -> dict[str, Any]:
    """Return the exact dedicated split-chain contract and its content hash."""
    payload = dict(_RNG_CONTRACT_BASE)
    payload["rng_contract_sha256"] = _canonical_sha256(dict(_RNG_CONTRACT_BASE))
    return payload


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ForagerRngParityError(f"{path} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ForagerRngParityError(f"{path} has unknown keys: {', '.join(unknown)}")


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ForagerRngParityError(f"{path} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _require_array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerRngParityError(f"{path} must be a JSON array")
    return value


def _require_string(value: Any, path: str) -> str:
    if type(value) is not str or not value or len(value) > 1_024:
        raise ForagerRngParityError(f"{path} must be a non-empty bounded string")
    return value


def _require_sha256(value: Any, path: str) -> str:
    result = _require_string(value, path)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ForagerRngParityError(f"{path} must be a lowercase SHA-256")
    return result


def _require_int(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForagerRngParityError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise ForagerRngParityError(f"{path} must lie in [{minimum}, {maximum}]")
    return value


def _key_words(key: Any) -> tuple[int, int]:
    _, _, jr = _jax_modules()
    words = np.asarray(jr.key_data(key))
    if words.shape != (2,) or words.dtype != np.dtype(np.uint32):
        raise ForagerRngParityError("environment key must contain two uint32 words")
    return (int(words[0]), int(words[1]))


def _jax_modules() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
    except ImportError as exc:  # pragma: no cover - runtime packaging failure
        raise ForagerRngParityError("JAX is required for the parity probe") from exc
    return jax, jnp, jr


def expected_key_schedule(
    config: FixedActionProbeConfig,
) -> tuple[KeyFrame, tuple[KeyFrame, ...]]:
    """Derive the exact partitionable-threefry reset and transition key frames."""
    jax, _, jr = _jax_modules()
    with jax.threefry_partitionable(True):
        key = jr.key(config.seed, impl=REQUIRED_PRNG_IMPL)

        def split_frame(current: Any) -> tuple[Any, KeyFrame]:
            next_key, environment_key = jr.split(current)
            return next_key, KeyFrame(
                input_key=_key_words(current),
                next_key=_key_words(next_key),
                environment_key=_key_words(environment_key),
            )

        key, reset = split_frame(key)
        transitions: list[KeyFrame] = []
        for _ in config.actions:
            key, frame = split_frame(key)
            transitions.append(frame)
    return reset, tuple(transitions)


def _path_component(value: Any) -> list[Any]:
    name = type(value).__name__
    if name == "DictKey":
        key = value.key
        if type(key) not in (str, int):
            raise ForagerRngParityError("pytree dictionary keys must be strings or integers")
        return ["dict", key]
    if name == "SequenceKey":
        return [
            "sequence",
            _require_int(
                value.idx,
                "pytree sequence index",
                minimum=0,
                maximum=2**31 - 1,
            ),
        ]
    if name == "GetAttrKey":
        return ["attribute", _require_string(value.name, "pytree attribute name")]
    if name == "FlattenedIndexKey":
        return [
            "flattened_index",
            _require_int(value.key, "pytree flattened index", minimum=0, maximum=2**31 - 1),
        ]
    raise ForagerRngParityError(f"unsupported pytree path entry {name!r}")


def _canonical_array(leaf: Any, path: str) -> tuple[dict[str, Any], str]:
    try:
        array = np.asarray(leaf)
    except (TypeError, ValueError) as exc:
        raise ForagerRngParityError(f"{path} is not an array-like numeric leaf") from exc
    if array.dtype.kind not in ("b", "i", "u", "f"):
        raise ForagerRngParityError(f"{path} has unsupported dtype {array.dtype}")
    if array.dtype.kind == "f" and not bool(np.all(np.isfinite(array))):
        raise ForagerRngParityError(f"{path} contains non-finite values")
    canonical_dtype = array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    contents = canonical.tobytes(order="C")
    descriptor = {
        "dtype": canonical.dtype.str,
        "shape": [int(size) for size in canonical.shape],
    }
    return descriptor, hashlib.sha256(contents).hexdigest()


def fingerprint_pytree(value: Any, *, label: str) -> TreeDigest:
    """Hash a finite numeric pytree without retaining or returning raw values."""
    jax, _, _ = _jax_modules()
    try:
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    except (TypeError, ValueError) as exc:
        raise ForagerRngParityError(f"{label} is not a supported JAX pytree") from exc
    if not path_leaves:
        raise ForagerRngParityError(f"{label} must contain at least one leaf")
    treedef_text = str(treedef)
    if not treedef_text or len(treedef_text) > 32_768 or "0x" in treedef_text:
        raise ForagerRngParityError(f"{label} has a noncanonical pytree definition")
    structure_leaves: list[dict[str, Any]] = []
    content_hashes: list[str] = []
    for index, (raw_path, leaf) in enumerate(path_leaves):
        path = [_path_component(component) for component in raw_path]
        descriptor, content_sha256 = _canonical_array(leaf, f"{label}.leaves[{index}]")
        structure_leaves.append({"path": path, **descriptor})
        content_hashes.append(content_sha256)
    structure = {
        "treedef": treedef_text,
        "leaves": structure_leaves,
    }
    structure_sha256 = _canonical_sha256(structure)
    content_sha256 = _canonical_sha256(
        {
            "structure_sha256": structure_sha256,
            "leaf_content_sha256": content_hashes,
        }
    )
    return TreeDigest(
        leaf_count=len(path_leaves),
        structure_sha256=structure_sha256,
        content_sha256=content_sha256,
    )


def _validate_key_frame(value: KeyFrame, expected: KeyFrame, path: str) -> None:
    for field_name, words in (
        ("input_key", value.input_key),
        ("next_key", value.next_key),
        ("environment_key", value.environment_key),
    ):
        if (
            not isinstance(words, tuple)
            or len(words) != 2
            or any(
                isinstance(word, bool)
                or not isinstance(word, int)
                or not 0 <= word <= _KEY_WORD_MAX
                for word in words
            )
        ):
            raise ForagerRngParityError(f"{path}.{field_name} must contain two uint32 words")
    if value != expected:
        raise ForagerRngParityError(f"{path} does not match the exact split-chain schedule")


def digest_environment_trace(
    config: FixedActionProbeConfig,
    trace: RawEnvironmentTrace,
    *,
    runner_label: str,
) -> EnvironmentTraceDigest:
    """Validate one executor trace against the fixed key/action schedule and hash it."""
    if not isinstance(trace, RawEnvironmentTrace):
        raise ForagerRngParityError(f"{runner_label} did not return a RawEnvironmentTrace")
    if not isinstance(trace.transitions, tuple):
        raise ForagerRngParityError(f"{runner_label} transitions must be an immutable tuple")
    if len(trace.transitions) != len(config.actions):
        raise ForagerRngParityError(
            f"{runner_label} transition count does not match the action sequence"
        )
    expected_reset, expected_transitions = expected_key_schedule(config)
    _validate_key_frame(trace.reset.keys, expected_reset, f"{runner_label}.reset.keys")
    reset_observation = fingerprint_pytree(
        trace.reset.observation,
        label=f"{runner_label}.reset.observation",
    )
    reset_state = fingerprint_pytree(
        trace.reset.state,
        label=f"{runner_label}.reset.state",
    )
    transitions: list[TransitionDigest] = []
    for index, (raw, action, expected_keys) in enumerate(
        zip(trace.transitions, config.actions, expected_transitions, strict=True)
    ):
        path = f"{runner_label}.transitions[{index}]"
        if raw.index != index:
            raise ForagerRngParityError(f"{path}.index is not exact")
        if raw.action != action:
            raise ForagerRngParityError(f"{path}.action does not match the fixed sequence")
        _validate_key_frame(raw.keys, expected_keys, f"{path}.keys")
        transitions.append(
            TransitionDigest(
                index=index,
                action=action,
                keys=raw.keys,
                observation=fingerprint_pytree(
                    raw.observation,
                    label=f"{path}.observation",
                ),
                reward=fingerprint_pytree(raw.reward, label=f"{path}.reward"),
                done=fingerprint_pytree(raw.done, label=f"{path}.done"),
                info=fingerprint_pytree(raw.info, label=f"{path}.info"),
                state=fingerprint_pytree(raw.state, label=f"{path}.state"),
            )
        )
    draft = EnvironmentTraceDigest(
        seed=config.seed,
        action_sequence_sha256=config.action_sequence_sha256,
        reset_keys=trace.reset.keys,
        reset_observation=reset_observation,
        reset_state=reset_state,
        transitions=tuple(transitions),
        trace_sha256="",
    )
    return replace(draft, trace_sha256=_canonical_sha256(draft.unsigned_dict()))


def _first_trace_mismatch(
    wrapper: EnvironmentTraceDigest,
    direct: EnvironmentTraceDigest,
) -> str | None:
    if wrapper.seed != direct.seed:
        return "seed"
    if wrapper.action_sequence_sha256 != direct.action_sequence_sha256:
        return "action_sequence_sha256"
    if wrapper.reset_keys != direct.reset_keys:
        return "reset.keys"
    if wrapper.reset_observation != direct.reset_observation:
        return "reset.observation"
    if wrapper.reset_state != direct.reset_state:
        return "reset.state"
    if len(wrapper.transitions) != len(direct.transitions):
        return "transitions.length"
    for index, (left, right) in enumerate(
        zip(wrapper.transitions, direct.transitions, strict=True)
    ):
        for field_name in (
            "index",
            "action",
            "keys",
            "observation",
            "reward",
            "done",
            "info",
            "state",
        ):
            if getattr(left, field_name) != getattr(right, field_name):
                return f"transitions[{index}].{field_name}"
    if wrapper.trace_sha256 != direct.trace_sha256:
        return "trace_sha256"
    return None


def _validate_runtime_identity(identity: VerifiedRuntimeIdentity) -> None:
    if type(identity.threefry_partitionable) is not bool:
        raise ForagerRngParityError("runtime.threefry_partitionable must be boolean")
    if type(identity.jax_enable_x64) is not bool:
        raise ForagerRngParityError("runtime.jax_enable_x64 must be boolean")
    expected: dict[str, Any] = {
        "required_oci_image_id": REQUIRED_OCI_IMAGE_ID,
        "build_attestation_sha256": REQUIRED_BUILD_ATTESTATION_SHA256,
        "source_repository": REQUIRED_SOURCE_REPOSITORY,
        "source_commit": REQUIRED_SOURCE_COMMIT,
        "source_tree_git_sha1": REQUIRED_SOURCE_TREE_GIT_SHA1,
        "source_archive_sha256": REQUIRED_SOURCE_ARCHIVE_SHA256,
        "source_archive_inventory_sha256": REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256,
        "dependency_lock_sha256": REQUIRED_DEPENDENCY_LOCK_SHA256,
        "wrapper_source_path": REQUIRED_WRAPPER_PATH.as_posix(),
        "wrapper_source_sha256": REQUIRED_WRAPPER_SHA256,
        "source_mount_mode": REQUIRED_SOURCE_MOUNT_MODE,
        "foragax_distribution": REQUIRED_FORAGAX_DISTRIBUTION,
        "foragax_version": REQUIRED_FORAGAX_VERSION,
        "foragax_wheel_sha256": REQUIRED_FORAGAX_WHEEL_SHA256,
        "foragax_install_tree_hash_scheme": REQUIRED_FORAGAX_INSTALL_TREE_HASH_SCHEME,
        "foragax_install_tree_sha256": REQUIRED_FORAGAX_INSTALL_TREE_SHA256,
        "python_version": REQUIRED_PYTHON_VERSION,
        "python_executable_sha256": REQUIRED_PYTHON_EXECUTABLE_SHA256,
        "jax_version": REQUIRED_JAX_VERSION,
        "jaxlib_version": REQUIRED_JAXLIB_VERSION,
        "backend": REQUIRED_BACKEND,
        "prng_impl": REQUIRED_PRNG_IMPL,
        "threefry_partitionable": True,
        "jax_enable_x64": False,
    }
    actual = identity.to_dict()
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ForagerRngParityError(
            "runtime identity differs from the qualified lock: " + ", ".join(mismatches)
        )
    _require_int(
        identity.cpu_device_count,
        "runtime.cpu_device_count",
        minimum=1,
        maximum=256,
    )
    _require_sha256(identity.probe_module_sha256, "runtime.probe_module_sha256")


def compare_fixed_action_traces(
    config: FixedActionProbeConfig,
    wrapper_trace: RawEnvironmentTrace,
    direct_trace: RawEnvironmentTrace,
    runtime: VerifiedRuntimeIdentity,
) -> ParityProbeResult:
    """Hash, compare, and return a result only when every component is exact."""
    _validate_runtime_identity(runtime)
    wrapper = digest_environment_trace(config, wrapper_trace, runner_label="wrapper")
    direct = digest_environment_trace(config, direct_trace, runner_label="direct")
    mismatch = _first_trace_mismatch(wrapper, direct)
    if mismatch is not None:
        raise ForagerRngParityMismatchError(
            f"fixed-action wrapper/direct parity mismatch at {mismatch}"
        )
    draft = ParityProbeResult(
        runtime=runtime,
        config=config,
        matched_trace=wrapper,
        wrapper_trace_sha256=wrapper.trace_sha256,
        direct_trace_sha256=direct.trace_sha256,
        payload_sha256="",
    )
    return replace(draft, payload_sha256=_canonical_sha256(draft.unsigned_dict()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exact_read_only_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ForagerRngParityError(f"could not stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ForagerRngParityError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > MAX_SOURCE_BYTES:
        raise ForagerRngParityError(f"{label} exceeds the source-size limit")
    try:
        if not bool(os.statvfs(path).f_flag & os.ST_RDONLY):
            raise ForagerRngParityError(f"{label} is not on a read-only mount")
        contents = path.read_bytes()
    except ForagerRngParityError:
        raise
    except OSError as exc:
        raise ForagerRngParityError(f"could not read {label}: {exc}") from exc
    if len(contents) != metadata.st_size:
        raise ForagerRngParityError(f"{label} changed while it was read")
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ForagerRngParityError(f"{label} SHA-256 differs from the qualified lock")
    return contents


def _foragax_install_tree_sha256() -> str:
    spec = importlib.util.find_spec("foragax")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise ForagerRngParityError("the qualified foragax package is not importable")
    files: list[tuple[str, Path]] = []
    for raw_root in locations:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            continue
        files.extend(
            (f"foragax/{path.relative_to(root).as_posix()}", path)
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    if not files:
        raise ForagerRngParityError("the qualified foragax package tree is empty")
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        encoded_path = relative.encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _read_probe_module_identity() -> str:
    path = Path(__file__)
    try:
        metadata = path.lstat()
        read_only = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError as exc:
        raise ForagerRngParityError(f"could not inspect probe module source: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ForagerRngParityError("probe module source must be a regular non-symlink file")
    if not read_only:
        raise ForagerRngParityError("probe module source is not on a read-only mount")
    if metadata.st_size > MAX_SOURCE_BYTES:
        raise ForagerRngParityError("probe module source exceeds the size limit")
    return _file_sha256(path)


def collect_verified_runtime_identity() -> VerifiedRuntimeIdentity:
    """Verify the qualified image contents visible to this process.

    The OCI image ID remains a required external identity.  An in-container
    process cannot authenticate its own image, which is why the returned
    result explicitly retains :data:`CONTENT_IDENTITY_BOUNDARY`.
    """
    attestation_bytes = _verify_exact_read_only_file(
        REQUIRED_BUILD_ATTESTATION_PATH,
        REQUIRED_BUILD_ATTESTATION_SHA256,
        label="build attestation",
    )
    _verify_exact_read_only_file(
        REQUIRED_WRAPPER_PATH,
        REQUIRED_WRAPPER_SHA256,
        label="upstream Foragax wrapper source",
    )
    try:
        if not bool(os.statvfs(REQUIRED_SOURCE_ROOT).f_flag & os.ST_RDONLY):
            raise ForagerRngParityError("upstream source root is not read-only")
    except OSError as exc:
        raise ForagerRngParityError(f"could not inspect upstream source root: {exc}") from exc

    attestation = _require_object(
        decode_strict_json(attestation_bytes),
        "build_attestation",
    )
    required_attestation_values = {
        "schema_version": REQUIRED_BUILD_ATTESTATION_SCHEMA,
        "source_archive_sha256": REQUIRED_SOURCE_ARCHIVE_SHA256,
        "source_archive_inventory_sha256": REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256,
        "source_commit": REQUIRED_SOURCE_COMMIT,
        "source_tree_git_sha1": REQUIRED_SOURCE_TREE_GIT_SHA1,
        "dependency_lock_sha256": REQUIRED_DEPENDENCY_LOCK_SHA256,
        "python_executable_sha256": REQUIRED_PYTHON_EXECUTABLE_SHA256,
    }
    drifted = [
        key
        for key, expected in required_attestation_values.items()
        if attestation.get(key) != expected
    ]
    if drifted:
        raise ForagerRngParityError(
            "build attestation differs from the qualified lock: " + ", ".join(drifted)
        )

    try:
        foragax_version = importlib.metadata.version(REQUIRED_FORAGAX_DISTRIBUTION)
        jaxlib_version = importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ForagerRngParityError("qualified scientific distributions are missing") from exc
    if foragax_version != REQUIRED_FORAGAX_VERSION:
        raise ForagerRngParityError("continual-foragax version differs from the lock")
    install_tree = _foragax_install_tree_sha256()
    if install_tree != REQUIRED_FORAGAX_INSTALL_TREE_SHA256:
        raise ForagerRngParityError("continual-foragax install tree differs from the lock")

    if Path(sys.executable) != REQUIRED_PYTHON_EXECUTABLE:
        raise ForagerRngParityError("probe is not running under the qualified interpreter")
    _verify_exact_read_only_file(
        REQUIRED_PYTHON_EXECUTABLE,
        REQUIRED_PYTHON_EXECUTABLE_SHA256,
        label="qualified Python interpreter",
    )
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    if python_version != REQUIRED_PYTHON_VERSION:
        raise ForagerRngParityError("Python version differs from the qualified lock")

    jax, _, jr = _jax_modules()
    backend = str(jax.default_backend())
    devices = tuple(jax.devices())
    if (
        backend != REQUIRED_BACKEND
        or not devices
        or any(str(device.platform) != REQUIRED_BACKEND for device in devices)
    ):
        raise ForagerRngParityError("probe must run exclusively on JAX CPU devices")
    prng_impl = str(jax.config.jax_default_prng_impl)
    key_impl = str(jr.key_impl(jr.key(0)))
    partitionable = bool(jax.config.jax_threefry_partitionable)
    enable_x64 = bool(jax.config.jax_enable_x64)
    if (
        str(jax.__version__) != REQUIRED_JAX_VERSION
        or jaxlib_version != REQUIRED_JAXLIB_VERSION
        or prng_impl != REQUIRED_PRNG_IMPL
        or key_impl != REQUIRED_PRNG_IMPL
        or not partitionable
        or enable_x64
    ):
        raise ForagerRngParityError("JAX CPU/threefry runtime differs from the lock")

    identity = VerifiedRuntimeIdentity(
        required_oci_image_id=REQUIRED_OCI_IMAGE_ID,
        build_attestation_sha256=REQUIRED_BUILD_ATTESTATION_SHA256,
        source_repository=REQUIRED_SOURCE_REPOSITORY,
        source_commit=REQUIRED_SOURCE_COMMIT,
        source_tree_git_sha1=REQUIRED_SOURCE_TREE_GIT_SHA1,
        source_archive_sha256=REQUIRED_SOURCE_ARCHIVE_SHA256,
        source_archive_inventory_sha256=REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256,
        dependency_lock_sha256=REQUIRED_DEPENDENCY_LOCK_SHA256,
        wrapper_source_path=REQUIRED_WRAPPER_PATH.as_posix(),
        wrapper_source_sha256=REQUIRED_WRAPPER_SHA256,
        source_mount_mode=REQUIRED_SOURCE_MOUNT_MODE,
        foragax_distribution=REQUIRED_FORAGAX_DISTRIBUTION,
        foragax_version=foragax_version,
        foragax_wheel_sha256=REQUIRED_FORAGAX_WHEEL_SHA256,
        foragax_install_tree_hash_scheme=REQUIRED_FORAGAX_INSTALL_TREE_HASH_SCHEME,
        foragax_install_tree_sha256=install_tree,
        python_version=python_version,
        python_executable_sha256=REQUIRED_PYTHON_EXECUTABLE_SHA256,
        jax_version=str(jax.__version__),
        jaxlib_version=jaxlib_version,
        backend=backend,
        cpu_device_count=len(devices),
        prng_impl=prng_impl,
        threefry_partitionable=partitionable,
        jax_enable_x64=enable_x64,
        probe_module_sha256=_read_probe_module_identity(),
    )
    _validate_runtime_identity(identity)
    return identity


def _task_kwargs() -> dict[str, Any]:
    return {
        "env_id": "ForagaxTwoBiomeLarge-v1",
        "aperture_size": 9,
        "observation_type": "color",
        "reward_delay": 0,
        "random_shift_max_steps": 0,
    }


def _load_exact_wrapper_class() -> Any:
    source_text = REQUIRED_SOURCE_SRC_ROOT.as_posix()
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("environments.Foragax")
    except ImportError as exc:
        raise ForagerRngParityError("could not import the exact upstream wrapper") from exc
    raw_module_path = getattr(module, "__file__", None)
    if type(raw_module_path) is not str:
        raise ForagerRngParityError("upstream wrapper module has no source path")
    module_path = Path(raw_module_path).resolve()
    if module_path != REQUIRED_WRAPPER_PATH:
        raise ForagerRngParityError("upstream wrapper import resolved outside the locked source")
    wrapper_class = getattr(module, "Foragax", None)
    if not isinstance(wrapper_class, type):
        raise ForagerRngParityError("upstream wrapper class is missing")
    init_code = getattr(getattr(wrapper_class, "__init__", None), "__code__", None)
    if init_code is None or Path(str(init_code.co_filename)).resolve() != REQUIRED_WRAPPER_PATH:
        raise ForagerRngParityError("upstream wrapper class was not defined by the locked source")
    return wrapper_class


def _frame_from_split(input_key: Any) -> tuple[Any, Any, KeyFrame]:
    _, _, jr = _jax_modules()
    next_key, environment_key = jr.split(input_key)
    return (
        next_key,
        environment_key,
        KeyFrame(
            input_key=_key_words(input_key),
            next_key=_key_words(next_key),
            environment_key=_key_words(environment_key),
        ),
    )


def _run_wrapper_trace(config: FixedActionProbeConfig) -> RawEnvironmentTrace:
    jax, jnp, jr = _jax_modules()
    wrapper_class = _load_exact_wrapper_class()
    try:
        wrapper = wrapper_class(config.seed, **_task_kwargs())
        root_key = wrapper.state.key
        if str(jr.key_impl(root_key)) != REQUIRED_PRNG_IMPL:
            raise ForagerRngParityError("wrapper root key does not use threefry2x32")
        expected_next, _, reset_frame = _frame_from_split(root_key)
        observation = wrapper.start()
        jax.block_until_ready((observation, wrapper.state))
        if _key_words(wrapper.state.key) != _key_words(expected_next):
            raise ForagerRngParityError("wrapper reset carry key drifted")
        reset = RawResetRecord(
            keys=reset_frame,
            observation=observation,
            state=wrapper.state.state,
        )
        transitions: list[RawTransitionRecord] = []
        for index, action in enumerate(config.actions):
            input_key = wrapper.state.key
            expected_next, _, frame = _frame_from_split(input_key)
            raw_output = wrapper.step(jnp.asarray(action, dtype=jnp.int32))
            jax.block_until_ready((raw_output, wrapper.state))
            if _key_words(wrapper.state.key) != _key_words(expected_next):
                raise ForagerRngParityError(f"wrapper transition {index} carry key drifted")
            if not isinstance(raw_output, tuple) or len(raw_output) != 5:
                raise ForagerRngParityError("wrapper step output does not have five fields")
            next_observation, reward, terminated, truncated, info = raw_output
            if fingerprint_pytree(
                terminated,
                label=f"wrapper.transitions[{index}].terminated",
            ) != fingerprint_pytree(
                truncated,
                label=f"wrapper.transitions[{index}].truncated",
            ):
                raise ForagerRngParityError(
                    f"wrapper transition {index} termination flags disagree"
                )
            transitions.append(
                RawTransitionRecord(
                    index=index,
                    action=action,
                    keys=frame,
                    observation=next_observation,
                    reward=reward,
                    done=terminated,
                    info=info,
                    state=wrapper.state.state,
                )
            )
    except ForagerRngParityError:
        raise
    except Exception as exc:
        raise ForagerRngParityError(
            f"upstream wrapper execution failed with {type(exc).__name__}"
        ) from exc
    return RawEnvironmentTrace(reset=reset, transitions=tuple(transitions))


def _run_direct_trace(config: FixedActionProbeConfig) -> RawEnvironmentTrace:
    jax, jnp, jr = _jax_modules()
    try:
        from foragax.registry import make

        environment = make(**_task_kwargs())
        params = environment.default_params
        key = jr.key(config.seed)
        if str(jr.key_impl(key)) != REQUIRED_PRNG_IMPL:
            raise ForagerRngParityError("direct environment root key is not threefry2x32")
        next_key, reset_key, reset_frame = _frame_from_split(key)
        observation, state = environment.reset(reset_key, params)
        jax.block_until_ready((observation, state))
        key = next_key
        reset = RawResetRecord(keys=reset_frame, observation=observation, state=state)
        transitions: list[RawTransitionRecord] = []
        for index, action in enumerate(config.actions):
            next_key, step_key, frame = _frame_from_split(key)
            next_observation, state, reward, done, info = environment.step(
                step_key,
                state,
                jnp.asarray(action, dtype=jnp.int32),
                params,
            )
            jax.block_until_ready((next_observation, state, reward, done, info))
            transitions.append(
                RawTransitionRecord(
                    index=index,
                    action=action,
                    keys=frame,
                    observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info,
                    state=state,
                )
            )
            key = next_key
    except ForagerRngParityError:
        raise
    except Exception as exc:
        raise ForagerRngParityError(
            f"direct Foragax execution failed with {type(exc).__name__}"
        ) from exc
    return RawEnvironmentTrace(reset=reset, transitions=tuple(transitions))


def run_live_parity_probe(config: FixedActionProbeConfig) -> ParityProbeResult:
    """Execute the exact wrapper and direct environment under fixed actions."""
    runtime = collect_verified_runtime_identity()
    wrapper = _run_wrapper_trace(config)
    direct = _run_direct_trace(config)
    return compare_fixed_action_traces(config, wrapper, direct, runtime)


def _parse_key_words(value: Any, path: str) -> tuple[int, int]:
    items = _require_array(value, path)
    if len(items) != 2:
        raise ForagerRngParityError(f"{path} must contain exactly two words")
    words = tuple(
        _require_int(item, f"{path}[{index}]", minimum=0, maximum=_KEY_WORD_MAX)
        for index, item in enumerate(items)
    )
    return cast(tuple[int, int], words)


def _parse_key_frame(value: Any, path: str) -> KeyFrame:
    payload = _require_object(value, path)
    _require_exact_keys(payload, {"input_key", "next_key", "environment_key"}, path)
    return KeyFrame(
        input_key=_parse_key_words(payload["input_key"], f"{path}.input_key"),
        next_key=_parse_key_words(payload["next_key"], f"{path}.next_key"),
        environment_key=_parse_key_words(payload["environment_key"], f"{path}.environment_key"),
    )


def _parse_tree_digest(value: Any, path: str) -> TreeDigest:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        {"leaf_count", "structure_sha256", "content_sha256"},
        path,
    )
    return TreeDigest(
        leaf_count=_require_int(
            payload["leaf_count"],
            f"{path}.leaf_count",
            minimum=1,
            maximum=100_000,
        ),
        structure_sha256=_require_sha256(payload["structure_sha256"], f"{path}.structure_sha256"),
        content_sha256=_require_sha256(payload["content_sha256"], f"{path}.content_sha256"),
    )


def _parse_trace_digest(
    value: Any,
    config: FixedActionProbeConfig,
) -> EnvironmentTraceDigest:
    path = "result.matched_trace"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "seed",
            "action_sequence_sha256",
            "reset",
            "transitions",
            "trace_sha256",
        },
        path,
    )
    if payload["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ForagerRngParityError(f"{path}.schema_version is unsupported")
    seed = _require_int(payload["seed"], f"{path}.seed", minimum=0, maximum=MAX_SEED)
    if seed != config.seed:
        raise ForagerRngParityError(f"{path}.seed does not match result.probe")
    action_sha256 = _require_sha256(
        payload["action_sequence_sha256"],
        f"{path}.action_sequence_sha256",
    )
    if action_sha256 != config.action_sequence_sha256:
        raise ForagerRngParityError(f"{path}.action_sequence_sha256 does not match result.probe")
    reset_payload = _require_object(payload["reset"], f"{path}.reset")
    _require_exact_keys(
        reset_payload,
        {"keys", "observation", "state"},
        f"{path}.reset",
    )
    reset_keys = _parse_key_frame(reset_payload["keys"], f"{path}.reset.keys")
    reset_observation = _parse_tree_digest(
        reset_payload["observation"], f"{path}.reset.observation"
    )
    reset_state = _parse_tree_digest(reset_payload["state"], f"{path}.reset.state")
    transition_payloads = _require_array(payload["transitions"], f"{path}.transitions")
    if len(transition_payloads) != len(config.actions):
        raise ForagerRngParityError(f"{path}.transitions length does not match result.probe")
    transitions: list[TransitionDigest] = []
    expected_reset, expected_transitions = expected_key_schedule(config)
    _validate_key_frame(reset_keys, expected_reset, f"{path}.reset.keys")
    for index, (raw, action, expected_keys) in enumerate(
        zip(transition_payloads, config.actions, expected_transitions, strict=True)
    ):
        item_path = f"{path}.transitions[{index}]"
        item = _require_object(raw, item_path)
        _require_exact_keys(
            item,
            {
                "index",
                "action",
                "keys",
                "observation",
                "reward",
                "done",
                "info",
                "state",
            },
            item_path,
        )
        parsed_index = _require_int(
            item["index"], item_path + ".index", minimum=0, maximum=MAX_ACTIONS - 1
        )
        parsed_action = _require_int(item["action"], item_path + ".action", minimum=0, maximum=3)
        if parsed_index != index or parsed_action != action:
            raise ForagerRngParityError(f"{item_path} index/action does not match result.probe")
        keys = _parse_key_frame(item["keys"], item_path + ".keys")
        _validate_key_frame(keys, expected_keys, item_path + ".keys")
        transitions.append(
            TransitionDigest(
                index=index,
                action=action,
                keys=keys,
                observation=_parse_tree_digest(item["observation"], item_path + ".observation"),
                reward=_parse_tree_digest(item["reward"], item_path + ".reward"),
                done=_parse_tree_digest(item["done"], item_path + ".done"),
                info=_parse_tree_digest(item["info"], item_path + ".info"),
                state=_parse_tree_digest(item["state"], item_path + ".state"),
            )
        )
    trace_sha256 = _require_sha256(payload["trace_sha256"], f"{path}.trace_sha256")
    parsed = EnvironmentTraceDigest(
        seed=seed,
        action_sequence_sha256=action_sha256,
        reset_keys=reset_keys,
        reset_observation=reset_observation,
        reset_state=reset_state,
        transitions=tuple(transitions),
        trace_sha256=trace_sha256,
    )
    if _canonical_sha256(parsed.unsigned_dict()) != trace_sha256:
        raise ForagerRngParityError(f"{path}.trace_sha256 does not verify")
    return parsed


def _parse_runtime_identity(value: Any) -> VerifiedRuntimeIdentity:
    path = "result.runtime"
    payload = _require_object(value, path)
    expected_keys = set(VerifiedRuntimeIdentity.__dataclass_fields__)
    _require_exact_keys(payload, expected_keys, path)
    string_fields = expected_keys - {
        "cpu_device_count",
        "threefry_partitionable",
        "jax_enable_x64",
    }
    strings = {key: _require_string(payload[key], f"{path}.{key}") for key in string_fields}
    if type(payload["threefry_partitionable"]) is not bool:
        raise ForagerRngParityError(f"{path}.threefry_partitionable must be boolean")
    if type(payload["jax_enable_x64"]) is not bool:
        raise ForagerRngParityError(f"{path}.jax_enable_x64 must be boolean")
    identity = VerifiedRuntimeIdentity(
        required_oci_image_id=strings["required_oci_image_id"],
        build_attestation_sha256=strings["build_attestation_sha256"],
        source_repository=strings["source_repository"],
        source_commit=strings["source_commit"],
        source_tree_git_sha1=strings["source_tree_git_sha1"],
        source_archive_sha256=strings["source_archive_sha256"],
        source_archive_inventory_sha256=strings["source_archive_inventory_sha256"],
        dependency_lock_sha256=strings["dependency_lock_sha256"],
        wrapper_source_path=strings["wrapper_source_path"],
        wrapper_source_sha256=strings["wrapper_source_sha256"],
        source_mount_mode=strings["source_mount_mode"],
        foragax_distribution=strings["foragax_distribution"],
        foragax_version=strings["foragax_version"],
        foragax_wheel_sha256=strings["foragax_wheel_sha256"],
        foragax_install_tree_hash_scheme=strings["foragax_install_tree_hash_scheme"],
        foragax_install_tree_sha256=strings["foragax_install_tree_sha256"],
        python_version=strings["python_version"],
        python_executable_sha256=strings["python_executable_sha256"],
        jax_version=strings["jax_version"],
        jaxlib_version=strings["jaxlib_version"],
        backend=strings["backend"],
        cpu_device_count=_require_int(
            payload["cpu_device_count"],
            f"{path}.cpu_device_count",
            minimum=1,
            maximum=256,
        ),
        prng_impl=strings["prng_impl"],
        threefry_partitionable=payload["threefry_partitionable"],
        jax_enable_x64=payload["jax_enable_x64"],
        probe_module_sha256=strings["probe_module_sha256"],
    )
    _validate_runtime_identity(identity)
    return identity


def validate_parity_result(
    value: Mapping[str, Any] | bytes | str,
    *,
    expected_payload_sha256: str | None = None,
) -> ParityProbeResult:
    """Validate a result and optionally bind it to an externally expected hash.

    Without ``expected_payload_sha256`` this validates content consistency, not
    authenticity.  A party able to alter a payload can also recompute its
    self-hash; callers needing identity must supply an independently obtained
    expected hash and replay the probe in the pinned OCI image.
    """
    if isinstance(value, (bytes, str)):
        decoded = decode_strict_json(value)
    else:
        decoded = decode_strict_json(canonical_json_bytes(value))
    payload = _require_object(decoded, "result")
    top_keys = {
        "schema_version",
        "status",
        "evidence_boundary",
        "promotion_authorized",
        "runtime",
        "task",
        "rng_contract",
        "probe",
        "matched_trace",
        "wrapper_trace_sha256",
        "direct_trace_sha256",
        "payload_sha256",
    }
    _require_exact_keys(payload, top_keys, "result")
    if payload["schema_version"] != PARITY_RESULT_SCHEMA_VERSION:
        raise ForagerRngParityError("result.schema_version is unsupported")
    if payload["status"] != MATCH_STATUS:
        raise ForagerRngParityError("result.status is not an exact parity match")
    if payload["evidence_boundary"] != CONTENT_IDENTITY_BOUNDARY:
        raise ForagerRngParityError("result.evidence_boundary was weakened")
    if payload["promotion_authorized"] is not False:
        raise ForagerRngParityError("result may never authorize promotion")
    declared_sha256 = _require_sha256(payload["payload_sha256"], "result.payload_sha256")
    unsigned = dict(payload)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared_sha256:
        raise ForagerRngParityError("result.payload_sha256 does not verify")
    if expected_payload_sha256 is not None:
        expected = _require_sha256(
            expected_payload_sha256,
            "expected_payload_sha256",
        )
        if declared_sha256 != expected:
            raise ForagerRngParityError("result does not match the externally expected hash")

    task_payload = _require_object(payload["task"], "result.task")
    if canonical_json_bytes(task_payload) != canonical_json_bytes(task_descriptor()):
        raise ForagerRngParityError("result.task differs from the exact FOV task identity")
    rng_payload = _require_object(payload["rng_contract"], "result.rng_contract")
    if canonical_json_bytes(rng_payload) != canonical_json_bytes(rng_contract_descriptor()):
        raise ForagerRngParityError("result.rng_contract differs from the split-chain lock")

    probe = _require_object(payload["probe"], "result.probe")
    _require_exact_keys(
        probe,
        {"seed", "actions", "action_count", "action_sequence_sha256"},
        "result.probe",
    )
    actions_raw = _require_array(probe["actions"], "result.probe.actions")
    actions = tuple(
        _require_int(item, f"result.probe.actions[{index}]", minimum=0, maximum=3)
        for index, item in enumerate(actions_raw)
    )
    config = FixedActionProbeConfig(
        seed=_require_int(probe["seed"], "result.probe.seed", minimum=0, maximum=MAX_SEED),
        actions=actions,
    )
    action_count = _require_int(
        probe["action_count"],
        "result.probe.action_count",
        minimum=1,
        maximum=MAX_ACTIONS,
    )
    if action_count != len(actions):
        raise ForagerRngParityError("result.probe.action_count does not verify")
    action_sequence_sha256 = _require_sha256(
        probe["action_sequence_sha256"],
        "result.probe.action_sequence_sha256",
    )
    if action_sequence_sha256 != config.action_sequence_sha256:
        raise ForagerRngParityError("result.probe.action_sequence_sha256 does not verify")

    runtime = _parse_runtime_identity(payload["runtime"])
    trace = _parse_trace_digest(payload["matched_trace"], config)
    wrapper_sha256 = _require_sha256(payload["wrapper_trace_sha256"], "result.wrapper_trace_sha256")
    direct_sha256 = _require_sha256(payload["direct_trace_sha256"], "result.direct_trace_sha256")
    if wrapper_sha256 != trace.trace_sha256 or direct_sha256 != trace.trace_sha256:
        raise ForagerRngParityError("result trace hashes do not identify matched_trace")
    return ParityProbeResult(
        runtime=runtime,
        config=config,
        matched_trace=trace,
        wrapper_trace_sha256=wrapper_sha256,
        direct_trace_sha256=direct_sha256,
        payload_sha256=declared_sha256,
    )


def _parse_actions_argument(value: str) -> tuple[int, ...]:
    if not value or value.strip() != value:
        raise argparse.ArgumentTypeError("actions must be a comma-separated list")
    parts = value.split(",")
    if any(not part or not part.isascii() or not part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("actions must contain decimal integers")
    try:
        actions = tuple(int(part) for part in parts)
        FixedActionProbeConfig(seed=0, actions=actions)
    except ForagerRngParityError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return actions


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the exact upstream wrapper and direct Foragax environment under "
            "one fixed action sequence. Output is hash-only content identity."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--actions",
        type=_parse_actions_argument,
        required=True,
        help="comma-separated action integers in [0,3]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = FixedActionProbeConfig(seed=args.seed, actions=args.actions)
        result = run_live_parity_probe(config)
    except ForagerRngParityError as exc:
        parser.exit(2, f"forager RNG parity probe failed: {exc}\n")
    sys.stdout.buffer.write(result.canonical_bytes + b"\n")
    return 0


__all__ = [
    "ACTION_SEQUENCE_SCHEMA_VERSION",
    "CONTENT_IDENTITY_BOUNDARY",
    "FixedActionProbeConfig",
    "ForagerRngParityError",
    "ForagerRngParityMismatchError",
    "KeyFrame",
    "MATCH_STATUS",
    "PARITY_RESULT_SCHEMA_VERSION",
    "ParityProbeResult",
    "REQUIRED_OCI_IMAGE_ID",
    "RawEnvironmentTrace",
    "RawResetRecord",
    "RawTransitionRecord",
    "TreeDigest",
    "VerifiedRuntimeIdentity",
    "canonical_json_bytes",
    "collect_verified_runtime_identity",
    "compare_fixed_action_traces",
    "decode_strict_json",
    "digest_environment_trace",
    "expected_key_schedule",
    "fingerprint_pytree",
    "rng_contract_descriptor",
    "run_live_parity_probe",
    "task_descriptor",
    "validate_parity_result",
]


if __name__ == "__main__":  # pragma: no cover - exercised in the qualified OCI runtime
    raise SystemExit(main())
