"""Bound exogenous source for the issued-unused cadence-separated protocol.

This module derives four typed Threefry keys from the frozen protocol config,
binds both earlier consumed development histories, constructs only exogenous
arrays, and fails closed against literal stream pins.
It does not initialize a learner, enter an operational attempt, execute an arm,
write output, produce a result, or authorize evidence or promotion.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import platform
from importlib.metadata import distribution as package_distribution
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as protocol,
)

DEVELOPMENT_ONLY: Final = True
SOURCE_GENERATION_ALLOWED: Final = True
OPERATIONAL_ENTRY_CONSUMED: Final = False
PANEL_EXECUTION_AUTHORIZED: Final = False
RESULT_AVAILABLE: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False

KEY_DERIVATION_DOMAIN: Final = "alberta-fu-v3-key"
PROTOCOL_CONFIG_SHA256: Final = (
    "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"
)
KEY_ROLES: Final = (
    "observations",
    "exploration",
    "random_actions",
    "learner_genesis",
)
KEY_MANIFEST: Final = MappingProxyType(
    {
        "observations": (1_180_172_486, 737_689_529),
        "exploration": (1_781_034_651, 2_339_008_862),
        "random_actions": (3_049_045_980, 2_471_128_907),
        "learner_genesis": (2_648_309_318, 4_192_599_369),
    }
)
KEY_MANIFEST_SHA256: Final = (
    "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"
)
CONSUMED_STREAM_SHA256S: Final = (
    "02fd5efbbb304b624fcfd29e259c361d5048233817e896300057d8e36f3fc036",
    "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c",
)
CONSUMED_STREAM_SHA256: Final[str] = CONSUMED_STREAM_SHA256S[-1]
CONSUMED_KEY_MANIFESTS: Final = (
    MappingProxyType(
        {
            "root": (0, 329_631_721),
            "observations": (2_316_273_231, 3_036_545_927),
            "exploration": (2_227_216_649, 3_977_711_669),
            "random_actions": (382_045_127, 333_255_797),
            "learner_genesis": (2_002_082_676, 3_427_004_161),
        }
    ),
    MappingProxyType(
        {
            "root": (0, 1_924_178_934),
            "observations": (1_189_056_302, 2_383_774_845),
            "exploration": (3_352_410_003, 3_947_271_724),
            "random_actions": (3_382_640_669, 4_117_898_437),
            "learner_genesis": (2_592_838_183, 3_227_537_730),
        }
    ),
)
CONSUMED_HISTORY_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-consumed-history.v1"
)
CONSUMED_HISTORY_SCOPE: Final = (
    "development-history reuse exclusion only; no result, evidence, or "
    "promotion authority"
)
CONSUMED_HISTORY_LANES: Final = (
    "compositional_future_utility_development_v1",
    "compositional_future_utility_calibration_v2",
)
CONSUMED_HISTORY_KEY_ROLES: Final = (
    "root",
    "observations",
    "exploration",
    "random_actions",
    "learner_genesis",
)
CONSUMED_HISTORY_SHA256: Final = (
    "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
)
STREAM_SHA256: Final[str] = (
    "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758"
)
CADENCE_BOUND_STREAM_SHA256: Final = (
    "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
)
CONTROL_PROTOCOL_CONFIG_SHA256: Final = (
    "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
)
STREAM_ENVELOPE_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-source-stream."
    "full-binding.v2"
)
STREAM_ENVELOPE_SHA256: Final = (
    "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
)
INSTALLED_DISTRIBUTION_MANIFEST_SCHEMA: Final = (
    "alberta.installed-python-distribution-manifest.v1"
)
BYTE_BOUND_RUNTIME_DISTRIBUTIONS: Final = (
    "chex",
    "jax",
    "jaxlib",
    "jaxtyping",
)
CHEX_DISTRIBUTION_FILE_MANIFEST_SHA256: Final = (
    "50ada405c5dd57cca4da5add6f4b8f63bb053d5d24407f48ab4f7b86e3015664"
)
JAX_DISTRIBUTION_FILE_MANIFEST_SHA256: Final = (
    "83109308c4587705e55cca19f7325b68a0f93542de44891404655ebab43dc52b"
)
JAXLIB_DISTRIBUTION_FILE_MANIFEST_SHA256: Final = (
    "08392cf01d90354a176bd0ce41e6602368b49069140639e0231358b8642a798b"
)
JAXTYPING_DISTRIBUTION_FILE_MANIFEST_SHA256: Final = (
    "60ec4237d3efc025a5af88eadbd9137ed83aed18130961e0d78b103bc62d301e"
)
RUNTIME_CONFIG: Final = MappingProxyType(
    {
        "python_version": "3.12.3",
        "platform_system": "Linux",
        "machine": "x86_64",
        "jax_version": "0.11.0",
        "jax_distribution_file_manifest_sha256": (
            JAX_DISTRIBUTION_FILE_MANIFEST_SHA256
        ),
        "jaxlib_version": "0.11.0",
        "jaxlib_distribution_file_manifest_sha256": (
            JAXLIB_DISTRIBUTION_FILE_MANIFEST_SHA256
        ),
        "numpy_version": "2.5.1",
        "chex_version": "0.1.92",
        "chex_distribution_file_manifest_sha256": (
            CHEX_DISTRIBUTION_FILE_MANIFEST_SHA256
        ),
        "jaxtyping_version": "0.3.11",
        "jaxtyping_distribution_file_manifest_sha256": (
            JAXTYPING_DISTRIBUTION_FILE_MANIFEST_SHA256
        ),
        "jax_backend": "cpu",
        "jax_device_count": 1,
        "jax_device_kinds": ("cpu",),
        "jax_enable_x64": False,
        "jax_threefry_partitionable": True,
        "jax_default_prng_impl": "threefry2x32",
        "jax_default_matmul_precision": None,
        "jax_disable_jit": False,
        "env_JAX_ENABLE_X64": None,
        "env_JAX_THREEFRY_PARTITIONABLE": None,
        "env_JAX_DEFAULT_PRNG_IMPL": None,
        "env_JAX_PLATFORMS": None,
        "env_JAX_PLATFORM_NAME": None,
        "env_XLA_FLAGS": None,
    }
)
RUNTIME_CONFIG_SHA256: Final = (
    "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6"
)
_MAPPING_PROXY_TYPE: Final[type] = type(MappingProxyType({}))
STREAM_ARRAY_NAMES: Final = (
    "observations",
    "phase_indices",
    "exploration_mask",
    "random_actions",
    "curation_due_mask",
)


@dataclasses.dataclass(frozen=True, slots=True)
class StreamArrayRecord:
    """Exact host representation and byte digest of one source array."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    bytes_sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in STREAM_ARRAY_NAMES:
            raise ValueError("stream array name is not declared")
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("stream array dtype must be an exact string")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError("stream array dtype is invalid") from error
        if dtype.str != self.dtype:
            raise ValueError("stream array dtype must use its canonical NumPy string")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value < 1 for value in self.shape)
        ):
            raise ValueError("stream array shape must contain positive exact integers")
        if type(self.nbytes) is not int or self.nbytes < 1:
            raise ValueError("stream array nbytes must be a positive exact integer")
        expected_nbytes = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
        if self.nbytes != expected_nbytes:
            raise ValueError("stream array nbytes do not match dtype and shape")
        if (
            type(self.bytes_sha256) is not str
            or len(self.bytes_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.bytes_sha256)
        ):
            raise ValueError("stream array digest must be lowercase SHA-256")

    def to_config(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "bytes_sha256": self.bytes_sha256,
        }


STREAM_RECORDS: Final = (
    StreamArrayRecord(
        "observations",
        "<f4",
        (8_998, 6),
        215_952,
        "af5d844a6d1fb34846c2546f0c65d40b08ecb6f73c43377462bab6b248eda037",
    ),
    StreamArrayRecord(
        "phase_indices",
        "<i4",
        (8_998,),
        35_992,
        "eb116ff28a0b48aff2103d139a3ee8a8bf1ee476aaa18234e3b39440ebbf43ad",
    ),
    StreamArrayRecord(
        "exploration_mask",
        "|b1",
        (8_998,),
        8_998,
        "332f464c2b993279deb337b51f297813599f02d3dbac60e099a6a3df9294ea91",
    ),
    StreamArrayRecord(
        "random_actions",
        "<i4",
        (8_998,),
        35_992,
        "2e4ef70a6e927a74ab016e921c1490421afca0a89d9b695251009343a1abaafe",
    ),
    StreamArrayRecord(
        "curation_due_mask",
        "|b1",
        (8_998,),
        8_998,
        "e845ea063d2a6c064374142714fe12b5a96afa7e615604fa57148b2445d7a707",
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class BoundV3Source:
    """Fully reconstructed source and its validated literal receipts."""

    control_protocol: control.CompositionalControlLifeProtocol
    source: control.BoundCompositionalControlLifeSource
    stream_records: tuple[StreamArrayRecord, ...]
    control_protocol_config_sha256: str
    runtime_config_sha256: str
    consumed_history_sha256: str
    key_manifest_sha256: str
    stream_sha256: str
    cadence_bound_stream_sha256: str
    stream_envelope_sha256: str
    validated: bool

    def __post_init__(self) -> None:
        validate_protocol_and_source_constants()
        if type(self.control_protocol) is not control.CompositionalControlLifeProtocol:
            raise TypeError("control protocol must be the exact control-life protocol")
        if (
            _canonical_sha256(self.control_protocol.to_config())
            != CONTROL_PROTOCOL_CONFIG_SHA256
        ):
            raise ValueError("control protocol receipt does not match its frozen config")
        if type(self.source) is not control.BoundCompositionalControlLifeSource:
            raise TypeError("source must be the exact bound control-life source")
        reconstructed_records = _validate_constructed_source(
            self.control_protocol,
            self.source,
        )
        canonical_source = dataclasses.replace(
            self.source,
            key_manifest=MappingProxyType(dict(self.source.key_manifest)),
        )
        object.__setattr__(self, "source", canonical_source)
        if self.stream_records != STREAM_RECORDS or reconstructed_records != STREAM_RECORDS:
            raise ValueError("stream array receipts do not match their frozen pins")
        if (
            self.control_protocol_config_sha256 != CONTROL_PROTOCOL_CONFIG_SHA256
            or self.runtime_config_sha256 != RUNTIME_CONFIG_SHA256
            or self.consumed_history_sha256 != CONSUMED_HISTORY_SHA256
            or self.key_manifest_sha256 != KEY_MANIFEST_SHA256
            or self.stream_sha256 != STREAM_SHA256
            or self.cadence_bound_stream_sha256 != CADENCE_BOUND_STREAM_SHA256
            or self.stream_envelope_sha256 != STREAM_ENVELOPE_SHA256
            or self.validated is not True
        ):
            raise ValueError("bound source receipts are not exact and valid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _validate_sha256(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")


def canonical_consumed_history_config() -> dict[str, object]:
    """Return the exact exclusion-only records for both consumed development lanes."""

    records: list[dict[str, object]] = []
    for lane, root, root_hex, stream_sha256, key_manifest in zip(
        CONSUMED_HISTORY_LANES,
        protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS,
        protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES,
        CONSUMED_STREAM_SHA256S,
        CONSUMED_KEY_MANIFESTS,
        strict=True,
    ):
        records.append(
            {
                "lane": lane,
                "development_root": root,
                "development_root_hex": root_hex,
                "stream_sha256": stream_sha256,
                "key_manifest": {
                    role: list(key_manifest[role])
                    for role in CONSUMED_HISTORY_KEY_ROLES
                },
            }
        )
    return {
        "schema": CONSUMED_HISTORY_SCHEMA,
        "scope": CONSUMED_HISTORY_SCOPE,
        "records": records,
    }


def _validate_consumed_history_constants() -> None:
    """Fail closed on any root, stream, role-key, or digest history drift."""

    record_count = 2
    collections = (
        ("consumed history lanes", CONSUMED_HISTORY_LANES),
        (
            "consumed development roots",
            protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS,
        ),
        (
            "consumed development root hexes",
            protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES,
        ),
        ("consumed streams", CONSUMED_STREAM_SHA256S),
        ("consumed key manifests", CONSUMED_KEY_MANIFESTS),
    )
    for label, values in collections:
        if type(values) is not tuple or len(values) != record_count:
            raise ValueError(f"{label} must contain exactly two frozen records")
    if (
        len(set(CONSUMED_HISTORY_LANES)) != record_count
        or any(type(lane) is not str or not lane for lane in CONSUMED_HISTORY_LANES)
    ):
        raise ValueError("consumed history lane identities are invalid")
    if CONSUMED_STREAM_SHA256 != CONSUMED_STREAM_SHA256S[-1]:
        raise ValueError("consumed stream compatibility alias drifted")
    if len(set(CONSUMED_STREAM_SHA256S)) != record_count:
        raise ValueError("consumed stream digests must be pairwise distinct")

    for root, root_hex, stream_sha256, key_manifest in zip(
        protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS,
        protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES,
        CONSUMED_STREAM_SHA256S,
        CONSUMED_KEY_MANIFESTS,
        strict=True,
    ):
        if type(root) is not int or not 0 <= root <= 0xFFFFFFFF:
            raise ValueError("consumed development root must be an unsigned 32-bit integer")
        if root_hex != protocol.format_root_hex(root):
            raise ValueError("consumed development root hexadecimal identity drifted")
        _validate_sha256(stream_sha256, label="consumed stream digest")
        if type(key_manifest) is not _MAPPING_PROXY_TYPE:
            raise ValueError("consumed history key manifest must be immutable")
        if tuple(key_manifest) != CONSUMED_HISTORY_KEY_ROLES:
            raise ValueError("consumed history key manifest roles are incomplete or reordered")
        for role in CONSUMED_HISTORY_KEY_ROLES:
            words = key_manifest[role]
            if (
                type(words) is not tuple
                or len(words) != 2
                or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words)
            ):
                raise ValueError(f"consumed history role key {role} is invalid")
        if key_manifest["root"] != (0, root):
            raise ValueError("consumed root key does not match its development identity")

    _validate_sha256(CONSUMED_HISTORY_SHA256, label="consumed history digest")
    if _canonical_sha256(canonical_consumed_history_config()) != CONSUMED_HISTORY_SHA256:
        raise ValueError("consumed history digest does not reconstruct")


def derive_role_key_words(protocol_config_sha256: str, role: str) -> tuple[int, int]:
    """Derive one typed-key payload from the frozen config digest and role."""

    _validate_sha256(protocol_config_sha256, label="protocol config digest")
    if type(role) is not str or role not in KEY_ROLES:
        raise ValueError("key role is not declared")
    digest = hashlib.sha256(
        KEY_DERIVATION_DOMAIN.encode("ascii")
        + b"\0"
        + protocol_config_sha256.encode("ascii")
        + b"\0"
        + role.encode("ascii")
    ).digest()
    return (
        int.from_bytes(digest[:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    )


def _key_manifest_config() -> dict[str, list[int]]:
    return {name: list(KEY_MANIFEST[name]) for name in KEY_ROLES}


def _stream_envelope_config(
    records: tuple[StreamArrayRecord, ...],
) -> dict[str, object]:
    return {
        "schema": STREAM_ENVELOPE_SCHEMA,
        "protocol_config_sha256": PROTOCOL_CONFIG_SHA256,
        "control_protocol_config_sha256": CONTROL_PROTOCOL_CONFIG_SHA256,
        "consumed_history_sha256": CONSUMED_HISTORY_SHA256,
        "key_derivation_domain": KEY_DERIVATION_DOMAIN,
        "key_manifest_sha256": KEY_MANIFEST_SHA256,
        "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
        "stream_sha256": STREAM_SHA256,
        "cadence_bound_stream_sha256": CADENCE_BOUND_STREAM_SHA256,
        "arrays": [record.to_config() for record in records],
    }


def _installed_distribution_manifest(distribution_name: str) -> dict[str, object]:
    """Hash exact installed files after checking their local RECORD receipts."""

    if (
        type(distribution_name) is not str
        or distribution_name not in BYTE_BOUND_RUNTIME_DISTRIBUTIONS
    ):
        raise ValueError("installed distribution name is not declared")
    distribution = package_distribution(distribution_name)
    observed_name = distribution.metadata["Name"]
    observed_version = distribution.version
    if observed_name != distribution_name or type(observed_version) is not str:
        raise ValueError("installed distribution identity is not exact")

    root_candidate = Path(str(distribution.locate_file("")))
    if not root_candidate.is_absolute() or root_candidate.is_symlink():
        raise ValueError("installed distribution root must be absolute and nonsymlinked")
    try:
        root = root_candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("installed distribution root cannot be resolved") from error
    if root != root_candidate or not root.is_dir():
        raise ValueError("installed distribution root must be an exact directory")

    package_files = distribution.files
    if package_files is None or not package_files:
        raise ValueError("installed distribution has no RECORD file inventory")
    records: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    record_self_rows = 0
    for package_path in sorted(package_files, key=lambda value: str(value)):
        relative_text = str(package_path)
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in relative_text
            or relative_text in seen_paths
        ):
            raise ValueError("installed distribution RECORD path is not canonical")
        seen_paths.add(relative_text)

        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("installed distribution RECORD path contains a symlink")
        located = Path(str(distribution.locate_file(package_path)))
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError("installed distribution RECORD file is missing") from error
        if (
            located != candidate
            or resolved != candidate
            or not resolved.is_relative_to(root)
            or not resolved.is_file()
        ):
            raise ValueError("installed distribution RECORD file is not exact")

        is_record_self = (
            relative.name == "RECORD"
            and relative.parent.name.endswith(".dist-info")
        )
        if is_record_self:
            record_self_rows += 1
            if package_path.hash is not None or package_path.size is not None:
                raise ValueError("installed distribution RECORD self-row must be unhashed")
            continue

        record_hash = package_path.hash
        record_size = package_path.size
        if (
            record_hash is None
            or record_hash.mode != "sha256"
            or type(record_hash.value) is not str
            or type(record_size) is not int
            or record_size < 0
        ):
            raise ValueError("installed distribution file lacks an exact SHA-256 receipt")
        raw = resolved.read_bytes()
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        record_sha256 = base64.urlsafe_b64encode(
            bytes.fromhex(observed_sha256)
        ).rstrip(b"=").decode("ascii")
        if record_size != len(raw) or record_hash.value != record_sha256:
            raise ValueError("installed distribution file differs from its RECORD receipt")
        records.append(
            {
                "path": relative_text,
                "sha256": observed_sha256,
                "nbytes": len(raw),
            }
        )
    if record_self_rows != 1:
        raise ValueError("installed distribution must contain one RECORD self-row")
    return {
        "schema": INSTALLED_DISTRIBUTION_MANIFEST_SCHEMA,
        "distribution": observed_name,
        "version": observed_version,
        "files": records,
    }


def _observed_runtime_config() -> dict[str, object]:
    chex_manifest = _installed_distribution_manifest("chex")
    jax_manifest = _installed_distribution_manifest("jax")
    jaxlib_manifest = _installed_distribution_manifest("jaxlib")
    jaxtyping_manifest = _installed_distribution_manifest("jaxtyping")
    precision = jax.config.jax_default_matmul_precision
    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "jax_version": jax.__version__,
        "jax_distribution_file_manifest_sha256": _canonical_sha256(jax_manifest),
        "jaxlib_version": package_version("jaxlib"),
        "jaxlib_distribution_file_manifest_sha256": _canonical_sha256(
            jaxlib_manifest
        ),
        "numpy_version": np.__version__,
        "chex_version": chex_manifest["version"],
        "chex_distribution_file_manifest_sha256": _canonical_sha256(
            chex_manifest
        ),
        "jaxtyping_version": jaxtyping_manifest["version"],
        "jaxtyping_distribution_file_manifest_sha256": _canonical_sha256(
            jaxtyping_manifest
        ),
        "jax_backend": jax.default_backend(),
        "jax_device_count": len(jax.devices()),
        "jax_device_kinds": tuple(device.device_kind for device in jax.devices()),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_default_matmul_precision": (
            None if precision is None else str(precision)
        ),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "env_JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "env_JAX_THREEFRY_PARTITIONABLE": os.environ.get(
            "JAX_THREEFRY_PARTITIONABLE"
        ),
        "env_JAX_DEFAULT_PRNG_IMPL": os.environ.get("JAX_DEFAULT_PRNG_IMPL"),
        "env_JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "env_JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        "env_XLA_FLAGS": os.environ.get("XLA_FLAGS"),
    }


def _control_protocol() -> control.CompositionalControlLifeProtocol:
    return control.CompositionalControlLifeProtocol(
        phase_lengths=protocol.PHASE_LENGTHS,
        epsilon=protocol.EPSILON,
        entry_window=protocol.ENTRY_WINDOW,
        tail_window=protocol.TAIL_WINDOW,
    )


def _validate_control_semantics() -> None:
    if (
        protocol.RAW_DIM != control.RAW_DIM
        or protocol.ACTIVE_SLOTS != control.ACTIVE_SLOTS
        or protocol.CANDIDATE_SLOTS != control.CANDIDATE_SLOTS
        or protocol.ACTION_HEADS != control.ACTION_HEADS
        or protocol.ALLOCATED_MAX_DEPTH != control.ALLOCATED_MAX_DEPTH
        or protocol.CURATION_INTERVAL != control.CURATION_INTERVAL
        or protocol.PHASE_ORDER != control.PHASE_ORDER
        or protocol.SIGNATURE_NAMES != control.SIGNATURE_NAMES
        or protocol.SIGNATURE_RAW_INDICES != control.SIGNATURE_RAW_INDICES
        or protocol.SIGNATURE_ROLES != control.SIGNATURE_ROLES
    ):
        raise ValueError("protocol and control-life geometry or semantics differ")

    source_arms = tuple(
        arm
        for arm in control.CONTROL_LIFE_ARMS
        if arm.name == protocol.LEFT_PACK_SOURCE_ARM
    )
    if len(source_arms) != 1 or source_arms[0].to_config() != dict(
        protocol.SOURCE_ARM_CONFIG
    ):
        raise ValueError("named control-life source arm differs from its protocol pin")

    historical = control.learner_config_for_arm(protocol.LEFT_PACK_SOURCE_ARM)
    for arm in protocol.ARMS:
        expected = dict(historical)
        expected.update(
            {
                "candidate_scoring_mode": "legacy",
                "candidate_novelty_admission_bonus": 0.0,
                "future_utility_trace_mode": "contribution",
                "future_utility_mix": arm.future_utility_mix,
                "future_utility_trace_decay": arm.future_utility_trace_decay,
                "future_utility_normalization": arm.future_utility_normalization,
                "future_utility_normalization_decay": 0.99,
                "future_utility_rare_task_power": 0.0,
            }
        )
        if expected != protocol.reconstruct_arm_learner_config(arm):
            raise ValueError("full production learner config differs from the protocol")

    sentinel = jnp.asarray((2.0, 3.0, 5.0, 7.0, 11.0, 13.0), dtype=jnp.float32)
    sentinel_host = np.asarray(sentinel)
    for phase, indices in enumerate(protocol.PHASE_TARGET_RAW_INDICES):
        expected_target = float(np.prod(sentinel_host[list(indices)]))
        observed_target = float(control._phase_target(sentinel, jnp.asarray(phase)))
        if observed_target != expected_target:
            raise ValueError("phase target semantics differ from their protocol map")

    if _canonical_sha256(_control_protocol().to_config()) != (
        CONTROL_PROTOCOL_CONFIG_SHA256
    ):
        raise ValueError("control-life protocol config differs from its source pin")


def validate_protocol_and_source_constants(
    *,
    observed_protocol_config_sha256: str = PROTOCOL_CONFIG_SHA256,
) -> None:
    """Validate all non-array bindings before source construction."""

    _validate_sha256(
        observed_protocol_config_sha256,
        label="protocol config digest",
    )
    if (
        observed_protocol_config_sha256 != PROTOCOL_CONFIG_SHA256
        or protocol.PROTOCOL_CONFIG_SHA256 != PROTOCOL_CONFIG_SHA256
        or protocol.protocol_config_sha256(
            protocol.CompositionalFutureUtilityCalibrationV3Protocol()
        )
        != PROTOCOL_CONFIG_SHA256
    ):
        raise ValueError("protocol config digest does not match the source binding")
    _validate_consumed_history_constants()
    observed_runtime = _observed_runtime_config()
    if observed_runtime != dict(RUNTIME_CONFIG) or _canonical_sha256(
        observed_runtime
    ) != RUNTIME_CONFIG_SHA256:
        raise ValueError("runtime configuration does not match the source binding")

    derived = {
        role: derive_role_key_words(PROTOCOL_CONFIG_SHA256, role) for role in KEY_ROLES
    }
    if derived != dict(KEY_MANIFEST) or len(set(derived.values())) != len(derived):
        raise ValueError("key manifest does not reconstruct uniquely")
    if _canonical_sha256(_key_manifest_config()) != KEY_MANIFEST_SHA256:
        raise ValueError("key manifest digest does not reconstruct")
    consumed_words = {
        words for manifest in CONSUMED_KEY_MANIFESTS for words in manifest.values()
    }
    if set(derived.values()) & consumed_words:
        raise ValueError("issued key manifest collides with a consumed manifest")
    if tuple(record.name for record in STREAM_RECORDS) != STREAM_ARRAY_NAMES:
        raise ValueError("stream record order does not match the declared array order")
    if _canonical_sha256(_stream_envelope_config(STREAM_RECORDS)) != STREAM_ENVELOPE_SHA256:
        raise ValueError("stream envelope digest does not reconstruct")
    if STREAM_SHA256 in CONSUMED_STREAM_SHA256S:
        raise ValueError("issued stream collides with a consumed stream")
    _validate_control_semantics()


def _typed_key(role: str) -> Array:
    return cast(
        Array,
        jr.wrap_key_data(
            jnp.asarray(KEY_MANIFEST[role], dtype=jnp.uint32),
            impl="threefry2x32",
        ),
    )


def _record_array(name: str, value: object) -> StreamArrayRecord:
    array = np.ascontiguousarray(np.asarray(value))
    return StreamArrayRecord(
        name=name,
        dtype=array.dtype.str,
        shape=tuple(array.shape),
        nbytes=int(array.nbytes),
        bytes_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    )


def _validate_constructed_source(
    control_protocol: control.CompositionalControlLifeProtocol,
    source: control.BoundCompositionalControlLifeSource,
) -> tuple[StreamArrayRecord, ...]:
    if type(control_protocol) is not control.CompositionalControlLifeProtocol:
        raise TypeError("control protocol must be exact")
    if type(source) is not control.BoundCompositionalControlLifeSource:
        raise TypeError("source must be exact")
    if type(source.key_manifest) is not _MAPPING_PROXY_TYPE:
        raise TypeError("source key manifest must be an immutable mapping proxy")
    if (
        not isinstance(source.learner_key, Array)
        or source.learner_key.shape != ()
        or str(jr.key_impl(source.learner_key)) != "threefry2x32"
    ):
        raise TypeError("source learner key must be a scalar typed Threefry JAX key")
    if _canonical_sha256(control_protocol.to_config()) != CONTROL_PROTOCOL_CONFIG_SHA256:
        raise ValueError("constructed source has the wrong control protocol")
    if source.scientific_promotion_allowed or source.evidence_authorized:
        raise ValueError("constructed source acquired scientific authority")
    if source.output_writes_allowed:
        raise ValueError("constructed source acquired output authority")

    arrays = (
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
        source.curation_due_mask,
    )
    if any(not isinstance(value, Array) for value in arrays):
        raise TypeError("constructed source arrays must be JAX arrays")
    records = tuple(
        _record_array(name, value)
        for name, value in zip(STREAM_ARRAY_NAMES, arrays, strict=True)
    )
    if records != STREAM_RECORDS:
        raise ValueError("constructed stream arrays do not match their frozen receipts")
    if dict(source.key_manifest) != dict(KEY_MANIFEST):
        raise ValueError("constructed source key manifest does not match its pin")
    learner_words = tuple(
        int(value)
        for value in np.asarray(jr.key_data(source.learner_key), dtype=np.uint32)
    )
    if learner_words != KEY_MANIFEST["learner_genesis"]:
        raise ValueError("constructed learner key does not match its pin")

    phase_indices = np.asarray(source.phase_indices)
    expected_phases = np.repeat(
        np.arange(len(protocol.PHASE_ORDER), dtype=np.int32),
        np.asarray(protocol.PHASE_LENGTHS, dtype=np.int32),
    )
    if not np.array_equal(phase_indices, expected_phases):
        raise ValueError("constructed phase-index runs do not match the exact schedule")
    due = np.asarray(source.curation_due_mask)
    expected_due = (
        np.arange(1, protocol.TOTAL_STEPS + 1, dtype=np.int32)
        % protocol.CURATION_INTERVAL
        == 0
    )
    if not np.array_equal(due, expected_due):
        raise ValueError("constructed due mask does not match the exact cadence")

    stream_arrays = arrays[:4]
    reconstructed_stream_sha256 = control._array_tree_sha256(stream_arrays)
    reconstructed_cadence_sha256 = control._array_tree_sha256(arrays)
    if (
        reconstructed_stream_sha256 != STREAM_SHA256
        or source.stream_sha256 != reconstructed_stream_sha256
        or reconstructed_cadence_sha256 != CADENCE_BOUND_STREAM_SHA256
        or source.cadence_bound_stream_sha256 != reconstructed_cadence_sha256
    ):
        raise ValueError("constructed combined stream digests do not match their pins")
    if _canonical_sha256(_stream_envelope_config(records)) != STREAM_ENVELOPE_SHA256:
        raise ValueError("constructed stream envelope does not match its pin")
    return records


def build_bound_v3_source() -> BoundV3Source:
    """Construct and validate exogenous arrays without entering an attempt."""

    validate_protocol_and_source_constants()
    control_protocol = _control_protocol()
    source = control.build_bound_compositional_control_life_source(
        control_protocol,
        observation_key=_typed_key("observations"),
        exploration_key=_typed_key("exploration"),
        random_action_key=_typed_key("random_actions"),
        learner_key=_typed_key("learner_genesis"),
    )
    records = _validate_constructed_source(control_protocol, source)
    return BoundV3Source(
        control_protocol=control_protocol,
        source=source,
        stream_records=records,
        control_protocol_config_sha256=CONTROL_PROTOCOL_CONFIG_SHA256,
        runtime_config_sha256=RUNTIME_CONFIG_SHA256,
        consumed_history_sha256=CONSUMED_HISTORY_SHA256,
        key_manifest_sha256=KEY_MANIFEST_SHA256,
        stream_sha256=STREAM_SHA256,
        cadence_bound_stream_sha256=CADENCE_BOUND_STREAM_SHA256,
        stream_envelope_sha256=STREAM_ENVELOPE_SHA256,
        validated=True,
    )


def validate_bound_v3_source(bound: BoundV3Source) -> BoundV3Source:
    """Revalidate every field and receipt, then return the exact supplied object.

    Re-entering the frozen dataclass constructor applies the same protocol,
    runtime, array, key, cadence, stream, envelope, and authority checks used at
    initial binding.  The reconstructed value is intentionally discarded: this
    postflight is an identity-preserving validation boundary, not a rebinding or
    source-generation operation.
    """

    if type(bound) is not BoundV3Source:
        raise TypeError("bound must be an exact BoundV3Source")
    dataclasses.replace(bound)
    return bound


__all__ = [
    "BYTE_BOUND_RUNTIME_DISTRIBUTIONS",
    "CADENCE_BOUND_STREAM_SHA256",
    "CONSUMED_HISTORY_KEY_ROLES",
    "CONSUMED_HISTORY_LANES",
    "CONSUMED_HISTORY_SCHEMA",
    "CONSUMED_HISTORY_SCOPE",
    "CONSUMED_HISTORY_SHA256",
    "CONSUMED_KEY_MANIFESTS",
    "CONSUMED_STREAM_SHA256",
    "CONSUMED_STREAM_SHA256S",
    "CONTROL_PROTOCOL_CONFIG_SHA256",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "KEY_DERIVATION_DOMAIN",
    "KEY_MANIFEST",
    "KEY_MANIFEST_SHA256",
    "KEY_ROLES",
    "OPERATIONAL_ENTRY_CONSUMED",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "PROTOCOL_CONFIG_SHA256",
    "RESULT_AVAILABLE",
    "RUNTIME_CONFIG",
    "RUNTIME_CONFIG_SHA256",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_GENERATION_ALLOWED",
    "STREAM_ARRAY_NAMES",
    "STREAM_ENVELOPE_SCHEMA",
    "STREAM_ENVELOPE_SHA256",
    "STREAM_RECORDS",
    "STREAM_SHA256",
    "BoundV3Source",
    "StreamArrayRecord",
    "build_bound_v3_source",
    "canonical_consumed_history_config",
    "derive_role_key_words",
    "validate_bound_v3_source",
    "validate_protocol_and_source_constants",
]
