"""Authority-free runtime provenance for noisy-world v6 DEVELOPMENT runs.

The record detects drift within one Python/JAX runtime and verifies the bytes
of every installed distribution file carrying a supported RECORD hash.
Unhashed RECORD entries are counted but intentionally excluded.  The record
makes no cross-backend reproducibility claim, derives no keys, sets no
thresholds, writes no files, and has no execution, evidence, artifact, or
promotion authority.
"""

from __future__ import annotations

import base64
import binascii
import csv
import dataclasses
import enum
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import platform
from typing import Any, cast

import jax

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.runtime-development.v2"
)
V6_RUNTIME_STATUS = "DEVELOPMENT_RUNTIME_PROVENANCE_NO_AUTHORITY"
V6_JAX_CONFIG_NORMALIZATION_SCHEMA = "alberta.jax-config.finite-json.v1"
V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA = "alberta.distribution-record-payload-verification.v1"

_SUPPORTED_RECORD_HASH_ALGORITHMS: tuple[str, ...] = (
    "blake2b",
    "blake2s",
    "sha224",
    "sha256",
    "sha384",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "sha512",
)
_PAYLOAD_READ_CHUNK_BYTES = 1024 * 1024

DEVELOPMENT_ONLY = True
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False
CROSS_BACKEND_REPRODUCIBILITY_CLAIMED = False

V6_RUNTIME_PACKAGE_ORDER: tuple[str, ...] = (
    "alberta-framework",
    "jax",
    "jaxlib",
    "numpy",
    "chex",
)

# Values are captured in this exact order, including explicit ``None`` for an
# absent variable.  These are inputs that can select a backend/device or alter
# JAX/XLA numerical, PRNG, compilation, allocation, or deterministic behavior.
V6_RUNTIME_ENVIRONMENT_ORDER: tuple[str, ...] = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH",
    "JAX_CUDA_VISIBLE_DEVICES",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DEFAULT_PRNG_IMPL",
    "JAX_DISABLE_JIT",
    "JAX_ENABLE_X64",
    "JAX_NUM_CPU_DEVICES",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "JAX_RANDOM_SEED_OFFSET",
    "JAX_ROCM_VISIBLE_DEVICES",
    "JAX_THREEFRY_GPU_KERNEL_LOWERING",
    "JAX_THREEFRY_PARTITIONABLE",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "NVIDIA_VISIBLE_DEVICES",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "ROCR_VISIBLE_DEVICES",
    "TF_DETERMINISTIC_OPS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TPU_VISIBLE_CHIPS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


class V6RuntimeValidationError(ValueError):
    """A runtime record is structurally invalid or differs from the live runtime."""


@dataclasses.dataclass(frozen=True, slots=True)
class V6RuntimePackageVersion:
    """One distribution and its verified RECORD-hashed payload identity."""

    name: str
    version: str
    record_sha256: str
    verified_hashed_file_count: int
    excluded_unhashed_entry_count: int
    verified_payload_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class V6RuntimeEnvironmentVariable:
    """One selected environment input, with absence represented explicitly."""

    name: str
    value: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class V6RuntimeDevice:
    """One JAX device in the exact order returned by ``jax.devices()``."""

    platform: str
    device_kind: str
    id: int
    process_index: int
    runtime_type: str
    platform_version: str
    ordinal: int


@dataclasses.dataclass(frozen=True, slots=True)
class V6RuntimeRecord:
    """Exact, immutable, authority-free identity of one DEVELOPMENT runtime."""

    schema: str
    status: str
    development_only: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    cross_backend_reproducibility_claimed: bool
    python_implementation: str
    python_version: str
    os_system: str
    os_release: str
    os_machine: str
    distribution_payload_verification_schema: str
    packages: tuple[V6RuntimePackageVersion, ...]
    jax_backend: str
    jax_process_count: int
    jax_process_index: int
    jax_enable_x64: bool
    jax_default_matmul_precision: str | None
    jax_default_prng_impl: str
    jax_config_normalization_schema: str
    jax_config_entry_count: int
    jax_config_sha256: str
    environment: tuple[V6RuntimeEnvironmentVariable, ...]
    devices: tuple[V6RuntimeDevice, ...]


_PACKAGE_FIELDS = (
    "name",
    "version",
    "record_sha256",
    "verified_hashed_file_count",
    "excluded_unhashed_entry_count",
    "verified_payload_sha256",
)
_ENVIRONMENT_FIELDS = ("name", "value")
_DEVICE_FIELDS = (
    "platform",
    "device_kind",
    "id",
    "process_index",
    "runtime_type",
    "platform_version",
    "ordinal",
)
_RECORD_FIELDS = (
    "schema",
    "status",
    "development_only",
    "execution_authorized",
    "evidence_authorized",
    "scientific_promotion_allowed",
    "cross_backend_reproducibility_claimed",
    "python_implementation",
    "python_version",
    "os_system",
    "os_release",
    "os_machine",
    "distribution_payload_verification_schema",
    "packages",
    "jax_backend",
    "jax_process_count",
    "jax_process_index",
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_default_prng_impl",
    "jax_config_normalization_schema",
    "jax_config_entry_count",
    "jax_config_sha256",
    "environment",
    "devices",
)


def _normalized_config_value(value: object) -> object:
    """Convert a JAX config value to finite, type-preserving JSON data."""

    if value is None:
        return {"kind": "none"}
    if isinstance(value, enum.Enum):
        enum_type = type(value)
        return {
            "class": f"{enum_type.__module__}.{enum_type.__qualname__}",
            "kind": "enum",
            "name": value.name,
            "value": _normalized_config_value(value.value),
        }
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise V6RuntimeValidationError("JAX config contains a non-finite float")
        return {"kind": "float-hex", "value": value.hex()}
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "value": [_normalized_config_value(item) for item in value],
        }
    if type(value) is list:
        return {
            "kind": "list",
            "value": [_normalized_config_value(item) for item in value],
        }
    if type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError("JAX config mappings must have exact built-in str keys")
        return {
            "kind": "dict",
            "value": [
                {"name": name, "value": _normalized_config_value(value[name])}
                for name in sorted(value)
            ],
        }
    value_type = type(value)
    raise TypeError(
        f"unsupported JAX config value type: {value_type.__module__}.{value_type.__qualname__}"
    )


def _jax_config_identity() -> tuple[int, str]:
    entries = [
        {"name": name, "value": _normalized_config_value(value)}
        for name, value in sorted(jax.config.values.items())
    ]
    raw = json.dumps(
        {
            "entries": entries,
            "normalization_schema": V6_JAX_CONFIG_NORMALIZATION_SCHEMA,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return len(entries), hashlib.sha256(raw).hexdigest()


def _exact_optional_config_string(name: str) -> str | None:
    value = jax.config.values[name]
    if value is None or type(value) is str:
        return value
    raise TypeError(f"JAX config {name} must be an exact built-in str or None")


def _exact_config_string(name: str) -> str:
    value = jax.config.values[name]
    if type(value) is not str:
        raise TypeError(f"JAX config {name} must be an exact built-in str")
    return value


def _decode_record_digest(algorithm: str, encoded: str, *, distribution_name: str) -> bytes:
    if algorithm not in _SUPPORTED_RECORD_HASH_ALGORITHMS:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} uses unsupported RECORD hash {algorithm!r}"
        )
    if not encoded or "=" in encoded:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} has malformed RECORD hash encoding"
        )
    padding = "=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(
            (encoded + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} has malformed RECORD hash encoding"
        ) from exc
    expected_size = hashlib.new(algorithm).digest_size
    if len(decoded) != expected_size:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} has a wrong-length RECORD hash"
        )
    return decoded


def _stream_payload_hashes(
    distribution: importlib.metadata.Distribution,
    relative_path: str,
    algorithm: str,
) -> tuple[int, bytes, str]:
    """Read one installed payload once and return size, RECORD hash, and SHA-256."""

    record_hasher = hashlib.new(algorithm)
    aggregate_hasher = record_hasher if algorithm == "sha256" else hashlib.sha256()
    located = distribution.locate_file(importlib.metadata.PackagePath(relative_path))
    byte_count = 0
    try:
        with cast(Any, located).open("rb") as handle:
            while True:
                chunk = handle.read(_PAYLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if type(chunk) is not bytes:
                    raise TypeError("installed distribution payload reads must return bytes")
                byte_count += len(chunk)
                record_hasher.update(chunk)
                if aggregate_hasher is not record_hasher:
                    aggregate_hasher.update(chunk)
    except (OSError, TypeError) as exc:
        raise V6RuntimeValidationError(
            f"installed payload is missing or unreadable: {relative_path!r}"
        ) from exc
    return byte_count, record_hasher.digest(), aggregate_hasher.hexdigest()


def _distribution_payload_identity(
    distribution: importlib.metadata.Distribution,
    distribution_name: str,
    record_text: str,
) -> tuple[int, int, str]:
    """Verify all supported RECORD-hashed payloads and aggregate their identity."""

    verified_entries: list[dict[str, object]] = []
    excluded_unhashed_count = 0
    seen_paths: set[str] = set()
    try:
        rows = csv.reader(io.StringIO(record_text, newline=""))
        for row_index, row in enumerate(rows):
            if len(row) != 3:
                raise V6RuntimeValidationError(
                    f"distribution {distribution_name!r} RECORD row {row_index} is malformed"
                )
            relative_path, hash_field, size_field = row
            if not relative_path or relative_path in seen_paths:
                raise V6RuntimeValidationError(
                    f"distribution {distribution_name!r} RECORD paths are empty or duplicated"
                )
            seen_paths.add(relative_path)
            declared_size: int | None = None
            if size_field:
                if not size_field.isascii() or not size_field.isdecimal():
                    raise V6RuntimeValidationError(
                        f"distribution {distribution_name!r} has malformed RECORD size"
                    )
                declared_size = int(size_field)
            if not hash_field:
                excluded_unhashed_count += 1
                continue
            if hash_field.count("=") != 1:
                raise V6RuntimeValidationError(
                    f"distribution {distribution_name!r} has malformed RECORD hash"
                )
            algorithm, encoded_digest = hash_field.split("=", 1)
            expected_digest = _decode_record_digest(
                algorithm,
                encoded_digest,
                distribution_name=distribution_name,
            )
            actual_size, actual_digest, payload_sha256 = _stream_payload_hashes(
                distribution,
                relative_path,
                algorithm,
            )
            if not hmac.compare_digest(actual_digest, expected_digest):
                raise V6RuntimeValidationError(
                    f"installed payload hash differs from RECORD: {relative_path!r}"
                )
            if declared_size is not None and actual_size != declared_size:
                raise V6RuntimeValidationError(
                    f"installed payload size differs from RECORD: {relative_path!r}"
                )
            verified_entries.append(
                {
                    "declared_size": declared_size,
                    "path": relative_path,
                    "payload_sha256": payload_sha256,
                    "record_hash": hash_field,
                    "verified_size": actual_size,
                }
            )
    except csv.Error as exc:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} RECORD CSV is malformed"
        ) from exc
    if not verified_entries:
        raise V6RuntimeValidationError(
            f"distribution {distribution_name!r} has no verifiable hashed RECORD entries"
        )
    verified_entries.sort(key=lambda entry: cast(str, entry["path"]))
    aggregate = json.dumps(
        {
            "entries": verified_entries,
            "excluded_unhashed_entry_count": excluded_unhashed_count,
            "schema": V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return (
        len(verified_entries),
        excluded_unhashed_count,
        hashlib.sha256(aggregate).hexdigest(),
    )


def _capture_packages() -> tuple[V6RuntimePackageVersion, ...]:
    try:
        records: list[V6RuntimePackageVersion] = []
        for name in V6_RUNTIME_PACKAGE_ORDER:
            distribution = importlib.metadata.distribution(name)
            record_text = distribution.read_text("RECORD")
            if record_text is None:
                raise V6RuntimeValidationError(
                    f"required runtime distribution has no RECORD: {name!r}"
                )
            verified_count, excluded_count, payload_sha256 = _distribution_payload_identity(
                distribution, name, record_text
            )
            records.append(
                V6RuntimePackageVersion(
                    name=name,
                    version=distribution.version,
                    record_sha256=hashlib.sha256(record_text.encode("utf-8")).hexdigest(),
                    verified_hashed_file_count=verified_count,
                    excluded_unhashed_entry_count=excluded_count,
                    verified_payload_sha256=payload_sha256,
                )
            )
        return tuple(records)
    except importlib.metadata.PackageNotFoundError as exc:
        raise V6RuntimeValidationError(
            f"required runtime distribution is not installed: {exc.name}"
        ) from exc


def _capture_devices() -> tuple[V6RuntimeDevice, ...]:
    records: list[V6RuntimeDevice] = []
    for ordinal, device in enumerate(jax.devices()):
        runtime_type = getattr(device.client, "runtime_type", None)
        platform_version = getattr(device.client, "platform_version", None)
        if type(runtime_type) is not str or type(platform_version) is not str:
            raise TypeError(
                "JAX device clients must expose exact runtime_type and platform_version strings"
            )
        records.append(
            V6RuntimeDevice(
                platform=str(device.platform),
                device_kind=str(device.device_kind),
                id=int(device.id),
                process_index=int(device.process_index),
                runtime_type=runtime_type,
                platform_version=platform_version,
                ordinal=ordinal,
            )
        )
    return tuple(records)


def capture_v6_runtime_record() -> V6RuntimeRecord:
    """Capture the exact live runtime without granting any scientific authority."""

    entry_count, config_sha256 = _jax_config_identity()
    enable_x64 = jax.config.values["jax_enable_x64"]
    if type(enable_x64) is not bool:
        raise TypeError("JAX config jax_enable_x64 must be an exact built-in bool")
    backend = jax.default_backend()
    if type(backend) is not str:
        raise TypeError("JAX default backend must be an exact built-in str")
    process_count = jax.process_count()
    process_index = jax.process_index()
    if type(process_count) is not int or type(process_index) is not int:
        raise TypeError("JAX process count and index must be exact built-in ints")
    record = V6RuntimeRecord(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA,
        status=V6_RUNTIME_STATUS,
        development_only=DEVELOPMENT_ONLY,
        execution_authorized=EXECUTION_AUTHORIZED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        cross_backend_reproducibility_claimed=CROSS_BACKEND_REPRODUCIBILITY_CLAIMED,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        os_system=platform.system(),
        os_release=platform.release(),
        os_machine=platform.machine(),
        distribution_payload_verification_schema=(V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA),
        packages=_capture_packages(),
        jax_backend=backend,
        jax_process_count=process_count,
        jax_process_index=process_index,
        jax_enable_x64=enable_x64,
        jax_default_matmul_precision=_exact_optional_config_string("jax_default_matmul_precision"),
        jax_default_prng_impl=_exact_config_string("jax_default_prng_impl"),
        jax_config_normalization_schema=V6_JAX_CONFIG_NORMALIZATION_SCHEMA,
        jax_config_entry_count=entry_count,
        jax_config_sha256=config_sha256,
        environment=tuple(
            V6RuntimeEnvironmentVariable(name=name, value=os.environ.get(name))
            for name in V6_RUNTIME_ENVIRONMENT_ORDER
        ),
        devices=_capture_devices(),
    )
    return validate_v6_runtime_record(record, require_live_match=False)


def _require_dataclass_shape(
    value: object,
    expected_type: type[Any],
    expected_fields: tuple[str, ...],
    *,
    path: str,
) -> None:
    if type(value) is not expected_type:
        raise TypeError(f"{path} must be an exact {expected_type.__name__}")
    actual_fields = tuple(field.name for field in dataclasses.fields(cast(Any, type(value))))
    if actual_fields != expected_fields:
        raise V6RuntimeValidationError(f"{path} dataclass field order differs from schema")


def _require_exact_string(value: object, *, path: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact built-in str")
    if not allow_empty and not value:
        raise V6RuntimeValidationError(f"{path} must be non-empty")
    return value


def _require_sha256(value: object, *, path: str) -> str:
    digest = _require_exact_string(value, path=path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise V6RuntimeValidationError(f"{path} must be 64 lowercase hexadecimal characters")
    return digest


def _validate_package_records(packages: object) -> None:
    if type(packages) is not tuple:
        raise TypeError("record.packages must be an exact built-in tuple")
    names: list[str] = []
    for index, package in enumerate(packages):
        path = f"record.packages[{index}]"
        _require_dataclass_shape(
            package,
            V6RuntimePackageVersion,
            _PACKAGE_FIELDS,
            path=path,
        )
        names.append(_require_exact_string(package.name, path=f"{path}.name"))
        _require_exact_string(package.version, path=f"{path}.version")
        _require_sha256(package.record_sha256, path=f"{path}.record_sha256")
        _require_sha256(
            package.verified_payload_sha256,
            path=f"{path}.verified_payload_sha256",
        )
        if type(package.verified_hashed_file_count) is not int:
            raise TypeError(f"{path}.verified_hashed_file_count must be an exact built-in int")
        if package.verified_hashed_file_count <= 0:
            raise V6RuntimeValidationError(f"{path}.verified_hashed_file_count must be positive")
        if type(package.excluded_unhashed_entry_count) is not int:
            raise TypeError(f"{path}.excluded_unhashed_entry_count must be an exact built-in int")
        if package.excluded_unhashed_entry_count < 0:
            raise V6RuntimeValidationError(
                f"{path}.excluded_unhashed_entry_count must be non-negative"
            )
    if len(names) != len(set(names)):
        raise V6RuntimeValidationError("record.packages contains duplicate names")
    if tuple(names) != V6_RUNTIME_PACKAGE_ORDER:
        raise V6RuntimeValidationError("record.packages differs from the required order")


def _validate_environment_records(environment: object) -> None:
    if type(environment) is not tuple:
        raise TypeError("record.environment must be an exact built-in tuple")
    names: list[str] = []
    for index, variable in enumerate(environment):
        path = f"record.environment[{index}]"
        _require_dataclass_shape(
            variable,
            V6RuntimeEnvironmentVariable,
            _ENVIRONMENT_FIELDS,
            path=path,
        )
        names.append(_require_exact_string(variable.name, path=f"{path}.name"))
        if variable.value is not None and type(variable.value) is not str:
            raise TypeError(f"{path}.value must be an exact built-in str or None")
    if len(names) != len(set(names)):
        raise V6RuntimeValidationError("record.environment contains duplicate names")
    if tuple(names) != V6_RUNTIME_ENVIRONMENT_ORDER:
        raise V6RuntimeValidationError("record.environment differs from the required order")


def _validate_device_records(devices: object) -> None:
    if type(devices) is not tuple:
        raise TypeError("record.devices must be an exact built-in tuple")
    if not devices:
        raise V6RuntimeValidationError("record.devices must contain at least one device")
    identities: list[tuple[int, int]] = []
    validated_devices: list[V6RuntimeDevice] = []
    for index, device in enumerate(devices):
        path = f"record.devices[{index}]"
        _require_dataclass_shape(device, V6RuntimeDevice, _DEVICE_FIELDS, path=path)
        _require_exact_string(device.platform, path=f"{path}.platform")
        _require_exact_string(device.device_kind, path=f"{path}.device_kind")
        _require_exact_string(device.runtime_type, path=f"{path}.runtime_type")
        _require_exact_string(device.platform_version, path=f"{path}.platform_version")
        if type(device.id) is not int:
            raise TypeError(f"{path}.id must be an exact built-in int")
        if type(device.process_index) is not int:
            raise TypeError(f"{path}.process_index must be an exact built-in int")
        if type(device.ordinal) is not int:
            raise TypeError(f"{path}.ordinal must be an exact built-in int")
        if device.id < 0 or device.process_index < 0 or device.ordinal < 0:
            raise V6RuntimeValidationError(f"{path} indices must be non-negative")
        if device.ordinal != index:
            raise V6RuntimeValidationError("record.devices differs from captured enumeration order")
        identities.append((device.process_index, device.id))
        validated_devices.append(device)
    if len(identities) != len(set(identities)):
        raise V6RuntimeValidationError("record.devices contains duplicate process/device ids")
    if tuple(validated_devices) != devices:
        raise V6RuntimeValidationError("record.devices has an invalid tuple representation")


def _validate_v6_runtime_record_structure(record: object) -> V6RuntimeRecord:
    _require_dataclass_shape(record, V6RuntimeRecord, _RECORD_FIELDS, path="record")
    assert type(record) is V6RuntimeRecord
    exact_constants: tuple[tuple[str, object, object], ...] = (
        (
            "schema",
            record.schema,
            HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA,
        ),
        ("status", record.status, V6_RUNTIME_STATUS),
        ("development_only", record.development_only, DEVELOPMENT_ONLY),
        ("execution_authorized", record.execution_authorized, EXECUTION_AUTHORIZED),
        ("evidence_authorized", record.evidence_authorized, EVIDENCE_AUTHORIZED),
        (
            "scientific_promotion_allowed",
            record.scientific_promotion_allowed,
            SCIENTIFIC_PROMOTION_ALLOWED,
        ),
        (
            "cross_backend_reproducibility_claimed",
            record.cross_backend_reproducibility_claimed,
            CROSS_BACKEND_REPRODUCIBILITY_CLAIMED,
        ),
        (
            "jax_config_normalization_schema",
            record.jax_config_normalization_schema,
            V6_JAX_CONFIG_NORMALIZATION_SCHEMA,
        ),
        (
            "distribution_payload_verification_schema",
            record.distribution_payload_verification_schema,
            V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA,
        ),
    )
    for name, actual, expected in exact_constants:
        if type(actual) is not type(expected):
            raise TypeError(f"record.{name} has the wrong exact built-in type")
        if actual != expected:
            raise V6RuntimeValidationError(f"record.{name} differs from the schema")

    for name in (
        "python_implementation",
        "python_version",
        "os_system",
        "os_release",
        "os_machine",
        "jax_backend",
        "jax_default_prng_impl",
    ):
        _require_exact_string(getattr(record, name), path=f"record.{name}")
    if record.jax_default_matmul_precision is not None:
        _require_exact_string(
            record.jax_default_matmul_precision,
            path="record.jax_default_matmul_precision",
            allow_empty=True,
        )
    if type(record.jax_enable_x64) is not bool:
        raise TypeError("record.jax_enable_x64 must be an exact built-in bool")
    if type(record.jax_process_count) is not int:
        raise TypeError("record.jax_process_count must be an exact built-in int")
    if type(record.jax_process_index) is not int:
        raise TypeError("record.jax_process_index must be an exact built-in int")
    if record.jax_process_count <= 0:
        raise V6RuntimeValidationError("record.jax_process_count must be positive")
    if not 0 <= record.jax_process_index < record.jax_process_count:
        raise V6RuntimeValidationError("record.jax_process_index must be in [0, jax_process_count)")
    if type(record.jax_config_entry_count) is not int:
        raise TypeError("record.jax_config_entry_count must be an exact built-in int")
    if record.jax_config_entry_count <= 0:
        raise V6RuntimeValidationError("record.jax_config_entry_count must be positive")
    _require_sha256(record.jax_config_sha256, path="record.jax_config_sha256")
    _validate_package_records(record.packages)
    _validate_environment_records(record.environment)
    _validate_device_records(record.devices)
    if any(device.process_index >= record.jax_process_count for device in record.devices):
        raise V6RuntimeValidationError(
            "record.devices contains a process index outside jax_process_count"
        )
    return record


def validate_v6_runtime_record(
    record: object,
    *,
    require_live_match: bool = True,
) -> V6RuntimeRecord:
    """Validate exact structure and, by default, equality to the live runtime."""

    if type(require_live_match) is not bool:
        raise TypeError("require_live_match must be an exact built-in bool")
    validated = _validate_v6_runtime_record_structure(record)
    if require_live_match:
        live = capture_v6_runtime_record()
        for field in dataclasses.fields(V6RuntimeRecord):
            if getattr(validated, field.name) != getattr(live, field.name):
                raise V6RuntimeValidationError(f"record.{field.name} differs from the live runtime")
    return validated


__all__ = [
    "CROSS_BACKEND_REPRODUCIBILITY_CLAIMED",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA",
    "V6_JAX_CONFIG_NORMALIZATION_SCHEMA",
    "V6_RUNTIME_ENVIRONMENT_ORDER",
    "V6_RUNTIME_PACKAGE_ORDER",
    "V6_RUNTIME_STATUS",
    "V6RuntimeDevice",
    "V6RuntimeEnvironmentVariable",
    "V6RuntimePackageVersion",
    "V6RuntimeRecord",
    "V6RuntimeValidationError",
    "capture_v6_runtime_record",
    "validate_v6_runtime_record",
]
