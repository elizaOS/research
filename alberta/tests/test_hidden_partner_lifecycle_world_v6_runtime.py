"""Focused tests for authority-free noisy-world v6 runtime provenance."""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
from typing import Any, cast

import jax
import pytest

import alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime as runtime
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import (
    CROSS_BACKEND_REPRODUCIBILITY_CLAIMED,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA,
    V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA,
    V6_RUNTIME_ENVIRONMENT_ORDER,
    V6_RUNTIME_PACKAGE_ORDER,
    V6_RUNTIME_STATUS,
    V6RuntimeDevice,
    V6RuntimeEnvironmentVariable,
    V6RuntimePackageVersion,
    V6RuntimeRecord,
    V6RuntimeValidationError,
    capture_v6_runtime_record,
    validate_v6_runtime_record,
)

pytestmark = pytest.mark.unit


def _replace(record: V6RuntimeRecord, **changes: object) -> V6RuntimeRecord:
    return cast(V6RuntimeRecord, dataclasses.replace(cast(Any, record), **changes))


def test_capture_is_frozen_canonical_and_matches_live_runtime() -> None:
    record = capture_v6_runtime_record()
    repeated = capture_v6_runtime_record()

    assert type(record) is V6RuntimeRecord
    assert record == repeated
    assert validate_v6_runtime_record(record) is record
    assert record.schema == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNTIME_SCHEMA
    assert record.status == V6_RUNTIME_STATUS
    assert record.development_only is True
    assert record.execution_authorized is False
    assert record.evidence_authorized is False
    assert record.scientific_promotion_allowed is False
    assert record.cross_backend_reproducibility_claimed is (CROSS_BACKEND_REPRODUCIBILITY_CLAIMED)
    assert record.cross_backend_reproducibility_claimed is False
    assert record.jax_process_count == jax.process_count()
    assert record.jax_process_index == jax.process_index()
    assert record.jax_config_entry_count == len(jax.config.values)
    assert len(record.jax_config_sha256) == 64
    assert set(record.jax_config_sha256) <= set("0123456789abcdef")
    assert (
        record.distribution_payload_verification_schema
        == V6_DISTRIBUTION_PAYLOAD_VERIFICATION_SCHEMA
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = "changed"  # type: ignore[misc]


def test_capture_binds_exact_package_environment_and_device_orders() -> None:
    record = capture_v6_runtime_record()

    assert tuple(package.name for package in record.packages) == V6_RUNTIME_PACKAGE_ORDER
    assert tuple(package.version for package in record.packages) == tuple(
        importlib.metadata.version(name) for name in V6_RUNTIME_PACKAGE_ORDER
    )
    for package in record.packages:
        distribution = importlib.metadata.distribution(package.name)
        record_text = distribution.read_text("RECORD")
        assert record_text is not None
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
        assert package.record_sha256 == hashlib.sha256(record_text.encode("utf-8")).hexdigest()
        assert package.verified_hashed_file_count == sum(bool(row[1]) for row in rows)
        assert package.excluded_unhashed_entry_count == sum(not row[1] for row in rows)
        assert package.verified_hashed_file_count > 0
        assert len(package.verified_payload_sha256) == 64
    assert tuple(variable.name for variable in record.environment) == (V6_RUNTIME_ENVIRONMENT_ORDER)
    assert tuple(variable.value for variable in record.environment) == tuple(
        os.environ.get(name) for name in V6_RUNTIME_ENVIRONMENT_ORDER
    )
    assert {
        "CUDA_DEVICE_ORDER",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "MKL_NUM_THREADS",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    } <= set(V6_RUNTIME_ENVIRONMENT_ORDER)
    assert record.devices
    assert tuple(device.ordinal for device in record.devices) == tuple(range(len(record.devices)))
    assert tuple(
        (
            device.platform,
            device.device_kind,
            device.id,
            device.process_index,
            device.platform_version,
        )
        for device in record.devices
    ) == tuple(
        (
            device.platform,
            device.device_kind,
            device.id,
            device.process_index,
            device.client.platform_version,
        )
        for device in jax.devices()
    )


def test_capture_fails_closed_when_installed_payload_differs_from_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime._stream_payload_hashes
    corrupted = False

    def mismatched_payload(
        distribution: importlib.metadata.Distribution,
        relative_path: str,
        algorithm: str,
    ) -> tuple[int, bytes, str]:
        nonlocal corrupted
        byte_count, digest, payload_sha256 = original(
            distribution,
            relative_path,
            algorithm,
        )
        if not corrupted:
            corrupted = True
            digest = bytes((digest[0] ^ 1, *digest[1:]))
        return byte_count, digest, payload_sha256

    monkeypatch.setattr(runtime, "_stream_payload_hashes", mismatched_payload)

    with pytest.raises(V6RuntimeValidationError, match="payload hash differs from RECORD"):
        capture_v6_runtime_record()
    assert corrupted


def test_config_normalization_is_finite_typed_and_deterministic() -> None:
    assert runtime._normalized_config_value(-0.0) == {
        "kind": "float-hex",
        "value": "-0x0.0p+0",
    }
    assert runtime._normalized_config_value(0) != runtime._normalized_config_value(False)
    assert runtime._normalized_config_value((1, "x")) == (
        runtime._normalized_config_value((1, "x"))
    )
    with pytest.raises(V6RuntimeValidationError, match="non-finite"):
        runtime._normalized_config_value(math.inf)
    with pytest.raises(V6RuntimeValidationError, match="non-finite"):
        runtime._normalized_config_value(math.nan)
    with pytest.raises(TypeError, match="unsupported"):
        runtime._normalized_config_value(object())


def test_validator_rejects_authority_digest_and_live_mismatches() -> None:
    record = capture_v6_runtime_record()

    with pytest.raises(V6RuntimeValidationError, match="execution_authorized"):
        validate_v6_runtime_record(
            _replace(record, execution_authorized=True),
            require_live_match=False,
        )
    with pytest.raises(V6RuntimeValidationError, match="jax_config_sha256"):
        validate_v6_runtime_record(
            _replace(record, jax_config_sha256="A" * 64),
            require_live_match=False,
        )
    with pytest.raises(
        V6RuntimeValidationError,
        match="distribution_payload_verification_schema",
    ):
        validate_v6_runtime_record(
            _replace(record, distribution_payload_verification_schema="stale"),
            require_live_match=False,
        )
    with pytest.raises(V6RuntimeValidationError, match="jax_config_sha256"):
        validate_v6_runtime_record(
            _replace(record, jax_config_sha256="0" * 63),
            require_live_match=False,
        )
    structurally_valid_stale = _replace(record, python_version=record.python_version + ".stale")
    assert (
        validate_v6_runtime_record(structurally_valid_stale, require_live_match=False)
        is structurally_valid_stale
    )
    with pytest.raises(V6RuntimeValidationError, match="python_version.*live runtime"):
        validate_v6_runtime_record(structurally_valid_stale)


def test_validator_rejects_top_level_subclasses_missing_and_extra_fields() -> None:
    record = capture_v6_runtime_record()

    class RuntimeSubclass(V6RuntimeRecord):
        pass

    subclass = RuntimeSubclass(
        **{field.name: getattr(record, field.name) for field in dataclasses.fields(record)}
    )

    @dataclasses.dataclass(frozen=True)
    class MissingFields:
        schema: str

    @dataclasses.dataclass(frozen=True)
    class ExtraFields:
        schema: str
        extra: str

    for malformed in (
        subclass,
        MissingFields(schema=record.schema),
        ExtraFields(schema=record.schema, extra="unexpected"),
        dataclasses.asdict(record),
    ):
        with pytest.raises(TypeError, match="exact V6RuntimeRecord"):
            validate_v6_runtime_record(malformed, require_live_match=False)
    with pytest.raises(TypeError, match="require_live_match"):
        validate_v6_runtime_record(record, require_live_match=cast(Any, 1))


def test_validator_rejects_package_duplicates_order_drift_and_wrong_types() -> None:
    record = capture_v6_runtime_record()
    duplicate = (record.packages[0],) * len(record.packages)

    with pytest.raises(V6RuntimeValidationError, match="duplicate"):
        validate_v6_runtime_record(
            _replace(record, packages=duplicate),
            require_live_match=False,
        )
    with pytest.raises(V6RuntimeValidationError, match="required order"):
        validate_v6_runtime_record(
            _replace(record, packages=tuple(reversed(record.packages))),
            require_live_match=False,
        )
    with pytest.raises(TypeError, match="exact built-in tuple"):
        validate_v6_runtime_record(
            _replace(record, packages=cast(Any, list(record.packages))),
            require_live_match=False,
        )
    malformed = V6RuntimePackageVersion(
        name=cast(Any, 1),
        version="1",
        record_sha256="0" * 64,
        verified_hashed_file_count=1,
        excluded_unhashed_entry_count=0,
        verified_payload_sha256="0" * 64,
    )
    with pytest.raises(TypeError, match="name"):
        validate_v6_runtime_record(
            _replace(record, packages=(malformed, *record.packages[1:])),
            require_live_match=False,
        )
    malformed_digest = dataclasses.replace(record.packages[0], record_sha256="G" * 64)
    with pytest.raises(V6RuntimeValidationError, match="record_sha256"):
        validate_v6_runtime_record(
            _replace(record, packages=(malformed_digest, *record.packages[1:])),
            require_live_match=False,
        )
    malformed_payload_digest = dataclasses.replace(
        record.packages[0],
        verified_payload_sha256="G" * 64,
    )
    with pytest.raises(V6RuntimeValidationError, match="verified_payload_sha256"):
        validate_v6_runtime_record(
            _replace(record, packages=(malformed_payload_digest, *record.packages[1:])),
            require_live_match=False,
        )
    zero_verified = dataclasses.replace(record.packages[0], verified_hashed_file_count=0)
    with pytest.raises(V6RuntimeValidationError, match="verified_hashed_file_count"):
        validate_v6_runtime_record(
            _replace(record, packages=(zero_verified, *record.packages[1:])),
            require_live_match=False,
        )


def test_validator_rejects_environment_duplicates_order_drift_and_wrong_types() -> None:
    record = capture_v6_runtime_record()
    duplicate = (record.environment[0],) * len(record.environment)

    with pytest.raises(V6RuntimeValidationError, match="duplicate"):
        validate_v6_runtime_record(
            _replace(record, environment=duplicate),
            require_live_match=False,
        )
    with pytest.raises(V6RuntimeValidationError, match="required order"):
        validate_v6_runtime_record(
            _replace(record, environment=tuple(reversed(record.environment))),
            require_live_match=False,
        )
    malformed = V6RuntimeEnvironmentVariable(
        name=record.environment[0].name,
        value=cast(Any, 1),
    )
    with pytest.raises(TypeError, match="value"):
        validate_v6_runtime_record(
            _replace(record, environment=(malformed, *record.environment[1:])),
            require_live_match=False,
        )


def test_validator_rejects_device_duplicates_order_drift_and_wrong_types() -> None:
    record = capture_v6_runtime_record()
    device_zero = V6RuntimeDevice(
        platform="cpu",
        device_kind="test-cpu",
        id=0,
        process_index=0,
        runtime_type="test",
        platform_version="test-platform",
        ordinal=0,
    )
    device_one = dataclasses.replace(device_zero, id=1, ordinal=1)

    with pytest.raises(V6RuntimeValidationError, match="duplicate"):
        validate_v6_runtime_record(
            _replace(
                record,
                devices=(device_zero, dataclasses.replace(device_zero, ordinal=1)),
            ),
            require_live_match=False,
        )
    with pytest.raises(V6RuntimeValidationError, match="enumeration order"):
        validate_v6_runtime_record(
            _replace(record, devices=(device_one, device_zero)),
            require_live_match=False,
        )
    malformed = dataclasses.replace(device_zero, id=cast(Any, True))
    with pytest.raises(TypeError, match=r"devices\[0\]\.id"):
        validate_v6_runtime_record(
            _replace(record, devices=(malformed,)),
            require_live_match=False,
        )
    malformed_platform_version = dataclasses.replace(
        device_zero,
        platform_version=cast(Any, 1),
    )
    with pytest.raises(TypeError, match="platform_version"):
        validate_v6_runtime_record(
            _replace(record, devices=(malformed_platform_version,)),
            require_live_match=False,
        )
    outside_process_count = dataclasses.replace(
        device_zero,
        process_index=record.jax_process_count,
    )
    with pytest.raises(V6RuntimeValidationError, match="outside jax_process_count"):
        validate_v6_runtime_record(
            _replace(record, devices=(outside_process_count,)),
            require_live_match=False,
        )


def test_module_has_no_cli_or_write_surface_and_record_has_no_run_authority_fields() -> None:
    source_path = Path(runtime.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    field_names = {field.name for field in dataclasses.fields(V6RuntimeRecord)}

    assert "argparse" not in imports
    assert "pathlib" not in imports
    assert "open" not in called_names
    assert not any(
        forbidden in name
        for name in field_names
        for forbidden in ("timestamp", "path", "argv", "key", "threshold", "artifact")
    )
