"""Fail-closed artifact tooling for the reserved-but-forbidden v4 hidden-partner tuning grid.

The v4 grid (``LEASE_TUNING_GRID`` in
:mod:`alberta_framework.evaluation.hidden_partner_lifecycle_v2`) tunes the
evidence-gated confirmed-memory lease of
:class:`~alberta_framework.core.integrated_hidden_partner.IntegratedHiddenPartnerAgent`:
eight immutable cells over retention grace steps {2048, 3072, 4096} crossed
with active-utility evidence thresholds 0.075–0.40, each to be run on eight
seed pairs derived from ``LEASE_TUNING_NAMESPACE`` and ranked by the frozen
lexicographic ``TUNING_SELECTION_RULE``.

The namespace was reserved fresh — earlier v3 development seeds were inspected
while diagnosing downstream consumer interference, so they could no longer
serve as untouched tuning seeds — but the grid was never executed, and the
namespace status is pinned ``FORBIDDEN/UNEXECUTED``.  Every production path
refuses while that status stands: :func:`build_lease_tuning_artifact` and
:func:`validate_lease_tuning_artifact` both call
``require_lease_tuning_execution_allowed()`` before touching seeds or
payloads.  Only the private ``_..._for_testing`` helpers bypass the gate, and
only for synthetic structural fixtures.

Validation trusts nothing in a candidate artifact: strict JSON loading rejects
duplicate keys and non-finite constants, the declared source closure is
re-hashed against the working tree, and every derived quantity — per-window
NLL/accuracy metrics, consumer read/write gates, confirmed feature-memory
commitment, lifecycle claims, per-cell aggregates, and the selected cell — is
reconstructed from run primitives and compared exactly.  The artifact is
development-only and can never promote scientific evidence.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import functools
import hashlib
import json
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerAgent,
)
from alberta_framework.evaluation.hidden_partner_development import (
    HiddenPartnerCondition,
    HiddenPartnerDevelopmentProtocol,
    derive_hidden_partner_seed_pairs,
    hidden_partner_run_summary_from_dict,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_v2 import (
    CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD,
    CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD,
    CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD,
    CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD,
    CRITICAL_MASKED_NLL_INCREASE_THRESHOLD,
    CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD,
    CRITICAL_RUN_PRIMITIVES_SCHEMA,
    FEATURE_LEARNING_WINDOW,
    FINAL_ABSENCE_WINDOW,
    HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA,
    INITIAL_LATE_REWARD_THRESHOLD,
    LEASE_TUNING_GRID,
    LEASE_TUNING_NAMESPACE,
    LEASE_TUNING_SCOPE_LIMITS,
    LEASE_TUNING_SEED_COUNT,
    MINIMUM_JOINT_SUCCESS_FRACTION,
    RECURRENT_EARLY_REWARD_THRESHOLD,
    RECURRENT_ENTRY_WINDOW,
    RETENTION_RATIO_THRESHOLD,
    RETIREMENT_CONFIRMATION_WINDOW,
    TUNING_SELECTION_RULE,
    CriticalLifecycleV2Summary,
    require_lease_tuning_execution_allowed,
)
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingWorld,
)

LEASE_TUNING_ARTIFACT_SCHEMA = "alberta.hidden-partner-lease-tuning-artifact.v4"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _algorithm_source_paths() -> tuple[Path, ...]:
    """Return the declared hidden-partner algorithm and dependency closure.

    The complete core and utility layers are pinned because the integrated
    kernel imports through those package surfaces. Only the directly relevant
    stream and evaluation family is included; unrelated benchmark drivers do
    not participate in this artifact and must not invalidate a long grid run.
    """
    package = REPO_ROOT / "alberta_framework"
    selected = {
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("alberta_framework/__init__.py"),
        Path("alberta_framework/streams/__init__.py"),
        Path("alberta_framework/streams/hidden_partner_mapping.py"),
        Path("alberta_framework/evaluation/__init__.py"),
    }
    for directory in ("core", "utils"):
        selected.update(
            path.relative_to(REPO_ROOT)
            for path in (package / directory).rglob("*.py")
        )
    selected.update(
        path.relative_to(REPO_ROOT)
        for path in (package / "evaluation").glob("hidden_partner*.py")
    )
    return tuple(sorted(selected))


SOURCE_PATHS = _algorithm_source_paths()


@dataclass(frozen=True)
class LeaseTuningArtifactValidation:
    """Structural validity and separately reported grid feasibility."""

    valid: bool
    feasible_cell_selected: bool
    errors: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (OverflowError, TypeError, ValueError):
        return False


def source_snapshot(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the declared algorithmic source closure for the tuning record."""
    return {
        relative.as_posix(): hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


@functools.lru_cache(maxsize=1)
def _expected_integrated_state_nbytes() -> int:
    agent = IntegratedHiddenPartnerAgent(LEASE_TUNING_GRID[0].agent_config())
    start = agent.start(
        jnp.zeros((8,), dtype=jnp.float32),
        jr.key(0),
    )
    return agent.resource_budget(start.state).total_state_nbytes


def _build_lease_tuning_artifact_for_testing(
    record: Mapping[str, object],
    *,
    operational_metadata: Mapping[str, object],
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build a synthetic artifact for structural unit tests only.

    This private helper intentionally does not authorize execution or artifact
    issuance.  Production callers must use :func:`build_lease_tuning_artifact`,
    which enforces the namespace status.
    """
    payload = {**dict(record), "source_sha256": source_snapshot(root)}
    return {
        "schema_version": LEASE_TUNING_ARTIFACT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "scientific_payload": payload,
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": _canonical_sha256(payload),
        },
        "operational_metadata": dict(operational_metadata),
    }


def build_lease_tuning_artifact(
    record: Mapping[str, object],
    *,
    operational_metadata: Mapping[str, object],
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Issue an artifact only when the namespace is explicitly executable."""
    require_lease_tuning_execution_allowed()
    return _build_lease_tuning_artifact_for_testing(
        record,
        operational_metadata=operational_metadata,
        root=root,
    )


def lease_tuning_artifact_json(artifact: Mapping[str, object]) -> str:
    """Serialize strict pretty JSON."""
    return (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_lease_tuning_artifact(path: Path) -> dict[str, object]:
    """Load strict JSON while rejecting duplicate fields and constants."""

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def parse_integer(value: str) -> int:
        if len(value.lstrip("-")) > 20:
            raise ValueError("JSON integer exceeds the artifact size bound")
        return int(value)

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON float is forbidden")
        return parsed

    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_int=parse_integer,
        parse_float=parse_float,
    )
    if not isinstance(parsed, dict):
        raise ValueError("lease tuning artifact must be one JSON object")
    return parsed


def _validate_operational_metadata(
    operational: object,
    errors: list[str],
) -> None:
    expected = {
        "argv",
        "generated_at_utc",
        "jax_backend",
        "jax_device_count",
        "jax_devices",
        "jaxlib_version",
        "jax_version",
        "numpy_version",
        "platform",
        "python_version",
        "wall_seconds",
    }
    if not isinstance(operational, Mapping) or set(operational) != expected:
        errors.append("operational_metadata fields do not match the schema")
        return
    if not isinstance(operational.get("argv"), list) or any(
        not isinstance(item, str) for item in cast(list[object], operational["argv"])
    ):
        errors.append("operational argv must be an array of strings")
    for field in (
        "generated_at_utc",
        "jax_backend",
        "jax_version",
        "jaxlib_version",
        "numpy_version",
        "platform",
        "python_version",
    ):
        value = operational.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"operational {field} must be a non-empty string")
    timestamp = operational.get("generated_at_utc")
    if isinstance(timestamp, str) and timestamp:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            errors.append("operational generated_at_utc must be ISO-8601")
        else:
            utc_offset = parsed_timestamp.utcoffset()
            if utc_offset is None or utc_offset.total_seconds() != 0.0:
                errors.append("operational generated_at_utc must identify UTC")
    count = operational.get("jax_device_count")
    if type(count) is not int or count <= 0:
        errors.append("operational jax_device_count must be positive")
    devices = operational.get("jax_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, str) or not item for item in devices)
        or (type(count) is int and len(devices) != count)
    ):
        errors.append("operational jax_devices must match jax_device_count")
    wall = operational.get("wall_seconds")
    try:
        valid_wall = (
            not isinstance(wall, bool)
            and isinstance(wall, Real)
            and math.isfinite(float(wall))
            and float(wall) >= 0.0
        )
    except (OverflowError, TypeError, ValueError):
        valid_wall = False
    if not valid_wall:
        errors.append("operational wall_seconds must be finite and non-negative")


def _validate_rle(
    raw: object,
    state_count: int,
    field: str,
    errors: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not isinstance(raw, list) or not raw:
        errors.append(f"{field} must be a non-empty RLE array")
        return None
    expected_fields = {
        "start",
        "end_exclusive",
        "deployed_slot",
        "shadow_slot",
        "candidate_slot",
    }
    previous_end = 0
    previous_state: tuple[int, int, int] | None = None
    valid = True
    deployed_slots = np.full(state_count, -2, dtype=np.int32)
    shadow_slots = np.full(state_count, -2, dtype=np.int32)
    candidate_slots = np.full(state_count, -2, dtype=np.int32)
    for index, interval in enumerate(raw):
        if not isinstance(interval, Mapping) or set(interval) != expected_fields:
            errors.append(f"{field}[{index}] fields are invalid")
            valid = False
            continue
        values = [interval.get(name) for name in expected_fields]
        if any(type(value) is not int for value in values):
            errors.append(f"{field}[{index}] values must be integers")
            valid = False
            continue
        start = cast(int, interval["start"])
        end = cast(int, interval["end_exclusive"])
        deployed = cast(int, interval["deployed_slot"])
        shadow = cast(int, interval["shadow_slot"])
        candidate = cast(int, interval["candidate_slot"])
        interval_valid = True
        if start != previous_end or not start < end <= state_count:
            errors.append(f"{field}[{index}] is noncontiguous or empty")
            interval_valid = False
        if not -1 <= deployed < 12 or not -1 <= shadow < 12:
            errors.append(f"{field}[{index}] active slot is out of bounds")
            interval_valid = False
        if not 0 <= candidate < 66:
            errors.append(f"{field}[{index}] candidate slot is out of bounds")
            interval_valid = False
        state = (deployed, shadow, candidate)
        if previous_state == state:
            errors.append(f"{field}[{index}] is a noncanonical split")
            interval_valid = False
        if interval_valid:
            deployed_slots[start:end] = deployed
            shadow_slots[start:end] = shadow
            candidate_slots[start:end] = candidate
        else:
            valid = False
        previous_state = state
        previous_end = end
    if previous_end != state_count:
        errors.append(f"{field} does not cover the complete life")
        valid = False
    if (
        not valid
        or np.any(deployed_slots == -2)
        or np.any(shadow_slots == -2)
        or np.any(candidate_slots == -2)
    ):
        return None
    return deployed_slots, shadow_slots, candidate_slots


def _strict_int(
    value: object,
    field: str,
    errors: list[str],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int:
        errors.append(f"{field} must be a non-boolean integer")
        return None
    result = value
    if minimum is not None and result < minimum:
        errors.append(f"{field} is below its minimum")
    if maximum is not None and result > maximum:
        errors.append(f"{field} exceeds its maximum")
    return result


def _strict_float(
    value: object,
    field: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    try:
        if isinstance(value, bool) or not isinstance(value, Real):
            valid_type = False
            result = math.nan
        else:
            valid_type = True
            result = float(value)
    except (OverflowError, TypeError, ValueError):
        valid_type = False
        result = math.nan
    if not valid_type or not math.isfinite(result):
        errors.append(f"{field} must be a finite real number")
        return None
    if minimum is not None and result < minimum:
        errors.append(f"{field} is below its minimum")
    if maximum is not None and result > maximum:
        errors.append(f"{field} exceeds its maximum")
    return result


def _strict_bool(
    value: object,
    field: str,
    errors: list[str],
) -> bool | None:
    if type(value) is not bool:
        errors.append(f"{field} must be boolean")
        return None
    return value


def _strict_int_list(
    value: object,
    field: str,
    errors: list[str],
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be an integer array")
        return None
    parsed: list[int] = []
    for index, item in enumerate(value):
        result = _strict_int(
            item,
            f"{field}[{index}]",
            errors,
            minimum=minimum,
            maximum=maximum,
        )
        if result is not None:
            parsed.append(result)
    if len(parsed) != len(value):
        return None
    return tuple(parsed)


def _strict_float_list(
    value: object,
    field: str,
    errors: list[str],
    *,
    minimum: float,
) -> tuple[float, ...] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a numeric array")
        return None
    parsed: list[float] = []
    for index, item in enumerate(value):
        result = _strict_float(
            item,
            f"{field}[{index}]",
            errors,
            minimum=minimum,
        )
        if result is not None:
            parsed.append(result)
    if len(parsed) != len(value):
        return None
    return tuple(parsed)


@dataclass(frozen=True)
class _PrimitiveWindowMetrics:
    online_nll: float
    entry_frozen_nll: float
    learning_nll_gain: float
    learning_positive_fraction: float
    online_accuracy: float
    entry_frozen_accuracy: float
    learning_accuracy_gain: float
    masked_nll_increase: float
    masked_nll_positive_fraction: float


@dataclass(frozen=True)
class _ValidatedCriticalRunPrimitives:
    rewards: np.ndarray
    evidence_refresh: np.ndarray
    deployed_states: np.ndarray
    shadow_states: np.ndarray
    representation_link_valid: bool
    consumer_gate_valid: bool
    feature_memory_enabled: bool
    feature_memory_contract_valid: bool
    candidate_archive_valid: bool
    counter_contract_valid: bool
    causal_contract_valid: bool
    all_finite: bool
    windows: Mapping[str, _PrimitiveWindowMetrics]


def _decode_bool_payload(
    raw: object,
    expected_shape: tuple[int, ...],
    field: str,
    errors: list[str],
) -> np.ndarray | None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "shape",
        "bitorder",
        "data_base64",
    }:
        errors.append(f"{field} fields are invalid")
        return None
    if raw.get("shape") != list(expected_shape):
        errors.append(f"{field}.shape is invalid")
        return None
    if raw.get("bitorder") != "little":
        errors.append(f"{field}.bitorder must be little")
        return None
    encoded = raw.get("data_base64")
    if not isinstance(encoded, str):
        errors.append(f"{field}.data_base64 must be a string")
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        errors.append(f"{field}.data_base64 is invalid")
        return None
    if base64.b64encode(decoded).decode("ascii") != encoded:
        errors.append(f"{field}.data_base64 is not canonical")
        return None
    bit_count = math.prod(expected_shape)
    expected_bytes = (bit_count + 7) // 8
    if len(decoded) != expected_bytes:
        errors.append(f"{field}.data_base64 has the wrong byte length")
        return None
    unpacked = np.unpackbits(
        np.frombuffer(decoded, dtype=np.uint8),
        bitorder="little",
    )
    if np.any(unpacked[bit_count:]):
        errors.append(f"{field}.data_base64 has nonzero padding bits")
        return None
    return unpacked[:bit_count].astype(np.bool_).reshape(expected_shape)


def _decode_float32_xor_state_payload(
    raw: object,
    expected_shape: tuple[int, ...],
    field: str,
    errors: list[str],
) -> np.ndarray | None:
    """Decode one bounded, exact little-endian float32 state series."""
    expected_fields = {
        "shape",
        "dtype",
        "byteorder",
        "delta",
        "codec",
        "data_base64",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        errors.append(f"{field} fields are invalid")
        return None
    shape = raw.get("shape")
    if (
        not isinstance(shape, list)
        or any(type(value) is not int for value in shape)
        or shape != list(expected_shape)
    ):
        errors.append(f"{field}.shape is invalid")
        return None
    for name, expected in (
        ("dtype", "float32"),
        ("byteorder", "little"),
        ("delta", "uint32-xor"),
        ("codec", "zlib"),
    ):
        if raw.get(name) != expected:
            errors.append(f"{field}.{name} is invalid")
    encoded = raw.get("data_base64")
    if not isinstance(encoded, str):
        errors.append(f"{field}.data_base64 must be a string")
        return None

    expected_nbytes = math.prod(expected_shape) * np.dtype("<f4").itemsize
    # A valid DEFLATE stream has very small worst-case overhead. Bound the
    # compressed and base64 inputs before allocating either decoded payload.
    maximum_compressed_nbytes = (
        expected_nbytes + max(64, expected_nbytes // 1_000 + 64)
    )
    maximum_base64_length = 4 * ((maximum_compressed_nbytes + 2) // 3)
    if len(encoded) > maximum_base64_length:
        errors.append(f"{field}.data_base64 exceeds the compressed size bound")
        return None
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        errors.append(f"{field}.data_base64 is invalid")
        return None
    if base64.b64encode(compressed).decode("ascii") != encoded:
        errors.append(f"{field}.data_base64 is not canonical")
        return None
    if len(compressed) > maximum_compressed_nbytes:
        errors.append(f"{field} compressed payload exceeds its size bound")
        return None

    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(
            compressed,
            expected_nbytes + 1,
        )
    except zlib.error:
        errors.append(f"{field} is not a valid zlib stream")
        return None
    if len(decoded) > expected_nbytes:
        errors.append(f"{field} has the wrong uncompressed byte length")
        return None
    if decompressor.unconsumed_tail:
        errors.append(f"{field} zlib stream has unconsumed data")
        return None
    if len(decoded) != expected_nbytes:
        errors.append(f"{field} has the wrong uncompressed byte length")
        return None
    if not decompressor.eof:
        errors.append(f"{field} zlib stream did not reach EOF")
        return None
    if decompressor.unused_data:
        errors.append(f"{field} zlib stream has trailing data")
        return None

    deltas = np.frombuffer(decoded, dtype="<u4").reshape(expected_shape)
    state_bits = np.bitwise_xor.accumulate(deltas, axis=0)
    return state_bits.view("<f4")


def _parse_descriptor_bank(
    raw: object,
    field: str,
    errors: list[str],
) -> np.ndarray | None:
    if not isinstance(raw, list) or len(raw) != 12:
        errors.append(f"{field} must contain exactly 12 descriptors")
        return None
    parsed = np.empty((12, 2), dtype=np.int32)
    live_pairs: set[tuple[int, int]] = set()
    valid = True
    for slot, descriptor in enumerate(raw):
        if (
            not isinstance(descriptor, list)
            or len(descriptor) != 2
            or any(type(value) is not int for value in descriptor)
        ):
            errors.append(f"{field}[{slot}] must be an integer pair")
            valid = False
            continue
        left, right = cast(list[int], descriptor)
        inactive = left == -1 and right == -1
        live = 0 <= left < right < 12
        if not (inactive or live):
            errors.append(f"{field}[{slot}] is not a canonical descriptor")
            valid = False
            continue
        pair = (left, right)
        if live and pair in live_pairs:
            errors.append(f"{field} contains a duplicate live descriptor")
            valid = False
            continue
        if live:
            live_pairs.add(pair)
        parsed[slot] = pair
    return parsed if valid else None


def _validate_bank_state_rle(
    raw: object,
    state_count: int,
    field: str,
    errors: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(raw, list) or not raw:
        errors.append(f"{field} must be a non-empty RLE array")
        return None
    deployed = np.full((state_count, 12, 2), -2, dtype=np.int32)
    shadow = np.full((state_count, 12, 2), -2, dtype=np.int32)
    previous_end = 0
    previous_flat: tuple[int, ...] | None = None
    valid = True
    for index, interval in enumerate(raw):
        if not isinstance(interval, Mapping) or set(interval) != {
            "start",
            "end_exclusive",
            "deployed_descriptors",
            "shadow_descriptors",
        }:
            errors.append(f"{field}[{index}] fields are invalid")
            valid = False
            continue
        start = _strict_int(
            interval.get("start"),
            f"{field}[{index}].start",
            errors,
            minimum=0,
            maximum=state_count - 1,
        )
        end = _strict_int(
            interval.get("end_exclusive"),
            f"{field}[{index}].end_exclusive",
            errors,
            minimum=1,
            maximum=state_count,
        )
        deployed_bank = _parse_descriptor_bank(
            interval.get("deployed_descriptors"),
            f"{field}[{index}].deployed_descriptors",
            errors,
        )
        shadow_bank = _parse_descriptor_bank(
            interval.get("shadow_descriptors"),
            f"{field}[{index}].shadow_descriptors",
            errors,
        )
        if start is None or end is None or deployed_bank is None or shadow_bank is None:
            valid = False
            continue
        if start != previous_end or start >= end:
            errors.append(f"{field}[{index}] is noncontiguous or empty")
            valid = False
        flat = tuple(
            int(value)
            for value in np.concatenate((deployed_bank.reshape(-1), shadow_bank.reshape(-1)))
        )
        if flat == previous_flat:
            errors.append(f"{field}[{index}] is a noncanonical split")
            valid = False
        if start == previous_end and start < end:
            deployed[start:end] = deployed_bank
            shadow[start:end] = shadow_bank
        previous_end = end
        previous_flat = flat
    if previous_end != state_count:
        errors.append(f"{field} does not cover the complete life")
        valid = False
    if not valid or np.any(deployed == -2) or np.any(shadow == -2):
        return None
    return deployed, shadow


def _validate_candidate_bank_state_rle(
    raw: object,
    state_count: int,
    field: str,
    errors: list[str],
) -> bool:
    """Require the exact immutable lexicographic 66-pair archive."""
    if not isinstance(raw, list) or len(raw) != 1:
        errors.append(f"{field} must be one canonical full-life interval")
        return False
    interval = raw[0]
    if not isinstance(interval, Mapping) or set(interval) != {
        "start",
        "end_exclusive",
        "candidate_descriptors",
    }:
        errors.append(f"{field}[0] fields are invalid")
        return False
    if type(interval.get("start")) is not int or interval.get("start") != 0:
        errors.append(f"{field}[0].start must be zero")
    if (
        type(interval.get("end_exclusive")) is not int
        or interval.get("end_exclusive") != state_count
    ):
        errors.append(f"{field}[0].end_exclusive must cover every state")
    descriptors = interval.get("candidate_descriptors")
    canonical = [
        [left, right]
        for left in range(12)
        for right in range(left + 1, 12)
    ]
    if (
        not isinstance(descriptors, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            for pair in descriptors
        )
        or descriptors != canonical
    ):
        errors.append(
            f"{field}[0].candidate_descriptors must be the exact canonical archive"
        )
    return not any(error.startswith(field) for error in errors)


def _pair_slots_from_bank(
    states: np.ndarray,
    pair: tuple[int, int],
    field: str,
    errors: list[str],
) -> np.ndarray | None:
    matches = np.all(
        states == np.asarray(pair, dtype=np.int32),
        axis=-1,
    )
    counts = np.sum(matches, axis=1)
    if np.any(counts > 1):
        errors.append(f"{field} duplicates critical pair {pair!r}")
        return None
    return np.where(counts == 1, np.argmax(matches, axis=1), -1).astype(np.int32)


def _binary_margin_nll(margin: float, intended_action: int) -> float:
    sign = 2.0 * intended_action - 1.0
    return float(np.logaddexp(0.0, -sign * margin))


def _validate_critical_window(
    raw: object,
    *,
    pair: tuple[int, int],
    entry_step: int,
    window_start: int,
    window_end_exclusive: int,
    deployed_states: np.ndarray,
    consumer_mask_pre: np.ndarray,
    behavior_states: np.ndarray,
    field: str,
    errors: list[str],
) -> _PrimitiveWindowMetrics | None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "pair",
        "entry_step",
        "window_start",
        "window_end_exclusive",
        "entry_critical_weight_margin",
        "rows",
    }:
        errors.append(f"{field} fields are invalid")
        return None
    if raw.get("pair") != list(pair):
        errors.append(f"{field}.pair is invalid")
    for name, expected in (
        ("entry_step", entry_step),
        ("window_start", window_start),
        ("window_end_exclusive", window_end_exclusive),
    ):
        if raw.get(name) != expected or type(raw.get(name)) is not int:
            errors.append(f"{field}.{name} is invalid")
    entry_margin = _strict_float(
        raw.get("entry_critical_weight_margin"),
        f"{field}.entry_critical_weight_margin",
        errors,
    )
    rows = raw.get("rows")
    window_length = window_end_exclusive - window_start
    if not isinstance(rows, list) or len(rows) != window_length:
        errors.append(f"{field}.rows must contain exactly {window_length} entries")
        return None
    target = np.asarray(pair, dtype=np.int32)
    entry_matches = np.all(deployed_states[entry_step] == target, axis=1)
    if np.sum(entry_matches) > 1:
        errors.append(f"{field} entry duplicates the critical pair")
    expected_entry_margin = 0.0
    if np.any(entry_matches):
        entry_slot = int(np.argmax(entry_matches))
        expected_entry_margin = float(
            behavior_states[entry_step, 1, entry_slot]
            - behavior_states[entry_step, 0, entry_slot]
        )
    if entry_margin != expected_entry_margin:
        errors.append(f"{field} entry column margin disagrees with numeric state")

    online_losses: list[float] = []
    entry_losses: list[float] = []
    zero_losses: list[float] = []
    online_correct: list[float] = []
    entry_correct: list[float] = []
    valid = entry_margin is not None
    for offset, row in enumerate(rows):
        step = window_start + offset
        row_field = f"{field}.rows[{offset}]"
        if not isinstance(row, Mapping) or set(row) != {
            "step",
            "intended_action",
            "online_logit_margin",
            "critical_activation",
            "current_critical_weight_margin",
        }:
            errors.append(f"{row_field} fields are invalid")
            valid = False
            continue
        if type(row.get("step")) is not int or row.get("step") != step:
            errors.append(f"{row_field}.step is invalid")
            valid = False
        action = _strict_int(
            row.get("intended_action"),
            f"{row_field}.intended_action",
            errors,
            minimum=0,
            maximum=1,
        )
        margin = _strict_float(
            row.get("online_logit_margin"),
            f"{row_field}.online_logit_margin",
            errors,
        )
        activation = _strict_float(
            row.get("critical_activation"),
            f"{row_field}.critical_activation",
            errors,
        )
        current_margin = _strict_float(
            row.get("current_critical_weight_margin"),
            f"{row_field}.current_critical_weight_margin",
            errors,
        )
        matches = np.all(deployed_states[step] == target, axis=1)
        if np.sum(matches) > 1:
            errors.append(f"{row_field} duplicates the critical pair")
            valid = False
        elif not np.any(matches):
            if activation != 0.0 or current_margin != 0.0:
                errors.append(f"{row_field} absent pair must have zero activation and margin")
                valid = False
        elif activation is not None:
            slot = int(np.argmax(matches))
            if not bool(consumer_mask_pre[step, slot]) and activation != 0.0:
                errors.append(f"{row_field} closed consumer read is nonzero")
                valid = False
            if current_margin is not None:
                expected_current_margin = float(
                    behavior_states[step, 1, slot]
                    - behavior_states[step, 0, slot]
                )
                if current_margin != expected_current_margin:
                    errors.append(
                        f"{row_field} column margin disagrees with numeric state"
                    )
                    valid = False
        if (
            action is None
            or margin is None
            or activation is None
            or current_margin is None
            or entry_margin is None
        ):
            valid = False
            continue
        zero_margin = margin - current_margin * activation
        frozen_margin = zero_margin + entry_margin * activation
        online_losses.append(_binary_margin_nll(margin, action))
        zero_losses.append(_binary_margin_nll(zero_margin, action))
        entry_losses.append(_binary_margin_nll(frozen_margin, action))
        online_correct.append(float((1 if margin > 0.0 else 0) == action))
        entry_correct.append(float((1 if frozen_margin > 0.0 else 0) == action))
    if not valid or len(online_losses) != window_length:
        return None
    online_array = np.asarray(online_losses, dtype=np.float64)
    entry_array = np.asarray(entry_losses, dtype=np.float64)
    zero_array = np.asarray(zero_losses, dtype=np.float64)
    learning_gain = entry_array - online_array
    mask_gain = zero_array - online_array
    online_accuracy = float(np.mean(online_correct))
    frozen_accuracy = float(np.mean(entry_correct))
    return _PrimitiveWindowMetrics(
        online_nll=float(np.mean(online_array)),
        entry_frozen_nll=float(np.mean(entry_array)),
        learning_nll_gain=float(np.mean(learning_gain)),
        learning_positive_fraction=float(np.mean(learning_gain > 0.0)),
        online_accuracy=online_accuracy,
        entry_frozen_accuracy=frozen_accuracy,
        learning_accuracy_gain=online_accuracy - frozen_accuracy,
        masked_nll_increase=float(np.mean(mask_gain)),
        masked_nll_positive_fraction=float(np.mean(mask_gain > 0.0)),
    )


def _expected_consumer_gate_arrays(
    deployed_states: np.ndarray,
    evidence: np.ndarray,
    *,
    write_confirmation_steps: int,
    read_confirmation_steps: int,
    read_lease_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact gates by descriptor identity without a stepwise Python scan."""
    cycle_steps = evidence.shape[0]
    state_count = cycle_steps + 1
    left = deployed_states[..., 0]
    right = deployed_states[..., 1]
    live = (0 <= left) & (left < right) & (right < 12)
    # Lexicographic ``combinations(range(12), 2)`` index: left*(23-left)//2
    # counts every pair whose first element precedes ``left``, and
    # (right-left-1) offsets within that block. This independently yields the
    # pinned C=1 and D=38 candidate identities.
    identity = np.where(
        live,
        left * (23 - left) // 2 + (right - left - 1),
        -1,
    ).astype(np.int32)
    pre_identity = identity[:-1]
    post_identity = identity[1:]
    live_pre = live[:-1]
    live_post = live[1:]

    slot_by_state_identity = np.full(
        (state_count, 66),
        -1,
        dtype=np.int8,
    )
    state_rows, state_slots = np.nonzero(live)
    live_ids = identity[state_rows, state_slots]
    slot_by_state_identity[state_rows, live_ids] = state_slots.astype(
        np.int8,
    )

    evidence_by_identity = np.zeros((cycle_steps, 66), dtype=np.bool_)
    rows, slots = np.nonzero(live_pre)
    evidence_by_identity[rows, pre_identity[rows, slots]] = evidence[rows, slots]
    write_by_identity = np.zeros_like(evidence_by_identity)
    mask_by_state_identity = np.zeros((state_count, 66), dtype=np.bool_)

    for descriptor_identity in np.unique(live_ids):
        present = slot_by_state_identity[:, descriptor_identity] >= 0
        padded = np.concatenate(
            (
                np.asarray((False,), dtype=np.bool_),
                present,
                np.asarray((False,), dtype=np.bool_),
            )
        )
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end_exclusive in zip(starts, ends, strict=True):
            start_int = int(start)
            end_int = int(end_exclusive)
            transition_end = min(end_int, cycle_steps)
            if start_int >= transition_end:
                continue
            identity_evidence = evidence_by_identity[
                start_int:transition_end,
                descriptor_identity,
            ]
            offsets = np.arange(identity_evidence.size, dtype=np.int64)
            last_false = np.maximum.accumulate(
                np.where(identity_evidence, -1, offsets)
            )
            consecutive = offsets - last_false
            write_by_identity[
                start_int:transition_end,
                descriptor_identity,
            ] = identity_evidence & (
                consecutive >= write_confirmation_steps
            )
            read_acquire = identity_evidence & (
                consecutive >= read_confirmation_steps
            )

            # Only transitions whose post-state retains this identity carry a
            # lease. Evidence events form clusters bridged by at most the
            # configured number of idle transitions. The first qualifying
            # acquisition in a cluster opens reads at the next decision.
            surviving_count = max(
                min(end_int - 1, cycle_steps) - start_int,
                0,
            )
            surviving_evidence = identity_evidence[:surviving_count]
            surviving_acquire = read_acquire[:surviving_count]
            evidence_offsets = np.flatnonzero(surviving_evidence)
            if evidence_offsets.size == 0:
                continue
            cluster_starts = np.concatenate(
                (
                    np.asarray((0,), dtype=np.int64),
                    np.flatnonzero(
                        np.diff(evidence_offsets) > read_lease_steps + 1
                    )
                    + 1,
                )
            )
            cluster_ends = np.concatenate(
                (
                    cluster_starts[1:],
                    np.asarray((evidence_offsets.size,), dtype=np.int64),
                )
            )
            for cluster_start, cluster_end in zip(
                cluster_starts,
                cluster_ends,
                strict=True,
            ):
                cluster_events = evidence_offsets[cluster_start:cluster_end]
                acquisition_events = cluster_events[
                    surviving_acquire[cluster_events]
                ]
                if acquisition_events.size == 0:
                    continue
                open_state = start_int + int(acquisition_events[0]) + 1
                close_state = min(
                    end_int,
                    start_int
                    + int(cluster_events[-1])
                    + read_lease_steps
                    + 2,
                )
                mask_by_state_identity[
                    open_state:close_state,
                    descriptor_identity,
                ] = True

    expected_write = np.zeros((cycle_steps, 12), dtype=np.bool_)
    expected_mask_pre = np.zeros((cycle_steps, 12), dtype=np.bool_)
    expected_mask_post = np.zeros((cycle_steps, 12), dtype=np.bool_)
    pre_rows, pre_slots = np.nonzero(live_pre)
    pre_ids = pre_identity[pre_rows, pre_slots]
    expected_write[pre_rows, pre_slots] = write_by_identity[
        pre_rows,
        pre_ids,
    ]
    expected_mask_pre[pre_rows, pre_slots] = mask_by_state_identity[
        pre_rows,
        pre_ids,
    ]
    post_rows, post_slots = np.nonzero(live_post)
    post_ids = post_identity[post_rows, post_slots]
    expected_mask_post[post_rows, post_slots] = mask_by_state_identity[
        post_rows + 1,
        post_ids,
    ]
    return expected_write, expected_mask_pre, expected_mask_post


def _expected_feature_memory_arrays(
    shadow_states: np.ndarray,
    raw_evidence: np.ndarray,
    *,
    confirmation_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct confirmed memory state by descriptor identity.

    Commitment begins only after a descriptor receives the configured number
    of consecutive evidence events. It follows a surviving descriptor across
    slot moves and resets whenever that identity leaves the active bank.
    """
    cycle_steps = raw_evidence.shape[0]
    state_count = cycle_steps + 1
    left = shadow_states[..., 0]
    right = shadow_states[..., 1]
    live = (0 <= left) & (left < right) & (right < 12)
    # Same lexicographic ``combinations(range(12), 2)`` identity index as in
    # _expected_consumer_gate_arrays; being slot-independent, it lets
    # commitment follow a surviving descriptor across slot moves.
    identity = np.where(
        live,
        left * (23 - left) // 2 + (right - left - 1),
        -1,
    ).astype(np.int32)
    pre_identity = identity[:-1]
    post_identity = identity[1:]
    live_pre = live[:-1]
    live_post = live[1:]

    slot_by_state_identity = np.full(
        (state_count, 66),
        -1,
        dtype=np.int8,
    )
    state_rows, state_slots = np.nonzero(live)
    live_ids = identity[state_rows, state_slots]
    slot_by_state_identity[state_rows, live_ids] = state_slots.astype(
        np.int8,
    )

    evidence_by_identity = np.zeros((cycle_steps, 66), dtype=np.bool_)
    pre_rows, pre_slots = np.nonzero(live_pre)
    pre_ids = pre_identity[pre_rows, pre_slots]
    evidence_by_identity[pre_rows, pre_ids] = raw_evidence[
        pre_rows,
        pre_slots,
    ]
    confirmed_by_identity = np.zeros_like(evidence_by_identity)
    committed_by_state_identity = np.zeros(
        (state_count, 66),
        dtype=np.bool_,
    )

    for descriptor_identity in np.unique(live_ids):
        present = slot_by_state_identity[:, descriptor_identity] >= 0
        padded = np.concatenate(
            (
                np.asarray((False,), dtype=np.bool_),
                present,
                np.asarray((False,), dtype=np.bool_),
            )
        )
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end_exclusive in zip(starts, ends, strict=True):
            start_int = int(start)
            end_int = int(end_exclusive)
            transition_end = min(end_int, cycle_steps)
            if start_int >= transition_end:
                continue
            identity_evidence = evidence_by_identity[
                start_int:transition_end,
                descriptor_identity,
            ]
            offsets = np.arange(identity_evidence.size, dtype=np.int64)
            last_false = np.maximum.accumulate(
                np.where(identity_evidence, -1, offsets)
            )
            consecutive = offsets - last_false
            confirmed = identity_evidence & (
                consecutive >= confirmation_steps
            )
            confirmed_by_identity[
                start_int:transition_end,
                descriptor_identity,
            ] = confirmed

            # A confirmation on transition t commits the head in state t+1,
            # provided the same descriptor survives into that state.
            surviving_transition_count = max(
                min(end_int - 1, cycle_steps) - start_int,
                0,
            )
            if surviving_transition_count:
                committed_by_state_identity[
                    start_int + 1 : start_int + 1 + surviving_transition_count,
                    descriptor_identity,
                ] = np.maximum.accumulate(
                    confirmed[:surviving_transition_count]
                )

    expected_confirmed = np.zeros((cycle_steps, 12), dtype=np.bool_)
    expected_committed_pre = np.zeros_like(expected_confirmed)
    expected_committed_post = np.zeros_like(expected_confirmed)
    expected_confirmed[pre_rows, pre_slots] = confirmed_by_identity[
        pre_rows,
        pre_ids,
    ]
    expected_committed_pre[pre_rows, pre_slots] = (
        committed_by_state_identity[pre_rows, pre_ids]
    )
    post_rows, post_slots = np.nonzero(live_post)
    post_ids = post_identity[post_rows, post_slots]
    expected_committed_post[post_rows, post_slots] = (
        committed_by_state_identity[post_rows + 1, post_ids]
    )

    identity_survives = np.zeros((cycle_steps, 12), dtype=np.bool_)
    identity_survives[pre_rows, pre_slots] = (
        slot_by_state_identity[pre_rows + 1, pre_ids] >= 0
    )
    return (
        expected_confirmed,
        expected_committed_pre,
        expected_committed_post,
        identity_survives,
    )


def _float32_bit_view(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype="<f4").view("<u4")


def _identity_routed_feature_head_changes(
    shadow_states: np.ndarray,
    feature_head_states: np.ndarray,
) -> np.ndarray:
    """Derive exact bitwise head changes for surviving identities."""
    shadow_pre = shadow_states[:-1]
    shadow_post = shadow_states[1:]
    live_pre = np.all(shadow_pre >= 0, axis=2)
    live_post = np.all(shadow_post >= 0, axis=2)
    pre_codes = shadow_pre[..., 0] * 12 + shadow_pre[..., 1]
    post_codes = shadow_post[..., 0] * 12 + shadow_post[..., 1]
    identity_matches = (
        (post_codes[:, :, None] == pre_codes[:, None, :])
        & live_post[:, :, None]
        & live_pre[:, None, :]
    )
    destination_for_pre = np.argmax(identity_matches, axis=1)
    pre_has_destination = np.any(identity_matches, axis=1)
    pre_bits = _float32_bit_view(feature_head_states[:-1])
    post_bits = _float32_bit_view(feature_head_states[1:])
    routed_post_bits = np.take_along_axis(
        post_bits,
        destination_for_pre[:, None, :],
        axis=2,
    )
    return (
        live_pre
        & pre_has_destination
        & np.any(pre_bits != routed_post_bits, axis=1)
    )


def _consumer_write_violation_bits(
    deployed_states: np.ndarray,
    write_gate: np.ndarray,
    behavior_states: np.ndarray,
    control_q_states: np.ndarray,
    control_q_trace_states: np.ndarray,
    *,
    evidence_gated: bool,
) -> np.ndarray:
    """Audit every persisted consumer column from exact numeric states."""
    cycle_steps = write_gate.shape[0]
    deployed_pre = deployed_states[:-1]
    deployed_post = deployed_states[1:]
    live_pre = np.all(deployed_pre >= 0, axis=2)
    live_post = np.all(deployed_post >= 0, axis=2)
    vacancy_pre = np.all(deployed_pre == -1, axis=2)
    pre_codes = deployed_pre[..., 0] * 12 + deployed_pre[..., 1]
    post_codes = deployed_post[..., 0] * 12 + deployed_post[..., 1]
    identity_matches = (
        (post_codes[:, :, None] == pre_codes[:, None, :])
        & live_post[:, :, None]
        & live_pre[:, None, :]
    )
    post_has_source = np.any(identity_matches, axis=2)
    destination_for_pre = np.argmax(identity_matches, axis=1)
    pre_has_destination = np.any(identity_matches, axis=1)

    state_arrays = (
        behavior_states,
        control_q_states,
        control_q_trace_states,
    )
    state_bits = tuple(_float32_bit_view(values) for values in state_arrays)
    violations = np.zeros((cycle_steps, 12), dtype=np.bool_)
    for values in state_arrays:
        violations |= np.any(~np.isfinite(values[:-1]), axis=1)
        violations |= np.any(~np.isfinite(values[1:]), axis=1)

    # All initial columns, every vacancy, and every post-state identity without
    # a pre-state source must contain exact positive-zero bits. This rejects
    # negative zero as durable state rather than treating it as numeric zero.
    for bits in state_bits:
        violations[0] |= np.any(bits[0] != 0, axis=0)
        violations |= vacancy_pre & np.any(bits[:-1] != 0, axis=1)
        violations |= (~post_has_source) & np.any(bits[1:] != 0, axis=1)

    if evidence_gated:
        closed_survivor = live_pre & pre_has_destination & ~write_gate
        for bits in state_bits[:2]:
            routed_post = np.take_along_axis(
                bits[1:],
                destination_for_pre[:, None, :],
                axis=2,
            )
            violations |= closed_survivor & np.any(
                bits[:-1] != routed_post,
                axis=1,
            )
        routed_trace_post = np.take_along_axis(
            state_bits[2][1:],
            destination_for_pre[:, None, :],
            axis=2,
        )
        violations |= closed_survivor & np.any(
            routed_trace_post != 0,
            axis=1,
        )
    return violations


def _validate_critical_run_primitives(
    raw: object,
    *,
    cycle_steps: int,
    segment_lengths: tuple[int, ...],
    consumer_write_confirmation_steps: int,
    consumer_read_confirmation_steps: int,
    consumer_read_lease_steps: int,
    feature_memory_enabled: bool,
    feature_evidence_confirmation_steps: int,
    field: str,
    errors: list[str],
) -> _ValidatedCriticalRunPrimitives | None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "cycle_steps",
        "reward_one_bits",
        "evidence_refresh_bits",
        "retention_evidence_refresh_bits",
        "feature_memory_committed_pre_bits",
        "feature_memory_committed_post_bits",
        "identity_routed_head_changed_bits",
        "feature_memory_contract_violation_bits",
        "feature_memory_enabled",
        "feature_head_state_xor",
        "consumer_write_gate_bits",
        "consumer_write_contract_violation_bits",
        "behavior_pair_weight_state_xor",
        "control_q_pair_weight_state_xor",
        "control_q_trace_state_xor",
        "consumer_active_mask_pre_bits",
        "consumer_active_mask_post_bits",
        "closed_consumer_read_violation_bits",
        "representation_link_violation_bits",
        "counter_contract_violation_bits",
        "causal_contract_violation_bits",
        "finite_violation_bits",
        "bank_state_rle",
        "candidate_bank_state_rle",
        "critical_windows",
    }:
        errors.append(f"{field} fields are invalid")
        return None
    if raw.get("schema_version") != CRITICAL_RUN_PRIMITIVES_SCHEMA:
        errors.append(f"{field}.schema_version is unsupported")
    if type(raw.get("cycle_steps")) is not int or raw.get("cycle_steps") != cycle_steps:
        errors.append(f"{field}.cycle_steps is invalid")
    rewards = _decode_bool_payload(
        raw.get("reward_one_bits"),
        (cycle_steps,),
        f"{field}.reward_one_bits",
        errors,
    )
    evidence = _decode_bool_payload(
        raw.get("evidence_refresh_bits"),
        (cycle_steps, 12),
        f"{field}.evidence_refresh_bits",
        errors,
    )
    confirmed_evidence = _decode_bool_payload(
        raw.get("retention_evidence_refresh_bits"),
        (cycle_steps, 12),
        f"{field}.retention_evidence_refresh_bits",
        errors,
    )
    feature_committed_pre = _decode_bool_payload(
        raw.get("feature_memory_committed_pre_bits"),
        (cycle_steps, 12),
        f"{field}.feature_memory_committed_pre_bits",
        errors,
    )
    feature_committed_post = _decode_bool_payload(
        raw.get("feature_memory_committed_post_bits"),
        (cycle_steps, 12),
        f"{field}.feature_memory_committed_post_bits",
        errors,
    )
    identity_routed_head_changed = _decode_bool_payload(
        raw.get("identity_routed_head_changed_bits"),
        (cycle_steps, 12),
        f"{field}.identity_routed_head_changed_bits",
        errors,
    )
    feature_memory_violations = _decode_bool_payload(
        raw.get("feature_memory_contract_violation_bits"),
        (cycle_steps, 12),
        f"{field}.feature_memory_contract_violation_bits",
        errors,
    )
    primitive_feature_memory_enabled = _strict_bool(
        raw.get("feature_memory_enabled"),
        f"{field}.feature_memory_enabled",
        errors,
    )
    feature_head_states = _decode_float32_xor_state_payload(
        raw.get("feature_head_state_xor"),
        (cycle_steps + 1, 1, 12),
        f"{field}.feature_head_state_xor",
        errors,
    )
    write_gate = _decode_bool_payload(
        raw.get("consumer_write_gate_bits"),
        (cycle_steps, 12),
        f"{field}.consumer_write_gate_bits",
        errors,
    )
    consumer_write_violations = _decode_bool_payload(
        raw.get("consumer_write_contract_violation_bits"),
        (cycle_steps, 12),
        f"{field}.consumer_write_contract_violation_bits",
        errors,
    )
    behavior_states = _decode_float32_xor_state_payload(
        raw.get("behavior_pair_weight_state_xor"),
        (cycle_steps + 1, 2, 12),
        f"{field}.behavior_pair_weight_state_xor",
        errors,
    )
    control_q_states = _decode_float32_xor_state_payload(
        raw.get("control_q_pair_weight_state_xor"),
        (cycle_steps + 1, 2, 12),
        f"{field}.control_q_pair_weight_state_xor",
        errors,
    )
    control_q_trace_states = _decode_float32_xor_state_payload(
        raw.get("control_q_trace_state_xor"),
        (cycle_steps + 1, 2, 12),
        f"{field}.control_q_trace_state_xor",
        errors,
    )
    mask_pre = _decode_bool_payload(
        raw.get("consumer_active_mask_pre_bits"),
        (cycle_steps, 12),
        f"{field}.consumer_active_mask_pre_bits",
        errors,
    )
    mask_post = _decode_bool_payload(
        raw.get("consumer_active_mask_post_bits"),
        (cycle_steps, 12),
        f"{field}.consumer_active_mask_post_bits",
        errors,
    )
    closed_read_violations = _decode_bool_payload(
        raw.get("closed_consumer_read_violation_bits"),
        (cycle_steps, 12),
        f"{field}.closed_consumer_read_violation_bits",
        errors,
    )
    link_violations = _decode_bool_payload(
        raw.get("representation_link_violation_bits"),
        (cycle_steps,),
        f"{field}.representation_link_violation_bits",
        errors,
    )
    counter_violations = _decode_bool_payload(
        raw.get("counter_contract_violation_bits"),
        (cycle_steps,),
        f"{field}.counter_contract_violation_bits",
        errors,
    )
    causal_violations = _decode_bool_payload(
        raw.get("causal_contract_violation_bits"),
        (cycle_steps,),
        f"{field}.causal_contract_violation_bits",
        errors,
    )
    finite_violations = _decode_bool_payload(
        raw.get("finite_violation_bits"),
        (cycle_steps,),
        f"{field}.finite_violation_bits",
        errors,
    )
    bank_states = _validate_bank_state_rle(
        raw.get("bank_state_rle"),
        cycle_steps + 1,
        f"{field}.bank_state_rle",
        errors,
    )
    candidate_archive_valid = _validate_candidate_bank_state_rle(
        raw.get("candidate_bank_state_rle"),
        cycle_steps + 1,
        f"{field}.candidate_bank_state_rle",
        errors,
    )
    if any(
        value is None
        for value in (
            rewards,
            evidence,
            confirmed_evidence,
            feature_committed_pre,
            feature_committed_post,
            identity_routed_head_changed,
            feature_memory_violations,
            primitive_feature_memory_enabled,
            feature_head_states,
            write_gate,
            consumer_write_violations,
            behavior_states,
            control_q_states,
            control_q_trace_states,
            mask_pre,
            mask_post,
            closed_read_violations,
            link_violations,
            counter_violations,
            causal_violations,
            finite_violations,
            bank_states,
        )
    ):
        return None
    assert rewards is not None
    assert evidence is not None
    assert confirmed_evidence is not None
    assert feature_committed_pre is not None
    assert feature_committed_post is not None
    assert identity_routed_head_changed is not None
    assert feature_memory_violations is not None
    assert primitive_feature_memory_enabled is not None
    assert feature_head_states is not None
    assert write_gate is not None
    assert consumer_write_violations is not None
    assert behavior_states is not None
    assert control_q_states is not None
    assert control_q_trace_states is not None
    assert mask_pre is not None
    assert mask_post is not None
    assert closed_read_violations is not None
    assert link_violations is not None
    assert counter_violations is not None
    assert causal_violations is not None
    assert finite_violations is not None
    assert bank_states is not None
    deployed_states, shadow_states = bank_states

    deployed_pre = deployed_states[:-1]
    deployed_post = deployed_states[1:]
    shadow_pre = shadow_states[:-1]
    shadow_post = shadow_states[1:]
    live_pre = np.all(deployed_pre >= 0, axis=2)
    if np.any(evidence & ~live_pre):
        errors.append(f"{field} refreshes evidence for an inactive descriptor")

    (
        expected_confirmed_evidence,
        expected_feature_committed_pre,
        expected_feature_committed_post,
        _,
    ) = _expected_feature_memory_arrays(
        shadow_states,
        evidence,
        confirmation_steps=feature_evidence_confirmation_steps,
    )
    reconstructed_feature_violations = np.zeros(
        (cycle_steps, 12),
        dtype=np.bool_,
    )
    shadow_live_pre = np.all(shadow_pre >= 0, axis=2)
    reconstructed_feature_violations |= evidence & ~shadow_live_pre
    reconstructed_feature_violations |= (
        confirmed_evidence != expected_confirmed_evidence
    )
    reconstructed_feature_violations |= (
        feature_committed_pre != expected_feature_committed_pre
    )
    reconstructed_feature_violations |= (
        feature_committed_post != expected_feature_committed_post
    )
    reconstructed_feature_violations |= np.any(
        deployed_pre != shadow_pre,
        axis=2,
    )
    reconstructed_feature_violations |= np.any(
        deployed_post != shadow_post,
        axis=2,
    )
    expected_head_changed = _identity_routed_feature_head_changes(
        shadow_states,
        feature_head_states,
    )
    reconstructed_feature_violations |= np.any(
        ~np.isfinite(feature_head_states[:-1]),
        axis=1,
    )
    reconstructed_feature_violations |= np.any(
        ~np.isfinite(feature_head_states[1:]),
        axis=1,
    )
    reconstructed_feature_violations |= (
        expected_feature_committed_pre
        & ~expected_confirmed_evidence
        & expected_head_changed
    )
    feature_enabled_valid = bool(
        feature_memory_enabled
        and primitive_feature_memory_enabled is feature_memory_enabled
    )
    feature_memory_contract_valid = bool(
        feature_enabled_valid
        and np.array_equal(
            confirmed_evidence,
            expected_confirmed_evidence,
        )
        and np.array_equal(
            feature_committed_pre,
            expected_feature_committed_pre,
        )
        and np.array_equal(
            feature_committed_post,
            expected_feature_committed_post,
        )
        and np.array_equal(
            identity_routed_head_changed,
            expected_head_changed,
        )
        and np.array_equal(
            feature_memory_violations,
            reconstructed_feature_violations,
        )
        and not np.any(feature_memory_violations)
    )

    (
        expected_write,
        expected_pre_mask,
        expected_post_mask,
    ) = _expected_consumer_gate_arrays(
        deployed_states,
        evidence,
        write_confirmation_steps=consumer_write_confirmation_steps,
        read_confirmation_steps=consumer_read_confirmation_steps,
        read_lease_steps=consumer_read_lease_steps,
    )
    expected_consumer_write_violations = _consumer_write_violation_bits(
        deployed_states,
        write_gate,
        behavior_states,
        control_q_states,
        control_q_trace_states,
        evidence_gated=True,
    )
    consumer_gate_valid = bool(
        not np.any(mask_pre[0])
        and np.array_equal(write_gate, expected_write)
        and (cycle_steps == 1 or np.array_equal(mask_pre[1:], mask_post[:-1]))
        and np.array_equal(mask_pre, expected_pre_mask)
        and np.array_equal(mask_post, expected_post_mask)
        and not np.any(closed_read_violations)
        and np.array_equal(
            consumer_write_violations,
            expected_consumer_write_violations,
        )
        and not np.any(consumer_write_violations)
    )
    representation_valid = bool(
        not np.any(link_violations) and np.array_equal(deployed_states, shadow_states)
    )

    ends = np.cumsum((0, *segment_lengths), dtype=np.int64)
    d_start, d_end = int(ends[3]), int(ends[4])
    c_start, c_end = int(ends[5]), int(ends[6])
    recurrent_c_start = int(ends[8])
    expected_windows = {
        "c_first_late": (
            (0, 2),
            c_start,
            c_end - FEATURE_LEARNING_WINDOW,
            c_end,
        ),
        "d_late": (
            (4, 5),
            d_start,
            d_end - FEATURE_LEARNING_WINDOW,
            d_end,
        ),
        "c_recurrent_early": (
            (0, 2),
            c_start,
            recurrent_c_start,
            recurrent_c_start + RECURRENT_ENTRY_WINDOW,
        ),
    }
    windows_raw = raw.get("critical_windows")
    if not isinstance(windows_raw, Mapping) or set(windows_raw) != set(expected_windows):
        errors.append(f"{field}.critical_windows fields are invalid")
        return None
    windows: dict[str, _PrimitiveWindowMetrics] = {}
    for name, (
        pair,
        entry_step,
        window_start,
        window_end,
    ) in expected_windows.items():
        metrics = _validate_critical_window(
            windows_raw.get(name),
            pair=pair,
            entry_step=entry_step,
            window_start=window_start,
            window_end_exclusive=window_end,
            deployed_states=deployed_states,
            consumer_mask_pre=mask_pre,
            behavior_states=behavior_states,
            field=f"{field}.critical_windows.{name}",
            errors=errors,
        )
        if metrics is not None:
            windows[name] = metrics
    if len(windows) != len(expected_windows):
        return None
    return _ValidatedCriticalRunPrimitives(
        rewards=rewards.astype(np.float64),
        evidence_refresh=evidence,
        deployed_states=deployed_states,
        shadow_states=shadow_states,
        representation_link_valid=representation_valid,
        consumer_gate_valid=consumer_gate_valid,
        feature_memory_enabled=feature_enabled_valid,
        feature_memory_contract_valid=feature_memory_contract_valid,
        candidate_archive_valid=candidate_archive_valid,
        counter_contract_valid=not bool(np.any(counter_violations)),
        causal_contract_valid=not bool(np.any(causal_violations)),
        all_finite=not bool(np.any(finite_violations)),
        windows=windows,
    )


def _target_acquisition_from_rle(
    evidence_refresh_steps: tuple[int, ...],
    present_states: np.ndarray,
    start: int,
    end_exclusive: int,
) -> int | None:
    for event_step in evidence_refresh_steps:
        effective_step = event_step + 1
        confirmation_end = effective_step + FEATURE_LEARNING_WINDOW
        if (
            start <= event_step < end_exclusive
            and confirmation_end <= end_exclusive
            and bool(present_states[event_step])
            and bool(np.all(present_states[event_step:confirmation_end]))
        ):
            return effective_step
    return None


def _transition_count(values: np.ndarray, *, rising: bool) -> int:
    previous = np.asarray(values[:-1], dtype=np.bool_)
    current = np.asarray(values[1:], dtype=np.bool_)
    selected = (~previous & current) if rising else (previous & ~current)
    return int(np.sum(selected))


def _derived_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, Real)
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    return type(actual) is type(expected) and actual == expected


def _validate_lifecycle(
    raw: object,
    summary_cycle_steps: int,
    segment_lengths: tuple[int, ...],
    primitives: _ValidatedCriticalRunPrimitives,
    field: str,
    errors: list[str],
) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping):
        errors.append(f"{field} must be an object")
        return None
    expected_fields = {item.name for item in dataclasses.fields(CriticalLifecycleV2Summary)}
    if set(raw) != expected_fields:
        errors.append(f"{field} fields are invalid")
        return None

    before = len(errors)
    cycle_steps = _strict_int(
        raw.get("cycle_steps"),
        f"{field}.cycle_steps",
        errors,
        minimum=1,
    )
    state_count = _strict_int(
        raw.get("decision_state_count"),
        f"{field}.decision_state_count",
        errors,
        minimum=2,
    )
    if cycle_steps != summary_cycle_steps:
        errors.append(f"{field}.cycle_steps is inconsistent with the run")
    if cycle_steps is not None and state_count != cycle_steps + 1:
        errors.append(f"{field}.decision_state_count must equal cycle_steps + 1")
    if cycle_steps is None or state_count is None:
        return None

    c_rle = _validate_rle(
        raw.get("c_lifecycle_rle"),
        state_count,
        f"{field}.c_lifecycle_rle",
        errors,
    )
    d_rle = _validate_rle(
        raw.get("d_lifecycle_rle"),
        state_count,
        f"{field}.d_lifecycle_rle",
        errors,
    )

    required_ints = {
        "c_shadow_deployed_mismatch_steps": (0, state_count),
        "d_shadow_deployed_mismatch_steps": (0, state_count),
        "c_survival_end_exclusive": (0, cycle_steps),
        "c_evictions_after_acquisition": (0, cycle_steps),
        "c_repromotions_after_acquisition": (0, cycle_steps),
        "d_post_exit_live_slot_steps": (0, cycle_steps),
        "d_post_exit_promotion_count": (0, cycle_steps),
        "d_repromotions_after_retirement": (0, cycle_steps),
        "d_retirement_event_count": (0, cycle_steps),
        "d_matching_candidate_reset_count": (0, cycle_steps),
    }
    for name, bounds in required_ints.items():
        _strict_int(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=bounds[0],
            maximum=bounds[1],
        )
    optional_ints = {
        "c_acquisition_step": (0, cycle_steps),
        "c_survival_gap_steps": (0, cycle_steps),
        "c_first_survival_gap_step": (0, cycle_steps - 1),
        "d_acquisition_step": (0, cycle_steps),
        "d_retirement_event_step": (0, cycle_steps - 1),
        "d_retirement_step": (0, cycle_steps),
        "d_retirement_event_latency_steps": (0, cycle_steps),
        "d_retirement_latency_steps": (0, cycle_steps),
        "d_linked_matching_candidate_reset_count": (0, 1),
        "d_linked_candidate_age_post": (0, 2**31 - 1),
    }
    for name, bounds in optional_ints.items():
        _strict_int(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=bounds[0],
            maximum=bounds[1],
            optional=True,
        )

    unit_floats = (
        "c_first_late_reward",
        "c_first_late_intended_accuracy",
        "c_critical_column_learning_positive_fraction",
        "c_first_late_entry_frozen_critical_accuracy",
        "c_first_late_masked_nll_positive_fraction",
        "c_recurrent_early_reward",
        "c_recurrent_early_intended_accuracy",
        "c_recurrent_early_masked_nll_positive_fraction",
        "d_late_reward",
        "d_late_intended_accuracy",
        "d_critical_column_learning_positive_fraction",
        "d_late_entry_frozen_critical_accuracy",
        "d_late_masked_nll_positive_fraction",
        "d_post_exit_live_fraction",
    )
    for name in unit_floats:
        _strict_float(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
    for name in (
        "c_critical_column_learning_nll_gain",
        "c_critical_column_learning_accuracy_gain",
        "c_critical_column_target_created_share",
        "c_first_late_masked_nll_increase",
        "c_recurrent_early_excess_reward_retention",
        "c_recurrent_early_masked_nll_increase",
        "d_critical_column_learning_nll_gain",
        "d_critical_column_learning_accuracy_gain",
        "d_critical_column_target_created_share",
        "d_late_masked_nll_increase",
    ):
        _strict_float(
            raw.get(name),
            f"{field}.{name}",
            errors,
        )
    for name in (
        "c_first_late_online_nll",
        "c_first_late_entry_frozen_critical_nll",
        "d_late_online_nll",
        "d_late_entry_frozen_critical_nll",
    ):
        _strict_float(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=0.0,
        )
    for name in (
        "c_candidate_utility_at_life_end",
        "d_candidate_utility_at_life_end",
    ):
        _strict_float(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=0.0,
        )
    for name in (
        "d_linked_candidate_utility_post",
        "d_linked_candidate_head_linf_post",
    ):
        _strict_float(
            raw.get(name),
            f"{field}.{name}",
            errors,
            minimum=0.0,
            optional=True,
        )

    bool_fields = (
        "representation_link_contract_valid",
        "consumer_gate_contract_valid",
        "feature_memory_enabled",
        "feature_memory_contract_valid",
        "c_task_learned",
        "c_continuously_survived",
        "c_retained_and_used",
        "d_deployed_through_exit",
        "d_task_learned",
        "d_absent_entire_final_window",
        "d_retirement_event_aligned",
        "d_learned_then_stably_retired",
        "joint_memory_management_success",
        "candidate_archive_contract_valid",
    )
    for name in bool_fields:
        _strict_bool(
            raw.get(name),
            f"{field}.{name}",
            errors,
        )

    c_promotions = _strict_int_list(
        raw.get("c_promotion_event_steps"),
        f"{field}.c_promotion_event_steps",
        errors,
        minimum=0,
        maximum=cycle_steps - 1,
    )
    d_promotions = _strict_int_list(
        raw.get("d_promotion_event_steps"),
        f"{field}.d_promotion_event_steps",
        errors,
        minimum=0,
        maximum=cycle_steps - 1,
    )
    c_refreshes = _strict_int_list(
        raw.get("c_target_evidence_refresh_steps"),
        f"{field}.c_target_evidence_refresh_steps",
        errors,
        minimum=0,
        maximum=cycle_steps - 1,
    )
    d_refreshes = _strict_int_list(
        raw.get("d_target_evidence_refresh_steps"),
        f"{field}.d_target_evidence_refresh_steps",
        errors,
        minimum=0,
        maximum=cycle_steps - 1,
    )
    d_event_steps = _strict_int_list(
        raw.get("d_retirement_event_steps"),
        f"{field}.d_retirement_event_steps",
        errors,
        minimum=0,
        maximum=cycle_steps - 1,
    )
    d_event_resets = _strict_int_list(
        raw.get("d_retirement_event_reset_counts"),
        f"{field}.d_retirement_event_reset_counts",
        errors,
        minimum=0,
        maximum=1,
    )
    d_event_ages = _strict_int_list(
        raw.get("d_retirement_event_candidate_age_post"),
        f"{field}.d_retirement_event_candidate_age_post",
        errors,
        minimum=0,
        maximum=2**31 - 1,
    )
    d_event_utilities = _strict_float_list(
        raw.get("d_retirement_event_candidate_utility_post"),
        f"{field}.d_retirement_event_candidate_utility_post",
        errors,
        minimum=0.0,
    )
    d_event_heads = _strict_float_list(
        raw.get("d_retirement_event_candidate_head_linf_post"),
        f"{field}.d_retirement_event_candidate_head_linf_post",
        errors,
        minimum=0.0,
    )

    if len(errors) != before or c_rle is None or d_rle is None:
        return None
    assert c_promotions is not None
    assert d_promotions is not None
    assert c_refreshes is not None
    assert d_refreshes is not None
    assert d_event_steps is not None
    assert d_event_resets is not None
    assert d_event_ages is not None
    assert d_event_utilities is not None
    assert d_event_heads is not None
    if (
        tuple(sorted(set(c_promotions))) != c_promotions
        or tuple(sorted(set(d_promotions))) != d_promotions
        or tuple(sorted(set(c_refreshes))) != c_refreshes
        or tuple(sorted(set(d_refreshes))) != d_refreshes
        or tuple(sorted(set(d_event_steps))) != d_event_steps
    ):
        errors.append(f"{field} event arrays must be strictly increasing")
        return None
    if not (
        len(d_event_steps)
        == len(d_event_resets)
        == len(d_event_ages)
        == len(d_event_utilities)
        == len(d_event_heads)
    ):
        errors.append(f"{field} D retirement event arrays have unequal lengths")
        return None

    c_deployed, c_shadow, c_candidates = c_rle
    d_deployed, d_shadow, d_candidates = d_rle
    bank_c_deployed = _pair_slots_from_bank(
        primitives.deployed_states,
        (0, 2),
        f"{field}.primitive_deployed_bank",
        errors,
    )
    bank_d_deployed = _pair_slots_from_bank(
        primitives.deployed_states,
        (4, 5),
        f"{field}.primitive_deployed_bank",
        errors,
    )
    bank_c_shadow = _pair_slots_from_bank(
        primitives.shadow_states,
        (0, 2),
        f"{field}.primitive_shadow_bank",
        errors,
    )
    bank_d_shadow = _pair_slots_from_bank(
        primitives.shadow_states,
        (4, 5),
        f"{field}.primitive_shadow_bank",
        errors,
    )
    if any(
        slots is None
        for slots in (
            bank_c_deployed,
            bank_d_deployed,
            bank_c_shadow,
            bank_d_shadow,
        )
    ):
        return None
    assert bank_c_deployed is not None
    assert bank_d_deployed is not None
    assert bank_c_shadow is not None
    assert bank_d_shadow is not None
    if not np.array_equal(c_deployed, bank_c_deployed):
        errors.append(f"{field}.c_lifecycle_rle disagrees with the primitive bank")
    if not np.array_equal(d_deployed, bank_d_deployed):
        errors.append(f"{field}.d_lifecycle_rle disagrees with the primitive bank")
    if not np.array_equal(c_shadow, bank_c_shadow):
        errors.append(f"{field}.c_lifecycle_rle shadow disagrees with the primitive bank")
    if not np.array_equal(d_shadow, bank_d_shadow):
        errors.append(f"{field}.d_lifecycle_rle shadow disagrees with the primitive bank")
    simultaneous_deployed = (c_deployed >= 0) & (d_deployed >= 0)
    simultaneous_shadow = (c_shadow >= 0) & (d_shadow >= 0)
    if np.any(simultaneous_deployed & (c_deployed == d_deployed)):
        errors.append(f"{field} assigns C and D to the same deployed slot")
    if np.any(simultaneous_shadow & (c_shadow == d_shadow)):
        errors.append(f"{field} assigns C and D to the same shadow slot")
    if not np.all(c_candidates == 1):
        errors.append(f"{field} C candidate index must remain canonical index 1")
    if not np.all(d_candidates == 38):
        errors.append(f"{field} D candidate index must remain canonical index 38")
    if np.any(c_candidates == d_candidates):
        errors.append(f"{field} assigns C and D to the same candidate slot")
    c_present = c_deployed >= 0
    d_present = d_deployed >= 0
    expected_c_promotions = tuple(
        int(step) for step in np.flatnonzero(~c_present[:-1] & c_present[1:])
    )
    expected_d_promotions = tuple(
        int(step) for step in np.flatnonzero(~d_present[:-1] & d_present[1:])
    )
    if c_promotions != expected_c_promotions:
        errors.append(f"{field}.c_promotion_event_steps do not reconstruct from the C lifecycle")
    if d_promotions != expected_d_promotions:
        errors.append(f"{field}.d_promotion_event_steps do not reconstruct from the D lifecycle")
    ends = np.cumsum((0, *segment_lengths), dtype=np.int64)
    d_start, d_end = int(ends[3]), int(ends[4])
    c_start, c_end = int(ends[5]), int(ends[6])
    recurrent_c_start = int(ends[8])
    expected_c_refreshes = tuple(
        step
        for step in range(c_start, c_end)
        if bank_c_deployed[step] >= 0
        and bool(
            primitives.evidence_refresh[
                step,
                bank_c_deployed[step],
            ]
        )
    )
    expected_d_refreshes = tuple(
        step
        for step in range(d_start, d_end)
        if bank_d_deployed[step] >= 0
        and bool(
            primitives.evidence_refresh[
                step,
                bank_d_deployed[step],
            ]
        )
    )
    if c_refreshes != expected_c_refreshes:
        errors.append(f"{field}.c_target_evidence_refresh_steps do not reconstruct from primitives")
    if d_refreshes != expected_d_refreshes:
        errors.append(f"{field}.d_target_evidence_refresh_steps do not reconstruct from primitives")
    if any(not c_start <= step < c_end or not bool(c_present[step]) for step in c_refreshes):
        errors.append(f"{field}.c_target_evidence_refresh_steps are not live C events")
    if any(not d_start <= step < d_end or not bool(d_present[step]) for step in d_refreshes):
        errors.append(f"{field}.d_target_evidence_refresh_steps are not live D events")
    if len(errors) != before:
        return None
    c_acquisition = _target_acquisition_from_rle(
        c_refreshes,
        c_present,
        c_start,
        c_end,
    )
    d_acquisition = _target_acquisition_from_rle(
        d_refreshes,
        d_present,
        d_start,
        d_end,
    )
    c_survival_end = min(
        cycle_steps,
        recurrent_c_start + RECURRENT_ENTRY_WINDOW,
    )
    if c_acquisition is None:
        c_gap_steps = None
        c_first_gap = None
        c_evictions = 0
        c_repromotions = 0
        c_continuous = False
    else:
        c_interval = c_present[c_acquisition:c_survival_end]
        missing = np.flatnonzero(~c_interval)
        c_gap_steps = int(missing.size)
        c_first_gap = None if missing.size == 0 else int(c_acquisition + missing[0])
        c_evictions = _transition_count(c_interval, rising=False)
        c_repromotions = sum(
            c_acquisition < event_step + 1 < c_survival_end for event_step in c_promotions
        )
        c_continuous = c_gap_steps == 0

    c_mismatches = int(np.sum(c_deployed != c_shadow))
    d_mismatches = int(np.sum(d_deployed != d_shadow))
    representation_valid = primitives.representation_link_valid
    consumer_gate_valid = primitives.consumer_gate_valid
    feature_memory_enabled = primitives.feature_memory_enabled
    feature_memory_contract_valid = primitives.feature_memory_contract_valid
    c_first_window = primitives.windows["c_first_late"]
    d_late_window = primitives.windows["d_late"]
    c_recurrent_window = primitives.windows["c_recurrent_early"]
    c_first_reward = float(np.mean(primitives.rewards[c_end - FEATURE_LEARNING_WINDOW : c_end]))
    d_late_reward = float(np.mean(primitives.rewards[d_end - FEATURE_LEARNING_WINDOW : d_end]))
    c_recurrent_reward = float(
        np.mean(primitives.rewards[recurrent_c_start : recurrent_c_start + RECURRENT_ENTRY_WINDOW])
    )
    # Retention is the fraction of C's first-visit excess-over-chance reward
    # (chance = 0.5 on this binary task) preserved on re-entry.
    c_retention_ratio = (c_recurrent_reward - 0.5) / max(c_first_reward - 0.5, 1e-7)
    c_target_created_share = c_first_window.learning_nll_gain / max(
        c_first_window.masked_nll_increase, 1e-12
    )
    d_target_created_share = d_late_window.learning_nll_gain / max(
        d_late_window.masked_nll_increase, 1e-12
    )
    # The gate constants below (reward floors, accuracy/NLL-gain thresholds,
    # retention ratio, joint-success fraction) are defined and frozen next to
    # the grid in hidden_partner_lifecycle_v2; this validator consumes them
    # unchanged so a reconstructed lifecycle claim passes or fails under
    # exactly the gates the frozen protocol declares.
    c_task_learned = (
        representation_valid
        and c_mismatches == 0
        and c_acquisition is not None
        and c_first_reward >= INITIAL_LATE_REWARD_THRESHOLD
        and c_first_window.online_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and c_first_window.learning_nll_gain >= CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD
        and c_first_window.learning_positive_fraction
        >= CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD
        and c_target_created_share >= CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD
        and c_first_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and c_first_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )
    c_retained = (
        c_task_learned
        and c_continuous
        and c_recurrent_reward >= RECURRENT_EARLY_REWARD_THRESHOLD
        and c_retention_ratio >= RETENTION_RATIO_THRESHOLD
        and c_recurrent_window.online_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and c_recurrent_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and c_recurrent_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )

    d_through_exit = bool(np.all(d_present[d_end - FEATURE_LEARNING_WINDOW : d_end]))
    d_task_learned = (
        representation_valid
        and d_mismatches == 0
        and d_acquisition is not None
        and d_through_exit
        and d_late_reward >= INITIAL_LATE_REWARD_THRESHOLD
        and d_late_window.online_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and d_late_window.learning_nll_gain >= CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD
        and d_late_window.learning_positive_fraction
        >= CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD
        and d_target_created_share >= CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD
        and d_late_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and d_late_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )
    d_post_live_steps = int(np.sum(d_present[d_end:cycle_steps]))
    d_post_live_fraction = d_post_live_steps / max(
        cycle_steps - d_end,
        1,
    )
    d_post_promotions = sum(event_step >= d_end for event_step in d_promotions)
    d_final_absent = bool(np.all(~d_present[-(FINAL_ABSENCE_WINDOW + 1) :]))

    aligned_event_step: int | None = None
    aligned_effective_step: int | None = None
    aligned_event_latency: int | None = None
    aligned_latency: int | None = None
    aligned_reset: int | None = None
    aligned_utility: float | None = None
    aligned_head: float | None = None
    aligned_age: int | None = None
    for index, event_step in enumerate(d_event_steps):
        effective_step = event_step + 1
        confirmation_end = effective_step + RETIREMENT_CONFIRMATION_WINDOW
        aligned = (
            event_step >= d_end
            and confirmation_end <= state_count
            and bool(d_present[event_step])
            and not bool(d_present[effective_step])
            and bool(np.all(~d_present[effective_step:confirmation_end]))
            and d_event_resets[index] == 1
            and d_event_utilities[index] == 0.0
            and d_event_heads[index] == 0.0
            and d_event_ages[index] == 0
            and event_step not in d_promotions
        )
        if aligned:
            aligned_event_step = event_step
            aligned_effective_step = effective_step
            aligned_event_latency = event_step - d_end
            aligned_latency = effective_step - d_end
            aligned_reset = d_event_resets[index]
            aligned_utility = d_event_utilities[index]
            aligned_head = d_event_heads[index]
            aligned_age = d_event_ages[index]
            break
    d_repromotions = (
        0
        if aligned_event_step is None
        else sum(event_step > aligned_event_step for event_step in d_promotions)
    )
    d_event_aligned = aligned_event_step is not None
    d_stable = d_task_learned and d_event_aligned and d_final_absent and d_repromotions == 0
    candidate_archive_valid = bool(
        primitives.candidate_archive_valid
        and np.all(c_candidates >= 0)
        and np.all(d_candidates >= 0)
    )
    joint = bool(
        representation_valid
        and consumer_gate_valid
        and feature_memory_enabled
        and feature_memory_contract_valid
        and candidate_archive_valid
        and c_retained
        and d_stable
    )

    derived: dict[str, object] = {
        "representation_link_contract_valid": representation_valid,
        "consumer_gate_contract_valid": consumer_gate_valid,
        "feature_memory_enabled": feature_memory_enabled,
        "feature_memory_contract_valid": feature_memory_contract_valid,
        "c_shadow_deployed_mismatch_steps": c_mismatches,
        "d_shadow_deployed_mismatch_steps": d_mismatches,
        "c_acquisition_step": c_acquisition,
        "c_first_late_reward": c_first_reward,
        "c_first_late_intended_accuracy": (c_first_window.online_accuracy),
        "c_first_late_online_nll": c_first_window.online_nll,
        "c_first_late_entry_frozen_critical_nll": (c_first_window.entry_frozen_nll),
        "c_critical_column_learning_nll_gain": (c_first_window.learning_nll_gain),
        "c_critical_column_learning_positive_fraction": (c_first_window.learning_positive_fraction),
        "c_critical_column_target_created_share": (c_target_created_share),
        "c_first_late_entry_frozen_critical_accuracy": (c_first_window.entry_frozen_accuracy),
        "c_critical_column_learning_accuracy_gain": (c_first_window.learning_accuracy_gain),
        "c_first_late_masked_nll_increase": (c_first_window.masked_nll_increase),
        "c_first_late_masked_nll_positive_fraction": (c_first_window.masked_nll_positive_fraction),
        "c_task_learned": c_task_learned,
        "c_survival_end_exclusive": c_survival_end,
        "c_survival_gap_steps": c_gap_steps,
        "c_first_survival_gap_step": c_first_gap,
        "c_evictions_after_acquisition": c_evictions,
        "c_repromotions_after_acquisition": c_repromotions,
        "c_continuously_survived": c_continuous,
        "c_recurrent_early_reward": c_recurrent_reward,
        "c_recurrent_early_excess_reward_retention": c_retention_ratio,
        "c_recurrent_early_intended_accuracy": (c_recurrent_window.online_accuracy),
        "c_recurrent_early_masked_nll_increase": (c_recurrent_window.masked_nll_increase),
        "c_recurrent_early_masked_nll_positive_fraction": (
            c_recurrent_window.masked_nll_positive_fraction
        ),
        "c_retained_and_used": c_retained,
        "d_late_reward": d_late_reward,
        "d_late_intended_accuracy": d_late_window.online_accuracy,
        "d_late_online_nll": d_late_window.online_nll,
        "d_late_entry_frozen_critical_nll": (d_late_window.entry_frozen_nll),
        "d_critical_column_learning_nll_gain": (d_late_window.learning_nll_gain),
        "d_critical_column_learning_positive_fraction": (d_late_window.learning_positive_fraction),
        "d_critical_column_target_created_share": d_target_created_share,
        "d_late_entry_frozen_critical_accuracy": (d_late_window.entry_frozen_accuracy),
        "d_critical_column_learning_accuracy_gain": (d_late_window.learning_accuracy_gain),
        "d_late_masked_nll_increase": (d_late_window.masked_nll_increase),
        "d_late_masked_nll_positive_fraction": (d_late_window.masked_nll_positive_fraction),
        "d_acquisition_step": d_acquisition,
        "d_deployed_through_exit": d_through_exit,
        "d_task_learned": d_task_learned,
        "d_retirement_event_step": aligned_event_step,
        "d_retirement_step": aligned_effective_step,
        "d_retirement_event_latency_steps": aligned_event_latency,
        "d_retirement_latency_steps": aligned_latency,
        "d_post_exit_live_slot_steps": d_post_live_steps,
        "d_post_exit_live_fraction": d_post_live_fraction,
        "d_post_exit_promotion_count": d_post_promotions,
        "d_repromotions_after_retirement": d_repromotions,
        "d_absent_entire_final_window": d_final_absent,
        "d_retirement_event_count": len(d_event_steps),
        "d_matching_candidate_reset_count": sum(d_event_resets),
        "d_linked_matching_candidate_reset_count": aligned_reset,
        "d_linked_candidate_utility_post": aligned_utility,
        "d_linked_candidate_head_linf_post": aligned_head,
        "d_linked_candidate_age_post": aligned_age,
        "d_retirement_event_aligned": d_event_aligned,
        "d_learned_then_stably_retired": d_stable,
        "joint_memory_management_success": joint,
        "candidate_archive_contract_valid": candidate_archive_valid,
    }
    for name, expected in derived.items():
        if not _derived_value_matches(raw.get(name), expected):
            errors.append(f"{field}.{name} does not reconstruct from primitive fields")
    if len(errors) != before:
        return None
    return raw


def _reconstruct_aggregates(
    runs: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    required = math.ceil(MINIMUM_JOINT_SUCCESS_FRACTION * LEASE_TUNING_SEED_COUNT)
    aggregates: list[dict[str, object]] = []
    for cell in LEASE_TUNING_GRID:
        cell_runs = [run for run in runs if run.get("cell_index") == cell.index]
        rewards = np.asarray(
            [run["_reconstructed_mean_reward"] for run in cell_runs],
            dtype=np.float64,
        )
        lifecycles = [cast(Mapping[str, object], run["critical_lifecycle"]) for run in cell_runs]
        valid = all(run.get("_reconstructed_contract_valid") is True for run in cell_runs)
        joint_count = sum(
            lifecycle.get("joint_memory_management_success") is True for lifecycle in lifecycles
        )
        c_count = sum(lifecycle.get("c_retained_and_used") is True for lifecycle in lifecycles)
        d_count = sum(
            lifecycle.get("d_learned_then_stably_retired") is True for lifecycle in lifecycles
        )
        latencies = [
            cast(int, lifecycle["d_retirement_latency_steps"])
            for lifecycle in lifecycles
            if lifecycle.get("d_retirement_latency_steps") is not None
        ]
        feasible = (
            valid
            and float(np.mean(rewards)) >= 0.85
            and float(np.min(rewards)) >= 0.80
            and joint_count >= required
            and c_count >= required
            and d_count >= required
        )
        aggregates.append(
            {
                "cell_index": cell.index,
                "cell_digest": _canonical_sha256(cell.to_dict()),
                "seed_count": LEASE_TUNING_SEED_COUNT,
                "required_success_count": required,
                "finite_contract_valid": valid,
                "mean_reward": float(np.mean(rewards)),
                "minimum_seed_reward": float(np.min(rewards)),
                "joint_success_count": joint_count,
                "c_retained_and_used_count": c_count,
                "d_learned_then_stably_retired_count": d_count,
                "total_d_repromotions": int(
                    sum(
                        cast(
                            int,
                            lifecycle["d_repromotions_after_retirement"],
                        )
                        for lifecycle in lifecycles
                    )
                ),
                "median_d_retirement_latency_steps": (
                    None if not latencies else float(np.median(latencies))
                ),
                "feasible": feasible,
            }
        )
    return aggregates


def _select_cell(
    aggregates: list[dict[str, object]],
) -> dict[str, object] | None:
    feasible = [record for record in aggregates if record["feasible"]]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda record: (
            cast(int, record["joint_success_count"]),
            cast(float, record["minimum_seed_reward"]),
            cast(float, record["mean_reward"]),
            -cast(int, record["total_d_repromotions"]),
            -cast(float, record["median_d_retirement_latency_steps"]),
            -cast(int, record["cell_index"]),
        ),
    )


def _validate_lease_tuning_artifact(
    artifact: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> LeaseTuningArtifactValidation:
    """Fail closed on source, digest, seed, grid, run, RLE, or aggregate drift."""
    errors: list[str] = []
    expected_top = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "scientific_payload",
        "scientific_digest",
        "operational_metadata",
    }
    if set(artifact) != expected_top:
        errors.append("top-level artifact fields do not match the schema")
    if artifact.get("schema_version") != LEASE_TUNING_ARTIFACT_SCHEMA:
        errors.append("artifact schema version is unsupported")
    if artifact.get("development_only") is not True:
        errors.append("artifact must remain development-only")
    if artifact.get("scientific_promotion_allowed") is not False:
        errors.append("artifact must forbid promotion")
    _validate_operational_metadata(
        artifact.get("operational_metadata"),
        errors,
    )

    payload = artifact.get("scientific_payload")
    if not isinstance(payload, Mapping):
        errors.append("scientific_payload must be an object")
        payload = {}
    expected_payload = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "protocol",
        "seed_namespace",
        "seed_pairs",
        "grid",
        "selection_rule",
        "aggregates",
        "selected_cell",
        "runs",
        "scope_limits",
        "source_sha256",
    }
    if set(payload) != expected_payload:
        errors.append("scientific_payload fields do not match the schema")
    if payload.get("schema_version") != HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA:
        errors.append("lifecycle record schema is unsupported")
    if payload.get("development_only") is not True:
        errors.append("scientific payload must remain development-only")
    if payload.get("scientific_promotion_allowed") is not False:
        errors.append("scientific payload must forbid promotion")
    protocol_object = HiddenPartnerDevelopmentProtocol()
    expected_protocol = protocol_object.to_config()
    if not _strict_json_equal(payload.get("protocol"), expected_protocol):
        errors.append("tuning protocol changed")
    if payload.get("seed_namespace") != LEASE_TUNING_NAMESPACE:
        errors.append("tuning seed namespace is invalid")
    if not _strict_json_equal(
        payload.get("selection_rule"),
        TUNING_SELECTION_RULE,
    ):
        errors.append("tuning selection rule changed")
    if not _strict_json_equal(
        payload.get("grid"),
        [cell.to_dict() for cell in LEASE_TUNING_GRID],
    ):
        errors.append("tuning grid changed")
    expected_seeds = [
        pair.to_dict()
        for pair in derive_hidden_partner_seed_pairs(
            LEASE_TUNING_NAMESPACE,
            LEASE_TUNING_SEED_COUNT,
        )
    ]
    if not _strict_json_equal(payload.get("seed_pairs"), expected_seeds):
        errors.append("tuning seed pairs changed")
    if not _strict_json_equal(
        payload.get("scope_limits"),
        list(LEASE_TUNING_SCOPE_LIMITS),
    ):
        errors.append("tuning scope limits changed")

    digest = artifact.get("scientific_digest")
    if not isinstance(digest, Mapping) or set(digest) != {
        "algorithm",
        "scope",
        "sha256",
    }:
        errors.append("scientific_digest fields are invalid")
    else:
        try:
            expected_digest = _canonical_sha256(payload)
        except (OverflowError, TypeError, ValueError) as error:
            errors.append(f"scientific payload is not strict JSON: {error}")
        else:
            if digest.get("algorithm") != "sha256":
                errors.append("scientific digest algorithm must be sha256")
            if digest.get("scope") != "$.scientific_payload":
                errors.append("scientific digest scope is invalid")
            if digest.get("sha256") != expected_digest:
                errors.append("scientific payload digest mismatch")

    try:
        expected_sources = source_snapshot(root)
    except OSError as error:
        errors.append(f"cannot hash current tuning sources: {error}")
    else:
        if not _strict_json_equal(
            payload.get("source_sha256"),
            expected_sources,
        ):
            errors.append("source hashes do not match pinned tuning sources")

    runs_payload = payload.get("runs")
    reconstructed_runs: list[Mapping[str, object]] = []
    expected_run_count = len(LEASE_TUNING_GRID) * LEASE_TUNING_SEED_COUNT
    fixed_world = HiddenPartnerMappingWorld(protocol_object.environment)
    expected_state_nbytes = _expected_integrated_state_nbytes()
    if not isinstance(runs_payload, list) or len(runs_payload) != expected_run_count:
        errors.append(f"runs must contain exactly {expected_run_count} records")
    else:
        for index, run in enumerate(runs_payload):
            # Structural or semantic failure in an earlier record already
            # invalidates the artifact. Avoid adversarially forcing full-grid
            # decompression after the validator has a decisive error.
            if errors:
                break
            run_error_count = len(errors)
            if not isinstance(run, Mapping) or set(run) != {
                "cell_index",
                "cell_digest",
                "condition_config",
                "seed_pair",
                "run_summary",
                "critical_lifecycle",
                "critical_run_primitives",
            }:
                errors.append(f"runs[{index}] fields are invalid")
                continue
            expected_cell = index // LEASE_TUNING_SEED_COUNT
            expected_seed = index % LEASE_TUNING_SEED_COUNT
            cell_index = run.get("cell_index")
            if type(cell_index) is not int or cell_index != expected_cell:
                errors.append(f"runs[{index}] cell ordering is invalid")
            cell = LEASE_TUNING_GRID[expected_cell]
            cell_config = cell.agent_config()
            expected_cell_digest = _canonical_sha256(cell.to_dict())
            if run.get("cell_digest") != expected_cell_digest:
                errors.append(f"runs[{index}] cell digest is invalid")
            expected_condition = HiddenPartnerCondition(
                name="full",
                config=cell_config,
                isolated_question=(f"evidence-lease tuning cell {expected_cell}"),
            ).to_config()
            if not _strict_json_equal(
                run.get("condition_config"),
                expected_condition,
            ):
                errors.append(f"runs[{index}] condition config is invalid")
            if not _strict_json_equal(
                run.get("seed_pair"),
                expected_seeds[expected_seed],
            ):
                errors.append(f"runs[{index}] seed pairing is invalid")
            summary_payload = run.get("run_summary")
            if not isinstance(summary_payload, Mapping):
                errors.append(f"runs[{index}].run_summary must be an object")
                continue
            try:
                summary = hidden_partner_run_summary_from_dict(summary_payload)
            except (
                ArithmeticError,
                OverflowError,
                TypeError,
                ValueError,
            ) as error:
                errors.append(f"runs[{index}].run_summary is invalid: {error}")
                continue
            if summary.condition != "full":
                errors.append(f"runs[{index}] summary condition must be full")
            if not _strict_json_equal(
                summary.seed_pair.to_dict(),
                expected_seeds[expected_seed],
            ):
                errors.append(f"runs[{index}] summary seed pairing is invalid")
            seeded_environment = fixed_world.init(jr.key(summary.seed_pair.stream_seed))
            expected_lengths = tuple(
                int(value)
                for value in np.asarray(
                    seeded_environment.segment_lengths,
                    dtype=np.int64,
                )
            )
            if summary.segment_lengths != expected_lengths:
                errors.append(
                    f"runs[{index}] segment lengths do not match the exact seeded schedule"
                )
            primitives = _validate_critical_run_primitives(
                run.get("critical_run_primitives"),
                cycle_steps=summary.cycle_steps,
                segment_lengths=summary.segment_lengths,
                consumer_write_confirmation_steps=(
                    cell_config.consumer_evidence_confirmation_steps
                ),
                consumer_read_confirmation_steps=(
                    cell_config.consumer_read_confirmation_steps
                ),
                consumer_read_lease_steps=cell_config.consumer_read_lease_steps,
                feature_memory_enabled=(
                    cell_config.evidence_gated_feature_memory
                ),
                feature_evidence_confirmation_steps=(
                    cell_config.feature_evidence_confirmation_steps
                ),
                field=f"runs[{index}].critical_run_primitives",
                errors=errors,
            )
            if primitives is None:
                continue
            for name, expected_value in (
                ("counter_contract_valid", primitives.counter_contract_valid),
                ("causal_contract_valid", primitives.causal_contract_valid),
                ("all_finite", primitives.all_finite),
            ):
                if getattr(summary, name) is not expected_value:
                    errors.append(
                        f"runs[{index}].run_summary.{name} does not reconstruct from contract bits"
                    )
            resource_shape_valid = (
                summary.initial_state_nbytes == expected_state_nbytes
                and summary.final_state_nbytes == expected_state_nbytes
            )
            if not resource_shape_valid:
                errors.append(
                    f"runs[{index}].run_summary state bytes do not match "
                    "the exact integrated budget"
                )
            if summary.resource_shape_matched is not resource_shape_valid:
                errors.append(
                    f"runs[{index}].run_summary.resource_shape_matched does not reconstruct"
                )
            expected_mean_reward = float(np.mean(primitives.rewards))
            if not _derived_value_matches(
                summary.mean_reward,
                expected_mean_reward,
            ):
                errors.append(
                    f"runs[{index}].run_summary.mean_reward does not reconstruct from rewards"
                )
            perfect_reward = (
                1.0 - cell_config.epsilon
            ) * 0.95 + cell_config.epsilon * 0.5
            expected_normalized = (expected_mean_reward - 0.5) / max(perfect_reward - 0.5, 1e-7)
            if not _derived_value_matches(
                summary.normalized_control_score,
                expected_normalized,
            ):
                errors.append(
                    f"runs[{index}].run_summary.normalized_control_score does not reconstruct"
                )
            segment_start = 0
            for segment_index, (
                segment,
                segment_length,
            ) in enumerate(
                zip(
                    summary.segments,
                    summary.segment_lengths,
                    strict=True,
                )
            ):
                segment_end = segment_start + segment_length
                segment_rewards = primitives.rewards[segment_start:segment_end]
                window = min(
                    protocol_object.early_late_window,
                    segment_length,
                )
                expected_segment_values = {
                    "mean_reward": float(np.mean(segment_rewards)),
                    "early_reward": float(np.mean(segment_rewards[:window])),
                    "late_reward": float(np.mean(segment_rewards[-window:])),
                }
                for name, expected_segment_value in expected_segment_values.items():
                    if not _derived_value_matches(
                        getattr(segment, name),
                        expected_segment_value,
                    ):
                        errors.append(
                            f"runs[{index}].run_summary.segments[{segment_index}]."
                            f"{name} does not reconstruct from rewards"
                        )
                segment_start = segment_end
            lifecycle = _validate_lifecycle(
                run.get("critical_lifecycle"),
                summary.cycle_steps,
                summary.segment_lengths,
                primitives,
                f"runs[{index}].critical_lifecycle",
                errors,
            )
            if lifecycle is not None and len(errors) == run_error_count:
                reconstructed_runs.append(
                    {
                        **dict(run),
                        "critical_lifecycle": lifecycle,
                        "_reconstructed_mean_reward": (expected_mean_reward),
                        "_reconstructed_contract_valid": (
                            primitives.all_finite
                            and primitives.counter_contract_valid
                            and primitives.causal_contract_valid
                            and resource_shape_valid
                            and primitives.representation_link_valid
                            and primitives.consumer_gate_valid
                            and primitives.feature_memory_enabled
                            and primitives.feature_memory_contract_valid
                            and primitives.candidate_archive_valid
                            and lifecycle.get(
                                "candidate_archive_contract_valid"
                            )
                            is True
                        ),
                    }
                )

    expected_selection: dict[str, object] | None = None
    if len(reconstructed_runs) == expected_run_count:
        expected_aggregates = _reconstruct_aggregates(reconstructed_runs)
        if not _strict_json_equal(
            payload.get("aggregates"),
            expected_aggregates,
        ):
            errors.append("aggregates do not reconstruct exactly from runs")
        expected_selection = _select_cell(expected_aggregates)
        if not _strict_json_equal(
            payload.get("selected_cell"),
            expected_selection,
        ):
            errors.append("selected cell does not follow the frozen rule")

    return LeaseTuningArtifactValidation(
        valid=not errors,
        feasible_cell_selected=(not errors and expected_selection is not None),
        errors=tuple(errors),
    )


def validate_lease_tuning_artifact(
    artifact: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> LeaseTuningArtifactValidation:
    """Validate adversarial input and reject an unexecuted namespace."""
    try:
        require_lease_tuning_execution_allowed()
    except RuntimeError as error:
        return LeaseTuningArtifactValidation(
            valid=False,
            feasible_cell_selected=False,
            errors=(str(error),),
        )
    return _validate_lease_tuning_artifact_for_testing(artifact, root=root)


def _validate_lease_tuning_artifact_for_testing(
    artifact: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> LeaseTuningArtifactValidation:
    """Exercise the structural validator for synthetic unit fixtures only."""
    try:
        return _validate_lease_tuning_artifact(
            artifact,
            root=root,
        )
    except Exception as error:  # noqa: BLE001 - this is a fail-closed boundary
        return LeaseTuningArtifactValidation(
            valid=False,
            feasible_cell_selected=False,
            errors=(f"artifact validation failed closed: {type(error).__name__}: {error}",),
        )


__all__ = [
    "LEASE_TUNING_ARTIFACT_SCHEMA",
    "LeaseTuningArtifactValidation",
    "REPO_ROOT",
    "SOURCE_PATHS",
    "build_lease_tuning_artifact",
    "lease_tuning_artifact_json",
    "load_lease_tuning_artifact",
    "source_snapshot",
    "validate_lease_tuning_artifact",
]
