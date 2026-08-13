"""Pure configuration bindings for the 14 native matched-v3 Alberta arms.

The historical matched-current builder already defines the exact worker
envelopes for the causal-map grid, Horde variants, and local RTU/RTRL arm.
This module gives those envelopes an explicit matched-v3 binding without
copying or silently redefining them.  The historical builder is imported only
when a caller explicitly requests a configuration; this module itself has no
eager JAX-backed import and executes no workload at import time.

Every build validates the exact candidate set, rejects aliased or non-plain
JSON, canonicalizes the selected worker envelope, and checks its frozen
SHA-256.  The implementation is deliberately ``implemented_unqualified``:
configuration completeness does not qualify a source snapshot, authorize an
execution, promote an artifact, or support a performance claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, cast

LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_local_configuration_source.v1"
)
LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_local_configuration_builder.v1"
)
LOCAL_CONFIGURATION_BUILDER_STATUS: Final = "implemented_unqualified"
_MAX_CONFIGURATION_BYTES: Final = 2 * 1024 * 1024

MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS: Final = (
    "causal_e025_q050",
    "causal_e025_q075",
    "causal_e025_q090",
    "causal_e050_q050",
    "causal_e050_q075",
    "causal_e050_q090",
    "causal_e100_q050",
    "causal_e100_q075",
    "causal_e100_q090",
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
    "alberta_rtu_h08_taylor",
)

EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "causal_e025_q050": (
                "1290335563481b7ac2fd3eda91ef9c63216684fd096f3ab5b16591de0870c736"
            ),
            "causal_e025_q075": (
                "69a5df44db99866a0ee3967677fad66ea94c60b1bfa8317936e2c142fac34ed1"
            ),
            "causal_e025_q090": (
                "e21692571fc751bdf2c4fa0e89ad43b12dbd51c72a0821d5839fc82f1031f8f4"
            ),
            "causal_e050_q050": (
                "916bd37e04c39dc16c19153032fc1c3baf12a941efb3df95860ee9f03c1ef331"
            ),
            "causal_e050_q075": (
                "afaa3ea47cd410a43541c85976fa6f718c5f70504494f70496385ec37ea84a63"
            ),
            "causal_e050_q090": (
                "ab555510e08a98e733d01a9b145d19073bb17ba31681a459a55a978d5a4faf33"
            ),
            "causal_e100_q050": (
                "00390162a1950e976a7b3e216b8c6d94a76427c38c8e30bbdc25fa583bf018a8"
            ),
            "causal_e100_q075": (
                "8d7a8afdb204c1837834ef633e2524bf569180c763a34a96c883c6e2cd33fb48"
            ),
            "causal_e100_q090": (
                "899658dff1eeaadf59de8dc437d1324429306b8a427a4ed67ccf54437931955c"
            ),
            "alberta_horde_default": (
                "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a"
            ),
            "alberta_horde_eps05": (
                "ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f"
            ),
            "alberta_horde_recurrent64": (
                "870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef"
            ),
            "alberta_horde_step3e3": (
                "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5"
            ),
            "alberta_rtu_h08_taylor": (
                "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
            ),
        }
    )
)

BUILDER_ID_BY_CANDIDATE: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{
            candidate_id: (
                "alberta.forager_matched_v3.generated_local.causal_map_grid.v1"
            )
            for candidate_id in MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS[:9]
        },
        **{
            candidate_id: (
                "alberta.forager_matched_v3.generated_local.horde_actor_critic.v1"
            )
            for candidate_id in MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS[9:13]
        },
        MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS[13]: (
            "alberta.forager_matched_v3.generated_local.rtu_h08_taylor.v1"
        ),
    }
)

_CLAIMS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
)


class ForagerMatchedV3LocalConfigurationError(ValueError):
    """The local candidate identity, source set, or configuration is invalid."""


@dataclass(frozen=True)
class BuiltMatchedV3LocalConfiguration:
    """One immutable, content-addressed local worker-envelope binding."""

    candidate_id: str
    builder_id: str
    status: Literal["implemented_unqualified"]
    canonical_json_bytes: bytes
    configuration_sha256: str
    source_descriptor_sha256: str
    builder_descriptor_sha256: str
    configuration_complete: bool = True
    execution_ready: bool = False
    execution_authorized: bool = False
    scientific_promotion_allowed: bool = False
    universal_sota_claim_allowed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or self.candidate_id not in EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE
        ):
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration has an invalid candidate_id"
            )
        if self.builder_id != BUILDER_ID_BY_CANDIDATE[self.candidate_id]:
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration has a mismatched builder_id"
            )
        if type(self.status) is not str or self.status != LOCAL_CONFIGURATION_BUILDER_STATUS:
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration status must be implemented_unqualified"
            )
        if type(self.canonical_json_bytes) is not bytes:
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration payload must be exact bytes"
            )
        expected = EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE[self.candidate_id]
        if (
            self.configuration_sha256 != expected
            or not hmac.compare_digest(_sha256(self.canonical_json_bytes), expected)
        ):
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration payload identity drifted"
            )
        if self.source_descriptor_sha256 != (
            MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        ):
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration source descriptor drifted"
            )
        if self.builder_descriptor_sha256 != (
            MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
        ):
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration builder descriptor drifted"
            )
        if self.configuration_complete is not True or any(
            value is not False
            for value in (
                self.execution_ready,
                self.execution_authorized,
                self.scientific_promotion_allowed,
                self.universal_sota_claim_allowed,
            )
        ):
            raise ForagerMatchedV3LocalConfigurationError(
                "built local configuration readiness or authority drifted"
            )

    def payload(self) -> dict[str, Any]:
        """Return a newly decoded, detached worker-envelope payload."""
        value = json.loads(self.canonical_json_bytes)
        if type(value) is not dict:  # pragma: no cover - construction guarantees this
            raise AssertionError("canonical local configuration is not a JSON object")
        return cast(dict[str, Any], value)

    def binding(self) -> dict[str, Any]:
        """Return a detached non-authorizing record for a configuration plan."""
        return {
            "kind": "generated_local",
            "repository_id": "local_alberta",
            "builder_id": self.builder_id,
            "builder_status": self.status,
            "builder_descriptor_sha256": self.builder_descriptor_sha256,
            "source_descriptor_sha256": self.source_descriptor_sha256,
            "worker_envelope_sha256": self.configuration_sha256,
            "configuration_complete": self.configuration_complete,
            "source_snapshot_status": "unqualified_current_checkout",
            "execution_ready": self.execution_ready,
            "execution_authorized": self.execution_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "universal_sota_claim_allowed": self.universal_sota_claim_allowed,
        }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_unaliased_plain_json(value: object, *, label: str) -> None:
    pending = [value]
    container_ids: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if id(item) in container_ids:
                raise ForagerMatchedV3LocalConfigurationError(
                    f"{label} contains aliased or cyclic containers"
                )
            container_ids.add(id(item))
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3LocalConfigurationError(
                    f"{label} contains a non-string JSON object key"
                )
            pending.extend(mapping.values())
        elif type(item) is list:
            if id(item) in container_ids:
                raise ForagerMatchedV3LocalConfigurationError(
                    f"{label} contains aliased or cyclic containers"
                )
            container_ids.add(id(item))
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3LocalConfigurationError(
                    f"{label} contains a non-finite JSON number"
                )
        elif item is not None and type(item) not in {str, int, bool}:
            raise ForagerMatchedV3LocalConfigurationError(
                f"{label} contains non-plain JSON type {type(item).__name__}"
            )


def _canonical_json_bytes(value: object, *, label: str) -> bytes:
    _assert_unaliased_plain_json(value, label=label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ForagerMatchedV3LocalConfigurationError(
            f"{label} is not canonicalizable plain JSON"
        ) from exc
    if len(raw) > _MAX_CONFIGURATION_BYTES:
        raise ForagerMatchedV3LocalConfigurationError(
            f"{label} exceeds the configuration byte limit"
        )
    return raw


def _source_descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_CONFIGURATION_BUILDER_STATUS,
        "classification": "local_configuration_source_nonpromoting",
        "repository_id": "local_alberta",
        "source_snapshot_status": "unqualified_current_checkout",
        "source_builder": {
            "module": (
                "alberta_framework.benchmarks.forager_matched_open_protocol"
            ),
            "path": (
                "alberta_framework/benchmarks/forager_matched_open_protocol.py"
            ),
            "callable": "matched_current_alberta_configurations",
            "invocation": "explicit_build_call_only",
            "relationship": (
                "direct_worker_envelope_binding_without_configuration_transform"
            ),
        },
        "canonicalization": {
            "encoding": "utf-8",
            "object_keys": "lexicographic",
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
            "accepted_container_types": ["dict", "list"],
            "aliases_allowed": False,
        },
        "candidate_ids": list(MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS),
        "expected_configuration_sha256_by_candidate": dict(
            EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE
        ),
        "claims": dict(_CLAIMS),
        "limitations": [
            "The current checkout is not a qualified content-addressed source snapshot.",
            "Configuration construction executes no benchmark workload or protected seed.",
            "Configuration completeness does not grant execution or promotion authority.",
        ],
    }


_SOURCE_DESCRIPTOR: Final = _source_descriptor()
_SOURCE_DESCRIPTOR_BYTES: Final = _canonical_json_bytes(
    _SOURCE_DESCRIPTOR,
    label="matched-v3 local configuration source descriptor",
)
MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256: Final = (
    "d15d70b55d965b2c135f1dcaa36a74173e4023e4fdc9430c43660df54f1bb38c"
)
if not hmac.compare_digest(
    _sha256(_SOURCE_DESCRIPTOR_BYTES),
    MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256,
):
    raise AssertionError("canonical matched-v3 local source descriptor drifted")


def _builder_descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_CONFIGURATION_BUILDER_STATUS,
        "classification": "implemented_unqualified_non_authorizing_builder",
        "module": (
            "alberta_framework.benchmarks.forager_matched_v3_local_configuration"
        ),
        "path": (
            "alberta_framework/benchmarks/forager_matched_v3_local_configuration.py"
        ),
        "single_builder_callable": "build_matched_v3_local_configuration",
        "set_builder_callable": "build_all_matched_v3_local_configurations",
        "import_time_jax_execution": False,
        "workload_execution": False,
        "source_descriptor_sha256": (
            MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        ),
        "candidate_bindings": [
            {
                "candidate_id": candidate_id,
                "builder_id": BUILDER_ID_BY_CANDIDATE[candidate_id],
                "expected_configuration_sha256": (
                    EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE[candidate_id]
                ),
            }
            for candidate_id in MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS
        ],
        "claims": dict(_CLAIMS),
        "limitations": [
            "The builder validates configuration identity but not runtime capability.",
            "No build result is an execution receipt or scientific evidence artifact.",
        ],
    }


_BUILDER_DESCRIPTOR: Final = _builder_descriptor()
_BUILDER_DESCRIPTOR_BYTES: Final = _canonical_json_bytes(
    _BUILDER_DESCRIPTOR,
    label="matched-v3 local configuration builder descriptor",
)
MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256: Final = (
    "1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b"
)
if not hmac.compare_digest(
    _sha256(_BUILDER_DESCRIPTOR_BYTES),
    MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256,
):
    raise AssertionError("canonical matched-v3 local builder descriptor drifted")


def _require_candidate_id(candidate_id: object) -> str:
    if type(candidate_id) is not str:
        raise ForagerMatchedV3LocalConfigurationError(
            "candidate_id must be an exact string"
        )
    value = candidate_id
    if value not in EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE:
        raise ForagerMatchedV3LocalConfigurationError(
            f"unknown matched-v3 local candidate {value!r}"
        )
    return value


def _load_and_validate_source_configurations() -> dict[str, dict[str, Any]]:
    # Deliberately lazy: the source module imports learner/configuration modules
    # backed by JAX, while this binding module must remain import-time pure.
    from alberta_framework.benchmarks import forager_matched_open_protocol

    configurations = (
        forager_matched_open_protocol.matched_current_alberta_configurations()
    )
    if type(configurations) is not dict:
        raise ForagerMatchedV3LocalConfigurationError(
            "matched-current source configurations must be a plain dictionary"
        )
    if tuple(configurations) != MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS:
        raise ForagerMatchedV3LocalConfigurationError(
            "matched-current source candidate order or membership drifted"
        )
    _assert_unaliased_plain_json(
        configurations,
        label="matched-current source configuration set",
    )
    return configurations


def _build_from_validated_source(
    candidate_id: str,
    source: Mapping[str, dict[str, Any]],
) -> BuiltMatchedV3LocalConfiguration:
    canonical = _canonical_json_bytes(
        source[candidate_id],
        label=f"local worker envelope {candidate_id}",
    )
    actual_sha256 = _sha256(canonical)
    expected_sha256 = EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE[candidate_id]
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ForagerMatchedV3LocalConfigurationError(
            f"local worker envelope drift for {candidate_id}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return BuiltMatchedV3LocalConfiguration(
        candidate_id=candidate_id,
        builder_id=BUILDER_ID_BY_CANDIDATE[candidate_id],
        status="implemented_unqualified",
        canonical_json_bytes=canonical,
        configuration_sha256=actual_sha256,
        source_descriptor_sha256=(
            MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        ),
        builder_descriptor_sha256=(
            MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
        ),
    )


def build_all_matched_v3_local_configurations() -> tuple[
    BuiltMatchedV3LocalConfiguration, ...
]:
    """Build and content-verify the exact ordered set of 14 local envelopes."""
    source = _load_and_validate_source_configurations()
    return tuple(
        _build_from_validated_source(candidate_id, source)
        for candidate_id in MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS
    )


def build_matched_v3_local_configuration(
    candidate_id: object,
) -> BuiltMatchedV3LocalConfiguration:
    """Build one envelope after verifying the atomic 14-arm source identity."""
    exact_candidate_id = _require_candidate_id(candidate_id)
    source = _load_and_validate_source_configurations()
    built = {
        source_candidate_id: _build_from_validated_source(source_candidate_id, source)
        for source_candidate_id in MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS
    }
    return built[exact_candidate_id]


def canonical_matched_v3_local_configuration_bytes(candidate_id: object) -> bytes:
    """Return exact canonical worker-envelope JSON for one local candidate."""
    return build_matched_v3_local_configuration(candidate_id).canonical_json_bytes


def matched_v3_local_configuration_sha256(candidate_id: object) -> str:
    """Return the verified canonical worker-envelope digest for one candidate."""
    return build_matched_v3_local_configuration(candidate_id).configuration_sha256


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalConfigurationError(
                f"duplicate configuration key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerMatchedV3LocalConfigurationError(
        f"non-finite configuration number {token!r} is forbidden"
    )


def parse_matched_v3_local_configuration_payload(
    candidate_id: object,
    raw: bytes,
) -> BuiltMatchedV3LocalConfiguration:
    """Accept only the exact canonical payload bound to ``candidate_id``."""
    exact_candidate_id = _require_candidate_id(candidate_id)
    if type(raw) is not bytes:
        raise ForagerMatchedV3LocalConfigurationError(
            "local configuration payload must be exact bytes"
        )
    if len(raw) > _MAX_CONFIGURATION_BYTES:
        raise ForagerMatchedV3LocalConfigurationError(
            "local configuration payload exceeds the byte limit"
        )
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3LocalConfigurationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ForagerMatchedV3LocalConfigurationError(
            "local configuration payload is not strict UTF-8 JSON"
        ) from exc
    canonical = _canonical_json_bytes(
        decoded,
        label="parsed local configuration payload",
    )
    if not hmac.compare_digest(raw, canonical):
        raise ForagerMatchedV3LocalConfigurationError(
            "local configuration payload is not canonical JSON"
        )
    expected = build_matched_v3_local_configuration(exact_candidate_id)
    if not hmac.compare_digest(raw, expected.canonical_json_bytes):
        raise ForagerMatchedV3LocalConfigurationError(
            f"local configuration payload does not match {exact_candidate_id}"
        )
    return expected


def matched_v3_local_configuration_source_descriptor() -> dict[str, Any]:
    """Return a detached plain-JSON source descriptor."""
    return cast(
        dict[str, Any],
        json.loads(_SOURCE_DESCRIPTOR_BYTES.decode("utf-8")),
    )


def canonical_matched_v3_local_configuration_source_descriptor_bytes() -> bytes:
    """Return canonical bytes for the stable source descriptor."""
    return _SOURCE_DESCRIPTOR_BYTES


def matched_v3_local_configuration_builder_descriptor() -> dict[str, Any]:
    """Return a detached plain-JSON builder descriptor."""
    return cast(
        dict[str, Any],
        json.loads(_BUILDER_DESCRIPTOR_BYTES.decode("utf-8")),
    )


def canonical_matched_v3_local_configuration_builder_descriptor_bytes() -> bytes:
    """Return canonical bytes for the stable builder descriptor."""
    return _BUILDER_DESCRIPTOR_BYTES


__all__ = [
    "BUILDER_ID_BY_CANDIDATE",
    "EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE",
    "LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_CONFIGURATION_BUILDER_STATUS",
    "LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SCHEMA_VERSION",
    "MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256",
    "MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS",
    "MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256",
    "BuiltMatchedV3LocalConfiguration",
    "ForagerMatchedV3LocalConfigurationError",
    "build_all_matched_v3_local_configurations",
    "build_matched_v3_local_configuration",
    "canonical_matched_v3_local_configuration_builder_descriptor_bytes",
    "canonical_matched_v3_local_configuration_bytes",
    "canonical_matched_v3_local_configuration_source_descriptor_bytes",
    "matched_v3_local_configuration_builder_descriptor",
    "matched_v3_local_configuration_sha256",
    "matched_v3_local_configuration_source_descriptor",
    "parse_matched_v3_local_configuration_payload",
]
