"""Source-only normalized publication commitments for matched Forager v3.

This module defines one canonical, case-specific metadata wrapper shared by the
local, external, and adapter publication families.  It neither publishes nor
reloads files.  The wrapper commits to the digest expected from a later reload;
the later host phase must perform that reload and validate equality itself.

Only role, name, size, and digest records cross this boundary.  No reward,
score, trace, payload, database, video, or manifest bytes are accepted.  A
canonical wrapper is not execution authority, an observation, a qualification
decision, or evidence.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, cast

QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_contract_descriptor.v1"
)
QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)

LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_metadata.v1"
)
LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_metadata.v1"
)
EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_publication_metadata.v1"
)
STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_publication_descriptor.v1"
)
ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_atomic_publication_descriptor.v1"
)
EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1"
)

QUALIFICATION_PUBLICATION_COMMITMENT_STATUS: Final = (
    "implemented_source_only_expected_reload_commitment_non_authorizing"
)
QUALIFICATION_PUBLICATION_COMMITMENT_CLASSIFICATION: Final = (
    "score_blind_metadata_only_normalized_commitment_non_authorizing"
)

MATCHED_V3_LOCAL_CANDIDATE_IDS: Final = (
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
MATCHED_V3_EXTERNAL_CANDIDATE_IDS: Final = (
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
MATCHED_V3_ADAPTER_CANDIDATE_IDS: Final = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
)
MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS: Final = (
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
)

LOCAL_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("local_bundle_manifest", "local-bundle-manifest.json"),
    ("bootstrap_receipt", "bootstrap-receipt.json"),
    ("bootstrap_child_record", "bootstrap-child-record.json"),
    ("local_runner_receipt", "local-runner-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
EXTERNAL_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("outcome_manifest", "external-outcome-manifest.json"),
    ("execution_receipt", "external-execution-receipt.json"),
    ("conversion_receipt", "external-conversion-receipt.json"),
    ("upstream_reward_npz", "upstream-reward.npz"),
    ("upstream_results_database", "upstream-results.db"),
    ("upstream_video_slot", "upstream-video-slot.bin"),
    ("canonical_reward_npz", "reward-trace.npz"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
ADAPTER_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("adapter_bundle_manifest", "adapter-bundle-manifest.json"),
    ("runner_result_receipt", "runner-result-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
)

MAX_PUBLICATION_FILE_BYTES: Final = 1024 * 1024 * 1024
MAX_PUBLICATION_TOTAL_BYTES: Final = 1024 * 1024 * 1024
EMPTY_FILE_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# These identities are content that obs-v2 already excludes from strict adapter
# publisher and strict adapter atomic-producer slots.  Retaining the complete
# set prevents an old runner or compiled addendum component from being relabelled.
INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
)
INCOMPATIBLE_ADAPTER_SOURCE_SHA256S: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
)
INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S: Final = frozenset(
    (*INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S, *INCOMPATIBLE_ADAPTER_SOURCE_SHA256S)
)

# Content keys and reverse-link receipt keys are rejected recursively before
# reconstruction.  Bare role nouns such as ``issuer`` and ``evaluator`` are
# deliberately absent so false capability envelopes cannot collide.
FORBIDDEN_CONTENT_KEYS: Final = frozenset(
    {
        "bundle_bytes",
        "content_bytes",
        "cumulative_reward",
        "database_bytes",
        "decoded_score",
        "file_bytes",
        "manifest_bytes",
        "mean_reward",
        "npz_bytes",
        "payload",
        "payload_bytes",
        "performance_score",
        "publication_bytes",
        "raw_result",
        "raw_results",
        "raw_reward",
        "result_bytes",
        "reward_magnitude",
        "reward_sum",
        "reward_total",
        "reward_trace",
        "score",
        "scores",
        "total_reward",
        "trace_bytes",
        "video_bytes",
    }
)
FORBIDDEN_REVERSE_BINDING_KEYS: Final = frozenset(
    {
        "acceptance_decision",
        "evaluator_receipt",
        "full_resource_merger_receipt",
        "host_execution_receipt",
        "host_success_receipt",
        "host_terminal_metadata",
        "issuer_receipt",
        "observation_handoff",
        "qualification_decision",
        "storage_boundary_receipt",
        "storage_write_seal",
        "terminal_metadata",
        "terminal_receipt",
    }
)
FORBIDDEN_RECURSIVE_KEYS: Final = FORBIDDEN_CONTENT_KEYS | FORBIDDEN_REVERSE_BINDING_KEYS

_MAX_ARTIFACT_BYTES: Final = 512 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_NODES: Final = 20_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_MAX_INTEGER_DIGITS: Final = 19
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


class ForagerMatchedV3QualificationPublicationCommitmentError(ValueError):
    """A normalized publication commitment failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationPublicationCommitmentError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact boolean")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded portable identifier")
    return value


def _require_file_name(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be exact text")
    if not value or len(value) > _MAX_TEXT_LENGTH:
        _fail(f"{label} must be bounded nonempty text")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        _fail(f"{label} must contain printable ASCII only")
    _require_identifier(value.replace(".", "_"), label)
    return value


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationPublicationCommitmentError(
            "publication commitment value is not canonical ASCII JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _body_sha256(value: object) -> str:
    return _sha256(_canonical_json(value, newline=False))


def _with_body_sha256(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(value)
    result[field_name] = _body_sha256(result)
    return result


def _require_body_digest(
    value: dict[str, Any],
    field_name: str,
    label: str,
) -> dict[str, Any]:
    body = dict(value)
    supplied = _require_sha256(body.pop(field_name, None), f"{label} body")
    if not hmac.compare_digest(supplied, _body_sha256(body)):
        _fail(f"{label} body digest differs")
    return body


def _reject_float(value: str) -> NoReturn:
    _fail(f"publication commitment JSON contains forbidden float {value!r}")


def _reject_constant(value: str) -> NoReturn:
    _fail(f"publication commitment JSON contains non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_INTEGER_DIGITS:
        _fail("publication commitment JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"publication commitment JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_bounds(value: object) -> None:
    nodes = 0

    def walk(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("publication commitment JSON exceeds its structural bound")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("publication commitment JSON text is outside printable ASCII bounds")
            return
        if type(item) is list:
            for child in cast(list[object], item):
                walk(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("publication commitment JSON object key is not text")
                walk(key, depth + 1)
                walk(child, depth + 1)
            return
        _fail("publication commitment JSON contains an unsupported value")

    walk(value, 0)


def _reject_forbidden_keys(value: object) -> None:
    if type(value) is dict:
        for key, child in cast(dict[object, object], value).items():
            if type(key) is not str:
                _fail("publication commitment JSON object key is not text")
            normalized = key.lower().replace("-", "_")
            if normalized in FORBIDDEN_RECURSIVE_KEYS:
                _fail(f"publication commitment contains forbidden key {key!r}")
            _reject_forbidden_keys(child)
    elif type(value) is list:
        for child in cast(list[object], value):
            _reject_forbidden_keys(child)


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("publication commitment JSON bytes are empty, inexact, or oversized")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3QualificationPublicationCommitmentError(
            "publication commitment JSON must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_int=_parse_int,
            parse_constant=_reject_constant,
        )
    except (RecursionError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3QualificationPublicationCommitmentError(
            "publication commitment JSON is invalid"
        ) from exc
    _validate_json_bounds(value)
    _reject_forbidden_keys(value)
    if type(value) is not dict:
        _fail("publication commitment JSON root must be one object")
    if not hmac.compare_digest(_canonical_json(value), raw):
        _fail("publication commitment JSON is not canonical")
    return cast(dict[str, Any], value)


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _validate_caller_file_pin(raw: bytes, expected_file_sha256: str, label: str) -> None:
    if type(raw) is not bytes:
        _fail(f"{label} bytes must use the exact bytes type")
    expected = _require_sha256(expected_file_sha256, f"{label} caller file pin")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail(f"{label} caller file pin differs")


def _family_for_candidate(candidate_id: str) -> Literal["local", "external", "adapter"]:
    if candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if candidate_id in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("publication commitment candidate is outside the exact 28-candidate order")


def _role_paths_for_family(
    family: Literal["local", "external", "adapter"],
) -> tuple[tuple[str, str], ...]:
    if family == "local":
        return LOCAL_PUBLICATION_ROLE_PATHS
    if family == "external":
        return EXTERNAL_PUBLICATION_ROLE_PATHS
    return ADAPTER_PUBLICATION_ROLE_PATHS


def _publisher_profile(
    family: Literal["local", "external", "adapter"],
) -> tuple[str, str, str, str | None]:
    if family == "local":
        return (
            LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            None,
        )
    if family == "external":
        return (
            EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        )
    return (
        STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION,
        STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    )


def _capabilities() -> dict[str, bool]:
    return {
        "acceptance_evaluation": False,
        "case_issuance": False,
        "content_value_decoding": False,
        "execution": False,
        "file_publication": False,
        "host_provisioning": False,
        "payload_byte_transport": False,
        "publication_reload": False,
        "reload_digest_equality_validation": False,
    }


def _readiness() -> dict[str, bool]:
    return {
        "host_execution_ready": False,
        "observation_ready": False,
        "publication_ready": False,
        "qualification_ready": False,
        "reload_observed": False,
    }


def _authority() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "observation_issuance_authorized": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "scientific_evidence_created": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "build_qualified": False,
        "performance_claim_allowed": False,
        "publisher_qualified": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "source_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The wrapper authenticates metadata commitments and never reads publication files.",
        "The reload digest is expected content; this wrapper performs no reload.",
        "A later phase must observe reload output and validate exact digest equality.",
        "Native receipts remain separate artifacts and are not replaced by this wrapper.",
        (
            "Canonical metadata grants no execution, observation, qualification, "
            "or evidence authority."
        ),
    ]


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """One schema-bound canonical artifact identity with no artifact bytes."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact body")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProducerIdentityV1:
    """One descriptor/source producer identity with no implementation loader."""

    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.descriptor_schema_version, "producer descriptor schema")
        _require_sha256(self.descriptor_sha256, "producer descriptor")
        _require_sha256(self.source_sha256, "producer source")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class PublicationFileRecordV1:
    """One immutable role/name/size/digest record; never file content."""

    role: str
    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, "publication file role")
        _require_file_name(self.name, "publication file name")
        _require_int(
            self.size_bytes,
            "publication file size",
            maximum=MAX_PUBLICATION_FILE_BYTES,
        )
        _require_sha256(self.sha256, "publication file")
        if (self.size_bytes == 0) is not (self.sha256 == EMPTY_FILE_SHA256):
            _fail("publication file zero length and empty digest must agree exactly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def publication_file_inventory_sha256(
    files: tuple[PublicationFileRecordV1, ...],
) -> str:
    """Hash the exact role-bearing inventory projection without file bytes."""

    if type(files) is not tuple or any(type(item) is not PublicationFileRecordV1 for item in files):
        raise TypeError("publication files must use one exact immutable record tuple")
    return _sha256(
        _canonical_json(
            {"files": [item.to_dict() for item in files]},
            newline=False,
        )
    )


@dataclass(frozen=True, slots=True)
class PublicationCommitmentBindingsV1:
    """Externally expected metadata bindings used to build and cross-check a wrapper."""

    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    publisher: ProducerIdentityV1
    publisher_metadata: ArtifactIdentityV1
    native_atomic_producer: ProducerIdentityV1
    native_publication_receipt: ArtifactIdentityV1 | None
    publication_address_sha256: str
    publication_manifest_file_sha256: str
    publication_manifest_body_sha256: str
    file_inventory_sha256: str
    published_bundle_sha256: str
    expected_reload_observation_sha256: str
    file_count: int
    total_size_bytes: int
    maximum_total_size_bytes: int
    video_slot_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ]
    files: tuple[PublicationFileRecordV1, ...]

    def __post_init__(self) -> None:
        _validate_bindings(self)

    def to_dict(self) -> dict[str, Any]:
        return _bindings_dict(self)


@dataclass(frozen=True, slots=True)
class QualificationPublicationCommitmentWrapperV1(PublicationCommitmentBindingsV1):
    """Canonical expected-reload commitment; it is not an observed reload receipt."""

    schema_version: str
    reload_performed_by_wrapper: bool
    reload_digest_equality_validated_by_wrapper: bool
    content_values_read_by_wrapper: bool
    payload_bytes_transported_by_wrapper: bool
    wrapper_body_sha256: str

    def __post_init__(self) -> None:
        PublicationCommitmentBindingsV1.__post_init__(self)
        if (
            type(self.schema_version) is not str
            or self.schema_version != QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION
        ):
            _fail("publication commitment wrapper schema differs")
        if (
            _require_bool(self.reload_performed_by_wrapper, "reload performed by wrapper")
            or _require_bool(
                self.reload_digest_equality_validated_by_wrapper,
                "reload digest equality validated by wrapper",
            )
            or _require_bool(self.content_values_read_by_wrapper, "content values read")
            or _require_bool(
                self.payload_bytes_transported_by_wrapper,
                "payload bytes transported",
            )
        ):
            _fail("publication commitment wrapper cannot claim operational work")
        supplied = _require_sha256(self.wrapper_body_sha256, "publication wrapper body")
        if not hmac.compare_digest(supplied, _body_sha256(self.to_body_dict())):
            _fail("publication commitment wrapper body digest differs")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": QUALIFICATION_PUBLICATION_COMMITMENT_STATUS,
            "classification": QUALIFICATION_PUBLICATION_COMMITMENT_CLASSIFICATION,
            **_bindings_dict(self),
            "reload_performed_by_wrapper": self.reload_performed_by_wrapper,
            "reload_digest_equality_validated_by_wrapper": (
                self.reload_digest_equality_validated_by_wrapper
            ),
            "content_values_read_by_wrapper": self.content_values_read_by_wrapper,
            "payload_bytes_transported_by_wrapper": (self.payload_bytes_transported_by_wrapper),
            "capabilities": _capabilities(),
            "readiness": _readiness(),
            "authority": _authority(),
            "claims": _claims(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.to_body_dict()
        body["wrapper_body_sha256"] = self.wrapper_body_sha256
        return body

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


def _bindings_dict(value: PublicationCommitmentBindingsV1) -> dict[str, Any]:
    return {
        "case_spine_sha256": value.case_spine_sha256,
        "case_ordinal": value.case_ordinal,
        "candidate_id": value.candidate_id,
        "candidate_family": value.candidate_family,
        "qualification_case_id": value.qualification_case_id,
        "publisher": value.publisher.to_dict(),
        "publisher_metadata": value.publisher_metadata.to_dict(),
        "native_atomic_producer": value.native_atomic_producer.to_dict(),
        "native_publication_receipt": (
            None
            if value.native_publication_receipt is None
            else value.native_publication_receipt.to_dict()
        ),
        "publication_address_sha256": value.publication_address_sha256,
        "publication_manifest_file_sha256": value.publication_manifest_file_sha256,
        "publication_manifest_body_sha256": value.publication_manifest_body_sha256,
        "file_inventory_sha256": value.file_inventory_sha256,
        "published_bundle_sha256": value.published_bundle_sha256,
        "expected_reload_observation_sha256": value.expected_reload_observation_sha256,
        "file_count": value.file_count,
        "total_size_bytes": value.total_size_bytes,
        "maximum_total_size_bytes": value.maximum_total_size_bytes,
        "video_slot_mode": value.video_slot_mode,
        "files": [item.to_dict() for item in value.files],
    }


def _validate_bindings(value: PublicationCommitmentBindingsV1) -> None:
    _require_sha256(value.case_spine_sha256, "publication case spine")
    ordinal = _require_int(
        value.case_ordinal,
        "publication case ordinal",
        maximum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS) - 1,
    )
    candidate_id = _require_identifier(value.candidate_id, "publication candidate ID")
    if candidate_id != MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS[ordinal]:
        _fail("publication ordinal and candidate differ from the exact global order")
    candidate_family = _require_identifier(
        value.candidate_family,
        "publication candidate family",
    )
    expected_family = _family_for_candidate(candidate_id)
    if candidate_family != expected_family:
        _fail("publication candidate family differs")
    qualification_case_id = _require_identifier(
        value.qualification_case_id,
        "publication qualification case ID",
    )
    if qualification_case_id != f"qualification_{ordinal:02d}_{candidate_id}":
        _fail("publication qualification case ID differs")
    if type(value.publisher) is not ProducerIdentityV1:
        _fail("publication publisher identity type differs")
    if type(value.publisher_metadata) is not ArtifactIdentityV1:
        _fail("publication publisher metadata identity type differs")
    if type(value.native_atomic_producer) is not ProducerIdentityV1:
        _fail("publication native atomic producer identity type differs")
    (
        expected_publisher_schema,
        expected_metadata_schema,
        expected_atomic_schema,
        expected_native_receipt_schema,
    ) = _publisher_profile(expected_family)
    if value.publisher.descriptor_schema_version != expected_publisher_schema:
        _fail("publication publisher descriptor schema differs from its family")
    if value.publisher_metadata.schema_version != expected_metadata_schema:
        _fail("publication publisher metadata schema differs from its family")
    if value.native_atomic_producer.descriptor_schema_version != expected_atomic_schema:
        _fail("publication native atomic producer schema differs from its family")
    if expected_native_receipt_schema is None:
        if value.native_publication_receipt is not None:
            _fail("local publication cannot carry a native publication receipt")
    else:
        if type(value.native_publication_receipt) is not ArtifactIdentityV1:
            _fail("nonlocal publication requires one exact native publication receipt")
        if value.native_publication_receipt.schema_version != expected_native_receipt_schema:
            _fail("native publication receipt schema differs from its family")
    adapter_identity_slots = (
        value.publisher.descriptor_sha256,
        value.publisher.source_sha256,
        value.native_atomic_producer.descriptor_sha256,
        value.native_atomic_producer.source_sha256,
    )
    if expected_family == "adapter" and any(
        digest in INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S for digest in adapter_identity_slots
    ):
        _fail("historical or unqualified adapter identity cannot fill a strict slot")
    for digest, label in (
        (value.publication_address_sha256, "publication address"),
        (value.publication_manifest_file_sha256, "publication manifest file"),
        (value.publication_manifest_body_sha256, "publication manifest body"),
        (value.file_inventory_sha256, "publication file inventory"),
        (value.published_bundle_sha256, "published bundle"),
        (value.expected_reload_observation_sha256, "expected reload observation"),
    ):
        _require_sha256(digest, label)
    if type(value.files) is not tuple or any(
        type(item) is not PublicationFileRecordV1 for item in value.files
    ):
        _fail("publication files must be one exact immutable tuple")
    expected_paths = _role_paths_for_family(expected_family)
    if tuple((item.role, item.name) for item in value.files) != expected_paths:
        _fail("publication role/name inventory differs from its family")
    if len({item.role for item in value.files}) != len(value.files) or len(
        {item.name for item in value.files}
    ) != len(value.files):
        _fail("publication role or filename is duplicated")
    empty_permitted_roles = {"stdout", "stderr", "upstream_video_slot"}
    if any(item.size_bytes == 0 and item.role not in empty_permitted_roles for item in value.files):
        _fail("publication canonical manifest, receipt, or data artifact is empty")
    if (
        value.publication_address_sha256 != value.files[0].sha256
        or value.publication_manifest_file_sha256 != value.files[0].sha256
    ):
        _fail("publication address and manifest file must equal publication.json SHA")
    file_count = _require_int(
        value.file_count,
        "publication file count",
        maximum=len(EXTERNAL_PUBLICATION_ROLE_PATHS),
    )
    if file_count != len(expected_paths):
        _fail("publication file count differs from its exact family inventory")
    total_size = sum(item.size_bytes for item in value.files)
    recorded_total_size = _require_int(
        value.total_size_bytes,
        "publication aggregate size",
        maximum=MAX_PUBLICATION_TOTAL_BYTES,
    )
    if recorded_total_size != total_size:
        _fail("publication aggregate size differs from its exact inventory")
    maximum_total_size = _require_int(
        value.maximum_total_size_bytes,
        "publication aggregate ceiling",
        maximum=MAX_PUBLICATION_TOTAL_BYTES,
    )
    if maximum_total_size != MAX_PUBLICATION_TOTAL_BYTES:
        _fail("publication aggregate ceiling must remain exact 1 GiB")
    if total_size > maximum_total_size:
        _fail("publication aggregate exceeds its frozen ceiling")
    if not hmac.compare_digest(
        value.file_inventory_sha256,
        publication_file_inventory_sha256(value.files),
    ):
        _fail("role-bearing publication inventory digest does not replay")
    video_slot_mode = _require_identifier(
        value.video_slot_mode,
        "publication video slot mode",
    )
    if video_slot_mode not in {
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    }:
        _fail("publication video slot mode differs")
    if expected_family == "external":
        video = value.files[
            EXTERNAL_PUBLICATION_ROLE_PATHS.index(
                ("upstream_video_slot", "upstream-video-slot.bin")
            )
        ]
        if candidate_id in MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS:
            if video_slot_mode != "opaque_ppo_video" or video.size_bytes < 1:
                _fail("PPO external publication requires one nonempty opaque video slot")
        elif (
            video_slot_mode != "absent_for_continuing_zero_length_slot"
            or video.size_bytes != 0
            or video.sha256 != EMPTY_FILE_SHA256
        ):
            _fail("continuing external publication requires the exact empty video sentinel")
    elif video_slot_mode != "not_applicable":
        _fail("local and adapter publications require video_slot_mode not_applicable")


def build_matched_v3_qualification_publication_commitment(
    bindings: PublicationCommitmentBindingsV1,
) -> QualificationPublicationCommitmentWrapperV1:
    """Normalize externally supplied metadata without publishing or reloading files."""

    if type(bindings) is not PublicationCommitmentBindingsV1:
        raise TypeError("publication commitment bindings must use the exact frozen type")
    provisional = {
        "schema_version": QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "status": QUALIFICATION_PUBLICATION_COMMITMENT_STATUS,
        "classification": QUALIFICATION_PUBLICATION_COMMITMENT_CLASSIFICATION,
        **bindings.to_dict(),
        "reload_performed_by_wrapper": False,
        "reload_digest_equality_validated_by_wrapper": False,
        "content_values_read_by_wrapper": False,
        "payload_bytes_transported_by_wrapper": False,
        "capabilities": _capabilities(),
        "readiness": _readiness(),
        "authority": _authority(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return QualificationPublicationCommitmentWrapperV1(
        **{name: getattr(bindings, name) for name in bindings.__dataclass_fields__},
        schema_version=QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        reload_performed_by_wrapper=False,
        reload_digest_equality_validated_by_wrapper=False,
        content_values_read_by_wrapper=False,
        payload_bytes_transported_by_wrapper=False,
        wrapper_body_sha256=_body_sha256(provisional),
    )


def canonical_matched_v3_qualification_publication_commitment_bytes(
    wrapper: QualificationPublicationCommitmentWrapperV1,
) -> bytes:
    """Serialize one exact metadata-only wrapper with one trailing LF."""

    if type(wrapper) is not QualificationPublicationCommitmentWrapperV1:
        raise TypeError("publication commitment wrapper must use the exact frozen type")
    return _canonical_json(wrapper.to_dict())


def validate_matched_v3_qualification_publication_commitment_bindings(
    wrapper: QualificationPublicationCommitmentWrapperV1,
    *,
    expected: PublicationCommitmentBindingsV1,
) -> None:
    """Cross-check all case and publication projections against external expectations."""

    if type(wrapper) is not QualificationPublicationCommitmentWrapperV1:
        raise TypeError("publication commitment wrapper must use the exact frozen type")
    if type(expected) is not PublicationCommitmentBindingsV1:
        raise TypeError("publication commitment expectations must use the exact frozen type")
    observed = PublicationCommitmentBindingsV1(
        **{name: getattr(wrapper, name) for name in expected.__dataclass_fields__}
    )
    if observed != expected:
        _fail("publication commitment differs from its external expected bindings")


def _artifact_identity_from_dict(value: object, label: str) -> ArtifactIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset({"schema_version", "file_sha256", "body_sha256"}),
        label,
    )
    return ArtifactIdentityV1(**item)


def _optional_artifact_identity_from_dict(
    value: object,
    label: str,
) -> ArtifactIdentityV1 | None:
    if value is None:
        return None
    return _artifact_identity_from_dict(value, label)


def _producer_identity_from_dict(value: object, label: str) -> ProducerIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset({"descriptor_schema_version", "descriptor_sha256", "source_sha256"}),
        label,
    )
    return ProducerIdentityV1(**item)


def _file_record_from_dict(value: object) -> PublicationFileRecordV1:
    item = _require_exact_keys(
        value,
        frozenset({"role", "name", "size_bytes", "sha256"}),
        "publication file record",
    )
    return PublicationFileRecordV1(**item)


_WRAPPER_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "classification",
        "case_spine_sha256",
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
        "publisher",
        "publisher_metadata",
        "native_atomic_producer",
        "native_publication_receipt",
        "publication_address_sha256",
        "publication_manifest_file_sha256",
        "publication_manifest_body_sha256",
        "file_inventory_sha256",
        "published_bundle_sha256",
        "expected_reload_observation_sha256",
        "file_count",
        "total_size_bytes",
        "maximum_total_size_bytes",
        "video_slot_mode",
        "files",
        "reload_performed_by_wrapper",
        "reload_digest_equality_validated_by_wrapper",
        "content_values_read_by_wrapper",
        "payload_bytes_transported_by_wrapper",
        "capabilities",
        "readiness",
        "authority",
        "claims",
        "limitations",
    }
)


def parse_matched_v3_qualification_publication_commitment(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> QualificationPublicationCommitmentWrapperV1:
    """Parse one wrapper only after verifying its caller-supplied full-file pin."""

    _validate_caller_file_pin(raw, expected_file_sha256, "publication commitment wrapper")
    value = _strict_json_load(raw)
    item = _require_exact_keys(
        value,
        _WRAPPER_BODY_KEYS | {"wrapper_body_sha256"},
        "publication commitment wrapper",
    )
    supplied_body = _require_sha256(
        item["wrapper_body_sha256"],
        "publication commitment wrapper body",
    )
    body = {key: child for key, child in item.items() if key != "wrapper_body_sha256"}
    if not hmac.compare_digest(supplied_body, _body_sha256(body)):
        _fail("publication commitment wrapper body digest differs")
    files_value = body["files"]
    if type(files_value) is not list:
        _fail("publication commitment files must be one list")
    wrapper = QualificationPublicationCommitmentWrapperV1(
        case_spine_sha256=body["case_spine_sha256"],
        case_ordinal=body["case_ordinal"],
        candidate_id=body["candidate_id"],
        candidate_family=body["candidate_family"],
        qualification_case_id=body["qualification_case_id"],
        publisher=_producer_identity_from_dict(body["publisher"], "publisher identity"),
        publisher_metadata=_artifact_identity_from_dict(
            body["publisher_metadata"],
            "publisher metadata identity",
        ),
        native_atomic_producer=_producer_identity_from_dict(
            body["native_atomic_producer"],
            "native atomic producer identity",
        ),
        native_publication_receipt=_optional_artifact_identity_from_dict(
            body["native_publication_receipt"],
            "native publication receipt identity",
        ),
        publication_address_sha256=body["publication_address_sha256"],
        publication_manifest_file_sha256=body["publication_manifest_file_sha256"],
        publication_manifest_body_sha256=body["publication_manifest_body_sha256"],
        file_inventory_sha256=body["file_inventory_sha256"],
        published_bundle_sha256=body["published_bundle_sha256"],
        expected_reload_observation_sha256=body["expected_reload_observation_sha256"],
        file_count=body["file_count"],
        total_size_bytes=body["total_size_bytes"],
        maximum_total_size_bytes=body["maximum_total_size_bytes"],
        video_slot_mode=body["video_slot_mode"],
        files=tuple(_file_record_from_dict(child) for child in files_value),
        schema_version=body["schema_version"],
        reload_performed_by_wrapper=body["reload_performed_by_wrapper"],
        reload_digest_equality_validated_by_wrapper=body[
            "reload_digest_equality_validated_by_wrapper"
        ],
        content_values_read_by_wrapper=body["content_values_read_by_wrapper"],
        payload_bytes_transported_by_wrapper=body["payload_bytes_transported_by_wrapper"],
        wrapper_body_sha256=supplied_body,
    )
    if raw != canonical_matched_v3_qualification_publication_commitment_bytes(wrapper):
        _fail("publication commitment wrapper canonical replay differs")
    return wrapper


def _contract_descriptor() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": (QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SCHEMA_VERSION),
        "status": QUALIFICATION_PUBLICATION_COMMITMENT_STATUS,
        "classification": QUALIFICATION_PUBLICATION_COMMITMENT_CLASSIFICATION,
        "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
        "full_file_caller_pin_required": True,
        "wrapper_schema_version": (QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION),
        "candidate_order": list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
        "candidate_families": {
            "local": list(MATCHED_V3_LOCAL_CANDIDATE_IDS),
            "external": list(MATCHED_V3_EXTERNAL_CANDIDATE_IDS),
            "adapter": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
        },
        "publication_role_paths": {
            "local": [list(item) for item in LOCAL_PUBLICATION_ROLE_PATHS],
            "external": [list(item) for item in EXTERNAL_PUBLICATION_ROLE_PATHS],
            "adapter": [list(item) for item in ADAPTER_PUBLICATION_ROLE_PATHS],
        },
        "publisher_descriptor_schemas": {
            "local": LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "external": EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "adapter": STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        },
        "publisher_metadata_schemas": {
            "local": LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            "external": EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            "adapter": STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION,
        },
        "native_atomic_descriptor_schemas": {
            "local": ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "external": ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "adapter": STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        },
        "native_receipt_schemas": {
            "local": None,
            "external": EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "adapter": STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        },
        "inventory_contract": {
            "digest_projection": "canonical_{files:[role,name,size_bytes,sha256]}",
            "role_bearing": True,
            "atomic_v2_roleless_digest_compatible": False,
            "maximum_file_bytes": MAX_PUBLICATION_FILE_BYTES,
            "maximum_total_bytes": MAX_PUBLICATION_TOTAL_BYTES,
            "address_equals_publication_manifest_file_sha256": True,
        },
        "reload_contract": {
            "expected_digest_committed": True,
            "reload_performed_here": False,
            "digest_equality_validated_here": False,
            "later_phase_validation_required": True,
        },
        "native_receipt_preserved_not_replaced": True,
        "forbidden_recursive_keys": sorted(FORBIDDEN_RECURSIVE_KEYS),
        "incompatible_adapter_descriptor_sha256s": list(INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S),
        "incompatible_adapter_source_sha256s": list(INCOMPATIBLE_ADAPTER_SOURCE_SHA256S),
        "incompatible_adapter_identity_sha256s": sorted(INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S),
        "public_api": {
            "metadata_builder": "build_matched_v3_qualification_publication_commitment",
            "canonical_serializer": (
                "canonical_matched_v3_qualification_publication_commitment_bytes"
            ),
            "strict_parser": "parse_matched_v3_qualification_publication_commitment",
            "external_binding_validator": (
                "validate_matched_v3_qualification_publication_commitment_bindings"
            ),
            "mutation_or_execution_apis": [],
        },
        "capabilities": _capabilities(),
        "readiness": _readiness(),
        "authority": _authority(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return _with_body_sha256(body, "descriptor_body_sha256")


_DESCRIPTOR: Final = _contract_descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR)

# Root replaces this only after static review and independent canonical replay.
PINNED_QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
)


def matched_v3_qualification_publication_commitment_contract_descriptor() -> dict[str, Any]:
    """Return detached source-only publication-commitment descriptor content."""

    return copy.deepcopy(_DESCRIPTOR)


def canonical_matched_v3_qualification_publication_commitment_contract_descriptor_bytes() -> bytes:
    """Return canonical descriptor bytes without bypassing the audit pin."""

    return bytes(_DESCRIPTOR_BYTES)


def matched_v3_qualification_publication_commitment_contract_descriptor_sha256() -> str:
    """Return the descriptor identity only after its literal is independently pinned."""

    observed = _sha256(_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(
        observed,
        PINNED_QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SHA256,
    ):
        _fail("publication commitment contract descriptor pin is not finalized or drifted")
    return observed


def parse_matched_v3_qualification_publication_commitment_contract_descriptor(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse the exact descriptor only after its literal pin is finalized."""

    _validate_caller_file_pin(raw, expected_file_sha256, "publication contract descriptor")
    value = _strict_json_load(raw)
    _require_body_digest(value, "descriptor_body_sha256", "publication contract descriptor")
    pinned = matched_v3_qualification_publication_commitment_contract_descriptor_sha256()
    if not hmac.compare_digest(expected_file_sha256, pinned):
        _fail("publication contract descriptor caller pin differs from literal pin")
    if raw != _DESCRIPTOR_BYTES:
        _fail("publication commitment contract descriptor differs")
    return copy.deepcopy(value)


__all__ = [
    "ADAPTER_PUBLICATION_ROLE_PATHS",
    "ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "ArtifactIdentityV1",
    "EMPTY_FILE_SHA256",
    "EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_ROLE_PATHS",
    "FORBIDDEN_RECURSIVE_KEYS",
    "ForagerMatchedV3QualificationPublicationCommitmentError",
    "INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S",
    "INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S",
    "INCOMPATIBLE_ADAPTER_SOURCE_SHA256S",
    "LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION",
    "LOCAL_PUBLICATION_ROLE_PATHS",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS",
    "MAX_PUBLICATION_FILE_BYTES",
    "MAX_PUBLICATION_TOTAL_BYTES",
    "PINNED_QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SHA256",
    "ProducerIdentityV1",
    "PublicationCommitmentBindingsV1",
    "PublicationFileRecordV1",
    "QUALIFICATION_PUBLICATION_COMMITMENT_CLASSIFICATION",
    "QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_PUBLICATION_COMMITMENT_STATUS",
    "QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION",
    "QualificationPublicationCommitmentWrapperV1",
    "STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION",
    "build_matched_v3_qualification_publication_commitment",
    "canonical_matched_v3_qualification_publication_commitment_bytes",
    "canonical_matched_v3_qualification_publication_commitment_contract_descriptor_bytes",
    "matched_v3_qualification_publication_commitment_contract_descriptor",
    "matched_v3_qualification_publication_commitment_contract_descriptor_sha256",
    "parse_matched_v3_qualification_publication_commitment",
    "parse_matched_v3_qualification_publication_commitment_contract_descriptor",
    "publication_file_inventory_sha256",
    "validate_matched_v3_qualification_publication_commitment_bindings",
]
