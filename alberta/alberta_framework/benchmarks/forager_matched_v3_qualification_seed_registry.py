"""Pure deterministic seed-registry content for Forager matched-v3 qualification.

The caller supplies one complete drand Quicknet pulse record and the detached
identities of an independently managed trust-root receipt.  This module checks
their canonical structure, derives the exact ordered 28-case seed registry, and
can replay that derivation from canonical bytes with independent caller pins.

This is deliberately not a pulse fetcher, a BLS verifier, a seed issuer, a
chronology authority, or a production registry.  Quicknet authenticates only
its unchained round message.  The pulse timestamp is deterministic chain data;
by itself it does not prove when an external receipt was accepted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_registry_descriptor.v1"
)
QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_registry.v2"
)
QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_drand_pulse_record.v1"
)
QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_trust_root_receipt_identity.v1"
)
QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_trust_root_receipt.v2"
)
QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_derivation.v1"
)
QUALIFICATION_SEED_DERIVATION_DOMAIN: Final = (
    "alberta.forager.matched_v3.public_qualification.seed.v1"
)
QUALIFICATION_SEED_REGISTRY_STATUS: Final = (
    "implemented_deterministic_derivation_no_offline_signature_verifier_no_issuer"
)
QUALIFICATION_SEED_REGISTRY_CLASSIFICATION: Final = (
    "pure_content_replay_unauthenticated_non_authorizing_nonproduction"
)
QUALIFICATION_SEED_MATERIAL_CLASS: Final = (
    "public_nonbenchmark_permanently_consumed"
)
QUALIFICATION_SEED_DERIVATION_ALGORITHM: Final = "sha256_canonical_json_high31_v1"

QUICKNET_PROVIDER_ID: Final = "drand_quicknet"
QUICKNET_CHAIN_HASH: Final = (
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
QUICKNET_SIGNATURE_SCHEME: Final = "bls-unchained-g1-rfc9380"
QUICKNET_PUBLIC_KEY_HEX: Final = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183"
    "c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4"
    "bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
QUICKNET_PUBLIC_KEY_RAW_SHA256: Final = (
    "96e74fcdd3a118406d3800a4e4935e67450a6befde915d47a0d6a13519cee134"
)
QUICKNET_GENESIS_TIME_UNIX: Final = 1_692_803_367
QUICKNET_PERIOD_SECONDS: Final = 3
QUICKNET_BLS_MESSAGE_SCOPE: Final = "unchained_round_only"
QUICKNET_RANDOMNESS_DERIVATION: Final = "sha256_raw_signature_bytes"
QUICKNET_TIMESTAMP_SOURCE: Final = "drand_quicknet_round_time"
QUICKNET_RAW_SIGNATURE_BYTES: Final = 48
UINT31_MAX: Final = (1 << 31) - 1

_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 48
_MAX_JSON_NODES: Final = 10_000
_MAX_TEXT_LENGTH: Final = 1_024
_MAX_INTEGER: Final = (1 << 63) - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_RAW_SIGNATURE_RE: Final = re.compile(rf"[0-9a-f]{{{QUICKNET_RAW_SIGNATURE_BYTES * 2}}}\Z")

MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS: Final = (
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
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
    "random_policy",
    "search_nearest",
    "search_oracle",
)

if (
    len(QUICKNET_PUBLIC_KEY_HEX) != 192
    or hashlib.sha256(bytes.fromhex(QUICKNET_PUBLIC_KEY_HEX)).hexdigest()
    != QUICKNET_PUBLIC_KEY_RAW_SHA256
):
    raise AssertionError("frozen Quicknet public-key identity drifted")


class ForagerMatchedV3QualificationSeedRegistryError(ValueError):
    """Seed-registry content or an independent caller pin failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3QualificationSeedRegistryError(
        f"seed-registry JSON contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3QualificationSeedRegistryError(
        f"seed-registry JSON contains forbidden float {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry JSON integer exceeds its lexical bound"
        )
    result = int(value)
    if not -_MAX_INTEGER <= result <= _MAX_INTEGER:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry JSON integer exceeds its value bound"
        )
    return result


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                f"seed-registry JSON contains duplicate key {key!r}"
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
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "seed-registry JSON exceeds its node limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "seed-registry JSON exceeds its depth limit"
            )
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                raise ForagerMatchedV3QualificationSeedRegistryError(
                    "seed-registry JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                raise ForagerMatchedV3QualificationSeedRegistryError(
                    "seed-registry JSON integer exceeds its value bound"
                )
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "seed-registry JSON must contain only exact JSON types"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "seed-registry JSON contains an aliased or cyclic container"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3QualificationSeedRegistryError(
                        "seed-registry JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = cast(dict[str, Any], left)
        right_dict = cast(dict[str, Any], right)
        return left_dict.keys() == right_dict.keys() and all(
            _exact_json_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if type(left) is list:
        left_list = left
        right_list = cast(list[Any], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode one bounded, plain, unaliased object as canonical ASCII JSON."""

    if type(value) is not dict:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "canonical seed-registry JSON root must be one plain object"
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
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact exceeds its byte limit"
        )
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact must be exact bytes"
        )
    if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact violates its byte bound"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact must have one canonical trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3QualificationSeedRegistryError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact is not bounded strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact root must be one plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(canonical_json_bytes(result), raw):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "seed-registry artifact is not in exact canonical form"
        )
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3QualificationSeedRegistryError(f"{label} keys are not exact")


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _require_flag(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            f"{label} must remain exactly {expected}"
        )


@dataclass(frozen=True, slots=True)
class QuicknetPulseRecord:
    """Caller-carried raw Quicknet round content; no signature verification claim."""

    schema_version: str
    provider_id: str
    provider_chain_hash: str
    signature_scheme: str
    provider_public_key_hex: str
    provider_public_key_raw_sha256: str
    beacon_round: int
    beacon_time_unix: int
    raw_signature_hex: str
    raw_signature_sha256: str
    randomness_hex: str
    bls_message_scope: str
    randomness_derivation: str
    timestamp_source: str
    offline_signature_verification_required: bool
    offline_signature_verified_here: bool
    cryptographic_authentication_accepted_here: bool

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet pulse-record schema drifted"
            )
        if self.provider_id != QUICKNET_PROVIDER_ID:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse provider is not the frozen Quicknet provider"
            )
        if self.provider_chain_hash != QUICKNET_CHAIN_HASH:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse Quicknet chain hash drifted"
            )
        if self.signature_scheme != QUICKNET_SIGNATURE_SCHEME:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse Quicknet signature scheme drifted"
            )
        if self.provider_public_key_hex != QUICKNET_PUBLIC_KEY_HEX:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse Quicknet raw public key drifted"
            )
        if self.provider_public_key_raw_sha256 != QUICKNET_PUBLIC_KEY_RAW_SHA256:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse Quicknet public-key digest drifted"
            )
        _require_exact_int(self.beacon_round, "Quicknet round", minimum=1)
        _require_exact_int(self.beacon_time_unix, "Quicknet round time", minimum=1)
        expected_time = QUICKNET_GENESIS_TIME_UNIX + (
            self.beacon_round - 1
        ) * QUICKNET_PERIOD_SECONDS
        if self.beacon_time_unix != expected_time:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "pulse time is not the deterministic Quicknet round time"
            )
        if (
            type(self.raw_signature_hex) is not str
            or _RAW_SIGNATURE_RE.fullmatch(self.raw_signature_hex) is None
            or self.raw_signature_hex == "0" * (QUICKNET_RAW_SIGNATURE_BYTES * 2)
        ):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet raw signature must be one nonzero 48-byte lowercase hex value"
            )
        expected_randomness = hashlib.sha256(bytes.fromhex(self.raw_signature_hex)).hexdigest()
        _require_sha256(self.raw_signature_sha256, "Quicknet raw-signature digest")
        _require_sha256(self.randomness_hex, "Quicknet randomness")
        if not hmac.compare_digest(self.raw_signature_sha256, expected_randomness):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet raw-signature digest is not derived from the raw signature"
            )
        if not hmac.compare_digest(self.randomness_hex, expected_randomness):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet randomness is not SHA-256 of the raw signature"
            )
        if self.bls_message_scope != QUICKNET_BLS_MESSAGE_SCOPE:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet signature scope must remain the unchained round only"
            )
        if self.randomness_derivation != QUICKNET_RANDOMNESS_DERIVATION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet randomness derivation drifted"
            )
        if self.timestamp_source != QUICKNET_TIMESTAMP_SOURCE:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "Quicknet timestamp source drifted"
            )
        _require_flag(
            self.offline_signature_verification_required,
            True,
            "offline signature verification requirement",
        )
        _require_flag(
            self.offline_signature_verified_here,
            False,
            "offline signature verification result",
        )
        _require_flag(
            self.cryptographic_authentication_accepted_here,
            False,
            "cryptographic authentication acceptance",
        )

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_chain_hash": self.provider_chain_hash,
            "signature_scheme": self.signature_scheme,
            "provider_public_key_hex": self.provider_public_key_hex,
            "provider_public_key_raw_sha256": self.provider_public_key_raw_sha256,
            "beacon_round": self.beacon_round,
            "beacon_time_unix": self.beacon_time_unix,
            "raw_signature_hex": self.raw_signature_hex,
            "raw_signature_sha256": self.raw_signature_sha256,
            "randomness_hex": self.randomness_hex,
            "bls_message_scope": self.bls_message_scope,
            "randomness_derivation": self.randomness_derivation,
            "timestamp_source": self.timestamp_source,
            "offline_signature_verification_required": (
                self.offline_signature_verification_required
            ),
            "offline_signature_verified_here": self.offline_signature_verified_here,
            "cryptographic_authentication_accepted_here": (
                self.cryptographic_authentication_accepted_here
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.to_body_dict()
        return {**body, "pulse_record_body_sha256": self.body_sha256}

    @property
    def body_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_body_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustRootReceiptIdentity:
    """Detached caller-carried identity of an externally managed receipt."""

    schema_version: str
    receipt_schema_version: str
    receipt_file_sha256: str
    receipt_body_sha256: str
    provider_id: str
    provider_chain_hash: str
    signature_scheme: str
    provider_public_key_raw_sha256: str
    pulse_record_schema_version: str
    pulse_record_file_sha256: str
    pulse_record_body_sha256: str
    beacon_round: int
    beacon_time_unix: int
    observation_cutoff_unix: int
    raw_signature_sha256: str
    randomness_hex: str
    offline_signature_verification_required: bool
    offline_signature_verified_here: bool
    external_preacceptance_required: bool
    external_preacceptance_accepted_here: bool
    preacceptance_chronology_required: bool
    preacceptance_chronology_accepted_here: bool

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_IDENTITY_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "trust-root receipt identity schema drifted"
            )
        if self.receipt_schema_version != QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "trust-root receipt artifact schema drifted"
            )
        _require_sha256(self.receipt_file_sha256, "trust-root receipt file")
        _require_sha256(self.receipt_body_sha256, "trust-root receipt body")
        if (
            self.provider_id != QUICKNET_PROVIDER_ID
            or self.provider_chain_hash != QUICKNET_CHAIN_HASH
            or self.signature_scheme != QUICKNET_SIGNATURE_SCHEME
            or self.provider_public_key_raw_sha256 != QUICKNET_PUBLIC_KEY_RAW_SHA256
        ):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "trust-root receipt does not carry the exact Quicknet identity"
            )
        if self.pulse_record_schema_version != QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "trust-root receipt pulse-record schema drifted"
            )
        _require_sha256(self.pulse_record_file_sha256, "receipt-bound pulse-record file")
        _require_sha256(self.pulse_record_body_sha256, "receipt-bound pulse-record body")
        _require_exact_int(self.beacon_round, "receipt-bound Quicknet round", minimum=1)
        _require_exact_int(self.beacon_time_unix, "receipt-bound Quicknet round time", minimum=1)
        _require_exact_int(
            self.observation_cutoff_unix,
            "receipt-bound observation cutoff",
            minimum=1,
        )
        expected_beacon_time = QUICKNET_GENESIS_TIME_UNIX + (
            self.beacon_round - 1
        ) * QUICKNET_PERIOD_SECONDS
        if self.beacon_time_unix != expected_beacon_time:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "receipt-bound beacon time is not the deterministic Quicknet round time"
            )
        if self.beacon_time_unix >= self.observation_cutoff_unix:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "receipt-bound beacon time must precede the observation cutoff"
            )
        _require_sha256(self.raw_signature_sha256, "receipt-bound raw signature")
        _require_sha256(self.randomness_hex, "receipt-bound randomness")
        if not hmac.compare_digest(self.randomness_hex, self.raw_signature_sha256):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "receipt-bound randomness must equal SHA-256 of the raw signature"
            )
        _require_flag(
            self.offline_signature_verification_required,
            True,
            "receipt offline signature verification requirement",
        )
        _require_flag(
            self.offline_signature_verified_here,
            False,
            "receipt offline signature verification result",
        )
        _require_flag(
            self.external_preacceptance_required,
            True,
            "external receipt preacceptance requirement",
        )
        _require_flag(
            self.external_preacceptance_accepted_here,
            False,
            "external receipt preacceptance acceptance",
        )
        _require_flag(
            self.preacceptance_chronology_required,
            True,
            "external preacceptance chronology requirement",
        )
        _require_flag(
            self.preacceptance_chronology_accepted_here,
            False,
            "external preacceptance chronology acceptance",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_schema_version": self.receipt_schema_version,
            "receipt_file_sha256": self.receipt_file_sha256,
            "receipt_body_sha256": self.receipt_body_sha256,
            "provider_id": self.provider_id,
            "provider_chain_hash": self.provider_chain_hash,
            "signature_scheme": self.signature_scheme,
            "provider_public_key_raw_sha256": self.provider_public_key_raw_sha256,
            "pulse_record_schema_version": self.pulse_record_schema_version,
            "pulse_record_file_sha256": self.pulse_record_file_sha256,
            "pulse_record_body_sha256": self.pulse_record_body_sha256,
            "beacon_round": self.beacon_round,
            "beacon_time_unix": self.beacon_time_unix,
            "observation_cutoff_unix": self.observation_cutoff_unix,
            "raw_signature_sha256": self.raw_signature_sha256,
            "randomness_hex": self.randomness_hex,
            "offline_signature_verification_required": (
                self.offline_signature_verification_required
            ),
            "offline_signature_verified_here": self.offline_signature_verified_here,
            "external_preacceptance_required": self.external_preacceptance_required,
            "external_preacceptance_accepted_here": (
                self.external_preacceptance_accepted_here
            ),
            "preacceptance_chronology_required": self.preacceptance_chronology_required,
            "preacceptance_chronology_accepted_here": (
                self.preacceptance_chronology_accepted_here
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class QualificationSeedAuthorityState:
    """All authentication and authority facts that remain deliberately unaccepted."""

    deterministic_derivation_implemented_here: bool
    offline_signature_verification_required: bool
    offline_signature_verified_here: bool
    external_preacceptance_required: bool
    external_preacceptance_accepted_here: bool
    preacceptance_chronology_required: bool
    preacceptance_chronology_accepted_here: bool
    pulse_time_alone_proves_preacceptance_chronology: bool
    quicknet_signature_authenticates_round_only: bool
    quicknet_signature_authenticates_registry: bool
    quicknet_signature_authenticates_trust_root_receipt: bool
    qualification_cases_issued_here: bool
    production_registry: bool
    execution_authorized: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        expected = {
            "deterministic_derivation_implemented_here": True,
            "offline_signature_verification_required": True,
            "offline_signature_verified_here": False,
            "external_preacceptance_required": True,
            "external_preacceptance_accepted_here": False,
            "preacceptance_chronology_required": True,
            "preacceptance_chronology_accepted_here": False,
            "pulse_time_alone_proves_preacceptance_chronology": False,
            "quicknet_signature_authenticates_round_only": True,
            "quicknet_signature_authenticates_registry": False,
            "quicknet_signature_authenticates_trust_root_receipt": False,
            "qualification_cases_issued_here": False,
            "production_registry": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
        }
        for field_name, expected_value in expected.items():
            _require_flag(getattr(self, field_name), expected_value, field_name)

    def to_dict(self) -> dict[str, bool]:
        return {
            "deterministic_derivation_implemented_here": (
                self.deterministic_derivation_implemented_here
            ),
            "offline_signature_verification_required": (
                self.offline_signature_verification_required
            ),
            "offline_signature_verified_here": self.offline_signature_verified_here,
            "external_preacceptance_required": self.external_preacceptance_required,
            "external_preacceptance_accepted_here": (
                self.external_preacceptance_accepted_here
            ),
            "preacceptance_chronology_required": self.preacceptance_chronology_required,
            "preacceptance_chronology_accepted_here": (
                self.preacceptance_chronology_accepted_here
            ),
            "pulse_time_alone_proves_preacceptance_chronology": (
                self.pulse_time_alone_proves_preacceptance_chronology
            ),
            "quicknet_signature_authenticates_round_only": (
                self.quicknet_signature_authenticates_round_only
            ),
            "quicknet_signature_authenticates_registry": (
                self.quicknet_signature_authenticates_registry
            ),
            "quicknet_signature_authenticates_trust_root_receipt": (
                self.quicknet_signature_authenticates_trust_root_receipt
            ),
            "qualification_cases_issued_here": self.qualification_cases_issued_here,
            "production_registry": self.production_registry,
            "execution_authorized": self.execution_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


@dataclass(frozen=True, slots=True)
class QualificationSeedCase:
    """One deterministic case record; equal numeric roots remain valid collisions."""

    case_id: str
    candidate_id: str
    material_class: str
    registry_case_ordinal: int
    derivation_payload_sha256: str
    environment_seed: int
    agent_seed: int
    environment_seed_derivation_sha256: str
    agent_seed_derivation_sha256: str

    def __post_init__(self) -> None:
        if self.candidate_id not in MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed case candidate is unknown"
            )
        ordinal = _require_exact_int(
            self.registry_case_ordinal,
            "qualification seed case ordinal",
            maximum=len(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS) - 1,
        )
        if MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[ordinal] != self.candidate_id:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed case candidate order drifted"
            )
        if self.case_id != f"qualification_{ordinal:02d}_{self.candidate_id}":
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed case ID drifted"
            )
        if self.material_class != QUALIFICATION_SEED_MATERIAL_CLASS:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed case material class drifted"
            )
        _require_sha256(self.derivation_payload_sha256, "case derivation payload")
        _require_exact_int(self.environment_seed, "environment seed", maximum=UINT31_MAX)
        _require_exact_int(self.agent_seed, "agent seed", maximum=UINT31_MAX)
        _require_sha256(
            self.environment_seed_derivation_sha256,
            "environment-seed derivation",
        )
        _require_sha256(self.agent_seed_derivation_sha256, "agent-seed derivation")
        expected_environment_seed = (
            int.from_bytes(bytes.fromhex(self.environment_seed_derivation_sha256)[:4], "big")
            & UINT31_MAX
        )
        expected_agent_seed = (
            int.from_bytes(bytes.fromhex(self.agent_seed_derivation_sha256)[:4], "big")
            & UINT31_MAX
        )
        if self.environment_seed != expected_environment_seed:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "environment seed is not the uint31 projection of its derivation identity"
            )
        if self.agent_seed != expected_agent_seed:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "agent seed is not the uint31 projection of its derivation identity"
            )
        if hmac.compare_digest(
            self.environment_seed_derivation_sha256,
            self.agent_seed_derivation_sha256,
        ):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "environment and agent derivation identities must remain distinct"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "material_class": self.material_class,
            "registry_case_ordinal": self.registry_case_ordinal,
            "derivation_payload_sha256": self.derivation_payload_sha256,
            "environment_seed": self.environment_seed,
            "agent_seed": self.agent_seed,
            "environment_seed_derivation_sha256": (
                self.environment_seed_derivation_sha256
            ),
            "agent_seed_derivation_sha256": self.agent_seed_derivation_sha256,
        }


def uint31_seed_from_derivation_sha256(value: str) -> int:
    """Project one exact derivation identity to uint31 without an inequality rule."""

    digest = _require_sha256(value, "seed derivation identity")
    return int.from_bytes(bytes.fromhex(digest)[:4], "big") & UINT31_MAX


def _derive_case(
    pulse_record: QuicknetPulseRecord, candidate_id: str, ordinal: int
) -> QualificationSeedCase:
    payload = {
        "schema_version": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        "domain": QUALIFICATION_SEED_DERIVATION_DOMAIN,
        "algorithm": QUALIFICATION_SEED_DERIVATION_ALGORITHM,
        "provider_chain_hash": pulse_record.provider_chain_hash,
        "beacon_round": pulse_record.beacon_round,
        "beacon_randomness_hex": pulse_record.randomness_hex,
        "candidate_id": candidate_id,
        "registry_case_ordinal": ordinal,
    }
    payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    lane_digests = tuple(
        hashlib.sha256(canonical_json_bytes({**payload, "lane": lane})).hexdigest()
        for lane in ("environment", "agent")
    )
    return QualificationSeedCase(
        case_id=f"qualification_{ordinal:02d}_{candidate_id}",
        candidate_id=candidate_id,
        material_class=QUALIFICATION_SEED_MATERIAL_CLASS,
        registry_case_ordinal=ordinal,
        derivation_payload_sha256=payload_sha256,
        environment_seed=uint31_seed_from_derivation_sha256(lane_digests[0]),
        agent_seed=uint31_seed_from_derivation_sha256(lane_digests[1]),
        environment_seed_derivation_sha256=lane_digests[0],
        agent_seed_derivation_sha256=lane_digests[1],
    )


def _derive_cases(pulse_record: QuicknetPulseRecord) -> tuple[QualificationSeedCase, ...]:
    return tuple(
        _derive_case(pulse_record, candidate_id, ordinal)
        for ordinal, candidate_id in enumerate(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS)
    )


def _authority_state() -> QualificationSeedAuthorityState:
    return QualificationSeedAuthorityState(
        deterministic_derivation_implemented_here=True,
        offline_signature_verification_required=True,
        offline_signature_verified_here=False,
        external_preacceptance_required=True,
        external_preacceptance_accepted_here=False,
        preacceptance_chronology_required=True,
        preacceptance_chronology_accepted_here=False,
        pulse_time_alone_proves_preacceptance_chronology=False,
        quicknet_signature_authenticates_round_only=True,
        quicknet_signature_authenticates_registry=False,
        quicknet_signature_authenticates_trust_root_receipt=False,
        qualification_cases_issued_here=False,
        production_registry=False,
        execution_authorized=False,
        scientific_promotion_allowed=False,
    )


def _validate_pulse_receipt_binding(
    pulse_record: QuicknetPulseRecord,
    receipt_identity: TrustRootReceiptIdentity,
) -> None:
    expected = (
        pulse_record.provider_id,
        pulse_record.provider_chain_hash,
        pulse_record.signature_scheme,
        pulse_record.provider_public_key_raw_sha256,
        pulse_record.schema_version,
        pulse_record.file_sha256,
        pulse_record.body_sha256,
        pulse_record.beacon_round,
        pulse_record.beacon_time_unix,
        pulse_record.raw_signature_sha256,
        pulse_record.randomness_hex,
    )
    observed = (
        receipt_identity.provider_id,
        receipt_identity.provider_chain_hash,
        receipt_identity.signature_scheme,
        receipt_identity.provider_public_key_raw_sha256,
        receipt_identity.pulse_record_schema_version,
        receipt_identity.pulse_record_file_sha256,
        receipt_identity.pulse_record_body_sha256,
        receipt_identity.beacon_round,
        receipt_identity.beacon_time_unix,
        receipt_identity.raw_signature_sha256,
        receipt_identity.randomness_hex,
    )
    if observed != expected:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt identity is cross-wired from its Quicknet pulse"
        )


@dataclass(frozen=True, slots=True)
class QualificationSeedRegistry:
    """Immutable deterministic registry content; no issuance or authority semantics."""

    schema_version: str
    status: str
    material_class: str
    pulse_record: QuicknetPulseRecord
    trust_root_receipt_identity: TrustRootReceiptIdentity
    derivation_schema_version: str
    derivation_domain: str
    derivation_algorithm: str
    candidate_order: tuple[str, ...]
    cases: tuple[QualificationSeedCase, ...]
    authority: QualificationSeedAuthorityState

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed registry schema drifted"
            )
        if self.status != QUALIFICATION_SEED_REGISTRY_STATUS:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed registry status drifted"
            )
        if self.material_class != QUALIFICATION_SEED_MATERIAL_CLASS:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed registry material class drifted"
            )
        if type(self.pulse_record) is not QuicknetPulseRecord:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed registry pulse type is invalid"
            )
        if type(self.trust_root_receipt_identity) is not TrustRootReceiptIdentity:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed registry receipt identity type is invalid"
            )
        if self.derivation_schema_version != QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed derivation schema drifted"
            )
        if self.derivation_domain != QUALIFICATION_SEED_DERIVATION_DOMAIN:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed derivation domain drifted"
            )
        if self.derivation_algorithm != QUALIFICATION_SEED_DERIVATION_ALGORITHM:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "qualification seed derivation algorithm drifted"
            )
        if (
            type(self.candidate_order) is not tuple
            or self.candidate_order != MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
        ):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "registry must bind the exact qualification-v2 28-candidate order"
            )
        if type(self.cases) is not tuple or any(
            type(item) is not QualificationSeedCase for item in self.cases
        ):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "registry cases must be exact immutable case records"
            )
        if type(self.authority) is not QualificationSeedAuthorityState:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "registry authority state type is invalid"
            )
        _validate_pulse_receipt_binding(
            self.pulse_record,
            self.trust_root_receipt_identity,
        )
        expected_cases = _derive_cases(self.pulse_record)
        if self.cases != expected_cases:
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "registry cases do not exactly replay the deterministic pulse derivation"
            )
        derivation_identities = tuple(
            digest
            for item in self.cases
            for digest in (
                item.derivation_payload_sha256,
                item.environment_seed_derivation_sha256,
                item.agent_seed_derivation_sha256,
            )
        )
        if len(derivation_identities) != len(set(derivation_identities)):
            raise ForagerMatchedV3QualificationSeedRegistryError(
                "registry derivation identities are not globally distinct"
            )

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "material_class": self.material_class,
            "pulse_record": self.pulse_record.to_dict(),
            "trust_root_receipt_identity": self.trust_root_receipt_identity.to_dict(),
            "trust_root_receipt_identity_binding_sha256": (
                self.trust_root_receipt_identity.binding_sha256
            ),
            "derivation": {
                "schema_version": self.derivation_schema_version,
                "domain": self.derivation_domain,
                "algorithm": self.derivation_algorithm,
            },
            "candidate_order": list(self.candidate_order),
            "cases": [item.to_dict() for item in self.cases],
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.to_body_dict()
        return {**body, "registry_body_sha256": self.body_sha256}

    @property
    def body_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_body_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def canonical_quicknet_pulse_record_bytes(pulse_record: QuicknetPulseRecord) -> bytes:
    """Serialize one structurally validated, caller-supplied pulse record."""

    if type(pulse_record) is not QuicknetPulseRecord:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse record must be one exact immutable record"
        )
    return canonical_json_bytes(pulse_record.to_dict())


def canonical_trust_root_receipt_identity_bytes(
    receipt_identity: TrustRootReceiptIdentity,
) -> bytes:
    """Serialize the detached receipt identity, not the external receipt itself."""

    if type(receipt_identity) is not TrustRootReceiptIdentity:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt identity must be one exact immutable record"
        )
    return canonical_json_bytes(receipt_identity.to_dict())


def _validate_caller_pins(
    pulse_record: QuicknetPulseRecord,
    receipt_identity: TrustRootReceiptIdentity,
    *,
    expected_pulse_record_file_sha256: str,
    expected_pulse_record_body_sha256: str,
    expected_trust_root_receipt_file_sha256: str,
    expected_trust_root_receipt_body_sha256: str,
    expected_trust_root_receipt_binding_sha256: str,
) -> None:
    expected_pulse_file = _require_sha256(
        expected_pulse_record_file_sha256,
        "independently expected pulse-record file",
    )
    expected_pulse_body = _require_sha256(
        expected_pulse_record_body_sha256,
        "independently expected pulse-record body",
    )
    expected_receipt_file = _require_sha256(
        expected_trust_root_receipt_file_sha256,
        "independently expected trust-root receipt file",
    )
    expected_receipt_body = _require_sha256(
        expected_trust_root_receipt_body_sha256,
        "independently expected trust-root receipt body",
    )
    expected_receipt_binding = _require_sha256(
        expected_trust_root_receipt_binding_sha256,
        "independently expected trust-root receipt identity binding",
    )
    if not hmac.compare_digest(pulse_record.file_sha256, expected_pulse_file):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record full-file digest differs from its caller pin"
        )
    if not hmac.compare_digest(pulse_record.body_sha256, expected_pulse_body):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record body digest differs from its caller pin"
        )
    if not hmac.compare_digest(receipt_identity.receipt_file_sha256, expected_receipt_file):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt full-file digest differs from its caller pin"
        )
    if not hmac.compare_digest(receipt_identity.receipt_body_sha256, expected_receipt_body):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt body digest differs from its caller pin"
        )
    if not hmac.compare_digest(receipt_identity.binding_sha256, expected_receipt_binding):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt identity binding differs from its caller pin"
        )
    _validate_pulse_receipt_binding(pulse_record, receipt_identity)


def derive_matched_v3_qualification_seed_registry(
    pulse_record: QuicknetPulseRecord,
    receipt_identity: TrustRootReceiptIdentity,
    *,
    expected_pulse_record_file_sha256: str,
    expected_pulse_record_body_sha256: str,
    expected_trust_root_receipt_file_sha256: str,
    expected_trust_root_receipt_body_sha256: str,
    expected_trust_root_receipt_binding_sha256: str,
) -> QualificationSeedRegistry:
    """Derive content only from caller-carried records and independent pins."""

    if type(pulse_record) is not QuicknetPulseRecord:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse record must be one exact immutable record"
        )
    if type(receipt_identity) is not TrustRootReceiptIdentity:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "receipt identity must be one exact immutable record"
        )
    _validate_caller_pins(
        pulse_record,
        receipt_identity,
        expected_pulse_record_file_sha256=expected_pulse_record_file_sha256,
        expected_pulse_record_body_sha256=expected_pulse_record_body_sha256,
        expected_trust_root_receipt_file_sha256=(
            expected_trust_root_receipt_file_sha256
        ),
        expected_trust_root_receipt_body_sha256=(
            expected_trust_root_receipt_body_sha256
        ),
        expected_trust_root_receipt_binding_sha256=(
            expected_trust_root_receipt_binding_sha256
        ),
    )
    return QualificationSeedRegistry(
        schema_version=QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        status=QUALIFICATION_SEED_REGISTRY_STATUS,
        material_class=QUALIFICATION_SEED_MATERIAL_CLASS,
        pulse_record=pulse_record,
        trust_root_receipt_identity=receipt_identity,
        derivation_schema_version=QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        derivation_domain=QUALIFICATION_SEED_DERIVATION_DOMAIN,
        derivation_algorithm=QUALIFICATION_SEED_DERIVATION_ALGORITHM,
        candidate_order=MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS,
        cases=_derive_cases(pulse_record),
        authority=_authority_state(),
    )


def canonical_matched_v3_qualification_seed_registry_bytes(
    registry: QualificationSeedRegistry,
) -> bytes:
    """Serialize one fully replayed deterministic registry."""

    if type(registry) is not QualificationSeedRegistry:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry must be one exact immutable record"
        )
    return canonical_json_bytes(registry.to_dict())


def _parse_pulse_record_dict(value: Mapping[str, Any]) -> QuicknetPulseRecord:
    _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "provider_id",
                "provider_chain_hash",
                "signature_scheme",
                "provider_public_key_hex",
                "provider_public_key_raw_sha256",
                "beacon_round",
                "beacon_time_unix",
                "raw_signature_hex",
                "raw_signature_sha256",
                "randomness_hex",
                "bls_message_scope",
                "randomness_derivation",
                "timestamp_source",
                "offline_signature_verification_required",
                "offline_signature_verified_here",
                "cryptographic_authentication_accepted_here",
                "pulse_record_body_sha256",
            }
        ),
        "Quicknet pulse record",
    )
    body = dict(value)
    supplied_body_sha256 = body.pop("pulse_record_body_sha256")
    _require_sha256(supplied_body_sha256, "pulse-record body")
    expected_body_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, expected_body_sha256):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record body digest disagrees"
        )
    result = QuicknetPulseRecord(
        schema_version=value["schema_version"],
        provider_id=value["provider_id"],
        provider_chain_hash=value["provider_chain_hash"],
        signature_scheme=value["signature_scheme"],
        provider_public_key_hex=value["provider_public_key_hex"],
        provider_public_key_raw_sha256=value["provider_public_key_raw_sha256"],
        beacon_round=value["beacon_round"],
        beacon_time_unix=value["beacon_time_unix"],
        raw_signature_hex=value["raw_signature_hex"],
        raw_signature_sha256=value["raw_signature_sha256"],
        randomness_hex=value["randomness_hex"],
        bls_message_scope=value["bls_message_scope"],
        randomness_derivation=value["randomness_derivation"],
        timestamp_source=value["timestamp_source"],
        offline_signature_verification_required=(
            value["offline_signature_verification_required"]
        ),
        offline_signature_verified_here=value["offline_signature_verified_here"],
        cryptographic_authentication_accepted_here=(
            value["cryptographic_authentication_accepted_here"]
        ),
    )
    if not _exact_json_equal(result.to_dict(), value):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record representation is not exact"
        )
    return result


def _parse_receipt_identity_dict(value: Mapping[str, Any]) -> TrustRootReceiptIdentity:
    _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "receipt_schema_version",
                "receipt_file_sha256",
                "receipt_body_sha256",
                "provider_id",
                "provider_chain_hash",
                "signature_scheme",
                "provider_public_key_raw_sha256",
                "pulse_record_schema_version",
                "pulse_record_file_sha256",
                "pulse_record_body_sha256",
                "beacon_round",
                "beacon_time_unix",
                "observation_cutoff_unix",
                "raw_signature_sha256",
                "randomness_hex",
                "offline_signature_verification_required",
                "offline_signature_verified_here",
                "external_preacceptance_required",
                "external_preacceptance_accepted_here",
                "preacceptance_chronology_required",
                "preacceptance_chronology_accepted_here",
            }
        ),
        "trust-root receipt identity",
    )
    return TrustRootReceiptIdentity(
        schema_version=value["schema_version"],
        receipt_schema_version=value["receipt_schema_version"],
        receipt_file_sha256=value["receipt_file_sha256"],
        receipt_body_sha256=value["receipt_body_sha256"],
        provider_id=value["provider_id"],
        provider_chain_hash=value["provider_chain_hash"],
        signature_scheme=value["signature_scheme"],
        provider_public_key_raw_sha256=value["provider_public_key_raw_sha256"],
        pulse_record_schema_version=value["pulse_record_schema_version"],
        pulse_record_file_sha256=value["pulse_record_file_sha256"],
        pulse_record_body_sha256=value["pulse_record_body_sha256"],
        beacon_round=value["beacon_round"],
        beacon_time_unix=value["beacon_time_unix"],
        observation_cutoff_unix=value["observation_cutoff_unix"],
        raw_signature_sha256=value["raw_signature_sha256"],
        randomness_hex=value["randomness_hex"],
        offline_signature_verification_required=(
            value["offline_signature_verification_required"]
        ),
        offline_signature_verified_here=value["offline_signature_verified_here"],
        external_preacceptance_required=value["external_preacceptance_required"],
        external_preacceptance_accepted_here=(
            value["external_preacceptance_accepted_here"]
        ),
        preacceptance_chronology_required=value["preacceptance_chronology_required"],
        preacceptance_chronology_accepted_here=(
            value["preacceptance_chronology_accepted_here"]
        ),
    )


def _parse_authority_dict(value: Mapping[str, Any]) -> QualificationSeedAuthorityState:
    expected_keys = frozenset(
        {
            "deterministic_derivation_implemented_here",
            "offline_signature_verification_required",
            "offline_signature_verified_here",
            "external_preacceptance_required",
            "external_preacceptance_accepted_here",
            "preacceptance_chronology_required",
            "preacceptance_chronology_accepted_here",
            "pulse_time_alone_proves_preacceptance_chronology",
            "quicknet_signature_authenticates_round_only",
            "quicknet_signature_authenticates_registry",
            "quicknet_signature_authenticates_trust_root_receipt",
            "qualification_cases_issued_here",
            "production_registry",
            "execution_authorized",
            "scientific_promotion_allowed",
        }
    )
    _require_exact_keys(value, expected_keys, "qualification seed authority state")
    return QualificationSeedAuthorityState(
        deterministic_derivation_implemented_here=(
            value["deterministic_derivation_implemented_here"]
        ),
        offline_signature_verification_required=(
            value["offline_signature_verification_required"]
        ),
        offline_signature_verified_here=value["offline_signature_verified_here"],
        external_preacceptance_required=value["external_preacceptance_required"],
        external_preacceptance_accepted_here=(
            value["external_preacceptance_accepted_here"]
        ),
        preacceptance_chronology_required=value["preacceptance_chronology_required"],
        preacceptance_chronology_accepted_here=(
            value["preacceptance_chronology_accepted_here"]
        ),
        pulse_time_alone_proves_preacceptance_chronology=(
            value["pulse_time_alone_proves_preacceptance_chronology"]
        ),
        quicknet_signature_authenticates_round_only=(
            value["quicknet_signature_authenticates_round_only"]
        ),
        quicknet_signature_authenticates_registry=(
            value["quicknet_signature_authenticates_registry"]
        ),
        quicknet_signature_authenticates_trust_root_receipt=(
            value["quicknet_signature_authenticates_trust_root_receipt"]
        ),
        qualification_cases_issued_here=value["qualification_cases_issued_here"],
        production_registry=value["production_registry"],
        execution_authorized=value["execution_authorized"],
        scientific_promotion_allowed=value["scientific_promotion_allowed"],
    )


def _parse_case_dict(value: Mapping[str, Any]) -> QualificationSeedCase:
    _require_exact_keys(
        value,
        frozenset(
            {
                "case_id",
                "candidate_id",
                "material_class",
                "registry_case_ordinal",
                "derivation_payload_sha256",
                "environment_seed",
                "agent_seed",
                "environment_seed_derivation_sha256",
                "agent_seed_derivation_sha256",
            }
        ),
        "qualification seed case",
    )
    return QualificationSeedCase(
        case_id=value["case_id"],
        candidate_id=value["candidate_id"],
        material_class=value["material_class"],
        registry_case_ordinal=value["registry_case_ordinal"],
        derivation_payload_sha256=value["derivation_payload_sha256"],
        environment_seed=value["environment_seed"],
        agent_seed=value["agent_seed"],
        environment_seed_derivation_sha256=(
            value["environment_seed_derivation_sha256"]
        ),
        agent_seed_derivation_sha256=value["agent_seed_derivation_sha256"],
    )


def _parse_registry_dict(value: Mapping[str, Any]) -> QualificationSeedRegistry:
    _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "material_class",
                "pulse_record",
                "trust_root_receipt_identity",
                "trust_root_receipt_identity_binding_sha256",
                "derivation",
                "candidate_order",
                "cases",
                "authority",
                "registry_body_sha256",
            }
        ),
        "qualification seed registry",
    )
    body = dict(value)
    supplied_body_sha256 = body.pop("registry_body_sha256")
    _require_sha256(supplied_body_sha256, "qualification seed registry body")
    expected_body_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, expected_body_sha256):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry body digest disagrees"
        )
    pulse_value = value["pulse_record"]
    receipt_value = value["trust_root_receipt_identity"]
    derivation_value = value["derivation"]
    candidate_order_value = value["candidate_order"]
    cases_value = value["cases"]
    authority_value = value["authority"]
    if type(pulse_value) is not dict or type(receipt_value) is not dict:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "registry pulse and receipt identities must be plain objects"
        )
    if type(derivation_value) is not dict:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "registry derivation must be one plain object"
        )
    _require_exact_keys(
        derivation_value,
        frozenset({"schema_version", "domain", "algorithm"}),
        "registry derivation",
    )
    if type(candidate_order_value) is not list or any(
        type(item) is not str for item in candidate_order_value
    ):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "registry candidate order must be one exact string list"
        )
    if type(cases_value) is not list or any(type(item) is not dict for item in cases_value):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "registry cases must be one exact object list"
        )
    if type(authority_value) is not dict:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "registry authority must be one plain object"
        )
    result = QualificationSeedRegistry(
        schema_version=value["schema_version"],
        status=value["status"],
        material_class=value["material_class"],
        pulse_record=_parse_pulse_record_dict(pulse_value),
        trust_root_receipt_identity=_parse_receipt_identity_dict(receipt_value),
        derivation_schema_version=derivation_value["schema_version"],
        derivation_domain=derivation_value["domain"],
        derivation_algorithm=derivation_value["algorithm"],
        candidate_order=tuple(candidate_order_value),
        cases=tuple(_parse_case_dict(item) for item in cases_value),
        authority=_parse_authority_dict(authority_value),
    )
    if not _exact_json_equal(result.to_dict(), value):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry representation is not exact"
        )
    return result


def parse_quicknet_pulse_record_artifact(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> QuicknetPulseRecord:
    """Parse raw pulse content only with independent full-file and body pins."""

    expected_file = _require_sha256(expected_file_sha256, "expected pulse-record file")
    expected_body = _require_sha256(expected_body_sha256, "expected pulse-record body")
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record artifact must be exact bytes"
        )
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_file):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record full-file digest disagrees"
        )
    result = _parse_pulse_record_dict(_strict_json_load(raw))
    if not hmac.compare_digest(result.body_sha256, expected_body):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "pulse-record body digest disagrees with its caller pin"
        )
    return result


def parse_trust_root_receipt_identity_binding(
    raw: bytes,
    *,
    expected_binding_sha256: str,
    expected_receipt_file_sha256: str,
    expected_receipt_body_sha256: str,
) -> TrustRootReceiptIdentity:
    """Parse a detached identity binding, never the external receipt itself."""

    expected_binding = _require_sha256(
        expected_binding_sha256,
        "expected trust-root receipt identity binding",
    )
    expected_receipt_file = _require_sha256(
        expected_receipt_file_sha256,
        "expected trust-root receipt file",
    )
    expected_receipt_body = _require_sha256(
        expected_receipt_body_sha256,
        "expected trust-root receipt body",
    )
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt identity binding must be exact bytes"
        )
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_binding):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt identity binding digest disagrees"
        )
    result = _parse_receipt_identity_dict(_strict_json_load(raw))
    if not hmac.compare_digest(result.receipt_file_sha256, expected_receipt_file):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt file identity disagrees with its caller pin"
        )
    if not hmac.compare_digest(result.receipt_body_sha256, expected_receipt_body):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "trust-root receipt body identity disagrees with its caller pin"
        )
    return result


def parse_matched_v3_qualification_seed_registry_artifact(
    raw: bytes,
    *,
    expected_registry_file_sha256: str,
    expected_registry_body_sha256: str,
    expected_pulse_record_file_sha256: str,
    expected_pulse_record_body_sha256: str,
    expected_trust_root_receipt_file_sha256: str,
    expected_trust_root_receipt_body_sha256: str,
    expected_trust_root_receipt_binding_sha256: str,
) -> QualificationSeedRegistry:
    """Strictly parse, independently pin, and replay all 28 derivations."""

    expected_registry_file = _require_sha256(
        expected_registry_file_sha256,
        "expected qualification seed registry file",
    )
    expected_registry_body = _require_sha256(
        expected_registry_body_sha256,
        "expected qualification seed registry body",
    )
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry artifact must be exact bytes"
        )
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_registry_file):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry full-file digest disagrees"
        )
    result = _parse_registry_dict(_strict_json_load(raw))
    if not hmac.compare_digest(result.body_sha256, expected_registry_body):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed registry body digest differs from its caller pin"
        )
    _validate_caller_pins(
        result.pulse_record,
        result.trust_root_receipt_identity,
        expected_pulse_record_file_sha256=expected_pulse_record_file_sha256,
        expected_pulse_record_body_sha256=expected_pulse_record_body_sha256,
        expected_trust_root_receipt_file_sha256=(
            expected_trust_root_receipt_file_sha256
        ),
        expected_trust_root_receipt_body_sha256=(
            expected_trust_root_receipt_body_sha256
        ),
        expected_trust_root_receipt_binding_sha256=(
            expected_trust_root_receipt_binding_sha256
        ),
    )
    return result


def _canonicalization_descriptor() -> dict[str, Any]:
    return {
        "format": "json",
        "encoding": "ascii",
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
        "floats_allowed": False,
        "separators": [",", ":"],
        "trailing_newline": True,
        "duplicate_keys_rejected": True,
        "container_aliases_and_cycles_rejected": True,
        "bool_integer_confusion_rejected": True,
        "maximum_bytes": _MAX_ARTIFACT_BYTES,
        "maximum_depth": _MAX_JSON_DEPTH,
        "maximum_nodes": _MAX_JSON_NODES,
        "maximum_text_length": _MAX_TEXT_LENGTH,
        "maximum_integer": _MAX_INTEGER,
    }


def _descriptor() -> dict[str, Any]:
    candidate_order_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {"candidate_order": list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS)}
        )
    ).hexdigest()
    return {
        "schema_version": QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SCHEMA_VERSION,
        "status": QUALIFICATION_SEED_REGISTRY_STATUS,
        "classification": QUALIFICATION_SEED_REGISTRY_CLASSIFICATION,
        "schemas": {
            "pulse_record": QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
            "trust_root_receipt_identity": (
                QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_IDENTITY_SCHEMA_VERSION
            ),
            "trust_root_receipt": QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION,
            "registry": QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
            "derivation": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        },
        "candidate_order": list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS),
        "candidate_order_sha256": candidate_order_sha256,
        "candidate_count": len(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS),
        "material_class": QUALIFICATION_SEED_MATERIAL_CLASS,
        "quicknet": {
            "provider_id": QUICKNET_PROVIDER_ID,
            "chain_hash": QUICKNET_CHAIN_HASH,
            "signature_scheme": QUICKNET_SIGNATURE_SCHEME,
            "provider_public_key_hex": QUICKNET_PUBLIC_KEY_HEX,
            "provider_public_key_raw_sha256": QUICKNET_PUBLIC_KEY_RAW_SHA256,
            "provider_public_key_hash_input": "hex_decoded_96_bytes",
            "raw_signature_bytes": QUICKNET_RAW_SIGNATURE_BYTES,
            "genesis_time_unix": QUICKNET_GENESIS_TIME_UNIX,
            "period_seconds": QUICKNET_PERIOD_SECONDS,
            "bls_message_scope": QUICKNET_BLS_MESSAGE_SCOPE,
            "randomness_derivation": QUICKNET_RANDOMNESS_DERIVATION,
            "timestamp_source": QUICKNET_TIMESTAMP_SOURCE,
            "raw_public_key_signature_and_randomness_required": True,
            "signature_authenticates_round_only": True,
            "signature_authenticates_registry": False,
            "signature_authenticates_trust_root_receipt": False,
        },
        "derivation": {
            "schema_version": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
            "domain": QUALIFICATION_SEED_DERIVATION_DOMAIN,
            "algorithm": QUALIFICATION_SEED_DERIVATION_ALGORITHM,
            "candidate_order_exact": True,
            "case_count": 28,
            "lanes": ["environment", "agent"],
            "numeric_seed_domain": "uint31",
            "numeric_seed_collisions_allowed": True,
            "numeric_seed_inequality_required": False,
            "derivation_identity_distinctness_required": True,
        },
        "caller_pins": {
            "pulse_record_full_file_sha256_required": True,
            "pulse_record_body_sha256_required": True,
            "trust_root_receipt_full_file_sha256_required": True,
            "trust_root_receipt_body_sha256_required": True,
            "trust_root_receipt_identity_binding_sha256_required": True,
            "registry_full_file_sha256_required_on_parse": True,
            "registry_body_sha256_required_on_parse": True,
            "receipt_bytes_parsed_here": False,
        },
        "authentication": {
            "offline_signature_verification_required": True,
            "offline_signature_verification_implemented_here": False,
            "offline_signature_verified_here": False,
            "external_preacceptance_required": True,
            "external_preacceptance_accepted_here": False,
            "preacceptance_chronology_required": True,
            "preacceptance_chronology_accepted_here": False,
            "beacon_time_precedes_observation_cutoff_structurally": True,
            "pulse_time_alone_proves_preacceptance_chronology": False,
        },
        "canonicalization": _canonicalization_descriptor(),
        "capabilities": {
            "network_or_fetch_api_exposed": False,
            "filesystem_access": False,
            "clock_access": False,
            "bls_verifier_implemented": False,
            "default_pulse_available": False,
            "seed_issuer_api_exposed": False,
            "chronology_acceptor_implemented": False,
            "publication_implemented": False,
            "production_registry_available": False,
            "authority_granted": False,
        },
        "claims": {
            "deterministic_derivation_implemented": True,
            "cryptographic_authentication_accepted": False,
            "external_preacceptance_accepted": False,
            "qualification_cases_issued": False,
            "production_registry": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
        },
        "limitations": [
            "Quicknet signs only its unchained round message, not this registry or receipt.",
            "Pulse time alone is not evidence of external preacceptance chronology.",
            "The caller-carried receipt identities are not authenticated or accepted here.",
            "No pulse is fetched and no default or production pulse is embedded.",
            "No seed case is issued, reserved, published, or granted execution authority.",
        ],
    }


_DESCRIPTOR_BYTES: Final = canonical_json_bytes(_descriptor())
QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256: Final = (
    "fba1ab637f72de87c926169f2e0df5e66a8a2c7dcf855f00442a33dbe42fbef2"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 qualification seed-registry descriptor drifted")


def matched_v3_qualification_seed_registry_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the pure seed-registry descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_qualification_seed_registry_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_qualification_seed_registry_descriptor_sha256() -> str:
    """Return the frozen descriptor full-file digest."""

    return QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256


def parse_matched_v3_qualification_seed_registry_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Accept only the exact canonical descriptor and frozen identity."""

    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3QualificationSeedRegistryError(
            "qualification seed-registry descriptor identity drifted"
        )
    return value


__all__ = [
    "ForagerMatchedV3QualificationSeedRegistryError",
    "MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS",
    "QUALIFICATION_SEED_DERIVATION_ALGORITHM",
    "QUALIFICATION_SEED_DERIVATION_DOMAIN",
    "QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION",
    "QUALIFICATION_SEED_MATERIAL_CLASS",
    "QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION",
    "QUALIFICATION_SEED_REGISTRY_CLASSIFICATION",
    "QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256",
    "QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION",
    "QUALIFICATION_SEED_REGISTRY_STATUS",
    "QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_IDENTITY_SCHEMA_VERSION",
    "QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION",
    "QUICKNET_BLS_MESSAGE_SCOPE",
    "QUICKNET_CHAIN_HASH",
    "QUICKNET_GENESIS_TIME_UNIX",
    "QUICKNET_PERIOD_SECONDS",
    "QUICKNET_PROVIDER_ID",
    "QUICKNET_PUBLIC_KEY_HEX",
    "QUICKNET_PUBLIC_KEY_RAW_SHA256",
    "QUICKNET_RANDOMNESS_DERIVATION",
    "QUICKNET_RAW_SIGNATURE_BYTES",
    "QUICKNET_SIGNATURE_SCHEME",
    "QUICKNET_TIMESTAMP_SOURCE",
    "QualificationSeedAuthorityState",
    "QualificationSeedCase",
    "QualificationSeedRegistry",
    "QuicknetPulseRecord",
    "TrustRootReceiptIdentity",
    "UINT31_MAX",
    "canonical_json_bytes",
    "canonical_matched_v3_qualification_seed_registry_bytes",
    "canonical_matched_v3_qualification_seed_registry_descriptor_bytes",
    "canonical_quicknet_pulse_record_bytes",
    "canonical_trust_root_receipt_identity_bytes",
    "derive_matched_v3_qualification_seed_registry",
    "matched_v3_qualification_seed_registry_descriptor",
    "matched_v3_qualification_seed_registry_descriptor_sha256",
    "parse_matched_v3_qualification_seed_registry_artifact",
    "parse_matched_v3_qualification_seed_registry_descriptor",
    "parse_quicknet_pulse_record_artifact",
    "parse_trust_root_receipt_identity_binding",
    "uint31_seed_from_derivation_sha256",
]
