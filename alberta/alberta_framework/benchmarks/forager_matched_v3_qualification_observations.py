"""Strict score-blind observation envelopes for matched-v3 qualification.

This module defines content-only structural validators for the seven observation
schemas named by the matched-v3 qualification-plan v2 descriptor.  It has no
observation issuer, executor, filesystem API, clock, network client, default
artifact, acceptance decision, qualification bundle, or authority surface.

Every observation is one canonical ASCII JSON envelope.  The envelope binds a
caller-supplied qualification-plan digest, the registry descriptor, one exact
kind-specific payload, and authority-denying claims.  Payload booleans are
observed facts only: this module validates their exact type but deliberately does
not evaluate them or turn them into acceptance or qualification.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v2 as plan_v2

QUALIFICATION_OBSERVATION_ENVELOPE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_envelope.v1"
)
QUALIFICATION_OBSERVATION_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v1"
)
QUALIFICATION_OBSERVATION_STATUS: Final = (
    "implemented_structural_validators_no_observation_issuer"
)
QUALIFICATION_OBSERVATION_CLASSIFICATION: Final = (
    "score_blind_content_only_observation_structure_non_authorizing"
)

SOURCE_OBSERVATION_KIND: Final = "source_observation"
RUNTIME_OBSERVATION_KIND: Final = "runtime_observation"
QUALIFICATION_SEED_OBSERVATION_KIND: Final = "qualification_seed_observation"
CANDIDATE_OBSERVATION_KIND: Final = "candidate_observation"
RESOURCE_OBSERVATION_KIND: Final = "resource_observation"
RESULT_PUBLICATION_OBSERVATION_KIND: Final = "result_publication_observation"
FRESH_REPLAY_OBSERVATION_KIND: Final = "fresh_replay_observation"

SOURCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.source_observation.v1"
)
RUNTIME_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_observation.v1"
)
QUALIFICATION_SEED_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_observation.v1"
)
CANDIDATE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.candidate_observation.v1"
)
RESOURCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.resource_observation.v1"
)
RESULT_PUBLICATION_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.result_publication_observation.v1"
)
FRESH_REPLAY_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.fresh_replay_observation.v1"
)

QUALIFICATION_OBSERVATION_KINDS: Final = (
    SOURCE_OBSERVATION_KIND,
    RUNTIME_OBSERVATION_KIND,
    QUALIFICATION_SEED_OBSERVATION_KIND,
    CANDIDATE_OBSERVATION_KIND,
    RESOURCE_OBSERVATION_KIND,
    RESULT_PUBLICATION_OBSERVATION_KIND,
    FRESH_REPLAY_OBSERVATION_KIND,
)
QUALIFICATION_OBSERVATION_SCHEMA_VERSIONS: Final = (
    SOURCE_OBSERVATION_SCHEMA_VERSION,
    RUNTIME_OBSERVATION_SCHEMA_VERSION,
    QUALIFICATION_SEED_OBSERVATION_SCHEMA_VERSION,
    CANDIDATE_OBSERVATION_SCHEMA_VERSION,
    RESOURCE_OBSERVATION_SCHEMA_VERSION,
    RESULT_PUBLICATION_OBSERVATION_SCHEMA_VERSION,
    FRESH_REPLAY_OBSERVATION_SCHEMA_VERSION,
)
_SCHEMA_BY_KIND: Final = dict(
    zip(
        QUALIFICATION_OBSERVATION_KINDS,
        QUALIFICATION_OBSERVATION_SCHEMA_VERSIONS,
        strict=True,
    )
)

_CANDIDATE_IDS: Final = tuple(plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS)
_RESOURCE_FIELDS: Final = tuple(plan_v2.RESOURCE_CEILING_FIELDS)
_HELPER_IDS: Final = ("drand_verify", "oci_runtime", "resource_observer")
_EXPECTED_FORAGAX_INSTALL_TREE_SHA256: Final = (
    "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
)

_SEED_ACCEPTANCE_FIELDS: Final = (
    "registry_full_file_and_body_digests_exact",
    "independent_trust_root_receipt_file_pin_exact",
    "independent_trust_root_receipt_binding_pin_exact",
    "provider_chain_public_key_and_signature_scheme_exact",
    "pulse_record_exact",
    "beacon_round_time_signature_and_randomness_exact",
    "offline_verifier_source_closure_membership_exact",
    "offline_signature_verification_exact",
    "deterministic_28_case_seed_pair_derivation_exact",
    "deterministic_registry_file_and_body_digests_exact",
    "derivation_schema_and_domain_exact",
    "case_derivation_payload_membership_exact",
    "beacon_time_precedes_observation_cutoff_exact",
    "external_receipt_preacceptance_chronology_exact",
)
_CANDIDATE_ACCEPTANCE_FIELDS: Final = (
    "source_membership_exact",
    "configuration_membership_exact",
    "entrypoint_import_exact",
    "agent_seed_transport_exact",
    "environment_agent_derivations_distinct",
    "candidate_rng_membership_exact",
)
_RESOURCE_ACCEPTANCE_FIELDS: Final = (
    "horizon_accounting_exact",
    "reward_membership_structural_only",
    "all_resource_observations_within_predeclared_integer_ceilings",
)
_PUBLICATION_ACCEPTANCE_FIELDS: Final = (
    "publisher_descriptor_membership_exact",
    "publisher_source_closure_membership_exact",
    "reload_validator_membership_exact",
    "atomic_publication_exact",
    "strict_reload_exact",
    "full_file_digest_equivalence_exact",
    "score_and_reward_magnitude_not_decoded",
)
_FRESH_REPLAY_ACCEPTANCE_FIELDS: Final = (
    "environment_seed_transport_exact",
    "reset_step_key_schedule_exact",
    "structural_replay_exact",
)
_V2_SEED_COMPATIBILITY_FIELDS: Final = (
    "independent_trust_root_receipt_pins_exact",
    "preobservation_chronology_exact",
    "deterministic_case_seed_derivation_exact",
)
_V2_RESOURCE_COMPATIBILITY_FIELD: Final = (
    "resource_observations_within_predeclared_integer_ceilings"
)

_MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_MAX_SOURCE_ENTRIES: Final = 2_000_000
_MAX_PUBLICATION_FILES: Final = 100_000
_MAX_INTERACTIONS: Final = 2**63 - 1

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PORTABLE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\Z")
_FORBIDDEN_PAYLOAD_KEYS: Final = frozenset(
    {
        "acceptance",
        "accepted",
        "candidate_rank",
        "candidate_ranking",
        "cumulative_reward",
        "mean_reward",
        "performance_score",
        "rank",
        "ranking",
        "raw_reward",
        "reward_magnitude",
        "reward_sum",
        "reward_total",
        "score",
        "scores",
        "total_reward",
    }
)


class ForagerMatchedV3QualificationObservationError(ValueError):
    """An observation descriptor, envelope, payload, or caller pin failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationObservationError(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_image_id(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact sha256 image ID")
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


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or any(ord(character) < 0x20 for character in value)
    ):
        _fail(f"{label} must be one bounded nonempty exact text value")
    return value


def _require_portable_id(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _PORTABLE_ID_RE.fullmatch(text) is None:
        _fail(f"{label} is not one portable identifier")
    return text


def _require_relative_path(value: object, label: str) -> str:
    text = _require_text(value, label)
    if (
        _RELATIVE_PATH_RE.fullmatch(text) is None
        or any(component in {".", ".."} for component in text.split("/"))
    ):
        _fail(f"{label} is not one portable relative path")
    return text


def _require_candidate_id(value: object) -> str:
    candidate_id = _require_text(value, "candidate ID")
    if candidate_id not in _CANDIDATE_IDS:
        _fail("candidate ID is outside qualification-plan v2")
    return candidate_id


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"qualification observation JSON contains non-finite constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"qualification observation JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("qualification observation JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"qualification observation JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("qualification observation JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("qualification observation JSON exceeds its depth bound")
        if item is None or type(item) in {bool, int}:
            if type(item) is int:
                _require_int(
                    item,
                    "qualification observation JSON integer",
                    minimum=-_MAX_INTEGER,
                )
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH:
                _fail("qualification observation JSON string exceeds its length bound")
            return
        if type(item) not in {dict, list}:
            _fail("qualification observation JSON contains a non-JSON or inexact value")
        identity = id(item)
        if identity in seen:
            _fail("qualification observation JSON containers must be unaliased and acyclic")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        for key, child in cast(dict[object, object], item).items():
            if type(key) is not str or len(key) > _MAX_TEXT_LENGTH:
                _fail("qualification observation JSON object key is invalid")
            visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationObservationError(
            "qualification observation JSON cannot be canonically encoded"
        ) from exc
    if newline:
        raw += b"\n"
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("qualification observation JSON exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES
        or not raw.endswith(b"\n")
    ):
        _fail("qualification observation bytes are invalid, oversized, or unterminated")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationObservationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationObservationError(
            "qualification observation bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("qualification observation JSON root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(raw, _canonical_json(result)):
        _fail("qualification observation bytes are not exact canonical JSON")
    return result


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[str, object], left)
        right_mapping = cast(dict[str, object], right)
        return frozenset(left_mapping) == frozenset(right_mapping) and all(
            _exact_json_equal(left_mapping[key], right_mapping[key])
            for key in left_mapping
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _reject_forbidden_payload_keys(value: Mapping[str, Any]) -> None:
    pending: list[Mapping[str, Any]] = [value]
    while pending:
        current = pending.pop()
        for key, child in current.items():
            if key in _FORBIDDEN_PAYLOAD_KEYS:
                _fail(f"qualification observation payload contains forbidden field {key!r}")
            if type(child) is dict:
                pending.append(cast(dict[str, Any], child))
            elif type(child) is list:
                pending.extend(
                    cast(dict[str, Any], item)
                    for item in child
                    if type(item) is dict
                )


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "artifact_accepted": False,
        "evidence_authority_granted": False,
        "execution_authority_granted": False,
        "observation_issuer_available": False,
        "performance_claim_allowed": False,
        "production_plan_issued": False,
        "publication_authority_granted": False,
        "qualification_authority_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "seed_authority_granted": False,
        "universal_sota_claim_allowed": False,
        "workload_executed": False,
    }


def _limitations() -> list[str]:
    return [
        "This envelope validates structure and content identities only.",
        "No observation issuer, executor, filesystem, clock, or network capability exists.",
        "Observed booleans are recorded but are not evaluated as acceptance conditions.",
        "No default qualification plan, image, runtime, source, helper, case, or seed is bound.",
        "No payload may carry score, ranking, reward magnitude, or performance values.",
        "No envelope grants qualification, evidence, promotion, publication, or execution.",
    ]


@dataclass(frozen=True, slots=True)
class ExternalSourceObservationPayload:
    """Exact external materialization, staging, publication, and archive identities."""

    source_id: str
    producer_kind: str
    publication_receipt_schema_version: str
    publication_receipt_file_sha256: str
    publication_receipt_body_sha256: str
    publication_contract_descriptor_sha256: str
    materialization_manifest_schema_version: str
    materialization_manifest_file_sha256: str
    materialization_manifest_body_sha256: str
    materialization_payload_sha256: str
    source_tree_sha256: str
    source_inventory_sha256: str
    staging_manifest_schema_version: str
    staging_manifest_file_sha256: str
    staging_manifest_body_sha256: str
    archive_file_sha256: str
    archive_inventory_sha256: str
    tracked_entry_count: int
    materialized_file_count: int
    excluded_gitlink_count: int
    archive_member_count: int
    materialized_total_size_bytes: int
    archive_size_bytes: int
    producer_receipt_replay_exact: bool
    manifest_file_body_binding_exact: bool
    source_tree_inventory_exact: bool
    archive_inventory_exact: bool
    counts_exact: bool

    def __post_init__(self) -> None:
        if self.source_id != "external_foragax_agents":
            _fail("external source observation source ID differs")
        if self.producer_kind != "durable_external_source_publication_v1_materialization_v2":
            _fail("external source observation producer kind differs")
        if (
            self.publication_receipt_schema_version
            != plan_v2.EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
            or self.materialization_manifest_schema_version
            != plan_v2.EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
            or self.staging_manifest_schema_version
            != plan_v2.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION
        ):
            _fail("external source observation producer schema differs")
        for digest_value, label in (
            (self.publication_receipt_file_sha256, "external publication receipt file"),
            (self.publication_receipt_body_sha256, "external publication receipt body"),
            (self.publication_contract_descriptor_sha256, "external publication descriptor"),
            (self.materialization_manifest_file_sha256, "external materialization manifest"),
            (self.materialization_manifest_body_sha256, "external materialization body"),
            (self.materialization_payload_sha256, "external materialization payload"),
            (self.source_tree_sha256, "external source tree"),
            (self.source_inventory_sha256, "external source inventory"),
            (self.staging_manifest_file_sha256, "external staging manifest"),
            (self.staging_manifest_body_sha256, "external staging body"),
            (self.archive_file_sha256, "external archive file"),
            (self.archive_inventory_sha256, "external archive inventory"),
        ):
            _require_sha256(digest_value, label)
        for count_value, label in (
            (self.tracked_entry_count, "external tracked-entry count"),
            (self.materialized_file_count, "external materialized-file count"),
            (self.excluded_gitlink_count, "external excluded-gitlink count"),
            (self.archive_member_count, "external archive-member count"),
        ):
            _require_int(count_value, label, maximum=_MAX_SOURCE_ENTRIES)
        _require_int(self.materialized_total_size_bytes, "external materialized bytes")
        _require_int(self.archive_size_bytes, "external archive bytes", minimum=1)
        for flag_value, label in (
            (self.producer_receipt_replay_exact, "external producer receipt replay"),
            (self.manifest_file_body_binding_exact, "external manifest file/body binding"),
            (self.source_tree_inventory_exact, "external source tree inventory"),
            (self.archive_inventory_exact, "external archive inventory"),
            (self.counts_exact, "external counts"),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "producer_kind": self.producer_kind,
            "publication_receipt_schema_version": self.publication_receipt_schema_version,
            "publication_receipt_file_sha256": self.publication_receipt_file_sha256,
            "publication_receipt_body_sha256": self.publication_receipt_body_sha256,
            "publication_contract_descriptor_sha256": (
                self.publication_contract_descriptor_sha256
            ),
            "materialization_manifest_schema_version": (
                self.materialization_manifest_schema_version
            ),
            "materialization_manifest_file_sha256": self.materialization_manifest_file_sha256,
            "materialization_manifest_body_sha256": self.materialization_manifest_body_sha256,
            "materialization_payload_sha256": self.materialization_payload_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "staging_manifest_schema_version": self.staging_manifest_schema_version,
            "staging_manifest_file_sha256": self.staging_manifest_file_sha256,
            "staging_manifest_body_sha256": self.staging_manifest_body_sha256,
            "archive_file_sha256": self.archive_file_sha256,
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "tracked_entry_count": self.tracked_entry_count,
            "materialized_file_count": self.materialized_file_count,
            "excluded_gitlink_count": self.excluded_gitlink_count,
            "archive_member_count": self.archive_member_count,
            "materialized_total_size_bytes": self.materialized_total_size_bytes,
            "archive_size_bytes": self.archive_size_bytes,
            "producer_receipt_replay_exact": self.producer_receipt_replay_exact,
            "manifest_file_body_binding_exact": self.manifest_file_body_binding_exact,
            "source_tree_inventory_exact": self.source_tree_inventory_exact,
            "archive_inventory_exact": self.archive_inventory_exact,
            "counts_exact": self.counts_exact,
        }


@dataclass(frozen=True, slots=True)
class LocalSourceObservationPayload:
    """Exact local snapshot, bundle receipt, archive, tree, and count identities."""

    source_id: str
    producer_kind: str
    snapshot_descriptor_sha256: str
    snapshot_manifest_schema_version: str
    snapshot_manifest_file_sha256: str
    snapshot_manifest_body_sha256: str
    snapshot_tree_schema_version: str
    snapshot_tree_sha256: str
    bundle_descriptor_sha256: str
    bundle_receipt_schema_version: str
    bundle_receipt_file_sha256: str
    bundle_receipt_body_sha256: str
    archive_file_sha256: str
    member_inventory_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int
    archive_member_count: int
    archive_size_bytes: int
    producer_receipt_replay_exact: bool
    manifest_file_body_binding_exact: bool
    source_tree_inventory_exact: bool
    archive_inventory_exact: bool
    counts_exact: bool

    def __post_init__(self) -> None:
        if self.source_id != "local_alberta":
            _fail("local source observation source ID differs")
        if self.producer_kind != "local_source_snapshot_and_retained_bundle_v1":
            _fail("local source observation producer kind differs")
        if (
            self.snapshot_manifest_schema_version
            != plan_v2.LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
            or self.snapshot_tree_schema_version
            != plan_v2.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION
            or self.bundle_receipt_schema_version
            != plan_v2.LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION
        ):
            _fail("local source observation producer schema differs")
        for digest_value, label in (
            (self.snapshot_descriptor_sha256, "local snapshot descriptor"),
            (self.snapshot_manifest_file_sha256, "local snapshot manifest file"),
            (self.snapshot_manifest_body_sha256, "local snapshot manifest body"),
            (self.snapshot_tree_sha256, "local snapshot tree"),
            (self.bundle_descriptor_sha256, "local bundle descriptor"),
            (self.bundle_receipt_file_sha256, "local bundle receipt file"),
            (self.bundle_receipt_body_sha256, "local bundle receipt body"),
            (self.archive_file_sha256, "local archive file"),
            (self.member_inventory_sha256, "local member inventory"),
        ):
            _require_sha256(digest_value, label)
        _require_int(self.directory_count, "local directory count", minimum=1)
        _require_int(self.file_count, "local file count", minimum=2)
        _require_int(self.total_size_bytes, "local total bytes")
        member_count = _require_int(
            self.archive_member_count,
            "local archive member count",
            minimum=2,
        )
        _require_int(self.archive_size_bytes, "local archive bytes", minimum=1)
        if member_count != self.file_count:
            _fail("local archive member count differs from snapshot file count")
        for flag_value, label in (
            (self.producer_receipt_replay_exact, "local producer receipt replay"),
            (self.manifest_file_body_binding_exact, "local manifest file/body binding"),
            (self.source_tree_inventory_exact, "local source tree inventory"),
            (self.archive_inventory_exact, "local archive inventory"),
            (self.counts_exact, "local counts"),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "producer_kind": self.producer_kind,
            "snapshot_descriptor_sha256": self.snapshot_descriptor_sha256,
            "snapshot_manifest_schema_version": self.snapshot_manifest_schema_version,
            "snapshot_manifest_file_sha256": self.snapshot_manifest_file_sha256,
            "snapshot_manifest_body_sha256": self.snapshot_manifest_body_sha256,
            "snapshot_tree_schema_version": self.snapshot_tree_schema_version,
            "snapshot_tree_sha256": self.snapshot_tree_sha256,
            "bundle_descriptor_sha256": self.bundle_descriptor_sha256,
            "bundle_receipt_schema_version": self.bundle_receipt_schema_version,
            "bundle_receipt_file_sha256": self.bundle_receipt_file_sha256,
            "bundle_receipt_body_sha256": self.bundle_receipt_body_sha256,
            "archive_file_sha256": self.archive_file_sha256,
            "member_inventory_sha256": self.member_inventory_sha256,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "archive_member_count": self.archive_member_count,
            "archive_size_bytes": self.archive_size_bytes,
            "producer_receipt_replay_exact": self.producer_receipt_replay_exact,
            "manifest_file_body_binding_exact": self.manifest_file_body_binding_exact,
            "source_tree_inventory_exact": self.source_tree_inventory_exact,
            "archive_inventory_exact": self.archive_inventory_exact,
            "counts_exact": self.counts_exact,
        }


@dataclass(frozen=True, slots=True)
class RuntimeHelperIdentity:
    """One caller-pinned helper identity; helper order is fixed by the runtime payload."""

    helper_id: str
    descriptor_schema_version: str
    descriptor_sha256: str
    implementation_path: str
    implementation_source_sha256: str
    entrypoint: str
    entrypoint_sha256: str
    executable_sha256: str
    version_output_sha256: str

    def __post_init__(self) -> None:
        if self.helper_id not in _HELPER_IDS:
            _fail("runtime helper ID differs")
        _require_text(self.descriptor_schema_version, "runtime helper descriptor schema")
        _require_sha256(self.descriptor_sha256, "runtime helper descriptor")
        _require_relative_path(self.implementation_path, "runtime helper implementation path")
        _require_sha256(self.implementation_source_sha256, "runtime helper source")
        _require_portable_id(self.entrypoint, "runtime helper entrypoint")
        _require_sha256(self.entrypoint_sha256, "runtime helper entrypoint")
        _require_sha256(self.executable_sha256, "runtime helper executable")
        _require_sha256(self.version_output_sha256, "runtime helper version output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "helper_id": self.helper_id,
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "implementation_path": self.implementation_path,
            "implementation_source_sha256": self.implementation_source_sha256,
            "entrypoint": self.entrypoint,
            "entrypoint_sha256": self.entrypoint_sha256,
            "executable_sha256": self.executable_sha256,
            "version_output_sha256": self.version_output_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeObservationPayload:
    """Executor, image, installed runtime, sandbox, and ordered helper identities."""

    executor_kind: str
    executor_descriptor_sha256: str
    executor_source_sha256: str
    executor_receipt_schema_version: str
    executor_receipt_file_sha256: str
    executor_receipt_body_sha256: str
    runtime_executable_sha256: str
    runtime_version_output_sha256: str
    image_id: str
    image_config_sha256: str
    runtime_profile_sha256: str
    runtime_identity_sha256: str
    runtime_inventory_sha256: str
    source_import_inventory_sha256: str
    python_implementation: str
    platform: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    foragax_version: str
    foragax_install_tree_sha256: str
    jax_backend: str
    default_prng_impl: str
    jax_enable_x64: bool
    threefry_partitionable: bool
    sandbox_policy_sha256: str
    sandbox_observation_sha256: str
    helpers: tuple[RuntimeHelperIdentity, ...]
    executor_identity_exact: bool
    image_identity_exact: bool
    runtime_inventory_exact: bool
    sandbox_policy_exact: bool
    helper_order_and_identity_exact: bool
    fresh_process_observation: bool
    network_disabled_observed: bool
    cpu_only_observed: bool
    unprivileged_user_observed: bool
    read_only_source_observed: bool
    bytecode_cache_disabled_observed: bool

    def __post_init__(self) -> None:
        if self.executor_kind != "networkless_oci_cpu":
            _fail("runtime executor kind differs")
        for digest_value, label in (
            (self.executor_descriptor_sha256, "runtime executor descriptor"),
            (self.executor_source_sha256, "runtime executor source"),
            (self.executor_receipt_file_sha256, "runtime executor receipt"),
            (self.executor_receipt_body_sha256, "runtime executor receipt body"),
            (self.runtime_executable_sha256, "runtime executable"),
            (self.runtime_version_output_sha256, "runtime version output"),
            (self.image_config_sha256, "runtime image config"),
            (self.runtime_profile_sha256, "runtime profile"),
            (self.runtime_identity_sha256, "runtime identity"),
            (self.runtime_inventory_sha256, "runtime inventory"),
            (self.source_import_inventory_sha256, "runtime source-import inventory"),
            (self.foragax_install_tree_sha256, "runtime Foragax install tree"),
            (self.sandbox_policy_sha256, "runtime sandbox policy"),
            (self.sandbox_observation_sha256, "runtime sandbox observation"),
        ):
            _require_sha256(digest_value, label)
        _require_text(self.executor_receipt_schema_version, "runtime executor receipt schema")
        _require_image_id(self.image_id, "runtime image")
        if not hmac.compare_digest(
            self.image_id.removeprefix("sha256:"), self.image_config_sha256
        ):
            _fail("runtime image ID/config digest binding differs")
        if self.python_implementation != "CPython" or self.python_version != "3.12.3":
            _fail("runtime Python identity differs from exact CPython 3.12.3")
        if self.jax_version != "0.11.0" or self.jaxlib_version != "0.11.0":
            _fail("runtime JAX/JAXlib versions differ from 0.11.0")
        if self.foragax_version != "0.55.0":
            _fail("runtime Foragax version differs from 0.55.0")
        if not hmac.compare_digest(
            self.foragax_install_tree_sha256,
            _EXPECTED_FORAGAX_INSTALL_TREE_SHA256,
        ):
            _fail("runtime Foragax install tree differs")
        if self.platform != "linux/amd64" or self.jax_backend != "cpu":
            _fail("runtime platform/backend differs from linux/amd64 CPU")
        if self.default_prng_impl != "threefry2x32":
            _fail("runtime default PRNG differs from threefry2x32")
        if type(self.jax_enable_x64) is not bool or self.jax_enable_x64 is not False:
            _fail("runtime JAX x64 mode must be exact false")
        if (
            type(self.threefry_partitionable) is not bool
            or self.threefry_partitionable is not True
        ):
            _fail("runtime partitionable Threefry mode must be exact true")
        if type(self.helpers) is not tuple or len(self.helpers) != len(_HELPER_IDS):
            _fail("runtime helper bindings must be one exact ordered tuple")
        if any(type(helper) is not RuntimeHelperIdentity for helper in self.helpers):
            _fail("runtime helper binding type differs")
        if tuple(helper.helper_id for helper in self.helpers) != _HELPER_IDS:
            _fail("runtime helper order differs")
        for flag_value, label in (
            (self.executor_identity_exact, "runtime executor identity"),
            (self.image_identity_exact, "runtime image identity"),
            (self.runtime_inventory_exact, "runtime inventory"),
            (self.sandbox_policy_exact, "runtime sandbox policy"),
            (self.helper_order_and_identity_exact, "runtime helper identity"),
            (self.fresh_process_observation, "runtime fresh-process observation"),
            (self.network_disabled_observed, "runtime network-disabled observation"),
            (self.cpu_only_observed, "runtime CPU-only observation"),
            (self.unprivileged_user_observed, "runtime unprivileged-user observation"),
            (self.read_only_source_observed, "runtime read-only-source observation"),
            (self.bytecode_cache_disabled_observed, "runtime bytecode-cache observation"),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor_kind": self.executor_kind,
            "executor_descriptor_sha256": self.executor_descriptor_sha256,
            "executor_source_sha256": self.executor_source_sha256,
            "executor_receipt_schema_version": self.executor_receipt_schema_version,
            "executor_receipt_file_sha256": self.executor_receipt_file_sha256,
            "executor_receipt_body_sha256": self.executor_receipt_body_sha256,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "runtime_version_output_sha256": self.runtime_version_output_sha256,
            "image_id": self.image_id,
            "image_config_sha256": self.image_config_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "runtime_inventory_sha256": self.runtime_inventory_sha256,
            "source_import_inventory_sha256": self.source_import_inventory_sha256,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "foragax_version": self.foragax_version,
            "foragax_install_tree_sha256": self.foragax_install_tree_sha256,
            "jax_backend": self.jax_backend,
            "default_prng_impl": self.default_prng_impl,
            "jax_enable_x64": self.jax_enable_x64,
            "threefry_partitionable": self.threefry_partitionable,
            "sandbox_policy_sha256": self.sandbox_policy_sha256,
            "sandbox_observation_sha256": self.sandbox_observation_sha256,
            "helpers": [helper.to_dict() for helper in self.helpers],
            "executor_identity_exact": self.executor_identity_exact,
            "image_identity_exact": self.image_identity_exact,
            "runtime_inventory_exact": self.runtime_inventory_exact,
            "sandbox_policy_exact": self.sandbox_policy_exact,
            "helper_order_and_identity_exact": self.helper_order_and_identity_exact,
            "fresh_process_observation": self.fresh_process_observation,
            "network_disabled_observed": self.network_disabled_observed,
            "cpu_only_observed": self.cpu_only_observed,
            "unprivileged_user_observed": self.unprivileged_user_observed,
            "read_only_source_observed": self.read_only_source_observed,
            "bytecode_cache_disabled_observed": self.bytecode_cache_disabled_observed,
        }


@dataclass(frozen=True, slots=True)
class QualificationSeedObservationPayload:
    """Content-addressed public-case provenance without raw seed issuance."""

    qualification_case_id: str
    qualification_case_manifest_sha256: str
    trust_root_receipt_sha256: str
    signature_bundle_sha256: str
    derivation_descriptor_sha256: str
    seed_commitment_sha256: str
    draw_index: int
    registry_full_file_and_body_digests_exact: bool
    independent_trust_root_receipt_file_pin_exact: bool
    independent_trust_root_receipt_binding_pin_exact: bool
    provider_chain_public_key_and_signature_scheme_exact: bool
    pulse_record_exact: bool
    beacon_round_time_signature_and_randomness_exact: bool
    offline_verifier_source_closure_membership_exact: bool
    offline_signature_verification_exact: bool
    deterministic_28_case_seed_pair_derivation_exact: bool
    deterministic_registry_file_and_body_digests_exact: bool
    derivation_schema_and_domain_exact: bool
    case_derivation_payload_membership_exact: bool
    beacon_time_precedes_observation_cutoff_exact: bool
    external_receipt_preacceptance_chronology_exact: bool

    def __post_init__(self) -> None:
        _require_portable_id(self.qualification_case_id, "qualification case ID")
        for digest_value, label in (
            (self.qualification_case_manifest_sha256, "qualification case manifest"),
            (self.trust_root_receipt_sha256, "qualification trust-root receipt"),
            (self.signature_bundle_sha256, "qualification signature bundle"),
            (self.derivation_descriptor_sha256, "qualification derivation descriptor"),
            (self.seed_commitment_sha256, "qualification seed commitment"),
        ):
            _require_sha256(digest_value, label)
        _require_int(self.draw_index, "qualification draw index")
        for field_name in _SEED_ACCEPTANCE_FIELDS:
            _require_bool(getattr(self, field_name), f"qualification seed {field_name}")

    @property
    def independent_trust_root_receipt_pins_exact(self) -> bool:
        """Qualification-plan v2 summary; never a caller-controlled fact."""

        return (
            self.independent_trust_root_receipt_file_pin_exact
            and self.independent_trust_root_receipt_binding_pin_exact
        )

    @property
    def preobservation_chronology_exact(self) -> bool:
        """Qualification-plan v2 chronology summary without extra authority."""

        return (
            self.beacon_time_precedes_observation_cutoff_exact
            and self.external_receipt_preacceptance_chronology_exact
        )

    @property
    def deterministic_case_seed_derivation_exact(self) -> bool:
        """Qualification-plan v2 deterministic-derivation summary."""

        return (
            self.deterministic_28_case_seed_pair_derivation_exact
            and self.deterministic_registry_file_and_body_digests_exact
            and self.derivation_schema_and_domain_exact
            and self.case_derivation_payload_membership_exact
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_case_id": self.qualification_case_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "trust_root_receipt_sha256": self.trust_root_receipt_sha256,
            "signature_bundle_sha256": self.signature_bundle_sha256,
            "derivation_descriptor_sha256": self.derivation_descriptor_sha256,
            "seed_commitment_sha256": self.seed_commitment_sha256,
            "draw_index": self.draw_index,
            **{field_name: getattr(self, field_name) for field_name in _SEED_ACCEPTANCE_FIELDS},
            **{
                field_name: getattr(self, field_name)
                for field_name in _V2_SEED_COMPATIBILITY_FIELDS
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateObservationPayload:
    """Candidate source, configuration, import, and seed-transport structure."""

    candidate_id: str
    qualification_case_manifest_sha256: str
    source_tree_sha256: str
    configuration_record_sha256: str
    entrypoint_source_sha256: str
    agent_seed_commitment_sha256: str
    environment_seed_commitment_sha256: str
    candidate_rng_trace_sha256: str
    source_membership_exact: bool
    configuration_membership_exact: bool
    entrypoint_import_exact: bool
    agent_seed_transport_exact: bool
    environment_agent_derivations_distinct: bool
    candidate_rng_membership_exact: bool

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id)
        for digest_value, label in (
            (self.qualification_case_manifest_sha256, "candidate qualification case"),
            (self.source_tree_sha256, "candidate source tree"),
            (self.configuration_record_sha256, "candidate configuration record"),
            (self.entrypoint_source_sha256, "candidate entrypoint source"),
            (self.agent_seed_commitment_sha256, "candidate agent-seed commitment"),
            (self.environment_seed_commitment_sha256, "candidate environment commitment"),
            (self.candidate_rng_trace_sha256, "candidate RNG trace"),
        ):
            _require_sha256(digest_value, label)
        for flag_value, label in (
            (self.source_membership_exact, "candidate source membership"),
            (self.configuration_membership_exact, "candidate configuration membership"),
            (self.entrypoint_import_exact, "candidate entrypoint import"),
            (self.agent_seed_transport_exact, "candidate agent-seed transport"),
            (
                self.environment_agent_derivations_distinct,
                "candidate environment/agent derivation distinction",
            ),
            (self.candidate_rng_membership_exact, "candidate RNG membership"),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "configuration_record_sha256": self.configuration_record_sha256,
            "entrypoint_source_sha256": self.entrypoint_source_sha256,
            "agent_seed_commitment_sha256": self.agent_seed_commitment_sha256,
            "environment_seed_commitment_sha256": self.environment_seed_commitment_sha256,
            "candidate_rng_trace_sha256": self.candidate_rng_trace_sha256,
            "source_membership_exact": self.source_membership_exact,
            "configuration_membership_exact": self.configuration_membership_exact,
            "entrypoint_import_exact": self.entrypoint_import_exact,
            "agent_seed_transport_exact": self.agent_seed_transport_exact,
            "environment_agent_derivations_distinct": (
                self.environment_agent_derivations_distinct
            ),
            "candidate_rng_membership_exact": self.candidate_rng_membership_exact,
        }


def _validate_resource_values(
    values: tuple[tuple[str, int], ...],
    *,
    label: str,
) -> None:
    if (
        type(values) is not tuple
        or any(type(item) is not tuple or len(item) != 2 for item in values)
        or tuple(item[0] for item in values) != _RESOURCE_FIELDS
    ):
        _fail(f"{label} must use qualification-plan v2 resource-field order")
    for name, value in values:
        _require_int(value, f"{label} {name}")


@dataclass(frozen=True, slots=True)
class ResourceObservationPayload:
    """Predeclared ceilings and observed integer resources, without gate evaluation."""

    candidate_id: str
    qualification_case_manifest_sha256: str
    resource_requirement_body_sha256: str
    resource_observation_sha256: str
    declared_ceilings: tuple[tuple[str, int], ...]
    observed_values: tuple[tuple[str, int], ...]
    horizon_accounting_exact: bool
    reward_membership_structural_only: bool
    all_resource_observations_within_predeclared_integer_ceilings: bool

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id)
        _require_sha256(
            self.qualification_case_manifest_sha256,
            "resource qualification case",
        )
        _require_sha256(
            self.resource_requirement_body_sha256,
            "resource requirement body",
        )
        _require_sha256(self.resource_observation_sha256, "resource observation")
        _validate_resource_values(self.declared_ceilings, label="declared resource ceilings")
        _validate_resource_values(self.observed_values, label="observed resource values")
        _require_bool(self.horizon_accounting_exact, "resource horizon accounting")
        _require_bool(
            self.reward_membership_structural_only,
            "resource reward-membership structure",
        )
        _require_bool(
            self.all_resource_observations_within_predeclared_integer_ceilings,
            "resource ceiling observation",
        )

    @property
    def resource_observations_within_predeclared_integer_ceilings(self) -> bool:
        """Qualification-plan v2 exact alias; never a caller-controlled fact."""

        return self.all_resource_observations_within_predeclared_integer_ceilings

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "resource_observation_sha256": self.resource_observation_sha256,
            "declared_ceilings": dict(self.declared_ceilings),
            "observed_values": dict(self.observed_values),
            "horizon_accounting_exact": self.horizon_accounting_exact,
            "reward_membership_structural_only": self.reward_membership_structural_only,
            "all_resource_observations_within_predeclared_integer_ceilings": (
                self.all_resource_observations_within_predeclared_integer_ceilings
            ),
            _V2_RESOURCE_COMPATIBILITY_FIELD: (
                self.resource_observations_within_predeclared_integer_ceilings
            ),
        }


@dataclass(frozen=True, slots=True)
class ResultPublicationObservationPayload:
    """Content-addressed result publication structure with no decoded performance value."""

    candidate_id: str
    qualification_case_manifest_sha256: str
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    publisher_source_tree_sha256: str
    publication_manifest_sha256: str
    publication_receipt_sha256: str
    published_bundle_sha256: str
    reload_observation_sha256: str
    publication_file_count: int
    publication_total_size_bytes: int
    publisher_descriptor_membership_exact: bool
    publisher_source_closure_membership_exact: bool
    reload_validator_membership_exact: bool
    atomic_publication_exact: bool
    strict_reload_exact: bool
    full_file_digest_equivalence_exact: bool
    score_and_reward_magnitude_not_decoded: bool

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id)
        for digest_value, label in (
            (self.qualification_case_manifest_sha256, "publication qualification case"),
            (self.publisher_descriptor_sha256, "publication publisher descriptor"),
            (self.publisher_source_sha256, "publication publisher source"),
            (self.publisher_source_tree_sha256, "publication source tree"),
            (self.publication_manifest_sha256, "publication manifest"),
            (self.publication_receipt_sha256, "publication receipt"),
            (self.published_bundle_sha256, "published bundle"),
            (self.reload_observation_sha256, "publication reload observation"),
        ):
            _require_sha256(digest_value, label)
        _require_int(
            self.publication_file_count,
            "publication file count",
            minimum=1,
            maximum=_MAX_PUBLICATION_FILES,
        )
        _require_int(self.publication_total_size_bytes, "publication total bytes", minimum=1)
        for flag_value, label in (
            (self.publisher_descriptor_membership_exact, "publisher descriptor membership"),
            (
                self.publisher_source_closure_membership_exact,
                "publisher source-closure membership",
            ),
            (self.reload_validator_membership_exact, "publication reload-validator membership"),
            (self.atomic_publication_exact, "atomic publication"),
            (self.strict_reload_exact, "strict publication reload"),
            (self.full_file_digest_equivalence_exact, "publication file digest equivalence"),
            (
                self.score_and_reward_magnitude_not_decoded,
                "publication score-blind decoding",
            ),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "publisher_descriptor_sha256": self.publisher_descriptor_sha256,
            "publisher_source_sha256": self.publisher_source_sha256,
            "publisher_source_tree_sha256": self.publisher_source_tree_sha256,
            "publication_manifest_sha256": self.publication_manifest_sha256,
            "publication_receipt_sha256": self.publication_receipt_sha256,
            "published_bundle_sha256": self.published_bundle_sha256,
            "reload_observation_sha256": self.reload_observation_sha256,
            "publication_file_count": self.publication_file_count,
            "publication_total_size_bytes": self.publication_total_size_bytes,
            "publisher_descriptor_membership_exact": self.publisher_descriptor_membership_exact,
            "publisher_source_closure_membership_exact": (
                self.publisher_source_closure_membership_exact
            ),
            "reload_validator_membership_exact": self.reload_validator_membership_exact,
            "atomic_publication_exact": self.atomic_publication_exact,
            "strict_reload_exact": self.strict_reload_exact,
            "full_file_digest_equivalence_exact": self.full_file_digest_equivalence_exact,
            "score_and_reward_magnitude_not_decoded": (
                self.score_and_reward_magnitude_not_decoded
            ),
        }


@dataclass(frozen=True, slots=True)
class FreshReplayObservationPayload:
    """Seed transport, key schedule, and structural replay identities."""

    candidate_id: str
    qualification_case_manifest_sha256: str
    runtime_identity_sha256: str
    environment_seed_commitment_sha256: str
    key_schedule_descriptor_sha256: str
    first_structure_trace_sha256: str
    replay_structure_trace_sha256: str
    interaction_count: int
    environment_seed_transport_exact: bool
    reset_step_key_schedule_exact: bool
    structural_replay_exact: bool

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id)
        for digest_value, label in (
            (self.qualification_case_manifest_sha256, "replay qualification case"),
            (self.runtime_identity_sha256, "replay runtime identity"),
            (self.environment_seed_commitment_sha256, "replay environment commitment"),
            (self.key_schedule_descriptor_sha256, "replay key-schedule descriptor"),
            (self.first_structure_trace_sha256, "first structural trace"),
            (self.replay_structure_trace_sha256, "replayed structural trace"),
        ):
            _require_sha256(digest_value, label)
        _require_int(
            self.interaction_count,
            "replay interaction count",
            maximum=_MAX_INTERACTIONS,
        )
        for flag_value, label in (
            (self.environment_seed_transport_exact, "replay environment-seed transport"),
            (self.reset_step_key_schedule_exact, "replay reset/step key schedule"),
            (self.structural_replay_exact, "structural replay"),
        ):
            _require_bool(flag_value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "environment_seed_commitment_sha256": self.environment_seed_commitment_sha256,
            "key_schedule_descriptor_sha256": self.key_schedule_descriptor_sha256,
            "first_structure_trace_sha256": self.first_structure_trace_sha256,
            "replay_structure_trace_sha256": self.replay_structure_trace_sha256,
            "interaction_count": self.interaction_count,
            "environment_seed_transport_exact": self.environment_seed_transport_exact,
            "reset_step_key_schedule_exact": self.reset_step_key_schedule_exact,
            "structural_replay_exact": self.structural_replay_exact,
        }


type QualificationObservationPayload = (
    ExternalSourceObservationPayload
    | LocalSourceObservationPayload
    | RuntimeObservationPayload
    | QualificationSeedObservationPayload
    | CandidateObservationPayload
    | ResourceObservationPayload
    | ResultPublicationObservationPayload
    | FreshReplayObservationPayload
)

_PAYLOAD_TYPE_TO_KIND: Final = {
    ExternalSourceObservationPayload: SOURCE_OBSERVATION_KIND,
    LocalSourceObservationPayload: SOURCE_OBSERVATION_KIND,
    RuntimeObservationPayload: RUNTIME_OBSERVATION_KIND,
    QualificationSeedObservationPayload: QUALIFICATION_SEED_OBSERVATION_KIND,
    CandidateObservationPayload: CANDIDATE_OBSERVATION_KIND,
    ResourceObservationPayload: RESOURCE_OBSERVATION_KIND,
    ResultPublicationObservationPayload: RESULT_PUBLICATION_OBSERVATION_KIND,
    FreshReplayObservationPayload: FRESH_REPLAY_OBSERVATION_KIND,
}


def _payload_kind(payload: QualificationObservationPayload) -> str:
    kind = _PAYLOAD_TYPE_TO_KIND.get(type(payload))
    if kind is None:
        _fail("qualification observation payload type differs")
    return kind


def _payload_to_dict(payload: QualificationObservationPayload) -> dict[str, Any]:
    if type(payload) is ExternalSourceObservationPayload:
        return payload.to_dict()
    if type(payload) is LocalSourceObservationPayload:
        return payload.to_dict()
    if type(payload) is RuntimeObservationPayload:
        return payload.to_dict()
    if type(payload) is QualificationSeedObservationPayload:
        return payload.to_dict()
    if type(payload) is CandidateObservationPayload:
        return payload.to_dict()
    if type(payload) is ResourceObservationPayload:
        return payload.to_dict()
    if type(payload) is ResultPublicationObservationPayload:
        return payload.to_dict()
    if type(payload) is FreshReplayObservationPayload:
        return payload.to_dict()
    _fail("qualification observation payload type differs")


def _registry_payload_contracts() -> list[dict[str, Any]]:
    return [
        {
            "kind": SOURCE_OBSERVATION_KIND,
            "schema_version": SOURCE_OBSERVATION_SCHEMA_VERSION,
            "variants": [
                "durable_external_source_publication_v1_materialization_v2",
                "local_source_snapshot_and_retained_bundle_v1",
            ],
            "records_acceptance": False,
        },
        {
            "kind": RUNTIME_OBSERVATION_KIND,
            "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
            "ordered_helper_ids": list(_HELPER_IDS),
            "exact_runtime": {
                "executor_kind": "networkless_oci_cpu",
                "python_implementation": "CPython",
                "python_version": "3.12.3",
                "jax_version": "0.11.0",
                "jaxlib_version": "0.11.0",
                "foragax_version": "0.55.0",
                "foragax_install_tree_sha256": (
                    _EXPECTED_FORAGAX_INSTALL_TREE_SHA256
                ),
                "platform": "linux/amd64",
                "jax_backend": "cpu",
                "default_prng_impl": "threefry2x32",
                "jax_enable_x64": False,
                "threefry_partitionable": True,
            },
            "records_acceptance": False,
        },
        {
            "kind": QUALIFICATION_SEED_OBSERVATION_KIND,
            "schema_version": QUALIFICATION_SEED_OBSERVATION_SCHEMA_VERSION,
            "acceptance_fields": list(_SEED_ACCEPTANCE_FIELDS),
            "qualification_plan_v2_derived_summary_fields": list(
                _V2_SEED_COMPATIBILITY_FIELDS
            ),
            "raw_seed_material_allowed": False,
            "records_acceptance": False,
        },
        {
            "kind": CANDIDATE_OBSERVATION_KIND,
            "schema_version": CANDIDATE_OBSERVATION_SCHEMA_VERSION,
            "acceptance_fields": list(_CANDIDATE_ACCEPTANCE_FIELDS),
            "candidate_order": list(_CANDIDATE_IDS),
            "records_acceptance": False,
        },
        {
            "kind": RESOURCE_OBSERVATION_KIND,
            "schema_version": RESOURCE_OBSERVATION_SCHEMA_VERSION,
            "acceptance_fields": list(_RESOURCE_ACCEPTANCE_FIELDS),
            "qualification_plan_v2_derived_summary_fields": [
                _V2_RESOURCE_COMPATIBILITY_FIELD
            ],
            "resource_fields": list(_RESOURCE_FIELDS),
            "ceiling_evaluation_performed": False,
            "records_acceptance": False,
        },
        {
            "kind": RESULT_PUBLICATION_OBSERVATION_KIND,
            "schema_version": RESULT_PUBLICATION_OBSERVATION_SCHEMA_VERSION,
            "acceptance_fields": list(_PUBLICATION_ACCEPTANCE_FIELDS),
            "performance_values_allowed": False,
            "records_acceptance": False,
        },
        {
            "kind": FRESH_REPLAY_OBSERVATION_KIND,
            "schema_version": FRESH_REPLAY_OBSERVATION_SCHEMA_VERSION,
            "acceptance_fields": list(_FRESH_REPLAY_ACCEPTANCE_FIELDS),
            "structural_content_only": True,
            "records_acceptance": False,
        },
    ]


def _v2_compatibility() -> dict[str, Any]:
    return {
        "qualification_plan_schema_version": plan_v2.QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
        "mode": "strict_additive_derived_summary_refinement",
        "summary_fields_are_caller_controlled": False,
        "acceptance_evaluation_performed_here": False,
        "authority_granted": False,
        "seed_summary_formulas": {
            "independent_trust_root_receipt_pins_exact": {
                "operator": "all",
                "inputs": [
                    "independent_trust_root_receipt_file_pin_exact",
                    "independent_trust_root_receipt_binding_pin_exact",
                ],
            },
            "preobservation_chronology_exact": {
                "operator": "all",
                "inputs": [
                    "beacon_time_precedes_observation_cutoff_exact",
                    "external_receipt_preacceptance_chronology_exact",
                ],
            },
            "deterministic_case_seed_derivation_exact": {
                "operator": "all",
                "inputs": [
                    "deterministic_28_case_seed_pair_derivation_exact",
                    "deterministic_registry_file_and_body_digests_exact",
                    "derivation_schema_and_domain_exact",
                    "case_derivation_payload_membership_exact",
                ],
            },
        },
        "resource_summary_formulas": {
            _V2_RESOURCE_COMPATIBILITY_FIELD: {
                "operator": "identity",
                "input": (
                    "all_resource_observations_within_predeclared_integer_ceilings"
                ),
            }
        },
        "publication_profile": {
            "v2_fields_retained": True,
            "reload_validator_membership_exact_additionally_required": True,
        },
    }


def _registry_descriptor() -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_OBSERVATION_REGISTRY_SCHEMA_VERSION,
        "status": QUALIFICATION_OBSERVATION_STATUS,
        "classification": QUALIFICATION_OBSERVATION_CLASSIFICATION,
        "envelope": {
            "schema_version": QUALIFICATION_OBSERVATION_ENVELOPE_SCHEMA_VERSION,
            "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
            "full_file_sha256_caller_pin_required": True,
            "qualification_plan_sha256_caller_pin_required": True,
            "registry_descriptor_sha256_embedded": True,
            "one_payload_kind_only": True,
        },
        "qualification_plan_binding": {
            "schema_version": plan_v2.QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
            "candidate_order": list(_CANDIDATE_IDS),
            "resource_fields": list(_RESOURCE_FIELDS),
            "default_plan_embedded": False,
            "plan_loaded_or_validated_here": False,
        },
        "v2_compatibility": _v2_compatibility(),
        "payload_contracts": _registry_payload_contracts(),
        "strict_json": {
            "duplicate_keys_allowed": False,
            "floats_allowed": False,
            "noncanonical_bytes_allowed": False,
            "container_aliases_or_cycles_allowed": False,
            "unknown_keys_allowed": False,
            "maximum_depth": _MAX_JSON_DEPTH,
            "maximum_nodes": _MAX_JSON_NODES,
            "maximum_text_length": _MAX_TEXT_LENGTH,
            "maximum_artifact_bytes": _MAX_ARTIFACT_BYTES,
        },
        "prohibited_payload_fields": sorted(_FORBIDDEN_PAYLOAD_KEYS),
        "capabilities": {
            "acceptance_evaluator": False,
            "clock": False,
            "default_inputs": False,
            "executor": False,
            "filesystem": False,
            "network": False,
            "observation_issuer": False,
            "qualification_bundle_builder": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_REGISTRY_DESCRIPTOR_BYTES: Final = _canonical_json(_registry_descriptor())
QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256: Final = (
    "f28d01ae9750ee5989f613dbdc64b91f8a8a500faa460b9b5a8c89aa59b31c09"
)
if not hmac.compare_digest(
    hashlib.sha256(_REGISTRY_DESCRIPTOR_BYTES).hexdigest(),
    QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256,
):
    raise RuntimeError("qualification observation registry descriptor identity drifted")


@dataclass(frozen=True, slots=True)
class MatchedV3QualificationObservation:
    """Immutable parsed or caller-built structural observation."""

    qualification_plan_sha256: str
    payload: QualificationObservationPayload

    def __post_init__(self) -> None:
        _require_sha256(self.qualification_plan_sha256, "qualification observation plan")
        _payload_kind(self.payload)
        payload = _payload_to_dict(self.payload)
        _assert_plain_unaliased_json(payload)
        _reject_forbidden_payload_keys(payload)

    @property
    def observation_kind(self) -> str:
        return _payload_kind(self.payload)

    @property
    def observation_schema_version(self) -> str:
        return _SCHEMA_BY_KIND[self.observation_kind]

    def to_dict(self) -> dict[str, Any]:
        payload = _payload_to_dict(self.payload)
        body: dict[str, Any] = {
            "schema_version": QUALIFICATION_OBSERVATION_ENVELOPE_SCHEMA_VERSION,
            "status": QUALIFICATION_OBSERVATION_STATUS,
            "classification": QUALIFICATION_OBSERVATION_CLASSIFICATION,
            "registry_descriptor_sha256": (
                QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256
            ),
            "qualification_plan": {
                "schema_version": plan_v2.QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
                "sha256": self.qualification_plan_sha256,
            },
            "observation_kind": self.observation_kind,
            "observation_schema_version": self.observation_schema_version,
            "payload": payload,
            "claims": _claims(),
            "limitations": _limitations(),
        }
        return {
            **body,
            "envelope_body_sha256": hashlib.sha256(
                _canonical_json(body, newline=False)
            ).hexdigest(),
        }


def _external_payload_from_dict(value: object) -> ExternalSourceObservationPayload:
    keys = frozenset(ExternalSourceObservationPayload.__dataclass_fields__)
    return ExternalSourceObservationPayload(
        **_require_exact_keys(value, keys, "external source observation payload")
    )


def _local_payload_from_dict(value: object) -> LocalSourceObservationPayload:
    keys = frozenset(LocalSourceObservationPayload.__dataclass_fields__)
    return LocalSourceObservationPayload(
        **_require_exact_keys(value, keys, "local source observation payload")
    )


def _helper_from_dict(value: object) -> RuntimeHelperIdentity:
    keys = frozenset(RuntimeHelperIdentity.__dataclass_fields__)
    return RuntimeHelperIdentity(**_require_exact_keys(value, keys, "runtime helper identity"))


def _runtime_payload_from_dict(value: object) -> RuntimeObservationPayload:
    keys = frozenset(RuntimeObservationPayload.__dataclass_fields__)
    item = dict(_require_exact_keys(value, keys, "runtime observation payload"))
    raw_helpers = item.pop("helpers")
    if type(raw_helpers) is not list:
        _fail("runtime observation helpers must be one exact list")
    helpers = tuple(_helper_from_dict(helper) for helper in raw_helpers)
    return RuntimeObservationPayload(**item, helpers=helpers)


def _seed_payload_from_dict(value: object) -> QualificationSeedObservationPayload:
    dataclass_keys = frozenset(QualificationSeedObservationPayload.__dataclass_fields__)
    keys = dataclass_keys | frozenset(_V2_SEED_COMPATIBILITY_FIELDS)
    item = dict(_require_exact_keys(value, keys, "qualification-seed observation payload"))
    supplied_compatibility = {
        field_name: item.pop(field_name) for field_name in _V2_SEED_COMPATIBILITY_FIELDS
    }
    result = QualificationSeedObservationPayload(**item)
    for field_name, supplied in supplied_compatibility.items():
        _require_bool(supplied, f"qualification-plan v2 seed summary {field_name}")
        if supplied is not getattr(result, field_name):
            _fail(f"qualification-plan v2 seed summary {field_name} differs")
    return result


def _candidate_payload_from_dict(value: object) -> CandidateObservationPayload:
    keys = frozenset(CandidateObservationPayload.__dataclass_fields__)
    return CandidateObservationPayload(
        **_require_exact_keys(value, keys, "candidate observation payload")
    )


def _resource_mapping(value: object, label: str) -> tuple[tuple[str, int], ...]:
    item = _require_exact_keys(value, frozenset(_RESOURCE_FIELDS), label)
    return tuple((name, item[name]) for name in _RESOURCE_FIELDS)


def _resource_payload_from_dict(value: object) -> ResourceObservationPayload:
    dataclass_keys = frozenset(ResourceObservationPayload.__dataclass_fields__)
    keys = dataclass_keys | frozenset({_V2_RESOURCE_COMPATIBILITY_FIELD})
    item = dict(_require_exact_keys(value, keys, "resource observation payload"))
    supplied_v2_summary = item.pop(_V2_RESOURCE_COMPATIBILITY_FIELD)
    ceilings = _resource_mapping(item.pop("declared_ceilings"), "declared resource ceilings")
    observed = _resource_mapping(item.pop("observed_values"), "observed resource values")
    result = ResourceObservationPayload(
        **item,
        declared_ceilings=ceilings,
        observed_values=observed,
    )
    _require_bool(supplied_v2_summary, "qualification-plan v2 resource summary")
    if supplied_v2_summary is not (
        result.resource_observations_within_predeclared_integer_ceilings
    ):
        _fail("qualification-plan v2 resource summary differs")
    return result


def _publication_payload_from_dict(value: object) -> ResultPublicationObservationPayload:
    keys = frozenset(ResultPublicationObservationPayload.__dataclass_fields__)
    return ResultPublicationObservationPayload(
        **_require_exact_keys(value, keys, "result-publication observation payload")
    )


def _replay_payload_from_dict(value: object) -> FreshReplayObservationPayload:
    keys = frozenset(FreshReplayObservationPayload.__dataclass_fields__)
    return FreshReplayObservationPayload(
        **_require_exact_keys(value, keys, "fresh-replay observation payload")
    )


def _payload_from_dict(kind: str, value: object) -> QualificationObservationPayload:
    if type(value) is not dict:
        _fail("qualification observation payload must be one exact object")
    payload = cast(dict[str, Any], value)
    _reject_forbidden_payload_keys(payload)
    if kind == SOURCE_OBSERVATION_KIND:
        producer_kind = payload.get("producer_kind")
        if producer_kind == "durable_external_source_publication_v1_materialization_v2":
            return _external_payload_from_dict(payload)
        if producer_kind == "local_source_snapshot_and_retained_bundle_v1":
            return _local_payload_from_dict(payload)
        _fail("source observation producer kind differs")
    if kind == RUNTIME_OBSERVATION_KIND:
        return _runtime_payload_from_dict(payload)
    if kind == QUALIFICATION_SEED_OBSERVATION_KIND:
        return _seed_payload_from_dict(payload)
    if kind == CANDIDATE_OBSERVATION_KIND:
        return _candidate_payload_from_dict(payload)
    if kind == RESOURCE_OBSERVATION_KIND:
        return _resource_payload_from_dict(payload)
    if kind == RESULT_PUBLICATION_OBSERVATION_KIND:
        return _publication_payload_from_dict(payload)
    if kind == FRESH_REPLAY_OBSERVATION_KIND:
        return _replay_payload_from_dict(payload)
    _fail("qualification observation kind differs")


def _parse_envelope_value(
    value: Mapping[str, Any],
    *,
    expected_qualification_plan_sha256: str,
) -> MatchedV3QualificationObservation:
    envelope = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "registry_descriptor_sha256",
                "qualification_plan",
                "observation_kind",
                "observation_schema_version",
                "payload",
                "claims",
                "limitations",
                "envelope_body_sha256",
            }
        ),
        "qualification observation envelope",
    )
    if (
        envelope["schema_version"] != QUALIFICATION_OBSERVATION_ENVELOPE_SCHEMA_VERSION
        or envelope["status"] != QUALIFICATION_OBSERVATION_STATUS
        or envelope["classification"] != QUALIFICATION_OBSERVATION_CLASSIFICATION
        or envelope["registry_descriptor_sha256"]
        != QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256
    ):
        _fail("qualification observation envelope identity differs")
    plan = _require_exact_keys(
        envelope["qualification_plan"],
        frozenset({"schema_version", "sha256"}),
        "qualification observation plan binding",
    )
    expected_plan = _require_sha256(
        expected_qualification_plan_sha256,
        "expected qualification-plan file",
    )
    if (
        plan["schema_version"] != plan_v2.QUALIFICATION_PLAN_V2_SCHEMA_VERSION
        or type(plan["sha256"]) is not str
        or not hmac.compare_digest(plan["sha256"], expected_plan)
    ):
        _fail("qualification observation plan caller pin differs")
    kind = envelope["observation_kind"]
    if type(kind) is not str or kind not in _SCHEMA_BY_KIND:
        _fail("qualification observation kind differs")
    if envelope["observation_schema_version"] != _SCHEMA_BY_KIND[kind]:
        _fail("qualification observation kind/schema binding differs")
    claims = _require_exact_keys(
        envelope["claims"],
        frozenset(_claims()),
        "qualification observation claims",
    )
    if not _exact_json_equal(claims, _claims()) or any(
        value is not False for value in claims.values()
    ):
        _fail("qualification observation authority claim became true")
    if not _exact_json_equal(envelope["limitations"], _limitations()):
        _fail("qualification observation limitations differ")
    supplied_body = _require_sha256(
        envelope["envelope_body_sha256"],
        "qualification observation envelope body",
    )
    body = copy.deepcopy(envelope)
    body.pop("envelope_body_sha256")
    expected_body = hashlib.sha256(_canonical_json(body, newline=False)).hexdigest()
    if not hmac.compare_digest(supplied_body, expected_body):
        _fail("qualification observation envelope body digest differs")
    payload = _payload_from_dict(kind, envelope["payload"])
    result = MatchedV3QualificationObservation(
        qualification_plan_sha256=expected_plan,
        payload=payload,
    )
    if not _exact_json_equal(result.to_dict(), envelope):
        _fail("qualification observation canonical reconstruction differs")
    return result


def canonical_matched_v3_qualification_observation_bytes(
    observation: MatchedV3QualificationObservation,
) -> bytes:
    """Validate and encode one immutable structural observation."""

    if type(observation) is not MatchedV3QualificationObservation:
        _fail("canonical observation serialization requires the exact observation type")
    value = observation.to_dict()
    _parse_envelope_value(
        value,
        expected_qualification_plan_sha256=observation.qualification_plan_sha256,
    )
    return _canonical_json(value)


def parse_matched_v3_qualification_observation(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_qualification_plan_sha256: str,
) -> MatchedV3QualificationObservation:
    """Parse canonical bytes under independent full-file and plan SHA-256 pins."""

    expected_file = _require_sha256(expected_file_sha256, "expected observation file")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_file
    ):
        _fail("qualification observation full-file SHA-256 differs")
    value = _strict_json_load(raw)
    return _parse_envelope_value(
        value,
        expected_qualification_plan_sha256=expected_qualification_plan_sha256,
    )


def replay_matched_v3_qualification_observation(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_qualification_plan_sha256: str,
) -> MatchedV3QualificationObservation:
    """Replay alias preserving both independent caller pins."""

    return parse_matched_v3_qualification_observation(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_qualification_plan_sha256=expected_qualification_plan_sha256,
    )


def matched_v3_qualification_observation_registry_descriptor() -> dict[str, Any]:
    """Return detached structural registry content; this grants no authority."""

    return _strict_json_load(_REGISTRY_DESCRIPTOR_BYTES)


def canonical_matched_v3_qualification_observation_registry_descriptor_bytes() -> bytes:
    """Return the exact canonical registry descriptor bytes."""

    return _REGISTRY_DESCRIPTOR_BYTES


def matched_v3_qualification_observation_registry_descriptor_sha256() -> str:
    """Return the exact registry descriptor SHA-256."""

    return QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256


def parse_matched_v3_qualification_observation_registry_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Parse only the exact frozen registry descriptor."""

    value = _strict_json_load(raw)
    if (
        not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256,
        )
        or not _exact_json_equal(value, _registry_descriptor())
    ):
        _fail("qualification observation registry descriptor differs")
    return value


__all__ = [
    "CANDIDATE_OBSERVATION_KIND",
    "CANDIDATE_OBSERVATION_SCHEMA_VERSION",
    "CandidateObservationPayload",
    "ExternalSourceObservationPayload",
    "FRESH_REPLAY_OBSERVATION_KIND",
    "FRESH_REPLAY_OBSERVATION_SCHEMA_VERSION",
    "ForagerMatchedV3QualificationObservationError",
    "FreshReplayObservationPayload",
    "LocalSourceObservationPayload",
    "MatchedV3QualificationObservation",
    "QUALIFICATION_OBSERVATION_CLASSIFICATION",
    "QUALIFICATION_OBSERVATION_ENVELOPE_SCHEMA_VERSION",
    "QUALIFICATION_OBSERVATION_KINDS",
    "QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256",
    "QUALIFICATION_OBSERVATION_REGISTRY_SCHEMA_VERSION",
    "QUALIFICATION_OBSERVATION_SCHEMA_VERSIONS",
    "QUALIFICATION_OBSERVATION_STATUS",
    "QUALIFICATION_SEED_OBSERVATION_KIND",
    "QUALIFICATION_SEED_OBSERVATION_SCHEMA_VERSION",
    "QualificationObservationPayload",
    "QualificationSeedObservationPayload",
    "RESOURCE_OBSERVATION_KIND",
    "RESOURCE_OBSERVATION_SCHEMA_VERSION",
    "RESULT_PUBLICATION_OBSERVATION_KIND",
    "RESULT_PUBLICATION_OBSERVATION_SCHEMA_VERSION",
    "RUNTIME_OBSERVATION_KIND",
    "RUNTIME_OBSERVATION_SCHEMA_VERSION",
    "ResourceObservationPayload",
    "ResultPublicationObservationPayload",
    "RuntimeHelperIdentity",
    "RuntimeObservationPayload",
    "SOURCE_OBSERVATION_KIND",
    "SOURCE_OBSERVATION_SCHEMA_VERSION",
    "canonical_matched_v3_qualification_observation_bytes",
    "canonical_matched_v3_qualification_observation_registry_descriptor_bytes",
    "matched_v3_qualification_observation_registry_descriptor",
    "matched_v3_qualification_observation_registry_descriptor_sha256",
    "parse_matched_v3_qualification_observation",
    "parse_matched_v3_qualification_observation_registry_descriptor",
    "replay_matched_v3_qualification_observation",
]
