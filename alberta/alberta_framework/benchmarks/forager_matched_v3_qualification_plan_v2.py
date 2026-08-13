"""Descriptor-only, nonauthorizing Forager matched-v3 qualification plan v2.

The v1 qualification-plan contract predates the durable external-source v2
materialization and the local source-bundle producer.  This additive contract
binds those producers explicitly and carries the context, execution,
publication, and image identities of one already-published CPU OCI build as
four independent caller pins.

This module does not issue qualification cases, execute probes, inspect result
payloads, or grant any authority.  Only the two adapter candidates currently
have an implemented strict result publisher.  The other 26 publisher slots
remain explicit gaps, so every artifact built here has status
``contract_implemented_no_production_plan``.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

QUALIFICATION_PLAN_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_descriptor.v2"
)
QUALIFICATION_PLAN_V2_STATUS: Final = "contract_implemented_no_production_plan"
QUALIFICATION_PLAN_V2_CLASSIFICATION: Final = (
    "content_only_unexecuted_incomplete_publisher_registry_non_authorizing"
)

EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_source_publication_receipt.v1"
)
EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization.v2"
)
EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_manifest.v1"
)
LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_bundle_receipt.v1"
)
LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_manifest.v1"
)
LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_tree.v1"
)
CPU_OCI_BUILD_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.cpu_oci_build_plan.v1"
CPU_OCI_BUILD_INTENT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.cpu_oci_build_intent.v1"
CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_context_receipt.v1"
)
CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_execution_receipt.v1"
)
CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_publication.v1"
)

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 4_096
_MAX_INTEGER: Final = 2**63 - 1
_MAX_SOURCE_ENTRIES: Final = 2_000_000
_HORIZON: Final = 499_712
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")

_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
_CANDIDATE_UNIVERSE_SHA256: Final = (
    "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
)
_CUMULATIVE_REWARD_METRIC_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
_TRIAL_BLOCK_GENERATOR_PLAN_SHA256: Final = (
    "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
)
_EXTERNAL_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "926b56b62f992bc12fc4abe2455992ecfc89fa48df92a232aa556b4bf517f04a"
)
_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256: Final = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
_EXTERNAL_MATERIALIZER_SOURCE_SHA256: Final = (
    "3ff59a9f88d79b122fa66a1cdca009a68ff524806a7a7c58e5d565cd30ecaafe"
)
_LOCAL_SNAPSHOT_DESCRIPTOR_SHA256: Final = (
    "5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"
)
_LOCAL_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "52a48f3258aff9c7f2e80033187b85dd2924dc843d991ba7c2bac829f10c5e89"
)
_ADAPTER_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
_ADAPTER_PUBLICATION_SOURCE_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)
_ADAPTER_PUBLICATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
)

_LOCAL_CANDIDATE_IDS: Final = (
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
_EXTERNAL_CANDIDATE_IDS: Final = (
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
IMPLEMENTED_PUBLISHER_CANDIDATE_IDS: Final = ("adapted_full_rainbow", "adapted_ppo_gru")
MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS: Final = (
    _LOCAL_CANDIDATE_IDS
    + _EXTERNAL_CANDIDATE_IDS[:9]
    + IMPLEMENTED_PUBLISHER_CANDIDATE_IDS
    + _EXTERNAL_CANDIDATE_IDS[9:]
)
MISSING_PUBLISHER_CANDIDATE_IDS: Final = tuple(
    candidate_id
    for candidate_id in MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    if candidate_id not in IMPLEMENTED_PUBLISHER_CANDIDATE_IDS
)

_CONFIGURATION_RECORD_SHA256: Final = {
    "causal_e025_q050": "d780067bf7fc6582b7c30a4f7bcb14672ceb15d201fc76e0c7d6e233d0f0660c",
    "causal_e025_q075": "8c4488a4ca6a513731c5671cacdc55397aa7faaa07052ddb86000a53787aae8a",
    "causal_e025_q090": "130d243b230e8a9427f2f60b317eae463993f722ca45a275cb2d8398cff24afa",
    "causal_e050_q050": "373fb27a1566c280047b619c1c18f7065d4e11038c1b71939e1afa7d99ca1dda",
    "causal_e050_q075": "4e2ed83e2f40d6440e9b21f74c69f277abf23bd14e9383b5253e985dfcba731f",
    "causal_e050_q090": "deda929f9d606d08ed9c85c461eaeb8a7bc13c44e536d83eb861c94cbe2417dd",
    "causal_e100_q050": "7b6b85ec68afa398077170ac7fd90bb0256e4b687f4e2156dfc9e89f554aefca",
    "causal_e100_q075": "4b2b287c40d9a97d903e8150add4fd0190557befe9ab8a95cd2daa2c2d289afb",
    "causal_e100_q090": "9d4311599ba6eb46ad8098df0e57ca1fe2c1878cb0c62ea32830a4e3321652ff",
    "alberta_horde_default": "7dbd4f63c60484ffaadbd587c502de6d1079713cc4f044e54deadb6557b6a382",
    "alberta_horde_eps05": "73d818ae3ffaaedf5bbb40df5ff83d703fcd60c7a07a192436f3ba078d27e4b4",
    "alberta_horde_recurrent64": (
        "ac3fe6295280202a8a316d0a07e136cda3db806f9031163e3f767ae0bc0f30ea"
    ),
    "alberta_horde_step3e3": "cd11bac7f31e9a1a32c4a6c8c4706eea258bb040fdd21d220ed785a23a7ff014",
    "alberta_rtu_h08_taylor": "d804c8b79f29da16f085c7f1b4621ae479d780c3b23d30799367982353eb69df",
    "external_dqn_plain": "36af195ff30176f0b1d826fad4f2a7e3a820a5e378536c6635be9bb124caab5a",
    "external_dqn_crelu": "bed3acff23f2684a37e7d57f68f73c53f90c5037b0e5743bd6e39c0c2c420362",
    "external_dqn_redo": "6c65f8f7bfbcb92ef5b2bfd145ab0aa611bbae258b125032f7269af4eb0fd390",
    "external_dqn_reward_trace": (
        "9b78fdadd71f56c593eec14e8bd6d54e1e819053d06ee7d4f12c604d1eafd666"
    ),
    "external_dqn_l2_init": "a7b75eaee1ad9d62ff052373f8ba35b6ad5515af06e30baaa527a5c3cdcb5999",
    "external_pt_dqn_xfinal": "79afe7a23bb01cbe7f4130bfe831e9c7c65be5f4bd833cfee9d307b942df6d8a",
    "external_drqn_xfinal": "37bb4c7c17d9933a6ac54b064b81139a572ac43bc09163d62d14beb1ac6db387",
    "isolated_ppo_generic": "d88c5e7359085d362a235365e994117e46543e26e074491d2379e75efb499c47",
    "isolated_rtu_paper_scale": (
        "c01bfeb5af6af6c79b444214d09064fb23601d7a03d669ee9662d3181d139210"
    ),
    "adapted_full_rainbow": "4863d7d569def90d20f89a1aafa2e1984df93be1368070dffe655c7b2699d0b9",
    "adapted_ppo_gru": "4f8b429ff968213d0c05de87553456be7f2c1a67a806944357543025d725d7ca",
    "random_policy": "3646c050470e6ddbd817bae5512096c1225561367f486f1c2a5964e0848b2515",
    "search_nearest": "caf65fa4215b1c0d6a08b8ebbf6ffb481034eeb9f81d7ab0385d5181d45b685d",
    "search_oracle": "9214959547664a9d3d37e32ca472abcadb4b8e14d2e3d739b75b2d3721dbd5a8",
}

RESOURCE_CEILING_FIELDS: Final = (
    "max_environment_interactions",
    "max_optimizer_updates",
    "max_gradient_updates",
    "max_sample_updates",
    "max_trainable_parameters",
    "max_frozen_parameters",
    "max_optimizer_state_elements",
    "max_optimizer_state_bytes",
    "max_target_copy_elements",
    "max_target_copy_bytes",
    "max_replay_capacity_transitions",
    "max_replay_peak_bytes",
    "max_rollout_storage_elements",
    "max_rollout_peak_bytes",
    "max_recurrent_carry_elements",
    "max_recurrent_carry_bytes",
    "max_rtrl_sensitivity_elements",
    "max_rtrl_sensitivity_bytes",
    "max_eligibility_elements",
    "max_eligibility_bytes",
    "max_peak_rss_bytes",
    "max_cpu_time_ns",
    "max_wall_time_ns",
    "max_temporary_peak_bytes",
    "max_disk_peak_bytes",
    "max_thread_count",
    "max_attempt_count",
    "max_failure_count",
)

_OBSERVATION_SCHEMAS: Final = {
    "source_observation": "alberta.forager_matched_v3.source_observation.v1",
    "runtime_observation": "alberta.forager_matched_v3.runtime_observation.v1",
    "qualification_seed_observation": (
        "alberta.forager_matched_v3.qualification_seed_observation.v1"
    ),
    "candidate_observation": "alberta.forager_matched_v3.candidate_observation.v1",
    "resource_observation": "alberta.forager_matched_v3.resource_observation.v1",
    "publication_observation": (
        "alberta.forager_matched_v3.result_publication_observation.v1"
    ),
    "fresh_replay_observation": "alberta.forager_matched_v3.fresh_replay_observation.v1",
    "qualification_bundle": "alberta.forager_matched_v3.qualification_bundle.v1",
}


class ForagerMatchedV3QualificationPlanV2Error(ValueError):
    """A v2 binding or canonical descriptor failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationPlanV2Error(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_image_id(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one sha256: image ID")
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


def _require_relative_path(value: object, label: str) -> str:
    if type(value) is not str or _RELATIVE_PATH_RE.fullmatch(value) is None:
        _fail(f"{label} must be one portable relative path")
    return value


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"qualification-plan JSON contains non-finite constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"qualification-plan JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("qualification-plan JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"qualification-plan JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("qualification-plan JSON exceeds its node limit")
        if depth > _MAX_JSON_DEPTH:
            _fail("qualification-plan JSON exceeds its depth limit")
        if item is None or type(item) in {bool, int}:
            if type(item) is int:
                _require_int(item, "qualification-plan JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH:
                _fail("qualification-plan JSON string exceeds its length limit")
            return
        if type(item) not in {dict, list}:
            _fail("qualification-plan JSON contains a non-JSON or inexact value")
        identity = id(item)
        if identity in seen:
            _fail("qualification-plan JSON containers must be unaliased")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        for key, child in cast(dict[object, object], item).items():
            if type(key) is not str or len(key) > _MAX_TEXT_LENGTH:
                _fail("qualification-plan JSON object key is invalid")
            visit(child, depth + 1)

    visit(value, 0)


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[str, object], left)
        right_mapping = cast(dict[str, object], right)
        return set(left_mapping) == set(right_mapping) and all(
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
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ForagerMatchedV3QualificationPlanV2Error(
            "qualification-plan JSON cannot be canonically encoded"
        ) from exc
    if newline:
        raw += b"\n"
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("qualification-plan JSON exceeds its byte limit")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("qualification-plan bytes are invalid or exceed the byte limit")
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationPlanV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationPlanV2Error(
            "qualification-plan bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("qualification-plan root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if raw != _canonical_json(result):
        _fail("qualification-plan bytes are not exact canonical JSON")
    return result


@dataclass(frozen=True, slots=True)
class ExternalSourcePublicationBindingV2:
    """Receipt-native binding for the durable external source producer."""

    publication_receipt_schema_version: str
    publication_receipt_sha256: str
    publication_receipt_body_sha256: str
    publication_contract_descriptor_sha256: str
    materialization_manifest_schema_version: str
    materialization_manifest_sha256: str
    materialization_payload_sha256: str
    materialization_identity_sha256: str
    source_tree_sha256: str
    staging_manifest_schema_version: str
    staging_manifest_sha256: str
    staging_manifest_body_sha256: str
    archive_sha256: str
    archive_size_bytes: int
    archive_member_count: int
    archive_inventory_sha256: str

    def __post_init__(self) -> None:
        if (
            self.publication_receipt_schema_version
            != EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
            or self.publication_contract_descriptor_sha256
            != _EXTERNAL_PUBLICATION_DESCRIPTOR_SHA256
            or self.materialization_manifest_schema_version
            != EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
            or self.materialization_identity_sha256
            != _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
            or self.staging_manifest_schema_version != EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION
        ):
            _fail("external source producer schema or frozen identity differs")
        for value, label in (
            (self.publication_receipt_sha256, "external publication receipt"),
            (self.publication_receipt_body_sha256, "external publication receipt body"),
            (self.publication_contract_descriptor_sha256, "external publication descriptor"),
            (self.materialization_manifest_sha256, "external materialization manifest"),
            (self.materialization_payload_sha256, "external materialization payload"),
            (self.materialization_identity_sha256, "external materialization identity"),
            (self.source_tree_sha256, "external source tree"),
            (self.staging_manifest_sha256, "external staging manifest"),
            (self.staging_manifest_body_sha256, "external staging manifest body"),
            (self.archive_sha256, "external source archive"),
            (self.archive_inventory_sha256, "external archive inventory"),
        ):
            _require_sha256(value, label)
        _require_int(self.archive_size_bytes, "external archive size", minimum=1)
        _require_int(
            self.archive_member_count,
            "external archive member count",
            minimum=1,
            maximum=_MAX_SOURCE_ENTRIES,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_kind": "durable_external_source_publication_v1_materialization_v2",
            "publication_receipt_schema_version": self.publication_receipt_schema_version,
            "publication_receipt_sha256": self.publication_receipt_sha256,
            "publication_receipt_body_sha256": self.publication_receipt_body_sha256,
            "publication_contract_descriptor_sha256": (
                self.publication_contract_descriptor_sha256
            ),
            "materialization_manifest_schema_version": (
                self.materialization_manifest_schema_version
            ),
            "materialization_manifest_sha256": self.materialization_manifest_sha256,
            "materialization_payload_sha256": self.materialization_payload_sha256,
            "materialization_identity_sha256": self.materialization_identity_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "staging_manifest_schema_version": self.staging_manifest_schema_version,
            "staging_manifest_sha256": self.staging_manifest_sha256,
            "staging_manifest_body_sha256": self.staging_manifest_body_sha256,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "archive_member_count": self.archive_member_count,
            "archive_inventory_sha256": self.archive_inventory_sha256,
        }


@dataclass(frozen=True, slots=True)
class LocalSourceBundleBindingV2:
    """Receipt-native binding for the local snapshot and retained USTAR."""

    bundle_receipt_schema_version: str
    bundle_receipt_sha256: str
    bundle_receipt_body_sha256: str
    bundle_descriptor_sha256: str
    snapshot_manifest_schema_version: str
    snapshot_manifest_sha256: str
    snapshot_tree_schema_version: str
    snapshot_tree_sha256: str
    archive_sha256: str
    archive_size_bytes: int
    archive_member_count: int
    member_inventory_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int

    def __post_init__(self) -> None:
        if (
            self.bundle_receipt_schema_version != LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION
            or self.bundle_descriptor_sha256 != _LOCAL_BUNDLE_DESCRIPTOR_SHA256
            or self.snapshot_manifest_schema_version
            != LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
            or self.snapshot_tree_schema_version != LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION
        ):
            _fail("local source producer schema or frozen descriptor differs")
        for value, label in (
            (self.bundle_receipt_sha256, "local bundle receipt"),
            (self.bundle_receipt_body_sha256, "local bundle receipt body"),
            (self.bundle_descriptor_sha256, "local bundle descriptor"),
            (self.snapshot_manifest_sha256, "local snapshot manifest"),
            (self.snapshot_tree_sha256, "local snapshot tree"),
            (self.archive_sha256, "local source archive"),
            (self.member_inventory_sha256, "local member inventory"),
        ):
            _require_sha256(value, label)
        _require_int(self.archive_size_bytes, "local archive size", minimum=1)
        member_count = _require_int(
            self.archive_member_count,
            "local archive member count",
            minimum=2,
            maximum=_MAX_SOURCE_ENTRIES,
        )
        file_count = _require_int(
            self.file_count, "local snapshot file count", minimum=2, maximum=_MAX_SOURCE_ENTRIES
        )
        _require_int(
            self.directory_count,
            "local snapshot directory count",
            minimum=1,
            maximum=_MAX_SOURCE_ENTRIES,
        )
        _require_int(self.total_size_bytes, "local snapshot total size")
        if member_count != file_count:
            _fail("local bundle member count must equal the snapshot file count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_kind": "local_source_snapshot_and_retained_bundle_v1",
            "bundle_receipt_schema_version": self.bundle_receipt_schema_version,
            "bundle_receipt_sha256": self.bundle_receipt_sha256,
            "bundle_receipt_body_sha256": self.bundle_receipt_body_sha256,
            "bundle_descriptor_sha256": self.bundle_descriptor_sha256,
            "snapshot_manifest_schema_version": self.snapshot_manifest_schema_version,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_tree_schema_version": self.snapshot_tree_schema_version,
            "snapshot_tree_sha256": self.snapshot_tree_sha256,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "archive_member_count": self.archive_member_count,
            "member_inventory_sha256": self.member_inventory_sha256,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CpuOciBuildPublicationBindingV2:
    """Four independently pinned identities plus exact source linkage for one build."""

    build_plan_schema_version: str
    build_plan_sha256: str
    intent_schema_version: str
    intent_sha256: str
    context_receipt_schema_version: str
    context_receipt_sha256: str
    execution_receipt_schema_version: str
    execution_receipt_sha256: str
    publication_receipt_schema_version: str
    publication_receipt_sha256: str
    image_id: str
    external_source_receipt_sha256: str
    external_source_archive_sha256: str
    external_source_tree_sha256: str
    local_source_receipt_sha256: str
    local_source_archive_sha256: str
    local_snapshot_manifest_sha256: str
    local_snapshot_tree_sha256: str

    def __post_init__(self) -> None:
        if (
            self.build_plan_schema_version != CPU_OCI_BUILD_PLAN_SCHEMA_VERSION
            or self.intent_schema_version != CPU_OCI_BUILD_INTENT_SCHEMA_VERSION
            or self.context_receipt_schema_version
            != CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION
            or self.execution_receipt_schema_version
            != CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION
            or self.publication_receipt_schema_version
            != CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION
        ):
            _fail("CPU OCI build producer schema differs")
        for value, label in (
            (self.build_plan_sha256, "CPU OCI build plan"),
            (self.intent_sha256, "CPU OCI build intent"),
            (self.context_receipt_sha256, "CPU OCI context receipt"),
            (self.execution_receipt_sha256, "CPU OCI execution receipt"),
            (self.publication_receipt_sha256, "CPU OCI publication receipt"),
            (self.external_source_receipt_sha256, "build external source receipt"),
            (self.external_source_archive_sha256, "build external source archive"),
            (self.external_source_tree_sha256, "build external source tree"),
            (self.local_source_receipt_sha256, "build local source receipt"),
            (self.local_source_archive_sha256, "build local source archive"),
            (self.local_snapshot_manifest_sha256, "build local snapshot manifest"),
            (self.local_snapshot_tree_sha256, "build local snapshot tree"),
        ):
            _require_sha256(value, label)
        _require_image_id(self.image_id, "CPU OCI image")
        if len(
            {
                self.context_receipt_sha256,
                self.execution_receipt_sha256,
                self.publication_receipt_sha256,
            }
        ) != 3:
            _fail("context, execution, and publication receipts must be independently distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_kind": "durable_cpu_oci_build_publication_v1",
            "build_plan_schema_version": self.build_plan_schema_version,
            "build_plan_sha256": self.build_plan_sha256,
            "intent_schema_version": self.intent_schema_version,
            "intent_sha256": self.intent_sha256,
            "context_receipt_schema_version": self.context_receipt_schema_version,
            "context_receipt_sha256": self.context_receipt_sha256,
            "execution_receipt_schema_version": self.execution_receipt_schema_version,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "publication_receipt_schema_version": self.publication_receipt_schema_version,
            "publication_receipt_sha256": self.publication_receipt_sha256,
            "image_id": self.image_id,
            "source_linkage": {
                "external_source_receipt_sha256": self.external_source_receipt_sha256,
                "external_source_archive_sha256": self.external_source_archive_sha256,
                "external_source_tree_sha256": self.external_source_tree_sha256,
                "local_source_receipt_sha256": self.local_source_receipt_sha256,
                "local_source_archive_sha256": self.local_source_archive_sha256,
                "local_snapshot_manifest_sha256": self.local_snapshot_manifest_sha256,
                "local_snapshot_tree_sha256": self.local_snapshot_tree_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateResourceBindingV2:
    """Pre-observation integer ceilings for one candidate, in frozen field order."""

    candidate_id: str
    ceilings: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.candidate_id not in MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS:
            _fail("resource candidate is unknown")
        if (
            type(self.ceilings) is not tuple
            or any(type(item) is not tuple or len(item) != 2 for item in self.ceilings)
            or tuple(item[0] for item in self.ceilings) != RESOURCE_CEILING_FIELDS
        ):
            _fail("resource ceilings must use the exact 28-field order")
        values: dict[str, int] = {}
        for name, value in self.ceilings:
            values[name] = _require_int(value, f"resource ceiling {name}")
        if values["max_environment_interactions"] < _HORIZON:
            _fail("resource ceiling cannot cover one exact matched-v3 horizon")
        if values["max_thread_count"] < 1 or values["max_attempt_count"] < 1:
            _fail("resource thread and attempt ceilings must be positive")
        if values["max_failure_count"] >= values["max_attempt_count"]:
            _fail("resource failure ceiling must be below the attempt ceiling")

    @classmethod
    def from_mapping(
        cls,
        *,
        candidate_id: str,
        ceilings: Mapping[str, int],
    ) -> CandidateResourceBindingV2:
        """Create a binding only from a complete exact-key ceiling mapping."""

        if type(ceilings) is not dict or set(ceilings) != set(RESOURCE_CEILING_FIELDS):
            _fail("resource ceiling mapping must have the exact frozen fields")
        return cls(
            candidate_id=candidate_id,
            ceilings=tuple((name, ceilings[name]) for name in RESOURCE_CEILING_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **dict(self.ceilings)}


@dataclass(frozen=True, slots=True)
class CandidatePublisherBindingV2:
    """One real, complete, nonsynthetic strict publisher implementation."""

    candidate_id: str
    publisher_kind: str
    descriptor_schema_version: str
    descriptor_sha256: str
    publication_schema_version: str
    implementation_path: str
    implementation_source_sha256: str
    local_source_tree_sha256: str
    reload_validator_descriptor_sha256: str
    implementation_complete: bool
    synthetic: bool

    def __post_init__(self) -> None:
        if self.candidate_id not in IMPLEMENTED_PUBLISHER_CANDIDATE_IDS:
            _fail("candidate has no implemented publisher contract")
        if self.implementation_complete is not True or self.synthetic is not False:
            _fail("publisher bindings must be complete real implementations")
        if (
            self.publisher_kind != "adapter_reward_publication_v1"
            or self.descriptor_schema_version
            != "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            or self.descriptor_sha256 != _ADAPTER_PUBLICATION_DESCRIPTOR_SHA256
            or self.publication_schema_version
            != "alberta.forager_matched_v3.adapter_reward_publication.v1"
            or self.implementation_path != _ADAPTER_PUBLICATION_PATH
            or self.implementation_source_sha256 != _ADAPTER_PUBLICATION_SOURCE_SHA256
            or self.reload_validator_descriptor_sha256
            != _ADAPTER_PUBLICATION_DESCRIPTOR_SHA256
        ):
            _fail("adapter publisher differs from its implemented content identity")
        _require_relative_path(self.implementation_path, "publisher implementation path")
        for value, label in (
            (self.descriptor_sha256, "publisher descriptor"),
            (self.implementation_source_sha256, "publisher source"),
            (self.local_source_tree_sha256, "publisher local source tree"),
            (self.reload_validator_descriptor_sha256, "publisher reload validator"),
        ):
            _require_sha256(value, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "publisher_kind": self.publisher_kind,
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "publication_schema_version": self.publication_schema_version,
            "implementation_path": self.implementation_path,
            "implementation_source_sha256": self.implementation_source_sha256,
            "local_source_tree_sha256": self.local_source_tree_sha256,
            "reload_validator_descriptor_sha256": (
                self.reload_validator_descriptor_sha256
            ),
            "implementation_complete": self.implementation_complete,
            "synthetic": self.synthetic,
        }


def matched_v3_implemented_publisher_bindings_v2(
    *, local_source_tree_sha256: str
) -> tuple[CandidatePublisherBindingV2, ...]:
    """Return exact bindings for the only two implemented result publishers."""

    tree = _require_sha256(local_source_tree_sha256, "publisher local source tree")
    return tuple(
        CandidatePublisherBindingV2(
            candidate_id=candidate_id,
            publisher_kind="adapter_reward_publication_v1",
            descriptor_schema_version=(
                "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            ),
            descriptor_sha256=_ADAPTER_PUBLICATION_DESCRIPTOR_SHA256,
            publication_schema_version=(
                "alberta.forager_matched_v3.adapter_reward_publication.v1"
            ),
            implementation_path=_ADAPTER_PUBLICATION_PATH,
            implementation_source_sha256=_ADAPTER_PUBLICATION_SOURCE_SHA256,
            local_source_tree_sha256=tree,
            reload_validator_descriptor_sha256=_ADAPTER_PUBLICATION_DESCRIPTOR_SHA256,
            implementation_complete=True,
            synthetic=False,
        )
        for candidate_id in IMPLEMENTED_PUBLISHER_CANDIDATE_IDS
    )


def _dependencies() -> dict[str, Any]:
    return {
        "configuration_plan": {
            "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
        "candidate_universe": {
            "schema_version": "alberta.forager_matched_v3_candidate_universe.v1",
            "sha256": _CANDIDATE_UNIVERSE_SHA256,
        },
        "cumulative_reward_metric": {
            "schema_version": "alberta.forager_matched_v3.cumulative_reward_metric.v1",
            "sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
        },
        "trial_block_generator_plan": {
            "schema_version": "alberta.forager_matched_v3.trial_block_generator_plan.v1",
            "sha256": _TRIAL_BLOCK_GENERATOR_PLAN_SHA256,
        },
        "external_source_producer": {
            "publication_receipt_schema_version": (
                EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
            ),
            "publication_descriptor_sha256": _EXTERNAL_PUBLICATION_DESCRIPTOR_SHA256,
            "materialization_manifest_schema_version": (
                EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
            ),
            "materialization_identity_sha256": (
                _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
            ),
            "materializer_source_sha256": _EXTERNAL_MATERIALIZER_SOURCE_SHA256,
        },
        "local_source_producer": {
            "snapshot_manifest_schema_version": (
                LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
            ),
            "snapshot_tree_schema_version": LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
            "snapshot_descriptor_sha256": _LOCAL_SNAPSHOT_DESCRIPTOR_SHA256,
            "bundle_receipt_schema_version": LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION,
            "bundle_descriptor_sha256": _LOCAL_BUNDLE_DESCRIPTOR_SHA256,
        },
        "cpu_oci_build_producer": {
            "plan_schema_version": CPU_OCI_BUILD_PLAN_SCHEMA_VERSION,
            "intent_schema_version": CPU_OCI_BUILD_INTENT_SCHEMA_VERSION,
            "context_receipt_schema_version": CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
            "execution_receipt_schema_version": (
                CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION
            ),
            "publication_receipt_schema_version": (
                CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION
            ),
        },
    }


def _probe_profiles() -> list[dict[str, Any]]:
    return [
        {
            "profile_id": "qualification_seed_provenance_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS[
                "qualification_seed_observation"
            ],
            "acceptance_fields": [
                "independent_trust_root_receipt_pins_exact",
                "offline_signature_verification_exact",
                "preobservation_chronology_exact",
                "deterministic_case_seed_derivation_exact",
            ],
        },
        {
            "profile_id": "content_import_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS["candidate_observation"],
            "acceptance_fields": [
                "source_membership_exact",
                "configuration_membership_exact",
                "entrypoint_import_exact",
            ],
        },
        {
            "profile_id": "environment_rng_replay_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS["fresh_replay_observation"],
            "acceptance_fields": [
                "environment_seed_transport_exact",
                "reset_step_key_schedule_exact",
                "structural_replay_exact",
            ],
        },
        {
            "profile_id": "candidate_seed_transport_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS["candidate_observation"],
            "acceptance_fields": [
                "agent_seed_transport_exact",
                "environment_agent_derivations_distinct",
                "candidate_rng_membership_exact",
            ],
        },
        {
            "profile_id": "full_horizon_resource_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS["resource_observation"],
            "acceptance_fields": [
                "horizon_accounting_exact",
                "reward_membership_structural_only",
                "resource_observations_within_predeclared_integer_ceilings",
            ],
        },
        {
            "profile_id": "result_publication_roundtrip_v1",
            "required_observation_schema": _OBSERVATION_SCHEMAS["publication_observation"],
            "acceptance_fields": [
                "publisher_descriptor_membership_exact",
                "publisher_source_closure_membership_exact",
                "atomic_publication_exact",
                "strict_reload_exact",
                "full_file_digest_equivalence_exact",
                "score_and_reward_magnitude_not_decoded",
            ],
        },
    ]


def _acceptance_policy() -> dict[str, Any]:
    return {
        "required_observation_schemas": dict(_OBSERVATION_SCHEMAS),
        "all_28_candidate_publishers_required": True,
        "incomplete_publisher_allowed": False,
        "synthetic_publisher_allowed": False,
        "source_membership_exact": True,
        "configuration_membership_exact": True,
        "runtime_membership_exact": True,
        "seed_transport_replay_exact": True,
        "horizon_accounting_exact": True,
        "resource_accounting_complete": True,
        "result_publication_roundtrip_exact": True,
        "reward_magnitude_is_acceptance_input": False,
        "score_is_acceptance_input": False,
        "ranking_is_acceptance_input": False,
        "observation_validators_implemented_here": False,
    }


def _failure_policy() -> dict[str, Any]:
    return {
        "fixed_before_observation_required": True,
        "fixed_before_observation_verified_here": False,
        "fail_closed": True,
        "missing_required_observation": "reject_case",
        "missing_or_incomplete_publisher": "reject_plan",
        "synthetic_publisher": "reject_plan",
        "schema_or_digest_mismatch": "reject_plan",
        "build_identity_mismatch": "reject_plan",
        "resource_ceiling_exceeded": "reject_case_no_post_observation_retuning",
        "partial_candidate_coverage": "reject_plan",
        "reward_magnitude_is_failure_input": False,
        "score_is_failure_input": False,
        "ranking_is_failure_input": False,
    }


def _runtime_policy() -> dict[str, Any]:
    return {
        "build_publication_is_runtime_qualification": False,
        "image_identity_is_runtime_qualification": False,
        "fresh_runtime_observation_required": True,
        "networkless_linux_amd64_cpu_required": True,
        "required_helper_ids": ["drand_verify", "oci_runtime", "resource_observer"],
        "helper_implementations_bound_here": False,
        "runtime_qualified_here": False,
    }


def _issuance_policy() -> dict[str, Any]:
    return {
        "required_candidate_count": 28,
        "implemented_publisher_count": 2,
        "missing_publisher_count": 26,
        "publisher_registry_complete": False,
        "qualification_cases_issued": False,
        "qualification_seed_material_issued": False,
        "runtime_probe_authority_issued": False,
        "production_plan_issued": False,
        "production_plan_builder_available": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "build_qualified": False,
        "execution_authority_granted": False,
        "executed_bytecode_attested": False,
        "performance_claim_allowed": False,
        "production_plan_issued": False,
        "promotion_allowed": False,
        "publication_authority_granted": False,
        "publisher_registry_complete": False,
        "qualification_granted": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This artifact is a content descriptor and grants no execution or publication authority.",
        "The source receipts and CPU OCI build publication remain unqualified content lineage.",
        "A successful image build is not a runtime qualification or bytecode attestation.",
        "Only two of 28 strict candidate result publishers are implemented and bound.",
        "No qualification cases, seed material, probe receipts, or result bundles are issued.",
        "Observation schemas describe required future contracts; validators are not supplied here.",
        "No result permits promotion, performance, comparative, or universal SOTA claims.",
    ]


def _source_id(candidate_id: str) -> str:
    return "external_foragax_agents" if candidate_id in _EXTERNAL_CANDIDATE_IDS else "local_alberta"


def _publisher_requirement(
    candidate_id: str,
    publishers: Mapping[str, CandidatePublisherBindingV2],
) -> dict[str, Any]:
    binding = publishers.get(candidate_id)
    if binding is None:
        return {
            "status": "required_not_implemented",
            "binding_body_sha256": None,
            "publication_schema_version": None,
        }
    binding_dict = binding.to_dict()
    return {
        "status": "implemented_strict_publisher",
        "binding_body_sha256": hashlib.sha256(
            _canonical_json(binding_dict, newline=False)
        ).hexdigest(),
        "publication_schema_version": binding.publication_schema_version,
    }


def _candidate_requirements(
    resources: Mapping[str, CandidateResourceBindingV2],
    publishers: Mapping[str, CandidatePublisherBindingV2],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ordinal, candidate_id in enumerate(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS):
        resource = resources[candidate_id].to_dict()
        result.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                "configuration_record_sha256": _CONFIGURATION_RECORD_SHA256[candidate_id],
                "source_id": _source_id(candidate_id),
                "probe_profile_ids": [item["profile_id"] for item in _probe_profiles()],
                "resource_requirement_body_sha256": hashlib.sha256(
                    _canonical_json(resource, newline=False)
                ).hexdigest(),
                "publisher_requirement": _publisher_requirement(candidate_id, publishers),
                "acceptance_policy": _acceptance_policy(),
            }
        )
    return result


def _validate_exact_order(
    values: Sequence[object],
    *,
    expected_ids: tuple[str, ...],
    expected_type: type[object],
    label: str,
) -> None:
    if type(values) not in {tuple, list}:
        _fail(f"{label} must be one exact tuple or list")
    if len({id(value) for value in values}) != len(values):
        _fail(f"{label} contains aliased records")
    if any(type(value) is not expected_type for value in values):
        _fail(f"{label} contains an inexact record type")
    if tuple(getattr(value, "candidate_id", None) for value in values) != expected_ids:
        _fail(f"{label} coverage or order differs")


def _check_build_pins(
    build: CpuOciBuildPublicationBindingV2,
    *,
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_publication_receipt_sha256: str,
    expected_image_id: str,
) -> None:
    expected_context = _require_sha256(
        expected_context_receipt_sha256, "independent expected context receipt"
    )
    expected_execution = _require_sha256(
        expected_execution_receipt_sha256, "independent expected execution receipt"
    )
    expected_publication = _require_sha256(
        expected_publication_receipt_sha256, "independent expected publication receipt"
    )
    expected_image = _require_image_id(expected_image_id, "independent expected image ID")
    if not (
        hmac.compare_digest(build.context_receipt_sha256, expected_context)
        and hmac.compare_digest(build.execution_receipt_sha256, expected_execution)
        and hmac.compare_digest(build.publication_receipt_sha256, expected_publication)
        and hmac.compare_digest(build.image_id, expected_image)
    ):
        _fail("CPU OCI context, execution, publication, or image caller pin differs")


def _check_source_linkage(
    external: ExternalSourcePublicationBindingV2,
    local: LocalSourceBundleBindingV2,
    build: CpuOciBuildPublicationBindingV2,
) -> None:
    if (
        build.external_source_receipt_sha256 != external.publication_receipt_sha256
        or build.external_source_archive_sha256 != external.archive_sha256
        or build.external_source_tree_sha256 != external.source_tree_sha256
        or build.local_source_receipt_sha256 != local.bundle_receipt_sha256
        or build.local_source_archive_sha256 != local.archive_sha256
        or build.local_snapshot_manifest_sha256 != local.snapshot_manifest_sha256
        or build.local_snapshot_tree_sha256 != local.snapshot_tree_sha256
    ):
        _fail("CPU OCI build source linkage is cross-wired")
    if external.source_tree_sha256 == local.snapshot_tree_sha256:
        _fail("external and local source tree identities must be distinct")


def build_matched_v3_qualification_plan_v2(
    *,
    external_source: ExternalSourcePublicationBindingV2,
    local_source: LocalSourceBundleBindingV2,
    cpu_oci_build: CpuOciBuildPublicationBindingV2,
    resource_bindings: Sequence[CandidateResourceBindingV2],
    publisher_bindings: Sequence[CandidatePublisherBindingV2],
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_publication_receipt_sha256: str,
    expected_image_id: str,
) -> dict[str, Any]:
    """Build one canonicalizable descriptor without issuing a production plan."""

    if type(external_source) is not ExternalSourcePublicationBindingV2:
        _fail("external source binding type differs")
    if type(local_source) is not LocalSourceBundleBindingV2:
        _fail("local source binding type differs")
    if type(cpu_oci_build) is not CpuOciBuildPublicationBindingV2:
        _fail("CPU OCI build binding type differs")
    _validate_exact_order(
        resource_bindings,
        expected_ids=MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS,
        expected_type=CandidateResourceBindingV2,
        label="resource bindings",
    )
    _validate_exact_order(
        publisher_bindings,
        expected_ids=IMPLEMENTED_PUBLISHER_CANDIDATE_IDS,
        expected_type=CandidatePublisherBindingV2,
        label="implemented publisher bindings",
    )
    _check_source_linkage(external_source, local_source, cpu_oci_build)
    _check_build_pins(
        cpu_oci_build,
        expected_context_receipt_sha256=expected_context_receipt_sha256,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_publication_receipt_sha256=expected_publication_receipt_sha256,
        expected_image_id=expected_image_id,
    )
    if any(
        publisher.local_source_tree_sha256 != local_source.snapshot_tree_sha256
        for publisher in publisher_bindings
    ):
        _fail("implemented publisher is outside the bound local source tree")

    resources_by_id = {resource.candidate_id: resource for resource in resource_bindings}
    publishers_by_id = {publisher.candidate_id: publisher for publisher in publisher_bindings}
    body: dict[str, Any] = {
        "schema_version": QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
        "status": QUALIFICATION_PLAN_V2_STATUS,
        "classification": QUALIFICATION_PLAN_V2_CLASSIFICATION,
        "dependency_bindings": _dependencies(),
        "producer_bindings": {
            "external_source": external_source.to_dict(),
            "local_source": local_source.to_dict(),
            "cpu_oci_build": cpu_oci_build.to_dict(),
        },
        "candidate_order": list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS),
        "probe_profiles": _probe_profiles(),
        "candidate_requirements": _candidate_requirements(resources_by_id, publishers_by_id),
        "resource_contract": {
            "scope": "per_candidate_public_qualification_case_integer_ceilings_v2",
            "integer_ceiling_fields": list(RESOURCE_CEILING_FIELDS),
            "requirements": [resource.to_dict() for resource in resource_bindings],
            "compute_efficiency_claimed": False,
            "resource_matched_claimed": False,
        },
        "publisher_registry": {
            "required_candidate_ids": list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS),
            "implemented_bindings": [publisher.to_dict() for publisher in publisher_bindings],
            "missing_candidate_ids": list(MISSING_PUBLISHER_CANDIDATE_IDS),
            "implemented_count": 2,
            "missing_count": 26,
            "complete": False,
            "synthetic_bindings_allowed": False,
            "incomplete_bindings_allowed": False,
        },
        "acceptance_policy": _acceptance_policy(),
        "failure_policy": _failure_policy(),
        "runtime_policy": _runtime_policy(),
        "issuance_policy": _issuance_policy(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    plan = {
        **body,
        "plan_body_sha256": hashlib.sha256(_canonical_json(body, newline=False)).hexdigest(),
    }
    _validate_plan(plan)
    return copy.deepcopy(plan)


def _external_from_dict(value: object) -> ExternalSourcePublicationBindingV2:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(
                {
                "producer_kind",
                "publication_receipt_schema_version",
                "publication_receipt_sha256",
                "publication_receipt_body_sha256",
                "publication_contract_descriptor_sha256",
                "materialization_manifest_schema_version",
                "materialization_manifest_sha256",
                "materialization_payload_sha256",
                "materialization_identity_sha256",
                "source_tree_sha256",
                "staging_manifest_schema_version",
                "staging_manifest_sha256",
                "staging_manifest_body_sha256",
                "archive_sha256",
                "archive_size_bytes",
                "archive_member_count",
                "archive_inventory_sha256",
                }
            ),
            "external source binding",
        )
    )
    if item.pop("producer_kind") != "durable_external_source_publication_v1_materialization_v2":
        _fail("external source producer kind differs")
    return ExternalSourcePublicationBindingV2(**item)


def _local_from_dict(value: object) -> LocalSourceBundleBindingV2:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(
                {
                "producer_kind",
                "bundle_receipt_schema_version",
                "bundle_receipt_sha256",
                "bundle_receipt_body_sha256",
                "bundle_descriptor_sha256",
                "snapshot_manifest_schema_version",
                "snapshot_manifest_sha256",
                "snapshot_tree_schema_version",
                "snapshot_tree_sha256",
                "archive_sha256",
                "archive_size_bytes",
                "archive_member_count",
                "member_inventory_sha256",
                "directory_count",
                "file_count",
                "total_size_bytes",
                }
            ),
            "local source binding",
        )
    )
    if item.pop("producer_kind") != "local_source_snapshot_and_retained_bundle_v1":
        _fail("local source producer kind differs")
    return LocalSourceBundleBindingV2(**item)


def _build_from_dict(value: object) -> CpuOciBuildPublicationBindingV2:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(
                {
                "producer_kind",
                "build_plan_schema_version",
                "build_plan_sha256",
                "intent_schema_version",
                "intent_sha256",
                "context_receipt_schema_version",
                "context_receipt_sha256",
                "execution_receipt_schema_version",
                "execution_receipt_sha256",
                "publication_receipt_schema_version",
                "publication_receipt_sha256",
                "image_id",
                "source_linkage",
                }
            ),
            "CPU OCI build binding",
        )
    )
    if item.pop("producer_kind") != "durable_cpu_oci_build_publication_v1":
        _fail("CPU OCI build producer kind differs")
    linkage = _require_exact_keys(
        item.pop("source_linkage"),
        frozenset(
            {
                "external_source_receipt_sha256",
                "external_source_archive_sha256",
                "external_source_tree_sha256",
                "local_source_receipt_sha256",
                "local_source_archive_sha256",
                "local_snapshot_manifest_sha256",
                "local_snapshot_tree_sha256",
            }
        ),
        "CPU OCI build source linkage",
    )
    return CpuOciBuildPublicationBindingV2(**item, **linkage)


def _resource_from_dict(value: object) -> CandidateResourceBindingV2:
    item = _require_exact_keys(
        value,
        frozenset({"candidate_id", *RESOURCE_CEILING_FIELDS}),
        "candidate resource binding",
    )
    return CandidateResourceBindingV2(
        candidate_id=item["candidate_id"],
        ceilings=tuple((name, item[name]) for name in RESOURCE_CEILING_FIELDS),
    )


def _publisher_from_dict(value: object) -> CandidatePublisherBindingV2:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "candidate_id",
                "publisher_kind",
                "descriptor_schema_version",
                "descriptor_sha256",
                "publication_schema_version",
                "implementation_path",
                "implementation_source_sha256",
                "local_source_tree_sha256",
                "reload_validator_descriptor_sha256",
                "implementation_complete",
                "synthetic",
            }
        ),
        "candidate publisher binding",
    )
    return CandidatePublisherBindingV2(**item)


def _validate_plan(value: Mapping[str, Any]) -> None:
    _assert_plain_unaliased_json(value)
    plan = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "dependency_bindings",
                "producer_bindings",
                "candidate_order",
                "probe_profiles",
                "candidate_requirements",
                "resource_contract",
                "publisher_registry",
                "acceptance_policy",
                "failure_policy",
                "runtime_policy",
                "issuance_policy",
                "claims",
                "limitations",
                "plan_body_sha256",
            }
        ),
        "qualification-plan v2",
    )
    if (
        plan["schema_version"] != QUALIFICATION_PLAN_V2_SCHEMA_VERSION
        or plan["status"] != QUALIFICATION_PLAN_V2_STATUS
        or plan["classification"] != QUALIFICATION_PLAN_V2_CLASSIFICATION
    ):
        _fail("qualification-plan v2 identity differs")
    if not _exact_json_equal(plan["dependency_bindings"], _dependencies()):
        _fail("qualification-plan v2 dependency binding differs")
    producers = _require_exact_keys(
        plan["producer_bindings"],
        frozenset({"external_source", "local_source", "cpu_oci_build"}),
        "producer bindings",
    )
    external = _external_from_dict(producers["external_source"])
    local = _local_from_dict(producers["local_source"])
    build = _build_from_dict(producers["cpu_oci_build"])
    _check_source_linkage(external, local, build)

    if not _exact_json_equal(
        plan["candidate_order"],
        list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS),
    ):
        _fail("candidate order differs")
    if not _exact_json_equal(plan["probe_profiles"], _probe_profiles()):
        _fail("probe profile policy differs")

    resource_contract = _require_exact_keys(
        plan["resource_contract"],
        frozenset(
            {
                "scope",
                "integer_ceiling_fields",
                "requirements",
                "compute_efficiency_claimed",
                "resource_matched_claimed",
            }
        ),
        "resource contract",
    )
    if (
        resource_contract["scope"]
        != "per_candidate_public_qualification_case_integer_ceilings_v2"
        or resource_contract["integer_ceiling_fields"] != list(RESOURCE_CEILING_FIELDS)
        or resource_contract["compute_efficiency_claimed"] is not False
        or resource_contract["resource_matched_claimed"] is not False
    ):
        _fail("resource contract policy differs")
    raw_resources = resource_contract["requirements"]
    if type(raw_resources) is not list:
        _fail("resource requirements must be one list")
    resources = tuple(_resource_from_dict(item) for item in raw_resources)
    _validate_exact_order(
        resources,
        expected_ids=MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS,
        expected_type=CandidateResourceBindingV2,
        label="resource requirements",
    )

    registry = _require_exact_keys(
        plan["publisher_registry"],
        frozenset(
            {
                "required_candidate_ids",
                "implemented_bindings",
                "missing_candidate_ids",
                "implemented_count",
                "missing_count",
                "complete",
                "synthetic_bindings_allowed",
                "incomplete_bindings_allowed",
            }
        ),
        "publisher registry",
    )
    if (
        registry["required_candidate_ids"] != list(MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS)
        or registry["missing_candidate_ids"] != list(MISSING_PUBLISHER_CANDIDATE_IDS)
        or registry["implemented_count"] != 2
        or registry["missing_count"] != 26
        or registry["complete"] is not False
        or registry["synthetic_bindings_allowed"] is not False
        or registry["incomplete_bindings_allowed"] is not False
    ):
        _fail("publisher registry readiness differs")
    raw_publishers = registry["implemented_bindings"]
    if type(raw_publishers) is not list:
        _fail("implemented publisher bindings must be one list")
    publishers = tuple(_publisher_from_dict(item) for item in raw_publishers)
    _validate_exact_order(
        publishers,
        expected_ids=IMPLEMENTED_PUBLISHER_CANDIDATE_IDS,
        expected_type=CandidatePublisherBindingV2,
        label="implemented publisher bindings",
    )
    if any(
        publisher.local_source_tree_sha256 != local.snapshot_tree_sha256
        for publisher in publishers
    ):
        _fail("publisher source tree is cross-wired")

    resources_by_id = {resource.candidate_id: resource for resource in resources}
    publishers_by_id = {publisher.candidate_id: publisher for publisher in publishers}
    if not _exact_json_equal(
        plan["candidate_requirements"],
        _candidate_requirements(resources_by_id, publishers_by_id),
    ):
        _fail("candidate requirement order, policy, resource, or publisher binding differs")
    if not _exact_json_equal(plan["acceptance_policy"], _acceptance_policy()):
        _fail("acceptance policy differs")
    if not _exact_json_equal(plan["failure_policy"], _failure_policy()):
        _fail("failure policy differs")
    if not _exact_json_equal(plan["runtime_policy"], _runtime_policy()):
        _fail("runtime policy differs")
    if not _exact_json_equal(plan["issuance_policy"], _issuance_policy()):
        _fail("issuance policy differs")
    if not _exact_json_equal(plan["claims"], _claims()) or any(
        item is not False for item in plan["claims"].values()
    ):
        _fail("qualification-plan authority claim became true")
    if not _exact_json_equal(plan["limitations"], _limitations()):
        _fail("qualification-plan limitations differ")

    supplied_body_sha256 = _require_sha256(plan["plan_body_sha256"], "plan body")
    body = copy.deepcopy(plan)
    body.pop("plan_body_sha256")
    expected_body_sha256 = hashlib.sha256(_canonical_json(body, newline=False)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, expected_body_sha256):
        _fail("qualification-plan body digest differs")
    _assert_plain_unaliased_json(plan)
    _canonical_json(plan)


def canonical_matched_v3_qualification_plan_v2_bytes(
    plan: Mapping[str, Any],
    *,
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_publication_receipt_sha256: str,
    expected_image_id: str,
) -> bytes:
    """Validate and encode one exact descriptor with independent build pins."""

    _validate_plan(plan)
    build = _build_from_dict(cast(dict[str, Any], plan["producer_bindings"])["cpu_oci_build"])
    _check_build_pins(
        build,
        expected_context_receipt_sha256=expected_context_receipt_sha256,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_publication_receipt_sha256=expected_publication_receipt_sha256,
        expected_image_id=expected_image_id,
    )
    return _canonical_json(plan)


def replay_matched_v3_qualification_plan_v2(
    raw: bytes,
    *,
    expected_plan_sha256: str,
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_publication_receipt_sha256: str,
    expected_image_id: str,
) -> dict[str, Any]:
    """Strictly replay canonical bytes under five independent caller pins."""

    expected_plan = _require_sha256(expected_plan_sha256, "expected qualification plan")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_plan
    ):
        _fail("qualification-plan full-file SHA-256 differs")
    plan = _strict_json_load(raw)
    _validate_plan(plan)
    build = _build_from_dict(cast(dict[str, Any], plan["producer_bindings"])["cpu_oci_build"])
    _check_build_pins(
        build,
        expected_context_receipt_sha256=expected_context_receipt_sha256,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_publication_receipt_sha256=expected_publication_receipt_sha256,
        expected_image_id=expected_image_id,
    )
    return copy.deepcopy(plan)


def parse_matched_v3_qualification_plan_v2(
    raw: bytes,
    *,
    expected_plan_sha256: str,
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_publication_receipt_sha256: str,
    expected_image_id: str,
) -> dict[str, Any]:
    """Alias the strict replay API under the conventional parser name."""

    return replay_matched_v3_qualification_plan_v2(
        raw,
        expected_plan_sha256=expected_plan_sha256,
        expected_context_receipt_sha256=expected_context_receipt_sha256,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_publication_receipt_sha256=expected_publication_receipt_sha256,
        expected_image_id=expected_image_id,
    )


__all__ = [
    "CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_INTENT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_PLAN_SCHEMA_VERSION",
    "CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "CandidatePublisherBindingV2",
    "CandidateResourceBindingV2",
    "CpuOciBuildPublicationBindingV2",
    "EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION",
    "ExternalSourcePublicationBindingV2",
    "ForagerMatchedV3QualificationPlanV2Error",
    "IMPLEMENTED_PUBLISHER_CANDIDATE_IDS",
    "LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION",
    "LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION",
    "LocalSourceBundleBindingV2",
    "MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS",
    "MISSING_PUBLISHER_CANDIDATE_IDS",
    "QUALIFICATION_PLAN_V2_CLASSIFICATION",
    "QUALIFICATION_PLAN_V2_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V2_STATUS",
    "RESOURCE_CEILING_FIELDS",
    "build_matched_v3_qualification_plan_v2",
    "canonical_matched_v3_qualification_plan_v2_bytes",
    "matched_v3_implemented_publisher_bindings_v2",
    "parse_matched_v3_qualification_plan_v2",
    "replay_matched_v3_qualification_plan_v2",
]
