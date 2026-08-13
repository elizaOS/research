"""Focused tests for the source-only matched-v3 publication commitment wrapper."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
from typing import Any, Literal

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_publication_commitment as commitment,
)

EXPECTED_DESCRIPTOR_SHA256 = "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
EXPECTED_EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
EXPECTED_MAX_PUBLICATION_BYTES = 1024 * 1024 * 1024

EXPECTED_LOCAL_CANDIDATE_IDS = (
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
EXPECTED_EXTERNAL_CANDIDATE_IDS = (
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
EXPECTED_ADAPTER_CANDIDATE_IDS = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
EXPECTED_CANDIDATE_ORDER = (
    EXPECTED_LOCAL_CANDIDATE_IDS
    + EXPECTED_EXTERNAL_CANDIDATE_IDS[:9]
    + EXPECTED_ADAPTER_CANDIDATE_IDS
    + EXPECTED_EXTERNAL_CANDIDATE_IDS[9:]
)
EXPECTED_PPO_EXTERNAL_CANDIDATE_IDS = (
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
)

EXPECTED_LOCAL_ROLE_PATHS = (
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
EXPECTED_EXTERNAL_ROLE_PATHS = (
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
EXPECTED_ADAPTER_ROLE_PATHS = (
    ("publication_manifest", "publication.json"),
    ("adapter_bundle_manifest", "adapter-bundle-manifest.json"),
    ("runner_result_receipt", "runner-result-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
)

EXPECTED_WRAPPER_SCHEMA = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)
EXPECTED_CONTRACT_DESCRIPTOR_SCHEMA = (
    "alberta.forager_matched_v3.qualification_publication_commitment_contract_descriptor.v1"
)
EXPECTED_LOCAL_METADATA_SCHEMA = "alberta.forager_matched_v3.local_reward_publication_metadata.v1"
EXPECTED_LOCAL_PUBLISHER_SCHEMA = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
EXPECTED_EXTERNAL_METADATA_SCHEMA = (
    "alberta.forager_matched_v3.external_reward_publication_metadata.v1"
)
EXPECTED_EXTERNAL_PUBLISHER_SCHEMA = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
EXPECTED_ADAPTER_METADATA_SCHEMA = (
    "alberta.forager_matched_v3.adapter_qualification_publication_metadata.v1"
)
EXPECTED_ADAPTER_PUBLISHER_SCHEMA = (
    "alberta.forager_matched_v3.adapter_qualification_publication_descriptor.v1"
)
EXPECTED_GENERIC_ATOMIC_SCHEMA = "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
EXPECTED_ADAPTER_ATOMIC_SCHEMA = (
    "alberta.forager_matched_v3.adapter_qualification_atomic_publication_descriptor.v1"
)
EXPECTED_EXTERNAL_NATIVE_RECEIPT_SCHEMA = (
    "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"
)
EXPECTED_ADAPTER_NATIVE_RECEIPT_SCHEMA = (
    "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1"
)

EXPECTED_INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
)
EXPECTED_INCOMPATIBLE_ADAPTER_SOURCE_SHA256S = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
)
EXPECTED_INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S = (
    *EXPECTED_INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S,
    *EXPECTED_INCOMPATIBLE_ADAPTER_SOURCE_SHA256S,
)


class _StrSubclass(str):
    pass


class _IntSubclass(int):
    pass


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _inventory_digest(
    files: tuple[commitment.PublicationFileRecordV1, ...],
) -> str:
    projection = {
        "files": [
            {
                "role": item.role,
                "name": item.name,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in files
        ]
    }
    return hashlib.sha256(_canonical(projection, newline=False)).hexdigest()


def _rebody(value: dict[str, Any]) -> bytes:
    body = copy.deepcopy(value)
    body.pop("wrapper_body_sha256", None)
    value["wrapper_body_sha256"] = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    return _canonical(value)


def _artifact(schema: str, label: str) -> commitment.ArtifactIdentityV1:
    return commitment.ArtifactIdentityV1(
        schema_version=schema,
        file_sha256=_sha(f"{label}-file"),
        body_sha256=_sha(f"{label}-body"),
    )


def _producer(schema: str, label: str) -> commitment.ProducerIdentityV1:
    return commitment.ProducerIdentityV1(
        descriptor_schema_version=schema,
        descriptor_sha256=_sha(f"{label}-descriptor"),
        source_sha256=_sha(f"{label}-source"),
    )


def _family(candidate_id: str) -> Literal["local", "external", "adapter"]:
    if candidate_id in EXPECTED_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate_id in EXPECTED_EXTERNAL_CANDIDATE_IDS:
        return "external"
    return "adapter"


def _role_paths(
    family: Literal["local", "external", "adapter"],
) -> tuple[tuple[str, str], ...]:
    if family == "local":
        return EXPECTED_LOCAL_ROLE_PATHS
    if family == "external":
        return EXPECTED_EXTERNAL_ROLE_PATHS
    return EXPECTED_ADAPTER_ROLE_PATHS


def _profiles(
    family: Literal["local", "external", "adapter"],
) -> tuple[str, str, str, str | None]:
    if family == "local":
        return (
            EXPECTED_LOCAL_PUBLISHER_SCHEMA,
            EXPECTED_LOCAL_METADATA_SCHEMA,
            EXPECTED_GENERIC_ATOMIC_SCHEMA,
            None,
        )
    if family == "external":
        return (
            EXPECTED_EXTERNAL_PUBLISHER_SCHEMA,
            EXPECTED_EXTERNAL_METADATA_SCHEMA,
            EXPECTED_GENERIC_ATOMIC_SCHEMA,
            EXPECTED_EXTERNAL_NATIVE_RECEIPT_SCHEMA,
        )
    return (
        EXPECTED_ADAPTER_PUBLISHER_SCHEMA,
        EXPECTED_ADAPTER_METADATA_SCHEMA,
        EXPECTED_ADAPTER_ATOMIC_SCHEMA,
        EXPECTED_ADAPTER_NATIVE_RECEIPT_SCHEMA,
    )


def _files(
    family: Literal["local", "external", "adapter"],
    candidate_id: str,
    label: str,
) -> tuple[commitment.PublicationFileRecordV1, ...]:
    records: list[commitment.PublicationFileRecordV1] = []
    for index, (role, name) in enumerate(_role_paths(family)):
        empty_video = (
            family == "external"
            and role == "upstream_video_slot"
            and candidate_id not in EXPECTED_PPO_EXTERNAL_CANDIDATE_IDS
        )
        size = 0 if empty_video else index + 11
        digest = (
            EXPECTED_EMPTY_FILE_SHA256 if empty_video else _sha(f"{label}-{index}-{role}-{name}")
        )
        records.append(
            commitment.PublicationFileRecordV1(
                role=role,
                name=name,
                size_bytes=size,
                sha256=digest,
            )
        )
    return tuple(records)


def _bindings(
    ordinal: int,
    *,
    label: str | None = None,
) -> commitment.PublicationCommitmentBindingsV1:
    candidate_id = EXPECTED_CANDIDATE_ORDER[ordinal]
    family = _family(candidate_id)
    prefix = label or f"case-{ordinal:02d}-{candidate_id}"
    publisher_schema, metadata_schema, atomic_schema, receipt_schema = _profiles(family)
    files = _files(family, candidate_id, prefix)
    native_receipt = (
        None if receipt_schema is None else _artifact(receipt_schema, f"{prefix}-native-receipt")
    )
    video_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ]
    if family != "external":
        video_mode = "not_applicable"
    elif candidate_id in EXPECTED_PPO_EXTERNAL_CANDIDATE_IDS:
        video_mode = "opaque_ppo_video"
    else:
        video_mode = "absent_for_continuing_zero_length_slot"
    return commitment.PublicationCommitmentBindingsV1(
        case_spine_sha256=_sha(f"{prefix}-case-spine"),
        case_ordinal=ordinal,
        candidate_id=candidate_id,
        candidate_family=family,
        qualification_case_id=f"qualification_{ordinal:02d}_{candidate_id}",
        publisher=_producer(publisher_schema, f"{prefix}-publisher"),
        publisher_metadata=_artifact(metadata_schema, f"{prefix}-publisher-metadata"),
        native_atomic_producer=_producer(atomic_schema, f"{prefix}-atomic"),
        native_publication_receipt=native_receipt,
        publication_address_sha256=files[0].sha256,
        publication_manifest_file_sha256=files[0].sha256,
        publication_manifest_body_sha256=_sha(f"{prefix}-publication-body"),
        file_inventory_sha256=_inventory_digest(files),
        published_bundle_sha256=_sha(f"{prefix}-published-bundle"),
        expected_reload_observation_sha256=_sha(f"{prefix}-expected-reload"),
        file_count=len(files),
        total_size_bytes=sum(item.size_bytes for item in files),
        maximum_total_size_bytes=EXPECTED_MAX_PUBLICATION_BYTES,
        video_slot_mode=video_mode,
        files=files,
    )


def _bindings_with_files(
    bindings: commitment.PublicationCommitmentBindingsV1,
    files: tuple[commitment.PublicationFileRecordV1, ...],
) -> commitment.PublicationCommitmentBindingsV1:
    return dataclasses.replace(
        bindings,
        files=files,
        publication_address_sha256=files[0].sha256,
        publication_manifest_file_sha256=files[0].sha256,
        file_inventory_sha256=_inventory_digest(files),
        file_count=len(files),
        total_size_bytes=sum(item.size_bytes for item in files),
    )


@pytest.mark.parametrize("ordinal", range(28))
def test_every_family_builds_roundtrips_and_preserves_native_receipt(ordinal: int) -> None:
    bindings = _bindings(ordinal)
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(bindings)
    raw = commitment.canonical_matched_v3_qualification_publication_commitment_bytes(wrapper)
    parsed = commitment.parse_matched_v3_qualification_publication_commitment(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert parsed == wrapper
    commitment.validate_matched_v3_qualification_publication_commitment_bindings(
        parsed,
        expected=bindings,
    )
    assert parsed.native_publication_receipt == bindings.native_publication_receipt
    assert parsed.publication_address_sha256 == parsed.files[0].sha256
    assert parsed.publication_manifest_file_sha256 == parsed.files[0].sha256
    assert parsed.file_inventory_sha256 == _inventory_digest(parsed.files)
    assert parsed.file_inventory_sha256 == commitment.publication_file_inventory_sha256(
        parsed.files
    )
    value = parsed.to_dict()
    supplied_body = value.pop("wrapper_body_sha256")
    independently_replayed_body = hashlib.sha256(_canonical(value, newline=False)).hexdigest()
    assert supplied_body == independently_replayed_body
    assert parsed.body_sha256 == independently_replayed_body
    assert all(value is False for value in parsed.to_dict()["capabilities"].values())
    assert all(value is False for value in parsed.to_dict()["readiness"].values())
    assert all(value is False for value in parsed.to_dict()["authority"].values())
    assert all(value is False for value in parsed.to_dict()["claims"].values())
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.case_spine_sha256 = _sha("mutation")  # type: ignore[misc]


def test_literal_protocol_expectations_match_the_production_surface() -> None:
    assert commitment.MATCHED_V3_LOCAL_CANDIDATE_IDS == EXPECTED_LOCAL_CANDIDATE_IDS
    assert commitment.MATCHED_V3_EXTERNAL_CANDIDATE_IDS == EXPECTED_EXTERNAL_CANDIDATE_IDS
    assert commitment.MATCHED_V3_ADAPTER_CANDIDATE_IDS == EXPECTED_ADAPTER_CANDIDATE_IDS
    assert commitment.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS == EXPECTED_CANDIDATE_ORDER
    assert commitment.MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS == EXPECTED_PPO_EXTERNAL_CANDIDATE_IDS
    assert commitment.LOCAL_PUBLICATION_ROLE_PATHS == EXPECTED_LOCAL_ROLE_PATHS
    assert commitment.EXTERNAL_PUBLICATION_ROLE_PATHS == EXPECTED_EXTERNAL_ROLE_PATHS
    assert commitment.ADAPTER_PUBLICATION_ROLE_PATHS == EXPECTED_ADAPTER_ROLE_PATHS
    assert commitment.QUALIFICATION_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION == (
        EXPECTED_WRAPPER_SCHEMA
    )
    assert (
        commitment.QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
        == EXPECTED_CONTRACT_DESCRIPTOR_SCHEMA
    )
    assert commitment.LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION == (EXPECTED_LOCAL_METADATA_SCHEMA)
    assert commitment.LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION == (
        EXPECTED_LOCAL_PUBLISHER_SCHEMA
    )
    assert commitment.EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION == (
        EXPECTED_EXTERNAL_METADATA_SCHEMA
    )
    assert commitment.EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION == (
        EXPECTED_EXTERNAL_PUBLISHER_SCHEMA
    )
    assert commitment.STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION == (
        EXPECTED_ADAPTER_METADATA_SCHEMA
    )
    assert commitment.STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION == (
        EXPECTED_ADAPTER_PUBLISHER_SCHEMA
    )
    assert commitment.ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION == (
        EXPECTED_GENERIC_ATOMIC_SCHEMA
    )
    assert commitment.STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION == (
        EXPECTED_ADAPTER_ATOMIC_SCHEMA
    )
    assert commitment.EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION == (
        EXPECTED_EXTERNAL_NATIVE_RECEIPT_SCHEMA
    )
    assert commitment.STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION == (
        EXPECTED_ADAPTER_NATIVE_RECEIPT_SCHEMA
    )
    assert commitment.INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S == (
        EXPECTED_INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S
    )
    assert commitment.INCOMPATIBLE_ADAPTER_SOURCE_SHA256S == (
        EXPECTED_INCOMPATIBLE_ADAPTER_SOURCE_SHA256S
    )
    assert commitment.INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S == frozenset(
        EXPECTED_INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S
    )
    assert commitment.EMPTY_FILE_SHA256 == EXPECTED_EMPTY_FILE_SHA256
    assert commitment.MAX_PUBLICATION_FILE_BYTES == EXPECTED_MAX_PUBLICATION_BYTES
    assert commitment.MAX_PUBLICATION_TOTAL_BYTES == EXPECTED_MAX_PUBLICATION_BYTES


def test_descriptor_has_final_semantics_and_replays_its_literal_pin() -> None:
    descriptor = commitment.matched_v3_qualification_publication_commitment_contract_descriptor()
    descriptor_bytes = getattr(
        commitment,
        "canonical_matched_v3_qualification_publication_commitment_contract_descriptor_bytes",
    )
    raw = descriptor_bytes()
    descriptor_value = json.loads(raw)
    supplied_descriptor_body = descriptor_value.pop("descriptor_body_sha256")
    independently_replayed_descriptor_body = hashlib.sha256(
        _canonical(descriptor_value, newline=False)
    ).hexdigest()

    assert (
        commitment.PINNED_QUALIFICATION_PUBLICATION_COMMITMENT_CONTRACT_DESCRIPTOR_SHA256
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_DESCRIPTOR_SHA256
    assert supplied_descriptor_body == independently_replayed_descriptor_body
    assert descriptor["descriptor_body_sha256"] == independently_replayed_descriptor_body
    assert "zero" not in descriptor["status"]
    assert descriptor["inventory_contract"]["role_bearing"] is True
    assert descriptor["inventory_contract"]["atomic_v2_roleless_digest_compatible"] is False
    assert descriptor["reload_contract"] == {
        "expected_digest_committed": True,
        "reload_performed_here": False,
        "digest_equality_validated_here": False,
        "later_phase_validation_required": True,
    }
    assert descriptor["native_receipt_preserved_not_replaced"] is True
    for envelope in ("capabilities", "readiness", "authority", "claims"):
        assert all(value is False for value in descriptor[envelope].values())
    assert (
        commitment.matched_v3_qualification_publication_commitment_contract_descriptor_sha256()
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        commitment.parse_matched_v3_qualification_publication_commitment_contract_descriptor(
            raw,
            expected_file_sha256=EXPECTED_DESCRIPTOR_SHA256,
        )
        == descriptor
    )

    with pytest.raises(
        commitment.ForagerMatchedV3QualificationPublicationCommitmentError,
        match="body digest differs",
    ):
        invalid_body = copy.deepcopy(descriptor)
        invalid_body["status"] = "coherently_untrusted_status"
        invalid_raw = _canonical(invalid_body)
        commitment.parse_matched_v3_qualification_publication_commitment_contract_descriptor(
            invalid_raw,
            expected_file_sha256=hashlib.sha256(invalid_raw).hexdigest(),
        )


def test_inventory_digest_is_role_bearing_and_exactly_replayed() -> None:
    bindings = _bindings(23)
    roleless = hashlib.sha256(
        _canonical(
            [
                {
                    "name": item.name,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in bindings.files
            ],
            newline=False,
        )
    ).hexdigest()
    assert bindings.file_inventory_sha256 != roleless

    changed_first = dataclasses.replace(bindings.files[0], role="different_manifest_role")
    changed_files = (changed_first, *bindings.files[1:])
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings,
            files=changed_files,
            file_inventory_sha256=_inventory_digest(changed_files),
        )

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(bindings, files=tuple(reversed(bindings.files)))
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(bindings, file_count=bindings.file_count + 1)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(bindings, total_size_bytes=bindings.total_size_bytes + 1)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(bindings, maximum_total_size_bytes=1024)


def test_address_manifest_and_nonempty_file_rules_fail_closed() -> None:
    bindings = _bindings(23)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(bindings, publication_address_sha256=_sha("wrong-address"))
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings,
            publication_manifest_file_sha256=_sha("wrong-manifest-file"),
        )

    empty_receipt = dataclasses.replace(
        bindings.files[2],
        size_bytes=0,
        sha256=EXPECTED_EMPTY_FILE_SHA256,
    )
    files = (*bindings.files[:2], empty_receipt, *bindings.files[3:])
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings,
            files=files,
            total_size_bytes=sum(item.size_bytes for item in files),
            file_inventory_sha256=_inventory_digest(files),
        )

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.PublicationFileRecordV1(
            role="stdout",
            name="stdout.bin",
            size_bytes=0,
            sha256=_sha("nonempty-digest"),
        )


@pytest.mark.parametrize("ordinal", [0, 14])
def test_local_and_external_stdout_stderr_may_be_exactly_empty(ordinal: int) -> None:
    bindings = _bindings(ordinal)
    files = tuple(
        dataclasses.replace(
            item,
            size_bytes=0,
            sha256=EXPECTED_EMPTY_FILE_SHA256,
        )
        if item.role in {"stdout", "stderr"}
        else item
        for item in bindings.files
    )
    updated = _bindings_with_files(bindings, files)
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(updated)
    by_role = {item.role: item for item in wrapper.files}
    for role in ("stdout", "stderr"):
        assert by_role[role].size_bytes == 0
        assert by_role[role].sha256 == EXPECTED_EMPTY_FILE_SHA256


@pytest.mark.parametrize("file_index", range(5))
def test_each_adapter_publication_role_rejects_empty_content(file_index: int) -> None:
    bindings = _bindings(23)
    empty_record = dataclasses.replace(
        bindings.files[file_index],
        size_bytes=0,
        sha256=EXPECTED_EMPTY_FILE_SHA256,
    )
    files = (
        *bindings.files[:file_index],
        empty_record,
        *bindings.files[file_index + 1 :],
    )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        _bindings_with_files(bindings, files)


def test_external_video_sentinel_is_candidate_exact() -> None:
    continuing = _bindings(14)
    ppo = _bindings(21)
    continuing_video = continuing.files[6]
    ppo_video = ppo.files[6]

    assert continuing.video_slot_mode == "absent_for_continuing_zero_length_slot"
    assert continuing_video.size_bytes == 0
    assert continuing_video.sha256 == EXPECTED_EMPTY_FILE_SHA256
    assert ppo.video_slot_mode == "opaque_ppo_video"
    assert ppo_video.size_bytes > 0

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(continuing, video_slot_mode="opaque_ppo_video")
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(ppo, video_slot_mode="absent_for_continuing_zero_length_slot")
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(_bindings(0), video_slot_mode="opaque_ppo_video")

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        nonempty_video = dataclasses.replace(
            continuing_video,
            size_bytes=23,
            sha256=_sha("forbidden-continuing-video"),
        )
        files = (*continuing.files[:6], nonempty_video, *continuing.files[7:])
        _bindings_with_files(continuing, files)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        empty_video = dataclasses.replace(
            ppo_video,
            size_bytes=0,
            sha256=EXPECTED_EMPTY_FILE_SHA256,
        )
        files = (*ppo.files[:6], empty_video, *ppo.files[7:])
        _bindings_with_files(ppo, files)


def test_native_receipt_presence_and_schema_rules_are_exact() -> None:
    local = _bindings(0)
    external = _bindings(14)
    adapter = _bindings(23)

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            local,
            native_publication_receipt=_artifact(
                EXPECTED_EXTERNAL_NATIVE_RECEIPT_SCHEMA,
                "forbidden-local-native",
            ),
        )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(external, native_publication_receipt=None)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(adapter, native_publication_receipt=None)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            adapter,
            native_publication_receipt=_artifact(
                EXPECTED_EXTERNAL_NATIVE_RECEIPT_SCHEMA,
                "wrong-adapter-native-schema",
            ),
        )


def test_old_adapter_v1_v2_runner_and_compiled_identities_are_rejected() -> None:
    adapter = _bindings(23)
    for index, digest in enumerate(EXPECTED_INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                adapter,
                publisher=dataclasses.replace(
                    adapter.publisher,
                    descriptor_sha256=digest,
                    source_sha256=_sha(f"fresh-publisher-source-{index}"),
                ),
            )
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                adapter,
                native_atomic_producer=dataclasses.replace(
                    adapter.native_atomic_producer,
                    descriptor_sha256=digest,
                    source_sha256=_sha(f"fresh-atomic-source-{index}"),
                ),
            )
    for index, digest in enumerate(EXPECTED_INCOMPATIBLE_ADAPTER_SOURCE_SHA256S):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                adapter,
                publisher=dataclasses.replace(
                    adapter.publisher,
                    descriptor_sha256=_sha(f"fresh-publisher-descriptor-{index}"),
                    source_sha256=digest,
                ),
            )
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                adapter,
                native_atomic_producer=dataclasses.replace(
                    adapter.native_atomic_producer,
                    descriptor_sha256=_sha(f"fresh-atomic-descriptor-{index}"),
                    source_sha256=digest,
                ),
            )


@pytest.mark.parametrize("incompatible_digest", EXPECTED_INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S)
@pytest.mark.parametrize(
    "identity_role",
    (
        "publisher_descriptor",
        "publisher_source",
        "native_descriptor",
        "native_source",
    ),
)
def test_adapter_incompatible_digests_cannot_cross_identity_kinds(
    identity_role: str,
    incompatible_digest: str,
) -> None:
    adapter = _bindings(23)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        if identity_role == "publisher_descriptor":
            dataclasses.replace(
                adapter,
                publisher=dataclasses.replace(
                    adapter.publisher,
                    descriptor_sha256=incompatible_digest,
                ),
            )
        elif identity_role == "publisher_source":
            dataclasses.replace(
                adapter,
                publisher=dataclasses.replace(
                    adapter.publisher,
                    source_sha256=incompatible_digest,
                ),
            )
        elif identity_role == "native_descriptor":
            dataclasses.replace(
                adapter,
                native_atomic_producer=dataclasses.replace(
                    adapter.native_atomic_producer,
                    descriptor_sha256=incompatible_digest,
                ),
            )
        else:
            dataclasses.replace(
                adapter,
                native_atomic_producer=dataclasses.replace(
                    adapter.native_atomic_producer,
                    source_sha256=incompatible_digest,
                ),
            )


def test_family_publisher_metadata_and_atomic_schema_swaps_fail_closed() -> None:
    local = _bindings(0)
    external = _bindings(14)
    adapter = _bindings(23)

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(local, publisher=external.publisher)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(local, publisher_metadata=adapter.publisher_metadata)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(adapter, native_atomic_producer=external.native_atomic_producer)


def test_external_expectations_reject_cross_case_and_commitment_swaps() -> None:
    expected = _bindings(0, label="expected-local")
    other = _bindings(1, label="other-local")

    swapped_case = dataclasses.replace(
        expected,
        case_spine_sha256=other.case_spine_sha256,
    )
    swapped_publisher = dataclasses.replace(expected, publisher=other.publisher)
    swapped_reload = dataclasses.replace(
        expected,
        expected_reload_observation_sha256=other.expected_reload_observation_sha256,
    )
    swapped_inventory = dataclasses.replace(
        expected,
        files=other.files,
        publication_address_sha256=other.publication_address_sha256,
        publication_manifest_file_sha256=other.publication_manifest_file_sha256,
        file_inventory_sha256=other.file_inventory_sha256,
        file_count=other.file_count,
        total_size_bytes=other.total_size_bytes,
    )
    for swapped in (
        swapped_case,
        swapped_publisher,
        swapped_reload,
        swapped_inventory,
    ):
        wrapper = commitment.build_matched_v3_qualification_publication_commitment(swapped)
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            commitment.validate_matched_v3_qualification_publication_commitment_bindings(
                wrapper,
                expected=expected,
            )

    expected_external = _bindings(14, label="expected-external")
    other_external = _bindings(15, label="other-external")
    swapped_native = dataclasses.replace(
        expected_external,
        native_atomic_producer=other_external.native_atomic_producer,
        native_publication_receipt=other_external.native_publication_receipt,
    )
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(swapped_native)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.validate_matched_v3_qualification_publication_commitment_bindings(
            wrapper,
            expected=expected_external,
        )


def test_wrapper_cannot_claim_reload_or_content_work() -> None:
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(_bindings(23))
    for field in (
        "reload_performed_by_wrapper",
        "reload_digest_equality_validated_by_wrapper",
        "content_values_read_by_wrapper",
        "payload_bytes_transported_by_wrapper",
    ):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(wrapper, **{field: True})  # type: ignore[arg-type]

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        value = wrapper.to_dict()
        value["reload_performed_by_wrapper"] = True
        raw = _rebody(value)
        commitment.parse_matched_v3_qualification_publication_commitment(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_direct_dataclass_fields_reject_str_and_int_subclasses() -> None:
    bindings = _bindings(23)
    for field in (
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
        "video_slot_mode",
    ):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                bindings,
                **{field: _StrSubclass(getattr(bindings, field))},  # type: ignore[arg-type]
            )
    for field in (
        "case_ordinal",
        "file_count",
        "total_size_bytes",
        "maximum_total_size_bytes",
    ):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            dataclasses.replace(
                bindings,
                **{field: _IntSubclass(getattr(bindings, field))},  # type: ignore[arg-type]
            )

    wrapper = commitment.build_matched_v3_qualification_publication_commitment(bindings)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            wrapper,
            schema_version=_StrSubclass(EXPECTED_WRAPPER_SCHEMA),
        )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings.files[0],
            name=_StrSubclass(bindings.files[0].name),
        )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings.files[0],
            size_bytes=_IntSubclass(bindings.files[0].size_bytes),
        )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings.publisher,
            descriptor_schema_version=_StrSubclass(bindings.publisher.descriptor_schema_version),
        )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        dataclasses.replace(
            bindings.publisher_metadata,
            schema_version=_StrSubclass(bindings.publisher_metadata.schema_version),
        )


def test_non_string_filename_has_one_controlled_domain_error() -> None:
    bindings = _bindings(23)
    with pytest.raises(
        commitment.ForagerMatchedV3QualificationPublicationCommitmentError,
        match="must be exact text",
    ):
        dataclasses.replace(bindings.files[0], name=7)  # type: ignore[arg-type]


def test_parsed_bool_int_aliases_and_false_to_zero_envelopes_fail_closed() -> None:
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(_bindings(23))
    for field, alias in (
        ("case_ordinal", False),
        ("file_count", True),
        ("total_size_bytes", False),
        ("maximum_total_size_bytes", True),
        ("reload_performed_by_wrapper", 0),
    ):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            value = wrapper.to_dict()
            value[field] = alias
            raw = _rebody(value)
            commitment.parse_matched_v3_qualification_publication_commitment(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )

    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        value = wrapper.to_dict()
        value["files"][0]["size_bytes"] = True
        raw = _rebody(value)
        commitment.parse_matched_v3_qualification_publication_commitment(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )

    for envelope in ("capabilities", "readiness", "authority", "claims"):
        with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
            value = wrapper.to_dict()
            first_key = next(iter(value[envelope]))
            value[envelope][first_key] = 0
            raw = _rebody(value)
            commitment.parse_matched_v3_qualification_publication_commitment(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


@pytest.mark.parametrize("forbidden_key", ["score", "payload_bytes"])
def test_recursive_forbidden_keys_run_before_reconstruction(forbidden_key: str) -> None:
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(_bindings(23))
    with pytest.raises(
        commitment.ForagerMatchedV3QualificationPublicationCommitmentError,
        match="forbidden key",
    ):
        value = wrapper.to_dict()
        value["publisher_metadata"][forbidden_key] = _sha(forbidden_key)
        raw = _rebody(value)
        commitment.parse_matched_v3_qualification_publication_commitment(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_strict_parser_rejects_full_pin_duplicates_floats_and_noncanonical_json() -> None:
    wrapper = commitment.build_matched_v3_qualification_publication_commitment(_bindings(23))
    raw = commitment.canonical_matched_v3_qualification_publication_commitment_bytes(wrapper)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.parse_matched_v3_qualification_publication_commitment(
            raw,
            expected_file_sha256=_sha("wrong-full-file-pin"),
        )

    duplicate = raw.replace(
        b'"schema_version":',
        b'"schema_version":"duplicate","schema_version":',
        1,
    )
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.parse_matched_v3_qualification_publication_commitment(
            duplicate,
            expected_file_sha256=hashlib.sha256(duplicate).hexdigest(),
        )

    float_raw = raw.replace(b'"file_count":5', b'"file_count":5.0', 1)
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.parse_matched_v3_qualification_publication_commitment(
            float_raw,
            expected_file_sha256=hashlib.sha256(float_raw).hexdigest(),
        )

    noncanonical = json.dumps(wrapper.to_dict(), indent=2, sort_keys=True).encode("ascii")
    with pytest.raises(commitment.ForagerMatchedV3QualificationPublicationCommitmentError):
        commitment.parse_matched_v3_qualification_publication_commitment(
            noncanonical,
            expected_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_module_is_stdlib_only_and_has_no_operational_surface() -> None:
    source = inspect.getsource(commitment)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "copy",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "typing",
    }
    forbidden_calls = {
        "Popen",
        "__import__",
        "compile",
        "connect",
        "eval",
        "exec",
        "import_module",
        "open",
        "remove",
        "rename",
        "run",
        "socket",
        "system",
        "unlink",
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_forbidden_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        if node.func.attr in forbidden_calls
        and not (
            node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
        )
    }
    assert not (called_names & forbidden_calls)
    assert not called_forbidden_attributes
    assert "subprocess" not in source
    assert "docker" not in source.lower()
    assert "publication_reload" not in {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
