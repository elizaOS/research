"""Tests for the additive, descriptor-only matched-v3 qualification plan v2."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
from typing import Any, NamedTuple

import pytest

from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v2 as plan

_EXTERNAL_DESCRIPTOR_SHA256 = (
    "926b56b62f992bc12fc4abe2455992ecfc89fa48df92a232aa556b4bf517f04a"
)
_EXTERNAL_IDENTITY_SHA256 = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
_LOCAL_BUNDLE_DESCRIPTOR_SHA256 = (
    "52a48f3258aff9c7f2e80033187b85dd2924dc843d991ba7c2bac829f10c5e89"
)
_CONTEXT_SHA256 = "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330"
_EXECUTION_SHA256 = "38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d"
_PUBLICATION_SHA256 = "28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88"
_IMAGE_ID = "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269"
_EXPECTED_EXTERNAL_MATERIALIZATION_SCHEMA = (
    "alberta.forager_matched_v3_external_materialization.v2"
)
_EXPECTED_LOCAL_SNAPSHOT_MANIFEST_SCHEMA = (
    "alberta.forager_matched_v3.local_source_snapshot_manifest.v1"
)
_EXPECTED_LOCAL_SNAPSHOT_TREE_SCHEMA = (
    "alberta.forager_matched_v3.local_source_snapshot_tree.v1"
)
_EXPECTED_LOCAL_BUNDLE_RECEIPT_SCHEMA = (
    "alberta.forager_matched_v3.local_source_bundle_receipt.v1"
)
_EXPECTED_CANDIDATE_ORDER = (
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


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _rebody(value: dict[str, Any]) -> bytes:
    body = copy.deepcopy(value)
    body.pop("plan_body_sha256")
    value["plan_body_sha256"] = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    return _canonical(value)


class Inputs(NamedTuple):
    external: plan.ExternalSourcePublicationBindingV2
    local: plan.LocalSourceBundleBindingV2
    build: plan.CpuOciBuildPublicationBindingV2
    resources: tuple[plan.CandidateResourceBindingV2, ...]
    publishers: tuple[plan.CandidatePublisherBindingV2, ...]

    @property
    def pins(self) -> dict[str, str]:
        return {
            "expected_context_receipt_sha256": _CONTEXT_SHA256,
            "expected_execution_receipt_sha256": _EXECUTION_SHA256,
            "expected_publication_receipt_sha256": _PUBLICATION_SHA256,
            "expected_image_id": _IMAGE_ID,
        }


def _resource(candidate_id: str) -> plan.CandidateResourceBindingV2:
    ceilings = {field: 1 for field in plan.RESOURCE_CEILING_FIELDS}
    ceilings["max_environment_interactions"] = 499_712
    ceilings["max_attempt_count"] = 2
    ceilings["max_failure_count"] = 1
    return plan.CandidateResourceBindingV2.from_mapping(
        candidate_id=candidate_id,
        ceilings=ceilings,
    )


@pytest.fixture
def inputs() -> Inputs:
    external = plan.ExternalSourcePublicationBindingV2(
        publication_receipt_schema_version=(
            plan.EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
        ),
        publication_receipt_sha256=_sha("external-receipt"),
        publication_receipt_body_sha256=_sha("external-receipt-body"),
        publication_contract_descriptor_sha256=_EXTERNAL_DESCRIPTOR_SHA256,
        materialization_manifest_schema_version=(
            plan.EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        ),
        materialization_manifest_sha256=_sha("external-manifest"),
        materialization_payload_sha256=_sha("external-payload"),
        materialization_identity_sha256=_EXTERNAL_IDENTITY_SHA256,
        source_tree_sha256=_sha("external-tree"),
        staging_manifest_schema_version=plan.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        staging_manifest_sha256=_sha("external-staging-manifest"),
        staging_manifest_body_sha256=_sha("external-staging-body"),
        archive_sha256=_sha("external-archive"),
        archive_size_bytes=20_480,
        archive_member_count=3,
        archive_inventory_sha256=_sha("external-inventory"),
    )
    local = plan.LocalSourceBundleBindingV2(
        bundle_receipt_schema_version=plan.LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION,
        bundle_receipt_sha256=_sha("local-receipt"),
        bundle_receipt_body_sha256=_sha("local-receipt-body"),
        bundle_descriptor_sha256=_LOCAL_BUNDLE_DESCRIPTOR_SHA256,
        snapshot_manifest_schema_version=(
            plan.LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        ),
        snapshot_manifest_sha256=_sha("local-manifest"),
        snapshot_tree_schema_version=plan.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
        snapshot_tree_sha256=_sha("local-tree"),
        archive_sha256=_sha("local-archive"),
        archive_size_bytes=20_480,
        archive_member_count=3,
        member_inventory_sha256=_sha("local-inventory"),
        directory_count=2,
        file_count=3,
        total_size_bytes=17,
    )
    build = plan.CpuOciBuildPublicationBindingV2(
        build_plan_schema_version=plan.CPU_OCI_BUILD_PLAN_SCHEMA_VERSION,
        build_plan_sha256=_sha("build-plan"),
        intent_schema_version=plan.CPU_OCI_BUILD_INTENT_SCHEMA_VERSION,
        intent_sha256=_sha("build-intent"),
        context_receipt_schema_version=plan.CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
        context_receipt_sha256=_CONTEXT_SHA256,
        execution_receipt_schema_version=(
            plan.CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION
        ),
        execution_receipt_sha256=_EXECUTION_SHA256,
        publication_receipt_schema_version=(
            plan.CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION
        ),
        publication_receipt_sha256=_PUBLICATION_SHA256,
        image_id=_IMAGE_ID,
        external_source_receipt_sha256=external.publication_receipt_sha256,
        external_source_archive_sha256=external.archive_sha256,
        external_source_tree_sha256=external.source_tree_sha256,
        local_source_receipt_sha256=local.bundle_receipt_sha256,
        local_source_archive_sha256=local.archive_sha256,
        local_snapshot_manifest_sha256=local.snapshot_manifest_sha256,
        local_snapshot_tree_sha256=local.snapshot_tree_sha256,
    )
    resources = tuple(
        _resource(candidate_id)
        for candidate_id in plan.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    publishers = plan.matched_v3_implemented_publisher_bindings_v2(
        local_source_tree_sha256=local.snapshot_tree_sha256
    )
    return Inputs(external, local, build, resources, publishers)


def _build(inputs: Inputs) -> dict[str, Any]:
    return plan.build_matched_v3_qualification_plan_v2(
        external_source=inputs.external,
        local_source=inputs.local,
        cpu_oci_build=inputs.build,
        resource_bindings=inputs.resources,
        publisher_bindings=inputs.publishers,
        **inputs.pins,
    )


def _bytes(inputs: Inputs) -> tuple[dict[str, Any], bytes, str]:
    value = _build(inputs)
    raw = plan.canonical_matched_v3_qualification_plan_v2_bytes(value, **inputs.pins)
    return value, raw, hashlib.sha256(raw).hexdigest()


def test_build_is_descriptor_only_ordered_and_authority_denying(inputs: Inputs) -> None:
    value = _build(inputs)

    assert value["schema_version"] == plan.QUALIFICATION_PLAN_V2_SCHEMA_VERSION
    assert value["status"] == "contract_implemented_no_production_plan"
    assert plan.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS == _EXPECTED_CANDIDATE_ORDER
    assert value["candidate_order"] == list(_EXPECTED_CANDIDATE_ORDER)
    assert len(value["candidate_requirements"]) == 28
    assert all(claim is False for claim in value["claims"].values())
    assert value["issuance_policy"]["production_plan_issued"] is False
    assert value["runtime_policy"]["runtime_qualified_here"] is False


def test_bindings_use_current_producer_specific_schemas(inputs: Inputs) -> None:
    value = _build(inputs)
    external = value["producer_bindings"]["external_source"]
    local = value["producer_bindings"]["local_source"]

    assert (
        external["materialization_manifest_schema_version"]
        == _EXPECTED_EXTERNAL_MATERIALIZATION_SCHEMA
    )
    assert external["materialization_identity_sha256"] == _EXTERNAL_IDENTITY_SHA256
    assert (
        local["snapshot_manifest_schema_version"]
        == _EXPECTED_LOCAL_SNAPSHOT_MANIFEST_SCHEMA
    )
    assert (
        local["snapshot_tree_schema_version"]
        == _EXPECTED_LOCAL_SNAPSHOT_TREE_SCHEMA
    )
    assert (
        local["bundle_receipt_schema_version"]
        == _EXPECTED_LOCAL_BUNDLE_RECEIPT_SCHEMA
    )


def test_build_identity_has_four_independent_pins(inputs: Inputs) -> None:
    value = _build(inputs)
    build = value["producer_bindings"]["cpu_oci_build"]

    assert build["context_receipt_sha256"] == _CONTEXT_SHA256
    assert build["execution_receipt_sha256"] == _EXECUTION_SHA256
    assert build["publication_receipt_sha256"] == _PUBLICATION_SHA256
    assert build["image_id"] == _IMAGE_ID
    assert len({_CONTEXT_SHA256, _EXECUTION_SHA256, _PUBLICATION_SHA256}) == 3


def test_registry_binds_only_two_real_publishers_and_26_explicit_gaps(
    inputs: Inputs,
) -> None:
    value = _build(inputs)
    registry = value["publisher_registry"]

    assert [item["candidate_id"] for item in registry["implemented_bindings"]] == list(
        plan.IMPLEMENTED_PUBLISHER_CANDIDATE_IDS
    )
    assert registry["missing_candidate_ids"] == list(plan.MISSING_PUBLISHER_CANDIDATE_IDS)
    assert registry["implemented_count"] == 2
    assert registry["missing_count"] == 26
    assert registry["complete"] is False
    assert registry["synthetic_bindings_allowed"] is False
    assert registry["incomplete_bindings_allowed"] is False
    status_by_id = {
        item["candidate_id"]: item["publisher_requirement"]["status"]
        for item in value["candidate_requirements"]
    }
    assert sum(status == "implemented_strict_publisher" for status in status_by_id.values()) == 2
    assert sum(status == "required_not_implemented" for status in status_by_id.values()) == 26


def test_canonical_roundtrip_requires_plan_and_all_build_pins(inputs: Inputs) -> None:
    value, raw, digest = _bytes(inputs)

    replayed = plan.replay_matched_v3_qualification_plan_v2(
        raw,
        expected_plan_sha256=digest,
        **inputs.pins,
    )
    parsed = plan.parse_matched_v3_qualification_plan_v2(
        raw,
        expected_plan_sha256=digest,
        **inputs.pins,
    )

    assert replayed == value
    assert parsed == value
    assert replayed is not value
    assert raw.endswith(b"\n")
    assert _canonical(replayed) == raw


@pytest.mark.parametrize(
    ("pin_name", "replacement"),
    [
        ("expected_context_receipt_sha256", _sha("wrong-context")),
        ("expected_execution_receipt_sha256", _sha("wrong-execution")),
        ("expected_publication_receipt_sha256", _sha("wrong-publication")),
        ("expected_image_id", "sha256:" + _sha("wrong-image")),
    ],
)
def test_replay_rejects_each_independent_build_pin(
    inputs: Inputs,
    pin_name: str,
    replacement: str,
) -> None:
    _, raw, digest = _bytes(inputs)
    pins = dict(inputs.pins)
    pins[pin_name] = replacement

    with pytest.raises(
        plan.ForagerMatchedV3QualificationPlanV2Error,
        match="caller pin differs",
    ):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=digest,
            **pins,
        )


def test_replay_rejects_wrong_full_file_pin_and_noncanonical_bytes(inputs: Inputs) -> None:
    _, raw, digest = _bytes(inputs)

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="full-file"):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=_sha("wrong-plan"),
            **inputs.pins,
        )
    noncanonical = b" " + raw
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="not exact canonical"):
        plan.replay_matched_v3_qualification_plan_v2(
            noncanonical,
            expected_plan_sha256=hashlib.sha256(noncanonical).hexdigest(),
            **inputs.pins,
        )
    assert digest == hashlib.sha256(raw).hexdigest()


def test_parser_rejects_duplicate_keys_and_floats_before_validation(inputs: Inputs) -> None:
    _, raw, _ = _bytes(inputs)
    duplicate = raw.replace(
        b'{"acceptance_policy":',
        b'{"schema_version":"duplicate","acceptance_policy":',
        1,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="duplicate key"):
        plan.replay_matched_v3_qualification_plan_v2(
            duplicate,
            expected_plan_sha256=hashlib.sha256(duplicate).hexdigest(),
            **inputs.pins,
        )
    floating = raw.replace(b"499712", b"1.0", 1)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="forbidden float"):
        plan.replay_matched_v3_qualification_plan_v2(
            floating,
            expected_plan_sha256=hashlib.sha256(floating).hexdigest(),
            **inputs.pins,
        )


def test_parser_rejects_excessive_json_depth_with_contract_error(inputs: Inputs) -> None:
    deeply_nested = b'{"x":' * 100 + b"0" + b"}" * 100 + b"\n"

    with pytest.raises(
        plan.ForagerMatchedV3QualificationPlanV2Error,
        match="depth limit",
    ):
        plan.replay_matched_v3_qualification_plan_v2(
            deeply_nested,
            expected_plan_sha256=hashlib.sha256(deeply_nested).hexdigest(),
            **inputs.pins,
        )


def test_parser_wraps_json_decoder_recursion_error(
    inputs: Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"{}\n"

    def raise_recursion_error(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RecursionError("injected decoder recursion")

    monkeypatch.setattr(json, "loads", raise_recursion_error)
    with pytest.raises(
        plan.ForagerMatchedV3QualificationPlanV2Error,
        match="not strict ASCII JSON",
    ):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("candidate_requirements", 0, "ordinal"), False),
        (("candidate_requirements", 1, "ordinal"), True),
        (("acceptance_policy", "source_membership_exact"), 1),
        (
            (
                "candidate_requirements",
                0,
                "acceptance_policy",
                "source_membership_exact",
            ),
            1,
        ),
        (("failure_policy", "fail_closed"), 1),
        (("runtime_policy", "runtime_qualified_here"), 0),
        (("issuance_policy", "production_plan_issued"), 0),
        (("claims", "qualification_granted"), 0),
    ],
)
def test_recomputed_body_rejects_bool_integer_type_substitution(
    inputs: Inputs,
    path: tuple[str | int, ...],
    replacement: bool | int,
) -> None:
    value = _build(inputs)
    target: Any = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement
    raw = _rebody(value)

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )


def test_recomputed_body_cannot_enable_authority(inputs: Inputs) -> None:
    value = _build(inputs)
    value["claims"]["qualification_granted"] = True
    raw = _rebody(value)

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="claim became true"):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )


def test_recomputed_body_cannot_reorder_candidates_or_resources(inputs: Inputs) -> None:
    value = _build(inputs)
    value["candidate_order"][0], value["candidate_order"][1] = (
        value["candidate_order"][1],
        value["candidate_order"][0],
    )
    raw = _rebody(value)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="candidate order"):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )

    value = _build(inputs)
    requirements = value["resource_contract"]["requirements"]
    requirements[0], requirements[1] = requirements[1], requirements[0]
    raw = _rebody(value)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="order differs"):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )


def test_recomputed_body_cannot_fill_gap_with_synthetic_publisher(inputs: Inputs) -> None:
    value = _build(inputs)
    fake = copy.deepcopy(value["publisher_registry"]["implemented_bindings"][0])
    fake["candidate_id"] = plan.MISSING_PUBLISHER_CANDIDATE_IDS[0]
    fake["synthetic"] = True
    value["publisher_registry"]["implemented_bindings"].append(fake)
    value["publisher_registry"]["implemented_count"] = 3
    value["publisher_registry"]["missing_count"] = 25
    raw = _rebody(value)

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error):
        plan.replay_matched_v3_qualification_plan_v2(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            **inputs.pins,
        )


def test_publisher_type_rejects_synthetic_incomplete_and_unknown(inputs: Inputs) -> None:
    publisher = inputs.publishers[0]

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="complete real"):
        dataclasses.replace(publisher, synthetic=True)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="complete real"):
        dataclasses.replace(publisher, implementation_complete=False)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="no implemented"):
        dataclasses.replace(publisher, candidate_id=plan.MISSING_PUBLISHER_CANDIDATE_IDS[0])


def test_builder_rejects_partial_or_reordered_publisher_coverage(inputs: Inputs) -> None:
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="coverage or order"):
        plan.build_matched_v3_qualification_plan_v2(
            external_source=inputs.external,
            local_source=inputs.local,
            cpu_oci_build=inputs.build,
            resource_bindings=inputs.resources,
            publisher_bindings=inputs.publishers[:1],
            **inputs.pins,
        )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="coverage or order"):
        plan.build_matched_v3_qualification_plan_v2(
            external_source=inputs.external,
            local_source=inputs.local,
            cpu_oci_build=inputs.build,
            resource_bindings=inputs.resources,
            publisher_bindings=tuple(reversed(inputs.publishers)),
            **inputs.pins,
        )


def test_builder_rejects_resource_order_aliases_and_boolean_ceilings(inputs: Inputs) -> None:
    reordered = list(inputs.resources)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="coverage or order"):
        plan.build_matched_v3_qualification_plan_v2(
            external_source=inputs.external,
            local_source=inputs.local,
            cpu_oci_build=inputs.build,
            resource_bindings=reordered,
            publisher_bindings=inputs.publishers,
            **inputs.pins,
        )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="aliased"):
        plan.build_matched_v3_qualification_plan_v2(
            external_source=inputs.external,
            local_source=inputs.local,
            cpu_oci_build=inputs.build,
            resource_bindings=(inputs.resources[0],) * 28,
            publisher_bindings=inputs.publishers,
            **inputs.pins,
        )
    ceilings = {field: 1 for field in plan.RESOURCE_CEILING_FIELDS}
    ceilings["max_environment_interactions"] = True
    ceilings["max_attempt_count"] = 2
    ceilings["max_failure_count"] = 1
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="exact integer"):
        plan.CandidateResourceBindingV2.from_mapping(
            candidate_id=plan.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[0],
            ceilings=ceilings,
        )


def test_external_binding_rejects_v1_schema_and_stale_identity(inputs: Inputs) -> None:
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="schema"):
        dataclasses.replace(
            inputs.external,
            materialization_manifest_schema_version=(
                "alberta.forager_matched_v3_external_materialization.v1"
            ),
        )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="identity"):
        dataclasses.replace(
            inputs.external,
            materialization_identity_sha256=(
                "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
            ),
        )


def test_local_binding_rejects_old_generic_snapshot_schema(inputs: Inputs) -> None:
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="schema"):
        dataclasses.replace(
            inputs.local,
            snapshot_manifest_schema_version=(
                "alberta.forager_matched_v3.local_source_snapshot.v1"
            ),
        )


def test_builder_rejects_cross_wired_build_source_linkage(inputs: Inputs) -> None:
    cross_wired = dataclasses.replace(
        inputs.build,
        local_snapshot_tree_sha256=_sha("different-local-tree"),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV2Error, match="cross-wired"):
        plan.build_matched_v3_qualification_plan_v2(
            external_source=inputs.external,
            local_source=inputs.local,
            cpu_oci_build=cross_wired,
            resource_bindings=inputs.resources,
            publisher_bindings=inputs.publishers,
            **inputs.pins,
        )


def test_builder_has_no_defaults_or_implicit_production_inputs() -> None:
    signature = inspect.signature(plan.build_matched_v3_qualification_plan_v2)

    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert "production" not in signature.parameters
