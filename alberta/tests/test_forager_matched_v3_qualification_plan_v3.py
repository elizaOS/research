"""Static contract tests for the additive matched-v3 qualification plan v3."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v3 as plan

_EXPECTED_DESCRIPTOR_BODY_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
_EXPECTED_DESCRIPTOR_FILE_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
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
_EXPECTED_GAP_IDS = (
    "fresh_source_closure",
    "sealed_staging",
    "fresh_cpu_oci_build",
    "runtime_quicknet_verifier",
    "external_preacceptance_seed_chronology",
    "production_host_executor",
    "full_resource_merger",
    "observation_registry_v2",
    "separate_plan_issuer",
    "separate_acceptance_evaluator",
)
_EXPECTED_READINESS_IDS = (
    "acceptance_evaluator_available",
    "execution_ready",
    "fresh_build_available",
    "full_resource_merger_available",
    "observation_registry_v2_available",
    "plan_issuer_available",
    "production_host_executor_available",
    "publisher_registry_ready",
    "qualification_ready",
    "runtime_quicknet_verifier_available",
    "seed_chronology_acceptor_available",
    "structural_inputs_sufficient",
)
_EXPECTED_CLAIM_IDS = (
    "authority_granted",
    "build_qualified",
    "evidence_authority",
    "executed_bytecode_attested",
    "execution_authorized",
    "performance_claim_allowed",
    "production_plan_issued",
    "promotion_allowed",
    "publication_authority_granted",
    "publisher_registry_complete",
    "qualification_granted",
    "resource_matched",
    "runtime_qualified",
    "scientific_evidence_created",
    "seed_chronology_accepted",
    "source_qualified",
    "staging_qualified",
    "universal_sota_claim_allowed",
)
_FRESH_IMAGE_ID = (
    "sha256:93562b7037a45a69b4ac6bb67c8f7e06d21a13f6e05a5374eb10bce68f30f2b5"
)
_FRESH_LINEAGE = (
    "bbbd7186f054319380965bfed6c5db1bc7f196e50413c0c55a93797fcb2049e6",
    "01b85897159115991b3aa81239fd92eb272730459f166533d7e54fb0977e3f00",
    "fd5fa27da422667d48ac46764da6c194f50cc29651c7ffb589d21ff93be61078",
)


class _StringAlias(str):
    pass


class _BytesAlias(bytes):
    pass


class _DictAlias(dict[str, bool]):
    pass


class _ListAlias(list[Any]):
    pass


class _TupleAlias(tuple[Any, ...]):
    pass


class _EquivalentStringKey:
    def __init__(self, value: str) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return other == self.value


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
    value["plan_body_sha256"] = hashlib.sha256(
        _canonical(body, newline=False)
    ).hexdigest()
    return _canonical(value)


@pytest.fixture
def finalized_descriptor_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise finalized-only APIs without installing repository pin literals."""

    body = plan.canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes()
    file = plan.canonical_matched_v3_qualification_plan_v3_descriptor_bytes()
    assert file == body + b"\n"
    monkeypatch.setattr(
        plan,
        "QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256",
        hashlib.sha256(body).hexdigest(),
    )
    monkeypatch.setattr(
        plan,
        "QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256",
        hashlib.sha256(file).hexdigest(),
    )


def _build(
    *,
    closures: tuple[plan.PublisherClosureIdentityV3, ...] | None = None,
) -> dict[str, Any]:
    if closures is None:
        closures = (
            plan.matched_v3_local_publisher_closure_v3(),
            plan.matched_v3_external_publisher_closure_v3(),
        )
    return plan.build_matched_v3_qualification_plan_v3(
        structural_inputs=plan.synthetically_complete_matched_v3_structural_inputs_v3(),
        publisher_closures=closures,
        proposed_image_id=_FRESH_IMAGE_ID,
        proposed_build_lineage_components=_FRESH_LINEAGE,
    )


def _replay_mutation(value: dict[str, Any]) -> None:
    raw = _rebody(value)
    plan.replay_matched_v3_qualification_plan_v3(
        raw,
        expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    target: dict[str, Any] = value
    for key in path[:-1]:
        target = cast(dict[str, Any], target[key])
    target[path[-1]] = replacement


def test_zero_descriptor_pins_are_lazy_and_all_pin_bearing_apis_fail_closed() -> None:
    descriptor = plan.matched_v3_qualification_plan_v3_descriptor()
    body = plan.canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes()
    file = plan.canonical_matched_v3_qualification_plan_v3_descriptor_bytes()

    assert type(descriptor) is dict
    assert body and not body.endswith(b"\n")
    assert file == body + b"\n"
    assert plan.QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256 == (
        _EXPECTED_DESCRIPTOR_BODY_SHA256
    )
    assert plan.QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_FILE_SHA256
    )
    assert hashlib.sha256(body).hexdigest() != _EXPECTED_DESCRIPTOR_BODY_SHA256
    assert hashlib.sha256(file).hexdigest() != _EXPECTED_DESCRIPTOR_FILE_SHA256

    structural = plan.synthetically_complete_matched_v3_structural_inputs_v3()
    closures = (
        plan.matched_v3_local_publisher_closure_v3(),
        plan.matched_v3_external_publisher_closure_v3(),
    )
    expected_empty_file = hashlib.sha256(b"{}\n").hexdigest()
    pin_bearing_calls = (
        plan.matched_v3_qualification_plan_v3_descriptor_body_sha256,
        plan.matched_v3_qualification_plan_v3_descriptor_sha256,
        lambda: plan.parse_matched_v3_qualification_plan_v3_descriptor(file),
        lambda: plan.build_matched_v3_qualification_plan_v3(
            structural_inputs=structural,
            publisher_closures=closures,
            proposed_image_id=_FRESH_IMAGE_ID,
            proposed_build_lineage_components=_FRESH_LINEAGE,
        ),
        lambda: plan.canonical_matched_v3_qualification_plan_v3_bytes({}),
        lambda: plan.replay_matched_v3_qualification_plan_v3(
            b"{}\n",
            expected_plan_sha256=expected_empty_file,
        ),
        lambda: plan.parse_matched_v3_qualification_plan_v3(
            b"{}\n",
            expected_plan_sha256=expected_empty_file,
        ),
    )
    for call in pin_bearing_calls:
        with pytest.raises(
            plan.ForagerMatchedV3QualificationPlanV3Error,
            match="pins are not finalized",
        ):
            call()


def test_descriptor_identity_order_and_family_membership_are_exact(
    finalized_descriptor_pins: None,
) -> None:
    descriptor = plan.matched_v3_qualification_plan_v3_descriptor()
    body = plan.canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes()
    raw = plan.canonical_matched_v3_qualification_plan_v3_descriptor_bytes()
    universe = descriptor["universe"]

    assert plan.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS == _EXPECTED_CANDIDATE_ORDER
    assert universe["candidate_order"] == list(_EXPECTED_CANDIDATE_ORDER)
    assert tuple(_EXPECTED_CANDIDATE_ORDER[:14]) == plan.MATCHED_V3_LOCAL_CANDIDATE_IDS
    assert tuple(_EXPECTED_CANDIDATE_ORDER[14:23]) == (
        plan.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    )
    assert tuple(_EXPECTED_CANDIDATE_ORDER[23:25]) == plan.MATCHED_V3_ADAPTER_CANDIDATE_IDS
    assert tuple(_EXPECTED_CANDIDATE_ORDER[25:]) == (
        plan.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
    )
    assert universe["partitions"] == {
        "local": list(plan.MATCHED_V3_LOCAL_CANDIDATE_IDS),
        "external": list(plan.MATCHED_V3_EXTERNAL_CANDIDATE_IDS),
        "adapter": list(plan.MATCHED_V3_ADAPTER_CANDIDATE_IDS),
    }
    assert universe["partition_counts"] == {"local": 14, "external": 12, "adapter": 2}
    assert universe["candidate_count"] == 28
    assert len(set(_EXPECTED_CANDIDATE_ORDER)) == 28
    assert raw == body + b"\n"
    assert plan.matched_v3_qualification_plan_v3_descriptor_body_sha256() == (
        hashlib.sha256(body).hexdigest()
    )
    assert plan.matched_v3_qualification_plan_v3_descriptor_sha256() == (
        hashlib.sha256(raw).hexdigest()
    )
    assert plan.parse_matched_v3_qualification_plan_v3_descriptor(raw) == descriptor
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.parse_matched_v3_qualification_plan_v3_descriptor(body)


def test_descriptor_body_and_file_pins_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = plan.canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes()
    file = plan.canonical_matched_v3_qualification_plan_v3_descriptor_bytes()
    body_sha256 = hashlib.sha256(body).hexdigest()
    file_sha256 = hashlib.sha256(file).hexdigest()

    for body_pin, file_pin in (
        (_EXPECTED_DESCRIPTOR_BODY_SHA256, file_sha256),
        (body_sha256, _EXPECTED_DESCRIPTOR_FILE_SHA256),
        (file_sha256, file_sha256),
        (body_sha256, body_sha256),
        (file_sha256, body_sha256),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                plan,
                "QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256",
                body_pin,
            )
            patch.setattr(
                plan,
                "QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256",
                file_pin,
            )
            with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
                plan.matched_v3_qualification_plan_v3_descriptor_sha256()


def test_descriptor_is_additive_and_permanently_nonauthorizing() -> None:
    descriptor = plan.matched_v3_qualification_plan_v3_descriptor()

    assert descriptor["versioning"] == {
        "additive": True,
        "plan_v2_mutated": False,
        "plan_v2_accepted_as_v3": False,
        "frozen_outputs_modified": False,
    }
    assert tuple(descriptor["claims"]) == _EXPECTED_CLAIM_IDS
    assert tuple(descriptor["readiness"]) == _EXPECTED_READINESS_IDS
    assert len(descriptor["claims"]) == 18
    assert len(descriptor["readiness"]) == 12
    assert all(value is False for value in descriptor["claims"].values())
    assert all(value is False for value in descriptor["readiness"].values())
    required_gaps = descriptor["required_gap_ledger"]
    assert [item["gap_id"] for item in required_gaps] == list(_EXPECTED_GAP_IDS)
    assert len(required_gaps) == 10
    assert all(item["satisfied"] is False for item in required_gaps)
    assert all(item["accepted_artifact_sha256"] is None for item in required_gaps)
    assert all(
        item["caller_structural_presence_can_satisfy"] is False
        for item in required_gaps
    )
    assert descriptor["publisher_registry"]["strict_qualifying_count"] == 0
    assert descriptor["publisher_registry"]["complete"] is False


def test_gap_requirements_are_immutable_and_descriptor_results_are_detached(
    finalized_descriptor_pins: None,
) -> None:
    gap_id = _EXPECTED_GAP_IDS[0]
    expected_requirement = (
        plan.matched_v3_qualification_plan_v3_descriptor()["required_gap_ledger"][0][
            "required_future_artifact"
        ]
    )

    with pytest.raises(TypeError):
        cast(dict[str, str], plan._GAP_REQUIREMENTS)[gap_id] = "mutated requirement"

    detached = plan.matched_v3_qualification_plan_v3_descriptor()
    detached["required_gap_ledger"][0][
        "required_future_artifact"
    ] = "detached mutation"

    rebuilt_descriptor = plan.matched_v3_qualification_plan_v3_descriptor()
    assert rebuilt_descriptor["required_gap_ledger"][0] == {
        "gap_id": gap_id,
        "required_future_artifact": expected_requirement,
        "satisfied": False,
        "accepted_artifact_sha256": None,
        "caller_structural_presence_can_satisfy": False,
    }

    value = _build()
    assert value["blocker_ledger"][0]["required_future_artifact"] == expected_requirement
    raw = plan.canonical_matched_v3_qualification_plan_v3_bytes(value)
    assert plan.replay_matched_v3_qualification_plan_v3(
        raw,
        expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
    ) == value


def test_exact_local_and_external_closures_are_content_only_and_one_way() -> None:
    local = plan.matched_v3_local_publisher_closure_v3()
    external = plan.matched_v3_external_publisher_closure_v3()

    assert local.family == "local"
    assert local.candidate_ids == plan.MATCHED_V3_LOCAL_CANDIDATE_IDS
    assert local.trust_direction.endswith("no_cycle")
    assert [
        (item.role, item.descriptor_sha256, item.implementation_source_sha256)
        for item in local.components
    ] == [
        (
            "local_reward_bundle",
            "f1fb7d28f0508c38b0d53173707ea5cb006b669793d3401091a942874ee3b878",
            "93e824e2518ce405f457329d7c2aa77ddc0fd140d157d155f04f4a9342e0eb9f",
        ),
        (
            "local_reward_publisher",
            "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f",
            "48640a7e352383eac58fed24c8c36c77fcf3bbed8baf78ce663394d1f7e90200",
        ),
    ]
    assert external.family == "external"
    assert external.candidate_ids == plan.MATCHED_V3_EXTERNAL_CANDIDATE_IDS
    assert external.trust_direction.endswith("no_reverse_source_cycle")
    assert [item.role for item in external.components] == [
        "external_execution_contract",
        "external_execution_runner",
        "external_outcome_consumer",
        "external_reward_publisher",
    ]
    assert [
        (item.descriptor_sha256, item.implementation_source_sha256)
        for item in external.components
    ] == [
        (
            "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4",
            "b53381a21f47fd488e79f97630211c2e90ab43faf7775fb8d8ed5cbebcff76d2",
        ),
        (
            "0f0c12a93f458ded1188185fed8c0c97e5763f5efa5151f84b70f28b2c945636",
            "7ae6a28674076e3e8c0d862d13fc900e7a7c868ef1fa4cb3da333cca35dcc0d7",
        ),
        (
            "7c7d007f29b55d6e4a72467d72c4b793568847930d7eb0c17cc276b027e74ceb",
            "f3fba30c37500b73250992cdcef459fb9814aafce056af301e31a2f066a1ab3a",
        ),
        (
            "59d470d6c31e1d3dce8eded401e6331994ca007b94524d8e00714c1f2c66f30b",
            "645d232134b220f57b466d3f9c3e140ace8bad3835d9ed290fc066a3c257a80c",
        ),
    ]
    all_components = (*local.components, *external.components)
    assert all(item.implementation_path.endswith(".py") for item in all_components)
    assert all(
        "qualification_plan_v3" not in item.implementation_path
        for item in all_components
    )
    assert local.to_dict()["qualification_ready"] is False
    assert local.to_dict()["recognition_disposition"] == (
        "implementation_identity_only_nonqualifying"
    )
    assert external.to_dict()["qualification_ready"] is False


@pytest.mark.parametrize(
    ("closures", "expected_count"),
    [
        ((), 0),
        ((plan.matched_v3_local_publisher_closure_v3(),), 14),
        ((plan.matched_v3_external_publisher_closure_v3(),), 12),
        (
            (
                plan.matched_v3_local_publisher_closure_v3(),
                plan.matched_v3_external_publisher_closure_v3(),
            ),
            26,
        ),
    ],
)
def test_exact_closure_recognition_counts_remain_nonqualifying(
    closures: tuple[plan.PublisherClosureIdentityV3, ...],
    expected_count: int,
    finalized_descriptor_pins: None,
) -> None:
    value = _build(closures=closures)
    registry = value["publisher_registry"]

    assert registry["recognized_nonqualifying_count"] == expected_count
    assert registry["qualifying_count"] == 0
    assert registry["complete"] is False
    assert registry["missing_strict_adapter_count"] == 2
    assert registry["missing_strict_adapter_candidate_ids"] == [
        "adapted_full_rainbow",
        "adapted_ppo_gru",
    ]
    assert all(item["qualification_ready"] is False for item in registry["candidate_entries"])


def test_all_true_structural_inputs_satisfy_no_gap_or_claim(
    finalized_descriptor_pins: None,
) -> None:
    value = _build()
    registry = value["publisher_registry"]

    assert value["descriptor_binding"] == {
        "schema_version": plan.QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
        "body_sha256": plan.QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256,
        "file_sha256": plan.QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256,
    }
    assert value["candidate_order"] == list(_EXPECTED_CANDIDATE_ORDER)
    assert value["partition_counts"] == {"local": 14, "external": 12, "adapter": 2}

    assert set(value["structural_inputs"].values()) == {True}
    assert [item["gap_id"] for item in value["blocker_ledger"]] == list(
        _EXPECTED_GAP_IDS
    )
    assert all(
        item["caller_reports_structural_presence"] is True
        for item in value["blocker_ledger"]
    )
    assert all(item["satisfied"] is False for item in value["blocker_ledger"])
    assert all(
        item["accepted_artifact_sha256"] is None
        for item in value["blocker_ledger"]
    )
    assert all(
        item["caller_structural_presence_can_satisfy"] is False
        for item in value["blocker_ledger"]
    )
    assert registry["recognized_nonqualifying_count"] == 26
    assert registry["qualifying_count"] == 0
    assert [item["family"] for item in registry["recognized_closures"]] == [
        "local",
        "external",
    ]
    assert all(
        item["publisher_status"]
        == "recognized_nonqualifying_publisher_implementation"
        for item in registry["candidate_entries"][:23]
    )
    assert all(
        item["publisher_status"]
        == "strict_adapter_publisher_required_not_implemented"
        for item in registry["candidate_entries"][23:25]
    )
    assert all(
        item["publisher_status"]
        == "recognized_nonqualifying_publisher_implementation"
        for item in registry["candidate_entries"][25:]
    )
    expected_families = (
        ("local",) * 14
        + ("external",) * 9
        + ("adapter",) * 2
        + ("external",) * 3
    )
    assert [
        (item["ordinal"], item["candidate_id"], item["family"])
        for item in registry["candidate_entries"]
    ] == [
        (ordinal, candidate_id, expected_families[ordinal])
        for ordinal, candidate_id in enumerate(_EXPECTED_CANDIDATE_ORDER)
    ]
    assert len(registry["candidate_entries"]) == 28
    assert tuple(value["claims"]) == _EXPECTED_CLAIM_IDS
    assert tuple(value["readiness"]) == _EXPECTED_READINESS_IDS
    assert all(value is False for value in value["claims"].values())
    assert all(value is False for value in value["readiness"].values())
    assert value["future_authority_boundary"] == {
        "separate_plan_issuer_required": True,
        "separate_acceptance_evaluator_required": True,
        "implemented_here": False,
    }
    assert value["proposed_lineage"]["accepted_as_fresh_build"] is False
    assert value["proposed_lineage"]["qualifies_runtime"] is False


def test_observation_host_runtime_seed_and_resource_surfaces_remain_gaps() -> None:
    gaps = plan.matched_v3_qualification_plan_v3_descriptor()[
        "observation_and_execution_gaps"
    ]

    assert gaps["observation_registry_v1"] == {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v1"
        ),
        "descriptor_sha256": (
            "f28d01ae9750ee5989f613dbdc64b91f8a8a500faa460b9b5a8c89aa59b31c09"
        ),
        "status": "implemented_structural_validators_no_observation_issuer",
        "v2_compatible": False,
        "v3_compatible": False,
        "explicit_blocker": True,
        "may_fill_observation_registry_v2_gap": False,
    }
    assert gaps["observation_registry_v2"] == {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v2"
        ),
        "candidate_batch_schema_version": (
            "alberta.forager_matched_v3.qualification_observation_candidate_batch.v2"
        ),
        "implementation_path": (
            "alberta_framework/benchmarks/"
            "forager_matched_v3_qualification_observations_v2.py"
        ),
        "descriptor_body_sha256": None,
        "descriptor_file_sha256": None,
        "implementation_source_sha256": None,
        "structural_candidate_validator_implemented": True,
        "descriptor_finalized": False,
        "source_identity_finalized": False,
        "observation_issuer_available": False,
        "acceptance_evaluator_available": False,
        "production_registry_available": False,
        "gap_satisfied": False,
    }
    host_v1 = gaps["host_executor"]["incompatible_v1"]
    assert host_v1["schema_version"] == (
        "alberta.forager_matched_v3.host_qualification_executor_descriptor.v1"
    )
    assert host_v1["implementation_path"] == (
        "alberta_framework/benchmarks/forager_matched_v3_host_qualification_executor.py"
    )
    assert host_v1["descriptor_sha256"] == (
        "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
    )
    assert host_v1["implementation_source_sha256"] == (
        "d8bbc666a49e252662807f256c7f212c9a7c8c3be279b928a6a93ed77532a2e1"
    )
    assert host_v1["v2_compatible"] is False
    assert host_v1["permanently_excluded_from_v2_slot"] is True
    assert host_v1["may_fill_production_executor_gap"] is False

    host_v2 = gaps["host_executor"]["source_only_v2"]
    assert host_v2["schema_version"] == (
        "alberta.forager_matched_v3.host_qualification_executor_descriptor.v2"
    )
    assert host_v2["implementation_path"] == (
        "alberta_framework/benchmarks/forager_matched_v3_host_qualification_executor_v2.py"
    )
    assert host_v2["source_contract_implemented"] is True
    assert host_v2["descriptor_body_sha256"] is None
    assert host_v2["descriptor_file_sha256"] is None
    assert host_v2["implementation_source_sha256"] is None
    assert host_v2["descriptor_finalized"] is False
    assert host_v2["source_identity_finalized"] is False
    assert host_v2["production_backend_available"] is False
    assert host_v2["production_executor_available"] is False
    assert host_v2["may_self_issue"] is False
    assert host_v2["may_self_evaluate"] is False
    assert host_v2["gap_satisfied"] is False
    assert gaps["quicknet"]["runtime_verifier_available"] is False
    assert gaps["quicknet"]["runtime_verification_accepted"] is False
    assert gaps["quicknet"]["source_descriptor_sha256"] == (
        "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
    )
    assert gaps["seed_registry"]["descriptor_sha256"] == (
        "fba1ab637f72de87c926169f2e0df5e66a8a2c7dcf855f00442a33dbe42fbef2"
    )
    assert gaps["seed_registry"]["external_preacceptance_chronology_accepted"] is False
    assert gaps["seed_registry"]["seed_issuer_available"] is False
    assert gaps["resource_observation"]["endpoint_observer_descriptor_sha256"] == (
        "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
    )
    assert gaps["resource_observation"]["endpoint_observer_source_sha256"] == (
        "4d34951ccb4b265caa29794457cdd8a5dd837ecf4b73b7a44e4f849bf8c8106e"
    )
    assert (
        gaps["resource_observation"][
            "endpoint_observer_counts_as_full_28_field_merger"
        ]
        is False
    )
    assert gaps["resource_observation"]["full_resource_merger_available"] is False
    merger = gaps["full_resource_merger"]
    assert merger["descriptor_schema_version"] == (
        "alberta.forager_matched_v3.full_resource_merger_descriptor.v1"
    )
    assert merger["receipt_schema_version"] == (
        "alberta.forager_matched_v3.full_resource_merger_receipt.v1"
    )
    assert merger["structural_candidate_schema_version"] == (
        "alberta.forager_matched_v3.qualification_resource_merger_candidate.v2"
    )
    assert merger["structural_candidate_validator_path"] == (
        "alberta_framework/benchmarks/forager_matched_v3_qualification_observations_v2.py"
    )
    assert merger["structural_candidate_validator_implemented"] is True
    assert merger["implementation_path"] is None
    assert merger["descriptor_body_sha256"] is None
    assert merger["descriptor_file_sha256"] is None
    assert merger["implementation_source_sha256"] is None
    assert merger["descriptor_finalized"] is False
    assert merger["source_identity_finalized"] is False
    assert merger["production_merger_implemented"] is False
    assert merger["production_receipt_available"] is False
    assert merger["endpoint_observer_can_substitute"] is False
    assert merger["finalized_dependency_identity_can_substitute"] is False
    assert merger["gap_satisfied"] is False
    assert gaps["future_separation"]["plan_issuer_must_be_separate"] is True
    assert gaps["future_separation"]["acceptance_evaluator_must_be_separate"] is True
    assert gaps["future_separation"]["issuer_available"] is False
    assert gaps["future_separation"]["evaluator_available"] is False


def test_finalized_source_only_dependencies_and_retired_union_are_exact() -> None:
    gaps = plan.matched_v3_qualification_plan_v3_descriptor()[
        "observation_and_execution_gaps"
    ]
    dependencies = gaps["finalized_source_only_dependencies"]
    expected_pairs = {
        "algorithmic_resource_contract": (
            "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d",
            "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3",
        ),
        "storage_boundary_contract": (
            "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c",
            "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf",
        ),
        "normalized_publication_contract": (
            "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420",
            "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80",
        ),
    }
    assert set(dependencies) == set(expected_pairs)
    expected_statuses = {
        "algorithmic_resource_contract": (
            "implemented_source_only_contract_uninvoked_no_production_receipt",
            "score_blind_metadata_only_algorithmic_resource_contract_non_authorizing",
        ),
        "storage_boundary_contract": (
            "implemented_source_only_contract_uninvoked_no_production_receipt",
            "score_blind_metadata_only_storage_boundary_contract_non_authorizing",
        ),
        "normalized_publication_contract": (
            "implemented_source_only_expected_reload_commitment_non_authorizing",
            "score_blind_metadata_only_normalized_commitment_non_authorizing",
        ),
    }
    expected_metadata = {
        "algorithmic_resource_contract": (
            "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1",
            "alberta_framework/benchmarks/forager_matched_v3_algorithmic_resource_contract.py",
        ),
        "storage_boundary_contract": (
            "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1",
            "alberta_framework/benchmarks/forager_matched_v3_qualification_storage_boundary.py",
        ),
        "normalized_publication_contract": (
            (
                "alberta.forager_matched_v3."
                "qualification_publication_commitment_contract_descriptor.v1"
            ),
            (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_qualification_publication_commitment.py"
            ),
        ),
    }
    for name, pair in expected_pairs.items():
        dependency = dependencies[name]
        assert (
            dependency["descriptor_sha256"],
            dependency["implementation_source_sha256"],
        ) == pair
        assert (dependency["status"], dependency["classification"]) == (
            expected_statuses[name]
        )
        assert (dependency["schema_version"], dependency["implementation_path"]) == (
            expected_metadata[name]
        )
        assert dependency["recognition"] == "exact_finalized_source_only_identity"
        assert dependency["source_contract_implemented"] is True
        assert dependency["production_implementation_available"] is False
        assert dependency["production_receipt_available"] is False
        assert dependency["qualification_ready"] is False
        assert dependency["non_authorizing"] is True
        assert dependency["fills_any_structural_gap"] is False

    exclusions = gaps["cross_kind_identity_exclusions"]
    assert exclusions["algorithmic_resource_validator_identity_sha256_union"] == [
        *expected_pairs["algorithmic_resource_contract"],
        "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733",
        "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8",
    ]
    assert exclusions["retired_algorithmic_resource_validator"] == {
        "descriptor_sha256": (
            "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733"
        ),
        "source_sha256": (
            "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8"
        ),
        "retired": True,
        "relabel_allowed": False,
    }
    assert exclusions["descriptor_source_cross_kind_substitution_allowed"] is False
    assert (
        exclusions["algorithmic_validator_can_substitute_for_full_resource_merger"]
        is False
    )
    assert exclusions["storage_validator_can_substitute_for_full_resource_merger"] is False
    assert exclusions["endpoint_observer_can_substitute_for_full_resource_merger"] is False


def test_both_adapter_publications_and_all_runners_are_explicitly_unqualified() -> None:
    adapter = plan.matched_v3_qualification_plan_v3_descriptor()[
        "publisher_registry"
    ]["adapter"]

    assert adapter["strict_slot_count"] == 2
    assert adapter["recognized_strict_slot_count"] == 0
    assert adapter["historical_adapter_publication_v1"] == {
        "descriptor_sha256": (
            "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
        ),
        "source_sha256": (
            "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
        ),
        "counted_toward_strict_slots": False,
        "disposition": "historical_v1_excluded_not_reinterpreted",
    }
    assert adapter["adapter_atomic_publication_v2"]["descriptor_sha256"] == (
        "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6"
    )
    assert adapter["adapter_atomic_publication_v2"]["source_sha256"] == (
        "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2"
    )
    assert (
        adapter["historical_adapter_publication_v1"][
            "counted_toward_strict_slots"
        ]
        is False
    )
    assert (
        adapter["adapter_atomic_publication_v2"]["counted_toward_strict_slots"]
        is False
    )
    for disposition in adapter["runner_dispositions"].values():
        assert disposition["qualification_ready"] is False
        assert disposition["relabel_as_qualified_allowed"] is False
    assert adapter["runner_dispositions"]["adapted_full_rainbow"][
        "descriptor_sha256"
    ] == "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
    assert adapter["runner_dispositions"]["adapted_full_rainbow"][
        "source_sha256"
    ] == "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
    assert adapter["runner_dispositions"]["adapted_ppo_gru"][
        "descriptor_sha256"
    ] == "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
    assert adapter["runner_dispositions"]["adapted_ppo_gru"][
        "source_sha256"
    ] == "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"

    compiled = adapter["historical_compiled_ppo_gru_addendum"]
    assert compiled["candidate_id"] == "adapted_ppo_gru"
    assert compiled["fills_adapted_ppo_gru_slot"] is False
    assert compiled["counted_toward_strict_slots"] is False
    assert compiled["qualification_ready"] is False
    assert compiled["relabel_as_qualified_allowed"] is False
    assert compiled["escape_via_historical_addendum_allowed"] is False
    assert [
        (item["role"], item["descriptor_sha256"], item["source_sha256"])
        for item in compiled["components"]
    ] == [
        (
            "compiled_ppo_gru_runner",
            "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
            "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
        ),
        (
            "compiled_ppo_gru_bundle",
            "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
            "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
        ),
        (
            "compiled_ppo_gru_six_file_publication",
            "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
            "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
        ),
    ]


def test_historical_image_and_one_shot_lineage_exclusions_are_exact() -> None:
    assert plan.HISTORICAL_IMAGE_IDS == (
        "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269",
        "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768",
    )
    assert plan.HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS == (
        (
            "context_receipt_sha256",
            "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330",
        ),
        (
            "execution_receipt_sha256",
            "38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d",
        ),
        (
            "publication_receipt_sha256",
            "28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88",
        ),
    )


@pytest.mark.parametrize("historical_image_id", plan.HISTORICAL_IMAGE_IDS)
def test_each_historical_image_is_rejected(
    historical_image_id: str,
    finalized_descriptor_pins: None,
) -> None:
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.build_matched_v3_qualification_plan_v3(
            structural_inputs=(
                plan.synthetically_complete_matched_v3_structural_inputs_v3()
            ),
            publisher_closures=(),
            proposed_image_id=historical_image_id,
            proposed_build_lineage_components=(),
        )


@pytest.mark.parametrize(
    "historical_component",
    [item[1] for item in plan.HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS],
)
def test_each_prior_one_shot_lineage_component_is_rejected(
    historical_component: str,
    finalized_descriptor_pins: None,
) -> None:
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.build_matched_v3_qualification_plan_v3(
            structural_inputs=(
                plan.synthetically_complete_matched_v3_structural_inputs_v3()
            ),
            publisher_closures=(),
            proposed_image_id=None,
            proposed_build_lineage_components=(historical_component,),
        )


def test_zero_image_and_lineage_component_are_not_present_artifact_identities(
    finalized_descriptor_pins: None,
) -> None:
    structural = plan.synthetically_complete_matched_v3_structural_inputs_v3()

    for image_id, components in (
        ("sha256:" + "0" * 64, _FRESH_LINEAGE),
        (_FRESH_IMAGE_ID, ("0" * 64, *_FRESH_LINEAGE[1:])),
    ):
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.build_matched_v3_qualification_plan_v3(
                structural_inputs=structural,
                publisher_closures=(),
                proposed_image_id=image_id,
                proposed_build_lineage_components=components,
            )


def test_zero_image_and_lineage_component_fail_strict_replay(
    finalized_descriptor_pins: None,
) -> None:
    for path, replacement in (
        (("proposed_lineage", "image_id"), "sha256:" + "0" * 64),
        (("proposed_lineage", "build_lineage_components"), ["0" * 64]),
    ):
        value = _build()
        _set_nested(value, path, replacement)
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)


def test_drifted_reordered_and_aliased_publisher_closures_fail_closed(
    finalized_descriptor_pins: None,
) -> None:
    local = plan.matched_v3_local_publisher_closure_v3()
    external = plan.matched_v3_external_publisher_closure_v3()
    changed_component = dataclasses.replace(
        local.components[0],
        implementation_source_sha256="1" * 64,
    )
    changed_local = dataclasses.replace(
        local,
        components=(changed_component, *local.components[1:]),
    )
    reordered_external = dataclasses.replace(
        external,
        components=tuple(reversed(external.components)),
    )

    for closures in ((changed_local,), (reordered_external,), (local, local)):
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _build(closures=closures)


@pytest.mark.parametrize(
    "field_name",
    (
        "role",
        "descriptor_schema_version",
        "descriptor_sha256",
        "implementation_path",
        "implementation_source_sha256",
    ),
)
def test_component_identity_rejects_string_subclasses(field_name: str) -> None:
    component = plan.matched_v3_local_publisher_closure_v3().components[0]

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        dataclasses.replace(
            component,
            **{field_name: _StringAlias(getattr(component, field_name))},
        )


def test_closure_and_structural_inputs_require_exact_builtin_types() -> None:
    local = plan.matched_v3_local_publisher_closure_v3()
    closure_mutations = (
        lambda: dataclasses.replace(local, family=_StringAlias("local")),
        lambda: dataclasses.replace(
            local,
            family=cast(str, _EquivalentStringKey("local")),
        ),
        lambda: dataclasses.replace(
            local,
            candidate_ids=(_StringAlias(local.candidate_ids[0]), *local.candidate_ids[1:]),
        ),
        lambda: dataclasses.replace(
            local,
            trust_direction=_StringAlias(local.trust_direction),
        ),
        lambda: dataclasses.replace(
            local,
            candidate_ids=cast(tuple[str, ...], _TupleAlias(local.candidate_ids)),
        ),
        lambda: dataclasses.replace(
            local,
            components=cast(
                tuple[plan.PublisherComponentIdentityV3, ...],
                _TupleAlias(local.components),
            ),
        ),
    )
    for mutate in closure_mutations:
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            mutate()

    exact_mapping = {name: True for name in _EXPECTED_GAP_IDS}
    alias_mapping = dict(exact_mapping)
    first_value = alias_mapping.pop(_EXPECTED_GAP_IDS[0])
    alias_mapping[_StringAlias(_EXPECTED_GAP_IDS[0])] = first_value
    proxy_mapping = cast(dict[str, bool], dict(exact_mapping))
    proxy_mapping[cast(str, _EquivalentStringKey(_EXPECTED_GAP_IDS[0]))] = (
        proxy_mapping.pop(_EXPECTED_GAP_IDS[0])
    )
    for mapping in (
        cast(dict[str, bool], _DictAlias(exact_mapping)),
        alias_mapping,
        proxy_mapping,
    ):
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.QualificationPlanV3StructuralInputs.from_mapping(mapping)

    exact_signals = tuple((name, True) for name in _EXPECTED_GAP_IDS)
    inexact_signals = (
        ((_StringAlias(_EXPECTED_GAP_IDS[0]), True), *exact_signals[1:]),
        ((exact_signals[0][0], cast(bool, 1)), *exact_signals[1:]),
        cast(tuple[tuple[str, bool], ...], _TupleAlias(exact_signals)),
    )
    for signals in inexact_signals:
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.QualificationPlanV3StructuralInputs(signals)


def test_public_plan_inputs_reject_builtin_subclasses(
    finalized_descriptor_pins: None,
) -> None:
    structural = plan.synthetically_complete_matched_v3_structural_inputs_v3()
    local = plan.matched_v3_local_publisher_closure_v3()
    build_mutations = (
        {
            "publisher_closures": cast(
                tuple[plan.PublisherClosureIdentityV3, ...],
                _TupleAlias((local,)),
            ),
            "proposed_image_id": _FRESH_IMAGE_ID,
            "proposed_build_lineage_components": _FRESH_LINEAGE,
        },
        {
            "publisher_closures": cast(
                tuple[plan.PublisherClosureIdentityV3, ...],
                _ListAlias([local]),
            ),
            "proposed_image_id": _FRESH_IMAGE_ID,
            "proposed_build_lineage_components": _FRESH_LINEAGE,
        },
        {
            "publisher_closures": (local,),
            "proposed_image_id": _StringAlias(_FRESH_IMAGE_ID),
            "proposed_build_lineage_components": _FRESH_LINEAGE,
        },
        {
            "publisher_closures": (local,),
            "proposed_image_id": _FRESH_IMAGE_ID,
            "proposed_build_lineage_components": (
                _StringAlias(_FRESH_LINEAGE[0]),
                *_FRESH_LINEAGE[1:],
            ),
        },
    )
    for mutation in build_mutations:
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.build_matched_v3_qualification_plan_v3(
                structural_inputs=structural,
                **mutation,
            )

    value = _build()
    raw = plan.canonical_matched_v3_qualification_plan_v3_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.replay_matched_v3_qualification_plan_v3(
            _BytesAlias(raw),
            expected_plan_sha256=digest,
        )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.replay_matched_v3_qualification_plan_v3(
            raw,
            expected_plan_sha256=_StringAlias(digest),
        )


@pytest.mark.parametrize(
    "implementation_path",
    (".", "..", "./publisher.py", "publisher/../escape.py", "publisher/./file.py"),
)
def test_component_identity_rejects_noncanonical_relative_paths(
    implementation_path: str,
) -> None:
    component = plan.matched_v3_local_publisher_closure_v3().components[0]

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        dataclasses.replace(component, implementation_path=implementation_path)


@pytest.mark.parametrize(
    "implementation_path",
    ("../publisher.py", "publisher/../escape.py", "publisher/./file.py"),
)
def test_plan_parser_rejects_component_path_traversal(
    implementation_path: str,
    finalized_descriptor_pins: None,
) -> None:
    value = _build()
    value["publisher_registry"]["recognized_closures"][0]["components"][0][
        "implementation_path"
    ] = implementation_path

    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        _replay_mutation(value)


def test_plan_round_trip_requires_canonical_bytes_and_full_file_pin(
    finalized_descriptor_pins: None,
) -> None:
    value = _build()
    raw = plan.canonical_matched_v3_qualification_plan_v3_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    body = copy.deepcopy(value)
    supplied_body_sha256 = body.pop("plan_body_sha256")

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert supplied_body_sha256 == hashlib.sha256(
        _canonical(body, newline=False)
    ).hexdigest()
    assert digest != supplied_body_sha256
    assert plan.replay_matched_v3_qualification_plan_v3(
        raw,
        expected_plan_sha256=digest,
    ) == value
    assert plan.parse_matched_v3_qualification_plan_v3(
        raw,
        expected_plan_sha256=digest,
    ) == value
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.replay_matched_v3_qualification_plan_v3(
            raw,
            expected_plan_sha256="0" * 64,
        )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.replay_matched_v3_qualification_plan_v3(
            raw,
            expected_plan_sha256=supplied_body_sha256,
        )

    noncanonical = json.dumps(value, sort_keys=True).encode("ascii") + b"\n"
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.replay_matched_v3_qualification_plan_v3(
            noncanonical,
            expected_plan_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_duplicate_keys_and_floats_fail_strict_replay(
    finalized_descriptor_pins: None,
) -> None:
    value = _build()
    raw = plan.canonical_matched_v3_qualification_plan_v3_bytes(value)
    duplicate = b'{"schema_version":"duplicate",' + raw[1:]
    floating = raw.replace(b'"ordinal":0', b'"ordinal":0.0', 1)
    nonfinite = raw.replace(b'"ordinal":0', b'"ordinal":NaN', 1)

    for malformed in (duplicate, floating, nonfinite):
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.replay_matched_v3_qualification_plan_v3(
                malformed,
                expected_plan_sha256=hashlib.sha256(malformed).hexdigest(),
            )


def test_strict_json_alias_cycle_depth_node_text_integer_and_byte_bounds(
    finalized_descriptor_pins: None,
) -> None:
    aliased = _build()
    shared: list[Any] = []
    aliased["candidate_order"] = shared
    aliased["limitations"] = shared
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.canonical_matched_v3_qualification_plan_v3_bytes(aliased)

    cyclic = _build()
    cyclic["limitations"].append(cyclic["limitations"])
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.canonical_matched_v3_qualification_plan_v3_bytes(cyclic)

    too_deep = _build()
    nested: list[Any] = []
    cursor = nested
    for _index in range(plan._MAX_JSON_DEPTH + 1):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    too_deep["limitations"] = nested

    too_many_nodes = _build()
    too_many_nodes["limitations"] = [None] * (plan._MAX_JSON_NODES + 1)

    too_long = _build()
    too_long["limitations"][0] = "x" * (plan._MAX_TEXT_LENGTH + 1)

    out_of_range_int = plan.canonical_matched_v3_qualification_plan_v3_bytes(_build()).replace(
        b'"ordinal":0',
        b'"ordinal":9223372036854775808',
        1,
    )

    for malformed in (
        _rebody(too_deep),
        _rebody(too_many_nodes),
        _rebody(too_long),
        out_of_range_int,
        b" " * (plan._MAX_ARTIFACT_BYTES + 1),
    ):
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            plan.replay_matched_v3_qualification_plan_v3(
                malformed,
                expected_plan_sha256=hashlib.sha256(malformed).hexdigest(),
            )


def test_semantic_mutations_fail_even_with_recomputed_body_and_file_digests(
    finalized_descriptor_pins: None,
) -> None:
    mutations: list[dict[str, Any]] = []

    candidate_order = _build()
    candidate_order["candidate_order"][0], candidate_order["candidate_order"][1] = (
        candidate_order["candidate_order"][1],
        candidate_order["candidate_order"][0],
    )
    mutations.append(candidate_order)

    claim = _build()
    claim["claims"]["qualification_granted"] = True
    mutations.append(claim)

    readiness = _build()
    readiness["readiness"]["qualification_ready"] = True
    mutations.append(readiness)

    blocker = _build()
    blocker["blocker_ledger"][0]["satisfied"] = True
    blocker["blocker_ledger"][0]["accepted_artifact_sha256"] = "1" * 64
    mutations.append(blocker)

    registry = _build()
    registry["publisher_registry"]["qualifying_count"] = 1
    mutations.append(registry)

    host_descriptor = _build()
    host_descriptor["observation_and_execution_gaps"]["host_executor"][
        "incompatible_v1"
    ]["descriptor_sha256"] = "0" * 64
    mutations.append(host_descriptor)

    host_source = _build()
    host_source["observation_and_execution_gaps"]["host_executor"][
        "incompatible_v1"
    ]["implementation_source_sha256"] = "0" * 64
    mutations.append(host_source)

    structural_bool = _build()
    structural_bool["structural_inputs"][_EXPECTED_GAP_IDS[0]] = 1
    mutations.append(structural_bool)

    for mutation in mutations:
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(mutation)


def test_every_gap_readiness_and_claim_mutation_fails_closed(
    finalized_descriptor_pins: None,
) -> None:
    for section, keys in (
        ("claims", _EXPECTED_CLAIM_IDS),
        ("readiness", _EXPECTED_READINESS_IDS),
    ):
        for key in keys:
            value = _build()
            value[section][key] = True
            with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
                _replay_mutation(value)

    for index, expected_gap_id in enumerate(_EXPECTED_GAP_IDS):
        value = _build()
        assert value["blocker_ledger"][index]["gap_id"] == expected_gap_id
        value["blocker_ledger"][index]["satisfied"] = True
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    blocker_field_mutations = {
        "gap_id": "different_gap",
        "required_future_artifact": "different requirement",
        "accepted_artifact_sha256": "1" * 64,
        "caller_structural_presence_can_satisfy": True,
        "caller_reports_structural_presence": False,
    }
    for field_name, replacement in blocker_field_mutations.items():
        value = _build()
        value["blocker_ledger"][0][field_name] = replacement
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    for section, key in (
        ("claims", _EXPECTED_CLAIM_IDS[0]),
        ("readiness", _EXPECTED_READINESS_IDS[0]),
    ):
        value = _build()
        del value[section][key]
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)


def test_order_publisher_and_authority_ledger_mutations_fail_closed(
    finalized_descriptor_pins: None,
) -> None:
    mutations: list[dict[str, Any]] = []

    for path, replacement in (
        (("status",), "different_status"),
        (("classification",), "different_classification"),
        (("descriptor_binding", "schema_version"), "different.schema.v1"),
        (("descriptor_binding", "body_sha256"), "0" * 64),
        (("descriptor_binding", "file_sha256"), "0" * 64),
        (("partition_counts", "local"), 13),
        (("partition_counts", "external"), 13),
        (("partition_counts", "adapter"), 1),
        (("proposed_lineage", "accepted_as_fresh_build"), True),
        (("proposed_lineage", "qualifies_runtime"), True),
        (("future_authority_boundary", "separate_plan_issuer_required"), False),
        (
            ("future_authority_boundary", "separate_acceptance_evaluator_required"),
            False,
        ),
        (("future_authority_boundary", "implemented_here"), True),
    ):
        value = _build()
        _set_nested(value, path, replacement)
        mutations.append(value)

    for key, replacement in (
        ("recognized_nonqualifying_count", 25),
        ("missing_strict_adapter_count", 1),
        ("missing_strict_adapter_candidate_ids", ["adapted_full_rainbow"]),
        ("qualifying_count", 1),
        ("complete", True),
    ):
        value = _build()
        value["publisher_registry"][key] = replacement
        mutations.append(value)

    for field_name, replacement in (
        ("ordinal", 1),
        ("candidate_id", "different_candidate"),
        ("family", "adapter"),
        ("publisher_status", "qualified"),
        ("qualification_ready", True),
    ):
        value = _build()
        value["publisher_registry"]["candidate_entries"][0][field_name] = replacement
        mutations.append(value)

    reordered_entries = _build()
    reordered_entries["publisher_registry"]["candidate_entries"][0:2] = reversed(
        reordered_entries["publisher_registry"]["candidate_entries"][0:2]
    )
    mutations.append(reordered_entries)
    missing_entry = _build()
    missing_entry["publisher_registry"]["candidate_entries"].pop()
    mutations.append(missing_entry)
    duplicate_entry = _build()
    duplicate_entry["publisher_registry"]["candidate_entries"][-1] = copy.deepcopy(
        duplicate_entry["publisher_registry"]["candidate_entries"][0]
    )
    mutations.append(duplicate_entry)
    reordered_closures = _build()
    reordered_closures["publisher_registry"]["recognized_closures"].reverse()
    mutations.append(reordered_closures)
    changed_limitation = _build()
    changed_limitation["limitations"][0] = "different limitation"
    mutations.append(changed_limitation)

    for mutation in mutations:
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(mutation)


def test_dependency_incompatibility_and_placeholder_mutations_fail_closed(
    finalized_descriptor_pins: None,
) -> None:
    prefix = ("observation_and_execution_gaps",)
    identity_paths = (
        prefix
        + (
            "finalized_source_only_dependencies",
            "algorithmic_resource_contract",
            "descriptor_sha256",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "algorithmic_resource_contract",
            "implementation_source_sha256",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "storage_boundary_contract",
            "descriptor_sha256",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "storage_boundary_contract",
            "implementation_source_sha256",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "normalized_publication_contract",
            "descriptor_sha256",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "normalized_publication_contract",
            "implementation_source_sha256",
        ),
        prefix + ("host_executor", "incompatible_v1", "descriptor_sha256"),
        prefix
        + ("host_executor", "incompatible_v1", "implementation_source_sha256"),
        prefix + ("observation_registry_v1", "descriptor_sha256"),
        prefix + ("resource_observation", "endpoint_observer_descriptor_sha256"),
        prefix + ("resource_observation", "endpoint_observer_source_sha256"),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "retired_algorithmic_resource_validator",
            "descriptor_sha256",
        ),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "retired_algorithmic_resource_validator",
            "source_sha256",
        ),
    )
    for path in identity_paths:
        value = _build()
        _set_nested(value, path, "0" * 64)
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    for dependency_name in (
        "algorithmic_resource_contract",
        "storage_boundary_contract",
        "normalized_publication_contract",
    ):
        for field_name in ("schema_version", "status", "classification"):
            value = _build()
            _set_nested(
                value,
                prefix
                + ("finalized_source_only_dependencies", dependency_name, field_name),
                "different_value",
            )
            with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
                _replay_mutation(value)

    truth_mutations = (
        prefix + ("observation_registry_v1", "v2_compatible"),
        prefix + ("observation_registry_v1", "v3_compatible"),
        prefix + ("observation_registry_v1", "may_fill_observation_registry_v2_gap"),
        prefix + ("host_executor", "incompatible_v1", "v2_compatible"),
        prefix + ("host_executor", "incompatible_v1", "may_fill_production_executor_gap"),
        prefix + ("host_executor", "source_only_v2", "descriptor_finalized"),
        prefix + ("host_executor", "source_only_v2", "source_identity_finalized"),
        prefix + ("host_executor", "source_only_v2", "production_backend_available"),
        prefix + ("host_executor", "source_only_v2", "production_executor_available"),
        prefix + ("host_executor", "source_only_v2", "may_self_issue"),
        prefix + ("host_executor", "source_only_v2", "may_self_evaluate"),
        prefix + ("host_executor", "source_only_v2", "gap_satisfied"),
        prefix + ("observation_registry_v2", "descriptor_finalized"),
        prefix + ("observation_registry_v2", "source_identity_finalized"),
        prefix + ("observation_registry_v2", "observation_issuer_available"),
        prefix + ("observation_registry_v2", "acceptance_evaluator_available"),
        prefix + ("observation_registry_v2", "production_registry_available"),
        prefix + ("observation_registry_v2", "gap_satisfied"),
        prefix + ("full_resource_merger", "descriptor_finalized"),
        prefix + ("full_resource_merger", "source_identity_finalized"),
        prefix + ("full_resource_merger", "production_merger_implemented"),
        prefix + ("full_resource_merger", "production_receipt_available"),
        prefix + ("full_resource_merger", "endpoint_observer_can_substitute"),
        prefix
        + ("full_resource_merger", "finalized_dependency_identity_can_substitute"),
        prefix + ("full_resource_merger", "gap_satisfied"),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "descriptor_source_cross_kind_substitution_allowed",
        ),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "algorithmic_validator_can_substitute_for_full_resource_merger",
        ),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "storage_validator_can_substitute_for_full_resource_merger",
        ),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "endpoint_observer_can_substitute_for_full_resource_merger",
        ),
    )
    for path in truth_mutations:
        value = _build()
        _set_nested(value, path, True)
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    required_true_paths = (
        prefix + ("host_executor", "source_only_v2", "source_contract_implemented"),
        prefix + ("observation_registry_v2", "structural_candidate_validator_implemented"),
        prefix + ("full_resource_merger", "structural_candidate_validator_implemented"),
        prefix
        + (
            "finalized_source_only_dependencies",
            "algorithmic_resource_contract",
            "source_contract_implemented",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "storage_boundary_contract",
            "source_contract_implemented",
        ),
        prefix
        + (
            "finalized_source_only_dependencies",
            "normalized_publication_contract",
            "source_contract_implemented",
        ),
        prefix
        + (
            "cross_kind_identity_exclusions",
            "retired_algorithmic_resource_validator",
            "retired",
        ),
    )
    for path in required_true_paths:
        value = _build()
        _set_nested(value, path, False)
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    for dependency_name in (
        "algorithmic_resource_contract",
        "storage_boundary_contract",
        "normalized_publication_contract",
    ):
        for field_name in (
            "production_implementation_available",
            "production_receipt_available",
            "qualification_ready",
            "fills_any_structural_gap",
        ):
            value = _build()
            _set_nested(
                value,
                prefix
                + ("finalized_source_only_dependencies", dependency_name, field_name),
                True,
            )
            with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
                _replay_mutation(value)

    nullable_pin_paths = (
        prefix + ("host_executor", "source_only_v2", "descriptor_body_sha256"),
        prefix + ("host_executor", "source_only_v2", "descriptor_file_sha256"),
        prefix + ("host_executor", "source_only_v2", "implementation_source_sha256"),
        prefix + ("observation_registry_v2", "descriptor_body_sha256"),
        prefix + ("observation_registry_v2", "descriptor_file_sha256"),
        prefix + ("observation_registry_v2", "implementation_source_sha256"),
        prefix + ("full_resource_merger", "descriptor_body_sha256"),
        prefix + ("full_resource_merger", "descriptor_file_sha256"),
        prefix + ("full_resource_merger", "implementation_source_sha256"),
    )
    for path in nullable_pin_paths:
        value = _build()
        _set_nested(value, path, "1" * 64)
        with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
            _replay_mutation(value)

    union = _build()
    exclusions = union["observation_and_execution_gaps"][
        "cross_kind_identity_exclusions"
    ]["algorithmic_resource_validator_identity_sha256_union"]
    exclusions[0], exclusions[1] = exclusions[1], exclusions[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        _replay_mutation(union)


def test_descriptor_copies_are_detached_and_parser_rejects_mutation(
    finalized_descriptor_pins: None,
) -> None:
    first = plan.matched_v3_qualification_plan_v3_descriptor()
    second = plan.matched_v3_qualification_plan_v3_descriptor()
    first["claims"]["qualification_granted"] = True

    assert second["claims"]["qualification_granted"] is False
    raw = plan.canonical_matched_v3_qualification_plan_v3_descriptor_bytes()
    changed = raw.replace(
        b'"qualification_granted":false',
        b'"qualification_granted":true',
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanV3Error):
        plan.parse_matched_v3_qualification_plan_v3_descriptor(changed)


def test_public_build_and_replay_inputs_are_required_keyword_only() -> None:
    build_signature = inspect.signature(plan.build_matched_v3_qualification_plan_v3)
    replay_signature = inspect.signature(plan.replay_matched_v3_qualification_plan_v3)

    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in build_signature.parameters.values()
    )
    assert replay_signature.parameters["expected_plan_sha256"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert replay_signature.parameters["expected_plan_sha256"].default is (
        inspect.Parameter.empty
    )
