"""Focused tests for the source-only matched-v3 algorithmic resource contract."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_algorithmic_resource_contract as contract,
)

EXPECTED_DESCRIPTOR_SHA256 = "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"

# Test-local literal replay: these values must not be sourced from the module
# under test.  They are interleaved as the seven historical descriptor/source
# pairs while their rejection semantics deliberately form one cross-kind union.
EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
)


class _StrSubclass(str):
    pass


class _EqualityProxy:
    def __init__(self, target: str) -> None:
        self.target = target

    def __eq__(self, other: object) -> bool:
        return bool(other == self.target)

    def __ne__(self, other: object) -> bool:
        return not self == other


def _adversarial_text(value: str, adversary_kind: str) -> object:
    if adversary_kind == "str_subclass":
        return _StrSubclass(value)
    if adversary_kind == "equality_proxy":
        return _EqualityProxy(value)
    raise AssertionError("unknown test adversary")


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


def _rebody(value: dict[str, Any], field: str) -> bytes:
    body = copy.deepcopy(value)
    body.pop(field, None)
    value[field] = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    return _canonical(value)


def _family(case_ordinal: int) -> str:
    candidate = contract.MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS[case_ordinal]
    if candidate in contract.MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate in contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    return "adapter"


def _producer(case_ordinal: int) -> contract.ProducerIdentityV1:
    family = _family(case_ordinal)
    return contract.ProducerIdentityV1(
        descriptor_schema_version=(
            contract.algorithmic_resource_producer_schema_for_family(family)
        ),
        descriptor_sha256=_sha(f"{family}-producer-descriptor"),
        source_sha256=_sha(f"{family}-producer-source"),
    )


def _intent(case_ordinal: int = 0) -> contract.AlgorithmicResourceMeasurementIntentV1:
    candidate = contract.MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS[case_ordinal]
    family = _family(case_ordinal)
    policy = contract.matched_v3_algorithmic_resource_field_policy()
    return contract.AlgorithmicResourceMeasurementIntentV1(
        schema_version=contract.ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
        campaign_spine_sha256=_sha("campaign-spine"),
        case_spine_sha256=_sha(f"case-spine-{case_ordinal}"),
        case_ordinal=case_ordinal,
        candidate_id=candidate,
        candidate_family=family,  # type: ignore[arg-type]
        qualification_case_id=f"qualification_{case_ordinal:02d}_{candidate}",
        resource_requirement_body_sha256=_sha(f"requirement-{case_ordinal}"),
        configuration_sha256=_sha(f"configuration-{candidate}"),
        producer=_producer(case_ordinal),
        field_policy_inventory_sha256=(
            contract.algorithmic_resource_field_policy_inventory_sha256(policy)
        ),
        field_policy=policy,
    )


def _positive_fields() -> tuple[contract.AlgorithmicResourceMeasurementV1, ...]:
    result: list[contract.AlgorithmicResourceMeasurementV1] = []
    for index, policy in enumerate(
        contract.matched_v3_algorithmic_resource_field_policy(),
        start=1,
    ):
        value = (
            contract.MATCHED_V3_HORIZON
            if policy.field_name == "max_environment_interactions"
            else index
        )
        result.append(
            contract.AlgorithmicResourceMeasurementV1(
                field_name=policy.field_name,
                observed_value=value,
                measurement_kind=policy.positive_measurement_kind,
                measurement_scope=policy.measurement_scope,
                measurement_basis_body_sha256=_sha(f"basis-{policy.field_name}"),
                structural_absence_kind=contract.NOT_ABSENT,
            )
        )
    return tuple(result)


def _intent_identity(
    intent: contract.AlgorithmicResourceMeasurementIntentV1,
) -> contract.ArtifactIdentityV1:
    raw = contract.canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(intent)
    return contract.ArtifactIdentityV1(
        schema_version=contract.ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=intent.body_sha256,
    )


def _receipt(
    case_ordinal: int = 0,
    *,
    intent: contract.AlgorithmicResourceMeasurementIntentV1 | None = None,
    fields: tuple[contract.AlgorithmicResourceMeasurementV1, ...] | None = None,
) -> contract.AlgorithmicResourceReceiptV1:
    exact_intent = _intent(case_ordinal) if intent is None else intent
    exact_fields = _positive_fields() if fields is None else fields
    return contract.AlgorithmicResourceReceiptV1(
        schema_version=contract.algorithmic_resource_receipt_schema_for_family(
            exact_intent.candidate_family
        ),
        campaign_spine_sha256=exact_intent.campaign_spine_sha256,
        case_spine_sha256=exact_intent.case_spine_sha256,
        case_ordinal=exact_intent.case_ordinal,
        candidate_id=exact_intent.candidate_id,
        candidate_family=exact_intent.candidate_family,
        qualification_case_id=exact_intent.qualification_case_id,
        resource_requirement_body_sha256=(exact_intent.resource_requirement_body_sha256),
        configuration_sha256=exact_intent.configuration_sha256,
        producer=exact_intent.producer,
        measurement_intent=_intent_identity(exact_intent),
        runner_execution_receipt=contract.ArtifactIdentityV1(
            schema_version=contract.runner_execution_receipt_schema_for_candidate(
                exact_intent.candidate_id
            ),
            file_sha256=_sha(f"runner-file-{exact_intent.candidate_id}"),
            body_sha256=_sha(f"runner-body-{exact_intent.candidate_id}"),
        ),
        field_inventory_sha256=(
            contract.algorithmic_resource_measurement_inventory_sha256(exact_fields)
        ),
        fields=exact_fields,
    )


def _typed_zero_fields(
    *field_names: str,
) -> tuple[contract.AlgorithmicResourceMeasurementV1, ...]:
    """Apply typed absence and intrinsic subsystem/pair closure only."""

    requested = set(field_names)
    for left, right in contract.COUPLED_RESOURCE_FIELD_PAIRS:
        if left in requested or right in requested:
            requested.update((left, right))
    if "max_optimizer_updates" in requested:
        requested.update(("max_optimizer_state_elements", "max_optimizer_state_bytes"))
    result: list[contract.AlgorithmicResourceMeasurementV1] = []
    shared_pair_basis: dict[tuple[str, str], str] = {
        pair: _sha(f"absence-{'-'.join(pair)}") for pair in contract.COUPLED_RESOURCE_FIELD_PAIRS
    }
    for item in _positive_fields():
        if item.field_name not in requested:
            result.append(item)
            continue
        basis = _sha(f"absence-{item.field_name}")
        for pair, pair_basis in shared_pair_basis.items():
            if item.field_name in pair:
                basis = pair_basis
                break
        result.append(
            dataclasses.replace(
                item,
                observed_value=0,
                measurement_kind=contract.STRUCTURAL_ABSENCE,
                measurement_basis_body_sha256=basis,
                structural_absence_kind=(contract.FIELD_ALLOWED_ZERO_ABSENCE[item.field_name]),
            )
        )
    return tuple(result)


def _coherent_zero_fields(
    *field_names: str,
) -> tuple[contract.AlgorithmicResourceMeasurementV1, ...]:
    """Close typed zeroes over the frozen positive dependency implications."""

    requested = set(field_names)
    if requested.intersection({"max_sample_updates", "max_trainable_parameters"}):
        requested.add("max_optimizer_updates")
    if "max_optimizer_updates" in requested:
        requested.add("max_gradient_updates")
    return _typed_zero_fields(*requested)


def _all_false(value: object) -> bool:
    if type(value) is dict:
        return all(_all_false(child) for child in value.values())
    return value is False


def test_frozen_candidate_and_family_order() -> None:
    assert len(contract.MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS) == 28
    assert contract.MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS == (
        contract.MATCHED_V3_LOCAL_CANDIDATE_IDS
        + contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
        + contract.MATCHED_V3_ADAPTER_CANDIDATE_IDS
        + contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
    )
    assert len(contract.MATCHED_V3_LOCAL_CANDIDATE_IDS) == 14
    assert len(contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS) == 12
    assert len(contract.MATCHED_V3_ADAPTER_CANDIDATE_IDS) == 2


def test_frozen_first_twenty_field_order_and_optimizer_state_absence() -> None:
    assert len(contract.ALGORITHMIC_RESOURCE_FIELDS) == 20
    assert contract.ALGORITHMIC_RESOURCE_FIELDS[0] == "max_environment_interactions"
    assert contract.ALGORITHMIC_RESOURCE_FIELDS[-1] == "max_eligibility_bytes"
    assert (
        contract.FIELD_ALLOWED_ZERO_ABSENCE["max_optimizer_state_elements"]
        == "optimizer_state_tree_absent"
    )
    assert (
        contract.FIELD_ALLOWED_ZERO_ABSENCE["max_optimizer_state_bytes"]
        == "optimizer_state_tree_absent"
    )


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
@pytest.mark.parametrize(
    "field_name",
    (
        "field_name",
        "positive_measurement_kind",
        "measurement_scope",
        "zero_structural_absence_kind",
    ),
)
def test_field_policy_direct_string_fields_require_exact_strings(
    field_name: str,
    adversary_kind: str,
) -> None:
    policy = contract.matched_v3_algorithmic_resource_field_policy()[0]
    adversary = _adversarial_text(getattr(policy, field_name), adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            policy,
            **{field_name: adversary},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
@pytest.mark.parametrize(
    "field_name",
    (
        "field_name",
        "measurement_kind",
        "measurement_scope",
        "structural_absence_kind",
    ),
)
def test_measurement_direct_string_fields_require_exact_strings(
    field_name: str,
    adversary_kind: str,
) -> None:
    measurement = _positive_fields()[1]
    adversary = _adversarial_text(getattr(measurement, field_name), adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            measurement,
            **{field_name: adversary},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("case_ordinal", "family", "receipt_schema", "runner_schema"),
    [
        (
            0,
            "local",
            contract.LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            contract.LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        ),
        (
            14,
            "external",
            contract.EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            contract.EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        ),
        (
            23,
            "adapter",
            contract.ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            contract.FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        ),
        (
            24,
            "adapter",
            contract.ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            contract.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
        ),
        (
            27,
            "external",
            contract.EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            contract.EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        ),
    ],
)
def test_family_schema_projection(
    case_ordinal: int,
    family: str,
    receipt_schema: str,
    runner_schema: str,
) -> None:
    intent = _intent(case_ordinal)
    receipt = _receipt(case_ordinal, intent=intent)
    assert intent.candidate_family == family
    assert receipt.schema_version == receipt_schema
    assert receipt.runner_execution_receipt.schema_version == runner_schema


def test_descriptor_is_source_only_and_all_flags_are_false() -> None:
    descriptor = contract.matched_v3_algorithmic_resource_contract_descriptor()
    assert descriptor["status"] == contract.ALGORITHMIC_RESOURCE_CONTRACT_STATUS
    assert descriptor["candidate_values_supplied_or_inferred"] is False
    assert descriptor["ceiling_comparison_performed"] is False
    assert descriptor["runner_execution_receipts_are_complete_resource_records"] is False
    assert descriptor["runner_execution_receipt_schemas"] == {
        "local": contract.LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "external": contract.EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "adapted_full_rainbow": (contract.FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION),
        "adapted_ppo_gru": contract.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
    }
    assert descriptor["adapter_algorithmic_resource_producer_identity_policy"] == {
        "historical_or_unqualified_identity_sha256_union_denylist": list(
            EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST
        ),
        "historical_or_unqualified_pair_count": 7,
        "denylist_value_count": 14,
        "cross_kind_union_rejection": True,
        "applies_to_families": ["adapter"],
        "applies_to_artifacts": [
            "alberta.forager_matched_v3.algorithmic_resource_measurement_intent.v1",
            "alberta.forager_matched_v3.adapter_algorithmic_resource_receipt.v1",
        ],
        "applies_to_producer_fields": [
            "descriptor_sha256",
            "source_sha256",
        ],
        "future_full_resource_merger_must_pin_exact_production_producer_pair": True,
        "exact_production_producer_pair_pinned_here": False,
        "production_adapter_algorithmic_resource_producer_implemented_here": False,
        "full_resource_merger_implemented_here": False,
        "producer_descriptor_source_or_runner_bytes_read_here": False,
    }
    assert descriptor["adapted_ppo_gru_runner_schema_policy"] == {
        "accepted_schema_version": "alberta.forager_matched_v3.ppo_gru_result_receipt.v1",
        "compiled_v2_admitted": False,
    }
    for section in ("capabilities", "readiness", "authority", "claims"):
        assert _all_false(descriptor[section])
    assert descriptor["measurement_chain"] == [
        "pre_go_measurement_intent",
        "family_runner_execution_receipt",
        "algorithmic_resource_receipt",
        "future_terminal_v2",
        "future_host_success_v2",
        "future_full_resource_merger",
    ]
    body = copy.deepcopy(descriptor)
    supplied_body_sha256 = body.pop("descriptor_body_sha256")
    assert supplied_body_sha256 == hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    assert (
        contract.canonical_matched_v3_algorithmic_resource_contract_descriptor_bytes()
        == _canonical(descriptor)
    )


def test_adapter_producer_identity_denylist_is_exact_independent_literal_union() -> None:
    observed = contract.ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYLIST
    assert type(observed) is tuple
    assert observed == EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST
    assert len(observed) == 14
    assert len(set(observed)) == 14
    assert all(type(value) is str and len(value) == 64 for value in observed)


def test_source_import_and_api_surface_remains_stdlib_only_and_nonoperational() -> None:
    tree = ast.parse(inspect.getsource(contract))
    imported_roots: set[str] = set()
    public_functions: set[str] = set()
    forbidden_calls: set[str] = set()
    operational_prefixes = (
        "connect_",
        "evaluate_",
        "execute_",
        "invoke_",
        "issue_",
        "open_",
        "publish_",
        "qualify_",
        "read_",
        "run_",
        "spawn_",
        "write_",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public_functions.add(node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "compile", "__import__"}:
                forbidden_calls.add(node.func.id)
    assert imported_roots == {
        "__future__",
        "collections",
        "copy",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "types",
        "typing",
    }
    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "jax",
            "numpy",
            "os",
            "pathlib",
            "requests",
            "socket",
            "subprocess",
            "urllib",
        }
    )
    assert forbidden_calls == set()
    assert not {name for name in public_functions if name.startswith(operational_prefixes)}


def test_descriptor_pin_matches_independent_schema_audit() -> None:
    assert (
        contract.PINNED_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        contract.matched_v3_algorithmic_resource_contract_descriptor_sha256()
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert contract.parse_matched_v3_algorithmic_resource_contract_descriptor(
        contract.canonical_matched_v3_algorithmic_resource_contract_descriptor_bytes()
    ) == contract.matched_v3_algorithmic_resource_contract_descriptor()


@pytest.mark.parametrize("case_ordinal", [0, 14, 23, 24, 27])
def test_intent_and_receipt_canonical_round_trip(case_ordinal: int) -> None:
    intent = _intent(case_ordinal)
    intent_raw = contract.canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(intent)
    parsed_intent = contract.parse_matched_v3_algorithmic_resource_measurement_intent(
        intent_raw,
        expected_file_sha256=hashlib.sha256(intent_raw).hexdigest(),
    )
    assert parsed_intent == intent
    receipt = _receipt(case_ordinal, intent=intent)
    receipt_raw = contract.canonical_matched_v3_algorithmic_resource_receipt_bytes(receipt)
    parsed_receipt = contract.parse_matched_v3_algorithmic_resource_receipt(
        receipt_raw,
        expected_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )
    assert parsed_receipt == receipt
    contract.validate_matched_v3_algorithmic_resource_receipt_chain(
        parsed_intent,
        parsed_receipt,
    )


def test_receipt_has_exact_six_key_measurement_records() -> None:
    fields = _receipt().to_dict()["fields"]
    assert all(
        list(item)
        == [
            "field_name",
            "observed_value",
            "measurement_kind",
            "measurement_scope",
            "measurement_basis_body_sha256",
            "structural_absence_kind",
        ]
        for item in fields
    )


def test_horizon_must_equal_499712() -> None:
    fields = list(_positive_fields())
    fields[0] = dataclasses.replace(fields[0], observed_value=contract.MATCHED_V3_HORIZON + 1)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


@pytest.mark.parametrize(
    "field_name",
    [
        name
        for name in contract.ALGORITHMIC_RESOURCE_FIELDS
        if name != "max_environment_interactions"
    ],
)
def test_every_nonhorizon_zero_requires_and_accepts_only_its_typed_absence(
    field_name: str,
) -> None:
    fields = _coherent_zero_fields(field_name)
    receipt = _receipt(fields=fields)
    by_name = {item.field_name: item for item in receipt.fields}
    assert by_name[field_name].observed_value == 0
    assert by_name[field_name].measurement_kind == contract.STRUCTURAL_ABSENCE
    assert (
        by_name[field_name].structural_absence_kind
        == contract.FIELD_ALLOWED_ZERO_ABSENCE[field_name]
    )


def test_zero_horizon_is_forbidden() -> None:
    item = _positive_fields()[0]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            item,
            observed_value=0,
            measurement_kind=contract.STRUCTURAL_ABSENCE,
            structural_absence_kind=contract.ZERO_FORBIDDEN,
        )


def test_zero_with_wrong_absence_kind_is_rejected() -> None:
    item = _positive_fields()[1]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            item,
            observed_value=0,
            measurement_kind=contract.STRUCTURAL_ABSENCE,
            structural_absence_kind="optimizer_state_tree_absent",
        )


def test_zero_with_positive_measurement_kind_is_rejected() -> None:
    item = _positive_fields()[2]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            item,
            observed_value=0,
            structural_absence_kind="gradient_update_path_absent",
        )


def test_positive_with_structural_absence_is_rejected() -> None:
    item = _positive_fields()[6]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(item, structural_absence_kind="optimizer_state_tree_absent")


@pytest.mark.parametrize("bad_value", [True, -1, 2**63])
def test_non_exact_or_out_of_range_measurement_is_rejected(bad_value: object) -> None:
    item = _positive_fields()[1]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(item, observed_value=bad_value)  # type: ignore[arg-type]


def test_wrong_measurement_kind_scope_or_basis_is_rejected() -> None:
    item = _positive_fields()[4]
    for changes in (
        {"measurement_kind": contract.EXACT_RUNTIME_COUNTER},
        {"measurement_scope": "wrong_scope"},
        {"measurement_basis_body_sha256": "0" * 64},
    ):
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            dataclasses.replace(item, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("pair", contract.COUPLED_RESOURCE_FIELD_PAIRS)
def test_coupled_fields_reject_mixed_zero_nonzero_state(pair: tuple[str, str]) -> None:
    fields = list(_typed_zero_fields(*pair))
    positive = {item.field_name: item for item in _positive_fields()}
    index = contract.ALGORITHMIC_RESOURCE_FIELDS.index(pair[1])
    fields[index] = positive[pair[1]]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


def test_coupled_zero_fields_require_one_absence_basis() -> None:
    fields = list(
        _typed_zero_fields(
            "max_optimizer_state_elements",
            "max_optimizer_state_bytes",
        )
    )
    index = contract.ALGORITHMIC_RESOURCE_FIELDS.index("max_optimizer_state_bytes")
    fields[index] = dataclasses.replace(
        fields[index],
        measurement_basis_body_sha256=_sha("cross-wired-absence-basis"),
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


def test_absent_optimizer_subsystem_cannot_retain_optimizer_state() -> None:
    fields = list(_positive_fields())
    index = contract.ALGORITHMIC_RESOURCE_FIELDS.index("max_optimizer_updates")
    fields[index] = dataclasses.replace(
        fields[index],
        observed_value=0,
        measurement_kind=contract.STRUCTURAL_ABSENCE,
        structural_absence_kind="optimizer_subsystem_absent",
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


@pytest.mark.parametrize(
    "absent_dependency",
    [
        "max_optimizer_updates",
        "max_sample_updates",
        "max_trainable_parameters",
    ],
)
def test_positive_gradient_updates_require_positive_learning_dependencies(
    absent_dependency: str,
) -> None:
    fields = _typed_zero_fields(absent_dependency)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(fields)


@pytest.mark.parametrize(
    "absent_dependency",
    ["max_sample_updates", "max_trainable_parameters"],
)
def test_positive_optimizer_updates_require_samples_and_trainable_parameters(
    absent_dependency: str,
) -> None:
    fields = list(_typed_zero_fields(absent_dependency))
    gradient_index = contract.ALGORITHMIC_RESOURCE_FIELDS.index("max_gradient_updates")
    gradient = fields[gradient_index]
    fields[gradient_index] = dataclasses.replace(
        gradient,
        observed_value=0,
        measurement_kind=contract.STRUCTURAL_ABSENCE,
        measurement_basis_body_sha256=_sha("gradient-path-absence"),
        structural_absence_kind="gradient_update_path_absent",
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


@pytest.mark.parametrize("operation", ["missing", "duplicate", "reordered"])
def test_measurement_inventory_requires_exact_first_twenty_order(operation: str) -> None:
    fields = list(_positive_fields())
    if operation == "missing":
        fields.pop()
    elif operation == "duplicate":
        fields[-1] = fields[-2]
    else:
        fields[1], fields[2] = fields[2], fields[1]
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.algorithmic_resource_measurement_inventory_sha256(tuple(fields))


@pytest.mark.parametrize(
    ("case_ordinal", "change"),
    [
        (0, {"candidate_id": "causal_e025_q075"}),
        (0, {"candidate_family": "external"}),
        (0, {"qualification_case_id": "qualification_00_wrong"}),
        (14, {"case_ordinal": 15}),
    ],
)
def test_intent_rejects_case_projection_cross_wires(
    case_ordinal: int,
    change: dict[str, object],
) -> None:
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(_intent(case_ordinal), **change)  # type: ignore[arg-type]


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
@pytest.mark.parametrize(
    "field_name",
    ("schema_version", "candidate_id", "qualification_case_id"),
)
def test_intent_direct_string_fields_require_exact_strings(
    field_name: str,
    adversary_kind: str,
) -> None:
    intent = _intent(23)
    adversary = _adversarial_text(getattr(intent, field_name), adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            intent,
            **{field_name: adversary},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
def test_intent_field_policy_inventory_digest_requires_an_exact_string(
    adversary_kind: str,
) -> None:
    intent = _intent(23)
    adversary = _adversarial_text(intent.field_policy_inventory_sha256, adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            intent,
            field_policy_inventory_sha256=adversary,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
@pytest.mark.parametrize(
    "field_name",
    ("schema_version", "candidate_id", "qualification_case_id"),
)
def test_receipt_direct_string_fields_require_exact_strings(
    field_name: str,
    adversary_kind: str,
) -> None:
    receipt = _receipt(23)
    adversary = _adversarial_text(getattr(receipt, field_name), adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            receipt,
            **{field_name: adversary},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
def test_receipt_field_inventory_digest_requires_an_exact_string(
    adversary_kind: str,
) -> None:
    receipt = _receipt(23)
    adversary = _adversarial_text(receipt.field_inventory_sha256, adversary_kind)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            receipt,
            field_inventory_sha256=adversary,  # type: ignore[arg-type]
        )


def test_intent_rejects_wrong_family_producer_schema() -> None:
    intent = _intent()
    producer = dataclasses.replace(
        intent.producer,
        descriptor_schema_version=(
            contract.EXTERNAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
        ),
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(intent, producer=producer)


@pytest.mark.parametrize("denylist_index", range(14))
@pytest.mark.parametrize("producer_slot", ("descriptor_sha256", "source_sha256"))
@pytest.mark.parametrize("artifact_kind", ("intent", "receipt"))
def test_adapter_intent_and_receipt_reject_entire_producer_union_in_either_slot(
    denylist_index: int,
    producer_slot: str,
    artifact_kind: str,
) -> None:
    digest = EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST[denylist_index]
    if artifact_kind == "intent":
        intent = _intent(23)
        producer = dataclasses.replace(
            intent.producer,
            **{producer_slot: digest},
        )
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            dataclasses.replace(intent, producer=producer)
    else:
        receipt = _receipt(23)
        producer = dataclasses.replace(
            receipt.producer,
            **{producer_slot: digest},
        )
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            dataclasses.replace(receipt, producer=producer)


def test_adapter_producer_denylist_rejects_original_identity_kinds_when_cross_wired() -> None:
    adapter = _intent(24)
    historical_descriptors = EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST[0::2]
    historical_sources = EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST[1::2]
    for descriptor_digest, source_digest in zip(
        historical_descriptors,
        historical_sources,
        strict=True,
    ):
        for producer_slot, cross_kind_digest in (
            ("source_sha256", descriptor_digest),
            ("descriptor_sha256", source_digest),
        ):
            producer = dataclasses.replace(
                adapter.producer,
                **{producer_slot: cross_kind_digest},
            )
            with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
                dataclasses.replace(adapter, producer=producer)


@pytest.mark.parametrize("case_ordinal", (0, 14))
def test_local_and_external_producers_do_not_collide_with_adapter_only_denylist(
    case_ordinal: int,
) -> None:
    for denylist_index in range(14):
        digest = EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST[denylist_index]
        for producer_slot in ("descriptor_sha256", "source_sha256"):
            intent = _intent(case_ordinal)
            producer = dataclasses.replace(
                intent.producer,
                **{producer_slot: digest},
            )
            accepted_intent = dataclasses.replace(intent, producer=producer)
            accepted_receipt = _receipt(case_ordinal, intent=accepted_intent)
            assert getattr(accepted_intent.producer, producer_slot) == digest
            assert getattr(accepted_receipt.producer, producer_slot) == digest


def test_fresh_adapter_producer_pair_remains_structurally_accepted() -> None:
    intent = _intent(23)
    assert intent.producer.descriptor_sha256 not in set(
        EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST
    )
    assert intent.producer.source_sha256 not in set(
        EXPECTED_ADAPTER_PRODUCER_IDENTITY_SHA256_DENYLIST
    )
    receipt = _receipt(23, intent=intent)
    assert receipt.producer == intent.producer


@pytest.mark.parametrize("adversary_kind", ("str_subclass", "equality_proxy"))
@pytest.mark.parametrize("artifact_kind", ("intent", "receipt"))
def test_adapter_family_projection_requires_an_exact_string(
    artifact_kind: str,
    adversary_kind: str,
) -> None:
    adversary = _adversarial_text("adapter", adversary_kind)
    if artifact_kind == "intent":
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            dataclasses.replace(
                _intent(23),
                candidate_family=adversary,  # type: ignore[arg-type]
            )
    else:
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            dataclasses.replace(
                _receipt(23),
                candidate_family=adversary,  # type: ignore[arg-type]
            )


def test_parsers_reconstruct_only_exact_direct_string_fields() -> None:
    intent = _intent(23)
    intent_raw = contract.canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(intent)
    parsed_intent = contract.parse_matched_v3_algorithmic_resource_measurement_intent(
        intent_raw,
        expected_file_sha256=hashlib.sha256(intent_raw).hexdigest(),
    )
    for field_name in ("schema_version", "candidate_id", "qualification_case_id"):
        assert type(getattr(parsed_intent, field_name)) is str
    for policy in parsed_intent.field_policy:
        for field_name in (
            "field_name",
            "positive_measurement_kind",
            "measurement_scope",
            "zero_structural_absence_kind",
        ):
            assert type(getattr(policy, field_name)) is str

    receipt = _receipt(23, intent=intent)
    receipt_raw = contract.canonical_matched_v3_algorithmic_resource_receipt_bytes(receipt)
    parsed_receipt = contract.parse_matched_v3_algorithmic_resource_receipt(
        receipt_raw,
        expected_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )
    for field_name in ("schema_version", "candidate_id", "qualification_case_id"):
        assert type(getattr(parsed_receipt, field_name)) is str
    for measurement in parsed_receipt.fields:
        for field_name in (
            "field_name",
            "measurement_kind",
            "measurement_scope",
            "structural_absence_kind",
        ):
            assert type(getattr(measurement, field_name)) is str


@pytest.mark.parametrize("artifact_kind", ("intent", "receipt"))
def test_parser_rejects_nonstring_inventory_digest_after_coherent_body_rehash(
    artifact_kind: str,
) -> None:
    if artifact_kind == "intent":
        value = _intent(23).to_dict()
        exact_digest = value["field_policy_inventory_sha256"]
        value["field_policy_inventory_sha256"] = [exact_digest]
        raw = _rebody(value, "intent_body_sha256")
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_measurement_intent(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )
    else:
        value = _receipt(23).to_dict()
        exact_digest = value["field_inventory_sha256"]
        value["field_inventory_sha256"] = [exact_digest]
        raw = _rebody(value, "receipt_body_sha256")
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


@pytest.mark.parametrize("case_ordinal", [0, 14, 23, 24])
def test_receipt_rejects_wrong_runner_execution_schema(case_ordinal: int) -> None:
    receipt = _receipt(case_ordinal)
    wrong = dataclasses.replace(
        receipt.runner_execution_receipt,
        schema_version="alberta.forager_matched_v3.unrelated_runner_receipt.v1",
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(receipt, runner_execution_receipt=wrong)


def test_adapted_ppo_gru_rejects_explicitly_excluded_compiled_v2_receipt() -> None:
    receipt = _receipt(24)
    assert (
        receipt.runner_execution_receipt.schema_version
        == contract.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    )
    compiled = dataclasses.replace(
        receipt.runner_execution_receipt,
        schema_version=("alberta.forager_matched_v3.ppo_gru_compiled_result_receipt.v2"),
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(receipt, runner_execution_receipt=compiled)


@pytest.mark.parametrize(
    "field",
    [
        "campaign_spine_sha256",
        "case_spine_sha256",
        "resource_requirement_body_sha256",
        "configuration_sha256",
    ],
)
def test_chain_rejects_intent_projection_cross_wires(field: str) -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    cross_wired = dataclasses.replace(
        receipt,
        **{field: _sha(f"wrong-{field}")},  # type: ignore[arg-type]
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.validate_matched_v3_algorithmic_resource_receipt_chain(
            intent,
            cross_wired,
        )


def test_chain_rejects_wrong_intent_file_or_body_identity() -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    for field in ("file_sha256", "body_sha256"):
        identity = dataclasses.replace(
            receipt.measurement_intent,
            **{field: _sha(f"wrong-intent-{field}")},
        )
        cross_wired = dataclasses.replace(receipt, measurement_intent=identity)
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.validate_matched_v3_algorithmic_resource_receipt_chain(
                intent,
                cross_wired,
            )


def test_intent_and_receipt_are_frozen_dataclasses() -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.candidate_id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.candidate_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "reverse_key",
    [
        "publication_metadata",
        "terminal_metadata",
        "host_success_receipt",
        "storage_receipt",
        "full_resource_merger_receipt",
    ],
)
def test_receipt_parser_rejects_reverse_or_cyclic_pins(reverse_key: str) -> None:
    value = _receipt().to_dict()
    value[reverse_key] = _sha(reverse_key)
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_current_runner_receipt_shape_cannot_substitute_for_resource_receipt() -> None:
    raw = _canonical(
        {
            "schema_version": contract.LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "receipt_body_sha256": _sha("body"),
        }
    )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_requires_caller_full_file_pin() -> None:
    intent_raw = contract.canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(
        _intent()
    )
    receipt_raw = contract.canonical_matched_v3_algorithmic_resource_receipt_bytes(_receipt())
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_measurement_intent(
            intent_raw,
            expected_file_sha256=_sha("wrong-intent-file"),
        )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_receipt(
            receipt_raw,
            expected_file_sha256=_sha("wrong-receipt-file"),
        )


@pytest.mark.parametrize("artifact_kind", ["intent", "receipt"])
def test_parser_rejects_body_digest_tampering(artifact_kind: str) -> None:
    if artifact_kind == "intent":
        value = _intent().to_dict()
        value["configuration_sha256"] = _sha("tampered-configuration")
        raw = _canonical(value)
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_measurement_intent(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )
    else:
        value = _receipt().to_dict()
        value["configuration_sha256"] = _sha("tampered-configuration")
        raw = _canonical(value)
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_parser_rejects_field_inventory_digest_tampering() -> None:
    value = _receipt().to_dict()
    value["field_inventory_sha256"] = _sha("wrong-inventory")
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_rejects_true_capability_readiness_authority_or_claim() -> None:
    for section in ("capabilities", "readiness", "authority", "claims"):
        value = _receipt().to_dict()
        first_key = next(iter(value[section]))
        value[section][first_key] = True
        raw = _rebody(value, "receipt_body_sha256")
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_parser_rejects_boolean_alias_for_integer() -> None:
    value = _receipt().to_dict()
    value["fields"][1]["observed_value"] = True
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.parse_matched_v3_algorithmic_resource_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_rejects_float_nan_duplicate_key_and_noncanonical_bytes() -> None:
    receipt = _receipt()
    value = receipt.to_dict()
    value["fields"][1]["observed_value"] = 1.5
    float_raw = _canonical(value)
    canonical_raw = contract.canonical_matched_v3_algorithmic_resource_receipt_bytes(receipt)
    duplicate_raw = b'{"schema_version":"duplicate",' + canonical_raw[1:]
    nan_raw = canonical_raw.replace(b'"observed_value":2', b'"observed_value":NaN', 1)
    for raw in (float_raw, duplicate_raw, nan_raw, canonical_raw + b"\n"):
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_parser_rejects_non_ascii_and_oversized_bytes() -> None:
    raw = b'{"candidate_id":"\xc3\xa9"}\n'
    oversized = b" " * (1024 * 1024 + 1)
    for candidate in (raw, oversized):
        with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
            contract.parse_matched_v3_algorithmic_resource_receipt(
                candidate,
                expected_file_sha256=hashlib.sha256(candidate).hexdigest(),
            )


def test_artifact_and_producer_identities_reject_zero_hashes() -> None:
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.ArtifactIdentityV1(
            schema_version="alberta.example.v1",
            file_sha256="0" * 64,
            body_sha256=_sha("body"),
        )
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        contract.ProducerIdentityV1(
            descriptor_schema_version=(
                contract.LOCAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=_sha("descriptor"),
            source_sha256="0" * 64,
        )


@pytest.mark.parametrize("producer_slot", ("descriptor_sha256", "source_sha256"))
def test_producer_identity_hash_slots_require_exact_strings(producer_slot: str) -> None:
    producer = _producer(23)
    with pytest.raises(contract.ForagerMatchedV3AlgorithmicResourceContractError):
        dataclasses.replace(
            producer,
            **{producer_slot: _StrSubclass(_sha(f"subclass-{producer_slot}"))},
        )
