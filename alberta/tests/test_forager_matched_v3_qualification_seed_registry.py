from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import FrozenInstanceError
from typing import Any, NamedTuple, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v2 as plan_v2
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_seed_registry as registry,
)

_EXPECTED_DESCRIPTOR_SHA256 = (
    "fba1ab637f72de87c926169f2e0df5e66a8a2c7dcf855f00442a33dbe42fbef2"
)
_SYNTHETIC_ROUND = 1_234_567


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


class SyntheticInputs(NamedTuple):
    pulse: registry.QuicknetPulseRecord
    receipt: registry.TrustRootReceiptIdentity
    derived: registry.QualificationSeedRegistry
    raw: bytes


def _synthetic_pulse() -> registry.QuicknetPulseRecord:
    raw_signature_hex = bytes(range(1, registry.QUICKNET_RAW_SIGNATURE_BYTES + 1)).hex()
    randomness = hashlib.sha256(bytes.fromhex(raw_signature_hex)).hexdigest()
    return registry.QuicknetPulseRecord(
        schema_version=registry.QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
        provider_id=registry.QUICKNET_PROVIDER_ID,
        provider_chain_hash=registry.QUICKNET_CHAIN_HASH,
        signature_scheme=registry.QUICKNET_SIGNATURE_SCHEME,
        provider_public_key_hex=registry.QUICKNET_PUBLIC_KEY_HEX,
        provider_public_key_raw_sha256=registry.QUICKNET_PUBLIC_KEY_RAW_SHA256,
        beacon_round=_SYNTHETIC_ROUND,
        beacon_time_unix=(
            registry.QUICKNET_GENESIS_TIME_UNIX
            + (_SYNTHETIC_ROUND - 1) * registry.QUICKNET_PERIOD_SECONDS
        ),
        raw_signature_hex=raw_signature_hex,
        raw_signature_sha256=randomness,
        randomness_hex=randomness,
        bls_message_scope=registry.QUICKNET_BLS_MESSAGE_SCOPE,
        randomness_derivation=registry.QUICKNET_RANDOMNESS_DERIVATION,
        timestamp_source=registry.QUICKNET_TIMESTAMP_SOURCE,
        offline_signature_verification_required=True,
        offline_signature_verified_here=False,
        cryptographic_authentication_accepted_here=False,
    )


def _synthetic_receipt(
    pulse: registry.QuicknetPulseRecord,
) -> registry.TrustRootReceiptIdentity:
    return registry.TrustRootReceiptIdentity(
        schema_version=(
            registry.QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_IDENTITY_SCHEMA_VERSION
        ),
        receipt_schema_version=(
            registry.QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION
        ),
        receipt_file_sha256=_sha("synthetic-external-receipt-file"),
        receipt_body_sha256=_sha("synthetic-external-receipt-body"),
        provider_id=pulse.provider_id,
        provider_chain_hash=pulse.provider_chain_hash,
        signature_scheme=pulse.signature_scheme,
        provider_public_key_raw_sha256=pulse.provider_public_key_raw_sha256,
        pulse_record_schema_version=pulse.schema_version,
        pulse_record_file_sha256=pulse.file_sha256,
        pulse_record_body_sha256=pulse.body_sha256,
        beacon_round=pulse.beacon_round,
        beacon_time_unix=pulse.beacon_time_unix,
        observation_cutoff_unix=pulse.beacon_time_unix + 60,
        raw_signature_sha256=pulse.raw_signature_sha256,
        randomness_hex=pulse.randomness_hex,
        offline_signature_verification_required=True,
        offline_signature_verified_here=False,
        external_preacceptance_required=True,
        external_preacceptance_accepted_here=False,
        preacceptance_chronology_required=True,
        preacceptance_chronology_accepted_here=False,
    )


def _derive(
    pulse: registry.QuicknetPulseRecord,
    receipt: registry.TrustRootReceiptIdentity,
) -> registry.QualificationSeedRegistry:
    return registry.derive_matched_v3_qualification_seed_registry(
        pulse,
        receipt,
        expected_pulse_record_file_sha256=pulse.file_sha256,
        expected_pulse_record_body_sha256=pulse.body_sha256,
        expected_trust_root_receipt_file_sha256=receipt.receipt_file_sha256,
        expected_trust_root_receipt_body_sha256=receipt.receipt_body_sha256,
        expected_trust_root_receipt_binding_sha256=receipt.binding_sha256,
    )


def _synthetic_inputs() -> SyntheticInputs:
    pulse = _synthetic_pulse()
    receipt = _synthetic_receipt(pulse)
    derived = _derive(pulse, receipt)
    raw = registry.canonical_matched_v3_qualification_seed_registry_bytes(derived)
    return SyntheticInputs(pulse, receipt, derived, raw)


def _parse_kwargs(inputs: SyntheticInputs) -> dict[str, str]:
    return {
        "expected_registry_file_sha256": inputs.derived.file_sha256,
        "expected_registry_body_sha256": inputs.derived.body_sha256,
        "expected_pulse_record_file_sha256": inputs.pulse.file_sha256,
        "expected_pulse_record_body_sha256": inputs.pulse.body_sha256,
        "expected_trust_root_receipt_file_sha256": inputs.receipt.receipt_file_sha256,
        "expected_trust_root_receipt_body_sha256": inputs.receipt.receipt_body_sha256,
        "expected_trust_root_receipt_binding_sha256": inputs.receipt.binding_sha256,
    }


def _payload(inputs: SyntheticInputs) -> dict[str, Any]:
    value = json.loads(inputs.raw.decode("ascii"))
    assert type(value) is dict
    return cast(dict[str, Any], value)


def _rehash_registry_payload(value: dict[str, Any]) -> tuple[bytes, str]:
    body = copy.deepcopy(value)
    body.pop("registry_body_sha256", None)
    body_sha256 = hashlib.sha256(_canonical(body)).hexdigest()
    full = {**body, "registry_body_sha256": body_sha256}
    return _canonical(full), body_sha256


def _parse_rehashed_mutation(
    inputs: SyntheticInputs,
    value: dict[str, Any],
) -> registry.QualificationSeedRegistry:
    raw, body_sha256 = _rehash_registry_payload(value)
    kwargs = _parse_kwargs(inputs)
    kwargs["expected_registry_file_sha256"] = hashlib.sha256(raw).hexdigest()
    kwargs["expected_registry_body_sha256"] = body_sha256
    return registry.parse_matched_v3_qualification_seed_registry_artifact(raw, **kwargs)


@pytest.mark.unit
def test_descriptor_is_frozen_current_score_blind_and_non_authorizing() -> None:
    descriptor = registry.matched_v3_qualification_seed_registry_descriptor()
    raw = registry.canonical_matched_v3_qualification_seed_registry_descriptor_bytes()

    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256
    assert registry.QUALIFICATION_SEED_REGISTRY_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert registry.matched_v3_qualification_seed_registry_descriptor_sha256() == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert registry.parse_matched_v3_qualification_seed_registry_descriptor(raw) == descriptor
    assert descriptor["status"] == (
        "implemented_deterministic_derivation_no_offline_signature_verifier_no_issuer"
    )
    assert descriptor["candidate_order"] == list(
        plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    assert descriptor["candidate_count"] == 28
    assert descriptor["derivation"]["numeric_seed_collisions_allowed"] is True
    assert descriptor["derivation"]["numeric_seed_inequality_required"] is False
    assert descriptor["quicknet"]["signature_authenticates_round_only"] is True
    assert descriptor["quicknet"]["signature_authenticates_registry"] is False
    assert descriptor["quicknet"]["signature_authenticates_trust_root_receipt"] is False
    assert descriptor["authentication"] == {
        "offline_signature_verification_required": True,
        "offline_signature_verification_implemented_here": False,
        "offline_signature_verified_here": False,
        "external_preacceptance_required": True,
        "external_preacceptance_accepted_here": False,
        "preacceptance_chronology_required": True,
        "preacceptance_chronology_accepted_here": False,
        "beacon_time_precedes_observation_cutoff_structurally": True,
        "pulse_time_alone_proves_preacceptance_chronology": False,
    }
    assert all(value is False for value in descriptor["capabilities"].values())
    assert descriptor["claims"]["deterministic_derivation_implemented"] is True
    assert all(
        value is False
        for key, value in descriptor["claims"].items()
        if key != "deterministic_derivation_implemented"
    )
    serialized = raw.decode("ascii")
    assert '"score":' not in serialized
    assert '"reward":' not in serialized
    assert '"ranking":' not in serialized

    descriptor["claims"]["execution_authorized"] = True
    assert registry.matched_v3_qualification_seed_registry_descriptor()["claims"][
        "execution_authorized"
    ] is False


@pytest.mark.unit
def test_exact_qualification_v2_order_is_repeated_without_importing_it_in_module() -> None:
    assert registry.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS == (
        plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    assert len(set(registry.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS)) == 28
    module_names = registry.__dict__
    assert "plan_v2" not in module_names
    assert "forager_matched_v3_qualification_plan" not in module_names


@pytest.mark.unit
def test_deterministic_derivation_and_strict_replay_cover_every_candidate() -> None:
    inputs = _synthetic_inputs()
    repeated = _derive(inputs.pulse, inputs.receipt)
    replayed = registry.parse_matched_v3_qualification_seed_registry_artifact(
        inputs.raw,
        **_parse_kwargs(inputs),
    )

    assert repeated == inputs.derived
    assert replayed == inputs.derived
    assert registry.canonical_matched_v3_qualification_seed_registry_bytes(replayed) == (
        inputs.raw
    )
    assert hashlib.sha256(inputs.raw).hexdigest() == inputs.derived.file_sha256
    assert replayed.to_dict()["trust_root_receipt_identity_binding_sha256"] == (
        inputs.receipt.binding_sha256
    )
    assert [item.candidate_id for item in replayed.cases] == list(
        registry.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    assert [item.registry_case_ordinal for item in replayed.cases] == list(range(28))

    for ordinal, (candidate_id, case) in enumerate(
        zip(replayed.candidate_order, replayed.cases, strict=True)
    ):
        payload = {
            "schema_version": registry.QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
            "domain": registry.QUALIFICATION_SEED_DERIVATION_DOMAIN,
            "algorithm": registry.QUALIFICATION_SEED_DERIVATION_ALGORITHM,
            "provider_chain_hash": inputs.pulse.provider_chain_hash,
            "beacon_round": inputs.pulse.beacon_round,
            "beacon_randomness_hex": inputs.pulse.randomness_hex,
            "candidate_id": candidate_id,
            "registry_case_ordinal": ordinal,
        }
        expected_lane_digests = tuple(
            hashlib.sha256(_canonical({**payload, "lane": lane})).hexdigest()
            for lane in ("environment", "agent")
        )
        assert case.case_id == f"qualification_{ordinal:02d}_{candidate_id}"
        assert case.derivation_payload_sha256 == hashlib.sha256(
            _canonical(payload)
        ).hexdigest()
        assert case.environment_seed_derivation_sha256 == expected_lane_digests[0]
        assert case.agent_seed_derivation_sha256 == expected_lane_digests[1]
        assert case.environment_seed == (
            int.from_bytes(bytes.fromhex(expected_lane_digests[0])[:4], "big")
            & registry.UINT31_MAX
        )
        assert case.agent_seed == (
            int.from_bytes(bytes.fromhex(expected_lane_digests[1])[:4], "big")
            & registry.UINT31_MAX
        )


@pytest.mark.unit
def test_raw_quicknet_structure_is_bound_but_not_cryptographically_accepted() -> None:
    inputs = _synthetic_inputs()
    pulse_raw = registry.canonical_quicknet_pulse_record_bytes(inputs.pulse)
    parsed = registry.parse_quicknet_pulse_record_artifact(
        pulse_raw,
        expected_file_sha256=inputs.pulse.file_sha256,
        expected_body_sha256=inputs.pulse.body_sha256,
    )
    assert parsed == inputs.pulse
    assert len(bytes.fromhex(parsed.provider_public_key_hex)) == 96
    assert len(bytes.fromhex(parsed.raw_signature_hex)) == 48
    assert parsed.raw_signature_sha256 == hashlib.sha256(
        bytes.fromhex(parsed.raw_signature_hex)
    ).hexdigest()
    assert parsed.randomness_hex == parsed.raw_signature_sha256
    assert parsed.bls_message_scope == "unchained_round_only"
    assert parsed.offline_signature_verified_here is False
    assert parsed.cryptographic_authentication_accepted_here is False

    different_signature = bytes(range(2, registry.QUICKNET_RAW_SIGNATURE_BYTES + 2)).hex()
    different_randomness = hashlib.sha256(bytes.fromhex(different_signature)).hexdigest()
    structurally_coherent = dataclasses.replace(
        inputs.pulse,
        raw_signature_hex=different_signature,
        raw_signature_sha256=different_randomness,
        randomness_hex=different_randomness,
    )
    assert structurally_coherent.offline_signature_verified_here is False
    assert structurally_coherent.cryptographic_authentication_accepted_here is False


@pytest.mark.unit
def test_official_quicknet_round_1000_vector_matches_frozen_semantics() -> None:
    signature = (
        "b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125"
        "e342b73a8dd2bacbe47e4b6b63ed5e39"
    )
    randomness = "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd"
    assert len(bytes.fromhex(signature)) == registry.QUICKNET_RAW_SIGNATURE_BYTES
    assert hashlib.sha256(bytes.fromhex(signature)).hexdigest() == randomness
    assert registry.QUICKNET_GENESIS_TIME_UNIX + 999 * registry.QUICKNET_PERIOD_SECONDS == (
        1_692_806_364
    )

    pulse = dataclasses.replace(
        _synthetic_pulse(),
        beacon_round=1_000,
        beacon_time_unix=1_692_806_364,
        raw_signature_hex=signature,
        raw_signature_sha256=randomness,
        randomness_hex=randomness,
    )
    assert pulse.randomness_hex == randomness
    assert pulse.offline_signature_verified_here is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provider_public_key_hex", "00" * 96, "public key"),
        ("raw_signature_hex", "01" * 48, "raw-signature digest"),
        ("randomness_hex", _sha("unbound-randomness"), "randomness"),
        ("beacon_time_unix", 1, "round time"),
        ("bls_message_scope", "registry_and_receipt", "round only"),
        ("offline_signature_verified_here", True, "verification result"),
        ("cryptographic_authentication_accepted_here", True, "acceptance"),
    ],
)
@pytest.mark.unit
def test_quicknet_structure_and_unaccepted_authentication_state_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match=match,
    ):
        replace(_synthetic_pulse(), **{field: value})


@pytest.mark.unit
def test_receipt_binding_is_detached_independently_pinned_and_still_unaccepted() -> None:
    inputs = _synthetic_inputs()
    raw = registry.canonical_trust_root_receipt_identity_bytes(inputs.receipt)
    parsed = registry.parse_trust_root_receipt_identity_binding(
        raw,
        expected_binding_sha256=inputs.receipt.binding_sha256,
        expected_receipt_file_sha256=inputs.receipt.receipt_file_sha256,
        expected_receipt_body_sha256=inputs.receipt.receipt_body_sha256,
    )
    assert parsed == inputs.receipt
    assert parsed.offline_signature_verified_here is False
    assert parsed.external_preacceptance_accepted_here is False
    assert parsed.preacceptance_chronology_accepted_here is False
    assert parsed.beacon_time_unix == inputs.pulse.beacon_time_unix
    assert parsed.beacon_time_unix < parsed.observation_cutoff_unix
    assert inputs.derived.authority.pulse_time_alone_proves_preacceptance_chronology is False

    replace = cast(Any, dataclasses.replace)
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="precede the observation cutoff",
    ):
        replace(parsed, observation_cutoff_unix=parsed.beacon_time_unix)
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="deterministic Quicknet round time",
    ):
        replace(parsed, beacon_time_unix=parsed.beacon_time_unix + 1)
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="randomness must equal SHA-256",
    ):
        replace(parsed, randomness_hex=_sha("detached-randomness"))


@pytest.mark.unit
def test_receipt_parser_rejects_rehashed_detached_randomness_mismatch() -> None:
    inputs = _synthetic_inputs()
    value = inputs.receipt.to_dict()
    value["randomness_hex"] = _sha("detached-randomness")
    raw = _canonical(value)

    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="randomness must equal SHA-256",
    ):
        registry.parse_trust_root_receipt_identity_binding(
            raw,
            expected_binding_sha256=hashlib.sha256(raw).hexdigest(),
            expected_receipt_file_sha256=inputs.receipt.receipt_file_sha256,
            expected_receipt_body_sha256=inputs.receipt.receipt_body_sha256,
        )


@pytest.mark.unit
def test_distinct_derivation_identities_may_retain_equal_uint31_collisions() -> None:
    first_digest = "80000001" + "00" * 28
    second_digest = "00000001" + "00" * 28
    assert first_digest != second_digest
    assert registry.uint31_seed_from_derivation_sha256(first_digest) == 1
    assert registry.uint31_seed_from_derivation_sha256(second_digest) == 1

    case = _synthetic_inputs().derived.cases[0]
    collision = dataclasses.replace(
        case,
        environment_seed=1,
        agent_seed=1,
        environment_seed_derivation_sha256=first_digest,
        agent_seed_derivation_sha256=second_digest,
    )
    assert collision.environment_seed == collision.agent_seed
    assert collision.environment_seed_derivation_sha256 != (
        collision.agent_seed_derivation_sha256
    )


@pytest.mark.unit
def test_seed_case_rejects_numeric_values_not_projected_from_lane_digests() -> None:
    case = _synthetic_inputs().derived.cases[0]
    replace = cast(Any, dataclasses.replace)

    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="environment seed is not the uint31 projection",
    ):
        replace(case, environment_seed=case.environment_seed ^ 1)
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="agent seed is not the uint31 projection",
    ):
        replace(case, agent_seed=case.agent_seed ^ 1)


@pytest.mark.unit
def test_immutable_records_reject_bool_integer_confusion() -> None:
    inputs = _synthetic_inputs()
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        dataclasses.replace(inputs.pulse, beacon_round=True)
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        dataclasses.replace(inputs.derived.cases[0], registry_case_ordinal=False)
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        dataclasses.replace(inputs.derived.cases[0], environment_seed=True)
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        replace(inputs.receipt, external_preacceptance_required=1)
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        replace(inputs.derived.authority, execution_authorized=0)

    changed = _payload(inputs)
    changed["cases"][0]["environment_seed"] = True
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        _parse_rehashed_mutation(inputs, changed)


@pytest.mark.unit
def test_all_records_are_frozen() -> None:
    inputs = _synthetic_inputs()
    records = (
        inputs.pulse,
        inputs.receipt,
        inputs.derived.authority,
        inputs.derived.cases[0],
        inputs.derived,
    )
    for item in records:
        field_name = dataclasses.fields(item)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(item, field_name, getattr(item, field_name))


@pytest.mark.parametrize(
    "pin_name",
    [
        "expected_registry_file_sha256",
        "expected_registry_body_sha256",
        "expected_pulse_record_file_sha256",
        "expected_pulse_record_body_sha256",
        "expected_trust_root_receipt_file_sha256",
        "expected_trust_root_receipt_body_sha256",
        "expected_trust_root_receipt_binding_sha256",
    ],
)
@pytest.mark.unit
def test_every_independent_full_file_body_and_binding_pin_is_load_bearing(
    pin_name: str,
) -> None:
    inputs = _synthetic_inputs()
    kwargs = _parse_kwargs(inputs)
    kwargs[pin_name] = _sha(f"mutated-{pin_name}")
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        registry.parse_matched_v3_qualification_seed_registry_artifact(
            inputs.raw,
            **kwargs,
        )


@pytest.mark.unit
def test_wrong_candidate_or_case_order_fails_even_after_coherent_rehash() -> None:
    inputs = _synthetic_inputs()
    wrong_candidates = _payload(inputs)
    wrong_candidates["candidate_order"][0], wrong_candidates["candidate_order"][1] = (
        wrong_candidates["candidate_order"][1],
        wrong_candidates["candidate_order"][0],
    )
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="28-candidate order",
    ):
        _parse_rehashed_mutation(inputs, wrong_candidates)

    wrong_cases = _payload(inputs)
    wrong_cases["cases"][0], wrong_cases["cases"][1] = (
        wrong_cases["cases"][1],
        wrong_cases["cases"][0],
    )
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        _parse_rehashed_mutation(inputs, wrong_cases)


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
@pytest.mark.unit
def test_every_authority_mutation_fails_after_coherent_rehash(field: str) -> None:
    inputs = _synthetic_inputs()
    changed = _payload(inputs)
    current = changed["authority"][field]
    assert type(current) is bool
    changed["authority"][field] = not current
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        _parse_rehashed_mutation(inputs, changed)


@pytest.mark.unit
def test_case_derivation_or_pulse_receipt_cross_wiring_fails_after_rehash() -> None:
    inputs = _synthetic_inputs()
    changed_seed = _payload(inputs)
    changed_seed["cases"][7]["agent_seed"] ^= 1
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="uint31 projection",
    ):
        _parse_rehashed_mutation(inputs, changed_seed)

    changed_receipt = _payload(inputs)
    changed_receipt["trust_root_receipt_identity"]["pulse_record_file_sha256"] = _sha(
        "cross-wired-pulse"
    )
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        _parse_rehashed_mutation(inputs, changed_receipt)


@pytest.mark.unit
def test_strict_json_rejects_alias_cycle_float_depth_node_and_text_inputs() -> None:
    shared: list[object] = []
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="aliased"):
        registry.canonical_json_bytes({"left": shared, "right": shared})

    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="cyclic"):
        registry.canonical_json_bytes(cycle)

    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="JSON types"):
        registry.canonical_json_bytes({"float": 1.0})
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="integer"):
        registry.canonical_json_bytes({"integer": 1 << 63})
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="depth"):
        nested: Any = None
        for _ in range(registry._MAX_JSON_DEPTH + 1):
            nested = [nested]
        registry.canonical_json_bytes({"nested": nested})
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="node"):
        registry.canonical_json_bytes(
            {"nodes": [None] * (registry._MAX_JSON_NODES + 1)}
        )
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError, match="strings"):
        registry.canonical_json_bytes({"text": "x" * (registry._MAX_TEXT_LENGTH + 1)})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"duplicate":1,"duplicate":2}\n',
        b'{"float":1.0}\n',
        b'{"constant":NaN}\n',
    ],
)
@pytest.mark.unit
def test_strict_parser_rejects_duplicate_float_and_nonfinite_json(raw: bytes) -> None:
    inputs = _synthetic_inputs()
    kwargs = _parse_kwargs(inputs)
    kwargs["expected_registry_file_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        registry.parse_matched_v3_qualification_seed_registry_artifact(raw, **kwargs)


@pytest.mark.unit
def test_strict_parser_rejects_depth_node_text_and_noncanonical_bytes() -> None:
    inputs = _synthetic_inputs()
    malformed_values: list[dict[str, Any]] = []
    nested: Any = None
    for _ in range(registry._MAX_JSON_DEPTH + 2):
        nested = [nested]
    malformed_values.append({"nested": nested})
    malformed_values.append({"nodes": [None] * (registry._MAX_JSON_NODES + 1)})
    malformed_values.append({"text": "x" * (registry._MAX_TEXT_LENGTH + 1)})
    for value in malformed_values:
        raw = _canonical(value)
        kwargs = _parse_kwargs(inputs)
        kwargs["expected_registry_file_sha256"] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
            registry.parse_matched_v3_qualification_seed_registry_artifact(raw, **kwargs)

    noncanonical = inputs.raw.replace(b'"authority":', b'"authority" :', 1)
    kwargs = _parse_kwargs(inputs)
    kwargs["expected_registry_file_sha256"] = hashlib.sha256(noncanonical).hexdigest()
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="canonical form",
    ):
        registry.parse_matched_v3_qualification_seed_registry_artifact(
            noncanonical,
            **kwargs,
        )


@pytest.mark.unit
def test_descriptor_and_registry_reject_digest_and_body_mutations() -> None:
    descriptor_raw = registry.canonical_matched_v3_qualification_seed_registry_descriptor_bytes()
    with pytest.raises(registry.ForagerMatchedV3QualificationSeedRegistryError):
        registry.parse_matched_v3_qualification_seed_registry_descriptor(
            descriptor_raw.replace(b"no_issuer", b"has_issuer", 1)
        )

    inputs = _synthetic_inputs()
    changed = _payload(inputs)
    changed["cases"][0]["agent_seed"] ^= 1
    raw_with_stale_body = _canonical(changed)
    kwargs = _parse_kwargs(inputs)
    kwargs["expected_registry_file_sha256"] = hashlib.sha256(raw_with_stale_body).hexdigest()
    with pytest.raises(
        registry.ForagerMatchedV3QualificationSeedRegistryError,
        match="body digest",
    ):
        registry.parse_matched_v3_qualification_seed_registry_artifact(
            raw_with_stale_body,
            **kwargs,
        )


@pytest.mark.unit
def test_no_issuer_fetch_verifier_clock_filesystem_or_production_default_api_exists() -> None:
    forbidden_fragments = (
        "issue",
        "fetch",
        "download",
        "network",
        "clock",
        "verify_bls",
        "verify_signature",
        "publish",
    )
    public_callables = {
        name
        for name in registry.__all__
        if callable(getattr(registry, name, None))
    }
    assert not any(
        fragment in name.lower()
        for name in public_callables
        for fragment in forbidden_fragments
    )
    assert not hasattr(registry, "DEFAULT_PULSE")
    assert not hasattr(registry, "PRODUCTION_REGISTRY")
    assert not hasattr(registry, "issue_qualification_seed_registry")
    assert not hasattr(registry, "fetch_quicknet_pulse")
    assert not hasattr(registry, "verify_quicknet_bls_signature")
    assert "pathlib" not in registry.__dict__
    assert "time" not in registry.__dict__
    assert "urllib" not in registry.__dict__
    assert "requests" not in registry.__dict__
