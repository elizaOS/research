"""Pure trust-contract tests for matched-v3 host provisioning v3.

These tests construct and corrupt canonical metadata only.  They never verify
an Ed25519 signature, inspect the host, open a file, or invoke a process,
container runtime, cgroup API, issuer, workload, or qualification path.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_host_provisioning_v3 as trust

ZERO_SHA256 = "0" * 64
EXPECTED_DESCRIPTOR_BODY_SHA256 = (
    "0e75dc103dc9b5b4f6d50b35e0832a11396a5f18b839deb05604548b1aacc54a"
)
EXPECTED_DESCRIPTOR_FILE_SHA256 = (
    "1ff3b76662504333749529926120c0f9a49dfd7aa010f5fc5951282feed4cf56"
)
PLAN_SCHEMA = "alberta.forager_matched_v3.qualification_plan.v3"
DESCRIPTOR_BODY_FIELD = "host_provisioning_trust_contract_descriptor_body_sha256"
POLICY_BODY_FIELD = "host_trust_policy_body_sha256"
STATEMENT_BODY_FIELD = "host_provisioning_statement_body_sha256"
VERIFICATION_BODY_FIELD = "host_signature_verification_receipt_body_sha256"
LIVE_BODY_FIELD = "host_live_validation_receipt_body_sha256"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


@contextmanager
def _observed_descriptor_pin() -> Any:
    descriptor = trust.HostProvisioningTrustContractDescriptorV1()
    raw = trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
        descriptor
    )
    observed = hashlib.sha256(raw).hexdigest()
    previous = trust.PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256
    setattr(trust, "PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256", observed)
    try:
        yield observed
    finally:
        setattr(trust, "PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256", previous)


def _artifact(schema: str, label: str) -> trust.ArtifactIdentityV1:
    return trust.ArtifactIdentityV1(
        schema_version=schema,
        file_sha256=_hash(label + ":file"),
        body_sha256=_hash(label + ":body"),
    )


def _component(label: str) -> trust.PinnedComponentIdentityV1:
    return trust.PinnedComponentIdentityV1(
        component_id=label,
        descriptor_schema_version=f"alberta.forager_matched_v3.{label}_descriptor.v1",
        descriptor_file_sha256=_hash(label + ":descriptor"),
        source_sha256=_hash(label + ":source"),
        runtime_artifact_sha256=_hash(label + ":runtime"),
    )


def _facts() -> trust.HostFactsInventoryV1:
    return trust.HostFactsInventoryV1(
        kernel=trust.HostKernelIdentityV1(
            host_identity_sha256=_hash("host"),
            machine_id_sha256=_hash("machine"),
            boot_id="01234567-89ab-4def-8123-456789abcdef",
            architecture="x86_64",
            kernel_release="6.8.0-qualified",
            kernel_build_sha256=_hash("kernel-build"),
            kernel_command_line_sha256=_hash("kernel-command-line"),
        ),
        cgroup=trust.CgroupV2IdentityV1(
            mount_path="/sys/fs/cgroup",
            mount_device_major=0,
            mount_device_minor=29,
            mount_inode=101,
            filesystem_magic="0x63677270",
            unified_hierarchy=True,
            delegate_path="/sys/fs/cgroup/alberta-qualified-host",
            delegate_device_major=0,
            delegate_device_minor=29,
            delegate_inode=202,
            delegate_uid=0,
            delegate_gid=0,
            delegate_mode=0o750,
            delegated_controllers=("cpu", "memory", "pids"),
            subtree_control=("cpu", "memory", "pids"),
        ),
        docker=trust.DockerDaemonIdentityV1(
            socket_path="/run/docker.sock",
            socket_device_major=0,
            socket_device_minor=8,
            socket_inode=303,
            socket_uid=0,
            socket_gid=0,
            socket_mode=0o660,
            daemon_id="qualified-daemon-01",
            daemon_pid=404,
            daemon_start_ticks=505,
            rootful=True,
            cgroup_driver="cgroupfs",
            version="29.0.1",
            api_version="1.52",
            config_sha256=_hash("docker-config"),
            root_dir_path="/var/lib/docker-qualified",
            root_dir_device_major=8,
            root_dir_device_minor=1,
            root_dir_inode=606,
        ),
        components=trust.HostComponentInventoryV1(
            oci_runtime=_component("oci-runtime"),
            membership_observer=_component("membership-observer"),
            storage_measurement_producer=_component("storage-producer"),
            storage_terminal_relay=_component("storage-relay"),
            security_profile=_component("security-profile"),
        ),
    )


def _policy(*, facts: trust.HostFactsInventoryV1 | None = None) -> trust.HostTrustPolicyV1:
    expected = _facts() if facts is None else facts
    return trust.HostTrustPolicyV1(
        policy_id="matched-v3-qualified-host-policy-01",
        policy_nonce_sha256=_hash("policy-nonce"),
        qualification_plan=_artifact(PLAN_SCHEMA, "plan"),
        issued_at_unix_ns=1_000,
        valid_from_unix_ns=1_000,
        valid_until_unix_ns=9_000,
        signer_key_id="host-provisioner-ed25519-key-01",
        signer_public_key_sha256=_hash("public-key"),
        independent_verifier=_component("signature-verifier"),
        live_validator=_component("live-validator"),
        supported_host_tuple=trust.SupportedHostTupleV1.from_facts(
            "qualified-linux-docker-cgroupfs-tuple-01", expected
        ),
        expected_facts=expected,
    )


def _statement(
    policy: trust.HostTrustPolicyV1,
    *,
    facts: trust.HostFactsInventoryV1 | None = None,
    signature_hex: str = "ab" * 64,
) -> trust.HostProvisioningStatementV1:
    return trust.HostProvisioningStatementV1(
        policy=trust.host_trust_policy_identity_v1(policy),
        observed_at_unix_ns=2_000,
        observed_at_monotonic_ns=20_000,
        signer_key_id=policy.signer_key_id,
        signer_public_key_sha256=policy.signer_public_key_sha256,
        facts=policy.expected_facts if facts is None else facts,
        signature_hex=signature_hex,
    )


def _verification(
    policy: trust.HostTrustPolicyV1,
    statement: trust.HostProvisioningStatementV1,
    *,
    verifier: trust.PinnedComponentIdentityV1 | None = None,
) -> trust.HostSignatureVerificationReceiptV1:
    return trust.HostSignatureVerificationReceiptV1(
        policy=trust.host_trust_policy_identity_v1(policy),
        statement=trust.host_provisioning_statement_identity_v1(statement),
        verifier=policy.independent_verifier if verifier is None else verifier,
        verification_run_id_sha256=_hash("verification-run"),
        verification_started_at_unix_ns=2_100,
        verification_completed_at_unix_ns=2_200,
        verification_started_at_monotonic_ns=20_100,
        verification_completed_at_monotonic_ns=20_200,
        signer_key_id=statement.signer_key_id,
        signer_public_key_sha256=statement.signer_public_key_sha256,
        signed_payload_sha256=statement.signed_payload_sha256,
        signature_sha256=hashlib.sha256(bytes.fromhex(statement.signature_hex)).hexdigest(),
    )


def _live(
    policy: trust.HostTrustPolicyV1,
    statement: trust.HostProvisioningStatementV1,
    verification: trust.HostSignatureVerificationReceiptV1,
    checkpoint: str,
    *,
    previous: trust.HostLiveValidationReceiptV1 | None,
    facts: trust.HostFactsInventoryV1 | None = None,
) -> trust.HostLiveValidationReceiptV1:
    ordinal = trust.LIVE_VALIDATION_CHECKPOINTS.index(checkpoint)
    return trust.HostLiveValidationReceiptV1(
        checkpoint=checkpoint,
        checkpoint_ordinal=ordinal,
        policy=trust.host_trust_policy_identity_v1(policy),
        statement=trust.host_provisioning_statement_identity_v1(statement),
        signature_verification_receipt=(
            trust.host_signature_verification_receipt_identity_v1(verification)
        ),
        previous_live_validation_receipt=(
            None if previous is None else trust.host_live_validation_receipt_identity_v1(previous)
        ),
        validator=policy.live_validator,
        validation_run_id_sha256=_hash(f"live-validation:{checkpoint}"),
        validated_at_unix_ns=3_000 + ordinal * 1_000,
        validated_at_monotonic_ns=30_000 + ordinal * 1_000,
        facts=statement.facts if facts is None else facts,
    )


def _chain() -> tuple[
    trust.HostTrustPolicyV1,
    trust.HostProvisioningStatementV1,
    trust.HostSignatureVerificationReceiptV1,
    tuple[trust.HostLiveValidationReceiptV1, ...],
]:
    policy = _policy()
    statement = _statement(policy)
    verification = _verification(policy, statement)
    receipts: list[trust.HostLiveValidationReceiptV1] = []
    previous = None
    for checkpoint in trust.LIVE_VALIDATION_CHECKPOINTS:
        current = _live(
            policy,
            statement,
            verification,
            checkpoint,
            previous=previous,
        )
        receipts.append(current)
        previous = current
    return policy, statement, verification, tuple(receipts)


def _rewrite(
    raw: bytes,
    body_field: str,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[bytes, str]:
    payload = json.loads(raw)
    payload.pop(body_field)
    mutate(payload)
    body = trust.canonical_provisioning_json_bytes(payload, trailing_lf=False)
    payload[body_field] = hashlib.sha256(body).hexdigest()
    rewritten = trust.canonical_provisioning_json_bytes(payload, trailing_lf=True)
    return rewritten, hashlib.sha256(rewritten).hexdigest()


def _round_trip(
    value: Any,
    file_builder: Callable[[Any], bytes],
    parser: Callable[..., Any],
) -> None:
    raw = file_builder(value)
    assert parser(raw, expected_file_sha256=hashlib.sha256(raw).hexdigest()) == value


def test_four_artifacts_round_trip_and_complete_chain_is_non_authorizing() -> None:
    policy, statement, verification, receipts = _chain()
    _round_trip(
        policy,
        trust.canonical_host_trust_policy_v1_file_bytes,
        trust.parse_host_trust_policy_v1,
    )
    _round_trip(
        statement,
        trust.canonical_host_provisioning_statement_v1_file_bytes,
        trust.parse_host_provisioning_statement_v1,
    )
    _round_trip(
        verification,
        trust.canonical_host_signature_verification_receipt_v1_file_bytes,
        trust.parse_host_signature_verification_receipt_v1,
    )
    for receipt in receipts:
        _round_trip(
            receipt,
            trust.canonical_host_live_validation_receipt_v1_file_bytes,
            trust.parse_host_live_validation_receipt_v1,
        )

    trust.validate_host_provisioning_trust_chain_v1(
        policy, statement, verification, receipts
    )
    descriptor = trust.HostProvisioningTrustContractDescriptorV1()
    assert descriptor.operational_apis == ()
    assert descriptor.signature_algorithm == "ed25519"
    assert descriptor.signature_domain == trust.ED25519_SIGNATURE_DOMAIN_LABEL
    assert descriptor.live_validation_checkpoints == trust.LIVE_VALIDATION_CHECKPOINTS
    assert trust.REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS == (
        "memory.events:oom",
        "memory.events:oom_kill",
        "pids.events:max",
    )
    assert trust.POSITIVE_RESOURCE_EVENT_DELTA_POLICY == (
        "any_positive_delta_makes_case_resource_ineligible_even_when_worker_exit_zero"
    )
    assert trust.RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY == (
        "retained_initial_and_terminal_unsigned_counters_must_be_monotonic"
    )
    for mapping in (
        descriptor.capabilities,
        descriptor.readiness,
        descriptor.authority,
        descriptor.claims,
    ):
        assert mapping and not any(mapping.values())


def test_descriptor_body_file_digest_parser_and_source_only_surface_are_exact() -> None:
    descriptor = trust.HostProvisioningTrustContractDescriptorV1()
    body = trust.canonical_host_provisioning_trust_contract_descriptor_v1_body_bytes(
        descriptor
    )
    raw = trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
        descriptor
    )
    body_payload = descriptor.to_body_dict()
    file_payload = json.loads(raw)

    assert body == trust.canonical_provisioning_json_bytes(body_payload, trailing_lf=False)
    assert not body.endswith(b"\n")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert set(file_payload) == set(body_payload) | {DESCRIPTOR_BODY_FIELD}
    assert file_payload[DESCRIPTOR_BODY_FIELD] == hashlib.sha256(body).hexdigest()
    with _observed_descriptor_pin() as observed:
        assert trust.parse_host_provisioning_trust_contract_descriptor_v1(
            raw,
            expected_file_sha256=observed,
        ) == descriptor
        assert trust.host_provisioning_v3_descriptor_sha256() == observed
    assert descriptor.safety_posture == {
        "capabilities": trust.SOURCE_ONLY_CAPABILITIES,
        "readiness": trust.SOURCE_ONLY_READINESS,
        "authority": trust.SOURCE_ONLY_AUTHORITY,
        "claims": trust.SOURCE_ONLY_CLAIMS,
    }
    assert all(
        value is False
        for values in descriptor.safety_posture.values()
        for value in values.values()
    )


def test_public_surface_is_exact_and_unique() -> None:
    expected = (
        "HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
        "HOST_TRUST_POLICY_SCHEMA_VERSION",
        "HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION",
        "HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION",
        "HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION",
        "QUALIFICATION_PLAN_V3_SCHEMA_VERSION",
        "ED25519_SIGNATURE_ALGORITHM",
        "ED25519_SIGNATURE_DOMAIN_LABEL",
        "ED25519_SIGNATURE_DOMAIN",
        "LIVE_VALIDATION_CHECKPOINTS",
        "REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS",
        "RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY",
        "POSITIVE_RESOURCE_EVENT_DELTA_POLICY",
        "PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256",
        "SOURCE_ONLY_CAPABILITIES",
        "SOURCE_ONLY_READINESS",
        "SOURCE_ONLY_AUTHORITY",
        "SOURCE_ONLY_CLAIMS",
        "SOURCE_ONLY_SAFETY_POSTURE",
        "ForagerMatchedV3HostProvisioningV3Error",
        "ArtifactIdentityV1",
        "PinnedComponentIdentityV1",
        "HostKernelIdentityV1",
        "CgroupV2IdentityV1",
        "DockerDaemonIdentityV1",
        "HostComponentInventoryV1",
        "HostFactsInventoryV1",
        "SupportedHostTupleV1",
        "HostProvisioningTrustContractDescriptorV1",
        "HostTrustPolicyV1",
        "HostProvisioningStatementV1",
        "HostSignatureVerificationReceiptV1",
        "HostLiveValidationReceiptV1",
        "canonical_provisioning_json_bytes",
        "decode_canonical_provisioning_json",
        "canonical_host_provisioning_trust_contract_descriptor_v1_body_bytes",
        "canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes",
        "canonical_host_trust_policy_v1_body_bytes",
        "canonical_host_trust_policy_v1_file_bytes",
        "canonical_host_provisioning_statement_v1_signed_payload_bytes",
        "canonical_host_provisioning_statement_v1_body_bytes",
        "canonical_host_provisioning_statement_v1_file_bytes",
        "canonical_host_signature_verification_receipt_v1_body_bytes",
        "canonical_host_signature_verification_receipt_v1_file_bytes",
        "canonical_host_live_validation_receipt_v1_body_bytes",
        "canonical_host_live_validation_receipt_v1_file_bytes",
        "host_provisioning_v3_descriptor_sha256",
        "host_trust_policy_identity_v1",
        "host_provisioning_statement_identity_v1",
        "host_signature_verification_receipt_identity_v1",
        "host_live_validation_receipt_identity_v1",
        "parse_host_provisioning_trust_contract_descriptor_v1",
        "parse_host_trust_policy_v1",
        "parse_host_provisioning_statement_v1",
        "parse_host_signature_verification_receipt_v1",
        "parse_host_live_validation_receipt_v1",
        "validate_host_provisioning_statement_against_policy_v1",
        "validate_host_signature_verification_receipt_v1",
        "validate_host_live_validation_receipt_v1",
        "validate_host_provisioning_trust_chain_v1",
    )
    assert trust.__all__ == expected
    assert len(trust.__all__) == len(set(trust.__all__))
    assert all(getattr(trust, name) is not None for name in trust.__all__)


def test_statement_signed_payload_is_domain_separated_but_not_verified_by_parser() -> None:
    policy = _policy()
    statement = _statement(policy)
    signed = trust.canonical_host_provisioning_statement_v1_signed_payload_bytes(statement)
    assert signed.startswith(trust.ED25519_SIGNATURE_DOMAIN)
    assert hashlib.sha256(signed).hexdigest() == statement.signed_payload_sha256
    parsed = trust.parse_host_provisioning_statement_v1(
        trust.canonical_host_provisioning_statement_v1_file_bytes(statement),
        expected_file_sha256=trust.host_provisioning_statement_identity_v1(
            statement
        ).file_sha256,
    )
    assert parsed.signature_hex == "ab" * 64
    assert (
        parsed.safety_posture["claims"]["signature_cryptographically_verified_by_parser"]
        is False
    )


def test_statement_signed_payload_obeys_exact_ed25519_domain_byte_equation() -> None:
    statement = _statement(_policy())
    unsigned = trust.canonical_provisioning_json_bytes(
        statement.to_unsigned_dict(),
        trailing_lf=False,
    )
    expected_domain = trust.ED25519_SIGNATURE_DOMAIN_LABEL.encode("ascii") + b"\x00"
    payload = trust.canonical_host_provisioning_statement_v1_signed_payload_bytes(
        statement
    )

    assert trust.ED25519_SIGNATURE_DOMAIN == expected_domain
    assert payload == expected_domain + unsigned
    assert payload.count(b"\x00") == 1
    assert not unsigned.endswith(b"\n")
    assert statement.signed_payload_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("builder", "parser"),
    (
        (
            trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes,
            trust.parse_host_provisioning_trust_contract_descriptor_v1,
        ),
        (
            trust.canonical_host_trust_policy_v1_file_bytes,
            trust.parse_host_trust_policy_v1,
        ),
        (
            trust.canonical_host_provisioning_statement_v1_file_bytes,
            trust.parse_host_provisioning_statement_v1,
        ),
        (
            trust.canonical_host_signature_verification_receipt_v1_file_bytes,
            trust.parse_host_signature_verification_receipt_v1,
        ),
        (
            trust.canonical_host_live_validation_receipt_v1_file_bytes,
            trust.parse_host_live_validation_receipt_v1,
        ),
    ),
)
def test_parsers_require_correct_nonzero_caller_file_pin(
    builder: Callable[[Any], bytes], parser: Callable[..., Any]
) -> None:
    policy, statement, verification, receipts = _chain()
    values = {
        trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes: (
            trust.HostProvisioningTrustContractDescriptorV1()
        ),
        trust.canonical_host_trust_policy_v1_file_bytes: policy,
        trust.canonical_host_provisioning_statement_v1_file_bytes: statement,
        trust.canonical_host_signature_verification_receipt_v1_file_bytes: verification,
        trust.canonical_host_live_validation_receipt_v1_file_bytes: receipts[0],
    }
    raw = builder(values[builder])
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(raw, expected_file_sha256=_hash("wrong-file"))
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(raw, expected_file_sha256=ZERO_SHA256)


@pytest.mark.parametrize(
    ("value_factory", "builder", "parser", "body_field"),
    (
        (
            lambda: trust.HostProvisioningTrustContractDescriptorV1(),
            trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes,
            trust.parse_host_provisioning_trust_contract_descriptor_v1,
            DESCRIPTOR_BODY_FIELD,
        ),
        (
            lambda: _policy(),
            trust.canonical_host_trust_policy_v1_file_bytes,
            trust.parse_host_trust_policy_v1,
            POLICY_BODY_FIELD,
        ),
        (
            lambda: _statement(_policy()),
            trust.canonical_host_provisioning_statement_v1_file_bytes,
            trust.parse_host_provisioning_statement_v1,
            STATEMENT_BODY_FIELD,
        ),
        (
            lambda: _verification(_policy(), _statement(_policy())),
            trust.canonical_host_signature_verification_receipt_v1_file_bytes,
            trust.parse_host_signature_verification_receipt_v1,
            VERIFICATION_BODY_FIELD,
        ),
        (
            lambda: _chain()[3][0],
            trust.canonical_host_live_validation_receipt_v1_file_bytes,
            trust.parse_host_live_validation_receipt_v1,
            LIVE_BODY_FIELD,
        ),
    ),
)
def test_schema_keys_and_body_digest_fail_closed(
    value_factory: Callable[[], Any],
    builder: Callable[[Any], bytes],
    parser: Callable[..., Any],
    body_field: str,
) -> None:
    raw = builder(value_factory())
    wrong_schema, pin = _rewrite(
        raw,
        body_field,
        lambda body: body.__setitem__("schema_version", "wrong.v1"),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(wrong_schema, expected_file_sha256=pin)
    extra, pin = _rewrite(raw, body_field, lambda body: body.__setitem__("extra", False))
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(extra, expected_file_sha256=pin)

    payload = json.loads(raw)
    payload["status"] = "tampered-without-body-rewrite"
    tampered = trust.canonical_provisioning_json_bytes(payload, trailing_lf=True)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(tampered, expected_file_sha256=hashlib.sha256(tampered).hexdigest())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("status", "operational"),
        ("artifact_schema_versions", [trust.HOST_TRUST_POLICY_SCHEMA_VERSION]),
        ("signature_algorithm", "hmac-sha256"),
        ("signature_domain", "wrong-domain"),
        ("signature_length_bytes", 32),
        ("live_validation_checkpoints", list(reversed(trust.LIVE_VALIDATION_CHECKPOINTS))),
        (
            "required_retained_resource_event_counters",
            ["memory.events:oom", "pids.events:max"],
        ),
        ("resource_event_counter_monotonicity_policy", "allow_counter_reset"),
        ("positive_resource_event_delta_policy", "ignore_positive_delta"),
        ("operational_apis", ["execute"]),
        ("executor_signing_secret_policy", "executor_holds_signing_secret"),
        ("hmac_policy", "allowed"),
        ("parsing_semantics", "parsing_verifies_signature"),
        ("audit_pin_state", "audited"),
    ),
)
def test_descriptor_parser_rejects_every_contract_literal_drift(
    field: str,
    replacement: object,
) -> None:
    raw = trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
        trust.HostProvisioningTrustContractDescriptorV1()
    )
    corrupted, pin = _rewrite(
        raw,
        DESCRIPTOR_BODY_FIELD,
        lambda body: body.__setitem__(field, replacement),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_provisioning_trust_contract_descriptor_v1(
            corrupted,
            expected_file_sha256=pin,
        )


@pytest.mark.parametrize(
    ("section", "key"),
    tuple(
        (section, key)
        for section, values in trust.SOURCE_ONLY_SAFETY_POSTURE.items()
        for key in values
    ),
)
def test_descriptor_parser_rejects_every_true_safety_claim(
    section: str,
    key: str,
) -> None:
    raw = trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
        trust.HostProvisioningTrustContractDescriptorV1()
    )

    def mutate(body: dict[str, Any]) -> None:
        body["safety_posture"][section][key] = True

    corrupted, pin = _rewrite(raw, DESCRIPTOR_BODY_FIELD, mutate)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_provisioning_trust_contract_descriptor_v1(
            corrupted,
            expected_file_sha256=pin,
        )


@pytest.mark.parametrize(
    "raw",
    (
        b'{"a":1,"a":1}\n',
        b'{"a":1.0}\n',
        b'{"a":NaN}\n',
        b'{"a":"\xc3\xa9"}\n',
        b'{"a":1}',
        b' {"a":1}\n',
        b'[]\n',
    ),
)
def test_strict_ascii_canonical_json_rejects_duplicates_floats_and_noncanonical_bytes(
    raw: bytes,
) -> None:
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.decode_canonical_provisioning_json(raw)


def test_canonical_builder_rejects_aliases_cycles_and_non_plain_values() -> None:
    shared: list[object] = []
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes(
            {"first": shared, "second": shared}, trailing_lf=False
        )
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes({"cycle": cycle}, trailing_lf=False)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes({"set": {1}}, trailing_lf=False)


@pytest.mark.parametrize("trailing_lf", (None, 0, 1, "yes", (), []))
def test_canonical_builder_requires_trailing_lf_to_be_one_exact_bool(
    trailing_lf: object,
) -> None:
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes(
            {"bounded": True},
            trailing_lf=trailing_lf,  # type: ignore[arg-type]
        )


def test_json_recursion_errors_are_normalized_to_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recurse(*args: object, **kwargs: object) -> object:
        raise RecursionError("synthetic recursion exhaustion")

    monkeypatch.setattr(json, "loads", recurse)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.decode_canonical_provisioning_json(b'{}\n')

    monkeypatch.undo()
    monkeypatch.setattr(json, "dumps", recurse)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes({}, trailing_lf=False)


def test_json_depth_node_and_file_bounds_fail_closed() -> None:
    nested: object = 0
    for _ in range(65):
        nested = [nested]
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes({"nested": nested}, trailing_lf=False)

    too_many_nodes = {"nodes": [0] * 100_000}
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes(too_many_nodes, trailing_lf=False)

    deep_raw = b'{"nested":' + b"[" * 65 + b"0" + b"]" * 65 + b"}\n"
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.decode_canonical_provisioning_json(deep_raw)

    many_nodes_raw = json.dumps(
        too_many_nodes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.decode_canonical_provisioning_json(many_nodes_raw)

    oversized = b"{" + b" " * (4 * 1024 * 1024) + b"}\n"
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.decode_canonical_provisioning_json(oversized)


def test_artifact_parser_rejects_oversized_bytes_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_hash(raw: bytes) -> str:
        raise AssertionError(f"hashed oversized input of {len(raw)} bytes")

    monkeypatch.setattr(trust, "_sha256", forbidden_hash)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(
            b"x" * (4 * 1024 * 1024 + 1),
            expected_file_sha256=_hash("oversized-file-pin"),
        )


def test_canonical_builder_enforces_text_integer_and_hash_sentinel_bounds() -> None:
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes(
            {"text": "x" * 16_385},
            trailing_lf=False,
        )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.canonical_provisioning_json_bytes(
            {"integer": 2**63},
            trailing_lf=False,
        )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.ArtifactIdentityV1(PLAN_SCHEMA, ZERO_SHA256, _hash("body"))
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(_component("zero-source"), source_sha256=ZERO_SHA256)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(_policy(), policy_nonce_sha256=ZERO_SHA256)


@pytest.mark.parametrize("signature", ("ab" * 63, "ab" * 65, "AB" * 64, "00" * 64))
def test_signature_metadata_requires_exact_nonzero_64_byte_lowercase_ed25519_signature(
    signature: str,
) -> None:
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        _statement(_policy(), signature_hex=signature)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("signature_algorithm", "hmac-sha256"),
        ("signature_algorithm", "ed448"),
        ("signature_domain", "alberta.forager_matched_v3.other-domain.v1"),
        ("hmac_used", True),
        ("executor_held_signing_secret", True),
        ("signature_verified_by_parser", True),
    ),
)
def test_statement_rejects_algorithm_domain_key_hmac_and_false_claim_drift(
    field: str, replacement: object
) -> None:
    statement = _statement(_policy())
    raw = trust.canonical_host_provisioning_statement_v1_file_bytes(statement)
    corrupted, pin = _rewrite(
        raw,
        STATEMENT_BODY_FIELD,
        lambda body: body.__setitem__(field, replacement),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_provisioning_statement_v1(corrupted, expected_file_sha256=pin)


@pytest.mark.parametrize(
    "statement",
    (
        replace(_statement(_policy()), signer_key_id="other-key"),
        replace(
            _statement(_policy()),
            signer_public_key_sha256=_hash("other-public-key"),
        ),
    ),
)
def test_statement_key_crosswire_is_rejected_by_chain_not_structural_parser(
    statement: trust.HostProvisioningStatementV1,
) -> None:
    policy = _policy()
    raw = trust.canonical_host_provisioning_statement_v1_file_bytes(statement)
    parsed = trust.parse_host_provisioning_statement_v1(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_provisioning_statement_against_policy_v1(policy, parsed)


def test_structural_statement_parser_does_not_substitute_for_chain_validation() -> None:
    policy = _policy()
    statement = _statement(
        policy,
        facts=replace(
            policy.expected_facts,
            kernel=replace(
                policy.expected_facts.kernel,
                boot_id="fedcba98-7654-4abc-9234-56789abcdef0",
            ),
        ),
    )
    raw = trust.canonical_host_provisioning_statement_v1_file_bytes(statement)
    trust.parse_host_provisioning_statement_v1(
        raw, expected_file_sha256=hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_provisioning_statement_against_policy_v1(policy, statement)


def test_signature_verification_receipt_rejects_crosswired_verifier_and_statement() -> None:
    policy = _policy()
    statement = _statement(policy)
    wrong_verifier = _verification(
        policy, statement, verifier=_component("untrusted-verifier")
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_signature_verification_receipt_v1(
            policy, statement, wrong_verifier
        )

    receipt = _verification(policy, statement)
    crosswired = replace(
        receipt,
        statement=_artifact(
            trust.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
            "other-statement",
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_signature_verification_receipt_v1(
            policy, statement, crosswired
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "policy",
            _artifact(trust.HOST_TRUST_POLICY_SCHEMA_VERSION, "other-receipt-policy"),
        ),
        ("signer_key_id", "other-signer-key"),
        ("signer_public_key_sha256", _hash("other-receipt-public-key")),
        ("signed_payload_sha256", _hash("other-signed-payload")),
        ("signature_sha256", _hash("other-signature")),
    ),
)
def test_signature_verification_receipt_rejects_every_material_crosswire(
    field: str,
    replacement: object,
) -> None:
    policy = _policy()
    statement = _statement(policy)
    receipt = replace(
        _verification(policy, statement),
        **{field: replacement},  # type: ignore[arg-type]
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_signature_verification_receipt_v1(
            policy,
            statement,
            receipt,
        )


def test_signature_verification_chronology_cannot_precede_statement() -> None:
    policy = _policy()
    statement = _statement(policy)
    receipt = replace(
        _verification(policy, statement),
        verification_started_at_unix_ns=statement.observed_at_unix_ns - 1,
        verification_started_at_monotonic_ns=statement.observed_at_monotonic_ns - 1,
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_signature_verification_receipt_v1(
            policy,
            statement,
            receipt,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("verification_started_at_unix_ns", 1_999),
        ("verification_started_at_monotonic_ns", 19_999),
        ("verification_completed_at_unix_ns", 9_001),
    ),
)
def test_signature_verification_boundary_chronology_checks_each_clock(
    field: str,
    value: int,
) -> None:
    policy = _policy()
    statement = _statement(policy)
    receipt = replace(
        _verification(policy, statement),
        **{field: value},  # type: ignore[arg-type]
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_signature_verification_receipt_v1(
            policy,
            statement,
            receipt,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("signature_algorithm", "hmac-sha256"),
        ("signature_domain", "alberta.forager_matched_v3.wrong-domain.v1"),
        ("verification_method", "caller_assertion"),
        ("cryptographic_verification_performed_by_parser", True),
    ),
)
def test_verification_receipt_rejects_algorithm_domain_method_or_parser_claim(
    field: str,
    replacement: object,
) -> None:
    policy = _policy()
    statement = _statement(policy)
    raw = trust.canonical_host_signature_verification_receipt_v1_file_bytes(
        _verification(policy, statement)
    )
    corrupted, pin = _rewrite(
        raw,
        VERIFICATION_BODY_FIELD,
        lambda body: body.__setitem__(field, replacement),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_signature_verification_receipt_v1(
            corrupted,
            expected_file_sha256=pin,
        )


@pytest.mark.parametrize(
    "drift",
    (
        "host_identity",
        "machine_id",
        "boot",
        "architecture",
        "kernel_release",
        "kernel",
        "kernel_command_line",
        "daemon",
        "daemon_id",
        "daemon_pid",
        "socket",
        "socket_path",
        "socket_device",
        "socket_uid",
        "socket_gid",
        "socket_mode",
        "mount",
        "mount_inode",
        "delegate_inode",
        "delegate_uid",
        "delegate_gid",
        "delegate_mode",
        "docker_version",
        "docker_api_version",
        "docker_config",
        "docker_root_path",
        "docker_root_device",
        "docker_root_inode",
        "runtime",
        "observer",
        "storage_producer",
        "storage_relay",
        "security_profile",
    ),
)
def test_live_validation_rejects_every_material_fact_drift(drift: str) -> None:
    policy, statement, verification, receipts = _chain()
    facts = statement.facts
    kernel_fields: dict[str, object] = {
        "host_identity": _hash("drift-host-identity"),
        "machine_id": _hash("drift-machine-id"),
        "boot": "fedcba98-7654-4abc-9234-56789abcdef0",
        "architecture": "aarch64",
        "kernel_release": "6.8.1-qualified",
        "kernel": _hash("drift-kernel"),
        "kernel_command_line": _hash("drift-kernel-command-line"),
    }
    kernel_field_names = {
        "host_identity": "host_identity_sha256",
        "machine_id": "machine_id_sha256",
        "boot": "boot_id",
        "architecture": "architecture",
        "kernel_release": "kernel_release",
        "kernel": "kernel_build_sha256",
        "kernel_command_line": "kernel_command_line_sha256",
    }
    docker_fields: dict[str, object] = {
        "daemon": 999,
        "daemon_id": "drift-daemon-id",
        "daemon_pid": 999,
        "socket": 999,
        "socket_path": "/run/drift-docker.sock",
        "socket_uid": 1,
        "socket_gid": 1,
        "socket_mode": 0o600,
        "docker_version": "29.0.2",
        "docker_api_version": "1.53",
        "docker_config": _hash("drift-docker-config"),
        "docker_root_path": "/var/lib/drift-docker",
        "docker_root_inode": 999,
    }
    docker_field_names = {
        "daemon": "daemon_start_ticks",
        "daemon_id": "daemon_id",
        "daemon_pid": "daemon_pid",
        "socket": "socket_inode",
        "socket_path": "socket_path",
        "socket_uid": "socket_uid",
        "socket_gid": "socket_gid",
        "socket_mode": "socket_mode",
        "docker_version": "version",
        "docker_api_version": "api_version",
        "docker_config": "config_sha256",
        "docker_root_path": "root_dir_path",
        "docker_root_inode": "root_dir_inode",
    }
    if drift in kernel_fields:
        changed = replace(
            facts,
            kernel=replace(
                facts.kernel,
                **{  # type: ignore[arg-type]
                    kernel_field_names[drift]: kernel_fields[drift]
                },
            ),
        )
    elif drift in docker_fields:
        changed = replace(
            facts,
            docker=replace(
                facts.docker,
                **{  # type: ignore[arg-type]
                    docker_field_names[drift]: docker_fields[drift]
                },
            ),
        )
    elif drift == "socket_device":
        changed = replace(
            facts,
            docker=replace(facts.docker, socket_device_major=1, socket_device_minor=9),
        )
    elif drift == "docker_root_device":
        changed = replace(
            facts,
            docker=replace(
                facts.docker,
                root_dir_device_major=9,
                root_dir_device_minor=2,
            ),
        )
    elif drift == "mount":
        changed = replace(
            facts,
            cgroup=replace(
                facts.cgroup,
                mount_device_minor=30,
                delegate_device_minor=30,
            ),
        )
    elif drift == "mount_inode":
        changed = replace(facts, cgroup=replace(facts.cgroup, mount_inode=999))
    elif drift == "delegate_inode":
        changed = replace(facts, cgroup=replace(facts.cgroup, delegate_inode=999))
    elif drift == "delegate_uid":
        changed = replace(facts, cgroup=replace(facts.cgroup, delegate_uid=1))
    elif drift == "delegate_gid":
        changed = replace(facts, cgroup=replace(facts.cgroup, delegate_gid=1))
    elif drift == "delegate_mode":
        changed = replace(facts, cgroup=replace(facts.cgroup, delegate_mode=0o700))
    else:
        field = {
            "runtime": "oci_runtime",
            "observer": "membership_observer",
            "storage_producer": "storage_measurement_producer",
            "storage_relay": "storage_terminal_relay",
            "security_profile": "security_profile",
        }[drift]
        changed_components = replace(facts.components, **{field: _component("drift-" + drift)})
        changed = replace(facts, components=changed_components)
    corrupted = replace(receipts[2], facts=changed)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            corrupted,
            previous_receipt=receipts[1],
        )


@pytest.mark.parametrize(
    ("first", "second"),
    (
        ("oci_runtime", "membership_observer"),
        ("oci_runtime", "storage_measurement_producer"),
        ("oci_runtime", "storage_terminal_relay"),
        ("oci_runtime", "security_profile"),
        ("membership_observer", "storage_measurement_producer"),
        ("membership_observer", "storage_terminal_relay"),
        ("membership_observer", "security_profile"),
        ("storage_measurement_producer", "storage_terminal_relay"),
        ("storage_measurement_producer", "security_profile"),
        ("storage_terminal_relay", "security_profile"),
    ),
)
def test_host_component_inventory_rejects_every_duplicate_role_pair(
    first: str,
    second: str,
) -> None:
    components = _facts().components
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(components, **{second: getattr(components, first)})


def test_policy_separates_verifier_validator_and_material_host_component_roles() -> None:
    policy = _policy()
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(policy, live_validator=policy.independent_verifier)

    for component in policy.expected_facts.components.to_dict():
        reused_id = getattr(policy.expected_facts.components, component).component_id
        with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
            replace(
                policy,
                independent_verifier=replace(
                    policy.independent_verifier,
                    component_id=reused_id,
                ),
            )


@pytest.mark.parametrize(
    "crosswire",
    (
        "policy",
        "statement",
        "verification",
        "validator",
    ),
)
def test_current_live_receipt_rejects_every_identity_or_role_crosswire(
    crosswire: str,
) -> None:
    policy, statement, verification, receipts = _chain()
    replacements: dict[str, object] = {
        "policy": {
            "policy": _artifact(trust.HOST_TRUST_POLICY_SCHEMA_VERSION, "other-policy")
        },
        "statement": {
            "statement": _artifact(
                trust.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
                "other-statement",
            )
        },
        "verification": {
            "signature_verification_receipt": _artifact(
                trust.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
                "other-verification",
            )
        },
        "validator": {"validator": _component("other-live-validator")},
    }
    corrupted = replace(receipts[2], **replacements[crosswire])  # type: ignore[arg-type]
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            corrupted,
            previous_receipt=receipts[1],
        )


@pytest.mark.parametrize(
    "crosswire",
    (
        "policy",
        "statement",
        "verification",
        "validator",
        "facts",
    ),
)
def test_standalone_live_validation_rejects_semantically_crosswired_predecessor(
    crosswire: str,
) -> None:
    policy, statement, verification, receipts = _chain()
    drifted_facts = replace(
        statement.facts,
        kernel=replace(
            statement.facts.kernel,
            host_identity_sha256=_hash("predecessor-host-drift"),
        ),
    )
    replacements: dict[str, dict[str, object]] = {
        "policy": {
            "policy": _artifact(trust.HOST_TRUST_POLICY_SCHEMA_VERSION, "previous-policy")
        },
        "statement": {
            "statement": _artifact(
                trust.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
                "previous-statement",
            )
        },
        "verification": {
            "signature_verification_receipt": _artifact(
                trust.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
                "previous-verification",
            )
        },
        "validator": {"validator": _component("previous-validator")},
        "facts": {"facts": drifted_facts},
    }
    previous = replace(receipts[1], **replacements[crosswire])  # type: ignore[arg-type]
    current = replace(
        receipts[2],
        previous_live_validation_receipt=(
            trust.host_live_validation_receipt_identity_v1(previous)
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            current,
            previous_receipt=previous,
        )


@pytest.mark.parametrize("clock", ("unix", "monotonic"))
def test_standalone_live_validation_rechecks_predecessor_genesis_chronology(
    clock: str,
) -> None:
    policy, statement, verification, receipts = _chain()
    field = (
        "validated_at_unix_ns" if clock == "unix" else "validated_at_monotonic_ns"
    )
    completion = (
        verification.verification_completed_at_unix_ns
        if clock == "unix"
        else verification.verification_completed_at_monotonic_ns
    )
    previous = replace(
        receipts[0],
        **{field: completion - 1},  # type: ignore[arg-type]
    )
    current = replace(
        receipts[1],
        previous_live_validation_receipt=(
            trust.host_live_validation_receipt_identity_v1(previous)
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            current,
            previous_receipt=previous,
        )


def test_checkpoints_order_previous_links_and_chronology_are_exact() -> None:
    policy, statement, verification, receipts = _chain()
    wrong_order = (receipts[1], receipts[0], receipts[2], receipts[3])
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_provisioning_trust_chain_v1(
            policy, statement, verification, wrong_order
        )

    wrong_previous = replace(
        receipts[2], previous_live_validation_receipt=_artifact(
            trust.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION, "crosswire"
        )
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            wrong_previous,
            previous_receipt=receipts[1],
        )

    nonmonotonic_unix = replace(
        receipts[2],
        validated_at_unix_ns=receipts[1].validated_at_unix_ns,
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            nonmonotonic_unix,
            previous_receipt=receipts[1],
        )


@pytest.mark.parametrize("clock", ("unix", "monotonic"))
def test_live_chain_genesis_cannot_precede_signature_verification(clock: str) -> None:
    policy, statement, verification, receipts = _chain()
    field = (
        "validated_at_unix_ns" if clock == "unix" else "validated_at_monotonic_ns"
    )
    completion = (
        verification.verification_completed_at_unix_ns
        if clock == "unix"
        else verification.verification_completed_at_monotonic_ns
    )
    genesis = replace(
        receipts[0],
        **{field: completion - 1},  # type: ignore[arg-type]
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            genesis,
            previous_receipt=None,
        )


@pytest.mark.parametrize(
    "receipts_factory",
    (
        lambda receipts: receipts[:-1],
        lambda receipts: receipts + (receipts[-1],),
        lambda receipts: list(receipts),
        lambda receipts: (),
    ),
)
def test_complete_chain_requires_exact_tuple_type_and_cardinality(
    receipts_factory: Callable[[tuple[trust.HostLiveValidationReceiptV1, ...]], object],
) -> None:
    policy, statement, verification, receipts = _chain()
    corrupted = receipts_factory(receipts)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_provisioning_trust_chain_v1(
            policy,
            statement,
            verification,
            corrupted,  # type: ignore[arg-type]
        )

    nonmonotonic = replace(
        receipts[2],
        validated_at_monotonic_ns=receipts[1].validated_at_monotonic_ns,
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.validate_host_live_validation_receipt_v1(
            policy,
            statement,
            verification,
            nonmonotonic,
            previous_receipt=receipts[1],
        )


def test_policy_host_tuple_and_validity_chronology_fail_closed() -> None:
    facts = _facts()
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(
            _policy(facts=facts),
            supported_host_tuple=replace(
                trust.SupportedHostTupleV1.from_facts("tuple", facts),
                docker_version="different",
            ),
        )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        replace(_policy(), valid_until_unix_ns=999)


def test_safety_maps_are_immutable_and_any_true_claim_is_rejected() -> None:
    descriptor = trust.HostProvisioningTrustContractDescriptorV1()
    with pytest.raises(TypeError):
        descriptor.capabilities["executes_host_operations"] = True  # type: ignore[index]

    policy = _policy()
    raw = trust.canonical_host_trust_policy_v1_file_bytes(policy)

    def mutate(body: dict[str, Any]) -> None:
        body["safety_posture"]["claims"]["host_facts_true_by_parsing"] = True

    corrupted, pin = _rewrite(raw, POLICY_BODY_FIELD, mutate)
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(corrupted, expected_file_sha256=pin)


def test_policy_rejects_resource_event_counter_or_positive_delta_rule_drift() -> None:
    raw = trust.canonical_host_trust_policy_v1_file_bytes(_policy())
    wrong_counters, pin = _rewrite(
        raw,
        POLICY_BODY_FIELD,
        lambda body: body.__setitem__(
            "required_executor_handoff_resource_event_counters",
            ["memory.events:oom", "pids.events:max"],
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(wrong_counters, expected_file_sha256=pin)
    wrong_rule, pin = _rewrite(
        raw,
        POLICY_BODY_FIELD,
        lambda body: body.__setitem__(
            "positive_resource_event_delta_policy",
            "worker_exit_zero_overrides_positive_delta",
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(wrong_rule, expected_file_sha256=pin)
    wrong_monotonicity, pin = _rewrite(
        raw,
        POLICY_BODY_FIELD,
        lambda body: body.__setitem__(
            "resource_event_counter_monotonicity_policy",
            "terminal_counter_may_decrease",
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(
            wrong_monotonicity,
            expected_file_sha256=pin,
        )


@pytest.mark.parametrize(
    ("value_factory", "builder", "parser", "body_field", "field", "replacement"),
    (
        (
            lambda: trust.HostProvisioningTrustContractDescriptorV1(),
            trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes,
            trust.parse_host_provisioning_trust_contract_descriptor_v1,
            DESCRIPTOR_BODY_FIELD,
            "hmac_policy",
            "allowed",
        ),
        (
            lambda: _policy(),
            trust.canonical_host_trust_policy_v1_file_bytes,
            trust.parse_host_trust_policy_v1,
            POLICY_BODY_FIELD,
            "hmac_allowed",
            True,
        ),
        (
            lambda: _statement(_policy()),
            trust.canonical_host_provisioning_statement_v1_file_bytes,
            trust.parse_host_provisioning_statement_v1,
            STATEMENT_BODY_FIELD,
            "hmac_used",
            True,
        ),
        (
            lambda: _chain()[2],
            trust.canonical_host_signature_verification_receipt_v1_file_bytes,
            trust.parse_host_signature_verification_receipt_v1,
            VERIFICATION_BODY_FIELD,
            "hmac_used",
            True,
        ),
    ),
)
def test_every_artifact_that_mentions_hmac_enforces_its_exact_prohibition(
    value_factory: Callable[[], Any],
    builder: Callable[[Any], bytes],
    parser: Callable[..., Any],
    body_field: str,
    field: str,
    replacement: object,
) -> None:
    raw = builder(value_factory())
    corrupted, pin = _rewrite(
        raw,
        body_field,
        lambda body: body.__setitem__(field, replacement),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        parser(corrupted, expected_file_sha256=pin)


@pytest.mark.parametrize(
    "replacement",
    (
        list(reversed(trust.REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS)),
        list(trust.REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS) + ["cpu.events:throttled"],
        ["memory.events:oom", "pids.events:max"],
        ["memory.events:oom", "memory.events:oom", "pids.events:max"],
    ),
)
def test_descriptor_and_policy_require_exact_ordered_resource_counter_contract(
    replacement: list[str],
) -> None:
    descriptor_raw = (
        trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
            trust.HostProvisioningTrustContractDescriptorV1()
        )
    )
    corrupted, pin = _rewrite(
        descriptor_raw,
        DESCRIPTOR_BODY_FIELD,
        lambda body: body.__setitem__(
            "required_retained_resource_event_counters",
            replacement,
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_provisioning_trust_contract_descriptor_v1(
            corrupted,
            expected_file_sha256=pin,
        )

    policy_raw = trust.canonical_host_trust_policy_v1_file_bytes(_policy())
    corrupted, pin = _rewrite(
        policy_raw,
        POLICY_BODY_FIELD,
        lambda body: body.__setitem__(
            "required_executor_handoff_resource_event_counters",
            replacement,
        ),
    )
    with pytest.raises(trust.ForagerMatchedV3HostProvisioningV3Error):
        trust.parse_host_trust_policy_v1(corrupted, expected_file_sha256=pin)


def test_descriptor_file_pin_is_final_and_source_identity_remains_external() -> None:
    descriptor = trust.HostProvisioningTrustContractDescriptorV1()
    body = trust.canonical_host_provisioning_trust_contract_descriptor_v1_body_bytes(
        descriptor
    )
    raw = trust.canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
        descriptor
    )
    assert hashlib.sha256(body).hexdigest() == EXPECTED_DESCRIPTOR_BODY_SHA256
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_DESCRIPTOR_FILE_SHA256
    assert (
        trust.PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256
        == EXPECTED_DESCRIPTOR_FILE_SHA256
    )
    assert trust.host_provisioning_v3_descriptor_sha256() == EXPECTED_DESCRIPTOR_FILE_SHA256
    assert not hasattr(trust, "PINNED_HOST_PROVISIONING_V3_SOURCE_SHA256")
    assert trust.HostProvisioningTrustContractDescriptorV1().audit_pin_state == (
        "descriptor_file_pin_required_source_identity_external"
    )


def test_source_is_pure_and_contains_no_operational_or_secret_holding_imports() -> None:
    source_path = Path(trust.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert imported.isdisjoint(
        {
            "asyncio",
            "cryptography",
            "docker",
            "nacl",
            "os",
            "pathlib",
            "socket",
            "subprocess",
            "tempfile",
        }
    )
    assert calls.isdisjoint(
        {
            "open",
            "system",
            "popen",
            "run",
            "Popen",
            "fork",
            "execve",
            "mount",
            "kill",
        }
    )
