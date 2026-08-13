"""Tests for the additive compiled-PPO qualification content overlay."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_qualification_addendum as addendum,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_plan as base_qualification,
)
from tests import test_forager_matched_v3_qualification_plan as base_test

_DESCRIPTOR_SHA256 = "b5f7df77cd3f6e35126ed7c9f4b7acacdaa8237e8242241f658a95d21e9e3b06"
_BASE_DESCRIPTOR_SHA256 = "258b9e376b82127f912bf2828a6d4e5c7a257ed2a990cd15bf4c9cbd81c17788"
_BASE_SOURCE_SHA256 = "d84eb2322dc902dc912e79d9b14295f5d580bcdedf3e8870027854ca344e1ebf"
_CONFIGURATION_PLAN_SHA256 = "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
_CONFIGURATION_RECORD_SHA256 = "4f8b429ff968213d0c05de87553456be7f2c1a67a806944357543025d725d7ca"
_COMPILED_RUNNER_DESCRIPTOR_SHA256 = (
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
)
_COMPILED_RUNNER_SOURCE_SHA256 = "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f"
_COMPILED_BUNDLE_DESCRIPTOR_SHA256 = (
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
)
_COMPILED_BUNDLE_SOURCE_SHA256 = "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e"
_COMPILED_PUBLICATION_DESCRIPTOR_SHA256 = (
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500"
)
_COMPILED_PUBLICATION_SOURCE_SHA256 = (
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71"
)
_BASE_PUBLICATION_DESCRIPTOR_SHA256 = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)


class SyntheticArtifacts(NamedTuple):
    base_plan: dict[str, Any]
    base_raw: bytes
    base_sha256: str
    receipt_file_sha256: str
    receipt_binding_sha256: str
    value: dict[str, Any]
    raw: bytes
    sha256: str


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


@lru_cache(maxsize=1)
def _synthetic_artifacts() -> SyntheticArtifacts:
    inputs = base_test._synthetic_inputs()
    resources = list(inputs.resources)
    index = [item.candidate_id for item in resources].index("adapted_ppo_gru")
    resources[index] = dataclasses.replace(
        resources[index],
        max_optimizer_updates=15_616,
        max_gradient_updates=15_616,
    )
    inputs = inputs._replace(resources=tuple(resources))
    base_plan = base_test._build(inputs)
    base_raw = base_qualification.canonical_matched_v3_qualification_plan_bytes(
        base_plan,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            inputs.expected_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            inputs.expected_trust_root_receipt_binding_sha256
        ),
    )
    base_sha256 = hashlib.sha256(base_raw).hexdigest()
    value = addendum.build_compiled_ppo_qualification_addendum(
        base_plan_raw=base_raw,
        expected_base_plan_file_sha256=base_sha256,
        expected_base_seed_trust_receipt_file_sha256=(
            inputs.expected_trust_root_receipt_file_sha256
        ),
        expected_base_seed_trust_receipt_binding_sha256=(
            inputs.expected_trust_root_receipt_binding_sha256
        ),
    )
    raw = addendum.canonical_compiled_ppo_qualification_addendum_bytes(
        value,
        base_plan_raw=base_raw,
        expected_base_plan_file_sha256=base_sha256,
        expected_base_seed_trust_receipt_file_sha256=(
            inputs.expected_trust_root_receipt_file_sha256
        ),
        expected_base_seed_trust_receipt_binding_sha256=(
            inputs.expected_trust_root_receipt_binding_sha256
        ),
    )
    return SyntheticArtifacts(
        base_plan=base_plan,
        base_raw=base_raw,
        base_sha256=base_sha256,
        receipt_file_sha256=inputs.expected_trust_root_receipt_file_sha256,
        receipt_binding_sha256=inputs.expected_trust_root_receipt_binding_sha256,
        value=value,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _parse(raw: bytes, *, expected_file_sha256: str | None = None) -> dict[str, Any]:
    artifacts = _synthetic_artifacts()
    return addendum.parse_compiled_ppo_qualification_addendum_artifact(
        raw,
        expected_file_sha256=(
            hashlib.sha256(raw).hexdigest()
            if expected_file_sha256 is None
            else expected_file_sha256
        ),
        base_plan_raw=artifacts.base_raw,
        expected_base_plan_file_sha256=artifacts.base_sha256,
        expected_base_seed_trust_receipt_file_sha256=artifacts.receipt_file_sha256,
        expected_base_seed_trust_receipt_binding_sha256=artifacts.receipt_binding_sha256,
    )


def _coherently_rehash(value: dict[str, Any]) -> bytes:
    result = copy.deepcopy(value)
    body = copy.deepcopy(result)
    body.pop("addendum_body_sha256", None)
    result["addendum_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(result)


def test_descriptor_identity_is_literal_canonical_and_detached() -> None:
    raw = addendum.canonical_compiled_qualification_addendum_descriptor_bytes()
    assert len(raw) == 8_423
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert addendum.compiled_qualification_addendum_descriptor_sha256() == (_DESCRIPTOR_SHA256)
    first = addendum.compiled_qualification_addendum_descriptor()
    first["claims"]["execution_authorized"] = True
    assert (
        addendum.compiled_qualification_addendum_descriptor()["claims"]["execution_authorized"]
        is False
    )
    assert addendum.parse_compiled_qualification_addendum_descriptor(raw) == (
        addendum.compiled_qualification_addendum_descriptor()
    )


def test_import_does_not_execute_configuration_plan_or_materializer() -> None:
    script = """
import sys
from alberta_framework.benchmarks import forager_matched_v3_compiled_qualification_addendum
for name in (
    'alberta_framework.benchmarks.forager_matched_v3_configuration_plan',
    'alberta_framework.benchmarks.forager_matched_v3_external_materialization',
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_descriptor_hard_pins_every_stable_dependency() -> None:
    dependencies = addendum.compiled_qualification_addendum_descriptor()["dependencies"]
    assert dependencies["base_qualification_plan"]["descriptor_sha256"] == (_BASE_DESCRIPTOR_SHA256)
    assert dependencies["base_qualification_plan"]["source_sha256"] == _BASE_SOURCE_SHA256
    assert dependencies["configuration_plan"]["sha256"] == _CONFIGURATION_PLAN_SHA256
    assert dependencies["configuration_plan"]["candidate_record_sha256"] == (
        _CONFIGURATION_RECORD_SHA256
    )
    assert dependencies["compiled_runner"]["descriptor_sha256"] == (
        _COMPILED_RUNNER_DESCRIPTOR_SHA256
    )
    assert dependencies["compiled_runner"]["source_sha256"] == (_COMPILED_RUNNER_SOURCE_SHA256)
    assert dependencies["compiled_reward_bundle"]["descriptor_sha256"] == (
        _COMPILED_BUNDLE_DESCRIPTOR_SHA256
    )
    assert dependencies["compiled_reward_bundle"]["source_sha256"] == (
        _COMPILED_BUNDLE_SOURCE_SHA256
    )
    assert dependencies["compiled_reward_publication"]["descriptor_sha256"] == (
        _COMPILED_PUBLICATION_DESCRIPTOR_SHA256
    )
    assert dependencies["compiled_reward_publication"]["source_sha256"] == (
        _COMPILED_PUBLICATION_SOURCE_SHA256
    )
    assert dependencies["foragax_bridge"]["descriptor_sha256"] == (
        "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
    )
    assert dependencies["ppo_gru_core"]["configuration_sha256"] == (
        "07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70"
    )
    assert dependencies["ppo_gru_core"]["descriptor_sha256"] == (
        "64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca"
    )


def test_build_reuses_only_the_exact_base_candidate_requirements() -> None:
    artifacts = _synthetic_artifacts()
    reused = artifacts.value["reused_base_requirements"]
    base_candidate = next(
        item
        for item in artifacts.base_plan["candidate_requirements"]
        if item["candidate_id"] == "adapted_ppo_gru"
    )
    base_case = next(
        item
        for item in artifacts.base_plan["seed_boundary"]["cases"]
        if item["candidate_id"] == "adapted_ppo_gru"
    )
    base_resource = next(
        item
        for item in artifacts.base_plan["resource_contract"]["requirements"]
        if item["candidate_id"] == "adapted_ppo_gru"
    )
    base_source = next(
        item
        for item in artifacts.base_plan["source_requirements"]
        if item["source_id"] == "local_alberta"
    )
    assert reused["candidate_static_contract"] == {
        key: value for key, value in base_candidate.items() if key != "result_publication_binding"
    }
    assert reused["qualification_seed_case"] == base_case
    assert reused["resource_requirement"] == base_resource
    assert reused["local_source_requirement"] == base_source
    assert reused["shared_runtime_requirement"] == artifacts.base_plan["runtime_requirement"]
    assert "external_foragax_agents" not in json.dumps(reused, sort_keys=True)


def test_base_plan_is_full_digest_bound_not_embedded_amended_or_superseded() -> None:
    artifacts = _synthetic_artifacts()
    binding = artifacts.value["base_plan_binding"]
    assert binding["full_file_sha256"] == artifacts.base_sha256
    assert binding["body_sha256"] == artifacts.base_plan["plan_body_sha256"]
    assert binding["trust_receipt_file_sha256"] == artifacts.receipt_file_sha256
    assert binding["trust_receipt_binding_sha256"] == artifacts.receipt_binding_sha256
    assert binding["strictly_parsed_with_independent_pins"] is True
    assert binding["embedded"] is False
    assert binding["amended"] is False
    assert binding["superseded"] is False


def test_base_publisher_is_explicitly_excluded_and_not_compiled() -> None:
    excluded = _synthetic_artifacts().value["reused_base_requirements"][
        "excluded_base_result_publication"
    ]
    assert excluded["binding"]["descriptor_sha256"] == _BASE_PUBLICATION_DESCRIPTOR_SHA256
    assert excluded["binding"]["descriptor_sha256"] != (_COMPILED_PUBLICATION_DESCRIPTOR_SHA256)
    assert excluded["disposition"] == "excluded_not_reused_not_reinterpreted_as_compiled"
    overlay = _synthetic_artifacts().value["compiled_execution_and_publication_overlay"]
    assert overlay["base_result_publication_reused"] is False
    assert overlay["v1_runner_selected_for_execution"] is False


def test_compiled_chain_is_exactly_runner_bundle_six_file_publication() -> None:
    overlay = _synthetic_artifacts().value["compiled_execution_and_publication_overlay"]
    chain = overlay["ordered_chain"]
    assert [item["descriptor_sha256"] for item in chain] == [
        _COMPILED_RUNNER_DESCRIPTOR_SHA256,
        _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
        _COMPILED_PUBLICATION_DESCRIPTOR_SHA256,
    ]
    assert overlay["publication"]["exact_file_count"] == 6
    assert overlay["publication"]["exact_files"] == {
        "compiled_bundle_manifest": "compiled-bundle-manifest.json",
        "publication_manifest": "publication.json",
        "reward_trace": "reward-trace.npz",
        "runner_result_receipt": "runner-result-receipt.json",
        "runtime_identity": "runtime-identity.json",
        "score_receipt": "score-receipt.json",
    }
    assert overlay["publication"]["canonical_reward_npz_size_bytes"] == 499_980
    assert overlay["publication"]["result_accepted_here"] is False


def test_compiled_geometry_and_rng_accounting_are_exact() -> None:
    overlay = _synthetic_artifacts().value["compiled_execution_and_publication_overlay"]
    assert overlay["accounting"] == {
        "action_draws": 499_712,
        "automatic_resets": 0,
        "bridge_environment_key_uses": 499_713,
        "bridge_resets": 1,
        "compiled_chunk_count": 976,
        "compiled_chunk_steps": 512,
        "environment_interactions": 499_712,
        "optimizer_updates": 15_616,
        "parameter_initialization_draws": 1,
        "permutation_draws": 3_904,
        "segment_steps": 128,
        "segments_per_rollout": 4,
        "total_agent_draws": 503_617,
        "update_epochs": 4,
    }
    assert overlay["rng"] == {
        "agent_action": "continuation,action_key=split(continuation)",
        "agent_initialization": "continuation,init_key=split(agent_root)",
        "agent_permutation": "one continuation split per epoch",
        "automatic_resets": 0,
        "categorical_mode": "low",
        "environment_reset": "continuation,reset_key=split(root)",
        "environment_root": "jax.random.key(environment_seed)",
        "environment_transition": "continuation,step_key=split(continuation)",
        "implementation": "threefry2x32",
        "ppo_environment_chain_consumed": False,
    }


def test_all_claims_false_and_future_qualification_is_explicit() -> None:
    for artifact in (
        _synthetic_artifacts().value,
        addendum.compiled_qualification_addendum_descriptor(),
    ):
        assert artifact["claims"]
        assert all(value is False for value in artifact["claims"].values())
        future = artifact["future_qualification_requirements"]
        assert future["future_executor_required"] is True
        assert future["future_result_validator_required"] is True
        assert future["source_closure_qualification_required"] is True
        assert future["runtime_qualification_required"] is True
        assert future["resource_observation_required"] is True
        assert future["implemented_here"] is False


def test_artifact_roundtrip_requires_independent_full_file_pin() -> None:
    artifacts = _synthetic_artifacts()
    assert _parse(artifacts.raw, expected_file_sha256=artifacts.sha256) == artifacts.value
    assert (
        addendum.compiled_ppo_qualification_addendum_sha256(
            artifacts.value,
            base_plan_raw=artifacts.base_raw,
            expected_base_plan_file_sha256=artifacts.base_sha256,
            expected_base_seed_trust_receipt_file_sha256=artifacts.receipt_file_sha256,
            expected_base_seed_trust_receipt_binding_sha256=artifacts.receipt_binding_sha256,
        )
        == artifacts.sha256
    )
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(artifacts.raw, expected_file_sha256=artifacts.value["addendum_body_sha256"])


@pytest.mark.parametrize(
    "pin",
    ["base_file", "receipt_file", "receipt_binding"],
)
def test_base_and_trust_receipt_pins_are_independently_required(pin: str) -> None:
    artifacts = _synthetic_artifacts()
    arguments = {
        "base_plan_raw": artifacts.base_raw,
        "expected_base_plan_file_sha256": artifacts.base_sha256,
        "expected_base_seed_trust_receipt_file_sha256": artifacts.receipt_file_sha256,
        "expected_base_seed_trust_receipt_binding_sha256": artifacts.receipt_binding_sha256,
    }
    field = {
        "base_file": "expected_base_plan_file_sha256",
        "receipt_file": "expected_base_seed_trust_receipt_file_sha256",
        "receipt_binding": "expected_base_seed_trust_receipt_binding_sha256",
    }[pin]
    arguments[field] = hashlib.sha256(f"wrong:{pin}".encode()).hexdigest()
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        cast(Any, addendum.build_compiled_ppo_qualification_addendum)(**arguments)


@pytest.mark.parametrize("location", ["claim", "future", "overlay"])
def test_bool_int_alias_with_coherent_rehash_fails(location: str) -> None:
    value = copy.deepcopy(_synthetic_artifacts().value)
    if location == "claim":
        value["claims"]["execution_authorized"] = 0
    elif location == "future":
        value["future_qualification_requirements"]["future_executor_required"] = 1
    else:
        value["compiled_execution_and_publication_overlay"]["base_plan_amended"] = 0
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(_coherently_rehash(value))


def test_base_publisher_cannot_be_silently_relabelled_as_compiled() -> None:
    value = copy.deepcopy(_synthetic_artifacts().value)
    excluded = value["reused_base_requirements"]["excluded_base_result_publication"]
    excluded["binding"]["descriptor_sha256"] = _COMPILED_PUBLICATION_DESCRIPTOR_SHA256
    excluded["binding"]["implementation_source_sha256"] = _COMPILED_PUBLICATION_SOURCE_SHA256
    excluded["disposition"] = "compiled"
    excluded["binding_sha256"] = hashlib.sha256(_canonical(excluded["binding"])).hexdigest()
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(_coherently_rehash(value))


def test_unrelated_runner_cannot_replace_compiled_runner_with_coherent_rehash() -> None:
    value = copy.deepcopy(_synthetic_artifacts().value)
    runner = value["compiled_execution_and_publication_overlay"]["ordered_chain"][0]
    runner["descriptor_sha256"] = "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
    runner["source_sha256"] = "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(_coherently_rehash(value))


def test_live_dependency_source_drift_fails_before_addendum_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _synthetic_artifacts()
    original = addendum._bounded_source_sha256

    def drifted(module_file: object, expected_suffix: str) -> str:
        if expected_suffix.endswith("forager_matched_v3_ppo_gru_compiled_runner.py"):
            return hashlib.sha256(b"drifted compiled runner").hexdigest()
        return original(module_file, expected_suffix)

    monkeypatch.setattr(addendum, "_bounded_source_sha256", drifted)
    with pytest.raises(
        addendum.ForagerMatchedV3CompiledQualificationAddendumError,
        match="compiled runner source binding drifted",
    ):
        addendum.build_compiled_ppo_qualification_addendum(
            base_plan_raw=artifacts.base_raw,
            expected_base_plan_file_sha256=artifacts.base_sha256,
            expected_base_seed_trust_receipt_file_sha256=artifacts.receipt_file_sha256,
            expected_base_seed_trust_receipt_binding_sha256=(artifacts.receipt_binding_sha256),
        )


def test_dependency_source_verifier_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "dependency.py"
    os.mkfifo(fifo)
    with pytest.raises(
        addendum.ForagerMatchedV3CompiledQualificationAddendumError,
        match="bounded single-link file",
    ):
        addendum._bounded_source_sha256(str(fifo), "/dependency.py")


def test_compiled_resource_minimums_fail_closed() -> None:
    inputs = base_test._synthetic_inputs()
    base_plan = base_test._build(inputs)
    raw = base_qualification.canonical_matched_v3_qualification_plan_bytes(
        base_plan,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            inputs.expected_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            inputs.expected_trust_root_receipt_binding_sha256
        ),
    )
    with pytest.raises(
        addendum.ForagerMatchedV3CompiledQualificationAddendumError,
        match="max_optimizer_updates",
    ):
        addendum.build_compiled_ppo_qualification_addendum(
            base_plan_raw=raw,
            expected_base_plan_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_base_seed_trust_receipt_file_sha256=(
                inputs.expected_trust_root_receipt_file_sha256
            ),
            expected_base_seed_trust_receipt_binding_sha256=(
                inputs.expected_trust_root_receipt_binding_sha256
            ),
        )


def test_strict_parser_rejects_duplicate_noncanonical_float_depth_nodes_and_bytes() -> None:
    artifacts = _synthetic_artifacts()
    duplicate = b'{"schema_version":"duplicate",' + artifacts.raw[1:]
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(duplicate)
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError):
        _parse(b" " + artifacts.raw)
    float_raw = b'{"x":1.0}\n'
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError, match="float"):
        _parse(float_raw)
    nested: Any = 0
    for _ in range(70):
        nested = [nested]
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError, match="depth"):
        _parse(_canonical({"x": nested}))
    with pytest.raises(addendum.ForagerMatchedV3CompiledQualificationAddendumError, match="node"):
        _parse(_canonical({"x": [0] * 100_001}))
    oversized = b'{"x":"' + b"a" * (2 * 1024 * 1024) + b'"}\n'
    with pytest.raises(
        addendum.ForagerMatchedV3CompiledQualificationAddendumError, match="bounded"
    ):
        _parse(oversized)


def test_canonicalizer_rejects_container_aliases() -> None:
    shared: list[object] = []
    with pytest.raises(
        addendum.ForagerMatchedV3CompiledQualificationAddendumError,
        match="aliased",
    ):
        cast(Any, addendum._canonical_json)({"first": shared, "second": shared})


def test_no_workload_seed_issuer_loader_acceptance_or_default_production_api() -> None:
    for name in (
        "open_compiled_runtime",
        "execute_compiled_qualification",
        "run_compiled_qualification",
        "issue_qualification_seed",
        "load_and_accept_result",
        "accept_compiled_publication",
        "DEFAULT_COMPILED_QUALIFICATION_ADDENDUM",
        "PRODUCTION_COMPILED_QUALIFICATION_ADDENDUM",
        "compiled_runner",
        "compiled_bundle",
        "compiled_publication",
    ):
        assert not hasattr(addendum, name)
    descriptor = addendum.compiled_qualification_addendum_descriptor()
    assert all(value is False for value in descriptor["apis"].values())
    signature = inspect.signature(addendum.build_compiled_ppo_qualification_addendum)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )


def test_build_does_not_open_runtime_run_workload_or_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _synthetic_artifacts()
    private_addendum = cast(Any, addendum)

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("addendum construction called a workload or publication API")

    for module, names in (
        (
            private_addendum._compiled_runner,
            ("open_matched_v3_ppo_gru_compiled_runtime", "run_matched_v3_ppo_gru_compiled"),
        ),
        (private_addendum._compiled_bundle, ("build_ppo_gru_compiled_reward_bundle",)),
        (
            private_addendum._compiled_publication,
            ("publish_compiled_reward_bundle", "load_compiled_reward_publication"),
        ),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    built = addendum.build_compiled_ppo_qualification_addendum(
        base_plan_raw=artifacts.base_raw,
        expected_base_plan_file_sha256=artifacts.base_sha256,
        expected_base_seed_trust_receipt_file_sha256=artifacts.receipt_file_sha256,
        expected_base_seed_trust_receipt_binding_sha256=artifacts.receipt_binding_sha256,
    )
    assert built["claims"]["qualification_executed"] is False
