from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_external_reward_publication as pub

_ROOT = Path(__file__).resolve().parents[1]
_ATOMIC_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "_forager_matched_v3_atomic_publication.py"
)
_PUBLISHER_PATH = Path(pub.__file__).resolve()
_CONSUMER_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_external_outcome_consumer.py"
)
_RUNNER_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_external_execution_runner.py"
)
_ATOMIC_NAME = "_alberta_forager_matched_v3_atomic_publication_isolated_v1"
_PUBLISHER_NAME = (
    "_alberta_forager_matched_v3_external_reward_publication_isolated_v1"
)
_MAXIMUM = 8 * 1024 * 1024
_RELOAD_FORBIDDEN_MODULES = (
    "_alberta_forager_matched_v3_external_outcome_consumer_isolated_v1",
    "_alberta_forager_matched_v3_external_execution_runner_isolated_v1",
    "alberta_framework.benchmarks._forager_matched_v3_external_result_bridge",
    "alberta_framework.benchmarks._forager_matched_v3_scorer",
    "alberta_framework.benchmarks.forager_matched_v3_protocol",
)
_RELOAD_FORBIDDEN_ATTRIBUTES = (
    "forager_matched_v3_external_outcome_consumer",
    "forager_matched_v3_external_execution_runner",
    "_forager_matched_v3_external_result_bridge",
    "_forager_matched_v3_scorer",
    "forager_matched_v3_protocol",
)
_RELOAD_CALLER_PIN_FIELDS = (
    "expected_address",
    "expected_candidate_id",
    "expected_qualification_plan_sha256",
    "expected_qualification_case_manifest_sha256",
    "expected_publisher_source_tree_sha256",
    "expected_workload_source_tree_sha256",
    "expected_staging_manifest_sha256",
    "expected_environment_seed_commitment_sha256",
    "expected_agent_seed_commitment_sha256",
    "expected_consumer_descriptor_sha256",
    "expected_consumer_source_sha256",
    "expected_runner_descriptor_sha256",
    "expected_runner_source_sha256",
    "expected_bridge_descriptor_sha256",
    "expected_bridge_source_sha256",
    "expected_scorer_source_sha256",
    "expected_protocol_source_sha256",
    "expected_metric_descriptor_sha256",
    "expected_execution_contract_descriptor_sha256",
    "expected_staging_descriptor_sha256",
    "expected_seed_transport_descriptor_sha256",
    "expected_production_runner_exact",
    "expected_maximum_publication_total_bytes",
)


@dataclass(frozen=True)
class _PublisherStack:
    atomic: types.ModuleType
    publisher: types.ModuleType
    publisher_source_sha256: str
    consumer_descriptor_sha256: str
    consumer_source_sha256: str


def _load_module(
    path: Path,
    name: str,
    injections: dict[str, object],
) -> tuple[types.ModuleType, str]:
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__dict__.update(injections)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def publisher_stack() -> Iterator[_PublisherStack]:
    consumer_descriptor = "8" * 64
    consumer_source = "9" * 64
    atomic, _atomic_sha256 = _load_module(_ATOMIC_PATH, _ATOMIC_NAME, {})
    publisher_source = hashlib.sha256(_PUBLISHER_PATH.read_bytes()).hexdigest()
    publisher, observed = _load_module(
        _PUBLISHER_PATH,
        _PUBLISHER_NAME,
        {
            "_MATCHED_V3_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256": publisher_source,
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256": consumer_source,
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256": (
                consumer_descriptor
            ),
        },
    )
    assert observed == publisher_source
    yield _PublisherStack(
        atomic=atomic,
        publisher=publisher,
        publisher_source_sha256=publisher_source,
        consumer_descriptor_sha256=consumer_descriptor,
        consumer_source_sha256=consumer_source,
    )
    if sys.modules.get(_PUBLISHER_NAME) is publisher:
        del sys.modules[_PUBLISHER_NAME]
    if sys.modules.get(_ATOMIC_NAME) is atomic:
        del sys.modules[_ATOMIC_NAME]


def _new_parent(tmp_path: Path, name: str = "output") -> Path:
    parent = tmp_path / name
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    return parent


def _content_payloads() -> tuple[tuple[str, bytes], ...]:
    return (
        ("external-execution-receipt.json", b"execution-receipt\n"),
        ("external-conversion-receipt.json", b"conversion-receipt\n"),
        ("upstream-reward.npz", b"opaque-upstream-reward"),
        ("upstream-results.db", b"opaque-results-database"),
        ("upstream-video-slot.bin", b""),
        ("reward-trace.npz", b"opaque-canonical-reward"),
        ("stdout.bin", b"stdout"),
        ("stderr.bin", b"stderr"),
    )


def _facts(stack: _PublisherStack) -> object:
    publisher = stack.publisher
    content = dict(_content_payloads())
    return publisher._ExternalPublicationFacts(
        candidate_id="external_dqn_plain",
        external_candidate_ordinal=0,
        family="continuing",
        qualification_plan_sha256="1" * 64,
        qualification_case_manifest_sha256="2" * 64,
        publisher_source_tree_sha256="3" * 64,
        workload_source_tree_sha256="4" * 64,
        staging_manifest_sha256="5" * 64,
        environment_seed_commitment_sha256="6" * 64,
        agent_seed_commitment_sha256="7" * 64,
        runner_descriptor_sha256="a" * 64,
        runner_source_sha256="b" * 64,
        consumer_descriptor_sha256=stack.consumer_descriptor_sha256,
        consumer_source_sha256=stack.consumer_source_sha256,
        bridge_descriptor_sha256="c" * 64,
        bridge_source_sha256="d" * 64,
        scorer_source_sha256="e" * 64,
        protocol_source_sha256="f" * 64,
        metric_descriptor_sha256="1" * 64,
        execution_contract_descriptor_sha256="2" * 64,
        staging_descriptor_sha256="3" * 64,
        seed_transport_descriptor_sha256="4" * 64,
        execution_receipt_sha256=hashlib.sha256(
            content["external-execution-receipt.json"]
        ).hexdigest(),
        conversion_receipt_sha256=hashlib.sha256(
            content["external-conversion-receipt.json"]
        ).hexdigest(),
        production_runner_exact=False,
        video_slot_mode="absent_for_continuing_zero_length_slot",
        maximum_publication_total_bytes=_MAXIMUM,
    )


def _role_payloads(stack: _PublisherStack, facts: object) -> tuple[tuple[str, bytes], ...]:
    content = _content_payloads()
    outcome = stack.publisher._build_external_outcome_manifest(
        facts=facts,
        content_payloads=content,
    )
    return (("external-outcome-manifest.json", outcome), *content)


def _publish(
    stack: _PublisherStack,
    parent: Path,
) -> tuple[Any, object, tuple[tuple[str, bytes], ...]]:
    facts = _facts(stack)
    role_payloads = _role_payloads(stack, facts)
    metadata = stack.publisher._publish_consumed_external_outcome_payload(
        publication_parent=parent,
        role_payloads=role_payloads,
        facts=facts,
    )
    return metadata, facts, role_payloads


def _reload_kwargs(metadata: Any) -> dict[str, object]:
    return {
        "publication_parent": metadata.publication_root.parent,
        "expected_address": metadata.address,
        "expected_file_records": metadata.files,
        "expected_candidate_id": metadata.candidate_id,
        "expected_qualification_plan_sha256": metadata.qualification_plan_sha256,
        "expected_qualification_case_manifest_sha256": (
            metadata.qualification_case_manifest_sha256
        ),
        "expected_publisher_source_tree_sha256": (
            metadata.publisher_source_tree_sha256
        ),
        "expected_workload_source_tree_sha256": metadata.workload_source_tree_sha256,
        "expected_staging_manifest_sha256": metadata.staging_manifest_sha256,
        "expected_environment_seed_commitment_sha256": (
            metadata.environment_seed_commitment_sha256
        ),
        "expected_agent_seed_commitment_sha256": (
            metadata.agent_seed_commitment_sha256
        ),
        "expected_consumer_descriptor_sha256": metadata.consumer_descriptor_sha256,
        "expected_consumer_source_sha256": metadata.consumer_source_sha256,
        "expected_runner_descriptor_sha256": metadata.runner_descriptor_sha256,
        "expected_runner_source_sha256": metadata.runner_source_sha256,
        "expected_bridge_descriptor_sha256": metadata.bridge_descriptor_sha256,
        "expected_bridge_source_sha256": metadata.bridge_source_sha256,
        "expected_scorer_source_sha256": metadata.scorer_source_sha256,
        "expected_protocol_source_sha256": metadata.protocol_source_sha256,
        "expected_metric_descriptor_sha256": metadata.metric_descriptor_sha256,
        "expected_execution_contract_descriptor_sha256": (
            metadata.execution_contract_descriptor_sha256
        ),
        "expected_staging_descriptor_sha256": metadata.staging_descriptor_sha256,
        "expected_seed_transport_descriptor_sha256": (
            metadata.seed_transport_descriptor_sha256
        ),
        "expected_production_runner_exact": metadata.production_runner_exact,
        "expected_maximum_publication_total_bytes": (
            metadata.maximum_publication_total_bytes
        ),
    }


@contextmanager
def _without_score_modules() -> Iterator[None]:
    saved_modules = {
        name: sys.modules.pop(name)
        for name in _RELOAD_FORBIDDEN_MODULES
        if name in sys.modules
    }
    package = sys.modules.get("alberta_framework.benchmarks")
    saved_attributes: dict[str, object] = {}
    if type(package) is types.ModuleType:
        for name in _RELOAD_FORBIDDEN_ATTRIBUTES:
            if hasattr(package, name):
                saved_attributes[name] = getattr(package, name)
                delattr(package, name)
    try:
        yield
    finally:
        if type(package) is types.ModuleType:
            for name, value in saved_attributes.items():
                setattr(package, name, value)
        sys.modules.update(saved_modules)


def _tampered_pin(field: str, current: object) -> object:
    if field == "expected_candidate_id":
        return "external_dqn_crelu"
    if field == "expected_production_runner_exact":
        return not current
    if field == "expected_maximum_publication_total_bytes":
        assert type(current) is int
        return current + 1
    changed = hashlib.sha256(f"tampered:{field}".encode("ascii")).hexdigest()
    assert changed != current
    return changed


def _assert_metadata_only(value: object) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        assert type(item) is not bytes
        dataclass_fields = getattr(item, "__dataclass_fields__", None)
        if type(dataclass_fields) is dict:
            for field in cast(dict[str, Any], dataclass_fields).values():
                assert field.name not in {
                    "content",
                    "conversion",
                    "payload",
                    "raw_trace",
                    "reward",
                    "rewards",
                    "score",
                    "trace",
                    "trace_bytes",
                }
                pending.append(getattr(item, field.name))
        elif type(item) is tuple:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())


def test_descriptor_freezes_metadata_only_ten_file_contract() -> None:
    descriptor = pub.external_reward_publication_descriptor()
    assert descriptor["schema_version"] == pub.EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor["classification"] == (
        "payload_score_fields_not_decoded_atomic_external_publication_"
        "metadata_only_non_authorizing"
    )
    assert descriptor["candidate_count"] == 12
    assert descriptor["candidate_order"] == list(pub.EXTERNAL_PUBLICATION_CANDIDATE_IDS)
    assert descriptor["publication"]["exact_filenames"] == list(pub.EXTERNAL_PUBLICATION_FILENAMES)
    assert descriptor["publication"]["exact_file_count"] == 10
    assert descriptor["publication"]["address"] == "full_sha256_of_publication_json"
    assert descriptor["publication"]["maximum_total_bytes"] == 1024 * 1024 * 1024
    assert descriptor["publication"]["atomic_helper_call_count"] == 1
    assert descriptor["publication"]["collision_retry"] is False
    assert descriptor["publication"]["uncertain_state_retry"] is False
    assert descriptor["publisher_policy"]["safe_parent_preflight_before_runner_claim"]
    assert descriptor["publisher_policy"]["atomic_commit_reopens_and_reverifies_parent"]
    assert descriptor["publisher_policy"]["parent_preflight_eliminates_toctou"] is False
    assert descriptor["publication"]["terminal_metadata"] == {
        "interaction_horizon_explicit": True,
        "publication_committed_explicit": True,
        "atomic_helper_intent_sha256": True,
        "atomic_publication_receipt_sha256": True,
        "atomic_publication_receipt_semantics": (
            "canonical_digest_of_content_verified_helper_result_metadata"
        ),
        "reload_observation_sha256": True,
        "score_or_reward_value_fields": False,
        "raw_score_reward_or_payload_fields_recursively_forbidden": True,
    }
    assert descriptor["public_publish_interface"] == {
        "accepts_live_outcome_capability": True,
        "accepts_public_completion": False,
        "accepts_bridge_conversion": False,
        "accepts_payload_bytes": False,
        "accepts_callback_or_sink": False,
        "returns_immutable_metadata_only": True,
    }
    assert all(value is False for value in descriptor["claims"].values())


def test_exact_semantic_role_and_filename_order() -> None:
    assert pub.EXTERNAL_PUBLICATION_CANDIDATE_IDS == (
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
    assert pub.EXTERNAL_PUBLICATION_ROLE_PATHS == (
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
    assert pub.EXTERNAL_PUBLICATION_FILENAMES == tuple(
        path for _role, path in pub.EXTERNAL_PUBLICATION_ROLE_PATHS
    )


def test_publisher_source_has_no_static_project_import() -> None:
    tree = ast.parse(_PUBLISHER_PATH.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert all(not name.startswith("alberta_framework") for name in imports)


@pytest.mark.parametrize(
    ("source_path", "constant_name", "expected"),
    (
        (
            _PUBLISHER_PATH,
            "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
            "59d470d6c31e1d3dce8eded401e6331994ca007b94524d8e00714c1f2c66f30b",
        ),
        (
            _CONSUMER_PATH,
            "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256",
            "7c7d007f29b55d6e4a72467d72c4b793568847930d7eb0c17cc276b027e74ceb",
        ),
        (
            _RUNNER_PATH,
            "EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256",
            "0f0c12a93f458ded1188185fed8c0c97e5763f5efa5151f84b70f28b2c945636",
        ),
    ),
)
def test_external_chain_descriptor_hash_assignments_are_string_literals(
    source_path: Path,
    constant_name: str,
    expected: str,
) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == constant_name
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Constant)
    assert type(value.value) is str
    assert value.value == expected


def test_metadata_and_file_records_reject_bytes_aliases_and_wrong_order(tmp_path: Path) -> None:
    record = pub.MatchedV3ExternalPublicationFile(
        role="publication_manifest",
        name="publication.json",
        size_bytes=1,
        sha256=hashlib.sha256(b"x").hexdigest(),
    )
    assert not hasattr(record, "content")
    with pytest.raises(pub.ForagerMatchedV3ExternalRewardPublicationError):
        pub.MatchedV3ExternalPublicationFile(
            role="stdout",
            name="stderr.bin",
            size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
        )
    with pytest.raises(pub.ForagerMatchedV3ExternalRewardPublicationError):
        pub.load_matched_v3_external_reward_publication(
            publication_parent=tmp_path,
            expected_address=record.sha256,
            expected_file_records=(record,),
            expected_candidate_id="external_dqn_plain",
            expected_qualification_plan_sha256="1" * 64,
            expected_qualification_case_manifest_sha256="2" * 64,
            expected_publisher_source_tree_sha256="3" * 64,
            expected_workload_source_tree_sha256="4" * 64,
            expected_staging_manifest_sha256="5" * 64,
            expected_environment_seed_commitment_sha256="6" * 64,
            expected_agent_seed_commitment_sha256="7" * 64,
            expected_consumer_descriptor_sha256="8" * 64,
            expected_consumer_source_sha256="9" * 64,
            expected_runner_descriptor_sha256="a" * 64,
            expected_runner_source_sha256="b" * 64,
            expected_bridge_descriptor_sha256="c" * 64,
            expected_bridge_source_sha256="d" * 64,
            expected_scorer_source_sha256="e" * 64,
            expected_protocol_source_sha256="f" * 64,
            expected_metric_descriptor_sha256="1" * 64,
            expected_execution_contract_descriptor_sha256="2" * 64,
            expected_staging_descriptor_sha256="3" * 64,
            expected_seed_transport_descriptor_sha256="4" * 64,
            expected_production_runner_exact=False,
            expected_maximum_publication_total_bytes=1024,
        )


def test_video_slot_semantics_are_family_derived() -> None:
    assert pub._video_slot_mode(candidate_id="external_dqn_plain", raw=b"") == (
        "absent_for_continuing_zero_length_slot"
    )
    with pytest.raises(pub.ForagerMatchedV3ExternalRewardPublicationError):
        pub._video_slot_mode(candidate_id="external_dqn_plain", raw=b"video")
    assert pub._video_slot_mode(
        candidate_id="isolated_ppo_generic", raw=b"video"
    ) == "opaque_ppo_video"
    with pytest.raises(pub.ForagerMatchedV3ExternalRewardPublicationError):
        pub._video_slot_mode(candidate_id="isolated_ppo_generic", raw=b"")


def test_outcome_manifest_validates_family_video_bytes_before_publication(
    publisher_stack: _PublisherStack,
) -> None:
    publisher = publisher_stack.publisher
    continuing = _facts(publisher_stack)
    continuing_payloads = list(_content_payloads())
    continuing_payloads[4] = ("upstream-video-slot.bin", b"unexpected-video")
    with pytest.raises(publisher.ForagerMatchedV3ExternalRewardPublicationError):
        publisher._build_external_outcome_manifest(
            facts=continuing,
            content_payloads=tuple(continuing_payloads),
        )

    ppo = replace(
        cast(Any, continuing),
        candidate_id="isolated_ppo_generic",
        external_candidate_ordinal=7,
        family="ppo",
        video_slot_mode="opaque_ppo_video",
    )
    with pytest.raises(publisher.ForagerMatchedV3ExternalRewardPublicationError):
        publisher._build_external_outcome_manifest(
            facts=ppo,
            content_payloads=_content_payloads(),
        )
    ppo_payloads = list(_content_payloads())
    ppo_payloads[4] = ("upstream-video-slot.bin", b"opaque-video")
    manifest = publisher._build_external_outcome_manifest(
        facts=ppo,
        content_payloads=tuple(ppo_payloads),
    )
    assert json.loads(manifest)["candidate"]["video_slot_mode"] == "opaque_ppo_video"


def test_descriptor_and_metadata_parsers_require_exact_canonical_bytes() -> None:
    raw = pub.canonical_external_reward_publication_descriptor_bytes()
    assert pub.parse_external_reward_publication_descriptor(raw) == (
        pub.external_reward_publication_descriptor()
    )
    changed = json.dumps(json.loads(raw), indent=2, sort_keys=True).encode("ascii") + b"\n"
    with pytest.raises(pub.ForagerMatchedV3ExternalRewardPublicationError):
        pub.parse_external_reward_publication_descriptor(changed)


def test_isolated_descriptor_self_replay_and_source_drift_fail_closed(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = publisher_stack.publisher
    descriptor = publisher.canonical_external_reward_publication_descriptor_bytes()
    assert hashlib.sha256(descriptor).hexdigest() == (
        publisher.EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    )
    assert publisher.parse_external_reward_publication_descriptor(descriptor) == (
        publisher.external_reward_publication_descriptor()
    )
    assert publisher._require_boundary() == publisher_stack.publisher_source_sha256

    drifted = tmp_path / _PUBLISHER_PATH.name
    drifted.write_bytes(_PUBLISHER_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(publisher, "__file__", str(drifted))
    with pytest.raises(publisher.ForagerMatchedV3ExternalRewardPublicationError):
        publisher._require_boundary()


def test_isolated_descriptor_runtime_drift_fails_closed(
    publisher_stack: _PublisherStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = publisher_stack.publisher
    monkeypatch.setattr(publisher, "_DESCRIPTOR_BYTES", publisher._DESCRIPTOR_BYTES + b" ")
    with pytest.raises(publisher.ForagerMatchedV3ExternalRewardPublicationError):
        publisher._require_boundary()


def test_valid_publication_metadata_round_trips_without_content(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
) -> None:
    metadata, _facts_value, _payloads = _publish(
        publisher_stack, _new_parent(tmp_path)
    )
    publisher = publisher_stack.publisher
    assert type(metadata) is publisher.MatchedV3ExternalPublicationMetadata
    assert metadata.operation == "published"
    assert metadata.interaction_horizon == 499_712
    assert metadata.publication_committed is True
    assert metadata.address == hashlib.sha256(
        (metadata.publication_root / "publication.json").read_bytes()
    ).hexdigest()
    root_metadata = metadata.publication_root.stat(follow_symlinks=False)
    assert stat.S_ISDIR(root_metadata.st_mode)
    assert stat.S_IMODE(root_metadata.st_mode) == 0o700
    assert root_metadata.st_nlink == 2
    assert tuple(
        sorted(
            (path.name for path in metadata.publication_root.iterdir()),
            key=os.fsencode,
        )
    ) == tuple(
        sorted(publisher.EXTERNAL_PUBLICATION_FILENAMES, key=os.fsencode)
    )
    for name in publisher.EXTERNAL_PUBLICATION_FILENAMES:
        file_metadata = (metadata.publication_root / name).stat(
            follow_symlinks=False
        )
        assert stat.S_ISREG(file_metadata.st_mode)
        assert stat.S_IMODE(file_metadata.st_mode) == 0o600
        assert file_metadata.st_nlink == 1
    assert metadata.files == tuple(
        publisher.MatchedV3ExternalPublicationFile(
            role=role,
            name=name,
            size_bytes=(metadata.publication_root / name).stat().st_size,
            sha256=hashlib.sha256(
                (metadata.publication_root / name).read_bytes()
            ).hexdigest(),
        )
        for role, name in publisher.EXTERNAL_PUBLICATION_ROLE_PATHS
    )
    _assert_metadata_only(metadata)
    raw = publisher.canonical_external_publication_metadata_bytes(metadata)
    reconstructed = publisher.parse_external_publication_metadata(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert reconstructed == metadata
    _assert_metadata_only(reconstructed)


@pytest.mark.parametrize("target_name", pub.EXTERNAL_PUBLICATION_FILENAMES[1:])
def test_reload_rejects_each_nonself_file_tamper_with_updated_caller_record(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
    target_name: str,
) -> None:
    metadata, _facts_value, _payloads = _publish(
        publisher_stack, _new_parent(tmp_path)
    )
    publisher = publisher_stack.publisher
    target = metadata.publication_root / target_name
    changed = target.read_bytes() + b"\x00"
    target.write_bytes(changed)
    records = tuple(
        publisher.MatchedV3ExternalPublicationFile(
            role=record.role,
            name=record.name,
            size_bytes=len(changed) if record.name == target_name else record.size_bytes,
            sha256=(
                hashlib.sha256(changed).hexdigest()
                if record.name == target_name
                else record.sha256
            ),
        )
        for record in metadata.files
    )
    kwargs = _reload_kwargs(metadata)
    kwargs["expected_file_records"] = records
    with _without_score_modules(), pytest.raises(
        publisher.ForagerMatchedV3ExternalRewardPublicationError
    ):
        publisher.load_matched_v3_external_reward_publication(**kwargs)


@pytest.mark.parametrize("pin_field", _RELOAD_CALLER_PIN_FIELDS)
def test_reload_rejects_each_tampered_caller_carried_component_pin(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
    pin_field: str,
) -> None:
    metadata, _facts_value, _payloads = _publish(
        publisher_stack, _new_parent(tmp_path)
    )
    publisher = publisher_stack.publisher
    kwargs = _reload_kwargs(metadata)
    kwargs[pin_field] = _tampered_pin(pin_field, kwargs[pin_field])
    with _without_score_modules(), pytest.raises(
        publisher.ForagerMatchedV3ExternalRewardPublicationError
    ):
        publisher.load_matched_v3_external_reward_publication(**kwargs)


def test_collision_is_one_atomic_call_and_never_retried(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _new_parent(tmp_path)
    _publish(publisher_stack, parent)
    publisher = publisher_stack.publisher
    original = publisher._ATOMIC_PUBLISH_AT_LOAD
    calls = 0

    def counting_publish(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(publisher, "_ATOMIC_PUBLISH_AT_LOAD", counting_publish)
    with pytest.raises(
        publisher_stack.atomic.ForagerMatchedV3AtomicPublicationCollisionError
    ):
        _publish(publisher_stack, parent)
    assert calls == 1


def test_postcommit_validation_failure_is_uncertain_after_one_call_without_retry(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _new_parent(tmp_path)
    publisher = publisher_stack.publisher
    original = publisher._ATOMIC_PUBLISH_AT_LOAD
    calls = 0

    def invalid_result_after_commit(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        committed = original(*args, **kwargs)
        assert type(committed) is publisher_stack.atomic.ContentVerifiedFlatPublication
        return object()

    monkeypatch.setattr(
        publisher, "_ATOMIC_PUBLISH_AT_LOAD", invalid_result_after_commit
    )
    with pytest.raises(
        publisher_stack.atomic.ForagerMatchedV3AtomicPublicationUncertainError
    ) as raised:
        _publish(publisher_stack, parent)
    assert calls == 1
    assert raised.value.committed is True
    assert raised.value.destination.is_dir()


def test_malformed_nested_metadata_record_is_wrapped_as_publication_error(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
) -> None:
    metadata, _facts_value, _payloads = _publish(
        publisher_stack, _new_parent(tmp_path)
    )
    publisher = publisher_stack.publisher
    raw = publisher.canonical_external_publication_metadata_bytes(metadata)
    malformed = json.loads(raw)
    del malformed["files"][0]["role"]
    malformed_raw = json.dumps(
        malformed,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    with pytest.raises(publisher.ForagerMatchedV3ExternalRewardPublicationError):
        publisher.parse_external_publication_metadata(
            malformed_raw,
            expected_file_sha256=hashlib.sha256(malformed_raw).hexdigest(),
        )


def test_fresh_isolated_reload_succeeds_and_rejects_module_or_package_presence(
    publisher_stack: _PublisherStack,
    tmp_path: Path,
) -> None:
    metadata, _facts_value, _payloads = _publish(
        publisher_stack, _new_parent(tmp_path)
    )
    kwargs = _reload_kwargs(metadata)
    handoff = {
        "address": metadata.address,
        "records": [
            {
                "role": record.role,
                "name": record.name,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in metadata.files
        ],
        "pins": {
            name: value
            for name, value in kwargs.items()
            if name
            not in {"publication_parent", "expected_address", "expected_file_records"}
        },
    }
    script = r'''\
import hashlib
import json
import sys
import types
from pathlib import Path

def load(path, name, injections):
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ''
    module.__dict__.update(injections)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

atomic_path = Path(sys.argv[1]).resolve()
publisher_path = Path(sys.argv[2]).resolve()
parent = Path(sys.argv[3]).resolve()
handoff = json.loads(sys.argv[4])
atomic_name = '_alberta_forager_matched_v3_atomic_publication_isolated_v1'
publisher_name = '_alberta_forager_matched_v3_external_reward_publication_isolated_v1'
load(atomic_path, atomic_name, {})
publisher_source = hashlib.sha256(publisher_path.read_bytes()).hexdigest()
publisher = load(
    publisher_path,
    publisher_name,
    {
        '_MATCHED_V3_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256': publisher_source,
        '_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256': (
            handoff['pins']['expected_consumer_source_sha256']
        ),
        '_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256': (
            handoff['pins']['expected_consumer_descriptor_sha256']
        ),
    },
)
records = tuple(
    publisher.MatchedV3ExternalPublicationFile(**item)
    for item in handoff['records']
)
call = dict(
    publication_parent=parent,
    expected_address=handoff['address'],
    expected_file_records=records,
    **handoff['pins'],
)
result = publisher.load_matched_v3_external_reward_publication(**call)
assert result.operation == 'reloaded'
assert result.address == handoff['address']

protocol_name = 'alberta_framework.benchmarks.forager_matched_v3_protocol'
sys.modules[protocol_name] = types.ModuleType(protocol_name)
try:
    publisher.load_matched_v3_external_reward_publication(**call)
except publisher.ForagerMatchedV3ExternalRewardPublicationError:
    pass
else:
    raise AssertionError('reload accepted a loaded score-bearing module')
del sys.modules[protocol_name]

package_name = 'alberta_framework.benchmarks'
package = types.ModuleType(package_name)
package.forager_matched_v3_protocol = object()
sys.modules[package_name] = package
try:
    publisher.load_matched_v3_external_reward_publication(**call)
except publisher.ForagerMatchedV3ExternalRewardPublicationError:
    pass
else:
    raise AssertionError('reload accepted a lingering score-bearing package alias')
print(json.dumps({'address': result.address, 'operation': result.operation}))
'''
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(_ATOMIC_PATH),
            str(_PUBLISHER_PATH),
            str(metadata.publication_root.parent),
            json.dumps(handoff, separators=(",", ":"), sort_keys=True),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "address": metadata.address,
        "operation": "reloaded",
    }


def test_public_surface_has_no_byte_or_conversion_ingestion_api() -> None:
    assert pub.__all__ == [
        "EXTERNAL_PUBLICATION_CANDIDATE_IDS",
        "EXTERNAL_PUBLICATION_FILENAMES",
        "EXTERNAL_PUBLICATION_ROLE_PATHS",
        "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
        "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
        "EXTERNAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION",
        "EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION",
        "EXTERNAL_REWARD_PUBLICATION_STATUS",
        "ForagerMatchedV3ExternalRewardPublicationError",
        "MATCHED_V3_INTERACTION_HORIZON",
        "MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES",
        "MatchedV3ExternalPublicationFile",
        "MatchedV3ExternalPublicationMetadata",
        "canonical_external_publication_metadata_bytes",
        "canonical_external_reward_publication_descriptor_bytes",
        "external_reward_publication_descriptor",
        "load_matched_v3_external_reward_publication",
        "parse_external_publication_metadata",
        "parse_external_reward_publication_descriptor",
        "publish_matched_v3_external_outcome_capability",
    ]
    exported = set(pub.__all__)
    assert "_publish_consumed_external_outcome_payload" not in exported
    assert not any("conversion" in name and name.startswith("publish") for name in exported)
    assert not any("bytes" in name and name.startswith("publish") for name in exported)
    forbidden_parameters = {"raw", "payload", "payloads", "content", "bytes"}
    for name in (
        "publish_matched_v3_external_outcome_capability",
        "load_matched_v3_external_reward_publication",
    ):
        assert forbidden_parameters.isdisjoint(
            inspect.signature(getattr(pub, name)).parameters
        )
    assert "raw" in inspect.signature(
        pub.parse_external_publication_metadata
    ).parameters
    assert "raw" in inspect.signature(
        pub.parse_external_reward_publication_descriptor
    ).parameters
