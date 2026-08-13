"""Adversarial contracts for durable, non-authorizing v3 adapter publications.

Every result in this module is synthetic.  Most mutation tests patch the private
adapter-bundle receipt reader; dedicated structural-receipt and fresh-process tests use
the real frozen receipt parsers.  No test executes either candidate runner or reads a
repository evidence artifact.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_publication as publication,
)
from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import (
    forager_matched_v3_full_rainbow_runner as full_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_runner as ppo_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[1]
_PUBLICATION_OS: Any = getattr(publication, "os")
_EXPECTED_NAMES = frozenset(
    {
        publication.PUBLICATION_FILENAME,
        publication.ADAPTER_BUNDLE_MANIFEST_FILENAME,
        publication.RUNNER_RESULT_RECEIPT_FILENAME,
        publication.REWARD_TRACE_FILENAME,
        publication.SCORE_RECEIPT_FILENAME,
    }
)
assert _EXPECTED_NAMES == {
    "publication.json",
    "adapter-bundle-manifest.json",
    "runner-result-receipt.json",
    "reward-trace.npz",
    "score-receipt.json",
}
_LIVE_CAPABILITY_SENTINEL = b"synthetic-live-capability-must-never-be-persisted-5de8"

SyntheticBundleFactory = Callable[
    [str, int, str], bundle.MatchedV3AdapterRewardBundle
]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _structural_ppo_result_receipt(raw_trace: bytes) -> bytes:
    geometry = ppo_runner.MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY
    identity = bridge.MatchedV3ForagaxRuntimeIdentity(
        jax_version=bridge.JAX_REQUIRED_VERSION,
        jaxlib_version=bridge.JAXLIB_REQUIRED_VERSION,
        default_prng_impl="threefry2x32",
        threefry_partitionable=True,
        jax_enable_x64=False,
        backend="cpu",
        foragax_version=bridge.FORAGAX_REQUIRED_VERSION,
        foragax_install_tree_sha256=bridge.FORAGAX_INSTALL_TREE_SHA256,
        foragax_package_root="/synthetic/nonexecuted/foragax",
        runtime_qualified=False,
    )
    runtime_identity = ppo_runner._bridge_runtime_identity_dict(identity)
    body: dict[str, Any] = {
        "schema_version": ppo_runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
        "candidate_id": "adapted_ppo_gru",
        "classification": "production_runtime_unqualified_non_authorizing",
        "runner": ppo_runner._frozen_runner_binding(),
        "dependencies": ppo_runner._frozen_dependency_binding(),
        "seeds": {
            "environment_seed": 0,
            "agent_seed": 0,
            "provenance": "caller_supplied_unverified",
            "upstream_receipt_bound": False,
            "protected_seed_status": "unverified",
        },
        "geometry": geometry.to_dict(),
        "accounting": {
            "environment_interactions": geometry.horizon,
            "rollout_count": geometry.rollout_count,
            "optimizer_update_count": geometry.optimizer_update_count,
            "parameter_initialization_draw_count": 1,
            "action_draw_count": geometry.action_draw_count,
            "permutation_draw_count": geometry.permutation_draw_count,
            "total_agent_draw_count": geometry.total_agent_draw_count,
            "ppo_environment_draw_count": 0,
            "bridge_reset_count": 1,
            "bridge_step_count": geometry.horizon,
            "bridge_environment_key_use_count": 1 + geometry.horizon,
        },
        "raw_reward_trace": {
            "encoding": "signed_int8_twos_complement",
            "length": len(raw_trace),
            "sha256": _sha256(raw_trace),
            "score_reduction": "exact_int64_sum",
            "score_scaling": "none",
        },
        "raw_cumulative_score": 0,
        "trace_chain_sha256": _sha256(b"synthetic-structural-trace-chain"),
        "production_horizon_complete": True,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": _sha256(_canonical_json(runtime_identity)),
        "claims": ppo_runner._non_authorizing_claims(),
        "limitations": list(ppo_runner._receipt_limitations()),
    }
    payload = dict(body)
    payload["receipt_sha256"] = _sha256(_canonical_json(body))
    raw = _canonical_json(payload)
    ppo_runner.parse_ppo_gru_result_receipt(raw)
    return raw


def _structural_full_rainbow_result_receipt(raw_trace: bytes) -> bytes:
    schedule = full_runner.production_full_rainbow_schedule()
    base_accounting = full_runner.full_rainbow_schedule_accounting(schedule)
    update_transitions = [
        transition
        for transition in range(1, schedule.horizon + 1)
        if transition - schedule.update_horizon > schedule.minimum_replay_history
        and transition % schedule.update_period == 0
    ]
    target_sync_transitions = [
        transition
        for transition in range(1, schedule.horizon + 1)
        if transition - schedule.update_horizon > schedule.minimum_replay_history
        and transition % schedule.target_update_period == 0
    ]
    accounting: dict[str, Any] = {
        **base_accounting,
        "update_transitions": update_transitions,
        "target_sync_transitions": target_sync_transitions,
    }
    updates = cast(int, accounting["optimizer_updates"])
    dependencies = full_runner.FullRainbowRunnerDependencies(
        dependency_identity="production_bridge_and_compiled_full_rainbow_v1",
        environment_runtime=SimpleNamespace(initialize=lambda *args, **kwargs: None),
        step_environment=cast(Any, lambda *args, **kwargs: None),
        initialize_core=cast(Any, lambda *args, **kwargs: None),
        action_q_values=cast(Any, lambda *args, **kwargs: None),
        update_core=cast(Any, lambda *args, **kwargs: None),
        sync_target=cast(Any, lambda *args, **kwargs: None),
        runtime_identity={
            "backend": "synthetic_structural_nonexecuted",
            "runtime_qualified": False,
            "foragax_runtime_parity_executed": False,
        },
        compiled_action_kernel=True,
        compiled_update_kernel=True,
    )
    raw = full_runner._receipt(
        environment_seed=0,
        agent_seed=0,
        schedule=schedule,
        dependencies=dependencies,
        raw_trace=raw_trace,
        cumulative_score=0,
        accounting=accounting,
        rng_accounting={
            "agent_continuation_split_calls": 1 + schedule.horizon + 2 * updates,
            "agent_parameter_initialization_subkeys": 2,
            "agent_action_split_calls": schedule.horizon,
            "agent_action_subkeys_produced_per_call": 3,
            "agent_replay_sampling_split_calls": updates,
            "agent_update_split_calls": updates,
            "agent_update_subkeys_produced_per_call": 2,
            "core_environment_key_consumptions": 0,
            "bridge_environment_key_consumptions": schedule.horizon + 1,
        },
        production_runtime=True,
    )
    full_runner.parse_full_rainbow_result_receipt(raw)
    return raw


def _rewrite_publication_body(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = dict(payload)
    body.pop("publication_body_sha256", None)
    body_sha256 = _sha256(_canonical_json(body))
    rewritten = dict(body)
    rewritten["publication_body_sha256"] = body_sha256
    raw = _canonical_json(rewritten)
    return raw, _sha256(raw)


def _rewrite_bundle_manifest_body(payload: dict[str, Any]) -> bytes:
    body = dict(payload)
    body.pop("manifest_body_sha256", None)
    body_sha256 = _sha256(_canonical_json(body))
    rewritten = dict(body)
    rewritten["manifest_body_sha256"] = body_sha256
    return _canonical_json(rewritten)


def _nested_mappings(value: object) -> Iterator[dict[str, Any]]:
    if type(value) is dict:
        mapping = cast(dict[str, Any], value)
        yield mapping
        for item in mapping.values():
            yield from _nested_mappings(item)
    elif type(value) is list:
        for item in value:
            yield from _nested_mappings(item)


def _file_record(payload: dict[str, Any], filename: str) -> dict[str, Any]:
    matches = [
        value
        for value in _nested_mappings(payload)
        if value.get("path") == filename
        and "sha256" in value
        and "size_bytes" in value
    ]
    assert len(matches) == 1
    return matches[0]


def _replace_file_record(
    payload: dict[str, Any], filename: str, raw: bytes
) -> None:
    record = _file_record(payload, filename)
    record["sha256"] = _sha256(raw)
    record["size_bytes"] = len(raw)


def _write_rehashed_publication(root: Path, payload: dict[str, Any]) -> str:
    raw, full_sha256 = _rewrite_publication_body(payload)
    (root / "publication.json").write_bytes(raw)
    return full_sha256


def _read_publication(root: Path) -> dict[str, Any]:
    value = json.loads((root / "publication.json").read_bytes())
    assert type(value) is dict
    return cast(dict[str, Any], value)


@pytest.fixture
def synthetic_bundle_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> SyntheticBundleFactory:
    facts_by_receipt: dict[bytes, tuple[str, int, str, int]] = {}

    def receipt_facts(
        candidate_id: str, runner_receipt_bytes: bytes
    ) -> tuple[dict[str, Any], int, str, int]:
        try:
            expected_candidate, score, trace_sha256, trace_length = facts_by_receipt[
                runner_receipt_bytes
            ]
        except KeyError as exc:
            raise bundle.ForagerMatchedV3AdapterRewardBundleError(
                "synthetic runner receipt is not registered"
            ) from exc
        if candidate_id != expected_candidate:
            raise bundle.ForagerMatchedV3AdapterRewardBundleError(
                "synthetic candidate differs"
            )
        return (
            {"private_live_capability": _LIVE_CAPABILITY_SENTINEL.decode("ascii")},
            score,
            trace_sha256,
            trace_length,
        )

    monkeypatch.setattr(bundle, "_runner_receipt_facts", receipt_facts)

    def build(
        candidate_id: str = "adapted_full_rainbow",
        encoded_reward: int = 0,
        label: str = "a",
    ) -> bundle.MatchedV3AdapterRewardBundle:
        if encoded_reward not in {0, 1, 30, 255}:
            raise AssertionError("synthetic reward must belong to the frozen support")
        raw_trace = bytes((encoded_reward,)) + bytes(protocol.MATCHED_V3_HORIZON - 1)
        score = -1 if encoded_reward == 255 else encoded_reward
        runner_receipt_bytes = _canonical_json(
            {
                "candidate_id": candidate_id,
                "classification": "synthetic_structural_receipt_non_authorizing",
                "label": label,
            }
        )
        facts_by_receipt[runner_receipt_bytes] = (
            candidate_id,
            score,
            _sha256(raw_trace),
            len(raw_trace),
        )
        return bundle._build_bundle(
            candidate_id=candidate_id,
            runner_receipt_bytes=runner_receipt_bytes,
            raw_trace=raw_trace,
            expected_score=score,
        )

    return build


@pytest.fixture
def published_zero_bundle(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
) -> tuple[
    Path,
    bundle.MatchedV3AdapterRewardBundle,
    publication.ContentVerifiedAdapterRewardPublication,
]:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, "published-zero")
    root = tmp_path / "published"
    verified = publication.publish_adapter_reward_bundle(built, root)
    return root, built, verified


def test_descriptor_is_canonical_detached_dependency_bound_and_non_authorizing() -> None:
    raw = publication.canonical_adapter_reward_publication_descriptor_bytes()
    parsed = publication.parse_adapter_reward_publication_descriptor(raw)

    assert len(raw) == 2_739
    assert _sha256(raw) == publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    assert publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256 == (
        "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
    )
    assert publication.adapter_reward_publication_descriptor() == parsed
    assert parsed["status"] == publication.ADAPTER_REWARD_PUBLICATION_STATUS
    assert publication.ADAPTER_REWARD_PUBLICATION_STATUS == "implemented_unexecuted"
    assert set(cast(dict[str, Any], parsed["claims"]).values()) == {False}
    assert b"TO_BE_" not in raw
    assert b'"authority_granted":true' not in raw
    assert b'"scientific_promotion_allowed":true' not in raw
    assert b'"universal_sota_claim_allowed":true' not in raw

    source_bindings = [
        value
        for value in _nested_mappings(parsed)
        if type(value.get("source_path")) is str
        and type(value.get("source_sha256")) is str
    ]
    assert source_bindings
    assert any(
        value["source_path"]
        == "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py"
        for value in source_bindings
    )
    for value in source_bindings:
        source = _ROOT / cast(str, value["source_path"])
        assert _sha256(source.read_bytes()) == value["source_sha256"]

    cast(dict[str, Any], parsed["claims"])["authority_granted"] = True
    assert publication.adapter_reward_publication_descriptor()["claims"][
        "authority_granted"
    ] is False
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.parse_adapter_reward_publication_descriptor(b" " + raw)


@pytest.mark.parametrize(
    ("candidate_id", "encoded_reward"),
    (("adapted_full_rainbow", 0), ("adapted_ppo_gru", 30)),
)
def test_publish_and_load_round_trip_exact_five_files_and_bytes(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    candidate_id: str,
    encoded_reward: int,
) -> None:
    built = synthetic_bundle_factory(candidate_id, encoded_reward, candidate_id)
    root = tmp_path / candidate_id
    published = publication.publish_adapter_reward_bundle(built, root)

    assert type(published) is publication.ContentVerifiedAdapterRewardPublication
    assert published.output_root == root
    assert published.candidate_id == candidate_id
    assert set(path.name for path in root.iterdir()) == _EXPECTED_NAMES
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert root.stat().st_uid == os.geteuid()
    for path in root.iterdir():
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_uid == os.geteuid()

    expected_bytes = {
        "adapter-bundle-manifest.json": built.manifest_bytes,
        "runner-result-receipt.json": built.runner_receipt_bytes,
        "reward-trace.npz": built.reward_artifact_bytes,
        "score-receipt.json": built.score_receipt_bytes,
    }
    for name, raw in expected_bytes.items():
        assert (root / name).read_bytes() == raw

    publication_raw = (root / "publication.json").read_bytes()
    assert _sha256(publication_raw) == published.publication_file_sha256
    parsed_publication = publication.parse_adapter_reward_publication_manifest(
        publication_raw,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    assert parsed_publication["candidate_id"] == published.manifest["candidate_id"]
    assert parsed_publication["publication_body_sha256"] == (
        published.manifest["publication_body_sha256"]
    )
    assert parsed_publication["claims"] == dict(published.manifest["claims"])
    assert tuple(parsed_publication["limitations"]) == published.manifest["limitations"]
    assert set(parsed_publication) == {
        "schema_version",
        "classification",
        "candidate_id",
        "publication_descriptor",
        "adapter_reward_bundle",
        "files",
        "writer_contract",
        "claims",
        "limitations",
        "publication_body_sha256",
    }
    assert parsed_publication["schema_version"] == (
        publication.ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
    )
    assert parsed_publication["publication_descriptor"] == {
        "schema_version": publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "sha256": publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    }
    file_records = cast(dict[str, Any], parsed_publication["files"])
    assert set(file_records) == {
        "adapter_bundle_manifest",
        "runner_result_receipt",
        "reward_trace",
        "score_receipt",
    }
    assert {
        cast(dict[str, Any], record)["path"] for record in file_records.values()
    } == _EXPECTED_NAMES - {"publication.json"}
    for role, record_value in file_records.items():
        record = cast(dict[str, Any], record_value)
        assert set(record) == {"path", "role", "sha256", "size_bytes"}
        assert record["role"] == role

    loaded = publication.load_adapter_reward_bundle_publication(
        root,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    assert loaded == published
    assert loaded.bundle == built
    assert scorer.extract_canonical_reward_trace(loaded.bundle.reward_artifact_bytes)[0] == (
        encoded_reward
    )
    assert bundle.validate_adapter_reward_bundle(loaded.bundle) is loaded.bundle


def test_external_full_file_pin_rejects_coherent_substitution(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
) -> None:
    first_bundle = synthetic_bundle_factory("adapted_full_rainbow", 0, "first")
    second_bundle = synthetic_bundle_factory("adapted_ppo_gru", 1, "second")
    first = publication.publish_adapter_reward_bundle(first_bundle, tmp_path / "first")
    second = publication.publish_adapter_reward_bundle(second_bundle, tmp_path / "second")

    assert first.publication_file_sha256 != second.publication_file_sha256
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            second.output_root,
            expected_publication_file_sha256=first.publication_file_sha256,
        )
    assert (
        publication.load_adapter_reward_bundle_publication(
            second.output_root,
            expected_publication_file_sha256=second.publication_file_sha256,
        ).bundle
        == second_bundle
    )


def test_copied_valid_content_remains_structural_and_non_authorizing(
    tmp_path: Path,
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, original = published_zero_bundle
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    loaded = publication.load_adapter_reward_bundle_publication(
        copied,
        expected_publication_file_sha256=original.publication_file_sha256,
    )

    assert loaded.output_root == copied
    assert loaded.bundle == original.bundle
    assert type(loaded.bundle) is bundle.MatchedV3AdapterRewardBundle
    assert not isinstance(loaded.bundle, full_runner.FullRainbowRunnerResult)
    assert not isinstance(loaded.bundle, ppo_runner.PPOGRURunnerOutcome)
    persisted = b"".join((copied / name).read_bytes() for name in sorted(_EXPECTED_NAMES))
    assert _LIVE_CAPABILITY_SENTINEL not in persisted
    assert set(cast(dict[str, Any], loaded.bundle.manifest()["claims"]).values()) == {
        False
    }
    assert set(cast(dict[str, Any], loaded.manifest["claims"]).values()) == {False}


def test_manifest_loader_returns_detached_immutable_content(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, published = published_zero_bundle
    first = publication.load_adapter_reward_bundle_publication(
        root,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    second = publication.load_adapter_reward_bundle_publication(
        root,
        expected_publication_file_sha256=published.publication_file_sha256,
    )

    assert first == second
    assert first.manifest is not second.manifest
    first_manifest = first.manifest
    with pytest.raises(TypeError):
        first_manifest["claims"] = {}  # type: ignore[index]
    first_claims = cast(Mapping[str, Any], first.manifest["claims"])
    with pytest.raises(TypeError):
        first_claims["authority_granted"] = True  # type: ignore[index]
    assert second.manifest["claims"]["authority_granted"] is False
    forbidden_fields = {
        "authority",
        "capability",
        "execution_token",
        "ingestion_token",
        "production_result",
        "runtime",
    }
    assert forbidden_fields.isdisjoint(field.name for field in dataclasses.fields(first))
    assert forbidden_fields.isdisjoint(
        field.name for field in dataclasses.fields(first.bundle)
    )


def test_publication_parser_requires_exact_external_pin_and_canonical_bytes(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, published = published_zero_bundle
    raw = (root / "publication.json").read_bytes()

    parsed = publication.parse_adapter_reward_publication_manifest(
        raw,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    assert parsed["publication_body_sha256"] == published.publication_body_sha256
    for changed, digest in (
        (b" " + raw, _sha256(b" " + raw)),
        (raw + b"\n", _sha256(raw + b"\n")),
        (raw, "0" * 64),
    ):
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.parse_adapter_reward_publication_manifest(
                changed,
                expected_publication_file_sha256=digest,
            )

    duplicate = raw.replace(
        b"{",
        b'{"schema_version":"duplicate",',
        1,
    )
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.parse_adapter_reward_publication_manifest(
            duplicate,
            expected_publication_file_sha256=_sha256(duplicate),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"classification": "authoritative_evidence"}),
        lambda value: value.update({"candidate_id": "../../outside"}),
        lambda value: cast(dict[str, Any], value["claims"]).update(
            {"authority_granted": True}
        ),
        lambda value: cast(dict[str, Any], value["claims"]).update(
            {"execution_authorized": 0}
        ),
        lambda value: cast(dict[str, Any], value["publication_descriptor"]).update(
            {"sha256": "0" * 64}
        ),
        lambda value: cast(dict[str, Any], value["adapter_reward_bundle"]).update(
            {"manifest_body_sha256": "0" * 64}
        ),
        lambda value: cast(dict[str, Any], value["adapter_reward_bundle"]).update(
            {"manifest_file_sha256": "1" * 64}
        ),
        lambda value: cast(dict[str, Any], value["writer_contract"]).update(
            {"writer_contract_independently_attests_execution": 0}
        ),
        lambda value: cast(dict[str, Any], value["writer_contract"]).update(
            {"staged_files_fsynced": 1}
        ),
        lambda value: _file_record(value, "reward-trace.npz").update(
            {"path": "../reward-trace.npz"}
        ),
        lambda value: _file_record(value, "runner-result-receipt.json").update(
            {"role": "reward_trace"}
        ),
        lambda value: _file_record(value, "score-receipt.json").update(
            {"size_bytes": True}
        ),
    ),
)
def test_rehashed_publication_manifest_semantic_mutations_fail_closed(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    root, _built, _published = published_zero_bundle
    payload = _read_publication(root)
    mutation(payload)
    raw, digest = _rewrite_publication_body(payload)

    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.parse_adapter_reward_publication_manifest(
            raw,
            expected_publication_file_sha256=digest,
        )


@pytest.mark.parametrize(
    "missing_name",
    sorted(_EXPECTED_NAMES),
)
def test_loader_rejects_every_missing_payload(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    missing_name: str,
) -> None:
    root, _built, published = published_zero_bundle
    (root / missing_name).unlink()
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.parametrize("extra_kind", ("file", "directory"))
def test_loader_rejects_extra_or_nested_inventory(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    extra_kind: str,
) -> None:
    root, _built, published = published_zero_bundle
    extra = root / "unexpected"
    if extra_kind == "file":
        extra.write_bytes(b"unexpected")
    else:
        extra.mkdir()
        (extra / "nested").write_bytes(b"unexpected")
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.parametrize(
    "payload_name",
    sorted(_EXPECTED_NAMES - {"publication.json"}),
)
def test_payload_bit_flip_fails_cross_file_replay(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    payload_name: str,
) -> None:
    root, _built, published = published_zero_bundle
    path = root / payload_name
    changed = bytearray(path.read_bytes())
    changed[len(changed) // 2] ^= 1
    path.write_bytes(changed)
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.parametrize(
    ("payload_name", "new_size"),
    (
        ("runner-result-receipt.json", 4 * 1024 * 1024 + 1),
        ("adapter-bundle-manifest.json", 256 * 1024 + 1),
        ("score-receipt.json", 64 * 1024 + 1),
        ("reward-trace.npz", scorer.CANONICAL_NPZ_SIZE_BYTES + 1),
    ),
)
def test_loader_rejects_oversized_sparse_payload_before_content_replay(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    payload_name: str,
    new_size: int,
) -> None:
    root, _built, published = published_zero_bundle
    os.truncate(root / payload_name, new_size)
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.parametrize("replacement", ("symlink", "hardlink", "directory", "socket"))
def test_loader_rejects_payload_aliases_and_special_files(
    tmp_path: Path,
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    replacement: str,
) -> None:
    root, _built, published = published_zero_bundle
    target = root / "score-receipt.json"
    original = target.read_bytes()
    backup = tmp_path / "score-receipt-backup.json"
    backup.write_bytes(original)
    target.unlink()

    opened_socket: socket.socket | None = None
    if replacement == "symlink":
        target.symlink_to(backup)
    elif replacement == "hardlink":
        os.link(backup, target)
    elif replacement == "directory":
        target.mkdir()
    else:
        opened_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        opened_socket.bind(str(target))
    try:
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.load_adapter_reward_bundle_publication(
                root,
                expected_publication_file_sha256=published.publication_file_sha256,
            )
    finally:
        if opened_socket is not None:
            opened_socket.close()


@pytest.mark.parametrize("target_name", (None, "runner-result-receipt.json"))
def test_loader_rejects_group_or_world_writable_publication_content(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    target_name: str | None,
) -> None:
    root, _built, published = published_zero_bundle
    target = root if target_name is None else root / target_name
    target.chmod(stat.S_IMODE(target.stat().st_mode) | 0o022)
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux FIFO contract")
def test_loader_rejects_fifo_without_blocking(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, published = published_zero_bundle
    publication_path = root / "publication.json"
    publication_path.unlink()
    os.mkfifo(publication_path, 0o600)
    script = """
from pathlib import Path
import sys
from alberta_framework.benchmarks import forager_matched_v3_adapter_reward_publication as p
try:
    p.load_adapter_reward_bundle_publication(
        Path(sys.argv[1]), expected_publication_file_sha256=sys.argv[2]
    )
except p.ForagerMatchedV3AdapterRewardPublicationError:
    raise SystemExit(0)
raise SystemExit(2)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            published.publication_file_sha256,
        ],
        check=False,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("destination_kind", ("directory", "file", "symlink", "fifo", "socket"))
def test_publisher_preserves_every_preexisting_destination(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    destination_kind: str,
) -> None:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, destination_kind)
    destination = tmp_path / "destination"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel")
    opened_socket: socket.socket | None = None
    if destination_kind == "directory":
        destination.mkdir()
        (destination / "keep").write_bytes(b"keep")
    elif destination_kind == "file":
        destination.write_bytes(b"keep")
    elif destination_kind == "symlink":
        destination.symlink_to(sentinel)
    elif destination_kind == "fifo":
        os.mkfifo(destination, 0o600)
    else:
        opened_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        opened_socket.bind(str(destination))
    before = destination.lstat()

    try:
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.publish_adapter_reward_bundle(built, destination)
        after = destination.lstat()
        assert (after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode)) == (
            before.st_dev,
            before.st_ino,
            stat.S_IFMT(before.st_mode),
        )
        if destination_kind == "directory":
            assert (destination / "keep").read_bytes() == b"keep"
        elif destination_kind == "file":
            assert destination.read_bytes() == b"keep"
        elif destination_kind == "symlink":
            assert destination.readlink() == sentinel
            assert sentinel.read_bytes() == b"sentinel"
    finally:
        if opened_socket is not None:
            opened_socket.close()


def test_mixed_publication_rehashed_at_outer_layer_still_fails_bundle_replay(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
) -> None:
    first_bundle = synthetic_bundle_factory("adapted_full_rainbow", 0, "first")
    second_bundle = synthetic_bundle_factory("adapted_full_rainbow", 1, "second")
    first = publication.publish_adapter_reward_bundle(first_bundle, tmp_path / "first")
    second = publication.publish_adapter_reward_bundle(second_bundle, tmp_path / "second")
    root = first.output_root

    replacement = (second.output_root / "runner-result-receipt.json").read_bytes()
    (root / "runner-result-receipt.json").write_bytes(replacement)
    outer = _read_publication(root)
    _replace_file_record(outer, "runner-result-receipt.json", replacement)
    new_pin = _write_rehashed_publication(root, outer)

    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=new_pin,
        )


def test_rehashed_inner_and_outer_manifests_cannot_hide_runner_trace_disagreement(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
) -> None:
    first_bundle = synthetic_bundle_factory("adapted_full_rainbow", 0, "first")
    second_bundle = synthetic_bundle_factory("adapted_full_rainbow", 30, "second")
    first = publication.publish_adapter_reward_bundle(first_bundle, tmp_path / "first")
    second = publication.publish_adapter_reward_bundle(second_bundle, tmp_path / "second")
    root = first.output_root

    runner_bytes = (second.output_root / "runner-result-receipt.json").read_bytes()
    (root / "runner-result-receipt.json").write_bytes(runner_bytes)
    inner_path = root / "adapter-bundle-manifest.json"
    inner = cast(dict[str, Any], json.loads(inner_path.read_bytes()))
    inner_runner = cast(dict[str, Any], inner["runner_receipt"])
    inner_runner["sha256"] = _sha256(runner_bytes)
    inner_runner["size_bytes"] = len(runner_bytes)
    inner_bytes = _rewrite_bundle_manifest_body(inner)
    inner_path.write_bytes(inner_bytes)

    outer = _read_publication(root)
    _replace_file_record(outer, "runner-result-receipt.json", runner_bytes)
    _replace_file_record(outer, "adapter-bundle-manifest.json", inner_bytes)
    inner_body_sha256 = cast(str, json.loads(inner_bytes)["manifest_body_sha256"])
    outer_bundle = cast(dict[str, Any], outer["adapter_reward_bundle"])
    outer_bundle["manifest_body_sha256"] = inner_body_sha256
    new_pin = _write_rehashed_publication(root, outer)

    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=new_pin,
        )


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires procfs")
def test_repeated_loader_failures_do_not_leak_file_descriptors(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, published = published_zero_bundle
    (root / "score-receipt.json").unlink()
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )
    before = len(tuple(Path("/proc/self/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.load_adapter_reward_bundle_publication(
                root,
                expected_publication_file_sha256=published.publication_file_sha256,
            )
    after = len(tuple(Path("/proc/self/fd").iterdir()))
    assert after == before


def test_root_aliases_and_non_directories_fail_closed(
    tmp_path: Path,
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, published = published_zero_bundle
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    regular = tmp_path / "regular"
    regular.write_bytes(b"not a directory")
    for candidate in (alias, regular):
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.load_adapter_reward_bundle_publication(
                candidate,
                expected_publication_file_sha256=published.publication_file_sha256,
            )


def test_manifest_size_and_json_complexity_are_bounded() -> None:
    oversized = b"{" + b" " * (1024 * 1024) + b"}"
    huge_integer = b'{"value":' + b"9" * 5_000 + b"}"
    deeply_nested = b"[" * 2_000 + b"]" * 2_000
    for raw in (oversized, huge_integer, deeply_nested):
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.parse_adapter_reward_publication_manifest(
                raw,
                expected_publication_file_sha256=_sha256(raw),
            )


def test_source_has_no_execution_subprocess_or_authority_shortcut() -> None:
    source = (
        _ROOT
        / "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
    ).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "Popen(" not in source
    assert "execution_authorized\": True" not in source
    assert "scientific_promotion_allowed\": True" not in source
    assert "universal_sota_claim_allowed\": True" not in source
    assert "TO_BE_" not in source


def test_valid_candidate_substitution_fails_full_bundle_replay(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
) -> None:
    root, _built, _published = published_zero_bundle
    payload = _read_publication(root)
    assert payload["candidate_id"] == "adapted_full_rainbow"
    payload["candidate_id"] = "adapted_ppo_gru"
    new_pin = _write_rehashed_publication(root, payload)

    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=new_pin,
        )


def test_successful_move_then_injected_exception_is_uncertain_and_recoverable(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, "move-gap")
    root = tmp_path / "move-gap"
    real_publish = publication._publish_verified_no_replace

    def publish_then_fail(*args: Any, **kwargs: Any) -> None:
        real_publish(*args, **kwargs)
        raise RuntimeError("injected after verified move")

    monkeypatch.setattr(publication, "_publish_verified_no_replace", publish_then_fail)
    with pytest.raises(
        publication.PublishedAdapterRewardPublicationUncertainError
    ) as captured:
        publication.publish_adapter_reward_bundle(built, root)

    error = captured.value
    assert error.destination == root
    assert error.publication_file_sha256 == _sha256(
        (root / publication.PUBLICATION_FILENAME).read_bytes()
    )
    parsed = publication.parse_adapter_reward_publication_manifest(
        (root / publication.PUBLICATION_FILENAME).read_bytes(),
        expected_publication_file_sha256=error.publication_file_sha256,
    )
    assert error.publication_body_sha256 == parsed["publication_body_sha256"]
    monkeypatch.setattr(publication, "_publish_verified_no_replace", real_publish)
    assert (
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=error.publication_file_sha256,
        ).bundle
        == built
    )


def test_parent_fsync_failure_is_uncertain_with_recovery_pin(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = synthetic_bundle_factory("adapted_ppo_gru", 0, "parent-fsync")
    root = tmp_path / "parent-fsync"

    def fail_parent_sync(parent: object) -> None:
        del parent
        raise OSError(errno.EIO, "injected parent fsync failure")

    monkeypatch.setattr(publication, "_sync_publication_parent", fail_parent_sync)
    with pytest.raises(
        publication.PublishedAdapterRewardPublicationUncertainError
    ) as captured:
        publication.publish_adapter_reward_bundle(built, root)

    assert root.is_dir()
    assert captured.value.publication_file_sha256 == _sha256(
        (root / publication.PUBLICATION_FILENAME).read_bytes()
    )


def test_rename_moves_then_reports_error_is_uncertain_and_preserves_destination(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, "rename-ambiguous")
    root = tmp_path / "rename-ambiguous"
    real_rename = publication._rename_no_replace

    def rename_then_report_error(
        parent: Any,
        source_name: str,
        destination_name: str,
    ) -> None:
        real_rename(parent, source_name, destination_name)
        raise OSError(errno.EIO, "injected ambiguous rename result")

    monkeypatch.setattr(publication, "_rename_no_replace", rename_then_report_error)
    with pytest.raises(
        publication.PublishedAdapterRewardPublicationUncertainError
    ) as captured:
        publication.publish_adapter_reward_bundle(built, root)

    assert root.is_dir()
    assert captured.value.publication_file_sha256 == _sha256(
        (root / publication.PUBLICATION_FILENAME).read_bytes()
    )
    assert not tuple(tmp_path.glob(".forager-v3-adapter-partial-*"))


@pytest.mark.parametrize("stage", ("write", "staged_replay", "tree_sync"))
def test_prepublication_oserrors_are_normalized_and_owned_staging_is_cleaned(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, stage)
    root = tmp_path / f"pre-{stage}"

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise OSError(errno.EIO, f"injected {stage} failure")

    target = {
        "write": "_write_exclusive_at",
        "staged_replay": "_load_from_open_root",
        "tree_sync": "_durably_sync_open_tree",
    }[stage]
    monkeypatch.setattr(publication, target, fail)
    with pytest.raises(
        publication.ForagerMatchedV3AdapterRewardPublicationError
    ) as captured:
        publication.publish_adapter_reward_bundle(built, root)

    assert not isinstance(
        captured.value,
        publication.PublishedAdapterRewardPublicationUncertainError,
    )
    assert not root.exists()
    assert not tuple(tmp_path.glob(".forager-v3-adapter-partial-*"))


def test_concurrent_destination_wins_without_replacement(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = synthetic_bundle_factory("adapted_ppo_gru", 0, "rename-race")
    root = tmp_path / "rename-race"
    real_rename = publication._rename_no_replace

    def create_winner_then_rename(
        parent: Any,
        source_name: str,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, 0o700, dir_fd=parent.descriptor)
        real_rename(parent, source_name, destination_name)

    monkeypatch.setattr(publication, "_rename_no_replace", create_winner_then_rename)
    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.publish_adapter_reward_bundle(built, root)

    assert root.is_dir()
    assert not tuple(root.iterdir())
    assert not tuple(tmp_path.glob(".forager-v3-adapter-partial-*"))


@pytest.mark.parametrize("failure", ("stat", "open", "fchmod"))
def test_partial_staging_initialization_failure_is_cleaned(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    built = synthetic_bundle_factory("adapted_full_rainbow", 0, failure)
    root = tmp_path / f"staging-{failure}"
    if failure == "stat":
        real_stat = _PUBLICATION_OS.stat
        injected = False

        def fail_first_staging_stat(path: object, *args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if (
                not injected
                and type(path) is str
                and path.startswith(".forager-v3-adapter-partial-")
            ):
                injected = True
                raise OSError(errno.EIO, "injected first staging stat failure")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(_PUBLICATION_OS, "stat", fail_first_staging_stat)
    elif failure == "open":
        real_open = publication._open_stable_directory_at
        injected = False

        def fail_first_staging_open(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if not injected and args[3] == "adapter publication staging directory":
                injected = True
                raise OSError(errno.EIO, "injected staging open failure")
            return real_open(*args, **kwargs)

        monkeypatch.setattr(
            publication,
            "_open_stable_directory_at",
            fail_first_staging_open,
        )
    else:
        real_fchmod = _PUBLICATION_OS.fchmod
        injected = False

        def fail_first_fchmod(descriptor: int, mode: int) -> None:
            nonlocal injected
            if not injected:
                injected = True
                raise OSError(errno.EIO, "injected staging fchmod failure")
            real_fchmod(descriptor, mode)

        monkeypatch.setattr(_PUBLICATION_OS, "fchmod", fail_first_fchmod)

    with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
        publication.publish_adapter_reward_bundle(built, root)
    assert not root.exists()
    assert not tuple(tmp_path.glob(".forager-v3-adapter-partial-*"))


def test_publisher_rejects_untrusted_parent_namespace(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
) -> None:
    built = synthetic_bundle_factory("adapted_ppo_gru", 0, "unsafe-parent")
    parent = tmp_path / "unsafe-parent"
    parent.mkdir()
    parent.chmod(0o777)
    try:
        with pytest.raises(publication.ForagerMatchedV3AdapterRewardPublicationError):
            publication.publish_adapter_reward_bundle(built, parent / "publication")
        assert not (parent / "publication").exists()
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize("operation", ("scandir", "read"))
def test_loader_normalizes_post_open_oserrors(
    published_zero_bundle: tuple[
        Path,
        bundle.MatchedV3AdapterRewardBundle,
        publication.ContentVerifiedAdapterRewardPublication,
    ],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root, _built, published = published_zero_bundle

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise OSError(errno.EIO, f"injected {operation} failure")

    monkeypatch.setattr(_PUBLICATION_OS, operation, fail)
    with pytest.raises(
        publication.ForagerMatchedV3AdapterRewardPublicationError
    ) as captured:
        publication.load_adapter_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )
    assert not isinstance(captured.value, OSError)


def test_close_errors_do_not_mask_fsynced_publication(
    tmp_path: Path,
    synthetic_bundle_factory: SyntheticBundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = synthetic_bundle_factory("adapted_ppo_gru", 0, "close-errors")
    root = tmp_path / "close-errors"
    real_close = _PUBLICATION_OS.close
    closed: list[int] = []

    def close_then_report_error(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        raise OSError(errno.EIO, "injected post-close report")

    monkeypatch.setattr(_PUBLICATION_OS, "close", close_then_report_error)
    published = publication.publish_adapter_reward_bundle(built, root)
    assert len(closed) >= 7
    assert published.bundle == built
    assert root.is_dir()


@pytest.mark.parametrize(
    ("candidate_id", "receipt_builder"),
    (
        ("adapted_full_rainbow", _structural_full_rainbow_result_receipt),
        ("adapted_ppo_gru", _structural_ppo_result_receipt),
    ),
)
def test_publication_round_trip_uses_real_frozen_runner_receipt_parsers(
    tmp_path: Path,
    candidate_id: str,
    receipt_builder: Callable[[bytes], bytes],
) -> None:
    raw_trace = bytes(protocol.MATCHED_V3_HORIZON)
    receipt = receipt_builder(raw_trace)
    built = bundle._build_bundle(
        candidate_id=candidate_id,
        runner_receipt_bytes=receipt,
        raw_trace=raw_trace,
        expected_score=0,
    )
    published = publication.publish_adapter_reward_bundle(
        built,
        tmp_path / candidate_id,
    )

    loaded = publication.load_adapter_reward_bundle_publication(
        published.output_root,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    assert loaded.bundle == built
    assert loaded.candidate_id == candidate_id


def test_structural_publication_replays_in_fresh_process_without_monkeypatch(
    tmp_path: Path,
) -> None:
    raw_trace = bytes(protocol.MATCHED_V3_HORIZON)
    receipt = _structural_ppo_result_receipt(raw_trace)
    built = bundle._build_bundle(
        candidate_id="adapted_ppo_gru",
        runner_receipt_bytes=receipt,
        raw_trace=raw_trace,
        expected_score=0,
    )
    published = publication.publish_adapter_reward_bundle(built, tmp_path / "fresh")
    script = """
import sys
from pathlib import Path
from alberta_framework.benchmarks import forager_matched_v3_adapter_reward_publication as p

loaded = p.load_adapter_reward_bundle_publication(
    Path(sys.argv[1]),
    expected_publication_file_sha256=sys.argv[2],
)
assert loaded.candidate_id == "adapted_ppo_gru"
assert set(loaded.manifest["claims"].values()) == {False}
assert set(loaded.bundle.manifest()["claims"].values()) == {False}
print(loaded.publication_file_sha256)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(published.output_root),
            published.publication_file_sha256,
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == published.publication_file_sha256
