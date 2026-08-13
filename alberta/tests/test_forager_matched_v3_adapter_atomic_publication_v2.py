"""Focused contracts for the additive adapter atomic-v2 publisher.

All persisted content in these tests is synthetic.  No candidate workload is run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PosixPath
from typing import Any, Literal, cast

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_atomic_publication as atomic
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_atomic_publication_v2 as publication,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as bundle,
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
_SOURCE_TREE_SHA256 = hashlib.sha256(b"synthetic-local-source-tree").hexdigest()
_EXACT_NAMES = (
    "publication.json",
    "adapter-bundle-manifest.json",
    "runner-result-receipt.json",
    "reward-trace.npz",
    "score-receipt.json",
)


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
    """Build parser-valid, permanently nonexecuted receipt bytes."""

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


@pytest.fixture
def synthetic_bundle(monkeypatch: pytest.MonkeyPatch) -> bundle.MatchedV3AdapterRewardBundle:
    trace = bytes(protocol.MATCHED_V3_HORIZON)
    receipt = _structural_ppo_result_receipt(trace)

    def receipt_facts(
        candidate_id: str, supplied: bytes
    ) -> tuple[dict[str, Any], int, str, int]:
        assert candidate_id == "adapted_ppo_gru"
        assert supplied == receipt
        return {}, 0, _sha256(trace), len(trace)

    with monkeypatch.context() as scoped:
        scoped.setattr(bundle, "_runner_receipt_facts", receipt_facts)
        return bundle._build_bundle(
            candidate_id="adapted_ppo_gru",
            runner_receipt_bytes=receipt,
            raw_trace=trace,
            expected_score=0,
        )


def _safe_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "adapter-publications"
    parent.mkdir(mode=0o700)
    return parent


def test_descriptor_is_exact_source_bound_honest_and_non_authorizing() -> None:
    raw = publication.canonical_adapter_atomic_publication_v2_descriptor_bytes()
    parsed = publication.parse_adapter_atomic_publication_v2_descriptor(raw)

    assert _sha256(raw) == publication.ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256
    assert parsed == publication.adapter_atomic_publication_v2_descriptor()
    assert parsed["status"] == (
        "implemented_unexecuted_unqualified_surfaces_host_isolation_unproven"
    )
    assert set(cast(dict[str, Any], parsed["claims"]).values()) == {False}
    readiness = cast(dict[str, Any], parsed["readiness"])
    assert set(readiness.values()) == {False}
    limitations = cast(list[str], parsed["limitations"])
    assert any("digests and sizes" in item for item in limitations)
    assert any("Full Rainbow" in item and "unqualified" in item for item in limitations)
    assert any("PPO-GRU" in item and "unqualified" in item for item in limitations)
    assert any("cgroup" in item for item in limitations)
    assert any("normal reload" in item and "fresh host" in item for item in limitations)
    assert any("stale or unqualified image" in item for item in limitations)
    assert not any("score-free" in item or "score-opaque" in item for item in limitations)
    registry = cast(dict[str, Any], parsed["qualification_registry"])
    assert registry == {
        "implemented_strict_qualification_publisher": False,
        "only_executable_public_surfaces_are_explicitly_unqualified": True,
        "v3_adapter_publisher_registry_gap_remains": True,
    }
    fused = cast(dict[str, Any], parsed["fused_contract"])
    assert fused["one_atomic_publish_call_per_public_invocation"] is True
    assert fused["global_retry_or_single_use_coordinator"] is False
    assert publication.__all__ == [
        "ADAPTER_ATOMIC_PUBLICATION_V2_CANDIDATE_IDS",
        "ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SCHEMA_VERSION",
        "ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256",
        "ADAPTER_ATOMIC_PUBLICATION_V2_FILENAMES",
        "ADAPTER_ATOMIC_PUBLICATION_V2_METADATA_SCHEMA_VERSION",
        "ADAPTER_ATOMIC_PUBLICATION_V2_STATUS",
        "ForagerMatchedV3AdapterAtomicPublicationV2Error",
        "ForagerMatchedV3AdapterAtomicPublicationV2CollisionError",
        "MatchedV3AdapterAtomicPublicationFileV2",
        "MatchedV3AdapterAtomicPublicationMetadataV2",
        "PublishedAdapterAtomicPublicationV2UncertainError",
        "adapter_atomic_publication_v2_descriptor",
        "canonical_adapter_atomic_publication_v2_descriptor_bytes",
        "canonical_adapter_atomic_publication_v2_metadata_bytes",
        "parse_adapter_atomic_publication_v2_descriptor",
        "parse_adapter_atomic_publication_v2_metadata",
        "reload_matched_v3_adapter_atomic_publication_v2",
        "run_and_publish_matched_v3_full_rainbow_adapter_v2",
        "run_and_publish_matched_v3_ppo_gru_adapter_v2",
    ]
    assert not any(name.startswith("_") for name in publication.__all__)

    for binding in cast(list[dict[str, Any]], parsed["source_bindings"]):
        source = _ROOT / binding["implementation_path"]
        assert _sha256(source.read_bytes()) == binding["implementation_source_sha256"]

    for name in (
        "run_and_publish_matched_v3_full_rainbow_adapter_v2",
        "run_and_publish_matched_v3_ppo_gru_adapter_v2",
    ):
        signature = inspect.signature(getattr(publication, name))
        forbidden = {"result", "outcome", "bundle", "payload", "callback", "bytes"}
        assert forbidden.isdisjoint(signature.parameters)
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )

    for dependency in (
        _ROOT
        / "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py",
        _ROOT / "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py",
        _ROOT / "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py",
    ):
        assert "forager_matched_v3_adapter_atomic_publication_v2" not in dependency.read_text()

    implementation = (
        _ROOT
        / "alberta_framework/benchmarks/forager_matched_v3_adapter_atomic_publication_v2.py"
    ).read_text()
    assert publication.ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256 in implementation
    assert "ADAPTER_ATOMIC_PUBLICATION_V2_DESCRIPTOR_SHA256: Final = hashlib" not in implementation

    detached = publication.adapter_atomic_publication_v2_descriptor()
    cast(dict[str, Any], detached["claims"])["authority_granted"] = True
    assert publication.adapter_atomic_publication_v2_descriptor() == parsed
    assert publication.canonical_adapter_atomic_publication_v2_descriptor_bytes() == raw
    reformatted = json.dumps(json.loads(raw), indent=2, sort_keys=True).encode("ascii")
    tampered = raw.replace(b'"qualification_ready":false', b'"qualification_ready":true')
    for rejected in (reformatted, tampered, raw + b"\n"):
        with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
            publication.parse_adapter_atomic_publication_v2_descriptor(rejected)


@pytest.mark.parametrize("drift", ["source", "descriptor", "function", "default"])
def test_dependency_drift_fails_during_closure_load(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if drift == "source":
        specs = tuple(
            (
                component,
                module_name,
                path,
                "1" * 64 if component == "full_rainbow_runner" else digest,
            )
            for component, module_name, path, digest in publication._SOURCE_BINDING_SPECS
        )
        monkeypatch.setattr(publication, "_SOURCE_BINDING_SPECS", specs)
    elif drift == "descriptor":
        monkeypatch.setattr(full_runner, "FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256", "1" * 64)
    elif drift == "function":

        def replaced_runner(**_kwargs: object) -> None:
            raise AssertionError("a drifted runner must never execute")

        monkeypatch.setattr(full_runner, "run_matched_v3_full_rainbow", replaced_runner)
    else:
        monkeypatch.setattr(
            full_runner.run_matched_v3_full_rainbow,
            "__kwdefaults__",
            {"unqualified_engineering": True},
        )

    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication._load_dependency_closure()


def test_dependency_code_bindings_are_portable_to_a_second_absolute_root(
    tmp_path: Path,
) -> None:
    relocated_root = tmp_path / "relocated-source-root"
    shutil.copytree(
        _ROOT / "alberta_framework",
        relocated_root / "alberta_framework",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    script = """
from pathlib import Path
from alberta_framework.benchmarks import forager_matched_v3_adapter_atomic_publication_v2 as p
closure = p._load_dependency_closure()
print(Path(p.__file__).resolve())
print(Path(closure.full_runner.__file__).resolve())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=relocated_root,
        env={**os.environ, "PYTHONPATH": str(relocated_root)},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    loaded_paths = tuple(Path(line) for line in completed.stdout.strip().splitlines())
    assert len(loaded_paths) == 2
    assert all(path.is_relative_to(relocated_root) for path in loaded_paths)


def test_explicit_opt_ins_fail_before_any_unqualified_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)

    def forbidden_dependency_load() -> None:
        raise AssertionError("dependency load happened before both opt-ins")

    monkeypatch.setattr(publication, "_load_dependency_closure", forbidden_dependency_load)
    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication.run_and_publish_matched_v3_full_rainbow_adapter_v2(
            environment_seed=0,
            agent_seed=0,
            publication_parent=parent,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            explicit_unqualified_execution=False,
            explicit_publication_opt_in=True,
        )
    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication.run_and_publish_matched_v3_ppo_gru_adapter_v2(
            environment_seed=0,
            agent_seed=0,
            publication_parent=parent,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            explicit_unqualified_execution=True,
            explicit_publication_opt_in=False,
        )
    assert list(parent.iterdir()) == []


def test_publication_parent_is_rejected_before_heavy_runner_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = _safe_parent(tmp_path)
    unsafe_mode = tmp_path / "unsafe-mode"
    unsafe_mode.mkdir(mode=0o700)
    unsafe_mode.chmod(0o755)
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(safe, target_is_directory=True)

    class PathSubclass(PosixPath):
        pass

    def forbidden_dependency_load() -> None:
        raise AssertionError("dependency load happened before parent preflight")

    monkeypatch.setattr(publication, "_load_dependency_closure", forbidden_dependency_load)
    for invalid in (
        Path("relative-parent"),
        Path("/"),
        PathSubclass(safe),
        unsafe_mode,
        symlink_parent,
        safe / ".." / safe.name,
        tmp_path / "missing-parent",
    ):
        with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
            publication.run_and_publish_matched_v3_full_rainbow_adapter_v2(
                environment_seed=0,
                agent_seed=0,
                publication_parent=invalid,
                expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
                explicit_unqualified_execution=True,
                explicit_publication_opt_in=True,
            )
    assert list(safe.iterdir()) == []


def test_private_synthetic_publish_and_public_reload_return_metadata_only(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
) -> None:
    parent = _safe_parent(tmp_path)
    metadata = publication._publish_validated_adapter_bundle(
        adapter_bundle=synthetic_bundle,
        publication_parent=parent,
        expected_environment_seed=0,
        expected_agent_seed=0,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        required_pid=os.getpid(),
    )

    assert type(metadata) is publication.MatchedV3AdapterAtomicPublicationMetadataV2
    assert not isinstance(metadata, bytes)
    assert not hasattr(metadata, "bundle")
    assert not hasattr(metadata, "files_bytes")
    assert all(
        not isinstance(getattr(metadata, field.name), bytes)
        for field in dataclasses.fields(metadata)
    )
    assert all(
        getattr(metadata, field.name) is not synthetic_bundle
        for field in dataclasses.fields(metadata)
    )
    assert metadata.operation == "published"
    assert metadata.operation_pid == os.getpid()
    assert metadata.candidate_id == "adapted_ppo_gru"
    assert metadata.environment_seed == 0
    assert metadata.agent_seed == 0
    assert metadata.file_count == 5
    assert tuple(item.name for item in metadata.files) == _EXACT_NAMES
    assert tuple(sorted(path.name for path in metadata.publication_root.iterdir())) == tuple(
        sorted(_EXACT_NAMES)
    )
    for record in metadata.files:
        file_path = metadata.publication_root / record.name
        raw_file = file_path.read_bytes()
        file_stat = file_path.stat(follow_symlinks=False)
        assert len(raw_file) == record.size_bytes
        assert _sha256(raw_file) == record.sha256
        assert stat.S_ISREG(file_stat.st_mode)
        assert stat.S_IMODE(file_stat.st_mode) == 0o600
        assert file_stat.st_nlink == 1
        assert file_stat.st_uid == os.geteuid()
        assert (file_stat.st_uid, file_stat.st_gid) == (
            metadata.publication_root.stat().st_uid,
            metadata.publication_root.stat().st_gid,
        )

    raw = publication.canonical_adapter_atomic_publication_v2_metadata_bytes(metadata)
    file_sha256 = _sha256(raw)
    parsed = publication.parse_adapter_atomic_publication_v2_metadata(
        raw,
        expected_full_file_sha256=file_sha256,
    )
    assert parsed == metadata
    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication.parse_adapter_atomic_publication_v2_metadata(
            raw,
            expected_full_file_sha256=metadata.metadata_body_sha256,
        )

    reloaded = publication.reload_matched_v3_adapter_atomic_publication_v2(
        publication_parent=parent,
        expected_address=metadata.address,
        expected_file_records=metadata.files,
        expected_candidate_id="adapted_ppo_gru",
        expected_environment_seed=0,
        expected_agent_seed=0,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
    )
    assert type(reloaded) is publication.MatchedV3AdapterAtomicPublicationMetadataV2
    assert reloaded.operation == "reloaded"
    assert reloaded.content_projection_sha256 == metadata.content_projection_sha256
    assert reloaded.inventory_sha256 == metadata.inventory_sha256
    assert not hasattr(reloaded, "bundle")


def test_metadata_parser_and_reload_fail_closed(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
) -> None:
    parent = _safe_parent(tmp_path)
    metadata = publication._publish_validated_adapter_bundle(
        adapter_bundle=synthetic_bundle,
        publication_parent=parent,
        expected_environment_seed=0,
        expected_agent_seed=0,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        required_pid=os.getpid(),
    )
    raw = publication.canonical_adapter_atomic_publication_v2_metadata_bytes(metadata)
    duplicate = raw.replace(
        b'{"address":',
        b'{"address":"' + metadata.address.encode() + b'","address":',
        1,
    )
    for changed in (b" " + raw, duplicate, raw.replace(b'"file_count":5', b'"file_count":5.0')):
        with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
            publication.parse_adapter_atomic_publication_v2_metadata(
                changed,
                expected_full_file_sha256=_sha256(changed),
            )

    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication.reload_matched_v3_adapter_atomic_publication_v2(
            publication_parent=parent,
            expected_address=metadata.address,
            expected_file_records=metadata.files,
            expected_candidate_id="adapted_ppo_gru",
            expected_environment_seed=1,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        )

    for index, record in enumerate(metadata.files):
        changed_record = dataclasses.replace(record, sha256="1" * 64)
        changed_records = list(metadata.files)
        changed_records[index] = changed_record
        with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
            publication.reload_matched_v3_adapter_atomic_publication_v2(
                publication_parent=parent,
                expected_address=metadata.address,
                expected_file_records=tuple(changed_records),
                expected_candidate_id="adapted_ppo_gru",
                expected_environment_seed=0,
                expected_agent_seed=0,
                expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            )

    score_path = metadata.publication_root / "score-receipt.json"
    score_raw = score_path.read_bytes()
    score_path.write_bytes(bytes([score_raw[0] ^ 1]) + score_raw[1:])
    with pytest.raises(publication.ForagerMatchedV3AdapterAtomicPublicationV2Error):
        publication.reload_matched_v3_adapter_atomic_publication_v2(
            publication_parent=parent,
            expected_address=metadata.address,
            expected_file_records=metadata.files,
            expected_candidate_id="adapted_ppo_gru",
            expected_environment_seed=0,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        )


def test_publish_requires_same_pid(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
) -> None:
    parent = _safe_parent(tmp_path)
    with pytest.raises(
        publication.ForagerMatchedV3AdapterAtomicPublicationV2Error,
        match="PID",
    ):
        publication._publish_validated_adapter_bundle(
            adapter_bundle=synthetic_bundle,
            publication_parent=parent,
            expected_environment_seed=0,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            required_pid=os.getpid() + 1,
        )
    assert list(parent.iterdir()) == []


def test_collision_is_typed_and_never_retried(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
) -> None:
    parent = _safe_parent(tmp_path)
    first = publication._publish_validated_adapter_bundle(
        adapter_bundle=synthetic_bundle,
        publication_parent=parent,
        expected_environment_seed=0,
        expected_agent_seed=0,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        required_pid=os.getpid(),
    )
    before = {
        path.name: _sha256(path.read_bytes())
        for path in first.publication_root.iterdir()
    }
    with pytest.raises(
        publication.ForagerMatchedV3AdapterAtomicPublicationV2CollisionError
    ):
        publication._publish_validated_adapter_bundle(
            adapter_bundle=synthetic_bundle,
            publication_parent=parent,
            expected_environment_seed=0,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            required_pid=os.getpid(),
        )
    after = {
        path.name: _sha256(path.read_bytes())
        for path in first.publication_root.iterdir()
    }
    assert after == before


def test_injected_success_seam_calls_atomic_publish_exactly_once(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    closure = publication._load_dependency_closure()
    exact_atomic_publish_once = publication._atomic_publish_once
    calls = 0

    def counting_publish(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return exact_atomic_publish_once(*args, **kwargs)

    monkeypatch.setattr(publication, "_atomic_publish_once", counting_publish)
    metadata = publication._publish_adapter_bundle_with_closure(
        closure=closure,
        adapter_bundle=synthetic_bundle,
        publication_parent=parent,
        expected_environment_seed=0,
        expected_agent_seed=0,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        required_pid=os.getpid(),
    )
    assert calls == 1
    assert metadata.operation == "published"
    assert metadata.publication_root.is_dir()


@pytest.mark.parametrize("committed", [True, None])
def test_atomic_uncertainty_preserves_state_and_is_not_retried(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
    committed: Literal[True] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    closure = publication._load_dependency_closure()
    calls = 0
    observed_address = ""

    def uncertain_publish(
        _closure: object,
        _parent: Path,
        *,
        address: str,
        records: object,
        payloads: object,
    ) -> None:
        del records, payloads
        nonlocal calls, observed_address
        calls += 1
        observed_address = address
        raise atomic.ForagerMatchedV3AtomicPublicationUncertainError(
            parent / address,
            address,
            "synthetic atomic uncertainty",
            committed=committed,
        )

    monkeypatch.setattr(publication, "_atomic_publish_once", uncertain_publish)
    with pytest.raises(
        publication.PublishedAdapterAtomicPublicationV2UncertainError
    ) as caught:
        publication._publish_adapter_bundle_with_closure(
            closure=closure,
            adapter_bundle=synthetic_bundle,
            publication_parent=parent,
            expected_environment_seed=0,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            required_pid=os.getpid(),
        )
    assert calls == 1
    assert caught.value.committed is committed
    assert caught.value.address == observed_address
    assert caught.value.destination == parent / observed_address
    assert list(parent.iterdir()) == []


def test_post_commit_metadata_failure_reports_committed_uncertainty(
    tmp_path: Path,
    synthetic_bundle: bundle.MatchedV3AdapterRewardBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _safe_parent(tmp_path)
    closure = publication._load_dependency_closure()

    def fail_metadata_projection(**_kwargs: object) -> None:
        raise RuntimeError("synthetic post-commit metadata failure")

    monkeypatch.setattr(
        publication,
        "_metadata_from_flat_publication",
        fail_metadata_projection,
    )
    with pytest.raises(
        publication.PublishedAdapterAtomicPublicationV2UncertainError
    ) as caught:
        publication._publish_adapter_bundle_with_closure(
            closure=closure,
            adapter_bundle=synthetic_bundle,
            publication_parent=parent,
            expected_environment_seed=0,
            expected_agent_seed=0,
            expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
            required_pid=os.getpid(),
        )
    assert caught.value.committed is True
    assert caught.value.destination == parent / caught.value.address
    assert caught.value.destination.is_dir()
    assert tuple(
        sorted(path.name for path in caught.value.destination.iterdir())
    ) == tuple(sorted(_EXACT_NAMES))
