"""Filesystem and structural contracts for compiled reward publication."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_reward_bundle as bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_reward_publication as publication,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_compiled_runner as compiled_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_FILES = {
    "publication.json",
    "compiled-bundle-manifest.json",
    "runner-result-receipt.json",
    "runtime-identity.json",
    "reward-trace.npz",
    "score-receipt.json",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _synthetic_runtime_identity_bytes() -> bytes:
    runner_descriptor = compiled_runner.matched_v3_ppo_gru_compiled_runner_descriptor()
    identity = {
        "schema_version": compiled_runner.PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "classification": "observed_compiled_runtime_unqualified_non_authorizing",
        "bindings": {
            "compiled_runner_descriptor_sha256": (
                compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
            ),
            "bridge_descriptor_sha256": compiled_runner.BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_sha256": (compiled_runner.BOUND_BRIDGE_IMPLEMENTATION_SHA256),
            "core_configuration_sha256": compiled_runner.BOUND_CORE_CONFIGURATION_SHA256,
            "core_implementation_sha256": compiled_runner.BOUND_CORE_IMPLEMENTATION_SHA256,
            "foragax_install_tree_sha256": compiled_runner.BOUND_FORAGAX_INSTALL_TREE_SHA256,
        },
        "runtime": {
            "jax_version": "0.11.0",
            "jaxlib_version": "0.11.0",
            "default_prng_impl": "threefry2x32",
            "threefry_partitionable": True,
            "jax_enable_x64": False,
            "backend": "synthetic-publication-test-only",
            "foragax_version": "0.55.0",
            "foragax_install_tree_sha256": compiled_runner.BOUND_FORAGAX_INSTALL_TREE_SHA256,
            "foragax_package_root": "/synthetic/not-opened",
            "runtime_qualified": False,
        },
        "kernel": {
            "chunk_steps": compiled_runner.PPO_GRU_COMPILED_CHUNK_STEPS,
            "constructed": True,
            "full_horizon_executed": False,
            "runtime_qualified": False,
        },
        "claims": runner_descriptor["claims"],
    }
    return _canonical(identity)


@dataclass(frozen=True, slots=True)
class _SyntheticOutcome:
    raw_reward_trace: bytes
    raw_cumulative_score: int
    interactions: int
    rollout_count: int
    optimizer_update_count: int
    total_agent_draw_count: int
    bridge_environment_key_use_count: int
    trace_chain_sha256: str
    runtime_identity_bytes: bytes
    receipt_bytes: bytes
    production_runtime: bool


@lru_cache(maxsize=1)
def _synthetic_contents() -> tuple[_SyntheticOutcome, bytes]:
    trace = bytes(protocol.MATCHED_V3_HORIZON)
    runtime_identity_bytes = _synthetic_runtime_identity_bytes()
    receipt = compiled_runner._receipt_bytes_from_fields(
        environment_seed=17,
        agent_seed=29,
        runtime_identity_bytes=runtime_identity_bytes,
        raw_reward_trace=trace,
        raw_cumulative_score=0,
        trace_chain_sha256="1" * 64,
    )
    outcome = _SyntheticOutcome(
        raw_reward_trace=trace,
        raw_cumulative_score=0,
        interactions=compiled_runner.MATCHED_V3_HORIZON,
        rollout_count=compiled_runner.PPO_GRU_COMPILED_CHUNK_COUNT,
        optimizer_update_count=compiled_runner.PPO_GRU_OPTIMIZER_UPDATES,
        total_agent_draw_count=compiled_runner.PPO_GRU_TOTAL_AGENT_DRAWS,
        bridge_environment_key_use_count=compiled_runner.PPO_GRU_BRIDGE_KEY_USES,
        trace_chain_sha256="1" * 64,
        runtime_identity_bytes=runtime_identity_bytes,
        receipt_bytes=receipt,
        production_runtime=True,
    )
    return outcome, receipt


@pytest.fixture
def compiled_reward_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> bundle.MatchedV3CompiledRewardBundle:
    outcome, receipt = _synthetic_contents()

    def validate(value: object) -> bytes:
        if value is not outcome:
            raise compiled_runner.ForagerMatchedV3PPOGRUCompiledRunnerError(
                "synthetic validator received another object"
            )
        return receipt

    monkeypatch.setattr(
        compiled_runner,
        "canonical_ppo_gru_compiled_result_receipt_bytes",
        validate,
    )
    return bundle.build_ppo_gru_compiled_reward_bundle(cast(Any, outcome))


def _publish(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
    *,
    name: str = "compiled-publication",
) -> publication.ContentVerifiedCompiledRewardPublication:
    tmp_path.chmod(0o700)
    return publication.publish_compiled_reward_bundle(
        compiled_reward_bundle,
        tmp_path / name,
    )


def _rewrite_publication_manifest(
    root: Path,
    mutate: Any,
) -> tuple[bytes, str]:
    path = root / publication.PUBLICATION_MANIFEST_FILENAME
    payload = cast(dict[str, Any], json.loads(path.read_bytes()))
    mutate(payload)
    payload.pop("publication_body_sha256", None)
    body_digest = hashlib.sha256(_canonical(payload)).hexdigest()
    payload["publication_body_sha256"] = body_digest
    raw = _canonical(payload)
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw, hashlib.sha256(raw).hexdigest()


def _partial_names(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if "compiled-partial" in path.name)


def test_descriptor_is_hardcoded_dependency_bound_and_nonauthorizing() -> None:
    raw = publication.canonical_compiled_reward_publication_descriptor_bytes()
    descriptor = publication.parse_compiled_reward_publication_descriptor(raw)

    assert len(raw) == 3_115
    assert publication.COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256 == (
        "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500"
    )
    assert hashlib.sha256(raw).hexdigest() == (
        publication.COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    )
    assert descriptor["dependency"] == {
        "source_path": (
            "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_bundle.py"
        ),
        "source_sha256": "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
        "descriptor_schema_version": (
            "alberta.forager_matched_v3.compiled_reward_bundle_descriptor.v1"
        ),
        "descriptor_sha256": ("cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"),
        "manifest_schema_version": (
            "alberta.forager_matched_v3.compiled_reward_bundle_manifest.v1"
        ),
    }
    assert set(descriptor["exact_files"].values()) == _EXPECTED_FILES
    assert descriptor["bounds"]["root_entries"] == 6
    assert descriptor["strict_scorer"]["canonical_npz_size_bytes"] == 499_980
    assert set(descriptor["claims"].values()) == {False}
    assert descriptor["loading"]["live_capability_reconstruction"] is False
    source = _ROOT / descriptor["dependency"]["source_path"]
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == (descriptor["dependency"]["source_sha256"])
    )
    descriptor["claims"]["authority_granted"] = True
    assert (
        publication.compiled_reward_publication_descriptor()["claims"]["authority_granted"] is False
    )
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.parse_compiled_reward_publication_descriptor(cast(Any, bytearray(raw)))


def test_publish_and_load_exact_six_files_without_reconstructing_live_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    def forbidden(_value: object) -> bytes:
        raise AssertionError("publication must not consult the live outcome capability")

    monkeypatch.setattr(
        compiled_runner,
        "canonical_ppo_gru_compiled_result_receipt_bytes",
        forbidden,
    )
    published = _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"

    assert published.output_root == root.resolve()
    assert {path.name for path in root.iterdir()} == _EXPECTED_FILES
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.iterdir())
    assert (root / publication.COMPILED_BUNDLE_MANIFEST_FILENAME).read_bytes() == (
        compiled_reward_bundle.manifest_bytes
    )
    assert (root / publication.RUNNER_RESULT_RECEIPT_FILENAME).read_bytes() == (
        compiled_reward_bundle.runner_receipt_bytes
    )
    assert (root / publication.RUNTIME_IDENTITY_FILENAME).read_bytes() == (
        compiled_reward_bundle.runtime_identity_bytes
    )
    assert (root / publication.REWARD_TRACE_FILENAME).read_bytes() == (
        compiled_reward_bundle.reward_artifact_bytes
    )
    assert (root / publication.SCORE_RECEIPT_FILENAME).read_bytes() == (
        compiled_reward_bundle.score_receipt_bytes
    )
    assert (
        hashlib.sha256((root / publication.PUBLICATION_MANIFEST_FILENAME).read_bytes()).hexdigest()
        == published.publication_file_sha256
    )
    assert set(published.manifest["claims"].values()) == {False}
    assert published.bundle == compiled_reward_bundle

    loaded = publication.load_compiled_reward_bundle_publication(
        root,
        expected_publication_file_sha256=published.publication_file_sha256,
    )
    assert loaded.bundle == compiled_reward_bundle
    assert loaded.publication_file_sha256 == published.publication_file_sha256
    assert isinstance(loaded.manifest, MappingProxyType)
    with pytest.raises(TypeError):
        loaded.manifest["candidate_id"] = "changed"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.candidate_id = "changed"  # type: ignore[misc]


def test_loader_requires_external_full_file_pin_and_exact_types(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    published = _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"
    raw = (root / publication.PUBLICATION_MANIFEST_FILENAME).read_bytes()

    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.load_compiled_reward_bundle_publication(
            root,
            expected_publication_file_sha256="1" * 64,
        )
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.load_compiled_reward_bundle_publication(
            cast(Any, str(root)),
            expected_publication_file_sha256=published.publication_file_sha256,
        )
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.parse_compiled_reward_publication_manifest(
            cast(Any, bytearray(raw)),
            expected_publication_file_sha256=published.publication_file_sha256,
        )
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.parse_compiled_reward_publication_manifest(
            raw,
            expected_publication_file_sha256=cast(Any, True),
        )


def test_rehashed_bool_int_alias_and_semantic_mutation_fail_closed(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"

    raw, digest = _rewrite_publication_manifest(
        root,
        lambda value: value["claims"].update({"authority_granted": 0}),
    )
    with pytest.raises(
        publication.ForagerMatchedV3CompiledRewardPublicationError,
        match="exact false",
    ):
        publication.parse_compiled_reward_publication_manifest(
            raw,
            expected_publication_file_sha256=digest,
        )

    raw, digest = _rewrite_publication_manifest(
        root,
        lambda value: value["writer_contract"].update({"staged_files_fsynced": 1}),
    )
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.parse_compiled_reward_publication_manifest(
            raw,
            expected_publication_file_sha256=digest,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "symlink", "hardlink", "fifo"))
def test_loader_rejects_inventory_links_and_special_files(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
    mutation: str,
) -> None:
    published = _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"
    runtime = root / publication.RUNTIME_IDENTITY_FILENAME
    runner = root / publication.RUNNER_RESULT_RECEIPT_FILENAME

    if mutation == "missing":
        runtime.unlink()
    elif mutation == "extra":
        extra = root / "extra.json"
        extra.write_bytes(b"{}")
        extra.chmod(0o600)
    elif mutation == "symlink":
        runtime.unlink()
        runtime.symlink_to(runner.name)
    elif mutation == "hardlink":
        runtime.unlink()
        os.link(runner, runtime)
    else:
        runtime.unlink()
        os.mkfifo(runtime, 0o600)

    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.load_compiled_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


def test_coherently_rehashed_outer_runtime_substitution_fails_bundle_replay(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"
    runtime_path = root / publication.RUNTIME_IDENTITY_FILENAME
    replacement = b"{}"
    runtime_path.write_bytes(replacement)
    runtime_path.chmod(0o600)

    def mutate(value: dict[str, Any]) -> None:
        record = value["files"]["runtime_identity"]
        record["sha256"] = hashlib.sha256(replacement).hexdigest()
        record["size_bytes"] = len(replacement)

    _raw, digest = _rewrite_publication_manifest(root, mutate)
    with pytest.raises(
        publication.ForagerMatchedV3CompiledRewardPublicationError,
        match="structural replay",
    ):
        publication.load_compiled_reward_bundle_publication(
            root,
            expected_publication_file_sha256=digest,
        )


def test_loader_rejects_oversized_or_noncanonical_payload_before_replay(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    published = _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"
    runtime = root / publication.RUNTIME_IDENTITY_FILENAME
    with runtime.open("wb") as stream:
        stream.truncate(publication._MAX_RUNTIME_IDENTITY_BYTES + 1)
    runtime.chmod(0o600)

    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.load_compiled_reward_bundle_publication(
            root,
            expected_publication_file_sha256=published.publication_file_sha256,
        )


@pytest.mark.parametrize("kind", ("file", "directory", "symlink"))
def test_publisher_never_overwrites_preexisting_destination(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
    kind: str,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "occupied"
    if kind == "file":
        destination.write_bytes(b"sentinel")
    elif kind == "directory":
        destination.mkdir(mode=0o700)
        (destination / "sentinel").write_bytes(b"sentinel")
    else:
        target = tmp_path / "target"
        target.write_bytes(b"sentinel")
        destination.symlink_to(target.name)
    before = destination.lstat()

    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    after = destination.lstat()
    assert (before.st_dev, before.st_ino, before.st_mode) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
    )
    assert _partial_names(tmp_path) == []


def test_concurrent_destination_wins_without_replacement_and_staging_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "raced"
    original = publication._rename_no_replace

    def race(
        parent: publication._OpenDirectory,
        source_name: str,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, 0o700, dir_fd=parent.descriptor)
        original(parent, source_name, destination_name)

    monkeypatch.setattr(publication, "_rename_no_replace", race)
    with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert _partial_names(tmp_path) == []


def test_prepublication_failure_cleans_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "failed"
    original = publication._write_exclusive_at
    calls = 0

    def fail_second(
        root: publication._OpenDirectory,
        name: str,
        raw: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise publication.ForagerMatchedV3CompiledRewardPublicationError(
                "injected prepublication failure"
            )
        original(root, name, raw)

    monkeypatch.setattr(publication, "_write_exclusive_at", fail_second)
    with pytest.raises(
        publication.ForagerMatchedV3CompiledRewardPublicationError,
        match="injected",
    ):
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    assert not destination.exists()
    assert _partial_names(tmp_path) == []


def test_final_replay_failure_after_visibility_is_explicitly_uncertain_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "uncertain"
    original = publication._load_from_open_root
    calls = 0

    def fail_second(
        root: publication._OpenDirectory,
        *,
        expected_publication_file_sha256: str,
    ) -> publication.ContentVerifiedCompiledRewardPublication:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise publication.ForagerMatchedV3CompiledRewardPublicationError(
                "injected final replay failure"
            )
        return original(
            root,
            expected_publication_file_sha256=expected_publication_file_sha256,
        )

    monkeypatch.setattr(publication, "_load_from_open_root", fail_second)
    with pytest.raises(publication.PublishedCompiledRewardPublicationUncertainError) as caught:
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    assert destination.is_dir()
    assert _partial_names(tmp_path) == []
    recovered = publication.load_compiled_reward_bundle_publication(
        destination,
        expected_publication_file_sha256=caught.value.publication_file_sha256,
    )
    assert recovered.bundle == compiled_reward_bundle


def test_parent_fsync_failure_after_move_is_uncertain_with_recovery_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "fsync-uncertain"

    def fail(_parent: publication._OpenDirectory) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(publication, "_sync_publication_parent", fail)
    with pytest.raises(publication.PublishedCompiledRewardPublicationUncertainError) as caught:
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    assert destination.is_dir()
    recovered = publication.load_compiled_reward_bundle_publication(
        destination,
        expected_publication_file_sha256=caught.value.publication_file_sha256,
    )
    assert recovered.bundle == compiled_reward_bundle


def test_rename_that_moves_then_reports_error_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "rename-uncertain"
    original = publication._rename_no_replace

    def move_then_fail(
        parent: publication._OpenDirectory,
        source_name: str,
        destination_name: str,
    ) -> None:
        original(parent, source_name, destination_name)
        raise OSError("injected post-rename report failure")

    monkeypatch.setattr(publication, "_rename_no_replace", move_then_fail)
    with pytest.raises(publication.PublishedCompiledRewardPublicationUncertainError) as caught:
        publication.publish_compiled_reward_bundle(compiled_reward_bundle, destination)

    assert destination.is_dir()
    recovered = publication.load_compiled_reward_bundle_publication(
        destination,
        expected_publication_file_sha256=caught.value.publication_file_sha256,
    )
    assert recovered.bundle == compiled_reward_bundle


def test_untrusted_parent_and_non_path_destination_fail_before_staging(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    tmp_path.chmod(0o770)
    try:
        with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
            publication.publish_compiled_reward_bundle(
                compiled_reward_bundle,
                tmp_path / "unsafe-parent",
            )
        with pytest.raises(publication.ForagerMatchedV3CompiledRewardPublicationError):
            publication.publish_compiled_reward_bundle(
                compiled_reward_bundle,
                cast(Any, str(tmp_path / "not-a-path")),
            )
    finally:
        tmp_path.chmod(0o700)
    assert _partial_names(tmp_path) == []


def test_dependency_source_reader_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    relative = Path("alberta_framework/benchmarks/forager_matched_v3_compiled_reward_bundle.py")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    expected = (_ROOT / relative).read_bytes()
    source.write_bytes(expected)
    assert (
        publication._source_sha256(str(source), str(relative))
        == hashlib.sha256(expected).hexdigest()
    )

    symlink = tmp_path / "symlink" / relative
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(source)
    with pytest.raises(RuntimeError):
        publication._source_sha256(str(symlink), str(relative))

    hardlink = tmp_path / "hardlink" / relative
    hardlink.parent.mkdir(parents=True)
    os.link(source, hardlink)
    with pytest.raises(RuntimeError):
        publication._source_sha256(str(source), str(relative))


def test_structural_publication_reloads_in_fresh_process(
    tmp_path: Path,
    compiled_reward_bundle: bundle.MatchedV3CompiledRewardBundle,
) -> None:
    published = _publish(tmp_path, compiled_reward_bundle)
    root = tmp_path / "compiled-publication"
    script = """
import sys
from pathlib import Path
from alberta_framework.benchmarks import forager_matched_v3_compiled_reward_publication as p
loaded = p.load_compiled_reward_bundle_publication(
    Path(sys.argv[1]), expected_publication_file_sha256=sys.argv[2]
)
assert loaded.candidate_id == 'adapted_ppo_gru'
assert set(loaded.manifest['claims'].values()) == {False}
print(loaded.publication_file_sha256)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), published.publication_file_sha256],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert published.publication_file_sha256 in result.stdout


def test_source_has_no_execution_v1_qualification_or_authority_shortcut() -> None:
    path = _ROOT / "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_publication.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "open_matched_v3_ppo_gru_compiled_runtime" not in called_attributes
    assert "run_matched_v3_ppo_gru_compiled" not in called_attributes
    assert "canonical_ppo_gru_compiled_result_receipt_bytes" not in called_attributes
    assert "subprocess" not in source
    assert "forager_matched_v3_adapter_reward_publication" not in source
    assert "forager_matched_v3_qualification_plan" not in source
    assert '"execution_authorized": True' not in source
    assert '"execution_ready": True' not in source
    assert '"runtime_qualified": True' not in source
    assert '"scientific_promotion_allowed": True' not in source
    assert '"universal_sota_claim_allowed": True' not in source
    assert "TO_BE_" not in source
