"""Adversarial tests for direct matched-v3 local atomic publication."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import subprocess
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ATOMIC_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "_forager_matched_v3_atomic_publication.py"
)
_PUBLISHER_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_reward_publication.py"
)
_BOOTSTRAP_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_execution_bootstrap.py"
)
_HANDOFF_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_result_handoff.py"
)
_BUNDLE_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_reward_bundle.py"
)
_BUNDLE_TEST_PATH = _ROOT / "tests" / "test_forager_matched_v3_local_reward_bundle.py"
_EXPECTED_FILENAMES = (
    "publication.json",
    "local-bundle-manifest.json",
    "bootstrap-receipt.json",
    "bootstrap-child-record.json",
    "local-runner-receipt.json",
    "reward-trace.npz",
    "score-receipt.json",
    "stdout.bin",
    "stderr.bin",
)


def _load_bundle_test_helpers() -> types.ModuleType:
    name = "_matched_v3_local_bundle_test_helpers_for_publication"
    existing = sys.modules.get(name)
    if type(existing) is types.ModuleType:
        return existing
    spec = importlib.util.spec_from_file_location(name, _BUNDLE_TEST_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load local bundle test helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Stack:
    helper: types.ModuleType
    modules: Any


@pytest.fixture(scope="module")
def stack() -> Iterator[_Stack]:
    helper = _load_bundle_test_helpers()
    generator = helper.modules.__wrapped__()
    modules = next(generator)
    yield _Stack(helper, modules)
    with pytest.raises(StopIteration):
        next(generator)


def _new_parent(tmp_path: Path, name: str = "publication-root") -> Path:
    parent = tmp_path / name
    parent.mkdir(mode=0o700)
    os.chmod(parent, 0o700)
    return parent


def _issue(stack: _Stack) -> object:
    return cast(object, stack.helper._issue(stack.modules))


def _publish(stack: _Stack, capability: object, parent: Path) -> Any:
    return stack.modules.publisher.publish_matched_v3_local_reward_capability(
        bundle_capability=capability,
        publication_parent=parent,
        expected_candidate_id=stack.helper._CANDIDATE_ID,
        expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
        expected_agent_seed=stack.helper._AGENT_SEED,
        expected_local_source_tree_sha256=stack.helper._SOURCE_TREE_SHA256,
        explicit_publication_opt_in=True,
    )


def _reload(stack: _Stack, metadata: Any, parent: Path) -> Any:
    return stack.modules.publisher.load_matched_v3_local_reward_publication(
        publication_parent=parent,
        expected_address=metadata.address,
        expected_file_records=metadata.files,
        expected_candidate_id=stack.helper._CANDIDATE_ID,
        expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
        expected_agent_seed=stack.helper._AGENT_SEED,
        expected_local_source_tree_sha256=stack.helper._SOURCE_TREE_SHA256,
    )


def _assert_no_public_bytes(value: object) -> None:
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        assert type(item) is not bytes
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                lowered = field.name.lower()
                assert "score" not in lowered
                assert "reward" not in lowered
                visit(getattr(item, field.name))
        elif type(item) is tuple:
            for child in cast(tuple[object, ...], item):
                visit(child)
        elif type(item) is dict:
            for key, child in cast(dict[object, object], item).items():
                visit(key)
                visit(child)

    visit(value)


def test_descriptor_and_public_api_are_narrow_and_non_authorizing(stack: _Stack) -> None:
    publisher = stack.modules.publisher
    raw = publisher.canonical_matched_v3_local_reward_publication_descriptor_bytes()
    assert publisher.LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256 == (
        "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f"
    )
    assert hashlib.sha256(raw).hexdigest() == (
        publisher.LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    )
    descriptor = publisher.matched_v3_local_reward_publication_descriptor()
    assert descriptor["publication"]["exact_filenames"] == list(_EXPECTED_FILENAMES)
    assert descriptor["publication"]["address"] == "full_sha256_of_publication_json"
    assert descriptor["publication"]["collision_retry"] is False
    assert descriptor["publication"]["uncertain_state_retry"] is False
    assert descriptor["publication_parent"] == {
        "exact_platform_path_required": True,
        "absolute_non_root_canonical_path_required": True,
        "must_exist_before_capability_claim": True,
        "symlinks_allowed": False,
        "effective_uid_ownership_required": True,
        "required_mode": "0700",
        "captured_atomic_open_and_close_primitives": True,
        "safe_preflight_before_bundle_capability_claim": True,
        "preflight_performs_writes": False,
        "atomic_commit_reopens_and_reverifies_parent": True,
        "preflight_eliminates_toctou": False,
    }
    assert any("TOCTOU" in item for item in descriptor["limitations"])
    assert descriptor["metadata_visibility"] == {
        "plaintext_score_returned": False,
        "published_file_bytes_returned": False,
        "exact_file_digests_and_sizes_returned": True,
        "information_theoretic_score_opacity_claimed": False,
        "qualification_controller_may_branch_on_content_digests_or_sizes": False,
    }
    assert descriptor["metadata_receipt"] == {
        "canonical_score_field_free_bytes_exported": True,
        "strict_parser_exported": True,
        "caller_carried_full_file_sha256_required": True,
        "body_digest_is_not_substituted_for_full_file_digest": True,
    }
    assert descriptor["claims"] and all(value is False for value in descriptor["claims"].values())
    signature = inspect.signature(publisher.publish_matched_v3_local_reward_capability)
    assert tuple(signature.parameters) == (
        "bundle_capability",
        "publication_parent",
        "expected_candidate_id",
        "expected_environment_seed",
        "expected_agent_seed",
        "expected_local_source_tree_sha256",
        "explicit_publication_opt_in",
    )
    assert not {"bundle", "payload", "bytes", "callback", "sink"} & set(signature.parameters)
    assert "_preflight_publication_parent" not in publisher.__all__


def test_direct_publish_uses_full_manifest_digest_and_returns_metadata_only(
    stack: _Stack,
    tmp_path: Path,
) -> None:
    parent = _new_parent(tmp_path)
    metadata = _publish(stack, _issue(stack), parent)
    assert type(metadata) is stack.modules.publisher.MatchedV3LocalRewardPublicationMetadata
    assert metadata.operation == "published"
    assert metadata.publication_root == parent / metadata.address
    assert tuple(item.name for item in metadata.files) == _EXPECTED_FILENAMES
    publication_raw = (metadata.publication_root / "publication.json").read_bytes()
    assert metadata.address == hashlib.sha256(publication_raw).hexdigest()
    assert metadata.address == metadata.publication_manifest_sha256
    assert metadata.file_count == 9
    assert metadata.total_size_bytes == sum(item.size_bytes for item in metadata.files)
    assert stat.S_IMODE(metadata.publication_root.stat().st_mode) == 0o700
    for record in metadata.files:
        path = metadata.publication_root / record.name
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_size == record.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256
    _assert_no_public_bytes(metadata)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.address = "f" * 64


def test_inspection_then_publication_fails_closed(stack: _Stack, tmp_path: Path) -> None:
    capability = _issue(stack)
    stack.helper._consume(stack.modules, capability)
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _publish(stack, capability, _new_parent(tmp_path))


def test_publication_then_inspection_fails_closed(stack: _Stack, tmp_path: Path) -> None:
    capability = _issue(stack)
    _publish(stack, capability, _new_parent(tmp_path))
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        stack.helper._consume(stack.modules, capability)


def test_wrong_capability_and_missing_opt_in_fail_closed(stack: _Stack, tmp_path: Path) -> None:
    parent = _new_parent(tmp_path)
    publisher = stack.modules.publisher
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _publish(stack, object(), parent)
    capability = _issue(stack)
    with pytest.raises(publisher.ForagerMatchedV3LocalRewardPublicationError):
        publisher.publish_matched_v3_local_reward_capability(
            bundle_capability=capability,
            publication_parent=parent,
            expected_candidate_id=stack.helper._CANDIDATE_ID,
            expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
            expected_agent_seed=stack.helper._AGENT_SEED,
            expected_local_source_tree_sha256=stack.helper._SOURCE_TREE_SHA256,
            explicit_publication_opt_in=False,
        )
    assert type(stack.helper._consume(stack.modules, capability)) is (
        stack.modules.bundle.MatchedV3LocalRewardBundle
    )


def test_public_parent_mode_failure_preserves_same_capability(
    stack: _Stack,
    tmp_path: Path,
) -> None:
    parent = _new_parent(tmp_path)
    parent.chmod(0o755)
    capability = _issue(stack)
    with pytest.raises(stack.modules.atomic.ForagerMatchedV3AtomicPublicationError):
        _publish(stack, capability, parent)
    assert stack.modules.bundle._BUNDLE_CAPABILITIES[capability].status == "live"
    parent.chmod(0o700)
    metadata = _publish(stack, capability, parent)
    assert type(metadata) is stack.modules.publisher.MatchedV3LocalRewardPublicationMetadata


@pytest.mark.parametrize("primitive_name", ("_open_parent", "_close_no_raise"))
def test_atomic_parent_preflight_primitive_drift_precedes_capability_claim(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primitive_name: str,
) -> None:
    atomic = stack.modules.atomic
    capability = _issue(stack)
    original = getattr(atomic, primitive_name)

    def replacement(*args: object, **kwargs: object) -> object:
        return cast(Any, original)(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(atomic, primitive_name, replacement)
        with pytest.raises(
            stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError
        ):
            _publish(stack, capability, _new_parent(tmp_path, f"drift-{primitive_name}"))
        assert stack.modules.bundle._BUNDLE_CAPABILITIES[capability].status == "live"
    metadata = _publish(stack, capability, _new_parent(tmp_path, f"clean-{primitive_name}"))
    assert type(metadata) is stack.modules.publisher.MatchedV3LocalRewardPublicationMetadata


def test_publisher_source_binding_drift_precedes_capability_claim(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = stack.modules.publisher
    capability = _issue(stack)
    with monkeypatch.context() as patch:
        patch.setattr(publisher, "_PUBLICATION_SOURCE_SHA256_INPUT", "a" * 64)
        with pytest.raises(publisher.ForagerMatchedV3LocalRewardPublicationError):
            _publish(stack, capability, _new_parent(tmp_path, "source-drift"))
        assert stack.modules.bundle._BUNDLE_CAPABILITIES[capability].status == "live"
    metadata = _publish(stack, capability, _new_parent(tmp_path, "source-clean"))
    assert type(metadata) is publisher.MatchedV3LocalRewardPublicationMetadata


def test_captured_parent_preflight_function_drift_precedes_capability_claim(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = stack.modules.publisher
    capability = _issue(stack)
    original = publisher._preflight_publication_parent

    def replacement(*, publication_parent: Path) -> Path:
        return cast(Path, original(publication_parent=publication_parent))

    with monkeypatch.context() as patch:
        patch.setattr(publisher, "_preflight_publication_parent", replacement)
        with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
            stack.helper._publish_direct(
                stack.modules,
                capability,
                _new_parent(tmp_path, "preflight-drift"),
            )
        assert stack.modules.bundle._BUNDLE_CAPABILITIES[capability].status == "live"
    metadata = _publish(stack, capability, _new_parent(tmp_path, "preflight-clean"))
    assert type(metadata) is publisher.MatchedV3LocalRewardPublicationMetadata


def test_pid_drift_consumes_and_rejects_capability(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _issue(stack)
    original = os.getpid()
    monkeypatch.setattr(stack.modules.bundle.os, "getpid", lambda: original + 1)
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError, match="PID"):
        _publish(stack, capability, _new_parent(tmp_path))
    monkeypatch.undo()
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        stack.helper._consume(stack.modules, capability)


@pytest.mark.parametrize("role", tuple(range(9)))
def test_each_sealed_file_tamper_fails_before_publication(
    stack: _Stack,
    tmp_path: Path,
    role: int,
) -> None:
    capability = _issue(stack)
    state = stack.modules.bundle._BUNDLE_CAPABILITIES[capability]
    payloads = list(state.sealed_payload.role_payloads)
    item_role, path, raw = payloads[role]
    payloads[role] = (item_role, path, raw + b"tampered")
    object.__setattr__(state.sealed_payload, "role_payloads", tuple(payloads))
    parent = _new_parent(tmp_path)
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _publish(stack, capability, parent)
    assert tuple(parent.iterdir()) == ()


def test_exact_collision_propagates_without_retry(stack: _Stack, tmp_path: Path) -> None:
    capability = _issue(stack)
    state = stack.modules.bundle._BUNDLE_CAPABILITIES[capability]
    address = hashlib.sha256(state.sealed_payload.role_payloads[0][2]).hexdigest()
    parent = _new_parent(tmp_path)
    (parent / address).mkdir(mode=0o700)
    with pytest.raises(
        stack.modules.atomic.ForagerMatchedV3AtomicPublicationCollisionError
    ):
        _publish(stack, capability, parent)
    assert tuple(path.name for path in parent.iterdir()) == (address,)
    with pytest.raises(stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        stack.helper._consume(stack.modules, capability)


def test_atomic_uncertain_error_propagates_once(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atomic = stack.modules.atomic
    publisher = stack.modules.publisher
    calls = 0
    expected = atomic.ForagerMatchedV3AtomicPublicationUncertainError(
        tmp_path / "unknown",
        "a" * 64,
        "synthetic post-commit verification failure",
        committed=None,
    )

    def raising_publish(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise expected

    raising_publish.__module__ = publisher.PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
    monkeypatch.setattr(atomic, "publish_exact_flat_publication", raising_publish)
    monkeypatch.setattr(publisher, "_ATOMIC_PUBLISH_AT_LOAD", raising_publish)
    surface = tuple(
        sorted(
            (
                (name, value, value.__code__, value.__defaults__, value.__kwdefaults__)
                for name, value in vars(atomic).items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__
                == publisher.PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    monkeypatch.setattr(publisher, "_ATOMIC_FUNCTION_SURFACE_AT_LOAD", surface)
    capability = _issue(stack)
    with pytest.raises(type(expected)) as raised:
        _publish(stack, capability, _new_parent(tmp_path))
    assert raised.value is expected
    assert calls == 1


def test_atomic_commit_reopens_parent_once_after_successful_preflight(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atomic = stack.modules.atomic
    publisher = stack.modules.publisher
    parent = _new_parent(tmp_path)
    capability = _issue(stack)
    original_publish = atomic.publish_exact_flat_publication
    calls = 0

    def change_mode_then_publish(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        publication_parent = cast(Path, args[0])
        publication_parent.chmod(0o755)
        return cast(Any, original_publish)(*args, **kwargs)

    change_mode_then_publish.__module__ = (
        publisher.PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
    )
    monkeypatch.setattr(
        atomic,
        "publish_exact_flat_publication",
        change_mode_then_publish,
    )
    monkeypatch.setattr(publisher, "_ATOMIC_PUBLISH_AT_LOAD", change_mode_then_publish)
    surface = tuple(
        sorted(
            (
                (name, value, value.__code__, value.__defaults__, value.__kwdefaults__)
                for name, value in vars(atomic).items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__
                == publisher.PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    monkeypatch.setattr(publisher, "_ATOMIC_FUNCTION_SURFACE_AT_LOAD", surface)
    with pytest.raises(atomic.ForagerMatchedV3AtomicPublicationError):
        _publish(stack, capability, parent)
    assert calls == 1
    assert tuple(parent.iterdir()) == ()
    assert stack.modules.bundle._BUNDLE_CAPABILITIES[capability].status == "consumed"


def test_reload_requires_caller_records_and_replays_structure(
    stack: _Stack,
    tmp_path: Path,
) -> None:
    parent = _new_parent(tmp_path)
    published = _publish(stack, _issue(stack), parent)
    reloaded = _reload(stack, published, parent)
    assert reloaded.operation == "reloaded"
    assert reloaded.address == published.address
    assert reloaded.files == published.files
    assert reloaded.metadata_body_sha256 != published.metadata_body_sha256
    _assert_no_public_bytes(reloaded)
    with pytest.raises(stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError):
        stack.modules.publisher.load_matched_v3_local_reward_publication(
            publication_parent=parent,
            expected_address=published.address,
            expected_file_records=tuple(reversed(published.files)),
            expected_candidate_id=stack.helper._CANDIDATE_ID,
            expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
            expected_agent_seed=stack.helper._AGENT_SEED,
            expected_local_source_tree_sha256=stack.helper._SOURCE_TREE_SHA256,
        )


def test_canonical_metadata_receipts_use_full_file_digests(
    stack: _Stack,
    tmp_path: Path,
) -> None:
    parent = _new_parent(tmp_path)
    published = _publish(stack, _issue(stack), parent)
    reloaded = _reload(stack, published, parent)
    publisher = stack.modules.publisher
    for metadata in (published, reloaded):
        raw = publisher.canonical_matched_v3_local_reward_publication_metadata_bytes(metadata)
        full_file_sha256 = hashlib.sha256(raw).hexdigest()
        assert full_file_sha256 != metadata.metadata_body_sha256
        parsed = publisher.parse_matched_v3_local_reward_publication_metadata(
            raw,
            expected_full_file_sha256=full_file_sha256,
        )
        assert parsed == metadata
        _assert_no_public_bytes(parsed)
        with pytest.raises(publisher.ForagerMatchedV3LocalRewardPublicationError):
            publisher.parse_matched_v3_local_reward_publication_metadata(
                raw,
                expected_full_file_sha256=metadata.metadata_body_sha256,
            )
        with pytest.raises(publisher.ForagerMatchedV3LocalRewardPublicationError):
            publisher.parse_matched_v3_local_reward_publication_metadata(
                raw + b"drift",
                expected_full_file_sha256=hashlib.sha256(raw + b"drift").hexdigest(),
            )


@pytest.mark.parametrize("file_index", tuple(range(9)))
def test_reload_rejects_tamper_of_every_file(
    stack: _Stack,
    tmp_path: Path,
    file_index: int,
) -> None:
    parent = _new_parent(tmp_path)
    metadata = _publish(stack, _issue(stack), parent)
    target = metadata.publication_root / metadata.files[file_index].name
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(stack.modules.atomic.ForagerMatchedV3AtomicPublicationError):
        _reload(stack, metadata, parent)


def test_reload_rejects_wrong_cell_and_source_tree(stack: _Stack, tmp_path: Path) -> None:
    parent = _new_parent(tmp_path)
    metadata = _publish(stack, _issue(stack), parent)
    publisher = stack.modules.publisher
    for candidate, tree in (
        ("causal_e025_q075", stack.helper._SOURCE_TREE_SHA256),
        (stack.helper._CANDIDATE_ID, "8" * 64),
    ):
        with pytest.raises(publisher.ForagerMatchedV3LocalRewardPublicationError):
            publisher.load_matched_v3_local_reward_publication(
                publication_parent=parent,
                expected_address=metadata.address,
                expected_file_records=metadata.files,
                expected_candidate_id=candidate,
                expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
                expected_agent_seed=stack.helper._AGENT_SEED,
                expected_local_source_tree_sha256=tree,
            )


@pytest.mark.parametrize("file_index", tuple(range(2, 9)))
def test_reload_rejects_tamper_even_with_updated_caller_record(
    stack: _Stack,
    tmp_path: Path,
    file_index: int,
) -> None:
    """Caller records cannot replace either manifest's committed file inventory."""

    parent = _new_parent(tmp_path)
    metadata = _publish(stack, _issue(stack), parent)
    target = metadata.publication_root / metadata.files[file_index].name
    tampered = target.read_bytes() + b"tampered"
    target.write_bytes(tampered)
    records = list(metadata.files)
    records[file_index] = stack.modules.publisher.MatchedV3LocalRewardPublicationFile(
        name=records[file_index].name,
        size_bytes=len(tampered),
        sha256=hashlib.sha256(tampered).hexdigest(),
    )
    with pytest.raises(stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError):
        stack.modules.publisher.load_matched_v3_local_reward_publication(
            publication_parent=parent,
            expected_address=metadata.address,
            expected_file_records=tuple(records),
            expected_candidate_id=stack.helper._CANDIDATE_ID,
            expected_environment_seed=stack.helper._ENVIRONMENT_SEED,
            expected_agent_seed=stack.helper._AGENT_SEED,
            expected_local_source_tree_sha256=stack.helper._SOURCE_TREE_SHA256,
        )


def test_bundle_descriptor_bytes_are_replayed_before_capability_consumption(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _issue(stack)
    original = stack.modules.bundle._DESCRIPTOR_BYTES
    monkeypatch.setattr(stack.modules.bundle, "_DESCRIPTOR_BYTES", original + b"drift")
    with pytest.raises(
        stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError,
        match="descriptor replay failed",
    ):
        _publish(stack, capability, _new_parent(tmp_path))
    monkeypatch.undo()
    assert type(stack.helper._consume(stack.modules, capability)) is (
        stack.modules.bundle.MatchedV3LocalRewardBundle
    )


@pytest.mark.parametrize("dependency", ("publisher", "atomic"))
def test_publisher_and_atomic_descriptor_bytes_are_replayed_before_consumption(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    capability = _issue(stack)
    module = getattr(stack.modules, dependency)
    original = module._DESCRIPTOR_BYTES
    monkeypatch.setattr(module, "_DESCRIPTOR_BYTES", original + b"drift")
    with pytest.raises(stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError):
        _publish(stack, capability, _new_parent(tmp_path))
    monkeypatch.undo()
    assert type(stack.helper._consume(stack.modules, capability)) is (
        stack.modules.bundle.MatchedV3LocalRewardBundle
    )


def test_publisher_sink_code_drift_fails_closed(
    stack: _Stack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = stack.modules.publisher._publish_consumed_local_reward_payload

    def forged_sink(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(sink, "__code__", forged_sink.__code__)
    with pytest.raises(
        (
            stack.modules.publisher.ForagerMatchedV3LocalRewardPublicationError,
            stack.modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        )
    ):
        _publish(stack, _issue(stack), _new_parent(tmp_path))


def test_fresh_isolated_reload_handshake_without_guard_replacement(
    stack: _Stack,
    tmp_path: Path,
) -> None:
    parent = _new_parent(tmp_path)
    metadata = _publish(stack, _issue(stack), parent)
    records_json = json.dumps(
        [[item.name, item.size_bytes, item.sha256] for item in metadata.files],
        separators=(",", ":"),
    )
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
    return module, hashlib.sha256(source).hexdigest()

atomic_path, publisher_path, bootstrap_path, handoff_path, bundle_path = (
    Path(value).resolve() for value in sys.argv[1:6]
)
parent = Path(sys.argv[6])
address = sys.argv[7]
records_data = json.loads(sys.argv[8])
atomic_name = '_alberta_forager_matched_v3_atomic_publication_isolated_v1'
publisher_name = '_alberta_forager_matched_v3_local_reward_publication_isolated_v1'
bootstrap_name = '_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1'
handoff_name = '_alberta_forager_matched_v3_local_result_handoff_isolated_v1'
bundle_name = '_alberta_forager_matched_v3_local_reward_bundle_isolated_v1'
atomic, atomic_sha = load(atomic_path, atomic_name, {})
publisher_source = hashlib.sha256(publisher_path.read_bytes()).hexdigest()
publisher, observed_publisher = load(
    publisher_path,
    publisher_name,
    {'_MATCHED_V3_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256': publisher_source},
)
assert observed_publisher == publisher_source
bootstrap_source = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
bootstrap, _ = load(
    bootstrap_path,
    bootstrap_name,
    {'_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256': bootstrap_source},
)
handoff_source = hashlib.sha256(handoff_path.read_bytes()).hexdigest()
handoff, _ = load(
    handoff_path,
    handoff_name,
    {
        '_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256': handoff_source,
        '_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256': bootstrap_source,
    },
)
bundle_source = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
bundle, _ = load(
    bundle_path,
    bundle_name,
    {
        '_MATCHED_V3_LOCAL_REWARD_BUNDLE_SOURCE_SHA256': bundle_source,
        '_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256': handoff_source,
    },
)
assert publisher._ISOLATED_PUBLICATION_BOUNDARY is True
assert bundle._ISOLATED_BUNDLE_BOUNDARY is True
assert bundle._PUBLISHER_SINK_AT_LOAD is publisher._publish_consumed_local_reward_payload
records = tuple(
    publisher.MatchedV3LocalRewardPublicationFile(
        name=item[0], size_bytes=item[1], sha256=item[2]
    )
    for item in records_data
)
result = publisher.load_matched_v3_local_reward_publication(
    publication_parent=parent,
    expected_address=address,
    expected_file_records=records,
    expected_candidate_id='causal_e025_q050',
    expected_environment_seed=17,
    expected_agent_seed=23,
    expected_local_source_tree_sha256='7' * 64,
)
assert result.operation == 'reloaded'
assert result.address == address
assert type(result.files) is tuple
for item in result.files:
    assert type(item.name) is str
    assert type(item.size_bytes) is int
    assert type(item.sha256) is str
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
            str(_BOOTSTRAP_PATH),
            str(_HANDOFF_PATH),
            str(_BUNDLE_PATH),
            str(parent),
            metadata.address,
            records_json,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
