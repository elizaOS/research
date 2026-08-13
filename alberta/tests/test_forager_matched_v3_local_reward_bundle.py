"""Adversarial tests for the score-bearing matched-v3 inspection bundle."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
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
_ATOMIC_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "_forager_matched_v3_atomic_publication.py"
)
_PUBLISHER_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_reward_publication.py"
)
_HANDOFF_TEST_PATH = _ROOT / "tests" / "test_forager_matched_v3_local_result_handoff.py"
_BOOTSTRAP_NAME = "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
_HANDOFF_NAME = "_alberta_forager_matched_v3_local_result_handoff_isolated_v1"
_BUNDLE_NAME = "_alberta_forager_matched_v3_local_reward_bundle_isolated_v1"
_ATOMIC_NAME = "_alberta_forager_matched_v3_atomic_publication_isolated_v1"
_PUBLISHER_NAME = "_alberta_forager_matched_v3_local_reward_publication_isolated_v1"
_CANDIDATE_ID = "causal_e025_q050"
_ENVIRONMENT_SEED = 17
_AGENT_SEED = 23
_SOURCE_FULL_SHA256 = "6" * 64
_SOURCE_TREE_SHA256 = "7" * 64
_TRACE = b"\x00" * 499_712
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
_RECEIPT_METHOD_NAMES = (
    "__delattr__",
    "__eq__",
    "__getstate__",
    "__hash__",
    "__init__",
    "__post_init__",
    "__repr__",
    "__setattr__",
    "__setstate__",
    "canonical_body",
    "canonical_json",
    "to_body",
    "to_payload",
)


def _load_handoff_test_helpers() -> types.ModuleType:
    name = "_matched_v3_local_handoff_test_helpers"
    existing = sys.modules.get(name)
    if type(existing) is types.ModuleType:
        return existing
    spec = importlib.util.spec_from_file_location(name, _HANDOFF_TEST_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the local handoff test helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Modules:
    atomic: types.ModuleType
    publisher: types.ModuleType
    bootstrap: types.ModuleType
    handoff: types.ModuleType
    bundle: types.ModuleType
    helpers: types.ModuleType


@pytest.fixture(scope="module")
def modules() -> Iterator[_Modules]:
    helpers = _load_handoff_test_helpers()
    atomic = helpers._direct_module(
        source_path=_ATOMIC_PATH,
        module_name=_ATOMIC_NAME,
        injections={},
    )
    publisher_source = _PUBLISHER_PATH.read_bytes()
    publisher = helpers._direct_module(
        source_path=_PUBLISHER_PATH,
        module_name=_PUBLISHER_NAME,
        injections={
            "_MATCHED_V3_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256": (
                hashlib.sha256(publisher_source).hexdigest()
            ),
        },
    )
    setattr(publisher, "_ISOLATED_PUBLICATION_BOUNDARY", True)
    setattr(publisher, "_live_forbidden_modules", lambda: ())
    setattr(
        publisher,
        "_SELF_FUNCTION_SURFACE_AT_READY",
        publisher._current_self_function_surface(),
    )
    bootstrap_source = _BOOTSTRAP_PATH.read_bytes()
    bootstrap_sha256 = hashlib.sha256(bootstrap_source).hexdigest()
    bootstrap = helpers._direct_module(
        source_path=_BOOTSTRAP_PATH,
        module_name=_BOOTSTRAP_NAME,
        injections={
            "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256": bootstrap_sha256,
        },
    )
    setattr(bootstrap, "_ISOLATED_PARENT_BOUNDARY", True)
    setattr(bootstrap, "_live_forbidden_modules", lambda: ())

    handoff_source = _HANDOFF_PATH.read_bytes()
    handoff_sha256 = hashlib.sha256(handoff_source).hexdigest()
    handoff = helpers._direct_module(
        source_path=_HANDOFF_PATH,
        module_name=_HANDOFF_NAME,
        injections={
            "_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256": handoff_sha256,
            "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256": bootstrap_sha256,
        },
    )
    setattr(handoff, "_ISOLATED_HANDOFF_BOUNDARY", True)
    setattr(handoff, "_live_forbidden_modules", lambda: ())

    bundle_source = _BUNDLE_PATH.read_bytes()
    bundle = helpers._direct_module(
        source_path=_BUNDLE_PATH,
        module_name=_BUNDLE_NAME,
        injections={
            "_MATCHED_V3_LOCAL_REWARD_BUNDLE_SOURCE_SHA256": (
                hashlib.sha256(bundle_source).hexdigest()
            ),
            "_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256": handoff_sha256,
        },
    )
    # This in-process structural fixture runs under pytest after JAX has been
    # loaded elsewhere.  Only the ambient observation is bypassed here; the
    # real production guards run without replacement in the subprocess test.
    setattr(bundle, "_ISOLATED_BUNDLE_BOUNDARY", True)
    setattr(bundle, "_live_forbidden_modules", lambda: ())
    setattr(
        bundle,
        "_SELF_FUNCTION_SURFACE_AT_READY",
        bundle._current_self_function_surface(),
    )
    yield _Modules(atomic, publisher, bootstrap, handoff, bundle, helpers)
    sys.modules.pop(_BUNDLE_NAME, None)
    sys.modules.pop(_HANDOFF_NAME, None)
    sys.modules.pop(_BOOTSTRAP_NAME, None)
    sys.modules.pop(_PUBLISHER_NAME, None)
    sys.modules.pop(_ATOMIC_NAME, None)


def _handoff_capability(
    modules: _Modules,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    trace: bytes = _TRACE,
) -> object:
    handoff_modules = modules.helpers._Modules(
        modules.bootstrap,
        modules.handoff,
        hashlib.sha256(_BOOTSTRAP_PATH.read_bytes()).hexdigest(),
        hashlib.sha256(_HANDOFF_PATH.read_bytes()).hexdigest(),
    )
    fixture = modules.helpers._authentic_outcome(
        handoff_modules,
        stdout=stdout,
        stderr=stderr,
        reward_trace=trace,
    )
    return modules.handoff.issue_matched_v3_local_result_handoff(
        bootstrap_outcome_capability=fixture.outcome,
        explicit_handoff_opt_in=True,
    )


def _issue(
    modules: _Modules,
    *,
    handoff_capability: object | None = None,
    candidate_id: str = _CANDIDATE_ID,
    environment_seed: int = _ENVIRONMENT_SEED,
    agent_seed: int = _AGENT_SEED,
    source_full_sha256: str = _SOURCE_FULL_SHA256,
    source_tree_sha256: str = _SOURCE_TREE_SHA256,
    explicit_opt_in: bool = True,
) -> object:
    capability = handoff_capability
    if capability is None:
        capability = _handoff_capability(modules)
    return modules.bundle.issue_matched_v3_local_reward_bundle(
        handoff_capability=capability,
        expected_candidate_id=candidate_id,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
        expected_local_source_full_sha256=source_full_sha256,
        expected_local_source_tree_sha256=source_tree_sha256,
        explicit_bundle_opt_in=explicit_opt_in,
    )


def _consume(modules: _Modules, capability: object, *, opt_in: bool = True) -> Any:
    return modules.bundle.consume_matched_v3_local_reward_bundle(
        bundle_capability=capability,
        explicit_content_access_opt_in=opt_in,
    )


def _publish_direct(
    modules: _Modules,
    capability: object,
    publication_parent: object,
) -> Any:
    return modules.bundle._consume_matched_v3_local_reward_capability_to_captured_sink(
        bundle_capability=capability,
        publication_parent=cast(Any, publication_parent),
        expected_candidate_id=_CANDIDATE_ID,
        expected_environment_seed=_ENVIRONMENT_SEED,
        expected_agent_seed=_AGENT_SEED,
        expected_local_source_tree_sha256=_SOURCE_TREE_SHA256,
        explicit_publication_opt_in=True,
    )


def test_descriptor_is_score_bearing_permanently_nonqualifying_inspection(
    modules: _Modules,
) -> None:
    bundle = modules.bundle
    raw = bundle.canonical_matched_v3_local_reward_bundle_descriptor_bytes()
    assert bundle.LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256 == (
        "f1fb7d28f0508c38b0d53173707ea5cb006b669793d3401091a942874ee3b878"
    )
    assert bundle.PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256 == (
        "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f"
    )
    assert bundle.PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256 == (
        "48640a7e352383eac58fed24c8c36c77fcf3bbed8baf78ce663394d1f7e90200"
    )
    assert hashlib.sha256(raw).hexdigest() == bundle.LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256
    parsed = bundle.parse_matched_v3_local_reward_bundle_descriptor(raw)
    assert parsed == bundle.matched_v3_local_reward_bundle_descriptor()
    assert parsed["inventory"]["exact_filenames"] == list(_EXPECTED_FILENAMES)
    assert parsed["inspection"]["public_bundle_is_score_bearing"] is True
    assert parsed["inspection"]["public_bundle_is_permanently_nonqualifying"] is True
    assert parsed["inspection"]["module_decodes_score_fields"] is False
    assert parsed["inspection"]["module_decodes_reward_npz"] is False
    assert parsed["qualification_publisher"]["must_consume_live_capability_directly"] is True
    assert parsed["qualification_publisher"]["may_accept_public_bundle_object"] is False
    assert parsed["qualification_publisher"]["exact_publisher_sink_captured_at_bundle_load"]
    assert parsed["qualification_publisher"][
        "exact_parent_preflight_captured_at_bundle_load"
    ]
    assert parsed["qualification_publisher"][
        "safe_parent_preflight_precedes_capability_claim"
    ]
    assert parsed["qualification_publisher"][
        "atomic_commit_reopens_and_reverifies_parent"
    ]
    assert (
        parsed["qualification_publisher"]["parent_preflight_eliminates_toctou"]
        is False
    )
    assert any("TOCTOU" in item for item in parsed["limitations"])
    assert parsed["capability"]["registry_retains_private_sealed_payload_not_public_bundle"]
    assert parsed["filesystem"]["writes"] is False
    assert parsed["claims"] and all(value is False for value in parsed["claims"].values())
    assert bundle._PUBLISHER_PARENT_PREFLIGHT_AT_LOAD is (
        modules.publisher._preflight_publication_parent
    )


def test_true_isolated_production_guards_and_scorer_load_without_overrides() -> None:
    script = r'''\
import hashlib
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

atomic_path = Path(sys.argv[1]).resolve()
publisher_path = Path(sys.argv[2]).resolve()
bootstrap_path = Path(sys.argv[3]).resolve()
handoff_path = Path(sys.argv[4]).resolve()
bundle_path = Path(sys.argv[5]).resolve()
atomic_name = '_alberta_forager_matched_v3_atomic_publication_isolated_v1'
publisher_name = '_alberta_forager_matched_v3_local_reward_publication_isolated_v1'
bootstrap_name = '_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1'
handoff_name = '_alberta_forager_matched_v3_local_result_handoff_isolated_v1'
bundle_name = '_alberta_forager_matched_v3_local_reward_bundle_isolated_v1'
atomic, observed_atomic = load(atomic_path, atomic_name, {})
assert observed_atomic == '8e7ccf6333c7cd8d932a190bc69aed969be93fdad450df7d5b6f8cbb785fc587'
publisher_source = hashlib.sha256(publisher_path.read_bytes()).hexdigest()
publisher, observed_publisher = load(
    publisher_path,
    publisher_name,
    {'_MATCHED_V3_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256': publisher_source},
)
assert publisher_source == observed_publisher
bootstrap_source = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
bootstrap, observed_bootstrap = load(
    bootstrap_path,
    bootstrap_name,
    {'_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256': bootstrap_source},
)
assert bootstrap_source == observed_bootstrap
handoff_source = hashlib.sha256(handoff_path.read_bytes()).hexdigest()
handoff, observed_handoff = load(
    handoff_path,
    handoff_name,
    {
        '_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256': handoff_source,
        '_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256': bootstrap_source,
    },
)
assert handoff_source == observed_handoff
bundle_source = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
module, observed_bundle = load(
    bundle_path,
    bundle_name,
    {
        '_MATCHED_V3_LOCAL_REWARD_BUNDLE_SOURCE_SHA256': bundle_source,
        '_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256': handoff_source,
    },
)
assert bundle_source == observed_bundle
assert bootstrap._ISOLATED_PARENT_BOUNDARY is True
assert handoff._ISOLATED_HANDOFF_BOUNDARY is True
assert module._ISOLATED_BUNDLE_BOUNDARY is True
assert publisher._ISOLATED_PUBLICATION_BOUNDARY is True
assert bootstrap._require_parent_boundary(require_current_source=True) == bootstrap_source
assert handoff._require_handoff_boundary(require_current_source=True) == handoff_source
assert module._require_bundle_boundary(reject_runtime_modules=True) == bundle_source
module._require_exact_handoff_module()
module._require_exact_publisher_module()
before = tuple(
    key for key in sys.modules
    if key == 'alberta_framework' or key.startswith('alberta_framework.')
)
assert before == ()
api = module._require_scorer_api()
assert type(api.module) is types.ModuleType
assert sys.modules.get(api.module.__name__) is not api.module
protocol = api.module.protocol
assert type(protocol) is types.ModuleType
assert sys.modules.get(protocol.__name__) is not protocol
after = tuple(
    key for key in sys.modules
    if key == 'alberta_framework' or key.startswith('alberta_framework.')
)
assert after == ()
assert module._require_bundle_boundary(reject_runtime_modules=True) == bundle_source
module._require_exact_handoff_module()
for forbidden in ('jax', 'jaxlib', 'numpy', 'scipy', 'foragax', 'chex'):
    assert forbidden not in sys.modules
live_guard = module._live_forbidden_modules
def forged_live_guard():
    return ()
live_guard.__code__ = forged_live_guard.__code__
try:
    module.matched_v3_local_reward_bundle_descriptor()
except module.ForagerMatchedV3LocalRewardBundleError:
    pass
else:
    raise AssertionError('production live-module guard code drift was accepted')
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
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_inspection_bytes_are_parseable_and_score_bearing(modules: _Modules) -> None:
    result = _consume(modules, _issue(modules))
    api = modules.bundle._SCORER_API_CACHE
    receipt = api.parse(result.score_receipt_bytes)
    assert type(receipt) is api.receipt_type
    assert api.receipt_canonical_json(receipt) == result.score_receipt_bytes
    decoded_trace = api.module.extract_canonical_reward_trace(result.reward_artifact_bytes)
    assert decoded_trace == _TRACE
    assert b'"cumulative_reward":' in result.score_receipt_bytes


def test_builds_exact_immutable_inventory_with_zero_byte_streams(modules: _Modules) -> None:
    capability = _issue(
        modules,
        handoff_capability=_handoff_capability(modules, stdout=b"", stderr=b""),
    )
    result = _consume(modules, capability)
    assert type(result) is modules.bundle.MatchedV3LocalRewardBundle
    assert tuple(item.path for item in result.inventory) == _EXPECTED_FILENAMES
    assert tuple(result.file_bytes(item.path) for item in result.inventory)
    assert result.file_bytes("stdout.bin") == b""
    assert result.file_bytes("stderr.bin") == b""
    assert result.file_bytes("reward-trace.npz") != _TRACE
    for item in result.inventory:
        raw = result.file_bytes(item.path)
        assert type(raw) is bytes
        assert len(raw) == item.size_bytes
        assert hashlib.sha256(raw).hexdigest() == item.sha256
        assert "/" not in item.path and "\\" not in item.path
    manifest = modules.bundle.parse_matched_v3_local_reward_bundle_manifest(
        result.local_bundle_manifest_bytes,
        expected_full_file_sha256=hashlib.sha256(
            result.local_bundle_manifest_bytes
        ).hexdigest(),
    )
    publication = modules.bundle.parse_matched_v3_local_reward_publication_manifest(
        result.publication_manifest_bytes,
        expected_full_file_sha256=hashlib.sha256(
            result.publication_manifest_bytes
        ).hexdigest(),
    )
    assert manifest["cell"] == {
        "candidate_id": _CANDIDATE_ID,
        "environment_seed": _ENVIRONMENT_SEED,
        "agent_seed": _AGENT_SEED,
    }
    assert manifest["source_binding"]["local_source_tree"]["full_sha256"] == (
        _SOURCE_FULL_SHA256
    )
    assert manifest["source_binding"]["local_source_tree"]["tree_sha256"] == (
        _SOURCE_TREE_SHA256
    )
    assert publication["inventory"]["exact_filenames"] == list(_EXPECTED_FILENAMES)
    assert manifest["inspection"]["public_bundle_is_score_bearing"] is True
    assert manifest["inspection"]["public_bundle_is_permanently_nonqualifying"] is True
    assert manifest["qualification_publisher"]["may_accept_public_bundle_object"] is False
    assert b'"cumulative_reward":' not in result.local_bundle_manifest_bytes
    assert b'"cumulative_reward":' not in result.publication_manifest_bytes


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("candidate_id", "causal_e025_q075"),
        ("environment_seed", 18),
        ("agent_seed", 24),
        ("source_full_sha256", "8" * 64),
        ("source_tree_sha256", "9" * 64),
    ],
)
def test_cross_pairing_and_cell_mismatch_fail_closed(
    modules: _Modules,
    override: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {override: value}
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        cast(Any, _issue)(modules, **kwargs)


def test_creation_and_access_require_exact_separate_opt_ins(modules: _Modules) -> None:
    handoff = _handoff_capability(modules)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules, handoff_capability=handoff, explicit_opt_in=False)
    capability = _issue(modules, handoff_capability=handoff)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _consume(modules, capability, opt_in=False)
    assert type(_consume(modules, capability)) is modules.bundle.MatchedV3LocalRewardBundle


@pytest.mark.parametrize(
    "unsafe_kind",
    ("mode_0755", "symlink", "missing", "noncanonical", "wrong_type"),
)
def test_predictably_unsafe_publication_parent_does_not_claim_live_capability(
    modules: _Modules,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    safe = tmp_path / f"safe-{unsafe_kind}"
    safe.mkdir(mode=0o700)
    safe.chmod(0o700)
    unsafe: object
    if unsafe_kind == "mode_0755":
        safe.chmod(0o755)
        unsafe = safe
    elif unsafe_kind == "symlink":
        alias = tmp_path / "publication-parent-alias"
        alias.symlink_to(safe, target_is_directory=True)
        unsafe = alias
    elif unsafe_kind == "missing":
        unsafe = tmp_path / "missing-publication-parent"
    elif unsafe_kind == "noncanonical":
        unsafe = safe / ".." / safe.name
    else:
        unsafe = str(safe)

    capability = _issue(modules)
    with pytest.raises(
        (
            modules.atomic.ForagerMatchedV3AtomicPublicationError,
            modules.publisher.ForagerMatchedV3LocalRewardPublicationError,
            modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        )
    ):
        _publish_direct(modules, capability, unsafe)
    assert modules.bundle._BUNDLE_CAPABILITIES[capability].status == "live"
    assert tuple(safe.iterdir()) == ()

    safe.chmod(0o700)
    metadata = _publish_direct(modules, capability, safe)
    assert type(metadata) is modules.publisher.MatchedV3LocalRewardPublicationMetadata
    assert metadata.publication_root.parent == safe
    assert modules.bundle._BUNDLE_CAPABILITIES[capability].status == "consumed"


def test_bundle_capability_is_single_use_uncopyable_and_unserializable(modules: _Modules) -> None:
    capability = _issue(modules)
    with pytest.raises(TypeError):
        copy.copy(capability)
    with pytest.raises(TypeError):
        copy.deepcopy(capability)
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    _consume(modules, capability)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _consume(modules, capability)


def test_live_registry_retains_only_private_sealed_payload(modules: _Modules) -> None:
    capability = _issue(modules)
    state = modules.bundle._BUNDLE_CAPABILITIES[capability]
    assert type(state.sealed_payload) is modules.bundle._SealedLocalRewardPayload
    assert not hasattr(state, "bundle")
    assert not any(
        type(value) is modules.bundle.MatchedV3LocalRewardBundle
        for value in (
            state.handoff_capability,
            state.handoff_content,
            state.sealed_payload,
        )
    )
    _consume(modules, capability)


def test_bundle_capability_cannot_cross_pid_boundary(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _issue(modules)
    original_pid = os.getpid()
    monkeypatch.setattr(modules.bundle.os, "getpid", lambda: original_pid + 1)
    with pytest.raises(
        modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        match="PID",
    ):
        _consume(modules, capability)
    monkeypatch.undo()
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _consume(modules, capability)


@pytest.mark.parametrize(
    "role",
    (
        "publication_manifest",
        "local_bundle_manifest",
        "bootstrap_receipt",
        "bootstrap_child_record",
        "local_runner_receipt",
        "reward_trace",
        "score_receipt",
        "stdout",
        "stderr",
    ),
)
def test_registry_detects_every_sealed_payload_byte_tamper(
    modules: _Modules,
    role: str,
) -> None:
    capability = _issue(modules)
    state = modules.bundle._BUNDLE_CAPABILITIES[capability]
    mutated = tuple(
        (item_role, path, raw + b"tampered" if item_role == role else raw)
        for item_role, path, raw in state.sealed_payload.role_payloads
    )
    object.__setattr__(state.sealed_payload, "role_payloads", mutated)
    with pytest.raises(
        modules.bundle.ForagerMatchedV3LocalRewardBundleError,
    ):
        _consume(modules, capability)


def test_wrong_or_forged_capability_is_rejected(modules: _Modules) -> None:
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _consume(modules, object())
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules, handoff_capability=object())


def test_scorer_in_memory_function_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consume(modules, _issue(modules))
    scorer = modules.bundle._SCORER_API_CACHE.module
    monkeypatch.setattr(scorer, "canonical_reward_npz_bytes", lambda _trace: b"forged")
    with pytest.raises(
        modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        match="identity drifted",
    ):
        _issue(modules)


@pytest.mark.parametrize(
    "method_name",
    _RECEIPT_METHOD_NAMES,
)
def test_receipt_method_in_place_code_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    _consume(modules, _issue(modules))
    receipt_type = modules.bundle._SCORER_API_CACHE.receipt_type
    method = vars(receipt_type)[method_name]
    code = method.__code__
    drifted = code.replace(co_consts=(*code.co_consts, "adversarial-code-drift"))
    monkeypatch.setattr(method, "__code__", drifted)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules)


def test_receipt_semantic_surface_covers_every_method_and_property(
    modules: _Modules,
) -> None:
    _consume(modules, _issue(modules))
    receipt_type = modules.bundle._SCORER_API_CACHE.receipt_type
    surface = modules.bundle._receipt_accessor_codes(receipt_type)
    callable_accessors = cast(dict[str, Any], surface["callable_accessors"])
    expected = {
        name
        for name, value in vars(receipt_type).items()
        if type(value) in {types.FunctionType, property, staticmethod, classmethod}
    }
    assert set(callable_accessors) == expected
    assert expected == {*_RECEIPT_METHOD_NAMES, "receipt_sha256"}


def test_receipt_property_accessor_in_place_code_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consume(modules, _issue(modules))
    receipt_type = modules.bundle._SCORER_API_CACHE.receipt_type
    receipt_property = vars(receipt_type)["receipt_sha256"]
    accessor = receipt_property.fget
    assert accessor is not None
    code = accessor.__code__
    drifted = code.replace(co_consts=(*code.co_consts, "adversarial-accessor-drift"))
    monkeypatch.setattr(accessor, "__code__", drifted)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules)


def test_raw_trace_digest_domain_global_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consume(modules, _issue(modules))
    scorer = modules.bundle._SCORER_API_CACHE.module
    monkeypatch.setattr(scorer, "RAW_TRACE_DIGEST_DOMAIN", b"forged-domain")
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules)


def test_protocol_mutable_metric_global_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consume(modules, _issue(modules))
    protocol = modules.bundle._SCORER_API_CACHE.protocol_module
    metric = protocol._CUMULATIVE_REWARD_METRIC
    monkeypatch.setitem(metric, "horizon", 1)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        _issue(modules)


def test_clean_semantic_surface_rejects_drift_before_initial_cache_validation() -> None:
    script = r'''\
import hashlib
import sys
import types
from pathlib import Path

path = Path(sys.argv[1]).resolve()
source = path.read_bytes()
name = '_alberta_forager_matched_v3_local_reward_bundle_isolated_v1'
module = types.ModuleType(name)
module.__file__ = str(path)
module.__package__ = ''
module.__dict__['_MATCHED_V3_LOCAL_REWARD_BUNDLE_SOURCE_SHA256'] = (
    hashlib.sha256(source).hexdigest()
)
module.__dict__['_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256'] = (
    'a5275d77d9b0870214b19c31acad73841f12c217f6eb411a6f8c56e317cc0819'
)
sys.modules[name] = module
exec(compile(source, str(path), 'exec'), module.__dict__)
scorer = module._direct_load_stdlib_scorer()
method = vars(scorer.MatchedV3ScoreReceipt)['to_body']
def replacement(_self):
    return {}
method.__code__ = replacement.__code__
try:
    module._validate_scorer_module(scorer)
except module.ForagerMatchedV3LocalRewardBundleError:
    pass
else:
    raise AssertionError('pre-cache semantic scorer drift was accepted')
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(_BUNDLE_PATH)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_own_helper_in_place_code_drift_fails_closed(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = modules.bundle._read_exact_source_bytes

    def replacement(*_args: object, **_kwargs: object) -> bytes:
        return b""

    monkeypatch.setattr(helper, "__code__", replacement.__code__)
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        modules.bundle.matched_v3_local_reward_bundle_descriptor()


def _rehash_json(payload: dict[str, Any], body_digest_key: str) -> bytes:
    body = dict(payload)
    body.pop(body_digest_key, None)
    canonical_body = (
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    body[body_digest_key] = hashlib.sha256(canonical_body).hexdigest()
    return (
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def test_publication_manifest_body_binding_cross_checks_local_manifest(
    modules: _Modules,
) -> None:
    result = _consume(modules, _issue(modules))
    payload = cast(dict[str, Any], json.loads(result.publication_manifest_bytes))
    binding = cast(dict[str, Any], payload["bundle_binding"])
    binding["manifest_body_sha256"] = "a" * 64
    raw = _rehash_json(payload, "publication_body_sha256")
    object.__setattr__(result, "publication_manifest_bytes", raw)
    role_bytes = modules.bundle._bundle_role_bytes(result)
    object.__setattr__(
        result,
        "inventory",
        modules.bundle._inventory_from_role_bytes(role_bytes),
    )
    with pytest.raises(
        modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        match="manifest body",
    ):
        modules.bundle._validate_bundle_structure(result, scorer_api=None)


def test_manifest_parser_rejects_duplicate_keys(modules: _Modules) -> None:
    result = _consume(modules, _issue(modules))
    raw = result.local_bundle_manifest_bytes.replace(
        b'{"cell":', b'{"cell":{},"cell":', 1
    )
    with pytest.raises(
        modules.bundle.ForagerMatchedV3LocalRewardBundleError,
        match="duplicate",
    ):
        modules.bundle.parse_matched_v3_local_reward_bundle_manifest(
            raw,
            expected_full_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_manifest_parser_rejects_path_traversal_even_when_rehashed(modules: _Modules) -> None:
    result = _consume(modules, _issue(modules))
    payload = cast(dict[str, Any], json.loads(result.publication_manifest_bytes))
    files = cast(dict[str, Any], payload["files"])
    record = cast(dict[str, Any], files["stdout"])
    record["path"] = "../stdout.bin"
    raw = _rehash_json(payload, "publication_body_sha256")
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        modules.bundle.parse_matched_v3_local_reward_publication_manifest(
            raw,
            expected_full_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_manifest_parser_rejects_nonfinite_or_float_shapes(modules: _Modules) -> None:
    result = _consume(modules, _issue(modules))
    for raw in (
        result.local_bundle_manifest_bytes.replace(b'"agent_seed":23', b'"agent_seed":NaN'),
        result.local_bundle_manifest_bytes.replace(b'"agent_seed":23', b'"agent_seed":23.0'),
    ):
        with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
            modules.bundle.parse_matched_v3_local_reward_bundle_manifest(
                raw,
                expected_full_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_bundle_file_lookup_rejects_unknown_and_nonexact_names(modules: _Modules) -> None:
    result = _consume(modules, _issue(modules))
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        result.file_bytes("../stdout.bin")
    with pytest.raises(modules.bundle.ForagerMatchedV3LocalRewardBundleError):
        result.file_bytes(cast(Any, Path("stdout.bin")))


def test_handoff_is_consumed_when_bundle_issuance_succeeds(modules: _Modules) -> None:
    handoff = _handoff_capability(modules)
    capability = _issue(modules, handoff_capability=handoff)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError):
        modules.handoff.consume_matched_v3_local_result_handoff(
            handoff_capability=handoff,
            explicit_content_access_opt_in=True,
        )
    _consume(modules, capability)


def test_bundle_binds_final_audited_handoff_identities(modules: _Modules) -> None:
    assert modules.bundle.PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256 == (
        "a5275d77d9b0870214b19c31acad73841f12c217f6eb411a6f8c56e317cc0819"
    )
    assert modules.bundle.PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256 == (
        "dc488f74d50ef224309e89968559df4671f4a3f954144530a9e4424e3cabba03"
    )
