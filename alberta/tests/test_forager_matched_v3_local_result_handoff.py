"""Adversarial tests for the isolated matched-v3 local result handoff."""

from __future__ import annotations

import copy
import gc
import hashlib
import os
import pickle
import subprocess
import sys
import types
import weakref
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
_BOOTSTRAP_NAME = "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
_HANDOFF_NAME = "_alberta_forager_matched_v3_local_result_handoff_isolated_v1"


def _direct_module(
    *,
    source_path: Path,
    module_name: str,
    injections: dict[str, Any],
) -> types.ModuleType:
    source = source_path.read_bytes()
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    module.__dict__.update(injections)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class _Modules:
    bootstrap: types.ModuleType
    handoff: types.ModuleType
    bootstrap_source_sha256: str
    handoff_source_sha256: str


@pytest.fixture(scope="module")
def modules() -> Iterator[_Modules]:
    bootstrap_source = _BOOTSTRAP_PATH.read_bytes()
    bootstrap_sha256 = hashlib.sha256(bootstrap_source).hexdigest()
    bootstrap = _direct_module(
        source_path=_BOOTSTRAP_PATH,
        module_name=_BOOTSTRAP_NAME,
        injections={
            "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256": bootstrap_sha256,
        },
    )
    # The test process may already contain numerical libraries.  The exact
    # source/module/function gates remain active; only the ambient observation
    # is bypassed for this in-process structural fixture.
    setattr(bootstrap, "_ISOLATED_PARENT_BOUNDARY", True)
    setattr(bootstrap, "_live_forbidden_modules", lambda: ())

    handoff_source = _HANDOFF_PATH.read_bytes()
    handoff_sha256 = hashlib.sha256(handoff_source).hexdigest()
    handoff = _direct_module(
        source_path=_HANDOFF_PATH,
        module_name=_HANDOFF_NAME,
        injections={
            "_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256": handoff_sha256,
            "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256": bootstrap_sha256,
        },
    )
    setattr(handoff, "_ISOLATED_HANDOFF_BOUNDARY", True)
    setattr(handoff, "_live_forbidden_modules", lambda: ())
    yield _Modules(bootstrap, handoff, bootstrap_sha256, handoff_sha256)
    sys.modules.pop(_HANDOFF_NAME, None)
    sys.modules.pop(_BOOTSTRAP_NAME, None)


@dataclass(frozen=True)
class _AuthenticFixture:
    outcome: object
    completion: Any
    bootstrap_receipt: bytes
    child_record: bytes
    local_receipt: bytes
    reward_trace: bytes
    stdout: bytes
    stderr: bytes


def _local_receipt(
    bootstrap: types.ModuleType,
    *,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    reward_trace: bytes,
) -> bytes:
    body = {
        "schema_version": "alberta.forager_matched_v3.local_runner_completion.v1",
        "status": "completed_unqualified_non_authorizing",
        "classification": "content_only_unqualified_execution_completion",
        "candidate": {
            "candidate_id": candidate_id,
            "implementation_kind": "test_only_structural_fixture",
        },
        "seed_transport": {
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
            "environment_transport": "runner_seeds_single_lane_uint31",
            "agent_transport": "runner_agent_seeds_single_lane_uint31",
            "environment_agent_seed_collision": environment_seed == agent_seed,
            "environment_agent_seed_collisions_allowed": True,
        },
        "reward_trace": {
            "size_bytes": len(reward_trace),
            "sha256": hashlib.sha256(reward_trace).hexdigest(),
        },
        "claims": {"scientific_evidence_created": False},
    }
    return cast(
        bytes,
        bootstrap._canonical_json(
            {
                **body,
                "receipt_body_sha256": hashlib.sha256(
                    bootstrap._canonical_json(body)
                ).hexdigest(),
            }
        ),
    )


def _authentic_outcome(
    modules: _Modules,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    reward_trace: bytes = b"\x00\x01\xff",
) -> _AuthenticFixture:
    """Insert one narrow test-only authentic outcome into bootstrap registries."""

    bootstrap = modules.bootstrap
    candidate_id = "causal_e025_q050"
    environment_seed = 17
    agent_seed = 23
    local_receipt = _local_receipt(
        bootstrap,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        reward_trace=reward_trace,
    )
    repository_path = "/fixture/repository"
    runtime_paths = ("/fixture/runtime",)
    stdlib_paths = ("/fixture/stdlib",)
    final_paths = (repository_path, *runtime_paths, *stdlib_paths)
    request_contract = {
        "argv_sha256": "1" * 64,
        "argv_count": 12,
        "environment_sha256": "2" * 64,
        "cwd_path_sha256": bootstrap._path_sha256(repository_path),
        "executable_path_sha256": "3" * 64,
        "executable_source_sha256": "4" * 64,
        "executable_proc_path_sha256": "5" * 64,
        "executable_fd": 7,
        "cache_proc_path": "/proc/self/fd/9",
        "jax_platform_selector": "cpu",
        "jax_compilation_cache_enabled": False,
    }
    child_process = bootstrap._child_process_record(
        request_contract=request_contract,
        original_stdlib_paths=stdlib_paths,
        final_sys_path=final_paths,
        repository_path=repository_path,
        runtime_paths=runtime_paths,
        cache_proc_path=request_contract["cache_proc_path"],
    )
    manifest_full_sha256 = "6" * 64
    manifest_tree_sha256 = "7" * 64
    child_record = bootstrap._build_child_record(
        bootstrap_source_sha256=modules.bootstrap_source_sha256,
        request_sha256="8" * 64,
        manifest_full_sha256=manifest_full_sha256,
        manifest_tree_sha256=manifest_tree_sha256,
        pre_full_sha256=manifest_full_sha256,
        pre_tree_sha256=manifest_tree_sha256,
        post_full_sha256=manifest_full_sha256,
        post_tree_sha256=manifest_tree_sha256,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        process_contract=child_process,
        local_receipt=local_receipt,
        reward_trace=reward_trace,
    )
    process_contract = {
        **request_contract,
        "launcher_sha256": bootstrap._CHILD_LAUNCHER_SHA256,
        "home_xdg_cache_proc_path": request_contract["cache_proc_path"],
        "required_flags": ["-I", "-S", "-B"],
        "stdin": "devnull_closed_to_input",
        "start_new_session": True,
        "site_initialization": False,
        "pth_processing": False,
    }
    manifest_identity = bootstrap._SnapshotManifestIdentity(
        full_sha256=manifest_full_sha256,
        tree_sha256=manifest_tree_sha256,
        snapshot_source_size=1,
        snapshot_source_sha256=bootstrap.PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
        runner_source_size=1,
        runner_source_sha256=bootstrap.PINNED_LOCAL_RUNNER_SOURCE_SHA256,
    )
    process_result = bootstrap._ProcessResult(0, stdout, stderr)
    frame = bootstrap._result_frame(child_record, local_receipt, reward_trace)
    ceilings = {
        "maximum_request_bytes": 1024 * 1024,
        "maximum_bootstrap_source_bytes": 8 * 1024 * 1024,
        "maximum_result_bytes": 16 * 1024 * 1024,
        "maximum_stdout_bytes": 1024 * 1024,
        "maximum_stderr_bytes": 1024 * 1024,
        "timeout_seconds": 60,
    }
    bootstrap_receipt = bootstrap._build_bootstrap_receipt(
        bootstrap_source_sha256=modules.bootstrap_source_sha256,
        manifest_identity=manifest_identity,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        process_contract=process_contract,
        process_result=process_result,
        frame=frame,
        child_record=child_record,
        local_receipt=local_receipt,
        reward_trace=reward_trace,
        ceilings=ceilings,
    )
    receipt_sha256 = hashlib.sha256(bootstrap_receipt).hexdigest()
    completion = bootstrap.MatchedV3LocalBootstrapCompletion(
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        canonical_receipt_bytes=bootstrap_receipt,
        receipt_sha256=receipt_sha256,
        canonical_child_record_bytes=child_record,
        local_completion_receipt_bytes=local_receipt,
        reward_trace=reward_trace,
        stdout=stdout,
        stderr=stderr,
    )
    execution = bootstrap._ParentExecutionCapability()
    outcome = bootstrap._ParentOutcomeCapability()
    with bootstrap._CAPABILITY_LOCK:
        bootstrap._EXECUTION_CAPABILITIES[execution] = bootstrap._ExecutionState(
            pid=os.getpid(),
            status="consumed",
        )
        bootstrap._OUTCOME_CAPABILITIES[outcome] = bootstrap._OutcomeState(
            pid=os.getpid(),
            status="live",
            execution_capability=execution,
            execution_identity=id(execution),
            bootstrap_source_sha256=modules.bootstrap_source_sha256,
            receipt_sha256=receipt_sha256,
            child_record_sha256=hashlib.sha256(child_record).hexdigest(),
            local_receipt_sha256=hashlib.sha256(local_receipt).hexdigest(),
            reward_trace_sha256=hashlib.sha256(reward_trace).hexdigest(),
            completion=completion,
        )
    return _AuthenticFixture(
        outcome=outcome,
        completion=completion,
        bootstrap_receipt=bootstrap_receipt,
        child_record=child_record,
        local_receipt=local_receipt,
        reward_trace=reward_trace,
        stdout=stdout,
        stderr=stderr,
    )


def _issue(modules: _Modules, fixture: _AuthenticFixture) -> object:
    return modules.handoff.issue_matched_v3_local_result_handoff(
        bootstrap_outcome_capability=fixture.outcome,
        explicit_handoff_opt_in=True,
    )


def _consume(modules: _Modules, capability: object) -> Any:
    return modules.handoff.consume_matched_v3_local_result_handoff(
        handoff_capability=capability,
        explicit_content_access_opt_in=True,
    )


def test_descriptor_is_canonical_pinned_and_all_authority_is_false(modules: _Modules) -> None:
    handoff = modules.handoff
    raw = handoff.canonical_matched_v3_local_result_handoff_descriptor_bytes()
    assert hashlib.sha256(raw).hexdigest() == handoff.LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256
    assert (
        handoff.LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256
        == "dc488f74d50ef224309e89968559df4671f4a3f954144530a9e4424e3cabba03"
    )
    descriptor = handoff.parse_matched_v3_local_result_handoff_descriptor(raw)
    assert descriptor == handoff.matched_v3_local_result_handoff_descriptor()
    assert descriptor["status"] == "implemented_unexecuted_non_authorizing"
    assert descriptor["module_import"]["workload_execution"] is False
    assert descriptor["module_import"]["default_workload_paths"] is False
    assert descriptor["handoff"]["serialized_bytes_accepted"] is False
    assert descriptor["handoff"]["plain_bootstrap_completion_accepted"] is False
    assert descriptor["claims"]
    assert all(value is False for value in descriptor["claims"].values())


def test_true_isolated_loader_rejects_preloaded_forbidden_module() -> None:
    script = """
import hashlib, sys, types
from pathlib import Path
path = Path(sys.argv[1]).resolve()
source = path.read_bytes()
sys.modules['numpy'] = types.ModuleType('numpy')
name = '_alberta_forager_matched_v3_local_result_handoff_isolated_v1'
module = types.ModuleType(name)
module.__file__ = str(path)
module.__package__ = ''
module.__dict__['_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256'] = (
    hashlib.sha256(source).hexdigest()
)
module.__dict__['_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256'] = '1' * 64
sys.modules[name] = module
exec(compile(source, str(path), 'exec'), module.__dict__)
try:
    module._require_handoff_boundary(require_current_source=True)
except module.ForagerMatchedV3LocalResultHandoffError as exc:
    assert 'isolated direct-byte' in str(exc) or 'preloaded runtime' in str(exc)
else:
    raise AssertionError('preloaded forbidden module was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(_HANDOFF_PATH)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_authentic_outcome_becomes_new_one_shot_handoff_with_exact_zero_stdio(
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(modules, stdout=b"", stderr=b"")
    capability = _issue(modules, fixture)
    assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "consumed"
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(capability)
    content = _consume(modules, capability)
    assert content.candidate_id == "causal_e025_q050"
    assert content.environment_seed == 17
    assert content.agent_seed == 23
    assert content.canonical_bootstrap_receipt_bytes == fixture.bootstrap_receipt
    assert content.canonical_bootstrap_child_record_bytes == fixture.child_record
    assert content.canonical_local_runner_receipt_bytes == fixture.local_receipt
    assert content.raw_reward_trace_bytes == fixture.reward_trace
    assert content.stdout_bytes == b""
    assert content.stderr_bytes == b""
    assert content.stdout_sha256 == hashlib.sha256(b"").hexdigest()
    assert content.stderr_sha256 == hashlib.sha256(b"").hexdigest()
    record = content.record()
    assert record["provenance"]["authentic_bootstrap_outcome_consumed"] is True
    assert record["provenance"]["bootstrap_completion_returned_to_creation_caller"] is False
    assert record["source_binding"]["handoff_source_sha256"] == modules.handoff_source_sha256
    assert (
        record["source_binding"]["bootstrap"]["source_sha256"]
        == modules.bootstrap_source_sha256
    )
    assert all(value is False for value in record["claims"].values())
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="consumed"):
        _consume(modules, capability)


def test_nonempty_stdout_stderr_are_preserved_without_interpretation(modules: _Modules) -> None:
    fixture = _authentic_outcome(
        modules,
        stdout=b"opaque stdout\x00bytes",
        stderr=b"opaque stderr\xffbytes",
    )
    content = _consume(modules, _issue(modules, fixture))
    assert content.stdout_bytes == fixture.stdout
    assert content.stderr_bytes == fixture.stderr
    record = content.record()
    assert record["artifacts"]["stdout"] == {
        "size_bytes": len(fixture.stdout),
        "content_sha256": hashlib.sha256(fixture.stdout).hexdigest(),
    }
    assert record["artifacts"]["stderr"] == {
        "size_bytes": len(fixture.stderr),
        "content_sha256": hashlib.sha256(fixture.stderr).hexdigest(),
    }


def test_wrong_opt_ins_do_not_spend_live_capabilities(modules: _Modules) -> None:
    fixture = _authentic_outcome(modules)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="opt-in"):
        modules.handoff.issue_matched_v3_local_result_handoff(
            bootstrap_outcome_capability=fixture.outcome,
            explicit_handoff_opt_in=1,
        )
    capability = _issue(modules, fixture)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="opt-in"):
        modules.handoff.consume_matched_v3_local_result_handoff(
            handoff_capability=capability,
            explicit_content_access_opt_in=1,
        )
    assert _consume(modules, capability).candidate_id == "causal_e025_q050"


@pytest.mark.parametrize("kind", ["completion", "bytes", "forged_handoff"])
def test_plain_serialized_or_forged_values_never_recreate_authority(
    modules: _Modules,
    kind: str,
) -> None:
    fixture = _authentic_outcome(modules)
    if kind == "completion":
        value = fixture.completion
        with pytest.raises(
            modules.handoff.ForagerMatchedV3LocalResultHandoffError,
            match="authentic opaque bootstrap outcome",
        ):
            modules.handoff.issue_matched_v3_local_result_handoff(
                bootstrap_outcome_capability=value,
                explicit_handoff_opt_in=True,
            )
        assert _consume(modules, _issue(modules, fixture)).candidate_id
    elif kind == "bytes":
        with pytest.raises(
            modules.handoff.ForagerMatchedV3LocalResultHandoffError,
            match="authentic opaque bootstrap outcome",
        ):
            modules.handoff.issue_matched_v3_local_result_handoff(
                bootstrap_outcome_capability=fixture.bootstrap_receipt,
                explicit_handoff_opt_in=True,
            )
        assert _consume(modules, _issue(modules, fixture)).candidate_id
    else:
        forged = object.__new__(modules.handoff._LocalResultHandoffCapability)
        with pytest.raises(
            modules.handoff.ForagerMatchedV3LocalResultHandoffError,
            match="unknown",
        ):
            _consume(modules, forged)


def test_already_consumed_bootstrap_outcome_cannot_create_handoff(modules: _Modules) -> None:
    fixture = _authentic_outcome(modules)
    modules.bootstrap.consume_matched_v3_local_bootstrap_outcome(
        outcome_capability=fixture.outcome,
        explicit_content_access_opt_in=True,
    )
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="consumption failed",
    ):
        _issue(modules, fixture)


def test_handoff_is_pid_bound_and_pid_failure_permanently_spends_it(modules: _Modules) -> None:
    fixture = _authentic_outcome(modules)
    capability = _issue(modules, fixture)
    modules.handoff._HANDOFF_CAPABILITIES[capability].pid += 1
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="PID"):
        _consume(modules, capability)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="consumed"):
        _consume(modules, capability)


def test_handoff_source_drift_is_fail_closed_and_permanently_consuming(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
) -> None:
    capability = _issue(modules, _authentic_outcome(modules))
    original = modules.handoff._current_handoff_source_sha256
    monkeypatch.setattr(modules.handoff, "_current_handoff_source_sha256", lambda: "d" * 64)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="stale"):
        _consume(modules, capability)
    monkeypatch.setattr(modules.handoff, "_current_handoff_source_sha256", original)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="consumed"):
        _consume(modules, capability)


def test_source_drift_before_issue_does_not_expose_completion_or_spend_outcome(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(modules)
    original = modules.handoff._current_handoff_source_sha256
    monkeypatch.setattr(modules.handoff, "_current_handoff_source_sha256", lambda: "e" * 64)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="stale"):
        _issue(modules, fixture)
    assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "live"
    monkeypatch.setattr(modules.handoff, "_current_handoff_source_sha256", original)
    assert _consume(modules, _issue(modules, fixture)).candidate_id


def test_bootstrap_source_drift_is_fail_closed_and_permanently_consuming(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
) -> None:
    capability = _issue(modules, _authentic_outcome(modules))
    original = modules.handoff._read_exact_source_sha256

    def drift_bootstrap(module: types.ModuleType, label: str) -> str:
        if label == "local bootstrap":
            return "f" * 64
        return cast(str, original(module, label))

    monkeypatch.setattr(modules.handoff, "_read_exact_source_sha256", drift_bootstrap)
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="bootstrap source bytes",
    ):
        _consume(modules, capability)
    monkeypatch.setattr(modules.handoff, "_read_exact_source_sha256", original)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="consumed"):
        _consume(modules, capability)


@pytest.mark.parametrize(
    "function_name",
    [
        "consume_matched_v3_local_bootstrap_outcome",
        "parse_matched_v3_local_bootstrap_receipt",
    ],
)
def test_bootstrap_function_monkeypatch_is_rejected_before_outcome_consumption(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
    function_name: str,
) -> None:
    fixture = _authentic_outcome(modules)
    original = getattr(modules.bootstrap, function_name)
    monkeypatch.setattr(modules.bootstrap, function_name, lambda **_kwargs: fixture.completion)
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="function",
    ):
        _issue(modules, fixture)
    assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "live"
    monkeypatch.setattr(modules.bootstrap, function_name, original)
    assert _consume(modules, _issue(modules, fixture)).candidate_id


def test_replaced_fake_bootstrap_module_is_rejected_before_outcome_consumption(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(modules)
    fake = types.ModuleType(_BOOTSTRAP_NAME)
    fake.__package__ = ""
    fake.__file__ = str(_BOOTSTRAP_PATH)
    setattr(
        fake,
        "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION",
        modules.bootstrap.LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
    )
    setattr(
        fake,
        "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256",
        modules.bootstrap.LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
    )
    setattr(fake, "LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME", _BOOTSTRAP_NAME)
    setattr(
        fake,
        "consume_matched_v3_local_bootstrap_outcome",
        modules.bootstrap.consume_matched_v3_local_bootstrap_outcome,
    )
    setattr(
        fake,
        "parse_matched_v3_local_bootstrap_receipt",
        modules.bootstrap.parse_matched_v3_local_bootstrap_receipt,
    )
    monkeypatch.setitem(sys.modules, _BOOTSTRAP_NAME, fake)
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="absent, replaced",
    ):
        _issue(modules, fixture)
    assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "live"
    monkeypatch.setitem(sys.modules, _BOOTSTRAP_NAME, modules.bootstrap)
    assert _consume(modules, _issue(modules, fixture)).candidate_id


def test_nested_bootstrap_validation_function_monkeypatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(modules)
    original = modules.bootstrap._validate_bootstrap_receipt
    monkeypatch.setattr(modules.bootstrap, "_validate_bootstrap_receipt", lambda *_a, **_k: None)
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="function surface",
    ):
        _issue(modules, fixture)
    assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "live"
    monkeypatch.setattr(modules.bootstrap, "_validate_bootstrap_receipt", original)
    assert _consume(modules, _issue(modules, fixture)).candidate_id


def test_in_place_bootstrap_helper_code_replacement_is_rejected(
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(modules)
    helper = modules.bootstrap._validate_bootstrap_receipt
    original_code = helper.__code__
    helper.__code__ = (lambda *_args, **_kwargs: None).__code__
    try:
        with pytest.raises(
            modules.handoff.ForagerMatchedV3LocalResultHandoffError,
            match="function surface",
        ):
            _issue(modules, fixture)
        assert modules.bootstrap._OUTCOME_CAPABILITIES[fixture.outcome].status == "live"
    finally:
        helper.__code__ = original_code
    assert _consume(modules, _issue(modules, fixture)).candidate_id


def test_cross_pairing_authentic_receipt_with_other_valid_records_is_rejected(
    modules: _Modules,
) -> None:
    left = _authentic_outcome(modules, reward_trace=b"\x00\x01\xff")
    right = _authentic_outcome(modules, reward_trace=b"\x01\x00\xff")
    with pytest.raises(
        modules.bootstrap.ForagerMatchedV3LocalExecutionBootstrapError,
        match="linkage drifted",
    ):
        modules.bootstrap.parse_matched_v3_local_bootstrap_receipt(
            left.bootstrap_receipt,
            expected_receipt_sha256=hashlib.sha256(left.bootstrap_receipt).hexdigest(),
            canonical_child_record_bytes=right.child_record,
            local_completion_receipt_bytes=left.local_receipt,
            reward_trace=left.reward_trace,
            stdout=left.stdout,
            stderr=left.stderr,
        )
    with pytest.raises(
        modules.bootstrap.ForagerMatchedV3LocalExecutionBootstrapError,
        match="linkage drifted",
    ):
        modules.bootstrap.parse_matched_v3_local_bootstrap_receipt(
            left.bootstrap_receipt,
            expected_receipt_sha256=hashlib.sha256(left.bootstrap_receipt).hexdigest(),
            canonical_child_record_bytes=left.child_record,
            local_completion_receipt_bytes=right.local_receipt,
            reward_trace=left.reward_trace,
            stdout=left.stdout,
            stderr=left.stderr,
        )


def test_all_six_stream_lengths_and_digests_are_bound_before_content_consumption(
    modules: _Modules,
) -> None:
    fixture = _authentic_outcome(
        modules,
        stdout=b"stdout-before-consume",
        stderr=b"stderr-before-consume",
    )
    capability = _issue(modules, fixture)
    content = modules.handoff._HANDOFF_CAPABILITIES[capability].content
    record = content.record()
    expected = {
        "bootstrap_receipt": (
            fixture.bootstrap_receipt,
            "full_file_sha256",
        ),
        "bootstrap_child_record": (
            fixture.child_record,
            "full_file_sha256",
        ),
        "local_runner_receipt": (
            fixture.local_receipt,
            "full_file_sha256",
        ),
        "raw_reward_trace": (fixture.reward_trace, "content_sha256"),
        "stdout": (fixture.stdout, "content_sha256"),
        "stderr": (fixture.stderr, "content_sha256"),
    }
    assert set(record["artifacts"]) == set(expected)
    for name, (raw, digest_key) in expected.items():
        artifact = record["artifacts"][name]
        assert artifact["size_bytes"] == len(raw)
        assert artifact[digest_key] == hashlib.sha256(raw).hexdigest()
    assert modules.handoff._HANDOFF_CAPABILITIES[capability].status == "live"
    consumed = _consume(modules, capability)
    assert consumed is content


def test_tampered_immutable_content_is_rejected_after_capability_is_spent(
    modules: _Modules,
) -> None:
    capability = _issue(modules, _authentic_outcome(modules))
    content = modules.handoff._HANDOFF_CAPABILITIES[capability].content
    object.__setattr__(content, "stdout_bytes", b"tampered")
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="content bytes are stale",
    ):
        _consume(modules, capability)
    with pytest.raises(modules.handoff.ForagerMatchedV3LocalResultHandoffError, match="consumed"):
        _consume(modules, capability)


def test_detached_record_replay_rejects_any_artifact_tamper(modules: _Modules) -> None:
    content = _consume(modules, _issue(modules, _authentic_outcome(modules)))
    with pytest.raises(
        modules.handoff.ForagerMatchedV3LocalResultHandoffError,
        match="digest linkage",
    ):
        modules.handoff.parse_matched_v3_local_result_handoff_record(
            content.canonical_handoff_record_bytes,
            expected_record_sha256=content.handoff_record_sha256,
            bootstrap_receipt_bytes=content.canonical_bootstrap_receipt_bytes,
            bootstrap_child_record_bytes=content.canonical_bootstrap_child_record_bytes,
            local_runner_receipt_bytes=content.canonical_local_runner_receipt_bytes,
            raw_reward_trace_bytes=b"\x01" + content.raw_reward_trace_bytes[1:],
            stdout_bytes=content.stdout_bytes,
            stderr_bytes=content.stderr_bytes,
        )


def test_weak_registry_does_not_keep_dropped_handoff_alive(modules: _Modules) -> None:
    fixture = _authentic_outcome(modules)
    capability = _issue(modules, fixture)
    reference = weakref.ref(capability)
    assert capability in modules.handoff._HANDOFF_CAPABILITIES
    del capability
    gc.collect()
    assert reference() is None
    assert len(modules.handoff._HANDOFF_CAPABILITIES) == 0


def test_no_workload_or_publication_surface_is_exported(modules: _Modules) -> None:
    exported = set(modules.handoff.__all__)
    assert not any("execute" in name or "publish" in name or "qualify" in name for name in exported)
    source = _HANDOFF_PATH.read_text(encoding="utf-8")
    assert "import jax" not in source
    assert "import numpy" not in source
    assert "import foragax" not in source
    assert "import subprocess" not in source
