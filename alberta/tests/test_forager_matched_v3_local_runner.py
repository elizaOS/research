"""Cheap fake-runtime tests for the capability-gated matched-v3 local runner."""

from __future__ import annotations

import builtins
import copy
import dataclasses
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import subprocess
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

_RUNNER_MODULE_NAME = "_alberta_forager_matched_v3_local_runner_isolated_v1"
_RUNNER_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_runner.py"
)
_BOOTSTRAP_GLOBAL_NAME = "_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"


def _load_test_runner_at_isolated_boundary() -> Any:
    # The repository-wide pytest conftest imports JAX, so this in-process instance
    # can only exercise the fake-runtime state machine. Patch its private captured
    # boundary after loading; subprocess tests below exercise the real boundary.
    source_bytes = _RUNNER_MODULE_PATH.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    spec = importlib.util.spec_from_file_location(
        _RUNNER_MODULE_NAME,
        _RUNNER_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__[_BOOTSTRAP_GLOBAL_NAME] = source_sha256
    sys.modules[_RUNNER_MODULE_NAME] = module
    exec(compile(source_bytes, str(_RUNNER_MODULE_PATH), "exec"), module.__dict__)
    module.__dict__["_PRELOADED_EXECUTION_MODULES"] = ()
    module.__dict__["_ISOLATED_TOP_LEVEL_NAME_BOUNDARY"] = True
    module.__dict__["_ISOLATED_TOP_LEVEL_LOAD"] = True
    module.__dict__["_current_execution_module_bindings"] = lambda: ()
    return module


runner = _load_test_runner_at_isolated_boundary()

_DESCRIPTOR_SHA256 = "2237914749f353d2700bbb0f33a66d8789268a5e156f2961be2e626f42efd2a1"
_HORIZON = 499_712
_TRACE_PATTERN = bytes((255, 0, 1, 30))
_EXPECTED_TRACE = _TRACE_PATTERN * (_HORIZON // len(_TRACE_PATTERN))
_EXPECTED_TOTAL = 30 * (_HORIZON // 4)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any, *, allow_nan: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=allow_nan,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _rehash_receipt(receipt: dict[str, Any]) -> bytes:
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256", None)
    receipt["receipt_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(receipt)


@dataclass
class FakeState:
    behavior: str = "success"
    runner_calls: list[dict[str, Any]] = field(default_factory=list)
    builder_calls: list[str] = field(default_factory=list)
    parser_calls: list[tuple[str, MappingLike]] = field(default_factory=list)
    env_apertures: list[int] = field(default_factory=list)
    benchmark_kwargs: list[dict[str, Any]] = field(default_factory=list)


MappingLike = dict[str, Any]


class _FakeEnvironment:
    resolved_env_id = "ForagaxTwoBiomeLarge-v1"
    resolved_observation_type = "color"

    def __init__(self, aperture_size: int) -> None:
        self.aperture_size = aperture_size


class _FakeBuiltConfiguration:
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.configuration_sha256 = runner._WORKER_ENVELOPE_SHA256_BY_CANDIDATE[candidate_id]
        self.source_descriptor_sha256 = runner._LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        self.builder_descriptor_sha256 = runner._LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "alberta.forager_matched_worker_configuration.v1",
            "implementation_kind": runner._IMPLEMENTATION_KIND_BY_CANDIDATE[self.candidate_id],
            "configuration": {"fake_candidate_id": self.candidate_id},
        }


def _runtime_dependencies(
    state: FakeState,
    *,
    source_overrides: dict[str, str] | None = None,
) -> Any:
    def build(candidate_id: str) -> _FakeBuiltConfiguration:
        state.builder_calls.append(candidate_id)
        return _FakeBuiltConfiguration(candidate_id)

    def parse_agent(kind: str, payload: MappingLike) -> SimpleNamespace:
        state.parser_calls.append((kind, dict(payload)))
        return SimpleNamespace(kind=kind, payload=dict(payload))

    class FakeEnvConfig:
        @classmethod
        def paper_field_of_view(cls, *, aperture_size: int) -> _FakeEnvironment:
            del cls
            state.env_apertures.append(aperture_size)
            return _FakeEnvironment(aperture_size)

    class FakeBenchmarkConfig:
        def __init__(self, **kwargs: Any) -> None:
            state.benchmark_kwargs.append(dict(kwargs))
            self.__dict__.update(kwargs)

    def execute(
        agent_configuration: Any,
        benchmark: Any,
        seeds: tuple[int, ...],
        *,
        agent_seeds: tuple[int, ...],
        mode: str,
        reward_trace_sink_factory: Any,
    ) -> tuple[SimpleNamespace, ...]:
        state.runner_calls.append(
            {
                "agent_configuration": agent_configuration,
                "benchmark": benchmark,
                "seeds": seeds,
                "agent_seeds": agent_seeds,
                "mode": mode,
            }
        )
        if state.behavior == "raise_before_sink":
            raise RuntimeError("synthetic runner failure")
        sink = reward_trace_sink_factory(seeds[0], benchmark.steps)
        if state.behavior == "bad_reward":
            sink.append(
                np.asarray([2.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
            )
            raise AssertionError("bad reward was accepted")
        if state.behavior == "incomplete":
            sink.append(
                np.asarray([-1.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
            )
            sink.finalize()
            raise AssertionError("incomplete trace was finalized")
        base = np.asarray([-1.0, 0.0, 1.0, 30.0], dtype=np.float32)
        chunk_size = 8_192
        completed = 0
        while completed < benchmark.steps:
            active = min(chunk_size, benchmark.steps - completed)
            rewards = np.resize(base, active).astype(np.float32, copy=False)
            regrets = np.zeros((active,), dtype=np.float32)
            sink.append(rewards, regrets)
            completed += active
        metadata = sink.finalize()
        assert metadata["count"] == benchmark.steps
        result_steps = benchmark.steps - 1 if state.behavior == "bad_result" else benchmark.steps
        return (
            SimpleNamespace(
                seed=seeds[0],
                steps=result_steps,
                total_reward=float(_EXPECTED_TOTAL),
                agent="fake_local_agent",
            ),
        )

    source_hashes = dict(runner._PINNED_SOURCE_SHA256)
    source_hashes["local_runner_observed"] = cast(str, runner._BOOTSTRAP_SOURCE_SHA256)
    if source_overrides is not None:
        source_hashes.update(source_overrides)
    forager = SimpleNamespace(
        ForagerEnvConfig=FakeEnvConfig,
        ForagerBenchmarkConfig=FakeBenchmarkConfig,
        run_alberta_forager_seeds=execute,
        run_rtu_rtrl_forager_seeds=execute,
    )
    causal = SimpleNamespace(run_causal_map_forager_seeds=execute)
    return runner._RuntimeDependencies(
        numpy=np,
        local_configuration=SimpleNamespace(build_matched_v3_local_configuration=build),
        worker=SimpleNamespace(_parse_agent_configuration=parse_agent),
        forager=forager,
        causal_map_forager=causal,
        source_sha256_by_id=source_hashes,
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    behavior: str = "success",
    source_overrides: dict[str, str] | None = None,
) -> FakeState:
    state = FakeState(behavior=behavior)
    dependencies = _runtime_dependencies(state, source_overrides=source_overrides)
    monkeypatch.setattr(runner, "_load_runtime_dependencies", lambda: dependencies)
    monkeypatch.setattr(runner, "_validate_runtime_import_transition", lambda: None)
    observed = dependencies.source_sha256_by_id["local_runner_observed"]
    monkeypatch.setattr(runner, "_current_runner_source_sha256", lambda: observed)
    return state


def _execute(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_id: str = "causal_e025_q050",
    environment_seed: int = 7,
    agent_seed: int = 11,
    behavior: str = "success",
) -> tuple[object, FakeState, object]:
    state = _install_fake_runtime(monkeypatch, behavior=behavior)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    outcome = runner.run_matched_v3_local_candidate(
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        explicit_execution_opt_in=True,
        execution_capability=capability,
    )
    return outcome, state, capability


def _consume_outcome(outcome: object) -> Any:
    return runner.consume_matched_v3_local_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )


def test_descriptor_has_literal_canonical_identity_and_replays() -> None:
    raw = runner.canonical_matched_v3_local_runner_descriptor_bytes()
    assert len(raw) == 9_162
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert runner.LOCAL_RUNNER_DESCRIPTOR_SHA256 == _DESCRIPTOR_SHA256
    assert runner.matched_v3_local_runner_descriptor_sha256() == _DESCRIPTOR_SHA256
    assert runner.parse_matched_v3_local_runner_descriptor(raw) == (
        runner.matched_v3_local_runner_descriptor()
    )


def test_bounded_source_hash_accepts_exact_concrete_regular_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    payload = b"VALUE = 1\n"
    source.write_bytes(payload)

    assert type(source) is type(Path())
    assert runner._bounded_source_sha256(source) == hashlib.sha256(payload).hexdigest()


def test_bounded_source_hash_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    symbolic = tmp_path / "symbolic.py"
    symbolic.symlink_to(source)
    hard = tmp_path / "hard.py"
    os.link(source, hard)

    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        runner._bounded_source_sha256(symbolic)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="single-link"):
        runner._bounded_source_sha256(source)


def test_declared_relevant_source_pins_match_current_single_link_files() -> None:
    source_paths = runner._runtime_source_paths()

    assert {
        source_id: runner._bounded_source_sha256(source_paths[source_id])
        for source_id in runner._PINNED_SOURCE_SHA256
    } == dict(runner._PINNED_SOURCE_SHA256)


def test_lazy_loader_hashes_strict_scorer_but_never_imports_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_paths = {
        source_id: tmp_path / f"{source_id}.py"
        for source_id in (*runner._PINNED_SOURCE_SHA256, "local_runner_observed")
    }
    source_id_by_path = {path: source_id for source_id, path in source_paths.items()}
    hash_calls: list[str] = []
    import_calls: list[str] = []

    def fake_hash(path: Path) -> str:
        source_id = source_id_by_path[path]
        hash_calls.append(source_id)
        if source_id == "local_runner_observed":
            return cast(str, runner._BOOTSTRAP_SOURCE_SHA256)
        return cast(str, runner._PINNED_SOURCE_SHA256[source_id])

    def fake_import(module_name: str) -> SimpleNamespace:
        import_calls.append(module_name)
        if module_name.endswith("._forager_matched_v3_scorer"):
            raise AssertionError("strict scalar scorer must not be imported")
        return SimpleNamespace(__file__=str(tmp_path / f"{module_name}.py"))

    transition = runner._RuntimeImportTransition(
        pid=os.getpid(), status="loading", module_bindings=()
    )

    def finish_transition(exact_transition: Any, required_modules: dict[str, object]) -> None:
        assert exact_transition is transition
        assert "numpy" in required_modules
        transition.status = "loaded"

    monkeypatch.setattr(runner, "_runtime_source_paths", lambda: source_paths)
    monkeypatch.setattr(runner, "_bounded_source_sha256", fake_hash)
    monkeypatch.setattr(runner, "_verify_imported_module_path", lambda *_args: None)
    monkeypatch.setattr(runner, "_begin_runtime_import_transition", lambda: transition)
    monkeypatch.setattr(runner, "_finish_runtime_import_transition", finish_transition)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    dependencies = runner._load_runtime_dependencies()

    assert hash_calls.count("strict_reward_scorer") == 2
    assert hash_calls.count("local_runner_observed") == 2
    assert dependencies.source_sha256_by_id["local_runner_observed"] == (
        runner._BOOTSTRAP_SOURCE_SHA256
    )
    assert dependencies.source_sha256_by_id["strict_reward_scorer"] == (
        "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert not any(name.endswith("._forager_matched_v3_scorer") for name in import_calls)
    assert import_calls[-1] == "numpy"


def test_descriptor_binds_exact_14_ordered_configurations_and_sources() -> None:
    descriptor = runner.matched_v3_local_runner_descriptor()
    bindings = descriptor["candidate_bindings"]
    assert [item["candidate_id"] for item in bindings] == list(
        runner.MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS
    )
    assert len(bindings) == 14
    assert len({item["configuration_record_sha256"] for item in bindings}) == 14
    assert len({item["worker_envelope_sha256"] for item in bindings}) == 14
    assert descriptor["configuration_plan"]["sha256"] == (
        "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
    )
    assert descriptor["metric"]["sha256"] == (
        "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
    )
    assert descriptor["metric"]["strict_scorer_source_sha256"] == (
        "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert descriptor["relevant_source_sha256"] == dict(runner._PINNED_SOURCE_SHA256)


def test_descriptor_task_sink_capability_and_claims_are_fail_closed() -> None:
    descriptor = runner.matched_v3_local_runner_descriptor()
    assert descriptor["task"] == {
        "preset": "field_of_view",
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "observation_type": "color",
        "aperture_size": 9,
        "horizon": _HORIZON,
        "reward_values": [-1, 0, 1, 30],
    }
    assert descriptor["reward_sink"]["maximum_payload_bytes"] == _HORIZON
    assert descriptor["reward_sink"]["full_transition_retention"] is False
    assert descriptor["reward_sink"]["filesystem_artifact_written"] is False
    assert descriptor["source_closure"] == {
        "checked_relevant_pin_count": 7,
        "checked_scope": "declared_relevant_source_subset_only",
        "complete_local_source_closure": False,
        "complete_distribution_source_closure": False,
        "complete_closure_required_externally": True,
        "qualified": False,
    }
    assert (
        descriptor["execution_capability"]["isolated_top_level_direct_file_load_required"] is True
    )
    assert descriptor["execution_capability"]["isolated_top_level_module_name"] == (
        _RUNNER_MODULE_NAME
    )
    assert descriptor["execution_capability"]["bootstrap_injected_source_sha256_required"] is True
    assert descriptor["execution_capability"]["bootstrap_loader_contract"] == (
        "read_hash_compile_exec_exact_bytes_v1"
    )
    assert descriptor["execution_capability"]["plain_spec_loader_grants_capability"] is False
    assert descriptor["execution_capability"]["normal_package_import_grants_capability"] is False
    assert (
        descriptor["execution_capability"]["forbidden_prefixes_rechecked_at_capability_issue"]
        is True
    )
    assert (
        descriptor["execution_capability"][
            "forbidden_prefixes_rechecked_at_pre_import_run_boundary"
        ]
        is True
    )
    assert descriptor["execution_capability"]["pre_import_boundary_failure_consumes_capability"]
    assert descriptor["execution_capability"]["runner_owned_import_transition_required"]
    assert descriptor["execution_capability"][
        "post_import_exact_module_identity_validation_required"
    ]
    assert descriptor["execution_capability"]["preloaded_module_prefixes_rejected"] == [
        "alberta_framework",
        "chex",
        "foragax",
        "jax",
        "jaxlib",
        "ml_dtypes",
        "numpy",
        "scipy",
    ]
    assert descriptor["execution_capability"]["weakly_registered"] is True
    assert descriptor["execution_capability"]["pid_bound"] is True
    assert descriptor["execution_capability"]["single_use"] is True
    assert descriptor["execution_capability"]["serializable"] is False
    assert descriptor["outcome_capability"]["weakly_registered"] is True
    assert descriptor["outcome_capability"]["pid_bound"] is True
    assert descriptor["outcome_capability"]["single_use_content_access"] is True
    assert descriptor["outcome_capability"]["serializable"] is False
    assert descriptor["import_contract"] == {
        "scope": "exact_isolated_top_level_direct_file_load_only",
        "bootstrap_source_identity_required": True,
        "plain_spec_loader_execution_eligible": False,
        "normal_package_import_execution_eligible": False,
        "normal_package_import_may_preload_parent_dependencies": True,
        "preloaded_at_scoped_load_allowed": False,
        "preloaded_at_capability_issue_allowed": False,
        "preloaded_at_pre_import_run_boundary_allowed": False,
        "post_import_dependencies_allowed_only_after_runner_owned_transition": True,
        "jax_imported_by_scoped_load": False,
        "numpy_imported_by_scoped_load": False,
        "forager_runner_imported_by_scoped_load": False,
        "filesystem_inspection_by_scoped_load": False,
        "subprocess_by_scoped_load": False,
        "workload_by_scoped_load": False,
    }
    assert all(value is False for value in descriptor["claims"].values())


def test_clean_exact_top_level_direct_file_load_can_issue_in_subprocess() -> None:
    script = r"""
import importlib.util
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
capability = module.issue_matched_v3_local_execution_capability(
    explicit_execution_opt_in=True
)
print(json.dumps({
    "isolated": module._ISOLATED_TOP_LEVEL_LOAD,
    "preloaded": list(module._PRELOADED_EXECUTION_MODULES),
    "capability": repr(capability),
    "bootstrap_sha256": module._BOOTSTRAP_SOURCE_SHA256,
    "compiled_source_sha256": source_sha256,
    "jax": "jax" in sys.modules,
    "numpy": "numpy" in sys.modules,
    "alberta": "alberta_framework" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    expected_source_sha256 = hashlib.sha256(_RUNNER_MODULE_PATH.read_bytes()).hexdigest()
    assert payload == {
        "isolated": True,
        "preloaded": [],
        "capability": "<matched-v3 local execution capability>",
        "bootstrap_sha256": expected_source_sha256,
        "compiled_source_sha256": expected_source_sha256,
        "jax": False,
        "numpy": False,
        "alberta": False,
    }


def test_exact_prefix_inserted_after_clean_load_rejects_capability_issue() -> None:
    script = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
sys.modules["alberta_framework"] = types.ModuleType("alberta_framework")
error = None
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({
    "error": error,
    "capability_count": len(module._CAPABILITIES),
    "captured_at_load": list(module._PRELOADED_EXECUTION_MODULES),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "error": (
            "capability issuance rejects preloaded runtime dependencies or a prior "
            "runtime import transition: alberta_framework"
        ),
        "capability_count": 0,
        "captured_at_load": [],
    }


def test_exact_prefix_inserted_after_issue_is_rejected_and_consumes_capability() -> None:
    script = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
capability = module.issue_matched_v3_local_execution_capability(
    explicit_execution_opt_in=True
)
load_calls = []
module._load_runtime_dependencies = lambda: load_calls.append("reached")
sys.modules["numpy"] = types.ModuleType("numpy")
error = None
try:
    module.run_matched_v3_local_candidate(
        candidate_id="causal_e025_q050",
        environment_seed=1,
        agent_seed=2,
        explicit_execution_opt_in=True,
        execution_capability=capability,
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({
    "error": error,
    "status": module._CAPABILITIES[capability].status,
    "load_calls": load_calls,
    "captured_at_load": list(module._PRELOADED_EXECUTION_MODULES),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "error": (
            "pre-import run boundary rejects preloaded runtime dependencies or a prior "
            "runtime import transition: numpy"
        ),
        "status": "consumed",
        "load_calls": [],
        "captured_at_load": [],
    }


@pytest.mark.parametrize("stage", ["load", "issue", "run"])
def test_nonexact_string_module_key_cannot_bypass_load_or_live_prefix_checks(stage: str) -> None:
    script = r"""
import hashlib
import json
import pathlib
import sys
import types

class StringKey(str):
    pass

path = pathlib.Path(sys.argv[1])
stage = sys.argv[2]
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
module = types.ModuleType(name)
module.__file__ = str(path)
module.__package__ = ""
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
if stage == "load":
    sys.modules[StringKey("jax")] = types.ModuleType("jax")
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
capability = None
if stage == "run":
    capability = module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
if stage != "load":
    sys.modules[StringKey("jax")] = types.ModuleType("jax")
load_calls = []
module._load_runtime_dependencies = lambda: load_calls.append("reached")
error = None
try:
    if stage in {"load", "issue"}:
        module.issue_matched_v3_local_execution_capability(
            explicit_execution_opt_in=True
        )
    else:
        module.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({
    "error": error,
    "status": None if capability is None else module._CAPABILITIES[capability].status,
    "load_calls": load_calls,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH), stage],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if stage == "load":
        assert "exact isolated top-level" in payload["error"]
    else:
        assert payload["error"] == "runtime module registry contains a non-exact-string key"
    assert payload["status"] == ("consumed" if stage == "run" else None)
    assert payload["load_calls"] == []


def test_post_import_module_binding_drift_poisons_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    replacement = object()
    transition = runner._RuntimeImportTransition(
        pid=os.getpid(),
        status="loaded",
        module_bindings=(("jax", original),),
    )
    monkeypatch.setattr(runner, "_RUNTIME_IMPORT_TRANSITION", transition)
    monkeypatch.setattr(
        runner,
        "_current_execution_module_bindings",
        lambda: (("jax", replacement),),
    )

    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="identities drifted"):
        runner._validate_runtime_import_transition()
    assert transition.status == "poisoned"


def test_post_import_registry_observation_failure_poisons_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = runner._RuntimeImportTransition(
        pid=os.getpid(),
        status="loaded",
        module_bindings=(("jax", object()),),
    )
    monkeypatch.setattr(runner, "_RUNTIME_IMPORT_TRANSITION", transition)

    def fail_observation() -> tuple[tuple[str, object], ...]:
        raise runner.ForagerMatchedV3LocalRunnerError("synthetic registry anomaly")

    monkeypatch.setattr(runner, "_current_execution_module_bindings", fail_observation)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="registry anomaly"):
        runner._validate_runtime_import_transition()
    assert transition.status == "poisoned"


def test_normal_package_import_rejects_issue_and_run_in_subprocess() -> None:
    script = r"""
import importlib
import json
import sys

module = importlib.import_module(
    "alberta_framework.benchmarks.forager_matched_v3_local_runner"
)
errors = []
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    errors.append(str(exc))
try:
    module.run_matched_v3_local_candidate(
        candidate_id="causal_e025_q050",
        environment_seed=1,
        agent_seed=2,
        explicit_execution_opt_in=True,
        execution_capability=object(),
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    errors.append(str(exc))
print(json.dumps({
    "isolated": module._ISOLATED_TOP_LEVEL_LOAD,
    "errors": errors,
    "jax": "jax" in sys.modules,
    "numpy": "numpy" in sys.modules,
    "forager": "alberta_framework.benchmarks.forager" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_RUNNER_MODULE_PATH.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["isolated"] is False
    assert len(payload["errors"]) == 2
    assert all("isolated top-level" in message for message in payload["errors"])
    assert payload["jax"] is True
    assert payload["numpy"] is True
    assert payload["forager"] is True


def test_preloaded_dependencies_reject_exact_name_direct_load_in_subprocess() -> None:
    script = r"""
import importlib.util
import hashlib
import json
import pathlib
import sys

import alberta_framework
import numpy

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
error = None
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({
    "isolated": module._ISOLATED_TOP_LEVEL_LOAD,
    "preloaded": list(module._PRELOADED_EXECUTION_MODULES),
    "error": error,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(_RUNNER_MODULE_PATH)],
        cwd=_RUNNER_MODULE_PATH.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["isolated"] is False
    assert payload["error"] is not None
    assert "preloaded runtime dependencies" in payload["error"]
    assert "numpy" in payload["preloaded"]
    assert any(name.startswith("alberta_framework") for name in payload["preloaded"])


def test_plain_spec_loader_without_bootstrap_binding_cannot_issue() -> None:
    script = r"""
import importlib.util
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
error = None
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({
    "name_boundary": module._ISOLATED_TOP_LEVEL_NAME_BOUNDARY,
    "isolated": module._ISOLATED_TOP_LEVEL_LOAD,
    "bootstrap": module._BOOTSTRAP_SOURCE_SHA256,
    "error": error,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["name_boundary"] is True
    assert payload["isolated"] is False
    assert payload["bootstrap"] is None
    assert "bootstrap-injected SHA-256" in payload["error"]


def test_forged_bootstrap_source_binding_cannot_issue() -> None:
    script = r"""
import importlib.util
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = "0" * 64
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
error = None
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({"isolated": module._ISOLATED_TOP_LEVEL_LOAD, "error": error}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_RUNNER_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["isolated"] is True
    assert "nonzero" in payload["error"]


def test_stale_bootstrap_source_binding_cannot_issue(tmp_path: Path) -> None:
    copied_runner = tmp_path / "forager_matched_v3_local_runner.py"
    copied_runner.write_bytes(_RUNNER_MODULE_PATH.read_bytes())
    script = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
source_bytes = path.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
module.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = source_sha256
sys.modules[name] = module
exec(compile(source_bytes, str(path), "exec"), module.__dict__)
path.write_bytes(source_bytes + b"# post-load drift\n")
error = None
try:
    module.issue_matched_v3_local_execution_capability(
        explicit_execution_opt_in=True
    )
except module.ForagerMatchedV3LocalRunnerError as exc:
    error = str(exc)
print(json.dumps({"isolated": module._ISOLATED_TOP_LEVEL_LOAD, "error": error}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(copied_runner)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["isolated"] is True
    assert "stale or forged" in payload["error"]


def test_descriptor_snapshot_is_detached_and_mutation_is_rejected() -> None:
    first = runner.matched_v3_local_runner_descriptor()
    first["claims"]["performance_claim_allowed"] = True
    assert (
        runner.matched_v3_local_runner_descriptor()["claims"]["performance_claim_allowed"] is False
    )
    raw = _canonical(first)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        runner.parse_matched_v3_local_runner_descriptor(raw)


@pytest.mark.parametrize("opt_in", [False, 0, 1, None])
def test_capability_issue_requires_exact_explicit_true(opt_in: object) -> None:
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        cast(Any, runner.issue_matched_v3_local_execution_capability)(
            explicit_execution_opt_in=opt_in
        )


def test_forged_capability_is_rejected_before_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_runtime_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("lazy import reached")),
    )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="authentic"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=object(),
        )


def test_run_opt_in_false_does_not_execute_or_consume_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_runtime(monkeypatch)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="run-time opt-in"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=False,
            execution_capability=capability,
        )
    outcome = runner.run_matched_v3_local_candidate(
        candidate_id="causal_e025_q050",
        environment_seed=1,
        agent_seed=2,
        explicit_execution_opt_in=True,
        execution_capability=capability,
    )
    completion = _consume_outcome(outcome)
    assert completion.environment_seed == 1
    assert len(state.runner_calls) == 1


def test_capability_is_single_use_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _state, capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    assert len(completion.reward_trace) == _HORIZON
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=7,
            agent_seed=11,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


def test_capability_is_pid_bound_and_poisoned_on_fork_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="PID"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    monkeypatch.setattr(os, "getpid", lambda: parent_pid)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


def test_capability_is_weak_opaque_uncopyable_and_unserializable() -> None:
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    assert repr(capability) == "<matched-v3 local execution capability>"
    import copy as copy_module

    with pytest.raises(TypeError):
        copy_module.copy(capability)
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    reference = weakref.ref(capability)
    assert capability in runner._CAPABILITIES
    del capability
    gc.collect()
    assert reference() is None


def test_success_returns_only_opaque_weak_outcome_bound_to_exact_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, execution_capability = _execute(monkeypatch)
    assert repr(outcome) == "<matched-v3 local outcome capability>"
    assert not hasattr(outcome, "reward_trace")
    assert not hasattr(outcome, "canonical_receipt_bytes")
    state = runner._OUTCOMES[cast(Any, outcome)]
    assert state.status == "live"
    assert state.execution_capability is execution_capability
    assert state.execution_capability_identity == id(execution_capability)
    assert state.trace_sha256 == hashlib.sha256(_EXPECTED_TRACE).hexdigest()
    assert state.receipt_sha256 == state.completion.receipt_sha256
    assert state.runner_source_sha256 == runner._BOOTSTRAP_SOURCE_SHA256


def test_outcome_is_uncopyable_unserializable_and_not_dataclass_replaceable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    import copy as copy_module

    with pytest.raises(TypeError):
        copy_module.copy(outcome)
    with pytest.raises(TypeError):
        copy_module.deepcopy(outcome)
    with pytest.raises(TypeError):
        pickle.dumps(outcome)
    with pytest.raises(TypeError):
        dataclasses.replace(cast(Any, outcome))


def test_forged_outcome_cannot_expose_structural_content() -> None:
    forged = runner._LocalOutcomeCapability()
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="authentic"):
        runner.consume_matched_v3_local_outcome(
            outcome_capability=object(),
            explicit_content_access_opt_in=True,
        )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="stale"):
        runner.consume_matched_v3_local_outcome(
            outcome_capability=forged,
            explicit_content_access_opt_in=True,
        )


def test_outcome_content_opt_in_false_does_not_consume_live_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="explicit opt-in"):
        runner.consume_matched_v3_local_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=False,
        )
    assert len(_consume_outcome(outcome).reward_trace) == _HORIZON


def test_outcome_content_access_is_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    assert completion.receipt_sha256
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        _consume_outcome(outcome)


def test_outcome_is_pid_bound_and_fork_attempt_permanently_stales_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid)
    outcome, _state, _capability = _execute(monkeypatch)
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="PID"):
        _consume_outcome(outcome)
    monkeypatch.setattr(os, "getpid", lambda: parent_pid)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        _consume_outcome(outcome)


def test_outcome_source_drift_stales_handle_before_content_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    monkeypatch.setattr(runner, "_current_runner_source_sha256", lambda: _sha("drifted"))
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="source is stale"):
        _consume_outcome(outcome)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        _consume_outcome(outcome)


def test_outcome_rejects_lost_execution_capability_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, execution_capability = _execute(monkeypatch)
    del runner._CAPABILITIES[cast(Any, execution_capability)]
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="binding"):
        _consume_outcome(outcome)


@pytest.mark.parametrize("field", ["trace_sha256", "receipt_sha256"])
def test_outcome_digest_binding_tamper_stales_handle(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    outcome_state = runner._OUTCOMES[cast(Any, outcome)]
    setattr(outcome_state, field, "0" * 64)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="nonzero"):
        _consume_outcome(outcome)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        _consume_outcome(outcome)


def test_outcome_registry_is_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    reference = weakref.ref(outcome)
    assert outcome in runner._OUTCOMES
    del outcome
    gc.collect()
    assert reference() is None


def test_invalid_candidate_consumes_capability_before_any_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_runtime_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("lazy import reached")),
    )
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="unknown"):
        runner.run_matched_v3_local_candidate(
            candidate_id="unknown",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


@pytest.mark.parametrize("field", ["environment_seed", "agent_seed"])
@pytest.mark.parametrize("bad_seed", [True, -1, 2**31, 1.0])
def test_seed_boundary_is_exact_uint31_and_failure_poisoned(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_seed: object,
) -> None:
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    kwargs: dict[str, Any] = {
        "candidate_id": "causal_e025_q050",
        "environment_seed": 1,
        "agent_seed": 2,
        "explicit_execution_opt_in": True,
        "execution_capability": capability,
    }
    kwargs[field] = bad_seed
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="uint31"):
        cast(Any, runner.run_matched_v3_local_candidate)(**kwargs)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


def test_runner_source_drift_after_issue_is_rejected_after_capability_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_runtime(monkeypatch)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    monkeypatch.setattr(runner, "_current_runner_source_sha256", lambda: _sha("drifted"))

    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="stale or forged"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    assert state.builder_calls == []
    monkeypatch.setattr(
        runner,
        "_current_runner_source_sha256",
        lambda: runner._BOOTSTRAP_SOURCE_SHA256,
    )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


def test_runner_source_drift_during_execution_fails_before_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_runtime(monkeypatch)
    expected = cast(str, runner._BOOTSTRAP_SOURCE_SHA256)
    current_calls = 0

    def current_source() -> str:
        nonlocal current_calls
        current_calls += 1
        return expected if current_calls <= 3 else _sha("post-run-drift")

    monkeypatch.setattr(runner, "_current_runner_source_sha256", current_source)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)

    with pytest.raises(
        runner.ForagerMatchedV3LocalRunnerError,
        match="source drifted during candidate execution",
    ):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    assert current_calls == 4
    assert state.builder_calls == ["causal_e025_q050"]


def test_distinct_seed_transport_reaches_underlying_runner_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, state, _capability = _execute(monkeypatch, environment_seed=123, agent_seed=456)
    completion = _consume_outcome(outcome)
    assert state.runner_calls[0]["seeds"] == (123,)
    assert state.runner_calls[0]["agent_seeds"] == (456,)
    assert state.runner_calls[0]["mode"] == "strict"
    receipt = completion.receipt()
    assert receipt["seed_transport"]["environment_agent_seed_collision"] is False


def test_equal_environment_and_agent_seed_is_allowed_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, state, _capability = _execute(
        monkeypatch, environment_seed=2_147_483_647, agent_seed=2_147_483_647
    )
    completion = _consume_outcome(outcome)
    assert state.runner_calls[0]["seeds"] == (2_147_483_647,)
    assert state.runner_calls[0]["agent_seeds"] == (2_147_483_647,)
    receipt = completion.receipt()
    assert receipt["seed_transport"]["environment_agent_seed_collision"] is True
    assert receipt["seed_transport"]["environment_agent_seed_collisions_allowed"] is True


@pytest.mark.parametrize(
    ("candidate_id", "kind"),
    [
        ("causal_e025_q050", "alberta_causal_map"),
        ("alberta_horde_default", "alberta_horde_actor_critic"),
        ("alberta_rtu_h08_taylor", "alberta_rtu_rtrl"),
    ],
)
def test_all_three_local_families_route_through_exact_typed_configuration(
    monkeypatch: pytest.MonkeyPatch,
    candidate_id: str,
    kind: str,
) -> None:
    outcome, state, _capability = _execute(monkeypatch, candidate_id=candidate_id)
    completion = _consume_outcome(outcome)
    assert state.builder_calls == [candidate_id]
    assert state.parser_calls == [(kind, {"fake_candidate_id": candidate_id})]
    assert completion.receipt()["candidate"]["implementation_kind"] == kind


def test_exact_field_of_view_task_and_horizon_are_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, state, _capability = _execute(monkeypatch, environment_seed=99)
    _consume_outcome(outcome)
    assert state.env_apertures == [9]
    benchmark = state.benchmark_kwargs[0]
    assert benchmark["steps"] == _HORIZON
    assert benchmark["seed"] == 99
    assert benchmark["environment"].resolved_env_id == "ForagaxTwoBiomeLarge-v1"
    assert benchmark["environment"].resolved_observation_type == "color"
    assert benchmark["ewm_decay"] == 0.999
    assert benchmark["record_every"] == 100


def test_exact_reward_sequence_count_digest_and_nontransition_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    assert type(completion.reward_trace) is bytes
    assert len(completion.reward_trace) == _HORIZON
    assert completion.reward_trace == _EXPECTED_TRACE
    receipt = completion.receipt()
    reward = receipt["reward_trace"]
    assert reward["count"] == _HORIZON
    assert reward["size_bytes"] == _HORIZON
    assert reward["sha256"] == hashlib.sha256(_EXPECTED_TRACE).hexdigest()
    assert reward["cumulative_reward"] == _EXPECTED_TOTAL
    assert reward["allowed_values"] == [-1, 0, 1, 30]
    assert reward["full_transition_retention"] is False
    assert "observations" not in receipt and "actions" not in receipt


def test_completion_receipt_replays_with_external_digest_and_is_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    parsed = runner.parse_matched_v3_local_completion_receipt(
        completion.canonical_receipt_bytes,
        reward_trace=completion.reward_trace,
        expected_receipt_sha256=completion.receipt_sha256,
    )
    assert (
        parsed["bindings"]["cumulative_reward_metric"]["strict_scorer_source_sha256"]
        == "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert (
        parsed["bindings"]["relevant_source_sha256"]["strict_reward_scorer"]
        == "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert (
        parsed["bindings"]["relevant_source_sha256"]["local_runner_observed"]
        == runner._BOOTSTRAP_SOURCE_SHA256
    )
    parsed["claims"]["performance_claim_allowed"] = True
    assert completion.receipt()["claims"]["performance_claim_allowed"] is False
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="nonzero"):
        runner.parse_matched_v3_local_completion_receipt(
            completion.canonical_receipt_bytes,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "alberta.forager_matched_v3.local_runner_completion.v2"),
        (("reward_trace", "count"), _HORIZON - 1),
        (("reward_trace", "sha256"), "0" * 64),
        (("bindings", "configuration_plan", "sha256"), "0" * 64),
        (("bindings", "cumulative_reward_metric", "sha256"), "0" * 64),
        (("bindings", "relevant_source_sha256", "forager"), "0" * 64),
        (("bindings", "relevant_source_sha256", "local_runner_observed"), "0" * 64),
    ],
)
def test_coherently_rehashed_receipt_semantic_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    receipt = completion.receipt()
    target: dict[str, Any] = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    raw = _rehash_receipt(receipt)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        runner.parse_matched_v3_local_completion_receipt(
            raw,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("underlying_result", "environment_seed"), False),
        (("reward_trace", "allowed_values"), [-1, False, True, 30]),
        (("reward_trace", "full_transition_retention"), 0),
    ],
)
def test_coherently_rehashed_receipt_rejects_bool_int_type_confusion(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch, environment_seed=0)
    completion = _consume_outcome(outcome)
    receipt = completion.receipt()
    target: dict[str, Any] = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    raw = _rehash_receipt(receipt)

    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        runner.parse_matched_v3_local_completion_receipt(
            raw,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_trace_length_and_reward_domain_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="length"):
        runner.parse_matched_v3_local_completion_receipt(
            completion.canonical_receipt_bytes,
            reward_trace=completion.reward_trace[:-1],
            expected_receipt_sha256=completion.receipt_sha256,
        )
    mutated = bytearray(completion.reward_trace)
    mutated[0] = 2
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="domain"):
        runner.parse_matched_v3_local_completion_receipt(
            completion.canonical_receipt_bytes,
            reward_trace=bytes(mutated),
            expected_receipt_sha256=completion.receipt_sha256,
        )


@pytest.mark.parametrize(
    "behavior",
    ["raise_before_sink", "bad_reward", "incomplete", "bad_result"],
)
def test_underlying_exception_or_contract_failure_poisons_capability(
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
) -> None:
    _install_fake_runtime(monkeypatch, behavior=behavior)
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="consumed"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )


def test_source_digest_drift_fails_before_builder_and_poisons_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_runtime(monkeypatch, source_overrides={"forager": _sha("drifted")})
    capability = runner.issue_matched_v3_local_execution_capability(explicit_execution_opt_in=True)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="source drifted"):
        runner.run_matched_v3_local_candidate(
            candidate_id="causal_e025_q050",
            environment_seed=1,
            agent_seed=2,
            explicit_execution_opt_in=True,
            execution_capability=capability,
        )
    assert state.builder_calls == []


def test_strict_parser_rejects_duplicate_noncanonical_and_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _state, _capability = _execute(monkeypatch)
    completion = _consume_outcome(outcome)
    raw = completion.canonical_receipt_bytes
    duplicate = b'{"schema_version":"duplicate",' + raw[1:]
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="duplicate"):
        runner.parse_matched_v3_local_completion_receipt(
            duplicate,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256=hashlib.sha256(duplicate).hexdigest(),
        )
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="canonical"):
        runner.parse_matched_v3_local_completion_receipt(
            b" " + raw,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256=hashlib.sha256(b" " + raw).hexdigest(),
        )
    nonfinite = raw.replace(b'"steps":499712', b'"steps":NaN', 1)
    with pytest.raises(runner.ForagerMatchedV3LocalRunnerError, match="non-finite"):
        runner.parse_matched_v3_local_completion_receipt(
            nonfinite,
            reward_trace=completion.reward_trace,
            expected_receipt_sha256=hashlib.sha256(nonfinite).hexdigest(),
        )


def test_importing_source_directly_performs_no_heavy_import_fs_pid_or_subprocess_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path(runner.__file__)
    original_import = builtins.__import__
    pid_calls = 0

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.split(".", 1)[0] in {"jax", "numpy", "foragax"}:
            raise AssertionError(f"heavy import at module execution: {name}")
        return original_import(name, globals, locals, fromlist, level)

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("filesystem or subprocess work at module execution")

    def counted_pid() -> int:
        nonlocal pid_calls
        pid_calls += 1
        return 123

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "getpid", counted_pid)
    module_name = "_matched_v3_local_runner_import_probe"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    assert pid_calls == 0


def test_public_execution_apis_have_no_defaults_or_production_plan() -> None:
    issue_signature = inspect.signature(runner.issue_matched_v3_local_execution_capability)
    run_signature = inspect.signature(runner.run_matched_v3_local_candidate)
    consume_signature = inspect.signature(runner.consume_matched_v3_local_outcome)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in issue_signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in run_signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in consume_signature.parameters.values()
    )
    assert not hasattr(runner, "DEFAULT_EXECUTION_CAPABILITY")
    assert not hasattr(runner, "PRODUCTION_EXECUTION_PLAN")
