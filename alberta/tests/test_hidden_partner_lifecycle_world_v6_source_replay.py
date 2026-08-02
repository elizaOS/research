"""Focused contracts for explicit v6 canonical-source replay verification."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation import hidden_partner_lifecycle_world_v6_source_replay as replay
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    build_v6_primary_controls,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    HiddenPartnerLifecycleWorldV6Runner,
    V6DevelopmentRun,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_source_replay import (
    SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
    SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN,
    SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN,
    compare_v6_development_runs_bit_exact,
    verify_v6_development_run_source_replay,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
    STRUCTURALLY_INVALID_DEVELOPMENT_RUN,
    STRUCTURALLY_VALID_DEVELOPMENT_RUN,
    V6DevelopmentRunValidation,
)

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True, slots=True)
class _TinyState:
    parameters: jax.Array
    host_value: float


@dataclasses.dataclass(frozen=True, slots=True)
class _TinyRng:
    supplied_key_data: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _TinyHostRecord:
    name: str
    values: tuple[int, ...]


def _member_validation(*, valid: bool = True) -> V6DevelopmentRunValidation:
    return V6DevelopmentRunValidation(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
        status=(
            STRUCTURALLY_VALID_DEVELOPMENT_RUN if valid else STRUCTURALLY_INVALID_DEVELOPMENT_RUN
        ),
        development_only=True,
        structural_only=True,
        replay_verified=False,
        execution_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        errors=(),
        lifecycle=cast(Any, None),
        coverage=cast(Any, None),
        quality=cast(Any, None),
    )


@pytest.fixture(autouse=True)
def _valid_structural_preflights(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.name == "test_real_source_replay_opt_in_large_memory":
        return
    verdict = _member_validation()
    monkeypatch.setattr(
        replay,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        lambda _run: verdict,
    )


def _run(
    *,
    parameter: float = 0.25,
    host_value: float = 0.0,
    control_name: str = "full",
    primary: bool = True,
) -> V6DevelopmentRun:
    state = _TinyState(
        parameters=jnp.asarray((parameter, -0.5), dtype=jnp.float32),
        host_value=host_value,
    )
    aggregate = _TinyHostRecord(name="aggregate", values=(1, 2, 3))
    return V6DevelopmentRun(
        control_name=control_name,
        primary=primary,
        plan=cast(Any, _TinyHostRecord(name="plan", values=(30_000,))),
        control_config_sha256="a" * 64,
        control_matrix_sha256="b" * 64,
        bridge_config_sha256="c" * 64,
        runner_config_sha256="d" * 64,
        source_closure_hashes=cast(
            Any,
            (_TinyHostRecord(name="source.py", values=(1,)),),
        ),
        runtime=cast(Any, _TinyHostRecord(name="runtime", values=(2,))),
        initial_state=cast(Any, state),
        final_state=cast(Any, state),
        windows=cast(Any, aggregate),
        row_heads=cast(Any, aggregate),
        filter_totals=cast(Any, aggregate),
        action_totals=cast(Any, aggregate),
        audits=cast(Any, aggregate),
        ledger=cast(Any, aggregate),
        lifecycle=cast(Any, aggregate),
        rng=cast(
            Any,
            _TinyRng(supplied_key_data=jnp.asarray(((11, 12), (21, 22)), dtype=jnp.uint32)),
        ),
        resources=cast(Any, _TinyHostRecord(name="resources", values=(3,))),
        stream_code=jnp.asarray((0, 1, 2), dtype=jnp.uint8),
    )


def _codes(differences: tuple[object, ...]) -> set[str]:
    return {difference.code for difference in differences}  # type: ignore[attr-defined]


def test_comparator_rejects_finite_parameter_tamper_and_signed_zero() -> None:
    baseline = _run()
    parameter_tamper = _run(parameter=0.375)
    signed_zero_array = _run(parameter=-0.0)
    positive_zero_array = _run(parameter=0.0)
    signed_zero_host = _run(host_value=-0.0)

    parameter_differences = compare_v6_development_runs_bit_exact(parameter_tamper, baseline)
    array_zero_differences = compare_v6_development_runs_bit_exact(
        signed_zero_array,
        positive_zero_array,
    )
    host_zero_differences = compare_v6_development_runs_bit_exact(signed_zero_host, baseline)

    assert "ARRAY_BITS" in _codes(parameter_differences)
    assert any(item.path == "run.initial_state.parameters" for item in parameter_differences)
    assert "ARRAY_BITS" in _codes(array_zero_differences)
    assert "HOST_FLOAT_BITS" in _codes(host_zero_differences)


def test_comparator_rejects_numpy_arrays_even_when_bytes_match() -> None:
    baseline = _run()
    numpy_stream = np.asarray((0, 1, 2), dtype=np.uint8)
    left = dataclasses.replace(baseline, stream_code=cast(Any, numpy_stream))
    right = dataclasses.replace(baseline, stream_code=cast(Any, numpy_stream.copy()))

    differences = compare_v6_development_runs_bit_exact(left, right)

    assert "NUMPY_ARRAY" in _codes(differences)
    assert any(item.path == "run.stream_code" for item in differences)


def test_invalid_candidate_preflight_avoids_runner_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    class ForbiddenRunner:
        def __init__(self, _control: object) -> None:
            nonlocal constructed
            constructed = True
            raise AssertionError("runner must not be constructed")

    monkeypatch.setattr(
        replay,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        lambda _run: _member_validation(valid=False),
    )
    monkeypatch.setattr(replay, "HiddenPartnerLifecycleWorldV6Runner", ForbiddenRunner)

    result = verify_v6_development_run_source_replay(object())

    assert result.status == SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN
    assert not result.replay_attempted
    assert not result.replay_executed
    assert not result.replay_verified
    assert not result.candidate_structurally_valid
    assert not constructed
    assert "CANDIDATE_PREFLIGHT" in _codes(result.differences)


def test_canonical_control_and_explicit_threefry_keys_drive_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _run()
    observed: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, control: object) -> None:
            observed["control"] = control

        def run(self, world_key: jax.Array, agent_key: jax.Array) -> V6DevelopmentRun:
            observed["world_key"] = world_key
            observed["agent_key"] = agent_key
            return candidate

    monkeypatch.setattr(replay, "HiddenPartnerLifecycleWorldV6Runner", FakeRunner)

    result = verify_v6_development_run_source_replay(candidate)

    assert result.status == SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN
    assert result.replay_attempted
    assert result.replay_executed
    assert result.replay_verified
    assert result.candidate_structurally_valid
    assert result.fresh_structurally_valid
    assert not result.structural_only
    assert not result.independent_learner_or_accumulator_oracle
    assert not result.evidence_authorized
    assert not result.scientific_promotion_allowed
    assert cast(Any, observed["control"]).name == "full"
    world_key = cast(jax.Array, observed["world_key"])
    agent_key = cast(jax.Array, observed["agent_key"])
    assert str(jr.key_impl(world_key)) == "threefry2x32"
    assert str(jr.key_impl(agent_key)) == "threefry2x32"
    np.testing.assert_array_equal(jr.key_data(world_key), np.asarray((11, 12), dtype=np.uint32))
    np.testing.assert_array_equal(jr.key_data(agent_key), np.asarray((21, 22), dtype=np.uint32))


def test_finite_candidate_mismatch_returns_path_addressed_replay_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _run(parameter=0.375)
    fresh = _run(parameter=0.25)

    class FakeRunner:
        def __init__(self, _control: object) -> None:
            pass

        def run(self, _world_key: jax.Array, _agent_key: jax.Array) -> V6DevelopmentRun:
            return fresh

    monkeypatch.setattr(replay, "HiddenPartnerLifecycleWorldV6Runner", FakeRunner)

    result = verify_v6_development_run_source_replay(candidate)

    assert result.status == SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN
    assert result.replay_executed
    assert not result.replay_verified
    assert result.candidate_structurally_valid
    assert result.fresh_structurally_valid
    assert any(item.path == "run.initial_state.parameters" for item in result.differences)


def test_fresh_run_structural_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _run()
    calls = 0

    def validate(_run: object) -> V6DevelopmentRunValidation:
        nonlocal calls
        calls += 1
        return _member_validation(valid=calls == 1)

    class FakeRunner:
        def __init__(self, _control: object) -> None:
            pass

        def run(self, _world_key: jax.Array, _agent_key: jax.Array) -> V6DevelopmentRun:
            return candidate

    monkeypatch.setattr(
        replay,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        validate,
    )
    monkeypatch.setattr(replay, "HiddenPartnerLifecycleWorldV6Runner", FakeRunner)

    result = verify_v6_development_run_source_replay(candidate)

    assert result.status == SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN
    assert result.replay_attempted
    assert result.replay_executed
    assert result.candidate_structurally_valid
    assert not result.fresh_structurally_valid
    assert not result.replay_verified
    assert "FRESH_PREFLIGHT" in _codes(result.differences)


def test_runner_exception_is_reported_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _run()

    class ExplodingRunner:
        def __init__(self, _control: object) -> None:
            pass

        def run(self, _world_key: jax.Array, _agent_key: jax.Array) -> V6DevelopmentRun:
            raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(replay, "HiddenPartnerLifecycleWorldV6Runner", ExplodingRunner)

    result = verify_v6_development_run_source_replay(candidate)

    assert result.status == SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN
    assert result.replay_attempted
    assert not result.replay_executed
    assert not result.replay_verified
    assert "REPLAY_EXECUTION" in _codes(result.differences)


@pytest.mark.development
@pytest.mark.skipif(
    os.environ.get("ALBERTA_RUN_V6_SOURCE_REPLAY") != "1",
    reason="set ALBERTA_RUN_V6_SOURCE_REPLAY=1 for the opt-in large-memory replay",
)
def test_real_source_replay_opt_in_large_memory() -> None:
    """Execute two full v6 scans only under an explicit development opt-in."""

    control = build_v6_primary_controls()[0]
    candidate = HiddenPartnerLifecycleWorldV6Runner(control).run(jr.key(97_001), jr.key(97_002))

    result = verify_v6_development_run_source_replay(candidate)

    assert result.status == SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN
    assert result.replay_verified
    assert result.differences == ()
