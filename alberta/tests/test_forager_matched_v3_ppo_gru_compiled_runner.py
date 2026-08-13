"""Fast fake-JAX checks for the additive matched-v3 compiled PPO-GRU runner."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru_compiled_runner as runner
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru_runner as v1_runner

_HIDDEN_SIZE = 4


class _FakeEnvironmentState(NamedTuple):
    time: Array
    reset_count: Array
    step_call_count: Array


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _observation(time: Array) -> Array:
    row = jnp.mod(time, jnp.int32(9))
    column = jnp.mod(time // jnp.int32(9), jnp.int32(9))
    channel = jnp.mod(time, jnp.int32(3))
    return jnp.zeros((9, 9, 3), dtype=jnp.float32).at[row, column, channel].set(1.0)


def _info() -> dict[str, Array]:
    return {
        "discount": jnp.asarray(1.0, dtype=jnp.float32),
        "temperatures": jnp.zeros((4,), dtype=jnp.float32),
        "biome_id": jnp.asarray(0, dtype=jnp.int16),
        "object_collected_id": jnp.asarray(-1, dtype=jnp.int32),
        "current_biome_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "max_biome_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "biome_regret": jnp.asarray(0.0, dtype=jnp.float32),
        "biome_rank": jnp.asarray(1, dtype=jnp.int32),
        "rewards": jnp.zeros((9, 9), dtype=jnp.float16),
    }


def _environment_step(
    key: Array,
    state: _FakeEnvironmentState,
    action: Array,
    params: object,
) -> tuple[Array, _FakeEnvironmentState, Array, Array, dict[str, Array]]:
    del params
    draw = jr.randint(key, (), 0, 4, dtype=jnp.int32)
    support = jnp.asarray((-1.0, 0.0, 1.0, 30.0), dtype=jnp.float32)
    reward = support[jnp.mod(action + draw, jnp.int32(4))]
    next_time = state.time + jnp.int32(1)
    next_state = _FakeEnvironmentState(
        time=next_time,
        reset_count=state.reset_count,
        step_call_count=state.step_call_count + jnp.int32(1),
    )
    return (
        _observation(next_time),
        next_state,
        reward,
        jnp.asarray(False, dtype=jnp.bool_),
        _info(),
    )


def _policy_apply(
    variables: object,
    carry: Array,
    observation: Array,
    reset_before: Array,
) -> tuple[Array, Array, Array]:
    del variables
    reset_carry = jnp.where(reset_before, jnp.zeros_like(carry), carry)
    signal = jnp.sum(observation[..., 0], dtype=jnp.float32)
    increment = jnp.asarray((0.125, 0.25, -0.125, 0.5), dtype=jnp.float32)
    outgoing = reset_carry + increment + signal * jnp.float32(0.01)
    logits = jnp.asarray((0.25, -0.5, 0.75, 0.0), dtype=jnp.float32) + outgoing
    value = jnp.sum(outgoing, dtype=jnp.float32) * jnp.float32(0.125)
    return outgoing, logits, value


def _keys(seed: int) -> tuple[Array, Array]:
    continuation, _consumed = jr.split(jr.key(seed, impl="threefry2x32"))
    return continuation, _consumed


def _initial_inputs(
    *, environment_seed: int = 17, agent_seed: int = 29
) -> tuple[_FakeEnvironmentState, Array, Array, Array, Array, Array]:
    environment_key, _reset_key = _keys(environment_seed)
    agent_key, _initialization_key = _keys(agent_seed)
    state = _FakeEnvironmentState(
        time=jnp.asarray(0, dtype=jnp.int32),
        reset_count=jnp.asarray(1, dtype=jnp.int32),
        step_call_count=jnp.asarray(0, dtype=jnp.int32),
    )
    return (
        state,
        _observation(state.time),
        environment_key,
        jnp.zeros((_HIDDEN_SIZE,), dtype=jnp.float32),
        agent_key,
        jnp.asarray(0, dtype=jnp.int32),
    )


def _kernel(
    *,
    chunk_steps: int,
    environment_step: Any = _environment_step,
    policy_apply: Any = _policy_apply,
) -> Any:
    return runner._build_chunk_kernel(
        environment_step=environment_step,
        environment_params=None,
        policy_apply=policy_apply,
        hidden_size=_HIDDEN_SIZE,
        chunk_steps=chunk_steps,
    )


def _scalar_rollout(
    *,
    steps: int,
    environment_state: _FakeEnvironmentState,
    observation: Array,
    environment_key: Array,
    carry: Array,
    agent_key: Array,
) -> dict[str, Any]:
    observations: list[Array] = []
    incoming_carries: list[Array] = []
    outgoing_carries: list[Array] = []
    action_key_words: list[Array] = []
    logits_trace: list[Array] = []
    actions: list[Array] = []
    old_log_probs: list[Array] = []
    old_values: list[Array] = []
    rewards: list[Array] = []
    next_observations: list[Array] = []
    for _ in range(steps):
        agent_key, action_key = jr.split(agent_key)
        outgoing, logits, value = _policy_apply(None, carry, observation, jnp.bool_(False))
        action = jr.categorical(
            action_key,
            logits,
            axis=-1,
            mode=ppo_gru.PPO_GRU_CATEGORICAL_MODE,
        ).astype(jnp.int32)
        log_prob = jnp.sum(
            jax.nn.log_softmax(logits) * jax.nn.one_hot(action, 4, dtype=jnp.float32)
        )
        environment_key, step_key = jr.split(environment_key)
        next_observation, environment_state, reward, done, info = _environment_step(
            step_key, environment_state, action, None
        )
        assert not bool(np.asarray(done))
        assert set(info) == runner._EXPECTED_INFO_KEYS
        observations.append(observation)
        incoming_carries.append(carry)
        outgoing_carries.append(outgoing)
        action_key_words.append(jr.key_data(action_key))
        logits_trace.append(logits)
        actions.append(action)
        old_log_probs.append(log_prob)
        old_values.append(value)
        rewards.append(reward)
        next_observations.append(next_observation)
        observation = next_observation
        carry = outgoing
    _bootstrap_carry, _bootstrap_logits, bootstrap_value = _policy_apply(
        None, carry, observation, jnp.bool_(False)
    )
    return {
        "environment_state": environment_state,
        "observation": observation,
        "environment_key": environment_key,
        "gru_carry": carry,
        "agent_key": agent_key,
        "observations": jnp.stack(observations),
        "incoming_carries": jnp.stack(incoming_carries),
        "outgoing_carries": jnp.stack(outgoing_carries),
        "action_key_words": jnp.stack(action_key_words),
        "logits": jnp.stack(logits_trace),
        "actions": jnp.stack(actions),
        "old_log_probs": jnp.stack(old_log_probs),
        "old_values": jnp.stack(old_values),
        "raw_rewards": jnp.stack(rewards),
        "next_observations": jnp.stack(next_observations),
        "bootstrap_value": bootstrap_value,
    }


def _assert_array_equal(first: object, second: object) -> None:
    def comparable(value: object) -> np.ndarray[Any, Any]:
        try:
            return cast(np.ndarray[Any, Any], np.asarray(value))
        except TypeError:
            return cast(np.ndarray[Any, Any], np.asarray(jr.key_data(cast(Any, value))))

    np.testing.assert_array_equal(comparable(first), comparable(second))


def _fake_bridge_runtime() -> bridge.MatchedV3ForagaxRuntime:
    identity = bridge.MatchedV3ForagaxRuntimeIdentity(
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        default_prng_impl="threefry2x32",
        threefry_partitionable=True,
        jax_enable_x64=False,
        backend="cpu",
        foragax_version="0.55.0",
        foragax_install_tree_sha256=bridge.FORAGAX_INSTALL_TREE_SHA256,
        foragax_package_root="/synthetic/foragax",
        runtime_qualified=False,
    )
    return bridge.MatchedV3ForagaxRuntime(
        runtime_identity=identity,
        _environment=object(),
        _params=object(),
        _capability=bridge._RuntimeCapability(),
    )


def _runtime_identity_bytes(runtime: bridge.MatchedV3ForagaxRuntime) -> bytes:
    return _canonical(runner._runtime_identity(runtime))


def _receipt() -> bytes:
    return runner._receipt_bytes_from_fields(
        environment_seed=17,
        agent_seed=29,
        runtime_identity_bytes=_runtime_identity_bytes(_fake_bridge_runtime()),
        raw_reward_trace=bytes(runner.MATCHED_V3_HORIZON),
        raw_cumulative_score=0,
        trace_chain_sha256="1" * 64,
    )


def _resign_receipt(receipt: dict[str, Any]) -> bytes:
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256", None)
    receipt["receipt_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(receipt)


@pytest.mark.unit
def test_descriptor_is_distinct_detached_unexecuted_and_non_authorizing() -> None:
    raw = runner.canonical_matched_v3_ppo_gru_compiled_runner_descriptor_bytes()
    descriptor = runner.parse_matched_v3_ppo_gru_compiled_runner_descriptor(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
    )
    assert runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256 == (
        "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
    )
    assert len(raw) == 4_795
    assert descriptor["schema_version"] == (
        runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION
    )
    assert descriptor["status"] == "implemented_unexecuted"
    assert descriptor["implementation"]["source_self_hash_bound"] is False
    assert descriptor["claims"] and all(value is False for value in descriptor["claims"].values())
    assert descriptor["rng"]["categorical_mode"] == "low"
    assert descriptor["kernel"]["host_action_key_chain_replayed"] is True
    assert descriptor["state_lifecycle"]["completion_registration"] == (
        "run_closure_after_exact_horizon_checks_only"
    )
    assert any("not bound by qualification-plan v1" in item for item in descriptor["limitations"])
    assert any("pre-observation addendum or plan" in item for item in descriptor["limitations"])
    assert raw != v1_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()

    detached = runner.matched_v3_ppo_gru_compiled_runner_descriptor()
    detached["claims"]["authority_granted"] = True
    assert runner.matched_v3_ppo_gru_compiled_runner_descriptor()["claims"][
        "authority_granted"
    ] is False
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError):
        runner.parse_matched_v3_ppo_gru_compiled_runner_descriptor(
            v1_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
        )
    with pytest.raises(v1_runner.ForagerMatchedV3PPOGRURunnerError):
        v1_runner.parse_matched_v3_ppo_gru_runner_descriptor(raw)


@pytest.mark.unit
def test_descriptor_and_receipt_json_parsers_reject_ambiguity() -> None:
    descriptor = runner.canonical_matched_v3_ppo_gru_compiled_runner_descriptor_bytes()
    receipt = _receipt()
    receipt_digest = hashlib.sha256(receipt).hexdigest()

    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError):
        runner.parse_matched_v3_ppo_gru_compiled_runner_descriptor(b" " + descriptor)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="duplicate"):
        duplicate = b'{"schema_version":"duplicate",' + descriptor[1:]
        runner.parse_matched_v3_ppo_gru_compiled_runner_descriptor(duplicate)
    duplicate_receipt = b'{"accounting":{},' + receipt[1:]
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="duplicate"):
        runner.parse_ppo_gru_compiled_result_receipt(
            duplicate_receipt,
            expected_receipt_sha256=hashlib.sha256(duplicate_receipt).hexdigest(),
        )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="non-finite"):
        runner.parse_ppo_gru_compiled_result_receipt(
            receipt.replace(b'"raw_cumulative_score":0', b'"raw_cumulative_score":NaN'),
            expected_receipt_sha256=hashlib.sha256(
                receipt.replace(
                    b'"raw_cumulative_score":0', b'"raw_cumulative_score":NaN'
                )
            ).hexdigest(),
        )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="full-file"):
        runner.parse_ppo_gru_compiled_result_receipt(
            receipt, expected_receipt_sha256="0" * 64
        )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError):
        runner.parse_ppo_gru_compiled_result_receipt(
            b" " + receipt,
            expected_receipt_sha256=hashlib.sha256(b" " + receipt).hexdigest(),
        )
    oversized = b"{" + b" " * runner._MAX_RECEIPT_BYTES + b"}"
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="byte limit"):
        runner.parse_ppo_gru_compiled_result_receipt(
            oversized,
            expected_receipt_sha256=hashlib.sha256(oversized).hexdigest(),
        )
    too_deep = (
        b'{"claims":'
        + b"[" * (runner._MAX_JSON_DEPTH + 1)
        + b"0"
        + b"]" * (runner._MAX_JSON_DEPTH + 1)
        + b"}"
    )
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="nesting-depth"
    ):
        runner.parse_ppo_gru_compiled_result_receipt(
            too_deep,
            expected_receipt_sha256=hashlib.sha256(too_deep).hexdigest(),
        )
    too_many_nodes = (
        b'{"claims":[' + b"0," * runner._MAX_JSON_NODES + b"0]}"
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="node limit"):
        runner.parse_ppo_gru_compiled_result_receipt(
            too_many_nodes,
            expected_receipt_sha256=hashlib.sha256(too_many_nodes).hexdigest(),
        )
    with pytest.raises(TypeError):
        cast(Any, runner.parse_ppo_gru_compiled_result_receipt)(receipt)
    assert runner.parse_ppo_gru_compiled_result_receipt(
        receipt, expected_receipt_sha256=receipt_digest
    )["claims"]["authority_granted"] is False


@pytest.mark.unit
def test_exact_source_bindings_and_workload_arithmetic_remain_stable() -> None:
    sources = (
        (
            bridge.__file__,
            runner.BOUND_BRIDGE_IMPLEMENTATION_SHA256,
        ),
        (
            ppo_gru.__file__,
            runner.BOUND_CORE_IMPLEMENTATION_SHA256,
        ),
        (
            v1_runner.__file__,
            runner.BOUND_V1_RUNNER_IMPLEMENTATION_SHA256,
        ),
    )
    for source, expected in sources:
        assert type(source) is str
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected

    accounting = runner.matched_v3_ppo_gru_compiled_accounting()
    assert accounting == {
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
    assert accounting["compiled_chunk_count"] * accounting["compiled_chunk_steps"] == (
        accounting["environment_interactions"]
    )
    assert accounting["optimizer_updates"] == 976 * 4 * 4


@pytest.mark.unit
def test_compiled_scan_matches_scalar_transition_and_rng_semantics() -> None:
    inputs = _initial_inputs()
    result = _kernel(chunk_steps=4)(None, *inputs)
    result = runner._require_clean_chunk(
        result,
        chunk_steps=4,
        hidden_size=_HIDDEN_SIZE,
        expected_final_step=4,
        initial_environment_key=inputs[2],
        initial_agent_key=inputs[4],
    )
    scalar = _scalar_rollout(
        steps=4,
        environment_state=inputs[0],
        observation=inputs[1],
        environment_key=inputs[2],
        carry=inputs[3],
        agent_key=inputs[4],
    )
    for name in (
        "observation",
        "environment_key",
        "gru_carry",
        "agent_key",
        "observations",
        "incoming_carries",
        "outgoing_carries",
        "action_key_words",
        "logits",
        "actions",
        "old_log_probs",
        "old_values",
        "raw_rewards",
        "next_observations",
        "bootstrap_value",
    ):
        _assert_array_equal(getattr(result, name), scalar[name])
    scalar_state = cast(_FakeEnvironmentState, scalar["environment_state"])
    for field in _FakeEnvironmentState._fields:
        _assert_array_equal(getattr(result.environment_state, field), getattr(scalar_state, field))
    assert int(np.asarray(result.environment_state.reset_count)) == 1
    assert int(np.asarray(result.environment_state.step_call_count)) == 4


@pytest.mark.unit
def test_two_consecutive_chunks_equal_one_longer_scan_exactly() -> None:
    inputs = _initial_inputs()
    four = _kernel(chunk_steps=4)
    first = four(None, *inputs)
    second = four(
        None,
        first.environment_state,
        first.observation,
        first.environment_key,
        first.gru_carry,
        first.agent_key,
        first.absolute_step,
    )
    combined = _kernel(chunk_steps=8)(None, *inputs)

    for name in (
        "observations",
        "incoming_carries",
        "outgoing_carries",
        "action_key_words",
        "logits",
        "actions",
        "old_log_probs",
        "old_values",
        "raw_rewards",
        "next_observations",
    ):
        _assert_array_equal(
            jnp.concatenate((getattr(first, name), getattr(second, name))),
            getattr(combined, name),
        )
    for name in (
        "observation",
        "environment_key",
        "gru_carry",
        "agent_key",
        "absolute_step",
        "bootstrap_value",
    ):
        _assert_array_equal(getattr(second, name), getattr(combined, name))
    for field in _FakeEnvironmentState._fields:
        _assert_array_equal(
            getattr(second.environment_state, field),
            getattr(combined.environment_state, field),
        )


@pytest.mark.unit
def test_agent_keys_do_not_change_environment_key_schedule_and_bootstrap_draws_no_key() -> None:
    first_inputs = _initial_inputs(agent_seed=29)
    second_inputs = _initial_inputs(agent_seed=31)
    kernel = _kernel(chunk_steps=6)
    first = kernel(None, *first_inputs)
    second = kernel(None, *second_inputs)

    _assert_array_equal(first.environment_key, second.environment_key)
    expected_agent_key = first_inputs[4]
    for _ in range(6):
        expected_agent_key, _action_key = jr.split(expected_agent_key)
    _assert_array_equal(first.agent_key, expected_agent_key)
    assert not np.array_equal(
        np.asarray(first.action_key_words), np.asarray(second.action_key_words)
    )


def _violating_environment_step(kind: str) -> Any:
    def step(
        key: Array,
        state: _FakeEnvironmentState,
        action: Array,
        params: object,
    ) -> tuple[Array, _FakeEnvironmentState, Array, Array, dict[str, Array]]:
        observation, next_state, reward, done, info = _environment_step(
            key, state, action, params
        )
        if kind == "observation":
            observation = observation.at[0, 0, 0].set(jnp.float32(2.0))
        elif kind == "reward":
            reward = jnp.asarray(2.0, dtype=jnp.float32)
        elif kind == "done":
            done = jnp.asarray(True, dtype=jnp.bool_)
        elif kind == "info":
            info["biome_regret"] = jnp.asarray(-1.0, dtype=jnp.float32)
        elif kind == "time":
            next_state = next_state._replace(time=next_state.time + jnp.int32(1))
        return observation, next_state, reward, done, info

    return step


def _violating_policy(
    variables: object,
    carry: Array,
    observation: Array,
    reset_before: Array,
) -> tuple[Array, Array, Array]:
    outgoing, logits, value = _policy_apply(
        variables, carry, observation, reset_before
    )
    return outgoing.at[0].set(jnp.nan), logits, value


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "expected_bit"),
    [
        ("observation", runner.VIOLATION_OBSERVATION),
        ("reward", runner.VIOLATION_REWARD),
        ("done", runner.VIOLATION_DONE),
        ("info", runner.VIOLATION_INFO),
        ("time", runner.VIOLATION_ENVIRONMENT_TIME),
    ],
)
def test_violation_poisoning_stops_after_first_transition(
    kind: str, expected_bit: int
) -> None:
    inputs = _initial_inputs()
    result = _kernel(
        chunk_steps=4,
        environment_step=_violating_environment_step(kind),
    )(None, *inputs)

    assert int(np.asarray(result.violation_mask)) == expected_bit
    assert int(np.asarray(result.first_invalid_offset)) == 0
    assert int(np.asarray(result.absolute_step)) == 1
    assert int(np.asarray(result.environment_state.step_call_count)) == 1
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="poisoned or invalid"
    ):
        runner._require_clean_chunk(
            result,
            chunk_steps=4,
            hidden_size=_HIDDEN_SIZE,
            expected_final_step=4,
            initial_environment_key=inputs[2],
            initial_agent_key=inputs[4],
        )


@pytest.mark.unit
def test_invalid_policy_poisoning_precedes_key_splits_and_environment_step() -> None:
    inputs = _initial_inputs()
    result = _kernel(
        chunk_steps=4,
        policy_apply=_violating_policy,
    )(None, *inputs)

    assert int(np.asarray(result.violation_mask)) == runner.VIOLATION_POLICY
    assert int(np.asarray(result.first_invalid_offset)) == 0
    assert int(np.asarray(result.absolute_step)) == 0
    assert int(np.asarray(result.environment_state.step_call_count)) == 0
    _assert_array_equal(result.environment_key, inputs[2])
    _assert_array_equal(result.agent_key, inputs[4])
    np.testing.assert_array_equal(
        np.asarray(result.action_key_words),
        np.zeros((4, 2), dtype=np.uint32),
    )
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="poisoned or invalid"
    ):
        runner._require_clean_chunk(
            result,
            chunk_steps=4,
            hidden_size=_HIDDEN_SIZE,
            expected_final_step=4,
            initial_environment_key=inputs[2],
            initial_agent_key=inputs[4],
        )


@pytest.mark.unit
def test_host_boundary_replays_keys_actions_log_probs_and_endpoints() -> None:
    inputs = _initial_inputs()
    result = _kernel(chunk_steps=4)(None, *inputs)
    runner._require_clean_chunk(
        result,
        chunk_steps=4,
        hidden_size=_HIDDEN_SIZE,
        expected_final_step=4,
        initial_environment_key=inputs[2],
        initial_agent_key=inputs[4],
    )

    changed_action_keys = result.action_key_words.at[0, 0].set(
        result.action_key_words[0, 0] ^ jnp.uint32(1)
    )
    changed_actions = result.actions.at[0].set(
        jnp.mod(result.actions[0] + jnp.int32(1), jnp.int32(4))
    )
    changed_log_probs = result.old_log_probs.at[0].add(jnp.float32(0.25))
    mutations = (
        (result._replace(action_key_words=changed_action_keys), "action-key"),
        (result._replace(actions=changed_actions), "action replay"),
        (result._replace(old_log_probs=changed_log_probs), "log-probability replay"),
        (result._replace(agent_key=inputs[4]), "agent-key"),
        (result._replace(environment_key=inputs[2]), "environment-key"),
    )
    for changed, message in mutations:
        with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match=message):
            runner._require_clean_chunk(
                changed,
                chunk_steps=4,
                hidden_size=_HIDDEN_SIZE,
                expected_final_step=4,
                initial_environment_key=inputs[2],
                initial_agent_key=inputs[4],
            )


@pytest.mark.unit
def test_runtime_capability_is_pid_bound_single_use_and_copy_resistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_runtime = _fake_bridge_runtime()
    runtime = runner._register_runtime(
        bridge_runtime=bridge_runtime,
        kernel=cast(Any, lambda *args: args),
        runtime_identity_bytes=_runtime_identity_bytes(bridge_runtime),
    )
    assert runner._validated_runtime_binding(runtime).bridge_runtime is bridge_runtime
    with pytest.raises(TypeError):
        copy.copy(runtime)
    with pytest.raises(TypeError):
        copy.deepcopy(runtime)
    with pytest.raises(TypeError):
        pickle.dumps(runtime)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="forged"):
        runner._validated_runtime_binding(dataclasses.replace(runtime))
    forged = runner.PPOGRUCompiledRuntime(
        bridge_runtime=runtime.bridge_runtime,
        runtime_identity_bytes=runtime.runtime_identity_bytes,
        _kernel=runtime._kernel,
        _capability=runner._RuntimeCapability(),
        _pid=os.getpid(),
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="forged"):
        runner._validated_runtime_binding(forged)

    stale = runner._register_runtime(
        bridge_runtime=bridge_runtime,
        kernel=cast(Any, lambda *args: args),
        runtime_identity_bytes=_runtime_identity_bytes(bridge_runtime),
    )
    object.__setattr__(stale, "runtime_identity_bytes", b"{}")
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="stale"):
        runner._validated_runtime_binding(stale)

    original_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: original_pid + 1)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="forked"):
        runner._validated_runtime_binding(runtime)
    monkeypatch.undo()
    runner._claim_runtime(runtime)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="single-use"):
        runner._claim_runtime(runtime)


@pytest.mark.unit
def test_execution_opt_in_is_exact_and_remains_nonauthorizing() -> None:
    bridge_runtime = _fake_bridge_runtime()
    runtime = runner._register_runtime(
        bridge_runtime=bridge_runtime,
        kernel=cast(Any, lambda *args: args),
        runtime_identity_bytes=_runtime_identity_bytes(bridge_runtime),
    )
    for invalid in (False, 1, "true", None):
        with pytest.raises(runner.PPOGRUCompiledExecutionBlockedError):
            runner.run_matched_v3_ppo_gru_compiled(
                environment_seed=17,
                agent_seed=29,
                runtime=runtime,
                unqualified_engineering=cast(Any, invalid),
            )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="uint31"):
        runner.run_matched_v3_ppo_gru_compiled(
            environment_seed=-1,
            agent_seed=29,
            runtime=runtime,
            unqualified_engineering=True,
        )
    assert all(
        value is False
        for value in runner.matched_v3_ppo_gru_compiled_runner_descriptor()["claims"].values()
    )


@pytest.mark.unit
def test_result_receipt_requires_full_digest_rejects_old_schema_and_bounds() -> None:
    raw = _receipt()
    digest = hashlib.sha256(raw).hexdigest()
    parsed = runner.parse_ppo_gru_compiled_result_receipt(
        raw, expected_receipt_sha256=digest
    )
    assert parsed["schema_version"] == runner.PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION
    assert parsed["accounting"] == runner.matched_v3_ppo_gru_compiled_accounting()
    assert parsed["completion"]["content_independently_proves_execution"] is False
    assert any("not bound by qualification-plan v1" in item for item in parsed["limitations"])
    assert any("pre-observation addendum or plan" in item for item in parsed["limitations"])

    mutation = copy.deepcopy(parsed)
    mutation["seeds"]["environment_seed"] = -1
    changed = _resign_receipt(mutation)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="seed"):
        runner.parse_ppo_gru_compiled_result_receipt(
            changed, expected_receipt_sha256=hashlib.sha256(changed).hexdigest()
        )

    authority = copy.deepcopy(parsed)
    authority["claims"]["promotion_authorized"] = True
    changed = _resign_receipt(authority)
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError):
        runner.parse_ppo_gru_compiled_result_receipt(
            changed, expected_receipt_sha256=hashlib.sha256(changed).hexdigest()
        )

    with pytest.raises(v1_runner.ForagerMatchedV3PPOGRURunnerError):
        v1_runner.parse_ppo_gru_result_receipt(raw, expected_receipt_sha256=digest)
    old_descriptor = v1_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError):
        runner.parse_ppo_gru_compiled_result_receipt(
            old_descriptor,
            expected_receipt_sha256=hashlib.sha256(old_descriptor).hexdigest(),
        )


@pytest.mark.unit
def test_no_execution_path_can_register_a_live_completion() -> None:
    assert not hasattr(runner, "_complete_runtime")
    assert not hasattr(runner, "_register_outcome")
    bridge_runtime = _fake_bridge_runtime()
    identity_bytes = _runtime_identity_bytes(bridge_runtime)
    runtime = runner._register_runtime(
        bridge_runtime=bridge_runtime,
        kernel=cast(Any, lambda *args: args),
        runtime_identity_bytes=identity_bytes,
    )
    binding = runner._claim_runtime(runtime)
    receipt = _receipt()
    outcome = runner.PPOGRUCompiledOutcome(
        raw_reward_trace=bytes(runner.MATCHED_V3_HORIZON),
        raw_cumulative_score=0,
        interactions=runner.MATCHED_V3_HORIZON,
        rollout_count=runner.PPO_GRU_COMPILED_CHUNK_COUNT,
        optimizer_update_count=runner.PPO_GRU_OPTIMIZER_UPDATES,
        total_agent_draw_count=runner.PPO_GRU_TOTAL_AGENT_DRAWS,
        bridge_environment_key_use_count=runner.PPO_GRU_BRIDGE_KEY_USES,
        trace_chain_sha256="1" * 64,
        runtime_identity_bytes=identity_bytes,
        receipt_bytes=receipt,
        production_runtime=True,
        _capability=runner._OutcomeCapability(),
        _pid=os.getpid(),
    )

    assert binding.in_flight is True
    assert binding.completed is False
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="forged"):
        runner.canonical_ppo_gru_compiled_result_receipt_bytes(outcome)


@pytest.mark.unit
def test_forged_outcome_capability_and_copy_pickle_attempts_fail() -> None:
    receipt = _receipt()
    outcome = runner.PPOGRUCompiledOutcome(
        raw_reward_trace=bytes(runner.MATCHED_V3_HORIZON),
        raw_cumulative_score=0,
        interactions=runner.MATCHED_V3_HORIZON,
        rollout_count=runner.PPO_GRU_COMPILED_CHUNK_COUNT,
        optimizer_update_count=runner.PPO_GRU_OPTIMIZER_UPDATES,
        total_agent_draw_count=runner.PPO_GRU_TOTAL_AGENT_DRAWS,
        bridge_environment_key_use_count=runner.PPO_GRU_BRIDGE_KEY_USES,
        trace_chain_sha256="1" * 64,
        runtime_identity_bytes=_runtime_identity_bytes(_fake_bridge_runtime()),
        receipt_bytes=receipt,
        production_runtime=True,
        _capability=runner._OutcomeCapability(),
        _pid=os.getpid(),
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRUCompiledRunnerError, match="forged"):
        runner.canonical_ppo_gru_compiled_result_receipt_bytes(outcome)
    with pytest.raises(TypeError):
        copy.copy(outcome)
    with pytest.raises(TypeError):
        copy.deepcopy(outcome)
    with pytest.raises(TypeError):
        pickle.dumps(outcome)
