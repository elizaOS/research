"""Exact schedule and atomicity tests for Step 2 discovery streams."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.feature_discovery import (
    FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES,
    FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES,
    FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA,
    INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA,
    INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA,
    NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA,
    NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA,
    FeatureDiscoveryStreamResourceBudget,
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryStream,
    feature_discovery_stream_clock_nbytes,
    measure_feature_discovery_stream_state_nbytes,
    migrate_legacy_interaction_feature_discovery_state,
    migrate_legacy_nonlinear_feature_discovery_state,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _stream(kind: str) -> Any:
    if kind == "nonlinear":
        return NonlinearFeatureDiscoveryStream(
            feature_dim=3,
            n_tasks=2,
            n_latents=4,
            n_contexts=5,
            context_length=3,
            active_latents_per_context=2,
            noise_std=0.0,
        )
    return InteractionFeatureDiscoveryStream(
        feature_dim=3,
        n_tasks=2,
        n_contexts=5,
        context_length=3,
        active_pairs_per_context=2,
        noise_std=0.0,
    )


def _assert_bit_exact(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        try:
            left_array = np.asarray(left_leaf)
            right_array = np.asarray(right_leaf)
        except TypeError:
            left_array = np.asarray(jr.key_data(left_leaf))
            right_array = np.asarray(jr.key_data(right_leaf))
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes() == right_array.tobytes()


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_exact_schedule_identity_beyond_2p32_and_low_word_carry(kind: str) -> None:
    stream = _stream(kind)
    event = (1 << 32) + 17
    state = stream.init(jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((1, 17), dtype=jnp.uint32),
    )

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.update_applied)
    assert int(result.context_index) == (event // 3) % 5
    np.testing.assert_array_equal(result.post_step_words, (1, 18))
    assert int(result.state.step_count) == _INT32_MAX

    carry_state = state.replace(
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32)
    )
    carry = stream.step_result(carry_state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(carry.update_applied)
    assert int(carry.context_index) == (_UINT32_MAX // 3) % 5
    np.testing.assert_array_equal(carry.post_step_words, (1, 0))


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_terminal_all_ones_refuses_and_rolls_back_rng_and_oracle_state(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
    )

    result = jax.jit(stream.step_result)(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    _assert_bit_exact(result.state, state)
    assert bool(jnp.all(jnp.isnan(result.timestep.observation)))
    assert bool(jnp.all(jnp.isnan(result.timestep.target)))


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_invalid_input_and_corrupt_clock_are_atomic_noops(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(2))

    bad_input = stream.step_result(
        state,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    assert not bool(bad_input.input_valid)
    assert not bool(bad_input.update_applied)
    _assert_bit_exact(bad_input.state, state)

    corrupt = state.replace(
        step_words=jnp.asarray((0, 9), dtype=jnp.uint32)
    )
    bad_clock = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(bad_clock.lifetime_counter_valid)
    assert not bool(bad_clock.update_applied)
    _assert_bit_exact(bad_clock.state, corrupt)


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_nonfinite_persistent_state_is_an_atomic_noop(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(3))
    if kind == "nonlinear":
        corrupt = state.replace(
            latent_weights=state.latent_weights.at[0, 0].set(jnp.nan)
        )
    else:
        corrupt = state.replace(
            context_weights=state.context_weights.at[0, 0, 0].set(jnp.inf)
        )

    result = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))

    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, corrupt)


def test_corrupt_interaction_pair_identity_is_an_atomic_noop() -> None:
    stream = _stream("interaction")
    state = stream.init(jr.key(4))
    corrupt = state.replace(
        pair_left=state.pair_left.at[0].set(2)
    )

    result = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))

    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, corrupt)


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_tuple_step_api_matches_diagnostic_step_result(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(5))
    idx = jnp.asarray(0, dtype=jnp.int32)

    expected = stream.step_result(state, idx)
    timestep, new_state = stream.step(state, idx)

    _assert_bit_exact(timestep, expected.timestep)
    _assert_bit_exact(new_state, expected.state)


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_eager_jit_and_scan_preserve_exact_schedule(kind: str) -> None:
    stream = _stream(kind)
    initial_event = _UINT32_MAX - 1
    state = stream.init(jr.key(6)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, initial_event), dtype=jnp.uint32),
    )
    idx = jnp.asarray(0, dtype=jnp.int32)
    eager = stream.step_result(state, idx)
    compiled = jax.jit(stream.step_result)(state, idx)
    _assert_bit_exact(eager, compiled)

    def scan_step(carry: Any, scan_idx: jax.Array) -> tuple[Any, jax.Array]:
        result = stream.step_result(carry, scan_idx)
        return result.state, result.context_index

    final_state, contexts = jax.lax.scan(
        scan_step,
        state,
        jnp.arange(4, dtype=jnp.int32),
    )
    expected_contexts = np.asarray(
        [((initial_event + offset) // 3) % 5 for offset in range(4)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(contexts, expected_contexts)
    np.testing.assert_array_equal(final_state.step_words, (1, 2))
    assert int(final_state.step_count) == _INT32_MAX


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_strict_config_schema_and_resource_contract(kind: str) -> None:
    stream = _stream(kind)
    config = stream.to_config()
    if kind == "nonlinear":
        assert config["config_schema"] == NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA
        assert config["state_schema"] == NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA
        restored: Any = NonlinearFeatureDiscoveryStream.from_config(config)
    else:
        assert config["config_schema"] == INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA
        assert config["state_schema"] == INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA
        restored = InteractionFeatureDiscoveryStream.from_config(config)
    assert restored.to_config() == config

    extra = copy.deepcopy(config)
    extra["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        type(stream).from_config(extra)

    state = stream.init(jr.key(7))
    budget = stream.resource_budget
    assert budget.state_nbytes == measure_feature_discovery_stream_state_nbytes(state)
    assert budget.exact_clock_nbytes == FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES
    assert budget.exact_clock_delta_nbytes == (
        FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES
    )
    assert feature_discovery_stream_clock_nbytes() == (
        FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES
    )
    old_nbytes = budget.state_nbytes - state.step_words.nbytes
    assert budget.state_nbytes - old_nbytes == FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES
    serialized_budget = budget.to_dict()
    assert serialized_budget["schema"] == FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA
    assert FeatureDiscoveryStreamResourceBudget.from_dict(serialized_budget) == budget


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_explicit_legacy_state_migration_rejects_ambiguous_history(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(8))
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }
    legacy["step_count"] = jnp.asarray(23, dtype=jnp.int32)
    if kind == "nonlinear":
        migrated: Any = migrate_legacy_nonlinear_feature_discovery_state(
            legacy,
            stream=stream,
        )
    else:
        migrated = migrate_legacy_interaction_feature_discovery_state(
            legacy,
            stream=stream,
        )
    np.testing.assert_array_equal(migrated.step_words, (0, 23))
    assert bool(stream.state_is_valid(migrated))

    ambiguous = dict(legacy)
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    migration = (
        migrate_legacy_nonlinear_feature_discovery_state
        if kind == "nonlinear"
        else migrate_legacy_interaction_feature_discovery_state
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migration(ambiguous, stream=stream)


@pytest.mark.parametrize("kind", ["nonlinear", "interaction"])
def test_structural_contracts_fail_fast(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(9))
    with pytest.raises(ValueError, match="idx must be scalar"):
        stream.step_result(state, jnp.zeros((1,), dtype=jnp.int32))
    malformed = state.replace(
        step_words=jnp.zeros((3,), dtype=jnp.uint32)
    )
    with pytest.raises(ValueError, match="step_words"):
        stream.step_result(malformed, jnp.asarray(0, dtype=jnp.int32))


def test_constructor_rejects_nonfinite_schedule_configuration() -> None:
    with pytest.raises(ValueError, match="context_length"):
        NonlinearFeatureDiscoveryStream(feature_dim=2, context_length=_INT32_MAX + 1)
    with pytest.raises(ValueError, match="noise_std"):
        InteractionFeatureDiscoveryStream(feature_dim=2, noise_std=float("nan"))
