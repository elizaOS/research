"""Focused exact-clock contracts for the shared-trunk off-policy Horde."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.normalizers import EMANormalizer, WelfordNormalizer
from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit

_FLOAT32_INTEGER_LIMIT = 2**24
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
_NEXT_OBSERVATION = jnp.asarray((-0.75, 0.125), dtype=jnp.float32)
_CUMULANT = jnp.asarray((0.75,), dtype=jnp.float32)
_RHO = jnp.asarray((1.25,), dtype=jnp.float32)
_DISCOUNT = jnp.asarray((0.6,), dtype=jnp.float32)


def _learner(*, normalizer=None) -> OffPolicyHordeLearner:
    spec = create_horde_spec(
        (
            GVFSpec(
                name="horizon_demon",
                demon_type=DemonType.PREDICTION,
                gamma=0.6,
                lamda=0.4,
                cumulant_index=0,
            ),  # type: ignore[call-arg]
        )
    )
    return OffPolicyHordeLearner(
        spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
        sparsity=0.0,
        use_layer_norm=False,
        ratio_clip=2.0,
        trace_ratio_clip=2.0,
    )


def _update(learner: OffPolicyHordeLearner, state):  # type: ignore[no-untyped-def]
    return learner.update_with_ratios_and_discounts(
        state,
        _OBSERVATION,
        _CUMULANT,
        _NEXT_OBSERVATION,
        _RHO,
        _DISCOUNT,
    )


def _assert_persistent_array_tree_bit_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        # The inherited Python timing floats are host-only lifecycle metadata.
        if not isinstance(first_leaf, jax.Array) or not isinstance(
            second_leaf,
            jax.Array,
        ):
            continue
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def test_ordinary_early_update_is_unchanged_and_exact_in_eager_and_jit() -> None:
    normalizer = EMANormalizer(decay=0.9)
    learner = _learner(normalizer=normalizer)
    state = learner.init(2, jax.random.key(0))
    expected_next_predictions = learner.predict(state, _NEXT_OBSERVATION)

    with jax.disable_jit():
        eager = _update(learner, state)
    compiled = _update(learner, state)

    for result in (eager, compiled):
        assert bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        assert bool(result.update_applied)
        np.testing.assert_array_equal(result.pre_step_words, (0, 0))
        np.testing.assert_array_equal(result.post_step_words, (0, 1))
        np.testing.assert_array_equal(result.state.step_words, (0, 1))
        assert int(result.state.step_count) == 1
        assert result.state.normalizer_state is not None
        np.testing.assert_array_equal(
            result.state.normalizer_state.sample_count_words,
            result.state.step_words,
        )
        np.testing.assert_allclose(
            result.td_targets,
            _CUMULANT + _DISCOUNT * expected_next_predictions,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.td_errors,
            result.td_targets - result.predictions,
            rtol=0.0,
            atol=0.0,
        )
        assert not np.array_equal(
            np.asarray(result.state.head_params.biases[0]),
            np.asarray(state.head_params.biases[0]),
        )

    _assert_persistent_array_tree_bit_equal(eager.state, compiled.state)
    np.testing.assert_array_equal(eager.per_demon_metrics, compiled.per_demon_metrics)


def test_scan_crosses_uint32_carry_with_saturating_telemetry() -> None:
    learner = _learner()
    state = learner.init(2, jax.random.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )

    def step(carry, unused):  # type: ignore[no-untyped-def]
        del unused
        result = _update(learner, carry)
        return result.state, (
            result.pre_step_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, (pre_words, post_words, applied) = jax.lax.scan(
        step,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(applied, (True, True))
    np.testing.assert_array_equal(
        pre_words,
        np.asarray(((0, _UINT32_MAX), (1, 0)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        post_words,
        np.asarray(((1, 0), (1, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(final_state.step_words, (1, 1))
    assert int(final_state.step_count) == _INT32_MAX


def test_welford_horizon_refuses_the_whole_transaction() -> None:
    learner = _learner(normalizer=WelfordNormalizer())
    state = learner.init(2, jax.random.key(2))
    assert state.normalizer_state is not None
    state = state.replace(
        step_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
        step_words=jnp.asarray((0, _FLOAT32_INTEGER_LIMIT), dtype=jnp.uint32),
        normalizer_state=state.normalizer_state.replace(
            sample_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
            sample_count_words=jnp.asarray(
                (0, _FLOAT32_INTEGER_LIMIT),
                dtype=jnp.uint32,
            ),
        ),
    )

    result = _update(learner, state)

    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert not bool(result.normalizer_estimator_capacity_available)
    assert not bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, state.step_words)
    np.testing.assert_array_equal(result.post_step_words, state.step_words)
    _assert_persistent_array_tree_bit_equal(result.state, state)


def test_misaligned_normalizer_clock_is_an_atomic_noop() -> None:
    learner = _learner(normalizer=EMANormalizer(decay=0.9))
    state = learner.init(2, jax.random.key(3))
    assert state.normalizer_state is not None
    state = state.replace(
        normalizer_state=state.normalizer_state.replace(
            sample_count=jnp.asarray(1, dtype=jnp.int32),
            sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    )

    with jax.disable_jit():
        eager = _update(learner, state)
    compiled = _update(learner, state)

    for result in (eager, compiled):
        assert not bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert not bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        assert not bool(result.update_applied)
        np.testing.assert_array_equal(result.pre_step_words, (0, 0))
        np.testing.assert_array_equal(result.post_step_words, (0, 0))
        _assert_persistent_array_tree_bit_equal(result.state, state)


def test_all_ones_nested_clock_is_a_diagnosed_atomic_noop() -> None:
    learner = _learner(normalizer=EMANormalizer(decay=0.9))
    state = learner.init(2, jax.random.key(4))
    assert state.normalizer_state is not None
    terminal_words = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    state = state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=terminal_words,
        normalizer_state=state.normalizer_state.replace(
            sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            sample_count_words=terminal_words,
        ),
    )

    result = _update(learner, state)

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert bool(result.normalizer_estimator_capacity_available)
    assert not bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, terminal_words)
    np.testing.assert_array_equal(result.post_step_words, terminal_words)
    _assert_persistent_array_tree_bit_equal(result.state, state)
