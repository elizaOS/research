"""Focused exact-clock contracts for the on-policy shared-trunk Horde."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.normalizers import EMANormalizer, WelfordNormalizer
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit

_FLOAT32_INTEGER_LIMIT = 2**24
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
_NEXT_OBSERVATION = jnp.asarray((-0.75, 0.125), dtype=jnp.float32)
_CUMULANT = jnp.asarray((0.75,), dtype=jnp.float32)
_DISCOUNT = jnp.asarray((0.35,), dtype=jnp.float32)


def _learner(*, normalizer=None) -> HordeLearner:  # type: ignore[no-untyped-def]
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
    return HordeLearner(
        spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _update(learner: HordeLearner, state):  # type: ignore[no-untyped-def]
    return learner.update(
        state,
        _OBSERVATION,
        _CUMULANT,
        _NEXT_OBSERVATION,
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


def _assert_refused(result, state) -> None:  # type: ignore[no-untyped-def]
    assert not bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, state.step_words)
    np.testing.assert_array_equal(result.post_step_words, state.step_words)
    _assert_persistent_array_tree_bit_equal(result.state, state)


def test_ordinary_early_update_matches_child_and_exposes_transaction() -> None:
    learner = _learner(normalizer=EMANormalizer(decay=0.9))
    state = learner.init(2, jax.random.key(0))
    next_predictions = learner.predict(state, _NEXT_OBSERVATION)
    target = _CUMULANT + learner.horde_spec.gammas * next_predictions
    direct_child = learner.learner.update(state, _OBSERVATION, target)

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
        np.testing.assert_allclose(result.td_targets, target, rtol=0.0, atol=0.0)
        _assert_persistent_array_tree_bit_equal(result.state, direct_child.state)
        np.testing.assert_array_equal(result.predictions, direct_child.predictions)
        np.testing.assert_array_equal(result.td_errors, direct_child.errors)
        np.testing.assert_array_equal(
            result.per_demon_metrics,
            direct_child.per_head_metrics,
        )

    _assert_persistent_array_tree_bit_equal(eager.state, compiled.state)


def test_explicit_discounts_preserve_the_same_child_transaction_contract() -> None:
    learner = _learner()
    state = learner.init(2, jax.random.key(1))
    next_predictions = learner.predict(state, _NEXT_OBSERVATION)
    target = _CUMULANT + _DISCOUNT * next_predictions
    direct_child = learner.learner.update(state, _OBSERVATION, target)

    result = learner.update_with_discounts(
        state,
        _OBSERVATION,
        _CUMULANT,
        _NEXT_OBSERVATION,
        _DISCOUNT,
    )

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, direct_child.pre_step_words)
    np.testing.assert_array_equal(result.post_step_words, direct_child.post_step_words)
    np.testing.assert_array_equal(result.td_targets, target)
    _assert_persistent_array_tree_bit_equal(result.state, direct_child.state)


def test_scan_crosses_uint32_carry_with_saturating_telemetry() -> None:
    learner = _learner()
    state = learner.init(2, jax.random.key(2)).replace(
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


@pytest.mark.parametrize("failure", ("outer_invalid", "normalizer_misaligned"))
def test_invalid_or_misaligned_clock_is_a_diagnosed_atomic_noop(
    failure: str,
) -> None:
    learner = _learner(normalizer=EMANormalizer(decay=0.9))
    state = learner.init(2, jax.random.key(3))
    assert state.normalizer_state is not None
    if failure == "outer_invalid":
        state = state.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    else:
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
        assert bool(result.normalizer_estimator_capacity_available)
        assert bool(result.normalizer_counter_aligned) is (failure == "outer_invalid")
        _assert_refused(result, state)


def test_welford_estimator_horizon_refuses_the_whole_transaction() -> None:
    learner = _learner(normalizer=WelfordNormalizer())
    state = learner.init(2, jax.random.key(4))
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

    result = learner.update_with_discounts(
        state,
        _OBSERVATION,
        _CUMULANT,
        _NEXT_OBSERVATION,
        _DISCOUNT,
    )

    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert not bool(result.normalizer_estimator_capacity_available)
    _assert_refused(result, state)


def test_all_ones_nested_clock_is_a_diagnosed_atomic_noop() -> None:
    learner = _learner(normalizer=EMANormalizer(decay=0.9))
    state = learner.init(2, jax.random.key(5))
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
    _assert_refused(result, state)
