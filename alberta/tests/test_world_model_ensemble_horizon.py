"""Focused exact-clock contracts for compound world-model ensembles."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.normalizers import EMANormalizer
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    save_world_model_ensemble_checkpoint,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_EVENT = (
    jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    jnp.asarray(1, dtype=jnp.int32),
    jnp.asarray(0.3, dtype=jnp.float32),
    jnp.asarray(0.9, dtype=jnp.float32),
    jnp.asarray((0.15, -0.23), dtype=jnp.float32),
)


def _ensemble() -> WorldModelEnsemble:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_decay=0.8,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )


def _ensemble_with_normalizer(normalizer) -> WorldModelEnsemble:  # type: ignore[no-untyped-def]
    ensemble = _ensemble()
    model = ActionConditionedWorldModel(
        ensemble.config.model,
        normalizer=normalizer,
    )
    ensemble._model = model
    template = model.init(jr.key(0))
    leaves, structure = jax.tree_util.tree_flatten(template)
    ensemble._member_state_static_signature = (
        structure,
        tuple((jnp.asarray(leaf).shape, jnp.asarray(leaf).dtype) for leaf in leaves),
    )
    return ensemble


def _assert_tree_bit_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        # Inherited host timing floats are outside the persistent JAX contract.
        if not isinstance(first_leaf, jax.Array) or not isinstance(second_leaf, jax.Array):
            continue
        first_dtype = getattr(first_leaf, "dtype", None)
        second_dtype = getattr(second_leaf, "dtype", None)
        if first_dtype is not None and jax.dtypes.issubdtype(
            first_dtype,
            jax.dtypes.prng_key,
        ):
            first_leaf = jr.key_data(first_leaf)
        if second_dtype is not None and jax.dtypes.issubdtype(
            second_dtype,
            jax.dtypes.prng_key,
        ):
            second_leaf = jr.key_data(second_leaf)
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def _assert_tree_numerically_close(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if not isinstance(first_leaf, jax.Array) or not isinstance(second_leaf, jax.Array):
            continue
        if jax.dtypes.issubdtype(first_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jr.key_data(first_leaf),
                jr.key_data(second_leaf),
            )
        elif jnp.issubdtype(first_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(first_leaf, second_leaf, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(first_leaf, second_leaf)


def _long_horizon_state(ensemble: WorldModelEnsemble):  # type: ignore[no-untyped-def]
    state = ensemble.init(jr.key(10))
    words = jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32)
    members = []
    for member in state.member_states:
        members.append(
            member.replace(
                learner_state=member.learner_state.replace(
                    step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
                    step_words=words,
                ),
                observation_min=jnp.asarray((-1.0, -1.0), dtype=jnp.float32),
                observation_max=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
                reward_min=jnp.asarray(-1.0, dtype=jnp.float32),
                reward_max=jnp.asarray(1.0, dtype=jnp.float32),
                step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
                step_words=words,
            )
        )
    signal_state = state.signal_state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        valid_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=words,
        valid_words=words,
    )
    return state.replace(
        member_states=tuple(members),
        signal_state=signal_state,
        bootstrap_key=jr.key(0),  # first two-member mask is deterministically all true
        member_update_counts=jnp.full((2,), _INT32_MAX, dtype=jnp.int32),
        member_update_count_words=jnp.tile(words[None, :], (2, 1)),
        event_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        event_count_words=words,
    )


def test_early_real_update_matches_direct_members_eager_and_jit() -> None:
    ensemble = _ensemble()
    state = ensemble.init(jr.key(0)).replace(bootstrap_key=jr.key(0))
    direct = tuple(
        ensemble.member_model.update(member, *_EVENT)
        for member in state.member_states
    )

    with jax.disable_jit():
        eager = ensemble.update(state, *_EVENT)
    compiled = ensemble.update(state, *_EVENT)

    for result in (eager, compiled):
        assert bool(result.diagnostics.applied)
        np.testing.assert_array_equal(result.bootstrap_mask, (True, True))
        np.testing.assert_array_equal(result.pre_event_words, (0, 0))
        np.testing.assert_array_equal(result.post_event_words, (0, 1))
        assert bool(result.event_counter_valid)
        assert bool(result.event_capacity_available)
        np.testing.assert_array_equal(result.member_pre_step_words, ((0, 0), (0, 0)))
        np.testing.assert_array_equal(result.member_post_step_words, ((0, 1), (0, 1)))
        np.testing.assert_array_equal(result.member_wrapper_counter_aligned, (True, True))
        np.testing.assert_array_equal(result.member_ensemble_counter_aligned, (True, True))
        np.testing.assert_array_equal(result.member_lifetime_counter_valid, (True, True))
        np.testing.assert_array_equal(
            result.member_lifetime_capacity_available,
            (True, True),
        )
        np.testing.assert_array_equal(result.member_updates_applied, (True, True))
        np.testing.assert_array_equal(result.state.event_count_words, (0, 1))
        np.testing.assert_array_equal(result.state.signal_state.step_words, (0, 1))
        np.testing.assert_array_equal(result.state.signal_state.valid_words, (0, 1))
        np.testing.assert_array_equal(
            result.state.member_update_count_words,
            ((0, 1), (0, 1)),
        )
        for candidate, expected in zip(
            result.state.member_states,
            direct,
            strict=True,
        ):
            _assert_tree_numerically_close(candidate, expected.state)

    _assert_tree_numerically_close(eager, compiled)


def test_one_member_wrapper_mismatch_rejects_without_advancing_any_lane() -> None:
    ensemble = _ensemble()
    state = ensemble.init(jr.key(1)).replace(bootstrap_key=jr.key(0))
    corrupt_member = state.member_states[0].replace(
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )
    corrupt = state.replace(
        member_states=(corrupt_member, state.member_states[1])
    )

    with jax.disable_jit():
        eager = ensemble.update(corrupt, *_EVENT)
    compiled = ensemble.update(corrupt, *_EVENT)

    for result in (eager, compiled):
        assert not bool(result.diagnostics.state_valid)
        assert not bool(result.diagnostics.applied)
        np.testing.assert_array_equal(
            result.member_wrapper_counter_aligned,
            (False, True),
        )
        np.testing.assert_array_equal(result.member_updates_applied, (False, False))
        np.testing.assert_array_equal(result.pre_event_words, (0, 0))
        np.testing.assert_array_equal(result.post_event_words, (0, 0))
        _assert_tree_bit_equal(result.state, corrupt)


def test_scan_crosses_member_and_global_uint32_carry() -> None:
    ensemble = _ensemble()
    state = _long_horizon_state(ensemble)
    assert bool(ensemble.state_valid(state))

    def step(carry, unused):  # type: ignore[no-untyped-def]
        del unused
        result = ensemble.update(carry, *_EVENT)
        return result.state, (
            result.pre_event_words,
            result.post_event_words,
            result.member_pre_step_words,
            result.member_post_step_words,
            result.diagnostics.applied,
        )

    final_state, trace = jax.lax.scan(
        step,
        state,
        jnp.arange(1, dtype=jnp.int32),
    )
    pre_event, post_event, member_pre, member_post, applied = trace

    np.testing.assert_array_equal(applied, (True,))
    np.testing.assert_array_equal(pre_event, ((0, _UINT32_MAX),))
    np.testing.assert_array_equal(post_event, ((1, 0),))
    np.testing.assert_array_equal(
        member_pre,
        (((0, _UINT32_MAX), (0, _UINT32_MAX)),),
    )
    np.testing.assert_array_equal(member_post, (((1, 0), (1, 0)),))
    np.testing.assert_array_equal(final_state.event_count_words, (1, 0))
    np.testing.assert_array_equal(
        final_state.member_update_count_words,
        ((1, 0), (1, 0)),
    )
    assert int(final_state.event_count) == _INT32_MAX
    np.testing.assert_array_equal(
        final_state.member_update_counts,
        (_INT32_MAX, _INT32_MAX),
    )


def test_required_all_ones_member_refusal_rolls_back_compound_real_update() -> None:
    ensemble = _ensemble()
    state = ensemble.init(jr.key(2)).replace(bootstrap_key=jr.key(0))
    terminal = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    first = state.member_states[0].replace(
        learner_state=state.member_states[0].learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=terminal,
        ),
        observation_min=jnp.asarray((-1.0, -1.0), dtype=jnp.float32),
        observation_max=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        reward_min=jnp.asarray(-1.0, dtype=jnp.float32),
        reward_max=jnp.asarray(1.0, dtype=jnp.float32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=terminal,
    )
    state = state.replace(
        member_states=(first, state.member_states[1]),
        replay_event_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        replay_event_count_words=terminal,
        replay_member_update_counts=jnp.asarray(
            (_INT32_MAX, 0),
            dtype=jnp.int32,
        ),
        replay_member_update_count_words=jnp.asarray(
            ((_UINT32_MAX, _UINT32_MAX), (0, 0)),
            dtype=jnp.uint32,
        ),
    )
    assert bool(ensemble.state_valid(state))

    result = ensemble.update(state, *_EVENT)

    assert not bool(result.diagnostics.member_updates_valid)
    assert not bool(result.diagnostics.applied)
    np.testing.assert_array_equal(
        result.member_lifetime_capacity_available,
        (False, True),
    )
    np.testing.assert_array_equal(result.member_updates_applied, (False, False))
    np.testing.assert_array_equal(result.post_event_words, state.event_count_words)
    np.testing.assert_array_equal(
        result.member_post_step_words,
        np.stack(
            [member.learner_state.step_words for member in state.member_states]
        ),
    )
    _assert_tree_bit_equal(result.state, state)


def test_replay_uses_isolated_exact_cadence_and_preserves_real_lane() -> None:
    ensemble = _ensemble()
    state = ensemble.init(jr.key(3)).replace(replay_bootstrap_key=jr.key(0))

    result = ensemble.replay_update(
        state,
        *_EVENT,
        jnp.asarray(True),
    )

    assert bool(result.diagnostics.applied)
    np.testing.assert_array_equal(result.bootstrap_mask, (True, True))
    np.testing.assert_array_equal(result.pre_event_words, (0, 0))
    np.testing.assert_array_equal(result.post_event_words, (0, 1))
    np.testing.assert_array_equal(result.state.replay_event_count_words, (0, 1))
    np.testing.assert_array_equal(result.state.event_count_words, (0, 0))
    np.testing.assert_array_equal(
        result.state.replay_member_update_count_words,
        ((0, 1), (0, 1)),
    )
    np.testing.assert_array_equal(
        result.state.member_update_count_words,
        ((0, 0), (0, 0)),
    )
    _assert_tree_bit_equal(result.state.signal_state, state.signal_state)
    _assert_tree_bit_equal(result.state.residual_variances, state.residual_variances)
    _assert_tree_bit_equal(result.state.bootstrap_key, state.bootstrap_key)


def test_saturated_wrapper_word_drift_rejects_ensemble_transaction() -> None:
    ensemble = _ensemble()
    state = _long_horizon_state(ensemble)
    first = state.member_states[0].replace(
        step_words=jnp.asarray((0, _INT32_MAX), dtype=jnp.uint32)
    )
    corrupt = state.replace(member_states=(first, state.member_states[1]))

    assert not bool(ensemble.state_valid(corrupt))
    result = ensemble.update(corrupt, *_EVENT)

    assert not bool(result.diagnostics.state_valid)
    np.testing.assert_array_equal(
        result.member_wrapper_counter_aligned,
        (False, True),
    )
    np.testing.assert_array_equal(result.member_updates_applied, (False, False))
    _assert_tree_bit_equal(result.state, corrupt)


def test_nested_normalizer_status_drives_predict_rejection_and_diagnostics(
    tmp_path: Path,
) -> None:
    ensemble = _ensemble_with_normalizer(EMANormalizer(decay=0.9))
    state = ensemble.init(jr.key(20))
    first = state.member_states[0]
    normalizer_state = first.learner_state.normalizer_state
    assert normalizer_state is not None
    first = first.replace(
        learner_state=first.learner_state.replace(
            normalizer_state=normalizer_state.replace(
                sample_count=jnp.asarray(1, dtype=jnp.int32),
                sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            )
        )
    )
    corrupt = state.replace(member_states=(first, state.member_states[1]))

    assert not bool(ensemble.state_valid(corrupt))
    prediction = ensemble.predict(corrupt, _EVENT[0], _EVENT[1])
    assert not bool(prediction.valid)
    np.testing.assert_array_equal(prediction.member_raw_predictions, 0.0)

    real = ensemble.update(corrupt, *_EVENT)
    replay = ensemble.replay_update(corrupt, *_EVENT, jnp.asarray(True))
    for result in (real, replay):
        np.testing.assert_array_equal(
            result.member_normalizer_counter_aligned,
            (False, True),
        )
        np.testing.assert_array_equal(
            result.member_normalizer_estimator_capacity_available,
            (True, True),
        )
        assert not bool(result.diagnostics.state_valid)
        assert not bool(result.diagnostics.applied)
        _assert_tree_bit_equal(result.state, corrupt)

    with pytest.raises(ValueError, match="cannot save an invalid"):
        save_world_model_ensemble_checkpoint(
            ensemble,
            corrupt,
            tmp_path / "normalizer-corrupt",
        )


def test_normalizer_estimator_horizon_is_valid_but_reports_real_capacity() -> None:
    ensemble = _ensemble_with_normalizer(EMANormalizer(decay=1.0))
    state = ensemble.init(jr.key(21)).replace(bootstrap_key=jr.key(0))
    horizon = 2**24
    words = jnp.asarray((0, horizon), dtype=jnp.uint32)
    members = []
    for member in state.member_states:
        normalizer_state = member.learner_state.normalizer_state
        assert normalizer_state is not None
        members.append(
            member.replace(
                learner_state=member.learner_state.replace(
                    step_count=jnp.asarray(horizon, dtype=jnp.int32),
                    step_words=words,
                    normalizer_state=normalizer_state.replace(
                        sample_count=jnp.asarray(horizon, dtype=jnp.int32),
                        sample_count_words=words,
                    ),
                ),
                observation_min=jnp.asarray((-1.0, -1.0), dtype=jnp.float32),
                observation_max=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
                reward_min=jnp.asarray(-1.0, dtype=jnp.float32),
                reward_max=jnp.asarray(1.0, dtype=jnp.float32),
                step_count=jnp.asarray(horizon, dtype=jnp.int32),
                step_words=words,
            )
        )
    signal_state = state.signal_state.replace(
        step_count=jnp.asarray(horizon, dtype=jnp.int32),
        valid_count=jnp.asarray(horizon, dtype=jnp.int32),
        step_words=words,
        valid_words=words,
    )
    state = state.replace(
        member_states=tuple(members),
        signal_state=signal_state,
        member_update_counts=jnp.full((2,), horizon, dtype=jnp.int32),
        member_update_count_words=jnp.tile(words[None, :], (2, 1)),
        event_count=jnp.asarray(horizon, dtype=jnp.int32),
        event_count_words=words,
    )

    assert bool(ensemble.state_valid(state))
    result = ensemble.update(state, *_EVENT)

    np.testing.assert_array_equal(
        result.member_normalizer_counter_aligned,
        (True, True),
    )
    np.testing.assert_array_equal(
        result.member_normalizer_estimator_capacity_available,
        (False, False),
    )
    assert not bool(result.diagnostics.member_updates_valid)
    assert not bool(result.diagnostics.applied)
    _assert_tree_bit_equal(result.state, state)


@pytest.mark.parametrize("corruption", ("signal-word-drift", "invalid-partition"))
def test_signal_exact_partition_must_bind_to_real_event_clock(
    corruption: str,
) -> None:
    ensemble = _ensemble()
    state = ensemble.init(jr.key(22))
    if corruption == "signal-word-drift":
        event_words = jnp.asarray((1, 0), dtype=jnp.uint32)
        signal_words = jnp.asarray((0, _INT32_MAX), dtype=jnp.uint32)
        signal_state = state.signal_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            valid_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=signal_words,
            valid_words=signal_words,
        )
        corrupt = state.replace(
            signal_state=signal_state,
            event_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            event_count_words=event_words,
        )
    else:
        one_word = jnp.asarray((0, 1), dtype=jnp.uint32)
        signal_state = state.signal_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32),
            invalid_count=jnp.asarray(1, dtype=jnp.int32),
            step_words=one_word,
            invalid_words=one_word,
        )
        corrupt = state.replace(
            signal_state=signal_state,
            event_count=jnp.asarray(1, dtype=jnp.int32),
            event_count_words=one_word,
        )

    signal_status = ensemble.signal_estimator.counter_status(corrupt.signal_state)
    assert bool(signal_status.lifetime_counter_valid)
    assert not bool(ensemble.state_valid(corrupt))
    prediction = ensemble.predict(corrupt, _EVENT[0], _EVENT[1])
    result = ensemble.update(corrupt, *_EVENT)
    assert not bool(prediction.valid)
    assert not bool(result.diagnostics.state_valid)
    _assert_tree_bit_equal(result.state, corrupt)
