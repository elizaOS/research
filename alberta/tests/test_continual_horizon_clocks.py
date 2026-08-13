"""Focused long-horizon clock contracts for normalizers and world learners."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint
from alberta_framework.core.latent_world_model import (
    LATENT_WORLD_MODEL_CONFIG_SCHEMA,
    LATENT_WORLD_MODEL_STATE_SCHEMA,
    LatentWorldModel,
    LatentWorldModelConfig,
    latent_world_model_lifetime_counter_nbytes,
    latent_world_model_wrapper_state_nbytes_formula,
    measure_latent_world_model_wrapper_state_nbytes,
    migrate_legacy_latent_world_model_state,
    run_latent_world_model_learning_loop,
)
from alberta_framework.core.multi_head_learner import (
    MULTI_HEAD_MLP_STATE_SCHEMA,
    MultiHeadMLPLearner,
    measure_multi_head_mlp_state_nbytes,
    migrate_legacy_multi_head_mlp_state,
    multi_head_lifetime_counter_nbytes,
    run_multi_head_learning_loop,
)
from alberta_framework.core.normalizers import (
    BOUNDED_RECENCY_ESTIMATOR_SEMANTICS,
    CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS,
    NORMALIZER_STATE_SCHEMA,
    STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS,
    WELFORD_ESTIMATOR_SCHEMA,
    EMANormalizer,
    StreamingBatchNormalizer,
    WelfordNormalizer,
    measure_normalizer_state_nbytes,
    migrate_legacy_normalizer_state,
    normalizer_from_config,
    normalizer_state_nbytes_formula,
)

pytestmark = pytest.mark.unit

_FLOAT32_INTEGER_LIMIT = 2**24
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)


def _assert_persistent_array_tree_bit_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        # MultiHeadMLPState's pre-existing host-only timing floats are coerced
        # by JIT and are outside this persistent learning/counter contract.
        if not isinstance(first_leaf, jax.Array) or not isinstance(
            second_leaf,
            jax.Array,
        ):
            continue
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


@pytest.mark.parametrize(
    ("normalizer", "expected_bytes"),
    [
        (EMANormalizer(decay=0.9), 8 * 3 + 16),
        (WelfordNormalizer(), 12 * 3 + 12),
        (StreamingBatchNormalizer(momentum=0.9), 8 * 3 + 16),
    ],
)
def test_normalizer_counter_init_and_resource_formula_are_exact(
    normalizer,
    expected_bytes: int,
) -> None:
    state = normalizer.init(3)
    assert state.sample_count.dtype == jnp.dtype(jnp.int32)
    assert state.sample_count_words.dtype == jnp.dtype(jnp.uint32)
    np.testing.assert_array_equal(state.sample_count_words, (0, 0))
    assert normalizer_state_nbytes_formula(type(normalizer).__name__, 3) == expected_bytes
    assert measure_normalizer_state_nbytes(state) == expected_bytes


@pytest.mark.parametrize(
    "normalizer",
    [
        EMANormalizer(decay=0.9),
        WelfordNormalizer(),
        StreamingBatchNormalizer(momentum=0.9),
    ],
)
def test_normalizer_statistics_preserve_legacy_early_float32_trajectory(
    normalizer,
) -> None:
    observations = (
        _OBSERVATION,
        -_OBSERVATION,
        jnp.asarray((1.0, 0.75), dtype=jnp.float32),
    )
    state = normalizer.init(2)
    legacy_mean = state.mean
    legacy_var = state.var
    legacy_count = jnp.asarray(0.0, dtype=jnp.float32)
    legacy_p = jnp.zeros_like(state.mean)

    for observation in observations:
        new_count = legacy_count + 1.0
        if isinstance(normalizer, EMANormalizer):
            effective_decay = jnp.minimum(
                state.decay,
                1.0 - 1.0 / (new_count + 1.0),
            )
            delta = observation - legacy_mean
            next_mean = legacy_mean + (1.0 - effective_decay) * delta
            delta2 = observation - next_mean
            next_var = jnp.maximum(
                effective_decay * legacy_var + (1.0 - effective_decay) * delta * delta2,
                normalizer._epsilon,
            )
        elif isinstance(normalizer, WelfordNormalizer):
            delta = observation - legacy_mean
            next_mean = legacy_mean + delta / new_count
            delta2 = observation - next_mean
            legacy_p = legacy_p + delta * delta2
            next_var = jnp.where(
                new_count >= 2.0,
                legacy_p / (new_count - 1.0),
                jnp.ones_like(legacy_p),
            )
        else:
            is_first = legacy_count <= 0.0
            one_minus_m = 1.0 - state.momentum
            next_mean = jnp.where(
                is_first,
                observation,
                state.momentum * legacy_mean + one_minus_m * observation,
            )
            centered = observation - legacy_mean
            next_var = jnp.maximum(
                jnp.where(
                    is_first,
                    jnp.ones_like(legacy_var),
                    state.momentum * legacy_var + one_minus_m * centered * centered,
                ),
                normalizer._epsilon,
            )
        expected_normalized = (observation - next_mean) / (jnp.sqrt(next_var) + normalizer._epsilon)
        with jax.disable_jit():
            normalized, state = normalizer.normalize(state, observation)
        np.testing.assert_array_equal(normalized, expected_normalized)
        np.testing.assert_array_equal(state.mean, next_mean)
        np.testing.assert_array_equal(state.var, next_var)
        legacy_mean = next_mean
        legacy_var = next_var
        legacy_count = new_count


@pytest.mark.parametrize(
    "normalizer",
    [EMANormalizer(decay=0.9), StreamingBatchNormalizer(momentum=0.9)],
    ids=("ema-recency", "streaming-recency"),
)
def test_recency_normalizers_cross_float32_freeze_and_uint32_carry(normalizer) -> None:
    state = normalizer.init(2).replace(
        sample_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
        sample_count_words=jnp.asarray((0, _FLOAT32_INTEGER_LIMIT), dtype=jnp.uint32),
    )
    compiled = jax.jit(normalizer.normalize_with_diagnostics)
    result = compiled(state, _OBSERVATION)
    assert bool(result.update_applied)
    assert int(result.state.sample_count) == _FLOAT32_INTEGER_LIMIT + 1
    np.testing.assert_array_equal(
        result.state.sample_count_words,
        (0, _FLOAT32_INTEGER_LIMIT + 1),
    )

    near_carry = result.state.replace(
        sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        sample_count_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    carry = compiled(near_carry, _OBSERVATION)
    assert int(carry.state.sample_count) == _INT32_MAX
    np.testing.assert_array_equal(carry.state.sample_count_words, (1, 0))


@pytest.mark.parametrize(
    "normalizer",
    [WelfordNormalizer(), EMANormalizer(decay=1.0)],
    ids=("welford", "ema-cumulative"),
)
def test_float32_cumulative_estimators_accept_boundary_then_fail_stop(normalizer) -> None:
    initial = normalizer.init(2).replace(
        sample_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT - 1, dtype=jnp.int32),
        sample_count_words=jnp.asarray(
            (0, _FLOAT32_INTEGER_LIMIT - 1),
            dtype=jnp.uint32,
        ),
    )
    with jax.disable_jit():
        eager_boundary = normalizer.normalize_with_diagnostics(
            initial,
            _OBSERVATION,
        )
    compiled_boundary = jax.jit(normalizer.normalize_with_diagnostics)(
        initial,
        _OBSERVATION,
    )
    for boundary in (eager_boundary, compiled_boundary):
        assert bool(boundary.update_applied)
        assert bool(boundary.estimator_capacity_available)
        np.testing.assert_array_equal(
            boundary.state.sample_count_words,
            (0, _FLOAT32_INTEGER_LIMIT),
        )
    _assert_persistent_array_tree_bit_equal(
        eager_boundary.state,
        compiled_boundary.state,
    )

    def scan_step(state, observation):
        result = normalizer.normalize_with_diagnostics(state, observation)
        return result.state, (
            result.update_applied,
            result.estimator_capacity_available,
            result.post_sample_count_words,
        )

    final_state, (applied, estimator_capacity, words) = jax.lax.scan(
        scan_step,
        initial,
        jnp.stack((_OBSERVATION, _OBSERVATION)),
    )
    np.testing.assert_array_equal(applied, (True, False))
    np.testing.assert_array_equal(estimator_capacity, (True, False))
    np.testing.assert_array_equal(
        words,
        (
            (0, _FLOAT32_INTEGER_LIMIT),
            (0, _FLOAT32_INTEGER_LIMIT),
        ),
    )
    assert int(final_state.sample_count) == _FLOAT32_INTEGER_LIMIT

    terminal = normalizer.normalize_with_diagnostics(final_state, -_OBSERVATION)
    assert not bool(terminal.update_applied)
    _assert_persistent_array_tree_bit_equal(terminal.state, final_state)


def test_normalizer_all_ones_is_diagnosed_bit_exact_noop_eager_and_jit() -> None:
    normalizer = EMANormalizer(decay=0.9)
    state = normalizer.init(2).replace(
        sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        sample_count_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
    )
    with jax.disable_jit():
        eager = normalizer.normalize_with_diagnostics(state, _OBSERVATION)
    compiled = jax.jit(normalizer.normalize_with_diagnostics)(state, _OBSERVATION)
    for result in (eager, compiled):
        assert bool(result.counter_valid)
        assert not bool(result.lifetime_capacity_available)
        assert not bool(result.update_applied)
        _assert_persistent_array_tree_bit_equal(result.state, state)


def test_normalizer_config_modes_and_strict_scalars() -> None:
    ema_recency = EMANormalizer(decay=0.9).to_config()
    ema_cumulative = EMANormalizer(decay=1.0).to_config()
    static_batch = StreamingBatchNormalizer(momentum=1.0).to_config()
    rounded_cumulative = EMANormalizer(decay=1.0 - 1e-10).to_config()
    rounded_static = StreamingBatchNormalizer(momentum=1.0 - 1e-10).to_config()
    assert ema_recency["state_schema"] == NORMALIZER_STATE_SCHEMA
    assert ema_recency["estimator_semantics"] == BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
    assert ema_cumulative["estimator_semantics"] == CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
    assert static_batch["estimator_semantics"] == STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS
    assert rounded_cumulative["estimator_semantics"] == CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
    assert rounded_static["estimator_semantics"] == STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS
    assert normalizer_from_config(ema_cumulative).to_config() == ema_cumulative
    welford = WelfordNormalizer().to_config()
    assert welford["estimator_schema"] == WELFORD_ESTIMATOR_SCHEMA
    assert normalizer_from_config(welford).to_config() == welford

    bad_semantics = dict(ema_cumulative)
    bad_semantics["estimator_semantics"] = BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
    with pytest.raises(ValueError, match="does not match"):
        normalizer_from_config(bad_semantics)
    for value in (True, "0.9", float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            EMANormalizer(decay=value)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            StreamingBatchNormalizer(momentum=value)  # type: ignore[arg-type]
    for value in (0.0, -1.0, True, "1e-8", float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            EMANormalizer(epsilon=value)  # type: ignore[arg-type]

    # Preflight derives semantics from persisted state, but it also binds that
    # state to the configured normalizer so a mismatched checkpoint cannot
    # silently switch estimator class or horizon.
    configured_recency = EMANormalizer(decay=0.9)
    tampered_cumulative = configured_recency.init(2).replace(
        sample_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
        sample_count_words=jnp.asarray(
            (0, _FLOAT32_INTEGER_LIMIT),
            dtype=jnp.uint32,
        ),
        decay=jnp.asarray(1.0, dtype=jnp.float32),
    )
    stopped = configured_recency.normalize_with_diagnostics(
        tampered_cumulative,
        _OBSERVATION,
    )
    assert not bool(stopped.counter_valid)
    assert not bool(stopped.estimator_capacity_available)
    assert not bool(stopped.update_applied)
    invalid = configured_recency.normalize_with_diagnostics(
        tampered_cumulative.replace(decay=jnp.asarray(jnp.nan, dtype=jnp.float32)),
        _OBSERVATION,
    )
    assert not bool(invalid.counter_valid)
    assert not bool(invalid.update_applied)


def test_legacy_normalizer_migration_rejects_ambiguous_float_count() -> None:
    state = EMANormalizer(decay=0.9).init(2)
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name != "sample_count_words"
    }
    legacy["sample_count"] = jnp.asarray(7.0, dtype=jnp.float32)
    migrated = migrate_legacy_normalizer_state(
        legacy,
        normalizer_type="EMANormalizer",
    )
    assert int(migrated.sample_count) == 7
    np.testing.assert_array_equal(migrated.sample_count_words, (0, 7))

    ambiguous = dict(legacy)
    ambiguous["sample_count"] = jnp.asarray(
        float(_FLOAT32_INTEGER_LIMIT),
        dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_normalizer_state(
            ambiguous,
            normalizer_type="EMANormalizer",
        )


def _small_multi_head(*, normalizer=None) -> MultiHeadMLPLearner:
    return MultiHeadMLPLearner(
        n_heads=1,
        hidden_sizes=(),
        normalizer=normalizer,
        step_size=0.0,
        sparsity=0.0,
        use_layer_norm=False,
    )


def test_multi_head_nested_clock_carry_and_atomic_refusal() -> None:
    normalizer = EMANormalizer(decay=0.9)
    learner = _small_multi_head(normalizer=normalizer)
    initial = learner.init(2, jr.key(1))
    assert initial.normalizer_state is not None
    near_carry = initial.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        normalizer_state=initial.normalizer_state.replace(
            sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            sample_count_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        ),
    )
    result = learner.update(
        near_carry,
        _OBSERVATION,
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.state.step_words, (1, 0))
    assert result.state.normalizer_state is not None
    np.testing.assert_array_equal(
        result.state.normalizer_state.sample_count_words,
        result.state.step_words,
    )
    assert int(result.state.step_count) == _INT32_MAX

    exhausted = result.state.replace(
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        normalizer_state=result.state.normalizer_state.replace(
            sample_count_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        ),
    )
    stopped = learner.update(
        exhausted,
        _OBSERVATION,
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert not bool(stopped.update_applied)
    assert bool(stopped.normalizer_counter_aligned)
    assert not bool(stopped.lifetime_capacity_available)
    _assert_persistent_array_tree_bit_equal(stopped.state, exhausted)

    misaligned = exhausted.replace(
        step_words=jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    rejected = learner.update(
        misaligned,
        _OBSERVATION,
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert not bool(rejected.lifetime_counter_valid)
    assert not bool(rejected.normalizer_counter_aligned)
    assert not bool(rejected.update_applied)
    _assert_persistent_array_tree_bit_equal(rejected.state, misaligned)


def test_multi_head_welford_horizon_refuses_whole_learner_transaction() -> None:
    learner = _small_multi_head(normalizer=WelfordNormalizer())
    state = learner.init(2, jr.key(2))
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
    result = learner.update(
        state,
        _OBSERVATION,
        jnp.asarray((1.0,), dtype=jnp.float32),
    )
    assert bool(result.lifetime_capacity_available)
    assert not bool(result.normalizer_estimator_capacity_available)
    assert not bool(result.update_applied)
    _assert_persistent_array_tree_bit_equal(result.state, state)


def test_multi_head_nonfinite_inputs_and_source_state_are_atomic_noops() -> None:
    learner = _small_multi_head()
    state = learner.init(2, jr.key(20))

    for observation, targets in (
        (
            jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32),
            jnp.asarray((0.0,), dtype=jnp.float32),
        ),
        (
            _OBSERVATION,
            jnp.asarray((jnp.inf,), dtype=jnp.float32),
        ),
    ):
        result = learner.update(state, observation, targets)
        assert not bool(result.update_applied)
        _assert_persistent_array_tree_bit_equal(result.state, state)

    corrupt = state.replace(
        head_params=state.head_params.replace(
            weights=(state.head_params.weights[0].at[0, 0].set(jnp.nan),)
        )
    )
    rejected = learner.update(
        corrupt,
        _OBSERVATION,
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert not bool(rejected.update_applied)
    _assert_persistent_array_tree_bit_equal(rejected.state, corrupt)


def test_multi_head_scan_reports_terminal_refusals() -> None:
    learner = _small_multi_head()
    state = learner.init(2, jr.key(21)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
    )
    result = run_multi_head_learning_loop(
        learner,
        state,
        jnp.ones((3, 2), dtype=jnp.float32),
        jnp.zeros((3, 1), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(result.updates_applied, (False, False, False))
    _assert_persistent_array_tree_bit_equal(result.state, state)


def test_multi_head_migration_config_resources_and_checkpoint(tmp_path) -> None:
    learner = _small_multi_head()
    state = learner.init(2, jr.key(3)).replace(
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=jnp.asarray((0, 7), dtype=jnp.uint32),
    )
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name != "step_words"
    }
    migrated = migrate_legacy_multi_head_mlp_state(legacy)
    np.testing.assert_array_equal(migrated.step_words, (0, 7))
    assert learner.to_config()["state_schema"] == MULTI_HEAD_MLP_STATE_SCHEMA
    assert MultiHeadMLPLearner.from_config(learner.to_config()).to_config() == learner.to_config()
    assert multi_head_lifetime_counter_nbytes(has_normalizer=False) == 12
    assert measure_multi_head_mlp_state_nbytes(state) >= 12

    save_checkpoint(state, tmp_path / "multi-head-v2")
    template = learner.init(2, jr.key(4))
    loaded, _ = load_checkpoint(template, tmp_path / "multi-head-v2")
    np.testing.assert_array_equal(loaded.step_words, state.step_words)

    ambiguous = dict(legacy)
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_multi_head_mlp_state(ambiguous)

    normalized_learner = _small_multi_head(normalizer=EMANormalizer(decay=0.9))
    normalized_state = normalized_learner.init(2, jr.key(8))
    assert normalized_state.normalizer_state is not None
    normalized_state = normalized_state.replace(
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=jnp.asarray((0, 7), dtype=jnp.uint32),
        normalizer_state=normalized_state.normalizer_state.replace(
            sample_count=jnp.asarray(7, dtype=jnp.int32),
            sample_count_words=jnp.asarray((0, 7), dtype=jnp.uint32),
        ),
    )
    legacy_normalizer = {
        field.name: getattr(normalized_state.normalizer_state, field.name)
        for field in dataclasses.fields(type(normalized_state.normalizer_state))
        if field.name != "sample_count_words"
    }
    legacy_normalizer["sample_count"] = jnp.asarray(7.0, dtype=jnp.float32)
    legacy_normalized_state = {
        field.name: getattr(normalized_state, field.name)
        for field in dataclasses.fields(type(normalized_state))
        if field.name != "step_words"
    }
    legacy_normalized_state["normalizer_state"] = legacy_normalizer
    migrated_normalized = migrate_legacy_multi_head_mlp_state(legacy_normalized_state)
    assert migrated_normalized.normalizer_state is not None
    np.testing.assert_array_equal(
        migrated_normalized.normalizer_state.sample_count_words,
        migrated_normalized.step_words,
    )

    misaligned_normalizer = dict(legacy_normalizer)
    misaligned_normalizer["sample_count"] = jnp.asarray(6.0, dtype=jnp.float32)
    misaligned_legacy = dict(legacy_normalized_state)
    misaligned_legacy["normalizer_state"] = misaligned_normalizer
    with pytest.raises(ValueError, match="not aligned"):
        migrate_legacy_multi_head_mlp_state(misaligned_legacy)


def _small_world() -> LatentWorldModel:
    return LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            step_size=0.01,
            sparsity=0.0,
            min_latent_std=0.0,
        )
    )


_WORLD_ARGS = (
    _OBSERVATION,
    jnp.asarray(0, dtype=jnp.int32),
    jnp.asarray(0.5, dtype=jnp.float32),
    jnp.asarray(0.99, dtype=jnp.float32),
    -_OBSERVATION,
)


def test_lifetime_contracts_are_public_from_package_and_core() -> None:
    import alberta_framework as package_api
    import alberta_framework.core as core_api

    required = {
        "BEHAVIOR_MODEL_STATE_SCHEMA",
        "JOINT_OUTCOME_STATE_SCHEMA",
        "NORMALIZER_STATE_SCHEMA",
        "MULTI_HEAD_MLP_STATE_SCHEMA",
        "LATENT_WORLD_MODEL_STATE_SCHEMA",
        "migrate_legacy_behavior_model_state",
        "migrate_legacy_bounded_joint_outcome_state",
        "migrate_legacy_normalizer_state",
        "migrate_legacy_multi_head_mlp_state",
        "migrate_legacy_latent_world_model_state",
        "normalizer_state_nbytes_formula",
        "multi_head_lifetime_counter_nbytes",
        "latent_world_model_lifetime_counter_nbytes",
    }
    for module in (package_api, core_api):
        assert required <= set(module.__all__)
        assert all(hasattr(module, name) for name in required)


def test_latent_world_clock_carry_scan_and_all_ones_noop() -> None:
    model = _small_world()
    initial = model.init(jr.key(5))
    near_carry = initial.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        learner_state=initial.learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        ),
    )

    def scan_step(state, _):
        result = model.update(state, *_WORLD_ARGS)
        return result.state, (result.update_applied, result.post_step_words)

    scan_state, (applied, words) = jax.lax.scan(
        scan_step,
        near_carry,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(applied, (True, True))
    np.testing.assert_array_equal(words, ((1, 0), (1, 1)))
    assert int(scan_state.step_count) == _INT32_MAX
    assert int(scan_state.learner_state.step_count) == _INT32_MAX
    np.testing.assert_array_equal(
        scan_state.learner_state.step_words,
        scan_state.step_words,
    )

    exhausted = scan_state.replace(
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        learner_state=scan_state.learner_state.replace(
            step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        ),
    )
    stopped = model.update(exhausted, *_WORLD_ARGS)
    assert bool(stopped.lifetime_counter_valid)
    assert bool(stopped.learner_counter_aligned)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    assert not bool(stopped.learner_result.update_applied)
    assert not bool(stopped.encoder_update_applied)
    _assert_persistent_array_tree_bit_equal(stopped.state, exhausted)


def test_latent_world_propagates_child_and_input_refusals_atomically() -> None:
    model = _small_world()
    state = model.init(jr.key(22))
    invalid_transitions = (
        (
            jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32),
            _WORLD_ARGS[1],
            _WORLD_ARGS[2],
            _WORLD_ARGS[3],
            _WORLD_ARGS[4],
        ),
        (
            _WORLD_ARGS[0],
            jnp.asarray(2, dtype=jnp.int32),
            _WORLD_ARGS[2],
            _WORLD_ARGS[3],
            _WORLD_ARGS[4],
        ),
        (
            _WORLD_ARGS[0],
            _WORLD_ARGS[1],
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            _WORLD_ARGS[3],
            _WORLD_ARGS[4],
        ),
        (
            _WORLD_ARGS[0],
            _WORLD_ARGS[1],
            _WORLD_ARGS[2],
            jnp.asarray(1.1, dtype=jnp.float32),
            _WORLD_ARGS[4],
        ),
    )
    for transition in invalid_transitions:
        result = model.update(state, *transition)
        assert not bool(result.update_applied)
        assert not bool(result.learner_result.update_applied)
        assert not bool(result.encoder_update_applied)
        _assert_persistent_array_tree_bit_equal(result.state, state)

    corrupt = state.replace(
        encoder_matrix=state.encoder_matrix.at[0, 0].set(jnp.nan)
    )
    rejected = model.update(corrupt, *_WORLD_ARGS)
    assert not bool(rejected.update_applied)
    assert not bool(rejected.learner_result.update_applied)
    _assert_persistent_array_tree_bit_equal(rejected.state, corrupt)


def test_latent_world_scan_surfaces_terminal_refusals() -> None:
    model = _small_world()
    state = model.init(jr.key(23)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
    )
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=state.step_words,
        )
    )
    result = run_latent_world_model_learning_loop(
        model,
        state,
        jnp.ones((3, 2), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.float32),
        -jnp.ones((3, 2), dtype=jnp.float32),
        jnp.ones((3,), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(result.updates_applied, (False, False, False))
    _assert_persistent_array_tree_bit_equal(result.state, state)


def test_latent_world_misalignment_migration_config_and_resource_formula() -> None:
    model = _small_world()
    state = model.init(jr.key(6))
    misaligned = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32),
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    )
    rejected = model.update(misaligned, *_WORLD_ARGS)
    assert not bool(rejected.lifetime_counter_valid)
    assert not bool(rejected.learner_counter_aligned)
    assert not bool(rejected.update_applied)
    assert not bool(rejected.learner_result.update_applied)
    _assert_persistent_array_tree_bit_equal(rejected.state, misaligned)

    legacy_learner = {
        field.name: getattr(state.learner_state, field.name)
        for field in dataclasses.fields(type(state.learner_state))
        if field.name != "step_words"
    }
    legacy_world = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name != "step_words"
    }
    legacy_world["learner_state"] = legacy_learner
    migrated = migrate_legacy_latent_world_model_state(legacy_world)
    np.testing.assert_array_equal(migrated.step_words, (0, 0))
    np.testing.assert_array_equal(
        migrated.learner_state.step_words,
        migrated.step_words,
    )

    config = model.to_config()
    assert config["state_schema"] == LATENT_WORLD_MODEL_STATE_SCHEMA
    assert config["config"]["schema"] == LATENT_WORLD_MODEL_CONFIG_SCHEMA
    assert LatentWorldModel.from_config(config).to_config() == config
    expected_wrapper = latent_world_model_wrapper_state_nbytes_formula(model.config)
    assert measure_latent_world_model_wrapper_state_nbytes(state) == expected_wrapper
    assert latent_world_model_lifetime_counter_nbytes() == 24
