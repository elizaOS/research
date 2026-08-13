# mypy: disable-error-code="call-arg,untyped-decorator"
"""Tests for causal temporal/context features."""

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.temporal_context import (
    TEMPORAL_CONTEXT_LIFETIME_COUNTER_DELTA_NBYTES,
    TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES,
    TEMPORAL_CONTEXT_STATE_SCHEMA,
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
    measure_temporal_context_state_nbytes,
    migrate_legacy_temporal_context_state,
    temporal_context_lifetime_counter_nbytes,
    transform_temporal_context_arrays,
)


def test_temporal_context_shapes_and_roundtrip() -> None:
    config = TemporalContextConfig(input_dim=3, periods=(5.0, 10.0))
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()

    features = featurizer.features(state, jnp.ones(3))

    assert config.output_dim() == 13
    chex.assert_shape(features, (13,))
    chex.assert_tree_all_finite(features)
    assert TemporalContextConfig.from_config(config.to_config()) == config


def test_temporal_context_step_is_causal() -> None:
    config = TemporalContextConfig(input_dim=2, ema_decay=0.5, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()
    observation = jnp.asarray([2.0, -2.0], dtype=jnp.float32)

    next_state, features = featurizer.step(state, observation)

    chex.assert_trees_all_close(features[:2], observation)
    chex.assert_trees_all_close(features[2:4], jnp.zeros(2))
    chex.assert_trees_all_close(features[4:6], observation)
    chex.assert_trees_all_close(next_state.observation_ema, observation * 0.5)
    assert int(next_state.step_count) == 1


def test_temporal_context_phase_products_expand_with_input() -> None:
    config = TemporalContextConfig(
        input_dim=2,
        include_phase_products=True,
        periods=(4.0,),
    )
    featurizer = TemporalContextFeaturizer(config)

    features = featurizer.features(featurizer.init(), jnp.asarray([3.0, -1.0]))

    assert config.output_dim() == 12
    chex.assert_shape(features, (12,))
    chex.assert_tree_all_finite(features)


def test_temporal_context_array_transform_is_jittable() -> None:
    config = TemporalContextConfig(input_dim=2, periods=(4.0,))
    featurizer = TemporalContextFeaturizer(config)
    observations = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    @jax.jit
    def run(initial_state: TemporalContextState):
        return transform_temporal_context_arrays(
            featurizer,
            observations,
            state=initial_state,
        )

    state, features = run(featurizer.init())

    chex.assert_shape(features, (2, config.output_dim()))
    assert int(state.step_count) == 2
    chex.assert_tree_all_finite(features)


def test_phase_uses_exact_words_beyond_float32_integer_alias_boundary() -> None:
    featurizer = TemporalContextFeaturizer(
        TemporalContextConfig(
            input_dim=1,
            include_ema=False,
            include_delta=False,
            periods=(5.0,),
        )
    )
    observation = jnp.asarray([2.0], dtype=jnp.float32)
    large_step = 2**24 + 1
    large = featurizer.init().replace(
        step_count=jnp.asarray(large_step, dtype=jnp.int32),
        step_words=jnp.asarray((0, large_step), dtype=jnp.uint32),
    )
    aliased_predecessor = large.replace(
        step_count=jnp.asarray(large_step - 1, dtype=jnp.int32),
        step_words=jnp.asarray((0, large_step - 1), dtype=jnp.uint32)
    )
    short = featurizer.init().replace(
        step_count=jnp.asarray(large_step % 5, dtype=jnp.int32),
        step_words=jnp.asarray((0, large_step % 5), dtype=jnp.uint32),
    )

    exact = jax.jit(featurizer.features)(large, observation)
    predecessor = featurizer.features(aliased_predecessor, observation)
    reference = featurizer.features(short, observation)

    chex.assert_trees_all_equal(exact, reference)
    assert bool(jnp.any(exact != predecessor))


def test_exact_update_carries_and_terminal_or_corrupt_state_is_a_full_noop() -> None:
    featurizer = TemporalContextFeaturizer(
        TemporalContextConfig(input_dim=1, periods=(4.0,))
    )
    near_carry = featurizer.init().replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        step_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )
    carried = jax.jit(featurizer.update_result)(
        near_carry,
        jnp.asarray([1.0], dtype=jnp.float32),
    )
    assert bool(carried.update_applied)
    assert int(carried.state.step_count) == 2**31 - 1
    chex.assert_trees_all_equal(
        carried.state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )

    terminal = carried.state.replace(
        step_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32)
    )
    stopped = featurizer.update_result(
        terminal,
        jnp.asarray([9.0], dtype=jnp.float32),
    )
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    chex.assert_trees_all_equal(stopped.state, terminal)

    corrupt = carried.state.replace(
        step_words=jnp.asarray((0, 1), dtype=jnp.uint32)
    )
    rejected = featurizer.update_result(
        corrupt,
        jnp.asarray([9.0], dtype=jnp.float32),
    )
    assert not bool(rejected.state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, corrupt)
    chex.assert_trees_all_equal(
        featurizer.features(corrupt, jnp.asarray([9.0], dtype=jnp.float32)),
        jnp.zeros((featurizer.config.output_dim(),), dtype=jnp.float32),
    )


def test_schema_resources_migration_and_period_domain_are_explicit() -> None:
    featurizer = TemporalContextFeaturizer(
        TemporalContextConfig(input_dim=3, periods=(4.0, 7.0))
    )
    config = featurizer.config.to_config()
    assert config["state_schema"] == TEMPORAL_CONTEXT_STATE_SCHEMA
    assert TemporalContextConfig.from_config(config).to_config() == config
    with pytest.raises(ValueError, match="manifest is not exact"):
        TemporalContextConfig.from_config(
            {key: value for key, value in config.items() if key != "state_schema"}
        )
    for period in (1.5, float("inf"), float(2**24 + 1)):
        with pytest.raises(ValueError, match="exact positive float32 integers"):
            TemporalContextFeaturizer(
                TemporalContextConfig(input_dim=1, periods=(period,))
            )

    budget = featurizer.resource_budget()
    state = featurizer.init()
    assert budget.state_bytes == measure_temporal_context_state_nbytes(state)
    assert budget.exact_lifetime_counter_bytes == (
        TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES
    )
    assert temporal_context_lifetime_counter_nbytes() == (
        TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES
    )
    assert TEMPORAL_CONTEXT_LIFETIME_COUNTER_DELTA_NBYTES == 8

    migrated = migrate_legacy_temporal_context_state(
        {
            "observation_ema": jnp.ones((3,), dtype=jnp.float32),
            "step_count": jnp.asarray(7, dtype=jnp.int32),
        }
    )
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 7), dtype=jnp.uint32),
    )
    assert bool(featurizer.state_valid(migrated))
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_temporal_context_state(
            {
                "observation_ema": jnp.ones((3,), dtype=jnp.float32),
                "step_count": jnp.asarray(2**31 - 1, dtype=jnp.int32),
            }
        )
