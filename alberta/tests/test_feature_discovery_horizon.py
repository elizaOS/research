"""Exact long-horizon contracts for general fixed-budget feature discovery."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.feature_discovery import (
    FEATURE_DISCOVERY_CHECKPOINT_SCHEMA,
    FEATURE_DISCOVERY_LIFETIME_COUNTER_DELTA_NBYTES,
    FEATURE_DISCOVERY_LIFETIME_COUNTER_NBYTES,
    FEATURE_DISCOVERY_STATE_SCHEMA,
    FEATURE_DISCOVERY_TRANSACTION_CLOCK_DELTA_NBYTES,
    FEATURE_DISCOVERY_TRANSACTION_CLOCK_NBYTES,
    FeatureDiscoveryState,
    FixedBudgetFeatureLearner,
    feature_discovery_lifetime_counter_nbytes,
    feature_discovery_transaction_clock_nbytes,
    load_feature_discovery_checkpoint,
    measure_feature_discovery_state_nbytes,
    migrate_legacy_feature_discovery_state,
    save_feature_discovery_checkpoint,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.5, -0.25), dtype=jnp.float32)
_TARGET = jnp.asarray((0.75,), dtype=jnp.float32)


def _learner(
    *,
    replacement_interval: int = 0,
    candidate_count: int = 0,
    min_feature_age: int = 0,
    candidate_min_age: int = 0,
) -> FixedBudgetFeatureLearner:
    return FixedBudgetFeatureLearner(
        n_features=1,
        n_tasks=1,
        candidate_count=candidate_count,
        replacement_interval=replacement_interval,
        min_feature_age=min_feature_age,
        candidate_min_age=candidate_min_age,
    )


def _near_rollover_state(
    learner: FixedBudgetFeatureLearner,
    *,
    replacement_phase: int = 0,
) -> FeatureDiscoveryState:
    return learner.init(2, jr.key(17)).replace(  # type: ignore[attr-defined,no-any-return]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        replacement_phase=jnp.asarray(replacement_phase, dtype=jnp.int32),
    )


def test_low_word_rollover_is_exact_under_eager_jit_and_scan() -> None:
    learner = _learner()
    state = _near_rollover_state(learner)

    with jax.disable_jit():
        eager = learner.update(state, _OBSERVATION, _TARGET)
    compiled = jax.jit(learner.update)(state, _OBSERVATION, _TARGET)

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
        np.testing.assert_array_equal(result.post_step_words, (1, 0))
        assert int(result.state.step_count) == _INT32_MAX
    chex.assert_trees_all_equal(eager.state, compiled.state)

    def step(
        carry: FeatureDiscoveryState,
        _: jax.Array,
    ) -> tuple[FeatureDiscoveryState, jax.Array]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, result.post_step_words

    final_state, words = jax.lax.scan(
        step,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        words,
        np.asarray(((1, 0), (1, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(final_state.step_words, (1, 1))


def test_nondivisor_replacement_phase_survives_low_word_carry() -> None:
    learner = _learner(
        replacement_interval=7,
        min_feature_age=_INT32_MAX,
    )
    # 2**32 - 1 == 3 (mod 7), so the fourth accepted update is due.
    state = _near_rollover_state(learner, replacement_phase=3)

    def step(
        carry: FeatureDiscoveryState,
        _: jax.Array,
    ) -> tuple[FeatureDiscoveryState, tuple[jax.Array, jax.Array]]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, (
            result.state.replacement_phase,
            result.curation_attempted,
        )

    final_state, (phases, attempted) = jax.lax.scan(
        step,
        state,
        jnp.arange(5, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(phases, (4, 5, 6, 0, 1))
    np.testing.assert_array_equal(attempted, (False, False, False, True, False))
    np.testing.assert_array_equal(final_state.step_words, (1, 4))


def test_counter_and_phase_corruption_are_diagnosed_atomic_noops() -> None:
    learner = _learner(replacement_interval=3)
    state = learner.init(2, jr.key(2))
    corrupt_counter = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )
    counter_result = jax.jit(learner.update)(
        corrupt_counter,
        _OBSERVATION,
        _TARGET,
    )

    assert not bool(counter_result.lifetime_counter_valid)
    assert bool(counter_result.lifetime_capacity_available)
    assert not bool(counter_result.state_valid)
    assert not bool(counter_result.update_applied)
    assert bool(counter_result.update_rejected)
    chex.assert_trees_all_equal(counter_result.state, corrupt_counter)
    np.testing.assert_array_equal(
        counter_result.pre_step_words,
        counter_result.post_step_words,
    )

    for bad_phase in (1, 3):
        corrupt_phase = state.replace(  # type: ignore[attr-defined]
            replacement_phase=jnp.asarray(bad_phase, dtype=jnp.int32)
        )
        phase_result = learner.update(corrupt_phase, _OBSERVATION, _TARGET)
        assert bool(phase_result.lifetime_counter_valid)
        assert not bool(phase_result.state_valid)
        assert not bool(phase_result.update_applied)
        assert bool(phase_result.update_rejected)
        chex.assert_trees_all_equal(phase_result.state, corrupt_phase)


def test_maturity_ages_saturate_and_remain_eligible_for_promotion() -> None:
    learner = _learner(
        replacement_interval=1,
        candidate_count=1,
        min_feature_age=_INT32_MAX,
        candidate_min_age=_INT32_MAX,
    )
    state = learner.init(2, jr.key(8)).replace(  # type: ignore[attr-defined]
        ages=jnp.asarray((_INT32_MAX,), dtype=jnp.int32),
        candidate_ages=jnp.asarray((_INT32_MAX,), dtype=jnp.int32),
        utilities=jnp.asarray((0.0,), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((1.0,), dtype=jnp.float32),
    )

    result = learner.update(state, _OBSERVATION, _TARGET)

    assert bool(result.update_applied)
    assert bool(result.curation_attempted)
    assert int(result.promoted_candidate) == 0
    assert int(result.replaced_slot) == 0
    # Both mature slots were eligible at INT32_MAX and their new identities reset.
    assert int(result.state.ages[0]) == 0
    assert int(result.state.candidate_ages[0]) == 0


def test_invalid_unused_source_leaves_are_atomic_noops() -> None:
    learner = _learner(candidate_count=1)
    state = learner.init(2, jr.key(31))
    invalid_states = (
        state.replace(  # type: ignore[attr-defined]
            plasticity_log_weights=jnp.asarray((0.0, jnp.nan, 0.0), dtype=jnp.float32),
        ),
        state.replace(  # type: ignore[attr-defined]
            candidate_utility_contribution_trace=jnp.asarray(
                ((jnp.nan,),),
                dtype=jnp.float32,
            ),
        ),
        state.replace(  # type: ignore[attr-defined]
            candidate_generator=jnp.asarray((3,), dtype=jnp.int32),
        ),
    )

    for invalid in invalid_states:
        result = learner.update(invalid, _OBSERVATION, _TARGET)
        assert not bool(result.state_valid)
        assert not bool(result.update_applied)
        assert bool(result.update_rejected)
        np.testing.assert_array_equal(result.pre_step_words, result.post_step_words)
        assert result.state is not None
        for actual, expected in zip(
            jax.tree.leaves(result.state),
            jax.tree.leaves(invalid),
            strict=True,
        ):
            if jnp.issubdtype(actual.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jr.key_data(actual),
                    jr.key_data(expected),
                )
            else:
                np.testing.assert_array_equal(actual, expected)


def test_nonfinite_candidate_is_rejected_after_finite_source() -> None:
    learner = _learner()
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    state = learner.init(2, jr.key(32)).replace(  # type: ignore[attr-defined]
        feature_weights=jnp.zeros((1, 2), dtype=jnp.float32),
        feature_biases=jnp.asarray((10.0,), dtype=jnp.float32),
        output_weights=jnp.asarray(((maximum,),), dtype=jnp.float32),
    )

    result = learner.update(
        state,
        _OBSERVATION,
        jnp.asarray((-maximum,), dtype=jnp.float32),
    )

    assert bool(result.state_valid)
    assert not bool(result.candidate_state_valid)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)


def test_all_ones_exhaustion_is_a_diagnosed_atomic_noop() -> None:
    learner = _learner()
    state = learner.init(2, jr.key(3)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
    )

    result = learner.update(state, _OBSERVATION, _TARGET)

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)


def test_learned_replacement_accumulator_stays_a_bounded_residual() -> None:
    learner = FixedBudgetFeatureLearner(
        n_features=1,
        n_tasks=1,
        replacement_interval=1,
        min_feature_age=_INT32_MAX,
        learn_feature_resources=True,
        resource_learning_rate=0.0,
        resource_exploration=0.0,
    )
    state = learner.init(2, jr.key(41)).replace(  # type: ignore[attr-defined]
        plasticity_log_weights=jnp.asarray((-20.0, -20.0, 20.0), dtype=jnp.float32)
    )

    def step(
        carry: FeatureDiscoveryState,
        _: jax.Array,
    ) -> tuple[FeatureDiscoveryState, jax.Array]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, result.state.replacement_accumulator

    final_state, residuals = jax.lax.scan(
        step,
        state,
        jnp.arange(256, dtype=jnp.int32),
    )

    assert bool(jnp.all(residuals >= 0.0))
    assert bool(jnp.all(residuals < 1.0))
    assert 0.0 <= float(final_state.replacement_accumulator) < 1.0


def test_schema_migration_and_resource_accounting_are_exact() -> None:
    learner = _learner(replacement_interval=7, candidate_count=1)
    state = learner.init(2, jr.key(6))
    config = learner.to_config()
    assert config["state_schema"] == FEATURE_DISCOVERY_STATE_SCHEMA
    assert FixedBudgetFeatureLearner.from_config(config).to_config() == config
    with pytest.raises(ValueError, match="state schema"):
        FixedBudgetFeatureLearner.from_config(
            {**config, "state_schema": "alberta.feature-discovery-state.v1"}
        )

    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)  # type: ignore[arg-type]
        if field.name not in {"step_words", "replacement_phase"}
    }
    legacy["step_count"] = jnp.asarray(19, dtype=jnp.int32)
    legacy["ages"] = jnp.asarray((5,), dtype=jnp.int32)
    legacy["candidate_ages"] = jnp.asarray((7,), dtype=jnp.int32)
    legacy["birth_timestamp"] = float(state.birth_timestamp)
    legacy["uptime_s"] = float(state.uptime_s)
    migrated = migrate_legacy_feature_discovery_state(
        legacy,
        replacement_interval=7,
    )
    np.testing.assert_array_equal(migrated.step_words, (0, 19))
    assert int(migrated.replacement_phase) == 5
    assert migrated.birth_timestamp.shape == ()
    assert migrated.birth_timestamp.dtype == jnp.dtype(jnp.float32)
    assert migrated.uptime_s.shape == ()
    assert migrated.uptime_s.dtype == jnp.dtype(jnp.float32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_feature_discovery_state(
            {**legacy, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
            replacement_interval=7,
        )
    with pytest.raises(ValueError, match="negative.*step_count"):
        migrate_legacy_feature_discovery_state(
            {**legacy, "step_count": jnp.asarray(-1, dtype=jnp.int32)},
            replacement_interval=7,
        )
    with pytest.raises(ValueError, match="saturated.*ages"):
        migrate_legacy_feature_discovery_state(
            {**legacy, "ages": jnp.asarray((_INT32_MAX,), dtype=jnp.int32)},
            replacement_interval=7,
        )
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_feature_discovery_state(
            {**legacy, "extra": 1},
            replacement_interval=7,
        )

    without_new_clock = 0
    for field in dataclasses.fields(state):  # type: ignore[arg-type]
        value = getattr(state, field.name)
        if field.name not in {"step_words", "replacement_phase"} and isinstance(
            value,
            jax.Array,
        ):
            without_new_clock += int(value.size) * int(value.dtype.itemsize)
    assert measure_feature_discovery_state_nbytes(state) == (
        without_new_clock + FEATURE_DISCOVERY_TRANSACTION_CLOCK_DELTA_NBYTES
    )
    accounting = learner.memory_accounting(state)
    assert accounting["persistent_array_bytes"] == measure_feature_discovery_state_nbytes(
        state
    )
    assert accounting["lifetime_counter_bytes"] == 12
    assert accounting["replacement_phase_bytes"] == 4
    assert accounting["transaction_clock_bytes"] == 16
    assert feature_discovery_lifetime_counter_nbytes() == 12
    assert feature_discovery_transaction_clock_nbytes() == 16
    assert FEATURE_DISCOVERY_LIFETIME_COUNTER_NBYTES == 12
    assert FEATURE_DISCOVERY_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert FEATURE_DISCOVERY_TRANSACTION_CLOCK_NBYTES == 16
    assert FEATURE_DISCOVERY_TRANSACTION_CLOCK_DELTA_NBYTES == 12


def test_checkpoint_v2_resumes_exactly_and_v1_is_rejected(tmp_path: Path) -> None:
    learner = _learner(replacement_interval=7, candidate_count=1)
    state = _near_rollover_state(learner, replacement_phase=3)
    path = tmp_path / "current"
    save_feature_discovery_checkpoint(learner, state, path, feature_dim=2)

    loaded_learner, loaded_state = load_feature_discovery_checkpoint(path)
    chex.assert_trees_all_equal(loaded_state, state)
    direct = learner.update(state, _OBSERVATION, _TARGET)
    resumed = loaded_learner.update(loaded_state, _OBSERVATION, _TARGET)
    chex.assert_trees_all_equal(resumed, direct)
    assert FEATURE_DISCOVERY_CHECKPOINT_SCHEMA.endswith(".v2")

    legacy_path = tmp_path / "legacy"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.feature-discovery-checkpoint.v1",
            "learner_config": learner.to_config(),
            "feature_dim": 2,
            "memory_accounting": learner.memory_accounting(state),
        },
    )
    with pytest.raises(ValueError, match="lacks exact step_words.*resave"):
        load_feature_discovery_checkpoint(legacy_path)
