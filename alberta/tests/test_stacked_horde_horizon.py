"""Exact-horizon and atomicity contracts for the stacked linear Horde."""

from __future__ import annotations

import copy
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.stacked_horde import (
    STACKED_HORDE_CONFIG_SCHEMA,
    STACKED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES,
    STACKED_HORDE_LIFETIME_COUNTER_NBYTES,
    STACKED_HORDE_RESOURCE_SCHEMA,
    STACKED_HORDE_STATE_SCHEMA,
    StackedHordeConfig,
    StackedHordeResourceBudget,
    StackedLinearHorde,
    measure_stacked_horde_state_nbytes,
    migrate_legacy_stacked_horde_config,
    migrate_legacy_stacked_horde_state,
    run_stacked_horde_scan,
    stacked_horde_lifetime_counter_nbytes,
    stacked_horde_state_nbytes_formula,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _config(*, step_size: float = 0.1) -> StackedHordeConfig:
    return StackedHordeConfig(
        n_demons=2,
        feature_dim=2,
        gammas=(0.5, 0.9),
        lamdas=(0.4, 0.7),
        cumulant_indices=(0, 1),
        step_size=step_size,
    )


def _transition() -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray((1.0, 0.5), dtype=jnp.float32),
        jnp.asarray((0.25, 1.0), dtype=jnp.float32),
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
    )


def _assert_bit_exact(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes() == right_array.tobytes()


def test_exact_low_word_carry_and_saturating_telemetry() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init().replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    features, next_features, cumulants = _transition()

    result = horde.update(state, features, next_features, cumulants)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
    np.testing.assert_array_equal(result.post_step_words, (1, 0))
    assert int(result.state.step_count) == _INT32_MAX
    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)


def test_all_ones_terminal_clock_refuses_with_bit_exact_rollback() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init().replace(  # type: ignore[attr-defined]
        weights=jnp.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=jnp.float32),
        traces=jnp.asarray(((0.5, 0.25), (0.75, 1.0)), dtype=jnp.float32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
    )
    features, next_features, cumulants = _transition()

    result = jax.jit(horde.update)(state, features, next_features, cumulants)

    assert not bool(result.update_applied)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.lifetime_counter_valid)
    _assert_bit_exact(result.state, state)
    np.testing.assert_array_equal(result.post_step_words, state.step_words)


def test_corrupt_clock_and_nonfinite_state_fail_closed() -> None:
    horde = StackedLinearHorde(_config())
    features, next_features, cumulants = _transition()
    corrupt_clock = horde.init().replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(0, dtype=jnp.int32),
        step_words=jnp.asarray((0, 7), dtype=jnp.uint32),
    )
    clock_result = horde.update(
        corrupt_clock,
        features,
        next_features,
        cumulants,
    )
    assert not bool(clock_result.lifetime_counter_valid)
    assert not bool(clock_result.update_applied)
    _assert_bit_exact(clock_result.state, corrupt_clock)

    corrupt_weights = horde.init().replace(  # type: ignore[attr-defined]
        weights=horde.init().weights.at[1, 0].set(jnp.nan)
    )
    state_result = horde.update(
        corrupt_weights,
        features,
        next_features,
        cumulants,
    )
    assert not bool(state_result.state_valid)
    assert not bool(state_result.update_applied)
    _assert_bit_exact(state_result.state, corrupt_weights)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("features", jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32)),
        ("next_features", jnp.asarray((0.0, -jnp.inf), dtype=jnp.float32)),
        ("cumulants", jnp.asarray((1.0, jnp.inf), dtype=jnp.float32)),
        ("rho", jnp.asarray((1.0, -0.1), dtype=jnp.float32)),
    ],
)
def test_invalid_transition_sources_roll_back(field: str, bad_value: jax.Array) -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init()
    features, next_features, cumulants = _transition()
    values: dict[str, jax.Array | float] = {
        "features": features,
        "next_features": next_features,
        "cumulants": cumulants,
        "rho": 1.0,
    }
    values[field] = bad_value

    result = horde.update(
        state,
        cast(jax.Array, values["features"]),
        cast(jax.Array, values["next_features"]),
        cast(jax.Array, values["cumulants"]),
        values["rho"],
    )

    assert not bool(result.source_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)


def test_one_nonfinite_demon_candidate_rolls_back_every_demon() -> None:
    horde = StackedLinearHorde(_config(step_size=1.0))
    state = horde.init()
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    features = jnp.asarray((2.0, 0.0), dtype=jnp.float32)
    next_features = jnp.zeros((2,), dtype=jnp.float32)
    cumulants = jnp.asarray((1.0, maximum), dtype=jnp.float32)

    result = horde.update(state, features, next_features, cumulants)

    assert bool(result.source_valid)
    np.testing.assert_array_equal(result.per_demon_candidate_valid, (True, False))
    assert not bool(result.candidate_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)


def test_nan_inactive_demon_is_an_accepted_semantic_event() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init().replace(  # type: ignore[attr-defined]
        traces=jnp.ones((2, 2), dtype=jnp.float32),
    )
    features, next_features, _ = _transition()
    cumulants = jnp.asarray((jnp.nan, 1.0), dtype=jnp.float32)

    result = horde.update(state, features, next_features, cumulants)

    assert bool(result.source_valid)
    assert bool(result.update_applied)
    assert bool(jnp.isnan(result.td_errors[0]))
    np.testing.assert_allclose(
        result.state.traces[0],
        0.5 * 0.4 * state.traces[0],
    )
    np.testing.assert_array_equal(result.state.weights[0], state.weights[0])


def test_eager_and_jit_results_are_bit_exact() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init()
    features, next_features, cumulants = _transition()

    eager = horde.update(state, features, next_features, cumulants)
    compiled = jax.jit(horde.update)(state, features, next_features, cumulants)

    _assert_bit_exact(eager, compiled)


def test_scan_advances_exact_clock_across_low_word_carry() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init().replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX - 1), dtype=jnp.uint32),
    )
    features = jnp.asarray(
        ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)),
        dtype=jnp.float32,
    )
    cumulants = jnp.ones((4, 2), dtype=jnp.float32)

    final_state, td_errors = run_stacked_horde_scan(
        horde,
        state,
        features,
        cumulants,
    )

    np.testing.assert_array_equal(final_state.step_words, (1, 1))
    assert int(final_state.step_count) == _INT32_MAX
    assert td_errors.shape == (3, 2)

    terminal = state.replace(
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32)
    )
    refused_state, _ = run_stacked_horde_scan(
        horde,
        terminal,
        features,
        cumulants,
    )
    _assert_bit_exact(refused_state, terminal)


def test_strict_array_contracts_reject_shape_and_dtype_drift() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init()
    features, next_features, cumulants = _transition()

    with pytest.raises(TypeError, match="features.*float32"):
        horde.update(
            state,
            features.astype(jnp.int32),
            next_features,
            cumulants,
        )
    with pytest.raises(ValueError, match="cumulant_source"):
        horde.update(state, features, next_features, cumulants[:1])
    with pytest.raises(TypeError, match="rho arrays.*float32"):
        horde.update(
            state,
            features,
            next_features,
            cumulants,
            jnp.ones((2,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="rho must"):
        horde.update(
            state,
            features,
            next_features,
            cumulants,
            jnp.ones((3,), dtype=jnp.float32),
        )


def test_v2_config_schema_and_explicit_legacy_migration() -> None:
    cfg = _config()
    serialized = cfg.to_config()
    assert serialized["schema"] == STACKED_HORDE_CONFIG_SCHEMA
    assert serialized["state_schema"] == STACKED_HORDE_STATE_SCHEMA
    assert StackedHordeConfig.from_config(serialized) == cfg
    assert StackedLinearHorde.from_config(StackedLinearHorde(cfg).to_config()).config == cfg

    legacy = {
        key: value
        for key, value in serialized.items()
        if key not in {"schema", "state_schema"}
    }
    with pytest.raises(ValueError, match="manifest"):
        StackedHordeConfig.from_config(legacy)
    migrated = migrate_legacy_stacked_horde_config(legacy)
    assert StackedHordeConfig.from_config(migrated) == cfg

    corrupt = copy.deepcopy(serialized)
    corrupt["unknown"] = 1
    with pytest.raises(ValueError, match="manifest"):
        StackedHordeConfig.from_config(corrupt)


def test_state_migration_and_resource_accounting_are_exact() -> None:
    horde = StackedLinearHorde(_config())
    state = horde.init()
    legacy = {
        "weights": state.weights,
        "traces": state.traces,
        "step_count": jnp.asarray(17, dtype=jnp.int32),
    }
    migrated = migrate_legacy_stacked_horde_state(legacy, config=horde.config)
    np.testing.assert_array_equal(migrated.step_words, (0, 17))
    assert bool(horde.state_valid(migrated))

    saturated = dict(legacy)
    saturated["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_stacked_horde_state(saturated, config=horde.config)

    budget = horde.resource_budget
    assert budget.lifetime_counter_nbytes == STACKED_HORDE_LIFETIME_COUNTER_NBYTES
    assert stacked_horde_lifetime_counter_nbytes() == (
        STACKED_HORDE_LIFETIME_COUNTER_NBYTES
    )
    assert budget.state_nbytes == measure_stacked_horde_state_nbytes(state)
    assert budget.state_nbytes == stacked_horde_state_nbytes_formula(2, 2)
    legacy_nbytes = state.weights.nbytes + state.traces.nbytes + state.step_count.nbytes
    assert budget.state_nbytes - legacy_nbytes == (
        STACKED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES
    )
    serialized_budget = budget.to_dict()
    assert serialized_budget["schema"] == STACKED_HORDE_RESOURCE_SCHEMA
    assert StackedHordeResourceBudget.from_dict(serialized_budget) == budget
    corrupt_budget = dict(serialized_budget)
    corrupt_budget["state_nbytes"] = int(corrupt_budget["state_nbytes"]) + 4
    with pytest.raises(ValueError, match="total byte"):
        StackedHordeResourceBudget.from_dict(corrupt_budget)
