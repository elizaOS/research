# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Contracts for the paper-defined Kondo forward/backward boundary."""

from __future__ import annotations

import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.kondo_gate import (
    KONDO_GATE_SCHEMA,
    KondoGate,
    KondoGateConfig,
    KondoGateState,
)

pytestmark = pytest.mark.unit


def _arrays(batch_size: int = 6) -> tuple[jax.Array, ...]:
    advantage = jnp.asarray([1.0, -1.0, 2.0, 0.5, 1.0, -0.25], dtype=jnp.float32)
    log_probability = jnp.asarray(
        [-1.0, -2.0, -0.25, -4.0, -1.0, -3.0], dtype=jnp.float32
    )
    valid = jnp.ones((batch_size,), dtype=jnp.bool_)
    forced = jnp.zeros((batch_size,), dtype=jnp.bool_)
    return (
        advantage[:batch_size],
        log_probability[:batch_size],
        valid,
        forced,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = (
            np.asarray(jr.key_data(lhs))
            if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key)
            else np.asarray(lhs)
        )
        rhs_array = (
            np.asarray(jr.key_data(rhs))
            if jax.dtypes.issubdtype(rhs.dtype, jax.dtypes.prng_key)
            else np.asarray(rhs)
        )
        np.testing.assert_array_equal(lhs_array, rhs_array)


def test_config_names_the_paper_semantics_and_roundtrips_strictly() -> None:
    assert alberta.KondoGate is KondoGate
    assert core.KondoGate is KondoGate
    config = KondoGateConfig(batch_size=10, target_rate=0.26)
    payload = config.to_config()

    assert payload["schema"] == KONDO_GATE_SCHEMA
    assert payload["delight_semantics"] == "advantage-times-action-surprisal"
    assert payload["sparks_joy_semantics"] == "selected-for-backward-pass"
    assert payload["generic_gradient_quality_audit"] is False
    assert payload["wall_clock_savings_claimed"] is False
    assert payload["backward_capacity"] == 3
    assert KondoGateConfig.from_config(payload) == config
    assert KondoGate.from_config(payload).to_config() == payload

    integer_spelling = KondoGateConfig(
        batch_size=10,
        target_rate=1,  # type: ignore[arg-type]
        price=0,  # type: ignore[arg-type]
        temperature=1,  # type: ignore[arg-type]
    )
    assert type(integer_spelling.target_rate) is float
    assert type(integer_spelling.price) is float
    assert type(integer_spelling.temperature) is float
    assert KondoGateConfig.from_config(integer_spelling.to_config()) == integer_spelling
    assert KondoGateConfig(batch_size=10, target_rate=0.25).backward_capacity == 2

    malformed = dict(payload)
    malformed["target_rate"] = 1
    with pytest.raises(ValueError, match="target_rate must be a float"):
        KondoGateConfig.from_config(malformed)
    malformed = dict(payload)
    malformed["sparks_joy_semantics"] = "generic-gradient-quality"
    with pytest.raises(ValueError, match="sparks_joy_semantics"):
        KondoGateConfig.from_config(malformed)
    malformed = dict(payload)
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="fields"):
        KondoGateConfig.from_config(malformed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"mode": "unknown"},
        {"mode": np.str_("top_k_rate")},
        {"target_rate": 0.0},
        {"target_rate": 1.0e-100},
        {"target_rate": float("nan")},
        {"target_rate": True},
        {"price": float("inf")},
        {"price": -0.01},
        {"price": 1.0e-100},
        {"temperature": 0.0},
        {"temperature": 1.0e-100},
        {"minimum_uniform_keep": -1},
        {"max_screenings": 0},
        {"mode": "top_k_rate", "sparse_capacity": 2},
        {"mode": "bernoulli_price", "sparse_capacity": None},
        {"mode": "bernoulli_price", "sparse_capacity": 7},
        {"target_rate": 0.2, "minimum_uniform_keep": 2},
    ],
)
def test_config_rejects_unsafe_or_ambiguous_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {"batch_size": 6}
    values.update(kwargs)
    with pytest.raises(ValueError):
        KondoGateConfig(**values)  # type: ignore[arg-type]


def test_top_k_delight_is_exact_and_sparks_joy_means_backward_selection() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(9))
    advantage, log_probability, valid, forced = _arrays()

    result = gate.screen(state, advantage, log_probability, valid, forced)
    expected_surprisal = -log_probability
    expected_delight = advantage * expected_surprisal

    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(result.action_surprisal, jnp.int32),
        jax.lax.bitcast_convert_type(expected_surprisal, jnp.int32),
    )
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(result.delight, jnp.int32),
        jax.lax.bitcast_convert_type(expected_delight, jnp.int32),
    )
    # Delight values are [1, -2, .5, 2, 1, -.75]. Lowest index wins the
    # exact 1.0 tie for the third sparse slot.
    np.testing.assert_array_equal(
        result.selected_mask,
        jnp.asarray([True, False, False, True, True, False]),
    )
    np.testing.assert_array_equal(result.selected_by_delight_gate, result.selected_mask)
    np.testing.assert_array_equal(result.sparks_joy, result.selected_mask)
    np.testing.assert_array_equal(result.selected_indices, jnp.asarray([3, 0, 4]))
    assert int(result.selected_count) == 3
    assert int(result.state.forward_slots_screened) == 6
    assert int(result.state.valid_examples_screened) == 6
    assert int(result.state.examples_selected) == 3
    assert bool(result.sparse_backward_available)
    assert not bool(result.full_shape_masked_backward_required)
    assert int(result.random_draw_count) == 0
    np.testing.assert_array_equal(jr.key_data(result.state.rng_key), jr.key_data(state.rng_key))


def test_padding_is_not_selected_and_ties_use_lowest_source_index() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(0))
    advantage = jnp.ones((6,), dtype=jnp.float32)
    log_probability = -jnp.ones((6,), dtype=jnp.float32)
    valid = jnp.asarray([True, False, True, False, True, True], dtype=jnp.bool_)
    forced = jnp.zeros((6,), dtype=jnp.bool_)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    np.testing.assert_array_equal(result.selected_indices, jnp.asarray([0, 2, 0]))
    np.testing.assert_array_equal(
        result.selected_slot_mask,
        jnp.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        result.selected_mask,
        jnp.asarray([True, False, True, False, False, False]),
    )
    assert int(result.selected_count) == 2
    assert int(result.valid_count) == 4
    assert int(result.state.forward_slots_screened) == 6
    assert int(result.state.valid_examples_screened) == 4
    assert int(result.state.examples_selected) == 2


def test_force_keep_is_independent_and_displaces_lower_delight() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(0))
    advantage, log_probability, valid, _ = _arrays()
    forced = jnp.asarray([False, True, False, False, False, False], dtype=jnp.bool_)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    assert bool(result.selected_mask[1])
    assert int(result.selected_count) == 3
    np.testing.assert_array_equal(result.force_keep_mask, forced)
    assert bool(result.capacity_sufficient)


def test_forced_overflow_never_drops_forced_samples_and_requires_full_shape() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.25))
    state = gate.init(jr.key(0))
    advantage, log_probability, valid, _ = _arrays()
    forced = jnp.asarray([True, True, True, False, False, False], dtype=jnp.bool_)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    assert not bool(result.capacity_sufficient)
    assert not bool(result.sparse_backward_available)
    assert bool(result.full_shape_masked_backward_required)
    assert bool(jnp.all(result.selected_mask[:3]))
    assert int(result.selected_count) >= 3
    with pytest.raises(ValueError, match="full-shape"):
        gate.gather_sparse({"x": jnp.arange(6)}, result)


def test_bernoulli_price_matches_the_exact_paper_draw_and_key_progression() -> None:
    config = KondoGateConfig(
        batch_size=6,
        mode="bernoulli_price",
        sparse_capacity=6,
        price=0.25,
        temperature=0.75,
    )
    gate = KondoGate(config)
    key = jr.key(123)
    state = gate.init(key)
    advantage, log_probability, valid, forced = _arrays()

    result = gate.screen(state, advantage, log_probability, valid, forced)
    expected_next_key, draw_key = jr.split(key)
    expected_probability = jax.nn.sigmoid(
        ((advantage * -log_probability) - jnp.float32(0.25)) / jnp.float32(0.75)
    )
    expected_selected = jr.uniform(draw_key, (6,), dtype=jnp.float32) < expected_probability

    np.testing.assert_array_equal(result.gate_probability, expected_probability)
    np.testing.assert_array_equal(result.selected_by_delight_gate, expected_selected)
    np.testing.assert_array_equal(result.selected_mask, expected_selected)
    np.testing.assert_array_equal(jr.key_data(result.state.rng_key), jr.key_data(expected_next_key))
    assert int(result.random_draw_count) == 6


def test_bernoulli_probability_is_zero_on_padding() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=6,
            price=0.25,
            temperature=0.75,
        )
    )
    advantage, log_probability, _, forced = _arrays()
    valid = jnp.asarray([True, False, True, False, True, True], dtype=jnp.bool_)

    result = gate.screen(
        gate.init(jr.key(123)), advantage, log_probability, valid, forced
    )

    np.testing.assert_array_equal(
        result.gate_probability[~valid],
        jnp.zeros((2,), dtype=jnp.float32),
    )


def test_bernoulli_overflow_preserves_gate_mask_but_discloses_no_sparse_saving() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=1,
            price=0.0,
            temperature=1.0,
        )
    )
    _, log_probability, valid, forced = _arrays()
    advantage = jnp.full((6,), 100.0, dtype=jnp.float32)
    result = gate.screen(
        gate.init(jr.key(0)), advantage, log_probability, valid, forced
    )

    assert int(result.selected_count) == 6
    assert bool(jnp.all(result.selected_mask))
    assert bool(result.full_shape_masked_backward_required)
    assert not bool(result.sparse_backward_available)


def test_uniform_reserve_uses_valid_rows_and_is_reported_separately() -> None:
    gate = KondoGate(
        KondoGateConfig(batch_size=6, target_rate=0.5, minimum_uniform_keep=2)
    )
    advantage, log_probability, _, forced = _arrays()
    valid = jnp.asarray([False, True, False, True, True, True], dtype=jnp.bool_)
    result = gate.screen(
        gate.init(jr.key(5)), advantage, log_probability, valid, forced
    )

    assert int(jnp.sum(result.uniformly_reserved.astype(jnp.int32))) == 2
    assert not bool(jnp.any(result.uniformly_reserved & ~valid))
    assert bool(jnp.all(result.selected_mask[result.uniformly_reserved]))
    assert int(result.random_draw_count) == 6


def test_bernoulli_and_uniform_reserve_use_independent_random_streams() -> None:
    config = KondoGateConfig(
        batch_size=6,
        mode="bernoulli_price",
        sparse_capacity=6,
        price=0.0,
        temperature=1.0,
        minimum_uniform_keep=2,
    )
    gate = KondoGate(config)
    key = jr.key(31)
    advantage, log_probability, valid, forced = _arrays()

    result = gate.screen(gate.init(key), advantage, log_probability, valid, forced)
    expected_next_key, gate_key, reserve_key = jr.split(key, 3)
    expected_probability = jax.nn.sigmoid(advantage * -log_probability)
    expected_gate = jr.uniform(gate_key, (6,), dtype=jnp.float32) < expected_probability
    reserve_uniforms = jr.uniform(reserve_key, (6,), dtype=jnp.float32)
    reserve_order = jnp.argsort(reserve_uniforms, stable=True)
    expected_reserve = jnp.zeros((6,), dtype=jnp.bool_).at[reserve_order[:2]].set(True)

    np.testing.assert_array_equal(result.selected_by_delight_gate, expected_gate)
    np.testing.assert_array_equal(result.uniformly_reserved, expected_reserve)
    np.testing.assert_array_equal(
        jr.key_data(result.state.rng_key), jr.key_data(expected_next_key)
    )
    assert int(result.random_draw_count) == 12
    assert gate.resource_declaration().maximum_random_draws_per_screen == 12


def test_sparse_gather_really_reduces_the_autodiff_input_shape() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    advantage, log_probability, valid, forced = _arrays()
    result = gate.screen(
        gate.init(jr.key(0)), advantage, log_probability, valid, forced
    )
    full_batch = {
        "features": jnp.arange(12, dtype=jnp.float32).reshape(6, 2),
        "targets": jnp.arange(6, dtype=jnp.float32),
    }

    sparse = gate.gather_sparse(full_batch, result)

    assert sparse.data["features"].shape == (3, 2)
    assert sparse.data["targets"].shape == (3,)
    assert sparse.sample_mask.shape == (3,)
    np.testing.assert_array_equal(
        sparse.data["features"], full_batch["features"][result.selected_indices]
    )

    def sparse_loss(weight: jax.Array, features: jax.Array, mask: jax.Array) -> jax.Array:
        values = features @ weight
        return jnp.sum(jnp.square(values) * mask.astype(jnp.float32))

    grad = jax.grad(sparse_loss)(
        jnp.ones((2,), dtype=jnp.float32),
        sparse.data["features"],
        sparse.sample_mask,
    )
    assert grad.shape == (2,)
    jaxpr = str(
        jax.make_jaxpr(jax.grad(sparse_loss))(
            jnp.ones((2,), dtype=jnp.float32),
            sparse.data["features"],
            sparse.sample_mask,
        )
    )
    assert "f32[3,2]" in jaxpr
    assert "f32[6,2]" not in jaxpr


def test_sparse_gather_padding_is_dummy_and_loss_mask_is_required() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    advantage = jnp.ones((6,), dtype=jnp.float32)
    log_probability = -jnp.ones((6,), dtype=jnp.float32)
    valid = jnp.asarray([True, False, True, False, True, True], dtype=jnp.bool_)
    forced = jnp.zeros((6,), dtype=jnp.bool_)
    result = gate.screen(
        gate.init(jr.key(0)), advantage, log_probability, valid, forced
    )
    values = jnp.asarray([2.0, 100.0, 3.0, 100.0, 50.0, 60.0], dtype=jnp.float32)

    sparse = gate.gather_sparse({"value": values}, result)

    np.testing.assert_array_equal(sparse.sample_mask, jnp.asarray([True, True, False]))
    assert float(
        jnp.sum(sparse.data["value"] * sparse.sample_mask.astype(jnp.float32))
    ) == 5.0


def test_sparse_gather_rejects_results_from_other_configurations() -> None:
    source = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = source.screen(source.init(jr.key(0)), *_arrays())
    same_shape_other_config = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=3,
            price=0.0,
        )
    )
    different_capacity = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.25))
    batch = {"x": jnp.arange(6, dtype=jnp.float32)}

    with pytest.raises(ValueError, match="different gate configuration"):
        same_shape_other_config.gather_sparse(batch, result)
    with pytest.raises(ValueError, match="selected_indices"):
        different_capacity.gather_sparse(batch, result)


@pytest.mark.parametrize(
    "batch",
    [
        {},
        {"x": jnp.asarray(1.0, dtype=jnp.float32)},
        {"x": jnp.zeros((5, 2), dtype=jnp.float32)},
    ],
)
def test_sparse_gather_rejects_malformed_batch_trees(batch: object) -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    with pytest.raises(ValueError):
        gate.gather_sparse(batch, result)


@pytest.mark.parametrize("field", ["force_keep_mask", "uniformly_reserved"])
def test_sparse_gather_rejects_unselected_mandatory_rows(field: str) -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    unselected = int(jnp.argmax(~result.selected_mask))
    tampered = replace(
        result,
        **{field: getattr(result, field).at[unselected].set(True)},
    )

    with pytest.raises(ValueError, match="accounting"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_state_that_claims_a_full_fallback() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    tampered = replace(
        result,
        state=replace(
            result.state,
            sparse_batch_count=jnp.asarray(0, dtype=jnp.int32),
            full_fallback_count=jnp.asarray(1, dtype=jnp.int32),
        ),
    )

    with pytest.raises(ValueError, match="accounting"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_impossible_predecessor_counters() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    advantage, log_probability, _, forced = _arrays()
    result = gate.screen(
        gate.init(jr.key(0)),
        advantage,
        log_probability,
        jnp.zeros((6,), dtype=jnp.bool_),
        forced,
    )
    tampered = replace(
        result,
        state=replace(
            result.state,
            screen_count=jnp.asarray(2, dtype=jnp.int32),
            forward_slots_screened=jnp.asarray(12, dtype=jnp.int32),
            valid_examples_screened=jnp.asarray(10, dtype=jnp.int32),
            sparse_batch_count=jnp.asarray(2, dtype=jnp.int32),
        ),
    )
    assert bool(gate._state_valid(tampered.state))

    with pytest.raises(ValueError, match="predecessor accounting"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_dropped_bernoulli_gate_survivor() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=6,
            price=100.0,
        )
    )
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    unselected = int(jnp.argmax(~result.selected_mask))
    tampered = replace(
        result,
        selected_by_delight_gate=(
            result.selected_by_delight_gate.at[unselected].set(True)
        ),
    )

    with pytest.raises(ValueError, match="Bernoulli selection union"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_coordinated_bernoulli_selection_rewrite() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=6,
            price=0.0,
        )
    )
    result = gate.screen(gate.init(jr.key(1)), *_arrays())
    authentic = np.flatnonzero(np.asarray(result.selected_mask))
    omitted = np.flatnonzero(~np.asarray(result.selected_mask))
    assert authentic.size > 0
    assert omitted.size > 0
    source = int(authentic[0])
    destination = int(omitted[0])
    selected_mask = result.selected_mask.at[source].set(False).at[
        destination
    ].set(True)
    selected_by_gate = result.selected_by_delight_gate.at[source].set(False).at[
        destination
    ].set(True)
    selected_order = jnp.argsort(~selected_mask, stable=True)
    selected_indices = selected_order[:6].astype(jnp.int32)
    selected_slot_mask = selected_mask[selected_indices]
    selected_indices = jnp.where(
        selected_slot_mask,
        selected_indices,
        jnp.zeros_like(selected_indices),
    )
    tampered = replace(
        result,
        selected_by_delight_gate=selected_by_gate,
        selected_mask=selected_mask,
        selected_indices=selected_indices,
        selected_slot_mask=selected_slot_mask,
    )

    with pytest.raises(ValueError, match="Bernoulli selection union"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_rng_key_not_derived_from_screen_key() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=6,
            price=100.0,
        )
    )
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    tampered = replace(
        result,
        state=replace(result.state, rng_key=jr.key(999)),
    )

    with pytest.raises(ValueError, match="RNG key is inconsistent"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_reordered_active_indices() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    assert bool(result.selected_slot_mask[0] & result.selected_slot_mask[1])
    reordered = result.selected_indices.at[0].set(result.selected_indices[1]).at[
        1
    ].set(result.selected_indices[0])
    tampered = replace(result, selected_indices=reordered)

    with pytest.raises(ValueError, match="index order"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_same_count_uniform_reserve_rewrite() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            target_rate=0.5,
            minimum_uniform_keep=1,
        )
    )
    result = gate.screen(gate.init(jr.key(7)), *_arrays())
    reserved = int(jnp.argmax(result.uniformly_reserved))
    replacement = int(jnp.argmax(~result.uniformly_reserved))
    tampered_reserve = result.uniformly_reserved.at[reserved].set(False).at[
        replacement
    ].set(True)
    tampered = replace(result, uniformly_reserved=tampered_reserve)

    with pytest.raises(ValueError, match="uniform-reserve accounting"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


def test_sparse_gather_rejects_valid_mask_that_excludes_a_selected_row() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    selected = int(jnp.argmax(result.selected_mask))
    unselected = int(jnp.argmax(~result.selected_mask))
    tampered_valid = result.valid_mask.at[selected].set(False).at[unselected].set(True)
    tampered = replace(result, valid_mask=tampered_valid)

    with pytest.raises(ValueError, match="selection accounting"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


@pytest.mark.parametrize(
    "mutation",
    ["gate_count", "selected_count", "non_gate_selection", "reserve_count"],
)
def test_sparse_gather_rejects_inconsistent_top_k_components(
    mutation: str,
) -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            target_rate=0.5,
            minimum_uniform_keep=1,
        )
    )
    result = gate.screen(gate.init(jr.key(7)), *_arrays())
    if mutation == "gate_count":
        selected_by_gate = result.selected_by_delight_gate.at[
            int(jnp.argmax(result.selected_by_delight_gate))
        ].set(False)
        tampered = replace(result, selected_by_delight_gate=selected_by_gate)
        match = "top-k selection accounting"
    elif mutation == "selected_count":
        tampered = replace(
            result,
            selected_count=result.selected_count - jnp.asarray(1, dtype=jnp.int32),
        )
        match = "accounting"
    elif mutation == "non_gate_selection":
        selected = np.flatnonzero(
            np.asarray(result.selected_mask)
            & ~np.asarray(result.uniformly_reserved)
            & ~np.asarray(result.force_keep_mask)
        )
        replacement = np.flatnonzero(
            ~np.asarray(result.selected_by_delight_gate)
            & ~np.asarray(result.uniformly_reserved)
            & ~np.asarray(result.force_keep_mask)
        )
        assert selected.size > 0
        assert replacement.size > 0
        source = int(selected[-1])
        destination = int(replacement[0])
        selected_mask = result.selected_mask.at[source].set(False).at[
            destination
        ].set(True)
        active_slot = int(jnp.argmax(result.selected_indices == source))
        selected_indices = result.selected_indices.at[active_slot].set(destination)
        tampered = replace(
            result,
            selected_mask=selected_mask,
            selected_indices=selected_indices,
        )
        match = "top-k selection accounting"
    else:
        reserve = result.uniformly_reserved.at[
            int(jnp.argmax(result.uniformly_reserved))
        ].set(False)
        tampered = replace(result, uniformly_reserved=reserve)
        match = "uniform-reserve accounting"

    with pytest.raises(ValueError, match=match):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sparse_backward_available", False),
        ("full_shape_masked_backward_required", True),
    ],
)
def test_sparse_gather_rejects_inconsistent_backward_route(
    field: str,
    value: bool,
) -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    result = gate.screen(gate.init(jr.key(0)), *_arrays())
    tampered = replace(
        result,
        **{field: jnp.asarray(value, dtype=jnp.bool_)},
    )

    with pytest.raises(ValueError, match="decision is inconsistent"):
        gate.gather_sparse({"x": jnp.arange(6, dtype=jnp.float32)}, tampered)


@pytest.mark.parametrize(
    "mutation",
    [
        "nan_advantage",
        "positive_log_probability",
        "forced_padding",
    ],
)
def test_dynamic_invalid_inputs_are_exact_state_noops(mutation: str) -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(7))
    advantage, log_probability, valid, forced = _arrays()
    if mutation == "nan_advantage":
        advantage = advantage.at[1].set(jnp.nan)
    elif mutation == "positive_log_probability":
        log_probability = log_probability.at[1].set(0.01)
    else:
        valid = valid.at[1].set(False)
        forced = forced.at[1].set(True)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    assert not bool(result.transaction_applied)
    assert not bool(result.input_valid)
    _assert_tree_equal(result.state, state)
    assert not bool(jnp.any(result.selected_mask))
    assert not bool(jnp.any(result.delight))


def test_finite_inputs_whose_delight_overflows_fail_closed() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(7))
    advantage, log_probability, valid, forced = _arrays()
    advantage = advantage.at[0].set(jnp.finfo(jnp.float32).max)
    log_probability = log_probability.at[0].set(-2.0)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    assert not bool(result.transaction_applied)
    assert not bool(result.input_valid)
    _assert_tree_equal(result.state, state)
    assert not bool(jnp.any(result.delight))


@pytest.mark.parametrize("state_kind", ["exhausted", "corrupt"])
def test_input_validity_is_independent_of_state_and_capacity(state_kind: str) -> None:
    gate = KondoGate(
        KondoGateConfig(batch_size=6, target_rate=0.5, max_screenings=1)
    )
    state = gate.init(jr.key(7))
    if state_kind == "exhausted":
        state = gate.screen(state, *_arrays()).state
    else:
        state = replace(
            state,
            forward_slots_screened=jnp.asarray(1, dtype=jnp.int32),
        )
    advantage, log_probability, valid, forced = _arrays()
    advantage = advantage.at[0].set(jnp.finfo(jnp.float32).max)
    log_probability = log_probability.at[0].set(-2.0)

    result = gate.screen(state, advantage, log_probability, valid, forced)

    assert not bool(result.input_valid)
    assert not bool(result.transaction_applied)
    _assert_tree_equal(result.state, state)


def test_only_threefry_typed_keys_are_checkpointable() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    with pytest.raises(TypeError, match="threefry2x32"):
        gate.init(jr.key(0, impl="rbg"))


def test_resource_declaration_does_not_depend_on_default_prng_impl() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    previous = jax.config.jax_default_prng_impl
    try:
        jax.config.update("jax_default_prng_impl", "rbg")
        resources = gate.resource_declaration()
    finally:
        jax.config.update("jax_default_prng_impl", previous)
    assert resources.forward_slots_per_screen == 6


def test_corrupt_state_and_exhausted_capacity_fail_closed() -> None:
    gate = KondoGate(
        KondoGateConfig(batch_size=6, target_rate=0.5, max_screenings=1)
    )
    state = gate.init(jr.key(0))
    first = gate.screen(state, *_arrays())
    exhausted = gate.screen(first.state, *_arrays())

    assert not bool(exhausted.transaction_applied)
    assert bool(exhausted.state_valid)
    _assert_tree_equal(exhausted.state, first.state)

    corrupt = replace(
        state,
        forward_slots_screened=jnp.asarray(1, dtype=jnp.int32),
    )
    rejected = gate.screen(corrupt, *_arrays())
    assert not bool(rejected.transaction_applied)
    assert not bool(rejected.state_valid)
    _assert_tree_equal(rejected.state, corrupt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("advantage", jnp.zeros((5,), dtype=jnp.float32)),
        ("advantage", jnp.zeros((6,), dtype=jnp.bfloat16)),
        ("action_log_probability", jnp.zeros((6,), dtype=jnp.int32)),
        ("valid_mask", jnp.zeros((6,), dtype=jnp.int32)),
        ("force_keep_mask", jnp.zeros((5,), dtype=jnp.bool_)),
    ],
)
def test_static_input_contracts_raise_before_tracing(field: str, value: jax.Array) -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    inputs = dict(
        zip(
            ("advantage", "action_log_probability", "valid_mask", "force_keep_mask"),
            _arrays(),
            strict=True,
        )
    )
    inputs[field] = value
    with pytest.raises((TypeError, ValueError)):
        gate.screen(gate.init(jr.key(0)), **inputs)


def test_eager_jit_and_scan_are_exact_and_source_state_is_immutable() -> None:
    gate = KondoGate(KondoGateConfig(batch_size=6, target_rate=0.5))
    state = gate.init(jr.key(42))
    inputs = _arrays()
    before = state

    eager = gate.screen(state, *inputs)
    compiled = jax.jit(gate.screen)(state, *inputs)
    _assert_tree_equal(eager, compiled)
    _assert_tree_equal(state, before)

    def body(carry: KondoGateState, _: jax.Array) -> tuple[KondoGateState, jax.Array]:
        result = gate.screen(carry, *inputs)
        return result.state, result.selected_count

    final, counts = jax.lax.scan(body, state, jnp.arange(3))
    assert int(final.screen_count) == 3
    assert int(final.forward_slots_screened) == 18
    assert int(final.valid_examples_screened) == 18
    np.testing.assert_array_equal(counts, jnp.asarray([3, 3, 3], dtype=jnp.int32))


def test_checkpoint_roundtrip_and_resource_accounting_are_strict() -> None:
    gate = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=4,
            minimum_uniform_keep=1,
        )
    )
    result = gate.screen(gate.init(jr.key(8)), *_arrays())
    payload = gate.checkpoint_payload(result.state)
    restored_gate, restored_state = KondoGate.from_checkpoint_payload(payload)

    assert restored_gate.to_config() == gate.to_config()
    _assert_tree_equal(restored_state, result.state)
    resources = gate.resource_declaration(result.state)
    assert resources.forward_slots_per_screen == 6
    assert resources.sparse_backward_capacity == 4
    assert resources.maximum_random_draws_per_screen == 12
    assert resources.maximum_delight_products_per_screen == 6
    assert resources.persistent_state_bytes > 0
    assert resources.full_shape_fallback_possible
    assert not resources.wall_clock_savings_claimed
    no_fallback = KondoGate(
        KondoGateConfig(
            batch_size=6,
            mode="bernoulli_price",
            sparse_capacity=6,
        )
    ).resource_declaration()
    assert not no_fallback.full_shape_fallback_possible

    malformed = copy.deepcopy(payload)
    malformed["state"]["rng_key_data"] = [0]  # type: ignore[index]
    with pytest.raises(ValueError, match="rng_key_data"):
        KondoGate.from_checkpoint_payload(malformed)
    malformed = copy.deepcopy(payload)
    malformed["state"]["forward_slots_screened"] += 1  # type: ignore[index]
    with pytest.raises(ValueError, match="state is invalid"):
        KondoGate.from_checkpoint_payload(malformed)
