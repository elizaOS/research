"""Strict mechanism contracts for recurrent RTU generate-and-test."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.recurrent_trace_actor_critic import (
    RTUSensitivities,
    RTUState,
)
from alberta_framework.core.rtu_generate_and_test import (
    RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED,
    RTUGenerateAndTest,
    RTUGenerateAndTestCompositionState,
    RTUGenerateAndTestConfig,
    load_rtu_generate_and_test_checkpoint,
    save_rtu_generate_and_test_checkpoint,
)
from alberta_framework.core.state_builder import RecurrentTraceUnitStateBuilderConfig

pytestmark = pytest.mark.unit


def _config(
    *,
    hidden_dim: int = 3,
    utility_decay: float = 0.0,
    replacement_interval: int = 1,
    replacement_quota: int = 1,
    warmup_observations: int = 0,
    minimum_age: int = 0,
    minimum_support: int = 0,
    minimum_causal_evidence: int = 1,
    protected_units: tuple[int, ...] = (),
    taylor: bool = False,
) -> RTUGenerateAndTestConfig:
    return RTUGenerateAndTestConfig(
        builder=RecurrentTraceUnitStateBuilderConfig(
            observation_dim=2,
            hidden_dim=hidden_dim,
            rtrl_taylor_correction=taylor,
        ),
        utility_decay=utility_decay,
        replacement_interval=replacement_interval,
        replacement_quota=replacement_quota,
        warmup_observations=warmup_observations,
        minimum_age=minimum_age,
        minimum_support=minimum_support,
        minimum_causal_evidence=minimum_causal_evidence,
        protected_units=protected_units,
    )


def _tree_assert_exact(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        if jnp.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _payload_builder(
    lifecycle: RTUGenerateAndTest,
    *,
    taylor_scale: float = 0.0,
) -> Any:
    state = lifecycle.builder.init(jr.key(90, impl="threefry2x32"))
    hidden = lifecycle.config.builder.hidden_dim
    event = lifecycle.config.builder.event_dim()
    polar = jnp.arange(1, 2 * hidden + 1, dtype=jnp.float32).reshape((2, hidden))
    inputs = jnp.arange(
        1,
        2 * hidden * event + 1,
        dtype=jnp.float32,
    ).reshape((2, hidden, event))
    sensitivities = RTUSensitivities(
        nu_log=polar,
        theta_log=polar + 20.0,
        b_real=inputs,
        b_imag=inputs + 100.0,
    )
    updates: dict[str, Any] = {
        "rtu_state": RTUState(
            real=jnp.arange(1, hidden + 1, dtype=jnp.float32),
            imaginary=jnp.arange(11, 11 + hidden, dtype=jnp.float32),
        ),
        "sensitivities": sensitivities,
        "last_gradient_norm": jnp.asarray(7.5, dtype=jnp.float32),
    }
    if lifecycle.config.builder.rtrl_taylor_correction:
        updates["taylor_trace"] = RTUSensitivities(
            nu_log=taylor_scale + polar + 200.0,
            theta_log=taylor_scale + polar + 220.0,
            b_real=taylor_scale + inputs + 300.0,
            b_imag=taylor_scale + inputs + 400.0,
        )
    state = state.replace(**updates)
    assert bool(lifecycle.builder.state_valid(state))
    return state


def _gradient(
    lifecycle: RTUGenerateAndTest,
    real: tuple[float, ...],
    imaginary: tuple[float, ...] | None = None,
) -> jax.Array:
    hidden = lifecycle.config.builder.hidden_dim
    imaginary_values = imaginary if imaginary is not None else (0.0,) * hidden
    return jnp.concatenate(
        (
            jnp.zeros((lifecycle.config.builder.observation_dim,), dtype=jnp.float32),
            jnp.asarray(real, dtype=jnp.float32),
            jnp.asarray(imaginary_values, dtype=jnp.float32),
        )
    )


def _parameter_parts(lifecycle: RTUGenerateAndTest, parameters: jax.Array) -> tuple[Any, ...]:
    hidden = lifecycle.config.builder.hidden_dim
    event = lifecycle.config.builder.event_dim()
    offset = 2 * hidden
    return (
        parameters[:hidden],
        parameters[hidden:offset],
        parameters[offset : offset + hidden * event].reshape((hidden, event)),
        parameters[offset + hidden * event :].reshape((hidden, event)),
    )


def test_initial_state_is_fixed_capacity_and_not_assessed() -> None:
    lifecycle = RTUGenerateAndTest(
        RTUGenerateAndTestConfig(
            builder=RecurrentTraceUnitStateBuilderConfig(
                observation_dim=2,
                hidden_dim=3,
            )
        )
    )
    state = lifecycle.init(jr.key(1, impl="threefry2x32"))

    assert state.utility.shape == (3,)
    assert state.causal_utility.shape == (3,)
    assert state.causal_evidence_count.shape == (3,)
    np.testing.assert_array_equal(state.causal_evidence_count, 0)
    assert lifecycle.evidence_level == "L0"
    assert lifecycle.mechanism_status == "not_assessed"
    assert lifecycle.scientific_promotion_allowed is False
    assert RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert bool(lifecycle.state_valid(state))


def test_state_validator_rejects_unreachable_causal_and_replacement_telemetry() -> None:
    lifecycle = RTUGenerateAndTest(_config())
    state = lifecycle.init(jr.key(301, impl="threefry2x32"))

    count_before_age = state.replace(
        causal_evidence_count=state.causal_evidence_count.at[0].set(1)
    )
    unsupported_ema = state.replace(
        causal_utility=state.causal_utility.at[0].set(jnp.float32(0.5))
    )
    stale_last_change = state.replace(
        last_causal_deletion_loss_change=(
            state.last_causal_deletion_loss_change.at[0].set(jnp.float32(0.25))
        )
    )
    unsanitized_replacement = state.replace(
        last_replaced_mask=state.last_replaced_mask.at[0].set(True),
        utility=state.utility.at[0].set(jnp.float32(1.0)),
    )
    orphaned_scrubbed_replacement = state.replace(
        last_replaced_mask=state.last_replaced_mask.at[0].set(True)
    )
    evidence_before_any_observation = state.replace(
        age=jnp.full_like(state.age, 2),
        causal_utility=jnp.full_like(state.causal_utility, jnp.float32(0.5)),
        causal_evidence_count=jnp.full_like(state.causal_evidence_count, 2),
        last_causal_deletion_loss_change=jnp.full_like(
            state.last_causal_deletion_loss_change,
            jnp.float32(1.0),
        ),
    )
    causal_ema_outside_reachable_range = state.replace(
        observation_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        observation_count=jnp.asarray(1, dtype=jnp.int32),
        age=jnp.ones_like(state.age),
        causal_utility=jnp.full_like(state.causal_utility, jnp.float32(1.01)),
        causal_evidence_count=jnp.ones_like(state.causal_evidence_count),
        last_causal_deletion_loss_change=jnp.ones_like(
            state.last_causal_deletion_loss_change
        ),
    )
    replacement_event_without_observation = state.replace(
        replacement_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        replacement_count=jnp.asarray(1, dtype=jnp.int32),
        replacement_event_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        replacement_event_count=jnp.asarray(1, dtype=jnp.int32),
    )
    replacement_quota_exceeded = state.replace(
        observation_words=jnp.asarray([0, 2], dtype=jnp.uint32),
        observation_count=jnp.asarray(2, dtype=jnp.int32),
        age=jnp.ones_like(state.age),
        replacement_words=jnp.asarray([0, 2], dtype=jnp.uint32),
        replacement_count=jnp.asarray(2, dtype=jnp.int32),
        replacement_event_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        replacement_event_count=jnp.asarray(1, dtype=jnp.int32),
    )
    slow_ema_lifecycle = RTUGenerateAndTest(_config(utility_decay=0.99))
    slow_ema_state = slow_ema_lifecycle.init(jr.key(302, impl="threefry2x32"))
    unreachable_first_causal_ema = slow_ema_state.replace(
        observation_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        observation_count=jnp.asarray(1, dtype=jnp.int32),
        age=jnp.ones_like(slow_ema_state.age),
        causal_utility=jnp.full_like(
            slow_ema_state.causal_utility,
            jnp.float32(0.5),
        ),
        causal_evidence_count=jnp.ones_like(
            slow_ema_state.causal_evidence_count
        ),
        last_causal_deletion_loss_change=jnp.ones_like(
            slow_ema_state.last_causal_deletion_loss_change
        ),
    )
    quota_two_lifecycle = RTUGenerateAndTest(_config(replacement_quota=2))
    quota_two_state = quota_two_lifecycle.init(jr.key(303, impl="threefry2x32"))
    reachable_two_unit_replacement = quota_two_state.replace(
        observation_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        observation_count=jnp.asarray(1, dtype=jnp.int32),
        replacement_words=jnp.asarray([0, 2], dtype=jnp.uint32),
        replacement_count=jnp.asarray(2, dtype=jnp.int32),
        replacement_event_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        replacement_event_count=jnp.asarray(1, dtype=jnp.int32),
        age=jnp.asarray([0, 0, 1], dtype=jnp.uint32),
        last_replaced_mask=jnp.asarray([True, True, False], dtype=jnp.bool_),
    )

    assert not bool(lifecycle.state_valid(count_before_age))
    assert not bool(lifecycle.state_valid(unsupported_ema))
    assert not bool(lifecycle.state_valid(stale_last_change))
    assert not bool(lifecycle.state_valid(unsanitized_replacement))
    assert not bool(lifecycle.state_valid(orphaned_scrubbed_replacement))
    assert not bool(lifecycle.state_valid(evidence_before_any_observation))
    assert not bool(lifecycle.state_valid(causal_ema_outside_reachable_range))
    assert not bool(lifecycle.state_valid(replacement_event_without_observation))
    assert not bool(lifecycle.state_valid(replacement_quota_exceeded))
    assert not bool(slow_ema_lifecycle.state_valid(unreachable_first_causal_ema))
    assert bool(quota_two_lifecycle.state_valid(reachable_two_unit_replacement))


def test_hand_computed_pre_update_effective_contribution_utility() -> None:
    lifecycle = RTUGenerateAndTest(
        _config(replacement_interval=17, minimum_age=8, minimum_support=8)
    )
    state = lifecycle.init(jr.key(2, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(
        lifecycle,
        (0.5, -1.0, 2.0),
        (1.0, 0.25, -0.5),
    )

    proposal = lifecycle.propose(state, builder, gradient)

    # |[1,2,3] * [.5,-1,2]| + |[11,12,13] * [1,.25,-.5]|
    expected = jnp.asarray((11.5, 5.0, 12.5), dtype=jnp.float32)
    np.testing.assert_array_equal(proposal.effective_contribution, expected)
    np.testing.assert_array_equal(proposal.observed_utility, expected)
    assert not bool(jnp.any(proposal.selected_mask))

    committed = lifecycle.commit(state, builder, proposal)
    assert bool(committed.diagnostics.applied)
    np.testing.assert_array_equal(committed.state.utility, expected)
    np.testing.assert_array_equal(committed.state.last_effective_contribution, expected)
    np.testing.assert_array_equal(committed.state.age, np.ones(3, dtype=np.uint32))
    np.testing.assert_array_equal(committed.state.support, np.ones(3, dtype=np.uint32))


def test_stable_lowest_utility_selection_honors_fixed_quota() -> None:
    lifecycle = RTUGenerateAndTest(_config(hidden_dim=4, replacement_quota=2))
    state = lifecycle.init(jr.key(3, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    # Units 0 and 1 tie at zero. Stable selection must keep ascending indices.
    gradient = _gradient(lifecycle, (0.0, 0.0, 1.0, 2.0))

    proposal = lifecycle.propose(state, builder, gradient)

    np.testing.assert_array_equal(proposal.selected_indices, np.asarray((0, 1)))
    np.testing.assert_array_equal(proposal.selected_slots, np.asarray((True, True)))
    np.testing.assert_array_equal(
        proposal.selected_mask,
        np.asarray((True, True, False, False)),
    )
    committed = lifecycle.commit(state, builder, proposal)
    assert int(committed.diagnostics.selected_count) == 2
    np.testing.assert_array_equal(committed.state.replacement_words, np.asarray((0, 2)))
    np.testing.assert_array_equal(
        committed.state.replacement_event_words,
        np.asarray((0, 1)),
    )
    assert int(committed.state.replacement_event_count) == 1


def test_required_causal_deletion_evidence_overrides_sensitivity_ranking() -> None:
    lifecycle = RTUGenerateAndTest(_config(hidden_dim=4, replacement_quota=2))
    state = lifecycle.init(jr.key(301, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    # The legacy sensitivity proxy would choose units 0 and 1.  Frozen-head
    # causal deletion says units 1 and 3 have the least retained utility.
    gradient = _gradient(lifecycle, (0.0, 1.0, 2.0, 3.0))
    deletion_loss_change = jnp.asarray((100.0, 0.0, 50.0, 1.0), dtype=jnp.float32)

    proposal = lifecycle.propose(
        state,
        builder,
        gradient,
        causal_deletion_loss_change=deletion_loss_change,
        require_causal_evidence=True,
    )

    np.testing.assert_array_equal(proposal.selected_indices, np.asarray((1, 3)))
    np.testing.assert_array_equal(
        proposal.selected_mask,
        np.asarray((False, True, False, True)),
    )
    assert bool(proposal.causal_deletion_evidence_available)
    assert bool(proposal.causal_evidence_required)
    assert bool(jnp.all(proposal.observed_causal_evidence_count == 1))
    np.testing.assert_array_equal(
        proposal.selection_utility,
        proposal.observed_causal_utility,
    )


def test_missing_required_causal_evidence_defers_only_replacement() -> None:
    lifecycle = RTUGenerateAndTest(_config(hidden_dim=3, replacement_quota=1))
    state = lifecycle.init(jr.key(302, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)

    proposal = lifecycle.propose(
        state,
        builder,
        _gradient(lifecycle, (0.0, 1.0, 2.0)),
        require_causal_evidence=True,
    )
    committed = lifecycle.commit(state, builder, proposal)

    assert bool(proposal.valid)
    assert not bool(proposal.causal_deletion_evidence_available)
    assert not bool(jnp.any(proposal.selected_mask))
    assert bool(committed.diagnostics.applied)
    np.testing.assert_array_equal(committed.state.observation_words, [0, 1])
    np.testing.assert_array_equal(committed.state.replacement_event_words, [0, 0])
    np.testing.assert_array_equal(
        jr.key_data(committed.state.rng_key),
        jr.key_data(state.rng_key),
    )


def test_missing_vector_available_flag_is_jax_safe_and_fails_closed() -> None:
    lifecycle = RTUGenerateAndTest(_config(hidden_dim=3, replacement_quota=1))
    state = lifecycle.init(jr.key(304, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (0.0, 1.0, 2.0))

    compiled = jax.jit(
        lambda available: lifecycle.propose(
            state,
            builder,
            gradient,
            causal_deletion_evidence_available=available,
            require_causal_evidence=True,
        )
    )
    missing = compiled(jnp.asarray(False, dtype=jnp.bool_))
    contradictory = compiled(jnp.asarray(True, dtype=jnp.bool_))

    assert bool(missing.valid)
    assert not bool(jnp.any(missing.selected_mask))
    assert not bool(contradictory.causal_deletion_evidence_valid)
    assert not bool(contradictory.valid)


def test_immature_causal_evidence_accumulates_without_legacy_rank_fallback() -> None:
    lifecycle = RTUGenerateAndTest(
        _config(
            hidden_dim=3,
            replacement_quota=1,
            minimum_support=0,
            minimum_causal_evidence=3,
        )
    )
    state = lifecycle.init(jr.key(303, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (0.0, 1.0, 2.0))
    deletion = jnp.asarray((0.0, 1.0, 2.0), dtype=jnp.float32)

    results = []
    for _ in range(3):
        proposal = lifecycle.propose(
            state,
            builder,
            gradient,
            causal_deletion_loss_change=deletion,
            require_causal_evidence=True,
        )
        result = lifecycle.commit(state, builder, proposal)
        results.append(result)
        state = result.state
        builder = result.builder_state

    assert not bool(jnp.any(results[0].diagnostics.selected_mask))
    assert not bool(jnp.any(results[1].diagnostics.selected_mask))
    assert bool(results[2].diagnostics.selected_mask[0])
    np.testing.assert_array_equal(results[1].state.causal_evidence_count, 2)
    assert int(results[2].state.causal_evidence_count[0]) == 0


def test_replacement_event_clock_is_distinct_and_active_owner_can_defer() -> None:
    lifecycle = RTUGenerateAndTest(
        _config(hidden_dim=4, replacement_quota=2)
    )
    state = lifecycle.init(jr.key(31, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (0.0, 0.0, 1.0, 2.0))

    deferred = lifecycle.propose(
        state,
        builder,
        gradient,
        replacement_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert not bool(jnp.any(deferred.selected_mask))
    observed = lifecycle.commit(state, builder, deferred)
    assert bool(observed.diagnostics.applied)
    np.testing.assert_array_equal(observed.state.observation_words, [0, 1])
    np.testing.assert_array_equal(observed.state.replacement_words, [0, 0])
    np.testing.assert_array_equal(
        observed.state.replacement_event_words,
        [0, 0],
    )

    first = lifecycle.transact(observed.state, observed.builder_state, gradient)
    second = lifecycle.transact(first.state, first.builder_state, gradient)
    assert bool(jnp.all(first.diagnostics.selected_mask[:2]))
    assert bool(jnp.all(second.diagnostics.selected_mask[:2]))
    np.testing.assert_array_equal(second.state.replacement_words, [0, 4])
    np.testing.assert_array_equal(
        second.state.replacement_event_words,
        [0, 2],
    )
    np.testing.assert_array_equal(second.builder_state.update_words, [0, 2])


def test_protection_warmup_age_and_support_floors_gate_replacement() -> None:
    lifecycle = RTUGenerateAndTest(
        _config(
            warmup_observations=3,
            minimum_age=3,
            minimum_support=3,
            protected_units=(0,),
        )
    )
    composition = lifecycle.init_composition(
        jr.key(4, impl="threefry2x32"),
        jr.key(5, impl="threefry2x32"),
    )
    composition = composition.replace(builder=_payload_builder(lifecycle))
    gradient = _gradient(lifecycle, (0.1, 0.2, 0.3))

    first = lifecycle.transact(composition.lifecycle, composition.builder, gradient)
    second = lifecycle.transact(first.state, first.builder_state, gradient)
    third = lifecycle.transact(second.state, second.builder_state, gradient)

    assert not bool(jnp.any(first.diagnostics.selected_mask))
    assert not bool(jnp.any(second.diagnostics.selected_mask))
    assert bool(third.diagnostics.selected_mask[1])
    assert not bool(third.diagnostics.selected_mask[0])
    assert int(third.state.age[0]) == 3
    assert int(third.state.support[0]) == 3


def test_whole_unit_scrub_preserves_every_nonselected_builder_bit() -> None:
    lifecycle = RTUGenerateAndTest(_config())
    state = lifecycle.init(jr.key(6, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    before_parts = _parameter_parts(lifecycle, builder.parameters)
    # Contributions [1, 0, 6] select only unit 1.
    proposal = lifecycle.propose(state, builder, _gradient(lifecycle, (1.0, 0.0, 2.0)))
    assert proposal.ordinary_learning_diagnostics is None
    assert int(proposal.selected_indices[0]) == 1
    committed = lifecycle.commit(state, builder, proposal)
    after = committed.builder_state
    after_parts = _parameter_parts(lifecycle, after.parameters)
    survivors = jnp.asarray((0, 2), dtype=jnp.int32)

    for before, candidate in zip(before_parts, after_parts, strict=True):
        if before.ndim == 1:
            np.testing.assert_array_equal(candidate[survivors], before[survivors])
            assert float(candidate[1]) != float(before[1])
        else:
            np.testing.assert_array_equal(candidate[survivors], before[survivors])
            assert not np.array_equal(np.asarray(candidate[1]), np.asarray(before[1]))
    np.testing.assert_array_equal(after.rtu_state.real, np.asarray((1.0, 0.0, 3.0)))
    np.testing.assert_array_equal(
        after.rtu_state.imaginary,
        np.asarray((11.0, 0.0, 13.0)),
    )
    for before, candidate in zip(
        jax.tree.leaves(builder.sensitivities),
        jax.tree.leaves(after.sensitivities),
        strict=True,
    ):
        np.testing.assert_array_equal(candidate[:, survivors], before[:, survivors])
        np.testing.assert_array_equal(candidate[:, 1], np.zeros_like(candidate[:, 1]))
    np.testing.assert_array_equal(after.step_words, builder.step_words)
    np.testing.assert_array_equal(after.update_words, np.asarray((0, 1), dtype=np.uint32))
    assert float(after.last_gradient_norm) == 7.5
    assert bool(lifecycle.builder.state_valid(after))


def test_taylor_state_scrubs_selected_trace_source_and_delta_slices() -> None:
    lifecycle = RTUGenerateAndTest(_config(taylor=True))
    state = lifecycle.init(jr.key(7, impl="threefry2x32"))
    pre_update = _payload_builder(lifecycle, taylor_scale=0.5)
    gradient = _gradient(lifecycle, (0.0, 1.0, 2.0), (0.0, 0.25, 0.5))
    learning_proposal = lifecycle.builder.propose_learning_update(pre_update, gradient)
    live, learning_diagnostics = lifecycle.builder.commit_learning_update(
        pre_update,
        learning_proposal,
    )
    assert bool(learning_diagnostics.applied)
    assert live.sensitivity_parameter_delta is not None
    assert bool(jnp.any(live.sensitivity_parameter_delta != 0.0))

    proposal = lifecycle.propose(state, pre_update, gradient, learning_proposal)
    assert bool(proposal.ordinary_advance_valid)
    selected = int(proposal.selected_indices[0])
    committed = lifecycle.commit(state, live, proposal)
    candidate = committed.builder_state
    assert bool(committed.diagnostics.applied)
    assert candidate.taylor_trace is not None
    assert candidate.sensitivity_source_parameters is not None
    assert candidate.sensitivity_parameter_delta is not None
    mask = np.ones(lifecycle.config.builder.hidden_dim, dtype=bool)
    mask[selected] = False
    for before, after in zip(
        jax.tree.leaves(cast(RTUSensitivities, live.taylor_trace)),
        jax.tree.leaves(candidate.taylor_trace),
        strict=True,
    ):
        np.testing.assert_array_equal(after[:, mask], before[:, mask])
        np.testing.assert_array_equal(after[:, selected], np.zeros_like(after[:, selected]))

    source_parts = _parameter_parts(
        lifecycle,
        candidate.sensitivity_source_parameters,
    )
    parameter_parts = _parameter_parts(lifecycle, candidate.parameters)
    delta_parts = _parameter_parts(lifecycle, candidate.sensitivity_parameter_delta)
    for source, parameters, delta in zip(
        source_parts,
        parameter_parts,
        delta_parts,
        strict=True,
    ):
        np.testing.assert_array_equal(source[selected], parameters[selected])
        np.testing.assert_array_equal(delta[selected], np.zeros_like(delta[selected]))
    assert bool(lifecycle.builder.state_valid(candidate))


def test_advanced_destination_is_exact_and_unverified_destination_rolls_back() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=50))
    state = lifecycle.init(jr.key(8, impl="threefry2x32"))
    pre_update = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (1.0, 2.0, 3.0))
    learning_proposal = lifecycle.builder.propose_learning_update(pre_update, gradient)
    live, diagnostics = lifecycle.builder.commit_learning_update(
        pre_update,
        learning_proposal,
    )
    assert bool(diagnostics.applied)
    proposal = lifecycle.propose(state, pre_update, gradient, learning_proposal)
    assert proposal.ordinary_learning_diagnostics is not None
    _tree_assert_exact(proposal.ordinary_learning_diagnostics, diagnostics)

    wrong_destination = lifecycle.commit(state, pre_update, proposal)
    assert not bool(wrong_destination.diagnostics.applied)
    _tree_assert_exact(wrong_destination.state, state)
    _tree_assert_exact(wrong_destination.builder_state, pre_update)

    accepted = lifecycle.commit(state, live, proposal)
    assert bool(accepted.diagnostics.applied)
    np.testing.assert_array_equal(accepted.builder_state.parameters, live.parameters)
    np.testing.assert_array_equal(accepted.builder_state.update_words, live.update_words)

    tampered_diagnostics = proposal.ordinary_learning_diagnostics.replace(
        applied=~proposal.ordinary_learning_diagnostics.applied,
    )
    tampered_proposal = proposal.replace(
        ordinary_learning_diagnostics=tampered_diagnostics,
    )
    rejected = lifecycle.commit(state, live, tampered_proposal)
    assert not bool(rejected.diagnostics.proposal_integrity)
    assert not bool(rejected.diagnostics.applied)
    _tree_assert_exact(rejected.state, state)
    _tree_assert_exact(rejected.builder_state, live)


def test_content_bound_normal_advance_recomputes_exact_live_destination() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=50))
    lifecycle_state = lifecycle.init(jr.key(81, impl="threefry2x32"))
    source_builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (0.5, -1.0, 2.0), (1.0, 0.25, -0.5))
    next_observation = jnp.asarray((0.25, -0.75), dtype=jnp.float32)
    receipt = lifecycle.make_advance_receipt(
        source_builder,
        bootstrap_observation=next_observation,
        previous_action=jnp.asarray(-1, dtype=jnp.int32),
        previous_reward=jnp.asarray(0.5, dtype=jnp.float32),
        previous_discount=jnp.asarray(0.9, dtype=jnp.float32),
        episode_boundary=jnp.asarray(False, dtype=jnp.bool_),
        restart_observation=next_observation,
    )
    learning = lifecycle.builder.propose_learning_update(source_builder, gradient)
    manual_advance = lifecycle.builder.update_with_status(
        source_builder,
        next_observation,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    manual_live, manual_learning = lifecycle.builder.commit_learning_update(
        manual_advance.state,
        learning,
    )
    assert bool(manual_advance.transition_applied)
    assert bool(manual_learning.applied)

    proposal = lifecycle.propose(
        lifecycle_state,
        source_builder,
        gradient,
        learning,
        receipt,
    )
    assert bool(proposal.advance_receipt_valid)
    assert bool(proposal.ordinary_advance_valid)
    _tree_assert_exact(proposal.live_builder_state, manual_live)
    # Utility uses the causal source activation, never the advanced activation.
    np.testing.assert_array_equal(
        proposal.effective_contribution,
        np.asarray((11.5, 5.0, 12.5), dtype=np.float32),
    )
    committed = lifecycle.commit(lifecycle_state, manual_live, proposal)
    assert bool(committed.diagnostics.applied)
    _tree_assert_exact(committed.builder_state, manual_live)

    invented = lifecycle.builder.update_with_status(
        source_builder,
        jnp.asarray((-0.9, 0.4), dtype=jnp.float32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    ).state
    invented, invented_learning = lifecycle.builder.commit_learning_update(
        invented,
        learning,
    )
    assert bool(invented_learning.applied)
    rejected = lifecycle.commit(lifecycle_state, invented, proposal)
    assert not bool(rejected.diagnostics.live_builder_matches)
    assert not bool(rejected.diagnostics.applied)
    _tree_assert_exact(rejected.state, lifecycle_state)
    _tree_assert_exact(rejected.builder_state, invented)


def test_boundary_receipt_recomputes_bootstrap_reset_restart_sequence() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=50, taylor=True))
    lifecycle_state = lifecycle.init(jr.key(82, impl="threefry2x32"))
    source_builder = _payload_builder(lifecycle, taylor_scale=0.25)
    gradient = _gradient(lifecycle, (0.1, 0.2, 0.3), (0.3, 0.2, 0.1))
    bootstrap_observation = jnp.asarray((0.7, -0.2), dtype=jnp.float32)
    restart_observation = jnp.asarray((-0.4, 0.8), dtype=jnp.float32)
    receipt = lifecycle.make_advance_receipt(
        source_builder,
        bootstrap_observation=bootstrap_observation,
        previous_action=jnp.asarray(-1, dtype=jnp.int32),
        previous_reward=jnp.asarray(-0.25, dtype=jnp.float32),
        previous_discount=jnp.asarray(0.0, dtype=jnp.float32),
        episode_boundary=jnp.asarray(True, dtype=jnp.bool_),
        restart_observation=restart_observation,
    )
    learning = lifecycle.builder.propose_learning_update(source_builder, gradient)
    bootstrap = lifecycle.builder.update_with_status(
        source_builder,
        bootstrap_observation,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(-0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    reset = lifecycle.builder.reset_episode(bootstrap.state)
    restart = lifecycle.builder.update_with_status(
        reset,
        restart_observation,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    manual_live, learning_diagnostics = lifecycle.builder.commit_learning_update(
        restart.state,
        learning,
    )
    assert bool(bootstrap.transition_applied)
    assert bool(restart.transition_applied)
    assert bool(learning_diagnostics.applied)
    proposal = lifecycle.propose(
        lifecycle_state,
        source_builder,
        gradient,
        learning,
        receipt,
    )
    assert int(receipt.sequence_length) == 2
    _tree_assert_exact(proposal.live_builder_state, manual_live)
    result = lifecycle.commit(lifecycle_state, manual_live, proposal)
    assert bool(result.diagnostics.applied)
    np.testing.assert_array_equal(result.builder_state.step_words, (0, 2))


def test_stale_or_mutated_advance_receipt_is_an_exact_noop() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=50))
    lifecycle_state = lifecycle.init(jr.key(83, impl="threefry2x32"))
    source_builder = _payload_builder(lifecycle)
    observation = jnp.asarray((0.3, -0.1), dtype=jnp.float32)
    receipt = lifecycle.make_advance_receipt(
        source_builder,
        bootstrap_observation=observation,
        previous_action=jnp.asarray(-1, dtype=jnp.int32),
        previous_reward=jnp.asarray(0.1, dtype=jnp.float32),
        previous_discount=jnp.asarray(0.95, dtype=jnp.float32),
        episode_boundary=jnp.asarray(False, dtype=jnp.bool_),
        restart_observation=observation,
    )
    gradient = _gradient(lifecycle, (0.2, 0.4, 0.6))
    learning = lifecycle.builder.propose_learning_update(source_builder, gradient)
    canonical = lifecycle.propose(
        lifecycle_state,
        source_builder,
        gradient,
        learning,
        receipt,
    )

    mutated_receipt = receipt.replace(
        bootstrap_observation=receipt.bootstrap_observation.at[0].add(
            jnp.asarray(0.5, dtype=jnp.float32)
        )
    )
    invalid = lifecycle.propose(
        lifecycle_state,
        source_builder,
        gradient,
        learning,
        mutated_receipt,
    )
    assert not bool(invalid.advance_receipt_valid)
    assert not bool(invalid.valid)
    invalid_result = lifecycle.commit(
        lifecycle_state,
        invalid.live_builder_state,
        invalid,
    )
    assert not bool(invalid_result.diagnostics.applied)
    _tree_assert_exact(invalid_result.state, lifecycle_state)

    tampered_proposal = canonical.replace(advance_receipt=mutated_receipt)
    tampered = lifecycle.commit(
        lifecycle_state,
        canonical.live_builder_state,
        tampered_proposal,
    )
    assert not bool(tampered.diagnostics.proposal_integrity)
    assert not bool(tampered.diagnostics.applied)

    newer_source = lifecycle.builder.update_with_status(
        source_builder,
        observation,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(0.95, dtype=jnp.float32),
    ).state
    stale_learning = lifecycle.builder.propose_learning_update(newer_source, gradient)
    stale = lifecycle.propose(
        lifecycle_state,
        newer_source,
        gradient,
        stale_learning,
        receipt,
    )
    assert not bool(stale.advance_receipt_valid)
    assert not bool(stale.valid)


def test_stale_and_tampered_proposals_fail_closed_with_exact_rollback() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=10))
    state = lifecycle.init(jr.key(9, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (1.0, 2.0, 3.0))
    proposal = lifecycle.propose(state, builder, gradient)
    newer = lifecycle.transact(state, builder, gradient)

    stale = lifecycle.commit(newer.state, builder, proposal)
    assert not bool(stale.diagnostics.lifecycle_source_matches)
    assert not bool(stale.diagnostics.applied)
    _tree_assert_exact(stale.state, newer.state)
    _tree_assert_exact(stale.builder_state, builder)

    tampered = proposal.replace(
        observed_utility=proposal.observed_utility.at[0].add(jnp.float32(1.0))
    )
    rejected = lifecycle.commit(state, builder, tampered)
    assert not bool(rejected.diagnostics.proposal_integrity)
    assert not bool(rejected.diagnostics.applied)
    _tree_assert_exact(rejected.state, state)
    _tree_assert_exact(rejected.builder_state, builder)


def test_numeric_and_clock_capacity_failures_do_not_advance_any_owned_state() -> None:
    lifecycle = RTUGenerateAndTest(_config())
    state = lifecycle.init(jr.key(10, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (1.0, 2.0, 3.0))

    bad_gradient = gradient.at[-1].set(jnp.nan)
    numeric = lifecycle.propose(state, builder, bad_gradient)
    assert not bool(numeric.input_valid)
    numeric_result = lifecycle.commit(state, builder, numeric)
    _tree_assert_exact(numeric_result.state, state)
    _tree_assert_exact(numeric_result.builder_state, builder)

    bad_causal_evidence = lifecycle.propose(
        state,
        builder,
        gradient,
        causal_deletion_loss_change=jnp.asarray(
            (0.0, jnp.nan, 1.0),
            dtype=jnp.float32,
        ),
        require_causal_evidence=True,
    )
    assert not bool(bad_causal_evidence.causal_deletion_evidence_valid)
    assert not bool(bad_causal_evidence.valid)
    bad_causal_result = lifecycle.commit(
        state,
        builder,
        bad_causal_evidence,
    )
    assert not bool(bad_causal_result.diagnostics.applied)
    _tree_assert_exact(bad_causal_result.state, state)
    _tree_assert_exact(bad_causal_result.builder_state, builder)

    exhausted = state.replace(
        observation_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        observation_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32),
        replacement_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        replacement_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32),
        replacement_event_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        replacement_event_words=jnp.asarray(
            (2**32 - 1, 2**32 - 1),
            dtype=jnp.uint32,
        ),
        age=jnp.asarray((0, 1, 2), dtype=jnp.uint32),
        last_replaced_mask=jnp.asarray((True, False, False), dtype=jnp.bool_),
    )
    assert bool(lifecycle.state_valid(exhausted))
    clock = lifecycle.propose(exhausted, builder, gradient)
    assert not bool(clock.observation_capacity_available)
    clock_result = lifecycle.commit(exhausted, builder, clock)
    _tree_assert_exact(clock_result.state, exhausted)

    aged = state.replace(
        age=jnp.full((3,), 2**32 - 1, dtype=jnp.uint32),
        observation_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        observation_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(aged))
    age_proposal = lifecycle.propose(aged, builder, gradient)
    assert not bool(age_proposal.per_unit_capacity_available)
    age_result = lifecycle.commit(aged, builder, age_proposal)
    _tree_assert_exact(age_result.state, aged)

    causal_exhausted = state.replace(
        age=jnp.full((3,), 2**32 - 1, dtype=jnp.uint32),
        causal_evidence_count=jnp.full((3,), 2**32 - 1, dtype=jnp.uint32),
        observation_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        observation_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(causal_exhausted))
    causal_capacity = lifecycle.propose(
        causal_exhausted,
        builder,
        gradient,
        causal_deletion_loss_change=jnp.zeros((3,), dtype=jnp.float32),
        require_causal_evidence=True,
    )
    assert not bool(causal_capacity.per_unit_capacity_available)
    causal_capacity_result = lifecycle.commit(
        causal_exhausted,
        builder,
        causal_capacity,
    )
    _tree_assert_exact(causal_capacity_result.state, causal_exhausted)

    builder_exhausted = builder.replace(
        update_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        update_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32),
    )
    assert bool(lifecycle.builder.state_valid(builder_exhausted))
    builder_proposal = lifecycle.propose(state, builder_exhausted, gradient)
    assert not bool(builder_proposal.builder_capacity_available)
    builder_result = lifecycle.commit(state, builder_exhausted, builder_proposal)
    _tree_assert_exact(builder_result.state, state)
    _tree_assert_exact(builder_result.builder_state, builder_exhausted)


def test_fixed_rng_draw_accounting_is_pure_and_rejection_retains_key() -> None:
    lifecycle = RTUGenerateAndTest(
        _config(replacement_interval=100, replacement_quota=2)
    )
    state = lifecycle.init(jr.key(11, impl="threefry2x32"))
    builder = _payload_builder(lifecycle)
    gradient = _gradient(lifecycle, (1.0, 2.0, 3.0))
    first = lifecycle.propose(state, builder, gradient)
    second = lifecycle.propose(state, builder, gradient)
    _tree_assert_exact(first, second)
    np.testing.assert_array_equal(
        jr.key_data(first.candidate_state.rng_key),
        jr.key_data(state.rng_key),
    )
    accepted = lifecycle.commit(state, builder, first)
    assert bool(accepted.diagnostics.applied)
    np.testing.assert_array_equal(
        jr.key_data(accepted.state.rng_key),
        jr.key_data(state.rng_key),
    )

    rejected = lifecycle.commit(
        state,
        builder,
        first.replace(fresh_parameter_slices=first.fresh_parameter_slices.at[0, 0].add(1.0)),
    )
    assert not bool(rejected.diagnostics.applied)
    np.testing.assert_array_equal(jr.key_data(rejected.state.rng_key), jr.key_data(state.rng_key))


def test_checkpoint_config_and_resource_budget_are_exact(
    tmp_path: Path,
) -> None:
    lifecycle = RTUGenerateAndTest(_config(taylor=True, replacement_quota=2))
    composition = lifecycle.init_composition(
        jr.key(12, impl="threefry2x32"),
        jr.key(13, impl="threefry2x32"),
    )
    composition = composition.replace(builder=_payload_builder(lifecycle))
    result = lifecycle.transact(
        composition.lifecycle,
        composition.builder,
        _gradient(lifecycle, (1.0, 0.5, 2.0)),
    )
    budget = lifecycle.resource_budget(result.state, result.builder_state)
    assert budget.composition_state_nbytes == (
        budget.lifecycle_state_nbytes + budget.builder_state_nbytes
    )
    assert budget.maximum_proposal_nbytes > budget.composition_state_nbytes
    assert budget.random_replacement_roots_per_observation == 2
    assert budget.random_subkeys_per_replacement_root == 4
    assert RTUGenerateAndTestConfig.from_config(lifecycle.to_config()) == lifecycle.config

    path = tmp_path / "rtu-generate-and-test"
    save_rtu_generate_and_test_checkpoint(lifecycle, result.composition, path)
    restored_lifecycle, restored = load_rtu_generate_and_test_checkpoint(path)
    assert restored_lifecycle.to_config() == lifecycle.to_config()
    _tree_assert_exact(restored, result.composition)
    assert restored_lifecycle.resource_budget(
        restored.lifecycle,
        restored.builder,
    ) == budget


def test_eager_jit_and_scan_are_bit_exact() -> None:
    lifecycle = RTUGenerateAndTest(_config(replacement_interval=2))
    composition = lifecycle.init_composition(
        jr.key(14, impl="threefry2x32"),
        jr.key(15, impl="threefry2x32"),
    )
    composition = composition.replace(builder=_payload_builder(lifecycle))
    gradient = _gradient(lifecycle, (1.0, 2.0, 3.0))
    eager = lifecycle.transact(composition.lifecycle, composition.builder, gradient)
    compiled = jax.jit(
        lambda state, builder, value: lifecycle.transact(state, builder, value)
    )(composition.lifecycle, composition.builder, gradient)
    _tree_assert_exact(eager, compiled)

    gradients = jnp.stack((gradient, gradient * 0.5, gradient * 0.25))

    def scan_body(
        carry: RTUGenerateAndTestCompositionState,
        value: jax.Array,
    ) -> tuple[RTUGenerateAndTestCompositionState, jax.Array]:
        result = lifecycle.transact(carry.lifecycle, carry.builder, value)
        return result.composition, result.diagnostics.selected_count

    eager_scan = jax.lax.scan(scan_body, composition, gradients)
    compiled_scan = jax.jit(lambda carry: jax.lax.scan(scan_body, carry, gradients))(
        composition
    )
    _tree_assert_exact(eager_scan, compiled_scan)


@pytest.mark.parametrize(
    "bad_key",
    (
        pytest.param(jr.PRNGKey(1), id="legacy-key"),
        pytest.param(jr.key(1, impl="rbg"), id="non-threefry-key"),
    ),
)
def test_typed_threefry_and_strict_configuration_contracts(bad_key: jax.Array) -> None:
    lifecycle = RTUGenerateAndTest(_config())
    with pytest.raises(TypeError, match="typed Threefry"):
        lifecycle.init(bad_key)
    with pytest.raises(ValueError, match="sorted and unique"):
        dataclasses.replace(_config(), protected_units=(2, 1))
    with pytest.raises(ValueError, match="replacement_quota"):
        dataclasses.replace(_config(), replacement_quota=4)


def test_public_roots_export_exact_rtu_lifecycle_symbols() -> None:
    import alberta_framework as package_root
    import alberta_framework.core as core_root
    from alberta_framework.core import rtu_generate_and_test

    for name in rtu_generate_and_test.__all__:
        implementation = getattr(rtu_generate_and_test, name)
        assert getattr(core_root, name) is implementation
        assert getattr(package_root, name) is implementation
        assert name in core_root.__all__
        assert name in package_root.__all__
