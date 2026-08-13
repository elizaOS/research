"""Shared internal mechanics for strict RTU objective-feature recycling.

The functions here carry no policy, target, evidence, or promotion authority.
They only enforce the common uint32 transaction horizon and zero every
comprehensive-objective parameter/cache axis owned by recycled complex units.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Bool

from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectivesState,
)
from alberta_framework.core.options import STOMPState
from alberta_framework.core.state_builder import RecurrentTraceUnitStateBuilderState

_UINT32_MAX = 2**32 - 1


def _rtu_global_lifetime_state_valid(words: Array) -> Bool[Array, ""]:
    """Require the composed RTU transaction clock to remain uint32-bounded."""

    return words[0] == jnp.asarray(0, dtype=jnp.uint32)


def _rtu_global_lifetime_capacity(words: Array) -> Bool[Array, ""]:
    """Report whether one more RTU transaction fits the declared fail-stop."""

    return _rtu_global_lifetime_state_valid(words) & (
        words[1] < jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    )


def _selected_float32_axes_are_positive_zero(
    value: Array,
    selected_mask: Array,
) -> Bool[Array, ""]:
    """Require selected last-axis entries to have the canonical +0.0 bits."""

    positive_zero = lax.bitcast_convert_type(value, jnp.uint32) == jnp.uint32(0)
    return jnp.all((~selected_mask) | positive_zero)


def _scrub_objective_representation_axes(
    state: ComprehensiveStateObjectivesState,
    reset_mask: Array,
) -> ComprehensiveStateObjectivesState:
    """Zero every comprehensive-head column owned by recycled features."""

    def scrub(value: Array) -> Array:
        return jnp.where(reset_mask, jnp.asarray(0.0, dtype=value.dtype), value)

    return cast(
        ComprehensiveStateObjectivesState,
        dataclasses.replace(
            cast(Any, state),
            observation_weights=scrub(state.observation_weights),
            latent_weights=scrub(state.latent_weights),
            reward_weights=scrub(state.reward_weights),
            termination_weights=scrub(state.termination_weights),
            gvf_weights=scrub(state.gvf_weights),
            value_weights=scrub(state.value_weights),
            advantage_weights=scrub(state.advantage_weights),
            inverse_current_weights=scrub(state.inverse_current_weights),
            inverse_next_weights=scrub(state.inverse_next_weights),
            pending_representation=scrub(state.pending_representation),
        ),
    )


def _rtu_builder_replacement_scrub_valid(
    state: RecurrentTraceUnitStateBuilderState,
    selected_mask: Array,
    *,
    event_dim: int,
) -> Bool[Array, ""]:
    """Validate every reset RTU activation, trace, and Taylor delta slice."""

    polar_keep = (~selected_mask)[None, :]
    input_keep = (~selected_mask)[None, :, None]
    valid = (
        _selected_float32_axes_are_positive_zero(state.rtu_state.real, selected_mask)
        & _selected_float32_axes_are_positive_zero(
            state.rtu_state.imaginary,
            selected_mask,
        )
        & _selected_float32_axes_are_positive_zero(
            state.sensitivities.nu_log,
            ~polar_keep,
        )
        & _selected_float32_axes_are_positive_zero(
            state.sensitivities.theta_log,
            ~polar_keep,
        )
        & _selected_float32_axes_are_positive_zero(
            state.sensitivities.b_real,
            ~input_keep,
        )
        & _selected_float32_axes_are_positive_zero(
            state.sensitivities.b_imag,
            ~input_keep,
        )
    )
    if state.taylor_trace is not None:
        valid = (
            valid
            & _selected_float32_axes_are_positive_zero(
                state.taylor_trace.nu_log,
                ~polar_keep,
            )
            & _selected_float32_axes_are_positive_zero(
                state.taylor_trace.theta_log,
                ~polar_keep,
            )
            & _selected_float32_axes_are_positive_zero(
                state.taylor_trace.b_real,
                ~input_keep,
            )
            & _selected_float32_axes_are_positive_zero(
                state.taylor_trace.b_imag,
                ~input_keep,
            )
        )
    if state.sensitivity_parameter_delta is not None:
        parameter_mask = jnp.concatenate(
            (
                selected_mask,
                selected_mask,
                jnp.repeat(selected_mask, event_dim),
                jnp.repeat(selected_mask, event_dim),
            )
        )
        valid = valid & _selected_float32_axes_are_positive_zero(
            state.sensitivity_parameter_delta,
            parameter_mask,
        )
    return valid


def _linear_stomp_replacement_scrub_valid(
    state: STOMPState,
    reset_mask: Array,
) -> Bool[Array, ""]:
    """Validate every linear STOMP coefficient/cache axis reset at recycling."""

    checks = [
        _selected_float32_axes_are_positive_zero(state.base_last_obs, reset_mask),
        _selected_float32_axes_are_positive_zero(
            state.option_policies.q_weights,
            reset_mask,
        ),
        _selected_float32_axes_are_positive_zero(
            state.option_policies.traces,
            reset_mask,
        ),
    ]
    checks.extend(
        _selected_float32_axes_are_positive_zero(weights, reset_mask)
        for weights in state.base_learner_state.head_params.weights
    )
    checks.extend(
        _selected_float32_axes_are_positive_zero(weight_trace, reset_mask)
        for weight_trace, _ in state.base_learner_state.head_traces
    )
    model_reset = reset_mask[None, :, None] | reset_mask[None, None, :]
    checks.append(
        jnp.all(
            (~model_reset)
            | (
                lax.bitcast_convert_type(
                    state.option_models.next_state_weights,
                    jnp.uint32,
                )
                == jnp.uint32(0)
            )
        )
    )
    return jnp.all(jnp.stack(tuple(checks)))


__all__ = [
    "_linear_stomp_replacement_scrub_valid",
    "_rtu_builder_replacement_scrub_valid",
    "_rtu_global_lifetime_capacity",
    "_rtu_global_lifetime_state_valid",
    "_selected_float32_axes_are_positive_zero",
    "_scrub_objective_representation_axes",
]
