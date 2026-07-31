"""Numerical contracts for the opt-in RTU Taylor-trace correction."""

from __future__ import annotations

import math
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.flatten_util import ravel_pytree

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
    RTUParameters,
    RTUSensitivities,
    RTUState,
    obgd_update,
    rtu_forward,
    rtu_network_encode,
    rtu_step,
    rtu_taylor_step,
    zero_rtu_sensitivities,
    zero_rtu_state,
)

pytestmark = pytest.mark.slow

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT_CONFIG_FIELDS = (
    "gamma",
    "actor_lamda",
    "critic_lamda",
    "actor_alpha",
    "critic_alpha",
    "actor_kappa",
    "critic_kappa",
    "entropy_coefficient",
    "temperature",
    "sparsity",
    "r_min",
    "r_max",
    "max_phase",
    "rtu_epsilon",
    "layer_norm_epsilon",
    "leaky_relu_slope",
    "normalization_epsilon",
)


def _assert_tree_all_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert str(actual_structure) == str(expected_structure)
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        assert jnp.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def _assert_tree_array_equal(actual: Any, expected: Any) -> None:
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert str(actual_structure) == str(expected_structure)
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        assert jnp.array_equal(actual_leaf, expected_leaf)


def _zero_parameter_delta(params: RTUParameters) -> RTUParameters:
    return cast(RTUParameters, jax.tree_util.tree_map(jnp.zeros_like, params))


def _expanded_flat_sensitivity(sensitivities: RTUSensitivities) -> Array:
    """Expand compressed unit-aligned columns in flattened parameter order."""
    hidden_size = sensitivities.nu_log.shape[1]
    input_dim = sensitivities.b_real.shape[2]
    indices = jnp.arange(hidden_size)

    def recurrent_leaf(compressed: Array) -> Array:
        expanded = jnp.zeros((2 * hidden_size, hidden_size), dtype=compressed.dtype)
        expanded = expanded.at[indices, indices].set(compressed[0])
        return expanded.at[hidden_size + indices, indices].set(compressed[1])

    def input_leaf(compressed: Array) -> Array:
        expanded = jnp.zeros(
            (2 * hidden_size, hidden_size, input_dim),
            dtype=compressed.dtype,
        )
        expanded = expanded.at[indices, indices, :].set(compressed[0])
        return expanded.at[hidden_size + indices, indices, :].set(compressed[1])

    expanded = RTUParameters(
        nu_log=recurrent_leaf(sensitivities.nu_log),
        theta_log=recurrent_leaf(sensitivities.theta_log),
        b_real=input_leaf(sensitivities.b_real),
        b_imag=input_leaf(sensitivities.b_imag),
    )
    return jnp.concatenate(
        tuple(
            leaf.reshape((2 * hidden_size, -1))
            for leaf in jax.tree_util.tree_leaves(expanded)
        ),
        axis=1,
    )


def _interior_params() -> RTUParameters:
    return RTUParameters(
        nu_log=jnp.asarray((-1.2, -0.35), dtype=jnp.float32),
        theta_log=jnp.asarray((-0.8, 0.45), dtype=jnp.float32),
        b_real=jnp.asarray(((0.2, -0.3), (-0.1, 0.5)), dtype=jnp.float32),
        b_imag=jnp.asarray(((0.15, 0.35), (0.45, -0.25)), dtype=jnp.float32),
    )


def test_taylor_trace_matches_full_autodiff_hessian_diagonal() -> None:
    """The analytic trace injection covers every trainable RTU parameter leaf."""
    params = _interior_params()
    state = RTUState(
        real=jnp.asarray((0.21, -0.34), dtype=jnp.float32),
        imaginary=jnp.asarray((-0.13, 0.27), dtype=jnp.float32),
    )
    inputs = jnp.asarray((0.4, -0.7), dtype=jnp.float32)
    zeros = zero_rtu_sensitivities(hidden_size=2, input_dim=2)
    _, _, taylor_trace = rtu_taylor_step(
        params,
        state,
        zeros,
        zeros,
        _zero_parameter_delta(params),
        inputs,
    )

    flat_params, unravel = ravel_pytree(params)

    def flat_output(candidate: Array) -> Array:
        next_state = rtu_forward(unravel(candidate), state, inputs)
        return jnp.concatenate((next_state.real, next_state.imaginary))

    full_hessian = jax.jacfwd(jax.jacfwd(flat_output))(flat_params)
    full_diagonal = jnp.diagonal(full_hessian, axis1=1, axis2=2)
    assert jnp.allclose(
        _expanded_flat_sensitivity(taylor_trace),
        full_diagonal,
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "branch_case",
    (
        "nu_below_clamp",
        "nu_above_clamp",
        "theta_below_clamp",
        "theta_above_clamp",
        "normalization_floor",
        "nu_lower_boundary",
        "nu_upper_boundary",
        "theta_lower_boundary",
        "theta_upper_boundary",
    ),
)
def test_taylor_hessian_matches_autodiff_across_clamp_and_floor_branches(
    branch_case: str,
) -> None:
    """The analytic branch convention matches the piecewise RTU forward map."""
    minimum_exp_log = math.log(_FLOAT32_TINY)
    maximum_nu_log = math.log(-math.log(_FLOAT32_TINY))
    maximum_theta_log = math.log(2.0 * math.pi)
    values = {
        "nu_log": -0.7,
        "theta_log": -0.2,
    }
    overrides = {
        "nu_below_clamp": ("nu_log", minimum_exp_log - 1.0),
        "nu_above_clamp": ("nu_log", maximum_nu_log + 1.0),
        "theta_below_clamp": ("theta_log", minimum_exp_log - 1.0),
        "theta_above_clamp": ("theta_log", maximum_theta_log + 1.0),
        "normalization_floor": ("nu_log", -20.0),
        "nu_lower_boundary": ("nu_log", minimum_exp_log),
        "nu_upper_boundary": ("nu_log", maximum_nu_log),
        "theta_lower_boundary": ("theta_log", minimum_exp_log),
        "theta_upper_boundary": ("theta_log", maximum_theta_log),
    }
    field_name, field_value = overrides[branch_case]
    values[field_name] = field_value
    params = RTUParameters(
        nu_log=jnp.asarray((values["nu_log"],), dtype=jnp.float32),
        theta_log=jnp.asarray((values["theta_log"],), dtype=jnp.float32),
        b_real=jnp.asarray(((0.35,),), dtype=jnp.float32),
        b_imag=jnp.asarray(((-0.25,),), dtype=jnp.float32),
    )
    state = RTUState(
        real=jnp.asarray((0.23,), dtype=jnp.float32),
        imaginary=jnp.asarray((-0.31,), dtype=jnp.float32),
    )
    inputs = jnp.asarray((0.47,), dtype=jnp.float32)
    zeros = zero_rtu_sensitivities(1, 1)
    _, _, taylor_trace = rtu_taylor_step(
        params,
        state,
        zeros,
        zeros,
        _zero_parameter_delta(params),
        inputs,
    )

    flat_params, unravel = ravel_pytree(params)

    def flat_output(candidate: Array) -> Array:
        next_state = rtu_forward(unravel(candidate), state, inputs)
        return jnp.concatenate((next_state.real, next_state.imaginary))

    full_hessian = jax.jacfwd(jax.jacfwd(flat_output))(flat_params)
    full_diagonal = jnp.diagonal(full_hessian, axis1=1, axis2=2)
    actual = _expanded_flat_sensitivity(taylor_trace)
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.all(jnp.isfinite(full_diagonal))
    assert jnp.allclose(actual, full_diagonal, atol=2e-6, rtol=2e-5)


def test_diagonal_correction_retains_first_order_mixed_parameter_residual() -> None:
    """Equation 16 does not inherit full-Omega second-order scaling in general."""
    params = RTUParameters(
        nu_log=jnp.asarray((-0.7,), dtype=jnp.float32),
        theta_log=jnp.asarray((-0.2,), dtype=jnp.float32),
        b_real=jnp.asarray(((0.35,),), dtype=jnp.float32),
        b_imag=jnp.asarray(((-0.25,),), dtype=jnp.float32),
    )
    state = RTUState(
        real=jnp.asarray((0.2,), dtype=jnp.float32),
        imaginary=jnp.asarray((-0.3,), dtype=jnp.float32),
    )
    inputs = jnp.asarray((0.4,), dtype=jnp.float32)
    zeros = zero_rtu_sensitivities(1, 1)
    _, _, taylor_trace = rtu_taylor_step(
        params,
        state,
        zeros,
        zeros,
        _zero_parameter_delta(params),
        inputs,
    )
    diagonal_hessian = _expanded_flat_sensitivity(taylor_trace)
    flat_params, unravel = ravel_pytree(params)

    def flat_output(candidate: Array) -> Array:
        next_state = rtu_forward(unravel(candidate), state, inputs)
        return jnp.concatenate((next_state.real, next_state.imaginary))

    def immediate_jacobian(candidate: Array) -> Array:
        return cast(Array, jax.jacfwd(flat_output)(candidate))

    initial_jacobian = immediate_jacobian(flat_params)
    full_hessian = jax.jacfwd(immediate_jacobian)(flat_params)
    direction = jnp.asarray((0.7, -0.5, 0.9, -0.8), dtype=jnp.float32)
    full_linear_change = jnp.einsum("opq,q->op", full_hessian, direction)
    diagonal_linear_change = diagonal_hessian * direction[None, :]
    mixed_change = full_linear_change - diagonal_linear_change
    assert jnp.linalg.norm(mixed_change) > 0.5 * jnp.linalg.norm(full_linear_change)

    def errors(scale: float) -> tuple[Array, Array]:
        parameter_delta = jnp.asarray(scale, dtype=jnp.float32) * direction
        actual = immediate_jacobian(flat_params + parameter_delta)
        diagonal_prediction = initial_jacobian + diagonal_hessian * parameter_delta[None, :]
        full_prediction = initial_jacobian + jnp.einsum(
            "opq,q->op",
            full_hessian,
            parameter_delta,
        )
        return (
            jnp.linalg.norm(actual - diagonal_prediction),
            jnp.linalg.norm(actual - full_prediction),
        )

    diagonal_large, full_large = errors(0.04)
    diagonal_small, full_small = errors(0.02)
    assert 0.45 < diagonal_small / diagonal_large < 0.55
    assert 0.20 < full_small / full_large < 0.30
    assert diagonal_large > 20.0 * full_large


def test_taylor_trace_matches_central_difference_of_immediate_sensitivity() -> None:
    """An independent finite difference checks all four diagonal injections."""
    params = RTUParameters(
        nu_log=jnp.asarray((-0.7,), dtype=jnp.float32),
        theta_log=jnp.asarray((-0.2,), dtype=jnp.float32),
        b_real=jnp.asarray(((0.35,),), dtype=jnp.float32),
        b_imag=jnp.asarray(((-0.25,),), dtype=jnp.float32),
    )
    state = RTUState(
        real=jnp.asarray((0.2,), dtype=jnp.float32),
        imaginary=jnp.asarray((-0.3,), dtype=jnp.float32),
    )
    inputs = jnp.asarray((0.4,), dtype=jnp.float32)
    zeros = zero_rtu_sensitivities(1, 1)
    _, _, taylor_trace = rtu_taylor_step(
        params,
        state,
        zeros,
        zeros,
        _zero_parameter_delta(params),
        inputs,
    )
    displacement = jnp.asarray(2e-3, dtype=jnp.float32)
    for name in ("nu_log", "theta_log", "b_real", "b_imag"):
        value = _coordinate(params, name)
        plus = _replace_coordinate(params, name, value + displacement)
        minus = _replace_coordinate(params, name, value - displacement)
        _, plus_immediate = rtu_step(plus, state, zeros, inputs)
        _, minus_immediate = rtu_step(minus, state, zeros, inputs)
        finite_difference = (
            _sensitivity_column(plus_immediate, name)
            - _sensitivity_column(minus_immediate, name)
        ) / (2.0 * displacement)
        assert jnp.allclose(
            _sensitivity_column(taylor_trace, name),
            finite_difference,
            atol=3e-5,
            rtol=4e-4,
        )


def test_zero_parameter_delta_is_bitwise_standard_rtrl() -> None:
    """With fixed parameters, correction reduces exactly to the existing step."""
    params = _interior_params()
    zeros = zero_rtu_sensitivities(hidden_size=2, input_dim=2)
    first_state, first_sensitivities, first_taylor_trace = rtu_taylor_step(
        params,
        zero_rtu_state(2),
        zeros,
        zeros,
        _zero_parameter_delta(params),
        jnp.asarray((0.2, -0.4), dtype=jnp.float32),
    )
    next_inputs = jnp.asarray((-0.3, 0.7), dtype=jnp.float32)
    expected_state, expected_sensitivities = rtu_step(
        params,
        first_state,
        first_sensitivities,
        next_inputs,
    )
    actual_state, actual_sensitivities, _ = rtu_taylor_step(
        params,
        first_state,
        first_sensitivities,
        first_taylor_trace,
        _zero_parameter_delta(params),
        next_inputs,
    )
    _assert_tree_array_equal(actual_state, expected_state)
    _assert_tree_array_equal(actual_sensitivities, expected_sensitivities)


def _coordinate(params: RTUParameters, name: str) -> Array:
    leaf = getattr(params, name)
    return cast(Array, leaf[0] if leaf.ndim == 1 else leaf[0, 0])


def _replace_coordinate(
    params: RTUParameters,
    name: str,
    value: Array,
) -> RTUParameters:
    leaf = getattr(params, name)
    replacement = leaf.at[0].set(value) if leaf.ndim == 1 else leaf.at[0, 0].set(value)
    return params._replace(**{name: replacement})


def _sensitivity_column(sensitivities: RTUSensitivities, name: str) -> Array:
    leaf = getattr(sensitivities, name)
    return cast(Array, leaf[:, 0] if leaf.ndim == 2 else leaf[:, 0, 0])


def _state_vector(state: RTUState) -> Array:
    return jnp.concatenate((state.real, state.imaginary))


def _state_from_vector(vector: Array) -> RTUState:
    return RTUState(real=vector[:1], imaginary=vector[1:])


def _online_parameter_motion_errors(name: str, scale: float) -> tuple[float, float]:
    """Compare columns with the formal Equation 8 fixed-trajectory target."""
    base = RTUParameters(
        nu_log=jnp.asarray((-0.7,), dtype=jnp.float32),
        theta_log=jnp.asarray((-0.2,), dtype=jnp.float32),
        b_real=jnp.asarray(((0.35,),), dtype=jnp.float32),
        b_imag=jnp.asarray(((-0.25,),), dtype=jnp.float32),
    )
    inputs = jnp.asarray(((0.4,), (-0.7,), (0.2,), (0.9,), (-0.3,)))
    offsets = jnp.asarray((0.0, 0.5, -0.25, 0.8, -0.6))
    initial_value = _coordinate(base, name)
    schedule = tuple(
        _replace_coordinate(base, name, initial_value + scale * offset)
        for offset in offsets
    )

    state = zero_rtu_state(1)
    stale = zero_rtu_sensitivities(1, 1)
    corrected = zero_rtu_sensitivities(1, 1)
    taylor_trace = zero_rtu_sensitivities(1, 1)
    history: list[tuple[RTUParameters, RTUState, Array]] = []
    for index, (params, current_inputs) in enumerate(
        zip(schedule, inputs, strict=True)
    ):
        previous_state = state
        previous_params = schedule[index - 1] if index else params
        parameter_delta = jax.tree_util.tree_map(
            lambda current, previous: current - previous,
            params,
            previous_params,
        )
        state, stale = rtu_step(params, previous_state, stale, current_inputs)
        corrected_state, corrected, taylor_trace = rtu_taylor_step(
            params,
            previous_state,
            corrected,
            taylor_trace,
            parameter_delta,
            current_inputs,
        )
        _assert_tree_array_equal(corrected_state, state)
        history.append((params, previous_state, current_inputs))

    # This is S*_t from Appendix C.2: historical J_k products are retained,
    # while every local I_k column is reevaluated at the final parameter.
    current_params = schedule[-1]
    ideal_column = jnp.zeros((2,), dtype=jnp.float32)
    for historical_params, previous_state, historical_inputs in history:
        state_jacobian = jax.jacfwd(
            lambda vector: _state_vector(
                rtu_forward(
                    historical_params,
                    _state_from_vector(vector),
                    historical_inputs,
                )
            )
        )(_state_vector(previous_state))
        current_value = _coordinate(current_params, name)
        immediate_column = jax.jacfwd(
            lambda value: _state_vector(
                rtu_forward(
                    _replace_coordinate(current_params, name, value),
                    previous_state,
                    historical_inputs,
                )
            )
        )(current_value)
        ideal_column = state_jacobian @ ideal_column + immediate_column

    stale_error = jnp.linalg.norm(_sensitivity_column(stale, name) - ideal_column)
    corrected_error = jnp.linalg.norm(
        _sensitivity_column(corrected, name) - ideal_column
    )
    return float(stale_error), float(corrected_error)


@pytest.mark.parametrize("parameter_name", ("nu_log", "theta_log", "b_real", "b_imag"))
def test_online_parameter_motion_has_first_vs_second_order_error(
    parameter_name: str,
) -> None:
    """Multiple online moves expose sign, index, leaf, and ordering mistakes."""
    stale_large, corrected_large = _online_parameter_motion_errors(
        parameter_name,
        0.04,
    )
    stale_small, corrected_small = _online_parameter_motion_errors(
        parameter_name,
        0.02,
    )

    # Halving parameter motion halves ordinary staleness but quarters the
    # Taylor residual.  Loose margins leave room for float32 roundoff while
    # still rejecting an omitted, sign-flipped, or one-step-shifted term.
    assert 0.42 < stale_small / stale_large < 0.58
    assert corrected_small / corrected_large < 0.35
    assert corrected_large < 0.12 * stale_large


def _agent_config(**overrides: Any) -> RecurrentTraceActorCriticConfig:
    values: dict[str, Any] = {
        "n_actions": 2,
        "hidden_size": 2,
        "encoder_width": 2,
        "output_width": 3,
        "sparsity": 0.0,
        "r_min": 0.1,
        "r_max": 0.9,
        "normalize_observations": False,
        "normalize_rewards": False,
        "rtrl_taylor_correction": True,
    }
    values.update(overrides)
    return RecurrentTraceActorCriticConfig(**values)


def _config_with_float_override(
    field_name: str,
    value: float,
) -> RecurrentTraceActorCriticConfig:
    values: dict[str, Any] = {
        "n_actions": 2,
        field_name: value,
    }
    return RecurrentTraceActorCriticConfig(**values)


def test_taylor_option_is_default_off_validated_and_serialized() -> None:
    default = RecurrentTraceActorCriticConfig(n_actions=2)
    assert default.rtrl_taylor_correction is False
    default_state = RecurrentTraceActorCriticAgent(default).init(2, jr.key(79))
    assert default_state.actor_taylor_trace is None
    assert default_state.critic_taylor_trace is None
    enabled = _agent_config()
    assert RecurrentTraceActorCriticConfig.from_config(
        enabled.to_config()
    ) == enabled
    with pytest.raises(ValueError, match="rtrl_taylor_correction must be a bool"):
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            rtrl_taylor_correction=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field_name", _FLOAT_CONFIG_FIELDS)
@pytest.mark.parametrize(
    "invalid_value",
    (
        _FLOAT32_MAX * 2.0,
        _FLOAT32_TINY / 2.0,
        1e-300,
    ),
)
def test_config_rejects_values_outside_finite_normal_float32(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field_name}.*finite normal float32",
    ):
        _config_with_float_override(field_name, invalid_value)


@pytest.mark.parametrize(
    "field_name",
    (
        "gamma",
        "actor_lamda",
        "critic_lamda",
        "actor_alpha",
        "critic_alpha",
        "entropy_coefficient",
        "sparsity",
        "r_min",
        "leaky_relu_slope",
    ),
)
def test_config_retains_exact_zero_where_the_field_contract_allows_it(
    field_name: str,
) -> None:
    config = _config_with_float_override(field_name, 0.0)
    assert getattr(config, field_name) == 0.0


@pytest.mark.parametrize(
    "field_name",
    (
        "actor_kappa",
        "critic_kappa",
        "temperature",
        "r_max",
        "max_phase",
        "rtu_epsilon",
        "layer_norm_epsilon",
        "normalization_epsilon",
    ),
)
def test_config_still_rejects_zero_for_positive_only_fields(field_name: str) -> None:
    with pytest.raises(ValueError):
        _config_with_float_override(field_name, 0.0)


def test_config_rejects_nonzero_radius_minimum_with_subnormal_square() -> None:
    with pytest.raises(ValueError, match="nonzero r_min squared.*normal float32"):
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            r_min=math.sqrt(_FLOAT32_TINY) / 2.0,
            r_max=0.5,
        )


def test_config_rejects_radius_interval_collapsed_by_float32_conversion() -> None:
    with pytest.raises(ValueError, match="non-empty float32 radius-squared interval"):
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            r_min=0.5,
            r_max=math.nextafter(0.5, 1.0),
        )


def test_config_rejects_radius_maximum_at_the_numerical_floor() -> None:
    with pytest.raises(ValueError, match="radius-squared|numerical floor"):
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            r_max=math.sqrt(_FLOAT32_TINY),
        )


def test_episode_boundary_resets_taylor_history_and_parameter_delta() -> None:
    """A new episode receives only its own immediate S and omega injections."""
    config = _agent_config(actor_alpha=0.2, critic_alpha=0.2)
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(80))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.3, -0.2), dtype=jnp.float32),
    )
    for reward, observation in (
        (0.7, (0.1, 0.5)),
        (-0.4, (-0.6, 0.2)),
    ):
        state = agent.update(
            state,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(observation, dtype=jnp.float32),
        ).state
    assert state.actor_taylor_trace is not None
    assert any(
        bool(jnp.any(leaf != 0.0))
        for leaf in jax.tree_util.tree_leaves(state.actor_taylor_trace)
    )

    reset_observation = jnp.asarray((0.8, -0.1), dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(1.1, dtype=jnp.float32),
        jnp.asarray((-0.2, 0.4), dtype=jnp.float32),
        episode_boundary=jnp.asarray(True),
        reset_observation=reset_observation,
    )
    actor_inputs = rtu_network_encode(
        result.state.actor_params,
        reset_observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    zeros = zero_rtu_sensitivities(config.hidden_size, config.encoder_width)
    expected_state, expected_sensitivities, expected_taylor_trace = rtu_taylor_step(
        result.state.actor_params.rtu,
        zero_rtu_state(config.hidden_size),
        zeros,
        zeros,
        _zero_parameter_delta(result.state.actor_params.rtu),
        actor_inputs,
        epsilon=config.rtu_epsilon,
    )
    _assert_tree_all_close(result.state.actor_rtu_state, expected_state)
    _assert_tree_all_close(result.state.actor_sensitivities, expected_sensitivities)
    _assert_tree_all_close(result.state.actor_taylor_trace, expected_taylor_trace)


def test_agent_uses_effective_post_projection_parameter_delta() -> None:
    """A clamped ObGD proposal must not enter the Taylor correction as motion."""
    config = _agent_config(actor_alpha=0.0, critic_alpha=0.5)
    agent = RecurrentTraceActorCriticAgent(config)
    warm_agent = RecurrentTraceActorCriticAgent(
        _agent_config(actor_alpha=0.0, critic_alpha=0.0)
    )
    state = agent.init(feature_dim=2, key=jr.key(81))
    maximum_theta_log = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))
    clamped_theta = jnp.full_like(
        state.critic_params.rtu.theta_log,
        maximum_theta_log,
    )
    state = state.replace(
        critic_params=state.critic_params._replace(
            rtu=state.critic_params.rtu._replace(theta_log=clamped_theta)
        )
    )
    state, _, _ = warm_agent.start(
        state,
        jnp.asarray((0.25, -0.35), dtype=jnp.float32),
    )
    state = warm_agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((0.15, 0.45), dtype=jnp.float32),
    ).state
    assert state.critic_taylor_trace is not None
    assert bool(jnp.any(state.critic_taylor_trace.theta_log != 0.0))
    forced_trace = state.critic_traces._replace(
        rtu=state.critic_traces.rtu._replace(
            theta_log=jnp.full_like(state.critic_traces.rtu.theta_log, 100.0)
        )
    )
    state = state.replace(critic_traces=forced_trace)
    critic_taylor_trace = state.critic_taylor_trace
    assert critic_taylor_trace is not None
    observation = jnp.asarray((-0.4, 0.6), dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(10.0, dtype=jnp.float32),
        observation,
    )

    raw_obgd = obgd_update(
        result.state.critic_traces,
        result.td_error,
        alpha=config.critic_alpha,
        kappa=config.critic_kappa,
    )
    assert jnp.all(raw_obgd.updates.rtu.theta_log > 0.0)
    assert jnp.allclose(result.state.critic_params.rtu.theta_log, clamped_theta)

    effective_delta = jax.tree_util.tree_map(
        lambda current, previous: current - previous,
        result.state.critic_params.rtu,
        state.critic_params.rtu,
    )
    critic_inputs = rtu_network_encode(
        result.state.critic_params,
        observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    expected_state, expected_sensitivities, expected_taylor_trace = rtu_taylor_step(
        result.state.critic_params.rtu,
        state.critic_rtu_state,
        state.critic_sensitivities,
        critic_taylor_trace,
        effective_delta,
        critic_inputs,
        epsilon=config.rtu_epsilon,
    )
    _assert_tree_all_close(result.state.critic_rtu_state, expected_state)
    _assert_tree_all_close(result.state.critic_sensitivities, expected_sensitivities)
    _assert_tree_all_close(result.state.critic_taylor_trace, expected_taylor_trace)

    wrong_delta = raw_obgd.updates.rtu
    _, wrong_sensitivities, _ = rtu_taylor_step(
        result.state.critic_params.rtu,
        state.critic_rtu_state,
        state.critic_sensitivities,
        critic_taylor_trace,
        wrong_delta,
        critic_inputs,
        epsilon=config.rtu_epsilon,
    )
    assert not jnp.allclose(
        result.state.critic_sensitivities.theta_log,
        wrong_sensitivities.theta_log,
        atol=1e-6,
        rtol=1e-6,
    )
