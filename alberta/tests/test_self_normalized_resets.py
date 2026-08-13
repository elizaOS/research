# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for the paper-grounded Self-Normalized Resets core."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.self_normalized_resets import (
    SELF_NORMALIZED_RESETS_SCHEMA,
    DenseReLUFreshParameters,
    DenseReLUOptimizerState,
    DenseReLUParameters,
    DenseReLUResetTarget,
    SelfNormalizedResetConfig,
    SelfNormalizedResetResult,
    SelfNormalizedResets,
    SelfNormalizedResetState,
    empty_dense_relu_optimizer_state,
    zero_adam_state,
)

pytestmark = pytest.mark.unit


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config(**overrides: object) -> SelfNormalizedResetConfig:
    values: dict[str, object] = {
        "input_dim": 2,
        "unit_count": 3,
        "output_dim": 2,
        "source_sha256": _digest("source-a"),
        "representation_sha256": _digest("representation-a"),
        "window_size": 4,
        "min_intervals": 1,
        "warmup_observations": 0,
        "rejection_percentile": 0.25,
        "optimizer_kind": "none",
        "initialization_mode": "caller_provided",
    }
    values.update(overrides)
    return SelfNormalizedResetConfig(**values)  # type: ignore[arg-type]


def _parameters(config: SelfNormalizedResetConfig, offset: float = 0.0) -> DenseReLUParameters:
    incoming = jnp.arange(config.input_dim * config.unit_count, dtype=jnp.float32).reshape(
        config.input_dim, config.unit_count
    )
    outgoing = jnp.arange(config.unit_count * config.output_dim, dtype=jnp.float32).reshape(
        config.unit_count, config.output_dim
    )
    return DenseReLUParameters(
        incoming_weight=incoming + jnp.asarray(offset, dtype=jnp.float32),
        bias=jnp.arange(config.unit_count, dtype=jnp.float32) + offset,
        outgoing_weight=outgoing + jnp.asarray(offset + 20.0, dtype=jnp.float32),
    )


def _filled_adam(parameters: DenseReLUParameters, value: float = 5.0) -> DenseReLUOptimizerState:
    return cast(
        DenseReLUOptimizerState,
        zero_adam_state(parameters).replace(
            count=jnp.asarray(17, dtype=jnp.int32),
            incoming_first_moment=jnp.full_like(parameters.incoming_weight, value),
            incoming_second_moment=jnp.full_like(parameters.incoming_weight, value + 1),
            bias_first_moment=jnp.full_like(parameters.bias, value + 2),
            bias_second_moment=jnp.full_like(parameters.bias, value + 3),
            outgoing_first_moment=jnp.full_like(parameters.outgoing_weight, value + 4),
            outgoing_second_moment=jnp.full_like(parameters.outgoing_weight, value + 5),
        ),
    )


def _target(config: SelfNormalizedResetConfig, offset: float = 0.0) -> DenseReLUResetTarget:
    parameters = _parameters(config, offset)
    optimizer = (
        _filled_adam(parameters)
        if config.optimizer_kind == "adam"
        else empty_dense_relu_optimizer_state()
    )
    return DenseReLUResetTarget(parameters=parameters, optimizer=optimizer)


def _fresh(config: SelfNormalizedResetConfig, value: float = 77.0) -> DenseReLUFreshParameters:
    return DenseReLUFreshParameters(
        incoming_weight=jnp.full((config.input_dim, config.unit_count), value, dtype=jnp.float32),
        bias=jnp.full((config.unit_count,), value + 1, dtype=jnp.float32),
    )


def _array_bits(value: jax.Array) -> np.ndarray:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return np.asarray(jr.key_data(value))
    if value.dtype == jnp.float32:
        return np.asarray(jax.lax.bitcast_convert_type(value, jnp.uint32))
    return np.asarray(value)


def _assert_tree_bits_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(_array_bits(lhs), _array_bits(rhs))


def _step(
    controller: SelfNormalizedResets,
    state: SelfNormalizedResetState,
    activation: list[float],
    *,
    target: DenseReLUResetTarget | None = None,
    fresh: DenseReLUFreshParameters | None = None,
) -> SelfNormalizedResetResult:
    resolved_state = state
    resolved_target = target if target is not None else resolved_state.target
    return controller.step(
        resolved_state,
        jnp.asarray(activation, dtype=jnp.float32),
        resolved_target,
        fresh if fresh is not None else _fresh(controller.config),
    )


def test_config_serializes_paper_law_support_ordering_and_bounds() -> None:
    config = _config()
    payload = config.to_config()

    assert payload["schema"] == SELF_NORMALIZED_RESETS_SCHEMA
    assert payload["algorithm"] == "self-normalized-resets"
    assert payload["paper"] == "Farias-Jozefiak-ICLR-2025-arXiv-2410.20098"
    assert payload["geometric_support"] == "positive-integers"
    assert payload["completed_interval"] == "pre-firing-age-plus-one"
    assert payload["reset_ordering"] == "observe-then-caller-optimize-then-reset"
    assert payload["support_choice_authority"] == ("alberta-explicit-positive-support-resolution")
    assert payload["official_histogram_code_bit_equivalent"] is False
    assert payload["integrity_authenticated"] is False
    assert payload["scientific_evidence_claimed"] is False
    assert SelfNormalizedResetConfig.from_config(payload) == config

    malformed = dict(payload)
    malformed["geometric_support"] = "zero-based"
    with pytest.raises(ValueError, match="geometric_support"):
        SelfNormalizedResetConfig.from_config(malformed)
    with pytest.raises(ValueError):
        _config(rejection_percentile=0.0)
    with pytest.raises(ValueError):
        _config(min_intervals=5)
    with pytest.raises(ValueError):
        _config(window_size=0)
    with pytest.raises(ValueError):
        _config(maximum_updates=2**64)


def test_hand_calculation_uses_positive_geometric_support_and_inclusive_eta() -> None:
    config = _config(unit_count=1, window_size=2, rejection_percentile=0.25)
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(1), _target(config))
    fresh = _fresh(config)

    # Fires at observations 1 and 3 complete A=2: one silent observation plus
    # the firing endpoint. The estimated law is Geometric(p=1/2).
    for activation in ([1.0], [0.0], [1.0]):
        result = _step(controller, state, activation, fresh=fresh)
        state = result.state
    np.testing.assert_array_equal(state.intervals_words[0, 0], [0, 2])

    first_silent = _step(controller, state, [0.0], fresh=fresh)
    assert not bool(first_silent.reset_mask[0])
    assert float(first_silent.log_survival[0]) == pytest.approx(np.log(0.5))
    second_silent = _step(controller, first_silent.state, [0.0], fresh=fresh)
    assert bool(second_silent.reset_mask[0])
    # Exact equality exercises the specified inclusive ``survival <= eta`` gate.
    assert float(jnp.exp(second_silent.log_survival[0])) == pytest.approx(0.25)


def test_min_history_and_post_observation_warmup_boundaries_are_exact() -> None:
    config = _config(
        unit_count=1,
        window_size=3,
        min_intervals=2,
        warmup_observations=6,
        rejection_percentile=0.25,
    )
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(101), _target(config))

    for activation in ([1.0], [0.0], [1.0]):
        state = _step(controller, state, activation).state
    before_min_history = _step(controller, state, [0.0])
    assert int(before_min_history.state.interval_count[0]) == 1
    assert not bool(before_min_history.eligible_mask[0])

    second_interval = _step(controller, before_min_history.state, [1.0])
    assert int(second_interval.state.interval_count[0]) == 2
    at_warmup_boundary = _step(controller, second_interval.state, [0.0])
    np.testing.assert_array_equal(at_warmup_boundary.state.epoch_observations_words, [[0, 6]])
    assert bool(at_warmup_boundary.eligible_mask[0])
    assert not bool(at_warmup_boundary.reset_mask[0])
    equality_reset = _step(controller, at_warmup_boundary.state, [0.0])
    assert bool(equality_reset.reset_mask[0])


def test_trailing_window_recurrence_and_eviction_are_exact() -> None:
    config = _config(
        unit_count=1,
        window_size=2,
        warmup_observations=100,
        rejection_percentile=1.0e-6,
    )
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(2), _target(config))

    # Completed positive-support gaps are A=1, A=3, A=2. The third evicts A=1.
    for activation in ([1.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0]):
        state = _step(controller, state, activation).state
    np.testing.assert_array_equal(state.intervals_words[0], [[0, 2], [0, 3]])
    np.testing.assert_array_equal(state.interval_count, [2])
    np.testing.assert_array_equal(state.interval_cursor, [1])

    diagnostic = _step(controller, state, [0.0])
    assert float(diagnostic.estimated_mean_interval[0]) == pytest.approx(2.5)


def test_never_always_and_mixed_units_have_distinct_self_normalized_timing() -> None:
    config = _config(unit_count=3, window_size=2, rejection_percentile=0.25)
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(3), _target(config))

    sequence = [
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    result = None
    for activation in sequence:
        result = _step(controller, state, activation)
        state = result.state
    assert result is not None
    np.testing.assert_array_equal(result.reset_mask, [False, False, True])
    np.testing.assert_array_equal(state.interval_count, [0, 2, 1])
    assert not bool(state.has_fired[0])
    assert bool(state.has_fired[1])
    assert not bool(state.has_fired[2])  # reset starts a fresh firing epoch


def test_post_optimizer_ordering_consumer_and_adam_moment_clearing() -> None:
    config = _config(
        unit_count=3,
        window_size=2,
        optimizer_kind="adam",
        rejection_percentile=0.25,
        max_resets_per_step=2,
    )
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(4), _target(config))
    fresh = _fresh(config, 101.0)
    for activation in ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]):
        state = _step(controller, state, list(activation), fresh=fresh).state

    post_optimizer = _target(config, offset=1_000.0)
    result = _step(
        controller,
        state,
        [0.0, 0.0, 0.0],
        target=post_optimizer,
        fresh=fresh,
    )
    np.testing.assert_array_equal(result.reset_mask, [True, True, False])
    parameters = result.state.target.parameters
    np.testing.assert_array_equal(parameters.incoming_weight[:, :2], 101.0)
    np.testing.assert_array_equal(parameters.bias[:2], 102.0)
    np.testing.assert_array_equal(parameters.outgoing_weight[:2], 0.0)
    # Lowest indices win the deterministic same-step cap; every unselected bit
    # comes from the post-optimizer target, not the pre-optimizer state.
    np.testing.assert_array_equal(
        _array_bits(parameters.incoming_weight[:, 2]),
        _array_bits(post_optimizer.parameters.incoming_weight[:, 2]),
    )
    np.testing.assert_array_equal(
        _array_bits(parameters.bias[2]), _array_bits(post_optimizer.parameters.bias[2])
    )
    np.testing.assert_array_equal(
        _array_bits(parameters.outgoing_weight[2]),
        _array_bits(post_optimizer.parameters.outgoing_weight[2]),
    )
    optimizer = result.state.target.optimizer
    assert int(optimizer.count) == int(post_optimizer.optimizer.count)
    np.testing.assert_array_equal(optimizer.incoming_first_moment[:, :2], 0.0)
    np.testing.assert_array_equal(optimizer.incoming_second_moment[:, :2], 0.0)
    np.testing.assert_array_equal(optimizer.bias_first_moment[:2], 0.0)
    np.testing.assert_array_equal(optimizer.bias_second_moment[:2], 0.0)
    np.testing.assert_array_equal(optimizer.outgoing_first_moment[:2], 0.0)
    np.testing.assert_array_equal(optimizer.outgoing_second_moment[:2], 0.0)
    np.testing.assert_array_equal(
        _array_bits(optimizer.incoming_first_moment[:, 2]),
        _array_bits(post_optimizer.optimizer.incoming_first_moment[:, 2]),
    )


def test_owned_rng_is_deterministic_and_no_reset_does_not_consume_it() -> None:
    config = _config(
        unit_count=1,
        window_size=2,
        initialization_mode="owned_lecun_uniform",
        initial_bias=0.125,
    )
    controller = SelfNormalizedResets(config)
    initial_target = _target(config)
    states = [controller.init(jr.key(9), initial_target) for _ in range(2)]
    initial_key_data = np.asarray(jr.key_data(states[0].rng_key))
    for activation in ([1.0], [1.0]):
        states = [
            controller.step(state, jnp.asarray(activation, jnp.float32), state.target).state
            for state in states
        ]
    np.testing.assert_array_equal(jr.key_data(states[0].rng_key), initial_key_data)

    results = [
        controller.step(state, jnp.asarray([0.0], jnp.float32), state.target) for state in states
    ]
    assert all(bool(result.reset_mask[0]) for result in results)
    _assert_tree_bits_equal(results[0].state, results[1].state)
    assert not np.array_equal(jr.key_data(results[0].state.rng_key), initial_key_data)
    assert float(results[0].state.target.parameters.bias[0]) == pytest.approx(0.125)


def test_no_reset_and_disabled_paths_preserve_target_bits() -> None:
    config = _config(unit_count=1)
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(10), _target(config))
    post_optimizer = _target(config, offset=500.0)
    result = _step(controller, state, [1.0], target=post_optimizer)
    _assert_tree_bits_equal(result.state.target, post_optimizer)
    np.testing.assert_array_equal(jr.key_data(result.state.rng_key), jr.key_data(state.rng_key))

    disabled_config = replace(config, enabled=False, initialization_mode="owned_lecun_uniform")
    disabled = SelfNormalizedResets(disabled_config)
    disabled_state = disabled.init(jr.key(11), _target(disabled_config))
    disabled_result = disabled.step(
        disabled_state,
        jnp.asarray([0.0], dtype=jnp.float32),
        _target(disabled_config, offset=900.0),
    )
    assert bool(disabled_result.update_applied)
    np.testing.assert_array_equal(disabled_result.reset_mask, [False])
    np.testing.assert_array_equal(disabled_result.state.step_words, [0, 0])
    np.testing.assert_array_equal(disabled_result.state.interval_count, [0])
    _assert_tree_bits_equal(disabled_result.state.target, _target(disabled_config, offset=900.0))


def test_shape_nonfinite_negative_and_live_state_tamper_fail_closed() -> None:
    config = _config(unit_count=1)
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(12), _target(config))
    fresh = _fresh(config)
    with pytest.raises(ValueError, match="shape"):
        controller.step(state, jnp.zeros((2,), jnp.float32), state.target, fresh)
    with pytest.raises(TypeError, match="dtype"):
        controller.step(state, jnp.zeros((1,), jnp.int32), state.target, fresh)
    unsafe_optimizer = state.target.optimizer.replace(
        incoming_first_moment=jnp.zeros((1,), dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="shape"):
        controller.init(jr.key(120), state.target.replace(optimizer=unsafe_optimizer))

    for bad_activation in (jnp.asarray([jnp.nan], jnp.float32), jnp.asarray([-1.0], jnp.float32)):
        result = controller.step(state, bad_activation, state.target, fresh)
        assert bool(result.update_rejected)
        assert not bool(result.activation_valid)
        _assert_tree_bits_equal(result.state, state)

    corrupt = state.replace(ages_words=state.ages_words.at[0, 1].set(1))
    result = controller.step(corrupt, jnp.asarray([0.0], jnp.float32), corrupt.target, fresh)
    assert not bool(result.state_valid)
    assert bool(result.update_rejected)
    _assert_tree_bits_equal(result.state, corrupt)

    bad_target = state.target.replace(
        parameters=state.target.parameters.replace(
            bias=state.target.parameters.bias.at[0].set(jnp.nan)
        )
    )
    result = controller.step(state, jnp.asarray([0.0], jnp.float32), bad_target, fresh)
    assert not bool(result.target_valid)
    _assert_tree_bits_equal(result.state, state)

    bad_fresh = fresh.replace(bias=fresh.bias.at[0].set(jnp.inf))
    result = controller.step(state, jnp.asarray([0.0], jnp.float32), state.target, bad_fresh)
    assert not bool(result.fresh_values_valid)
    _assert_tree_bits_equal(result.state, state)

    # Structural impossibility remains invalid even if an untrusted writer can
    # recompute the explicitly non-authenticating live checksum.
    impossible = state.replace(
        intervals_words=state.intervals_words.at[0, 0].set(jnp.asarray([0, 2], dtype=jnp.uint32)),
        interval_count=jnp.asarray([1], dtype=jnp.int32),
        interval_cursor=jnp.asarray([1], dtype=jnp.int32),
        integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
    )
    impossible = impossible.replace(integrity_tag=controller._integrity_tag(impossible))
    result = controller.step(impossible, jnp.asarray([0.0], jnp.float32), impossible.target, fresh)
    assert not bool(result.state_valid)
    _assert_tree_bits_equal(result.state, impossible)

    # This impossible sum collided under the former modulo-uint32 check.
    count_config = _config(unit_count=4)
    count_controller = SelfNormalizedResets(count_config)
    count_state = count_controller.init(jr.key(121), _target(count_config))
    modulo_corrupt = count_state.replace(
        unit_reset_count=jnp.asarray([2**31 - 1, 2**31 - 1, 2**31 - 1, 2], dtype=jnp.int32),
        total_reset_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
    )
    modulo_corrupt = modulo_corrupt.replace(
        integrity_tag=count_controller._integrity_tag(modulo_corrupt)
    )
    assert not bool(count_controller._state_valid(modulo_corrupt))


def test_exact_clock_carry_update_cap_and_resource_declaration() -> None:
    carry_config = _config(unit_count=1, maximum_updates=2**32 + 1)
    controller = SelfNormalizedResets(carry_config)
    state = controller.init(jr.key(13), _target(carry_config))
    near_carry = state.replace(
        step_words=jnp.asarray([0, 2**32 - 1], dtype=jnp.uint32),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
    )
    near_carry = near_carry.replace(integrity_tag=controller._integrity_tag(near_carry))
    result = _step(controller, near_carry, [1.0])
    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.state.step_words, [1, 0])
    assert int(result.state.step_count) == 2**31 - 1

    # Per-unit inactivity and epoch counters use the same exact word carry.
    age_carry = state.replace(
        ages_words=jnp.asarray([[0, 2**32 - 1]], dtype=jnp.uint32),
        epoch_observations_words=jnp.asarray([[0, 2**32 - 1]], dtype=jnp.uint32),
        has_fired=jnp.asarray([True], dtype=jnp.bool_),
        step_words=jnp.asarray([1, 0], dtype=jnp.uint32),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
    )
    age_carry = age_carry.replace(integrity_tag=controller._integrity_tag(age_carry))
    age_result = _step(controller, age_carry, [0.0])
    assert bool(age_result.update_applied)
    np.testing.assert_array_equal(age_result.state.ages_words, [[1, 0]])
    np.testing.assert_array_equal(age_result.state.epoch_observations_words, [[1, 0]])

    capped_config = _config(unit_count=1, maximum_updates=1)
    capped = SelfNormalizedResets(capped_config)
    capped_state = capped.init(jr.key(14), _target(capped_config))
    first = _step(capped, capped_state, [1.0])
    second = _step(capped, first.state, [1.0])
    assert bool(first.update_applied)
    assert not bool(second.lifetime_capacity_available)
    assert bool(second.update_rejected)
    _assert_tree_bits_equal(second.state, first.state)

    resources = controller.resource_declaration(state)
    assert resources.interval_slots == carry_config.unit_count * carry_config.window_size
    assert resources.dense_parameter_count == carry_config.parameter_count
    assert resources.persistent_state_bytes > 0
    assert resources.maximum_updates == 2**32 + 1
    assert resources.temporary_bytes_scope == (
        "source-level-named-arrays; excludes-compiler-and-xla-workspaces; "
        "not-a-measured-device-peak"
    )
    assert resources.checkpoint_host_only
    assert not resources.integrity_authenticated
    assert not resources.external_side_effects


def test_large_mean_log1p_survival_remains_finite_negative_and_resets() -> None:
    config = _config(
        unit_count=1,
        window_size=1,
        rejection_percentile=0.5,
    )
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(140), _target(config))
    large_mean = 2**30
    large_age_before_update = 2**31 - 1
    state = state.replace(
        ages_words=jnp.asarray([[0, large_age_before_update]], dtype=jnp.uint32),
        epoch_observations_words=jnp.asarray([[0, large_age_before_update]], dtype=jnp.uint32),
        has_fired=jnp.asarray([True], dtype=jnp.bool_),
        intervals_words=jnp.asarray([[[0, large_mean]]], dtype=jnp.uint32),
        interval_count=jnp.asarray([1], dtype=jnp.int32),
        interval_cursor=jnp.asarray([0], dtype=jnp.int32),
        step_words=jnp.asarray([0, 2**31], dtype=jnp.uint32),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
    )
    state = state.replace(integrity_tag=controller._integrity_tag(state))

    result = _step(controller, state, [0.0])

    # p=2^-30 and age becomes 2^31, so log survival is approximately -2.
    # Computing log(float32(1 - p)) would instead round to log(1)=0.
    assert np.isfinite(float(result.log_survival[0]))
    assert float(result.log_survival[0]) == pytest.approx(-2.0, rel=1.0e-5)
    assert bool(result.reset_mask[0])


def test_checkpoint_resume_tamper_detection_and_explicit_rebind() -> None:
    config = _config(unit_count=1, window_size=2)
    controller = SelfNormalizedResets(config)
    state = controller.init(jr.key(15), _target(config))
    for activation in ([1.0], [0.0], [1.0], [0.0]):
        state = _step(controller, state, activation).state
    payload = controller.checkpoint_payload(state)
    assert payload["integrity_notice"] == ("unkeyed-sha256-detects-corruption-not-authentication")
    restored, restored_state = SelfNormalizedResets.from_checkpoint_payload(
        payload,
        expected_source_sha256=config.source_sha256,
        expected_representation_sha256=config.representation_sha256,
    )
    _assert_tree_bits_equal(state, restored_state)
    resumed_left = _step(controller, state, [0.0])
    resumed_right = _step(restored, restored_state, [0.0])
    _assert_tree_bits_equal(resumed_left, resumed_right)

    tampered = copy.deepcopy(payload)
    tampered_state = tampered["state"]
    assert isinstance(tampered_state, dict)
    step_payload = tampered_state["step_words"]
    assert isinstance(step_payload, dict)
    step_hex = str(step_payload["data_hex"])
    replacement = "0" if step_hex[-1] != "0" else "1"
    step_payload["data_hex"] = step_hex[:-1] + replacement
    with pytest.raises(ValueError, match="integrity check"):
        SelfNormalizedResets.from_checkpoint_payload(
            tampered,
            expected_source_sha256=config.source_sha256,
            expected_representation_sha256=config.representation_sha256,
        )
    with pytest.raises(ValueError, match="source binding"):
        SelfNormalizedResets.from_checkpoint_payload(
            payload,
            expected_source_sha256=_digest("wrong"),
            expected_representation_sha256=config.representation_sha256,
        )

    rebound, rebound_state = controller.rebind_reset(
        state,
        rng_key=jr.key(99),
        source_sha256=_digest("source-b"),
        representation_sha256=_digest("representation-b"),
    )
    _assert_tree_bits_equal(rebound_state.target, state.target)
    np.testing.assert_array_equal(rebound_state.step_words, [0, 0])
    np.testing.assert_array_equal(rebound_state.interval_count, [0])
    assert rebound.config.source_sha256 == _digest("source-b")
    assert rebound.config.representation_sha256 == _digest("representation-b")
