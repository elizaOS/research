"""Exact-horizon and atomicity tests for the partial-observation wrapper."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.partial_observation import (
    PARTIAL_OBSERVATION_CONFIG_SCHEMA,
    PARTIAL_OBSERVATION_RESOURCE_SCHEMA,
    PARTIAL_OBSERVATION_STATE_SCHEMA,
    MaskMode,
    PartialObservationResourceBudget,
    PartialObservationState,
    PartialObservationWrapper,
    migrate_legacy_partial_observation_state,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


@chex.dataclass(frozen=True)
class _ToyState:
    count: jax.Array
    payload: jax.Array


class _ToyStream:
    """Small deterministic child with controllable dynamic failures."""

    feature_dim = 3

    def __init__(
        self,
        *,
        invalid_output: bool = False,
        invalid_candidate: bool = False,
        wrong_observation_shape: bool = False,
        wrong_candidate_shape: bool = False,
    ) -> None:
        self.invalid_output = invalid_output
        self.invalid_candidate = invalid_candidate
        self.wrong_observation_shape = wrong_observation_shape
        self.wrong_candidate_shape = wrong_candidate_shape

    def init(self, key: jax.Array) -> _ToyState:
        del key
        return _ToyState(
            count=jnp.asarray(0, dtype=jnp.int32),
            payload=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        )

    def step(self, state: _ToyState, idx: jax.Array) -> tuple[TimeStep, _ToyState]:
        del idx
        observation = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
        if self.wrong_observation_shape:
            observation = observation[:2]
        if self.invalid_output:
            observation = observation.at[0].set(jnp.nan)
        payload = state.payload + 1.0
        if self.invalid_candidate:
            payload = payload.at[0].set(jnp.nan)
        if self.wrong_candidate_shape:
            payload = jnp.concatenate((payload, jnp.ones((1,), dtype=payload.dtype)))
        return (
            TimeStep(
                observation=observation,
                target=jnp.asarray([4.0], dtype=jnp.float32),
            ),
            _ToyState(count=state.count + 1, payload=payload),
        )


class _SemanticValidatorStream(_ToyStream):
    def state_is_valid(self, state: _ToyState) -> jax.Array:
        return jnp.all(state.payload >= 0.0)


def _wrapper(
    mode: MaskMode,
    *,
    inner: Any | None = None,
) -> PartialObservationWrapper[Any]:
    child = _ToyStream() if inner is None else inner
    if mode is MaskMode.FIXED:
        return PartialObservationWrapper(
            child,
            mode=mode,
            fixed_mask=jnp.asarray([True, False, True], dtype=jnp.bool_),
            sentinel=-9.0,
        )
    if mode is MaskMode.RANDOM:
        return PartialObservationWrapper(child, mode=mode, mask_prob=0.5, sentinel=-9.0)
    return PartialObservationWrapper(
        child,
        mode=mode,
        schedule=(
            jnp.asarray([True, False, False], dtype=jnp.bool_),
            jnp.asarray([False, True, False], dtype=jnp.bool_),
            jnp.asarray([False, False, True], dtype=jnp.bool_),
        ),
        sentinel=-9.0,
    )


@pytest.mark.parametrize("mode", list(MaskMode))
def test_all_modes_own_exact_identity_and_saturating_telemetry(mode: MaskMode) -> None:
    stream = _wrapper(mode)
    state = stream.init(jr.key(0))
    assert state.step_count.dtype == jnp.int32
    assert state.step_words.dtype == jnp.uint32
    chex.assert_trees_all_equal(state.step_count, jnp.asarray(0, dtype=jnp.int32))
    chex.assert_trees_all_equal(state.step_words, jnp.zeros((2,), dtype=jnp.uint32))

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.state.step_count, jnp.asarray(1, dtype=jnp.int32))
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )

    high_state = dataclasses.replace(
        result.state,
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((1, 17), dtype=jnp.uint32),
    )
    high_result = stream.step_result(high_state, jnp.asarray(1, dtype=jnp.int32))
    assert bool(high_result.update_applied)
    chex.assert_trees_all_equal(
        high_result.state.step_count,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        high_result.state.step_words,
        jnp.asarray((1, 18), dtype=jnp.uint32),
    )


def test_periodic_schedule_is_exact_beyond_two_to_the_32() -> None:
    stream = _wrapper(MaskMode.PERIODIC)
    state = stream.init(jr.key(4))
    # 2**32 + 4 has remainder 2 modulo the three-element schedule.
    state = dataclasses.replace(
        state,
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((1, 4), dtype=jnp.uint32),
    )
    result = jax.jit(stream.step_result)(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(result.schedule_index) == 2
    chex.assert_trees_all_equal(
        result.visibility_mask,
        jnp.asarray([False, False, True], dtype=jnp.bool_),
    )
    chex.assert_trees_all_close(
        result.timestep.observation,
        jnp.asarray([-9.0, -9.0, 3.0], dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    "event",
    [
        0,
        1,
        2**31 - 1,
        2**31,
        2**32 - 1,
        2**32,
        2**32 + 1,
        2**48 + 12345,
        2**63 + 7,
        2**64 - 2,
    ],
)
def test_periodic_schedule_matches_host_integer_arithmetic_at_boundaries(
    event: int,
) -> None:
    stream = _wrapper(MaskMode.PERIODIC)
    state = stream.init(jr.key(8))
    telemetry = min(event, _INT32_MAX)
    state = dataclasses.replace(
        state,
        step_count=jnp.asarray(telemetry, dtype=jnp.int32),
        step_words=jnp.asarray(
            ((event >> 32) & _UINT32_MAX, event & _UINT32_MAX),
            dtype=jnp.uint32,
        ),
    )
    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(result.update_applied)
    assert int(result.schedule_index) == event % 3
    next_event = event + 1
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray(
            ((next_event >> 32) & _UINT32_MAX, next_event & _UINT32_MAX),
            dtype=jnp.uint32,
        ),
    )


@pytest.mark.parametrize("mode", list(MaskMode))
def test_invalid_child_output_rolls_back_entire_wrapper(mode: MaskMode) -> None:
    stream = _wrapper(mode, inner=_ToyStream(invalid_output=True))
    state = stream.init(jr.key(7))
    result = jax.jit(stream.step_result)(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(result.update_rejected)
    assert not bool(result.output_valid)
    chex.assert_trees_all_equal(result.state, state)
    assert bool(jnp.all(jnp.isnan(result.timestep.observation)))
    assert bool(jnp.all(jnp.isnan(result.timestep.target)))


def test_invalid_child_candidate_rolls_back_entire_wrapper() -> None:
    stream = _wrapper(MaskMode.RANDOM, inner=_ToyStream(invalid_candidate=True))
    state = stream.init(jr.key(9))
    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(result.update_rejected)
    assert not bool(result.candidate_state_valid)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("mode", list(MaskMode))
def test_exhausted_wrapper_identity_rolls_back_child_rng_and_counters(
    mode: MaskMode,
) -> None:
    stream = _wrapper(mode)
    state = stream.init(jr.key(2))
    state = dataclasses.replace(
        state,
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
    )
    result = jax.jit(stream.step_result)(state, jnp.asarray(11, dtype=jnp.int32))
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)


def test_invalid_input_and_tampered_telemetry_fail_closed() -> None:
    stream = _wrapper(MaskMode.RANDOM)
    state = stream.init(jr.key(2))
    invalid_input = stream.step_result(state, jnp.asarray(jnp.nan, dtype=jnp.float32))
    assert not bool(invalid_input.input_valid)
    chex.assert_trees_all_equal(invalid_input.state, state)

    tampered = dataclasses.replace(
        state,
        step_count=jnp.asarray(8, dtype=jnp.int32),
    )
    invalid_state = stream.step_result(tampered, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(invalid_state.lifetime_counter_valid)
    chex.assert_trees_all_equal(invalid_state.state, tampered)


def test_optional_child_validator_is_honored_without_claiming_generic_semantics() -> None:
    stream = _wrapper(MaskMode.FIXED, inner=_SemanticValidatorStream())
    state = stream.init(jr.key(0))
    state = dataclasses.replace(
        state,
        inner_state=dataclasses.replace(
            state.inner_state,
            payload=jnp.asarray([-1.0, 2.0], dtype=jnp.float32),
        ),
    )
    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(result.state_valid)
    chex.assert_trees_all_equal(result.state, state)


def test_static_child_output_contract_rejects_wrong_shape() -> None:
    stream = _wrapper(MaskMode.FIXED, inner=_ToyStream(wrong_observation_shape=True))
    state = stream.init(jr.key(0))
    with pytest.raises(ValueError, match="observation shape"):
        stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))


def test_static_child_state_contract_rejects_shape_change_before_commit() -> None:
    stream = _wrapper(MaskMode.RANDOM, inner=_ToyStream(wrong_candidate_shape=True))
    state = stream.init(jr.key(0))
    with pytest.raises(ValueError, match="child state leaf 1 shape changed"):
        stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))


@pytest.mark.parametrize("mode", list(MaskMode))
def test_step_compatibility_jit_and_scan(mode: MaskMode) -> None:
    stream = _wrapper(mode)
    state = stream.init(jr.key(3))
    timestep, stepped = jax.jit(stream.step)(state, jnp.asarray(0, dtype=jnp.int32))
    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    chex.assert_trees_all_equal(timestep, result.timestep)
    chex.assert_trees_all_equal(stepped, result.state)

    def body(
        carry: PartialObservationState[_ToyState],
        idx: jax.Array,
    ) -> tuple[PartialObservationState[_ToyState], jax.Array]:
        item, next_state = stream.step(carry, idx)
        return next_state, item.observation

    final, observations = jax.lax.scan(body, state, jnp.arange(8, dtype=jnp.int32))
    chex.assert_shape(observations, (8, 3))
    chex.assert_trees_all_equal(final.step_words, jnp.asarray((0, 8), dtype=jnp.uint32))
    chex.assert_trees_all_equal(final.step_count, jnp.asarray(8, dtype=jnp.int32))


def test_config_round_trip_is_strict_and_child_is_explicitly_external() -> None:
    child = _ToyStream()
    stream = _wrapper(MaskMode.PERIODIC, inner=child)
    payload = stream.to_config()
    assert payload["config_schema"] == PARTIAL_OBSERVATION_CONFIG_SCHEMA
    assert payload["state_schema"] == PARTIAL_OBSERVATION_STATE_SCHEMA
    assert payload["resource_schema"] == PARTIAL_OBSERVATION_RESOURCE_SCHEMA
    rebuilt = PartialObservationWrapper.from_config(payload, inner=child)
    assert rebuilt.to_config() == payload

    with pytest.raises(ValueError, match="fields"):
        PartialObservationWrapper.from_config({**payload, "extra": 1}, inner=child)
    with pytest.raises(ValueError, match="config schema"):
        PartialObservationWrapper.from_config(
            {**payload, "config_schema": "wrong"},
            inner=child,
        )
    with pytest.raises(ValueError, match="inner stream type"):
        PartialObservationWrapper.from_config(payload, inner=_SemanticValidatorStream())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mode": "fixed", "fixed_mask": jnp.ones(3, dtype=jnp.bool_)}, "MaskMode"),
        ({"mode": MaskMode.FIXED, "fixed_mask": jnp.ones(3, dtype=jnp.int32)}, "dtype"),
        ({"mode": MaskMode.RANDOM, "fixed_mask": jnp.ones(3, dtype=jnp.bool_)}, "fixed_mask"),
        (
            {
                "mode": MaskMode.FIXED,
                "fixed_mask": jnp.ones(3, dtype=jnp.bool_),
                "schedule": (jnp.ones(3, dtype=jnp.bool_),),
            },
            "schedule",
        ),
        ({"mode": MaskMode.RANDOM, "mask_prob": True}, "mask_prob"),
        ({"mode": MaskMode.RANDOM, "mask_prob": float("nan")}, "mask_prob"),
        (
            {
                "mode": MaskMode.PERIODIC,
                "schedule": [jnp.ones(3, dtype=jnp.bool_)],
            },
            "tuple",
        ),
        (
            {
                "mode": MaskMode.PERIODIC,
                "schedule": (jnp.ones(3, dtype=jnp.int32),),
            },
            "dtype",
        ),
        (
            {
                "mode": MaskMode.FIXED,
                "fixed_mask": jnp.ones(3, dtype=jnp.bool_),
                "sentinel": float("inf"),
            },
            "sentinel",
        ),
    ],
)
def test_constructor_rejects_ambiguous_or_lossy_configuration(
    kwargs: dict[str, Any],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        PartialObservationWrapper(_ToyStream(), **kwargs)


def test_legacy_periodic_migration_only_accepts_representable_identity() -> None:
    stream = _wrapper(MaskMode.PERIODIC)
    initialized = stream.init(jr.key(1))
    legacy = {
        "inner_state": initialized.inner_state,
        "key": initialized.key,
        "period_index": jnp.asarray(17, dtype=jnp.int32),
    }
    migrated = migrate_legacy_partial_observation_state(legacy, wrapper=stream)
    chex.assert_trees_all_equal(migrated.step_count, jnp.asarray(17, dtype=jnp.int32))
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 17), dtype=jnp.uint32),
    )

    for mode in (MaskMode.FIXED, MaskMode.RANDOM):
        with pytest.raises(ValueError, match="not representable"):
            migrate_legacy_partial_observation_state(legacy, wrapper=_wrapper(mode))
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_partial_observation_state(
            {**legacy, "period_index": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
            wrapper=stream,
        )


@pytest.mark.parametrize("mode", list(MaskMode))
def test_resource_budget_separates_child_from_wrapper_ownership(mode: MaskMode) -> None:
    stream = _wrapper(mode)
    budget = stream.resource_budget
    assert budget.schema == PARTIAL_OBSERVATION_RESOURCE_SCHEMA
    assert budget.wrapper_state_nbytes == 20  # key data (8) + int32 (4) + words (8)
    assert budget.exact_clock_nbytes == 12
    assert budget.exact_clock_delta_nbytes == 8
    expected_mask_bytes = {
        MaskMode.FIXED: 3,
        MaskMode.RANDOM: 0,
        MaskMode.PERIODIC: 9,
    }[mode]
    assert budget.mask_metadata_nbytes == expected_mask_bytes
    assert budget.child_state_nbytes == 12
    assert budget.wrapper_owned_nbytes == 20 + expected_mask_bytes
    assert budget.composed_persistent_nbytes == 32 + expected_mask_bytes
    assert budget.child_state_accounting == "declared-separately-excluded-from-wrapper-owned"
    assert PartialObservationResourceBudget.from_dict(budget.to_dict()) == budget

    with pytest.raises(ValueError, match="wrapper-owned"):
        PartialObservationResourceBudget.from_dict(
            {**budget.to_dict(), "wrapper_owned_nbytes": budget.wrapper_owned_nbytes + 1}
        )
