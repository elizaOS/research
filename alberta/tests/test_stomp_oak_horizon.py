# mypy: disable-error-code="attr-defined"
"""Exact lifetime and atomic transaction contracts for STOMP and OaK."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import (
    OAK_LIFETIME_COUNTER_DELTA_NBYTES,
    OAK_LIFETIME_COUNTER_NBYTES,
    OaKAgent,
    OaKConfig,
    OaKState,
    measure_oak_state_nbytes,
    measure_oak_wrapper_state_nbytes,
    migrate_legacy_oak_state,
    oak_lifetime_counter_nbytes,
    oak_total_lifetime_counter_nbytes,
)
from alberta_framework.core.options import (
    STOMP_LIFETIME_COUNTER_DELTA_NBYTES,
    STOMP_LIFETIME_COUNTER_NBYTES,
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
    load_stomp_state_with_migration,
    measure_stomp_state_nbytes,
    measure_stomp_wrapper_state_nbytes,
    stomp_lifetime_counter_nbytes,
    stomp_state_to_checkpoint_payload,
)

_I32_MAX = 2**31 - 1
_U32_MAX = 2**32 - 1
_OBS = jnp.asarray((1.0, 0.0), dtype=jnp.float32)


def _config(*, n_options: int = 1, planning: int = 0) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=tuple(
            SubtaskSpec(
                feature_index=index,
                threshold=1.0e6,
                max_option_steps=32,
            )
            for index in range(n_options)
        ),
        observation_dim=max(2, n_options),
        n_primitive_actions=2,
        base_step_size=0.05,
        option_model_decay=0.0,
        option_planning_backups_per_step=planning,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _idle_state(agent: STOMPAgent) -> STOMPState:
    state = agent.init(jr.key(1))
    return cast(
        STOMPState,
        state.replace(
            base_last_obs=jnp.zeros(agent.config.observation_dim, dtype=jnp.float32),
            base_last_action=jnp.asarray(0, dtype=jnp.int32),
            last_primitive_action=jnp.asarray(0, dtype=jnp.int32),
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
        ),
    )


def _active_state(agent: STOMPAgent, *, option: int = 0) -> STOMPState:
    state = _idle_state(agent)
    return cast(
        STOMPState,
        state.replace(
            base_last_action=jnp.asarray(
                agent.config.n_primitive_actions + option,
                dtype=jnp.int32,
            ),
            executing_option=jnp.asarray(option, dtype=jnp.int32),
            option_start_obs=jnp.zeros(agent.config.observation_dim, dtype=jnp.float32),
            option_last_intra_action=jnp.asarray(0, dtype=jnp.int32),
            option_steps=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _with_stomp_clock(state: STOMPState, high: int, low: int) -> STOMPState:
    words = jnp.asarray((high, low), dtype=jnp.uint32)
    telemetry = jnp.asarray(min((high << 32) + low, _I32_MAX), dtype=jnp.int32)
    return cast(STOMPState, state.replace(step_words=words, step_count=telemetry))


def _with_base_clock(state: STOMPState, high: int, low: int) -> STOMPState:
    words = jnp.asarray((high, low), dtype=jnp.uint32)
    telemetry = jnp.asarray(min((high << 32) + low, _I32_MAX), dtype=jnp.int32)
    return cast(
        STOMPState,
        state.replace(
            base_learner_state=state.base_learner_state.replace(
                step_words=words,
                step_count=telemetry,
            )
        ),
    )


def _without_host_timing(state: STOMPState) -> STOMPState:
    """Normalize legacy host-only MultiHead timing metadata for comparisons."""

    learner = state.base_learner_state.replace(
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    return cast(STOMPState, state.replace(base_learner_state=learner))


def _assert_stomp_persistent_equal(actual: STOMPState, expected: STOMPState) -> None:
    chex.assert_trees_all_equal(
        _without_host_timing(actual),
        _without_host_timing(expected),
    )


def _assert_oak_persistent_equal(actual: OaKState, expected: OaKState) -> None:
    _assert_stomp_persistent_equal(actual.stomp_state, expected.stomp_state)
    chex.assert_trees_all_equal(
        actual.replace(stomp_state=expected.stomp_state),
        expected,
    )


def test_stomp_initializes_exact_clock_and_accounts_for_its_bytes() -> None:
    agent = STOMPAgent(_config())
    state = agent.init(jr.key(2))

    np.testing.assert_array_equal(np.asarray(state.step_words), np.zeros(2, np.uint32))
    assert state.step_words.dtype == jnp.uint32
    assert STOMP_LIFETIME_COUNTER_NBYTES == 12
    assert STOMP_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert int(state.step_words.nbytes) == STOMP_LIFETIME_COUNTER_DELTA_NBYTES
    assert measure_stomp_state_nbytes(state) > int(state.step_words.nbytes)
    assert stomp_lifetime_counter_nbytes() == 24
    assert measure_stomp_wrapper_state_nbytes(state) < measure_stomp_state_nbytes(state)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((0, _I32_MAX), (0, _I32_MAX + 1)),
        ((7, _U32_MAX), (8, 0)),
    ],
)
def test_stomp_exact_clock_advances_after_telemetry_saturates(
    before: tuple[int, int],
    after: tuple[int, int],
) -> None:
    agent = STOMPAgent(_config())
    state = _with_stomp_clock(_idle_state(agent), *before)

    result = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    assert bool(result.update_applied)
    assert bool(result.proposed_state_valid)
    assert int(result.state.step_count) == _I32_MAX
    np.testing.assert_array_equal(np.asarray(result.pre_step_words), before)
    np.testing.assert_array_equal(np.asarray(result.post_step_words), after)
    np.testing.assert_array_equal(np.asarray(result.state.step_words), after)


def test_stomp_exhaustion_is_atomic_under_update_and_scan() -> None:
    agent = STOMPAgent(_config())
    state = _with_stomp_clock(_idle_state(agent), _U32_MAX, _U32_MAX)

    with jax.disable_jit():
        eager = agent.update(state, jnp.asarray(1.0, jnp.float32), _OBS)
    scanned = agent.scan(
        state,
        jnp.asarray((1.0,), dtype=jnp.float32),
        _OBS[None, :],
    )

    _assert_stomp_persistent_equal(eager.state, state)
    _assert_stomp_persistent_equal(scanned.state, state)
    assert not bool(eager.lifetime_capacity_available)
    assert not bool(eager.update_applied)
    assert not bool(scanned.update_applied[0])
    assert int(eager.planning_backups) == 0
    assert np.isfinite(float(eager.td_error))


def test_stomp_rejects_an_unauthenticated_outer_clock_atomically() -> None:
    agent = STOMPAgent(_config())
    state = _idle_state(agent).replace(
        step_words=jnp.asarray((0, 5), dtype=jnp.uint32),
        step_count=jnp.asarray(4, dtype=jnp.int32),
    )

    result = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    _assert_stomp_persistent_equal(result.state, state)
    assert not bool(result.lifetime_counter_valid)
    assert not bool(result.update_applied)


def test_stomp_nonfinite_candidate_rolls_back_under_eager_and_scan() -> None:
    agent = STOMPAgent(_config())
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    source = _idle_state(agent).replace(base_last_obs=jnp.full((2,), maximum, dtype=jnp.float32))

    with jax.disable_jit():
        eager = agent.update(source, maximum, _OBS)
    scanned = agent.scan(source, maximum[None], _OBS[None, :])

    _assert_stomp_persistent_equal(eager.state, source)
    _assert_stomp_persistent_equal(scanned.state, source)
    assert bool(eager.inputs_valid)
    assert not bool(eager.proposed_state_valid)
    assert not bool(eager.update_applied)
    assert not bool(scanned.proposed_state_valid[0])
    assert not bool(scanned.update_applied[0])


def test_stomp_preflights_all_nested_planning_updates_as_one_transaction() -> None:
    agent = STOMPAgent(_config(planning=2))
    state = _active_state(agent)
    state = state.replace(
        option_models=state.option_models.replace(
            n_completions=jnp.asarray((1,), dtype=jnp.int32),
            env_return_ema=jnp.asarray((1.0,), dtype=jnp.float32),
        )
    )
    state = _with_base_clock(state, _U32_MAX, _U32_MAX - 1)

    result = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    _assert_stomp_persistent_equal(result.state, state)
    assert int(result.nested_updates_required) == 2
    assert int(result.nested_updates_applied) == 0
    assert not bool(result.nested_lifetime_capacity_available)
    assert not bool(result.update_applied)


def test_planning_round_robin_uses_exact_outer_phase_after_int32_saturation() -> None:
    agent = STOMPAgent(_config(n_options=2, planning=1))
    base = _active_state(agent)
    zero_weights = tuple(
        jnp.zeros_like(value) for value in base.base_learner_state.head_params.weights
    )
    base = base.replace(
        base_learner_state=base.base_learner_state.replace(
            head_params=base.base_learner_state.head_params.replace(weights=zero_weights)
        ),
        option_models=base.option_models.replace(
            n_completions=jnp.asarray((1, 1), dtype=jnp.int32),
            env_return_ema=jnp.asarray((1.0, 2.0), dtype=jnp.float32),
            discount_ema=jnp.zeros((2,), dtype=jnp.float32),
        ),
    )
    even = _with_stomp_clock(base, 0, _I32_MAX + 1)
    odd = _with_stomp_clock(base, 0, _I32_MAX + 2)

    even_result = agent.update(even, jnp.asarray(0.0, jnp.float32), _OBS)
    odd_result = agent.update(odd, jnp.asarray(0.0, jnp.float32), _OBS)

    option_head_0 = agent.config.n_primitive_actions
    option_head_1 = option_head_0 + 1
    assert bool(
        jnp.any(
            even_result.state.base_learner_state.head_params.weights[option_head_0]
            != zero_weights[option_head_0]
        )
    )
    assert bool(
        jnp.any(
            odd_result.state.base_learner_state.head_params.weights[option_head_1]
            != zero_weights[option_head_1]
        )
    )


def test_option_completion_telemetry_saturates_without_blocking_learning() -> None:
    config = dataclasses.replace(
        _config(),
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.5),),
    )
    agent = STOMPAgent(config)
    state = _active_state(agent).replace(
        option_models=_active_state(agent).option_models.replace(
            n_completions=jnp.asarray((_I32_MAX,), dtype=jnp.int32)
        )
    )

    result = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    assert bool(result.update_applied)
    assert int(result.state.option_models.n_completions[0]) == _I32_MAX


def test_negative_option_model_counter_rejects_the_whole_source() -> None:
    agent = STOMPAgent(_config())
    source = _idle_state(agent)
    source = source.replace(
        option_models=source.option_models.replace(
            n_completions=jnp.asarray((-1,), dtype=jnp.int32)
        )
    )

    result = agent.update(source, jnp.asarray(0.0, jnp.float32), _OBS)

    _assert_stomp_persistent_equal(result.state, source)
    assert not bool(result.update_applied)


def test_candidate_validation_accepts_both_post_update_action_owners() -> None:
    config = dataclasses.replace(
        _config(),
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=1),),
    )
    agent = STOMPAgent(config)
    state = _idle_state(agent)
    option_action = config.n_primitive_actions
    option_weights = tuple(
        jnp.asarray(
            [[10.0 if action == option_action else -10.0, 0.0]],
            dtype=jnp.float32,
        )
        for action in range(config.n_total_actions)
    )
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(
            head_params=state.base_learner_state.head_params.replace(weights=option_weights)
        )
    )

    started = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    assert bool(started.update_applied)
    assert bool(started.proposed_state_valid)
    assert int(started.state.executing_option) == 0
    assert int(started.state.base_last_action) == option_action
    assert int(started.state.option_last_intra_action) == int(started.state.last_primitive_action)

    primitive_weights = tuple(
        jnp.asarray(
            [[10.0 if action == 1 else -10.0, 0.0]],
            dtype=jnp.float32,
        )
        for action in range(config.n_total_actions)
    )
    active = started.state.replace(
        base_learner_state=started.state.base_learner_state.replace(
            head_params=started.state.base_learner_state.head_params.replace(
                weights=primitive_weights
            )
        )
    )

    terminated = agent.update(active, jnp.asarray(0.0, jnp.float32), _OBS)

    assert bool(terminated.update_applied)
    assert bool(terminated.proposed_state_valid)
    assert int(terminated.state.executing_option) == -1
    assert int(terminated.state.base_last_action) == 1
    assert int(terminated.state.last_primitive_action) == 1


def test_stomp_checkpoint_migration_authenticates_only_unambiguous_legacy_clock() -> None:
    agent = STOMPAgent(_config())
    state = _with_stomp_clock(_idle_state(agent), 0, 17)
    payload = stomp_state_to_checkpoint_payload(state)
    del payload["step_words"]

    migrated = load_stomp_state_with_migration(payload)

    np.testing.assert_array_equal(np.asarray(migrated.step_words), (0, 17))
    ambiguous = dict(payload)
    ambiguous["step_count"] = jnp.asarray(_I32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        load_stomp_state_with_migration(ambiguous)


def _oak_agent(*, min_steps: int = 0) -> OaKAgent:
    return OaKAgent(
        OaKConfig(
            stomp=_config(),
            min_steps_before_curation=min_steps,
        )
    )


def _with_oak_clock(state: OaKState, high: int, low: int) -> OaKState:
    stomp = _with_stomp_clock(state.stomp_state, high, low)
    words = jnp.asarray((high, low), dtype=jnp.uint32)
    telemetry = jnp.asarray(min((high << 32) + low, _I32_MAX), dtype=jnp.int32)
    return cast(
        OaKState,
        state.replace(
            stomp_state=stomp,
            step_words=words,
            step_count=telemetry,
        ),
    )


def test_oak_exact_clock_is_aligned_with_stomp_and_counts_saturate() -> None:
    agent = _oak_agent()
    state = _with_oak_clock(agent.init(jr.key(3)), 4, _U32_MAX)
    state = state.replace(execution_counts=jnp.asarray((_I32_MAX,), jnp.int32))

    result = agent.update(state, jnp.asarray(0.0, jnp.float32), _OBS)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.state.step_words), (5, 0))
    np.testing.assert_array_equal(
        np.asarray(result.state.stomp_state.step_words),
        (5, 0),
    )
    assert int(result.state.step_count) == _I32_MAX
    assert int(result.state.execution_counts[0]) == _I32_MAX
    assert OAK_LIFETIME_COUNTER_NBYTES == 12
    assert OAK_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert oak_lifetime_counter_nbytes() == 24
    assert oak_total_lifetime_counter_nbytes() == 36
    assert measure_oak_wrapper_state_nbytes(result.state) == 24
    assert (
        measure_oak_state_nbytes(result.state)
        == measure_stomp_state_nbytes(result.state.stomp_state) + 24
    )


def test_oak_misaligned_outer_and_stomp_clocks_reject_atomically_under_jit() -> None:
    agent = _oak_agent()
    state = agent.init(jr.key(4)).replace(
        step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        step_count=jnp.asarray(1, dtype=jnp.int32),
    )
    update = jax.jit(lambda carry: agent.update(carry, jnp.asarray(0.0, jnp.float32), _OBS))

    result = update(state)

    _assert_oak_persistent_equal(result.state, state)
    assert not bool(result.nested_counter_aligned)
    assert not bool(result.update_applied)


def test_oak_nonfinite_candidate_rolls_back_under_jit() -> None:
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    agent = OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=float(maximum),
                        pseudo_reward_scale=float(maximum),
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        )
    )
    source = agent.init(jr.key(44))
    source = source.replace(
        execution_counts=jnp.asarray((1,), dtype=jnp.int32),
        cumulative_pseudo_rewards=jnp.asarray((maximum,), dtype=jnp.float32),
        stomp_state=source.stomp_state.replace(
            base_last_action=jnp.asarray(2, dtype=jnp.int32),
            last_primitive_action=jnp.asarray(0, dtype=jnp.int32),
            executing_option=jnp.asarray(0, dtype=jnp.int32),
            option_last_intra_action=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    update = jax.jit(lambda carry: agent.update(carry, jnp.asarray(0.0, jnp.float32), _OBS))

    with jax.disable_jit():
        eager = agent.update(source, jnp.asarray(0.0, jnp.float32), _OBS)
    result = update(source)

    _assert_oak_persistent_equal(eager.state, source)
    _assert_oak_persistent_equal(result.state, source)
    assert bool(eager.nested_update_applied)
    assert not bool(eager.proposed_state_valid)
    assert not bool(eager.update_applied)
    assert bool(result.nested_update_applied)
    assert not bool(result.proposed_state_valid)
    assert not bool(result.update_applied)


def test_oak_curation_minimum_uptime_uses_exact_clock_not_saturated_telemetry() -> None:
    minimum = _I32_MAX + 9
    agent = _oak_agent(min_steps=minimum)
    base = agent.init(jr.key(5)).replace(utility_ema=jnp.asarray((0.0,), dtype=jnp.float32))
    before = _with_oak_clock(base, 0, minimum - 1)
    eligible = _with_oak_clock(base, 0, minimum)

    same_agent, same_state = agent.curate(
        before,
        jr.key(6),
        available_feature_indices=[1],
    )
    new_agent, new_state = agent.curate(
        eligible,
        jr.key(6),
        available_feature_indices=[1],
    )

    assert same_agent is agent
    assert same_state is before
    assert new_agent is not agent
    assert new_agent.config.stomp.subtask_specs[0].feature_index == 1
    np.testing.assert_array_equal(np.asarray(new_state.step_words), (0, minimum))


def test_legacy_oak_migration_requires_an_exact_aligned_nested_clock() -> None:
    agent = _oak_agent()
    state = _with_oak_clock(agent.init(jr.key(70)), 0, 17)
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))  # type: ignore[arg-type]
        if field.name != "step_words"
    }

    migrated = migrate_legacy_oak_state(legacy)

    np.testing.assert_array_equal(np.asarray(migrated.step_words), (0, 17))
    chex.assert_trees_all_equal(migrated, state)

    misaligned = dict(legacy)
    misaligned["stomp_state"] = _with_stomp_clock(
        state.stomp_state,
        0,
        18,
    )
    with pytest.raises(ValueError, match="not aligned"):
        migrate_legacy_oak_state(misaligned)


def test_legacy_oak_migration_rejects_saturation_and_mixed_manifests() -> None:
    state = _oak_agent().init(jr.key(71))
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))  # type: ignore[arg-type]
        if field.name != "step_words"
    }
    saturated = dict(legacy)
    saturated["step_count"] = jnp.asarray(_I32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_oak_state(saturated)

    mixed = dict(legacy)
    mixed["step_words"] = jnp.zeros((2,), dtype=jnp.uint32)
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_oak_state(mixed)
