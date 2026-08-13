# mypy: disable-error-code="attr-defined,call-arg"
"""Exact-lifetime boundary tests for the PrototypeAgent outer composer.

These are contract tests, not scientific evidence.  They exercise only the
outer real-transition/observation identities and their alignment with the
nested OaK/STOMP learner clocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PROTOTYPE_LIFETIME_COUNTER_DELTA_NBYTES,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    load_prototype_checkpoint,
    prototype_lifetime_counter_nbytes,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_agent import (
    _lifetime_words_modulo as _words_modulo,
)
from alberta_framework.core.prototype_agent import (
    _unambiguous_legacy_int32_counter_words as _legacy_words,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _agent(
    *,
    auto_curate_every: int = 0,
    world_model_step_size: float | None = None,
) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=2,
                    n_primitive_actions=2,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            auto_curate_every=auto_curate_every,
            world_model=(
                None
                if world_model_step_size is None
                else ActionConditionedWorldModelConfig(
                    observation_dim=2,
                    n_actions=2,
                    hidden_sizes=(),
                    step_size=world_model_step_size,
                    sparsity=0.0,
                    use_layer_norm=False,
                )
            ),
            buffer_capacity=4,
        )
    )


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(
            dtype,
            jax.dtypes.prng_key,
        ):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _at_exact_step(
    state: PrototypeAgentState,
    *,
    step_words: tuple[int, int],
    observation_words: tuple[int, int] | None = None,
) -> PrototypeAgentState:
    """Move a minimal valid state to an exact synthetic lifetime."""

    exact_step = jnp.asarray(step_words, dtype=jnp.uint32)
    exact_observation = jnp.asarray(
        step_words if observation_words is None else observation_words,
        dtype=jnp.uint32,
    )
    telemetry = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    base = state.oak_state.stomp_state.base_learner_state.replace(
        step_count=telemetry,
        step_words=exact_step,
    )
    stomp = state.oak_state.stomp_state.replace(
        base_learner_state=base,
        step_count=telemetry,
        step_words=exact_step,
    )
    oak = state.oak_state.replace(
        stomp_state=stomp,
        step_count=telemetry,
        step_words=exact_step,
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=oak,
            step_count=telemetry,
            step_words=exact_step,
            observation_event_count=telemetry,
            observation_event_words=exact_observation,
        ),
    )


def test_init_start_and_update_align_exact_outer_and_oak_clocks() -> None:
    agent = _agent()
    initial = agent.init(jr.key(0))
    chex.assert_trees_all_equal(initial.step_words, jnp.zeros(2, dtype=jnp.uint32))
    chex.assert_trees_all_equal(
        initial.observation_event_words,
        jnp.zeros(2, dtype=jnp.uint32),
    )

    state = agent.start(initial, jnp.zeros(2, dtype=jnp.float32))
    chex.assert_trees_all_equal(
        state.observation_event_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    result = agent.update(state, jnp.asarray(0.0), jnp.ones(2, dtype=jnp.float32))

    assert bool(result.transition_diagnostics.outer_counter_valid)
    assert bool(result.transition_diagnostics.current_counter_capacity_available)
    chex.assert_trees_all_equal(
        result.transition_diagnostics.pre_step_words,
        jnp.zeros(2, dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.transition_diagnostics.proposed_step_words,
        result.state.step_words,
    )
    chex.assert_trees_all_equal(
        result.transition_diagnostics.proposed_observation_event_words,
        result.state.observation_event_words,
    )
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(result.state.step_words, result.state.oak_state.step_words)
    chex.assert_trees_all_equal(
        result.state.step_words,
        result.state.oak_state.stomp_state.step_words,
    )
    chex.assert_trees_all_equal(
        result.state.observation_event_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )


def test_legacy_direct_world_model_refusal_remains_best_effort() -> None:
    agent = _agent(world_model_step_size=1.0e20)
    state = agent.start(agent.init(jr.key(101)), jnp.zeros(2, dtype=jnp.float32))

    first = agent.update(
        state,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert bool(first.transition_diagnostics.valid)
    first_world = agent._action_world_model_component_state(
        first.state.world_model_state
    )
    chex.assert_trees_all_equal(
        first_world.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )

    second = agent.update(
        first.state,
        jnp.asarray(0.75, dtype=jnp.float32),
        -jnp.ones(2, dtype=jnp.float32),
    )
    assert bool(second.transition_diagnostics.valid)
    assert int(second.state.step_count) == 2
    second_world = agent._action_world_model_component_state(
        second.state.world_model_state
    )
    chex.assert_trees_all_equal(second_world.step_words, first_world.step_words)
    assert int(second.state.buffer_state.size) == 2
    assert bool(agent._checkpoint_state_valid(second.state))

    third = agent.update(
        second.state,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert bool(third.transition_diagnostics.valid)
    assert int(third.state.step_count) == 3
    third_world = agent._action_world_model_component_state(
        third.state.world_model_state
    )
    chex.assert_trees_all_equal(third_world.step_words, first_world.step_words)


def test_transition_continues_after_int32_telemetry_saturates() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(1)), jnp.zeros(2, dtype=jnp.float32))
    state = _at_exact_step(
        state,
        step_words=(0, _INT32_MAX),
        observation_words=(0, _INT32_MAX + 1),
    )

    result = jax.jit(agent.update)(
        state,
        jnp.asarray(0.0),
        jnp.ones(2, dtype=jnp.float32),
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.state.started)
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.observation_event_count) == _INT32_MAX
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((0, _INT32_MAX + 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.observation_event_words,
        jnp.asarray((0, _INT32_MAX + 2), dtype=jnp.uint32),
    )


def test_transition_carries_low_step_word_without_x64() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(11)), jnp.zeros(2, dtype=jnp.float32))
    state = _at_exact_step(
        state,
        step_words=(0, _UINT32_MAX),
        observation_words=(1, 0),
    )

    result = jax.jit(agent.update)(
        state,
        jnp.asarray(0.0),
        jnp.ones(2, dtype=jnp.float32),
    )

    assert bool(result.transition_diagnostics.valid)
    chex.assert_trees_all_equal(
        result.transition_diagnostics.pre_step_words,
        jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(result.state.step_words, result.state.oak_state.step_words)


def test_jitted_lax_scan_preserves_sequential_step_word_carry() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(12)), jnp.zeros(2, dtype=jnp.float32))
    state = _at_exact_step(
        state,
        step_words=(0, _UINT32_MAX - 1),
        observation_words=(0, _UINT32_MAX),
    )
    observations = jnp.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=jnp.float32)

    def run_scan(
        source: PrototypeAgentState,
    ) -> tuple[PrototypeAgentState, jax.Array]:
        def body(
            carry: PrototypeAgentState,
            observation: jax.Array,
        ) -> tuple[PrototypeAgentState, jax.Array]:
            updated = agent.update(carry, jnp.asarray(0.0), observation)
            return updated.state, updated.state.step_words

        return jax.lax.scan(body, source, observations)

    final_state, word_trace = jax.jit(run_scan)(state)

    chex.assert_trees_all_equal(
        word_trace,
        jnp.asarray(
            ((0, _UINT32_MAX), (1, 0)),
            dtype=jnp.uint32,
        ),
    )
    chex.assert_trees_all_equal(final_state.step_words, word_trace[-1])


@pytest.mark.parametrize("corrupt_nested", [False, True])
def test_corrupt_outer_or_nested_exact_clock_rejects_atomically(
    corrupt_nested: bool,
) -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(2)), jnp.zeros(2, dtype=jnp.float32))
    if corrupt_nested:
        oak = state.oak_state.replace(
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32)
        )
        corrupt = state.replace(oak_state=oak)
    else:
        corrupt = state.replace(step_words=jnp.asarray((0, 1), dtype=jnp.uint32))

    result = jax.jit(agent.update)(
        corrupt,
        jnp.asarray(0.0),
        jnp.ones(2, dtype=jnp.float32),
    )

    assert not bool(result.transition_diagnostics.state_consistent)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(corrupt),
    )


@pytest.mark.parametrize(
    "observation_words",
    [
        pytest.param((1, 3), id="observation-not-after-step"),
        pytest.param((2, 10), id="observation-exceeds-two-per-step"),
    ],
)
def test_impossible_outer_observation_history_rejects_bit_exactly(
    observation_words: tuple[int, int],
) -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(21)), jnp.zeros(2, dtype=jnp.float32))
    corrupt = _at_exact_step(
        state,
        step_words=(1, 4),
        observation_words=observation_words,
    )

    result = jax.jit(agent.update)(
        corrupt,
        jnp.asarray(0.0),
        jnp.ones(2, dtype=jnp.float32),
    )

    assert not bool(result.transition_diagnostics.outer_counter_valid)
    assert not bool(result.transition_diagnostics.state_consistent)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(corrupt),
    )


def test_boundary_records_two_exact_observation_events() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(3)), jnp.zeros(2, dtype=jnp.float32))
    result = agent.update_transition(
        state,
        PrototypeTransition(
            observation=state.current_raw_observation,
            action=state.current_action,
            decision_id=state.current_decision_id,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(0.0),
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
            next_observation=jnp.ones(2, dtype=jnp.float32),
            next_decision_observation=-jnp.ones(2, dtype=jnp.float32),
        ),
    )

    assert bool(result.transition_diagnostics.valid)
    chex.assert_trees_all_equal(
        result.state.observation_event_words,
        jnp.asarray((0, 3), dtype=jnp.uint32),
    )


def test_last_fully_reservable_exact_step_is_processed_then_state_disarms() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(4)), jnp.zeros(2, dtype=jnp.float32))
    state = _at_exact_step(
        state,
        step_words=(_UINT32_MAX, _UINT32_MAX - 3),
        observation_words=(_UINT32_MAX, _UINT32_MAX - 2),
    )

    result = agent.update(state, jnp.asarray(0.0), jnp.ones(2, dtype=jnp.float32))

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.transition_diagnostics.next_counter_capacity_available)
    assert not bool(result.state.started)
    assert int(result.action) == -1
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((_UINT32_MAX, _UINT32_MAX - 2), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.observation_event_words,
        jnp.asarray((_UINT32_MAX, _UINT32_MAX - 1), dtype=jnp.uint32),
    )
    assert bool(agent._checkpoint_state_valid(result.state))


def test_maybe_curate_uses_exact_clock_above_uint32_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(auto_curate_every=5)
    state = agent.start(agent.init(jr.key(5)), jnp.zeros(2, dtype=jnp.float32))
    # 2**32 + 4 is divisible by five; saturated int32 telemetry is not.
    state = _at_exact_step(state, step_words=(1, 4), observation_words=(1, 5))
    assert int(_words_modulo(state.step_words, 5)) == 0
    called = False

    def fake_curate(
        self: PrototypeAgent,
        source: PrototypeAgentState,
        key: jax.Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[PrototypeAgent, PrototypeAgentState]:
        del key, available_feature_indices
        nonlocal called
        called = True
        return self, source

    monkeypatch.setattr(PrototypeAgent, "curate", fake_curate)
    agent.maybe_curate(state, jr.key(6))
    assert called


def test_auto_curate_interval_outside_exact_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="exact cadence"):
        _agent(auto_curate_every=_INT32_MAX + 1)


def test_checkpoint_roundtrip_authenticates_exact_outer_clocks(tmp_path: Path) -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(7)), jnp.zeros(2, dtype=jnp.float32))
    state = agent.update(state, jnp.asarray(0.0), jnp.ones(2, dtype=jnp.float32)).state
    path = tmp_path / "prototype-exact"

    save_prototype_checkpoint(agent, state, path)
    restored_agent, restored = load_prototype_checkpoint(path)

    assert PROTOTYPE_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v13"
    assert bool(restored_agent._checkpoint_state_valid(restored))
    chex.assert_trees_all_equal(
        _materialize_keys(restored),
        _materialize_keys(state),
    )


def test_exact_outer_clock_resource_delta_is_declared() -> None:
    agent = _agent()
    state = agent.init(jr.key(8))
    assert PROTOTYPE_LIFETIME_COUNTER_DELTA_NBYTES == 16
    assert prototype_lifetime_counter_nbytes() == 24
    assert state.step_words.nbytes + state.observation_event_words.nbytes == 16


def test_exact_outer_clock_resources_are_publicly_exported() -> None:
    import alberta_framework as public
    import alberta_framework.core as core

    assert public.PROTOTYPE_LIFETIME_COUNTER_DELTA_NBYTES == 16
    assert core.PROTOTYPE_LIFETIME_COUNTER_NBYTES == 24
    assert public.prototype_lifetime_counter_nbytes is prototype_lifetime_counter_nbytes


def test_legacy_saturated_counter_migration_is_ambiguous() -> None:
    with pytest.raises(ValueError, match="saturated.*ambiguous"):
        _legacy_words(
            jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            name="test counter",
        )
