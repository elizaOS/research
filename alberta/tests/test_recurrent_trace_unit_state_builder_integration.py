"""Ordinary PrototypeAgent integration for the RTU state-builder path."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.state_builder import RecurrentTraceUnitStateBuilderConfig

pytestmark = pytest.mark.integration


def _agent() -> PrototypeAgent:
    builder = RecurrentTraceUnitStateBuilderConfig(
        observation_dim=2,
        n_actions=2,
        hidden_dim=2,
        include_raw_observation=True,
        rtrl_taylor_correction=False,
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=builder.feature_dim(),
                    n_primitive_actions=2,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=builder,
        )
    )


def _transition(
    state: Any,
    next_observation: jax.Array,
    reward: jax.Array,
) -> PrototypeTransition:
    return PrototypeTransition(  # type: ignore[call-arg]
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=reward,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype,
            jax.dtypes.prng_key,
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_rtu_builder_runs_through_prototype_start_update_and_authoritative_scan() -> None:
    agent = _agent()
    initial = agent.start(
        agent.init(jr.key(10, impl="threefry2x32")),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    assert int(initial.state_builder_state.step_count) == 1
    assert initial.current_representation.shape == (6,)

    observations = jnp.asarray(
        [[0.3, 0.4], [-0.5, 0.2], [0.8, -0.1]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    loop_state = initial
    transitions = []
    actions = []
    for observation, reward in zip(observations, rewards, strict=True):
        transition = _transition(loop_state, observation, reward)
        transitions.append(transition)
        result = agent.update_transition(loop_state, transition)
        assert bool(result.transition_diagnostics.valid)
        loop_state = result.state
        actions.append(result.action)

    batched = jax.tree.map(lambda *values: jnp.stack(values), *transitions)
    scanned = jax.jit(agent.scan_transitions)(initial, batched)
    chex.assert_trees_all_close(
        _materialize_keys(scanned.state),
        _materialize_keys(loop_state),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    chex.assert_trees_all_equal(scanned.actions, jnp.stack(actions))
    chex.assert_trees_all_equal(
        scanned.transition_valid,
        jnp.ones((3,), dtype=jnp.bool_),
    )
