"""Prototype integration regression for IA executed-action credit."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig

OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)
EXECUTED_ACTION = 0
IA_OWN_ACTION = 1


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_trace_decay=0.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _changed_base_heads(before, after) -> list[bool]:
    before_params = before.stomp_state.base_learner_state.head_params
    after_params = after.stomp_state.base_learner_state.head_params
    return [
        bool(jnp.any(w_before != w_after)) or bool(jnp.any(b_before != b_after))
        for w_before, w_after, b_before, b_after in zip(
            before_params.weights,
            after_params.weights,
            before_params.biases,
            after_params.biases,
        )
    ]


def test_prototype_ia_credits_primitive_actually_dispatched_by_oak() -> None:
    """IA's own action may disagree, but only the executed head is updated."""
    ia_config = IAConfig(
        cerebellum=ExoCerebellumConfig(n_demons=1, obs_dim=2),
        cortex=_oak_config(),
    )
    agent = PrototypeAgent(
        PrototypeAgentConfig(oak=_oak_config(), ia=ia_config)
    )
    state = agent.start(agent.init(jr.key(0)), OBS)

    # The main OaK agent dispatched action 0, while the independent IA cortex
    # believes it selected action 1. PrototypeAgent.update must route the
    # former into IAAgent.update(partner_action=...).
    main_stomp = state.oak_state.stomp_state.replace(
        last_primitive_action=jnp.array(EXECUTED_ACTION, dtype=jnp.int32)
    )
    ia_cortex_stomp = state.ia_state.cortex_state.stomp_state.replace(
        base_last_action=jnp.array(IA_OWN_ACTION, dtype=jnp.int32),
        last_primitive_action=jnp.array(IA_OWN_ACTION, dtype=jnp.int32),
        executing_option=jnp.array(-1, dtype=jnp.int32),
    )
    state = state.replace(
        oak_state=state.oak_state.replace(stomp_state=main_stomp),
        ia_state=state.ia_state.replace(
            cortex_state=state.ia_state.cortex_state.replace(
                stomp_state=ia_cortex_stomp
            )
        ),
    )
    before_cortex = state.ia_state.cortex_state

    result = agent.update(state, jnp.array(2.0, dtype=jnp.float32), OBS)
    changed = _changed_base_heads(
        before_cortex,
        result.state.ia_state.cortex_state,
    )

    assert changed[EXECUTED_ACTION]
    assert not changed[IA_OWN_ACTION]
    assert sum(changed) == 1
