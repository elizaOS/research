# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Explicit continuation-discount regressions for Intelligence Amplification."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAConfig,
    IAState,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)

OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)
REWARD = jnp.array(1.7, dtype=jnp.float32)
EXECUTED_ACTION = 0
IA_OWN_ACTION = 1
BASE_AVERAGE_REWARD = 0.4
BASE_STEP_SIZE = 0.05
NEXT_MAX_Q = 2.0
EXECUTED_Q = 1.0


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
            base_step_size=BASE_STEP_SIZE,
            base_avg_reward_step_size=0.01,
            base_trace_decay=0.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _ia_config() -> IAConfig:
    return IAConfig(
        cerebellum=ExoCerebellumConfig(n_demons=1, obs_dim=2),
        cortex=_oak_config(),
    )


def _prepare_ia_state(state: IAState) -> IAState:
    """Give the cortex hand-derived linear Q-values and a disagreeing action."""
    learner = state.cortex_state.stomp_state.base_learner_state
    weights = (
        jnp.array([[EXECUTED_Q, 0.0]], dtype=jnp.float32),
        jnp.array([[NEXT_MAX_Q, 0.0]], dtype=jnp.float32),
        jnp.array([[-100.0, 0.0]], dtype=jnp.float32),
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=weights,
            biases=tuple(jnp.zeros_like(bias) for bias in learner.head_params.biases),
        )
    )
    stomp = state.cortex_state.stomp_state.replace(
        base_learner_state=learner,
        base_average_reward=jnp.array(BASE_AVERAGE_REWARD, dtype=jnp.float32),
        base_last_obs=OBS,
        base_last_action=jnp.array(IA_OWN_ACTION, dtype=jnp.int32),
        last_primitive_action=jnp.array(IA_OWN_ACTION, dtype=jnp.int32),
        option_last_intra_action=jnp.array(IA_OWN_ACTION, dtype=jnp.int32),
        executing_option=jnp.array(-1, dtype=jnp.int32),
    )
    return state.replace(cortex_state=state.cortex_state.replace(stomp_state=stomp))


def _head_weight(state: IAState, action: int) -> float:
    weights = state.cortex_state.stomp_state.base_learner_state.head_params.weights
    return float(weights[action][0, 0])


def _expected_td(discount: float) -> float:
    return float(REWARD) - BASE_AVERAGE_REWARD + discount * NEXT_MAX_Q - EXECUTED_Q


def _materialize_typed_keys(tree: object) -> object:
    """Convert typed PRNG leaves so Chex can compare complete states."""

    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


@pytest.mark.parametrize("discount", [0.0, 0.25])
def test_ia_backup_uses_explicit_terminal_and_fractional_discount(
    discount: float,
) -> None:
    """The executed primitive is credited with the exact supplied bootstrap."""
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(0)), OBS))

    result = agent.update(
        state,
        OBS,
        REWARD,
        OBS,
        partner_action=jnp.array(EXECUTED_ACTION, dtype=jnp.int32),
        discount=jnp.array(discount, dtype=jnp.float32),
    )

    expected_td = _expected_td(discount)
    np.testing.assert_allclose(
        np.asarray(result.cortex_td_error).item(),
        expected_td,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _head_weight(result.state, EXECUTED_ACTION),
        EXECUTED_Q + BASE_STEP_SIZE * expected_td,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _head_weight(result.state, IA_OWN_ACTION),
        NEXT_MAX_Q,
        rtol=0.0,
        atol=0.0,
    )


def test_ia_legacy_update_preserves_unit_primitive_bootstrap() -> None:
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(1)), OBS))

    result = agent.update(
        state,
        OBS,
        REWARD,
        OBS,
        partner_action=jnp.array(EXECUTED_ACTION, dtype=jnp.int32),
    )

    expected_td = _expected_td(1.0)
    np.testing.assert_allclose(
        np.asarray(result.cortex_td_error).item(),
        expected_td,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _head_weight(result.state, EXECUTED_ACTION),
        EXECUTED_Q + BASE_STEP_SIZE * expected_td,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("invalid_action", [-1, 2])
def test_ia_rejects_out_of_range_executed_action_eagerly(
    invalid_action: int,
) -> None:
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(11)), OBS))

    with pytest.raises(ValueError, match=r"partner_action must be in \[0, 2\)"):
        agent.update(
            state,
            OBS,
            REWARD,
            OBS,
            partner_action=jnp.array(invalid_action, dtype=jnp.int32),
            discount=jnp.array(0.5, dtype=jnp.float32),
        )


def test_ia_rejects_non_scalar_or_non_integer_executed_action() -> None:
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(12)), OBS))

    with pytest.raises(ValueError, match="partner_action must be scalar"):
        agent.update(
            state,
            OBS,
            REWARD,
            OBS,
            partner_action=jnp.array([0], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="partner_action must have an integer dtype"):
        agent.update(
            state,
            OBS,
            REWARD,
            OBS,
            partner_action=jnp.array(0.0, dtype=jnp.float32),
        )


def test_ia_invalid_traced_action_fails_closed_with_nonfinite_td_error() -> None:
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(13)), OBS))

    compiled_td_error = jax.jit(
        lambda action: agent.update(
            state,
            OBS,
            REWARD,
            OBS,
            partner_action=action,
            discount=jnp.array(0.5, dtype=jnp.float32),
        ).cortex_td_error
    )

    assert not bool(
        jnp.isfinite(compiled_td_error(jnp.array(-1, dtype=jnp.int32)))
    )


@pytest.mark.parametrize(("discount", "expected_discount"), [(0.0, 0.0), (0.4, 0.4)])
def test_prototype_explicit_transition_routes_discount_to_ia(
    discount: float,
    expected_discount: float,
) -> None:
    """Prototype forwards continuation separately from the dispatched action."""
    agent = PrototypeAgent(PrototypeAgentConfig(oak=_oak_config(), ia=_ia_config()))
    state = agent.start(agent.init(jr.key(2)), OBS)
    main_stomp = state.oak_state.stomp_state.replace(
        base_last_obs=OBS,
        last_primitive_action=jnp.array(EXECUTED_ACTION, dtype=jnp.int32),
    )
    state = state.replace(
        oak_state=state.oak_state.replace(stomp_state=main_stomp),
        ia_state=_prepare_ia_state(state.ia_state),
    )

    result = agent.update_transition(
        state,
        PrototypeTransition(
            reward=REWARD,
            next_observation=OBS,
            discount=jnp.array(discount, dtype=jnp.float32),
        ),
    )

    expected_td = _expected_td(expected_discount)
    np.testing.assert_allclose(
        _head_weight(result.state.ia_state, EXECUTED_ACTION),
        EXECUTED_Q + BASE_STEP_SIZE * expected_td,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _head_weight(result.state.ia_state, IA_OWN_ACTION),
        NEXT_MAX_Q,
        rtol=0.0,
        atol=0.0,
    )


def test_prototype_legacy_update_keeps_ia_legacy_bootstrap() -> None:
    agent = PrototypeAgent(PrototypeAgentConfig(oak=_oak_config(), ia=_ia_config()))
    state = agent.start(agent.init(jr.key(3)), OBS)
    state = state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=state.oak_state.stomp_state.replace(
                base_last_obs=OBS,
                last_primitive_action=jnp.array(EXECUTED_ACTION, dtype=jnp.int32),
            )
        ),
        ia_state=_prepare_ia_state(state.ia_state),
    )

    result = agent.update(state, REWARD, OBS)

    expected_td = _expected_td(1.0)
    np.testing.assert_allclose(
        _head_weight(result.state.ia_state, EXECUTED_ACTION),
        EXECUTED_Q + BASE_STEP_SIZE * expected_td,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _head_weight(result.state.ia_state, IA_OWN_ACTION),
        NEXT_MAX_Q,
        rtol=0.0,
        atol=0.0,
    )


def test_discounted_scan_matches_loop_and_jit() -> None:
    agent = IAAgent(_ia_config())
    state = _prepare_ia_state(agent.start(agent.init(jr.key(4)), OBS))
    num_steps = 4
    observations = jnp.tile(OBS, (num_steps, 1))
    rewards = jnp.array([1.7, -0.2, 0.5, 1.1], dtype=jnp.float32)
    actions = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    discounts = jnp.array([0.0, 0.25, 0.8, 1.0], dtype=jnp.float32)

    loop_state = state
    for index in range(num_steps):
        loop_state = agent.update(
            loop_state,
            observations[index],
            rewards[index],
            observations[index],
            partner_action=actions[index],
            discount=discounts[index],
        ).state

    scan_result = agent.scan(
        state,
        observations,
        rewards,
        observations,
        partner_actions=actions,
        discounts=discounts,
    )
    jit_result = jax.jit(agent.scan)(
        state,
        observations,
        rewards,
        observations,
        partner_actions=actions,
        discounts=discounts,
    )

    chex.assert_trees_all_close(
        _materialize_typed_keys(scan_result.state),
        _materialize_typed_keys(loop_state),
        rtol=1e-6,
        atol=1e-6,
    )
    chex.assert_trees_all_close(
        _materialize_typed_keys(jit_result),
        _materialize_typed_keys(scan_result),
        rtol=1e-6,
        atol=1e-6,
    )
