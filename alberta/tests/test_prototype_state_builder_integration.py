"""Authoritative PrototypeTransition and causal StateBuilder integration tests."""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    GRUPerceptionConfig,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    _prototype_config_digest,
    _PrototypeAgentStateV1,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.state_builder import (
    FixedTraceStateBuilderConfig,
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    StateBuilderConfig,
    state_builder_from_config,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = pytest.mark.unit

RAW_DIM = 2
N_ACTIONS = 2


def _builder_configs() -> tuple[StateBuilderConfig, ...]:
    return (
        IdentityStateBuilderConfig(observation_dim=RAW_DIM),
        FixedTraceStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            outcome_decay_rates=(),
            include_raw_observation=True,
        ),
        OnlineGatedStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=2,
            include_raw_observation=True,
            step_size=0.01,
        ),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _agent(
    builder_config: StateBuilderConfig,
    *,
    world_model: bool = False,
) -> PrototypeAgent:
    builder = state_builder_from_config(builder_config.to_config())
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=builder.feature_dim(),
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            state_builder=builder_config,
            world_model=(
                ActionConditionedWorldModelConfig(
                    observation_dim=builder.feature_dim(),
                    n_actions=N_ACTIONS,
                    hidden_sizes=(),
                    step_size=0.1,
                    gamma=0.95,
                )
                if world_model
                else None
            ),
        )
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.5,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    observation: jax.Array | None = None,
    action: jax.Array | None = None,
    decision_id: jax.Array | None = None,
    next_decision_observation: jax.Array | None = None,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=(
            state.current_raw_observation if observation is None else observation
        ),
        action=state.current_action if action is None else action,
        decision_id=(
            state.current_decision_id if decision_id is None else decision_id
        ),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
        next_observation=next_observation,
        next_decision_observation=(
            next_observation
            if next_decision_observation is None
            else next_decision_observation
        ),
    )


def test_state_builder_configuration_is_dimensioned_and_mutually_exclusive() -> None:
    identity = IdentityStateBuilderConfig(observation_dim=RAW_DIM)
    valid = _agent(identity).config
    with pytest.raises(ValueError, match="mutually exclusive"):
        PrototypeAgentConfig(
            oak=valid.oak,
            state_builder=identity,
            gru_perception=GRUPerceptionConfig(
                observation_dim=RAW_DIM,
                hidden_dim=1,
            ),
        )
    with pytest.raises(ValueError, match="feature_dim"):
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=RAW_DIM + 1,
                    n_primitive_actions=N_ACTIONS,
                )
            ),
            state_builder=identity,
        )
    with pytest.raises(ValueError, match="n_actions"):
        _agent(
            FixedTraceStateBuilderConfig(
                observation_dim=RAW_DIM,
                n_actions=3,
                observation_decay_rates=(),
                action_decay_rates=(),
                outcome_decay_rates=(),
            )
        )


@pytest.mark.parametrize("builder_config", _builder_configs())
def test_start_advances_builder_once_and_caches_the_dispatched_decision(
    builder_config: StateBuilderConfig,
) -> None:
    agent = _agent(builder_config)
    initial = agent.init(jr.key(1))
    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)

    state = agent.start(initial, observation)

    assert int(state.state_builder_state.step_count) == 1
    chex.assert_trees_all_equal(state.current_raw_observation, observation)
    chex.assert_trees_all_close(
        state.current_representation,
        state.oak_state.stomp_state.base_last_obs,
    )
    chex.assert_trees_all_equal(
        state.current_action,
        state.oak_state.stomp_state.last_primitive_action,
    )

    state_before = jax.tree.map(lambda leaf: leaf.copy(), state)
    eager_action = agent.act(state, observation)
    compiled_action = jax.jit(agent.act)(state, observation)
    chex.assert_trees_all_equal(eager_action, state.current_action)
    chex.assert_trees_all_equal(compiled_action, state.current_action)
    chex.assert_trees_all_equal(state, state_before)


def test_act_mismatch_fails_eager_and_returns_unarmed_sentinel_when_traced() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(2)), jnp.zeros(RAW_DIM))
    mismatched = jnp.ones(RAW_DIM, dtype=jnp.float32)

    with pytest.raises(ValueError, match="cached decision"):
        agent.act(state, mismatched)

    traced = jax.jit(agent.act)(state, mismatched)
    chex.assert_trees_all_equal(traced, jnp.asarray(-1, dtype=jnp.int32))


def test_start_is_one_shot_and_dynamic_invalid_start_is_atomic() -> None:
    agent = _agent(_builder_configs()[2])
    initial = agent.init(jr.key(20))
    observation = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    started = agent.start(initial, observation)

    with pytest.raises(RuntimeError, match="fresh unstarted"):
        agent.start(started, observation)

    repeated = jax.jit(agent.start)(started, observation)
    nan_start = jax.jit(agent.start)(
        initial,
        jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        _materialize_keys(repeated),
        _materialize_keys(started),
    )
    chex.assert_trees_all_equal(
        _materialize_keys(nan_start),
        _materialize_keys(initial),
    )


def test_start_rejects_a_nonfinite_derived_state_atomically() -> None:
    agent = _agent(_builder_configs()[2])
    initial = agent.init(jr.key(1))
    maximum = jnp.finfo(jnp.float32).max
    finite_extreme = jnp.asarray([maximum, -maximum], dtype=jnp.float32)

    with pytest.raises(ValueError, match="non-finite agent state"):
        agent.start(initial, finite_extreme)

    compiled = jax.jit(agent.start)(initial, finite_extreme)
    chex.assert_trees_all_equal(
        _materialize_keys(compiled),
        _materialize_keys(initial),
    )


def test_explicit_decision_id_rejects_aba_and_cross_lifecycle_replay() -> None:
    agent = _agent(_builder_configs()[0])
    observation = jnp.zeros(RAW_DIM, dtype=jnp.float32)
    first = agent.start(
        agent.init(jr.key(21), lifecycle_id=jnp.asarray([1, 2], dtype=jnp.uint32)),
        observation,
    )
    first_transition = _transition(first, observation)
    second = agent.update_transition(first, first_transition).state

    stale = _transition(
        second,
        observation,
        decision_id=first.current_decision_id,
    )
    stale_result = agent.update_transition(second, stale)
    assert bool(stale_result.transition_diagnostics.observation_matches)
    assert bool(stale_result.transition_diagnostics.action_matches)
    assert not bool(stale_result.transition_diagnostics.decision_id_matches)
    assert bool(stale_result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(stale_result.state),
        _materialize_keys(second),
    )

    other = agent.start(
        agent.init(jr.key(21), lifecycle_id=jnp.asarray([3, 4], dtype=jnp.uint32)),
        observation,
    )
    cross_session = agent.update_transition(other, first_transition)
    assert not bool(cross_session.transition_diagnostics.decision_id_matches)
    assert bool(cross_session.transition_diagnostics.rejected)


@pytest.mark.parametrize("action", [np.uint64(2**32), np.uint64(2**32 + 1)])
def test_action_ownership_rejects_lossy_integer_aliases(action: np.uint64) -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(22)), jnp.zeros(RAW_DIM))

    with pytest.raises(ValueError, match="losslessly representable as int32"):
        agent.update_transition(
            state,
            _transition(state, jnp.ones(RAW_DIM), action=action),
        )


def test_lossless_integer_action_dtype_has_eager_jit_parity() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(221)), jnp.zeros(RAW_DIM))
    transition = _transition(
        state,
        jnp.ones(RAW_DIM),
        action=jnp.asarray(state.current_action, dtype=jnp.int16),
    )

    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)

    assert bool(eager.transition_diagnostics.valid)
    assert bool(compiled.transition_diagnostics.valid)
    chex.assert_trees_all_close(
        _materialize_keys(compiled),
        _materialize_keys(eager),
        atol=1.0e-6,
    )


def test_traced_lossy_action_alias_is_an_atomic_rejection() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(222)), jnp.zeros(RAW_DIM))
    transition = _transition(
        state,
        jnp.ones(RAW_DIM),
        action=jnp.asarray(2**32 - 1, dtype=jnp.uint32),
    )

    result = jax.jit(agent.update_transition)(state, transition)

    assert not bool(result.transition_diagnostics.action_in_range)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(state),
    )


def test_corrupt_recurrent_state_is_an_atomic_transition_rejection() -> None:
    agent = _agent(_builder_configs()[2])
    state = agent.start(agent.init(jr.key(23)), jnp.zeros(RAW_DIM))
    corrupted_builder = state.state_builder_state.replace(
        hidden=state.state_builder_state.hidden + 100.0
    )
    corrupted = state.replace(state_builder_state=corrupted_builder)

    result = agent.update_transition(
        corrupted,
        _transition(corrupted, jnp.ones(RAW_DIM)),
    )

    assert not bool(result.transition_diagnostics.state_consistent)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(corrupted),
    )


def test_final_decision_generation_processes_outcome_then_disarms() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(24)), jnp.zeros(RAW_DIM))
    maximum = jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)
    exhausted_id = state.current_decision_id.at[2:].set(maximum)
    state = state.replace(current_decision_id=exhausted_id)

    result = agent.update_transition(
        state,
        _transition(state, jnp.ones(RAW_DIM)),
    )

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.transition_diagnostics.next_generation_available)
    assert int(result.state.step_count) == 1
    assert not bool(result.state.started)
    assert int(result.action) == -1
    chex.assert_trees_all_equal(result.state.current_decision_id, exhausted_id)

    for decision in (
        agent.decision(result.state),
        jax.jit(agent.decision)(result.state),
    ):
        assert not bool(decision.armed)
        assert int(decision.action) == -1
        chex.assert_trees_all_equal(decision.decision_id, exhausted_id)


def test_signed_counter_capacity_processes_final_outcome_then_disarms() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(241)), jnp.zeros(RAW_DIM))
    maximum = np.iinfo(np.int32).max
    near_maximum = state.replace(
        state_builder_state=state.state_builder_state.replace(
            step_count=jnp.asarray(maximum - 2, dtype=jnp.int32)
        ),
        observation_event_count=jnp.asarray(maximum - 2, dtype=jnp.int32),
        step_count=jnp.asarray(maximum - 1, dtype=jnp.int32),
        oak_state=state.oak_state.replace(
            step_count=jnp.asarray(maximum - 1, dtype=jnp.int32),
            stomp_state=state.oak_state.stomp_state.replace(
                step_count=jnp.asarray(maximum - 1, dtype=jnp.int32)
            ),
        ),
    )

    result = agent.update_transition(
        near_maximum,
        _transition(near_maximum, jnp.ones(RAW_DIM)),
    )

    assert bool(result.transition_diagnostics.valid)
    assert not bool(
        result.transition_diagnostics.next_counter_capacity_available
    )
    assert int(result.state.step_count) == maximum
    assert int(result.state.oak_state.step_count) == maximum
    assert int(result.state.oak_state.stomp_state.step_count) == maximum
    assert not bool(result.state.started)
    assert int(result.action) == -1
    assert bool(agent._checkpoint_state_valid(result.state))
    assert not bool(agent.decision(result.state).armed)

    corrupt_armed = near_maximum.replace(
        step_count=jnp.asarray(maximum, dtype=jnp.int32),
        oak_state=near_maximum.oak_state.replace(
            step_count=jnp.asarray(maximum, dtype=jnp.int32),
            stomp_state=near_maximum.oak_state.stomp_state.replace(
                step_count=jnp.asarray(maximum, dtype=jnp.int32)
            ),
        ),
    )
    rejected = jax.jit(agent.update_transition)(
        corrupt_armed,
        _transition(corrupt_armed, jnp.ones(RAW_DIM)),
    )
    assert not bool(rejected.transition_diagnostics.state_consistent)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected.state),
        _materialize_keys(corrupt_armed),
    )


@pytest.mark.parametrize("builder_config", _builder_configs())
def test_explicit_transition_advances_builder_once_with_supplied_outcomes(
    builder_config: StateBuilderConfig,
) -> None:
    agent = _agent(builder_config)
    state = agent.start(agent.init(jr.key(3)), jnp.zeros(RAW_DIM))
    next_observation = jnp.asarray([0.8, -0.2], dtype=jnp.float32)
    transition = _transition(state, next_observation, reward=0.7, discount=0.4)
    assert agent.state_builder is not None
    expected_builder_state, expected_representation = agent.state_builder.update(
        state.state_builder_state,
        next_observation,
        state.current_action,
        0.7,
        0.4,
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert int(result.state.step_count) == 1
    chex.assert_trees_all_close(result.state.state_builder_state, expected_builder_state)
    chex.assert_trees_all_close(
        result.state.current_representation,
        expected_representation,
    )
    chex.assert_trees_all_equal(
        result.state.current_action,
        result.action,
    )


@pytest.mark.parametrize(
    ("discount", "terminated", "truncated", "expected_valid"),
    (
        (0.9, False, False, True),
        (0.0, True, False, True),
        (0.8, False, True, True),
        (0.0, True, True, True),
        (0.9, True, False, False),
        (0.0, False, True, False),
    ),
)
def test_terminal_and_truncation_semantics_are_explicit_and_fail_closed(
    discount: float,
    terminated: bool,
    truncated: bool,
    expected_valid: bool,
) -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(4)), jnp.zeros(RAW_DIM))
    transition = _transition(
        state,
        jnp.ones(RAW_DIM),
        discount=discount,
        terminated=terminated,
        truncated=truncated,
    )

    result = jax.jit(agent.update_transition)(state, transition)

    assert bool(result.transition_diagnostics.valid) is expected_valid
    if expected_valid:
        assert int(result.state.step_count) == 1
    else:
        chex.assert_trees_all_equal(result.state, state)
        assert int(result.action) == int(state.current_action)
        assert float(result.oak_td_error) == 0.0


@pytest.mark.parametrize("builder_config", _builder_configs())
def test_autoreset_uses_bootstrap_for_learning_and_reset_for_next_decision(
    builder_config: StateBuilderConfig,
) -> None:
    agent = _agent(builder_config, world_model=True)
    initial_raw = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(40)), initial_raw)
    final_raw = jnp.asarray([0.9, 0.1], dtype=jnp.float32)
    reset_raw = jnp.asarray([-0.8, 0.3], dtype=jnp.float32)
    reward = 0.7
    discount = 0.8
    assert agent.state_builder is not None
    bootstrap_builder_state, bootstrap_representation = agent.state_builder.update(
        state.state_builder_state,
        final_raw,
        state.current_action,
        reward,
        discount,
    )
    reset_builder_state = agent.state_builder.reset_episode(
        bootstrap_builder_state
    )
    expected_builder_state, expected_decision_representation = (
        agent.state_builder.start(reset_builder_state, reset_raw)
    )
    assert agent._world_model is not None
    prediction = agent._world_model.predict(
        state.world_model_state,
        state.current_representation,
        state.current_action,
    )
    expected_model_error = (
        jnp.mean(
            (prediction.next_observation - bootstrap_representation) ** 2
        )
        + (prediction.reward - reward) ** 2
        + (prediction.discount - discount) ** 2
    )

    result = agent.update_transition(
        state,
        _transition(
            state,
            final_raw,
            reward=reward,
            discount=discount,
            truncated=True,
            next_decision_observation=reset_raw,
        ),
    )

    assert bool(result.transition_diagnostics.valid)
    assert int(result.state.step_count) == 1
    assert int(result.state.observation_event_count) == 3
    chex.assert_trees_all_close(
        result.state.state_builder_state,
        expected_builder_state,
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(
        result.state.current_raw_observation,
        reset_raw,
    )
    chex.assert_trees_all_close(
        result.state.current_representation,
        expected_decision_representation,
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(
        result.state.oak_state.stomp_state.base_last_obs,
        expected_decision_representation,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        result.world_model_error,
        expected_model_error,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.state.buffer_state is not None
    assert int(result.state.buffer_state.size) == 2
    chex.assert_trees_all_close(
        result.state.buffer_state.observations[0],
        bootstrap_representation,
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(
        result.state.buffer_state.observations[1],
        expected_decision_representation,
        atol=1.0e-6,
    )


def test_distinct_decision_observation_without_boundary_is_rejected() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(41)), jnp.zeros(RAW_DIM))
    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.ones(RAW_DIM),
            next_decision_observation=-jnp.ones(RAW_DIM),
        ),
    )

    assert not bool(result.transition_diagnostics.boundary_semantics_valid)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(state),
    )


@pytest.mark.parametrize("failure", ("observation", "action", "nonfinite"))
def test_dynamic_transition_ownership_failures_are_atomic_eager_and_jit(
    failure: str,
) -> None:
    agent = _agent(_builder_configs()[2])
    state = agent.start(agent.init(jr.key(5)), jnp.zeros(RAW_DIM))
    kwargs: dict[str, object] = {}
    if failure == "observation":
        kwargs["observation"] = jnp.ones(RAW_DIM, dtype=jnp.float32)
    elif failure == "action":
        kwargs["action"] = (state.current_action + 1) % N_ACTIONS
    else:
        kwargs["reward"] = float("nan")
    transition = _transition(
        state,
        jnp.ones(RAW_DIM, dtype=jnp.float32),
        **kwargs,  # type: ignore[arg-type]
    )

    for result in (
        agent.update_transition(state, transition),
        jax.jit(agent.update_transition)(state, transition),
    ):
        assert not bool(result.transition_diagnostics.valid)
        assert bool(result.transition_diagnostics.rejected)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(result.action, state.current_action)
        assert float(result.oak_td_error) == 0.0
        assert not bool(result.transition_diagnostics.post_update_checked)


def test_nonfinite_candidate_state_is_atomic_eager_jit_and_scan() -> None:
    agent = _agent(_builder_configs()[2])
    state = agent.start(agent.init(jr.key(51)), jnp.zeros(RAW_DIM))
    extreme = jnp.finfo(jnp.float32).max
    next_observation = jnp.asarray([extreme, -extreme], dtype=jnp.float32)
    transition = _transition(state, next_observation)
    assert agent.state_builder is not None
    candidate_builder, _ = agent.state_builder.update(
        state.state_builder_state,
        next_observation,
        state.current_action,
        transition.reward,
        transition.discount,
    )
    assert not bool(jnp.all(jnp.isfinite(candidate_builder.parameter_sensitivity)))

    for result in (
        agent.update_transition(state, transition),
        jax.jit(agent.update_transition)(state, transition),
    ):
        assert bool(result.transition_diagnostics.post_update_checked)
        assert not bool(result.transition_diagnostics.post_update_finite)
        assert bool(result.transition_diagnostics.post_update_consistent)
        assert not bool(result.transition_diagnostics.valid)
        assert bool(result.transition_diagnostics.rejected)
        assert float(result.oak_td_error) == 0.0
        chex.assert_trees_all_equal(
            _materialize_keys(result.state),
            _materialize_keys(state),
        )

    batched = jax.tree.map(lambda leaf: leaf[None], transition)
    scanned = jax.jit(agent.scan_transitions)(state, batched)
    chex.assert_trees_all_equal(scanned.transition_valid, jnp.asarray([False]))
    chex.assert_trees_all_equal(
        _materialize_keys(scanned.state),
        _materialize_keys(state),
    )
    chex.assert_trees_all_equal(scanned.actions, state.current_action[None])
    chex.assert_trees_all_equal(
        scanned.oak_td_errors,
        jnp.zeros((1,), dtype=jnp.float32),
    )


def test_static_transition_shape_action_and_decision_id_drift_raise() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(6)), jnp.zeros(RAW_DIM))
    valid = _transition(state, jnp.ones(RAW_DIM))

    with pytest.raises(ValueError, match="shape"):
        agent.update_transition(
            state,
            valid.replace(observation=jnp.zeros((1, RAW_DIM))),
        )
    with pytest.raises(ValueError, match="scalar integer"):
        agent.update_transition(
            state,
            valid.replace(action=jnp.asarray(0.0, dtype=jnp.float32)),
        )
    with pytest.raises(ValueError, match=r"shape \(4,\).*uint32"):
        agent.update_transition(
            state,
            valid.replace(decision_id=jnp.zeros((2,), dtype=jnp.uint32)),
        )
    with pytest.raises(ValueError, match="dtype uint32"):
        agent.update_transition(
            state,
            valid.replace(decision_id=jnp.zeros((4,), dtype=jnp.int32)),
        )


def test_legacy_update_and_array_scan_reject_canonical_state_builder() -> None:
    agent = _agent(_builder_configs()[0])
    state = agent.start(agent.init(jr.key(61)), jnp.zeros(RAW_DIM))
    with pytest.raises(ValueError, match="legacy update"):
        agent.update(state, jnp.asarray(0.0), jnp.ones(RAW_DIM))
    with pytest.raises(ValueError, match="legacy array scan"):
        agent.scan(
            state,
            jnp.zeros(1),
            jnp.ones((1, RAW_DIM)),
        )


@pytest.mark.parametrize("builder_config", _builder_configs())
def test_scan_transitions_matches_repeated_authoritative_updates(
    builder_config: StateBuilderConfig,
) -> None:
    agent = _agent(builder_config)
    initial = agent.start(agent.init(jr.key(7)), jnp.zeros(RAW_DIM))
    next_observations = jnp.asarray(
        [[0.2, 0.4], [-0.3, 0.8], [0.7, -0.1]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.5, -0.2, 1.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.6, 0.0], dtype=jnp.float32)
    loop_state = initial
    transitions: list[PrototypeTransition] = []
    loop_actions = []
    for index in range(next_observations.shape[0]):
        transition = _transition(
            loop_state,
            next_observations[index],
            reward=float(rewards[index]),
            discount=float(discounts[index]),
            terminated=bool(discounts[index] == 0.0),
        )
        transitions.append(transition)
        result = agent.update_transition(loop_state, transition)
        loop_state = result.state
        loop_actions.append(result.action)

    batched = PrototypeTransition(
        observation=jnp.stack([item.observation for item in transitions]),
        action=jnp.stack([item.action for item in transitions]),
        decision_id=jnp.stack([item.decision_id for item in transitions]),
        reward=jnp.stack([item.reward for item in transitions]),
        discount=jnp.stack([item.discount for item in transitions]),
        terminated=jnp.stack([item.terminated for item in transitions]),
        truncated=jnp.stack([item.truncated for item in transitions]),
        next_observation=jnp.stack(
            [item.next_observation for item in transitions]
        ),
        next_decision_observation=jnp.stack(
            [item.next_decision_observation for item in transitions]
        ),
    )

    scanned = jax.jit(agent.scan_transitions)(initial, batched)

    chex.assert_trees_all_close(
        _materialize_keys(scanned.state),
        _materialize_keys(loop_state),
        atol=1.0e-6,
    )
    chex.assert_trees_all_equal(scanned.actions, jnp.stack(loop_actions))
    chex.assert_trees_all_equal(
        scanned.transition_valid,
        jnp.ones(len(transitions), dtype=jnp.bool_),
    )


def test_scan_invalid_element_is_transactional_and_later_owned_step_recovers() -> None:
    agent = _agent(_builder_configs()[2], world_model=True)
    initial = agent.start(agent.init(jr.key(70)), jnp.zeros(RAW_DIM))
    first = _transition(initial, jnp.asarray([0.2, 0.4]))
    after_first = agent.update_transition(initial, first).state
    invalid = _transition(
        after_first,
        jnp.asarray([-0.9, 0.8]),
        decision_id=initial.current_decision_id,
    )
    third = _transition(after_first, jnp.asarray([0.7, -0.1]))
    transitions = (first, invalid, third)
    batched = PrototypeTransition(
        observation=jnp.stack([item.observation for item in transitions]),
        action=jnp.stack([item.action for item in transitions]),
        decision_id=jnp.stack([item.decision_id for item in transitions]),
        reward=jnp.stack([item.reward for item in transitions]),
        discount=jnp.stack([item.discount for item in transitions]),
        terminated=jnp.stack([item.terminated for item in transitions]),
        truncated=jnp.stack([item.truncated for item in transitions]),
        next_observation=jnp.stack(
            [item.next_observation for item in transitions]
        ),
        next_decision_observation=jnp.stack(
            [item.next_decision_observation for item in transitions]
        ),
    )

    loop_state = initial
    loop_results = []
    for transition in transitions:
        result = agent.update_transition(loop_state, transition)
        loop_results.append(result)
        loop_state = result.state
    scanned = jax.jit(agent.scan_transitions)(initial, batched)

    chex.assert_trees_all_equal(
        scanned.transition_valid,
        jnp.asarray([True, False, True]),
    )
    chex.assert_trees_all_equal(
        _materialize_keys(loop_results[1].state),
        _materialize_keys(after_first),
    )
    chex.assert_trees_all_close(
        _materialize_keys(scanned.state),
        _materialize_keys(loop_state),
        atol=1.0e-6,
    )


def test_generic_builder_checkpoint_roundtrip_restores_caches_and_state(
    tmp_path: Path,
) -> None:
    agent = _agent(_builder_configs()[2])
    state = agent.start(agent.init(jr.key(8)), jnp.asarray([0.2, -0.4]))
    state = agent.update_transition(
        state,
        _transition(state, jnp.asarray([0.7, 0.1])),
    ).state
    checkpoint = tmp_path / "prototype-v2"

    save_prototype_checkpoint(agent, state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)

    assert PROTOTYPE_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v3"
    assert restored_agent.to_config() == agent.to_config()
    chex.assert_trees_all_close(
        _materialize_keys(restored_state),
        _materialize_keys(state),
    )


def test_v2_checkpoint_roundtrips_pristine_state_and_rejects_corrupt_cache(
    tmp_path: Path,
) -> None:
    agent = _agent(_builder_configs()[2])
    initial = agent.init(jr.key(81))
    pristine_checkpoint = tmp_path / "prototype-v2-pristine"
    save_prototype_checkpoint(agent, initial, pristine_checkpoint)
    restored_agent, restored = load_prototype_checkpoint(pristine_checkpoint)
    chex.assert_trees_all_equal(
        _materialize_keys(restored),
        _materialize_keys(initial),
    )
    observation = jnp.asarray([0.1, -0.2], dtype=jnp.float32)
    chex.assert_trees_all_close(
        _materialize_keys(restored_agent.start(restored, observation)),
        _materialize_keys(agent.start(initial, observation)),
        atol=1.0e-6,
    )

    started = agent.start(agent.init(jr.key(82)), observation)
    corrupt_builder = started.state_builder_state.replace(
        hidden=started.state_builder_state.hidden + 1.0
    )
    corrupt = started.replace(state_builder_state=corrupt_builder)
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            agent,
            corrupt,
            tmp_path / "prototype-v2-corrupt",
        )

    poisoned_builder = initial.state_builder_state.replace(
        parameters=initial.state_builder_state.parameters.at[0].set(jnp.nan)
    )
    poisoned_pristine = initial.replace(state_builder_state=poisoned_builder)
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            agent,
            poisoned_pristine,
            tmp_path / "prototype-v2-nonfinite",
        )


@pytest.mark.parametrize("use_gru", [False, True])
def test_v1_checkpoint_requires_trust_then_migrates_raw_and_gru_exactly(
    tmp_path: Path,
    use_gru: bool,
) -> None:
    representation_dim = RAW_DIM + 2 if use_gru else RAW_DIM
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=representation_dim,
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    config = PrototypeAgentConfig(
        oak=oak,
        gru_perception=(
            GRUPerceptionConfig(observation_dim=RAW_DIM, hidden_dim=2)
            if use_gru
            else None
        ),
    )
    agent = PrototypeAgent(config)
    key = jr.key(25)
    state = agent.start(
        agent.init(
            key,
            lifecycle_id=jnp.asarray([11, 12], dtype=jnp.uint32),
        ),
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    )
    legacy_state = _PrototypeAgentStateV1(
        oak_state=state.oak_state,
        world_model_state=state.world_model_state,
        buffer_state=state.buffer_state,
        horde_state=state.horde_state,
        ia_state=state.ia_state,
        gru_state=state.gru_state,
        step_count=state.step_count,
    )
    payload = agent.to_config()
    checkpoint = tmp_path / f"prototype-v1-{use_gru}"
    save_checkpoint(
        legacy_state,
        checkpoint,
        metadata={
            "schema": "alberta.prototype_agent.v1",
            "agent_config": payload,
            "config_sha256": _prototype_config_digest(payload),
        },
    )

    with pytest.raises(ValueError, match="lifecycle is ambiguous"):
        load_prototype_checkpoint(checkpoint, template_key=key)

    restored_agent, restored = load_prototype_checkpoint(
        checkpoint,
        template_key=key,
        trust_v1_started=True,
    )
    # A caller-owned lifecycle was not persisted by v1. The trusted loader
    # derives a replacement session from template_key and arms generation 0.
    expected = state.replace(
        current_decision_id=agent.init(key).current_decision_id,
    )
    chex.assert_trees_all_close(
        _materialize_keys(restored),
        _materialize_keys(expected),
        atol=1.0e-6,
    )

    next_observation = jnp.asarray([-0.1, 0.75], dtype=jnp.float32)
    uninterrupted = agent.update_transition(
        expected,
        _transition(expected, next_observation),
    )
    resumed = restored_agent.update_transition(
        restored,
        _transition(restored, next_observation),
    )
    chex.assert_trees_all_close(
        _materialize_keys(resumed),
        _materialize_keys(uninterrupted),
        atol=1.0e-6,
    )
