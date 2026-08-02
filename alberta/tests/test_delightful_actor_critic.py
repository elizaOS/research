# mypy: disable-error-code="attr-defined,call-arg"
"""Contracts for the isolated stateful Delightful Policy Gradient core."""

from __future__ import annotations

import json

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
import alberta_framework.core.delightful_actor_critic as module
from alberta_framework.core.delight import (
    DelightfulPolicyGradientConfig,
    discrete_delightful_policy_gradient,
)
from alberta_framework.core.delightful_actor_critic import (
    DelightfulActorCriticAgent,
    DelightfulActorCriticConfig,
    DelightfulActorCriticState,
    DelightfulChannelAvailability,
    run_delightful_actor_critic_from_arrays,
)

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> DelightfulActorCriticConfig:
    values: dict[str, object] = {
        "observation_dim": 2,
        "n_actions": 2,
        "mode": "delightful_pg",
        "actor_step_size": 0.1,
        "critic_step_size": 0.2,
        "average_reward_step_size": 0.05,
        "critic_trace_lambda": 0.0,
        "policy_temperature": 1.0,
        "delight_temperature": 0.75,
        "diagnostics_epsilon": 1.0e-8,
        "max_input_magnitude": 100.0,
        "max_parameter_magnitude": 100.0,
        "max_update_component_magnitude": 10.0,
        "max_updates": 100,
    }
    values.update(overrides)
    return DelightfulActorCriticConfig(**values)  # type: ignore[arg-type]


def _availability(
    *,
    safety: bool = True,
    model: bool = True,
    representation: bool = True,
) -> DelightfulChannelAvailability:
    return DelightfulChannelAvailability(
        safety=jnp.asarray(safety, dtype=jnp.bool_),
        model=jnp.asarray(model, dtype=jnp.bool_),
        representation=jnp.asarray(representation, dtype=jnp.bool_),
    )


def _started(
    agent: DelightfulActorCriticAgent,
    *,
    key: int = 0,
    observation: tuple[float, float] = (1.0, 0.0),
) -> DelightfulActorCriticState:
    state = agent.init(jax.random.key(key))
    result = agent.start(state, jnp.asarray(observation, dtype=jnp.float32))
    assert bool(result.applied)
    return result.state


def _assert_byte_identical(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = jnp.asarray(lhs)
        rhs_array = jnp.asarray(rhs)
        if jax.dtypes.issubdtype(lhs_array.dtype, jax.dtypes.prng_key):
            lhs_array = jax.random.key_data(lhs_array)
        if jax.dtypes.issubdtype(rhs_array.dtype, jax.dtypes.prng_key):
            rhs_array = jax.random.key_data(rhs_array)
        lhs_host = np.asarray(jax.device_get(lhs_array))
        rhs_host = np.asarray(jax.device_get(rhs_array))
        assert lhs_host.dtype == rhs_host.dtype
        assert lhs_host.shape == rhs_host.shape
        assert lhs_host.tobytes() == rhs_host.tobytes()


def _assert_trees_all_close(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = jnp.asarray(lhs)
        rhs_array = jnp.asarray(rhs)
        if jax.dtypes.issubdtype(lhs_array.dtype, jax.dtypes.prng_key):
            lhs_array = jax.random.key_data(lhs_array)
        if jax.dtypes.issubdtype(rhs_array.dtype, jax.dtypes.prng_key):
            rhs_array = jax.random.key_data(rhs_array)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(lhs_array)),
            np.asarray(jax.device_get(rhs_array)),
            atol=1.0e-7,
            rtol=1.0e-7,
        )


def test_public_exports_resolve_to_new_core_module() -> None:
    for name in module.__all__:
        implementation = getattr(module, name)
        assert name in core.__all__
        assert name in alberta.__all__
        assert getattr(core, name) is implementation
        assert getattr(alberta, name) is implementation


def test_strict_config_roundtrip_and_actor_trace_is_fixed_at_zero() -> None:
    config = _config()
    payload = json.loads(json.dumps(config.to_config()))
    assert DelightfulActorCriticConfig.from_config(payload) == config
    agent = DelightfulActorCriticAgent(config)
    assert DelightfulActorCriticAgent.from_config(agent.to_config()).config == config

    malformed = dict(payload)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        DelightfulActorCriticConfig.from_config(malformed)
    with pytest.raises(ValueError, match="mode"):
        _config(mode="unknown")
    with pytest.raises(ValueError, match="actor_trace_lambda"):
        _config(actor_trace_lambda=0.5)
    with pytest.raises(ValueError, match="actor_trace_lambda"):
        _config(actor_trace_lambda=False)
    with pytest.raises(ValueError, match="delight_temperature"):
        _config(delight_temperature=0.0)
    with pytest.raises(ValueError, match="policy_temperature"):
        _config(policy_temperature=float("nan"))
    with pytest.raises(ValueError, match="diagnostics_epsilon"):
        _config(diagnostics_epsilon=0.0)
    with pytest.raises(ValueError, match="actor_step_size"):
        _config(actor_step_size=1.0e-100)
    with pytest.raises(ValueError, match="max_update_component_magnitude"):
        _config(max_update_component_magnitude=0.0)
    with pytest.raises(ValueError, match="max_updates"):
        _config(max_updates=True)

    conceptual_bound = 1.00000006
    canonical = _config(max_update_component_magnitude=conceptual_bound)
    expected_float32_bound = float(np.float32(conceptual_bound))
    assert canonical.max_update_component_magnitude == expected_float32_bound
    assert (
        DelightfulActorCriticAgent(canonical).resource_budget.max_update_component_magnitude
        == expected_float32_bound
    )


def test_start_is_reproducible_and_records_exact_on_policy_semantics() -> None:
    agent = DelightfulActorCriticAgent(_config())
    observation = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    state_a = agent.init(jax.random.key(2))
    state_b = agent.init(jax.random.key(2))
    result_a = agent.start(state_a, observation)
    result_b = agent.start(state_b, observation)

    chex.assert_trees_all_equal(result_a, result_b)
    assert bool(result_a.applied)
    sample = result_a.state.last_sample
    chex.assert_trees_all_equal(sample.target_policy, sample.behavior_policy)
    chex.assert_trees_all_equal(sample.target_probability, sample.behavior_probability)
    chex.assert_trees_all_equal(sample.target_log_probability, sample.behavior_log_probability)
    assert int(sample.action) == int(result_a.action)
    assert float(sample.target_probability) == pytest.approx(
        float(sample.target_policy[sample.action])
    )
    assert float(sample.target_log_probability) == pytest.approx(
        float(jnp.log(sample.target_probability))
    )
    assert bool(agent.state_valid(result_a.state))

    duplicate = agent.start(result_a.state, observation)
    assert not bool(duplicate.applied)
    _assert_byte_identical(duplicate.state, result_a.state)

    with pytest.raises(ValueError, match="typed Threefry"):
        agent.init(jax.random.PRNGKey(2))


def test_update_before_start_and_mismatched_behavior_target_fail_closed() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = agent.init(jax.random.key(3))
    before_start = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(before_start.diagnostics.sample_available)
    assert bool(before_start.diagnostics.rejected)
    _assert_byte_identical(before_start.state, state)

    started = _started(agent, key=3)
    dishonest_sample = started.last_sample.replace(
        behavior_policy=started.last_sample.behavior_policy.at[0].set(0.25)
    )
    dishonest = started.replace(last_sample=dishonest_sample)
    rejected = agent.update(
        dishonest,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(rejected.diagnostics.state_valid)
    assert not bool(rejected.diagnostics.applied)
    _assert_byte_identical(rejected.state, dishonest)

    delta = jnp.asarray([4.0e-7, -4.0e-7], dtype=jnp.float32)
    almost_target = started.last_sample.target_policy + delta
    almost_sample = started.last_sample.replace(
        target_policy=almost_target,
        behavior_policy=almost_target,
    )
    almost_on_policy = started.replace(last_sample=almost_sample)
    almost_result = agent.update(
        almost_on_policy,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(almost_result.diagnostics.state_valid)
    _assert_byte_identical(almost_result.state, almost_on_policy)


def test_paper_specific_dg_delight_drives_only_current_actor_sample() -> None:
    config = _config(critic_trace_lambda=0.8)
    agent = DelightfulActorCriticAgent(config)
    initial = agent.init(jax.random.key(4)).replace(
        actor_bias=jnp.log(jnp.asarray([0.8, 0.2], dtype=jnp.float32))
    )
    start = agent.start(initial, jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    assert bool(start.applied)
    old = start.state
    result = agent.update(
        old,
        jnp.asarray(2.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    diagnostics = result.diagnostics

    log_probability = old.last_sample.behavior_log_probability
    advantage = jnp.asarray(2.0, dtype=jnp.float32)
    surprisal = -log_probability
    paper_dg_delight = advantage * surprisal
    gate = jax.nn.sigmoid(paper_dg_delight / config.delight_temperature)
    one_hot = jax.nn.one_hot(old.last_sample.action, 2, dtype=jnp.float32)
    score_bias = one_hot - old.last_sample.target_policy
    expected_actor_weights = old.actor_weights + (
        config.actor_step_size * gate * advantage * score_bias[:, None] * old.last_observation
    )
    expected_actor_bias = old.actor_bias + config.actor_step_size * gate * advantage * score_bias

    assert float(diagnostics.selected_log_probability) == pytest.approx(float(log_probability))
    assert float(diagnostics.action_surprisal) == pytest.approx(float(surprisal))
    assert float(diagnostics.advantage) == pytest.approx(float(advantage))
    assert float(diagnostics.delight) == pytest.approx(float(paper_dg_delight))
    assert float(diagnostics.gate_weight) == pytest.approx(float(gate))
    chex.assert_trees_all_close(result.state.actor_weights, expected_actor_weights)
    chex.assert_trees_all_close(result.state.actor_bias, expected_actor_bias)
    chex.assert_trees_all_equal(result.state.critic_trace_weights, old.last_observation)
    # Actor lambda is zero even though the critic trace is enabled.
    assert not hasattr(result.state, "actor_trace_weights")


def test_ordinary_mode_is_matched_and_gate_never_touches_baseline_channels() -> None:
    ordinary = DelightfulActorCriticAgent(_config(mode="ordinary_pg"))
    delightful = DelightfulActorCriticAgent(_config(mode="delightful_pg"))
    ordinary_state = _started(ordinary, key=5)
    delightful_state = _started(delightful, key=5)
    chex.assert_trees_all_equal(ordinary_state, delightful_state)

    reward = jnp.asarray(-2.0, dtype=jnp.float32)
    next_observation = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    ordinary_result = ordinary.update(ordinary_state, reward, next_observation, _availability())
    delightful_result = delightful.update(
        delightful_state, reward, next_observation, _availability()
    )

    assert float(ordinary_result.diagnostics.gate_weight) == 1.0
    assert float(delightful_result.diagnostics.gate_weight) < 0.5
    chex.assert_trees_all_equal(
        ordinary_result.state.critic_weights,
        delightful_result.state.critic_weights,
    )
    chex.assert_trees_all_equal(
        ordinary_result.state.critic_bias,
        delightful_result.state.critic_bias,
    )
    chex.assert_trees_all_equal(
        ordinary_result.state.critic_trace_weights,
        delightful_result.state.critic_trace_weights,
    )
    chex.assert_trees_all_equal(
        ordinary_result.state.average_reward,
        delightful_result.state.average_reward,
    )
    chex.assert_trees_all_equal(
        jax.random.key_data(ordinary_result.state.rng_key),
        jax.random.key_data(delightful_result.state.rng_key),
    )
    chex.assert_trees_all_equal(
        ordinary_result.state.transition_count,
        delightful_result.state.transition_count,
    )
    for routing in (
        ordinary_result.diagnostics.routing,
        delightful_result.diagnostics.routing,
    ):
        assert float(routing.critic_weight) == 1.0
        assert float(routing.average_reward_weight) == 1.0
        assert float(routing.safety_weight) == 1.0
        assert float(routing.model_weight) == 1.0
        assert float(routing.representation_weight) == 1.0


def test_external_availability_is_explicit_and_independent_of_actor_gate() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = _started(agent, key=6)
    result = agent.update(
        state,
        jnp.asarray(-3.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(safety=True, model=False, representation=True),
    )
    routing = result.diagnostics.routing
    assert float(result.diagnostics.gate_weight) < 0.5
    assert bool(routing.safety_available)
    assert not bool(routing.model_available)
    assert bool(routing.representation_available)
    assert float(routing.safety_weight) == 1.0
    assert float(routing.model_weight) == 0.0
    assert float(routing.representation_weight) == 1.0

    with pytest.raises(ValueError, match="availability.safety"):
        agent.update(
            result.state,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            DelightfulChannelAvailability(
                safety=jnp.asarray(1, dtype=jnp.int32),
                model=jnp.asarray(True),
                representation=jnp.asarray(True),
            ),
        )


def test_nonfinite_corrupt_and_exhausted_transactions_are_atomic_noops() -> None:
    agent = DelightfulActorCriticAgent(_config(max_updates=1))
    state = _started(agent, key=7)
    nan_input = agent.update(
        state,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(nan_input.diagnostics.input_valid)
    _assert_byte_identical(nan_input.state, state)

    corrupt = state.replace(actor_weights=state.actor_weights.at[0, 0].set(jnp.nan))
    corrupt_result = agent.update(
        corrupt,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(corrupt_result.diagnostics.state_valid)
    _assert_byte_identical(corrupt_result.state, corrupt)

    corrupt_outputs = corrupt.replace(
        average_reward=jnp.asarray(jnp.nan, dtype=jnp.float32),
        last_sample=corrupt.last_sample.replace(
            target_policy=corrupt.last_sample.target_policy.at[0].set(jnp.nan),
            behavior_policy=corrupt.last_sample.behavior_policy.at[0].set(jnp.nan),
        ),
    )
    sanitized = agent.update(
        corrupt_outputs,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert int(sanitized.action) == -1
    assert bool(jnp.all(jnp.isfinite(sanitized.target_policy)))
    assert bool(jnp.all(jnp.isfinite(sanitized.behavior_policy)))
    assert float(sanitized.average_reward) == 0.0
    routing = sanitized.diagnostics.routing
    assert bool(routing.safety_available)
    assert float(routing.safety_weight) == 0.0
    assert float(routing.model_weight) == 0.0
    assert float(routing.representation_weight) == 0.0
    _assert_byte_identical(sanitized.state, corrupt_outputs)

    accepted = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    exhausted = agent.update(
        accepted.state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        _availability(),
    )
    assert bool(exhausted.diagnostics.state_valid)
    assert not bool(exhausted.diagnostics.capacity_available)
    assert bool(exhausted.diagnostics.rejected)
    _assert_byte_identical(exhausted.state, accepted.state)


def test_policy_falls_back_to_uniform_when_finite_inputs_overflow_logits() -> None:
    agent = DelightfulActorCriticAgent(
        _config(
            max_input_magnitude=1.0e10,
            max_parameter_magnitude=1.0e30,
        )
    )
    state = agent.init(jax.random.key(71)).replace(
        actor_weights=jnp.full((2, 2), 1.0e30, dtype=jnp.float32)
    )
    observation = jnp.full((2,), 1.0e10, dtype=jnp.float32)
    policy = agent.policy(state, observation)
    chex.assert_trees_all_equal(
        policy,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    start = agent.start(state, observation)
    assert not bool(start.applied)
    assert bool(jnp.all(jnp.isfinite(start.target_policy)))
    _assert_byte_identical(start.state, state)


def test_out_of_bound_candidate_and_counter_corruption_do_not_partially_commit() -> None:
    agent = DelightfulActorCriticAgent(
        _config(
            actor_step_size=10.0,
            critic_step_size=10.0,
            average_reward_step_size=10.0,
            max_parameter_magnitude=0.1,
            max_update_component_magnitude=10.0,
        )
    )
    state = _started(agent, key=8)
    rejected = agent.update(
        state,
        jnp.asarray(10.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(rejected.diagnostics.candidate_state_valid)
    assert not bool(rejected.diagnostics.applied)
    _assert_byte_identical(rejected.state, state)

    normal = DelightfulActorCriticAgent(_config())
    normal_state = _started(normal, key=9)
    corrupt_count = normal_state.replace(actor_update_count=jnp.asarray(1, dtype=jnp.int32))
    count_result = normal.update(
        corrupt_count,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(count_result.diagnostics.state_valid)
    _assert_byte_identical(count_result.state, corrupt_count)


@pytest.mark.parametrize(
    "step_size_field",
    ("actor_step_size", "critic_step_size", "average_reward_step_size"),
)
def test_raw_update_overflow_is_rejected_before_component_clipping(
    step_size_field: str,
) -> None:
    agent = DelightfulActorCriticAgent(_config(**{step_size_field: 3.0e38}))
    state = _started(agent, key=91, observation=(1.0, 1.0))
    result = agent.update(
        state,
        jnp.asarray(10.0, dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert not bool(result.diagnostics.signals_finite)
    assert bool(result.diagnostics.candidate_state_valid)
    assert not bool(result.diagnostics.applied)
    assert int(result.action) == -1
    _assert_byte_identical(result.state, state)


def test_counter_and_rng_accounting_advance_exactly_once_per_commit() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = _started(agent, key=10)
    old_key = state.rng_key
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    )
    assert bool(result.diagnostics.applied)
    assert int(result.state.transition_count) == 1
    assert int(result.state.actor_update_count) == 1
    assert int(result.state.critic_update_count) == 1
    assert int(result.state.average_reward_update_count) == 1
    assert not bool(
        jnp.array_equal(jax.random.key_data(old_key), jax.random.key_data(result.state.rng_key))
    )

    rejected = agent.update(
        result.state,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        _availability(),
    )
    _assert_byte_identical(rejected.state, result.state)


def test_static_shape_and_dtype_contracts_raise_before_dynamic_update() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = _started(agent, key=11)
    with pytest.raises(ValueError, match="reward"):
        agent.update(
            state,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            _availability(),
        )
    with pytest.raises(ValueError, match="next_observation"):
        agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            _availability(),
        )
    with pytest.raises(ValueError, match="next_observation"):
        agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            np.asarray([0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
            _availability(),
        )
    malformed_state = state.replace(critic_weights=jnp.zeros((3,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.critic_weights"):
        agent.update(
            malformed_state,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            _availability(),
        )


def test_eager_jit_and_scan_paths_are_identical() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = _started(agent, key=12)
    reward = jnp.asarray(1.0, dtype=jnp.float32)
    observation = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    availability = _availability()
    with jax.disable_jit():
        eager = agent.update(state, reward, observation, availability)
    compiled = agent.update(state, reward, observation, availability)
    _assert_trees_all_close(eager, compiled)

    rewards = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32)
    observations = jnp.asarray([[0.0, 1.0], [1.0, 1.0], [-0.5, 0.25]], dtype=jnp.float32)
    batch_availability = DelightfulChannelAvailability(
        safety=jnp.ones((3,), dtype=jnp.bool_),
        model=jnp.asarray([True, False, True], dtype=jnp.bool_),
        representation=jnp.ones((3,), dtype=jnp.bool_),
    )
    scanned = run_delightful_actor_critic_from_arrays(
        agent, state, rewards, observations, batch_availability
    )
    manual_state = state
    manual_results = []
    for index in range(3):
        item = agent.update(
            manual_state,
            rewards[index],
            observations[index],
            DelightfulChannelAvailability(
                safety=batch_availability.safety[index],
                model=batch_availability.model[index],
                representation=batch_availability.representation[index],
            ),
        )
        manual_state = item.state
        manual_results.append(item)
    _assert_trees_all_close(scanned.state, manual_state)
    chex.assert_trees_all_equal(
        scanned.actions,
        jnp.stack([item.action for item in manual_results]),
    )
    assert int(scanned.diagnostics.attempted_count) == 3
    assert int(scanned.diagnostics.accepted_count) == 3
    expected_ess = float(jnp.sum(scanned.gate_weights) ** 2 / jnp.sum(scanned.gate_weights**2))
    assert float(scanned.diagnostics.effective_sample_size) == pytest.approx(expected_ess)

    tiny_gate_reward = jnp.asarray([-16.6], dtype=jnp.float32)
    one_step = run_delightful_actor_critic_from_arrays(
        agent,
        state,
        tiny_gate_reward,
        observations[:1],
        DelightfulChannelAvailability(
            safety=batch_availability.safety[:1],
            model=batch_availability.model[:1],
            representation=batch_availability.representation[:1],
        ),
    )
    direct = agent.update(
        state,
        tiny_gate_reward[0],
        observations[0],
        DelightfulChannelAvailability(
            safety=batch_availability.safety[0],
            model=batch_availability.model[0],
            representation=batch_availability.representation[0],
        ),
    )
    assert float(one_step.gate_weights[0]) < 1.0e-5
    assert float(one_step.diagnostics.effective_sample_size) == pytest.approx(
        float(direct.diagnostics.effective_sample_size), rel=1.0e-6
    )


def test_checkpoint_config_and_state_roundtrip_are_exact_and_fail_closed() -> None:
    agent = DelightfulActorCriticAgent(_config())
    state = _started(agent, key=13)
    state = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _availability(),
    ).state
    payload = json.loads(json.dumps(agent.checkpoint_payload(state)))
    restored_agent, restored_state = DelightfulActorCriticAgent.from_checkpoint_payload(payload)
    assert restored_agent.config == agent.config
    chex.assert_trees_all_equal(restored_state, state)

    malformed = dict(payload)
    malformed["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        DelightfulActorCriticAgent.from_checkpoint_payload(malformed)
    nonfinite = json.loads(json.dumps(payload))
    nonfinite["state"]["actor_weights"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DelightfulActorCriticAgent.from_checkpoint_payload(nonfinite)
    boolean_number = json.loads(json.dumps(payload))
    boolean_number["state"]["actor_weights"][0][0] = False
    with pytest.raises(ValueError, match="JSON real numbers"):
        DelightfulActorCriticAgent.from_checkpoint_payload(boolean_number)
    numeric_string = json.loads(json.dumps(payload))
    numeric_string["state"]["actor_weights"][0][0] = "0.0"
    with pytest.raises(ValueError, match="JSON real numbers"):
        DelightfulActorCriticAgent.from_checkpoint_payload(numeric_string)
    dishonest = json.loads(json.dumps(payload))
    dishonest["state"]["last_sample"]["behavior_probability"] = 0.123
    with pytest.raises(ValueError, match="invalid"):
        DelightfulActorCriticAgent.from_checkpoint_payload(dishonest)


def test_resource_budget_exactly_matches_fixed_state_and_update_bounds() -> None:
    agent = DelightfulActorCriticAgent(_config(max_updates=17))
    state = agent.init(jax.random.key(14))
    budget = agent.resource_budget
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))
    expected_trainable = 2 * 2 + 2 + 2 + 1
    assert budget.trainable_float32_scalars == expected_trainable
    assert budget.state_nbytes == actual_nbytes
    assert budget.max_transitions == 17
    assert budget.max_actor_updates_per_transition == 1
    assert budget.max_critic_updates_per_transition == 1
    assert budget.max_average_reward_updates_per_transition == 1
    assert budget.actor_scalar_updates_per_transition == 6
    assert budget.critic_scalar_updates_per_transition == 3
    assert budget.average_reward_scalar_updates_per_transition == 1
    assert budget.max_update_component_magnitude == 10.0
    assert budget.max_actor_update_l2_norm == pytest.approx(10.0 * np.sqrt(6.0))
    assert budget.max_critic_update_l2_norm == pytest.approx(10.0 * np.sqrt(3.0))
    assert budget.max_average_reward_update_abs == 10.0
    assert budget.max_external_routes_per_transition == 3
    assert budget.scan_output_nbytes_per_transition == 8 * 2 + 25
    assert budget.batch_diagnostics_nbytes == 52
    assert budget.scan_result_nbytes(3) == actual_nbytes + 52 + 3 * (8 * 2 + 25)
    with pytest.raises(ValueError, match="num_steps"):
        budget.scan_result_nbytes(0)
    assert budget.replay_capacity == 0
    assert budget.to_dict()["state_nbytes"] == actual_nbytes


def test_heteroskedastic_gambling_contract_is_deterministic_calculation_only() -> None:
    """A lucky rare outcome receives a large gate; this is not a quality claim."""
    common_probability = jnp.asarray(0.9, dtype=jnp.float32)
    rare_probability = jnp.asarray(0.1, dtype=jnp.float32)
    log_probabilities = jnp.log(
        jnp.asarray(
            [
                common_probability,
                common_probability,
                common_probability,
                common_probability,
                rare_probability,
                rare_probability,
                rare_probability,
                rare_probability,
            ],
            dtype=jnp.float32,
        )
    )
    # Common outcomes have low variance and positive mean.  The rare gamble has
    # negative mean but one large lucky realization, the exact pathology the
    # contract must expose rather than interpret as evidence of improvement.
    advantages = jnp.asarray([1.0, 1.0, -0.5, -0.5, 12.0, -8.0, -8.0, -8.0], dtype=jnp.float32)
    result = discrete_delightful_policy_gradient(
        log_probabilities,
        advantages,
        DelightfulPolicyGradientConfig(mode="delightful_pg", temperature=1.0),
    )
    expected_surprisal = -log_probabilities
    expected_delight = advantages * expected_surprisal
    expected_gate = jax.nn.sigmoid(expected_delight)

    chex.assert_trees_all_close(result.action_surprisal, expected_surprisal)
    chex.assert_trees_all_close(result.delight, expected_delight)
    chex.assert_trees_all_close(result.sample_weights, expected_gate)
    assert float(jnp.mean(advantages[4:])) < 0.0
    assert float(result.sample_weights[4]) > 0.999
    assert float(jnp.max(result.sample_weights[5:])) < 1.0e-7
    assert float(result.actor_coefficients[4]) > 10.0
