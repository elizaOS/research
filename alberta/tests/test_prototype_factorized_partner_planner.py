# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Focused contracts for the defaults-off factorized Prototype sidecar."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterator
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
    PrototypeAgentState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    BASE_FALLBACK_SOURCE_BINDING_CLAIMED,
    CHECKPOINT_RESUME_CLAIMED,
    CONFIG_TOKEN_NBYTES,
    DEVELOPMENT_ONLY,
    POST_MEMORY_TRANSITION_BINDING_CLAIMED,
    REPLAY_CAPACITY,
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLD_CALIBRATION_CLAIMED,
    FactorizedPartnerPlannerAgentState,
    PrototypeFactorizedPartnerPlanner,
    PrototypeFactorizedPartnerPlannerConfig,
    PrototypeFactorizedPartnerPlannerState,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RAW_DIM = 8
REPRESENTATION_DIM = 12
N_ACTIONS = 2
MASKS = jnp.ones((2, N_ACTIONS), dtype=jnp.bool_)


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    """Keep ordinary cases eager; the one parity case deliberately compiles."""

    if request.node.name == "test_prepare_and_completed_transition_have_eager_jit_parity":
        yield
    else:
        with jax.disable_jit():
            yield


def _prototype() -> PrototypeAgent:
    builder = OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=REPRESENTATION_DIM - RAW_DIM,
        step_size=0.01,
        include_raw_observation=True,
    )
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=REPRESENTATION_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    return PrototypeAgent(PrototypeAgentConfig(oak=oak, state_builder=builder))


def _planner(
    prototype: PrototypeAgent,
    *,
    planning_enabled: bool = False,
    uniform_partner_belief: bool = False,
) -> PrototypeFactorizedPartnerPlanner:
    return PrototypeFactorizedPartnerPlanner(
        prototype,
        PrototypeFactorizedPartnerPlannerConfig(
            observation_dim=RAW_DIM,
            prototype_representation_dim=REPRESENTATION_DIM,
            n_actions=N_ACTIONS,
            planning_enabled=planning_enabled,
            uniform_partner_belief=uniform_partner_belief,
        ),
    )


def _started_pair(prototype: PrototypeAgent) -> tuple[PrototypeAgentState, PrototypeAgentState]:
    observations = (
        jnp.linspace(-0.4, 0.3, RAW_DIM, dtype=jnp.float32),
        jnp.linspace(0.35, -0.35, RAW_DIM, dtype=jnp.float32),
    )
    states = tuple(
        prototype.start(
            prototype.init(
                jr.key(100 + index),
                lifecycle_id=jnp.asarray((40 + index, 70 + index), dtype=jnp.uint32),
            ),
            observations[index],
        )
        for index in range(2)
    )
    return states[0], states[1]


def _force_action(
    prototype: PrototypeAgent,
    state: PrototypeAgentState,
    action: int,
) -> PrototypeAgentState:
    replacement = prototype.replace_cached_primitive_action(
        state,
        decision_id=state.current_decision_id,
        decision_observation=state.current_representation,
        proposed_action=jnp.asarray(action, dtype=jnp.int32),
        safety_action_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
    )
    assert bool(replacement.committed)
    return replacement.state


def _with_model_values(
    state: PrototypeFactorizedPartnerPlannerState,
    *,
    reward_cells: jax.Array,
    behavior_bias: jax.Array | None = None,
) -> PrototypeFactorizedPartnerPlannerState:
    cells = jnp.asarray(reward_cells, dtype=jnp.float32)
    if cells.shape != (N_ACTIONS, N_ACTIONS):
        raise ValueError("reward_cells has the wrong shape")

    def replace_agent(
        agent: FactorizedPartnerPlannerAgentState,
    ) -> FactorizedPartnerPlannerAgentState:
        grounded_bias = jnp.zeros_like(agent.grounded.bias).at[:, RAW_DIM].set(cells.reshape((-1,)))
        behavior = agent.behavior
        if behavior_bias is not None:
            behavior = behavior.replace(
                weights=jnp.zeros_like(behavior.weights),
                bias=jnp.asarray(behavior_bias, dtype=jnp.float32),
            )
        return agent.replace(
            behavior=behavior,
            grounded=agent.grounded.replace(
                weights=jnp.zeros_like(agent.grounded.weights),
                bias=grounded_bias,
            ),
        )

    return state.replace(
        agent_0=replace_agent(state.agent_0),
        agent_1=replace_agent(state.agent_1),
    )


def _transition(
    state: PrototypeAgentState,
    *,
    reward: float,
    next_observation: jax.Array,
    discount: float = 0.9,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=jnp.asarray(next_observation, dtype=jnp.float32),
        next_decision_observation=jnp.asarray(next_observation, dtype=jnp.float32),
    )


def _advance_pair(
    prototype: PrototypeAgent,
    source_0: PrototypeAgentState,
    source_1: PrototypeAgentState,
    rewards: jax.Array,
    next_observations: jax.Array,
    discount: float = 0.9,
) -> tuple[PrototypeAgentState, PrototypeAgentState]:
    updates = tuple(
        prototype.update_transition(
            source,
            _transition(
                source,
                reward=float(rewards[index]),
                next_observation=next_observations[index],
                discount=discount,
            ),
        )
        for index, source in enumerate((source_0, source_1))
    )
    assert all(bool(update.transition_diagnostics.valid) for update in updates)
    return updates[0].state, updates[1].state


def _materialize_keys(tree: object) -> object:
    def convert(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(leaf)
        return leaf

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    def exact_bits(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            if dtype == jnp.float32:
                return jax.lax.bitcast_convert_type(leaf, jnp.uint32)
            if dtype == jnp.float16:
                return jax.lax.bitcast_convert_type(leaf, jnp.uint16)
        return leaf

    chex.assert_trees_all_equal(
        jax.tree.map(exact_bits, _materialize_keys(left)),
        jax.tree.map(exact_bits, _materialize_keys(right)),
    )


def test_config_defaults_and_exact_resource_formulas_are_fail_closed() -> None:
    prototype = _prototype()
    planner = _planner(prototype)
    config = planner.config
    restored = PrototypeFactorizedPartnerPlannerConfig.from_config(config.to_config())
    state = planner.init(jr.key(1))
    resources = planner.resource_budget(state)

    assert restored == config
    assert not config.planning_enabled
    assert not config.uniform_partner_belief
    assert DEVELOPMENT_ONLY
    assert not SCIENTIFIC_PROMOTION_ALLOWED
    assert not CHECKPOINT_RESUME_CLAIMED
    assert not THRESHOLD_CALIBRATION_CLAIMED
    assert not POST_MEMORY_TRANSITION_BINDING_CLAIMED
    assert not BASE_FALLBACK_SOURCE_BINDING_CLAIMED
    assert config.to_config()["post_memory_transition_binding_claimed"] is False
    assert config.to_config()["base_fallback_source_binding_claimed"] is False
    invalid_binding_claim = config.to_config()
    invalid_binding_claim["base_fallback_source_binding_claimed"] = True
    with pytest.raises(ValueError, match="base/fallback source binding"):
        PrototypeFactorizedPartnerPlannerConfig.from_config(invalid_binding_claim)
    assert REPLAY_CAPACITY == 0
    assert resources.behavior_state_nbytes_per_agent == 104
    assert resources.grounded_state_nbytes_per_agent == 1_452
    assert resources.cache_nbytes_per_agent == 307
    assert resources.state_nbytes_per_agent == 1_863
    assert resources.config_token_nbytes == CONFIG_TOKEN_NBYTES == 32
    assert resources.pair_state_nbytes == 3_758
    assert resources.measured_pair_nbytes == 3_758
    assert resources.exact_tree_match
    assert resources.post_init_random_draws_per_event == 0

    standalone_work = planner.standalone_prepare_work_budget()
    assert standalone_work.operation == "standalone_prepare"
    assert standalone_work.behavior_probability_vector_evaluations == 2
    assert standalone_work.grounded_joint_cell_prediction_equivalents == 8
    assert standalone_work.expected_reward_marginalization_products == 8
    assert standalone_work.prototype_replacement_candidates == 2
    assert standalone_work.behavior_parameter_update_attempts == 0
    assert standalone_work.grounded_parameter_update_attempts == 0
    assert standalone_work.atomic_pair_commit_decisions == 1

    completed_work = planner.completed_transition_work_budget()
    assert completed_work.operation == "completed_transition"
    assert completed_work.cache_authentication_evaluations == 2
    assert completed_work.behavior_probability_vector_evaluations == 8
    assert completed_work.grounded_joint_cell_prediction_equivalents == 18
    assert completed_work.expected_reward_marginalization_products == 16
    assert completed_work.prototype_replacement_candidates == 2
    assert completed_work.behavior_parameter_update_attempts == 2
    assert completed_work.grounded_parameter_update_attempts == 2
    assert completed_work.atomic_pair_commit_decisions == 2
    assert completed_work.environment_transition_proposals == 0
    assert completed_work.replay_updates == 0
    assert completed_work.post_init_random_draws == 0

    enabled_state = _planner(prototype, planning_enabled=True).init(jr.key(1))
    assert not bool(jnp.array_equal(state.config_token, enabled_state.config_token))


def test_work_budget_matches_spied_authentication_and_inference_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind logical work constants to the eager calls that perform the work."""

    prototype = _prototype()
    sources = _started_pair(prototype)
    planner = _planner(prototype)
    prepared = planner.prepare_pair(planner.init(jr.key(29)), *sources, MASKS)
    next_observations = jnp.stack(
        (
            jnp.linspace(0.05, 0.4, RAW_DIM, dtype=jnp.float32),
            jnp.linspace(-0.4, -0.05, RAW_DIM, dtype=jnp.float32),
        )
    )
    rewards = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    post_0, post_1 = _advance_pair(
        prototype,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        rewards,
        next_observations,
    )
    executed_actions = jnp.stack(
        (
            prepared.prototype_agent_0.current_action,
            prepared.prototype_agent_1.current_action,
        )
    ).astype(jnp.int32)

    def function_tree(function: Any) -> ast.AST:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
        return ast.parse(source)

    def named_call_count(function: Any, name: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            for node in ast.walk(function_tree(function))
        )

    def reward_marginalization_count(function: Any) -> int:
        return sum(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.MatMult)
            and isinstance(node.left, ast.Name)
            and node.left.id == "reward_cells"
            and isinstance(node.right, ast.Name)
            and node.right.id == "applied"
            for node in ast.walk(function_tree(function))
        )

    planner_type = type(planner)
    behavior_type = type(planner.behavior_model)
    assert reward_marginalization_count(planner_type._cache_matches_agent) == 1
    assert reward_marginalization_count(planner_type._evaluate_agent) == 1
    assert named_call_count(behavior_type.predict_probabilities, "softmax") == 1
    assert named_call_count(behavior_type.update, "softmax") == 1

    counts = {
        "pair_authentications": 0,
        "cache_authentication_evaluations": 0,
        "behavior_probability_predictions": 0,
        "behavior_probability_updates": 0,
        "grounded_joint_cell_predictions": 0,
        "decision_evaluations": 0,
    }
    original_authenticate_pair = planner.authenticate_pair
    original_cache_matches = planner._cache_matches_agent
    original_evaluate_agent = planner._evaluate_agent
    original_behavior_prediction = planner.behavior_model.predict_probabilities
    original_behavior_update = planner.behavior_model.update
    original_grounded_prediction = planner.grounded_world_model._predict_unchecked

    def counted_authenticate_pair(*args: Any, **kwargs: Any) -> Any:
        counts["pair_authentications"] += 1
        return original_authenticate_pair(*args, **kwargs)

    def counted_cache_matches(*args: Any, **kwargs: Any) -> Any:
        counts["cache_authentication_evaluations"] += 1
        return original_cache_matches(*args, **kwargs)

    def counted_evaluate_agent(*args: Any, **kwargs: Any) -> Any:
        counts["decision_evaluations"] += 1
        return original_evaluate_agent(*args, **kwargs)

    def counted_behavior_prediction(*args: Any, **kwargs: Any) -> Any:
        counts["behavior_probability_predictions"] += 1
        return original_behavior_prediction(*args, **kwargs)

    def counted_behavior_update(*args: Any, **kwargs: Any) -> Any:
        counts["behavior_probability_updates"] += 1
        return original_behavior_update(*args, **kwargs)

    def counted_grounded_prediction(*args: Any, **kwargs: Any) -> Any:
        counts["grounded_joint_cell_predictions"] += 1
        return original_grounded_prediction(*args, **kwargs)

    monkeypatch.setattr(planner, "authenticate_pair", counted_authenticate_pair)
    monkeypatch.setattr(planner, "_cache_matches_agent", counted_cache_matches)
    monkeypatch.setattr(planner, "_evaluate_agent", counted_evaluate_agent)
    monkeypatch.setattr(
        planner.behavior_model,
        "predict_probabilities",
        counted_behavior_prediction,
    )
    monkeypatch.setattr(planner.behavior_model, "update", counted_behavior_update)
    monkeypatch.setattr(
        planner.grounded_world_model,
        "_predict_unchecked",
        counted_grounded_prediction,
    )

    authenticated = planner.authenticate_pair(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
    )
    chex.assert_trees_all_equal(authenticated, jnp.ones((2,), dtype=jnp.bool_))
    assert counts == {
        "pair_authentications": 1,
        "cache_authentication_evaluations": 2,
        "behavior_probability_predictions": 2,
        "behavior_probability_updates": 0,
        "grounded_joint_cell_predictions": 8,
        "decision_evaluations": 0,
    }
    assert (
        counts["cache_authentication_evaluations"]
        + counts["decision_evaluations"]
    ) * N_ACTIONS**2 == 8

    for name in counts:
        counts[name] = 0
    completed = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        post_0,
        post_1,
        executed_actions,
        rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )
    assert bool(completed.diagnostics.transaction_committed)

    budget = planner.completed_transition_work_budget()
    assert counts["pair_authentications"] == 1
    assert (
        counts["cache_authentication_evaluations"]
        == budget.cache_authentication_evaluations
        == 2
    )
    assert (
        counts["behavior_probability_predictions"]
        + counts["behavior_probability_updates"]
        == budget.behavior_probability_vector_evaluations
        == 8
    )
    assert counts["behavior_probability_predictions"] == 6
    assert (
        counts["behavior_probability_updates"]
        == budget.behavior_parameter_update_attempts
        == 2
    )
    assert (
        counts["grounded_joint_cell_predictions"]
        == budget.grounded_joint_cell_prediction_equivalents
        == 18
    )
    assert counts["decision_evaluations"] == 2
    assert (
        counts["cache_authentication_evaluations"]
        + counts["decision_evaluations"]
    ) * N_ACTIONS**2 == budget.expected_reward_marginalization_products == 16


def test_one_causal_partner_belief_scores_every_joint_reward_cell() -> None:
    prototype = _prototype()
    sources = _started_pair(prototype)
    reward_cells = jnp.asarray(((1.0, 3.0), (2.0, 4.0)), dtype=jnp.float32)
    behavior_bias = jnp.asarray((1.5, -0.5), dtype=jnp.float32)

    learned_planner = _planner(prototype)
    learned_state = _with_model_values(
        learned_planner.init(jr.key(2)),
        reward_cells=reward_cells,
        behavior_bias=behavior_bias,
    )
    learned = learned_planner.prepare_pair(learned_state, *sources, MASKS)

    assert bool(learned.diagnostics.pair_committed)
    chex.assert_trees_all_equal(
        learned.diagnostics.learned_partner_probabilities,
        learned.diagnostics.applied_partner_probabilities,
    )
    for agent_index in range(2):
        chex.assert_trees_all_equal(
            learned.diagnostics.world_reward_cells[agent_index],
            reward_cells,
        )
        # There is exactly one belief vector per agent. Both candidate-own-action
        # rows are scored against that same simultaneous partner distribution.
        chex.assert_trees_all_close(
            learned.diagnostics.expected_rewards[agent_index],
            reward_cells @ learned.diagnostics.applied_partner_probabilities[agent_index],
        )

    uniform_planner = _planner(prototype, uniform_partner_belief=True)
    uniform_state = _with_model_values(
        uniform_planner.init(jr.key(2)),
        reward_cells=reward_cells,
        behavior_bias=behavior_bias,
    )
    uniform = uniform_planner.prepare_pair(uniform_state, *sources, MASKS)
    assert not bool(
        jnp.array_equal(
            uniform.diagnostics.learned_partner_probabilities,
            uniform.diagnostics.applied_partner_probabilities,
        )
    )
    chex.assert_trees_all_close(
        uniform.diagnostics.applied_partner_probabilities,
        jnp.full((2, N_ACTIONS), 0.5, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        uniform.diagnostics.expected_rewards,
        jnp.broadcast_to(reward_cells @ jnp.asarray((0.5, 0.5)), (2, N_ACTIONS)),
    )
    uniform_behavior_tamper = uniform.state.replace(
        agent_0=uniform.state.agent_0.replace(
            behavior=uniform.state.agent_0.behavior.replace(
                bias=uniform.state.agent_0.behavior.bias.at[0].add(0.25)
            )
        )
    )
    chex.assert_trees_all_equal(
        uniform_planner.authenticate_pair(
            uniform_behavior_tamper,
            uniform.prototype_agent_0,
            uniform.prototype_agent_1,
        ),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    tied_state = _with_model_values(
        learned_planner.init(jr.key(3)),
        reward_cells=jnp.zeros((N_ACTIONS, N_ACTIONS), dtype=jnp.float32),
    )
    tied = learned_planner.prepare_pair(tied_state, *sources, MASKS)
    chex.assert_trees_all_equal(
        tied.diagnostics.proposed_actions,
        jnp.zeros((2,), dtype=jnp.int32),
    )


def test_planner_selection_uses_public_replacement_and_synchronizes_stomp() -> None:
    prototype = _prototype()
    initial_sources = _started_pair(prototype)
    sources = tuple(_force_action(prototype, state, 0) for state in initial_sources)
    reward_cells = jnp.asarray(((0.0, 0.0), (5.0, 5.0)), dtype=jnp.float32)

    enabled_planner = _planner(prototype, planning_enabled=True)
    enabled_state = _with_model_values(
        enabled_planner.init(jr.key(4)),
        reward_cells=reward_cells,
    )
    enabled = enabled_planner.prepare_pair(enabled_state, *sources, MASKS)

    disabled_planner = _planner(prototype, planning_enabled=False)
    disabled_state = _with_model_values(
        disabled_planner.init(jr.key(4)),
        reward_cells=reward_cells,
    )
    disabled = disabled_planner.prepare_pair(disabled_state, *sources, MASKS)

    chex.assert_trees_all_equal(
        enabled.diagnostics.proposed_actions,
        disabled.diagnostics.proposed_actions,
    )
    chex.assert_trees_all_equal(
        enabled.diagnostics.world_raw_predictions,
        disabled.diagnostics.world_raw_predictions,
    )
    chex.assert_trees_all_equal(
        enabled.diagnostics.proposed_actions,
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        enabled.diagnostics.base_actions,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        enabled.diagnostics.effective_actions,
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        disabled.diagnostics.effective_actions,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    for selected in (enabled.prototype_agent_0, enabled.prototype_agent_1):
        assert int(selected.current_action) == 1
        assert int(selected.oak_state.stomp_state.last_primitive_action) == 1
        assert bool(prototype.validate_state(selected))
    for source, selected in zip(sources, (enabled.prototype_agent_0, enabled.prototype_agent_1)):
        chex.assert_trees_all_equal(
            jr.key_data(source.oak_state.stomp_state.rng_key),
            jr.key_data(selected.oak_state.stomp_state.rng_key),
        )
    for agent in (enabled.state.agent_0, enabled.state.agent_1):
        assert int(agent.cache.base_action) == 0
        assert int(agent.cache.base_action_guard) == -1
        assert int(agent.cache.effective_action) == 1

    fallback_masks = jnp.asarray(((True, False), (True, False)), dtype=jnp.bool_)
    fallback = enabled_planner.prepare_pair(
        enabled_state,
        *sources,
        fallback_masks,
    )
    assert bool(fallback.diagnostics.pair_committed)
    chex.assert_trees_all_equal(
        fallback.diagnostics.proposed_actions,
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        fallback.diagnostics.effective_actions,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        enabled_planner.authenticate_pair(
            fallback.state,
            fallback.prototype_agent_0,
            fallback.prototype_agent_1,
        ),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    unsafe_base_masks = jnp.asarray(((False, True), (False, True)), dtype=jnp.bool_)
    disabled_unsafe = disabled_planner.prepare_pair(
        disabled_state,
        *sources,
        unsafe_base_masks,
    )
    assert not bool(disabled_unsafe.diagnostics.pair_committed)
    _assert_tree_equal(disabled_unsafe.state, disabled_state)
    _assert_tree_equal(disabled_unsafe.prototype_agent_0, sources[0])
    _assert_tree_equal(disabled_unsafe.prototype_agent_1, sources[1])


def test_transition_updates_use_the_correct_per_agent_joint_orientation() -> None:
    prototype = _prototype()
    start_0, start_1 = _started_pair(prototype)
    source_0 = _force_action(prototype, start_0, 0)
    source_1 = _force_action(prototype, start_1, 1)
    planner = _planner(prototype)
    prepared = planner.prepare_pair(
        planner.init(jr.key(5)),
        source_0,
        source_1,
        MASKS,
    )
    next_observations = jnp.stack(
        (
            jnp.linspace(0.1, 0.8, RAW_DIM, dtype=jnp.float32),
            jnp.linspace(-0.8, -0.1, RAW_DIM, dtype=jnp.float32),
        )
    )
    rewards = jnp.asarray((0.75, -0.25), dtype=jnp.float32)
    post_0, post_1 = _advance_pair(
        prototype,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        rewards,
        next_observations,
    )
    result = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        post_0,
        post_1,
        jnp.asarray((0, 1), dtype=jnp.int32),
        rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )

    assert bool(result.diagnostics.transaction_committed)
    chex.assert_trees_all_equal(
        result.diagnostics.observed_partner_actions,
        jnp.asarray((1, 0), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.grounded_joint_action_indices,
        jnp.asarray((1, 2), dtype=jnp.int32),
    )
    expected_targets = jnp.concatenate(
        (
            next_observations,
            rewards[:, None],
            jnp.full((2, 1), 0.9, dtype=jnp.float32),
        ),
        axis=1,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.grounded_targets,
        expected_targets,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.behavior_update_applied,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.grounded_update_applied,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.candidate_generation_aligned,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert float(result.state.agent_0.behavior.bias[1]) > 0.0
    assert float(result.state.agent_0.behavior.bias[0]) < 0.0
    assert float(result.state.agent_1.behavior.bias[0]) > 0.0
    assert float(result.state.agent_1.behavior.bias[1]) < 0.0
    assert int(result.state.agent_0.grounded.update_count) == 1
    assert int(result.state.agent_1.grounded.update_count) == 1
    chex.assert_trees_all_equal(
        jr.key_data(result.state.agent_0.behavior.rng_key),
        jr.key_data(prepared.state.agent_0.behavior.rng_key),
    )
    chex.assert_trees_all_equal(
        jr.key_data(result.state.agent_1.behavior.rng_key),
        jr.key_data(prepared.state.agent_1.behavior.rng_key),
    )

    nll_limit = -jnp.log(
        jnp.asarray(planner.behavior_model.config.min_probability, dtype=jnp.float32)
    )
    invalid_positive_diagnostics = (
        result.state.replace(
            agent_0=result.state.agent_0.replace(
                behavior=result.state.agent_0.behavior.replace(
                    nll_ema=nll_limit + jnp.asarray(1.0, dtype=jnp.float32)
                )
            )
        ),
        result.state.replace(
            agent_0=result.state.agent_0.replace(
                behavior=result.state.agent_0.behavior.replace(
                    confidence_ema=jnp.asarray(0.25, dtype=jnp.float32)
                )
            )
        ),
    )
    for tampered in invalid_positive_diagnostics:
        rejected = planner.prepare_pair(
            tampered,
            result.prototype_agent_0,
            result.prototype_agent_1,
            MASKS,
        )
        assert not bool(rejected.diagnostics.pair_committed)
        _assert_tree_equal(rejected.state, tampered)


def test_consumed_and_unconsumed_planners_have_identical_preselection_learning() -> None:
    prototype = _prototype()
    starts = _started_pair(prototype)
    sources = tuple(_force_action(prototype, state, 0) for state in starts)
    reward_cells = jnp.asarray(((2.0, 2.0), (0.0, 0.0)), dtype=jnp.float32)
    planners = (
        _planner(prototype, planning_enabled=False),
        _planner(prototype, planning_enabled=True),
    )
    states = tuple(
        _with_model_values(planner.init(jr.key(6)), reward_cells=reward_cells)
        for planner in planners
    )
    prepared = tuple(
        planner.prepare_pair(state, *sources, MASKS) for planner, state in zip(planners, states)
    )
    chex.assert_trees_all_equal(
        prepared[0].diagnostics.proposed_actions,
        prepared[1].diagnostics.proposed_actions,
    )
    _assert_tree_equal(prepared[0].prototype_agent_0, prepared[1].prototype_agent_0)
    _assert_tree_equal(prepared[0].prototype_agent_1, prepared[1].prototype_agent_1)

    next_observations = jnp.stack(
        (
            jnp.full((RAW_DIM,), 0.2, dtype=jnp.float32),
            jnp.full((RAW_DIM,), -0.2, dtype=jnp.float32),
        )
    )
    rewards = jnp.asarray((0.4, 0.6), dtype=jnp.float32)
    post_0, post_1 = _advance_pair(
        prototype,
        prepared[0].prototype_agent_0,
        prepared[0].prototype_agent_1,
        rewards,
        next_observations,
    )
    results = tuple(
        planner.completed_transition(
            item.state,
            item.prototype_agent_0,
            item.prototype_agent_1,
            post_0,
            post_1,
            jnp.zeros((2,), dtype=jnp.int32),
            rewards,
            next_observations,
            jnp.asarray(0.9, dtype=jnp.float32),
            MASKS,
        )
        for planner, item in zip(planners, prepared)
    )
    assert all(bool(result.diagnostics.transaction_committed) for result in results)
    for agent_name in ("agent_0", "agent_1"):
        left = getattr(results[0].state, agent_name)
        right = getattr(results[1].state, agent_name)
        _assert_tree_equal(left.behavior, right.behavior)
        _assert_tree_equal(left.grounded, right.grounded)
    chex.assert_trees_all_equal(
        results[0].diagnostics.next_prepare.proposed_actions,
        results[1].diagnostics.next_prepare.proposed_actions,
    )
    chex.assert_trees_all_equal(
        results[0].diagnostics.next_prepare.world_raw_predictions,
        results[1].diagnostics.next_prepare.world_raw_predictions,
    )


def test_cache_source_and_config_token_tamper_are_recomputed() -> None:
    prototype = _prototype()
    starts = _started_pair(prototype)
    sources = tuple(_force_action(prototype, state, 0) for state in starts)
    planner = _planner(prototype, planning_enabled=True)
    state = _with_model_values(
        planner.init(jr.key(7)),
        reward_cells=jnp.asarray(((0.0, 0.0), (3.0, 3.0)), dtype=jnp.float32),
    )
    prepared = planner.prepare_pair(state, *sources, MASKS)
    prototypes = (prepared.prototype_agent_0, prepared.prototype_agent_1)
    chex.assert_trees_all_equal(
        planner.authenticate_pair(prepared.state, *prototypes),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    prediction_tamper = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            cache=prepared.state.agent_0.cache.replace(
                world_raw_predictions=(
                    prepared.state.agent_0.cache.world_raw_predictions.at[0, 0, RAW_DIM].add(1.0)
                )
            )
        )
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(prediction_tamper, *prototypes),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    diagnostic_tamper = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            behavior=prepared.state.agent_0.behavior.replace(
                nll_ema=prepared.state.agent_0.behavior.nll_ema + 0.25
            )
        )
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(diagnostic_tamper, *prototypes),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    zero_step_diagnostic_tampers = (
        prepared.state.replace(
            agent_0=prepared.state.agent_0.replace(
                behavior=prepared.state.agent_0.behavior.replace(
                    nll_ema=jnp.asarray(0.25, dtype=jnp.float32)
                )
            )
        ),
        prepared.state.replace(
            agent_0=prepared.state.agent_0.replace(
                behavior=prepared.state.agent_0.behavior.replace(
                    accuracy_ema=jnp.asarray(0.25, dtype=jnp.float32)
                )
            )
        ),
        prepared.state.replace(
            agent_0=prepared.state.agent_0.replace(
                behavior=prepared.state.agent_0.behavior.replace(
                    confidence_ema=jnp.asarray(0.25, dtype=jnp.float32)
                )
            )
        ),
    )
    for tampered in zero_step_diagnostic_tampers:
        rejected = planner.prepare_pair(tampered, *prototypes, MASKS)
        assert not bool(rejected.diagnostics.pair_committed)
        _assert_tree_equal(rejected.state, tampered)

    invalid_behavior_counter = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            behavior=prepared.state.agent_0.behavior.replace(
                step_count=jnp.asarray(1, dtype=jnp.int32)
            )
        )
    )
    rejected_behavior_counter = planner.prepare_pair(
        invalid_behavior_counter,
        *prototypes,
        MASKS,
    )
    assert not bool(rejected_behavior_counter.diagnostics.pair_committed)
    _assert_tree_equal(rejected_behavior_counter.state, invalid_behavior_counter)

    invalid_grounded_counter = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            grounded=prepared.state.agent_0.grounded.replace(
                update_count=jnp.asarray(1, dtype=jnp.int32)
            )
        )
    )
    rejected_grounded_counter = planner.prepare_pair(
        invalid_grounded_counter,
        *prototypes,
        MASKS,
    )
    assert not bool(rejected_grounded_counter.diagnostics.pair_committed)
    _assert_tree_equal(rejected_grounded_counter.state, invalid_grounded_counter)

    nan_diagnostic = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            behavior=prepared.state.agent_0.behavior.replace(
                nll_ema=jnp.asarray(jnp.nan, dtype=jnp.float32)
            )
        )
    )
    rejected_nan = planner.prepare_pair(nan_diagnostic, *prototypes, MASKS)
    assert not bool(rejected_nan.diagnostics.pair_committed)
    _assert_tree_equal(rejected_nan.state, nan_diagnostic)
    _assert_tree_equal(rejected_nan.prototype_agent_0, prototypes[0])
    _assert_tree_equal(rejected_nan.prototype_agent_1, prototypes[1])

    malformed_key = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            behavior=prepared.state.agent_0.behavior.replace(
                rng_key=jnp.zeros((2,), dtype=jnp.uint32)
            )
        )
    )
    with pytest.raises(TypeError, match="scalar typed PRNG key"):
        planner.authenticate_pair(malformed_key, *prototypes)

    base_tamper = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            cache=prepared.state.agent_0.cache.replace(base_action=jnp.asarray(1, dtype=jnp.int32))
        )
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(base_tamper, *prototypes),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    # The original pre-replacement base no longer has an independent source.
    # A coordinated receipt+guard rewrite therefore remains outside cache
    # identity when the exact effective action is the recomputed proposal.
    assert not BASE_FALLBACK_SOURCE_BINDING_CLAIMED
    coordinated_base_receipt_tamper = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            cache=prepared.state.agent_0.cache.replace(
                base_action=jnp.asarray(1, dtype=jnp.int32),
                base_action_guard=jnp.asarray(-2, dtype=jnp.int32),
            )
        )
    )
    assert int(prepared.state.agent_0.cache.effective_action) == 1
    assert int(prepared.diagnostics.proposed_actions[0]) == 1
    chex.assert_trees_all_equal(
        planner.authenticate_pair(coordinated_base_receipt_tamper, *prototypes),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    model_tamper = prepared.state.replace(
        agent_0=prepared.state.agent_0.replace(
            grounded=prepared.state.agent_0.grounded.replace(
                bias=prepared.state.agent_0.grounded.bias.at[0, RAW_DIM].add(0.5)
            )
        )
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(model_tamper, *prototypes),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    token_tamper = prepared.state.replace(
        config_token=prepared.state.config_token.at[0].set(
            prepared.state.config_token[0] ^ jnp.asarray(1, dtype=jnp.uint8)
        )
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(token_tamper, *prototypes),
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    rejected_token = planner.prepare_pair(token_tamper, *prototypes, MASKS)
    assert not bool(rejected_token.diagnostics.config_token_valid)
    assert not bool(rejected_token.diagnostics.pair_committed)
    _assert_tree_equal(rejected_token.state, token_tamper)
    _assert_tree_equal(rejected_token.prototype_agent_0, prototypes[0])
    _assert_tree_equal(rejected_token.prototype_agent_1, prototypes[1])

    source_tamper = prototypes[0].replace(
        current_raw_observation=prototypes[0].current_raw_observation.at[0].add(0.25)
    )
    chex.assert_trees_all_equal(
        planner.authenticate_pair(prepared.state, source_tamper, prototypes[1]),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )

    alternate_effective = prototype.replace_cached_primitive_action(
        prototypes[0],
        decision_id=prototypes[0].current_decision_id,
        decision_observation=prototypes[0].current_representation,
        proposed_action=jnp.asarray(0, dtype=jnp.int32),
        safety_action_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
    )
    assert bool(alternate_effective.committed)
    assert bool(prototype.validate_state(alternate_effective.state))
    chex.assert_trees_all_equal(
        planner.authenticate_pair(
            prepared.state,
            alternate_effective.state,
            prototypes[1],
        ),
        jnp.asarray((False, True), dtype=jnp.bool_),
    )


def test_nonfinite_completed_transition_rolls_back_every_paired_child() -> None:
    prototype = _prototype()
    sources = _started_pair(prototype)
    planner = _planner(prototype, planning_enabled=True)
    prepared = planner.prepare_pair(planner.init(jr.key(8)), *sources, MASKS)
    next_observations = jnp.stack(
        (
            jnp.full((RAW_DIM,), 0.3, dtype=jnp.float32),
            jnp.full((RAW_DIM,), -0.3, dtype=jnp.float32),
        )
    )
    finite_rewards = jnp.asarray((0.25, 0.5), dtype=jnp.float32)
    post_0, post_1 = _advance_pair(
        prototype,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        finite_rewards,
        next_observations,
    )
    actions = jnp.stack(
        (
            prepared.prototype_agent_0.current_action,
            prepared.prototype_agent_1.current_action,
        )
    )
    result = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        post_0,
        post_1,
        actions,
        jnp.asarray((jnp.nan, 0.5), dtype=jnp.float32),
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )

    assert not bool(result.diagnostics.candidate_valid)
    assert not bool(result.diagnostics.transaction_committed)
    _assert_tree_equal(result.state, prepared.state)
    _assert_tree_equal(result.prototype_agent_0, prepared.prototype_agent_0)
    _assert_tree_equal(result.prototype_agent_1, prepared.prototype_agent_1)

    wrong_actions = actions.at[0].set(jnp.asarray(1, dtype=jnp.int32) - actions[0])
    bounded_world_rejection_rewards = finite_rewards.at[0].set(
        jnp.asarray(1_001.0, dtype=jnp.float32)
    )
    rollback_cases = (
        (
            wrong_actions,
            finite_rewards,
            jnp.asarray(0.9, dtype=jnp.float32),
        ),
        (
            actions,
            finite_rewards,
            jnp.asarray(1.1, dtype=jnp.float32),
        ),
        (
            actions,
            bounded_world_rejection_rewards,
            jnp.asarray(0.9, dtype=jnp.float32),
        ),
    )
    for attempted_actions, attempted_rewards, attempted_discount in rollback_cases:
        rejected = planner.completed_transition(
            prepared.state,
            prepared.prototype_agent_0,
            prepared.prototype_agent_1,
            post_0,
            post_1,
            attempted_actions,
            attempted_rewards,
            next_observations,
            attempted_discount,
            MASKS,
        )
        assert not bool(rejected.diagnostics.transaction_committed)
        _assert_tree_equal(rejected.state, prepared.state)
        _assert_tree_equal(rejected.prototype_agent_0, prepared.prototype_agent_0)
        _assert_tree_equal(rejected.prototype_agent_1, prepared.prototype_agent_1)

    other_lifecycle_post_0 = post_0.replace(
        current_decision_id=post_0.current_decision_id.at[0].add(jnp.asarray(1, dtype=jnp.uint32))
    )
    assert bool(prototype.validate_state(other_lifecycle_post_0))
    wrong_successor = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        other_lifecycle_post_0,
        post_1,
        actions,
        finite_rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )
    chex.assert_trees_all_equal(
        wrong_successor.diagnostics.candidate_generation_aligned,
        jnp.asarray((False, True), dtype=jnp.bool_),
    )
    assert not bool(wrong_successor.diagnostics.transaction_committed)
    _assert_tree_equal(wrong_successor.state, prepared.state)
    _assert_tree_equal(
        wrong_successor.prototype_agent_0,
        prepared.prototype_agent_0,
    )
    _assert_tree_equal(
        wrong_successor.prototype_agent_1,
        prepared.prototype_agent_1,
    )

    wrong_clock_post_0 = post_0.replace(
        step_count=post_0.step_count + jnp.asarray(1, dtype=jnp.int32),
        step_words=post_0.step_words.at[1].add(jnp.asarray(1, dtype=jnp.uint32)),
    )
    wrong_clock = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        wrong_clock_post_0,
        post_1,
        actions,
        finite_rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )
    chex.assert_trees_all_equal(
        wrong_clock.diagnostics.candidate_clock_aligned,
        jnp.asarray((False, True), dtype=jnp.bool_),
    )
    assert not bool(wrong_clock.diagnostics.transaction_committed)
    _assert_tree_equal(wrong_clock.state, prepared.state)

    wrong_raw_post_0 = post_0.replace(
        current_raw_observation=post_0.current_raw_observation.at[0].add(0.125)
    )
    wrong_raw = planner.completed_transition(
        prepared.state,
        prepared.prototype_agent_0,
        prepared.prototype_agent_1,
        wrong_raw_post_0,
        post_1,
        actions,
        finite_rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )
    chex.assert_trees_all_equal(
        wrong_raw.diagnostics.next_observations_match,
        jnp.asarray((False, True), dtype=jnp.bool_),
    )
    assert not bool(wrong_raw.diagnostics.transaction_committed)
    _assert_tree_equal(wrong_raw.state, prepared.state)


def test_prepare_and_completed_transition_have_eager_jit_parity() -> None:
    prototype = _prototype()
    sources = _started_pair(prototype)
    planner = _planner(prototype, planning_enabled=True)
    state = _with_model_values(
        planner.init(jr.key(9)),
        reward_cells=jnp.asarray(((0.0, 1.0), (2.0, 3.0)), dtype=jnp.float32),
        behavior_bias=jnp.asarray((0.4, -0.2), dtype=jnp.float32),
    )
    with jax.disable_jit():
        eager_prepare = planner.prepare_pair(state, *sources, MASKS)
    compiled_prepare = planner.prepare_pair(state, *sources, MASKS)
    chex.assert_trees_all_close(
        _materialize_keys(eager_prepare),
        _materialize_keys(compiled_prepare),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    next_observations = jnp.stack(
        (
            jnp.linspace(-0.1, 0.6, RAW_DIM, dtype=jnp.float32),
            jnp.linspace(0.6, -0.1, RAW_DIM, dtype=jnp.float32),
        )
    )
    rewards = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    with jax.disable_jit():
        post_0, post_1 = _advance_pair(
            prototype,
            eager_prepare.prototype_agent_0,
            eager_prepare.prototype_agent_1,
            rewards,
            next_observations,
        )
        actions = jnp.stack(
            (
                eager_prepare.prototype_agent_0.current_action,
                eager_prepare.prototype_agent_1.current_action,
            )
        )
        eager_result = planner.completed_transition(
            eager_prepare.state,
            eager_prepare.prototype_agent_0,
            eager_prepare.prototype_agent_1,
            post_0,
            post_1,
            actions,
            rewards,
            next_observations,
            jnp.asarray(0.9, dtype=jnp.float32),
            MASKS,
        )
    compiled_result = planner.completed_transition(
        eager_prepare.state,
        eager_prepare.prototype_agent_0,
        eager_prepare.prototype_agent_1,
        post_0,
        post_1,
        actions,
        rewards,
        next_observations,
        jnp.asarray(0.9, dtype=jnp.float32),
        MASKS,
    )
    chex.assert_trees_all_close(
        _materialize_keys(eager_result),
        _materialize_keys(compiled_result),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
