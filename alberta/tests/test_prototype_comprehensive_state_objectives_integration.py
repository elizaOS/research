"""Boundary, compiled scan, and checkpoint contracts for the WP3 adapter."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.core.prototype_comprehensive_state_objectives as pco_module
from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_comprehensive_state_objectives import (
    PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RTU_MAX_TRANSITIONS,
    PrototypeComprehensiveObjectivesScanResult,
    PrototypeComprehensiveObjectivesState,
    PrototypeComprehensiveStateObjectives,
    PrototypeComprehensiveTargetReceipt,
    load_prototype_comprehensive_objectives_checkpoint,
    run_prototype_comprehensive_objectives_scan,
    save_prototype_comprehensive_objectives_checkpoint,
)
from alberta_framework.core.rtu_generate_and_test import (
    RTUGenerateAndTest,
    RTUGenerateAndTestConfig,
)
from alberta_framework.core.state_builder import (
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
    RecurrentTraceUnitStateBuilder,
    RecurrentTraceUnitStateBuilderConfig,
    RecurrentTraceUnitStateBuilderState,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = pytest.mark.integration

RAW_DIM = 2
FEATURE_DIM = 3
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


def _adapter(*, full_gru: bool = False) -> PrototypeComprehensiveStateObjectives:
    builder = (
        LearnableGRUStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=1,
            include_raw_observation=True,
            step_size=0.04,
            gradient_clip=8.0,
            initialization_scale=0.1,
        )
        if full_gru
        else OnlineGatedStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=1,
            include_raw_observation=True,
            step_size=0.04,
            gradient_clip=8.0,
            initialization_scale=0.1,
        )
    )
    prototype = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=FEATURE_DIM,
                    n_primitive_actions=N_ACTIONS,
                    base_hidden_sizes=(),
                    base_step_size=0.02,
                    option_step_size=0.02,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=builder,
        )
    )
    objectives = ComprehensiveStateObjectives(
        ComprehensiveStateObjectivesConfig(
            representation_dim=FEATURE_DIM,
            observation_target_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            gvf_discounts=(0.2, 0.7, 0.95),
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )
    return PrototypeComprehensiveStateObjectives(prototype, objectives)


def _rtu_adapter(
    *,
    with_generate_and_test: bool = False,
    replacement_interval: int = 100,
    hidden_dim: int = 1,
    utility_decay: float = 0.99,
) -> PrototypeComprehensiveStateObjectives:
    feature_dim = RAW_DIM + 2 * hidden_dim
    builder = RecurrentTraceUnitStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=hidden_dim,
        include_raw_observation=True,
        step_size=0.04,
        gradient_clip=8.0,
        r_min=0.2,
        r_max=0.95,
    )
    prototype = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=feature_dim,
                    n_primitive_actions=N_ACTIONS,
                    base_hidden_sizes=(),
                    base_step_size=0.02,
                    option_step_size=0.02,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=builder,
        )
    )
    objectives = ComprehensiveStateObjectives(
        ComprehensiveStateObjectivesConfig(
            representation_dim=feature_dim,
            observation_target_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            gvf_discounts=(0.2, 0.7, 0.95),
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )
    lifecycle = (
        RTUGenerateAndTest(
            RTUGenerateAndTestConfig(
                builder=builder,
                utility_decay=utility_decay,
                replacement_interval=replacement_interval,
                replacement_quota=1,
                minimum_age=(0 if replacement_interval == 1 else 100),
                minimum_support=(0 if replacement_interval == 1 else 1),
            )
        )
        if with_generate_and_test
        else None
    )
    return PrototypeComprehensiveStateObjectives(prototype, objectives, lifecycle)


def _transition(
    state: PrototypeComprehensiveObjectivesState,
    next_observation: jax.Array,
    *,
    reward: jax.Array,
    discount: jax.Array,
    terminated: jax.Array | None = None,
    truncated: jax.Array | None = None,
    next_decision_observation: jax.Array | None = None,
) -> PrototypeTransition:
    prototype = state.prototype_state
    terminated_value = jnp.asarray(False, dtype=jnp.bool_) if terminated is None else terminated
    truncated_value = jnp.asarray(False, dtype=jnp.bool_) if truncated is None else truncated
    return PrototypeTransition(  # type: ignore[call-arg]
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=prototype.current_decision_id,
        reward=reward,
        discount=discount,
        terminated=terminated_value,
        truncated=truncated_value,
        next_observation=next_observation,
        next_decision_observation=(
            next_observation if next_decision_observation is None else next_decision_observation
        ),
    )


def _target(
    adapter: PrototypeComprehensiveStateObjectives,
    state: PrototypeComprehensiveObjectivesState,
) -> PrototypeComprehensiveTargetReceipt:
    return adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.25, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.8, dtype=jnp.float32),
        control_value_target=jnp.asarray(0.3, dtype=jnp.float32),
        selected_action_advantage_target=jnp.asarray(-0.1, dtype=jnp.float32),
        source_revision_words=jnp.asarray([2, 5], dtype=jnp.uint32),
        provenance_words=jnp.asarray([23, 29, 31, 37], dtype=jnp.uint32),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_allclose(left: object, right: object) -> None:
    left = _materialize_keys(left)
    right = _materialize_keys(right)
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(left_array, right_array, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_terminal_objectives_use_final_observation_before_autoreset_owner() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(20)),
        jnp.asarray([0.2, -0.4], dtype=jnp.float32),
    ).state
    final_observation = jnp.asarray([0.7, 0.1], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.8, 0.5], dtype=jnp.float32)
    source_builder = state.prototype_state.state_builder_state
    assert type(source_builder) is OnlineGatedStateBuilderState
    builder = adapter.builder
    assert type(builder) is OnlineGatedStateBuilder
    expected_bootstrap = builder.update_with_status(
        source_builder,
        final_observation,
        state.prototype_state.current_action,
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    action = int(state.prototype_state.current_action)
    observation_prediction = (
        state.objectives_state.observation_weights[action]
        @ state.prototype_state.current_representation
        + state.objectives_state.observation_bias[action]
    )
    expected_final_loss = 0.5 * jnp.mean(jnp.square(observation_prediction - final_observation))
    autoreset_loss = 0.5 * jnp.mean(jnp.square(observation_prediction - reset_observation))
    result = adapter.update_transition(
        state,
        _transition(
            state,
            final_observation,
            reward=jnp.asarray(0.4, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(True, dtype=jnp.bool_),
            next_decision_observation=reset_observation,
        ),
        _target(adapter, state),
    )
    assert bool(result.update_applied)
    np.testing.assert_allclose(
        result.bootstrap_representation,
        expected_bootstrap.representation,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.objective_update.observation_loss,
        expected_final_loss,
        rtol=1e-6,
        atol=1e-7,
    )
    assert not np.isclose(
        float(result.objective_update.observation_loss),
        float(autoreset_loss),
    )
    np.testing.assert_array_equal(
        result.objective_update.next_representation_revision_words,
        [0, 2],
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.observation_event_words,
        [0, 3],
    )
    np.testing.assert_array_equal(
        result.state.objectives_state.pending_representation_revision_words,
        [0, 3],
    )
    np.testing.assert_allclose(
        result.state.prototype_state.current_raw_observation,
        reset_observation,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.array_equal(
        np.asarray(result.bootstrap_representation),
        np.asarray(result.state.prototype_state.current_representation),
    )


def test_full_gru_rtrl_builder_commits_current_and_successor_gradients_once() -> None:
    adapter = _adapter(full_gru=True)
    assert type(adapter.prototype.state_builder) is LearnableGRUStateBuilder
    restored = PrototypeComprehensiveStateObjectives.from_config(adapter.to_config())
    assert type(restored.prototype.state_builder) is LearnableGRUStateBuilder
    assert restored.to_config() == adapter.to_config()
    state = adapter.start(
        adapter.init(jr.key(24)),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    ).state
    transition = _transition(
        state,
        jnp.asarray([-0.4, 0.7], dtype=jnp.float32),
        reward=jnp.asarray(0.35, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    result = adapter.update_transition(state, transition, _target(adapter, state))
    assert bool(result.update_applied)
    assert bool(result.builder_transaction_applied)
    assert float(jnp.linalg.norm(result.objective_update.current_representation_gradient)) > 0.0
    assert float(jnp.linalg.norm(result.objective_update.next_representation_gradient)) > 0.0

    source_builder = state.prototype_state.state_builder_state
    current_proposal = adapter.builder.propose_learning_update(
        source_builder,
        result.objective_update.current_representation_gradient,
    )
    successor_proposal = adapter.builder.propose_learning_update(
        result.bootstrap_builder_transition.state,
        result.objective_update.next_representation_gradient,
    )
    raw_combined = (
        current_proposal.raw_parameter_gradient + successor_proposal.raw_parameter_gradient
    )
    raw_norm = jnp.linalg.norm(raw_combined)
    clip_factor = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(adapter.builder.config.gradient_clip, dtype=jnp.float32)
        / jnp.maximum(raw_norm, jnp.asarray(1.0e-30, dtype=jnp.float32)),
    )
    expected_parameters = (
        result.bootstrap_builder_transition.state.parameters
        - jnp.asarray(adapter.builder.config.step_size, dtype=jnp.float32)
        * raw_combined
        * clip_factor
    )
    np.testing.assert_allclose(
        result.state.prototype_state.state_builder_state.parameters,
        expected_parameters,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(result.builder_learning.pre_update_words, [0, 0])
    np.testing.assert_array_equal(result.builder_learning.post_update_words, [0, 1])
    np.testing.assert_array_equal(result.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(result.state.target_receipt_words, [0, 1])


def test_rtu_builder_composes_with_both_comprehensive_gradient_sources() -> None:
    adapter = _rtu_adapter()
    assert type(adapter.prototype.state_builder) is RecurrentTraceUnitStateBuilder
    restored = PrototypeComprehensiveStateObjectives.from_config(adapter.to_config())
    assert type(restored.prototype.state_builder) is RecurrentTraceUnitStateBuilder
    assert restored.to_config() == adapter.to_config()
    state = adapter.start(
        adapter.init(jr.key(25)),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    ).state
    assert type(state.prototype_state.state_builder_state) is RecurrentTraceUnitStateBuilderState
    transition = _transition(
        state,
        jnp.asarray([-0.4, 0.7], dtype=jnp.float32),
        reward=jnp.asarray(0.35, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    result = adapter.update_transition(state, transition, _target(adapter, state))
    assert bool(result.update_applied)
    assert bool(result.builder_transaction_applied)
    assert bool(result.builder_sources_match)
    assert bool(result.builder_destination_matches)
    assert float(jnp.linalg.norm(result.objective_update.current_representation_gradient)) > 0.0
    assert float(jnp.linalg.norm(result.objective_update.next_representation_gradient)) > 0.0
    assert not np.array_equal(
        result.state.prototype_state.state_builder_state.parameters,
        state.prototype_state.state_builder_state.parameters,
    )
    np.testing.assert_array_equal(result.builder_learning.pre_update_words, [0, 0])
    np.testing.assert_array_equal(result.builder_learning.post_update_words, [0, 1])


def test_rtu_lifecycle_no_replacement_is_bit_exact_to_legacy_adapter() -> None:
    legacy = _rtu_adapter()
    integrated = _rtu_adapter(with_generate_and_test=True)
    observation = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    legacy_state = legacy.start(legacy.init(jr.key(251)), observation).state
    lifecycle = integrated.rtu_generate_and_test
    assert lifecycle is not None
    integrated_state = dataclasses.replace(  # type: ignore[type-var]
        legacy_state,
        rtu_generate_and_test_state=lifecycle.init(jr.key(252)),
    )
    assert bool(integrated.state_valid(integrated_state))
    next_observation = jnp.asarray([-0.4, 0.7], dtype=jnp.float32)
    legacy_result = legacy.update_transition(
        legacy_state,
        _transition(
            legacy_state,
            next_observation,
            reward=jnp.asarray(0.35, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        _target(legacy, legacy_state),
    )
    integrated_result = integrated.update_transition(
        integrated_state,
        _transition(
            integrated_state,
            next_observation,
            reward=jnp.asarray(0.35, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        _target(integrated, integrated_state),
    )
    assert bool(legacy_result.update_applied)
    assert bool(integrated_result.update_applied)
    assert bool(integrated_result.rtu_observation_proposal_valid)
    assert bool(integrated_result.rtu_observation_transaction_applied)
    assert bool(integrated_result.rtu_replacement_cache_safe)
    assert not bool(integrated_result.rtu_replacement_requires_pre_action_hook)
    assert integrated_result.rtu_generate_and_test is not None
    assert integrated_result.rtu_advance_receipt is not None
    assert int(integrated_result.rtu_advance_receipt.sequence_length) == 1
    chex.assert_trees_all_equal(
        _materialize_keys(integrated_result.state.prototype_state),
        _materialize_keys(legacy_result.state.prototype_state),
    )
    chex.assert_trees_all_equal(
        _materialize_keys(integrated_result.state.objectives_state),
        _materialize_keys(legacy_result.state.objectives_state),
    )
    chex.assert_trees_all_equal(
        _materialize_keys(integrated_result.rtu_generate_and_test.builder_state),
        _materialize_keys(integrated_result.state.prototype_state.state_builder_state),
    )
    rtu_state = integrated_result.state.rtu_generate_and_test_state
    assert rtu_state is not None
    np.testing.assert_array_equal(rtu_state.observation_words, [0, 1])
    assert bool(integrated.state_valid(integrated_result.state))


def test_rtu_lifecycle_boundary_replays_bootstrap_reset_restart_exactly() -> None:
    adapter = _rtu_adapter(with_generate_and_test=True)
    state = adapter.start(
        adapter.init(jr.key(253)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    ).state
    source_builder = state.prototype_state.state_builder_state
    final_observation = jnp.asarray([0.8, -0.5], dtype=jnp.float32)
    restart_observation = jnp.asarray([-0.7, 0.4], dtype=jnp.float32)
    result = adapter.update_transition(
        state,
        _transition(
            state,
            final_observation,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(True, dtype=jnp.bool_),
            next_decision_observation=restart_observation,
        ),
        _target(adapter, state),
    )
    assert bool(result.update_applied)
    assert result.rtu_advance_receipt is not None
    assert result.rtu_generate_and_test is not None
    assert int(result.rtu_advance_receipt.sequence_length) == 2
    builder = adapter.builder
    bootstrap = builder.update_with_status(
        source_builder,
        final_observation,
        state.prototype_state.current_action,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    reset = builder.reset_episode(bootstrap.state)
    restart = builder.update_with_status(
        reset,
        restart_observation,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.rtu_generate_and_test.builder_state.step_words,
        restart.state.step_words,
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.step_words,
        restart.state.step_words,
    )
    assert bool(adapter.state_valid(result.state))


def test_live_rtu_replacement_occurs_after_truncated_restart_before_action() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    ).state
    final_observation = jnp.asarray([0.8, -0.5], dtype=jnp.float32)
    restart_observation = jnp.asarray([-0.7, 0.4], dtype=jnp.float32)
    result = adapter.update_transition(
        state,
        _transition(
            state,
            final_observation,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            discount=jnp.asarray(0.8, dtype=jnp.float32),
            truncated=jnp.asarray(True, dtype=jnp.bool_),
            next_decision_observation=restart_observation,
        ),
        _target(adapter, state),
    )
    assert bool(result.update_applied)
    assert result.rtu_generate_and_test is not None
    assert bool(jnp.any(result.rtu_generate_and_test.diagnostics.selected_mask))
    np.testing.assert_array_equal(
        result.state.prototype_state.current_raw_observation,
        restart_observation,
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.current_representation[RAW_DIM:],
        0.0,
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.step_words,
        [0, 3],
    )
    assert not np.array_equal(
        np.asarray(result.bootstrap_representation),
        np.asarray(result.state.prototype_state.current_representation),
    )


def test_rtu_replacement_uses_pre_action_hook_and_commits_live() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    result = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.4, -0.3], dtype=jnp.float32),
            reward=jnp.asarray(0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        _target(adapter, state),
    )
    assert result.rtu_generate_and_test is not None
    assert bool(result.rtu_generate_and_test.diagnostics.applied)
    assert bool(result.rtu_observation_proposal_valid)
    assert bool(result.rtu_lifecycle_source_matches)
    assert bool(jnp.any(result.rtu_generate_and_test.diagnostics.selected_mask))
    assert bool(result.rtu_replacement_cache_safe)
    assert not bool(result.rtu_replacement_requires_pre_action_hook)
    assert bool(result.rtu_observation_transaction_applied)
    assert bool(result.update_applied)
    assert bool(result.builder_learning.applied)
    np.testing.assert_array_equal(result.builder_learning.pre_update_words, [0, 0])
    np.testing.assert_array_equal(result.builder_learning.post_update_words, [0, 1])
    np.testing.assert_array_equal(
        result.rtu_generate_and_test.diagnostics.pre_builder_update_words,
        [0, 1],
    )
    np.testing.assert_array_equal(
        result.rtu_generate_and_test.diagnostics.post_builder_update_words,
        [0, 2],
    )
    assert not np.array_equal(
        np.asarray(
            result.state.prototype_state.state_builder_state.parameters
        ),
        np.asarray(state.prototype_state.state_builder_state.parameters),
    )
    np.testing.assert_array_equal(
        result.state.rtu_generate_and_test_state.replacement_event_words,
        [0, 1],
    )


def test_live_rtu_replacement_uses_internally_owned_frozen_head_deletion_loss() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(1255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    representation = state.prototype_state.current_representation
    real_axis = RAW_DIM
    real_value = representation[real_axis]
    assert float(real_value) != 0.0

    objectives = state.objectives_state
    zeroed = objectives.replace(
        observation_weights=jnp.zeros_like(objectives.observation_weights),
        observation_bias=jnp.zeros_like(objectives.observation_bias),
        latent_weights=jnp.zeros_like(objectives.latent_weights),
        latent_bias=jnp.zeros_like(objectives.latent_bias),
        reward_weights=jnp.zeros_like(objectives.reward_weights),
        reward_bias=jnp.zeros_like(objectives.reward_bias),
        termination_weights=jnp.zeros_like(objectives.termination_weights),
        termination_bias=jnp.zeros_like(objectives.termination_bias),
        gvf_weights=jnp.zeros_like(objectives.gvf_weights),
        value_weights=jnp.zeros_like(objectives.value_weights).at[real_axis].set(2.0),
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_weights=jnp.zeros_like(objectives.advantage_weights),
        advantage_bias=jnp.zeros_like(objectives.advantage_bias),
        inverse_current_weights=jnp.zeros_like(objectives.inverse_current_weights),
        inverse_next_weights=jnp.zeros_like(objectives.inverse_next_weights),
        inverse_bias=jnp.zeros_like(objectives.inverse_bias),
    )
    state = dataclasses.replace(state, objectives_state=zeroed)  # type: ignore[type-var]
    assert bool(adapter.state_valid(state))
    target = adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.0, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.0, dtype=jnp.float32),
        control_value_target=jnp.float32(2.0) * real_value,
        selected_action_advantage_target=jnp.asarray(0.0, dtype=jnp.float32),
        source_revision_words=jnp.asarray([2, 5], dtype=jnp.uint32),
        provenance_words=jnp.asarray([23, 29, 31, 37], dtype=jnp.uint32),
    )
    result = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.4, -0.3], dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        target,
    )

    assert bool(result.update_applied)
    lifecycle_result = result.rtu_generate_and_test
    assert lifecycle_result is not None
    diagnostics = lifecycle_result.diagnostics
    assert bool(diagnostics.causal_deletion_evidence_available)
    assert bool(diagnostics.causal_evidence_required)
    expected_change = (
        jnp.float32(adapter.objectives.config.control_group_weight)
        * jnp.square(real_value)
    )
    np.testing.assert_allclose(
        diagnostics.causal_deletion_loss_change,
        jnp.asarray([expected_change], dtype=jnp.float32),
        rtol=2e-6,
        atol=1e-7,
    )
    assert bool(jnp.any(diagnostics.selected_mask))


def test_live_internal_causal_rank_overrides_opposite_proxy_rank_and_updates_head() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
        hidden_dim=2,
        utility_decay=0.0,
    )
    state = adapter.start(
        adapter.init(jr.key(2255)),
        jnp.asarray([0.35, -0.2], dtype=jnp.float32),
    ).state
    representation = state.prototype_state.current_representation
    real_axes = jnp.asarray([RAW_DIM, RAW_DIM + 1], dtype=jnp.int32)
    real_values = representation[real_axes]
    assert bool(jnp.all(jnp.abs(real_values) > jnp.float32(1.0e-5)))

    # Fix contributions at c0=+1 and c1=-0.1.  With factual value error +1,
    # activation-gradient proxy utility ranks unit 1 lower, while deleting unit
    # 0 removes the error and gives it the lower causal-deletion utility.
    value_weights = jnp.zeros_like(state.objectives_state.value_weights)
    value_weights = value_weights.at[real_axes[0]].set(
        jnp.float32(1.0) / real_values[0]
    )
    value_weights = value_weights.at[real_axes[1]].set(
        jnp.float32(-0.1) / real_values[1]
    )
    objectives = state.objectives_state
    controlled = objectives.replace(
        observation_weights=jnp.zeros_like(objectives.observation_weights),
        observation_bias=jnp.zeros_like(objectives.observation_bias),
        latent_weights=jnp.zeros_like(objectives.latent_weights),
        latent_bias=jnp.zeros_like(objectives.latent_bias),
        reward_weights=jnp.zeros_like(objectives.reward_weights),
        reward_bias=jnp.zeros_like(objectives.reward_bias),
        termination_weights=jnp.zeros_like(objectives.termination_weights),
        termination_bias=jnp.zeros_like(objectives.termination_bias),
        gvf_weights=jnp.zeros_like(objectives.gvf_weights),
        value_weights=value_weights,
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_weights=jnp.zeros_like(objectives.advantage_weights),
        advantage_bias=jnp.zeros_like(objectives.advantage_bias),
        inverse_current_weights=jnp.zeros_like(objectives.inverse_current_weights),
        inverse_next_weights=jnp.zeros_like(objectives.inverse_next_weights),
        inverse_bias=jnp.zeros_like(objectives.inverse_bias),
    )
    state = dataclasses.replace(state, objectives_state=controlled)  # type: ignore[type-var]
    next_observation = jnp.asarray([0.4, -0.3], dtype=jnp.float32)
    control_value_target = jnp.asarray(-0.1, dtype=jnp.float32)
    target = adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.0, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.0, dtype=jnp.float32),
        control_value_target=control_value_target,
        selected_action_advantage_target=jnp.asarray(0.0, dtype=jnp.float32),
        source_revision_words=jnp.asarray([2, 5], dtype=jnp.uint32),
        provenance_words=jnp.asarray([23, 29, 31, 37], dtype=jnp.uint32),
    )
    result = adapter.update_transition(
        state,
        _transition(
            state,
            next_observation,
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        target,
    )

    assert bool(result.update_applied)
    assert result.rtu_generate_and_test is not None
    diagnostics = result.rtu_generate_and_test.diagnostics
    assert float(diagnostics.effective_contribution[0]) > float(
        diagnostics.effective_contribution[1]
    )
    assert float(diagnostics.causal_deletion_loss_change[0]) < float(
        diagnostics.causal_deletion_loss_change[1]
    )
    expected_pre_update_change = (
        jnp.float32(adapter.objectives.config.control_group_weight)
        * jnp.float32(0.25)
        * jnp.asarray([-1.0, 0.21], dtype=jnp.float32)
    )
    np.testing.assert_allclose(
        diagnostics.causal_deletion_loss_change,
        expected_pre_update_change,
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(diagnostics.selected_mask, [True, False])
    assert not np.array_equal(
        np.asarray(result.objective_update.state.value_weights),
        np.asarray(controlled.value_weights),
    )

    # A post-SGD recomputation is deliberately different: the authoritative
    # score above was produced from the frozen source heads, before this real
    # transition updated any objective parameter.
    post_cache = adapter.objectives.cache_action(
        result.objective_update.state,
        representation,
        state.prototype_state.current_action,
        state.prototype_state.observation_event_words,
    )
    assert bool(post_cache.cache_applied)

    def evaluate_post_update_head(current_representation: jax.Array) -> jax.Array:
        counterfactual_state = post_cache.state.replace(
            pending_representation=current_representation
        )
        counterfactual_receipt = post_cache.receipt.replace(
            representation=current_representation
        )
        update = adapter.objectives.update(
            counterfactual_state,
            counterfactual_receipt,
            result.bootstrap_representation,
            result.objective_update.next_representation_revision_words,
            next_observation,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            control_value_target,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert bool(update.update_applied)
        return update.balanced_loss

    post_factual_loss = evaluate_post_update_head(representation)
    post_update_change = jnp.stack(
        tuple(
            evaluate_post_update_head(
                representation.at[RAW_DIM + unit_index]
                .set(jnp.float32(0.0))
                .at[RAW_DIM + 2 + unit_index]
                .set(jnp.float32(0.0))
            )
            - post_factual_loss
            for unit_index in range(2)
        )
    )
    assert float(
        jnp.max(jnp.abs(post_update_change - diagnostics.causal_deletion_loss_change))
    ) > 1.0e-5


def test_invalid_internal_causal_deletion_rejects_the_outer_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(with_generate_and_test=True, replacement_interval=1)
    state = adapter.start(
        adapter.init(jr.key(3255)),
        jnp.asarray([0.3, -0.15], dtype=jnp.float32),
    ).state
    assert float(state.prototype_state.current_representation[RAW_DIM]) != 0.0
    original_update = adapter.objectives._update_jit

    def invalidate_deleted_representation(*args: Any) -> Any:
        result = original_update(*args)
        receipt = args[1]
        counterfactual_valid = receipt.representation[RAW_DIM] != jnp.float32(0.0)
        return result.replace(
            update_applied=result.update_applied & counterfactual_valid,
        )

    monkeypatch.setattr(
        adapter.objectives,
        "_update_jit",
        invalidate_deleted_representation,
    )
    with jax.disable_jit():
        result = adapter.update_transition(
            state,
            _transition(
                state,
                jnp.asarray([0.45, -0.25], dtype=jnp.float32),
                reward=jnp.asarray(0.2, dtype=jnp.float32),
                discount=jnp.asarray(0.9, dtype=jnp.float32),
            ),
            _target(adapter, state),
        )

    assert bool(result.rtu_causal_deletion_evidence_attempted)
    assert not bool(result.rtu_causal_deletion_evidence_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(state),
    )


def test_live_replacement_scrubs_every_stomp_axis_before_single_selection() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    source_stomp = state.prototype_state.oak_state.stomp_state
    assert int(source_stomp.executing_option) == -1
    next_observation = jnp.asarray([0.4, -0.3], dtype=jnp.float32)
    transition = _transition(
        state,
        next_observation,
        reward=jnp.asarray(0.2, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    initial_preparation = adapter.prototype.prepare_rtu_transition(
        state.prototype_state,
        transition,
    )
    recycled_axis = RAW_DIM
    axis_value = float(initial_preparation.decision_representation[recycled_axis])
    assert axis_value != 0.0
    coefficient = 100.0 if axis_value > 0.0 else -100.0

    base_state = source_stomp.base_learner_state
    base_weights = [jnp.zeros_like(value) for value in base_state.head_params.weights]
    base_weights[N_ACTIONS] = base_weights[N_ACTIONS].at[0, recycled_axis].set(
        coefficient
    )
    biased_base = base_state.replace(
        head_params=base_state.head_params.replace(weights=tuple(base_weights))
    )
    option_weights = jnp.zeros_like(source_stomp.option_policies.q_weights)
    option_weights = option_weights.at[0, 1, recycled_axis].set(coefficient)
    biased_stomp = source_stomp.replace(
        base_learner_state=biased_base,
        option_policies=source_stomp.option_policies.replace(
            q_weights=option_weights,
        ),
    )
    biased_oak = state.prototype_state.oak_state.replace(stomp_state=biased_stomp)
    biased_prototype = state.prototype_state.replace(oak_state=biased_oak)
    state = dataclasses.replace(  # type: ignore[type-var]
        state,
        prototype_state=biased_prototype,
    )
    assert bool(adapter.state_valid(state))
    transition = _transition(
        state,
        next_observation,
        reward=jnp.asarray(0.2, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    preparation = adapter.prototype.prepare_rtu_transition(
        state.prototype_state,
        transition,
    )
    no_reset = adapter.prototype.oak_agent.update(
        biased_oak,
        transition.reward,
        preparation.bootstrap_transition.representation,
        transition.discount,
        decision_observation=preparation.decision_representation,
    )
    assert int(no_reset.state.stomp_state.base_last_action) >= N_ACTIONS

    result = adapter.update_transition(
        state,
        transition,
        _target(adapter, state),
    )
    assert bool(result.update_applied)
    assert result.rtu_generate_and_test is not None
    assert bool(jnp.any(result.rtu_generate_and_test.diagnostics.selected_mask))
    final_stomp = result.state.prototype_state.oak_state.stomp_state
    assert int(final_stomp.base_last_action) < N_ACTIONS
    np.testing.assert_array_equal(
        final_stomp.base_last_obs,
        result.state.prototype_state.current_representation,
    )
    np.testing.assert_array_equal(
        result.bootstrap_representation,
        preparation.bootstrap_transition.representation,
    )
    assert not np.array_equal(
        np.asarray(result.bootstrap_representation),
        np.asarray(result.state.prototype_state.current_representation),
    )

    recycled_axes = np.asarray([RAW_DIM, RAW_DIM + 1])
    for weights in final_stomp.base_learner_state.head_params.weights:
        np.testing.assert_array_equal(weights[:, recycled_axes], 0.0)
    for weight_trace, _ in final_stomp.base_learner_state.head_traces:
        np.testing.assert_array_equal(weight_trace[:, recycled_axes], 0.0)
    np.testing.assert_array_equal(
        final_stomp.option_policies.q_weights[..., recycled_axes],
        0.0,
    )
    np.testing.assert_array_equal(
        final_stomp.option_policies.traces[..., recycled_axes],
        0.0,
    )
    np.testing.assert_array_equal(
        final_stomp.option_models.next_state_weights[:, recycled_axes, :],
        0.0,
    )
    np.testing.assert_array_equal(
        final_stomp.option_models.next_state_weights[:, :, recycled_axes],
        0.0,
    )
    final_objectives = result.state.objectives_state
    for weights in (
        final_objectives.observation_weights,
        final_objectives.latent_weights,
        final_objectives.reward_weights,
        final_objectives.termination_weights,
        final_objectives.gvf_weights,
        final_objectives.value_weights,
        final_objectives.advantage_weights,
        final_objectives.inverse_current_weights,
        final_objectives.inverse_next_weights,
    ):
        np.testing.assert_array_equal(weights[..., recycled_axes], 0.0)
    np.testing.assert_array_equal(
        final_stomp.step_words,
        np.asarray([0, 1], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        final_stomp.base_learner_state.step_words,
        np.asarray([0, 1], dtype=np.uint32),
    )
    expected_rng = jr.split(biased_stomp.rng_key, 3)[0]
    np.testing.assert_array_equal(
        jr.key_data(final_stomp.rng_key),
        jr.key_data(expected_rng),
    )
    np.testing.assert_array_equal(result.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        [0, 2],
    )
    np.testing.assert_array_equal(
        result.state.pending_builder_update_words,
        [0, 2],
    )


def test_active_option_defers_replacement_without_rolling_back_learning() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(254)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    assert int(state.prototype_state.oak_state.stomp_state.executing_option) >= 0
    result = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.4, -0.3], dtype=jnp.float32),
            reward=jnp.asarray(0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
        ),
        _target(adapter, state),
    )
    assert bool(result.update_applied)
    assert result.rtu_generate_and_test is not None
    assert not bool(jnp.any(result.rtu_generate_and_test.diagnostics.selected_mask))
    lifecycle_state = result.state.rtu_generate_and_test_state
    assert lifecycle_state is not None
    np.testing.assert_array_equal(lifecycle_state.observation_words, [0, 1])
    np.testing.assert_array_equal(
        lifecycle_state.replacement_event_words,
        [0, 0],
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        [0, 1],
    )


def test_rtu_prepare_finalize_requires_recomputed_provenance_and_rejects_replay() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=17,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    transition = _transition(
        state,
        jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        reward=jnp.asarray(0.2, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    prototype = adapter.prototype
    lifecycle = adapter.rtu_generate_and_test
    lifecycle_state = state.rtu_generate_and_test_state
    assert lifecycle is not None
    assert lifecycle_state is not None
    preparation = prototype.prepare_rtu_transition(
        state.prototype_state,
        transition,
    )
    builder = adapter.builder
    assert type(builder) is RecurrentTraceUnitStateBuilder
    learning_proposal = builder.propose_learning_update(
        preparation.source_builder_state,
        jnp.zeros((builder.feature_dim(),), dtype=jnp.float32),
    )
    advance_receipt = lifecycle.make_advance_receipt(
        preparation.source_builder_state,
        bootstrap_observation=transition.next_observation,
        previous_action=transition.action,
        previous_reward=transition.reward,
        previous_discount=transition.discount,
        episode_boundary=transition.terminated | transition.truncated,
        restart_observation=transition.next_decision_observation,
    )
    rtu_proposal = lifecycle.propose(
        lifecycle_state,
        preparation.source_builder_state,
        jnp.zeros((builder.feature_dim(),), dtype=jnp.float32),
        learning_proposal,
        advance_receipt,
    )
    authorized = lifecycle.commit(
        lifecycle_state,
        rtu_proposal.live_builder_state,
        rtu_proposal,
    )
    assert bool(authorized.diagnostics.applied)
    receipt = prototype.bind_rtu_finalization(
        preparation,
        authorized.builder_state,
        rtu_proposal.selected_mask,
        rtu_proposal,
    )
    accepted = prototype.finalize_rtu_transition(
        state.prototype_state,
        transition,
        receipt,
        lifecycle,
    )
    assert bool(accepted.transition_diagnostics.valid)
    mismatched_lifecycle = RTUGenerateAndTest(
        dataclasses.replace(
            lifecycle.config,
            builder=dataclasses.replace(
                lifecycle.config.builder,
                hidden_dim=lifecycle.config.builder.hidden_dim + 1,
            ),
        )
    )
    with pytest.raises(ValueError, match="must exactly match Prototype"):
        prototype.finalize_rtu_transition(
            state.prototype_state,
            transition,
            receipt,
            mismatched_lifecycle,
        )
    compiled = jax.jit(
        lambda source, event, authorization: prototype.finalize_rtu_transition(
            source,
            event,
            authorization,
            lifecycle,
        )
    )(
        state.prototype_state,
        transition,
        receipt,
    )
    chex.assert_trees_all_equal(
        _materialize_keys(compiled.state),
        _materialize_keys(accepted.state),
    )

    # The standalone seam proves derivation from caller-supplied lifecycle,
    # gradient, and learning-proposal inputs; it cannot authenticate authority
    # over any of them.  The live adapter constructs and owns all three.
    supplied_foreign_source = lifecycle.init(jr.key(432, impl="threefry2x32"))
    foreign_proposal = lifecycle.propose(
        supplied_foreign_source,
        preparation.source_builder_state,
        jnp.zeros((builder.feature_dim(),), dtype=jnp.float32),
        learning_proposal,
        advance_receipt,
    )
    foreign_authorized = lifecycle.commit(
        supplied_foreign_source,
        foreign_proposal.live_builder_state,
        foreign_proposal,
    )
    foreign_receipt = prototype.bind_rtu_finalization(
        preparation,
        foreign_authorized.builder_state,
        foreign_proposal.selected_mask,
        foreign_proposal,
    )
    foreign_accepted = prototype.finalize_rtu_transition(
        state.prototype_state,
        transition,
        foreign_receipt,
        lifecycle,
    )
    assert bool(foreign_accepted.transition_diagnostics.valid)

    tampered_preparation = preparation.replace(
        decision_representation=(preparation.decision_representation + 1.0)
    )
    invented_destination = authorized.builder_state.replace(
        parameters=authorized.builder_state.parameters.at[0].add(1.0)
    )
    invented_mask = rtu_proposal.selected_mask.at[0].set(
        ~rtu_proposal.selected_mask[0]
    )
    invented_proposal = rtu_proposal.replace(
        downstream_loss_gradient=(rtu_proposal.downstream_loss_gradient + 1.0)
    )
    invented_source = rtu_proposal.source_state.replace(
        rng_key=jr.key(987, impl="threefry2x32")
    )
    invented_source_proposal = rtu_proposal.replace(
        source_state=invented_source,
    )
    wrong_transition_receipt = lifecycle.make_advance_receipt(
        preparation.source_builder_state,
        bootstrap_observation=transition.next_observation,
        previous_action=transition.action,
        previous_reward=transition.reward + jnp.asarray(0.25, dtype=jnp.float32),
        previous_discount=transition.discount,
        episode_boundary=transition.terminated | transition.truncated,
        restart_observation=transition.next_decision_observation,
    )
    wrong_transition_proposal = lifecycle.propose(
        lifecycle_state,
        preparation.source_builder_state,
        jnp.zeros((builder.feature_dim(),), dtype=jnp.float32),
        learning_proposal,
        wrong_transition_receipt,
    )
    wrong_transition_authorized = lifecycle.commit(
        lifecycle_state,
        wrong_transition_proposal.live_builder_state,
        wrong_transition_proposal,
    )
    assert bool(wrong_transition_authorized.diagnostics.applied)
    receipts = (
        receipt.replace(preparation=tampered_preparation),
        receipt.replace(rtu_proposal=invented_proposal),
        prototype.bind_rtu_finalization(
            preparation,
            invented_destination,
            rtu_proposal.selected_mask,
            rtu_proposal,
        ),
        prototype.bind_rtu_finalization(
            preparation,
            authorized.builder_state,
            invented_mask,
            rtu_proposal,
        ),
        prototype.bind_rtu_finalization(
            preparation,
            authorized.builder_state,
            rtu_proposal.selected_mask,
            invented_proposal,
        ),
        prototype.bind_rtu_finalization(
            preparation,
            authorized.builder_state,
            rtu_proposal.selected_mask,
            invented_source_proposal,
        ),
        prototype.bind_rtu_finalization(
            preparation,
            wrong_transition_authorized.builder_state,
            wrong_transition_proposal.selected_mask,
            wrong_transition_proposal,
        ),
    )
    for candidate in receipts:
        rejected = prototype.finalize_rtu_transition(
            state.prototype_state,
            transition,
            candidate,
            lifecycle,
        )
        assert not bool(rejected.transition_diagnostics.valid)
        chex.assert_trees_all_equal(
            _materialize_keys(rejected.state),
            _materialize_keys(state.prototype_state),
        )

    replayed = prototype.finalize_rtu_transition(
        accepted.state,
        transition,
        receipt,
        lifecycle,
    )
    assert not bool(replayed.transition_diagnostics.valid)
    chex.assert_trees_all_equal(
        _materialize_keys(replayed.state),
        _materialize_keys(accepted.state),
    )


def test_live_adapter_rejects_proposal_not_owned_by_its_lifecycle_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    lifecycle = adapter.rtu_generate_and_test
    assert lifecycle is not None
    original_propose = lifecycle.propose

    def propose_with_invented_source(*args: Any, **kwargs: Any) -> Any:
        proposal = original_propose(*args, **kwargs)
        invented_source = proposal.source_state.replace(
            rng_key=jr.key(654, impl="threefry2x32")
        )
        return proposal.replace(source_state=invented_source)

    monkeypatch.setattr(lifecycle, "propose", propose_with_invented_source)
    with jax.disable_jit():
        rejected = adapter.update_transition(
            state,
            _transition(
                state,
                jnp.asarray([0.4, -0.3], dtype=jnp.float32),
                reward=jnp.asarray(0.2, dtype=jnp.float32),
                discount=jnp.asarray(0.9, dtype=jnp.float32),
            ),
            _target(adapter, state),
        )
    assert not bool(rejected.rtu_lifecycle_source_matches)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected.state),
        _materialize_keys(state),
    )


def test_rtu_adapter_resource_count_matches_physical_and_logical_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    state = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.2, 0.1], dtype=jnp.float32),
    ).state
    budget = adapter.resource_budget(state)
    physical_commit_evaluations = 0
    physical_rtu_commit_evaluations = 0
    original_commit = RecurrentTraceUnitStateBuilder.commit_learning_update
    original_rtu_commit = RTUGenerateAndTest.commit

    def counted_commit(
        builder: RecurrentTraceUnitStateBuilder,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal physical_commit_evaluations
        physical_commit_evaluations += 1
        return original_commit(builder, *args, **kwargs)

    def counted_rtu_commit(
        lifecycle: RTUGenerateAndTest,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal physical_rtu_commit_evaluations
        physical_rtu_commit_evaluations += 1
        return original_rtu_commit(lifecycle, *args, **kwargs)

    monkeypatch.setattr(
        RecurrentTraceUnitStateBuilder,
        "commit_learning_update",
        counted_commit,
    )
    monkeypatch.setattr(RTUGenerateAndTest, "commit", counted_rtu_commit)
    with jax.disable_jit():
        result = adapter.update_transition(
            state,
            _transition(
                state,
                jnp.asarray([0.4, -0.3], dtype=jnp.float32),
                reward=jnp.asarray(0.2, dtype=jnp.float32),
                discount=jnp.asarray(0.9, dtype=jnp.float32),
            ),
            _target(adapter, state),
        )

    assert bool(result.update_applied)
    assert physical_commit_evaluations == 4
    assert physical_rtu_commit_evaluations == 2
    assert result.rtu_generate_and_test is not None
    np.testing.assert_array_equal(result.builder_learning.pre_update_words, [0, 0])
    np.testing.assert_array_equal(result.builder_learning.post_update_words, [0, 1])
    np.testing.assert_array_equal(
        result.rtu_generate_and_test.state.replacement_event_words,
        [0, 1],
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        [0, 2],
    )
    assert budget.max_builder_commits_per_transition == physical_commit_evaluations
    assert (
        budget.max_rtu_generate_and_test_commits_per_transition
        == physical_rtu_commit_evaluations
    )


def test_live_rtu_envelope_rejects_unsupported_consumers_and_semantics() -> None:
    valid = _rtu_adapter(with_generate_and_test=True)
    lifecycle = valid.rtu_generate_and_test
    assert lifecycle is not None
    config = valid.prototype.config

    nonlinear_stomp = dataclasses.replace(
        config.oak.stomp,
        base_hidden_sizes=(3,),
    )
    planning_stomp = dataclasses.replace(
        config.oak.stomp,
        option_planning_backups_per_step=1,
    )
    hidden_subtask_stomp = dataclasses.replace(
        config.oak.stomp,
        subtask_specs=(SubtaskSpec(feature_index=RAW_DIM),),
    )
    rejected_configs = (
        dataclasses.replace(
            config,
            oak=dataclasses.replace(config.oak, stomp=nonlinear_stomp),
        ),
        dataclasses.replace(
            config,
            oak=dataclasses.replace(config.oak, stomp=planning_stomp),
        ),
        dataclasses.replace(
            config,
            oak=dataclasses.replace(
                config.oak,
                stomp=hidden_subtask_stomp,
            ),
        ),
        dataclasses.replace(
            config,
            world_model=ActionConditionedWorldModelConfig(
                observation_dim=config.oak.observation_dim,
                n_actions=N_ACTIONS,
            ),
        ),
    )
    for rejected_config in rejected_configs:
        with pytest.raises(ValueError, match="live RTU replacement"):
            PrototypeComprehensiveStateObjectives(
                PrototypeAgent(rejected_config),
                valid.objectives,
                lifecycle,
            )

    protected_lifecycle = RTUGenerateAndTest(
        dataclasses.replace(
            lifecycle.config,
            protected_units=(0,),
        )
    )
    protected_adapter = PrototypeComprehensiveStateObjectives(
        PrototypeAgent(
            dataclasses.replace(
                config,
                oak=dataclasses.replace(
                    config.oak,
                    stomp=hidden_subtask_stomp,
                ),
            )
        ),
        valid.objectives,
        protected_lifecycle,
    )
    assert protected_adapter.rtu_generate_and_test is protected_lifecycle


def _run_scan(
    adapter: PrototypeComprehensiveStateObjectives,
    initial: PrototypeComprehensiveObjectivesState,
    next_observations: jax.Array,
    rewards: jax.Array,
    discounts: jax.Array,
) -> PrototypeComprehensiveObjectivesScanResult:
    steps = next_observations.shape[0]
    return run_prototype_comprehensive_objectives_scan(
        adapter,
        initial,
        next_observations,
        next_observations,
        rewards,
        discounts,
        jnp.zeros((steps,), dtype=jnp.bool_),
        jnp.zeros((steps,), dtype=jnp.bool_),
        jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        jnp.asarray([0.9, 0.8, 0.7], dtype=jnp.float32),
        jnp.asarray([0.2, 0.1, -0.1], dtype=jnp.float32),
        jnp.asarray([-0.2, 0.0, 0.3], dtype=jnp.float32),
        jnp.asarray([[0, 1], [0, 2], [0, 3]], dtype=jnp.uint32),
        jnp.asarray(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=jnp.uint32,
        ),
    )


def test_eager_jit_and_public_scan_preserve_clocks_and_outputs() -> None:
    adapter = _adapter()
    initial = adapter.start(
        adapter.init(jr.key(21)),
        jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    ).state
    observations = jnp.asarray(
        [[0.2, 0.3], [-0.4, 0.5], [0.6, -0.1]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.8, 1.0], dtype=jnp.float32)
    with jax.disable_jit():
        eager = _run_scan(adapter, initial, observations, rewards, discounts)
    compiled = jax.jit(_run_scan, static_argnums=(0,))(
        adapter,
        initial,
        observations,
        rewards,
        discounts,
    )
    _assert_tree_allclose(eager, compiled)
    assert bool(jnp.all(compiled.update_applied))
    np.testing.assert_array_equal(
        compiled.transaction_words,
        [[0, 1], [0, 2], [0, 3]],
    )
    np.testing.assert_array_equal(
        compiled.target_identity_words,
        [[0, 1], [0, 2], [0, 3]],
    )
    np.testing.assert_array_equal(compiled.state.transaction_words, [0, 3])
    np.testing.assert_array_equal(compiled.state.target_receipt_words, [0, 3])
    assert bool(adapter.state_valid(compiled.state))


def test_rtu_lifecycle_scan_jit_resource_and_checkpoint_are_exact(
    tmp_path: Path,
) -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=1,
    )
    config = adapter.to_config()
    assert "rtu_generate_and_test_config" in config
    restored_config = PrototypeComprehensiveStateObjectives.from_config(config)
    assert restored_config.to_config() == config
    initial = adapter.start(
        adapter.init(jr.key(255)),
        jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    ).state
    initial_stomp = initial.prototype_state.oak_state.stomp_state
    initial_biases = list(initial_stomp.base_learner_state.head_params.biases)
    initial_biases[N_ACTIONS] = jnp.full_like(
        initial_biases[N_ACTIONS],
        -100.0,
    )
    primitive_only_base = initial_stomp.base_learner_state.replace(
        head_params=initial_stomp.base_learner_state.head_params.replace(
            biases=tuple(initial_biases),
        )
    )
    primitive_only_stomp = initial_stomp.replace(
        base_learner_state=primitive_only_base,
    )
    initial = dataclasses.replace(  # type: ignore[type-var]
        initial,
        prototype_state=initial.prototype_state.replace(
            oak_state=initial.prototype_state.oak_state.replace(
                stomp_state=primitive_only_stomp,
            )
        ),
    )
    assert bool(adapter.state_valid(initial))
    observations = jnp.asarray(
        [[0.2, 0.3], [-0.4, 0.5], [0.6, -0.1]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.8, 1.0], dtype=jnp.float32)
    with jax.disable_jit():
        eager = _run_scan(adapter, initial, observations, rewards, discounts)
    compiled = jax.jit(_run_scan, static_argnums=(0,))(
        adapter,
        initial,
        observations,
        rewards,
        discounts,
    )
    _assert_tree_allclose(eager, compiled)
    assert bool(jnp.all(compiled.update_applied))
    lifecycle_state = compiled.state.rtu_generate_and_test_state
    assert lifecycle_state is not None
    np.testing.assert_array_equal(lifecycle_state.observation_words, [0, 3])
    np.testing.assert_array_equal(lifecycle_state.replacement_words, [0, 3])
    np.testing.assert_array_equal(
        lifecycle_state.replacement_event_words,
        [0, 3],
    )
    np.testing.assert_array_equal(
        compiled.state.prototype_state.state_builder_state.update_words,
        [0, 6],
    )
    budget = adapter.resource_budget(compiled.state)
    assert budget.rtu_generate_and_test_state_nbytes > 0
    # Counts all source-level builder-commit evaluations: proposal preflight,
    # live materialization, outer commit recomputation, and Prototype's
    # independent authorization recomputation.  Only one ordinary update
    # revision enters persistent state.
    assert budget.max_builder_commits_per_transition == 4
    assert budget.max_rtu_generate_and_test_proposals_per_transition == 1
    assert budget.max_rtu_generate_and_test_commits_per_transition == 2
    assert budget.max_causal_deletion_units_scored_per_transition == 1
    assert budget.max_causal_deletion_frozen_head_evaluations_per_transition == 8
    assert budget.max_accepted_transitions == (
        PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RTU_MAX_TRANSITIONS
    )
    assert budget.max_accepted_transitions == 2**32 - 1
    assert adapter.to_config()["max_transitions"] == budget.max_accepted_transitions

    checkpoint = tmp_path / "prototype-comprehensive-rtu-lifecycle"
    save_prototype_comprehensive_objectives_checkpoint(
        adapter,
        compiled.state,
        checkpoint,
    )
    restored_adapter, restored_state = (
        load_prototype_comprehensive_objectives_checkpoint(checkpoint)
    )
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(
        _materialize_keys(restored_state),
        _materialize_keys(compiled.state),
    )
    assert bool(restored_adapter.state_valid(restored_state))

    lifecycle_state = compiled.state.rtu_generate_and_test_state
    assert lifecycle_state is not None
    selected_unit = int(np.flatnonzero(np.asarray(lifecycle_state.last_replaced_mask))[0])
    selected_real_axis = RAW_DIM + selected_unit
    assert (
        np.asarray(
            compiled.state.objectives_state.value_weights[selected_real_axis]
        ).view(np.uint32)
        == np.uint32(0)
    )
    corrupt_head_state = dataclasses.replace(  # type: ignore[type-var]
        compiled.state,
        objectives_state=compiled.state.objectives_state.replace(
            value_weights=(
                compiled.state.objectives_state.value_weights.at[
                    selected_real_axis
                ].set(jnp.float32(1.0))
            )
        ),
    )
    corrupt_pending_state = dataclasses.replace(  # type: ignore[type-var]
        compiled.state,
        objectives_state=compiled.state.objectives_state.replace(
            pending_representation=(
                compiled.state.objectives_state.pending_representation.at[
                    selected_real_axis
                ].set(jnp.float32(1.0))
            )
        ),
    )
    corrupt_signed_zero_state = dataclasses.replace(  # type: ignore[type-var]
        compiled.state,
        objectives_state=compiled.state.objectives_state.replace(
            value_weights=(
                compiled.state.objectives_state.value_weights.at[
                    selected_real_axis
                ].set(jnp.float32(-0.0))
            )
        ),
    )
    builder = cast(Any, compiled.state.prototype_state.state_builder_state)
    corrupt_sensitivities = builder.sensitivities._replace(
        b_real=builder.sensitivities.b_real.at[0, selected_unit, 0].set(
            jnp.float32(1.0)
        )
    )
    corrupt_builder_state = dataclasses.replace(  # type: ignore[type-var]
        compiled.state,
        prototype_state=compiled.state.prototype_state.replace(
            state_builder_state=builder.replace(
                sensitivities=corrupt_sensitivities
            )
        ),
    )
    stomp = compiled.state.prototype_state.oak_state.stomp_state
    base = stomp.base_learner_state
    corrupt_weights = list(base.head_params.weights)
    corrupt_weights[0] = corrupt_weights[0].at[0, selected_real_axis].set(
        jnp.float32(1.0)
    )
    corrupt_stomp_state = dataclasses.replace(  # type: ignore[type-var]
        compiled.state,
        prototype_state=compiled.state.prototype_state.replace(
            oak_state=compiled.state.prototype_state.oak_state.replace(
                stomp_state=stomp.replace(
                    base_learner_state=base.replace(
                        head_params=base.head_params.replace(
                            weights=tuple(corrupt_weights)
                        )
                    )
                )
            )
        ),
    )
    assert not bool(adapter.state_valid(corrupt_head_state))
    assert not bool(adapter.state_valid(corrupt_pending_state))
    assert not bool(adapter.state_valid(corrupt_signed_zero_state))
    assert not bool(adapter.state_valid(corrupt_builder_state))
    assert not bool(adapter.state_valid(corrupt_stomp_state))
    with pytest.raises(ValueError, match="invalid composed state"):
        save_prototype_comprehensive_objectives_checkpoint(
            adapter,
            corrupt_head_state,
            tmp_path / "corrupt-prototype-comprehensive-rtu-lifecycle",
        )


def test_rtu_global_lifetime_fail_stop_is_exact_at_the_uint32_boundary() -> None:
    penultimate = jnp.asarray([0, 2**32 - 2], dtype=jnp.uint32)
    maximum = jnp.asarray([0, 2**32 - 1], dtype=jnp.uint32)
    past_declared_horizon = jnp.asarray([1, 0], dtype=jnp.uint32)

    assert bool(pco_module._rtu_global_lifetime_state_valid(penultimate))
    assert bool(pco_module._rtu_global_lifetime_capacity(penultimate))
    assert bool(pco_module._rtu_global_lifetime_state_valid(maximum))
    assert not bool(pco_module._rtu_global_lifetime_capacity(maximum))
    assert not bool(pco_module._rtu_global_lifetime_state_valid(past_declared_horizon))
    assert not bool(pco_module._rtu_global_lifetime_capacity(past_declared_horizon))


def test_checkpoint_resume_preserves_pending_owner_and_target_provenance(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(22), lifecycle_id=jnp.asarray([11, 13], dtype=jnp.uint32)),
        jnp.asarray([0.3, 0.6], dtype=jnp.float32),
    ).state
    first = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-0.2, 0.4], dtype=jnp.float32),
            reward=jnp.asarray(0.1, dtype=jnp.float32),
            discount=jnp.asarray(0.95, dtype=jnp.float32),
        ),
        _target(adapter, state),
    )
    assert bool(first.update_applied)
    checkpoint = tmp_path / "prototype-comprehensive"
    save_prototype_comprehensive_objectives_checkpoint(adapter, first.state, checkpoint)
    restored_adapter, restored_state = load_prototype_comprehensive_objectives_checkpoint(
        checkpoint
    )
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(
        _materialize_keys(restored_state),
        _materialize_keys(first.state),
    )

    next_observation = jnp.asarray([0.8, -0.7], dtype=jnp.float32)
    uninterrupted = adapter.update_transition(
        first.state,
        _transition(
            first.state,
            next_observation,
            reward=jnp.asarray(-0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.85, dtype=jnp.float32),
        ),
        _target(adapter, first.state),
    )
    resumed = restored_adapter.update_transition(
        restored_state,
        _transition(
            restored_state,
            next_observation,
            reward=jnp.asarray(-0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.85, dtype=jnp.float32),
        ),
        _target(restored_adapter, restored_state),
    )
    assert bool(uninterrupted.update_applied)
    assert bool(resumed.update_applied)
    _assert_tree_allclose(uninterrupted.state, resumed.state)

    malformed = tmp_path / "malformed-prototype-comprehensive"
    save_checkpoint(
        first.state,
        malformed,
        metadata={"schema": "missing-required-fields"},
    )
    with pytest.raises(ValueError, match="manifest is not exact"):
        load_prototype_comprehensive_objectives_checkpoint(malformed)
