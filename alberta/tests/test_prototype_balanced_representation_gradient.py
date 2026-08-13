# mypy: disable-error-code="attr-defined,call-arg,operator"
"""Balanced causal control/world gradients at the Prototype builder boundary."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.delight import GradientJoyConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, STOMPState, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.representation_gradient_mixer import (
    GradientMixMode,
    RepresentationGradientMixerConfig,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
)
from alberta_framework.core.types import MLPParams
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = pytest.mark.unit

RAW_DIM = 1
FEATURE_DIM = 2
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_jit_scan_and_checkpoint_preserve_mixer_contract":
        yield
    else:
        with jax.disable_jit():
            yield


def _builder_config() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=1,
        include_raw_observation=True,
        step_size=0.2,
        gradient_clip=100.0,
    )


def _oak_config(*, threshold: float = 100.0) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=threshold,
                    max_option_steps=8,
                ),
            ),
            observation_dim=FEATURE_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _ensemble_config() -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.95,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.8,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=FEATURE_DIM + 2,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            fast_loss_decay=0.5,
            slow_loss_decay=0.9,
            max_input_magnitude=100.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=2,
        bootstrap_probability=0.8,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
    )


def _mixer(
    mode: GradientMixMode,
    *,
    behavior_weight: float = 1.0,
    world_weight: float = 1.0,
) -> RepresentationGradientMixerConfig:
    return RepresentationGradientMixerConfig(
        representation_dim=FEATURE_DIM,
        mode=mode,
        behavior_weight=behavior_weight,
        grounded_world_weight=world_weight,
    )


def _agent(
    mode: GradientMixMode,
    *,
    ensemble: bool = False,
    behavior_weight: float = 1.0,
    world_weight: float = 1.0,
    candidate_audit: bool = False,
    option_threshold: float = 100.0,
) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(threshold=option_threshold),
            state_builder=_builder_config(),
            world_model_ensemble=_ensemble_config() if ensemble else None,
            representation_gradient_mixer=_mixer(
                mode,
                behavior_weight=behavior_weight,
                world_weight=world_weight,
            ),
            gradient_joy=(
                GradientJoyConfig(
                    candidate_semantics="update",
                    max_update_norm=100.0,
                    alignment_temperature=1.0,
                    norm_temperature=1.0,
                    diagnostics_epsilon=1.0e-12,
                )
                if candidate_audit
                else None
            ),
        )
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: float,
    *,
    reward: float = 0.3,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    next_decision_observation: float | None = None,
) -> PrototypeTransition:
    bootstrap = jnp.asarray([next_observation], dtype=jnp.float32)
    decision = jnp.asarray(
        [
            next_observation
            if next_decision_observation is None
            else next_decision_observation
        ],
        dtype=jnp.float32,
    )
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
        next_observation=bootstrap,
        next_decision_observation=decision,
    )


def _force_idle(state: PrototypeAgentState) -> PrototypeAgentState:
    stomp = cast(
        STOMPState,
        state.oak_state.stomp_state.replace(
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
            base_last_action=state.current_action,
        ),
    )
    oak = cast(OaKState, state.oak_state.replace(stomp_state=stomp))
    return cast(PrototypeAgentState, state.replace(oak_state=oak))


def _force_option(
    state: PrototypeAgentState,
    *,
    option_start_observation: jax.Array | None = None,
) -> PrototypeAgentState:
    stomp = cast(
        STOMPState,
        state.oak_state.stomp_state.replace(
            executing_option=jnp.asarray(0, dtype=jnp.int32),
            base_last_action=jnp.asarray(N_ACTIONS, dtype=jnp.int32),
            option_last_intra_action=state.current_action,
            option_start_obs=(
                state.current_representation
                if option_start_observation is None
                else option_start_observation
            ),
            option_steps=jnp.asarray(1, dtype=jnp.int32),
        ),
    )
    oak = cast(OaKState, state.oak_state.replace(stomp_state=stomp))
    return cast(PrototypeAgentState, state.replace(oak_state=oak))


def _set_base_linear_heads(
    state: PrototypeAgentState,
    selected_weight: jax.Array,
) -> PrototypeAgentState:
    stomp = state.oak_state.stomp_state
    learner = stomp.base_learner_state
    weights = tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights)
    weights = tuple(
        selected_weight.reshape(weight.shape) if index == int(state.current_action) else weight
        for index, weight in enumerate(weights)
    )
    biases = tuple(jnp.zeros_like(bias) for bias in learner.head_params.biases)
    learner = learner.replace(
        head_params=MLPParams(weights=weights, biases=biases),
    )
    stomp = cast(STOMPState, stomp.replace(base_learner_state=learner))
    oak = cast(OaKState, state.oak_state.replace(stomp_state=stomp))
    return cast(PrototypeAgentState, state.replace(oak_state=oak))


def _set_option_linear_heads(
    state: PrototypeAgentState,
    selected_weight: jax.Array,
) -> PrototypeAgentState:
    stomp = state.oak_state.stomp_state
    action = int(state.current_action)
    q_weights = jnp.zeros_like(stomp.option_policies.q_weights)
    q_weights = q_weights.at[0, action].set(selected_weight)
    option_policies = stomp.option_policies.replace(
        q_weights=q_weights,
        average_rewards=jnp.zeros_like(stomp.option_policies.average_rewards),
    )
    stomp = cast(STOMPState, stomp.replace(option_policies=option_policies))
    oak = cast(OaKState, state.oak_state.replace(stomp_state=stomp))
    return cast(PrototypeAgentState, state.replace(oak_state=oak))


def _assert_tree_equal(left: object, right: object) -> None:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    left = jax.tree.map(materialize, left)
    right = jax.tree.map(materialize, right)
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_is_opt_in_strict_and_preserves_legacy_serialization() -> None:
    legacy = PrototypeAgentConfig(
        oak=_oak_config(),
        state_builder=_builder_config(),
        world_model_ensemble=_ensemble_config(),
        learn_state_builder_from_world_model=True,
    )
    assert "representation_gradient_mixer" not in legacy.to_config()
    assert PrototypeAgentConfig.from_config(legacy.to_config()).to_config() == legacy.to_config()

    behavior_only = _agent("behavior_only")
    restored = PrototypeAgent.from_config(behavior_only.to_config())
    assert restored.to_config() == behavior_only.to_config()
    assert restored.config.learn_state_builder_from_world_model is False

    with pytest.raises(ValueError, match="representation_dim"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=_builder_config(),
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=3,
                mode="behavior_only",
            ),
        )
    with pytest.raises(ValueError, match="grounded-world"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=_builder_config(),
            representation_gradient_mixer=_mixer("world_only"),
        )
    with pytest.raises(ValueError, match="OnlineGated"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=FEATURE_DIM),
            representation_gradient_mixer=_mixer("behavior_only"),
        )


def test_idle_primitive_gradient_is_frozen_target_td_loss_and_drives_builder() -> None:
    agent = _agent("behavior_only")
    state = agent.start(agent.init(jr.key(1)), jnp.asarray([0.6], dtype=jnp.float32))
    state = _force_idle(state)
    selected_weight = jnp.asarray([0.7, -0.3], dtype=jnp.float32)
    state = _set_base_linear_heads(state, selected_weight)
    transition = _transition(state, -0.2, reward=0.4, discount=0.8)
    builder = agent.state_builder
    assert builder is not None
    _, bootstrap = builder.update(
        state.state_builder_state,
        transition.next_observation,
        transition.action,
        transition.reward,
        transition.discount,
    )
    base = state.oak_state.stomp_state
    next_values = agent.oak_agent.base_q_values(state.oak_state, bootstrap)
    target = transition.reward - base.base_average_reward + transition.discount * jnp.max(
        next_values
    )
    prediction = selected_weight @ state.current_representation
    td_error = target - prediction
    expected_gradient = -td_error * selected_weight
    proposal = builder.propose_learning_update(
        state.state_builder_state,
        expected_gradient,
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_representation_gradient_valid)
    assert bool(result.behavior_gradient_result.diagnostics.idle_base_source)
    assert not bool(result.behavior_gradient_result.diagnostics.intra_option_source)
    chex.assert_trees_all_close(result.behavior_representation_gradient, expected_gradient)
    chex.assert_trees_all_close(
        result.behavior_gradient_result.diagnostics.td_error,
        result.oak_td_error,
    )
    chex.assert_trees_all_close(result.mixed_representation_gradient, expected_gradient)
    assert bool(result.mixed_representation_gradient_valid)
    assert bool(result.state_builder_learning_diagnostics.applied)
    chex.assert_trees_all_close(
        result.state.state_builder_state.parameters,
        state.state_builder_state.parameters + proposal.candidate_parameter_update,
    )


def test_full_mix_discloses_raw_sources_and_routes_exact_mixed_candidate() -> None:
    agent = _agent(
        "full",
        ensemble=True,
        behavior_weight=1.5,
        world_weight=0.25,
    )
    state = agent.start(agent.init(jr.key(2)), jnp.asarray([0.2], dtype=jnp.float32))
    state = _force_idle(state)
    state = _set_base_linear_heads(
        state,
        jnp.asarray([0.4, -0.1], dtype=jnp.float32),
    )
    result = agent.update_transition(state, _transition(state, 0.1))
    expected = mix_representation_gradients(
        cast(RepresentationGradientMixerConfig, agent.config.representation_gradient_mixer),
        result.behavior_representation_gradient,
        result.world_model_representation_gradient,
        behavior_valid=result.behavior_representation_gradient_valid,
        grounded_world_valid=result.world_model_representation_gradient_valid,
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_representation_gradient_valid)
    assert bool(result.world_model_representation_gradient_valid)
    chex.assert_trees_all_close(result.representation_gradient_mix, expected)
    assert bool(result.representation_gradient_mix.diagnostics.behavior_active)
    assert bool(result.representation_gradient_mix.diagnostics.grounded_world_active)
    assert bool(result.state_builder_learning_diagnostics.applied)


def test_world_only_ignores_invalid_inactive_behavior_source() -> None:
    agent = _agent("world_only", ensemble=True)
    state = agent.start(agent.init(jr.key(21)), jnp.asarray([1.0], dtype=jnp.float32))
    state = _force_idle(state)
    state = _set_base_linear_heads(
        state,
        jnp.asarray([1.0e20, 0.0], dtype=jnp.float32),
    )
    result = agent.update_transition(
        state,
        _transition(state, 0.0, reward=0.0, discount=0.9),
    )

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.behavior_representation_gradient_valid)
    assert bool(result.world_model_representation_gradient_valid)
    assert not bool(result.representation_gradient_mix.diagnostics.behavior_active)
    assert bool(result.representation_gradient_mix.diagnostics.grounded_world_active)
    assert bool(result.representation_gradient_mix.applied)
    assert not bool(result.representation_gradient_mix.rejected)
    assert bool(result.state_builder_learning_diagnostics.applied)


def test_discard_mode_is_valid_zero_but_does_not_consume_builder_capacity() -> None:
    agent = _agent("discard")
    state = agent.start(agent.init(jr.key(22)), jnp.asarray([0.2], dtype=jnp.float32))
    state = _force_idle(state)
    before = state.state_builder_state.update_count
    result = agent.update_transition(state, _transition(state, 0.1))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.representation_gradient_mix.valid)
    assert not bool(result.representation_gradient_mix.applied)
    assert bool(result.representation_gradient_mix.zero_output)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    np.testing.assert_array_equal(
        np.asarray(result.state.state_builder_state.update_count),
        np.asarray(before),
    )


@pytest.mark.parametrize(
    ("terminated", "truncated", "discount", "expected_bootstrap"),
    [
        (False, True, 0.8, 0.8),
        (True, False, 0.0, 0.0),
    ],
)
def test_executing_option_uses_current_intra_option_loss_across_boundaries(
    terminated: bool,
    truncated: bool,
    discount: float,
    expected_bootstrap: float,
) -> None:
    agent = _agent("behavior_only")
    state = agent.start(agent.init(jr.key(3)), jnp.asarray([0.25], dtype=jnp.float32))
    state = _force_option(
        state,
        option_start_observation=jnp.asarray([9.0, -7.0], dtype=jnp.float32),
    )
    option_weight = jnp.asarray([0.6, -0.2], dtype=jnp.float32)
    state = _set_option_linear_heads(state, option_weight)
    state = _set_base_linear_heads(
        state,
        jnp.asarray([-4.0, 5.0], dtype=jnp.float32),
    )
    transition = _transition(
        state,
        -0.1,
        reward=0.7,
        discount=discount,
        terminated=terminated,
        truncated=truncated,
        next_decision_observation=0.9 if terminated or truncated else None,
    )
    builder = agent.state_builder
    assert builder is not None
    _, bootstrap = builder.update(
        state.state_builder_state,
        transition.next_observation,
        transition.action,
        transition.reward,
        transition.discount,
    )
    option_q = state.oak_state.stomp_state.option_policies.q_weights[0]
    pseudo_reward = bootstrap[0]
    target = pseudo_reward + expected_bootstrap * jnp.max(option_q @ bootstrap)
    prediction = option_weight @ state.current_representation
    expected_gradient = -(target - prediction) * option_weight

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.diagnostics.intra_option_source)
    assert not bool(result.behavior_gradient_result.diagnostics.idle_base_source)
    chex.assert_trees_all_close(
        result.behavior_gradient_result.diagnostics.bootstrap_discount,
        jnp.asarray(expected_bootstrap, dtype=jnp.float32),
    )
    assert bool(result.behavior_gradient_result.diagnostics.option_terminates) == terminated
    chex.assert_trees_all_close(result.behavior_representation_gradient, expected_gradient)


def test_natural_option_completion_zeros_only_the_intra_option_bootstrap() -> None:
    agent = _agent("behavior_only", option_threshold=0.05)
    state = agent.start(agent.init(jr.key(23)), jnp.asarray([0.1], dtype=jnp.float32))
    state = _force_option(state)
    state = _set_option_linear_heads(
        state,
        jnp.asarray([0.4, -0.1], dtype=jnp.float32),
    )
    result = agent.update_transition(
        state,
        _transition(state, 0.2, discount=0.9),
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.diagnostics.intra_option_source)
    assert bool(result.behavior_gradient_result.diagnostics.option_terminates)
    chex.assert_trees_all_close(
        result.behavior_gradient_result.diagnostics.bootstrap_discount,
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def test_nonfinite_behavior_loss_fails_closed_without_blocking_valid_control() -> None:
    agent = _agent("behavior_only")
    state = agent.start(agent.init(jr.key(4)), jnp.asarray([1.0], dtype=jnp.float32))
    state = _force_idle(state)
    state = _set_base_linear_heads(
        state,
        jnp.asarray([1.0e20, 0.0], dtype=jnp.float32),
    )
    before_updates = state.state_builder_state.update_count
    result = agent.update_transition(
        state,
        _transition(state, 0.0, reward=0.0, discount=0.9),
    )

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.behavior_representation_gradient_valid)
    assert not bool(result.behavior_gradient_result.diagnostics.loss_finite)
    assert not bool(result.behavior_gradient_result.diagnostics.gradient_finite)
    assert bool(result.representation_gradient_mix.rejected)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    for record in (
        result.behavior_gradient_result,
        result.representation_gradient_mix,
    ):
        for leaf in jax.tree.leaves(record):
            dtype = getattr(leaf, "dtype", None)
            if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
                assert bool(jnp.all(jnp.isfinite(leaf)))
    np.testing.assert_array_equal(
        np.asarray(result.state.state_builder_state.update_count),
        np.asarray(before_updates),
    )


@pytest.mark.parametrize("executing_option", [False, True])
def test_valid_range_control_owner_tamper_rejects_transition_and_checkpoint(
    executing_option: bool,
    tmp_path: Path,
) -> None:
    agent = _agent("behavior_only")
    state = agent.start(agent.init(jr.key(31)), jnp.asarray([0.2], dtype=jnp.float32))
    state = _force_option(state) if executing_option else _force_idle(state)
    stomp = state.oak_state.stomp_state
    wrong_action = jnp.asarray(1, dtype=jnp.int32) - state.current_action
    stomp = cast(
        STOMPState,
        stomp.replace(
            option_last_intra_action=(
                wrong_action if executing_option else stomp.option_last_intra_action
            ),
            base_last_action=(
                stomp.base_last_action if executing_option else wrong_action
            ),
        ),
    )
    tampered = cast(
        PrototypeAgentState,
        state.replace(
            oak_state=cast(OaKState, state.oak_state.replace(stomp_state=stomp))
        ),
    )

    result = agent.update_transition(tampered, _transition(tampered, 0.1))

    assert not bool(result.transition_diagnostics.state_consistent)
    assert not bool(result.transition_diagnostics.valid)
    assert not bool(result.behavior_gradient_result.diagnostics.source_available)
    assert not bool(result.representation_gradient_mix.applied)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    _assert_tree_equal(result.state, tampered)
    with pytest.raises(ValueError, match="inconsistent PrototypeAgent state"):
        save_prototype_checkpoint(
            agent,
            tampered,
            tmp_path / ("option-owner" if executing_option else "base-owner"),
        )


def _complete_candidate_update_audit_evidence(
    state: PrototypeAgentState,
    parameter_count: int,
) -> PrototypeCandidateUpdateAuditEvidence:
    available = jnp.asarray(True, dtype=jnp.bool_)
    return PrototypeCandidateUpdateAuditEvidence(
        decision_id=state.current_decision_id,
        objective_probe_gradient=jnp.ones(parameter_count, dtype=jnp.float32),
        retention_probe_gradient=jnp.linspace(
            0.5,
            1.5,
            parameter_count,
            dtype=jnp.float32,
        ),
        safety_cost_gradient=jnp.ones(parameter_count, dtype=jnp.float32),
        objective_probe_available=available,
        retention_probe_available=available,
        safety_probe_available=available,
        probe_independence_attested=available,
        advantage=jnp.asarray(1.0, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_available=available,
        action_surprisal_available=available,
        safety_cost_available=available,
    )


def test_candidate_update_audit_checks_the_mixed_builder_candidate() -> None:
    agent = _agent("full", ensemble=True, candidate_audit=True)
    state = agent.start(agent.init(jr.key(5)), jnp.asarray([0.3], dtype=jnp.float32))
    state = _force_idle(state)
    state = _set_base_linear_heads(
        state,
        jnp.asarray([0.2, -0.4], dtype=jnp.float32),
    )
    builder = agent.state_builder
    assert builder is not None
    sidecar = _complete_candidate_update_audit_evidence(
        state,
        cast(OnlineGatedStateBuilderConfig, agent.config.state_builder).parameter_count(),
    )
    result = agent.update_transition(state, _transition(state, 0.15), sidecar)
    proposal = builder.propose_learning_update(
        state.state_builder_state,
        result.mixed_representation_gradient,
    )
    application = result.candidate_update_audit_application
    assert application is not None

    chex.assert_trees_all_close(
        application.assessment.candidate_update,
        proposal.candidate_parameter_update,
    )
    assert bool(result.candidate_update_audit_evidence_supplied)
    assert bool(result.candidate_update_audit_decision_id_matches)


def test_jit_scan_and_checkpoint_preserve_mixer_contract(tmp_path: Path) -> None:
    agent = _agent("behavior_only")
    initial = agent.start(agent.init(jr.key(6)), jnp.asarray([0.1], dtype=jnp.float32))
    initial = _force_idle(initial)
    initial = _set_base_linear_heads(
        initial,
        jnp.asarray([0.3, 0.1], dtype=jnp.float32),
    )
    first_transition = _transition(initial, 0.2)
    first = agent.update_transition(initial, first_transition)
    second_transition = _transition(first.state, -0.1, reward=0.2)
    transitions = jax.tree.map(
        lambda first_value, second_value: (
            None
            if first_value is None
            else jnp.stack((first_value, second_value))
        ),
        first_transition,
        second_transition,
        is_leaf=lambda value: value is None,
    )
    scanned = jax.jit(agent.scan_transitions)(initial, transitions)
    assert bool(jnp.all(scanned.transition_valid))
    assert bool(jnp.all(scanned.state_builder_learning_applied))

    checkpoint = tmp_path / "balanced-prototype"
    save_prototype_checkpoint(agent, scanned.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_equal(restored_state, scanned.state)
