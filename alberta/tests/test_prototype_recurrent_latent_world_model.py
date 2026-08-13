# mypy: disable-error-code="attr-defined,call-arg"
"""Causal Prototype integration for the recurrent latent world-model lane."""

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

from alberta_framework.core.delight import GradientJoyConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeRecurrentLatentDiagnostics,
    PrototypeRecurrentLatentWorldModelState,
    PrototypeTransition,
    PrototypeUpdateResult,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentTransitionRecord,
    RecurrentLatentWorldModelEnsembleConfig,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilderConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = pytest.mark.unit

OBSERVATION_DIM = 2
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_jit_and_scan_match_sequential_recurrent_lifecycle":
        yield
    else:
        with jax.disable_jit():
            yield


def _model_config(
    *,
    observation_dim: int = OBSERVATION_DIM,
    n_actions: int = N_ACTIONS,
    warmup: int = 1,
    max_updates: int = 100,
) -> RecurrentLatentWorldModelEnsembleConfig:
    return RecurrentLatentWorldModelEnsembleConfig(
        observation_dim=observation_dim,
        n_actions=n_actions,
        latent_dim=2,
        ensemble_size=2,
        learning_rate=0.01,
        bootstrap_probability=0.7,
        uncertainty_warmup_steps=warmup,
        initialization_scale=0.1,
        max_updates=max_updates,
    )


def _oak_config(*, observation_dim: int = OBSERVATION_DIM) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=100.0),),
            observation_dim=observation_dim,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _agent(
    *,
    warmup: int = 1,
    max_updates: int = 100,
    online_builder: bool = False,
    mixer: bool = False,
    candidate_audit: bool = False,
) -> PrototypeAgent:
    state_builder = (
        OnlineGatedStateBuilderConfig(
            observation_dim=1,
            n_actions=N_ACTIONS,
            hidden_dim=1,
            include_raw_observation=True,
            step_size=0.05,
            gradient_clip=100.0,
        )
        if online_builder
        else None
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=state_builder,
            recurrent_latent_world_model_ensemble=_model_config(
                warmup=warmup,
                max_updates=max_updates,
            ),
            representation_gradient_mixer=(
                RepresentationGradientMixerConfig(
                    representation_dim=OBSERVATION_DIM,
                    mode="world_only",
                    behavior_weight=0.0,
                    grounded_world_weight=1.0,
                )
                if mixer
                else None
            ),
            gradient_joy=(
                GradientJoyConfig(
                    candidate_semantics="update",
                    max_update_norm=100.0,
                )
                if candidate_audit
                else None
            ),
        )
    )


def _initial_observation(*, online_builder: bool = False) -> jax.Array:
    values = [0.2] if online_builder else [0.2, -0.1]
    return jnp.asarray(values, dtype=jnp.float32)


def _next_observation(value: float, *, online_builder: bool = False) -> jax.Array:
    values = [value] if online_builder else [value, -0.5 * value]
    return jnp.asarray(values, dtype=jnp.float32)


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.3,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    next_decision_observation: jax.Array | None = None,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=(
            next_observation if next_decision_observation is None else next_decision_observation
        ),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert cast(Any, left_structure) == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _recurrent_diagnostics(
    result: PrototypeUpdateResult,
) -> PrototypeRecurrentLatentDiagnostics:
    diagnostics = result.recurrent_latent_world_model_diagnostics
    assert isinstance(diagnostics, PrototypeRecurrentLatentDiagnostics)
    return diagnostics


def test_config_round_trip_mutual_exclusion_dimensions_and_legacy_shape() -> None:
    agent = _agent()
    restored = PrototypeAgent.from_config(agent.to_config())
    assert restored.to_config() == agent.to_config()
    assert restored.recurrent_latent_world_model_ensemble is not None

    default_config = PrototypeAgentConfig().to_config()
    assert "recurrent_latent_world_model_ensemble" not in default_config
    default_state = PrototypeAgent(PrototypeAgentConfig()).init(jr.key(0))
    assert default_state.world_model_state is None
    assert [
        field.name
        for field in dataclasses.fields(cast(Any, PrototypeAgentState))
    ] == [
        "oak_state",
        "world_model_state",
        "buffer_state",
        "horde_state",
        "ia_state",
        "gru_state",
        "state_builder_state",
        "current_raw_observation",
        "current_representation",
        "current_action",
        "current_decision_id",
        "started",
        "observation_event_count",
        "step_count",
    ]

    legacy = ActionConditionedWorldModelConfig(
        observation_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            world_model=legacy,
            recurrent_latent_world_model_ensemble=_model_config(),
        )
    with pytest.raises(ValueError, match="observation_dim"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            recurrent_latent_world_model_ensemble=_model_config(observation_dim=3),
        )
    with pytest.raises(ValueError, match="n_actions"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            recurrent_latent_world_model_ensemble=_model_config(n_actions=3),
        )
    with pytest.raises(ValueError, match="legacy world_model"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            recurrent_latent_world_model_ensemble=_model_config(),
            n_dreams_per_step=1,
        )


def test_start_binds_exact_dispatched_representation_action_and_strict_cache() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(3)), _initial_observation())
    wrapper = cast(PrototypeRecurrentLatentWorldModelState, state.world_model_state)
    model = agent.recurrent_latent_world_model_ensemble
    assert model is not None
    expected = model.decide(
        wrapper.model_state,
        model.start(wrapper.model_state, state.current_representation),
        state.current_action,
    )
    _assert_tree_equal(wrapper.decision_cache, expected)
    assert bool(wrapper.decision_cache.valid)
    assert int(wrapper.decision_cache.owner_event_count) == 0

    tampered_cache = wrapper.decision_cache.replace(observation=jnp.zeros((3,), dtype=jnp.float32))
    tampered = state.replace(world_model_state=wrapper.replace(decision_cache=tampered_cache))
    with pytest.raises(ValueError, match="decision_cache"):
        agent.decision(tampered)


def test_transition_matches_standalone_update_and_caches_next_decision_once() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(5)), _initial_observation())
    wrapper = cast(PrototypeRecurrentLatentWorldModelState, state.world_model_state)
    model = agent.recurrent_latent_world_model_ensemble
    assert model is not None
    transition = _transition(state, _next_observation(0.4))
    expected = model.update(
        wrapper.model_state,
        wrapper.decision_cache,
        RecurrentLatentTransitionRecord(
            observation=state.current_representation,
            action=state.current_action,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            bootstrap_observation=transition.next_observation,
            next_decision_observation=transition.next_decision_observation,
        ),
    )

    result = agent.update_transition(state, transition)
    next_wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        result.state.world_model_state,
    )
    expected_cache = model.decide(
        expected.state,
        expected.next_start_cache,
        result.action,
    )
    assert bool(result.transition_diagnostics.valid)
    recurrent_diagnostics = _recurrent_diagnostics(result)
    assert bool(recurrent_diagnostics.transaction_applied)
    assert bool(recurrent_diagnostics.next_decision_cached)
    _assert_tree_equal(next_wrapper.model_state, expected.state)
    _assert_tree_equal(next_wrapper.decision_cache, expected_cache)
    assert int(next_wrapper.model_state.event_count) == 1
    assert int(next_wrapper.model_state.recurrent_advance_count) == 1
    assert int(next_wrapper.signal_state.step_count) == 1
    chex.assert_trees_all_close(
        result.world_model_representation_gradient,
        expected.representation_gradient,
    )
    chex.assert_trees_all_close(
        result.world_model_error,
        expected.mean_negative_log_likelihood,
    )


def test_boundary_target_is_final_observation_then_recurrent_context_resets() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(7)), _initial_observation())
    final_observation = jnp.asarray([0.8, -0.6], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.7, 0.5], dtype=jnp.float32)
    transition = _transition(
        state,
        final_observation,
        reward=1.0,
        discount=0.0,
        terminated=True,
        next_decision_observation=reset_observation,
    )
    result = agent.update_transition(state, transition)
    wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        result.state.world_model_state,
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(_recurrent_diagnostics(result).model.recurrent_reset)
    chex.assert_trees_all_equal(
        wrapper.model_state.member_hidden_states,
        jnp.zeros_like(wrapper.model_state.member_hidden_states),
    )
    chex.assert_trees_all_equal(
        wrapper.decision_cache.observation,
        reset_observation,
    )
    assert int(wrapper.model_state.boundary_count) == 1
    # The exact training target is available through the standalone model;
    # parameter parity distinguishes it from substituting the reset state.
    source_wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        state.world_model_state,
    )
    model = agent.recurrent_latent_world_model_ensemble
    assert model is not None
    expected = model.update(
        source_wrapper.model_state,
        source_wrapper.decision_cache,
        RecurrentLatentTransitionRecord(
            observation=state.current_representation,
            action=state.current_action,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            bootstrap_observation=final_observation,
            next_decision_observation=reset_observation,
        ),
    )
    _assert_tree_equal(wrapper.model_state, expected.state)


def test_uncertainty_warmup_and_raw_uncalibrated_disclosure_are_explicit() -> None:
    agent = _agent(warmup=2)
    state = agent.start(agent.init(jr.key(11)), _initial_observation())
    results = []
    for index in range(3):
        result = agent.update_transition(
            state,
            _transition(state, _next_observation(0.3 + 0.1 * index)),
        )
        assert bool(result.transition_diagnostics.valid)
        results.append(result)
        state = result.state

    assert not bool(results[0].learning_signals.availability.epistemic)
    assert not bool(results[1].learning_signals.availability.epistemic)
    assert bool(results[2].learning_signals.availability.epistemic)
    assert bool(results[2].learning_signals.availability.aleatoric)
    last_diagnostics = _recurrent_diagnostics(results[2])
    assert not bool(last_diagnostics.raw_uncertainty_calibrated)
    assert bool(last_diagnostics.prediction_availability.epistemic)


def test_recurrent_rejection_rolls_back_whole_prototype_transition() -> None:
    agent = _agent(max_updates=1)
    state = agent.start(agent.init(jr.key(13)), _initial_observation())
    first = agent.update_transition(
        state,
        _transition(state, _next_observation(0.4)),
    )
    assert bool(first.transition_diagnostics.valid)
    before_rejected = first.state
    rejected = agent.update_transition(
        before_rejected,
        _transition(before_rejected, _next_observation(0.6)),
    )

    assert not bool(rejected.transition_diagnostics.valid)
    assert bool(rejected.transition_diagnostics.rejected)
    rejected_diagnostics = _recurrent_diagnostics(rejected)
    assert not bool(rejected_diagnostics.model.capacity_available)
    assert not bool(rejected_diagnostics.transaction_applied)
    _assert_tree_equal(rejected.state, before_rejected)
    assert int(rejected.action) == int(before_rejected.current_action)


def test_world_nll_gradient_reaches_builder_and_missing_candidate_audit_vetoes_only_it() -> None:
    agent = _agent(online_builder=True, mixer=True, candidate_audit=True)
    state = agent.start(
        agent.init(jr.key(17)),
        _initial_observation(online_builder=True),
    )
    before_wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        state.world_model_state,
    )
    result = agent.update_transition(
        state,
        _transition(
            state,
            _next_observation(0.5, online_builder=True),
        ),
    )
    after_wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        result.state.world_model_state,
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.world_model_representation_gradient_valid)
    assert bool(result.representation_gradient_mix.applied)
    chex.assert_trees_all_close(
        result.mixed_representation_gradient,
        result.world_model_representation_gradient,
    )
    assert not bool(result.candidate_update_audit_passed)
    assert not bool(result.audited_candidate_update_applied)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 0
    # The candidate-update audit vetoes representation, never the accepted real model.
    assert int(after_wrapper.model_state.event_count) == (
        int(before_wrapper.model_state.event_count) + 1
    )


def test_world_nll_gradient_commits_builder_without_candidate_update_audit() -> None:
    agent = _agent(online_builder=True, mixer=True)
    state = agent.start(
        agent.init(jr.key(18)),
        _initial_observation(online_builder=True),
    )
    result = agent.update_transition(
        state,
        _transition(
            state,
            _next_observation(0.55, online_builder=True),
        ),
    )
    wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        result.state.world_model_state,
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 1
    assert bool(wrapper.decision_cache.valid)
    chex.assert_trees_all_equal(
        wrapper.decision_cache.observation,
        result.state.current_representation,
    )
    assert bool(agent.decision(result.state).armed)


def test_checkpoint_round_trip_preserves_wrapper_cache_rng_and_resources(
    tmp_path: Path,
) -> None:
    default_agent = PrototypeAgent(PrototypeAgentConfig())
    default_state = default_agent.start(
        default_agent.init(jr.key(18)),
        jnp.zeros((default_agent.config.oak.observation_dim,), dtype=jnp.float32),
    )
    default_path = tmp_path / "prototype-default"
    save_prototype_checkpoint(default_agent, default_state, default_path)
    restored_default_agent, restored_default_state = load_prototype_checkpoint(
        default_path
    )
    assert "recurrent_latent_world_model_ensemble" not in (
        restored_default_agent.to_config()
    )
    _assert_tree_equal(restored_default_state, default_state)

    agent = _agent()
    state = agent.start(agent.init(jr.key(19)), _initial_observation())
    result = agent.update_transition(
        state,
        _transition(state, _next_observation(0.4)),
    )
    path = tmp_path / "prototype-recurrent"
    save_prototype_checkpoint(agent, result.state, path)
    restored_agent, restored_state = load_prototype_checkpoint(path)

    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_equal(restored_state, result.state)
    wrapper = cast(
        PrototypeRecurrentLatentWorldModelState,
        restored_state.world_model_state,
    )
    model = restored_agent.recurrent_latent_world_model_ensemble
    assert model is not None
    budget = model.resource_budget(wrapper.model_state)
    assert budget.replay_capacity == 0
    assert budget.recurrent_advances_per_accepted_event == 1
    assert budget.member_gradient_candidates_per_event == 2
    assert int(wrapper.model_state.event_count) == int(wrapper.signal_state.step_count)


def test_jit_and_scan_match_sequential_recurrent_lifecycle() -> None:
    agent = _agent()
    initial = _initial_observation()
    state = agent.start(agent.init(jr.key(23)), initial)
    transition = _transition(state, _next_observation(0.4))
    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)
    chex.assert_trees_all_close(
        _materialize_keys(compiled),
        _materialize_keys(eager),
    )

    rewards = jnp.asarray([0.3, -0.1], dtype=jnp.float32)
    observations = jnp.stack(
        (_next_observation(0.4), _next_observation(0.6)),
        axis=0,
    )
    discounts = jnp.asarray([0.9, 0.8], dtype=jnp.float32)
    scan_state = agent.start(agent.init(jr.key(29)), initial)
    compiled_scan = jax.jit(agent.scan)(
        scan_state,
        rewards,
        observations,
        discounts=discounts,
    )
    sequential_state = scan_state
    sequential_valid = []
    for reward, observation, discount in zip(
        rewards,
        observations,
        discounts,
        strict=True,
    ):
        step = agent.update_transition(
            sequential_state,
            _transition(
                sequential_state,
                observation,
                reward=float(reward),
                discount=float(discount),
            ),
        )
        sequential_state = step.state
        sequential_valid.append(step.transition_diagnostics.valid)
    chex.assert_trees_all_close(
        _materialize_keys(compiled_scan.state),
        _materialize_keys(sequential_state),
    )
    chex.assert_trees_all_equal(
        compiled_scan.transition_valid,
        jnp.stack(sequential_valid),
    )


def test_plain_ensemble_and_recurrent_lane_are_mutually_exclusive() -> None:
    # A small malformed composition test also protects the lane count when new
    # model variants are added later.
    plain_ensemble = WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=OBSERVATION_DIM,
            n_actions=N_ACTIONS,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=OBSERVATION_DIM + 2,
        ),
        ensemble_size=2,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            world_model_ensemble=plain_ensemble,
            recurrent_latent_world_model_ensemble=_model_config(),
        )
