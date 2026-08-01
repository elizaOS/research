"""Causal PrototypeAgent integration for the bounded world-model ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.unit

OBSERVATION_DIM = 2
N_ACTIONS = 2


@chex.dataclass(frozen=True)
class _LegacyWorldModelEnsembleState:
    """Exact Prototype v2 ensemble subtree fixture."""

    member_states: tuple[Any, ...]
    residual_variances: Any
    signal_state: Any
    bootstrap_key: jax.Array
    last_bootstrap_mask: Any
    member_update_counts: Any
    event_count: Any


def _ensemble_config(
    *,
    observation_dim: int = OBSERVATION_DIM,
    max_input_magnitude: float = 100.0,
) -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=observation_dim,
            n_actions=N_ACTIONS,
            gamma=0.95,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.8,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=observation_dim + 2,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            fast_loss_decay=0.5,
            slow_loss_decay=0.9,
            max_input_magnitude=max_input_magnitude,
            max_predicted_variance=max_input_magnitude**2,
            max_observed_loss=max_input_magnitude**2,
        ),
        ensemble_size=2,
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
    )


def _agent(*, max_input_magnitude: float = 100.0) -> PrototypeAgent:
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=OBSERVATION_DIM,
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            state_builder=IdentityStateBuilderConfig(observation_dim=OBSERVATION_DIM),
            world_model_ensemble=_ensemble_config(max_input_magnitude=max_input_magnitude),
        )
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.4,
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
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
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
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_ensemble_config_is_strict_dimensioned_and_round_trips() -> None:
    agent = _agent()
    restored = PrototypeAgent.from_config(agent.to_config())
    assert restored.to_config() == agent.to_config()
    assert restored.world_model_ensemble is not None

    legacy_model = ActionConditionedWorldModelConfig(
        observation_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            world_model=legacy_model,
            world_model_ensemble=_ensemble_config(),
        )

    with pytest.raises(ValueError, match="observation_dim"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            world_model_ensemble=_ensemble_config(observation_dim=3),
        )

    with pytest.raises(ValueError, match="legacy world_model"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            world_model_ensemble=_ensemble_config(),
            n_dreams_per_step=1,
        )


def test_transition_matches_standalone_preupdate_ensemble_transaction() -> None:
    agent = _agent()
    ensemble = agent.world_model_ensemble
    assert ensemble is not None
    state = agent.start(
        agent.init(jr.key(7)),
        jnp.asarray([0.2, -0.3], dtype=jnp.float32),
    )
    next_observation = jnp.asarray([0.5, 0.1], dtype=jnp.float32)
    transition = _transition(state, next_observation)

    expected = ensemble.update(
        state.world_model_state,
        state.current_representation,
        state.current_action,
        transition.reward,
        transition.discount,
        next_observation,
    )
    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.world_model_ensemble_diagnostics.applied)
    assert bool(result.world_model_representation_gradient_valid)
    _assert_tree_equal(result.state.world_model_state, expected.state)
    chex.assert_trees_all_close(result.learning_signals, expected.signals)
    chex.assert_trees_all_close(
        result.world_model_representation_gradient,
        expected.representation_gradient,
    )
    chex.assert_trees_all_close(result.world_model_error, expected.observed_loss)


def test_boundary_model_target_uses_final_state_not_reset_decision_state() -> None:
    agent = _agent()
    ensemble = agent.world_model_ensemble
    assert ensemble is not None
    state = agent.start(
        agent.init(jr.key(11)),
        jnp.asarray([-0.2, 0.4], dtype=jnp.float32),
    )
    final_observation = jnp.asarray([0.7, -0.6], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.9, 0.8], dtype=jnp.float32)
    transition = _transition(
        state,
        final_observation,
        reward=1.0,
        discount=0.0,
        terminated=True,
        next_decision_observation=reset_observation,
    )
    expected = ensemble.update(
        state.world_model_state,
        state.current_representation,
        state.current_action,
        transition.reward,
        transition.discount,
        final_observation,
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    _assert_tree_equal(result.state.world_model_state, expected.state)
    chex.assert_trees_all_equal(
        result.state.current_raw_observation,
        reset_observation,
    )
    chex.assert_trees_all_equal(
        result.state.current_representation,
        reset_observation,
    )


def test_internal_ensemble_rejection_does_not_poison_or_rollback_control() -> None:
    agent = _agent(max_input_magnitude=2.0)
    state = agent.start(
        agent.init(jr.key(19)),
        jnp.asarray([0.1, -0.1], dtype=jnp.float32),
    )
    transition = _transition(
        state,
        jnp.asarray([0.2, -0.2], dtype=jnp.float32),
        reward=3.0,
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.world_model_ensemble_diagnostics.applied)
    assert bool(result.world_model_ensemble_diagnostics.rejected)
    assert not bool(result.world_model_representation_gradient_valid)
    assert not bool(result.learning_signals.availability.input_valid)
    chex.assert_trees_all_equal(
        result.world_model_representation_gradient,
        jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32),
    )
    _assert_tree_equal(result.state.world_model_state, state.world_model_state)
    assert int(result.state.step_count) == int(state.step_count) + 1


def test_ensemble_transition_matches_explicit_jit_and_checkpoint_resume(
    tmp_path: Path,
) -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(23)),
        jnp.asarray([0.3, 0.6], dtype=jnp.float32),
    )
    transition = _transition(
        state,
        jnp.asarray([-0.1, 0.4], dtype=jnp.float32),
    )

    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)
    _assert_tree_equal(compiled, eager)

    checkpoint = tmp_path / "prototype-ensemble"
    save_prototype_checkpoint(agent, eager.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_equal(restored_state, eager.state)


def test_prototype_v2_ensemble_checkpoint_migrates_only_isolated_replay_state(
    tmp_path: Path,
) -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(31)),
        jnp.asarray([0.4, -0.2], dtype=jnp.float32),
    )
    state = agent.update_transition(
        state,
        _transition(state, jnp.asarray([0.1, 0.7], dtype=jnp.float32)),
    ).state
    current_world = cast(WorldModelEnsembleState, state.world_model_state)
    legacy_world = _LegacyWorldModelEnsembleState(
        member_states=current_world.member_states,
        residual_variances=current_world.residual_variances,
        signal_state=current_world.signal_state,
        bootstrap_key=current_world.bootstrap_key,
        last_bootstrap_mask=current_world.last_bootstrap_mask,
        member_update_counts=current_world.member_update_counts,
        event_count=current_world.event_count,
    )
    legacy_state = state.replace(world_model_state=legacy_world)
    config = agent.to_config()
    config_digest = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    checkpoint = tmp_path / "prototype-v2-ensemble"
    save_checkpoint(
        legacy_state,
        checkpoint,
        metadata={
            "schema": "alberta.prototype_agent.v2",
            "agent_config": config,
            "config_sha256": config_digest,
        },
    )

    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    restored_world = cast(WorldModelEnsembleState, restored_state.world_model_state)
    assert PROTOTYPE_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v3"
    assert restored_agent.to_config() == config
    _assert_tree_equal(restored_world.member_states, legacy_world.member_states)
    _assert_tree_equal(restored_world.residual_variances, legacy_world.residual_variances)
    _assert_tree_equal(restored_world.signal_state, legacy_world.signal_state)
    _assert_tree_equal(restored_world.bootstrap_key, legacy_world.bootstrap_key)
    _assert_tree_equal(
        restored_world.last_bootstrap_mask,
        legacy_world.last_bootstrap_mask,
    )
    _assert_tree_equal(
        restored_world.member_update_counts,
        legacy_world.member_update_counts,
    )
    _assert_tree_equal(restored_world.event_count, legacy_world.event_count)
    np.testing.assert_array_equal(
        restored_world.last_replay_bootstrap_mask,
        np.zeros((2,), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        restored_world.replay_member_update_counts,
        np.zeros((2,), dtype=np.int32),
    )
    assert int(restored_world.replay_event_count) == 0
    assert not bool(
        jnp.array_equal(
            jr.key_data(restored_world.replay_bootstrap_key),
            jr.key_data(restored_world.bootstrap_key),
        )
    )
    assert bool(restored_agent._checkpoint_state_valid(restored_state))

    migrated_checkpoint = tmp_path / "prototype-v3-migrated"
    save_prototype_checkpoint(restored_agent, restored_state, migrated_checkpoint)
    reloaded_agent, reloaded_state = load_prototype_checkpoint(migrated_checkpoint)
    assert reloaded_agent.to_config() == config
    _assert_tree_equal(reloaded_state, restored_state)
