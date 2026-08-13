# mypy: disable-error-code="attr-defined,call-arg"
"""Prototype integration for bounded model-only dual-replay rehearsal."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.delight import GradientJoyConfig
from alberta_framework.core.dual_replay import DualReplayConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    StateBuilderConfig,
    state_builder_from_config,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = pytest.mark.unit

N_ACTIONS = 2
FEATURE_DIM = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_jit_scan_and_prototype_checkpoint_preserve_composition":
        yield
    else:
        with jax.disable_jit():
            yield


def _ensemble_config(*, max_input_magnitude: float = 100.0) -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=FEATURE_DIM,
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
            target_dim=FEATURE_DIM + 2,
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


def _rehearsal_config(*, max_input_magnitude: float = 100.0) -> ModelReplayRehearsalConfig:
    return ModelReplayRehearsalConfig(
        ensemble=_ensemble_config(max_input_magnitude=max_input_magnitude),
        replay=DualReplayConfig(
            total_capacity=4,
            short_term_capacity=2,
            observation_dim=FEATURE_DIM,
            action_dim=N_ACTIONS,
            short_term_sample_size=1,
            long_term_sample_size=1,
            long_term_policy="reservoir",
            max_representation_lag=0,
        ),
        action_encoding="one_hot",
    )


def _identity_builder() -> IdentityStateBuilderConfig:
    return IdentityStateBuilderConfig(observation_dim=FEATURE_DIM)


def _online_builder() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=1,
        n_actions=N_ACTIONS,
        hidden_dim=1,
        include_raw_observation=True,
        step_size=0.1,
        gradient_clip=10.0,
    )


def _agent(
    *,
    builder: StateBuilderConfig | None = None,
    learn_builder: bool = False,
    candidate_audit: bool = False,
    max_input_magnitude: float = 100.0,
) -> PrototypeAgent:
    resolved_builder = _identity_builder() if builder is None else builder
    feature_dim = state_builder_from_config(resolved_builder.to_config()).feature_dim()
    assert feature_dim == FEATURE_DIM
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=feature_dim,
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            state_builder=resolved_builder,
            model_replay_rehearsal=_rehearsal_config(
                max_input_magnitude=max_input_magnitude
            ),
            learn_state_builder_from_world_model=learn_builder,
            gradient_joy=(
                GradientJoyConfig(
                    candidate_semantics="update",
                    max_update_norm=10.0,
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
            next_observation
            if next_decision_observation is None
            else next_decision_observation
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


def test_config_round_trip_and_model_lane_exclusivity() -> None:
    agent = _agent()
    restored = PrototypeAgent.from_config(agent.to_config())
    assert restored.to_config() == agent.to_config()
    assert restored.model_replay_rehearsal is not None
    assert restored.world_model_ensemble is restored.model_replay_rehearsal.ensemble

    with pytest.raises(ValueError, match="mutually exclusive"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            world_model_ensemble=_ensemble_config(),
            model_replay_rehearsal=_rehearsal_config(),
        )
    with pytest.raises(ValueError, match="legacy world_model"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            model_replay_rehearsal=_rehearsal_config(),
            n_dreams_per_step=1,
        )


def test_transition_matches_standalone_atomic_composer() -> None:
    agent = _agent()
    composer = agent.model_replay_rehearsal
    assert composer is not None
    state = agent.start(
        agent.init(jr.key(1)),
        jnp.asarray([0.2, -0.3], dtype=jnp.float32),
    )
    next_observation = jnp.asarray([0.5, 0.1], dtype=jnp.float32)
    transition = _transition(state, next_observation)
    expected = composer.step(
        state.world_model_state,
        RealModelReplayEvent(
            observation=state.current_representation,
            action=state.current_action,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            next_observation=next_observation,
            representation_version=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=state.step_count,
            source_id=jnp.asarray(0, dtype=jnp.int32),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(False),
            valid=jnp.asarray(True),
        ),
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.model_replay_transaction_applied)
    assert bool(result.model_replay_recorded)
    assert bool(result.model_replay_sampled)
    assert int(result.model_replay_updates_applied) == 2
    assert int(result.model_replay_padding_count) == 0
    assert bool(result.world_model_ensemble_diagnostics.applied)
    _assert_tree_equal(result.state.world_model_state, expected.state)
    chex.assert_trees_all_close(result.learning_signals, expected.real_signals)
    chex.assert_trees_all_close(
        result.world_model_representation_gradient,
        expected.real_representation_gradient,
    )
    chex.assert_trees_all_close(result.world_model_error, expected.real_observed_loss)
    assert int(result.state.step_count) == 1
    assert int(result.state.world_model_state.accepted_real_event_count) == 1
    assert int(result.state.world_model_state.rehearsal_applied_count) == 2
    budget = composer.resource_budget(result.state.world_model_state)
    assert budget.max_actor_updates_per_event == 0
    assert budget.max_critic_updates_per_event == 0
    assert budget.max_state_builder_updates_per_event == 0


def test_truncation_replay_target_is_final_not_reset_observation() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(2)),
        jnp.asarray([-0.2, 0.4], dtype=jnp.float32),
    )
    final_observation = jnp.asarray([0.7, -0.6], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.9, 0.8], dtype=jnp.float32)
    result = agent.update_transition(
        state,
        _transition(
            state,
            final_observation,
            truncated=True,
            next_decision_observation=reset_observation,
        ),
    )

    replay = result.state.world_model_state.replay_state.short_term
    np.testing.assert_array_equal(replay.next_observations[0], final_observation)
    assert bool(replay.truncated[0])
    assert not bool(replay.terminated[0])
    assert float(replay.discounts[0]) > 0.0
    np.testing.assert_array_equal(result.state.current_representation, reset_observation)


def test_legacy_wrapper_uses_composed_ensemble_gamma() -> None:
    configured = _agent()
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=configured.config.oak,
            model_replay_rehearsal=_rehearsal_config(),
        )
    )
    state = agent.start(
        agent.init(jr.key(22)),
        jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    )
    result = agent.update(
        state,
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray([0.3, 0.2], dtype=jnp.float32),
    )
    replay = result.state.world_model_state.replay_state.short_term
    assert float(replay.discounts[0]) == pytest.approx(0.95)


def test_rehearsal_rejection_is_accounted_without_rolling_back_control() -> None:
    agent = _agent(max_input_magnitude=1.0)
    state = agent.start(
        agent.init(jr.key(3)),
        jnp.asarray([0.1, -0.1], dtype=jnp.float32),
    )
    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.2, -0.2], dtype=jnp.float32),
            reward=2.0,
        ),
    )

    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.model_replay_transaction_applied)
    assert not bool(result.model_replay_recorded)
    assert not bool(result.model_replay_sampled)
    assert int(result.model_replay_updates_applied) == 0
    assert float(result.world_model_error) == 0.0
    assert not bool(result.learning_signals.availability.input_valid)
    assert int(result.state.step_count) == int(state.step_count) + 1
    composer_state = result.state.world_model_state
    assert int(composer_state.real_attempt_count) == 1
    assert int(composer_state.accepted_real_event_count) == 0
    assert int(composer_state.rejected_real_event_count) == 1
    assert int(composer_state.ensemble_state.event_count) == 0
    assert int(composer_state.replay_state.accepted_transition_count) == 0


@pytest.mark.parametrize("candidate_audit_enabled", [False, True])
def test_only_real_gradient_can_reach_builder_and_candidate_audit_fails_closed(
    candidate_audit_enabled: bool,
) -> None:
    agent = _agent(
        builder=_online_builder(),
        learn_builder=True,
        candidate_audit=candidate_audit_enabled,
    )
    state = agent.start(
        agent.init(jr.key(4)),
        jnp.asarray([0.25], dtype=jnp.float32),
    )
    result = agent.update_transition(
        state,
        _transition(state, jnp.asarray([-0.1], dtype=jnp.float32)),
    )

    assert bool(result.model_replay_transaction_applied)
    assert int(result.model_replay_updates_applied) == 2
    expected_builder_updates = 0 if candidate_audit_enabled else 1
    assert int(result.state.state_builder_state.update_count) == expected_builder_updates
    assert bool(result.state_builder_learning_diagnostics.applied) is (
        not candidate_audit_enabled
    )
    if candidate_audit_enabled:
        assert not bool(result.candidate_update_audit_passed)
        assert not bool(result.audited_candidate_update_applied)
    assert int(result.state.world_model_state.ensemble_state.replay_event_count) == 2


def test_jit_scan_and_prototype_checkpoint_preserve_composition(tmp_path: Path) -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(5)),
        jnp.asarray([0.3, 0.6], dtype=jnp.float32),
    )
    first = _transition(
        state,
        jnp.asarray([-0.1, 0.4], dtype=jnp.float32),
    )
    eager_first = agent.update_transition(state, first)
    compiled_first = jax.jit(agent.update_transition)(state, first)
    _assert_tree_equal(compiled_first, eager_first)

    second = _transition(
        eager_first.state,
        jnp.asarray([0.2, -0.5], dtype=jnp.float32),
    )
    sequential = agent.update_transition(eager_first.state, second)
    stacked = jax.tree.map(lambda left, right: jnp.stack((left, right)), first, second)
    scanned = agent.scan_transitions(state, stacked)
    _assert_tree_equal(scanned.state, sequential.state)
    assert int(scanned.state.world_model_state.accepted_real_event_count) == 2
    assert int(scanned.state.world_model_state.ensemble_state.replay_event_count) == 4

    checkpoint = tmp_path / "prototype-model-replay"
    save_prototype_checkpoint(agent, sequential.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    assert restored_agent.model_replay_rehearsal is not None
    _assert_tree_equal(restored_state, sequential.state)
