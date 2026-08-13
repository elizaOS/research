# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def,override"
"""Stable-base world modeling under Prototype pair-feature curation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta_api
import alberta_framework.core as core_api
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
    PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeFeatureOaKState,
    PrototypeFeatureRepresentationState,
    PrototypeFeatureWorldModelState,
    PrototypeTransition,
    load_prototype_checkpoint,
    measure_prototype_agent_state_resources,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    PrototypeFeatureLifecycleResult,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    measure_action_conditioned_world_model_state_nbytes,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_DIM = 4
PAIR_SLOTS = 2
TOTAL_DIM = BASE_DIM + PAIR_SLOTS
N_ACTIONS = 2
BUFFER_CAPACITY = 3


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_forced_curation_has_eager_and_jit_scan_parity":
        yield
    else:
        with jax.disable_jit():
            yield


def _feature_config(*, replacement_interval: int = 0) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=PAIR_SLOTS,
        candidate_pair_slots=6,
        n_tasks=1,
        n_options=2,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0, 1),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
    )


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=8),
                SubtaskSpec(feature_index=1, threshold=1.0e6, max_option_steps=8),
            ),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _world_config(
    *,
    observation_dim: int = BASE_DIM,
    include_action_interactions: bool = True,
) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=observation_dim,
        n_actions=N_ACTIONS,
        hidden_sizes=(),
        step_size=0.02,
        sparsity=0.0,
        use_layer_norm=False,
        include_action_interactions=include_action_interactions,
    )


def _config(
    *,
    replacement_interval: int = 0,
    state_builder: Any = None,
    world_model: ActionConditionedWorldModelConfig | None = None,
    dreaming: DreamingConfig | None = None,
    n_dreams_per_step: int = 0,
) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=_oak_config(),
        state_builder=(
            IdentityStateBuilderConfig(observation_dim=BASE_DIM)
            if state_builder is None
            else state_builder
        ),
        prototype_feature_lifecycle=_feature_config(
            replacement_interval=replacement_interval
        ),
        world_model=_world_config() if world_model is None else world_model,
        dreaming=dreaming,
        n_dreams_per_step=n_dreams_per_step,
        buffer_capacity=BUFFER_CAPACITY,
    )


def _wrapper(state: PrototypeAgentState) -> PrototypeFeatureRepresentationState:
    assert type(state.state_builder_state) is PrototypeFeatureRepresentationState
    return state.state_builder_state


def _bound_oak(state: PrototypeAgentState) -> PrototypeFeatureOaKState:
    assert type(state.oak_state) is PrototypeFeatureOaKState
    return state.oak_state


def _oak(state: PrototypeAgentState) -> OaKState:
    return _bound_oak(state).oak_state


def _world_state(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> Any:
    return agent._action_world_model_component_state(state.world_model_state)


def _start_idle(agent: PrototypeAgent, observation: jax.Array) -> PrototypeAgentState:
    for seed in range(32):
        state = agent.start(agent.init(jr.key(seed)), observation)
        if int(_oak(state).stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not obtain an idle initial Prototype decision")


def _transition(
    state: PrototypeAgentState,
    bootstrap: jax.Array,
    *,
    decision: jax.Array | None = None,
    terminated: bool = False,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.75, dtype=jnp.float32),
        discount=jnp.asarray(0.0 if terminated else 0.9, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=bootstrap,
        next_decision_observation=bootstrap if decision is None else decision,
    )


def _force_promotion(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    wrapper = _wrapper(state)
    feature_state = wrapper.feature_lifecycle_state
    learner = feature_state.learner_state
    active = set(
        zip(
            np.asarray(learner.feature_left).tolist(),
            np.asarray(learner.feature_right).tolist(),
            strict=True,
        )
    )
    candidates = list(
        zip(
            np.asarray(learner.candidate_left).tolist(),
            np.asarray(learner.candidate_right).tolist(),
            strict=True,
        )
    )
    candidate_index = next(
        index for index, pair in enumerate(candidates) if pair not in active
    )
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner = learner.replace(
        utilities=jnp.asarray((0.0, 0.5), dtype=jnp.float32),
        candidate_utilities=candidate_utilities,
    )
    feature_state = feature_state.replace(learner_state=learner)
    assert bool(lifecycle.state_valid(feature_state))
    return cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature_state
            )
        ),
    )


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


class _RejectingFeatureLifecycle(PrototypeFeatureLifecycle):
    def observe_and_route(
        self,
        state,
        oak_state,
        consumer_binding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        curation_priority_override=None,
    ) -> PrototypeFeatureLifecycleResult:
        result = super().observe_and_route(
            state,
            oak_state,
            consumer_binding,
            event,
            curation_priority_override=curation_priority_override,
        )
        return cast(
            PrototypeFeatureLifecycleResult,
            result.replace(
                state=state,
                oak_state=oak_state,
                next_augmented_observation=self.augment(state, event.next_observation),
                diagnostics=result.diagnostics.replace(
                    learner_update_rejected=jnp.asarray(True, dtype=jnp.bool_),
                    transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                ),
            ),
        )


def test_config_accepts_only_exact_identity_and_stable_base_model() -> None:
    assert alberta_api.PrototypeFeatureWorldModelState is PrototypeFeatureWorldModelState
    assert core_api.PrototypeFeatureWorldModelState is PrototypeFeatureWorldModelState
    assert (
        alberta_api.PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES
        == core_api.PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES
        == PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES
    )

    class IdentitySubclass(IdentityStateBuilderConfig):
        pass

    @dataclasses.dataclass(frozen=True)
    class WorldConfigSubclass(ActionConditionedWorldModelConfig):
        extra: int = 7

    config = _config()
    encoded = config.to_config()
    assert PrototypeAgentConfig.from_config(encoded).to_config() == encoded

    with pytest.raises(ValueError, match="stable base_feature_dim"):
        _config(world_model=_world_config(observation_dim=TOTAL_DIM))
    with pytest.raises(ValueError, match="exact Identity"):
        _config(state_builder=IdentitySubclass(observation_dim=BASE_DIM))
    with pytest.raises(ValueError, match="exact ActionConditionedWorldModelConfig"):
        _config(
            world_model=WorldConfigSubclass(
                observation_dim=BASE_DIM,
                n_actions=N_ACTIONS,
                hidden_sizes=(),
                sparsity=0.0,
                use_layer_norm=False,
            )
        )
    with pytest.raises(ValueError, match="exact Identity"):
        _config(
            state_builder=OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=N_ACTIONS,
                hidden_dim=2,
                include_raw_observation=True,
            )
        )
    with pytest.raises(ValueError, match="dreaming is disabled"):
        _config(dreaming=DreamingConfig())
    with pytest.raises(ValueError, match="dreaming is disabled"):
        _config(n_dreams_per_step=1)


def test_model_updates_only_on_base_coordinates_without_curation() -> None:
    agent = PrototypeAgent(_config())
    initial_raw = jnp.asarray((1.0, -2.0, 0.5, 3.0), dtype=jnp.float32)
    state = _start_idle(agent, initial_raw)
    bootstrap = jnp.asarray((-1.0, 2.0, 4.0, 0.25), dtype=jnp.float32)
    transition = _transition(state, bootstrap)
    model = ActionConditionedWorldModel(cast(Any, agent.config.world_model))
    expected = model.update(
        _world_state(agent, state),
        state.current_raw_observation,
        transition.action,
        transition.reward,
        transition.discount,
        bootstrap,
    )

    result = agent.update_transition(state, transition)
    assert bool(result.transition_diagnostics.valid)
    _assert_tree_exact(_world_state(agent, result.state), expected.state)
    assert result.state.buffer_state.observations.shape == (
        BUFFER_CAPACITY,
        BASE_DIM,
    )
    np.testing.assert_array_equal(result.state.buffer_state.observations[0], bootstrap)


def test_forced_curation_preserves_the_direct_base_model_transaction() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _start_idle(
        agent,
        jnp.asarray((0.5, -0.25, 1.0, 2.0), dtype=jnp.float32),
    )
    state = _force_promotion(agent, state)
    bootstrap = jnp.asarray((2.0, 1.0, -1.0, 0.75), dtype=jnp.float32)
    transition = _transition(state, bootstrap)
    model = ActionConditionedWorldModel(cast(Any, agent.config.world_model))
    expected = model.update(
        _world_state(agent, state),
        state.current_raw_observation,
        transition.action,
        transition.reward,
        transition.discount,
        bootstrap,
    )
    old_generation = _bound_oak(state).consumer_binding.semantic_generation

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.lifecycle.curation_committed)
    assert int(_bound_oak(result.state).consumer_binding.semantic_generation) == (
        int(old_generation) + 1
    )
    _assert_tree_exact(_world_state(agent, result.state), expected.state)


def test_interaction_columns_and_autoreset_buffer_are_base_only() -> None:
    agent = PrototypeAgent(_config())
    model = cast(ActionConditionedWorldModel, agent._world_model)
    observation = jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float32)
    features = model.input_features(observation, jnp.asarray(1, dtype=jnp.int32))
    expected_interactions = jnp.asarray(
        (0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0),
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(features[:BASE_DIM], observation)
    np.testing.assert_array_equal(features[BASE_DIM : BASE_DIM + N_ACTIONS], (0.0, 1.0))
    np.testing.assert_array_equal(features[-BASE_DIM * N_ACTIONS :], expected_interactions)
    assert features.shape == (BASE_DIM + N_ACTIONS + BASE_DIM * N_ACTIONS,)

    state = _start_idle(agent, observation)
    bootstrap = jnp.asarray((-1.0, -2.0, -3.0, -4.0), dtype=jnp.float32)
    decision = jnp.asarray((4.0, 3.0, 2.0, 1.0), dtype=jnp.float32)
    result = agent.update_transition(
        state,
        _transition(state, bootstrap, decision=decision, terminated=True),
    )
    assert int(result.state.buffer_state.size) == 2
    np.testing.assert_array_equal(result.state.buffer_state.observations[0], bootstrap)
    np.testing.assert_array_equal(result.state.buffer_state.observations[1], decision)


def test_rejected_feature_transaction_rolls_back_model_and_buffer() -> None:
    agent = PrototypeAgent(_config())
    state = _start_idle(
        agent,
        jnp.asarray((0.2, -0.4, 0.6, -0.8), dtype=jnp.float32),
    )
    lifecycle_config = agent.config.prototype_feature_lifecycle
    assert lifecycle_config is not None
    agent._prototype_feature_lifecycle = _RejectingFeatureLifecycle(lifecycle_config)
    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray((0.7, 0.1, -0.5, 0.3), dtype=jnp.float32),
        ),
    )

    assert not bool(result.transition_diagnostics.valid)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert not bool(diagnostics.outer_transaction_committed)
    _assert_tree_exact(result.state, state)


def test_forced_curation_has_eager_and_jit_scan_parity() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_promotion(
        agent,
        _start_idle(
            agent,
            jnp.asarray((0.1, 0.2, -0.3, 0.4), dtype=jnp.float32),
        ),
    )
    transition = _transition(
        state,
        jnp.asarray((-0.2, 0.7, 0.5, -0.1), dtype=jnp.float32),
    )
    direct = agent.update_transition(state, transition)
    batched = jax.tree.map(
        lambda value: None if value is None else jnp.expand_dims(value, 0),
        transition,
        is_leaf=lambda value: value is None,
    )
    scanned = jax.jit(agent.scan_transitions)(state, batched)
    assert bool(jnp.all(scanned.transition_valid))
    _assert_tree_exact(scanned.state, direct.state)


def test_pristine_state_requires_empty_buffer_and_zero_model_clock(
    tmp_path: Path,
) -> None:
    agent = PrototypeAgent(_config())
    state = agent.init(jr.key(91))
    assert agent._buffer is not None
    assert agent._world_model is not None

    nonempty_buffer = cast(
        PrototypeAgentState,
        state.replace(
            buffer_state=agent._buffer.add(
                state.buffer_state,
                jnp.ones((BASE_DIM,), dtype=jnp.float32),
            )
        ),
    )
    assert not bool(agent._pristine_state_consistent(nonempty_buffer))
    with pytest.raises(RuntimeError, match="fresh unstarted"):
        agent.start(nonempty_buffer, jnp.zeros((BASE_DIM,), dtype=jnp.float32))

    advanced_model = agent._world_model.update(
        _world_state(agent, state),
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.ones((BASE_DIM,), dtype=jnp.float32),
    ).state
    advanced_state = cast(
        PrototypeAgentState,
        state.replace(
            world_model_state=agent._action_world_model_state_slot(
                advanced_model
            )
        ),
    )
    assert not bool(agent._pristine_state_consistent(advanced_state))
    with pytest.raises(RuntimeError, match="fresh unstarted"):
        agent.start(advanced_state, jnp.zeros((BASE_DIM,), dtype=jnp.float32))

    started = _start_idle(
        agent,
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
    )
    impossible_started_buffer = cast(
        PrototypeAgentState,
        started.replace(
            buffer_state=agent._buffer.add(
                started.buffer_state,
                jnp.ones((BASE_DIM,), dtype=jnp.float32),
            )
        ),
    )
    assert not bool(agent._checkpoint_state_valid(impossible_started_buffer))
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            agent,
            impossible_started_buffer,
            tmp_path / "impossible-started-buffer",
        )


def test_feature_world_model_refusal_rolls_back_the_complete_transaction() -> None:
    agent = PrototypeAgent(
        _config(
            world_model=dataclasses.replace(
                _world_config(),
                step_size=1.0e20,
            )
        )
    )
    state = _start_idle(
        agent,
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
    )
    first = agent.update_transition(
        state,
        _transition(state, jnp.ones((BASE_DIM,), dtype=jnp.float32)),
    )
    assert bool(first.transition_diagnostics.valid)
    source = first.state
    second_transition = _transition(
        source,
        -jnp.ones((BASE_DIM,), dtype=jnp.float32),
    )
    assert agent._world_model is not None
    direct_model_result = agent._world_model.update(
        _world_state(agent, source),
        source.current_raw_observation,
        source.current_action,
        second_transition.reward,
        second_transition.discount,
        second_transition.next_observation,
    )
    assert not bool(direct_model_result.update_applied)

    refused = agent.update_transition(source, second_transition)
    assert not bool(refused.transition_diagnostics.valid)
    _assert_tree_exact(refused.state, source)


def test_v17_checkpoint_and_resources_bind_the_composition(tmp_path: Path) -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_promotion(
        agent,
        _start_idle(agent, jnp.ones((BASE_DIM,), dtype=jnp.float32)),
    )
    before = measure_prototype_agent_state_resources(state)
    result = agent.update_transition(
        state,
        _transition(state, -jnp.ones((BASE_DIM,), dtype=jnp.float32)),
    )
    after = measure_prototype_agent_state_resources(result.state)
    assert before.world_model_bundle_nbytes == after.world_model_bundle_nbytes
    assert before.buffer_nbytes == after.buffer_nbytes == (
        4 * BUFFER_CAPACITY * BASE_DIM + 8
    )
    assert after.world_model_bundle_nbytes == (
        measure_action_conditioned_world_model_state_nbytes(
            _world_state(agent, result.state)
        )
        + PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES
    )

    checkpoint = tmp_path / "feature-world-v17"
    save_prototype_checkpoint(agent, result.state, checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["schema"] == PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_exact(restored_state, result.state)

    mismatched_agent = PrototypeAgent(
        _config(
            replacement_interval=1,
            world_model=dataclasses.replace(
                _world_config(),
                step_size=0.125,
            ),
        )
    )
    assert not bool(mismatched_agent._checkpoint_state_valid(result.state))
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            mismatched_agent,
            result.state,
            tmp_path / "feature-world-mismatched-config",
        )

    world_slot = result.state.world_model_state
    assert type(world_slot) is PrototypeFeatureWorldModelState
    model_state = world_slot.model_state
    head_optimizer_states = list(model_state.learner_state.head_optimizer_states)
    weight_optimizer, bias_optimizer = head_optimizer_states[0]
    head_optimizer_states[0] = (
        weight_optimizer.replace(step_size=jnp.asarray(0.125, dtype=jnp.float32)),
        bias_optimizer,
    )
    tampered_learner_state = model_state.learner_state.replace(
        head_optimizer_states=tuple(head_optimizer_states)
    )
    tampered_state = cast(
        PrototypeAgentState,
        result.state.replace(
            world_model_state=world_slot.replace(
                model_state=model_state.replace(
                    learner_state=tampered_learner_state
                )
            )
        ),
    )
    assert not bool(agent._checkpoint_state_valid(tampered_state))
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            agent,
            tampered_state,
            tmp_path / "feature-world-tampered-optimizer",
        )

    bad_metadata = dict(metadata)
    bad_metadata["schema"] = "alberta.prototype_agent.v13"
    bad_path = tmp_path / "feature-world-wrong-schema"
    save_checkpoint(result.state, bad_path, metadata=bad_metadata)
    with pytest.raises(ValueError, match="requires a v17"):
        load_prototype_checkpoint(bad_path)
