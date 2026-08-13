# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Prototype-owned integration of the bounded pair-feature lifecycle."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.option_search_control import OptionSearchControlConfig
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    check_option_terminated,
    compute_pseudo_reward,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeFeatureOaKState,
    PrototypeFeatureRepresentationState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    PrototypeFeatureLifecycleResult,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import (
    FixedTraceStateBuilderConfig,
    IdentityStateBuilderConfig,
    IdentityStateBuilderState,
    OnlineGatedStateBuilderConfig,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModelConfig,
)

pytestmark = pytest.mark.integration

BASE_DIM = 4
TOTAL_DIM = 6
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_jit_scan_and_checkpoint_round_trip_enabled_lane":
        yield
    else:
        with jax.disable_jit():
            yield


def _feature_config(
    *,
    replacement_interval: int = 0,
    max_observations: int = 100,
    n_tasks: int = 1,
    option_indices: tuple[int, ...] = (0, 1),
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=2,
        candidate_pair_slots=6,
        n_tasks=n_tasks,
        n_options=2,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=option_indices,
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=max_observations,
    )


def _oak_config(
    *,
    option_indices: tuple[int, ...] = (0, 1),
    hidden_sizes: tuple[int, ...] = (),
    observation_dim: int = TOTAL_DIM,
) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=tuple(
                SubtaskSpec(
                    feature_index=index,
                    threshold=1_000_000.0,
                    max_option_steps=8,
                )
                for index in option_indices
            ),
            observation_dim=observation_dim,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=hidden_sizes,
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _identity_agent(
    *,
    replacement_interval: int = 0,
    max_observations: int = 100,
) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(
                replacement_interval=replacement_interval,
                max_observations=max_observations,
            ),
        )
    )


def _online_agent(*, max_observations: int = 100) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=N_ACTIONS,
                hidden_dim=2,
                include_raw_observation=True,
                step_size=0.1,
                gradient_clip=100.0,
            ),
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=TOTAL_DIM,
                mode="behavior_only",
                behavior_weight=1.0,
                grounded_world_weight=0.0,
            ),
            prototype_feature_lifecycle=_feature_config(
                max_observations=max_observations,
            ),
        )
    )


def _start_idle(agent: PrototypeAgent, observation: jax.Array) -> PrototypeAgentState:
    for seed in range(32):
        state = agent.start(agent.init(jr.key(seed)), observation)
        if int(_oak(state).stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not obtain a deterministic idle initial decision")


def _force_next_extended_action(
    state: PrototypeAgentState,
    extended_action: int,
) -> PrototypeAgentState:
    bound_oak = _bound_oak(state)
    stomp = bound_oak.oak_state.stomp_state
    learner = stomp.base_learner_state
    weights = tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights)
    biases = tuple(
        jnp.full_like(bias, 100.0 if index == extended_action else -100.0)
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=weights,
            biases=biases,
        )
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=bound_oak.replace(
                oak_state=bound_oak.oak_state.replace(
                    stomp_state=stomp.replace(base_learner_state=learner)
                )
            )
        ),
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.5,
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
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _wrapper(state: PrototypeAgentState) -> PrototypeFeatureRepresentationState:
    assert isinstance(state.state_builder_state, PrototypeFeatureRepresentationState)
    return state.state_builder_state


def _bound_oak(state: PrototypeAgentState) -> PrototypeFeatureOaKState:
    assert type(state.oak_state) is PrototypeFeatureOaKState
    return state.oak_state


def _oak(state: PrototypeAgentState):
    return _bound_oak(state).oak_state


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
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
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
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(
                left_array,
                right_array,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        else:
            np.testing.assert_array_equal(left_array, right_array)


class _RejectingFeatureLifecycle(PrototypeFeatureLifecycle):
    """Failure injector for an unexpected owned-step rejection."""

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
                next_augmented_observation=self.augment(
                    state,
                    event.next_observation,
                ),
                diagnostics=result.diagnostics.replace(
                    learner_update_rejected=jnp.asarray(True, dtype=jnp.bool_),
                    transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                ),
            ),
        )


def test_config_round_trip_and_fail_closed_composition_contract() -> None:
    class MasqueradingIdentityConfig(IdentityStateBuilderConfig):
        def to_config(self) -> dict[str, Any]:
            return FixedTraceStateBuilderConfig(
                observation_dim=BASE_DIM,
                observation_decay_rates=(),
                action_decay_rates=(),
                outcome_decay_rates=(),
                include_raw_observation=True,
            ).to_config()

    config = _identity_agent().config
    encoded = config.to_config()
    assert "prototype_feature_lifecycle" in encoded
    assert PrototypeAgentConfig.from_config(encoded).to_config() == encoded

    with pytest.raises(ValueError, match="n_tasks must equal 1"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(n_tasks=2),
        )
    with pytest.raises(ValueError, match="exactly match"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(
                option_indices=(1, 0)
            ),
        )
    bool_index_stomp = dataclasses.replace(
        _oak_config().stomp,
        subtask_specs=(
            SubtaskSpec(feature_index=False),
            SubtaskSpec(feature_index=True),
        ),
    )
    with pytest.raises(ValueError, match="actual OaK config"):
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=bool_index_stomp),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
        )
    float_width_stomp = dataclasses.replace(
        _oak_config().stomp,
        observation_dim=cast(int, float(TOTAL_DIM)),
    )
    with pytest.raises(ValueError, match="actual OaK config"):
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=float_width_stomp),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="linear OaK"):
        PrototypeAgentConfig(
            oak=_oak_config(hidden_sizes=(3,)),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="Identity or OnlineGated"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="Identity or OnlineGated"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=MasqueradingIdentityConfig(
                observation_dim=BASE_DIM,
            ),
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="behavior_only or discard"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=N_ACTIONS,
                hidden_dim=2,
            ),
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=TOTAL_DIM,
                mode="full",
                behavior_weight=1.0,
                grounded_world_weight=0.0,
            ),
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="auto_curate_every == 0"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            auto_curate_every=1,
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="stable base_feature_dim"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            world_model=ActionConditionedWorldModelConfig(
                observation_dim=TOTAL_DIM,
                n_actions=N_ACTIONS,
                hidden_sizes=(),
            ),
            prototype_feature_lifecycle=_feature_config(),
        )


def test_online_target_pullback_update_and_max_capacity_freeze() -> None:
    agent = _online_agent(max_observations=1)
    state = _start_idle(
        agent,
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    )
    state = _force_next_extended_action(state, 0)
    next_observation = jnp.asarray([-0.75, 0.6], dtype=jnp.float32)
    transition = _transition(state, next_observation, reward=0.4, discount=0.8)

    wrapper = _wrapper(state)
    builder = agent.state_builder
    lifecycle = agent.prototype_feature_lifecycle
    mixer = agent.config.representation_gradient_mixer
    assert builder is not None
    assert lifecycle is not None
    assert mixer is not None
    _, next_base = builder.update(
        wrapper.builder_state,
        next_observation,
        transition.action,
        transition.reward,
        transition.discount,
    )
    next_augmented = lifecycle.augment(
        wrapper.feature_lifecycle_state,
        next_base,
    )
    behavior = agent._behavior_representation_gradient(
        state,
        transition.reward,
        next_augmented,
        transition.discount,
    )
    mixed = mix_representation_gradients(
        mixer,
        behavior.gradient,
        jnp.zeros_like(behavior.gradient),
        behavior_valid=behavior.valid,
        grounded_world_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    generation = wrapper.feature_lifecycle_state.router_state.generation_count
    expected_pullback = lifecycle.pullback_pair_gradient(
        wrapper.feature_lifecycle_state,
        state.current_representation[:BASE_DIM],
        mixed.gradient,
        generation,
        wrapper.feature_lifecycle_state.router_state.descriptors,
    )
    stale_pullback = lifecycle.pullback_pair_gradient(
        wrapper.feature_lifecycle_state,
        state.current_representation[:BASE_DIM],
        mixed.gradient,
        generation + jnp.asarray(1, dtype=jnp.int32),
        wrapper.feature_lifecycle_state.router_state.descriptors,
    )
    assert not bool(stale_pullback.valid)
    np.testing.assert_array_equal(
        np.asarray(stale_pullback.gradient),
        np.zeros((BASE_DIM,), dtype=np.float32),
    )

    first = agent.update_transition(state, transition)
    diagnostics = first.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(first.transition_diagnostics.valid)
    assert bool(diagnostics.outer_transaction_committed)
    assert bool(diagnostics.lifecycle.transaction_applied)
    np.testing.assert_array_equal(
        np.asarray(diagnostics.target),
        np.asarray(behavior.diagnostics.target),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.pullback_gradient),
        np.asarray(expected_pullback.gradient),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert bool(diagnostics.pullback_valid)
    assert bool(first.state_builder_learning_diagnostics.applied)
    assert int(_wrapper(first.state).feature_lifecycle_state.observe_count) == 1

    second_transition = _transition(
        first.state,
        jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        reward=-0.2,
    )
    second = agent.update_transition(first.state, second_transition)
    second_diagnostics = second.prototype_feature_lifecycle_diagnostics
    assert second_diagnostics is not None
    assert bool(second.transition_diagnostics.valid)
    assert bool(second_diagnostics.outer_transaction_committed)
    assert not bool(second_diagnostics.lifecycle.update_capacity_available)
    assert not bool(second_diagnostics.lifecycle.transaction_applied)
    assert int(_wrapper(second.state).feature_lifecycle_state.observe_count) == 1
    assert int(second.state.step_count) == 2


def test_safe_curation_routes_after_oak_without_changing_dispatch_rng() -> None:
    agent = _identity_agent(replacement_interval=1)
    state = _start_idle(
        agent,
        jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
    )
    state = _force_next_extended_action(state, 0)
    state = _force_promotion(agent, state)
    transition = _transition(
        state,
        jnp.asarray([-1.0, 2.0, 4.0, 0.25], dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    old_feature_state = _wrapper(state).feature_lifecycle_state
    expected_augmented = lifecycle.augment(
        old_feature_state,
        transition.next_observation,
    )
    expected_oak = agent.oak_agent.update(
        _oak(state),
        transition.reward,
        expected_augmented,
        transition.discount,
        decision_observation=expected_augmented,
        execution_boundary=jnp.asarray(False, dtype=jnp.bool_),
    )

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.lifecycle.curation_proposed)
    assert bool(diagnostics.lifecycle.safe_curation_boundary)
    assert bool(diagnostics.lifecycle.curation_committed)
    assert int(diagnostics.lifecycle.semantic_generation_after) == 1
    np.testing.assert_array_equal(
        np.asarray(result.action),
        np.asarray(expected_oak.primitive_action),
    )
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(_oak(result.state).stomp_state.rng_key)),
        np.asarray(jr.key_data(expected_oak.state.stomp_state.rng_key)),
    )

    with pytest.raises(ValueError, match="curate is unavailable"):
        agent.curate(result.state, jr.key(99))
    with pytest.raises(ValueError, match="maybe_curate is unavailable"):
        agent.maybe_curate(result.state, jr.key(99))


def test_bound_oak_subtree_rejects_zero_cache_stale_branch_and_checkpoint(
    tmp_path: Path,
) -> None:
    agent = _identity_agent(replacement_interval=1)
    state = _start_idle(
        agent,
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
    )
    state = _force_next_extended_action(state, 0)
    state = _force_promotion(agent, state)
    transition = _transition(
        state,
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    old_feature_state = _wrapper(state).feature_lifecycle_state
    next_representation = lifecycle.augment(
        old_feature_state,
        transition.next_observation,
    )
    stale_pre_route_oak = agent.oak_agent.update(
        _oak(state),
        transition.reward,
        next_representation,
        transition.discount,
        decision_observation=next_representation,
        execution_boundary=jnp.asarray(False, dtype=jnp.bool_),
    ).state

    committed = agent.update_transition(state, transition)
    diagnostics = committed.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.lifecycle.curation_committed)
    assert int(_bound_oak(committed.state).consumer_binding.semantic_generation) == 1

    stale_bound_oak = _bound_oak(state).replace(oak_state=stale_pre_route_oak)
    mixed = cast(
        PrototypeAgentState,
        committed.state.replace(oak_state=stale_bound_oak),
    )
    assert not bool(agent._state_cache_consistent(mixed))
    assert not bool(agent._checkpoint_state_valid(mixed))
    with pytest.raises(ValueError, match="inconsistent PrototypeAgent state"):
        save_prototype_checkpoint(agent, mixed, tmp_path / "stale-bound-oak")

    committed_bound_oak = _bound_oak(committed.state)
    forked_descriptors = committed_bound_oak.consumer_binding.descriptors.at[0].set(
        jnp.asarray([1, 3], dtype=jnp.int32)
    )
    descriptor_fork = cast(
        PrototypeAgentState,
        committed.state.replace(
            oak_state=committed_bound_oak.replace(
                consumer_binding=committed_bound_oak.consumer_binding.replace(
                    descriptors=forked_descriptors,
                )
            )
        ),
    )
    assert not bool(agent._checkpoint_state_valid(descriptor_fork))
    with pytest.raises(ValueError, match="inconsistent PrototypeAgent state"):
        save_prototype_checkpoint(
            agent,
            descriptor_fork,
            tmp_path / "same-generation-descriptor-fork",
        )

    rejected = agent.update_transition(
        mixed,
        _transition(
            mixed,
            jnp.zeros((BASE_DIM,), dtype=jnp.float32),
        ),
    )
    assert not bool(rejected.transition_diagnostics.valid)
    _assert_tree_exact(rejected.state, mixed)


def test_executing_option_target_is_exact_and_curation_defers_with_binding() -> None:
    agent = _identity_agent(replacement_interval=1)
    state = _start_idle(
        agent,
        jnp.asarray([0.25, -0.5, 0.75, 1.0], dtype=jnp.float32),
    )
    state = _force_next_extended_action(state, N_ACTIONS)
    entered = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.5, 0.25, -0.75, 1.25], dtype=jnp.float32),
        ),
    ).state
    entered_stomp = _oak(entered).stomp_state
    assert int(entered_stomp.executing_option) == 0

    entered = _force_promotion(agent, entered)
    old_feature_state = _wrapper(entered).feature_lifecycle_state
    old_binding = _bound_oak(entered).consumer_binding
    next_raw = jnp.asarray([-0.2, 0.4, 0.6, -0.8], dtype=jnp.float32)
    transition = _transition(entered, next_raw, reward=0.7, discount=0.85)
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    bootstrap = lifecycle.augment(old_feature_state, next_raw)
    option_index = entered_stomp.executing_option
    pseudo_reward = compute_pseudo_reward(
        agent.oak_agent.stomp_agent.spec_arrays,
        option_index,
        bootstrap,
    )
    option_terminates = check_option_terminated(
        agent.oak_agent.stomp_agent.spec_arrays,
        option_index,
        bootstrap,
        entered_stomp.option_steps + 1,
    )
    bootstrap_discount = jnp.where(
        option_terminates,
        jnp.asarray(0.0, dtype=jnp.float32),
        transition.discount,
    )
    expected_target = (
        pseudo_reward
        - entered_stomp.option_policies.average_rewards[option_index]
        + bootstrap_discount
        * jnp.max(entered_stomp.option_policies.q_weights[option_index] @ bootstrap)
    )

    result = agent.update_transition(entered, transition)
    feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert feature_diagnostics is not None
    behavior_diagnostics = result.behavior_gradient_result.diagnostics
    assert bool(behavior_diagnostics.intra_option_source)
    assert not bool(behavior_diagnostics.idle_base_source)
    np.testing.assert_array_equal(behavior_diagnostics.target, expected_target)
    np.testing.assert_array_equal(feature_diagnostics.target, expected_target)
    assert bool(feature_diagnostics.lifecycle.curation_proposed)
    assert not bool(feature_diagnostics.lifecycle.safe_curation_boundary)
    assert bool(feature_diagnostics.lifecycle.curation_deferred)
    assert not bool(feature_diagnostics.lifecycle.routing_attempted)
    np.testing.assert_array_equal(
        _wrapper(result.state).feature_lifecycle_state.router_state.descriptors,
        old_feature_state.router_state.descriptors,
    )
    _assert_tree_exact(_bound_oak(result.state).consumer_binding, old_binding)
    assert bool(agent._state_cache_consistent(result.state))


def test_option_search_and_feature_lifecycle_commit_in_one_valid_transition() -> None:
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(),
            option_search_control=OptionSearchControlConfig(
                backup_budget=1,
                min_model_completions=1,
            ),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
        )
    )
    state = _start_idle(
        agent,
        jnp.asarray([0.1, 0.2, -0.3, 0.4], dtype=jnp.float32),
    )
    state = _force_next_extended_action(state, N_ACTIONS)
    entered = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.3, -0.1, 0.5, -0.7], dtype=jnp.float32),
        ),
    ).state
    entered_oak = _oak(entered)
    assert int(entered_oak.stomp_state.executing_option) == 0
    np.testing.assert_array_equal(
        entered_oak.execution_counts,
        np.asarray([1, 0], dtype=np.int32),
    )

    stomp = entered_oak.stomp_state
    models = stomp.option_models.replace(
        n_completions=jnp.asarray([1, 0], dtype=jnp.int32),
        env_return_ema=jnp.asarray([4.0, 0.0], dtype=jnp.float32),
        cumreward_ema=jnp.asarray([4.0, 0.0], dtype=jnp.float32),
        duration_ema=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        baseline_mass_ema=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        discount_ema=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        next_state_weights=jnp.zeros_like(
            stomp.option_models.next_state_weights
        ),
    )
    bound = _bound_oak(entered)
    supported = cast(
        PrototypeAgentState,
        entered.replace(
            oak_state=bound.replace(
                oak_state=entered_oak.replace(
                    stomp_state=stomp.replace(option_models=models)
                )
            )
        ),
    )
    assert bool(agent._state_cache_consistent(supported))

    result = agent.update_transition(
        supported,
        _transition(
            supported,
            jnp.asarray([-0.6, 0.8, 0.2, -0.4], dtype=jnp.float32),
        ),
    )
    search_diagnostics = result.option_search_control_diagnostics
    feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert search_diagnostics is not None
    assert feature_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert int(search_diagnostics.applied_count) == 1
    assert bool(feature_diagnostics.lifecycle.transaction_applied)
    assert bool(feature_diagnostics.outer_transaction_committed)
    assert bool(agent._state_cache_consistent(result.state))


def test_unexpected_feature_step_rejection_rolls_back_outer_transition() -> None:
    agent = _identity_agent()
    state = _start_idle(
        agent,
        jnp.asarray([0.2, -0.4, 0.6, -0.8], dtype=jnp.float32),
    )
    lifecycle_config = agent.config.prototype_feature_lifecycle
    assert lifecycle_config is not None
    agent._prototype_feature_lifecycle = _RejectingFeatureLifecycle(
        lifecycle_config
    )
    transition = _transition(
        state,
        jnp.asarray([0.7, 0.1, -0.5, 0.3], dtype=jnp.float32),
    )

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    assert bool(diagnostics.lifecycle.learner_update_rejected)
    assert not bool(diagnostics.outer_transaction_committed)
    _assert_tree_exact(result.state, state)


def test_jit_scan_and_checkpoint_round_trip_enabled_lane(tmp_path: Path) -> None:
    agent = _identity_agent()
    initial = _start_idle(
        agent,
        jnp.asarray([0.1, 0.2, -0.3, 0.4], dtype=jnp.float32),
    )
    transition = _transition(
        initial,
        jnp.asarray([-0.2, 0.7, 0.5, -0.1], dtype=jnp.float32),
    )
    direct = agent.update_transition(initial, transition)
    batched = jax.tree.map(
        lambda value: None if value is None else jnp.expand_dims(value, 0),
        transition,
        is_leaf=lambda value: value is None,
    )
    scanned = jax.jit(agent.scan_transitions)(initial, batched)
    assert bool(jnp.all(scanned.transition_valid))
    _assert_tree_close(scanned.state, direct.state)

    checkpoint = tmp_path / "prototype-feature-lifecycle"
    save_prototype_checkpoint(agent, scanned.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_exact(restored_state, scanned.state)


def test_disabled_lane_preserves_legacy_config_state_and_checkpoint(
    tmp_path: Path,
) -> None:
    config = PrototypeAgentConfig(
        oak=_oak_config(observation_dim=BASE_DIM),
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
    )
    encoded = config.to_config()
    assert "prototype_feature_lifecycle" not in encoded
    assert PrototypeAgentConfig.from_config(encoded).to_config() == encoded
    agent = PrototypeAgent(config)
    initial = agent.init(jr.key(101))
    assert isinstance(initial.state_builder_state, IdentityStateBuilderState)
    state = agent.start(
        initial,
        jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32),
    )
    assert isinstance(state.state_builder_state, IdentityStateBuilderState)

    checkpoint = tmp_path / "prototype-feature-disabled"
    save_prototype_checkpoint(agent, state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert "prototype_feature_lifecycle" not in restored_agent.to_config()
    assert isinstance(
        restored_state.state_builder_state,
        IdentityStateBuilderState,
    )
    _assert_tree_exact(restored_state, state)
