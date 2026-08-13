# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Atomic Prototype composition of control and linear-Horde feature consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureRepresentationState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    HordeSpec,
    create_horde_spec,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_DIM = 3
ACTIVE_SLOTS = 2
TOTAL_DIM = BASE_DIM + ACTIVE_SLOTS
N_ACTIONS = 2
N_OPTIONS = 1
N_DEMONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_shared_lane_jit_and_v7_checkpoint_round_trip":
        yield
    else:
        with jax.disable_jit():
            yield


def _horde_spec(*, reverse: bool = False) -> HordeSpec:
    demons = [
        GVFSpec(
            name="instant",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        ),
        GVFSpec(
            name="temporal",
            demon_type=DemonType.PREDICTION,
            gamma=0.5,
            lamda=0.25,
            cumulant_index=1,
        ),
    ]
    return create_horde_spec(tuple(reversed(demons)) if reverse else demons)


def _feature_config(
    *,
    managed_horde_demons: int = N_DEMONS,
    replacement_interval: int = 0,
    max_observations: int = 100,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=3,
        n_tasks=1 + managed_horde_demons,
        n_options=N_OPTIONS,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
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
        managed_horde_demons=managed_horde_demons,
    )


def _oak_config(*, hidden_sizes: tuple[int, ...] = ()) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1_000_000.0,
                    max_option_steps=8,
                ),
            ),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=hidden_sizes,
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _agent(
    *,
    reverse_demons: bool = False,
    replacement_interval: int = 0,
    max_observations: int = 100,
) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(
                observation_dim=BASE_DIM,
            ),
            horde_spec=_horde_spec(reverse=reverse_demons),
            horde_hidden_sizes=(),
            horde_step_size=0.1,
            prototype_feature_lifecycle=_feature_config(
                replacement_interval=replacement_interval,
                max_observations=max_observations,
            ),
        )
    )


def _bundle(state: PrototypeAgentState) -> PrototypeFeatureOaKHordeState:
    assert type(state.oak_state) is PrototypeFeatureOaKHordeState
    return state.oak_state


def _representation_wrapper(
    state: PrototypeAgentState,
) -> PrototypeFeatureRepresentationState:
    assert type(state.state_builder_state) is PrototypeFeatureRepresentationState
    return state.state_builder_state


def _start_idle(agent: PrototypeAgent, observation: jax.Array) -> PrototypeAgentState:
    for seed in range(32):
        state = agent.start(agent.init(jr.key(seed)), observation)
        if int(_bundle(state).oak_state.stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not obtain a deterministic idle decision")


def _force_next_primitive(
    state: PrototypeAgentState,
    primitive_action: int = 0,
) -> PrototypeAgentState:
    bundle = _bundle(state)
    stomp = bundle.oak_state.stomp_state
    learner = stomp.base_learner_state
    biases = tuple(
        jnp.full_like(
            bias,
            100.0 if index == primitive_action else -100.0,
        )
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(biases=biases)
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=bundle.replace(
                oak_state=bundle.oak_state.replace(
                    stomp_state=stomp.replace(base_learner_state=learner),
                )
            )
        ),
    )


def _force_promotion(state: PrototypeAgentState) -> PrototypeAgentState:
    wrapper = _representation_wrapper(state)
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
    feature_state = feature_state.replace(
        learner_state=learner.replace(
            utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
            candidate_utilities=candidate_utilities,
        )
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature_state,
            )
        ),
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.5,
    discount: float = 0.9,
    next_decision_observation: jax.Array | None = None,
    truncated: bool = False,
    cumulants: jax.Array | None = None,
    horde_discounts: jax.Array | None = None,
) -> PrototypeTransition:
    decision_observation = (
        next_observation
        if next_decision_observation is None
        else next_decision_observation
    )
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=decision_observation,
        horde_cumulants=cumulants,
        horde_discounts=horde_discounts,
    )


def _materialize_keys(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: (
            jr.key_data(value)
            if getattr(value, "dtype", None) is not None
            and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        ),
        tree,
    )


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_horde_learning_equal(left: Any, right: Any) -> None:
    """Compare all algorithmic Horde state while excluding wall-clock timers."""

    _assert_tree_exact(
        left.replace(birth_timestamp=0.0, uptime_s=0.0),
        right.replace(birth_timestamp=0.0, uptime_s=0.0),
    )


def test_shared_config_group_balance_schema_identity_and_fail_closed_contract() -> None:
    agent = _agent()
    encoded = agent.to_config()
    assert PrototypeAgentConfig.from_config(encoded).to_config() == encoded
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    assert lifecycle.learner.to_config()["task_utility_weights"] == [
        0.5,
        0.25,
        0.25,
    ]
    state = agent.init(jr.key(0))
    assert state.horde_state is None
    assert bool(agent._checkpoint_state_valid(state))

    with pytest.raises(ValueError, match="demon count"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            horde_spec=_horde_spec(),
            horde_hidden_sizes=(),
            horde_step_size=0.1,
            prototype_feature_lifecycle=_feature_config(
                managed_horde_demons=1,
            ),
        )
    with pytest.raises(ValueError, match="horde_hidden_sizes =="):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            horde_spec=_horde_spec(),
            horde_hidden_sizes=(4,),
            horde_step_size=0.1,
            prototype_feature_lifecycle=_feature_config(),
        )
    with pytest.raises(ValueError, match="requires an exact HordeSpec"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            horde_hidden_sizes=(),
            horde_step_size=0.1,
            prototype_feature_lifecycle=_feature_config(),
        )

    reordered = _agent(reverse_demons=True).init(jr.key(0))
    assert not np.array_equal(
        np.asarray(_bundle(state).schema_digest),
        np.asarray(_bundle(reordered).schema_digest),
    )
    cross_schema = state.replace(
        oak_state=_bundle(state).replace(
            schema_digest=_bundle(reordered).schema_digest,
        )
    )
    assert not bool(agent._checkpoint_state_valid(cross_schema))


def test_control_then_horde_targets_and_old_bank_update_oracle() -> None:
    agent = _agent()
    initial = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float32)
    state = _start_idle(agent, initial)
    next_observation = jnp.asarray([-0.5, 1.5, 2.0], dtype=jnp.float32)
    transition = _transition(
        state,
        next_observation,
        reward=0.4,
        cumulants=jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        horde_discounts=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    feature_state = _representation_wrapper(state).feature_lifecycle_state
    next_augmented = lifecycle.augment(feature_state, next_observation)
    behavior = agent._behavior_representation_gradient(
        state,
        transition.reward,
        next_augmented,
        transition.discount,
    )
    expected_horde = agent._update_horde_for_transition(
        _bundle(state).horde_state,
        state.current_representation,
        next_augmented,
        transition,
    )

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.outer_transaction_committed)
    expected_targets = jnp.concatenate(
        (
            jnp.reshape(behavior.diagnostics.target, (1,)),
            expected_horde.td_targets,
        )
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.task_targets),
        np.asarray(expected_targets),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.task_target_available),
        np.ones((3,), dtype=np.bool_),
    )
    np.testing.assert_allclose(
        np.asarray(result.horde_td_errors),
        np.asarray(expected_horde.td_errors),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    _assert_horde_learning_equal(
        _bundle(result.state).horde_state,
        expected_horde.state,
    )
    assert result.state.horde_state is None
    assert int(result.state.step_count) == 1
    assert int(_bundle(result.state).horde_state.step_count) == 1
    assert bool(agent._checkpoint_state_valid(result.state))


def test_forced_curation_routes_post_learning_oak_and_horde_atomically() -> None:
    agent = _agent(replacement_interval=1)
    state = _start_idle(
        agent,
        jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float32),
    )
    state = _force_next_primitive(_force_promotion(state))
    transition = _transition(
        state,
        jnp.asarray([-2.0, 0.25, 3.0], dtype=jnp.float32),
        reward=0.3,
        cumulants=jnp.asarray([0.4, -0.2], dtype=jnp.float32),
        horde_discounts=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    old_feature = _representation_wrapper(state).feature_lifecycle_state
    old_bundle = _bundle(state)
    next_augmented = lifecycle.augment(old_feature, transition.next_observation)
    behavior = agent._behavior_representation_gradient(
        state,
        transition.reward,
        next_augmented,
        transition.discount,
    )
    oak_update = agent.oak_agent.update(
        old_bundle.oak_state,
        transition.reward,
        next_augmented,
        transition.discount,
        decision_observation=next_augmented,
        execution_boundary=jnp.asarray(False, dtype=jnp.bool_),
    )
    horde_update = agent._update_horde_for_transition(
        old_bundle.horde_state,
        state.current_representation,
        next_augmented,
        transition,
    )
    expected = lifecycle.observe_and_route_with_horde(
        old_feature,
        oak_update.state,
        horde_update.state,
        old_bundle.consumer_binding,
        PrototypeFeatureLifecycleEvent(
            observation=state.current_raw_observation,
            targets=jnp.concatenate(
                (
                    jnp.reshape(behavior.diagnostics.target, (1,)),
                    horde_update.td_targets,
                )
            ),
            next_observation=transition.next_observation,
            allow_curation=jnp.asarray(True, dtype=jnp.bool_),
        ),
    )
    assert bool(expected.diagnostics.curation_committed)

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.lifecycle.curation_committed)
    assert int(diagnostics.lifecycle.semantic_generation_after) == 1
    actual_bundle = _bundle(result.state)
    _assert_tree_exact(actual_bundle.consumer_binding, expected.consumer_binding)
    _assert_horde_learning_equal(actual_bundle.horde_state, expected.horde_state)
    for actual, wanted in zip(
        actual_bundle.horde_state.head_params.weights,
        expected.horde_state.head_params.weights,
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(wanted))
    for actual, wanted in zip(
        actual_bundle.horde_state.head_traces,
        expected.horde_state.head_traces,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual[0]),
            np.asarray(wanted[0]),
        )
        np.testing.assert_array_equal(
            np.asarray(actual[1]),
            np.asarray(wanted[1]),
        )
    np.testing.assert_array_equal(
        np.asarray(actual_bundle.oak_state.stomp_state.base_last_obs),
        np.asarray(expected.next_augmented_observation),
    )


def test_autoreset_bootstrap_and_nan_demon_are_auditable_and_inactive() -> None:
    agent = _agent()
    state = _start_idle(
        agent,
        jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32),
    )
    bootstrap = jnp.asarray([2.0, 0.25, -0.5], dtype=jnp.float32)
    reset = jnp.asarray([-3.0, 1.5, 0.75], dtype=jnp.float32)
    transition = _transition(
        state,
        bootstrap,
        reward=-0.2,
        next_decision_observation=reset,
        truncated=True,
        cumulants=jnp.asarray([0.3, jnp.nan], dtype=jnp.float32),
        horde_discounts=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    feature_state = _representation_wrapper(state).feature_lifecycle_state
    bootstrap_augmented = lifecycle.augment(feature_state, bootstrap)
    expected_horde = agent._update_horde_for_transition(
        _bundle(state).horde_state,
        state.current_representation,
        bootstrap_augmented,
        transition,
    )

    result = agent.update_transition(state, transition)
    diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    np.testing.assert_array_equal(
        np.asarray(diagnostics.task_target_available),
        np.asarray([True, True, False]),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.task_targets[1]),
        np.asarray(expected_horde.td_targets[0]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.task_targets[2]),
        np.asarray(0.0, dtype=np.float32),
    )
    old_horde = _bundle(state).horde_state
    new_horde = _bundle(result.state).horde_state
    np.testing.assert_array_equal(
        np.asarray(new_horde.head_params.weights[1]),
        np.asarray(old_horde.head_params.weights[1]),
    )
    np.testing.assert_array_equal(
        np.asarray(new_horde.head_traces[1][0]),
        np.asarray(old_horde.head_traces[1][0]),
    )
    expected_decision = lifecycle.augment(
        _representation_wrapper(result.state).feature_lifecycle_state,
        reset,
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.current_representation),
        np.asarray(expected_decision),
    )


def test_corrupt_bundle_schema_or_lms_step_rolls_back_and_cannot_checkpoint(
    tmp_path: Path,
) -> None:
    agent = _agent()
    state = _start_idle(
        agent,
        jnp.asarray([1.0, -1.0, 0.25], dtype=jnp.float32),
    )
    bundle = _bundle(state)
    bad_digest = bundle.schema_digest.at[0].set(bundle.schema_digest[0] ^ 1)
    stale = cast(
        PrototypeAgentState,
        state.replace(oak_state=bundle.replace(schema_digest=bad_digest)),
    )
    assert not bool(agent._checkpoint_state_valid(stale))
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(agent, stale, tmp_path / "stale")
    transition = _transition(
        stale,
        jnp.asarray([0.5, 0.75, -2.0], dtype=jnp.float32),
        cumulants=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
    )
    rejected = agent.update_transition(stale, transition)
    assert bool(rejected.transition_diagnostics.rejected)
    _assert_tree_exact(rejected.state, stale)

    horde = bundle.horde_state
    first_pair = horde.head_optimizer_states[0]
    bad_pair = (
        first_pair[0].replace(
            step_size=first_pair[0].step_size * jnp.asarray(2.0, dtype=jnp.float32)
        ),
        first_pair[1],
    )
    bad_horde = horde.replace(
        head_optimizer_states=(bad_pair, *horde.head_optimizer_states[1:])
    )
    bad_optimizer = state.replace(
        oak_state=bundle.replace(horde_state=bad_horde)
    )
    assert not bool(agent._checkpoint_state_valid(bad_optimizer))


def test_shared_lane_jit_and_v7_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = _agent(max_observations=2)
    state = _start_idle(
        agent,
        jnp.asarray([0.25, -0.75, 1.5], dtype=jnp.float32),
    )
    transition = _transition(
        state,
        jnp.asarray([-0.5, 2.0, 0.1], dtype=jnp.float32),
        cumulants=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        horde_discounts=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )
    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)
    _assert_tree_exact(eager.state, compiled.state)
    np.testing.assert_allclose(
        np.asarray(eager.prototype_feature_lifecycle_diagnostics.task_targets),
        np.asarray(compiled.prototype_feature_lifecycle_diagnostics.task_targets),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    checkpoint = tmp_path / "shared_feature_horde"
    save_prototype_checkpoint(agent, compiled.state, checkpoint)
    assert load_checkpoint_metadata(checkpoint)["schema"] == PROTOTYPE_CHECKPOINT_SCHEMA
    assert PROTOTYPE_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v13"
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_exact(restored_state, compiled.state)
    assert type(restored_state.oak_state) is PrototypeFeatureOaKHordeState
    assert restored_state.horde_state is None


def test_v3_label_accepts_exact_shape_but_rejects_impossible_shared_lane(
    tmp_path: Path,
) -> None:
    legacy_oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=BASE_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
        )
    )
    legacy_agent = PrototypeAgent(PrototypeAgentConfig(oak=legacy_oak))
    legacy_state = legacy_agent.init(jr.key(11))
    legacy_config = legacy_agent.to_config()
    legacy_digest = hashlib.sha256(
        json.dumps(
            legacy_config,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    legacy_path = tmp_path / "legacy-v3"
    save_checkpoint(
        legacy_state,
        legacy_path,
        metadata={
            "schema": "alberta.prototype_agent.v3",
            "agent_config": legacy_config,
            "config_sha256": legacy_digest,
        },
    )
    restored_agent, restored_state = load_prototype_checkpoint(legacy_path)
    assert restored_agent.to_config() == legacy_config
    _assert_tree_exact(restored_state, legacy_state)

    shared_agent = _agent()
    shared_state = shared_agent.init(jr.key(12))
    shared_config = shared_agent.to_config()
    shared_digest = hashlib.sha256(
        json.dumps(
            shared_config,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    impossible_path = tmp_path / "impossible-shared-v3"
    save_checkpoint(
        shared_state,
        impossible_path,
        metadata={
            "schema": "alberta.prototype_agent.v3",
            "agent_config": shared_config,
            "config_sha256": shared_digest,
        },
    )
    with pytest.raises(ValueError, match="requires a v7"):
        load_prototype_checkpoint(impossible_path)
