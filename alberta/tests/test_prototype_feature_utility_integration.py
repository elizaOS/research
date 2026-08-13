# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Prototype integration contracts for diagnostic-only causal feature utility."""

from __future__ import annotations

import dataclasses
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
    PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureOaKHordeUtilityState,
    PrototypeFeatureRepresentationState,
    PrototypeFeatureUtilityIntegrationDiagnostics,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_utility import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityDiagnostics,
    PrototypeFeatureUtilityState,
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
CANDIDATE_SLOTS = 3
TOTAL_DIM = BASE_DIM + ACTIVE_SLOTS
N_ACTIONS = 2
N_OPTIONS = 1
N_DEMONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_diagnostic_lane_is_bit_exact_under_jit":
        yield
    else:
        with jax.disable_jit():
            yield


def _horde_spec() -> HordeSpec:
    return create_horde_spec(
        (
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
        )
    )


def _feature_config(
    *,
    managed_horde_demons: int = N_DEMONS,
    replacement_interval: int = 0,
    max_observations: int = 8,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=CANDIDATE_SLOTS,
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


def _utility_config(*, max_observations: int = 8) -> PrototypeFeatureUtilityConfig:
    return PrototypeFeatureUtilityConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=CANDIDATE_SLOTS,
        managed_horde_demons=N_DEMONS,
        utility_decay=0.75,
        shadow_step_size=0.2,
        second_moment_decay=0.5,
        scale_epsilon=1.0e-6,
        max_observations=max_observations,
    )


def _oak_config() -> OaKConfig:
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
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _agent_config(
    *,
    utility: bool,
    replacement_interval: int = 0,
    max_observations: int = 8,
) -> PrototypeAgentConfig:
    kwargs: dict[str, Any] = {
        "oak": _oak_config(),
        "state_builder": IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        "horde_spec": _horde_spec(),
        "horde_hidden_sizes": (),
        "horde_step_size": 0.1,
        "prototype_feature_lifecycle": _feature_config(
            replacement_interval=replacement_interval,
            max_observations=max_observations,
        ),
    }
    if utility:
        kwargs["prototype_feature_utility"] = _utility_config(max_observations=max_observations)
    return PrototypeAgentConfig(**kwargs)


def _agent(
    *,
    utility: bool,
    replacement_interval: int = 0,
    max_observations: int = 8,
) -> PrototypeAgent:
    return PrototypeAgent(
        _agent_config(
            utility=utility,
            replacement_interval=replacement_interval,
            max_observations=max_observations,
        )
    )


def _bundle(state: PrototypeAgentState) -> Any:
    slot = state.oak_state
    assert type(slot) in {
        PrototypeFeatureOaKHordeState,
        PrototypeFeatureOaKHordeUtilityState,
    }
    return slot


def _feature_state(state: PrototypeAgentState) -> Any:
    wrapper = state.state_builder_state
    assert type(wrapper) is PrototypeFeatureRepresentationState
    return wrapper.feature_lifecycle_state


def _utility_state(state: PrototypeAgentState) -> PrototypeFeatureUtilityState:
    bundle = _bundle(state)
    assert type(bundle) is PrototypeFeatureOaKHordeUtilityState
    assert type(bundle.feature_utility_state) is PrototypeFeatureUtilityState
    return bundle.feature_utility_state


def _utility_diagnostics(result: Any) -> PrototypeFeatureUtilityDiagnostics:
    integration = _utility_integration_diagnostics(result)
    assert type(integration.observation) is PrototypeFeatureUtilityDiagnostics
    return integration.observation


def _utility_integration_diagnostics(
    result: Any,
) -> PrototypeFeatureUtilityIntegrationDiagnostics:
    diagnostics = result.prototype_feature_utility_diagnostics
    assert type(diagnostics) is PrototypeFeatureUtilityIntegrationDiagnostics
    return diagnostics


def _start_idle(
    agent: PrototypeAgent,
    observation: jax.Array,
    *,
    seed: int | None = None,
) -> tuple[PrototypeAgentState, int]:
    seeds = range(32) if seed is None else (seed,)
    for candidate in seeds:
        state = agent.start(agent.init(jr.key(candidate)), observation)
        if int(_bundle(state).oak_state.stomp_state.executing_option) == -1:
            return state, candidate
    raise AssertionError("could not obtain a deterministic idle decision")


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.4,
    cumulants: jax.Array | None = None,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=cumulants,
        horde_discounts=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )


def _pair_values(base: jax.Array, descriptors: jax.Array) -> jax.Array:
    return base[descriptors[:, 0]] * base[descriptors[:, 1]]


def _set_old_consumer_tails(
    state: PrototypeAgentState,
    *,
    executing_option: bool,
    control_tail: jax.Array,
    horde_tails: jax.Array,
) -> PrototypeAgentState:
    bundle = _bundle(state)
    stomp = bundle.oak_state.stomp_state
    action = int(state.current_action)
    base_head_index = N_ACTIONS if executing_option else action
    base_weights = [
        jnp.zeros_like(weight) for weight in stomp.base_learner_state.head_params.weights
    ]
    base_row = jnp.concatenate(
        (
            jnp.zeros((BASE_DIM,), dtype=jnp.float32),
            (jnp.asarray([7.0, -9.0], dtype=jnp.float32) if executing_option else control_tail),
        )
    )
    base_weights[base_head_index] = base_row.reshape(base_weights[base_head_index].shape)
    base_biases = tuple(
        jnp.zeros_like(bias) for bias in stomp.base_learner_state.head_params.biases
    )
    base_learner = stomp.base_learner_state.replace(
        head_params=stomp.base_learner_state.head_params.replace(
            weights=tuple(base_weights),
            biases=base_biases,
        )
    )
    option_policies = stomp.option_policies
    if executing_option:
        option_row = jnp.concatenate((jnp.zeros((BASE_DIM,), dtype=jnp.float32), control_tail))
        option_policies = option_policies.replace(
            q_weights=option_policies.q_weights.at[0, action].set(option_row)
        )
    stomp = stomp.replace(
        base_learner_state=base_learner,
        option_policies=option_policies,
        executing_option=jnp.asarray(0 if executing_option else -1, dtype=jnp.int32),
        base_last_action=jnp.asarray(base_head_index, dtype=jnp.int32),
        option_last_intra_action=jnp.asarray(action, dtype=jnp.int32),
        option_start_obs=state.current_representation,
        # No primitive transition has yet been observed at outer step zero.
        option_steps=jnp.asarray(0, dtype=jnp.int32),
    )

    horde = bundle.horde_state
    horde_weights = tuple(
        jnp.concatenate((jnp.zeros((BASE_DIM,), dtype=jnp.float32), horde_tails[index])).reshape(
            weight.shape
        )
        for index, weight in enumerate(horde.head_params.weights)
    )
    horde_biases = tuple(jnp.zeros_like(bias) for bias in horde.head_params.biases)
    horde = horde.replace(
        head_params=horde.head_params.replace(
            weights=horde_weights,
            biases=horde_biases,
        )
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=bundle.replace(
                consumer_state=bundle.consumer_state.replace(
                    oak_state=bundle.oak_state.replace(stomp_state=stomp),
                    horde_state=horde,
                ),
            )
        ),
    )


def _force_promotion(state: PrototypeAgentState) -> PrototypeAgentState:
    wrapper = cast(PrototypeFeatureRepresentationState, state.state_builder_state)
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
        index for index, descriptor in enumerate(candidates) if descriptor not in active
    )
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner = learner.replace(
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        candidate_utilities=candidate_utilities,
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature_state.replace(learner_state=learner)
            )
        ),
    )


def _force_next_primitive(state: PrototypeAgentState) -> PrototypeAgentState:
    """Keep the next decision at a lifecycle-safe primitive boundary."""

    bundle = _bundle(state)
    stomp = bundle.oak_state.stomp_state
    learner = stomp.base_learner_state
    biases = tuple(
        jnp.full_like(bias, 100.0 if index == 0 else -100.0)
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(head_params=learner.head_params.replace(biases=biases))
    consumer = (
        bundle.consumer_state if type(bundle) is (PrototypeFeatureOaKHordeUtilityState) else bundle
    )
    routed_consumer = consumer.replace(
        oak_state=consumer.oak_state.replace(stomp_state=stomp.replace(base_learner_state=learner))
    )
    oak_state = (
        bundle.replace(consumer_state=routed_consumer)
        if type(bundle) is PrototypeFeatureOaKHordeUtilityState
        else routed_consumer
    )
    return cast(
        PrototypeAgentState,
        state.replace(oak_state=oak_state),
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


def _assert_horde_exact(left: Any, right: Any) -> None:
    _assert_tree_exact(
        left.replace(birth_timestamp=0.0, uptime_s=0.0),
        right.replace(birth_timestamp=0.0, uptime_s=0.0),
    )


def _canonical_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_opt_in_is_strict_and_disabled_shared_lane_remains_v13() -> None:
    plain = _agent(utility=False)
    enabled = _agent(utility=True)

    assert "prototype_feature_utility" not in plain.to_config()
    assert "prototype_feature_utility" in enabled.to_config()
    assert PrototypeAgentConfig.from_config(enabled.to_config()).to_config() == (
        enabled.to_config()
    )
    assert PROTOTYPE_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v13"
    assert PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA == "alberta.prototype_agent.v14"
    assert not PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY

    plain_state = plain.init(jr.key(0))
    enabled_state = enabled.init(jr.key(0))
    assert type(plain_state.oak_state) is PrototypeFeatureOaKHordeState
    assert type(enabled_state.oak_state) is PrototypeFeatureOaKHordeUtilityState
    assert plain_state.horde_state is None
    assert enabled_state.horde_state is None
    assert plain.prototype_feature_utility_resource_budget is None
    assert enabled.prototype_feature_utility is not None
    assert enabled.prototype_feature_utility_resource_budget == (
        enabled.prototype_feature_utility.resource_budget()
    )
    assert not enabled.prototype_feature_utility_resource_budget.curation_authority
    assert enabled.prototype_feature_utility_resource_budget.router_calls_per_observe == 0
    assert enabled.prototype_feature_utility_resource_budget.consumer_updates_per_observe == 0

    utility_config = _utility_config()
    with pytest.raises(ValueError, match="prototype_feature_utility"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_utility=utility_config,
        )
    with pytest.raises(ValueError, match="prototype_feature_utility"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            horde_spec=_horde_spec(),
            horde_hidden_sizes=(),
            prototype_feature_lifecycle=_feature_config(managed_horde_demons=0),
            prototype_feature_utility=utility_config,
        )
    with pytest.raises(ValueError, match="prototype_feature_utility"):
        dataclasses.replace(
            _agent_config(utility=True),
            prototype_feature_utility=dataclasses.replace(
                utility_config,
                max_observations=7,
            ),
        )


@pytest.mark.parametrize("executing_option", [False, True])
def test_old_owner_and_declared_horde_channels_match_hand_oracle(
    executing_option: bool,
) -> None:
    agent = _agent(utility=True)
    base = jnp.asarray([1.5, -2.0, 0.75], dtype=jnp.float32)
    state, _ = _start_idle(agent, base)
    control_tail = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    horde_tails = jnp.asarray([[-0.4, 0.2], [0.3, 0.1]], dtype=jnp.float32)
    state = _set_old_consumer_tails(
        state,
        executing_option=executing_option,
        control_tail=control_tail,
        horde_tails=horde_tails,
    )
    assert bool(agent._checkpoint_state_valid(state))

    next_base = jnp.asarray([-0.5, 1.25, 2.0], dtype=jnp.float32)
    transition = _transition(
        state,
        next_base,
        cumulants=jnp.asarray([0.7, jnp.nan], dtype=jnp.float32),
    )
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    old_feature = _feature_state(state)
    next_augmented = lifecycle.augment(old_feature, next_base)
    expected_control = agent._behavior_representation_gradient(
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
    integration = _utility_integration_diagnostics(result)
    diagnostics = integration.observation
    assert bool(result.transition_diagnostics.valid)
    assert bool(integration.outer_transaction_committed)
    assert not bool(integration.rebind_required)
    assert not bool(integration.rebind.transaction_applied)
    assert bool(diagnostics.transaction_applied)
    expected_targets = np.asarray(
        [
            float(expected_control.diagnostics.target),
            float(expected_horde.td_targets[0]),
            0.0,
        ],
        dtype=np.float32,
    )
    expected_predictions = np.asarray(
        [
            float(expected_control.diagnostics.prediction),
            float(expected_horde.predictions[0]),
            float(expected_horde.predictions[1]),
        ],
        dtype=np.float32,
    )
    availability = np.asarray([True, True, False], dtype=np.bool_)
    np.testing.assert_allclose(diagnostics.targets, expected_targets, rtol=1e-6)
    np.testing.assert_allclose(
        diagnostics.predictions,
        expected_predictions,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(diagnostics.target_available, availability)

    descriptors = old_feature.router_state.descriptors
    pair_values = np.asarray(_pair_values(base, descriptors), dtype=np.float32)
    old_tail_weights = np.concatenate(
        (np.asarray(control_tail)[None, :], np.asarray(horde_tails)), axis=0
    )
    old_moments = np.asarray(_utility_state(state).target_second_moments)
    scales = np.maximum.reduce(
        (
            old_moments,
            expected_targets**2,
            expected_predictions**2,
            np.full((1 + N_DEMONS,), 1.0e-6, dtype=np.float32),
        )
    )
    errors = (expected_targets - expected_predictions) / np.sqrt(scales)
    contributions = old_tail_weights * pair_values[None, :] / np.sqrt(scales)[:, None]
    loss_changes = 0.5 * ((errors[:, None] + contributions) ** 2 - errors[:, None] ** 2)
    loss_changes = np.where(availability[:, None], loss_changes, 0.0)
    bounded = np.maximum(loss_changes, 0.0)
    bounded = bounded / (1.0 + bounded)
    task_weights = np.asarray([0.5, 0.25, 0.25], dtype=np.float32)

    np.testing.assert_allclose(diagnostics.active_values, pair_values, rtol=1e-6)
    np.testing.assert_allclose(
        diagnostics.active_normalized_contributions,
        contributions,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostics.active_loss_changes,
        loss_changes,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostics.active_bounded_gains,
        bounded,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostics.active_aggregate_signal,
        np.sum(task_weights[:, None] * bounded, axis=0),
        rtol=1e-6,
        atol=1e-7,
    )
    old_utility = _utility_state(state)
    updated_utility = _utility_state(result.state)
    np.testing.assert_array_equal(
        updated_utility.active_task_evidence_counts[2],
        old_utility.active_task_evidence_counts[2],
    )
    np.testing.assert_array_equal(
        updated_utility.candidate_task_evidence_counts[2],
        old_utility.candidate_task_evidence_counts[2],
    )
    np.testing.assert_array_equal(
        updated_utility.candidate_shadow_weights[2],
        old_utility.candidate_shadow_weights[2],
    )
    np.testing.assert_array_equal(
        updated_utility.target_second_moments[2],
        old_utility.target_second_moments[2],
    )
    if executing_option:
        assert bool(result.behavior_gradient_result.diagnostics.intra_option_source)
    else:
        assert bool(result.behavior_gradient_result.diagnostics.idle_base_source)


def test_diagnostic_lane_is_bit_exact_under_jit() -> None:
    audited = _agent(utility=True)
    plain = _agent(utility=False)
    observation = jnp.asarray([0.25, -0.75, 1.5], dtype=jnp.float32)
    audited_state, seed = _start_idle(audited, observation)
    plain_state, _ = _start_idle(plain, observation, seed=seed)
    transition = _transition(
        audited_state,
        jnp.asarray([-0.5, 2.0, 0.1], dtype=jnp.float32),
        cumulants=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
    )
    audited_result = jax.jit(audited.update_transition)(audited_state, transition)
    plain_result = jax.jit(plain.update_transition)(plain_state, transition)

    assert bool(audited_result.transition_diagnostics.valid)
    assert bool(plain_result.transition_diagnostics.valid)
    assert int(audited_result.action) == int(plain_result.action)
    np.testing.assert_array_equal(
        audited_result.state.current_decision_id,
        plain_result.state.current_decision_id,
    )
    audited_bundle = _bundle(audited_result.state)
    plain_bundle = _bundle(plain_result.state)
    _assert_tree_exact(audited_bundle.oak_state, plain_bundle.oak_state)
    _assert_horde_exact(audited_bundle.horde_state, plain_bundle.horde_state)
    _assert_tree_exact(
        _feature_state(audited_result.state),
        _feature_state(plain_result.state),
    )
    np.testing.assert_array_equal(audited_result.horde_td_errors, plain_result.horde_td_errors)
    _assert_tree_exact(
        audited_result.prototype_feature_lifecycle_diagnostics,
        plain_result.prototype_feature_lifecycle_diagnostics,
    )
    audited_diagnostics = _utility_integration_diagnostics(audited_result)
    assert bool(audited_diagnostics.observation.transaction_applied)
    assert not bool(audited_diagnostics.rebind_required)
    assert not bool(audited_diagnostics.rebind.transaction_applied)
    assert bool(audited_diagnostics.outer_transaction_committed)
    assert plain_result.prototype_feature_utility_diagnostics is None


def test_audit_values_have_no_authority_at_forced_curation_boundary() -> None:
    audited = _agent(utility=True, replacement_interval=1)
    plain = _agent(utility=False, replacement_interval=1)
    observation = jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float32)
    audited_state, seed = _start_idle(audited, observation)
    plain_state, _ = _start_idle(plain, observation, seed=seed)

    utility = _utility_state(audited_state)
    collision = jnp.any(
        jnp.all(
            utility.candidate_descriptors[:, None, :] == utility.active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    eligible = (~collision).astype(jnp.float32)
    utility = utility.replace(
        active_task_utilities=jnp.asarray(
            [[0.99, 0.01], [0.97, 0.03], [0.95, 0.05]],
            dtype=jnp.float32,
        ),
        candidate_shadow_weights=(
            jnp.asarray(
                [[20.0, -30.0, 40.0], [-50.0, 60.0, -70.0], [80.0, -90.0, 100.0]],
                dtype=jnp.float32,
            )
            * eligible[None, :]
        ),
        candidate_task_utilities=(
            jnp.asarray(
                [[0.91, 0.09, 0.81], [0.19, 0.71, 0.29], [0.61, 0.39, 0.51]],
                dtype=jnp.float32,
            )
            * eligible[None, :]
        ),
        candidate_second_moments=(jnp.asarray([11.0, 13.0, 17.0], dtype=jnp.float32) * eligible),
        target_second_moments=jnp.asarray([19.0, 23.0, 29.0], dtype=jnp.float32),
    )
    audited_bundle = _bundle(audited_state)
    audited_state = cast(
        PrototypeAgentState,
        audited_state.replace(oak_state=audited_bundle.replace(feature_utility_state=utility)),
    )
    audited_state = _force_next_primitive(_force_promotion(audited_state))
    plain_state = _force_next_primitive(_force_promotion(plain_state))
    assert bool(audited._checkpoint_state_valid(audited_state))
    assert bool(plain._checkpoint_state_valid(plain_state))

    next_observation = jnp.asarray([-2.0, 0.25, 3.0], dtype=jnp.float32)
    cumulants = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    audited_result = audited.update_transition(
        audited_state,
        _transition(audited_state, next_observation, cumulants=cumulants),
    )
    plain_result = plain.update_transition(
        plain_state,
        _transition(plain_state, next_observation, cumulants=cumulants),
    )

    assert bool(audited_result.transition_diagnostics.valid)
    assert bool(plain_result.transition_diagnostics.valid)
    audited_feature = audited_result.prototype_feature_lifecycle_diagnostics
    plain_feature = plain_result.prototype_feature_lifecycle_diagnostics
    assert audited_feature is not None
    assert plain_feature is not None
    assert bool(audited_feature.lifecycle.curation_committed)
    assert bool(plain_feature.lifecycle.curation_committed)
    _assert_tree_exact(audited_feature, plain_feature)
    audited_bundle = _bundle(audited_result.state)
    plain_bundle = _bundle(plain_result.state)
    _assert_tree_exact(audited_bundle.oak_state, plain_bundle.oak_state)
    _assert_horde_exact(audited_bundle.horde_state, plain_bundle.horde_state)
    _assert_tree_exact(
        _feature_state(audited_result.state),
        _feature_state(plain_result.state),
    )
    np.testing.assert_array_equal(audited_result.action, plain_result.action)
    np.testing.assert_array_equal(
        audited_result.state.current_decision_id,
        plain_result.state.current_decision_id,
    )
    audit = _utility_integration_diagnostics(audited_result)
    assert bool(audit.observation.transaction_applied)
    assert bool(audit.rebind_required)
    assert bool(audit.rebind.transaction_applied)
    assert bool(audit.outer_transaction_committed)


def test_forced_route_rebinds_by_identity_without_a_second_observation() -> None:
    agent = _agent(utility=True, replacement_interval=1)
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float32),
    )
    utility = _utility_state(state)
    candidate_collision = jnp.any(
        jnp.all(
            utility.candidate_descriptors[:, None, :] == utility.active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    candidate_eligible = (~candidate_collision).astype(jnp.float32)
    utility = utility.replace(
        active_task_utilities=jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=jnp.float32),
        candidate_shadow_weights=(
            jnp.asarray(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                dtype=jnp.float32,
            )
            * candidate_eligible[None, :]
        ),
        candidate_task_utilities=(
            jnp.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                dtype=jnp.float32,
            )
            * candidate_eligible[None, :]
        ),
        candidate_second_moments=(
            jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32) * candidate_eligible
        ),
    )
    bundle = _bundle(state)
    state = cast(
        PrototypeAgentState,
        state.replace(oak_state=bundle.replace(feature_utility_state=utility)),
    )
    state = _force_next_primitive(_force_promotion(state))
    assert bool(agent._checkpoint_state_valid(state))
    old_utility = _utility_state(state)
    transition = _transition(
        state,
        jnp.asarray([-2.0, 0.25, 3.0], dtype=jnp.float32),
        cumulants=jnp.asarray([0.4, -0.2], dtype=jnp.float32),
    )

    result = agent.update_transition(state, transition)
    feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert feature_diagnostics is not None
    assert bool(feature_diagnostics.lifecycle.curation_committed)
    integration = _utility_integration_diagnostics(result)
    observation_diagnostics = integration.observation
    rebind_diagnostics = integration.rebind
    assert bool(integration.outer_transaction_committed)
    assert bool(integration.rebind_required)
    assert bool(observation_diagnostics.transaction_applied)
    assert not bool(observation_diagnostics.binding_rebound)
    assert bool(rebind_diagnostics.transaction_applied)
    assert bool(rebind_diagnostics.binding_rebound)
    rebound = _utility_state(result.state)
    binding = _bundle(result.state).consumer_binding
    np.testing.assert_array_equal(rebound.active_descriptors, binding.descriptors)
    assert int(rebound.semantic_generation) == int(binding.semantic_generation)
    assert int(rebound.observation_count) == int(old_utility.observation_count) + 1
    np.testing.assert_array_equal(
        observation_diagnostics.source_active_descriptors,
        old_utility.active_descriptors,
    )
    np.testing.assert_array_equal(
        observation_diagnostics.source_candidate_descriptors,
        old_utility.candidate_descriptors,
    )
    assert int(observation_diagnostics.semantic_generation_before) == int(
        old_utility.semantic_generation
    )
    assert int(observation_diagnostics.semantic_generation_after) == int(
        old_utility.semantic_generation
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.source_active_descriptors,
        rebound.active_descriptors,
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.source_candidate_descriptors,
        rebound.candidate_descriptors,
    )
    assert int(rebind_diagnostics.semantic_generation_before) == int(
        old_utility.semantic_generation
    )
    assert int(rebind_diagnostics.semantic_generation_after) == int(rebound.semantic_generation)

    old_active = [tuple(row) for row in np.asarray(old_utility.active_descriptors)]
    old_candidates = [tuple(row) for row in np.asarray(old_utility.candidate_descriptors)]
    new_active = [tuple(row) for row in np.asarray(rebound.active_descriptors)]
    new_candidates = [tuple(row) for row in np.asarray(rebound.candidate_descriptors)]
    old_collision = np.asarray(
        [descriptor in old_active for descriptor in old_candidates],
        dtype=np.bool_,
    )
    active_survivors = np.asarray(
        [descriptor in old_active for descriptor in new_active],
        dtype=np.bool_,
    )
    new_collision = np.asarray(
        [descriptor in new_active for descriptor in new_candidates],
        dtype=np.bool_,
    )
    candidate_survivors = np.asarray(
        [
            descriptor in old_candidates and not new_collision[index]
            for index, descriptor in enumerate(new_candidates)
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(
        observation_diagnostics.candidate_collision_mask,
        old_collision,
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.active_survivor_mask,
        active_survivors,
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.candidate_survivor_mask,
        candidate_survivors,
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.candidate_collision_mask,
        new_collision,
    )
    np.testing.assert_array_equal(
        rebind_diagnostics.candidate_eligible_mask,
        ~new_collision,
    )
    for new_index, descriptor in enumerate(new_active):
        if descriptor in old_active:
            old_index = old_active.index(descriptor)
            np.testing.assert_array_equal(
                rebound.active_task_utilities[:, new_index],
                observation_diagnostics.active_task_utilities_after[:, old_index],
            )
            np.testing.assert_array_equal(
                rebound.active_task_evidence_counts[:, new_index],
                observation_diagnostics.active_task_evidence_counts_after[:, old_index],
            )
        else:
            np.testing.assert_array_equal(
                rebound.active_task_utilities[:, new_index],
                np.zeros((1 + N_DEMONS,), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                rebound.active_task_evidence_counts[:, new_index],
                np.zeros((1 + N_DEMONS,), dtype=np.int32),
            )
    for new_index, descriptor in enumerate(new_candidates):
        survives = descriptor in old_candidates and descriptor not in new_active
        if survives:
            old_index = old_candidates.index(descriptor)
            np.testing.assert_array_equal(
                rebound.candidate_shadow_weights[:, new_index],
                observation_diagnostics.candidate_shadow_weights_after[:, old_index],
            )
            np.testing.assert_array_equal(
                rebound.candidate_task_utilities[:, new_index],
                observation_diagnostics.candidate_task_utilities_after[:, old_index],
            )
            np.testing.assert_array_equal(
                rebound.candidate_task_evidence_counts[:, new_index],
                observation_diagnostics.candidate_task_evidence_counts_after[:, old_index],
            )
            np.testing.assert_array_equal(
                rebound.candidate_second_moments[new_index],
                observation_diagnostics.candidate_second_moments_after[old_index],
            )
        else:
            np.testing.assert_array_equal(
                rebound.candidate_shadow_weights[:, new_index],
                np.zeros((1 + N_DEMONS,), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                rebound.candidate_task_utilities[:, new_index],
                np.zeros((1 + N_DEMONS,), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                rebound.candidate_task_evidence_counts[:, new_index],
                np.zeros((1 + N_DEMONS,), dtype=np.int32),
            )
            np.testing.assert_array_equal(
                rebound.candidate_second_moments[new_index],
                np.asarray(0.0, dtype=np.float32),
            )
    np.testing.assert_array_equal(
        rebound.target_second_moments,
        observation_diagnostics.target_second_moments_after,
    )
    for collision_index in np.flatnonzero(new_collision):
        np.testing.assert_array_equal(
            rebound.candidate_shadow_weights[:, collision_index],
            np.zeros((1 + N_DEMONS,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            rebound.candidate_task_utilities[:, collision_index],
            np.zeros((1 + N_DEMONS,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            rebound.candidate_task_evidence_counts[:, collision_index],
            np.zeros((1 + N_DEMONS,), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            rebound.candidate_second_moments[collision_index],
            np.asarray(0.0, dtype=np.float32),
        )


def test_cap_is_accepted_but_corrupt_audit_or_digest_rolls_back(
    tmp_path: Path,
) -> None:
    agent = _agent(utility=True, max_observations=1)
    state, _ = _start_idle(
        agent,
        jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32),
    )
    first = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([1.0, 0.25, -0.5], dtype=jnp.float32),
            cumulants=jnp.asarray([0.2, -0.1], dtype=jnp.float32),
        ),
    )
    capped_before = _utility_state(first.state)
    feature_before = _feature_state(first.state)
    second = agent.update_transition(
        first.state,
        _transition(
            first.state,
            jnp.asarray([-0.75, 1.5, 0.1], dtype=jnp.float32),
            cumulants=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        ),
    )
    capped_integration = _utility_integration_diagnostics(second)
    capped = capped_integration.observation
    assert bool(second.transition_diagnostics.valid)
    assert bool(capped_integration.outer_transaction_committed)
    assert not bool(capped_integration.rebind_required)
    assert not bool(capped_integration.rebind.transaction_applied)
    assert bool(capped.capacity_capped)
    assert not bool(capped.transaction_applied)
    _assert_tree_exact(_utility_state(second.state), capped_before)
    _assert_tree_exact(_feature_state(second.state), feature_before)
    assert int(second.state.step_count) == 2
    assert int(_bundle(second.state).oak_state.step_count) == 2
    assert int(_bundle(second.state).horde_state.step_count) == 2

    bundle = _bundle(second.state)
    bad_utility = bundle.feature_utility_state.replace(
        active_task_utilities=bundle.feature_utility_state.active_task_utilities.at[0, 0].set(
            jnp.asarray(jnp.inf, dtype=jnp.float32)
        )
    )
    bad_digest = bundle.schema_digest.at[0].set(bundle.schema_digest[0] ^ 1)
    bad_consumer_digest = bundle.consumer_state.schema_digest.at[0].set(
        bundle.consumer_state.schema_digest[0] ^ 1
    )
    corrupt_states = (
        second.state.replace(oak_state=bundle.replace(feature_utility_state=bad_utility)),
        second.state.replace(oak_state=bundle.replace(schema_digest=bad_digest)),
        second.state.replace(
            oak_state=bundle.replace(
                consumer_state=bundle.consumer_state.replace(schema_digest=bad_consumer_digest)
            )
        ),
    )
    for index, corrupt in enumerate(corrupt_states):
        assert not bool(agent._checkpoint_state_valid(corrupt))
        with pytest.raises(ValueError, match="inconsistent"):
            save_prototype_checkpoint(agent, corrupt, tmp_path / f"corrupt-{index}")
        rejected = agent.update_transition(
            corrupt,
            _transition(
                corrupt,
                jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
                cumulants=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            ),
        )
        assert bool(rejected.transition_diagnostics.rejected)
        _assert_tree_exact(rejected.state, corrupt)


def test_v14_round_trip_and_v13_v14_loader_boundary_are_unambiguous(
    tmp_path: Path,
) -> None:
    enabled = _agent(utility=True, replacement_interval=1)
    enabled_state, _ = _start_idle(
        enabled,
        jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float32),
    )
    enabled_state = _force_next_primitive(_force_promotion(enabled_state))
    routed = enabled.update_transition(
        enabled_state,
        _transition(
            enabled_state,
            jnp.asarray([-2.0, 0.25, 3.0], dtype=jnp.float32),
            cumulants=jnp.asarray([0.4, -0.2], dtype=jnp.float32),
        ),
    )
    routed_diagnostics = _utility_integration_diagnostics(routed)
    assert bool(routed.transition_diagnostics.valid)
    assert bool(routed_diagnostics.rebind_required)
    assert bool(routed_diagnostics.rebind.transaction_applied)
    assert bool(routed_diagnostics.outer_transaction_committed)
    enabled_state = routed.state
    persisted_utility = _utility_state(enabled_state)
    assert int(persisted_utility.semantic_generation) > 0
    assert int(persisted_utility.observation_count) > 0
    assert np.any(np.asarray(persisted_utility.active_task_evidence_counts) > 0)
    assert np.any(np.asarray(persisted_utility.target_second_moments) > 0.0)
    enabled_path = tmp_path / "enabled-v14"
    save_prototype_checkpoint(enabled, enabled_state, enabled_path)
    assert load_checkpoint_metadata(enabled_path)["schema"] == (
        PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA
    )
    restored_enabled, restored_enabled_state = load_prototype_checkpoint(enabled_path)
    assert restored_enabled.to_config() == enabled.to_config()
    _assert_tree_exact(restored_enabled_state, enabled_state)

    plain = _agent(utility=False)
    plain_state = plain.init(jr.key(8))
    plain_path = tmp_path / "plain-v13"
    save_prototype_checkpoint(plain, plain_state, plain_path)
    assert load_checkpoint_metadata(plain_path)["schema"] == PROTOTYPE_CHECKPOINT_SCHEMA
    restored_plain, restored_plain_state = load_prototype_checkpoint(plain_path)
    assert restored_plain.to_config() == plain.to_config()
    _assert_tree_exact(restored_plain_state, plain_state)

    impossible_v13 = tmp_path / "audit-relabeled-v13"
    enabled_config = enabled.to_config()
    save_checkpoint(
        enabled_state,
        impossible_v13,
        metadata={
            "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
            "agent_config": enabled_config,
            "config_sha256": _canonical_digest(enabled_config),
        },
    )
    with pytest.raises(ValueError, match="v14"):
        load_prototype_checkpoint(impossible_v13)

    semantic_tamper = tmp_path / "audit-semantic-tamper-v14"
    tampered_utility_config = {
        **enabled_config["prototype_feature_utility"],
        "utility_decay": 0.5,
    }
    tampered_config = {
        **enabled_config,
        "prototype_feature_utility": tampered_utility_config,
    }
    save_checkpoint(
        enabled_state,
        semantic_tamper,
        metadata={
            "schema": PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
            "agent_config": tampered_config,
            "config_sha256": _canonical_digest(tampered_config),
        },
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_prototype_checkpoint(semantic_tamper)

    impossible_v14 = tmp_path / "plain-relabeled-v14"
    plain_config = plain.to_config()
    save_checkpoint(
        plain_state,
        impossible_v14,
        metadata={
            "schema": PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
            "agent_config": plain_config,
            "config_sha256": _canonical_digest(plain_config),
        },
    )
    with pytest.raises(ValueError, match="feature utility|utility-enabled"):
        load_prototype_checkpoint(impossible_v14)
