# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Standalone contracts for one lifecycle shared by OaK and a linear Horde."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.interaction_features import (
    InteractionCurationPriorityOverride,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_feature_lifecycle import (
    PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
    PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA,
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    PrototypeFeatureLifecycleHordeResult,
    load_prototype_feature_lifecycle_checkpoint,
    save_prototype_feature_lifecycle_checkpoint,
)
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    create_horde_spec,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _config(
    *,
    managed_horde_demons: int = 2,
    replacement_interval: int = 1,
    max_observations: int = 100,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=4,
        active_pair_slots=2,
        candidate_pair_slots=6,
        n_tasks=1 + managed_horde_demons if managed_horde_demons else 2,
        n_options=2,
        n_primitive_actions=2,
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
        max_observations=max_observations,
        managed_horde_demons=managed_horde_demons,
    )


def _oak_agent(config: PrototypeFeatureLifecycleConfig) -> OaKAgent:
    specs = tuple(
        SubtaskSpec(feature_index=index) for index in config.option_subtask_feature_indices
    )
    return OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=specs,
                observation_dim=config.total_feature_dim,
                n_primitive_actions=config.n_primitive_actions,
                base_hidden_sizes=(),
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        )
    )


def _materialize_host_timers(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_host_timers(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_host_timers(right))
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _tree_nbytes(tree: Any) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(tree))


def _consumer_binding(state) -> PrototypeFeatureConsumerBinding:
    return PrototypeFeatureConsumerBinding(
        semantic_generation=state.router_state.generation_count,
        semantic_generation_words=state.router_state.generation_words,
        descriptors=state.router_state.descriptors,
    )


def _event(*, allow_curation: bool = True) -> PrototypeFeatureLifecycleEvent:
    return PrototypeFeatureLifecycleEvent(
        observation=jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
        targets=jnp.asarray([0.75, jnp.nan, jnp.nan], dtype=jnp.float32),
        next_observation=jnp.asarray([-1.0, 2.0, 4.0, 0.25], dtype=jnp.float32),
        allow_curation=jnp.asarray(allow_curation, dtype=jnp.bool_),
    )


def _force_promotion(lifecycle: PrototypeFeatureLifecycle, state):
    learner = state.learner_state
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
    candidate_index = next(index for index, pair in enumerate(candidates) if pair not in active)
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner = learner.replace(
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        ages=jnp.full_like(learner.ages, 10),
        candidate_utilities=candidate_utilities,
        candidate_ages=jnp.full_like(learner.candidate_ages, 10),
        step_count=jnp.asarray(10, dtype=jnp.int32),
        step_words=jnp.asarray((0, 10), dtype=jnp.uint32),
    )
    forced = state.replace(
        learner_state=learner,
        observe_count=jnp.asarray(10, dtype=jnp.int32),
        observe_words=jnp.asarray((0, 10), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(forced))
    return forced


def _seed_oak(
    lifecycle: PrototypeFeatureLifecycle,
    state,
    next_observation: jax.Array,
    *,
    step_count: int,
) -> OaKState:
    config = lifecycle.config
    oak = _oak_agent(config).init(jr.key(71))
    stomp = oak.stomp_state
    width = config.total_feature_dim
    head_weights = tuple(
        jnp.arange(width, dtype=jnp.float32)[None, :] + 100.0 * (index + 1)
        for index in range(config.n_total_actions)
    )
    head_traces = tuple(
        (
            jnp.arange(width, dtype=jnp.float32)[None, :] + 1_000.0 * (index + 1),
            trace_pair[1],
        )
        for index, trace_pair in enumerate(stomp.base_learner_state.head_traces)
    )
    base_learner = stomp.base_learner_state.replace(
        head_params=stomp.base_learner_state.head_params.replace(weights=head_weights),
        head_traces=head_traces,
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        step_words=jnp.asarray((0, step_count), dtype=jnp.uint32),
    )
    policy_values = jnp.arange(
        config.n_options * config.n_primitive_actions * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, config.n_primitive_actions, width)
    model_values = jnp.arange(
        config.n_options * width * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, width, width)
    stomp = stomp.replace(
        base_learner_state=base_learner,
        base_last_obs=lifecycle.augment(state, next_observation),
        base_last_action=jnp.asarray(0, dtype=jnp.int32),
        last_primitive_action=jnp.asarray(0, dtype=jnp.int32),
        option_policies=stomp.option_policies.replace(
            q_weights=policy_values + 2_000.0,
            traces=policy_values + 3_000.0,
        ),
        option_models=stomp.option_models.replace(next_state_weights=model_values + 4_000.0),
        executing_option=jnp.asarray(-1, dtype=jnp.int32),
        option_start_obs=jnp.arange(width, dtype=jnp.float32) + 5_000.0,
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        step_words=jnp.asarray((0, step_count), dtype=jnp.uint32),
    )
    return cast(
        OaKState,
        oak.replace(
            stomp_state=stomp,
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
            step_words=jnp.asarray((0, step_count), dtype=jnp.uint32),
        ),
    )


def _seed_horde(
    lifecycle: PrototypeFeatureLifecycle,
    *,
    step_count: int,
) -> MultiHeadMLPState:
    demons = tuple(
        GVFSpec(
            name=f"demon-{index}",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.8,
            cumulant_index=index,
        )
        for index in range(lifecycle.config.managed_horde_demons)
    )
    horde = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(),
        step_size=0.1,
    ).init(lifecycle.config.total_feature_dim, jr.key(73))
    width = lifecycle.config.total_feature_dim
    weights = tuple(
        jnp.arange(width, dtype=jnp.float32)[None, :] + 6_000.0 * (index + 1)
        for index in range(lifecycle.config.managed_horde_demons)
    )
    traces = tuple(
        (
            jnp.arange(width, dtype=jnp.float32)[None, :] + 7_000.0 * (index + 1),
            trace_pair[1],
        )
        for index, trace_pair in enumerate(horde.head_traces)
    )
    return cast(
        MultiHeadMLPState,
        horde.replace(
            head_params=horde.head_params.replace(weights=weights),
            head_traces=traces,
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
            step_words=jnp.asarray((0, step_count), dtype=jnp.uint32),
        ),
    )


def _route_last_axis(
    values: jax.Array,
    old_descriptors: jax.Array,
    new_descriptors: jax.Array,
    base_dim: int,
) -> jax.Array:
    old_rows = [tuple(row) for row in np.asarray(old_descriptors).tolist()]
    new_rows = [tuple(row) for row in np.asarray(new_descriptors).tolist()]
    tail = jnp.zeros_like(values[..., base_dim:])
    for destination, descriptor in enumerate(new_rows):
        if descriptor in old_rows:
            source = old_rows.index(descriptor)
            tail = tail.at[..., destination].set(values[..., base_dim + source])
    return jnp.concatenate((values[..., :base_dim], tail), axis=-1)


def test_managed_horde_config_is_canonical_v2_and_balances_control_utility() -> None:
    config = _config()
    lifecycle = PrototypeFeatureLifecycle(config)
    payload = config.to_config()

    assert payload["schema"] == PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA
    assert payload["managed_horde_demons"] == 2
    assert PrototypeFeatureLifecycleConfig.from_config(payload) == config
    assert lifecycle.learner.to_config()["task_utility_weights"] == [0.5, 0.25, 0.25]

    legacy_payload = _config(managed_horde_demons=0).to_config()
    assert legacy_payload["schema"] == PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA
    assert "managed_horde_demons" not in legacy_payload

    noncanonical_v2 = dict(payload)
    noncanonical_v2["managed_horde_demons"] = 0
    with pytest.raises(ValueError, match="strict integer"):
        PrototypeFeatureLifecycleConfig.from_config(noncanonical_v2)
    with pytest.raises(ValueError, match="n_tasks"):
        PrototypeFeatureLifecycleConfig(
            **{
                **config.__dict__,
                "n_tasks": 2,
            }
        )


def test_horde_validator_accepts_only_exact_finite_linear_lms_state() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _seed_horde(lifecycle, step_count=1)

    assert bool(lifecycle.horde_state_valid(state))
    assert bool(jax.jit(lifecycle.horde_state_valid)(state))

    nonlinear = MultiHeadMLPLearner(
        n_heads=2,
        hidden_sizes=(3,),
        step_size=0.1,
    ).init(lifecycle.config.total_feature_dim, jr.key(5))
    wrong_head_count = MultiHeadMLPLearner(
        n_heads=1,
        hidden_sizes=(),
        step_size=0.1,
    ).init(lifecycle.config.total_feature_dim, jr.key(6))
    assert not bool(lifecycle.horde_state_valid(nonlinear))
    assert not bool(lifecycle.horde_state_valid(wrong_head_count))

    nan_weights = (
        state.head_params.weights[0].at[0, 0].set(jnp.nan),
        state.head_params.weights[1],
    )
    assert not bool(
        lifecycle.horde_state_valid(
            state.replace(head_params=state.head_params.replace(weights=nan_weights))
        )
    )
    unequal_optimizers = list(state.head_optimizer_states)
    unequal_optimizers[0] = (
        unequal_optimizers[0][0].replace(step_size=jnp.asarray(0.2, dtype=jnp.float32)),
        unequal_optimizers[0][1],
    )
    assert not bool(
        lifecycle.horde_state_valid(state.replace(head_optimizer_states=tuple(unequal_optimizers)))
    )
    assert not bool(lifecycle.horde_state_valid(state.replace(uptime_s=-1.0)))


def test_api_modes_reject_missing_or_unmanaged_horde() -> None:
    managed = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    managed_state = managed.init(jr.key(1))
    event = _event()
    oak = _seed_oak(managed, managed_state, event.next_observation, step_count=1)
    horde = _seed_horde(managed, step_count=1)

    with pytest.raises(ValueError, match="observe_and_route_with_horde"):
        managed.observe_and_route(
            managed_state,
            oak,
            _consumer_binding(managed_state),
            event,
        )
    nonlinear = MultiHeadMLPLearner(
        n_heads=2,
        hidden_sizes=(3,),
        step_size=0.1,
    ).init(managed.config.total_feature_dim, jr.key(3))
    with pytest.raises(ValueError, match="linear Horde"):
        managed.observe_and_route_with_horde(
            managed_state,
            oak,
            nonlinear,
            _consumer_binding(managed_state),
            event,
        )

    legacy = PrototypeFeatureLifecycle(_config(managed_horde_demons=0, replacement_interval=0))
    legacy_state = legacy.init(jr.key(2))
    legacy_event = PrototypeFeatureLifecycleEvent(
        observation=event.observation,
        targets=jnp.asarray([0.75, jnp.nan], dtype=jnp.float32),
        next_observation=event.next_observation,
        allow_curation=event.allow_curation,
    )
    legacy_oak = _seed_oak(
        legacy,
        legacy_state,
        legacy_event.next_observation,
        step_count=1,
    )
    with pytest.raises(ValueError, match="does not manage"):
        legacy.observe_and_route_with_horde(
            legacy_state,
            legacy_oak,
            horde,
            _consumer_binding(legacy_state),
            legacy_event,
        )


def test_safe_curation_routes_every_horde_weight_and_weight_trace_atomically() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _force_promotion(lifecycle, lifecycle.init(jr.key(11)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=11)
    horde = _seed_horde(lifecycle, step_count=11)
    old_descriptors = state.router_state.descriptors

    result = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        _consumer_binding(state),
        event,
    )

    assert isinstance(result, PrototypeFeatureLifecycleHordeResult)
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.curation_committed)
    assert bool(result.horde_diagnostics.horde_state_values_valid)
    assert bool(result.horde_diagnostics.pre_step_parity_valid)
    assert bool(result.horde_diagnostics.post_step_parity_valid)
    assert not bool(result.horde_diagnostics.lifecycle_capacity_capped)
    assert int(result.state.observe_count) == 11
    assert int(result.oak_state.step_count) == 11
    assert int(result.horde_state.step_count) == 11

    expected_weights = tuple(
        _route_last_axis(
            values,
            old_descriptors,
            result.state.router_state.descriptors,
            lifecycle.config.base_feature_dim,
        )
        for values in horde.head_params.weights
    )
    expected_traces = tuple(
        (
            _route_last_axis(
                trace_pair[0],
                old_descriptors,
                result.state.router_state.descriptors,
                lifecycle.config.base_feature_dim,
            ),
            trace_pair[1],
        )
        for trace_pair in horde.head_traces
    )
    expected_horde = horde.replace(
        head_params=horde.head_params.replace(weights=expected_weights),
        head_traces=expected_traces,
    )
    _assert_tree_exact(result.horde_state, expected_horde)
    np.testing.assert_array_equal(
        result.consumer_binding.descriptors,
        result.state.router_state.descriptors,
    )
    assert int(result.consumer_binding.semantic_generation) == 1


def test_curation_priority_override_is_threaded_without_changing_promotion_gate() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _force_promotion(lifecycle, lifecycle.init(jr.key(111)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=11)
    horde = _seed_horde(lifecycle, step_count=11)
    binding = _consumer_binding(state)

    legacy = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        binding,
        event,
    )
    override = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(True, dtype=jnp.bool_),
        active_ranks=jnp.asarray([100.0, -100.0], dtype=jnp.float32),
        candidate_ranks=state.learner_state.candidate_utilities,
    )
    ranked = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        binding,
        event,
        curation_priority_override=override,
    )

    assert bool(legacy.diagnostics.curation_committed)
    assert bool(ranked.diagnostics.curation_committed)
    assert not bool(legacy.diagnostics.curation_priority_override_enabled)
    assert not bool(legacy.diagnostics.curation_priority_override_applied)
    assert bool(ranked.diagnostics.curation_priority_override_enabled)
    assert bool(ranked.diagnostics.curation_priority_override_applied)
    assert int(ranked.diagnostics.curation_selected_active_worst_slot) == 1
    legacy_changed = jnp.any(
        legacy.state.router_state.descriptors
        != state.router_state.descriptors,
        axis=1,
    )
    ranked_changed = jnp.any(
        ranked.state.router_state.descriptors
        != state.router_state.descriptors,
        axis=1,
    )
    np.testing.assert_array_equal(legacy_changed, jnp.asarray([True, False]))
    np.testing.assert_array_equal(ranked_changed, jnp.asarray([False, True]))
    assert int(legacy.state.router_state.route_count) == 1
    assert int(ranked.state.router_state.route_count) == 1


@pytest.mark.parametrize(
    "invalid_kind",
    ["stale_binding", "step_parity", "nonfinite_horde"],
)
def test_shared_precondition_failure_is_an_exact_three_state_noop(
    invalid_kind: str,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _force_promotion(lifecycle, lifecycle.init(jr.key(12)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=11)
    horde = _seed_horde(lifecycle, step_count=11)
    binding = _consumer_binding(state)
    if invalid_kind == "stale_binding":
        binding = binding.replace(
            semantic_generation=binding.semantic_generation + jnp.int32(1),
            semantic_generation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    elif invalid_kind == "step_parity":
        horde = horde.replace(
            step_count=jnp.asarray(12, dtype=jnp.int32),
            step_words=jnp.asarray((0, 12), dtype=jnp.uint32),
        )
    else:
        weights = (
            horde.head_params.weights[0].at[0, 0].set(jnp.nan),
            horde.head_params.weights[1],
        )
        horde = horde.replace(head_params=horde.head_params.replace(weights=weights))

    result = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        binding,
        event,
    )

    assert not bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.curation_committed)
    if invalid_kind == "step_parity":
        assert not bool(result.horde_diagnostics.pre_step_parity_valid)
        assert not bool(result.horde_diagnostics.post_step_parity_valid)
    if invalid_kind == "nonfinite_horde":
        assert not bool(result.horde_diagnostics.horde_state_values_valid)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)
    _assert_tree_exact(result.horde_state, horde)
    _assert_tree_exact(result.consumer_binding, binding)


def test_capacity_capped_shared_lifecycle_allows_equal_consumers_to_advance() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0, max_observations=1))
    state = lifecycle.init(jr.key(13))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=1)
    horde = _seed_horde(lifecycle, step_count=1)
    first = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        _consumer_binding(state),
        event,
    )
    assert bool(first.diagnostics.transaction_applied)
    assert int(first.state.observe_count) == 1

    advanced_step = jnp.asarray(5, dtype=jnp.int32)
    advanced_words = jnp.asarray((0, 5), dtype=jnp.uint32)
    advanced_oak = first.oak_state.replace(
        step_count=advanced_step,
        step_words=advanced_words,
        stomp_state=first.oak_state.stomp_state.replace(
            step_count=advanced_step,
            step_words=advanced_words,
            base_learner_state=(
                first.oak_state.stomp_state.base_learner_state.replace(
                    step_count=advanced_step,
                    step_words=advanced_words,
                )
            ),
        ),
    )
    advanced_horde = first.horde_state.replace(
        step_count=advanced_step,
        step_words=advanced_words,
    )
    capped = lifecycle.observe_and_route_with_horde(
        first.state,
        advanced_oak,
        advanced_horde,
        first.consumer_binding,
        event,
    )

    assert bool(capped.horde_diagnostics.lifecycle_capacity_capped)
    assert bool(capped.horde_diagnostics.pre_step_parity_valid)
    assert bool(capped.horde_diagnostics.post_step_parity_valid)
    assert not bool(capped.diagnostics.update_capacity_available)
    assert not bool(capped.diagnostics.transaction_applied)
    _assert_tree_exact(capped.state, first.state)
    _assert_tree_exact(capped.oak_state, advanced_oak)
    _assert_tree_exact(capped.horde_state, advanced_horde)
    _assert_tree_exact(capped.consumer_binding, first.consumer_binding)

    prepared = lifecycle.prepare_observe_and_route_with_horde(
        first.state,
        advanced_oak,
        advanced_horde,
        first.consumer_binding,
        event,
    )
    receipt = lifecycle.horde_external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    adopted = lifecycle.adopt_prepared_route_with_horde(
        first.state,
        advanced_oak,
        advanced_horde,
        first.consumer_binding,
        prepared,
        receipt,
    )
    expected_next = lifecycle.augment(first.state, event.next_observation)
    _assert_tree_exact(adopted.result.state, first.state)
    _assert_tree_exact(adopted.result.oak_state, advanced_oak)
    _assert_tree_exact(adopted.result.horde_state, advanced_horde)
    _assert_tree_exact(adopted.result.consumer_binding, first.consumer_binding)
    np.testing.assert_array_equal(
        np.asarray(adopted.result.next_augmented_observation),
        np.asarray(expected_next),
    )
    assert np.all(np.isnan(np.asarray(adopted.result.predictions)))
    assert np.all(np.isnan(np.asarray(adopted.result.errors)))
    assert not bool(adopted.result.diagnostics.transaction_applied)
    assert not bool(adopted.result.diagnostics.routing_attempted)
    assert not bool(adopted.diagnostics.transaction_applied)
    assert not bool(adopted.diagnostics.destination_adopted)
    assert not bool(adopted.diagnostics.ordinary_update_retained)
    assert bool(adopted.diagnostics.rejected)
    assert int(adopted.diagnostics.adoption_learner_update_evaluations) == 0

    untrusted = prepared.replace(
        destination_result=prepared.destination_result.replace(
            next_augmented_observation=jnp.full_like(expected_next, 99.0),
        )
    )
    untrusted_receipt = lifecycle.horde_external_readiness_receipt(
        untrusted,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    sanitized = lifecycle.adopt_prepared_route_with_horde(
        first.state,
        advanced_oak,
        advanced_horde,
        first.consumer_binding,
        untrusted,
        untrusted_receipt,
    )
    np.testing.assert_array_equal(
        np.asarray(sanitized.result.next_augmented_observation),
        np.asarray(expected_next),
    )


def test_horde_resource_accounting_and_v2_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    state = lifecycle.init(jr.key(14))
    horde = _seed_horde(lifecycle, step_count=1)
    budget = lifecycle.resource_budget(state, horde)
    width = lifecycle.config.total_feature_dim
    demons = lifecycle.config.managed_horde_demons

    assert budget.managed_horde_demons == demons
    assert budget.horde_persistent_state_nbytes == _tree_nbytes(horde)
    assert budget.managed_horde_consumer_nbytes == 2 * demons * width * 4
    assert budget.managed_total_consumer_nbytes == (
        budget.managed_oak_consumer_nbytes + budget.managed_horde_consumer_nbytes
    )
    assert budget.input_route_feature_groups == 33
    assert budget.internal_horde_template_nbytes > 0
    assert budget.to_config()["managed_horde_demons"] == demons

    path = tmp_path / "managed-horde-lifecycle"
    save_prototype_feature_lifecycle_checkpoint(lifecycle, state, path)
    restored_lifecycle, restored_state = load_prototype_feature_lifecycle_checkpoint(path)
    assert restored_lifecycle.config == lifecycle.config
    _assert_tree_exact(restored_state, state)
    assert (
        restored_lifecycle.to_config()["schema"] == PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA
    )


def test_shared_transaction_is_eager_jit_exact() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _force_promotion(lifecycle, lifecycle.init(jr.key(15)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=11)
    horde = _seed_horde(lifecycle, step_count=11)
    binding = _consumer_binding(state)

    eager = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        binding,
        event,
    )
    compiled = jax.jit(lifecycle.observe_and_route_with_horde)(
        state,
        oak,
        horde,
        binding,
        event,
    )
    _assert_tree_exact(compiled, eager)


def test_external_horde_veto_retains_both_ordinary_consumers_and_clocks() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _force_promotion(lifecycle, lifecycle.init(jr.key(616)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation, step_count=11)
    horde = _seed_horde(lifecycle, step_count=11)
    binding = _consumer_binding(state)
    prepared = lifecycle.prepare_observe_and_route_with_horde(
        state,
        oak,
        horde,
        binding,
        event,
    )
    receipt = lifecycle.horde_external_readiness_receipt(
        prepared,
        jnp.asarray(False, dtype=jnp.bool_),
    )

    eager = lifecycle.adopt_prepared_route_with_horde(
        state,
        oak,
        horde,
        binding,
        prepared,
        receipt,
    )
    compiled = jax.jit(lifecycle.adopt_prepared_route_with_horde)(
        state,
        oak,
        horde,
        binding,
        prepared,
        receipt,
    )

    _assert_tree_exact(eager, compiled)
    _assert_tree_exact(eager.result, prepared.ordinary_result)
    _assert_tree_exact(eager.result.oak_state, oak)
    _assert_tree_exact(eager.result.horde_state, horde)
    _assert_tree_exact(eager.result.consumer_binding, binding)
    assert bool(eager.diagnostics.ordinary_update_retained)
    assert bool(eager.diagnostics.external_curation_rolled_back)
    assert int(eager.result.state.observe_count) == 11
    assert int(eager.result.state.committed_curation_count) == 0
    assert int(eager.result.state.rolled_back_curation_count) == 1
    np.testing.assert_array_equal(eager.result.state.observe_words, (0, 11))
    np.testing.assert_array_equal(eager.result.oak_state.step_words, (0, 11))
    np.testing.assert_array_equal(eager.result.horde_state.step_words, (0, 11))
    transient = lifecycle.external_transaction_resource_budget(prepared, receipt)
    assert transient.managed_horde_demons == lifecycle.config.managed_horde_demons
    assert transient.prepared_route_logical_nbytes == _tree_nbytes(prepared)
    assert transient.readiness_receipt_logical_nbytes == _tree_nbytes(receipt)
    assert transient.source_horde_state_nbytes == _tree_nbytes(horde)
    assert (
        transient.lifecycle_persistent_state_nbytes_before
        == lifecycle.resource_budget(state, horde).lifecycle_state_nbytes
    )
    assert (
        transient.lifecycle_persistent_state_nbytes_after
        == transient.lifecycle_persistent_state_nbytes_before
    )
    assert transient.learner_update_evaluations_per_prepare == 1
    assert transient.learner_update_evaluations_per_adopt == 0
    assert transient.router_evaluations_per_prepare == 2
    assert transient.router_evaluations_per_adopt == 0
    assert transient.persistent_capacity_growth == 0
