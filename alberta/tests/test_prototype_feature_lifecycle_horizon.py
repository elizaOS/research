# mypy: disable-error-code="attr-defined,call-arg,no-untyped-call,no-untyped-def"
"""Exact-horizon contracts for the fixed-bank Prototype feature lifecycle."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouter,
    FeatureBankRouteResult,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_feature_lifecycle import (
    PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_DELTA_NBYTES,
    PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_NBYTES,
    PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA,
    PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
    PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_DELTA_NBYTES,
    PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES,
    PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_DELTA_NBYTES,
    PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES,
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    load_prototype_feature_lifecycle_checkpoint,
    measure_prototype_feature_lifecycle_state_nbytes,
    migrate_legacy_prototype_feature_consumer_binding,
    migrate_legacy_prototype_feature_lifecycle_state,
    prototype_feature_lifecycle_counter_nbytes,
    prototype_feature_lifecycle_lifetime_counter_nbytes,
    save_prototype_feature_lifecycle_checkpoint,
)

pytestmark = pytest.mark.integration

_INT32_MAX = 2_147_483_647
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name in {
        "test_jit_and_scan_cross_low_word_carry_atomically",
        "test_terminal_all_ones_rejects_eager_and_jit_without_key_or_state_change",
    }:
        yield
    else:
        with jax.disable_jit():
            yield


def _config(
    *,
    replacement_interval: int = 0,
    max_observations: int = _UINT64_MAX,
    managed_horde_demons: int = 0,
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


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _telemetry(value: int) -> jax.Array:
    return jnp.asarray(min(value, _INT32_MAX), dtype=jnp.int32)


def _next_value(value: int) -> int:
    return value if value == _UINT64_MAX else value + 1


def _event(*, allow_curation: bool = False, n_tasks: int = 2):
    targets = [0.75, *([jnp.nan] * (n_tasks - 1))]
    return PrototypeFeatureLifecycleEvent(
        observation=jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
        targets=jnp.asarray(targets, dtype=jnp.float32),
        next_observation=jnp.asarray([-1.0, 2.0, 4.0, 0.25], dtype=jnp.float32),
        allow_curation=jnp.asarray(allow_curation, dtype=jnp.bool_),
    )


def _binding(state) -> PrototypeFeatureConsumerBinding:
    return PrototypeFeatureConsumerBinding(
        semantic_generation=state.router_state.generation_count,
        semantic_generation_words=state.router_state.generation_words,
        descriptors=state.router_state.descriptors,
    )


def _with_history(
    lifecycle: PrototypeFeatureLifecycle,
    state,
    observations: int,
    *,
    deferred: int = 0,
    committed: int = 0,
    rolled_back: int = 0,
):
    interval = lifecycle.config.replacement_interval
    learner = state.learner_state.replace(
        step_count=_telemetry(observations),
        step_words=_words(observations),
        replacement_phase=jnp.asarray(
            0 if interval == 0 else observations % interval,
            dtype=jnp.int32,
        ),
    )
    router = dataclasses.replace(
        state.router_state,
        route_count=_telemetry(committed),
        generation_count=_telemetry(committed),
        route_words=_words(committed),
        generation_words=_words(committed),
    )
    return state.replace(
        learner_state=learner,
        router_state=router,
        observe_count=_telemetry(observations),
        observe_words=_words(observations),
        deferred_curation_count=_telemetry(deferred),
        deferred_curation_words=_words(deferred),
        committed_curation_count=_telemetry(committed),
        committed_curation_words=_words(committed),
        rolled_back_curation_count=_telemetry(rolled_back),
        rolled_back_curation_words=_words(rolled_back),
    )


def _oak_agent(config: PrototypeFeatureLifecycleConfig) -> OaKAgent:
    return OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(SubtaskSpec(0), SubtaskSpec(1)),
                observation_dim=config.total_feature_dim,
                n_primitive_actions=config.n_primitive_actions,
                base_hidden_sizes=(),
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        )
    )


def _post_update_oak(
    lifecycle: PrototypeFeatureLifecycle,
    state,
    event: PrototypeFeatureLifecycleEvent,
    *,
    source_value: int,
) -> OaKState:
    post_value = _next_value(source_value)
    oak = _oak_agent(lifecycle.config).init(jr.key(71))
    stomp = oak.stomp_state.replace(
        base_last_obs=lifecycle.augment(state, event.next_observation),
        base_last_action=jnp.asarray(0, dtype=jnp.int32),
        last_primitive_action=jnp.asarray(0, dtype=jnp.int32),
        executing_option=jnp.asarray(-1, dtype=jnp.int32),
        step_count=_telemetry(post_value),
        step_words=_words(post_value),
    )
    return cast(
        OaKState,
        oak.replace(
            stomp_state=stomp,
            step_count=_telemetry(post_value),
            step_words=_words(post_value),
        ),
    )


def _post_update_horde(lifecycle: PrototypeFeatureLifecycle, source_value: int):
    state = MultiHeadMLPLearner(
        n_heads=lifecycle.config.managed_horde_demons,
        hidden_sizes=(),
        step_size=0.1,
    ).init(lifecycle.config.total_feature_dim, jr.key(73))
    post_value = _next_value(source_value)
    return state.replace(
        step_count=_telemetry(post_value),
        step_words=_words(post_value),
    )


def _force_promotion(state):
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
    candidate_index = next(
        index for index, pair in enumerate(candidates) if pair not in active
    )
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    return state.replace(
        learner_state=learner.replace(
            utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
            ages=jnp.full_like(learner.ages, 10),
            candidate_utilities=candidate_utilities,
            candidate_ages=jnp.full_like(learner.candidate_ages, 10),
        )
    )


def _materialize_host_leaves(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_host_leaves(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_host_leaves(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


class _DuplicateDescriptorRouter(FeatureBankRouter):
    def route(
        self,
        state,
        consumers,
        new_descriptors,
        *,
        feature_axes=None,
        carry_survivors=True,
    ) -> FeatureBankRouteResult:
        duplicate = new_descriptors.at[1].set(new_descriptors[0])
        return super().route(
            state,
            consumers,
            duplicate,
            feature_axes=feature_axes,
            carry_survivors=carry_survivors,
        )


def test_full_default_lifetime_schema_and_exact_resource_delta() -> None:
    config = _config()
    lifecycle = PrototypeFeatureLifecycle(config)
    state = lifecycle.init(jr.key(1))
    budget = lifecycle.resource_budget(state)

    assert config.max_observations == _UINT64_MAX
    assert config.to_config()["schema"] == PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA
    assert PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA.endswith(".v2")
    assert PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES == 12
    assert PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert prototype_feature_lifecycle_lifetime_counter_nbytes() == 12
    assert PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES == 48
    assert PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_DELTA_NBYTES == 32
    assert prototype_feature_lifecycle_counter_nbytes() == 48
    assert budget.lifecycle_telemetry_counter_nbytes == 16
    assert budget.lifecycle_exact_counter_nbytes == 32
    assert budget.lifecycle_counter_delta_nbytes == 32
    assert budget.lifecycle_counter_nbytes == 48
    assert budget.lifecycle_state_nbytes == measure_prototype_feature_lifecycle_state_nbytes(
        state
    )
    assert budget.consumer_binding_persistent_nbytes == 28
    assert PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_NBYTES == 12
    assert PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_DELTA_NBYTES == 8
    assert budget.consumer_binding_generation_nbytes == 12
    assert budget.consumer_binding_generation_delta_nbytes == 8

    legacy_config = dataclasses.replace(config, max_observations=100)
    legacy_payload = legacy_config.to_config()
    legacy_payload["schema"] = "alberta.prototype-feature-lifecycle.config.v1"
    legacy_payload.pop("state_schema")
    assert PrototypeFeatureLifecycleConfig.from_config(legacy_payload) == legacy_config
    with pytest.raises(ValueError, match="legacy max_observations"):
        PrototypeFeatureLifecycleConfig.from_config(
            {**legacy_payload, "max_observations": _UINT64_MAX}
        )
    with pytest.raises(ValueError, match="max_observations"):
        dataclasses.replace(config, max_observations=_UINT64_MAX + 1)


def test_eager_low_word_carry_saturated_telemetry_and_exact_cadence() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=7))
    source_value = (9 << 32) | _UINT32_MAX
    state = _with_history(lifecycle, lifecycle.init(jr.key(2)), source_value)
    event = _event()
    oak = _post_update_oak(lifecycle, state, event, source_value=source_value)
    saturated_per_option = jnp.full(
        (lifecycle.config.n_options,),
        _INT32_MAX,
        dtype=jnp.int32,
    )
    oak = oak.replace(
        execution_counts=saturated_per_option,
        stomp_state=oak.stomp_state.replace(
            option_models=oak.stomp_state.option_models.replace(
                n_completions=saturated_per_option,
            )
        ),
    )

    assert bool(lifecycle.state_valid(state))
    assert bool(lifecycle._oak_values_valid(oak))
    result = lifecycle.observe_and_route(state, oak, _binding(state), event)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.post_update_consumer_clock_valid)
    np.testing.assert_array_equal(result.state.observe_words, [10, 0])
    np.testing.assert_array_equal(result.state.learner_state.step_words, [10, 0])
    assert int(result.state.observe_count) == _INT32_MAX
    assert int(result.state.learner_state.step_count) == _INT32_MAX
    assert int(result.state.learner_state.replacement_phase) == (source_value + 1) % 7
    assert bool(lifecycle.state_valid(result.state))


@pytest.mark.parametrize(
    ("outcome", "allow_curation", "inject_route_failure"),
    (
        ("committed", True, False),
        ("deferred", False, False),
        ("rolled_back", True, True),
    ),
)
def test_curation_outcome_words_commit_defer_and_rollback(
    outcome: str,
    allow_curation: bool,
    inject_route_failure: bool,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=1))
    source_value = (7 << 32) | _UINT32_MAX
    near_carry = (2 << 32) | _UINT32_MAX
    kwargs = {outcome: near_carry}
    state = _with_history(
        lifecycle,
        lifecycle.init(jr.key(3)),
        source_value,
        **kwargs,
    )
    state = _force_promotion(state)
    if inject_route_failure:
        lifecycle._router = _DuplicateDescriptorRouter(lifecycle.router.config)
    event = _event(allow_curation=allow_curation)
    oak = _post_update_oak(lifecycle, state, event, source_value=source_value)

    assert bool(lifecycle.state_valid(state))
    result = lifecycle.observe_and_route(state, oak, _binding(state), event)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(getattr(result.diagnostics, f"curation_{outcome}"))
    words = getattr(result.state, f"{outcome}_curation_words")
    np.testing.assert_array_equal(words, [3, 0])
    assert int(getattr(result.state, f"{outcome}_curation_count")) == _INT32_MAX
    if outcome == "committed":
        np.testing.assert_array_equal(result.state.router_state.route_words, [3, 0])
        np.testing.assert_array_equal(
            result.consumer_binding.semantic_generation_words,
            [3, 0],
        )
    assert bool(lifecycle.state_valid(result.state))


def test_impossible_exact_histories_and_consumer_clocks_rollback_bit_exactly() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    source_value = (3 << 32) | 17
    state = _with_history(lifecycle, lifecycle.init(jr.key(4)), source_value)
    event = _event()
    oak = _post_update_oak(lifecycle, state, event, source_value=source_value)
    binding = _binding(state)

    corrupt_state = state.replace(observe_words=_words(source_value + 1))
    rejected_state = lifecycle.observe_and_route(
        corrupt_state,
        oak,
        binding,
        event,
    )
    assert not bool(rejected_state.diagnostics.state_values_valid)
    assert not bool(rejected_state.diagnostics.transaction_applied)
    _assert_tree_exact(rejected_state.state, corrupt_state)

    stale_oak = oak.replace(
        step_words=state.observe_words,
        stomp_state=oak.stomp_state.replace(step_words=state.observe_words),
    )
    rejected_oak = lifecycle.observe_and_route(state, stale_oak, binding, event)
    assert not bool(rejected_oak.diagnostics.post_update_consumer_clock_valid)
    assert not bool(rejected_oak.diagnostics.transaction_applied)
    _assert_tree_exact(rejected_oak.state, state)
    _assert_tree_exact(rejected_oak.oak_state, stale_oak)

    stale_binding = binding.replace(
        semantic_generation_words=jnp.asarray([1, 0], dtype=jnp.uint32)
    )
    rejected_binding = lifecycle.observe_and_route(state, oak, stale_binding, event)
    assert not bool(rejected_binding.diagnostics.consumer_binding_valid)
    assert not bool(rejected_binding.diagnostics.transaction_applied)
    _assert_tree_exact(rejected_binding.state, state)


def test_gradient_pullback_requires_exact_generation_after_telemetry_saturation() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    generation = (2 << 32) | 9
    observations = generation + 17
    state = _with_history(
        lifecycle,
        lifecycle.init(jr.key(41)),
        observations,
        committed=generation,
    )
    observation = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    gradient = jnp.ones((lifecycle.config.total_feature_dim,), dtype=jnp.float32)

    ambiguous = lifecycle.pullback_pair_gradient(
        state,
        observation,
        gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    assert not bool(ambiguous.valid)
    np.testing.assert_array_equal(ambiguous.semantic_generation_words, [0, 0])

    exact = lifecycle.pullback_pair_gradient(
        state,
        observation,
        gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
        expected_generation_words=state.router_state.generation_words,
    )
    assert bool(exact.valid)
    np.testing.assert_array_equal(
        exact.semantic_generation_words,
        state.router_state.generation_words,
    )


def test_terminal_all_ones_rejects_eager_and_jit_without_key_or_state_change() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=7))
    state = _with_history(lifecycle, lifecycle.init(jr.key(5)), _UINT64_MAX)
    event = _event()
    oak = _post_update_oak(lifecycle, state, event, source_value=_UINT64_MAX)
    binding = _binding(state)

    eager = lifecycle.observe_and_route(state, oak, binding, event)
    compiled = jax.jit(lifecycle.observe_and_route)(state, oak, binding, event)

    assert not bool(eager.diagnostics.update_capacity_available)
    assert not bool(eager.diagnostics.transaction_applied)
    _assert_tree_exact(eager.state, state)
    _assert_tree_exact(eager.oak_state, oak)
    _assert_tree_exact(compiled, eager)


def test_explicit_budget_above_low_word_wrap_uses_exact_words() -> None:
    budget = (4 << 32) | 13
    lifecycle = PrototypeFeatureLifecycle(_config(max_observations=budget))
    state = _with_history(lifecycle, lifecycle.init(jr.key(51)), budget)
    event = _event()
    oak = _post_update_oak(lifecycle, state, event, source_value=budget)

    assert bool(lifecycle.state_valid(state))
    result = lifecycle.observe_and_route(state, oak, _binding(state), event)

    assert not bool(result.diagnostics.update_capacity_available)
    assert bool(result.diagnostics.post_update_consumer_clock_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)


def test_managed_horde_uses_exact_post_update_clock_and_rejects_a_fork() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(managed_horde_demons=2))
    source_value = (5 << 32) | 11
    state = _with_history(lifecycle, lifecycle.init(jr.key(6)), source_value)
    event = _event(n_tasks=3)
    oak = _post_update_oak(lifecycle, state, event, source_value=source_value)
    horde = _post_update_horde(lifecycle, source_value)

    result = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        horde,
        _binding(state),
        event,
    )
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.horde_diagnostics.pre_step_parity_valid)
    assert bool(result.horde_diagnostics.post_step_parity_valid)
    np.testing.assert_array_equal(result.state.observe_words, result.horde_state.step_words)

    forked_horde = horde.replace(step_words=_words(source_value))
    rejected = lifecycle.observe_and_route_with_horde(
        state,
        oak,
        forked_horde,
        _binding(state),
        event,
    )
    assert bool(rejected.horde_diagnostics.horde_state_values_valid)
    assert not bool(rejected.horde_diagnostics.pre_step_parity_valid)
    assert not bool(rejected.diagnostics.transaction_applied)
    _assert_tree_exact(rejected.state, state)
    _assert_tree_exact(rejected.horde_state, forked_horde)


def test_strict_legacy_migration_and_v2_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = _with_history(lifecycle, lifecycle.init(jr.key(7)), 19, deferred=2)
    binding = _binding(state)
    exact_word_names = {
        "observe_words",
        "deferred_curation_words",
        "committed_curation_words",
        "rolled_back_curation_words",
    }
    legacy_state = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name not in exact_word_names
    }
    migrated = migrate_legacy_prototype_feature_lifecycle_state(
        lifecycle,
        legacy_state,
    )
    _assert_tree_exact(migrated, state)
    legacy_binding = {
        "semantic_generation": binding.semantic_generation,
        "descriptors": binding.descriptors,
    }
    migrated_binding = migrate_legacy_prototype_feature_consumer_binding(
        lifecycle,
        state,
        legacy_binding,
    )
    _assert_tree_exact(migrated_binding, binding)

    with pytest.raises(ValueError, match="saturated.*ambiguous"):
        migrate_legacy_prototype_feature_lifecycle_state(
            lifecycle,
            {**legacy_state, "observe_count": jnp.asarray(_INT32_MAX, jnp.int32)},
        )
    with pytest.raises(ValueError, match="saturated.*ambiguous"):
        migrate_legacy_prototype_feature_consumer_binding(
            lifecycle,
            state,
            {
                **legacy_binding,
                "semantic_generation": jnp.asarray(_INT32_MAX, jnp.int32),
            },
        )

    path = tmp_path / "exact-lifecycle"
    save_prototype_feature_lifecycle_checkpoint(lifecycle, state, path)
    restored_lifecycle, restored = load_prototype_feature_lifecycle_checkpoint(path)
    assert restored_lifecycle.to_config() == lifecycle.to_config()
    _assert_tree_exact(restored, state)

    legacy_path = tmp_path / "legacy-lifecycle"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.prototype-feature-lifecycle.checkpoint.v1",
            "mechanism_status": "development_mechanism_only",
            "scientific_promotion_allowed": False,
            "config": lifecycle.to_config(),
            "resource_budget": lifecycle.resource_budget(state).to_config(),
        },
    )
    with pytest.raises(ValueError, match="lacks exact outer clocks"):
        load_prototype_feature_lifecycle_checkpoint(legacy_path)


def test_jit_and_scan_cross_low_word_carry_atomically() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    source_value = _UINT32_MAX - 1
    state = _with_history(lifecycle, lifecycle.init(jr.key(8)), source_value)
    event = _event()
    oak = _post_update_oak(lifecycle, state, event, source_value=source_value)
    eager = lifecycle.observe_and_route(state, oak, _binding(state), event)
    compiled = jax.jit(lifecycle.observe_and_route)(state, oak, _binding(state), event)
    _assert_tree_exact(compiled, eager)

    initial_oak = _oak_agent(lifecycle.config).init(jr.key(81))
    initial_oak = initial_oak.replace(
        stomp_state=initial_oak.stomp_state.replace(
            base_last_action=jnp.asarray(0, dtype=jnp.int32),
            last_primitive_action=jnp.asarray(0, dtype=jnp.int32),
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
        )
    )
    events = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf, leaf)), event)

    def body(carry, one_event):
        one_state, one_oak, one_binding = carry
        one = jnp.asarray(1, dtype=jnp.uint32)
        low = one_state.observe_words[1] + one
        carry_word = (low == 0).astype(jnp.uint32)
        post_words = jnp.stack((one_state.observe_words[0] + carry_word, low))
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        post_count = (
            jnp.minimum(jnp.maximum(one_state.observe_count, 0), maximum - 1) + 1
        )
        post_oak = one_oak.replace(
            step_count=post_count,
            step_words=post_words,
            stomp_state=one_oak.stomp_state.replace(
                base_last_obs=lifecycle.augment(
                    one_state,
                    one_event.next_observation,
                ),
                step_count=post_count,
                step_words=post_words,
            ),
        )
        result = lifecycle.observe_and_route(
            one_state,
            post_oak,
            one_binding,
            one_event,
        )
        return (result.state, result.oak_state, result.consumer_binding), (
            result.diagnostics.transaction_applied,
            result.diagnostics.observe_words_before,
            result.diagnostics.observe_words_after,
        )

    scanned, audit = jax.lax.scan(
        body,
        (state, initial_oak, _binding(state)),
        events,
    )
    assert bool(jnp.all(audit[0]))
    np.testing.assert_array_equal(audit[1][0], [0, _UINT32_MAX - 1])
    np.testing.assert_array_equal(audit[2][0], [0, _UINT32_MAX])
    np.testing.assert_array_equal(audit[2][1], [1, 0])
    np.testing.assert_array_equal(scanned[0].observe_words, [1, 1])
    assert bool(lifecycle.state_valid(scanned[0]))
