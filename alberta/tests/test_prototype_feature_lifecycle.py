# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Standalone L0 contracts for bounded prototype feature discovery."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.core.prototype_feature_lifecycle as lifecycle_module
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouter,
    FeatureBankRouteResult,
)
from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    OaKState,
    learned_feature_subtask_specs,
)
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_feature_lifecycle import (
    PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
    PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    load_prototype_feature_lifecycle_checkpoint,
    save_prototype_feature_lifecycle_checkpoint,
)

pytestmark = pytest.mark.unit


def _config(*, replacement_interval: int = 1) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=4,
        active_pair_slots=2,
        candidate_pair_slots=6,
        n_tasks=2,
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
        max_observations=100,
    )


def _primitive_only_config() -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=3,
        active_pair_slots=1,
        candidate_pair_slots=3,
        n_tasks=1,
        n_options=0,
        n_primitive_actions=2,
        option_subtask_feature_indices=(),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=0,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
    )


def _prefix_pair_config(
    *,
    replacement_interval: int = 0,
) -> PrototypeFeatureLifecycleConfig:
    """Return the HCCL-sized base/pair-source separation contract."""

    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=23,
        pair_source_feature_dim=16,
        active_pair_slots=12,
        candidate_pair_slots=120,
        n_tasks=2,
        n_options=2,
        n_primitive_actions=2,
        option_subtask_feature_indices=(0, 22),
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


def _oak_agent(config: PrototypeFeatureLifecycleConfig, *, hidden: bool = False) -> OaKAgent:
    specs = tuple(
        SubtaskSpec(feature_index=index)
        for index in config.option_subtask_feature_indices
    )
    return OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=specs,
                observation_dim=config.total_feature_dim,
                n_primitive_actions=config.n_primitive_actions,
                base_hidden_sizes=((3,) if hidden else ()),
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        )
    )


def test_nonnegative_int32_sum_within_accepts_empty_vector_without_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = jnp.empty((0,), dtype=jnp.int32)
    limit = jnp.asarray(7, dtype=jnp.int32)

    def forbidden_scan(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("empty int32 sums must not enter lax.scan")

    with monkeypatch.context() as local:
        local.setattr(lifecycle_module.jax.lax, "scan", forbidden_scan)
        with jax.disable_jit():
            eager_result = lifecycle_module._nonnegative_int32_sum_within(values, limit)
        jitted_result = jax.jit(
            lifecycle_module._nonnegative_int32_sum_within
        )(values, limit)

    for result in (eager_result, jitted_result):
        assert result.shape == ()
        assert result.dtype == jnp.bool_
        assert bool(result)


def test_nonnegative_int32_sum_within_preserves_nonempty_and_jit_contract() -> None:
    values = jnp.asarray((1, 2, 3), dtype=jnp.int32)
    helper = jax.jit(lifecycle_module._nonnegative_int32_sum_within)

    accepted = helper(values, jnp.asarray(6, dtype=jnp.int32))
    rejected_over_limit = helper(values, jnp.asarray(5, dtype=jnp.int32))
    rejected_negative = helper(
        values.at[1].set(jnp.asarray(-1, dtype=jnp.int32)),
        jnp.asarray(6, dtype=jnp.int32),
    )

    for result in (accepted, rejected_over_limit, rejected_negative):
        assert result.shape == ()
        assert result.dtype == jnp.bool_
    assert bool(accepted)
    assert not bool(rejected_over_limit)
    assert not bool(rejected_negative)


def test_primitive_only_lifecycle_and_oak_have_no_dummy_option_or_curation() -> None:
    config = _primitive_only_config()
    assert PrototypeFeatureLifecycleConfig.from_config(config.to_config()) == config
    lifecycle = PrototypeFeatureLifecycle(config)
    oak_agent = _oak_agent(config)
    lifecycle.require_compatible_oak_config(oak_agent.config)

    lifecycle_state = lifecycle.init(jr.key(81))
    oak_state = oak_agent.init(jr.key(82))
    observation = jnp.asarray((0.2, -0.3, 0.5, 0.1), dtype=jnp.float32)
    oak_state = oak_agent.start(oak_state, observation)
    assert bool(lifecycle.state_valid(lifecycle_state))
    assert int(oak_state.stomp_state.executing_option) == -1
    assert oak_state.execution_counts.shape == (0,)
    assert oak_state.utility_ema.shape == (0,)

    updated = oak_agent.update(
        oak_state,
        jnp.asarray(0.25, dtype=jnp.float32),
        observation.at[0].add(0.1),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(updated.update_applied)
    assert int(updated.planning_backups) == 0
    assert updated.utility_ema.shape == (0,)
    generated = learned_feature_subtask_specs(updated.state, n_subtasks=2)
    assert len(generated) == 2

    curated_agent, curated_state = oak_agent.curate(updated.state, jr.key(83))
    assert curated_agent is oak_agent
    assert curated_state is updated.state

    budget = lifecycle.resource_budget(lifecycle_state)
    assert budget.output_route_feature_groups == 0
    assert budget.managed_oak_feature_width == config.total_feature_dim

    next_base = jnp.asarray((-0.4, 0.6, 0.2), dtype=jnp.float32)
    routed = lifecycle.observe_and_route(
        lifecycle_state,
        _seed_oak(lifecycle, lifecycle_state, next_base),
        _consumer_binding(lifecycle_state),
        PrototypeFeatureLifecycleEvent(
            observation=jnp.asarray((0.5, -0.25, 1.0), dtype=jnp.float32),
            targets=jnp.asarray((0.3,), dtype=jnp.float32),
            next_observation=next_base,
            allow_curation=jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    assert bool(routed.diagnostics.transaction_applied)
    assert routed.oak_state.stomp_state.option_policies.q_weights.shape == (
        0,
        config.n_primitive_actions,
        config.total_feature_dim,
    )


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        # MultiHeadMLPState's two host-only timing leaves are Python floats in
        # eager state and traced float32 scalars under JIT.  They are not
        # scientific state; normalize only those host scalars for comparison.
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
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
            np.testing.assert_allclose(left_array, right_array, rtol=1.0e-6, atol=1.0e-7)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def _force_promotion(lifecycle: PrototypeFeatureLifecycle, state):
    learner_state = state.learner_state
    active = set(
        zip(
            np.asarray(learner_state.feature_left).tolist(),
            np.asarray(learner_state.feature_right).tolist(),
            strict=True,
        )
    )
    candidates = list(
        zip(
            np.asarray(learner_state.candidate_left).tolist(),
            np.asarray(learner_state.candidate_right).tolist(),
            strict=True,
        )
    )
    candidate_index = next(index for index, pair in enumerate(candidates) if pair not in active)
    candidate_utilities = jnp.zeros_like(learner_state.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner_state = learner_state.replace(
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        ages=jnp.full_like(learner_state.ages, 10),
        candidate_utilities=candidate_utilities,
        candidate_ages=jnp.full_like(learner_state.candidate_ages, 10),
        step_count=jnp.asarray(10, dtype=jnp.int32),
        step_words=jnp.asarray((0, 10), dtype=jnp.uint32),
    )
    forced = state.replace(
        learner_state=learner_state,
        observe_count=jnp.asarray(10, dtype=jnp.int32),
        observe_words=jnp.asarray((0, 10), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(forced))
    return forced, candidates[candidate_index]


def _consumer_binding(state) -> PrototypeFeatureConsumerBinding:
    return PrototypeFeatureConsumerBinding(
        semantic_generation=state.router_state.generation_count,
        semantic_generation_words=state.router_state.generation_words,
        descriptors=state.router_state.descriptors,
    )


def _seed_oak(
    lifecycle: PrototypeFeatureLifecycle,
    lifecycle_state,
    next_base_observation: jax.Array,
) -> OaKState:
    config = lifecycle.config
    oak = _oak_agent(config).init(jr.key(71))
    stomp = oak.stomp_state
    width = config.total_feature_dim
    next_step = int(lifecycle_state.observe_count) + 1
    next_step_count = jnp.asarray(
        min(next_step, np.iinfo(np.int32).max),
        dtype=jnp.int32,
    )
    next_step_words = jnp.asarray((0, next_step), dtype=jnp.uint32)

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
        step_count=next_step_count,
        step_words=next_step_words,
    )
    policy_values = jnp.arange(
        config.n_options * config.n_primitive_actions * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, config.n_primitive_actions, width)
    model_values = jnp.arange(
        config.n_options * width * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, width, width)
    policies = stomp.option_policies.replace(
        q_weights=policy_values + 2_000.0,
        traces=policy_values + 3_000.0,
    )
    models = stomp.option_models.replace(next_state_weights=model_values + 4_000.0)
    stomp = stomp.replace(
        base_learner_state=base_learner,
        base_last_obs=lifecycle.augment(lifecycle_state, next_base_observation),
        base_last_action=jnp.asarray(0, dtype=jnp.int32),
        option_policies=policies,
        option_models=models,
        executing_option=jnp.asarray(-1, dtype=jnp.int32),
        option_start_obs=jnp.arange(width, dtype=jnp.float32) + 5_000.0,
        step_count=next_step_count,
        step_words=next_step_words,
    )
    return cast(
        OaKState,
        oak.replace(
            stomp_state=stomp,
            step_count=next_step_count,
            step_words=next_step_words,
        ),
    )


def _advance_oak_clock(oak: OaKState) -> OaKState:
    """Stage the next caller-owned post-transition OaK identity."""

    maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    next_count = jnp.where(oak.step_count < maximum, oak.step_count + 1, maximum)
    next_low = oak.step_words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (next_low < oak.step_words[1]).astype(jnp.uint32)
    next_words = jnp.stack((oak.step_words[0] + carry, next_low))
    base = oak.stomp_state.base_learner_state.replace(
        step_count=next_count,
        step_words=next_words,
    )
    stomp = oak.stomp_state.replace(
        base_learner_state=base,
        step_count=next_count,
        step_words=next_words,
    )
    return cast(
        OaKState,
        oak.replace(
            stomp_state=stomp,
            step_count=next_count,
            step_words=next_words,
        ),
    )


def _event(*, allow_curation: bool = True) -> PrototypeFeatureLifecycleEvent:
    return PrototypeFeatureLifecycleEvent(
        observation=jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
        targets=jnp.asarray([0.75, jnp.nan], dtype=jnp.float32),
        next_observation=jnp.asarray([-1.0, 2.0, 4.0, 0.25], dtype=jnp.float32),
        allow_curation=jnp.asarray(allow_curation, dtype=jnp.bool_),
    )


class _DuplicateDescriptorRouter(FeatureBankRouter):
    """Failure injector: make every proposed bank duplicate slot zero."""

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


class _NoUpdateProxy:
    """Delegate every learner operation except forbidden adoption recompute."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adoption must not evaluate the learner")


class _NoRouteProxy:
    """Delegate router validation/init while forbidding a new route call."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def route(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adoption must not evaluate the router")


class _InvalidPostconditionRouter(FeatureBankRouter):
    """Failure injector: valid route diagnostics with corrupt returned state."""

    def route(
        self,
        state,
        consumers,
        new_descriptors,
        *,
        feature_axes=None,
        carry_survivors=True,
    ) -> FeatureBankRouteResult:
        result = super().route(
            state,
            consumers,
            new_descriptors,
            feature_axes=feature_axes,
            carry_survivors=carry_survivors,
        )
        corrupt_descriptors = result.state.descriptors.at[1].set(
            result.state.descriptors[0]
        )
        corrupt_state = dataclasses.replace(
            result.state,
            descriptors=corrupt_descriptors,
        )
        return dataclasses.replace(result, state=corrupt_state)


def test_config_is_strict_versioned_mechanism_only_and_round_trips() -> None:
    class ConfigSubclass(PrototypeFeatureLifecycleConfig):
        pass

    class OaKConfigSubclass(OaKConfig):
        pass

    class STOMPConfigSubclass(STOMPConfig):
        pass

    class TupleSubclass(tuple[int, ...]):
        pass

    class ListSubclass(list[int]):
        pass

    config = _config()
    lifecycle = PrototypeFeatureLifecycle(config)
    assert PrototypeFeatureLifecycleConfig.from_config(config.to_config()) == config
    lifecycle.require_compatible_oak_config(_oak_agent(config).config)
    assert config.to_config()["mechanism_status"] == (
        PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
    )
    assert config.to_config()["scientific_promotion_allowed"] is False
    assert PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED is False

    config_subclass = ConfigSubclass(**dataclasses.asdict(config))
    with pytest.raises(TypeError, match="PrototypeFeatureLifecycleConfig"):
        PrototypeFeatureLifecycle(config_subclass)
    assert type(config_subclass.from_config(config.to_config())) is (
        PrototypeFeatureLifecycleConfig
    )
    oak_subclass = OaKConfigSubclass(stomp=_oak_agent(config).config.stomp)
    with pytest.raises(TypeError, match="OaKConfig"):
        lifecycle.require_compatible_oak_config(oak_subclass)
    exact_stomp = _oak_agent(config).config.stomp
    stomp_subclass = STOMPConfigSubclass(
        **{
            field.name: getattr(exact_stomp, field.name)
            for field in dataclasses.fields(exact_stomp)
        }
    )
    with pytest.raises(TypeError, match="exact STOMPConfig"):
        lifecycle.require_compatible_oak_config(OaKConfig(stomp=stomp_subclass))

    payload = config.to_config()
    payload["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="cannot claim promotion"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    payload = config.to_config()
    payload["active_pair_slots"] = True
    with pytest.raises(ValueError, match="active_pair_slots"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    with pytest.raises(ValueError, match="pair space"):
        dataclasses.replace(config, active_pair_slots=7)
    with pytest.raises(ValueError, match="requires candidate_pair_slots"):
        dataclasses.replace(config, candidate_pair_slots=0)
    with pytest.raises(ValueError, match="strict float"):
        dataclasses.replace(config, utility_decay=1)
    with pytest.raises(ValueError, match="stable base"):
        dataclasses.replace(config, option_subtask_feature_indices=(0, 4))
    with pytest.raises(ValueError, match="exact tuple"):
        dataclasses.replace(
            config,
            option_subtask_feature_indices=TupleSubclass((0, 1)),
        )
    payload = config.to_config()
    payload["option_subtask_feature_indices"] = ListSubclass([0, 1])
    with pytest.raises(ValueError, match="JSON integer list"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, step_size_output=1.0e300)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, scale_normalizer_epsilon=1.0e-300)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, utility_decay=0.999999999)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, scale_normalizer_decay=0.999999999)
    mismatched_oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0), SubtaskSpec(feature_index=4)),
            observation_dim=config.total_feature_dim,
            n_primitive_actions=config.n_primitive_actions,
        )
    )
    with pytest.raises(ValueError, match="subtask attestation"):
        lifecycle.require_compatible_oak_config(mismatched_oak)
    for malformed_index in (True, 1.0):
        malformed_specs = (
            SubtaskSpec(feature_index=0),
            SubtaskSpec(feature_index=cast(int, malformed_index)),
        )
        malformed_oak = OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=malformed_specs,
                observation_dim=config.total_feature_dim,
                n_primitive_actions=config.n_primitive_actions,
            )
        )
        with pytest.raises(ValueError, match="subtask attestation"):
            lifecycle.require_compatible_oak_config(malformed_oak)
    for malformed_dimension in (float(config.total_feature_dim),):
        malformed_oak = OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(feature_index=0),
                    SubtaskSpec(feature_index=1),
                ),
                observation_dim=cast(int, malformed_dimension),
                n_primitive_actions=config.n_primitive_actions,
            )
        )
        with pytest.raises(ValueError, match="subtask attestation"):
            lifecycle.require_compatible_oak_config(malformed_oak)
    malformed_actions = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=config.total_feature_dim,
            n_primitive_actions=cast(int, float(config.n_primitive_actions)),
        )
    )
    with pytest.raises(ValueError, match="subtask attestation"):
        lifecycle.require_compatible_oak_config(malformed_actions)
    single_action_config = dataclasses.replace(config, n_primitive_actions=1)
    single_action_lifecycle = PrototypeFeatureLifecycle(single_action_config)
    boolean_actions = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=single_action_config.total_feature_dim,
            n_primitive_actions=cast(int, True),
        )
    )
    with pytest.raises(ValueError, match="subtask attestation"):
        single_action_lifecycle.require_compatible_oak_config(boolean_actions)


def test_pair_source_prefix_config_is_opt_in_strict_and_legacy_byte_stable() -> None:
    legacy = _config()
    legacy_payload = legacy.to_config()
    assert legacy.pair_source_feature_dim is None
    assert legacy.effective_pair_source_feature_dim == legacy.base_feature_dim
    assert "pair_source_feature_dim" not in legacy_payload
    assert PrototypeFeatureLifecycleConfig.from_config(legacy_payload).to_config() == (
        legacy_payload
    )

    scoped = _prefix_pair_config()
    scoped_payload = scoped.to_config()
    assert scoped.effective_pair_source_feature_dim == 16
    assert scoped.total_feature_dim == 35
    assert scoped_payload["schema"] == (
        "alberta.prototype-feature-lifecycle.config.v5"
    )
    assert scoped_payload["pair_source_feature_dim"] == 16
    assert PrototypeFeatureLifecycleConfig.from_config(scoped_payload) == scoped

    scoped_horde = dataclasses.replace(
        scoped,
        n_tasks=2,
        managed_horde_demons=1,
    )
    scoped_horde_payload = scoped_horde.to_config()
    assert scoped_horde_payload["schema"] == (
        "alberta.prototype-feature-lifecycle.config.v6"
    )
    assert PrototypeFeatureLifecycleConfig.from_config(scoped_horde_payload) == (
        scoped_horde
    )

    for invalid_source in (True, 1.0, 1, 23, 24):
        with pytest.raises(ValueError, match="pair_source_feature_dim"):
            dataclasses.replace(
                scoped,
                pair_source_feature_dim=cast(int | None, invalid_source),
            )
    with pytest.raises(ValueError, match="pair space"):
        dataclasses.replace(scoped, active_pair_slots=121)
    with pytest.raises(ValueError, match="pair space"):
        dataclasses.replace(scoped, candidate_pair_slots=121)

    missing_source = dict(scoped_payload)
    missing_source.pop("pair_source_feature_dim")
    with pytest.raises(ValueError, match="config fields"):
        PrototypeFeatureLifecycleConfig.from_config(missing_source)
    null_source = dict(scoped_payload)
    null_source["pair_source_feature_dim"] = None
    with pytest.raises(ValueError, match="pair_source_feature_dim"):
        PrototypeFeatureLifecycleConfig.from_config(null_source)
    wrong_schema = dict(legacy_payload)
    wrong_schema["pair_source_feature_dim"] = 16
    with pytest.raises(ValueError, match="config fields"):
        PrototypeFeatureLifecycleConfig.from_config(wrong_schema)


def test_allocation_and_python_collection_ceilings_fail_before_construction() -> None:
    config = _config()
    with pytest.raises(ValueError, match="active descriptor comparison"):
        dataclasses.replace(
            config,
            base_feature_dim=100,
            active_pair_slots=2_049,
            candidate_pair_slots=0,
            replacement_interval=0,
        )
    with pytest.raises(ValueError, match="all-pairs candidate enumeration"):
        dataclasses.replace(
            config,
            base_feature_dim=400,
            active_pair_slots=1,
            candidate_pair_slots=1,
        )
    with pytest.raises(ValueError, match="base-head collection"):
        dataclasses.replace(
            config,
            n_options=3_000,
            n_primitive_actions=2_000,
            option_subtask_feature_indices=(0,) * 3_000,
        )


def test_init_augmentation_state_and_exact_resource_contracts() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(3))

    np.testing.assert_array_equal(
        state.router_state.descriptors,
        np.asarray([[0, 1], [0, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        state.router_state.descriptors[:, 0],
        state.learner_state.feature_left,
    )
    np.testing.assert_array_equal(
        state.router_state.descriptors[:, 1],
        state.learner_state.feature_right,
    )
    assert bool(lifecycle.state_valid(state))
    binding = _consumer_binding(state)
    assert bool(lifecycle.consumer_binding_valid(state, binding))

    observation = jnp.asarray([2.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        lifecycle.augment(state, observation),
        np.asarray([2.0, 3.0, 5.0, 7.0, 6.0, 10.0], dtype=np.float32),
    )
    budget = lifecycle.resource_budget(state)
    expected_lifecycle_state_bytes = sum(
        int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(state)
    )
    expected_learner_template_bytes = sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(state.learner_state)
    )
    expected_oak_template_bytes = sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(_oak_agent(lifecycle.config).init(jr.key(0)))
    )
    assert budget.lifecycle_state_nbytes == expected_lifecycle_state_bytes
    assert budget.consumer_binding_persistent_nbytes == sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(binding)
    )
    assert (
        budget.internal_learner_template_nbytes
        == expected_learner_template_bytes
    )
    assert budget.internal_oak_template_nbytes == expected_oak_template_bytes
    assert budget.internal_template_nbytes == (
        expected_learner_template_bytes + expected_oak_template_bytes
    )
    assert budget.owned_persistent_state_nbytes == (
        expected_lifecycle_state_bytes
        + expected_learner_template_bytes
        + expected_oak_template_bytes
    )
    assert budget.router_calls_per_committed_curation == 2
    assert budget.router_calls_per_observe == 2
    assert budget.max_active_pair_products_per_observe == 10
    assert budget.max_candidate_pair_products_per_observe == 6
    assert budget.managed_oak_feature_width == lifecycle.config.total_feature_dim
    assert budget.rebuilt_base_cache_nbytes == 4 * lifecycle.config.total_feature_dim
    assert budget.scientific_promotion_allowed is False


def test_pair_source_prefix_has_complete_raw_pair_universe_and_full_base_output() -> None:
    lifecycle = PrototypeFeatureLifecycle(_prefix_pair_config())
    state = lifecycle.init(jr.key(0))
    source_dim = lifecycle.config.effective_pair_source_feature_dim

    active = np.stack(
        (
            np.asarray(state.learner_state.feature_left),
            np.asarray(state.learner_state.feature_right),
        ),
        axis=1,
    )
    candidates = np.stack(
        (
            np.asarray(state.learner_state.candidate_left),
            np.asarray(state.learner_state.candidate_right),
        ),
        axis=1,
    )
    expected_candidates = {
        (left, right)
        for left in range(source_dim)
        for right in range(left + 1, source_dim)
    }
    assert active.shape == (12, 2)
    assert candidates.shape == (120, 2)
    assert int(np.max(active)) < source_dim
    assert {tuple(pair) for pair in candidates.tolist()} == expected_candidates
    assert bool(lifecycle.state_valid(state))

    observation = jnp.arange(1, 24, dtype=jnp.float32)
    augmented = lifecycle.augment(state, observation)
    np.testing.assert_array_equal(augmented[:23], observation)
    np.testing.assert_array_equal(
        augmented[23:],
        observation[active[:, 0]] * observation[active[:, 1]],
    )

    non_source_changed = observation.at[16:].add(1_000.0)
    changed_augmented = lifecycle.augment(state, non_source_changed)
    np.testing.assert_array_equal(changed_augmented[:23], non_source_changed)
    np.testing.assert_array_equal(changed_augmented[23:], augmented[23:])

    budget = lifecycle.resource_budget(state)
    assert budget.base_feature_slots == 23
    assert budget.pair_source_feature_slots == 16
    assert budget.canonical_pair_universe_slots == 120
    assert budget.managed_oak_feature_width == 35
    assert budget.max_active_pair_products_per_observe == 60
    assert budget.max_candidate_pair_products_per_observe == 120

    invalid_active_descriptors = state.router_state.descriptors.at[0].set(
        jnp.asarray((0, 16), dtype=jnp.int32)
    )
    invalid_active = state.replace(
        learner_state=state.learner_state.replace(
            feature_left=state.learner_state.feature_left.at[0].set(0),
            feature_right=state.learner_state.feature_right.at[0].set(16),
        ),
        router_state=dataclasses.replace(
            state.router_state,
            descriptors=invalid_active_descriptors,
        ),
    )
    assert not bool(lifecycle.state_valid(invalid_active))

    invalid_candidate = state.replace(
        learner_state=state.learner_state.replace(
            candidate_left=state.learner_state.candidate_left.at[0].set(0),
            candidate_right=state.learner_state.candidate_right.at[0].set(16),
        )
    )
    assert not bool(lifecycle.state_valid(invalid_candidate))


def test_pair_source_prefix_pullback_and_learning_ignore_non_source_coordinates() -> None:
    lifecycle = PrototypeFeatureLifecycle(_prefix_pair_config())
    state = lifecycle.init(jr.key(5))
    observation = jnp.linspace(-2.0, 3.0, 23, dtype=jnp.float32)
    augmented_gradient = jnp.linspace(
        -0.7,
        0.9,
        lifecycle.config.total_feature_dim,
        dtype=jnp.float32,
    )

    pullback = lifecycle.pullback_pair_gradient(
        state,
        observation,
        augmented_gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    expected = jax.grad(
        lambda value: jnp.vdot(
            lifecycle.augment(state, value),
            augmented_gradient,
        )
    )(observation)
    assert bool(pullback.valid)
    np.testing.assert_array_equal(pullback.gradient, expected)
    np.testing.assert_array_equal(
        pullback.gradient[16:],
        augmented_gradient[16:23],
    )

    next_observation = jnp.linspace(0.1, 2.3, 23, dtype=jnp.float32)
    oak = _seed_oak(lifecycle, state, next_observation)
    binding = _consumer_binding(state)
    base_event = PrototypeFeatureLifecycleEvent(
        observation=observation,
        targets=jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        next_observation=next_observation,
        allow_curation=jnp.asarray(False, dtype=jnp.bool_),
    )
    changed_event = cast(
        PrototypeFeatureLifecycleEvent,
        base_event.replace(
            observation=observation.at[16:].add(10_000.0),
        ),
    )
    base_result = lifecycle.observe_and_route(state, oak, binding, base_event)
    changed_result = lifecycle.observe_and_route(
        state,
        oak,
        binding,
        changed_event,
    )
    assert bool(base_result.diagnostics.transaction_applied)
    assert bool(changed_result.diagnostics.transaction_applied)
    _assert_tree_exact(
        base_result.state.learner_state,
        changed_result.state.learner_state,
    )


def test_unavailable_diagnostics_are_finite_neutral_and_jax_shape_compatible() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(4))
    generation = state.router_state.generation_count

    eager = lifecycle.unavailable_diagnostics(generation)
    compiled = jax.jit(lifecycle.unavailable_diagnostics)(generation)

    _assert_tree_exact(compiled, eager)
    assert not bool(eager.available)
    assert not bool(eager.consumer_binding_valid)
    assert not bool(eager.transaction_applied)
    assert not bool(eager.routing_attempted)
    assert not bool(eager.postcondition_checked)
    assert not bool(eager.postcondition_valid)
    assert not bool(eager.postcondition_rolled_back)
    assert int(eager.semantic_generation_before) == int(generation)
    assert int(eager.semantic_generation_after) == int(generation)
    for leaf in jax.tree.leaves(eager):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.inexact):
            assert np.all(np.isfinite(array))


def test_pair_gradient_pullback_matches_jax_autodiff_oracle() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(5))
    observation = jnp.asarray([0.5, -2.0, 3.0, 4.0], dtype=jnp.float32)
    augmented_gradient = jnp.asarray(
        [0.2, -0.3, 0.5, 1.0, 1.5, -2.0],
        dtype=jnp.float32,
    )

    result = lifecycle.pullback_pair_gradient(
        state,
        observation,
        augmented_gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    expected = jax.grad(
        lambda value: jnp.vdot(
            lifecycle.augment(state, value),
            augmented_gradient,
        )
    )(observation)

    assert bool(result.valid)
    assert int(result.semantic_generation) == 0
    np.testing.assert_allclose(result.gradient, expected, rtol=0.0, atol=0.0)

    invalid = lifecycle.pullback_pair_gradient(
        state,
        observation.at[0].set(jnp.inf),
        augmented_gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    assert not bool(invalid.valid)
    np.testing.assert_array_equal(invalid.gradient, jnp.zeros_like(observation))

    stale = lifecycle.pullback_pair_gradient(
        state,
        observation,
        augmented_gradient,
        jnp.asarray(1, dtype=jnp.int32),
        state.router_state.descriptors,
    )
    assert not bool(stale.valid)
    np.testing.assert_array_equal(stale.gradient, jnp.zeros_like(observation))


def test_descriptor_semantics_cannot_advance_ahead_of_generation() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(6))
    mutated_descriptors = jnp.asarray(
        [[1, 2], [2, 3]],
        dtype=jnp.int32,
    )
    mutated = state.replace(
        learner_state=state.learner_state.replace(
            feature_left=mutated_descriptors[:, 0],
            feature_right=mutated_descriptors[:, 1],
        ),
        router_state=dataclasses.replace(
            state.router_state,
            descriptors=mutated_descriptors,
        ),
    )
    assert not bool(lifecycle.state_valid(mutated))
    pullback = lifecycle.pullback_pair_gradient(
        mutated,
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.ones((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        mutated.router_state.descriptors,
    )
    assert not bool(pullback.valid)
    np.testing.assert_array_equal(
        pullback.gradient,
        np.zeros((lifecycle.config.base_feature_dim,), dtype=np.float32),
    )

    one_generation = mutated.replace(
        learner_state=mutated.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32),
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        ),
        router_state=dataclasses.replace(
            mutated.router_state,
            route_count=jnp.asarray(1, dtype=jnp.int32),
            route_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            generation_count=jnp.asarray(1, dtype=jnp.int32),
            generation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
        observe_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        committed_curation_count=jnp.asarray(1, dtype=jnp.int32),
        committed_curation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert not bool(lifecycle.state_valid(one_generation))


def test_same_generation_descriptor_forks_cannot_cross_bind_or_pull_back() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    canonical = lifecycle.init(jr.key(61))

    def branch(descriptors: jax.Array):
        return canonical.replace(
            learner_state=canonical.learner_state.replace(
                feature_left=descriptors[:, 0],
                feature_right=descriptors[:, 1],
                step_count=jnp.asarray(1, dtype=jnp.int32),
                step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            ),
            router_state=dataclasses.replace(
                canonical.router_state,
                descriptors=descriptors,
                route_count=jnp.asarray(1, dtype=jnp.int32),
                route_words=jnp.asarray((0, 1), dtype=jnp.uint32),
                generation_count=jnp.asarray(1, dtype=jnp.int32),
                generation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            ),
            observe_count=jnp.asarray(1, dtype=jnp.int32),
            observe_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            committed_curation_count=jnp.asarray(1, dtype=jnp.int32),
            committed_curation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )

    branch_a = branch(jnp.asarray([[0, 1], [0, 3]], dtype=jnp.int32))
    branch_b = branch(jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32))
    assert bool(lifecycle.state_valid(branch_a))
    assert bool(lifecycle.state_valid(branch_b))
    binding_b = _consumer_binding(branch_b)
    assert not bool(lifecycle.consumer_binding_valid(branch_a, binding_b))

    pullback = lifecycle.pullback_pair_gradient(
        branch_a,
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.ones((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
        branch_b.router_state.generation_count,
        branch_b.router_state.descriptors,
    )
    assert not bool(pullback.valid)
    np.testing.assert_array_equal(
        pullback.gradient,
        np.zeros((lifecycle.config.base_feature_dim,), dtype=np.float32),
    )


def test_augmentation_rejects_finite_coordinates_whose_products_overflow() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(7))
    observation = jnp.full((4,), 1.0e20, dtype=jnp.float32)

    augmented = lifecycle.augment(state, observation)

    np.testing.assert_array_equal(
        augmented,
        jnp.zeros((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
    )


def test_safe_curation_routes_every_linear_oak_consumer_on_both_model_axes() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, promoted_pair = _force_promotion(lifecycle, lifecycle.init(jr.key(11)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    old_oak = oak
    old_descriptors = np.asarray(state.router_state.descriptors)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    diagnostics = result.diagnostics
    assert bool(diagnostics.transaction_applied)
    assert bool(diagnostics.curation_proposed)
    assert bool(diagnostics.safe_curation_boundary)
    assert bool(diagnostics.routing_attempted)
    assert bool(diagnostics.input_route_valid)
    assert bool(diagnostics.output_route_valid)
    assert bool(diagnostics.route_states_match)
    assert bool(diagnostics.curation_committed)
    assert not bool(diagnostics.curation_deferred)
    assert not bool(diagnostics.curation_rolled_back)
    assert bool(result.input_route_diagnostics.valid)
    assert bool(result.output_route_diagnostics.valid)

    new_descriptors = np.asarray(result.state.router_state.descriptors)
    assert tuple(new_descriptors[0]) == promoted_pair
    np.testing.assert_array_equal(new_descriptors[1], old_descriptors[1])
    assert int(result.state.router_state.route_count) == 1
    assert int(result.state.router_state.generation_count) == 1
    assert int(result.consumer_binding.semantic_generation) == 1
    np.testing.assert_array_equal(
        result.consumer_binding.descriptors,
        result.state.router_state.descriptors,
    )
    assert bool(
        lifecycle.consumer_binding_valid(
            result.state,
            result.consumer_binding,
        )
    )
    assert int(result.state.committed_curation_count) == 1
    assert int(result.state.observe_count) == 11
    assert bool(lifecycle.state_valid(result.state))

    width = lifecycle.config.total_feature_dim
    base = lifecycle.config.base_feature_dim
    old_stomp = old_oak.stomp_state
    new_stomp = result.oak_state.stomp_state
    for old_weight, new_weight in zip(
        old_stomp.base_learner_state.head_params.weights,
        new_stomp.base_learner_state.head_params.weights,
        strict=True,
    ):
        np.testing.assert_array_equal(new_weight[..., :base], old_weight[..., :base])
        np.testing.assert_array_equal(new_weight[..., base], 0.0)
        np.testing.assert_array_equal(new_weight[..., base + 1], old_weight[..., base + 1])
    for old_trace, new_trace in zip(
        old_stomp.base_learner_state.head_traces,
        new_stomp.base_learner_state.head_traces,
        strict=True,
    ):
        np.testing.assert_array_equal(new_trace[0][..., base], 0.0)
        np.testing.assert_array_equal(new_trace[0][..., base + 1], old_trace[0][..., base + 1])
        np.testing.assert_array_equal(new_trace[1], old_trace[1])
    np.testing.assert_array_equal(
        new_stomp.option_policies.q_weights[..., base],
        jnp.zeros_like(new_stomp.option_policies.q_weights[..., base]),
    )
    np.testing.assert_array_equal(
        new_stomp.option_policies.q_weights[..., base + 1],
        old_stomp.option_policies.q_weights[..., base + 1],
    )
    np.testing.assert_array_equal(
        new_stomp.option_policies.traces[..., base],
        jnp.zeros_like(new_stomp.option_policies.traces[..., base]),
    )

    old_models = old_stomp.option_models.next_state_weights
    new_models = new_stomp.option_models.next_state_weights
    np.testing.assert_array_equal(new_models[:, base, :], 0.0)
    np.testing.assert_array_equal(new_models[:, :, base], 0.0)
    np.testing.assert_array_equal(
        new_models[:, base + 1, base + 1],
        old_models[:, base + 1, base + 1],
    )
    np.testing.assert_array_equal(
        new_models[:, :base, :base],
        old_models[:, :base, :base],
    )
    assert new_models.shape == (lifecycle.config.n_options, width, width)
    np.testing.assert_array_equal(
        new_stomp.base_last_obs,
        lifecycle.augment(result.state, event.next_observation),
    )
    np.testing.assert_array_equal(result.next_augmented_observation, new_stomp.base_last_obs)
    np.testing.assert_array_equal(new_stomp.option_start_obs[base], 0.0)
    np.testing.assert_array_equal(
        new_stomp.option_start_obs[base + 1],
        old_stomp.option_start_obs[base + 1],
    )
    np.testing.assert_array_equal(
        jr.key_data(new_stomp.rng_key),
        jr.key_data(old_stomp.rng_key),
    )
    np.testing.assert_array_equal(new_stomp.base_last_action, old_stomp.base_last_action)


def test_no_carry_ablation_zeros_every_dynamic_consumer_axis() -> None:
    config = dataclasses.replace(_config(), carry_survivors=False)
    lifecycle = PrototypeFeatureLifecycle(config)
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(12)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert bool(result.diagnostics.curation_committed)
    base = config.base_feature_dim
    stomp = result.oak_state.stomp_state
    for weight in stomp.base_learner_state.head_params.weights:
        np.testing.assert_array_equal(weight[..., base:], 0.0)
    for trace in stomp.base_learner_state.head_traces:
        np.testing.assert_array_equal(trace[0][..., base:], 0.0)
    np.testing.assert_array_equal(stomp.option_policies.q_weights[..., base:], 0.0)
    np.testing.assert_array_equal(stomp.option_policies.traces[..., base:], 0.0)
    np.testing.assert_array_equal(
        stomp.option_models.next_state_weights[:, base:, :],
        0.0,
    )
    np.testing.assert_array_equal(
        stomp.option_models.next_state_weights[:, :, base:],
        0.0,
    )


def test_zero_cache_collision_rejects_stale_oak_and_binding_exactly() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(62)))
    binding = _consumer_binding(state)
    event = _event().replace(
        next_observation=jnp.zeros((lifecycle.config.base_feature_dim,), dtype=jnp.float32)
    )
    stale_oak = _seed_oak(lifecycle, state, event.next_observation)

    committed = lifecycle.observe_and_route(
        state,
        stale_oak,
        binding,
        event,
    )
    assert bool(committed.diagnostics.curation_committed)
    assert int(committed.consumer_binding.semantic_generation) == 1
    assert bool(
        lifecycle.consumer_binding_valid(
            committed.state,
            committed.consumer_binding,
        )
    )

    rejected = lifecycle.observe_and_route(
        committed.state,
        stale_oak,
        binding,
        event,
    )
    assert bool(rejected.diagnostics.next_observation_matches_oak_cache)
    assert not bool(rejected.diagnostics.consumer_binding_valid)
    assert not bool(rejected.diagnostics.transaction_applied)
    _assert_tree_exact(rejected.state, committed.state)
    _assert_tree_exact(rejected.oak_state, stale_oak)
    _assert_tree_exact(rejected.consumer_binding, binding)

    fresh = lifecycle.observe_and_route(
        committed.state,
        _advance_oak_clock(committed.oak_state),
        committed.consumer_binding,
        event,
    )
    assert bool(fresh.diagnostics.consumer_binding_valid)
    assert bool(fresh.diagnostics.transaction_applied)


def test_duplicate_descriptor_route_failure_preserves_pre_curation_learning() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(41)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    raw_update = lifecycle.learner.update(
        state.learner_state,
        event.observation,
        event.targets,
    )
    lifecycle._router = _DuplicateDescriptorRouter(lifecycle.router.config)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.routing_attempted)
    assert not bool(result.diagnostics.input_route_valid)
    assert not bool(result.diagnostics.output_route_valid)
    assert bool(result.diagnostics.curation_rolled_back)
    assert bool(result.diagnostics.postcondition_valid)
    assert not bool(result.diagnostics.postcondition_rolled_back)
    _assert_tree_exact(result.state.learner_state, raw_update.pre_curation_state)
    _assert_tree_exact(result.state.router_state, state.router_state)
    _assert_tree_exact(result.oak_state, oak)
    assert int(result.state.rolled_back_curation_count) == 1
    assert bool(lifecycle.state_valid(result.state))


def test_invalid_route_postcondition_rolls_back_the_whole_observe_transaction() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(43)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    lifecycle._router = _InvalidPostconditionRouter(lifecycle.router.config)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert bool(result.diagnostics.routing_attempted)
    assert bool(result.diagnostics.input_route_valid)
    assert bool(result.diagnostics.output_route_valid)
    assert not bool(result.diagnostics.postcondition_valid)
    assert bool(result.diagnostics.postcondition_rolled_back)
    assert not bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.curation_committed)
    assert bool(result.diagnostics.curation_rolled_back)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)


@pytest.mark.parametrize("blocked_by", ["caller", "active_option"])
def test_unsafe_curation_is_deferred_after_ordinary_learning(blocked_by: str) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(13)))
    event = _event(allow_curation=blocked_by != "caller")
    oak = _seed_oak(lifecycle, state, event.next_observation)
    if blocked_by == "active_option":
        oak = oak.replace(
            stomp_state=oak.stomp_state.replace(
                executing_option=jnp.asarray(0, dtype=jnp.int32),
                base_last_action=jnp.asarray(
                    lifecycle.config.n_primitive_actions,
                    dtype=jnp.int32,
                ),
            )
        )

    raw_update = lifecycle.learner.update(
        state.learner_state,
        event.observation,
        event.targets,
    )
    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.curation_proposed)
    assert not bool(result.diagnostics.safe_curation_boundary)
    assert bool(result.diagnostics.curation_deferred)
    assert not bool(result.diagnostics.routing_attempted)
    assert not bool(result.diagnostics.curation_committed)
    _assert_tree_exact(result.state.learner_state, raw_update.pre_curation_state)
    _assert_tree_exact(result.state.router_state, state.router_state)
    _assert_tree_exact(result.oak_state, oak)
    assert int(result.state.observe_count) == 11
    assert int(result.state.deferred_curation_count) == 1
    assert int(result.state.committed_curation_count) == 0
    assert bool(lifecycle.state_valid(result.state))


def test_invalid_dynamic_input_is_an_exact_atomic_noop() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(17))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    invalid = event.replace(observation=event.observation.at[1].set(jnp.inf))

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), invalid)

    assert not bool(result.diagnostics.event_values_valid)
    assert not bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.curation_committed)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)
    np.testing.assert_array_equal(
        result.next_augmented_observation,
        lifecycle.augment(state, event.next_observation),
    )


@pytest.mark.parametrize(
    "counter_name",
    (
        "ages",
        "candidate_ages",
        "evidence_idle_steps",
        "utility_evidence_streak",
        "candidate_promotion_evidence_streak",
    ),
)
def test_owned_learner_counters_cannot_exceed_observed_steps(
    counter_name: str,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(43))
    counter = getattr(state.learner_state, counter_name)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{
                counter_name: jnp.full_like(counter, 2_147_483_647),
            }
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))

    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    result = lifecycle.observe_and_route(
        corrupt,
        oak,
        _consumer_binding(corrupt),
        event,
    )
    assert not bool(result.diagnostics.state_values_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_exact(result.state, corrupt)
    _assert_tree_exact(result.oak_state, oak)


@pytest.mark.parametrize("timer_name", ("birth_timestamp", "uptime_s"))
def test_owned_learner_timers_must_be_nonnegative(timer_name: str) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(45))
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{
                timer_name: jnp.asarray(-1.0, dtype=jnp.float32),
            }
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("feature_parent_a", 0),
        ("feature_parent_b", 0),
        ("candidate_parent_a", 0),
        ("candidate_parent_b", 0),
        ("feature_generator", 1),
        ("candidate_generator", 1),
    ),
)
def test_pair_only_lifecycle_rejects_compositional_provenance(
    field: str,
    replacement: int,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(46))
    current = getattr(state.learner_state, field)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{field: jnp.full_like(current, replacement)}
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


def test_fixed_disabled_learner_substate_must_remain_reachable() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(48))
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32),
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
        observe_words=jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(state))
    learner = state.learner_state
    corruptions = (
        learner.replace(
            relevance_probe_weights=learner.relevance_probe_weights.at[0, 0].set(1.0)
        ),
        learner.replace(
            relevance_probe_biases=learner.relevance_probe_biases.at[0].set(1.0)
        ),
        learner.replace(
            evidence_idle_steps=learner.evidence_idle_steps.at[0].set(1)
        ),
        learner.replace(
            utility_evidence_streak=learner.utility_evidence_streak.at[0].set(1)
        ),
        learner.replace(
            candidate_promotion_evidence_streak=(
                learner.candidate_promotion_evidence_streak.at[0].set(1)
            )
        ),
        learner.replace(
            active_output_memory_committed=(
                learner.active_output_memory_committed.at[0].set(True)
            )
        ),
        learner.replace(
            candidate_reacquisition_required=(
                learner.candidate_reacquisition_required.at[0].set(True)
            )
        ),
    )
    for corrupt_learner in corruptions:
        corrupt = state.replace(learner_state=corrupt_learner)
        assert not bool(lifecycle.state_valid(corrupt))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("utilities", -1.0),
        ("utilities", 2.0),
        ("candidate_utilities", -1.0),
        ("candidate_utilities", 2.0),
    ),
)
def test_scale_robust_utility_emas_remain_bounded(
    field: str,
    value: float,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(49))
    current = getattr(state.learner_state, field)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{field: current.at[0].set(value)}
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


def test_full_oak_audit_rejects_ownership_model_counter_and_outer_corruption() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(47))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    stomp = oak.stomp_state
    corruptions = (
        oak.replace(
            stomp_state=stomp.replace(
                base_last_action=jnp.asarray(1, dtype=jnp.int32)
            )
        ),
            oak.replace(
                stomp_state=stomp.replace(option_steps=jnp.asarray(2, dtype=jnp.int32))
        ),
        oak.replace(
            stomp_state=stomp.replace(
                option_models=stomp.option_models.replace(
                    duration_ema=stomp.option_models.duration_ema.at[0].set(-1.0)
                )
            )
        ),
            oak.replace(
                execution_counts=jnp.asarray([3, 0], dtype=jnp.int32)
        ),
        oak.replace(
            stomp_state=stomp.replace(
                option_models=stomp.option_models.replace(
                    n_completions=jnp.asarray(
                        [2_147_483_647, 0],
                        dtype=jnp.int32,
                    )
                )
            )
        ),
        oak.replace(
            stomp_state=stomp.replace(
                base_learner_state=stomp.base_learner_state.replace(
                    birth_timestamp=-1.0
                )
            )
        ),
        oak.replace(
            stomp_state=stomp.replace(
                base_learner_state=stomp.base_learner_state.replace(
                    uptime_s=jnp.asarray(-1.0, dtype=jnp.float32)
                )
            )
        ),
    )

    for corrupt in corruptions:
        result = lifecycle.observe_and_route(
            state,
            corrupt,
            _consumer_binding(state),
            event,
        )
        assert not bool(result.diagnostics.oak_values_valid)
        assert not bool(result.diagnostics.transaction_applied)
        _assert_tree_exact(result.state, state)
        _assert_tree_exact(result.oak_state, corrupt)


def test_max_observation_capacity_is_an_exact_atomic_noop() -> None:
    config = dataclasses.replace(_config(replacement_interval=0), max_observations=1)
    lifecycle = PrototypeFeatureLifecycle(config)
    state = lifecycle.init(jr.key(53))
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32),
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
        observe_words=jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert bool(lifecycle.state_valid(state))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert not bool(result.diagnostics.update_capacity_available)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)

    binding = _consumer_binding(state)
    prepared = lifecycle.prepare_observe_and_route(state, oak, binding, event)
    receipt = lifecycle.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    adopted = lifecycle.adopt_prepared_route(
        state,
        oak,
        binding,
        prepared,
        receipt,
    )
    expected_next = lifecycle.augment(state, event.next_observation)
    _assert_tree_exact(adopted.result.state, state)
    _assert_tree_exact(adopted.result.oak_state, oak)
    _assert_tree_exact(adopted.result.consumer_binding, binding)
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
    untrusted_receipt = lifecycle.external_readiness_receipt(
        untrusted,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    sanitized = lifecycle.adopt_prepared_route(
        state,
        oak,
        binding,
        untrusted,
        untrusted_receipt,
    )
    np.testing.assert_array_equal(
        np.asarray(sanitized.result.next_augmented_observation),
        np.asarray(expected_next),
    )


def test_static_contract_rejects_wrong_event_dtype_and_nonlinear_oak() -> None:
    class EventSubclass(PrototypeFeatureLifecycleEvent):
        pass

    class BindingSubclass(PrototypeFeatureConsumerBinding):
        pass

    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(19))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    with pytest.raises(TypeError, match="event.observation"):
        lifecycle.observe_and_route(
            state,
            oak,
            _consumer_binding(state),
            event.replace(observation=event.observation.astype(jnp.float16)),
        )
    subclass_event = EventSubclass(
        observation=event.observation,
        targets=event.targets,
        next_observation=event.next_observation,
        allow_curation=event.allow_curation,
    )
    with pytest.raises(TypeError, match="PrototypeFeatureLifecycleEvent"):
        lifecycle.observe_and_route(
            state,
            oak,
            _consumer_binding(state),
            subclass_event,
        )
    binding = _consumer_binding(state)
    binding_subclass = BindingSubclass(
        semantic_generation=binding.semantic_generation,
        semantic_generation_words=binding.semantic_generation_words,
        descriptors=binding.descriptors,
    )
    with pytest.raises(TypeError, match="PrototypeFeatureConsumerBinding"):
        lifecycle.observe_and_route(state, oak, binding_subclass, event)
    with pytest.raises(TypeError, match="consumer_binding.descriptors"):
        lifecycle.observe_and_route(
            state,
            oak,
            binding.replace(descriptors=binding.descriptors.astype(jnp.float32)),
            event,
        )
    nonlinear = _oak_agent(lifecycle.config, hidden=True).init(jr.key(20))
    nonlinear = nonlinear.replace(
        stomp_state=nonlinear.stomp_state.replace(
            base_last_obs=lifecycle.augment(state, event.next_observation)
        )
    )
    with pytest.raises(ValueError, match="linear OaK"):
        lifecycle.observe_and_route(
            state,
            nonlinear,
            _consumer_binding(state),
            event,
        )
    for malformed_timer in (
        jnp.asarray(1.0, dtype=jnp.float16),
        jnp.asarray(1.0 + 2.0j, dtype=jnp.complex64),
    ):
        malformed_oak = oak.replace(
            stomp_state=oak.stomp_state.replace(
                base_learner_state=oak.stomp_state.base_learner_state.replace(
                    birth_timestamp=malformed_timer
                )
            )
        )
        with pytest.raises(ValueError, match="linear OaK"):
            lifecycle.observe_and_route(
                state,
                malformed_oak,
                _consumer_binding(state),
                event,
            )


def test_eager_and_jit_transactions_are_exact_and_scan_is_recurrent() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(23)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    binding = _consumer_binding(state)
    eager = lifecycle.observe_and_route(state, oak, binding, event)
    compiled = jax.jit(lifecycle.observe_and_route)(state, oak, binding, event)
    _assert_tree_exact(compiled, eager)

    scan_lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    scan_state, scan_binding = scan_lifecycle.init_bound(jr.key(29))
    scan_oak = _seed_oak(scan_lifecycle, scan_state, event.next_observation)
    events = jax.tree.map(lambda value: jnp.stack([value, value, value]), event)

    def body(carry, one_event):
        one_state, one_oak, one_binding = carry
        result = scan_lifecycle.observe_and_route(
            one_state,
            one_oak,
            one_binding,
            one_event,
        )
        return (
            result.state,
            _advance_oak_clock(result.oak_state),
            result.consumer_binding,
        ), result.diagnostics.transaction_applied

    (scanned_state, scanned_oak, scanned_binding), applied = jax.jit(
        lambda initial, values: jax.lax.scan(body, initial, values)
    )((scan_state, scan_oak, scan_binding), events)
    loop_state, loop_oak, loop_binding = scan_state, scan_oak, scan_binding
    for _ in range(3):
        result = scan_lifecycle.observe_and_route(
            loop_state,
            loop_oak,
            loop_binding,
            event,
        )
        loop_state, loop_oak, loop_binding = (
            result.state,
            _advance_oak_clock(result.oak_state),
            result.consumer_binding,
        )
    # XLA may reassociate the scan's float32 EMA arithmetic by one ULP; all
    # integer, descriptor, counter, and key leaves remain bit exact.
    _assert_tree_close(scanned_state, loop_state)
    _assert_tree_close(scanned_oak, loop_oak)
    _assert_tree_exact(scanned_binding, loop_binding)
    np.testing.assert_array_equal(applied, np.ones(3, dtype=np.bool_))
    assert int(scanned_state.observe_count) == 3
    assert int(scanned_state.router_state.generation_count) == 0


def test_external_readiness_adopts_route_or_retains_exact_ordinary_successor() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(611)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    binding = _consumer_binding(state)

    prepared = lifecycle.prepare_observe_and_route(state, oak, binding, event)
    assert bool(prepared.internally_valid)
    assert bool(prepared.destination_result.diagnostics.curation_committed)
    assert int(prepared.preparation_learner_update_evaluations) == 1
    lifecycle._learner = cast(Any, _NoUpdateProxy(lifecycle._learner))
    lifecycle._router = cast(Any, _NoRouteProxy(lifecycle._router))

    ready_receipt = lifecycle.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    ready = lifecycle.adopt_prepared_route(
        state,
        oak,
        binding,
        prepared,
        ready_receipt,
    )
    _assert_tree_exact(ready.result, prepared.destination_result)
    assert bool(ready.diagnostics.destination_adopted)
    assert not bool(ready.diagnostics.external_curation_rolled_back)

    veto_receipt = lifecycle.external_readiness_receipt(
        prepared,
        jnp.asarray(False, dtype=jnp.bool_),
    )
    veto = lifecycle.adopt_prepared_route(
        state,
        oak,
        binding,
        prepared,
        veto_receipt,
    )
    _assert_tree_exact(veto.result, prepared.ordinary_result)
    _assert_tree_exact(veto.result.oak_state, oak)
    _assert_tree_exact(veto.result.consumer_binding, binding)
    _assert_tree_exact(veto.result.state.router_state, state.router_state)
    assert int(veto.result.state.observe_count) == int(state.observe_count) + 1
    assert int(veto.result.state.committed_curation_count) == int(
        state.committed_curation_count
    )
    assert int(veto.result.state.rolled_back_curation_count) == int(
        state.rolled_back_curation_count
    ) + 1
    assert not bool(veto.result.diagnostics.curation_deferred)
    assert bool(veto.result.diagnostics.curation_rolled_back)
    assert bool(veto.diagnostics.ordinary_update_retained)
    assert bool(veto.diagnostics.external_curation_rolled_back)
    assert int(veto.diagnostics.adoption_learner_update_evaluations) == 0
    assert int(veto.diagnostics.total_learner_update_evaluations) == 1
    transient = lifecycle.external_transaction_resource_budget(
        prepared,
        veto_receipt,
    )
    prepared_nbytes = sum(
        int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(prepared)
    )
    receipt_nbytes = sum(
        int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(veto_receipt)
    )
    assert transient.prepared_route_logical_nbytes == prepared_nbytes
    assert transient.readiness_receipt_logical_nbytes == receipt_nbytes
    assert transient.simultaneous_logical_transient_nbytes == (
        prepared_nbytes + receipt_nbytes
    )
    assert (
        transient.lifecycle_persistent_state_nbytes_before
        == lifecycle.resource_budget(state).lifecycle_state_nbytes
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


def test_external_readiness_rejects_internal_invalidity_staleness_and_tampering() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(612)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    binding = _consumer_binding(state)
    prepared = lifecycle.prepare_observe_and_route(state, oak, binding, event)
    receipt = lifecycle.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )

    tampered = prepared.replace(
        destination_result=prepared.destination_result.replace(
            predictions=prepared.destination_result.predictions.at[0].add(1.0),
        )
    )
    refused = lifecycle.adopt_prepared_route(
        state,
        oak,
        binding,
        tampered,
        receipt,
    )
    assert not bool(refused.diagnostics.receipt_matches_preparation)
    assert bool(refused.diagnostics.rejected)
    _assert_tree_exact(refused.result.state, state)
    _assert_tree_exact(refused.result.oak_state, oak)
    np.testing.assert_array_equal(
        np.asarray(refused.result.next_augmented_observation),
        np.zeros_like(np.asarray(refused.result.next_augmented_observation)),
    )

    stale_state = prepared.destination_result.state
    stale = lifecycle.adopt_prepared_route(
        stale_state,
        oak,
        binding,
        prepared,
        receipt,
    )
    assert not bool(stale.diagnostics.source_state_matches)
    _assert_tree_exact(stale.result.state, stale_state)

    invalid_event = event.replace(observation=event.observation.at[0].set(jnp.inf))
    invalid_prepared = lifecycle.prepare_observe_and_route(
        state,
        oak,
        binding,
        invalid_event,
    )
    for ready in (False, True):
        invalid_receipt = lifecycle.external_readiness_receipt(
            invalid_prepared,
            jnp.asarray(ready, dtype=jnp.bool_),
        )
        invalid = lifecycle.adopt_prepared_route(
            state,
            oak,
            binding,
            invalid_prepared,
            invalid_receipt,
        )
        assert not bool(invalid.diagnostics.preparation_internally_valid)
        assert bool(invalid.diagnostics.rejected)
        _assert_tree_exact(invalid.result.state, state)
        _assert_tree_exact(invalid.result.oak_state, oak)
        np.testing.assert_array_equal(
            np.asarray(invalid.result.next_augmented_observation),
            np.zeros_like(np.asarray(invalid.result.next_augmented_observation)),
        )


def test_external_readiness_adoption_is_eager_jit_exact() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(613)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    binding = _consumer_binding(state)
    prepared = lifecycle.prepare_observe_and_route(state, oak, binding, event)
    receipt = lifecycle.external_readiness_receipt(
        prepared,
        jnp.asarray(False, dtype=jnp.bool_),
    )

    eager = lifecycle.adopt_prepared_route(state, oak, binding, prepared, receipt)
    compiled = jax.jit(lifecycle.adopt_prepared_route)(
        state,
        oak,
        binding,
        prepared,
        receipt,
    )
    _assert_tree_exact(eager, compiled)


def test_checkpoint_round_trip_is_strict_and_resource_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    state = lifecycle.init(jr.key(31))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    state = lifecycle.observe_and_route(
        state,
        oak,
        _consumer_binding(state),
        event,
    ).state
    path = tmp_path / "feature-lifecycle"

    save_prototype_feature_lifecycle_checkpoint(lifecycle, state, path)
    restored_lifecycle, restored_state = load_prototype_feature_lifecycle_checkpoint(path)

    assert restored_lifecycle.config == lifecycle.config
    assert restored_lifecycle.resource_budget(restored_state) == lifecycle.resource_budget(state)
    assert bool(restored_lifecycle.state_valid(restored_state))
    _assert_tree_exact(restored_state, state)

    metadata = load_checkpoint_metadata(path)
    raw_budget = cast(dict[str, object], metadata["resource_budget"])
    type_tampers = (
        ("scientific_promotion_allowed", 0),
        (
            "owned_persistent_state_nbytes",
            float(cast(int, raw_budget["owned_persistent_state_nbytes"])),
        ),
    )
    for field, replacement in type_tampers:
        tampered = copy.deepcopy(metadata)
        tampered_budget = cast(dict[str, object], tampered["resource_budget"])
        tampered_budget[field] = replacement
        monkeypatch.setattr(
            lifecycle_module,
            "load_checkpoint_metadata",
            lambda _path, payload=tampered: payload,
        )
        monkeypatch.setattr(
            lifecycle_module,
            "load_checkpoint",
            lambda _template, _path, payload=tampered: (state, payload),
        )
        with pytest.raises(ValueError, match="resource contract changed"):
            load_prototype_feature_lifecycle_checkpoint(path)

    changed_between_reads = copy.deepcopy(metadata)
    changed_budget = cast(
        dict[str, object],
        changed_between_reads["resource_budget"],
    )
    changed_budget["scientific_promotion_allowed"] = 0
    monkeypatch.setattr(
        lifecycle_module,
        "load_checkpoint_metadata",
        lambda _path: metadata,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "load_checkpoint",
        lambda _template, _path: (state, changed_between_reads),
    )
    with pytest.raises(ValueError, match="metadata changed between reads"):
        load_prototype_feature_lifecycle_checkpoint(path)

    corrupt = state.replace(observe_count=jnp.asarray(-1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="state is invalid"):
        save_prototype_feature_lifecycle_checkpoint(
            lifecycle,
            corrupt,
            tmp_path / "corrupt",
        )
