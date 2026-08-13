# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Real old-bank consumer updates through the compositional router.

This is a consumed-key, development-only transaction test.  It establishes
live public-update wiring; it does not assess feature benefit or close the
Prototype integration gap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.compositional_consumer_router import (
    CompositionalConsumerRouter,
    CompositionalConsumerState,
)
from alberta_framework.core.compositional_feature_adapter import (
    CompositionalFeatureAdapter,
)
from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = [pytest.mark.integration, pytest.mark.development]


def _tree_bits_equal(left: object, right: object) -> bool:
    if str(jax.tree_util.tree_structure(left)) != str(
        jax.tree_util.tree_structure(right)
    ):
        return False
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        left_dtype = getattr(left_leaf, "dtype", None)
        if left_dtype is not None and jax.dtypes.issubdtype(
            left_dtype,
            jax.dtypes.prng_key,
        ):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        if left_array.dtype == jnp.dtype(jnp.float32):
            left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        if not bool(jnp.all(left_array == right_array)):
            return False
    return True


def _float32_bits(value: Array) -> np.ndarray:
    array = np.asarray(value)
    assert array.dtype == np.float32
    return array.view(np.uint32)


def _assert_last_axis_route(
    post_update: Array,
    routed: Array,
    changed: np.ndarray,
) -> None:
    before_bits = _float32_bits(post_update)
    after_bits = _float32_bits(routed)
    np.testing.assert_array_equal(after_bits[..., ~changed], before_bits[..., ~changed])
    np.testing.assert_array_equal(
        after_bits[..., changed],
        np.zeros_like(after_bits[..., changed]),
    )


def _assert_matrix_axis_route(
    post_update: Array,
    routed: Array,
    changed: np.ndarray,
) -> None:
    before_bits = _float32_bits(post_update)
    after_bits = _float32_bits(routed)
    survivor_cells = (~changed)[None, :, None] & (~changed)[None, None, :]
    np.testing.assert_array_equal(after_bits[survivor_cells], before_bits[survivor_cells])
    np.testing.assert_array_equal(
        after_bits[~survivor_cells],
        np.zeros_like(after_bits[~survivor_cells]),
    )


def _head_weights_changed(left: object, right: object) -> bool:
    left_weights = left.head_params.weights
    right_weights = right.head_params.weights
    return any(
        not np.array_equal(_float32_bits(before), _float32_bits(after))
        for before, after in zip(left_weights, right_weights, strict=True)
    )


def _safe_curation_boundary(state: OaKState, n_primitive_actions: int) -> Array:
    """Derive permission only from the completed OaK transition."""

    stomp = state.stomp_state
    return (
        (stomp.executing_option == jnp.asarray(-1, dtype=jnp.int32))
        & (stomp.base_last_action >= jnp.asarray(0, dtype=jnp.int32))
        & (
            stomp.base_last_action
            < jnp.asarray(n_primitive_actions, dtype=jnp.int32)
        )
    )


def _current_outcome_cumulants(
    reward: float,
    current_raw: Array,
    next_raw: Array,
) -> Array:
    """Use only reward and observation change owned by this transition."""

    return jnp.asarray(
        (reward, next_raw[1] - current_raw[1]),
        dtype=jnp.float32,
    )


def test_real_oak_horde_updates_commit_unsafe_learning_then_one_safe_birth() -> None:
    adapter = CompositionalFeatureAdapter(
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            candidate_count=0,
            step_size_output=0.05,
            step_size_theta=0.0,
            utility_decay=0.999,
            replacement_interval=1,
            min_feature_age=0,
            use_obgd=False,
            train_candidate_theta=False,
        ),
        base_feature_dim=2,
    )
    adapter_state0 = adapter.init(jr.key(7000))

    oak_config = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=1,
                ),
            ),
            observation_dim=adapter.n_features,
            n_primitive_actions=2,
            base_hidden_sizes=(),
            base_step_size=0.1,
            base_avg_reward_step_size=0.01,
            base_trace_decay=0.0,
            option_step_size=0.1,
            option_avg_reward_step_size=0.01,
            option_trace_decay=0.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    oak_agent = OaKAgent(oak_config)
    pristine_oak = oak_agent.init(jr.key(0))
    horde_spec = create_horde_spec(
        (
            GVFSpec(
                name="reward",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
            GVFSpec(
                name="raw_delta",
                demon_type=DemonType.PREDICTION,
                gamma=0.9,
                lamda=0.8,
                cumulant_index=1,
            ),
        )
    )
    horde = HordeLearner(horde_spec, hidden_sizes=(), step_size=0.1)
    pristine_horde = horde.init(adapter.n_features, jr.key(7002))
    router = CompositionalConsumerRouter(
        adapter,
        oak_config,
        pristine_oak,
        pristine_horde,
    )
    consumer_state0 = router.bind_pristine(
        adapter_state0,
        pristine_oak,
        pristine_horde,
    )

    raw0 = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
    raw1 = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    raw2 = jnp.asarray((-0.25, 0.75), dtype=jnp.float32)
    phi0 = adapter.representation(adapter_state0, raw0)
    started_oak = oak_agent.start(consumer_state0.oak_state, phi0)
    consumer_state0 = consumer_state0.replace(oak_state=started_oak)

    assert bool(router.state_valid(consumer_state0, adapter_state0))
    assert int(started_oak.stomp_state.executing_option) == -1
    assert int(started_oak.stomp_state.base_last_action) == 1
    assert _tree_bits_equal(started_oak.stomp_state.base_last_obs, phi0)

    # Transition 1: both consumers learn from A0 representations before any
    # adapter proposal exists.  All signals are owned by this observed
    # transition; no task label, task id, or future sample controls curation.
    reward1 = 0.3
    phi1_old = adapter.representation(adapter_state0, raw1)
    oak_update1 = oak_agent.update(
        consumer_state0.oak_state,
        jnp.asarray(reward1, dtype=jnp.float32),
        phi1_old,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert consumer_state0.horde_state is not None
    horde_update1 = horde.update(
        consumer_state0.horde_state,
        phi0,
        _current_outcome_cumulants(reward1, raw0, raw1),
        phi1_old,
    )
    post_update1 = CompositionalConsumerState(
        binding=consumer_state0.binding,
        oak_state=oak_update1.state,
        horde_state=horde_update1.state,
    )
    curation_allowed1 = _safe_curation_boundary(
        oak_update1.state,
        oak_config.n_primitive_actions,
    )

    assert bool(oak_update1.update_applied)
    assert bool(horde_update1.update_applied)
    assert not bool(curation_allowed1)
    assert int(oak_update1.state.stomp_state.executing_option) == 0
    assert int(oak_update1.state.stomp_state.base_last_action) == 2
    assert _head_weights_changed(
        consumer_state0.oak_state.stomp_state.base_learner_state,
        oak_update1.state.stomp_state.base_learner_state,
    )
    assert _head_weights_changed(consumer_state0.horde_state, horde_update1.state)
    assert _tree_bits_equal(oak_update1.state.stomp_state.base_last_obs, phi1_old)

    proposal1 = adapter.prepare_update(
        adapter_state0,
        raw1,
        jnp.asarray((reward1,), dtype=jnp.float32),
        curation_allowed=curation_allowed1,
    )
    denied_trace = proposal1.curation_trace
    assert not bool(proposal1.curation_allowed)
    assert not bool(proposal1.diagnostics.active_bank_changed)
    assert not bool(denied_trace.should_try_replace)
    assert not bool(denied_trace.proposal_formed)
    assert not bool(denied_trace.has_event)
    assert not bool(jnp.any(denied_trace.active_change_mask))
    assert int(denied_trace.logical_event_count) == 0
    assert int(denied_trace.pre_replacement_phase) == 0
    assert int(denied_trace.post_replacement_phase) == 0
    assert int(proposal1.candidate_state.learner_state.replacement_phase) == 0
    assert float(proposal1.candidate_state.learner_state.replacement_accumulator) == 0.0

    route1 = router.prepare_route(consumer_state0, post_update1, proposal1)
    committed1 = router.commit_prepared_route(
        adapter_state0,
        consumer_state0,
        route1,
    )

    assert bool(route1.receipt.consumers_ready)
    assert bool(committed1.diagnostics.applied)
    assert not bool(committed1.diagnostics.rejected)
    assert _tree_bits_equal(committed1.adapter_state, proposal1.candidate_state)
    assert _tree_bits_equal(committed1.consumer_state, post_update1)
    assert bool(router.state_valid(committed1.consumer_state, committed1.adapter_state))
    np.testing.assert_array_equal(route1.receipt.changed_birth_mask, (False,) * 4)
    committed_horde1 = committed1.consumer_state.horde_state
    assert committed_horde1 is not None
    for words in (
        committed1.adapter_state.learner_state.step_words,
        committed1.consumer_state.oak_state.step_words,
        committed1.consumer_state.oak_state.stomp_state.step_words,
        committed1.consumer_state.oak_state.stomp_state.base_learner_state.step_words,
        committed_horde1.step_words,
    ):
        np.testing.assert_array_equal(words, (0, 1))
    assert _tree_bits_equal(
        committed1.consumer_state.oak_state.stomp_state.base_last_obs,
        adapter.representation(committed1.adapter_state, raw1),
    )
    np.testing.assert_array_equal(
        committed1.adapter_state.binding.semantic_generation_words,
        (0, 0),
    )

    # Transition 2 starts from the committed learned states and again uses
    # only the current A1 bank.  The one-step option terminates through the
    # ordinary API and selects a primitive action, making curation safe.
    reward2 = -0.4
    phi1_current = adapter.representation(committed1.adapter_state, raw1)
    phi2_old = adapter.representation(committed1.adapter_state, raw2)
    oak_update2 = oak_agent.update(
        committed1.consumer_state.oak_state,
        jnp.asarray(reward2, dtype=jnp.float32),
        phi2_old,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert committed1.consumer_state.horde_state is not None
    horde_update2 = horde.update(
        committed1.consumer_state.horde_state,
        phi1_current,
        _current_outcome_cumulants(reward2, raw1, raw2),
        phi2_old,
    )
    post_update2 = CompositionalConsumerState(
        binding=committed1.consumer_state.binding,
        oak_state=oak_update2.state,
        horde_state=horde_update2.state,
    )
    curation_allowed2 = _safe_curation_boundary(
        oak_update2.state,
        oak_config.n_primitive_actions,
    )

    assert bool(oak_update2.update_applied)
    assert bool(horde_update2.update_applied)
    assert bool(oak_update2.option_terminated)
    assert bool(curation_allowed2)
    assert int(oak_update2.state.stomp_state.executing_option) == -1
    assert int(oak_update2.state.stomp_state.base_last_action) == 0
    assert _head_weights_changed(
        committed1.consumer_state.oak_state.stomp_state.base_learner_state,
        oak_update2.state.stomp_state.base_learner_state,
    )
    assert _head_weights_changed(
        committed1.consumer_state.horde_state,
        horde_update2.state,
    )
    assert _tree_bits_equal(oak_update2.state.stomp_state.base_last_obs, phi2_old)

    proposal2 = adapter.prepare_update(
        committed1.adapter_state,
        raw2,
        jnp.asarray((reward2,), dtype=jnp.float32),
        curation_allowed=curation_allowed2,
    )
    assert bool(proposal2.curation_allowed)
    assert bool(proposal2.curation_trace.should_try_replace)
    assert bool(proposal2.curation_trace.has_event)
    assert bool(proposal2.diagnostics.active_bank_changed)
    np.testing.assert_array_equal(
        proposal2.candidate_state.binding.semantic_generation_words,
        (0, 1),
    )

    route2 = router.prepare_route(committed1.consumer_state, post_update2, proposal2)
    committed2 = router.commit_prepared_route(
        committed1.adapter_state,
        committed1.consumer_state,
        route2,
    )
    changed = np.asarray(route2.receipt.changed_birth_mask, dtype=np.bool_)

    assert bool(route2.receipt.consumers_ready)
    assert bool(route2.diagnostics.survivor_columns_bit_exact)
    assert bool(route2.diagnostics.changed_columns_scrubbed)
    assert bool(route2.diagnostics.optimizer_state_authenticated)
    assert bool(committed2.diagnostics.applied)
    assert not bool(committed2.diagnostics.rejected)
    assert bool(router.state_valid(committed2.consumer_state, committed2.adapter_state))
    np.testing.assert_array_equal(changed, (False, False, True, False))
    np.testing.assert_array_equal(
        proposal2.curation_trace.active_change_mask,
        changed,
    )
    committed_horde2 = committed2.consumer_state.horde_state
    assert committed_horde2 is not None
    for words in (
        committed2.adapter_state.learner_state.step_words,
        committed2.consumer_state.oak_state.step_words,
        committed2.consumer_state.oak_state.stomp_state.step_words,
        committed2.consumer_state.oak_state.stomp_state.base_learner_state.step_words,
        committed_horde2.step_words,
    ):
        np.testing.assert_array_equal(words, (0, 2))

    # The route is checked against the actual public-update results, not a
    # synthetic fixture: survivor columns retain their exact bits, every
    # changed feature axis is scrubbed, and scalar/bias optimizer state stays
    # exactly as learned before routing.
    post_stomp = post_update2.oak_state.stomp_state
    routed_stomp = committed2.consumer_state.oak_state.stomp_state
    for post_weight, routed_weight in zip(
        post_stomp.base_learner_state.head_params.weights,
        routed_stomp.base_learner_state.head_params.weights,
        strict=True,
    ):
        _assert_last_axis_route(post_weight, routed_weight, changed)
    for post_trace, routed_trace in zip(
        post_stomp.base_learner_state.head_traces,
        routed_stomp.base_learner_state.head_traces,
        strict=True,
    ):
        _assert_last_axis_route(post_trace[0], routed_trace[0], changed)
    _assert_last_axis_route(
        post_stomp.option_start_obs,
        routed_stomp.option_start_obs,
        changed,
    )
    _assert_last_axis_route(
        post_stomp.option_policies.q_weights,
        routed_stomp.option_policies.q_weights,
        changed,
    )
    _assert_last_axis_route(
        post_stomp.option_policies.traces,
        routed_stomp.option_policies.traces,
        changed,
    )
    _assert_matrix_axis_route(
        post_stomp.option_models.next_state_weights,
        routed_stomp.option_models.next_state_weights,
        changed,
    )

    assert post_update2.horde_state is not None
    assert committed2.consumer_state.horde_state is not None
    for post_weight, routed_weight in zip(
        post_update2.horde_state.head_params.weights,
        committed2.consumer_state.horde_state.head_params.weights,
        strict=True,
    ):
        _assert_last_axis_route(post_weight, routed_weight, changed)
    for post_trace, routed_trace in zip(
        post_update2.horde_state.head_traces,
        committed2.consumer_state.horde_state.head_traces,
        strict=True,
    ):
        _assert_last_axis_route(post_trace[0], routed_trace[0], changed)

    assert _tree_bits_equal(
        post_stomp.base_learner_state.head_params.biases,
        routed_stomp.base_learner_state.head_params.biases,
    )
    assert _tree_bits_equal(
        tuple(pair[1] for pair in post_stomp.base_learner_state.head_traces),
        tuple(pair[1] for pair in routed_stomp.base_learner_state.head_traces),
    )
    assert _tree_bits_equal(
        post_stomp.base_learner_state.head_optimizer_states,
        routed_stomp.base_learner_state.head_optimizer_states,
    )
    assert _tree_bits_equal(
        post_update2.horde_state.head_params.biases,
        committed2.consumer_state.horde_state.head_params.biases,
    )
    assert _tree_bits_equal(
        tuple(pair[1] for pair in post_update2.horde_state.head_traces),
        tuple(
            pair[1]
            for pair in committed2.consumer_state.horde_state.head_traces
        ),
    )
    assert _tree_bits_equal(
        post_update2.horde_state.head_optimizer_states,
        committed2.consumer_state.horde_state.head_optimizer_states,
    )

    candidate_cache = adapter.representation(committed2.adapter_state, raw2)
    assert _tree_bits_equal(routed_stomp.base_last_obs, candidate_cache)
    assert not _tree_bits_equal(routed_stomp.base_last_obs, phi2_old)
    np.testing.assert_array_equal(
        committed2.adapter_state.binding.semantic_generation_words,
        (0, 1),
    )
