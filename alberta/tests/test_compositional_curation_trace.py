"""Public compositional-curation trace contracts."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CURATION_DESTINATION_ACTIVE,
    CURATION_DESTINATION_CANDIDATE,
    CURATION_DESTINATION_NONE,
    CompositionalCurationTrace,
)
from alberta_framework.core import compositional_features as cf


class _TraceProposalLearner(cf.CompositionalFeatureLearner):
    def __init__(
        self,
        *args: Any,
        proposal_parent: int,
        proposal_op: int = cf.OP_PRODUCT,
        **kwargs: Any,
    ) -> None:
        self.proposal_parent = proposal_parent
        self.proposal_op = proposal_op
        super().__init__(*args, **kwargs)

    def _generate_one(
        self,
        key: jax.Array,
        existing_depth: jax.Array,
        existing_utilities: jax.Array | None = None,
        existing_ages: jax.Array | None = None,
        feature_values: jax.Array | None = None,
        feature_credit: jax.Array | None = None,
        forced_op: jax.Array | None = None,
        parent_mode: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del (
            key,
            existing_utilities,
            existing_ages,
            feature_values,
            feature_credit,
            forced_op,
            parent_mode,
        )
        parent_a = jnp.asarray(self.proposal_parent, dtype=jnp.int32)
        parent_b = jnp.asarray(0, dtype=jnp.int32)
        proposal_depth = (
            jnp.maximum(existing_depth[parent_a], existing_depth[parent_b]) + 1
        )
        return (
            jnp.asarray(self.proposal_op, dtype=jnp.int32),
            parent_a,
            parent_b,
            jnp.asarray((0.5, -0.5), dtype=jnp.float32),
            proposal_depth.astype(jnp.int32),
        )

    def _initial_candidate_output_weights(
        self,
        op: jax.Array,
        parent_a: jax.Array,
        parent_b: jax.Array,
        theta: jax.Array,
        active_values: jax.Array,
        observation: jax.Array,
        errors: jax.Array,
        active_count: jax.Array,
        imprint_scale: jax.Array | None = None,
    ) -> jax.Array:
        del (
            op,
            parent_a,
            parent_b,
            theta,
            active_values,
            observation,
            active_count,
            imprint_scale,
        )
        return jnp.full_like(errors, 23.0)


def _assert_key_schedule(
    pre_state: cf.CompositionalFeatureState,
    result: cf.CompositionalFeatureUpdateResult,
) -> None:
    expected_post, expected_decision, expected_curation = jr.split(pre_state.key, 3)
    expected_proposal, expected_cascade = cf.compositional_curation_keys(
        expected_curation
    )
    expected_overdepth_regeneration = jr.fold_in(
        expected_curation,
        jnp.uint32(cf.COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL),
    )
    trace = result.curation_trace
    assert int(trace.pre_step) == int(pre_state.step_count)
    assert int(trace.post_step) == int(result.state.step_count)
    for actual, expected in (
        (result.state.key, expected_post),
        (trace.decision_key, expected_decision),
        (trace.curation_key, expected_curation),
        (trace.proposal_key, expected_proposal),
        (trace.cascade_key, expected_cascade),
        (
            trace.candidate_overdepth_regeneration_key,
            expected_overdepth_regeneration,
        ),
    ):
        actual_data = np.asarray(jr.key_data(actual))
        expected_data = np.asarray(jr.key_data(expected))
        assert actual_data.dtype == np.uint32
        np.testing.assert_array_equal(actual_data, expected_data)
    key_words = {
        tuple(int(word) for word in jr.key_data(derived_key))
        for derived_key in (
            expected_proposal,
            expected_cascade,
            expected_overdepth_regeneration,
        )
    }
    assert len(key_words) == 3


def _assert_count_consistency(result: cf.CompositionalFeatureUpdateResult) -> None:
    trace = result.curation_trace
    root_count = int(jnp.sum(trace.root_change_mask.astype(jnp.int32)))
    cascade_count = int(jnp.sum(trace.cascade_refill_mask.astype(jnp.int32)))
    ordinary_count = int(
        jnp.sum(trace.ordinary_candidate_refresh_mask.astype(jnp.int32))
    )
    post_promotion_count = int(
        jnp.sum(trace.post_promotion_candidate_refresh_mask.astype(jnp.int32))
    )
    refresh_count = int(jnp.sum(trace.candidate_refresh_mask.astype(jnp.int32)))
    rebound_count = int(jnp.sum(trace.candidate_rebound_mask.astype(jnp.int32)))
    regeneration_count = int(
        jnp.sum(trace.candidate_overdepth_regeneration_mask.astype(jnp.int32))
    )
    assert int(trace.proposal_count) == int(trace.proposal_formed)
    assert int(trace.root_change_count) == root_count
    assert int(trace.promotion_count) == int(trace.promotion_applied)
    assert int(trace.cascade_refill_count) == cascade_count
    assert int(trace.ordinary_candidate_refresh_count) == ordinary_count
    assert int(trace.post_promotion_candidate_refresh_count) == post_promotion_count
    assert int(trace.candidate_refresh_count) == refresh_count
    assert int(trace.candidate_rebound_count) == rebound_count
    assert int(trace.candidate_overdepth_regeneration_count) == regeneration_count
    expected_logical = (
        root_count
        + cascade_count
        + refresh_count
        + rebound_count
        + regeneration_count
    )
    assert int(trace.logical_event_count) == expected_logical
    assert bool(trace.has_event) == (expected_logical > 0)


def _assert_final_descriptor_snapshots(
    result: cf.CompositionalFeatureUpdateResult,
) -> None:
    trace = result.curation_trace
    post = result.state
    for actual, expected in (
        (trace.cascade_final_ops, post.ops),
        (trace.cascade_final_parent_a, post.parent_a),
        (trace.cascade_final_parent_b, post.parent_b),
        (trace.cascade_final_theta, post.theta),
        (trace.cascade_final_depth, post.depth),
        (trace.cascade_final_generator_policy, post.feature_generator_policy),
        (trace.candidate_final_ops, post.candidate_ops),
        (trace.candidate_final_parent_a, post.candidate_parent_a),
        (trace.candidate_final_parent_b, post.candidate_parent_b),
        (trace.candidate_final_theta, post.candidate_theta),
        (trace.candidate_final_depth, post.candidate_depth),
        (
            trace.candidate_final_generator_policy,
            post.candidate_generator_policy,
        ),
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _promotion_state(
    learner: cf.CompositionalFeatureLearner,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=3, key=jr.key(1201))
    return state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        theta=jnp.asarray(
            ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.1, 0.2), (0.3, 0.4)),
            dtype=jnp.float32,
        ),
        depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        output_weights=jnp.zeros_like(state.output_weights),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
        feature_generator_policy=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_PRODUCT), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_theta=jnp.asarray(((0.1, 0.2), (0.3, 0.4)), dtype=jnp.float32),
        candidate_depth=jnp.asarray((1, 3), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((6.0, 7.0),), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((100.0, 1.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10, 10), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((3, 2), dtype=jnp.int32),
    )


def _promotion_result(
    proposal_parent: int,
) -> tuple[cf.CompositionalFeatureState, cf.CompositionalFeatureUpdateResult]:
    learner = _TraceProposalLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        proposal_parent=proposal_parent,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        use_obgd=False,
    )
    state = _promotion_state(learner)
    result = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    return state, result


def test_no_event_trace_is_scan_compatible_and_has_exact_keys() -> None:
    learner = _TraceProposalLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        proposal_parent=0,
        replacement_interval=32,
        min_feature_age=100,
        candidate_min_age=16,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(1200))
    observation = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
    target = jnp.asarray((0.75,), dtype=jnp.float32)

    def scan_step(
        carry: cf.CompositionalFeatureState,
        _index: jax.Array,
    ) -> tuple[cf.CompositionalFeatureState, cf.CompositionalCurationTrace]:
        result = learner.update(carry, observation, target)
        return result.state, result.curation_trace

    _post, stacked_trace = jax.lax.scan(scan_step, state, jnp.arange(1))
    trace = jax.tree.map(lambda value: value[0], stacked_trace)
    direct_result = learner.update(state, observation, target)

    _assert_key_schedule(state, direct_result)
    _assert_count_consistency(direct_result)
    _assert_final_descriptor_snapshots(direct_result)
    assert isinstance(trace, CompositionalCurationTrace)
    assert int(trace.pre_step) == 0
    assert int(trace.post_step) == 1
    assert not bool(trace.should_try_replace)
    assert not bool(trace.has_event)
    assert not bool(trace.proposal_formed)
    assert int(trace.proposal_destination_bank) == CURATION_DESTINATION_NONE
    assert int(trace.proposal_destination_slot) == -1
    assert int(trace.logical_event_count) == 0
    assert not bool(jnp.any(trace.root_change_mask))
    assert not bool(jnp.any(trace.cascade_refill_mask))
    assert not bool(jnp.any(trace.candidate_refresh_mask))
    assert not bool(jnp.any(trace.candidate_rebound_mask))
    assert not bool(jnp.any(trace.candidate_overdepth_regeneration_mask))
    assert trace.candidate_overdepth_regeneration_mask.shape == (1,)
    assert int(trace.candidate_overdepth_regeneration_count) == 0


def test_ordinary_candidate_refresh_trace_matches_post_state() -> None:
    learner = _TraceProposalLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        proposal_parent=0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0e6,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(1202)).replace(  # type: ignore[attr-defined]
        ages=jnp.full((4,), 10, dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 1.0, 1.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((0.0,), dtype=jnp.float32),
    )
    result = learner.update(
        state,
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    trace = result.curation_trace

    _assert_key_schedule(state, result)
    _assert_count_consistency(result)
    _assert_final_descriptor_snapshots(result)
    assert bool(trace.should_try_replace)
    assert bool(trace.has_event)
    assert bool(trace.proposal_formed)
    assert int(trace.proposal_destination_bank) == CURATION_DESTINATION_CANDIDATE
    assert int(trace.proposal_destination_slot) == 0
    np.testing.assert_array_equal(np.asarray(trace.ordinary_candidate_refresh_mask), [True])
    np.testing.assert_array_equal(
        np.asarray(trace.post_promotion_candidate_refresh_mask), [False]
    )
    assert not bool(trace.promotion_applied)
    assert int(result.replaced_slot) == -1
    assert int(result.promoted_candidate) == -1
    assert int(trace.proposal_op) == int(result.state.candidate_ops[0])
    assert int(trace.proposal_parent_a) == int(result.state.candidate_parent_a[0])
    np.testing.assert_array_equal(
        np.asarray(trace.proposal_theta),
        np.asarray(result.state.candidate_theta[0]),
    )
    assert int(trace.proposal_count) == 1
    assert int(trace.candidate_refresh_count) == 1
    assert int(trace.logical_event_count) == 1


def test_promotion_trace_preserves_root_only_fresh_proposal_imprint() -> None:
    state, result = _promotion_result(proposal_parent=3)
    trace = result.curation_trace

    _assert_key_schedule(state, result)
    _assert_count_consistency(result)
    _assert_final_descriptor_snapshots(result)
    assert bool(trace.promotion_applied)
    assert int(trace.promotion_source_candidate) == 0
    assert int(trace.promotion_destination_active) == 3
    assert int(result.promoted_candidate) == 0
    assert int(result.replaced_slot) == 3
    np.testing.assert_array_equal(
        np.asarray(trace.root_change_mask),
        np.asarray((False, False, False, True, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.cascade_refill_mask),
        np.asarray((False, False, False, False, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.active_change_mask),
        np.asarray((False, False, False, True, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.post_promotion_candidate_refresh_mask),
        np.asarray((True, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.candidate_rebound_mask),
        np.asarray((False, True)),
    )
    assert int(trace.proposal_destination_bank) == CURATION_DESTINATION_CANDIDATE
    assert int(trace.proposal_destination_slot) == 0
    assert int(trace.proposal_parent_a) == 3
    assert float(result.state.candidate_output_weights[0, 0]) == 23.0
    assert int(trace.promoted_pre_refresh_op) == int(state.candidate_ops[0])
    assert int(trace.promoted_pre_refresh_parent_a) == int(state.candidate_parent_a[0])
    assert int(trace.promoted_pre_refresh_generator_policy) == 3
    assert int(trace.post_root_pre_cascade_slot) == 3
    assert int(trace.post_root_pre_cascade_op) == int(state.candidate_ops[0])
    assert int(trace.post_root_pre_cascade_generator_policy) == 3
    assert int(trace.logical_event_count) == 4


def test_promotion_trace_marks_fresh_proposal_rebound_by_later_cascade() -> None:
    state, result = _promotion_result(proposal_parent=4)
    trace = result.curation_trace

    _assert_key_schedule(state, result)
    _assert_count_consistency(result)
    _assert_final_descriptor_snapshots(result)
    np.testing.assert_array_equal(
        np.asarray(trace.candidate_rebound_mask),
        np.asarray((True, True)),
    )
    assert int(trace.proposal_parent_a) == 4
    assert int(trace.candidate_rebound_count) == 2
    assert int(trace.logical_event_count) == 5
    imprint_bits = np.asarray(result.state.candidate_output_weights[:, 0]).view(
        np.uint32
    )
    np.testing.assert_array_equal(
        imprint_bits,
        np.zeros(imprint_bits.shape, dtype=np.uint32),
    )


def test_zero_candidate_direct_replacement_trace_has_fixed_empty_shapes() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=0,
        replacement_interval=2,
        min_feature_age=0,
        learn_generator_resources=True,
        generator_resource_learning_rate=0.0,
        generator_resource_exploration=0.0,
        use_obgd=False,
    )
    state = learner.init(feature_dim=3, key=jr.key(1203))
    state = state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
        feature_generator_policy=jnp.asarray((0, 0, 0, 1, 3), dtype=jnp.int32),
        generator_resource_state=state.generator_resource_state.replace(  # type: ignore[attr-defined]
            log_weights=jnp.asarray(
                ((-1.0e9, -1.0e9, 0.0, -1.0e9),), dtype=jnp.float32
            )
        ),
        replacement_accumulator=jnp.asarray(0.5, dtype=jnp.float32),
    )
    result = learner.update(
        state,
        jnp.asarray((0.2, -0.4, 0.6), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    trace = result.curation_trace

    _assert_key_schedule(state, result)
    _assert_count_consistency(result)
    _assert_final_descriptor_snapshots(result)
    assert bool(trace.generator_policy_sampled)
    assert int(trace.generator_policy_id) == 2
    assert int(trace.proposal_destination_bank) == CURATION_DESTINATION_ACTIVE
    assert int(trace.proposal_destination_slot) == 3
    assert int(result.replaced_slot) == 3
    np.testing.assert_array_equal(
        np.asarray(trace.root_change_mask),
        np.asarray((False, False, False, True, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.cascade_refill_mask),
        np.asarray((False, False, False, False, True)),
    )
    assert int(trace.post_root_pre_cascade_op) == int(result.state.ops[3])
    assert int(trace.post_root_pre_cascade_parent_a) == int(result.state.parent_a[3])
    assert int(trace.proposal_generator_policy) == 2
    assert int(result.state.feature_generator_policy[3]) == 2
    assert int(result.state.feature_generator_policy[4]) == 2
    assert trace.candidate_refresh_mask.shape == (0,)
    assert trace.candidate_rebound_mask.shape == (0,)
    assert trace.candidate_overdepth_regeneration_mask.shape == (0,)
    assert int(trace.candidate_overdepth_regeneration_count) == 0
    assert trace.candidate_final_ops.shape == (0,)
    assert trace.candidate_final_theta.shape == (0, 2)
    assert int(trace.logical_event_count) == 2


def test_decision_audit_reconstructs_exact_promotion_algebra() -> None:
    """The public trace binds every array used by admission, not a later proxy."""

    learner = _TraceProposalLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        proposal_parent=3,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.75,
        max_depth=4,
        use_obgd=False,
        retention_slow_utility_decay=0.9,
        ancestor_utility_backup_decay=0.5,
    )
    state = _promotion_state(learner)
    result = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    trace = result.curation_trace

    for actual, expected in (
        (trace.decision_active_ops, state.ops),
        (trace.decision_active_parent_a, state.parent_a),
        (trace.decision_active_parent_b, state.parent_b),
        (trace.decision_active_depth, state.depth),
        (trace.decision_active_generator_policy, state.feature_generator_policy),
        (trace.decision_candidate_ops, state.candidate_ops),
        (trace.decision_candidate_parent_a, state.candidate_parent_a),
        (trace.decision_candidate_parent_b, state.candidate_parent_b),
        (trace.decision_candidate_depth, state.candidate_depth),
        (
            trace.decision_candidate_generator_policy,
            state.candidate_generator_policy,
        ),
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(trace.decision_active_ages), np.asarray(state.ages) + 1
    )
    np.testing.assert_array_equal(
        np.asarray(trace.decision_candidate_ages),
        np.asarray(state.candidate_ages) + 1,
    )

    direct_active = learner._retention_score(  # noqa: SLF001
        trace.decision_active_fast_utilities,
        trace.decision_active_slow_utilities,
    )
    backed_active = learner._ancestor_backed_retention_score(  # noqa: SLF001
        direct_active,
        state.ops,
        state.parent_a,
        state.parent_b,
    )
    direct_candidate = learner._retention_score(  # noqa: SLF001
        trace.decision_candidate_fast_utilities,
        trace.decision_candidate_slow_utilities,
    )
    np.testing.assert_array_equal(
        np.asarray(trace.decision_active_direct_scores), np.asarray(direct_active)
    )
    np.testing.assert_array_equal(
        np.asarray(trace.decision_active_backed_scores), np.asarray(backed_active)
    )
    np.testing.assert_array_equal(
        np.asarray(trace.decision_candidate_direct_scores),
        np.asarray(direct_candidate),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.decision_candidate_augmented_scores),
        np.asarray(direct_candidate + trace.decision_candidate_novelty_scores),
    )

    n_features = state.ops.shape[0]
    slot_indices = np.arange(n_features, dtype=np.int32)
    candidate_pa = np.asarray(state.candidate_parent_a)
    candidate_pb = np.asarray(state.candidate_parent_b)
    parent_max = np.maximum(candidate_pa, np.where(candidate_pb >= 0, candidate_pb, -1))
    topology = slot_indices[None, :] > parent_max[:, None]
    recomputed_depth = np.maximum(
        np.asarray(state.depth)[np.clip(candidate_pa, 0, n_features - 1)],
        np.where(
            candidate_pb >= 0,
            np.asarray(state.depth)[np.clip(candidate_pb, 0, n_features - 1)],
            0,
        ),
    ) + 1
    depth = np.broadcast_to((recomputed_depth <= 4)[:, None], topology.shape)
    compatible = (
        np.asarray(trace.decision_candidate_mature)[:, None]
        & np.asarray(trace.decision_active_eligible)[None, :]
        & topology
        & depth
        & np.asarray(trace.decision_candidate_headroom_compatible)
    )
    np.testing.assert_array_equal(trace.decision_candidate_recomputed_depth, recomputed_depth)
    np.testing.assert_array_equal(trace.decision_candidate_topology_compatible, topology)
    np.testing.assert_array_equal(trace.decision_candidate_depth_compatible, depth)
    np.testing.assert_array_equal(
        trace.decision_candidate_headroom_compatible,
        np.ones_like(depth, dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        trace.decision_candidate_destination_compatible, compatible
    )
    np.testing.assert_array_equal(
        trace.decision_candidate_has_destination, np.any(compatible, axis=1)
    )

    candidate_scores = np.asarray(trace.decision_candidate_ranking_scores)
    selected_candidate = int(
        np.argmax(
            np.where(
                np.asarray(trace.decision_candidate_has_destination),
                candidate_scores,
                -np.inf,
            )
        )
    )
    selected_destination = int(
        np.argmin(
            np.where(
                compatible[selected_candidate],
                np.asarray(trace.decision_active_backed_scores),
                np.inf,
            )
        )
    )
    assert int(trace.decision_selected_candidate) == selected_candidate
    assert int(trace.decision_selected_destination) == selected_destination
    expected_rhs = np.float32(0.75) * np.asarray(
        trace.decision_active_backed_scores
    )[selected_destination]
    assert np.asarray(trace.decision_margin_rhs).view(np.uint32) == np.asarray(
        expected_rhs, dtype=np.float32
    ).view(np.uint32)
    assert bool(trace.decision_margin_passed) == (
        np.asarray(trace.decision_candidate_augmented_scores)[selected_candidate]
        > expected_rhs
    )
    assert bool(trace.decision_should_promote) == bool(trace.promotion_applied)
    assert bool(trace.decision_selected_topology_ok)
    assert bool(trace.decision_selected_depth_ok)
    assert bool(trace.decision_selected_headroom_ok)
    assert bool(trace.decision_selected_can_promote)
    assert bool(trace.decision_commit_available)


def test_rejected_decision_audit_is_ephemeral_and_state_rolls_back_bit_exactly() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=2,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        candidate_novelty_admission_bonus=1.0,
        candidate_scoring_mode="energy_novelty",
        ancestor_utility_backup_decay=0.5,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(1807)).replace(  # type: ignore[attr-defined]
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    result = learner.update(
        state,
        jnp.asarray((jnp.nan, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    trace = result.curation_trace

    for actual, expected in zip(
        jax.tree_util.tree_leaves(result.state),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            actual.dtype, jax.dtypes.prng_key
        ):
            actual = jr.key_data(actual)
            expected = jr.key_data(expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not bool(trace.decision_update_available)
    assert not bool(trace.decision_commit_available)
    assert not bool(trace.should_try_replace)
    assert not bool(trace.promotion_applied)
    assert not bool(trace.has_event)
    assert trace.decision_candidate_destination_compatible.shape == (2, 4)
    np.testing.assert_array_equal(trace.decision_active_ops, state.ops)
    np.testing.assert_array_equal(trace.decision_candidate_ops, state.candidate_ops)
