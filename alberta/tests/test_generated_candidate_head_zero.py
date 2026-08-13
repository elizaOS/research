"""Focused contracts for the conditional candidate descriptor sanitizer."""

from __future__ import annotations

import dataclasses
import struct
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    OP_TANH,
    CompositionalFeatureLearner,
)
from alberta_framework.evaluation.generated_candidate_head_zero import (
    CANDIDATE_DESCRIPTOR_LEAF_PATHS,
    CANDIDATE_PROVENANCE_LEAF_PATHS,
    CANDIDATE_RESET_LEAF_PATHS,
    POST_UPDATE_PRESERVED_LEAF_PATHS,
    GeneratedCandidateHeadZeroConfig,
    apply_generated_candidate_head_zero,
    build_generated_candidate_head_zero_transaction,
    candidate_descriptor_dependency_change_mask,
    validate_generated_candidate_head_zero_transaction,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    COMPOSITIONAL_STATE_LEAF_PATHS,
    compositional_state_leaf_paths,
)

pytestmark = pytest.mark.unit


def _config(
    *,
    feature_dim: int = 2,
    active_slots: int = 6,
    candidate_slots: int = 3,
    n_tasks: int = 2,
    **changes: object,
) -> GeneratedCandidateHeadZeroConfig:
    values: dict[str, object] = {
        "feature_dim": feature_dim,
        "active_slots": active_slots,
        "candidate_slots": candidate_slots,
        "n_tasks": n_tasks,
    }
    values.update(changes)
    return GeneratedCandidateHeadZeroConfig(**values)  # type: ignore[arg-type]


def _states():
    learner = CompositionalFeatureLearner(
        n_features=6,
        n_tasks=2,
        candidate_count=3,
        replacement_interval=0,
        max_depth=4,
    )
    pre = learner.init(2, jr.key(801)).replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM, OP_GATED, OP_TANH),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 0, 2, 0, 4), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 0, 1, 1), dtype=jnp.int32),
        theta=jnp.arange(12, dtype=jnp.float32).reshape(6, 2) + 0.25,
        depth=jnp.asarray((0, 0, 1, 2, 1, 2), dtype=jnp.int32),
        output_weights=jnp.arange(12, dtype=jnp.float32).reshape(2, 6) + 1.0,
        output_bias=jnp.asarray((2.0, 3.0), dtype=jnp.float32),
        utilities=jnp.arange(6, dtype=jnp.float32) + 1.0,
        utility_contribution_trace=(
            jnp.arange(12, dtype=jnp.float32).reshape(2, 6) + 2.0
        ),
        utility_error_trace=jnp.asarray((3.0, 4.0), dtype=jnp.float32),
        utility_feature_trace=jnp.arange(6, dtype=jnp.float32) + 4.0,
        utility_feature_energy_trace=jnp.arange(6, dtype=jnp.float32) + 5.0,
        utility_signal_second_moment=jnp.arange(6, dtype=jnp.float32) + 6.0,
        feature_score_residual_trace=(
            jnp.arange(12, dtype=jnp.float32).reshape(2, 6) + 7.0
        ),
        feature_score_energy_trace=jnp.arange(6, dtype=jnp.float32) + 8.0,
        retention_slow_utilities=jnp.arange(6, dtype=jnp.float32) + 9.0,
        task_activity_ema=jnp.asarray((0.4, 0.6), dtype=jnp.float32),
        ages=jnp.arange(6, dtype=jnp.int32) + 10,
        candidate_ops=jnp.asarray((OP_PRODUCT, OP_GATED, OP_TANH), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0, 2, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1, 0, 1), dtype=jnp.int32),
        candidate_theta=jnp.asarray(
            ((0.0, 0.0), (0.25, -0.5), (0.75, -1.25)), dtype=jnp.float32
        ),
        candidate_depth=jnp.asarray((1, 2, 2), dtype=jnp.int32),
        candidate_output_weights=(
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 11.0
        ),
        candidate_utilities=jnp.arange(3, dtype=jnp.float32) + 12.0,
        candidate_utility_contribution_trace=(
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 13.0
        ),
        candidate_utility_feature_trace=jnp.arange(3, dtype=jnp.float32) + 14.0,
        candidate_utility_feature_energy_trace=(
            jnp.arange(3, dtype=jnp.float32) + 15.0
        ),
        candidate_utility_signal_second_moment=(
            jnp.arange(3, dtype=jnp.float32) + 16.0
        ),
        candidate_score_residual_trace=(
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 17.0
        ),
        candidate_score_energy_trace=jnp.arange(3, dtype=jnp.float32) + 18.0,
        candidate_retention_slow_utilities=(
            jnp.arange(3, dtype=jnp.float32) + 19.0
        ),
        candidate_active_correlation_trace=(
            jnp.arange(18, dtype=jnp.float32).reshape(3, 6) + 20.0
        ),
        candidate_ages=jnp.asarray((21, 22, 23), dtype=jnp.int32),
        candidate_selector_log_weights=jnp.arange(3, dtype=jnp.float32) + 24.0,
        candidate_selector_cumulative_loss=(
            jnp.arange(3, dtype=jnp.float32) + 25.0
        ),
        candidate_selector_action_counts=(
            jnp.arange(3, dtype=jnp.float32) + 26.0
        ),
        feature_generator_policy=jnp.asarray((0, 1, 2, 3, 1, 2), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((1, 2, 3), dtype=jnp.int32),
        replacement_accumulator=jnp.asarray(0.25, dtype=jnp.float32),
        step_count=jnp.asarray(31, dtype=jnp.int32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    generator = pre.generator_resource_state.replace(  # type: ignore[attr-defined]
        log_weights=pre.generator_resource_state.log_weights.at[0, 0].set(0.5),
        reward_ema=pre.generator_resource_state.reward_ema.at[0, 1].set(0.75),
        action_counts=pre.generator_resource_state.action_counts.at[0, 2].set(2.0),
        step_count=jnp.asarray(4, dtype=jnp.int32),
    )
    pre = pre.replace(generator_resource_state=generator)  # type: ignore[attr-defined]

    # Slot 1 is an ordinary/post-promotion-style fresh identity.  Candidate 0
    # carries exact negative zeros and is intentionally unchanged, so tests can
    # prove that the adapter neither normalizes nor resets an unmasked slot.
    negative_zero = jax.lax.bitcast_convert_type(
        jnp.asarray(0x80000000, dtype=jnp.uint32), jnp.float32
    )
    post = pre.replace(  # type: ignore[attr-defined]
        key=jr.key(802),
        output_weights=pre.output_weights.at[0, 5].set(101.0),
        output_bias=jnp.asarray((102.0, 103.0), dtype=jnp.float32),
        candidate_ops=pre.candidate_ops.at[1].set(OP_SUM),
        candidate_theta=pre.candidate_theta.at[1].set(
            jnp.asarray((negative_zero, 4.5), dtype=jnp.float32)
        ),
        candidate_output_weights=(
            pre.candidate_output_weights.at[:, 0]
            .set(negative_zero)
            .at[:, 1]
            .set(jnp.asarray((111.0, 112.0), dtype=jnp.float32))
        ),
        candidate_utilities=(
            pre.candidate_utilities.at[0].set(negative_zero).at[1].set(113.0)
        ),
        candidate_utility_contribution_trace=(
            pre.candidate_utility_contribution_trace.at[:, 1].set(
                jnp.asarray((114.0, 115.0), dtype=jnp.float32)
            )
        ),
        candidate_utility_feature_trace=(
            pre.candidate_utility_feature_trace.at[1].set(116.0)
        ),
        candidate_utility_feature_energy_trace=(
            pre.candidate_utility_feature_energy_trace.at[1].set(117.0)
        ),
        candidate_utility_signal_second_moment=(
            pre.candidate_utility_signal_second_moment.at[1].set(118.0)
        ),
        candidate_score_residual_trace=(
            pre.candidate_score_residual_trace.at[:, 1].set(
                jnp.asarray((119.0, 120.0), dtype=jnp.float32)
            )
        ),
        candidate_score_energy_trace=(
            pre.candidate_score_energy_trace.at[1].set(121.0)
        ),
        candidate_retention_slow_utilities=(
            pre.candidate_retention_slow_utilities.at[1].set(122.0)
        ),
        candidate_active_correlation_trace=(
            pre.candidate_active_correlation_trace.at[1].set(
                jnp.arange(6, dtype=jnp.float32) + 123.0
            )
        ),
        candidate_ages=pre.candidate_ages.at[1].set(124),
        candidate_selector_log_weights=(
            pre.candidate_selector_log_weights.at[1].set(125.0)
        ),
        candidate_selector_cumulative_loss=(
            pre.candidate_selector_cumulative_loss.at[1].set(126.0)
        ),
        candidate_selector_action_counts=(
            pre.candidate_selector_action_counts.at[1].set(127.0)
        ),
        candidate_generator_policy=pre.candidate_generator_policy.at[1].set(0),
        replacement_accumulator=jnp.asarray(0.75, dtype=jnp.float32),
        step_count=jnp.asarray(32, dtype=jnp.int32),
        birth_timestamp=jnp.asarray(10.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(11.0, dtype=jnp.float32),
    )
    post = post.replace(  # type: ignore[attr-defined]
        generator_resource_state=post.generator_resource_state.replace(  # type: ignore[attr-defined]
            reward_ema=post.generator_resource_state.reward_ema.at[0, 3].set(1.5),
            step_count=jnp.asarray(5, dtype=jnp.int32),
        )
    )
    return pre, post


def _leaf_bytes(value: object) -> bytes:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        array = np.asarray(jr.key_data(value))
        return b"key:" + array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()
    if isinstance(value, jax.Array):
        array = np.asarray(value)
        return array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()
    if type(value) is float:
        return b"float:" + struct.pack(">d", value)
    raise TypeError(type(value))


def _tree_bit_exact(first: object, second: object) -> bool:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    return first_tree == second_tree and all(
        _leaf_bytes(a) == _leaf_bytes(b)
        for a, b in zip(first_leaves, second_leaves, strict=True)
    )


def _path_value(state: object, path: str) -> Any:
    value = state
    for name in path.split("."):
        value = getattr(value, name)
    return value


def _assert_positive_zero(value: jax.Array) -> None:
    np.testing.assert_array_equal(np.asarray(value).view(np.uint32), 0)


def test_leaf_partition_and_authority_disclosures_are_exact() -> None:
    pre, _ = _states()
    groups = (
        CANDIDATE_DESCRIPTOR_LEAF_PATHS,
        CANDIDATE_PROVENANCE_LEAF_PATHS,
        CANDIDATE_RESET_LEAF_PATHS,
        POST_UPDATE_PRESERVED_LEAF_PATHS,
    )
    assert compositional_state_leaf_paths(pre) == COMPOSITIONAL_STATE_LEAF_PATHS
    assert set().union(*groups) == set(COMPOSITIONAL_STATE_LEAF_PATHS)
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            assert first.isdisjoint(second)

    config = _config()
    assert config.development_only
    assert config.complete_descriptor_exact_bits
    assert config.candidate_theta_learning_disabled_required
    assert config.external_birth_event_ledger_required
    assert config.gradient_descriptor_drift_can_trigger_mask
    assert config.descriptor_collision_can_hide_birth
    assert config.host_audit_not_jittable
    assert not config.post_update_origin_authenticated
    assert not config.event_identity_authenticated
    assert not config.lifecycle_prerequisite_complete
    assert not config.fresh_rng_epoch_claimed
    assert not config.future_target_isolation_claimed
    assert not config.structural_deletion_claimed
    assert not config.acquisition_claimed
    assert not config.outcome_claimed
    assert not config.execution_authorized
    assert not config.runner_authorized
    assert not config.artifact_writes_authorized
    assert not config.evidence_authorized
    assert not config.scientific_promotion_allowed

    with pytest.raises(ValueError):
        _config(candidate_theta_learning_disabled_required=False)
    with pytest.raises(ValueError):
        _config(external_birth_event_ledger_required=False)
    with pytest.raises(ValueError):
        _config(gradient_descriptor_drift_can_trigger_mask=False)
    with pytest.raises(ValueError):
        _config(descriptor_collision_can_hide_birth=False)
    with pytest.raises(ValueError):
        _config(event_identity_authenticated=True)


@pytest.mark.parametrize(
    "field, replacement",
    (
        ("candidate_ops", OP_SUM),
        ("candidate_parent_a", 1),
        ("candidate_parent_b", 0),
        ("candidate_depth", 2),
    ),
)
def test_every_integer_descriptor_word_changes_identity(
    field: str,
    replacement: int,
) -> None:
    pre, _ = _states()
    current = getattr(pre, field)
    post = pre.replace(**{field: current.at[0].set(replacement)})  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        candidate_descriptor_dependency_change_mask(pre, post),
        [True, False, False],
    )


@pytest.mark.parametrize("theta_column", (0, 1))
def test_theta_signed_zero_is_an_exact_descriptor_change(theta_column: int) -> None:
    pre, _ = _states()
    pre = pre.replace(  # type: ignore[attr-defined]
        candidate_theta=pre.candidate_theta.at[2, theta_column].set(
            jnp.asarray(0.0, dtype=jnp.float32)
        )
    )
    negative_zero = jax.lax.bitcast_convert_type(
        jnp.asarray(0x80000000, dtype=jnp.uint32), jnp.float32
    )
    post = pre.replace(  # type: ignore[attr-defined]
        candidate_theta=pre.candidate_theta.at[2, theta_column].set(negative_zero)
    )

    assert float(pre.candidate_theta[2, theta_column]) == float(
        post.candidate_theta[2, theta_column]
    )
    np.testing.assert_array_equal(
        candidate_descriptor_dependency_change_mask(pre, post),
        [False, False, True],
    )


def test_commit_zeros_all_changed_local_carry_and_preserves_post_state() -> None:
    pre, post = _states()
    result = apply_generated_candidate_head_zero(
        pre,
        post,
        jnp.asarray(True),
        config=_config(),
    )
    diagnostics = result.diagnostics

    assert bool(diagnostics.descriptor_sanitizer_valid)
    assert bool(diagnostics.descriptor_sanitizer_committed)
    assert not bool(diagnostics.post_update_origin_authenticated)
    assert not bool(diagnostics.event_identity_authenticated)
    assert bool(diagnostics.external_birth_event_ledger_required)
    assert bool(diagnostics.gradient_descriptor_drift_can_trigger_mask)
    assert bool(diagnostics.descriptor_collision_can_hide_birth)
    assert not bool(diagnostics.lifecycle_prerequisite_complete)
    np.testing.assert_array_equal(
        diagnostics.candidate_reset_mask, [False, True, False]
    )
    assert int(diagnostics.candidate_reset_count) == 1

    # The complete new descriptor and its newly supplied generator provenance
    # survive without normalization, including theta's negative-zero bit.
    for path in CANDIDATE_DESCRIPTOR_LEAF_PATHS | CANDIDATE_PROVENANCE_LEAF_PATHS:
        assert _leaf_bytes(_path_value(result.state, path)) == _leaf_bytes(
            _path_value(post, path)
        )
    assert int(
        jax.lax.bitcast_convert_type(result.state.candidate_theta[1, 0], jnp.uint32)
    ) == 0x80000000

    changed = 1
    _assert_positive_zero(result.state.candidate_output_weights[:, changed])
    _assert_positive_zero(result.state.candidate_utilities[changed : changed + 1])
    _assert_positive_zero(
        result.state.candidate_utility_contribution_trace[:, changed]
    )
    _assert_positive_zero(result.state.candidate_utility_feature_trace[changed : changed + 1])
    _assert_positive_zero(
        result.state.candidate_utility_feature_energy_trace[changed : changed + 1]
    )
    _assert_positive_zero(
        result.state.candidate_utility_signal_second_moment[changed : changed + 1]
    )
    _assert_positive_zero(result.state.candidate_score_residual_trace[:, changed])
    _assert_positive_zero(result.state.candidate_score_energy_trace[changed : changed + 1])
    _assert_positive_zero(
        result.state.candidate_retention_slow_utilities[changed : changed + 1]
    )
    _assert_positive_zero(result.state.candidate_active_correlation_trace[changed])
    assert int(result.state.candidate_ages[changed]) == 0
    _assert_positive_zero(result.state.candidate_selector_log_weights[changed : changed + 1])
    _assert_positive_zero(
        result.state.candidate_selector_cumulative_loss[changed : changed + 1]
    )
    _assert_positive_zero(
        result.state.candidate_selector_action_counts[changed : changed + 1]
    )

    for path in POST_UPDATE_PRESERVED_LEAF_PATHS:
        assert _leaf_bytes(_path_value(result.state, path)) == _leaf_bytes(
            _path_value(post, path)
        )
    # Unchanged candidate 0's negative zeros remain negative zeros.
    assert int(
        jax.lax.bitcast_convert_type(result.state.candidate_output_weights[0, 0], jnp.uint32)
    ) == 0x80000000
    assert int(
        jax.lax.bitcast_convert_type(result.state.candidate_utilities[0], jnp.uint32)
    ) == 0x80000000


def test_promotion_cascade_parent_change_resets_locally_unchanged_candidate() -> None:
    pre, _ = _states()
    # Candidate 1 depends on active slot 3.  Slot 3's local descriptor stays
    # fixed, but its parent slot 2 is replaced as a promotion could replace an
    # active identity.  The meaning change must propagate 2 -> 3 -> candidate.
    pre = pre.replace(  # type: ignore[attr-defined]
        candidate_parent_a=pre.candidate_parent_a.at[1].set(3),
        candidate_depth=pre.candidate_depth.at[1].set(3),
    )
    post = pre.replace(  # type: ignore[attr-defined]
        ops=pre.ops.at[2].set(OP_SUM),
        candidate_output_weights=pre.candidate_output_weights.at[:, 1].set(
            jnp.asarray((201.0, 202.0), dtype=jnp.float32)
        ),
        candidate_utilities=pre.candidate_utilities.at[1].set(203.0),
        candidate_active_correlation_trace=(
            pre.candidate_active_correlation_trace.at[1].set(
                jnp.arange(6, dtype=jnp.float32) + 204.0
            )
        ),
    )
    result = apply_generated_candidate_head_zero(
        pre,
        post,
        jnp.asarray(True),
        config=_config(),
    )

    np.testing.assert_array_equal(
        result.diagnostics.active_local_descriptor_change_mask,
        [False, False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        result.diagnostics.active_propagated_descriptor_change_mask,
        [False, False, True, True, False, False],
    )
    np.testing.assert_array_equal(
        result.diagnostics.candidate_local_descriptor_change_mask,
        [False, False, False],
    )
    np.testing.assert_array_equal(
        result.diagnostics.candidate_active_parent_dependency_change_mask,
        [False, True, False],
    )
    np.testing.assert_array_equal(
        result.diagnostics.candidate_reset_mask,
        [False, True, False],
    )
    assert int(result.diagnostics.active_local_descriptor_change_count) == 1
    assert int(result.diagnostics.active_propagated_descriptor_change_count) == 2
    assert int(result.diagnostics.candidate_local_descriptor_change_count) == 0
    assert int(
        result.diagnostics.candidate_active_parent_dependency_change_count
    ) == 1
    assert bool(result.diagnostics.descriptor_sanitizer_committed)
    _assert_positive_zero(result.state.candidate_output_weights[:, 1])
    _assert_positive_zero(result.state.candidate_utilities[1:2])
    _assert_positive_zero(result.state.candidate_active_correlation_trace[1])
    np.testing.assert_array_equal(result.state.candidate_ops, post.candidate_ops)
    np.testing.assert_array_equal(
        result.state.candidate_generator_policy, post.candidate_generator_policy
    )


def test_sham_and_no_change_are_bit_exact_but_never_causal_commits() -> None:
    pre, post = _states()
    sham = build_generated_candidate_head_zero_transaction(
        pre,
        post,
        jnp.asarray(False),
        config=_config(),
    )
    assert _tree_bit_exact(sham.result.state, post)
    assert bool(sham.result.diagnostics.sham_noop)
    assert not bool(sham.result.diagnostics.descriptor_sanitizer_committed)
    sham_validation = validate_generated_candidate_head_zero_transaction(
        pre,
        post,
        jnp.asarray(False),
        sham,
        config=_config(),
    )
    assert sham_validation.valid
    assert sham_validation.sham_noop_valid
    assert not sham_validation.descriptor_sanitizer_commit_valid
    assert not sham_validation.causal_refresh_commit_valid

    no_change = build_generated_candidate_head_zero_transaction(
        pre,
        pre,
        jnp.asarray(True),
        config=_config(),
    )
    assert _tree_bit_exact(no_change.result.state, pre)
    assert bool(no_change.result.diagnostics.no_change_noop)
    assert bool(no_change.result.diagnostics.rolled_back)
    assert not bool(no_change.result.diagnostics.descriptor_sanitizer_valid)
    assert not bool(no_change.result.diagnostics.descriptor_sanitizer_committed)
    no_change_validation = validate_generated_candidate_head_zero_transaction(
        pre,
        pre,
        jnp.asarray(True),
        no_change,
        config=_config(),
    )
    assert no_change_validation.valid
    assert no_change_validation.no_change_noop_valid
    assert not no_change_validation.descriptor_sanitizer_commit_valid
    assert not no_change_validation.causal_refresh_commit_valid


def test_committed_transform_is_idempotent_for_the_same_pre_post_pair() -> None:
    pre, post = _states()
    first = apply_generated_candidate_head_zero(
        pre, post, jnp.asarray(True), config=_config()
    )
    second = apply_generated_candidate_head_zero(
        pre, first.state, jnp.asarray(True), config=_config()
    )

    assert bool(second.diagnostics.descriptor_sanitizer_committed)
    assert _tree_bit_exact(first.state, second.state)


def test_kernel_is_bit_exact_between_eager_and_jit_with_canonical_timing() -> None:
    pre, post = _states()
    config = _config()
    eager = apply_generated_candidate_head_zero(
        pre, post, jnp.asarray(True), config=config
    )
    compiled = jax.jit(
        lambda before, after, commit: apply_generated_candidate_head_zero(
            before, after, commit, config=config
        )
    )(pre, post, jnp.asarray(True))

    assert _tree_bit_exact(eager, compiled)


def test_actual_ordinary_refresh_is_detected_and_zeroed() -> None:
    learner = CompositionalFeatureLearner(
        n_features=6,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=2,
        min_feature_age=100,
        candidate_min_age=0,
        promotion_margin=1000.0,
        learn_generator_resources=True,
        generator_resource_learning_rate=0.0,
        generator_resource_exploration=0.0,
    )
    pre = learner.init(feature_dim=3, key=jr.key(811)).replace(  # type: ignore[attr-defined]
        generator_resource_state=learner.init(
            feature_dim=3, key=jr.key(811)
        ).generator_resource_state.replace(  # type: ignore[attr-defined]
            log_weights=jnp.asarray(((-10.0, -10.0, 10.0, -10.0),), dtype=jnp.float32)
        ),
        candidate_utilities=jnp.asarray((0.0,), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        replacement_accumulator=jnp.asarray(0.5, dtype=jnp.float32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    update = learner.update(
        pre,
        jnp.asarray((0.2, -0.4, 0.6), dtype=jnp.float32),
        jnp.asarray((1.0,), dtype=jnp.float32),
    )
    assert int(update.promoted_candidate) == -1
    np.testing.assert_array_equal(
        candidate_descriptor_dependency_change_mask(pre, update.state), [True]
    )

    transaction = build_generated_candidate_head_zero_transaction(
        pre,
        update.state,
        jnp.asarray(True),
        config=_config(feature_dim=3, active_slots=6, candidate_slots=1, n_tasks=1),
    )
    assert bool(transaction.result.diagnostics.descriptor_sanitizer_committed)
    np.testing.assert_array_equal(
        transaction.result.diagnostics.candidate_local_descriptor_change_mask,
        [True],
    )
    assert not bool(transaction.result.diagnostics.post_update_origin_authenticated)
    _assert_positive_zero(transaction.result.state.candidate_output_weights[:, 0])
    assert int(transaction.result.state.candidate_generator_policy[0]) == int(
        update.state.candidate_generator_policy[0]
    )


def test_actual_promotion_refresh_preserves_promoted_active_state_and_zeros_candidate() -> None:
    learner = CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        utility_decay=0.99,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        use_obgd=False,
    )
    pre = learner.init(feature_dim=2, key=jr.key(812)).replace(  # type: ignore[attr-defined]
        ops=jnp.asarray((OP_RAW, OP_RAW, OP_PRODUCT), dtype=jnp.int32),
        parent_a=jnp.asarray((0, 1, 0), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1), dtype=jnp.int32),
        depth=jnp.asarray((0, 0, 1), dtype=jnp.int32),
        utilities=jnp.asarray((0.0, 0.0, 0.0), dtype=jnp.float32),
        candidate_ops=jnp.asarray((OP_TANH,), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0,), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1,), dtype=jnp.int32),
        candidate_theta=jnp.asarray(((0.5, -0.75),), dtype=jnp.float32),
        candidate_depth=jnp.asarray((1,), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((1.0,), dtype=jnp.float32),
        candidate_ages=jnp.asarray((5,), dtype=jnp.int32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    update = learner.update(
        pre,
        jnp.asarray((1.0, -1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert int(update.promoted_candidate) == 0
    assert int(update.replaced_slot) == 2
    np.testing.assert_array_equal(
        candidate_descriptor_dependency_change_mask(pre, update.state), [True]
    )

    transaction = build_generated_candidate_head_zero_transaction(
        pre,
        update.state,
        jnp.asarray(True),
        config=_config(feature_dim=2, active_slots=3, candidate_slots=1, n_tasks=1),
    )
    assert bool(transaction.result.diagnostics.descriptor_sanitizer_committed)
    np.testing.assert_array_equal(
        transaction.result.diagnostics.candidate_local_descriptor_change_mask,
        [True],
    )
    assert int(
        transaction.result.diagnostics.active_local_descriptor_change_count
    ) >= 1
    assert not bool(transaction.result.diagnostics.post_update_origin_authenticated)
    _assert_positive_zero(transaction.result.state.candidate_output_weights[:, 0])
    for path in POST_UPDATE_PRESERVED_LEAF_PATHS:
        assert _leaf_bytes(_path_value(transaction.result.state, path)) == _leaf_bytes(
            _path_value(update.state, path)
        )


def test_strict_validator_rejects_state_diagnostics_audit_and_input_bit_attacks() -> None:
    pre, post = _states()
    config = _config()
    transaction = build_generated_candidate_head_zero_transaction(
        pre, post, jnp.asarray(True), config=config
    )
    valid = validate_generated_candidate_head_zero_transaction(
        pre, post, jnp.asarray(True), transaction, config=config
    )
    assert valid.valid and valid.descriptor_sanitizer_commit_valid
    assert not valid.causal_refresh_commit_valid
    assert transaction.audit.candidate_reset_count == 1
    assert not transaction.audit.post_update_origin_authenticated
    assert not transaction.audit.event_identity_authenticated
    assert transaction.audit.candidate_theta_learning_disabled_required
    assert transaction.audit.external_birth_event_ledger_required
    assert transaction.audit.gradient_descriptor_drift_can_trigger_mask
    assert transaction.audit.descriptor_collision_can_hide_birth
    assert not transaction.audit.lifecycle_prerequisite_complete
    assert not transaction.audit.runner_authorized
    assert not transaction.audit.evidence_authorized

    forged_state = transaction.result.state.replace(  # type: ignore[attr-defined]
        output_bias=transaction.result.state.output_bias.at[0].add(1.0)
    )
    forged_result = transaction.result.replace(state=forged_state)  # type: ignore[attr-defined]
    forged_transaction = dataclasses.replace(transaction, result=forged_result)
    assert not validate_generated_candidate_head_zero_transaction(
        pre, post, jnp.asarray(True), forged_transaction, config=config
    ).valid

    forged_unmasked_candidate = transaction.result.state.replace(  # type: ignore[attr-defined]
        candidate_output_weights=(
            transaction.result.state.candidate_output_weights.at[0, 0].add(1.0)
        )
    )
    forged_result = transaction.result.replace(  # type: ignore[attr-defined]
        state=forged_unmasked_candidate
    )
    assert not validate_generated_candidate_head_zero_transaction(
        pre,
        post,
        jnp.asarray(True),
        dataclasses.replace(transaction, result=forged_result),
        config=config,
    ).valid

    forged_diagnostics = transaction.result.diagnostics.replace(  # type: ignore[attr-defined]
        post_update_origin_authenticated=jnp.asarray(True)
    )
    forged_transaction = dataclasses.replace(
        transaction,
        result=transaction.result.replace(diagnostics=forged_diagnostics),  # type: ignore[attr-defined]
    )
    assert not validate_generated_candidate_head_zero_transaction(
        pre, post, jnp.asarray(True), forged_transaction, config=config
    ).valid

    forged_audit = dataclasses.replace(transaction.audit, returned_state_bit_sha256="0" * 64)
    assert not validate_generated_candidate_head_zero_transaction(
        pre,
        post,
        jnp.asarray(True),
        dataclasses.replace(transaction, audit=forged_audit),
        config=config,
    ).valid

    altered_pre = pre.replace(  # type: ignore[attr-defined]
        output_bias=pre.output_bias.at[0].add(0.25)
    )
    assert not validate_generated_candidate_head_zero_transaction(
        altered_pre, post, jnp.asarray(True), transaction, config=config
    ).valid
    altered_post = post.replace(key=jr.key(899))  # type: ignore[attr-defined]
    assert not validate_generated_candidate_head_zero_transaction(
        pre, altered_post, jnp.asarray(True), transaction, config=config
    ).valid


def test_resource_signature_drift_rolls_back_and_cannot_be_causal() -> None:
    pre, post = _states()
    generator = post.generator_resource_state
    drifted = post.replace(  # type: ignore[attr-defined]
        feature_generator_policy=jnp.minimum(post.feature_generator_policy, 2),
        candidate_generator_policy=jnp.minimum(post.candidate_generator_policy, 2),
        generator_resource_state=generator.replace(  # type: ignore[attr-defined]
            log_weights=generator.log_weights[:, :3],
            reward_ema=generator.reward_ema[:, :3],
            action_counts=generator.action_counts[:, :3],
        ),
    )
    transaction = build_generated_candidate_head_zero_transaction(
        pre,
        drifted,
        jnp.asarray(True),
        config=_config(),
    )

    assert not bool(transaction.result.diagnostics.resource_shapes_match)
    assert bool(transaction.result.diagnostics.rolled_back)
    assert not bool(transaction.result.diagnostics.descriptor_sanitizer_committed)
    assert _tree_bit_exact(transaction.result.state, drifted)
    assert (
        transaction.audit.pre_resource_signature_sha256
        != transaction.audit.post_update_resource_signature_sha256
    )
    validation = validate_generated_candidate_head_zero_transaction(
        pre,
        drifted,
        jnp.asarray(True),
        transaction,
        config=_config(),
    )
    assert validation.valid
    assert not validation.descriptor_sanitizer_commit_valid
    assert not validation.causal_refresh_commit_valid
