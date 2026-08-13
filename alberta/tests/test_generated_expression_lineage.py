"""Focused contracts for exact expanded-expression lineage compilation."""

from __future__ import annotations

import dataclasses
import struct

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
    CompositionalFeatureState,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    GeneratedClassScrubConfig,
    persistent_compositional_state_nbytes,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    GeneratedExpression,
    expression_digest,
    gate_expression,
    product_expression,
    raw_expression,
    sum_expression,
    tanh_expression,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    EXPANDED_EXPRESSION_LINEAGE_SCHEMA,
    EXPANDED_EXPRESSION_LINEAGE_STATUS,
    ExpandedExpressionLineageConfig,
    compile_expanded_expression_lineage_masks,
    validate_post_scrub_expanded_expression_absence,
)

pytestmark = pytest.mark.unit


def _tree_bit_records(value: object) -> tuple[tuple[str, str, tuple[int, ...], bytes], ...]:
    records: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        if isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jr.key_data(leaf))
            records.append((str(path), str(leaf.dtype), array.shape, array.tobytes()))
        elif isinstance(leaf, jax.Array):
            array = np.asarray(leaf)
            records.append((str(path), array.dtype.str, array.shape, array.tobytes()))
        elif type(leaf) is float:
            records.append((str(path), "python-float", (), struct.pack(">d", leaf)))
        else:
            raise TypeError(type(leaf))
    return tuple(records)


def _state_target_and_config() -> tuple[
    CompositionalFeatureState,
    GeneratedExpression,
    ExpandedExpressionLineageConfig,
]:
    learner = CompositionalFeatureLearner(
        n_features=8,
        n_tasks=1,
        candidate_count=4,
        replacement_interval=0,
        max_depth=3,
    )
    state = learner.init(3, jr.key(402))
    theta = (
        jnp.zeros((8, 2), dtype=jnp.float32).at[6].set(jnp.asarray((-1.0, 2.0), dtype=jnp.float32))
    )
    state = state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (OP_RAW, OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM, OP_GATED, OP_TANH, OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3, 1, 1, 4), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2, 2, 0, 5), dtype=jnp.int32),
        theta=theta,
        depth=jnp.asarray((0, 0, 0, 1, 2, 1, 1, 3), dtype=jnp.int32),
        candidate_ops=jnp.asarray((OP_GATED, OP_PRODUCT, OP_RAW, OP_SUM), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((4, 6, 0, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((0, 2, -1, 5), dtype=jnp.int32),
        candidate_theta=jnp.zeros((4, 2), dtype=jnp.float32),
        candidate_depth=jnp.asarray((3, 2, 0, 3), dtype=jnp.int32),
        feature_generator_policy=jnp.asarray((0, 1, 2, 3, 0, 1, 2, 3), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((3, 2, 1, 0), dtype=jnp.int32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    target = product_expression(raw_expression(0), raw_expression(1))
    config = ExpandedExpressionLineageConfig(
        feature_dim=3,
        active_slots=8,
        candidate_slots=4,
        n_tasks=1,
        generator_contexts=1,
        generator_policy_count=4,
    )
    return state, target, config


def _scrub(
    state: CompositionalFeatureState,
    target: GeneratedExpression,
    config: ExpandedExpressionLineageConfig,
):
    plan = compile_expanded_expression_lineage_masks(state, target, config=config)
    scrub_config = _scrub_config(config)
    result = scrub_compositional_feature_state(
        state,
        plan.active_mask,
        plan.candidate_mask,
        jnp.asarray(True),
        config=scrub_config,
    )
    assert bool(result.diagnostics.committed)
    return plan, result.state


def _scrub_config(
    config: ExpandedExpressionLineageConfig,
) -> GeneratedClassScrubConfig:
    return GeneratedClassScrubConfig(
        feature_dim=config.feature_dim,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        n_tasks=config.n_tasks,
        filler_op=config.filler_op,
        filler_parent_a=config.filler_parent_a,
        filler_parent_b=config.filler_parent_b,
    )


def test_compiler_returns_exact_transitive_masks_and_raw_audit() -> None:
    state, target, config = _state_target_and_config()
    before = _tree_bit_records(state)
    plan = compile_expanded_expression_lineage_masks(state, target, config=config)
    replay = compile_expanded_expression_lineage_masks(state, target, config=config)
    assert _tree_bit_records(state) == before

    np.testing.assert_array_equal(
        plan.active_mask,
        np.asarray((False, False, False, True, True, False, False, True)),
    )
    np.testing.assert_array_equal(
        plan.candidate_mask,
        np.asarray((True, False, False, True)),
    )
    assert plan.active_mask.shape == (8,) and plan.active_mask.dtype == jnp.bool_
    assert plan.candidate_mask.shape == (4,) and plan.candidate_mask.dtype == jnp.bool_

    audit = plan.audit
    assert audit.active_root_count == 8
    assert audit.candidate_root_count == 4
    assert audit.active_exact_target_root_count == 1
    assert audit.active_roots_containing_target == 3
    assert audit.candidate_roots_containing_target == 2
    assert audit.active_target_subtree_occurrences == 3
    assert audit.candidate_target_subtree_occurrences == 2
    assert audit.active_descendant_dependency_roots == 2
    assert audit.candidate_active_dependency_roots == 2
    assert audit.active_mask_count == 3
    assert audit.candidate_mask_count == 2
    assert audit.pre_target_present
    assert audit.nonempty_causal_plan
    assert audit.active_expanded_node_visits == 26
    assert audit.candidate_expanded_node_visits == 22
    assert audit.logical_subtree_identity_comparisons == 48
    assert audit.active_parent_edges_audited == 10
    assert audit.candidate_parent_edges_audited == 6
    assert audit.mask_persistent_array_nbytes == 12
    assert audit.state_persistent_array_nbytes == persistent_compositional_state_nbytes(state)
    assert audit.state_persistent_array_nbytes == audit.expected_state_persistent_array_nbytes
    assert len(audit.target_expression_sha256) == 64
    assert len(audit.filler_expression_sha256) == 64
    assert len(audit.pre_state_bit_sha256) == 64
    assert len(audit.persistent_resource_signature_sha256) == 64
    assert len(audit.active_root_bank_sha256) == 64
    assert len(audit.candidate_root_bank_sha256) == 64
    assert len(audit.plan_sha256) == 64
    assert audit == replay.audit
    np.testing.assert_array_equal(plan.active_mask, replay.active_mask)
    np.testing.assert_array_equal(plan.candidate_mask, replay.candidate_mask)


def test_public_canonical_identity_handles_commutative_and_tanh_pair_swaps() -> None:
    state, _, config = _state_target_and_config()
    commutative_target = product_expression(raw_expression(1), raw_expression(0))
    commutative = compile_expanded_expression_lineage_masks(
        state,
        commutative_target,
        config=config,
    )
    np.testing.assert_array_equal(
        commutative.active_mask,
        np.asarray((False, False, False, True, True, False, False, True)),
    )

    # Slot 6 stores the reversed child/coefficient pairs (x1, -1), (x0, 2).
    # The public tanh constructor canonicalizes those pairs together.
    tanh_target = tanh_expression(
        raw_expression(0),
        raw_expression(1),
        theta0=2.0,
        theta1=-1.0,
    )
    tanh_plan = compile_expanded_expression_lineage_masks(state, tanh_target, config=config)
    np.testing.assert_array_equal(
        tanh_plan.active_mask,
        np.asarray((False, False, False, False, False, False, True, False)),
    )
    np.testing.assert_array_equal(
        tanh_plan.candidate_mask,
        np.asarray((False, True, False, False)),
    )
    assert tanh_plan.audit.active_exact_target_root_count == 1
    assert tanh_plan.audit.candidate_active_dependency_roots == 1


def test_ordered_gate_swap_remains_distinct_in_compiler_masks() -> None:
    state, _, config = _state_target_and_config()
    x1 = raw_expression(1)
    x2 = raw_expression(2)
    stored_gate = gate_expression(x1, x2)
    swapped_gate = gate_expression(x2, x1)
    assert expression_digest(stored_gate) != expression_digest(swapped_gate)

    stored = compile_expanded_expression_lineage_masks(state, stored_gate, config=config)
    np.testing.assert_array_equal(
        stored.active_mask,
        np.asarray((False, False, False, False, False, True, False, True)),
    )
    np.testing.assert_array_equal(
        stored.candidate_mask,
        np.asarray((False, False, False, True)),
    )
    swapped = compile_expanded_expression_lineage_masks(state, swapped_gate, config=config)
    assert not swapped.audit.pre_target_present
    assert not bool(jnp.any(swapped.active_mask))
    assert not bool(jnp.any(swapped.candidate_mask))


def test_tanh_signed_zero_and_float32_coefficient_bits_remain_significant() -> None:
    x0 = raw_expression(0)
    x1 = raw_expression(1)
    positive_zero = tanh_expression(x0, x1, theta0=0.0, theta1=1.0)
    negative_zero = tanh_expression(x0, x1, theta0=-0.0, theta1=1.0)
    rounded_same = tanh_expression(x0, x1, theta0=1e-50, theta1=1.0)
    next_float32 = float(np.nextafter(np.float32(1.0), np.float32(2.0), dtype=np.float32))
    next_coefficient = tanh_expression(x0, x1, theta0=0.0, theta1=next_float32)

    assert expression_digest(positive_zero) != expression_digest(negative_zero)
    assert expression_digest(positive_zero) == expression_digest(rounded_same)
    assert expression_digest(positive_zero) != expression_digest(next_coefficient)


def test_clipping_sensitive_associative_grouping_is_not_flattened() -> None:
    state, _, config = _state_target_and_config()
    grouped = state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (OP_RAW, OP_RAW, OP_RAW, OP_SUM, OP_SUM, OP_SUM, OP_SUM, OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3, 1, 0, 4), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2, 2, 5, 5), dtype=jnp.int32),
        theta=jnp.zeros((8, 2), dtype=jnp.float32),
        depth=jnp.asarray((0, 0, 0, 1, 2, 1, 2, 3), dtype=jnp.int32),
        candidate_depth=jnp.asarray((3, 3, 0, 3), dtype=jnp.int32),
    )
    x0 = raw_expression(0)
    x1 = raw_expression(1)
    x2 = raw_expression(2)
    left_grouped = sum_expression(sum_expression(x0, x1), x2)
    right_grouped = sum_expression(x0, sum_expression(x1, x2))
    assert expression_digest(left_grouped) != expression_digest(right_grouped)

    plan = compile_expanded_expression_lineage_masks(grouped, left_grouped, config=config)
    np.testing.assert_array_equal(
        plan.active_mask,
        np.asarray((False, False, False, False, True, False, False, True)),
    )
    np.testing.assert_array_equal(
        plan.candidate_mask,
        np.asarray((True, False, False, True)),
    )
    right = compile_expanded_expression_lineage_masks(grouped, right_grouped, config=config)
    np.testing.assert_array_equal(
        right.active_mask,
        np.asarray((False, False, False, False, False, False, True, False)),
    )
    np.testing.assert_array_equal(
        right.candidate_mask,
        np.asarray((False, True, False, False)),
    )


def test_canonical_tanh_duplicate_with_different_local_words_is_also_masked() -> None:
    state, _, config = _state_target_and_config()
    duplicate = state.replace(  # type: ignore[attr-defined]
        ops=state.ops.at[5].set(OP_TANH),
        parent_a=state.parent_a.at[5].set(0),
        parent_b=state.parent_b.at[5].set(1),
        theta=state.theta.at[5].set(jnp.asarray((2.0, -1.0), dtype=jnp.float32)),
    )
    target = tanh_expression(
        raw_expression(0),
        raw_expression(1),
        theta0=2.0,
        theta1=-1.0,
    )
    plan = compile_expanded_expression_lineage_masks(duplicate, target, config=config)

    # Slots 5 and 6 use different local parent/theta words but canonicalize to
    # the same expanded tanh expression. Slot 7 descends from slot 5.
    np.testing.assert_array_equal(
        plan.active_mask,
        np.asarray((False, False, False, False, False, True, True, True)),
    )
    np.testing.assert_array_equal(
        plan.candidate_mask,
        np.asarray((False, True, False, True)),
    )
    assert plan.audit.active_exact_target_root_count == 2

    attacked_mask = plan.active_mask.at[6].set(False)
    attacked_plan = dataclasses.replace(plan, active_mask=attacked_mask)
    locally_scrubbed = scrub_compositional_feature_state(
        duplicate,
        attacked_mask,
        plan.candidate_mask,
        jnp.asarray(True),
        config=_scrub_config(config),
    )
    assert bool(locally_scrubbed.diagnostics.committed)

    validation = validate_post_scrub_expanded_expression_absence(
        duplicate,
        locally_scrubbed.state,
        target,
        attacked_plan,
        config=config,
        scrub_config=_scrub_config(config),
    )
    assert not validation.plan_matches_canonical
    assert validation.active_roots_containing_target_after == 1
    assert not validation.target_absent_from_all_expanded_trees
    assert not validation.valid


def test_candidate_exact_target_is_masked_without_an_active_occurrence() -> None:
    state, _, config = _state_target_and_config()
    candidate_only = state.replace(  # type: ignore[attr-defined]
        candidate_ops=state.candidate_ops.at[2].set(OP_GATED),
        candidate_parent_a=state.candidate_parent_a.at[2].set(2),
        candidate_parent_b=state.candidate_parent_b.at[2].set(0),
        candidate_depth=state.candidate_depth.at[2].set(1),
    )
    target = GeneratedExpression(
        op="gate",
        left=raw_expression(2),
        right=raw_expression(0),
    )
    plan = compile_expanded_expression_lineage_masks(candidate_only, target, config=config)

    np.testing.assert_array_equal(plan.active_mask, np.zeros((8,), dtype=np.bool_))
    np.testing.assert_array_equal(
        plan.candidate_mask,
        np.asarray((False, False, True, False)),
    )
    assert plan.audit.active_roots_containing_target == 0
    assert plan.audit.candidate_exact_target_root_count == 1


def test_post_scrub_validator_proves_expanded_target_absence() -> None:
    state, target, config = _state_target_and_config()
    plan, scrubbed = _scrub(state, target, config)

    validation = validate_post_scrub_expanded_expression_absence(
        state,
        scrubbed,
        target,
        plan,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert validation.pre_target_present
    assert validation.nonempty_causal_plan
    assert validation.plan_matches_canonical
    assert validation.recomputed_commit_succeeded
    assert validation.transaction_matches_recomputed_commit
    assert validation.static_resources_preserved
    assert validation.persistent_array_nbytes_preserved
    assert validation.active_roots_containing_target_after == 0
    assert validation.candidate_roots_containing_target_after == 0
    assert validation.active_target_subtree_occurrences_after == 0
    assert validation.candidate_target_subtree_occurrences_after == 0
    assert validation.target_absent_from_all_expanded_trees
    assert validation.valid
    assert not validation.behavioral_memory_erasure_claimed
    assert not validation.execution_authorized
    assert not validation.evidence_authorized
    assert not validation.scientific_promotion_allowed
    assert len(validation.validation_sha256) == 64
    np.testing.assert_array_equal(jr.key_data(scrubbed.key), jr.key_data(state.key))
    for name in ("log_weights", "reward_ema", "action_counts", "step_count"):
        np.testing.assert_array_equal(
            getattr(scrubbed.generator_resource_state, name),
            getattr(state.generator_resource_state, name),
        )
    np.testing.assert_array_equal(scrubbed.feature_generator_policy[plan.active_mask], 0)
    np.testing.assert_array_equal(
        scrubbed.feature_generator_policy[~plan.active_mask],
        state.feature_generator_policy[~plan.active_mask],
    )
    np.testing.assert_array_equal(
        scrubbed.candidate_generator_policy[plan.candidate_mask],
        0,
    )
    np.testing.assert_array_equal(
        scrubbed.candidate_generator_policy[~plan.candidate_mask],
        state.candidate_generator_policy[~plan.candidate_mask],
    )


@pytest.mark.parametrize(
    ("mask_name", "index", "replacement"),
    (
        ("active_mask", 3, False),
        ("candidate_mask", 0, False),
        ("active_mask", 5, True),
        ("candidate_mask", 1, True),
    ),
)
def test_post_scrub_validator_rejects_false_mask_plans(
    mask_name: str,
    index: int,
    replacement: bool,
) -> None:
    state, target, config = _state_target_and_config()
    plan, scrubbed = _scrub(state, target, config)
    false_mask = getattr(plan, mask_name).at[index].set(replacement)
    attacked = dataclasses.replace(plan, **{mask_name: false_mask})

    validation = validate_post_scrub_expanded_expression_absence(
        state,
        scrubbed,
        target,
        attacked,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert not validation.plan_matches_canonical
    assert validation.target_absent_from_all_expanded_trees
    assert not validation.valid


@pytest.mark.parametrize("drift", ("key", "provenance", "resource-value"))
def test_post_scrub_validator_rejects_plan_stale_for_pre_state_bits(drift: str) -> None:
    state, target, config = _state_target_and_config()
    stale_plan = compile_expanded_expression_lineage_masks(state, target, config=config)
    if drift == "key":
        changed = state.replace(key=jr.key(999))  # type: ignore[attr-defined]
    elif drift == "provenance":
        changed = state.replace(  # type: ignore[attr-defined]
            feature_generator_policy=state.feature_generator_policy.at[0].set(1)
        )
    else:
        generator = state.generator_resource_state.replace(  # type: ignore[attr-defined]
            log_weights=state.generator_resource_state.log_weights.at[0, 0].set(0.25)
        )
        changed = state.replace(generator_resource_state=generator)  # type: ignore[attr-defined]
    _, scrubbed = _scrub(changed, target, config)

    validation = validate_post_scrub_expanded_expression_absence(
        changed,
        scrubbed,
        target,
        stale_plan,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert not validation.plan_matches_canonical
    assert validation.transaction_matches_recomputed_commit
    assert validation.target_absent_from_all_expanded_trees
    assert not validation.valid


def test_post_scrub_validator_rejects_filler_target_collision() -> None:
    state, target, config = _state_target_and_config()
    plan, scrubbed = _scrub(state, target, config)
    collision = scrubbed.replace(  # type: ignore[attr-defined]
        ops=scrubbed.ops.at[3].set(OP_PRODUCT),
        parent_a=scrubbed.parent_a.at[3].set(0),
        parent_b=scrubbed.parent_b.at[3].set(1),
        theta=scrubbed.theta.at[3].set(jnp.zeros((2,), dtype=jnp.float32)),
        depth=scrubbed.depth.at[3].set(1),
    )

    validation = validate_post_scrub_expanded_expression_absence(
        state,
        collision,
        target,
        plan,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert validation.plan_matches_canonical
    assert validation.active_roots_containing_target_after == 1
    assert validation.active_target_subtree_occurrences_after == 1
    assert not validation.target_absent_from_all_expanded_trees
    assert not validation.valid


def test_post_scrub_validator_rejects_arbitrary_unmasked_state_mutation() -> None:
    state, target, config = _state_target_and_config()
    plan, scrubbed = _scrub(state, target, config)
    mutated = scrubbed.replace(  # type: ignore[attr-defined]
        output_bias=scrubbed.output_bias.at[0].add(jnp.float32(1.0))
    )

    validation = validate_post_scrub_expanded_expression_absence(
        state,
        mutated,
        target,
        plan,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert validation.plan_matches_canonical
    assert validation.target_absent_from_all_expanded_trees
    assert not validation.transaction_matches_recomputed_commit
    assert not validation.valid


def test_target_absent_prestate_and_empty_plan_cannot_validate() -> None:
    state, _, config = _state_target_and_config()
    absent = product_expression(raw_expression(0), raw_expression(2))
    plan = compile_expanded_expression_lineage_masks(state, absent, config=config)

    assert not plan.audit.pre_target_present
    assert not plan.audit.nonempty_causal_plan
    assert not bool(jnp.any(plan.active_mask))
    assert not bool(jnp.any(plan.candidate_mask))
    validation = validate_post_scrub_expanded_expression_absence(
        state,
        state,
        absent,
        plan,
        config=config,
        scrub_config=_scrub_config(config),
    )

    assert not validation.pre_target_present
    assert not validation.nonempty_causal_plan
    assert not validation.recomputed_commit_succeeded
    assert not validation.valid


def test_compiler_fails_closed_on_noncanonical_target_and_malformed_state() -> None:
    state, target, config = _state_target_and_config()
    noncanonical = GeneratedExpression(
        op="product",
        left=raw_expression(1),
        right=raw_expression(0),
    )
    with pytest.raises(ValueError, match="canonical"):
        compile_expanded_expression_lineage_masks(state, noncanonical, config=config)

    with pytest.raises(ValueError, match="raw target"):
        compile_expanded_expression_lineage_masks(
            state,
            raw_expression(0),
            config=config,
        )
    collision_config = dataclasses.replace(
        config,
        filler_op=OP_PRODUCT,
        filler_parent_a=0,
        filler_parent_b=1,
    )
    with pytest.raises(ValueError, match="filler"):
        compile_expanded_expression_lineage_masks(state, target, config=collision_config)

    cyclic = state.replace(parent_a=state.parent_a.at[7].set(7))  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="topological"):
        compile_expanded_expression_lineage_masks(cyclic, target, config=config)

    wrong_dtype = state.replace(ops=state.ops.astype(jnp.float32))  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="state.ops"):
        compile_expanded_expression_lineage_masks(wrong_dtype, target, config=config)

    legacy_key = state.replace(key=jr.PRNGKey(3))  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="state.key"):
        compile_expanded_expression_lineage_masks(legacy_key, target, config=config)

    bad_provenance = state.replace(  # type: ignore[attr-defined]
        candidate_generator_policy=state.candidate_generator_policy.at[0].set(4)
    )
    with pytest.raises(ValueError, match="candidate_generator_policy"):
        compile_expanded_expression_lineage_masks(bad_provenance, target, config=config)

    drifted_config = dataclasses.replace(config, candidate_slots=5)
    with pytest.raises(ValueError, match="state.candidate_ops"):
        compile_expanded_expression_lineage_masks(state, target, config=drifted_config)


def test_contract_is_development_host_only_and_grants_no_authority() -> None:
    state, target, config = _state_target_and_config()
    plan = compile_expanded_expression_lineage_masks(state, target, config=config)

    assert config.schema == EXPANDED_EXPRESSION_LINEAGE_SCHEMA
    assert config.status == EXPANDED_EXPRESSION_LINEAGE_STATUS
    assert config.development_only
    assert config.host_only_not_jittable
    assert not config.target_d_special_casing
    assert not config.behavioral_memory_erasure_claimed
    assert not config.execution_authorized
    assert not config.runner_authorized
    assert not config.campaign_authorized
    assert not config.evidence_authorized
    assert not config.artifact_writes_authorized
    assert not config.scientific_promotion_allowed
    assert plan.audit.host_audit_metadata_bytes_included is False
    assert plan.audit.learner_update_jax_kernel_operations == 0
    assert "validator scrub-kernel work are excluded" in plan.audit.operation_accounting_scope
    assert plan.audit.wall_clock_threshold is None

    for authority in (
        "target_d_special_casing",
        "behavioral_memory_erasure_claimed",
        "execution_authorized",
        "runner_authorized",
        "campaign_authorized",
        "evidence_authorized",
        "artifact_writes_authorized",
        "scientific_promotion_allowed",
    ):
        with pytest.raises(ValueError, match="authority|special casing"):
            dataclasses.replace(config, **{authority: True})


def test_config_rejects_nonexact_types() -> None:
    _, _, config = _state_target_and_config()
    with pytest.raises(TypeError, match="feature_dim"):
        dataclasses.replace(config, feature_dim=np.int32(3))
    with pytest.raises(TypeError, match="development_only"):
        dataclasses.replace(config, development_only=1)
    with pytest.raises(TypeError, match="schema"):
        dataclasses.replace(config, schema=object())
    assert isinstance(jnp.asarray(True), jax.Array)
