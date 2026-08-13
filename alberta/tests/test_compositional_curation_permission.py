# mypy: disable-error-code="attr-defined,call-arg,no-any-return,no-untyped-def"
"""Dynamic, causally consumed compositional-curation permission."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_feature_adapter import (
    COMPOSITIONAL_FEATURE_ADAPTER_PREPARED_CURATION_PERMISSION_NBYTES,
    CompositionalFeatureAdapter,
)
from alberta_framework.core.compositional_features import (
    DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS,
    CompositionalFeatureLearner,
)

pytestmark = pytest.mark.unit


def _learner(
    *,
    replacement_interval: int,
    learned_resources: bool = False,
    candidate_count: int = 0,
) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=candidate_count,
        step_size_output=0.03,
        step_size_theta=0.0,
        utility_decay=0.99,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        use_obgd=False,
        train_candidate_theta=False,
        learn_generator_resources=learned_resources,
    )


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
            left_dtype, jax.dtypes.prng_key
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


def _assert_no_structural_event(result) -> None:
    trace = result.curation_trace
    assert not bool(trace.should_try_replace)
    assert not bool(trace.proposal_formed)
    assert not bool(trace.root_change_applied)
    assert not bool(trace.promotion_applied)
    assert not bool(trace.has_event)
    assert not bool(jnp.any(trace.root_change_mask))
    assert not bool(jnp.any(trace.cascade_refill_mask))
    assert not bool(jnp.any(trace.active_change_mask))
    assert not bool(jnp.any(trace.candidate_refresh_mask))
    assert not bool(jnp.any(trace.candidate_rebound_mask))
    assert not bool(jnp.any(trace.candidate_overdepth_regeneration_mask))
    assert int(trace.logical_event_count) == 0


@pytest.mark.parametrize(
    ("learned_resources", "replacement_interval"),
    ((False, 1), (True, 2)),
)
def test_default_true_is_bit_exact_and_permission_is_a_dynamic_bool_scalar(
    learned_resources: bool,
    replacement_interval: int,
) -> None:
    learner = _learner(
        replacement_interval=replacement_interval,
        learned_resources=learned_resources,
    )
    source = learner.init(2, jr.key(4101))
    if learned_resources:
        source = source.replace(replacement_accumulator=jnp.float32(0.99))
    observation = jnp.asarray((0.5, -0.25), dtype=jnp.float32)
    targets = jnp.asarray((1.0,), dtype=jnp.float32)

    default = learner.update(source, observation, targets)
    explicit = learner.update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert _tree_bits_equal(default, explicit)
    assert bool(default.curation_trace.should_try_replace)
    with pytest.raises((TypeError, ValueError)):
        learner.update(source, observation, targets, curation_allowed=jnp.int32(1))
    with pytest.raises((TypeError, ValueError)):
        learner.update(
            source,
            observation,
            targets,
            curation_allowed=jnp.asarray((True,), dtype=jnp.bool_),
        )


def test_fixed_cadence_denial_consumes_due_event_and_later_curation_is_bounded() -> None:
    learner = _learner(replacement_interval=2)
    state0 = learner.init(2, jr.key(4102))
    observation = jnp.asarray((0.75, 0.5), dtype=jnp.float32)
    targets = jnp.asarray((1.0,), dtype=jnp.float32)

    first = learner.update(state0, observation, targets, curation_allowed=False)
    denied = learner.update(
        first.state,
        observation,
        targets,
        curation_allowed=False,
    )

    assert int(first.state.replacement_phase) == 1
    assert int(denied.state.replacement_phase) == 0
    np.testing.assert_array_equal(denied.state.step_words, (0, 2))
    np.testing.assert_array_equal(
        jr.key_data(denied.state.key),
        jr.key_data(jr.split(first.state.key, 3)[0]),
    )
    assert not _tree_bits_equal(denied.state.output_weights, first.state.output_weights)
    np.testing.assert_array_equal(denied.state.ops, first.state.ops)
    np.testing.assert_array_equal(denied.state.parent_a, first.state.parent_a)
    np.testing.assert_array_equal(denied.state.parent_b, first.state.parent_b)
    np.testing.assert_array_equal(denied.state.depth, first.state.depth)
    _assert_no_structural_event(denied)

    third = learner.update(
        denied.state,
        observation,
        targets,
        curation_allowed=True,
    )
    admitted = learner.update(
        third.state,
        observation,
        targets,
        curation_allowed=True,
    )
    assert int(third.state.replacement_phase) == 1
    assert int(admitted.state.replacement_phase) == 0
    assert bool(admitted.curation_trace.should_try_replace)
    assert bool(admitted.curation_trace.has_event)


def test_learned_resource_denial_consumes_credit_without_backlog() -> None:
    learner = _learner(
        replacement_interval=2,
        learned_resources=True,
        candidate_count=1,
    )
    source = learner.init(2, jr.key(4103)).replace(
        replacement_accumulator=jnp.float32(0.99)
    )
    observation = jnp.asarray((0.25, 0.75), dtype=jnp.float32)
    targets = jnp.asarray((0.5,), dtype=jnp.float32)

    denied = learner.update(
        source,
        observation,
        targets,
        context_id=jnp.int32(0),
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    policy = int(denied.curation_trace.generator_policy_id)
    expected = (
        0.99
        + DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS[policy] / 2.0
        - 1.0
    )
    np.testing.assert_allclose(denied.state.replacement_accumulator, expected)
    assert 0.0 <= float(denied.state.replacement_accumulator) < 1.0
    assert int(denied.state.generator_resource_state.step_count) == 1
    np.testing.assert_array_equal(
        jr.key_data(denied.state.key),
        jr.key_data(jr.split(source.key, 3)[0]),
    )
    _assert_no_structural_event(denied)

    def step(state, _):
        result = learner.update(
            state,
            observation,
            targets,
            context_id=jnp.int32(0),
            curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
        )
        return result.state, result.state.replacement_accumulator

    final, accumulators = jax.jit(
        lambda state: jax.lax.scan(step, state, jnp.arange(64))
    )(denied.state)
    assert bool(jnp.all((accumulators >= 0.0) & (accumulators < 1.0)))
    np.testing.assert_array_equal(final.step_words, (0, 65))
    assert int(final.generator_resource_state.step_count) == 65


def test_denial_suppresses_mature_candidate_promotion_and_refresh() -> None:
    learner = _learner(replacement_interval=1, candidate_count=1)
    source = learner.init(2, jr.key(4106)).replace(
        candidate_utilities=jnp.asarray((100.0,), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 0.0, 1.0), dtype=jnp.float32),
        ages=jnp.asarray((10, 10, 10, 10), dtype=jnp.int32),
    )
    observation = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    targets = jnp.asarray((0.0,), dtype=jnp.float32)

    denied = learner.update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )

    _assert_no_structural_event(denied)
    np.testing.assert_array_equal(denied.state.candidate_ops, source.candidate_ops)
    np.testing.assert_array_equal(
        denied.state.candidate_parent_a,
        source.candidate_parent_a,
    )
    np.testing.assert_array_equal(
        denied.state.candidate_parent_b,
        source.candidate_parent_b,
    )
    np.testing.assert_array_equal(
        denied.state.candidate_depth,
        source.candidate_depth,
    )
    np.testing.assert_array_equal(denied.state.candidate_ages, (11,))
    np.testing.assert_array_equal(denied.state.step_words, (0, 1))


def test_adapter_captures_permission_recomputes_it_and_counts_one_transient_byte() -> None:
    adapter = CompositionalFeatureAdapter(
        _learner(replacement_interval=1),
        base_feature_dim=2,
    )
    source = adapter.init(jr.key(4104))
    observation = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    targets = jnp.asarray((0.0,), dtype=jnp.float32)

    proposal = adapter.prepare_update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert proposal.curation_allowed.shape == ()
    assert proposal.curation_allowed.dtype == jnp.dtype(jnp.bool_)
    assert not bool(proposal.curation_allowed)
    assert proposal.curation_allowed.nbytes == (
        COMPOSITIONAL_FEATURE_ADAPTER_PREPARED_CURATION_PERMISSION_NBYTES
    )
    assert bool(proposal.diagnostics.transaction_applied)
    assert not bool(proposal.diagnostics.active_bank_changed)
    assert _tree_bits_equal(proposal.candidate_state.binding, source.binding)
    direct = adapter.update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert _tree_bits_equal(direct.state, proposal.candidate_state)
    assert _tree_bits_equal(direct.predictions, proposal.predictions)
    assert _tree_bits_equal(direct.errors, proposal.errors)
    assert _tree_bits_equal(direct.metrics, proposal.metrics)
    assert _tree_bits_equal(direct.curation_trace, proposal.curation_trace)
    assert _tree_bits_equal(direct.diagnostics, proposal.diagnostics)

    default_true = adapter.prepare_update(source, observation, targets)
    explicit_true = adapter.prepare_update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert _tree_bits_equal(default_true, explicit_true)

    committed = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(committed.diagnostics.proposal_integrity)
    assert bool(committed.diagnostics.applied)
    np.testing.assert_array_equal(committed.state.learner_state.step_words, (0, 1))

    tampered = proposal.replace(
        curation_allowed=jnp.asarray(True, dtype=jnp.bool_)
    )
    rejected = adapter.commit_prepared_update(
        source,
        tampered,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(rejected.diagnostics.proposal_integrity)
    assert not bool(rejected.diagnostics.applied)
    assert _tree_bits_equal(rejected.state, source)


def test_permission_is_eager_jit_and_scan_safe() -> None:
    adapter = CompositionalFeatureAdapter(
        _learner(replacement_interval=1),
        base_feature_dim=2,
    )
    source = adapter.init(jr.key(4105))
    observation = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    targets = jnp.asarray((0.0,), dtype=jnp.float32)

    eager = adapter.prepare_update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    compiled = jax.jit(adapter.prepare_update)(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert _tree_bits_equal(eager, compiled)
    assert adapter.measure_prepared_update_nbytes(eager) == (
        adapter.measure_prepared_update_nbytes(compiled)
    )

    def step(state, allowed):
        proposal = adapter.prepare_update(
            state,
            observation,
            targets,
            curation_allowed=allowed,
        )
        result = adapter.commit_prepared_update(
            state,
            proposal,
            consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
        )
        return result.state, (
            result.diagnostics.applied,
            proposal.diagnostics.active_bank_changed,
        )

    final, (applied, changed) = jax.jit(
        lambda state: jax.lax.scan(
            step,
            state,
            jnp.asarray((False, False, True), dtype=jnp.bool_),
        )
    )(source)
    np.testing.assert_array_equal(applied, (True, True, True))
    np.testing.assert_array_equal(changed, (False, False, True))
    np.testing.assert_array_equal(final.learner_state.step_words, (0, 3))
