"""Focused contracts for the development-only compositional scrub kernel."""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Callable

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
    CompositionalFeatureLearner,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    ACTIVE_MASKED_LEAF_PATHS,
    CANDIDATE_MASKED_LEAF_PATHS,
    COMPOSITIONAL_STATE_LEAF_PATHS,
    CROSS_MASKED_LEAF_PATHS,
    PRESERVED_LEAF_PATHS,
    GeneratedClassScrubConfig,
    compositional_state_leaf_paths,
    persistent_compositional_state_nbytes,
    scrub_compositional_feature_state,
)

pytestmark = pytest.mark.unit


def _state_and_masks():
    learner = CompositionalFeatureLearner(
        n_features=6,
        n_tasks=1,
        candidate_count=3,
        replacement_interval=0,
        max_depth=3,
    )
    state = learner.init(2, jr.key(71))
    state = state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM, OP_GATED, OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 0, 2, 0, 4), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 0, 1, 1), dtype=jnp.int32),
        theta=jnp.arange(12, dtype=jnp.float32).reshape(6, 2) + 0.25,
        depth=jnp.asarray((0, 0, 1, 2, 1, 2), dtype=jnp.int32),
        output_weights=jnp.arange(6, dtype=jnp.float32)[None, :] + 1.0,
        output_bias=jnp.asarray((91.0,), dtype=jnp.float32),
        utilities=jnp.arange(6, dtype=jnp.float32) + 2.0,
        utility_contribution_trace=(jnp.arange(6, dtype=jnp.float32)[None, :] + 3.0),
        utility_error_trace=jnp.asarray((92.0,), dtype=jnp.float32),
        utility_feature_trace=jnp.arange(6, dtype=jnp.float32) + 4.0,
        utility_feature_energy_trace=jnp.arange(6, dtype=jnp.float32) + 5.0,
        utility_signal_second_moment=jnp.arange(6, dtype=jnp.float32) + 6.0,
        feature_score_residual_trace=(jnp.arange(6, dtype=jnp.float32)[None, :] + 7.0),
        feature_score_energy_trace=jnp.arange(6, dtype=jnp.float32) + 8.0,
        retention_slow_utilities=jnp.arange(6, dtype=jnp.float32) + 9.0,
        ages=jnp.arange(6, dtype=jnp.int32) + 10,
        candidate_ops=jnp.asarray((OP_GATED, OP_SUM, OP_PRODUCT), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((2, 4, 0), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((0, 1, 1), dtype=jnp.int32),
        candidate_theta=jnp.arange(6, dtype=jnp.float32).reshape(3, 2) + 10.0,
        candidate_depth=jnp.asarray((2, 2, 1), dtype=jnp.int32),
        candidate_output_weights=(jnp.arange(3, dtype=jnp.float32)[None, :] + 11.0),
        candidate_utilities=jnp.arange(3, dtype=jnp.float32) + 12.0,
        candidate_utility_contribution_trace=(jnp.arange(3, dtype=jnp.float32)[None, :] + 13.0),
        candidate_utility_feature_trace=jnp.arange(3, dtype=jnp.float32) + 14.0,
        candidate_utility_feature_energy_trace=(jnp.arange(3, dtype=jnp.float32) + 15.0),
        candidate_utility_signal_second_moment=(jnp.arange(3, dtype=jnp.float32) + 16.0),
        candidate_score_residual_trace=(jnp.arange(3, dtype=jnp.float32)[None, :] + 17.0),
        candidate_score_energy_trace=jnp.arange(3, dtype=jnp.float32) + 18.0,
        candidate_retention_slow_utilities=(jnp.arange(3, dtype=jnp.float32) + 19.0),
        candidate_active_correlation_trace=(jnp.arange(18, dtype=jnp.float32).reshape(3, 6) + 20.0),
        candidate_ages=jnp.arange(3, dtype=jnp.int32) + 20,
        candidate_selector_log_weights=jnp.arange(3, dtype=jnp.float32) + 21.0,
        candidate_selector_cumulative_loss=(jnp.arange(3, dtype=jnp.float32) + 22.0),
        candidate_selector_action_counts=(jnp.arange(3, dtype=jnp.float32) + 23.0),
        feature_generator_policy=jnp.asarray((0, 1, 2, 3, 1, 2), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((1, 2, 3), dtype=jnp.int32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    active_mask = jnp.asarray((False, False, True, True, False, False))
    candidate_mask = jnp.asarray((True, False, True))
    return state, active_mask, candidate_mask


def _config(**changes: object) -> GeneratedClassScrubConfig:
    values: dict[str, object] = {
        "feature_dim": 2,
        "active_slots": 6,
        "candidate_slots": 3,
        "n_tasks": 1,
        "filler_op": OP_GATED,
        "filler_parent_a": 0,
        "filler_parent_b": 1,
    }
    values.update(changes)
    return GeneratedClassScrubConfig(**values)  # type: ignore[arg-type]


def _leaf_bytes(value: object) -> bytes:
    if isinstance(value, jax.Array):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            array = np.asarray(jr.key_data(value))
            return b"key:" + repr(array.shape).encode() + array.tobytes()
        array = np.asarray(value)
        return array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()
    if isinstance(value, float):
        return b"float:" + struct.pack(">d", value)
    raise TypeError(type(value))


def _assert_tree_bit_exact(first: object, second: object) -> None:
    first_with_paths, first_tree = jax.tree_util.tree_flatten_with_path(first)
    second_with_paths, second_tree = jax.tree_util.tree_flatten_with_path(second)
    assert first_tree == second_tree
    assert [str(path) for path, _ in first_with_paths] == [
        str(path) for path, _ in second_with_paths
    ]
    for (_, first_leaf), (_, second_leaf) in zip(
        first_with_paths,
        second_with_paths,
        strict=True,
    ):
        assert _leaf_bytes(first_leaf) == _leaf_bytes(second_leaf)


def test_leaf_partition_is_exact_exhaustive_and_narrow() -> None:
    state, _, _ = _state_and_masks()
    actual = compositional_state_leaf_paths(state)
    groups = (
        ACTIVE_MASKED_LEAF_PATHS,
        CANDIDATE_MASKED_LEAF_PATHS,
        CROSS_MASKED_LEAF_PATHS,
        PRESERVED_LEAF_PATHS,
    )

    assert actual == COMPOSITIONAL_STATE_LEAF_PATHS
    assert set().union(*groups) == set(actual)
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            assert set(first).isdisjoint(second)
    assert "output_bias" in PRESERVED_LEAF_PATHS
    assert "utility_error_trace" in PRESERVED_LEAF_PATHS
    assert "step_words" in PRESERVED_LEAF_PATHS
    assert "replacement_phase" in PRESERVED_LEAF_PATHS

    config = _config()
    assert config.development_only
    assert config.identity_head_lineage_only
    assert not config.behavioral_information_erasure_claimed
    assert not config.expanded_expression_absence_claimed
    assert config.host_timing_canonicalization_required_for_jit_bit_equality
    assert not config.execution_authorized
    assert not config.evidence_authorized


def test_commit_scrubs_every_local_leaf_and_both_correlation_axes() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    result = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=_config(),
    )

    assert bool(result.diagnostics.plan_valid)
    assert bool(result.diagnostics.committed)
    assert not bool(result.diagnostics.behavioral_information_erasure_claimed)
    assert int(result.diagnostics.active_scrub_count) == 2
    assert int(result.diagnostics.candidate_scrub_count) == 2
    assert bool(result.diagnostics.masked_local_state_reset_exact)
    assert bool(result.diagnostics.no_masked_old_local_descriptor_remains)
    assert not bool(result.diagnostics.expanded_expression_absence_claimed)
    assert bool(result.diagnostics.host_timing_canonicalization_required_for_jit_bit_equality)
    scrubbed = result.state

    np.testing.assert_array_equal(scrubbed.ops[active_mask], OP_GATED)
    np.testing.assert_array_equal(scrubbed.parent_a[active_mask], 0)
    np.testing.assert_array_equal(scrubbed.parent_b[active_mask], 1)
    np.testing.assert_array_equal(scrubbed.depth[active_mask], 1)
    np.testing.assert_array_equal(scrubbed.theta[active_mask], 0.0)
    np.testing.assert_array_equal(scrubbed.candidate_ops[candidate_mask], OP_GATED)
    np.testing.assert_array_equal(scrubbed.candidate_parent_a[candidate_mask], 0)
    np.testing.assert_array_equal(scrubbed.candidate_parent_b[candidate_mask], 1)
    np.testing.assert_array_equal(scrubbed.candidate_depth[candidate_mask], 1)
    np.testing.assert_array_equal(scrubbed.candidate_theta[candidate_mask], 0.0)

    active_axis0 = (
        "utilities",
        "utility_feature_trace",
        "utility_feature_energy_trace",
        "utility_signal_second_moment",
        "feature_score_energy_trace",
        "retention_slow_utilities",
        "ages",
        "feature_generator_policy",
    )
    active_axis1 = (
        "output_weights",
        "utility_contribution_trace",
        "feature_score_residual_trace",
    )
    for name in active_axis0:
        np.testing.assert_array_equal(getattr(scrubbed, name)[active_mask], 0)
        np.testing.assert_array_equal(
            getattr(scrubbed, name)[~active_mask],
            getattr(state, name)[~active_mask],
        )
    for name in active_axis1:
        np.testing.assert_array_equal(getattr(scrubbed, name)[:, active_mask], 0)
        np.testing.assert_array_equal(
            getattr(scrubbed, name)[:, ~active_mask],
            getattr(state, name)[:, ~active_mask],
        )

    candidate_axis0 = (
        "candidate_utilities",
        "candidate_utility_feature_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_signal_second_moment",
        "candidate_score_energy_trace",
        "candidate_retention_slow_utilities",
        "candidate_ages",
        "candidate_selector_log_weights",
        "candidate_selector_cumulative_loss",
        "candidate_selector_action_counts",
        "candidate_generator_policy",
    )
    candidate_axis1 = (
        "candidate_output_weights",
        "candidate_utility_contribution_trace",
        "candidate_score_residual_trace",
    )
    for name in candidate_axis0:
        np.testing.assert_array_equal(getattr(scrubbed, name)[candidate_mask], 0)
        np.testing.assert_array_equal(
            getattr(scrubbed, name)[~candidate_mask],
            getattr(state, name)[~candidate_mask],
        )
    for name in candidate_axis1:
        np.testing.assert_array_equal(getattr(scrubbed, name)[:, candidate_mask], 0)
        np.testing.assert_array_equal(
            getattr(scrubbed, name)[:, ~candidate_mask],
            getattr(state, name)[:, ~candidate_mask],
        )

    expected_correlation = np.asarray(state.candidate_active_correlation_trace).copy()
    expected_correlation[np.asarray(candidate_mask), :] = 0.0
    expected_correlation[:, np.asarray(active_mask)] = 0.0
    np.testing.assert_array_equal(
        scrubbed.candidate_active_correlation_trace,
        expected_correlation,
    )
    np.testing.assert_array_equal(scrubbed.output_bias, state.output_bias)
    np.testing.assert_array_equal(scrubbed.utility_error_trace, state.utility_error_trace)


def test_masked_float_leaves_are_exact_positive_zero_bits() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    scrubbed = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=_config(),
    ).state

    def assert_positive_zero(value: object) -> None:
        array = np.asarray(value)
        assert array.dtype == np.float32
        np.testing.assert_array_equal(array.view(np.uint32), np.uint32(0))

    assert_positive_zero(scrubbed.theta[active_mask])
    for name in (
        "utilities",
        "utility_feature_trace",
        "utility_feature_energy_trace",
        "utility_signal_second_moment",
        "feature_score_energy_trace",
        "retention_slow_utilities",
    ):
        assert_positive_zero(getattr(scrubbed, name)[active_mask])
    for name in (
        "output_weights",
        "utility_contribution_trace",
        "feature_score_residual_trace",
    ):
        assert_positive_zero(getattr(scrubbed, name)[:, active_mask])

    assert_positive_zero(scrubbed.candidate_theta[candidate_mask])
    for name in (
        "candidate_utilities",
        "candidate_utility_feature_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_signal_second_moment",
        "candidate_score_energy_trace",
        "candidate_retention_slow_utilities",
        "candidate_selector_log_weights",
        "candidate_selector_cumulative_loss",
        "candidate_selector_action_counts",
    ):
        assert_positive_zero(getattr(scrubbed, name)[candidate_mask])
    for name in (
        "candidate_output_weights",
        "candidate_utility_contribution_trace",
        "candidate_score_residual_trace",
    ):
        assert_positive_zero(getattr(scrubbed, name)[:, candidate_mask])

    correlation_mask = np.asarray(candidate_mask)[:, None] | np.asarray(active_mask)[None, :]
    assert_positive_zero(np.asarray(scrubbed.candidate_active_correlation_trace)[correlation_mask])


def test_sham_uses_valid_plan_but_is_a_bit_exact_noop() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    sham = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(False),
        config=_config(),
    )

    assert bool(sham.diagnostics.plan_valid)
    assert bool(sham.diagnostics.sham_noop)
    assert not bool(sham.diagnostics.committed)
    assert not bool(sham.diagnostics.rolled_back)
    assert int(sham.diagnostics.active_scrub_count) == 2
    assert int(sham.diagnostics.candidate_scrub_count) == 2
    _assert_tree_bit_exact(sham.state, state)


@pytest.mark.parametrize(
    "mutate_masks_or_state",
    (
        pytest.param(
            lambda state, active, candidate: (
                state,
                active.at[0].set(True),
                candidate,
            ),
            id="raw-prefix-mask",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state,
                active.at[3].set(False),
                candidate,
            ),
            id="active-descendant-closure",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state,
                active,
                candidate.at[0].set(False),
            ),
            id="candidate-parent-closure",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state.replace(  # type: ignore[attr-defined]
                    ops=state.ops.at[5].set(OP_RAW),
                    parent_a=state.parent_a.at[5].set(0),
                    parent_b=state.parent_b.at[5].set(-1),
                    depth=state.depth.at[5].set(0),
                ),
                active,
                candidate,
            ),
            id="raw-slot-outside-prefix",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state.replace(  # type: ignore[attr-defined]
                    parent_a=state.parent_a.at[5].set(5)
                ),
                active,
                candidate,
            ),
            id="invalid-dag",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state.replace(  # type: ignore[attr-defined]
                    candidate_utilities=state.candidate_utilities.at[1].set(jnp.nan)
                ),
                active,
                candidate,
            ),
            id="nonfinite-state",
        ),
        pytest.param(
            lambda state, active, candidate: (
                state,
                jnp.zeros_like(active),
                jnp.zeros_like(candidate),
            ),
            id="empty-plan",
        ),
    ),
)
def test_invalid_plans_roll_back_every_bit(
    mutate_masks_or_state: Callable,
) -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    bad_state, bad_active, bad_candidate = mutate_masks_or_state(
        state,
        active_mask,
        candidate_mask,
    )
    result = scrub_compositional_feature_state(
        bad_state,
        bad_active,
        bad_candidate,
        jnp.asarray(True),
        config=_config(),
    )

    assert not bool(result.diagnostics.plan_valid)
    assert not bool(result.diagnostics.committed)
    assert bool(result.diagnostics.rolled_back)
    _assert_tree_bit_exact(result.state, bad_state)


@pytest.mark.parametrize(
    "policy_field",
    ("feature_generator_policy", "candidate_generator_policy"),
)
def test_out_of_range_generator_policy_ids_roll_back_atomically(policy_field: str) -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    policies = getattr(state, policy_field).at[0].set(4)
    bad_state = state.replace(**{policy_field: policies})  # type: ignore[attr-defined]

    result = scrub_compositional_feature_state(
        bad_state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=_config(),
    )

    assert not bool(result.diagnostics.generator_policy_ids_valid)
    assert not bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.plan_valid)
    assert bool(result.diagnostics.rolled_back)
    _assert_tree_bit_exact(result.state, bad_state)


def test_unmasked_duplicate_old_descriptor_fails_postcondition_atomically() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    # Candidate 2 duplicates masked active slot 2's local descriptor.  Leaving
    # it unmasked would preserve an old identity/head lineage.
    candidate_mask = candidate_mask.at[2].set(False)
    result = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=_config(),
    )

    assert not bool(result.diagnostics.no_masked_old_local_descriptor_remains)
    assert not bool(result.diagnostics.plan_valid)
    _assert_tree_bit_exact(result.state, state)


def test_filler_equal_to_masked_descriptor_is_unsafe_and_rolls_back() -> None:
    state, _, _ = _state_and_masks()
    # Slot 4 already has the canonical gate(0, 1) filler.  Its descendant and
    # dependent candidate are included, so filler identity is the only defect.
    active_mask = jnp.asarray((False, False, False, False, True, True))
    candidate_mask = jnp.asarray((False, True, False))
    result = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=_config(),
    )

    assert not bool(result.diagnostics.filler_distinct_from_masked_local_descriptors)
    assert not bool(result.diagnostics.plan_valid)
    _assert_tree_bit_exact(result.state, state)


def test_eager_and_jit_are_bit_exact_and_preserve_persistent_resource_bytes() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    config = _config()

    eager = scrub_compositional_feature_state(
        state,
        active_mask,
        candidate_mask,
        jnp.asarray(True),
        config=config,
    )
    compiled = jax.jit(
        lambda current, active, candidate, commit: scrub_compositional_feature_state(
            current,
            active,
            candidate,
            commit,
            config=config,
        )
    )(state, active_mask, candidate_mask, jnp.asarray(True))

    _assert_tree_bit_exact(eager.state, compiled.state)
    _assert_tree_bit_exact(eager.diagnostics, compiled.diagnostics)
    before = persistent_compositional_state_nbytes(state)
    assert persistent_compositional_state_nbytes(eager.state) == before
    assert persistent_compositional_state_nbytes(compiled.state) == before
    assert bool(eager.diagnostics.resource_shape_preserved)


def test_static_shape_and_dtype_contracts_fail_before_tracing() -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    with pytest.raises(ValueError, match="active_mask"):
        scrub_compositional_feature_state(
            state,
            active_mask[:-1],
            candidate_mask,
            jnp.asarray(True),
            config=_config(),
        )
    with pytest.raises(TypeError, match="candidate_mask"):
        scrub_compositional_feature_state(
            state,
            active_mask,
            candidate_mask.astype(jnp.int32),
            jnp.asarray(True),
            config=_config(),
        )
    with pytest.raises(ValueError, match="state active-slot"):
        scrub_compositional_feature_state(
            state,
            active_mask,
            candidate_mask,
            jnp.asarray(True),
            config=dataclasses.replace(_config(), active_slots=7),
        )


@pytest.mark.parametrize(
    "bad_key",
    (
        pytest.param(jr.PRNGKey(9), id="legacy-uint32-key"),
        pytest.param(jr.key(9, impl="rbg"), id="non-threefry-typed-key"),
    ),
)
def test_static_key_contract_rejects_legacy_and_non_threefry_keys(bad_key: jax.Array) -> None:
    state, active_mask, candidate_mask = _state_and_masks()
    bad_state = state.replace(key=bad_key)  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="state.key"):
        scrub_compositional_feature_state(
            bad_state,
            active_mask,
            candidate_mask,
            jnp.asarray(True),
            config=_config(),
        )


def test_config_requires_exact_builtin_types_and_canonical_disclosures() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="feature_dim"):
        _config(feature_dim=np.int32(2))
    with pytest.raises(TypeError, match="development_only"):
        _config(development_only=1)
    with pytest.raises(TypeError, match="schema"):
        _config(schema=StringSubclass("alberta.generated-class-lifecycle-scrub.development.v0"))
    with pytest.raises(ValueError, match="expanded-expression"):
        _config(expanded_expression_absence_claimed=True)
    with pytest.raises(ValueError, match="timing"):
        _config(host_timing_canonicalization_required_for_jit_bit_equality=False)
