"""Focused unit tests for principled multi-step compositional admission/retention."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

pytestmark = pytest.mark.unit


def _novelty_learner(bonus: float) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        utility_decay=0.0,
        replacement_interval=4,
        min_feature_age=0,
        candidate_min_age=3,
        promotion_margin=1.0,
        max_depth=2,
        use_obgd=False,
        candidate_scoring_mode="energy_novelty",
        candidate_score_trace_decay=0.99,
        candidate_novelty_weight=0.0,
        candidate_novelty_admission_bonus=bonus,
    )


def _novelty_state(learner: CompositionalFeatureLearner) -> CompositionalFeatureState:
    return cast(
        CompositionalFeatureState,
        learner.init(feature_dim=2, key=jr.key(0)).replace(  # type: ignore[attr-defined]
            ops=jnp.array([OP_RAW, OP_RAW, OP_SUM], dtype=jnp.int32),
            parent_a=jnp.array([0, 1, 0], dtype=jnp.int32),
            parent_b=jnp.array([-1, -1, 1], dtype=jnp.int32),
            depth=jnp.array([0, 0, 1], dtype=jnp.int32),
            candidate_ops=jnp.array([OP_PRODUCT], dtype=jnp.int32),
            candidate_parent_a=jnp.array([0], dtype=jnp.int32),
            candidate_parent_b=jnp.array([1], dtype=jnp.int32),
            candidate_depth=jnp.array([1], dtype=jnp.int32),
        ),
    )


def test_novelty_bonus_admits_mature_zero_direct_utility_intermediate() -> None:
    enabled = _novelty_learner(1.0)
    disabled = _novelty_learner(0.0)
    enabled_state = _novelty_state(enabled)
    disabled_state = _novelty_state(disabled)
    observations = (
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([1.0, -1.0], dtype=jnp.float32),
        jnp.array([-1.0, 1.0], dtype=jnp.float32),
        jnp.array([-1.0, -1.0], dtype=jnp.float32),
    )

    for observation in observations[:3]:
        enabled_state = enabled.update(
            enabled_state, observation, jnp.zeros((1,), dtype=jnp.float32)
        ).state
        disabled_state = disabled.update(
            disabled_state, observation, jnp.zeros((1,), dtype=jnp.float32)
        ).state

    diagnostics = enabled.ranking_diagnostics(enabled_state, feature_dim=2)
    assert bool(diagnostics.contract_valid)
    assert bool(diagnostics.candidate_mature[0])
    assert float(diagnostics.direct_candidate_scores[0]) == 0.0
    assert float(diagnostics.candidate_novelty_scores[0]) > 0.0
    assert float(diagnostics.augmented_candidate_scores[0]) > 0.0

    enabled_result = enabled.update(
        enabled_state, observations[3], jnp.zeros((1,), dtype=jnp.float32)
    )
    disabled_result = disabled.update(
        disabled_state, observations[3], jnp.zeros((1,), dtype=jnp.float32)
    )

    assert int(enabled_result.promoted_candidate) == 0
    assert bool(enabled_result.curation_trace.promotion_applied)
    assert int(enabled_result.state.ops[2]) == OP_PRODUCT
    assert float(enabled_state.candidate_utilities[0]) == 0.0
    assert int(disabled_result.promoted_candidate) == -1
    assert not bool(disabled_result.curation_trace.promotion_applied)
    assert int(disabled_result.state.ops[2]) == OP_SUM


def test_novelty_bonus_cannot_bypass_maturity_or_destination_compatibility() -> None:
    immature = CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        utility_decay=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=3,
        promotion_margin=1.0,
        max_depth=3,
        use_obgd=False,
        candidate_scoring_mode="energy_novelty",
        candidate_score_trace_decay=0.99,
        candidate_novelty_admission_bonus=10.0,
    )
    immature_result = immature.update(
        _novelty_state(immature),
        jnp.array([1.0, -1.0], dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    assert int(immature_result.promoted_candidate) == -1
    assert not bool(immature_result.curation_trace.promotion_applied)

    incompatible_state = cast(
        CompositionalFeatureState,
        _novelty_state(immature).replace(  # type: ignore[attr-defined]
            candidate_parent_a=jnp.array([2], dtype=jnp.int32),
            candidate_parent_b=jnp.array([0], dtype=jnp.int32),
            candidate_depth=jnp.array([2], dtype=jnp.int32),
            candidate_ages=jnp.array([3], dtype=jnp.int32),
            candidate_score_energy_trace=jnp.array([1.0], dtype=jnp.float32),
            feature_score_energy_trace=jnp.ones((3,), dtype=jnp.float32),
            candidate_active_correlation_trace=jnp.zeros(
                (1, 3), dtype=jnp.float32
            ),
        ),
    )
    incompatible_result = immature.update(
        incompatible_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    assert int(incompatible_result.promoted_candidate) == -1
    assert not bool(incompatible_result.curation_trace.promotion_applied)


def _retention_learner(decay: float) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=0,
        step_size_output=0.0,
        utility_decay=0.999,
        replacement_interval=1,
        min_feature_age=0,
        max_depth=3,
        use_obgd=False,
        ancestor_utility_backup_decay=decay,
    )


def _retention_state(
    learner: CompositionalFeatureLearner,
) -> CompositionalFeatureState:
    return cast(
        CompositionalFeatureState,
        learner.init(feature_dim=2, key=jr.key(1)).replace(  # type: ignore[attr-defined]
            ops=jnp.array(
                [OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT, OP_SUM], dtype=jnp.int32
            ),
            parent_a=jnp.array([0, 1, 0, 2, 0], dtype=jnp.int32),
            parent_b=jnp.array([-1, -1, 1, 0, 1], dtype=jnp.int32),
            depth=jnp.array([0, 0, 1, 2, 1], dtype=jnp.int32),
            utilities=jnp.array(
                [0.0, 0.0, 0.0, 1.0, 0.25], dtype=jnp.float32
            ),
            ages=jnp.full((5,), 100, dtype=jnp.int32),
        ),
    )


def test_descendant_utility_backs_up_transitively_for_replacement_only() -> None:
    enabled = _retention_learner(0.5)
    disabled = _retention_learner(0.0)
    enabled_state = _retention_state(enabled)
    disabled_state = _retention_state(disabled)

    diagnostics = enabled.ranking_diagnostics(enabled_state, feature_dim=2)
    np.testing.assert_array_equal(
        np.asarray(diagnostics.direct_active_scores),
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.25], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.backed_active_scores),
        np.asarray([0.5, 0.25, 0.5, 1.0, 0.25], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(enabled_state.utilities),
        np.asarray(diagnostics.direct_active_scores),
    )

    observation = jnp.array([1.0, -1.0], dtype=jnp.float32)
    inactive_target = jnp.array([jnp.nan], dtype=jnp.float32)
    enabled_result = enabled.update(enabled_state, observation, inactive_target)
    disabled_result = disabled.update(disabled_state, observation, inactive_target)

    assert int(enabled_result.replaced_slot) == 4
    assert int(disabled_result.replaced_slot) == 2


@pytest.mark.parametrize(
    ("name", "value", "error"),
    (
        ("candidate_novelty_admission_bonus", True, TypeError),
        ("candidate_novelty_admission_bonus", 1, TypeError),
        ("candidate_novelty_admission_bonus", float("nan"), ValueError),
        ("ancestor_utility_backup_decay", True, TypeError),
        ("ancestor_utility_backup_decay", 1, TypeError),
        ("ancestor_utility_backup_decay", float("inf"), ValueError),
        ("ancestor_utility_backup_decay", 1.1, ValueError),
    ),
)
def test_new_config_values_are_exact_and_finite(
    name: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            **cast(Any, {name: value}),
        )


def test_novelty_admission_requires_evidence_mode_and_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            candidate_count=0,
            candidate_scoring_mode="energy_novelty",
            candidate_novelty_admission_bonus=1.0,
        )
    with pytest.raises(ValueError, match="energy_novelty"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            candidate_count=1,
            candidate_novelty_admission_bonus=1.0,
        )


def test_config_roundtrip_and_explicit_disabled_defaults_are_exact() -> None:
    implicit = CompositionalFeatureLearner(n_features=5, n_tasks=1, candidate_count=1)
    explicit = CompositionalFeatureLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=1,
        candidate_novelty_admission_bonus=0.0,
        ancestor_utility_backup_decay=0.0,
    )
    assert implicit.to_config() == explicit.to_config()
    restored = CompositionalFeatureLearner.from_config(implicit.to_config())
    assert restored.to_config() == implicit.to_config()
    legacy_config = implicit.to_config()
    legacy_config.pop("candidate_novelty_admission_bonus")
    legacy_config.pop("ancestor_utility_backup_decay")
    legacy_restored = CompositionalFeatureLearner.from_config(legacy_config)
    assert legacy_restored.to_config()["candidate_novelty_admission_bonus"] == 0.0
    assert legacy_restored.to_config()["ancestor_utility_backup_decay"] == 0.0

    state = implicit.init(feature_dim=2, key=jr.key(7))
    observation = jnp.array([0.25, -0.5], dtype=jnp.float32)
    target = jnp.array([0.75], dtype=jnp.float32)
    implicit_result = implicit.update(state, observation, target)
    explicit_result = explicit.update(state, observation, target)
    chex.assert_trees_all_equal(implicit_result, explicit_result)


def test_opt_in_ranking_refuses_bad_topology_and_nonfinite_input_atomically() -> None:
    learner = _retention_learner(0.5)
    # One valid update canonicalizes host timing metadata into JAX leaves, so
    # subsequent exact rollback comparison covers the complete PyTree.
    initial = _retention_state(learner)
    state = learner.update(
        initial,
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
    ).state
    invalid_topology = cast(
        CompositionalFeatureState,
        state.replace(
            parent_a=state.parent_a.at[2].set(2)
        ),
    )
    diagnostics = learner.ranking_diagnostics(invalid_topology, feature_dim=2)
    assert not bool(diagnostics.contract_valid)

    bad_topology_result = learner.update(
        invalid_topology,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(bad_topology_result.state, invalid_topology)
    assert int(bad_topology_result.replaced_slot) == -1
    assert int(bad_topology_result.curation_trace.logical_event_count) == 0

    aliased_raw = cast(
        CompositionalFeatureState,
        state.replace(
            parent_a=state.parent_a.at[1].set(0)
        ),
    )
    assert not bool(learner.ranking_diagnostics(aliased_raw, feature_dim=2).contract_valid)
    aliased_raw_result = learner.update(
        aliased_raw,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(aliased_raw_result.state, aliased_raw)
    assert int(aliased_raw_result.curation_trace.logical_event_count) == 0

    nonfinite_result = learner.update(
        state,
        jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(nonfinite_result.state, state)
    assert int(nonfinite_result.replaced_slot) == -1
    assert int(nonfinite_result.curation_trace.logical_event_count) == 0


def test_novelty_admission_refuses_nonfinite_candidate_state() -> None:
    learner = _novelty_learner(1.0)
    initial = _novelty_state(learner)
    state = learner.update(
        initial,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    ).state
    corrupted = cast(
        CompositionalFeatureState,
        state.replace(
            candidate_utilities=jnp.array([jnp.nan], dtype=jnp.float32)
        ),
    )

    result = learner.update(
        corrupted,
        jnp.array([1.0, -1.0], dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )

    assert bool(jnp.isnan(result.state.candidate_utilities[0]))
    np.testing.assert_array_equal(
        np.asarray(result.state.step_words), np.asarray(corrupted.step_words)
    )
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(result.state.key)),
        np.asarray(jr.key_data(corrupted.key)),
    )
    assert int(result.promoted_candidate) == -1
    assert int(result.curation_trace.logical_event_count) == 0
