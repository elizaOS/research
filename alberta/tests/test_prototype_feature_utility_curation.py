# mypy: disable-error-code="attr-defined,call-arg,operator"
"""Unit contracts for stateless Prototype feature-utility audit ranking."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.interaction_features import (
    CURATION_ACTIVE_INELIGIBLE_RANK,
    CURATION_CANDIDATE_INELIGIBLE_RANK,
    InteractionCurationPriorityOverride,
)
from alberta_framework.core.prototype_feature_utility import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY as AUDITOR_CURATION_AUTHORITY,
)
from alberta_framework.core.prototype_feature_utility import (
    PrototypeFeatureUtilityAuditor,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityState,
)
from alberta_framework.core.prototype_feature_utility_curation import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA,
    PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_MECHANISM_STATUS,
    PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE,
    PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureUtilityCurationConfig,
    PrototypeFeatureUtilityCurationPolicy,
)

pytestmark = pytest.mark.unit


def _config(
    *,
    active_pair_slots: int = 2,
    candidate_pair_slots: int = 3,
) -> PrototypeFeatureUtilityConfig:
    return PrototypeFeatureUtilityConfig(
        base_feature_dim=4,
        active_pair_slots=active_pair_slots,
        candidate_pair_slots=candidate_pair_slots,
        managed_horde_demons=2,
        utility_decay=0.75,
        shadow_step_size=0.2,
        second_moment_decay=0.5,
        scale_epsilon=1.0e-6,
        max_observations=20,
    )


def _state(
    config: PrototypeFeatureUtilityConfig | None = None,
) -> PrototypeFeatureUtilityState:
    utility_config = _config() if config is None else config
    auditor = PrototypeFeatureUtilityAuditor(utility_config)
    active = jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32)[
        : utility_config.active_pair_slots
    ]
    # Slot one deliberately collides with active slot zero. The auditor's state
    # invariant requires every audit value/count for that candidate to stay zero.
    candidates = jnp.asarray([[0, 2], [0, 1], [2, 3]], dtype=jnp.int32)[
        : utility_config.candidate_pair_slots
    ]
    state = auditor.init(
        active_descriptors=active,
        candidate_descriptors=candidates,
        semantic_generation=jnp.asarray(7, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray([0, 7], dtype=jnp.uint32),
    )
    active_utilities = jnp.asarray(
        [[0.2, 0.8], [0.4, 0.0], [0.6, 0.4]],
        dtype=jnp.float32,
    )[:, : utility_config.active_pair_slots]
    active_counts = jnp.full_like(active_utilities, 3, dtype=jnp.int32)
    candidate_utilities = jnp.asarray(
        [[0.9, 0.0, 0.4], [0.1, 0.0, 0.8], [0.5, 0.0, 0.2]],
        dtype=jnp.float32,
    )[:, : utility_config.candidate_pair_slots]
    candidate_counts = jnp.asarray(
        [[3, 0, 3], [3, 0, 3], [3, 0, 3]],
        dtype=jnp.int32,
    )[:, : utility_config.candidate_pair_slots]
    return cast(
        PrototypeFeatureUtilityState,
        state.replace(
            observation_count=jnp.asarray(3, dtype=jnp.int32),
            observation_words=jnp.asarray([0, 3], dtype=jnp.uint32),
            active_task_utilities=active_utilities,
            active_task_evidence_counts=active_counts,
            candidate_task_utilities=candidate_utilities,
            candidate_task_evidence_counts=candidate_counts,
        ),
    )


def _policy(
    utility_config: PrototypeFeatureUtilityConfig,
    *,
    minimum_task_evidence: int = 3,
) -> PrototypeFeatureUtilityCurationPolicy:
    return PrototypeFeatureUtilityCurationPolicy(
        utility_config,
        PrototypeFeatureUtilityCurationConfig(
            minimum_task_evidence=minimum_task_evidence
        ),
    )


def _rank(
    policy: PrototypeFeatureUtilityCurationPolicy,
    state: PrototypeFeatureUtilityState,
    **overrides: Any,
) -> Any:
    source_generation = overrides.get(
        "source_semantic_generation",
        state.semantic_generation,
    )
    source_generation_words = overrides.get("source_semantic_generation_words")
    if source_generation_words is None:
        source_generation_words = (
            state.semantic_generation_words
            if "source_semantic_generation" not in overrides
            else jnp.asarray([0, int(source_generation)], dtype=jnp.uint32)
        )
    return policy.rank(
        state,
        source_semantic_generation=source_generation,
        source_semantic_generation_words=source_generation_words,
        source_active_descriptors=overrides.get(
            "source_active_descriptors",
            state.active_descriptors,
        ),
        source_candidate_descriptors=overrides.get(
            "source_candidate_descriptors",
            state.candidate_descriptors,
        ),
    )


def _assert_tree_floats_finite(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            assert np.all(np.isfinite(array))


def test_fixed_mass_aggregate_ranks_and_collision_mask() -> None:
    config = _config()
    state = _state(config)
    policy = _policy(config)

    result = _rank(policy, state)

    assert isinstance(result.override, InteractionCurationPriorityOverride)
    assert bool(result.override.enabled)
    np.testing.assert_allclose(
        result.override.active_ranks,
        jnp.asarray([0.35, 0.5], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        result.override.candidate_ranks[jnp.asarray([0, 2], dtype=jnp.int32)],
        jnp.asarray([0.6, 0.45], dtype=jnp.float32),
    )
    assert np.isfinite(float(result.override.candidate_ranks[1]))
    assert float(result.override.candidate_ranks[1]) == (
        CURATION_CANDIDATE_INELIGIBLE_RANK
    )
    np.testing.assert_array_equal(
        result.diagnostics.candidate_collision_mask,
        jnp.asarray([False, True, False]),
    )
    np.testing.assert_array_equal(
        result.diagnostics.candidate_rank_ready_mask,
        jnp.asarray([True, False, True]),
    )
    assert bool(result.diagnostics.override_enabled)
    assert bool(result.diagnostics.curation_ready)
    _assert_tree_floats_finite(result.override)


def test_capacity_cap_is_valid_but_returns_a_neutral_no_ranking_result() -> None:
    config = _config()
    state = _state(config)
    state = cast(
        PrototypeFeatureUtilityState,
        state.replace(
            observation_count=jnp.asarray(20, dtype=jnp.int32),
            observation_words=jnp.asarray([0, 20], dtype=jnp.uint32),
        ),
    )

    result = _rank(
        _policy(config),
        state,
    )

    assert not bool(result.override.enabled)
    assert bool(result.diagnostics.transaction_valid)
    assert bool(result.diagnostics.available)
    assert not bool(result.diagnostics.curation_ready)
    assert bool(result.diagnostics.observation_capacity_valid)
    assert not bool(result.diagnostics.observation_capacity_available)
    assert bool(result.diagnostics.observation_capacity_capped)
    assert int(result.diagnostics.observation_count) == 20
    assert int(result.diagnostics.maximum_observations) == 20
    np.testing.assert_array_equal(result.override.active_ranks, jnp.zeros((2,)))
    np.testing.assert_array_equal(result.override.candidate_ranks, jnp.zeros((3,)))


def test_strict_all_task_floor_masks_slot_without_renormalizing_mass() -> None:
    config = _config()
    state = _state(config)
    active_counts = state.active_task_evidence_counts.at[1, 0].set(2)
    state = cast(
        PrototypeFeatureUtilityState,
        state.replace(
            active_task_evidence_counts=active_counts,
        ),
    )
    policy = _policy(config)

    result = _rank(policy, state)

    assert bool(result.override.enabled)
    assert bool(result.diagnostics.transaction_valid)
    np.testing.assert_array_equal(
        result.diagnostics.active_all_tasks_evidence_ready,
        jnp.asarray([False, True]),
    )
    # The raw EMA aggregate retains the configured 0.5/0.25/0.25 mass even
    # though task one is below the evidence floor. Excluding that task and
    # renormalizing the remaining 0.75 mass would instead produce 1/3.
    assert float(
        result.diagnostics.raw_active_fixed_mass_utilities[0]
    ) == pytest.approx(0.35)
    assert float(
        result.diagnostics.raw_active_fixed_mass_utilities[0]
    ) != pytest.approx(1.0 / 3.0)
    assert float(result.override.active_ranks[0]) == CURATION_ACTIVE_INELIGIBLE_RANK
    assert bool(result.diagnostics.curation_ready)


@pytest.mark.parametrize("missing_side", ["active", "candidate"])
def test_valid_override_uses_sentinels_until_both_sides_are_ready(
    missing_side: str,
) -> None:
    config = _config()
    state = _state(config)
    if missing_side == "active":
        state = cast(
            PrototypeFeatureUtilityState,
            state.replace(
                active_task_evidence_counts=jnp.zeros_like(
                    state.active_task_evidence_counts
                )
            ),
        )
    else:
        state = cast(
            PrototypeFeatureUtilityState,
            state.replace(
                candidate_task_evidence_counts=jnp.zeros_like(
                    state.candidate_task_evidence_counts
                )
            ),
        )

    result = _rank(
        _policy(config),
        state,
    )

    assert bool(result.override.enabled)
    assert bool(result.diagnostics.transaction_valid)
    assert not bool(result.diagnostics.curation_ready)
    if missing_side == "active":
        np.testing.assert_array_equal(
            result.override.active_ranks,
            jnp.full((2,), CURATION_ACTIVE_INELIGIBLE_RANK, dtype=jnp.float32),
        )
    else:
        np.testing.assert_array_equal(
            result.override.candidate_ranks,
            jnp.full(
                (3,),
                CURATION_CANDIDATE_INELIGIBLE_RANK,
                dtype=jnp.float32,
            ),
        )


def test_stale_generation_and_same_generation_forks_fail_closed() -> None:
    config = _config()
    state = _state(config)
    policy = _policy(config)
    newer_generation = _rank(
        policy,
        state,
        source_semantic_generation=state.semantic_generation + 1,
    )
    older_generation = _rank(
        policy,
        state,
        source_semantic_generation=state.semantic_generation - 1,
    )
    forked_active = state.active_descriptors.at[1].set(
        jnp.asarray([1, 3], dtype=jnp.int32)
    )
    same_generation_fork = _rank(
        policy,
        state,
        source_active_descriptors=forked_active,
    )
    forked_candidate = state.candidate_descriptors.at[2].set(
        jnp.asarray([1, 3], dtype=jnp.int32)
    )
    same_generation_candidate_fork = _rank(
        policy,
        state,
        source_candidate_descriptors=forked_candidate,
    )

    assert not bool(newer_generation.override.enabled)
    assert not bool(newer_generation.diagnostics.transaction_valid)
    assert bool(newer_generation.diagnostics.stale_state_generation)
    assert not bool(newer_generation.diagnostics.source_binding_valid)
    assert not bool(older_generation.override.enabled)
    assert not bool(older_generation.diagnostics.transaction_valid)
    assert bool(older_generation.diagnostics.stale_source_generation)
    assert not bool(same_generation_fork.override.enabled)
    assert bool(same_generation_fork.diagnostics.same_generation_descriptor_fork)
    assert not bool(same_generation_fork.diagnostics.source_binding_valid)
    assert not bool(same_generation_candidate_fork.override.enabled)
    assert bool(
        same_generation_candidate_fork.diagnostics.same_generation_descriptor_fork
    )
    np.testing.assert_array_equal(
        same_generation_fork.override.active_ranks,
        jnp.zeros((2,), dtype=jnp.float32),
    )


@pytest.mark.parametrize("invalid_field", ["nonfinite_utility", "negative_count"])
def test_invalid_dynamic_state_fails_closed_without_rank_fallback(
    invalid_field: str,
) -> None:
    config = _config()
    state = _state(config)
    if invalid_field == "nonfinite_utility":
        state = cast(
            PrototypeFeatureUtilityState,
            state.replace(
                active_task_utilities=state.active_task_utilities.at[0, 0].set(
                    jnp.nan
                )
            ),
        )
    else:
        state = cast(
            PrototypeFeatureUtilityState,
            state.replace(
                active_task_evidence_counts=(
                    state.active_task_evidence_counts.at[0, 0].set(-1)
                )
            ),
        )

    result = _rank(
        _policy(config),
        state,
    )

    assert not bool(result.override.enabled)
    assert not bool(result.diagnostics.transaction_valid)
    assert not bool(result.diagnostics.state_valid)
    np.testing.assert_array_equal(result.override.active_ranks, jnp.zeros((2,)))
    np.testing.assert_array_equal(result.override.candidate_ranks, jnp.zeros((3,)))
    _assert_tree_floats_finite(result.override)
    if invalid_field == "nonfinite_utility":
        assert not bool(result.diagnostics.rank_values_finite)
        assert bool(jnp.isnan(result.diagnostics.raw_active_fixed_mass_utilities[0]))


def test_rank_is_jittable_and_has_a_fixed_pytree() -> None:
    config = _config()
    state = _state(config)
    policy = _policy(config)

    eager = _rank(policy, state)
    compiled = jax.jit(policy.rank)(
        state,
        source_semantic_generation=state.semantic_generation,
        source_semantic_generation_words=state.semantic_generation_words,
        source_active_descriptors=state.active_descriptors,
        source_candidate_descriptors=state.candidate_descriptors,
    )

    assert jax.tree_util.tree_structure(compiled) == jax.tree_util.tree_structure(eager)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compiled),
        jax.tree_util.tree_leaves(eager),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_resource_and_authority_declarations_are_narrow() -> None:
    config = _config()
    policy = _policy(config)

    budget = policy.resource_budget()

    assert budget.persistent_logical_scalars == 0
    assert budget.persistent_state_nbytes == 0
    assert budget.task_evidence_cells_per_rank == config.n_tasks * (
        config.active_pair_slots + config.candidate_pair_slots
    )
    assert (
        budget.task_aggregate_cells_per_rank
        == budget.task_evidence_cells_per_rank
    )
    assert budget.rng_draws_per_rank == 0
    assert budget.backward_passes_per_rank == 0
    assert budget.consumer_updates_per_rank == 0
    assert budget.router_calls_per_rank == 0
    assert budget.curation_decisions_per_rank == 0
    assert budget.ranking_influence is True
    assert budget.curation_authority is False
    assert budget.promotion_authority is False
    assert budget.go_no_go_authority is False
    assert budget.scientific_promotion_allowed is False
    assert budget.mechanism_status == PROTOTYPE_FEATURE_UTILITY_CURATION_MECHANISM_STATUS
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE is True
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert AUDITOR_CURATION_AUTHORITY is False


def test_zero_candidate_boundary_is_valid_and_jittable() -> None:
    config = _config(candidate_pair_slots=0)
    state = _state(config)
    policy = _policy(config)

    eager = _rank(policy, state)
    compiled = jax.jit(policy.rank)(
        state,
        source_semantic_generation=state.semantic_generation,
        source_semantic_generation_words=state.semantic_generation_words,
        source_active_descriptors=state.active_descriptors,
        source_candidate_descriptors=state.candidate_descriptors,
    )

    assert bool(eager.override.enabled)
    assert bool(eager.diagnostics.transaction_valid)
    assert not bool(eager.diagnostics.any_candidate_rank_ready)
    assert not bool(eager.diagnostics.curation_ready)
    assert eager.override.candidate_ranks.shape == (0,)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compiled),
        jax.tree_util.tree_leaves(eager),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("floor", [True, 0, -1, 1.0, "3"])
def test_evidence_floor_is_a_positive_exact_int(floor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        PrototypeFeatureUtilityCurationConfig(
            minimum_task_evidence=cast(Any, floor)
        )


def test_evidence_floor_cannot_exceed_auditor_lifetime() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        PrototypeFeatureUtilityCurationPolicy(
            _config(),
            PrototypeFeatureUtilityCurationConfig(minimum_task_evidence=21),
        )


def test_curation_config_round_trip_is_strict_and_versioned() -> None:
    config = PrototypeFeatureUtilityCurationConfig(minimum_task_evidence=3)
    payload = config.to_config()

    assert payload == {
        "schema_version": PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA,
        "minimum_task_evidence": 3,
    }
    assert PrototypeFeatureUtilityCurationConfig.from_config(payload) == config
    with pytest.raises(ValueError, match="keys differ"):
        PrototypeFeatureUtilityCurationConfig.from_config(
            {**payload, "unknown": 1}
        )
    with pytest.raises(ValueError, match="schema_version"):
        PrototypeFeatureUtilityCurationConfig.from_config(
            {**payload, "schema_version": "future"}
        )


def test_static_source_contract_drift_raises() -> None:
    config = _config()
    state = _state(config)
    policy = _policy(config)

    with pytest.raises(ValueError, match="source_active_descriptors"):
        policy.rank(
            state,
            source_semantic_generation=state.semantic_generation,
            source_semantic_generation_words=state.semantic_generation_words,
            source_active_descriptors=state.active_descriptors[:1],
            source_candidate_descriptors=state.candidate_descriptors,
        )
    with pytest.raises(TypeError, match="source_semantic_generation"):
        policy.rank(
            state,
            source_semantic_generation=state.semantic_generation.astype(jnp.float32),
            source_semantic_generation_words=state.semantic_generation_words,
            source_active_descriptors=state.active_descriptors,
            source_candidate_descriptors=state.candidate_descriptors,
        )
