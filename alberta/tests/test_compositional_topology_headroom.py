"""Exact defaults-off topology-headroom admission contracts."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import compositional_features as cf

pytestmark = pytest.mark.unit


def _learner(*, reserve: bool, max_depth: int = 3) -> cf.CompositionalFeatureLearner:
    return cf.CompositionalFeatureLearner(
        n_features=6,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0e6,
        max_depth=max_depth,
        topology_headroom_reserve=reserve,
        use_obgd=False,
    )


def _state(
    learner: cf.CompositionalFeatureLearner,
    *,
    candidate_parent_a: int,
    candidate_parent_b: int,
    candidate_depth: int,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=2, key=jr.key(2601))
    return cast(
        cf.CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                (
                    cf.OP_RAW,
                    cf.OP_RAW,
                    cf.OP_PRODUCT,
                    cf.OP_PRODUCT,
                    cf.OP_PRODUCT,
                    cf.OP_PRODUCT,
                ),
                dtype=jnp.int32,
            ),
            parent_a=jnp.asarray((0, 1, 0, 0, 2, 4), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, 1, 1, 0, 1), dtype=jnp.int32),
            depth=jnp.asarray((0, 0, 1, 1, 2, 3), dtype=jnp.int32),
            ages=jnp.full((6,), 10, dtype=jnp.int32),
            utilities=jnp.asarray((10.0, 10.0, 1.0, 2.0, 3.0, 4.0)),
            candidate_ops=jnp.asarray((cf.OP_PRODUCT,), dtype=jnp.int32),
            candidate_parent_a=jnp.asarray((candidate_parent_a,), dtype=jnp.int32),
            candidate_parent_b=jnp.asarray((candidate_parent_b,), dtype=jnp.int32),
            candidate_depth=jnp.asarray((candidate_depth,), dtype=jnp.int32),
            candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
            candidate_utilities=jnp.asarray((100.0,), dtype=jnp.float32),
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )


def _update(
    learner: cf.CompositionalFeatureLearner,
    state: cf.CompositionalFeatureState,
) -> cf.CompositionalFeatureUpdateResult:
    return learner.update(
        state,
        jnp.asarray((1.0, -1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if hasattr(left_leaf, "dtype") and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            left_leaf.dtype, jax.dtypes.prng_key
        ):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_depth_one_candidate_reserves_every_remaining_topology_level() -> None:
    learner = _learner(reserve=True)
    result = _update(
        learner,
        _state(
            learner,
            candidate_parent_a=0,
            candidate_parent_b=1,
            candidate_depth=1,
        ),
    )
    trace = result.curation_trace

    np.testing.assert_array_equal(
        trace.decision_candidate_topology_compatible,
        np.asarray(((False, False, True, True, True, True),)),
    )
    np.testing.assert_array_equal(
        trace.decision_candidate_headroom_compatible,
        np.asarray(((True, True, True, True, False, False),)),
    )
    np.testing.assert_array_equal(
        trace.decision_candidate_destination_compatible,
        np.asarray(((False, False, True, True, False, False),)),
    )


def test_max_depth_candidate_loses_no_destination_beyond_existing_topology() -> None:
    reserve = _learner(reserve=True)
    legacy = _learner(reserve=False)
    reserve_result = _update(
        reserve,
        _state(
            reserve,
            candidate_parent_a=4,
            candidate_parent_b=0,
            candidate_depth=3,
        ),
    )
    legacy_result = _update(
        legacy,
        _state(
            legacy,
            candidate_parent_a=4,
            candidate_parent_b=0,
            candidate_depth=3,
        ),
    )

    np.testing.assert_array_equal(
        reserve_result.curation_trace.decision_candidate_headroom_compatible,
        np.ones((1, 6), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        reserve_result.curation_trace.decision_candidate_destination_compatible,
        legacy_result.curation_trace.decision_candidate_destination_compatible,
    )


def test_flag_is_strict_round_tripped_and_false_path_is_bit_exact() -> None:
    with pytest.raises(TypeError, match="topology_headroom_reserve"):
        cf.CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            topology_headroom_reserve=cast(Any, 1),
        )

    enabled = _learner(reserve=True)
    assert enabled.to_config()["topology_headroom_reserve"] is True
    assert cf.CompositionalFeatureLearner.from_config(
        enabled.to_config()
    ).to_config() == enabled.to_config()

    default = cf.CompositionalFeatureLearner(
        n_features=6,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0e6,
        max_depth=3,
        use_obgd=False,
    )
    explicit = _learner(reserve=False)
    assert default.to_config() == explicit.to_config()
    default_state = _state(
        default,
        candidate_parent_a=0,
        candidate_parent_b=1,
        candidate_depth=1,
    )
    explicit_state = _state(
        explicit,
        candidate_parent_a=0,
        candidate_parent_b=1,
        candidate_depth=1,
    )
    _assert_tree_equal(default_state, explicit_state)
    _assert_tree_equal(
        _update(default, default_state),
        _update(explicit, explicit_state),
    )
    np.testing.assert_array_equal(
        _update(default, default_state)
        .curation_trace.decision_candidate_headroom_compatible,
        np.ones((1, 6), dtype=np.bool_),
    )


def test_impossible_headroom_is_nonadmission_without_topology_corruption() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        topology_headroom_reserve=True,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(2602)).replace(  # type: ignore[attr-defined]
        ages=jnp.full((4,), 10, dtype=jnp.int32),
        candidate_ops=jnp.asarray((cf.OP_PRODUCT,), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0,), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1,), dtype=jnp.int32),
        candidate_depth=jnp.asarray((1,), dtype=jnp.int32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((100.0,), dtype=jnp.float32),
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    result = _update(learner, state)
    trace = result.curation_trace

    assert not bool(trace.decision_candidate_has_destination[0])
    assert int(trace.decision_selected_candidate) == -1
    assert not bool(trace.decision_should_promote)
    assert not bool(trace.promotion_applied)
    assert not bool(jnp.any(trace.root_change_mask))
    np.testing.assert_array_equal(result.state.ops, state.ops)
    np.testing.assert_array_equal(result.state.parent_a, state.parent_a)
    np.testing.assert_array_equal(result.state.parent_b, state.parent_b)
    assert bool(learner._ranking_topology_valid(result.state, 2))  # noqa: SLF001


def _leftpack_learner(*, enabled: bool, max_depth: int = 3) -> cf.CompositionalFeatureLearner:
    return cf.CompositionalFeatureLearner(
        n_features=6,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        max_depth=max_depth,
        topology_headroom_reserve=True,
        topology_left_pack_destinations=enabled,
        use_obgd=False,
    )


def _placement_state(
    learner: cf.CompositionalFeatureLearner,
    *,
    active_slot_2_utility: float,
    active_slot_3_utility: float,
    active_slot_5_utility: float = 100.0,
    candidate_utility: float = 10.0,
    candidate_parent_a: int = 0,
    candidate_parent_b: int = 1,
    candidate_depth: int = 1,
) -> cf.CompositionalFeatureState:
    state = _state(
        learner,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_depth=candidate_depth,
    )
    return cast(
        cf.CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            utilities=jnp.asarray(
                (
                    100.0,
                    100.0,
                    active_slot_2_utility,
                    active_slot_3_utility,
                    100.0,
                    active_slot_5_utility,
                ),
                dtype=jnp.float32,
            ),
            candidate_utilities=jnp.asarray(
                (candidate_utility,), dtype=jnp.float32
            ),
        ),
    )


def test_leftpack_selects_lowest_margin_eligible_destination() -> None:
    learner = _leftpack_learner(enabled=True)
    result = _update(
        learner,
        _placement_state(
            learner,
            active_slot_2_utility=4.0,
            active_slot_3_utility=1.0,
        ),
    )
    trace = result.curation_trace

    np.testing.assert_array_equal(
        trace.decision_candidate_margin_eligible,
        np.asarray(((False, False, True, True, False, False),)),
    )
    assert bool(trace.decision_left_pack_destinations_enabled)
    assert bool(trace.decision_left_pack_destination_available)
    assert int(trace.decision_selected_destination) == 2
    assert int(result.replaced_slot) == 2


def test_leftpack_skips_low_index_destination_that_fails_strict_margin() -> None:
    learner = _leftpack_learner(enabled=True)
    result = _update(
        learner,
        _placement_state(
            learner,
            active_slot_2_utility=20.0,
            active_slot_3_utility=1.0,
        ),
    )

    np.testing.assert_array_equal(
        result.curation_trace.decision_candidate_margin_eligible,
        np.asarray(((False, False, False, True, False, False),)),
    )
    assert int(result.curation_trace.decision_selected_destination) == 3
    assert int(result.replaced_slot) == 3


def test_leftpack_with_no_margin_eligible_destination_does_not_promote() -> None:
    learner = _leftpack_learner(enabled=True)
    result = _update(
        learner,
        _placement_state(
            learner,
            active_slot_2_utility=20.0,
            active_slot_3_utility=15.0,
        ),
    )
    trace = result.curation_trace

    assert int(trace.decision_selected_candidate) == 0
    assert int(trace.decision_selected_destination) == -1
    assert not bool(jnp.any(trace.decision_candidate_margin_eligible))
    assert not bool(trace.decision_left_pack_destination_available)
    assert not bool(trace.decision_margin_passed)
    assert not bool(trace.decision_should_promote)
    assert not bool(trace.promotion_applied)
    assert int(result.replaced_slot) == -1


def test_leftpack_max_depth_candidate_keeps_existing_topological_destination() -> None:
    learner = _leftpack_learner(enabled=True)
    result = _update(
        learner,
        _placement_state(
            learner,
            active_slot_2_utility=100.0,
            active_slot_3_utility=100.0,
            active_slot_5_utility=1.0,
            candidate_parent_a=4,
            candidate_parent_b=0,
            candidate_depth=3,
        ),
    )

    np.testing.assert_array_equal(
        result.curation_trace.decision_candidate_headroom_compatible,
        np.ones((1, 6), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        result.curation_trace.decision_candidate_margin_eligible,
        np.asarray(((False, False, False, False, False, True),)),
    )
    assert int(result.replaced_slot) == 5


def test_leftpack_flag_is_strict_round_tripped_and_false_path_is_exact() -> None:
    with pytest.raises(TypeError, match="topology_left_pack_destinations"):
        cf.CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            topology_left_pack_destinations=cast(Any, 1),
        )

    enabled = _leftpack_learner(enabled=True)
    assert enabled.to_config()["topology_left_pack_destinations"] is True
    assert cf.CompositionalFeatureLearner.from_config(
        enabled.to_config()
    ).to_config() == enabled.to_config()

    default = _learner(reserve=True)
    explicit = cf.CompositionalFeatureLearner(
        **{
            key: value
            for key, value in default.to_config().items()
            if key not in {"type", "topology_left_pack_destinations"}
        },
        topology_left_pack_destinations=False,
    )
    assert default.to_config() == explicit.to_config()
    default_state = _placement_state(
        default,
        active_slot_2_utility=4.0,
        active_slot_3_utility=1.0,
    )
    explicit_state = _placement_state(
        explicit,
        active_slot_2_utility=4.0,
        active_slot_3_utility=1.0,
    )
    _assert_tree_equal(default_state, explicit_state)
    _assert_tree_equal(
        _update(default, default_state),
        _update(explicit, explicit_state),
    )
