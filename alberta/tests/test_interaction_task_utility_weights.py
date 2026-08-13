"""L0 contracts for optional interaction-feature task utility weights."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
    load_interaction_feature_checkpoint,
    save_interaction_feature_checkpoint,
)

pytestmark = pytest.mark.unit

_GROUP_WEIGHTS = (0.5, 0.125, 0.125, 0.125, 0.125)


def _learner(**kwargs: Any) -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=5,
        utility_decay=0.0,
        replacement_interval=0,
        use_obgd=False,
        **kwargs,
    )


def _utility_after_one_update(
    learner: FixedBudgetInteractionLearner,
    output_weights: jax.Array,
    targets: jax.Array | None = None,
) -> jax.Array:
    state = learner.init(feature_dim=2, key=jr.key(10))
    state = state.replace(output_weights=output_weights)
    if targets is None:
        targets = jnp.zeros((learner.n_tasks,), dtype=jnp.float32)
    result = learner.update(
        state,
        jnp.ones((2,), dtype=jnp.float32),
        targets,
    )
    assert not bool(result.update_rejected)
    return cast(jax.Array, result.state.utilities)


def _assert_inexact_tree_finite(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_task_utility_weights_have_an_exact_float32_safe_tuple_contract() -> None:
    class TupleSubclass(tuple[float, ...]):
        pass

    valid = _learner(task_utility_weights=(0.1, 0.2, 0.3, 0.2, 0.2))
    assert valid.to_config()["task_utility_weights"] == [0.1, 0.2, 0.3, 0.2, 0.2]

    cases = (
        (cast(object, [0.5, 0.125, 0.125, 0.125, 0.125]), "exact tuple", TypeError),
        (cast(object, TupleSubclass(_GROUP_WEIGHTS)), "exact tuple", TypeError),
        (cast(object, (0.5, 0.5)), "one entry per task", ValueError),
        (cast(object, (1, 0.0, 0.0, 0.0, 0.0)), "built-in floats", TypeError),
        (cast(object, (True, 0.0, 0.0, 0.0, 0.0)), "built-in floats", TypeError),
        (cast(object, (float("nan"), 1.0, 0.0, 0.0, 0.0)), "finite", ValueError),
        (cast(object, (float("inf"), 1.0, 0.0, 0.0, 0.0)), "finite", ValueError),
        (cast(object, (-0.1, 1.0, 0.0, 0.0, 0.0)), "nonnegative", ValueError),
        (cast(object, (0.0, 0.0, 0.0, 0.0, 0.0)), "positive mass", ValueError),
        (cast(object, (1.0e40, 0.0, 0.0, 0.0, 0.0)), "float32", ValueError),
        (cast(object, (1.0e-50, 0.0, 0.0, 0.0, 0.0)), "float32", ValueError),
        (cast(object, (3.0e38, 3.0e38, 0.0, 0.0, 0.0)), "float32", ValueError),
    )
    for raw_weights, match, error in cases:
        with pytest.raises(error, match=match):
            _learner(task_utility_weights=cast(tuple[float, ...] | None, raw_weights))


def test_task_utility_weights_round_trip_and_legacy_missing_field_defaults(
    tmp_path: Path,
) -> None:
    learner = _learner(
        task_utility_weights=_GROUP_WEIGHTS,
        utility_task_balancing="active_inverse_frequency",
        candidate_count=1,
        scale_robust=True,
    )
    payload = learner.to_config()

    assert payload["task_utility_weights"] == list(_GROUP_WEIGHTS)
    restored = FixedBudgetInteractionLearner.from_config(payload)
    assert restored.to_config() == payload

    state = learner.init(feature_dim=2, key=jr.key(13))
    checkpoint_path = tmp_path / "weighted"
    save_interaction_feature_checkpoint(
        learner,
        state,
        checkpoint_path,
        feature_dim=2,
    )
    loaded, loaded_state = load_interaction_feature_checkpoint(checkpoint_path)
    assert loaded.to_config() == payload
    chex.assert_trees_all_equal(loaded_state, state)

    legacy = _learner().to_config()
    legacy.pop("task_utility_weights")
    migrated = FixedBudgetInteractionLearner.from_config(legacy)
    assert migrated.to_config()["task_utility_weights"] is None

    malformed = dict(payload)
    malformed["task_utility_weights"] = tuple(_GROUP_WEIGHTS)
    with pytest.raises(TypeError, match="JSON list"):
        FixedBudgetInteractionLearner.from_config(malformed)


def test_default_none_path_is_tree_exact_and_keeps_legacy_mean() -> None:
    default = _learner()
    explicit = _learner(task_utility_weights=None)
    default_state = default.init(feature_dim=2, key=jr.key(11))
    explicit_state = explicit.init(feature_dim=2, key=jr.key(11))
    chex.assert_trees_all_equal(default_state, explicit_state)
    weights = jnp.asarray(((2.0,), (0.0,), (0.0,), (0.0,), (0.0,)), dtype=jnp.float32)
    default_state = default_state.replace(output_weights=weights)
    explicit_state = explicit_state.replace(output_weights=weights)
    observation = jnp.ones((2,), dtype=jnp.float32)
    targets = jnp.zeros((5,), dtype=jnp.float32)

    default_result = default.update(default_state, observation, targets)
    explicit_result = explicit.update(explicit_state, observation, targets)

    chex.assert_trees_all_equal(default_result, explicit_result)
    np.testing.assert_array_equal(default_result.state.utilities, np.asarray((0.4,), np.float32))
    assert default.to_config() == explicit.to_config()


def test_half_control_and_four_eighth_horde_weights_balance_group_utility() -> None:
    learner = _learner(task_utility_weights=_GROUP_WEIGHTS)
    control_only = jnp.asarray(((2.0,), (0.0,), (0.0,), (0.0,), (0.0,)), dtype=jnp.float32)
    horde_only = jnp.asarray(((0.0,), (2.0,), (2.0,), (2.0,), (2.0,)), dtype=jnp.float32)

    control_utility = _utility_after_one_update(learner, control_only)
    horde_utility = _utility_after_one_update(learner, horde_only)

    np.testing.assert_array_equal(control_utility, np.asarray((1.0,), dtype=np.float32))
    np.testing.assert_array_equal(horde_utility, np.asarray((1.0,), dtype=np.float32))


def test_weighted_mean_combines_active_and_inverse_frequency_factors() -> None:
    learner = _learner(
        task_utility_weights=_GROUP_WEIGHTS,
        utility_task_balancing="active_inverse_frequency",
        task_activity_decay=0.75,
    )
    output_weights = jnp.asarray(((1.0,), (1.0,), (9.0,), (9.0,), (9.0,)), dtype=jnp.float32)
    active_mask = jnp.asarray((True, True, False, False, False), dtype=jnp.bool_)
    activity = jnp.asarray((0.5, 0.25, 0.25, 0.25, 0.25), dtype=jnp.float32)

    current = learner._utility_signal(
        output_weights,
        jnp.ones((1,), dtype=jnp.float32),
        active_mask,
        activity,
    )
    scale_robust = learner._aggregate_task_feature_signal(
        jnp.abs(output_weights),
        active_mask,
        activity,
    )

    # ((.5 * 1 / .5) + (.125 * 1 / .25)) / (.5 + .125) == 2.4.
    expected = np.asarray((2.4,), dtype=np.float32)
    np.testing.assert_allclose(current, expected, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(scale_robust, expected, rtol=0.0, atol=1.0e-7)


def test_weighted_max_excludes_zero_weight_and_inactive_tasks_safely() -> None:
    zero_weighted = _learner(
        task_utility_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        utility_aggregation="max",
    )
    values = jnp.asarray(((2.0,), (100.0,), (100.0,), (100.0,), (100.0,)), dtype=jnp.float32)
    np.testing.assert_array_equal(
        _utility_after_one_update(zero_weighted, values),
        np.asarray((2.0,), dtype=np.float32),
    )

    active = _learner(
        task_utility_weights=(0.5, 0.5, 0.0, 0.0, 0.0),
        utility_aggregation="max",
        utility_task_balancing="active",
    )
    targets = jnp.asarray((jnp.nan, 0.0, jnp.nan, jnp.nan, jnp.nan), dtype=jnp.float32)
    np.testing.assert_array_equal(
        _utility_after_one_update(active, values, targets),
        np.asarray((100.0,), dtype=np.float32),
    )

    all_inactive = jnp.full((5,), jnp.nan, dtype=jnp.float32)
    np.testing.assert_array_equal(
        _utility_after_one_update(active, values, all_inactive),
        np.asarray((0.0,), dtype=np.float32),
    )


def test_weighted_topk_normalizes_only_selected_positive_weight_mass() -> None:
    learner = _learner(
        task_utility_weights=(0.5, 0.25, 0.25, 0.0, 0.0),
        utility_aggregation="topk",
        utility_top_k=2,
    )
    # Weighted ranks are 2.0, 1.0, 0.5, ineligible, ineligible. The selected
    # weighted signal is 3.0 and selected weight mass is .75.
    values = jnp.asarray(((4.0,), (4.0,), (2.0,), (100.0,), (100.0,)), dtype=jnp.float32)

    np.testing.assert_array_equal(
        _utility_after_one_update(learner, values),
        np.asarray((4.0,), dtype=np.float32),
    )

    sparse = _learner(
        task_utility_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        utility_aggregation="topk",
        utility_top_k=3,
        utility_task_balancing="active",
    )
    single_active = jnp.asarray(
        (0.0, jnp.nan, jnp.nan, jnp.nan, jnp.nan),
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        _utility_after_one_update(sparse, values, single_active),
        np.asarray((4.0,), dtype=np.float32),
    )
    all_inactive = jnp.full((5,), jnp.nan, dtype=jnp.float32)
    np.testing.assert_array_equal(
        _utility_after_one_update(sparse, values, all_inactive),
        np.asarray((0.0,), dtype=np.float32),
    )


def test_weighted_update_is_eager_jit_and_scan_compatible() -> None:
    learner = _learner(
        task_utility_weights=_GROUP_WEIGHTS,
        utility_task_balancing="active_inverse_frequency",
        task_activity_decay=0.5,
    )
    state = learner.init(feature_dim=2, key=jr.key(12))
    observation = jnp.asarray((0.5, -2.0), dtype=jnp.float32)
    target = jnp.asarray((1.0, 0.5, jnp.nan, -0.5, jnp.nan), dtype=jnp.float32)

    eager = learner.update(state, observation, target)
    compiled = jax.jit(learner.update)(state, observation, target)
    chex.assert_trees_all_equal(eager, compiled)

    observations = jnp.stack((observation, observation * 0.5, observation * -1.0))
    targets = jnp.stack((target, target, target))

    def scan(
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[InteractionFeatureState, jax.Array]:
        def body(
            carry: InteractionFeatureState,
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[InteractionFeatureState, jax.Array]:
            one_observation, one_target = inputs
            result = learner.update(carry, one_observation, one_target)
            return result.state, result.state.utilities

        return jax.lax.scan(body, state, values)

    final_state, utilities = jax.jit(scan)((observations, targets))

    _assert_inexact_tree_finite(final_state)
    _assert_inexact_tree_finite(utilities)
    chex.assert_shape(utilities, (3, 1))
    assert int(final_state.step_count) == 3
