"""Contracts for bounded, causal, consumer-specific learning-value routing."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.delight import LearningValue, LearningValueAvailability
from alberta_framework.core.learning_value_router import (
    LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
    LEARNING_VALUE_ROUTER_CONFIG_SCHEMA,
    MECHANISM_STATUS,
    AdaptationChangeLearningValueRoute,
    ExplorationLearningValueRoute,
    LearningValueRouter,
    LearningValueRouterConfig,
    LearningValueRouterState,
    LiteralGradientJoyEvidenceRoute,
    ModelMemoryReplayLearningValueRoute,
    PaperDGActorRoute,
    SafetyLearningValueRoute,
)

pytestmark = pytest.mark.unit

_NAMES = (
    "advantage",
    "action_surprisal",
    "delight",
    "epistemic_surprise",
    "aleatoric_uncertainty",
    "learning_progress",
    "change_probability",
    "safety_cost",
)

def _replace[T](value: T, **changes: object) -> T:
    """Use chex's immutable replacement method without losing static type."""

    return cast(T, cast(Any, value).replace(**changes))


def _value(**overrides: float) -> LearningValue:
    values = {
        "advantage": 2.0,
        "action_surprisal": 3.0,
        "delight": 6.0,
        "epistemic_surprise": 4.0,
        "aleatoric_uncertainty": 5.0,
        "learning_progress": -1.0,
        "change_probability": 0.25,
        "safety_cost": 0.75,
    }
    values.update(overrides)
    return LearningValue(
        **{name: jnp.asarray(values[name], dtype=jnp.float32) for name in _NAMES}
    )


def _availability(**overrides: bool) -> LearningValueAvailability:
    values = dict.fromkeys(_NAMES, True)
    values.update(overrides)
    return LearningValueAvailability(
        **{name: jnp.asarray(values[name], dtype=jnp.bool_) for name in _NAMES}
    )


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_tree_equal(first: object, second: object) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert str(first_structure) == str(second_structure)
    assert len(first_leaves) == len(second_leaves)
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def _assert_tree_allclose(first: object, second: object) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert str(first_structure) == str(second_structure)
    assert len(first_leaves) == len(second_leaves)
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(first_leaf),
            np.asarray(second_leaf),
            rtol=1e-6,
            atol=1e-7,
        )


def _assert_route_mask(
    route: object,
    expected_active: set[str],
    expected_values: LearningValue,
) -> None:
    for name in _NAMES:
        active = name in expected_active
        assert bool(getattr(route.availability, name)) is active  # type: ignore[attr-defined]
        actual = np.asarray(getattr(route.values, name))  # type: ignore[attr-defined]
        expected = np.asarray(getattr(expected_values, name)) if active else np.asarray(0.0)
        np.testing.assert_array_equal(actual, expected)
        assert not bool(getattr(route.normalized_availability, name))  # type: ignore[attr-defined]
        np.testing.assert_array_equal(
            np.asarray(getattr(route.normalized_values, name)),  # type: ignore[attr-defined]
            0.0,
        )


def test_config_roundtrip_is_strict_exact_and_json_compatible() -> None:
    config = LearningValueRouterConfig(
        normalization_min_count=4,
        max_steps=1234,
        max_abs_advantage=11.0,
        max_action_surprisal=12.0,
        max_abs_paper_dg_delight=13.0,
        max_epistemic_surprise=14.0,
        max_aleatoric_uncertainty=15.0,
        max_abs_learning_progress=16.0,
        max_safety_cost=17.0,
        advantage_scale_floor=0.1,
        action_surprisal_scale_floor=0.2,
        paper_dg_delight_scale_floor=0.3,
        epistemic_surprise_scale_floor=0.4,
        aleatoric_uncertainty_scale_floor=0.5,
        learning_progress_scale_floor=0.6,
        change_probability_scale_floor=0.01,
        safety_cost_scale_floor=0.8,
        normalization_clip=7.0,
    )
    payload = config.to_config()
    assert payload["schema"] == LEARNING_VALUE_ROUTER_CONFIG_SCHEMA
    assert payload["type"] == "LearningValueRouterConfig"
    assert payload["mechanism_status"] == MECHANISM_STATUS
    assert payload["scientific_promotion_allowed"] is False
    assert LearningValueRouterConfig.from_config(payload) == config
    serialized = cast(dict[str, object], json.loads(json.dumps(payload)))
    assert LearningValueRouter.from_config(serialized).config == config

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_steps = 9  # type: ignore[misc]

    malformed = (
        {**payload, "extra": 1},
        {key: value for key, value in payload.items() if key != "max_steps"},
        {**payload, "schema": "wrong"},
        {**payload, "type": "wrong"},
        {**payload, "mechanism_status": "accepted"},
        {**payload, "scientific_promotion_allowed": True},
        {**payload, "normalization_clip": 7},
        {**payload, "max_steps": 1234.0},
    )
    for candidate in malformed:
        with pytest.raises(ValueError):
            LearningValueRouterConfig.from_config(candidate)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normalization_min_count": True},
        {"normalization_min_count": 1},
        {"normalization_min_count": 3, "max_steps": 2},
        {"max_steps": True},
        {"max_steps": 0},
        {"max_steps": 715_827_883},
        {"max_abs_advantage": 0.0},
        {"max_action_surprisal": -1.0},
        {"max_abs_paper_dg_delight": float("inf")},
        {"max_epistemic_surprise": float("nan")},
        {"max_aleatoric_uncertainty": 1e-50},
        {"max_abs_learning_progress": True},
        {"max_safety_cost": 1e30, "max_steps": 2},
        {"advantage_scale_floor": 0.0},
        {"change_probability_scale_floor": float("nan")},
        {"normalization_clip": 0.0},
    ],
)
def test_config_rejects_invalid_or_numerically_unsafe_contracts(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        LearningValueRouterConfig(**kwargs)  # type: ignore[arg-type]


def test_metadata_names_all_producers_units_domains_and_paper_dg_semantics() -> None:
    router = LearningValueRouter(
        LearningValueRouterConfig(
            max_abs_advantage=11.0,
            max_action_surprisal=12.0,
            max_abs_paper_dg_delight=13.0,
            max_epistemic_surprise=14.0,
            max_aleatoric_uncertainty=15.0,
            max_abs_learning_progress=16.0,
            max_safety_cost=17.0,
            normalization_clip=7.0,
        )
    )
    metadata = router.channel_metadata()
    assert len(metadata) == 8
    assert tuple(item.index for item in metadata) == tuple(range(8))
    assert tuple(item.field_name for item in metadata) == _NAMES
    assert all(item.producer for item in metadata)
    assert all(item.causal_object for item in metadata)
    assert all(item.units for item in metadata)
    assert all(item.domain for item in metadata)
    assert all(item.normalization == "causal_pre_update_welford_zscore" for item in metadata)
    assert all(item.normalization_scale_floor > 0.0 for item in metadata)
    assert all(item.normalized_units == "dimensionless_zscore" for item in metadata)
    assert all(item.normalized_lower_bound == -7.0 for item in metadata)
    assert all(item.normalized_upper_bound == 7.0 for item in metadata)
    paper_dg = metadata[2]
    assert paper_dg.field_name == "delight"
    assert paper_dg.semantic_name == "paper_specific_dg_actor_sample_delight"
    assert "paper_specific_dg" in paper_dg.producer
    assert paper_dg.units == "return_nats"
    assert (paper_dg.lower_bound, paper_dg.upper_bound) == (-13.0, 13.0)
    assert (metadata[6].lower_bound, metadata[6].upper_bound) == (0.0, 1.0)
    assert "calibrated" not in metadata[3].producer
    assert "calibrated" not in metadata[3].units
    assert metadata[4].producer == "producer_declared_outcome_variance_estimator"
    assert "head" not in metadata[4].producer
    assert "calibrated" not in metadata[6].producer
    json.dumps([item.to_config() for item in metadata], allow_nan=False)


def test_resource_budget_matches_exact_state_and_result_leaf_counts() -> None:
    router = LearningValueRouter()
    state = router.init()
    budget = router.resource_budget()
    assert budget.channel_count == 8
    assert budget.consumer_route_count == 6
    assert budget.persistent_float32_scalars == 16
    assert budget.persistent_int32_scalars == 25
    assert budget.persistent_state_scalars == 41
    assert budget.persistent_state_bytes == 164
    assert budget.input_float32_scalars_per_step == 8
    assert budget.input_bool_scalars_per_step == 8
    assert budget.max_float32_state_scalars_touched_per_step == 16
    assert budget.max_int32_state_scalars_touched_per_step == 9
    assert budget.trainable_scalars == 0
    assert budget.rng_state_bytes == 0
    assert budget.replay_capacity == 0

    actual_state_bytes = sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(state))
    assert actual_state_bytes == budget.persistent_state_bytes
    _, result = router.route(state, _value(), _availability())
    float_count = 0
    bool_count = 0
    int_count = 0
    for leaf in jax.tree_util.tree_leaves(result):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            float_count += array.size
        elif np.issubdtype(array.dtype, np.bool_):
            bool_count += array.size
        elif np.issubdtype(array.dtype, np.integer):
            int_count += array.size
    assert float_count == budget.output_float32_scalars
    assert bool_count == budget.output_bool_scalars
    assert int_count == budget.output_int32_scalars
    assert budget.output_logical_bytes == 4 * (float_count + int_count) + bool_count


def test_routes_are_typed_masked_and_never_expose_a_generic_score() -> None:
    router = LearningValueRouter()
    next_state, result = router.route(router.init(), _value(), _availability())
    assert isinstance(result.paper_dg_actor, PaperDGActorRoute)
    assert isinstance(result.exploration, ExplorationLearningValueRoute)
    assert isinstance(result.model_memory_replay, ModelMemoryReplayLearningValueRoute)
    assert isinstance(result.adaptation_change, AdaptationChangeLearningValueRoute)
    assert isinstance(result.safety, SafetyLearningValueRoute)
    assert isinstance(result.literal_gradient_joy_evidence, LiteralGradientJoyEvidenceRoute)
    assert (
        result.candidate_update_audit_evidence
        is result.literal_gradient_joy_evidence
    )
    expected = _value()
    _assert_route_mask(
        result.paper_dg_actor,
        {"advantage", "action_surprisal", "delight"},
        expected,
    )
    _assert_route_mask(
        result.exploration,
        {
            "epistemic_surprise",
            "aleatoric_uncertainty",
            "learning_progress",
        },
        expected,
    )
    _assert_route_mask(
        result.model_memory_replay,
        {
            "epistemic_surprise",
            "aleatoric_uncertainty",
            "learning_progress",
        },
        expected,
    )
    _assert_route_mask(
        result.adaptation_change,
        {"learning_progress", "change_probability"},
        expected,
    )
    _assert_route_mask(result.safety, {"safety_cost"}, expected)
    _assert_route_mask(result.literal_gradient_joy_evidence, set(_NAMES), expected)
    assert bool(result.paper_dg_actor.ready)
    assert bool(result.exploration.ready)
    assert bool(result.model_memory_replay.ready)
    assert bool(result.adaptation_change.ready)
    assert bool(result.safety.ready)
    assert bool(result.literal_gradient_joy_evidence.ready)
    assert not any(
        hasattr(result, name)
        for name in ("score", "value_score", "aggregate", "utility", "sparks_joy")
    )
    assert int(next_state.step_count) == 1
    np.testing.assert_array_equal(next_state.channel_valid_counts, np.ones(8, dtype=np.int32))


def test_unrelated_invalid_and_unavailable_channels_do_not_suppress_safety() -> None:
    router = LearningValueRouter()
    values = _value(
        advantage=float("nan"),
        action_surprisal=-1.0,
        delight=float("inf"),
        epistemic_surprise=-3.0,
        learning_progress=float("inf"),
        change_probability=2.0,
        safety_cost=0.5,
    )
    availability = _availability(delight=False, aleatoric_uncertainty=False)
    next_state, result = router.route(router.init(), values, availability)

    assert bool(result.safety.ready)
    assert bool(result.safety.availability.safety_cost)
    np.testing.assert_array_equal(result.safety.values.safety_cost, 0.5)
    assert not bool(result.literal_gradient_joy_evidence.ready)
    assert not bool(result.paper_dg_actor.ready)
    assert not bool(result.exploration.ready)
    assert not bool(result.adaptation_change.ready)
    assert not bool(result.diagnostics.accepted.advantage)
    assert not bool(result.diagnostics.accepted.action_surprisal)
    assert not bool(result.diagnostics.accepted.delight)
    assert bool(result.diagnostics.accepted.safety_cost)
    assert not bool(result.diagnostics.finite.advantage)
    assert not bool(result.diagnostics.domain_valid.action_surprisal)
    assert not bool(result.diagnostics.declared_availability.delight)
    np.testing.assert_array_equal(
        next_state.channel_valid_counts,
        np.asarray([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_state.channel_unavailable_counts,
        np.asarray([0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_state.channel_invalid_counts,
        np.asarray([1, 1, 0, 1, 0, 1, 1, 0], dtype=np.int32),
    )


def test_paper_dg_delight_requires_prerequisites_and_exact_float32_identity() -> None:
    router = LearningValueRouter()
    _, exact = router.route(router.init(), _value(), _availability())
    assert bool(exact.diagnostics.paper_dg_delight_prerequisites_valid)
    assert bool(exact.diagnostics.paper_dg_delight_identity_valid)
    assert bool(exact.paper_dg_actor.ready)

    _, mismatch = router.route(
        router.init(),
        _value(delight=float(np.nextafter(np.float32(6.0), np.float32(7.0)))),
        _availability(),
    )
    assert bool(mismatch.diagnostics.paper_dg_delight_prerequisites_valid)
    assert not bool(mismatch.diagnostics.paper_dg_delight_identity_valid)
    assert bool(mismatch.diagnostics.domain_valid.delight)
    assert not bool(mismatch.diagnostics.accepted.delight)
    assert bool(mismatch.diagnostics.accepted.advantage)
    assert bool(mismatch.diagnostics.accepted.action_surprisal)
    assert not bool(mismatch.paper_dg_actor.ready)
    assert bool(mismatch.safety.ready)

    _, missing_prerequisite = router.route(
        router.init(),
        _value(),
        _availability(advantage=False),
    )
    assert not bool(
        missing_prerequisite.diagnostics.paper_dg_delight_prerequisites_valid
    )
    assert not bool(missing_prerequisite.diagnostics.paper_dg_delight_identity_valid)
    assert not bool(missing_prerequisite.diagnostics.accepted.delight)
    assert not bool(missing_prerequisite.paper_dg_actor.ready)
    assert bool(missing_prerequisite.safety.ready)


def test_change_availability_only_controls_adaptation_and_full_joy_routes() -> None:
    router = LearningValueRouter()
    _, result = router.route(
        router.init(),
        _value(),
        _availability(change_probability=False),
    )
    assert bool(result.exploration.ready)
    assert bool(result.model_memory_replay.ready)
    assert not bool(result.adaptation_change.ready)
    assert not bool(result.literal_gradient_joy_evidence.ready)
    assert bool(result.safety.ready)
    assert not bool(result.exploration.availability.change_probability)
    np.testing.assert_array_equal(result.exploration.values.change_probability, 0.0)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("advantage", 1.0e7),
        ("action_surprisal", -0.01),
        ("delight", -1.0e13),
        ("epistemic_surprise", -0.01),
        ("aleatoric_uncertainty", -0.01),
        ("learning_progress", float("inf")),
        ("change_probability", -0.01),
        ("change_probability", 1.01),
        ("safety_cost", -0.01),
    ],
)
def test_each_channel_has_independent_finite_and_domain_validation(
    field: str,
    invalid_value: float,
) -> None:
    router = LearningValueRouter()
    values = _value(**{field: invalid_value})
    _, result = router.route(router.init(), values, _availability())
    assert not bool(getattr(result.diagnostics.accepted, field))
    assert not bool(result.literal_gradient_joy_evidence.ready)
    consequentially_invalid = {field}
    if field in {"advantage", "action_surprisal"}:
        consequentially_invalid.add("delight")
    for unrelated in _NAMES:
        if unrelated not in consequentially_invalid:
            assert bool(getattr(result.diagnostics.accepted, unrelated))


def test_signed_channels_accept_negative_values_and_boundary_values_are_closed() -> None:
    config = LearningValueRouterConfig(
        max_abs_advantage=10.0,
        max_action_surprisal=11.0,
        max_abs_paper_dg_delight=120.0,
        max_epistemic_surprise=13.0,
        max_aleatoric_uncertainty=14.0,
        max_abs_learning_progress=15.0,
        max_safety_cost=16.0,
    )
    router = LearningValueRouter(config)
    values = _value(
        advantage=-10.0,
        action_surprisal=11.0,
        delight=-110.0,
        epistemic_surprise=13.0,
        aleatoric_uncertainty=14.0,
        learning_progress=-15.0,
        change_probability=1.0,
        safety_cost=16.0,
    )
    _, result = router.route(router.init(), values, _availability())
    assert bool(result.literal_gradient_joy_evidence.ready)
    assert all(bool(getattr(result.diagnostics.domain_valid, name)) for name in _NAMES)


def test_normalization_uses_only_pre_update_state_and_reports_calibration() -> None:
    router = LearningValueRouter(
        LearningValueRouterConfig(
            normalization_min_count=2,
            advantage_scale_floor=0.25,
        )
    )
    availability = _availability(
        action_surprisal=False,
        delight=False,
        epistemic_surprise=False,
        aleatoric_uncertainty=False,
        learning_progress=False,
        change_probability=False,
        safety_cost=False,
    )
    state = router.init()
    state, first = router.route(state, _value(advantage=1.0), availability)
    state, second = router.route(state, _value(advantage=3.0), availability)
    state, third = router.route(state, _value(advantage=5.0), availability)

    assert not bool(first.paper_dg_actor.normalized_availability.advantage)
    assert not bool(second.paper_dg_actor.normalized_availability.advantage)
    assert bool(third.paper_dg_actor.normalized_availability.advantage)
    np.testing.assert_array_equal(first.diagnostics.pre_update_count, np.zeros(8, dtype=np.int32))
    assert int(second.diagnostics.pre_update_count[0]) == 1
    assert int(third.diagnostics.pre_update_count[0]) == 2
    np.testing.assert_allclose(third.diagnostics.pre_update_mean.advantage, 2.0)
    np.testing.assert_allclose(third.diagnostics.pre_update_m2.advantage, 2.0)
    np.testing.assert_allclose(third.diagnostics.pre_update_scale.advantage, np.sqrt(2.0))
    np.testing.assert_allclose(
        third.paper_dg_actor.normalized_values.advantage,
        3.0 / np.sqrt(2.0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(state.channel_means[0], 3.0)
    np.testing.assert_allclose(state.channel_m2[0], 8.0)
    assert int(state.channel_valid_counts[0]) == 3
    np.testing.assert_array_equal(state.channel_unavailable_counts[1:], 3)
    assert not bool(third.paper_dg_actor.ready)
    assert not bool(third.literal_gradient_joy_evidence.ready)


def test_invalid_observation_does_not_enter_that_channels_normalization_state() -> None:
    router = LearningValueRouter()
    state = router.init()
    state, _ = router.route(state, _value(), _availability())
    before_safety_mean = np.asarray(state.channel_means[7]).copy()
    state, result = router.route(
        state,
        _value(advantage=float("nan"), safety_cost=1.25),
        _availability(),
    )
    assert int(state.channel_valid_counts[0]) == 1
    assert int(state.channel_invalid_counts[0]) == 1
    assert int(state.channel_valid_counts[7]) == 2
    assert int(state.channel_invalid_counts[7]) == 0
    np.testing.assert_array_equal(state.channel_means[0], 2.0)
    assert np.asarray(state.channel_means[7]) != before_safety_mean
    assert bool(result.safety.ready)
    assert not bool(result.literal_gradient_joy_evidence.ready)


def test_normalization_is_clipped_and_outputs_are_stop_gradient() -> None:
    router = LearningValueRouter(
        LearningValueRouterConfig(normalization_clip=2.0, advantage_scale_floor=1e-6)
    )
    availability = _availability(
        action_surprisal=False,
        delight=False,
        epistemic_surprise=False,
        aleatoric_uncertainty=False,
        learning_progress=False,
        change_probability=False,
        safety_cost=False,
    )
    state = router.init()
    state, _ = router.route(state, _value(advantage=0.0), availability)
    state, _ = router.route(state, _value(advantage=0.0), availability)
    _, result = router.route(state, _value(advantage=100.0), availability)
    np.testing.assert_array_equal(result.paper_dg_actor.normalized_values.advantage, 2.0)

    def routed_advantage(raw_advantage: jax.Array) -> jax.Array:
        values = _replace(_value(), advantage=raw_advantage)
        return router.route(state, values, _availability())[1].paper_dg_actor.values.advantage

    np.testing.assert_array_equal(jax.grad(routed_advantage)(jnp.asarray(1.0)), 0.0)


def test_eager_jit_scan_and_manual_sequence_have_parity() -> None:
    router = LearningValueRouter(LearningValueRouterConfig(normalization_min_count=2))
    values = [
        _value(advantage=1.0, change_probability=0.1),
        _value(advantage=3.0, change_probability=0.2),
        _value(advantage=5.0, change_probability=0.3),
        _value(advantage=float("nan"), change_probability=0.4),
    ]
    availability = [
        _availability(),
        _availability(delight=False),
        _availability(),
        _availability(epistemic_surprise=False),
    ]
    eager_state, eager_result = router.route(router.init(), values[0], availability[0])
    jit_state, jit_result = jax.jit(router.route)(
        router.init(), values[0], availability[0]
    )
    _assert_tree_allclose(eager_state, jit_state)
    _assert_tree_allclose(eager_result, jit_result)

    sequence_values = jax.tree_util.tree_map(
        lambda *parts: jnp.stack(parts),
        *values,
    )
    sequence_availability = jax.tree_util.tree_map(
        lambda *parts: jnp.stack(parts),
        *availability,
    )
    scan_state, scan_results = router.scan(
        router.init(),
        sequence_values,
        sequence_availability,
    )
    jit_scan_state, jit_scan_results = jax.jit(router.scan)(
        router.init(),
        sequence_values,
        sequence_availability,
    )
    _assert_tree_allclose(scan_state, jit_scan_state)
    _assert_tree_allclose(scan_results, jit_scan_results)

    manual_state = router.init()
    manual_results = []
    for event, flags in zip(values, availability, strict=True):
        manual_state, result = router.route(manual_state, event, flags)
        manual_results.append(result)
    stacked_manual = jax.tree_util.tree_map(lambda *parts: jnp.stack(parts), *manual_results)
    _assert_tree_allclose(scan_state, manual_state)
    _assert_tree_allclose(scan_results, stacked_manual)


def test_corrupt_dynamic_state_fails_all_routes_closed_and_remains_unchanged() -> None:
    router = LearningValueRouter()
    initial = router.init()
    corrupt = _replace(
        initial,
        channel_means=initial.channel_means.at[3].set(jnp.asarray(float("nan"))),
    )
    next_state, result = jax.jit(router.route)(corrupt, _value(), _availability())
    _assert_tree_equal(next_state, corrupt)
    assert not bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.normalization_state_updated)
    assert not bool(result.safety.ready)
    assert not bool(result.literal_gradient_joy_evidence.ready)
    for route in (
        result.paper_dg_actor,
        result.exploration,
        result.model_memory_replay,
        result.adaptation_change,
        result.safety,
        result.literal_gradient_joy_evidence,
    ):
        for name in _NAMES:
            np.testing.assert_array_equal(getattr(route.values, name), 0.0)
            np.testing.assert_array_equal(getattr(route.normalized_values, name), 0.0)
            assert not bool(getattr(route.availability, name))
            assert not bool(getattr(route.normalized_availability, name))
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(result)
        if np.issubdtype(np.asarray(leaf).dtype, np.floating)
    )
    with pytest.raises(ValueError):
        router.validate_state(corrupt)
    with pytest.raises(ValueError):
        router.checkpoint_payload(corrupt)


def test_counter_category_corruption_is_detected() -> None:
    router = LearningValueRouter()
    state, _ = router.route(router.init(), _value(), _availability())
    corrupt = _replace(
        state,
        channel_invalid_counts=state.channel_invalid_counts.at[0].set(1),
    )
    assert not bool(router.state_valid(corrupt))
    _, result = router.route(corrupt, _value(), _availability())
    assert not bool(result.safety.ready)


def test_counter_capacity_freezes_calibration_but_preserves_typed_raw_safety() -> None:
    router = LearningValueRouter(
        LearningValueRouterConfig(max_steps=2, normalization_min_count=2)
    )
    state = router.init()
    state, _ = router.route(state, _value(safety_cost=0.25), _availability())
    state, _ = router.route(state, _value(safety_cost=0.75), _availability())
    frozen = state
    state, result = router.route(state, _value(safety_cost=1.0), _availability())
    _assert_tree_equal(state, frozen)
    assert bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.counter_capacity_available)
    assert not bool(result.diagnostics.normalization_state_updated)
    assert bool(result.safety.ready)
    assert bool(result.safety.normalization_ready)
    np.testing.assert_array_equal(result.safety.values.safety_cost, 1.0)
    accounting = router.accounting(state)
    assert int(accounting.step_count) == 2
    assert bool(accounting.state_valid)
    assert not bool(accounting.counter_capacity_available)


def test_full_joy_evidence_route_requires_all_eight_and_normalization_requires_history() -> None:
    router = LearningValueRouter()
    state = router.init()
    for _ in range(2):
        state, result = router.route(state, _value(), _availability())
        assert bool(result.literal_gradient_joy_evidence.ready)
        assert not bool(result.literal_gradient_joy_evidence.normalization_ready)
    state, result = router.route(state, _value(), _availability())
    assert bool(result.literal_gradient_joy_evidence.ready)
    assert bool(result.literal_gradient_joy_evidence.normalization_ready)
    assert not hasattr(result.literal_gradient_joy_evidence, "sparks_joy")

    _, incomplete = router.route(
        state,
        _value(),
        _availability(learning_progress=False),
    )
    assert not bool(incomplete.literal_gradient_joy_evidence.ready)
    assert not bool(incomplete.literal_gradient_joy_evidence.normalization_ready)
    assert bool(incomplete.safety.ready)
    assert bool(incomplete.paper_dg_actor.ready)


def test_checkpoint_roundtrip_is_strict_json_safe_and_preserves_exact_state() -> None:
    router = LearningValueRouter(
        LearningValueRouterConfig(normalization_min_count=3, max_steps=100)
    )
    state = router.init()
    state, _ = router.route(state, _value(), _availability())
    state, _ = router.route(
        state,
        _value(advantage=float("nan"), safety_cost=1.25),
        _availability(delight=False),
    )
    payload = router.checkpoint_payload(state)
    assert payload["schema"] == LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA
    assert payload["mechanism_status"] == MECHANISM_STATUS
    assert payload["scientific_promotion_allowed"] is False
    serialized = cast(dict[str, object], json.loads(json.dumps(payload, allow_nan=False)))
    restored_router, restored_state = LearningValueRouter.from_checkpoint_payload(serialized)
    assert restored_router.config == router.config
    assert restored_router.resource_budget() == router.resource_budget()
    assert restored_router.channel_metadata() == router.channel_metadata()
    _assert_tree_equal(restored_state, state)
    assert restored_router.checkpoint_payload(restored_state) == payload


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema", "wrong"),
        ("mechanism_status", "accepted"),
        ("scientific_promotion_allowed", True),
        ("config_digest", "0" * 64),
        ("state_digest", "0" * 64),
    ],
)
def test_checkpoint_rejects_top_level_tampering(field: str, replacement: object) -> None:
    payload = LearningValueRouter().checkpoint_payload(LearningValueRouter().init())
    tampered = {**payload, field: replacement}
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(tampered)


def test_checkpoint_rejects_metadata_resource_config_and_state_tampering() -> None:
    router = LearningValueRouter()
    state, _ = router.route(router.init(), _value(), _availability())
    payload = router.checkpoint_payload(state)

    extra = {**payload, "extra": True}
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(extra)

    metadata = copy.deepcopy(payload)
    metadata_records = cast(list[dict[str, object]], metadata["channel_metadata"])
    metadata_records[2]["semantic_name"] = "generic_delight"
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(metadata)

    metadata_type = copy.deepcopy(payload)
    typed_metadata_records = cast(
        list[dict[str, object]], metadata_type["channel_metadata"]
    )
    typed_metadata_records[0]["index"] = 0.0
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(metadata_type)

    resources = copy.deepcopy(payload)
    resource_record = cast(dict[str, object], resources["resource_budget"])
    resource_record["trainable_scalars"] = 1
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(resources)

    resource_type = copy.deepcopy(payload)
    typed_resource_record = cast(dict[str, object], resource_type["resource_budget"])
    typed_resource_record["trainable_scalars"] = 0.0
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(resource_type)

    config = copy.deepcopy(payload)
    config_record = cast(dict[str, object], config["router"])
    config_record["normalization_clip"] = 9
    config["config_digest"] = _canonical_digest(config_record)
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(config)

    integer_float_state = copy.deepcopy(payload)
    integer_state = cast(dict[str, object], integer_float_state["state"])
    integer_means = cast(list[object], integer_state["channel_means"])
    integer_means[0] = 2
    integer_float_state["state_digest"] = _canonical_digest(integer_state)
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(integer_float_state)

    invalid_state = copy.deepcopy(payload)
    state_record = cast(dict[str, object], invalid_state["state"])
    invalid_counts = cast(list[int], state_record["channel_invalid_counts"])
    invalid_counts[0] = 1
    invalid_state["state_digest"] = _canonical_digest(state_record)
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(invalid_state)

    boolean_counter = copy.deepcopy(payload)
    boolean_state = cast(dict[str, object], boolean_counter["state"])
    boolean_counts = cast(list[object], boolean_state["channel_valid_counts"])
    boolean_counts[0] = True
    boolean_counter["state_digest"] = _canonical_digest(boolean_state)
    with pytest.raises(ValueError):
        LearningValueRouter.from_checkpoint_payload(boolean_counter)


def test_static_input_and_state_contracts_reject_shape_or_dtype_misuse() -> None:
    router = LearningValueRouter()
    state = router.init()
    with pytest.raises(TypeError):
        router.route(
            state,
            _replace(
                _value(), advantage=jnp.asarray(1, dtype=jnp.int32)
            ),
            _availability(),
        )
    with pytest.raises(ValueError):
        router.route(
            state,
            _replace(
                _value(), advantage=jnp.asarray([1.0], dtype=jnp.float32)
            ),
            _availability(),
        )
    with pytest.raises(TypeError):
        router.route(
            state,
            _value(),
            _replace(
                _availability(), advantage=jnp.asarray(1, dtype=jnp.int32)
            ),
        )
    with pytest.raises(ValueError):
        router.route(
            _replace(
                state, step_count=jnp.asarray([0], dtype=jnp.int32)
            ),
            _value(),
            _availability(),
        )
    with pytest.raises(ValueError):
        router.route(
            _replace(
                state, channel_means=jnp.zeros((7,), dtype=jnp.float32)
            ),
            _value(),
            _availability(),
        )


def test_scan_rejects_mismatched_sequence_contracts() -> None:
    router = LearningValueRouter()
    values = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), _value())
    availability = jax.tree_util.tree_map(
        lambda value: jnp.stack([value, value]), _availability()
    )
    with pytest.raises(ValueError):
        router.scan(
            router.init(),
            _replace(values, delight=jnp.ones((3,), dtype=jnp.float32)),
            availability,
        )
    with pytest.raises(ValueError):
        router.scan(router.init(), _value(), _availability())
    with pytest.raises(TypeError):
        router.scan(
            router.init(),
            values,
            _replace(
                availability, safety_cost=jnp.ones((2,), dtype=jnp.int32)
            ),
        )


def test_accounting_repeats_exact_counter_state_without_hidden_resources() -> None:
    router = LearningValueRouter()
    state = router.init()
    state, _ = router.route(
        state,
        _value(epistemic_surprise=-1.0),
        _availability(delight=False),
    )
    accounting = router.accounting(state)
    assert int(accounting.step_count) == 1
    np.testing.assert_array_equal(accounting.channel_valid_counts, state.channel_valid_counts)
    np.testing.assert_array_equal(
        accounting.channel_unavailable_counts,
        state.channel_unavailable_counts,
    )
    np.testing.assert_array_equal(accounting.channel_invalid_counts, state.channel_invalid_counts)
    assert bool(accounting.state_valid)
    assert bool(accounting.counter_capacity_available)
    assert set(router.resource_budget().to_config()) == {
        "channel_count",
        "consumer_route_count",
        "max_counter_steps",
        "persistent_float32_scalars",
        "persistent_int32_scalars",
        "persistent_state_scalars",
        "persistent_state_bytes",
        "input_float32_scalars_per_step",
        "input_bool_scalars_per_step",
        "max_float32_state_scalars_touched_per_step",
        "max_int32_state_scalars_touched_per_step",
        "output_float32_scalars",
        "output_bool_scalars",
        "output_int32_scalars",
        "output_logical_bytes",
        "trainable_scalars",
        "rng_state_bytes",
        "replay_capacity",
    }


def test_checkpoint_parser_rejects_non_mapping_sections() -> None:
    payload = LearningValueRouter().checkpoint_payload(LearningValueRouter().init())
    for field in ("router", "state"):
        malformed = {**payload, field: []}
        with pytest.raises(ValueError):
            LearningValueRouter.from_checkpoint_payload(malformed)


def test_state_class_remains_a_fixed_frozen_jax_pytree() -> None:
    state = LearningValueRouter().init()
    assert isinstance(state, LearningValueRouterState)
    assert len(jax.tree_util.tree_leaves(state)) == 6
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.step_count = jnp.asarray(1, dtype=jnp.int32)


def test_checkpoint_input_type_annotation_accepts_read_only_mapping() -> None:
    router = LearningValueRouter()
    payload: Mapping[str, Any] = router.checkpoint_payload(router.init())
    restored_router, restored_state = LearningValueRouter.from_checkpoint_payload(payload)
    assert restored_router.config == router.config
    _assert_tree_equal(restored_state, router.init())
