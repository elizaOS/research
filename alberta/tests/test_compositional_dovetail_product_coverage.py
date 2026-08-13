"""Contracts for task-agnostic product-grammar coverage generation."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    GENERATION_DOVETAIL_PRODUCT_COVERAGE,
    GENERATION_ROBUST_RECURSIVE,
    OP_PRODUCT,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    _dovetail_product_coverage_cursor,
    _dovetail_product_coverage_cycle,
)

pytestmark = pytest.mark.unit

_UINT32_MAX = 2**32 - 1
_RAW_DIM = 6
_ACTIVE_SLOTS = 11
_INTERVAL = 32


def _coverage_learner(**overrides: object) -> CompositionalFeatureLearner:
    values: dict[str, object] = {
        "n_features": _ACTIVE_SLOTS,
        "n_tasks": 1,
        "candidate_count": 1,
        "replacement_interval": _INTERVAL,
        "min_feature_age": 2**31 - 1,
        "candidate_min_age": 0,
        "promotion_margin": 1.0e6,
        "max_depth": 3,
        "use_obgd": False,
        "generation_strategy": GENERATION_DOVETAIL_PRODUCT_COVERAGE,
    }
    values.update(overrides)
    return CompositionalFeatureLearner(**values)  # type: ignore[arg-type]


def test_pre_step_dovetail_cursor_crosses_uint32_carry_and_uses_high_word() -> None:
    cycle = _dovetail_product_coverage_cycle(
        n_features=_ACTIVE_SLOTS,
        feature_dim=_RAW_DIM,
    )
    assert cycle == 60

    probes = (
        ((0, 31), 0),
        ((0, 63), 1),
        (
            (0, _UINT32_MAX),
            ((_UINT32_MAX // _INTERVAL) % cycle),
        ),
        (
            (1, 31),
            ((((1 << 32) + 31) // _INTERVAL) % cycle),
        ),
        (
            (37, 95),
            ((((37 << 32) + 95) // _INTERVAL) % cycle),
        ),
    )
    for words, expected in probes:
        actual = _dovetail_product_coverage_cursor(
            jnp.asarray(words, dtype=jnp.uint32),
            replacement_interval=_INTERVAL,
            n_features=_ACTIVE_SLOTS,
            feature_dim=_RAW_DIM,
        )
        assert int(actual) == expected

    compiled = jax.jit(
        lambda words: _dovetail_product_coverage_cursor(
            words,
            replacement_interval=_INTERVAL,
            n_features=_ACTIVE_SLOTS,
            feature_dim=_RAW_DIM,
        )
    )
    assert int(compiled(jnp.asarray((1, 31), dtype=jnp.uint32))) == probes[3][1]


def test_dovetail_rejects_static_modulus_overflow() -> None:
    with pytest.raises(ValueError, match="16-bit-limb modulus"):
        _dovetail_product_coverage_cursor(
            jnp.zeros((2,), dtype=jnp.uint32),
            replacement_interval=2_000,
            n_features=_ACTIVE_SLOTS,
            feature_dim=_RAW_DIM,
        )


def test_even_proposals_cover_every_distinct_raw_pair_exactly_once() -> None:
    learner = _coverage_learner()
    state = learner.init(_RAW_DIM, jr.key(71))
    actual: list[tuple[int, int]] = []
    for pair_index in range(15):
        op, parent_a, parent_b, _, depth = learner._generate_one(
            jr.key(100 + pair_index),
            state.depth,
            coverage_cursor=jnp.asarray(2 * pair_index, dtype=jnp.int32),
            feature_dim=_RAW_DIM,
        )
        assert int(op) == OP_PRODUCT
        assert int(depth) == 1
        actual.append((int(parent_a), int(parent_b)))

    assert actual == [
        (left, right)
        for left in range(_RAW_DIM)
        for right in range(left + 1, _RAW_DIM)
    ]


def test_odd_proposals_dovetail_every_depth1_slot_with_every_raw_parent() -> None:
    learner = _coverage_learner()
    state = learner.init(_RAW_DIM, jr.key(72))
    actual: list[tuple[int, int]] = []
    for extension_index in range((_ACTIVE_SLOTS - _RAW_DIM) * _RAW_DIM):
        op, parent_a, parent_b, _, depth = learner._generate_one(
            jr.key(200 + extension_index),
            state.depth,
            coverage_cursor=jnp.asarray(2 * extension_index + 1, dtype=jnp.int32),
            feature_dim=_RAW_DIM,
        )
        assert int(op) == OP_PRODUCT
        assert int(depth) == 2
        actual.append((int(parent_a), int(parent_b)))

    assert actual == [
        (slot, raw_parent)
        for raw_parent in range(_RAW_DIM)
        for slot in range(_RAW_DIM, _ACTIVE_SLOTS)
    ]

    changed_depth = state.depth.at[8].set(2)
    _, parent_a, _, _, _ = learner._generate_one(
        jr.key(999),
        changed_depth,
        coverage_cursor=jnp.asarray(5, dtype=jnp.int32),
        feature_dim=_RAW_DIM,
        preferred_depth1_parent=jnp.asarray(8, dtype=jnp.int32),
    )
    assert int(parent_a) == 9  # cyclic next-valid fallback, not a random target tweak


def test_actual_curation_proposal_uses_pre_step_identity_at_low_word_carry() -> None:
    learner = _coverage_learner()
    initial = cast(
        CompositionalFeatureState,
        learner.init(_RAW_DIM, jr.key(73)).replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
            replacement_phase=jnp.asarray(_INTERVAL - 1, dtype=jnp.int32),
            ages=jnp.full((_ACTIVE_SLOTS,), 2**31 - 1, dtype=jnp.int32),
            candidate_ages=jnp.asarray((2**31 - 1,), dtype=jnp.int32),
        ),
    )
    result = learner.update(
        initial,
        jnp.ones((_RAW_DIM,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    cursor = ((_UINT32_MAX // _INTERVAL) % 60)
    extension_index = cursor // 2
    expected_extension = (
        _RAW_DIM + extension_index % (_ACTIVE_SLOTS - _RAW_DIM),
        (extension_index // (_ACTIVE_SLOTS - _RAW_DIM)) % _RAW_DIM,
    )

    np.testing.assert_array_equal(result.state.step_words, (1, 0))
    assert bool(result.curation_trace.proposal_formed)
    assert cursor % 2 == 1
    assert (
        int(result.curation_trace.proposal_parent_a),
        int(result.curation_trace.proposal_parent_b),
    ) == expected_extension


def test_terminal_lifetime_is_still_a_bit_exact_noop() -> None:
    learner = _coverage_learner()
    initial = learner.init(_RAW_DIM, jr.key(731))
    exhausted = cast(
        CompositionalFeatureState,
        initial.replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
            replacement_phase=jnp.asarray(0, dtype=jnp.int32),
            ages=jnp.full_like(initial.ages, 2**31 - 1),
            candidate_ages=jnp.full_like(initial.candidate_ages, 2**31 - 1),
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    result = learner.update(
        exhausted,
        jnp.ones((_RAW_DIM,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.state, exhausted)
    assert not bool(result.curation_trace.lifetime_capacity_available)
    assert not bool(result.curation_trace.has_event)


def test_strategy_is_defaults_off_restricted_and_round_trips_without_legacy_drift() -> None:
    coverage = _coverage_learner()
    restored = CompositionalFeatureLearner.from_config(coverage.to_config())
    assert restored.to_config() == coverage.to_config()
    assert coverage.to_config()["generation_strategy"] == (
        GENERATION_DOVETAIL_PRODUCT_COVERAGE
    )

    for changes, message in (
        ({"candidate_count": 0}, "candidate_count"),
        ({"replacement_interval": 0}, "fixed replacement"),
        ({"max_depth": 1}, "max_depth"),
        ({"learn_generator_resources": True}, "learn_generator_resources"),
        ({"operation_prior": (0.0, 0.25, 0.25, 0.25, 0.25)}, "product-only"),
    ):
        with pytest.raises(ValueError, match=message):
            _coverage_learner(**changes)

    legacy = CompositionalFeatureLearner(
        n_features=_ACTIVE_SLOTS,
        n_tasks=1,
        candidate_count=8,
        replacement_interval=_INTERVAL,
        max_depth=3,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
    )
    legacy_state = legacy.init(_RAW_DIM, jr.key(74)).replace(  # type: ignore[attr-defined]
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    round_trip_state = CompositionalFeatureLearner.from_config(
        legacy.to_config()
    ).init(_RAW_DIM, jr.key(74)).replace(  # type: ignore[attr-defined]
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    chex.assert_trees_all_equal(legacy_state, round_trip_state)
    np.testing.assert_array_equal(legacy_state.parent_a[6:], (0, 0, 0, 0, 0))
    np.testing.assert_array_equal(legacy_state.parent_b[6:], (1, 2, 3, 4, 5))
