"""Independent stale-retirement cadence contracts for interaction features."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.0, 0.0, 0.0), dtype=jnp.float32)
_TARGET = jnp.asarray((0.0,), dtype=jnp.float32)


def _retiring_learner(
    *,
    n_features: int = 1,
    replacement_interval: int = 0,
    stale_retirement_interval: int | None = 1,
) -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=n_features,
        n_tasks=1,
        step_size_output=0.01,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        utility_retention_grace_steps=0,
        utility_evidence_threshold=1.0,
        retire_stale_features=True,
        stale_retirement_interval=stale_retirement_interval,
        candidate_promotion_floor=1.0,
        use_obgd=False,
    )


def _all_stale_state(
    learner: FixedBudgetInteractionLearner,
    *,
    key: int = 0,
) -> InteractionFeatureState:
    state = learner.init(3, jr.key(key))
    return state.replace(  # type: ignore[attr-defined,no-any-return]
        evidence_idle_steps=jnp.ones_like(state.evidence_idle_steps),
        ages=jnp.ones_like(state.ages),
    )


def test_retirement_interval_validation_and_legacy_serialization() -> None:
    legacy = FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=1,
        replacement_interval=64,
    )
    assert legacy.to_config()["stale_retirement_interval"] is None
    assert FixedBudgetInteractionLearner.from_config(legacy.to_config()).to_config() == (
        legacy.to_config()
    )
    legacy_payload = legacy.to_config()
    legacy_payload.pop("stale_retirement_interval")
    assert (
        FixedBudgetInteractionLearner.from_config(legacy_payload)
        .to_config()["stale_retirement_interval"]
        is None
    )

    for invalid in (True, 0, -1, _INT32_MAX + 1):
        with pytest.raises(
            ValueError,
            match="stale_retirement_interval must be None or a positive int32-safe integer",
        ):
            FixedBudgetInteractionLearner(
                n_features=1,
                n_tasks=1,
                stale_retirement_interval=invalid,
            )

    with pytest.raises(ValueError, match="positive retirement cadence"):
        _retiring_learner(
            replacement_interval=0,
            stale_retirement_interval=None,
        )


def test_default_retirement_cadence_is_bit_exact_legacy_behavior() -> None:
    legacy = _retiring_learner(
        n_features=2,
        replacement_interval=3,
        stale_retirement_interval=None,
    )
    explicit = _retiring_learner(
        n_features=2,
        replacement_interval=3,
        stale_retirement_interval=3,
    )
    legacy_state = _all_stale_state(legacy, key=3)
    explicit_state = _all_stale_state(explicit, key=3)

    def run(
        learner: FixedBudgetInteractionLearner,
        state: InteractionFeatureState,
    ) -> tuple[InteractionFeatureState, tuple[jax.Array, jax.Array, jax.Array]]:
        def step(
            carry: InteractionFeatureState,
            _: jax.Array,
        ) -> tuple[InteractionFeatureState, tuple[jax.Array, jax.Array, jax.Array]]:
            result = learner.update(carry, _OBSERVATION, _TARGET)
            return result.state, (
                result.retired_slot,
                result.replaced_slot,
                result.post_step_words,
            )

        return jax.lax.scan(step, state, jnp.arange(8, dtype=jnp.int32))

    legacy_final, legacy_trace = run(legacy, legacy_state)
    explicit_final, explicit_trace = run(explicit, explicit_state)

    chex.assert_trees_all_equal(legacy_final, explicit_final)
    chex.assert_trees_all_equal(legacy_trace, explicit_trace)
    assert legacy.memory_accounting(legacy_final) == explicit.memory_accounting(
        explicit_final
    )


def test_independent_retirement_due_uses_exact_words_across_low_word_rollover() -> None:
    learner = _retiring_learner(stale_retirement_interval=5)
    state = _all_stale_state(learner, key=5).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        replacement_phase=jnp.asarray(0, dtype=jnp.int32),
    )

    def step(
        carry: InteractionFeatureState,
        _: jax.Array,
    ) -> tuple[InteractionFeatureState, tuple[jax.Array, jax.Array]]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, (result.retired_slot, result.post_step_words)

    final, (retired_slots, post_words) = jax.lax.scan(
        step,
        state,
        jnp.arange(5, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(retired_slots, (-1, -1, -1, -1, 0))
    np.testing.assert_array_equal(
        post_words,
        np.asarray(
            ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4)),
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(final.step_words, (1, 4))


def test_due_transaction_retires_at_most_one_stale_slot() -> None:
    learner = _retiring_learner(n_features=3, stale_retirement_interval=1)
    state = _all_stale_state(learner, key=7)

    first = learner.update(state, _OBSERVATION, _TARGET)
    assert int(first.retired_slot) >= 0
    assert int(first.live_feature_count) == 2
    assert int(first.vacancy_count) == 1
    changed_first = np.flatnonzero(
        np.asarray(first.state.feature_left) != np.asarray(state.feature_left)
    )
    np.testing.assert_array_equal(changed_first, (int(first.retired_slot),))

    second = learner.update(first.state, _OBSERVATION, _TARGET)
    # Without a confirmed candidate refill, one vacancy is a fail-stop guard:
    # faster cadence must not silently collapse the feature bank.
    assert int(second.retired_slot) == -1
    assert int(second.live_feature_count) == 2
    assert int(second.vacancy_count) == 1


def test_fast_retirement_refills_with_nonmatching_confirmed_candidate_only() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        replacement_interval=0,
        min_feature_age=0,
        candidate_count=3,
        candidate_min_age=0,
        candidate_strategy="all_pairs",
        utility_retention_grace_steps=0,
        utility_evidence_threshold=1.0,
        retire_stale_features=True,
        stale_retirement_interval=1,
        candidate_promotion_floor=0.1,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        use_obgd=False,
    )
    state = learner.init(3, jr.key(13)).replace(  # type: ignore[attr-defined]
        feature_left=jnp.asarray((0, 0), dtype=jnp.int32),
        feature_right=jnp.asarray((1, 2), dtype=jnp.int32),
        evidence_idle_steps=jnp.ones((2,), dtype=jnp.int32),
        ages=jnp.ones((2,), dtype=jnp.int32),
        candidate_left=jnp.asarray((0, 0, 1), dtype=jnp.int32),
        candidate_right=jnp.asarray((1, 2, 2), dtype=jnp.int32),
        # The retired identity is deliberately the highest-ranked archive
        # entry. A stale pre-reset confirmation must not reacquire it.
        candidate_utilities=jnp.asarray((10.0, 0.0, 5.0), dtype=jnp.float32),
        candidate_ages=jnp.ones((3,), dtype=jnp.int32),
    )

    result = learner.update(state, _OBSERVATION, _TARGET)

    assert int(result.retired_slot) == 0
    assert int(result.retired_left) == 0
    assert int(result.retired_right) == 1
    assert int(result.promoted_candidate) == 2
    assert int(result.replaced_slot) == 0
    assert bool(result.promoted_into_vacancy)
    assert int(result.live_feature_count) == 2
    assert int(result.vacancy_count) == 0
    np.testing.assert_array_equal(
        (result.state.feature_left[0], result.state.feature_right[0]),
        (1, 2),
    )
    assert bool(result.matching_candidate_reset_mask[0])
    assert float(result.state.candidate_utilities[0]) == 0.0
    assert int(result.state.candidate_ages[0]) == 0


def test_corrupt_exact_identity_rejects_before_retirement() -> None:
    learner = _retiring_learner(stale_retirement_interval=1)
    state = _all_stale_state(learner, key=11)
    corrupt = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(1, dtype=jnp.int32),
    )

    result = learner.update(corrupt, _OBSERVATION, _TARGET)

    assert bool(result.update_rejected)
    assert int(result.retired_slot) == -1
    chex.assert_trees_all_equal(result.state, corrupt)
