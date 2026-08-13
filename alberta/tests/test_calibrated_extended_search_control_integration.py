# mypy: disable-error-code="attr-defined,call-arg"
"""JIT, scan, causal warm-up, and resume integration for WP7.4 search."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.calibrated_extended_search_control import (
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    SEARCH_MODE_COMBINED,
    CalibratedExtendedSearchControl,
    CalibratedExtendedSearchControlConfig,
    CalibratedExtendedSearchControlState,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]

ANCHORS = jnp.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=jnp.float32)
SOURCE = jnp.asarray((17, 29), dtype=jnp.uint32)
DESCRIPTORS = jnp.asarray(((3, 5, 1, 7),), dtype=jnp.int32)
GENERATIONS = jnp.asarray((2,), dtype=jnp.int32)
PRIMITIVE_NEXT = jnp.asarray(
    (
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 0.0)),
    ),
    dtype=jnp.float32,
)
OPTION_NEXT = jnp.asarray((((1.0, 0.0),), ((0.0, 1.0),)), dtype=jnp.float32)


def _controller(*, backup_budget: int = 2) -> CalibratedExtendedSearchControl:
    return CalibratedExtendedSearchControl(
        CalibratedExtendedSearchControlConfig(
            mode=SEARCH_MODE_COMBINED,
            observation_dim=2,
            anchor_capacity=2,
            n_primitive_actions=2,
            n_options=1,
            backup_budget=backup_budget,
            calibration_evidence_floor=4,
            model_support_floor=4,
            confidence_scale=1.0,
            support_prior=4.0,
            model_error_scale=10.0,
            backup_step_size=0.2,
            max_observations=10_000,
        )
    )


def _state(
    controller: CalibratedExtendedSearchControl,
    *,
    calibrated: bool,
) -> CalibratedExtendedSearchControlState:
    state = controller.init(
        anchor_bank=ANCHORS,
        anchor_active=jnp.ones((2,), dtype=jnp.bool_),
        q_values=jnp.zeros((2, 3), dtype=jnp.float32),
        option_descriptors=DESCRIPTORS,
        option_generations=GENERATIONS,
        representation_generation=jnp.asarray(4, dtype=jnp.int32),
        source_digest=SOURCE,
    )
    if not calibrated:
        return state
    c = controller.config.candidate_capacity
    return cast(
        CalibratedExtendedSearchControlState,
        state.replace(
            last_realized_targets=jnp.ones((c,), dtype=jnp.float32),
            last_target_available=jnp.ones((c,), dtype=jnp.bool_),
            value_change_counts=jnp.full((c,), 4, dtype=jnp.int32),
            value_change_means=jnp.ones((c,), dtype=jnp.float32),
            value_change_m2=jnp.zeros((c,), dtype=jnp.float32),
            model_error_counts=jnp.full((c,), 4, dtype=jnp.int32),
            model_error_means=jnp.zeros((c,), dtype=jnp.float32),
            model_error_m2=jnp.zeros((c,), dtype=jnp.float32),
            support_counts=jnp.full((c,), 4, dtype=jnp.int32),
            anchor_revisit_trials=jnp.full((2,), 4, dtype=jnp.int32),
            anchor_revisit_successes=jnp.full((2,), 4, dtype=jnp.int32),
        ),
    )


def _arm(
    controller: CalibratedExtendedSearchControl,
    state: CalibratedExtendedSearchControlState,
    *,
    decision_id: jax.Array,
    anchor: jax.Array,
    kind: jax.Array,
    index: jax.Array,
) -> Any:
    return controller.arm(
        state,
        decision_id=decision_id,
        decision_observation=ANCHORS[anchor],
        decision_anchor_index=anchor,
        executed_kind=kind,
        executed_index=index,
        average_reward=jnp.asarray(0.0, dtype=jnp.float32),
        primitive_reward_predictions=jnp.ones((2, 2), dtype=jnp.float32),
        primitive_discount_predictions=jnp.zeros((2, 2), dtype=jnp.float32),
        primitive_next_anchor_probabilities=PRIMITIVE_NEXT,
        primitive_model_available=jnp.ones((2, 2), dtype=jnp.bool_),
        primitive_model_support=jnp.full((2, 2), 8, dtype=jnp.int32),
        option_return_predictions=jnp.ones((2, 1), dtype=jnp.float32),
        option_baseline_mass_predictions=jnp.ones((2, 1), dtype=jnp.float32),
        option_discount_predictions=jnp.zeros((2, 1), dtype=jnp.float32),
        option_next_anchor_probabilities=OPTION_NEXT,
        option_model_available=jnp.ones((2, 1), dtype=jnp.bool_),
        option_model_support=jnp.full((2, 1), 8, dtype=jnp.int32),
        option_initiation_available=jnp.ones((2, 1), dtype=jnp.bool_),
        representation_generation=state.representation_generation,
        source_digest=state.source_digest,
        option_descriptors=state.option_descriptors,
        option_generations=state.option_generations,
        learner_revision=state.learner_revision,
        primitive_model_revision=jnp.asarray(1, dtype=jnp.int32),
        option_model_revision=jnp.asarray(1, dtype=jnp.int32),
    )


def _observe(
    controller: CalibratedExtendedSearchControl,
    state: CalibratedExtendedSearchControlState,
    *,
    future_anchor: jax.Array,
    elapsed: jax.Array,
) -> Any:
    return controller.observe(
        state,
        decision_id=state.pending_decision_id,
        future_observation=ANCHORS[future_anchor],
        observed_future_anchor_mask=(
            jnp.arange(2, dtype=jnp.int32) == future_anchor
        ),
        external_return=jnp.asarray(1.0, dtype=jnp.float32),
        baseline_mass=jnp.where(
            state.pending_executed_kind == CANDIDATE_KIND_PRIMITIVE,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
        ),
        terminal_discount=jnp.asarray(0.0, dtype=jnp.float32),
        elapsed_primitive_steps=elapsed,
        natural_completion=jnp.asarray(True, dtype=jnp.bool_),
        censored=jnp.asarray(False, dtype=jnp.bool_),
        representation_generation=state.representation_generation,
        source_digest=state.source_digest,
        option_descriptors=state.option_descriptors,
        option_generations=state.option_generations,
        learner_revision=state.learner_revision,
        primitive_model_revision=state.primitive_model_revision,
        option_model_revision=state.option_model_revision,
    )


def test_eager_and_jit_arm_observe_are_bit_exact() -> None:
    controller = _controller()
    state = _state(controller, calibrated=True)
    decision_id = jnp.asarray((0, 0, 0, 1), dtype=jnp.uint32)
    anchor = jnp.asarray(0, dtype=jnp.int32)
    kind = jnp.asarray(CANDIDATE_KIND_PRIMITIVE, dtype=jnp.int32)
    index = jnp.asarray(0, dtype=jnp.int32)

    eager_arm = _arm(
        controller,
        state,
        decision_id=decision_id,
        anchor=anchor,
        kind=kind,
        index=index,
    )
    compiled_arm = jax.jit(_arm, static_argnums=0)(
        controller,
        state,
        decision_id=decision_id,
        anchor=anchor,
        kind=kind,
        index=index,
    )
    chex.assert_trees_all_equal(eager_arm, compiled_arm)

    future = jnp.asarray(1, dtype=jnp.int32)
    elapsed = jnp.asarray(1, dtype=jnp.int32)
    eager_observe = _observe(
        controller, eager_arm.state, future_anchor=future, elapsed=elapsed
    )
    compiled_observe = jax.jit(_observe, static_argnums=0)(
        controller, compiled_arm.state, future_anchor=future, elapsed=elapsed
    )
    chex.assert_trees_all_equal(eager_observe, compiled_observe)


def test_scan_matches_eager_lifecycle_and_preserves_exact_attempt_budget() -> None:
    controller = _controller(backup_budget=2)
    initial = _state(controller, calibrated=True)
    steps = jnp.arange(5, dtype=jnp.int32)

    def transition(
        state: CalibratedExtendedSearchControlState, step: jax.Array
    ) -> tuple[CalibratedExtendedSearchControlState, tuple[jax.Array, jax.Array]]:
        anchor = step % 2
        future = 1 - anchor
        decision_id = jnp.asarray((0, 0, 0, 0), dtype=jnp.uint32).at[3].set(
            step.astype(jnp.uint32) + jnp.uint32(1)
        )
        armed = _arm(
            controller,
            state,
            decision_id=decision_id,
            anchor=anchor,
            kind=jnp.asarray(CANDIDATE_KIND_PRIMITIVE, dtype=jnp.int32),
            index=jnp.asarray(0, dtype=jnp.int32),
        )
        observed = _observe(
            controller,
            armed.state,
            future_anchor=future,
            elapsed=jnp.asarray(1, dtype=jnp.int32),
        )
        return observed.state, (
            armed.diagnostics.backup_attempt_count,
            observed.diagnostics.learner_update_count,
        )

    eager_state = initial
    eager_attempts: list[int] = []
    eager_updates: list[int] = []
    for step in range(5):
        eager_state, diagnostics = transition(
            eager_state, jnp.asarray(step, dtype=jnp.int32)
        )
        eager_attempts.append(int(diagnostics[0]))
        eager_updates.append(int(diagnostics[1]))

    scanned_state, scanned = jax.jit(lambda s: jax.lax.scan(transition, s, steps))(
        initial
    )
    # A fused scan may contract the float32 multiply-add used by the tabular
    # backup; discrete lifecycle/counter state is exact and float state stays
    # within one float32 rounding unit.
    chex.assert_trees_all_close(eager_state, scanned_state, rtol=1.1e-7, atol=6.0e-8)
    chex.assert_trees_all_equal(eager_state.last_decision_id, scanned_state.last_decision_id)
    chex.assert_trees_all_equal(eager_state.support_counts, scanned_state.support_counts)
    chex.assert_trees_all_equal(eager_state.learner_revision, scanned_state.learner_revision)
    np.testing.assert_array_equal(np.asarray(scanned[0]), np.asarray(eager_attempts))
    np.testing.assert_array_equal(np.asarray(scanned[1]), np.asarray(eager_updates))
    np.testing.assert_array_equal(np.asarray(scanned[0]), np.full(5, 2))
    assert bool(jnp.all(scanned[1] <= 2))


def test_calibration_uses_only_prior_outcomes_and_opens_on_the_next_arm() -> None:
    controller = _controller(backup_budget=1)
    state = _state(controller, calibrated=False)
    for step in range(4):
        decision_id = jnp.asarray((0, 0, 0, step + 1), dtype=jnp.uint32)
        arm = _arm(
            controller,
            state,
            decision_id=decision_id,
            anchor=jnp.asarray(0, dtype=jnp.int32),
            kind=jnp.asarray(CANDIDATE_KIND_PRIMITIVE, dtype=jnp.int32),
            index=jnp.asarray(0, dtype=jnp.int32),
        )
        # Even the fourth outcome cannot leak into its already-frozen arm.
        assert not bool(arm.diagnostics.candidate_eligible[0])
        observed = _observe(
            controller,
            arm.state,
            future_anchor=jnp.asarray(0, dtype=jnp.int32),
            elapsed=jnp.asarray(1, dtype=jnp.int32),
        )
        assert bool(observed.diagnostics.transaction_valid)
        state = observed.state

    fifth = _arm(
        controller,
        state,
        decision_id=jnp.asarray((0, 0, 0, 5), dtype=jnp.uint32),
        anchor=jnp.asarray(0, dtype=jnp.int32),
        kind=jnp.asarray(CANDIDATE_KIND_PRIMITIVE, dtype=jnp.int32),
        index=jnp.asarray(0, dtype=jnp.int32),
    )
    assert int(state.value_change_counts[0]) == 4
    assert int(state.model_error_counts[0]) == 4
    assert int(state.support_counts[0]) == 4
    assert int(state.anchor_revisit_trials[0]) == 4
    assert bool(fifth.diagnostics.candidate_eligible[0])
    assert bool(fifth.diagnostics.selected_valid[0])


def test_mid_option_checkpoint_resume_matches_uninterrupted_natural_completion() -> None:
    controller = _controller()
    state = _state(controller, calibrated=True)
    armed = _arm(
        controller,
        state,
        decision_id=jnp.asarray((0, 0, 0, 44), dtype=jnp.uint32),
        anchor=jnp.asarray(0, dtype=jnp.int32),
        kind=jnp.asarray(CANDIDATE_KIND_OPTION, dtype=jnp.int32),
        index=jnp.asarray(0, dtype=jnp.int32),
    ).state
    payload = controller.checkpoint_payload(armed)
    restored = controller.restore_checkpoint(
        payload,
        representation_generation=armed.representation_generation,
        source_digest=armed.source_digest,
        option_descriptors=armed.option_descriptors,
        option_generations=armed.option_generations,
        learner_revision=armed.learner_revision,
        primitive_model_revision=armed.primitive_model_revision,
        option_model_revision=armed.option_model_revision,
    )
    uninterrupted = _observe(
        controller,
        armed,
        future_anchor=jnp.asarray(1, dtype=jnp.int32),
        elapsed=jnp.asarray(3, dtype=jnp.int32),
    )
    resumed = _observe(
        controller,
        restored,
        future_anchor=jnp.asarray(1, dtype=jnp.int32),
        elapsed=jnp.asarray(3, dtype=jnp.int32),
    )

    chex.assert_trees_all_equal(uninterrupted, resumed)
    assert bool(resumed.diagnostics.natural_resolution)
    assert int(resumed.diagnostics.backup_attempt_count) == 2
    assert int(resumed.diagnostics.learner_update_count) <= 2
