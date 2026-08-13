# mypy: disable-error-code="attr-defined,call-arg"
"""Unit contracts for calibrated fixed-budget extended-action search."""

from __future__ import annotations

import copy
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.calibrated_extended_search_control import (
    CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA,
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    SEARCH_MODE_COMBINED,
    SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
    SEARCH_MODE_OPTION_MODEL,
    SEARCH_MODE_PRIMITIVE_MODEL,
    CalibratedExtendedSearchControl,
    CalibratedExtendedSearchControlConfig,
    CalibratedExtendedSearchControlState,
)

pytestmark = pytest.mark.unit

ANCHORS = jnp.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=jnp.float32)
ACTIVE = jnp.ones((2,), dtype=jnp.bool_)
SOURCE = jnp.asarray((0x12345678, 0x9ABCDEF0), dtype=jnp.uint32)
DESCRIPTORS = jnp.asarray(((7, 11, 1, 101),), dtype=jnp.int32)
GENERATIONS = jnp.asarray((3,), dtype=jnp.int32)


def _config(
    mode: str = SEARCH_MODE_COMBINED,
    *,
    backup_budget: int = 2,
    max_observations: int = 1_000,
) -> CalibratedExtendedSearchControlConfig:
    return CalibratedExtendedSearchControlConfig(
        mode=mode,
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
        backup_step_size=0.25,
        max_observations=max_observations,
    )


def _controller(
    mode: str = SEARCH_MODE_COMBINED,
    *,
    backup_budget: int = 2,
    max_observations: int = 1_000,
) -> CalibratedExtendedSearchControl:
    return CalibratedExtendedSearchControl(
        _config(
            mode,
            backup_budget=backup_budget,
            max_observations=max_observations,
        )
    )


def _state(
    controller: CalibratedExtendedSearchControl,
    *,
    q_values: jax.Array | None = None,
    calibrated: bool = True,
) -> CalibratedExtendedSearchControlState:
    cfg = controller.config
    state = controller.init(
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        q_values=(
            jnp.zeros((2, 3), dtype=jnp.float32)
            if q_values is None
            else q_values
        ),
        option_descriptors=DESCRIPTORS,
        option_generations=GENERATIONS,
        representation_generation=jnp.asarray(5, dtype=jnp.int32),
        source_digest=SOURCE,
    )
    if not calibrated:
        return state
    c = cfg.candidate_capacity
    return cast(
        CalibratedExtendedSearchControlState,
        state.replace(
            last_realized_targets=jnp.linspace(1.0, 2.0, c, dtype=jnp.float32),
            last_target_available=jnp.ones((c,), dtype=jnp.bool_),
            value_change_counts=jnp.full((c,), 4, dtype=jnp.int32),
            value_change_means=jnp.full((c,), 0.8, dtype=jnp.float32),
            value_change_m2=jnp.zeros((c,), dtype=jnp.float32),
            model_error_counts=jnp.full((c,), 4, dtype=jnp.int32),
            model_error_means=jnp.full((c,), 0.25, dtype=jnp.float32),
            model_error_m2=jnp.zeros((c,), dtype=jnp.float32),
            support_counts=jnp.full((c,), 4, dtype=jnp.int32),
            anchor_revisit_trials=jnp.full((2,), 4, dtype=jnp.int32),
            anchor_revisit_successes=jnp.full((2,), 4, dtype=jnp.int32),
        ),
    )


def _arm_inputs(
    state: CalibratedExtendedSearchControlState,
    *,
    decision_word: int = 1,
    anchor: int = 0,
    executed_kind: int = CANDIDATE_KIND_PRIMITIVE,
    executed_index: int = 0,
) -> dict[str, Any]:
    primitive_next = jnp.asarray(
        (
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 1.0), (1.0, 0.0)),
        ),
        dtype=jnp.float32,
    )
    option_next = jnp.asarray(
        (((0.0, 1.0),), ((1.0, 0.0),)), dtype=jnp.float32
    )
    return {
        "decision_id": jnp.asarray((0, 0, 0, decision_word), dtype=jnp.uint32),
        "decision_observation": ANCHORS[anchor],
        "decision_anchor_index": jnp.asarray(anchor, dtype=jnp.int32),
        "executed_kind": jnp.asarray(executed_kind, dtype=jnp.int32),
        "executed_index": jnp.asarray(executed_index, dtype=jnp.int32),
        "average_reward": jnp.asarray(0.4, dtype=jnp.float32),
        "primitive_reward_predictions": jnp.asarray(
            ((2.0, 1.0), (1.0, 1.0)), dtype=jnp.float32
        ),
        "primitive_discount_predictions": jnp.asarray(
            ((0.5, 0.0), (0.0, 0.0)), dtype=jnp.float32
        ),
        "primitive_next_anchor_probabilities": primitive_next,
        "primitive_model_available": jnp.ones((2, 2), dtype=jnp.bool_),
        "primitive_model_support": jnp.full((2, 2), 8, dtype=jnp.int32),
        "option_return_predictions": jnp.asarray(((2.5,), (1.0,)), dtype=jnp.float32),
        "option_baseline_mass_predictions": jnp.asarray(
            ((1.25,), (1.0,)), dtype=jnp.float32
        ),
        "option_discount_predictions": jnp.asarray(((0.5,), (0.0,)), dtype=jnp.float32),
        "option_next_anchor_probabilities": option_next,
        "option_model_available": jnp.ones((2, 1), dtype=jnp.bool_),
        "option_model_support": jnp.full((2, 1), 8, dtype=jnp.int32),
        "option_initiation_available": jnp.ones((2, 1), dtype=jnp.bool_),
        "representation_generation": state.representation_generation,
        "source_digest": state.source_digest,
        "option_descriptors": state.option_descriptors,
        "option_generations": state.option_generations,
        "learner_revision": state.learner_revision,
        "primitive_model_revision": jnp.asarray(7, dtype=jnp.int32),
        "option_model_revision": jnp.asarray(9, dtype=jnp.int32),
    }


def _observe_inputs(
    armed: CalibratedExtendedSearchControlState,
    *,
    natural: bool = True,
    censored: bool = False,
    elapsed: int = 1,
    future_anchor: int = 1,
) -> dict[str, Any]:
    mask = jnp.zeros((2,), dtype=jnp.bool_)
    if not censored:
        mask = mask.at[future_anchor].set(True)
    return {
        "decision_id": armed.pending_decision_id,
        "future_observation": ANCHORS[future_anchor],
        "observed_future_anchor_mask": mask,
        "external_return": jnp.asarray(2.0, dtype=jnp.float32),
        "baseline_mass": jnp.asarray(
            1.0 if int(armed.pending_executed_kind) == CANDIDATE_KIND_PRIMITIVE else 1.25,
            dtype=jnp.float32,
        ),
        "terminal_discount": jnp.asarray(0.5, dtype=jnp.float32),
        "elapsed_primitive_steps": jnp.asarray(elapsed, dtype=jnp.int32),
        "natural_completion": jnp.asarray(natural, dtype=jnp.bool_),
        "censored": jnp.asarray(censored, dtype=jnp.bool_),
        "representation_generation": armed.representation_generation,
        "source_digest": armed.source_digest,
        "option_descriptors": armed.option_descriptors,
        "option_generations": armed.option_generations,
        "learner_revision": armed.learner_revision,
        "primitive_model_revision": armed.primitive_model_revision,
        "option_model_revision": armed.option_model_revision,
    }


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(left, right)


def test_config_resources_and_fixed_candidate_capacity_are_exact() -> None:
    controller = _controller(backup_budget=3)
    config = controller.config
    payload = config.to_config()
    assert CalibratedExtendedSearchControlConfig.from_config(payload) == config
    assert payload["scientific_promotion_allowed"] is False
    assert payload["policy_authority"] is False
    with pytest.raises(ValueError, match="fields"):
        CalibratedExtendedSearchControlConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="mode"):
        CalibratedExtendedSearchControlConfig.from_config(
            {**payload, "mode": "dynamic"}
        )

    state = _state(controller)
    leaves = jax.tree_util.tree_leaves(state)
    resource = controller.resource_budget
    assert resource.candidate_capacity == 2 * (2 + 1)
    assert resource.candidate_predictions_per_ranking == 6
    assert resource.candidate_evaluations_per_ranking == 6
    assert resource.max_candidate_comparisons_per_ranking == 18
    assert resource.backup_attempts_per_committed_observation == 3
    assert resource.max_learner_updates_per_committed_observation == 3
    assert resource.pending_arm_slots == 1
    assert resource.arm_diagnostic_payload_bytes == 12 + 26 * 6 + 25 * 3
    assert resource.observe_diagnostic_payload_bytes == 46 + 21 * 3
    assert resource.max_diagnostic_payload_bytes_per_call == (
        resource.arm_diagnostic_payload_bytes
    )
    assert resource.random_draws_total == 0
    assert resource.persistent_state_growth_per_observation_bytes == 0
    assert resource.policy_authority is False
    assert resource.scientific_promotion_allowed is False
    assert resource.persistent_logical_scalars == sum(leaf.size for leaf in leaves)
    assert resource.persistent_state_nbytes == sum(
        leaf.size * leaf.dtype.itemsize for leaf in leaves
    )


def test_primitive_and_option_differential_targets_use_every_declared_field() -> None:
    controller = _controller(SEARCH_MODE_COMBINED)
    q_values = jnp.asarray(((1.0, 0.0, 0.5), (3.0, 2.0, 1.0)), dtype=jnp.float32)
    state = _state(controller, q_values=q_values)
    diagnostics = controller.arm(state, **_arm_inputs(state)).diagnostics

    # Candidate order is primitive action, anchor; then option, anchor.
    # Primitive a0@anchor0: 2.0 - 0.4 + 0.5 * maxQ(anchor0)=2.1.
    assert float(diagnostics.candidate_targets[0]) == pytest.approx(2.1)
    # Primitive a0@anchor1: 1.0 - 0.4 + 0 * maxQ(anchor1)=0.6.
    assert float(diagnostics.candidate_targets[1]) == pytest.approx(0.6)
    # Option 0@anchor0: 2.5 - 0.4*1.25 + 0.5*maxQ(anchor1)=3.5.
    assert float(diagnostics.candidate_targets[4]) == pytest.approx(3.5)
    # All four fields are live: changing mass alone changes only the option term.
    changed = _arm_inputs(state, decision_word=2)
    changed["option_baseline_mass_predictions"] = jnp.asarray(
        ((2.25,), (1.0,)), dtype=jnp.float32
    )
    changed_diagnostics = controller.arm(state, **changed).diagnostics
    assert float(changed_diagnostics.candidate_targets[4]) == pytest.approx(3.1)
    np.testing.assert_allclose(
        np.asarray(changed_diagnostics.candidate_targets[:4]),
        np.asarray(diagnostics.candidate_targets[:4]),
    )


@pytest.mark.parametrize(
    "field,expected_delta",
    (
        ("primitive_reward", 1.0),
        ("primitive_discount", 0.1),
        ("primitive_next", 1.0),
        ("option_return", 1.0),
        ("option_mass", -0.4),
        ("option_discount", 0.3),
        ("option_next", -1.0),
    ),
)
def test_each_primitive_and_option_target_field_has_its_analytic_effect(
    field: str, expected_delta: float
) -> None:
    controller = _controller(SEARCH_MODE_COMBINED)
    q_values = jnp.asarray(((1.0, 0.0, 0.5), (3.0, 2.0, 1.0)), dtype=jnp.float32)
    state = _state(controller, q_values=q_values)
    base_inputs = _arm_inputs(state, decision_word=10)
    base = controller.arm(state, **base_inputs).diagnostics.candidate_targets
    changed = _arm_inputs(state, decision_word=11)
    candidate = 0 if field.startswith("primitive") else 4
    if field == "primitive_reward":
        values = cast_array(changed["primitive_reward_predictions"])
        changed["primitive_reward_predictions"] = values.at[0, 0].add(1.0)
    elif field == "primitive_discount":
        values = cast_array(changed["primitive_discount_predictions"])
        changed["primitive_discount_predictions"] = values.at[0, 0].add(0.1)
    elif field == "primitive_next":
        values = cast_array(changed["primitive_next_anchor_probabilities"])
        changed["primitive_next_anchor_probabilities"] = values.at[0, 0].set(
            jnp.asarray((0.0, 1.0), dtype=jnp.float32)
        )
    elif field == "option_return":
        values = cast_array(changed["option_return_predictions"])
        changed["option_return_predictions"] = values.at[0, 0].add(1.0)
    elif field == "option_mass":
        values = cast_array(changed["option_baseline_mass_predictions"])
        changed["option_baseline_mass_predictions"] = values.at[0, 0].add(1.0)
    elif field == "option_discount":
        values = cast_array(changed["option_discount_predictions"])
        changed["option_discount_predictions"] = values.at[0, 0].add(0.1)
    else:
        values = cast_array(changed["option_next_anchor_probabilities"])
        changed["option_next_anchor_probabilities"] = values.at[0, 0].set(
            jnp.asarray((1.0, 0.0), dtype=jnp.float32)
        )
    result = controller.arm(state, **changed).diagnostics.candidate_targets

    assert float(result[candidate] - base[candidate]) == pytest.approx(
        expected_delta, abs=1.0e-6
    )
    untouched = np.ones(controller.config.candidate_capacity, dtype=np.bool_)
    untouched[candidate] = False
    np.testing.assert_allclose(np.asarray(result)[untouched], np.asarray(base)[untouched])


def test_priority_is_exact_noncompensating_product_and_factors_are_independent() -> None:
    controller = _controller()
    state = _state(controller)
    diagnostics = controller.arm(state, **_arm_inputs(state)).diagnostics

    # 0.8 value LCB * 1.0 reach LCB * (1-.25 error UCB) * 4/(4+4).
    assert float(diagnostics.priorities[0]) == pytest.approx(0.3)
    assert bool(diagnostics.candidate_eligible[0])

    higher_value = state.replace(
        value_change_means=state.value_change_means.at[0].set(1.0)
    )
    higher = controller.arm(higher_value, **_arm_inputs(higher_value)).diagnostics
    assert float(higher.priorities[0]) == pytest.approx(0.375)
    np.testing.assert_allclose(
        np.asarray(higher.priorities[1:]), np.asarray(diagnostics.priorities[1:])
    )

    higher_error = state.replace(
        model_error_means=state.model_error_means.at[0].set(0.5)
    )
    error_result = controller.arm(higher_error, **_arm_inputs(higher_error)).diagnostics
    assert float(error_result.priorities[0]) == pytest.approx(0.2)
    np.testing.assert_allclose(
        np.asarray(error_result.priorities[1:]), np.asarray(diagnostics.priorities[1:])
    )


@pytest.mark.parametrize("missing_factor", ("value", "error", "reach", "support", "model"))
def test_each_unavailable_factor_is_independently_ineligible(missing_factor: str) -> None:
    controller = _controller()
    state = _state(controller)
    inputs = _arm_inputs(state)
    if missing_factor == "value":
        state = state.replace(
            value_change_counts=state.value_change_counts.at[0].set(3)
        )
    elif missing_factor == "error":
        state = state.replace(model_error_counts=state.model_error_counts.at[0].set(3))
    elif missing_factor == "reach":
        state = state.replace(
            anchor_revisit_trials=state.anchor_revisit_trials.at[0].set(3),
            anchor_revisit_successes=state.anchor_revisit_successes.at[0].set(3),
        )
    elif missing_factor == "support":
        state = state.replace(support_counts=state.support_counts.at[0].set(3))
    else:
        available = cast_array(inputs["primitive_model_available"])
        inputs["primitive_model_available"] = available.at[0, 0].set(False)
    inputs.update(
        representation_generation=state.representation_generation,
        source_digest=state.source_digest,
        option_descriptors=state.option_descriptors,
        option_generations=state.option_generations,
        learner_revision=state.learner_revision,
    )
    diagnostics = controller.arm(state, **inputs).diagnostics
    assert not bool(diagnostics.candidate_eligible[0])
    assert float(diagnostics.priorities[0]) == 0.0


def cast_array(value: object) -> jax.Array:
    assert isinstance(value, jax.Array)
    return value


def test_reachability_requires_real_future_revisit_not_current_anchor_availability() -> None:
    controller = _controller()
    state = _state(controller)
    state = state.replace(
        anchor_revisit_trials=jnp.full((2,), 100, dtype=jnp.int32),
        anchor_revisit_successes=jnp.asarray((0, 100), dtype=jnp.int32),
    )
    diagnostics = controller.arm(state, **_arm_inputs(state)).diagnostics

    # The decision itself is exactly anchor 0, but no future revisit evidence exists.
    assert bool(diagnostics.decision_anchor_matches)
    assert float(diagnostics.reachability_lcb[0]) == 0.0
    assert not bool(diagnostics.candidate_eligible[0])
    assert float(diagnostics.reachability_lcb[1]) == 1.0


@pytest.mark.parametrize(
    "mode,expected_valid_kinds",
    (
        (SEARCH_MODE_MODEL_FREE_EXTENDED_Q, {0, 1}),
        (SEARCH_MODE_PRIMITIVE_MODEL, {0}),
        (SEARCH_MODE_OPTION_MODEL, {1}),
        (SEARCH_MODE_COMBINED, {0, 1}),
    ),
)
def test_all_static_modes_share_one_exact_total_budget(
    mode: str, expected_valid_kinds: set[int]
) -> None:
    controller = _controller(mode, backup_budget=3)
    state = _state(controller)
    diagnostics = controller.arm(state, **_arm_inputs(state)).diagnostics

    assert diagnostics.selected_candidate_indices.shape == (3,)
    assert diagnostics.selected_valid.shape == (3,)
    assert int(diagnostics.backup_attempt_count) == 3
    expected_updates = 2 if mode == SEARCH_MODE_OPTION_MODEL else 3
    assert int(jnp.sum(diagnostics.selected_valid)) == expected_updates
    selected_kinds = set(
        int(value)
        for value in np.asarray(diagnostics.selected_kinds)[
            np.asarray(diagnostics.selected_valid)
        ]
    )
    assert selected_kinds <= expected_valid_kinds
    if mode == SEARCH_MODE_OPTION_MODEL:
        assert selected_kinds == {1}
    if mode == SEARCH_MODE_PRIMITIVE_MODEL:
        assert selected_kinds == {0}
    # Combined has three attempts total, not three primitive plus three option.
    assert controller.resource_budget.backup_attempts_per_committed_observation == 3


def test_cross_family_ties_are_stable_by_kind_semantic_index_then_anchor() -> None:
    controller = _controller(SEARCH_MODE_COMBINED, backup_budget=6)
    state = _state(controller)
    inputs = _arm_inputs(state)
    inputs["primitive_reward_predictions"] = jnp.full((2, 2), 1.0, dtype=jnp.float32)
    inputs["primitive_discount_predictions"] = jnp.zeros((2, 2), dtype=jnp.float32)
    inputs["option_return_predictions"] = jnp.full((2, 1), 1.4, dtype=jnp.float32)
    inputs["option_baseline_mass_predictions"] = jnp.ones((2, 1), dtype=jnp.float32)
    inputs["option_discount_predictions"] = jnp.zeros((2, 1), dtype=jnp.float32)
    diagnostics = controller.arm(state, **inputs).diagnostics

    np.testing.assert_array_equal(
        np.asarray(diagnostics.selected_kinds),
        np.asarray((0, 0, 0, 0, 1, 1), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.selected_semantic_indices),
        np.asarray((0, 0, 1, 1, 0, 0), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.selected_anchor_indices),
        np.asarray((0, 1, 0, 1, 0, 1), dtype=np.int32),
    )


def test_arm_freezes_schedule_and_observe_cannot_leak_outcome_into_backups() -> None:
    controller = _controller(SEARCH_MODE_COMBINED, backup_budget=1)
    state = _state(controller)
    arm = controller.arm(state, **_arm_inputs(state))
    selected = int(arm.diagnostics.selected_candidate_indices[0])
    selected_anchor = int(arm.diagnostics.selected_anchor_indices[0])
    selected_extended = (
        int(arm.diagnostics.selected_semantic_indices[0])
        + (
            controller.config.n_primitive_actions
            if int(arm.diagnostics.selected_kinds[0]) == CANDIDATE_KIND_OPTION
            else 0
        )
    )
    frozen_target = float(arm.diagnostics.selected_targets[0])
    observe_inputs = _observe_inputs(arm.state)
    observe_inputs["external_return"] = jnp.asarray(100.0, dtype=jnp.float32)
    result = controller.observe(arm.state, **observe_inputs)

    assert int(result.diagnostics.selected_candidate_indices[0]) == selected
    expected = 0.25 * frozen_target
    assert float(result.state.q_values[selected_anchor, selected_extended]) == pytest.approx(
        expected
    )
    assert float(result.diagnostics.realized_differential_target) != pytest.approx(
        frozen_target
    )


@pytest.mark.parametrize("stale_field", ("decision", "source", "learner", "model"))
def test_stale_or_misattributed_observation_is_an_atomic_noop(stale_field: str) -> None:
    controller = _controller()
    state = _state(controller)
    armed = controller.arm(state, **_arm_inputs(state)).state
    inputs = _observe_inputs(armed)
    if stale_field == "decision":
        inputs["decision_id"] = jnp.asarray((0, 0, 0, 99), dtype=jnp.uint32)
    elif stale_field == "source":
        inputs["source_digest"] = jnp.asarray((1, 2), dtype=jnp.uint32)
    elif stale_field == "learner":
        inputs["learner_revision"] = armed.learner_revision + 1
    else:
        inputs["option_model_revision"] = armed.option_model_revision + 1
    result = controller.observe(armed, **inputs)

    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, armed)


def test_primitive_resolves_at_one_transition_and_option_only_naturally() -> None:
    controller = _controller()
    state = _state(controller)
    primitive = controller.arm(state, **_arm_inputs(state)).state
    invalid_primitive = controller.observe(
        primitive, **_observe_inputs(primitive, elapsed=2)
    )
    assert not bool(invalid_primitive.diagnostics.transaction_valid)
    _assert_tree_equal(invalid_primitive.state, primitive)

    option_inputs = _arm_inputs(
        state,
        decision_word=2,
        executed_kind=CANDIDATE_KIND_OPTION,
        executed_index=0,
    )
    option = controller.arm(state, **option_inputs).state
    mid_option = controller.observe(
        option,
        **_observe_inputs(option, natural=False, censored=False, elapsed=2),
    )
    assert not bool(mid_option.diagnostics.transaction_valid)
    _assert_tree_equal(mid_option.state, option)
    completed = controller.observe(
        option, **_observe_inputs(option, natural=True, elapsed=3)
    )
    assert bool(completed.diagnostics.natural_resolution)
    assert bool(completed.diagnostics.calibration_updated)


def test_censoring_closes_pending_arm_without_support_calibration_or_backups() -> None:
    controller = _controller(backup_budget=2)
    state = _state(controller)
    armed = controller.arm(
        state,
        **_arm_inputs(
            state,
            executed_kind=CANDIDATE_KIND_OPTION,
            executed_index=0,
        ),
    ).state
    result = controller.observe(
        armed,
        **_observe_inputs(armed, natural=False, censored=True, elapsed=3),
    )

    assert bool(result.diagnostics.transaction_valid)
    assert bool(result.diagnostics.censored_resolution)
    assert not bool(result.diagnostics.calibration_updated)
    assert not bool(result.diagnostics.reachability_updated)
    assert int(result.diagnostics.learner_update_count) == 0
    assert not bool(result.state.pending)
    _assert_tree_equal(result.state.q_values, armed.q_values)
    _assert_tree_equal(result.state.support_counts, armed.support_counts)
    _assert_tree_equal(result.state.value_change_counts, armed.value_change_counts)
    _assert_tree_equal(result.state.anchor_revisit_trials, armed.anchor_revisit_trials)


def test_counter_overflow_fails_closed_before_any_write() -> None:
    controller = _controller(max_observations=8)
    state = _state(controller)
    armed = controller.arm(state, **_arm_inputs(state)).state
    candidate = 0
    armed = armed.replace(
        support_counts=armed.support_counts.at[candidate].set(8),
    )
    # Runtime pending integrity excludes calibration arrays; the overflow is a
    # valid state but cannot accept another observation for this candidate.
    result = controller.observe(armed, **_observe_inputs(armed))
    assert not bool(result.diagnostics.capacity_available)
    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, armed)


def test_nonfinite_unavailable_model_cell_does_not_get_licensed_by_mask() -> None:
    controller = _controller()
    state = _state(controller)
    inputs = _arm_inputs(state)
    rewards = cast_array(inputs["primitive_reward_predictions"])
    available = cast_array(inputs["primitive_model_available"])
    inputs["primitive_reward_predictions"] = rewards.at[1, 1].set(jnp.nan)
    inputs["primitive_model_available"] = available.at[1, 1].set(False)
    result = controller.arm(state, **inputs)

    assert not bool(result.diagnostics.inputs_finite)
    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, state)


def test_finite_arm_operands_cannot_overflow_into_a_committed_pending_state() -> None:
    controller = _controller()
    state = _state(controller)
    inputs = _arm_inputs(state)
    float32_max = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    inputs["average_reward"] = -float32_max
    inputs["primitive_reward_predictions"] = jnp.full(
        (2, 2), float32_max, dtype=jnp.float32
    )

    result = controller.arm(state, **inputs)

    assert bool(result.diagnostics.inputs_finite)
    assert not bool(result.diagnostics.derived_values_valid)
    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, state)
    assert bool(
        controller.validate_state(
            result.state,
            representation_generation=state.representation_generation,
            source_digest=state.source_digest,
            option_descriptors=state.option_descriptors,
            option_generations=state.option_generations,
        )
    )


def test_finite_observation_operands_cannot_overflow_into_a_committed_update() -> None:
    controller = _controller()
    float32_max = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    state = _state(
        controller,
        q_values=jnp.full((2, 3), float32_max, dtype=jnp.float32),
    )
    armed = controller.arm(state, **_arm_inputs(state)).state
    assert bool(armed.pending)
    inputs = _observe_inputs(armed)
    inputs["external_return"] = float32_max
    inputs["terminal_discount"] = jnp.asarray(1.0, dtype=jnp.float32)

    result = controller.observe(armed, **inputs)

    assert bool(result.diagnostics.inputs_finite)
    assert not bool(result.diagnostics.derived_values_valid)
    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, armed)
    assert bool(
        controller.validate_state(
            result.state,
            representation_generation=armed.representation_generation,
            source_digest=armed.source_digest,
            option_descriptors=armed.option_descriptors,
            option_generations=armed.option_generations,
        )
    )


def test_runtime_pending_cache_tamper_fails_closed_even_with_valid_shapes() -> None:
    controller = _controller()
    state = _state(controller)
    armed = controller.arm(state, **_arm_inputs(state)).state
    tampered = armed.replace(
        pending_selected_targets=armed.pending_selected_targets.at[0].add(1.0)
    )
    result = controller.observe(tampered, **_observe_inputs(tampered))

    assert not bool(result.diagnostics.pending_cache_valid)
    assert not bool(result.diagnostics.transaction_valid)
    _assert_tree_equal(result.state, tampered)


def test_replacement_advances_generation_and_invalidates_only_option_semantics() -> None:
    controller = _controller()
    state = _state(controller)
    state = state.replace(
        q_values=jnp.arange(6, dtype=jnp.float32).reshape((2, 3)) + 1.0,
    )
    armed = controller.arm(
        state,
        **_arm_inputs(
            state,
            executed_kind=CANDIDATE_KIND_OPTION,
            executed_index=0,
        ),
    ).state
    replacement_descriptors = DESCRIPTORS.at[0, 3].set(202)
    replacement_generations = GENERATIONS.at[0].set(4)
    replaced = controller.replace_option_universe(
        armed,
        option_descriptors=replacement_descriptors,
        option_generations=replacement_generations,
    )

    assert not bool(replaced.pending)
    np.testing.assert_allclose(
        np.asarray(replaced.q_values[:, :2]), np.asarray(armed.q_values[:, :2])
    )
    np.testing.assert_array_equal(np.asarray(replaced.q_values[:, 2]), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(replaced.support_counts[:4]), np.full(4, 4))
    np.testing.assert_array_equal(np.asarray(replaced.support_counts[4:]), np.zeros(2))
    assert not bool(jnp.any(replaced.last_target_available[4:]))
    with pytest.raises(ValueError, match="strictly advance"):
        controller.replace_option_universe(
            replaced,
            option_descriptors=DESCRIPTORS,
            option_generations=GENERATIONS,
        )


def test_observe_isolates_internal_q_and_exact_executed_candidate_statistics() -> None:
    controller = _controller(backup_budget=1)
    state = _state(controller)
    armed = controller.arm(state, **_arm_inputs(state)).state
    result = controller.observe(armed, **_observe_inputs(armed))

    assert int(result.state.value_change_counts[0]) == 5
    np.testing.assert_array_equal(
        np.asarray(result.state.value_change_counts[1:]),
        np.asarray(armed.value_change_counts[1:]),
    )
    _assert_tree_equal(result.state.anchor_bank, armed.anchor_bank)
    _assert_tree_equal(result.state.anchor_active, armed.anchor_active)
    _assert_tree_equal(result.state.option_descriptors, armed.option_descriptors)
    _assert_tree_equal(result.state.option_generations, armed.option_generations)
    _assert_tree_equal(result.state.source_digest, armed.source_digest)
    assert int(result.diagnostics.rng_draw_count) == 0
    assert controller.resource_budget.external_model_updates_per_observation == 0
    assert controller.resource_budget.policy_dispatches_per_observation == 0


def test_model_free_mode_uses_only_last_causally_resolved_targets() -> None:
    controller = _controller(SEARCH_MODE_MODEL_FREE_EXTENDED_Q)
    state = _state(controller)
    first = _arm_inputs(state, decision_word=1)
    second = _arm_inputs(state, decision_word=2)
    second["primitive_reward_predictions"] = jnp.full((2, 2), 999.0, dtype=jnp.float32)
    second["option_return_predictions"] = jnp.full((2, 1), -999.0, dtype=jnp.float32)
    first_result = controller.arm(state, **first).diagnostics
    second_result = controller.arm(state, **second).diagnostics

    np.testing.assert_allclose(
        np.asarray(first_result.candidate_targets),
        np.asarray(state.last_realized_targets),
    )
    np.testing.assert_allclose(
        np.asarray(second_result.candidate_targets),
        np.asarray(first_result.candidate_targets),
    )
    np.testing.assert_array_equal(
        np.asarray(second_result.selected_candidate_indices),
        np.asarray(first_result.selected_candidate_indices),
    )


def test_checkpoint_binds_pending_arm_resources_all_identities_and_tamper() -> None:
    controller = _controller()
    state = _state(controller)
    armed = controller.arm(state, **_arm_inputs(state)).state
    payload = controller.checkpoint_payload(armed)
    assert payload["schema_version"] == CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA
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
    _assert_tree_equal(restored, armed)
    assert bool(restored.pending)

    tampered = copy.copy(payload)
    tampered["state"] = armed.replace(
        pending_selected_targets=armed.pending_selected_targets.at[0].add(1.0)
    )
    with pytest.raises(ValueError, match="digest"):
        controller.restore_checkpoint(
            tampered,
            representation_generation=armed.representation_generation,
            source_digest=armed.source_digest,
            option_descriptors=armed.option_descriptors,
            option_generations=armed.option_generations,
            learner_revision=armed.learner_revision,
            primitive_model_revision=armed.primitive_model_revision,
            option_model_revision=armed.option_model_revision,
        )

    resources = copy.deepcopy(payload)
    resource_payload = dict(cast_dict(resources["resource_budget"]))
    resource_payload["backup_budget"] = 999
    resources["resource_budget"] = resource_payload
    with pytest.raises(ValueError, match="resource"):
        controller.restore_checkpoint(
            resources,
            representation_generation=armed.representation_generation,
            source_digest=armed.source_digest,
            option_descriptors=armed.option_descriptors,
            option_generations=armed.option_generations,
            learner_revision=armed.learner_revision,
            primitive_model_revision=armed.primitive_model_revision,
            option_model_revision=armed.option_model_revision,
        )

    with pytest.raises(ValueError, match="identity"):
        controller.restore_checkpoint(
            payload,
            representation_generation=armed.representation_generation,
            source_digest=jnp.asarray((1, 2), dtype=jnp.uint32),
            option_descriptors=armed.option_descriptors,
            option_generations=armed.option_generations,
            learner_revision=armed.learner_revision,
            primitive_model_revision=armed.primitive_model_revision,
            option_model_revision=armed.option_model_revision,
        )


def cast_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value
