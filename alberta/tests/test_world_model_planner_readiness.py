# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator,type-var"
"""Versioned calibration-readiness sidecars for existing WP4 planners."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
from alberta_framework.core.one_step_dyna import (
    OneStepDynaAuthority,
    OneStepDynaConfig,
    RealStateDynaAnchor,
    RealStateOneStepDyna,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)
from alberta_framework.core.world_model_planner_readiness import (
    WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA,
    WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL,
    WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS,
    WorldModelPlannerReadiness,
    WorldModelPlannerReadinessConfig,
    load_world_model_planner_readiness_checkpoint,
    save_world_model_planner_readiness_checkpoint,
)
from alberta_framework.core.world_model_region_calibration import (
    WorldModelCalibrationOutcome,
    WorldModelRegionCalibration,
    WorldModelRegionCalibrationConfig,
)

pytestmark = pytest.mark.integration

OBSERVATION_DIM = 2
N_ACTIONS = 2
ENSEMBLE_SIZE = 2
LIFECYCLE = jnp.asarray((0, 17), dtype=jnp.uint32)
REPRESENTATION_REVISION = jnp.asarray((0, 7), dtype=jnp.uint32)


@pytest.fixture(autouse=True)
def _eager_contracts() -> Iterator[None]:
    with jax.disable_jit():
        yield


def _words(value: int) -> jax.Array:
    return jnp.asarray((0, value), dtype=jnp.uint32)


def _ensemble() -> WorldModelEnsemble:
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=ActionConditionedWorldModelConfig(
                observation_dim=OBSERVATION_DIM,
                n_actions=N_ACTIONS,
                gamma=0.95,
                hidden_sizes=(),
                step_size=0.05,
                sparsity=0.0,
                use_layer_norm=False,
                error_decay=0.8,
            ),
            signal_estimator=LearningSignalEstimatorConfig(
                ensemble_size=ENSEMBLE_SIZE,
                target_dim=OBSERVATION_DIM + 2,
                progress_warmup_steps=2,
                change_calibration_steps=2,
                max_input_magnitude=1_000.0,
                max_predicted_variance=10_000.0,
                max_observed_loss=10_000.0,
            ),
            ensemble_size=ENSEMBLE_SIZE,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-3,
        )
    )


def _set_constant_model_outputs(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
) -> WorldModelEnsembleState:
    outputs = (0.0, 0.0, 1.0, 0.5)
    members = []
    for member in state.member_states:
        learner = member.learner_state
        heads = learner.head_params.replace(
            weights=tuple(
                jnp.zeros_like(weight) for weight in learner.head_params.weights
            ),
            biases=tuple(
                jnp.asarray((value,), dtype=jnp.float32) for value in outputs
            ),
        )
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    result = cast(WorldModelEnsembleState, state.replace(member_states=tuple(members)))
    assert bool(ensemble.state_valid(result))
    return result


def _calibrator() -> WorldModelRegionCalibration:
    return WorldModelRegionCalibration(
        WorldModelRegionCalibrationConfig(
            observation_dim=OBSERVATION_DIM,
            n_actions=N_ACTIONS,
            n_regions=2,
            ensemble_size=ENSEMBLE_SIZE,
            capacity_per_cell=4,
            min_samples=2,
            min_termination_samples=2,
            min_termination_class_support=1,
            alpha=0.1,
            variance_floor=1.0e-6,
            max_aleatoric_variance=10.0,
            max_gaussian_nll=10.0,
            max_next_state_rmse=10.0,
            max_reward_abs_error=10.0,
            max_epistemic_next_state_bound=10.0,
            max_epistemic_reward_bound=10.0,
            max_termination_brier=1.0,
            max_prediction_magnitude=100.0,
            max_outcome_magnitude=100.0,
        )
    )


def _calibrated_state(calibrator: WorldModelRegionCalibration) -> Any:
    state = calibrator.init(LIFECYCLE)
    means = jnp.asarray(
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=jnp.float32,
    )
    variances = jnp.full((2, 3), 0.1, dtype=jnp.float32)
    for decision, terminal_probability, terminated in (
        (1, 0.1, False),
        (2, 0.9, True),
    ):
        receipt = calibrator.issue_prediction(
            state,
            lifecycle_id_words=LIFECYCLE,
            decision_id_words=_words(decision),
            model_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            representation_revision_words=REPRESENTATION_REVISION,
            action_revision_words=_words(decision),
            region_revision_words=_words(10 + decision),
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            member_mean_predictions=means,
            member_aleatoric_variances=variances,
            member_termination_probabilities=jnp.full(
                (2,),
                terminal_probability,
                dtype=jnp.float32,
            ),
        )
        outcome = WorldModelCalibrationOutcome(
            lifecycle_id_words=LIFECYCLE,
            decision_id_words=_words(decision),
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            next_state=jnp.zeros((2,), dtype=jnp.float32),
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(terminated, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
        )
        settled = calibrator.settle(state, receipt, outcome)
        assert bool(settled.transaction.applied)
        state = settled.state
    return state


def _control() -> MultiHeadMLPLearner:
    return MultiHeadMLPLearner(
        n_heads=N_ACTIONS,
        hidden_sizes=(),
        step_size=0.1,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _systems() -> tuple[
    WorldModelPlannerReadiness,
    WorldModelRegionCalibration,
    Any,
    RealStateOneStepDyna,
    Any,
    Any,
    OneStepDynaAuthority,
    EnsembleShortRolloutPlanner,
    Any,
    Any,
]:
    ensemble = _ensemble()
    model_state = _set_constant_model_outputs(
        ensemble,
        ensemble.init(jr.key(1)),
    )
    control = _control()
    control_state = control.init(OBSERVATION_DIM, jr.key(2))
    dyna = RealStateOneStepDyna(
        ensemble,
        control,
        OneStepDynaConfig(
            anchor_capacity=2,
            backup_budget=1,
            min_action_support=1,
            max_epistemic_disagreement=100.0,
            max_residual_variance=100.0,
            require_residual_proxy_ready=False,
            max_anchor_records=20,
            max_planning_calls=20,
            max_planned_backups=20,
        ),
    )
    authority = OneStepDynaAuthority(
        representation_revision_words=REPRESENTATION_REVISION,
        model_revision_words=model_state.event_count_words,
        control_revision_words=control_state.step_words,
    )
    dyna_state = dyna.init(
        jr.key(3),
        REPRESENTATION_REVISION,
        model_state,
        control_state,
    )
    recorded = dyna.record_real_anchor(
        dyna_state,
        model_state,
        control_state,
        RealStateDynaAnchor(
            observation=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            primitive_action=jnp.asarray(0, dtype=jnp.int32),
            decision_id_words=_words(10),
            authority=authority,
        ),
    )
    assert bool(recorded.diagnostics.applied)

    rollout = EnsembleShortRolloutPlanner(
        ensemble,
        EnsembleShortRolloutConfig(
            rollout_horizon=3,
            rollout_budget=1,
            require_residual_proxy_ready=False,
            max_epistemic_disagreement=100.0,
            max_residual_variance=100.0,
            max_proposal_calls=20,
            max_rollout_attempts=20,
            max_imagined_steps=60,
        ),
    )
    rollout_authority = rollout.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray((20.0, -20.0), dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(1.0, dtype=jnp.float32),
        action_support_counts=jnp.asarray((10, 10), dtype=jnp.int32),
        source_revision_words=REPRESENTATION_REVISION,
        model_state=model_state,
        policy_revision_words=_words(8),
        value_revision_words=_words(9),
    )
    rollout_state = rollout.init(jr.key(4), model_state, rollout_authority)
    rollout_anchor = rollout.bind_real_anchor(
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        _words(11),
        rollout_authority,
    )

    calibrator = _calibrator()
    calibration_state = _calibrated_state(calibrator)
    readiness = WorldModelPlannerReadiness(
        calibrator,
        dyna,
        rollout,
        WorldModelPlannerReadinessConfig(
            max_dyna_executions=20,
            max_rollout_executions=20,
        ),
    )
    return (
        readiness,
        calibrator,
        calibration_state,
        dyna,
        recorded.state,
        (model_state, control_state),
        authority,
        rollout,
        (rollout_state, rollout_authority, rollout_anchor),
        control,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(
            jr.key_data(left_leaf)
            if jax.dtypes.issubdtype(jnp.asarray(left_leaf).dtype, jax.dtypes.prng_key)
            else left_leaf
        )
        right_array = np.asarray(
            jr.key_data(right_leaf)
            if jax.dtypes.issubdtype(jnp.asarray(right_leaf).dtype, jax.dtypes.prng_key)
            else right_leaf
        )
        np.testing.assert_array_equal(left_array, right_array)


def test_public_config_resource_and_checkpoint_surface_is_strict(tmp_path: Path) -> None:
    readiness, _, calibration_state, *_ = _systems()
    assert core.WorldModelPlannerReadiness is WorldModelPlannerReadiness
    assert alberta.WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA == (
        WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA
    )
    config = readiness.to_config()
    restored_owner = WorldModelPlannerReadiness.from_config(config)
    assert restored_owner.to_config() == config
    assert config["evidence_level"] == WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL == "L0"
    assert config["outcome_status"] == WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS
    assert config["outcome_status"] == "not_assessed"
    state = readiness.init(calibration_state)
    budget = readiness.resource_budget
    assert budget.planning_authority == 0
    assert budget.safety_authority == 0
    assert budget.model_state_owned == 0
    assert budget.calibration_state_owned == 0

    checkpoint = tmp_path / "readiness"
    save_world_model_planner_readiness_checkpoint(readiness, state, checkpoint)
    restored_state = load_world_model_planner_readiness_checkpoint(
        readiness,
        checkpoint,
    )
    _assert_tree_equal(restored_state, state)


def test_dyna_calibration_is_an_additional_noncompensating_conjunction() -> None:
    (
        readiness,
        _,
        calibration_state,
        dyna,
        dyna_state,
        model_and_control,
        authority,
        *_,
    ) = _systems()
    model_state, control_state = model_and_control
    readiness_state = readiness.init(calibration_state)
    regions = jnp.asarray((0,), dtype=jnp.int32)
    action_revisions = jnp.asarray(((0, 1),), dtype=jnp.uint32)
    region_revisions = jnp.asarray(((0, 2),), dtype=jnp.uint32)
    receipt = readiness.prepare_dyna(
        readiness_state,
        calibration_state,
        dyna_state,
        model_state,
        control_state,
        authority,
        region_ids=regions,
        action_revision_words=action_revisions,
        region_revision_words=region_revisions,
    )
    assert bool(receipt.valid)
    assert bool(receipt.legacy_guard_passed[0])
    assert bool(receipt.calibration_gate_passed[0])
    assert bool(receipt.combined_gate_passed[0])
    assert not bool(receipt.planning_authority)
    assert not bool(receipt.safety_authority)
    executed = readiness.execute_dyna(
        readiness_state,
        calibration_state,
        dyna_state,
        model_state,
        control_state,
        authority,
        receipt,
    )
    assert bool(executed.diagnostics.applied)
    assert int(executed.readiness_state.dyna_execution_count_words[1]) == 1
    assert int(executed.dyna_state.planning_call_count_words[1]) == 1
    assert int(executed.control_state.step_words[1]) == 1

    # Reusing the exact receipt is stale at the readiness-owner boundary.
    replay = readiness.execute_dyna(
        executed.readiness_state,
        calibration_state,
        executed.dyna_state,
        model_state,
        executed.control_state,
        OneStepDynaAuthority(
            representation_revision_words=authority.representation_revision_words,
            model_revision_words=authority.model_revision_words,
            control_revision_words=executed.control_state.step_words,
        ),
        receipt,
    )
    assert bool(replay.diagnostics.rejected)
    _assert_tree_equal(replay.readiness_state, executed.readiness_state)
    _assert_tree_equal(replay.dyna_state, executed.dyna_state)
    _assert_tree_equal(replay.control_state, executed.control_state)

    # The unchanged legacy planner can propose an update from an uncalibrated
    # state, while the sidecar vetoes it and rolls back every child state.
    unready_calibration = readiness.calibrator.init(LIFECYCLE)
    unready_owner_state = readiness.init(unready_calibration)
    legacy = dyna.plan(dyna_state, model_state, control_state, authority)
    assert bool(legacy.diagnostics.guard_passed[0])
    unready_receipt = readiness.prepare_dyna(
        unready_owner_state,
        unready_calibration,
        dyna_state,
        model_state,
        control_state,
        authority,
        region_ids=regions,
        action_revision_words=action_revisions,
        region_revision_words=region_revisions,
    )
    assert bool(unready_receipt.valid)
    assert not bool(unready_receipt.calibration_gate_passed[0])
    vetoed = readiness.execute_dyna(
        unready_owner_state,
        unready_calibration,
        dyna_state,
        model_state,
        control_state,
        authority,
        unready_receipt,
    )
    assert bool(vetoed.diagnostics.rejected)
    _assert_tree_equal(vetoed.readiness_state, unready_owner_state)
    _assert_tree_equal(vetoed.dyna_state, dyna_state)
    _assert_tree_equal(vetoed.control_state, control_state)


def test_dyna_receipt_tamper_and_newer_calibration_state_are_atomic_rejections() -> None:
    (
        readiness,
        calibrator,
        calibration_state,
        _,
        dyna_state,
        model_and_control,
        authority,
        *_,
    ) = _systems()
    model_state, control_state = model_and_control
    readiness_state = readiness.init(calibration_state)
    receipt = readiness.prepare_dyna(
        readiness_state,
        calibration_state,
        dyna_state,
        model_state,
        control_state,
        authority,
        region_ids=jnp.asarray((0,), dtype=jnp.int32),
        action_revision_words=jnp.asarray(((0, 1),), dtype=jnp.uint32),
        region_revision_words=jnp.asarray(((0, 2),), dtype=jnp.uint32),
    )
    tampered = dataclasses.replace(
        receipt,
        calibration_cell_content_tags=receipt.calibration_cell_content_tags.at[0].add(1),
    )
    rejected = readiness.execute_dyna(
        readiness_state,
        calibration_state,
        dyna_state,
        model_state,
        control_state,
        authority,
        tampered,
    )
    assert bool(rejected.diagnostics.rejected)
    _assert_tree_equal(rejected.readiness_state, readiness_state)

    newer_prediction = calibrator.issue_prediction(
        calibration_state,
        lifecycle_id_words=LIFECYCLE,
        decision_id_words=_words(3),
        model_revision_words=model_state.event_count_words,
        representation_revision_words=REPRESENTATION_REVISION,
        action_revision_words=_words(3),
        region_revision_words=_words(13),
        action=jnp.asarray(0, dtype=jnp.int32),
        region=jnp.asarray(0, dtype=jnp.int32),
        member_mean_predictions=jnp.asarray(
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            dtype=jnp.float32,
        ),
        member_aleatoric_variances=jnp.full((2, 3), 0.1, dtype=jnp.float32),
        member_termination_probabilities=jnp.full((2,), 0.1, dtype=jnp.float32),
    )
    newer = calibrator.settle(
        calibration_state,
        newer_prediction,
        WorldModelCalibrationOutcome(
            lifecycle_id_words=LIFECYCLE,
            decision_id_words=_words(3),
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            next_state=jnp.zeros((2,), dtype=jnp.float32),
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    assert bool(newer.transaction.applied)
    stale = readiness.execute_dyna(
        readiness_state,
        newer.state,
        dyna_state,
        model_state,
        control_state,
        authority,
        receipt,
    )
    assert bool(stale.diagnostics.rejected)
    _assert_tree_equal(stale.readiness_state, readiness_state)


def test_short_rollout_calibration_is_prefix_closed_and_rolls_back_whole_call() -> None:
    (
        readiness,
        _,
        calibration_state,
        _,
        _,
        model_and_control,
        _,
        rollout,
        rollout_bundle,
        _,
    ) = _systems()
    model_state, _ = model_and_control
    rollout_state, authority, anchor = rollout_bundle
    readiness_state = readiness.init(calibration_state)
    regions = jnp.zeros((1, 3), dtype=jnp.int32)
    action_revisions = jnp.broadcast_to(_words(20), (1, 3, 2))
    region_revisions = jnp.broadcast_to(_words(21), (1, 3, 2))
    receipt = readiness.prepare_short_rollout(
        readiness_state,
        calibration_state,
        rollout_state,
        model_state,
        authority,
        anchor,
        region_ids=regions,
        action_revision_words=action_revisions,
        region_revision_words=region_revisions,
    )
    assert bool(receipt.valid)
    assert bool(jnp.all(receipt.legacy_guard_passed))
    assert bool(jnp.all(receipt.calibration_gate_passed))
    assert bool(jnp.all(receipt.prefix_eligible))
    assert not bool(receipt.planning_authority)
    executed = readiness.execute_short_rollout(
        readiness_state,
        calibration_state,
        rollout_state,
        model_state,
        authority,
        anchor,
        receipt,
    )
    assert bool(executed.diagnostics.applied)
    assert bool(executed.proposals.path_accepted[0])
    assert int(executed.readiness_state.rollout_execution_count_words[1]) == 1

    legacy = rollout.propose(rollout_state, model_state, authority, anchor)
    assert bool(legacy.proposals.path_accepted[0])
    mixed_regions = regions.at[0, 1].set(1)
    veto_receipt = readiness.prepare_short_rollout(
        readiness_state,
        calibration_state,
        rollout_state,
        model_state,
        authority,
        anchor,
        region_ids=mixed_regions,
        action_revision_words=action_revisions,
        region_revision_words=region_revisions,
    )
    np.testing.assert_array_equal(veto_receipt.calibration_gate_passed[0], [True, False, True])
    np.testing.assert_array_equal(veto_receipt.prefix_eligible[0], [True, False, False])
    vetoed = readiness.execute_short_rollout(
        readiness_state,
        calibration_state,
        rollout_state,
        model_state,
        authority,
        anchor,
        veto_receipt,
    )
    assert bool(vetoed.diagnostics.rejected)
    _assert_tree_equal(vetoed.readiness_state, readiness_state)
    _assert_tree_equal(vetoed.rollout_state, rollout_state)
    assert not bool(jnp.any(vetoed.proposals.transition_valid))
