# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator,type-var"
"""Contracts for bounded online state/action-region world-model calibration."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.world_model_region_calibration import (
    WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA,
    WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL,
    WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS,
    WorldModelCalibrationOutcome,
    WorldModelPredictBeforeOutcomeReceipt,
    WorldModelRegionCalibration,
    WorldModelRegionCalibrationConfig,
    load_world_model_region_calibration_checkpoint,
    measure_world_model_region_calibration_state_nbytes,
    save_world_model_region_calibration_checkpoint,
)

pytestmark = pytest.mark.integration

OBSERVATION_DIM = 1
N_ACTIONS = 2
N_REGIONS = 2
ENSEMBLE_SIZE = 2


@pytest.fixture(autouse=True)
def _bounded_jax(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_eager_jit_and_scan_are_deterministic":
        yield
    else:
        with jax.disable_jit():
            yield


def _config(**overrides: object) -> WorldModelRegionCalibrationConfig:
    values: dict[str, object] = {
        "observation_dim": OBSERVATION_DIM,
        "n_actions": N_ACTIONS,
        "n_regions": N_REGIONS,
        "ensemble_size": ENSEMBLE_SIZE,
        "capacity_per_cell": 4,
        "min_samples": 2,
        "min_termination_samples": 2,
        "min_termination_class_support": 1,
        "alpha": 0.1,
        "max_aleatoric_variance": 1.0,
        "max_gaussian_nll": 10.0,
        "max_next_state_rmse": 10.0,
        "max_reward_abs_error": 10.0,
        "max_epistemic_next_state_bound": 10.0,
        "max_epistemic_reward_bound": 10.0,
        "max_termination_brier": 1.0,
        "max_prediction_magnitude": 100.0,
        "max_outcome_magnitude": 100.0,
    }
    values.update(overrides)
    return WorldModelRegionCalibrationConfig(**values)  # type: ignore[arg-type]


def _words(value: int) -> jax.Array:
    return jnp.asarray([0, value], dtype=jnp.uint32)


def _issue(
    owner: WorldModelRegionCalibration,
    state: Any,
    decision: int,
    *,
    action: int = 0,
    region: int = 0,
    means: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (2.0, 2.0),
    ),
    variances: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 1.0),
        (1.0, 1.0),
    ),
    termination: tuple[float, float] = (0.2, 0.4),
) -> WorldModelPredictBeforeOutcomeReceipt:
    return owner.issue_prediction(
        state,
        lifecycle_id_words=_words(17),
        decision_id_words=_words(decision),
        model_revision_words=_words(100 + decision),
        representation_revision_words=_words(200 + decision),
        action_revision_words=_words(300 + decision),
        region_revision_words=_words(400 + decision),
        action=jnp.asarray(action, dtype=jnp.int32),
        region=jnp.asarray(region, dtype=jnp.int32),
        member_mean_predictions=jnp.asarray(means, dtype=jnp.float32),
        member_aleatoric_variances=jnp.asarray(variances, dtype=jnp.float32),
        member_termination_probabilities=jnp.asarray(
            termination,
            dtype=jnp.float32,
        ),
    )


def _outcome(
    decision: int,
    *,
    action: int = 0,
    region: int = 0,
    next_state: float = 2.0,
    reward: float = 3.0,
    terminated: bool = False,
    truncated: bool = False,
) -> WorldModelCalibrationOutcome:
    return WorldModelCalibrationOutcome(
        lifecycle_id_words=_words(17),
        decision_id_words=_words(decision),
        action=jnp.asarray(action, dtype=jnp.int32),
        region=jnp.asarray(region, dtype=jnp.int32),
        next_state=jnp.asarray([next_state], dtype=jnp.float32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_public_surface_initial_state_and_resources_are_strict() -> None:
    config = _config()
    payload = config.to_config()
    assert WorldModelRegionCalibrationConfig.from_config(payload) == config
    assert payload["evidence_level"] == WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL == "L0"
    assert payload["outcome_status"] == WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS
    assert payload["outcome_status"] == "not_assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert core.WorldModelRegionCalibration is WorldModelRegionCalibration
    assert alberta.WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA == (
        WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA
    )

    extra = dict(payload)
    extra["invented"] = True
    with pytest.raises(ValueError, match="fields"):
        WorldModelRegionCalibrationConfig.from_config(extra)
    with pytest.raises(ValueError, match="alpha"):
        dataclasses.replace(config, alpha=0.0)

    owner = WorldModelRegionCalibration(config)
    state = owner.init(_words(17))
    assert bool(owner.state_valid(state))
    assert int(state.cell_sizes.sum()) == 0
    assert int(state.accepted_count_words[1]) == 0
    budget = owner.resource_budget
    assert budget.persistent_state_bytes == measure_world_model_region_calibration_state_nbytes(
        state
    )
    assert budget.region_action_cells == N_REGIONS * N_ACTIONS
    assert budget.records_per_cell == config.capacity_per_cell
    assert budget.planning_authority == 0
    assert budget.safety_authority == 0
    assert budget.model_state_owned == 0


def test_hand_computed_channels_use_only_causal_preupdate_cell_history() -> None:
    owner = WorldModelRegionCalibration(_config())
    state = owner.init(_words(17))

    first_receipt = _issue(owner, state, 1)
    assert bool(owner.receipt_valid(state, first_receipt))
    assert not bool(first_receipt.gates.epistemic.available)
    assert not bool(first_receipt.gates.aleatoric.available)
    assert not bool(first_receipt.gates.next_state_error.available)
    assert not bool(first_receipt.gates.reward_error.available)
    assert not bool(first_receipt.gates.termination.available)
    assert not bool(first_receipt.planning_authority)
    assert not bool(first_receipt.safety_authority)
    first = owner.settle(state, first_receipt, _outcome(1, terminated=False))
    assert bool(first.transaction.applied)
    np.testing.assert_allclose(first.outcome.next_state_squared_error, 1.0)
    np.testing.assert_allclose(first.outcome.reward_squared_error, 4.0)
    np.testing.assert_allclose(first.outcome.next_state_epistemic_disagreement, 1.0)
    np.testing.assert_allclose(first.outcome.reward_epistemic_disagreement, 1.0)
    np.testing.assert_allclose(first.outcome.next_state_mean_aleatoric_variance, 1.0)
    np.testing.assert_allclose(first.outcome.reward_aleatoric_variance, 1.0)
    np.testing.assert_allclose(first.outcome.absolute_standardized_residuals, [1.0, 2.0])
    expected_nll = 0.5 * (np.log(2.0 * np.pi) + np.asarray([1.0, 4.0]))
    np.testing.assert_allclose(first.outcome.gaussian_nll, expected_nll, rtol=1e-6)
    assert bool(first.outcome.termination_observed)
    assert not bool(first.outcome.termination_target)
    np.testing.assert_allclose(first.outcome.termination_brier_error, 0.09)

    second_receipt = _issue(owner, first.state, 2)
    assert not bool(second_receipt.gates.next_state_error.available)
    second = owner.settle(
        first.state,
        second_receipt,
        _outcome(2, terminated=True),
    )
    assert bool(second.transaction.applied)

    # The third receipt sees exactly the first two settled records.  No third
    # outcome has entered any threshold or quantile.
    third = _issue(owner, second.state, 3)
    assert int(third.gates.next_state_error.support_count) == 2
    assert bool(third.gates.next_state_error.available)
    assert bool(third.gates.reward_error.available)
    np.testing.assert_allclose(third.gates.next_state_error.preupdate_rmse_quantile, 1.0)
    np.testing.assert_allclose(third.gates.reward_error.preupdate_abs_error_quantile, 2.0)
    np.testing.assert_allclose(
        third.gates.epistemic.preupdate_next_state_error_ratio_quantile,
        0.0,
    )
    np.testing.assert_allclose(
        third.gates.epistemic.preupdate_reward_error_ratio_quantile,
        3.0,
    )
    np.testing.assert_allclose(
        third.gates.aleatoric.preupdate_standardized_residual_quantiles,
        [1.0, 2.0],
    )
    assert int(third.gates.termination.terminal_support_count) == 1
    assert int(third.gates.termination.continuing_support_count) == 1
    assert bool(third.gates.termination.available)


def test_region_and_action_cells_isolate_support_and_ood_is_unavailable() -> None:
    owner = WorldModelRegionCalibration(_config())
    state = owner.init(_words(17))
    for decision in (1, 2):
        receipt = _issue(owner, state, decision, action=0, region=0)
        result = owner.settle(
            state,
            receipt,
            _outcome(decision, action=0, region=0, terminated=decision == 2),
        )
        assert bool(result.transaction.applied)
        state = result.state

    known = _issue(owner, state, 3, action=0, region=0)
    ood_region = _issue(owner, state, 3, action=0, region=1)
    unseen_action = _issue(owner, state, 3, action=1, region=0)
    assert bool(known.gates.next_state_error.available)
    assert not bool(ood_region.gates.next_state_error.available)
    assert not bool(unseen_action.gates.next_state_error.available)
    assert int(known.cell_revision_words[1]) == 2
    assert int(ood_region.cell_revision_words[1]) == 0
    assert int(unseen_action.cell_revision_words[1]) == 0


def test_noisy_tv_high_aleatoric_channel_is_a_separate_noncompensating_veto() -> None:
    owner = WorldModelRegionCalibration(_config(max_aleatoric_variance=0.5))
    state = owner.init(_words(17))
    low_variance = ((0.1, 0.1), (0.1, 0.1))
    for decision in (1, 2):
        receipt = _issue(owner, state, decision, variances=low_variance)
        result = owner.settle(
            state,
            receipt,
            _outcome(decision, next_state=1.0, reward=1.0, terminated=decision == 2),
        )
        assert bool(result.transaction.applied)
        state = result.state

    noisy = _issue(
        owner,
        state,
        3,
        means=((1.0, 1.0), (1.0, 1.0)),
        variances=((10.0, 10.0), (10.0, 10.0)),
    )
    assert bool(noisy.gates.aleatoric.available)
    assert bool(noisy.gates.aleatoric.noise_vetoed)
    assert not bool(noisy.gates.aleatoric.passed)
    # Zero member disagreement remains zero; raw/noisy realized error is never
    # relabeled as epistemic disagreement.
    np.testing.assert_array_equal(noisy.gates.epistemic.current_disagreements, [0.0, 0.0])
    assert not bool(noisy.planning_authority)


def test_stale_tampered_nonfinite_and_terminal_semantics_roll_back_atomically() -> None:
    owner = WorldModelRegionCalibration(_config())
    initial = owner.init(_words(17))
    receipt = _issue(owner, initial, 1)
    accepted = owner.settle(initial, receipt, _outcome(1))
    assert bool(accepted.transaction.applied)

    stale = owner.settle(accepted.state, receipt, _outcome(1))
    assert bool(stale.transaction.rejected)
    _assert_tree_equal(stale.state, accepted.state)

    fresh = _issue(owner, accepted.state, 2)
    tampered = dataclasses.replace(
        fresh,
        member_mean_predictions=fresh.member_mean_predictions.at[0, 0].add(1.0),
    )
    rejected_tamper = owner.settle(accepted.state, tampered, _outcome(2))
    assert bool(rejected_tamper.transaction.rejected)
    _assert_tree_equal(rejected_tamper.state, accepted.state)

    nonfinite = dataclasses.replace(
        _outcome(2),
        reward=jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    rejected_nonfinite = owner.settle(accepted.state, fresh, nonfinite)
    assert bool(rejected_nonfinite.transaction.rejected)
    _assert_tree_equal(rejected_nonfinite.state, accepted.state)

    both_boundaries = _outcome(2, terminated=True, truncated=True)
    rejected_boundaries = owner.settle(accepted.state, fresh, both_boundaries)
    assert bool(rejected_boundaries.transaction.rejected)
    _assert_tree_equal(rejected_boundaries.state, accepted.state)

    truncated = owner.settle(accepted.state, fresh, _outcome(2, truncated=True))
    assert bool(truncated.transaction.applied)
    assert not bool(truncated.outcome.termination_observed)
    assert not bool(truncated.outcome.termination_target)
    assert int(truncated.state.termination_support_counts[0, 0]) == 1


def test_checkpoint_roundtrip_and_resource_shape_are_lifecycle_bound(tmp_path: Path) -> None:
    owner = WorldModelRegionCalibration(_config())
    state = owner.init(_words(17))
    for decision in range(1, 7):
        result = owner.settle(
            state,
            _issue(owner, state, decision),
            _outcome(decision, terminated=decision % 2 == 0),
        )
        assert bool(result.transaction.applied)
        state = result.state
    assert int(state.cell_sizes[0, 0]) == 4
    assert int(state.cell_write_indices[0, 0]) == 2
    assert int(state.cell_count_words[0, 0, 1]) == 6
    assert bool(jnp.all(state.record_valid[0, 0]))
    checkpoint = tmp_path / "region-calibration"
    save_world_model_region_calibration_checkpoint(owner, state, checkpoint)
    restored_owner, restored_state = load_world_model_region_calibration_checkpoint(
        checkpoint
    )
    assert restored_owner.to_config() == owner.to_config()
    _assert_tree_equal(restored_state, state)
    assert restored_owner.resource_budget == owner.resource_budget

    tampered_checkpoint = tmp_path / "wrong-schema"
    tampered_metadata = load_checkpoint_metadata(checkpoint)
    tampered_metadata["schema"] = "alberta.world-model-region-calibration.checkpoint.v2"
    save_checkpoint(state, tampered_checkpoint, metadata=tampered_metadata)
    with pytest.raises(ValueError, match="v1 checkpoint"):
        load_world_model_region_calibration_checkpoint(tampered_checkpoint)

    wrong_lifecycle = dataclasses.replace(
        _outcome(7),
        lifecycle_id_words=_words(99),
    )
    receipt = _issue(restored_owner, restored_state, 7)
    rejected = restored_owner.settle(restored_state, receipt, wrong_lifecycle)
    assert bool(rejected.transaction.rejected)
    _assert_tree_equal(rejected.state, restored_state)


def test_eager_jit_and_scan_are_deterministic() -> None:
    owner = WorldModelRegionCalibration(_config())
    initial = owner.init(_words(17))

    def step(state: Any, decision: jax.Array) -> tuple[Any, Any]:
        receipt = owner.issue_prediction(
            state,
            lifecycle_id_words=_words(17),
            decision_id_words=jnp.stack(
                (jnp.asarray(0, dtype=jnp.uint32), decision.astype(jnp.uint32))
            ),
            model_revision_words=_words(100),
            representation_revision_words=_words(200),
            action_revision_words=_words(300),
            region_revision_words=_words(400),
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            member_mean_predictions=jnp.asarray(((0.0, 0.0), (2.0, 2.0)), jnp.float32),
            member_aleatoric_variances=jnp.ones((2, 2), dtype=jnp.float32),
            member_termination_probabilities=jnp.asarray((0.2, 0.4), jnp.float32),
        )
        outcome = WorldModelCalibrationOutcome(
            lifecycle_id_words=_words(17),
            decision_id_words=receipt.decision_id_words,
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            next_state=jnp.asarray((2.0,), dtype=jnp.float32),
            reward=jnp.asarray(3.0, dtype=jnp.float32),
            terminated=(decision == jnp.asarray(2, dtype=jnp.int32)),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
        )
        settled = owner.settle(state, receipt, outcome)
        return settled.state, settled.transaction.applied

    decisions = jnp.asarray((1, 2, 3), dtype=jnp.int32)
    eager_state = initial
    eager_applied = []
    with jax.disable_jit():
        for decision in decisions:
            eager_state, applied = step(eager_state, decision)
            eager_applied.append(applied)
    jitted_state, jitted_applied = jax.jit(lambda s, xs: jax.lax.scan(step, s, xs))(
        initial,
        decisions,
    )
    _assert_tree_equal(eager_state, jitted_state)
    np.testing.assert_array_equal(jitted_applied, jnp.stack(eager_applied))
