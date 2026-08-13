"""Structural checks for the development-only world/agent factorization probe."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import math
from collections.abc import Generator
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.behavior_model import measure_behavior_model_state_nbytes
from alberta_framework.core.grounded_joint_world_model import (
    measure_grounded_joint_world_state_nbytes,
)
from alberta_framework.evaluation import (
    world_agent_factorization_recurrence_development as factorization_module,
)
from alberta_framework.evaluation.world_agent_factorization_recurrence_development import (
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    DEVELOPMENT_KEY_FROZEN,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SCHEMA,
    EVIDENCE_CLAIMED,
    OUTPUT_WRITES_ALLOWED,
    PHASE_NAMES,
    RESETS_EXPOSED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    TASK_IDENTIFIERS_EXPOSED,
    THRESHOLDS_FROZEN,
    FactorizationFeedback,
    FactorizationMetrics,
    FactorizationPreObservation,
    WorldAgentFactorizationRecurrenceConfig,
    WorldAgentFactorizationRecurrenceReport,
    WorldAgentFactorizationSource,
    build_world_agent_factorization_source,
    run_world_agent_factorization_recurrence_development,
    validate_world_agent_factorization_recurrence_report,
    validate_world_agent_factorization_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module", autouse=True)
def _release_compilation_cache() -> Generator[None, None, None]:
    """Release the small module's compiled executables after all checks."""

    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()


@pytest.fixture(scope="module")
def small_report() -> WorldAgentFactorizationRecurrenceReport:
    return run_world_agent_factorization_recurrence_development(
        WorldAgentFactorizationRecurrenceConfig(
            phase_steps=8,
            summary_window=3,
            behavior_step_size=0.2,
            world_step_size=0.2,
            development_key=17,
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("phase_steps", True, TypeError),
        ("phase_steps", 3, ValueError),
        ("phase_steps", 6, ValueError),
        ("summary_window", 0, ValueError),
        ("summary_window", 33, ValueError),
        ("behavior_step_size", 1, TypeError),
        ("behavior_step_size", np.float32(0.2), TypeError),
        ("behavior_step_size", float("nan"), ValueError),
        ("behavior_step_size", 1.0e300, ValueError),
        ("behavior_step_size", 1.0e-300, ValueError),
        ("world_step_size", 1, TypeError),
        ("world_step_size", np.float32(0.2), TypeError),
        ("world_step_size", 1.1, ValueError),
        ("world_step_size", 1.0e-300, ValueError),
        ("world_initialization_scale", False, TypeError),
        ("world_initialization_scale", np.float32(0.01), TypeError),
        ("world_initialization_scale", 1.0e300, ValueError),
        ("world_initialization_scale", 1.0e-300, ValueError),
        ("development_key", -1, ValueError),
    ],
)
def test_config_fails_closed(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        WorldAgentFactorizationRecurrenceConfig(**{field: value})  # type: ignore[arg-type]


def test_source_is_deterministic_uninterrupted_and_hides_phase_from_learner() -> None:
    config = WorldAgentFactorizationRecurrenceConfig(
        phase_steps=8,
        summary_window=2,
        development_key=23,
    )
    source = build_world_agent_factorization_source(config)
    replay = build_world_agent_factorization_source(config)

    assert validate_world_agent_factorization_source(source) == ()
    assert source.input_sha256 == replay.input_sha256
    assert source.generator_contract_sha256 == replay.generator_contract_sha256
    assert source.behavior_features.shape == (24, 2)
    assert source.behavior_features.dtype == jnp.float32
    assert source.world_representations.shape == (24, 2)
    assert source.focal_actions.shape == (24,)
    assert source.focal_actions.dtype == jnp.int32
    assert source.partner_actions.shape == (24,)
    assert source.next_observations.shape == (24, 2)
    assert source.rewards.shape == (24,)
    assert source.discounts.shape == (24,)
    assert source.evaluator_phase_ids.shape == (24,)

    signal = np.argmax(np.asarray(source.behavior_features), axis=1)
    partner = np.asarray(source.partner_actions)
    np.testing.assert_array_equal(partner[:8], signal[:8])
    np.testing.assert_array_equal(partner[8:16], 1 - signal[8:16])
    np.testing.assert_array_equal(partner[16:], signal[16:])
    np.testing.assert_array_equal(
        np.asarray(source.focal_actions),
        np.tile(np.asarray((0, 0, 1, 1), dtype=np.int32), 6),
    )
    np.testing.assert_array_equal(
        np.asarray(source.evaluator_phase_ids),
        np.repeat(np.arange(3, dtype=np.int32), 8),
    )

    assert {field.name for field in dataclasses.fields(FactorizationPreObservation)} == {
        "behavior_features",
        "world_representation",
        "focal_action",
    }
    assert {field.name for field in dataclasses.fields(FactorizationFeedback)} == {
        "partner_action",
        "next_observation",
        "reward",
        "discount",
    }
    learner_fields = {
        field.name for field in dataclasses.fields(FactorizationPreObservation)
    } | {field.name for field in dataclasses.fields(FactorizationFeedback)}
    assert not learner_fields & {
        "phase",
        "phase_id",
        "task",
        "task_id",
        "reset",
        "step",
        "evaluator_phase_ids",
    }


def test_pre_action_prediction_boundary_precedes_feedback_access() -> None:
    parameters = tuple(
        inspect.signature(factorization_module._pre_action_predictions).parameters
    )
    assert parameters == (
        "behavior",
        "behavior_state",
        "world",
        "world_state",
        "pre",
    )
    runner_source = inspect.getsource(run_world_agent_factorization_recurrence_development)
    assert runner_source.index("_pre_action_predictions(") < runner_source.index(
        "_feedback_at("
    )


def test_report_is_explicitly_not_assessed_and_has_exact_resources(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    report = small_report
    assert validate_world_agent_factorization_recurrence_report(report) == ()
    assert DEVELOPMENT_SCHEMA == "alberta.world-agent-factorization-recurrence.development.v1"
    assert DEVELOPMENT_ONLY is True
    assert ASSESSMENT_STATUS == "not_assessed"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert BENCHMARK_EXECUTION_AUTHORITY is False
    assert ARTIFACT_AUTHORITY is False
    assert OUTPUT_WRITES_ALLOWED is False
    assert EVIDENCE_CLAIMED is False
    assert THRESHOLDS_FROZEN is False
    assert DEVELOPMENT_KEY_FROZEN is False
    assert TASK_IDENTIFIERS_EXPOSED is False
    assert RESETS_EXPOSED is False

    assert report.development_only is True
    assert report.assessment_status == "not_assessed"
    assert report.scientific_promotion_allowed is False
    assert report.benchmark_execution_authority is False
    assert report.artifact_authority is False
    assert report.output_writes_allowed is False
    assert report.evidence_claimed is False
    assert report.task_identifiers_exposed is False
    assert report.resets_exposed is False
    assert report.learner_reset_count == 0
    assert report.descriptive_claims_only is True
    assert len(report.limitations) >= 5

    resource = report.resource
    assert resource.behavior_initial_state_nbytes == measure_behavior_model_state_nbytes(
        report.initial_behavior_state
    )
    assert resource.behavior_final_state_nbytes == measure_behavior_model_state_nbytes(
        report.final_behavior_state
    )
    assert resource.world_initial_state_nbytes == measure_grounded_joint_world_state_nbytes(
        report.initial_world_state
    )
    assert resource.world_final_state_nbytes == measure_grounded_joint_world_state_nbytes(
        report.final_world_state
    )
    assert resource.initial_total_state_nbytes == (
        resource.behavior_initial_state_nbytes + resource.world_initial_state_nbytes
    )
    assert resource.initial_total_state_nbytes == resource.final_total_state_nbytes
    assert resource.fixed_state_nbytes is True
    assert resource.logical_preupdate_float32_scalars_per_step == 18
    assert resource.logical_preupdate_work_nbytes_per_step == 72
    expected_trajectory_nbytes = sum(
        int(np.asarray(getattr(report.trajectory, field.name)).nbytes)
        for field in dataclasses.fields(report.trajectory)
    )
    assert resource.trajectory_nbytes == expected_trajectory_nbytes
    assert resource.partner_world_cells_evaluated_per_step == 2
    assert resource.behavior_updates_per_step == 1
    assert resource.world_updates_per_step == 1
    assert resource.replay_capacity == 0
    assert resource.passes_over_source == 1


def test_validators_guard_exact_types_before_dereference_or_reconstruction(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    wrong_report = cast(WorldAgentFactorizationRecurrenceReport, object())
    assert validate_world_agent_factorization_recurrence_report(wrong_report) == (
        "report type differs",
    )

    wrong_source = cast(WorldAgentFactorizationSource, object())
    assert validate_world_agent_factorization_source(wrong_source) == (
        "source type differs",
    )

    wrong_config = cast(WorldAgentFactorizationRecurrenceConfig, object())
    source_with_wrong_config = dataclasses.replace(
        small_report.source,
        config=wrong_config,
    )
    assert validate_world_agent_factorization_source(source_with_wrong_config) == (
        "source config type differs",
    )

    report_with_wrong_config = dataclasses.replace(
        small_report,
        config=wrong_config,
    )
    report_errors = validate_world_agent_factorization_recurrence_report(
        report_with_wrong_config
    )
    assert "report config type differs" in report_errors
    assert "report config is not exactly bound to the source config" in report_errors

    malformed_cases = (
        (
            dataclasses.replace(small_report, trajectory=cast(Any, object())),
            "report trajectory type differs",
        ),
        (
            dataclasses.replace(small_report, resource=cast(Any, object())),
            "report resource type differs",
        ),
        (
            dataclasses.replace(small_report, recurrence=cast(Any, object())),
            "report recurrence type differs",
        ),
        (
            dataclasses.replace(small_report, initial_behavior_state=cast(Any, object())),
            "initial behavior state type differs",
        ),
        (
            dataclasses.replace(small_report, final_behavior_state=cast(Any, object())),
            "final behavior state type differs",
        ),
        (
            dataclasses.replace(small_report, initial_world_state=cast(Any, object())),
            "initial world state type differs",
        ),
        (
            dataclasses.replace(small_report, final_world_state=cast(Any, object())),
            "final world state type differs",
        ),
        (
            dataclasses.replace(small_report, phase_summaries=cast(Any, object())),
            "report phase summaries type differs",
        ),
    )
    for malformed, expected_error in malformed_cases:
        assert expected_error in validate_world_agent_factorization_recurrence_report(
            malformed
        )


def test_report_config_limitations_and_derived_metrics_are_exact_bindings(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    mismatched_config = dataclasses.replace(
        small_report.config,
        development_key=small_report.config.development_key + 1,
    )
    config_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(small_report, config=mismatched_config)
    )
    assert "report config is not exactly bound to the source config" in config_errors

    limitation_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            limitations=small_report.limitations + ("tampered",),
        )
    )
    assert "report limitations differ" in limitation_errors

    first_phase = small_report.phase_summaries[0]
    phase_metrics = dataclasses.replace(
        first_phase.metrics,
        behavior_nll=math.nextafter(first_phase.metrics.behavior_nll, math.inf),
    )
    phase_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            phase_summaries=(
                dataclasses.replace(first_phase, metrics=phase_metrics),
                small_report.phase_summaries[1],
                small_report.phase_summaries[2],
            ),
        )
    )
    assert "phase summary A_initial differs" in phase_errors

    recurrence_late = small_report.recurrence.recurrence_late
    recurrence_metrics = dataclasses.replace(
        recurrence_late,
        marginal_world_mse=math.nextafter(
            recurrence_late.marginal_world_mse,
            math.inf,
        ),
    )
    recurrence_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            recurrence=dataclasses.replace(
                small_report.recurrence,
                recurrence_late=recurrence_metrics,
            ),
        )
    )
    assert "recurrence summary recurrence_late differs" in recurrence_errors


def test_predictions_are_bound_before_updates_and_clocks_never_reset(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    report = small_report
    source = report.source
    trajectory = report.trajectory
    total = report.config.total_steps

    np.testing.assert_array_equal(
        trajectory.behavior_probabilities_pre,
        trajectory.behavior_probabilities_update,
    )
    rows = np.arange(total)
    partner = np.asarray(source.partner_actions)
    conditional_from_cells = np.asarray(trajectory.world_predictions_by_partner_pre)[
        rows, partner
    ]
    np.testing.assert_array_equal(
        conditional_from_cells,
        np.asarray(trajectory.conditional_world_predictions_pre),
    )
    np.testing.assert_array_equal(
        conditional_from_cells,
        np.asarray(trajectory.conditional_world_predictions_update),
    )
    assert bool(jnp.all(trajectory.behavior_prediction_bound))
    assert bool(jnp.all(trajectory.world_prediction_bound))
    assert bool(jnp.all(trajectory.behavior_update_applied))
    assert bool(jnp.all(trajectory.world_update_applied))

    pre_words = np.stack(
        (
            np.zeros((total,), dtype=np.uint32),
            np.arange(total, dtype=np.uint32),
        ),
        axis=1,
    )
    post_words = pre_words.copy()
    post_words[:, 1] += np.uint32(1)
    np.testing.assert_array_equal(trajectory.behavior_pre_words, pre_words)
    np.testing.assert_array_equal(trajectory.behavior_post_words, post_words)
    np.testing.assert_array_equal(trajectory.world_pre_words, pre_words)
    np.testing.assert_array_equal(trajectory.world_post_words, post_words)
    np.testing.assert_array_equal(
        report.final_behavior_state.step_words,
        np.asarray((0, total), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        report.final_world_state.update_words,
        np.asarray((0, total), dtype=np.uint32),
    )

    selected = np.asarray(source.focal_actions) * 2 + partner
    np.testing.assert_array_equal(trajectory.selected_joint_action_index, selected)
    selected_mask = np.eye(4, dtype=np.bool_)[selected]
    assert not bool(
        np.any(np.asarray(trajectory.world_weight_row_change_mask) & ~selected_mask)
    )
    assert not bool(
        np.any(np.asarray(trajectory.world_bias_row_change_mask) & ~selected_mask)
    )


def test_four_metric_channels_are_independently_reconstructable(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    report = small_report
    trajectory = report.trajectory
    probabilities = np.asarray(trajectory.behavior_probabilities_pre)
    partner = np.asarray(report.source.partner_actions)
    rows = np.arange(report.config.total_steps)
    one_hot = np.eye(2, dtype=np.float32)[partner]
    cells = np.asarray(trajectory.world_predictions_by_partner_pre)
    targets = np.asarray(trajectory.world_targets)
    conditional = cells[rows, partner]
    marginal = np.einsum("ta,tad->td", probabilities, cells)

    expected_nll = -np.log(np.maximum(probabilities[rows, partner], np.float32(1.0e-6)))
    expected_brier = np.sum(np.square(probabilities - one_hot), axis=1)
    expected_conditional = np.mean(np.square(conditional - targets), axis=1)
    expected_marginal = np.mean(np.square(marginal - targets), axis=1)
    np.testing.assert_allclose(trajectory.behavior_nll, expected_nll, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        trajectory.behavior_brier,
        expected_brier,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        trajectory.conditional_world_mse,
        expected_conditional,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        trajectory.marginal_world_mse,
        expected_marginal,
        rtol=1e-6,
        atol=1e-7,
    )

    # These are construction checks, not success claims: the partner loss is
    # not an alias of a world loss, and conditional scoring is not the same as
    # the pre-action belief mixture.
    assert not np.array_equal(
        np.asarray(trajectory.behavior_nll),
        np.asarray(trajectory.conditional_world_mse),
    )
    assert bool(
        np.any(
            np.asarray(trajectory.conditional_world_mse)
            != np.asarray(trajectory.marginal_world_mse)
        )
    )


def test_recurrence_endpoints_are_arithmetic_not_a_verdict(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    report = small_report
    assert tuple(summary.name for summary in report.phase_summaries) == PHASE_NAMES
    assert tuple((summary.start, summary.stop) for summary in report.phase_summaries) == (
        (0, 8),
        (8, 16),
        (16, 24),
    )
    recurrence = report.recurrence
    for metric_field in dataclasses.fields(FactorizationMetrics):
        name = metric_field.name
        initial = getattr(recurrence.initial_a_reference, name)
        entry = getattr(recurrence.recurrence_entry, name)
        late = getattr(recurrence.recurrence_late, name)
        assert getattr(recurrence.entry_forgetting, name) == pytest.approx(entry - initial)
        assert getattr(recurrence.within_recurrence_recovery, name) == pytest.approx(
            entry - late
        )
        assert getattr(recurrence.residual_forgetting, name) == pytest.approx(late - initial)


def _assert_tree_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert cast(object, left_tree) == cast(object, right_tree)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            left_leaf.dtype,
            jax.dtypes.prng_key,
        ):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_same_config_replays_bit_exactly(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    replay = run_world_agent_factorization_recurrence_development(small_report.config)
    assert replay.source.input_sha256 == small_report.source.input_sha256
    assert replay.trajectory_sha256 == small_report.trajectory_sha256
    _assert_tree_exact(replay.initial_behavior_state, small_report.initial_behavior_state)
    _assert_tree_exact(replay.final_behavior_state, small_report.final_behavior_state)
    _assert_tree_exact(replay.initial_world_state, small_report.initial_world_state)
    _assert_tree_exact(replay.final_world_state, small_report.final_world_state)
    for field in dataclasses.fields(small_report.trajectory):
        np.testing.assert_array_equal(
            np.asarray(getattr(replay.trajectory, field.name)),
            np.asarray(getattr(small_report.trajectory, field.name)),
        )


@pytest.mark.parametrize(
    ("field", "index", "expected_error"),
    [
        (
            "behavior_probabilities_pre",
            (0, 0),
            "behavior probabilities are non-finite",
        ),
        ("behavior_brier", (0,), "behavior Brier is non-finite"),
        (
            "world_predictions_by_partner_pre",
            (0, 0, 0),
            "world conditional cells are non-finite",
        ),
        ("world_targets", (0, 0), "world targets are non-finite"),
        ("marginal_world_mse", (0,), "marginal world MSE is non-finite"),
    ],
)
def test_nonfinite_report_channels_fail_explicitly(
    small_report: WorldAgentFactorizationRecurrenceReport,
    field: str,
    index: tuple[int, ...],
    expected_error: str,
) -> None:
    values = getattr(small_report.trajectory, field)
    tampered_values = values.at[index].set(jnp.float32(jnp.nan))
    tampered_trajectory = dataclasses.replace(
        small_report.trajectory,
        **{field: tampered_values},
    )
    errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(small_report, trajectory=tampered_trajectory)
    )
    assert expected_error in errors


def test_resealed_finite_trajectory_and_final_state_forgeries_fail_replay_binding(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    selected = int(np.asarray(small_report.trajectory.selected_joint_action_index)[0])
    weight_mask = small_report.trajectory.world_weight_row_change_mask
    forged_weight_mask = weight_mask.at[0, selected].set(~weight_mask[0, selected])
    forged_trajectory = dataclasses.replace(
        small_report.trajectory,
        world_weight_row_change_mask=forged_weight_mask,
    )
    forged_digest = factorization_module._array_manifest_sha256(
        factorization_module._trajectory_arrays(forged_trajectory),
        prefix=f"{DEVELOPMENT_SCHEMA}:{small_report.source.input_sha256}",
    )
    trajectory_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            trajectory=forged_trajectory,
            trajectory_sha256=forged_digest,
        )
    )
    assert (
        "trajectory world_weight_row_change_mask differs from deterministic execution"
        in trajectory_errors
    )

    forged_final_behavior = small_report.final_behavior_state.replace(  # type: ignore[attr-defined]
        weights=small_report.final_behavior_state.weights.at[0, 0].add(jnp.float32(0.125)),
    )
    state_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            final_behavior_state=forged_final_behavior,
        )
    )
    assert "final behavior state differs from deterministic execution" in state_errors


def test_resealed_signed_zero_trajectory_forgery_fails_byte_exact_replay_binding(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    targets = small_report.trajectory.world_targets
    zero_rows = np.argwhere(np.asarray(targets) == np.float32(0.0))
    assert zero_rows.size > 0
    row, column = (int(value) for value in zero_rows[0])
    assert np.asarray(targets)[row, column].tobytes() == np.float32(0.0).tobytes()
    forged_targets = targets.at[row, column].set(jnp.float32(-0.0))
    forged_trajectory = dataclasses.replace(
        small_report.trajectory,
        world_targets=forged_targets,
    )
    forged_digest = factorization_module._array_manifest_sha256(
        factorization_module._trajectory_arrays(forged_trajectory),
        prefix=f"{DEVELOPMENT_SCHEMA}:{small_report.source.input_sha256}",
    )
    errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(
            small_report,
            trajectory=forged_trajectory,
            trajectory_sha256=forged_digest,
        )
    )
    assert "trajectory world_targets differs from deterministic execution" in errors


def test_source_and_resource_tampering_fail_local_validation(
    small_report: WorldAgentFactorizationRecurrenceReport,
) -> None:
    tampered_source = dataclasses.replace(
        small_report.source,
        rewards=small_report.source.rewards.at[0].add(jnp.float32(0.25)),
    )
    source_errors = validate_world_agent_factorization_source(tampered_source)
    assert "source input digest does not match its arrays" in source_errors
    assert "rewards does not reconstruct bit-exactly" in source_errors

    tampered_resource = dataclasses.replace(
        small_report.resource,
        logical_preupdate_work_nbytes_per_step=(
            small_report.resource.logical_preupdate_work_nbytes_per_step + 4
        ),
    )
    tampered_report = dataclasses.replace(small_report, resource=tampered_resource)
    report_errors = validate_world_agent_factorization_recurrence_report(tampered_report)
    assert "fixed logical work or output byte accounting differs" in report_errors

    wrong_shape_trajectory = dataclasses.replace(
        small_report.trajectory,
        behavior_brier=small_report.trajectory.behavior_brier[:-1],
    )
    shape_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(small_report, trajectory=wrong_shape_trajectory)
    )
    assert "trajectory behavior_brier shape differs" in shape_errors

    wrong_dtype_trajectory = dataclasses.replace(
        small_report.trajectory,
        behavior_brier=small_report.trajectory.behavior_brier.astype(jnp.int32),
    )
    dtype_errors = validate_world_agent_factorization_recurrence_report(
        dataclasses.replace(small_report, trajectory=wrong_dtype_trajectory)
    )
    assert "trajectory behavior_brier dtype differs" in dtype_errors
