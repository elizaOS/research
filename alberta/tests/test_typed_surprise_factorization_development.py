"""Contracts for the development-only typed-surprise factorization probe."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import math
from collections.abc import Generator
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation import (
    typed_surprise_factorization_development as typed_module,
)
from alberta_framework.evaluation.typed_surprise_factorization_development import (
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    BRANCH_NAMES,
    DEVELOPMENT_KEY_FROZEN,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SCHEMA,
    DISTRACTOR_OBSERVATION_INDEX,
    EVIDENCE_CLAIMED,
    METRIC_NAMES,
    OUTPUT_WRITES_ALLOWED,
    PHYSICAL_TARGET_INDICES,
    RESETS_EXPOSED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    TASK_IDENTIFIERS_EXPOSED,
    THRESHOLDS_FROZEN,
    TypedSurpriseFactorizationConfig,
    TypedSurpriseFactorizationReport,
    TypedSurpriseFeedback,
    TypedSurprisePreObservation,
    TypedSurpriseSource,
    TypedSurpriseSummary,
    TypedSurpriseTrajectory,
    build_typed_surprise_source,
    run_typed_surprise_factorization_development,
    validate_typed_surprise_factorization_report,
    validate_typed_surprise_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module", autouse=True)
def _release_compilation_cache() -> Generator[None, None, None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()


@pytest.fixture(scope="module")
def small_config() -> TypedSurpriseFactorizationConfig:
    return TypedSurpriseFactorizationConfig(
        prefix_steps=8,
        continuation_steps=4,
        behavior_step_size=0.2,
        world_step_size=0.2,
        world_initialization_scale=0.01,
        development_key=17,
    )


@pytest.fixture(scope="module")
def small_report(
    small_config: TypedSurpriseFactorizationConfig,
) -> TypedSurpriseFactorizationReport:
    return run_typed_surprise_factorization_development(small_config)


def _array_bytes_equal(left: object, right: object) -> bool:
    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _trees_exact(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if hasattr(left_leaf, "dtype") and hasattr(right_leaf, "dtype"):
            left_value = left_leaf
            right_value = right_leaf
            if jax.dtypes.issubdtype(left_value.dtype, jax.dtypes.prng_key):  # type: ignore[attr-defined]
                left_value = jax.random.key_data(left_value)
                right_value = jax.random.key_data(right_value)
            if not _array_bytes_equal(left_value, right_value):
                return False
        elif type(left_leaf) is not type(right_leaf) or left_leaf != right_leaf:
            return False
    return True


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("prefix_steps", True, TypeError),
        ("prefix_steps", 3, ValueError),
        ("prefix_steps", 6, ValueError),
        ("continuation_steps", False, TypeError),
        ("continuation_steps", 0, ValueError),
        ("behavior_step_size", 1, TypeError),
        ("behavior_step_size", np.float32(0.2), TypeError),
        ("behavior_step_size", float("nan"), ValueError),
        ("behavior_step_size", float("inf"), ValueError),
        ("behavior_step_size", 1.0e300, ValueError),
        ("behavior_step_size", 1.0e-300, ValueError),
        ("world_step_size", 1, TypeError),
        ("world_step_size", 1.1, ValueError),
        ("world_initialization_scale", False, TypeError),
        ("world_initialization_scale", 1.0e-300, ValueError),
        ("development_key", -1, ValueError),
        ("development_key", 2**32, ValueError),
    ],
)
def test_config_fails_closed(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        TypedSurpriseFactorizationConfig(**{field: value})  # type: ignore[arg-type]


def test_config_canonically_narrows_all_learning_floats_to_float32() -> None:
    config = TypedSurpriseFactorizationConfig(
        behavior_step_size=0.2,
        world_step_size=0.3,
        world_initialization_scale=0.01,
    )
    assert type(config.behavior_step_size) is float
    assert type(config.world_step_size) is float
    assert type(config.world_initialization_scale) is float
    assert config.behavior_step_size == float(np.float32(0.2))
    assert config.world_step_size == float(np.float32(0.3))
    assert config.world_initialization_scale == float(np.float32(0.01))


def test_source_has_one_common_prefix_and_exactly_matched_component_branches(
    small_config: TypedSurpriseFactorizationConfig,
) -> None:
    source = build_typed_surprise_source(small_config)
    replay = build_typed_surprise_source(small_config)
    assert validate_typed_surprise_source(source) == ()
    assert source.input_sha256 == replay.input_sha256
    assert source.generator_contract_sha256 == replay.generator_contract_sha256
    assert tuple(branch.name for branch in source.branches) == BRANCH_NAMES

    control, partner, physical, noisy = (
        branch.segment for branch in source.branches
    )
    for candidate in (partner, physical, noisy):
        np.testing.assert_array_equal(
            np.asarray(candidate.behavior_features),
            np.asarray(control.behavior_features),
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.world_representations),
            np.asarray(control.world_representations),
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.focal_actions),
            np.asarray(control.focal_actions),
        )
    expected_world_context = np.zeros(
        (small_config.continuation_steps, 2), dtype=np.float32
    )
    expected_world_context[:, 0] = np.float32(1.0)
    np.testing.assert_array_equal(
        np.asarray(control.world_representations), expected_world_context
    )
    assert not np.array_equal(
        np.asarray(control.behavior_features),
        np.asarray(control.world_representations),
    )

    np.testing.assert_array_equal(
        np.asarray(partner.partner_actions),
        1 - np.asarray(control.partner_actions),
    )
    for candidate in (physical, noisy):
        np.testing.assert_array_equal(
            np.asarray(candidate.partner_actions),
            np.asarray(control.partner_actions),
        )

    np.testing.assert_array_equal(
        np.asarray(noisy.next_physical_observations),
        np.asarray(control.next_physical_observations),
    )
    np.testing.assert_array_equal(np.asarray(noisy.rewards), np.asarray(control.rewards))
    np.testing.assert_array_equal(
        np.asarray(noisy.discounts), np.asarray(control.discounts)
    )
    assert not np.array_equal(
        np.asarray(noisy.next_distractors), np.asarray(control.next_distractors)
    )

    np.testing.assert_array_equal(
        np.asarray(physical.next_distractors),
        np.asarray(control.next_distractors),
    )
    assert not np.array_equal(
        np.asarray(physical.next_physical_observations),
        np.asarray(control.next_physical_observations),
    )
    assert not np.array_equal(np.asarray(physical.rewards), np.asarray(control.rewards))
    assert not np.array_equal(
        np.asarray(physical.discounts), np.asarray(control.discounts)
    )

    focal = np.asarray(partner.focal_actions)
    partner_actions = np.asarray(partner.partner_actions)
    expected_first = (
        np.float32(0.375) * (np.float32(2.0) * focal - np.float32(1.0))
        - np.float32(0.125)
        * (np.float32(2.0) * partner_actions - np.float32(1.0))
    ).astype(np.float32)
    expected_second = (
        np.float32(0.50) * (np.float32(2.0) * focal - np.float32(1.0))
        + np.float32(0.25)
        * (np.float32(2.0) * partner_actions - np.float32(1.0))
    ).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(partner.next_physical_observations),
        np.stack((expected_first, expected_second), axis=1),
    )
    np.testing.assert_array_equal(
        np.asarray(partner.rewards),
        (focal == partner_actions).astype(np.float32),
    )


def test_learner_interfaces_hide_ids_and_prediction_precedes_reveal() -> None:
    assert {field.name for field in dataclasses.fields(TypedSurprisePreObservation)} == {
        "behavior_features",
        "world_representation",
        "focal_action",
    }
    assert {field.name for field in dataclasses.fields(TypedSurpriseFeedback)} == {
        "partner_action",
        "next_observation",
        "reward",
        "discount",
    }
    learner_fields = {
        field.name for field in dataclasses.fields(TypedSurprisePreObservation)
    } | {field.name for field in dataclasses.fields(TypedSurpriseFeedback)}
    assert not learner_fields & {
        "branch",
        "branch_id",
        "phase",
        "phase_id",
        "regime",
        "task",
        "task_id",
        "reset",
    }
    assert tuple(inspect.signature(typed_module._run_segment).parameters) == (
        "behavior",
        "behavior_state",
        "world",
        "world_state",
        "segment",
    )
    runner_source = inspect.getsource(typed_module._run_segment)
    assert runner_source.index("_pre_action_predictions(") < runner_source.index(
        "_feedback_at("
    )


def test_report_is_nonpromoting_exactly_bound_and_resource_accounted(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    report = small_report
    assert validate_typed_surprise_factorization_report(report) == ()
    assert DEVELOPMENT_SCHEMA == "alberta.typed-surprise-factorization.development.v1"
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
    assert report.learner_reset_count == 0
    assert report.descriptive_claims_only is True
    assert report.summary.metric_names == METRIC_NAMES
    assert report.summary.common_prefix_mean.shape == (5,)
    assert report.summary.branch_means.shape == (4, 5)
    assert report.summary.branch_minus_control.shape == (4, 5)
    np.testing.assert_array_equal(
        np.asarray(report.summary.branch_minus_control),
        np.asarray(report.summary.branch_means)
        - np.asarray(report.summary.branch_means)[0],
    )
    assert not {
        "threshold",
        "winner",
        "verdict",
        "accepted",
        "promoted",
    } & {field.name for field in dataclasses.fields(TypedSurpriseFactorizationReport)}

    resource = report.resource
    assert resource.fixed_state_nbytes is True
    assert resource.logical_preupdate_float32_scalars_per_step == 22
    assert resource.logical_preupdate_work_nbytes_per_step == 88
    assert resource.partner_world_cells_evaluated_per_step == 2
    assert resource.behavior_updates_per_step == 1
    assert resource.world_updates_per_step == 1
    assert resource.replay_capacity == 0
    assert resource.passes_over_prefix == 1
    assert resource.passes_over_each_continuation == 1
    assert all(
        value == resource.common_prefix_total_state_nbytes
        for value in resource.branch_initial_total_state_nbytes
        + resource.branch_final_total_state_nbytes
    )
    expected_prefix_bytes = sum(
        np.asarray(getattr(report.common_prefix_trajectory, field.name)).nbytes
        for field in dataclasses.fields(TypedSurpriseTrajectory)
    )
    expected_branch_bytes = tuple(
        sum(
            np.asarray(getattr(branch.trajectory, field.name)).nbytes
            for field in dataclasses.fields(TypedSurpriseTrajectory)
        )
        for branch in report.branches
    )
    assert resource.prefix_trajectory_nbytes == expected_prefix_bytes
    assert resource.branch_trajectory_nbytes == expected_branch_bytes
    assert resource.total_trajectory_nbytes == expected_prefix_bytes + sum(
        expected_branch_bytes
    )
    for digest in (
        report.source.input_sha256,
        report.implementation_source_sha256,
        report.common_prefix_state_sha256,
        report.branch_state_sha256,
        report.trajectory_sha256,
    ):
        assert type(digest) is str
        assert len(digest) == 64
        int(digest, 16)

    assert len(report.branch_audits) == len(BRANCH_NAMES)
    for audit in report.branch_audits:
        assert audit.transitions == report.config.continuation_steps
        assert (
            audit.behavior_pre_action_prediction_api_calls
            == report.config.continuation_steps
        )
        assert (
            audit.world_conditional_prediction_api_calls
            == 2 * report.config.continuation_steps
        )
        assert audit.behavior_update_api_calls == report.config.continuation_steps
        assert audit.world_update_api_calls == report.config.continuation_steps
        assert audit.behavior_rng_draws == 0
        assert audit.world_rng_draws == 0
        assert audit.copied_learner_states == 2
        assert (
            audit.logical_preupdate_work_nbytes
            == report.config.continuation_steps
            * resource.logical_preupdate_work_nbytes_per_step
        )
        assert len(audit.initial_behavior_rng_key_bytes_hex) == 16
        assert (
            audit.initial_behavior_rng_key_bytes_hex
            == audit.final_behavior_rng_key_bytes_hex
        )
    matched_receipts = tuple(
        dataclasses.astuple(audit)[1:] for audit in report.branch_audits
    )
    assert all(receipt == matched_receipts[0] for receipt in matched_receipts)


def test_physical_metrics_exclude_distractor_and_distractor_error_is_separate(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    assert PHYSICAL_TARGET_INDICES == (0, 1, 3, 4)
    assert DISTRACTOR_OBSERVATION_INDEX == 2
    assert DISTRACTOR_OBSERVATION_INDEX not in PHYSICAL_TARGET_INDICES
    physical_indices = np.asarray(PHYSICAL_TARGET_INDICES, dtype=np.int32)
    for branch_source, branch_run in zip(
        small_report.source.branches,
        small_report.branches,
        strict=True,
    ):
        trajectory = branch_run.trajectory
        partner_actions = np.asarray(branch_source.segment.partner_actions)
        rows = np.arange(small_report.config.continuation_steps)
        cells = np.asarray(trajectory.world_predictions_by_partner_pre)
        conditional = cells[rows, partner_actions]
        marginal = np.asarray(trajectory.marginal_world_predictions_pre)
        targets = np.asarray(trajectory.world_targets)
        conditional_physical_mse = np.mean(
            np.square(
                conditional[:, physical_indices] - targets[:, physical_indices]
            ),
            axis=1,
        ).astype(np.float32)
        marginal_physical_mse = np.mean(
            np.square(marginal[:, physical_indices] - targets[:, physical_indices]),
            axis=1,
        ).astype(np.float32)
        distractor_error = np.square(
            conditional[:, DISTRACTOR_OBSERVATION_INDEX]
            - targets[:, DISTRACTOR_OBSERVATION_INDEX]
        ).astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(trajectory.conditional_physical_world_mse),
            conditional_physical_mse,
        )
        np.testing.assert_array_equal(
            np.asarray(trajectory.marginal_physical_world_mse),
            marginal_physical_mse,
        )
        np.testing.assert_array_equal(
            np.asarray(trajectory.distractor_squared_error),
            distractor_error,
        )
        np.testing.assert_array_equal(
            targets[:, :2],
            np.asarray(branch_source.segment.next_physical_observations),
        )
        np.testing.assert_array_equal(
            targets[:, DISTRACTOR_OBSERVATION_INDEX],
            np.asarray(branch_source.segment.next_distractors),
        )
        np.testing.assert_array_equal(
            targets[:, 3], np.asarray(branch_source.segment.rewards)
        )
        np.testing.assert_array_equal(
            targets[:, 4], np.asarray(branch_source.segment.discounts)
        )


def test_each_branch_starts_from_an_exact_copy_and_clocks_continue_without_reset(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    report = small_report
    prefix_pre = np.stack(
        (
            np.zeros(report.config.prefix_steps, dtype=np.uint32),
            np.arange(report.config.prefix_steps, dtype=np.uint32),
        ),
        axis=1,
    )
    prefix_post = prefix_pre.copy()
    prefix_post[:, 1] += np.uint32(1)
    for name in ("behavior_pre_words", "world_pre_words"):
        np.testing.assert_array_equal(
            np.asarray(getattr(report.common_prefix_trajectory, name)), prefix_pre
        )
    for name in ("behavior_post_words", "world_post_words"):
        np.testing.assert_array_equal(
            np.asarray(getattr(report.common_prefix_trajectory, name)), prefix_post
        )

    branch_pre = np.stack(
        (
            np.zeros(report.config.continuation_steps, dtype=np.uint32),
            np.arange(
                report.config.prefix_steps,
                report.config.prefix_steps + report.config.continuation_steps,
                dtype=np.uint32,
            ),
        ),
        axis=1,
    )
    branch_post = branch_pre.copy()
    branch_post[:, 1] += np.uint32(1)
    for branch in report.branches:
        assert _trees_exact(
            branch.initial_behavior_state, report.common_prefix_behavior_state
        )
        assert _trees_exact(branch.initial_world_state, report.common_prefix_world_state)
        for name in ("behavior_pre_words", "world_pre_words"):
            np.testing.assert_array_equal(
                np.asarray(getattr(branch.trajectory, name)), branch_pre
            )
        for name in ("behavior_post_words", "world_post_words"):
            np.testing.assert_array_equal(
                np.asarray(getattr(branch.trajectory, name)), branch_post
            )
        for name in (
            "behavior_update_applied",
            "world_update_applied",
            "behavior_prediction_bound",
            "world_prediction_bound",
        ):
            assert bool(np.all(np.asarray(getattr(branch.trajectory, name))))


def test_source_validator_rejects_type_order_nonfinite_and_signed_zero_tamper(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    source = small_report.source
    assert validate_typed_surprise_source(cast(TypedSurpriseSource, object())) == (
        "source type differs",
    )
    wrong_config = dataclasses.replace(source, config=cast(Any, object()))
    assert validate_typed_surprise_source(wrong_config) == ("source config type differs",)
    wrong_prefix = dataclasses.replace(source, prefix=cast(Any, object()))
    assert "prefix segment type differs" in validate_typed_surprise_source(wrong_prefix)

    reordered = dataclasses.replace(
        source,
        branches=(
            source.branches[1],
            source.branches[0],
            source.branches[2],
            source.branches[3],
        ),
    )
    assert "source branch names or order differ" in validate_typed_surprise_source(
        reordered
    )

    control = source.branches[0]
    rewards = np.asarray(control.segment.rewards).copy()
    zero_index = int(np.flatnonzero(rewards == np.float32(0.0))[0])
    rewards[zero_index] = np.float32(-0.0)
    assert np.signbit(rewards[zero_index])
    signed_zero_segment = dataclasses.replace(
        control.segment,
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
    )
    signed_zero_source = dataclasses.replace(
        source,
        branches=(
            dataclasses.replace(control, segment=signed_zero_segment),
            source.branches[1],
            source.branches[2],
            source.branches[3],
        ),
    )
    signed_errors = validate_typed_surprise_source(signed_zero_source)
    assert "source input digest does not match its arrays" in signed_errors
    assert any("rewards does not reconstruct bit-exactly" in error for error in signed_errors)

    distractors = np.asarray(control.segment.next_distractors).copy()
    distractors[0] = np.float32(np.nan)
    nonfinite_segment = dataclasses.replace(
        control.segment,
        next_distractors=jnp.asarray(distractors),
    )
    nonfinite_source = dataclasses.replace(
        source,
        branches=(
            dataclasses.replace(control, segment=nonfinite_segment),
            source.branches[1],
            source.branches[2],
            source.branches[3],
        ),
    )
    assert any(
        "contains non-finite values" in error
        for error in validate_typed_surprise_source(nonfinite_source)
    )


def test_report_validator_guards_nested_types_before_dereference(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    assert validate_typed_surprise_factorization_report(
        cast(TypedSurpriseFactorizationReport, object())
    ) == ("report type differs",)
    cases = (
        (
            dataclasses.replace(small_report, config=cast(Any, object())),
            "report config type differs",
        ),
        (
            dataclasses.replace(small_report, source=cast(Any, object())),
            "report source type differs",
        ),
        (
            dataclasses.replace(
                small_report,
                common_prefix_trajectory=cast(Any, object()),
            ),
            "common-prefix trajectory type differs",
        ),
        (
            dataclasses.replace(small_report, branches=cast(Any, object())),
            "report branches type or cardinality differs",
        ),
        (
            dataclasses.replace(small_report, summary=cast(Any, object())),
            "report summary type differs",
        ),
        (
            dataclasses.replace(small_report, resource=cast(Any, object())),
            "report resource type differs",
        ),
        (
            dataclasses.replace(small_report, branch_audits=cast(Any, object())),
            "report branch audits type or cardinality differs",
        ),
    )
    for malformed, expected_error in cases:
        assert expected_error in validate_typed_surprise_factorization_report(malformed)


def test_report_validator_rejects_raw_summary_state_resource_hash_and_order_tamper(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    branch = small_report.branches[0]
    nll = np.asarray(branch.trajectory.behavior_nll).copy()
    nll[0] = np.nextafter(nll[0], np.float32(math.inf), dtype=np.float32)
    changed_trajectory = dataclasses.replace(
        branch.trajectory,
        behavior_nll=jnp.asarray(nll, dtype=jnp.float32),
    )
    changed_branch = dataclasses.replace(branch, trajectory=changed_trajectory)
    raw_tamper = dataclasses.replace(
        small_report,
        branches=(
            changed_branch,
            small_report.branches[1],
            small_report.branches[2],
            small_report.branches[3],
        ),
    )
    raw_errors = validate_typed_surprise_factorization_report(raw_tamper)
    assert any("behavior_nll differs from deterministic execution" in error for error in raw_errors)
    assert "trajectory digest differs" in raw_errors

    deltas = np.asarray(small_report.summary.branch_minus_control).copy()
    assert deltas[0, 0] == np.float32(0.0)
    deltas[0, 0] = np.float32(-0.0)
    assert np.signbit(deltas[0, 0])
    signed_summary = dataclasses.replace(
        small_report.summary,
        branch_minus_control=jnp.asarray(deltas, dtype=jnp.float32),
    )
    summary_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, summary=signed_summary)
    )
    assert "summary branch_minus_control differs from deterministic execution" in summary_errors

    weights = small_report.common_prefix_behavior_state.weights.at[0, 0].set(jnp.nan)
    bad_state = small_report.common_prefix_behavior_state.replace(weights=weights)  # type: ignore[attr-defined]
    state_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, common_prefix_behavior_state=bad_state)
    )
    assert "common-prefix behavior state differs from deterministic execution" in state_errors
    assert "common-prefix behavior state contains non-finite values" in state_errors
    assert "common-prefix state digest differs" in state_errors

    resource_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(
            small_report,
            resource=dataclasses.replace(
                small_report.resource,
                replay_capacity=np.int64(0),  # type: ignore[arg-type]
            ),
        )
    )
    assert "resource accounting differs" in resource_errors

    hash_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, trajectory_sha256="0" * 64)
    )
    assert "trajectory digest differs" in hash_errors

    implementation_hash_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, implementation_source_sha256="0" * 64)
    )
    assert "implementation source digest differs" in implementation_hash_errors

    audit = small_report.branch_audits[0]
    changed_audit = dataclasses.replace(
        audit,
        behavior_rng_draws=1,
    )
    audit_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(
            small_report,
            branch_audits=(
                changed_audit,
                small_report.branch_audits[1],
                small_report.branch_audits[2],
                small_report.branch_audits[3],
            ),
        )
    )
    assert "matched branch call, work, or RNG audit differs" in audit_errors

    reordered = dataclasses.replace(
        small_report,
        branches=(
            small_report.branches[1],
            small_report.branches[0],
            small_report.branches[2],
            small_report.branches[3],
        ),
    )
    assert "report branch names or order differ" in (
        validate_typed_surprise_factorization_report(reordered)
    )


def test_report_validator_rejects_nonfinite_trajectory_and_limitations_tamper(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    branch = small_report.branches[3]
    distractor_error = np.asarray(branch.trajectory.distractor_squared_error).copy()
    distractor_error[0] = np.float32(np.inf)
    trajectory = dataclasses.replace(
        branch.trajectory,
        distractor_squared_error=jnp.asarray(distractor_error),
    )
    malformed_branch = dataclasses.replace(branch, trajectory=trajectory)
    report = dataclasses.replace(
        small_report,
        branches=(
            small_report.branches[0],
            small_report.branches[1],
            small_report.branches[2],
            malformed_branch,
        ),
    )
    errors = validate_typed_surprise_factorization_report(report)
    assert any("contains non-finite values" in error for error in errors)

    limitation_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(
            small_report,
            limitations=small_report.limitations + ("tampered",),
        )
    )
    assert "report limitations differ" in limitation_errors


def test_public_builders_reject_config_substitutes() -> None:
    with pytest.raises(TypeError, match="TypedSurpriseFactorizationConfig"):
        build_typed_surprise_source(cast(Any, object()))
    with pytest.raises(TypeError, match="TypedSurpriseFactorizationConfig"):
        run_typed_surprise_factorization_development(cast(Any, object()))


def test_validators_reconstruct_config_and_reject_post_init_mutation(
    small_report: TypedSurpriseFactorizationReport,
) -> None:
    bool_alias = TypedSurpriseFactorizationConfig(prefix_steps=8, continuation_steps=4)
    object.__setattr__(bool_alias, "prefix_steps", True)
    bool_source = dataclasses.replace(small_report.source, config=bool_alias)
    assert validate_typed_surprise_source(bool_source) == (
        "source config fields are not canonical",
    )
    bool_report_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, config=bool_alias)
    )
    assert "report config fields are not canonical" in bool_report_errors

    nonfinite = TypedSurpriseFactorizationConfig(prefix_steps=8, continuation_steps=4)
    object.__setattr__(nonfinite, "world_step_size", float("nan"))
    nonfinite_source = dataclasses.replace(small_report.source, config=nonfinite)
    assert validate_typed_surprise_source(nonfinite_source) == (
        "source config fields are not canonical",
    )
    with pytest.raises(ValueError):
        build_typed_surprise_source(nonfinite)

    class StringAlias(str):
        pass

    alias_limitations = (
        StringAlias(small_report.limitations[0]),
        *small_report.limitations[1:],
    )
    assert alias_limitations == small_report.limitations
    alias_errors = validate_typed_surprise_factorization_report(
        dataclasses.replace(small_report, limitations=alias_limitations)
    )
    assert "report limitations differ" in alias_errors


def test_raw_schema_contains_only_exact_trajectory_arrays() -> None:
    assert all(
        field.name
        in {
            "behavior_probabilities_pre",
            "behavior_probabilities_update",
            "world_predictions_by_partner_pre",
            "conditional_world_predictions_pre",
            "conditional_world_predictions_update",
            "marginal_world_predictions_pre",
            "world_targets",
            "behavior_nll",
            "behavior_brier",
            "conditional_physical_world_mse",
            "marginal_physical_world_mse",
            "distractor_squared_error",
            "behavior_pre_words",
            "behavior_post_words",
            "world_pre_words",
            "world_post_words",
            "behavior_update_applied",
            "world_update_applied",
            "behavior_prediction_bound",
            "world_prediction_bound",
            "selected_joint_action_index",
            "world_weight_row_change_mask",
            "world_bias_row_change_mask",
        }
        for field in dataclasses.fields(TypedSurpriseTrajectory)
    )
    assert {field.name for field in dataclasses.fields(TypedSurpriseSummary)} == {
        "metric_names",
        "common_prefix_mean",
        "branch_means",
        "branch_minus_control",
    }
