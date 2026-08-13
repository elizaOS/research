"""Focused contracts for authority-free future-utility cadence accounting."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation import compositional_future_utility_panel_core as core

pytestmark = pytest.mark.unit


def test_partition_conserves_due_off_and_all_step_counts() -> None:
    mask = np.asarray((True, False, True, True, False, True), dtype=np.bool_)
    due = np.asarray((False, True, True, False, False, True), dtype=np.bool_)

    partition = core.partition_on_opportunity(mask, due)

    assert partition == core.OpportunityPartition(
        all_step_count=4,
        due_opportunity_count=2,
        off_opportunity_count=2,
    )
    assert partition.all_step_count == (
        partition.due_opportunity_count + partition.off_opportunity_count
    )


def test_exact_consumed_v2_cadence_regression_is_partitioned_not_rejected() -> None:
    margin_passed = np.zeros((32,), dtype=np.bool_)
    margin_passed[0] = True
    margin_passed[31] = True
    due = np.zeros((32,), dtype=np.bool_)
    due[31] = True

    assert core.partition_on_opportunity(margin_passed, due) == (
        core.OpportunityPartition(
            all_step_count=2,
            due_opportunity_count=1,
            off_opportunity_count=1,
        )
    )


def test_candidate_promote_and_refresh_style_masks_use_the_leading_step_axis() -> None:
    due = jnp.asarray((False, True, False, True), dtype=jnp.bool_)
    candidate_pairs = jnp.zeros((4, 2, 3), dtype=jnp.bool_).at[0, 0, 0].set(True)
    candidate_pairs = candidate_pairs.at[1, 0, 1].set(True).at[3, 1, 2].set(True)
    should_promote = jnp.asarray((True, True, False, False), dtype=jnp.bool_)
    should_refresh = jnp.asarray((False, True, True, True), dtype=jnp.bool_)

    assert core.partition_on_opportunity(candidate_pairs, due) == (
        core.OpportunityPartition(3, 2, 1)
    )
    assert core.partition_on_opportunity(should_promote, due) == (
        core.OpportunityPartition(2, 1, 1)
    )
    assert core.partition_on_opportunity(should_refresh, due) == (
        core.OpportunityPartition(3, 2, 1)
    )


def test_declared_mutations_are_returned_read_only_when_cadence_is_valid() -> None:
    due = np.asarray((False, True, False, True), dtype=np.bool_)
    promotion = np.asarray((False, True, False, False), dtype=np.bool_)
    root_changes = np.zeros((4, 3), dtype=np.bool_)
    root_changes[3, 1:] = True
    refreshes = np.zeros((4, 2), dtype=np.bool_)
    refreshes[1, 0] = True

    partitions = core.validate_mutation_masks_on_opportunity(
        {
            "promotion_applied": promotion,
            "root_change_mask": root_changes,
            "candidate_refresh_mask": refreshes,
        },
        due,
    )

    assert partitions == {
        "promotion_applied": core.OpportunityPartition(1, 1, 0),
        "root_change_mask": core.OpportunityPartition(2, 2, 0),
        "candidate_refresh_mask": core.OpportunityPartition(1, 1, 0),
    }
    assert isinstance(partitions, Mapping)
    with pytest.raises(TypeError):
        partitions["extra"] = core.OpportunityPartition(0, 0, 0)  # type: ignore[index]


def test_off_opportunity_mutation_is_rejected_with_its_declared_name() -> None:
    due = np.asarray((False, True, False, True), dtype=np.bool_)
    refreshes = np.asarray((False, True, True, False), dtype=np.bool_)

    with pytest.raises(
        ValueError,
        match="candidate_refresh_mask.*1 true off-opportunity cells",
    ):
        core.validate_mutation_masks_on_opportunity(
            {"candidate_refresh_mask": refreshes},
            due,
        )


@pytest.mark.parametrize(
    ("mask", "due", "error", "match"),
    (
        (True, np.asarray((True,), dtype=np.bool_), ValueError, "leading step"),
        (
            np.asarray((True, False), dtype=np.bool_),
            np.asarray(True, dtype=np.bool_),
            ValueError,
            "one-dimensional",
        ),
        (
            np.asarray((True,), dtype=np.bool_),
            np.asarray((), dtype=np.bool_),
            ValueError,
            "at least one step",
        ),
        (
            np.asarray((True, False), dtype=np.bool_),
            np.asarray((True,), dtype=np.bool_),
            ValueError,
            "same step dimension",
        ),
        (
            np.asarray((1, 0), dtype=np.int32),
            np.asarray((True, False), dtype=np.bool_),
            TypeError,
            "mask must have boolean dtype",
        ),
        (
            np.asarray((True, False), dtype=np.bool_),
            np.asarray((1, 0), dtype=np.int32),
            TypeError,
            "due_mask must have boolean dtype",
        ),
    ),
)
def test_partition_rejects_shape_and_dtype_drift(
    mask: object,
    due: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        core.partition_on_opportunity(mask, due)


def test_mutation_declaration_requires_a_nonempty_named_mapping() -> None:
    due = np.asarray((True,), dtype=np.bool_)
    with pytest.raises(TypeError, match="must be a mapping"):
        core.validate_mutation_masks_on_opportunity([], due)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one mutation"):
        core.validate_mutation_masks_on_opportunity({}, due)
    with pytest.raises(TypeError, match="non-empty exact strings"):
        core.validate_mutation_masks_on_opportunity({"": due}, due)


def test_opportunity_partition_is_frozen_and_validates_its_conservation() -> None:
    partition = core.OpportunityPartition(2, 1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        partition.all_step_count = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="must equal due plus off"):
        core.OpportunityPartition(3, 1, 1)
    with pytest.raises(TypeError, match="exact integer"):
        core.OpportunityPartition(True, 1, 0)


def test_panel_core_has_no_execution_or_scientific_authority() -> None:
    assert core.DEVELOPMENT_ONLY
    assert not core.PANEL_EXECUTION_AUTHORIZED
    assert not core.SOURCE_GENERATION_AUTHORIZED
    assert not core.RUNNER_AVAILABLE
    assert not core.OUTPUT_WRITES_ALLOWED
    assert not core.EVIDENCE_AUTHORIZED
    assert not core.SCIENTIFIC_PROMOTION_ALLOWED
    assert not hasattr(core, "run_compositional_future_utility_calibration_v2_development")


def _fixed_cadence_inputs(
    steps: int = 65,
) -> dict[str, np.ndarray]:
    post_step = np.arange(1, steps + 1, dtype=np.int32)
    due = post_step % 32 == 0
    return {
        "post_step": post_step,
        "decision_update_available": np.ones((steps,), dtype=np.bool_),
        "pre_replacement_phase": (post_step - 1) % 32,
        "post_replacement_phase": post_step % 32,
        "should_try_replace": due.copy(),
        "pinned_due_mask": due,
    }


def _empty_mutation_masks(steps: int = 65) -> dict[str, np.ndarray]:
    return {
        "decision_should_promote": np.zeros((steps,), dtype=np.bool_),
        "decision_should_refresh": np.zeros((steps,), dtype=np.bool_),
        "proposal_formed": np.zeros((steps,), dtype=np.bool_),
        "has_event": np.zeros((steps,), dtype=np.bool_),
        "promotion_applied": np.zeros((steps,), dtype=np.bool_),
        "root_change_applied": np.zeros((steps,), dtype=np.bool_),
        "root_change_mask": np.zeros((steps, 2), dtype=np.bool_),
        "cascade_refill_mask": np.zeros((steps, 2), dtype=np.bool_),
        "active_change_mask": np.zeros((steps, 2), dtype=np.bool_),
        "ordinary_candidate_refresh_mask": np.zeros((steps, 3), dtype=np.bool_),
        "post_promotion_candidate_refresh_mask": np.zeros(
            (steps, 3), dtype=np.bool_
        ),
        "candidate_refresh_mask": np.zeros((steps, 3), dtype=np.bool_),
        "candidate_rebound_mask": np.zeros((steps, 3), dtype=np.bool_),
        "candidate_overdepth_regeneration_mask": np.zeros(
            (steps, 3), dtype=np.bool_
        ),
    }


def test_fixed_cadence_domain_accepts_off_due_diagnostics_and_partitions_65_steps() -> None:
    domain = core.build_fixed_curation_opportunity_domain(
        **_fixed_cadence_inputs(),
        replacement_interval=32,
    )
    margin = np.zeros((65,), dtype=np.bool_)
    margin[[0, 31, 32, 63]] = True
    candidate_margin = np.zeros((65, 2, 3), dtype=np.bool_)
    candidate_margin[0, 0, 0] = True
    candidate_margin[31, 0, 1] = True
    candidate_margin[32, 1, 0] = True
    candidate_margin[63, 1, 2] = True

    audit = core.build_future_utility_cadence_audit(
        domain,
        decision_margin_passed=margin,
        decision_candidate_margin_eligible=candidate_margin,
        mutation_masks=_empty_mutation_masks(),
    )

    assert domain.steps == 65
    assert domain.due_opportunity_count == 2
    assert audit.diagnostic_partitions["decision_margin_passed"] == (
        core.OpportunityPartition(4, 2, 2)
    )
    assert audit.diagnostic_partitions[
        "decision_candidate_margin_eligible"
    ] == core.OpportunityPartition(4, 2, 2)
    assert all(
        partition.off_opportunity_count == 0
        for partition in audit.mutation_partitions.values()
    )


@pytest.mark.parametrize("name", tuple(_empty_mutation_masks()))
def test_fixed_cadence_audit_rejects_each_off_due_mutation(name: str) -> None:
    domain = core.build_fixed_curation_opportunity_domain(
        **_fixed_cadence_inputs(),
        replacement_interval=32,
    )
    mutations = _empty_mutation_masks()
    mutations[name].reshape(65, -1)[0, 0] = True

    with pytest.raises(ValueError, match=name):
        core.build_future_utility_cadence_audit(
            domain,
            decision_margin_passed=np.zeros((65,), dtype=np.bool_),
            decision_candidate_margin_eligible=np.zeros(
                (65, 2, 3), dtype=np.bool_
            ),
            mutation_masks=mutations,
        )


def test_fixed_cadence_audit_requires_the_complete_mutation_surface() -> None:
    domain = core.build_fixed_curation_opportunity_domain(
        **_fixed_cadence_inputs(),
        replacement_interval=32,
    )
    mutations = _empty_mutation_masks()
    mutations.pop("active_change_mask")
    with pytest.raises(ValueError, match="exact cadence mutation set"):
        core.build_future_utility_cadence_audit(
            domain,
            decision_margin_passed=np.zeros((65,), dtype=np.bool_),
            decision_candidate_margin_eligible=np.zeros(
                (65, 2, 3), dtype=np.bool_
            ),
            mutation_masks=mutations,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        (
            "post_step",
            np.arange(0, 65, dtype=np.int32),
            "exact sequence",
        ),
        (
            "pre_replacement_phase",
            np.zeros((65,), dtype=np.int32),
            "pre-replacement phase",
        ),
        (
            "post_replacement_phase",
            np.zeros((65,), dtype=np.int32),
            "post-replacement phase",
        ),
        (
            "pinned_due_mask",
            np.zeros((65,), dtype=np.bool_),
            "pinned due mask",
        ),
        (
            "should_try_replace",
            np.zeros((65,), dtype=np.bool_),
            "should-try-replace",
        ),
    ),
)
def test_fixed_cadence_domain_rejects_clock_phase_and_binding_drift(
    field: str,
    replacement: np.ndarray,
    match: str,
) -> None:
    inputs = _fixed_cadence_inputs()
    inputs[field] = replacement
    with pytest.raises(ValueError, match=match):
        core.build_fixed_curation_opportunity_domain(
            **inputs,
            replacement_interval=32,
        )
