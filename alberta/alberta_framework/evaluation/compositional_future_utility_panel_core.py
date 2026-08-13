"""Authority-free cadence accounting for future-utility panel mechanisms.

This module contains only host-side, protocol-neutral accounting helpers.  It
does not declare a development root, construct a source stream, execute an arm
or panel, write output, select a winner, or authorize evidence or promotion.

Inputs may be NumPy arrays or JAX arrays that support host conversion through
``numpy.asarray``.  Boolean dtype and leading step dimensions are validated
strictly; numeric masks are never silently coerced to booleans.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import numpy.typing as npt

DEVELOPMENT_ONLY: Final = True
PANEL_EXECUTION_AUTHORIZED: Final = False
SOURCE_GENERATION_AUTHORIZED: Final = False
RUNNER_AVAILABLE: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False

REQUIRED_CADENCE_MUTATION_MASK_NAMES: Final = (
    "decision_should_promote",
    "decision_should_refresh",
    "proposal_formed",
    "has_event",
    "promotion_applied",
    "root_change_applied",
    "root_change_mask",
    "cascade_refill_mask",
    "active_change_mask",
    "ordinary_candidate_refresh_mask",
    "post_promotion_candidate_refresh_mask",
    "candidate_refresh_mask",
    "candidate_rebound_mask",
    "candidate_overdepth_regeneration_mask",
)


@dataclasses.dataclass(frozen=True, slots=True)
class OpportunityPartition:
    """Exact true-cell counts partitioned by declared update opportunity.

    Counts apply to every cell in ``mask``.  A one-dimensional mask therefore
    counts events, while a mask with trailing candidate/destination axes counts
    true diagnostic or mutation cells.
    """

    all_step_count: int
    due_opportunity_count: int
    off_opportunity_count: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an exact integer")
            if value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        if self.all_step_count != (
            self.due_opportunity_count + self.off_opportunity_count
        ):
            raise ValueError("all-step count must equal due plus off-opportunity counts")


@dataclasses.dataclass(frozen=True, slots=True)
class FixedCurationOpportunityDomain:
    """Authenticated fixed-cadence step domain, independent of panel identity."""

    steps: int
    replacement_interval: int
    due_mask: tuple[bool, ...]
    due_opportunity_count: int

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive exact integer")
        if type(self.replacement_interval) is not int or self.replacement_interval < 1:
            raise ValueError("replacement_interval must be a positive exact integer")
        if (
            type(self.due_mask) is not tuple
            or len(self.due_mask) != self.steps
            or any(type(value) is not bool for value in self.due_mask)
        ):
            raise ValueError("due_mask must be an exact bool tuple over every step")
        if (
            type(self.due_opportunity_count) is not int
            or self.due_opportunity_count != sum(self.due_mask)
        ):
            raise ValueError("due_opportunity_count does not reconstruct")


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityCadenceAudit:
    """Read-only diagnostic and mutation partitions for one fixed-cadence trace."""

    steps: int
    due_opportunity_count: int
    diagnostic_partitions: Mapping[str, OpportunityPartition]
    mutation_partitions: Mapping[str, OpportunityPartition]


def _host_bool_array(value: object, *, field: str) -> npt.NDArray[np.bool_]:
    """Return one strict host boolean array without numeric coercion."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must support host array conversion") from error
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field} must have boolean dtype")
    return cast(npt.NDArray[np.bool_], array)


def _due_opportunity_mask(value: object) -> npt.NDArray[np.bool_]:
    due = _host_bool_array(value, field="due_mask")
    if due.ndim != 1:
        raise ValueError("due_mask must be one-dimensional")
    if due.shape[0] == 0:
        raise ValueError("due_mask must contain at least one step")
    return due


def _host_integer_vector(value: object, *, field: str) -> npt.NDArray[np.integer]:
    """Return a strict one-dimensional host integer array."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must support host array conversion") from error
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{field} must have integer dtype")
    return cast(npt.NDArray[np.integer], array)


def _host_bool_vector(value: object, *, field: str) -> npt.NDArray[np.bool_]:
    array = _host_bool_array(value, field=field)
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    return array


def build_fixed_curation_opportunity_domain(
    *,
    post_step: object,
    decision_update_available: object,
    pre_replacement_phase: object,
    post_replacement_phase: object,
    should_try_replace: object,
    pinned_due_mask: object,
    replacement_interval: int,
) -> FixedCurationOpportunityDomain:
    """Reconstruct and bind the exact fixed-cadence opportunity domain."""

    if type(replacement_interval) is not int or replacement_interval < 1:
        raise ValueError("replacement_interval must be a positive exact integer")
    steps_array = _host_integer_vector(post_step, field="post_step")
    steps = int(steps_array.shape[0])
    if steps < 1 or not np.array_equal(
        steps_array,
        np.arange(1, steps + 1, dtype=steps_array.dtype),
    ):
        raise ValueError("post_step must be the exact sequence 1..T")

    update_available = _host_bool_vector(
        decision_update_available,
        field="decision_update_available",
    )
    pre_phase = _host_integer_vector(
        pre_replacement_phase,
        field="pre_replacement_phase",
    )
    post_phase = _host_integer_vector(
        post_replacement_phase,
        field="post_replacement_phase",
    )
    should_try = _host_bool_vector(should_try_replace, field="should_try_replace")
    pinned_due = _host_bool_vector(pinned_due_mask, field="pinned_due_mask")
    arrays = (update_available, pre_phase, post_phase, should_try, pinned_due)
    if any(array.shape != (steps,) for array in arrays):
        raise ValueError("curation-domain arrays must share the post_step shape")

    expected_pre = (steps_array - 1) % replacement_interval
    expected_post = steps_array % replacement_interval
    if not np.array_equal(pre_phase, expected_pre):
        raise ValueError("pre-replacement phase does not match the fixed clock")
    if not np.array_equal(post_phase, expected_post):
        raise ValueError("post-replacement phase does not match the fixed clock")
    derived_due = (
        update_available
        & (pre_phase == replacement_interval - 1)
        & (post_phase == 0)
    )
    if not np.array_equal(derived_due, pinned_due):
        raise ValueError("derived and pinned due masks differ")
    if not np.array_equal(should_try, derived_due):
        raise ValueError("should-try-replace mask differs from the due domain")
    due_tuple = tuple(bool(value) for value in derived_due)
    return FixedCurationOpportunityDomain(
        steps=steps,
        replacement_interval=replacement_interval,
        due_mask=due_tuple,
        due_opportunity_count=sum(due_tuple),
    )


def partition_on_opportunity(
    mask: object,
    due_mask: object,
) -> OpportunityPartition:
    """Partition true cells into all-step, due, and off-opportunity counts.

    ``mask`` must be a non-scalar boolean host-convertible array whose leading
    dimension is the step axis.  Any trailing axes are retained as diagnostic
    cells and the one-dimensional ``due_mask`` broadcasts across them.
    """

    observed = _host_bool_array(mask, field="mask")
    due = _due_opportunity_mask(due_mask)
    if observed.ndim == 0:
        raise ValueError("mask must have a leading step dimension")
    if observed.shape[0] != due.shape[0]:
        raise ValueError("mask and due_mask must have the same step dimension")

    broadcast_shape = (due.shape[0],) + (1,) * (observed.ndim - 1)
    due_cells = due.reshape(broadcast_shape)
    all_step_count = int(np.count_nonzero(observed))
    due_count = int(np.count_nonzero(observed & due_cells))
    off_count = int(np.count_nonzero(observed & ~due_cells))
    return OpportunityPartition(
        all_step_count=all_step_count,
        due_opportunity_count=due_count,
        off_opportunity_count=off_count,
    )


def validate_mutation_masks_on_opportunity(
    mutation_masks: Mapping[str, object],
    due_mask: object,
) -> Mapping[str, OpportunityPartition]:
    """Validate that every declared mutation mask is false off-opportunity.

    The returned read-only mapping contains the exact partition for each named
    mutation.  All-step diagnostics should use :func:`partition_on_opportunity`
    directly; this helper is intentionally strict for state-changing masks.
    """

    if not isinstance(mutation_masks, Mapping):
        raise TypeError("mutation_masks must be a mapping")
    if not mutation_masks:
        raise ValueError("mutation_masks must declare at least one mutation")
    due = _due_opportunity_mask(due_mask)
    partitions: dict[str, OpportunityPartition] = {}
    for name, mask in mutation_masks.items():
        if type(name) is not str or not name:
            raise TypeError("mutation mask names must be non-empty exact strings")
        partition = partition_on_opportunity(mask, due)
        if partition.off_opportunity_count != 0:
            raise ValueError(
                f"mutation mask {name!r} has "
                f"{partition.off_opportunity_count} true off-opportunity cells"
            )
        partitions[name] = partition
    return MappingProxyType(partitions)


def build_future_utility_cadence_audit(
    domain: FixedCurationOpportunityDomain,
    *,
    decision_margin_passed: object,
    decision_candidate_margin_eligible: object,
    mutation_masks: Mapping[str, object],
) -> FutureUtilityCadenceAudit:
    """Partition raw diagnostics and reject every off-opportunity mutation."""

    if type(domain) is not FixedCurationOpportunityDomain:
        raise TypeError("domain must be an exact FixedCurationOpportunityDomain")
    if not isinstance(mutation_masks, Mapping):
        raise TypeError("mutation_masks must be a mapping")
    if set(mutation_masks) != set(REQUIRED_CADENCE_MUTATION_MASK_NAMES):
        raise ValueError("mutation_masks must declare the exact cadence mutation set")
    due = np.asarray(domain.due_mask, dtype=np.bool_)
    diagnostic_partitions = MappingProxyType(
        {
            "decision_margin_passed": partition_on_opportunity(
                decision_margin_passed,
                due,
            ),
            "decision_candidate_margin_eligible": partition_on_opportunity(
                decision_candidate_margin_eligible,
                due,
            ),
        }
    )
    mutation_partitions = validate_mutation_masks_on_opportunity(
        {name: mutation_masks[name] for name in REQUIRED_CADENCE_MUTATION_MASK_NAMES},
        due,
    )
    return FutureUtilityCadenceAudit(
        steps=domain.steps,
        due_opportunity_count=domain.due_opportunity_count,
        diagnostic_partitions=diagnostic_partitions,
        mutation_partitions=mutation_partitions,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "REQUIRED_CADENCE_MUTATION_MASK_NAMES",
    "RUNNER_AVAILABLE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_GENERATION_AUTHORIZED",
    "FixedCurationOpportunityDomain",
    "FutureUtilityCadenceAudit",
    "OpportunityPartition",
    "build_fixed_curation_opportunity_domain",
    "build_future_utility_cadence_audit",
    "partition_on_opportunity",
    "validate_mutation_masks_on_opportunity",
]
