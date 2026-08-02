"""Strict host validation for one in-memory noisy-world v6 DEVELOPMENT run.

This module has no execution, evidence, or promotion authority.  It derives no
keys, chooses no seeds, sets no thresholds, writes no files, exposes no CLI,
and creates no artifact.  Structural validation is fail-closed.  Empirical
coverage, lifecycle, and quality observations remain separate DEVELOPMENT
outcomes, so a causally intact negative result is still a valid run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    BASE_FEATURE_DIM,
    CANDIDATE_PAIR_SLOTS,
    IntegratedHiddenPartnerState,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    V6_DIAGNOSTIC_ORDER,
    V6_REPRESENTATION_LOSS_WEIGHTS,
    HiddenPartnerLifecycleWorldV6Control,
    build_v6_diagnostic_controls,
    build_v6_primary_controls,
    validate_v6_control,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_intervention_audit import (
    V6_CONTROL_REQUIRED_WITNESSES,
    V6_FLOAT32_REPLAY_ATOL,
    V6_FLOAT32_REPLAY_RTOL,
    V6_INTERVENTION_AUDIT_ORDER,
    V6_INTERVENTION_WITNESS_ORDER,
    required_v6_intervention_witness_names,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    CRITICAL_CANDIDATE_INDICES,
    CRITICAL_PAIRS,
    CURATION_INTERVAL,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNNER_SCHEMA,
    JOINT_ACTION_ROWS,
    MAX_CADENCE_LEDGER_ENTRIES,
    MAX_SCAN_STEPS,
    RUNNER_STATUS,
    TARGET_HEADS,
    V6_APPLIED_EVENT_ORDER,
    V6_CANDIDATE_FLAG_ORDER,
    V6_CANDIDATE_STREAK_ENDPOINT_ORDER,
    V6_COMPONENT_DELTA_ORDER,
    V6_CONSUMER_MASK_ORDER,
    V6_CONTRACT_AUDIT_ORDER,
    V6_CRITICAL_STAGE_ORDER,
    V6_POLICY_KEY_ENDPOINT_ORDER,
    V6_PROPOSAL_EVENT_ORDER,
    V6_RANDOM_CURATION_FLAG_ORDER,
    V6_RANDOM_CURATION_SELECTED_ORDER,
    V6_ROUTER_COUNT_ORDER,
    V6_ROUTER_FLAG_ORDER,
    V6_ROUTER_MASK_ORDER,
    V6_SOURCE_CLOSURE_PATHS,
    V6_TARGET_HEAD_ORDER,
    V6_WORLD_RNG_KEY_ORDER,
    WINDOW_BIN_COUNT,
    HiddenPartnerLifecycleWorldV6Runner,
    V6ActionTotals,
    V6AuditTotals,
    V6CadenceLedger,
    V6DevelopmentRun,
    V6FilterTotals,
    V6LifecycleChainState,
    V6ResourceRecord,
    V6RngRecord,
    V6RowHeadTotals,
    V6SourceClosureHash,
    V6WindowTotals,
    compute_v6_source_closure_hashes,
    reconstruct_v6_stream_code,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    DEVELOPMENT_ONLY as RUNNER_DEVELOPMENT_ONLY,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    EVIDENCE_AUTHORIZED as RUNNER_EVIDENCE_AUTHORIZED,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    EXECUTION_AUTHORIZED as RUNNER_EXECUTION_AUTHORIZED,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    SCIENTIFIC_PROMOTION_ALLOWED as RUNNER_SCIENTIFIC_PROMOTION_ALLOWED,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import (
    V6RuntimeRecord,
    validate_v6_runtime_record,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    ENTRY_WINDOW_STEPS,
    FINAL_WINDOW_STEPS,
    HiddenPartnerLifecycleWorldV6ScanPlan,
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    require_v6_control_suite_ready,
    validate_hidden_partner_lifecycle_world_v6_scan_plan,
    validate_v6_control_suite_readiness,
)
from alberta_framework.evaluation.hidden_partner_world_filter import (
    HiddenPartnerWorldFilterState,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldOnlineResourceBudget,
    HiddenPartnerWorldOnlineState,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackState,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.validator-development.v1"
)
STRUCTURALLY_VALID_DEVELOPMENT_RUN: Literal["STRUCTURALLY_VALID_DEVELOPMENT_RUN"] = (
    "STRUCTURALLY_VALID_DEVELOPMENT_RUN"
)
STRUCTURALLY_INVALID_DEVELOPMENT_RUN: Literal["STRUCTURALLY_INVALID_DEVELOPMENT_RUN"] = (
    "STRUCTURALLY_INVALID_DEVELOPMENT_RUN"
)
DEVELOPMENT_ONLY = True
STRUCTURAL_ONLY = True
REPLAY_VERIFIED = False
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

ValidationStatus = Literal[
    "STRUCTURALLY_VALID_DEVELOPMENT_RUN",
    "STRUCTURALLY_INVALID_DEVELOPMENT_RUN",
]


@dataclasses.dataclass(frozen=True)
class V6ValidationError:
    """One deterministic structural validation failure."""

    code: str
    path: str
    message: str


@dataclasses.dataclass(frozen=True)
class V6LifecycleOutcome:
    """C/D chain recomputed from occupied cadence-ledger rows."""

    structural_chain_consistent: bool
    c_first_acquisition_step: int
    c_ever_acquired: bool
    c_retained_after_acquisition: bool
    c_acquired_in_first_c: bool
    d_phase: int
    d_first_acquisition_step: int
    d_retirement_step: int
    d_reacquisition_step: int
    d_first_acquisition_in_first_d: bool
    d_retirement_while_absent: bool
    d_retirement_reset_exact: bool
    d_reacquisition_in_second_d: bool
    d_reacquisition_gate_observed: bool
    d_ordered_outcome: bool
    out_of_order_event_count: int


@dataclasses.dataclass(frozen=True)
class V6CoverageOutcome:
    """Exact support observations; false values do not invalidate structure."""

    complete_window_support: bool
    complete_accepted_window_support: bool
    complete_row_head_support: bool
    complete_filter_cue_support: bool
    complete_cadence_support: bool
    balanced_external_action_support: bool


@dataclasses.dataclass(frozen=True)
class V6QualityOutcome:
    """Threshold-free DEVELOPMENT quality observations."""

    all_steps_accepted: bool
    all_evaluator_lanes_present: bool
    grounded_lane_present: bool
    c_identity_outcome: bool
    d_identity_outcome: bool


@dataclasses.dataclass(frozen=True)
class V6DevelopmentRunValidation:
    """Structural-only result with no replay-authenticity or evidence authority."""

    schema: str
    status: ValidationStatus
    development_only: bool
    structural_only: bool
    replay_verified: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    errors: tuple[V6ValidationError, ...]
    lifecycle: V6LifecycleOutcome
    coverage: V6CoverageOutcome
    quality: V6QualityOutcome


_NEGATIVE_LIFECYCLE = V6LifecycleOutcome(
    structural_chain_consistent=False,
    c_first_acquisition_step=-1,
    c_ever_acquired=False,
    c_retained_after_acquisition=True,
    c_acquired_in_first_c=False,
    d_phase=0,
    d_first_acquisition_step=-1,
    d_retirement_step=-1,
    d_reacquisition_step=-1,
    d_first_acquisition_in_first_d=False,
    d_retirement_while_absent=False,
    d_retirement_reset_exact=False,
    d_reacquisition_in_second_d=False,
    d_reacquisition_gate_observed=False,
    d_ordered_outcome=False,
    out_of_order_event_count=0,
)
_NEGATIVE_COVERAGE = V6CoverageOutcome(False, False, False, False, False, False)
_NEGATIVE_QUALITY = V6QualityOutcome(False, False, False, False, False)


class _ValidationContext:
    def __init__(self) -> None:
        self.errors: list[V6ValidationError] = []

    def add(self, code: str, path: str, message: str) -> None:
        error = V6ValidationError(code=code, path=path, message=message)
        if error not in self.errors:
            self.errors.append(error)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _exact_sha256(ctx: _ValidationContext, value: object, *, path: str) -> str | None:
    if type(value) is not str:
        ctx.add("TYPE", path, "must be an exact built-in str")
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        ctx.add("SHA256", path, "must be one lowercase SHA-256 hex digest")
        return None
    return value


def _array(
    ctx: _ValidationContext,
    value: object,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        ctx.add("TRACER", path, "must not contain a JAX tracer")
        return None
    if not isinstance(value, jax.Array):
        ctx.add("ARRAY_CLASS", path, "must be one concrete JAX Array")
        return None
    expected_dtype = np.dtype(dtype)
    if value.shape != shape:
        ctx.add("SHAPE", path, f"must have exact shape {shape}, got {value.shape}")
        return None
    try:
        actual_dtype = np.dtype(value.dtype)
    except TypeError:
        ctx.add("DTYPE", path, f"must have exact dtype {expected_dtype}")
        return None
    if actual_dtype != expected_dtype:
        ctx.add(
            "DTYPE",
            path,
            f"must have exact dtype {expected_dtype}, got {actual_dtype}",
        )
        return None
    try:
        result = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as exc:
        ctx.add("CONCRETE", path, f"must be concrete: {exc}")
        return None
    if result.shape != shape or result.dtype != expected_dtype:
        ctx.add("ARRAY_CLASS", path, "changed shape or dtype during host transfer")
        return None
    return result


def _scalar_int(ctx: _ValidationContext, value: object, *, path: str) -> int | None:
    result = _array(ctx, value, path=path, shape=(), dtype=np.int32)
    return None if result is None else int(result.item())


def _scalar_bool(ctx: _ValidationContext, value: object, *, path: str) -> bool | None:
    result = _array(ctx, value, path=path, shape=(), dtype=np.bool_)
    return None if result is None else bool(result.item())


def _host_int(ctx: _ValidationContext, value: object, *, path: str) -> int | None:
    if type(value) is not int:
        ctx.add("TYPE", path, "must be an exact built-in int")
        return None
    return value


def _host_bool(ctx: _ValidationContext, value: object, *, path: str) -> bool | None:
    if type(value) is not bool:
        ctx.add("TYPE", path, "must be an exact built-in bool")
        return None
    return value


def _host_str(ctx: _ValidationContext, value: object, *, path: str) -> str | None:
    if type(value) is not str:
        ctx.add("TYPE", path, "must be an exact built-in str")
        return None
    return value


def _all_finite(ctx: _ValidationContext, value: np.ndarray | None, *, path: str) -> bool:
    if value is None:
        return False
    if not bool(np.all(np.isfinite(value))):
        ctx.add("NONFINITE", path, "must contain only finite values")
        return False
    return True


def _nonnegative(ctx: _ValidationContext, value: np.ndarray | None, *, path: str) -> bool:
    if value is None:
        return False
    if not bool(np.all(value >= 0)):
        ctx.add("DOMAIN", path, "must contain only non-negative values")
        return False
    return True


def _positive_zero(value: np.ndarray) -> bool:
    if value.dtype != np.dtype(np.float32):
        return False
    return bool(np.all(value.view(np.uint32) == np.uint32(0)))


def _expect_equal(
    ctx: _ValidationContext,
    actual: object,
    expected: object,
    *,
    path: str,
    code: str = "ALGEBRA",
) -> bool:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        equal = bool(np.array_equal(np.asarray(actual), np.asarray(expected)))
    else:
        equal = actual == expected
    if not equal:
        ctx.add(code, path, f"must equal {expected!r}, got {actual!r}")
    return equal


def _expect_array_equal(
    ctx: _ValidationContext,
    actual: np.ndarray | None,
    expected: np.ndarray,
    *,
    path: str,
    code: str = "ALGEBRA",
) -> bool:
    if actual is None:
        return False
    if not bool(np.array_equal(actual, expected)):
        ctx.add(code, path, "does not equal the exact reconstructed value")
        return False
    return True


def _expect_domain(
    ctx: _ValidationContext,
    value: np.ndarray | None,
    predicate: np.ndarray | bool,
    *,
    path: str,
    message: str,
) -> bool:
    if value is None:
        return False
    if not bool(np.all(predicate)):
        ctx.add("DOMAIN", path, message)
        return False
    return True


def _walk_concrete(ctx: _ValidationContext, value: object, *, path: str) -> None:
    if isinstance(value, jax.core.Tracer):
        ctx.add("TRACER", path, "must not contain a JAX tracer")
        return
    if isinstance(value, np.ndarray):
        ctx.add("ARRAY_CLASS", path, "NumPy arrays are not accepted in run records")
        return
    if isinstance(value, jax.Array):
        try:
            if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
                jax.device_get(jr.key_data(value))
            else:
                host = np.asarray(jax.device_get(value))
                if np.issubdtype(host.dtype, np.inexact) and not bool(np.all(np.isfinite(host))):
                    ctx.add("NONFINITE", path, "array must contain only finite values")
        except (TypeError, ValueError) as exc:
            ctx.add("CONCRETE", path, f"array must be concrete: {exc}")
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _walk_concrete(ctx, getattr(value, field.name), path=f"{path}.{field.name}")
        return
    if type(value) is tuple:
        for index, item in enumerate(value):
            _walk_concrete(ctx, item, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        ctx.add("NONFINITE", path, "host float must be finite")


def _exact_class(
    ctx: _ValidationContext,
    value: object,
    expected: type[object],
    *,
    path: str,
) -> bool:
    if type(value) is not expected:
        ctx.add("CLASS", path, f"must be exact {expected.__name__}")
        return False
    return True


_WINDOW_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "scheduled_support": ((WINDOW_BIN_COUNT,), np.int32),
    "accepted_support": ((WINDOW_BIN_COUNT,), np.int32),
    "reward_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "behavior_nll_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "behavior_brier_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "behavior_correct_count": ((WINDOW_BIN_COUNT,), np.int32),
    "filter_selected_regret_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "filter_planner_regret_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "full_information_selected_regret_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "full_information_planner_regret_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "world_posterior_nll_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "world_posterior_brier_sum": ((WINDOW_BIN_COUNT,), np.float32),
    "grounded_support": ((WINDOW_BIN_COUNT,), np.int32),
    "grounded_fit_loss_by_head_sum": ((WINDOW_BIN_COUNT, TARGET_HEADS), np.float32),
    "grounded_representation_loss_by_head_sum": (
        (WINDOW_BIN_COUNT, TARGET_HEADS),
        np.float32,
    ),
    "grounded_representation_gradient_norm_by_head_sum": (
        (WINDOW_BIN_COUNT, TARGET_HEADS),
        np.float32,
    ),
    "critical_present_count": ((WINDOW_BIN_COUNT, len(CRITICAL_PAIRS)), np.int32),
    "critical_durable_read_count": (
        (WINDOW_BIN_COUNT, len(CRITICAL_PAIRS)),
        np.int32,
    ),
    "critical_evidence_refresh_count": (
        (WINDOW_BIN_COUNT, len(CRITICAL_PAIRS)),
        np.int32,
    ),
    "critical_relevance_score_sum": (
        (WINDOW_BIN_COUNT, len(CRITICAL_PAIRS)),
        np.float32,
    ),
    "critical_relevance_error_abs_sum": (
        (WINDOW_BIN_COUNT, len(CRITICAL_PAIRS)),
        np.float32,
    ),
}

_ROW_HEAD_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "support": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.int32),
    "absolute_error_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "fit_loss_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "representation_loss_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "representation_gradient_norm_sum": (
        (JOINT_ACTION_ROWS, TARGET_HEADS),
        np.float32,
    ),
    "feature_contribution_abs_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "row_bias_abs_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "executed_weight_delta_norm_sum": (
        (JOINT_ACTION_ROWS, TARGET_HEADS),
        np.float32,
    ),
    "executed_bias_delta_abs_sum": ((JOINT_ACTION_ROWS, TARGET_HEADS), np.float32),
    "proposed_weight_change_count": ((JOINT_ACTION_ROWS,), np.int32),
    "proposed_bias_change_count": ((JOINT_ACTION_ROWS,), np.int32),
    "row_isolation_failure_count": ((), np.int32),
    "row_head_algebra_failure_count": ((), np.int32),
    "nonfinite_row_head_count": ((), np.int32),
}

_FILTER_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "support": ((), np.int32),
    "optimal_value_sum": ((), np.float32),
    "selected_value_sum": ((), np.float32),
    "selected_regret_sum": ((), np.float32),
    "margin_sum": ((), np.float32),
    "tied_support": ((), np.int32),
    "nontied_support": ((), np.int32),
    "tied_selected_regret_sum": ((), np.float32),
    "tied_focal_action_support": ((2,), np.int32),
    "cue_pattern_support": ((4,), np.int32),
    "cue_flip_support": ((2,), np.int32),
    "cue_flip_count": ((2,), np.int32),
    "filter_recurrence_failure_count": ((), np.int32),
}

_ACTION_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "focal_action_support": ((2,), np.int32),
    "partner_action_support": ((2,), np.int32),
    "joint_row_support": ((JOINT_ACTION_ROWS,), np.int32),
    "ordinary_policy_action_support": ((2,), np.int32),
    "explored_count": ((), np.int32),
    "externally_forced_count": ((), np.int32),
    "policy_schedule_failure_count": ((), np.int32),
    "policy_replay_failure_count": ((), np.int32),
    "rng_chain_failure_count": ((), np.int32),
    "decision_count": ((), np.int32),
}

_AUDIT_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "contract_failure_counts": ((len(V6_CONTRACT_AUDIT_ORDER),), np.int32),
    "intervention_failure_counts": ((len(V6_INTERVENTION_AUDIT_ORDER),), np.int32),
    "intervention_witness_counts": ((len(V6_INTERVENTION_WITNESS_ORDER),), np.int32),
    "component_delta_sums": ((len(V6_COMPONENT_DELTA_ORDER),), np.int32),
    "component_delta_failure_counts": ((len(V6_COMPONENT_DELTA_ORDER),), np.int32),
    "active_steps": ((), np.int32),
    "accepted_steps": ((), np.int32),
    "learner_valid_steps": ((), np.int32),
    "filter_valid_steps": ((), np.int32),
    "oracle_valid_steps": ((), np.int32),
    "mechanism_valid_steps": ((), np.int32),
    "all_finite_steps": ((), np.int32),
    "curation_attempt_count": ((), np.int32),
    "ledger_count": ((), np.int32),
    "ledger_overflow": ((), np.bool_),
}

_LEDGER_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "occupied": ((MAX_CADENCE_LEDGER_ENTRIES,), np.bool_),
    "transition_step": ((MAX_CADENCE_LEDGER_ENTRIES,), np.int32),
    "regime_id": ((MAX_CADENCE_LEDGER_ENTRIES,), np.int32),
    "pre_descriptors": ((MAX_CADENCE_LEDGER_ENTRIES, ACTIVE_PAIR_SLOTS, 2), np.int32),
    "proposal_descriptors": (
        (MAX_CADENCE_LEDGER_ENTRIES, ACTIVE_PAIR_SLOTS, 2),
        np.int32,
    ),
    "applied_descriptors": (
        (MAX_CADENCE_LEDGER_ENTRIES, ACTIVE_PAIR_SLOTS, 2),
        np.int32,
    ),
    "proposal_event": ((MAX_CADENCE_LEDGER_ENTRIES, 6), np.int32),
    "applied_event": ((MAX_CADENCE_LEDGER_ENTRIES, 6), np.int32),
    "critical_slot": ((MAX_CADENCE_LEDGER_ENTRIES, 3, len(CRITICAL_PAIRS)), np.int32),
    "critical_candidate_streak": (
        (MAX_CADENCE_LEDGER_ENTRIES, len(CRITICAL_PAIRS), 2),
        np.int32,
    ),
    "critical_candidate_flags": (
        (MAX_CADENCE_LEDGER_ENTRIES, len(CRITICAL_PAIRS), 6),
        np.bool_,
    ),
    "candidate_reset_mask": (
        (MAX_CADENCE_LEDGER_ENTRIES, 2, CANDIDATE_PAIR_SLOTS),
        np.bool_,
    ),
    "random_curation_flags": ((MAX_CADENCE_LEDGER_ENTRIES, 3), np.bool_),
    "random_curation_selected": ((MAX_CADENCE_LEDGER_ENTRIES, 3), np.int32),
    "random_active_priorities": (
        (MAX_CADENCE_LEDGER_ENTRIES, ACTIVE_PAIR_SLOTS),
        np.float32,
    ),
    "random_candidate_priorities": (
        (MAX_CADENCE_LEDGER_ENTRIES, CANDIDATE_PAIR_SLOTS),
        np.float32,
    ),
    "consumer_masks": ((MAX_CADENCE_LEDGER_ENTRIES, 9, ACTIVE_PAIR_SLOTS), np.bool_),
    "router_source_slots": ((MAX_CADENCE_LEDGER_ENTRIES, ACTIVE_PAIR_SLOTS), np.int32),
    "router_masks": ((MAX_CADENCE_LEDGER_ENTRIES, 3, ACTIVE_PAIR_SLOTS), np.bool_),
    "router_flags": ((MAX_CADENCE_LEDGER_ENTRIES, 4), np.bool_),
    "router_counts": ((MAX_CADENCE_LEDGER_ENTRIES, 4), np.int32),
    "transaction_exact": ((MAX_CADENCE_LEDGER_ENTRIES,), np.bool_),
    "identity_carry_exact": ((MAX_CADENCE_LEDGER_ENTRIES,), np.bool_),
    "retired_identity_reset_exact": ((MAX_CADENCE_LEDGER_ENTRIES,), np.bool_),
}

_LIFECYCLE_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "structural_valid": ((), np.bool_),
    "c_first_acquisition_step": ((), np.int32),
    "c_ever_acquired": ((), np.bool_),
    "c_retained_after_acquisition": ((), np.bool_),
    "c_acquired_in_first_c": ((), np.bool_),
    "d_phase": ((), np.int32),
    "d_first_acquisition_step": ((), np.int32),
    "d_retirement_step": ((), np.int32),
    "d_reacquisition_step": ((), np.int32),
    "d_first_acquisition_in_first_d": ((), np.bool_),
    "d_retirement_while_absent": ((), np.bool_),
    "d_retirement_reset_exact": ((), np.bool_),
    "d_reacquisition_in_second_d": ((), np.bool_),
    "d_reacquisition_gate_observed": ((), np.bool_),
    "d_ordered_outcome": ((), np.bool_),
    "out_of_order_event_count": ((), np.int32),
}

_RNG_CONTRACT: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "supplied_key_data": ((2, 2), np.uint32),
    "initial_world_key_data": ((len(V6_WORLD_RNG_KEY_ORDER), 2), np.uint32),
    "final_world_key_data": ((len(V6_WORLD_RNG_KEY_ORDER), 2), np.uint32),
    "initial_policy_key_data": ((len(V6_POLICY_KEY_ENDPOINT_ORDER), 2), np.uint32),
    "final_policy_key_data": ((len(V6_POLICY_KEY_ENDPOINT_ORDER), 2), np.uint32),
    "initial_interaction_key_data": ((2,), np.uint32),
    "final_interaction_key_data": ((2,), np.uint32),
    "initial_stream_bits": ((), np.uint8),
    "world_draw_counts": ((len(V6_WORLD_RNG_KEY_ORDER),), np.int32),
    "interaction_key_advance_count": ((), np.int32),
    "policy_decision_count": ((), np.int32),
}


def _record_arrays(
    ctx: _ValidationContext,
    value: object,
    expected: type[object],
    contract: dict[str, tuple[tuple[int, ...], type[np.generic]]],
    *,
    path: str,
) -> dict[str, np.ndarray]:
    if not _exact_class(ctx, value, expected, path=path):
        return {}
    actual_fields = tuple(field.name for field in dataclasses.fields(cast(Any, value)))
    if actual_fields != tuple(contract):
        ctx.add("SCHEMA", path, "record field order differs from the reviewed contract")
        return {}
    arrays: dict[str, np.ndarray] = {}
    for field_name, (shape, dtype) in contract.items():
        result = _array(
            ctx,
            getattr(value, field_name),
            path=f"{path}.{field_name}",
            shape=shape,
            dtype=dtype,
        )
        if result is not None:
            arrays[field_name] = result
    return arrays


def _validate_static_orders(ctx: _ValidationContext) -> None:
    expected = {
        "V6_PROPOSAL_EVENT_ORDER": (
            "replaced_slot",
            "promoted_candidate",
            "refreshed_candidate",
            "retired_slot",
            "retired_left",
            "retired_right",
        ),
        "V6_APPLIED_EVENT_ORDER": (
            "replaced_slot",
            "promoted_candidate",
            "refreshed_candidate",
            "retired_slot",
            "retired_left",
            "retired_right",
        ),
        "V6_CRITICAL_STAGE_ORDER": ("pre", "proposal", "applied"),
        "V6_CANDIDATE_STREAK_ENDPOINT_ORDER": ("pre", "post"),
        "V6_CANDIDATE_FLAG_ORDER": (
            "promotion_raw_evidence",
            "promotion_confirmed",
            "reacquisition_required_pre",
            "reacquisition_required_proposal_post",
            "reacquisition_required_post",
            "reacquisition_confirmed",
        ),
        "V6_RANDOM_CURATION_FLAG_ORDER": ("enabled", "attempted", "applied"),
        "V6_RANDOM_CURATION_SELECTED_ORDER": (
            "active_worst_slot",
            "promotion_candidate",
            "refresh_candidate",
        ),
        "V6_CONSUMER_MASK_ORDER": (
            "durable_read",
            "read_acquire_pre",
            "read_acquire_post",
            "confirmed_write_pre",
            "confirmed_write_post",
            "read_pre",
            "read_post",
            "active_pre",
            "active_post",
        ),
        "V6_ROUTER_MASK_ORDER": ("survivor", "new", "evicted"),
        "V6_ROUTER_FLAG_ORDER": (
            "valid",
            "applied",
            "carry_survivors",
            "descriptors_changed",
        ),
        "V6_ROUTER_COUNT_ORDER": (
            "route_before",
            "route_after",
            "generation_before",
            "generation_after",
        ),
        "V6_POLICY_KEY_ENDPOINT_ORDER": ("before", "after"),
    }
    live: dict[str, tuple[str, ...]] = {
        "V6_PROPOSAL_EVENT_ORDER": V6_PROPOSAL_EVENT_ORDER,
        "V6_APPLIED_EVENT_ORDER": V6_APPLIED_EVENT_ORDER,
        "V6_CRITICAL_STAGE_ORDER": V6_CRITICAL_STAGE_ORDER,
        "V6_CANDIDATE_STREAK_ENDPOINT_ORDER": V6_CANDIDATE_STREAK_ENDPOINT_ORDER,
        "V6_CANDIDATE_FLAG_ORDER": V6_CANDIDATE_FLAG_ORDER,
        "V6_RANDOM_CURATION_FLAG_ORDER": V6_RANDOM_CURATION_FLAG_ORDER,
        "V6_RANDOM_CURATION_SELECTED_ORDER": V6_RANDOM_CURATION_SELECTED_ORDER,
        "V6_CONSUMER_MASK_ORDER": V6_CONSUMER_MASK_ORDER,
        "V6_ROUTER_MASK_ORDER": V6_ROUTER_MASK_ORDER,
        "V6_ROUTER_FLAG_ORDER": V6_ROUTER_FLAG_ORDER,
        "V6_ROUTER_COUNT_ORDER": V6_ROUTER_COUNT_ORDER,
        "V6_POLICY_KEY_ENDPOINT_ORDER": V6_POLICY_KEY_ENDPOINT_ORDER,
    }
    for name, exact in expected.items():
        if live[name] != exact:
            ctx.add("LIVE_ORDER", name, "reviewed ordered binding has drifted")
    expected_targets = (
        "x",
        "previous_contextual_outcome",
        "previous_partner_action",
        "has_partner_history",
        "u",
        "v",
        "world_cue_1",
        "world_cue_2",
        "reward",
        "discount",
    )
    if V6_TARGET_HEAD_ORDER != expected_targets:
        ctx.add("LIVE_ORDER", "V6_TARGET_HEAD_ORDER", "reviewed target-head order has drifted")
    if V6_WORLD_RNG_KEY_ORDER != ("signal", "partner", "world", "cue", "outcome"):
        ctx.add("LIVE_ORDER", "V6_WORLD_RNG_KEY_ORDER", "reviewed world-key order has drifted")
    expected_component = (
        "state_builder_step",
        "state_builder_learning",
        "behavior",
        "interaction",
        "table_world",
        "grounded_world",
        "control",
        "router_route",
        "router_generation",
        "integrated",
    )
    if V6_COMPONENT_DELTA_ORDER != expected_component:
        ctx.add("LIVE_ORDER", "V6_COMPONENT_DELTA_ORDER", "reviewed component order has drifted")
    expected_audits = (
        "entry_state_contract",
        "config_token",
        "counter_synchronization",
        "action_domain",
        "action_policy",
        "selection_binding",
        "policy_replay",
        "next_action_policy",
        "next_selection_binding",
        "next_policy_replay",
        "next_selection_diagnostics",
        "learner_trace",
        "filter_trace",
        "oracle_trace",
        "all_finite",
        "mechanism_trace",
        "oracle_schedule",
        "stream_domain",
        "grounded_target_algebra",
        "grounded_row_head_algebra",
        "descriptor_domain",
        "lifecycle_event_cadence",
        "router_transaction",
        "consumer_identity_carry",
        "retired_identity_reset",
        "rng_chain",
        "filter_recurrence",
    )
    if V6_CONTRACT_AUDIT_ORDER != expected_audits:
        ctx.add("LIVE_ORDER", "V6_CONTRACT_AUDIT_ORDER", "reviewed audit order has drifted")
    expected_interventions = (
        "behavior_credit_replay",
        "grounded_credit_replay",
        "gradient_mix_mode_bounded_replay",
        "gradient_chain_bounded_replay",
        "state_learning_gate_bounded_replay",
        "grounded_learning_gate_exact",
        "memory_mask_exact",
        "planner_reward_source_exact",
        "planning_application_exact",
        "partner_belief_exact",
        "lifecycle_commit_gate_exact",
        "identity_carry_mode_exact",
        "retention_floor_exact",
        "retirement_gate_exact",
        "random_curation_exact",
        "uniform_action_exact",
        "cue_sampling_exact",
        "row_bias_exact",
    )
    if V6_INTERVENTION_AUDIT_ORDER != expected_interventions:
        ctx.add(
            "LIVE_ORDER",
            "V6_INTERVENTION_AUDIT_ORDER",
            "reviewed intervention audit order has drifted",
        )
    expected_witnesses = (
        "behavior_credit_nonzero",
        "grounded_credit_nonzero",
        "state_parameter_proposal_nonzero",
        "grounded_parameter_proposal_nonzero",
        "lifecycle_proposal_event",
        "applied_descriptor_change",
        "retention_floor_counterfactual_bind",
        "retirement_eligible",
        "random_selection_differs_from_utility_selection",
        "masked_hidden_state_downstream_learning_effect",
        "table_and_grounded_rewards_disagree",
        "planner_model_term_nonzero",
        "partner_prediction_nonuniform",
        "forced_action_differs_from_ordinary",
        "equal_cue_differs_from_base_counterfactual",
        "row_bias_proposal_nonzero",
    )
    if V6_INTERVENTION_WITNESS_ORDER != expected_witnesses:
        ctx.add(
            "LIVE_ORDER",
            "V6_INTERVENTION_WITNESS_ORDER",
            "reviewed intervention witness order has drifted",
        )
    expected_required = (
        ("full", ()),
        ("grounded_model_frozen", ("grounded_parameter_proposal_nonzero",)),
        ("world_credit_off", ("behavior_credit_nonzero", "grounded_credit_nonzero")),
        ("behavior_credit_off", ("behavior_credit_nonzero", "grounded_credit_nonzero")),
        (
            "all_representation_credit_off",
            ("behavior_credit_nonzero", "grounded_credit_nonzero"),
        ),
        ("state_frozen", ("state_parameter_proposal_nonzero",)),
        (
            "recurrent_memory_masked",
            ("masked_hidden_state_downstream_learning_effect",),
        ),
        ("table_planner", ("table_and_grounded_rewards_disagree",)),
        ("no_planning", ("planner_model_term_nonzero",)),
        ("uniform_partner", ("partner_prediction_nonuniform",)),
        ("lifecycle_frozen", ("lifecycle_proposal_event",)),
        ("no_identity_carry", ("applied_descriptor_change",)),
        ("no_retention_floor", ("retention_floor_counterfactual_bind",)),
        ("retirement_disabled", ("retirement_eligible",)),
        ("random_curation", ("random_selection_differs_from_utility_selection",)),
        ("uniform_action", ("forced_action_differs_from_ordinary",)),
        ("equal_cue", ("equal_cue_differs_from_base_counterfactual",)),
        ("row_bias", ("row_bias_proposal_nonzero",)),
    )
    if V6_CONTROL_REQUIRED_WITNESSES != expected_required:
        ctx.add(
            "LIVE_ORDER",
            "V6_CONTROL_REQUIRED_WITNESSES",
            "reviewed control witness mapping has drifted",
        )
    if (
        V6_FLOAT32_REPLAY_RTOL != 2.0**-20
        or V6_FLOAT32_REPLAY_ATOL != 2.0**-22
    ):
        ctx.add(
            "LIVE_ORDER",
            "V6_FLOAT32_REPLAY_TOLERANCE",
            "reviewed bounded-replay tolerance has drifted",
        )


def _select_live_control(
    ctx: _ValidationContext,
    *,
    name: object,
    primary: object,
) -> HiddenPartnerLifecycleWorldV6Control | None:
    control_name = _host_str(ctx, name, path="run.control_name")
    primary_value = _host_bool(ctx, primary, path="run.primary")
    if control_name is None or primary_value is None:
        return None
    declared_order = PRIMARY_CONDITION_ORDER if primary_value else V6_DIAGNOSTIC_ORDER
    if control_name not in declared_order:
        ctx.add("CONTROL_NAME", "run.control_name", "is not in the selected live family order")
        return None
    try:
        controls = build_v6_primary_controls() if primary_value else build_v6_diagnostic_controls()
        matches = tuple(control for control in controls if control.name == control_name)
        if len(matches) != 1:
            ctx.add("LIVE_BINDING", "run.control_name", "must resolve to one live control")
            return None
        return validate_v6_control(matches[0])
    except (TypeError, ValueError, RuntimeError) as exc:
        ctx.add("LIVE_BINDING", "run.control_name", f"live control construction failed: {exc}")
        return None


def _validate_live_binding(
    ctx: _ValidationContext,
    run: V6DevelopmentRun,
) -> tuple[HiddenPartnerLifecycleWorldV6Runner | None, HiddenPartnerLifecycleWorldV6Control | None]:
    if HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNNER_SCHEMA != (
        "alberta.hidden-partner-lifecycle-world.runner-development.v1"
    ):
        ctx.add("LIVE_SCHEMA", "runner.schema", "runner schema has drifted")
    if RUNNER_STATUS != "DEVELOPMENT_RUNNER_NO_EVIDENCE_AUTHORITY":
        ctx.add("AUTHORITY", "runner.status", "runner status has unexpected authority")
    if (
        RUNNER_DEVELOPMENT_ONLY is not True
        or RUNNER_EXECUTION_AUTHORIZED is not False
        or RUNNER_EVIDENCE_AUTHORIZED is not False
        or RUNNER_SCIENTIFIC_PROMOTION_ALLOWED is not False
    ):
        ctx.add("AUTHORITY", "runner", "live runner authority constants must remain disabled")

    control = _select_live_control(ctx, name=run.control_name, primary=run.primary)
    if control is None:
        return None, None
    try:
        readiness = validate_v6_control_suite_readiness(require_v6_control_suite_ready())
    except (TypeError, ValueError, RuntimeError) as exc:
        ctx.add("READINESS", "live.readiness", f"live control suite is not ready: {exc}")
        return None, control
    family = "primary" if control.primary else "diagnostic"
    matches = tuple(
        binding
        for binding in readiness.bindings
        if binding.family == family and binding.name == control.name
    )
    if len(matches) != 1:
        ctx.add("READINESS", "live.binding", "selected control needs one ordered readiness binding")
        return None, control
    binding = matches[0]
    digest_values = {
        "control_config_sha256": binding.control_config_sha256,
        "control_matrix_sha256": readiness.control_matrix_sha256,
        "bridge_config_sha256": binding.bridge_config_sha256,
    }
    for field_name, expected in digest_values.items():
        actual = _exact_sha256(ctx, getattr(run, field_name), path=f"run.{field_name}")
        if actual is not None and actual != expected:
            ctx.add("LIVE_DIGEST", f"run.{field_name}", "differs from the live ordered binding")
    try:
        runner = HiddenPartnerLifecycleWorldV6Runner(control)
    except (TypeError, ValueError, RuntimeError, FileNotFoundError) as exc:
        ctx.add("RUNNER_CONSTRUCTION", "live.runner", f"live runner construction failed: {exc}")
        return None, control
    if runner.bridge.config_token_hex != binding.bridge_config_sha256:
        ctx.add("LIVE_DIGEST", "live.bridge", "constructed bridge differs from readiness")

    try:
        source_closure = compute_v6_source_closure_hashes()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("SOURCE_CLOSURE_MISSING", "run.source_closure_hashes", str(exc))
        return runner, control
    if type(run.source_closure_hashes) is not tuple:
        ctx.add("TYPE", "run.source_closure_hashes", "must be an exact tuple")
    else:
        if len(run.source_closure_hashes) != len(V6_SOURCE_CLOSURE_PATHS):
            ctx.add("SOURCE_CLOSURE", "run.source_closure_hashes", "has the wrong length")
        for index, expected_source in enumerate(source_closure):
            path = f"run.source_closure_hashes[{index}]"
            if index >= len(run.source_closure_hashes):
                ctx.add("SOURCE_CLOSURE_MISSING", path, "source record is missing")
                continue
            source_record = run.source_closure_hashes[index]
            if not _exact_class(ctx, source_record, V6SourceClosureHash, path=path):
                continue
            exact_source = source_record
            relative_path = _host_str(
                ctx,
                exact_source.relative_path,
                path=f"{path}.relative_path",
            )
            digest = _exact_sha256(ctx, exact_source.sha256, path=f"{path}.sha256")
            if relative_path != expected_source.relative_path or digest != expected_source.sha256:
                ctx.add("SOURCE_CLOSURE", path, "differs from the live ordered source closure")
        if len(run.source_closure_hashes) > len(source_closure):
            ctx.add("SOURCE_CLOSURE", "run.source_closure_hashes", "contains extra records")
    if tuple(record.relative_path for record in source_closure) != V6_SOURCE_CLOSURE_PATHS:
        ctx.add("SOURCE_CLOSURE", "live.source_closure", "live source order differs")

    try:
        # Runtime and every source record were independently checked against
        # the live process above; private reconstruction preserves the exact
        # historical digest without reopening public provenance injection.
        config = runner._config_for_source_closure(  # noqa: SLF001
            run.source_closure_hashes,
            run.runtime,
        )
        if config.get("development_only") is not True:
            ctx.add("AUTHORITY", "live.runner_config.development_only", "must be true")
        for key in ("execution_authorized", "evidence_authorized", "scientific_promotion_allowed"):
            if config.get(key) is not False:
                ctx.add("AUTHORITY", f"live.runner_config.{key}", "must be false")
        if config.get("thresholds") is not None or config.get("outcomes") is not None:
            ctx.add("AUTHORITY", "live.runner_config", "must not define thresholds or outcomes")
        if config.get("seed_namespace") is not None or config.get("writes_files") is not False:
            ctx.add("AUTHORITY", "live.runner_config", "must not derive seeds or write files")
        expected_runner_digest = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
        actual_runner_digest = _exact_sha256(
            ctx,
            run.runner_config_sha256,
            path="run.runner_config_sha256",
        )
        if actual_runner_digest is not None and actual_runner_digest != expected_runner_digest:
            ctx.add(
                "LIVE_DIGEST",
                "run.runner_config_sha256",
                "differs from the canonical bound runner config",
            )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("RUNNER_CONFIG", "live.runner_config", f"cannot reconstruct config: {exc}")
    return runner, control


def _validate_runtime_provenance(
    ctx: _ValidationContext,
    runtime: object,
) -> V6RuntimeRecord | None:
    """Require one exact runtime record that still matches the live process."""

    try:
        return validate_v6_runtime_record(runtime, require_live_match=True)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add(
            "RUNTIME_PROVENANCE",
            "run.runtime",
            f"runtime provenance is invalid or stale: {exc}",
        )
        return None


def _exact_tree_contract(
    ctx: _ValidationContext,
    actual: object,
    expected: object,
    *,
    path: str,
    compare_values: bool,
) -> None:
    if isinstance(expected, jax.Array):
        if isinstance(actual, jax.core.Tracer):
            ctx.add("TRACER", path, "must not contain a JAX tracer")
            return
        if not isinstance(actual, jax.Array):
            ctx.add("ARRAY_CLASS", path, "must be one concrete JAX Array")
            return
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            ctx.add("STATE_CONTRACT", path, "shape or dtype differs from live initialization")
            return
        try:
            if jnp.issubdtype(actual.dtype, jax.dtypes.prng_key):
                actual_host = np.asarray(jax.device_get(jr.key_data(actual)))
                expected_host = np.asarray(jax.device_get(jr.key_data(expected)))
            else:
                actual_host = np.asarray(jax.device_get(actual))
                expected_host = np.asarray(jax.device_get(expected))
        except (TypeError, ValueError) as exc:
            ctx.add("CONCRETE", path, f"state leaf must be concrete: {exc}")
            return
        if compare_values and not bool(np.array_equal(actual_host, expected_host)):
            ctx.add("INITIAL_STATE", path, "differs from deterministic live initialization")
        return
    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        if type(actual) is not type(expected):
            ctx.add("CLASS", path, f"must be exact {type(expected).__name__}")
            return
        for field in dataclasses.fields(expected):
            _exact_tree_contract(
                ctx,
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
                compare_values=compare_values,
            )
        return
    if type(expected) is tuple:
        if type(actual) is not tuple or len(cast(tuple[object, ...], actual)) != len(expected):
            ctx.add("STATE_CONTRACT", path, "tuple structure differs from live initialization")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _exact_tree_contract(
                ctx,
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                compare_values=compare_values,
            )
        return
    if actual is None or expected is None:
        if actual is not expected:
            ctx.add("STATE_CONTRACT", path, "optional structure differs from live initialization")
        return
    if type(actual) is not type(expected):
        ctx.add("TYPE", path, f"must be exact {type(expected).__name__}")
    elif compare_values and actual != expected:
        ctx.add("INITIAL_STATE", path, "differs from deterministic live initialization")


def _validate_plan(
    ctx: _ValidationContext,
    run: V6DevelopmentRun,
) -> HiddenPartnerLifecycleWorldV6ScanPlan | None:
    if not _exact_class(
        ctx,
        run.plan,
        HiddenPartnerLifecycleWorldV6ScanPlan,
        path="run.plan",
    ):
        return None
    try:
        validated = validate_hidden_partner_lifecycle_world_v6_scan_plan(run.plan)
    except (TypeError, ValueError, RuntimeError) as exc:
        ctx.add("PLAN", "run.plan", f"plan validation failed: {exc}")
        return None
    if type(run.initial_state) is not HiddenPartnerWorldOnlineState:
        ctx.add("CLASS", "run.initial_state", "must be exact HiddenPartnerWorldOnlineState")
        return validated
    try:
        reconstructed = validate_hidden_partner_lifecycle_world_v6_scan_plan(
            build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(run.initial_state.world)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        ctx.add("PLAN_RECONSTRUCTION", "run.plan", f"cannot rebuild from initial world: {exc}")
        return validated
    if validated != reconstructed:
        ctx.add("PLAN_RECONSTRUCTION", "run.plan", "differs from exact initial-world geometry")
    return validated


def _key_data(key: object) -> np.ndarray | None:
    if not isinstance(key, jax.Array) or isinstance(key, jax.core.Tracer):
        return None
    try:
        return np.asarray(jax.device_get(jr.key_data(key)), dtype=np.uint32)
    except (TypeError, ValueError):
        return None


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...] | None:
    signature: list[tuple[tuple[int, ...], str]] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.core.Tracer) or not isinstance(leaf, jax.Array):
            return None
        signature.append((tuple(leaf.shape), str(leaf.dtype)))
    return tuple(signature)


def _tree_nbytes(tree: object) -> int | None:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.core.Tracer) or not isinstance(leaf, jax.Array):
            return None
        total += int(leaf.nbytes)
    return total


def _state_counter(
    ctx: _ValidationContext,
    state: HiddenPartnerWorldOnlineState,
    dotted: str,
    *,
    endpoint: str,
) -> int | None:
    value: object = state
    for part in dotted.split("."):
        value = getattr(value, part)
    return _scalar_int(ctx, value, path=f"run.{endpoint}_state.{dotted}")


def _validate_state_endpoints(
    ctx: _ValidationContext,
    run: V6DevelopmentRun,
    runner: HiddenPartnerLifecycleWorldV6Runner | None,
    rng: dict[str, np.ndarray],
) -> None:
    initial_exact = _exact_class(
        ctx,
        run.initial_state,
        HiddenPartnerWorldOnlineState,
        path="run.initial_state",
    )
    final_exact = _exact_class(
        ctx,
        run.final_state,
        HiddenPartnerWorldOnlineState,
        path="run.final_state",
    )
    if not initial_exact or not final_exact:
        return
    for endpoint, state in (("initial", run.initial_state), ("final", run.final_state)):
        _exact_class(
            ctx, state.world, HiddenPartnerWorldFeedbackState, path=f"run.{endpoint}_state.world"
        )
        _exact_class(
            ctx, state.agent, IntegratedHiddenPartnerState, path=f"run.{endpoint}_state.agent"
        )
        _exact_class(
            ctx,
            state.world_filter,
            HiddenPartnerWorldFilterState,
            path=f"run.{endpoint}_state.world_filter",
        )
        _walk_concrete(ctx, state, path=f"run.{endpoint}_state")
        _array(
            ctx,
            state.config_token,
            path=f"run.{endpoint}_state.config_token",
            shape=(32,),
            dtype=np.uint8,
        )
        action = _scalar_int(ctx, state.action, path=f"run.{endpoint}_state.action")
        if action is not None and action not in (0, 1):
            ctx.add("DOMAIN", f"run.{endpoint}_state.action", "must be binary")
        _scalar_bool(ctx, state.valid, path=f"run.{endpoint}_state.valid")
        _scalar_int(ctx, state.step_count, path=f"run.{endpoint}_state.step_count")

    if runner is None or "supplied_key_data" not in rng:
        return
    try:
        supplied = rng["supplied_key_data"]
        world_key = jr.wrap_key_data(jnp.asarray(supplied[0], dtype=jnp.uint32))
        agent_key = jr.wrap_key_data(jnp.asarray(supplied[1], dtype=jnp.uint32))
        expected_initial = runner.initialize(world_key, agent_key)
    except (TypeError, ValueError, RuntimeError) as exc:
        ctx.add("INITIAL_STATE", "run.initial_state", f"live initialization failed: {exc}")
        return
    _exact_tree_contract(
        ctx,
        run.initial_state,
        expected_initial,
        path="run.initial_state",
        compare_values=True,
    )
    _exact_tree_contract(
        ctx,
        run.final_state,
        expected_initial,
        path="run.final_state",
        compare_values=False,
    )
    expected_token = np.asarray(jax.device_get(expected_initial.config_token))
    final_token = np.asarray(jax.device_get(run.final_state.config_token))
    if not bool(np.array_equal(final_token, expected_token)):
        ctx.add("CONFIG_TOKEN", "run.final_state.config_token", "must preserve the live token")


def _validate_windows(
    ctx: _ValidationContext,
    arrays: dict[str, np.ndarray],
) -> tuple[bool, bool]:
    if len(arrays) != len(_WINDOW_CONTRACT):
        return False, False
    expected_scheduled = np.asarray(
        (ENTRY_WINDOW_STEPS,) * 36 + (FINAL_WINDOW_STEPS,),
        dtype=np.int32,
    )
    scheduled = arrays["scheduled_support"]
    accepted = arrays["accepted_support"]
    grounded = arrays["grounded_support"]
    _expect_array_equal(
        ctx,
        scheduled,
        expected_scheduled,
        path="run.windows.scheduled_support",
        code="WINDOW_GEOMETRY",
    )
    _expect_array_equal(
        ctx,
        accepted,
        expected_scheduled,
        path="run.windows.accepted_support",
        code="WINDOW_ACCEPTANCE",
    )
    for name, value in arrays.items():
        if value.dtype == np.dtype(np.float32):
            _all_finite(ctx, value, path=f"run.windows.{name}")
    for name in (
        "scheduled_support",
        "accepted_support",
        "behavior_correct_count",
        "grounded_support",
        "critical_present_count",
        "critical_durable_read_count",
        "critical_evidence_refresh_count",
    ):
        _nonnegative(ctx, arrays[name], path=f"run.windows.{name}")
    for name in (
        "reward_sum",
        "behavior_nll_sum",
        "behavior_brier_sum",
        "filter_selected_regret_sum",
        "filter_planner_regret_sum",
        "full_information_selected_regret_sum",
        "full_information_planner_regret_sum",
        "world_posterior_nll_sum",
        "world_posterior_brier_sum",
        "grounded_fit_loss_by_head_sum",
        "grounded_representation_loss_by_head_sum",
        "grounded_representation_gradient_norm_by_head_sum",
        "critical_relevance_error_abs_sum",
    ):
        _nonnegative(ctx, arrays[name], path=f"run.windows.{name}")
    bounded_by_accepted = (
        "reward_sum",
        "filter_selected_regret_sum",
        "filter_planner_regret_sum",
        "full_information_selected_regret_sum",
        "full_information_planner_regret_sum",
        "world_posterior_brier_sum",
    )
    for name in bounded_by_accepted:
        if not bool(np.all(arrays[name] <= accepted.astype(np.float32))):
            ctx.add("WINDOW_ALGEBRA", f"run.windows.{name}", "exceeds its exact support bound")
    if not bool(np.all(arrays["behavior_brier_sum"] <= 2.0 * accepted.astype(np.float32))):
        ctx.add(
            "WINDOW_ALGEBRA",
            "run.windows.behavior_brier_sum",
            "exceeds binary Brier support bound",
        )
    if not bool(np.all(accepted <= scheduled)):
        ctx.add("WINDOW_ALGEBRA", "run.windows.accepted_support", "must not exceed scheduled")
    if not bool(np.all(grounded <= accepted)):
        ctx.add("WINDOW_ALGEBRA", "run.windows.grounded_support", "must not exceed accepted")
    if not bool(np.all(arrays["behavior_correct_count"] <= accepted)):
        ctx.add("WINDOW_ALGEBRA", "run.windows.behavior_correct_count", "exceeds accepted")
    present = arrays["critical_present_count"]
    if not bool(np.all(present <= accepted[:, None])):
        ctx.add("WINDOW_ALGEBRA", "run.windows.critical_present_count", "exceeds accepted")
    for name in ("critical_durable_read_count", "critical_evidence_refresh_count"):
        if not bool(np.all(arrays[name] <= present)):
            ctx.add("WINDOW_ALGEBRA", f"run.windows.{name}", "exceeds critical presence")

    accepted_zero = accepted == 0
    for name in (
        "reward_sum",
        "behavior_nll_sum",
        "behavior_brier_sum",
        "filter_selected_regret_sum",
        "filter_planner_regret_sum",
        "full_information_selected_regret_sum",
        "full_information_planner_regret_sum",
        "world_posterior_nll_sum",
        "world_posterior_brier_sum",
    ):
        if not _positive_zero(arrays[name][accepted_zero]):
            ctx.add("POSITIVE_ZERO", f"run.windows.{name}", "zero-support bins must be +0.0")
    grounded_zero = grounded == 0
    for name in (
        "grounded_fit_loss_by_head_sum",
        "grounded_representation_loss_by_head_sum",
        "grounded_representation_gradient_norm_by_head_sum",
    ):
        if not _positive_zero(arrays[name][grounded_zero, :]):
            ctx.add("POSITIVE_ZERO", f"run.windows.{name}", "zero-support bins must be +0.0")
    present_zero = present == 0
    for name in ("critical_relevance_score_sum", "critical_relevance_error_abs_sum"):
        if not _positive_zero(arrays[name][present_zero]):
            ctx.add("POSITIVE_ZERO", f"run.windows.{name}", "absent identities must sum to +0.0")
    return bool(np.all(scheduled > 0)), bool(np.all(accepted > 0))


def _validate_row_heads(
    ctx: _ValidationContext,
    arrays: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    control: HiddenPartnerLifecycleWorldV6Control | None,
) -> bool:
    if len(arrays) != len(_ROW_HEAD_CONTRACT):
        return False
    support = arrays["support"]
    for name, value in arrays.items():
        if value.dtype == np.dtype(np.float32):
            _all_finite(ctx, value, path=f"run.row_heads.{name}")
        _nonnegative(ctx, value, path=f"run.row_heads.{name}")
    for name in (
        "row_isolation_failure_count",
        "row_head_algebra_failure_count",
        "nonfinite_row_head_count",
    ):
        if int(arrays[name].item()) != 0:
            ctx.add("ROW_HEAD_AUDIT", f"run.row_heads.{name}", "must be zero")
    row_support = action.get("joint_row_support")
    grounded_present = bool(
        control is not None
        and control.agent_config is not None
        and control.agent_config.grounded_world_model is not None
    )
    if row_support is not None:
        expected = row_support[:, None] if grounded_present else np.zeros_like(support)
        expected = np.broadcast_to(expected, support.shape)
        _expect_array_equal(
            ctx,
            support,
            expected,
            path="run.row_heads.support",
            code="ROW_HEAD_ALGEBRA",
        )
    zero_cells = support == 0
    for name in (
        "absolute_error_sum",
        "fit_loss_sum",
        "representation_loss_sum",
        "representation_gradient_norm_sum",
        "feature_contribution_abs_sum",
        "row_bias_abs_sum",
        "executed_weight_delta_norm_sum",
        "executed_bias_delta_abs_sum",
    ):
        if not _positive_zero(arrays[name][zero_cells]):
            ctx.add("POSITIVE_ZERO", f"run.row_heads.{name}", "zero-support cells must be +0.0")
    weights = np.asarray(V6_REPRESENTATION_LOSS_WEIGHTS, dtype=np.float32)
    expected_rep = arrays["fit_loss_sum"] * weights[None, :]
    if not bool(np.allclose(arrays["representation_loss_sum"], expected_rep, rtol=2e-6, atol=2e-6)):
        ctx.add(
            "ROW_HEAD_ALGEBRA",
            "run.row_heads.representation_loss_sum",
            "must equal the weighted fit-loss accumulation",
        )
    zero_weight = weights == np.float32(0.0)
    if not _positive_zero(arrays["representation_loss_sum"][:, zero_weight]):
        ctx.add(
            "POSITIVE_ZERO",
            "run.row_heads.representation_loss_sum",
            "zero-weight heads must be +0.0",
        )
    if grounded_present and row_support is not None:
        for name in ("proposed_weight_change_count", "proposed_bias_change_count"):
            if not bool(np.all(arrays[name] <= row_support)):
                ctx.add("ROW_HEAD_ALGEBRA", f"run.row_heads.{name}", "exceeds executed-row support")
    return bool(np.all(support > 0)) if grounded_present else False


def _validate_filter_and_action(
    ctx: _ValidationContext,
    filters: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    control: HiddenPartnerLifecycleWorldV6Control | None,
) -> tuple[bool, bool]:
    if len(filters) != len(_FILTER_CONTRACT) or len(action) != len(_ACTION_CONTRACT):
        return False, False
    for name, value in filters.items():
        if value.dtype == np.dtype(np.float32):
            _all_finite(ctx, value, path=f"run.filter_totals.{name}")
        _nonnegative(ctx, value, path=f"run.filter_totals.{name}")
    for name, value in action.items():
        _nonnegative(ctx, value, path=f"run.action_totals.{name}")
    support = int(filters["support"].item())
    tied = int(filters["tied_support"].item())
    nontied = int(filters["nontied_support"].item())
    if tied + nontied != support:
        ctx.add("FILTER_ALGEBRA", "run.filter_totals", "tied plus nontied must equal support")
    if int(np.sum(filters["tied_focal_action_support"])) != tied:
        ctx.add(
            "FILTER_ALGEBRA",
            "run.filter_totals.tied_focal_action_support",
            "must sum to tied support",
        )
    if int(np.sum(filters["cue_pattern_support"])) != support:
        ctx.add("FILTER_ALGEBRA", "run.filter_totals.cue_pattern_support", "must sum to support")
    _expect_array_equal(
        ctx,
        filters["cue_flip_support"],
        np.full((2,), support, dtype=np.int32),
        path="run.filter_totals.cue_flip_support",
        code="FILTER_ALGEBRA",
    )
    if not bool(np.all(filters["cue_flip_count"] <= filters["cue_flip_support"])):
        ctx.add("FILTER_ALGEBRA", "run.filter_totals.cue_flip_count", "exceeds cue support")
    if int(filters["filter_recurrence_failure_count"].item()) != 0:
        ctx.add("FILTER_AUDIT", "run.filter_totals.filter_recurrence_failure_count", "must be zero")
    regret_difference = np.float32(filters["optimal_value_sum"] - filters["selected_value_sum"])
    if not bool(
        np.isclose(filters["selected_regret_sum"], regret_difference, rtol=2e-6, atol=2e-6)
    ):
        ctx.add(
            "FILTER_ALGEBRA",
            "run.filter_totals.selected_regret_sum",
            "must equal optimal minus selected",
        )
    if not _positive_zero(filters["tied_selected_regret_sum"].reshape((1,))):
        ctx.add(
            "FILTER_ALGEBRA",
            "run.filter_totals.tied_selected_regret_sum",
            "ties must have exact +0.0 regret",
        )

    focal = action["focal_action_support"]
    partner = action["partner_action_support"]
    joint = action["joint_row_support"]
    ordinary = action["ordinary_policy_action_support"]
    accepted = int(np.sum(focal))
    if support != accepted:
        ctx.add(
            "FILTER_ALGEBRA",
            "run.filter_totals.support",
            "must equal accepted action support",
        )
    for name in (
        "optimal_value_sum",
        "selected_value_sum",
        "selected_regret_sum",
        "margin_sum",
    ):
        if float(filters[name].item()) > float(support):
            ctx.add(
                "FILTER_ALGEBRA",
                f"run.filter_totals.{name}",
                "exceeds its exact support bound",
            )
    if (
        int(np.sum(partner)) != accepted
        or int(np.sum(joint)) != accepted
        or int(np.sum(ordinary)) != accepted
    ):
        ctx.add("ACTION_ALGEBRA", "run.action_totals", "all action support marginals must agree")
    _expect_array_equal(
        ctx,
        focal,
        np.asarray((joint[0] + joint[1], joint[2] + joint[3]), dtype=np.int32),
        path="run.action_totals.focal_action_support",
        code="ACTION_ALGEBRA",
    )
    _expect_array_equal(
        ctx,
        partner,
        np.asarray((joint[0] + joint[2], joint[1] + joint[3]), dtype=np.int32),
        path="run.action_totals.partner_action_support",
        code="ACTION_ALGEBRA",
    )
    if int(action["decision_count"].item()) != accepted + 1:
        ctx.add(
            "ACTION_ALGEBRA",
            "run.action_totals.decision_count",
            "must equal accepted steps plus one",
        )
    if int(action["explored_count"].item()) > accepted:
        ctx.add("ACTION_ALGEBRA", "run.action_totals.explored_count", "exceeds accepted support")
    for name in (
        "policy_schedule_failure_count",
        "policy_replay_failure_count",
        "rng_chain_failure_count",
    ):
        if int(action[name].item()) != 0:
            ctx.add("ACTION_AUDIT", f"run.action_totals.{name}", "must be zero")
    balanced = control is not None and control.focal_action_policy == "balanced_external"
    forced_expected = accepted if balanced else 0
    if int(action["externally_forced_count"].item()) != forced_expected:
        ctx.add(
            "ACTION_ALGEBRA",
            "run.action_totals.externally_forced_count",
            "differs from control policy",
        )
    if balanced:
        expected_focal = np.asarray(((accepted + 1) // 2, accepted // 2), dtype=np.int32)
        _expect_array_equal(
            ctx,
            focal,
            expected_focal,
            path="run.action_totals.focal_action_support",
            code="ACTION_ALGEBRA",
        )
    return bool(np.all(filters["cue_pattern_support"] > 0)), bool(focal[0] == focal[1])


def _validate_audits_and_counters(
    ctx: _ValidationContext,
    arrays: dict[str, np.ndarray],
    run: V6DevelopmentRun,
    plan: HiddenPartnerLifecycleWorldV6ScanPlan | None,
    control: HiddenPartnerLifecycleWorldV6Control | None,
) -> tuple[int | None, int | None]:
    if len(arrays) != len(_AUDIT_CONTRACT) or plan is None:
        return None, None
    for name, value in arrays.items():
        if value.dtype != np.dtype(np.bool_):
            _nonnegative(ctx, value, path=f"run.audits.{name}")
    run_steps = plan.run_steps
    active_steps = int(arrays["active_steps"].item())
    accepted_steps = int(arrays["accepted_steps"].item())
    if active_steps != run_steps:
        ctx.add("ACTIVE_PREFIX", "run.audits.active_steps", "must equal exact plan run_steps")
    if accepted_steps != active_steps:
        ctx.add(
            "ACTIVE_REJECTION",
            "run.audits.accepted_steps",
            "every active-prefix bridge transition must be accepted",
        )
    for name in (
        "learner_valid_steps",
        "filter_valid_steps",
        "oracle_valid_steps",
        "mechanism_valid_steps",
        "all_finite_steps",
    ):
        if int(arrays[name].item()) != active_steps:
            ctx.add("AUDIT_LANE", f"run.audits.{name}", "must equal active_steps")
    if bool(np.any(arrays["contract_failure_counts"] != 0)):
        ctx.add("CONTRACT_AUDIT", "run.audits.contract_failure_counts", "must be all zero")
    intervention_failures = arrays["intervention_failure_counts"]
    intervention_witnesses = arrays["intervention_witness_counts"]
    if bool(np.any(intervention_failures != 0)):
        ctx.add(
            "INTERVENTION_AUDIT",
            "run.audits.intervention_failure_counts",
            "must be all zero",
        )
    for name, values in (
        ("intervention_failure_counts", intervention_failures),
        ("intervention_witness_counts", intervention_witnesses),
    ):
        if bool(np.any(values > accepted_steps)):
            ctx.add(
                "INTERVENTION_COUNT",
                f"run.audits.{name}",
                "each per-step count must be bounded by accepted_steps",
            )
    if control is not None:
        required_witnesses = required_v6_intervention_witness_names(control)
        missing_witnesses = tuple(
            name
            for name in required_witnesses
            if int(intervention_witnesses[V6_INTERVENTION_WITNESS_ORDER.index(name)]) <= 0
        )
        if missing_witnesses:
            ctx.add(
                "INTERVENTION_WITNESS",
                "run.audits.intervention_witness_counts",
                "missing required positive support: " + ", ".join(missing_witnesses),
            )
    if bool(np.any(arrays["component_delta_failure_counts"] != 0)):
        ctx.add(
            "COMPONENT_AUDIT",
            "run.audits.component_delta_failure_counts",
            "must be all zero",
        )
    expected_cadence = run_steps // CURATION_INTERVAL
    if int(arrays["curation_attempt_count"].item()) != expected_cadence:
        ctx.add("CADENCE_COUNT", "run.audits.curation_attempt_count", "differs from plan")
    if int(arrays["ledger_count"].item()) != expected_cadence:
        ctx.add("CADENCE_COUNT", "run.audits.ledger_count", "differs from plan")
    if bool(arrays["ledger_overflow"].item()):
        ctx.add("LEDGER_OVERFLOW", "run.audits.ledger_overflow", "must remain false")

    if (
        type(run.initial_state) is not HiddenPartnerWorldOnlineState
        or type(run.final_state) is not HiddenPartnerWorldOnlineState
    ):
        return active_steps, accepted_steps
    initial_valid = _scalar_bool(ctx, run.initial_state.valid, path="run.initial_state.valid")
    final_valid = _scalar_bool(ctx, run.final_state.valid, path="run.final_state.valid")
    if initial_valid is not True:
        ctx.add("STATE_VALID", "run.initial_state.valid", "must start true")
    if final_valid is not True:
        ctx.add("STATE_LATCHED", "run.final_state.valid", "must remain true")

    component_paths = (
        "agent.state_builder.step_count",
        "agent.state_builder.update_count",
        "agent.behavior.step_count",
        "agent.interaction.step_count",
        "agent.joint_world.step_count",
        "agent.grounded_world.update_count",
        "agent.control.step_count",
        "agent.router.route_count",
        "agent.router.generation_count",
        "agent.step_count",
    )
    grounded_present = bool(
        control is not None
        and control.agent_config is not None
        and control.agent_config.grounded_world_model is not None
    )
    grounded_learning = bool(
        grounded_present
        and control is not None
        and control.agent_config is not None
        and control.agent_config.grounded_world_learning_enabled
    )
    state_learning = bool(
        control is not None
        and control.agent_config is not None
        and control.agent_config.state_learning_enabled
    )
    expected_static = np.asarray(
        (
            accepted_steps,
            accepted_steps * int(state_learning),
            accepted_steps,
            accepted_steps,
            accepted_steps,
            accepted_steps * int(grounded_learning),
            accepted_steps,
            accepted_steps,
            -1,
            accepted_steps,
        ),
        dtype=np.int32,
    )
    endpoint_deltas: list[int] = []
    for index, dotted in enumerate(component_paths):
        if dotted == "agent.grounded_world.update_count" and not grounded_present:
            endpoint_deltas.append(0)
            continue
        initial_value = _state_counter(ctx, run.initial_state, dotted, endpoint="initial")
        final_value = _state_counter(ctx, run.final_state, dotted, endpoint="final")
        if initial_value is None or final_value is None:
            endpoint_deltas.append(0)
            continue
        delta = final_value - initial_value
        endpoint_deltas.append(delta)
        if expected_static[index] >= 0 and delta != int(expected_static[index]):
            ctx.add("COUNTER_DELTA", f"run.final_state.{dotted}", "has a noncanonical delta")
    delta_array = np.asarray(endpoint_deltas, dtype=np.int32)
    _expect_array_equal(
        ctx,
        arrays["component_delta_sums"],
        delta_array,
        path="run.audits.component_delta_sums",
        code="COMPONENT_ALGEBRA",
    )
    for dotted in ("step_count", "world.step_count", "world_filter.step_count"):
        initial_value = _state_counter(ctx, run.initial_state, dotted, endpoint="initial")
        final_value = _state_counter(ctx, run.final_state, dotted, endpoint="final")
        if initial_value is not None and final_value is not None:
            if final_value - initial_value != accepted_steps:
                ctx.add(
                    "COUNTER_DELTA", f"run.final_state.{dotted}", "must advance per accepted step"
                )
    return active_steps, accepted_steps


def _candidate_pair(index: int) -> tuple[int, int] | None:
    if index < 0 or index >= CANDIDATE_PAIR_SLOTS:
        return None
    cursor = 0
    for left in range(BASE_FEATURE_DIM):
        for right in range(left + 1, BASE_FEATURE_DIM):
            if cursor == index:
                return left, right
            cursor += 1
    return None


def _pair_candidate(pair: tuple[int, int]) -> int | None:
    left, right = pair
    if not (0 <= left < right < BASE_FEATURE_DIM):
        return None
    return left * (2 * BASE_FEATURE_DIM - left - 1) // 2 + right - left - 1


def _live_mask(bank: np.ndarray) -> np.ndarray:
    return (bank[:, 0] >= 0) & (bank[:, 0] < bank[:, 1]) & (bank[:, 1] < BASE_FEATURE_DIM)


def _bank_valid(bank: np.ndarray) -> bool:
    inactive = np.all(bank == -1, axis=1)
    live = _live_mask(bank)
    if not bool(np.all(inactive | live)):
        return False
    identities = [tuple(map(int, row)) for row in bank[live]]
    return len(identities) == len(set(identities))


def _critical_slots_host(bank: np.ndarray) -> np.ndarray:
    slots: list[int] = []
    for pair in CRITICAL_PAIRS:
        matches = np.all(bank == np.asarray(pair, dtype=np.int32), axis=1)
        indices = np.flatnonzero(matches)
        slots.append(int(indices[0]) if len(indices) == 1 else -1)
    return np.asarray(slots, dtype=np.int32)


def _route_mask(values: np.ndarray, sources: np.ndarray, survivor: np.ndarray) -> np.ndarray:
    routed = np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.bool_)
    if bool(np.any(survivor)):
        routed[survivor] = values[sources[survivor]]
    return routed


def _occurrence_at(
    plan: HiddenPartnerLifecycleWorldV6ScanPlan,
    step: int,
) -> tuple[int, int] | None:
    for occurrence in plan.segment_occurrences:
        if occurrence.start <= step < occurrence.end_exclusive:
            return occurrence.occurrence_index, occurrence.regime_id
    return None


def _validate_event_row(
    ctx: _ValidationContext,
    *,
    index: int,
    prefix: str,
    pre: np.ndarray,
    proposed: np.ndarray,
    event: np.ndarray,
    reset: np.ndarray,
) -> None:
    replaced, promoted, refreshed, retired_slot, retired_left, retired_right = map(int, event)
    for value, name, limit in (
        (replaced, "replaced_slot", ACTIVE_PAIR_SLOTS),
        (retired_slot, "retired_slot", ACTIVE_PAIR_SLOTS),
        (promoted, "promoted_candidate", CANDIDATE_PAIR_SLOTS),
        (refreshed, "refreshed_candidate", CANDIDATE_PAIR_SLOTS),
    ):
        if value < -1 or value >= limit:
            ctx.add("EVENT_DOMAIN", f"{prefix}.{name}[{index}]", "is outside its exact domain")
    promotion = replaced >= 0 or promoted >= 0
    retirement = retired_slot >= 0 or retired_left >= 0 or retired_right >= 0
    if (replaced >= 0) != (promoted >= 0):
        ctx.add("EVENT_ALGEBRA", f"{prefix}[{index}]", "promotion slot and candidate must co-occur")
    pair_present = retired_left >= 0 or retired_right >= 0
    if (retired_slot >= 0) != pair_present:
        ctx.add("EVENT_ALGEBRA", f"{prefix}[{index}]", "retirement slot and pair must co-occur")
    if pair_present and not (0 <= retired_left < retired_right < BASE_FEATURE_DIM):
        ctx.add("EVENT_DOMAIN", f"{prefix}[{index}]", "retired pair is invalid")
    if promotion and retirement:
        ctx.add("EVENT_ALGEBRA", f"{prefix}[{index}]", "promotion and retirement are exclusive")
    expected = pre.copy()
    if replaced >= 0 and promoted >= 0:
        pair = _candidate_pair(promoted)
        if pair is not None:
            expected[replaced] = np.asarray(pair, dtype=np.int32)
    elif retired_slot >= 0:
        if tuple(map(int, pre[retired_slot])) != (retired_left, retired_right):
            ctx.add("EVENT_ALGEBRA", f"{prefix}[{index}]", "retired pair differs from pre bank")
        expected[retired_slot] = np.asarray((-1, -1), dtype=np.int32)
    _expect_array_equal(
        ctx,
        proposed,
        expected,
        path=f"{prefix}.descriptors[{index}]",
        code="EVENT_ALGEBRA",
    )
    expected_reset = np.zeros((CANDIDATE_PAIR_SLOTS,), dtype=np.bool_)
    if retired_slot >= 0:
        candidate = _pair_candidate((retired_left, retired_right))
        if candidate is not None:
            expected_reset[candidate] = True
    _expect_array_equal(
        ctx,
        reset,
        expected_reset,
        path=f"{prefix}.candidate_reset_mask[{index}]",
        code="EVENT_ALGEBRA",
    )


def _validate_ledger(
    ctx: _ValidationContext,
    arrays: dict[str, np.ndarray],
    run: V6DevelopmentRun,
    plan: HiddenPartnerLifecycleWorldV6ScanPlan | None,
    control: HiddenPartnerLifecycleWorldV6Control | None,
) -> bool:
    if len(arrays) != len(_LEDGER_CONTRACT) or plan is None:
        return False
    count = plan.run_steps // CURATION_INTERVAL
    expected_occupied = np.arange(MAX_CADENCE_LEDGER_ENTRIES) < count
    _expect_array_equal(
        ctx,
        arrays["occupied"],
        expected_occupied,
        path="run.ledger.occupied",
        code="LEDGER_PREFIX",
    )
    for name, value in arrays.items():
        suffix = value[count:]
        if value.dtype == np.dtype(np.int32):
            if not bool(np.all(suffix == -1)):
                ctx.add("PADDING_SENTINEL", f"run.ledger.{name}", "int suffix must be exact -1")
        elif value.dtype == np.dtype(np.bool_):
            if bool(np.any(suffix)):
                ctx.add("PADDING_SENTINEL", f"run.ledger.{name}", "bool suffix must be false")
        elif value.dtype == np.dtype(np.float32):
            if not _positive_zero(suffix):
                ctx.add("PADDING_SENTINEL", f"run.ledger.{name}", "float suffix must be +0.0")
    if count == 0:
        return False
    active = slice(0, count)
    for name in ("random_active_priorities", "random_candidate_priorities"):
        _all_finite(ctx, arrays[name][active], path=f"run.ledger.{name}[:ledger_count]")
    if not bool(np.all(arrays["critical_candidate_streak"][active] >= 0)):
        ctx.add(
            "DOMAIN", "run.ledger.critical_candidate_streak", "occupied streaks must be nonnegative"
        )
    if not bool(np.all(arrays["router_counts"][active] >= 0)):
        ctx.add("DOMAIN", "run.ledger.router_counts", "occupied counts must be nonnegative")
    expected_steps = CURATION_INTERVAL * np.arange(1, count + 1, dtype=np.int32)
    _expect_array_equal(
        ctx,
        arrays["transition_step"][:count],
        expected_steps,
        path="run.ledger.transition_step[:ledger_count]",
        code="CADENCE_ALGEBRA",
    )

    initial_bank = np.asarray(jax.device_get(run.initial_state.agent.router.descriptors))
    final_bank = np.asarray(jax.device_get(run.final_state.agent.router.descriptors))
    if not bool(np.array_equal(arrays["pre_descriptors"][0], initial_bank)):
        ctx.add("LEDGER_CHAIN", "run.ledger.pre_descriptors[0]", "differs from initial router bank")
    if count > 1 and not bool(
        np.array_equal(
            arrays["pre_descriptors"][1:count], arrays["applied_descriptors"][: count - 1]
        )
    ):
        ctx.add(
            "LEDGER_CHAIN", "run.ledger.pre_descriptors", "does not chain from prior applied bank"
        )
    if not bool(np.array_equal(arrays["applied_descriptors"][count - 1], final_bank)):
        ctx.add(
            "LEDGER_CHAIN", "run.ledger.applied_descriptors", "last bank differs from final router"
        )

    config = None if control is None else control.agent_config
    lifecycle_enabled = bool(config is not None and config.feature_lifecycle_enabled)
    carry_configured = bool(config is not None and config.carry_survivors)
    random_enabled = bool(config is not None and config.random_feature_curation)
    initial_route = int(
        np.asarray(jax.device_get(run.initial_state.agent.router.route_count)).item()
    )
    initial_generation = int(
        np.asarray(jax.device_get(run.initial_state.agent.router.generation_count)).item()
    )
    cumulative_generation = initial_generation
    for index in range(count):
        row_path = f"run.ledger[{index}]"
        event_step = int(arrays["transition_step"][index]) - 1
        occurrence = _occurrence_at(plan, event_step)
        if occurrence is None:
            ctx.add("CADENCE_ALGEBRA", f"{row_path}.transition_step", "is outside plan")
        else:
            _, expected_regime = occurrence
            if int(arrays["regime_id"][index]) != expected_regime:
                ctx.add("CADENCE_ALGEBRA", f"{row_path}.regime_id", "differs from plan")
        stages = (
            arrays["pre_descriptors"][index],
            arrays["proposal_descriptors"][index],
            arrays["applied_descriptors"][index],
        )
        for stage_name, bank in zip(V6_CRITICAL_STAGE_ORDER, stages, strict=True):
            if not _bank_valid(bank):
                ctx.add("DESCRIPTOR_BANK", f"{row_path}.{stage_name}_descriptors", "is invalid")
        expected_critical = np.stack(tuple(_critical_slots_host(bank) for bank in stages))
        _expect_array_equal(
            ctx,
            arrays["critical_slot"][index],
            expected_critical,
            path=f"{row_path}.critical_slot",
            code="DESCRIPTOR_ALGEBRA",
        )

        proposal_event = arrays["proposal_event"][index]
        applied_event = arrays["applied_event"][index]
        proposal_reset = arrays["candidate_reset_mask"][index, 0]
        applied_reset = arrays["candidate_reset_mask"][index, 1]
        _validate_event_row(
            ctx,
            index=index,
            prefix="run.ledger.proposal_event",
            pre=stages[0],
            proposed=stages[1],
            event=proposal_event,
            reset=proposal_reset,
        )
        if lifecycle_enabled:
            _expect_array_equal(
                ctx,
                applied_event,
                proposal_event,
                path=f"{row_path}.applied_event",
                code="EVENT_ALGEBRA",
            )
            _expect_array_equal(
                ctx,
                stages[2],
                stages[1],
                path=f"{row_path}.applied_descriptors",
                code="EVENT_ALGEBRA",
            )
            _expect_array_equal(
                ctx,
                applied_reset,
                proposal_reset,
                path=f"{row_path}.candidate_reset_mask[1]",
                code="EVENT_ALGEBRA",
            )
        else:
            _expect_array_equal(
                ctx,
                applied_event,
                np.full((6,), -1, dtype=np.int32),
                path=f"{row_path}.applied_event",
                code="FROZEN_LIFECYCLE",
            )
            _expect_array_equal(
                ctx,
                stages[2],
                stages[0],
                path=f"{row_path}.applied_descriptors",
                code="FROZEN_LIFECYCLE",
            )
            _expect_array_equal(
                ctx,
                applied_reset,
                np.zeros((CANDIDATE_PAIR_SLOTS,), dtype=np.bool_),
                path=f"{row_path}.candidate_reset_mask[1]",
                code="FROZEN_LIFECYCLE",
            )

        flags = arrays["critical_candidate_flags"][index]
        if bool(np.any(flags[:, 5] & ~(flags[:, 0] & flags[:, 1] & flags[:, 2]))):
            ctx.add(
                "CANDIDATE_ALGEBRA",
                f"{row_path}.critical_candidate_flags",
                "invalid reacquisition confirmation",
            )
        for critical_index, candidate_index in enumerate(CRITICAL_CANDIDATE_INDICES):
            if bool(applied_reset[candidate_index]):
                if int(arrays["critical_candidate_streak"][index, critical_index, 1]) != 0:
                    ctx.add(
                        "CANDIDATE_ALGEBRA",
                        f"{row_path}.critical_candidate_streak",
                        "reset streak must be zero",
                    )
                if not bool(flags[critical_index, 4]):
                    ctx.add(
                        "CANDIDATE_ALGEBRA",
                        f"{row_path}.critical_candidate_flags",
                        "reset must require reacquisition",
                    )

        pre, applied = stages[0], stages[2]
        pre_live = _live_mask(pre)
        applied_live = _live_mask(applied)
        identity = np.all(applied[:, None, :] == pre[None, :, :], axis=2)
        identity &= applied_live[:, None] & pre_live[None, :]
        survivor = applied_live & np.any(identity, axis=1)
        new = applied_live & ~survivor
        evicted = pre_live & ~np.any(identity, axis=0)
        sources = np.where(survivor, np.argmax(identity, axis=1), -1).astype(np.int32)
        expected_masks = np.stack((survivor, new, evicted))
        _expect_array_equal(
            ctx,
            arrays["router_source_slots"][index],
            sources,
            path=f"{row_path}.router_source_slots",
            code="ROUTER_ALGEBRA",
        )
        _expect_array_equal(
            ctx,
            arrays["router_masks"][index],
            expected_masks,
            path=f"{row_path}.router_masks",
            code="ROUTER_ALGEBRA",
        )
        changed = bool(np.any(pre != applied))
        expected_router_flags = np.asarray(
            (True, True, carry_configured or not changed, changed),
            dtype=np.bool_,
        )
        _expect_array_equal(
            ctx,
            arrays["router_flags"][index],
            expected_router_flags,
            path=f"{row_path}.router_flags",
            code="ROUTER_ALGEBRA",
        )
        before_route = initial_route + event_step
        after_route = before_route + 1
        before_generation = cumulative_generation
        cumulative_generation += int(changed)
        expected_counts = np.asarray(
            (before_route, after_route, before_generation, cumulative_generation),
            dtype=np.int32,
        )
        _expect_array_equal(
            ctx,
            arrays["router_counts"][index],
            expected_counts,
            path=f"{row_path}.router_counts",
            code="ROUTER_ALGEBRA",
        )
        for name in ("transaction_exact", "identity_carry_exact", "retired_identity_reset_exact"):
            if not bool(arrays[name][index]):
                ctx.add("ROUTER_AUDIT", f"{row_path}.{name}", "must be true")

        consumer = arrays["consumer_masks"][index]
        durable, acquire_pre, acquire_post, confirmed_pre, confirmed_post = consumer[:5]
        read_pre, read_post, active_pre, active_post = consumer[5:]
        if not bool(np.array_equal(read_pre, active_pre)) or not bool(
            np.array_equal(read_post, active_post)
        ):
            ctx.add(
                "CONSUMER_ALGEBRA", f"{row_path}.consumer_masks", "read and active masks differ"
            )
        if bool(np.any(durable & ~(active_pre & pre_live))):
            ctx.add(
                "CONSUMER_ALGEBRA",
                f"{row_path}.consumer_masks.durable",
                "is not a live active subset",
            )
        if bool(np.any(confirmed_pre & ~acquire_pre)) or bool(
            np.any(confirmed_post & ~acquire_post)
        ):
            ctx.add(
                "CONSUMER_ALGEBRA",
                f"{row_path}.consumer_masks.confirmed",
                "is not an acquire subset",
            )
        expected_acquire_post = _route_mask(acquire_pre, sources, survivor)
        expected_confirmed_post = _route_mask(confirmed_pre, sources, survivor)
        _expect_array_equal(
            ctx,
            acquire_post,
            expected_acquire_post,
            path=f"{row_path}.consumer_masks.acquire_post",
            code="CONSUMER_ALGEBRA",
        )
        _expect_array_equal(
            ctx,
            confirmed_post,
            expected_confirmed_post,
            path=f"{row_path}.consumer_masks.confirmed_post",
            code="CONSUMER_ALGEBRA",
        )
        possible_active_post = (
            _route_mask(active_pre | acquire_pre, sources, survivor) & applied_live
        )
        if bool(np.any(active_post & ~possible_active_post)):
            ctx.add(
                "CONSUMER_ALGEBRA",
                f"{row_path}.consumer_masks.active_post",
                "cannot arise by identity routing",
            )
        if bool(np.any(acquire_post & ~active_post)):
            ctx.add(
                "CONSUMER_ALGEBRA",
                f"{row_path}.consumer_masks.acquire_post",
                "must be active post-route",
            )

        expected_random_flags = np.asarray((random_enabled, True, random_enabled), dtype=np.bool_)
        _expect_array_equal(
            ctx,
            arrays["random_curation_flags"][index],
            expected_random_flags,
            path=f"{row_path}.random_curation_flags",
            code="RANDOM_CURATION_ALGEBRA",
        )
        active_priorities = arrays["random_active_priorities"][index]
        candidate_priorities = arrays["random_candidate_priorities"][index]
        if not bool(
            np.array_equal(
                np.sort(active_priorities), np.arange(ACTIVE_PAIR_SLOTS, dtype=np.float32)
            )
        ):
            ctx.add(
                "RANDOM_CURATION_ALGEBRA",
                f"{row_path}.random_active_priorities",
                "must be an exact permutation",
            )
        if not bool(
            np.array_equal(
                np.sort(candidate_priorities), np.arange(CANDIDATE_PAIR_SLOTS, dtype=np.float32)
            )
        ):
            ctx.add(
                "RANDOM_CURATION_ALGEBRA",
                f"{row_path}.random_candidate_priorities",
                "must be an exact permutation",
            )
        selected_active, selected_promotion, selected_refresh = map(
            int,
            arrays["random_curation_selected"][index],
        )
        if (
            not (-1 <= selected_active < ACTIVE_PAIR_SLOTS)
            or not (-1 <= selected_promotion < CANDIDATE_PAIR_SLOTS)
            or not (0 <= selected_refresh < CANDIDATE_PAIR_SLOTS)
        ):
            ctx.add(
                "RANDOM_CURATION_ALGEBRA",
                f"{row_path}.random_curation_selected",
                "is outside its domain",
            )
        if random_enabled and selected_refresh != int(np.argmin(candidate_priorities)):
            ctx.add(
                "RANDOM_CURATION_ALGEBRA",
                f"{row_path}.random_curation_selected.refresh",
                "must follow random priority",
            )

    final_route = int(np.asarray(jax.device_get(run.final_state.agent.router.route_count)).item())
    final_generation = int(
        np.asarray(jax.device_get(run.final_state.agent.router.generation_count)).item()
    )
    if final_route != initial_route + plan.run_steps:
        ctx.add(
            "ROUTER_ENDPOINT",
            "run.final_state.agent.router.route_count",
            "differs from active scan",
        )
    if final_generation != cumulative_generation:
        ctx.add(
            "ROUTER_ENDPOINT",
            "run.final_state.agent.router.generation_count",
            "differs from ledger changes",
        )
    return bool(np.all(arrays["occupied"][:count]))


def _contains_pair(bank: np.ndarray, pair: tuple[int, int]) -> bool:
    return bool(np.any(np.all(bank == np.asarray(pair, dtype=np.int32), axis=1)))


def _recompute_lifecycle(
    ledger: dict[str, np.ndarray],
    plan: HiddenPartnerLifecycleWorldV6ScanPlan | None,
) -> V6LifecycleOutcome:
    if plan is None or len(ledger) != len(_LEDGER_CONTRACT):
        return _NEGATIVE_LIFECYCLE
    count = plan.run_steps // CURATION_INTERVAL
    structural = True
    c_first_step = -1
    c_ever = False
    c_retained = True
    c_first_occurrence = False
    d_phase = 0
    d_first_step = -1
    d_retirement_step = -1
    d_reacquisition_step = -1
    d_first_in_first = False
    d_retired_absent = False
    d_reset = False
    d_reacquired_second = False
    d_gate = False
    out_of_order = 0
    for index in range(count):
        transition_step = int(ledger["transition_step"][index])
        occurrence = _occurrence_at(plan, transition_step - 1)
        if occurrence is None:
            structural = False
            occurrence_index = -1
            regime_id = -1
        else:
            occurrence_index, regime_id = occurrence
        pre = ledger["pre_descriptors"][index]
        post = ledger["applied_descriptors"][index]
        structural &= (
            _bank_valid(pre)
            and _bank_valid(post)
            and bool(ledger["transaction_exact"][index])
            and bool(ledger["identity_carry_exact"][index])
            and bool(ledger["retired_identity_reset_exact"][index])
        )
        c_pre = _contains_pair(pre, CRITICAL_PAIRS[0])
        c_post = _contains_pair(post, CRITICAL_PAIRS[0])
        c_acquired = not c_pre and c_post
        c_lost = c_pre and not c_post
        first_c = c_acquired and not c_ever
        if first_c:
            c_first_step = transition_step
        c_ever = c_ever or c_acquired
        c_retained = c_retained and not (c_ever and not first_c and c_lost)
        c_first_occurrence = c_first_occurrence or (first_c and occurrence_index == 5)

        d_pre = _contains_pair(pre, CRITICAL_PAIRS[1])
        d_post = _contains_pair(post, CRITICAL_PAIRS[1])
        d_acquired = not d_pre and d_post
        d_disappeared = d_pre and not d_post
        applied_event = ledger["applied_event"][index]
        retired_d = tuple(map(int, applied_event[4:6])) == CRITICAL_PAIRS[1]
        retirement_reset_exact = retired_d and bool(ledger["retired_identity_reset_exact"][index])
        exact_d_retirement = d_disappeared and retirement_reset_exact
        first_acquisition = d_phase == 0 and d_acquired
        ordered_retirement = d_phase == 1 and exact_d_retirement
        ordered_reacquisition = d_phase == 2 and d_acquired
        event_out_of_order = (
            (d_phase == 0 and d_disappeared)
            or (d_phase == 1 and (d_acquired or (d_disappeared and not exact_d_retirement)))
            or (d_phase == 2 and d_disappeared)
            or (d_phase == 3 and (d_acquired or d_disappeared))
        )
        out_of_order += int(event_out_of_order)
        promoted = int(ledger["applied_event"][index, 1])
        d_flags = ledger["critical_candidate_flags"][index, 1]
        if first_acquisition:
            d_phase = 1
            d_first_step = transition_step
            d_first_in_first = d_first_in_first or (
                occurrence_index == 3 and promoted == CRITICAL_CANDIDATE_INDICES[1]
            )
        elif ordered_retirement:
            d_phase = 2
            d_retirement_step = transition_step
            d_retired_absent = d_retired_absent or regime_id != 3
            d_reset = d_reset or retirement_reset_exact
        elif ordered_reacquisition:
            d_phase = 3
            d_reacquisition_step = transition_step
            d_reacquired_second = d_reacquired_second or (
                occurrence_index == 12 and promoted == CRITICAL_CANDIDATE_INDICES[1]
            )
            d_gate = d_gate or (bool(d_flags[2]) and bool(d_flags[5]))
    d_ordered = (
        d_phase == 3
        and d_first_in_first
        and d_retired_absent
        and d_reset
        and d_reacquired_second
        and d_gate
    )
    return V6LifecycleOutcome(
        structural_chain_consistent=structural,
        c_first_acquisition_step=c_first_step,
        c_ever_acquired=c_ever,
        c_retained_after_acquisition=c_retained,
        c_acquired_in_first_c=c_first_occurrence,
        d_phase=d_phase,
        d_first_acquisition_step=d_first_step,
        d_retirement_step=d_retirement_step,
        d_reacquisition_step=d_reacquisition_step,
        d_first_acquisition_in_first_d=d_first_in_first,
        d_retirement_while_absent=d_retired_absent,
        d_retirement_reset_exact=d_reset,
        d_reacquisition_in_second_d=d_reacquired_second,
        d_reacquisition_gate_observed=d_gate,
        d_ordered_outcome=d_ordered,
        out_of_order_event_count=out_of_order,
    )


def _validate_lifecycle_summary(
    ctx: _ValidationContext,
    arrays: dict[str, np.ndarray],
    outcome: V6LifecycleOutcome,
) -> None:
    if len(arrays) != len(_LIFECYCLE_CONTRACT):
        return
    expected: dict[str, int | bool] = {
        "structural_valid": outcome.structural_chain_consistent,
        "c_first_acquisition_step": outcome.c_first_acquisition_step,
        "c_ever_acquired": outcome.c_ever_acquired,
        "c_retained_after_acquisition": outcome.c_retained_after_acquisition,
        "c_acquired_in_first_c": outcome.c_acquired_in_first_c,
        "d_phase": outcome.d_phase,
        "d_first_acquisition_step": outcome.d_first_acquisition_step,
        "d_retirement_step": outcome.d_retirement_step,
        "d_reacquisition_step": outcome.d_reacquisition_step,
        "d_first_acquisition_in_first_d": outcome.d_first_acquisition_in_first_d,
        "d_retirement_while_absent": outcome.d_retirement_while_absent,
        "d_retirement_reset_exact": outcome.d_retirement_reset_exact,
        "d_reacquisition_in_second_d": outcome.d_reacquisition_in_second_d,
        "d_reacquisition_gate_observed": outcome.d_reacquisition_gate_observed,
        "d_ordered_outcome": outcome.d_ordered_outcome,
        "out_of_order_event_count": outcome.out_of_order_event_count,
    }
    for name, exact in expected.items():
        actual = arrays[name].item()
        if actual != exact:
            ctx.add(
                "LIFECYCLE_RECOMPUTATION",
                f"run.lifecycle.{name}",
                "differs from occupied-ledger reconstruction",
            )
    if outcome.d_phase not in (0, 1, 2, 3):
        ctx.add("LIFECYCLE_DOMAIN", "run.lifecycle.d_phase", "must be in 0..3")
    if outcome.out_of_order_event_count < 0:
        ctx.add("LIFECYCLE_DOMAIN", "run.lifecycle.out_of_order_event_count", "must be nonnegative")


def _budget_values(
    ctx: _ValidationContext,
    budget: object,
    *,
    path: str,
) -> dict[str, int]:
    if not _exact_class(ctx, budget, HiddenPartnerWorldOnlineResourceBudget, path=path):
        return {}
    values: dict[str, int] = {}
    for field in dataclasses.fields(HiddenPartnerWorldOnlineResourceBudget):
        value = _host_int(ctx, getattr(budget, field.name), path=f"{path}.{field.name}")
        if value is not None:
            values[field.name] = value
            if value < 0:
                ctx.add("DOMAIN", f"{path}.{field.name}", "must be nonnegative")
    if len(values) != len(dataclasses.fields(HiddenPartnerWorldOnlineResourceBudget)):
        return values
    if values["component_state_nbytes"] != (
        values["world_state_nbytes"] + values["agent_state_nbytes"] + values["filter_state_nbytes"]
    ):
        ctx.add("RESOURCE_ALGEBRA", path, "component bytes do not sum")
    if values["bridge_metadata_nbytes"] != (
        values["config_token_nbytes"]
        + values["action_nbytes"]
        + values["valid_nbytes"]
        + values["step_count_nbytes"]
    ):
        ctx.add("RESOURCE_ALGEBRA", path, "metadata bytes do not sum")
    if values["total_state_nbytes"] != (
        values["component_state_nbytes"] + values["bridge_metadata_nbytes"]
    ):
        ctx.add("RESOURCE_ALGEBRA", path, "total bytes do not sum")
    if values["replay_capacity"] != (
        values["world_replay_capacity"] + values["agent_replay_capacity"]
    ):
        ctx.add("RESOURCE_ALGEBRA", path, "replay capacities do not sum")
    return values


def _expected_budget_from_state(state: HiddenPartnerWorldOnlineState) -> dict[str, int] | None:
    world = _tree_nbytes(state.world)
    agent = _tree_nbytes(state.agent)
    world_filter = _tree_nbytes(state.world_filter)
    token = _tree_nbytes(state.config_token)
    action = _tree_nbytes(state.action)
    valid = _tree_nbytes(state.valid)
    step = _tree_nbytes(state.step_count)
    if None in (world, agent, world_filter, token, action, valid, step):
        return None
    component = cast(int, world) + cast(int, agent) + cast(int, world_filter)
    metadata = cast(int, token) + cast(int, action) + cast(int, valid) + cast(int, step)
    return {
        "world_state_nbytes": cast(int, world),
        "agent_state_nbytes": cast(int, agent),
        "filter_state_nbytes": cast(int, world_filter),
        "component_state_nbytes": component,
        "config_token_nbytes": cast(int, token),
        "action_nbytes": cast(int, action),
        "valid_nbytes": cast(int, valid),
        "step_count_nbytes": cast(int, step),
        "bridge_metadata_nbytes": metadata,
        "total_state_nbytes": component + metadata,
    }


def _validate_resources(ctx: _ValidationContext, run: V6DevelopmentRun) -> None:
    if not _exact_class(ctx, run.resources, V6ResourceRecord, path="run.resources"):
        return
    initial = _budget_values(ctx, run.resources.initial, path="run.resources.initial")
    final = _budget_values(ctx, run.resources.final, path="run.resources.final")
    for endpoint, state, values in (
        ("initial", run.initial_state, initial),
        ("final", run.final_state, final),
    ):
        if type(state) is not HiddenPartnerWorldOnlineState:
            continue
        expected = _expected_budget_from_state(state)
        if expected is None:
            ctx.add(
                "RESOURCE_ALGEBRA",
                f"run.resources.{endpoint}",
                "state leaves are not concrete arrays",
            )
            continue
        for name, exact in expected.items():
            if values.get(name) != exact:
                ctx.add(
                    "RESOURCE_ALGEBRA",
                    f"run.resources.{endpoint}.{name}",
                    "differs from exact state tree",
                )
    if initial and final and initial != final:
        ctx.add("RESOURCE_STATIC", "run.resources", "endpoint budgets must be identical")
    peak = _host_int(
        ctx, run.resources.peak_total_state_nbytes, path="run.resources.peak_total_state_nbytes"
    )
    static = _host_bool(
        ctx, run.resources.static_total_state_nbytes, path="run.resources.static_total_state_nbytes"
    )
    zero_replay = _host_bool(ctx, run.resources.zero_replay, path="run.resources.zero_replay")
    tree_structure_equal = _host_bool(
        ctx, run.resources.tree_structure_equal, path="run.resources.tree_structure_equal"
    )
    tree_signature_equal = _host_bool(
        ctx, run.resources.tree_signature_equal, path="run.resources.tree_signature_equal"
    )
    if (
        initial
        and final
        and peak != max(initial["total_state_nbytes"], final["total_state_nbytes"])
    ):
        ctx.add(
            "RESOURCE_ALGEBRA",
            "run.resources.peak_total_state_nbytes",
            "must equal endpoint maximum",
        )
    if static is not True or tree_structure_equal is not True or tree_signature_equal is not True:
        ctx.add("RESOURCE_STATIC", "run.resources", "all static resource flags must be true")
    expected_zero_replay = bool(
        initial and final and initial["replay_capacity"] == 0 and final["replay_capacity"] == 0
    )
    if zero_replay != expected_zero_replay or zero_replay is not True:
        ctx.add("RESOURCE_REPLAY", "run.resources.zero_replay", "must exactly report zero replay")
    initial_signature = _tree_signature(run.initial_state)
    final_signature = _tree_signature(run.final_state)
    for signature_name, stored_signature, exact_signature in (
        ("initial_tree_signature", run.resources.initial_tree_signature, initial_signature),
        ("final_tree_signature", run.resources.final_tree_signature, final_signature),
    ):
        if type(stored_signature) is not tuple:
            ctx.add("TYPE", f"run.resources.{signature_name}", "must be an exact tuple")
        elif exact_signature is None or stored_signature != exact_signature:
            ctx.add(
                "RESOURCE_SIGNATURE",
                f"run.resources.{signature_name}",
                "differs from exact tree",
            )
    initial_structure = str(jax.tree_util.tree_structure(run.initial_state))
    final_structure = str(jax.tree_util.tree_structure(run.final_state))
    if initial_structure != final_structure:
        ctx.add(
            "RESOURCE_STRUCTURE", "run.resources.tree_structure_equal", "state structures differ"
        )


def _advance_split_root(keys: Array, steps: int, *, width: int) -> Array:
    def body(_: int, current: Array) -> Array:
        return jax.vmap(lambda key: jr.split(key, width)[0])(current)

    return cast(Array, jax.lax.fori_loop(0, steps, body, keys))


def _validate_rng_and_stream(
    ctx: _ValidationContext,
    rng: dict[str, np.ndarray],
    stream: np.ndarray | None,
    run: V6DevelopmentRun,
    plan: HiddenPartnerLifecycleWorldV6ScanPlan | None,
    accepted_steps: int | None,
    action: dict[str, np.ndarray],
    control: HiddenPartnerLifecycleWorldV6Control | None,
) -> None:
    if len(rng) != len(_RNG_CONTRACT):
        return
    if (
        type(run.initial_state) is not HiddenPartnerWorldOnlineState
        or type(run.final_state) is not HiddenPartnerWorldOnlineState
    ):
        return
    initial_world_keys = (
        run.initial_state.world.signal_key,
        run.initial_state.world.partner_key,
        run.initial_state.world.world_key,
        run.initial_state.world.cue_key,
        run.initial_state.world.outcome_key,
    )
    final_world_keys = (
        run.final_state.world.signal_key,
        run.final_state.world.partner_key,
        run.final_state.world.world_key,
        run.final_state.world.cue_key,
        run.final_state.world.outcome_key,
    )
    for field_name, world_key_tuple in (
        ("initial_world_key_data", initial_world_keys),
        ("final_world_key_data", final_world_keys),
    ):
        expected_rows = tuple(_key_data(key) for key in world_key_tuple)
        if any(row is None for row in expected_rows):
            ctx.add("RNG_KEY", f"run.rng.{field_name}", "state contains an invalid typed key")
        else:
            expected = np.stack(cast(tuple[np.ndarray, ...], expected_rows))
            _expect_array_equal(
                ctx, rng[field_name], expected, path=f"run.rng.{field_name}", code="RNG_ENDPOINT"
            )
    policy_keys = {
        "initial_policy_key_data": (
            run.initial_state.agent.current_selection.rng_key_before,
            run.initial_state.agent.current_selection.rng_key_after,
        ),
        "final_policy_key_data": (
            run.final_state.agent.current_selection.rng_key_before,
            run.final_state.agent.current_selection.rng_key_after,
        ),
    }
    for field_name, policy_key_tuple in policy_keys.items():
        rows = tuple(_key_data(key) for key in policy_key_tuple)
        if any(row is None for row in rows):
            ctx.add("RNG_KEY", f"run.rng.{field_name}", "state contains an invalid typed key")
        else:
            expected = np.stack(cast(tuple[np.ndarray, ...], rows))
            _expect_array_equal(
                ctx, rng[field_name], expected, path=f"run.rng.{field_name}", code="RNG_ENDPOINT"
            )
            before = jr.wrap_key_data(jnp.asarray(expected[0], dtype=jnp.uint32))
            expected_after = np.asarray(jax.device_get(jr.key_data(jr.split(before, 4)[0])))
            if not bool(np.array_equal(expected[1], expected_after)):
                ctx.add(
                    "RNG_CHAIN",
                    f"run.rng.{field_name}",
                    "policy before/after is not one split-4 root",
                )
    for field_name, key in (
        ("initial_interaction_key_data", run.initial_state.agent.interaction.key),
        ("final_interaction_key_data", run.final_state.agent.interaction.key),
    ):
        expected_interaction_endpoint = _key_data(key)
        if expected_interaction_endpoint is None:
            ctx.add("RNG_KEY", f"run.rng.{field_name}", "state contains an invalid typed key")
        else:
            _expect_array_equal(
                ctx,
                rng[field_name],
                expected_interaction_endpoint,
                path=f"run.rng.{field_name}",
                code="RNG_ENDPOINT",
            )

    initial = run.initial_state.world
    bits = np.concatenate(
        (
            np.asarray(jax.device_get(initial.current_signals > 0.0), dtype=np.uint8),
            np.asarray(
                [bool(np.asarray(jax.device_get(initial.world_sign > 0.0)))], dtype=np.uint8
            ),
            np.asarray(jax.device_get(initial.current_cues > 0.0), dtype=np.uint8),
            np.asarray(
                [bool(np.asarray(jax.device_get(initial.previous_outcome > 0.0)))], dtype=np.uint8
            ),
            np.asarray(
                [bool(np.asarray(jax.device_get(initial.has_partner_history)))], dtype=np.uint8
            ),
        )
    )
    initial_stream = np.sum(np.left_shift(bits, np.arange(8, dtype=np.uint8)), dtype=np.uint8)
    if rng["initial_stream_bits"].item() != initial_stream.item():
        ctx.add("RNG_STREAM", "run.rng.initial_stream_bits", "differs from initial world bits")
    if accepted_steps is not None:
        expected_draws = np.full((len(V6_WORLD_RNG_KEY_ORDER),), accepted_steps, dtype=np.int32)
        _expect_array_equal(
            ctx,
            rng["world_draw_counts"],
            expected_draws,
            path="run.rng.world_draw_counts",
            code="RNG_COUNT",
        )
        if int(rng["interaction_key_advance_count"].item()) != accepted_steps:
            ctx.add(
                "RNG_COUNT", "run.rng.interaction_key_advance_count", "must equal accepted steps"
            )
        decision_count = (
            None if "decision_count" not in action else int(action["decision_count"].item())
        )
        if int(rng["policy_decision_count"].item()) != decision_count:
            ctx.add("RNG_COUNT", "run.rng.policy_decision_count", "differs from action decisions")
        try:
            world_roots = jnp.stack(initial_world_keys)
            expected_world_final = _advance_split_root(world_roots, accepted_steps, width=2)
            expected_world_data = np.asarray(
                jax.device_get(jax.vmap(jr.key_data)(expected_world_final))
            )
            _expect_array_equal(
                ctx,
                rng["final_world_key_data"],
                expected_world_data,
                path="run.rng.final_world_key_data",
                code="RNG_CHAIN",
            )
            interaction_root = jnp.reshape(run.initial_state.agent.interaction.key, (1,))
            expected_interaction = _advance_split_root(interaction_root, accepted_steps, width=2)[0]
            expected_interaction_data = np.asarray(
                jax.device_get(jr.key_data(expected_interaction))
            )
            _expect_array_equal(
                ctx,
                rng["final_interaction_key_data"],
                expected_interaction_data,
                path="run.rng.final_interaction_key_data",
                code="RNG_CHAIN",
            )
            if accepted_steps > 0:
                policy_root = jnp.reshape(
                    run.initial_state.agent.current_selection.rng_key_after, (1,)
                )
                expected_policy_after = _advance_split_root(policy_root, accepted_steps, width=4)[0]
                expected_policy_before = _advance_split_root(
                    policy_root, accepted_steps - 1, width=4
                )[0]
                expected_policy = np.stack(
                    (
                        np.asarray(jax.device_get(jr.key_data(expected_policy_before))),
                        np.asarray(jax.device_get(jr.key_data(expected_policy_after))),
                    )
                )
                _expect_array_equal(
                    ctx,
                    rng["final_policy_key_data"],
                    expected_policy,
                    path="run.rng.final_policy_key_data",
                    code="RNG_CHAIN",
                )
        except (TypeError, ValueError, RuntimeError) as exc:
            ctx.add("RNG_CHAIN", "run.rng", f"cannot reconstruct key chain: {exc}")
    if stream is not None and plan is not None:
        if control is not None:
            expected_stream = np.asarray(
                jax.device_get(
                    reconstruct_v6_stream_code(
                        run.initial_state.world,
                        control.world_config,
                        plan.run_steps,
                    )
                ),
                dtype=np.uint8,
            )
            if not bool(
                np.array_equal(stream[: plan.run_steps], expected_stream[: plan.run_steps])
            ):
                ctx.add(
                    "STREAM_RECONSTRUCTION",
                    "run.stream_code",
                    "active prefix differs from exact named-world-key replay",
                )
        if not bool(np.all(stream[plan.run_steps :] == np.uint8(0))):
            ctx.add("STREAM_PADDING", "run.stream_code", "padding suffix must be exact zero")


def validate_hidden_partner_lifecycle_world_v6_development_run(
    run: object,
) -> V6DevelopmentRunValidation:
    """Strictly validate one already-executed in-memory DEVELOPMENT run.

    This function performs host verification only.  It neither executes the
    scan nor grants evidence, promotion, artifact, threshold, or seed
    authority and does not establish replay authenticity.  Its two statuses
    describe structural integrity only; empirical lifecycle and coverage
    failures are returned separately.
    """

    ctx = _ValidationContext()
    _validate_static_orders(ctx)
    if not _exact_class(ctx, run, V6DevelopmentRun, path="run"):
        return V6DevelopmentRunValidation(
            schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
            status=STRUCTURALLY_INVALID_DEVELOPMENT_RUN,
            development_only=True,
            structural_only=True,
            replay_verified=False,
            execution_authorized=False,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
            errors=tuple(ctx.errors),
            lifecycle=_NEGATIVE_LIFECYCLE,
            coverage=_NEGATIVE_COVERAGE,
            quality=_NEGATIVE_QUALITY,
        )
    exact_run = cast(V6DevelopmentRun, run)
    _walk_concrete(ctx, exact_run, path="run")
    _validate_runtime_provenance(ctx, exact_run.runtime)

    windows = _record_arrays(
        ctx,
        exact_run.windows,
        V6WindowTotals,
        _WINDOW_CONTRACT,
        path="run.windows",
    )
    rows = _record_arrays(
        ctx,
        exact_run.row_heads,
        V6RowHeadTotals,
        _ROW_HEAD_CONTRACT,
        path="run.row_heads",
    )
    filters = _record_arrays(
        ctx,
        exact_run.filter_totals,
        V6FilterTotals,
        _FILTER_CONTRACT,
        path="run.filter_totals",
    )
    action = _record_arrays(
        ctx,
        exact_run.action_totals,
        V6ActionTotals,
        _ACTION_CONTRACT,
        path="run.action_totals",
    )
    audits = _record_arrays(
        ctx,
        exact_run.audits,
        V6AuditTotals,
        _AUDIT_CONTRACT,
        path="run.audits",
    )
    ledger = _record_arrays(
        ctx,
        exact_run.ledger,
        V6CadenceLedger,
        _LEDGER_CONTRACT,
        path="run.ledger",
    )
    lifecycle_arrays = _record_arrays(
        ctx,
        exact_run.lifecycle,
        V6LifecycleChainState,
        _LIFECYCLE_CONTRACT,
        path="run.lifecycle",
    )
    rng = _record_arrays(
        ctx,
        exact_run.rng,
        V6RngRecord,
        _RNG_CONTRACT,
        path="run.rng",
    )
    stream = _array(
        ctx,
        exact_run.stream_code,
        path="run.stream_code",
        shape=(MAX_SCAN_STEPS,),
        dtype=np.uint8,
    )

    runner: HiddenPartnerLifecycleWorldV6Runner | None = None
    control: HiddenPartnerLifecycleWorldV6Control | None = None
    plan: HiddenPartnerLifecycleWorldV6ScanPlan | None = None
    try:
        runner, control = _validate_live_binding(ctx, exact_run)
    except (AttributeError, IndexError, OSError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "live_binding", f"host validation failed closed: {exc}")
    try:
        plan = _validate_plan(ctx, exact_run)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "plan", f"host validation failed closed: {exc}")
    try:
        _validate_state_endpoints(ctx, exact_run, runner, rng)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "state_endpoints", f"host validation failed closed: {exc}")
    try:
        _validate_resources(ctx, exact_run)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "resources", f"host validation failed closed: {exc}")

    complete_windows = False
    accepted_windows = False
    row_support = False
    cue_support = False
    balanced_actions = False
    cadence_support = False
    active_steps: int | None = None
    accepted_steps: int | None = None
    try:
        complete_windows, accepted_windows = _validate_windows(ctx, windows)
        cue_support, balanced_actions = _validate_filter_and_action(
            ctx,
            filters,
            action,
            control,
        )
        row_support = _validate_row_heads(ctx, rows, action, control)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "aggregates", f"host validation failed closed: {exc}")
    try:
        active_steps, accepted_steps = _validate_audits_and_counters(
            ctx,
            audits,
            exact_run,
            plan,
            control,
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "audits", f"host validation failed closed: {exc}")
    if accepted_steps is not None and "focal_action_support" in action:
        if int(np.sum(action["focal_action_support"])) != accepted_steps:
            ctx.add(
                "AGGREGATE_CROSSCHECK",
                "run.action_totals.focal_action_support",
                "must sum to audited accepted_steps",
            )
    grounded_present = bool(
        control is not None
        and control.agent_config is not None
        and control.agent_config.grounded_world_model is not None
    )
    if grounded_present and {
        "grounded_support",
        "accepted_support",
    }.issubset(windows):
        _expect_array_equal(
            ctx,
            windows["grounded_support"],
            windows["accepted_support"],
            path="run.windows.grounded_support",
            code="AGGREGATE_CROSSCHECK",
        )
    try:
        cadence_support = _validate_ledger(ctx, ledger, exact_run, plan, control)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "ledger", f"host validation failed closed: {exc}")

    outcome = _recompute_lifecycle(ledger, plan)
    try:
        _validate_lifecycle_summary(ctx, lifecycle_arrays, outcome)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "lifecycle", f"host validation failed closed: {exc}")
    if not outcome.structural_chain_consistent:
        ctx.add(
            "LIFECYCLE_STRUCTURE", "run.lifecycle", "observed lifecycle chain is not structural"
        )
    try:
        _validate_rng_and_stream(
            ctx,
            rng,
            stream,
            exact_run,
            plan,
            accepted_steps,
            action,
            control,
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("VALIDATOR_STAGE", "rng_stream", f"host validation failed closed: {exc}")

    coverage = V6CoverageOutcome(
        complete_window_support=complete_windows,
        complete_accepted_window_support=accepted_windows,
        complete_row_head_support=row_support,
        complete_filter_cue_support=cue_support,
        complete_cadence_support=cadence_support,
        balanced_external_action_support=balanced_actions,
    )
    lane_values = tuple(
        int(audits[name].item())
        for name in (
            "learner_valid_steps",
            "filter_valid_steps",
            "oracle_valid_steps",
            "mechanism_valid_steps",
            "all_finite_steps",
        )
        if name in audits
    )
    quality = V6QualityOutcome(
        all_steps_accepted=(
            active_steps is not None
            and accepted_steps is not None
            and accepted_steps == active_steps
        ),
        all_evaluator_lanes_present=(
            active_steps is not None
            and len(lane_values) == 5
            and all(value == active_steps for value in lane_values)
        ),
        grounded_lane_present=(
            "grounded_support" in windows and bool(np.any(windows["grounded_support"] > 0))
        ),
        c_identity_outcome=(
            outcome.c_ever_acquired
            and outcome.c_retained_after_acquisition
            and outcome.c_acquired_in_first_c
        ),
        d_identity_outcome=outcome.d_ordered_outcome,
    )
    status: ValidationStatus = (
        STRUCTURALLY_VALID_DEVELOPMENT_RUN
        if not ctx.errors
        else STRUCTURALLY_INVALID_DEVELOPMENT_RUN
    )
    return V6DevelopmentRunValidation(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
        status=status,
        development_only=True,
        structural_only=True,
        replay_verified=False,
        execution_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        errors=tuple(ctx.errors),
        lifecycle=outcome,
        coverage=coverage,
        quality=quality,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA",
    "REPLAY_VERIFIED",
    "STRUCTURALLY_INVALID_DEVELOPMENT_RUN",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "STRUCTURAL_ONLY",
    "STRUCTURALLY_VALID_DEVELOPMENT_RUN",
    "V6CoverageOutcome",
    "V6DevelopmentRunValidation",
    "V6LifecycleOutcome",
    "V6QualityOutcome",
    "V6ValidationError",
    "validate_hidden_partner_lifecycle_world_v6_development_run",
]
