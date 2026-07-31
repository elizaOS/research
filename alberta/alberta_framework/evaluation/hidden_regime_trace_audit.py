"""Independent, fail-closed audit of hidden-regime primitive traces.

The development evaluator emits a complete primitive trace, but its own replay
is necessarily coupled to the implementation that produced the trace.  This
module instead translates every trace row into the separate host role and
world oracle records.  Those oracles canonicalize the records and reconstruct
the state machines without calling the learner update, world step, or an
evaluator runner.

The audit remains development infrastructure.  Random decisions and cue draws
use the same JAX backend's threefry key semantics, and transition arithmetic is
float32.  The role and world transition state machines are independently
implemented on the host, but this is neither a cross-backend portability proof
nor scientific evidence or an Alberta Plan completion claim.

Primitive-trace v3 records ``transition.terminated`` and
``transition.discount``.  They are supplied to the independent world oracle
as actual leaves rather than synthesized protocol constants.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    N_SLOT_ACTIONS,
    N_SLOT_INPUTS,
    N_SLOTS,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    DurableWritePolicy,
    ReplacementTargetPolicy,
    SlotSignalingState,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
    HIDDEN_REGIME_TRACE_SCHEMA,
    HiddenRegimeDevelopmentCondition,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeResourceReport,
    HiddenRegimeRunResult,
    HiddenRegimeRunSummary,
    HiddenRegimeSeedPair,
)
from alberta_framework.evaluation.hidden_regime_summary_oracle import (
    validate_hidden_regime_summary,
)
from alberta_framework.evaluation.hidden_regime_world_oracle import (
    DeliveryContract,
    HiddenRegimeWorldDiagnosticsSnapshot,
    HiddenRegimeWorldOracleConfig,
    HiddenRegimeWorldStateSnapshot,
    HiddenRegimeWorldTransitionRecord,
    validate_hidden_regime_world_state_continuity,
    validate_hidden_regime_world_transition,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    INPUT_DELIVERED_MESSAGE,
    INPUT_PRIVATE_CUE,
    ROLE_BENEFICIARY,
    ROLE_HELPER,
    RoleName,
    SlotRoleDecisionSnapshot,
    SlotRoleOracleConfig,
    SlotRoleStateSnapshot,
    SlotRoleTransitionRecord,
    SlotRoleUpdateDiagnosticsSnapshot,
    validate_slot_role_state_continuity,
    validate_slot_role_transition,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CONSTANT_ZERO_TERNARY_CHANNEL,
    DIRECT_TERNARY_CHANNEL,
    SHUFFLED_TERNARY_CHANNEL,
)

HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA = "alberta.hidden-regime.trace-audit-input.v3"
HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA = "alberta.hidden-regime.trace-audit-report.v3"

# These literals intentionally do not follow a future evaluator schema silently.
SUPPORTED_DEVELOPMENT_SCHEMA = "alberta.hidden-regime-signaling.development.v5"
SUPPORTED_TRACE_SCHEMA = "alberta.hidden-regime-signaling.primitive-trace.v3"

EVIDENCE_BOUNDARY = (
    "development-only audit using same-backend JAX threefry key/draw semantics and "
    "float32 arithmetic; role and world transitions are independently implemented host "
    "state machines, including actual runtime terminated and discount leaves; only an "
    "enumerated set of exact fused, unfused, and algebraically reassociated float32 "
    "relevance contractions is admitted and disclosed; every derived summary and "
    "resource field is independently reconstructed from primitives and the final state; "
    "this is not cross-backend portability evidence, scientific promotion, or an "
    "Alberta Plan completion claim"
)
UNOBSERVED_TRANSITION_FIELDS: tuple[str, ...] = ()

# Stable named-stream tags are repeated intentionally instead of calling either
# production key-container helper.  Changing a producer tag without changing
# the trace schema therefore fails row-zero provenance.
_HELPER_RNG_TAG = 0x53484C50  # ASCII "SHLP"
_BENEFICIARY_RNG_TAG = 0x53424E46  # ASCII "SBNF"
_CUE_RNG_TAG = 0x48524355  # ASCII "HRCU"
_CHANNEL_RNG_TAG = 0x48524348  # ASCII "HRCH"


@dataclasses.dataclass(frozen=True)
class _IndependentConditionSpec:
    """Audit-local binding of one serialized intervention identifier.

    The eight bindings are intentionally duplicated rather than obtained from
    the development evaluator.  A producer-side condition-binding defect must
    therefore disagree with the trace instead of teaching the audit the same
    defect.
    """

    channel: DeliveryContract
    helper_write: bool
    beneficiary_write: bool
    durable_write_policy: DurableWritePolicy
    replacement_target_policy: ReplacementTargetPolicy

    @property
    def durable_writes_enabled(self) -> bool:
        """Whether ordinary writes may mutate already-durable values."""

        return self.durable_write_policy == DURABLE_WRITE_WRITABLE


_INDEPENDENT_CONDITION_SPECS: Mapping[str, _IndependentConditionSpec] = {
    "selective_full": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
    "writable_evidence": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_WRITABLE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
    "selective_lru": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_LRU,
    ),
    "writable_lru": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_WRITABLE,
        REPLACEMENT_TARGET_LRU,
    ),
    "helper_frozen": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        False,
        True,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
    "beneficiary_frozen": _IndependentConditionSpec(
        DIRECT_TERNARY_CHANNEL,
        True,
        False,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
    "constant_channel_0": _IndependentConditionSpec(
        CONSTANT_ZERO_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
    "shuffled_channel": _IndependentConditionSpec(
        SHUFFLED_TERNARY_CHANNEL,
        True,
        True,
        DURABLE_WRITE_SELECTIVE,
        REPLACEMENT_TARGET_EVIDENCE,
    ),
}


def _independent_condition_spec(
    condition: HiddenRegimeDevelopmentCondition,
) -> _IndependentConditionSpec:
    """Resolve a condition only through the audit-owned literal table."""

    try:
        return _INDEPENDENT_CONDITION_SPECS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown hidden-regime development condition: {condition!r}") from exc


_INT32_FIELDS = frozenset(
    {
        "step_index",
        "segment_index",
        "segment_step",
        "world_step_count_pre",
        "world_step_count_post",
        "world_schedule_position_pre",
        "world_schedule_position_post",
        "helper_decision_slot",
        "helper_private_input",
        "helper_decision_action",
        "beneficiary_decision_slot",
        "beneficiary_private_input",
        "beneficiary_decision_action",
        "helper_failed_leases_pre",
        "helper_failed_leases_post",
        "beneficiary_failed_leases_pre",
        "beneficiary_failed_leases_post",
        "helper_idle_leases_pre",
        "helper_idle_leases_post",
        "beneficiary_idle_leases_pre",
        "beneficiary_idle_leases_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
        "helper_candidate_confirmations_pre",
        "helper_candidate_confirmations_post",
        "beneficiary_candidate_confirmations_pre",
        "beneficiary_candidate_confirmations_post",
        "helper_scratch_failed_leases_pre",
        "helper_scratch_failed_leases_post",
        "beneficiary_scratch_failed_leases_pre",
        "beneficiary_scratch_failed_leases_post",
        "helper_lease_offset_pre",
        "helper_lease_offset_post",
        "beneficiary_lease_offset_pre",
        "beneficiary_lease_offset_post",
        "helper_remaining_durable_tests_pre",
        "helper_remaining_durable_tests_post",
        "beneficiary_remaining_durable_tests_pre",
        "beneficiary_remaining_durable_tests_post",
        "helper_search_cursor_pre",
        "helper_search_cursor_post",
        "beneficiary_search_cursor_pre",
        "beneficiary_search_cursor_post",
        "helper_next_generation_pre",
        "helper_next_generation_post",
        "beneficiary_next_generation_pre",
        "beneficiary_next_generation_post",
        "helper_committed_slot",
        "helper_committed_generation",
        "helper_retired_slot",
        "helper_retired_generation",
        "beneficiary_committed_slot",
        "beneficiary_committed_generation",
        "beneficiary_retired_slot",
        "beneficiary_retired_generation",
    }
)

_INT8_FIELDS = frozenset(
    {
        "regime_id",
        "world_cue_pre",
        "world_cue_post",
        "helper_cue",
        "oracle_target",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "helper_active_slot_pre",
        "helper_active_slot_post",
        "beneficiary_active_slot_pre",
        "beneficiary_active_slot_post",
        "helper_status_pre",
        "helper_status_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
    }
)

_BOOL_FIELDS = frozenset(
    {
        "helper_write_enabled",
        "beneficiary_write_enabled",
        "helper_value_write",
        "beneficiary_value_write",
        "helper_lease_boundary",
        "beneficiary_lease_boundary",
        "helper_candidate_lease_success",
        "beneficiary_candidate_lease_success",
        "helper_relevance_ready",
        "beneficiary_relevance_ready",
        "helper_durable_relevant",
        "beneficiary_durable_relevant",
        "helper_candidate_relevant",
        "beneficiary_candidate_relevant",
        "helper_generation_exhausted",
        "beneficiary_generation_exhausted",
        "helper_scratch_retest_started",
        "beneficiary_scratch_retest_started",
        "world_terminated",
        "lifecycle_synchronized",
        "helper_selective_mutation_violation",
        "beneficiary_selective_mutation_violation",
    }
)

_UINT32_FIELDS = frozenset(
    {
        "helper_value_bits_pre",
        "helper_value_bits_post",
        "beneficiary_value_bits_pre",
        "beneficiary_value_bits_post",
        "world_cue_key_data_pre",
        "world_cue_key_data_post",
        "world_channel_key_data_pre",
        "world_channel_key_data_post",
        "helper_policy_key_data_pre",
        "helper_policy_key_data_post",
        "beneficiary_policy_key_data_pre",
        "beneficiary_policy_key_data_post",
    }
)

_FLOAT32_FIELDS = frozenset(
    {
        "reward",
        "world_discount",
        "helper_selected_value",
        "beneficiary_selected_value",
        "helper_relevance_mean_pre",
        "helper_relevance_mean_post",
        "beneficiary_relevance_mean_pre",
        "beneficiary_relevance_mean_post",
        "helper_relevance_mass_pre",
        "helper_relevance_mass_post",
        "beneficiary_relevance_mass_pre",
        "beneficiary_relevance_mass_post",
        "helper_lease_reward_sum_pre",
        "helper_lease_reward_sum_post",
        "beneficiary_lease_reward_sum_pre",
        "beneficiary_lease_reward_sum_post",
        "helper_value_pre",
        "helper_candidate_value",
        "helper_value_post",
        "beneficiary_value_pre",
        "beneficiary_candidate_value",
        "beneficiary_value_post",
        "helper_lease_reward_mean",
        "beneficiary_lease_reward_mean",
    }
)

_VECTOR_FIELDS = frozenset(
    {
        "helper_relevance_mean_pre",
        "helper_relevance_mean_post",
        "beneficiary_relevance_mean_pre",
        "beneficiary_relevance_mean_post",
        "helper_relevance_mass_pre",
        "helper_relevance_mass_post",
        "beneficiary_relevance_mass_pre",
        "beneficiary_relevance_mass_post",
        "helper_failed_leases_pre",
        "helper_failed_leases_post",
        "beneficiary_failed_leases_pre",
        "beneficiary_failed_leases_post",
        "helper_idle_leases_pre",
        "helper_idle_leases_post",
        "beneficiary_idle_leases_pre",
        "beneficiary_idle_leases_post",
        "helper_status_pre",
        "helper_status_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
        "helper_selective_mutation_violation",
        "beneficiary_selective_mutation_violation",
    }
)

_BANK_FIELDS = frozenset(
    {
        "helper_value_bits_pre",
        "helper_value_bits_post",
        "beneficiary_value_bits_pre",
        "beneficiary_value_bits_post",
    }
)

_KEY_FIELDS = _UINT32_FIELDS - _BANK_FIELDS
_DECLARED_TRACE_FIELDS = (
    _INT32_FIELDS | _INT8_FIELDS | _BOOL_FIELDS | _UINT32_FIELDS | _FLOAT32_FIELDS
)

type TraceArrays = Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]


@dataclasses.dataclass(frozen=True)
class HiddenRegimeTraceAuditInput:
    """Runtime envelope binding a trace to its seed, condition, config, and output."""

    condition: HiddenRegimeDevelopmentCondition
    seed_pair: HiddenRegimeSeedPair
    config: HiddenRegimeDevelopmentConfig
    trace: HiddenRegimePrimitiveTrace
    final_learner_state: SlotSignalingState
    summary: HiddenRegimeRunSummary
    resource: HiddenRegimeResourceReport
    schema: str = HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA
    development_schema: str = HIDDEN_REGIME_DEVELOPMENT_SCHEMA
    trace_schema: str = HIDDEN_REGIME_TRACE_SCHEMA

    @classmethod
    def from_run_result(cls, run: HiddenRegimeRunResult) -> HiddenRegimeTraceAuditInput:
        """Bind directly to the complete output of one development run."""

        if not isinstance(run, HiddenRegimeRunResult):
            raise TypeError("run must be a HiddenRegimeRunResult")
        return cls(
            condition=run.condition,
            seed_pair=run.seed_pair,
            config=run.config,
            trace=run.trace,
            final_learner_state=run.final_state,
            summary=run.summary,
            resource=run.resource,
        )


@dataclasses.dataclass(frozen=True)
class HiddenRegimeTraceAuditReport:
    """Result of auditing every role and world transition in one trace."""

    valid: bool
    expected_steps: int
    rows_checked: int
    helper_transitions_checked: int
    beneficiary_transitions_checked: int
    world_transitions_checked: int
    mismatches: tuple[str, ...]
    commit_lineages_checked: int = 0
    recurrence_records_checked: int = 0
    retention_aggregate_fields_checked: int = 0
    summary_fields_checked: int = 0
    resource_fields_checked: int = 0
    accepted_float32_contractions: tuple[str, ...] = ()
    unobserved_transition_fields: tuple[str, ...] = UNOBSERVED_TRANSITION_FIELDS
    evidence_boundary: str = EVIDENCE_BOUNDARY
    schema: str = HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "expected_steps": self.expected_steps,
            "rows_checked": self.rows_checked,
            "helper_transitions_checked": self.helper_transitions_checked,
            "beneficiary_transitions_checked": self.beneficiary_transitions_checked,
            "world_transitions_checked": self.world_transitions_checked,
            "commit_lineages_checked": self.commit_lineages_checked,
            "recurrence_records_checked": self.recurrence_records_checked,
            "retention_aggregate_fields_checked": (self.retention_aggregate_fields_checked),
            "summary_fields_checked": self.summary_fields_checked,
            "resource_fields_checked": self.resource_fields_checked,
            "mismatches": list(self.mismatches),
            "accepted_float32_contractions": list(self.accepted_float32_contractions),
            "unobserved_transition_fields": list(self.unobserved_transition_fields),
            "evidence_boundary": self.evidence_boundary,
        }


def _key_data(key: Array) -> tuple[int, int]:
    host = np.asarray(jr.key_data(key), dtype=np.uint32)
    if host.shape != (2,):
        raise ValueError("named root key must use two-word threefry data")
    return cast(tuple[int, int], tuple(int(value) for value in host.tolist()))


def _expected_shape(field: str, steps: int) -> tuple[int, ...]:
    if field in _BANK_FIELDS:
        return (steps, N_SLOTS, N_SLOT_INPUTS, N_SLOT_ACTIONS)
    if field in _KEY_FIELDS:
        return (steps, 2)
    if field in _VECTOR_FIELDS:
        return (steps, N_SLOTS)
    return (steps,)


def _expected_dtype(field: str) -> np.dtype[np.generic]:
    if field in _INT32_FIELDS:
        return np.dtype(np.int32)
    if field in _INT8_FIELDS:
        return np.dtype(np.int8)
    if field in _BOOL_FIELDS:
        return np.dtype(np.bool_)
    if field in _UINT32_FIELDS:
        return np.dtype(np.uint32)
    return np.dtype(np.float32)


def _validate_envelope(
    audit_input: object,
    mismatches: list[str],
) -> HiddenRegimeTraceAuditInput | None:
    if not isinstance(audit_input, HiddenRegimeTraceAuditInput):
        mismatches.append("input: expected HiddenRegimeTraceAuditInput")
        return None
    if audit_input.schema != HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA:
        mismatches.append("input.schema")
    if audit_input.development_schema != SUPPORTED_DEVELOPMENT_SCHEMA:
        mismatches.append("input.development_schema")
    if audit_input.trace_schema != SUPPORTED_TRACE_SCHEMA:
        mismatches.append("input.trace_schema")
    if HIDDEN_REGIME_DEVELOPMENT_SCHEMA != SUPPORTED_DEVELOPMENT_SCHEMA:
        mismatches.append("runtime.development_schema")
    if HIDDEN_REGIME_TRACE_SCHEMA != SUPPORTED_TRACE_SCHEMA:
        mismatches.append("runtime.trace_schema")
    if audit_input.condition not in {
        "selective_full",
        "writable_evidence",
        "selective_lru",
        "writable_lru",
        "helper_frozen",
        "beneficiary_frozen",
        "constant_channel_0",
        "shuffled_channel",
    }:
        mismatches.append("input.condition")
    if not isinstance(audit_input.seed_pair, HiddenRegimeSeedPair):
        mismatches.append("input.seed_pair")
    if not isinstance(audit_input.config, HiddenRegimeDevelopmentConfig):
        mismatches.append("input.config")
    if not isinstance(audit_input.trace, HiddenRegimePrimitiveTrace):
        mismatches.append("input.trace")
    if not isinstance(audit_input.final_learner_state, SlotSignalingState):
        mismatches.append("input.final_learner_state")
    if not isinstance(audit_input.summary, HiddenRegimeRunSummary):
        mismatches.append("input.summary")
    if not isinstance(audit_input.resource, HiddenRegimeResourceReport):
        mismatches.append("input.resource")
    return audit_input


def _trace_arrays(
    trace: HiddenRegimePrimitiveTrace,
    steps: int,
    mismatches: list[str],
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]] | None:
    runtime_fields = {field.name for field in dataclasses.fields(trace)}
    class_fields = {field.name for field in dataclasses.fields(HiddenRegimePrimitiveTrace)}
    if runtime_fields != class_fields:
        mismatches.append("trace.fields.runtime_class_mismatch")
        return None
    if class_fields != _DECLARED_TRACE_FIELDS:
        missing = sorted(_DECLARED_TRACE_FIELDS - class_fields)
        extra = sorted(class_fields - _DECLARED_TRACE_FIELDS)
        mismatches.append(f"trace.fields.schema_drift: missing={missing}, extra={extra}")
        return None

    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]] = {}
    for field in sorted(class_fields):
        try:
            array = np.asarray(getattr(trace, field))
        except (TypeError, ValueError) as error:
            mismatches.append(f"trace.{field}: cannot materialize array: {error}")
            continue
        expected_shape = _expected_shape(field, steps)
        if array.shape != expected_shape:
            mismatches.append(f"trace.{field}.shape: expected {expected_shape}, got {array.shape}")
            continue
        expected_dtype = _expected_dtype(field)
        if array.dtype != expected_dtype:
            mismatches.append(f"trace.{field}.dtype: expected {expected_dtype}, got {array.dtype}")
            continue
        if field in _FLOAT32_FIELDS and not bool(np.all(np.isfinite(array))):
            mismatches.append(f"trace.{field}: contains nonfinite float32")
            continue
        arrays[field] = array
    if len(arrays) != len(class_fields):
        return None
    return arrays


def _float_values_from_bits(
    bits: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    contiguous = np.ascontiguousarray(bits, dtype=np.uint32)
    values = contiguous.view(np.float32)
    return tuple(
        tuple(tuple(float(value) for value in row) for row in matrix) for matrix in values.tolist()
    )


def _ints(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> tuple[int, ...]:
    return tuple(int(value) for value in array.tolist())


def _floats(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> tuple[float, ...]:
    return tuple(float(value) for value in array.tolist())


def _row_role_state(
    arrays: TraceArrays,
    row: int,
    role: RoleName,
    phase: Literal["pre", "post"],
) -> SlotRoleStateSnapshot:
    prefix = str(role)
    return SlotRoleStateSnapshot(
        values=_float_values_from_bits(arrays[f"{prefix}_value_bits_{phase}"][row]),
        relevance_mean=_floats(arrays[f"{prefix}_relevance_mean_{phase}"][row]),
        relevance_mass=_floats(arrays[f"{prefix}_relevance_mass_{phase}"][row]),
        failed_leases=_ints(arrays[f"{prefix}_failed_leases_{phase}"][row]),
        idle_leases=_ints(arrays[f"{prefix}_idle_leases_{phase}"][row]),
        status=_ints(arrays[f"{prefix}_status_{phase}"][row]),
        generation=_ints(arrays[f"{prefix}_generation_{phase}"][row]),
        active_slot=int(arrays[f"{prefix}_active_slot_{phase}"][row]),
        lease_offset=int(arrays[f"{prefix}_lease_offset_{phase}"][row]),
        lease_reward_sum=float(arrays[f"{prefix}_lease_reward_sum_{phase}"][row]),
        remaining_durable_tests=int(arrays[f"{prefix}_remaining_durable_tests_{phase}"][row]),
        search_cursor=int(arrays[f"{prefix}_search_cursor_{phase}"][row]),
        candidate_successful_leases=int(arrays[f"{prefix}_candidate_confirmations_{phase}"][row]),
        next_generation=int(arrays[f"{prefix}_next_generation_{phase}"][row]),
        key_data=cast(
            tuple[int, int],
            tuple(int(value) for value in arrays[f"{prefix}_policy_key_data_{phase}"][row]),
        ),
    )


def _row_decision(
    arrays: TraceArrays,
    row: int,
    role: RoleName,
) -> SlotRoleDecisionSnapshot:
    prefix = str(role)
    return SlotRoleDecisionSnapshot(
        slot=int(arrays[f"{prefix}_decision_slot"][row]),
        private_input=int(arrays[f"{prefix}_private_input"][row]),
        action=int(arrays[f"{prefix}_decision_action"][row]),
        selected_value=float(arrays[f"{prefix}_selected_value"][row]),
        next_key_data=cast(
            tuple[int, int],
            tuple(int(value) for value in arrays[f"{prefix}_policy_key_data_post"][row]),
        ),
    )


def _row_diagnostics(
    arrays: TraceArrays,
    row: int,
    role: RoleName,
) -> SlotRoleUpdateDiagnosticsSnapshot:
    prefix = str(role)
    return SlotRoleUpdateDiagnosticsSnapshot(
        value_pre=float(arrays[f"{prefix}_value_pre"][row]),
        candidate_value=float(arrays[f"{prefix}_candidate_value"][row]),
        value_post=float(arrays[f"{prefix}_value_post"][row]),
        value_write=bool(arrays[f"{prefix}_value_write"][row]),
        lease_boundary=bool(arrays[f"{prefix}_lease_boundary"][row]),
        lease_reward_mean=float(arrays[f"{prefix}_lease_reward_mean"][row]),
        relevance_ready=bool(arrays[f"{prefix}_relevance_ready"][row]),
        durable_relevant=bool(arrays[f"{prefix}_durable_relevant"][row]),
        candidate_relevant=bool(arrays[f"{prefix}_candidate_relevant"][row]),
        candidate_lease_success=bool(arrays[f"{prefix}_candidate_lease_success"][row]),
        scratch_failed_leases_pre=int(arrays[f"{prefix}_scratch_failed_leases_pre"][row]),
        scratch_failed_leases_post=int(arrays[f"{prefix}_scratch_failed_leases_post"][row]),
        scratch_retest_started=bool(arrays[f"{prefix}_scratch_retest_started"][row]),
        generation_exhausted=bool(arrays[f"{prefix}_generation_exhausted"][row]),
        committed_slot=int(arrays[f"{prefix}_committed_slot"][row]),
        committed_generation=int(arrays[f"{prefix}_committed_generation"][row]),
        retired_slot=int(arrays[f"{prefix}_retired_slot"][row]),
        retired_generation=int(arrays[f"{prefix}_retired_generation"][row]),
    )


def _row_role_record(
    arrays: TraceArrays,
    row: int,
    role: RoleName,
    config: SlotRoleOracleConfig,
    lifecycle_write: bool,
) -> SlotRoleTransitionRecord:
    prefix = str(role)
    input_stream = INPUT_PRIVATE_CUE if role == ROLE_HELPER else INPUT_DELIVERED_MESSAGE
    return SlotRoleTransitionRecord(
        role=role,
        input_stream=input_stream,
        config=config,
        old_state=_row_role_state(arrays, row, role, "pre"),
        decision=_row_decision(arrays, row, role),
        reward=float(arrays["reward"][row]),
        external_value_write=bool(arrays[f"{prefix}_write_enabled"][row]),
        lifecycle_write=lifecycle_write,
        diagnostics=_row_diagnostics(arrays, row, role),
        new_state=_row_role_state(arrays, row, role, "post"),
    )


def _row_world_state(
    arrays: TraceArrays,
    row: int,
    phase: Literal["pre", "post"],
) -> HiddenRegimeWorldStateSnapshot:
    return HiddenRegimeWorldStateSnapshot(
        cue_key_data=cast(
            tuple[int, int],
            tuple(int(value) for value in arrays[f"world_cue_key_data_{phase}"][row]),
        ),
        channel_key_data=cast(
            tuple[int, int],
            tuple(int(value) for value in arrays[f"world_channel_key_data_{phase}"][row]),
        ),
        cue=int(arrays[f"world_cue_{phase}"][row]),
        step_count=int(arrays[f"world_step_count_{phase}"][row]),
        schedule_position=int(arrays[f"world_schedule_position_{phase}"][row]),
    )


def _row_world_record(
    arrays: TraceArrays,
    row: int,
    config: HiddenRegimeWorldOracleConfig,
    delivery_contract: DeliveryContract,
) -> HiddenRegimeWorldTransitionRecord:
    return HiddenRegimeWorldTransitionRecord(
        config=config,
        old_state=_row_world_state(arrays, row, "pre"),
        delivery_contract=delivery_contract,
        helper_message=int(arrays["helper_message"][row]),
        delivered_message=int(arrays["delivered_message"][row]),
        beneficiary_action=int(arrays["beneficiary_action"][row]),
        diagnostics=HiddenRegimeWorldDiagnosticsSnapshot(
            observation_helper_cue=int(arrays["helper_cue"][row]),
            reward=float(arrays["reward"][row]),
            next_observation_helper_cue=int(arrays["world_cue_post"][row]),
            terminated=bool(arrays["world_terminated"][row]),
            discount=float(arrays["world_discount"][row]),
            oracle_step_count=int(arrays["step_index"][row]),
            oracle_segment_index=int(arrays["segment_index"][row]),
            oracle_segment_step=int(arrays["segment_step"][row]),
            oracle_regime_id=int(arrays["regime_id"][row]),
            oracle_target=int(arrays["oracle_target"][row]),
        ),
        new_state=_row_world_state(arrays, row, "post"),
    )


def _zero_role_state(key: Array) -> SlotRoleStateSnapshot:
    zero_row = tuple(0.0 for _ in range(N_SLOT_ACTIONS))
    zero_matrix = tuple(zero_row for _ in range(N_SLOT_INPUTS))
    return SlotRoleStateSnapshot(
        values=tuple(zero_matrix for _ in range(N_SLOTS)),
        relevance_mean=tuple(0.0 for _ in range(N_SLOTS)),
        relevance_mass=tuple(0.0 for _ in range(N_SLOTS)),
        failed_leases=tuple(0 for _ in range(N_SLOTS)),
        idle_leases=tuple(0 for _ in range(N_SLOTS)),
        status=(SLOT_SCRATCH, 0, 0, 0),
        generation=tuple(0 for _ in range(N_SLOTS)),
        active_slot=SCRATCH_SLOT,
        lease_offset=0,
        lease_reward_sum=0.0,
        remaining_durable_tests=0,
        search_cursor=1,
        candidate_successful_leases=0,
        next_generation=1,
        key_data=_key_data(key),
    )


def _initial_states(
    seed_pair: HiddenRegimeSeedPair,
) -> tuple[SlotRoleStateSnapshot, SlotRoleStateSnapshot, HiddenRegimeWorldStateSnapshot]:
    learner_root = jr.key(seed_pair.learner_seed)
    helper_key = jr.fold_in(learner_root, _HELPER_RNG_TAG)
    beneficiary_key = jr.fold_in(learner_root, _BENEFICIARY_RNG_TAG)
    world_root = jr.key(seed_pair.world_seed)
    cue_key = jr.fold_in(world_root, _CUE_RNG_TAG)
    channel_key = jr.fold_in(world_root, _CHANNEL_RNG_TAG)
    cue_draw_key, next_cue_key = jr.split(cue_key)
    initial_cue = int(
        np.asarray(
            jr.randint(
                cue_draw_key,
                (),
                0,
                N_SLOT_INPUTS,
                dtype=jnp.int32,
            )
        )
    )
    world = HiddenRegimeWorldStateSnapshot(
        cue_key_data=_key_data(next_cue_key),
        channel_key_data=_key_data(channel_key),
        cue=initial_cue,
        step_count=0,
        schedule_position=0,
    )
    return _zero_role_state(helper_key), _zero_role_state(beneficiary_key), world


def _float32_bits(value: float) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def _lifecycle_equal(left: SlotRoleStateSnapshot, right: SlotRoleStateSnapshot) -> bool:
    float_fields = ("relevance_mean", "relevance_mass")
    for field in float_fields:
        left_values = cast(tuple[float, ...], getattr(left, field))
        right_values = cast(tuple[float, ...], getattr(right, field))
        if tuple(_float32_bits(value) for value in left_values) != tuple(
            _float32_bits(value) for value in right_values
        ):
            return False
    scalar_float_equal = _float32_bits(left.lease_reward_sum) == _float32_bits(
        right.lease_reward_sum
    )
    return scalar_float_equal and all(
        getattr(left, field) == getattr(right, field)
        for field in (
            "failed_leases",
            "idle_leases",
            "status",
            "generation",
            "active_slot",
            "lease_offset",
            "remaining_durable_tests",
            "search_cursor",
            "candidate_successful_leases",
            "next_generation",
        )
    )


def _expected_selective_violation(
    arrays: TraceArrays,
    row: int,
    role: RoleName,
    durable_writes_enabled: bool,
) -> tuple[bool, ...]:
    if durable_writes_enabled:
        return tuple(False for _ in range(N_SLOTS))
    prefix = str(role)
    before = arrays[f"{prefix}_value_bits_pre"][row]
    after = arrays[f"{prefix}_value_bits_post"][row]
    status_pre = arrays[f"{prefix}_status_pre"][row]
    status_post = arrays[f"{prefix}_status_post"][row]
    generation_pre = arrays[f"{prefix}_generation_pre"][row]
    generation_post = arrays[f"{prefix}_generation_post"][row]
    committed_slot = int(arrays[f"{prefix}_committed_slot"][row])
    committed_generation = int(arrays[f"{prefix}_committed_generation"][row])
    retired_slot = int(arrays[f"{prefix}_retired_slot"][row])
    retired_generation = int(arrays[f"{prefix}_retired_generation"][row])
    result: list[bool] = []
    for slot in range(N_SLOTS):
        changed = not bool(np.array_equal(before[slot], after[slot]))
        atomic_replacement = (
            retired_slot == slot
            and committed_slot == slot
            and retired_generation == int(generation_pre[slot])
            and committed_generation == int(generation_post[slot])
            and int(generation_pre[slot]) != int(generation_post[slot])
            and int(status_post[slot]) == SLOT_DURABLE
        )
        result.append(int(status_pre[slot]) == SLOT_DURABLE and changed and not atomic_replacement)
    return tuple(result)


def _float32_tuple_bits(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(_float32_bits(value) for value in values)


def _relevance_contraction_labeled_values(
    old_value: np.float32,
    gain: np.float32,
    reward: np.float32,
) -> tuple[tuple[str, np.float32], ...]:
    """Enumerate the named exact source/FMA/reassociation results."""

    delta = np.float32(reward - old_value)
    gain_old = np.float32(gain * old_value)
    gain_reward = np.float32(gain * reward)
    one_minus_gain = np.float32(np.float32(1.0) - gain)
    return (
        # Source order, with and without multiply-add contraction.
        (
            "source_unfused",
            np.float32(old_value + np.float32(gain * delta)),
        ),
        (
            "source_fma",
            np.float32(
                np.float64(old_value)
                + np.float64(gain)
                * (np.float64(reward) - np.float64(old_value))
            ),
        ),
        # Algebraically equivalent forms emitted by XLA's reassociation pass.
        (
            "weighted_sum_unfused",
            np.float32(np.float32(one_minus_gain * old_value) + gain_reward),
        ),
        (
            "weighted_sum_fma",
            np.float32(
                np.float64(one_minus_gain) * np.float64(old_value)
                + np.float64(gain_reward)
            ),
        ),
        (
            "subtract_then_add",
            np.float32(np.float32(old_value - gain_old) + gain_reward),
        ),
        (
            "add_then_subtract",
            np.float32(np.float32(old_value + gain_reward) - gain_old),
        ),
        (
            "negative_gain_fma_after_rounded_sum",
            np.float32(
                np.float64(-gain) * np.float64(old_value)
                + np.float64(np.float32(old_value + gain_reward))
            ),
        ),
    )


def _relevance_contraction_values(
    old_value: np.float32,
    gain: np.float32,
    reward: np.float32,
) -> tuple[np.float32, ...]:
    """Return admitted results deduplicated and ordered by uint32 value."""

    labeled_values = _relevance_contraction_labeled_values(
        old_value,
        gain,
        reward,
    )
    by_bits = {
        _float32_bits(float(value)): value for _, value in labeled_values
    }
    return tuple(by_bits[key] for key in sorted(by_bits))


def _relevance_contraction_formula_ids(
    old_value: np.float32,
    gain: np.float32,
    reward: np.float32,
) -> dict[int, tuple[str, ...]]:
    """Map every admitted active-value word to its exact formula identifiers."""

    labeled_values = _relevance_contraction_labeled_values(
        old_value,
        gain,
        reward,
    )
    labels_by_bits: dict[int, list[str]] = {}
    for label, value in labeled_values:
        labels_by_bits.setdefault(_float32_bits(float(value)), []).append(label)
    return {
        bits: tuple(labels_by_bits[bits])
        for bits in sorted(labels_by_bits)
    }


def _portable_relevance_contraction(
    record: SlotRoleTransitionRecord,
    oracle_mismatches: tuple[str, ...],
) -> str | None:
    """Admit only exact fused/unfused variants of the active relevance update.

    A JIT scan may contract ``old + gain * (reward - old)`` while the row-wise
    host oracle evaluates separate float32 operations.  No tolerance is used:
    the complete recorded relevance vector must equal one of the explicitly
    enumerated exact source-order, FMA, or reassociated float32 outcomes after
    applying any atomic scratch commit.
    """

    if not oracle_mismatches or any(
        not path.startswith("new_state.relevance_mean[") for path in oracle_mismatches
    ):
        return None
    old = record.old_state
    slot = record.decision.slot
    if not 0 <= slot < N_SLOTS:
        return None
    old_value = np.float32(old.relevance_mean[slot])
    old_mass = np.float32(old.relevance_mass[slot])
    active_mass = np.minimum(
        np.float32(old_mass + np.float32(1.0)),
        np.float32(16_777_216.0),
    )
    gain = np.maximum(
        np.float32(record.config.relevance_rate),
        np.float32(np.float32(1.0) / active_mass),
    )
    reward = np.float32(record.reward)
    formula_ids = _relevance_contraction_formula_ids(old_value, gain, reward)

    candidates: dict[tuple[int, ...], tuple[int, tuple[str, ...]]] = {}
    for active_bits, labels in formula_ids.items():
        active_relevance = np.asarray(active_bits, dtype=np.uint32).view(np.float32)
        expected = np.asarray(old.relevance_mean, dtype=np.float32).copy()
        expected[slot] = active_relevance
        target = record.diagnostics.committed_slot
        if target >= 0:
            if target >= N_SLOTS:
                return None
            expected[target] = expected[SCRATCH_SLOT]
            expected[SCRATCH_SLOT] = np.float32(0.0)
        post_bits = tuple(int(value) for value in expected.view(np.uint32).tolist())
        candidates[post_bits] = (active_bits, labels)
    actual = _float32_tuple_bits(record.new_state.relevance_mean)
    match = candidates.get(actual)
    if match is None:
        return None
    active_bits, labels = match
    post_words = ",".join(f"0x{word:08x}" for word in actual)
    return (
        f"active_value_uint32=0x{active_bits:08x};"
        f"formula_ids={','.join(labels)};"
        f"post_vector_uint32={post_words}"
    )


def _add_validation_mismatches(
    prefix: str,
    valid: bool,
    details: tuple[str, ...],
    mismatches: list[str],
) -> None:
    if valid:
        return
    if not details:
        mismatches.append(prefix)
        return
    mismatches.extend(f"{prefix}.{detail}" for detail in details)


def _audit_information_flow(
    arrays: TraceArrays,
    row: int,
    helper_record: SlotRoleTransitionRecord,
    beneficiary_record: SlotRoleTransitionRecord,
    world_record: HiddenRegimeWorldTransitionRecord,
    expected_helper_write: bool,
    expected_beneficiary_write: bool,
    mismatches: list[str],
) -> None:
    bindings: tuple[tuple[str, object, object], ...] = (
        (
            "helper_private_input_to_world_cue",
            helper_record.decision.private_input,
            world_record.old_state.cue,
        ),
        (
            "helper_observation_to_world_cue",
            world_record.diagnostics.observation_helper_cue,
            world_record.old_state.cue,
        ),
        ("helper_decision_to_message", helper_record.decision.action, world_record.helper_message),
        (
            "beneficiary_private_input_to_delivery",
            beneficiary_record.decision.private_input,
            world_record.delivered_message,
        ),
        (
            "beneficiary_decision_to_action",
            beneficiary_record.decision.action,
            world_record.beneficiary_action,
        ),
        ("role_reward", helper_record.reward, beneficiary_record.reward),
        ("world_helper_reward", world_record.diagnostics.reward, helper_record.reward),
        ("helper_write_permit", helper_record.external_value_write, expected_helper_write),
        (
            "beneficiary_write_permit",
            beneficiary_record.external_value_write,
            expected_beneficiary_write,
        ),
        (
            "world_step_to_trace_index",
            world_record.old_state.step_count,
            int(arrays["step_index"][row]),
        ),
    )
    for name, actual, expected in bindings:
        if isinstance(actual, float) or isinstance(expected, float):
            equal = _float32_bits(cast(float, actual)) == _float32_bits(cast(float, expected))
        else:
            equal = actual == expected
        if not equal:
            mismatches.append(f"rows[{row}].binding.{name}")


def audit_hidden_regime_trace(audit_input: object) -> HiddenRegimeTraceAuditReport:
    """Audit every primitive row without evaluator replay or production transitions."""

    mismatches: list[str] = []
    envelope = _validate_envelope(audit_input, mismatches)
    if envelope is None:
        return HiddenRegimeTraceAuditReport(
            valid=False,
            expected_steps=0,
            rows_checked=0,
            helper_transitions_checked=0,
            beneficiary_transitions_checked=0,
            world_transitions_checked=0,
            mismatches=tuple(mismatches),
        )

    if mismatches:
        expected_steps = (
            envelope.config.num_steps
            if isinstance(envelope.config, HiddenRegimeDevelopmentConfig)
            else 0
        )
        return HiddenRegimeTraceAuditReport(
            valid=False,
            expected_steps=expected_steps,
            rows_checked=0,
            helper_transitions_checked=0,
            beneficiary_transitions_checked=0,
            world_transitions_checked=0,
            mismatches=tuple(mismatches),
        )

    expected_steps = envelope.config.num_steps
    arrays = _trace_arrays(envelope.trace, expected_steps, mismatches)
    if arrays is None:
        return HiddenRegimeTraceAuditReport(
            valid=False,
            expected_steps=expected_steps,
            rows_checked=0,
            helper_transitions_checked=0,
            beneficiary_transitions_checked=0,
            world_transitions_checked=0,
            mismatches=tuple(mismatches),
        )

    expected_indices = np.arange(expected_steps, dtype=np.int32)
    if not bool(np.array_equal(arrays["step_index"], expected_indices)):
        mismatches.append("trace.step_index.sequence")

    spec = _independent_condition_spec(envelope.condition)
    effective_learner = dataclasses.replace(
        envelope.config.learner,
        writable_lru_ablation=False,
        durable_write_policy=spec.durable_write_policy,
        replacement_target_policy=spec.replacement_target_policy,
    )
    role_config = SlotRoleOracleConfig.from_config(effective_learner)
    world_config = HiddenRegimeWorldOracleConfig.from_config(envelope.config.world)
    delivery_contract = spec.channel
    lifecycle_write = spec.helper_write and spec.beneficiary_write

    initial_helper, initial_beneficiary, initial_world = _initial_states(envelope.seed_pair)
    first_helper = _row_role_state(arrays, 0, ROLE_HELPER, "pre")
    first_beneficiary = _row_role_state(arrays, 0, ROLE_BENEFICIARY, "pre")
    first_world = _row_world_state(arrays, 0, "pre")
    for name, validation in (
        (
            "initial.helper",
            validate_slot_role_state_continuity(initial_helper, first_helper),
        ),
        (
            "initial.beneficiary",
            validate_slot_role_state_continuity(initial_beneficiary, first_beneficiary),
        ),
        (
            "initial.world",
            validate_hidden_regime_world_state_continuity(initial_world, first_world),
        ),
    ):
        _add_validation_mismatches(name, validation.valid, validation.mismatches, mismatches)

    previous_helper: SlotRoleStateSnapshot | None = None
    previous_beneficiary: SlotRoleStateSnapshot | None = None
    previous_world: HiddenRegimeWorldStateSnapshot | None = None
    helper_checked = 0
    beneficiary_checked = 0
    world_checked = 0
    accepted_contractions: list[str] = []

    for row in range(expected_steps):
        helper = _row_role_record(
            arrays,
            row,
            ROLE_HELPER,
            role_config,
            lifecycle_write,
        )
        beneficiary = _row_role_record(
            arrays,
            row,
            ROLE_BENEFICIARY,
            role_config,
            lifecycle_write,
        )
        world = _row_world_record(arrays, row, world_config, delivery_contract)

        if previous_helper is not None:
            continuity = validate_slot_role_state_continuity(previous_helper, helper.old_state)
            _add_validation_mismatches(
                f"rows[{row}].helper", continuity.valid, continuity.mismatches, mismatches
            )
        if previous_beneficiary is not None:
            continuity = validate_slot_role_state_continuity(
                previous_beneficiary, beneficiary.old_state
            )
            _add_validation_mismatches(
                f"rows[{row}].beneficiary",
                continuity.valid,
                continuity.mismatches,
                mismatches,
            )
        if previous_world is not None:
            world_continuity = validate_hidden_regime_world_state_continuity(
                previous_world, world.old_state
            )
            _add_validation_mismatches(
                f"rows[{row}].world",
                world_continuity.valid,
                world_continuity.mismatches,
                mismatches,
            )

        helper_validation = validate_slot_role_transition(helper)
        helper_checked += 1
        helper_contraction = _portable_relevance_contraction(
            helper,
            helper_validation.mismatches,
        )
        if helper_contraction is not None:
            accepted_contractions.append(
                f"rows[{row}].helper.relevance_mean;{helper_contraction}"
            )
        else:
            _add_validation_mismatches(
                f"rows[{row}].helper.oracle",
                helper_validation.valid,
                helper_validation.mismatches,
                mismatches,
            )
        beneficiary_validation = validate_slot_role_transition(beneficiary)
        beneficiary_checked += 1
        beneficiary_contraction = _portable_relevance_contraction(
            beneficiary,
            beneficiary_validation.mismatches,
        )
        if beneficiary_contraction is not None:
            accepted_contractions.append(
                f"rows[{row}].beneficiary.relevance_mean;{beneficiary_contraction}"
            )
        else:
            _add_validation_mismatches(
                f"rows[{row}].beneficiary.oracle",
                beneficiary_validation.valid,
                beneficiary_validation.mismatches,
                mismatches,
            )
        world_validation = validate_hidden_regime_world_transition(world)
        world_checked += 1
        _add_validation_mismatches(
            f"rows[{row}].world.oracle",
            world_validation.valid,
            world_validation.mismatches,
            mismatches,
        )

        _audit_information_flow(
            arrays,
            row,
            helper,
            beneficiary,
            world,
            spec.helper_write,
            spec.beneficiary_write,
            mismatches,
        )

        expected_synchronized = _lifecycle_equal(helper.new_state, beneficiary.new_state)
        if bool(arrays["lifecycle_synchronized"][row]) != expected_synchronized:
            mismatches.append(f"rows[{row}].lifecycle_synchronized")
        for role, record in ((ROLE_HELPER, helper), (ROLE_BENEFICIARY, beneficiary)):
            expected_violation = _expected_selective_violation(
                arrays,
                row,
                role,
                spec.durable_writes_enabled,
            )
            actual_violation = tuple(
                bool(value) for value in arrays[f"{role}_selective_mutation_violation"][row]
            )
            if actual_violation != expected_violation:
                mismatches.append(f"rows[{row}].{role}.selective_mutation_violation")

        previous_helper = helper.new_state
        previous_beneficiary = beneficiary.new_state
        previous_world = world.new_state

    try:
        final_helper = SlotRoleStateSnapshot.from_state(envelope.final_learner_state.helper)
        final_beneficiary = SlotRoleStateSnapshot.from_state(
            envelope.final_learner_state.beneficiary
        )
    except (TypeError, ValueError) as error:
        mismatches.append(f"final.input: {error}")
    else:
        if previous_helper is not None:
            validation = validate_slot_role_state_continuity(previous_helper, final_helper)
            _add_validation_mismatches(
                "final.helper", validation.valid, validation.mismatches, mismatches
            )
        if previous_beneficiary is not None:
            validation = validate_slot_role_state_continuity(
                previous_beneficiary, final_beneficiary
            )
            _add_validation_mismatches(
                "final.beneficiary", validation.valid, validation.mismatches, mismatches
            )

    summary_validation = validate_hidden_regime_summary(
        envelope.trace,
        envelope.config,
        envelope.condition,
        envelope.final_learner_state,
        envelope.summary,
        envelope.resource,
    )
    mismatches.extend(f"derived.{path}" for path in summary_validation.mismatches)
    expected_summary = (
        None
        if summary_validation.expected is None
        else summary_validation.expected.summary
    )

    return HiddenRegimeTraceAuditReport(
        valid=not mismatches,
        expected_steps=expected_steps,
        rows_checked=expected_steps,
        helper_transitions_checked=helper_checked,
        beneficiary_transitions_checked=beneficiary_checked,
        world_transitions_checked=world_checked,
        commit_lineages_checked=(
            0 if expected_summary is None else len(expected_summary.commit_generation_lineages)
        ),
        recurrence_records_checked=(
            0 if expected_summary is None else len(expected_summary.recurrence_retention)
        ),
        retention_aggregate_fields_checked=(
            0
            if expected_summary is None
            else len(dataclasses.fields(expected_summary.retention))
        ),
        summary_fields_checked=summary_validation.summary_fields_checked,
        resource_fields_checked=summary_validation.resource_fields_checked,
        mismatches=tuple(mismatches),
        accepted_float32_contractions=tuple(accepted_contractions),
    )


def audit_hidden_regime_run_result(run: HiddenRegimeRunResult) -> HiddenRegimeTraceAuditReport:
    """Convenience entry point that preserves the run-result provenance envelope."""

    try:
        audit_input = HiddenRegimeTraceAuditInput.from_run_result(run)
    except (TypeError, ValueError) as error:
        return HiddenRegimeTraceAuditReport(
            valid=False,
            expected_steps=0,
            rows_checked=0,
            helper_transitions_checked=0,
            beneficiary_transitions_checked=0,
            world_transitions_checked=0,
            mismatches=(f"input: {error}",),
        )
    return audit_hidden_regime_trace(audit_input)
