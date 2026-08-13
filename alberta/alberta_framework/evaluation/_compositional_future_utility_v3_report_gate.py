"""Pure-stdlib fail-closed gate for the one-shot v3 descriptive report.

The gate accepts only an already-completed, strict-JSON report.  It validates
source bindings, exact endpoint algebra, bounded state-gate claims, exact
reward counts, cross-arm resource accounting, and canonical content hashes.
It cannot issue or consume a root, import an evaluator, execute an arm, write
an artifact, apply a threshold, select a winner/default, or authorize evidence
or scientific promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from typing import Final, cast

REPORT_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-cadence-separated."
    "descriptive-report.v1"
)
REPORT_STATUS: Final = "completed-descriptive-only"
STATE_GATE_SCHEMA: Final = (
    "alberta.compositional-future-utility-state-gate.contribution-endpoint.v1"
)
REWARD_COUNT_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-cadence-separated."
    "exact-reward-counts.v1"
)
ARM_EXECUTION_RECEIPT_SCHEMA: Final = (
    "alberta.compositional-control-life-development.arm-execution-receipt.v1"
)
SOURCE_ARM_NAME: Final = "dovetail_coverage_ancestor_headroom_leftpack"

TOTAL_STEPS: Final = 8_998
CURATION_INTERVAL: Final = 32
TOTAL_CURATION_OPPORTUNITIES: Final = 281
ENTRY_WINDOW: Final = 64
TAIL_WINDOW: Final = 64
RAW_DIM: Final = 6
ACTIVE_SLOTS: Final = 11
CANDIDATE_SLOTS: Final = 8
ACTION_HEADS: Final = 2
PERSISTENT_STATE_NBYTES: Final = 2_072
FINAL_REPLACEMENT_PHASE: Final = 6

PHASE_ORDER: Final = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
PHASE_LENGTHS: Final = (773, 811, 839, 877, 907, 937, 967, 999, 1020, 868)
PHASE_BOUNDARIES: Final = (
    0,
    773,
    1_584,
    2_423,
    3_300,
    4_207,
    5_144,
    6_111,
    7_110,
    8_130,
    8_998,
)
TARGET_NAMES: Final = ("A", "B", "C")
ARM_ORDER: Final = (
    "current_mix0_decay095_none",
    "future_mix1_decay095_none",
    "calibrated_mix05_decay095_none",
    "normalized_mix1_decay095_uncertainty_age",
    "horizon_mix1_decay883_uncertainty_age",
)
PRIMARY_ENDPOINT_ORDER: Final = (
    "margin_passes",
    "promotions",
    "candidate_refreshes",
    "cascade_losses",
    "target_admission_loss_end",
    "pre_recurrence_presence",
    "target_occupancy",
    "pre_recurrence_ranks",
)
REWARD_RECORD_FIELDS: Final = (
    "steps",
    "executed_reward_sum",
    "greedy_reward_sum",
    "executed_action_one_count",
    "greedy_action_one_count",
    "explored_count",
)

_LEARNER_CONFIG_SHA256: Final = (
    "5bca00ecc8a3c14dff9eb1afbd7af2e0d6cfc371e80fad21da4a5239af7548e7",
    "34d98992313753d1e810a22714cd22bf4199cfcdb9359eff1b4e887564ca1392",
    "590a9e5f757cffcc9ca8aac120a57b34ebf7ffce53f57b96974433f3e9c1778f",
    "f1ddcfde6a7d3ed6cf5f238afa95e1846bf2367315c112e5b9cc811d3590a269",
    "defe82edf61c6e7fbbd3f5dce7c4353738bfead2f5e13858245c9ecd393dc12e",
)
_TRACE_DECAY_BITS: Final = ("3f733333", "3f733333", "3f733333", "3f733333", "3f7fcc93")
_RAW_ENERGY_BITS: Final = (1_101_004_788,) * 4 + (1_151_281_462,)
_NORMALIZATION_POLICIES: Final = (
    "disabled-exact-zero",
    "disabled-exact-zero",
    "disabled-exact-zero",
    "enabled-bounded-endpoint",
    "enabled-bounded-endpoint",
)
_STATE_FIELD_MANIFEST_SHA256: Final = (
    "834498ba4ed937d814590c2852d756164a80377124ae11fc15ad22ed17cfc9bd"
)
_STATE_NONCLAIMS: Final = (
    "per-step-contribution-transition-not-proven",
    "candidate-trace-transition-not-proven",
    "mixed-utility-equation-not-proven",
    "normalization-use-in-ranking-not-proven",
    "trace-reset-and-promotion-transfer-not-proven",
)

_STABLE_BINDING_ITEMS: Final = (
    ("development_root", 317_707_403),
    ("development_root_hex", "0x12EFD48B"),
    (
        "protocol_config_sha256",
        "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c",
    ),
    (
        "control_protocol_config_sha256",
        "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca",
    ),
    (
        "runtime_config_sha256",
        "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6",
    ),
    (
        "consumed_history_sha256",
        "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6",
    ),
    (
        "key_manifest_sha256",
        "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5",
    ),
    (
        "stream_sha256",
        "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758",
    ),
    (
        "cadence_bound_stream_sha256",
        "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1",
    ),
    (
        "source_envelope_sha256",
        "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41",
    ),
)

_DYNAMIC_BINDING_FIELDS: Final = (
    "execution_source_closure_sha256",
    "bootstrap_sha256",
    "ledger_primitive_sha256",
    "declared_loader_sha256",
    "genesis_sha256",
    "started_sha256",
)

_RECURRENCE_SCHEDULE: Final = (
    ("A", 2, 2, 1_584),
    ("A", 3, 4, 3_300),
    ("A", 4, 6, 5_144),
    ("B", 2, 7, 6_111),
    ("C", 2, 8, 7_110),
    ("A", 5, 9, 8_130),
)

_MUTATION_MASK_NAMES: Final = (
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
_MUTATION_TOTAL_NAMES: Final = (
    "proposal",
    "root_change",
    "promotion",
    "cascade_refill",
    "ordinary_candidate_refresh",
    "post_promotion_candidate_refresh",
    "candidate_refresh",
    "candidate_rebound",
    "candidate_overdepth_regeneration",
)

WORK_RESOURCE_CONTRACT_SHA256: Final = (
    "67368883be0f864a61da48d14a6bfd0137a32c557129a7aff4f9352c0ed3ee6d"
)
_RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent learner-state bytes, matched shared-base logical cell/update "
    "counts, intervention-specific named cell counts, and measured curation-audit "
    "arrays; excludes behavior-dependent branch work, source arrays, full scan "
    "telemetry, compiler workspaces, and compiled FLOPs"
)


def _validate_json_tree(value: object, path: str = "report") -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        number = value
        if not math.isfinite(number):
            raise ValueError(f"{path} contains a non-finite float")
        if number == 0.0 and math.copysign(1.0, number) < 0.0:
            raise ValueError(f"{path} contains negative zero")
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string key")
            _validate_json_tree(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported exact type {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return the exact ASCII canonical JSON representation of a strict tree."""

    _validate_json_tree(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    """Hash the exact canonical JSON bytes of a strict tree."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object, path: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{path} must be 64 lowercase hexadecimal characters")
    return cast(str, value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_json(value: object, expected: object, message: str) -> None:
    if canonical_json(value) != canonical_json(expected):
        raise ValueError(message)


def _plain_dict(
    value: object,
    path: str,
    fields: tuple[str, ...],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dict")
    record = cast(dict[str, object], value)
    if set(record) != set(fields):
        missing = sorted(set(fields) - set(record))
        extra = sorted(set(record) - set(fields))
        raise ValueError(f"{path} has an inexact field set; missing={missing}, extra={extra}")
    return record


def _plain_list(value: object, path: str, *, length: int | None = None) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{path} must be an exact list")
    items = cast(list[object], value)
    if length is not None and len(items) != length:
        raise ValueError(f"{path} must contain exactly {length} items")
    return items


def _exact_int(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an exact integer")
    result = value
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} is below {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path} is above {maximum}")
    return result


def _exact_bool(value: object, path: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{path} must be exact {expected}")


def _exact_string(value: object, path: str, expected: str | None = None) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = value
    if expected is not None and result != expected:
        raise ValueError(f"{path} differs from its frozen literal")
    return result


def _exact_string_list(value: object, path: str, expected: tuple[str, ...]) -> None:
    items = _plain_list(value, path, length=len(expected))
    if items != list(expected) or any(type(item) is not str for item in items):
        raise ValueError(f"{path} differs from its frozen order")


@dataclasses.dataclass(frozen=True, slots=True)
class ExpectedExecutionBindings:
    """Caller-supplied source/ledger digests unavailable to this pure gate."""

    execution_source_closure_sha256: str
    bootstrap_sha256: str
    ledger_primitive_sha256: str
    declared_loader_sha256: str
    genesis_sha256: str
    started_sha256: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _sha256(getattr(self, field.name), field.name)

    def to_config(self) -> dict[str, object]:
        """Return a fresh strict-JSON binding record."""

        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


def work_resource_contract_config() -> dict[str, object]:
    """Reconstruct the exact engine work-resource contract without importing it."""

    active_cells = TOTAL_STEPS * ACTIVE_SLOTS
    candidate_cells = TOTAL_STEPS * CANDIDATE_SLOTS
    intervention_cells = active_cells + candidate_cells
    shared: dict[str, object] = {
        "learner_updates": TOTAL_STEPS,
        "curation_update_opportunities": TOTAL_CURATION_OPPORTUNITIES,
        "behavior_active_feature_value_cells": active_cells,
        "learner_update_active_feature_value_cells": active_cells,
        "total_active_feature_value_cells": active_cells * 2,
        "learner_update_candidate_feature_value_cells": candidate_cells,
        "evaluator_full_q_dot_products": TOTAL_STEPS,
        "evaluator_raw_q_dot_products": TOTAL_STEPS,
        "learner_prediction_q_dot_products": TOTAL_STEPS,
        "full_and_raw_q_dot_products": TOTAL_STEPS * 2,
        "total_q_dot_products": TOTAL_STEPS * 3,
        "total_q_head_scalar_outputs": TOTAL_STEPS * ACTION_HEADS * 3,
        "ranking_diagnostic_calls": TOTAL_STEPS + 1,
        "active_future_reduction_cells": TOTAL_STEPS * ACTION_HEADS * ACTIVE_SLOTS,
        "candidate_future_reduction_cells": (
            TOTAL_STEPS * ACTION_HEADS * CANDIDATE_SLOTS
        ),
        "future_contribution_trace_cells": (
            TOTAL_STEPS * ACTION_HEADS * (ACTIVE_SLOTS + CANDIDATE_SLOTS)
        ),
        "future_feature_energy_trace_cells": intervention_cells,
        "persistent_candidate_active_correlation_cells": (
            ACTIVE_SLOTS * CANDIDATE_SLOTS
        ),
        "candidate_active_correlation_statistical_accumulation_cells": 0,
        "candidate_active_correlation_reset_mask_cells": (
            TOTAL_STEPS * ACTIVE_SLOTS * CANDIDATE_SLOTS
        ),
        "ranking_candidate_active_correlation_cells": (
            (TOTAL_STEPS + 1) * ACTIVE_SLOTS * CANDIDATE_SLOTS
        ),
        "persistent_state_nbytes": PERSISTENT_STATE_NBYTES,
        "persistent_search_archive_entries": 0,
        "keys_stream_shapes_and_update_opportunities_matched": True,
        "behavioral_experience_matching_claimed": False,
        "compiled_flop_equivalence_claimed": False,
    }

    def intervention(*, mixture: bool, normalized: bool) -> dict[str, object]:
        return {
            "utility_mixture_cells": intervention_cells if mixture else 0,
            "active_second_moment_cells": active_cells if normalized else 0,
            "candidate_second_moment_cells": candidate_cells if normalized else 0,
            "active_age_debias_cells": active_cells if normalized else 0,
            "candidate_age_debias_cells": candidate_cells if normalized else 0,
            "active_uncertainty_normalization_cells": (
                active_cells if normalized else 0
            ),
            "candidate_uncertainty_normalization_cells": (
                candidate_cells if normalized else 0
            ),
        }

    interventions = {
        ARM_ORDER[0]: intervention(mixture=False, normalized=False),
        ARM_ORDER[1]: intervention(mixture=True, normalized=False),
        ARM_ORDER[2]: intervention(mixture=True, normalized=False),
        ARM_ORDER[3]: intervention(mixture=True, normalized=True),
        ARM_ORDER[4]: intervention(mixture=True, normalized=True),
    }
    return {
        "selected_arm_count": len(ARM_ORDER),
        "per_arm_shared_base": shared,
        "intervention_specific_per_arm": interventions,
        "panel_learner_updates": TOTAL_STEPS * len(ARM_ORDER),
        "panel_curation_update_opportunities": (
            TOTAL_CURATION_OPPORTUNITIES * len(ARM_ORDER)
        ),
        "aggregate_arm_state_byte_equivalent": (
            PERSISTENT_STATE_NBYTES * len(ARM_ORDER)
        ),
        "aggregate_arm_state_byte_equivalent_is_peak_memory": False,
        "accounting_scope": _RESOURCE_ACCOUNTING_SCOPE,
        "shared_base_logical_work_matched": True,
        "stream_shapes_and_update_opportunities_matched": True,
        "intervention_specific_logical_work_matched": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "persistent_shapes_matched": True,
        "source_array_bytes_included": False,
        "full_scan_telemetry_bytes_included": False,
        "compiler_workspace_bytes_included": False,
        "compiled_flop_equivalence_claimed": False,
        "behavioral_experience_matching_claimed": False,
    }


_STATE_RECEIPT_FIELDS: Final = (
    "schema",
    "steps",
    "trace_decay_f32_bits",
    "expected_raw_energy_f32_bits",
    "normalization_moment_policy",
    "field_manifest_sha256",
    "initial_subset_sha256",
    "final_subset_sha256",
    "initial_fields_all_zero",
    "all_fields_finite",
    "contribution_mode_zero_marginal_traces",
    "raw_slots_untouched_by_curation",
    "raw_energy_bits_exact",
    "normalization_moment_policy_exact",
    "utility_event_final_rows_exact",
    "nonclaims",
    "development_only",
    "panel_execution_authorized",
    "result_authorized",
    "output_writes_allowed",
    "evidence_authorized",
    "scientific_promotion_allowed",
)
_REWARD_PROJECTION_FIELDS: Final = (
    "schema",
    "phase_order",
    "lifetime",
    "whole_phases",
    "entry_windows",
    "tail_windows",
    "experience_semantics_validated",
    "development_only",
    "execution_authorized",
    "output_writes_allowed",
    "evidence_authorized",
    "scientific_promotion_allowed",
)


def _validate_state_receipt(
    value: object,
    *,
    arm_index: int,
) -> tuple[str, str]:
    path = f"report.runs[{arm_index}].state_gate_receipt"
    receipt = _plain_dict(value, path, _STATE_RECEIPT_FIELDS)
    _exact_string(receipt["schema"], f"{path}.schema", STATE_GATE_SCHEMA)
    _require(
        _exact_int(receipt["steps"], f"{path}.steps") == TOTAL_STEPS,
        f"{path}.steps differs from v3",
    )
    _exact_string(
        receipt["trace_decay_f32_bits"],
        f"{path}.trace_decay_f32_bits",
        _TRACE_DECAY_BITS[arm_index],
    )
    _require(
        _exact_int(
            receipt["expected_raw_energy_f32_bits"],
            f"{path}.expected_raw_energy_f32_bits",
        )
        == _RAW_ENERGY_BITS[arm_index],
        f"{path}.expected_raw_energy_f32_bits differs from its mechanism pin",
    )
    _exact_string(
        receipt["normalization_moment_policy"],
        f"{path}.normalization_moment_policy",
        _NORMALIZATION_POLICIES[arm_index],
    )
    _exact_string(
        receipt["field_manifest_sha256"],
        f"{path}.field_manifest_sha256",
        _STATE_FIELD_MANIFEST_SHA256,
    )
    initial_subset = _sha256(
        receipt["initial_subset_sha256"],
        f"{path}.initial_subset_sha256",
    )
    final_subset = _sha256(
        receipt["final_subset_sha256"],
        f"{path}.final_subset_sha256",
    )
    _require(initial_subset != final_subset, f"{path} state subset did not advance")
    for field in (
        "initial_fields_all_zero",
        "all_fields_finite",
        "contribution_mode_zero_marginal_traces",
        "raw_slots_untouched_by_curation",
        "raw_energy_bits_exact",
        "normalization_moment_policy_exact",
        "utility_event_final_rows_exact",
        "development_only",
    ):
        _exact_bool(receipt[field], f"{path}.{field}", True)
    for field in (
        "panel_execution_authorized",
        "result_authorized",
        "output_writes_allowed",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        _exact_bool(receipt[field], f"{path}.{field}", False)
    _exact_string_list(receipt["nonclaims"], f"{path}.nonclaims", _STATE_NONCLAIMS)
    return initial_subset, final_subset


def _validate_reward_record(
    value: object,
    path: str,
    *,
    expected_steps: int,
) -> dict[str, int]:
    record = _plain_dict(value, path, REWARD_RECORD_FIELDS)
    values = {
        field: _exact_int(record[field], f"{path}.{field}")
        for field in REWARD_RECORD_FIELDS
    }
    steps = values["steps"]
    _require(steps == expected_steps, f"{path}.steps differs from its exact window")
    for field in ("executed_reward_sum", "greedy_reward_sum"):
        reward_sum = values[field]
        _require(
            -steps <= reward_sum <= steps,
            f"{path}.{field} is outside the exact binary-reward range",
        )
        _require(
            (steps + reward_sum) % 2 == 0,
            f"{path}.{field} violates exact binary-reward parity",
        )
    for field in (
        "executed_action_one_count",
        "greedy_action_one_count",
        "explored_count",
    ):
        _require(
            0 <= values[field] <= steps,
            f"{path}.{field} is outside its exact count bounds",
        )
    return values


def _validate_reward_projection(value: object, *, arm_index: int) -> None:
    path = f"report.runs[{arm_index}].reward_counts"
    projection = _plain_dict(value, path, _REWARD_PROJECTION_FIELDS)
    _exact_string(projection["schema"], f"{path}.schema", REWARD_COUNT_SCHEMA)
    _exact_string_list(projection["phase_order"], f"{path}.phase_order", PHASE_ORDER)
    lifetime = _validate_reward_record(
        projection["lifetime"],
        f"{path}.lifetime",
        expected_steps=TOTAL_STEPS,
    )
    whole_values = _plain_list(
        projection["whole_phases"],
        f"{path}.whole_phases",
        length=len(PHASE_ORDER),
    )
    phases = [
        _validate_reward_record(
            record,
            f"{path}.whole_phases[{index}]",
            expected_steps=PHASE_LENGTHS[index],
        )
        for index, record in enumerate(whole_values)
    ]
    entry_values = _plain_list(
        projection["entry_windows"],
        f"{path}.entry_windows",
        length=len(PHASE_ORDER),
    )
    tail_values = _plain_list(
        projection["tail_windows"],
        f"{path}.tail_windows",
        length=len(PHASE_ORDER),
    )
    for index, record in enumerate(entry_values):
        _validate_reward_record(
            record,
            f"{path}.entry_windows[{index}]",
            expected_steps=ENTRY_WINDOW,
        )
    for index, record in enumerate(tail_values):
        _validate_reward_record(
            record,
            f"{path}.tail_windows[{index}]",
            expected_steps=TAIL_WINDOW,
        )
    for field in REWARD_RECORD_FIELDS:
        _require(
            lifetime[field] == sum(phase[field] for phase in phases),
            f"{path}.lifetime.{field} does not equal the whole-phase sum",
        )
    for field, expected in (
        ("experience_semantics_validated", True),
        ("development_only", True),
        ("execution_authorized", False),
        ("output_writes_allowed", False),
        ("evidence_authorized", False),
        ("scientific_promotion_allowed", False),
    ):
        _exact_bool(projection[field], f"{path}.{field}", expected)


_RANK_RECORD_FIELDS: Final = (
    "present",
    "matching_score_f32_bits",
    "best_score_f32_bits",
    "descending_rank_interval",
)
_RECURRENCE_RECORD_FIELDS: Final = (
    "target",
    "occurrence",
    "recurrence_phase_index",
    "pre_recurrence_post_step",
    "active_present",
    "candidate_present",
    "active_slot_count",
    "candidate_slot_count",
    "matching_active_slots",
    "matching_candidate_slots",
    "direct_rank",
    "ancestor_backed_rank",
    "candidate_direct_rank",
    "candidate_augmented_rank",
)
_PRESENCE_RECORD_FIELDS: Final = (
    "target",
    "occurrence",
    "pre_recurrence_post_step",
    "active_present",
    "candidate_present",
    "active_slot_count",
    "candidate_slot_count",
)


def _f32_from_uint_bits(value: int) -> float:
    return float(struct.unpack(">f", struct.pack(">I", value))[0])


def _slot_list(
    value: object,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> list[int]:
    raw = _plain_list(value, path)
    slots = [
        _exact_int(item, f"{path}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(raw)
    ]
    _require(slots == sorted(set(slots)), f"{path} must be sorted and unique")
    return slots


def _validate_rank(
    value: object,
    path: str,
    *,
    slot_field: str,
    expected_slots: list[int],
    bank_capacity: int,
) -> None:
    fields = (*_RANK_RECORD_FIELDS[:1], slot_field, *_RANK_RECORD_FIELDS[1:])
    rank = _plain_dict(value, path, fields)
    present = rank["present"]
    if type(present) is not bool:
        raise TypeError(f"{path}.present must be an exact bool")
    slots = _slot_list(
        rank[slot_field],
        f"{path}.{slot_field}",
        minimum=RAW_DIM if slot_field == "matching_composed_slots" else 0,
        maximum=(ACTIVE_SLOTS - 1 if slot_field == "matching_composed_slots" else 7),
    )
    _require(slots == expected_slots, f"{path}.{slot_field} differs from structure")
    raw_bits = _plain_list(
        rank["matching_score_f32_bits"],
        f"{path}.matching_score_f32_bits",
    )
    score_bits = [
        _exact_int(
            item,
            f"{path}.matching_score_f32_bits[{index}]",
            minimum=0,
            maximum=0xFFFFFFFF,
        )
        for index, item in enumerate(raw_bits)
    ]
    _require(len(score_bits) == len(slots), f"{path} score/slot lengths differ")
    if present is False:
        _require(not slots and not score_bits, f"{path} absent rank must have no matches")
        _require(
            rank["best_score_f32_bits"] is None,
            f"{path}.best_score_f32_bits must be null when absent",
        )
        _require(
            rank["descending_rank_interval"] is None,
            f"{path}.descending_rank_interval must be null when absent",
        )
        return
    _require(bool(slots), f"{path} present rank must have a matching slot")
    scores = [_f32_from_uint_bits(bits) for bits in score_bits]
    _require(all(math.isfinite(score) for score in scores), f"{path} scores are not finite")
    best_bits = _exact_int(
        rank["best_score_f32_bits"],
        f"{path}.best_score_f32_bits",
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    best_score = _f32_from_uint_bits(best_bits)
    _require(math.isfinite(best_score), f"{path}.best_score_f32_bits is non-finite")
    _require(best_bits in score_bits, f"{path} best-score bits are not a matching score")
    _require(best_score == max(scores), f"{path} best score does not close")
    interval = _plain_list(
        rank["descending_rank_interval"],
        f"{path}.descending_rank_interval",
        length=2,
    )
    lower = _exact_int(interval[0], f"{path}.descending_rank_interval[0]", minimum=1)
    upper = _exact_int(
        interval[1],
        f"{path}.descending_rank_interval[1]",
        minimum=lower,
        maximum=bank_capacity,
    )
    _require(lower <= upper, f"{path}.descending_rank_interval is reversed")


def _validate_recurrence_records(
    presence_value: object,
    rank_value: object,
    *,
    path: str,
) -> list[dict[str, object]]:
    presence_records = _plain_list(
        presence_value,
        f"{path}.pre_recurrence_presence",
        length=len(_RECURRENCE_SCHEDULE),
    )
    rank_container = _plain_dict(
        rank_value,
        f"{path}.pre_recurrence_ranks",
        ("active_definition", "candidate_definition", "records"),
    )
    _exact_string(
        rank_container["active_definition"],
        f"{path}.pre_recurrence_ranks.active_definition",
        (
            "best matching target slot among composed slots RAW_DIM:ACTIVE_SLOTS; "
            "tie-aware descending rank interval, with rank 1 highest"
        ),
    )
    _exact_string(
        rank_container["candidate_definition"],
        f"{path}.pre_recurrence_ranks.candidate_definition",
        (
            "best matching target slot among all candidate slots; direct and "
            "novelty-augmented scores each use a tie-aware descending rank interval, "
            "with rank 1 highest"
        ),
    )
    rank_records = _plain_list(
        rank_container["records"],
        f"{path}.pre_recurrence_ranks.records",
        length=len(_RECURRENCE_SCHEDULE),
    )
    validated: list[dict[str, object]] = []
    for index, schedule in enumerate(_RECURRENCE_SCHEDULE):
        target, occurrence, phase_index, post_step = schedule
        presence_path = f"{path}.pre_recurrence_presence[{index}]"
        rank_path = f"{path}.pre_recurrence_ranks.records[{index}]"
        presence = _plain_dict(
            presence_records[index],
            presence_path,
            _PRESENCE_RECORD_FIELDS,
        )
        record = _plain_dict(rank_records[index], rank_path, _RECURRENCE_RECORD_FIELDS)
        expected_common = {
            "target": target,
            "occurrence": occurrence,
            "pre_recurrence_post_step": post_step,
        }
        _exact_string(presence["target"], f"{presence_path}.target", target)
        _exact_string(record["target"], f"{rank_path}.target", target)
        for field in ("occurrence", "pre_recurrence_post_step"):
            expected = cast(int, expected_common[field])
            _require(
                _exact_int(presence[field], f"{presence_path}.{field}") == expected,
                f"{presence_path}.{field} differs",
            )
            _require(
                _exact_int(record[field], f"{rank_path}.{field}") == expected,
                f"{rank_path}.{field} differs",
            )
        _require(
            _exact_int(
                record["recurrence_phase_index"],
                f"{rank_path}.recurrence_phase_index",
            )
            == phase_index,
            f"{rank_path}.recurrence_phase_index differs",
        )
        for field in (
            "active_present",
            "candidate_present",
            "active_slot_count",
            "candidate_slot_count",
        ):
            _require_exact_json(
                presence[field],
                record[field],
                f"{presence_path}.{field} differs from rank structure",
            )
        active_present = record["active_present"]
        candidate_present = record["candidate_present"]
        if type(active_present) is not bool or type(candidate_present) is not bool:
            raise TypeError(f"{rank_path} presence flags must be exact bools")
        active_count = _exact_int(
            record["active_slot_count"],
            f"{rank_path}.active_slot_count",
            minimum=0,
            maximum=ACTIVE_SLOTS - RAW_DIM,
        )
        candidate_count = _exact_int(
            record["candidate_slot_count"],
            f"{rank_path}.candidate_slot_count",
            minimum=0,
            maximum=CANDIDATE_SLOTS,
        )
        active_slots = _slot_list(
            record["matching_active_slots"],
            f"{rank_path}.matching_active_slots",
            minimum=RAW_DIM,
            maximum=ACTIVE_SLOTS - 1,
        )
        candidate_slots = _slot_list(
            record["matching_candidate_slots"],
            f"{rank_path}.matching_candidate_slots",
            minimum=0,
            maximum=CANDIDATE_SLOTS - 1,
        )
        _require(active_count == len(active_slots), f"{rank_path} active count differs")
        _require(
            candidate_count == len(candidate_slots),
            f"{rank_path} candidate count differs",
        )
        _require(active_present is bool(active_slots), f"{rank_path} active presence differs")
        _require(
            candidate_present is bool(candidate_slots),
            f"{rank_path} candidate presence differs",
        )
        _validate_rank(
            record["direct_rank"],
            f"{rank_path}.direct_rank",
            slot_field="matching_composed_slots",
            expected_slots=active_slots,
            bank_capacity=ACTIVE_SLOTS - RAW_DIM,
        )
        _validate_rank(
            record["ancestor_backed_rank"],
            f"{rank_path}.ancestor_backed_rank",
            slot_field="matching_composed_slots",
            expected_slots=active_slots,
            bank_capacity=ACTIVE_SLOTS - RAW_DIM,
        )
        _validate_rank(
            record["candidate_direct_rank"],
            f"{rank_path}.candidate_direct_rank",
            slot_field="matching_candidate_slots",
            expected_slots=candidate_slots,
            bank_capacity=CANDIDATE_SLOTS,
        )
        _validate_rank(
            record["candidate_augmented_rank"],
            f"{rank_path}.candidate_augmented_rank",
            slot_field="matching_candidate_slots",
            expected_slots=candidate_slots,
            bank_capacity=CANDIDATE_SLOTS,
        )
        validated.append(record)
    return validated


def _target_dict(value: object, path: str) -> dict[str, object]:
    return _plain_dict(value, path, TARGET_NAMES)


def _validate_target_lifecycle(
    lifecycle_value: object,
    cascade_value: object,
    *,
    path: str,
) -> tuple[dict[str, bool], dict[str, int]]:
    lifecycles = _target_dict(lifecycle_value, f"{path}.target_admission_loss_end")
    cascade_losses = _target_dict(cascade_value, f"{path}.cascade_losses")
    present_at_end: dict[str, bool] = {}
    totals = {
        "direct_admission": 0,
        "loss_episode": 0,
        "root_lost_slot": 0,
        "cascade_lost_slot": 0,
    }
    for target in TARGET_NAMES:
        target_path = f"{path}.target_admission_loss_end.{target}"
        lifecycle = _plain_dict(
            lifecycles[target],
            target_path,
            (
                "direct_candidate_admission_count",
                "admission_episode_count",
                "loss_episode_count",
                "present_at_end",
                "structural_reacquisition_count",
            ),
        )
        direct = _exact_int(
            lifecycle["direct_candidate_admission_count"],
            f"{target_path}.direct_candidate_admission_count",
            minimum=0,
            maximum=TOTAL_CURATION_OPPORTUNITIES,
        )
        acquisitions = _exact_int(
            lifecycle["admission_episode_count"],
            f"{target_path}.admission_episode_count",
            minimum=0,
            maximum=TOTAL_CURATION_OPPORTUNITIES,
        )
        losses = _exact_int(
            lifecycle["loss_episode_count"],
            f"{target_path}.loss_episode_count",
            minimum=0,
            maximum=TOTAL_CURATION_OPPORTUNITIES,
        )
        end = lifecycle["present_at_end"]
        if type(end) is not bool:
            raise TypeError(f"{target_path}.present_at_end must be an exact bool")
        reacquisitions = _exact_int(
            lifecycle["structural_reacquisition_count"],
            f"{target_path}.structural_reacquisition_count",
            minimum=0,
        )
        _require(direct >= acquisitions, f"{target_path} episodes exceed admissions")
        _require(
            acquisitions - losses == int(end),
            f"{target_path} admission/loss/end lifecycle does not close",
        )
        _require(
            reacquisitions == max(0, acquisitions - 1),
            f"{target_path} structural reacquisition count does not close",
        )
        cascade_path = f"{path}.cascade_losses.{target}"
        cascade = _plain_dict(
            cascade_losses[target],
            cascade_path,
            (
                "loss_episode_count",
                "root_replacement_lost_slot_count",
                "cascade_dependency_refill_lost_slot_count",
                "all_changed_slots_accounted",
            ),
        )
        _require(
            _exact_int(
                cascade["loss_episode_count"],
                f"{cascade_path}.loss_episode_count",
                minimum=0,
            )
            == losses,
            f"{cascade_path}.loss_episode_count differs from lifecycle",
        )
        root_lost = _exact_int(
            cascade["root_replacement_lost_slot_count"],
            f"{cascade_path}.root_replacement_lost_slot_count",
            minimum=0,
        )
        cascade_lost = _exact_int(
            cascade["cascade_dependency_refill_lost_slot_count"],
            f"{cascade_path}.cascade_dependency_refill_lost_slot_count",
            minimum=0,
        )
        _exact_bool(
            cascade["all_changed_slots_accounted"],
            f"{cascade_path}.all_changed_slots_accounted",
            True,
        )
        present_at_end[target] = end
        totals["direct_admission"] += direct
        totals["loss_episode"] += losses
        totals["root_lost_slot"] += root_lost
        totals["cascade_lost_slot"] += cascade_lost
    return present_at_end, totals


def _fraction(value: object, numerator: int, denominator: int, path: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{path} must be an exact float")
    _require(
        value == numerator / denominator,
        f"{path} does not equal its exact count fraction",
    )


def _validate_occupancy(
    value: object,
    *,
    present_at_end: dict[str, bool],
    path: str,
) -> tuple[dict[str, int], dict[str, int]]:
    occupancy = _plain_dict(
        value,
        f"{path}.target_occupancy",
        (
            "post_update_state_count",
            "per_target",
            "coexistence",
            "steps_by_distinct_active_target_count",
            "maximum_distinct_active_target_count",
            "final_active_targets",
        ),
    )
    _require(
        _exact_int(
            occupancy["post_update_state_count"],
            f"{path}.target_occupancy.post_update_state_count",
        )
        == TOTAL_STEPS,
        f"{path}.target_occupancy.post_update_state_count differs",
    )
    per_target = _target_dict(
        occupancy["per_target"],
        f"{path}.target_occupancy.per_target",
    )
    active_presence_total = 0
    active_by_target: dict[str, int] = {}
    candidate_by_target: dict[str, int] = {}
    for target in TARGET_NAMES:
        target_path = f"{path}.target_occupancy.per_target.{target}"
        record = _plain_dict(
            per_target[target],
            target_path,
            (
                "active_present_post_steps",
                "active_presence_fraction",
                "active_slot_step_cells",
                "candidate_present_post_steps",
                "candidate_presence_fraction",
                "candidate_slot_step_cells",
            ),
        )
        active_present = _exact_int(
            record["active_present_post_steps"],
            f"{target_path}.active_present_post_steps",
            minimum=0,
            maximum=TOTAL_STEPS,
        )
        active_cells = _exact_int(
            record["active_slot_step_cells"],
            f"{target_path}.active_slot_step_cells",
            minimum=0,
            maximum=TOTAL_STEPS * (ACTIVE_SLOTS - RAW_DIM),
        )
        candidate_present = _exact_int(
            record["candidate_present_post_steps"],
            f"{target_path}.candidate_present_post_steps",
            minimum=0,
            maximum=TOTAL_STEPS,
        )
        candidate_cells = _exact_int(
            record["candidate_slot_step_cells"],
            f"{target_path}.candidate_slot_step_cells",
            minimum=0,
            maximum=TOTAL_STEPS * CANDIDATE_SLOTS,
        )
        _fraction(
            record["active_presence_fraction"],
            active_present,
            TOTAL_STEPS,
            f"{target_path}.active_presence_fraction",
        )
        _fraction(
            record["candidate_presence_fraction"],
            candidate_present,
            TOTAL_STEPS,
            f"{target_path}.candidate_presence_fraction",
        )
        _require(
            active_present <= active_cells <= active_present * (ACTIVE_SLOTS - RAW_DIM),
            f"{target_path} active slot-cell occupancy does not close",
        )
        _require(
            candidate_present <= candidate_cells <= candidate_present * CANDIDATE_SLOTS,
            f"{target_path} candidate slot-cell occupancy does not close",
        )
        if present_at_end[target]:
            _require(active_present > 0, f"{target_path} end presence lacks occupancy")
        active_by_target[target] = active_present
        candidate_by_target[target] = candidate_present
        active_presence_total += active_present
    coexistence_path = f"{path}.target_occupancy.coexistence"
    coexistence = _plain_dict(
        occupancy["coexistence"],
        coexistence_path,
        (
            "target_order",
            "steps",
            "steps_by_active_target_count",
            "maximum_active_target_count",
            "all_targets_present_steps",
            "all_targets_presence_fraction",
            "first_all_targets_post_step",
            "last_all_targets_post_step",
            "active_targets_at_end",
        ),
    )
    _exact_string_list(
        coexistence["target_order"],
        f"{coexistence_path}.target_order",
        TARGET_NAMES,
    )
    _require(
        _exact_int(coexistence["steps"], f"{coexistence_path}.steps") == TOTAL_STEPS,
        f"{coexistence_path}.steps differs",
    )
    raw_histogram = _plain_list(
        coexistence["steps_by_active_target_count"],
        f"{coexistence_path}.steps_by_active_target_count",
        length=len(TARGET_NAMES) + 1,
    )
    histogram = [
        _exact_int(
            count,
            f"{coexistence_path}.steps_by_active_target_count[{index}]",
            minimum=0,
            maximum=TOTAL_STEPS,
        )
        for index, count in enumerate(raw_histogram)
    ]
    _require(sum(histogram) == TOTAL_STEPS, f"{coexistence_path} histogram does not close")
    maximum = _exact_int(
        coexistence["maximum_active_target_count"],
        f"{coexistence_path}.maximum_active_target_count",
        minimum=0,
        maximum=len(TARGET_NAMES),
    )
    derived_maximum = max(index for index, count in enumerate(histogram) if count > 0)
    _require(maximum == derived_maximum, f"{coexistence_path} maximum does not close")
    _require(
        active_presence_total == sum(index * count for index, count in enumerate(histogram)),
        f"{coexistence_path} target occupancy cells do not close",
    )
    all_target_steps = _exact_int(
        coexistence["all_targets_present_steps"],
        f"{coexistence_path}.all_targets_present_steps",
        minimum=0,
        maximum=TOTAL_STEPS,
    )
    _require(all_target_steps == histogram[-1], f"{coexistence_path} all-target count differs")
    _fraction(
        coexistence["all_targets_presence_fraction"],
        all_target_steps,
        TOTAL_STEPS,
        f"{coexistence_path}.all_targets_presence_fraction",
    )
    first = coexistence["first_all_targets_post_step"]
    last = coexistence["last_all_targets_post_step"]
    if all_target_steps == 0:
        _require(first is None and last is None, f"{coexistence_path} timings must be null")
    else:
        first_step = _exact_int(
            first,
            f"{coexistence_path}.first_all_targets_post_step",
            minimum=1,
            maximum=TOTAL_STEPS,
        )
        last_step = _exact_int(
            last,
            f"{coexistence_path}.last_all_targets_post_step",
            minimum=first_step,
            maximum=TOTAL_STEPS,
        )
        _require(first_step <= last_step, f"{coexistence_path} timings are reversed")
    final_targets = tuple(target for target in TARGET_NAMES if present_at_end[target])
    _exact_string_list(
        coexistence["active_targets_at_end"],
        f"{coexistence_path}.active_targets_at_end",
        final_targets,
    )
    _require_exact_json(
        occupancy["steps_by_distinct_active_target_count"],
        raw_histogram,
        f"{path}.target_occupancy histogram alias differs",
    )
    _require(
        _exact_int(
            occupancy["maximum_distinct_active_target_count"],
            f"{path}.target_occupancy.maximum_distinct_active_target_count",
        )
        == maximum,
        f"{path}.target_occupancy maximum alias differs",
    )
    _require_exact_json(
        occupancy["final_active_targets"],
        list(final_targets),
        f"{path}.target_occupancy final-target alias differs",
    )
    return active_by_target, candidate_by_target


def _validate_target_retention(
    value: object,
    *,
    recurrence_records: list[dict[str, object]],
    present_at_end: dict[str, bool],
    path: str,
) -> None:
    retention = _target_dict(value, f"{path}.target_retention")
    for target in TARGET_NAMES:
        target_path = f"{path}.target_retention.{target}"
        record = _plain_dict(
            retention[target],
            target_path,
            (
                "pre_recurrence_phase_indices",
                "pre_recurrence_presence",
                "present_at_end",
            ),
        )
        target_records = [
            recurrence
            for recurrence in recurrence_records
            if recurrence["target"] == target
        ]
        expected_indices = [
            cast(int, recurrence["recurrence_phase_index"])
            for recurrence in target_records
        ]
        expected_presence = [
            cast(bool, recurrence["active_present"])
            for recurrence in target_records
        ]
        _require_exact_json(
            record["pre_recurrence_phase_indices"],
            expected_indices,
            f"{target_path}.pre_recurrence_phase_indices differs",
        )
        _require_exact_json(
            record["pre_recurrence_presence"],
            expected_presence,
            f"{target_path}.pre_recurrence_presence differs",
        )
        _exact_bool(
            record["present_at_end"],
            f"{target_path}.present_at_end",
            present_at_end[target],
        )


_PARTITION_FIELDS: Final = (
    "all_step_count",
    "due_opportunity_count",
    "off_opportunity_count",
)


def _validate_partition(
    value: object,
    path: str,
    *,
    maximum_all: int,
    require_zero_off: bool,
) -> dict[str, int]:
    partition = _plain_dict(value, path, _PARTITION_FIELDS)
    counts = {
        field: _exact_int(
            partition[field],
            f"{path}.{field}",
            minimum=0,
            maximum=maximum_all,
        )
        for field in _PARTITION_FIELDS
    }
    _require(
        counts["all_step_count"]
        == counts["due_opportunity_count"] + counts["off_opportunity_count"],
        f"{path} all/due/off counts do not close",
    )
    if require_zero_off:
        _require(counts["off_opportunity_count"] == 0, f"{path} mutated off cadence")
    return counts


def _mutation_width(name: str) -> int:
    if name in {
        "root_change_mask",
        "cascade_refill_mask",
        "active_change_mask",
    }:
        return ACTIVE_SLOTS
    if name in {
        "ordinary_candidate_refresh_mask",
        "post_promotion_candidate_refresh_mask",
        "candidate_refresh_mask",
        "candidate_rebound_mask",
        "candidate_overdepth_regeneration_mask",
    }:
        return CANDIDATE_SLOTS
    return 1


def _validate_cadence_integrity(value: object, *, path: str) -> dict[str, int]:
    cadence_path = f"{path}.cadence_integrity"
    cadence = _plain_dict(
        value,
        cadence_path,
        (
            "diagnostic_partitions",
            "mutation_partitions",
            "all_mutations_off_opportunity_count",
            "curation_counts_close",
            "curation_count_closure",
            "eventwise_curation_closure",
        ),
    )
    diagnostic = _plain_dict(
        cadence["diagnostic_partitions"],
        f"{cadence_path}.diagnostic_partitions",
        ("decision_margin_passed", "decision_candidate_margin_eligible"),
    )
    selected_margin = _validate_partition(
        diagnostic["decision_margin_passed"],
        f"{cadence_path}.diagnostic_partitions.decision_margin_passed",
        maximum_all=TOTAL_STEPS,
        require_zero_off=False,
    )
    candidate_margin = _validate_partition(
        diagnostic["decision_candidate_margin_eligible"],
        f"{cadence_path}.diagnostic_partitions.decision_candidate_margin_eligible",
        maximum_all=TOTAL_STEPS * CANDIDATE_SLOTS * ACTIVE_SLOTS,
        require_zero_off=False,
    )
    mutations = _plain_dict(
        cadence["mutation_partitions"],
        f"{cadence_path}.mutation_partitions",
        _MUTATION_MASK_NAMES,
    )
    mutation_counts: dict[str, dict[str, int]] = {}
    for name in _MUTATION_MASK_NAMES:
        mutation_counts[name] = _validate_partition(
            mutations[name],
            f"{cadence_path}.mutation_partitions.{name}",
            maximum_all=TOTAL_CURATION_OPPORTUNITIES * _mutation_width(name),
            require_zero_off=True,
        )
    _require(
        _exact_int(
            cadence["all_mutations_off_opportunity_count"],
            f"{cadence_path}.all_mutations_off_opportunity_count",
        )
        == 0,
        f"{cadence_path}.all_mutations_off_opportunity_count must be zero",
    )
    _exact_bool(
        cadence["curation_counts_close"],
        f"{cadence_path}.curation_counts_close",
        True,
    )
    closure_path = f"{cadence_path}.curation_count_closure"
    closure = _plain_dict(
        cadence["curation_count_closure"],
        closure_path,
        (
            "all_checked_counts_close",
            "curation_due_count",
            "mutation_counts",
            "logical_event_count",
            "event_bearing_opportunity_count",
        ),
    )
    _exact_bool(
        closure["all_checked_counts_close"],
        f"{closure_path}.all_checked_counts_close",
        True,
    )
    _require(
        _exact_int(
            closure["curation_due_count"],
            f"{closure_path}.curation_due_count",
        )
        == TOTAL_CURATION_OPPORTUNITIES,
        f"{closure_path}.curation_due_count differs from v3 cadence",
    )
    totals_raw = _plain_dict(
        closure["mutation_counts"],
        f"{closure_path}.mutation_counts",
        _MUTATION_TOTAL_NAMES,
    )
    total_to_mask = {
        "proposal": "proposal_formed",
        "root_change": "root_change_mask",
        "promotion": "promotion_applied",
        "cascade_refill": "cascade_refill_mask",
        "ordinary_candidate_refresh": "ordinary_candidate_refresh_mask",
        "post_promotion_candidate_refresh": "post_promotion_candidate_refresh_mask",
        "candidate_refresh": "candidate_refresh_mask",
        "candidate_rebound": "candidate_rebound_mask",
        "candidate_overdepth_regeneration": "candidate_overdepth_regeneration_mask",
    }
    totals: dict[str, int] = {}
    for name, mask_name in total_to_mask.items():
        count = _exact_int(
            totals_raw[name],
            f"{closure_path}.mutation_counts.{name}",
            minimum=0,
        )
        _require(
            count == mutation_counts[mask_name]["due_opportunity_count"],
            f"{closure_path}.mutation_counts.{name} differs from its mask",
        )
        totals[name] = count
    logical = _exact_int(
        closure["logical_event_count"],
        f"{closure_path}.logical_event_count",
        minimum=0,
    )
    event_bearing = _exact_int(
        closure["event_bearing_opportunity_count"],
        f"{closure_path}.event_bearing_opportunity_count",
        minimum=0,
        maximum=TOTAL_CURATION_OPPORTUNITIES,
    )
    _require(
        totals["proposal"]
        == totals["promotion"] + totals["ordinary_candidate_refresh"],
        f"{closure_path} proposal count does not close",
    )
    _require(
        totals["post_promotion_candidate_refresh"] == totals["promotion"],
        f"{closure_path} post-promotion refresh count does not close",
    )
    _require(
        totals["candidate_refresh"]
        == totals["ordinary_candidate_refresh"]
        + totals["post_promotion_candidate_refresh"],
        f"{closure_path} candidate refresh union does not close",
    )
    _require(
        mutation_counts["decision_should_promote"]["due_opportunity_count"]
        == totals["promotion"],
        f"{closure_path} should-promote count differs",
    )
    _require(
        mutation_counts["root_change_applied"]["due_opportunity_count"]
        == totals["root_change"],
        f"{closure_path} root-change event count differs",
    )
    _require(
        mutation_counts["decision_should_refresh"]["due_opportunity_count"]
        == totals["ordinary_candidate_refresh"],
        f"{closure_path} should-refresh count differs",
    )
    _require(
        mutation_counts["active_change_mask"]["due_opportunity_count"]
        == totals["root_change"] + totals["cascade_refill"],
        f"{closure_path} active-change union does not close",
    )
    expected_logical = (
        totals["root_change"]
        + totals["candidate_refresh"]
        + totals["cascade_refill"]
        + totals["candidate_rebound"]
        + totals["candidate_overdepth_regeneration"]
    )
    _require(logical == expected_logical, f"{closure_path} logical count does not close")
    _require(
        event_bearing == mutation_counts["has_event"]["due_opportunity_count"],
        f"{closure_path} event-bearing opportunity count differs",
    )
    _require(event_bearing <= logical, f"{closure_path} event opportunities exceed events")
    eventwise_path = f"{cadence_path}.eventwise_curation_closure"
    eventwise = _plain_dict(
        cadence["eventwise_curation_closure"],
        eventwise_path,
        (
            "all_eventwise_curation_semantics_match",
            "promotion_event_count",
            "ordinary_refresh_event_count",
            "event_bearing_opportunity_count",
        ),
    )
    _exact_bool(
        eventwise["all_eventwise_curation_semantics_match"],
        f"{eventwise_path}.all_eventwise_curation_semantics_match",
        True,
    )
    expected_eventwise = {
        "promotion_event_count": totals["promotion"],
        "ordinary_refresh_event_count": totals["ordinary_candidate_refresh"],
        "event_bearing_opportunity_count": event_bearing,
    }
    for field, expected in expected_eventwise.items():
        _require(
            _exact_int(eventwise[field], f"{eventwise_path}.{field}") == expected,
            f"{eventwise_path}.{field} differs",
        )
    totals["logical_event"] = logical
    totals["event_bearing_opportunity"] = event_bearing
    totals["selected_margin_due"] = selected_margin["due_opportunity_count"]
    totals["selected_margin_all"] = selected_margin["all_step_count"]
    totals["selected_margin_off"] = selected_margin["off_opportunity_count"]
    totals["candidate_margin_due"] = candidate_margin["due_opportunity_count"]
    totals["candidate_margin_all"] = candidate_margin["all_step_count"]
    totals["candidate_margin_off"] = candidate_margin["off_opportunity_count"]
    return totals


_PRIMARY_ENDPOINT_FIELDS: Final = (
    "endpoint_order",
    "margin_passes",
    "promotions",
    "cascade_refill_slot_count",
    "candidate_refreshes",
    "cascade_losses",
    "cascade_loss_definition",
    "target_admission_loss_end",
    "pre_recurrence_presence",
    "target_retention",
    "target_occupancy",
    "pre_recurrence_ranks",
    "cadence_integrity",
    "identity_reacquisition_claimed",
)


def _validate_primary_endpoints(value: object, *, arm_index: int) -> None:
    path = f"report.runs[{arm_index}].primary_endpoints"
    endpoints = _plain_dict(value, path, _PRIMARY_ENDPOINT_FIELDS)
    _exact_string_list(
        endpoints["endpoint_order"],
        f"{path}.endpoint_order",
        PRIMARY_ENDPOINT_ORDER,
    )
    cadence_totals = _validate_cadence_integrity(endpoints["cadence_integrity"], path=path)
    margin_path = f"{path}.margin_passes"
    margin = _plain_dict(
        endpoints["margin_passes"],
        margin_path,
        (
            "selected_strict_margin_pass_count",
            "selected_strict_margin_all_step_diagnostic_count",
            "selected_strict_margin_off_opportunity_diagnostic_count",
            "candidate_destination_strict_margin_pair_count",
            "candidate_destination_strict_margin_all_step_diagnostic_count",
            "candidate_destination_strict_margin_off_opportunity_diagnostic_count",
            "due_curation_event_count",
        ),
    )
    expected_margin = {
        "selected_strict_margin_pass_count": cadence_totals["selected_margin_due"],
        "selected_strict_margin_all_step_diagnostic_count": cadence_totals[
            "selected_margin_all"
        ],
        "selected_strict_margin_off_opportunity_diagnostic_count": cadence_totals[
            "selected_margin_off"
        ],
        "candidate_destination_strict_margin_pair_count": cadence_totals[
            "candidate_margin_due"
        ],
        "candidate_destination_strict_margin_all_step_diagnostic_count": cadence_totals[
            "candidate_margin_all"
        ],
        "candidate_destination_strict_margin_off_opportunity_diagnostic_count": (
            cadence_totals["candidate_margin_off"]
        ),
        "due_curation_event_count": TOTAL_CURATION_OPPORTUNITIES,
    }
    for field, expected in expected_margin.items():
        _require(
            _exact_int(margin[field], f"{margin_path}.{field}") == expected,
            f"{margin_path}.{field} differs from cadence",
        )
    promotions_path = f"{path}.promotions"
    promotions = _plain_dict(endpoints["promotions"], promotions_path, ("event_count",))
    _require(
        _exact_int(promotions["event_count"], f"{promotions_path}.event_count")
        == cadence_totals["promotion"],
        f"{promotions_path}.event_count differs from cadence",
    )
    _require(
        cadence_totals["promotion"]
        <= cadence_totals["selected_margin_due"]
        <= cadence_totals["candidate_margin_due"],
        f"{path} promotion/margin nesting does not close",
    )
    _require(
        _exact_int(
            endpoints["cascade_refill_slot_count"],
            f"{path}.cascade_refill_slot_count",
        )
        == cadence_totals["cascade_refill"],
        f"{path}.cascade_refill_slot_count differs from cadence",
    )
    refresh_path = f"{path}.candidate_refreshes"
    refreshes = _plain_dict(
        endpoints["candidate_refreshes"],
        refresh_path,
        (
            "decision_should_refresh_event_count",
            "ordinary_refreshed_slot_count",
            "post_promotion_refreshed_slot_count",
            "total_refreshed_slot_count",
        ),
    )
    expected_refreshes = {
        "decision_should_refresh_event_count": cadence_totals[
            "ordinary_candidate_refresh"
        ],
        "ordinary_refreshed_slot_count": cadence_totals["ordinary_candidate_refresh"],
        "post_promotion_refreshed_slot_count": cadence_totals[
            "post_promotion_candidate_refresh"
        ],
        "total_refreshed_slot_count": cadence_totals["candidate_refresh"],
    }
    for field, expected in expected_refreshes.items():
        _require(
            _exact_int(refreshes[field], f"{refresh_path}.{field}") == expected,
            f"{refresh_path}.{field} differs",
        )
    _exact_string(
        endpoints["cascade_loss_definition"],
        f"{path}.cascade_loss_definition",
        (
            "target-signature lost slots whose exact decision audit cause is "
            "cascade_dependency_refill"
        ),
    )
    present_at_end, lifecycle_totals = _validate_target_lifecycle(
        endpoints["target_admission_loss_end"],
        endpoints["cascade_losses"],
        path=path,
    )
    _require(
        lifecycle_totals["direct_admission"] <= cadence_totals["promotion"],
        f"{path} target admissions exceed promotions",
    )
    _require(
        lifecycle_totals["root_lost_slot"] <= cadence_totals["root_change"],
        f"{path} root-replacement losses exceed root changes",
    )
    _require(
        lifecycle_totals["cascade_lost_slot"] <= cadence_totals["cascade_refill"],
        f"{path} cascade losses exceed cascade refills",
    )
    _require(
        lifecycle_totals["loss_episode"]
        <= lifecycle_totals["root_lost_slot"] + lifecycle_totals["cascade_lost_slot"],
        f"{path} loss episodes exceed accounted lost slots",
    )
    recurrence_records = _validate_recurrence_records(
        endpoints["pre_recurrence_presence"],
        endpoints["pre_recurrence_ranks"],
        path=path,
    )
    _validate_target_retention(
        endpoints["target_retention"],
        recurrence_records=recurrence_records,
        present_at_end=present_at_end,
        path=path,
    )
    active_occupancy, candidate_occupancy = _validate_occupancy(
        endpoints["target_occupancy"],
        present_at_end=present_at_end,
        path=path,
    )
    for record in recurrence_records:
        target = cast(str, record["target"])
        if record["active_present"] is True:
            _require(
                active_occupancy[target] > 0,
                f"{path} pre-recurrence active presence lacks aggregate occupancy",
            )
        if record["candidate_present"] is True:
            _require(
                candidate_occupancy[target] > 0,
                f"{path} pre-recurrence candidate presence lacks aggregate occupancy",
            )
    _exact_bool(
        endpoints["identity_reacquisition_claimed"],
        f"{path}.identity_reacquisition_claimed",
        False,
    )


_EXECUTION_RECEIPT_FIELDS: Final = (
    "schema",
    "total_steps",
    "initial_state_sha256",
    "final_state_sha256",
    "trace_sha256",
    "expected_persistent_state_nbytes",
    "initial_persistent_state_nbytes",
    "final_persistent_state_nbytes",
    "final_step_count",
    "final_step_words_uint32",
    "final_replacement_phase",
    "initial_state_finite",
    "final_state_finite",
    "all_lifetime_counters_valid",
    "all_lifetime_capacity_available",
    "all_ranking_contracts_valid",
    "all_core_predictions_match_full_q",
    "initial_target_signature_counts_zero",
    "scientific_promotion_allowed",
    "evidence_authorized",
    "output_writes_allowed",
)


def _validate_execution_receipt(
    value: object,
    *,
    arm_index: int,
) -> tuple[str, str, str]:
    path = f"report.runs[{arm_index}].execution_receipt"
    receipt = _plain_dict(value, path, _EXECUTION_RECEIPT_FIELDS)
    _exact_string(
        receipt["schema"],
        f"{path}.schema",
        ARM_EXECUTION_RECEIPT_SCHEMA,
    )
    _require(
        _exact_int(receipt["total_steps"], f"{path}.total_steps") == TOTAL_STEPS,
        f"{path}.total_steps differs",
    )
    initial_sha256 = _sha256(receipt["initial_state_sha256"], f"{path}.initial_state_sha256")
    final_sha256 = _sha256(receipt["final_state_sha256"], f"{path}.final_state_sha256")
    trace_sha256 = _sha256(receipt["trace_sha256"], f"{path}.trace_sha256")
    _require(initial_sha256 != final_sha256, f"{path} state did not advance")
    for field in (
        "expected_persistent_state_nbytes",
        "initial_persistent_state_nbytes",
        "final_persistent_state_nbytes",
    ):
        _require(
            _exact_int(receipt[field], f"{path}.{field}") == PERSISTENT_STATE_NBYTES,
            f"{path}.{field} differs from the exact state formula",
        )
    _require(
        _exact_int(receipt["final_step_count"], f"{path}.final_step_count")
        == TOTAL_STEPS,
        f"{path}.final_step_count differs",
    )
    words = _plain_list(
        receipt["final_step_words_uint32"],
        f"{path}.final_step_words_uint32",
        length=2,
    )
    word_values = [
        _exact_int(word, f"{path}.final_step_words_uint32[{index}]")
        for index, word in enumerate(words)
    ]
    _require(word_values == [0, TOTAL_STEPS], f"{path}.final_step_words_uint32 differs")
    _require(
        _exact_int(
            receipt["final_replacement_phase"],
            f"{path}.final_replacement_phase",
        )
        == FINAL_REPLACEMENT_PHASE,
        f"{path}.final_replacement_phase differs",
    )
    for field in (
        "initial_state_finite",
        "final_state_finite",
        "all_lifetime_counters_valid",
        "all_lifetime_capacity_available",
        "all_ranking_contracts_valid",
        "all_core_predictions_match_full_q",
        "initial_target_signature_counts_zero",
    ):
        _exact_bool(receipt[field], f"{path}.{field}", True)
    for field in (
        "scientific_promotion_allowed",
        "evidence_authorized",
        "output_writes_allowed",
    ):
        _exact_bool(receipt[field], f"{path}.{field}", False)
    return initial_sha256, final_sha256, trace_sha256


_RUN_FIELDS: Final = (
    "arm",
    "source_arm_name",
    "learner_config_sha256",
    "execution_receipt",
    "state_gate_receipt",
    "primary_endpoints",
    "reward_counts",
    "arm_record_sha256",
)


def _validate_run(
    value: object,
    *,
    arm_index: int,
) -> tuple[str, str]:
    path = f"report.runs[{arm_index}]"
    run = _plain_dict(value, path, _RUN_FIELDS)
    _exact_string(run["arm"], f"{path}.arm", ARM_ORDER[arm_index])
    _exact_string(run["source_arm_name"], f"{path}.source_arm_name", SOURCE_ARM_NAME)
    _exact_string(
        run["learner_config_sha256"],
        f"{path}.learner_config_sha256",
        _LEARNER_CONFIG_SHA256[arm_index],
    )
    initial_state_sha256, _final_state_sha256, _trace_sha256 = (
        _validate_execution_receipt(run["execution_receipt"], arm_index=arm_index)
    )
    initial_subset_sha256, _final_subset_sha256 = _validate_state_receipt(
        run["state_gate_receipt"],
        arm_index=arm_index,
    )
    _validate_primary_endpoints(run["primary_endpoints"], arm_index=arm_index)
    _validate_reward_projection(run["reward_counts"], arm_index=arm_index)
    arm_sha256 = _sha256(run["arm_record_sha256"], f"{path}.arm_record_sha256")
    body = {field: run[field] for field in _RUN_FIELDS if field != "arm_record_sha256"}
    _require(
        arm_sha256 == canonical_json_sha256(body),
        f"{path}.arm_record_sha256 does not reconstruct",
    )
    return initial_state_sha256, initial_subset_sha256


_EXECUTION_FIELDS: Final = (
    "attempt_index",
    "attempts_authorized",
    "attempts_consumed",
    "root_consumed",
    "attempt_consumed_before_evaluator_import",
    "retry_or_recovery_authorized",
    "panel_completed",
    "arm_count",
)
_AUTHORITY_FIELDS: Final = (
    "development_only",
    "descriptive_result_available",
    "scientific_promotion_allowed",
    "evidence_authorized",
    "experiment_output_writes_allowed",
    "artifact_authorized",
    "threshold_defined_or_applied",
    "winner_or_default_selected",
    "search_or_tuning_performed",
    "retry_or_recovery_authorized",
)
_CROSS_ARM_FIELDS: Final = (
    "shared_initial_state_sha256",
    "shared_initial_subset_sha256",
    "shared_protocol_source_and_genesis",
    "shared_base_logical_work_matched",
    "stream_shapes_and_update_opportunities_matched",
    "persistent_shapes_and_bytes_matched",
    "intervention_specific_logical_work_matched",
    "total_named_logical_work_equivalence_claimed",
    "behavior_dependent_branch_work_equivalence_claimed",
    "behavioral_experience_matching_claimed",
    "compiled_flop_equivalence_claimed",
    "work_resource_contract_embedded",
    "work_resource_contract_sha256_bound",
)
_TOP_LEVEL_FIELDS: Final = (
    "schema",
    "status",
    "bindings",
    "execution",
    "authority",
    "arm_order",
    "primary_endpoint_order",
    "reward_metric_order",
    "runs",
    "cross_arm_contract",
    "work_resource_contract",
    "work_resource_contract_sha256",
    "report_sha256",
)


def _validate_bindings(value: object, expected: ExpectedExecutionBindings) -> None:
    expected_record = {**dict(_STABLE_BINDING_ITEMS), **expected.to_config()}
    fields = tuple(expected_record)
    bindings = _plain_dict(value, "report.bindings", fields)
    _require_exact_json(
        bindings,
        expected_record,
        "report.bindings differ from the exact source closure",
    )


def _validate_execution(value: object) -> None:
    execution = _plain_dict(value, "report.execution", _EXECUTION_FIELDS)
    expected: dict[str, object] = {
        "attempt_index": 1,
        "attempts_authorized": 1,
        "attempts_consumed": 1,
        "root_consumed": True,
        "attempt_consumed_before_evaluator_import": True,
        "retry_or_recovery_authorized": False,
        "panel_completed": True,
        "arm_count": len(ARM_ORDER),
    }
    _require_exact_json(
        execution,
        expected,
        "report.execution differs from the one-shot completion",
    )


def _validate_authority(value: object) -> None:
    authority = _plain_dict(value, "report.authority", _AUTHORITY_FIELDS)
    expected = {
        "development_only": True,
        "descriptive_result_available": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "experiment_output_writes_allowed": False,
        "artifact_authorized": False,
        "threshold_defined_or_applied": False,
        "winner_or_default_selected": False,
        "search_or_tuning_performed": False,
        "retry_or_recovery_authorized": False,
    }
    _require_exact_json(
        authority,
        expected,
        "report.authority exceeds descriptive authority",
    )


def _validate_cross_arm_contract(
    value: object,
    *,
    initial_state_sha256: str,
    initial_subset_sha256: str,
) -> None:
    contract = _plain_dict(value, "report.cross_arm_contract", _CROSS_ARM_FIELDS)
    expected: dict[str, object] = {
        "shared_initial_state_sha256": initial_state_sha256,
        "shared_initial_subset_sha256": initial_subset_sha256,
        "shared_protocol_source_and_genesis": True,
        "shared_base_logical_work_matched": True,
        "stream_shapes_and_update_opportunities_matched": True,
        "persistent_shapes_and_bytes_matched": True,
        "intervention_specific_logical_work_matched": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "behavioral_experience_matching_claimed": False,
        "compiled_flop_equivalence_claimed": False,
        "work_resource_contract_embedded": True,
        "work_resource_contract_sha256_bound": True,
    }
    _require_exact_json(contract, expected, "report.cross_arm_contract is not exact")


def validate_v3_descriptive_report(
    report: object,
    expected_bindings: ExpectedExecutionBindings,
) -> None:
    """Raise unless ``report`` is the exact bounded one-shot descriptive payload."""

    if type(expected_bindings) is not ExpectedExecutionBindings:
        raise TypeError("expected_bindings must be an exact ExpectedExecutionBindings")
    _validate_json_tree(report)
    root = _plain_dict(report, "report", _TOP_LEVEL_FIELDS)
    _exact_string(root["schema"], "report.schema", REPORT_SCHEMA)
    _exact_string(root["status"], "report.status", REPORT_STATUS)
    _validate_bindings(root["bindings"], expected_bindings)
    _validate_execution(root["execution"])
    _validate_authority(root["authority"])
    _exact_string_list(root["arm_order"], "report.arm_order", ARM_ORDER)
    _exact_string_list(
        root["primary_endpoint_order"],
        "report.primary_endpoint_order",
        PRIMARY_ENDPOINT_ORDER,
    )
    _exact_string_list(
        root["reward_metric_order"],
        "report.reward_metric_order",
        REWARD_RECORD_FIELDS,
    )
    runs = _plain_list(root["runs"], "report.runs", length=len(ARM_ORDER))
    initial_states: list[str] = []
    initial_subsets: list[str] = []
    for arm_index, run in enumerate(runs):
        initial_state, initial_subset = _validate_run(run, arm_index=arm_index)
        initial_states.append(initial_state)
        initial_subsets.append(initial_subset)
    _require(len(set(initial_states)) == 1, "report.runs do not share one initial state")
    _require(len(set(initial_subsets)) == 1, "report.runs do not share one initial subset")
    _validate_cross_arm_contract(
        root["cross_arm_contract"],
        initial_state_sha256=initial_states[0],
        initial_subset_sha256=initial_subsets[0],
    )
    expected_work = work_resource_contract_config()
    _require(
        canonical_json_sha256(expected_work) == WORK_RESOURCE_CONTRACT_SHA256,
        "frozen work-resource contract digest does not reconstruct",
    )
    _require_exact_json(
        root["work_resource_contract"],
        expected_work,
        "report.work_resource_contract differs from the exact engine contract",
    )
    _exact_string(
        root["work_resource_contract_sha256"],
        "report.work_resource_contract_sha256",
        WORK_RESOURCE_CONTRACT_SHA256,
    )
    report_sha256 = _sha256(root["report_sha256"], "report.report_sha256")
    body = {
        field: root[field]
        for field in _TOP_LEVEL_FIELDS
        if field != "report_sha256"
    }
    _require(
        report_sha256 == canonical_json_sha256(body),
        "report.report_sha256 does not reconstruct",
    )


def serialize_v3_descriptive_report(
    report: object,
    expected_bindings: ExpectedExecutionBindings,
) -> str:
    """Validate and return canonical JSON; this function performs no writes."""

    validate_v3_descriptive_report(report, expected_bindings)
    return canonical_json(report)


__all__ = [
    "ACTION_HEADS",
    "ACTIVE_SLOTS",
    "ARM_EXECUTION_RECEIPT_SCHEMA",
    "ARM_ORDER",
    "CANDIDATE_SLOTS",
    "CURATION_INTERVAL",
    "ENTRY_WINDOW",
    "ExpectedExecutionBindings",
    "FINAL_REPLACEMENT_PHASE",
    "PERSISTENT_STATE_NBYTES",
    "PHASE_BOUNDARIES",
    "PHASE_LENGTHS",
    "PHASE_ORDER",
    "PRIMARY_ENDPOINT_ORDER",
    "RAW_DIM",
    "REPORT_SCHEMA",
    "REPORT_STATUS",
    "REWARD_COUNT_SCHEMA",
    "REWARD_RECORD_FIELDS",
    "SOURCE_ARM_NAME",
    "STATE_GATE_SCHEMA",
    "TAIL_WINDOW",
    "TARGET_NAMES",
    "TOTAL_CURATION_OPPORTUNITIES",
    "TOTAL_STEPS",
    "WORK_RESOURCE_CONTRACT_SHA256",
    "canonical_json",
    "canonical_json_sha256",
    "serialize_v3_descriptive_report",
    "validate_v3_descriptive_report",
    "work_resource_contract_config",
]
