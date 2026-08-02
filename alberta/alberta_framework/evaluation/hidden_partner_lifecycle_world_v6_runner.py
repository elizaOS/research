"""Fixed-shape DEVELOPMENT runner for the noisy-world v6 mechanism study.

This module is deliberately not an evidence protocol.  It derives no seeds,
owns no seed namespace, contains no thresholds, writes no artifact, and makes
no promotion decision.  A caller must explicitly supply both typed PRNG keys.
The production scan always has 30,318 iterations, but executes the bridge only
on the scan plan's active prefix.  Padding preserves the complete runtime carry
bit-exact and emits one canonical zero byte.

The scan consumes the bridge's compact transient trace into fixed-size sums and
a 473-entry curation ledger.  It never stacks the full bridge or mechanism
trace.  Lifecycle success fields are observed outcomes, not structural-validity
gates: a causally intact run may validly fail to acquire, retain, retire, or
reacquire a critical descriptor.
"""

# mypy: disable-error-code="attr-defined,call-arg,no-any-return,no-untyped-call,operator"

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    BASE_FEATURE_DIM,
    CANDIDATE_PAIR_SLOTS,
    DEPLOYED_FEATURE_DIM,
    IntegratedHiddenPartnerAgent,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    FORBIDDEN_SEED_NAMESPACES,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
    V6_REPRESENTATION_LOSS_WEIGHTS,
    V6_TARGET_HEAD_ORDER,
    HiddenPartnerLifecycleWorldV6Control,
    validate_v6_control,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_intervention_audit import (
    V6_CONTROL_REQUIRED_WITNESSES,
    V6_FLOAT32_REPLAY_ATOL,
    V6_FLOAT32_REPLAY_RTOL,
    V6_INTERVENTION_AUDIT_ORDER,
    V6_INTERVENTION_WITNESS_ORDER,
    V6InterventionStepAudit,
    audit_v6_intervention_step,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import (
    V6RuntimeRecord,
    capture_v6_runtime_record,
    validate_v6_runtime_record,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    ENTRY_WINDOW_STEPS,
    FINAL_WINDOW_STEPS,
    N_SEGMENTS,
    TAIL_WINDOW_STEPS,
    HiddenPartnerLifecycleWorldV6ScanPlan,
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    require_v6_control_suite_ready,
    validate_hidden_partner_lifecycle_world_v6_scan_plan,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    MAX_SCAN_STEPS as _MAX_SCAN_STEPS,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldOnlineBridge,
    HiddenPartnerWorldOnlineResourceBudget,
    HiddenPartnerWorldOnlineState,
    HiddenPartnerWorldOnlineTrace,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackState,
    HiddenPartnerWorldFeedbackWorld,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNNER_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.runner-development.v1"
)
RUNNER_STATUS = "DEVELOPMENT_RUNNER_NO_EVIDENCE_AUTHORITY"
DEVELOPMENT_ONLY = True
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

MAX_SCAN_STEPS = _MAX_SCAN_STEPS
WINDOW_BIN_COUNT = 37
JOINT_ACTION_ROWS = 4
TARGET_HEADS = 10
CRITICAL_PAIR_COUNT = 2
CURATION_INTERVAL = 64
MAX_CADENCE_LEDGER_ENTRIES = MAX_SCAN_STEPS // CURATION_INTERVAL
_INT32_MAX = 2**31 - 1

C_PAIR = (0, 2)
D_PAIR = (4, 5)
CRITICAL_PAIRS = (C_PAIR, D_PAIR)
CRITICAL_CANDIDATE_INDICES = (1, 38)


def require_v6_development_seed_namespace(seed_namespace: object) -> None:
    """Refuse execution under any seed namespace, reserved or otherwise.

    The development runner owns no seed namespace: both PRNG keys are caller
    supplied and its bound config pins ``seed_namespace`` to ``None``.  A
    certification process that later binds this runner must present its
    namespace here first.  Every reserved v6 evidence namespace is refused by
    name, and any other namespace is refused because development machinery
    has no namespace authority at all.
    """

    if seed_namespace is None:
        return
    if not isinstance(seed_namespace, str):
        raise TypeError("v6 seed namespace must be None or an exact string")
    if seed_namespace in FORBIDDEN_SEED_NAMESPACES:
        raise PermissionError(
            "v6 development runner refuses the reserved evidence namespace "
            f"{seed_namespace!r}"
        )
    raise PermissionError(
        "v6 development runner has no seed-namespace authority; "
        f"refusing execution under {seed_namespace!r}"
    )


def _discover_v6_source_closure_paths() -> tuple[str, ...]:
    """Inventory every package source plus the two dependency manifests."""

    repository_root = Path(__file__).resolve().parents[2]
    framework_root = repository_root / "alberta_framework"
    if not framework_root.is_dir():
        raise FileNotFoundError("v6 source-closure package root is missing")
    python_paths = tuple(
        sorted(
            path.relative_to(repository_root).as_posix()
            for path in framework_root.rglob("*.py")
            if path.is_file()
        )
    )
    if not python_paths or len(python_paths) != len(set(python_paths)):
        raise RuntimeError("v6 package source inventory is empty or duplicated")
    return ("pyproject.toml", "uv.lock", *python_paths)


# Import-time inventory, not a hand-maintained approximation of eager imports.
# Added and deleted Python sources are independently detected by the fresh
# inventory in ``compute_v6_source_closure_hashes``.
V6_SOURCE_CLOSURE_PATHS = _discover_v6_source_closure_paths()

V6_PROPOSAL_EVENT_ORDER: tuple[str, ...] = (
    "replaced_slot",
    "promoted_candidate",
    "refreshed_candidate",
    "retired_slot",
    "retired_left",
    "retired_right",
)
V6_APPLIED_EVENT_ORDER: tuple[str, ...] = V6_PROPOSAL_EVENT_ORDER
V6_CRITICAL_STAGE_ORDER: tuple[str, ...] = ("pre", "proposal", "applied")
V6_CANDIDATE_STREAK_ENDPOINT_ORDER: tuple[str, ...] = ("pre", "post")
V6_CANDIDATE_FLAG_ORDER: tuple[str, ...] = (
    "promotion_raw_evidence",
    "promotion_confirmed",
    "reacquisition_required_pre",
    "reacquisition_required_proposal_post",
    "reacquisition_required_post",
    "reacquisition_confirmed",
)
V6_RANDOM_CURATION_FLAG_ORDER: tuple[str, ...] = (
    "enabled",
    "attempted",
    "applied",
)
V6_RANDOM_CURATION_SELECTED_ORDER: tuple[str, ...] = (
    "active_worst_slot",
    "promotion_candidate",
    "refresh_candidate",
)
V6_CONSUMER_MASK_ORDER: tuple[str, ...] = (
    "durable_read",
    "read_acquire_pre",
    "read_acquire_post",
    "confirmed_write_pre",
    "confirmed_write_post",
    "read_pre",
    "read_post",
    "active_pre",
    "active_post",
)
V6_ROUTER_MASK_ORDER: tuple[str, ...] = ("survivor", "new", "evicted")
V6_ROUTER_FLAG_ORDER: tuple[str, ...] = (
    "valid",
    "applied",
    "carry_survivors",
    "descriptors_changed",
)
V6_ROUTER_COUNT_ORDER: tuple[str, ...] = (
    "route_before",
    "route_after",
    "generation_before",
    "generation_after",
)
V6_WORLD_RNG_KEY_ORDER: tuple[str, ...] = (
    "signal",
    "partner",
    "world",
    "cue",
    "outcome",
)
V6_POLICY_KEY_ENDPOINT_ORDER: tuple[str, ...] = ("before", "after")

V6_TRANSITION_STREAM_BIT_ORDER: tuple[str, ...] = (
    "signal_0_positive",
    "signal_1_positive",
    "signal_2_positive",
    "partner_flipped",
    "world_flipped",
    "cue_0_flipped",
    "cue_1_flipped",
    "outcome_flipped",
)
V6_INITIAL_STREAM_BIT_ORDER: tuple[str, ...] = (
    "signal_0_positive",
    "signal_1_positive",
    "signal_2_positive",
    "world_sign_positive",
    "cue_0_positive",
    "cue_1_positive",
    "previous_outcome_positive",
    "has_partner_history",
)

V6_COMPONENT_DELTA_ORDER: tuple[str, ...] = (
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

V6_CONTRACT_AUDIT_ORDER: tuple[str, ...] = (
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

if len(V6_CONTRACT_AUDIT_ORDER) != 27:
    raise RuntimeError("v6 active-transition contract audit width must remain exactly 27")
if len(V6_COMPONENT_DELTA_ORDER) != 10:
    raise RuntimeError("v6 component-delta width must remain exactly 10")
if len(V6_INTERVENTION_AUDIT_ORDER) != 18:
    raise RuntimeError("v6 intervention audit width must remain exactly 18")
if len(V6_INTERVENTION_WITNESS_ORDER) != 16:
    raise RuntimeError("v6 intervention witness width must remain exactly 16")
if MAX_CADENCE_LEDGER_ENTRIES != 473:
    raise RuntimeError("v6 cadence ledger capacity has drifted from 473")
if tuple(map(len, (V6_PROPOSAL_EVENT_ORDER, V6_CRITICAL_STAGE_ORDER))) != (6, 3):
    raise RuntimeError("v6 event or critical-stage order has drifted")
if tuple(
    map(
        len,
        (
            V6_CANDIDATE_STREAK_ENDPOINT_ORDER,
            V6_CANDIDATE_FLAG_ORDER,
            V6_RANDOM_CURATION_FLAG_ORDER,
            V6_RANDOM_CURATION_SELECTED_ORDER,
            V6_CONSUMER_MASK_ORDER,
            V6_ROUTER_MASK_ORDER,
            V6_ROUTER_FLAG_ORDER,
            V6_ROUTER_COUNT_ORDER,
            V6_WORLD_RNG_KEY_ORDER,
            V6_POLICY_KEY_ENDPOINT_ORDER,
            V6_TARGET_HEAD_ORDER,
        ),
    )
) != (2, 6, 3, 3, 9, 3, 4, 4, 5, 2, 10):
    raise RuntimeError("v6 reviewed column order has drifted")
if (
    len(V6_TRANSITION_STREAM_BIT_ORDER) != 8
    or len(set(V6_TRANSITION_STREAM_BIT_ORDER)) != 8
    or len(V6_INITIAL_STREAM_BIT_ORDER) != 8
    or len(set(V6_INITIAL_STREAM_BIT_ORDER)) != 8
):
    raise RuntimeError("v6 reviewed stream-bit order must contain eight unique names")


@chex.dataclass(frozen=True)
class V6WindowTotals:
    """Thirty-six occurrence windows plus the overlapping terminal window."""

    scheduled_support: Int[Array, " 37"]
    accepted_support: Int[Array, " 37"]
    reward_sum: Float[Array, " 37"]
    behavior_nll_sum: Float[Array, " 37"]
    behavior_brier_sum: Float[Array, " 37"]
    behavior_correct_count: Int[Array, " 37"]
    filter_selected_regret_sum: Float[Array, " 37"]
    filter_planner_regret_sum: Float[Array, " 37"]
    full_information_selected_regret_sum: Float[Array, " 37"]
    full_information_planner_regret_sum: Float[Array, " 37"]
    world_posterior_nll_sum: Float[Array, " 37"]
    world_posterior_brier_sum: Float[Array, " 37"]
    grounded_support: Int[Array, " 37"]
    grounded_fit_loss_by_head_sum: Float[Array, "37 10"]
    grounded_representation_loss_by_head_sum: Float[Array, "37 10"]
    grounded_representation_gradient_norm_by_head_sum: Float[Array, "37 10"]
    critical_present_count: Int[Array, "37 2"]
    critical_durable_read_count: Int[Array, "37 2"]
    critical_evidence_refresh_count: Int[Array, "37 2"]
    critical_relevance_score_sum: Float[Array, "37 2"]
    critical_relevance_error_abs_sum: Float[Array, "37 2"]


@chex.dataclass(frozen=True)
class V6RowHeadTotals:
    """Executed-row prequential support and loss for all forty row/head cells."""

    support: Int[Array, "4 10"]
    absolute_error_sum: Float[Array, "4 10"]
    fit_loss_sum: Float[Array, "4 10"]
    representation_loss_sum: Float[Array, "4 10"]
    representation_gradient_norm_sum: Float[Array, "4 10"]
    feature_contribution_abs_sum: Float[Array, "4 10"]
    row_bias_abs_sum: Float[Array, "4 10"]
    executed_weight_delta_norm_sum: Float[Array, "4 10"]
    executed_bias_delta_abs_sum: Float[Array, "4 10"]
    proposed_weight_change_count: Int[Array, " 4"]
    proposed_bias_change_count: Int[Array, " 4"]
    row_isolation_failure_count: Int[Array, ""]
    row_head_algebra_failure_count: Int[Array, ""]
    nonfinite_row_head_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class V6FilterTotals:
    """Tie-aware evaluator-filter accounting."""

    support: Int[Array, ""]
    optimal_value_sum: Float[Array, ""]
    selected_value_sum: Float[Array, ""]
    selected_regret_sum: Float[Array, ""]
    margin_sum: Float[Array, ""]
    tied_support: Int[Array, ""]
    nontied_support: Int[Array, ""]
    tied_selected_regret_sum: Float[Array, ""]
    tied_focal_action_support: Int[Array, " 2"]
    cue_pattern_support: Int[Array, " 4"]
    cue_flip_support: Int[Array, " 2"]
    cue_flip_count: Int[Array, " 2"]
    filter_recurrence_failure_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class V6ActionTotals:
    """Applied and ordinary-policy action support with RNG accounting."""

    focal_action_support: Int[Array, " 2"]
    partner_action_support: Int[Array, " 2"]
    joint_row_support: Int[Array, " 4"]
    ordinary_policy_action_support: Int[Array, " 2"]
    explored_count: Int[Array, ""]
    externally_forced_count: Int[Array, ""]
    policy_schedule_failure_count: Int[Array, ""]
    policy_replay_failure_count: Int[Array, ""]
    rng_chain_failure_count: Int[Array, ""]
    decision_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class V6AuditTotals:
    """Structural failure counts; lifecycle quality is intentionally separate."""

    contract_failure_counts: Int[Array, " 27"]
    intervention_failure_counts: Int[Array, " 18"]
    intervention_witness_counts: Int[Array, " 16"]
    component_delta_sums: Int[Array, " 10"]
    component_delta_failure_counts: Int[Array, " 10"]
    active_steps: Int[Array, ""]
    accepted_steps: Int[Array, ""]
    learner_valid_steps: Int[Array, ""]
    filter_valid_steps: Int[Array, ""]
    oracle_valid_steps: Int[Array, ""]
    mechanism_valid_steps: Int[Array, ""]
    all_finite_steps: Int[Array, ""]
    curation_attempt_count: Int[Array, ""]
    ledger_count: Int[Array, ""]
    ledger_overflow: Bool[Array, ""]


@chex.dataclass(frozen=True)
class V6CadenceObservation:
    """One fixed-width curation-boundary observation before ledger insertion."""

    transition_step: Int[Array, ""]
    regime_id: Int[Array, ""]
    pre_descriptors: Int[Array, "12 2"]
    proposal_descriptors: Int[Array, "12 2"]
    applied_descriptors: Int[Array, "12 2"]
    proposal_event: Int[Array, " 6"]
    applied_event: Int[Array, " 6"]
    critical_slot: Int[Array, "3 2"]
    critical_candidate_streak: Int[Array, "2 2"]
    critical_candidate_flags: Bool[Array, "2 6"]
    candidate_reset_mask: Bool[Array, "2 66"]
    random_curation_flags: Bool[Array, " 3"]
    random_curation_selected: Int[Array, " 3"]
    random_active_priorities: Float[Array, " 12"]
    random_candidate_priorities: Float[Array, " 66"]
    consumer_masks: Bool[Array, "9 12"]
    router_source_slots: Int[Array, " 12"]
    router_masks: Bool[Array, "3 12"]
    router_flags: Bool[Array, " 4"]
    router_counts: Int[Array, " 4"]
    transaction_exact: Bool[Array, ""]
    identity_carry_exact: Bool[Array, ""]
    retired_identity_reset_exact: Bool[Array, ""]


@chex.dataclass(frozen=True)
class V6CadenceLedger:
    """Every possible 64-step curation boundary in the maximum scan."""

    occupied: Bool[Array, " 473"]
    transition_step: Int[Array, " 473"]
    regime_id: Int[Array, " 473"]
    pre_descriptors: Int[Array, "473 12 2"]
    proposal_descriptors: Int[Array, "473 12 2"]
    applied_descriptors: Int[Array, "473 12 2"]
    proposal_event: Int[Array, "473 6"]
    applied_event: Int[Array, "473 6"]
    critical_slot: Int[Array, "473 3 2"]
    critical_candidate_streak: Int[Array, "473 2 2"]
    critical_candidate_flags: Bool[Array, "473 2 6"]
    candidate_reset_mask: Bool[Array, "473 2 66"]
    random_curation_flags: Bool[Array, "473 3"]
    random_curation_selected: Int[Array, "473 3"]
    random_active_priorities: Float[Array, "473 12"]
    random_candidate_priorities: Float[Array, "473 66"]
    consumer_masks: Bool[Array, "473 9 12"]
    router_source_slots: Int[Array, "473 12"]
    router_masks: Bool[Array, "473 3 12"]
    router_flags: Bool[Array, "473 4"]
    router_counts: Int[Array, "473 4"]
    transaction_exact: Bool[Array, " 473"]
    identity_carry_exact: Bool[Array, " 473"]
    retired_identity_reset_exact: Bool[Array, " 473"]


@chex.dataclass(frozen=True)
class V6LifecycleObservation:
    """Minimal identity observation for the online C/D outcome state machine."""

    transition_step: Int[Array, ""]
    occurrence_index: Int[Array, ""]
    regime_id: Int[Array, ""]
    pre_descriptors: Int[Array, "12 2"]
    applied_descriptors: Int[Array, "12 2"]
    promoted_candidate: Int[Array, ""]
    retired_pair: Int[Array, " 2"]
    d_reacquisition_required_pre: Bool[Array, ""]
    d_reacquisition_confirmed: Bool[Array, ""]
    d_reset_exact: Bool[Array, ""]
    structural_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class V6LifecycleChainState:
    """Observed lifecycle ordering; negative values remain valid outcomes."""

    structural_valid: Bool[Array, ""]
    c_first_acquisition_step: Int[Array, ""]
    c_ever_acquired: Bool[Array, ""]
    c_retained_after_acquisition: Bool[Array, ""]
    c_acquired_in_first_c: Bool[Array, ""]
    d_phase: Int[Array, ""]
    d_first_acquisition_step: Int[Array, ""]
    d_retirement_step: Int[Array, ""]
    d_reacquisition_step: Int[Array, ""]
    d_first_acquisition_in_first_d: Bool[Array, ""]
    d_retirement_while_absent: Bool[Array, ""]
    d_retirement_reset_exact: Bool[Array, ""]
    d_reacquisition_in_second_d: Bool[Array, ""]
    d_reacquisition_gate_observed: Bool[Array, ""]
    d_ordered_outcome: Bool[Array, ""]
    out_of_order_event_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class V6RngRecord:
    """Exact key-data endpoints and reviewed draw counts."""

    supplied_key_data: UInt[Array, "2 2"]
    initial_world_key_data: UInt[Array, "5 2"]
    final_world_key_data: UInt[Array, "5 2"]
    initial_policy_key_data: UInt[Array, "2 2"]
    final_policy_key_data: UInt[Array, "2 2"]
    initial_interaction_key_data: UInt[Array, " 2"]
    final_interaction_key_data: UInt[Array, " 2"]
    initial_stream_bits: UInt[Array, ""]
    world_draw_counts: Int[Array, " 5"]
    interaction_key_advance_count: Int[Array, ""]
    policy_decision_count: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class V6SourceClosureHash:
    """One dynamic lowercase SHA-256 binding for a reviewed local source."""

    relative_path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class V6ResourceRecord:
    """Host-side exact persistent resource endpoints."""

    initial: HiddenPartnerWorldOnlineResourceBudget
    final: HiddenPartnerWorldOnlineResourceBudget
    peak_total_state_nbytes: int
    static_total_state_nbytes: bool
    zero_replay: bool
    initial_tree_signature: tuple[tuple[tuple[int, ...], str], ...]
    final_tree_signature: tuple[tuple[tuple[int, ...], str], ...]
    tree_structure_equal: bool
    tree_signature_equal: bool


@chex.dataclass(frozen=True)
class V6ScanCarry:
    """Bridge state plus compact fixed-shape evaluator accumulation."""

    bridge_state: HiddenPartnerWorldOnlineState
    windows: V6WindowTotals
    row_heads: V6RowHeadTotals
    filter_totals: V6FilterTotals
    action_totals: V6ActionTotals
    audits: V6AuditTotals
    ledger: V6CadenceLedger
    lifecycle: V6LifecycleChainState


@dataclasses.dataclass(frozen=True)
class V6DevelopmentRun:
    """In-memory development output; intentionally has no artifact serializer."""

    control_name: str
    primary: bool
    plan: HiddenPartnerLifecycleWorldV6ScanPlan
    control_config_sha256: str
    control_matrix_sha256: str
    bridge_config_sha256: str
    runner_config_sha256: str
    source_closure_hashes: tuple[V6SourceClosureHash, ...]
    runtime: V6RuntimeRecord
    initial_state: HiddenPartnerWorldOnlineState
    final_state: HiddenPartnerWorldOnlineState
    windows: V6WindowTotals
    row_heads: V6RowHeadTotals
    filter_totals: V6FilterTotals
    action_totals: V6ActionTotals
    audits: V6AuditTotals
    ledger: V6CadenceLedger
    lifecycle: V6LifecycleChainState
    rng: V6RngRecord
    resources: V6ResourceRecord
    stream_code: UInt[Array, " 30318"]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("v6 runner config must contain finite canonical JSON data") from exc


def _read_v6_source_closure_hashes(
    paths: tuple[str, ...],
) -> tuple[V6SourceClosureHash, ...]:
    """Read one ordered on-disk snapshot of the reviewed closure."""

    repository_root = Path(__file__).resolve().parents[2]
    records: list[V6SourceClosureHash] = []
    for relative_path in paths:
        source = repository_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"v6 source-closure path is missing: {relative_path}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        records.append(V6SourceClosureHash(relative_path=relative_path, sha256=digest))
    return tuple(records)


# Freeze the disk identities when this runner module is imported. A run from a
# long-lived interpreter must not execute already-imported code while claiming
# hashes read from files that changed later in the process lifetime.
_V6_IMPORT_SOURCE_CLOSURE_HASHES = _read_v6_source_closure_hashes(V6_SOURCE_CLOSURE_PATHS)


def compute_v6_source_closure_hashes() -> tuple[V6SourceClosureHash, ...]:
    """Return the import-time closure only if the live files remain identical."""

    live_paths = _discover_v6_source_closure_paths()
    if live_paths != V6_SOURCE_CLOSURE_PATHS:
        raise RuntimeError(
            "v6 package source inventory changed after runner import; start a fresh Python process"
        )
    live = _read_v6_source_closure_hashes(live_paths)
    if live != _V6_IMPORT_SOURCE_CLOSURE_HASHES:
        raise RuntimeError(
            "v6 reviewed sources changed after runner import; start a fresh Python process"
        )
    return _V6_IMPORT_SOURCE_CLOSURE_HASHES


def _tree_shape_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    records: list[tuple[tuple[int, ...], str]] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError("v6 persistent state tree must contain only array leaves")
        records.append((tuple(shape), str(dtype)))
    return tuple(records)


def _zero_i32(shape: tuple[int, ...] = ()) -> Array:
    return jnp.zeros(shape, dtype=jnp.int32)


def _zero_f32(shape: tuple[int, ...] = ()) -> Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def empty_v6_window_totals() -> V6WindowTotals:
    """Return exact positive-zero window accumulators."""

    i37 = _zero_i32((WINDOW_BIN_COUNT,))
    f37 = _zero_f32((WINDOW_BIN_COUNT,))
    f3710 = _zero_f32((WINDOW_BIN_COUNT, TARGET_HEADS))
    i372 = _zero_i32((WINDOW_BIN_COUNT, CRITICAL_PAIR_COUNT))
    f372 = _zero_f32((WINDOW_BIN_COUNT, CRITICAL_PAIR_COUNT))
    return V6WindowTotals(
        scheduled_support=i37,
        accepted_support=i37,
        reward_sum=f37,
        behavior_nll_sum=f37,
        behavior_brier_sum=f37,
        behavior_correct_count=i37,
        filter_selected_regret_sum=f37,
        filter_planner_regret_sum=f37,
        full_information_selected_regret_sum=f37,
        full_information_planner_regret_sum=f37,
        world_posterior_nll_sum=f37,
        world_posterior_brier_sum=f37,
        grounded_support=i37,
        grounded_fit_loss_by_head_sum=f3710,
        grounded_representation_loss_by_head_sum=f3710,
        grounded_representation_gradient_norm_by_head_sum=f3710,
        critical_present_count=i372,
        critical_durable_read_count=i372,
        critical_evidence_refresh_count=i372,
        critical_relevance_score_sum=f372,
        critical_relevance_error_abs_sum=f372,
    )


def empty_v6_row_head_totals() -> V6RowHeadTotals:
    """Return exact positive-zero row/head accumulators."""

    f = _zero_f32((JOINT_ACTION_ROWS, TARGET_HEADS))
    return V6RowHeadTotals(
        support=_zero_i32((JOINT_ACTION_ROWS, TARGET_HEADS)),
        absolute_error_sum=f,
        fit_loss_sum=f,
        representation_loss_sum=f,
        representation_gradient_norm_sum=f,
        feature_contribution_abs_sum=f,
        row_bias_abs_sum=f,
        executed_weight_delta_norm_sum=f,
        executed_bias_delta_abs_sum=f,
        proposed_weight_change_count=_zero_i32((JOINT_ACTION_ROWS,)),
        proposed_bias_change_count=_zero_i32((JOINT_ACTION_ROWS,)),
        row_isolation_failure_count=_zero_i32(),
        row_head_algebra_failure_count=_zero_i32(),
        nonfinite_row_head_count=_zero_i32(),
    )


def empty_v6_filter_totals() -> V6FilterTotals:
    """Return exact positive-zero filter accumulators."""

    return V6FilterTotals(
        support=_zero_i32(),
        optimal_value_sum=_zero_f32(),
        selected_value_sum=_zero_f32(),
        selected_regret_sum=_zero_f32(),
        margin_sum=_zero_f32(),
        tied_support=_zero_i32(),
        nontied_support=_zero_i32(),
        tied_selected_regret_sum=_zero_f32(),
        tied_focal_action_support=_zero_i32((2,)),
        cue_pattern_support=_zero_i32((4,)),
        cue_flip_support=_zero_i32((2,)),
        cue_flip_count=_zero_i32((2,)),
        filter_recurrence_failure_count=_zero_i32(),
    )


def empty_v6_action_totals() -> V6ActionTotals:
    """Return exact action accumulators before the initial decision is counted."""

    return V6ActionTotals(
        focal_action_support=_zero_i32((2,)),
        partner_action_support=_zero_i32((2,)),
        joint_row_support=_zero_i32((4,)),
        ordinary_policy_action_support=_zero_i32((2,)),
        explored_count=_zero_i32(),
        externally_forced_count=_zero_i32(),
        policy_schedule_failure_count=_zero_i32(),
        policy_replay_failure_count=_zero_i32(),
        rng_chain_failure_count=_zero_i32(),
        decision_count=jnp.asarray(1, dtype=jnp.int32),
    )


def empty_v6_audit_totals() -> V6AuditTotals:
    """Return exact structural audit counters."""

    return V6AuditTotals(
        contract_failure_counts=_zero_i32((len(V6_CONTRACT_AUDIT_ORDER),)),
        intervention_failure_counts=_zero_i32((len(V6_INTERVENTION_AUDIT_ORDER),)),
        intervention_witness_counts=_zero_i32((len(V6_INTERVENTION_WITNESS_ORDER),)),
        component_delta_sums=_zero_i32((len(V6_COMPONENT_DELTA_ORDER),)),
        component_delta_failure_counts=_zero_i32((len(V6_COMPONENT_DELTA_ORDER),)),
        active_steps=_zero_i32(),
        accepted_steps=_zero_i32(),
        learner_valid_steps=_zero_i32(),
        filter_valid_steps=_zero_i32(),
        oracle_valid_steps=_zero_i32(),
        mechanism_valid_steps=_zero_i32(),
        all_finite_steps=_zero_i32(),
        curation_attempt_count=_zero_i32(),
        ledger_count=_zero_i32(),
        ledger_overflow=jnp.asarray(False, dtype=jnp.bool_),
    )


def empty_v6_cadence_ledger() -> V6CadenceLedger:
    """Return the exact 473-row canonical-sentinel ledger."""

    n = MAX_CADENCE_LEDGER_ENTRIES

    def absent(shape: tuple[int, ...]) -> Array:
        return jnp.full(shape, -1, dtype=jnp.int32)

    return V6CadenceLedger(
        occupied=jnp.zeros((n,), dtype=jnp.bool_),
        transition_step=absent((n,)),
        regime_id=absent((n,)),
        pre_descriptors=absent((n, ACTIVE_PAIR_SLOTS, 2)),
        proposal_descriptors=absent((n, ACTIVE_PAIR_SLOTS, 2)),
        applied_descriptors=absent((n, ACTIVE_PAIR_SLOTS, 2)),
        proposal_event=absent((n, 6)),
        applied_event=absent((n, 6)),
        critical_slot=absent((n, 3, CRITICAL_PAIR_COUNT)),
        critical_candidate_streak=absent((n, CRITICAL_PAIR_COUNT, 2)),
        critical_candidate_flags=jnp.zeros((n, CRITICAL_PAIR_COUNT, 6), dtype=jnp.bool_),
        candidate_reset_mask=jnp.zeros((n, 2, CANDIDATE_PAIR_SLOTS), dtype=jnp.bool_),
        random_curation_flags=jnp.zeros((n, 3), dtype=jnp.bool_),
        random_curation_selected=absent((n, 3)),
        random_active_priorities=_zero_f32((n, ACTIVE_PAIR_SLOTS)),
        random_candidate_priorities=_zero_f32((n, CANDIDATE_PAIR_SLOTS)),
        consumer_masks=jnp.zeros((n, 9, ACTIVE_PAIR_SLOTS), dtype=jnp.bool_),
        router_source_slots=absent((n, ACTIVE_PAIR_SLOTS)),
        router_masks=jnp.zeros((n, 3, ACTIVE_PAIR_SLOTS), dtype=jnp.bool_),
        router_flags=jnp.zeros((n, 4), dtype=jnp.bool_),
        router_counts=absent((n, 4)),
        transaction_exact=jnp.zeros((n,), dtype=jnp.bool_),
        identity_carry_exact=jnp.zeros((n,), dtype=jnp.bool_),
        retired_identity_reset_exact=jnp.zeros((n,), dtype=jnp.bool_),
    )


def empty_v6_lifecycle_chain_state() -> V6LifecycleChainState:
    """Return the initial, structurally valid, outcome-negative chain state."""

    absent = jnp.asarray(-1, dtype=jnp.int32)
    false = jnp.asarray(False, dtype=jnp.bool_)
    return V6LifecycleChainState(
        structural_valid=jnp.asarray(True, dtype=jnp.bool_),
        c_first_acquisition_step=absent,
        c_ever_acquired=false,
        c_retained_after_acquisition=jnp.asarray(True, dtype=jnp.bool_),
        c_acquired_in_first_c=false,
        d_phase=jnp.asarray(0, dtype=jnp.int32),
        d_first_acquisition_step=absent,
        d_retirement_step=absent,
        d_reacquisition_step=absent,
        d_first_acquisition_in_first_d=false,
        d_retirement_while_absent=false,
        d_retirement_reset_exact=false,
        d_reacquisition_in_second_d=false,
        d_reacquisition_gate_observed=false,
        d_ordered_outcome=false,
        out_of_order_event_count=_zero_i32(),
    )


def empty_v6_scan_carry(state: HiddenPartnerWorldOnlineState) -> V6ScanCarry:
    """Construct the complete fixed-shape evaluator carry."""

    return V6ScanCarry(
        bridge_state=state,
        windows=empty_v6_window_totals(),
        row_heads=empty_v6_row_head_totals(),
        filter_totals=empty_v6_filter_totals(),
        action_totals=empty_v6_action_totals(),
        audits=empty_v6_audit_totals(),
        ledger=empty_v6_cadence_ledger(),
        lifecycle=empty_v6_lifecycle_chain_state(),
    )


def v6_window_membership(
    step: Array,
    occurrence_starts: Array,
    occurrence_ends: Array,
    run_steps: Array,
) -> Array:
    """Return exact bool[37] entry/tail/final membership for one scan index."""

    index = jnp.asarray(step, dtype=jnp.int32)
    starts = jnp.asarray(occurrence_starts, dtype=jnp.int32)
    ends = jnp.asarray(occurrence_ends, dtype=jnp.int32)
    limit = jnp.asarray(run_steps, dtype=jnp.int32)
    if starts.shape != (18,) or ends.shape != (18,):
        raise ValueError("occurrence starts and ends must both have shape (18,)")
    entry = (index >= starts) & (index < starts + ENTRY_WINDOW_STEPS)
    tail = (index >= ends - TAIL_WINDOW_STEPS) & (index < ends)
    occurrence = jnp.stack((entry, tail), axis=1).reshape((36,))
    final = (index >= limit - FINAL_WINDOW_STEPS) & (index < limit)
    return jnp.concatenate((occurrence, jnp.reshape(final, (1,))))


def expected_v6_focal_action(step: Array, initial_external_action: int = 0) -> Array:
    """Return the exact alternating action used by the uniform diagnostic."""

    if type(initial_external_action) is not int or initial_external_action not in (0, 1):
        raise ValueError("initial_external_action must be the exact integer 0 or 1")
    index = jnp.asarray(step, dtype=jnp.int32)
    return jnp.bitwise_xor(
        jnp.asarray(initial_external_action, dtype=jnp.int32),
        jnp.bitwise_and(index, jnp.asarray(1, dtype=jnp.int32)),
    )


def pack_v6_stream_code(
    *,
    next_signals: Array,
    partner_flipped: Array,
    world_flipped: Array,
    next_cue_flipped: Array,
    outcome_flipped: Array,
) -> Array:
    """Pack one action-independent transition realization into one uint8."""

    signals = jnp.asarray(next_signals, dtype=jnp.float32)
    cues = jnp.asarray(next_cue_flipped, dtype=jnp.bool_)
    if signals.shape != (3,) or cues.shape != (2,):
        raise ValueError("stream packing requires three signals and two cue bits")
    bits = jnp.concatenate(
        (
            signals > 0.0,
            jnp.reshape(jnp.asarray(partner_flipped, dtype=jnp.bool_), (1,)),
            jnp.reshape(jnp.asarray(world_flipped, dtype=jnp.bool_), (1,)),
            cues,
            jnp.reshape(jnp.asarray(outcome_flipped, dtype=jnp.bool_), (1,)),
        )
    ).astype(jnp.uint8)
    shifts = jnp.arange(8, dtype=jnp.uint8)
    return jnp.sum(jnp.left_shift(bits, shifts), dtype=jnp.uint8)


def reconstruct_v6_stream_code(
    initial_world: HiddenPartnerWorldFeedbackState,
    world_config: HiddenPartnerWorldFeedbackConfig,
    run_steps: int,
) -> Array:
    """Replay the exact action-independent world draws into a fixed stream."""

    if type(initial_world) is not HiddenPartnerWorldFeedbackState:
        raise TypeError("initial_world must be an exact HiddenPartnerWorldFeedbackState")
    if type(world_config) is not HiddenPartnerWorldFeedbackConfig:
        raise TypeError("world_config must be an exact HiddenPartnerWorldFeedbackConfig")
    if type(run_steps) is not int:
        raise TypeError("run_steps must be an exact built-in int")
    if not 0 <= run_steps <= MAX_SCAN_STEPS:
        raise ValueError("run_steps must lie in the fixed scan range")

    initial_keys = (
        initial_world.signal_key,
        initial_world.partner_key,
        initial_world.world_key,
        initial_world.cue_key,
        initial_world.outcome_key,
    )
    partner_probability = jnp.asarray(
        world_config.partner_flip_probability,
        dtype=jnp.float32,
    )
    world_probability = jnp.asarray(
        world_config.world_flip_probability,
        dtype=jnp.float32,
    )
    cue_probabilities = jnp.asarray(
        world_config.cue_flip_probabilities,
        dtype=jnp.float32,
    )
    outcome_probability = jnp.asarray(
        world_config.outcome_flip_probability,
        dtype=jnp.float32,
    )

    def body(
        keys: tuple[Array, Array, Array, Array, Array],
        _: None,
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        signal_key, partner_key, world_key, cue_key, outcome_key = keys
        next_signal_key, signal_sample_key = jr.split(signal_key)
        next_partner_key, partner_flip_key = jr.split(partner_key)
        next_world_key, world_flip_key = jr.split(world_key)
        next_cue_key, cue_sample_key = jr.split(cue_key)
        next_outcome_key, outcome_flip_key = jr.split(outcome_key)
        positive_signals = jr.bernoulli(
            signal_sample_key,
            p=0.5,
            shape=(3,),
        )
        partner_flipped = jr.bernoulli(partner_flip_key, p=partner_probability)
        world_flipped = jr.bernoulli(world_flip_key, p=world_probability)
        next_cue_flipped = jr.bernoulli(
            cue_sample_key,
            p=cue_probabilities,
            shape=(2,),
        )
        outcome_flipped = jr.bernoulli(outcome_flip_key, p=outcome_probability)
        code = pack_v6_stream_code(
            next_signals=jnp.where(positive_signals, 1.0, -1.0).astype(jnp.float32),
            partner_flipped=partner_flipped,
            world_flipped=world_flipped,
            next_cue_flipped=next_cue_flipped,
            outcome_flipped=outcome_flipped,
        )
        return (
            (
                next_signal_key,
                next_partner_key,
                next_world_key,
                next_cue_key,
                next_outcome_key,
            ),
            code,
        )

    _, active = jax.lax.scan(body, initial_keys, xs=None, length=run_steps)
    return jnp.concatenate(
        (
            active.astype(jnp.uint8),
            jnp.zeros((MAX_SCAN_STEPS - run_steps,), dtype=jnp.uint8),
        )
    )


def _descriptor_presence_and_slot(descriptors: Array, pair: tuple[int, int]) -> tuple[Array, Array]:
    target = jnp.asarray(pair, dtype=jnp.int32)
    matches = jnp.all(jnp.asarray(descriptors, dtype=jnp.int32) == target, axis=1)
    present = jnp.any(matches)
    slot = jnp.where(present, jnp.argmax(matches).astype(jnp.int32), -1)
    return present, slot


def _descriptor_bank_valid(descriptors: Array) -> Array:
    values = jnp.asarray(descriptors, dtype=jnp.int32)
    left = values[:, 0]
    right = values[:, 1]
    inactive = (left == -1) & (right == -1)
    live = (left >= 0) & (left < right) & (right < BASE_FEATURE_DIM)
    domain = jnp.all(inactive | live)
    codes = left * BASE_FEATURE_DIM + right
    duplicate = live[:, None] & live[None, :] & (codes[:, None] == codes[None, :])
    duplicate = duplicate & ~jnp.eye(ACTIVE_PAIR_SLOTS, dtype=jnp.bool_)
    return domain & ~jnp.any(duplicate)


def update_v6_lifecycle_chain(
    state: V6LifecycleChainState,
    observation: V6LifecycleObservation,
) -> V6LifecycleChainState:
    """Update C/D outcomes without turning a negative outcome into invalidity."""

    c_pre, _ = _descriptor_presence_and_slot(observation.pre_descriptors, C_PAIR)
    c_post, _ = _descriptor_presence_and_slot(observation.applied_descriptors, C_PAIR)
    d_pre, _ = _descriptor_presence_and_slot(observation.pre_descriptors, D_PAIR)
    d_post, _ = _descriptor_presence_and_slot(observation.applied_descriptors, D_PAIR)
    descriptor_valid = _descriptor_bank_valid(observation.pre_descriptors) & _descriptor_bank_valid(
        observation.applied_descriptors
    )
    structural_valid = state.structural_valid & observation.structural_valid & descriptor_valid

    c_acquired = ~c_pre & c_post
    c_lost = c_pre & ~c_post
    first_c = c_acquired & ~state.c_ever_acquired
    c_first_step = jnp.where(first_c, observation.transition_step, state.c_first_acquisition_step)
    c_ever = state.c_ever_acquired | c_acquired
    c_retained = state.c_retained_after_acquisition & ~(state.c_ever_acquired & c_lost)
    c_acquired_first = state.c_acquired_in_first_c | (first_c & (observation.occurrence_index == 5))

    d_acquired = ~d_pre & d_post
    d_disappeared = d_pre & ~d_post
    explicit_d_retirement = jnp.all(
        observation.retired_pair == jnp.asarray(D_PAIR, dtype=jnp.int32)
    )
    exact_d_retirement = d_disappeared & explicit_d_retirement & observation.d_reset_exact
    first_acquisition = (state.d_phase == 0) & d_acquired
    ordered_retirement = (state.d_phase == 1) & exact_d_retirement
    ordered_reacquisition = (state.d_phase == 2) & d_acquired
    out_of_order = (
        ((state.d_phase == 0) & d_disappeared)
        | ((state.d_phase == 1) & (d_acquired | (d_disappeared & ~exact_d_retirement)))
        | ((state.d_phase == 2) & d_disappeared)
        | ((state.d_phase == 3) & (d_acquired | d_disappeared))
    )
    phase = jnp.where(
        first_acquisition,
        1,
        jnp.where(ordered_retirement, 2, jnp.where(ordered_reacquisition, 3, state.d_phase)),
    ).astype(jnp.int32)
    first_in_d = state.d_first_acquisition_in_first_d | (
        first_acquisition
        & (observation.occurrence_index == 3)
        & (observation.promoted_candidate == CRITICAL_CANDIDATE_INDICES[1])
    )
    retired_absent = state.d_retirement_while_absent | (
        ordered_retirement & (observation.regime_id != 3)
    )
    retirement_reset = state.d_retirement_reset_exact | (
        ordered_retirement & observation.d_reset_exact
    )
    reacquired_second = state.d_reacquisition_in_second_d | (
        ordered_reacquisition
        & (observation.occurrence_index == 12)
        & (observation.promoted_candidate == CRITICAL_CANDIDATE_INDICES[1])
    )
    reacquisition_gate = state.d_reacquisition_gate_observed | (
        ordered_reacquisition
        & observation.d_reacquisition_required_pre
        & observation.d_reacquisition_confirmed
    )
    ordered_outcome = (
        (phase == 3)
        & first_in_d
        & retired_absent
        & retirement_reset
        & reacquired_second
        & reacquisition_gate
    )
    return V6LifecycleChainState(
        structural_valid=structural_valid,
        c_first_acquisition_step=c_first_step,
        c_ever_acquired=c_ever,
        c_retained_after_acquisition=c_retained,
        c_acquired_in_first_c=c_acquired_first,
        d_phase=phase,
        d_first_acquisition_step=jnp.where(
            first_acquisition,
            observation.transition_step,
            state.d_first_acquisition_step,
        ),
        d_retirement_step=jnp.where(
            ordered_retirement,
            observation.transition_step,
            state.d_retirement_step,
        ),
        d_reacquisition_step=jnp.where(
            ordered_reacquisition,
            observation.transition_step,
            state.d_reacquisition_step,
        ),
        d_first_acquisition_in_first_d=first_in_d,
        d_retirement_while_absent=retired_absent,
        d_retirement_reset_exact=retirement_reset,
        d_reacquisition_in_second_d=reacquired_second,
        d_reacquisition_gate_observed=reacquisition_gate,
        d_ordered_outcome=ordered_outcome,
        out_of_order_event_count=(state.out_of_order_event_count + out_of_order.astype(jnp.int32)),
    )


def record_v6_cadence_observation(
    ledger: V6CadenceLedger,
    count: Array,
    observation: V6CadenceObservation,
) -> tuple[V6CadenceLedger, Array, Array]:
    """Insert one cadence row or preserve the full ledger on overflow."""

    index = jnp.asarray(count, dtype=jnp.int32)
    capacity = jnp.asarray(MAX_CADENCE_LEDGER_ENTRIES, dtype=jnp.int32)
    available = (index >= 0) & (index < capacity)
    safe_index = jnp.clip(index, 0, MAX_CADENCE_LEDGER_ENTRIES - 1)
    proposed = ledger.replace(
        occupied=ledger.occupied.at[safe_index].set(True),
        transition_step=ledger.transition_step.at[safe_index].set(observation.transition_step),
        regime_id=ledger.regime_id.at[safe_index].set(observation.regime_id),
        pre_descriptors=ledger.pre_descriptors.at[safe_index].set(observation.pre_descriptors),
        proposal_descriptors=ledger.proposal_descriptors.at[safe_index].set(
            observation.proposal_descriptors
        ),
        applied_descriptors=ledger.applied_descriptors.at[safe_index].set(
            observation.applied_descriptors
        ),
        proposal_event=ledger.proposal_event.at[safe_index].set(observation.proposal_event),
        applied_event=ledger.applied_event.at[safe_index].set(observation.applied_event),
        critical_slot=ledger.critical_slot.at[safe_index].set(observation.critical_slot),
        critical_candidate_streak=ledger.critical_candidate_streak.at[safe_index].set(
            observation.critical_candidate_streak
        ),
        critical_candidate_flags=ledger.critical_candidate_flags.at[safe_index].set(
            observation.critical_candidate_flags
        ),
        candidate_reset_mask=ledger.candidate_reset_mask.at[safe_index].set(
            observation.candidate_reset_mask
        ),
        random_curation_flags=ledger.random_curation_flags.at[safe_index].set(
            observation.random_curation_flags
        ),
        random_curation_selected=ledger.random_curation_selected.at[safe_index].set(
            observation.random_curation_selected
        ),
        random_active_priorities=ledger.random_active_priorities.at[safe_index].set(
            observation.random_active_priorities
        ),
        random_candidate_priorities=ledger.random_candidate_priorities.at[safe_index].set(
            observation.random_candidate_priorities
        ),
        consumer_masks=ledger.consumer_masks.at[safe_index].set(observation.consumer_masks),
        router_source_slots=ledger.router_source_slots.at[safe_index].set(
            observation.router_source_slots
        ),
        router_masks=ledger.router_masks.at[safe_index].set(observation.router_masks),
        router_flags=ledger.router_flags.at[safe_index].set(observation.router_flags),
        router_counts=ledger.router_counts.at[safe_index].set(observation.router_counts),
        transaction_exact=ledger.transaction_exact.at[safe_index].set(
            observation.transaction_exact
        ),
        identity_carry_exact=ledger.identity_carry_exact.at[safe_index].set(
            observation.identity_carry_exact
        ),
        retired_identity_reset_exact=ledger.retired_identity_reset_exact.at[safe_index].set(
            observation.retired_identity_reset_exact
        ),
    )
    selected = cast(
        V6CadenceLedger,
        jax.lax.cond(available, lambda _: proposed, lambda _: ledger, operand=None),
    )
    next_count = jnp.where(available, index + 1, jnp.minimum(jnp.maximum(index, 0), capacity))
    return selected, next_count.astype(jnp.int32), ~available


def update_v6_row_head_totals(
    totals: V6RowHeadTotals,
    *,
    accepted: Array,
    grounded_valid: Array,
    executed_row: Array,
    feature_contribution: Array,
    row_bias: Array,
    raw_predictions: Array,
    targets: Array,
    errors: Array,
    fit_loss_by_head: Array,
    representation_loss_by_head: Array,
    representation_gradient: Array,
    representation_gradient_by_head: Array,
    representation_gradient_norm_by_head: Array,
    proposed_weight_change_mask: Array,
    proposed_bias_change_mask: Array,
    executed_weight_delta_norm_by_head: Array,
    executed_bias_delta_by_head: Array,
    row_update_isolated: Array,
    target_weights: Array,
) -> tuple[V6RowHeadTotals, Array]:
    """Accumulate one executed row and return its exact algebra verdict."""

    participate = jnp.asarray(accepted, dtype=jnp.bool_) & jnp.asarray(
        grounded_valid, dtype=jnp.bool_
    )
    row = jnp.asarray(executed_row, dtype=jnp.int32)
    feature = jnp.asarray(feature_contribution, dtype=jnp.float32)
    bias = jnp.asarray(row_bias, dtype=jnp.float32)
    raw = jnp.asarray(raw_predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    error = jnp.asarray(errors, dtype=jnp.float32)
    fit = jnp.asarray(fit_loss_by_head, dtype=jnp.float32)
    representation_loss = jnp.asarray(representation_loss_by_head, dtype=jnp.float32)
    gradient = jnp.asarray(representation_gradient, dtype=jnp.float32)
    gradient_by_head = jnp.asarray(representation_gradient_by_head, dtype=jnp.float32)
    gradient_norms = jnp.asarray(representation_gradient_norm_by_head, dtype=jnp.float32)
    weight_mask = jnp.asarray(proposed_weight_change_mask, dtype=jnp.bool_)
    bias_mask = jnp.asarray(proposed_bias_change_mask, dtype=jnp.bool_)
    weight_delta = jnp.asarray(executed_weight_delta_norm_by_head, dtype=jnp.float32)
    bias_delta = jnp.asarray(executed_bias_delta_by_head, dtype=jnp.float32)
    weights = jnp.asarray(target_weights, dtype=jnp.float32)
    expected_shapes = (
        feature.shape == (TARGET_HEADS,),
        bias.shape == (TARGET_HEADS,),
        raw.shape == (TARGET_HEADS,),
        target.shape == (TARGET_HEADS,),
        error.shape == (TARGET_HEADS,),
        fit.shape == (TARGET_HEADS,),
        representation_loss.shape == (TARGET_HEADS,),
        gradient.shape == (DEPLOYED_FEATURE_DIM,),
        gradient_by_head.shape == (TARGET_HEADS, DEPLOYED_FEATURE_DIM),
        gradient_norms.shape == (TARGET_HEADS,),
        weight_mask.shape == (JOINT_ACTION_ROWS,),
        bias_mask.shape == (JOINT_ACTION_ROWS,),
        weight_delta.shape == (TARGET_HEADS,),
        bias_delta.shape == (TARGET_HEADS,),
        weights.shape == (TARGET_HEADS,),
    )
    if not all(expected_shapes):
        raise ValueError("row/head update arrays differ from the fixed v6 shapes")
    safe_row = jnp.clip(row, 0, JOINT_ACTION_ROWS - 1)
    row_one_hot = jax.nn.one_hot(safe_row, JOINT_ACTION_ROWS, dtype=jnp.int32)
    allowed_row = row_one_hot.astype(jnp.bool_)
    finite = jnp.all(
        jnp.stack(
            (
                jnp.all(jnp.isfinite(feature)),
                jnp.all(jnp.isfinite(bias)),
                jnp.all(jnp.isfinite(raw)),
                jnp.all(jnp.isfinite(target)),
                jnp.all(jnp.isfinite(error)),
                jnp.all(jnp.isfinite(fit)),
                jnp.all(jnp.isfinite(representation_loss)),
                jnp.all(jnp.isfinite(gradient)),
                jnp.all(jnp.isfinite(gradient_by_head)),
                jnp.all(jnp.isfinite(gradient_norms)),
                jnp.all(jnp.isfinite(weight_delta)),
                jnp.all(jnp.isfinite(bias_delta)),
            )
        )
    )
    divisor = jnp.asarray(TARGET_HEADS, dtype=jnp.float32)
    algebra = (
        (row >= 0)
        & (row < JOINT_ACTION_ROWS)
        & finite
        & jnp.array_equal(raw, feature + bias)
        & jnp.array_equal(error, raw - target)
        & jnp.array_equal(fit, 0.5 * jnp.square(error) / divisor)
        & jnp.array_equal(representation_loss, fit * weights)
        & jnp.allclose(
            gradient,
            jnp.sum(gradient_by_head, axis=0),
            atol=2e-6,
            rtol=1e-6,
        )
        & jnp.allclose(
            gradient_norms,
            jnp.linalg.norm(gradient_by_head, axis=1),
            atol=2e-6,
            rtol=1e-6,
        )
        & ~jnp.any(weight_mask & ~allowed_row)
        & ~jnp.any(bias_mask & ~allowed_row)
        & jnp.asarray(row_update_isolated, dtype=jnp.bool_)
    )
    accepted_algebra = ~participate | algebra
    include = participate & finite & (row >= 0) & (row < JOINT_ACTION_ROWS)
    cell_mask_i = (row_one_hot[:, None] * include.astype(jnp.int32)).astype(jnp.int32)
    cell_mask_f = cell_mask_i.astype(jnp.float32)
    return (
        totals.replace(
            support=totals.support + cell_mask_i,
            absolute_error_sum=totals.absolute_error_sum + cell_mask_f * jnp.abs(error)[None, :],
            fit_loss_sum=totals.fit_loss_sum + cell_mask_f * fit[None, :],
            representation_loss_sum=(
                totals.representation_loss_sum + cell_mask_f * representation_loss[None, :]
            ),
            representation_gradient_norm_sum=(
                totals.representation_gradient_norm_sum + cell_mask_f * gradient_norms[None, :]
            ),
            feature_contribution_abs_sum=(
                totals.feature_contribution_abs_sum + cell_mask_f * jnp.abs(feature)[None, :]
            ),
            row_bias_abs_sum=totals.row_bias_abs_sum + cell_mask_f * jnp.abs(bias)[None, :],
            executed_weight_delta_norm_sum=(
                totals.executed_weight_delta_norm_sum + cell_mask_f * weight_delta[None, :]
            ),
            executed_bias_delta_abs_sum=(
                totals.executed_bias_delta_abs_sum + cell_mask_f * jnp.abs(bias_delta)[None, :]
            ),
            proposed_weight_change_count=(
                totals.proposed_weight_change_count + (weight_mask & participate).astype(jnp.int32)
            ),
            proposed_bias_change_count=(
                totals.proposed_bias_change_count + (bias_mask & participate).astype(jnp.int32)
            ),
            row_isolation_failure_count=(
                totals.row_isolation_failure_count
                + (participate & ~jnp.asarray(row_update_isolated, dtype=jnp.bool_)).astype(
                    jnp.int32
                )
            ),
            row_head_algebra_failure_count=(
                totals.row_head_algebra_failure_count + (participate & ~algebra).astype(jnp.int32)
            ),
            nonfinite_row_head_count=(
                totals.nonfinite_row_head_count + (participate & ~finite).astype(jnp.int32)
            ),
        ),
        accepted_algebra,
    )


def _safe_action_value(values: Array, action: Array) -> Array:
    """Gather one binary-action value without permitting an invalid index."""

    index = jnp.clip(jnp.asarray(action, dtype=jnp.int32), 0, 1)
    return jnp.asarray(values, dtype=jnp.float32)[index]


def _binary_metric_terms(probabilities: Array, target: Array) -> tuple[Array, Array, Array]:
    """Return finite prequential NLL, Brier score, and correctness."""

    values = jnp.asarray(probabilities, dtype=jnp.float32)
    target_index = jnp.clip(jnp.asarray(target, dtype=jnp.int32), 0, 1)
    target_probability = jnp.clip(values[target_index], 1e-7, 1.0)
    one_hot = jax.nn.one_hot(target_index, 2, dtype=jnp.float32)
    nll = -jnp.log(target_probability)
    brier = jnp.sum(jnp.square(values - one_hot))
    correct = jnp.argmax(values).astype(jnp.int32) == target_index
    return nll, brier, correct


def _critical_slot_values(
    descriptors: Array,
    values: Array,
) -> tuple[Array, Array]:
    """Gather active-slot values for C and D with absent entries neutralized."""

    c_present, c_slot = _descriptor_presence_and_slot(descriptors, C_PAIR)
    d_present, d_slot = _descriptor_presence_and_slot(descriptors, D_PAIR)
    present = jnp.stack((c_present, d_present))
    slots = jnp.stack((c_slot, d_slot)).astype(jnp.int32)
    gathered = jnp.asarray(values)[jnp.clip(slots, 0, ACTIVE_PAIR_SLOTS - 1)]
    neutral = jnp.zeros_like(gathered)
    return present, jnp.where(present, gathered, neutral)


def _update_v6_window_totals(
    totals: V6WindowTotals,
    *,
    membership: Array,
    accepted: Array,
    pre_state: HiddenPartnerWorldOnlineState,
    trace: HiddenPartnerWorldOnlineTrace,
) -> V6WindowTotals:
    """Accumulate one transition into all overlapping geometry windows."""

    member = jnp.asarray(membership, dtype=jnp.bool_)
    include = member & jnp.asarray(accepted, dtype=jnp.bool_)
    member_i = member.astype(jnp.int32)
    include_i = include.astype(jnp.int32)
    include_f = include.astype(jnp.float32)

    behavior_probabilities = pre_state.agent.current_evaluation.predicted_partner_probabilities
    behavior_nll, behavior_brier, behavior_correct = _binary_metric_terms(
        behavior_probabilities,
        trace.partner_action,
    )
    filter_values = trace.agent_partner_belief_conditioned_expected_rewards
    filter_optimal = jnp.max(filter_values)
    planner_action = pre_state.agent.current_evaluation.greedy_action
    filter_planner_regret = filter_optimal - _safe_action_value(
        filter_values,
        planner_action,
    )
    full_information_values = trace.oracle_realized_counterfactual_rewards
    full_information_optimal = jnp.max(full_information_values)
    full_information_selected_regret = full_information_optimal - _safe_action_value(
        full_information_values,
        trace.focal_action,
    )
    full_information_planner_regret = full_information_optimal - _safe_action_value(
        full_information_values,
        planner_action,
    )

    world_probability = jnp.clip(
        0.5 * (1.0 + trace.oracle_world_sign * trace.filter_mean_pre),
        1e-7,
        1.0,
    )
    world_nll = -jnp.log(world_probability)
    world_brier = jnp.square(1.0 - world_probability)

    mechanism = trace.mechanism
    critical_present, critical_durable = _critical_slot_values(
        mechanism.lifecycle_pre_descriptors,
        mechanism.lifecycle_durable_read_mask,
    )
    _, critical_refreshed = _critical_slot_values(
        mechanism.lifecycle_pre_descriptors,
        mechanism.lifecycle_active_evidence_refreshed,
    )
    _, critical_scores = _critical_slot_values(
        mechanism.lifecycle_pre_descriptors,
        mechanism.lifecycle_relevance_probe_scores,
    )
    _, critical_errors = _critical_slot_values(
        mechanism.lifecycle_pre_descriptors,
        jnp.abs(mechanism.lifecycle_relevance_probe_errors[0]),
    )
    grounded_valid = (
        jnp.asarray(accepted, dtype=jnp.bool_)
        & mechanism.grounded_enabled
        & mechanism.grounded_prediction_valid
        & mechanism.grounded_target_valid
        & mechanism.grounded_gradient_valid
    )
    grounded_i = (member & grounded_valid).astype(jnp.int32)
    grounded_f = grounded_i.astype(jnp.float32)
    critical_mask_i = include_i[:, None] * critical_present.astype(jnp.int32)[None, :]
    critical_mask_f = critical_mask_i.astype(jnp.float32)
    return totals.replace(
        scheduled_support=totals.scheduled_support + member_i,
        accepted_support=totals.accepted_support + include_i,
        reward_sum=totals.reward_sum + include_f * trace.reward,
        behavior_nll_sum=totals.behavior_nll_sum + include_f * behavior_nll,
        behavior_brier_sum=totals.behavior_brier_sum + include_f * behavior_brier,
        behavior_correct_count=(
            totals.behavior_correct_count + include_i * behavior_correct.astype(jnp.int32)
        ),
        filter_selected_regret_sum=(
            totals.filter_selected_regret_sum
            + include_f * trace.agent_partner_belief_conditioned_selected_regret
        ),
        filter_planner_regret_sum=(
            totals.filter_planner_regret_sum + include_f * filter_planner_regret
        ),
        full_information_selected_regret_sum=(
            totals.full_information_selected_regret_sum
            + include_f * full_information_selected_regret
        ),
        full_information_planner_regret_sum=(
            totals.full_information_planner_regret_sum + include_f * full_information_planner_regret
        ),
        world_posterior_nll_sum=(totals.world_posterior_nll_sum + include_f * world_nll),
        world_posterior_brier_sum=(totals.world_posterior_brier_sum + include_f * world_brier),
        grounded_support=totals.grounded_support + grounded_i,
        grounded_fit_loss_by_head_sum=(
            totals.grounded_fit_loss_by_head_sum
            + grounded_f[:, None] * mechanism.grounded_fit_loss_by_head[None, :]
        ),
        grounded_representation_loss_by_head_sum=(
            totals.grounded_representation_loss_by_head_sum
            + grounded_f[:, None] * mechanism.grounded_representation_loss_by_head[None, :]
        ),
        grounded_representation_gradient_norm_by_head_sum=(
            totals.grounded_representation_gradient_norm_by_head_sum
            + grounded_f[:, None] * mechanism.grounded_representation_gradient_norm_by_head[None, :]
        ),
        critical_present_count=totals.critical_present_count + critical_mask_i,
        critical_durable_read_count=(
            totals.critical_durable_read_count
            + critical_mask_i * critical_durable.astype(jnp.int32)[None, :]
        ),
        critical_evidence_refresh_count=(
            totals.critical_evidence_refresh_count
            + critical_mask_i * critical_refreshed.astype(jnp.int32)[None, :]
        ),
        critical_relevance_score_sum=(
            totals.critical_relevance_score_sum + critical_mask_f * critical_scores[None, :]
        ),
        critical_relevance_error_abs_sum=(
            totals.critical_relevance_error_abs_sum + critical_mask_f * critical_errors[None, :]
        ),
    )


def _update_v6_filter_totals(
    totals: V6FilterTotals,
    *,
    accepted: Array,
    trace: HiddenPartnerWorldOnlineTrace,
) -> V6FilterTotals:
    include = jnp.asarray(accepted, dtype=jnp.bool_) & trace.filter_trace_valid
    include_i = include.astype(jnp.int32)
    include_f = include.astype(jnp.float32)
    values = trace.agent_partner_belief_conditioned_expected_rewards
    optimal = jnp.max(values)
    selected = _safe_action_value(values, trace.focal_action)
    regret = optimal - selected
    tied = include & trace.agent_partner_belief_conditioned_tied
    focal = jnp.clip(trace.focal_action, 0, 1)
    focal_one_hot = jax.nn.one_hot(focal, 2, dtype=jnp.int32)
    cues = trace.observation_pre[jnp.asarray((6, 7), dtype=jnp.int32)] > 0.0
    cue_pattern = cues[0].astype(jnp.int32) + 2 * cues[1].astype(jnp.int32)
    cue_one_hot = jax.nn.one_hot(cue_pattern, 4, dtype=jnp.int32)
    return totals.replace(
        support=totals.support + include_i,
        optimal_value_sum=totals.optimal_value_sum + include_f * optimal,
        selected_value_sum=totals.selected_value_sum + include_f * selected,
        selected_regret_sum=totals.selected_regret_sum + include_f * regret,
        margin_sum=(
            totals.margin_sum + include_f * trace.agent_partner_belief_conditioned_action_margin
        ),
        tied_support=totals.tied_support + tied.astype(jnp.int32),
        nontied_support=totals.nontied_support + (include & ~tied).astype(jnp.int32),
        tied_selected_regret_sum=(
            totals.tied_selected_regret_sum + tied.astype(jnp.float32) * regret
        ),
        tied_focal_action_support=(
            totals.tied_focal_action_support + focal_one_hot * tied.astype(jnp.int32)
        ),
        cue_pattern_support=totals.cue_pattern_support + cue_one_hot * include_i,
        cue_flip_support=totals.cue_flip_support + jnp.full((2,), include_i),
        cue_flip_count=(
            totals.cue_flip_count + trace.oracle_world_cue_flipped.astype(jnp.int32) * include_i
        ),
        filter_recurrence_failure_count=(
            totals.filter_recurrence_failure_count
            + (jnp.asarray(accepted, dtype=jnp.bool_) & ~trace.filter_trace_valid).astype(jnp.int32)
        ),
    )


def _update_v6_action_totals(
    totals: V6ActionTotals,
    *,
    accepted: Array,
    trace: HiddenPartnerWorldOnlineTrace,
    initial_external_action: int,
    balanced_external: bool,
) -> tuple[V6ActionTotals, Array]:
    include = jnp.asarray(accepted, dtype=jnp.bool_)
    include_i = include.astype(jnp.int32)
    focal = jnp.clip(trace.focal_action, 0, 1)
    partner = jnp.clip(trace.partner_action, 0, 1)
    row = 2 * focal + partner
    ordinary = jnp.clip(trace.focal_action_ordinary_policy_action, 0, 1)
    alternating_expected = expected_v6_focal_action(trace.step, initial_external_action)
    external_phase_valid = (~jnp.asarray(balanced_external, dtype=jnp.bool_)) | (
        trace.expected_focal_action == alternating_expected
    )
    schedule_valid = ~include | (
        (trace.focal_action == trace.expected_focal_action) & external_phase_valid
    )
    policy_replay_valid = ~include | (
        trace.action_policy_valid
        & trace.selection_binding_valid
        & trace.policy_replay_valid
        & trace.next_action_policy_valid
        & trace.next_selection_binding_valid
        & trace.next_policy_replay_valid
    )
    rng_chain_valid = ~include | jnp.array_equal(
        trace.focal_action_policy_rng_after,
        trace.next_action_policy_rng_before,
    )
    return (
        totals.replace(
            focal_action_support=(
                totals.focal_action_support + jax.nn.one_hot(focal, 2, dtype=jnp.int32) * include_i
            ),
            partner_action_support=(
                totals.partner_action_support
                + jax.nn.one_hot(partner, 2, dtype=jnp.int32) * include_i
            ),
            joint_row_support=(
                totals.joint_row_support
                + jax.nn.one_hot(row, JOINT_ACTION_ROWS, dtype=jnp.int32) * include_i
            ),
            ordinary_policy_action_support=(
                totals.ordinary_policy_action_support
                + jax.nn.one_hot(ordinary, 2, dtype=jnp.int32) * include_i
            ),
            explored_count=totals.explored_count
            + (include & trace.focal_action_explored).astype(jnp.int32),
            externally_forced_count=(
                totals.externally_forced_count
                + (include & trace.focal_action_externally_forced).astype(jnp.int32)
            ),
            policy_schedule_failure_count=(
                totals.policy_schedule_failure_count + (include & ~schedule_valid).astype(jnp.int32)
            ),
            policy_replay_failure_count=(
                totals.policy_replay_failure_count
                + (include & ~policy_replay_valid).astype(jnp.int32)
            ),
            rng_chain_failure_count=(
                totals.rng_chain_failure_count + (include & ~rng_chain_valid).astype(jnp.int32)
            ),
            decision_count=totals.decision_count + include_i,
        ),
        rng_chain_valid,
    )


def _critical_slots(descriptors: Array) -> Array:
    return jnp.stack(
        tuple(_descriptor_presence_and_slot(descriptors, pair)[1] for pair in CRITICAL_PAIRS)
    ).astype(jnp.int32)


def _candidate_index(left: Array, right: Array) -> Array:
    left_value = jnp.asarray(left, dtype=jnp.int32)
    right_value = jnp.asarray(right, dtype=jnp.int32)
    return (
        left_value * (2 * BASE_FEATURE_DIM - left_value - 1) // 2 + right_value - left_value - 1
    ).astype(jnp.int32)


def _retired_identity_reset_exact(mechanism: Any) -> Array:
    """Check both consumer destinations and candidate archive after retirement."""

    left = mechanism.lifecycle_applied_retired_left
    right = mechanism.lifecycle_applied_retired_right
    retired = (left >= 0) & (left < right) & (right < BASE_FEATURE_DIM)
    candidate = _candidate_index(left, right)
    safe_candidate = jnp.clip(candidate, 0, CANDIDATE_PAIR_SLOTS - 1)
    expected = (
        jax.nn.one_hot(
            safe_candidate,
            CANDIDATE_PAIR_SLOTS,
            dtype=jnp.bool_,
        )
        & retired
    )
    actual = mechanism.lifecycle_applied_candidate_reset_mask
    reset_values = (
        mechanism.lifecycle_candidate_promotion_evidence_streak_post[safe_candidate] == 0
    ) & mechanism.lifecycle_candidate_reacquisition_required_post[safe_candidate]
    return (
        mechanism.consumer_lifecycle_destination_reset_exact
        & jnp.array_equal(actual, expected)
        & (~retired | reset_values)
    )


def _router_transaction_exact(
    pre_state: HiddenPartnerWorldOnlineState,
    post_state: HiddenPartnerWorldOnlineState,
    trace: HiddenPartnerWorldOnlineTrace,
) -> Array:
    mechanism = trace.mechanism
    return (
        _descriptor_bank_valid(mechanism.lifecycle_pre_descriptors)
        & _descriptor_bank_valid(mechanism.lifecycle_proposal_descriptors)
        & _descriptor_bank_valid(mechanism.lifecycle_applied_descriptors)
        & jnp.array_equal(
            mechanism.lifecycle_pre_descriptors,
            pre_state.agent.router.descriptors,
        )
        & jnp.array_equal(
            mechanism.lifecycle_applied_descriptors,
            post_state.agent.router.descriptors,
        )
        & jnp.array_equal(
            mechanism.router_descriptors_changed,
            jnp.any(mechanism.lifecycle_pre_descriptors != mechanism.lifecycle_applied_descriptors),
        )
        & mechanism.router_valid
        & mechanism.router_applied
    )


def expected_v6_carry_survivors(
    configured_carry_survivors: bool,
    descriptors_changed: Array,
) -> Array:
    """Return core router carry semantics, including unchanged-route vacuity."""

    return jnp.asarray(configured_carry_survivors, dtype=jnp.bool_) | ~jnp.asarray(
        descriptors_changed,
        dtype=jnp.bool_,
    )


def _identity_carry_exact(mechanism: Any, *, carry_survivors: bool) -> Array:
    expected_carry = expected_v6_carry_survivors(
        carry_survivors,
        mechanism.router_descriptors_changed,
    )
    sources = mechanism.router_source_slots
    survivor = mechanism.router_survivor_mask
    source_domain = jnp.all((~survivor) | ((sources >= 0) & (sources < ACTIVE_PAIR_SLOTS)))
    destination_partition = ~jnp.any(mechanism.router_survivor_mask & mechanism.router_new_mask)
    consumer_values_exact = jnp.all(
        jnp.stack(
            (
                mechanism.consumer_route_source_slots_exact,
                mechanism.consumer_route_identity_masks_exact,
                mechanism.consumer_route_stable_prefix_exact,
                mechanism.consumer_route_survivor_values_exact,
                mechanism.consumer_route_reset_values_exact,
                mechanism.consumer_route_no_carry_reset_exact,
                mechanism.consumer_route_behavior_values_exact,
                mechanism.consumer_route_q_values_exact,
                mechanism.consumer_route_trace_values_exact,
                mechanism.consumer_route_last_observation_exact,
                mechanism.consumer_route_grounded_values_exact,
                mechanism.consumer_route_values_exact,
            )
        )
    )
    return (
        mechanism.router_valid
        & (mechanism.router_carry_survivors == expected_carry)
        & source_domain
        & destination_partition
        & consumer_values_exact
    )


def _cadence_observation(
    *,
    transition_step: Array,
    regime_id: Array,
    pre_state: HiddenPartnerWorldOnlineState,
    post_state: HiddenPartnerWorldOnlineState,
    trace: HiddenPartnerWorldOnlineTrace,
    carry_survivors: bool,
) -> V6CadenceObservation:
    mechanism = trace.mechanism
    candidates = jnp.asarray(CRITICAL_CANDIDATE_INDICES, dtype=jnp.int32)
    pre_slots = _critical_slots(mechanism.lifecycle_pre_descriptors)
    proposal_slots = _critical_slots(mechanism.lifecycle_proposal_descriptors)
    applied_slots = _critical_slots(mechanism.lifecycle_applied_descriptors)
    streak = jnp.stack(
        (
            mechanism.lifecycle_candidate_promotion_evidence_streak_pre[candidates],
            mechanism.lifecycle_candidate_promotion_evidence_streak_post[candidates],
        ),
        axis=1,
    )
    flags = jnp.stack(
        (
            mechanism.lifecycle_candidate_promotion_raw_evidence[candidates],
            mechanism.lifecycle_candidate_promotion_confirmed[candidates],
            mechanism.lifecycle_candidate_reacquisition_required_pre[candidates],
            mechanism.lifecycle_candidate_reacquisition_required_proposal_post[candidates],
            mechanism.lifecycle_candidate_reacquisition_required_post[candidates],
            mechanism.lifecycle_candidate_reacquisition_confirmed[candidates],
        ),
        axis=1,
    )
    transaction_exact = _router_transaction_exact(pre_state, post_state, trace)
    identity_exact = _identity_carry_exact(
        mechanism,
        carry_survivors=carry_survivors,
    )
    reset_exact = _retired_identity_reset_exact(mechanism)
    return V6CadenceObservation(
        transition_step=jnp.asarray(transition_step, dtype=jnp.int32),
        regime_id=jnp.asarray(regime_id, dtype=jnp.int32),
        pre_descriptors=mechanism.lifecycle_pre_descriptors,
        proposal_descriptors=mechanism.lifecycle_proposal_descriptors,
        applied_descriptors=mechanism.lifecycle_applied_descriptors,
        proposal_event=jnp.stack(
            (
                mechanism.lifecycle_proposal_replaced_slot,
                mechanism.lifecycle_proposal_promoted_candidate,
                mechanism.lifecycle_proposal_refreshed_candidate,
                mechanism.lifecycle_proposal_retired_slot,
                mechanism.lifecycle_proposal_retired_left,
                mechanism.lifecycle_proposal_retired_right,
            )
        ).astype(jnp.int32),
        applied_event=jnp.stack(
            (
                mechanism.lifecycle_applied_replaced_slot,
                mechanism.lifecycle_applied_promoted_candidate,
                mechanism.lifecycle_applied_refreshed_candidate,
                mechanism.lifecycle_applied_retired_slot,
                mechanism.lifecycle_applied_retired_left,
                mechanism.lifecycle_applied_retired_right,
            )
        ).astype(jnp.int32),
        critical_slot=jnp.stack((pre_slots, proposal_slots, applied_slots)),
        critical_candidate_streak=streak,
        critical_candidate_flags=flags,
        candidate_reset_mask=jnp.stack(
            (
                mechanism.lifecycle_candidate_reset_mask,
                mechanism.lifecycle_applied_candidate_reset_mask,
            )
        ),
        random_curation_flags=jnp.stack(
            (
                mechanism.random_curation_enabled,
                mechanism.random_curation_attempted,
                mechanism.random_curation_applied,
            )
        ),
        random_curation_selected=jnp.stack(
            (
                mechanism.random_curation_selected_active_worst_slot,
                mechanism.random_curation_selected_promotion_candidate,
                mechanism.random_curation_selected_refresh_candidate,
            )
        ).astype(jnp.int32),
        random_active_priorities=mechanism.random_curation_active_priorities,
        random_candidate_priorities=mechanism.random_curation_candidate_priorities,
        consumer_masks=jnp.stack(
            (
                mechanism.lifecycle_durable_read_mask,
                mechanism.consumer_read_acquire_pre,
                mechanism.consumer_read_acquire_post,
                mechanism.consumer_confirmed_write_pre,
                mechanism.consumer_confirmed_write_post,
                mechanism.consumer_read_mask_pre,
                mechanism.consumer_read_mask_post,
                mechanism.consumer_active_mask_pre,
                mechanism.consumer_active_mask_post,
            )
        ),
        router_source_slots=mechanism.router_source_slots,
        router_masks=jnp.stack(
            (
                mechanism.router_survivor_mask,
                mechanism.router_new_mask,
                mechanism.router_evicted_mask,
            )
        ),
        router_flags=jnp.stack(
            (
                mechanism.router_valid,
                mechanism.router_applied,
                mechanism.router_carry_survivors,
                mechanism.router_descriptors_changed,
            )
        ),
        router_counts=jnp.stack(
            (
                mechanism.router_route_count_before,
                mechanism.router_route_count_after,
                mechanism.router_generation_count_before,
                mechanism.router_generation_count_after,
            )
        ).astype(jnp.int32),
        transaction_exact=transaction_exact,
        identity_carry_exact=identity_exact,
        retired_identity_reset_exact=reset_exact,
    )


def _lifecycle_observation(
    *,
    transition_step: Array,
    occurrence_index: Array,
    regime_id: Array,
    pre_state: HiddenPartnerWorldOnlineState,
    post_state: HiddenPartnerWorldOnlineState,
    trace: HiddenPartnerWorldOnlineTrace,
    carry_survivors: bool,
) -> V6LifecycleObservation:
    mechanism = trace.mechanism
    d_index = jnp.asarray(CRITICAL_CANDIDATE_INDICES[1], dtype=jnp.int32)
    structural = (
        _router_transaction_exact(pre_state, post_state, trace)
        & _identity_carry_exact(mechanism, carry_survivors=carry_survivors)
        & _retired_identity_reset_exact(mechanism)
    )
    d_retired = (mechanism.lifecycle_applied_retired_left == D_PAIR[0]) & (
        mechanism.lifecycle_applied_retired_right == D_PAIR[1]
    )
    return V6LifecycleObservation(
        transition_step=jnp.asarray(transition_step, dtype=jnp.int32),
        occurrence_index=jnp.asarray(occurrence_index, dtype=jnp.int32),
        regime_id=jnp.asarray(regime_id, dtype=jnp.int32),
        pre_descriptors=mechanism.lifecycle_pre_descriptors,
        applied_descriptors=mechanism.lifecycle_applied_descriptors,
        promoted_candidate=mechanism.lifecycle_applied_promoted_candidate,
        retired_pair=jnp.stack(
            (
                mechanism.lifecycle_applied_retired_left,
                mechanism.lifecycle_applied_retired_right,
            )
        ).astype(jnp.int32),
        d_reacquisition_required_pre=(
            mechanism.lifecycle_candidate_reacquisition_required_pre[d_index]
        ),
        d_reacquisition_confirmed=(mechanism.lifecycle_candidate_reacquisition_confirmed[d_index]),
        d_reset_exact=(d_retired & _retired_identity_reset_exact(mechanism)),
        structural_valid=structural,
    )


def _oracle_schedule_exact(
    trace: HiddenPartnerWorldOnlineTrace,
    *,
    step: Array,
    cycle_length: Array,
    occurrence_starts: Array,
    occurrence_ends: Array,
    occurrence_regimes: Array,
) -> tuple[Array, Array, Array]:
    index = jnp.asarray(step, dtype=jnp.int32)
    cycle = index // cycle_length
    cycle_step = index % cycle_length
    first_cycle_starts = occurrence_starts[:N_SEGMENTS]
    first_cycle_ends = occurrence_ends[:N_SEGMENTS]
    segment_index = jnp.sum(cycle_step >= first_cycle_ends, dtype=jnp.int32)
    safe_segment = jnp.clip(segment_index, 0, N_SEGMENTS - 1)
    safe_occurrence = jnp.clip(cycle * N_SEGMENTS + safe_segment, 0, 2 * N_SEGMENTS - 1)
    segment_step = cycle_step - first_cycle_starts[safe_segment]
    segment_length = first_cycle_ends[safe_segment] - first_cycle_starts[safe_segment]
    next_index = index + jnp.asarray(1, dtype=jnp.int32)
    next_cycle = next_index // cycle_length
    next_cycle_step = next_index % cycle_length
    next_segment = jnp.sum(next_cycle_step >= first_cycle_ends, dtype=jnp.int32)
    safe_next_segment = jnp.clip(next_segment, 0, N_SEGMENTS - 1)
    schedule_exact = (
        (trace.oracle_step_count == index)
        & (trace.oracle_cycle_index == cycle)
        & (trace.oracle_cycle_step == cycle_step)
        & (trace.oracle_cycle_length == cycle_length)
        & (trace.oracle_segment_index == safe_segment)
        & (trace.oracle_segment_step == segment_step)
        & (trace.oracle_segment_length == segment_length)
        & (trace.oracle_regime_id == occurrence_regimes[safe_segment])
        & (trace.oracle_next_cycle_index == next_cycle)
        & (trace.oracle_next_segment_index == safe_next_segment)
        & (trace.oracle_next_regime_id == occurrence_regimes[safe_next_segment])
        & (
            trace.oracle_schedule_switched
            == ((next_cycle != cycle) | (safe_next_segment != safe_segment))
        )
    )
    return schedule_exact, safe_occurrence, occurrence_regimes[safe_segment]


def _oracle_algebra_exact(
    trace: HiddenPartnerWorldOnlineTrace,
    *,
    partner_flip_probability: float,
    outcome_flip_probability: float,
) -> Array:
    """Reconstruct the privileged trace, including full-information ties."""

    noise_coefficient = jnp.asarray(
        (1.0 - 2.0 * partner_flip_probability) * (1.0 - 2.0 * outcome_flip_probability),
        dtype=jnp.float32,
    )
    coefficient = noise_coefficient * trace.oracle_world_sign * trace.oracle_partner_intended_sign
    optimal_action = jnp.where(coefficient >= 0.0, 1, 0).astype(jnp.int32)
    expected_partner_sign = jnp.where(
        trace.oracle_partner_flipped,
        -trace.oracle_partner_intended_sign,
        trace.oracle_partner_intended_sign,
    )
    expected_world_sign = jnp.where(
        trace.oracle_world_flipped,
        -trace.oracle_world_sign,
        trace.oracle_world_sign,
    )
    outcome_noise_sign = jnp.where(trace.oracle_outcome_flipped, -1.0, 1.0)
    focal_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
    expected_counterfactual = 0.5 * (
        1.0
        + trace.oracle_world_sign
        * focal_signs
        * trace.oracle_partner_action_sign
        * outcome_noise_sign
    )
    expected_current_cues = jnp.where(
        trace.oracle_world_cue_flipped,
        -trace.oracle_world_sign,
        trace.oracle_world_sign,
    )
    expected_next_cues = jnp.where(
        trace.oracle_next_world_cue_flipped,
        -trace.oracle_next_world_sign,
        trace.oracle_next_world_sign,
    )
    return (
        (trace.oracle_partner_intended_action >= 0)
        & (trace.oracle_partner_intended_action < 2)
        & (
            trace.oracle_partner_intended_action
            == ((trace.oracle_partner_intended_sign + 1.0) / 2.0).astype(jnp.int32)
        )
        & jnp.array_equal(trace.oracle_partner_action_sign, expected_partner_sign)
        & jnp.array_equal(
            trace.oracle_focal_action_sign,
            2.0 * trace.focal_action.astype(jnp.float32) - 1.0,
        )
        & jnp.array_equal(
            trace.oracle_partner_action_sign,
            2.0 * trace.partner_action.astype(jnp.float32) - 1.0,
        )
        & jnp.array_equal(trace.oracle_next_world_sign, expected_world_sign)
        & jnp.array_equal(trace.observation_pre[6:8], expected_current_cues)
        & jnp.array_equal(trace.next_observation[6:8], expected_next_cues)
        & (trace.oracle_full_information_action == optimal_action)
        & jnp.array_equal(
            trace.oracle_full_information_action_margin,
            jnp.abs(coefficient),
        )
        & (trace.oracle_full_information_action_tied == (coefficient == 0.0))
        & jnp.array_equal(
            trace.oracle_realized_counterfactual_rewards,
            expected_counterfactual,
        )
    )


def _stream_domain_valid(trace: HiddenPartnerWorldOnlineTrace) -> Array:
    signs = jnp.concatenate(
        (
            jnp.reshape(trace.oracle_world_sign, (1,)),
            jnp.reshape(trace.oracle_next_world_sign, (1,)),
            jnp.reshape(trace.oracle_partner_intended_sign, (1,)),
            jnp.reshape(trace.oracle_focal_action_sign, (1,)),
            jnp.reshape(trace.oracle_partner_action_sign, (1,)),
        )
    )
    return (
        jnp.all((signs == -1.0) | (signs == 1.0))
        & (trace.focal_action >= 0)
        & (trace.focal_action < 2)
        & (trace.partner_action >= 0)
        & (trace.partner_action < 2)
        & jnp.all(jnp.isfinite(trace.observation_pre))
        & jnp.all(jnp.isfinite(trace.next_observation))
    )


def _component_deltas(trace: HiddenPartnerWorldOnlineTrace) -> Array:
    mechanism = trace.mechanism
    return jnp.stack(
        (
            mechanism.state_builder_step_delta,
            mechanism.state_builder_learning_delta,
            mechanism.behavior_step_delta,
            mechanism.interaction_step_delta,
            mechanism.table_world_step_delta,
            mechanism.grounded_world_step_delta,
            mechanism.control_step_delta,
            mechanism.router_route_delta,
            mechanism.router_generation_delta,
            mechanism.integrated_step_delta,
        )
    ).astype(jnp.int32)


def _expected_component_deltas(
    trace: HiddenPartnerWorldOnlineTrace,
    *,
    state_learning_enabled: bool,
    grounded_present: bool,
    grounded_learning_enabled: bool,
) -> Array:
    accepted = trace.accepted.astype(jnp.int32)
    grounded = int(grounded_present and grounded_learning_enabled)
    expected = jnp.asarray(
        (
            1,
            int(state_learning_enabled),
            1,
            1,
            1,
            grounded,
            1,
            1,
            0,
            1,
        ),
        dtype=jnp.int32,
    )
    expected = expected.at[8].set(trace.mechanism.router_descriptors_changed.astype(jnp.int32))
    return expected * accepted


def _grounded_target_algebra(trace: HiddenPartnerWorldOnlineTrace) -> Array:
    mechanism = trace.mechanism
    expected_targets = jnp.concatenate(
        (
            trace.next_observation,
            jnp.reshape(trace.reward, (1,)),
            jnp.ones((1,), dtype=jnp.float32),
        )
    )
    return ~mechanism.grounded_enabled | (
        mechanism.grounded_prediction_valid
        & mechanism.grounded_target_valid
        & mechanism.grounded_gradient_valid
        & mechanism.grounded_prediction_matches_decision
        & jnp.array_equal(mechanism.grounded_targets, expected_targets)
        & (mechanism.grounded_executed_joint_index == 2 * trace.focal_action + trace.partner_action)
    )


def _accumulate_v6_intervention_audit(
    totals: V6AuditTotals,
    audit: V6InterventionStepAudit,
) -> V6AuditTotals:
    """Accumulate one active step without retaining its transient audit vectors."""

    checks = jnp.asarray(audit.checks)
    witnesses = jnp.asarray(audit.witnesses)
    if checks.shape != (len(V6_INTERVENTION_AUDIT_ORDER),) or checks.dtype != jnp.bool_:
        raise TypeError("v6 intervention checks must be bool[18]")
    if witnesses.shape != (len(V6_INTERVENTION_WITNESS_ORDER),) or witnesses.dtype != jnp.bool_:
        raise TypeError("v6 intervention witnesses must be bool[16]")
    return totals.replace(
        intervention_failure_counts=_saturating_int32_event_counts(
            totals.intervention_failure_counts,
            ~checks,
        ),
        intervention_witness_counts=_saturating_int32_event_counts(
            totals.intervention_witness_counts,
            witnesses,
        ),
    )


def _saturating_int32_event_counts(counts: Array, events: Array) -> Array:
    """Add boolean events without signed overflow; corrupt inputs stay invalid."""

    current = jnp.asarray(counts)
    observed = jnp.asarray(events)
    if current.dtype != jnp.int32 or observed.dtype != jnp.bool_:
        raise TypeError("saturating v6 event counts require int32 counts and bool events")
    if current.shape != observed.shape:
        raise ValueError("saturating v6 event counts require equal shapes")
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    valid = (current >= 0) & (current <= maximum)
    safe_increment_base = jnp.minimum(current, maximum - jnp.int32(1))
    incremented = jnp.where(
        observed,
        safe_increment_base + jnp.int32(1),
        current,
    )
    return jnp.where(valid, incremented, maximum)


def _update_v6_audits(
    totals: V6AuditTotals,
    *,
    trace: HiddenPartnerWorldOnlineTrace,
    row_head_algebra_valid: Array,
    oracle_schedule_valid: Array,
    stream_domain_valid: Array,
    descriptor_domain_valid: Array,
    lifecycle_cadence_valid: Array,
    router_transaction_valid: Array,
    identity_carry_valid: Array,
    retired_reset_valid: Array,
    rng_chain_valid: Array,
    component_expected: Array,
    expected_cadence: Array,
    ledger_count: Array,
    ledger_overflow: Array,
    intervention: V6InterventionStepAudit,
) -> V6AuditTotals:
    mechanism = trace.mechanism
    accepted = trace.accepted
    target_algebra = _grounded_target_algebra(trace)
    filter_recurrence = ~accepted | trace.filter_trace_valid
    checks = jnp.stack(
        (
            trace.entry_state_contract_valid,
            trace.config_token_valid,
            trace.counters_synchronized,
            trace.action_valid,
            trace.action_policy_valid,
            trace.selection_binding_valid,
            trace.policy_replay_valid,
            trace.next_action_policy_valid,
            trace.next_selection_binding_valid,
            trace.next_policy_replay_valid,
            trace.next_selection_diagnostics_valid,
            trace.learner_trace_valid,
            trace.filter_trace_valid,
            trace.oracle_trace_valid,
            trace.all_finite,
            mechanism.valid,
            oracle_schedule_valid,
            stream_domain_valid,
            target_algebra,
            row_head_algebra_valid,
            descriptor_domain_valid,
            lifecycle_cadence_valid,
            router_transaction_valid,
            identity_carry_valid,
            retired_reset_valid,
            rng_chain_valid,
            filter_recurrence,
        )
    )
    deltas = _component_deltas(trace)
    delta_valid = deltas == component_expected
    updated = totals.replace(
        contract_failure_counts=(totals.contract_failure_counts + (~checks).astype(jnp.int32)),
        component_delta_sums=totals.component_delta_sums + deltas,
        component_delta_failure_counts=(
            totals.component_delta_failure_counts + (~delta_valid).astype(jnp.int32)
        ),
        active_steps=totals.active_steps + jnp.asarray(1, dtype=jnp.int32),
        accepted_steps=totals.accepted_steps + accepted.astype(jnp.int32),
        learner_valid_steps=(
            totals.learner_valid_steps + trace.learner_trace_valid.astype(jnp.int32)
        ),
        filter_valid_steps=(totals.filter_valid_steps + trace.filter_trace_valid.astype(jnp.int32)),
        oracle_valid_steps=(totals.oracle_valid_steps + trace.oracle_trace_valid.astype(jnp.int32)),
        mechanism_valid_steps=(totals.mechanism_valid_steps + mechanism.valid.astype(jnp.int32)),
        all_finite_steps=totals.all_finite_steps + trace.all_finite.astype(jnp.int32),
        curation_attempt_count=(
            totals.curation_attempt_count
            + jnp.asarray(expected_cadence, dtype=jnp.bool_).astype(jnp.int32)
        ),
        ledger_count=jnp.asarray(ledger_count, dtype=jnp.int32),
        ledger_overflow=totals.ledger_overflow | ledger_overflow,
    )
    return _accumulate_v6_intervention_audit(updated, intervention)


def _require_typed_key(key: Array, *, name: str) -> None:
    if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
        getattr(key, "dtype", None),
        jax.dtypes.prng_key,
    ):
        raise TypeError(f"{name} must be a scalar typed PRNG key")
    key_data = jr.key_data(key)
    if (
        str(jr.key_impl(key)) != "threefry2x32"
        or key_data.shape != (2,)
        or key_data.dtype != jnp.dtype(jnp.uint32)
    ):
        raise TypeError(f"{name} must use the exact threefry2x32 uint32[2] key contract")


def _world_key_data(state: HiddenPartnerWorldOnlineState) -> Array:
    world = state.world
    return jnp.stack(
        tuple(
            jr.key_data(key)
            for key in (
                world.signal_key,
                world.partner_key,
                world.world_key,
                world.cue_key,
                world.outcome_key,
            )
        )
    ).astype(jnp.uint32)


def _policy_key_data(state: HiddenPartnerWorldOnlineState) -> Array:
    selection = state.agent.current_selection
    return jnp.stack(
        (
            jr.key_data(selection.rng_key_before),
            jr.key_data(selection.rng_key_after),
        )
    ).astype(jnp.uint32)


def _initial_stream_bits(state: HiddenPartnerWorldOnlineState) -> Array:
    world = state.world
    bits = jnp.concatenate(
        (
            world.current_signals > 0.0,
            jnp.reshape(world.world_sign > 0.0, (1,)),
            world.current_cues > 0.0,
            jnp.reshape(world.previous_outcome > 0.0, (1,)),
            jnp.reshape(world.has_partner_history, (1,)),
        )
    ).astype(jnp.uint8)
    return jnp.sum(
        jnp.left_shift(bits, jnp.arange(8, dtype=jnp.uint8)),
        dtype=jnp.uint8,
    )


class HiddenPartnerLifecycleWorldV6Runner:
    """Execute one exact v6 control as an authority-free in-memory run."""

    def __init__(
        self,
        control: HiddenPartnerLifecycleWorldV6Control,
        *,
        seed_namespace: str | None = None,
    ) -> None:
        require_v6_development_seed_namespace(seed_namespace)
        readiness = require_v6_control_suite_ready()
        validated = validate_v6_control(control)
        if not validated.execution_ready or validated.agent_config is None:
            raise ValueError("v6 runner requires an execution-ready live control")
        self._control = validated
        self._world = HiddenPartnerWorldFeedbackWorld(validated.world_config)
        self._agent = IntegratedHiddenPartnerAgent(validated.agent_config)
        self._bridge = HiddenPartnerWorldOnlineBridge(
            world=self._world,
            agent=self._agent,
            focal_action_policy=validated.focal_action_policy,
            initial_external_action=validated.initial_external_action,
        )
        family = "primary" if validated.primary else "diagnostic"
        matching_bindings = tuple(
            binding
            for binding in readiness.bindings
            if binding.family == family and binding.name == validated.name
        )
        if len(matching_bindings) != 1:
            raise RuntimeError("v6 readiness must contain one exact selected binding")
        binding = matching_bindings[0]
        if binding.bridge_config_sha256 != self._bridge.config_token_hex:
            raise RuntimeError("v6 selected binding differs from the constructed bridge")
        self._readiness = readiness
        self._binding = binding
        self._state_learning_enabled = validated.agent_config.state_learning_enabled
        self._grounded_present = validated.agent_config.grounded_world_model is not None
        self._grounded_learning_enabled = validated.agent_config.grounded_world_learning_enabled
        self._carry_survivors = validated.agent_config.carry_survivors
        self._partner_flip_probability = validated.world_config.partner_flip_probability
        self._outcome_flip_probability = validated.world_config.outcome_flip_probability

    @property
    def control(self) -> HiddenPartnerLifecycleWorldV6Control:
        return self._control

    @property
    def bridge(self) -> HiddenPartnerWorldOnlineBridge:
        return self._bridge

    def initialize(
        self,
        world_key: Array,
        agent_key: Array,
    ) -> HiddenPartnerWorldOnlineState:
        """Return the deterministic v6 initial state for two typed keys.

        The general bridge deliberately preserves host lifecycle timestamps.
        They are useful operational metadata, but they are neither learned
        state nor a deterministic function of the v6 inputs.  The v6 runner
        therefore canonicalizes exactly the interaction and control birth
        timestamps to positive float32 zero at its boundary.  Component
        uptimes must already be positive float32 zero and are preserved
        bit-exactly; any drift fails closed.
        """

        _require_typed_key(world_key, name="world_key")
        _require_typed_key(agent_key, name="agent_key")
        initial = self._bridge.initialize(world_key, agent_key)
        interaction_birth = jnp.asarray(initial.agent.interaction.birth_timestamp)
        control_birth = jnp.asarray(initial.agent.control.birth_timestamp)
        interaction_uptime = jnp.asarray(initial.agent.interaction.uptime_s)
        control_uptime = jnp.asarray(initial.agent.control.uptime_s)
        for name, birth in (
            ("agent.interaction.birth_timestamp", interaction_birth),
            ("agent.control.birth_timestamp", control_birth),
        ):
            if birth.shape != () or birth.dtype != jnp.dtype(jnp.float32):
                raise RuntimeError(f"v6 {name} must be a scalar float32")
            valid_birth = jnp.isfinite(birth) & (birth > jnp.asarray(0.0, dtype=jnp.float32))
            if not bool(jax.device_get(valid_birth)):
                raise RuntimeError(f"v6 {name} must be finite and positive before canonicalization")
        for name, uptime in (
            ("agent.interaction.uptime_s", interaction_uptime),
            ("agent.control.uptime_s", control_uptime),
        ):
            if uptime.shape != () or uptime.dtype != jnp.dtype(jnp.float32):
                raise RuntimeError(f"v6 {name} must be a scalar float32")
            uptime_bits = jax.lax.bitcast_convert_type(uptime, jnp.uint32)
            if int(jax.device_get(uptime_bits)) != 0:
                raise RuntimeError(f"v6 {name} must initialize to positive zero")

        zero = jnp.asarray(0.0, dtype=jnp.float32)
        canonical_agent = initial.agent.replace(
            interaction=initial.agent.interaction.replace(birth_timestamp=zero),
            control=initial.agent.control.replace(birth_timestamp=zero),
        )
        canonical = initial.replace(agent=canonical_agent)
        for before, after in (
            (interaction_uptime, canonical.agent.interaction.uptime_s),
            (control_uptime, canonical.agent.control.uptime_s),
        ):
            before_bits = jax.lax.bitcast_convert_type(before, jnp.uint32)
            after_bits = jax.lax.bitcast_convert_type(after, jnp.uint32)
            if not bool(jax.device_get(jnp.array_equal(before_bits, after_bits))):
                raise RuntimeError("v6 timestamp canonicalization changed component uptime")
        return canonical

    def _config_for_source_closure(
        self,
        source_closure: tuple[V6SourceClosureHash, ...],
        runtime: V6RuntimeRecord,
    ) -> dict[str, object]:
        """Reconstruct metadata after the caller has validated both inputs."""

        validated_runtime = validate_v6_runtime_record(runtime, require_live_match=False)
        return {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNNER_SCHEMA,
            "status": RUNNER_STATUS,
            "phase": "PHASE_1_RUNNER_PHASE_2_VALIDATOR_SEPARATE",
            "development_only": DEVELOPMENT_ONLY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "control_schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
            "control_matrix_sha256": self._readiness.control_matrix_sha256,
            "readiness": self._readiness.to_config(),
            "selected_control_binding": self._binding.to_config(),
            "control": self._control.to_config(),
            "control_config_sha256": self._binding.control_config_sha256,
            "bridge_schema": self._binding.bridge_schema,
            "bridge_config": self._bridge.to_config(),
            "bridge_config_sha256": self._binding.bridge_config_sha256,
            "initial_state_canonicalization": {
                "scope": "v6_runner_only",
                "canonicalized_leaves": [
                    {
                        "path": "agent.interaction.birth_timestamp",
                        "required_precondition": "scalar finite positive float32",
                        "canonical_value": "+0.0",
                        "canonical_uint32_bits": 0,
                    },
                    {
                        "path": "agent.control.birth_timestamp",
                        "required_precondition": "scalar finite positive float32",
                        "canonical_value": "+0.0",
                        "canonical_uint32_bits": 0,
                    },
                ],
                "required_preserved_leaves": [
                    {
                        "path": "agent.interaction.uptime_s",
                        "dtype": "float32",
                        "required_uint32_bits": 0,
                    },
                    {
                        "path": "agent.control.uptime_s",
                        "dtype": "float32",
                        "required_uint32_bits": 0,
                    },
                ],
            },
            "source_closure": [dataclasses.asdict(record) for record in source_closure],
            "runtime": dataclasses.asdict(validated_runtime),
            "fixed_scan": {
                "iterations": MAX_SCAN_STEPS,
                "active_expression": "scan_index < plan.run_steps",
                "active_branch": "bridge_step_and_compact_accumulation",
                "padding_branch": "bit_exact_carry_and_uint8_zero",
                "window_bins": WINDOW_BIN_COUNT,
                "row_head_cells": JOINT_ACTION_ROWS * TARGET_HEADS,
                "cadence_interval": CURATION_INTERVAL,
                "cadence_ledger_capacity": MAX_CADENCE_LEDGER_ENTRIES,
                "stream_code_dtype": "uint8",
                "full_trace_stacked": False,
                "intervention_audit_order": list(V6_INTERVENTION_AUDIT_ORDER),
                "intervention_witness_order": list(V6_INTERVENTION_WITNESS_ORDER),
                "float32_bounded_replay": {
                    "rtol": V6_FLOAT32_REPLAY_RTOL,
                    "atol": V6_FLOAT32_REPLAY_ATOL,
                    "disabled_persistence_bit_exact": True,
                },
                "control_required_witnesses": [
                    {
                        "control": name,
                        "required": list(required),
                    }
                    for name, required in V6_CONTROL_REQUIRED_WITNESSES
                ],
            },
            "seed_namespace": None,
            "thresholds": None,
            "outcomes": None,
            "writes_files": False,
        }

    def to_config(self) -> dict[str, object]:
        """Live-capture canonical development metadata and exact provenance."""

        runtime = capture_v6_runtime_record()
        validate_v6_runtime_record(runtime, require_live_match=True)
        source_closure = compute_v6_source_closure_hashes()
        return self._config_for_source_closure(source_closure, runtime)

    def _advance_active(
        self,
        carry: V6ScanCarry,
        step: Array,
        *,
        run_steps: Array,
        cycle_length: Array,
        occurrence_starts: Array,
        occurrence_ends: Array,
        occurrence_regimes: Array,
    ) -> tuple[V6ScanCarry, Array]:
        result = self._bridge.step(carry.bridge_state)
        intervention = audit_v6_intervention_step(
            self._control,
            self._agent,
            carry.bridge_state,
            result,
        )
        trace = result.trace
        membership = v6_window_membership(
            step,
            occurrence_starts,
            occurrence_ends,
            run_steps,
        )
        windows = _update_v6_window_totals(
            carry.windows,
            membership=membership,
            accepted=trace.accepted,
            pre_state=carry.bridge_state,
            trace=trace,
        )
        mechanism = trace.mechanism
        row_heads, row_head_valid = update_v6_row_head_totals(
            carry.row_heads,
            accepted=trace.accepted,
            grounded_valid=(
                mechanism.grounded_enabled
                & mechanism.grounded_prediction_valid
                & mechanism.grounded_target_valid
                & mechanism.grounded_gradient_valid
            ),
            executed_row=mechanism.grounded_executed_joint_index,
            feature_contribution=mechanism.grounded_feature_contribution,
            row_bias=mechanism.grounded_row_bias,
            raw_predictions=mechanism.grounded_raw_predictions,
            targets=mechanism.grounded_targets,
            errors=mechanism.grounded_errors,
            fit_loss_by_head=mechanism.grounded_fit_loss_by_head,
            representation_loss_by_head=(mechanism.grounded_representation_loss_by_head),
            representation_gradient=mechanism.grounded_representation_gradient,
            representation_gradient_by_head=(mechanism.grounded_representation_gradient_by_head),
            representation_gradient_norm_by_head=(
                mechanism.grounded_representation_gradient_norm_by_head
            ),
            proposed_weight_change_mask=(mechanism.grounded_proposed_weight_row_bit_change_mask),
            proposed_bias_change_mask=(mechanism.grounded_proposed_bias_row_bit_change_mask),
            executed_weight_delta_norm_by_head=(
                mechanism.grounded_executed_weight_row_delta_norm_by_head
            ),
            executed_bias_delta_by_head=(mechanism.grounded_executed_bias_row_delta_by_head),
            row_update_isolated=mechanism.grounded_row_update_isolated,
            target_weights=jnp.asarray(
                V6_REPRESENTATION_LOSS_WEIGHTS,
                dtype=jnp.float32,
            ),
        )
        filter_totals = _update_v6_filter_totals(
            carry.filter_totals,
            accepted=trace.accepted,
            trace=trace,
        )
        action_totals, rng_chain_valid = _update_v6_action_totals(
            carry.action_totals,
            accepted=trace.accepted,
            trace=trace,
            initial_external_action=self._control.initial_external_action,
            balanced_external=(self._control.focal_action_policy == "balanced_external"),
        )
        oracle_schedule_valid, occurrence_index, regime_id = _oracle_schedule_exact(
            trace,
            step=step,
            cycle_length=cycle_length,
            occurrence_starts=occurrence_starts,
            occurrence_ends=occurrence_ends,
            occurrence_regimes=occurrence_regimes,
        )
        oracle_schedule_valid = oracle_schedule_valid & _oracle_algebra_exact(
            trace,
            partner_flip_probability=self._partner_flip_probability,
            outcome_flip_probability=self._outcome_flip_probability,
        )
        expected_cadence = jnp.equal(
            jnp.mod(step + jnp.asarray(1, dtype=jnp.int32), CURATION_INTERVAL),
            0,
        )
        descriptor_domain_valid = (
            _descriptor_bank_valid(mechanism.lifecycle_pre_descriptors)
            & _descriptor_bank_valid(mechanism.lifecycle_proposal_descriptors)
            & _descriptor_bank_valid(mechanism.lifecycle_applied_descriptors)
        )
        event = (
            mechanism.lifecycle_proposed
            | mechanism.lifecycle_applied
            | mechanism.router_descriptors_changed
            | mechanism.random_curation_attempted
        )
        lifecycle_cadence_valid = ~event | expected_cadence
        router_transaction_valid = _router_transaction_exact(
            carry.bridge_state,
            result.state,
            trace,
        )
        identity_carry_valid = _identity_carry_exact(
            mechanism,
            carry_survivors=self._carry_survivors,
        )
        retired_reset_valid = _retired_identity_reset_exact(mechanism)
        cadence = _cadence_observation(
            transition_step=step + jnp.asarray(1, dtype=jnp.int32),
            regime_id=regime_id,
            pre_state=carry.bridge_state,
            post_state=result.state,
            trace=trace,
            carry_survivors=self._carry_survivors,
        )

        def insert_ledger(_: None) -> tuple[V6CadenceLedger, Array, Array]:
            return record_v6_cadence_observation(
                carry.ledger,
                carry.audits.ledger_count,
                cadence,
            )

        def preserve_ledger(_: None) -> tuple[V6CadenceLedger, Array, Array]:
            return (
                carry.ledger,
                carry.audits.ledger_count,
                jnp.asarray(False, dtype=jnp.bool_),
            )

        ledger, ledger_count, ledger_overflow = cast(
            tuple[V6CadenceLedger, Array, Array],
            jax.lax.cond(
                expected_cadence,
                insert_ledger,
                preserve_ledger,
                operand=None,
            ),
        )
        lifecycle = update_v6_lifecycle_chain(
            carry.lifecycle,
            _lifecycle_observation(
                # Lifecycle positions use the same one-based completed-transition
                # unit as the cadence ledger.  Only schedule-array lookup uses the
                # zero-based scan index above.
                transition_step=step + jnp.asarray(1, dtype=jnp.int32),
                occurrence_index=occurrence_index,
                regime_id=regime_id,
                pre_state=carry.bridge_state,
                post_state=result.state,
                trace=trace,
                carry_survivors=self._carry_survivors,
            ),
        )
        component_expected = _expected_component_deltas(
            trace,
            state_learning_enabled=self._state_learning_enabled,
            grounded_present=self._grounded_present,
            grounded_learning_enabled=self._grounded_learning_enabled,
        )
        audits = _update_v6_audits(
            carry.audits,
            trace=trace,
            row_head_algebra_valid=row_head_valid,
            oracle_schedule_valid=oracle_schedule_valid,
            stream_domain_valid=_stream_domain_valid(trace),
            descriptor_domain_valid=descriptor_domain_valid,
            lifecycle_cadence_valid=lifecycle_cadence_valid,
            router_transaction_valid=router_transaction_valid,
            identity_carry_valid=identity_carry_valid,
            retired_reset_valid=retired_reset_valid,
            rng_chain_valid=rng_chain_valid,
            component_expected=component_expected,
            expected_cadence=expected_cadence,
            ledger_count=ledger_count,
            ledger_overflow=ledger_overflow,
            intervention=intervention,
        )
        stream_code = jnp.where(
            trace.accepted,
            pack_v6_stream_code(
                next_signals=result.state.world.current_signals,
                partner_flipped=trace.oracle_partner_flipped,
                world_flipped=trace.oracle_world_flipped,
                next_cue_flipped=trace.oracle_next_world_cue_flipped,
                outcome_flipped=trace.oracle_outcome_flipped,
            ),
            jnp.asarray(0, dtype=jnp.uint8),
        )
        return (
            V6ScanCarry(
                bridge_state=result.state,
                windows=windows,
                row_heads=row_heads,
                filter_totals=filter_totals,
                action_totals=action_totals,
                audits=audits,
                ledger=ledger,
                lifecycle=lifecycle,
            ),
            stream_code,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _fixed_scan(
        self,
        carry: V6ScanCarry,
        run_steps: Array,
        cycle_length: Array,
        occurrence_starts: Array,
        occurrence_ends: Array,
        occurrence_regimes: Array,
    ) -> tuple[V6ScanCarry, Array]:
        """Run the exact static scan with an explicit active-v-padding branch."""

        def body(current: V6ScanCarry, index: Array) -> tuple[V6ScanCarry, Array]:
            return cast(
                tuple[V6ScanCarry, Array],
                jax.lax.cond(
                    index < run_steps,
                    lambda value: self._advance_active(
                        value,
                        index,
                        run_steps=run_steps,
                        cycle_length=cycle_length,
                        occurrence_starts=occurrence_starts,
                        occurrence_ends=occurrence_ends,
                        occurrence_regimes=occurrence_regimes,
                    ),
                    lambda value: (value, jnp.asarray(0, dtype=jnp.uint8)),
                    current,
                ),
            )

        return cast(
            tuple[V6ScanCarry, Array],
            jax.lax.scan(
                body,
                carry,
                jnp.arange(MAX_SCAN_STEPS, dtype=jnp.int32),
            ),
        )

    def run(self, world_key: Array, agent_key: Array) -> V6DevelopmentRun:
        """Run in memory from two caller-supplied scalar typed PRNG keys."""

        _require_typed_key(world_key, name="world_key")
        _require_typed_key(agent_key, name="agent_key")
        runtime = capture_v6_runtime_record()
        validate_v6_runtime_record(runtime, require_live_match=True)
        source_closure = compute_v6_source_closure_hashes()
        runner_config = self._config_for_source_closure(source_closure, runtime)
        runner_config_sha256 = hashlib.sha256(_canonical_json_bytes(runner_config)).hexdigest()
        initial = self.initialize(world_key, agent_key)
        plan = validate_hidden_partner_lifecycle_world_v6_scan_plan(
            build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(initial.world)
        )
        occurrence_starts = jnp.asarray(
            tuple(item.start for item in plan.segment_occurrences),
            dtype=jnp.int32,
        )
        occurrence_ends = jnp.asarray(
            tuple(item.end_exclusive for item in plan.segment_occurrences),
            dtype=jnp.int32,
        )
        occurrence_regimes = jnp.asarray(
            tuple(item.regime_id for item in plan.segment_occurrences),
            dtype=jnp.int32,
        )
        initial_budget = self._bridge.resource_budget(initial)
        final_carry, stream_code = self._fixed_scan(
            empty_v6_scan_carry(initial),
            jnp.asarray(plan.run_steps, dtype=jnp.int32),
            jnp.asarray(plan.cycle_length, dtype=jnp.int32),
            occurrence_starts,
            occurrence_ends,
            occurrence_regimes,
        )
        final_carry, stream_code = jax.block_until_ready((final_carry, stream_code))
        validate_v6_runtime_record(runtime, require_live_match=True)
        post_scan_source_closure = compute_v6_source_closure_hashes()
        if post_scan_source_closure != source_closure:
            raise RuntimeError("v6 source closure changed during the fixed scan")
        final = final_carry.bridge_state
        final_budget = self._bridge.resource_budget(final)
        accepted_steps = final_carry.audits.accepted_steps
        world_draws = jnp.full((5,), accepted_steps, dtype=jnp.int32)
        rng = V6RngRecord(
            supplied_key_data=jnp.stack((jr.key_data(world_key), jr.key_data(agent_key))).astype(
                jnp.uint32
            ),
            initial_world_key_data=_world_key_data(initial),
            final_world_key_data=_world_key_data(final),
            initial_policy_key_data=_policy_key_data(initial),
            final_policy_key_data=_policy_key_data(final),
            initial_interaction_key_data=jr.key_data(initial.agent.interaction.key).astype(
                jnp.uint32
            ),
            final_interaction_key_data=jr.key_data(final.agent.interaction.key).astype(jnp.uint32),
            initial_stream_bits=_initial_stream_bits(initial),
            world_draw_counts=world_draws,
            interaction_key_advance_count=accepted_steps,
            policy_decision_count=final_carry.action_totals.decision_count,
        )
        resources = V6ResourceRecord(
            initial=initial_budget,
            final=final_budget,
            peak_total_state_nbytes=max(
                initial_budget.total_state_nbytes,
                final_budget.total_state_nbytes,
            ),
            static_total_state_nbytes=(
                initial_budget.total_state_nbytes == final_budget.total_state_nbytes
            ),
            zero_replay=(initial_budget.replay_capacity == 0 and final_budget.replay_capacity == 0),
            initial_tree_signature=_tree_shape_signature(initial),
            final_tree_signature=_tree_shape_signature(final),
            tree_structure_equal=(
                jax.tree_util.tree_structure(initial) == jax.tree_util.tree_structure(final)
            ),
            tree_signature_equal=(_tree_shape_signature(initial) == _tree_shape_signature(final)),
        )
        return V6DevelopmentRun(
            control_name=self._control.name,
            primary=self._control.primary,
            plan=plan,
            control_config_sha256=self._binding.control_config_sha256,
            control_matrix_sha256=self._readiness.control_matrix_sha256,
            bridge_config_sha256=self._bridge.config_token_hex,
            runner_config_sha256=runner_config_sha256,
            source_closure_hashes=source_closure,
            runtime=runtime,
            initial_state=initial,
            final_state=final,
            windows=final_carry.windows,
            row_heads=final_carry.row_heads,
            filter_totals=final_carry.filter_totals,
            action_totals=final_carry.action_totals,
            audits=final_carry.audits,
            ledger=final_carry.ledger,
            lifecycle=final_carry.lifecycle,
            rng=rng,
            resources=resources,
            stream_code=stream_code,
        )


__all__ = [
    "CRITICAL_CANDIDATE_INDICES",
    "CRITICAL_PAIRS",
    "CURATION_INTERVAL",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_RUNNER_SCHEMA",
    "HiddenPartnerLifecycleWorldV6Runner",
    "JOINT_ACTION_ROWS",
    "MAX_CADENCE_LEDGER_ENTRIES",
    "MAX_SCAN_STEPS",
    "RUNNER_STATUS",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TARGET_HEADS",
    "V6ActionTotals",
    "V6AuditTotals",
    "V6CadenceLedger",
    "V6CadenceObservation",
    "V6DevelopmentRun",
    "V6FilterTotals",
    "V6LifecycleChainState",
    "V6LifecycleObservation",
    "V6ResourceRecord",
    "V6RngRecord",
    "V6RowHeadTotals",
    "V6ScanCarry",
    "V6SourceClosureHash",
    "V6WindowTotals",
    "V6_APPLIED_EVENT_ORDER",
    "V6_CANDIDATE_FLAG_ORDER",
    "V6_CANDIDATE_STREAK_ENDPOINT_ORDER",
    "V6_COMPONENT_DELTA_ORDER",
    "V6_CONSUMER_MASK_ORDER",
    "V6_CONTRACT_AUDIT_ORDER",
    "V6_CRITICAL_STAGE_ORDER",
    "V6_INTERVENTION_AUDIT_ORDER",
    "V6_INTERVENTION_WITNESS_ORDER",
    "V6_INITIAL_STREAM_BIT_ORDER",
    "V6_POLICY_KEY_ENDPOINT_ORDER",
    "V6_PROPOSAL_EVENT_ORDER",
    "V6_RANDOM_CURATION_FLAG_ORDER",
    "V6_RANDOM_CURATION_SELECTED_ORDER",
    "V6_ROUTER_COUNT_ORDER",
    "V6_ROUTER_FLAG_ORDER",
    "V6_ROUTER_MASK_ORDER",
    "V6_SOURCE_CLOSURE_PATHS",
    "V6_TARGET_HEAD_ORDER",
    "V6_TRANSITION_STREAM_BIT_ORDER",
    "V6_WORLD_RNG_KEY_ORDER",
    "WINDOW_BIN_COUNT",
    "compute_v6_source_closure_hashes",
    "empty_v6_action_totals",
    "empty_v6_audit_totals",
    "empty_v6_cadence_ledger",
    "empty_v6_filter_totals",
    "empty_v6_lifecycle_chain_state",
    "empty_v6_row_head_totals",
    "empty_v6_scan_carry",
    "empty_v6_window_totals",
    "expected_v6_carry_survivors",
    "expected_v6_focal_action",
    "pack_v6_stream_code",
    "record_v6_cadence_observation",
    "reconstruct_v6_stream_code",
    "require_v6_development_seed_namespace",
    "update_v6_lifecycle_chain",
    "update_v6_row_head_totals",
    "v6_window_membership",
]
