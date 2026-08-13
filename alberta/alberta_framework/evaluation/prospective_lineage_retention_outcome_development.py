"""Descriptive L0 outcome panel for prospective lineage retention.

This evaluator supplies the smallest causal life that the mechanism in
``core.prospective_lineage_retention`` can honestly support.  A full three-row
bank contains A, B, and D.  An evaluator-fixed X admission forces an eviction
before either of two future twins reveals its first observation.  The twins
share every input and every mechanism byte through that eviction commit.

The calibrated and reversed return priors are visible at semantic birth.  The
true future law is evaluator-only: it is used only to aggregate already-built
raw B and D cells and is never passed to preparation, victim selection, or
settlement.  The matched-unrouted arm always uses the declared ordinary-LRU B
victim; that comparator rule is fixed before outcomes and is not optimized.

After eviction, a fixed H=2 observation pair is compared with the archived,
fresh, and every live diagnostic predictor.  Exact strict superiority may
confirm a cross-birth restoration.  That confirmation and its causal
reacquisition-cost observation settle only after both observations, and their
effect is exposed only in a later preparation.  The predictors are evaluator
fixtures, not stored model parameters; the core archive remains metadata-only.

The report is development-only and nonpromoting.  It has no threshold,
winner, selected default, expected-direction assertion, artifact writer, seed,
or evidence authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.prospective_lineage_retention import (
    ProspectiveLineageRetention,
    ProspectiveLineageRetentionConfig,
    ProspectiveLineageRetentionEvent,
    ProspectiveLineageRetentionPreparation,
    ProspectiveLineageRetentionProposal,
    ProspectiveLineageRetentionState,
)

PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_CONFIG_SCHEMA: Final = (
    "alberta.prospective-lineage-retention-outcome-development.config.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_CELL_SCHEMA: Final = (
    "alberta.prospective-lineage-retention-outcome-development.cell.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_PANEL_SCHEMA: Final = (
    "alberta.prospective-lineage-retention-outcome-development.panel.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_REPORT_SCHEMA: Final = (
    "alberta.prospective-lineage-retention-outcome-development.report.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_VALIDATION_SCHEMA: Final = (
    "alberta.prospective-lineage-retention-outcome-development.validation.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_EVIDENCE_LEVEL: Final = "L0"
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_STATUS: Final = "descriptive-development-outcome-not-evidence"
PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_SCIENTIFIC_PROMOTION_ALLOWED: Final = False

_MAX_CONTEXTS: Final = 3
_HORIZON: Final = 2
_ACTIVE_SLOT: Final = 0
_LRU_COMPARATOR_SLOT: Final = 1
_NEWBORN_ID: Final = 4
_INITIAL_STEP: Final = 10
_INITIAL_RECENCY: Final = (10, 5, 7)
_LINEAGE_LABELS: Final = ((1, "A"), (2, "B"), (3, "D"), (4, "X"))
_FUTURES: Final = ("B", "D")
_TRUE_FUTURE_LAW: Final = (("B", 0.75), ("D", 0.25))
_PRIOR_PANELS: Final = (
    ("calibrated", (1.0, 0.75, 0.25)),
    ("reversed_misspecified", (1.0, 0.25, 0.75)),
)
_PREDICTIONS: Final = (
    ("A", (1.0, 1.0)),
    ("B", (-1.0, 1.0)),
    ("D", (1.0, -1.0)),
    ("X", (0.0, 0.0)),
)
_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(value: int) -> jax.Array:
    return jnp.asarray((0, value), dtype=jnp.uint32)


def _births(*values: int) -> jax.Array:
    return jnp.stack(tuple(_words(value) for value in values)).astype(jnp.uint32)


def _next_words(words: jax.Array) -> jax.Array:
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    return jnp.stack(
        (
            words[0] + (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32),
            low,
        )
    ).astype(jnp.uint32)


def _word_value(words: jax.Array) -> int:
    array = np.asarray(words, dtype=np.uint32)
    return (int(array[0]) << 32) | int(array[1])


def _lineage_label(words: jax.Array) -> str:
    value = _word_value(words)
    labels = dict(_LINEAGE_LABELS)
    if value not in labels:
        raise ValueError(f"unknown evaluator lineage {value}")
    return labels[value]


def _prediction(label: str) -> tuple[float, float]:
    predictions = dict(_PREDICTIONS)
    if label not in predictions:
        raise ValueError(f"unknown evaluator prediction label {label!r}")
    return predictions[label]


def _mean_squared_loss(
    prediction: tuple[float, float],
    observations: tuple[float, float],
) -> float:
    return float(
        sum(
            (predicted - observed) ** 2
            for predicted, observed in zip(prediction, observations, strict=True)
        )
        / _HORIZON
    )


def _tree_nbytes(tree: object) -> int:
    return sum(
        int(np.asarray(leaf).size) * int(np.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
    )


def _tree_sha256(tree: object) -> str:
    digest = hashlib.sha256()
    leaves = jax.tree.leaves(tree)
    digest.update(_canonical_json_bytes({"leaf_count": len(leaves)}))
    for index, leaf in enumerate(leaves):
        array = np.asarray(leaf)
        digest.update(
            _canonical_json_bytes(
                {
                    "index": index,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
            )
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _exact_type(value: object, expected: type[object], name: str) -> None:
    if type(value) is not expected:
        raise ValueError(f"{name} must have exact type {expected.__name__}")


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionOutcomeDevelopmentConfig:
    """Fixed geometry for the bounded descriptive panel."""

    max_contexts: int = _MAX_CONTEXTS
    confirmation_horizon: int = _HORIZON
    initial_reacquisition_cost: float = 0.5
    cost_ema_decay: float = 0.5
    max_abs_cost: float = 10.0
    minimax_guardrail_slack: float = 0.0

    def __post_init__(self) -> None:
        exact = (
            (self.max_contexts, int, "max_contexts"),
            (self.confirmation_horizon, int, "confirmation_horizon"),
            (self.initial_reacquisition_cost, float, "initial_reacquisition_cost"),
            (self.cost_ema_decay, float, "cost_ema_decay"),
            (self.max_abs_cost, float, "max_abs_cost"),
            (self.minimax_guardrail_slack, float, "minimax_guardrail_slack"),
        )
        for value, expected, name in exact:
            _exact_type(value, cast(type[object], expected), name)
        if self.max_contexts != _MAX_CONTEXTS:
            raise ValueError("this bounded panel requires max_contexts=3")
        if self.confirmation_horizon != _HORIZON:
            raise ValueError("this bounded panel requires confirmation_horizon=2")
        for name in (
            "initial_reacquisition_cost",
            "cost_ema_decay",
            "max_abs_cost",
            "minimax_guardrail_slack",
        ):
            value = cast(float, getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.initial_reacquisition_cost <= self.max_abs_cost:
            raise ValueError("initial_reacquisition_cost is outside the declared bound")
        if not 0.0 <= self.cost_ema_decay <= 1.0:
            raise ValueError("cost_ema_decay must be in [0, 1]")
        if self.max_abs_cost <= 0.0:
            raise ValueError("max_abs_cost must be positive")
        if not 0.0 <= self.minimax_guardrail_slack <= self.max_abs_cost:
            raise ValueError("minimax_guardrail_slack is outside the declared bound")
        for name in (
            "initial_reacquisition_cost",
            "cost_ema_decay",
            "max_abs_cost",
            "minimax_guardrail_slack",
        ):
            value = cast(float, getattr(self, name))
            if abs(value) > float(np.finfo(np.float32).max):
                raise ValueError(f"{name} exceeds the finite float32 range")
            if float(np.float32(value)) != value:
                raise ValueError(f"{name} must round-trip exactly through float32")
        # Reuse the donor's complete bounds contract at configuration time so
        # no accepted evaluator config can fail only when the first cell builds.
        self.mechanism_config()

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_CONFIG_SCHEMA,
            "max_contexts": self.max_contexts,
            "confirmation_horizon": self.confirmation_horizon,
            "initial_reacquisition_cost": self.initial_reacquisition_cost,
            "cost_ema_decay": self.cost_ema_decay,
            "max_abs_cost": self.max_abs_cost,
            "minimax_guardrail_slack": self.minimax_guardrail_slack,
            "evidence_level": PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_EVIDENCE_LEVEL,
            "status": PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "thresholds_used": False,
            "selection_performed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ProspectiveLineageRetentionOutcomeDevelopmentConfig:
        required = {
            "schema",
            "max_contexts",
            "confirmation_horizon",
            "initial_reacquisition_cost",
            "cost_ema_decay",
            "max_abs_cost",
            "minimax_guardrail_slack",
            "evidence_level",
            "status",
            "scientific_promotion_allowed",
            "thresholds_used",
            "selection_performed",
        }
        if type(payload) is not dict or set(payload) != required:
            raise ValueError("outcome-development config fields do not match")
        exact_types: dict[str, type[object]] = {
            "schema": str,
            "max_contexts": int,
            "confirmation_horizon": int,
            "initial_reacquisition_cost": float,
            "cost_ema_decay": float,
            "max_abs_cost": float,
            "minimax_guardrail_slack": float,
            "evidence_level": str,
            "status": str,
            "scientific_promotion_allowed": bool,
            "thresholds_used": bool,
            "selection_performed": bool,
        }
        if any(type(payload[name]) is not expected for name, expected in exact_types.items()):
            raise ValueError("outcome-development config field types are not canonical")
        candidate = cls(
            max_contexts=cast(int, payload["max_contexts"]),
            confirmation_horizon=cast(int, payload["confirmation_horizon"]),
            initial_reacquisition_cost=cast(float, payload["initial_reacquisition_cost"]),
            cost_ema_decay=cast(float, payload["cost_ema_decay"]),
            max_abs_cost=cast(float, payload["max_abs_cost"]),
            minimax_guardrail_slack=cast(float, payload["minimax_guardrail_slack"]),
        )
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("outcome-development config is not canonical")
        return candidate

    def mechanism_config(self) -> ProspectiveLineageRetentionConfig:
        return ProspectiveLineageRetentionConfig(
            max_contexts=self.max_contexts,
            initial_reacquisition_cost=self.initial_reacquisition_cost,
            cost_ema_decay=self.cost_ema_decay,
            max_abs_cost=self.max_abs_cost,
            minimax_guardrail_slack=self.minimax_guardrail_slack,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionOutcomeCell:
    """One raw prior/route/future cell; probabilities are intentionally absent."""

    schema: str
    prior_panel: str
    routed: bool
    future_lineage: str
    supplied_return_priors: tuple[float, float, float]
    prior_declaration_sha256: str
    true_evaluation_law_supplied_to_cell: bool
    initial_expected_scores: tuple[float, float, float]
    initial_minimax_scores: tuple[float, float, float]
    initial_protection_to_route: tuple[float, float, float]
    initial_expected_victim_slot: int
    initial_minimax_victim_slot: int
    initial_selected_victim_slot: int
    ordinary_lru_comparator_slot: int
    committed_victim_slot: int
    evicted_lineage: str
    future_lineage_retained_after_eviction: bool
    future_lineage_archived_after_eviction: bool
    future_observations_seen_before_eviction: int
    future_observations: tuple[float, float]
    archived_prediction: tuple[float, float]
    fresh_prediction: tuple[float, float]
    live_predictions: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    archived_loss: float
    fresh_loss: float
    live_losses: tuple[float, float, float]
    strict_h2_confirmation: bool
    core_confirmation_requested: bool
    restoration_applied: bool
    prior_restored: bool
    cost_updated: bool
    parameter_transplanted: bool
    core_cost_observation: float
    initial_archived_reacquisition_cost: float
    restored_reacquisition_cost: float
    pre_confirmation_target_prior: float
    pre_confirmation_target_score: float
    post_confirmation_target_prior: float
    post_confirmation_target_score: float
    raw_recurrence_cost: float
    retention_count: int
    restoration_count: int
    allocation_count: int
    eviction_count: int
    birth_churn_count: int
    lineage_rebind_count: int
    prefix_sha256: str
    eviction_state_sha256: str
    future_observation_sha256: str
    final_state_sha256: str
    preparations: int
    settlements: int
    host_squared_error_cells: int
    random_draws: int
    rng_stream_nbytes: int
    rng_stream_sha256: str
    state_nbytes: int
    fixed_output_nbytes: int
    fixed_output_sha256: str
    every_core_update_applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionPriorPanel:
    """Post-cell aggregation under the evaluator-only declared future law."""

    schema: str
    prior_panel: str
    supplied_return_priors: tuple[float, float, float]
    prior_declaration_sha256: str
    true_future_law: tuple[tuple[str, float], tuple[str, float]]
    true_future_law_sha256: str
    true_future_law_evaluator_only: bool
    raw_cell_keys: tuple[str, str, str, str]
    routed_expected_recurrence_cost: float
    unrouted_expected_recurrence_cost: float
    routed_minimax_recurrence_cost: float
    unrouted_minimax_recurrence_cost: float
    routed_expected_retention_count: float
    unrouted_expected_retention_count: float
    routed_expected_restoration_count: float
    unrouted_expected_restoration_count: float
    routed_total_birth_churn: int
    unrouted_total_birth_churn: int
    threshold_used: bool
    winner_selected: bool
    default_selected: bool
    expected_direction_asserted: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionMatchedAudit:
    """Exact branch-neutral resource, work, RNG, and timing receipts."""

    future_twins_bit_exact_through_eviction_commit: bool
    prefix_groups_checked: int
    priors_bound_at_birth_before_preparation: bool
    true_law_absent_from_raw_cells: bool
    ordinary_lru_b_fixed_before_outcomes: bool
    first_future_observation_is_first_branch_divergence: bool
    restoration_settled_after_h2: bool
    restoration_effect_visible_only_to_later_preparation: bool
    all_core_work_equal: bool
    core_preparations_per_cell: int
    core_settlements_per_cell: int
    core_authentication_repreparations_per_cell: int
    core_total_score_preparations_per_cell: int
    core_score_products_per_cell: int
    all_host_work_equal: bool
    host_squared_error_cells_per_cell: int
    all_rng_streams_equal: bool
    rng_stream_nbytes_per_cell: int
    rng_stream_sha256: str
    all_state_nbytes_equal: bool
    state_nbytes_per_cell: int
    all_fixed_output_nbytes_equal: bool
    fixed_output_nbytes_per_cell: int
    persistent_capacity_growth: int
    replay_capacity: int
    archive_capacity: int


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionScalingRecord:
    """Asymptotic declarations for this fixed-K, finite-future panel."""

    live_capacity_symbol: str
    confirmation_horizon_symbol: str
    future_count_symbol: str
    prior_panel_count_symbol: str
    route_count_symbol: str
    persistent_state_formula_bytes: str
    measured_state_bytes_at_k3: int
    archive_capacity: int
    core_prepare_work: str
    core_settle_work_including_authentication: str
    host_confirmation_work: str
    exhaustive_panel_work: str
    report_cell_count_formula: str
    realized_report_cell_count: int
    unbounded_history_retained: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionOutcomeDevelopmentReport:
    """Complete deterministic, nonwriting L0 report."""

    schema: str
    evidence_level: str
    status: str
    config: ProspectiveLineageRetentionOutcomeDevelopmentConfig
    config_sha256: str
    protocol_sha256: str
    core_source_sha256: str
    evaluator_source_sha256: str
    true_future_law_sha256: str
    cells: tuple[
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
        ProspectiveLineageRetentionOutcomeCell,
    ]
    prior_panels: tuple[
        ProspectiveLineageRetentionPriorPanel,
        ProspectiveLineageRetentionPriorPanel,
    ]
    matched_audit: ProspectiveLineageRetentionMatchedAudit
    scaling: ProspectiveLineageRetentionScalingRecord
    timing: tuple[str, ...]
    limitations: tuple[str, ...]
    thresholds_used: bool
    winner_selected: bool
    default_selected: bool
    expected_direction_asserted: bool
    artifact_written: bool
    evidence_claimed: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionOutcomeValidationReceipt:
    """Strict deterministic reconstruction receipt, not evidence."""

    schema: str
    valid: bool
    report_sha256: str
    raw_cell_count: int
    deterministic_reconstruction_exact: bool
    future_prefixes_exact: bool
    equal_work_rng_and_bytes: bool
    evidence_level: str
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _EvictionPrefix:
    mechanism: ProspectiveLineageRetention
    prior_panel: str
    routed: bool
    supplied_priors: tuple[float, float, float]
    prior_sha256: str
    initial_state: ProspectiveLineageRetentionState
    preparation: ProspectiveLineageRetentionPreparation
    eviction: ProspectiveLineageRetentionProposal
    target_slot: int
    recency_after_eviction: jax.Array
    prefix_sha256: str


def _allocation_event(
    state: ProspectiveLineageRetentionState,
    *,
    target_slot: int,
) -> ProspectiveLineageRetentionEvent:
    post_births = state.bound_birth_words[:_MAX_CONTEXTS].at[target_slot].set(_words(_NEWBORN_ID))
    return ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
        source_step_words=state.step_words,
        post_step_words=_next_words(state.step_words),
        source_birth_words=state.bound_birth_words[:_MAX_CONTEXTS],
        post_birth_words=post_births,
        source_in_use=state.live_in_use,
        post_in_use=state.live_in_use,
        allocated=jnp.asarray(True, dtype=jnp.bool_),
        evicted=jnp.asarray(True, dtype=jnp.bool_),
        target_slot=jnp.asarray(target_slot, dtype=jnp.int32),
        newborn_return_prior=jnp.asarray(0.0, dtype=jnp.float32),
        lineage_transfer_confirmed=jnp.asarray(False, dtype=jnp.bool_),
        transfer_slot=jnp.asarray(-1, dtype=jnp.int32),
        transferred_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        archived_loss=jnp.asarray(0.0, dtype=jnp.float32),
        fresh_loss=jnp.asarray(0.0, dtype=jnp.float32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def _neutral_confirmation_event(
    state: ProspectiveLineageRetentionState,
    *,
    target_slot: int,
    confirm: bool,
    archived_loss: float,
    fresh_loss: float,
) -> ProspectiveLineageRetentionEvent:
    archive_index = _MAX_CONTEXTS
    return ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
        source_step_words=state.step_words,
        post_step_words=_next_words(state.step_words),
        source_birth_words=state.bound_birth_words[:_MAX_CONTEXTS],
        post_birth_words=state.bound_birth_words[:_MAX_CONTEXTS],
        source_in_use=state.live_in_use,
        post_in_use=state.live_in_use,
        allocated=jnp.asarray(False, dtype=jnp.bool_),
        evicted=jnp.asarray(False, dtype=jnp.bool_),
        target_slot=jnp.asarray(-1, dtype=jnp.int32),
        newborn_return_prior=jnp.asarray(0.0, dtype=jnp.float32),
        lineage_transfer_confirmed=jnp.asarray(confirm, dtype=jnp.bool_),
        transfer_slot=jnp.asarray(target_slot if confirm else -1, dtype=jnp.int32),
        transferred_lineage_words=(
            state.lineage_words[archive_index] if confirm else jnp.zeros((2,), dtype=jnp.uint32)
        ),
        archived_loss=jnp.asarray(archived_loss if confirm else 0.0, dtype=jnp.float32),
        fresh_loss=jnp.asarray(fresh_loss if confirm else 0.0, dtype=jnp.float32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def _prepare(
    mechanism: ProspectiveLineageRetention,
    state: ProspectiveLineageRetentionState,
    *,
    active_slot: int,
    recency: jax.Array,
    routed: bool,
) -> ProspectiveLineageRetentionPreparation:
    return mechanism.prepare(
        state,
        source_birth_words=state.bound_birth_words[:_MAX_CONTEXTS],
        source_in_use=state.live_in_use,
        active_slot=jnp.asarray(active_slot, dtype=jnp.int32),
        last_active_words=recency,
        route_protection=jnp.asarray(routed, dtype=jnp.bool_),
    )


def _build_eviction_prefix(
    config: ProspectiveLineageRetentionOutcomeDevelopmentConfig,
    *,
    prior_panel: str,
    supplied_priors: tuple[float, float, float],
    routed: bool,
) -> _EvictionPrefix:
    """Build the entire common prefix without a future or probability input."""

    mechanism = ProspectiveLineageRetention(config.mechanism_config())
    initial_state = mechanism.init(
        step_words=_words(_INITIAL_STEP),
        birth_words=_births(1, 2, 3),
        in_use=jnp.ones((_MAX_CONTEXTS,), dtype=jnp.bool_),
        return_priors=jnp.asarray(supplied_priors, dtype=jnp.float32),
    )
    recency = _births(*_INITIAL_RECENCY)
    preparation = _prepare(
        mechanism,
        initial_state,
        active_slot=_ACTIVE_SLOT,
        recency=recency,
        routed=routed,
    )
    target = int(preparation.selected_victim) if routed else _LRU_COMPARATOR_SLOT
    eviction = mechanism.settle(
        initial_state,
        preparation,
        _allocation_event(initial_state, target_slot=target),
    )
    if not bool(preparation.preparation_valid) or not bool(eviction.update_applied):
        raise RuntimeError("fixed eviction prefix did not satisfy the mechanism contract")
    recency_after = recency.at[target].set(eviction.state.step_words)
    prior_payload = {
        "panel": prior_panel,
        "birth_labels": ["A", "B", "D"],
        "return_priors": list(supplied_priors),
        "visible_at_birth": True,
    }
    prefix_sha = _sha256(
        {
            "prior_sha256": _sha256(prior_payload),
            "routed": routed,
            "initial_state_sha256": _tree_sha256(initial_state),
            "preparation_sha256": _tree_sha256(preparation),
            "eviction_proposal_sha256": _tree_sha256(eviction),
            "recency_after_eviction_sha256": _tree_sha256(recency_after),
            "future_observation_count": 0,
        }
    )
    return _EvictionPrefix(
        mechanism=mechanism,
        prior_panel=prior_panel,
        routed=routed,
        supplied_priors=supplied_priors,
        prior_sha256=_sha256(prior_payload),
        initial_state=initial_state,
        preparation=preparation,
        eviction=eviction,
        target_slot=target,
        recency_after_eviction=recency_after,
        prefix_sha256=prefix_sha,
    )


def _fixed_output_bytes(
    cell_values: Sequence[float],
) -> bytes:
    if len(cell_values) != 32:
        raise ValueError("fixed numeric output must contain exactly 32 values")
    return struct.pack(">32d", *(float(value) for value in cell_values))


def _complete_raw_cell(
    prefix: _EvictionPrefix,
    *,
    future_lineage: str,
) -> ProspectiveLineageRetentionOutcomeCell:
    """Diverge after the committed prefix using observations, never a law."""

    if future_lineage not in _FUTURES:
        raise ValueError(f"unsupported future lineage {future_lineage!r}")
    mechanism = prefix.mechanism
    post_eviction = prefix.eviction.state
    archive_index = _MAX_CONTEXTS
    observations = _prediction(future_lineage)
    evicted_lineage = _lineage_label(post_eviction.lineage_words[archive_index])

    # This preparation precedes both observations.  Its values therefore
    # cannot contain the H=2 restoration result that follows.
    pre_confirmation = _prepare(
        mechanism,
        post_eviction,
        active_slot=prefix.target_slot,
        recency=prefix.recency_after_eviction,
        routed=prefix.routed,
    )
    if not bool(pre_confirmation.preparation_valid):
        raise RuntimeError("pre-confirmation preparation is invalid")

    archived_prediction = _prediction(evicted_lineage)
    fresh_prediction = _prediction("X")
    live_labels = tuple(
        _lineage_label(post_eviction.lineage_words[index]) for index in range(_MAX_CONTEXTS)
    )
    live_predictions = cast(
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        tuple(_prediction(label) for label in live_labels),
    )
    archived_loss = _mean_squared_loss(archived_prediction, observations)
    fresh_loss = _mean_squared_loss(fresh_prediction, observations)
    live_losses = cast(
        tuple[float, float, float],
        tuple(_mean_squared_loss(prediction, observations) for prediction in live_predictions),
    )
    strict_confirmation = archived_loss < fresh_loss and all(
        archived_loss < loss for loss in live_losses
    )
    confirmation = mechanism.settle(
        post_eviction,
        pre_confirmation,
        _neutral_confirmation_event(
            post_eviction,
            target_slot=prefix.target_slot,
            confirm=strict_confirmation,
            archived_loss=archived_loss,
            fresh_loss=fresh_loss,
        ),
    )
    if not bool(confirmation.update_applied):
        raise RuntimeError("fixed confirmation settlement did not apply")

    post_confirmation = _prepare(
        mechanism,
        confirmation.state,
        active_slot=prefix.target_slot,
        recency=prefix.recency_after_eviction,
        routed=prefix.routed,
    )
    if not bool(post_confirmation.preparation_valid):
        raise RuntimeError("post-confirmation preparation is invalid")

    future_retained = future_lineage in live_labels
    future_archived = future_lineage == evicted_lineage
    if future_retained == future_archived:
        raise RuntimeError("future lineage must be in exactly one post-eviction location")
    raw_cost = min(live_losses) if future_retained else fresh_loss
    target = prefix.target_slot
    numeric_values = (
        float(prefix.routed),
        float(0 if future_lineage == "B" else 1),
        *prefix.supplied_priors,
        *tuple(float(value) for value in np.asarray(prefix.preparation.expected_scores)),
        *tuple(float(value) for value in np.asarray(prefix.preparation.minimax_scores)),
        float(prefix.preparation.selected_victim),
        float(prefix.target_slot),
        *observations,
        *archived_prediction,
        *fresh_prediction,
        archived_loss,
        fresh_loss,
        *live_losses,
        float(strict_confirmation),
        float(confirmation.lineage_restored),
        float(raw_cost),
        float(pre_confirmation.expected_scores[target]),
        float(post_confirmation.expected_scores[target]),
        float(confirmation.state.reacquisition_cost[target]),
        float(future_retained),
        float(future_archived),
    )
    output_bytes = _fixed_output_bytes(numeric_values)
    return ProspectiveLineageRetentionOutcomeCell(
        schema=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_CELL_SCHEMA,
        prior_panel=prefix.prior_panel,
        routed=prefix.routed,
        future_lineage=future_lineage,
        supplied_return_priors=prefix.supplied_priors,
        prior_declaration_sha256=prefix.prior_sha256,
        true_evaluation_law_supplied_to_cell=False,
        initial_expected_scores=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.asarray(prefix.preparation.expected_scores)),
        ),
        initial_minimax_scores=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.asarray(prefix.preparation.minimax_scores)),
        ),
        initial_protection_to_route=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.asarray(prefix.preparation.protection_to_route)),
        ),
        initial_expected_victim_slot=int(prefix.preparation.expected_victim),
        initial_minimax_victim_slot=int(prefix.preparation.minimax_victim),
        initial_selected_victim_slot=int(prefix.preparation.selected_victim),
        ordinary_lru_comparator_slot=_LRU_COMPARATOR_SLOT,
        committed_victim_slot=prefix.target_slot,
        evicted_lineage=evicted_lineage,
        future_lineage_retained_after_eviction=future_retained,
        future_lineage_archived_after_eviction=future_archived,
        future_observations_seen_before_eviction=0,
        future_observations=observations,
        archived_prediction=archived_prediction,
        fresh_prediction=fresh_prediction,
        live_predictions=live_predictions,
        archived_loss=archived_loss,
        fresh_loss=fresh_loss,
        live_losses=live_losses,
        strict_h2_confirmation=strict_confirmation,
        core_confirmation_requested=strict_confirmation,
        restoration_applied=bool(confirmation.lineage_restored),
        prior_restored=bool(confirmation.prior_restored),
        cost_updated=bool(confirmation.cost_updated),
        parameter_transplanted=bool(confirmation.parameter_transplanted),
        core_cost_observation=float(confirmation.cost_observation),
        initial_archived_reacquisition_cost=float(post_eviction.reacquisition_cost[archive_index]),
        restored_reacquisition_cost=float(confirmation.state.reacquisition_cost[target]),
        pre_confirmation_target_prior=float(post_eviction.return_prior[target]),
        pre_confirmation_target_score=float(pre_confirmation.expected_scores[target]),
        post_confirmation_target_prior=float(confirmation.state.return_prior[target]),
        post_confirmation_target_score=float(post_confirmation.expected_scores[target]),
        raw_recurrence_cost=raw_cost,
        retention_count=int(future_retained),
        restoration_count=int(bool(confirmation.lineage_restored)),
        allocation_count=1,
        eviction_count=1,
        birth_churn_count=1,
        lineage_rebind_count=int(bool(confirmation.lineage_restored)),
        prefix_sha256=prefix.prefix_sha256,
        eviction_state_sha256=_tree_sha256(post_eviction),
        future_observation_sha256=_sha256({"observations": observations}),
        final_state_sha256=_tree_sha256(confirmation.state),
        preparations=3,
        settlements=2,
        host_squared_error_cells=(_MAX_CONTEXTS + 2) * _HORIZON,
        random_draws=0,
        rng_stream_nbytes=0,
        rng_stream_sha256=_EMPTY_SHA256,
        state_nbytes=_tree_nbytes(confirmation.state),
        fixed_output_nbytes=len(output_bytes),
        fixed_output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        every_core_update_applied=(
            bool(prefix.eviction.update_applied) and bool(confirmation.update_applied)
        ),
    )


def _cell_key(cell: ProspectiveLineageRetentionOutcomeCell) -> str:
    route = "routed" if cell.routed else "unrouted"
    return f"{cell.prior_panel}:{route}:{cell.future_lineage}"


def _aggregate(
    cells: Sequence[ProspectiveLineageRetentionOutcomeCell],
    *,
    prior_panel: str,
    supplied_priors: tuple[float, float, float],
) -> ProspectiveLineageRetentionPriorPanel:
    selected = tuple(cell for cell in cells if cell.prior_panel == prior_panel)
    if len(selected) != 4:
        raise RuntimeError("each prior panel must contain four raw cells")
    law = dict(_TRUE_FUTURE_LAW)

    def route_cells(routed: bool) -> tuple[ProspectiveLineageRetentionOutcomeCell, ...]:
        result = tuple(cell for cell in selected if cell.routed is routed)
        if len(result) != 2:
            raise RuntimeError("each route must contain both future cells")
        return result

    def expected(field: str, routed: bool) -> float:
        return float(
            sum(
                law[cell.future_lineage] * float(getattr(cell, field))
                for cell in route_cells(routed)
            )
        )

    def worst(field: str, routed: bool) -> float:
        return max(float(getattr(cell, field)) for cell in route_cells(routed))

    routed_cells = route_cells(True)
    unrouted_cells = route_cells(False)
    prior_sha = routed_cells[0].prior_declaration_sha256
    return ProspectiveLineageRetentionPriorPanel(
        schema=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_PANEL_SCHEMA,
        prior_panel=prior_panel,
        supplied_return_priors=supplied_priors,
        prior_declaration_sha256=prior_sha,
        true_future_law=_TRUE_FUTURE_LAW,
        true_future_law_sha256=_sha256({"true_future_law": _TRUE_FUTURE_LAW}),
        true_future_law_evaluator_only=True,
        raw_cell_keys=cast(
            tuple[str, str, str, str],
            tuple(_cell_key(cell) for cell in selected),
        ),
        routed_expected_recurrence_cost=expected("raw_recurrence_cost", True),
        unrouted_expected_recurrence_cost=expected("raw_recurrence_cost", False),
        routed_minimax_recurrence_cost=worst("raw_recurrence_cost", True),
        unrouted_minimax_recurrence_cost=worst("raw_recurrence_cost", False),
        routed_expected_retention_count=expected("retention_count", True),
        unrouted_expected_retention_count=expected("retention_count", False),
        routed_expected_restoration_count=expected("restoration_count", True),
        unrouted_expected_restoration_count=expected("restoration_count", False),
        routed_total_birth_churn=sum(cell.birth_churn_count for cell in routed_cells),
        unrouted_total_birth_churn=sum(cell.birth_churn_count for cell in unrouted_cells),
        threshold_used=False,
        winner_selected=False,
        default_selected=False,
        expected_direction_asserted=False,
    )


def _report_sha256(report: ProspectiveLineageRetentionOutcomeDevelopmentReport) -> str:
    return _sha256(dataclasses.asdict(report))


def _require_exact_structure_types(
    actual: object,
    expected: object,
    *,
    path: str,
) -> None:
    """Reject Python-equal bool/int and container/dataclass type aliases."""

    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        if type(actual) is not type(expected):
            raise ValueError(f"{path} does not have the exact reconstructed dataclass type")
        for field in dataclasses.fields(expected):
            _require_exact_structure_types(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if type(expected) is tuple:
        if type(actual) is not tuple or len(cast(tuple[object, ...], actual)) != len(expected):
            raise ValueError(f"{path} does not have the exact reconstructed tuple type")
        for index, (actual_item, expected_item) in enumerate(
            zip(cast(tuple[object, ...], actual), expected, strict=True)
        ):
            _require_exact_structure_types(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"{path} does not have the exact reconstructed scalar type")


def build_prospective_lineage_retention_outcome_development_report(
    config: ProspectiveLineageRetentionOutcomeDevelopmentConfig | None = None,
) -> ProspectiveLineageRetentionOutcomeDevelopmentReport:
    """Build all raw cells first, then aggregate under the evaluator-only law."""

    resolved = ProspectiveLineageRetentionOutcomeDevelopmentConfig() if config is None else config
    if type(resolved) is not ProspectiveLineageRetentionOutcomeDevelopmentConfig:
        raise TypeError("config must be ProspectiveLineageRetentionOutcomeDevelopmentConfig")

    # No prefix builder accepts a future identity, observation, probability, or
    # true-law input.  All eight independently executed prefixes exist before
    # any future identity is paired with one or any raw future is completed.
    prefixes = tuple(
        _build_eviction_prefix(
            resolved,
            prior_panel=panel,
            supplied_priors=priors,
            routed=routed,
        )
        for panel, priors in _PRIOR_PANELS
        for routed in (True, False)
        for _replica in range(len(_FUTURES))
    )
    future_assignments = tuple(
        future
        for _panel, _priors in _PRIOR_PANELS
        for _routed in (True, False)
        for future in _FUTURES
    )
    raw_cells = tuple(
        _complete_raw_cell(prefix, future_lineage=future)
        for prefix, future in zip(prefixes, future_assignments, strict=True)
    )
    cells = cast(
        tuple[
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
            ProspectiveLineageRetentionOutcomeCell,
        ],
        raw_cells,
    )
    prior_panels = cast(
        tuple[
            ProspectiveLineageRetentionPriorPanel,
            ProspectiveLineageRetentionPriorPanel,
        ],
        tuple(
            _aggregate(cells, prior_panel=panel, supplied_priors=priors)
            for panel, priors in _PRIOR_PANELS
        ),
    )

    prefix_groups: dict[tuple[str, bool], set[str]] = {}
    for cell in cells:
        prefix_groups.setdefault((cell.prior_panel, cell.routed), set()).add(cell.prefix_sha256)
    work = ProspectiveLineageRetention(resolved.mechanism_config()).work_record(
        preparations=3,
        settlements=2,
    )
    resources = ProspectiveLineageRetention(resolved.mechanism_config()).resource_record()
    fixed_output_sizes = {cell.fixed_output_nbytes for cell in cells}
    state_sizes = {cell.state_nbytes for cell in cells}
    rng_receipts = {(cell.rng_stream_nbytes, cell.rng_stream_sha256) for cell in cells}
    core_work_receipts = {
        (
            cell.preparations,
            cell.settlements,
            work.authentication_repreparations,
            work.total_score_preparations,
            work.score_products,
        )
        for cell in cells
    }
    matched_audit = ProspectiveLineageRetentionMatchedAudit(
        future_twins_bit_exact_through_eviction_commit=all(
            len(digests) == 1 for digests in prefix_groups.values()
        ),
        prefix_groups_checked=len(prefix_groups),
        priors_bound_at_birth_before_preparation=True,
        true_law_absent_from_raw_cells=all(
            not cell.true_evaluation_law_supplied_to_cell for cell in cells
        ),
        ordinary_lru_b_fixed_before_outcomes=all(
            cell.ordinary_lru_comparator_slot == _LRU_COMPARATOR_SLOT
            and (cell.routed or cell.committed_victim_slot == _LRU_COMPARATOR_SLOT)
            for cell in cells
        ),
        first_future_observation_is_first_branch_divergence=True,
        restoration_settled_after_h2=True,
        restoration_effect_visible_only_to_later_preparation=True,
        all_core_work_equal=len(core_work_receipts) == 1,
        core_preparations_per_cell=3,
        core_settlements_per_cell=2,
        core_authentication_repreparations_per_cell=work.authentication_repreparations,
        core_total_score_preparations_per_cell=work.total_score_preparations,
        core_score_products_per_cell=work.score_products,
        all_host_work_equal=len({cell.host_squared_error_cells for cell in cells}) == 1,
        host_squared_error_cells_per_cell=(_MAX_CONTEXTS + 2) * _HORIZON,
        all_rng_streams_equal=len(rng_receipts) == 1,
        rng_stream_nbytes_per_cell=0,
        rng_stream_sha256=_EMPTY_SHA256,
        all_state_nbytes_equal=len(state_sizes) == 1,
        state_nbytes_per_cell=resources.state_nbytes,
        all_fixed_output_nbytes_equal=len(fixed_output_sizes) == 1,
        fixed_output_nbytes_per_cell=32 * 8,
        persistent_capacity_growth=resources.persistent_capacity_growth,
        replay_capacity=resources.replay_capacity,
        archive_capacity=resources.archive_capacity,
    )
    protocol = {
        "initial_bank": ["A", "B", "D"],
        "active_slot": _ACTIVE_SLOT,
        "initial_recency": list(_INITIAL_RECENCY),
        "ordinary_lru_comparator_slot": _LRU_COMPARATOR_SLOT,
        "ordinary_lru_comparator_lineage": "B",
        "newborn": "X",
        "confirmation_horizon": _HORIZON,
        "future_observations": {future: list(_prediction(future)) for future in _FUTURES},
        "prior_panels": [
            {"name": name, "return_priors": list(priors)} for name, priors in _PRIOR_PANELS
        ],
        "true_future_law": list(_TRUE_FUTURE_LAW),
        "true_law_use": "post-cell-aggregation-only",
    }
    core_path = Path(__file__).resolve().parents[1] / "core" / "prospective_lineage_retention.py"
    return ProspectiveLineageRetentionOutcomeDevelopmentReport(
        schema=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_REPORT_SCHEMA,
        evidence_level=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_EVIDENCE_LEVEL,
        status=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_STATUS,
        config=resolved,
        config_sha256=_sha256(resolved.to_config()),
        protocol_sha256=_sha256(protocol),
        core_source_sha256=_file_sha256(core_path),
        evaluator_source_sha256=_file_sha256(Path(__file__).resolve()),
        true_future_law_sha256=_sha256({"true_future_law": _TRUE_FUTURE_LAW}),
        cells=cells,
        prior_panels=prior_panels,
        matched_audit=matched_audit,
        scaling=ProspectiveLineageRetentionScalingRecord(
            live_capacity_symbol="K",
            confirmation_horizon_symbol="H",
            future_count_symbol="F",
            prior_panel_count_symbol="P",
            route_count_symbol="R",
            persistent_state_formula_bytes="64 + 57 * (K + 1) + K",
            measured_state_bytes_at_k3=resources.state_nbytes,
            archive_capacity=resources.archive_capacity,
            core_prepare_work="O(K)",
            core_settle_work_including_authentication="O(K)",
            host_confirmation_work="O(H * (K + 2))",
            exhaustive_panel_work="O(P * R * F * H * K)",
            report_cell_count_formula="P * R * F",
            realized_report_cell_count=len(cells),
            unbounded_history_retained=False,
        ),
        timing=(
            "Each A/B/D return-prior vector is supplied and receipted at genesis birth.",
            "The X admission and ordinary-LRU=B comparator rule are fixed before outcomes.",
            "Preparation and victim selection occur with zero future observations revealed.",
            "Both future twins are bit-identical through the eviction settlement commit.",
            "The twins first diverge at the first of two post-eviction observations.",
            "All H=2 losses are computed before any restoration confirmation is settled.",
            "Restored prior and cost can affect only the later third preparation.",
            "The true future law is applied only after all eight raw cells exist.",
        ),
        limitations=(
            "The supplied birth-time return priors are declared inputs, not learned hazards.",
            "The two future probabilities are an evaluator law, not a population estimate.",
            "Fixed diagnostic predictors are fixtures, not learned general features.",
            (
                "The core cannot authenticate host losses; this evaluator binds one "
                "synthetic H=2 rule."
            ),
            "The archive stores metadata only and never transplants predictor parameters.",
            "Restoration is lineage/prior/cost metadata, not demonstrated behavioral recovery.",
            "The life has no reward learner, action learning, multi-agent adaptation, or control.",
            (
                "The exhaustive two-future result does not establish whole-agent "
                "forgetting resistance."
            ),
            "Expected and minimax costs are descriptive and have no statistical interval.",
            "Zero RNG is appropriate only to this deterministic finite panel.",
            "No threshold, winner, default, artifact, evidence, or promotion follows.",
        ),
        thresholds_used=False,
        winner_selected=False,
        default_selected=False,
        expected_direction_asserted=False,
        artifact_written=False,
        evidence_claimed=False,
        scientific_promotion_allowed=False,
    )


def validate_prospective_lineage_retention_outcome_development_report(
    report: ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> ProspectiveLineageRetentionOutcomeValidationReceipt:
    """Fail closed by exact deterministic reconstruction from canonical config."""

    if type(report) is not ProspectiveLineageRetentionOutcomeDevelopmentReport:
        raise TypeError("report must be ProspectiveLineageRetentionOutcomeDevelopmentReport")
    if type(report.config) is not ProspectiveLineageRetentionOutcomeDevelopmentConfig:
        raise ValueError("report config type is not canonical")
    canonical_config = ProspectiveLineageRetentionOutcomeDevelopmentConfig.from_config(
        report.config.to_config()
    )
    expected = build_prospective_lineage_retention_outcome_development_report(canonical_config)
    _require_exact_structure_types(report, expected, path="report")
    try:
        actual_bytes = _canonical_json_bytes(dataclasses.asdict(report))
        expected_bytes = _canonical_json_bytes(dataclasses.asdict(expected))
    except (TypeError, ValueError) as error:
        raise ValueError("outcome-development report is not canonical JSON") from error
    if actual_bytes != expected_bytes:
        raise ValueError("outcome-development report canonical bytes differ from reconstruction")
    if report != expected:
        raise ValueError("outcome-development report differs from deterministic reconstruction")
    if not (
        report.matched_audit.future_twins_bit_exact_through_eviction_commit
        and report.matched_audit.all_core_work_equal
        and report.matched_audit.all_host_work_equal
        and report.matched_audit.all_rng_streams_equal
        and report.matched_audit.all_state_nbytes_equal
        and report.matched_audit.all_fixed_output_nbytes_equal
    ):
        raise ValueError("outcome-development matched audit is incomplete")
    return ProspectiveLineageRetentionOutcomeValidationReceipt(
        schema=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_VALIDATION_SCHEMA,
        valid=True,
        report_sha256=_report_sha256(report),
        raw_cell_count=len(report.cells),
        deterministic_reconstruction_exact=True,
        future_prefixes_exact=True,
        equal_work_rng_and_bytes=True,
        evidence_level=PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_EVIDENCE_LEVEL,
        scientific_promotion_allowed=False,
    )


__all__ = [
    "PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_CONFIG_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_EVIDENCE_LEVEL",
    "PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_REPORT_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_STATUS",
    "ProspectiveLineageRetentionMatchedAudit",
    "ProspectiveLineageRetentionOutcomeCell",
    "ProspectiveLineageRetentionOutcomeDevelopmentConfig",
    "ProspectiveLineageRetentionOutcomeDevelopmentReport",
    "ProspectiveLineageRetentionOutcomeValidationReceipt",
    "ProspectiveLineageRetentionPriorPanel",
    "ProspectiveLineageRetentionScalingRecord",
    "build_prospective_lineage_retention_outcome_development_report",
    "validate_prospective_lineage_retention_outcome_development_report",
]
