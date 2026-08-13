# mypy: disable-error-code="call-arg,name-defined"
"""Bounded prior-conditioned prospective retention for semantic lineages.

This module is a mechanism-only L0 sidecar.  It turns an immutable, birth-bound
return prior and a causally observed reacquisition-cost estimate into a
full-bank eviction-protection score::

    protection = return_prior * causal_reacquisition_cost

The preparation is strictly pre-outcome and does not mutate state.  A separate
settlement may bind one completed allocation/eviction or one independently
confirmed cross-birth lineage restoration.  Settlement cannot change the
preparation that selected the current eviction.  A composing outer transaction
must commit this sidecar together with its context bank and lineage-confirmation
mechanism, or commit none of them.

The fixed metadata bank has ``K + 1`` rows: ``K`` live context rows followed by
one archive row.  The archive retains retention metadata only; this module does
not store or transplant a context model.  A confirmed restoration preserves
the archived return prior, updates its cost estimate from the non-negative
``fresh_loss - archived_loss`` advantage, and rebinds the stable lineage to the
new semantic birth.

The source-bound receipts and content checksum are integrity checks, not
cryptographic authentication or proof of caller provenance.  In particular,
this core cannot verify that caller-supplied losses came from a genuine H=2
comparison.  The outer evaluator owns that causal binding.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

PROSPECTIVE_LINEAGE_RETENTION_CONFIG_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.config.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_STATE_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.state.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_PREPARATION_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.preparation.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_EVENT_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.event.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_PROPOSAL_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.proposal.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_RESOURCE_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.resource.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_WORK_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.work.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA: Final = (
    "alberta.prospective-lineage-retention.checkpoint.v1"
)
PROSPECTIVE_LINEAGE_RETENTION_EVIDENCE_LEVEL: Final = "L0"
PROSPECTIVE_LINEAGE_RETENTION_STATUS: Final = "mechanism-only-not-assessed"
PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY: Final = 1
PROSPECTIVE_LINEAGE_RETENTION_REPLAY_CAPACITY: Final = 0
PROSPECTIVE_LINEAGE_RETENTION_EXTERNAL_PROVENANCE_CLAIMED: Final = False
PROSPECTIVE_LINEAGE_RETENTION_HOST_EVENT_BINDING_CLAIMED: Final = False
PROSPECTIVE_LINEAGE_RETENTION_SCIENTIFIC_PROMOTION_ALLOWED: Final = False

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_CONFIG_TOKEN_NBYTES = 32
_CONTENT_TOKEN_WORDS = 4


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _finite_float(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum or converted > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return converted


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32), axis=-1)


def _rows_unique(words: Array, mask: Array) -> Array:
    equal = jnp.all(words[:, None, :] == words[None, :, :], axis=-1)
    relevant = mask[:, None] & mask[None, :]
    diagonal = jnp.eye(words.shape[0], dtype=jnp.bool_)
    return ~jnp.any(equal & relevant & ~diagonal)


def _rotate_left(value: Array, amount: int) -> Array:
    return (value << jnp.asarray(amount, dtype=jnp.uint32)) | (
        value >> jnp.asarray(32 - amount, dtype=jnp.uint32)
    )


def _float_words(values: Array) -> Array:
    return jax.lax.bitcast_convert_type(values.astype(jnp.float32), jnp.uint32)


def _prior_receipts(source_birth_words: Array, return_prior: Array, valid: Array) -> Array:
    bits = _float_words(return_prior)
    receipt = jnp.stack(
        (
            source_birth_words[:, 0] ^ jnp.asarray(0x50524C31, dtype=jnp.uint32),
            source_birth_words[:, 1] ^ jnp.asarray(0x50524C32, dtype=jnp.uint32),
            bits ^ jnp.asarray(0x50524C33, dtype=jnp.uint32),
            (
                source_birth_words[:, 0]
                + _rotate_left(source_birth_words[:, 1], 11)
                + _rotate_left(bits, 19)
                + jnp.asarray(0x50524C34, dtype=jnp.uint32)
            ),
        ),
        axis=1,
    ).astype(jnp.uint32)
    return jnp.where(valid[:, None], receipt, jnp.zeros_like(receipt))


def _preparation_receipt(source_content_token: Array, route_protection: Array) -> Array:
    route_word = jnp.where(
        route_protection,
        jnp.asarray(0x52544F4E, dtype=jnp.uint32),
        jnp.asarray(0x52544F46, dtype=jnp.uint32),
    )
    return (
        source_content_token
        ^ jnp.asarray(
            (0x50524531, 0x50524532, 0x50524533, 0x50524534),
            dtype=jnp.uint32,
        )
        ^ jnp.stack(
            (
                route_word,
                _rotate_left(route_word, 7),
                _rotate_left(route_word, 13),
                _rotate_left(route_word, 23),
            )
        )
    ).astype(jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionConfig:
    """Static geometry and conservative cost/guardrail settings."""

    max_contexts: int
    initial_reacquisition_cost: float = 1.0
    cost_ema_decay: float = 0.9
    max_abs_cost: float = 1_000.0
    minimax_guardrail_slack: float = 0.0

    def __post_init__(self) -> None:
        if type(self.max_contexts) is not int or self.max_contexts < 2:
            raise ValueError("max_contexts must be an integer of at least two")
        initial = _finite_float(
            self.initial_reacquisition_cost,
            name="initial_reacquisition_cost",
            minimum=0.0,
            maximum=float(self.max_abs_cost),
        )
        maximum = _finite_float(
            self.max_abs_cost,
            name="max_abs_cost",
            minimum=max(initial, 1.0e-12),
            maximum=_FLOAT32_MAX,
        )
        _finite_float(
            self.cost_ema_decay,
            name="cost_ema_decay",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_float(
            self.minimax_guardrail_slack,
            name="minimax_guardrail_slack",
            minimum=0.0,
            maximum=maximum,
        )

    @property
    def metadata_capacity(self) -> int:
        return self.max_contexts + PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROSPECTIVE_LINEAGE_RETENTION_CONFIG_SCHEMA,
            "max_contexts": self.max_contexts,
            "metadata_capacity": self.metadata_capacity,
            "archive_capacity": PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY,
            "initial_reacquisition_cost": float(self.initial_reacquisition_cost),
            "cost_ema_decay": float(self.cost_ema_decay),
            "max_abs_cost": float(self.max_abs_cost),
            "minimax_guardrail_slack": float(self.minimax_guardrail_slack),
            "score_semantics": "return_prior_times_causal_reacquisition_cost",
            "evidence_level": PROSPECTIVE_LINEAGE_RETENTION_EVIDENCE_LEVEL,
            "status": PROSPECTIVE_LINEAGE_RETENTION_STATUS,
            "scientific_promotion_allowed": False,
            "external_provenance_claimed": False,
            "host_event_binding_claimed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ProspectiveLineageRetentionConfig:
        required = {
            "schema",
            "max_contexts",
            "metadata_capacity",
            "archive_capacity",
            "initial_reacquisition_cost",
            "cost_ema_decay",
            "max_abs_cost",
            "minimax_guardrail_slack",
            "score_semantics",
            "evidence_level",
            "status",
            "scientific_promotion_allowed",
            "external_provenance_claimed",
            "host_event_binding_claimed",
        }
        if type(payload) is not dict or set(payload) != required:
            raise ValueError("prospective-retention config fields do not match")
        exact_types: dict[str, type[object]] = {
            "schema": str,
            "max_contexts": int,
            "metadata_capacity": int,
            "archive_capacity": int,
            "initial_reacquisition_cost": float,
            "cost_ema_decay": float,
            "max_abs_cost": float,
            "minimax_guardrail_slack": float,
            "score_semantics": str,
            "evidence_level": str,
            "status": str,
            "scientific_promotion_allowed": bool,
            "external_provenance_claimed": bool,
            "host_event_binding_claimed": bool,
        }
        if any(type(payload[name]) is not expected for name, expected in exact_types.items()):
            raise ValueError("prospective-retention config field types are not canonical")
        candidate = cls(
            max_contexts=cast(int, payload["max_contexts"]),
            initial_reacquisition_cost=cast(float, payload["initial_reacquisition_cost"]),
            cost_ema_decay=cast(float, payload["cost_ema_decay"]),
            max_abs_cost=cast(float, payload["max_abs_cost"]),
            minimax_guardrail_slack=cast(float, payload["minimax_guardrail_slack"]),
        )
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("prospective-retention config is not canonical")
        return candidate


@chex.dataclass(frozen=True)
class ProspectiveLineageRetentionState:
    """Fixed ``K + 1`` metadata bank and exact source revision."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 4"]
    step_words: UInt[Array, " 2"]
    revision_words: UInt[Array, " 2"]
    valid: Bool[Array, " metadata_capacity"]
    bound_birth_words: UInt[Array, "metadata_capacity 2"]
    lineage_words: UInt[Array, "metadata_capacity 2"]
    prior_source_birth_words: UInt[Array, "metadata_capacity 2"]
    prior_receipt_words: UInt[Array, "metadata_capacity 4"]
    return_prior: Float[Array, " metadata_capacity"]
    reacquisition_cost: Float[Array, " metadata_capacity"]
    cost_support_words: UInt[Array, "metadata_capacity 2"]
    live_in_use: Bool[Array, " max_contexts"]


@chex.dataclass(frozen=True)
class ProspectiveLineageRetentionPreparation:
    """Pre-outcome scores and an exact snapshot of their source authority."""

    source_config_token: UInt[Array, " 32"]
    source_content_token: UInt[Array, " 4"]
    preparation_receipt_words: UInt[Array, " 4"]
    source_step_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    source_birth_words: UInt[Array, "max_contexts 2"]
    source_lineage_words: UInt[Array, "max_contexts 2"]
    source_in_use: Bool[Array, " max_contexts"]
    active_slot: Int[Array, ""]
    last_active_words: UInt[Array, "max_contexts 2"]
    eligible_mask: Bool[Array, " max_contexts"]
    expected_scores: Float[Array, " max_contexts"]
    minimax_scores: Float[Array, " max_contexts"]
    raw_protection: Float[Array, " max_contexts"]
    protection_to_route: Float[Array, " max_contexts"]
    expected_victim: Int[Array, ""]
    minimax_victim: Int[Array, ""]
    selected_victim: Int[Array, ""]
    expected_victim_worst_cost: Float[Array, ""]
    minimax_victim_worst_cost: Float[Array, ""]
    guardrail_passed: Bool[Array, ""]
    guardrail_fallback_used: Bool[Array, ""]
    route_protection: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ProspectiveLineageRetentionEvent:
    """One completed host lifecycle event supplied for atomic settlement."""

    source_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_birth_words: UInt[Array, "max_contexts 2"]
    post_birth_words: UInt[Array, "max_contexts 2"]
    source_in_use: Bool[Array, " max_contexts"]
    post_in_use: Bool[Array, " max_contexts"]
    allocated: Bool[Array, ""]
    evicted: Bool[Array, ""]
    target_slot: Int[Array, ""]
    newborn_return_prior: Float[Array, ""]
    lineage_transfer_confirmed: Bool[Array, ""]
    transfer_slot: Int[Array, ""]
    transferred_lineage_words: UInt[Array, " 2"]
    archived_loss: Float[Array, ""]
    fresh_loss: Float[Array, ""]
    context_update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ProspectiveLineageRetentionProposal:
    """Fail-closed successor and bounded settlement diagnostics."""

    state: ProspectiveLineageRetentionState
    source_state_valid: Bool[Array, ""]
    preparation_authenticated: Bool[Array, ""]
    event_valid: Bool[Array, ""]
    step_capacity_available: Bool[Array, ""]
    revision_capacity_available: Bool[Array, ""]
    cost_support_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    routed_eviction_binding_valid: Bool[Array, ""]
    archive_created: Bool[Array, ""]
    archive_replaced: Bool[Array, ""]
    newborn_bound: Bool[Array, ""]
    lineage_restored: Bool[Array, ""]
    prior_restored: Bool[Array, ""]
    cost_updated: Bool[Array, ""]
    cost_observation: Float[Array, ""]
    parameter_transplanted: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionResourceRecord:
    """Exact persistent bytes and fixed-capacity declarations."""

    schema: str
    max_contexts: int
    metadata_capacity: int
    archive_capacity: int
    config_token_nbytes: int
    content_token_nbytes: int
    state_nbytes: int
    measured_state_nbytes: int
    replay_capacity: int
    persistent_capacity_growth: int
    parameter_transplant_allowed: bool
    external_provenance_claimed: bool
    host_event_binding_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveLineageRetentionWorkRecord:
    """Named fixed logical work, not a FLOP or latency claim."""

    schema: str
    preparations: int
    settlements: int
    authentication_repreparations: int
    total_score_preparations: int
    score_products: int
    expected_selection_cells: int
    minimax_selection_cells: int
    guardrail_comparisons: int
    metadata_route_cells: int
    settlement_proposals: int
    random_draws: int
    replay_updates: int
    reset_callbacks: int
    routed_unrouted_same_preparation_work: bool
    exhaustive_primitive_operation_count_claimed: bool
    compiled_flop_count_claimed: bool


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        total += int(value.size) * int(value.dtype.itemsize)
    return total


def _tree_arrays_equal(left: object, right: object) -> Array:
    comparisons = jax.tree.leaves(
        jax.tree.map(lambda lhs, rhs: jnp.array_equal(lhs, rhs), left, right)
    )
    return jnp.all(jnp.stack(tuple(comparisons)))


def _state_content_words(state: ProspectiveLineageRetentionState) -> Array:
    return jnp.concatenate(
        (
            state.config_token.astype(jnp.uint32),
            state.step_words,
            state.revision_words,
            state.valid.astype(jnp.uint32),
            state.bound_birth_words.reshape(-1),
            state.lineage_words.reshape(-1),
            state.prior_source_birth_words.reshape(-1),
            state.prior_receipt_words.reshape(-1),
            _float_words(state.return_prior),
            _float_words(state.reacquisition_cost),
            state.cost_support_words.reshape(-1),
            state.live_in_use.astype(jnp.uint32),
        )
    ).astype(jnp.uint32)


def _state_content_token(state: ProspectiveLineageRetentionState) -> Array:
    words = _state_content_words(state)
    initial = jnp.asarray(
        (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A),
        dtype=jnp.uint32,
    )

    def fold(index: int, carry: Array) -> Array:
        word = words[index] + jnp.asarray(index, dtype=jnp.uint32) * jnp.asarray(
            0x9E3779B9, dtype=jnp.uint32
        )
        return jnp.stack(
            (
                _rotate_left(carry[0] ^ word, 5) + carry[1],
                _rotate_left(carry[1] + word, 11) ^ carry[2],
                _rotate_left(carry[2] ^ (word + carry[0]), 17) + carry[3],
                _rotate_left(carry[3] + (word ^ carry[1]), 23) ^ carry[0],
            )
        ).astype(jnp.uint32)

    return cast(Array, jax.lax.fori_loop(0, words.shape[0], fold, initial))


def _config_token(config: ProspectiveLineageRetentionConfig) -> Array:
    digest = hashlib.sha256(_canonical_json_bytes(config.to_config())).digest()
    return jnp.asarray(tuple(digest), dtype=jnp.uint8)


def _oldest_masked(mask: Array, last_active_words: Array) -> Array:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    high = jnp.where(mask, last_active_words[:, 0], maximum)
    minimum_high = jnp.min(high)
    low = jnp.where(
        mask & (last_active_words[:, 0] == minimum_high),
        last_active_words[:, 1],
        maximum,
    )
    minimum_low = jnp.min(low)
    oldest = mask & (last_active_words[:, 0] == minimum_high) & (
        last_active_words[:, 1] == minimum_low
    )
    return jnp.argmax(oldest.astype(jnp.int32)).astype(jnp.int32)


def _select_score_victim(scores: Array, eligible: Array, last_active_words: Array) -> Array:
    masked = jnp.where(eligible, scores, jnp.asarray(jnp.inf, dtype=jnp.float32))
    minimum = jnp.min(masked)
    return _oldest_masked(eligible & (scores == minimum), last_active_words)


class ProspectiveLineageRetention:
    """Pure fixed-capacity prior-conditioned retention transaction."""

    def __init__(self, config: ProspectiveLineageRetentionConfig):
        if type(config) is not ProspectiveLineageRetentionConfig:
            raise TypeError("config must be ProspectiveLineageRetentionConfig")
        self._config = config
        self._config_token = _config_token(config)

    @property
    def config(self) -> ProspectiveLineageRetentionConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> ProspectiveLineageRetention:
        return cls(ProspectiveLineageRetentionConfig.from_config(payload))

    def _require_state_contract(self, state: ProspectiveLineageRetentionState) -> None:
        k = self._config.max_contexts
        capacity = self._config.metadata_capacity
        _require_array(state.config_token, name="config_token", shape=(32,), dtype=jnp.uint8)
        _require_array(
            state.content_token,
            name="content_token",
            shape=(_CONTENT_TOKEN_WORDS,),
            dtype=jnp.uint32,
        )
        for name in ("step_words", "revision_words"):
            _require_array(getattr(state, name), name=name, shape=(2,), dtype=jnp.uint32)
        _require_array(state.valid, name="valid", shape=(capacity,), dtype=jnp.bool_)
        for name in ("bound_birth_words", "lineage_words", "prior_source_birth_words"):
            _require_array(
                getattr(state, name),
                name=name,
                shape=(capacity, 2),
                dtype=jnp.uint32,
            )
        _require_array(
            state.prior_receipt_words,
            name="prior_receipt_words",
            shape=(capacity, 4),
            dtype=jnp.uint32,
        )
        for name in ("return_prior", "reacquisition_cost"):
            _require_array(
                getattr(state, name),
                name=name,
                shape=(capacity,),
                dtype=jnp.float32,
            )
        _require_array(
            state.cost_support_words,
            name="cost_support_words",
            shape=(capacity, 2),
            dtype=jnp.uint32,
        )
        _require_array(
            state.live_in_use,
            name="live_in_use",
            shape=(k,),
            dtype=jnp.bool_,
        )

    def _with_content_token(
        self,
        state: ProspectiveLineageRetentionState,
    ) -> ProspectiveLineageRetentionState:
        """Reseal mechanism-produced state; this proves no external provenance."""

        unsigned = cast(Any, state).replace(
            content_token=jnp.zeros((_CONTENT_TOKEN_WORDS,), dtype=jnp.uint32)
        )
        return cast(
            ProspectiveLineageRetentionState,
            unsigned.replace(content_token=_state_content_token(unsigned)),
        )

    def init_empty(self) -> ProspectiveLineageRetentionState:
        """Return an empty fixed-shape template, suitable for checkpoint restore."""

        k = self._config.max_contexts
        capacity = self._config.metadata_capacity
        state = ProspectiveLineageRetentionState(
            config_token=self._config_token,
            content_token=jnp.zeros((_CONTENT_TOKEN_WORDS,), dtype=jnp.uint32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            valid=jnp.zeros((capacity,), dtype=jnp.bool_),
            bound_birth_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            lineage_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            prior_source_birth_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            prior_receipt_words=jnp.zeros((capacity, 4), dtype=jnp.uint32),
            return_prior=jnp.zeros((capacity,), dtype=jnp.float32),
            reacquisition_cost=jnp.zeros((capacity,), dtype=jnp.float32),
            cost_support_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            live_in_use=jnp.zeros((k,), dtype=jnp.bool_),
        )
        return self._with_content_token(state)

    def init(
        self,
        *,
        step_words: UInt[Array, " 2"],
        birth_words: UInt[Array, "max_contexts 2"],
        in_use: Bool[Array, " max_contexts"],
        return_priors: Float[Array, " max_contexts"],
    ) -> ProspectiveLineageRetentionState:
        """Bind genesis metadata to exact live semantic births."""

        k = self._config.max_contexts
        checked_step = _require_array(
            step_words, name="step_words", shape=(2,), dtype=jnp.uint32
        )
        checked_births = _require_array(
            birth_words, name="birth_words", shape=(k, 2), dtype=jnp.uint32
        )
        checked_in_use = _require_array(
            in_use, name="in_use", shape=(k,), dtype=jnp.bool_
        )
        checked_priors = _require_array(
            return_priors,
            name="return_priors",
            shape=(k,),
            dtype=jnp.float32,
        )
        valid_inputs = (
            jnp.all(jnp.isfinite(checked_priors))
            & jnp.all((checked_priors >= 0.0) & (checked_priors <= 1.0))
            & jnp.all(jnp.where(checked_in_use, _words_nonzero(checked_births), True))
            & jnp.all(
                jnp.where(
                    checked_in_use[:, None],
                    True,
                    checked_births == jnp.zeros((k, 2), dtype=jnp.uint32),
                )
            )
            & jnp.all(jnp.where(checked_in_use, True, checked_priors == 0.0))
            & _rows_unique(checked_births, checked_in_use)
        )
        if not bool(valid_inputs):
            raise ValueError("genesis births or return priors are invalid")
        capacity = self._config.metadata_capacity
        valid = jnp.concatenate(
            (checked_in_use, jnp.zeros((1,), dtype=jnp.bool_))
        )
        padded_births = jnp.concatenate(
            (checked_births, jnp.zeros((1, 2), dtype=jnp.uint32)), axis=0
        )
        padded_priors = jnp.concatenate(
            (checked_priors, jnp.zeros((1,), dtype=jnp.float32))
        )
        masked_births = jnp.where(valid[:, None], padded_births, 0)
        state = ProspectiveLineageRetentionState(
            config_token=self._config_token,
            content_token=jnp.zeros((_CONTENT_TOKEN_WORDS,), dtype=jnp.uint32),
            step_words=checked_step,
            revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            valid=valid,
            bound_birth_words=masked_births,
            lineage_words=masked_births,
            prior_source_birth_words=masked_births,
            prior_receipt_words=_prior_receipts(masked_births, padded_priors, valid),
            return_prior=jnp.where(valid, padded_priors, 0.0).astype(jnp.float32),
            reacquisition_cost=jnp.where(
                valid,
                jnp.float32(self._config.initial_reacquisition_cost),
                jnp.float32(0.0),
            ),
            cost_support_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            live_in_use=checked_in_use,
        )
        sealed = self._with_content_token(state)
        if not bool(self.state_valid(sealed)):
            raise RuntimeError("initialized prospective-retention state is invalid")
        return sealed

    def state_valid(self, state: ProspectiveLineageRetentionState) -> Array:
        """Validate complete fixed metadata and its integrity checksum."""

        self._require_state_contract(state)
        k = self._config.max_contexts
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_receipt = jnp.zeros((4,), dtype=jnp.uint32)
        expected_receipts = _prior_receipts(
            state.prior_source_birth_words,
            state.return_prior,
            state.valid,
        )
        live_valid_bound = jnp.all(state.valid[:k] == state.live_in_use)
        valid_rows = (
            _words_nonzero(state.bound_birth_words)
            & _words_nonzero(state.lineage_words)
            & _words_nonzero(state.prior_source_birth_words)
            & jnp.all(state.lineage_words == state.prior_source_birth_words, axis=1)
            & jnp.all(state.prior_receipt_words == expected_receipts, axis=1)
            & jnp.isfinite(state.return_prior)
            & (state.return_prior >= 0.0)
            & (state.return_prior <= 1.0)
            & jnp.isfinite(state.reacquisition_cost)
            & (state.reacquisition_cost >= 0.0)
            & (state.reacquisition_cost <= jnp.float32(self._config.max_abs_cost))
        )
        invalid_rows = (
            jnp.all(state.bound_birth_words == zero_words, axis=1)
            & jnp.all(state.lineage_words == zero_words, axis=1)
            & jnp.all(state.prior_source_birth_words == zero_words, axis=1)
            & jnp.all(state.prior_receipt_words == zero_receipt, axis=1)
            & (state.return_prior == 0.0)
            & (state.reacquisition_cost == 0.0)
            & jnp.all(state.cost_support_words == zero_words, axis=1)
        )
        unsigned = cast(Any, state).replace(
            content_token=jnp.zeros((_CONTENT_TOKEN_WORDS,), dtype=jnp.uint32)
        )
        return (
            jnp.array_equal(state.config_token, self._config_token)
            & jnp.array_equal(state.content_token, _state_content_token(unsigned))
            & live_valid_bound
            & jnp.all(jnp.where(state.valid, valid_rows, invalid_rows))
            & _rows_unique(state.bound_birth_words, state.valid)
            & _rows_unique(state.lineage_words, state.valid)
        )

    def _require_preparation_inputs(
        self,
        source_birth_words: Any,
        source_in_use: Any,
        active_slot: Any,
        last_active_words: Any,
        route_protection: Any,
    ) -> tuple[Array, Array, Array, Array, Array]:
        k = self._config.max_contexts
        return (
            _require_array(
                source_birth_words,
                name="source_birth_words",
                shape=(k, 2),
                dtype=jnp.uint32,
            ),
            _require_array(
                source_in_use,
                name="source_in_use",
                shape=(k,),
                dtype=jnp.bool_,
            ),
            _require_array(active_slot, name="active_slot", shape=(), dtype=jnp.int32),
            _require_array(
                last_active_words,
                name="last_active_words",
                shape=(k, 2),
                dtype=jnp.uint32,
            ),
            _require_array(
                route_protection,
                name="route_protection",
                shape=(),
                dtype=jnp.bool_,
            ),
        )

    def prepare(
        self,
        state: ProspectiveLineageRetentionState,
        *,
        source_birth_words: UInt[Array, "max_contexts 2"],
        source_in_use: Bool[Array, " max_contexts"],
        active_slot: Int[Array, ""],
        last_active_words: UInt[Array, "max_contexts 2"],
        route_protection: Bool[Array, ""],
    ) -> ProspectiveLineageRetentionPreparation:
        """Prepare expected-value scores before any current outcome exists."""

        self._require_state_contract(state)
        births, in_use, active, recency, route = self._require_preparation_inputs(
            source_birth_words,
            source_in_use,
            active_slot,
            last_active_words,
            route_protection,
        )
        k = self._config.max_contexts
        source_state_valid = self.state_valid(state)
        safe_active = jnp.clip(active, 0, k - 1)
        live_count = jnp.sum(in_use.astype(jnp.int32))
        active_valid = jnp.where(
            live_count == 0,
            active == -1,
            (active >= 0) & (active < k) & in_use[safe_active],
        )
        birth_binding = jnp.all(
            jnp.where(
                in_use[:, None],
                births == state.bound_birth_words[:k],
                births == jnp.zeros((k, 2), dtype=jnp.uint32),
            )
        )
        recency_within_clock = (recency[:, 0] < state.step_words[0]) | (
            (recency[:, 0] == state.step_words[0])
            & (recency[:, 1] <= state.step_words[1])
        )
        recency_valid = jnp.all(
            jnp.where(
                in_use,
                recency_within_clock,
                jnp.all(recency == jnp.zeros((k, 2), dtype=jnp.uint32), axis=1),
            )
        )
        bank_full = jnp.all(in_use)
        eligible = (
            bank_full
            & in_use
            & (jnp.arange(k, dtype=jnp.int32) != safe_active)
        )
        has_eligible = jnp.any(eligible)
        eligible_valid = ~bank_full | has_eligible
        input_valid = (
            jnp.array_equal(in_use, state.live_in_use)
            & birth_binding
            & active_valid
            & recency_valid
            & eligible_valid
        )
        priors = state.return_prior[:k]
        costs = state.reacquisition_cost[:k]
        expected_scores = (priors * costs).astype(jnp.float32)
        minimax_scores = costs.astype(jnp.float32)
        expected_victim = _select_score_victim(expected_scores, eligible, recency)
        minimax_victim = _select_score_victim(minimax_scores, eligible, recency)
        expected_worst = jnp.where(
            has_eligible,
            costs[expected_victim],
            jnp.float32(0.0),
        )
        minimax_worst = jnp.where(
            has_eligible,
            costs[minimax_victim],
            jnp.float32(0.0),
        )
        guardrail_passed = expected_worst <= (
            minimax_worst + jnp.float32(self._config.minimax_guardrail_slack)
        )
        selected = jnp.where(
            has_eligible,
            jnp.where(guardrail_passed, expected_victim, minimax_victim),
            jnp.int32(-1),
        )
        raw = jnp.where(guardrail_passed, expected_scores, minimax_scores)
        raw = jnp.where(in_use, raw, jnp.float32(0.0)).astype(jnp.float32)
        valid = source_state_valid & input_valid
        zero_scores = jnp.zeros((k,), dtype=jnp.float32)
        return ProspectiveLineageRetentionPreparation(
            source_config_token=state.config_token,
            source_content_token=state.content_token,
            preparation_receipt_words=_preparation_receipt(state.content_token, route),
            source_step_words=state.step_words,
            source_revision_words=state.revision_words,
            source_birth_words=births,
            source_lineage_words=state.lineage_words[:k],
            source_in_use=in_use,
            active_slot=active,
            last_active_words=recency,
            eligible_mask=jnp.where(valid, eligible, jnp.zeros_like(eligible)),
            expected_scores=jnp.where(valid, expected_scores, zero_scores),
            minimax_scores=jnp.where(valid, minimax_scores, zero_scores),
            raw_protection=jnp.where(valid, raw, zero_scores),
            protection_to_route=jnp.where(
                valid & route & bank_full,
                raw,
                zero_scores,
            ),
            expected_victim=jnp.where(
                valid & has_eligible,
                expected_victim,
                jnp.int32(-1),
            ),
            minimax_victim=jnp.where(
                valid & has_eligible,
                minimax_victim,
                jnp.int32(-1),
            ),
            selected_victim=jnp.where(valid, selected, jnp.int32(-1)),
            expected_victim_worst_cost=jnp.where(valid, expected_worst, jnp.float32(0.0)),
            minimax_victim_worst_cost=jnp.where(valid, minimax_worst, jnp.float32(0.0)),
            guardrail_passed=valid & has_eligible & guardrail_passed,
            guardrail_fallback_used=valid & has_eligible & ~guardrail_passed,
            route_protection=route,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            preparation_valid=valid,
        )

    def _require_preparation_contract(
        self,
        preparation: ProspectiveLineageRetentionPreparation,
    ) -> None:
        k = self._config.max_contexts
        _require_array(
            preparation.source_config_token,
            name="source_config_token",
            shape=(32,),
            dtype=jnp.uint8,
        )
        for name in ("source_content_token", "preparation_receipt_words"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(4,),
                dtype=jnp.uint32,
            )
        for name in ("source_step_words", "source_revision_words"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(2,),
                dtype=jnp.uint32,
            )
        for name in ("source_birth_words", "source_lineage_words", "last_active_words"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(k, 2),
                dtype=jnp.uint32,
            )
        for name in ("source_in_use", "eligible_mask"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(k,),
                dtype=jnp.bool_,
            )
        for name in (
            "expected_scores",
            "minimax_scores",
            "raw_protection",
            "protection_to_route",
        ):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(k,),
                dtype=jnp.float32,
            )
        for name in ("active_slot", "expected_victim", "minimax_victim", "selected_victim"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(),
                dtype=jnp.int32,
            )
        for name in ("expected_victim_worst_cost", "minimax_victim_worst_cost"):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(),
                dtype=jnp.float32,
            )
        for name in (
            "guardrail_passed",
            "guardrail_fallback_used",
            "route_protection",
            "source_state_valid",
            "input_valid",
            "preparation_valid",
        ):
            _require_array(
                getattr(preparation, name),
                name=name,
                shape=(),
                dtype=jnp.bool_,
            )

    def _require_event_contract(self, event: ProspectiveLineageRetentionEvent) -> None:
        k = self._config.max_contexts
        for name in ("source_step_words", "post_step_words", "transferred_lineage_words"):
            _require_array(getattr(event, name), name=name, shape=(2,), dtype=jnp.uint32)
        for name in ("source_birth_words", "post_birth_words"):
            _require_array(
                getattr(event, name), name=name, shape=(k, 2), dtype=jnp.uint32
            )
        for name in ("source_in_use", "post_in_use"):
            _require_array(getattr(event, name), name=name, shape=(k,), dtype=jnp.bool_)
        for name in (
            "allocated",
            "evicted",
            "lineage_transfer_confirmed",
            "context_update_applied",
        ):
            _require_array(getattr(event, name), name=name, shape=(), dtype=jnp.bool_)
        for name in ("target_slot", "transfer_slot"):
            _require_array(getattr(event, name), name=name, shape=(), dtype=jnp.int32)
        for name in ("newborn_return_prior", "archived_loss", "fresh_loss"):
            _require_array(getattr(event, name), name=name, shape=(), dtype=jnp.float32)

    def _preparation_authenticated(
        self,
        state: ProspectiveLineageRetentionState,
        preparation: ProspectiveLineageRetentionPreparation,
        event: ProspectiveLineageRetentionEvent,
    ) -> Array:
        canonical = self.prepare(
            state,
            source_birth_words=preparation.source_birth_words,
            source_in_use=preparation.source_in_use,
            active_slot=preparation.active_slot,
            last_active_words=preparation.last_active_words,
            route_protection=preparation.route_protection,
        )
        event_source_bound = (
            jnp.array_equal(event.source_step_words, preparation.source_step_words)
            & jnp.array_equal(event.source_birth_words, preparation.source_birth_words)
            & jnp.array_equal(event.source_in_use, preparation.source_in_use)
        )
        return (
            canonical.preparation_valid
            & _tree_arrays_equal(preparation, canonical)
            & event_source_bound
        )

    def _event_valid(
        self,
        state: ProspectiveLineageRetentionState,
        preparation: ProspectiveLineageRetentionPreparation,
        event: ProspectiveLineageRetentionEvent,
        proposed_step_words: Array,
    ) -> tuple[Array, Array, Array]:
        k = self._config.max_contexts
        archive_index = k
        safe_target = jnp.clip(event.target_slot, 0, k - 1)
        safe_transfer = jnp.clip(event.transfer_slot, 0, k - 1)
        target_valid = (event.target_slot >= 0) & (event.target_slot < k)
        transfer_valid = (event.transfer_slot >= 0) & (event.transfer_slot < k)
        other_slots = jnp.arange(k, dtype=jnp.int32) != safe_target
        clock_valid = (
            jnp.array_equal(event.source_step_words, state.step_words)
            & jnp.array_equal(event.post_step_words, proposed_step_words)
        )
        source_binding = (
            jnp.array_equal(event.source_birth_words, state.bound_birth_words[:k])
            & jnp.array_equal(event.source_in_use, state.live_in_use)
        )
        no_lifecycle = (
            ~event.allocated
            & ~event.evicted
            & (event.target_slot == -1)
            & (event.newborn_return_prior == 0.0)
            & jnp.array_equal(event.post_birth_words, event.source_birth_words)
            & jnp.array_equal(event.post_in_use, event.source_in_use)
        )
        newborn_prior_valid = (
            jnp.isfinite(event.newborn_return_prior)
            & (event.newborn_return_prior >= 0.0)
            & (event.newborn_return_prior <= 1.0)
        )
        allocation = (
            event.allocated
            & target_valid
            & (event.evicted == event.source_in_use[safe_target])
            & (~event.evicted | jnp.all(event.source_in_use))
            & event.post_in_use[safe_target]
            & _words_nonzero(event.post_birth_words[safe_target])
            & ~jnp.all(
                event.post_birth_words[safe_target]
                == event.source_birth_words[safe_target]
            )
            & jnp.all(
                jnp.where(
                    other_slots[:, None],
                    event.post_birth_words == event.source_birth_words,
                    jnp.asarray(True, dtype=jnp.bool_),
                )
            )
            & jnp.all(
                jnp.where(
                    other_slots,
                    event.post_in_use == event.source_in_use,
                    jnp.asarray(True, dtype=jnp.bool_),
                )
            )
            & _rows_unique(event.post_birth_words, event.post_in_use)
            & newborn_prior_valid
        )
        routed_eviction_binding_valid = (
            ~event.evicted
            | ~preparation.route_protection
            | (event.target_slot == preparation.selected_victim)
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        no_transfer = (
            ~event.lineage_transfer_confirmed
            & (event.transfer_slot == -1)
            & jnp.array_equal(event.transferred_lineage_words, zero_words)
            & (event.archived_loss == 0.0)
            & (event.fresh_loss == 0.0)
        )
        losses_valid = (
            jnp.isfinite(event.archived_loss)
            & jnp.isfinite(event.fresh_loss)
            & (event.archived_loss >= 0.0)
            & (event.fresh_loss >= 0.0)
            & (event.archived_loss <= jnp.float32(self._config.max_abs_cost))
            & (event.fresh_loss <= jnp.float32(self._config.max_abs_cost))
        )
        confirmation = (
            event.lineage_transfer_confirmed
            & ~event.allocated
            & transfer_valid
            & event.post_in_use[safe_transfer]
            & state.valid[archive_index]
            & jnp.array_equal(
                event.transferred_lineage_words,
                state.lineage_words[archive_index],
            )
            & ~jnp.array_equal(
                state.lineage_words[safe_transfer],
                state.lineage_words[archive_index],
            )
            & losses_valid
        )
        lifecycle_valid = no_lifecycle | allocation
        transfer_contract = no_transfer | confirmation
        valid = (
            event.context_update_applied
            & clock_valid
            & source_binding
            & lifecycle_valid
            & transfer_contract
            & routed_eviction_binding_valid
        )
        return valid, routed_eviction_binding_valid, confirmation

    def settle(
        self,
        state: ProspectiveLineageRetentionState,
        preparation: ProspectiveLineageRetentionPreparation,
        event: ProspectiveLineageRetentionEvent,
    ) -> ProspectiveLineageRetentionProposal:
        """Settle one completed event without changing its prior preparation."""

        self._require_state_contract(state)
        self._require_preparation_contract(preparation)
        self._require_event_contract(event)
        k = self._config.max_contexts
        archive_index = k
        source_state_valid = self.state_valid(state)
        authenticated = self._preparation_authenticated(state, preparation, event)
        proposed_step, step_capacity = _checked_words_increment(state.step_words)
        proposed_revision, revision_capacity = _checked_words_increment(state.revision_words)
        event_valid, routed_binding, confirmation_requested = self._event_valid(
            state,
            preparation,
            event,
            proposed_step,
        )
        safe_target = jnp.clip(event.target_slot, 0, k - 1)
        safe_transfer = jnp.clip(event.transfer_slot, 0, k - 1)

        valid = state.valid
        bound_births = state.bound_birth_words
        lineages = state.lineage_words
        prior_sources = state.prior_source_birth_words
        prior_receipts = state.prior_receipt_words
        priors = state.return_prior
        costs = state.reacquisition_cost
        support = state.cost_support_words

        archive_was_valid = state.valid[archive_index]
        copy_victim = event.allocated & event.evicted
        valid = valid.at[archive_index].set(jnp.where(copy_victim, True, valid[archive_index]))
        bound_births = bound_births.at[archive_index].set(
            jnp.where(
                copy_victim,
                state.bound_birth_words[safe_target],
                bound_births[archive_index],
            )
        )
        lineages = lineages.at[archive_index].set(
            jnp.where(copy_victim, state.lineage_words[safe_target], lineages[archive_index])
        )
        prior_sources = prior_sources.at[archive_index].set(
            jnp.where(
                copy_victim,
                state.prior_source_birth_words[safe_target],
                prior_sources[archive_index],
            )
        )
        prior_receipts = prior_receipts.at[archive_index].set(
            jnp.where(
                copy_victim,
                state.prior_receipt_words[safe_target],
                prior_receipts[archive_index],
            )
        )
        priors = priors.at[archive_index].set(
            jnp.where(copy_victim, state.return_prior[safe_target], priors[archive_index])
        )
        costs = costs.at[archive_index].set(
            jnp.where(copy_victim, state.reacquisition_cost[safe_target], costs[archive_index])
        )
        support = support.at[archive_index].set(
            jnp.where(copy_victim, state.cost_support_words[safe_target], support[archive_index])
        )

        new_birth = event.post_birth_words[safe_target]
        new_prior = event.newborn_return_prior
        new_receipt = _prior_receipts(
            new_birth[None, :],
            new_prior[None],
            jnp.ones((1,), dtype=jnp.bool_),
        )[0]
        valid = valid.at[safe_target].set(jnp.where(event.allocated, True, valid[safe_target]))
        bound_births = bound_births.at[safe_target].set(
            jnp.where(event.allocated, new_birth, bound_births[safe_target])
        )
        lineages = lineages.at[safe_target].set(
            jnp.where(event.allocated, new_birth, lineages[safe_target])
        )
        prior_sources = prior_sources.at[safe_target].set(
            jnp.where(event.allocated, new_birth, prior_sources[safe_target])
        )
        prior_receipts = prior_receipts.at[safe_target].set(
            jnp.where(event.allocated, new_receipt, prior_receipts[safe_target])
        )
        priors = priors.at[safe_target].set(
            jnp.where(event.allocated, new_prior, priors[safe_target])
        )
        costs = costs.at[safe_target].set(
            jnp.where(
                event.allocated,
                jnp.float32(self._config.initial_reacquisition_cost),
                costs[safe_target],
            )
        )
        support = support.at[safe_target].set(
            jnp.where(
                event.allocated,
                jnp.zeros((2,), dtype=jnp.uint32),
                support[safe_target],
            )
        )

        cost_observation = jnp.clip(
            event.fresh_loss - event.archived_loss,
            jnp.float32(0.0),
            jnp.float32(self._config.max_abs_cost),
        )
        old_archive_cost = costs[archive_index]
        updated_cost = (
            jnp.float32(self._config.cost_ema_decay) * old_archive_cost
            + jnp.float32(1.0 - self._config.cost_ema_decay) * cost_observation
        )
        incremented_support, support_capacity = _checked_words_increment(
            support[archive_index]
        )
        cost_support_capacity = ~confirmation_requested | support_capacity
        confirmation = confirmation_requested & support_capacity
        bound_births = bound_births.at[safe_transfer].set(
            jnp.where(
                confirmation,
                event.post_birth_words[safe_transfer],
                bound_births[safe_transfer],
            )
        )
        lineages = lineages.at[safe_transfer].set(
            jnp.where(confirmation, lineages[archive_index], lineages[safe_transfer])
        )
        prior_sources = prior_sources.at[safe_transfer].set(
            jnp.where(
                confirmation,
                prior_sources[archive_index],
                prior_sources[safe_transfer],
            )
        )
        prior_receipts = prior_receipts.at[safe_transfer].set(
            jnp.where(
                confirmation,
                prior_receipts[archive_index],
                prior_receipts[safe_transfer],
            )
        )
        priors = priors.at[safe_transfer].set(
            jnp.where(confirmation, priors[archive_index], priors[safe_transfer])
        )
        costs = costs.at[safe_transfer].set(
            jnp.where(confirmation, updated_cost, costs[safe_transfer])
        )
        support = support.at[safe_transfer].set(
            jnp.where(confirmation, incremented_support, support[safe_transfer])
        )

        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_receipt = jnp.zeros((4,), dtype=jnp.uint32)
        valid = valid.at[archive_index].set(jnp.where(confirmation, False, valid[archive_index]))
        bound_births = bound_births.at[archive_index].set(
            jnp.where(confirmation, zero_words, bound_births[archive_index])
        )
        lineages = lineages.at[archive_index].set(
            jnp.where(confirmation, zero_words, lineages[archive_index])
        )
        prior_sources = prior_sources.at[archive_index].set(
            jnp.where(confirmation, zero_words, prior_sources[archive_index])
        )
        prior_receipts = prior_receipts.at[archive_index].set(
            jnp.where(confirmation, zero_receipt, prior_receipts[archive_index])
        )
        priors = priors.at[archive_index].set(
            jnp.where(confirmation, jnp.float32(0.0), priors[archive_index])
        )
        costs = costs.at[archive_index].set(
            jnp.where(confirmation, jnp.float32(0.0), costs[archive_index])
        )
        support = support.at[archive_index].set(
            jnp.where(confirmation, zero_words, support[archive_index])
        )

        unsigned_candidate = ProspectiveLineageRetentionState(
            config_token=state.config_token,
            content_token=jnp.zeros((_CONTENT_TOKEN_WORDS,), dtype=jnp.uint32),
            step_words=proposed_step,
            revision_words=proposed_revision,
            valid=valid,
            bound_birth_words=bound_births,
            lineage_words=lineages,
            prior_source_birth_words=prior_sources,
            prior_receipt_words=prior_receipts,
            return_prior=priors,
            reacquisition_cost=costs,
            cost_support_words=support,
            live_in_use=event.post_in_use,
        )
        candidate = self._with_content_token(unsigned_candidate)
        candidate_valid = self.state_valid(candidate)
        capacity_available = step_capacity & revision_capacity & cost_support_capacity
        update_applied = (
            source_state_valid
            & authenticated
            & event_valid
            & capacity_available
            & candidate_valid
        )
        committed = cast(
            ProspectiveLineageRetentionState,
            jax.tree.map(
                lambda proposed, current: jnp.where(update_applied, proposed, current),
                candidate,
                state,
            ),
        )
        archive_created = copy_victim & ~archive_was_valid
        archive_replaced = copy_victim & archive_was_valid
        return ProspectiveLineageRetentionProposal(
            state=committed,
            source_state_valid=source_state_valid,
            preparation_authenticated=authenticated,
            event_valid=event_valid,
            step_capacity_available=step_capacity,
            revision_capacity_available=revision_capacity,
            cost_support_capacity_available=cost_support_capacity,
            candidate_state_valid=candidate_valid,
            routed_eviction_binding_valid=routed_binding,
            archive_created=update_applied & archive_created,
            archive_replaced=update_applied & archive_replaced,
            newborn_bound=update_applied & event.allocated,
            lineage_restored=update_applied & confirmation,
            prior_restored=update_applied & confirmation,
            cost_updated=update_applied & confirmation,
            cost_observation=jnp.where(
                update_applied & confirmation,
                cost_observation,
                jnp.float32(0.0),
            ),
            parameter_transplanted=jnp.asarray(False, dtype=jnp.bool_),
            update_applied=update_applied,
        )

    def resource_record(self) -> ProspectiveLineageRetentionResourceRecord:
        """Return exact fixed state bytes and explicit nonclaims."""

        k = self._config.max_contexts
        capacity = self._config.metadata_capacity
        formula = 64 + 57 * capacity + k
        measured = _tree_nbytes(self.init_empty())
        if measured != formula:
            raise AssertionError(f"state byte formula {formula} differs from measured {measured}")
        return ProspectiveLineageRetentionResourceRecord(
            schema=PROSPECTIVE_LINEAGE_RETENTION_RESOURCE_SCHEMA,
            max_contexts=k,
            metadata_capacity=capacity,
            archive_capacity=PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY,
            config_token_nbytes=_CONFIG_TOKEN_NBYTES,
            content_token_nbytes=4 * _CONTENT_TOKEN_WORDS,
            state_nbytes=formula,
            measured_state_nbytes=measured,
            replay_capacity=PROSPECTIVE_LINEAGE_RETENTION_REPLAY_CAPACITY,
            persistent_capacity_growth=0,
            parameter_transplant_allowed=False,
            external_provenance_claimed=(
                PROSPECTIVE_LINEAGE_RETENTION_EXTERNAL_PROVENANCE_CLAIMED
            ),
            host_event_binding_claimed=(
                PROSPECTIVE_LINEAGE_RETENTION_HOST_EVENT_BINDING_CLAIMED
            ),
        )

    def work_record(
        self,
        *,
        preparations: int,
        settlements: int,
    ) -> ProspectiveLineageRetentionWorkRecord:
        """Return fixed named logical work for a declared invocation schedule."""

        if type(preparations) is not int or preparations < 0:
            raise ValueError("preparations must be a nonnegative integer")
        if type(settlements) is not int or settlements < 0:
            raise ValueError("settlements must be a nonnegative integer")
        k = self._config.max_contexts
        total_score_preparations = preparations + settlements
        return ProspectiveLineageRetentionWorkRecord(
            schema=PROSPECTIVE_LINEAGE_RETENTION_WORK_SCHEMA,
            preparations=preparations,
            settlements=settlements,
            authentication_repreparations=settlements,
            total_score_preparations=total_score_preparations,
            score_products=total_score_preparations * k,
            expected_selection_cells=total_score_preparations * k,
            minimax_selection_cells=total_score_preparations * k,
            guardrail_comparisons=total_score_preparations,
            metadata_route_cells=settlements * self._config.metadata_capacity,
            settlement_proposals=settlements,
            random_draws=0,
            replay_updates=0,
            reset_callbacks=0,
            routed_unrouted_same_preparation_work=True,
            exhaustive_primitive_operation_count_claimed=False,
            compiled_flop_count_claimed=False,
        )


def measure_prospective_lineage_retention_state_nbytes(
    state: ProspectiveLineageRetentionState,
) -> int:
    """Measure every persistent JAX-array leaf in one sidecar."""

    return _tree_nbytes(state)


def save_prospective_lineage_retention_checkpoint(
    mechanism: ProspectiveLineageRetention,
    state: ProspectiveLineageRetentionState,
    path: str | Path,
) -> None:
    """Save one config-bound mechanism state."""

    if not bool(mechanism.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid prospective-retention state")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA,
            "config": mechanism.to_config(),
        },
    )


def load_prospective_lineage_retention_checkpoint(
    path: str | Path,
) -> tuple[ProspectiveLineageRetention, ProspectiveLineageRetentionState]:
    """Load and validate one config-bound mechanism state."""

    metadata = load_checkpoint_metadata(path)
    if set(metadata) != {"schema", "config"}:
        raise ValueError("prospective-retention checkpoint metadata fields do not match")
    if metadata["schema"] != PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA:
        raise ValueError("prospective-retention checkpoint schema is unsupported")
    config_payload = metadata["config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("prospective-retention checkpoint config is not a mapping")
    mechanism = ProspectiveLineageRetention.from_config(config_payload)
    loaded, restored_metadata = load_checkpoint(mechanism.init_empty(), path)
    if restored_metadata != metadata:
        raise ValueError("prospective-retention checkpoint metadata changed during load")
    state = cast(ProspectiveLineageRetentionState, loaded)
    if not bool(mechanism.state_valid(state)):
        raise ValueError("prospective-retention checkpoint state is invalid")
    return mechanism, state


__all__ = [
    "PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY",
    "PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_CONFIG_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_EVENT_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_EVIDENCE_LEVEL",
    "PROSPECTIVE_LINEAGE_RETENTION_EXTERNAL_PROVENANCE_CLAIMED",
    "PROSPECTIVE_LINEAGE_RETENTION_HOST_EVENT_BINDING_CLAIMED",
    "PROSPECTIVE_LINEAGE_RETENTION_PREPARATION_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_PROPOSAL_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_REPLAY_CAPACITY",
    "PROSPECTIVE_LINEAGE_RETENTION_RESOURCE_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROSPECTIVE_LINEAGE_RETENTION_STATE_SCHEMA",
    "PROSPECTIVE_LINEAGE_RETENTION_STATUS",
    "PROSPECTIVE_LINEAGE_RETENTION_WORK_SCHEMA",
    "ProspectiveLineageRetention",
    "ProspectiveLineageRetentionConfig",
    "ProspectiveLineageRetentionEvent",
    "ProspectiveLineageRetentionPreparation",
    "ProspectiveLineageRetentionProposal",
    "ProspectiveLineageRetentionResourceRecord",
    "ProspectiveLineageRetentionState",
    "ProspectiveLineageRetentionWorkRecord",
    "load_prospective_lineage_retention_checkpoint",
    "measure_prospective_lineage_retention_state_nbytes",
    "save_prospective_lineage_retention_checkpoint",
]
