# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded one-step, real-state-anchored Dyna for the ensemble lane.

This module closes only the L0 mechanism gap in WP4.6 of
``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``.  It composes a read-only
:class:`WorldModelEnsemble` snapshot with a caller-owned
:class:`MultiHeadMLPLearner` Q target.  It does not claim calibrated
uncertainty, useful planning, improved return, retention, or scientific
evidence.

The causal order for one real decision is explicit:

1. Before the current real transition is submitted to the world model, the
   caller records the encountered representation and executed primitive
   action against the exact decision-time model and control revision words.
2. After caller-owned real model/control updates commit, a planning call may
   bind only their exact current, monotonically newer revisions. It samples
   recorded real anchors with its own typed Threefry key; the current model
   snapshot is read but never updated by planning.
3. For each accepted anchor, the ensemble predicts reward, successor
   representation, and continuation.  Observed action support, residual-proxy
   readiness, epistemic disagreement, residual variance, finite values, and
   member termination agreement are checked before any control update.
4. The caller-owned target is formed *before* the update as
   ``reward_hat + continuation_hat * max_a Q(snapshot, next_representation)``.
   A single selected primitive Q head is then optimized.  Synthetic traces
   start at zero; real traces and hidden-unit utility/lifecycle diagnostics are
   restored exactly after the parameter/optimizer proposal.

The anchor representation, primitive action, decision identity, model
revision, and control revision carry a deterministic integrity tag.  This is
an accidental-corruption/tamper detector, not a cryptographic authenticator.
Observed action counts are only a bounded support veto; they are not calibrated
state-action density or an OOD guarantee.

Planning state, RNG, and clocks are disjoint from the model, its bootstrap
keys and signal calibrator, any actor, state builder, safety envelope, or
environment.  The checkpoint stores only this planner-owned state.  Model and
control states remain caller-owned and must be checkpointed by their owners.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleState,
)

ONE_STEP_DYNA_CONFIG_SCHEMA = "alberta.real-state-one-step-dyna.config.v1"
ONE_STEP_DYNA_CHECKPOINT_SCHEMA = "alberta.real-state-one-step-dyna.checkpoint.v1"
ONE_STEP_DYNA_MECHANISM_STATUS = "l0-mechanism-only-not-assessed"
ONE_STEP_DYNA_SCIENTIFIC_PROMOTION_ALLOWED = False
ONE_STEP_DYNA_EVIDENCE_LEVEL = "L0"

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_MAX_BACKUP_BUDGET = 4_096
_MAX_ANCHOR_CAPACITY = 4_096
_MAX_DIAGNOSTIC_SLOTS = 262_144
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_TAG_SALT = 0x44594E41


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return value


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must be finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not underflow in float32")
    if narrowed < minimum or (strictly_positive and narrowed == 0.0):
        comparator = "positive" if strictly_positive else f">= {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return narrowed


def _typed_threefry_key(value: object, *, name: str) -> Array:
    try:
        key = jnp.asarray(value)
        implementation = str(jr.key_impl(cast(Any, value)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key") from exc
    if (
        key.shape != ()
        or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        or implementation != "threefry2x32"
        or jr.key_data(cast(Any, value)).shape != (2,)
        or jr.key_data(cast(Any, value)).dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key")
    return key


def _array_has_contract(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and cast(Any, value).shape == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _tree_static_signature(tree: object) -> tuple[object, tuple[tuple[object, ...], ...]]:
    leaves, structure = jax.tree.flatten(tree)
    signatures: list[tuple[object, ...]] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            signatures.append((array.shape, "typed-prng", str(jr.key_impl(leaf))))
        else:
            signatures.append((array.shape, np.dtype(array.dtype).str))
    return structure, tuple(signatures)


def _tree_is_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _tree_exactly_equal(left: object, right: object) -> Bool[Array, ""]:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if (
        cast(Any, left_structure) != cast(Any, right_structure)
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_leaf)
        if jax.dtypes.issubdtype(right_array.dtype, jax.dtypes.prng_key):
            right_array = jr.key_data(right_leaf)
        equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def _words_nonzero(words: Array) -> Bool[Array, ""]:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_less(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _checked_words_add_small(words: Array, amount: Array | int) -> tuple[Array, Array]:
    increment = jnp.asarray(amount, dtype=jnp.uint32)
    low = words[1] + increment
    carry = (low < words[1]).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _checked_words_add(left: Array, right: Array) -> tuple[Array, Array]:
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_high = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_carry = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    return jnp.stack((high, low)), ~(overflow_high | overflow_carry)


def _words_leq_limit(words: Array, limit: int) -> Bool[Array, ""]:
    high = jnp.asarray((limit >> 32) & _UINT32_MAX, dtype=jnp.uint32)
    low = jnp.asarray(limit & _UINT32_MAX, dtype=jnp.uint32)
    limit_words = jnp.stack((high, low))
    return _words_less_equal(words, limit_words)


def _words_times_small(words: Array, multiplier: int) -> tuple[Array, Array]:
    """Multiply a word pair by a <=4096 scalar without requiring JAX x64."""

    multiplier_u = jnp.asarray(multiplier, dtype=jnp.uint32)
    low_lo = words[1] & jnp.asarray(0xFFFF, dtype=jnp.uint32)
    low_hi = words[1] >> jnp.asarray(16, dtype=jnp.uint32)
    p0 = low_lo * multiplier_u
    p1 = low_hi * multiplier_u
    shifted = (p1 & jnp.asarray(0xFFFF, dtype=jnp.uint32)) << jnp.asarray(
        16, dtype=jnp.uint32
    )
    result_low = p0 + shifted
    carry = (result_low < p0).astype(jnp.uint32)
    result_high = (
        words[0] * multiplier_u
        + (p1 >> jnp.asarray(16, dtype=jnp.uint32))
        + carry
    )
    # Configured plan calls are int32-bounded, so a nonzero input high word is
    # already invalid.  Keep the general overflow verdict explicit anyway.
    high_product_overflow = (words[0] != 0) & (
        (result_high // multiplier_u) != words[0]
    )
    return jnp.stack((result_high, result_low)), ~high_product_overflow


def _words_mod_small(words: Array, modulus: int) -> Int[Array, ""]:
    modulus_u = jnp.asarray(modulus, dtype=jnp.uint32)
    two32_mod = jnp.asarray((1 << 32) % modulus, dtype=jnp.uint32)
    high_term = (words[0] % modulus_u) * two32_mod
    remainder = (high_term + (words[1] % modulus_u)) % modulus_u
    return remainder.astype(jnp.int32)


def _saturating_size(words: Array, capacity: int) -> Int[Array, ""]:
    reached = (words[0] != 0) | (
        words[1] >= jnp.asarray(capacity, dtype=jnp.uint32)
    )
    return jnp.where(
        reached,
        jnp.asarray(capacity, dtype=jnp.int32),
        words[1].astype(jnp.int32),
    )


def _config_digest(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class OneStepDynaConfig:
    """Static memory, work, guard, and lifetime limits.

    Thresholds are user declarations and are not empirically calibrated by
    this module.  ``min_action_support`` counts retained real anchors for the
    primitive action, not local state-action density.
    """

    anchor_capacity: int = 64
    backup_budget: int = 1
    min_action_support: int = 1
    max_epistemic_disagreement: float = 1.0
    max_residual_variance: float = 1.0
    require_residual_proxy_ready: bool = True
    terminal_discount_threshold: float = 0.0
    require_termination_agreement: bool = True
    max_anchor_magnitude: float = 1_000.0
    max_abs_control_target: float = 1_000_000.0
    max_anchor_records: int = _INT32_MAX
    max_planning_calls: int = _INT32_MAX
    max_planned_backups: int = _INT32_MAX

    def __post_init__(self) -> None:
        _positive_int(
            self.anchor_capacity,
            name="anchor_capacity",
            maximum=_MAX_ANCHOR_CAPACITY,
        )
        _positive_int(
            self.backup_budget,
            name="backup_budget",
            maximum=_MAX_BACKUP_BUDGET,
        )
        _positive_int(
            self.min_action_support,
            name="min_action_support",
            maximum=self.anchor_capacity,
        )
        _positive_int(
            self.max_anchor_records,
            name="max_anchor_records",
            maximum=_INT32_MAX,
        )
        _positive_int(
            self.max_planning_calls,
            name="max_planning_calls",
            maximum=_INT32_MAX,
        )
        _positive_int(
            self.max_planned_backups,
            name="max_planned_backups",
            maximum=_INT32_MAX,
        )
        if self.max_planned_backups > self.max_planning_calls * self.backup_budget:
            raise ValueError(
                "max_planned_backups cannot exceed max_planning_calls * backup_budget"
            )
        if type(self.require_residual_proxy_ready) is not bool:
            raise ValueError("require_residual_proxy_ready must be a strict boolean")
        if type(self.require_termination_agreement) is not bool:
            raise ValueError("require_termination_agreement must be a strict boolean")
        for name, minimum, positive in (
            ("max_epistemic_disagreement", 0.0, False),
            ("max_residual_variance", 0.0, False),
            ("terminal_discount_threshold", 0.0, False),
            ("max_anchor_magnitude", _FLOAT32_TINY, True),
            ("max_abs_control_target", _FLOAT32_TINY, True),
        ):
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    strictly_positive=positive,
                ),
            )
        if self.backup_budget * self.anchor_capacity > _MAX_DIAGNOSTIC_SLOTS:
            raise ValueError(
                "backup_budget * anchor_capacity exceeds the diagnostic slot ceiling"
            )

    @property
    def max_planning_attempts(self) -> int:
        """Exact lifetime attempted-candidate ceiling."""

        return self.backup_budget * self.max_planning_calls

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible L0 configuration."""

        return {
            "schema": ONE_STEP_DYNA_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": ONE_STEP_DYNA_MECHANISM_STATUS,
            "evidence_level": ONE_STEP_DYNA_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": (
                ONE_STEP_DYNA_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "anchor_capacity": self.anchor_capacity,
            "backup_budget": self.backup_budget,
            "min_action_support": self.min_action_support,
            "max_epistemic_disagreement": self.max_epistemic_disagreement,
            "max_residual_variance": self.max_residual_variance,
            "require_residual_proxy_ready": self.require_residual_proxy_ready,
            "terminal_discount_threshold": self.terminal_discount_threshold,
            "require_termination_agreement": self.require_termination_agreement,
            "max_anchor_magnitude": self.max_anchor_magnitude,
            "max_abs_control_target": self.max_abs_control_target,
            "max_anchor_records": self.max_anchor_records,
            "max_planning_calls": self.max_planning_calls,
            "max_planned_backups": self.max_planned_backups,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> OneStepDynaConfig:
        """Strictly restore the sole v1 schema; no migration is needed."""

        payload = dict(config)
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("one-step Dyna config fields do not match v1")
        if payload.pop("schema") != ONE_STEP_DYNA_CONFIG_SCHEMA:
            raise ValueError("unsupported one-step Dyna config schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected one-step Dyna config type")
        if payload.pop("mechanism_status") != ONE_STEP_DYNA_MECHANISM_STATUS:
            raise ValueError("one-step Dyna must remain mechanism-only")
        if payload.pop("evidence_level") != ONE_STEP_DYNA_EVIDENCE_LEVEL:
            raise ValueError("one-step Dyna evidence level must remain L0")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("one-step Dyna cannot claim scientific promotion")
        for name in (
            "anchor_capacity",
            "backup_budget",
            "min_action_support",
            "max_anchor_records",
            "max_planning_calls",
            "max_planned_backups",
        ):
            if type(payload[name]) is not int:
                raise ValueError(f"serialized {name} must be a JSON integer")
        for name in (
            "max_epistemic_disagreement",
            "max_residual_variance",
            "terminal_discount_threshold",
            "max_anchor_magnitude",
            "max_abs_control_target",
        ):
            if type(payload[name]) is not float:
                raise ValueError(f"serialized {name} must be a canonical JSON float")
        for name in (
            "require_residual_proxy_ready",
            "require_termination_agreement",
        ):
            if type(payload[name]) is not bool:
                raise ValueError(f"serialized {name} must be a JSON boolean")
        restored = cls(**cast(dict[str, Any], payload))
        if restored.to_config() != dict(config):
            raise ValueError("one-step Dyna config is not canonical")
        return restored


@dataclasses.dataclass(frozen=True)
class OneStepDynaResourceBudget:
    """Exact module-owned persistent bytes and hard per-call work maxima."""

    persistent_bytes_scope: str
    diagnostic_bytes_scope: str
    temporary_bytes_scope: str
    observation_dim: int
    n_primitive_actions: int
    ensemble_size: int
    anchor_capacity: int
    backup_budget: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_uint32_scalars: int
    persistent_bool_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    planning_prng_keys: int
    planning_prng_uint32_scalars: int
    record_diagnostics_scalars: int
    record_diagnostics_bytes: int
    plan_diagnostics_scalars: int
    plan_diagnostics_bytes: int
    max_ensemble_prediction_calls_per_call: int
    max_member_model_predictions_per_call: int
    max_control_target_forward_calls_per_call: int
    max_control_update_forward_calls_per_call: int
    max_control_forward_calls_per_call: int
    max_control_backward_calls_per_call: int
    max_control_updates_per_call: int
    max_planning_rng_splits_per_call: int
    max_planning_rng_draws_per_call: int
    max_anchor_records: int
    max_planning_calls: int
    max_planning_attempts: int
    max_planned_backups: int
    model_state_owned: int
    control_state_owned: int
    replay_capacity: int

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class OneStepDynaAuthority:
    """Caller assertion of the representation/model/control snapshot."""

    representation_revision_words: UInt[Array, " 2"]
    model_revision_words: UInt[Array, " 2"]
    control_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RealStateDynaAnchor:
    """One encountered representation and executed primitive action."""

    observation: Float[Array, " observation_dim"]
    primitive_action: Int[Array, ""]
    decision_id_words: UInt[Array, " 2"]
    authority: OneStepDynaAuthority


@chex.dataclass(frozen=True)
class OneStepDynaState:
    """Complete bounded planner-owned state; child learner states are absent."""

    anchor_observations: Float[Array, "anchor_capacity observation_dim"]
    anchor_actions: Int[Array, " anchor_capacity"]
    anchor_decision_id_words: UInt[Array, "anchor_capacity 2"]
    anchor_representation_revision_words: UInt[Array, "anchor_capacity 2"]
    anchor_model_revision_words: UInt[Array, "anchor_capacity 2"]
    anchor_control_revision_words: UInt[Array, "anchor_capacity 2"]
    anchor_integrity_tags: UInt[Array, " anchor_capacity"]
    anchor_valid: Bool[Array, " anchor_capacity"]
    action_support_counts: Int[Array, " n_primitive_actions"]
    size: Int[Array, ""]
    write_index: Int[Array, ""]
    bound_representation_revision_words: UInt[Array, " 2"]
    bound_model_revision_words: UInt[Array, " 2"]
    bound_control_revision_words: UInt[Array, " 2"]
    last_decision_id_words: UInt[Array, " 2"]
    record_count_words: UInt[Array, " 2"]
    planning_call_count_words: UInt[Array, " 2"]
    planning_attempt_count_words: UInt[Array, " 2"]
    planned_backup_count_words: UInt[Array, " 2"]
    rejected_backup_count_words: UInt[Array, " 2"]
    planning_key: Array


@chex.dataclass(frozen=True)
class OneStepDynaRecordDiagnostics:
    """Atomic real-anchor recording verdict."""

    state_static_contract_valid: Bool[Array, ""]
    model_state_static_contract_valid: Bool[Array, ""]
    control_state_static_contract_valid: Bool[Array, ""]
    anchor_static_contract_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    model_state_valid: Bool[Array, ""]
    control_state_valid: Bool[Array, ""]
    anchor_values_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    revisions_monotonic: Bool[Array, ""]
    decision_identity_valid: Bool[Array, ""]
    record_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    written_index: Int[Array, ""]
    integrity_tag: UInt[Array, ""]
    pre_record_count_words: UInt[Array, " 2"]
    post_record_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OneStepDynaRecordResult:
    """Planner state after one attempted real-anchor record."""

    state: OneStepDynaState
    diagnostics: OneStepDynaRecordDiagnostics


@chex.dataclass(frozen=True)
class OneStepDynaDiagnostics:
    """Fixed-budget guard, target, child-update, and clock audit."""

    state_static_contract_valid: Bool[Array, ""]
    model_state_static_contract_valid: Bool[Array, ""]
    control_state_static_contract_valid: Bool[Array, ""]
    authority_static_contract_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    model_state_valid: Bool[Array, ""]
    control_state_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    anchors_available: Bool[Array, ""]
    planning_call_capacity_available: Bool[Array, ""]
    planning_attempt_capacity_available: Bool[Array, ""]
    planned_backup_capacity_available: Bool[Array, ""]
    selected_anchor_indices: Int[Array, " backup_budget"]
    selected_actions: Int[Array, " backup_budget"]
    selected_decision_id_words: UInt[Array, "backup_budget 2"]
    selected_anchor_model_revision_words: UInt[Array, "backup_budget 2"]
    action_support_counts: Int[Array, " backup_budget"]
    predicted_next_observations: Float[Array, "backup_budget observation_dim"]
    predicted_rewards: Float[Array, " backup_budget"]
    predicted_continuations: Float[Array, " backup_budget"]
    epistemic_disagreements: Float[Array, " backup_budget"]
    residual_variance_maxima: Float[Array, " backup_budget"]
    successor_values: Float[Array, " backup_budget"]
    control_targets: Float[Array, " backup_budget"]
    td_errors: Float[Array, " backup_budget"]
    anchor_identity_valid: Bool[Array, " backup_budget"]
    support_valid: Bool[Array, " backup_budget"]
    model_prediction_valid: Bool[Array, " backup_budget"]
    residual_proxy_ready: Bool[Array, " backup_budget"]
    epistemic_valid: Bool[Array, " backup_budget"]
    residual_variance_valid: Bool[Array, " backup_budget"]
    continuation_valid: Bool[Array, " backup_budget"]
    termination_agreement: Bool[Array, " backup_budget"]
    finite_target_valid: Bool[Array, " backup_budget"]
    per_backup_capacity_available: Bool[Array, " backup_budget"]
    guard_passed: Bool[Array, " backup_budget"]
    child_update_applied: Bool[Array, " backup_budget"]
    child_transaction_authenticated: Bool[Array, " backup_budget"]
    trace_and_utility_isolation_preserved: Bool[Array, " backup_budget"]
    applied: Bool[Array, " backup_budget"]
    applied_count: Int[Array, ""]
    transaction_applied: Bool[Array, ""]
    pre_model_revision_words: UInt[Array, " 2"]
    post_model_revision_words: UInt[Array, " 2"]
    pre_control_revision_words: UInt[Array, " 2"]
    post_control_revision_words: UInt[Array, " 2"]
    pre_planning_call_count_words: UInt[Array, " 2"]
    post_planning_call_count_words: UInt[Array, " 2"]
    pre_planning_attempt_count_words: UInt[Array, " 2"]
    post_planning_attempt_count_words: UInt[Array, " 2"]
    pre_planned_backup_count_words: UInt[Array, " 2"]
    post_planned_backup_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OneStepDynaResult:
    """Planner and caller-owned control target after one planning call."""

    state: OneStepDynaState
    control_state: MultiHeadMLPState
    diagnostics: OneStepDynaDiagnostics


class RealStateOneStepDyna:
    """One-step Dyna composer over an ensemble and caller-owned Q target."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble,
        control_learner: MultiHeadMLPLearner,
        config: OneStepDynaConfig | None = None,
    ) -> None:
        self._ensemble = ensemble
        self._control = control_learner
        self._config = config or OneStepDynaConfig()
        model_cfg = ensemble.config.model
        if control_learner.n_heads != model_cfg.n_actions:
            raise ValueError(
                "control learner heads must equal ensemble primitive actions"
            )
        if control_learner.normalizer is not None:
            raise ValueError(
                "Dyna control target must not own an online normalizer; encode "
                "and normalize real representations in the caller-owned state builder"
            )
        if self._config.terminal_discount_threshold > model_cfg.gamma:
            raise ValueError(
                "terminal_discount_threshold cannot exceed the model discount bound"
            )
        self._reference_model_state = ensemble.init(
            jr.key(0, impl="threefry2x32")
        )
        self._reference_control_state = control_learner.init(
            model_cfg.observation_dim,
            jr.key(1, impl="threefry2x32"),
        )
        self._model_signature = _tree_static_signature(self._reference_model_state)
        self._control_signature = _tree_static_signature(
            self._reference_control_state
        )

    @property
    def config(self) -> OneStepDynaConfig:
        """Return the immutable planner configuration."""

        return self._config

    @property
    def ensemble(self) -> WorldModelEnsemble:
        """Return the read-only world-model implementation."""

        return self._ensemble

    @property
    def control_learner(self) -> MultiHeadMLPLearner:
        """Return the caller-owned control-target implementation."""

        return self._control

    @property
    def observation_dim(self) -> int:
        return self._ensemble.config.model.observation_dim

    @property
    def n_primitive_actions(self) -> int:
        return self._ensemble.config.model.n_actions

    def to_config(self) -> dict[str, object]:
        """Serialize child constructions and this L0 composition contract."""

        return {
            "type": type(self).__name__,
            "planner": self._config.to_config(),
            "ensemble": self._ensemble.to_config(),
            "control_learner": self._control.to_config(),
            "child_states_owned": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> RealStateOneStepDyna:
        """Strictly reconstruct compatible child implementations and planner."""

        payload = dict(config)
        expected = {
            "type",
            "planner",
            "ensemble",
            "control_learner",
            "child_states_owned",
            "scientific_promotion_allowed",
        }
        if set(payload) != expected:
            raise ValueError("one-step Dyna construction fields do not match v1")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected one-step Dyna construction type")
        if payload.pop("child_states_owned") is not False:
            raise ValueError("model and control states must remain caller-owned")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("one-step Dyna cannot claim scientific promotion")
        planner_payload = payload.pop("planner")
        ensemble_payload = payload.pop("ensemble")
        control_payload = payload.pop("control_learner")
        if not all(
            isinstance(item, Mapping)
            for item in (planner_payload, ensemble_payload, control_payload)
        ):
            raise ValueError("one-step Dyna child configurations must be mappings")
        instance = cls(
            WorldModelEnsemble.from_config(dict(cast(Mapping[str, Any], ensemble_payload))),
            MultiHeadMLPLearner.from_config(
                dict(cast(Mapping[str, Any], control_payload))
            ),
            OneStepDynaConfig.from_config(
                cast(Mapping[str, object], planner_payload)
            ),
        )
        if instance.to_config() != dict(config):
            raise ValueError("one-step Dyna construction is not canonical")
        return instance

    def _model_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, WorldModelEnsembleState)
            and _tree_static_signature(state) == self._model_signature
        )

    def _control_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, MultiHeadMLPState)
            and _tree_static_signature(state) == self._control_signature
        )

    def _model_valid(self, state: WorldModelEnsembleState) -> Array:
        return self._ensemble.state_valid(state)

    def _control_valid(self, state: MultiHeadMLPState) -> Array:
        status = self._control._counter_status(state)
        integer_valid = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in jax.tree.leaves(state):
            array = jnp.asarray(leaf)
            if jnp.issubdtype(array.dtype, jnp.signedinteger):
                integer_valid = integer_valid & jnp.all(array >= 0)
        return (
            _tree_is_finite(state)
            & integer_valid
            & status.lifetime_counter_valid
            & status.normalizer_counter_aligned
            & (state.normalizer_state is None)
        )

    def _state_static_valid(self, state: object) -> bool:
        if not isinstance(state, OneStepDynaState):
            return False
        cfg = self._config
        capacity = cfg.anchor_capacity
        observation_dim = self.observation_dim
        n_actions = self.n_primitive_actions
        contracts = (
            (state.anchor_observations, (capacity, observation_dim), jnp.float32),
            (state.anchor_actions, (capacity,), jnp.int32),
            (state.anchor_decision_id_words, (capacity, 2), jnp.uint32),
            (
                state.anchor_representation_revision_words,
                (capacity, 2),
                jnp.uint32,
            ),
            (state.anchor_model_revision_words, (capacity, 2), jnp.uint32),
            (state.anchor_control_revision_words, (capacity, 2), jnp.uint32),
            (state.anchor_integrity_tags, (capacity,), jnp.uint32),
            (state.anchor_valid, (capacity,), jnp.bool_),
            (state.action_support_counts, (n_actions,), jnp.int32),
            (state.size, (), jnp.int32),
            (state.write_index, (), jnp.int32),
            (state.bound_representation_revision_words, (2,), jnp.uint32),
            (state.bound_model_revision_words, (2,), jnp.uint32),
            (state.bound_control_revision_words, (2,), jnp.uint32),
            (state.last_decision_id_words, (2,), jnp.uint32),
            (state.record_count_words, (2,), jnp.uint32),
            (state.planning_call_count_words, (2,), jnp.uint32),
            (state.planning_attempt_count_words, (2,), jnp.uint32),
            (state.planned_backup_count_words, (2,), jnp.uint32),
            (state.rejected_backup_count_words, (2,), jnp.uint32),
        )
        if not all(
            _array_has_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in contracts
        ):
            return False
        try:
            _typed_threefry_key(state.planning_key, name="state.planning_key")
        except TypeError:
            return False
        return True

    def _authority_static_valid(self, authority: object) -> bool:
        return isinstance(authority, OneStepDynaAuthority) and all(
            _array_has_contract(value, shape=(2,), dtype=jnp.uint32)
            for value in (
                authority.representation_revision_words,
                authority.model_revision_words,
                authority.control_revision_words,
            )
        )

    def _anchor_static_valid(self, anchor: object) -> bool:
        return (
            isinstance(anchor, RealStateDynaAnchor)
            and _array_has_contract(
                anchor.observation,
                shape=(self.observation_dim,),
                dtype=jnp.float32,
            )
            and _array_has_contract(
                anchor.primitive_action,
                shape=(),
                dtype=jnp.int32,
            )
            and _array_has_contract(
                anchor.decision_id_words,
                shape=(2,),
                dtype=jnp.uint32,
            )
            and self._authority_static_valid(anchor.authority)
        )

    def _anchor_tag(
        self,
        observation: Array,
        action: Array,
        decision_words: Array,
        representation_words: Array,
        model_words: Array,
        control_words: Array,
    ) -> UInt[Array, ""]:
        observation_words = jax.lax.bitcast_convert_type(
            observation,
            jnp.uint32,
        )
        words = jnp.concatenate(
            (
                jnp.asarray([_TAG_SALT], dtype=jnp.uint32),
                decision_words,
                representation_words,
                model_words,
                control_words,
                jnp.reshape(action.astype(jnp.uint32), (1,)),
                observation_words,
            )
        )

        def mix(tag: Array, word: Array) -> Array:
            return (tag ^ word) * jnp.asarray(_TAG_PRIME, dtype=jnp.uint32)

        tag = jax.lax.fori_loop(
            0,
            words.shape[0],
            lambda index, carry: mix(carry, words[index]),
            jnp.asarray(_TAG_OFFSET, dtype=jnp.uint32),
        )
        return jnp.where(
            tag == jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(_TAG_SALT, dtype=jnp.uint32),
            tag,
        )

    def _state_valid(self, state: OneStepDynaState) -> Bool[Array, ""]:
        cfg = self._config
        capacity = cfg.anchor_capacity
        positions = jnp.arange(capacity, dtype=jnp.int32)
        expected_size = _saturating_size(state.record_count_words, capacity)
        expected_write = _words_mod_small(state.record_count_words, capacity)
        expected_valid = positions < expected_size
        safe_actions = jnp.clip(state.anchor_actions, 0, self.n_primitive_actions - 1)
        recomputed_support = jnp.sum(
            jax.nn.one_hot(
                safe_actions,
                self.n_primitive_actions,
                dtype=jnp.int32,
            )
            * state.anchor_valid[:, None].astype(jnp.int32),
            axis=0,
        )
        bound = jnp.asarray(cfg.max_anchor_magnitude, dtype=jnp.float32)
        observation_valid = jnp.all(
            jnp.where(
                state.anchor_valid[:, None],
                jnp.isfinite(state.anchor_observations)
                & (jnp.abs(state.anchor_observations) <= bound),
                state.anchor_observations == 0.0,
            )
        )
        action_valid = jnp.all(
            jnp.where(
                state.anchor_valid,
                (state.anchor_actions >= 0)
                & (state.anchor_actions < self.n_primitive_actions),
                state.anchor_actions == -1,
            )
        )
        computed_tags = jax.vmap(self._anchor_tag)(
            state.anchor_observations,
            safe_actions,
            state.anchor_decision_id_words,
            state.anchor_representation_revision_words,
            state.anchor_model_revision_words,
            state.anchor_control_revision_words,
        )
        tags_valid = jnp.all(
            jnp.where(
                state.anchor_valid,
                state.anchor_integrity_tags == computed_tags,
                state.anchor_integrity_tags == 0,
            )
        )
        unused_words_zero = jnp.all(
            jnp.where(
                state.anchor_valid[:, None],
                True,
                (state.anchor_decision_id_words == 0)
                & (state.anchor_representation_revision_words == 0)
                & (state.anchor_model_revision_words == 0)
                & (state.anchor_control_revision_words == 0),
            )
        )
        entry_revisions_valid = jnp.all(
            jnp.where(
                state.anchor_valid,
                jnp.all(
                    state.anchor_representation_revision_words
                    == state.bound_representation_revision_words[None, :],
                    axis=1,
                )
                & jax.vmap(_words_less_equal)(
                    state.anchor_model_revision_words,
                    jnp.broadcast_to(
                        state.bound_model_revision_words,
                        state.anchor_model_revision_words.shape,
                    ),
                )
                & jax.vmap(_words_less_equal)(
                    state.anchor_control_revision_words,
                    jnp.broadcast_to(
                        state.bound_control_revision_words,
                        state.anchor_control_revision_words.shape,
                    ),
                )
                & jax.vmap(_words_nonzero)(state.anchor_decision_id_words),
                True,
            )
        )
        oldest = jnp.where(expected_size == capacity, state.write_index, 0)
        chronological = (oldest + positions) % capacity
        ordered_decisions = state.anchor_decision_id_words[chronological]
        ordered_models = state.anchor_model_revision_words[chronological]
        ordered_controls = state.anchor_control_revision_words[chronological]
        if capacity > 1:
            pair_mask = positions[:-1] + 1 < expected_size
            decisions_increase = jnp.all(
                jnp.where(
                    pair_mask,
                    jax.vmap(_words_less)(
                        ordered_decisions[:-1], ordered_decisions[1:]
                    ),
                    True,
                )
            )
            models_monotonic = jnp.all(
                jnp.where(
                    pair_mask,
                    jax.vmap(_words_less_equal)(
                        ordered_models[:-1], ordered_models[1:]
                    ),
                    True,
                )
            )
            controls_monotonic = jnp.all(
                jnp.where(
                    pair_mask,
                    jax.vmap(_words_less_equal)(
                        ordered_controls[:-1], ordered_controls[1:]
                    ),
                    True,
                )
            )
        else:
            decisions_increase = jnp.asarray(True, dtype=jnp.bool_)
            models_monotonic = jnp.asarray(True, dtype=jnp.bool_)
            controls_monotonic = jnp.asarray(True, dtype=jnp.bool_)
        newest_index = chronological[jnp.maximum(expected_size - 1, 0)]
        newest_decision = state.anchor_decision_id_words[newest_index]
        last_decision_valid = jnp.where(
            expected_size > 0,
            jnp.all(newest_decision == state.last_decision_id_words),
            jnp.all(state.last_decision_id_words == 0),
        )
        attempts_sum, attempts_sum_valid = _checked_words_add(
            state.planned_backup_count_words,
            state.rejected_backup_count_words,
        )
        expected_attempts, multiplication_valid = _words_times_small(
            state.planning_call_count_words,
            cfg.backup_budget,
        )
        return (
            _words_nonzero(state.bound_representation_revision_words)
            & (state.size == expected_size)
            & (state.write_index == expected_write)
            & jnp.array_equal(state.anchor_valid, expected_valid)
            & observation_valid
            & action_valid
            & tags_valid
            & unused_words_zero
            & entry_revisions_valid
            & decisions_increase
            & models_monotonic
            & controls_monotonic
            & last_decision_valid
            & jnp.array_equal(state.action_support_counts, recomputed_support)
            & jnp.all(state.action_support_counts >= 0)
            & (jnp.sum(state.action_support_counts) == state.size)
            & _words_leq_limit(state.record_count_words, cfg.max_anchor_records)
            & _words_leq_limit(
                state.planning_call_count_words,
                cfg.max_planning_calls,
            )
            & _words_leq_limit(
                state.planning_attempt_count_words,
                cfg.max_planning_attempts,
            )
            & _words_leq_limit(
                state.planned_backup_count_words,
                cfg.max_planned_backups,
            )
            & attempts_sum_valid
            & multiplication_valid
            & jnp.array_equal(attempts_sum, state.planning_attempt_count_words)
            & jnp.array_equal(expected_attempts, state.planning_attempt_count_words)
        )

    def state_valid(self, state: OneStepDynaState) -> Bool[Array, ""]:
        """Return the dynamic planner-state verdict after static validation."""

        if not self._state_static_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        return self._state_valid(state)

    def validate_state(self, state: OneStepDynaState) -> None:
        """Raise on a malformed or dynamically inconsistent planner state."""

        if not self._state_static_valid(state):
            raise ValueError("one-step Dyna state has an invalid static contract")
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("one-step Dyna state is dynamically invalid")

    def _empty_state(
        self,
        key: Array,
        representation_revision_words: Array,
        model_revision_words: Array,
        control_revision_words: Array,
    ) -> OneStepDynaState:
        capacity = self._config.anchor_capacity
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return OneStepDynaState(
            anchor_observations=jnp.zeros(
                (capacity, self.observation_dim), dtype=jnp.float32
            ),
            anchor_actions=jnp.full((capacity,), -1, dtype=jnp.int32),
            anchor_decision_id_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            anchor_representation_revision_words=jnp.zeros(
                (capacity, 2), dtype=jnp.uint32
            ),
            anchor_model_revision_words=jnp.zeros(
                (capacity, 2), dtype=jnp.uint32
            ),
            anchor_control_revision_words=jnp.zeros(
                (capacity, 2), dtype=jnp.uint32
            ),
            anchor_integrity_tags=jnp.zeros((capacity,), dtype=jnp.uint32),
            anchor_valid=jnp.zeros((capacity,), dtype=jnp.bool_),
            action_support_counts=jnp.zeros(
                (self.n_primitive_actions,), dtype=jnp.int32
            ),
            size=jnp.asarray(0, dtype=jnp.int32),
            write_index=jnp.asarray(0, dtype=jnp.int32),
            bound_representation_revision_words=representation_revision_words,
            bound_model_revision_words=model_revision_words,
            bound_control_revision_words=control_revision_words,
            last_decision_id_words=zero_words,
            record_count_words=zero_words,
            planning_call_count_words=zero_words,
            planning_attempt_count_words=zero_words,
            planned_backup_count_words=zero_words,
            rejected_backup_count_words=zero_words,
            planning_key=key,
        )

    def init(
        self,
        key: Array,
        representation_revision_words: Array,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
    ) -> OneStepDynaState:
        """Bind an empty planner to exact external snapshot revisions."""

        checked_key = _typed_threefry_key(key, name="key")
        if not _array_has_contract(
            representation_revision_words,
            shape=(2,),
            dtype=jnp.uint32,
        ):
            raise ValueError("representation_revision_words must be uint32[2]")
        if not self._model_static_valid(model_state):
            raise ValueError("model_state has an incompatible static contract")
        if not self._control_static_valid(control_state):
            raise ValueError("control_state has an incompatible static contract")
        if not bool(jax.device_get(self._model_valid(model_state))):
            raise ValueError("cannot bind an invalid model state")
        if not bool(jax.device_get(self._control_valid(control_state))):
            raise ValueError("cannot bind an invalid control state")
        if not bool(jax.device_get(_words_nonzero(representation_revision_words))):
            raise ValueError("representation revision must be nonzero")
        state = self._empty_state(
            checked_key,
            representation_revision_words,
            model_state.event_count_words,
            control_state.step_words,
        )
        self.validate_state(state)
        return state

    def _unavailable_record_diagnostics(
        self,
        state: object,
        *,
        state_static: bool,
        model_static: bool,
        control_static: bool,
        anchor_static: bool,
    ) -> OneStepDynaRecordDiagnostics:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        pre_words = (
            state.record_count_words
            if isinstance(state, OneStepDynaState)
            and _array_has_contract(
                state.record_count_words, shape=(2,), dtype=jnp.uint32
            )
            else zero_words
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        return OneStepDynaRecordDiagnostics(
            state_static_contract_valid=jnp.asarray(state_static),
            model_state_static_contract_valid=jnp.asarray(model_static),
            control_state_static_contract_valid=jnp.asarray(control_static),
            anchor_static_contract_valid=jnp.asarray(anchor_static),
            state_valid=false,
            model_state_valid=false,
            control_state_valid=false,
            anchor_values_valid=false,
            authority_valid=false,
            revisions_monotonic=false,
            decision_identity_valid=false,
            record_capacity_available=false,
            candidate_state_valid=false,
            applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
            written_index=jnp.asarray(-1, dtype=jnp.int32),
            integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            pre_record_count_words=pre_words,
            post_record_count_words=pre_words,
        )

    def record_real_anchor(
        self,
        state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
        anchor: RealStateDynaAnchor,
    ) -> OneStepDynaRecordResult:
        """Atomically record one pre-model-update real state/action snapshot."""

        state_static = self._state_static_valid(state)
        model_static = self._model_static_valid(model_state)
        control_static = self._control_static_valid(control_state)
        anchor_static = self._anchor_static_valid(anchor)
        if not (state_static and model_static and control_static and anchor_static):
            return OneStepDynaRecordResult(
                state=state,
                diagnostics=self._unavailable_record_diagnostics(
                    state,
                    state_static=state_static,
                    model_static=model_static,
                    control_static=control_static,
                    anchor_static=anchor_static,
                ),
            )

        state_dynamic_valid = self._state_valid(state)
        model_dynamic_valid = self._model_valid(model_state)
        control_dynamic_valid = self._control_valid(control_state)
        authority = anchor.authority
        action = anchor.primitive_action
        safe_action = jnp.clip(action, 0, self.n_primitive_actions - 1)
        bound = jnp.asarray(self._config.max_anchor_magnitude, dtype=jnp.float32)
        anchor_values_valid = (
            jnp.all(jnp.isfinite(anchor.observation))
            & jnp.all(jnp.abs(anchor.observation) <= bound)
            & (action >= 0)
            & (action < self.n_primitive_actions)
        )
        authority_valid = (
            jnp.array_equal(
                authority.representation_revision_words,
                state.bound_representation_revision_words,
            )
            & jnp.array_equal(
                authority.model_revision_words,
                model_state.event_count_words,
            )
            & jnp.array_equal(
                authority.control_revision_words,
                control_state.step_words,
            )
        )
        revisions_monotonic = (
            _words_less_equal(
                state.bound_model_revision_words,
                authority.model_revision_words,
            )
            & _words_less_equal(
                state.bound_control_revision_words,
                authority.control_revision_words,
            )
        )
        decision_valid = _words_nonzero(anchor.decision_id_words) & jnp.where(
            state.size > 0,
            _words_less(state.last_decision_id_words, anchor.decision_id_words),
            True,
        )
        proposed_record_words, record_clock_available = _checked_words_add_small(
            state.record_count_words, 1
        )
        record_capacity = record_clock_available & _words_leq_limit(
            proposed_record_words,
            self._config.max_anchor_records,
        )
        preflight = (
            state_dynamic_valid
            & model_dynamic_valid
            & control_dynamic_valid
            & anchor_values_valid
            & authority_valid
            & revisions_monotonic
            & decision_valid
            & record_capacity
        )
        index = state.write_index
        replacing = state.size == self._config.anchor_capacity
        old_action = jnp.clip(
            state.anchor_actions[index], 0, self.n_primitive_actions - 1
        )
        support = state.action_support_counts.at[old_action].add(
            jnp.where(replacing, -1, 0)
        )
        support = support.at[safe_action].add(1)
        tag = self._anchor_tag(
            anchor.observation,
            safe_action,
            anchor.decision_id_words,
            authority.representation_revision_words,
            authority.model_revision_words,
            authority.control_revision_words,
        )
        candidate = state.replace(
            anchor_observations=state.anchor_observations.at[index].set(
                anchor.observation
            ),
            anchor_actions=state.anchor_actions.at[index].set(safe_action),
            anchor_decision_id_words=state.anchor_decision_id_words.at[index].set(
                anchor.decision_id_words
            ),
            anchor_representation_revision_words=(
                state.anchor_representation_revision_words.at[index].set(
                    authority.representation_revision_words
                )
            ),
            anchor_model_revision_words=state.anchor_model_revision_words.at[
                index
            ].set(authority.model_revision_words),
            anchor_control_revision_words=state.anchor_control_revision_words.at[
                index
            ].set(authority.control_revision_words),
            anchor_integrity_tags=state.anchor_integrity_tags.at[index].set(tag),
            anchor_valid=state.anchor_valid.at[index].set(True),
            action_support_counts=support,
            size=jnp.minimum(
                state.size + 1,
                jnp.asarray(self._config.anchor_capacity, dtype=jnp.int32),
            ),
            write_index=(index + 1) % self._config.anchor_capacity,
            bound_model_revision_words=authority.model_revision_words,
            bound_control_revision_words=authority.control_revision_words,
            last_decision_id_words=anchor.decision_id_words,
            record_count_words=proposed_record_words,
        )
        candidate_valid = self._state_valid(candidate)
        applied = preflight & candidate_valid
        next_state = cast(
            OneStepDynaState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        return OneStepDynaRecordResult(
            state=next_state,
            diagnostics=OneStepDynaRecordDiagnostics(
                state_static_contract_valid=jnp.asarray(True),
                model_state_static_contract_valid=jnp.asarray(True),
                control_state_static_contract_valid=jnp.asarray(True),
                anchor_static_contract_valid=jnp.asarray(True),
                state_valid=state_dynamic_valid,
                model_state_valid=model_dynamic_valid,
                control_state_valid=control_dynamic_valid,
                anchor_values_valid=anchor_values_valid,
                authority_valid=authority_valid,
                revisions_monotonic=revisions_monotonic,
                decision_identity_valid=decision_valid,
                record_capacity_available=record_capacity,
                candidate_state_valid=candidate_valid,
                applied=applied,
                rejected=~applied,
                written_index=jnp.where(applied, index, -1),
                integrity_tag=jnp.where(applied, tag, 0),
                pre_record_count_words=state.record_count_words,
                post_record_count_words=next_state.record_count_words,
            ),
        )

    def _zero_plan_diagnostics(
        self,
        state: OneStepDynaState,
        *,
        state_static: bool,
        model_static: bool,
        control_static: bool,
        authority_static: bool,
    ) -> OneStepDynaDiagnostics:
        budget = self._config.backup_budget
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        state_model_words = (
            state.bound_model_revision_words
            if self._state_static_valid(state)
            else zero_words
        )
        state_control_words = (
            state.bound_control_revision_words
            if self._state_static_valid(state)
            else zero_words
        )
        call_words = (
            state.planning_call_count_words
            if self._state_static_valid(state)
            else zero_words
        )
        attempt_words = (
            state.planning_attempt_count_words
            if self._state_static_valid(state)
            else zero_words
        )
        backup_words = (
            state.planned_backup_count_words
            if self._state_static_valid(state)
            else zero_words
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        bools = jnp.zeros((budget,), dtype=jnp.bool_)
        floats = jnp.zeros((budget,), dtype=jnp.float32)
        return OneStepDynaDiagnostics(
            state_static_contract_valid=jnp.asarray(state_static),
            model_state_static_contract_valid=jnp.asarray(model_static),
            control_state_static_contract_valid=jnp.asarray(control_static),
            authority_static_contract_valid=jnp.asarray(authority_static),
            state_valid=false,
            model_state_valid=false,
            control_state_valid=false,
            authority_valid=false,
            anchors_available=false,
            planning_call_capacity_available=false,
            planning_attempt_capacity_available=false,
            planned_backup_capacity_available=false,
            selected_anchor_indices=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_actions=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_decision_id_words=jnp.zeros((budget, 2), dtype=jnp.uint32),
            selected_anchor_model_revision_words=jnp.zeros(
                (budget, 2), dtype=jnp.uint32
            ),
            action_support_counts=jnp.zeros((budget,), dtype=jnp.int32),
            predicted_next_observations=jnp.zeros(
                (budget, self.observation_dim), dtype=jnp.float32
            ),
            predicted_rewards=floats,
            predicted_continuations=floats,
            epistemic_disagreements=floats,
            residual_variance_maxima=floats,
            successor_values=floats,
            control_targets=floats,
            td_errors=floats,
            anchor_identity_valid=bools,
            support_valid=bools,
            model_prediction_valid=bools,
            residual_proxy_ready=bools,
            epistemic_valid=bools,
            residual_variance_valid=bools,
            continuation_valid=bools,
            termination_agreement=bools,
            finite_target_valid=bools,
            per_backup_capacity_available=bools,
            guard_passed=bools,
            child_update_applied=bools,
            child_transaction_authenticated=bools,
            trace_and_utility_isolation_preserved=bools,
            applied=bools,
            applied_count=jnp.asarray(0, dtype=jnp.int32),
            transaction_applied=false,
            pre_model_revision_words=state_model_words,
            post_model_revision_words=state_model_words,
            pre_control_revision_words=state_control_words,
            post_control_revision_words=state_control_words,
            pre_planning_call_count_words=call_words,
            post_planning_call_count_words=call_words,
            pre_planning_attempt_count_words=attempt_words,
            post_planning_attempt_count_words=attempt_words,
            pre_planned_backup_count_words=backup_words,
            post_planned_backup_count_words=backup_words,
        )

    def _plan_preflight(
        self,
        state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
        authority: OneStepDynaAuthority,
    ) -> tuple[Array, ...]:
        state_valid = self._state_valid(state)
        model_valid = self._model_valid(model_state)
        control_valid = self._control_valid(control_state)
        authority_valid = (
            jnp.array_equal(
                authority.representation_revision_words,
                state.bound_representation_revision_words,
            )
            & jnp.array_equal(
                authority.model_revision_words,
                model_state.event_count_words,
            )
            & jnp.array_equal(
                authority.control_revision_words,
                control_state.step_words,
            )
            & _words_less_equal(
                state.bound_model_revision_words,
                authority.model_revision_words,
            )
            & _words_less_equal(
                state.bound_control_revision_words,
                authority.control_revision_words,
            )
        )
        proposed_calls, call_clock = _checked_words_add_small(
            state.planning_call_count_words, 1
        )
        call_capacity = call_clock & _words_leq_limit(
            proposed_calls, self._config.max_planning_calls
        )
        proposed_attempts, attempt_clock = _checked_words_add_small(
            state.planning_attempt_count_words,
            self._config.backup_budget,
        )
        attempt_capacity = attempt_clock & _words_leq_limit(
            proposed_attempts,
            self._config.max_planning_attempts,
        )
        backup_capacity = _words_leq_limit(
            state.planned_backup_count_words,
            self._config.max_planned_backups - 1,
        )
        anchors_available = state.size > 0
        valid = (
            state_valid
            & model_valid
            & control_valid
            & authority_valid
            & anchors_available
            & call_capacity
            & attempt_capacity
            & backup_capacity
        )
        return (
            valid,
            state_valid,
            model_valid,
            control_valid,
            authority_valid,
            anchors_available,
            call_capacity,
            attempt_capacity,
            backup_capacity,
            proposed_calls,
            proposed_attempts,
        )

    def plan(
        self,
        state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
        authority: OneStepDynaAuthority,
    ) -> OneStepDynaResult:
        """Apply at most ``backup_budget`` guarded caller-owned Q updates.

        A top-level invalid state, child snapshot, authority, or exhausted
        lifetime cap is an exact no-op including the planning RNG.  A valid
        call consumes exactly ``backup_budget`` anchor draws; individual
        unsupported, uncertain, non-finite, or termination-ambiguous dreams
        advance attempt/rejection clocks but cannot mutate the control target.
        """

        state_static = self._state_static_valid(state)
        model_static = self._model_static_valid(model_state)
        control_static = self._control_static_valid(control_state)
        authority_static = self._authority_static_valid(authority)
        if not (state_static and model_static and control_static and authority_static):
            return OneStepDynaResult(
                state=state,
                control_state=control_state,
                diagnostics=self._zero_plan_diagnostics(
                    state,
                    state_static=state_static,
                    model_static=model_static,
                    control_static=control_static,
                    authority_static=authority_static,
                ),
            )

        (
            preflight,
            state_dynamic_valid,
            model_dynamic_valid,
            control_dynamic_valid,
            authority_valid,
            anchors_available,
            call_capacity,
            attempt_capacity,
            initial_backup_capacity,
            proposed_calls,
            proposed_attempts,
        ) = self._plan_preflight(state, model_state, control_state, authority)

        zero_diagnostics = self._zero_plan_diagnostics(
            state,
            state_static=True,
            model_static=True,
            control_static=True,
            authority_static=True,
        ).replace(
            state_valid=state_dynamic_valid,
            model_state_valid=model_dynamic_valid,
            control_state_valid=control_dynamic_valid,
            authority_valid=authority_valid,
            anchors_available=anchors_available,
            planning_call_capacity_available=call_capacity,
            planning_attempt_capacity_available=attempt_capacity,
            planned_backup_capacity_available=initial_backup_capacity,
        )

        def do_plan(_: None) -> OneStepDynaResult:
            budget = self._config.backup_budget
            selected_indices = jnp.full((budget,), -1, dtype=jnp.int32)
            selected_actions = jnp.full((budget,), -1, dtype=jnp.int32)
            selected_decisions = jnp.zeros((budget, 2), dtype=jnp.uint32)
            selected_model_revisions = jnp.zeros((budget, 2), dtype=jnp.uint32)
            support_counts = jnp.zeros((budget,), dtype=jnp.int32)
            predicted_next = jnp.zeros(
                (budget, self.observation_dim), dtype=jnp.float32
            )
            rewards = jnp.zeros((budget,), dtype=jnp.float32)
            continuations = jnp.zeros((budget,), dtype=jnp.float32)
            epistemic = jnp.zeros((budget,), dtype=jnp.float32)
            residual_max = jnp.zeros((budget,), dtype=jnp.float32)
            successor_values = jnp.zeros((budget,), dtype=jnp.float32)
            targets_log = jnp.zeros((budget,), dtype=jnp.float32)
            td_errors = jnp.zeros((budget,), dtype=jnp.float32)
            bool_log = jnp.zeros((budget,), dtype=jnp.bool_)
            carry: tuple[Any, ...] = (
                control_state,
                state.planning_key,
                state.planned_backup_count_words,
                selected_indices,
                selected_actions,
                selected_decisions,
                selected_model_revisions,
                support_counts,
                predicted_next,
                rewards,
                continuations,
                epistemic,
                residual_max,
                successor_values,
                targets_log,
                td_errors,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
                bool_log,
            )

            def backup_body(index: int, loop: tuple[Any, ...]) -> tuple[Any, ...]:
                (
                    current_control,
                    current_key,
                    applied_words,
                    index_log,
                    action_log,
                    decision_log,
                    model_revision_log,
                    support_log,
                    next_log,
                    reward_log,
                    continuation_log,
                    epistemic_log,
                    residual_log,
                    successor_log,
                    target_log,
                    td_log,
                    identity_log,
                    support_valid_log,
                    prediction_valid_log,
                    residual_ready_log,
                    epistemic_valid_log,
                    residual_valid_log,
                    continuation_valid_log,
                    termination_log,
                    finite_log,
                    capacity_log,
                    guard_log,
                    child_applied_log,
                    child_authenticated_log,
                    isolation_log,
                    applied_log,
                ) = loop
                next_key, draw_key = jr.split(current_key)
                anchor_index = jr.randint(
                    draw_key,
                    (),
                    0,
                    state.size,
                    dtype=jnp.int32,
                )
                observation = state.anchor_observations[anchor_index]
                action = state.anchor_actions[anchor_index]
                safe_action = jnp.clip(action, 0, self.n_primitive_actions - 1)
                decision_words = state.anchor_decision_id_words[anchor_index]
                anchor_model_words = state.anchor_model_revision_words[anchor_index]
                support_count = state.action_support_counts[safe_action]
                tag = self._anchor_tag(
                    observation,
                    safe_action,
                    decision_words,
                    state.anchor_representation_revision_words[anchor_index],
                    anchor_model_words,
                    state.anchor_control_revision_words[anchor_index],
                )
                identity_valid = (
                    state.anchor_valid[anchor_index]
                    & (action == safe_action)
                    & jnp.array_equal(
                        tag, state.anchor_integrity_tags[anchor_index]
                    )
                    & jnp.array_equal(
                        state.anchor_representation_revision_words[anchor_index],
                        authority.representation_revision_words,
                    )
                    & _words_less_equal(
                        anchor_model_words,
                        authority.model_revision_words,
                    )
                    & _words_less_equal(
                        state.anchor_control_revision_words[anchor_index],
                        authority.control_revision_words,
                    )
                )
                supported = support_count >= self._config.min_action_support
                prediction = self._ensemble.predict(
                    model_state,
                    observation,
                    safe_action,
                )
                member_discounts = prediction.member_discounts
                threshold = jnp.asarray(
                    self._config.terminal_discount_threshold,
                    dtype=jnp.float32,
                )
                terminal_votes = member_discounts <= threshold
                all_terminal = jnp.all(terminal_votes)
                all_continuing = jnp.all(~terminal_votes)
                termination_agreement = all_terminal | all_continuing
                if not self._config.require_termination_agreement:
                    termination_agreement = jnp.asarray(True, dtype=jnp.bool_)
                    all_terminal = prediction.mean_discount <= threshold
                continuation = jnp.where(
                    all_terminal,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    prediction.mean_discount,
                )
                continuation_semantics = (
                    jnp.all(jnp.isfinite(member_discounts))
                    & jnp.all(member_discounts >= 0.0)
                    & jnp.all(
                        member_discounts
                        <= jnp.asarray(
                            self._ensemble.config.model.gamma,
                            dtype=jnp.float32,
                        )
                    )
                    & jnp.isfinite(continuation)
                    & (continuation >= 0.0)
                    & (
                        continuation
                        <= jnp.asarray(
                            self._ensemble.config.model.gamma,
                            dtype=jnp.float32,
                        )
                    )
                )
                residual_value = jnp.max(prediction.residual_variances)
                residual_ready = prediction.residual_proxy_ready
                if not self._config.require_residual_proxy_ready:
                    residual_ready = jnp.asarray(True, dtype=jnp.bool_)
                epistemic_ok = (
                    jnp.isfinite(prediction.epistemic_disagreement)
                    & (
                        prediction.epistemic_disagreement
                        <= jnp.asarray(
                            self._config.max_epistemic_disagreement,
                            dtype=jnp.float32,
                        )
                    )
                )
                residual_ok = (
                    jnp.all(jnp.isfinite(prediction.residual_variances))
                    & jnp.isfinite(residual_value)
                    & (
                        residual_value
                        <= jnp.asarray(
                            self._config.max_residual_variance,
                            dtype=jnp.float32,
                        )
                    )
                )
                current_values = self._control.predict(current_control, observation)
                next_values = self._control.predict(
                    current_control,
                    prediction.mean_next_observation,
                )
                successor_value = jnp.max(next_values)
                target = prediction.mean_reward + continuation * successor_value
                td_error = target - current_values[safe_action]
                finite_target = (
                    jnp.all(jnp.isfinite(prediction.member_raw_predictions))
                    & jnp.all(jnp.isfinite(prediction.member_next_observations))
                    & jnp.all(jnp.isfinite(prediction.member_rewards))
                    & jnp.all(jnp.isfinite(current_values))
                    & jnp.all(jnp.isfinite(next_values))
                    & jnp.isfinite(successor_value)
                    & jnp.isfinite(target)
                    & jnp.isfinite(td_error)
                    & (
                        jnp.abs(target)
                        <= jnp.asarray(
                            self._config.max_abs_control_target,
                            dtype=jnp.float32,
                        )
                    )
                )
                per_backup_capacity = _words_leq_limit(
                    applied_words,
                    self._config.max_planned_backups - 1,
                )
                guard = (
                    identity_valid
                    & supported
                    & prediction.valid
                    & residual_ready
                    & epistemic_ok
                    & residual_ok
                    & continuation_semantics
                    & termination_agreement
                    & finite_target
                    & per_backup_capacity
                )

                zero_traces = current_control.replace(
                    trunk_traces=tuple(
                        jnp.zeros_like(trace)
                        for trace in current_control.trunk_traces
                    ),
                    head_traces=tuple(
                        (
                            jnp.zeros_like(weight_trace),
                            jnp.zeros_like(bias_trace),
                        )
                        for weight_trace, bias_trace in current_control.head_traces
                    ),
                )
                control_targets = jnp.full(
                    (self.n_primitive_actions,),
                    jnp.nan,
                    dtype=jnp.float32,
                ).at[safe_action].set(target)
                expected_step_words, child_capacity = _checked_words_add_small(
                    current_control.step_words, 1
                )
                child_counter_valid = self._control._counter_status(
                    current_control
                ).lifetime_counter_valid

                def propose(_: None) -> tuple[Any, ...]:
                    update = self._control.update(
                        zero_traces,
                        observation,
                        control_targets,
                    )
                    restored = update.state.replace(
                        trunk_traces=current_control.trunk_traces,
                        head_traces=current_control.head_traces,
                        hidden_unit_utilities=current_control.hidden_unit_utilities,
                        normalizer_state=current_control.normalizer_state,
                    )
                    isolated = (
                        _tree_exactly_equal(
                            restored.trunk_traces,
                            current_control.trunk_traces,
                        )
                        & _tree_exactly_equal(
                            restored.head_traces,
                            current_control.head_traces,
                        )
                        & _tree_exactly_equal(
                            restored.hidden_unit_utilities,
                            current_control.hidden_unit_utilities,
                        )
                        & (restored.normalizer_state is None)
                    )
                    authenticated = (
                        jnp.array_equal(
                            update.pre_step_words, current_control.step_words
                        )
                        & jnp.array_equal(
                            update.post_step_words, update.state.step_words
                        )
                        & jnp.array_equal(
                            update.post_step_words, expected_step_words
                        )
                        & (update.lifetime_counter_valid == child_counter_valid)
                        & update.lifetime_capacity_available
                        & child_capacity
                        & update.normalizer_counter_aligned
                        & update.normalizer_estimator_capacity_available
                    )
                    accepted = (
                        update.update_applied
                        & authenticated
                        & isolated
                        & _tree_is_finite(restored)
                        & self._control_valid(restored)
                        & jnp.isfinite(update.errors[safe_action])
                    )
                    return (
                        cast(
                            MultiHeadMLPState,
                            jax.lax.cond(
                                accepted,
                                lambda _: restored,
                                lambda _: current_control,
                                None,
                            ),
                        ),
                        update.update_applied,
                        authenticated,
                        isolated,
                        accepted,
                        jnp.where(accepted, update.errors[safe_action], 0.0),
                    )

                def skip(_: None) -> tuple[Any, ...]:
                    false = jnp.asarray(False, dtype=jnp.bool_)
                    return (
                        current_control,
                        false,
                        false,
                        false,
                        false,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )

                (
                    next_control,
                    child_applied,
                    child_authenticated,
                    isolation_preserved,
                    accepted,
                    committed_td_error,
                ) = jax.lax.cond(guard, propose, skip, None)
                next_applied_words, applied_clock = _checked_words_add_small(
                    applied_words,
                    accepted.astype(jnp.uint32),
                )
                accepted = accepted & applied_clock
                next_applied_words = jnp.where(
                    accepted, next_applied_words, applied_words
                )
                next_control = cast(
                    MultiHeadMLPState,
                    jax.lax.cond(
                        accepted,
                        lambda _: next_control,
                        lambda _: current_control,
                        None,
                    ),
                )
                return (
                    next_control,
                    next_key,
                    next_applied_words,
                    index_log.at[index].set(anchor_index),
                    action_log.at[index].set(action),
                    decision_log.at[index].set(decision_words),
                    model_revision_log.at[index].set(anchor_model_words),
                    support_log.at[index].set(support_count),
                    next_log.at[index].set(
                        jnp.where(
                            prediction.valid,
                            prediction.mean_next_observation,
                            0.0,
                        )
                    ),
                    reward_log.at[index].set(
                        jnp.where(prediction.valid, prediction.mean_reward, 0.0)
                    ),
                    continuation_log.at[index].set(
                        jnp.where(prediction.valid, continuation, 0.0)
                    ),
                    epistemic_log.at[index].set(
                        jnp.where(
                            jnp.isfinite(prediction.epistemic_disagreement),
                            prediction.epistemic_disagreement,
                            0.0,
                        )
                    ),
                    residual_log.at[index].set(
                        jnp.where(jnp.isfinite(residual_value), residual_value, 0.0)
                    ),
                    successor_log.at[index].set(
                        jnp.where(finite_target, successor_value, 0.0)
                    ),
                    target_log.at[index].set(jnp.where(finite_target, target, 0.0)),
                    td_log.at[index].set(committed_td_error),
                    identity_log.at[index].set(identity_valid),
                    support_valid_log.at[index].set(supported),
                    prediction_valid_log.at[index].set(prediction.valid),
                    residual_ready_log.at[index].set(residual_ready),
                    epistemic_valid_log.at[index].set(epistemic_ok),
                    residual_valid_log.at[index].set(residual_ok),
                    continuation_valid_log.at[index].set(continuation_semantics),
                    termination_log.at[index].set(termination_agreement),
                    finite_log.at[index].set(finite_target),
                    capacity_log.at[index].set(per_backup_capacity),
                    guard_log.at[index].set(guard),
                    child_applied_log.at[index].set(child_applied),
                    child_authenticated_log.at[index].set(child_authenticated),
                    isolation_log.at[index].set(isolation_preserved),
                    applied_log.at[index].set(accepted),
                )

            final = jax.lax.fori_loop(0, budget, backup_body, carry)
            (
                final_control,
                final_key,
                final_applied_words,
                selected_indices,
                selected_actions,
                selected_decisions,
                selected_model_revisions,
                support_counts,
                predicted_next,
                rewards,
                continuations,
                epistemic,
                residual_max,
                successor_values,
                targets_log,
                td_errors,
                identity_log,
                support_valid_log,
                prediction_valid_log,
                residual_ready_log,
                epistemic_valid_log,
                residual_valid_log,
                continuation_valid_log,
                termination_log,
                finite_log,
                capacity_log,
                guard_log,
                child_applied_log,
                child_authenticated_log,
                isolation_log,
                applied_log,
            ) = final
            applied_count = jnp.sum(applied_log.astype(jnp.int32))
            rejected_count = budget - applied_count
            proposed_rejected, rejected_clock = _checked_words_add_small(
                state.rejected_backup_count_words,
                rejected_count,
            )
            expected_applied, applied_clock = _checked_words_add_small(
                state.planned_backup_count_words,
                applied_count,
            )
            clocks_match = jnp.array_equal(expected_applied, final_applied_words)
            candidate_state = state.replace(
                bound_model_revision_words=authority.model_revision_words,
                bound_control_revision_words=final_control.step_words,
                planning_call_count_words=proposed_calls,
                planning_attempt_count_words=proposed_attempts,
                planned_backup_count_words=final_applied_words,
                rejected_backup_count_words=proposed_rejected,
                planning_key=final_key,
            )
            candidate_valid = (
                rejected_clock
                & applied_clock
                & clocks_match
                & self._state_valid(candidate_state)
                & self._control_valid(final_control)
            )
            next_state = cast(
                OneStepDynaState,
                jax.lax.cond(
                    candidate_valid,
                    lambda _: candidate_state,
                    lambda _: state,
                    None,
                ),
            )
            next_control = cast(
                MultiHeadMLPState,
                jax.lax.cond(
                    candidate_valid,
                    lambda _: final_control,
                    lambda _: control_state,
                    None,
                ),
            )
            committed_applied = applied_log & candidate_valid
            diagnostics = OneStepDynaDiagnostics(
                state_static_contract_valid=jnp.asarray(True),
                model_state_static_contract_valid=jnp.asarray(True),
                control_state_static_contract_valid=jnp.asarray(True),
                authority_static_contract_valid=jnp.asarray(True),
                state_valid=state_dynamic_valid,
                model_state_valid=model_dynamic_valid,
                control_state_valid=control_dynamic_valid,
                authority_valid=authority_valid,
                anchors_available=anchors_available,
                planning_call_capacity_available=call_capacity,
                planning_attempt_capacity_available=attempt_capacity,
                planned_backup_capacity_available=initial_backup_capacity,
                selected_anchor_indices=selected_indices,
                selected_actions=selected_actions,
                selected_decision_id_words=selected_decisions,
                selected_anchor_model_revision_words=selected_model_revisions,
                action_support_counts=support_counts,
                predicted_next_observations=predicted_next,
                predicted_rewards=rewards,
                predicted_continuations=continuations,
                epistemic_disagreements=epistemic,
                residual_variance_maxima=residual_max,
                successor_values=successor_values,
                control_targets=targets_log,
                td_errors=jnp.where(candidate_valid, td_errors, 0.0),
                anchor_identity_valid=identity_log,
                support_valid=support_valid_log,
                model_prediction_valid=prediction_valid_log,
                residual_proxy_ready=residual_ready_log,
                epistemic_valid=epistemic_valid_log,
                residual_variance_valid=residual_valid_log,
                continuation_valid=continuation_valid_log,
                termination_agreement=termination_log,
                finite_target_valid=finite_log,
                per_backup_capacity_available=capacity_log,
                guard_passed=guard_log,
                child_update_applied=child_applied_log,
                child_transaction_authenticated=child_authenticated_log,
                trace_and_utility_isolation_preserved=isolation_log,
                applied=committed_applied,
                applied_count=jnp.sum(committed_applied.astype(jnp.int32)),
                transaction_applied=candidate_valid,
                pre_model_revision_words=model_state.event_count_words,
                post_model_revision_words=model_state.event_count_words,
                pre_control_revision_words=control_state.step_words,
                post_control_revision_words=next_control.step_words,
                pre_planning_call_count_words=state.planning_call_count_words,
                post_planning_call_count_words=next_state.planning_call_count_words,
                pre_planning_attempt_count_words=state.planning_attempt_count_words,
                post_planning_attempt_count_words=(
                    next_state.planning_attempt_count_words
                ),
                pre_planned_backup_count_words=state.planned_backup_count_words,
                post_planned_backup_count_words=next_state.planned_backup_count_words,
            )
            return OneStepDynaResult(
                state=next_state,
                control_state=next_control,
                diagnostics=diagnostics,
            )

        def reject(_: None) -> OneStepDynaResult:
            return OneStepDynaResult(
                state=state,
                control_state=control_state,
                diagnostics=zero_diagnostics,
            )

        return cast(
            OneStepDynaResult,
            jax.lax.cond(preflight, do_plan, reject, None),
        )

    @property
    def resource_budget(self) -> OneStepDynaResourceBudget:
        """Measure exact module-owned state/output bytes and static work caps."""

        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        template = self._empty_state(
            jr.key(0, impl="threefry2x32"),
            jnp.asarray([0, 1], dtype=jnp.uint32),
            zero_words,
            zero_words,
        )
        state_scalars, state_bytes = _logical_tree_size(template)
        record = self._unavailable_record_diagnostics(
            template,
            state_static=True,
            model_static=True,
            control_static=True,
            anchor_static=True,
        )
        plan = self._zero_plan_diagnostics(
            template,
            state_static=True,
            model_static=True,
            control_static=True,
            authority_static=True,
        )
        record_scalars, record_bytes = _logical_tree_size(record)
        plan_scalars, plan_bytes = _logical_tree_size(plan)
        capacity = self._config.anchor_capacity
        persistent_float = capacity * self.observation_dim
        persistent_int = capacity + self.n_primitive_actions + 2
        persistent_bool = capacity
        # Nine uint32 words per anchor (four identities plus one tag), eight
        # bound/last-identity words, ten exact counter words, and one two-word
        # typed Threefry key.
        persistent_uint = 9 * capacity + 20
        if (
            4 * (persistent_float + persistent_int + persistent_uint)
            + persistent_bool
            != state_bytes
        ):
            raise ValueError("one-step Dyna state resource formula drifted")
        return OneStepDynaResourceBudget(
            persistent_bytes_scope=(
                "planner-owned-persistent-array-leaves-only; excludes-model,control,"
                "host-object-overhead,compiler-and-xla-workspaces"
            ),
            diagnostic_bytes_scope=(
                "full-fixed-shape-return-array-leaves; not-a-measured-device-peak"
            ),
            temporary_bytes_scope=(
                "not-measured; source-level-call-count-upper-bounds-only; excludes-"
                "compiler-and-xla-workspaces"
            ),
            observation_dim=self.observation_dim,
            n_primitive_actions=self.n_primitive_actions,
            ensemble_size=self._ensemble.config.ensemble_size,
            anchor_capacity=capacity,
            backup_budget=self._config.backup_budget,
            persistent_float32_scalars=persistent_float,
            persistent_int32_scalars=persistent_int,
            persistent_uint32_scalars=persistent_uint,
            persistent_bool_scalars=persistent_bool,
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            planning_prng_keys=1,
            planning_prng_uint32_scalars=2,
            record_diagnostics_scalars=record_scalars,
            record_diagnostics_bytes=record_bytes,
            plan_diagnostics_scalars=plan_scalars,
            plan_diagnostics_bytes=plan_bytes,
            max_ensemble_prediction_calls_per_call=self._config.backup_budget,
            max_member_model_predictions_per_call=(
                self._config.backup_budget * self._ensemble.config.ensemble_size
            ),
            max_control_target_forward_calls_per_call=(
                2 * self._config.backup_budget
            ),
            max_control_update_forward_calls_per_call=self._config.backup_budget,
            max_control_forward_calls_per_call=3 * self._config.backup_budget,
            max_control_backward_calls_per_call=self._config.backup_budget,
            max_control_updates_per_call=self._config.backup_budget,
            max_planning_rng_splits_per_call=self._config.backup_budget,
            max_planning_rng_draws_per_call=self._config.backup_budget,
            max_anchor_records=self._config.max_anchor_records,
            max_planning_calls=self._config.max_planning_calls,
            max_planning_attempts=self._config.max_planning_attempts,
            max_planned_backups=self._config.max_planned_backups,
            model_state_owned=0,
            control_state_owned=0,
            replay_capacity=capacity,
        )


def save_one_step_dyna_checkpoint(
    planner: RealStateOneStepDyna,
    state: OneStepDynaState,
    path: str | Path,
) -> None:
    """Persist only planner-owned anchors, identities, clocks, and RNG."""

    planner.validate_state(state)
    config = planner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": ONE_STEP_DYNA_CHECKPOINT_SCHEMA,
            "planner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": planner.resource_budget.to_config(),
            "model_state_included": False,
            "control_state_included": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_one_step_dyna_checkpoint(
    path: str | Path,
) -> tuple[RealStateOneStepDyna, OneStepDynaState]:
    """Restore the sole v1 planner checkpoint; there is no legacy migration."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != ONE_STEP_DYNA_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a one-step Dyna v1 checkpoint")
    config = metadata.get("planner_config")
    if not isinstance(config, Mapping):
        raise ValueError("one-step Dyna checkpoint is missing planner_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("one-step Dyna checkpoint config digest does not match")
    planner = RealStateOneStepDyna.from_config(config_dict)
    if metadata.get("resource_budget") != planner.resource_budget.to_config():
        raise ValueError("one-step Dyna checkpoint resource budget does not match")
    if metadata.get("model_state_included") is not False:
        raise ValueError("one-step Dyna checkpoint must not own model state")
    if metadata.get("control_state_included") is not False:
        raise ValueError("one-step Dyna checkpoint must not own control state")
    if metadata.get("scientific_promotion_allowed") is not False:
        raise ValueError("one-step Dyna checkpoint cannot claim promotion")
    zero_words = jnp.zeros((2,), dtype=jnp.uint32)
    template = planner._empty_state(
        jr.key(0, impl="threefry2x32"),
        zero_words,
        zero_words,
        zero_words,
    )
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("one-step Dyna checkpoint metadata changed between reads")
    state = cast(OneStepDynaState, restored)
    planner.validate_state(state)
    if _logical_tree_size(state)[1] != planner.resource_budget.persistent_state_bytes:
        raise ValueError("restored one-step Dyna state resource size is invalid")
    return planner, state


__all__ = [
    "ONE_STEP_DYNA_CHECKPOINT_SCHEMA",
    "ONE_STEP_DYNA_CONFIG_SCHEMA",
    "ONE_STEP_DYNA_EVIDENCE_LEVEL",
    "ONE_STEP_DYNA_MECHANISM_STATUS",
    "ONE_STEP_DYNA_SCIENTIFIC_PROMOTION_ALLOWED",
    "OneStepDynaAuthority",
    "OneStepDynaConfig",
    "OneStepDynaDiagnostics",
    "OneStepDynaRecordDiagnostics",
    "OneStepDynaRecordResult",
    "OneStepDynaResourceBudget",
    "OneStepDynaResult",
    "OneStepDynaState",
    "RealStateDynaAnchor",
    "RealStateOneStepDyna",
    "load_one_step_dyna_checkpoint",
    "save_one_step_dyna_checkpoint",
]
