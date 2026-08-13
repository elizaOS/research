"""Causal multi-timescale GVF and inverse objectives for learned state.

This isolated WP3 kernel closes one missing mechanism from section 3.3 of
``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``.  It owns separate linear heads for
one scalar cumulant at multiple prediction timescales and for inverse action
classification from a consecutive representation pair.  The two objective
groups have fixed, declared mass; the GVF group is averaged over heads before
its group weight is applied, so adding timescales cannot silently increase its
share of the representation gradient.

The causal boundary is explicit.  :meth:`BalancedStateObjectives.cache_action`
binds the representation and representation revision that owned the action
which actually executed.  :meth:`BalancedStateObjectives.update` accepts only
the bit-identical receipt, a successor representation with a nondecreasing
caller-owned revision, and one transition's cumulant/continuation.  Invalid,
stale, overflowing, or structurally corrupt transactions leave the complete
state unchanged.

The GVFs use a stopped-bootstrap one-step semi-gradient

``target_i = cumulant + continuation * gamma_i * stop_gradient(v_i(z_next))``.

The inverse head uses ordinary softmax cross entropy over the cached executed
action.  Head parameters update from their separate losses; the returned
current/successor representation gradients are the declared group-weighted
sum and can be routed to the matching state-builder owners by a higher-level
integration.  This module does not perform that commit itself.

This is an L0, ``not_assessed`` mechanism.  Its fixed weighting is not an
empirically calibrated balance, its linear heads are not a full learned-state
architecture, and it provides no feature-utility, causal deletion, retention,
Forager, control-benefit, Alberta Plan completion, or SOTA evidence.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
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

BALANCED_STATE_OBJECTIVES_CONFIG_SCHEMA = "alberta.balanced-state-objectives-config.v1"
BALANCED_STATE_OBJECTIVES_STATE_SCHEMA = "alberta.balanced-state-objectives-state.v1"
BALANCED_STATE_OBJECTIVES_CHECKPOINT_SCHEMA = (
    "alberta.balanced-state-objectives-checkpoint.v1"
)
BALANCED_STATE_OBJECTIVES_RESOURCE_SCHEMA = "alberta.balanced-state-objectives-resource.v1"
BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL = "L0"
BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS = "not_assessed"
BALANCED_STATE_OBJECTIVES_OWNERSHIP = (
    "cache-executed-action; bit-exact-receipt; caller-owned-representation-revisions"
)
BALANCED_STATE_OBJECTIVES_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
BALANCED_STATE_OBJECTIVES_MAX_DECISIONS = 2**64 - 1
BALANCED_STATE_OBJECTIVES_MAX_UPDATES = 2**64 - 1
BALANCED_STATE_OBJECTIVES_LIMITATIONS = (
    "single-scalar-cumulant-linear-semi-gradient-gvfs",
    "linear-consecutive-pair-inverse-action-head",
    "fixed-declared-not-empirically-calibrated-group-balance",
    "no-feature-utility-or-causal-deletion",
    "no-retention-control-forager-or-sota-evidence",
)

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Copy one exact dictionary after rejecting missing or unknown fields."""

    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _exact_int(value: Any, *, label: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _finite_real(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    below = scalar <= minimum if strict_minimum else scalar < minimum
    if below or (maximum is not None and scalar > maximum):
        relation = ">" if strict_minimum else ">="
        suffix = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{label} must be {relation} {minimum}{suffix}")
    if scalar > _FLOAT32_MAX:
        raise ValueError(f"{label} exceeds the finite float32 range")
    return scalar


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_float32_vector(value: Any, size: int, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(size,),
        dtype=jnp.dtype(jnp.float32),
    )


def _require_float32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.float32),
    )


def _require_int32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.bool_),
    )


def _require_words(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )


def _require_threefry_key(value: Any, *, label: str) -> None:
    try:
        key_data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry JAX key") from exc
    if (
        getattr(value, "shape", None) != ()
        or key_data.shape != (2,)
        or key_data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry JAX key")


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Return the exact big-endian uint64 successor and capacity verdict."""

    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    successor = jnp.stack(
        (
            words[0] + carry.astype(jnp.uint32),
            words[1] + jnp.asarray(1, dtype=jnp.uint32),
        )
    ).astype(jnp.uint32)
    return successor, capacity


def _words_not_earlier(candidate: Array, reference: Array) -> Bool[Array, ""]:
    """Return unsigned lexicographic ``candidate >= reference``."""

    return (candidate[0] > reference[0]) | (
        (candidate[0] == reference[0]) & (candidate[1] >= reference[1])
    )


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


def _safe_norm_and_clip(value: Array, limit: float) -> tuple[Array, Array]:
    """Return a finite-saturated norm and globally clipped vector."""

    scale = jnp.max(jnp.abs(value))
    safe_scale = jnp.where(scale > 0.0, scale, jnp.float32(1.0))
    scaled = value / safe_scale
    scaled_norm = jnp.sqrt(jnp.sum(jnp.square(scaled)))
    norm = scale * scaled_norm
    norm = jnp.nan_to_num(norm, nan=_FLOAT32_MAX, posinf=_FLOAT32_MAX)
    safe_norm = jnp.where(norm > 0.0, norm, jnp.float32(1.0))
    factor = jnp.minimum(jnp.float32(1.0), jnp.float32(limit) / safe_norm)
    return norm, value * factor


def _finite_or_max(value: Array) -> Array:
    """Sanitize a scalar diagnostic without granting transaction validity."""

    return jnp.nan_to_num(
        value,
        nan=_FLOAT32_MAX,
        posinf=_FLOAT32_MAX,
        neginf=_FLOAT32_MAX,
    )


@dataclasses.dataclass(frozen=True)
class BalancedStateObjectivesConfig:
    """Static shapes, head learning, group balance, and numerical bounds."""

    representation_dim: int
    n_actions: int
    gvf_discounts: tuple[float, ...] = (0.5, 0.9, 0.99)
    gvf_step_size: float = 0.01
    inverse_step_size: float = 0.01
    gvf_group_weight: float = 0.5
    inverse_group_weight: float = 0.5
    initialization_scale: float = 0.05
    representation_gradient_clip: float = 100.0
    max_abs_representation: float = 1.0e6
    max_abs_cumulant: float = 1.0e6

    def __post_init__(self) -> None:
        _exact_int(self.representation_dim, label="representation_dim", minimum=1)
        _exact_int(self.n_actions, label="n_actions", minimum=2)
        if type(self.gvf_discounts) is not tuple:
            raise TypeError("gvf_discounts must be a tuple")
        if len(self.gvf_discounts) < 2:
            raise ValueError("gvf_discounts must contain at least two timescales")
        discounts = tuple(
            _finite_real(value, label="gvf discount", minimum=0.0, maximum=1.0)
            for value in self.gvf_discounts
        )
        if any(value >= 1.0 for value in discounts):
            raise ValueError("every gvf discount must be < 1.0")
        if any(right <= left for left, right in zip(discounts, discounts[1:])):
            raise ValueError("gvf_discounts must be strictly increasing")
        for name in ("gvf_step_size", "inverse_step_size"):
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )
        gvf_weight = _finite_real(
            self.gvf_group_weight,
            label="gvf_group_weight",
            minimum=0.0,
            strict_minimum=True,
        )
        inverse_weight = _finite_real(
            self.inverse_group_weight,
            label="inverse_group_weight",
            minimum=0.0,
            strict_minimum=True,
        )
        if not math.isclose(gvf_weight + inverse_weight, 1.0, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("objective group weights must sum to one")
        for name in (
            "initialization_scale",
            "representation_gradient_clip",
            "max_abs_representation",
            "max_abs_cumulant",
        ):
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )

    @property
    def n_gvf_heads(self) -> int:
        return len(self.gvf_discounts)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "BalancedStateObjectives",
            "schema": BALANCED_STATE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": BALANCED_STATE_OBJECTIVES_STATE_SCHEMA,
            "evidence_level": BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": BALANCED_STATE_OBJECTIVES_OWNERSHIP,
            "limitations": list(BALANCED_STATE_OBJECTIVES_LIMITATIONS),
            "representation_dim": self.representation_dim,
            "n_actions": self.n_actions,
            "gvf_discounts": [float(value) for value in self.gvf_discounts],
            "gvf_step_size": float(self.gvf_step_size),
            "inverse_step_size": float(self.inverse_step_size),
            "gvf_group_weight": float(self.gvf_group_weight),
            "inverse_group_weight": float(self.inverse_group_weight),
            "initialization_scale": float(self.initialization_scale),
            "representation_gradient_clip": float(self.representation_gradient_clip),
            "max_abs_representation": float(self.max_abs_representation),
            "max_abs_cumulant": float(self.max_abs_cumulant),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> BalancedStateObjectivesConfig:
        expected = {
            "type",
            "schema",
            "state_schema",
            "evidence_level",
            "outcome_status",
            "ownership",
            "limitations",
            "representation_dim",
            "n_actions",
            "gvf_discounts",
            "gvf_step_size",
            "inverse_step_size",
            "gvf_group_weight",
            "inverse_group_weight",
            "initialization_scale",
            "representation_gradient_clip",
            "max_abs_representation",
            "max_abs_cumulant",
        }
        fields = _exact_manifest(payload, expected, label="balanced state objectives config")
        checks = {
            "type": "BalancedStateObjectives",
            "schema": BALANCED_STATE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": BALANCED_STATE_OBJECTIVES_STATE_SCHEMA,
            "evidence_level": BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": BALANCED_STATE_OBJECTIVES_OWNERSHIP,
            "limitations": list(BALANCED_STATE_OBJECTIVES_LIMITATIONS),
        }
        for name, expected_value in checks.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"balanced state objectives {name} is unsupported")
        discounts = fields.get("gvf_discounts")
        if type(discounts) is not list:
            raise TypeError("serialized gvf_discounts must be a list")
        fields["gvf_discounts"] = tuple(discounts)
        return cls(**fields)


@chex.dataclass(frozen=True)
class BalancedStateObjectivesState:
    """Bounded separate-head state and one pending causal action owner."""

    gvf_weights: Float[Array, "timescale representation"]
    inverse_current_weights: Float[Array, "action representation"]
    inverse_next_weights: Float[Array, "action representation"]
    inverse_bias: Float[Array, " action"]
    pending_representation: Float[Array, " representation"]
    pending_action: Int[Array, ""]
    pending_representation_revision_words: UInt[Array, " 2"]
    pending_action_identity_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    decision_words: UInt[Array, " 2"]
    update_words: UInt[Array, " 2"]
    head_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class StateObjectiveActionReceipt:
    """Transient exact record of the representation/action that executed."""

    representation: Float[Array, " representation"]
    action: Int[Array, ""]
    representation_revision_words: UInt[Array, " 2"]
    action_identity_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class StateObjectiveActionCacheResult:
    """Atomic result of binding one executed action to its representation."""

    state: BalancedStateObjectivesState
    receipt: StateObjectiveActionReceipt
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    cache_available: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    cache_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class BalancedStateObjectiveUpdateResult:
    """Separate losses/gradients plus one atomic head-update transaction."""

    state: BalancedStateObjectivesState
    gvf_predictions: Float[Array, " timescale"]
    gvf_targets: Float[Array, " timescale"]
    gvf_td_errors: Float[Array, " timescale"]
    inverse_probabilities: Float[Array, " action"]
    gvf_loss: Float[Array, ""]
    inverse_loss: Float[Array, ""]
    balanced_loss: Float[Array, ""]
    gvf_current_representation_gradient: Float[Array, " representation"]
    inverse_current_representation_gradient: Float[Array, " representation"]
    inverse_next_representation_gradient: Float[Array, " representation"]
    current_representation_gradient: Float[Array, " representation"]
    next_representation_gradient: Float[Array, " representation"]
    unclipped_current_gradient_norm: Float[Array, ""]
    unclipped_next_gradient_norm: Float[Array, ""]
    current_gradient_was_clipped: Bool[Array, ""]
    next_gradient_was_clipped: Bool[Array, ""]
    current_representation_revision_words: UInt[Array, " 2"]
    next_representation_revision_words: UInt[Array, " 2"]
    action_identity_words: UInt[Array, " 2"]
    pre_update_words: UInt[Array, " 2"]
    post_update_words: UInt[Array, " 2"]
    pre_head_revision_words: UInt[Array, " 2"]
    post_head_revision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    receipt_identity_valid: Bool[Array, ""]
    representation_revision_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class BalancedStateObjectivesResourceBudget:
    """Exact persistent-array accounting and one-update execution bound."""

    schema: str
    parameter_nbytes: int
    pending_cache_nbytes: int
    clock_and_revision_nbytes: int
    total_state_nbytes: int
    max_head_updates_per_transition: int
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class BalancedStateObjectivesScanResult:
    """Fixed-shape scan trace over explicit continuing transitions."""

    state: BalancedStateObjectivesState
    balanced_losses: Float[Array, " steps"]
    current_representation_gradients: Float[Array, "steps representation"]
    next_representation_gradients: Float[Array, "steps representation"]
    cache_applied: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]
    action_identity_words: UInt[Array, "steps 2"]


def _empty_pending(state: BalancedStateObjectivesState) -> BalancedStateObjectivesState:
    return dataclasses.replace(  # type: ignore[type-var]
        state,
        pending_representation=jnp.zeros_like(state.pending_representation),
        pending_action=jnp.asarray(-1, dtype=jnp.int32),
        pending_representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_action_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_valid=jnp.asarray(False, dtype=jnp.bool_),
    )


def _receipt_from_state(state: BalancedStateObjectivesState) -> StateObjectiveActionReceipt:
    return StateObjectiveActionReceipt(  # type: ignore[call-arg]
        representation=state.pending_representation,
        action=state.pending_action,
        representation_revision_words=state.pending_representation_revision_words,
        action_identity_words=state.pending_action_identity_words,
    )


def measure_balanced_state_objectives_state_nbytes(
    state: BalancedStateObjectivesState,
) -> int:
    """Return exact bytes occupied by every persistent JAX-array leaf."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class BalancedStateObjectives:
    """Separate multi-timescale GVF/inverse heads with causal ownership."""

    def __init__(self, config: BalancedStateObjectivesConfig) -> None:
        if type(config) is not BalancedStateObjectivesConfig:
            raise TypeError("config must be an exact BalancedStateObjectivesConfig")
        self._config = config
        self._discounts = jnp.asarray(config.gvf_discounts, dtype=jnp.float32)

    @property
    def config(self) -> BalancedStateObjectivesConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> BalancedStateObjectives:
        return cls(BalancedStateObjectivesConfig.from_config(payload))

    def init(self, key: Array) -> BalancedStateObjectivesState:
        """Initialize separate heads from one scalar typed Threefry key."""

        _require_threefry_key(key, label="key")
        cfg = self._config
        keys = jr.split(key, 4)
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)
        gvf = scale * jr.normal(
            keys[1],
            (cfg.n_gvf_heads, cfg.representation_dim),
            dtype=jnp.float32,
        )
        inverse_current = scale * jr.normal(
            keys[2],
            (cfg.n_actions, cfg.representation_dim),
            dtype=jnp.float32,
        )
        inverse_next = scale * jr.normal(
            keys[3],
            (cfg.n_actions, cfg.representation_dim),
            dtype=jnp.float32,
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return BalancedStateObjectivesState(  # type: ignore[call-arg]
            gvf_weights=gvf,
            inverse_current_weights=inverse_current,
            inverse_next_weights=inverse_next,
            inverse_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            pending_representation=jnp.zeros((cfg.representation_dim,), dtype=jnp.float32),
            pending_action=jnp.asarray(-1, dtype=jnp.int32),
            pending_representation_revision_words=zero_words,
            pending_action_identity_words=zero_words,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            decision_words=zero_words,
            update_words=zero_words,
            head_revision_words=zero_words,
        )

    def _require_state_contract(self, state: BalancedStateObjectivesState) -> None:
        if type(state) is not BalancedStateObjectivesState:
            raise TypeError("state must be an exact BalancedStateObjectivesState")
        cfg = self._config
        float_contracts = {
            "gvf_weights": ((cfg.n_gvf_heads, cfg.representation_dim), state.gvf_weights),
            "inverse_current_weights": (
                (cfg.n_actions, cfg.representation_dim),
                state.inverse_current_weights,
            ),
            "inverse_next_weights": (
                (cfg.n_actions, cfg.representation_dim),
                state.inverse_next_weights,
            ),
            "inverse_bias": ((cfg.n_actions,), state.inverse_bias),
            "pending_representation": (
                (cfg.representation_dim,),
                state.pending_representation,
            ),
        }
        for label, (shape, value) in float_contracts.items():
            _require_array(
                value,
                label=label,
                shape=shape,
                dtype=jnp.dtype(jnp.float32),
            )
        _require_int32_scalar(state.pending_action, label="pending_action")
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        for label in (
            "pending_representation_revision_words",
            "pending_action_identity_words",
            "decision_words",
            "update_words",
            "head_revision_words",
        ):
            _require_words(getattr(state, label), label=label)

    def _dynamic_state_valid(self, state: BalancedStateObjectivesState) -> Bool[Array, ""]:
        cfg = self._config
        finite = (
            jnp.all(jnp.isfinite(state.gvf_weights))
            & jnp.all(jnp.isfinite(state.inverse_current_weights))
            & jnp.all(jnp.isfinite(state.inverse_next_weights))
            & jnp.all(jnp.isfinite(state.inverse_bias))
            & jnp.all(jnp.isfinite(state.pending_representation))
        )
        pending_filled = (
            (state.pending_action >= 0)
            & (state.pending_action < cfg.n_actions)
            & jnp.all(
                jnp.abs(state.pending_representation)
                <= jnp.float32(cfg.max_abs_representation)
            )
            & jnp.all(state.pending_action_identity_words == state.decision_words)
        )
        pending_empty = (
            (state.pending_action == -1)
            & jnp.all(state.pending_representation == 0.0)
            & jnp.all(state.pending_representation_revision_words == 0)
            & jnp.all(state.pending_action_identity_words == 0)
        )
        expected_decision, update_has_successor = _increment_words(state.update_words)
        partition_valid = jnp.where(
            state.pending_valid,
            update_has_successor & jnp.all(state.decision_words == expected_decision),
            jnp.all(state.decision_words == state.update_words),
        )
        return (
            finite
            & jnp.all(state.head_revision_words == state.update_words)
            & partition_valid
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
        )

    def state_valid(self, state: BalancedStateObjectivesState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def cache_action(
        self,
        state: BalancedStateObjectivesState,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
    ) -> StateObjectiveActionCacheResult:
        """Bind one executed action to the exact representation that owned it."""

        self._require_state_contract(state)
        representation = _require_float32_vector(
            representation,
            self._config.representation_dim,
            label="representation",
        )
        action = _require_int32_scalar(action, label="action")
        representation_revision_words = _require_words(
            representation_revision_words,
            label="representation_revision_words",
        )
        return cast(
            StateObjectiveActionCacheResult,
            self._cache_action_jit(
                state,
                representation,
                action,
                representation_revision_words,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _cache_action_jit(
        self,
        state: BalancedStateObjectivesState,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
    ) -> StateObjectiveActionCacheResult:
        cfg = self._config
        state_valid = self._dynamic_state_valid(state)
        source_valid = (
            jnp.all(jnp.isfinite(representation))
            & jnp.all(jnp.abs(representation) <= jnp.float32(cfg.max_abs_representation))
            & (action >= 0)
            & (action < cfg.n_actions)
        )
        cache_available = ~state.pending_valid
        proposed_words, lifetime_capacity = _increment_words(state.decision_words)
        cache_applied = state_valid & source_valid & cache_available & lifetime_capacity
        candidate = dataclasses.replace(  # type: ignore[type-var]
            state,
            pending_representation=representation,
            pending_action=action,
            pending_representation_revision_words=representation_revision_words,
            pending_action_identity_words=proposed_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            decision_words=proposed_words,
        )
        next_state = jax.lax.cond(cache_applied, lambda: candidate, lambda: state)
        return StateObjectiveActionCacheResult(  # type: ignore[call-arg]
            state=next_state,
            receipt=_receipt_from_state(next_state),
            pre_decision_words=state.decision_words,
            post_decision_words=next_state.decision_words,
            state_valid=state_valid,
            source_valid=source_valid,
            cache_available=cache_available,
            lifetime_capacity_available=lifetime_capacity,
            cache_applied=cache_applied,
        )

    def _require_receipt_contract(self, receipt: StateObjectiveActionReceipt) -> None:
        if type(receipt) is not StateObjectiveActionReceipt:
            raise TypeError("receipt must be an exact StateObjectiveActionReceipt")
        _require_float32_vector(
            receipt.representation,
            self._config.representation_dim,
            label="receipt.representation",
        )
        _require_int32_scalar(receipt.action, label="receipt.action")
        _require_words(
            receipt.representation_revision_words,
            label="receipt.representation_revision_words",
        )
        _require_words(receipt.action_identity_words, label="receipt.action_identity_words")

    def update(
        self,
        state: BalancedStateObjectivesState,
        receipt: StateObjectiveActionReceipt,
        next_representation: Array,
        next_representation_revision_words: Array,
        cumulant: Array,
        continuation: Array,
    ) -> BalancedStateObjectiveUpdateResult:
        """Train separate heads and emit owner-bound balanced gradients."""

        self._require_state_contract(state)
        self._require_receipt_contract(receipt)
        next_representation = _require_float32_vector(
            next_representation,
            self._config.representation_dim,
            label="next_representation",
        )
        next_representation_revision_words = _require_words(
            next_representation_revision_words,
            label="next_representation_revision_words",
        )
        cumulant = _require_float32_scalar(cumulant, label="cumulant")
        continuation = _require_float32_scalar(continuation, label="continuation")
        return cast(
            BalancedStateObjectiveUpdateResult,
            self._update_jit(
                state,
                receipt,
                next_representation,
                next_representation_revision_words,
                cumulant,
                continuation,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: BalancedStateObjectivesState,
        receipt: StateObjectiveActionReceipt,
        next_representation: Array,
        next_representation_revision_words: Array,
        cumulant: Array,
        continuation: Array,
    ) -> BalancedStateObjectiveUpdateResult:
        cfg = self._config
        state_valid = self._dynamic_state_valid(state)
        receipt_identity_valid = (
            state.pending_valid
            & _float_bits_equal(receipt.representation, state.pending_representation)
            & (receipt.action == state.pending_action)
            & jnp.all(
                receipt.representation_revision_words
                == state.pending_representation_revision_words
            )
            & jnp.all(receipt.action_identity_words == state.pending_action_identity_words)
        )
        representation_revision_valid = _words_not_earlier(
            next_representation_revision_words,
            receipt.representation_revision_words,
        )
        source_valid = (
            jnp.all(jnp.isfinite(receipt.representation))
            & jnp.all(
                jnp.abs(receipt.representation)
                <= jnp.float32(cfg.max_abs_representation)
            )
            & jnp.all(jnp.isfinite(next_representation))
            & jnp.all(
                jnp.abs(next_representation) <= jnp.float32(cfg.max_abs_representation)
            )
            & jnp.isfinite(cumulant)
            & (jnp.abs(cumulant) <= jnp.float32(cfg.max_abs_cumulant))
            & jnp.isfinite(continuation)
            & (continuation >= 0.0)
            & (continuation <= 1.0)
            & (receipt.action >= 0)
            & (receipt.action < cfg.n_actions)
        )

        safe_current = jnp.clip(
            jnp.nan_to_num(receipt.representation),
            -jnp.float32(cfg.max_abs_representation),
            jnp.float32(cfg.max_abs_representation),
        )
        safe_next = jnp.clip(
            jnp.nan_to_num(next_representation),
            -jnp.float32(cfg.max_abs_representation),
            jnp.float32(cfg.max_abs_representation),
        )
        safe_cumulant = jnp.clip(
            jnp.nan_to_num(cumulant),
            -jnp.float32(cfg.max_abs_cumulant),
            jnp.float32(cfg.max_abs_cumulant),
        )
        safe_continuation = jnp.clip(jnp.nan_to_num(continuation), 0.0, 1.0)
        safe_action = jnp.clip(receipt.action, 0, cfg.n_actions - 1)

        gvf_predictions = state.gvf_weights @ safe_current
        gvf_next_predictions = state.gvf_weights @ safe_next
        gvf_targets = safe_cumulant + safe_continuation * self._discounts * jax.lax.stop_gradient(
            gvf_next_predictions
        )
        gvf_errors = gvf_predictions - gvf_targets
        n_heads = jnp.float32(cfg.n_gvf_heads)
        gvf_loss_raw = jnp.float32(0.5) * jnp.mean(jnp.square(gvf_errors))
        gvf_current_gradient = jnp.mean(
            gvf_errors[:, None] * state.gvf_weights,
            axis=0,
        )
        gvf_weight_gradient = gvf_errors[:, None] * safe_current[None, :] / n_heads

        inverse_logits = (
            state.inverse_current_weights @ safe_current
            + state.inverse_next_weights @ safe_next
            + state.inverse_bias
        )
        inverse_log_probabilities = jax.nn.log_softmax(inverse_logits)
        inverse_probabilities = jnp.exp(inverse_log_probabilities)
        action_one_hot = jax.nn.one_hot(safe_action, cfg.n_actions, dtype=jnp.float32)
        inverse_error = inverse_probabilities - action_one_hot
        inverse_loss_raw = -inverse_log_probabilities[safe_action]
        inverse_current_gradient = state.inverse_current_weights.T @ inverse_error
        inverse_next_gradient = state.inverse_next_weights.T @ inverse_error
        inverse_current_weight_gradient = inverse_error[:, None] * safe_current[None, :]
        inverse_next_weight_gradient = inverse_error[:, None] * safe_next[None, :]

        gvf_group_weight = jnp.float32(cfg.gvf_group_weight)
        inverse_group_weight = jnp.float32(cfg.inverse_group_weight)
        raw_current_gradient = (
            gvf_group_weight * gvf_current_gradient
            + inverse_group_weight * inverse_current_gradient
        )
        raw_next_gradient = inverse_group_weight * inverse_next_gradient
        current_norm, clipped_current_gradient = _safe_norm_and_clip(
            raw_current_gradient,
            cfg.representation_gradient_clip,
        )
        next_norm, clipped_next_gradient = _safe_norm_and_clip(
            raw_next_gradient,
            cfg.representation_gradient_clip,
        )
        balanced_loss_raw = (
            gvf_group_weight * gvf_loss_raw + inverse_group_weight * inverse_loss_raw
        )

        candidate_with_pending = dataclasses.replace(  # type: ignore[type-var]
            state,
            gvf_weights=(
                state.gvf_weights - jnp.float32(cfg.gvf_step_size) * gvf_weight_gradient
            ),
            inverse_current_weights=(
                state.inverse_current_weights
                - jnp.float32(cfg.inverse_step_size) * inverse_current_weight_gradient
            ),
            inverse_next_weights=(
                state.inverse_next_weights
                - jnp.float32(cfg.inverse_step_size) * inverse_next_weight_gradient
            ),
            inverse_bias=(
                state.inverse_bias - jnp.float32(cfg.inverse_step_size) * inverse_error
            ),
        )
        proposed_update_words, update_capacity = _increment_words(state.update_words)
        proposed_head_revision, head_capacity = _increment_words(state.head_revision_words)
        candidate = dataclasses.replace(  # type: ignore[type-var]
            _empty_pending(candidate_with_pending),
            update_words=proposed_update_words,
            head_revision_words=proposed_head_revision,
        )
        candidate_state_valid = self._dynamic_state_valid(candidate)
        numeric_candidate_valid = (
            jnp.all(jnp.isfinite(gvf_predictions))
            & jnp.all(jnp.isfinite(gvf_targets))
            & jnp.all(jnp.isfinite(gvf_errors))
            & jnp.all(jnp.isfinite(inverse_probabilities))
            & jnp.isfinite(gvf_loss_raw)
            & jnp.isfinite(inverse_loss_raw)
            & jnp.isfinite(balanced_loss_raw)
            & jnp.all(jnp.isfinite(clipped_current_gradient))
            & jnp.all(jnp.isfinite(clipped_next_gradient))
        )
        lifetime_capacity = update_capacity & head_capacity
        update_applied = (
            state_valid
            & source_valid
            & receipt_identity_valid
            & representation_revision_valid
            & lifetime_capacity
            & candidate_state_valid
            & numeric_candidate_valid
        )
        next_state = jax.lax.cond(update_applied, lambda: candidate, lambda: state)
        # Representation gradients are actionable outputs, not merely
        # observations.  Fail closed with the same atomic verdict as the head
        # transaction so a corrupt/exhausted source cannot update a builder
        # while these owned heads remain unchanged.
        valid_diagnostics = update_applied
        zero_current = jnp.zeros_like(clipped_current_gradient)
        zero_next = jnp.zeros_like(clipped_next_gradient)
        current_gradient = jnp.where(valid_diagnostics, clipped_current_gradient, zero_current)
        next_gradient = jnp.where(valid_diagnostics, clipped_next_gradient, zero_next)
        return BalancedStateObjectiveUpdateResult(  # type: ignore[call-arg]
            state=next_state,
            gvf_predictions=jnp.nan_to_num(gvf_predictions),
            gvf_targets=jnp.nan_to_num(gvf_targets),
            gvf_td_errors=jnp.nan_to_num(-gvf_errors),
            inverse_probabilities=jnp.nan_to_num(inverse_probabilities),
            gvf_loss=_finite_or_max(gvf_loss_raw),
            inverse_loss=_finite_or_max(inverse_loss_raw),
            balanced_loss=_finite_or_max(balanced_loss_raw),
            gvf_current_representation_gradient=jnp.where(
                valid_diagnostics,
                gvf_current_gradient,
                zero_current,
            ),
            inverse_current_representation_gradient=jnp.where(
                valid_diagnostics,
                inverse_current_gradient,
                zero_current,
            ),
            inverse_next_representation_gradient=jnp.where(
                valid_diagnostics,
                inverse_next_gradient,
                zero_next,
            ),
            current_representation_gradient=current_gradient,
            next_representation_gradient=next_gradient,
            unclipped_current_gradient_norm=_finite_or_max(current_norm),
            unclipped_next_gradient_norm=_finite_or_max(next_norm),
            current_gradient_was_clipped=(
                current_norm > jnp.float32(cfg.representation_gradient_clip)
            ),
            next_gradient_was_clipped=(
                next_norm > jnp.float32(cfg.representation_gradient_clip)
            ),
            current_representation_revision_words=receipt.representation_revision_words,
            next_representation_revision_words=next_representation_revision_words,
            action_identity_words=receipt.action_identity_words,
            pre_update_words=state.update_words,
            post_update_words=next_state.update_words,
            pre_head_revision_words=state.head_revision_words,
            post_head_revision_words=next_state.head_revision_words,
            state_valid=state_valid,
            source_valid=source_valid,
            receipt_identity_valid=receipt_identity_valid,
            representation_revision_valid=representation_revision_valid,
            lifetime_capacity_available=lifetime_capacity,
            candidate_state_valid=candidate_state_valid & numeric_candidate_valid,
            update_applied=update_applied,
        )

    def resource_budget(
        self,
        state: BalancedStateObjectivesState | None = None,
    ) -> BalancedStateObjectivesResourceBudget:
        """Return exact logical bytes for the configured fixed-shape state."""

        cfg = self._config
        parameter_nbytes = 4 * (
            cfg.n_gvf_heads * cfg.representation_dim
            + 2 * cfg.n_actions * cfg.representation_dim
            + cfg.n_actions
        )
        pending_cache_nbytes = 4 * cfg.representation_dim + 4 + 8 + 8 + 1
        clock_and_revision_nbytes = 3 * 8
        budget = BalancedStateObjectivesResourceBudget(
            schema=BALANCED_STATE_OBJECTIVES_RESOURCE_SCHEMA,
            parameter_nbytes=parameter_nbytes,
            pending_cache_nbytes=pending_cache_nbytes,
            clock_and_revision_nbytes=clock_and_revision_nbytes,
            total_state_nbytes=(
                parameter_nbytes + pending_cache_nbytes + clock_and_revision_nbytes
            ),
            max_head_updates_per_transition=1,
            temporary_bytes_scope=(
                "source-level-named-arrays; excludes-compiler-and-xla-workspaces; "
                "not-a-measured-device-peak"
            ),
        )
        if state is not None:
            self._require_state_contract(state)
            if measure_balanced_state_objectives_state_nbytes(state) != budget.total_state_nbytes:
                raise ValueError("balanced state objectives state allocation differs from config")
        return budget


def run_balanced_state_objectives_scan(
    objectives: BalancedStateObjectives,
    state: BalancedStateObjectivesState,
    current_representations: Array,
    next_representations: Array,
    actions: Array,
    cumulants: Array,
    continuations: Array,
    current_representation_revision_words: Array,
    next_representation_revision_words: Array,
) -> BalancedStateObjectivesScanResult:
    """Run cache/update transactions over fixed-shape continuing transitions."""

    if type(objectives) is not BalancedStateObjectives:
        raise TypeError("objectives must be an exact BalancedStateObjectives")
    objectives._require_state_contract(state)
    cfg = objectives.config
    if getattr(current_representations, "ndim", None) != 2:
        raise ValueError("current_representations must have rank two")
    steps = current_representations.shape[0]
    contracts = {
        "current_representations": (
            current_representations,
            (steps, cfg.representation_dim),
            jnp.dtype(jnp.float32),
        ),
        "next_representations": (
            next_representations,
            (steps, cfg.representation_dim),
            jnp.dtype(jnp.float32),
        ),
        "actions": (actions, (steps,), jnp.dtype(jnp.int32)),
        "cumulants": (cumulants, (steps,), jnp.dtype(jnp.float32)),
        "continuations": (continuations, (steps,), jnp.dtype(jnp.float32)),
        "current_representation_revision_words": (
            current_representation_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
        "next_representation_revision_words": (
            next_representation_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
    }
    for label, (value, shape, dtype) in contracts.items():
        _require_array(value, label=label, shape=shape, dtype=dtype)

    def body(
        carry: BalancedStateObjectivesState,
        inputs: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[BalancedStateObjectivesState, tuple[Array, ...]]:
        current, successor, action, cumulant, continuation, current_revision, next_revision = (
            inputs
        )
        cached = objectives.cache_action(carry, current, action, current_revision)
        # A failed cache must never let a still-pending older receipt consume
        # the new event's successor.  Materialize an invalid empty receipt on
        # that branch so :meth:`update` preserves the pending owner atomically.
        scan_receipt = StateObjectiveActionReceipt(  # type: ignore[call-arg]
            representation=jnp.where(
                cached.cache_applied,
                cached.receipt.representation,
                jnp.zeros_like(cached.receipt.representation),
            ),
            action=jnp.where(
                cached.cache_applied,
                cached.receipt.action,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            representation_revision_words=jnp.where(
                cached.cache_applied,
                cached.receipt.representation_revision_words,
                jnp.zeros_like(cached.receipt.representation_revision_words),
            ),
            action_identity_words=jnp.where(
                cached.cache_applied,
                cached.receipt.action_identity_words,
                jnp.zeros_like(cached.receipt.action_identity_words),
            ),
        )
        updated = objectives.update(
            cached.state,
            scan_receipt,
            successor,
            next_revision,
            cumulant,
            continuation,
        )
        return updated.state, (
            updated.balanced_loss,
            updated.current_representation_gradient,
            updated.next_representation_gradient,
            cached.cache_applied,
            updated.update_applied,
            updated.action_identity_words,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            current_representations,
            next_representations,
            actions,
            cumulants,
            continuations,
            current_representation_revision_words,
            next_representation_revision_words,
        ),
    )
    losses, current_gradients, next_gradients, cached, updated, identities = outputs
    return BalancedStateObjectivesScanResult(  # type: ignore[call-arg]
        state=final_state,
        balanced_losses=losses,
        current_representation_gradients=current_gradients,
        next_representation_gradients=next_gradients,
        cache_applied=cached,
        update_applied=updated,
        action_identity_words=identities,
    )


def save_balanced_state_objectives_checkpoint(
    objectives: BalancedStateObjectives,
    state: BalancedStateObjectivesState,
    path: str | Path,
) -> None:
    """Persist exact state with strict L0/config/resource metadata."""

    objectives._require_state_contract(state)
    if not bool(objectives.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid balanced state objectives state")
    config = objectives.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": BALANCED_STATE_OBJECTIVES_CHECKPOINT_SCHEMA,
            "evidence_level": BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": BALANCED_STATE_OBJECTIVES_OWNERSHIP,
            "objectives_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": objectives.resource_budget(state).to_config(),
        },
    )


def load_balanced_state_objectives_checkpoint(
    path: str | Path,
) -> tuple[BalancedStateObjectives, BalancedStateObjectivesState]:
    """Restore only a canonical, resource-consistent L0 checkpoint."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "evidence_level",
        "outcome_status",
        "ownership",
        "objectives_config",
        "config_sha256",
        "resource_budget",
    }
    fields = _exact_manifest(metadata, expected, label="balanced state objectives checkpoint")
    if fields["schema"] != BALANCED_STATE_OBJECTIVES_CHECKPOINT_SCHEMA:
        raise ValueError("balanced state objectives checkpoint schema is unsupported")
    if fields["evidence_level"] != BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL:
        raise ValueError("balanced state objectives checkpoint must remain L0")
    if fields["outcome_status"] != BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS:
        raise ValueError("balanced state objectives outcome must remain not_assessed")
    if fields["ownership"] != BALANCED_STATE_OBJECTIVES_OWNERSHIP:
        raise ValueError("balanced state objectives ownership contract differs")
    config = fields["objectives_config"]
    if type(config) is not dict:
        raise TypeError("balanced state objectives checkpoint config must be a dict")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("balanced state objectives checkpoint config digest differs")
    objectives = BalancedStateObjectives.from_config(config)
    if objectives.to_config() != config:
        raise ValueError("balanced state objectives checkpoint config is noncanonical")
    template = objectives.init(jr.key(0))
    expected_budget = objectives.resource_budget(template).to_config()
    if fields["resource_budget"] != expected_budget:
        raise ValueError("balanced state objectives checkpoint resource budget differs")
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("balanced state objectives checkpoint metadata changed between reads")
    state = cast(BalancedStateObjectivesState, restored)
    objectives._require_state_contract(state)
    if not bool(objectives.state_valid(state)):
        raise ValueError("restored balanced state objectives state is invalid")
    objectives.resource_budget(state)
    return objectives, state


__all__ = [
    "BALANCED_STATE_OBJECTIVES_CHECKPOINT_SCHEMA",
    "BALANCED_STATE_OBJECTIVES_CONFIG_SCHEMA",
    "BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL",
    "BALANCED_STATE_OBJECTIVES_LIFETIME_SEMANTICS",
    "BALANCED_STATE_OBJECTIVES_LIMITATIONS",
    "BALANCED_STATE_OBJECTIVES_MAX_DECISIONS",
    "BALANCED_STATE_OBJECTIVES_MAX_UPDATES",
    "BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS",
    "BALANCED_STATE_OBJECTIVES_OWNERSHIP",
    "BALANCED_STATE_OBJECTIVES_RESOURCE_SCHEMA",
    "BALANCED_STATE_OBJECTIVES_STATE_SCHEMA",
    "BalancedStateObjectiveUpdateResult",
    "BalancedStateObjectives",
    "BalancedStateObjectivesConfig",
    "BalancedStateObjectivesResourceBudget",
    "BalancedStateObjectivesScanResult",
    "BalancedStateObjectivesState",
    "StateObjectiveActionCacheResult",
    "StateObjectiveActionReceipt",
    "load_balanced_state_objectives_checkpoint",
    "measure_balanced_state_objectives_state_nbytes",
    "run_balanced_state_objectives_scan",
    "save_balanced_state_objectives_checkpoint",
]
