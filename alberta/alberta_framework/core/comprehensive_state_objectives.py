"""Bounded comprehensive auxiliary objectives for learnable state.

This isolated WP3 kernel implements the objective families named in section
3.3 of ``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md`` without changing the
ordinary Prototype path.  It owns separate linear heads for action-conditional
next-observation and next-latent prediction, reward, termination,
multiple-timescale GVFs, control value, selected-action advantage, and inverse
action classification.

The prediction pair and control pair are averaged inside their families, and
the GVFs are averaged across timescales, before six fixed group masses are
applied.  Target dimensions and head counts therefore cannot silently increase
a family's share of the representation gradient.  Every parameter head has
its own array, step size, and exact revision row.

An action must first be cached with the bit-exact representation and caller-
owned representation revision that selected it.  An update consumes only that
receipt, a non-earlier successor revision, and bounded transition targets.
Invalid sources, stale/tampered receipts, corrupt state, numerical failure, or
counter exhaustion preserve the complete pending transaction bit-for-bit and
emit zero actionable representation gradients.

This is an L0, nonpromoting, ``not_assessed`` mechanism.  The heads are linear,
targets and objective masses are caller-declared rather than calibrated, the
latent regression has no target network or anti-collapse mechanism, and the
control targets have no off-policy correction.  There is no Prototype
integration, feature lifecycle, retention result, Forager result, Alberta Plan
completion, or SOTA evidence here.
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

COMPREHENSIVE_STATE_OBJECTIVES_CONFIG_SCHEMA = (
    "alberta.comprehensive-state-objectives-config.v1"
)
COMPREHENSIVE_STATE_OBJECTIVES_STATE_SCHEMA = (
    "alberta.comprehensive-state-objectives-state.v1"
)
COMPREHENSIVE_STATE_OBJECTIVES_CHECKPOINT_SCHEMA = (
    "alberta.comprehensive-state-objectives-checkpoint.v1"
)
COMPREHENSIVE_STATE_OBJECTIVES_RESOURCE_SCHEMA = (
    "alberta.comprehensive-state-objectives-resource.v1"
)
COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL = "L0"
COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS = "not_assessed"
COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP = (
    "cache-executed-action; bit-exact-representation-receipt; "
    "caller-owned-monotone-representation-revisions"
)
COMPREHENSIVE_STATE_OBJECTIVES_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
COMPREHENSIVE_STATE_OBJECTIVES_MAX_DECISIONS = 2**64 - 1
COMPREHENSIVE_STATE_OBJECTIVES_MAX_UPDATES = 2**64 - 1
COMPREHENSIVE_STATE_OBJECTIVES_HEADS = (
    "next_observation",
    "next_latent",
    "reward",
    "termination",
    "multi_timescale_gvf",
    "control_value",
    "selected_action_advantage",
    "inverse_action",
)
COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS = (
    "linear-one-step-heads",
    "caller-declared-supervision-and-fixed-not-calibrated-group-masses",
    "latent-regression-without-target-network-or-anti-collapse-mechanism",
    "caller-supplied-control-targets-without-off-policy-correction",
    "no-prototype-integration-or-feature-lifecycle",
    "no-retention-forager-alberta-plan-or-sota-evidence",
)

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_N_PARAMETER_HEADS = len(COMPREHENSIVE_STATE_OBJECTIVES_HEADS)


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
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
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.float32))


def _require_int32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.int32))


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.bool_))


def _require_words(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(2,), dtype=jnp.dtype(jnp.uint32))


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
    scale = jnp.max(jnp.abs(value))
    safe_scale = jnp.where(scale > 0.0, scale, jnp.float32(1.0))
    scaled_norm = jnp.sqrt(jnp.sum(jnp.square(value / safe_scale)))
    norm = scale * scaled_norm
    norm = jnp.nan_to_num(norm, nan=_FLOAT32_MAX, posinf=_FLOAT32_MAX)
    safe_norm = jnp.where(norm > 0.0, norm, jnp.float32(1.0))
    factor = jnp.minimum(jnp.float32(1.0), jnp.float32(limit) / safe_norm)
    return norm, value * factor


def _finite_or_max(value: Array) -> Array:
    return jnp.nan_to_num(
        value,
        nan=_FLOAT32_MAX,
        posinf=_FLOAT32_MAX,
        neginf=_FLOAT32_MAX,
    )


@dataclasses.dataclass(frozen=True)
class ComprehensiveStateObjectivesConfig:
    """Static heads, learning rates, group masses, and numerical bounds."""

    representation_dim: int
    observation_target_dim: int
    n_actions: int
    gvf_discounts: tuple[float, ...] = (0.5, 0.9, 0.99)
    observation_step_size: float = 0.01
    latent_step_size: float = 0.01
    reward_step_size: float = 0.01
    termination_step_size: float = 0.01
    gvf_step_size: float = 0.01
    value_step_size: float = 0.01
    advantage_step_size: float = 0.01
    inverse_step_size: float = 0.01
    prediction_group_weight: float = 1.0 / 6.0
    reward_group_weight: float = 1.0 / 6.0
    termination_group_weight: float = 1.0 / 6.0
    gvf_group_weight: float = 1.0 / 6.0
    control_group_weight: float = 1.0 / 6.0
    inverse_group_weight: float = 1.0 / 6.0
    initialization_scale: float = 0.05
    representation_gradient_clip: float = 100.0
    max_abs_representation: float = 1.0e6
    max_abs_observation_target: float = 1.0e6
    max_abs_reward_target: float = 1.0e6
    max_abs_cumulant: float = 1.0e6
    max_abs_control_target: float = 1.0e6

    def __post_init__(self) -> None:
        _exact_int(self.representation_dim, label="representation_dim", minimum=1)
        _exact_int(self.observation_target_dim, label="observation_target_dim", minimum=1)
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
        for name in (
            "observation_step_size",
            "latent_step_size",
            "reward_step_size",
            "termination_step_size",
            "gvf_step_size",
            "value_step_size",
            "advantage_step_size",
            "inverse_step_size",
        ):
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )
        weight_names = (
            "prediction_group_weight",
            "reward_group_weight",
            "termination_group_weight",
            "gvf_group_weight",
            "control_group_weight",
            "inverse_group_weight",
        )
        weights = tuple(
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )
            for name in weight_names
        )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("objective group weights must sum to one")
        for name in (
            "initialization_scale",
            "representation_gradient_clip",
            "max_abs_representation",
            "max_abs_observation_target",
            "max_abs_reward_target",
            "max_abs_cumulant",
            "max_abs_control_target",
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
            "type": "ComprehensiveStateObjectives",
            "schema": COMPREHENSIVE_STATE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": COMPREHENSIVE_STATE_OBJECTIVES_STATE_SCHEMA,
            "evidence_level": COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP,
            "heads": list(COMPREHENSIVE_STATE_OBJECTIVES_HEADS),
            "limitations": list(COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS),
            "representation_dim": self.representation_dim,
            "observation_target_dim": self.observation_target_dim,
            "n_actions": self.n_actions,
            "gvf_discounts": [float(value) for value in self.gvf_discounts],
            "observation_step_size": float(self.observation_step_size),
            "latent_step_size": float(self.latent_step_size),
            "reward_step_size": float(self.reward_step_size),
            "termination_step_size": float(self.termination_step_size),
            "gvf_step_size": float(self.gvf_step_size),
            "value_step_size": float(self.value_step_size),
            "advantage_step_size": float(self.advantage_step_size),
            "inverse_step_size": float(self.inverse_step_size),
            "prediction_group_weight": float(self.prediction_group_weight),
            "reward_group_weight": float(self.reward_group_weight),
            "termination_group_weight": float(self.termination_group_weight),
            "gvf_group_weight": float(self.gvf_group_weight),
            "control_group_weight": float(self.control_group_weight),
            "inverse_group_weight": float(self.inverse_group_weight),
            "initialization_scale": float(self.initialization_scale),
            "representation_gradient_clip": float(self.representation_gradient_clip),
            "max_abs_representation": float(self.max_abs_representation),
            "max_abs_observation_target": float(self.max_abs_observation_target),
            "max_abs_reward_target": float(self.max_abs_reward_target),
            "max_abs_cumulant": float(self.max_abs_cumulant),
            "max_abs_control_target": float(self.max_abs_control_target),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> ComprehensiveStateObjectivesConfig:
        fields = _exact_manifest(
            payload,
            {
                "type",
                "schema",
                "state_schema",
                "evidence_level",
                "outcome_status",
                "ownership",
                "heads",
                "limitations",
                "representation_dim",
                "observation_target_dim",
                "n_actions",
                "gvf_discounts",
                "observation_step_size",
                "latent_step_size",
                "reward_step_size",
                "termination_step_size",
                "gvf_step_size",
                "value_step_size",
                "advantage_step_size",
                "inverse_step_size",
                "prediction_group_weight",
                "reward_group_weight",
                "termination_group_weight",
                "gvf_group_weight",
                "control_group_weight",
                "inverse_group_weight",
                "initialization_scale",
                "representation_gradient_clip",
                "max_abs_representation",
                "max_abs_observation_target",
                "max_abs_reward_target",
                "max_abs_cumulant",
                "max_abs_control_target",
            },
            label="comprehensive state objectives config",
        )
        checks = {
            "type": "ComprehensiveStateObjectives",
            "schema": COMPREHENSIVE_STATE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": COMPREHENSIVE_STATE_OBJECTIVES_STATE_SCHEMA,
            "evidence_level": COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP,
            "heads": list(COMPREHENSIVE_STATE_OBJECTIVES_HEADS),
            "limitations": list(COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS),
        }
        for name, expected in checks.items():
            if fields.pop(name) != expected:
                raise ValueError(f"comprehensive state objectives {name} is unsupported")
        discounts = fields.get("gvf_discounts")
        if type(discounts) is not list:
            raise TypeError("serialized gvf_discounts must be a list")
        fields["gvf_discounts"] = tuple(discounts)
        return cls(**fields)


@chex.dataclass(frozen=True)
class ComprehensiveStateObjectivesState:
    """All separate heads, one pending action owner, and exact clocks."""

    observation_weights: Float[Array, "action observation representation"]
    observation_bias: Float[Array, "action observation"]
    latent_weights: Float[Array, "action latent representation"]
    latent_bias: Float[Array, "action latent"]
    reward_weights: Float[Array, "action representation"]
    reward_bias: Float[Array, " action"]
    termination_weights: Float[Array, "action representation"]
    termination_bias: Float[Array, " action"]
    gvf_weights: Float[Array, "timescale representation"]
    value_weights: Float[Array, " representation"]
    value_bias: Float[Array, ""]
    advantage_weights: Float[Array, "action representation"]
    advantage_bias: Float[Array, " action"]
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
    head_revision_words: UInt[Array, "head 2"]


@chex.dataclass(frozen=True)
class ComprehensiveStateObjectiveActionReceipt:
    """Exact transient owner of one environment action."""

    representation: Float[Array, " representation"]
    action: Int[Array, ""]
    representation_revision_words: UInt[Array, " 2"]
    action_identity_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ComprehensiveStateObjectiveCacheResult:
    """Atomic result of caching an executed action owner."""

    state: ComprehensiveStateObjectivesState
    receipt: ComprehensiveStateObjectiveActionReceipt
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    cache_available: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    cache_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ComprehensiveStateObjectiveUpdateResult:
    """All head diagnostics and one atomic representation/head transaction."""

    state: ComprehensiveStateObjectivesState
    observation_prediction: Float[Array, " observation"]
    latent_prediction: Float[Array, " latent"]
    reward_prediction: Float[Array, ""]
    termination_probability: Float[Array, ""]
    gvf_predictions: Float[Array, " timescale"]
    gvf_targets: Float[Array, " timescale"]
    value_prediction: Float[Array, ""]
    advantage_prediction: Float[Array, ""]
    inverse_probabilities: Float[Array, " action"]
    observation_loss: Float[Array, ""]
    latent_loss: Float[Array, ""]
    reward_loss: Float[Array, ""]
    termination_loss: Float[Array, ""]
    gvf_loss: Float[Array, ""]
    value_loss: Float[Array, ""]
    advantage_loss: Float[Array, ""]
    inverse_loss: Float[Array, ""]
    prediction_group_loss: Float[Array, ""]
    control_group_loss: Float[Array, ""]
    balanced_loss: Float[Array, ""]
    prediction_current_representation_gradient: Float[Array, " representation"]
    prediction_next_representation_gradient: Float[Array, " representation"]
    reward_current_representation_gradient: Float[Array, " representation"]
    termination_current_representation_gradient: Float[Array, " representation"]
    gvf_current_representation_gradient: Float[Array, " representation"]
    control_current_representation_gradient: Float[Array, " representation"]
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
    pre_head_revision_words: UInt[Array, "head 2"]
    post_head_revision_words: UInt[Array, "head 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    receipt_identity_valid: Bool[Array, ""]
    representation_revision_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class ComprehensiveStateObjectivesResourceBudget:
    """Exact persistent allocation split by independently updated head."""

    schema: str
    observation_head_nbytes: int
    latent_head_nbytes: int
    reward_head_nbytes: int
    termination_head_nbytes: int
    gvf_head_nbytes: int
    value_head_nbytes: int
    advantage_head_nbytes: int
    inverse_head_nbytes: int
    parameter_nbytes: int
    pending_cache_nbytes: int
    clock_and_revision_nbytes: int
    total_state_nbytes: int
    max_parameter_head_updates_per_transition: int
    max_atomic_transactions_per_transition: int
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ComprehensiveStateObjectivesScanResult:
    """Fixed-shape scan trace over explicit continuing transitions."""

    state: ComprehensiveStateObjectivesState
    balanced_losses: Float[Array, " steps"]
    current_representation_gradients: Float[Array, "steps representation"]
    next_representation_gradients: Float[Array, "steps representation"]
    cache_applied: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]
    action_identity_words: UInt[Array, "steps 2"]


def _empty_pending(
    state: ComprehensiveStateObjectivesState,
) -> ComprehensiveStateObjectivesState:
    return dataclasses.replace(  # type: ignore[type-var]
        state,
        pending_representation=jnp.zeros_like(state.pending_representation),
        pending_action=jnp.asarray(-1, dtype=jnp.int32),
        pending_representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_action_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_valid=jnp.asarray(False, dtype=jnp.bool_),
    )


def _receipt_from_state(
    state: ComprehensiveStateObjectivesState,
) -> ComprehensiveStateObjectiveActionReceipt:
    return ComprehensiveStateObjectiveActionReceipt(  # type: ignore[call-arg]
        representation=state.pending_representation,
        action=state.pending_action,
        representation_revision_words=state.pending_representation_revision_words,
        action_identity_words=state.pending_action_identity_words,
    )


def measure_comprehensive_state_objectives_state_nbytes(
    state: ComprehensiveStateObjectivesState,
) -> int:
    """Return exact bytes occupied by every persistent JAX-array leaf."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class ComprehensiveStateObjectives:
    """Separate comprehensive heads with causal ownership and fixed masses."""

    def __init__(self, config: ComprehensiveStateObjectivesConfig) -> None:
        if type(config) is not ComprehensiveStateObjectivesConfig:
            raise TypeError("config must be an exact ComprehensiveStateObjectivesConfig")
        self._config = config
        self._discounts = jnp.asarray(config.gvf_discounts, dtype=jnp.float32)

    @property
    def config(self) -> ComprehensiveStateObjectivesConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> ComprehensiveStateObjectives:
        return cls(ComprehensiveStateObjectivesConfig.from_config(payload))

    def init(self, key: Array) -> ComprehensiveStateObjectivesState:
        """Initialize every head from one typed Threefry key."""

        _require_threefry_key(key, label="key")
        cfg = self._config
        keys = jr.split(key, 10)
        scale = jnp.float32(cfg.initialization_scale)
        normal = functools.partial(jr.normal, dtype=jnp.float32)
        zeros_words = jnp.zeros((2,), dtype=jnp.uint32)
        return ComprehensiveStateObjectivesState(  # type: ignore[call-arg]
            observation_weights=scale
            * normal(
                keys[1],
                (cfg.n_actions, cfg.observation_target_dim, cfg.representation_dim),
            ),
            observation_bias=jnp.zeros(
                (cfg.n_actions, cfg.observation_target_dim), dtype=jnp.float32
            ),
            latent_weights=scale
            * normal(
                keys[2],
                (cfg.n_actions, cfg.representation_dim, cfg.representation_dim),
            ),
            latent_bias=jnp.zeros(
                (cfg.n_actions, cfg.representation_dim), dtype=jnp.float32
            ),
            reward_weights=scale
            * normal(keys[3], (cfg.n_actions, cfg.representation_dim)),
            reward_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            termination_weights=scale
            * normal(keys[4], (cfg.n_actions, cfg.representation_dim)),
            termination_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            gvf_weights=scale
            * normal(keys[5], (cfg.n_gvf_heads, cfg.representation_dim)),
            value_weights=scale * normal(keys[6], (cfg.representation_dim,)),
            value_bias=jnp.asarray(0.0, dtype=jnp.float32),
            advantage_weights=scale
            * normal(keys[7], (cfg.n_actions, cfg.representation_dim)),
            advantage_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            inverse_current_weights=scale
            * normal(keys[8], (cfg.n_actions, cfg.representation_dim)),
            inverse_next_weights=scale
            * normal(keys[9], (cfg.n_actions, cfg.representation_dim)),
            inverse_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            pending_representation=jnp.zeros(
                (cfg.representation_dim,), dtype=jnp.float32
            ),
            pending_action=jnp.asarray(-1, dtype=jnp.int32),
            pending_representation_revision_words=zeros_words,
            pending_action_identity_words=zeros_words,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            decision_words=zeros_words,
            update_words=zeros_words,
            head_revision_words=jnp.zeros((_N_PARAMETER_HEADS, 2), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: ComprehensiveStateObjectivesState) -> None:
        if type(state) is not ComprehensiveStateObjectivesState:
            raise TypeError("state must be an exact ComprehensiveStateObjectivesState")
        cfg = self._config
        contracts = {
            "observation_weights": (
                (cfg.n_actions, cfg.observation_target_dim, cfg.representation_dim),
                state.observation_weights,
            ),
            "observation_bias": (
                (cfg.n_actions, cfg.observation_target_dim),
                state.observation_bias,
            ),
            "latent_weights": (
                (cfg.n_actions, cfg.representation_dim, cfg.representation_dim),
                state.latent_weights,
            ),
            "latent_bias": (
                (cfg.n_actions, cfg.representation_dim),
                state.latent_bias,
            ),
            "reward_weights": (
                (cfg.n_actions, cfg.representation_dim),
                state.reward_weights,
            ),
            "reward_bias": ((cfg.n_actions,), state.reward_bias),
            "termination_weights": (
                (cfg.n_actions, cfg.representation_dim),
                state.termination_weights,
            ),
            "termination_bias": ((cfg.n_actions,), state.termination_bias),
            "gvf_weights": (
                (cfg.n_gvf_heads, cfg.representation_dim),
                state.gvf_weights,
            ),
            "value_weights": ((cfg.representation_dim,), state.value_weights),
            "value_bias": ((), state.value_bias),
            "advantage_weights": (
                (cfg.n_actions, cfg.representation_dim),
                state.advantage_weights,
            ),
            "advantage_bias": ((cfg.n_actions,), state.advantage_bias),
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
        for label, (shape, value) in contracts.items():
            _require_array(value, label=label, shape=shape, dtype=jnp.dtype(jnp.float32))
        _require_int32_scalar(state.pending_action, label="pending_action")
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        for label in (
            "pending_representation_revision_words",
            "pending_action_identity_words",
            "decision_words",
            "update_words",
        ):
            _require_words(getattr(state, label), label=label)
        _require_array(
            state.head_revision_words,
            label="head_revision_words",
            shape=(_N_PARAMETER_HEADS, 2),
            dtype=jnp.dtype(jnp.uint32),
        )

    def _dynamic_state_valid(
        self, state: ComprehensiveStateObjectivesState
    ) -> Bool[Array, ""]:
        cfg = self._config
        parameter_leaves = (
            state.observation_weights,
            state.observation_bias,
            state.latent_weights,
            state.latent_bias,
            state.reward_weights,
            state.reward_bias,
            state.termination_weights,
            state.termination_bias,
            state.gvf_weights,
            state.value_weights,
            state.value_bias,
            state.advantage_weights,
            state.advantage_bias,
            state.inverse_current_weights,
            state.inverse_next_weights,
            state.inverse_bias,
        )
        finite = functools.reduce(
            jnp.logical_and,
            (jnp.all(jnp.isfinite(value)) for value in parameter_leaves),
            jnp.asarray(True, dtype=jnp.bool_),
        ) & jnp.all(jnp.isfinite(state.pending_representation))
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
            & jnp.all(state.head_revision_words == state.update_words[None, :])
            & partition_valid
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
        )

    def state_valid(self, state: ComprehensiveStateObjectivesState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def cache_action(
        self,
        state: ComprehensiveStateObjectivesState,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
    ) -> ComprehensiveStateObjectiveCacheResult:
        """Bind the exact representation and revision that selected an action."""

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
            ComprehensiveStateObjectiveCacheResult,
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
        state: ComprehensiveStateObjectivesState,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
    ) -> ComprehensiveStateObjectiveCacheResult:
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
        return ComprehensiveStateObjectiveCacheResult(  # type: ignore[call-arg]
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

    def _require_receipt_contract(
        self, receipt: ComprehensiveStateObjectiveActionReceipt
    ) -> None:
        if type(receipt) is not ComprehensiveStateObjectiveActionReceipt:
            raise TypeError(
                "receipt must be an exact ComprehensiveStateObjectiveActionReceipt"
            )
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
        state: ComprehensiveStateObjectivesState,
        receipt: ComprehensiveStateObjectiveActionReceipt,
        next_representation: Array,
        next_representation_revision_words: Array,
        next_observation_target: Array,
        reward_target: Array,
        terminated_target: Array,
        cumulant: Array,
        continuation: Array,
        control_value_target: Array,
        advantage_target: Array,
    ) -> ComprehensiveStateObjectiveUpdateResult:
        """Update all separate heads and return owner-bound balanced gradients."""

        self._require_state_contract(state)
        self._require_receipt_contract(receipt)
        cfg = self._config
        next_representation = _require_float32_vector(
            next_representation,
            cfg.representation_dim,
            label="next_representation",
        )
        next_representation_revision_words = _require_words(
            next_representation_revision_words,
            label="next_representation_revision_words",
        )
        next_observation_target = _require_float32_vector(
            next_observation_target,
            cfg.observation_target_dim,
            label="next_observation_target",
        )
        reward_target = _require_float32_scalar(reward_target, label="reward_target")
        terminated_target = _require_bool_scalar(
            terminated_target, label="terminated_target"
        )
        cumulant = _require_float32_scalar(cumulant, label="cumulant")
        continuation = _require_float32_scalar(continuation, label="continuation")
        control_value_target = _require_float32_scalar(
            control_value_target, label="control_value_target"
        )
        advantage_target = _require_float32_scalar(
            advantage_target, label="advantage_target"
        )
        return cast(
            ComprehensiveStateObjectiveUpdateResult,
            self._update_jit(
                state,
                receipt,
                next_representation,
                next_representation_revision_words,
                next_observation_target,
                reward_target,
                terminated_target,
                cumulant,
                continuation,
                control_value_target,
                advantage_target,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: ComprehensiveStateObjectivesState,
        receipt: ComprehensiveStateObjectiveActionReceipt,
        next_representation: Array,
        next_representation_revision_words: Array,
        next_observation_target: Array,
        reward_target: Array,
        terminated_target: Array,
        cumulant: Array,
        continuation: Array,
        control_value_target: Array,
        advantage_target: Array,
    ) -> ComprehensiveStateObjectiveUpdateResult:
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
            & jnp.all(jnp.isfinite(next_observation_target))
            & jnp.all(
                jnp.abs(next_observation_target)
                <= jnp.float32(cfg.max_abs_observation_target)
            )
            & jnp.isfinite(reward_target)
            & (jnp.abs(reward_target) <= jnp.float32(cfg.max_abs_reward_target))
            & jnp.isfinite(cumulant)
            & (jnp.abs(cumulant) <= jnp.float32(cfg.max_abs_cumulant))
            & jnp.isfinite(continuation)
            & (continuation >= 0.0)
            & (continuation <= 1.0)
            & jnp.isfinite(control_value_target)
            & (jnp.abs(control_value_target) <= jnp.float32(cfg.max_abs_control_target))
            & jnp.isfinite(advantage_target)
            & (jnp.abs(advantage_target) <= jnp.float32(cfg.max_abs_control_target))
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
        safe_observation = jnp.clip(
            jnp.nan_to_num(next_observation_target),
            -jnp.float32(cfg.max_abs_observation_target),
            jnp.float32(cfg.max_abs_observation_target),
        )
        safe_reward = jnp.clip(
            jnp.nan_to_num(reward_target),
            -jnp.float32(cfg.max_abs_reward_target),
            jnp.float32(cfg.max_abs_reward_target),
        )
        safe_cumulant = jnp.clip(
            jnp.nan_to_num(cumulant),
            -jnp.float32(cfg.max_abs_cumulant),
            jnp.float32(cfg.max_abs_cumulant),
        )
        safe_continuation = jnp.clip(jnp.nan_to_num(continuation), 0.0, 1.0)
        safe_value_target = jnp.clip(
            jnp.nan_to_num(control_value_target),
            -jnp.float32(cfg.max_abs_control_target),
            jnp.float32(cfg.max_abs_control_target),
        )
        safe_advantage_target = jnp.clip(
            jnp.nan_to_num(advantage_target),
            -jnp.float32(cfg.max_abs_control_target),
            jnp.float32(cfg.max_abs_control_target),
        )
        safe_terminated = terminated_target.astype(jnp.float32)
        safe_action = jnp.clip(receipt.action, 0, cfg.n_actions - 1)

        observation_weights = state.observation_weights[safe_action]
        observation_prediction = (
            observation_weights @ safe_current + state.observation_bias[safe_action]
        )
        observation_error = observation_prediction - safe_observation
        observation_denominator = jnp.float32(cfg.observation_target_dim)
        observation_loss = jnp.float32(0.5) * jnp.mean(jnp.square(observation_error))
        observation_current_gradient = (
            observation_weights.T @ observation_error / observation_denominator
        )
        observation_weight_gradient = (
            observation_error[:, None] * safe_current[None, :] / observation_denominator
        )
        observation_bias_gradient = observation_error / observation_denominator

        latent_weights = state.latent_weights[safe_action]
        latent_prediction = latent_weights @ safe_current + state.latent_bias[safe_action]
        latent_error = latent_prediction - safe_next
        latent_denominator = jnp.float32(cfg.representation_dim)
        latent_loss = jnp.float32(0.5) * jnp.mean(jnp.square(latent_error))
        latent_current_gradient = latent_weights.T @ latent_error / latent_denominator
        latent_next_gradient = -latent_error / latent_denominator
        latent_weight_gradient = (
            latent_error[:, None] * safe_current[None, :] / latent_denominator
        )
        latent_bias_gradient = latent_error / latent_denominator

        reward_prediction = (
            state.reward_weights[safe_action] @ safe_current
            + state.reward_bias[safe_action]
        )
        reward_error = reward_prediction - safe_reward
        reward_loss = jnp.float32(0.5) * jnp.square(reward_error)
        reward_current_gradient = reward_error * state.reward_weights[safe_action]
        reward_weight_gradient = reward_error * safe_current

        termination_logit = (
            state.termination_weights[safe_action] @ safe_current
            + state.termination_bias[safe_action]
        )
        termination_probability = jax.nn.sigmoid(termination_logit)
        termination_error = termination_probability - safe_terminated
        termination_loss = jax.nn.softplus(termination_logit) - (
            safe_terminated * termination_logit
        )
        termination_current_gradient = (
            termination_error * state.termination_weights[safe_action]
        )
        termination_weight_gradient = termination_error * safe_current

        gvf_predictions = state.gvf_weights @ safe_current
        gvf_next_predictions = state.gvf_weights @ safe_next
        gvf_targets = safe_cumulant + safe_continuation * self._discounts * (
            jax.lax.stop_gradient(gvf_next_predictions)
        )
        gvf_error = gvf_predictions - gvf_targets
        gvf_denominator = jnp.float32(cfg.n_gvf_heads)
        gvf_loss = jnp.float32(0.5) * jnp.mean(jnp.square(gvf_error))
        gvf_current_gradient = jnp.mean(
            gvf_error[:, None] * state.gvf_weights,
            axis=0,
        )
        gvf_weight_gradient = (
            gvf_error[:, None] * safe_current[None, :] / gvf_denominator
        )

        value_prediction = state.value_weights @ safe_current + state.value_bias
        value_error = value_prediction - safe_value_target
        value_loss = jnp.float32(0.5) * jnp.square(value_error)
        value_current_gradient = value_error * state.value_weights
        value_weight_gradient = value_error * safe_current

        advantage_prediction = (
            state.advantage_weights[safe_action] @ safe_current
            + state.advantage_bias[safe_action]
        )
        advantage_error = advantage_prediction - safe_advantage_target
        advantage_loss = jnp.float32(0.5) * jnp.square(advantage_error)
        advantage_current_gradient = (
            advantage_error * state.advantage_weights[safe_action]
        )
        advantage_weight_gradient = advantage_error * safe_current

        inverse_logits = (
            state.inverse_current_weights @ safe_current
            + state.inverse_next_weights @ safe_next
            + state.inverse_bias
        )
        inverse_log_probabilities = jax.nn.log_softmax(inverse_logits)
        inverse_probabilities = jnp.exp(inverse_log_probabilities)
        action_one_hot = jax.nn.one_hot(safe_action, cfg.n_actions, dtype=jnp.float32)
        inverse_error = inverse_probabilities - action_one_hot
        inverse_loss = -inverse_log_probabilities[safe_action]
        inverse_current_gradient = state.inverse_current_weights.T @ inverse_error
        inverse_next_gradient = state.inverse_next_weights.T @ inverse_error
        inverse_current_weight_gradient = inverse_error[:, None] * safe_current[None, :]
        inverse_next_weight_gradient = inverse_error[:, None] * safe_next[None, :]

        prediction_loss = jnp.float32(0.5) * (observation_loss + latent_loss)
        prediction_current_gradient = jnp.float32(0.5) * (
            observation_current_gradient + latent_current_gradient
        )
        prediction_next_gradient = jnp.float32(0.5) * latent_next_gradient
        control_loss = jnp.float32(0.5) * (value_loss + advantage_loss)
        control_current_gradient = jnp.float32(0.5) * (
            value_current_gradient + advantage_current_gradient
        )

        prediction_mass = jnp.float32(cfg.prediction_group_weight)
        reward_mass = jnp.float32(cfg.reward_group_weight)
        termination_mass = jnp.float32(cfg.termination_group_weight)
        gvf_mass = jnp.float32(cfg.gvf_group_weight)
        control_mass = jnp.float32(cfg.control_group_weight)
        inverse_mass = jnp.float32(cfg.inverse_group_weight)
        raw_current_gradient = (
            prediction_mass * prediction_current_gradient
            + reward_mass * reward_current_gradient
            + termination_mass * termination_current_gradient
            + gvf_mass * gvf_current_gradient
            + control_mass * control_current_gradient
            + inverse_mass * inverse_current_gradient
        )
        raw_next_gradient = (
            prediction_mass * prediction_next_gradient
            + inverse_mass * inverse_next_gradient
        )
        current_norm, clipped_current_gradient = _safe_norm_and_clip(
            raw_current_gradient,
            cfg.representation_gradient_clip,
        )
        next_norm, clipped_next_gradient = _safe_norm_and_clip(
            raw_next_gradient,
            cfg.representation_gradient_clip,
        )
        balanced_loss = (
            prediction_mass * prediction_loss
            + reward_mass * reward_loss
            + termination_mass * termination_loss
            + gvf_mass * gvf_loss
            + control_mass * control_loss
            + inverse_mass * inverse_loss
        )

        action_mask = action_one_hot
        observation_mask = action_mask[:, None, None]
        observation_bias_mask = action_mask[:, None]
        vector_mask = action_mask[:, None]
        candidate_with_pending = dataclasses.replace(  # type: ignore[type-var]
            state,
            observation_weights=(
                state.observation_weights
                - jnp.float32(cfg.observation_step_size)
                * observation_mask
                * observation_weight_gradient[None, :, :]
            ),
            observation_bias=(
                state.observation_bias
                - jnp.float32(cfg.observation_step_size)
                * observation_bias_mask
                * observation_bias_gradient[None, :]
            ),
            latent_weights=(
                state.latent_weights
                - jnp.float32(cfg.latent_step_size)
                * observation_mask
                * latent_weight_gradient[None, :, :]
            ),
            latent_bias=(
                state.latent_bias
                - jnp.float32(cfg.latent_step_size)
                * observation_bias_mask
                * latent_bias_gradient[None, :]
            ),
            reward_weights=(
                state.reward_weights
                - jnp.float32(cfg.reward_step_size)
                * vector_mask
                * reward_weight_gradient[None, :]
            ),
            reward_bias=(
                state.reward_bias
                - jnp.float32(cfg.reward_step_size) * action_mask * reward_error
            ),
            termination_weights=(
                state.termination_weights
                - jnp.float32(cfg.termination_step_size)
                * vector_mask
                * termination_weight_gradient[None, :]
            ),
            termination_bias=(
                state.termination_bias
                - jnp.float32(cfg.termination_step_size)
                * action_mask
                * termination_error
            ),
            gvf_weights=(
                state.gvf_weights
                - jnp.float32(cfg.gvf_step_size) * gvf_weight_gradient
            ),
            value_weights=(
                state.value_weights
                - jnp.float32(cfg.value_step_size) * value_weight_gradient
            ),
            value_bias=(
                state.value_bias - jnp.float32(cfg.value_step_size) * value_error
            ),
            advantage_weights=(
                state.advantage_weights
                - jnp.float32(cfg.advantage_step_size)
                * vector_mask
                * advantage_weight_gradient[None, :]
            ),
            advantage_bias=(
                state.advantage_bias
                - jnp.float32(cfg.advantage_step_size)
                * action_mask
                * advantage_error
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
        proposed_update_words, lifetime_capacity = _increment_words(state.update_words)
        candidate = dataclasses.replace(  # type: ignore[type-var]
            _empty_pending(candidate_with_pending),
            update_words=proposed_update_words,
            head_revision_words=jnp.broadcast_to(
                proposed_update_words[None, :], (_N_PARAMETER_HEADS, 2)
            ),
        )
        candidate_state_valid = self._dynamic_state_valid(candidate)
        diagnostics = (
            observation_prediction,
            latent_prediction,
            reward_prediction,
            termination_probability,
            gvf_predictions,
            gvf_targets,
            value_prediction,
            advantage_prediction,
            inverse_probabilities,
            observation_loss,
            latent_loss,
            reward_loss,
            termination_loss,
            gvf_loss,
            value_loss,
            advantage_loss,
            inverse_loss,
            prediction_loss,
            control_loss,
            balanced_loss,
            clipped_current_gradient,
            clipped_next_gradient,
        )
        numeric_candidate_valid = functools.reduce(
            jnp.logical_and,
            (jnp.all(jnp.isfinite(value)) for value in diagnostics),
            jnp.asarray(True, dtype=jnp.bool_),
        )
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
        zero_current = jnp.zeros_like(clipped_current_gradient)
        zero_next = jnp.zeros_like(clipped_next_gradient)
        valid = update_applied

        return ComprehensiveStateObjectiveUpdateResult(  # type: ignore[call-arg]
            state=next_state,
            observation_prediction=jnp.nan_to_num(observation_prediction),
            latent_prediction=jnp.nan_to_num(latent_prediction),
            reward_prediction=_finite_or_max(reward_prediction),
            termination_probability=jnp.nan_to_num(termination_probability),
            gvf_predictions=jnp.nan_to_num(gvf_predictions),
            gvf_targets=jnp.nan_to_num(gvf_targets),
            value_prediction=_finite_or_max(value_prediction),
            advantage_prediction=_finite_or_max(advantage_prediction),
            inverse_probabilities=jnp.nan_to_num(inverse_probabilities),
            observation_loss=_finite_or_max(observation_loss),
            latent_loss=_finite_or_max(latent_loss),
            reward_loss=_finite_or_max(reward_loss),
            termination_loss=_finite_or_max(termination_loss),
            gvf_loss=_finite_or_max(gvf_loss),
            value_loss=_finite_or_max(value_loss),
            advantage_loss=_finite_or_max(advantage_loss),
            inverse_loss=_finite_or_max(inverse_loss),
            prediction_group_loss=_finite_or_max(prediction_loss),
            control_group_loss=_finite_or_max(control_loss),
            balanced_loss=_finite_or_max(balanced_loss),
            prediction_current_representation_gradient=jnp.where(
                valid, prediction_current_gradient, zero_current
            ),
            prediction_next_representation_gradient=jnp.where(
                valid, prediction_next_gradient, zero_next
            ),
            reward_current_representation_gradient=jnp.where(
                valid, reward_current_gradient, zero_current
            ),
            termination_current_representation_gradient=jnp.where(
                valid, termination_current_gradient, zero_current
            ),
            gvf_current_representation_gradient=jnp.where(
                valid, gvf_current_gradient, zero_current
            ),
            control_current_representation_gradient=jnp.where(
                valid, control_current_gradient, zero_current
            ),
            inverse_current_representation_gradient=jnp.where(
                valid, inverse_current_gradient, zero_current
            ),
            inverse_next_representation_gradient=jnp.where(
                valid, inverse_next_gradient, zero_next
            ),
            current_representation_gradient=jnp.where(
                valid, clipped_current_gradient, zero_current
            ),
            next_representation_gradient=jnp.where(
                valid, clipped_next_gradient, zero_next
            ),
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
        state: ComprehensiveStateObjectivesState | None = None,
    ) -> ComprehensiveStateObjectivesResourceBudget:
        """Return exact logical bytes for every configured persistent array."""

        cfg = self._config
        observation = 4 * cfg.n_actions * cfg.observation_target_dim * (
            cfg.representation_dim + 1
        )
        latent = 4 * cfg.n_actions * cfg.representation_dim * (
            cfg.representation_dim + 1
        )
        reward = 4 * cfg.n_actions * (cfg.representation_dim + 1)
        termination = 4 * cfg.n_actions * (cfg.representation_dim + 1)
        gvf = 4 * cfg.n_gvf_heads * cfg.representation_dim
        value = 4 * (cfg.representation_dim + 1)
        advantage = 4 * cfg.n_actions * (cfg.representation_dim + 1)
        inverse = 4 * cfg.n_actions * (2 * cfg.representation_dim + 1)
        parameter = (
            observation
            + latent
            + reward
            + termination
            + gvf
            + value
            + advantage
            + inverse
        )
        pending = 4 * cfg.representation_dim + 4 + 8 + 8 + 1
        clocks = 8 + 8 + _N_PARAMETER_HEADS * 8
        budget = ComprehensiveStateObjectivesResourceBudget(
            schema=COMPREHENSIVE_STATE_OBJECTIVES_RESOURCE_SCHEMA,
            observation_head_nbytes=observation,
            latent_head_nbytes=latent,
            reward_head_nbytes=reward,
            termination_head_nbytes=termination,
            gvf_head_nbytes=gvf,
            value_head_nbytes=value,
            advantage_head_nbytes=advantage,
            inverse_head_nbytes=inverse,
            parameter_nbytes=parameter,
            pending_cache_nbytes=pending,
            clock_and_revision_nbytes=clocks,
            total_state_nbytes=parameter + pending + clocks,
            max_parameter_head_updates_per_transition=_N_PARAMETER_HEADS,
            max_atomic_transactions_per_transition=1,
            temporary_bytes_scope=(
                "source-level-named-arrays; excludes-compiler-and-xla-workspaces; "
                "not-a-measured-device-peak"
            ),
        )
        if state is not None:
            self._require_state_contract(state)
            if measure_comprehensive_state_objectives_state_nbytes(state) != (
                budget.total_state_nbytes
            ):
                raise ValueError(
                    "comprehensive state objectives state allocation differs from config"
                )
        return budget


def run_comprehensive_state_objectives_scan(
    objectives: ComprehensiveStateObjectives,
    state: ComprehensiveStateObjectivesState,
    current_representations: Array,
    next_representations: Array,
    actions: Array,
    next_observation_targets: Array,
    reward_targets: Array,
    terminated_targets: Array,
    cumulants: Array,
    continuations: Array,
    control_value_targets: Array,
    advantage_targets: Array,
    current_representation_revision_words: Array,
    next_representation_revision_words: Array,
) -> ComprehensiveStateObjectivesScanResult:
    """Run cache/update transactions over a fixed-shape transition sequence."""

    if type(objectives) is not ComprehensiveStateObjectives:
        raise TypeError("objectives must be an exact ComprehensiveStateObjectives")
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
        "next_observation_targets": (
            next_observation_targets,
            (steps, cfg.observation_target_dim),
            jnp.dtype(jnp.float32),
        ),
        "reward_targets": (reward_targets, (steps,), jnp.dtype(jnp.float32)),
        "terminated_targets": (terminated_targets, (steps,), jnp.dtype(jnp.bool_)),
        "cumulants": (cumulants, (steps,), jnp.dtype(jnp.float32)),
        "continuations": (continuations, (steps,), jnp.dtype(jnp.float32)),
        "control_value_targets": (
            control_value_targets,
            (steps,),
            jnp.dtype(jnp.float32),
        ),
        "advantage_targets": (advantage_targets, (steps,), jnp.dtype(jnp.float32)),
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
        carry: ComprehensiveStateObjectivesState,
        inputs: tuple[Array, ...],
    ) -> tuple[ComprehensiveStateObjectivesState, tuple[Array, ...]]:
        (
            current,
            successor,
            action,
            observation,
            reward,
            terminated,
            cumulant,
            continuation,
            value_target,
            advantage_target,
            current_revision,
            next_revision,
        ) = inputs
        cached = objectives.cache_action(carry, current, action, current_revision)
        scan_receipt = ComprehensiveStateObjectiveActionReceipt(  # type: ignore[call-arg]
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
            observation,
            reward,
            terminated,
            cumulant,
            continuation,
            value_target,
            advantage_target,
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
            next_observation_targets,
            reward_targets,
            terminated_targets,
            cumulants,
            continuations,
            control_value_targets,
            advantage_targets,
            current_representation_revision_words,
            next_representation_revision_words,
        ),
    )
    losses, current_gradients, next_gradients, cached, updated, identities = outputs
    return ComprehensiveStateObjectivesScanResult(  # type: ignore[call-arg]
        state=final_state,
        balanced_losses=losses,
        current_representation_gradients=current_gradients,
        next_representation_gradients=next_gradients,
        cache_applied=cached,
        update_applied=updated,
        action_identity_words=identities,
    )


def save_comprehensive_state_objectives_checkpoint(
    objectives: ComprehensiveStateObjectives,
    state: ComprehensiveStateObjectivesState,
    path: str | Path,
) -> None:
    """Persist exact state with strict L0/config/resource metadata."""

    objectives._require_state_contract(state)
    if not bool(objectives.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid comprehensive objectives state")
    config = objectives.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": COMPREHENSIVE_STATE_OBJECTIVES_CHECKPOINT_SCHEMA,
            "evidence_level": COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP,
            "objectives_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": objectives.resource_budget(state).to_config(),
        },
    )


def load_comprehensive_state_objectives_checkpoint(
    path: str | Path,
) -> tuple[ComprehensiveStateObjectives, ComprehensiveStateObjectivesState]:
    """Restore only a canonical, resource-consistent L0 checkpoint."""

    metadata = load_checkpoint_metadata(path)
    fields = _exact_manifest(
        metadata,
        {
            "schema",
            "evidence_level",
            "outcome_status",
            "ownership",
            "objectives_config",
            "config_sha256",
            "resource_budget",
        },
        label="comprehensive state objectives checkpoint",
    )
    if fields["schema"] != COMPREHENSIVE_STATE_OBJECTIVES_CHECKPOINT_SCHEMA:
        raise ValueError("comprehensive state objectives checkpoint schema is unsupported")
    if fields["evidence_level"] != COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL:
        raise ValueError("comprehensive state objectives checkpoint must remain L0")
    if fields["outcome_status"] != COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS:
        raise ValueError("comprehensive state objectives outcome must remain not_assessed")
    if fields["ownership"] != COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP:
        raise ValueError("comprehensive state objectives ownership contract differs")
    config = fields["objectives_config"]
    if type(config) is not dict:
        raise TypeError("comprehensive state objectives checkpoint config must be a dict")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("comprehensive state objectives checkpoint config digest differs")
    objectives = ComprehensiveStateObjectives.from_config(config)
    if objectives.to_config() != config:
        raise ValueError("comprehensive state objectives checkpoint config is noncanonical")
    template = objectives.init(jr.key(0))
    if fields["resource_budget"] != objectives.resource_budget(template).to_config():
        raise ValueError("comprehensive state objectives checkpoint resource budget differs")
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("comprehensive state objectives checkpoint metadata changed between reads")
    state = cast(ComprehensiveStateObjectivesState, restored)
    objectives._require_state_contract(state)
    if not bool(objectives.state_valid(state)):
        raise ValueError("restored comprehensive state objectives state is invalid")
    objectives.resource_budget(state)
    return objectives, state


__all__ = [
    "COMPREHENSIVE_STATE_OBJECTIVES_CHECKPOINT_SCHEMA",
    "COMPREHENSIVE_STATE_OBJECTIVES_CONFIG_SCHEMA",
    "COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL",
    "COMPREHENSIVE_STATE_OBJECTIVES_HEADS",
    "COMPREHENSIVE_STATE_OBJECTIVES_LIFETIME_SEMANTICS",
    "COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS",
    "COMPREHENSIVE_STATE_OBJECTIVES_MAX_DECISIONS",
    "COMPREHENSIVE_STATE_OBJECTIVES_MAX_UPDATES",
    "COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS",
    "COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP",
    "COMPREHENSIVE_STATE_OBJECTIVES_RESOURCE_SCHEMA",
    "COMPREHENSIVE_STATE_OBJECTIVES_STATE_SCHEMA",
    "ComprehensiveStateObjectiveActionReceipt",
    "ComprehensiveStateObjectiveCacheResult",
    "ComprehensiveStateObjectiveUpdateResult",
    "ComprehensiveStateObjectives",
    "ComprehensiveStateObjectivesConfig",
    "ComprehensiveStateObjectivesResourceBudget",
    "ComprehensiveStateObjectivesScanResult",
    "ComprehensiveStateObjectivesState",
    "load_comprehensive_state_objectives_checkpoint",
    "measure_comprehensive_state_objectives_state_nbytes",
    "run_comprehensive_state_objectives_scan",
    "save_comprehensive_state_objectives_checkpoint",
]
