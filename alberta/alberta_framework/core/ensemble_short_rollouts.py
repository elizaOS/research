# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded policy- and uncertainty-directed ensemble rollouts.

This module implements only the L0 mechanism described by items 2 and 3 of
WP4.6 in ``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``.  It reads an exact
``WorldModelEnsemble`` snapshot and a caller-owned immutable linear
policy/value snapshot, then returns fixed-shape imagined training proposals.
It never mutates or dispatches to an actor, critic, model, state builder,
safety envelope, or environment.

Every rollout starts from a caller-asserted real observation carrying an
ordered decision identity and exact source, model, policy, and value revision
words.  The model receipt covers every array word in the supplied ensemble
snapshot, including its child RNG state, so different content cannot alias an
unchanged model revision.  Non-cryptographic integrity tags detect accidental
corruption and revision aliasing; they do not prove that a caller actually
observed the environment.  Policy-directed paths sample the supported policy distribution.
Uncertainty-directed paths choose the greatest ensemble disagreement among
actions that already pass every safety/readiness gate.

Learned member discounts define both continuation and termination.  Members
must agree on terminal versus continuing semantics unless the configuration
explicitly relaxes that veto.  Terminal transitions receive continuation zero,
padding after termination is invalid, and the reverse multi-step return never
bootstraps across a terminal transition.  A horizon-truncated valid path may
bootstrap from the caller-owned value snapshot.

All memory and work are statically bounded.  The rollout lane owns one typed
Threefry key and exact fail-stop uint64 word-pair clocks.  Invalid, corrupt,
stale, or over-capacity calls are atomic no-ops, including RNG and clocks.
Thresholds are caller declarations: this module is not calibrated, not
assessed, and makes no control-benefit, retention, or promotion claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

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
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsemblePrediction,
    WorldModelEnsembleState,
)

ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA = "alberta.ensemble-short-rollout.config.v1"
ENSEMBLE_SHORT_ROLLOUT_CHECKPOINT_SCHEMA = (
    "alberta.ensemble-short-rollout.checkpoint.v1"
)
ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS = "l0-mechanism-only-not-assessed"
ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL = "L0"
ENSEMBLE_SHORT_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED = False

RolloutSelectionMode = Literal["policy_directed", "uncertainty_directed"]

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_MAX_HORIZON = 32
_MAX_ROLLOUT_BUDGET = 64
_MAX_TRANSITIONS_PER_CALL = 1_024
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_SOURCE_TAG_SALT = 0x53524345
_POLICY_TAG_SALT = 0x504F4C59
_VALUE_TAG_SALT = 0x56414C55
_MODEL_TAG_SALT = 0x4D4F444C
_AUTHORITY_TAG_SALT = 0x41555448
_ANCHOR_TAG_SALT = 0x414E4348


def _strict_positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return value


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
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
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
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


def _array_contract(
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


def _tree_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


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


def _words_times_small(words: Array, multiplier: int) -> tuple[Array, Array]:
    """Multiply a uint64 word pair by a small static scalar without x64."""

    multiplier_u = jnp.asarray(multiplier, dtype=jnp.uint32)
    low_lo = words[1] & jnp.asarray(0xFFFF, dtype=jnp.uint32)
    low_hi = words[1] >> jnp.asarray(16, dtype=jnp.uint32)
    product_low = low_lo * multiplier_u
    product_high = low_hi * multiplier_u
    shifted = (product_high & jnp.asarray(0xFFFF, dtype=jnp.uint32)) << jnp.asarray(
        16,
        dtype=jnp.uint32,
    )
    result_low = product_low + shifted
    carry = (result_low < product_low).astype(jnp.uint32)
    result_high = (
        words[0] * multiplier_u
        + (product_high >> jnp.asarray(16, dtype=jnp.uint32))
        + carry
    )
    high_overflow = (words[0] != 0) & (
        (result_high // multiplier_u) != words[0]
    )
    return jnp.stack((result_high, result_low)), ~high_overflow


def _words_leq_limit(words: Array, limit: int) -> Bool[Array, ""]:
    limit_words = jnp.asarray(
        [(limit >> 32) & _UINT32_MAX, limit & _UINT32_MAX],
        dtype=jnp.uint32,
    )
    return _words_less_equal(words, limit_words)


def _mix_words(words: Array, *, salt: int) -> UInt[Array, ""]:
    flat = jnp.ravel(words).astype(jnp.uint32)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ flat[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    tag = jax.lax.fori_loop(
        0,
        flat.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(
        tag == jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(salt, dtype=jnp.uint32),
        tag,
    )


def _float_words(value: Array) -> Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _tree_content_words(tree: object) -> Array:
    """Return canonical uint32 content words for a statically known PyTree."""

    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        if leaf is None:
            continue
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            words = jr.key_data(leaf)
        elif array.dtype == jnp.dtype(jnp.float32):
            words = _float_words(array)
        elif array.dtype in (
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported model-state receipt dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(parts)


def _config_digest(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class EnsembleShortRolloutConfig:
    """Static rollout, guard, work, and lifetime bounds.

    All thresholds are uncalibrated caller declarations.  ``action_support``
    counts are global primitive-action observations supplied by the source;
    they are not local density or an OOD certificate.
    """

    selection_mode: RolloutSelectionMode = "policy_directed"
    rollout_horizon: int = 3
    rollout_budget: int = 2
    min_action_support: int = 1
    min_policy_probability: float = 0.0
    policy_temperature: float = 1.0
    max_epistemic_disagreement: float = 1.0
    max_residual_variance: float = 1.0
    require_residual_proxy_ready: bool = True
    terminal_discount_threshold: float = 0.0
    require_termination_agreement: bool = True
    max_observation_magnitude: float = 1_000.0
    max_abs_reward: float = 1_000.0
    max_abs_value: float = 1_000_000.0
    max_abs_return: float = 1_000_000.0
    max_proposal_calls: int = _INT32_MAX
    max_rollout_attempts: int = _INT32_MAX
    max_imagined_steps: int = _INT32_MAX

    def __post_init__(self) -> None:
        if self.selection_mode not in (
            "policy_directed",
            "uncertainty_directed",
        ):
            raise ValueError("selection_mode is unsupported")
        _strict_positive_int(
            self.rollout_horizon,
            name="rollout_horizon",
            maximum=_MAX_HORIZON,
        )
        _strict_positive_int(
            self.rollout_budget,
            name="rollout_budget",
            maximum=_MAX_ROLLOUT_BUDGET,
        )
        if self.rollout_horizon * self.rollout_budget > _MAX_TRANSITIONS_PER_CALL:
            raise ValueError("rollout_horizon * rollout_budget exceeds the work ceiling")
        for name in (
            "min_action_support",
            "max_proposal_calls",
            "max_rollout_attempts",
            "max_imagined_steps",
        ):
            _strict_positive_int(getattr(self, name), name=name, maximum=_INT32_MAX)
        if self.max_rollout_attempts > self.max_proposal_calls * self.rollout_budget:
            raise ValueError(
                "max_rollout_attempts cannot exceed max_proposal_calls * rollout_budget"
            )
        if self.max_imagined_steps > self.max_rollout_attempts * self.rollout_horizon:
            raise ValueError(
                "max_imagined_steps cannot exceed max_rollout_attempts * rollout_horizon"
            )
        if self.max_imagined_steps < self.rollout_horizon:
            raise ValueError("max_imagined_steps must admit one full rollout")
        if type(self.require_residual_proxy_ready) is not bool:
            raise ValueError("require_residual_proxy_ready must be a strict boolean")
        if type(self.require_termination_agreement) is not bool:
            raise ValueError("require_termination_agreement must be a strict boolean")
        specs = (
            ("min_policy_probability", 0.0, 1.0, False),
            ("policy_temperature", _FLOAT32_TINY, None, True),
            ("max_epistemic_disagreement", 0.0, None, False),
            ("max_residual_variance", 0.0, None, False),
            ("terminal_discount_threshold", 0.0, None, False),
            ("max_observation_magnitude", _FLOAT32_TINY, None, True),
            ("max_abs_reward", _FLOAT32_TINY, None, True),
            ("max_abs_value", _FLOAT32_TINY, None, True),
            ("max_abs_return", _FLOAT32_TINY, None, True),
        )
        for name, minimum, maximum, positive in specs:
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                    strictly_positive=positive,
                ),
            )

    def to_config(self) -> dict[str, object]:
        """Return the canonical JSON-compatible L0 contract."""

        return {
            "schema": ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS,
            "evidence_level": ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": (
                ENSEMBLE_SHORT_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "control_benefit_assessed": False,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> EnsembleShortRolloutConfig:
        """Strictly restore the sole v1 configuration schema."""

        payload = dict(config)
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("ensemble short-rollout config fields do not match v1")
        if payload.pop("schema") != ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA:
            raise ValueError("unsupported ensemble short-rollout config schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected ensemble short-rollout config type")
        if payload.pop("mechanism_status") != ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS:
            raise ValueError("ensemble short rollouts must remain mechanism-only")
        if payload.pop("evidence_level") != ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL:
            raise ValueError("ensemble short-rollout evidence level must remain L0")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("ensemble short rollouts cannot claim promotion")
        if payload.pop("control_benefit_assessed") is not False:
            raise ValueError("control benefit remains not assessed")
        for name in (
            "rollout_horizon",
            "rollout_budget",
            "min_action_support",
            "max_proposal_calls",
            "max_rollout_attempts",
            "max_imagined_steps",
        ):
            if type(payload[name]) is not int:
                raise ValueError(f"serialized {name} must be a JSON integer")
        for name in (
            "min_policy_probability",
            "policy_temperature",
            "max_epistemic_disagreement",
            "max_residual_variance",
            "terminal_discount_threshold",
            "max_observation_magnitude",
            "max_abs_reward",
            "max_abs_value",
            "max_abs_return",
        ):
            if type(payload[name]) is not float:
                raise ValueError(f"serialized {name} must be a canonical JSON float")
        for name in (
            "require_residual_proxy_ready",
            "require_termination_agreement",
        ):
            if type(payload[name]) is not bool:
                raise ValueError(f"serialized {name} must be a JSON boolean")
        if type(payload["selection_mode"]) is not str:
            raise ValueError("serialized selection_mode must be a JSON string")
        restored = cls(**cast(dict[str, Any], payload))
        if restored.to_config() != dict(config):
            raise ValueError("ensemble short-rollout config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class RolloutPolicyValueAuthority:
    """Caller-owned immutable linear policy/value snapshot and provenance."""

    policy_weights: Float[Array, "n_actions observation_dim"]
    policy_bias: Float[Array, " n_actions"]
    value_weights: Float[Array, " observation_dim"]
    value_bias: Float[Array, ""]
    action_support_counts: Int[Array, " n_actions"]
    source_revision_words: UInt[Array, " 2"]
    model_revision_words: UInt[Array, " 2"]
    policy_revision_words: UInt[Array, " 2"]
    value_revision_words: UInt[Array, " 2"]
    source_integrity_tag: UInt[Array, ""]
    model_integrity_tag: UInt[Array, ""]
    policy_integrity_tag: UInt[Array, ""]
    value_integrity_tag: UInt[Array, ""]
    authority_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class RealStateRolloutAnchor:
    """Caller-asserted real observation with exact decision provenance."""

    observation: Float[Array, " observation_dim"]
    decision_id_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    model_revision_words: UInt[Array, " 2"]
    policy_revision_words: UInt[Array, " 2"]
    value_revision_words: UInt[Array, " 2"]
    authority_integrity_tag: UInt[Array, ""]
    model_integrity_tag: UInt[Array, ""]
    anchor_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class EnsembleShortRolloutState:
    """Complete planner-owned state; model and policy/value states are absent."""

    rollout_key: Array
    bound_source_revision_words: UInt[Array, " 2"]
    bound_model_revision_words: UInt[Array, " 2"]
    bound_policy_revision_words: UInt[Array, " 2"]
    bound_value_revision_words: UInt[Array, " 2"]
    bound_source_integrity_tag: UInt[Array, ""]
    bound_model_integrity_tag: UInt[Array, ""]
    bound_policy_integrity_tag: UInt[Array, ""]
    bound_value_integrity_tag: UInt[Array, ""]
    last_decision_id_words: UInt[Array, " 2"]
    proposal_call_count_words: UInt[Array, " 2"]
    rollout_attempt_count_words: UInt[Array, " 2"]
    accepted_rollout_count_words: UInt[Array, " 2"]
    rejected_rollout_count_words: UInt[Array, " 2"]
    imagined_step_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ImaginedRolloutBatch:
    """Fixed-shape, read-only training proposals with no dispatch authority."""

    observations: Float[Array, "rollout_budget rollout_horizon observation_dim"]
    actions: Int[Array, "rollout_budget rollout_horizon"]
    rewards: Float[Array, "rollout_budget rollout_horizon"]
    continuations: Float[Array, "rollout_budget rollout_horizon"]
    next_observations: Float[
        Array, "rollout_budget rollout_horizon observation_dim"
    ]
    return_targets: Float[Array, "rollout_budget rollout_horizon"]
    bootstrap_values: Float[Array, " rollout_budget"]
    root_returns: Float[Array, " rollout_budget"]
    transition_valid: Bool[Array, "rollout_budget rollout_horizon"]
    terminated: Bool[Array, "rollout_budget rollout_horizon"]
    path_accepted: Bool[Array, " rollout_budget"]
    decision_id_words: UInt[Array, "rollout_budget 2"]
    source_revision_words: UInt[Array, "rollout_budget 2"]
    model_revision_words: UInt[Array, "rollout_budget 2"]
    policy_revision_words: UInt[Array, "rollout_budget 2"]
    value_revision_words: UInt[Array, "rollout_budget 2"]
    source_integrity_tags: UInt[Array, " rollout_budget"]
    policy_integrity_tags: UInt[Array, " rollout_budget"]
    value_integrity_tags: UInt[Array, " rollout_budget"]
    authority_integrity_tags: UInt[Array, " rollout_budget"]
    model_integrity_tags: UInt[Array, " rollout_budget"]
    anchor_integrity_tags: UInt[Array, " rollout_budget"]


@chex.dataclass(frozen=True)
class EnsembleShortRolloutDiagnostics:
    """Preflight, per-step guard, return, and exact-clock audit."""

    state_static_contract_valid: Bool[Array, ""]
    model_state_static_contract_valid: Bool[Array, ""]
    authority_static_contract_valid: Bool[Array, ""]
    anchor_static_contract_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    model_state_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    anchor_identity_valid: Bool[Array, ""]
    revisions_monotonic: Bool[Array, ""]
    decision_identity_valid: Bool[Array, ""]
    call_capacity_available: Bool[Array, ""]
    attempt_capacity_available: Bool[Array, ""]
    selected_actions: Int[Array, "rollout_budget rollout_horizon"]
    selected_policy_probabilities: Float[
        Array, "rollout_budget rollout_horizon"
    ]
    epistemic_disagreements: Float[Array, "rollout_budget rollout_horizon"]
    residual_variance_maxima: Float[Array, "rollout_budget rollout_horizon"]
    support_valid: Bool[Array, "rollout_budget rollout_horizon"]
    policy_valid: Bool[Array, "rollout_budget rollout_horizon"]
    model_prediction_valid: Bool[Array, "rollout_budget rollout_horizon"]
    residual_proxy_ready: Bool[Array, "rollout_budget rollout_horizon"]
    epistemic_valid: Bool[Array, "rollout_budget rollout_horizon"]
    residual_variance_valid: Bool[Array, "rollout_budget rollout_horizon"]
    continuation_valid: Bool[Array, "rollout_budget rollout_horizon"]
    termination_agreement: Bool[Array, "rollout_budget rollout_horizon"]
    finite_values_valid: Bool[Array, "rollout_budget rollout_horizon"]
    guard_passed: Bool[Array, "rollout_budget rollout_horizon"]
    path_failed: Bool[Array, " rollout_budget"]
    path_return_valid: Bool[Array, " rollout_budget"]
    transaction_applied: Bool[Array, ""]
    pre_proposal_call_count_words: UInt[Array, " 2"]
    post_proposal_call_count_words: UInt[Array, " 2"]
    pre_rollout_attempt_count_words: UInt[Array, " 2"]
    post_rollout_attempt_count_words: UInt[Array, " 2"]
    pre_accepted_rollout_count_words: UInt[Array, " 2"]
    post_accepted_rollout_count_words: UInt[Array, " 2"]
    pre_rejected_rollout_count_words: UInt[Array, " 2"]
    post_rejected_rollout_count_words: UInt[Array, " 2"]
    pre_imagined_step_count_words: UInt[Array, " 2"]
    post_imagined_step_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class EnsembleShortRolloutResult:
    """Planner state and proposal-only output from one atomic call."""

    state: EnsembleShortRolloutState
    proposals: ImaginedRolloutBatch
    diagnostics: EnsembleShortRolloutDiagnostics


@dataclasses.dataclass(frozen=True)
class EnsembleShortRolloutResourceBudget:
    """Exact logical state/output bytes and source-level work ceilings."""

    persistent_bytes_scope: str
    proposal_bytes_scope: str
    diagnostic_bytes_scope: str
    temporary_bytes_scope: str
    observation_dim: int
    n_actions: int
    ensemble_size: int
    rollout_horizon: int
    rollout_budget: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    proposal_scalars: int
    proposal_bytes: int
    diagnostics_scalars: int
    diagnostics_bytes: int
    rollout_prng_keys: int
    rollout_prng_uint32_scalars: int
    max_ensemble_prediction_calls_per_call: int
    max_member_predictions_per_call: int
    max_policy_forward_calls_per_call: int
    max_value_forward_calls_per_call: int
    max_model_integrity_scalar_reads_per_call: int
    max_rng_splits_per_call: int
    max_rng_draws_per_call: int
    max_proposal_calls: int
    max_rollout_attempts: int
    max_imagined_steps: int
    model_state_owned: int
    policy_value_state_owned: int
    actor_or_critic_updates_per_call: int
    dispatch_authority: int

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class EnsembleShortRolloutPlanner:
    """Proposal-only short-rollout composer over a read-only ensemble."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble,
        config: EnsembleShortRolloutConfig | None = None,
    ) -> None:
        self._ensemble = ensemble
        self._config = config or EnsembleShortRolloutConfig()
        if (
            self._config.terminal_discount_threshold
            > self._ensemble.config.model.gamma
        ):
            raise ValueError(
                "terminal_discount_threshold cannot exceed the model discount bound"
            )
        self._reference_model_state = ensemble.init(
            jr.key(0, impl="threefry2x32")
        )
        self._model_signature = _tree_static_signature(self._reference_model_state)

    @property
    def config(self) -> EnsembleShortRolloutConfig:
        return self._config

    @property
    def ensemble(self) -> WorldModelEnsemble:
        return self._ensemble

    @property
    def observation_dim(self) -> int:
        return self._ensemble.config.model.observation_dim

    @property
    def n_actions(self) -> int:
        return self._ensemble.config.model.n_actions

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "planner": self._config.to_config(),
            "ensemble": self._ensemble.to_config(),
            "model_state_owned": False,
            "policy_value_state_owned": False,
            "dispatch_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> EnsembleShortRolloutPlanner:
        payload = dict(config)
        expected = {
            "type",
            "planner",
            "ensemble",
            "model_state_owned",
            "policy_value_state_owned",
            "dispatch_authority",
            "scientific_promotion_allowed",
        }
        if set(payload) != expected:
            raise ValueError("ensemble short-rollout construction fields do not match v1")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected ensemble short-rollout construction type")
        for name in (
            "model_state_owned",
            "policy_value_state_owned",
            "dispatch_authority",
            "scientific_promotion_allowed",
        ):
            if payload.pop(name) is not False:
                raise ValueError(f"{name} must remain false")
        planner_config = payload.pop("planner")
        ensemble_config = payload.pop("ensemble")
        if not isinstance(planner_config, Mapping) or not isinstance(
            ensemble_config, Mapping
        ):
            raise ValueError("ensemble short-rollout child configs must be mappings")
        instance = cls(
            WorldModelEnsemble.from_config(dict(cast(Mapping[str, Any], ensemble_config))),
            EnsembleShortRolloutConfig.from_config(
                cast(Mapping[str, object], planner_config)
            ),
        )
        if instance.to_config() != dict(config):
            raise ValueError("ensemble short-rollout construction is not canonical")
        return instance

    def _source_tag(self, support: Array, revision: Array) -> UInt[Array, ""]:
        words = jnp.concatenate((revision, support.astype(jnp.uint32)))
        return _mix_words(words, salt=_SOURCE_TAG_SALT)

    def _policy_tag(
        self,
        weights: Array,
        bias: Array,
        revision: Array,
    ) -> UInt[Array, ""]:
        words = jnp.concatenate(
            (revision, jnp.ravel(_float_words(weights)), _float_words(bias))
        )
        return _mix_words(words, salt=_POLICY_TAG_SALT)

    def _value_tag(
        self,
        weights: Array,
        bias: Array,
        revision: Array,
    ) -> UInt[Array, ""]:
        words = jnp.concatenate(
            (
                revision,
                _float_words(weights),
                jnp.reshape(_float_words(bias), (1,)),
            )
        )
        return _mix_words(words, salt=_VALUE_TAG_SALT)

    def _model_tag(
        self,
        model_state: WorldModelEnsembleState,
    ) -> UInt[Array, ""]:
        """Bind every caller-owned ensemble-state word to one receipt."""

        return _mix_words(
            _tree_content_words(model_state),
            salt=_MODEL_TAG_SALT,
        )

    def _authority_tag(
        self,
        source_tag: Array,
        model_tag: Array,
        policy_tag: Array,
        value_tag: Array,
        model_revision: Array,
    ) -> UInt[Array, ""]:
        words = jnp.concatenate(
            (
                jnp.reshape(source_tag, (1,)),
                model_revision,
                jnp.reshape(model_tag, (1,)),
                jnp.reshape(policy_tag, (1,)),
                jnp.reshape(value_tag, (1,)),
            )
        )
        return _mix_words(words, salt=_AUTHORITY_TAG_SALT)

    def _anchor_tag(self, anchor: RealStateRolloutAnchor) -> UInt[Array, ""]:
        words = jnp.concatenate(
            (
                anchor.decision_id_words,
                anchor.source_revision_words,
                anchor.model_revision_words,
                anchor.policy_revision_words,
                anchor.value_revision_words,
                jnp.reshape(anchor.authority_integrity_tag, (1,)),
                jnp.reshape(anchor.model_integrity_tag, (1,)),
                _float_words(anchor.observation),
            )
        )
        return _mix_words(words, salt=_ANCHOR_TAG_SALT)

    def _model_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, WorldModelEnsembleState)
            and _tree_static_signature(state) == self._model_signature
        )

    def _authority_static_valid(self, authority: object) -> bool:
        if not isinstance(authority, RolloutPolicyValueAuthority):
            return False
        contracts = (
            (
                authority.policy_weights,
                (self.n_actions, self.observation_dim),
                jnp.float32,
            ),
            (authority.policy_bias, (self.n_actions,), jnp.float32),
            (authority.value_weights, (self.observation_dim,), jnp.float32),
            (authority.value_bias, (), jnp.float32),
            (authority.action_support_counts, (self.n_actions,), jnp.int32),
            (authority.source_revision_words, (2,), jnp.uint32),
            (authority.model_revision_words, (2,), jnp.uint32),
            (authority.policy_revision_words, (2,), jnp.uint32),
            (authority.value_revision_words, (2,), jnp.uint32),
            (authority.source_integrity_tag, (), jnp.uint32),
            (authority.model_integrity_tag, (), jnp.uint32),
            (authority.policy_integrity_tag, (), jnp.uint32),
            (authority.value_integrity_tag, (), jnp.uint32),
            (authority.authority_integrity_tag, (), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in contracts
        )

    def _anchor_static_valid(self, anchor: object) -> bool:
        if not isinstance(anchor, RealStateRolloutAnchor):
            return False
        contracts = (
            (anchor.observation, (self.observation_dim,), jnp.float32),
            (anchor.decision_id_words, (2,), jnp.uint32),
            (anchor.source_revision_words, (2,), jnp.uint32),
            (anchor.model_revision_words, (2,), jnp.uint32),
            (anchor.policy_revision_words, (2,), jnp.uint32),
            (anchor.value_revision_words, (2,), jnp.uint32),
            (anchor.authority_integrity_tag, (), jnp.uint32),
            (anchor.model_integrity_tag, (), jnp.uint32),
            (anchor.anchor_integrity_tag, (), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in contracts
        )

    def _state_static_valid(self, state: object) -> bool:
        if not isinstance(state, EnsembleShortRolloutState):
            return False
        contracts = (
            (state.bound_source_revision_words, (2,), jnp.uint32),
            (state.bound_model_revision_words, (2,), jnp.uint32),
            (state.bound_policy_revision_words, (2,), jnp.uint32),
            (state.bound_value_revision_words, (2,), jnp.uint32),
            (state.bound_source_integrity_tag, (), jnp.uint32),
            (state.bound_model_integrity_tag, (), jnp.uint32),
            (state.bound_policy_integrity_tag, (), jnp.uint32),
            (state.bound_value_integrity_tag, (), jnp.uint32),
            (state.last_decision_id_words, (2,), jnp.uint32),
            (state.proposal_call_count_words, (2,), jnp.uint32),
            (state.rollout_attempt_count_words, (2,), jnp.uint32),
            (state.accepted_rollout_count_words, (2,), jnp.uint32),
            (state.rejected_rollout_count_words, (2,), jnp.uint32),
            (state.imagined_step_count_words, (2,), jnp.uint32),
        )
        if not all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in contracts
        ):
            return False
        try:
            _typed_threefry_key(state.rollout_key, name="state.rollout_key")
        except TypeError:
            return False
        return True

    def bind_authority(
        self,
        *,
        policy_weights: Array,
        policy_bias: Array,
        value_weights: Array,
        value_bias: Array,
        action_support_counts: Array,
        source_revision_words: Array,
        model_state: WorldModelEnsembleState,
        policy_revision_words: Array,
        value_revision_words: Array,
    ) -> RolloutPolicyValueAuthority:
        """Bind exact caller-owned arrays to deterministic integrity tags."""

        if not self._model_static_valid(model_state):
            raise TypeError("model_state does not match the ensemble static contract")
        if not bool(self._ensemble.state_valid(model_state)):
            raise ValueError("model_state is dynamically invalid")
        model_revision_words = model_state.event_count_words
        provisional = RolloutPolicyValueAuthority(
            policy_weights=policy_weights,
            policy_bias=policy_bias,
            value_weights=value_weights,
            value_bias=value_bias,
            action_support_counts=action_support_counts,
            source_revision_words=source_revision_words,
            model_revision_words=model_revision_words,
            policy_revision_words=policy_revision_words,
            value_revision_words=value_revision_words,
            source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            model_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            policy_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            value_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            authority_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        if not self._authority_static_valid(provisional):
            raise TypeError("authority arrays do not match the strict static contract")
        source_tag = self._source_tag(action_support_counts, source_revision_words)
        model_tag = self._model_tag(model_state)
        policy_tag = self._policy_tag(
            policy_weights,
            policy_bias,
            policy_revision_words,
        )
        value_tag = self._value_tag(
            value_weights,
            value_bias,
            value_revision_words,
        )
        return provisional.replace(
            source_integrity_tag=source_tag,
            model_integrity_tag=model_tag,
            policy_integrity_tag=policy_tag,
            value_integrity_tag=value_tag,
            authority_integrity_tag=self._authority_tag(
                source_tag,
                model_tag,
                policy_tag,
                value_tag,
                model_revision_words,
            ),
        )

    def bind_real_anchor(
        self,
        observation: Array,
        decision_id_words: Array,
        authority: RolloutPolicyValueAuthority,
    ) -> RealStateRolloutAnchor:
        """Bind a caller assertion that ``observation`` came from reality.

        The integrity receipt detects later accidental mutation; it cannot
        independently authenticate the caller's environment observation.
        """

        if not self._authority_static_valid(authority):
            raise TypeError("authority does not match the strict static contract")
        provisional = RealStateRolloutAnchor(
            observation=observation,
            decision_id_words=decision_id_words,
            source_revision_words=authority.source_revision_words,
            model_revision_words=authority.model_revision_words,
            policy_revision_words=authority.policy_revision_words,
            value_revision_words=authority.value_revision_words,
            authority_integrity_tag=authority.authority_integrity_tag,
            model_integrity_tag=authority.model_integrity_tag,
            anchor_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        if not self._anchor_static_valid(provisional):
            raise TypeError("real anchor arrays do not match the strict static contract")
        return provisional.replace(anchor_integrity_tag=self._anchor_tag(provisional))

    def _authority_valid(
        self,
        authority: RolloutPolicyValueAuthority,
        model_state: WorldModelEnsembleState,
    ) -> Bool[Array, ""]:
        source_tag = self._source_tag(
            authority.action_support_counts,
            authority.source_revision_words,
        )
        model_tag = self._model_tag(model_state)
        policy_tag = self._policy_tag(
            authority.policy_weights,
            authority.policy_bias,
            authority.policy_revision_words,
        )
        value_tag = self._value_tag(
            authority.value_weights,
            authority.value_bias,
            authority.value_revision_words,
        )
        authority_tag = self._authority_tag(
            source_tag,
            model_tag,
            policy_tag,
            value_tag,
            authority.model_revision_words,
        )
        bound = jnp.asarray(
            self._config.max_observation_magnitude,
            dtype=jnp.float32,
        )
        return (
            _tree_finite(authority)
            & jnp.all(jnp.abs(authority.policy_weights) <= bound)
            & jnp.all(jnp.abs(authority.policy_bias) <= bound)
            & jnp.all(jnp.abs(authority.value_weights) <= bound)
            & (jnp.abs(authority.value_bias) <= bound)
            & jnp.all(authority.action_support_counts >= 0)
            & _words_nonzero(authority.source_revision_words)
            & _words_nonzero(authority.policy_revision_words)
            & _words_nonzero(authority.value_revision_words)
            & jnp.array_equal(
                authority.model_revision_words,
                model_state.event_count_words,
            )
            & (authority.source_integrity_tag == source_tag)
            & (authority.model_integrity_tag == model_tag)
            & (authority.policy_integrity_tag == policy_tag)
            & (authority.value_integrity_tag == value_tag)
            & (authority.authority_integrity_tag == authority_tag)
        )

    def _anchor_valid(
        self,
        anchor: RealStateRolloutAnchor,
        authority: RolloutPolicyValueAuthority,
    ) -> Bool[Array, ""]:
        bound = jnp.asarray(
            self._config.max_observation_magnitude,
            dtype=jnp.float32,
        )
        return (
            jnp.all(jnp.isfinite(anchor.observation))
            & jnp.all(jnp.abs(anchor.observation) <= bound)
            & _words_nonzero(anchor.decision_id_words)
            & jnp.array_equal(
                anchor.source_revision_words,
                authority.source_revision_words,
            )
            & jnp.array_equal(
                anchor.model_revision_words,
                authority.model_revision_words,
            )
            & jnp.array_equal(
                anchor.policy_revision_words,
                authority.policy_revision_words,
            )
            & jnp.array_equal(
                anchor.value_revision_words,
                authority.value_revision_words,
            )
            & (anchor.authority_integrity_tag == authority.authority_integrity_tag)
            & (anchor.model_integrity_tag == authority.model_integrity_tag)
            & (anchor.anchor_integrity_tag == self._anchor_tag(anchor))
        )

    def _state_valid(self, state: EnsembleShortRolloutState) -> Bool[Array, ""]:
        total, total_ok = _checked_words_add(
            state.accepted_rollout_count_words,
            state.rejected_rollout_count_words,
        )
        attempted_step_words, max_steps_ok = _words_times_small(
            state.rollout_attempt_count_words,
            self._config.rollout_horizon,
        )
        revisions_nonzero = (
            _words_nonzero(state.bound_source_revision_words)
            & _words_nonzero(state.bound_policy_revision_words)
            & _words_nonzero(state.bound_value_revision_words)
        )
        tags_nonzero = (
            (state.bound_source_integrity_tag != 0)
            & (state.bound_model_integrity_tag != 0)
            & (state.bound_policy_integrity_tag != 0)
            & (state.bound_value_integrity_tag != 0)
        )
        return (
            total_ok
            & max_steps_ok
            & revisions_nonzero
            & tags_nonzero
            & jnp.array_equal(total, state.rollout_attempt_count_words)
            & _words_less_equal(
                state.imagined_step_count_words,
                attempted_step_words,
            )
            & _words_leq_limit(
                state.proposal_call_count_words,
                self._config.max_proposal_calls,
            )
            & _words_leq_limit(
                state.rollout_attempt_count_words,
                self._config.max_rollout_attempts,
            )
            & _words_leq_limit(
                state.imagined_step_count_words,
                self._config.max_imagined_steps,
            )
        )

    def state_valid(self, state: EnsembleShortRolloutState) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state does not match the strict static contract")
        return self._state_valid(state)

    def validate_state(self, state: EnsembleShortRolloutState) -> None:
        if not self._state_static_valid(state):
            raise TypeError("state does not match the strict static contract")
        if not bool(self._state_valid(state)):
            raise ValueError("ensemble short-rollout state is dynamically invalid")

    def _empty_state(
        self,
        key: Array,
        authority: RolloutPolicyValueAuthority,
    ) -> EnsembleShortRolloutState:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return EnsembleShortRolloutState(
            rollout_key=key,
            bound_source_revision_words=authority.source_revision_words,
            bound_model_revision_words=authority.model_revision_words,
            bound_policy_revision_words=authority.policy_revision_words,
            bound_value_revision_words=authority.value_revision_words,
            bound_source_integrity_tag=authority.source_integrity_tag,
            bound_model_integrity_tag=authority.model_integrity_tag,
            bound_policy_integrity_tag=authority.policy_integrity_tag,
            bound_value_integrity_tag=authority.value_integrity_tag,
            last_decision_id_words=zero_words,
            proposal_call_count_words=zero_words,
            rollout_attempt_count_words=zero_words,
            accepted_rollout_count_words=zero_words,
            rejected_rollout_count_words=zero_words,
            imagined_step_count_words=zero_words,
        )

    def init(
        self,
        key: Array,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
    ) -> EnsembleShortRolloutState:
        typed_key = _typed_threefry_key(key, name="key")
        if not self._model_static_valid(model_state):
            raise TypeError("model_state does not match the ensemble static contract")
        if not self._authority_static_valid(authority):
            raise TypeError("authority does not match the strict static contract")
        if not bool(self._ensemble.state_valid(model_state)):
            raise ValueError("model_state is dynamically invalid")
        if not bool(self._authority_valid(authority, model_state)):
            raise ValueError("authority is dynamically invalid")
        state = self._empty_state(typed_key, authority)
        self.validate_state(state)
        return state

    def _policy_distribution(
        self,
        authority: RolloutPolicyValueAuthority,
        observation: Array,
    ) -> tuple[Array, Array, Array]:
        logits = (
            authority.policy_weights @ observation + authority.policy_bias
        ) / jnp.asarray(self._config.policy_temperature, dtype=jnp.float32)
        support = authority.action_support_counts >= self._config.min_action_support
        has_support = jnp.any(support)
        safe_logits = jnp.where(
            support,
            logits,
            jnp.asarray(-1.0e30, dtype=jnp.float32),
        )
        probabilities = jax.nn.softmax(safe_logits)
        probabilities = jnp.where(
            has_support,
            probabilities,
            jnp.zeros_like(probabilities).at[0].set(1.0),
        )
        valid = (
            has_support
            & jnp.all(jnp.isfinite(logits))
            & jnp.all(jnp.isfinite(probabilities))
            & jnp.isclose(jnp.sum(probabilities), 1.0, atol=1.0e-5)
        )
        return probabilities, support, valid

    def _prediction_facts(
        self,
        prediction: WorldModelEnsemblePrediction,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        residual_max = jnp.max(prediction.residual_variances)
        residual_ready = prediction.residual_proxy_ready
        if not self._config.require_residual_proxy_ready:
            residual_ready = jnp.asarray(True, dtype=jnp.bool_)
        epistemic_valid = (
            jnp.isfinite(prediction.epistemic_disagreement)
            & (
                prediction.epistemic_disagreement
                <= jnp.asarray(
                    self._config.max_epistemic_disagreement,
                    dtype=jnp.float32,
                )
            )
        )
        residual_valid = (
            jnp.all(jnp.isfinite(prediction.residual_variances))
            & jnp.isfinite(residual_max)
            & (
                residual_max
                <= jnp.asarray(
                    self._config.max_residual_variance,
                    dtype=jnp.float32,
                )
            )
        )
        threshold = jnp.asarray(
            self._config.terminal_discount_threshold,
            dtype=jnp.float32,
        )
        terminal_votes = prediction.member_discounts <= threshold
        all_terminal = jnp.all(terminal_votes)
        all_continuing = jnp.all(~terminal_votes)
        agreement = all_terminal | all_continuing
        if not self._config.require_termination_agreement:
            agreement = jnp.asarray(True, dtype=jnp.bool_)
            all_terminal = prediction.mean_discount <= threshold
        continuation = jnp.where(
            all_terminal,
            jnp.asarray(0.0, dtype=jnp.float32),
            prediction.mean_discount,
        )
        gamma = jnp.asarray(self._ensemble.config.model.gamma, dtype=jnp.float32)
        continuation_valid = (
            jnp.all(jnp.isfinite(prediction.member_discounts))
            & jnp.all(prediction.member_discounts >= 0.0)
            & jnp.all(prediction.member_discounts <= gamma)
            & jnp.isfinite(continuation)
            & (continuation >= 0.0)
            & (continuation <= gamma)
        )
        observation_bound = jnp.asarray(
            self._config.max_observation_magnitude,
            dtype=jnp.float32,
        )
        reward_bound = jnp.asarray(self._config.max_abs_reward, dtype=jnp.float32)
        finite = (
            jnp.all(jnp.isfinite(prediction.member_raw_predictions))
            & jnp.all(jnp.isfinite(prediction.member_next_observations))
            & jnp.all(jnp.abs(prediction.member_next_observations) <= observation_bound)
            & jnp.all(jnp.isfinite(prediction.member_rewards))
            & jnp.all(jnp.abs(prediction.member_rewards) <= reward_bound)
            & jnp.all(jnp.isfinite(prediction.mean_next_observation))
            & jnp.all(jnp.abs(prediction.mean_next_observation) <= observation_bound)
            & jnp.isfinite(prediction.mean_reward)
            & (jnp.abs(prediction.mean_reward) <= reward_bound)
        )
        return (
            residual_max,
            residual_ready,
            epistemic_valid,
            residual_valid,
            continuation,
            continuation_valid,
            agreement,
            finite,
        )

    def _select_prediction(
        self,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
        observation: Array,
        draw_key: Array,
    ) -> tuple[Array, WorldModelEnsemblePrediction, tuple[Array, ...]]:
        probabilities, support_mask, policy_distribution_valid = (
            self._policy_distribution(authority, observation)
        )
        if self._config.selection_mode == "policy_directed":
            sampled = jr.categorical(
                draw_key,
                jnp.log(
                    jnp.maximum(
                        probabilities,
                        jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
                    )
                ),
            ).astype(jnp.int32)
            action = jnp.where(
                policy_distribution_valid,
                sampled,
                jnp.asarray(0, dtype=jnp.int32),
            )
            prediction = self._ensemble.predict(model_state, observation, action)
            facts = self._prediction_facts(prediction)
            selected_probability = probabilities[action]
            support_valid = support_mask[action]
            policy_valid = (
                policy_distribution_valid
                & jnp.isfinite(selected_probability)
                & (
                    selected_probability
                    >= jnp.asarray(
                        self._config.min_policy_probability,
                        dtype=jnp.float32,
                    )
                )
            )
            return action, prediction, (
                selected_probability,
                support_valid,
                policy_valid,
                *facts,
            )

        actions = jnp.arange(self.n_actions, dtype=jnp.int32)
        predictions = jax.vmap(
            lambda action: self._ensemble.predict(model_state, observation, action)
        )(actions)
        facts = jax.vmap(self._prediction_facts)(predictions)
        (
            residual_maxima,
            residual_ready,
            epistemic_valid,
            residual_valid,
            continuations,
            continuation_valid,
            agreements,
            finite,
        ) = facts
        candidate_valid = (
            support_mask
            & predictions.valid
            & residual_ready
            & epistemic_valid
            & residual_valid
            & continuation_valid
            & agreements
            & finite
        )
        scores = jnp.where(
            candidate_valid,
            predictions.epistemic_disagreement,
            jnp.asarray(-jnp.inf, dtype=jnp.float32),
        )
        action = jnp.argmax(scores).astype(jnp.int32)
        any_candidate = jnp.any(candidate_valid)
        action = jnp.where(any_candidate, action, jnp.asarray(0, dtype=jnp.int32))
        prediction = cast(
            WorldModelEnsemblePrediction,
            jax.tree.map(lambda value: value[action], predictions),
        )
        selected_probability = probabilities[action]
        return action, prediction, (
            selected_probability,
            support_mask[action] & any_candidate,
            policy_distribution_valid & any_candidate,
            residual_maxima[action],
            residual_ready[action],
            epistemic_valid[action],
            residual_valid[action],
            continuations[action],
            continuation_valid[action],
            agreements[action],
            finite[action],
        )

    def _zero_proposals(self) -> ImaginedRolloutBatch:
        budget = self._config.rollout_budget
        horizon = self._config.rollout_horizon
        observation_shape = (budget, horizon, self.observation_dim)
        step_shape = (budget, horizon)
        revision_shape = (budget, 2)
        return ImaginedRolloutBatch(
            observations=jnp.zeros(observation_shape, dtype=jnp.float32),
            actions=jnp.zeros(step_shape, dtype=jnp.int32),
            rewards=jnp.zeros(step_shape, dtype=jnp.float32),
            continuations=jnp.zeros(step_shape, dtype=jnp.float32),
            next_observations=jnp.zeros(observation_shape, dtype=jnp.float32),
            return_targets=jnp.zeros(step_shape, dtype=jnp.float32),
            bootstrap_values=jnp.zeros((budget,), dtype=jnp.float32),
            root_returns=jnp.zeros((budget,), dtype=jnp.float32),
            transition_valid=jnp.zeros(step_shape, dtype=jnp.bool_),
            terminated=jnp.zeros(step_shape, dtype=jnp.bool_),
            path_accepted=jnp.zeros((budget,), dtype=jnp.bool_),
            decision_id_words=jnp.zeros(revision_shape, dtype=jnp.uint32),
            source_revision_words=jnp.zeros(revision_shape, dtype=jnp.uint32),
            model_revision_words=jnp.zeros(revision_shape, dtype=jnp.uint32),
            policy_revision_words=jnp.zeros(revision_shape, dtype=jnp.uint32),
            value_revision_words=jnp.zeros(revision_shape, dtype=jnp.uint32),
            source_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
            policy_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
            value_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
            authority_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
            model_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
            anchor_integrity_tags=jnp.zeros((budget,), dtype=jnp.uint32),
        )

    def _zero_diagnostics(
        self,
        state: EnsembleShortRolloutState,
        *,
        state_static: bool,
        model_static: bool,
        authority_static: bool,
        anchor_static: bool,
    ) -> EnsembleShortRolloutDiagnostics:
        shape = (self._config.rollout_budget, self._config.rollout_horizon)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return EnsembleShortRolloutDiagnostics(
            state_static_contract_valid=jnp.asarray(state_static),
            model_state_static_contract_valid=jnp.asarray(model_static),
            authority_static_contract_valid=jnp.asarray(authority_static),
            anchor_static_contract_valid=jnp.asarray(anchor_static),
            state_valid=false,
            model_state_valid=false,
            authority_valid=false,
            anchor_identity_valid=false,
            revisions_monotonic=false,
            decision_identity_valid=false,
            call_capacity_available=false,
            attempt_capacity_available=false,
            selected_actions=jnp.zeros(shape, dtype=jnp.int32),
            selected_policy_probabilities=jnp.zeros(shape, dtype=jnp.float32),
            epistemic_disagreements=jnp.zeros(shape, dtype=jnp.float32),
            residual_variance_maxima=jnp.zeros(shape, dtype=jnp.float32),
            support_valid=jnp.zeros(shape, dtype=jnp.bool_),
            policy_valid=jnp.zeros(shape, dtype=jnp.bool_),
            model_prediction_valid=jnp.zeros(shape, dtype=jnp.bool_),
            residual_proxy_ready=jnp.zeros(shape, dtype=jnp.bool_),
            epistemic_valid=jnp.zeros(shape, dtype=jnp.bool_),
            residual_variance_valid=jnp.zeros(shape, dtype=jnp.bool_),
            continuation_valid=jnp.zeros(shape, dtype=jnp.bool_),
            termination_agreement=jnp.zeros(shape, dtype=jnp.bool_),
            finite_values_valid=jnp.zeros(shape, dtype=jnp.bool_),
            guard_passed=jnp.zeros(shape, dtype=jnp.bool_),
            path_failed=jnp.zeros((self._config.rollout_budget,), dtype=jnp.bool_),
            path_return_valid=jnp.zeros(
                (self._config.rollout_budget,), dtype=jnp.bool_
            ),
            transaction_applied=false,
            pre_proposal_call_count_words=state.proposal_call_count_words,
            post_proposal_call_count_words=state.proposal_call_count_words,
            pre_rollout_attempt_count_words=state.rollout_attempt_count_words,
            post_rollout_attempt_count_words=state.rollout_attempt_count_words,
            pre_accepted_rollout_count_words=state.accepted_rollout_count_words,
            post_accepted_rollout_count_words=state.accepted_rollout_count_words,
            pre_rejected_rollout_count_words=state.rejected_rollout_count_words,
            post_rejected_rollout_count_words=state.rejected_rollout_count_words,
            pre_imagined_step_count_words=state.imagined_step_count_words,
            post_imagined_step_count_words=state.imagined_step_count_words,
        )

    @jax.jit(static_argnums=(0,))
    def propose(
        self,
        state: EnsembleShortRolloutState,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
        anchor: RealStateRolloutAnchor,
    ) -> EnsembleShortRolloutResult:
        """Return fixed-shape proposals and atomically advance only lane state."""

        state_static = self._state_static_valid(state)
        model_static = self._model_static_valid(model_state)
        authority_static = self._authority_static_valid(authority)
        anchor_static = self._anchor_static_valid(anchor)
        zero_proposals = self._zero_proposals()
        zero_diagnostics = self._zero_diagnostics(
            state,
            state_static=state_static,
            model_static=model_static,
            authority_static=authority_static,
            anchor_static=anchor_static,
        )
        if not (state_static and model_static and authority_static and anchor_static):
            return EnsembleShortRolloutResult(
                state=state,
                proposals=zero_proposals,
                diagnostics=zero_diagnostics,
            )

        state_valid = self._state_valid(state)
        model_valid = self._ensemble.state_valid(model_state)
        authority_valid = self._authority_valid(authority, model_state)
        anchor_valid = self._anchor_valid(anchor, authority)
        source_monotonic = _words_less_equal(
            state.bound_source_revision_words,
            authority.source_revision_words,
        )
        model_monotonic = _words_less_equal(
            state.bound_model_revision_words,
            authority.model_revision_words,
        )
        policy_monotonic = _words_less_equal(
            state.bound_policy_revision_words,
            authority.policy_revision_words,
        )
        value_monotonic = _words_less_equal(
            state.bound_value_revision_words,
            authority.value_revision_words,
        )
        source_alias_valid = (~jnp.array_equal(
            state.bound_source_revision_words,
            authority.source_revision_words,
        )) | (state.bound_source_integrity_tag == authority.source_integrity_tag)
        model_alias_valid = (~jnp.array_equal(
            state.bound_model_revision_words,
            authority.model_revision_words,
        )) | (state.bound_model_integrity_tag == authority.model_integrity_tag)
        policy_alias_valid = (~jnp.array_equal(
            state.bound_policy_revision_words,
            authority.policy_revision_words,
        )) | (state.bound_policy_integrity_tag == authority.policy_integrity_tag)
        value_alias_valid = (~jnp.array_equal(
            state.bound_value_revision_words,
            authority.value_revision_words,
        )) | (state.bound_value_integrity_tag == authority.value_integrity_tag)
        revisions_monotonic = (
            source_monotonic
            & model_monotonic
            & policy_monotonic
            & value_monotonic
            & source_alias_valid
            & model_alias_valid
            & policy_alias_valid
            & value_alias_valid
        )
        decision_valid = _words_less(
            state.last_decision_id_words,
            anchor.decision_id_words,
        )
        proposed_calls, call_clock_valid = _checked_words_add_small(
            state.proposal_call_count_words,
            1,
        )
        proposed_attempts, attempt_clock_valid = _checked_words_add_small(
            state.rollout_attempt_count_words,
            self._config.rollout_budget,
        )
        call_capacity = call_clock_valid & _words_leq_limit(
            proposed_calls,
            self._config.max_proposal_calls,
        )
        attempt_capacity = attempt_clock_valid & _words_leq_limit(
            proposed_attempts,
            self._config.max_rollout_attempts,
        )
        preflight = (
            state_valid
            & model_valid
            & authority_valid
            & anchor_valid
            & revisions_monotonic
            & decision_valid
            & call_capacity
            & attempt_capacity
        )

        def do_propose(_: None) -> EnsembleShortRolloutResult:
            horizon = self._config.rollout_horizon
            budget = self._config.rollout_budget

            def one_path(path_key: Array) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
                observation_log = jnp.zeros(
                    (horizon, self.observation_dim), dtype=jnp.float32
                )
                action_log = jnp.zeros((horizon,), dtype=jnp.int32)
                reward_log = jnp.zeros((horizon,), dtype=jnp.float32)
                continuation_log = jnp.zeros((horizon,), dtype=jnp.float32)
                next_log = jnp.zeros(
                    (horizon, self.observation_dim), dtype=jnp.float32
                )
                valid_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                terminated_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                probability_log = jnp.zeros((horizon,), dtype=jnp.float32)
                epistemic_log = jnp.zeros((horizon,), dtype=jnp.float32)
                residual_log = jnp.zeros((horizon,), dtype=jnp.float32)
                support_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                policy_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                prediction_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                readiness_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                epistemic_valid_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                residual_valid_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                continuation_valid_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                agreement_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                finite_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                guard_log = jnp.zeros((horizon,), dtype=jnp.bool_)
                carry = (
                    anchor.observation,
                    path_key,
                    jnp.asarray(True, dtype=jnp.bool_),
                    jnp.asarray(False, dtype=jnp.bool_),
                    observation_log,
                    action_log,
                    reward_log,
                    continuation_log,
                    next_log,
                    valid_log,
                    terminated_log,
                    probability_log,
                    epistemic_log,
                    residual_log,
                    support_log,
                    policy_log,
                    prediction_log,
                    readiness_log,
                    epistemic_valid_log,
                    residual_valid_log,
                    continuation_valid_log,
                    agreement_log,
                    finite_log,
                    guard_log,
                )

                def step_body(index: int, loop: tuple[Array, ...]) -> tuple[Array, ...]:
                    (
                        observation,
                        key,
                        active,
                        failed,
                        observations,
                        actions,
                        rewards,
                        continuations,
                        next_observations,
                        valids,
                        terminateds,
                        probabilities,
                        epistemics,
                        residuals,
                        supports,
                        policies,
                        predictions,
                        readiness,
                        epistemic_valids,
                        residual_valids,
                        continuation_valids,
                        agreements,
                        finites,
                        guards,
                    ) = loop
                    next_key, draw_key = jr.split(key)
                    action, prediction, facts = self._select_prediction(
                        model_state,
                        authority,
                        observation,
                        draw_key,
                    )
                    (
                        probability,
                        support_valid,
                        policy_valid,
                        residual_max,
                        residual_ready,
                        epistemic_valid,
                        residual_valid,
                        continuation,
                        continuation_valid,
                        agreement,
                        finite,
                    ) = facts
                    prediction_valid = prediction.valid
                    guard = (
                        active
                        & support_valid
                        & policy_valid
                        & prediction_valid
                        & residual_ready
                        & epistemic_valid
                        & residual_valid
                        & continuation_valid
                        & agreement
                        & finite
                    )
                    terminated = guard & (
                        continuation == jnp.asarray(0.0, dtype=jnp.float32)
                    )
                    next_active = guard & ~terminated
                    next_failed = failed | (active & ~guard)
                    return (
                        jnp.where(
                            guard,
                            prediction.mean_next_observation,
                            observation,
                        ),
                        next_key,
                        next_active,
                        next_failed,
                        observations.at[index].set(jnp.where(active, observation, 0.0)),
                        actions.at[index].set(jnp.where(active, action, 0)),
                        rewards.at[index].set(
                            jnp.where(guard, prediction.mean_reward, 0.0)
                        ),
                        continuations.at[index].set(jnp.where(guard, continuation, 0.0)),
                        next_observations.at[index].set(
                            jnp.where(guard, prediction.mean_next_observation, 0.0)
                        ),
                        valids.at[index].set(guard),
                        terminateds.at[index].set(terminated),
                        probabilities.at[index].set(jnp.where(active, probability, 0.0)),
                        epistemics.at[index].set(
                            jnp.where(
                                active & jnp.isfinite(prediction.epistemic_disagreement),
                                prediction.epistemic_disagreement,
                                0.0,
                            )
                        ),
                        residuals.at[index].set(
                            jnp.where(active & jnp.isfinite(residual_max), residual_max, 0.0)
                        ),
                        supports.at[index].set(active & support_valid),
                        policies.at[index].set(active & policy_valid),
                        predictions.at[index].set(active & prediction_valid),
                        readiness.at[index].set(active & residual_ready),
                        epistemic_valids.at[index].set(active & epistemic_valid),
                        residual_valids.at[index].set(active & residual_valid),
                        continuation_valids.at[index].set(active & continuation_valid),
                        agreements.at[index].set(active & agreement),
                        finites.at[index].set(active & finite),
                        guards.at[index].set(guard),
                    )

                final = jax.lax.fori_loop(0, horizon, step_body, carry)
                (
                    final_observation,
                    _,
                    final_active,
                    failed,
                    observations,
                    actions,
                    rewards,
                    continuations,
                    next_observations,
                    valids,
                    terminateds,
                    probabilities,
                    epistemics,
                    residuals,
                    supports,
                    policies,
                    predictions,
                    readiness,
                    epistemic_valids,
                    residual_valids,
                    continuation_valids,
                    agreements,
                    finites,
                    guards,
                ) = final
                raw_bootstrap = (
                    jnp.dot(authority.value_weights, final_observation)
                    + authority.value_bias
                )
                bootstrap = jnp.where(final_active, raw_bootstrap, 0.0)
                value_valid = (~final_active) | (
                    jnp.isfinite(raw_bootstrap)
                    & (
                        jnp.abs(raw_bootstrap)
                        <= jnp.asarray(self._config.max_abs_value, dtype=jnp.float32)
                    )
                )

                def reverse_body(
                    carry_return: Array,
                    inputs: tuple[Array, ...],
                ) -> tuple[Array, Array]:
                    reward, continuation, valid = inputs
                    candidate = reward + continuation * carry_return
                    next_return = jnp.where(valid, candidate, carry_return)
                    return next_return, jnp.where(valid, candidate, 0.0)

                _, reversed_returns = jax.lax.scan(
                    reverse_body,
                    bootstrap,
                    (rewards[::-1], continuations[::-1], valids[::-1]),
                )
                returns = reversed_returns[::-1]
                return_valid = (
                    value_valid
                    & jnp.all(jnp.isfinite(returns))
                    & jnp.all(
                        jnp.abs(returns)
                        <= jnp.asarray(self._config.max_abs_return, dtype=jnp.float32)
                    )
                )
                accepted = ~failed & jnp.any(valids) & return_valid
                committed_valids = valids & accepted
                proposals = (
                    jnp.where(committed_valids[:, None], observations, 0.0),
                    jnp.where(committed_valids, actions, 0),
                    jnp.where(committed_valids, rewards, 0.0),
                    jnp.where(committed_valids, continuations, 0.0),
                    jnp.where(committed_valids[:, None], next_observations, 0.0),
                    jnp.where(committed_valids, returns, 0.0),
                    jnp.where(accepted, bootstrap, 0.0),
                    jnp.where(accepted, returns[0], 0.0),
                    committed_valids,
                    terminateds & committed_valids,
                    accepted,
                )
                diagnostics = (
                    actions,
                    probabilities,
                    epistemics,
                    residuals,
                    supports,
                    policies,
                    predictions,
                    readiness,
                    epistemic_valids,
                    residual_valids,
                    continuation_valids,
                    agreements,
                    finites,
                    guards,
                    failed,
                    return_valid,
                )
                return proposals, diagnostics

            def path_scan(
                master_key: Array,
                _: Array,
            ) -> tuple[Array, tuple[tuple[Array, ...], tuple[Array, ...]]]:
                next_master, path_key = jr.split(master_key)
                return next_master, one_path(path_key)

            final_key, (proposal_parts, diagnostic_parts) = jax.lax.scan(
                path_scan,
                state.rollout_key,
                jnp.arange(budget, dtype=jnp.int32),
            )
            (
                observations,
                actions,
                rewards,
                continuations,
                next_observations,
                returns,
                bootstraps,
                root_returns,
                transition_valid,
                terminated,
                path_accepted,
            ) = proposal_parts
            (
                selected_actions,
                probabilities,
                epistemics,
                residuals,
                supports,
                policies,
                predictions,
                readiness,
                epistemic_valids,
                residual_valids,
                continuation_valids,
                agreements,
                finites,
                guards,
                path_failed,
                path_return_valid,
            ) = diagnostic_parts
            accepted_count = jnp.sum(path_accepted.astype(jnp.int32))
            rejected_count = budget - accepted_count
            imagined_count = jnp.sum(transition_valid.astype(jnp.int32))
            proposed_accepted, accepted_clock_valid = _checked_words_add_small(
                state.accepted_rollout_count_words,
                accepted_count,
            )
            proposed_rejected, rejected_clock_valid = _checked_words_add_small(
                state.rejected_rollout_count_words,
                rejected_count,
            )
            proposed_imagined, imagined_clock_valid = _checked_words_add_small(
                state.imagined_step_count_words,
                imagined_count,
            )
            imagined_capacity = _words_leq_limit(
                proposed_imagined,
                self._config.max_imagined_steps,
            )
            candidate_state = state.replace(
                rollout_key=final_key,
                bound_source_revision_words=authority.source_revision_words,
                bound_model_revision_words=authority.model_revision_words,
                bound_policy_revision_words=authority.policy_revision_words,
                bound_value_revision_words=authority.value_revision_words,
                bound_source_integrity_tag=authority.source_integrity_tag,
                bound_model_integrity_tag=authority.model_integrity_tag,
                bound_policy_integrity_tag=authority.policy_integrity_tag,
                bound_value_integrity_tag=authority.value_integrity_tag,
                last_decision_id_words=anchor.decision_id_words,
                proposal_call_count_words=proposed_calls,
                rollout_attempt_count_words=proposed_attempts,
                accepted_rollout_count_words=proposed_accepted,
                rejected_rollout_count_words=proposed_rejected,
                imagined_step_count_words=proposed_imagined,
            )
            candidate_valid = (
                accepted_clock_valid
                & rejected_clock_valid
                & imagined_clock_valid
                & imagined_capacity
                & self._state_valid(candidate_state)
            )
            next_state = cast(
                EnsembleShortRolloutState,
                jax.lax.cond(
                    candidate_valid,
                    lambda _: candidate_state,
                    lambda _: state,
                    None,
                ),
            )
            proposals = ImaginedRolloutBatch(
                observations=jnp.where(candidate_valid, observations, 0.0),
                actions=jnp.where(candidate_valid, actions, 0),
                rewards=jnp.where(candidate_valid, rewards, 0.0),
                continuations=jnp.where(candidate_valid, continuations, 0.0),
                next_observations=jnp.where(candidate_valid, next_observations, 0.0),
                return_targets=jnp.where(candidate_valid, returns, 0.0),
                bootstrap_values=jnp.where(candidate_valid, bootstraps, 0.0),
                root_returns=jnp.where(candidate_valid, root_returns, 0.0),
                transition_valid=transition_valid & candidate_valid,
                terminated=terminated & candidate_valid,
                path_accepted=path_accepted & candidate_valid,
                decision_id_words=jnp.where(
                    candidate_valid,
                    jnp.broadcast_to(anchor.decision_id_words, (budget, 2)),
                    0,
                ),
                source_revision_words=jnp.where(
                    candidate_valid,
                    jnp.broadcast_to(anchor.source_revision_words, (budget, 2)),
                    0,
                ),
                model_revision_words=jnp.where(
                    candidate_valid,
                    jnp.broadcast_to(anchor.model_revision_words, (budget, 2)),
                    0,
                ),
                policy_revision_words=jnp.where(
                    candidate_valid,
                    jnp.broadcast_to(anchor.policy_revision_words, (budget, 2)),
                    0,
                ),
                value_revision_words=jnp.where(
                    candidate_valid,
                    jnp.broadcast_to(anchor.value_revision_words, (budget, 2)),
                    0,
                ),
                source_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        authority.source_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
                policy_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        authority.policy_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
                value_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        authority.value_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
                authority_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        anchor.authority_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
                model_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        anchor.model_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
                anchor_integrity_tags=jnp.where(
                    candidate_valid,
                    jnp.full(
                        (budget,),
                        anchor.anchor_integrity_tag,
                        dtype=jnp.uint32,
                    ),
                    0,
                ),
            )
            diagnostics = EnsembleShortRolloutDiagnostics(
                state_static_contract_valid=jnp.asarray(True),
                model_state_static_contract_valid=jnp.asarray(True),
                authority_static_contract_valid=jnp.asarray(True),
                anchor_static_contract_valid=jnp.asarray(True),
                state_valid=state_valid,
                model_state_valid=model_valid,
                authority_valid=authority_valid,
                anchor_identity_valid=anchor_valid,
                revisions_monotonic=revisions_monotonic,
                decision_identity_valid=decision_valid,
                call_capacity_available=call_capacity,
                attempt_capacity_available=attempt_capacity,
                selected_actions=selected_actions,
                selected_policy_probabilities=probabilities,
                epistemic_disagreements=epistemics,
                residual_variance_maxima=residuals,
                support_valid=supports,
                policy_valid=policies,
                model_prediction_valid=predictions,
                residual_proxy_ready=readiness,
                epistemic_valid=epistemic_valids,
                residual_variance_valid=residual_valids,
                continuation_valid=continuation_valids,
                termination_agreement=agreements,
                finite_values_valid=finites,
                guard_passed=guards,
                path_failed=path_failed,
                path_return_valid=path_return_valid,
                transaction_applied=candidate_valid,
                pre_proposal_call_count_words=state.proposal_call_count_words,
                post_proposal_call_count_words=next_state.proposal_call_count_words,
                pre_rollout_attempt_count_words=state.rollout_attempt_count_words,
                post_rollout_attempt_count_words=next_state.rollout_attempt_count_words,
                pre_accepted_rollout_count_words=state.accepted_rollout_count_words,
                post_accepted_rollout_count_words=next_state.accepted_rollout_count_words,
                pre_rejected_rollout_count_words=state.rejected_rollout_count_words,
                post_rejected_rollout_count_words=next_state.rejected_rollout_count_words,
                pre_imagined_step_count_words=state.imagined_step_count_words,
                post_imagined_step_count_words=next_state.imagined_step_count_words,
            )
            return EnsembleShortRolloutResult(
                state=next_state,
                proposals=proposals,
                diagnostics=diagnostics,
            )

        def reject(_: None) -> EnsembleShortRolloutResult:
            return EnsembleShortRolloutResult(
                state=state,
                proposals=zero_proposals,
                diagnostics=zero_diagnostics.replace(
                    state_valid=state_valid,
                    model_state_valid=model_valid,
                    authority_valid=authority_valid,
                    anchor_identity_valid=anchor_valid,
                    revisions_monotonic=revisions_monotonic,
                    decision_identity_valid=decision_valid,
                    call_capacity_available=call_capacity,
                    attempt_capacity_available=attempt_capacity,
                ),
            )

        return cast(
            EnsembleShortRolloutResult,
            jax.lax.cond(preflight, do_propose, reject, None),
        )

    def _template_authority(self) -> RolloutPolicyValueAuthority:
        one = jnp.asarray([0, 1], dtype=jnp.uint32)
        return self.bind_authority(
            policy_weights=jnp.zeros(
                (self.n_actions, self.observation_dim), dtype=jnp.float32
            ),
            policy_bias=jnp.zeros((self.n_actions,), dtype=jnp.float32),
            value_weights=jnp.zeros((self.observation_dim,), dtype=jnp.float32),
            value_bias=jnp.asarray(0.0, dtype=jnp.float32),
            action_support_counts=jnp.ones((self.n_actions,), dtype=jnp.int32),
            source_revision_words=one,
            model_state=self._reference_model_state,
            policy_revision_words=one,
            value_revision_words=one,
        )

    @property
    def resource_budget(self) -> EnsembleShortRolloutResourceBudget:
        """Return exact logical fixed-state/output sizes and work ceilings."""

        authority = self._template_authority()
        state = self._empty_state(jr.key(0, impl="threefry2x32"), authority)
        proposals = self._zero_proposals()
        diagnostics = self._zero_diagnostics(
            state,
            state_static=True,
            model_static=True,
            authority_static=True,
            anchor_static=True,
        )
        state_scalars, state_bytes = _logical_tree_size(state)
        proposal_scalars, proposal_bytes = _logical_tree_size(proposals)
        diagnostics_scalars, diagnostics_bytes = _logical_tree_size(diagnostics)
        predictions_per_step = (
            1
            if self._config.selection_mode == "policy_directed"
            else self.n_actions
        )
        transitions = self._config.rollout_budget * self._config.rollout_horizon
        model_state_scalars, _ = _logical_tree_size(self._reference_model_state)
        return EnsembleShortRolloutResourceBudget(
            persistent_bytes_scope=(
                "planner-owned-array-leaves-only; excludes-model,policy,value,host-"
                "object-overhead,compiler-and-xla-workspaces"
            ),
            proposal_bytes_scope=(
                "full-fixed-shape-imagined-proposal-array-leaves; not-dispatched"
            ),
            diagnostic_bytes_scope=(
                "full-fixed-shape-diagnostic-array-leaves; not-a-device-peak"
            ),
            temporary_bytes_scope=(
                "not-measured; source-level-call-count-upper-bounds-only; excludes-"
                "compiler-and-xla-workspaces"
            ),
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            ensemble_size=self._ensemble.config.ensemble_size,
            rollout_horizon=self._config.rollout_horizon,
            rollout_budget=self._config.rollout_budget,
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            proposal_scalars=proposal_scalars,
            proposal_bytes=proposal_bytes,
            diagnostics_scalars=diagnostics_scalars,
            diagnostics_bytes=diagnostics_bytes,
            rollout_prng_keys=1,
            rollout_prng_uint32_scalars=2,
            max_ensemble_prediction_calls_per_call=(
                transitions * predictions_per_step
            ),
            max_member_predictions_per_call=(
                transitions
                * predictions_per_step
                * self._ensemble.config.ensemble_size
            ),
            max_policy_forward_calls_per_call=transitions,
            max_value_forward_calls_per_call=self._config.rollout_budget,
            max_model_integrity_scalar_reads_per_call=model_state_scalars,
            max_rng_splits_per_call=transitions + self._config.rollout_budget,
            max_rng_draws_per_call=(
                transitions
                if self._config.selection_mode == "policy_directed"
                else 0
            ),
            max_proposal_calls=self._config.max_proposal_calls,
            max_rollout_attempts=self._config.max_rollout_attempts,
            max_imagined_steps=self._config.max_imagined_steps,
            model_state_owned=0,
            policy_value_state_owned=0,
            actor_or_critic_updates_per_call=0,
            dispatch_authority=0,
        )


def save_ensemble_short_rollout_checkpoint(
    planner: EnsembleShortRolloutPlanner,
    state: EnsembleShortRolloutState,
    path: str | Path,
) -> None:
    """Persist only rollout-lane revisions, tags, exact clocks, and RNG."""

    planner.validate_state(state)
    config = planner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": ENSEMBLE_SHORT_ROLLOUT_CHECKPOINT_SCHEMA,
            "planner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": planner.resource_budget.to_config(),
            "model_state_included": False,
            "policy_value_state_included": False,
            "proposals_included": False,
            "dispatch_authority": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_ensemble_short_rollout_checkpoint(
    path: str | Path,
) -> tuple[EnsembleShortRolloutPlanner, EnsembleShortRolloutState]:
    """Restore the sole v1 planner-owned checkpoint schema."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != ENSEMBLE_SHORT_ROLLOUT_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an ensemble short-rollout v1 checkpoint")
    config = metadata.get("planner_config")
    if not isinstance(config, Mapping):
        raise ValueError("ensemble short-rollout checkpoint is missing planner_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("ensemble short-rollout checkpoint config digest does not match")
    planner = EnsembleShortRolloutPlanner.from_config(config_dict)
    if metadata.get("resource_budget") != planner.resource_budget.to_config():
        raise ValueError("ensemble short-rollout checkpoint resource budget does not match")
    for field in (
        "model_state_included",
        "policy_value_state_included",
        "proposals_included",
        "dispatch_authority",
        "scientific_promotion_allowed",
    ):
        if metadata.get(field) is not False:
            raise ValueError(f"ensemble short-rollout checkpoint {field} must be false")
    template = planner._empty_state(
        jr.key(0, impl="threefry2x32"),
        planner._template_authority(),
    )
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("checkpoint metadata changed between reads")
    state = cast(EnsembleShortRolloutState, restored)
    planner.validate_state(state)
    if _logical_tree_size(state)[1] != planner.resource_budget.persistent_state_bytes:
        raise ValueError("restored ensemble short-rollout state size is invalid")
    return planner, state


__all__ = [
    "ENSEMBLE_SHORT_ROLLOUT_CHECKPOINT_SCHEMA",
    "ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA",
    "ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL",
    "ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS",
    "ENSEMBLE_SHORT_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED",
    "EnsembleShortRolloutConfig",
    "EnsembleShortRolloutDiagnostics",
    "EnsembleShortRolloutPlanner",
    "EnsembleShortRolloutResourceBudget",
    "EnsembleShortRolloutResult",
    "EnsembleShortRolloutState",
    "ImaginedRolloutBatch",
    "RealStateRolloutAnchor",
    "RolloutPolicyValueAuthority",
    "RolloutSelectionMode",
    "load_ensemble_short_rollout_checkpoint",
    "save_ensemble_short_rollout_checkpoint",
]
