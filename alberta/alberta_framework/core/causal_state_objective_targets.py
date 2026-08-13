# mypy: disable-error-code="call-arg"
"""Learner-owned causal targets for comprehensive learned-state objectives.

This opt-in WP3 composition removes ordinary prediction/control target choice
from the caller.  It owns :class:`ComprehensiveStateObjectives`, caches the
exact current observation/representation/action before an outcome, and accepts
only a content-bound real-transition receipt with the same lifecycle,
decision, action, representation revision, and internal action identity.

From that unchanged learner snapshot and accepted transition it derives,
with ``stop_gradient`` throughout:

* action-conditional next-observation and next-latent targets;
* reward, termination, raw discount, and effective continuation;
* one-step multiple-timescale GVF targets;
* a differential current-value target and selected-action TD-advantage target;
* the exact consecutive-pair inverse-action label and validity.

Natural termination suppresses all bootstrap terms even if the raw discount
is nonzero.  Truncation is not termination: it requires a valid final/bootstrap
representation and a positive discount, and bootstraps from that final state,
never from a post-reset decision observation.

The default GVF cumulant is the accepted environment reward.  An arbitrary
cumulant cannot be derived from the transition alone, so the only alternative
is an explicit typed receipt bound to the pending lifecycle/decision,
transition revision, owner digest, monotone source revision, nonzero provenance,
value bits, and a deterministic content tag.  This binding is integrity and
declared provenance, not proof that an external cumulant is semantically sound.

All objective head families remain separate in the owned objective kernel.
Invalid, stale, tampered, non-finite, ambiguous boundary, or exhausted-clock
transactions roll back both producer and objective state bit-for-bit and expose
zero actionable targets/gradients.  State is fixed-size, clocks are exact
uint64 fail-stop pairs, checkpoints are version/config/resource strict, and a
fixed-shape scan supports eager/JIT deterministic execution.

This is isolated L0 ``not_assessed`` machinery.  It does not change
PrototypeAgent, authenticate an environment, calibrate objective masses or
cumulants, establish target quality, pass Forager, complete WP3, or support an
evidence, efficacy, Alberta Plan, or SOTA claim.
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
from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectiveActionReceipt,
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
    ComprehensiveStateObjectivesState,
)

CAUSAL_STATE_OBJECTIVE_TARGET_CONFIG_SCHEMA = (
    "alberta.causal-state-objective-target-producer-config.v1"
)
CAUSAL_STATE_OBJECTIVE_TARGET_STATE_SCHEMA = (
    "alberta.causal-state-objective-target-producer-state.v1"
)
CAUSAL_STATE_OBJECTIVE_TARGET_CHECKPOINT_SCHEMA = (
    "alberta.causal-state-objective-target-producer-checkpoint.v1"
)
CAUSAL_STATE_OBJECTIVE_TARGET_RESOURCE_SCHEMA = (
    "alberta.causal-state-objective-target-producer-resource.v1"
)
CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL = "L0"
CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS = "not_assessed"
CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY = "learner-owned-causal-real-transition"
CAUSAL_STATE_OBJECTIVE_TARGET_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
CAUSAL_STATE_OBJECTIVE_TARGET_MAX_DECISIONS = 2**64 - 1
CAUSAL_STATE_OBJECTIVE_TARGET_MAX_TRANSITIONS = 2**64 - 1
CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES = (
    "action_conditional_next_observation",
    "action_conditional_next_latent",
    "reward",
    "termination_and_discount",
    "multiple_timescale_gvf",
    "current_value",
    "selected_action_advantage",
    "consecutive_pair_inverse_action",
)
CAUSAL_STATE_OBJECTIVE_TARGET_LIMITATIONS = (
    "accepted-transition-owner-digest-is-declared-metadata-not-authentication",
    "optional-arbitrary-cumulant-provenance-is-bound-but-not-semantically-proven",
    "one-step-semi-gradient-value-advantage-and-gvf-targets",
    "linear-owned-objective-heads-with-caller-configured-uncalibrated-masses",
    "no-prototype-integration-feature-lifecycle-forager-or-control-benefit",
    "no-evidence-alberta-plan-completion-or-sota-claim",
)

CausalCumulantMode = Literal["environment_reward", "bound_optional"]

_UINT32_MAX = 2**32 - 1
_OWNER_WORDS = 8
_IDENTITY_WORDS = 4


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    if set(fields) != expected:
        raise ValueError(f"{label} fields differ")
    return fields


def _exact_int(value: Any, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
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
    if scalar <= minimum if strict_minimum else scalar < minimum:
        raise ValueError(f"{label} is below its lower bound")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{label} exceeds its upper bound")
    narrowed = float(np.float32(scalar))
    if not math.isfinite(narrowed) or (scalar != 0.0 and narrowed == 0.0):
        raise ValueError(f"{label} must remain finite and nonzero in float32")
    return scalar


def _owner_digest(value: Any, *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != _OWNER_WORDS:
        raise ValueError(f"{label} must be an exact {_OWNER_WORDS}-word tuple")
    result: list[int] = []
    for index, word in enumerate(value):
        if type(word) is not int or not 0 <= word <= _UINT32_MAX:
            raise ValueError(f"{label}[{index}] must be a uint32 integer")
        result.append(word)
    return tuple(result)


def _cumulant_mode(value: Any) -> CausalCumulantMode:
    if value not in {"environment_reward", "bound_optional"}:
        raise ValueError("cumulant_mode must be environment_reward or bound_optional")
    return cast(CausalCumulantMode, value)


def _require_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    label: str,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _float_vector(value: Any, width: int, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(width,),
        dtype=jnp.dtype(jnp.float32),
        label=label,
    )


def _float_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.float32), label=label)


def _int_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.int32), label=label)


def _bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.bool_), label=label)


def _words(value: Any, width: int, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(width,),
        dtype=jnp.dtype(jnp.uint32),
        label=label,
    )


def _require_key(key: Any, *, label: str) -> None:
    try:
        data = jr.key_data(key)
        implementation = str(jr.key_impl(key))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a typed Threefry key") from error
    if (
        getattr(key, "shape", None) != ()
        or data.shape != (2,)
        or data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be a typed Threefry key")


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


def _rotate_left(value: Array, distance: Array) -> Array:
    right = (jnp.asarray(32, dtype=jnp.uint32) - distance) & jnp.uint32(31)
    return jnp.asarray((value << distance) | (value >> right), dtype=jnp.uint32)


def _content_tag(*values: Array) -> UInt[Array, " 4"]:
    words: list[Array] = []
    for value in values:
        array = jnp.asarray(value)
        if array.dtype == jnp.dtype(jnp.float32):
            converted = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.int32):
            converted = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.bool_):
            converted = array.astype(jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.uint32):
            converted = array
        else:
            raise TypeError("content tag values must use float32/int32/bool/uint32")
        words.append(jnp.reshape(converted, (-1,)))
    payload = jnp.concatenate(tuple(words)).astype(jnp.uint32)
    indices = jnp.arange(payload.shape[0], dtype=jnp.uint32)
    distances = (indices % jnp.uint32(31)) + jnp.uint32(1)
    mixed = _rotate_left(payload ^ (indices * jnp.uint32(0x9E3779B9)), distances)
    return jnp.stack(
        (
            jnp.bitwise_xor.reduce(mixed),
            jnp.sum(mixed * jnp.uint32(0x85EBCA6B), dtype=jnp.uint32),
            jnp.bitwise_xor.reduce(mixed * (indices + jnp.uint32(0xC2B2AE35))),
            jnp.sum(
                _rotate_left(
                    mixed,
                    ((indices * jnp.uint32(7)) % jnp.uint32(31)) + jnp.uint32(1),
                ),
                dtype=jnp.uint32,
            ),
        )
    ).astype(jnp.uint32)


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class CausalStateObjectiveTargetProducerConfig:
    """Nested objectives, declared owners, and cumulant authority mode."""

    objectives_config: ComprehensiveStateObjectivesConfig
    transition_owner_digest: tuple[int, ...]
    cumulant_mode: CausalCumulantMode = "environment_reward"
    cumulant_owner_digest: tuple[int, ...] = (0,) * _OWNER_WORDS

    def __post_init__(self) -> None:
        if type(self.objectives_config) is not ComprehensiveStateObjectivesConfig:
            raise TypeError("objectives_config must be exact ComprehensiveStateObjectivesConfig")
        _owner_digest(self.transition_owner_digest, label="transition_owner_digest")
        _owner_digest(self.cumulant_owner_digest, label="cumulant_owner_digest")
        _cumulant_mode(self.cumulant_mode)
        if self.cumulant_mode == "bound_optional" and not any(self.cumulant_owner_digest):
            raise ValueError("bound_optional cumulants require a nonzero owner digest")

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "CausalStateObjectiveTargetProducer",
            "schema": CAUSAL_STATE_OBJECTIVE_TARGET_CONFIG_SCHEMA,
            "state_schema": CAUSAL_STATE_OBJECTIVE_TARGET_STATE_SCHEMA,
            "evidence_level": CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL,
            "outcome_status": CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
            "lifetime_semantics": CAUSAL_STATE_OBJECTIVE_TARGET_LIFETIME_SEMANTICS,
            "head_families": list(CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES),
            "limitations": list(CAUSAL_STATE_OBJECTIVE_TARGET_LIMITATIONS),
            "arbitrary_cumulant_causal_derivation_claimed": False,
            "objectives_config": self.objectives_config.to_config(),
            "transition_owner_digest": list(self.transition_owner_digest),
            "cumulant_mode": self.cumulant_mode,
            "cumulant_owner_digest": list(self.cumulant_owner_digest),
        }

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
    ) -> CausalStateObjectiveTargetProducerConfig:
        fields = _exact_manifest(
            payload,
            {
                "type",
                "schema",
                "state_schema",
                "evidence_level",
                "outcome_status",
                "scientific_promotion_allowed",
                "target_authority",
                "lifetime_semantics",
                "head_families",
                "limitations",
                "arbitrary_cumulant_causal_derivation_claimed",
                "objectives_config",
                "transition_owner_digest",
                "cumulant_mode",
                "cumulant_owner_digest",
            },
            label="causal state objective target config",
        )
        fixed = {
            "type": "CausalStateObjectiveTargetProducer",
            "schema": CAUSAL_STATE_OBJECTIVE_TARGET_CONFIG_SCHEMA,
            "state_schema": CAUSAL_STATE_OBJECTIVE_TARGET_STATE_SCHEMA,
            "evidence_level": CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL,
            "outcome_status": CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
            "lifetime_semantics": CAUSAL_STATE_OBJECTIVE_TARGET_LIFETIME_SEMANTICS,
            "head_families": list(CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES),
            "limitations": list(CAUSAL_STATE_OBJECTIVE_TARGET_LIMITATIONS),
            "arbitrary_cumulant_causal_derivation_claimed": False,
        }
        for name, expected in fixed.items():
            if fields.pop(name) != expected:
                raise ValueError(f"causal target config {name} is unsupported")
        nested = fields.pop("objectives_config")
        if type(nested) is not dict:
            raise TypeError("objectives_config must serialize as an exact dict")
        for name in ("transition_owner_digest", "cumulant_owner_digest"):
            serialized = fields[name]
            if type(serialized) is not list:
                raise TypeError(f"{name} must serialize as a list")
            fields[name] = tuple(serialized)
        return cls(
            objectives_config=ComprehensiveStateObjectivesConfig.from_config(nested),
            **fields,
        )


@chex.dataclass(frozen=True)
class CausalStateObjectiveTargetProducerState:
    """Owned objectives, exact pending decision, source revision, and clocks."""

    objectives_state: ComprehensiveStateObjectivesState
    pending_observation: Float[Array, " observation"]
    pending_lifecycle_identity_words: UInt[Array, " 4"]
    pending_decision_identity_words: UInt[Array, " 4"]
    pending_objective_action_identity_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    decision_words: UInt[Array, " 2"]
    transition_words: UInt[Array, " 2"]
    last_cumulant_source_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CausalStateObjectiveDecisionReceipt:
    """Bit-exact cached decision and all target ownership identities."""

    observation: Float[Array, " observation"]
    representation: Float[Array, " representation"]
    action: Int[Array, ""]
    representation_revision_words: UInt[Array, " 2"]
    lifecycle_identity_words: UInt[Array, " 4"]
    decision_identity_words: UInt[Array, " 4"]
    objective_action_identity_words: UInt[Array, " 2"]
    producer_decision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CausalStateObjectiveAcceptedTransition:
    """Content-bound accepted real transition; caller authority is declared."""

    accepted: Bool[Array, ""]
    transition_owner_digest: UInt[Array, " 8"]
    transition_revision_words: UInt[Array, " 2"]
    lifecycle_identity_words: UInt[Array, " 4"]
    decision_identity_words: UInt[Array, " 4"]
    objective_action_identity_words: UInt[Array, " 2"]
    source_observation: Float[Array, " observation"]
    action: Int[Array, ""]
    source_representation_revision_words: UInt[Array, " 2"]
    next_observation: Float[Array, " observation"]
    next_representation: Float[Array, " representation"]
    next_representation_revision_words: UInt[Array, " 2"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    bootstrap_valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class CausalStateObjectiveCumulantReceipt:
    """Optional arbitrary cumulant with exact transition-bound provenance."""

    available: Bool[Array, ""]
    value: Float[Array, ""]
    cumulant_owner_digest: UInt[Array, " 8"]
    transition_revision_words: UInt[Array, " 2"]
    lifecycle_identity_words: UInt[Array, " 4"]
    decision_identity_words: UInt[Array, " 4"]
    source_revision_words: UInt[Array, " 2"]
    provenance_words: UInt[Array, " 4"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class CausalStateObjectiveTargets:
    """Detached target families derived from one accepted transition."""

    next_observation: Float[Array, " observation"]
    next_latent: Float[Array, " representation"]
    reward: Float[Array, ""]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    effective_continuation: Float[Array, ""]
    cumulant: Float[Array, ""]
    gvf_targets: Float[Array, " timescale"]
    current_value: Float[Array, ""]
    bootstrap_value: Float[Array, ""]
    control_value_target: Float[Array, ""]
    selected_action_advantage_target: Float[Array, ""]
    inverse_action_label: Int[Array, ""]
    inverse_pair_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CausalStateObjectiveCacheResult:
    state: CausalStateObjectiveTargetProducerState
    receipt: CausalStateObjectiveDecisionReceipt
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    objective_cache_applied: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    cache_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CausalStateObjectiveUpdateResult:
    state: CausalStateObjectiveTargetProducerState
    targets: CausalStateObjectiveTargets
    objective_gvf_targets: Float[Array, " timescale"]
    balanced_loss: Float[Array, ""]
    current_representation_gradient: Float[Array, " representation"]
    next_representation_gradient: Float[Array, " representation"]
    pre_transition_words: UInt[Array, " 2"]
    post_transition_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    decision_receipt_valid: Bool[Array, ""]
    transition_owner_valid: Bool[Array, ""]
    transition_identity_valid: Bool[Array, ""]
    transition_content_valid: Bool[Array, ""]
    transition_semantics_valid: Bool[Array, ""]
    representation_revision_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    cumulant_valid: Bool[Array, ""]
    target_numeric_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    objective_update_applied: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class CausalStateObjectiveTargetResourceBudget:
    schema: str
    objectives_state_nbytes: int
    pending_observation_nbytes: int
    pending_identity_nbytes: int
    pending_valid_nbytes: int
    clock_and_source_revision_nbytes: int
    producer_state_nbytes: int
    total_state_nbytes: int
    max_objective_head_updates_per_transition: int
    max_atomic_transactions_per_transition: int
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class CausalStateObjectiveTargetScanInputs:
    current_observations: Float[Array, "steps observation"]
    current_representations: Float[Array, "steps representation"]
    actions: Int[Array, " steps"]
    current_representation_revision_words: UInt[Array, "steps 2"]
    lifecycle_identity_words: UInt[Array, "steps 4"]
    decision_identity_words: UInt[Array, "steps 4"]
    next_observations: Float[Array, "steps observation"]
    next_representations: Float[Array, "steps representation"]
    next_representation_revision_words: UInt[Array, "steps 2"]
    rewards: Float[Array, " steps"]
    discounts: Float[Array, " steps"]
    terminated: Bool[Array, " steps"]
    truncated: Bool[Array, " steps"]
    bootstrap_valid: Bool[Array, " steps"]
    optional_cumulants: Float[Array, " steps"]
    optional_cumulant_available: Bool[Array, " steps"]
    cumulant_source_revision_words: UInt[Array, "steps 2"]
    cumulant_provenance_words: UInt[Array, "steps 4"]


@chex.dataclass(frozen=True)
class CausalStateObjectiveTargetScanResult:
    state: CausalStateObjectiveTargetProducerState
    gvf_targets: Float[Array, "steps timescale"]
    control_value_targets: Float[Array, " steps"]
    selected_action_advantage_targets: Float[Array, " steps"]
    inverse_action_labels: Int[Array, " steps"]
    current_representation_gradients: Float[Array, "steps representation"]
    next_representation_gradients: Float[Array, "steps representation"]
    cache_applied: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]
    transition_words: UInt[Array, "steps 2"]


def _empty_decision_receipt(
    config: CausalStateObjectiveTargetProducerConfig,
) -> CausalStateObjectiveDecisionReceipt:
    return CausalStateObjectiveDecisionReceipt(
        observation=jnp.zeros(
            (config.objectives_config.observation_target_dim,), dtype=jnp.float32
        ),
        representation=jnp.zeros((config.objectives_config.representation_dim,), dtype=jnp.float32),
        action=jnp.asarray(-1, dtype=jnp.int32),
        representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        lifecycle_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        decision_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        objective_action_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
        producer_decision_words=jnp.zeros((2,), dtype=jnp.uint32),
    )


def _receipt_from_state(
    state: CausalStateObjectiveTargetProducerState,
) -> CausalStateObjectiveDecisionReceipt:
    objectives = state.objectives_state
    return CausalStateObjectiveDecisionReceipt(
        observation=state.pending_observation,
        representation=objectives.pending_representation,
        action=objectives.pending_action,
        representation_revision_words=(objectives.pending_representation_revision_words),
        lifecycle_identity_words=state.pending_lifecycle_identity_words,
        decision_identity_words=state.pending_decision_identity_words,
        objective_action_identity_words=(state.pending_objective_action_identity_words),
        producer_decision_words=state.decision_words,
    )


def _empty_cumulant_receipt() -> CausalStateObjectiveCumulantReceipt:
    return CausalStateObjectiveCumulantReceipt(
        available=jnp.asarray(False, dtype=jnp.bool_),
        value=jnp.asarray(0.0, dtype=jnp.float32),
        cumulant_owner_digest=jnp.zeros((_OWNER_WORDS,), dtype=jnp.uint32),
        transition_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        lifecycle_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        decision_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        source_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        provenance_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        content_tag_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
    )


def _zero_targets(
    config: CausalStateObjectiveTargetProducerConfig,
) -> CausalStateObjectiveTargets:
    cfg = config.objectives_config
    return CausalStateObjectiveTargets(
        next_observation=jnp.zeros((cfg.observation_target_dim,), dtype=jnp.float32),
        next_latent=jnp.zeros((cfg.representation_dim,), dtype=jnp.float32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        effective_continuation=jnp.asarray(0.0, dtype=jnp.float32),
        cumulant=jnp.asarray(0.0, dtype=jnp.float32),
        gvf_targets=jnp.zeros((cfg.n_gvf_heads,), dtype=jnp.float32),
        current_value=jnp.asarray(0.0, dtype=jnp.float32),
        bootstrap_value=jnp.asarray(0.0, dtype=jnp.float32),
        control_value_target=jnp.asarray(0.0, dtype=jnp.float32),
        selected_action_advantage_target=jnp.asarray(0.0, dtype=jnp.float32),
        inverse_action_label=jnp.asarray(-1, dtype=jnp.int32),
        inverse_pair_valid=jnp.asarray(False, dtype=jnp.bool_),
    )


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def measure_causal_state_objective_target_state_nbytes(
    state: CausalStateObjectiveTargetProducerState,
) -> int:
    """Return exact bytes occupied by every persistent array leaf."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class CausalStateObjectiveTargetProducer:
    """Atomic owner of comprehensive heads and their causal supervision."""

    def __init__(self, config: CausalStateObjectiveTargetProducerConfig) -> None:
        if type(config) is not CausalStateObjectiveTargetProducerConfig:
            raise TypeError("config must be exact CausalStateObjectiveTargetProducerConfig")
        self._config = config
        self._objectives = ComprehensiveStateObjectives(config.objectives_config)
        self._discounts = jnp.asarray(
            config.objectives_config.gvf_discounts,
            dtype=jnp.float32,
        )

    @property
    def config(self) -> CausalStateObjectiveTargetProducerConfig:
        return self._config

    @property
    def objectives(self) -> ComprehensiveStateObjectives:
        return self._objectives

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
    ) -> CausalStateObjectiveTargetProducer:
        return cls(CausalStateObjectiveTargetProducerConfig.from_config(payload))

    def init(self, key: Array) -> CausalStateObjectiveTargetProducerState:
        _require_key(key, label="key")
        cfg = self._config.objectives_config
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return CausalStateObjectiveTargetProducerState(
            objectives_state=self._objectives.init(key),
            pending_observation=jnp.zeros((cfg.observation_target_dim,), dtype=jnp.float32),
            pending_lifecycle_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            pending_decision_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            pending_objective_action_identity_words=zero_words,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            decision_words=zero_words,
            transition_words=zero_words,
            last_cumulant_source_revision_words=zero_words,
        )

    def _require_state_contract(
        self,
        state: CausalStateObjectiveTargetProducerState,
    ) -> None:
        if type(state) is not CausalStateObjectiveTargetProducerState:
            raise TypeError("state must be exact CausalStateObjectiveTargetProducerState")
        cfg = self._config.objectives_config
        self._objectives.state_valid(state.objectives_state)
        _float_vector(
            state.pending_observation,
            cfg.observation_target_dim,
            label="state.pending_observation",
        )
        _words(
            state.pending_lifecycle_identity_words,
            _IDENTITY_WORDS,
            label="state.pending_lifecycle_identity_words",
        )
        _words(
            state.pending_decision_identity_words,
            _IDENTITY_WORDS,
            label="state.pending_decision_identity_words",
        )
        _words(
            state.pending_objective_action_identity_words,
            2,
            label="state.pending_objective_action_identity_words",
        )
        _bool_scalar(state.pending_valid, label="state.pending_valid")
        _words(state.decision_words, 2, label="state.decision_words")
        _words(state.transition_words, 2, label="state.transition_words")
        _words(
            state.last_cumulant_source_revision_words,
            2,
            label="state.last_cumulant_source_revision_words",
        )

    def _dynamic_state_valid(
        self,
        state: CausalStateObjectiveTargetProducerState,
    ) -> Bool[Array, ""]:
        objective = state.objectives_state
        pending_filled = (
            jnp.all(jnp.isfinite(state.pending_observation))
            & jnp.all(
                jnp.abs(state.pending_observation)
                <= jnp.float32(self._config.objectives_config.max_abs_observation_target)
            )
            & jnp.any(state.pending_lifecycle_identity_words != 0)
            & jnp.any(state.pending_decision_identity_words != 0)
            & jnp.all(
                state.pending_objective_action_identity_words
                == objective.pending_action_identity_words
            )
        )
        pending_empty = (
            jnp.all(state.pending_observation == 0.0)
            & jnp.all(state.pending_lifecycle_identity_words == 0)
            & jnp.all(state.pending_decision_identity_words == 0)
            & jnp.all(state.pending_objective_action_identity_words == 0)
        )
        return (
            self._objectives.state_valid(objective)
            & (state.pending_valid == objective.pending_valid)
            & jnp.all(state.decision_words == objective.decision_words)
            & jnp.all(state.transition_words == objective.update_words)
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
        )

    def state_valid(
        self,
        state: CausalStateObjectiveTargetProducerState,
    ) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def cache_decision(
        self,
        state: CausalStateObjectiveTargetProducerState,
        *,
        observation: Array,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
        lifecycle_identity_words: Array,
        decision_identity_words: Array,
    ) -> CausalStateObjectiveCacheResult:
        self._require_state_contract(state)
        cfg = self._config.objectives_config
        observation = _float_vector(
            observation,
            cfg.observation_target_dim,
            label="observation",
        )
        representation = _float_vector(
            representation,
            cfg.representation_dim,
            label="representation",
        )
        action = _int_scalar(action, label="action")
        representation_revision_words = _words(
            representation_revision_words,
            2,
            label="representation_revision_words",
        )
        lifecycle_identity_words = _words(
            lifecycle_identity_words,
            _IDENTITY_WORDS,
            label="lifecycle_identity_words",
        )
        decision_identity_words = _words(
            decision_identity_words,
            _IDENTITY_WORDS,
            label="decision_identity_words",
        )
        return cast(
            CausalStateObjectiveCacheResult,
            self._cache_decision_jit(
                state,
                observation,
                representation,
                action,
                representation_revision_words,
                lifecycle_identity_words,
                decision_identity_words,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _cache_decision_jit(
        self,
        state: CausalStateObjectiveTargetProducerState,
        observation: Array,
        representation: Array,
        action: Array,
        representation_revision_words: Array,
        lifecycle_identity_words: Array,
        decision_identity_words: Array,
    ) -> CausalStateObjectiveCacheResult:
        cfg = self._config.objectives_config
        state_valid = self._dynamic_state_valid(state)
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.abs(observation) <= jnp.float32(cfg.max_abs_observation_target))
            & jnp.any(lifecycle_identity_words != 0)
            & jnp.any(decision_identity_words != 0)
            & (~state.pending_valid)
        )
        proposed_words, capacity = _increment_words(state.decision_words)
        objective_cache = self._objectives.cache_action(
            state.objectives_state,
            representation,
            action,
            representation_revision_words,
        )
        objective_clock_matches = jnp.all(objective_cache.post_decision_words == proposed_words)
        candidate = CausalStateObjectiveTargetProducerState(
            objectives_state=objective_cache.state,
            pending_observation=observation,
            pending_lifecycle_identity_words=lifecycle_identity_words,
            pending_decision_identity_words=decision_identity_words,
            pending_objective_action_identity_words=(objective_cache.receipt.action_identity_words),
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            decision_words=proposed_words,
            transition_words=state.transition_words,
            last_cumulant_source_revision_words=(state.last_cumulant_source_revision_words),
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            state_valid
            & source_valid
            & capacity
            & objective_cache.cache_applied
            & objective_clock_matches
            & candidate_valid
        )
        next_state = cast(
            CausalStateObjectiveTargetProducerState,
            _tree_select(applied, candidate, state),
        )
        receipt = cast(
            CausalStateObjectiveDecisionReceipt,
            _tree_select(
                applied,
                _receipt_from_state(candidate),
                _empty_decision_receipt(self._config),
            ),
        )
        return CausalStateObjectiveCacheResult(
            state=next_state,
            receipt=receipt,
            pre_decision_words=state.decision_words,
            post_decision_words=next_state.decision_words,
            state_valid=state_valid,
            source_valid=source_valid,
            lifetime_capacity_available=capacity,
            objective_cache_applied=objective_cache.cache_applied,
            candidate_state_valid=candidate_valid,
            cache_applied=applied,
        )

    def bind_accepted_transition(
        self,
        state: CausalStateObjectiveTargetProducerState,
        *,
        next_observation: Array,
        next_representation: Array,
        next_representation_revision_words: Array,
        reward: Array,
        discount: Array,
        terminated: Array,
        truncated: Array,
        bootstrap_valid: Array,
    ) -> CausalStateObjectiveAcceptedTransition:
        """Content-bind one caller-declared accepted real transition."""

        self._require_state_contract(state)
        cfg = self._config.objectives_config
        next_observation = _float_vector(
            next_observation,
            cfg.observation_target_dim,
            label="next_observation",
        )
        next_representation = _float_vector(
            next_representation,
            cfg.representation_dim,
            label="next_representation",
        )
        next_representation_revision_words = _words(
            next_representation_revision_words,
            2,
            label="next_representation_revision_words",
        )
        reward = _float_scalar(reward, label="reward")
        discount = _float_scalar(discount, label="discount")
        terminated = _bool_scalar(terminated, label="terminated")
        truncated = _bool_scalar(truncated, label="truncated")
        bootstrap_valid = _bool_scalar(bootstrap_valid, label="bootstrap_valid")
        transition_words, _ = _increment_words(state.transition_words)
        objective = state.objectives_state
        owner = jnp.asarray(self._config.transition_owner_digest, dtype=jnp.uint32)
        accepted = jnp.asarray(True, dtype=jnp.bool_)
        content_tag = _content_tag(
            accepted,
            owner,
            transition_words,
            state.pending_lifecycle_identity_words,
            state.pending_decision_identity_words,
            state.pending_objective_action_identity_words,
            state.pending_observation,
            objective.pending_action,
            objective.pending_representation_revision_words,
            next_observation,
            next_representation,
            next_representation_revision_words,
            reward,
            discount,
            terminated,
            truncated,
            bootstrap_valid,
        )
        return CausalStateObjectiveAcceptedTransition(
            accepted=accepted,
            transition_owner_digest=owner,
            transition_revision_words=transition_words,
            lifecycle_identity_words=state.pending_lifecycle_identity_words,
            decision_identity_words=state.pending_decision_identity_words,
            objective_action_identity_words=(state.pending_objective_action_identity_words),
            source_observation=state.pending_observation,
            action=objective.pending_action,
            source_representation_revision_words=(objective.pending_representation_revision_words),
            next_observation=next_observation,
            next_representation=next_representation,
            next_representation_revision_words=(next_representation_revision_words),
            reward=reward,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            bootstrap_valid=bootstrap_valid,
            content_tag_words=content_tag,
        )

    def bind_optional_cumulant(
        self,
        state: CausalStateObjectiveTargetProducerState,
        *,
        value: Array,
        source_revision_words: Array,
        provenance_words: Array,
    ) -> CausalStateObjectiveCumulantReceipt:
        """Bind an arbitrary cumulant; provenance authority remains external."""

        if self._config.cumulant_mode != "bound_optional":
            raise ValueError("optional cumulants require bound_optional mode")
        self._require_state_contract(state)
        value = _float_scalar(value, label="value")
        source_revision_words = _words(
            source_revision_words,
            2,
            label="source_revision_words",
        )
        provenance_words = _words(
            provenance_words,
            _IDENTITY_WORDS,
            label="provenance_words",
        )
        transition_words, _ = _increment_words(state.transition_words)
        owner = jnp.asarray(self._config.cumulant_owner_digest, dtype=jnp.uint32)
        available = jnp.asarray(True, dtype=jnp.bool_)
        tag = _content_tag(
            available,
            value,
            owner,
            transition_words,
            state.pending_lifecycle_identity_words,
            state.pending_decision_identity_words,
            source_revision_words,
            provenance_words,
        )
        return CausalStateObjectiveCumulantReceipt(
            available=available,
            value=value,
            cumulant_owner_digest=owner,
            transition_revision_words=transition_words,
            lifecycle_identity_words=state.pending_lifecycle_identity_words,
            decision_identity_words=state.pending_decision_identity_words,
            source_revision_words=source_revision_words,
            provenance_words=provenance_words,
            content_tag_words=tag,
        )

    def _require_decision_receipt(
        self,
        receipt: CausalStateObjectiveDecisionReceipt,
    ) -> None:
        if type(receipt) is not CausalStateObjectiveDecisionReceipt:
            raise TypeError("decision_receipt must be exact")
        cfg = self._config.objectives_config
        _float_vector(receipt.observation, cfg.observation_target_dim, label="receipt.observation")
        _float_vector(
            receipt.representation,
            cfg.representation_dim,
            label="receipt.representation",
        )
        _int_scalar(receipt.action, label="receipt.action")
        _words(receipt.representation_revision_words, 2, label="receipt.revision")
        _words(receipt.lifecycle_identity_words, _IDENTITY_WORDS, label="receipt.lifecycle")
        _words(receipt.decision_identity_words, _IDENTITY_WORDS, label="receipt.decision")
        _words(receipt.objective_action_identity_words, 2, label="receipt.action_identity")
        _words(receipt.producer_decision_words, 2, label="receipt.producer_decision")

    def _require_transition(
        self,
        transition: CausalStateObjectiveAcceptedTransition,
    ) -> None:
        if type(transition) is not CausalStateObjectiveAcceptedTransition:
            raise TypeError("transition must be exact CausalStateObjectiveAcceptedTransition")
        cfg = self._config.objectives_config
        _bool_scalar(transition.accepted, label="transition.accepted")
        _words(transition.transition_owner_digest, _OWNER_WORDS, label="transition.owner")
        _words(transition.transition_revision_words, 2, label="transition.revision")
        _words(transition.lifecycle_identity_words, _IDENTITY_WORDS, label="transition.lifecycle")
        _words(transition.decision_identity_words, _IDENTITY_WORDS, label="transition.decision")
        _words(transition.objective_action_identity_words, 2, label="transition.action_identity")
        _float_vector(
            transition.source_observation,
            cfg.observation_target_dim,
            label="transition.source_observation",
        )
        _int_scalar(transition.action, label="transition.action")
        _words(
            transition.source_representation_revision_words,
            2,
            label="transition.source_representation_revision_words",
        )
        _float_vector(
            transition.next_observation,
            cfg.observation_target_dim,
            label="transition.next_observation",
        )
        _float_vector(
            transition.next_representation,
            cfg.representation_dim,
            label="transition.next_representation",
        )
        _words(
            transition.next_representation_revision_words,
            2,
            label="transition.next_representation_revision_words",
        )
        _float_scalar(transition.reward, label="transition.reward")
        _float_scalar(transition.discount, label="transition.discount")
        _bool_scalar(transition.terminated, label="transition.terminated")
        _bool_scalar(transition.truncated, label="transition.truncated")
        _bool_scalar(transition.bootstrap_valid, label="transition.bootstrap_valid")
        _words(transition.content_tag_words, _IDENTITY_WORDS, label="transition.content_tag")

    def _require_cumulant(
        self,
        receipt: CausalStateObjectiveCumulantReceipt,
    ) -> None:
        if type(receipt) is not CausalStateObjectiveCumulantReceipt:
            raise TypeError("cumulant_receipt must be exact")
        _bool_scalar(receipt.available, label="cumulant.available")
        _float_scalar(receipt.value, label="cumulant.value")
        _words(receipt.cumulant_owner_digest, _OWNER_WORDS, label="cumulant.owner")
        _words(receipt.transition_revision_words, 2, label="cumulant.transition")
        _words(receipt.lifecycle_identity_words, _IDENTITY_WORDS, label="cumulant.lifecycle")
        _words(receipt.decision_identity_words, _IDENTITY_WORDS, label="cumulant.decision")
        _words(receipt.source_revision_words, 2, label="cumulant.source_revision")
        _words(receipt.provenance_words, _IDENTITY_WORDS, label="cumulant.provenance")
        _words(receipt.content_tag_words, _IDENTITY_WORDS, label="cumulant.content_tag")

    def update(
        self,
        state: CausalStateObjectiveTargetProducerState,
        decision_receipt: CausalStateObjectiveDecisionReceipt,
        transition: CausalStateObjectiveAcceptedTransition,
        cumulant_receipt: CausalStateObjectiveCumulantReceipt | None = None,
    ) -> CausalStateObjectiveUpdateResult:
        """Derive and atomically consume learner-owned targets."""

        self._require_state_contract(state)
        self._require_decision_receipt(decision_receipt)
        self._require_transition(transition)
        cumulant_receipt = (
            _empty_cumulant_receipt() if cumulant_receipt is None else cumulant_receipt
        )
        self._require_cumulant(cumulant_receipt)
        return cast(
            CausalStateObjectiveUpdateResult,
            self._update_jit(
                state,
                decision_receipt,
                transition,
                cumulant_receipt,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: CausalStateObjectiveTargetProducerState,
        decision_receipt: CausalStateObjectiveDecisionReceipt,
        transition: CausalStateObjectiveAcceptedTransition,
        cumulant_receipt: CausalStateObjectiveCumulantReceipt,
    ) -> CausalStateObjectiveUpdateResult:
        cfg = self._config.objectives_config
        objective = state.objectives_state
        state_valid = self._dynamic_state_valid(state) & state.pending_valid
        decision_valid = (
            _float_bits_equal(decision_receipt.observation, state.pending_observation)
            & _float_bits_equal(
                decision_receipt.representation,
                objective.pending_representation,
            )
            & (decision_receipt.action == objective.pending_action)
            & jnp.all(
                decision_receipt.representation_revision_words
                == objective.pending_representation_revision_words
            )
            & jnp.all(
                decision_receipt.lifecycle_identity_words == state.pending_lifecycle_identity_words
            )
            & jnp.all(
                decision_receipt.decision_identity_words == state.pending_decision_identity_words
            )
            & jnp.all(
                decision_receipt.objective_action_identity_words
                == state.pending_objective_action_identity_words
            )
            & jnp.all(decision_receipt.producer_decision_words == state.decision_words)
        )
        proposed_transition_words, capacity = _increment_words(state.transition_words)
        owner_valid = jnp.all(
            transition.transition_owner_digest
            == jnp.asarray(self._config.transition_owner_digest, dtype=jnp.uint32)
        )
        transition_identity_valid = (
            transition.accepted
            & jnp.all(transition.transition_revision_words == proposed_transition_words)
            & jnp.all(transition.lifecycle_identity_words == state.pending_lifecycle_identity_words)
            & jnp.all(transition.decision_identity_words == state.pending_decision_identity_words)
            & jnp.all(
                transition.objective_action_identity_words
                == state.pending_objective_action_identity_words
            )
            & _float_bits_equal(
                transition.source_observation,
                state.pending_observation,
            )
            & (transition.action == objective.pending_action)
            & jnp.all(
                transition.source_representation_revision_words
                == objective.pending_representation_revision_words
            )
        )
        expected_transition_tag = _content_tag(
            transition.accepted,
            transition.transition_owner_digest,
            transition.transition_revision_words,
            transition.lifecycle_identity_words,
            transition.decision_identity_words,
            transition.objective_action_identity_words,
            transition.source_observation,
            transition.action,
            transition.source_representation_revision_words,
            transition.next_observation,
            transition.next_representation,
            transition.next_representation_revision_words,
            transition.reward,
            transition.discount,
            transition.terminated,
            transition.truncated,
            transition.bootstrap_valid,
        )
        transition_content_valid = jnp.all(transition.content_tag_words == expected_transition_tag)
        representation_revision_valid = _words_not_earlier(
            transition.next_representation_revision_words,
            objective.pending_representation_revision_words,
        )
        transition_semantics_valid = (
            ~(transition.terminated & transition.truncated)
            & (transition.discount >= 0.0)
            & (transition.discount <= 1.0)
            & (transition.bootstrap_valid == (~transition.terminated))
            & jnp.where(
                transition.truncated,
                transition.discount > 0.0,
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
        source_valid = (
            jnp.all(jnp.isfinite(transition.next_observation))
            & jnp.all(
                jnp.abs(transition.next_observation) <= jnp.float32(cfg.max_abs_observation_target)
            )
            & jnp.all(jnp.isfinite(transition.next_representation))
            & jnp.all(
                jnp.abs(transition.next_representation) <= jnp.float32(cfg.max_abs_representation)
            )
            & jnp.isfinite(transition.reward)
            & (jnp.abs(transition.reward) <= jnp.float32(cfg.max_abs_reward_target))
            & jnp.isfinite(transition.discount)
        )

        empty = _empty_cumulant_receipt()
        if self._config.cumulant_mode == "environment_reward":
            cumulant_valid = jax.tree.reduce(
                jnp.logical_and,
                jax.tree.map(lambda left, right: jnp.all(left == right), cumulant_receipt, empty),
            )
            safe_cumulant = transition.reward
            next_source_revision = state.last_cumulant_source_revision_words
        else:
            expected_cumulant_tag = _content_tag(
                cumulant_receipt.available,
                cumulant_receipt.value,
                cumulant_receipt.cumulant_owner_digest,
                cumulant_receipt.transition_revision_words,
                cumulant_receipt.lifecycle_identity_words,
                cumulant_receipt.decision_identity_words,
                cumulant_receipt.source_revision_words,
                cumulant_receipt.provenance_words,
            )
            cumulant_valid = (
                cumulant_receipt.available
                & jnp.isfinite(cumulant_receipt.value)
                & (jnp.abs(cumulant_receipt.value) <= jnp.float32(cfg.max_abs_cumulant))
                & jnp.all(
                    cumulant_receipt.cumulant_owner_digest
                    == jnp.asarray(
                        self._config.cumulant_owner_digest,
                        dtype=jnp.uint32,
                    )
                )
                & jnp.all(cumulant_receipt.transition_revision_words == proposed_transition_words)
                & jnp.all(
                    cumulant_receipt.lifecycle_identity_words
                    == state.pending_lifecycle_identity_words
                )
                & jnp.all(
                    cumulant_receipt.decision_identity_words
                    == state.pending_decision_identity_words
                )
                & _words_not_earlier(
                    cumulant_receipt.source_revision_words,
                    state.last_cumulant_source_revision_words,
                )
                & jnp.any(cumulant_receipt.provenance_words != 0)
                & jnp.all(cumulant_receipt.content_tag_words == expected_cumulant_tag)
            )
            safe_cumulant = cumulant_receipt.value
            next_source_revision = cumulant_receipt.source_revision_words

        safe_next_observation = jnp.nan_to_num(transition.next_observation)
        safe_next_representation = jnp.nan_to_num(transition.next_representation)
        safe_reward = jnp.nan_to_num(transition.reward)
        safe_discount = jnp.clip(jnp.nan_to_num(transition.discount), 0.0, 1.0)
        safe_cumulant = jnp.nan_to_num(safe_cumulant)
        effective_continuation = jnp.where(
            transition.terminated,
            jnp.asarray(0.0, dtype=jnp.float32),
            safe_discount,
        )
        current_value = (
            objective.value_weights @ objective.pending_representation + objective.value_bias
        )
        bootstrap_value = objective.value_weights @ safe_next_representation + objective.value_bias
        control_value_target = safe_reward + effective_continuation * bootstrap_value
        selected_advantage_target = control_value_target - current_value
        gvf_bootstrap = objective.gvf_weights @ safe_next_representation
        gvf_targets = safe_cumulant + (effective_continuation * self._discounts * gvf_bootstrap)
        targets = CausalStateObjectiveTargets(
            next_observation=jax.lax.stop_gradient(safe_next_observation),
            next_latent=jax.lax.stop_gradient(safe_next_representation),
            reward=jax.lax.stop_gradient(safe_reward),
            terminated=jax.lax.stop_gradient(transition.terminated),
            discount=jax.lax.stop_gradient(safe_discount),
            effective_continuation=jax.lax.stop_gradient(effective_continuation),
            cumulant=jax.lax.stop_gradient(safe_cumulant),
            gvf_targets=jax.lax.stop_gradient(gvf_targets),
            current_value=jax.lax.stop_gradient(current_value),
            bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
            control_value_target=jax.lax.stop_gradient(control_value_target),
            selected_action_advantage_target=jax.lax.stop_gradient(selected_advantage_target),
            inverse_action_label=jax.lax.stop_gradient(objective.pending_action),
            inverse_pair_valid=jax.lax.stop_gradient(jnp.asarray(True, dtype=jnp.bool_)),
        )
        target_numeric_valid = (
            jnp.all(jnp.isfinite(targets.next_observation))
            & jnp.all(jnp.isfinite(targets.next_latent))
            & jnp.isfinite(targets.reward)
            & jnp.isfinite(targets.discount)
            & jnp.isfinite(targets.effective_continuation)
            & jnp.isfinite(targets.cumulant)
            & jnp.all(jnp.isfinite(targets.gvf_targets))
            & jnp.isfinite(targets.current_value)
            & jnp.isfinite(targets.bootstrap_value)
            & jnp.isfinite(targets.control_value_target)
            & jnp.isfinite(targets.selected_action_advantage_target)
            & (jnp.abs(targets.cumulant) <= jnp.float32(cfg.max_abs_cumulant))
            & (jnp.abs(targets.control_value_target) <= jnp.float32(cfg.max_abs_control_target))
            & (
                jnp.abs(targets.selected_action_advantage_target)
                <= jnp.float32(cfg.max_abs_control_target)
            )
        )
        objective_receipt = ComprehensiveStateObjectiveActionReceipt(
            representation=objective.pending_representation,
            action=objective.pending_action,
            representation_revision_words=(objective.pending_representation_revision_words),
            action_identity_words=objective.pending_action_identity_words,
        )
        objective_update = self._objectives.update(
            objective,
            objective_receipt,
            targets.next_latent,
            transition.next_representation_revision_words,
            targets.next_observation,
            targets.reward,
            targets.terminated,
            targets.cumulant,
            targets.effective_continuation,
            targets.control_value_target,
            targets.selected_action_advantage_target,
        )
        objective_targets_match = _float_bits_equal(
            objective_update.gvf_targets,
            targets.gvf_targets,
        )
        candidate = CausalStateObjectiveTargetProducerState(
            objectives_state=objective_update.state,
            pending_observation=jnp.zeros_like(state.pending_observation),
            pending_lifecycle_identity_words=jnp.zeros_like(state.pending_lifecycle_identity_words),
            pending_decision_identity_words=jnp.zeros_like(state.pending_decision_identity_words),
            pending_objective_action_identity_words=jnp.zeros_like(
                state.pending_objective_action_identity_words
            ),
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            decision_words=state.decision_words,
            transition_words=proposed_transition_words,
            last_cumulant_source_revision_words=next_source_revision,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            state_valid
            & decision_valid
            & owner_valid
            & transition_identity_valid
            & transition_content_valid
            & transition_semantics_valid
            & representation_revision_valid
            & source_valid
            & cumulant_valid
            & target_numeric_valid
            & capacity
            & objective_update.update_applied
            & objective_targets_match
            & candidate_valid
        )
        next_state = cast(
            CausalStateObjectiveTargetProducerState,
            _tree_select(applied, candidate, state),
        )
        exposed_targets = cast(
            CausalStateObjectiveTargets,
            _tree_select(applied, targets, _zero_targets(self._config)),
        )
        zero_representation = jnp.zeros((cfg.representation_dim,), dtype=jnp.float32)
        return CausalStateObjectiveUpdateResult(
            state=next_state,
            targets=exposed_targets,
            objective_gvf_targets=jnp.where(
                applied,
                objective_update.gvf_targets,
                jnp.zeros_like(objective_update.gvf_targets),
            ),
            balanced_loss=jnp.where(
                applied,
                objective_update.balanced_loss,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            current_representation_gradient=jnp.where(
                applied,
                objective_update.current_representation_gradient,
                zero_representation,
            ),
            next_representation_gradient=jnp.where(
                applied,
                objective_update.next_representation_gradient,
                zero_representation,
            ),
            pre_transition_words=state.transition_words,
            post_transition_words=next_state.transition_words,
            state_valid=state_valid,
            decision_receipt_valid=decision_valid,
            transition_owner_valid=owner_valid,
            transition_identity_valid=transition_identity_valid,
            transition_content_valid=transition_content_valid,
            transition_semantics_valid=transition_semantics_valid,
            representation_revision_valid=representation_revision_valid,
            source_valid=source_valid,
            cumulant_valid=cumulant_valid,
            target_numeric_valid=target_numeric_valid,
            lifetime_capacity_available=capacity,
            objective_update_applied=objective_update.update_applied,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: CausalStateObjectiveTargetProducerState | None = None,
    ) -> CausalStateObjectiveTargetResourceBudget:
        cfg = self._config.objectives_config
        objectives_nbytes = self._objectives.resource_budget().total_state_nbytes
        observation = 4 * cfg.observation_target_dim
        identities = 4 * _IDENTITY_WORDS * 2 + 4 * 2
        pending_valid = 1
        clocks = 4 * 2 * 3
        producer = observation + identities + pending_valid + clocks
        budget = CausalStateObjectiveTargetResourceBudget(
            schema=CAUSAL_STATE_OBJECTIVE_TARGET_RESOURCE_SCHEMA,
            objectives_state_nbytes=objectives_nbytes,
            pending_observation_nbytes=observation,
            pending_identity_nbytes=identities,
            pending_valid_nbytes=pending_valid,
            clock_and_source_revision_nbytes=clocks,
            producer_state_nbytes=producer,
            total_state_nbytes=objectives_nbytes + producer,
            max_objective_head_updates_per_transition=len(
                CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES
            ),
            max_atomic_transactions_per_transition=1,
            temporary_bytes_scope=(
                "source-level-derived-targets-and-objective-temporaries; excludes-"
                "compiler-and-xla-workspaces; not-a-measured-device-peak"
            ),
        )
        if state is not None:
            self._require_state_contract(state)
            if measure_causal_state_objective_target_state_nbytes(state) != (
                budget.total_state_nbytes
            ):
                raise ValueError("causal target state allocation differs from config")
        return budget


def _require_scan_inputs(
    producer: CausalStateObjectiveTargetProducer,
    inputs: CausalStateObjectiveTargetScanInputs,
) -> int:
    if type(inputs) is not CausalStateObjectiveTargetScanInputs:
        raise TypeError("inputs must be exact CausalStateObjectiveTargetScanInputs")
    cfg = producer.config.objectives_config
    if getattr(inputs.current_observations, "ndim", None) != 2:
        raise ValueError("current_observations must have rank two")
    steps = inputs.current_observations.shape[0]
    contracts = {
        "current_observations": (
            inputs.current_observations,
            (steps, cfg.observation_target_dim),
            jnp.dtype(jnp.float32),
        ),
        "current_representations": (
            inputs.current_representations,
            (steps, cfg.representation_dim),
            jnp.dtype(jnp.float32),
        ),
        "actions": (inputs.actions, (steps,), jnp.dtype(jnp.int32)),
        "current_representation_revision_words": (
            inputs.current_representation_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
        "lifecycle_identity_words": (
            inputs.lifecycle_identity_words,
            (steps, _IDENTITY_WORDS),
            jnp.dtype(jnp.uint32),
        ),
        "decision_identity_words": (
            inputs.decision_identity_words,
            (steps, _IDENTITY_WORDS),
            jnp.dtype(jnp.uint32),
        ),
        "next_observations": (
            inputs.next_observations,
            (steps, cfg.observation_target_dim),
            jnp.dtype(jnp.float32),
        ),
        "next_representations": (
            inputs.next_representations,
            (steps, cfg.representation_dim),
            jnp.dtype(jnp.float32),
        ),
        "next_representation_revision_words": (
            inputs.next_representation_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
        "rewards": (inputs.rewards, (steps,), jnp.dtype(jnp.float32)),
        "discounts": (inputs.discounts, (steps,), jnp.dtype(jnp.float32)),
        "terminated": (inputs.terminated, (steps,), jnp.dtype(jnp.bool_)),
        "truncated": (inputs.truncated, (steps,), jnp.dtype(jnp.bool_)),
        "bootstrap_valid": (
            inputs.bootstrap_valid,
            (steps,),
            jnp.dtype(jnp.bool_),
        ),
        "optional_cumulants": (
            inputs.optional_cumulants,
            (steps,),
            jnp.dtype(jnp.float32),
        ),
        "optional_cumulant_available": (
            inputs.optional_cumulant_available,
            (steps,),
            jnp.dtype(jnp.bool_),
        ),
        "cumulant_source_revision_words": (
            inputs.cumulant_source_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
        "cumulant_provenance_words": (
            inputs.cumulant_provenance_words,
            (steps, _IDENTITY_WORDS),
            jnp.dtype(jnp.uint32),
        ),
    }
    for label, (value, shape, dtype) in contracts.items():
        _require_array(value, shape=shape, dtype=dtype, label=label)
    return steps


def run_causal_state_objective_target_scan(
    producer: CausalStateObjectiveTargetProducer,
    state: CausalStateObjectiveTargetProducerState,
    inputs: CausalStateObjectiveTargetScanInputs,
) -> CausalStateObjectiveTargetScanResult:
    """Run exact cache/derive/update transactions with fixed-shape scan inputs."""

    if type(producer) is not CausalStateObjectiveTargetProducer:
        raise TypeError("producer must be exact CausalStateObjectiveTargetProducer")
    producer._require_state_contract(state)
    _require_scan_inputs(producer, inputs)

    def body(
        carry: CausalStateObjectiveTargetProducerState,
        row: tuple[Array, ...],
    ) -> tuple[CausalStateObjectiveTargetProducerState, tuple[Array, ...]]:
        (
            current_observation,
            current_representation,
            action,
            current_revision,
            lifecycle,
            decision,
            next_observation,
            next_representation,
            next_revision,
            reward,
            discount,
            terminated,
            truncated,
            bootstrap_valid,
            optional_cumulant,
            optional_available,
            cumulant_revision,
            cumulant_provenance,
        ) = row
        cached = producer.cache_decision(
            carry,
            observation=current_observation,
            representation=current_representation,
            action=action,
            representation_revision_words=current_revision,
            lifecycle_identity_words=lifecycle,
            decision_identity_words=decision,
        )
        transition = producer.bind_accepted_transition(
            cached.state,
            next_observation=next_observation,
            next_representation=next_representation,
            next_representation_revision_words=next_revision,
            reward=reward,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            bootstrap_valid=bootstrap_valid,
        )
        if producer.config.cumulant_mode == "bound_optional":
            candidate_cumulant = producer.bind_optional_cumulant(
                cached.state,
                value=optional_cumulant,
                source_revision_words=cumulant_revision,
                provenance_words=cumulant_provenance,
            )
            cumulant = cast(
                CausalStateObjectiveCumulantReceipt,
                _tree_select(
                    optional_available,
                    candidate_cumulant,
                    _empty_cumulant_receipt(),
                ),
            )
        else:
            cumulant = _empty_cumulant_receipt()
        updated = producer.update(cached.state, cached.receipt, transition, cumulant)
        return updated.state, (
            updated.targets.gvf_targets,
            updated.targets.control_value_target,
            updated.targets.selected_action_advantage_target,
            updated.targets.inverse_action_label,
            updated.current_representation_gradient,
            updated.next_representation_gradient,
            cached.cache_applied,
            updated.update_applied,
            updated.post_transition_words,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            inputs.current_observations,
            inputs.current_representations,
            inputs.actions,
            inputs.current_representation_revision_words,
            inputs.lifecycle_identity_words,
            inputs.decision_identity_words,
            inputs.next_observations,
            inputs.next_representations,
            inputs.next_representation_revision_words,
            inputs.rewards,
            inputs.discounts,
            inputs.terminated,
            inputs.truncated,
            inputs.bootstrap_valid,
            inputs.optional_cumulants,
            inputs.optional_cumulant_available,
            inputs.cumulant_source_revision_words,
            inputs.cumulant_provenance_words,
        ),
    )
    (
        gvf_targets,
        control_targets,
        advantage_targets,
        inverse_labels,
        current_gradients,
        next_gradients,
        cached,
        updated,
        transition_words,
    ) = outputs
    return CausalStateObjectiveTargetScanResult(
        state=final_state,
        gvf_targets=gvf_targets,
        control_value_targets=control_targets,
        selected_action_advantage_targets=advantage_targets,
        inverse_action_labels=inverse_labels,
        current_representation_gradients=current_gradients,
        next_representation_gradients=next_gradients,
        cache_applied=cached,
        update_applied=updated,
        transition_words=transition_words,
    )


def save_causal_state_objective_target_checkpoint(
    producer: CausalStateObjectiveTargetProducer,
    state: CausalStateObjectiveTargetProducerState,
    path: str | Path,
) -> None:
    """Persist exact producer/objective state with strict versioned metadata."""

    producer._require_state_contract(state)
    if not bool(producer.state_valid(state)):
        raise ValueError("cannot checkpoint invalid causal target state")
    config = producer.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": CAUSAL_STATE_OBJECTIVE_TARGET_CHECKPOINT_SCHEMA,
            "evidence_level": CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL,
            "outcome_status": CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS,
            "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
            "producer_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": producer.resource_budget(state).to_config(),
        },
    )


def load_causal_state_objective_target_checkpoint(
    path: str | Path,
) -> tuple[
    CausalStateObjectiveTargetProducer,
    CausalStateObjectiveTargetProducerState,
]:
    """Restore only a canonical config/resource-compatible L0 checkpoint."""

    metadata = load_checkpoint_metadata(path)
    fields = _exact_manifest(
        metadata,
        {
            "schema",
            "evidence_level",
            "outcome_status",
            "target_authority",
            "producer_config",
            "config_sha256",
            "resource_budget",
        },
        label="causal target checkpoint",
    )
    fixed = {
        "schema": CAUSAL_STATE_OBJECTIVE_TARGET_CHECKPOINT_SCHEMA,
        "evidence_level": CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL,
        "outcome_status": CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS,
        "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
    }
    for name, expected in fixed.items():
        if fields[name] != expected:
            raise ValueError(f"causal target checkpoint {name} is unsupported")
    config = fields["producer_config"]
    if type(config) is not dict:
        raise TypeError("causal target checkpoint config must be an exact dict")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("causal target checkpoint config digest differs")
    producer = CausalStateObjectiveTargetProducer.from_config(config)
    if producer.to_config() != config:
        raise ValueError("causal target checkpoint config is noncanonical")
    template = producer.init(jr.key(0))
    if fields["resource_budget"] != producer.resource_budget(template).to_config():
        raise ValueError("causal target checkpoint resource budget differs")
    restored_raw, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("causal target checkpoint metadata changed between reads")
    restored = cast(CausalStateObjectiveTargetProducerState, restored_raw)
    producer._require_state_contract(restored)
    if not bool(producer.state_valid(restored)):
        raise ValueError("restored causal target state is invalid")
    producer.resource_budget(restored)
    return producer, restored


__all__ = [
    "CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY",
    "CAUSAL_STATE_OBJECTIVE_TARGET_CHECKPOINT_SCHEMA",
    "CAUSAL_STATE_OBJECTIVE_TARGET_CONFIG_SCHEMA",
    "CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL",
    "CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES",
    "CAUSAL_STATE_OBJECTIVE_TARGET_LIFETIME_SEMANTICS",
    "CAUSAL_STATE_OBJECTIVE_TARGET_LIMITATIONS",
    "CAUSAL_STATE_OBJECTIVE_TARGET_MAX_DECISIONS",
    "CAUSAL_STATE_OBJECTIVE_TARGET_MAX_TRANSITIONS",
    "CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS",
    "CAUSAL_STATE_OBJECTIVE_TARGET_RESOURCE_SCHEMA",
    "CAUSAL_STATE_OBJECTIVE_TARGET_STATE_SCHEMA",
    "CausalCumulantMode",
    "CausalStateObjectiveAcceptedTransition",
    "CausalStateObjectiveCacheResult",
    "CausalStateObjectiveCumulantReceipt",
    "CausalStateObjectiveDecisionReceipt",
    "CausalStateObjectiveTargetProducer",
    "CausalStateObjectiveTargetProducerConfig",
    "CausalStateObjectiveTargetProducerState",
    "CausalStateObjectiveTargetResourceBudget",
    "CausalStateObjectiveTargetScanInputs",
    "CausalStateObjectiveTargetScanResult",
    "CausalStateObjectiveTargets",
    "CausalStateObjectiveUpdateResult",
    "load_causal_state_objective_target_checkpoint",
    "measure_causal_state_objective_target_state_nbytes",
    "run_causal_state_objective_target_scan",
    "save_causal_state_objective_target_checkpoint",
]
