# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Stateless, fail-closed proposals from consolidated procedural memory.

``ConsolidatedProceduralMemoryPolicy`` consumes only an already-produced
``ProceduralMemoryRetrieval``.  It cannot query or mutate memory, dispatch an
action, mutate an agent, or use randomness.  Stored procedural payloads are
interpreted only as non-negative categorical score mass.  After strict
compatibility, evidence, Wilson success-bound, outcome-uncertainty, and caller
hard-safety gates, the policy proposes the lowest-index safe argmax.

This is an L0 mechanism contract.  It provides no efficacy, live-control,
Prototype integration, scientific-promotion, or SOTA claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.consolidated_memory import ProceduralMemoryRetrieval

CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CONFIG_SCHEMA = (
    "alberta.consolidated-procedural-memory-policy.config.v1"
)
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_STATELESS = True
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CHECKPOINT_REQUIRED = False
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROPOSAL_AUTHORITY = True
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_ACTION_DISPATCH_AUTHORITY = False
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_MEMORY_MUTATION_AUTHORITY = False
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_AGENT_MUTATION_AUTHORITY = False
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROMOTION_AUTHORITY = False
CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_SCIENTIFIC_PROMOTION_ALLOWED = False

POLICY_REASON_AVAILABLE = 0
POLICY_REASON_RETRIEVAL_CONTRACT = 1
POLICY_REASON_RETRIEVAL_UNAVAILABLE = 2
POLICY_REASON_COMPATIBILITY = 3
POLICY_REASON_LIFECYCLE = 4
POLICY_REASON_EVIDENCE = 5
POLICY_REASON_COUNT_INCONSISTENCY = 6
POLICY_REASON_SUCCESS_BOUND = 7
POLICY_REASON_OUTCOME = 8
POLICY_REASON_UNCERTAINTY = 9
POLICY_REASON_SCORE_MASS = 10
POLICY_REASON_NO_SAFE_ACTION = 11
POLICY_REASON_NO_SAFE_POSITIVE_MASS = 12

_DIGEST_BYTES = 32
_INT32_MAX = 2**31 - 1
_MAX_ACTIONS = 16_384
_MAX_OUTCOME_DIM = 16_384


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact Python int")
    return value


def _probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented) or not 0.0 <= represented <= 1.0:
        raise ValueError(f"{name} must remain in [0, 1] in float32")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite exact Python float")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented) or represented <= 0.0:
        raise ValueError(f"{name} must remain positive and finite in float32")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype metadata")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if jnp.dtype(value.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}, got {value.dtype}")


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        result = jnp.asarray(value, dtype=jnp.int32)
    else:
        result = jnp.asarray(value)
    _require_array(result, name=name, shape=(), dtype=jnp.int32)
    return result


def _digest_is_nonzero(value: Array) -> Array:
    return jnp.any(value != jnp.asarray(0, dtype=jnp.uint8))


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedProceduralMemoryPolicyConfig:
    """Static proposal gates and exact input dimensions."""

    n_actions: int
    outcome_dim: int
    min_evidence_count: int
    min_success_lower_bound: float
    wilson_z: float
    max_outcome_standard_error: float
    max_abs_outcome_mean: float

    SCHEMA_VERSION: ClassVar[str] = CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.n_actions, name="n_actions")
        _positive_int(self.outcome_dim, name="outcome_dim")
        _positive_int(self.min_evidence_count, name="min_evidence_count")
        if self.n_actions > _MAX_ACTIONS:
            raise ValueError("n_actions exceeds the fixed action ceiling")
        if self.outcome_dim > _MAX_OUTCOME_DIM:
            raise ValueError("outcome_dim exceeds the fixed outcome ceiling")
        if self.min_evidence_count < 2:
            raise ValueError("min_evidence_count must be at least 2 for uncertainty")
        if self.min_evidence_count > _INT32_MAX:
            raise ValueError("min_evidence_count exceeds signed-int32")
        _probability(self.min_success_lower_bound, name="min_success_lower_bound")
        _positive_float(self.wilson_z, name="wilson_z")
        _positive_float(
            self.max_outcome_standard_error,
            name="max_outcome_standard_error",
        )
        _positive_float(self.max_abs_outcome_mean, name="max_abs_outcome_mean")

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["schema"] = self.SCHEMA_VERSION
        payload["stateless"] = True
        payload["checkpoint_required"] = False
        payload["action_dispatch_authority"] = False
        payload["memory_mutation_authority"] = False
        payload["scientific_promotion_allowed"] = False
        return payload

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedProceduralMemoryPolicyConfig:
        if type(value) is not dict:
            raise ValueError("policy config must be an exact dict")
        raw = cast(dict[object, object], value)
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "stateless": True,
            "checkpoint_required": False,
            "action_dispatch_authority": False,
            "memory_mutation_authority": False,
            "scientific_promotion_allowed": False,
        }
        expected = {field.name for field in dataclasses.fields(cls)} | set(fixed)
        if set(raw) != expected:
            raise ValueError("policy config fields differ from schema v1")
        if any(
            type(raw[name]) is not type(item) or raw[name] != item for name, item in fixed.items()
        ):
            raise ValueError("policy config fixed fields differ")
        kwargs = {name: raw[name] for name in expected if name not in fixed}
        result = cls(**cast(Any, kwargs))
        if result.to_config() != raw:
            raise ValueError("policy config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryPolicyDiagnostics:
    """Every independent gate used by one proposal decision."""

    retrieval_contract_valid: Bool[Array, ""]
    retrieval_accepted: Bool[Array, ""]
    retrieval_transaction_applied: Bool[Array, ""]
    retrieval_state_valid: Bool[Array, ""]
    retrieval_request_valid: Bool[Array, ""]
    retrieval_identity_found: Bool[Array, ""]
    retrieval_compatible: Bool[Array, ""]
    retrieval_fresh: Bool[Array, ""]
    retrieval_confidence_ok: Bool[Array, ""]
    lifecycle_available: Bool[Array, ""]
    lifecycle_matches: Bool[Array, ""]
    evidence_ready: Bool[Array, ""]
    counts_consistent: Bool[Array, ""]
    success_lower_bound: Float[Array, ""]
    success_bound_ready: Bool[Array, ""]
    outcome_finite: Bool[Array, ""]
    outcome_in_bounds: Bool[Array, ""]
    uncertainty_available: Bool[Array, ""]
    uncertainty_finite: Bool[Array, ""]
    uncertainty_ready: Bool[Array, ""]
    outcome_standard_error: Float[Array, " outcome_dim"]
    score_mass_finite: Bool[Array, ""]
    score_mass_nonnegative: Bool[Array, ""]
    any_positive_mass: Bool[Array, ""]
    hard_safety_mask_provided: Bool[Array, ""]
    any_safe_action: Bool[Array, ""]
    any_safe_positive_mass: Bool[Array, ""]
    unavailable_reason: Int[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryProposal:
    """A deterministic suggestion only; ``action`` is never dispatched."""

    available: Bool[Array, ""]
    action: Int[Array, ""]
    selected_mass: Float[Array, ""]
    safe_mass_total: Float[Array, ""]
    categorical_score_mass: Float[Array, " n_actions"]
    hard_safety_mask: Bool[Array, " n_actions"]
    safe_positive_mask: Bool[Array, " n_actions"]
    diagnostics: ConsolidatedProceduralMemoryPolicyDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedProceduralMemoryPolicyResourceBudget:
    """Exact logical work and zero-state/no-authority declaration."""

    persistent_logical_scalars: int
    persistent_state_bytes: int
    checkpoint_bytes: int
    n_actions: int
    outcome_dim: int
    score_cells_scanned_per_proposal: int
    outcome_cells_scanned_per_proposal: int
    wilson_square_roots_per_proposal: int
    deterministic_argmax_calls_per_proposal: int
    random_generator_calls_per_proposal: int
    memory_queries_per_proposal: int
    memory_writes_per_proposal: int
    action_dispatches_per_proposal: int
    agent_parameter_mutations_per_proposal: int
    proposal_authority: bool
    action_dispatch_authority: bool
    memory_mutation_authority: bool
    agent_mutation_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_required: bool


class ConsolidatedProceduralMemoryPolicy:
    """Stateless conservative proposal boundary for procedural retrievals."""

    def __init__(self, config: ConsolidatedProceduralMemoryPolicyConfig) -> None:
        if type(config) is not ConsolidatedProceduralMemoryPolicyConfig:
            raise TypeError("config must be an exact ConsolidatedProceduralMemoryPolicyConfig")
        self._config = config
        encoded = json.dumps(config.to_config(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        self._identity_sha256 = hashlib.sha256(encoded).hexdigest()

    @property
    def config(self) -> ConsolidatedProceduralMemoryPolicyConfig:
        return self._config

    @property
    def identity_sha256(self) -> str:
        """Return the exact stateless policy/config identity."""

        return self._identity_sha256

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedProceduralMemoryPolicy:
        return cls(ConsolidatedProceduralMemoryPolicyConfig.from_config(value))

    @property
    def resource_budget(self) -> ConsolidatedProceduralMemoryPolicyResourceBudget:
        cfg = self._config
        return ConsolidatedProceduralMemoryPolicyResourceBudget(
            persistent_logical_scalars=0,
            persistent_state_bytes=0,
            checkpoint_bytes=0,
            n_actions=cfg.n_actions,
            outcome_dim=cfg.outcome_dim,
            score_cells_scanned_per_proposal=cfg.n_actions,
            outcome_cells_scanned_per_proposal=cfg.outcome_dim,
            wilson_square_roots_per_proposal=1,
            deterministic_argmax_calls_per_proposal=1,
            random_generator_calls_per_proposal=0,
            memory_queries_per_proposal=0,
            memory_writes_per_proposal=0,
            action_dispatches_per_proposal=0,
            agent_parameter_mutations_per_proposal=0,
            proposal_authority=True,
            action_dispatch_authority=False,
            memory_mutation_authority=False,
            agent_mutation_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_required=False,
        )

    def _validate_retrieval_static(self, retrieval: ProceduralMemoryRetrieval) -> None:
        if not isinstance(retrieval, ProceduralMemoryRetrieval):
            raise TypeError("retrieval must be a ProceduralMemoryRetrieval")
        cfg = self._config
        for name, shape in {
            "payload": (cfg.n_actions,),
            "outcome_mean": (cfg.outcome_dim,),
            "outcome_m2": (cfg.outcome_dim,),
        }.items():
            _require_array(
                getattr(retrieval, name),
                name=f"retrieval.{name}",
                shape=shape,
                dtype=jnp.float32,
            )
        _require_array(
            retrieval.lifecycle_digest,
            name="retrieval.lifecycle_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        for name in (
            "confidence",
            "evidence_mean",
            "evidence_m2",
        ):
            _require_array(
                getattr(retrieval, name),
                name=f"retrieval.{name}",
                shape=(),
                dtype=jnp.float32,
            )
        for name in (
            "slot",
            "evidence_count",
            "success_count",
            "failure_count",
            "lifecycle_generation",
            "lifecycle_revision",
        ):
            _require_array(
                getattr(retrieval, name),
                name=f"retrieval.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        for name in (
            "accepted",
            "transaction_applied",
            "lifecycle_link_available",
            "state_valid",
            "request_valid",
            "identity_found",
            "compatible",
            "fresh",
            "confidence_ok",
        ):
            _require_array(
                getattr(retrieval, name),
                name=f"retrieval.{name}",
                shape=(),
                dtype=jnp.bool_,
            )

    def propose(
        self,
        retrieval: ProceduralMemoryRetrieval,
        *,
        hard_safety_mask: Array,
        expected_lifecycle_digest: Array,
        expected_lifecycle_generation: int | Array,
        expected_lifecycle_revision: int | Array,
    ) -> ConsolidatedProceduralMemoryProposal:
        """Return a suggestion or explicit fail-closed diagnostics."""

        self._validate_retrieval_static(retrieval)
        _require_array(
            hard_safety_mask,
            name="hard_safety_mask",
            shape=(self._config.n_actions,),
            dtype=jnp.bool_,
        )
        _require_array(
            expected_lifecycle_digest,
            name="expected_lifecycle_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        generation = _int32_scalar(
            expected_lifecycle_generation,
            name="expected_lifecycle_generation",
        )
        revision = _int32_scalar(
            expected_lifecycle_revision,
            name="expected_lifecycle_revision",
        )
        return cast(
            ConsolidatedProceduralMemoryProposal,
            self._propose_jit(
                retrieval,
                hard_safety_mask,
                expected_lifecycle_digest,
                generation,
                revision,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_jit(
        self,
        retrieval: ProceduralMemoryRetrieval,
        hard_safety_mask: Array,
        expected_lifecycle_digest: Array,
        expected_lifecycle_generation: Array,
        expected_lifecycle_revision: Array,
    ) -> ConsolidatedProceduralMemoryProposal:
        cfg = self._config
        unavailable_payload_honest = (
            jnp.all(retrieval.payload == 0.0)
            & (retrieval.confidence == 0.0)
            & (retrieval.evidence_count == 0)
            & (retrieval.evidence_mean == 0.0)
            & (retrieval.evidence_m2 == 0.0)
            & (retrieval.success_count == 0)
            & (retrieval.failure_count == 0)
            & jnp.all(retrieval.outcome_mean == 0.0)
            & jnp.all(retrieval.outcome_m2 == 0.0)
            & (~retrieval.lifecycle_link_available)
            & (~_digest_is_nonzero(retrieval.lifecycle_digest))
            & (retrieval.lifecycle_generation == -1)
            & (retrieval.lifecycle_revision == -1)
        )
        slot_contract = jnp.where(retrieval.accepted, retrieval.slot >= 0, retrieval.slot == -1)
        retrieval_contract_valid = (
            jnp.isfinite(retrieval.confidence)
            & (retrieval.confidence >= 0.0)
            & (retrieval.confidence <= 1.0)
            & jnp.isfinite(retrieval.evidence_mean)
            & jnp.isfinite(retrieval.evidence_m2)
            & (retrieval.evidence_m2 >= 0.0)
            & slot_contract
            & (retrieval.accepted | unavailable_payload_honest)
        )
        retrieval_available = (
            retrieval.accepted
            & retrieval.transaction_applied
            & retrieval.state_valid
            & retrieval.request_valid
            & retrieval.identity_found
            & retrieval.compatible
            & retrieval.fresh
            & retrieval.confidence_ok
        )
        expected_lifecycle_valid = (
            _digest_is_nonzero(expected_lifecycle_digest)
            & (expected_lifecycle_generation >= 0)
            & (expected_lifecycle_revision >= 0)
        )
        lifecycle_available = retrieval.lifecycle_link_available & _digest_is_nonzero(
            retrieval.lifecycle_digest
        )
        lifecycle_matches = (
            expected_lifecycle_valid
            & lifecycle_available
            & jnp.array_equal(retrieval.lifecycle_digest, expected_lifecycle_digest)
            & (retrieval.lifecycle_generation == expected_lifecycle_generation)
            & (retrieval.lifecycle_revision == expected_lifecycle_revision)
        )
        counts_nonnegative = (
            (retrieval.evidence_count >= 0)
            & (retrieval.success_count >= 0)
            & (retrieval.failure_count >= 0)
        )
        counts_consistent = (
            counts_nonnegative
            & (retrieval.success_count <= retrieval.evidence_count)
            & (retrieval.failure_count == retrieval.evidence_count - retrieval.success_count)
        )
        evidence_ready = counts_consistent & (
            retrieval.evidence_count >= jnp.asarray(cfg.min_evidence_count, dtype=jnp.int32)
        )
        safe_count = jnp.maximum(retrieval.evidence_count, 1).astype(jnp.float32)
        successes = jnp.maximum(retrieval.success_count, 0).astype(jnp.float32)
        p_hat = successes / safe_count
        z = jnp.asarray(cfg.wilson_z, dtype=jnp.float32)
        z_squared = z * z
        denominator = 1.0 + z_squared / safe_count
        center = p_hat + z_squared / (2.0 * safe_count)
        margin = z * jnp.sqrt(
            jnp.maximum(
                p_hat * (1.0 - p_hat) / safe_count + z_squared / (4.0 * safe_count * safe_count),
                0.0,
            )
        )
        raw_success_lower_bound = (center - margin) / denominator
        success_lower_bound = jnp.where(
            counts_consistent, jnp.clip(raw_success_lower_bound, 0.0, 1.0), 0.0
        )
        success_bound_ready = evidence_ready & (
            success_lower_bound >= jnp.asarray(cfg.min_success_lower_bound, dtype=jnp.float32)
        )
        outcome_finite = jnp.all(jnp.isfinite(retrieval.outcome_mean)) & jnp.all(
            jnp.isfinite(retrieval.outcome_m2)
        )
        outcome_in_bounds = outcome_finite & jnp.all(
            jnp.abs(retrieval.outcome_mean)
            <= jnp.asarray(cfg.max_abs_outcome_mean, dtype=jnp.float32)
        )
        uncertainty_available = (
            counts_consistent
            & (retrieval.evidence_count >= 2)
            & jnp.all(retrieval.outcome_m2 >= 0.0)
        )
        variance_denominator = jnp.maximum(retrieval.evidence_count - 1, 1).astype(jnp.float32)
        sample_variance = jnp.maximum(retrieval.outcome_m2, 0.0) / variance_denominator
        outcome_standard_error = jnp.sqrt(
            sample_variance / jnp.maximum(retrieval.evidence_count, 1).astype(jnp.float32)
        )
        uncertainty_finite = uncertainty_available & jnp.all(jnp.isfinite(outcome_standard_error))
        uncertainty_ready = uncertainty_finite & jnp.all(
            outcome_standard_error <= jnp.asarray(cfg.max_outcome_standard_error, dtype=jnp.float32)
        )
        score_mass_total = jnp.sum(retrieval.payload)
        score_mass_finite = jnp.all(jnp.isfinite(retrieval.payload)) & jnp.isfinite(
            score_mass_total
        )
        score_mass_nonnegative = score_mass_finite & jnp.all(retrieval.payload >= 0.0)
        positive_mass = retrieval.payload > 0.0
        any_positive_mass = score_mass_nonnegative & jnp.any(positive_mass)
        any_safe_action = jnp.any(hard_safety_mask)
        safe_positive_mask = hard_safety_mask & positive_mass & score_mass_nonnegative
        any_safe_positive_mass = jnp.any(safe_positive_mask)
        safe_scores = jnp.where(
            safe_positive_mask,
            retrieval.payload,
            jnp.asarray(-jnp.inf, dtype=jnp.float32),
        )
        selected = jnp.argmax(safe_scores).astype(jnp.int32)
        available = (
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & evidence_ready
            & success_bound_ready
            & outcome_in_bounds
            & uncertainty_ready
            & score_mass_nonnegative
            & any_positive_mass
            & any_safe_action
            & any_safe_positive_mass
        )
        reason = jnp.asarray(POLICY_REASON_AVAILABLE, dtype=jnp.int32)
        reason = jnp.where(
            retrieval_contract_valid,
            reason,
            jnp.asarray(POLICY_REASON_RETRIEVAL_CONTRACT, dtype=jnp.int32),
        )
        reason = jnp.where(
            retrieval_contract_valid & (~retrieval_available),
            jnp.asarray(POLICY_REASON_RETRIEVAL_UNAVAILABLE, dtype=jnp.int32),
            reason,
        )
        compatibility_ok = (
            retrieval.state_valid
            & retrieval.request_valid
            & retrieval.identity_found
            & retrieval.compatible
            & retrieval.fresh
            & retrieval.confidence_ok
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval.transaction_applied
            & retrieval.identity_found
            & (~compatibility_ok),
            jnp.asarray(POLICY_REASON_COMPATIBILITY, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid & retrieval_available & (~lifecycle_matches),
            jnp.asarray(POLICY_REASON_LIFECYCLE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & counts_consistent
            & (~evidence_ready),
            jnp.asarray(POLICY_REASON_EVIDENCE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & (~counts_consistent),
            jnp.asarray(POLICY_REASON_COUNT_INCONSISTENCY, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & evidence_ready
            & (~success_bound_ready),
            jnp.asarray(POLICY_REASON_SUCCESS_BOUND, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & evidence_ready
            & success_bound_ready
            & (~outcome_in_bounds),
            jnp.asarray(POLICY_REASON_OUTCOME, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & evidence_ready
            & success_bound_ready
            & outcome_in_bounds
            & (~uncertainty_ready),
            jnp.asarray(POLICY_REASON_UNCERTAINTY, dtype=jnp.int32),
            reason,
        )
        pre_mass_gates = (
            retrieval_contract_valid
            & retrieval_available
            & lifecycle_matches
            & evidence_ready
            & success_bound_ready
            & outcome_in_bounds
            & uncertainty_ready
        )
        reason = jnp.where(
            pre_mass_gates & ((~score_mass_nonnegative) | (~any_positive_mass)),
            jnp.asarray(POLICY_REASON_SCORE_MASS, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            pre_mass_gates & score_mass_nonnegative & any_positive_mass & (~any_safe_action),
            jnp.asarray(POLICY_REASON_NO_SAFE_ACTION, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            pre_mass_gates
            & score_mass_nonnegative
            & any_positive_mass
            & any_safe_action
            & (~any_safe_positive_mass),
            jnp.asarray(POLICY_REASON_NO_SAFE_POSITIVE_MASS, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(available, POLICY_REASON_AVAILABLE, reason).astype(jnp.int32)
        selected_mass = jnp.where(available, retrieval.payload[selected], 0.0)
        safe_mass_total = jnp.where(
            score_mass_nonnegative,
            jnp.sum(jnp.where(hard_safety_mask, retrieval.payload, 0.0)),
            0.0,
        )
        diagnostics = ConsolidatedProceduralMemoryPolicyDiagnostics(
            retrieval_contract_valid=retrieval_contract_valid,
            retrieval_accepted=retrieval.accepted,
            retrieval_transaction_applied=retrieval.transaction_applied,
            retrieval_state_valid=retrieval.state_valid,
            retrieval_request_valid=retrieval.request_valid,
            retrieval_identity_found=retrieval.identity_found,
            retrieval_compatible=retrieval.compatible,
            retrieval_fresh=retrieval.fresh,
            retrieval_confidence_ok=retrieval.confidence_ok,
            lifecycle_available=lifecycle_available,
            lifecycle_matches=lifecycle_matches,
            evidence_ready=evidence_ready,
            counts_consistent=counts_consistent,
            success_lower_bound=success_lower_bound,
            success_bound_ready=success_bound_ready,
            outcome_finite=outcome_finite,
            outcome_in_bounds=outcome_in_bounds,
            uncertainty_available=uncertainty_available,
            uncertainty_finite=uncertainty_finite,
            uncertainty_ready=uncertainty_ready,
            outcome_standard_error=jnp.where(
                uncertainty_finite,
                outcome_standard_error,
                jnp.zeros_like(outcome_standard_error),
            ),
            score_mass_finite=score_mass_finite,
            score_mass_nonnegative=score_mass_nonnegative,
            any_positive_mass=any_positive_mass,
            hard_safety_mask_provided=jnp.asarray(True, dtype=jnp.bool_),
            any_safe_action=any_safe_action,
            any_safe_positive_mass=any_safe_positive_mass,
            unavailable_reason=reason,
        )
        return ConsolidatedProceduralMemoryProposal(
            available=available,
            action=jnp.where(available, selected, -1).astype(jnp.int32),
            selected_mass=selected_mass.astype(jnp.float32),
            safe_mass_total=safe_mass_total.astype(jnp.float32),
            categorical_score_mass=jnp.where(
                score_mass_finite,
                retrieval.payload,
                jnp.zeros_like(retrieval.payload),
            ),
            hard_safety_mask=hard_safety_mask,
            safe_positive_mask=safe_positive_mask,
            diagnostics=diagnostics,
        )


__all__ = [
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_ACTION_DISPATCH_AUTHORITY",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_AGENT_MUTATION_AUTHORITY",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CHECKPOINT_REQUIRED",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CONFIG_SCHEMA",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_MEMORY_MUTATION_AUTHORITY",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROMOTION_AUTHORITY",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROPOSAL_AUTHORITY",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_SCIENTIFIC_PROMOTION_ALLOWED",
    "CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_STATELESS",
    "POLICY_REASON_AVAILABLE",
    "POLICY_REASON_COMPATIBILITY",
    "POLICY_REASON_COUNT_INCONSISTENCY",
    "POLICY_REASON_EVIDENCE",
    "POLICY_REASON_LIFECYCLE",
    "POLICY_REASON_NO_SAFE_ACTION",
    "POLICY_REASON_NO_SAFE_POSITIVE_MASS",
    "POLICY_REASON_OUTCOME",
    "POLICY_REASON_RETRIEVAL_CONTRACT",
    "POLICY_REASON_RETRIEVAL_UNAVAILABLE",
    "POLICY_REASON_SCORE_MASS",
    "POLICY_REASON_SUCCESS_BOUND",
    "POLICY_REASON_UNCERTAINTY",
    "ConsolidatedProceduralMemoryPolicy",
    "ConsolidatedProceduralMemoryPolicyConfig",
    "ConsolidatedProceduralMemoryPolicyDiagnostics",
    "ConsolidatedProceduralMemoryPolicyResourceBudget",
    "ConsolidatedProceduralMemoryProposal",
]
