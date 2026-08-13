# mypy: disable-error-code="call-arg"
"""Mechanism-only adjacent-cube causal attribution for the staged HCCL design.

This standalone kernel implements only the first code rung specified in
``CONTINUAL_DYAD_BENCHMARK.md``: validate one immutable source receipt, one
immutable exogenous receipt, and distinct B/M/P action receipts for two agents;
stage the memory and planner dyad cubes in the exact eight-call order; require
the duplicated MM proposal to be bit-exact; compute typed immediate contrasts;
and persist only the PP proposal in one atomic state transaction.

It does not implement the HCCL environment, agents, stochastic draws, message
semantics, safety functions, run geometry, artifact schema, evaluator,
threshold, or claim.  In particular, the caller supplies already-materialized
source and exogenous receipts.  Their unkeyed tags provide deterministic
integrity binding, not authentication, and this module neither pins nor
generates missing noise/seed semantics.

Eager calls perform a host preflight when every input is concrete.  Invalid,
stale, tampered, identity-aliased, or exhausted sources then invoke zero
proposal callbacks and return the exact state.  Under JAX tracing, Python
callbacks must be staged while validity remains dynamic; final state still
rolls back, but tracing cannot promise zero callback staging.  This limitation
is explicit in config/checkpoint metadata and diagnostics.

All eight proposals consume the same source, exogenous receipt, hard mask, and
fixed delivered-message charge.  Calls are
fixed as ``MM, B0M1, M0B1, BB, PP, M0P1, P0M1, MM`` even when actions coincide.
Exactly seven slots are designated counterfactuals; only valid PP may persist.
If PP is rejected, all eight staged proposals are discarded.
This is L0 ``mechanism-only`` infrastructure with no HCCL execution, evidence,
promotion, Alberta Plan completion, or SOTA authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from enum import IntEnum
from numbers import Real
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

HCCL_CAUSAL_ATTRIBUTION_CONFIG_SCHEMA = "alberta.hccl-causal-attribution-config.v1"
HCCL_CAUSAL_ATTRIBUTION_STATE_SCHEMA = "alberta.hccl-causal-attribution-state.v1"
HCCL_CAUSAL_ATTRIBUTION_SOURCE_SCHEMA = "alberta.hccl-causal-source-receipt.v1"
HCCL_CAUSAL_ATTRIBUTION_EXOGENOUS_SCHEMA = "alberta.hccl-exogenous-receipt.v1"
HCCL_CAUSAL_ATTRIBUTION_ACTION_SCHEMA = "alberta.hccl-action-receipt.v1"
HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_SCHEMA = "alberta.hccl-transition-proposal.v1"
HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA = "alberta.hccl-causal-attribution-checkpoint.v1"
HCCL_CAUSAL_ATTRIBUTION_RESOURCE_SCHEMA = "alberta.hccl-causal-attribution-resource.v1"
HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL = "L0"
HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS = "mechanism-only"
HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER = (
    "MM-memory",
    "B0M1-memory",
    "M0B1-memory",
    "BB-memory",
    "PP-planner",
    "M0P1-planner",
    "P0M1-planner",
    "MM-planner-duplicate",
)
HCCL_CAUSAL_ATTRIBUTION_LIMITATIONS = (
    "host-preflight-cannot-suppress-traced-callback-staging",
    "proposal-callback-purity-and-side-effect-freedom-are-caller-obligations",
    "generic-kernel-does-not-enforce-equal-payloads-for-equal-effective-actions",
    "caller-materialized-source-and-exogenous-receipts-not-authenticated",
    "environment-noise-seed-and-draw-semantics-unpinned",
    "typed-message-and-safety-signals-have-no-pinned-environment-functions",
    "immediate-same-prestate-effects-not-long-run-policy-returns",
    "no-agent-environment-run-artifact-validator-threshold-or-evidence-authority",
)

_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_N_AGENTS = 2
_IDENTITY_WORDS = 4
_TAG_WORDS = 4
_OWNER_WORDS = 8
_N_PROPOSALS = 8
_N_NONCOMMITTING = 7
_PP_SLOT = 4
_FLOAT32_EPS = float(np.finfo(np.float32).eps)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


class HCCLActionLayer(IntEnum):
    """Typed source of one prepared effective primitive action."""

    BASE = 0
    MEMORY = 1
    PLANNER = 2


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


def _exact_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _finite_real(
    value: Any,
    *,
    label: str,
    minimum: float,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    if scalar <= minimum if strict_minimum else scalar < minimum:
        raise ValueError(f"{label} is below its lower bound")
    narrowed = float(np.float32(scalar))
    if not math.isfinite(narrowed) or (scalar != 0.0 and narrowed == 0.0):
        raise ValueError(f"{label} must remain finite and nonzero in float32")
    return scalar


def _owner_digest(value: Any) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != _OWNER_WORDS:
        raise ValueError("proposal_owner_digest must be an exact eight-word tuple")
    result: list[int] = []
    for index, word in enumerate(value):
        if type(word) is not int or not 0 <= word <= _UINT32_MAX:
            raise ValueError(f"proposal_owner_digest[{index}] must be uint32")
        result.append(word)
    return tuple(result)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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


def _words(value: Any, shape: tuple[int, ...], *, label: str) -> Array:
    return _require_array(value, shape=shape, dtype=jnp.dtype(jnp.uint32), label=label)


def _bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.bool_), label=label)


def _int_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.int32), label=label)


def _float_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.float32), label=label)


def _float_vector(value: Any, width: int, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(width,),
        dtype=jnp.dtype(jnp.float32),
        label=label,
    )


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


def _words_not_later(candidate: Array, reference: Array) -> Bool[Array, ""]:
    return (candidate[0] < reference[0]) | (
        (candidate[0] == reference[0]) & (candidate[1] <= reference[1])
    )


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _tree_exact_equal(left: Any, right: Any) -> Bool[Array, ""]:
    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(object, left_structure) != cast(object, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    result = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if left_array.dtype == jnp.dtype(jnp.float32):
            result = result & _float_bits_equal(left_array, right_array)
        else:
            result = result & jnp.array_equal(left_array, right_array)
    return result


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def _contains_tracer(value: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _rotate_left(value: Array, distance: Array) -> Array:
    right = (jnp.asarray(32, dtype=jnp.uint32) - distance) & jnp.uint32(31)
    return jnp.asarray((value << distance) | (value >> right), dtype=jnp.uint32)


def _content_tag(owner: Array, *values: Array) -> UInt[Array, " 4"]:
    words: list[Array] = [jnp.reshape(owner, (-1,))]
    for value in values:
        array = jax.lax.stop_gradient(jnp.asarray(value))
        if array.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
            converted = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.bool_):
            converted = array.astype(jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.uint32):
            converted = array
        else:
            raise TypeError("content tag values must be float32/int32/bool/uint32")
        words.append(jnp.reshape(converted, (-1,)))
    payload = jnp.concatenate(tuple(words)).astype(jnp.uint32)
    indices = jnp.arange(payload.shape[0], dtype=jnp.uint32)
    mixed = _rotate_left(
        payload ^ (indices * jnp.uint32(0x9E3779B9)),
        (indices % jnp.uint32(31)) + jnp.uint32(1),
    )
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


@dataclasses.dataclass(frozen=True)
class HCCLCausalAttributionConfig:
    """Fixed dimensions, bounds, owner digest, and exact lifetime."""

    source_dim: int
    exogenous_dim: int
    transition_dim: int
    n_actions: int
    proposal_owner_digest: tuple[int, ...]
    max_transactions: int = _UINT64_MAX
    max_abs_source: float = 1.0e6
    max_abs_exogenous: float = 1.0e6
    max_abs_transition: float = 1.0e6
    max_abs_signal: float = 1.0e6

    def __post_init__(self) -> None:
        _exact_int(self.source_dim, label="source_dim", minimum=1, maximum=4096)
        _exact_int(self.exogenous_dim, label="exogenous_dim", minimum=1, maximum=4096)
        _exact_int(self.transition_dim, label="transition_dim", minimum=1, maximum=4096)
        if self.n_actions != 2 or type(self.n_actions) is not int:
            raise ValueError("HCCL causal attribution requires exactly two primitive actions")
        _owner_digest(self.proposal_owner_digest)
        _exact_int(
            self.max_transactions,
            label="max_transactions",
            minimum=1,
            maximum=_UINT64_MAX,
        )
        for name in (
            "max_abs_source",
            "max_abs_exogenous",
            "max_abs_transition",
            "max_abs_signal",
        ):
            if type(getattr(self, name)) is not float:
                raise TypeError(f"{name} must be an exact float")
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "HCCLCausalAttributionKernel",
            "schema": HCCL_CAUSAL_ATTRIBUTION_CONFIG_SCHEMA,
            "state_schema": HCCL_CAUSAL_ATTRIBUTION_STATE_SCHEMA,
            "source_schema": HCCL_CAUSAL_ATTRIBUTION_SOURCE_SCHEMA,
            "exogenous_schema": HCCL_CAUSAL_ATTRIBUTION_EXOGENOUS_SCHEMA,
            "action_schema": HCCL_CAUSAL_ATTRIBUTION_ACTION_SCHEMA,
            "proposal_schema": HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_SCHEMA,
            "checkpoint_schema": HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA,
            "resource_schema": HCCL_CAUSAL_ATTRIBUTION_RESOURCE_SCHEMA,
            "evidence_level": HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL,
            "mechanism_status": HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "hccl_execution_authorized": False,
            "environment_implemented": False,
            "noise_and_seed_semantics_pinned": False,
            "artifact_or_claim_authority": False,
            "proposal_order": list(HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER),
            "committed_slot": "PP-planner",
            "proposal_calls_per_valid_transaction": _N_PROPOSALS,
            "designated_counterfactual_slots_per_transaction": _N_NONCOMMITTING,
            "max_discarded_proposals_per_transaction": _N_PROPOSALS,
            "max_unique_effective_joint_actions": min(
                self.n_actions**_N_AGENTS,
                _N_NONCOMMITTING,
            ),
            "max_unique_joint_action_receipt_vertices": _N_NONCOMMITTING,
            "limitations": list(HCCL_CAUSAL_ATTRIBUTION_LIMITATIONS),
            "source_dim": self.source_dim,
            "exogenous_dim": self.exogenous_dim,
            "transition_dim": self.transition_dim,
            "n_actions": self.n_actions,
            "proposal_owner_digest": list(self.proposal_owner_digest),
            "max_transactions": self.max_transactions,
            "max_abs_source": float(self.max_abs_source),
            "max_abs_exogenous": float(self.max_abs_exogenous),
            "max_abs_transition": float(self.max_abs_transition),
            "max_abs_signal": float(self.max_abs_signal),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> HCCLCausalAttributionConfig:
        fields = _exact_manifest(
            payload,
            {
                "type",
                "schema",
                "state_schema",
                "source_schema",
                "exogenous_schema",
                "action_schema",
                "proposal_schema",
                "checkpoint_schema",
                "resource_schema",
                "evidence_level",
                "mechanism_status",
                "scientific_promotion_allowed",
                "hccl_execution_authorized",
                "environment_implemented",
                "noise_and_seed_semantics_pinned",
                "artifact_or_claim_authority",
                "proposal_order",
                "committed_slot",
                "proposal_calls_per_valid_transaction",
                "designated_counterfactual_slots_per_transaction",
                "max_discarded_proposals_per_transaction",
                "max_unique_effective_joint_actions",
                "max_unique_joint_action_receipt_vertices",
                "limitations",
                "source_dim",
                "exogenous_dim",
                "transition_dim",
                "n_actions",
                "proposal_owner_digest",
                "max_transactions",
                "max_abs_source",
                "max_abs_exogenous",
                "max_abs_transition",
                "max_abs_signal",
            },
            label="HCCL causal attribution config",
        )
        fixed = {
            "type": "HCCLCausalAttributionKernel",
            "schema": HCCL_CAUSAL_ATTRIBUTION_CONFIG_SCHEMA,
            "state_schema": HCCL_CAUSAL_ATTRIBUTION_STATE_SCHEMA,
            "source_schema": HCCL_CAUSAL_ATTRIBUTION_SOURCE_SCHEMA,
            "exogenous_schema": HCCL_CAUSAL_ATTRIBUTION_EXOGENOUS_SCHEMA,
            "action_schema": HCCL_CAUSAL_ATTRIBUTION_ACTION_SCHEMA,
            "proposal_schema": HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_SCHEMA,
            "checkpoint_schema": HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA,
            "resource_schema": HCCL_CAUSAL_ATTRIBUTION_RESOURCE_SCHEMA,
            "evidence_level": HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL,
            "mechanism_status": HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "hccl_execution_authorized": False,
            "environment_implemented": False,
            "noise_and_seed_semantics_pinned": False,
            "artifact_or_claim_authority": False,
            "proposal_order": list(HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER),
            "committed_slot": "PP-planner",
            "proposal_calls_per_valid_transaction": _N_PROPOSALS,
            "designated_counterfactual_slots_per_transaction": _N_NONCOMMITTING,
            "max_discarded_proposals_per_transaction": _N_PROPOSALS,
            "max_unique_joint_action_receipt_vertices": _N_NONCOMMITTING,
            "limitations": list(HCCL_CAUSAL_ATTRIBUTION_LIMITATIONS),
        }
        for name, expected in fixed.items():
            if fields.pop(name) != expected:
                raise ValueError(f"HCCL causal attribution config {name} is unsupported")
        max_unique_actions = fields.pop("max_unique_effective_joint_actions")
        digest = fields.pop("proposal_owner_digest")
        if type(digest) is not list:
            raise TypeError("proposal_owner_digest must serialize as a list")
        candidate = cls(proposal_owner_digest=tuple(digest), **fields)
        expected_unique_actions = min(
            candidate.n_actions**_N_AGENTS,
            _N_NONCOMMITTING,
        )
        if max_unique_actions != expected_unique_actions:
            raise ValueError(
                "HCCL causal attribution config max_unique_effective_joint_actions "
                "is unsupported"
            )
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("HCCL causal attribution config types are noncanonical")
        return candidate


@chex.dataclass(frozen=True)
class HCCLTypedSignals:
    """Unambiguous proposal-specific task, net, safety, and message channels."""

    task_score: Float[Array, ""]
    net_reward: Float[Array, " 2"]
    safety_cost: Float[Array, " 2"]
    message_charge: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLSignalContrast:
    task_score: Float[Array, ""]
    net_reward: Float[Array, " 2"]
    safety_cost: Float[Array, " 2"]
    message_charge: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLCausalContrasts:
    memory_total: HCCLSignalContrast
    memory_interaction: HCCLSignalContrast
    planner_total: HCCLSignalContrast
    planner_interaction: HCCLSignalContrast
    pp_minus_bb: HCCLSignalContrast
    telescoping_sum: HCCLSignalContrast
    telescoping_residual: HCCLSignalContrast


@chex.dataclass(frozen=True)
class HCCLCausalSourceReceipt:
    source_vector: Float[Array, " source"]
    source_identity_words: UInt[Array, " 4"]
    agent_identity_words: UInt[Array, "2 4"]
    decision_words: UInt[Array, " 2"]
    source_transition_words: UInt[Array, " 2"]
    raw_observation_identity_words: UInt[Array, "2 4"]
    fast_state_words: UInt[Array, "2 2"]
    slow_context_birth_words: UInt[Array, "2 4"]
    feature_birth_words: UInt[Array, "2 4"]
    memory_generation_words: UInt[Array, "2 2"]
    planner_model_words: UInt[Array, "2 2"]
    hard_mask_generation_words: UInt[Array, "2 2"]
    rng_receipt_identity_words: UInt[Array, "2 4"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLExogenousReceipt:
    source_identity_words: UInt[Array, " 4"]
    decision_words: UInt[Array, " 2"]
    source_transition_words: UInt[Array, " 2"]
    exogenous_identity_words: UInt[Array, " 4"]
    exogenous_source_words: UInt[Array, " 2"]
    payload: Float[Array, " exogenous"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLActionReceipt:
    source: HCCLCausalSourceReceipt
    exogenous_identity_words: UInt[Array, " 4"]
    exogenous_content_tag_words: UInt[Array, " 4"]
    layer: Int[Array, ""]
    actions_before_mask: Int[Array, " 2"]
    actions_after_mask: Int[Array, " 2"]
    hard_action_masks: Bool[Array, "2 action"]
    action_receipt_identity_words: UInt[Array, "2 4"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLJointActionVertex:
    actions: Int[Array, " 2"]
    action_receipt_identity_words: UInt[Array, "2 4"]
    layer_codes: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLTransitionProposal:
    source_identity_words: UInt[Array, " 4"]
    source_content_tag_words: UInt[Array, " 4"]
    exogenous_identity_words: UInt[Array, " 4"]
    exogenous_content_tag_words: UInt[Array, " 4"]
    source_decision_words: UInt[Array, " 2"]
    source_transition_words: UInt[Array, " 2"]
    destination_transition_words: UInt[Array, " 2"]
    vertex: HCCLJointActionVertex
    candidate_transition: Float[Array, " transition"]
    signals: HCCLTypedSignals
    accepted: Bool[Array, ""]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLCausalAttributionState:
    transaction_words: UInt[Array, " 2"]
    decision_words: UInt[Array, " 2"]
    last_committed_pp: HCCLTransitionProposal
    last_contrasts: HCCLCausalContrasts
    last_attribution_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLCausalAttributionWork:
    proposal_calls: Int[Array, ""]
    designated_counterfactual_slots: Int[Array, ""]
    discarded_proposal_calls: Int[Array, ""]
    committed_pp_calls: Int[Array, ""]
    duplicate_mm_equality_checks: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLCausalAttributionResult:
    state: HCCLCausalAttributionState
    proposals: HCCLTransitionProposal
    contrasts: HCCLCausalContrasts
    work: HCCLCausalAttributionWork
    pre_transaction_words: UInt[Array, " 2"]
    post_transaction_words: UInt[Array, " 2"]
    committed_slot: Int[Array, ""]
    unique_joint_action_vertices: Int[Array, ""]
    unique_joint_action_receipt_vertices: Int[Array, ""]
    host_preflight_performed: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    source_receipt_valid: Bool[Array, ""]
    exogenous_receipt_valid: Bool[Array, ""]
    action_receipts_valid: Bool[Array, ""]
    receipt_identities_distinct: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    preflight_valid: Bool[Array, ""]
    all_child_proposals_valid: Bool[Array, ""]
    duplicate_mm_bit_exact: Bool[Array, ""]
    typed_signals_valid: Bool[Array, ""]
    telescoping_valid: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class HCCLCausalAttributionResourceBudget:
    schema: str
    clock_nbytes: int
    last_committed_proposal_nbytes: int
    last_contrasts_nbytes: int
    last_attribution_tag_nbytes: int
    total_state_nbytes: int
    max_proposal_calls_per_transaction: int
    designated_counterfactual_slots_per_transaction: int
    max_discarded_proposal_calls_per_transaction: int
    max_committed_calls_per_transaction: int
    max_unique_effective_joint_actions: int
    max_unique_joint_action_receipt_vertices: int
    max_transactions: int
    persistent_bytes_scope: str
    transient_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class HCCLCausalAttributionScanResult:
    state: HCCLCausalAttributionState
    memory_total_task_score: Float[Array, " steps"]
    planner_total_task_score: Float[Array, " steps"]
    pp_minus_bb_task_score: Float[Array, " steps"]
    transaction_words: UInt[Array, "steps 2"]
    proposal_calls: Int[Array, " steps"]
    update_applied: Bool[Array, " steps"]


HCCLProposalCallback = Callable[
    [HCCLCausalSourceReceipt, HCCLExogenousReceipt, HCCLJointActionVertex, Array],
    HCCLTransitionProposal,
]


def _empty_signals() -> HCCLTypedSignals:
    return HCCLTypedSignals(
        task_score=jnp.asarray(0.0, dtype=jnp.float32),
        net_reward=jnp.zeros((_N_AGENTS,), dtype=jnp.float32),
        safety_cost=jnp.zeros((_N_AGENTS,), dtype=jnp.float32),
        message_charge=jnp.zeros((_N_AGENTS,), dtype=jnp.float32),
    )


def _empty_contrast() -> HCCLSignalContrast:
    signals = _empty_signals()
    return HCCLSignalContrast(
        task_score=signals.task_score,
        net_reward=signals.net_reward,
        safety_cost=signals.safety_cost,
        message_charge=signals.message_charge,
    )


def _empty_contrasts() -> HCCLCausalContrasts:
    empty = _empty_contrast()
    return HCCLCausalContrasts(
        memory_total=empty,
        memory_interaction=empty,
        planner_total=empty,
        planner_interaction=empty,
        pp_minus_bb=empty,
        telescoping_sum=empty,
        telescoping_residual=empty,
    )


def _empty_vertex() -> HCCLJointActionVertex:
    return HCCLJointActionVertex(
        actions=jnp.full((_N_AGENTS,), -1, dtype=jnp.int32),
        action_receipt_identity_words=jnp.zeros((_N_AGENTS, _IDENTITY_WORDS), dtype=jnp.uint32),
        layer_codes=jnp.full((_N_AGENTS,), -1, dtype=jnp.int32),
    )


def _empty_proposal(config: HCCLCausalAttributionConfig) -> HCCLTransitionProposal:
    return HCCLTransitionProposal(
        source_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        source_content_tag_words=jnp.zeros((_TAG_WORDS,), dtype=jnp.uint32),
        exogenous_identity_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        exogenous_content_tag_words=jnp.zeros((_TAG_WORDS,), dtype=jnp.uint32),
        source_decision_words=jnp.zeros((2,), dtype=jnp.uint32),
        source_transition_words=jnp.zeros((2,), dtype=jnp.uint32),
        destination_transition_words=jnp.zeros((2,), dtype=jnp.uint32),
        vertex=_empty_vertex(),
        candidate_transition=jnp.zeros((config.transition_dim,), dtype=jnp.float32),
        signals=_empty_signals(),
        accepted=jnp.asarray(False, dtype=jnp.bool_),
        content_tag_words=jnp.zeros((_TAG_WORDS,), dtype=jnp.uint32),
    )


def _stack_proposals(
    proposals: tuple[HCCLTransitionProposal, ...],
) -> HCCLTransitionProposal:
    return cast(
        HCCLTransitionProposal,
        jax.tree.map(lambda *leaves: jnp.stack(leaves), *proposals),
    )


def _proposal_at(proposals: HCCLTransitionProposal, index: int) -> HCCLTransitionProposal:
    return cast(HCCLTransitionProposal, jax.tree.map(lambda leaf: leaf[index], proposals))


def _signals_at(proposals: HCCLTransitionProposal, index: int) -> HCCLTypedSignals:
    return _proposal_at(proposals, index).signals


def _signals_binary(
    left: HCCLTypedSignals | HCCLSignalContrast,
    right: HCCLTypedSignals | HCCLSignalContrast,
    operation: Callable[[Array, Array], Array],
) -> HCCLSignalContrast:
    return HCCLSignalContrast(
        task_score=operation(left.task_score, right.task_score),
        net_reward=operation(left.net_reward, right.net_reward),
        safety_cost=operation(left.safety_cost, right.safety_cost),
        message_charge=operation(left.message_charge, right.message_charge),
    )


def _signals_add(
    left: HCCLTypedSignals | HCCLSignalContrast,
    right: HCCLTypedSignals | HCCLSignalContrast,
) -> HCCLSignalContrast:
    return _signals_binary(left, right, jnp.add)


def _signals_subtract(
    left: HCCLTypedSignals | HCCLSignalContrast,
    right: HCCLTypedSignals | HCCLSignalContrast,
) -> HCCLSignalContrast:
    return _signals_binary(left, right, jnp.subtract)


def _signals_four_term(
    positive: HCCLTypedSignals,
    negative_left: HCCLTypedSignals,
    negative_right: HCCLTypedSignals,
    positive_base: HCCLTypedSignals,
) -> HCCLSignalContrast:
    return _signals_add(
        _signals_subtract(
            _signals_subtract(positive, negative_left),
            negative_right,
        ),
        positive_base,
    )


def _derive_contrasts(proposals: HCCLTransitionProposal) -> HCCLCausalContrasts:
    mm = _signals_at(proposals, 0)
    b0m1 = _signals_at(proposals, 1)
    m0b1 = _signals_at(proposals, 2)
    bb = _signals_at(proposals, 3)
    pp = _signals_at(proposals, 4)
    m0p1 = _signals_at(proposals, 5)
    p0m1 = _signals_at(proposals, 6)
    memory_total = _signals_subtract(mm, bb)
    planner_total = _signals_subtract(pp, mm)
    pp_minus_bb = _signals_subtract(pp, bb)
    telescoping_sum = _signals_add(memory_total, planner_total)
    return HCCLCausalContrasts(
        memory_total=memory_total,
        memory_interaction=_signals_four_term(mm, b0m1, m0b1, bb),
        planner_total=planner_total,
        planner_interaction=_signals_four_term(pp, m0p1, p0m1, mm),
        pp_minus_bb=pp_minus_bb,
        telescoping_sum=telescoping_sum,
        telescoping_residual=_signals_subtract(pp_minus_bb, telescoping_sum),
    )


def _signals_finite(value: HCCLTypedSignals | HCCLSignalContrast) -> Bool[Array, ""]:
    return (
        jnp.isfinite(value.task_score)
        & jnp.all(jnp.isfinite(value.net_reward))
        & jnp.all(jnp.isfinite(value.safety_cost))
        & jnp.all(jnp.isfinite(value.message_charge))
    )


def _contrasts_finite(contrasts: HCCLCausalContrasts) -> Bool[Array, ""]:
    result = jnp.asarray(True, dtype=jnp.bool_)
    for value in jax.tree.leaves(contrasts):
        result = result & jnp.all(jnp.isfinite(value))
    return result


def _contrasts_zero(value: HCCLSignalContrast) -> Bool[Array, ""]:
    return jax.tree.reduce(
        jnp.logical_and,
        jax.tree.map(lambda leaf: jnp.all(leaf == 0.0), value),
    )


def _telescoping_roundoff_valid(
    contrasts: HCCLCausalContrasts,
) -> Bool[Array, ""]:
    """Audit the algebra with a scale-aware float32 roundoff envelope.

    ``PP - BB`` and ``(MM - BB) + (PP - MM)`` are mathematically identical,
    but their separately rounded float32 evaluations need not be bit equal.
    The envelope is eight float32 epsilons times the largest stored
    intermediate magnitude, with a normal-minimum floor for subnormals.  This
    bounds the four rounded add/subtract operations without converting a valid
    nonzero residual into a causal effect.
    """

    expected_sum = _signals_add(contrasts.memory_total, contrasts.planner_total)
    expected_residual = _signals_subtract(
        contrasts.pp_minus_bb,
        contrasts.telescoping_sum,
    )
    exact_derivation = _tree_exact_equal(
        contrasts.telescoping_sum,
        expected_sum,
    ) & _tree_exact_equal(
        contrasts.telescoping_residual,
        expected_residual,
    )
    within_roundoff = jnp.asarray(True, dtype=jnp.bool_)
    groups = zip(
        jax.tree.leaves(contrasts.memory_total),
        jax.tree.leaves(contrasts.planner_total),
        jax.tree.leaves(contrasts.pp_minus_bb),
        jax.tree.leaves(contrasts.telescoping_sum),
        jax.tree.leaves(contrasts.telescoping_residual),
        strict=True,
    )
    for memory, planner, direct, summed, residual in groups:
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(memory), jnp.abs(planner)),
            jnp.maximum(jnp.abs(direct), jnp.abs(summed)),
        )
        scale = jnp.maximum(scale, jnp.float32(_FLOAT32_TINY))
        tolerance = jnp.float32(8.0 * _FLOAT32_EPS) * scale
        within_roundoff = within_roundoff & jnp.all(jnp.abs(residual) <= tolerance)
    return _contrasts_finite(contrasts) & exact_derivation & within_roundoff


def _fixed_message_delivery_valid(
    proposals: tuple[HCCLTransitionProposal, ...],
) -> Bool[Array, ""]:
    reference = proposals[0].signals.message_charge
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for proposal in proposals[1:]:
        valid = valid & _float_bits_equal(
            proposal.signals.message_charge,
            reference,
        )
    return valid


def _attribution_tag(
    owner: Array,
    words: Array,
    proposal: HCCLTransitionProposal,
    contrasts: HCCLCausalContrasts,
) -> UInt[Array, " 4"]:
    return _content_tag(
        owner,
        words,
        proposal.content_tag_words,
        contrasts.memory_total.task_score,
        contrasts.memory_total.net_reward,
        contrasts.memory_total.safety_cost,
        contrasts.memory_total.message_charge,
        contrasts.memory_interaction.task_score,
        contrasts.memory_interaction.net_reward,
        contrasts.memory_interaction.safety_cost,
        contrasts.memory_interaction.message_charge,
        contrasts.planner_total.task_score,
        contrasts.planner_total.net_reward,
        contrasts.planner_total.safety_cost,
        contrasts.planner_total.message_charge,
        contrasts.planner_interaction.task_score,
        contrasts.planner_interaction.net_reward,
        contrasts.planner_interaction.safety_cost,
        contrasts.planner_interaction.message_charge,
        contrasts.pp_minus_bb.task_score,
        contrasts.pp_minus_bb.net_reward,
        contrasts.pp_minus_bb.safety_cost,
        contrasts.pp_minus_bb.message_charge,
        contrasts.telescoping_sum.task_score,
        contrasts.telescoping_sum.net_reward,
        contrasts.telescoping_sum.safety_cost,
        contrasts.telescoping_sum.message_charge,
        contrasts.telescoping_residual.task_score,
        contrasts.telescoping_residual.net_reward,
        contrasts.telescoping_residual.safety_cost,
        contrasts.telescoping_residual.message_charge,
    )


def _state_nbytes(state: Any) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def measure_hccl_causal_attribution_state_nbytes(
    state: HCCLCausalAttributionState,
) -> int:
    if type(state) is not HCCLCausalAttributionState:
        raise TypeError("state must be exact HCCLCausalAttributionState")
    return _state_nbytes(state)


class HCCLCausalAttributionKernel:
    """Fixed-work, fail-closed attribution owner with no environment authority."""

    def __init__(self, config: HCCLCausalAttributionConfig) -> None:
        if type(config) is not HCCLCausalAttributionConfig:
            raise TypeError("config must be exact HCCLCausalAttributionConfig")
        self._config = config
        self._owner = jnp.asarray(config.proposal_owner_digest, dtype=jnp.uint32)
        self._max_words = jnp.asarray(
            (config.max_transactions >> 32, config.max_transactions & _UINT32_MAX),
            dtype=jnp.uint32,
        )

    @property
    def config(self) -> HCCLCausalAttributionConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> HCCLCausalAttributionKernel:
        return cls(HCCLCausalAttributionConfig.from_config(payload))

    def init(self) -> HCCLCausalAttributionState:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return HCCLCausalAttributionState(
            transaction_words=zero_words,
            decision_words=zero_words,
            last_committed_pp=_empty_proposal(self._config),
            last_contrasts=_empty_contrasts(),
            last_attribution_tag_words=jnp.zeros((_TAG_WORDS,), dtype=jnp.uint32),
        )

    def _require_signals_contract(self, signals: HCCLTypedSignals, *, label: str) -> None:
        if type(signals) is not HCCLTypedSignals:
            raise TypeError(f"{label} must be exact HCCLTypedSignals")
        _float_scalar(signals.task_score, label=f"{label}.task_score")
        _float_vector(signals.net_reward, _N_AGENTS, label=f"{label}.net_reward")
        _float_vector(signals.safety_cost, _N_AGENTS, label=f"{label}.safety_cost")
        _float_vector(signals.message_charge, _N_AGENTS, label=f"{label}.message_charge")

    def _require_vertex_contract(self, vertex: HCCLJointActionVertex, *, label: str) -> None:
        if type(vertex) is not HCCLJointActionVertex:
            raise TypeError(f"{label} must be exact HCCLJointActionVertex")
        _require_array(
            vertex.actions,
            shape=(_N_AGENTS,),
            dtype=jnp.dtype(jnp.int32),
            label=f"{label}.actions",
        )
        _words(
            vertex.action_receipt_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label=f"{label}.action_receipt_identity_words",
        )
        _require_array(
            vertex.layer_codes,
            shape=(_N_AGENTS,),
            dtype=jnp.dtype(jnp.int32),
            label=f"{label}.layer_codes",
        )

    def _require_source_contract(self, source: HCCLCausalSourceReceipt) -> None:
        if type(source) is not HCCLCausalSourceReceipt:
            raise TypeError("source must be exact HCCLCausalSourceReceipt")
        _float_vector(source.source_vector, self._config.source_dim, label="source_vector")
        _words(source.source_identity_words, (_IDENTITY_WORDS,), label="source_identity_words")
        _words(
            source.agent_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="agent_identity_words",
        )
        _words(source.decision_words, (2,), label="source.decision_words")
        _words(
            source.source_transition_words,
            (2,),
            label="source.source_transition_words",
        )
        _words(
            source.raw_observation_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="raw_observation_identity_words",
        )
        _words(source.fast_state_words, (_N_AGENTS, 2), label="fast_state_words")
        _words(
            source.slow_context_birth_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="slow_context_birth_words",
        )
        _words(
            source.feature_birth_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="feature_birth_words",
        )
        _words(
            source.memory_generation_words,
            (_N_AGENTS, 2),
            label="memory_generation_words",
        )
        _words(
            source.planner_model_words,
            (_N_AGENTS, 2),
            label="planner_model_words",
        )
        _words(
            source.hard_mask_generation_words,
            (_N_AGENTS, 2),
            label="hard_mask_generation_words",
        )
        _words(
            source.rng_receipt_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="rng_receipt_identity_words",
        )
        _words(source.content_tag_words, (_TAG_WORDS,), label="source.content_tag_words")

    def _require_exogenous_contract(self, receipt: HCCLExogenousReceipt) -> None:
        if type(receipt) is not HCCLExogenousReceipt:
            raise TypeError("exogenous must be exact HCCLExogenousReceipt")
        _words(receipt.source_identity_words, (_IDENTITY_WORDS,), label="exogenous.source")
        _words(receipt.decision_words, (2,), label="exogenous.decision_words")
        _words(receipt.source_transition_words, (2,), label="exogenous.transition")
        _words(
            receipt.exogenous_identity_words,
            (_IDENTITY_WORDS,),
            label="exogenous.identity",
        )
        _words(receipt.exogenous_source_words, (2,), label="exogenous.source_words")
        _float_vector(receipt.payload, self._config.exogenous_dim, label="exogenous.payload")
        _words(receipt.content_tag_words, (_TAG_WORDS,), label="exogenous.content_tag")

    def _require_action_contract(self, receipt: HCCLActionReceipt) -> None:
        if type(receipt) is not HCCLActionReceipt:
            raise TypeError("action receipt must be exact HCCLActionReceipt")
        self._require_source_contract(receipt.source)
        _words(receipt.exogenous_identity_words, (_IDENTITY_WORDS,), label="action.exogenous")
        _words(receipt.exogenous_content_tag_words, (_TAG_WORDS,), label="action.exogenous_tag")
        _int_scalar(receipt.layer, label="action.layer")
        for label, value in (
            ("actions_before_mask", receipt.actions_before_mask),
            ("actions_after_mask", receipt.actions_after_mask),
        ):
            _require_array(
                value,
                shape=(_N_AGENTS,),
                dtype=jnp.dtype(jnp.int32),
                label=label,
            )
        _require_array(
            receipt.hard_action_masks,
            shape=(_N_AGENTS, self._config.n_actions),
            dtype=jnp.dtype(jnp.bool_),
            label="hard_action_masks",
        )
        _words(
            receipt.action_receipt_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="action_receipt_identity_words",
        )
        _words(receipt.content_tag_words, (_TAG_WORDS,), label="action.content_tag")

    def _require_proposal_contract(self, proposal: HCCLTransitionProposal) -> None:
        if type(proposal) is not HCCLTransitionProposal:
            raise TypeError("callback must return exact HCCLTransitionProposal")
        for label, value in (
            ("proposal.source_identity", proposal.source_identity_words),
            ("proposal.source_tag", proposal.source_content_tag_words),
            ("proposal.exogenous_identity", proposal.exogenous_identity_words),
            ("proposal.exogenous_tag", proposal.exogenous_content_tag_words),
            ("proposal.content_tag", proposal.content_tag_words),
        ):
            _words(value, (_IDENTITY_WORDS,), label=label)
        for label, value in (
            ("proposal.source_decision", proposal.source_decision_words),
            ("proposal.source_transition", proposal.source_transition_words),
            ("proposal.destination_transition", proposal.destination_transition_words),
        ):
            _words(value, (2,), label=label)
        self._require_vertex_contract(proposal.vertex, label="proposal.vertex")
        _float_vector(
            proposal.candidate_transition,
            self._config.transition_dim,
            label="proposal.candidate_transition",
        )
        self._require_signals_contract(proposal.signals, label="proposal.signals")
        _bool_scalar(proposal.accepted, label="proposal.accepted")

    def _require_contrasts_contract(self, contrasts: HCCLCausalContrasts) -> None:
        if type(contrasts) is not HCCLCausalContrasts:
            raise TypeError("contrasts must be exact HCCLCausalContrasts")
        for name in (
            "memory_total",
            "memory_interaction",
            "planner_total",
            "planner_interaction",
            "pp_minus_bb",
            "telescoping_sum",
            "telescoping_residual",
        ):
            value = getattr(contrasts, name)
            if type(value) is not HCCLSignalContrast:
                raise TypeError(f"contrasts.{name} must be exact HCCLSignalContrast")
            _float_scalar(value.task_score, label=f"contrasts.{name}.task_score")
            _float_vector(value.net_reward, _N_AGENTS, label=f"contrasts.{name}.net")
            _float_vector(value.safety_cost, _N_AGENTS, label=f"contrasts.{name}.safety")
            _float_vector(value.message_charge, _N_AGENTS, label=f"contrasts.{name}.message")

    def _require_state_contract(self, state: HCCLCausalAttributionState) -> None:
        if type(state) is not HCCLCausalAttributionState:
            raise TypeError("state must be exact HCCLCausalAttributionState")
        _words(state.transaction_words, (2,), label="state.transaction_words")
        _words(state.decision_words, (2,), label="state.decision_words")
        self._require_proposal_contract(state.last_committed_pp)
        self._require_contrasts_contract(state.last_contrasts)
        _words(
            state.last_attribution_tag_words,
            (_TAG_WORDS,),
            label="state.last_attribution_tag_words",
        )

    def _signals_valid(self, signals: HCCLTypedSignals) -> Bool[Array, ""]:
        bound = jnp.float32(self._config.max_abs_signal)
        expected_net = signals.task_score - signals.message_charge - signals.safety_cost
        return cast(
            Bool[Array, ""],
            _signals_finite(signals)
            & (jnp.abs(signals.task_score) <= bound)
            & jnp.all(jnp.abs(signals.net_reward) <= bound)
            & jnp.all(signals.safety_cost >= 0.0)
            & jnp.all(signals.safety_cost <= bound)
            & jnp.all(signals.message_charge >= 0.0)
            & jnp.all(signals.message_charge <= bound)
            & _float_bits_equal(signals.net_reward, expected_net),
        )

    def _proposal_self_valid(self, proposal: HCCLTransitionProposal) -> Bool[Array, ""]:
        expected_tag = self._proposal_tag(proposal)
        return (
            proposal.accepted
            & jnp.any(proposal.source_identity_words != 0)
            & jnp.any(proposal.exogenous_identity_words != 0)
            & jnp.all(proposal.vertex.actions >= 0)
            & jnp.all(proposal.vertex.actions < self._config.n_actions)
            & jnp.all(
                (proposal.vertex.layer_codes >= int(HCCLActionLayer.BASE))
                & (proposal.vertex.layer_codes <= int(HCCLActionLayer.PLANNER))
            )
            & jnp.all(
                jnp.any(
                    proposal.vertex.action_receipt_identity_words != 0,
                    axis=1,
                )
            )
            & (~jnp.all(
                proposal.vertex.action_receipt_identity_words[0]
                == proposal.vertex.action_receipt_identity_words[1]
            ))
            & jnp.all(jnp.isfinite(proposal.candidate_transition))
            & jnp.all(
                jnp.abs(proposal.candidate_transition)
                <= jnp.float32(self._config.max_abs_transition)
            )
            & self._signals_valid(proposal.signals)
            & jnp.all(proposal.content_tag_words == expected_tag)
        )

    def _dynamic_state_valid(self, state: HCCLCausalAttributionState) -> Bool[Array, ""]:
        empty = self.init()
        predecessor_successor, predecessor_capacity = _increment_words(
            state.last_committed_pp.source_transition_words
        )
        message_contrasts_zero = jnp.asarray(True, dtype=jnp.bool_)
        for name in (
            "memory_total",
            "memory_interaction",
            "planner_total",
            "planner_interaction",
            "pp_minus_bb",
            "telescoping_sum",
            "telescoping_residual",
        ):
            contrast = getattr(state.last_contrasts, name)
            message_contrasts_zero = message_contrasts_zero & jnp.all(
                contrast.message_charge == 0.0
            )
        pristine = (
            jnp.all(state.transaction_words == 0)
            & _tree_exact_equal(state.last_committed_pp, empty.last_committed_pp)
            & _tree_exact_equal(state.last_contrasts, empty.last_contrasts)
            & jnp.all(state.last_attribution_tag_words == 0)
        )
        committed = (
            jnp.any(state.transaction_words != 0)
            & self._proposal_self_valid(state.last_committed_pp)
            & jnp.all(
                state.last_committed_pp.destination_transition_words == state.transaction_words
            )
            & predecessor_capacity
            & jnp.all(predecessor_successor == state.transaction_words)
            & jnp.all(
                state.last_committed_pp.source_decision_words
                == state.last_committed_pp.source_transition_words
            )
            & jnp.all(
                state.last_committed_pp.vertex.layer_codes
                == int(HCCLActionLayer.PLANNER)
            )
            & _telescoping_roundoff_valid(state.last_contrasts)
            & message_contrasts_zero
            & jnp.all(
                state.last_attribution_tag_words
                == _attribution_tag(
                    self._owner,
                    state.transaction_words,
                    state.last_committed_pp,
                    state.last_contrasts,
                )
            )
        )
        return (
            jnp.all(state.decision_words == state.transaction_words)
            & _words_not_later(state.transaction_words, self._max_words)
            & (pristine | committed)
        )

    def state_valid(self, state: HCCLCausalAttributionState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def bind_source(
        self,
        state: HCCLCausalAttributionState,
        *,
        source_vector: Array,
        source_identity_words: Array,
        agent_identity_words: Array,
        raw_observation_identity_words: Array,
        fast_state_words: Array,
        slow_context_birth_words: Array,
        feature_birth_words: Array,
        memory_generation_words: Array,
        planner_model_words: Array,
        hard_mask_generation_words: Array,
        rng_receipt_identity_words: Array,
    ) -> HCCLCausalSourceReceipt:
        self._require_state_contract(state)
        source_vector = _float_vector(
            source_vector,
            self._config.source_dim,
            label="source_vector",
        )
        source_identity_words = _words(
            source_identity_words,
            (_IDENTITY_WORDS,),
            label="source_identity_words",
        )
        agent_identity_words = _words(
            agent_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="agent_identity_words",
        )
        raw_observation_identity_words = _words(
            raw_observation_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="raw_observation_identity_words",
        )
        fast_state_words = _words(
            fast_state_words,
            (_N_AGENTS, 2),
            label="fast_state_words",
        )
        slow_context_birth_words = _words(
            slow_context_birth_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="slow_context_birth_words",
        )
        feature_birth_words = _words(
            feature_birth_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="feature_birth_words",
        )
        memory_generation_words = _words(
            memory_generation_words,
            (_N_AGENTS, 2),
            label="memory_generation_words",
        )
        planner_model_words = _words(
            planner_model_words,
            (_N_AGENTS, 2),
            label="planner_model_words",
        )
        hard_mask_generation_words = _words(
            hard_mask_generation_words,
            (_N_AGENTS, 2),
            label="hard_mask_generation_words",
        )
        rng_receipt_identity_words = _words(
            rng_receipt_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="rng_receipt_identity_words",
        )
        tag = _content_tag(
            self._owner,
            source_vector,
            source_identity_words,
            agent_identity_words,
            state.decision_words,
            state.transaction_words,
            raw_observation_identity_words,
            fast_state_words,
            slow_context_birth_words,
            feature_birth_words,
            memory_generation_words,
            planner_model_words,
            hard_mask_generation_words,
            rng_receipt_identity_words,
        )
        return HCCLCausalSourceReceipt(
            source_vector=source_vector,
            source_identity_words=source_identity_words,
            agent_identity_words=agent_identity_words,
            decision_words=state.decision_words,
            source_transition_words=state.transaction_words,
            raw_observation_identity_words=raw_observation_identity_words,
            fast_state_words=fast_state_words,
            slow_context_birth_words=slow_context_birth_words,
            feature_birth_words=feature_birth_words,
            memory_generation_words=memory_generation_words,
            planner_model_words=planner_model_words,
            hard_mask_generation_words=hard_mask_generation_words,
            rng_receipt_identity_words=rng_receipt_identity_words,
            content_tag_words=tag,
        )

    def _source_tag(self, source: HCCLCausalSourceReceipt) -> Array:
        return _content_tag(
            self._owner,
            source.source_vector,
            source.source_identity_words,
            source.agent_identity_words,
            source.decision_words,
            source.source_transition_words,
            source.raw_observation_identity_words,
            source.fast_state_words,
            source.slow_context_birth_words,
            source.feature_birth_words,
            source.memory_generation_words,
            source.planner_model_words,
            source.hard_mask_generation_words,
            source.rng_receipt_identity_words,
        )

    def bind_exogenous(
        self,
        source: HCCLCausalSourceReceipt,
        *,
        exogenous_identity_words: Array,
        exogenous_source_words: Array,
        payload: Array,
    ) -> HCCLExogenousReceipt:
        self._require_source_contract(source)
        exogenous_identity_words = _words(
            exogenous_identity_words,
            (_IDENTITY_WORDS,),
            label="exogenous_identity_words",
        )
        exogenous_source_words = _words(
            exogenous_source_words,
            (2,),
            label="exogenous_source_words",
        )
        payload = _float_vector(payload, self._config.exogenous_dim, label="payload")
        tag = _content_tag(
            self._owner,
            source.source_identity_words,
            source.content_tag_words,
            source.decision_words,
            source.source_transition_words,
            exogenous_identity_words,
            exogenous_source_words,
            payload,
        )
        return HCCLExogenousReceipt(
            source_identity_words=source.source_identity_words,
            decision_words=source.decision_words,
            source_transition_words=source.source_transition_words,
            exogenous_identity_words=exogenous_identity_words,
            exogenous_source_words=exogenous_source_words,
            payload=payload,
            content_tag_words=tag,
        )

    def _exogenous_tag(
        self,
        source: HCCLCausalSourceReceipt,
        receipt: HCCLExogenousReceipt,
    ) -> Array:
        return _content_tag(
            self._owner,
            receipt.source_identity_words,
            source.content_tag_words,
            receipt.decision_words,
            receipt.source_transition_words,
            receipt.exogenous_identity_words,
            receipt.exogenous_source_words,
            receipt.payload,
        )

    def bind_action_receipt(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        *,
        layer: HCCLActionLayer,
        actions_before_mask: Array,
        actions_after_mask: Array,
        hard_action_masks: Array,
        action_receipt_identity_words: Array,
    ) -> HCCLActionReceipt:
        self._require_source_contract(source)
        self._require_exogenous_contract(exogenous)
        if type(layer) is not HCCLActionLayer:
            raise TypeError("layer must be exact HCCLActionLayer")
        actions_before_mask = _require_array(
            actions_before_mask,
            shape=(_N_AGENTS,),
            dtype=jnp.dtype(jnp.int32),
            label="actions_before_mask",
        )
        actions_after_mask = _require_array(
            actions_after_mask,
            shape=(_N_AGENTS,),
            dtype=jnp.dtype(jnp.int32),
            label="actions_after_mask",
        )
        hard_action_masks = _require_array(
            hard_action_masks,
            shape=(_N_AGENTS, self._config.n_actions),
            dtype=jnp.dtype(jnp.bool_),
            label="hard_action_masks",
        )
        action_receipt_identity_words = _words(
            action_receipt_identity_words,
            (_N_AGENTS, _IDENTITY_WORDS),
            label="action_receipt_identity_words",
        )
        layer_array = jnp.asarray(int(layer), dtype=jnp.int32)
        tag = _content_tag(
            self._owner,
            source.content_tag_words,
            exogenous.exogenous_identity_words,
            exogenous.content_tag_words,
            layer_array,
            actions_before_mask,
            actions_after_mask,
            hard_action_masks,
            action_receipt_identity_words,
        )
        return HCCLActionReceipt(
            source=source,
            exogenous_identity_words=exogenous.exogenous_identity_words,
            exogenous_content_tag_words=exogenous.content_tag_words,
            layer=layer_array,
            actions_before_mask=actions_before_mask,
            actions_after_mask=actions_after_mask,
            hard_action_masks=hard_action_masks,
            action_receipt_identity_words=action_receipt_identity_words,
            content_tag_words=tag,
        )

    def _action_tag(self, receipt: HCCLActionReceipt) -> Array:
        return _content_tag(
            self._owner,
            receipt.source.content_tag_words,
            receipt.exogenous_identity_words,
            receipt.exogenous_content_tag_words,
            receipt.layer,
            receipt.actions_before_mask,
            receipt.actions_after_mask,
            receipt.hard_action_masks,
            receipt.action_receipt_identity_words,
        )

    def bind_proposal(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        *,
        candidate_transition: Array,
        signals: HCCLTypedSignals,
        accepted: Array,
    ) -> HCCLTransitionProposal:
        self._require_source_contract(source)
        self._require_exogenous_contract(exogenous)
        self._require_vertex_contract(vertex, label="vertex")
        candidate_transition = _float_vector(
            candidate_transition,
            self._config.transition_dim,
            label="candidate_transition",
        )
        self._require_signals_contract(signals, label="signals")
        accepted = _bool_scalar(accepted, label="accepted")
        destination, _ = _increment_words(source.source_transition_words)
        proposal = HCCLTransitionProposal(
            source_identity_words=source.source_identity_words,
            source_content_tag_words=source.content_tag_words,
            exogenous_identity_words=exogenous.exogenous_identity_words,
            exogenous_content_tag_words=exogenous.content_tag_words,
            source_decision_words=source.decision_words,
            source_transition_words=source.source_transition_words,
            destination_transition_words=destination,
            vertex=vertex,
            candidate_transition=candidate_transition,
            signals=signals,
            accepted=accepted,
            content_tag_words=jnp.zeros((_TAG_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTransitionProposal,
            cast(Any, proposal).replace(content_tag_words=self._proposal_tag(proposal)),
        )

    def _proposal_tag(self, proposal: HCCLTransitionProposal) -> Array:
        signals = proposal.signals
        return _content_tag(
            self._owner,
            proposal.source_identity_words,
            proposal.source_content_tag_words,
            proposal.exogenous_identity_words,
            proposal.exogenous_content_tag_words,
            proposal.source_decision_words,
            proposal.source_transition_words,
            proposal.destination_transition_words,
            proposal.vertex.actions,
            proposal.vertex.action_receipt_identity_words,
            proposal.vertex.layer_codes,
            proposal.candidate_transition,
            signals.task_score,
            signals.net_reward,
            signals.safety_cost,
            signals.message_charge,
            proposal.accepted,
        )

    def _source_valid(
        self,
        state: HCCLCausalAttributionState,
        source: HCCLCausalSourceReceipt,
    ) -> Bool[Array, ""]:
        return (
            jnp.all(source.decision_words == state.decision_words)
            & jnp.all(source.source_transition_words == state.transaction_words)
            & jnp.all(jnp.isfinite(source.source_vector))
            & jnp.all(jnp.abs(source.source_vector) <= jnp.float32(self._config.max_abs_source))
            & jnp.any(source.source_identity_words != 0)
            & jnp.all(jnp.any(source.agent_identity_words != 0, axis=1))
            & (~jnp.all(source.agent_identity_words[0] == source.agent_identity_words[1]))
            & jnp.all(jnp.any(source.raw_observation_identity_words != 0, axis=1))
            & jnp.all(jnp.any(source.slow_context_birth_words != 0, axis=1))
            & jnp.all(jnp.any(source.feature_birth_words != 0, axis=1))
            & jnp.all(jnp.any(source.rng_receipt_identity_words != 0, axis=1))
            & jnp.all(source.content_tag_words == self._source_tag(source))
        )

    def _exogenous_valid(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
    ) -> Bool[Array, ""]:
        return (
            jnp.all(exogenous.source_identity_words == source.source_identity_words)
            & jnp.all(exogenous.decision_words == source.decision_words)
            & jnp.all(exogenous.source_transition_words == source.source_transition_words)
            & jnp.any(exogenous.exogenous_identity_words != 0)
            & jnp.any(exogenous.exogenous_source_words != 0)
            & jnp.all(jnp.isfinite(exogenous.payload))
            & jnp.all(jnp.abs(exogenous.payload) <= jnp.float32(self._config.max_abs_exogenous))
            & jnp.all(exogenous.content_tag_words == self._exogenous_tag(source, exogenous))
        )

    def _action_valid(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        receipt: HCCLActionReceipt,
        expected_layer: HCCLActionLayer,
    ) -> Bool[Array, ""]:
        safe_actions = jnp.clip(
            receipt.actions_after_mask,
            0,
            self._config.n_actions - 1,
        )
        selected_safe = receipt.hard_action_masks[
            jnp.arange(_N_AGENTS),
            safe_actions,
        ]
        before_in_range = (receipt.actions_before_mask >= 0) & (
            receipt.actions_before_mask < self._config.n_actions
        )
        after_in_range = (receipt.actions_after_mask >= 0) & (
            receipt.actions_after_mask < self._config.n_actions
        )
        before_safe = receipt.hard_action_masks[
            jnp.arange(_N_AGENTS),
            jnp.clip(receipt.actions_before_mask, 0, self._config.n_actions - 1),
        ]
        replacement_semantics = (~before_safe) | (
            receipt.actions_after_mask == receipt.actions_before_mask
        )
        return (
            _tree_exact_equal(receipt.source, source)
            & jnp.all(receipt.exogenous_identity_words == exogenous.exogenous_identity_words)
            & jnp.all(receipt.exogenous_content_tag_words == exogenous.content_tag_words)
            & (receipt.layer == int(expected_layer))
            & jnp.all(before_in_range)
            & jnp.all(after_in_range)
            & jnp.all(selected_safe)
            & jnp.all(replacement_semantics)
            & jnp.all(jnp.any(receipt.hard_action_masks, axis=1))
            & jnp.all(jnp.any(receipt.action_receipt_identity_words != 0, axis=1))
            & jnp.all(receipt.content_tag_words == self._action_tag(receipt))
        )

    def _receipt_identities_distinct(
        self,
        receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    ) -> Bool[Array, ""]:
        identities = jnp.stack(tuple(receipt.action_receipt_identity_words for receipt in receipts))
        flattened = jnp.reshape(identities, (3 * _N_AGENTS, _IDENTITY_WORDS))
        distinct = jnp.asarray(True, dtype=jnp.bool_)
        for left in range(3 * _N_AGENTS):
            for right in range(left):
                distinct = distinct & (~jnp.all(flattened[left] == flattened[right]))
        return distinct

    def _vertices(
        self,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        planner: HCCLActionReceipt,
    ) -> tuple[HCCLJointActionVertex, ...]:
        def vertex(left: HCCLActionReceipt, right: HCCLActionReceipt) -> HCCLJointActionVertex:
            return HCCLJointActionVertex(
                actions=jnp.stack((left.actions_after_mask[0], right.actions_after_mask[1])).astype(
                    jnp.int32
                ),
                action_receipt_identity_words=jnp.stack(
                    (
                        left.action_receipt_identity_words[0],
                        right.action_receipt_identity_words[1],
                    )
                ).astype(jnp.uint32),
                layer_codes=jnp.stack((left.layer, right.layer)).astype(jnp.int32),
            )

        mm = vertex(memory, memory)
        return (
            mm,
            vertex(base, memory),
            vertex(memory, base),
            vertex(base, base),
            vertex(planner, planner),
            vertex(memory, planner),
            vertex(planner, memory),
            mm,
        )

    def _proposal_valid(
        self,
        proposal: HCCLTransitionProposal,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        expected_vertex: HCCLJointActionVertex,
        destination_words: Array,
    ) -> Bool[Array, ""]:
        return (
            self._proposal_self_valid(proposal)
            & jnp.all(proposal.source_identity_words == source.source_identity_words)
            & jnp.all(proposal.source_content_tag_words == source.content_tag_words)
            & jnp.all(proposal.exogenous_identity_words == exogenous.exogenous_identity_words)
            & jnp.all(proposal.exogenous_content_tag_words == exogenous.content_tag_words)
            & jnp.all(proposal.source_decision_words == source.decision_words)
            & jnp.all(proposal.source_transition_words == source.source_transition_words)
            & jnp.all(proposal.destination_transition_words == destination_words)
            & _tree_exact_equal(proposal.vertex, expected_vertex)
        )

    def _count_unique_vertices(
        self,
        vertices: tuple[HCCLJointActionVertex, ...],
        *,
        include_receipt_identity: bool,
    ) -> Int[Array, ""]:
        count = jnp.asarray(0, dtype=jnp.int32)
        for index in range(7):
            is_new = jnp.asarray(True, dtype=jnp.bool_)
            for previous in range(index):
                equal = jnp.all(vertices[index].actions == vertices[previous].actions)
                if include_receipt_identity:
                    equal = (
                        equal
                        & jnp.all(
                            vertices[index].action_receipt_identity_words
                            == vertices[previous].action_receipt_identity_words
                        )
                        & jnp.all(vertices[index].layer_codes == vertices[previous].layer_codes)
                    )
                is_new = is_new & (~equal)
            count = count + is_new.astype(jnp.int32)
        return count

    def _empty_result(
        self,
        state: HCCLCausalAttributionState,
        *,
        source_state_valid: Array,
        source_receipt_valid: Array,
        exogenous_receipt_valid: Array,
        action_receipts_valid: Array,
        identities_distinct: Array,
        capacity: Array,
        preflight_valid: Array,
    ) -> HCCLCausalAttributionResult:
        empty = _empty_proposal(self._config)
        proposals = _stack_proposals(tuple(empty for _ in range(_N_PROPOSALS)))
        zero = jnp.asarray(0, dtype=jnp.int32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return HCCLCausalAttributionResult(
            state=state,
            proposals=proposals,
            contrasts=_empty_contrasts(),
            work=HCCLCausalAttributionWork(
                proposal_calls=zero,
                designated_counterfactual_slots=zero,
                discarded_proposal_calls=zero,
                committed_pp_calls=zero,
                duplicate_mm_equality_checks=zero,
            ),
            pre_transaction_words=state.transaction_words,
            post_transaction_words=state.transaction_words,
            committed_slot=jnp.asarray(-1, dtype=jnp.int32),
            unique_joint_action_vertices=zero,
            unique_joint_action_receipt_vertices=zero,
            host_preflight_performed=jnp.asarray(True, dtype=jnp.bool_),
            source_state_valid=source_state_valid,
            source_receipt_valid=source_receipt_valid,
            exogenous_receipt_valid=exogenous_receipt_valid,
            action_receipts_valid=action_receipts_valid,
            receipt_identities_distinct=identities_distinct,
            lifetime_capacity_available=capacity,
            preflight_valid=preflight_valid,
            all_child_proposals_valid=false,
            duplicate_mm_bit_exact=false,
            typed_signals_valid=false,
            telescoping_valid=false,
            downstream_candidate_valid=false,
            candidate_state_valid=false,
            update_applied=false,
        )

    def stage(
        self,
        state: HCCLCausalAttributionState,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        planner: HCCLActionReceipt,
        proposal_callback: HCCLProposalCallback,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLCausalAttributionResult:
        """Stage both cubes and atomically persist only PP when every gate accepts."""

        self._require_state_contract(state)
        self._require_source_contract(source)
        self._require_exogenous_contract(exogenous)
        for receipt in (base, memory, planner):
            self._require_action_contract(receipt)
        if not callable(proposal_callback):
            raise TypeError("proposal_callback must be callable")
        downstream_candidate_valid = _bool_scalar(
            downstream_candidate_valid,
            label="downstream_candidate_valid",
        )
        source_state_valid = self._dynamic_state_valid(state)
        source_valid = self._source_valid(state, source)
        exogenous_valid = self._exogenous_valid(source, exogenous)
        action_valid = (
            self._action_valid(source, exogenous, base, HCCLActionLayer.BASE)
            & self._action_valid(source, exogenous, memory, HCCLActionLayer.MEMORY)
            & self._action_valid(source, exogenous, planner, HCCLActionLayer.PLANNER)
            & jnp.array_equal(base.hard_action_masks, memory.hard_action_masks)
            & jnp.array_equal(base.hard_action_masks, planner.hard_action_masks)
        )
        identities_distinct = self._receipt_identities_distinct((base, memory, planner))
        destination_words, uint64_capacity = _increment_words(state.transaction_words)
        configured_capacity = ~jnp.all(state.transaction_words == self._max_words)
        capacity = uint64_capacity & configured_capacity
        preflight = (
            source_state_valid
            & source_valid
            & exogenous_valid
            & action_valid
            & identities_distinct
            & capacity
        )
        traced = _contains_tracer(
            (
                state,
                source,
                exogenous,
                base,
                memory,
                planner,
                downstream_candidate_valid,
            )
        )
        if not traced and not bool(preflight):
            return self._empty_result(
                state,
                source_state_valid=source_state_valid,
                source_receipt_valid=source_valid,
                exogenous_receipt_valid=exogenous_valid,
                action_receipts_valid=action_valid,
                identities_distinct=identities_distinct,
                capacity=capacity,
                preflight_valid=preflight,
            )

        vertices = self._vertices(base, memory, planner)
        proposal_list: list[HCCLTransitionProposal] = []
        proposal_validity: list[Array] = []
        for slot, vertex in enumerate(vertices):
            proposal = proposal_callback(
                source,
                exogenous,
                vertex,
                jnp.asarray(slot, dtype=jnp.int32),
            )
            self._require_proposal_contract(proposal)
            proposal_list.append(proposal)
            proposal_validity.append(
                self._proposal_valid(
                    proposal,
                    source,
                    exogenous,
                    vertex,
                    destination_words,
                )
            )
        proposals = _stack_proposals(tuple(proposal_list))
        all_proposals_valid = jnp.all(jnp.stack(tuple(proposal_validity)))
        duplicate_mm = _tree_exact_equal(proposal_list[0], proposal_list[7])
        typed_signals_valid = jnp.all(
            jnp.stack(tuple(self._signals_valid(item.signals) for item in proposal_list))
        ) & _fixed_message_delivery_valid(tuple(proposal_list))
        contrasts = _derive_contrasts(proposals)
        telescoping_valid = _telescoping_roundoff_valid(contrasts)
        pp = proposal_list[_PP_SLOT]
        attribution_tag = _attribution_tag(
            self._owner,
            destination_words,
            pp,
            contrasts,
        )
        candidate = HCCLCausalAttributionState(
            transaction_words=destination_words,
            decision_words=destination_words,
            last_committed_pp=pp,
            last_contrasts=contrasts,
            last_attribution_tag_words=attribution_tag,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            preflight
            & all_proposals_valid
            & duplicate_mm
            & typed_signals_valid
            & telescoping_valid
            & downstream_candidate_valid
            & candidate_valid
        )
        final_state = cast(
            HCCLCausalAttributionState,
            _tree_select(applied, candidate, state),
        )
        return HCCLCausalAttributionResult(
            state=final_state,
            proposals=proposals,
            contrasts=contrasts,
            work=HCCLCausalAttributionWork(
                proposal_calls=jnp.asarray(_N_PROPOSALS, dtype=jnp.int32),
                designated_counterfactual_slots=jnp.asarray(
                    _N_NONCOMMITTING,
                    dtype=jnp.int32,
                ),
                discarded_proposal_calls=jnp.asarray(_N_PROPOSALS, dtype=jnp.int32)
                - applied.astype(jnp.int32),
                committed_pp_calls=applied.astype(jnp.int32),
                duplicate_mm_equality_checks=jnp.asarray(1, dtype=jnp.int32),
            ),
            pre_transaction_words=state.transaction_words,
            post_transaction_words=final_state.transaction_words,
            committed_slot=jnp.where(
                applied,
                jnp.asarray(_PP_SLOT, dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            unique_joint_action_vertices=self._count_unique_vertices(
                vertices,
                include_receipt_identity=False,
            ),
            unique_joint_action_receipt_vertices=self._count_unique_vertices(
                vertices,
                include_receipt_identity=True,
            ),
            host_preflight_performed=jnp.asarray(not traced, dtype=jnp.bool_),
            source_state_valid=source_state_valid,
            source_receipt_valid=source_valid,
            exogenous_receipt_valid=exogenous_valid,
            action_receipts_valid=action_valid,
            receipt_identities_distinct=identities_distinct,
            lifetime_capacity_available=capacity,
            preflight_valid=preflight,
            all_child_proposals_valid=all_proposals_valid,
            duplicate_mm_bit_exact=duplicate_mm,
            typed_signals_valid=typed_signals_valid,
            telescoping_valid=telescoping_valid,
            downstream_candidate_valid=downstream_candidate_valid,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLCausalAttributionState | None = None,
    ) -> HCCLCausalAttributionResourceBudget:
        reference = self.init() if state is None else state
        self._require_state_contract(reference)
        clock_bytes = 16
        proposal_bytes = _state_nbytes(reference.last_committed_pp)
        contrast_bytes = _state_nbytes(reference.last_contrasts)
        tag_bytes = 16
        budget = HCCLCausalAttributionResourceBudget(
            schema=HCCL_CAUSAL_ATTRIBUTION_RESOURCE_SCHEMA,
            clock_nbytes=clock_bytes,
            last_committed_proposal_nbytes=proposal_bytes,
            last_contrasts_nbytes=contrast_bytes,
            last_attribution_tag_nbytes=tag_bytes,
            total_state_nbytes=clock_bytes + proposal_bytes + contrast_bytes + tag_bytes,
            max_proposal_calls_per_transaction=_N_PROPOSALS,
            designated_counterfactual_slots_per_transaction=_N_NONCOMMITTING,
            max_discarded_proposal_calls_per_transaction=_N_PROPOSALS,
            max_committed_calls_per_transaction=1,
            max_unique_effective_joint_actions=min(
                self._config.n_actions**_N_AGENTS,
                _N_NONCOMMITTING,
            ),
            max_unique_joint_action_receipt_vertices=_N_NONCOMMITTING,
            max_transactions=self._config.max_transactions,
            persistent_bytes_scope=("kernel-clocks-last-PP-proposal-and-last-typed-contrasts-only"),
            transient_bytes_scope=(
                "source-exogenous-action-receipts-eight-proposals-and-callback-work; "
                "compiler-and-XLA-workspaces-excluded"
            ),
        )
        if measure_hccl_causal_attribution_state_nbytes(reference) != budget.total_state_nbytes:
            raise ValueError("HCCL attribution state allocation differs from resource declaration")
        return budget


def _leading_steps(tree: Any, *, label: str) -> int:
    leaves = jax.tree.leaves(tree)
    if not leaves or getattr(leaves[0], "ndim", 0) < 1:
        raise ValueError(f"{label} must have a leading step dimension")
    steps = int(leaves[0].shape[0])
    if steps < 1:
        raise ValueError(f"{label} must contain at least one row")
    for leaf in leaves:
        if getattr(leaf, "ndim", 0) < 1 or leaf.shape[0] != steps:
            raise ValueError(f"{label} leaves must share one leading step dimension")
    return steps


def run_hccl_causal_attribution_scan(
    kernel: HCCLCausalAttributionKernel,
    state: HCCLCausalAttributionState,
    sources: HCCLCausalSourceReceipt,
    exogenous: HCCLExogenousReceipt,
    base: HCCLActionReceipt,
    memory: HCCLActionReceipt,
    planner: HCCLActionReceipt,
    downstream_candidate_valid: Array,
    proposal_callback: HCCLProposalCallback,
) -> HCCLCausalAttributionScanResult:
    """Scan pre-bound rows; this does not create an HCCL environment or event draws."""

    if type(kernel) is not HCCLCausalAttributionKernel:
        raise TypeError("kernel must be exact HCCLCausalAttributionKernel")
    kernel._require_state_contract(state)
    steps = _leading_steps(sources, label="sources")
    for label, value in (
        ("exogenous", exogenous),
        ("base", base),
        ("memory", memory),
        ("planner", planner),
    ):
        if _leading_steps(value, label=label) != steps:
            raise ValueError(f"{label} step count differs")
    _require_array(
        downstream_candidate_valid,
        shape=(steps,),
        dtype=jnp.dtype(jnp.bool_),
        label="downstream_candidate_valid",
    )
    source_row = cast(
        HCCLCausalSourceReceipt,
        jax.tree.map(lambda leaf: leaf[0], sources),
    )
    exogenous_row = cast(HCCLExogenousReceipt, jax.tree.map(lambda leaf: leaf[0], exogenous))
    base_row = cast(HCCLActionReceipt, jax.tree.map(lambda leaf: leaf[0], base))
    memory_row = cast(HCCLActionReceipt, jax.tree.map(lambda leaf: leaf[0], memory))
    planner_row = cast(HCCLActionReceipt, jax.tree.map(lambda leaf: leaf[0], planner))
    kernel._require_source_contract(source_row)
    kernel._require_exogenous_contract(exogenous_row)
    for receipt in (base_row, memory_row, planner_row):
        kernel._require_action_contract(receipt)

    def body(
        carry: HCCLCausalAttributionState,
        row: tuple[
            HCCLCausalSourceReceipt,
            HCCLExogenousReceipt,
            HCCLActionReceipt,
            HCCLActionReceipt,
            HCCLActionReceipt,
            Array,
        ],
    ) -> tuple[HCCLCausalAttributionState, tuple[Array, ...]]:
        source_item, exogenous_item, base_item, memory_item, planner_item, gate = row
        result = kernel.stage(
            carry,
            source_item,
            exogenous_item,
            base_item,
            memory_item,
            planner_item,
            proposal_callback,
            downstream_candidate_valid=gate,
        )
        return result.state, (
            result.contrasts.memory_total.task_score,
            result.contrasts.planner_total.task_score,
            result.contrasts.pp_minus_bb.task_score,
            result.post_transaction_words,
            result.work.proposal_calls,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            sources,
            exogenous,
            base,
            memory,
            planner,
            downstream_candidate_valid,
        ),
    )
    memory_total, planner_total, stack_total, words, calls, applied = outputs
    return HCCLCausalAttributionScanResult(
        state=final_state,
        memory_total_task_score=memory_total,
        planner_total_task_score=planner_total,
        pp_minus_bb_task_score=stack_total,
        transaction_words=words,
        proposal_calls=calls,
        update_applied=applied,
    )


def save_hccl_causal_attribution_checkpoint(
    kernel: HCCLCausalAttributionKernel,
    state: HCCLCausalAttributionState,
    path: str | Path,
) -> None:
    """Persist mechanism state; metadata grants no HCCL execution authority."""

    if type(kernel) is not HCCLCausalAttributionKernel:
        raise TypeError("kernel must be exact HCCLCausalAttributionKernel")
    kernel._require_state_contract(state)
    if not bool(kernel.state_valid(state)):
        raise ValueError("cannot checkpoint invalid HCCL attribution state")
    config = kernel.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA,
            "evidence_level": HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL,
            "mechanism_status": HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS,
            "hccl_execution_authorized": False,
            "noise_and_seed_semantics_pinned": False,
            "artifact_or_claim_authority": False,
            "kernel_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": kernel.resource_budget(state).to_config(),
        },
    )


def load_hccl_causal_attribution_checkpoint(
    path: str | Path,
) -> tuple[HCCLCausalAttributionKernel, HCCLCausalAttributionState]:
    """Restore only canonical L0 mechanism state and authority metadata."""

    metadata = load_checkpoint_metadata(path)
    fields = _exact_manifest(
        metadata,
        {
            "schema",
            "evidence_level",
            "mechanism_status",
            "hccl_execution_authorized",
            "noise_and_seed_semantics_pinned",
            "artifact_or_claim_authority",
            "kernel_config",
            "config_sha256",
            "resource_budget",
        },
        label="HCCL causal attribution checkpoint",
    )
    fixed = {
        "schema": HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA,
        "evidence_level": HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL,
        "mechanism_status": HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS,
        "hccl_execution_authorized": False,
        "noise_and_seed_semantics_pinned": False,
        "artifact_or_claim_authority": False,
    }
    for name, expected in fixed.items():
        actual = fields[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"HCCL attribution checkpoint {name} is unsupported")
    config = fields["kernel_config"]
    if type(config) is not dict:
        raise TypeError("checkpoint kernel_config must be an exact dict")
    if type(fields["config_sha256"]) is not str:
        raise TypeError("checkpoint config_sha256 must be an exact str")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("checkpoint config digest differs")
    kernel = HCCLCausalAttributionKernel.from_config(config)
    if _canonical_json_bytes(kernel.to_config()) != _canonical_json_bytes(config):
        raise ValueError("checkpoint config is noncanonical")
    template = kernel.init()
    resource_budget = fields["resource_budget"]
    if type(resource_budget) is not dict:
        raise TypeError("checkpoint resource_budget must be an exact dict")
    if _canonical_json_bytes(resource_budget) != _canonical_json_bytes(
        kernel.resource_budget(template).to_config()
    ):
        raise ValueError("checkpoint resource budget differs")
    restored_raw, restored_metadata = load_checkpoint(template, path)
    if _canonical_json_bytes(restored_metadata) != _canonical_json_bytes(metadata):
        raise ValueError("checkpoint metadata changed between reads")
    restored = cast(HCCLCausalAttributionState, restored_raw)
    kernel._require_state_contract(restored)
    if not bool(kernel.state_valid(restored)):
        raise ValueError("restored HCCL attribution state is invalid")
    kernel.resource_budget(restored)
    return kernel, restored


__all__ = [
    "HCCLActionLayer",
    "HCCLActionReceipt",
    "HCCL_CAUSAL_ATTRIBUTION_ACTION_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_CHECKPOINT_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_CONFIG_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL",
    "HCCL_CAUSAL_ATTRIBUTION_EXOGENOUS_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_LIMITATIONS",
    "HCCL_CAUSAL_ATTRIBUTION_MECHANISM_STATUS",
    "HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER",
    "HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_RESOURCE_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_SOURCE_SCHEMA",
    "HCCL_CAUSAL_ATTRIBUTION_STATE_SCHEMA",
    "HCCLCausalAttributionConfig",
    "HCCLCausalAttributionKernel",
    "HCCLCausalAttributionResourceBudget",
    "HCCLCausalAttributionResult",
    "HCCLCausalAttributionScanResult",
    "HCCLCausalAttributionState",
    "HCCLCausalAttributionWork",
    "HCCLCausalContrasts",
    "HCCLCausalSourceReceipt",
    "HCCLExogenousReceipt",
    "HCCLJointActionVertex",
    "HCCLProposalCallback",
    "HCCLSignalContrast",
    "HCCLTransitionProposal",
    "HCCLTypedSignals",
    "load_hccl_causal_attribution_checkpoint",
    "measure_hccl_causal_attribution_state_nbytes",
    "run_hccl_causal_attribution_scan",
    "save_hccl_causal_attribution_checkpoint",
]
