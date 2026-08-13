# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Classified transient ownership trace for post-control STOMP mutations.

Prototype evaluates the real OaK→STOMP transition once, then some supported
compositions may update only narrowly classified parts of that same owner:
option-search learner backups, feature-axis routing, guarded Dyna backups, and
two ownership-correct primitive-dispatch replacements.  This module binds the
exact raw/final states plus narrow ordered witnesses for those stages without
rerunning any learner.  Digests and checksums are unkeyed integrity mechanisms,
never caller authentication or provenance credentials.
"""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.options import STOMPState

STOMP_OWNER_FINALIZATION_TRACE_SCHEMA = (
    "alberta.stomp-owner-finalization.trace.v1"
)
STOMP_OWNER_FINALIZATION_CALLER_AUTHENTICATED = False

STOMP_OWNER_STAGE_OPTION_SEARCH = 1
STOMP_OWNER_STAGE_FEATURE_ROUTE = 2
STOMP_OWNER_STAGE_DYNA = 3
STOMP_OWNER_STAGE_MEMORY_DISPATCH = 4
STOMP_OWNER_STAGE_PARTNER_DISPATCH = 5
STOMP_OWNER_STAGE_COUNT = 5

_TRACE_TAG = jnp.asarray((0x534F4631, 0x00000001), dtype=jnp.uint32)
_DIGEST_WORDS = 8


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    acc0 = jnp.uint32(0x9E3779B9)
    acc1 = jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _static_uint32_tag(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("ascii"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def stomp_typed_tree_digest(value: object) -> UInt[Array, " 8"]:
    """Return an array-only type/shape-tagged, unkeyed tree digest."""

    payload: list[Array] = []
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        array = jnp.asarray(leaf)
        dtype_text = str(array.dtype)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        elif array.dtype not in (jnp.float32, jnp.int32, jnp.uint32, jnp.bool_):
            raise TypeError(
                "STOMP finalization supports only float32, int32, uint32, "
                "bool, and typed PRNG-key leaves"
            )
        header = jnp.asarray(
            (
                index,
                _static_uint32_tag(dtype_text),
                array.ndim,
                array.size,
                *array.shape,
            ),
            dtype=jnp.uint32,
        )
        payload.extend((header, array))
    lanes = tuple(
        _checksum_arrays(
            (
                jnp.asarray((salt, lane), dtype=jnp.uint32),
                *payload,
            )
        )
        for lane, salt in enumerate(
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)
        )
    )
    return jnp.concatenate(lanes).astype(jnp.uint32)


def _tree_exact_equal(left: object, right: object) -> Bool[Array, ""]:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return jnp.asarray(False, dtype=jnp.bool_)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            valid = valid & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            valid = valid & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


@chex.dataclass(frozen=True)
class STOMPOwnerFinalizationStageReceipt:
    """Narrow digest/counter witness for one classified post-control stage."""

    stage_kind: Int[Array, ""]
    configured: Bool[Array, ""]
    evaluated: Bool[Array, ""]
    committed: Bool[Array, ""]
    classified_delta_valid: Bool[Array, ""]
    stomp_update_evaluations: Int[Array, ""]
    learner_updates_applied: Int[Array, ""]
    source_digest: UInt[Array, " 8"]
    destination_digest: UInt[Array, " 8"]
    source_step_count: Int[Array, ""]
    destination_step_count: Int[Array, ""]
    source_step_words: UInt[Array, " 2"]
    destination_step_words: UInt[Array, " 2"]
    source_executing_option: Int[Array, ""]
    destination_executing_option: Int[Array, ""]
    receipt_checksum: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class STOMPOwnerFinalizationTrace:
    """Ordered raw-to-final owner trace; transient and non-authenticating."""

    raw_state: STOMPState
    stages: tuple[
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
    ]
    final_state: STOMPState
    raw_digest: UInt[Array, " 8"]
    final_digest: UInt[Array, " 8"]
    real_control_stomp_evaluations: Int[Array, ""]
    imagined_stomp_evaluations: Int[Array, ""]
    option_search_learner_updates: Int[Array, ""]
    trace_checksum: UInt[Array, " 8"]
    caller_authenticated: Bool[Array, ""]


def make_stomp_owner_stage_receipt(
    source_state: STOMPState,
    destination_state: STOMPState,
    *,
    stage_kind: int,
    configured: Array,
    evaluated: Array,
    stomp_update_evaluations: Array,
    learner_updates_applied: Array,
    source_digest: Array | None = None,
    destination_digest: Array | None = None,
    classified_delta_valid: Array | None = None,
) -> STOMPOwnerFinalizationStageReceipt:
    """Bind one already-evaluated stage without retaining endpoint owners.

    ``source_digest``/``destination_digest`` and ``classified_delta_valid``
    let the trusted Prototype caller reuse already-computed narrow witnesses.
    When omitted they are recomputed from the supplied transient endpoints.
    The resulting receipt is unkeyed integrity, never authentication.
    """

    if type(source_state) is not STOMPState or type(destination_state) is not STOMPState:
        raise TypeError("stage endpoints must be exact STOMPState values")
    if stage_kind not in {
        STOMP_OWNER_STAGE_OPTION_SEARCH,
        STOMP_OWNER_STAGE_FEATURE_ROUTE,
        STOMP_OWNER_STAGE_DYNA,
        STOMP_OWNER_STAGE_MEMORY_DISPATCH,
        STOMP_OWNER_STAGE_PARTNER_DISPATCH,
    }:
        raise ValueError("stage_kind is unsupported")
    source_words = (
        stomp_typed_tree_digest(source_state)
        if source_digest is None
        else jnp.asarray(source_digest, dtype=jnp.uint32)
    )
    destination_words = (
        stomp_typed_tree_digest(destination_state)
        if destination_digest is None
        else jnp.asarray(destination_digest, dtype=jnp.uint32)
    )
    if source_words.shape != (_DIGEST_WORDS,) or destination_words.shape != (
        _DIGEST_WORDS,
    ):
        raise ValueError("stage endpoint digests must have shape (8,)")
    classification = (
        stomp_owner_stage_delta_valid(
            source_state,
            destination_state,
            stage_kind=stage_kind,
        )
        if classified_delta_valid is None
        else jnp.asarray(classified_delta_valid, dtype=jnp.bool_)
    )
    receipt = STOMPOwnerFinalizationStageReceipt(
        stage_kind=jnp.asarray(stage_kind, dtype=jnp.int32),
        configured=jnp.asarray(configured, dtype=jnp.bool_),
        evaluated=jnp.asarray(evaluated, dtype=jnp.bool_),
        committed=~jnp.array_equal(source_words, destination_words),
        classified_delta_valid=classification,
        stomp_update_evaluations=jnp.asarray(
            stomp_update_evaluations,
            dtype=jnp.int32,
        ),
        learner_updates_applied=jnp.asarray(
            learner_updates_applied,
            dtype=jnp.int32,
        ),
        source_digest=source_words,
        destination_digest=destination_words,
        source_step_count=source_state.step_count,
        destination_step_count=destination_state.step_count,
        source_step_words=source_state.step_words,
        destination_step_words=destination_state.step_words,
        source_executing_option=source_state.executing_option,
        destination_executing_option=destination_state.executing_option,
        receipt_checksum=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
    )
    return cast(
        STOMPOwnerFinalizationStageReceipt,
        receipt.replace(
            receipt_checksum=stomp_typed_tree_digest(
                _stage_receipt_payload(receipt)
            )
        ),
    )


def stomp_owner_stage_delta_valid(
    source: STOMPState,
    destination: STOMPState,
    *,
    stage_kind: int,
) -> Bool[Array, ""]:
    """Independently classify the exact delta of two transient endpoints."""

    if type(source) is not STOMPState or type(destination) is not STOMPState:
        raise TypeError("stage endpoints must be exact STOMPState values")
    option_search_or_dyna = _tree_exact_equal(
        destination,
        source.replace(base_learner_state=destination.base_learner_state),
    )
    feature_route = _tree_exact_equal(
        destination,
        source.replace(
            base_learner_state=destination.base_learner_state,
            base_last_obs=destination.base_last_obs,
            option_policies=destination.option_policies,
            option_models=destination.option_models,
            option_start_obs=destination.option_start_obs,
        ),
    )
    dispatch = _tree_exact_equal(
        destination,
        source.replace(
            base_last_action=destination.base_last_action,
            last_primitive_action=destination.last_primitive_action,
            option_last_intra_action=destination.option_last_intra_action,
        ),
    )
    return jnp.where(
        stage_kind == STOMP_OWNER_STAGE_OPTION_SEARCH,
        option_search_or_dyna,
        jnp.where(
            stage_kind == STOMP_OWNER_STAGE_FEATURE_ROUTE,
            feature_route,
            jnp.where(
                stage_kind == STOMP_OWNER_STAGE_DYNA,
                option_search_or_dyna,
                jnp.where(
                    (stage_kind == STOMP_OWNER_STAGE_MEMORY_DISPATCH)
                    | (stage_kind == STOMP_OWNER_STAGE_PARTNER_DISPATCH),
                    dispatch,
                    jnp.asarray(False, dtype=jnp.bool_),
                ),
            ),
        ),
    )


def _stage_receipt_payload(
    receipt: STOMPOwnerFinalizationStageReceipt,
) -> object:
    return (
        receipt.stage_kind,
        receipt.configured,
        receipt.evaluated,
        receipt.committed,
        receipt.classified_delta_valid,
        receipt.stomp_update_evaluations,
        receipt.learner_updates_applied,
        receipt.source_digest,
        receipt.destination_digest,
        receipt.source_step_count,
        receipt.destination_step_count,
        receipt.source_step_words,
        receipt.destination_step_words,
        receipt.source_executing_option,
        receipt.destination_executing_option,
    )


def _stage_receipt_valid(receipt: STOMPOwnerFinalizationStageReceipt) -> Array:
    noop = jnp.array_equal(receipt.source_digest, receipt.destination_digest)
    no_stomp_stage = receipt.stage_kind != STOMP_OWNER_STAGE_DYNA
    stage_work_valid = (
        (receipt.stomp_update_evaluations >= 0)
        & (receipt.learner_updates_applied >= 0)
        & jnp.where(
            no_stomp_stage,
            receipt.stomp_update_evaluations == 0,
            receipt.learner_updates_applied
            <= receipt.stomp_update_evaluations,
        )
        & jnp.where(
            receipt.stage_kind == STOMP_OWNER_STAGE_OPTION_SEARCH,
            receipt.learner_updates_applied >= 0,
            jnp.where(
                receipt.stage_kind == STOMP_OWNER_STAGE_DYNA,
                jnp.asarray(True, dtype=jnp.bool_),
                receipt.learner_updates_applied == 0,
            ),
        )
    )
    configured_valid = jnp.where(
        receipt.configured,
        receipt.evaluated,
        (~receipt.evaluated) & noop,
    )
    return (
        receipt.classified_delta_valid
        & stage_work_valid
        & configured_valid
        & (receipt.committed == (~noop))
        & (receipt.source_step_count == receipt.destination_step_count)
        & jnp.array_equal(
            receipt.source_step_words,
            receipt.destination_step_words,
        )
        & (
            receipt.source_executing_option
            == receipt.destination_executing_option
        )
        & jnp.array_equal(
            receipt.receipt_checksum,
            stomp_typed_tree_digest(_stage_receipt_payload(receipt)),
        )
    )


def _trace_payload(trace: STOMPOwnerFinalizationTrace) -> object:
    return (
        _TRACE_TAG,
        trace.stages,
        trace.raw_digest,
        trace.final_digest,
        trace.real_control_stomp_evaluations,
        trace.imagined_stomp_evaluations,
        trace.option_search_learner_updates,
        trace.caller_authenticated,
    )


def make_stomp_owner_finalization_trace(
    raw_state: STOMPState,
    stages: tuple[
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
        STOMPOwnerFinalizationStageReceipt,
    ],
    final_state: STOMPState,
    *,
    real_control_stomp_evaluations: Array,
    imagined_stomp_evaluations: Array,
    option_search_learner_updates: Array,
    raw_digest: Array | None = None,
    final_digest: Array | None = None,
) -> STOMPOwnerFinalizationTrace:
    """Construct a typed/checksummed transient trace from exact stage states."""

    trace = STOMPOwnerFinalizationTrace(
        raw_state=raw_state,
        stages=stages,
        final_state=final_state,
        raw_digest=(
            stomp_typed_tree_digest(raw_state)
            if raw_digest is None
            else jnp.asarray(raw_digest, dtype=jnp.uint32)
        ),
        final_digest=(
            stomp_typed_tree_digest(final_state)
            if final_digest is None
            else jnp.asarray(final_digest, dtype=jnp.uint32)
        ),
        real_control_stomp_evaluations=jnp.asarray(
            real_control_stomp_evaluations,
            dtype=jnp.int32,
        ),
        imagined_stomp_evaluations=jnp.asarray(
            imagined_stomp_evaluations,
            dtype=jnp.int32,
        ),
        option_search_learner_updates=jnp.asarray(
            option_search_learner_updates,
            dtype=jnp.int32,
        ),
        trace_checksum=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
    )
    return cast(
        STOMPOwnerFinalizationTrace,
        trace.replace(
            trace_checksum=stomp_typed_tree_digest(_trace_payload(trace))
        ),
    )


def stomp_owner_finalization_trace_valid(
    trace: STOMPOwnerFinalizationTrace,
) -> Bool[Array, ""]:
    """Validate stage order, exact chaining, classified deltas, and work."""

    if type(trace) is not STOMPOwnerFinalizationTrace:
        raise TypeError("trace must be an exact STOMPOwnerFinalizationTrace")
    if len(trace.stages) != STOMP_OWNER_STAGE_COUNT:
        raise ValueError("trace must contain exactly five ordered stages")
    expected_kinds = (
        STOMP_OWNER_STAGE_OPTION_SEARCH,
        STOMP_OWNER_STAGE_FEATURE_ROUTE,
        STOMP_OWNER_STAGE_DYNA,
        STOMP_OWNER_STAGE_MEMORY_DISPATCH,
        STOMP_OWNER_STAGE_PARTNER_DISPATCH,
    )
    valid = (
        (trace.real_control_stomp_evaluations == 1)
        & (trace.imagined_stomp_evaluations >= 0)
        & (trace.option_search_learner_updates >= 0)
        & (~trace.caller_authenticated)
        & jnp.array_equal(trace.raw_digest, stomp_typed_tree_digest(trace.raw_state))
        & jnp.array_equal(
            trace.final_digest,
            stomp_typed_tree_digest(trace.final_state),
        )
        & jnp.array_equal(
            trace.trace_checksum,
            stomp_typed_tree_digest(_trace_payload(trace)),
        )
    )
    previous_digest = trace.raw_digest
    for stage, expected_kind in zip(trace.stages, expected_kinds, strict=True):
        valid = (
            valid
            & (stage.stage_kind == expected_kind)
            & jnp.array_equal(stage.source_digest, previous_digest)
            & _stage_receipt_valid(stage)
        )
        previous_digest = stage.destination_digest
    return (
        valid
        & jnp.array_equal(previous_digest, trace.final_digest)
        & (trace.raw_state.step_count == trace.final_state.step_count)
        & jnp.array_equal(
            trace.raw_state.step_words,
            trace.final_state.step_words,
        )
        & (
            trace.raw_state.executing_option
            == trace.final_state.executing_option
        )
        & (
            trace.imagined_stomp_evaluations
            == trace.stages[2].stomp_update_evaluations
        )
        & (
            trace.option_search_learner_updates
            == trace.stages[0].learner_updates_applied
        )
    )


__all__ = [
    "STOMP_OWNER_FINALIZATION_CALLER_AUTHENTICATED",
    "STOMP_OWNER_FINALIZATION_TRACE_SCHEMA",
    "STOMP_OWNER_STAGE_COUNT",
    "STOMP_OWNER_STAGE_DYNA",
    "STOMP_OWNER_STAGE_FEATURE_ROUTE",
    "STOMP_OWNER_STAGE_MEMORY_DISPATCH",
    "STOMP_OWNER_STAGE_OPTION_SEARCH",
    "STOMP_OWNER_STAGE_PARTNER_DISPATCH",
    "STOMPOwnerFinalizationStageReceipt",
    "STOMPOwnerFinalizationTrace",
    "make_stomp_owner_finalization_trace",
    "make_stomp_owner_stage_receipt",
    "stomp_owner_finalization_trace_valid",
    "stomp_owner_stage_delta_valid",
    "stomp_typed_tree_digest",
]
