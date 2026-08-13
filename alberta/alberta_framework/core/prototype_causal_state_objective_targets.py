# mypy: disable-error-code="call-arg"
"""Atomic Prototype integration for learner-owned causal state targets.

This versioned, opt-in WP3 adapter is deliberately separate from
``PrototypeComprehensiveStateObjectives``.  The historical adapter keeps its
caller-supplied target receipt unchanged; this composition instead owns a
``CausalStateObjectiveTargetProducer`` and constructs every ordinary training
target from the exact dispatched Prototype decision plus one accepted real
``PrototypeTransition``.

The transaction caches the bit-exact raw observation, recurrent
representation, executed action, Prototype lifecycle/decision identity,
observation revision, and online-builder owner.  Prototype then evaluates the
real transition exactly once.  The adapter independently reconstructs the
final/bootstrap recurrent observation, content-binds it to the target owner,
and evaluates that owner exactly once.  The resulting detached prediction,
reward/boundary, GVF, value/advantage, and inverse-action targets are therefore
not caller inputs.  The only optional external learning signal is the target
producer's typed, content-bound cumulant receipt; its provenance binding is an
integrity claim, not proof of semantic correctness.

Both objective representation gradients are pulled back through the exact
current and final/bootstrap online recurrent sensitivities, combined, clipped,
and committed once to the Prototype destination.  Prototype state, target
owner/objective state, builder learning, the next dispatch cache, and all exact
uint64 clocks commit together or roll back bit-for-bit on any child refusal.
Natural termination suppresses bootstrap; truncation uses the final
observation before reset and requires a valid positive-discount bootstrap.

An exact RTU builder requires its matching generate-and-test lifecycle, which
prepares recurrence before
the target transaction, ranks whole complex units by internally owned frozen-
head deletion loss, atomically scrubs recycled axes from every target-owned
head, and finalizes Prototype only afterward.  Invalid attempted scoring rolls
the outer transaction back; valid immature evidence defers only replacement.
Learning-value routing, general feature lifecycles, Prototype-owned gradient
mixing, world-model builder learning, and automatic curation remain rejected.
This is isolated L0
``not_assessed`` machinery with uncalibrated objective masses.  It establishes
no causal-cumulant semantics, retention, control benefit, Forager result,
Alberta Plan completion, evidence promotion, or SOTA claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
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

from alberta_framework.core._rtu_objective_recycling import (
    _linear_stomp_replacement_scrub_valid,
    _rtu_builder_replacement_scrub_valid,
    _rtu_global_lifetime_capacity,
    _rtu_global_lifetime_state_valid,
    _scrub_objective_representation_axes,
    _selected_float32_axes_are_positive_zero,
)
from alberta_framework.core.causal_state_objective_targets import (
    CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
    CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL,
    CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES,
    CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS,
    CausalStateObjectiveAcceptedTransition,
    CausalStateObjectiveCacheResult,
    CausalStateObjectiveCumulantReceipt,
    CausalStateObjectiveDecisionReceipt,
    CausalStateObjectiveTargetProducer,
    CausalStateObjectiveTargetProducerState,
    CausalStateObjectiveTargets,
    CausalStateObjectiveUpdateResult,
    measure_causal_state_objective_target_state_nbytes,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.comprehensive_state_objectives import (
    COMPREHENSIVE_STATE_OBJECTIVES_HEADS,
    ComprehensiveStateObjectiveActionReceipt,
    ComprehensiveStateObjectivesState,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeRTUTransitionPreparation,
    PrototypeTransition,
    PrototypeTransitionDiagnostics,
    PrototypeUpdateResult,
    measure_prototype_agent_state_resources,
)
from alberta_framework.core.rtu_generate_and_test import (
    RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL,
    RTU_GENERATE_AND_TEST_MECHANISM_STATUS,
    RTUGenerateAndTest,
    RTUGenerateAndTestAdvanceReceipt,
    RTUGenerateAndTestCommitResult,
    RTUGenerateAndTestState,
)
from alberta_framework.core.state_builder import (
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
    OnlineGatedStateBuilderTransitionResult,
    RecurrentTraceUnitStateBuilder,
    RecurrentTraceUnitStateBuilderConfig,
    RecurrentTraceUnitStateBuilderState,
    RecurrentTraceUnitStateBuilderTransitionResult,
    StateBuilderLearningDiagnostics,
    replace_state_builder_learning_proposal_update,
)

PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CONFIG_SCHEMA = (
    "alberta.prototype-causal-state-objective-targets-config.v2"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_STATE_SCHEMA = (
    "alberta.prototype-causal-state-objective-targets-state.v2"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA = (
    "alberta.prototype-causal-state-objective-derived-target-receipt.v1"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA = (
    "alberta.prototype-causal-state-objective-targets-checkpoint.v2"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RESOURCE_SCHEMA = (
    "alberta.prototype-causal-state-objective-targets-resource.v2"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL = "L0"
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS = "not_assessed"
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIFETIME_SEMANTICS = (
    "exact-uint64-and-rtu-uint32-fail-stop"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP = (
    "exact-prototype-dispatch-and-transition; learner-owned-causal-targets; "
    "caller-training-targets-excluded; pre-update-frozen-head-whole-complex-"
    "RTU-causal-deletion-owner; atomic-recycled-axis-target-head-scrub"
)
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIMITATIONS = (
    "explicit-opt-in-online-RTRL-builder-only",
    "learning-value-router-and-feature-lifecycle-incompatible",
    "v18-atomic-feature-world-memory-identity-builder-owner-incompatible",
    "world-model-builder-learning-gradient-mixing-and-auto-curation-incompatible",
    "online-recurrent-sensitivity-approximation-after-parameter-updates",
    "optional-cumulant-provenance-bound-but-not-semantically-proven",
    "objective-group-masses-uncalibrated",
    "live-RTU-replacement-requires-statically-guarded-linear-OaK-STOMP-envelope",
    "exact-RTU-builder-requires-owned-generate-and-test-lifecycle",
    "live-RTU-causal-deletion-is-prequential-L0-not-held-out-outcome-evidence",
    "no-retention-control-forager-alberta-plan-evidence-or-sota-claim",
)
# One initial target cache and one successor cache accompany an armed stream.
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_MAX_TRANSITIONS = 2**64 - 2
PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS = 2**32 - 1

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_LIFECYCLE_NAMESPACE_WORDS = (0x43535450, 1)  # ``CSTP``, schema generation.


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


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare canonical JSON trees without Python's bool/int aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_fields = cast(dict[str, object], left)
        right_fields = cast(dict[str, object], right)
        return set(left_fields) == set(right_fields) and all(
            _exact_json_equal(left_fields[key], right_fields[key])
            for key in left_fields
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(
                left_items,
                right_items,
                strict=True,
            )
        )
    return bool(left == right)


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


def _require_words(value: Any, width: int, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(width,),
        dtype=jnp.dtype(jnp.uint32),
        label=label,
    )


def _require_bool(value: Any, *, label: str) -> Array:
    return _require_array(value, shape=(), dtype=jnp.dtype(jnp.bool_), label=label)


def _require_key(value: Any, *, label: str) -> None:
    try:
        data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a typed Threefry key") from error
    if (
        getattr(value, "shape", None) != ()
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


def _add_word_pairs(
    left: Array,
    right: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Add two exact uint64 word pairs and report overflow."""

    low = left[1] + right[1]
    low_carry = low < left[1]
    partial_high = left[0] + right[0]
    high_overflow = partial_high < left[0]
    high = partial_high + low_carry.astype(jnp.uint32)
    carry_overflow = low_carry & (high == 0)
    capacity = ~(high_overflow | carry_overflow)
    return jnp.stack((high, low)).astype(jnp.uint32), capacity


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
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jnp.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            equal = equal & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            equal = equal & _float_bits_equal(left_array, right_array)
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def _safe_clip_parameter_gradient(
    value: Array,
    limit: float,
) -> tuple[Bool[Array, ""], Float[Array, " parameter"], Float[Array, ""]]:
    finite = jnp.all(jnp.isfinite(value))
    safe = jnp.where(finite, value, jnp.zeros_like(value))
    scale = jnp.max(jnp.abs(safe))
    safe_scale = jnp.where(scale > 0.0, scale, jnp.float32(1.0))
    scaled_norm = jnp.sqrt(jnp.sum(jnp.square(safe / safe_scale)))
    raw_norm = scale * scaled_norm
    norm = jnp.where(
        finite & jnp.isfinite(raw_norm),
        raw_norm,
        jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
    )
    denominator = jnp.where(norm > 0.0, norm, jnp.float32(1.0))
    factor = jnp.minimum(jnp.float32(1.0), jnp.float32(limit) / denominator)
    clipped = safe * factor
    return finite & jnp.all(jnp.isfinite(clipped)), clipped, norm


def _state_nbytes(state: Any) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def _prototype_lifecycle_identity(decision_id: Array) -> UInt[Array, " 4"]:
    return jnp.concatenate(
        (
            decision_id[:2],
            jnp.asarray(_LIFECYCLE_NAMESPACE_WORDS, dtype=jnp.uint32),
        )
    ).astype(jnp.uint32)


def _rotate_left(value: Array, distance: Array) -> Array:
    right = (jnp.asarray(32, dtype=jnp.uint32) - distance) & jnp.uint32(31)
    return jnp.asarray((value << distance) | (value >> right), dtype=jnp.uint32)


def _derived_target_tag(
    prototype_decision_id: Array,
    action: Array,
    source_observation_event_words: Array,
    target_decision_words: Array,
    transition_content_tag_words: Array,
    targets: CausalStateObjectiveTargets,
) -> UInt[Array, " 4"]:
    values = (
        prototype_decision_id,
        action,
        source_observation_event_words,
        target_decision_words,
        transition_content_tag_words,
        targets.next_observation,
        targets.next_latent,
        targets.reward,
        targets.terminated,
        targets.discount,
        targets.effective_continuation,
        targets.cumulant,
        targets.gvf_targets,
        targets.current_value,
        targets.bootstrap_value,
        targets.control_value_target,
        targets.selected_action_advantage_target,
        targets.inverse_action_label,
        targets.inverse_pair_valid,
    )
    words: list[Array] = []
    for value in values:
        array = jax.lax.stop_gradient(jnp.asarray(value))
        if array.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
            converted = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.bool_):
            converted = array.astype(jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.uint32):
            converted = array
        else:
            raise TypeError("derived target tag fields must be float32/int32/bool/uint32")
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


def _empty_cumulant_receipt() -> CausalStateObjectiveCumulantReceipt:
    return CausalStateObjectiveCumulantReceipt(
        available=jnp.asarray(False, dtype=jnp.bool_),
        value=jnp.asarray(0.0, dtype=jnp.float32),
        cumulant_owner_digest=jnp.zeros((8,), dtype=jnp.uint32),
        transition_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        lifecycle_identity_words=jnp.zeros((4,), dtype=jnp.uint32),
        decision_identity_words=jnp.zeros((4,), dtype=jnp.uint32),
        source_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        provenance_words=jnp.zeros((4,), dtype=jnp.uint32),
        content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
    )


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsState:
    """One atomic Prototype, target owner/objectives, and builder-owner state."""

    prototype_state: PrototypeAgentState
    target_state: CausalStateObjectiveTargetProducerState
    pending_prototype_decision_id: UInt[Array, " 4"]
    pending_builder_step_words: UInt[Array, " 2"]
    pending_builder_update_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    transaction_words: UInt[Array, " 2"]
    rtu_generate_and_test_state: RTUGenerateAndTestState | None = None


@chex.dataclass(frozen=True)
class PrototypeCausalDerivedTargetReceipt:
    """Detached derived targets bound to the exact accepted Prototype owner."""

    prototype_decision_id: UInt[Array, " 4"]
    action: Int[Array, ""]
    source_observation_event_words: UInt[Array, " 2"]
    target_decision_words: UInt[Array, " 2"]
    transition_content_tag_words: UInt[Array, " 4"]
    targets: CausalStateObjectiveTargets
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsStartResult:
    state: PrototypeCausalStateObjectiveTargetsState
    target_cache: CausalStateObjectiveCacheResult
    source_state_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    start_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsWork:
    """Exact source-level evaluation counts for one attempted transaction."""

    prototype_update_evaluations: Int[Array, ""]
    target_owner_update_evaluations: Int[Array, ""]
    builder_proposal_evaluations: Int[Array, ""]
    builder_commit_evaluations: Int[Array, ""]
    causal_deletion_units_scored: Int[Array, ""]
    causal_deletion_frozen_head_evaluations: Int[Array, ""]
    rtu_generate_and_test_proposal_evaluations: Int[Array, ""]
    rtu_generate_and_test_commit_evaluations: Int[Array, ""]
    next_target_cache_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsUpdateResult:
    """Diagnostics for one all-or-nothing real-transition transaction."""

    state: PrototypeCausalStateObjectiveTargetsState
    action: Int[Array, ""]
    prototype_update: PrototypeUpdateResult
    prototype_transition: PrototypeTransitionDiagnostics
    accepted_target_transition: CausalStateObjectiveAcceptedTransition
    target_update: CausalStateObjectiveUpdateResult
    derived_target_receipt: PrototypeCausalDerivedTargetReceipt
    next_target_cache: CausalStateObjectiveCacheResult
    bootstrap_builder_transition: (
        OnlineGatedStateBuilderTransitionResult | RecurrentTraceUnitStateBuilderTransitionResult
    )
    builder_learning: StateBuilderLearningDiagnostics
    rtu_generate_and_test: RTUGenerateAndTestCommitResult | None
    rtu_advance_receipt: RTUGenerateAndTestAdvanceReceipt | None
    resource_work: PrototypeCausalStateObjectiveTargetsWork
    combined_raw_parameter_gradient_norm: Float[Array, ""]
    pre_transaction_words: UInt[Array, " 2"]
    post_transaction_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    transition_identity_matches: Bool[Array, ""]
    accepted_transition_matches_prototype: Bool[Array, ""]
    derived_target_receipt_valid: Bool[Array, ""]
    bootstrap_event_capacity_available: Bool[Array, ""]
    bootstrap_transition_applied: Bool[Array, ""]
    prototype_transaction_applied: Bool[Array, ""]
    target_transaction_applied: Bool[Array, ""]
    builder_sources_match: Bool[Array, ""]
    builder_destination_matches: Bool[Array, ""]
    builder_transaction_applied: Bool[Array, ""]
    rtu_observation_proposal_valid: Bool[Array, ""]
    rtu_lifecycle_source_matches: Bool[Array, ""]
    rtu_observation_transaction_applied: Bool[Array, ""]
    rtu_causal_deletion_evidence_attempted: Bool[Array, ""]
    rtu_causal_deletion_evidence_available: Bool[Array, ""]
    rtu_causal_deletion_evidence_valid: Bool[Array, ""]
    rtu_replacement_cache_safe: Bool[Array, ""]
    rtu_replacement_requires_pre_action_hook: Bool[Array, ""]
    next_target_cache_required: Bool[Array, ""]
    next_target_cache_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    derived_target_receipt_committed: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsResourceBudget:
    schema: str
    prototype_state_nbytes: int
    target_owner_state_nbytes: int
    rtu_generate_and_test_state_nbytes: int
    adapter_metadata_nbytes: int
    total_state_nbytes: int
    max_prototype_updates_per_transition: int
    max_target_owner_updates_per_transition: int
    max_objective_head_updates_per_transition: int
    max_causal_deletion_units_scored_per_transition: int
    max_causal_deletion_frozen_head_evaluations_per_transition: int
    max_builder_proposals_per_transition: int
    max_builder_commits_per_transition: int
    max_rtu_generate_and_test_proposals_per_transition: int
    max_rtu_generate_and_test_commits_per_transition: int
    max_next_target_cache_writes_per_transition: int
    max_accepted_transitions: int
    persistent_bytes_scope: str
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsScanInputs:
    next_observations: Float[Array, "steps observation"]
    next_decision_observations: Float[Array, "steps observation"]
    rewards: Float[Array, " steps"]
    discounts: Float[Array, " steps"]
    terminated: Bool[Array, " steps"]
    truncated: Bool[Array, " steps"]
    optional_cumulants: Float[Array, " steps"]
    optional_cumulant_available: Bool[Array, " steps"]
    cumulant_source_revision_words: UInt[Array, "steps 2"]
    cumulant_provenance_words: UInt[Array, "steps 4"]


@chex.dataclass(frozen=True)
class PrototypeCausalStateObjectiveTargetsScanResult:
    state: PrototypeCausalStateObjectiveTargetsState
    balanced_losses: Float[Array, " steps"]
    gvf_targets: Float[Array, "steps timescale"]
    control_value_targets: Float[Array, " steps"]
    actions: Int[Array, " steps"]
    transaction_words: UInt[Array, "steps 2"]
    update_applied: Bool[Array, " steps"]


def measure_prototype_causal_state_objective_targets_state_nbytes(
    state: PrototypeCausalStateObjectiveTargetsState,
) -> int:
    """Measure every persistent JAX-array leaf in the composition."""

    if type(state) is not PrototypeCausalStateObjectiveTargetsState:
        raise TypeError("state must be exact PrototypeCausalStateObjectiveTargetsState")
    return _state_nbytes(state)


def _zero_sized_array_leaf_indices(
    state: PrototypeCausalStateObjectiveTargetsState,
) -> list[int]:
    """Identify fixed empty leaves that Orbax cannot persist directly."""

    return [
        index
        for index, leaf in enumerate(jax.tree.leaves(state))
        if isinstance(leaf, Array) and leaf.size == 0
    ]


def _checkpoint_storage_state(
    state: PrototypeCausalStateObjectiveTargetsState,
) -> PrototypeCausalStateObjectiveTargetsState:
    """Encode config-fixed empty arrays as one-element storage sentinels."""

    return cast(
        PrototypeCausalStateObjectiveTargetsState,
        jax.tree.map(
            lambda leaf: (
                jnp.zeros((1,), dtype=leaf.dtype)
                if isinstance(leaf, Array) and leaf.size == 0
                else leaf
            ),
            state,
        ),
    )


def _restore_checkpoint_empty_arrays(
    restored: PrototypeCausalStateObjectiveTargetsState,
    template: PrototypeCausalStateObjectiveTargetsState,
) -> PrototypeCausalStateObjectiveTargetsState:
    """Decode storage sentinels from the canonical config-derived template."""

    return cast(
        PrototypeCausalStateObjectiveTargetsState,
        jax.tree.map(
            lambda stored, expected: (
                expected
                if isinstance(expected, Array) and expected.size == 0
                else stored
            ),
            restored,
            template,
        ),
    )


def _checkpoint_empty_array_storage_valid(
    restored: PrototypeCausalStateObjectiveTargetsState,
    template: PrototypeCausalStateObjectiveTargetsState,
) -> bool:
    """Require every encoded empty-array sentinel to be typed canonical zero."""

    restored_leaves, restored_structure = jax.tree.flatten(restored)
    template_leaves, template_structure = jax.tree.flatten(template)
    if cast(object, restored_structure) != cast(object, template_structure) or len(
        restored_leaves
    ) != len(template_leaves):
        return False
    for stored, expected in zip(restored_leaves, template_leaves, strict=True):
        if not isinstance(expected, Array) or expected.size != 0:
            continue
        if (
            not isinstance(stored, Array)
            or stored.shape != (1,)
            or stored.dtype != expected.dtype
        ):
            return False
        raw = np.asarray(stored)
        if raw.tobytes() != bytes(raw.nbytes):
            return False
    return True


class PrototypeCausalStateObjectiveTargets:
    """Single transaction owner for Prototype and causal comprehensive targets."""

    def __init__(
        self,
        prototype: PrototypeAgent,
        target_producer: CausalStateObjectiveTargetProducer,
        rtu_generate_and_test: RTUGenerateAndTest | None = None,
    ) -> None:
        if type(prototype) is not PrototypeAgent:
            raise TypeError("prototype must be an exact PrototypeAgent")
        if type(target_producer) is not CausalStateObjectiveTargetProducer:
            raise TypeError("target_producer must be an exact CausalStateObjectiveTargetProducer")
        if (
            rtu_generate_and_test is not None
            and type(rtu_generate_and_test) is not RTUGenerateAndTest
        ):
            raise TypeError("rtu_generate_and_test must be an exact RTUGenerateAndTest")
        config = prototype.config
        if config.learning_value_router is not None:
            raise ValueError("adapter does not compose with learning_value_router")
        if config.prototype_atomic_feature_world_memory is not None:
            raise ValueError(
                "adapter does not compose with v18 atomic feature/world/memory; "
                "that lane requires feature-lifecycle ownership and an Identity builder"
            )
        if config.prototype_feature_lifecycle is not None:
            raise ValueError("adapter does not compose with prototype_feature_lifecycle")
        if type(config.state_builder) not in {
            OnlineGatedStateBuilderConfig,
            LearnableGRUStateBuilderConfig,
            RecurrentTraceUnitStateBuilderConfig,
        }:
            raise ValueError("adapter requires an exact online RTRL state-builder config")
        if type(prototype.state_builder) not in {
            OnlineGatedStateBuilder,
            LearnableGRUStateBuilder,
            RecurrentTraceUnitStateBuilder,
        }:
            raise ValueError("adapter requires an exact online RTRL state-builder instance")
        if (
            type(config.state_builder) is RecurrentTraceUnitStateBuilderConfig
            and rtu_generate_and_test is None
        ):
            raise ValueError(
                "an exact RTU builder requires its owned generate-and-test lifecycle"
            )
        if config.learn_state_builder_from_world_model:
            raise ValueError("adapter requires Prototype world-model builder learning disabled")
        if config.representation_gradient_mixer is not None:
            raise ValueError("adapter requires Prototype representation gradient mixing disabled")
        if config.auto_curate_every != 0:
            raise ValueError("adapter requires auto_curate_every == 0")
        builder: Any = prototype.state_builder
        objectives = target_producer.objectives.config
        builder_config = cast(
            OnlineGatedStateBuilderConfig
            | LearnableGRUStateBuilderConfig
            | RecurrentTraceUnitStateBuilderConfig,
            config.state_builder,
        )
        if objectives.representation_dim != builder.feature_dim():
            raise ValueError("target objective representation_dim must match builder feature_dim")
        if objectives.observation_target_dim != builder_config.observation_dim:
            raise ValueError(
                "target objective observation_target_dim must match raw observation_dim"
            )
        if objectives.n_actions != config.oak.n_primitive_actions:
            raise ValueError("target objective n_actions must match Prototype primitive actions")
        if CAUSAL_STATE_OBJECTIVE_TARGET_EVIDENCE_LEVEL != "L0":
            raise RuntimeError("causal target producer must remain L0")
        if CAUSAL_STATE_OBJECTIVE_TARGET_OUTCOME_STATUS != "not_assessed":
            raise RuntimeError("causal target producer must remain not_assessed")
        if rtu_generate_and_test is not None:
            if type(prototype.state_builder) is not RecurrentTraceUnitStateBuilder:
                raise ValueError("rtu_generate_and_test requires an exact RTU Prototype builder")
            if type(config.state_builder) is not RecurrentTraceUnitStateBuilderConfig:
                raise ValueError("rtu_generate_and_test requires an exact RTU Prototype config")
            if rtu_generate_and_test.config.builder != config.state_builder:
                raise ValueError(
                    "rtu_generate_and_test builder config must exactly match Prototype"
                )
            if RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL != "L0":
                raise RuntimeError("RTU generate-and-test must remain an L0 mechanism")
            if RTU_GENERATE_AND_TEST_MECHANISM_STATUS != "not_assessed":
                raise RuntimeError("RTU generate-and-test must remain not_assessed")
            unsupported_components = {
                "world_model": config.world_model,
                "world_model_ensemble": config.world_model_ensemble,
                "model_replay_rehearsal": config.model_replay_rehearsal,
                "recurrent_latent_world_model_ensemble": (
                    config.recurrent_latent_world_model_ensemble
                ),
                "dreaming": config.dreaming,
                "horde_spec": config.horde_spec,
                "ia": config.ia,
                "partner_policy_fusion": config.partner_policy_fusion,
                "experiential_memory": config.experiential_memory,
                "experiential_memory_advantage_gate": (config.experiential_memory_advantage_gate),
                "gru_perception": config.gru_perception,
                "gradient_joy": config.gradient_joy,
                "option_search_control": config.option_search_control,
                "prototype_feature_lifecycle": config.prototype_feature_lifecycle,
                "prototype_feature_utility": config.prototype_feature_utility,
                "prototype_feature_utility_curation": (config.prototype_feature_utility_curation),
            }
            enabled_unsupported = sorted(
                name for name, component in unsupported_components.items() if component is not None
            )
            if enabled_unsupported:
                raise ValueError(
                    "live RTU replacement does not compose with: " + ", ".join(enabled_unsupported)
                )
            if config.n_dreams_per_step != 0:
                raise ValueError("live RTU replacement requires n_dreams_per_step == 0")
            if config.oak.stomp.base_hidden_sizes:
                raise ValueError("live RTU replacement requires a linear STOMP base learner")
            if config.oak.stomp.option_planning_backups_per_step != 0:
                raise ValueError("live RTU replacement requires option planning disabled")
            if not config.state_builder.include_raw_observation:
                raise ValueError("live RTU replacement requires raw observation features")
            raw_dim = config.state_builder.observation_dim
            hidden_dim = config.state_builder.hidden_dim
            protected = set(rtu_generate_and_test.config.protected_units)
            for subtask in config.oak.stomp.subtask_specs:
                feature_index = subtask.feature_index
                if feature_index < raw_dim:
                    continue
                relative = feature_index - raw_dim
                unit_index = relative % hidden_dim
                if relative >= 2 * hidden_dim or unit_index not in protected:
                    raise ValueError(
                        "live RTU replacement requires option subtask features "
                        "to be raw or owned by protected RTU units"
                    )
        self._prototype = prototype
        self._target_producer = target_producer
        self._builder = cast(
            OnlineGatedStateBuilder | RecurrentTraceUnitStateBuilder,
            builder,
        )
        self._rtu_generate_and_test = rtu_generate_and_test

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def target_producer(self) -> CausalStateObjectiveTargetProducer:
        return self._target_producer

    @property
    def builder(self) -> OnlineGatedStateBuilder | RecurrentTraceUnitStateBuilder:
        return self._builder

    @property
    def rtu_generate_and_test(self) -> RTUGenerateAndTest | None:
        return self._rtu_generate_and_test

    @property
    def max_accepted_transitions(self) -> int:
        if self._rtu_generate_and_test is not None:
            return PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS
        return PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_MAX_TRANSITIONS

    def to_config(self) -> dict[str, Any]:
        payload = {
            "type": "PrototypeCausalStateObjectiveTargets",
            "schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_STATE_SCHEMA,
            "receipt_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA,
            "checkpoint_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
            "ownership": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP,
            "lifetime_semantics": (PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIFETIME_SEMANTICS),
            "max_transitions": self.max_accepted_transitions,
            "limitations": list(PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIMITATIONS),
            "activation": "explicit-adapter-only",
            "prototype_config": self._prototype.to_config(),
            "target_producer_config": self._target_producer.to_config(),
        }
        if self._rtu_generate_and_test is not None:
            payload["rtu_generate_and_test_config"] = self._rtu_generate_and_test.to_config()
        return payload

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
    ) -> PrototypeCausalStateObjectiveTargets:
        expected_fields = {
            "type",
            "schema",
            "state_schema",
            "receipt_schema",
            "checkpoint_schema",
            "resource_schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "target_authority",
            "ownership",
            "lifetime_semantics",
            "max_transitions",
            "limitations",
            "activation",
            "prototype_config",
            "target_producer_config",
        }
        if "rtu_generate_and_test_config" in payload:
            expected_fields.add("rtu_generate_and_test_config")
        fields = _exact_manifest(
            payload,
            expected_fields,
            label="Prototype causal target adapter config",
        )
        fixed = {
            "type": "PrototypeCausalStateObjectiveTargets",
            "schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_STATE_SCHEMA,
            "receipt_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA,
            "checkpoint_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "target_authority": CAUSAL_STATE_OBJECTIVE_TARGET_AUTHORITY,
            "ownership": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP,
            "lifetime_semantics": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIFETIME_SEMANTICS,
            "limitations": list(PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIMITATIONS),
            "activation": "explicit-adapter-only",
        }
        for name, expected in fixed.items():
            if fields.pop(name) != expected:
                raise ValueError(f"Prototype causal target adapter {name} is unsupported")
        declared_max_transitions = fields.pop("max_transitions")
        prototype_config = fields.pop("prototype_config")
        target_config = fields.pop("target_producer_config")
        if type(prototype_config) is not dict or type(target_config) is not dict:
            raise TypeError("nested configs must be exact dictionaries")
        rtu_config_supplied = "rtu_generate_and_test_config" in fields
        rtu_config = fields.pop("rtu_generate_and_test_config", None)
        if rtu_config_supplied and type(rtu_config) is not dict:
            raise TypeError("RTU generate-and-test config must be an exact dictionary")
        restored = cls(
            PrototypeAgent.from_config(prototype_config),
            CausalStateObjectiveTargetProducer.from_config(target_config),
            (
                RTUGenerateAndTest.from_config(cast(dict[str, Any], rtu_config))
                if rtu_config_supplied
                else None
            ),
        )
        if (
            type(declared_max_transitions) is not int
            or declared_max_transitions != restored.max_accepted_transitions
        ):
            raise ValueError("Prototype causal target adapter max_transitions is unsupported")
        return restored

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeCausalStateObjectiveTargetsState:
        _require_key(key, label="key")
        if self._rtu_generate_and_test is None:
            prototype_key, target_key = jr.split(key)
            rtu_state = None
        else:
            prototype_key, target_key, rtu_key = jr.split(key, 3)
            rtu_state = self._rtu_generate_and_test.init(rtu_key)
        zero_two = jnp.zeros((2,), dtype=jnp.uint32)
        return PrototypeCausalStateObjectiveTargetsState(
            prototype_state=self._prototype.init(
                prototype_key,
                lifecycle_id=lifecycle_id,
            ),
            target_state=self._target_producer.init(target_key),
            pending_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_builder_step_words=zero_two,
            pending_builder_update_words=zero_two,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            transaction_words=zero_two,
            rtu_generate_and_test_state=rtu_state,
        )

    def _require_state_contract(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
    ) -> None:
        if type(state) is not PrototypeCausalStateObjectiveTargetsState:
            raise TypeError("state must be exact PrototypeCausalStateObjectiveTargetsState")
        if type(state.prototype_state) is not PrototypeAgentState:
            raise TypeError("state.prototype_state must be exact PrototypeAgentState")
        if type(state.target_state) is not CausalStateObjectiveTargetProducerState:
            raise TypeError("state.target_state must be exact causal target state")
        if self._rtu_generate_and_test is None:
            if state.rtu_generate_and_test_state is not None:
                raise TypeError("base adapter state cannot carry an RTU lifecycle")
        else:
            if type(state.rtu_generate_and_test_state) is not RTUGenerateAndTestState:
                raise TypeError("RTU-enabled adapter state must carry exact lifecycle state")
            self._rtu_generate_and_test.state_valid(state.rtu_generate_and_test_state)
        expected_builder_state_type = (
            RecurrentTraceUnitStateBuilderState
            if isinstance(self._builder, RecurrentTraceUnitStateBuilder)
            else OnlineGatedStateBuilderState
        )
        if type(state.prototype_state.state_builder_state) is not expected_builder_state_type:
            raise TypeError("Prototype state must contain the configured online builder state")
        self._target_producer.state_valid(state.target_state)
        cast(Any, self._builder).state_valid(state.prototype_state.state_builder_state)
        _require_words(
            state.pending_prototype_decision_id,
            4,
            label="pending_prototype_decision_id",
        )
        _require_words(state.pending_builder_step_words, 2, label="pending_builder_step_words")
        _require_words(
            state.pending_builder_update_words,
            2,
            label="pending_builder_update_words",
        )
        _require_bool(state.pending_valid, label="pending_valid")
        _require_words(state.transaction_words, 2, label="transaction_words")

    def _dynamic_state_valid(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
    ) -> Bool[Array, ""]:
        prototype = state.prototype_state
        target = state.target_state
        objectives = target.objectives_state
        builder = cast(
            OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
            prototype.state_builder_state,
        )
        rtu_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        rtu_global_lifetime_valid = jnp.asarray(True, dtype=jnp.bool_)
        target_replacement_scrub_valid = jnp.asarray(True, dtype=jnp.bool_)
        consumer_replacement_scrub_valid = jnp.asarray(True, dtype=jnp.bool_)
        replacement_event_words = jnp.zeros((2,), dtype=jnp.uint32)
        most_recent_replacement = jnp.asarray(False, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            rtu_state = cast(RTUGenerateAndTestState, state.rtu_generate_and_test_state)
            rtu_state_valid = self._rtu_generate_and_test.state_valid(rtu_state) & jnp.all(
                rtu_state.observation_words == state.transaction_words
            )
            rtu_global_lifetime_valid = _rtu_global_lifetime_state_valid(state.transaction_words)
            replacement_event_words = rtu_state.replacement_event_words
            most_recent_replacement = jnp.any(rtu_state.last_replaced_mask)
            rtu_builder_config = cast(
                RecurrentTraceUnitStateBuilderConfig,
                self._builder.config,
            )
            raw_dim = (
                rtu_builder_config.observation_dim
                if rtu_builder_config.include_raw_observation
                else 0
            )
            reset_mask = jnp.concatenate(
                (
                    jnp.zeros((raw_dim,), dtype=jnp.bool_),
                    rtu_state.last_replaced_mask,
                    rtu_state.last_replaced_mask,
                )
            )

            target_replacement_scrub_valid = jnp.all(
                jnp.stack(
                    tuple(
                        _selected_float32_axes_are_positive_zero(
                            value,
                            reset_mask,
                        )
                        for value in (
                            objectives.observation_weights,
                            objectives.latent_weights,
                            objectives.reward_weights,
                            objectives.termination_weights,
                            objectives.gvf_weights,
                            objectives.value_weights,
                            objectives.advantage_weights,
                            objectives.inverse_current_weights,
                            objectives.inverse_next_weights,
                            objectives.pending_representation,
                        )
                    )
                )
            )
            consumer_replacement_scrub_valid = (
                _rtu_builder_replacement_scrub_valid(
                    cast(RecurrentTraceUnitStateBuilderState, builder),
                    rtu_state.last_replaced_mask,
                    event_dim=rtu_builder_config.event_dim(),
                )
                & _linear_stomp_replacement_scrub_valid(
                    prototype.oak_state.stomp_state,
                    reset_mask,
                )
            )
        expected_builder_revision, builder_revision_capacity = _add_word_pairs(
            state.transaction_words,
            replacement_event_words,
        )
        owner_successor, owner_capacity = _increment_words(state.pending_builder_update_words)
        initial_owner = jnp.all(state.transaction_words == 0) & jnp.all(
            state.pending_builder_update_words == builder.update_words
        )
        advanced_owner = jnp.any(state.transaction_words != 0) & jnp.where(
            most_recent_replacement,
            jnp.all(state.pending_builder_update_words == builder.update_words),
            owner_capacity & jnp.all(owner_successor == builder.update_words),
        )
        expected_lifecycle = _prototype_lifecycle_identity(prototype.current_decision_id)
        pending_filled = (
            prototype.started
            & target.pending_valid
            & objectives.pending_valid
            & jnp.all(state.pending_prototype_decision_id == prototype.current_decision_id)
            & _float_bits_equal(
                target.pending_observation,
                prototype.current_raw_observation,
            )
            & _float_bits_equal(
                objectives.pending_representation,
                prototype.current_representation,
            )
            & (objectives.pending_action == prototype.current_action)
            & jnp.all(
                objectives.pending_representation_revision_words
                == prototype.observation_event_words
            )
            & jnp.all(target.pending_lifecycle_identity_words == expected_lifecycle)
            & jnp.all(target.pending_decision_identity_words == prototype.current_decision_id)
            & jnp.all(state.pending_builder_step_words == builder.step_words)
            & (initial_owner | advanced_owner)
        )
        pending_empty = (
            (~prototype.started)
            & (~target.pending_valid)
            & (~objectives.pending_valid)
            & jnp.all(state.pending_prototype_decision_id == 0)
            & jnp.all(state.pending_builder_step_words == 0)
            & jnp.all(state.pending_builder_update_words == 0)
        )
        builder_state_valid = cast(
            Array,
            cast(Any, self._builder).state_valid(builder),
        )
        return (
            self._prototype.validate_state(prototype)
            & self._target_producer.state_valid(target)
            & builder_state_valid
            & rtu_state_valid
            & rtu_global_lifetime_valid
            & target_replacement_scrub_valid
            & consumer_replacement_scrub_valid
            & (state.pending_valid == target.pending_valid)
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
            & jnp.all(state.transaction_words == prototype.step_words)
            & jnp.all(state.transaction_words == target.transition_words)
            & jnp.all(state.transaction_words == objectives.update_words)
            & builder_revision_capacity
            & jnp.all(expected_builder_revision == builder.update_words)
        )

    def state_valid(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
    ) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def _decision_receipt(
        self,
        target: CausalStateObjectiveTargetProducerState,
    ) -> CausalStateObjectiveDecisionReceipt:
        objectives = target.objectives_state
        return CausalStateObjectiveDecisionReceipt(
            observation=target.pending_observation,
            representation=objectives.pending_representation,
            action=objectives.pending_action,
            representation_revision_words=(objectives.pending_representation_revision_words),
            lifecycle_identity_words=target.pending_lifecycle_identity_words,
            decision_identity_words=target.pending_decision_identity_words,
            objective_action_identity_words=(target.pending_objective_action_identity_words),
            producer_decision_words=target.decision_words,
        )

    def start(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
        initial_observation: Array,
    ) -> PrototypeCausalStateObjectiveTargetsStartResult:
        self._require_state_contract(state)
        return cast(
            PrototypeCausalStateObjectiveTargetsStartResult,
            self._start_jit(state, initial_observation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _start_jit(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
        initial_observation: Array,
    ) -> PrototypeCausalStateObjectiveTargetsStartResult:
        source_valid = self._dynamic_state_valid(state) & (~state.pending_valid)
        prototype = self._prototype.start(
            state.prototype_state,
            initial_observation,
        )
        builder = cast(
            OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
            prototype.state_builder_state,
        )
        target_cache = self._target_producer.cache_decision(
            state.target_state,
            observation=prototype.current_raw_observation,
            representation=prototype.current_representation,
            action=prototype.current_action,
            representation_revision_words=prototype.observation_event_words,
            lifecycle_identity_words=_prototype_lifecycle_identity(prototype.current_decision_id),
            decision_identity_words=prototype.current_decision_id,
        )
        candidate = PrototypeCausalStateObjectiveTargetsState(
            prototype_state=prototype,
            target_state=target_cache.state,
            rtu_generate_and_test_state=state.rtu_generate_and_test_state,
            pending_prototype_decision_id=prototype.current_decision_id,
            pending_builder_step_words=builder.step_words,
            pending_builder_update_words=builder.update_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            transaction_words=state.transaction_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = source_valid & prototype.started & target_cache.cache_applied & candidate_valid
        return PrototypeCausalStateObjectiveTargetsStartResult(
            state=cast(
                PrototypeCausalStateObjectiveTargetsState,
                _tree_select(applied, candidate, state),
            ),
            target_cache=target_cache,
            source_state_valid=source_valid,
            candidate_state_valid=candidate_valid,
            start_applied=applied,
        )

    def bind_optional_cumulant(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
        *,
        value: Array,
        source_revision_words: Array,
        provenance_words: Array,
    ) -> CausalStateObjectiveCumulantReceipt:
        """Bind the only caller-supplied optional learning target."""

        self._require_state_contract(state)
        return self._target_producer.bind_optional_cumulant(
            state.target_state,
            value=value,
            source_revision_words=source_revision_words,
            provenance_words=provenance_words,
        )

    def update_transition(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
        transition: PrototypeTransition,
        optional_cumulant: CausalStateObjectiveCumulantReceipt | None = None,
    ) -> PrototypeCausalStateObjectiveTargetsUpdateResult:
        self._require_state_contract(state)
        if type(transition) is not PrototypeTransition:
            raise TypeError("transition must be an exact PrototypeTransition")
        if optional_cumulant is not None and type(optional_cumulant) is not (
            CausalStateObjectiveCumulantReceipt
        ):
            raise TypeError("optional_cumulant must be an exact cumulant receipt")
        receipt = _empty_cumulant_receipt() if optional_cumulant is None else optional_cumulant
        return cast(
            PrototypeCausalStateObjectiveTargetsUpdateResult,
            self._update_transition_jit(state, transition, receipt),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_transition_jit(
        self,
        state: PrototypeCausalStateObjectiveTargetsState,
        transition: PrototypeTransition,
        optional_cumulant: CausalStateObjectiveCumulantReceipt,
    ) -> PrototypeCausalStateObjectiveTargetsUpdateResult:
        source_valid = self._dynamic_state_valid(state) & state.pending_valid
        rtu_global_lifetime_capacity = jnp.asarray(True, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            rtu_global_lifetime_capacity = _rtu_global_lifetime_capacity(state.transaction_words)
        prototype_source = state.prototype_state
        target_source = state.target_state
        source_builder = cast(
            OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
            prototype_source.state_builder_state,
        )
        decision_receipt = self._decision_receipt(target_source)
        transition_identity_matches = (
            jnp.all(transition.decision_id == state.pending_prototype_decision_id)
            & (transition.action == prototype_source.current_action)
            & _float_bits_equal(
                transition.observation,
                prototype_source.current_raw_observation,
            )
            & _float_bits_equal(
                decision_receipt.observation,
                transition.observation,
            )
            & _float_bits_equal(
                decision_receipt.representation,
                prototype_source.current_representation,
            )
            & (decision_receipt.action == transition.action)
            & jnp.all(
                decision_receipt.representation_revision_words
                == prototype_source.observation_event_words
            )
        )
        proposed_transaction_words, transaction_capacity = _increment_words(state.transaction_words)
        boundary = transition.terminated | transition.truncated
        rtu_preparation: PrototypeRTUTransitionPreparation | None = None
        prototype_update: PrototypeUpdateResult | None = None
        bootstrap_transition: Any
        prototype_destination: Any
        if self._rtu_generate_and_test is not None:
            rtu_preparation = self._prototype.prepare_rtu_transition(
                prototype_source,
                transition,
            )
            bootstrap_transition = rtu_preparation.bootstrap_transition
            prototype_destination = rtu_preparation.decision_builder_state
            bootstrap_transition_applied = rtu_preparation.preparation_valid
            destination_matches = jnp.asarray(True, dtype=jnp.bool_)
            prototype_applied = rtu_preparation.preparation_valid
        else:
            prototype_update = self._prototype.update_transition(
                prototype_source,
                transition,
            )
            prototype_applied = prototype_update.transition_diagnostics.valid
            bootstrap_transition = self._builder.update_with_status(
                source_builder,
                transition.next_observation,
                transition.action,
                transition.reward,
                transition.discount,
            )
            reset_builder = self._builder.reset_episode(bootstrap_transition.state)
            restart_transition = self._builder.update_with_status(
                reset_builder,
                transition.next_decision_observation,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            expected_destination = cast(
                OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
                _tree_select(
                    boundary,
                    restart_transition.state,
                    bootstrap_transition.state,
                ),
            )
            bootstrap_transition_applied = bootstrap_transition.transition_applied & jnp.where(
                boundary,
                restart_transition.transition_applied,
                jnp.asarray(True, dtype=jnp.bool_),
            )
            prototype_destination = cast(
                OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
                prototype_update.state.state_builder_state,
            )
            destination_matches = _tree_exact_equal(
                expected_destination,
                prototype_destination,
            )
        bootstrap_event_words, bootstrap_event_capacity = _increment_words(
            prototype_source.observation_event_words
        )
        accepted_target_transition = self._target_producer.bind_accepted_transition(
            target_source,
            next_observation=transition.next_observation,
            next_representation=bootstrap_transition.representation,
            next_representation_revision_words=bootstrap_event_words,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            bootstrap_valid=~transition.terminated,
        )
        target_update = self._target_producer.update(
            target_source,
            decision_receipt,
            accepted_target_transition,
            optional_cumulant,
        )
        accepted_matches = (
            jnp.all(
                accepted_target_transition.transition_revision_words == proposed_transaction_words
            )
            & jnp.all(target_update.post_transition_words == proposed_transaction_words)
            & jnp.all(accepted_target_transition.decision_identity_words == transition.decision_id)
            & (accepted_target_transition.action == transition.action)
            & _float_bits_equal(
                accepted_target_transition.source_observation,
                transition.observation,
            )
            & _float_bits_equal(
                accepted_target_transition.next_observation,
                transition.next_observation,
            )
            & _float_bits_equal(
                accepted_target_transition.reward,
                transition.reward,
            )
            & _float_bits_equal(
                accepted_target_transition.discount,
                transition.discount,
            )
            & (accepted_target_transition.terminated == transition.terminated)
            & (accepted_target_transition.truncated == transition.truncated)
        )
        derived_tag = _derived_target_tag(
            transition.decision_id,
            transition.action,
            prototype_source.observation_event_words,
            target_source.decision_words,
            accepted_target_transition.content_tag_words,
            target_update.targets,
        )
        derived_receipt = PrototypeCausalDerivedTargetReceipt(
            prototype_decision_id=transition.decision_id,
            action=transition.action,
            source_observation_event_words=(prototype_source.observation_event_words),
            target_decision_words=target_source.decision_words,
            transition_content_tag_words=(accepted_target_transition.content_tag_words),
            targets=target_update.targets,
            content_tag_words=derived_tag,
        )
        derived_receipt_valid = (
            target_update.update_applied
            & jnp.all(derived_receipt.prototype_decision_id == state.pending_prototype_decision_id)
            & (derived_receipt.action == prototype_source.current_action)
            & jnp.all(
                derived_receipt.source_observation_event_words
                == prototype_source.observation_event_words
            )
            & jnp.all(derived_receipt.target_decision_words == target_source.decision_words)
            & jnp.all(
                derived_receipt.transition_content_tag_words
                == accepted_target_transition.content_tag_words
            )
            & jnp.all(
                derived_receipt.content_tag_words
                == _derived_target_tag(
                    derived_receipt.prototype_decision_id,
                    derived_receipt.action,
                    derived_receipt.source_observation_event_words,
                    derived_receipt.target_decision_words,
                    derived_receipt.transition_content_tag_words,
                    derived_receipt.targets,
                )
            )
        )
        rtu_causal_deletion_loss_change: Array | None = None
        rtu_causal_deletion_evidence_available: Array | None = None
        rtu_causal_deletion_evidence_attempted = jnp.asarray(
            False,
            dtype=jnp.bool_,
        )
        rtu_causal_deletion_evidence_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            rtu_causal_deletion_evidence_attempted = jnp.asarray(
                True,
                dtype=jnp.bool_,
            )
            rtu_builder_config = cast(
                RecurrentTraceUnitStateBuilderConfig,
                self._builder.config,
            )
            raw_dim = (
                rtu_builder_config.observation_dim
                if rtu_builder_config.include_raw_observation
                else 0
            )
            hidden_dim = rtu_builder_config.hidden_dim
            unit_ids = jnp.arange(hidden_dim, dtype=jnp.int32)
            objective_source = target_source.objectives_state
            objective_receipt = ComprehensiveStateObjectiveActionReceipt(
                representation=decision_receipt.representation,
                action=decision_receipt.action,
                representation_revision_words=(decision_receipt.representation_revision_words),
                action_identity_words=(decision_receipt.objective_action_identity_words),
            )

            def frozen_head_deletion_loss(
                unit_index: Array,
            ) -> tuple[Array, Array]:
                real_index = raw_dim + unit_index
                imaginary_index = raw_dim + hidden_dim + unit_index
                deleted_representation = (
                    objective_receipt.representation.at[real_index]
                    .set(jnp.float32(0.0))
                    .at[imaginary_index]
                    .set(jnp.float32(0.0))
                )
                deleted_state = cast(
                    ComprehensiveStateObjectivesState,
                    dataclasses.replace(
                        cast(Any, objective_source),
                        pending_representation=deleted_representation,
                    ),
                )
                deleted_receipt = cast(
                    ComprehensiveStateObjectiveActionReceipt,
                    dataclasses.replace(
                        cast(Any, objective_receipt),
                        representation=deleted_representation,
                    ),
                )
                counterfactual = self._target_producer.objectives._update_jit(
                    deleted_state,
                    deleted_receipt,
                    target_update.targets.next_latent,
                    accepted_target_transition.next_representation_revision_words,
                    target_update.targets.next_observation,
                    target_update.targets.reward,
                    target_update.targets.terminated,
                    target_update.targets.cumulant,
                    target_update.targets.effective_continuation,
                    target_update.targets.control_value_target,
                    target_update.targets.selected_action_advantage_target,
                )
                return counterfactual.balanced_loss, counterfactual.update_applied

            deleted_losses, deletion_transactions_applied = jax.vmap(frozen_head_deletion_loss)(
                unit_ids
            )
            raw_deletion_change = deleted_losses - target_update.balanced_loss
            rtu_causal_deletion_evidence_valid = (
                target_update.update_applied
                & jnp.all(deletion_transactions_applied)
                & jnp.isfinite(target_update.balanced_loss)
                & jnp.all(jnp.isfinite(raw_deletion_change))
            )
            rtu_causal_deletion_evidence_available = rtu_causal_deletion_evidence_valid
            rtu_causal_deletion_loss_change = jnp.where(
                rtu_causal_deletion_evidence_available,
                raw_deletion_change,
                jnp.zeros_like(raw_deletion_change),
            )
        current_proposal = cast(Any, self._builder).propose_learning_update(
            source_builder,
            target_update.current_representation_gradient,
        )
        next_proposal = cast(Any, self._builder).propose_learning_update(
            bootstrap_transition.state,
            target_update.next_representation_gradient,
        )
        builder_sources_match = (
            _float_bits_equal(
                current_proposal.source_parameters,
                next_proposal.source_parameters,
            )
            & (current_proposal.source_update_count == next_proposal.source_update_count)
            & jnp.all(current_proposal.source_update_words == next_proposal.source_update_words)
            & jnp.all(current_proposal.builder_fingerprint == next_proposal.builder_fingerprint)
        )
        combined_raw_gradient = (
            current_proposal.raw_parameter_gradient + next_proposal.raw_parameter_gradient
        )
        combined_valid, combined_clipped, combined_norm = _safe_clip_parameter_gradient(
            combined_raw_gradient,
            self._builder.config.gradient_clip,
        )
        combined_update = (
            -jnp.asarray(self._builder.config.step_size, dtype=jnp.float32) * combined_clipped
        )
        proposal_approved = (
            source_valid
            & transition_identity_matches
            & transaction_capacity
            & rtu_global_lifetime_capacity
            & prototype_applied
            & bootstrap_transition_applied
            & destination_matches
            & bootstrap_event_capacity
            & target_update.update_applied
            & accepted_matches
            & derived_receipt_valid
            & current_proposal.valid
            & next_proposal.valid
            & builder_sources_match
            & combined_valid
            & rtu_causal_deletion_evidence_valid
        )
        combined_proposal = replace_state_builder_learning_proposal_update(
            current_proposal,
            combined_update,
            proposal_approved,
        )
        learned_builder: OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState
        builder_learning: StateBuilderLearningDiagnostics
        rtu_result: RTUGenerateAndTestCommitResult | None = None
        rtu_receipt: RTUGenerateAndTestAdvanceReceipt | None = None
        rtu_proposal_valid = jnp.asarray(True, dtype=jnp.bool_)
        rtu_lifecycle_source_matches = jnp.asarray(True, dtype=jnp.bool_)
        rtu_transaction_applied = jnp.asarray(True, dtype=jnp.bool_)
        rtu_replacement_cache_safe = jnp.asarray(True, dtype=jnp.bool_)
        rtu_replacement_requires_pre_action_hook = jnp.asarray(
            False,
            dtype=jnp.bool_,
        )
        rtu_replacement_selected = jnp.asarray(False, dtype=jnp.bool_)
        next_rtu_state = state.rtu_generate_and_test_state
        post_replacement_target = target_update.state
        effective_learned_builder: (
            OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState
        )
        if self._rtu_generate_and_test is not None:
            rtu_source = cast(
                RTUGenerateAndTestState,
                state.rtu_generate_and_test_state,
            )
            rtu_receipt = self._rtu_generate_and_test.make_advance_receipt(
                cast(RecurrentTraceUnitStateBuilderState, source_builder),
                bootstrap_observation=transition.next_observation,
                previous_action=transition.action,
                previous_reward=transition.reward,
                previous_discount=transition.discount,
                episode_boundary=boundary,
                restart_observation=transition.next_decision_observation,
            )
            rtu_proposal = self._rtu_generate_and_test.propose(
                rtu_source,
                cast(RecurrentTraceUnitStateBuilderState, source_builder),
                target_update.current_representation_gradient,
                combined_proposal,
                rtu_receipt,
                replacement_allowed=(prototype_source.oak_state.stomp_state.executing_option < 0),
                causal_deletion_loss_change=rtu_causal_deletion_loss_change,
                causal_deletion_evidence_available=(rtu_causal_deletion_evidence_available),
                require_causal_evidence=True,
            )
            rtu_lifecycle_source_matches = _tree_exact_equal(
                rtu_proposal.source_state,
                rtu_source,
            )
            ordinary_learning_diagnostics = rtu_proposal.ordinary_learning_diagnostics
            if ordinary_learning_diagnostics is None:
                raise RuntimeError("RTU adapter requires ordinary builder-learning diagnostics")
            learned_builder = rtu_proposal.live_builder_state
            builder_learning = ordinary_learning_diagnostics
            rtu_result = self._rtu_generate_and_test.commit(
                rtu_source,
                rtu_proposal.live_builder_state,
                rtu_proposal,
            )
            rtu_proposal_valid = rtu_proposal.valid
            rtu_transaction_applied = rtu_result.diagnostics.applied
            rtu_replacement_selected = jnp.any(rtu_proposal.selected_mask)
            objective_reset_mask = jnp.concatenate(
                (
                    jnp.zeros((raw_dim,), dtype=jnp.bool_),
                    rtu_proposal.selected_mask,
                    rtu_proposal.selected_mask,
                )
            )
            post_replacement_objectives = cast(
                ComprehensiveStateObjectivesState,
                jax.lax.cond(
                    rtu_replacement_selected,
                    lambda: _scrub_objective_representation_axes(
                        target_update.state.objectives_state,
                        objective_reset_mask,
                    ),
                    lambda: target_update.state.objectives_state,
                ),
            )
            post_replacement_target = cast(
                CausalStateObjectiveTargetProducerState,
                dataclasses.replace(
                    cast(Any, target_update.state),
                    objectives_state=post_replacement_objectives,
                ),
            )
            if rtu_preparation is None:
                raise RuntimeError("RTU lifecycle requires a Prototype preparation")
            finalization_receipt = self._prototype.bind_rtu_finalization(
                rtu_preparation,
                rtu_result.builder_state,
                rtu_proposal.selected_mask,
                rtu_proposal,
            )
            prototype_update = self._prototype.finalize_rtu_transition(
                prototype_source,
                transition,
                finalization_receipt,
                self._rtu_generate_and_test,
            )
            prototype_applied = prototype_update.transition_diagnostics.valid
            rtu_replacement_cache_safe = prototype_applied
            effective_learned_builder = rtu_result.builder_state
            next_rtu_state = rtu_result.state
            learned_prototype = prototype_update.state
        else:
            learned_builder, builder_learning = self._builder.commit_learning_update(
                prototype_destination,
                combined_proposal,
            )
            effective_learned_builder = learned_builder
            if prototype_update is None:
                raise RuntimeError("base adapter requires a Prototype result")
            learned_prototype = cast(
                PrototypeAgentState,
                dataclasses.replace(
                    cast(Any, prototype_update.state),
                    state_builder_state=effective_learned_builder,
                ),
            )
        next_cache_required = learned_prototype.started
        next_target_cache = self._target_producer.cache_decision(
            post_replacement_target,
            observation=learned_prototype.current_raw_observation,
            representation=learned_prototype.current_representation,
            action=learned_prototype.current_action,
            representation_revision_words=(learned_prototype.observation_event_words),
            lifecycle_identity_words=_prototype_lifecycle_identity(
                learned_prototype.current_decision_id
            ),
            decision_identity_words=learned_prototype.current_decision_id,
        )
        next_cache_valid = jnp.where(
            next_cache_required,
            next_target_cache.cache_applied,
            ~post_replacement_target.pending_valid,
        )
        candidate_target = cast(
            CausalStateObjectiveTargetProducerState,
            _tree_select(
                next_cache_required,
                next_target_cache.state,
                post_replacement_target,
            ),
        )
        candidate_pending_decision = jnp.where(
            next_cache_required,
            learned_prototype.current_decision_id,
            jnp.zeros((4,), dtype=jnp.uint32),
        )
        candidate_pending_step = jnp.where(
            next_cache_required,
            learned_builder.step_words,
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate_pending_update = jnp.where(
            next_cache_required,
            jnp.where(
                rtu_replacement_selected,
                effective_learned_builder.update_words,
                prototype_destination.update_words,
            ),
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate = PrototypeCausalStateObjectiveTargetsState(
            prototype_state=learned_prototype,
            target_state=candidate_target,
            rtu_generate_and_test_state=next_rtu_state,
            pending_prototype_decision_id=candidate_pending_decision,
            pending_builder_step_words=candidate_pending_step,
            pending_builder_update_words=candidate_pending_update,
            pending_valid=next_cache_required,
            transaction_words=proposed_transaction_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        next_replacement_event_words = jnp.zeros((2,), dtype=jnp.uint32)
        rtu_observation_clock_matches = jnp.asarray(True, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            committed_rtu_state = cast(RTUGenerateAndTestState, next_rtu_state)
            next_replacement_event_words = committed_rtu_state.replacement_event_words
            rtu_observation_clock_matches = jnp.all(
                committed_rtu_state.observation_words == proposed_transaction_words
            )
        expected_builder_words, builder_words_capacity = _add_word_pairs(
            proposed_transaction_words,
            next_replacement_event_words,
        )
        if prototype_update is None:
            raise RuntimeError("adapter transaction requires a Prototype result")
        clocks_match = (
            jnp.all(prototype_update.state.step_words == proposed_transaction_words)
            & jnp.all(target_update.state.transition_words == proposed_transaction_words)
            & rtu_observation_clock_matches
            & builder_words_capacity
            & jnp.all(effective_learned_builder.update_words == expected_builder_words)
        )
        applied = (
            proposal_approved
            & destination_matches
            & builder_learning.applied
            & rtu_proposal_valid
            & rtu_lifecycle_source_matches
            & rtu_transaction_applied
            & rtu_replacement_cache_safe
            & next_cache_valid
            & clocks_match
            & candidate_valid
        )
        final_state = cast(
            PrototypeCausalStateObjectiveTargetsState,
            _tree_select(applied, candidate, state),
        )
        work = PrototypeCausalStateObjectiveTargetsWork(
            prototype_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            target_owner_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            builder_proposal_evaluations=jnp.asarray(2, dtype=jnp.int32),
            builder_commit_evaluations=jnp.asarray(
                4 if self._rtu_generate_and_test is not None else 1,
                dtype=jnp.int32,
            ),
            causal_deletion_units_scored=jnp.asarray(
                hidden_dim if self._rtu_generate_and_test is not None else 0,
                dtype=jnp.int32,
            ),
            causal_deletion_frozen_head_evaluations=jnp.asarray(
                (
                    len(COMPREHENSIVE_STATE_OBJECTIVES_HEADS) * hidden_dim
                    if self._rtu_generate_and_test is not None
                    else 0
                ),
                dtype=jnp.int32,
            ),
            rtu_generate_and_test_proposal_evaluations=jnp.asarray(
                1 if self._rtu_generate_and_test is not None else 0,
                dtype=jnp.int32,
            ),
            rtu_generate_and_test_commit_evaluations=jnp.asarray(
                2 if self._rtu_generate_and_test is not None else 0,
                dtype=jnp.int32,
            ),
            next_target_cache_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )
        return PrototypeCausalStateObjectiveTargetsUpdateResult(
            state=final_state,
            action=final_state.prototype_state.current_action,
            prototype_update=prototype_update,
            prototype_transition=prototype_update.transition_diagnostics,
            accepted_target_transition=accepted_target_transition,
            target_update=target_update,
            derived_target_receipt=derived_receipt,
            next_target_cache=next_target_cache,
            bootstrap_builder_transition=bootstrap_transition,
            builder_learning=builder_learning,
            rtu_generate_and_test=rtu_result,
            rtu_advance_receipt=rtu_receipt,
            resource_work=work,
            combined_raw_parameter_gradient_norm=combined_norm,
            pre_transaction_words=state.transaction_words,
            post_transaction_words=final_state.transaction_words,
            source_state_valid=source_valid,
            transition_identity_matches=transition_identity_matches,
            accepted_transition_matches_prototype=accepted_matches,
            derived_target_receipt_valid=derived_receipt_valid,
            bootstrap_event_capacity_available=bootstrap_event_capacity,
            bootstrap_transition_applied=bootstrap_transition_applied,
            prototype_transaction_applied=prototype_applied,
            target_transaction_applied=target_update.update_applied,
            builder_sources_match=builder_sources_match,
            builder_destination_matches=destination_matches,
            builder_transaction_applied=builder_learning.applied,
            rtu_observation_proposal_valid=rtu_proposal_valid,
            rtu_lifecycle_source_matches=rtu_lifecycle_source_matches,
            rtu_observation_transaction_applied=(applied & rtu_transaction_applied),
            rtu_causal_deletion_evidence_attempted=(rtu_causal_deletion_evidence_attempted),
            rtu_causal_deletion_evidence_available=(
                jnp.asarray(False, dtype=jnp.bool_)
                if rtu_causal_deletion_evidence_available is None
                else rtu_causal_deletion_evidence_available
            ),
            rtu_causal_deletion_evidence_valid=(rtu_causal_deletion_evidence_valid),
            rtu_replacement_cache_safe=rtu_replacement_cache_safe,
            rtu_replacement_requires_pre_action_hook=(rtu_replacement_requires_pre_action_hook),
            next_target_cache_required=next_cache_required,
            next_target_cache_valid=next_cache_valid,
            lifetime_capacity_available=(transaction_capacity & rtu_global_lifetime_capacity),
            candidate_state_valid=candidate_valid,
            derived_target_receipt_committed=applied,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: PrototypeCausalStateObjectiveTargetsState | None = None,
    ) -> PrototypeCausalStateObjectiveTargetsResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        prototype_nbytes = measure_prototype_agent_state_resources(
            reference.prototype_state
        ).total_nbytes
        target_nbytes = measure_causal_state_objective_target_state_nbytes(reference.target_state)
        rtu_nbytes = (
            _state_nbytes(reference.rtu_generate_and_test_state)
            if reference.rtu_generate_and_test_state is not None
            else 0
        )
        metadata = 16 + 8 + 8 + 1 + 8
        budget = PrototypeCausalStateObjectiveTargetsResourceBudget(
            schema=PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RESOURCE_SCHEMA,
            prototype_state_nbytes=prototype_nbytes,
            target_owner_state_nbytes=target_nbytes,
            rtu_generate_and_test_state_nbytes=rtu_nbytes,
            adapter_metadata_nbytes=metadata,
            total_state_nbytes=(prototype_nbytes + target_nbytes + rtu_nbytes + metadata),
            max_prototype_updates_per_transition=1,
            max_target_owner_updates_per_transition=1,
            max_objective_head_updates_per_transition=len(
                CAUSAL_STATE_OBJECTIVE_TARGET_HEAD_FAMILIES
            ),
            max_causal_deletion_units_scored_per_transition=(
                self._rtu_generate_and_test.config.builder.hidden_dim
                if self._rtu_generate_and_test is not None
                else 0
            ),
            max_causal_deletion_frozen_head_evaluations_per_transition=(
                len(COMPREHENSIVE_STATE_OBJECTIVES_HEADS)
                * self._rtu_generate_and_test.config.builder.hidden_dim
                if self._rtu_generate_and_test is not None
                else 0
            ),
            max_builder_proposals_per_transition=2,
            max_builder_commits_per_transition=(
                4 if self._rtu_generate_and_test is not None else 1
            ),
            max_rtu_generate_and_test_proposals_per_transition=(
                1 if self._rtu_generate_and_test is not None else 0
            ),
            max_rtu_generate_and_test_commits_per_transition=(
                2 if self._rtu_generate_and_test is not None else 0
            ),
            max_next_target_cache_writes_per_transition=1,
            max_accepted_transitions=self.max_accepted_transitions,
            persistent_bytes_scope=(
                "all-JAX-array-leaves-in-composed-state; Python composition excluded"
            ),
            temporary_bytes_scope=(
                "source-level-counterfactual-child-results; compiler/XLA workspace excluded"
            ),
        )
        if (
            measure_prototype_causal_state_objective_targets_state_nbytes(reference)
            != budget.total_state_nbytes
        ):
            raise ValueError("composed state allocation differs from resource declaration")
        return budget


def _require_scan_inputs(
    adapter: PrototypeCausalStateObjectiveTargets,
    inputs: PrototypeCausalStateObjectiveTargetsScanInputs,
) -> int:
    if type(inputs) is not PrototypeCausalStateObjectiveTargetsScanInputs:
        raise TypeError("inputs must be exact PrototypeCausalStateObjectiveTargetsScanInputs")
    if getattr(inputs.next_observations, "ndim", None) != 2:
        raise ValueError("next_observations must have rank two")
    steps = inputs.next_observations.shape[0]
    observation_dim = adapter.target_producer.config.objectives_config.observation_target_dim
    contracts = {
        "next_observations": (
            inputs.next_observations,
            (steps, observation_dim),
            jnp.dtype(jnp.float32),
        ),
        "next_decision_observations": (
            inputs.next_decision_observations,
            (steps, observation_dim),
            jnp.dtype(jnp.float32),
        ),
        "rewards": (inputs.rewards, (steps,), jnp.dtype(jnp.float32)),
        "discounts": (inputs.discounts, (steps,), jnp.dtype(jnp.float32)),
        "terminated": (inputs.terminated, (steps,), jnp.dtype(jnp.bool_)),
        "truncated": (inputs.truncated, (steps,), jnp.dtype(jnp.bool_)),
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
            (steps, 4),
            jnp.dtype(jnp.uint32),
        ),
    }
    for label, (value, shape, dtype) in contracts.items():
        _require_array(value, shape=shape, dtype=dtype, label=label)
    return steps


def run_prototype_causal_state_objective_targets_scan(
    adapter: PrototypeCausalStateObjectiveTargets,
    state: PrototypeCausalStateObjectiveTargetsState,
    inputs: PrototypeCausalStateObjectiveTargetsScanInputs,
) -> PrototypeCausalStateObjectiveTargetsScanResult:
    """Run fixed-shape accepted real transitions without caller training targets."""

    if type(adapter) is not PrototypeCausalStateObjectiveTargets:
        raise TypeError("adapter must be exact PrototypeCausalStateObjectiveTargets")
    adapter._require_state_contract(state)
    _require_scan_inputs(adapter, inputs)

    def body(
        carry: PrototypeCausalStateObjectiveTargetsState,
        row: tuple[Array, ...],
    ) -> tuple[PrototypeCausalStateObjectiveTargetsState, tuple[Array, ...]]:
        (
            next_observation,
            next_decision_observation,
            reward,
            discount,
            terminated,
            truncated,
            optional_cumulant,
            optional_available,
            source_revision,
            provenance,
        ) = row
        prototype = carry.prototype_state
        transition = PrototypeTransition(
            observation=prototype.current_raw_observation,
            action=prototype.current_action,
            decision_id=prototype.current_decision_id,
            reward=reward,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            next_observation=next_observation,
            next_decision_observation=next_decision_observation,
        )
        if adapter.target_producer.config.cumulant_mode == "bound_optional":
            candidate_cumulant = adapter.bind_optional_cumulant(
                carry,
                value=optional_cumulant,
                source_revision_words=source_revision,
                provenance_words=provenance,
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
        result = adapter.update_transition(carry, transition, cumulant)
        return result.state, (
            result.target_update.balanced_loss,
            result.target_update.targets.gvf_targets,
            result.target_update.targets.control_value_target,
            result.action,
            result.post_transaction_words,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            inputs.next_observations,
            inputs.next_decision_observations,
            inputs.rewards,
            inputs.discounts,
            inputs.terminated,
            inputs.truncated,
            inputs.optional_cumulants,
            inputs.optional_cumulant_available,
            inputs.cumulant_source_revision_words,
            inputs.cumulant_provenance_words,
        ),
    )
    losses, gvf, control, actions, words, applied = outputs
    return PrototypeCausalStateObjectiveTargetsScanResult(
        state=final_state,
        balanced_losses=losses,
        gvf_targets=gvf,
        control_value_targets=control,
        actions=actions,
        transaction_words=words,
        update_applied=applied,
    )


def save_prototype_causal_state_objective_targets_checkpoint(
    adapter: PrototypeCausalStateObjectiveTargets,
    state: PrototypeCausalStateObjectiveTargetsState,
    path: str | Path,
) -> None:
    """Persist the exact composition with strict versioned L0 metadata."""

    if type(adapter) is not PrototypeCausalStateObjectiveTargets:
        raise TypeError("adapter must be exact PrototypeCausalStateObjectiveTargets")
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("cannot checkpoint invalid Prototype causal target state")
    config = adapter.to_config()
    save_checkpoint(
        _checkpoint_storage_state(state),
        path,
        metadata={
            "schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA,
            "evidence_level": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS,
            "ownership": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP,
            "receipt_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA,
            "adapter_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": adapter.resource_budget(state).to_config(),
            "zero_sized_array_leaf_indices": _zero_sized_array_leaf_indices(
                state
            ),
        },
    )


def load_prototype_causal_state_objective_targets_checkpoint(
    path: str | Path,
) -> tuple[
    PrototypeCausalStateObjectiveTargets,
    PrototypeCausalStateObjectiveTargetsState,
]:
    """Restore only a canonical config- and resource-compatible checkpoint."""

    metadata = load_checkpoint_metadata(path)
    fields = _exact_manifest(
        metadata,
        {
            "schema",
            "evidence_level",
            "outcome_status",
            "ownership",
            "receipt_schema",
            "adapter_config",
            "config_sha256",
            "resource_budget",
            "zero_sized_array_leaf_indices",
        },
        label="Prototype causal target checkpoint",
    )
    fixed = {
        "schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA,
        "evidence_level": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL,
        "outcome_status": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS,
        "ownership": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP,
        "receipt_schema": PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA,
    }
    for name, expected in fixed.items():
        if not _exact_json_equal(fields[name], expected):
            raise ValueError(f"Prototype causal target checkpoint {name} is unsupported")
    config = fields["adapter_config"]
    if type(config) is not dict:
        raise TypeError("checkpoint adapter config must be an exact dict")
    if not _exact_json_equal(fields["config_sha256"], _canonical_digest(config)):
        raise ValueError("checkpoint adapter config digest differs")
    adapter = PrototypeCausalStateObjectiveTargets.from_config(config)
    if not _exact_json_equal(adapter.to_config(), config):
        raise ValueError("checkpoint adapter config is noncanonical")
    template = adapter.init(jr.key(0))
    expected_empty_leaves = _zero_sized_array_leaf_indices(template)
    if not _exact_json_equal(
        fields["zero_sized_array_leaf_indices"],
        expected_empty_leaves,
    ):
        raise ValueError("checkpoint empty-array storage manifest differs")
    if not _exact_json_equal(
        fields["resource_budget"],
        adapter.resource_budget(template).to_config(),
    ):
        raise ValueError("checkpoint resource budget differs")
    restored_storage, restored_metadata = load_checkpoint(
        _checkpoint_storage_state(template),
        path,
    )
    if not _exact_json_equal(restored_metadata, metadata):
        raise ValueError("checkpoint metadata changed between reads")
    if not _checkpoint_empty_array_storage_valid(restored_storage, template):
        raise ValueError("checkpoint empty-array storage sentinel differs")
    restored = _restore_checkpoint_empty_arrays(
        cast(PrototypeCausalStateObjectiveTargetsState, restored_storage),
        template,
    )
    adapter._require_state_contract(restored)
    if not bool(adapter.state_valid(restored)):
        raise ValueError("restored Prototype causal target state is invalid")
    adapter.resource_budget(restored)
    return adapter, restored


__all__ = [
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CHECKPOINT_SCHEMA",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_CONFIG_SCHEMA",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIFETIME_SEMANTICS",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_LIMITATIONS",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_MAX_TRANSITIONS",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OUTCOME_STATUS",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_OWNERSHIP",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RECEIPT_SCHEMA",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RESOURCE_SCHEMA",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS",
    "PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_STATE_SCHEMA",
    "PrototypeCausalDerivedTargetReceipt",
    "PrototypeCausalStateObjectiveTargets",
    "PrototypeCausalStateObjectiveTargetsResourceBudget",
    "PrototypeCausalStateObjectiveTargetsScanInputs",
    "PrototypeCausalStateObjectiveTargetsScanResult",
    "PrototypeCausalStateObjectiveTargetsStartResult",
    "PrototypeCausalStateObjectiveTargetsState",
    "PrototypeCausalStateObjectiveTargetsUpdateResult",
    "PrototypeCausalStateObjectiveTargetsWork",
    "load_prototype_causal_state_objective_targets_checkpoint",
    "measure_prototype_causal_state_objective_targets_state_nbytes",
    "run_prototype_causal_state_objective_targets_scan",
    "save_prototype_causal_state_objective_targets_checkpoint",
]
