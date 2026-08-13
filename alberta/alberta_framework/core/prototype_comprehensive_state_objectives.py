"""Transactional Prototype integration for comprehensive state objectives.

This opt-in WP3 adapter binds :class:`ComprehensiveStateObjectives` to the
exact primitive action dispatched by :class:`PrototypeAgent`.  The current
representation owner is the decision-time online-RTRL builder revision.  The
successor owner is the final/bootstrap observation reached by that action;
an autoreset observation is used only to cache the following decision.

The caller supplies cumulant, GVF continuation, control-value, and
selected-action-advantage targets in an explicit content-tagged receipt.  The
receipt is bound to the Prototype decision, action, observation event, builder
owner revisions, a sequential target identity, and caller-owned source and
provenance words.  Reward, termination, and next-observation supervision come
only from the authenticated real transition.  No privileged regime label is
accepted or inferred.

Both representation gradients are routed through their respective online
recurrent sensitivities, summed, globally clipped, and committed exactly once
to the already-advanced builder destination.  Prototype learner/RNG state,
all comprehensive heads, builder parameters, clocks, the next-action cache,
and the consumed target receipt commit together or roll back bit-for-bit.

An exact matching RTU generate-and-test lifecycle may own the same transaction.
Prototype first prepares only the real bootstrap/reset/restart recurrence.  The
adapter then commits both comprehensive gradients once, applies causal RTU
generate-and-test, scrubs every recycled axis from all comprehensive heads and
the statically supported linear OaK/STOMP consumers, and only then finalizes the
sole learner update and successor action selection.  Invalid internal deletion
scoring rejects this whole transaction.  The old bootstrap representation remains the
just-ended transition's target; the final builder internally emits the future
decision representation.  An active option defers replacement without losing
the real transition's ordinary learning.

This is an L0, nonpromoting, ``not_assessed`` mechanism.  Online recurrent
sensitivity after parameter updates is an explicit approximation; caller
targets and group masses are uncalibrated.  It establishes no retention,
Forager, control-benefit, Alberta Plan completion, or SOTA claim.
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
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.comprehensive_state_objectives import (
    COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL,
    COMPREHENSIVE_STATE_OBJECTIVES_HEADS,
    COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS,
    ComprehensiveStateObjectiveActionReceipt,
    ComprehensiveStateObjectiveCacheResult,
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesState,
    ComprehensiveStateObjectiveUpdateResult,
    measure_comprehensive_state_objectives_state_nbytes,
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

PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CONFIG_SCHEMA = (
    "alberta.prototype-comprehensive-state-objectives-config.v4"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_STATE_SCHEMA = (
    "alberta.prototype-comprehensive-state-objectives-state.v3"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA = (
    "alberta.prototype-comprehensive-state-objectives-target.v1"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA = (
    "alberta.prototype-comprehensive-state-objectives-checkpoint.v4"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RESOURCE_SCHEMA = (
    "alberta.prototype-comprehensive-state-objectives-resource.v4"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL = "L0"
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS = "not_assessed"
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIFETIME_SEMANTICS = (
    "exact-uint64-and-rtu-uint32-fail-stop"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP = (
    "exact-prototype-decision-and-action; bit-exact-current-representation; "
    "final-bootstrap-successor-before-autoreset; decision-time-online-builder-owner; "
    "content-tagged-caller-target-provenance; one-source-bound-builder-commit; "
    "pre-update-frozen-head-whole-complex-RTU-causal-deletion-owner; "
    "atomic-recycled-axis-objective-head-scrub"
)
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIMITATIONS = (
    "explicit-opt-in-online-RTRL-builder-only",
    "online-recurrent-sensitivity-approximation-after-parameter-updates",
    "caller-targets-and-objective-group-masses-uncalibrated",
    "content-tag-detects-mutation-but-does-not-authenticate-caller-source",
    "no-privileged-regime-labels-or-inferred-control-targets",
    "no-general-feature-lifecycle-or-external-concurrent-builder-learning",
    "live-RTU-replacement-requires-statically-guarded-linear-OaK-STOMP-envelope",
    "live-RTU-causal-deletion-is-prequential-L0-not-held-out-outcome-evidence",
    "no-retention-control-forager-alberta-plan-or-sota-evidence",
)
# Every accepted transition also reserves the next comprehensive action-cache
# identity.  The portable armed-continuation bound is therefore one below the
# uint64 maximum.
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_MAX_TRANSITIONS = 2**64 - 2
PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RTU_MAX_TRANSITIONS = 2**32 - 1

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_PROVENANCE_WIDTH = 4
_TARGET_PAYLOAD_WIDTH = 4
_TARGET_TAG_WIDTH = 4


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


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _require_words(value: Any, *, label: str, width: int = 2) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(width,),
        dtype=jnp.dtype(jnp.uint32),
    )


def _require_float32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.float32))


def _require_int32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.int32))


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.bool_))


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


def _words_not_earlier(candidate: Array, reference: Array) -> Bool[Array, ""]:
    return (candidate[0] > reference[0]) | (
        (candidate[0] == reference[0]) & (candidate[1] >= reference[1])
    )


def _float32_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _target_payload_words(
    cumulant: Array,
    continuation: Array,
    control_value_target: Array,
    selected_action_advantage_target: Array,
) -> UInt[Array, " 4"]:
    values = jnp.stack(
        (
            cumulant,
            continuation,
            control_value_target,
            selected_action_advantage_target,
        )
    ).astype(jnp.float32)
    return jax.lax.bitcast_convert_type(values, jnp.uint32)


def _rotate_left(value: Array, distance: Array) -> Array:
    right = (jnp.asarray(32, dtype=jnp.uint32) - distance) & jnp.uint32(31)
    return jnp.asarray((value << distance) | (value >> right), dtype=jnp.uint32)


def _target_content_tag(
    *,
    prototype_decision_id: Array,
    action: Array,
    observation_event_words: Array,
    builder_step_words: Array,
    builder_update_words: Array,
    target_identity_words: Array,
    source_revision_words: Array,
    provenance_words: Array,
    payload_words: Array,
) -> UInt[Array, " 4"]:
    """Return a deterministic integrity tag over every receipt owner word."""

    action_word = jax.lax.bitcast_convert_type(action, jnp.uint32)[None]
    words = jnp.concatenate(
        (
            prototype_decision_id,
            action_word,
            observation_event_words,
            builder_step_words,
            builder_update_words,
            target_identity_words,
            source_revision_words,
            provenance_words,
            payload_words,
        )
    ).astype(jnp.uint32)
    indices = jnp.arange(words.shape[0], dtype=jnp.uint32)
    distances = (indices % jnp.uint32(31)) + jnp.uint32(1)
    mixed = _rotate_left(
        words ^ (indices * jnp.uint32(0x9E3779B9)),
        distances,
    )
    tag0 = jnp.bitwise_xor.reduce(mixed, axis=0)
    tag1 = jnp.sum(mixed * jnp.uint32(0x85EBCA6B), dtype=jnp.uint32)
    tag2 = jnp.bitwise_xor.reduce(mixed * (indices + jnp.uint32(0xC2B2AE35)), axis=0)
    tag3 = jnp.sum(
        _rotate_left(mixed, ((indices * jnp.uint32(7)) % jnp.uint32(31)) + 1),
        dtype=jnp.uint32,
    )
    return jnp.stack((tag0, tag1, tag2, tag3)).astype(jnp.uint32)


def _exact_tree_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Compare one fixed-shape PyTree by exact typed leaf content."""

    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if (
        cast(object, left_structure) != cast(object, right_structure)
        or len(left_leaves) != len(right_leaves)
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
            equal = equal & _float32_bits_equal(left_array, right_array)
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _builder_states_equal(
    left: OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
    right: OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState,
) -> Bool[Array, ""]:
    return _exact_tree_equal(left, right)


def _safe_clip_parameter_gradient(value: Array, limit: float) -> tuple[Array, Array, Array]:
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
    valid = finite & jnp.all(jnp.isfinite(clipped))
    return valid, clipped, norm


def _state_array_nbytes(state: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(state):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


@chex.dataclass(frozen=True)
class PrototypeComprehensiveTargetReceipt:
    """Caller-owned targets and their exact decision/source provenance."""

    cumulant: Float[Array, ""]
    gvf_continuation: Float[Array, ""]
    control_value_target: Float[Array, ""]
    selected_action_advantage_target: Float[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    action: Int[Array, ""]
    observation_event_words: UInt[Array, " 2"]
    builder_step_words: UInt[Array, " 2"]
    builder_update_words: UInt[Array, " 2"]
    target_identity_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    provenance_words: UInt[Array, " 4"]
    payload_words: UInt[Array, " 4"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class PrototypeComprehensiveObjectivesState:
    """Atomic Prototype, objective, builder-owner, and target clocks."""

    prototype_state: PrototypeAgentState
    objectives_state: ComprehensiveStateObjectivesState
    pending_prototype_decision_id: UInt[Array, " 4"]
    pending_builder_step_words: UInt[Array, " 2"]
    pending_builder_update_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    transaction_words: UInt[Array, " 2"]
    target_receipt_words: UInt[Array, " 2"]
    last_target_prototype_decision_id: UInt[Array, " 4"]
    last_target_action: Int[Array, ""]
    last_target_observation_event_words: UInt[Array, " 2"]
    last_target_builder_step_words: UInt[Array, " 2"]
    last_target_builder_update_words: UInt[Array, " 2"]
    last_target_source_revision_words: UInt[Array, " 2"]
    last_target_provenance_words: UInt[Array, " 4"]
    last_target_payload_words: UInt[Array, " 4"]
    last_target_content_tag_words: UInt[Array, " 4"]
    rtu_generate_and_test_state: RTUGenerateAndTestState | None = None


@chex.dataclass(frozen=True)
class PrototypeComprehensiveObjectivesStartResult:
    """Atomic result of priming and binding the first dispatch."""

    state: PrototypeComprehensiveObjectivesState
    objective_cache: ComprehensiveStateObjectiveCacheResult
    source_state_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    start_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeComprehensiveObjectivesUpdateResult:
    """One attempted all-or-nothing Prototype/comprehensive transaction."""

    state: PrototypeComprehensiveObjectivesState
    action: Int[Array, ""]
    prototype_transition: PrototypeTransitionDiagnostics
    objective_update: ComprehensiveStateObjectiveUpdateResult
    next_objective_cache: ComprehensiveStateObjectiveCacheResult
    bootstrap_builder_transition: (
        OnlineGatedStateBuilderTransitionResult
        | RecurrentTraceUnitStateBuilderTransitionResult
    )
    builder_learning: StateBuilderLearningDiagnostics
    rtu_generate_and_test: RTUGenerateAndTestCommitResult | None
    rtu_advance_receipt: RTUGenerateAndTestAdvanceReceipt | None
    target_receipt: PrototypeComprehensiveTargetReceipt
    bootstrap_representation: Float[Array, " representation"]
    combined_raw_parameter_gradient_norm: Float[Array, ""]
    pre_transaction_words: UInt[Array, " 2"]
    post_transaction_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    transition_identity_matches: Bool[Array, ""]
    target_owner_matches: Bool[Array, ""]
    target_payload_matches: Bool[Array, ""]
    target_content_tag_matches: Bool[Array, ""]
    target_provenance_valid: Bool[Array, ""]
    target_source_revision_valid: Bool[Array, ""]
    target_identity_capacity_available: Bool[Array, ""]
    bootstrap_event_capacity_available: Bool[Array, ""]
    bootstrap_transition_applied: Bool[Array, ""]
    prototype_transaction_applied: Bool[Array, ""]
    objective_transaction_applied: Bool[Array, ""]
    builder_sources_match: Bool[Array, ""]
    builder_destination_matches: Bool[Array, ""]
    builder_transaction_applied: Bool[Array, ""]
    rtu_observation_proposal_valid: Bool[Array, ""]
    rtu_lifecycle_source_matches: Bool[Array, ""]
    rtu_observation_transaction_applied: Bool[Array, ""]
    rtu_causal_deletion_evidence_attempted: Bool[Array, ""]
    rtu_causal_deletion_evidence_valid: Bool[Array, ""]
    rtu_replacement_cache_safe: Bool[Array, ""]
    rtu_replacement_requires_pre_action_hook: Bool[Array, ""]
    next_cache_required: Bool[Array, ""]
    next_cache_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    target_receipt_committed: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeComprehensiveObjectivesResourceBudget:
    """Exact persistent allocation and fixed per-transition work bounds.

    Builder commit counts include every source-level
    ``commit_learning_update`` evaluation, including pure preflight and
    commit-time integrity recomputation.  They do not mean that each
    evaluation advances persistent builder state.
    """

    schema: str
    prototype_state_nbytes: int
    objectives_state_nbytes: int
    rtu_generate_and_test_state_nbytes: int
    adapter_metadata_nbytes: int
    total_state_nbytes: int
    max_prototype_updates_per_transition: int
    max_objective_parameter_head_updates_per_transition: int
    max_causal_deletion_units_scored_per_transition: int
    max_causal_deletion_frozen_head_evaluations_per_transition: int
    max_builder_proposals_per_transition: int
    max_builder_commits_per_transition: int
    max_rtu_generate_and_test_proposals_per_transition: int
    max_rtu_generate_and_test_commits_per_transition: int
    max_next_action_cache_writes_per_transition: int
    max_target_receipts_consumed_per_transition: int
    max_accepted_transitions: int
    persistent_bytes_scope: str
    diagnostic_bytes_scope: str
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypeComprehensiveObjectivesScanResult:
    """Fixed-shape trace for explicit real transitions and target receipts."""

    state: PrototypeComprehensiveObjectivesState
    balanced_losses: Float[Array, " steps"]
    actions: Int[Array, " steps"]
    target_identity_words: UInt[Array, "steps 2"]
    transaction_words: UInt[Array, "steps 2"]
    update_applied: Bool[Array, " steps"]


def measure_prototype_comprehensive_objectives_state_nbytes(
    state: PrototypeComprehensiveObjectivesState,
) -> int:
    """Measure every persistent JAX-array leaf in the composition."""

    if type(state) is not PrototypeComprehensiveObjectivesState:
        raise TypeError("state must be an exact PrototypeComprehensiveObjectivesState")
    return _state_array_nbytes(state)


class PrototypeComprehensiveStateObjectives:
    """Opt-in transactional adapter for Prototype and comprehensive heads.

    In the RTU lane this composition owns the lifecycle source and constructs
    both downstream objective gradients and the source-bound ordinary learning
    proposal internally.  Exact lifecycle-source matching plus Prototype's
    independent RTU recomputation closes the lower-level finalizer's three
    external-authority boundaries.
    """

    def __init__(
        self,
        prototype: PrototypeAgent,
        objectives: ComprehensiveStateObjectives,
        rtu_generate_and_test: RTUGenerateAndTest | None = None,
    ) -> None:
        if type(prototype) is not PrototypeAgent:
            raise TypeError("prototype must be an exact PrototypeAgent")
        if type(objectives) is not ComprehensiveStateObjectives:
            raise TypeError("objectives must be an exact ComprehensiveStateObjectives")
        if (
            rtu_generate_and_test is not None
            and type(rtu_generate_and_test) is not RTUGenerateAndTest
        ):
            raise TypeError("rtu_generate_and_test must be an exact RTUGenerateAndTest")
        config = prototype.config
        if config.learning_value_router is not None:
            raise ValueError("adapter does not compose with learning_value_router")
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
        if config.prototype_feature_lifecycle is not None:
            raise ValueError("adapter does not compose with prototype_feature_lifecycle")
        if config.learn_state_builder_from_world_model:
            raise ValueError("adapter requires Prototype world-model builder learning disabled")
        if config.representation_gradient_mixer is not None:
            raise ValueError("adapter requires Prototype representation gradient mixing disabled")
        if config.auto_curate_every != 0:
            raise ValueError("adapter requires auto_curate_every == 0")
        builder: Any = prototype.state_builder
        builder_config = cast(
            OnlineGatedStateBuilderConfig
            | LearnableGRUStateBuilderConfig
            | RecurrentTraceUnitStateBuilderConfig,
            config.state_builder,
        )
        if objectives.config.representation_dim != builder.feature_dim():
            raise ValueError("objective representation_dim must match the builder feature_dim")
        if objectives.config.observation_target_dim != builder_config.observation_dim:
            raise ValueError(
                "objective observation_target_dim must match Prototype raw observation_dim"
            )
        if objectives.config.n_actions != config.oak.n_primitive_actions:
            raise ValueError("objective n_actions must match Prototype primitive actions")
        if COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL != "L0":
            raise RuntimeError("comprehensive objectives must remain an L0 mechanism")
        if COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS != "not_assessed":
            raise RuntimeError("comprehensive objectives must remain not_assessed")
        if rtu_generate_and_test is not None:
            if type(prototype.state_builder) is not RecurrentTraceUnitStateBuilder:
                raise ValueError(
                    "rtu_generate_and_test requires an exact RTU Prototype builder"
                )
            if type(config.state_builder) is not RecurrentTraceUnitStateBuilderConfig:
                raise ValueError(
                    "rtu_generate_and_test requires an exact RTU Prototype config"
                )
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
                "experiential_memory_advantage_gate": (
                    config.experiential_memory_advantage_gate
                ),
                "gru_perception": config.gru_perception,
                "gradient_joy": config.gradient_joy,
                "option_search_control": config.option_search_control,
                "prototype_feature_lifecycle": (
                    config.prototype_feature_lifecycle
                ),
                "prototype_feature_utility": config.prototype_feature_utility,
                "prototype_feature_utility_curation": (
                    config.prototype_feature_utility_curation
                ),
            }
            enabled_unsupported = sorted(
                name
                for name, component in unsupported_components.items()
                if component is not None
            )
            if enabled_unsupported:
                raise ValueError(
                    "live RTU replacement does not compose with: "
                    + ", ".join(enabled_unsupported)
                )
            if config.n_dreams_per_step != 0:
                raise ValueError(
                    "live RTU replacement requires n_dreams_per_step == 0"
                )
            if config.oak.stomp.base_hidden_sizes:
                raise ValueError(
                    "live RTU replacement requires a linear STOMP base learner"
                )
            if config.oak.stomp.option_planning_backups_per_step != 0:
                raise ValueError(
                    "live RTU replacement requires option planning disabled"
                )
            if not config.state_builder.include_raw_observation:
                raise ValueError(
                    "live RTU replacement requires raw observation features"
                )
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
        self._objectives = objectives
        self._builder = cast(
            OnlineGatedStateBuilder | RecurrentTraceUnitStateBuilder,
            builder,
        )
        self._rtu_generate_and_test = rtu_generate_and_test

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def objectives(self) -> ComprehensiveStateObjectives:
        return self._objectives

    @property
    def builder(
        self,
    ) -> OnlineGatedStateBuilder | RecurrentTraceUnitStateBuilder:
        return self._builder

    @property
    def rtu_generate_and_test(self) -> RTUGenerateAndTest | None:
        """Return the explicit RTU lifecycle, if this adapter owns one."""

        return self._rtu_generate_and_test

    @property
    def max_accepted_transitions(self) -> int:
        """Return the strictest persistent-counter lifetime bound."""

        if self._rtu_generate_and_test is not None:
            return PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RTU_MAX_TRANSITIONS
        return PROTOTYPE_COMPREHENSIVE_OBJECTIVES_MAX_TRANSITIONS

    def to_config(self) -> dict[str, Any]:
        payload = {
            "type": "PrototypeComprehensiveStateObjectives",
            "schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_STATE_SCHEMA,
            "target_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA,
            "checkpoint_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP,
            "lifetime_semantics": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIFETIME_SEMANTICS,
            "max_transitions": self.max_accepted_transitions,
            "limitations": list(PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIMITATIONS),
            "activation": "explicit-adapter-only",
            "prototype_config": self._prototype.to_config(),
            "objectives_config": self._objectives.to_config(),
        }
        if self._rtu_generate_and_test is not None:
            payload["rtu_generate_and_test_config"] = (
                self._rtu_generate_and_test.to_config()
            )
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeComprehensiveStateObjectives:
        expected = {
            "type",
            "schema",
            "state_schema",
            "target_schema",
            "checkpoint_schema",
            "resource_schema",
            "evidence_level",
            "outcome_status",
            "ownership",
            "lifetime_semantics",
            "max_transitions",
            "limitations",
            "activation",
            "prototype_config",
            "objectives_config",
        }
        if "rtu_generate_and_test_config" in payload:
            expected.add("rtu_generate_and_test_config")
        fields = _exact_manifest(
            payload,
            expected,
            label="prototype comprehensive objectives config",
        )
        fixed = {
            "type": "PrototypeComprehensiveStateObjectives",
            "schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_STATE_SCHEMA,
            "target_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA,
            "checkpoint_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP,
            "lifetime_semantics": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIFETIME_SEMANTICS,
            "limitations": list(PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIMITATIONS),
            "activation": "explicit-adapter-only",
        }
        for name, expected_value in fixed.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"prototype comprehensive objectives {name} is unsupported")
        declared_max_transitions = fields.pop("max_transitions")
        prototype_config = fields["prototype_config"]
        objectives_config = fields["objectives_config"]
        if type(prototype_config) is not dict or type(objectives_config) is not dict:
            raise TypeError("nested Prototype and objectives configs must be exact dicts")
        rtu_config_supplied = "rtu_generate_and_test_config" in fields
        rtu_config = fields.get("rtu_generate_and_test_config")
        if rtu_config_supplied and type(rtu_config) is not dict:
            raise TypeError("RTU generate-and-test config must be an exact dict")
        restored = cls(
            PrototypeAgent.from_config(prototype_config),
            ComprehensiveStateObjectives.from_config(objectives_config),
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
            raise ValueError(
                "prototype comprehensive objectives max_transitions is unsupported"
            )
        return restored

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeComprehensiveObjectivesState:
        """Initialize both components without changing ordinary Prototype use."""

        _require_threefry_key(key, label="key")
        if self._rtu_generate_and_test is None:
            prototype_key, objectives_key = jr.split(key)
            rtu_state = None
        else:
            prototype_key, objectives_key, rtu_key = jr.split(key, 3)
            rtu_state = self._rtu_generate_and_test.init(rtu_key)
        zero_two = jnp.zeros((2,), dtype=jnp.uint32)
        zero_four = jnp.zeros((4,), dtype=jnp.uint32)
        return PrototypeComprehensiveObjectivesState(  # type: ignore[call-arg]
            prototype_state=self._prototype.init(
                prototype_key,
                lifecycle_id=lifecycle_id,
            ),
            objectives_state=self._objectives.init(objectives_key),
            rtu_generate_and_test_state=rtu_state,
            pending_prototype_decision_id=zero_four,
            pending_builder_step_words=zero_two,
            pending_builder_update_words=zero_two,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            transaction_words=zero_two,
            target_receipt_words=zero_two,
            last_target_prototype_decision_id=zero_four,
            last_target_action=jnp.asarray(-1, dtype=jnp.int32),
            last_target_observation_event_words=zero_two,
            last_target_builder_step_words=zero_two,
            last_target_builder_update_words=zero_two,
            last_target_source_revision_words=zero_two,
            last_target_provenance_words=zero_four,
            last_target_payload_words=zero_four,
            last_target_content_tag_words=zero_four,
        )

    def _require_state_contract(
        self,
        state: PrototypeComprehensiveObjectivesState,
    ) -> None:
        if type(state) is not PrototypeComprehensiveObjectivesState:
            raise TypeError("state must be an exact PrototypeComprehensiveObjectivesState")
        if type(state.prototype_state) is not PrototypeAgentState:
            raise TypeError("state.prototype_state must be an exact PrototypeAgentState")
        if type(state.objectives_state) is not ComprehensiveStateObjectivesState:
            raise TypeError(
                "state.objectives_state must be an exact ComprehensiveStateObjectivesState"
            )
        if self._rtu_generate_and_test is None:
            if state.rtu_generate_and_test_state is not None:
                raise TypeError("base adapter state cannot carry an RTU lifecycle")
        else:
            if type(state.rtu_generate_and_test_state) is not RTUGenerateAndTestState:
                raise TypeError(
                    "RTU-enabled adapter state must carry exact lifecycle state"
                )
            self._rtu_generate_and_test.state_valid(
                state.rtu_generate_and_test_state
            )
        expected_builder_state_type = (
            RecurrentTraceUnitStateBuilderState
            if isinstance(self._builder, RecurrentTraceUnitStateBuilder)
            else OnlineGatedStateBuilderState
        )
        if type(state.prototype_state.state_builder_state) is not expected_builder_state_type:
            raise TypeError("Prototype builder state does not match the configured RTRL builder")
        _require_words(
            state.pending_prototype_decision_id,
            label="pending_prototype_decision_id",
            width=4,
        )
        _require_words(state.pending_builder_step_words, label="pending_builder_step_words")
        _require_words(
            state.pending_builder_update_words,
            label="pending_builder_update_words",
        )
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        _require_words(state.transaction_words, label="transaction_words")
        _require_words(state.target_receipt_words, label="target_receipt_words")
        _require_words(
            state.last_target_prototype_decision_id,
            label="last_target_prototype_decision_id",
            width=4,
        )
        _require_int32_scalar(state.last_target_action, label="last_target_action")
        _require_words(
            state.last_target_observation_event_words,
            label="last_target_observation_event_words",
        )
        _require_words(
            state.last_target_builder_step_words,
            label="last_target_builder_step_words",
        )
        _require_words(
            state.last_target_builder_update_words,
            label="last_target_builder_update_words",
        )
        _require_words(
            state.last_target_source_revision_words,
            label="last_target_source_revision_words",
        )
        _require_words(
            state.last_target_provenance_words,
            label="last_target_provenance_words",
            width=_PROVENANCE_WIDTH,
        )
        _require_words(
            state.last_target_payload_words,
            label="last_target_payload_words",
            width=_TARGET_PAYLOAD_WIDTH,
        )
        _require_words(
            state.last_target_content_tag_words,
            label="last_target_content_tag_words",
            width=_TARGET_TAG_WIDTH,
        )
        self._objectives.state_valid(state.objectives_state)
        cast(Any, self._builder).state_valid(
            state.prototype_state.state_builder_state
        )

    def _dynamic_state_valid(
        self,
        state: PrototypeComprehensiveObjectivesState,
    ) -> Bool[Array, ""]:
        prototype_state = state.prototype_state
        objective_state = state.objectives_state
        builder_state = cast(
            OnlineGatedStateBuilderState,
            prototype_state.state_builder_state,
        )
        rtu_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        rtu_global_lifetime_valid = jnp.asarray(True, dtype=jnp.bool_)
        objective_replacement_scrub_valid = jnp.asarray(True, dtype=jnp.bool_)
        consumer_replacement_scrub_valid = jnp.asarray(True, dtype=jnp.bool_)
        replacement_event_words = jnp.zeros((2,), dtype=jnp.uint32)
        most_recent_replacement = jnp.asarray(False, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            rtu_state = cast(
                RTUGenerateAndTestState,
                state.rtu_generate_and_test_state,
            )
            rtu_state_valid = (
                self._rtu_generate_and_test.state_valid(rtu_state)
                & jnp.all(rtu_state.observation_words == state.transaction_words)
            )
            replacement_event_words = rtu_state.replacement_event_words
            most_recent_replacement = jnp.any(rtu_state.last_replaced_mask)
            rtu_global_lifetime_valid = _rtu_global_lifetime_state_valid(
                state.transaction_words
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
            reset_mask = jnp.concatenate(
                (
                    jnp.zeros((raw_dim,), dtype=jnp.bool_),
                    rtu_state.last_replaced_mask,
                    rtu_state.last_replaced_mask,
                )
            )

            objective_replacement_scrub_valid = jnp.all(
                jnp.stack(
                    tuple(
                        _selected_float32_axes_are_positive_zero(
                            value,
                            reset_mask,
                        )
                        for value in (
                            objective_state.observation_weights,
                            objective_state.latent_weights,
                            objective_state.reward_weights,
                            objective_state.termination_weights,
                            objective_state.gvf_weights,
                            objective_state.value_weights,
                            objective_state.advantage_weights,
                            objective_state.inverse_current_weights,
                            objective_state.inverse_next_weights,
                            objective_state.pending_representation,
                        )
                    )
                )
            )
            consumer_replacement_scrub_valid = (
                _rtu_builder_replacement_scrub_valid(
                    cast(RecurrentTraceUnitStateBuilderState, builder_state),
                    rtu_state.last_replaced_mask,
                    event_dim=rtu_builder_config.event_dim(),
                )
                & _linear_stomp_replacement_scrub_valid(
                    prototype_state.oak_state.stomp_state,
                    reset_mask,
                )
            )
        expected_builder_revision, builder_revision_capacity = _add_word_pairs(
            state.transaction_words,
            replacement_event_words,
        )
        owner_successor, owner_has_successor = _increment_words(state.pending_builder_update_words)
        initial_decision_owner = jnp.all(state.transaction_words == 0) & jnp.all(
            state.pending_builder_update_words == builder_state.update_words
        )
        post_update_decision_owner = (
            jnp.any(state.transaction_words != 0)
            & jnp.where(
                most_recent_replacement,
                jnp.all(
                    state.pending_builder_update_words
                    == builder_state.update_words
                ),
                owner_has_successor
                & jnp.all(owner_successor == builder_state.update_words),
            )
        )
        pending_filled = (
            prototype_state.started
            & objective_state.pending_valid
            & jnp.array_equal(
                state.pending_prototype_decision_id,
                prototype_state.current_decision_id,
            )
            & _float32_bits_equal(
                objective_state.pending_representation,
                prototype_state.current_representation,
            )
            & (objective_state.pending_action == prototype_state.current_action)
            & jnp.all(
                objective_state.pending_representation_revision_words
                == prototype_state.observation_event_words
            )
            & jnp.all(state.pending_builder_step_words == builder_state.step_words)
            & (initial_decision_owner | post_update_decision_owner)
        )
        pending_empty = (
            (~prototype_state.started)
            & (~objective_state.pending_valid)
            & jnp.all(state.pending_prototype_decision_id == 0)
            & jnp.all(state.pending_builder_step_words == 0)
            & jnp.all(state.pending_builder_update_words == 0)
        )
        no_target_committed = (
            jnp.all(state.transaction_words == 0)
            & jnp.all(state.last_target_prototype_decision_id == 0)
            & (state.last_target_action == -1)
            & jnp.all(state.last_target_observation_event_words == 0)
            & jnp.all(state.last_target_builder_step_words == 0)
            & jnp.all(state.last_target_builder_update_words == 0)
            & jnp.all(state.last_target_source_revision_words == 0)
            & jnp.all(state.last_target_provenance_words == 0)
            & jnp.all(state.last_target_payload_words == 0)
            & jnp.all(state.last_target_content_tag_words == 0)
        )
        expected_last_target_tag = _target_content_tag(
            prototype_decision_id=state.last_target_prototype_decision_id,
            action=state.last_target_action,
            observation_event_words=state.last_target_observation_event_words,
            builder_step_words=state.last_target_builder_step_words,
            builder_update_words=state.last_target_builder_update_words,
            target_identity_words=state.target_receipt_words,
            source_revision_words=state.last_target_source_revision_words,
            provenance_words=state.last_target_provenance_words,
            payload_words=state.last_target_payload_words,
        )
        committed_target_record = (
            jnp.any(state.transaction_words != 0)
            & (state.last_target_action >= 0)
            & (state.last_target_action < self._objectives.config.n_actions)
            & jnp.any(state.last_target_provenance_words != 0)
            & jnp.all(state.last_target_content_tag_words == expected_last_target_tag)
        )
        builder_state_valid = cast(
            Array,
            cast(Any, self._builder).state_valid(builder_state),
        )
        return (
            self._prototype.validate_state(prototype_state)
            & self._objectives.state_valid(objective_state)
            & builder_state_valid
            & rtu_state_valid
            & rtu_global_lifetime_valid
            & objective_replacement_scrub_valid
            & consumer_replacement_scrub_valid
            & (state.pending_valid == objective_state.pending_valid)
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
            & jnp.all(state.transaction_words == prototype_state.step_words)
            & jnp.all(state.transaction_words == objective_state.update_words)
            & builder_revision_capacity
            & jnp.all(expected_builder_revision == builder_state.update_words)
            & jnp.all(state.transaction_words == state.target_receipt_words)
            & (no_target_committed | committed_target_record)
        )

    def state_valid(
        self,
        state: PrototypeComprehensiveObjectivesState,
    ) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def _objective_receipt(
        self,
        state: ComprehensiveStateObjectivesState,
    ) -> ComprehensiveStateObjectiveActionReceipt:
        return ComprehensiveStateObjectiveActionReceipt(  # type: ignore[call-arg]
            representation=state.pending_representation,
            action=state.pending_action,
            representation_revision_words=state.pending_representation_revision_words,
            action_identity_words=state.pending_action_identity_words,
        )

    def make_target_receipt(
        self,
        state: PrototypeComprehensiveObjectivesState,
        *,
        cumulant: Array,
        gvf_continuation: Array,
        control_value_target: Array,
        selected_action_advantage_target: Array,
        source_revision_words: Array,
        provenance_words: Array,
    ) -> PrototypeComprehensiveTargetReceipt:
        """Bind caller targets to the exact currently dispatched decision.

        This function does not mutate ``state``.  A receipt is consumed only
        if the complete outer transaction commits, so the same receipt remains
        retryable after any refusal.  Its deterministic tag detects later
        mutation; caller source authority remains an external responsibility.
        """

        self._require_state_contract(state)
        cumulant = _require_float32_scalar(cumulant, label="cumulant")
        gvf_continuation = _require_float32_scalar(
            gvf_continuation,
            label="gvf_continuation",
        )
        control_value_target = _require_float32_scalar(
            control_value_target,
            label="control_value_target",
        )
        selected_action_advantage_target = _require_float32_scalar(
            selected_action_advantage_target,
            label="selected_action_advantage_target",
        )
        source_revision_words = _require_words(
            source_revision_words,
            label="source_revision_words",
        )
        provenance_words = _require_words(
            provenance_words,
            label="provenance_words",
            width=_PROVENANCE_WIDTH,
        )
        target_identity_words, _ = _increment_words(state.target_receipt_words)
        payload_words = _target_payload_words(
            cumulant,
            gvf_continuation,
            control_value_target,
            selected_action_advantage_target,
        )
        prototype_state = state.prototype_state
        content_tag_words = _target_content_tag(
            prototype_decision_id=state.pending_prototype_decision_id,
            action=prototype_state.current_action,
            observation_event_words=prototype_state.observation_event_words,
            builder_step_words=state.pending_builder_step_words,
            builder_update_words=state.pending_builder_update_words,
            target_identity_words=target_identity_words,
            source_revision_words=source_revision_words,
            provenance_words=provenance_words,
            payload_words=payload_words,
        )
        return PrototypeComprehensiveTargetReceipt(  # type: ignore[call-arg]
            cumulant=cumulant,
            gvf_continuation=gvf_continuation,
            control_value_target=control_value_target,
            selected_action_advantage_target=selected_action_advantage_target,
            prototype_decision_id=state.pending_prototype_decision_id,
            action=prototype_state.current_action,
            observation_event_words=prototype_state.observation_event_words,
            builder_step_words=state.pending_builder_step_words,
            builder_update_words=state.pending_builder_update_words,
            target_identity_words=target_identity_words,
            source_revision_words=source_revision_words,
            provenance_words=provenance_words,
            payload_words=payload_words,
            content_tag_words=content_tag_words,
        )

    def _require_target_receipt_contract(
        self,
        receipt: PrototypeComprehensiveTargetReceipt,
    ) -> None:
        if type(receipt) is not PrototypeComprehensiveTargetReceipt:
            raise TypeError("target_receipt must be an exact PrototypeComprehensiveTargetReceipt")
        for label in (
            "cumulant",
            "gvf_continuation",
            "control_value_target",
            "selected_action_advantage_target",
        ):
            _require_float32_scalar(getattr(receipt, label), label=f"target_receipt.{label}")
        _require_words(
            receipt.prototype_decision_id,
            label="target_receipt.prototype_decision_id",
            width=4,
        )
        _require_int32_scalar(receipt.action, label="target_receipt.action")
        for label in (
            "observation_event_words",
            "builder_step_words",
            "builder_update_words",
            "target_identity_words",
            "source_revision_words",
        ):
            _require_words(getattr(receipt, label), label=f"target_receipt.{label}")
        _require_words(
            receipt.provenance_words,
            label="target_receipt.provenance_words",
            width=_PROVENANCE_WIDTH,
        )
        _require_words(
            receipt.payload_words,
            label="target_receipt.payload_words",
            width=_TARGET_PAYLOAD_WIDTH,
        )
        _require_words(
            receipt.content_tag_words,
            label="target_receipt.content_tag_words",
            width=_TARGET_TAG_WIDTH,
        )

    def start(
        self,
        state: PrototypeComprehensiveObjectivesState,
        initial_observation: Array,
    ) -> PrototypeComprehensiveObjectivesStartResult:
        """Prime Prototype and cache the exact first comprehensive owner."""

        self._require_state_contract(state)
        return cast(
            PrototypeComprehensiveObjectivesStartResult,
            self._start_jit(state, initial_observation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _start_jit(
        self,
        state: PrototypeComprehensiveObjectivesState,
        initial_observation: Array,
    ) -> PrototypeComprehensiveObjectivesStartResult:
        source_valid = self._dynamic_state_valid(state) & (~state.pending_valid)
        candidate_prototype = self._prototype.start(
            state.prototype_state,
            initial_observation,
        )
        builder_state = cast(
            OnlineGatedStateBuilderState,
            candidate_prototype.state_builder_state,
        )
        cached = self._objectives.cache_action(
            state.objectives_state,
            candidate_prototype.current_representation,
            candidate_prototype.current_action,
            candidate_prototype.observation_event_words,
        )
        candidate = PrototypeComprehensiveObjectivesState(  # type: ignore[call-arg]
            prototype_state=candidate_prototype,
            objectives_state=cached.state,
            rtu_generate_and_test_state=state.rtu_generate_and_test_state,
            pending_prototype_decision_id=candidate_prototype.current_decision_id,
            pending_builder_step_words=builder_state.step_words,
            pending_builder_update_words=builder_state.update_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            transaction_words=state.transaction_words,
            target_receipt_words=state.target_receipt_words,
            last_target_prototype_decision_id=state.last_target_prototype_decision_id,
            last_target_action=state.last_target_action,
            last_target_observation_event_words=(state.last_target_observation_event_words),
            last_target_builder_step_words=state.last_target_builder_step_words,
            last_target_builder_update_words=state.last_target_builder_update_words,
            last_target_source_revision_words=state.last_target_source_revision_words,
            last_target_provenance_words=state.last_target_provenance_words,
            last_target_payload_words=state.last_target_payload_words,
            last_target_content_tag_words=state.last_target_content_tag_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            source_valid & candidate_prototype.started & cached.cache_applied & candidate_valid
        )
        final_state = cast(
            PrototypeComprehensiveObjectivesState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        return PrototypeComprehensiveObjectivesStartResult(  # type: ignore[call-arg]
            state=final_state,
            objective_cache=cached,
            source_state_valid=source_valid,
            candidate_state_valid=candidate_valid,
            start_applied=applied,
        )

    def update_transition(
        self,
        state: PrototypeComprehensiveObjectivesState,
        transition: PrototypeTransition,
        target_receipt: PrototypeComprehensiveTargetReceipt,
    ) -> PrototypeComprehensiveObjectivesUpdateResult:
        """Consume one real transition and exact caller-owned target receipt."""

        self._require_state_contract(state)
        if type(transition) is not PrototypeTransition:
            raise TypeError("transition must be an exact PrototypeTransition")
        self._require_target_receipt_contract(target_receipt)
        return cast(
            PrototypeComprehensiveObjectivesUpdateResult,
            self._update_transition_jit(state, transition, target_receipt),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_transition_jit(
        self,
        state: PrototypeComprehensiveObjectivesState,
        transition: PrototypeTransition,
        target_receipt: PrototypeComprehensiveTargetReceipt,
    ) -> PrototypeComprehensiveObjectivesUpdateResult:
        source_valid = self._dynamic_state_valid(state) & state.pending_valid
        rtu_global_lifetime_capacity = jnp.asarray(True, dtype=jnp.bool_)
        if self._rtu_generate_and_test is not None:
            rtu_global_lifetime_capacity = _rtu_global_lifetime_capacity(
                state.transaction_words
            )
        prototype_state = state.prototype_state
        source_builder: Any = prototype_state.state_builder_state
        objective_receipt = self._objective_receipt(state.objectives_state)
        transition_identity_matches = (
            jnp.array_equal(
                transition.decision_id,
                state.pending_prototype_decision_id,
            )
            & (transition.action == objective_receipt.action)
            & _float32_bits_equal(
                transition.observation,
                prototype_state.current_raw_observation,
            )
            & _float32_bits_equal(
                objective_receipt.representation,
                prototype_state.current_representation,
            )
            & jnp.all(
                objective_receipt.representation_revision_words
                == prototype_state.observation_event_words
            )
        )

        proposed_target_identity, target_identity_capacity = _increment_words(
            state.target_receipt_words
        )
        target_owner_matches = (
            jnp.array_equal(
                target_receipt.prototype_decision_id,
                state.pending_prototype_decision_id,
            )
            & (target_receipt.action == prototype_state.current_action)
            & jnp.all(
                target_receipt.observation_event_words == prototype_state.observation_event_words
            )
            & jnp.all(target_receipt.builder_step_words == state.pending_builder_step_words)
            & jnp.all(target_receipt.builder_update_words == state.pending_builder_update_words)
            & jnp.all(target_receipt.target_identity_words == proposed_target_identity)
        )
        expected_payload_words = _target_payload_words(
            target_receipt.cumulant,
            target_receipt.gvf_continuation,
            target_receipt.control_value_target,
            target_receipt.selected_action_advantage_target,
        )
        target_payload_matches = jnp.all(target_receipt.payload_words == expected_payload_words)
        expected_content_tag = _target_content_tag(
            prototype_decision_id=target_receipt.prototype_decision_id,
            action=target_receipt.action,
            observation_event_words=target_receipt.observation_event_words,
            builder_step_words=target_receipt.builder_step_words,
            builder_update_words=target_receipt.builder_update_words,
            target_identity_words=target_receipt.target_identity_words,
            source_revision_words=target_receipt.source_revision_words,
            provenance_words=target_receipt.provenance_words,
            payload_words=target_receipt.payload_words,
        )
        target_content_tag_matches = jnp.all(
            target_receipt.content_tag_words == expected_content_tag
        )
        target_provenance_valid = jnp.any(target_receipt.provenance_words != 0)
        target_source_revision_valid = _words_not_earlier(
            target_receipt.source_revision_words,
            state.last_target_source_revision_words,
        )

        boundary = transition.terminated | transition.truncated
        rtu_preparation: PrototypeRTUTransitionPreparation | None = None
        prototype_result: PrototypeUpdateResult | None = None
        bootstrap_transition: Any
        expected_destination: Any
        prototype_destination: Any
        if self._rtu_generate_and_test is not None:
            rtu_preparation = self._prototype.prepare_rtu_transition(
                prototype_state,
                transition,
            )
            bootstrap_transition = rtu_preparation.bootstrap_transition
            expected_destination = rtu_preparation.decision_builder_state
            expected_destination_valid = rtu_preparation.preparation_valid
            prototype_destination = expected_destination
            destination_matches = jnp.asarray(True, dtype=jnp.bool_)
            prototype_applied = rtu_preparation.preparation_valid
        else:
            prototype_result = self._prototype.update_transition(
                prototype_state,
                transition,
            )
            prototype_applied = prototype_result.transition_diagnostics.valid
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
                OnlineGatedStateBuilderState,
                jax.lax.cond(
                    boundary,
                    lambda: restart_transition.state,
                    lambda: bootstrap_transition.state,
                ),
            )
            expected_destination_valid = (
                bootstrap_transition.transition_applied
                & jnp.where(
                    boundary,
                    restart_transition.transition_applied,
                    jnp.asarray(True, dtype=jnp.bool_),
                )
            )
            prototype_destination = cast(
                OnlineGatedStateBuilderState,
                prototype_result.state.state_builder_state,
            )
            destination_matches = _builder_states_equal(
                expected_destination,
                prototype_destination,
            )

        bootstrap_event_words, bootstrap_event_capacity = _increment_words(
            prototype_state.observation_event_words
        )
        objective_update = self._objectives.update(
            state.objectives_state,
            objective_receipt,
            bootstrap_transition.representation,
            bootstrap_event_words,
            transition.next_observation,
            transition.reward,
            transition.terminated,
            target_receipt.cumulant,
            target_receipt.gvf_continuation,
            target_receipt.control_value_target,
            target_receipt.selected_action_advantage_target,
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
                        cast(Any, state.objectives_state),
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
                counterfactual = self._objectives._update_jit(
                    deleted_state,
                    deleted_receipt,
                    bootstrap_transition.representation,
                    bootstrap_event_words,
                    transition.next_observation,
                    transition.reward,
                    transition.terminated,
                    target_receipt.cumulant,
                    target_receipt.gvf_continuation,
                    target_receipt.control_value_target,
                    target_receipt.selected_action_advantage_target,
                )
                return counterfactual.balanced_loss, counterfactual.update_applied

            deleted_losses, deletion_transactions_applied = jax.vmap(
                frozen_head_deletion_loss
            )(unit_ids)
            raw_deletion_change = (
                deleted_losses - objective_update.balanced_loss
            )
            rtu_causal_deletion_evidence_valid = (
                objective_update.update_applied
                & jnp.all(deletion_transactions_applied)
                & jnp.isfinite(objective_update.balanced_loss)
                & jnp.all(jnp.isfinite(raw_deletion_change))
            )
            rtu_causal_deletion_evidence_available = (
                rtu_causal_deletion_evidence_valid
            )
            rtu_causal_deletion_loss_change = jnp.where(
                rtu_causal_deletion_evidence_available,
                raw_deletion_change,
                jnp.zeros_like(raw_deletion_change),
            )
        current_proposal = self._builder.propose_learning_update(
            source_builder,
            objective_update.current_representation_gradient,
        )
        next_proposal = self._builder.propose_learning_update(
            bootstrap_transition.state,
            objective_update.next_representation_gradient,
        )
        builder_sources_match = (
            _float32_bits_equal(
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
        combined_parameter_update = (
            -jnp.asarray(self._builder.config.step_size, dtype=jnp.float32) * combined_clipped
        )
        # An RTU lifecycle independently reconstructs the exact recurrence
        # destination from its content-bound receipt.  Let that pure proposal
        # form even when a larger compiled graph exposes a bit mismatch in the
        # duplicate Prototype reconstruction; the outer commit still requires
        # ``destination_matches`` and therefore remains fail closed.
        proposal_destination_verified = destination_matches
        if self._rtu_generate_and_test is not None:
            proposal_destination_verified = jnp.asarray(True, dtype=jnp.bool_)
        base_proposal_approved = (
            source_valid
            & transition_identity_matches
            & target_owner_matches
            & target_payload_matches
            & target_content_tag_matches
            & target_provenance_valid
            & target_source_revision_valid
            & target_identity_capacity
            & expected_destination_valid
            & prototype_applied
            & objective_update.update_applied
            & current_proposal.valid
            & next_proposal.valid
            & builder_sources_match
            & proposal_destination_verified
            & bootstrap_event_capacity
            & combined_valid
            & rtu_causal_deletion_evidence_valid
            & rtu_global_lifetime_capacity
        )
        combined_proposal = replace_state_builder_learning_proposal_update(
            current_proposal,
            combined_parameter_update,
            base_proposal_approved,
        )
        learned_builder: (
            OnlineGatedStateBuilderState | RecurrentTraceUnitStateBuilderState
        )
        builder_diagnostics: StateBuilderLearningDiagnostics
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
        post_replacement_objectives = objective_update.state
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
                objective_update.current_representation_gradient,
                combined_proposal,
                rtu_receipt,
                replacement_allowed=(
                    prototype_state.oak_state.stomp_state.executing_option < 0
                ),
                causal_deletion_loss_change=(
                    rtu_causal_deletion_loss_change
                ),
                causal_deletion_evidence_available=(
                    rtu_causal_deletion_evidence_available
                ),
                require_causal_evidence=True,
            )
            rtu_lifecycle_source_matches = _exact_tree_equal(
                rtu_proposal.source_state,
                rtu_source,
            )
            ordinary_learning_diagnostics = (
                rtu_proposal.ordinary_learning_diagnostics
            )
            if ordinary_learning_diagnostics is None:
                raise RuntimeError(
                    "RTU adapter requires ordinary builder-learning diagnostics"
                )
            learned_builder = rtu_proposal.live_builder_state
            builder_diagnostics = ordinary_learning_diagnostics
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
                        objective_update.state,
                        objective_reset_mask,
                    ),
                    lambda: objective_update.state,
                ),
            )
            if rtu_preparation is None:
                raise RuntimeError(
                    "RTU lifecycle requires a Prototype preparation"
                )
            finalization_receipt = self._prototype.bind_rtu_finalization(
                rtu_preparation,
                rtu_result.builder_state,
                rtu_proposal.selected_mask,
                rtu_proposal,
            )
            prototype_result = self._prototype.finalize_rtu_transition(
                prototype_state,
                transition,
                finalization_receipt,
                self._rtu_generate_and_test,
            )
            prototype_applied = prototype_result.transition_diagnostics.valid
            rtu_replacement_cache_safe = prototype_applied
            rtu_replacement_requires_pre_action_hook = jnp.asarray(
                False,
                dtype=jnp.bool_,
            )
            effective_learned_builder = rtu_result.builder_state
            next_rtu_state = rtu_result.state
            learned_prototype = prototype_result.state
        else:
            learned_builder, builder_diagnostics = (
                self._builder.commit_learning_update(
                    prototype_destination,
                    combined_proposal,
                )
            )
            effective_learned_builder = learned_builder
            if prototype_result is None:
                raise RuntimeError("base adapter requires a Prototype result")
            learned_prototype = cast(
                PrototypeAgentState,
                dataclasses.replace(
                    cast(Any, prototype_result.state),
                    state_builder_state=effective_learned_builder,
                ),
            )

        next_cache = self._objectives.cache_action(
            post_replacement_objectives,
            learned_prototype.current_representation,
            learned_prototype.current_action,
            learned_prototype.observation_event_words,
        )
        next_cache_required = learned_prototype.started
        next_cache_valid = jnp.where(
            next_cache_required,
            next_cache.cache_applied,
            ~post_replacement_objectives.pending_valid,
        )
        candidate_objectives = cast(
            ComprehensiveStateObjectivesState,
            jax.lax.cond(
                next_cache_required,
                lambda: next_cache.state,
                lambda: post_replacement_objectives,
            ),
        )
        proposed_transaction_words, transaction_capacity = _increment_words(state.transaction_words)
        transaction_identity_matches = jnp.all(
            proposed_transaction_words == proposed_target_identity
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
        candidate = PrototypeComprehensiveObjectivesState(  # type: ignore[call-arg]
            prototype_state=learned_prototype,
            objectives_state=candidate_objectives,
            rtu_generate_and_test_state=next_rtu_state,
            pending_prototype_decision_id=candidate_pending_decision,
            pending_builder_step_words=candidate_pending_step,
            pending_builder_update_words=candidate_pending_update,
            pending_valid=next_cache_required,
            transaction_words=proposed_transaction_words,
            target_receipt_words=proposed_target_identity,
            last_target_prototype_decision_id=target_receipt.prototype_decision_id,
            last_target_action=target_receipt.action,
            last_target_observation_event_words=target_receipt.observation_event_words,
            last_target_builder_step_words=target_receipt.builder_step_words,
            last_target_builder_update_words=target_receipt.builder_update_words,
            last_target_source_revision_words=target_receipt.source_revision_words,
            last_target_provenance_words=target_receipt.provenance_words,
            last_target_payload_words=target_receipt.payload_words,
            last_target_content_tag_words=target_receipt.content_tag_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            base_proposal_approved
            & destination_matches
            & builder_diagnostics.applied
            & rtu_proposal_valid
            & rtu_lifecycle_source_matches
            & rtu_transaction_applied
            & rtu_replacement_cache_safe
            & next_cache_valid
            & transaction_capacity
            & rtu_global_lifetime_capacity
            & transaction_identity_matches
            & candidate_valid
        )
        final_state = cast(
            PrototypeComprehensiveObjectivesState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        committed_prototype_result = prototype_result
        return PrototypeComprehensiveObjectivesUpdateResult(  # type: ignore[call-arg]
            state=final_state,
            action=final_state.prototype_state.current_action,
            prototype_transition=(
                committed_prototype_result.transition_diagnostics
            ),
            objective_update=objective_update,
            next_objective_cache=next_cache,
            bootstrap_builder_transition=bootstrap_transition,
            builder_learning=builder_diagnostics,
            rtu_generate_and_test=rtu_result,
            rtu_advance_receipt=rtu_receipt,
            target_receipt=target_receipt,
            bootstrap_representation=bootstrap_transition.representation,
            combined_raw_parameter_gradient_norm=combined_norm,
            pre_transaction_words=state.transaction_words,
            post_transaction_words=final_state.transaction_words,
            source_state_valid=source_valid,
            transition_identity_matches=transition_identity_matches,
            target_owner_matches=target_owner_matches,
            target_payload_matches=target_payload_matches,
            target_content_tag_matches=target_content_tag_matches,
            target_provenance_valid=target_provenance_valid,
            target_source_revision_valid=target_source_revision_valid,
            target_identity_capacity_available=target_identity_capacity,
            bootstrap_event_capacity_available=bootstrap_event_capacity,
            bootstrap_transition_applied=expected_destination_valid,
            prototype_transaction_applied=prototype_applied,
            objective_transaction_applied=objective_update.update_applied,
            builder_sources_match=builder_sources_match,
            builder_destination_matches=destination_matches,
            builder_transaction_applied=builder_diagnostics.applied,
            rtu_observation_proposal_valid=rtu_proposal_valid,
            rtu_lifecycle_source_matches=rtu_lifecycle_source_matches,
            rtu_observation_transaction_applied=(
                applied & rtu_transaction_applied
            ),
            rtu_causal_deletion_evidence_attempted=(
                rtu_causal_deletion_evidence_attempted
            ),
            rtu_causal_deletion_evidence_valid=(
                rtu_causal_deletion_evidence_valid
            ),
            rtu_replacement_cache_safe=rtu_replacement_cache_safe,
            rtu_replacement_requires_pre_action_hook=(
                rtu_replacement_requires_pre_action_hook
            ),
            next_cache_required=next_cache_required,
            next_cache_valid=next_cache_valid,
            lifetime_capacity_available=(
                transaction_capacity
                & target_identity_capacity
                & rtu_global_lifetime_capacity
            ),
            candidate_state_valid=candidate_valid,
            target_receipt_committed=applied,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: PrototypeComprehensiveObjectivesState | None = None,
    ) -> PrototypeComprehensiveObjectivesResourceBudget:
        """Return exact persistent bytes and declared fixed work bounds."""

        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        prototype_nbytes = measure_prototype_agent_state_resources(
            reference.prototype_state
        ).total_nbytes
        objectives_nbytes = measure_comprehensive_state_objectives_state_nbytes(
            reference.objectives_state
        )
        rtu_nbytes = (
            _state_array_nbytes(reference.rtu_generate_and_test_state)
            if reference.rtu_generate_and_test_state is not None
            else 0
        )
        metadata_nbytes = (
            16  # pending Prototype decision
            + 8  # pending builder step owner
            + 8  # pending builder update owner
            + 1  # pending-valid flag
            + 8  # transaction clock
            + 8  # target receipt clock
            + 16  # last target Prototype decision
            + 4  # last target action
            + 8  # last target observation event
            + 8  # last target builder step owner
            + 8  # last target builder update owner
            + 8  # last target source revision
            + 16  # last target provenance
            + 16  # last target payload bits
            + 16  # last target content tag
        )
        budget = PrototypeComprehensiveObjectivesResourceBudget(
            schema=PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RESOURCE_SCHEMA,
            prototype_state_nbytes=prototype_nbytes,
            objectives_state_nbytes=objectives_nbytes,
            rtu_generate_and_test_state_nbytes=rtu_nbytes,
            adapter_metadata_nbytes=metadata_nbytes,
            total_state_nbytes=(
                prototype_nbytes + objectives_nbytes + rtu_nbytes + metadata_nbytes
            ),
            max_prototype_updates_per_transition=1,
            max_objective_parameter_head_updates_per_transition=len(
                COMPREHENSIVE_STATE_OBJECTIVES_HEADS
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
                # RTU path: public proposal preflight, proposal materialization,
                # outer commit recomputation, and independent Prototype
                # authorization recomputation.  Exactly one resulting ordinary
                # update revision can enter persistent state.
                4 if self._rtu_generate_and_test is not None else 1
            ),
            max_rtu_generate_and_test_proposals_per_transition=(
                1 if self._rtu_generate_and_test is not None else 0
            ),
            max_rtu_generate_and_test_commits_per_transition=(
                # One outer lifecycle commit plus Prototype's independent
                # destination-derivation authorization.
                2 if self._rtu_generate_and_test is not None else 0
            ),
            max_next_action_cache_writes_per_transition=1,
            max_target_receipts_consumed_per_transition=1,
            max_accepted_transitions=self.max_accepted_transitions,
            persistent_bytes_scope=(
                "all-JAX-array-leaves-in-composed-state; excludes-Python-composition-objects"
            ),
            diagnostic_bytes_scope=(
                "result-component-and-transient-target-receipt-diagnostics-excluded"
            ),
            temporary_bytes_scope=(
                "source-level-named-arrays; excludes-compiler-and-XLA-workspaces; "
                "not-a-measured-device-peak"
            ),
        )
        if measure_prototype_comprehensive_objectives_state_nbytes(reference) != (
            budget.total_state_nbytes
        ):
            raise ValueError("composed state allocation differs from its resource declaration")
        return budget


def run_prototype_comprehensive_objectives_scan(
    adapter: PrototypeComprehensiveStateObjectives,
    state: PrototypeComprehensiveObjectivesState,
    next_observations: Array,
    next_decision_observations: Array,
    rewards: Array,
    discounts: Array,
    terminated: Array,
    truncated: Array,
    cumulants: Array,
    gvf_continuations: Array,
    control_value_targets: Array,
    selected_action_advantage_targets: Array,
    source_revision_words: Array,
    provenance_words: Array,
) -> PrototypeComprehensiveObjectivesScanResult:
    """Run fixed-shape explicit transitions through the atomic adapter."""

    if type(adapter) is not PrototypeComprehensiveStateObjectives:
        raise TypeError("adapter must be an exact PrototypeComprehensiveStateObjectives")
    adapter._require_state_contract(state)
    raw_dim = adapter.builder.config.observation_dim
    if getattr(next_observations, "ndim", None) != 2:
        raise ValueError("next_observations must have rank two")
    steps = next_observations.shape[0]
    contracts = {
        "next_observations": (
            next_observations,
            (steps, raw_dim),
            jnp.dtype(jnp.float32),
        ),
        "next_decision_observations": (
            next_decision_observations,
            (steps, raw_dim),
            jnp.dtype(jnp.float32),
        ),
        "rewards": (rewards, (steps,), jnp.dtype(jnp.float32)),
        "discounts": (discounts, (steps,), jnp.dtype(jnp.float32)),
        "terminated": (terminated, (steps,), jnp.dtype(jnp.bool_)),
        "truncated": (truncated, (steps,), jnp.dtype(jnp.bool_)),
        "cumulants": (cumulants, (steps,), jnp.dtype(jnp.float32)),
        "gvf_continuations": (
            gvf_continuations,
            (steps,),
            jnp.dtype(jnp.float32),
        ),
        "control_value_targets": (
            control_value_targets,
            (steps,),
            jnp.dtype(jnp.float32),
        ),
        "selected_action_advantage_targets": (
            selected_action_advantage_targets,
            (steps,),
            jnp.dtype(jnp.float32),
        ),
        "source_revision_words": (
            source_revision_words,
            (steps, 2),
            jnp.dtype(jnp.uint32),
        ),
        "provenance_words": (
            provenance_words,
            (steps, _PROVENANCE_WIDTH),
            jnp.dtype(jnp.uint32),
        ),
    }
    for label, (value, shape, dtype) in contracts.items():
        _require_array(value, label=label, shape=shape, dtype=dtype)

    def body(
        carry: PrototypeComprehensiveObjectivesState,
        inputs: tuple[Array, ...],
    ) -> tuple[PrototypeComprehensiveObjectivesState, tuple[Array, ...]]:
        (
            next_observation,
            next_decision_observation,
            reward,
            discount,
            is_terminated,
            is_truncated,
            cumulant,
            gvf_continuation,
            control_value_target,
            selected_action_advantage_target,
            source_revision,
            provenance,
        ) = inputs
        prototype_state = carry.prototype_state
        transition = PrototypeTransition(  # type: ignore[call-arg]
            observation=prototype_state.current_raw_observation,
            action=prototype_state.current_action,
            decision_id=prototype_state.current_decision_id,
            reward=reward,
            discount=discount,
            terminated=is_terminated,
            truncated=is_truncated,
            next_observation=next_observation,
            next_decision_observation=next_decision_observation,
        )
        target_receipt = adapter.make_target_receipt(
            carry,
            cumulant=cumulant,
            gvf_continuation=gvf_continuation,
            control_value_target=control_value_target,
            selected_action_advantage_target=selected_action_advantage_target,
            source_revision_words=source_revision,
            provenance_words=provenance,
        )
        result = adapter.update_transition(carry, transition, target_receipt)
        return result.state, (
            result.objective_update.balanced_loss,
            result.action,
            target_receipt.target_identity_words,
            result.post_transaction_words,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            next_observations,
            next_decision_observations,
            rewards,
            discounts,
            terminated,
            truncated,
            cumulants,
            gvf_continuations,
            control_value_targets,
            selected_action_advantage_targets,
            source_revision_words,
            provenance_words,
        ),
    )
    losses, actions, target_identities, transaction_words, applied = outputs
    return PrototypeComprehensiveObjectivesScanResult(  # type: ignore[call-arg]
        state=final_state,
        balanced_losses=losses,
        actions=actions,
        target_identity_words=target_identities,
        transaction_words=transaction_words,
        update_applied=applied,
    )


def save_prototype_comprehensive_objectives_checkpoint(
    adapter: PrototypeComprehensiveStateObjectives,
    state: PrototypeComprehensiveObjectivesState,
    path: str | Path,
) -> None:
    """Persist the complete composition with strict L0 metadata."""

    if type(adapter) is not PrototypeComprehensiveStateObjectives:
        raise TypeError("adapter must be an exact PrototypeComprehensiveStateObjectives")
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid composed state")
    config = adapter.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA,
            "evidence_level": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP,
            "target_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA,
            "adapter_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": adapter.resource_budget(state).to_config(),
        },
    )


def load_prototype_comprehensive_objectives_checkpoint(
    path: str | Path,
) -> tuple[
    PrototypeComprehensiveStateObjectives,
    PrototypeComprehensiveObjectivesState,
]:
    """Restore only a canonical, resource-consistent composed checkpoint."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "evidence_level",
        "outcome_status",
        "ownership",
        "target_schema",
        "adapter_config",
        "config_sha256",
        "resource_budget",
    }
    fields = _exact_manifest(
        metadata,
        expected,
        label="prototype comprehensive checkpoint",
    )
    fixed = {
        "schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA,
        "evidence_level": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL,
        "outcome_status": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS,
        "ownership": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP,
        "target_schema": PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA,
    }
    for name, expected_value in fixed.items():
        if fields[name] != expected_value:
            raise ValueError(f"prototype comprehensive checkpoint {name} is unsupported")
    config = fields["adapter_config"]
    if type(config) is not dict:
        raise TypeError("prototype comprehensive checkpoint config must be an exact dict")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("prototype comprehensive checkpoint config digest differs")
    adapter = PrototypeComprehensiveStateObjectives.from_config(config)
    if adapter.to_config() != config:
        raise ValueError("prototype comprehensive checkpoint config is noncanonical")
    template = adapter.init(jr.key(0))
    expected_budget = adapter.resource_budget(template).to_config()
    if fields["resource_budget"] != expected_budget:
        raise ValueError("prototype comprehensive checkpoint resource budget differs")
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("prototype comprehensive checkpoint metadata changed between reads")
    state = cast(PrototypeComprehensiveObjectivesState, restored)
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("restored prototype comprehensive state is invalid")
    adapter.resource_budget(state)
    return adapter, state


__all__ = [
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CHECKPOINT_SCHEMA",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_CONFIG_SCHEMA",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIFETIME_SEMANTICS",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_LIMITATIONS",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_MAX_TRANSITIONS",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RTU_MAX_TRANSITIONS",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OWNERSHIP",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_RESOURCE_SCHEMA",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_STATE_SCHEMA",
    "PROTOTYPE_COMPREHENSIVE_OBJECTIVES_TARGET_SCHEMA",
    "PrototypeComprehensiveObjectivesResourceBudget",
    "PrototypeComprehensiveObjectivesScanResult",
    "PrototypeComprehensiveObjectivesStartResult",
    "PrototypeComprehensiveObjectivesState",
    "PrototypeComprehensiveObjectivesUpdateResult",
    "PrototypeComprehensiveStateObjectives",
    "PrototypeComprehensiveTargetReceipt",
    "load_prototype_comprehensive_objectives_checkpoint",
    "measure_prototype_comprehensive_objectives_state_nbytes",
    "run_prototype_comprehensive_objectives_scan",
    "save_prototype_comprehensive_objectives_checkpoint",
]
