"""Development-only descriptor-change sanitizer for candidate local state.

``CompositionalFeatureLearner.update`` has two ways to recycle a candidate
slot: an ordinary refresh and the refresh that follows promotion.  Both paths
install a new local descriptor and generator provenance.  This module adds a
generic sanitizer boundary after either path.  It detects direct candidate
descriptor changes from all descriptor words (operation, both parents, both
exact float32 theta words, and depth).  It also propagates exact active-slot
descriptor changes through active descendants and treats a candidate as changed
when either referenced active parent changed meaning.  It preserves the
post-update descriptor and provenance, and sets every other changed-candidate
local carry to exact positive zero.

The sanitizer deliberately trusts neither a candidate age nor a caller-supplied
mask.  The fixed-shape mask is reconstructed from the complete pre/post active
and candidate descriptor banks.  It commits only when at least one direct or
dependency change exists; otherwise the supplied post-update state is returned
bit-exactly.  A matched ``commit=False`` call is also an explicit bit-exact
no-op.

Descriptor change is not authenticated birth identity.  Candidate theta can
itself be trained, active theta normally changes during online learning, and a
random refresh can collide with the old descriptor.  A runner must therefore
authenticate curation/birth events with an external ledger and must establish
that candidate-theta learning is disabled before this sanitizer can be placed
on the generated-class causal path.  This module receives neither proof and
marks the lifecycle prerequisite incomplete.

The JAX kernel validates state semantics and rolls an invalid transaction back
to the supplied post-update state.  The host transaction builder additionally
binds every pre, post, and returned state bit (including host timing metadata),
their resource signatures, the derived mask, and the configuration with
SHA-256.  The strict validator independently rebuilds the transaction.  These
hashes provide integrity binding, not proof that the supplied post state was
actually emitted by a learner update.

This is development infrastructure only.  It does not establish a fresh RNG
epoch, future-target isolation, structural deletion, acquisition, retention,
or any learning outcome, and grants no execution, runner, artifact, evidence,
or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int

from alberta_framework.core.compositional_features import (
    NUM_OPS,
    OP_RAW,
    CompositionalFeatureState,
)
from alberta_framework.core.resource_manager import GeneratorMetaResourceManagerState
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    COMPOSITIONAL_STATE_LEAF_PATHS,
    compositional_state_leaf_paths,
    persistent_compositional_state_nbytes,
)

GENERATED_CANDIDATE_HEAD_ZERO_SCHEMA = (
    "alberta.generated-candidate-head-zero.development.v0"
)
GENERATED_CANDIDATE_HEAD_ZERO_STATUS = "DEVELOPMENT_DESCRIPTOR_SANITIZER_ONLY"


# These sets are an exhaustive partition of the live state leaves.  A future
# CompositionalFeatureState field therefore fails at import rather than becoming
# an untracked memory channel.
CANDIDATE_DESCRIPTOR_LEAF_PATHS = frozenset(
    {
        "candidate_depth",
        "candidate_ops",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_theta",
    }
)
CANDIDATE_PROVENANCE_LEAF_PATHS = frozenset({"candidate_generator_policy"})
CANDIDATE_RESET_LEAF_PATHS = frozenset(
    {
        "candidate_active_correlation_trace",
        "candidate_ages",
        "candidate_output_weights",
        "candidate_retention_slow_utilities",
        "candidate_score_energy_trace",
        "candidate_score_residual_trace",
        "candidate_selector_action_counts",
        "candidate_selector_cumulative_loss",
        "candidate_selector_log_weights",
        "candidate_utilities",
        "candidate_utility_contribution_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_feature_trace",
        "candidate_utility_signal_second_moment",
    }
)
POST_UPDATE_PRESERVED_LEAF_PATHS = COMPOSITIONAL_STATE_LEAF_PATHS.difference(
    CANDIDATE_DESCRIPTOR_LEAF_PATHS
    | CANDIDATE_PROVENANCE_LEAF_PATHS
    | CANDIDATE_RESET_LEAF_PATHS
)


def _assert_exact_leaf_partition() -> None:
    groups = (
        CANDIDATE_DESCRIPTOR_LEAF_PATHS,
        CANDIDATE_PROVENANCE_LEAF_PATHS,
        CANDIDATE_RESET_LEAF_PATHS,
        POST_UPDATE_PRESERVED_LEAF_PATHS,
    )
    combined = frozenset().union(*groups)
    if combined != COMPOSITIONAL_STATE_LEAF_PATHS:
        missing = sorted(COMPOSITIONAL_STATE_LEAF_PATHS - combined)
        extra = sorted(combined - COMPOSITIONAL_STATE_LEAF_PATHS)
        raise RuntimeError(
            "candidate-head-zero leaf partition is stale; "
            f"missing={missing}, extra={extra}"
        )
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            if not first.isdisjoint(second):
                overlap = sorted(first.intersection(second))
                raise RuntimeError(
                    "candidate-head-zero leaf partition overlaps at " f"{overlap}"
                )


_assert_exact_leaf_partition()


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedCandidateHeadZeroConfig:
    """Static shape, prerequisite, and authority contract for the sanitizer."""

    feature_dim: int
    active_slots: int
    candidate_slots: int
    n_tasks: int
    schema: str = GENERATED_CANDIDATE_HEAD_ZERO_SCHEMA
    status: str = GENERATED_CANDIDATE_HEAD_ZERO_STATUS
    development_only: bool = True
    complete_descriptor_exact_bits: bool = True
    candidate_theta_learning_disabled_required: bool = True
    external_birth_event_ledger_required: bool = True
    gradient_descriptor_drift_can_trigger_mask: bool = True
    descriptor_collision_can_hide_birth: bool = True
    host_audit_not_jittable: bool = True
    host_timing_canonicalization_required_for_jit_bit_equality: bool = True
    post_update_origin_authenticated: bool = False
    event_identity_authenticated: bool = False
    lifecycle_prerequisite_complete: bool = False
    fresh_rng_epoch_claimed: bool = False
    future_target_isolation_claimed: bool = False
    structural_deletion_claimed: bool = False
    acquisition_claimed: bool = False
    outcome_claimed: bool = False
    execution_authorized: bool = False
    runner_authorized: bool = False
    artifact_writes_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("feature_dim", "active_slots", "candidate_slots", "n_tasks"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.active_slots < self.feature_dim:
            raise ValueError("active_slots must be at least feature_dim")
        if self.candidate_slots < 1:
            raise ValueError("candidate_slots must be positive")
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if type(self.schema) is not str or self.schema != GENERATED_CANDIDATE_HEAD_ZERO_SCHEMA:
            raise ValueError("candidate-head-zero schema is not canonical")
        if type(self.status) is not str or self.status != GENERATED_CANDIDATE_HEAD_ZERO_STATUS:
            raise ValueError("candidate-head-zero status is not canonical")
        boolean_fields = (
            "development_only",
            "complete_descriptor_exact_bits",
            "candidate_theta_learning_disabled_required",
            "external_birth_event_ledger_required",
            "gradient_descriptor_drift_can_trigger_mask",
            "descriptor_collision_can_hide_birth",
            "host_audit_not_jittable",
            "host_timing_canonicalization_required_for_jit_bit_equality",
            "post_update_origin_authenticated",
            "event_identity_authenticated",
            "lifecycle_prerequisite_complete",
            "fresh_rng_epoch_claimed",
            "future_target_isolation_claimed",
            "structural_deletion_claimed",
            "acquisition_claimed",
            "outcome_claimed",
            "execution_authorized",
            "runner_authorized",
            "artifact_writes_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact Python boolean")
        if (
            not self.development_only
            or not self.complete_descriptor_exact_bits
            or not self.candidate_theta_learning_disabled_required
            or not self.external_birth_event_ledger_required
            or not self.gradient_descriptor_drift_can_trigger_mask
            or not self.descriptor_collision_can_hide_birth
            or not self.host_audit_not_jittable
            or not self.host_timing_canonicalization_required_for_jit_bit_equality
        ):
            raise ValueError(
                "development, exact-bit, disabled-candidate-theta, external-ledger, "
                "and host-audit requirements must remain enabled"
            )
        forbidden = (
            self.post_update_origin_authenticated
            or self.event_identity_authenticated
            or self.lifecycle_prerequisite_complete
            or self.fresh_rng_epoch_claimed
            or self.future_target_isolation_claimed
            or self.structural_deletion_claimed
            or self.acquisition_claimed
            or self.outcome_claimed
            or self.execution_authorized
            or self.runner_authorized
            or self.artifact_writes_authorized
            or self.evidence_authorized
            or self.scientific_promotion_allowed
        )
        if forbidden:
            raise ValueError("candidate-head-zero configuration cannot grant causal claims")


@chex.dataclass(frozen=True)
class GeneratedCandidateHeadZeroDiagnostics:
    """Threshold-free diagnostics for one fixed-shape kernel call."""

    pre_state_valid: Bool[Array, ""]
    post_update_state_valid: Bool[Array, ""]
    resource_shapes_match: Bool[Array, ""]
    descriptor_dependency_masks_exact: Bool[Array, ""]
    nonempty_descriptor_or_dependency_change: Bool[Array, ""]
    proposal_state_valid: Bool[Array, ""]
    new_descriptors_preserved_exact: Bool[Array, ""]
    new_generator_provenance_preserved_exact: Bool[Array, ""]
    changed_candidate_local_state_positive_zero_exact: Bool[Array, ""]
    unchanged_candidates_post_update_bit_exact: Bool[Array, ""]
    active_global_key_generator_timing_post_update_bit_exact: Bool[Array, ""]
    descriptor_sanitizer_valid: Bool[Array, ""]
    commit_requested: Bool[Array, ""]
    descriptor_sanitizer_committed: Bool[Array, ""]
    sham_noop: Bool[Array, ""]
    no_change_noop: Bool[Array, ""]
    rolled_back: Bool[Array, ""]
    active_local_descriptor_change_mask: Bool[Array, " n_features"]
    active_propagated_descriptor_change_mask: Bool[Array, " n_features"]
    candidate_local_descriptor_change_mask: Bool[Array, " n_candidates"]
    candidate_active_parent_dependency_change_mask: Bool[Array, " n_candidates"]
    candidate_reset_mask: Bool[Array, " n_candidates"]
    active_local_descriptor_change_count: Int[Array, ""]
    active_propagated_descriptor_change_count: Int[Array, ""]
    candidate_local_descriptor_change_count: Int[Array, ""]
    candidate_active_parent_dependency_change_count: Int[Array, ""]
    candidate_reset_count: Int[Array, ""]
    post_update_origin_authenticated: Bool[Array, ""]
    event_identity_authenticated: Bool[Array, ""]
    candidate_theta_learning_disabled_required: Bool[Array, ""]
    external_birth_event_ledger_required: Bool[Array, ""]
    gradient_descriptor_drift_can_trigger_mask: Bool[Array, ""]
    descriptor_collision_can_hide_birth: Bool[Array, ""]
    lifecycle_prerequisite_complete: Bool[Array, ""]
    fresh_rng_epoch_claimed: Bool[Array, ""]
    future_target_isolation_claimed: Bool[Array, ""]
    structural_deletion_claimed: Bool[Array, ""]
    acquisition_claimed: Bool[Array, ""]
    outcome_claimed: Bool[Array, ""]
    execution_authorized: Bool[Array, ""]
    runner_authorized: Bool[Array, ""]
    artifact_writes_authorized: Bool[Array, ""]
    evidence_authorized: Bool[Array, ""]
    scientific_promotion_allowed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GeneratedCandidateHeadZeroKernelResult:
    """Atomic state and diagnostics returned by the JAX-compatible kernel."""

    state: CompositionalFeatureState
    diagnostics: GeneratedCandidateHeadZeroDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedCandidateHeadZeroAudit:
    """Host-only exact-bit and resource binding for one kernel result."""

    schema: str
    status: str
    config_sha256: str
    pre_state_bit_sha256: str
    post_update_state_bit_sha256: str
    returned_state_bit_sha256: str
    pre_resource_signature_sha256: str
    post_update_resource_signature_sha256: str
    returned_resource_signature_sha256: str
    active_local_descriptor_change_mask_sha256: str
    active_propagated_descriptor_change_mask_sha256: str
    candidate_local_descriptor_change_mask_sha256: str
    candidate_active_parent_dependency_change_mask_sha256: str
    candidate_reset_mask_sha256: str
    diagnostics_bit_sha256: str
    transaction_sha256: str
    pre_persistent_array_nbytes: int
    post_update_persistent_array_nbytes: int
    returned_persistent_array_nbytes: int
    active_local_descriptor_change_count: int
    active_propagated_descriptor_change_count: int
    candidate_local_descriptor_change_count: int
    candidate_active_parent_dependency_change_count: int
    candidate_reset_count: int
    reset_payload_nbytes_per_changed_candidate: int
    host_timing_metadata_bound_in_state_hashes: bool
    host_audit_not_jittable: bool
    post_update_origin_authenticated: bool
    event_identity_authenticated: bool
    candidate_theta_learning_disabled_required: bool
    external_birth_event_ledger_required: bool
    gradient_descriptor_drift_can_trigger_mask: bool
    descriptor_collision_can_hide_birth: bool
    lifecycle_prerequisite_complete: bool
    fresh_rng_epoch_claimed: bool
    future_target_isolation_claimed: bool
    structural_deletion_claimed: bool
    acquisition_claimed: bool
    outcome_claimed: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedCandidateHeadZeroTransaction:
    """Host-bound descriptor-sanitizer transaction."""

    result: GeneratedCandidateHeadZeroKernelResult
    audit: GeneratedCandidateHeadZeroAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedCandidateHeadZeroValidation:
    """Integrity outcome; standalone causal-refresh validity is always false."""

    valid: bool
    descriptor_sanitizer_commit_valid: bool
    causal_refresh_commit_valid: bool
    sham_noop_valid: bool
    no_change_noop_valid: bool
    errors: tuple[str, ...]
    execution_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False


_ACTIVE_INT_FIELDS = (
    "ops",
    "parent_a",
    "parent_b",
    "depth",
    "ages",
    "feature_generator_policy",
)
_ACTIVE_FLOAT_FIELDS = (
    "utilities",
    "utility_feature_trace",
    "utility_feature_energy_trace",
    "utility_signal_second_moment",
    "feature_score_energy_trace",
    "retention_slow_utilities",
)
_ACTIVE_TASK_FIELDS = (
    "output_weights",
    "utility_contribution_trace",
    "feature_score_residual_trace",
)
_CANDIDATE_INT_FIELDS = (
    "candidate_ops",
    "candidate_parent_a",
    "candidate_parent_b",
    "candidate_depth",
    "candidate_ages",
    "candidate_generator_policy",
)
_CANDIDATE_FLOAT_FIELDS = (
    "candidate_utilities",
    "candidate_utility_feature_trace",
    "candidate_utility_feature_energy_trace",
    "candidate_utility_signal_second_moment",
    "candidate_score_energy_trace",
    "candidate_retention_slow_utilities",
    "candidate_selector_log_weights",
    "candidate_selector_cumulative_loss",
    "candidate_selector_action_counts",
)
_CANDIDATE_TASK_FIELDS = (
    "candidate_output_weights",
    "candidate_utility_contribution_trace",
    "candidate_score_residual_trace",
)


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _require_static_state_contract(
    state: CompositionalFeatureState,
    config: GeneratedCandidateHeadZeroConfig,
) -> None:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    compositional_state_leaf_paths(state)
    n = config.active_slots
    c = config.candidate_slots
    h = config.n_tasks
    for name in _ACTIVE_INT_FIELDS:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(n,), dtype=jnp.int32)
    for name in _ACTIVE_FLOAT_FIELDS:
        _require_array(
            getattr(state, name), name=f"state.{name}", shape=(n,), dtype=jnp.float32
        )
    _require_array(state.theta, name="state.theta", shape=(n, 2), dtype=jnp.float32)
    for name in _ACTIVE_TASK_FIELDS:
        _require_array(
            getattr(state, name), name=f"state.{name}", shape=(h, n), dtype=jnp.float32
        )
    for name in ("output_bias", "utility_error_trace", "task_activity_ema"):
        _require_array(
            getattr(state, name), name=f"state.{name}", shape=(h,), dtype=jnp.float32
        )
    for name in _CANDIDATE_INT_FIELDS:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(c,), dtype=jnp.int32)
    for name in _CANDIDATE_FLOAT_FIELDS:
        _require_array(
            getattr(state, name), name=f"state.{name}", shape=(c,), dtype=jnp.float32
        )
    _require_array(
        state.candidate_theta,
        name="state.candidate_theta",
        shape=(c, 2),
        dtype=jnp.float32,
    )
    for name in _CANDIDATE_TASK_FIELDS:
        _require_array(
            getattr(state, name), name=f"state.{name}", shape=(h, c), dtype=jnp.float32
        )
    _require_array(
        state.candidate_active_correlation_trace,
        name="state.candidate_active_correlation_trace",
        shape=(c, n),
        dtype=jnp.float32,
    )
    _require_array(state.step_count, name="state.step_count", shape=(), dtype=jnp.int32)
    _require_array(
        state.replacement_accumulator,
        name="state.replacement_accumulator",
        shape=(),
        dtype=jnp.float32,
    )
    key_shape = getattr(state.key, "shape", None)
    key_dtype = getattr(state.key, "dtype", None)
    if (
        key_shape != ()
        or key_dtype is None
        or not jax.dtypes.issubdtype(key_dtype, jax.dtypes.prng_key)  # type: ignore[attr-defined]
        or str(jax.random.key_impl(state.key)) != "threefry2x32"
    ):
        raise TypeError("state.key must be one scalar typed Threefry key")
    key_words = jax.random.key_data(state.key)
    if key_words.shape != (2,) or key_words.dtype != jnp.uint32:
        raise TypeError("state.key must expose exactly two uint32 Threefry words")
    generator = state.generator_resource_state
    if type(generator) is not GeneratorMetaResourceManagerState:
        raise TypeError(
            "state.generator_resource_state must be an exact GeneratorMetaResourceManagerState"
        )
    generator_shape = jnp.asarray(generator.log_weights).shape
    if len(generator_shape) != 2 or generator_shape[0] < 1 or generator_shape[1] < 1:
        raise ValueError("generator resource matrices must have two positive dimensions")
    for name in ("log_weights", "reward_ema", "action_counts"):
        _require_array(
            getattr(generator, name),
            name=f"state.generator_resource_state.{name}",
            shape=generator_shape,
            dtype=jnp.float32,
        )
    _require_array(
        generator.step_count,
        name="state.generator_resource_state.step_count",
        shape=(),
        dtype=jnp.int32,
    )


def _finite_state(state: CompositionalFeatureState) -> Array:
    floating = (
        state.theta,
        state.output_weights,
        state.output_bias,
        state.utilities,
        state.utility_contribution_trace,
        state.utility_error_trace,
        state.utility_feature_trace,
        state.utility_feature_energy_trace,
        state.utility_signal_second_moment,
        state.feature_score_residual_trace,
        state.feature_score_energy_trace,
        state.retention_slow_utilities,
        state.task_activity_ema,
        state.candidate_theta,
        state.candidate_output_weights,
        state.candidate_utilities,
        state.candidate_utility_contribution_trace,
        state.candidate_utility_feature_trace,
        state.candidate_utility_feature_energy_trace,
        state.candidate_utility_signal_second_moment,
        state.candidate_score_residual_trace,
        state.candidate_score_energy_trace,
        state.candidate_retention_slow_utilities,
        state.candidate_active_correlation_trace,
        state.candidate_selector_log_weights,
        state.candidate_selector_cumulative_loss,
        state.candidate_selector_action_counts,
        state.generator_resource_state.log_weights,
        state.generator_resource_state.reward_ema,
        state.generator_resource_state.action_counts,
        state.replacement_accumulator,
        jnp.asarray(state.birth_timestamp),
        jnp.asarray(state.uptime_s),
    )
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for value in floating:
        valid = valid & jnp.all(jnp.isfinite(jnp.asarray(value)))
    return valid


def _dag_valid(state: CompositionalFeatureState, feature_dim: int) -> Array:
    n = state.ops.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    is_raw = state.ops == OP_RAW
    safe_a = jnp.clip(state.parent_a, 0, n - 1)
    safe_b = jnp.clip(state.parent_b, 0, n - 1)
    raw_valid = (
        (state.parent_a >= 0)
        & (state.parent_a < feature_dim)
        & (state.parent_b == -1)
        & (state.depth == 0)
    )
    binary_valid = (
        (state.parent_a >= 0)
        & (state.parent_a < indices)
        & (state.parent_b >= 0)
        & (state.parent_b < indices)
        & (state.depth == jnp.maximum(state.depth[safe_a], state.depth[safe_b]) + 1)
    )
    prefix = jnp.arange(feature_dim, dtype=jnp.int32)
    active_valid = (
        (state.ops >= OP_RAW)
        & (state.ops < NUM_OPS)
        & jnp.where(is_raw, raw_valid, binary_valid)
    )

    candidate_is_raw = state.candidate_ops == OP_RAW
    candidate_safe_a = jnp.clip(state.candidate_parent_a, 0, n - 1)
    candidate_safe_b = jnp.clip(state.candidate_parent_b, 0, n - 1)
    candidate_raw_valid = (
        (state.candidate_parent_a >= 0)
        & (state.candidate_parent_a < feature_dim)
        & (state.candidate_parent_b == -1)
        & (state.candidate_depth == 0)
    )
    candidate_binary_valid = (
        (state.candidate_parent_a >= 0)
        & (state.candidate_parent_a < n)
        & (state.candidate_parent_b >= 0)
        & (state.candidate_parent_b < n)
        & (
            state.candidate_depth
            == jnp.maximum(state.depth[candidate_safe_a], state.depth[candidate_safe_b]) + 1
        )
    )
    candidate_valid = (
        (state.candidate_ops >= OP_RAW)
        & (state.candidate_ops < NUM_OPS)
        & jnp.where(candidate_is_raw, candidate_raw_valid, candidate_binary_valid)
    )
    return (
        jnp.all(state.ops[:feature_dim] == OP_RAW)
        & jnp.all(state.parent_a[:feature_dim] == prefix)
        & jnp.all(state.parent_b[:feature_dim] == -1)
        & jnp.all(state.depth[:feature_dim] == 0)
        & jnp.all(state.ops[feature_dim:] != OP_RAW)
        & jnp.all(active_valid)
        & jnp.all(candidate_valid)
    )


def _state_valid(state: CompositionalFeatureState, feature_dim: int) -> Array:
    policy_count = state.generator_resource_state.log_weights.shape[1]
    counters_valid = (
        (state.step_count >= 0)
        & (state.generator_resource_state.step_count >= 0)
        & jnp.all(state.ages >= 0)
        & jnp.all(state.candidate_ages >= 0)
        & jnp.all(state.candidate_selector_action_counts >= 0.0)
        & jnp.all(state.generator_resource_state.action_counts >= 0.0)
        & (state.replacement_accumulator >= 0.0)
    )
    policy_ids_valid = (
        jnp.asarray(policy_count > 0, dtype=jnp.bool_)
        & jnp.all(state.feature_generator_policy >= 0)
        & jnp.all(state.feature_generator_policy < policy_count)
        & jnp.all(state.candidate_generator_policy >= 0)
        & jnp.all(state.candidate_generator_policy < policy_count)
    )
    return _finite_state(state) & _dag_valid(state, feature_dim) & counters_valid & policy_ids_valid


def _descriptor_words(
    ops: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    depth: Array,
) -> Array:
    """Return six exact descriptor words without algebraic canonicalization."""

    return jnp.stack(
        (
            jax.lax.bitcast_convert_type(ops, jnp.uint32),
            jax.lax.bitcast_convert_type(parent_a, jnp.uint32),
            jax.lax.bitcast_convert_type(parent_b, jnp.uint32),
            jax.lax.bitcast_convert_type(theta[:, 0], jnp.uint32),
            jax.lax.bitcast_convert_type(theta[:, 1], jnp.uint32),
            jax.lax.bitcast_convert_type(depth, jnp.uint32),
        ),
        axis=-1,
    )


def _active_descriptor_words(state: CompositionalFeatureState) -> Array:
    return _descriptor_words(
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        state.depth,
    )


def _candidate_descriptor_words(state: CompositionalFeatureState) -> Array:
    return _descriptor_words(
        state.candidate_ops,
        state.candidate_parent_a,
        state.candidate_parent_b,
        state.candidate_theta,
        state.candidate_depth,
    )


def _descriptor_dependency_masks(
    pre_state: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
) -> tuple[Array, Array, Array, Array, Array]:
    """Derive active local/expanded and candidate local/dependent reset masks."""

    active_local = jnp.any(
        _active_descriptor_words(pre_state) != _active_descriptor_words(post_update_state),
        axis=-1,
    )
    active_propagated = active_local
    n = post_update_state.ops.shape[0]
    # Active slots are topologically ordered.  Static Python unrolling keeps
    # this fixed-shape and JIT-compatible while propagating an ancestor's new
    # meaning through every unchanged local descendant descriptor.
    for index in range(n):
        safe_a = jnp.clip(post_update_state.parent_a[index], 0, n - 1)
        safe_b = jnp.clip(post_update_state.parent_b[index], 0, n - 1)
        parent_changed = (post_update_state.ops[index] != OP_RAW) & (
            active_propagated[safe_a] | active_propagated[safe_b]
        )
        active_propagated = active_propagated.at[index].set(
            active_local[index] | parent_changed
        )

    candidate_local = jnp.any(
        _candidate_descriptor_words(pre_state)
        != _candidate_descriptor_words(post_update_state),
        axis=-1,
    )
    safe_candidate_a = jnp.clip(post_update_state.candidate_parent_a, 0, n - 1)
    safe_candidate_b = jnp.clip(post_update_state.candidate_parent_b, 0, n - 1)
    candidate_parent_dependency = (post_update_state.candidate_ops != OP_RAW) & (
        active_propagated[safe_candidate_a] | active_propagated[safe_candidate_b]
    )
    candidate_changed = candidate_local | candidate_parent_dependency
    return (
        active_local,
        active_propagated,
        candidate_local,
        candidate_parent_dependency,
        candidate_changed,
    )


def candidate_descriptor_dependency_change_mask(
    pre_state: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
) -> Bool[Array, " n_candidates"]:
    """Detect direct descriptor or active-parent-dependent candidate changes."""

    return _descriptor_dependency_masks(pre_state, post_update_state)[-1]


def _float_bits_equal(first: Array, second: Array) -> Array:
    return jnp.all(
        jax.lax.bitcast_convert_type(first, jnp.uint32)
        == jax.lax.bitcast_convert_type(second, jnp.uint32)
    )


def _leaf_bit_equal(first: Any, second: Any) -> Array:
    first_dtype = getattr(first, "dtype", None)
    second_dtype = getattr(second, "dtype", None)
    if first_dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        first_dtype, jax.dtypes.prng_key
    ):
        if second_dtype is None or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            second_dtype, jax.dtypes.prng_key
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        return jnp.all(jax.random.key_data(first) == jax.random.key_data(second))
    first_array = jnp.asarray(first)
    second_array = jnp.asarray(second)
    if first_array.shape != second_array.shape or first_array.dtype != second_array.dtype:
        return jnp.asarray(False, dtype=jnp.bool_)
    if jnp.issubdtype(first_array.dtype, jnp.floating):
        if first_array.dtype == jnp.float32:
            return _float_bits_equal(first_array, second_array)
        if first_array.dtype == jnp.float64:
            return jnp.all(
                jax.lax.bitcast_convert_type(first_array, jnp.uint64)
                == jax.lax.bitcast_convert_type(second_array, jnp.uint64)
            )
    return jnp.all(first_array == second_array)


def _resolve_path(state: CompositionalFeatureState, path: str) -> Any:
    value: Any = state
    for name in path.split("."):
        value = getattr(value, name)
    return value


def _all_paths_bit_equal(
    first: CompositionalFeatureState,
    second: CompositionalFeatureState,
    paths: frozenset[str],
) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for path in sorted(paths):
        valid = valid & _leaf_bit_equal(_resolve_path(first, path), _resolve_path(second, path))
    return valid


def _build_proposal(
    post_update_state: CompositionalFeatureState,
    changed: Array,
) -> CompositionalFeatureState:
    columns = changed[None, :]
    rows = changed[:, None]
    return cast(
        CompositionalFeatureState,
        post_update_state.replace(  # type: ignore[attr-defined]
            candidate_output_weights=jnp.where(
                columns,
                jnp.zeros_like(post_update_state.candidate_output_weights),
                post_update_state.candidate_output_weights,
            ),
            candidate_utilities=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_utilities),
                post_update_state.candidate_utilities,
            ),
            candidate_utility_contribution_trace=jnp.where(
                columns,
                jnp.zeros_like(post_update_state.candidate_utility_contribution_trace),
                post_update_state.candidate_utility_contribution_trace,
            ),
            candidate_utility_feature_trace=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_utility_feature_trace),
                post_update_state.candidate_utility_feature_trace,
            ),
            candidate_utility_feature_energy_trace=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_utility_feature_energy_trace),
                post_update_state.candidate_utility_feature_energy_trace,
            ),
            candidate_utility_signal_second_moment=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_utility_signal_second_moment),
                post_update_state.candidate_utility_signal_second_moment,
            ),
            candidate_score_residual_trace=jnp.where(
                columns,
                jnp.zeros_like(post_update_state.candidate_score_residual_trace),
                post_update_state.candidate_score_residual_trace,
            ),
            candidate_score_energy_trace=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_score_energy_trace),
                post_update_state.candidate_score_energy_trace,
            ),
            candidate_retention_slow_utilities=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_retention_slow_utilities),
                post_update_state.candidate_retention_slow_utilities,
            ),
            candidate_active_correlation_trace=jnp.where(
                rows,
                jnp.zeros_like(post_update_state.candidate_active_correlation_trace),
                post_update_state.candidate_active_correlation_trace,
            ),
            candidate_ages=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_ages),
                post_update_state.candidate_ages,
            ),
            candidate_selector_log_weights=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_selector_log_weights),
                post_update_state.candidate_selector_log_weights,
            ),
            candidate_selector_cumulative_loss=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_selector_cumulative_loss),
                post_update_state.candidate_selector_cumulative_loss,
            ),
            candidate_selector_action_counts=jnp.where(
                changed,
                jnp.zeros_like(post_update_state.candidate_selector_action_counts),
                post_update_state.candidate_selector_action_counts,
            ),
        ),
    )


def _positive_zero_masked(value: Array, mask: Array, *, axis: int) -> Array:
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
    expanded = mask if axis == 0 else mask[None, :]
    if axis == 2:
        expanded = mask[:, None]
    return jnp.all(jnp.where(expanded, bits == 0, True))


def _changed_local_positive_zero_exact(
    state: CompositionalFeatureState,
    changed: Array,
) -> Array:
    valid = (
        _positive_zero_masked(state.candidate_output_weights, changed, axis=1)
        & _positive_zero_masked(state.candidate_utilities, changed, axis=0)
        & _positive_zero_masked(
            state.candidate_utility_contribution_trace, changed, axis=1
        )
        & _positive_zero_masked(state.candidate_utility_feature_trace, changed, axis=0)
        & _positive_zero_masked(
            state.candidate_utility_feature_energy_trace, changed, axis=0
        )
        & _positive_zero_masked(
            state.candidate_utility_signal_second_moment, changed, axis=0
        )
        & _positive_zero_masked(state.candidate_score_residual_trace, changed, axis=1)
        & _positive_zero_masked(state.candidate_score_energy_trace, changed, axis=0)
        & _positive_zero_masked(
            state.candidate_retention_slow_utilities, changed, axis=0
        )
        & _positive_zero_masked(
            state.candidate_active_correlation_trace, changed, axis=2
        )
        & jnp.all(jnp.where(changed, state.candidate_ages == 0, True))
        & _positive_zero_masked(state.candidate_selector_log_weights, changed, axis=0)
        & _positive_zero_masked(
            state.candidate_selector_cumulative_loss, changed, axis=0
        )
        & _positive_zero_masked(
            state.candidate_selector_action_counts, changed, axis=0
        )
    )
    return valid


def _unchanged_candidates_post_bit_exact(
    proposal: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
    changed: Array,
) -> Array:
    unchanged = ~changed

    def masked_equal(first: Array, second: Array, *, axis: int) -> Array:
        if jnp.issubdtype(first.dtype, jnp.floating):
            first_words = jax.lax.bitcast_convert_type(first, jnp.uint32)
            second_words = jax.lax.bitcast_convert_type(second, jnp.uint32)
            equal = first_words == second_words
        else:
            equal = first == second
        expanded = unchanged if axis == 0 else unchanged[None, :]
        if axis == 2:
            expanded = unchanged[:, None]
        return jnp.all(jnp.where(expanded, equal, True))

    axis0 = (
        "candidate_ops",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_depth",
        "candidate_utilities",
        "candidate_utility_feature_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_signal_second_moment",
        "candidate_score_energy_trace",
        "candidate_retention_slow_utilities",
        "candidate_ages",
        "candidate_selector_log_weights",
        "candidate_selector_cumulative_loss",
        "candidate_selector_action_counts",
        "candidate_generator_policy",
    )
    axis1 = (
        "candidate_output_weights",
        "candidate_utility_contribution_trace",
        "candidate_score_residual_trace",
    )
    valid = masked_equal(
        proposal.candidate_theta,
        post_update_state.candidate_theta,
        axis=2,
    )
    for name in axis0:
        valid = valid & masked_equal(
            getattr(proposal, name),
            getattr(post_update_state, name),
            axis=0,
        )
    for name in axis1:
        valid = valid & masked_equal(
            getattr(proposal, name),
            getattr(post_update_state, name),
            axis=1,
        )
    return valid & masked_equal(
        proposal.candidate_active_correlation_trace,
        post_update_state.candidate_active_correlation_trace,
        axis=2,
    )


def _select_returned_state(
    committed: Array,
    proposal: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
) -> CompositionalFeatureState:
    replacements = {
        path: jnp.where(
            committed,
            getattr(proposal, path),
            getattr(post_update_state, path),
        )
        for path in CANDIDATE_RESET_LEAF_PATHS
    }
    return cast(
        CompositionalFeatureState,
        post_update_state.replace(**replacements),  # type: ignore[attr-defined]
    )


def apply_generated_candidate_head_zero(
    pre_state: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
    commit: Array,
    *,
    config: GeneratedCandidateHeadZeroConfig,
) -> GeneratedCandidateHeadZeroKernelResult:
    """Apply or sham one exact candidate-head-zero transaction.

    Static shape/dtype errors raise before tracing.  Dynamic semantic failures
    and a requested commit with no descriptor/dependency change return
    ``post_update_state`` bit-exactly.  Callers needing bit-exact eager/JIT
    comparison must represent the two timing leaves canonically as arrays, as
    disclosed by ``config``.
    """

    if type(config) is not GeneratedCandidateHeadZeroConfig:
        raise TypeError("config must be an exact GeneratedCandidateHeadZeroConfig")
    _require_static_state_contract(pre_state, config)
    _require_static_state_contract(post_update_state, config)
    requested = _require_array(commit, name="commit", shape=(), dtype=jnp.bool_)
    resource_shapes_match = jnp.asarray(
        pre_state.generator_resource_state.log_weights.shape
        == post_update_state.generator_resource_state.log_weights.shape,
        dtype=jnp.bool_,
    )
    pre_valid = _state_valid(pre_state, config.feature_dim)
    post_valid = _state_valid(post_update_state, config.feature_dim)
    (
        active_local,
        active_propagated,
        candidate_local,
        candidate_parent_dependency,
        changed,
    ) = _descriptor_dependency_masks(pre_state, post_update_state)
    # Recomputing every mask is intentional: this asserts there is no
    # caller-supplied or age-derived mask path.
    independently_recomputed_masks = _descriptor_dependency_masks(
        pre_state, post_update_state
    )
    descriptor_dependency_masks_exact = jnp.asarray(True, dtype=jnp.bool_)
    for supplied, recomputed in zip(
        (
            active_local,
            active_propagated,
            candidate_local,
            candidate_parent_dependency,
            changed,
        ),
        independently_recomputed_masks,
        strict=True,
    ):
        descriptor_dependency_masks_exact = (
            descriptor_dependency_masks_exact & jnp.all(supplied == recomputed)
        )
    nonempty = jnp.any(changed)
    proposal = _build_proposal(post_update_state, changed)
    proposal_valid = _state_valid(proposal, config.feature_dim)
    descriptors_preserved = _leaf_bit_equal(
        _candidate_descriptor_words(proposal),
        _candidate_descriptor_words(post_update_state),
    )
    provenance_preserved = _leaf_bit_equal(
        proposal.candidate_generator_policy,
        post_update_state.candidate_generator_policy,
    )
    local_zero = _changed_local_positive_zero_exact(proposal, changed)
    unchanged_exact = _unchanged_candidates_post_bit_exact(
        proposal, post_update_state, changed
    )
    active_global_exact = _all_paths_bit_equal(
        proposal,
        post_update_state,
        POST_UPDATE_PRESERVED_LEAF_PATHS,
    )
    sanitizer_valid = (
        pre_valid
        & post_valid
        & resource_shapes_match
        & descriptor_dependency_masks_exact
        & nonempty
        & proposal_valid
        & descriptors_preserved
        & provenance_preserved
        & local_zero
        & unchanged_exact
        & active_global_exact
    )
    committed = requested & sanitizer_valid
    returned_state = _select_returned_state(committed, proposal, post_update_state)
    base_valid = (
        pre_valid
        & post_valid
        & resource_shapes_match
        & descriptor_dependency_masks_exact
    )
    diagnostics = GeneratedCandidateHeadZeroDiagnostics(  # type: ignore[call-arg]
        pre_state_valid=pre_valid,
        post_update_state_valid=post_valid,
        resource_shapes_match=resource_shapes_match,
        descriptor_dependency_masks_exact=descriptor_dependency_masks_exact,
        nonempty_descriptor_or_dependency_change=nonempty,
        proposal_state_valid=proposal_valid,
        new_descriptors_preserved_exact=descriptors_preserved,
        new_generator_provenance_preserved_exact=provenance_preserved,
        changed_candidate_local_state_positive_zero_exact=local_zero,
        unchanged_candidates_post_update_bit_exact=unchanged_exact,
        active_global_key_generator_timing_post_update_bit_exact=active_global_exact,
        descriptor_sanitizer_valid=sanitizer_valid,
        commit_requested=requested,
        descriptor_sanitizer_committed=committed,
        sham_noop=base_valid & (~requested),
        no_change_noop=base_valid & (~nonempty),
        rolled_back=requested & (~sanitizer_valid),
        active_local_descriptor_change_mask=active_local,
        active_propagated_descriptor_change_mask=active_propagated,
        candidate_local_descriptor_change_mask=candidate_local,
        candidate_active_parent_dependency_change_mask=candidate_parent_dependency,
        candidate_reset_mask=changed,
        active_local_descriptor_change_count=jnp.sum(active_local, dtype=jnp.int32),
        active_propagated_descriptor_change_count=jnp.sum(
            active_propagated, dtype=jnp.int32
        ),
        candidate_local_descriptor_change_count=jnp.sum(
            candidate_local, dtype=jnp.int32
        ),
        candidate_active_parent_dependency_change_count=jnp.sum(
            candidate_parent_dependency, dtype=jnp.int32
        ),
        candidate_reset_count=jnp.sum(changed, dtype=jnp.int32),
        post_update_origin_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        event_identity_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        candidate_theta_learning_disabled_required=jnp.asarray(True, dtype=jnp.bool_),
        external_birth_event_ledger_required=jnp.asarray(True, dtype=jnp.bool_),
        gradient_descriptor_drift_can_trigger_mask=jnp.asarray(True, dtype=jnp.bool_),
        descriptor_collision_can_hide_birth=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_prerequisite_complete=jnp.asarray(False, dtype=jnp.bool_),
        fresh_rng_epoch_claimed=jnp.asarray(False, dtype=jnp.bool_),
        future_target_isolation_claimed=jnp.asarray(False, dtype=jnp.bool_),
        structural_deletion_claimed=jnp.asarray(False, dtype=jnp.bool_),
        acquisition_claimed=jnp.asarray(False, dtype=jnp.bool_),
        outcome_claimed=jnp.asarray(False, dtype=jnp.bool_),
        execution_authorized=jnp.asarray(False, dtype=jnp.bool_),
        runner_authorized=jnp.asarray(False, dtype=jnp.bool_),
        artifact_writes_authorized=jnp.asarray(False, dtype=jnp.bool_),
        evidence_authorized=jnp.asarray(False, dtype=jnp.bool_),
        scientific_promotion_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    return GeneratedCandidateHeadZeroKernelResult(  # type: ignore[call-arg]
        state=returned_state,
        diagnostics=diagnostics,
    )


def _path_text(path: tuple[Any, ...]) -> str:
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if isinstance(name, str):
            names.append(name)
        else:
            names.append(str(key))
    return ".".join(names)


def _tree_leaf_records(value: Any) -> tuple[tuple[str, str, str, tuple[int, ...], bytes], ...]:
    records: list[tuple[str, str, str, tuple[int, ...], bytes]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        path_name = _path_text(path)
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf_dtype, jax.dtypes.prng_key
        ):
            array = np.asarray(jax.random.key_data(leaf))
            records.append((path_name, "typed-prng", str(leaf_dtype), array.shape, array.tobytes()))
        elif isinstance(leaf, Array):
            array = np.asarray(leaf)
            records.append((path_name, "jax-array", array.dtype.str, array.shape, array.tobytes()))
        elif type(leaf) is float:
            records.append((path_name, "python-float", ">f8", (), struct.pack(">d", leaf)))
        elif type(leaf) is bool:
            records.append((path_name, "python-bool", "bool", (), bytes((int(leaf),))))
        elif type(leaf) is int:
            records.append((path_name, "python-int", "int", (), str(leaf).encode("ascii")))
        else:
            raise TypeError(f"unsupported exact-binding leaf at {path_name}: {type(leaf)!r}")
    return tuple(records)


def _records_sha256(
    records: tuple[tuple[str, str, str, tuple[int, ...], bytes], ...],
    *,
    include_values: bool,
) -> str:
    digest = hashlib.sha256()
    for path, kind, dtype, shape, raw in records:
        metadata = json.dumps(
            {
                "path": path,
                "kind": kind,
                "dtype": dtype,
                "shape": list(shape),
                "nbytes": len(raw),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        if include_values:
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _state_bit_sha256(state: CompositionalFeatureState) -> str:
    return _records_sha256(_tree_leaf_records(state), include_values=True)


def _resource_signature_sha256(state: CompositionalFeatureState) -> str:
    records = tuple(
        record
        for record in _tree_leaf_records(state)
        if record[0] not in {"birth_timestamp", "uptime_s"}
    )
    return _records_sha256(records, include_values=False)


def _diagnostics_bit_sha256(diagnostics: GeneratedCandidateHeadZeroDiagnostics) -> str:
    return _records_sha256(_tree_leaf_records(diagnostics), include_values=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _mask_sha256(mask: Array) -> str:
    concrete = np.asarray(mask, dtype=np.bool_)
    return _sha256_json(
        {"dtype": "bool", "shape": list(concrete.shape), "values": concrete.tolist()}
    )


def _config_sha256(config: GeneratedCandidateHeadZeroConfig) -> str:
    return _sha256_json(dataclasses.asdict(config))


def build_generated_candidate_head_zero_transaction(
    pre_state: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
    commit: Array,
    *,
    config: GeneratedCandidateHeadZeroConfig,
) -> GeneratedCandidateHeadZeroTransaction:
    """Build a host-bound transaction from the independently derived kernel result."""

    result = apply_generated_candidate_head_zero(
        pre_state,
        post_update_state,
        commit,
        config=config,
    )
    requested = bool(np.asarray(commit))
    active_local_count = int(
        np.asarray(result.diagnostics.active_local_descriptor_change_count)
    )
    active_propagated_count = int(
        np.asarray(result.diagnostics.active_propagated_descriptor_change_count)
    )
    candidate_local_count = int(
        np.asarray(result.diagnostics.candidate_local_descriptor_change_count)
    )
    candidate_dependency_count = int(
        np.asarray(
            result.diagnostics.candidate_active_parent_dependency_change_count
        )
    )
    changed_count = int(np.asarray(result.diagnostics.candidate_reset_count))
    config_hash = _config_sha256(config)
    pre_hash = _state_bit_sha256(pre_state)
    post_hash = _state_bit_sha256(post_update_state)
    returned_hash = _state_bit_sha256(result.state)
    pre_resource = _resource_signature_sha256(pre_state)
    post_resource = _resource_signature_sha256(post_update_state)
    returned_resource = _resource_signature_sha256(result.state)
    active_local_mask_hash = _mask_sha256(
        result.diagnostics.active_local_descriptor_change_mask
    )
    active_propagated_mask_hash = _mask_sha256(
        result.diagnostics.active_propagated_descriptor_change_mask
    )
    candidate_local_mask_hash = _mask_sha256(
        result.diagnostics.candidate_local_descriptor_change_mask
    )
    candidate_dependency_mask_hash = _mask_sha256(
        result.diagnostics.candidate_active_parent_dependency_change_mask
    )
    candidate_mask_hash = _mask_sha256(
        result.diagnostics.candidate_reset_mask
    )
    diagnostics_hash = _diagnostics_bit_sha256(result.diagnostics)
    transaction_hash = _sha256_json(
        {
            "schema": config.schema,
            "status": config.status,
            "config_sha256": config_hash,
            "pre_state_bit_sha256": pre_hash,
            "post_update_state_bit_sha256": post_hash,
            "returned_state_bit_sha256": returned_hash,
            "pre_resource_signature_sha256": pre_resource,
            "post_update_resource_signature_sha256": post_resource,
            "returned_resource_signature_sha256": returned_resource,
            "active_local_descriptor_change_mask_sha256": active_local_mask_hash,
            "active_propagated_descriptor_change_mask_sha256": (
                active_propagated_mask_hash
            ),
            "candidate_local_descriptor_change_mask_sha256": candidate_local_mask_hash,
            "candidate_active_parent_dependency_change_mask_sha256": (
                candidate_dependency_mask_hash
            ),
            "candidate_reset_mask_sha256": candidate_mask_hash,
            "diagnostics_bit_sha256": diagnostics_hash,
            "commit_requested": requested,
        }
    )
    audit = GeneratedCandidateHeadZeroAudit(
        schema=config.schema,
        status=config.status,
        config_sha256=config_hash,
        pre_state_bit_sha256=pre_hash,
        post_update_state_bit_sha256=post_hash,
        returned_state_bit_sha256=returned_hash,
        pre_resource_signature_sha256=pre_resource,
        post_update_resource_signature_sha256=post_resource,
        returned_resource_signature_sha256=returned_resource,
        active_local_descriptor_change_mask_sha256=active_local_mask_hash,
        active_propagated_descriptor_change_mask_sha256=(
            active_propagated_mask_hash
        ),
        candidate_local_descriptor_change_mask_sha256=candidate_local_mask_hash,
        candidate_active_parent_dependency_change_mask_sha256=(
            candidate_dependency_mask_hash
        ),
        candidate_reset_mask_sha256=candidate_mask_hash,
        diagnostics_bit_sha256=diagnostics_hash,
        transaction_sha256=transaction_hash,
        pre_persistent_array_nbytes=persistent_compositional_state_nbytes(pre_state),
        post_update_persistent_array_nbytes=persistent_compositional_state_nbytes(
            post_update_state
        ),
        returned_persistent_array_nbytes=persistent_compositional_state_nbytes(result.state),
        active_local_descriptor_change_count=active_local_count,
        active_propagated_descriptor_change_count=active_propagated_count,
        candidate_local_descriptor_change_count=candidate_local_count,
        candidate_active_parent_dependency_change_count=candidate_dependency_count,
        candidate_reset_count=changed_count,
        reset_payload_nbytes_per_changed_candidate=(
            4 * (3 * config.n_tasks + config.active_slots + 10)
        ),
        host_timing_metadata_bound_in_state_hashes=True,
        host_audit_not_jittable=True,
        post_update_origin_authenticated=False,
        event_identity_authenticated=False,
        candidate_theta_learning_disabled_required=True,
        external_birth_event_ledger_required=True,
        gradient_descriptor_drift_can_trigger_mask=True,
        descriptor_collision_can_hide_birth=True,
        lifecycle_prerequisite_complete=False,
        fresh_rng_epoch_claimed=False,
        future_target_isolation_claimed=False,
        structural_deletion_claimed=False,
        acquisition_claimed=False,
        outcome_claimed=False,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
    return GeneratedCandidateHeadZeroTransaction(result=result, audit=audit)


def _objects_bit_exact(first: Any, second: Any) -> bool:
    try:
        return _tree_leaf_records(first) == _tree_leaf_records(second)
    except (TypeError, ValueError):
        return False


def validate_generated_candidate_head_zero_transaction(
    pre_state: CompositionalFeatureState,
    post_update_state: CompositionalFeatureState,
    commit: Array,
    transaction: GeneratedCandidateHeadZeroTransaction,
    *,
    config: GeneratedCandidateHeadZeroConfig,
) -> GeneratedCandidateHeadZeroValidation:
    """Strictly rebuild state, masks, diagnostics, and integrity bindings.

    ``valid`` means only that the supplied transaction matches this sanitizer's
    canonical reconstruction.  It does not authenticate a refresh/birth event;
    consequently ``causal_refresh_commit_valid`` is unconditionally false.
    """

    errors: list[str] = []
    if type(config) is not GeneratedCandidateHeadZeroConfig:
        errors.append("config must be an exact GeneratedCandidateHeadZeroConfig")
    if type(transaction) is not GeneratedCandidateHeadZeroTransaction:
        errors.append("transaction must be an exact GeneratedCandidateHeadZeroTransaction")
        return GeneratedCandidateHeadZeroValidation(
            valid=False,
            descriptor_sanitizer_commit_valid=False,
            causal_refresh_commit_valid=False,
            sham_noop_valid=False,
            no_change_noop_valid=False,
            errors=tuple(errors),
        )
    if type(transaction.result) is not GeneratedCandidateHeadZeroKernelResult:
        errors.append("transaction.result must be an exact kernel result")
    if type(transaction.audit) is not GeneratedCandidateHeadZeroAudit:
        errors.append("transaction.audit must be an exact audit")
    try:
        expected = build_generated_candidate_head_zero_transaction(
            pre_state,
            post_update_state,
            commit,
            config=config,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"canonical reconstruction failed: {exc}")
        return GeneratedCandidateHeadZeroValidation(
            valid=False,
            descriptor_sanitizer_commit_valid=False,
            causal_refresh_commit_valid=False,
            sham_noop_valid=False,
            no_change_noop_valid=False,
            errors=tuple(errors),
        )
    if type(transaction.result) is GeneratedCandidateHeadZeroKernelResult:
        if not _objects_bit_exact(transaction.result.state, expected.result.state):
            errors.append("returned state does not match the canonical transaction bit-exactly")
        if not _objects_bit_exact(
            transaction.result.diagnostics, expected.result.diagnostics
        ):
            errors.append("diagnostics do not match the canonical transaction bit-exactly")
    if type(transaction.audit) is GeneratedCandidateHeadZeroAudit:
        if transaction.audit != expected.audit:
            errors.append("audit does not match the canonical pre/post/resource binding")
    valid = not errors
    committed = bool(
        np.asarray(expected.result.diagnostics.descriptor_sanitizer_committed)
    )
    sham = bool(np.asarray(expected.result.diagnostics.sham_noop))
    no_change = bool(np.asarray(expected.result.diagnostics.no_change_noop))
    return GeneratedCandidateHeadZeroValidation(
        valid=valid,
        descriptor_sanitizer_commit_valid=valid and committed,
        causal_refresh_commit_valid=False,
        sham_noop_valid=valid and sham and not committed,
        no_change_noop_valid=valid and no_change and not committed,
        errors=tuple(errors),
    )


__all__ = [
    "CANDIDATE_DESCRIPTOR_LEAF_PATHS",
    "CANDIDATE_PROVENANCE_LEAF_PATHS",
    "CANDIDATE_RESET_LEAF_PATHS",
    "GENERATED_CANDIDATE_HEAD_ZERO_SCHEMA",
    "GENERATED_CANDIDATE_HEAD_ZERO_STATUS",
    "POST_UPDATE_PRESERVED_LEAF_PATHS",
    "GeneratedCandidateHeadZeroAudit",
    "GeneratedCandidateHeadZeroConfig",
    "GeneratedCandidateHeadZeroDiagnostics",
    "GeneratedCandidateHeadZeroKernelResult",
    "GeneratedCandidateHeadZeroTransaction",
    "GeneratedCandidateHeadZeroValidation",
    "apply_generated_candidate_head_zero",
    "build_generated_candidate_head_zero_transaction",
    "candidate_descriptor_dependency_change_mask",
    "validate_generated_candidate_head_zero_transaction",
]
